"""qwen4exp MTP: verify-sink rollback parity, drafter forward, remap."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.load.config_synth import synthesize_config
from gmlx.models.qwen4_exp.model import Model, ModelArgs, ensure_registered
from gmlx.models.qwen4_exp.mtp import (
    MTP_ARCH,
    QWEN4EXP_NEXTN_MAP,
    Qwen4ExpMTPConfig,
    Qwen4ExpMTPDrafter,
    Qwen4ExpSpecLM,
    remap_qwen4exp_mtp_arrays,
)
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta


def _args(**over):
    ensure_registered()
    cfg = synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)
    cfg.update(over)
    return ModelArgs.from_dict(cfg)


def _randomize(module, seed=0):
    mx.random.seed(seed)
    flat = {}
    for k, v in nn.utils.tree_flatten(module.parameters()):
        flat[k] = (mx.random.normal(v.shape) * 0.1).astype(v.dtype)
    module.load_weights(list(flat.items()), strict=False)
    mx.eval(module.parameters())


def _cache_state(caches):
    out = []
    for c in caches:
        st = c.state
        out.append([None if a is None else a for a in
                    (st if isinstance(st, (list, tuple)) else [st])])
    return out


def _assert_states_close(a, b, atol):
    assert len(a) == len(b)
    for ca, cb in zip(a, b):
        assert len(ca) == len(cb)
        for x, y in zip(ca, cb):
            if x is None or y is None:
                assert x is None and y is None
                continue
            assert x.shape == y.shape, (x.shape, y.shape)
            if x.dtype in (mx.int32, mx.int64, mx.uint64):
                assert mx.array_equal(x, y)
            else:
                assert mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max() < atol


@pytest.mark.parametrize("prompt_len", [3, 9])
def test_verify_rollback_matches_prefix_forward(prompt_len):
    """A verify block of 4 followed by rollback to 2 accepted (3 kept)
    leaves every cache leaf (GDN conv/scan, PLE history/conv, QSA K/V/ik)
    equal to a clean forward over prompt + the 3 kept tokens, and the next
    decode step's logits agree."""
    args = _args()
    lm = Qwen4ExpSpecLM(args)
    _randomize(lm)
    vocab = args.vocab_size
    mx.random.seed(1)
    prompt = mx.random.randint(0, vocab, (1, prompt_len))
    block = mx.random.randint(0, vocab, (1, 4))
    nxt = mx.random.randint(0, vocab, (1, 1))

    # Speculative path: prefill, verify 4, rollback to accepted=2.
    cache_a = lm.make_cache()
    out = lm(prompt, cache=cache_a, return_hidden=True)
    assert out.hidden_states[-1].shape == (1, prompt_len, args.hc_count, args.hidden_size)
    streams, _, sink = lm.speculative_verify_hidden(block, cache_a)
    assert streams.shape[1] == 4
    assert any(e["kind"] == "gdn" for e in sink)
    assert any(e["kind"] == "ple" for e in sink)
    lm.rollback_speculative_cache(cache_a, sink, 2, 4)
    logits_a = lm(nxt, cache=cache_a).logits
    mx.eval(logits_a)

    # Reference: one forward over prompt + the 3 kept tokens.
    cache_b = lm.make_cache()
    lm(mx.concatenate([prompt, block[:, :3]], axis=1), cache=cache_b)
    logits_b = lm(nxt, cache=cache_b).logits
    mx.eval(logits_b)

    for ca, cb in zip(cache_a, cache_b):
        assert ca.offset == cb.offset if hasattr(ca, "offset") else True
    _assert_states_close(_cache_state(cache_a), _cache_state(cache_b), 2e-4)
    assert mx.abs(logits_a - logits_b).max() < 2e-3


def test_logits_from_hidden_matches_forward():
    args = _args()
    lm = Qwen4ExpSpecLM(args)
    _randomize(lm)
    ids = mx.random.randint(0, args.vocab_size, (1, 5))
    out = lm(ids, return_hidden=True)
    ref = lm(ids).logits
    got = lm.speculative_logits_from_hidden(out.hidden_states[-1])
    assert mx.abs(got - ref).max() < 1e-5
    assert mx.array_equal(lm.speculative_argmax_from_hidden(out.hidden_states[-1]),
                          mx.argmax(ref, axis=-1))


def test_skip_logits_still_returns_the_streams():
    args = _args()
    lm = Qwen4ExpSpecLM(args)
    _randomize(lm)
    ids = mx.random.randint(0, args.vocab_size, (1, 5))
    out = lm(ids, return_hidden=True, skip_logits=True)
    assert out.logits is None
    got = lm.speculative_logits_from_hidden(out.hidden_states[-1])
    assert mx.abs(got - lm(ids).logits).max() < 1e-5


def test_drafter_forward_and_seed_cycle(monkeypatch):
    monkeypatch.setenv("GMLX_Q4_MTP_LADDER", "0")  # the fixed block's cycle
    args = _args()
    target = Qwen4ExpSpecLM(args)
    _randomize(target, seed=2)
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, block_size=3))
    _randomize(drafter, seed=3)
    names = {k for k, _ in nn.utils.tree_flatten(drafter.parameters())}
    for want in ("fc_embedding.weight", "fc_hidden.weight",
                 "pre_fc_norm_hidden.weight", "hyper_connection_mixer.down.weight",
                 "layers.0.self_attn.indexer.q_proj.weight",
                 "layers.0.hc_attn.inject.weight",
                 "layers.0.mlp.switch_mlp.down_proj.weight"):
        assert want in names, want
    assert not any(".ple." in n or "linear_attn" in n for n in names)
    assert drafter.pre_fc_norm_hidden.weight.shape == (args.hc_count * args.hidden_size,)

    class _Wrap(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm

    drafter.reset(_Wrap(target))
    prompt = mx.random.randint(0, args.vocab_size, (1, 6))
    cache = target.make_cache()
    out = target(prompt, cache=cache, return_hidden=True)
    first = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    drafter.prefill_from_target_hidden(
        prompt, out.hidden_states[-1], first, None, greedy=True)
    assert drafter._cache[0].offset == 6
    assert drafter._seed_hidden.shape == (1, 1, args.hc_count, args.hidden_size)
    draft = drafter.draft_block(first, out.hidden_states[-1][:, -1:], None, 3,
                                None, greedy=True)
    assert draft.shape == (1, 2)
    assert drafter._round_appended == 1
    # Accept one draft + a new bonus token: KV trims the rollout and re-adds.
    verify_h = mx.random.normal((1, 3, args.hc_count, args.hidden_size))
    drafter.accept_verified_tokens(verify_h, draft, 1, [int(draft[0, 0].item()), 7],
                                   None, greedy=True)
    assert drafter._cache[0].offset == 6 + 2
    assert drafter._seed_token is not None


def test_draft_ladder_sets_the_round_length(monkeypatch):
    """The seed's confidence picks the draft count from the ladder, the
    accept cap trims it after a short round, and a shorter round is a
    prefix of the fixed block (the same picks, cut short)."""
    args = _args()
    target = Qwen4ExpSpecLM(args)
    _randomize(target, seed=2)
    prompt = mx.random.randint(0, args.vocab_size, (1, 6))
    cache = target.make_cache()
    out = target(prompt, cache=cache, return_hidden=True)
    first = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())

    class _Wrap(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm

    def drafted(ladder, conf=None, last_round=(), streak_cap="0"):
        monkeypatch.setenv("GMLX_Q4_MTP_LADDER", ladder)
        monkeypatch.setenv("GMLX_Q4_MTP_STREAK_CAP", streak_cap)
        drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, block_size=6))
        _randomize(drafter, seed=3)
        drafter.reset(_Wrap(target))
        drafter.prefill_from_target_hidden(
            prompt, out.hidden_states[-1], first, None, greedy=True)
        assert (drafter._seed_conf is not None) == bool(ladder != "0")
        if conf is not None:
            drafter._seed_conf = mx.array([conf], dtype=mx.float32)
        drafter._last_round = last_round
        draft = drafter.draft_block(first, out.hidden_states[-1][:, -1:], None, 6,
                                    None, greedy=True)
        mx.eval(draft)
        return drafter, draft

    _, fixed = drafted("0")
    assert fixed.shape == (1, 5)
    ladder = "0.9:5,0.5:3,0:1"
    # The confidence is the seed's argmax probability; a malformed ladder
    # keeps the default one.
    logits = mx.random.normal((2, 1, args.vocab_size)) * 4
    probs = mx.softmax(logits.astype(mx.float32), axis=-1).max(axis=-1).reshape(-1)
    assert mx.allclose(Qwen4ExpMTPDrafter._conf(logits), probs, atol=1e-6)
    drafter, _ = drafted(ladder)
    assert drafter._ladder == ((0.9, 5), (0.5, 3), (0.0, 1))
    drafter, _ = drafted("0.9:5;0.5:3")
    assert drafter._ladder == ((0.9, 5), (0.5, 3), (0.0, 1))
    drafter, sure = drafted(ladder, conf=0.95)
    assert sure.shape == (1, 5) and drafter._round_appended == 4
    assert mx.array_equal(sure, fixed)
    drafter, mid = drafted(ladder, conf=0.7)
    assert mid.shape == (1, 3) and drafter._round_appended == 2
    assert mx.array_equal(mid, fixed[:, :3])
    drafter, doubt = drafted(ladder, conf=0.2)
    assert doubt.shape == (1, 1) and drafter._round_appended == 0
    assert mx.array_equal(doubt, fixed[:, :1])
    # The streak cap holds the long step back after a rejection.
    _, capped = drafted(ladder, conf=0.95, last_round=(5, 1), streak_cap="3")
    assert capped.shape == (1, 3)
    _, uncapped = drafted(ladder, conf=0.95, last_round=(5, 5), streak_cap="3")
    assert uncapped.shape == (1, 5)
    _, first = drafted(ladder, conf=0.95, last_round=(), streak_cap="3")
    assert first.shape == (1, 5)
    # The accept after a short round trims nothing, records the round and
    # re-seeds with a confidence.
    verify_h = mx.random.normal((1, 2, args.hc_count, args.hidden_size))
    drafter.accept_verified_tokens(verify_h, doubt, 0, [7], None, greedy=True)
    assert drafter._cache[0].offset == 6 + 1
    assert drafter._last_round == (1, 0)
    assert drafter._seed_conf is not None


def test_default_block_follows_the_ladder(monkeypatch):
    from gmlx.models.qwen4_exp import mtp

    monkeypatch.delenv("GMLX_Q4_MTP_LADDER", raising=False)
    assert mtp.default_block_size() == mtp.BLOCK_LADDER
    monkeypatch.setenv("GMLX_Q4_MTP_LADDER", "0")
    assert mtp.default_block_size() == mtp.BLOCK_FIXED
    assert mtp.draft_ladder() == ()
    monkeypatch.setenv("GMLX_Q4_MTP_LADDER", "0:2,0.8:4")
    assert mtp.draft_ladder() == ((0.8, 4), (0.0, 2))
    assert mtp.default_block_size() == 5


class _Wrap(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm


def test_fused_eh_proj_matches_split_projections():
    """The llama.cpp sidecar joins fc_embedding and fc_hidden into one
    eh_proj over [enorm(e), hnorm(h)], embedding half first. With the same
    weights the two forms draft the same streams."""
    args = _args()
    target = Qwen4ExpSpecLM(args)
    _randomize(target, seed=2)
    split = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args))
    _randomize(split, seed=3)
    fused = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, fused_eh_proj=True))
    weights = [(k, v) for k, v in nn.utils.tree_flatten(split.parameters())
               if not k.startswith("fc_")]
    weights.append(("eh_proj.weight", mx.concatenate(
        [split.fc_embedding.weight, split.fc_hidden.weight], axis=1)))
    fused.load_weights(weights, strict=True)
    for d in (split, fused):
        d.reset(_Wrap(target))
    mx.random.seed(4)
    tokens = mx.random.randint(0, args.vocab_size, (1, 3))
    hidden = mx.random.normal((1, 3, args.hc_count, args.hidden_size))
    want = split._forward(tokens, hidden, cache=split.make_cache())
    got = fused._forward(tokens, hidden, cache=fused.make_cache())
    mx.eval(want, got)
    assert mx.abs(got - want).max() < 1e-4


# Every tensor of unsloth's mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf, with
# the block index written as {n}.
_SIDECAR_TENSORS = (
    "attn_k", "attn_k_norm", "attn_output", "attn_q", "attn_q_norm", "attn_v",
    "ffn_down_exps", "ffn_down_shexp", "ffn_gate_exps", "ffn_gate_inp",
    "ffn_gate_inp_shexp", "ffn_gate_shexp", "ffn_up_exps", "ffn_up_shexp",
    "hc_attn_down", "hc_attn_inject", "hc_attn_norm", "hc_attn_up",
    "hc_ffn_down", "hc_ffn_inject", "hc_ffn_norm", "hc_ffn_up",
    "indexer.k_norm", "indexer.k_proj", "indexer.q_norm", "indexer.q_proj",
    "nextn.eh_proj", "nextn.enorm", "nextn.hc_head_down", "nextn.hc_head_norm",
    "nextn.hc_head_up", "nextn.hnorm",
)


def test_sidecar_remap_covers_the_fused_drafter_tree():
    from gmlx.load.wire import remap_mtp_arrays

    args = _args()
    n = args.num_hidden_layers
    arrays = {f"blk.{n}.{t}.weight": mx.zeros((4,)) for t in _SIDECAR_TENSORS}
    # The self-contained sidecar also carries these, and the drafter binds
    # the target's instead.
    arrays["token_embd.weight"] = mx.zeros((4,))
    arrays["output.weight"] = mx.zeros((4,))
    w, _, stats = remap_mtp_arrays(
        arrays, {}, "qwen4exp", first_mtp_block=n,
        n_head=args.num_attention_heads, n_head_kv=args.num_key_value_heads,
        nextn_map=QWEN4EXP_NEXTN_MAP)
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, fused_eh_proj=True))
    names = {k for k, _ in nn.utils.tree_flatten(drafter.parameters())}
    assert set(w) == names
    assert stats["skipped"] == 0


def test_drafter_is_b1_only():
    args = _args()
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args))
    assert len(drafter.make_cache([0])) == 1
    with pytest.raises(NotImplementedError):
        drafter.make_cache([0, 3])


def test_remap_strips_prefix_and_threads_codecs():
    arrays = {
        "mtp.fc_hidden.weight": mx.zeros((4, 4)),
        "mtp.layers.0.mlp.switch_mlp.down_proj.weight": mx.zeros((2, 8), dtype=mx.uint8),
        "mtp.layers.0.mlp.switch_mlp.down_proj.scales": mx.zeros((1,), dtype=mx.uint8),
        "stray.weight": mx.zeros((1,)),
    }
    kq = {"mtp.layers.0.mlp.switch_mlp.down_proj.weight": "q8_0"}
    w, codecs, stats = remap_qwen4exp_mtp_arrays(arrays, kq)
    assert stats == {"mapped": 2, "kquant": 1, "skipped": 1}
    assert set(w) == {"fc_hidden.weight", "layers.0.mlp.switch_mlp.down_proj.weight",
                      "layers.0.mlp.switch_mlp.down_proj.scales"}
    assert codecs == {"layers.0.mlp.switch_mlp.down_proj.weight": "q8_0"}
    assert MTP_ARCH == "qwen4exp-mtp"


def test_arch_table_and_loader_rows():
    import gmlx.load.arch_table as arch_table
    from gmlx.load.mtp_target import _MTP_TARGET_HOOKS_BY_TYPE, _mtp_target_classes
    from gmlx.spec.mtp_load import _assistant_kind

    assert arch_table.drafter_arches("qwen4_exp") == (MTP_ARCH, "qwen4exp")
    assert arch_table.drafter_serves(MTP_ARCH, "qwen4exp") is True
    assert arch_table.drafter_serves("qwen4exp", "qwen4exp") is True
    cls, build = _mtp_target_classes("qwen4_exp")
    assert cls is Qwen4ExpSpecLM
    for hook in _MTP_TARGET_HOOKS_BY_TYPE["qwen4_exp"]:
        assert hasattr(cls, hook), hook
    assert _assistant_kind("qwen4_exp", "/nonexistent.gguf") == "qwen4exp"
    lm = build(dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)))
    assert isinstance(lm, Model)


def test_default_lora_keys_train_through_the_hyper_connection_inject():
    """gmlx train wraps every Linear, the hyper-connection inject included,
    and the inject kernel reads a weight the wrapper does not have."""
    from mlx.utils import tree_flatten

    from gmlx.tune.indices import install_index_stop_gradient
    from gmlx.tune.lora import prepare_lora_student

    m = Model(_args())
    _randomize(m)
    prepare_lora_student(m, rank=4, scale=2.0, keys=None)
    m.train()
    restore = install_index_stop_gradient()
    try:
        ids = mx.array([[i % 30 + 1 for i in range(16)]])
        loss, g = nn.value_and_grad(m, lambda m, t: m(t).astype(mx.float32).sum())(m, ids)
        mx.eval(loss, g)
    finally:
        restore()
    assert any("inject.lora_a" in k for k, _ in tree_flatten(g))
