"""qwen4exp MTP: verify-sink rollback parity, drafter forward, remap."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.config_synth import synthesize_config
from gmlx.qwen4_exp_model import Model, ModelArgs, ensure_registered
from gmlx.qwen4_exp_mtp import (
    MTP_ARCH,
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


def test_drafter_forward_and_seed_cycle():
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
    from gmlx import arch_table
    from gmlx.loader import _MTP_TARGET_HOOKS_BY_TYPE, _mtp_target_classes
    from gmlx.mtp_load import _assistant_kind

    assert arch_table.drafter_arches("qwen4_exp") == (MTP_ARCH,)
    assert arch_table.drafter_serves(MTP_ARCH, "qwen4exp") is True
    cls, build = _mtp_target_classes("qwen4_exp")
    assert cls is Qwen4ExpSpecLM
    for hook in _MTP_TARGET_HOOKS_BY_TYPE["qwen4_exp"]:
        assert hasattr(cls, hook), hook
    assert _assistant_kind("qwen4_exp", "/nonexistent.gguf") == "qwen4exp"
    lm = build(dict(synthesize_config(_qwen4exp_meta(True, True), _QWEN4EXP_SHAPES)))
    assert isinstance(lm, Model)
