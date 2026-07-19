"""DeepSeek-V4-Flash MTP: drafter/SpecLM unit tests + tiny greedy-identity A/B.

The losslessness contract is drafter-independent (the verify walk emits the
target's own tokens), so the load-bearing gates are the CACHE paths:

- all-reject rounds: every round rolls back the verify write (2- or 3-wide;
  S=2/S=3), exercising the rotating-cache one-update undo log (the
  sliding-window caches rotate from the first decode step at prompt >
  window) and the PoolingCache remainder/undo trim across its full 4-round
  phase cycle -- including trims whose confirmed-prefix replay re-completes
  a pool window (S=3 only);
- all-accept rounds (oracle drafter): no rollback -- every verify token stays
  in the cache, gating the accept path's cache state. The S=3 oracle also
  gates the draft-time rollout's snapshot/restore (a leaked rollout KV entry
  would desync the head from round 2 on).

All must be token-identical to a plain greedy decode over the SAME prefill
chunking (the V4 graph is numerically path-dependent between chunked and
single-shot prefill, so the A/B shares it).
"""

import mlx.core as mx
import mlx.utils as mu
import pytest

from gmlx import deepseek_v4_model as v4
from gmlx.config_synth import synthesize_config
from gmlx.deepseek_v4_mtp import (
    DeepseekV4MTPConfig,
    DeepseekV4MTPDrafter,
    DeepseekV4SpecLM,
)

from test_config_synth import _DEEPSEEK4_SHAPES, _deepseek4_meta

PREFILL_CHUNK = 8
N_GEN = 24


def _tiny_config() -> dict:
    v4.ensure_registered()
    return synthesize_config(_deepseek4_meta(), tensor_shapes=_DEEPSEEK4_SHAPES)


def _randomize_zero_params(mod) -> None:
    """Give zero-initialized params (router, sinks, hc tables) small random
    values so routing and logits aren't degenerate ties."""
    new = []
    for k, val in mu.tree_flatten(mod.parameters()):
        if (
            val.dtype in (mx.float32, mx.bfloat16, mx.float16)
            and float(mx.abs(val).sum()) == 0.0
        ):
            new.append((k, mx.random.normal(val.shape).astype(val.dtype) * 0.02))
        else:
            new.append((k, val))
    mod.update(mu.tree_unflatten(new))
    mx.eval(mod.parameters())


def _build_target(cfg):
    from gmlx.loader import MTPTextTarget

    lm = DeepseekV4SpecLM(v4.ModelArgs.from_dict(cfg))
    mx.eval(lm.parameters())
    _randomize_zero_params(lm)
    return MTPTextTarget(lm, cfg), lm


def _build_drafter(cfg, cls=DeepseekV4MTPDrafter, block_size=2, **kw):
    args = v4.ModelArgs.from_dict(cfg)
    # __post_init__ truncates; the MTP layer's ratio is appended post-init,
    # exactly as mtp_load._load_deepseek4_mtp_drafter does.
    args.compress_ratios = list(args.compress_ratios) + [0]
    drafter = cls(DeepseekV4MTPConfig(text=args, block_size=block_size), **kw)
    # Pin the rollout confidence gates off so ambient GMLX_MTP_S3_TAU /
    # S4_TAU cannot flip draft widths under deterministic expectations; the
    # gated path has its own dedicated test.
    drafter._rollout_tau = 0.0
    drafter._rollout_tau2 = 0.0
    mx.eval(drafter.parameters())
    _randomize_zero_params(drafter)
    return drafter


def _greedy_reference(lm, prompt, n_gen):
    cache = lm.make_cache()
    i = 0
    while i < prompt.shape[1]:
        logits = lm(prompt[:, i : i + PREFILL_CHUNK], cache=cache).logits
        i += PREFILL_CHUNK
    tok = int(mx.argmax(logits[0, -1]).item())
    out = [tok]
    for _ in range(n_gen - 1):
        logits = lm(mx.array([[tok]], dtype=mx.int32), cache=cache).logits
        tok = int(mx.argmax(logits[0, -1]).item())
        out.append(tok)
    return out


def _spec_decode(target, lm, drafter, prompt, n_gen, block=2):
    from gmlx.speculative import stream_speculative

    drafter.reset(target)
    return list(
        stream_speculative(
            target,
            drafter,
            prompt,
            prompt_cache=lm.make_cache(),
            max_tokens=n_gen,
            sampler=None,
            draft_block_size=block,
            prefill_chunk=PREFILL_CHUNK,
        )
    )


class _OracleDrafter(DeepseekV4MTPDrafter):
    """Drafts a known continuation -> every draft accepted (no rollback)."""

    def __init__(self, config, script=None):
        super().__init__(config)
        self._script = script or []
        self._pos = 0

    def reset(self, target_model, left_padding=None):
        out = super().reset(target_model, left_padding)
        self._pos = 1  # position 0 is the engine-sampled first token
        return out

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler,
                    token_dtype=mx.int32, greedy=False):
        tok = super().draft_block(
            last_bonus, hidden, cache, block_size, sampler, token_dtype, greedy
        )
        n = int(tok.shape[1])
        window = self._script[self._pos : self._pos + n]
        if len(window) == n:
            tok = mx.array([[int(t) for t in window]], dtype=token_dtype)
        return tok

    def accept_verified_tokens(self, verify_hidden, draft_tokens, accepted,
                               new_tokens, sampler, token_dtype=mx.int32,
                               greedy=False):
        self._pos += int(accepted) + 1
        super().accept_verified_tokens(
            verify_hidden, draft_tokens, accepted, new_tokens, sampler,
            token_dtype, greedy,
        )


@pytest.mark.parametrize("block", [2, 3, 4])
def test_mtp_greedy_identity_reject_path(block):
    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    drafter = _build_drafter(cfg, block_size=block)
    # Prompt > sliding_window so the rotating caches rotate from the first
    # decode round; 41 % 4 != 0 so the PoolingCache remainder cycles through
    # the window-completion undo phase (at block 3 the 3-wide verify also
    # hits trims whose confirmed replay re-completes a window).
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)

    ref = _greedy_reference(lm, prompt, N_GEN)
    spec = _spec_decode(target, lm, drafter, prompt, N_GEN, block=block)

    assert spec == ref
    accepts = list(drafter.accept_lens)
    # A random drafter against a random target: rejections must dominate, so
    # the rollback path is genuinely exercised.
    assert any(a == 0 for a in accepts)


@pytest.mark.parametrize("block", [2, 3, 4])
def test_mtp_greedy_identity_accept_path(block):
    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)

    ref = _greedy_reference(lm, prompt, N_GEN)
    oracle = _build_drafter(cfg, cls=_OracleDrafter, block_size=block)
    oracle._script = ref
    spec = _spec_decode(target, lm, oracle, prompt, N_GEN, block=block)

    assert spec == ref
    accepts = list(oracle.accept_lens)
    full = block - 1
    assert sum(1 for a in accepts if a == full) >= len(accepts) - 1, (
        "oracle drafts must (near-)all be accepted (accept-path gate; the "
        "final round may be budget-clamped below the full draft width)"
    )


def test_rotating_undo_replay_is_bit_identical():
    # Cache-level gate for the one-update undo log: a rotated cache takes an
    # armed S=2 update, trims 1, and must equal a control cache that saw the
    # same stream with only the confirmed token.
    from mlx_lm.models.cache import RotatingKVCache

    from gmlx.deepseek_v4_cache import ensure_rollback_attached, set_undo_armed

    ensure_rollback_attached()
    mx.random.seed(3)

    def feed(cache, arrs):
        for a in arrs:
            cache.update_and_fetch(a, mx.zeros(a.shape[:2] + (a.shape[2], 0)))

    stream = [mx.random.normal((1, 1, 1, 4)) for _ in range(20)]
    confirmed = mx.random.normal((1, 1, 1, 4))
    rejected = mx.random.normal((1, 1, 1, 4))

    test_cache = RotatingKVCache(max_size=8)
    feed(test_cache, stream)
    assert test_cache._mtp_undo is None
    set_undo_armed(True)
    try:
        test_cache.update_and_fetch(
            mx.concatenate([confirmed, rejected], axis=2), mx.zeros((1, 1, 2, 0))
        )
    finally:
        set_undo_armed(False)
    assert test_cache.is_trimmable()
    assert test_cache.trim(1) == 1

    control = RotatingKVCache(max_size=8)
    feed(control, stream + [confirmed])

    assert control.offset == test_cache.offset
    assert control._idx == test_cache._idx
    assert mx.array_equal(control.keys, test_cache.keys)


def test_rotating_undo_unarmed_update_not_trimmable():
    from mlx_lm.models.cache import RotatingKVCache

    from gmlx.deepseek_v4_cache import ensure_rollback_attached

    ensure_rollback_attached()
    cache = RotatingKVCache(max_size=4)
    for _ in range(6):
        cache.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0)))
    # Rotated + unarmed S=2 update: stock semantics -- no phantom trimmability.
    cache.update_and_fetch(mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 0)))
    assert not cache.is_trimmable()
    assert cache.trim(1) == 0


def test_pooling_cache_trim_through_completed_window():
    # trim(1) when the rejected token completed a pool window must restore
    # the pre-update state and replay the confirmed token exactly.
    from gmlx.deepseek_v4_cache import PoolingCache

    mx.random.seed(5)
    ratio = 4
    tokens = [mx.random.normal((1, 1, 8)) for _ in range(3)]
    confirmed = mx.random.normal((1, 1, 8))
    rejected = mx.random.normal((1, 1, 8))
    gates = [mx.random.normal((1, 1, 2)) for _ in range(5)]

    test_cache = PoolingCache(ratio)
    for t, g in zip(tokens, gates[:3]):
        test_cache.accumulate_windows(t, g, 0)
    # S=2 verify-sized update completes the window (3 + 2 crosses ratio 4).
    r_kv, _, _ = test_cache.accumulate_windows(
        mx.concatenate([confirmed, rejected], axis=1),
        mx.concatenate([gates[3], gates[4]], axis=1),
        0,
    )
    # window emitted (would be poisoned); ratio-4 prepends the lookback rows
    assert r_kv.shape[1] == 2 * ratio
    assert test_cache.is_trimmable()
    assert test_cache.trim(1) == 1

    control = PoolingCache(ratio)
    for t, g in zip(tokens + [confirmed], gates[:4]):
        control.accumulate_windows(t, g, 0)

    assert control.remainder == test_cache.remainder
    assert mx.array_equal(
        control.buf_kv[:, : control.remainder],
        test_cache.buf_kv[:, : test_cache.remainder],
    )
    assert (control.pooled is None) == (test_cache.pooled is None)


@pytest.mark.parametrize("block", [3, 4])
def test_mtp_greedy_identity_gated_rollout(block):
    # Confidence-gated rounds draft fewer tokens than requested; the
    # engine must rebind bs to the actual width or rollback trims valid
    # tokens (the exact corruption an impossible tau forces every round).
    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    drafter = _build_drafter(cfg, block_size=block)
    drafter._rollout_tau = 2.0  # gate always fires -> 1 draft per round
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)

    ref = _greedy_reference(lm, prompt, N_GEN)
    spec = _spec_decode(target, lm, drafter, prompt, N_GEN, block=block)

    assert spec == ref
    assert all(d == 1 for d in drafter.draft_lens), (
        "every round must have gated down to the single seed draft"
    )


def test_rotating_undo_replay_3wide():
    # S=3 rounds write 3-wide verifies; trim(1) and trim(2) on a rotated
    # cache must each equal a control that saw only the confirmed prefix.
    from mlx_lm.models.cache import RotatingKVCache

    from gmlx.deepseek_v4_cache import ensure_rollback_attached, set_undo_armed

    ensure_rollback_attached()
    mx.random.seed(11)

    def feed(cache, arrs):
        for a in arrs:
            cache.update_and_fetch(a, mx.zeros(a.shape[:2] + (a.shape[2], 0)))

    stream = [mx.random.normal((1, 1, 1, 4)) for _ in range(20)]
    c0, c1, rej = (mx.random.normal((1, 1, 1, 4)) for _ in range(3))

    for n_trim, confirmed in ((1, [c0, c1]), (2, [c0])):
        test_cache = RotatingKVCache(max_size=8)
        feed(test_cache, stream)
        set_undo_armed(True)
        try:
            test_cache.update_and_fetch(
                mx.concatenate([c0, c1, rej], axis=2), mx.zeros((1, 1, 3, 0))
            )
        finally:
            set_undo_armed(False)
        assert test_cache.is_trimmable()
        assert test_cache.trim(n_trim) == n_trim

        control = RotatingKVCache(max_size=8)
        feed(control, stream)
        if len(confirmed) == 2:
            # trim(1)'s replay is a 2-wide concat update; mirror it.
            control.update_and_fetch(
                mx.concatenate(confirmed, axis=2), mx.zeros((1, 1, 2, 0))
            )
        else:
            feed(control, confirmed)

        assert control.offset == test_cache.offset
        assert control._idx == test_cache._idx
        assert mx.array_equal(control.keys, test_cache.keys)


def _pool_px(r_kv, ratio):
    """Deterministic stand-in for the compressor: mean over each window."""
    B, usable, D = r_kv.shape
    return r_kv.reshape(B, usable // ratio, ratio, D).mean(axis=2)


def test_pooling_cache_trim_window_recompletion():
    # S=3 trims whose confirmed-prefix replay RE-completes a pool window:
    # the pooled row must come back from the post-append stash (the
    # compressor inputs are gone), bit-identical to a control that only
    # ever saw the confirmed tokens.
    from gmlx.deepseek_v4_cache import PoolingCache

    mx.random.seed(13)
    ratio = 4
    D, G = 8, 2

    def run(pre_count, n_trim):
        pre = [mx.random.normal((1, 1, D)) for _ in range(pre_count)]
        pre_g = [mx.random.normal((1, 1, G)) for _ in range(pre_count)]
        upd = mx.random.normal((1, 3, D))
        upd_g = mx.random.normal((1, 3, G))

        test_cache = PoolingCache(ratio)
        for t, g in zip(pre, pre_g):
            r_kv, _, _ = test_cache.accumulate_windows(t, g, 0)
            assert r_kv.shape[1] == 0
        r_kv, _, _ = test_cache.accumulate_windows(upd, upd_g, 0)
        if r_kv.shape[1] > 0:
            test_cache.update_and_fetch(_pool_px(r_kv, ratio))
        assert test_cache._can_trim(n_trim)
        assert test_cache.trim(n_trim) == n_trim

        control = PoolingCache(ratio)
        k = 3 - n_trim
        for i in range(pre_count + k):
            t = pre[i] if i < pre_count else upd[:, i - pre_count : i - pre_count + 1]
            g = (
                pre_g[i]
                if i < pre_count
                else upd_g[:, i - pre_count : i - pre_count + 1]
            )
            r_kv, _, _ = control.accumulate_windows(t, g, 0)
            if r_kv.shape[1] > 0:
                control.update_and_fetch(_pool_px(r_kv, ratio))

        assert control.remainder == test_cache.remainder
        if control.remainder > 0:
            assert mx.array_equal(
                control.buf_kv[:, : control.remainder],
                test_cache.buf_kv[:, : test_cache.remainder],
            )
        if control.pooled is None:
            assert test_cache.pooled is None or test_cache.pooled.shape[1] == 0
        else:
            assert mx.array_equal(control.pooled, test_cache.pooled)

    # rem 2 + 3-wide (window completes): trim(1) replay re-completes it
    # (total 4), trim(2) replay stays in the buffer (total 3).
    run(2, 1)
    run(2, 2)
    # rem 3 + 3-wide: both trims re-complete a window (total 5 / 4) --
    # refused outright before the post-append pooled stash existed.
    run(3, 1)
    run(3, 2)


def test_draft_block_s3_rollout_restores_drafter_kv():
    # The block-3 rollout forward must leave no trace in the drafter KV:
    # the accept hook re-writes the position teacher-forced next round.
    mx.random.seed(17)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    drafter = _build_drafter(cfg, block_size=3)
    drafter.reset(target)

    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 12)).astype(mx.int32)
    cache = lm.make_cache()
    out = lm(prompt, cache=cache, return_hidden=True)
    hidden = out.hidden_states[-1]
    bonus = int(mx.argmax(out.logits[0, -1]).item())
    drafter.prefill_from_target_hidden(
        prompt, hidden, bonus, None, greedy=True
    )

    c0 = drafter._cache[0]
    keys0, values0 = c0.keys + 0, c0.values + 0
    offset0, idx0 = c0.offset, c0._idx

    toks = drafter.draft_block(None, None, None, 3, None, greedy=True)
    assert toks.shape == (1, 2)
    assert c0.offset == offset0
    assert c0._idx == idx0
    assert mx.array_equal(c0.keys, keys0)
    assert mx.array_equal(c0.values, values0)

    # Seed consumed: a second draft without an accept must fail loudly.
    with pytest.raises(RuntimeError, match="without a seed"):
        drafter.draft_block(None, None, None, 3, None, greedy=True)


def test_gemv_row_fusion_greedy_identity():
    # Row-concat fusion serves wq_a/wkv and compressor wkv/wgate from single
    # dispatches; each output row is an independent dot product, so a greedy
    # decode must be unchanged after install.
    mx.random.seed(7)
    cfg = _tiny_config()
    _target, lm = _build_target(cfg)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)
    ref = _greedy_reference(lm, prompt, N_GEN)

    n = v4.install_gemv_row_fusion(lm)
    assert n > 0
    fused = _greedy_reference(lm, prompt, N_GEN)
    assert fused == ref


def test_default_drafter_block_size_is_4():
    cfg = _tiny_config()
    args = v4.ModelArgs.from_dict(cfg)
    args.compress_ratios = list(args.compress_ratios) + [0]
    assert DeepseekV4MTPConfig(text=args).block_size == 4


def test_speclm_hooks_match_loader_contract():
    from gmlx.loader import _MTP_TARGET_HOOKS_BY_TYPE

    hooks = _MTP_TARGET_HOOKS_BY_TYPE["deepseek_v4"]
    for hook in hooks:
        assert hasattr(DeepseekV4SpecLM, hook), hook


def test_drafter_validates_and_rejects_batch():
    from gmlx.drafter_protocol import validate_drafter

    cfg = _tiny_config()
    target, _lm = _build_target(cfg)
    drafter = _build_drafter(cfg)
    drafter.reset(target)
    validate_drafter(drafter)
    assert drafter.requires_owned_engine
    assert drafter.config.block_size == 2
    with pytest.raises(NotImplementedError):
        drafter.reset(target, left_padding=[0, 0])


def test_drafter_requires_extended_compress_ratios():
    cfg = _tiny_config()
    args = v4.ModelArgs.from_dict(cfg)  # post-init truncated, NOT extended
    with pytest.raises(ValueError, match="compress_ratios"):
        DeepseekV4MTPDrafter(DeepseekV4MTPConfig(text=args, block_size=2))


def test_mtp_remap_covers_closed_tensor_set():
    from mlx.utils import tree_flatten

    from gmlx.mtp_load import (
        _DEEPSEEK4_MTP_MAP,
        _DEEPSEEK4_MTP_RAW,
        remap_deepseek4_mtp_arrays,
    )

    cfg = _tiny_config()
    drafter = _build_drafter(cfg)
    params = {k for k, _ in tree_flatten(drafter.parameters())}

    o_groups, o_lora = cfg["o_groups"], cfg["o_lora_rank"]
    arrays: dict = {}
    kquant_meta: dict = {}
    for base in _DEEPSEEK4_MTP_MAP:
        name = f"mtp.0.{base}.weight"
        if base == "attn_output_a":
            arrays[name] = mx.zeros((o_groups * o_lora, 8), dtype=mx.uint8)
            arrays[f"mtp.0.{base}.scales"] = mx.zeros(
                (o_groups * o_lora, 2), dtype=mx.uint8
            )
            kquant_meta[name] = "q8_0"
        else:
            arrays[name] = mx.zeros((4, 4))
    for base in _DEEPSEEK4_MTP_RAW:
        suffix = "" if base.endswith(".bias") else ".weight"
        arrays[f"mtp.0.{base}{suffix}"] = mx.zeros((4,))

    weights, meta, stats = remap_deepseek4_mtp_arrays(
        arrays, kquant_meta, o_groups=o_groups, o_lora_rank=o_lora
    )
    assert stats["mapped"] == len(_DEEPSEEK4_MTP_MAP) + len(_DEEPSEEK4_MTP_RAW)
    # Every remap target is a real drafter parameter.
    for key in weights:
        if key.endswith(".scales"):
            continue
        assert key in params, key
    # The wo_a reshape hit wire bytes and planar scales alike.
    assert weights["block.attn.wo_a.weight"].shape == (o_groups, o_lora, 8)
    assert weights["block.attn.wo_a.scales"].shape == (o_groups, o_lora, 2)
    assert meta["block.attn.wo_a.weight"] == "q8_0"

    # Inline-scales codecs (q8_0 in the real MTP GGUF) ship a size-1 .scales
    # placeholder; it must pass through un-reshaped (sanitize's ndim guard).
    w2, _, _ = remap_deepseek4_mtp_arrays(
        {
            "mtp.0.attn_output_a.weight": mx.zeros(
                (o_groups * o_lora, 8), dtype=mx.uint8
            ),
            "mtp.0.attn_output_a.scales": mx.zeros((1,)),
        },
        {"mtp.0.attn_output_a.weight": "q8_0"},
        o_groups=o_groups,
        o_lora_rank=o_lora,
    )
    assert w2["block.attn.wo_a.weight"].shape == (o_groups, o_lora, 8)
    assert w2["block.attn.wo_a.scales"].shape == (1,)

    with pytest.raises(RuntimeError, match="unknown tensor"):
        remap_deepseek4_mtp_arrays(
            {"mtp.0.mystery.weight": mx.zeros((2, 2))}, {},
            o_groups=o_groups, o_lora_rank=o_lora,
        )
    with pytest.raises(RuntimeError, match="non-mtp.0"):
        remap_deepseek4_mtp_arrays(
            {"blk.0.attn_q_a.weight": mx.zeros((2, 2))}, {},
            o_groups=o_groups, o_lora_rank=o_lora,
        )


def test_cacheless_forward_matches_cached():
    # The cache-less (training/eval) forward must apply the same causal
    # pooled-visibility mask the cached path gets from make_mask; without it
    # early positions attend to future pooled rows (maxdiff ~0.85 here).
    mx.random.seed(9)
    cfg = _tiny_config()
    _, lm = _build_target(cfg)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 37)).astype(mx.int32)
    ref = lm(prompt, cache=lm.make_cache()).logits
    got = lm(prompt).logits
    assert got.shape == ref.shape
    assert mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 1e-5
