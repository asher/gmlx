"""KVarNKVCache lifecycle: eager seal, trim plans, replay bit-equality,
serialization round trips. GPU-only where kq kernels dispatch."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import GROUP, KVarNKVCache, KVarNView
from kvarn_testlib import D, H, filled, needs_kvarn_ops, tokens


def _assert_same_content(a, b):
    assert (a.offset, a.n_sealed, a.live_len) == (b.offset, b.n_sealed, b.live_len)
    for f in ("codes_k", "codes_v", "axes_k", "axes_v"):
        x = np.array(getattr(a, f)[:, :, : a.n_sealed])
        y = np.array(getattr(b, f)[:, :, : b.n_sealed])
        assert np.array_equal(x, y), f
    for x, y in zip(a.materialize(), b.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


@needs_kvarn_ops
def test_eager_seal_watermarks():
    c = KVarNKVCache(tail_tokens=256)
    k, v = tokens(300)
    c.update_and_fetch(k[:, :, :127], v[:, :, :127])
    assert (c.offset, c.n_sealed, c.live_len) == (127, 0, 0)
    c.update_and_fetch(k[:, :, 127:129], v[:, :, 127:129])
    assert (c.offset, c.n_sealed, c.live_len) == (129, 0, 1)
    c.update_and_fetch(k[:, :, 129:256], v[:, :, 129:256])
    # token 256 completes the first post-sink group and seals it eagerly
    assert (c.offset, c.n_sealed, c.live_len) == (256, 1, 0)
    assert c.horizon_valid
    c.update_and_fetch(k[:, :, 256:300], v[:, :, 256:300])
    assert (c.offset, c.n_sealed, c.live_len) == (300, 1, 44)
    assert c.tail_len == 256


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_incremental_equals_bulk(d):
    # At d=256 this crosses both slice-transpose paths: single-group seals
    # on the incremental side, the multi-group bulk transpose on the other.
    k, v = tokens(700, d=d)
    inc = KVarNKVCache(tail_tokens=256)
    for i in range(0, 700, 13):
        inc.update_and_fetch(k[:, :, i : i + 13], v[:, :, i : i + 13])
    bulk = KVarNKVCache(tail_tokens=256)
    bulk.update_and_fetch(k, v)
    assert inc.head_dim == bulk.head_dim == d
    _assert_same_content(inc, bulk)


@needs_kvarn_ops
def test_views_and_bypass_errors():
    c = filled(10)
    kv, vv = c.update_and_fetch(*tokens(1, seed=9))
    assert isinstance(kv, KVarNView) and isinstance(vv, KVarNView)
    for op in (
        lambda: kv[0],
        lambda: len(kv),
        lambda: list(kv),
        lambda: np.array(kv),
    ):
        with pytest.raises(RuntimeError, match="kvarn SDPA route"):
            op()


@needs_kvarn_ops
def test_trim_truth_table():
    # 700 tokens, sink 128, sealed 4 (tokens 128..640), live 60,
    # tail covers [316, 700).
    c = filled(700)
    assert c._can_trim(60)  # inside live
    assert c._can_trim(160)  # horizon reopen (g=3)
    assert c._can_trim(250)  # tail rebuild (g=2, group start 384 >= 316)
    # frontier group starts before tail coverage: records dequantize
    assert c._trim_plan(450) == ("records", 0, 122)
    assert c._can_trim(600)  # into sink rows (always valid)
    assert c._can_trim(700)  # full reset
    for n in (60, 160, 250):
        assert c._trim_plan(n)[0] != "records"


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("n_trim", [60, 160, 250, 600])
def test_trim_replay_bit_equality(n_trim, d):
    k, v = tokens(700, d=d)
    ref = KVarNKVCache(tail_tokens=384)
    ref.update_and_fetch(k, v)
    c = KVarNKVCache(tail_tokens=384)
    c.update_and_fetch(k, v)
    assert c.trim(n_trim) == n_trim
    c.update_and_fetch(k[:, :, 700 - n_trim :], v[:, :, 700 - n_trim :])
    _assert_same_content(c, ref)


@needs_kvarn_ops
def test_horizon_single_use():
    c = filled(700)
    assert c.trim(160) == 160  # consumes the horizon (g=3)
    # A second trim crossing the new frontier group has no horizon and no
    # tail coverage for group 2 (tail 384 covers [316, 700), group 2 starts
    # at 384 but the tail shrank with the trim): only the records remain.
    assert c._trim_plan(160)[0] == "records"


@needs_kvarn_ops
def test_sink_extension():
    c = filled(700, sink_tokens=256)
    assert c.sink_cap == 256
    assert c.n_sealed == (700 - 256) // GROUP
    ref_k, _ = c.materialize()
    assert ref_k.shape == (1, H, 700, D)


@needs_kvarn_ops
def test_tail_disabled():
    c = filled(700, tail=0)
    assert c.tail_len == 0
    assert c._can_trim(60)
    assert c._can_trim(160)
    assert c._trim_plan(250)[0] == "records"  # no tail to rebuild from


@needs_kvarn_ops
@pytest.mark.parametrize("d", [128, 256, 512])
def test_state_meta_round_trip(d):
    ref = filled(700, d=d)
    r = KVarNKVCache.from_state(ref.state, ref.meta_state)
    _assert_same_content(r, ref)
    assert r.head_dim == d
    assert (r.k_bits, r.v_bits, r.sink_cap, r.tail_cap) == (6, 6, 128, 384)
    assert r.tail_len == ref.tail_len and r.horizon_valid == ref.horizon_valid


@needs_kvarn_ops
def test_state_is_content_sized_and_regrows():
    # 200 tokens seal nothing: the state carries one placeholder group,
    # the tail through tail_end and a one-row horizon, not the slabs.
    ref = filled(200)
    state_bytes = sum(a.nbytes for a in ref.state)
    assert state_bytes < ref.nbytes / 3
    r = KVarNKVCache.from_state(
        tuple(mx.contiguous(a) for a in ref.state), ref.meta_state
    )
    assert r.nbytes == state_bytes
    _assert_same_content(r, ref)
    # The first write regrows the tail ring and the record slabs.
    for c in (r, ref):
        c.update_and_fetch(*tokens(700, seed=3))
    _assert_same_content(r, ref)
    assert r.trim(130) == ref.trim(130) == 130
    _assert_same_content(r, ref)
    assert r.tail_len == ref.tail_len
    for x, y in zip(r.tail_slices(r.tail_len), ref.tail_slices(ref.tail_len)):
        assert np.array_equal(np.array(x), np.array(y))
    deep = filled(3000)
    assert sum(a.nbytes for a in deep.state) < deep.nbytes


@needs_kvarn_ops
def test_empty_cache_round_trip():
    c = KVarNKVCache(tail_tokens=256)
    r = KVarNKVCache.from_state(c.state, c.meta_state)
    assert r.offset == 0 and not r._allocated()
    r.update_and_fetch(*tokens(5))
    assert r.offset == 5


@needs_kvarn_ops
def test_layout_version_fail_closed():
    ref = filled(300)
    meta = list(ref.meta_state)
    meta[0] = str(int(meta[0]) + 1)
    with pytest.raises(ValueError, match="layout version"):
        KVarNKVCache.from_state(ref.state, tuple(meta))


@needs_kvarn_ops
def test_save_load_prompt_cache_file(tmp_path):
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    from gmlx.cache.kvarn_cache import ensure_registered

    ensure_registered()
    ref = filled(300)
    path = str(tmp_path / "kvarn.safetensors")
    save_prompt_cache(path, [ref])
    (r,) = load_prompt_cache(path)
    assert type(r).__name__ == "KVarNKVCache"
    _assert_same_content(r, ref)


@needs_kvarn_ops
def test_from_cache_conversion():
    from mlx_lm.models.cache import KVCache

    k, v = tokens(300)
    plain = KVCache()
    plain.update_and_fetch(k, v)
    conv = KVarNKVCache.from_cache(plain, tail_tokens=256)
    ref = KVarNKVCache(tail_tokens=256)
    ref.update_and_fetch(k, v)
    _assert_same_content(conv, ref)


@needs_kvarn_ops
def test_conversion_targets_plain_kv_only():
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    from gmlx.cache.kv_policy import quantize_kv_members
    from gmlx.gen.generation import resolve_kvarn_policy

    class _M:
        class args:
            model_type = "llama"
            head_dim = 128

    pc = [KVCache(), RotatingKVCache(max_size=64), ArraysCache(size=1), KVCache()]
    policy = resolve_kvarn_policy(_M(), None, 256, None, pc)
    assert policy.verdict == "partial"
    for i, plan in enumerate(policy.per_layer):
        if plan.quantize:
            pc[i], _ = quantize_kv_members(pc[i], policy)
    # index 3 is the last slot: the carve-out holds it fp16.
    assert type(pc[0]) is KVarNKVCache
    assert type(pc[1]).__name__ == "RotatingKVCache"
    assert type(pc[2]).__name__ == "ArraysCache"
    assert type(pc[3]).__name__ == "KVCache"


@needs_kvarn_ops
def test_nbytes_accounting():
    c = filled(700)
    total = sum(getattr(c, f).nbytes for f in c._STATE_FIELDS)
    assert c.nbytes == total > 0
    # 6-bit records store well under the fp16 equivalent
    rec_bytes = c.codes_k.nbytes + c.axes_k.nbytes
    fp16_equiv = c.codes_k.shape[2] * GROUP * D * 2 * H
    assert rec_bytes < 0.55 * fp16_equiv


@needs_kvarn_ops
def test_nbytes_tracks_active_memory():
    # The formula must reflect what the allocator holds once the fill's
    # transients settle: a state field pinning a retained fp16 graph
    # would push the active delta far past nbytes.
    mx.synchronize()
    before = mx.get_active_memory()
    c = filled(8192, tail=256)
    mx.eval(*(getattr(c, f) for f in c._STATE_FIELDS))
    mx.synchronize()
    delta = mx.get_active_memory() - before
    assert delta <= c.nbytes * 1.1 + (1 << 20)
    # at depth the whole cache (records + fp16 sink/stage/tail + slack)
    # stays well under the fp16 twin, between kv8 (~0.53) and the 6-bit
    # body floor (~0.40)
    fp16_twin = 8192 * H * D * 2 * 2
    assert c.nbytes < 0.55 * fp16_twin


def test_constructor_rejects_malformed():
    with pytest.raises(ValueError):
        KVarNKVCache(k_bits=7)
    with pytest.raises(ValueError):
        KVarNKVCache(tail_tokens=100)
    with pytest.raises(ValueError):
        KVarNKVCache(sink_tokens=0)


@needs_kvarn_ops
def test_update_rejects_malformed():
    c = KVarNKVCache()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="head_dim in"):
        bad = mx.array(rng.standard_normal((1, H, 4, 64)).astype(np.float16))
        c.update_and_fetch(bad, bad)
    with pytest.raises(ValueError, match="single-stream"):
        bad = mx.array(rng.standard_normal((2, H, 4, D)).astype(np.float16))
        c.update_and_fetch(bad, bad)


def test_trim_plan_records_is_the_fallback():
    # Plan arithmetic only: no arrays. sink 128, three sealed groups, five
    # live rows.
    c = KVarNKVCache(tail_tokens=0)
    c.offset, c.n_sealed, c.horizon_valid = 128 + 3 * GROUP + 5, 3, False
    assert c._trim_plan(5) == ("live",)
    assert c._trim_plan(10) == ("records", 2, 123)
    assert c._trim_plan(5 + 2 * GROUP) == ("records", 1, 0)
    assert c._trim_plan(5 + 3 * GROUP) == ("sink",)
    c.horizon_valid = True
    assert c._trim_plan(10) == ("horizon", 123)
    assert c._trim_plan(10 + GROUP) == ("records", 1, 123)


@needs_kvarn_ops
def test_trim_records_dequantizes_the_frontier_group():
    from gmlx.cache.kvarn_cache import _dequant_head

    k, v = tokens(700)
    c = KVarNKVCache(tail_tokens=0)
    c.update_and_fetch(k, v)
    # sealed 4 (tokens 128..640), live 60. Trim 250: frontier 450 lands
    # in group 2 with 66 live rows; no horizon, no tail.
    exp = [
        _dequant_head(cd[:, :, 2:3], ax[:, :, 2:3], 6, side, D, mx.float16)[:, :, :66]
        for side, cd, ax in (("k", c.codes_k, c.axes_k), ("v", c.codes_v, c.axes_v))
    ]
    assert c._trim_plan(250) == ("records", 2, 66)
    assert c.trim(250) == 250
    assert (c.offset, c.n_sealed, c.live_len, c.horizon_valid) == (450, 2, 66, False)
    for stage, e in zip((c.stage_k, c.stage_v), exp, strict=True):
        assert np.array_equal(np.array(stage[:, :, 128:194]), np.array(e))
    # Replay continues from the reopened frontier. Groups 0-1, group 3 and
    # the live rows match a bulk build exactly; group 2 re-seals over 66
    # reopened rows, so its axes and codes carry one extra quantization
    # round trip.
    c.update_and_fetch(k[:, :, 450:], v[:, :, 450:])
    ref = KVarNKVCache(tail_tokens=0)
    ref.update_and_fetch(k, v)
    assert (c.offset, c.n_sealed, c.live_len) == (700, ref.n_sealed, ref.live_len)
    for a, b in zip(c.materialize(), ref.materialize(), strict=True):
        a, b = np.array(a).astype(np.float32), np.array(b).astype(np.float32)
        assert np.array_equal(a[:, :, :384], b[:, :, :384])
        assert np.array_equal(a[:, :, 512:], b[:, :, 512:])
        seg_a, seg_b = a[:, :, 384:512], b[:, :, 384:512]
        rel = np.sqrt(np.mean((seg_a - seg_b) ** 2)) / np.sqrt(np.mean(seg_b**2))
        assert 0 < rel < 0.1
