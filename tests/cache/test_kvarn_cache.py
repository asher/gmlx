"""KVarNKVCache lifecycle: eager seal, trim plans, replay bit-equality,
serialization round trips. GPU-only where kq kernels dispatch."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.kvarn_cache import GROUP, KVarNKVCache, KVarNView, convert_prompt_cache

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn quantize/rotate dispatch Metal kernels; needs the GPU device",
)

H = 2
D = 128


def _tokens(n, seed=0, d=D):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    return k, v


def _filled(n, tail=384, seed=0, d=D, **kw):
    c = KVarNKVCache(tail_tokens=tail, **kw)
    c.update_and_fetch(*_tokens(n, seed, d=d))
    return c


def _assert_same_content(a, b):
    assert (a.offset, a.n_sealed, a.live_len) == (b.offset, b.n_sealed, b.live_len)
    for f in ("codes_k", "codes_v", "axes_k", "axes_v"):
        x = np.array(getattr(a, f)[:, :, : a.n_sealed])
        y = np.array(getattr(b, f)[:, :, : b.n_sealed])
        assert np.array_equal(x, y), f
    for x, y in zip(a.materialize(), b.materialize(), strict=True):
        assert np.array_equal(np.array(x), np.array(y))


@_NEEDS_GPU
def test_eager_seal_watermarks():
    c = KVarNKVCache(tail_tokens=256)
    k, v = _tokens(300)
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


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256, 512])
def test_incremental_equals_bulk(d):
    # At d=256 this crosses both slice-transpose paths: single-group seals
    # on the incremental side, the multi-group bulk transpose on the other.
    k, v = _tokens(700, d=d)
    inc = KVarNKVCache(tail_tokens=256)
    for i in range(0, 700, 13):
        inc.update_and_fetch(k[:, :, i : i + 13], v[:, :, i : i + 13])
    bulk = KVarNKVCache(tail_tokens=256)
    bulk.update_and_fetch(k, v)
    assert inc.head_dim == bulk.head_dim == d
    _assert_same_content(inc, bulk)


@_NEEDS_GPU
def test_views_and_bypass_errors():
    c = _filled(10)
    kv, vv = c.update_and_fetch(*_tokens(1, seed=9))
    assert isinstance(kv, KVarNView) and isinstance(vv, KVarNView)
    for op in (
        lambda: kv[0],
        lambda: len(kv),
        lambda: list(kv),
        lambda: np.array(kv),
    ):
        with pytest.raises(RuntimeError, match="kvarn SDPA route"):
            op()


@_NEEDS_GPU
def test_trim_truth_table():
    # 700 tokens, sink 128, sealed 4 (tokens 128..640), live 60,
    # tail covers [316, 700).
    c = _filled(700)
    assert c._can_trim(60)  # inside live
    assert c._can_trim(160)  # horizon reopen (g=3)
    assert c._can_trim(250)  # tail rebuild (g=2, group start 384 >= 316)
    assert not c._can_trim(450)  # frontier group starts before tail coverage
    assert c._can_trim(600)  # into sink rows (always valid)
    assert c._can_trim(700)  # full reset
    assert c.trim(450) == 0  # refusal is a no-op
    assert c.offset == 700


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256, 512])
@pytest.mark.parametrize("n_trim", [60, 160, 250, 600])
def test_trim_replay_bit_equality(n_trim, d):
    k, v = _tokens(700, d=d)
    ref = KVarNKVCache(tail_tokens=384)
    ref.update_and_fetch(k, v)
    c = KVarNKVCache(tail_tokens=384)
    c.update_and_fetch(k, v)
    assert c.trim(n_trim) == n_trim
    c.update_and_fetch(k[:, :, 700 - n_trim :], v[:, :, 700 - n_trim :])
    _assert_same_content(c, ref)


@_NEEDS_GPU
def test_horizon_single_use():
    c = _filled(700)
    assert c.trim(160) == 160  # consumes the horizon (g=3)
    # A second trim crossing the new frontier group has no horizon and no
    # tail coverage for group 2 (tail 384 covers [316, 700), group 2 starts
    # at 384 but the tail shrank with the trim).
    assert not c._can_trim(160)


@_NEEDS_GPU
def test_sink_extension():
    c = _filled(700, sink_tokens=256)
    assert c.sink_cap == 256
    assert c.n_sealed == (700 - 256) // GROUP
    ref_k, _ = c.materialize()
    assert ref_k.shape == (1, H, 700, D)


@_NEEDS_GPU
def test_tail_disabled():
    c = _filled(700, tail=0)
    assert c.tail_len == 0
    assert c._can_trim(60)
    assert c._can_trim(160)
    assert not c._can_trim(250)  # no tail to rebuild from


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256, 512])
def test_state_meta_round_trip(d):
    ref = _filled(700, d=d)
    r = KVarNKVCache.from_state(ref.state, ref.meta_state)
    _assert_same_content(r, ref)
    assert r.head_dim == d
    assert (r.k_bits, r.v_bits, r.sink_cap, r.tail_cap) == (6, 6, 128, 384)
    assert r.tail_len == ref.tail_len and r.horizon_valid == ref.horizon_valid


@_NEEDS_GPU
def test_empty_cache_round_trip():
    c = KVarNKVCache(tail_tokens=256)
    r = KVarNKVCache.from_state(c.state, c.meta_state)
    assert r.offset == 0 and not r._allocated()
    r.update_and_fetch(*_tokens(5))
    assert r.offset == 5


@_NEEDS_GPU
def test_layout_version_fail_closed():
    ref = _filled(300)
    meta = list(ref.meta_state)
    meta[0] = str(int(meta[0]) + 1)
    with pytest.raises(ValueError, match="layout version"):
        KVarNKVCache.from_state(ref.state, tuple(meta))


@_NEEDS_GPU
def test_save_load_prompt_cache_file(tmp_path):
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    from gmlx.kvarn_cache import ensure_registered

    ensure_registered()
    ref = _filled(300)
    path = str(tmp_path / "kvarn.safetensors")
    save_prompt_cache(path, [ref])
    (r,) = load_prompt_cache(path)
    assert type(r).__name__ == "KVarNKVCache"
    _assert_same_content(r, ref)


@_NEEDS_GPU
def test_from_cache_conversion():
    from mlx_lm.models.cache import KVCache

    k, v = _tokens(300)
    plain = KVCache()
    plain.update_and_fetch(k, v)
    conv = KVarNKVCache.from_cache(plain, tail_tokens=256)
    ref = KVarNKVCache(tail_tokens=256)
    ref.update_and_fetch(k, v)
    _assert_same_content(conv, ref)


@_NEEDS_GPU
def test_convert_prompt_cache_targets_plain_kv_only():
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    pc = [KVCache(), RotatingKVCache(max_size=64), ArraysCache(size=1), KVCache()]
    n = convert_prompt_cache(pc, tail_tokens=256)
    assert n == 2
    assert type(pc[0]) is KVarNKVCache and type(pc[3]) is KVarNKVCache
    assert type(pc[1]).__name__ == "RotatingKVCache"
    assert type(pc[2]).__name__ == "ArraysCache"


@_NEEDS_GPU
def test_nbytes_accounting():
    c = _filled(700)
    total = sum(getattr(c, f).nbytes for f in c._STATE_FIELDS)
    assert c.nbytes == total > 0
    # 6-bit records store well under the fp16 equivalent
    rec_bytes = c.codes_k.nbytes + c.axes_k.nbytes
    fp16_equiv = c.codes_k.shape[2] * GROUP * D * 2 * H
    assert rec_bytes < 0.55 * fp16_equiv


@_NEEDS_GPU
def test_nbytes_tracks_active_memory():
    # The formula must reflect what the allocator holds once the fill's
    # transients settle: a state field pinning a retained fp16 graph
    # would push the active delta far past nbytes.
    mx.synchronize()
    before = mx.get_active_memory()
    c = _filled(8192, tail=256)
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


@_NEEDS_GPU
def test_update_rejects_malformed():
    c = KVarNKVCache()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="head_dim in"):
        bad = mx.array(rng.standard_normal((1, H, 4, 64)).astype(np.float16))
        c.update_and_fetch(bad, bad)
    with pytest.raises(ValueError, match="single-stream"):
        bad = mx.array(rng.standard_normal((2, H, 4, D)).astype(np.float16))
        c.update_and_fetch(bad, bad)
