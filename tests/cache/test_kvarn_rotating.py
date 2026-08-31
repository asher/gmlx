"""KVarNRotatingKVCache: window geometry, entry compaction, context floor,
mask contract, trim/serialization, and post-wrap decode parity. GPU-only
where kq kernels dispatch."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import (
    GROUP,
    KVarNKVCache,
    KVarNRotatingKVCache,
)

_NEEDS_GPU = pytest.mark.skipif(
    mx.default_device() != mx.gpu,
    reason="kvarn kernels are Metal-only; needs the GPU device",
)

H = 2
HQ = 8
D = 128

# Small window for fast wraps: sink 128 + 4 record groups; tail 384 keeps
# the floor at exactly 640.
WIN = 640
TAIL = 384


def _tokens(n, seed=0, d=D):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    return k, v


def _rot(max_size=WIN, tail=TAIL, **kw):
    return KVarNRotatingKVCache(max_size, tail_tokens=tail, **kw)


def _feed(c, k, v, chunk):
    n = k.shape[2]
    for i in range(0, n, chunk):
        c.update_and_fetch(k[:, :, i : i + chunk], v[:, :, i : i + chunk])


def _assert_invariants(c, chunk=GROUP):
    assert c.visible == c.offset - c.evicted
    assert c.evicted % GROUP == 0
    assert c.size() == c.visible
    assert c.visible == c.sink_used + GROUP * c.n_sealed + c.live_len
    if c.offset > c.max_size:
        # Context floor: never fewer than g_max resident record groups;
        # the transient upper bound scales with the caller's chunk size.
        assert c.n_sealed >= c.g_max
        bound = c.g_max + c.evict_slack + (chunk + GROUP - 1) // GROUP + 1
        assert c.n_sealed <= bound


def _assert_common_window_equal(a, b):
    """Bit-equality over the content both caches retain: the trailing
    min(n_sealed) record groups plus sink, live rows and tail. Hysteresis
    timing may legitimately differ, so absolute evicted counts can too."""
    assert a.offset == b.offset
    assert a.live_len == b.live_len
    assert a.evicted + GROUP * a.n_sealed == b.evicted + GROUP * b.n_sealed
    g = min(a.n_sealed, b.n_sealed)
    for f in ("codes_k", "codes_v", "axes_k", "axes_v"):
        x = np.array(getattr(a, f)[:, :, a.n_sealed - g : a.n_sealed])
        y = np.array(getattr(b, f)[:, :, b.n_sealed - g : b.n_sealed])
        assert np.array_equal(x, y), f
    for f in ("stage_k", "stage_v"):
        s = a.sink_cap + a.live_len
        assert np.array_equal(
            np.array(getattr(a, f)[:, :, :s]), np.array(getattr(b, f)[:, :, :s])
        ), f
    t = a.tail_len
    assert t == b.tail_len
    if t:
        for x, y in zip(a.tail_slices(t), b.tail_slices(t), strict=True):
            assert np.array_equal(np.array(x), np.array(y))


# -- construction ------------------------------------------------------------


def test_floor_decline():
    with pytest.raises(ValueError, match="window floor"):
        KVarNRotatingKVCache(639, tail_tokens=TAIL)
    with pytest.raises(ValueError, match="window floor"):
        KVarNRotatingKVCache(383, tail_tokens=0)
    assert KVarNRotatingKVCache(384, tail_tokens=0).g_max == 2
    assert _rot().g_max == 4


def test_window_geometry():
    c = _rot()
    assert (c.max_size, c.g_max, c.evicted) == (WIN, 4, 0)
    # max_size rounds down to whole groups
    assert KVarNRotatingKVCache(WIN + 127, tail_tokens=TAIL).g_max == 4
    # initial record capacity clamps to the window (steady-state peak:
    # one seal past the compaction trigger), not the 32-group step
    assert c._initial_gcap() == c.g_max + c.evict_slack + 1
    big = KVarNRotatingKVCache(128 * 1024, tail_tokens=TAIL)
    assert big._initial_gcap() == big.gcap_step


# -- eviction ----------------------------------------------------------------


@_NEEDS_GPU
@pytest.mark.parametrize("chunk", [13, 128, 700])
def test_eviction_watermarks(chunk):
    c = _rot()
    k, v = _tokens(2000)
    seen = 0
    for i in range(0, 2000, chunk):
        c.update_and_fetch(k[:, :, i : i + chunk], v[:, :, i : i + chunk])
        seen += k[:, :, i : i + chunk].shape[2]
        assert c.offset == seen  # offset stays the absolute count
        _assert_invariants(c, chunk=chunk)
    assert c.evicted > 0


@_NEEDS_GPU
def test_nbytes_plateau():
    c = _rot()
    k, v = _tokens(4000)
    _feed(c, k[:, :, :2000], v[:, :, :2000], 128)
    mid = c.nbytes
    _feed(c, k[:, :, 2000:], v[:, :, 2000:], 128)
    assert c.nbytes == mid  # bounded: capacity stops growing past wrap
    unbounded = KVarNKVCache(tail_tokens=TAIL)
    unbounded.update_and_fetch(k, v)
    assert c.nbytes < unbounded.nbytes


@_NEEDS_GPU
@pytest.mark.parametrize("d", [128, 256])
def test_incremental_matches_bulk_common_window(d):
    k, v = _tokens(2000, d=d)
    inc = _rot()
    _feed(inc, k, v, 13)
    bulk = _rot()
    bulk.update_and_fetch(k, v)
    _assert_invariants(inc)
    _assert_invariants(bulk)
    _assert_common_window_equal(inc, bulk)


@_NEEDS_GPU
def test_rotating_matches_truncated_plain():
    k, v = _tokens(2000)
    rot = _rot()
    _feed(rot, k, v, 128)
    plain = KVarNKVCache(tail_tokens=TAIL)
    plain.update_and_fetch(k, v)
    shift = rot.evicted // GROUP
    for f in ("codes_k", "codes_v", "axes_k", "axes_v"):
        x = np.array(getattr(rot, f)[:, :, : rot.n_sealed])
        y = np.array(getattr(plain, f)[:, :, shift : shift + rot.n_sealed])
        assert np.array_equal(x, y), f


@_NEEDS_GPU
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_wrap_boundary_group_edges(delta):
    # Land the stream right at the compaction trigger and one token either
    # side of it: sink + (g_max + slack + 1) full groups.
    edge = 128 + (4 + 4 + 1) * GROUP + delta
    c = _rot()
    k, v = _tokens(edge + 300)
    _feed(c, k[:, :, :edge], v[:, :, :edge], 128)
    _assert_invariants(c)
    _feed(c, k[:, :, edge:], v[:, :, edge:], 1)
    _assert_invariants(c)
    assert c.evicted > 0


@_NEEDS_GPU
def test_bulk_overflow_skips_doomed_groups():
    # A single prefill 3x the window: only the last g_max groups are ever
    # quantized (record capacity never grows past the initial clamp).
    c = _rot()
    k, v = _tokens(3 * WIN)
    c.update_and_fetch(k, v)
    _assert_invariants(c)
    assert c.n_sealed == c.g_max
    assert c.codes_k.shape[2] == c._initial_gcap()


# -- compaction timing -------------------------------------------------------


@_NEEDS_GPU
def test_compaction_fires_at_entry_only(monkeypatch):
    c = _rot()
    events = []
    compact, seal = c._compact, c._seal
    monkeypatch.setattr(c, "_compact", lambda: (events.append("compact"), compact())[1])
    monkeypatch.setattr(
        c, "_seal", lambda rk, rv: (events.append("seal"), seal(rk, rv))[1]
    )
    k, v = _tokens(2000)
    for i in range(0, 2000, 128):
        events.clear()
        c.update_and_fetch(k[:, :, i : i + 128], v[:, :, i : i + 128])
        # every update opens with the compaction check; seals never precede it
        assert events[0] == "compact"
        assert "compact" not in events[1:]


@_NEEDS_GPU
def test_context_floor_across_wrap():
    # Every incoming row must see >= the resident window (sink + g_max
    # groups): compaction is sized by the pre-append frontier, so visibility
    # only grows between entry and attention.
    c = _rot()
    k, v = _tokens(3000)
    floor = c.sink_cap + c.g_max * GROUP
    for i in range(0, 3000, 97):
        n = k[:, :, i : i + 97].shape[2]
        pre_offset = c.offset
        c.update_and_fetch(k[:, :, i : i + 97], v[:, :, i : i + 97])
        # visible - n is the post-compaction, pre-append width row 0 saw
        assert c.visible - n >= min(pre_offset, floor)


# -- mask contract -----------------------------------------------------------


def test_make_mask_strings_only():
    c = _rot()
    assert c.make_mask(8) == "causal"
    assert c.make_mask(1) is None
    with pytest.raises(ValueError, match="string masks only"):
        c.make_mask(8, return_array=True)
    with pytest.raises(ValueError, match="string masks only"):
        c.make_mask(8, window_size=256)


@_NEEDS_GPU
def test_materialize_width_is_visible():
    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    mk, mv = c.materialize()
    assert mk.shape[2] == mv.shape[2] == c.visible


# -- trim --------------------------------------------------------------------


@_NEEDS_GPU
def test_trim_truth_table_post_wrap():
    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    live = c.live_len
    assert c._can_trim(live)  # inside live rows
    assert c._can_trim(live + GROUP)  # horizon reopen
    # crossing into evicted history refuses (chat rebuilds instead)
    beyond = c.visible - c.sink_cap + 1
    assert not c._can_trim(beyond)
    assert c.trim(beyond) == 0
    assert c.offset == 2000
    # trimming into the sink is always valid and resets eviction
    assert c._can_trim(2000 - 64)
    assert c.trim(2000 - 64) == 2000 - 64
    assert (c.offset, c.evicted, c.n_sealed) == (64, 0, 0)


@_NEEDS_GPU
def test_trim_replay_common_window():
    k, v = _tokens(2000)
    ref = _rot()
    _feed(ref, k, v, 128)
    c = _rot()
    _feed(c, k, v, 128)
    n = c.live_len + GROUP  # horizon reopen across the last group seam
    assert c.trim(n) == n
    _feed(c, k[:, :, 2000 - n :], v[:, :, 2000 - n :], 128)
    _assert_common_window_equal(c, ref)


@_NEEDS_GPU
def test_mtp_verify_trim_post_wrap():
    # verify rejections trim <= 3 tokens off the frontier, never the
    # evicted region; replaying them lands bit-identically
    k, v = _tokens(1500)
    ref = _rot()
    _feed(ref, k, v, 1)
    c = _rot()
    _feed(c, k, v, 1)
    assert c.evicted > 0
    assert c.trim(3) == 3
    _feed(c, k[:, :, 1497:], v[:, :, 1497:], 1)
    _assert_common_window_equal(c, ref)


# -- serialization -----------------------------------------------------------


@_NEEDS_GPU
def test_state_meta_round_trip():
    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    r = KVarNRotatingKVCache.from_state(c.state, c.meta_state)
    assert (r.max_size, r.evicted, r.offset) == (c.max_size, c.evicted, c.offset)
    _assert_common_window_equal(r, c)
    r.update_and_fetch(*_tokens(5, seed=9))
    _assert_invariants(r)


@_NEEDS_GPU
def test_meta_arity_fail_closed_both_directions():
    c = _rot()
    k, v = _tokens(300)
    c.update_and_fetch(k, v)
    plain = KVarNKVCache(tail_tokens=TAIL)
    plain.update_and_fetch(k, v)
    with pytest.raises(ValueError, match="different kvarn cache class"):
        KVarNKVCache.from_state(c.state, c.meta_state)  # 13 into 11
    with pytest.raises(ValueError, match="different kvarn cache class"):
        KVarNRotatingKVCache.from_state(plain.state, plain.meta_state)  # 11 into 13


@_NEEDS_GPU
def test_save_load_prompt_cache_file(tmp_path):
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    from gmlx.cache.kvarn_cache import ensure_registered

    ensure_registered()
    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    path = str(tmp_path / "kvarn-rot.safetensors")
    save_prompt_cache(path, [c])
    (r,) = load_prompt_cache(path)
    assert type(r).__name__ == "KVarNRotatingKVCache"
    assert (r.max_size, r.evicted) == (c.max_size, c.evicted)
    _assert_common_window_equal(r, c)


@_NEEDS_GPU
def test_clone_lm_twin():
    from gmlx.cache.snapshot import _clone_lm_twin

    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    targets = []
    r = _clone_lm_twin(c, targets)
    mx.eval(*targets)
    assert type(r) is KVarNRotatingKVCache
    assert (r.max_size, r.evicted) == (c.max_size, c.evicted)
    _assert_common_window_equal(r, c)


# -- conversion --------------------------------------------------------------


@_NEEDS_GPU
def test_from_cache_pre_wrap():
    from mlx_lm.models.cache import RotatingKVCache

    k, v = _tokens(500)
    src = RotatingKVCache(max_size=WIN, keep=4)
    src.update_and_fetch(k, v)
    conv = KVarNRotatingKVCache.from_cache(src, tail_tokens=TAIL)
    ref = _rot()
    ref.update_and_fetch(k, v)
    assert conv.max_size == WIN
    _assert_common_window_equal(conv, ref)


def test_from_cache_wrapped_refuses():
    from mlx_lm.models.cache import RotatingKVCache

    src = RotatingKVCache(max_size=256, keep=4)
    k, v = _tokens(300)
    src.update_and_fetch(k, v)
    with pytest.raises(ValueError, match="wrapped rotating cache"):
        KVarNRotatingKVCache.from_cache(src, tail_tokens=0)


@_NEEDS_GPU
def test_policy_converts_the_rotating_arm(_ops_ok):
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from gmlx.gen.generation import convert_kvarn_cache

    model = _MakeCacheLess()
    # without the window the rotating entry stays untouched
    plain = [KVCache(), RotatingKVCache(max_size=WIN, keep=4)]
    convert_kvarn_cache(model, plain, None, TAIL)
    assert type(plain[0]) is KVarNKVCache
    assert type(plain[1]) is RotatingKVCache

    pc = [KVCache(), RotatingKVCache(max_size=WIN, keep=4)]
    policy = convert_kvarn_cache(model, pc, None, TAIL, rotating_window=WIN)
    assert policy.verdict == "full" and policy.n_quant == 2
    assert type(pc[0]) is KVarNKVCache
    assert type(pc[1]) is KVarNRotatingKVCache
    # a rotating cache built for some other window is not ours to convert
    other = [RotatingKVCache(max_size=WIN * 2, keep=4)]
    assert convert_kvarn_cache(
        model, other, None, TAIL, rotating_window=WIN).verdict == "dropped"


# -- setup plumbing ----------------------------------------------------------


class _Args:
    def __init__(self, **kw):
        self.model_type = kw.pop("model_type", "llama")
        self.head_dim = kw.pop("head_dim", 128)
        for k, v in kw.items():
            setattr(self, k, v)


class _MakeCacheLess:
    """llama-class shape: no make_cache, so --max-kv-size manufactures a
    rotating stack via mlx-lm's make_prompt_cache."""

    def __init__(self, n_layers=2, **kw):
        self.args = _Args(**kw)
        self.layers = [None] * n_layers


@pytest.fixture
def _ops_ok(monkeypatch):
    from gmlx.cache import kvarn_sdpa

    monkeypatch.setattr(kvarn_sdpa, "_probe_result", (None,))
    monkeypatch.delenv("GMLX_KVARN", raising=False)
    monkeypatch.delenv("GMLX_KVARN_BITS", raising=False)


@_NEEDS_GPU
def test_setup_converts_rotating_stack(_ops_ok, capsys):
    from gmlx.gen.generation import setup_kvarn_cache

    pc = setup_kvarn_cache(_MakeCacheLess(), None, 1024, 4096)
    assert pc is not None and len(pc) == 2
    assert all(type(c).__name__ == "KVarNRotatingKVCache" for c in pc)
    assert all(c.max_size == 4096 for c in pc)
    err = capsys.readouterr().err
    assert "[kv] kvarn6 tail=1024 window=4096 -> quantized 2/2 attn layers" in err


def test_setup_declines_below_floor(_ops_ok, capsys):
    from gmlx.gen.generation import setup_kvarn_cache

    assert setup_kvarn_cache(_MakeCacheLess(), None, 1024, 512) is None
    err = capsys.readouterr().err
    assert "window floor" in err and "kv_tail_tokens" in err


@_NEEDS_GPU
def test_setup_without_window_stays_plain(_ops_ok, capsys):
    from gmlx.gen.generation import setup_kvarn_cache

    pc = setup_kvarn_cache(_MakeCacheLess(), None, 1024, None)
    assert pc is not None
    assert all(type(c).__name__ == "KVarNKVCache" for c in pc)
    assert "window" not in capsys.readouterr().err


def test_kv_bits_rotating_reason():
    """Affine's counterpart to the kvarn rotating path: --max-kv-size on a
    make_cache-less model builds an all-window stack the shared policy
    refuses, while the same model without it quantizes."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from gmlx.cache.kv_policy import resolve_kv_quant_policy

    plain = resolve_kv_quant_policy([KVCache(), KVCache()], kv_bits=8)
    assert plain.verdict == "full"
    rot = resolve_kv_quant_policy(
        [RotatingKVCache(max_size=4096), RotatingKVCache(max_size=4096)],
        kv_bits=8,
        max_kv_size=4096,
    )
    assert rot.verdict == "error"
    assert "max_kv_size" in rot.reason and "quantize" in rot.reason


# -- decode parity -----------------------------------------------------------


def _ref_attention(q, cache, qL):
    """fp32 attention over the exact values the route attends: rotated
    materialized body plus rotated tail rows over the visible window."""
    import mlx_kquant as kq

    d = q.shape[-1]
    n = cache.visible
    t = min(cache.tail_len, n)
    if 0 < n - t < qL:
        t = n - qL
    mat_k, mat_v = cache.materialize()
    parts_k, parts_v = [mat_k[:, :, : n - t]], [mat_v[:, :, : n - t]]
    if t:
        tk, tv = cache.tail_slices(t)
        parts_k.append(kq.kvarn_rotate(tk))
        parts_v.append(kq.kvarn_rotate(tv))
    k = mx.concatenate(parts_k, axis=2).astype(mx.float32)
    v = mx.concatenate(parts_v, axis=2).astype(mx.float32)
    qr = kq.kvarn_rotate(q.astype(mx.float16)).astype(mx.float32)
    qg = qr.reshape(1, H, HQ // H, qL, d)
    s = (qg @ k[:, :, None].transpose(0, 1, 2, 4, 3)) * (d**-0.5)
    kpos = mx.arange(n)[None, None, None, None, :]
    qpos = (n - qL + mx.arange(qL))[None, None, None, :, None]
    s = mx.where(kpos <= qpos, s, mx.array(-np.inf, mx.float32))
    o = (mx.softmax(s, axis=-1) @ v[:, :, None]).reshape(1, HQ, qL, d)
    return kq.kvarn_rotate(o.astype(mx.float16)).astype(mx.float32)


@_NEEDS_GPU
@pytest.mark.parametrize("ql", [1, 2, 4])
def test_post_wrap_decode_matches_reference(ql):
    from gmlx.cache.kvarn_sdpa import kvarn_attention

    c = _rot()
    k, v = _tokens(2000)
    _feed(c, k, v, 128)
    assert c.evicted > 0
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((1, HQ, ql, D)).astype(np.float16))
    out = kvarn_attention(q, c, D**-0.5, None if ql == 1 else "causal")
    ref = _ref_attention(q, c, ql)
    d = np.abs(np.array(out.astype(mx.float32)) - np.array(ref)).max()
    assert d < 5e-3, f"max|d|={d}"


@_NEEDS_GPU
def test_pre_wrap_tail_saturated_split():
    # Small fill where the precision tail covers nearly everything: the
    # decode split widens the body to qL (base-cache behavior preserved).
    from gmlx.cache.kvarn_sdpa import kvarn_attention

    c = _rot()
    k, v = _tokens(WIN - 255)
    c.update_and_fetch(k, v)
    assert c.evicted == 0 and 0 < c.visible - c.tail_len < 4
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((1, HQ, 4, D)).astype(np.float16))
    out = kvarn_attention(q, c, D**-0.5, "causal")
    ref = _ref_attention(q, c, 4)
    d = np.abs(np.array(out.astype(mx.float32)) - np.array(ref)).max()
    assert d < 5e-3, f"max|d|={d}"
