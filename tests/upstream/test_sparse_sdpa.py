"""Sparse top-k page decode route: claim gates, selection determinism
(route output == a manual paged-kernel call on the same selected pages),
forced sink/local residency, block-mean memo lifecycle, and install
hygiene (opt-in, idempotent, killable).

Numerics need the mlx_kquant paged op (GPU); those tests skip where it
is unavailable. Selection and memo tests are host logic and always run.
"""

import importlib

import mlx.core as mx
import pytest

from mlx_lm.models import llama as llama_mod

from gmlx import sparse_sdpa as sp

_SEAM_MODS = []
for _pkg, _name in sp._MODULES:
    try:
        _mod = importlib.import_module(f"{_pkg}.{_name}")
    except ImportError:
        continue
    if getattr(_mod, "scaled_dot_product_attention", None) is not None:
        _SEAM_MODS.append(_mod)

_ENTRY_SEAMS = {}


def _has_paged_op():
    try:
        from mlx_kquant import sdpa_decode_gqa_paged  # noqa: F401
    except ImportError:
        return False
    return mx.default_device() == mx.Device(mx.gpu)


@pytest.fixture(scope="module", autouse=True)
def _fresh_route_state():
    """Other suites can leave sp._installed latched while the seam symbols
    hold a chain without the sparse route (a teardown elsewhere rebound
    them). Clear the latch so installs wrap fresh on whatever chain exists,
    and restore the entry state on exit instead of severing chains to
    import-time symbols. Tests that need route-free seams mid-module reset
    to the entry snapshot, not to stock."""
    _ENTRY_SEAMS.clear()
    _ENTRY_SEAMS.update(
        {m: m.scaled_dot_product_attention for m in _SEAM_MODS})
    snap_latch = sp._installed
    sp._installed = False
    yield
    for m, fn in _ENTRY_SEAMS.items():
        m.scaled_dot_product_attention = fn
    sp._installed = snap_latch


class _FakeCache:
    left_padding = None
    _right_padding = None


def _install(monkeypatch):
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    assert sp.install_sparse_sdpa()


def _mk(B, hq, hkv, S, d=128, dtype=mx.float16, seed=5):
    mx.random.seed(seed)
    q = mx.random.normal((B, hq, 1, d)).astype(dtype)
    k = mx.random.normal((B, hkv, S, d)).astype(dtype)
    v = mx.random.normal((B, hkv, S, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


@pytest.mark.skipif(not _has_paged_op(), reason="paged op needs GPU")
def test_route_matches_manual_paged_call(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setenv("GMLX_SPARSE_MIN_S", "1024")
    monkeypatch.setenv("GMLX_SPARSE_K", "512")
    from mlx_kquant import sdpa_decode_gqa_paged

    q, k, v = _mk(2, 16, 8, 4096)
    cache = _FakeCache()
    n0 = sp.claims()
    got = llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert sp.claims() == n0 + 1
    # same cache => memoized means => identical selection => identical op
    pages = sp._select_pages(q, k, cache, 32, 512 // 32 + 3)
    want = sdpa_decode_gqa_paged(q, k, v, 0.125, pages)
    mx.eval(got, want)
    assert mx.array_equal(got, want)


@pytest.mark.skipif(not _has_paged_op(), reason="paged op needs GPU")
def test_non_claims_fall_through(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setenv("GMLX_SPARSE_MIN_S", "1024")
    monkeypatch.setenv("GMLX_SPARSE_K", "512")
    q, k, v = _mk(1, 16, 8, 4096)
    n0 = sp.claims()

    # no cache
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=None, scale=0.125, mask=None)
    # below MIN_S
    monkeypatch.setenv("GMLX_SPARSE_MIN_S", "65536")
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeCache(), scale=0.125, mask=None)
    monkeypatch.setenv("GMLX_SPARSE_MIN_S", "1024")
    # budget covers the whole cache -> pointless, decline
    monkeypatch.setenv("GMLX_SPARSE_K", "4096")
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeCache(), scale=0.125, mask=None)
    monkeypatch.setenv("GMLX_SPARSE_K", "512")
    # verify width
    q2 = mx.concatenate([q, q], axis=2)
    llama_mod.scaled_dot_product_attention(
        q2, k, v, cache=_FakeCache(), scale=0.125, mask=None)
    # quantized cache
    qc = _FakeCache()
    qc.bits = 8
    assert sp._claim(q, k, v, qc, 0.125, None, None) is None
    # a heavily padded row leaves too little effective context
    pc = _FakeCache()
    pc.left_padding = mx.array([0, 3500])
    qb, kb, vb = _mk(2, 16, 8, 4096)
    assert sp._claim(qb, kb, vb, pc, 0.125, None, None) is None
    assert sp.claims() == n0


@pytest.mark.skipif(not _has_paged_op(), reason="paged op needs GPU")
def test_route_left_padded_matches_manual_call(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setenv("GMLX_SPARSE_MIN_S", "1024")
    monkeypatch.setenv("GMLX_SPARSE_K", "512")
    from mlx_kquant import sdpa_decode_gqa_paged

    pads = [0, 100, 513]
    q, k, v = _mk(3, 16, 8, 4096)
    cache = _FakeCache()
    cache.left_padding = mx.array(pads)
    n0 = sp.claims()
    got = llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert sp.claims() == n0 + 1
    pages = sp._select_pages(q, k, cache, 32, 512 // 32 + 3, pads)
    want = sdpa_decode_gqa_paged(q, k, v, 0.125, pages,
                                 starts=mx.array(pads, dtype=mx.int32))
    mx.eval(got, want)
    assert mx.array_equal(got, want)


def test_select_pages_per_row_pads():
    pads = [0, 640, 1000]
    q, k, _ = _mk(3, 16, 8, 4096)
    nkeep = 15
    pages = sp._select_pages(q, k, _FakeCache(), 32, nkeep, pads)
    for b, row_pages in enumerate(pages.tolist()):
        sink = pads[b] // 32
        dead_below = pads[b] // 32  # pages < this hold only pad bytes
        for row in row_pages:
            assert sink in row
            assert all(pg >= dead_below for pg in row)


def test_select_pages_forced_residency():
    q, k, _ = _mk(1, 16, 8, 4096)
    cache = _FakeCache()
    nkeep = 19
    pages = sp._select_pages(q, k, cache, 32, nkeep)
    assert pages.shape == (1, 8, nkeep)
    assert pages.dtype == mx.int32
    pl = pages.tolist()[0]
    nb = 4096 // 32
    for row in pl:
        assert row == sorted(row)
        assert 0 in row and nb - 1 in row and nb - 2 in row
        assert len(set(row)) == nkeep


def test_select_pages_partial_tail_forced():
    q, k, _ = _mk(1, 16, 8, 4001)  # 126 pages, last one partial
    pages = sp._select_pages(q, k, _FakeCache(), 32, 11)
    nb_total = (4001 + 31) // 32
    for row in pages.tolist()[0]:
        assert nb_total - 1 in row and nb_total - 2 in row


def test_means_memo_lifecycle():
    cache = _FakeCache()
    _, k, _ = _mk(1, 16, 8, 4096)
    m1 = sp._block_means(cache, k, 32)
    assert m1.shape == (1, 8, 128, 128)
    memo1 = cache._gmlx_sp_means

    # append: only new full blocks are computed, prefix object is reused
    k2 = mx.concatenate([k, mx.ones((1, 8, 64, 128), dtype=k.dtype)], axis=2)
    m2 = sp._block_means(cache, k2, 32)
    assert m2.shape[2] == 130
    assert mx.array_equal(m2[:, :, :128], m1)

    # shrink (trim/filter) invalidates the memo wholesale
    m3 = sp._block_means(cache, k, 32)
    assert cache._gmlx_sp_means is not memo1
    assert m3.shape[2] == 128

    # batch-width change invalidates too
    kb = mx.concatenate([k2, k2], axis=0)
    m4 = sp._block_means(cache, kb, 32)
    assert m4.shape[0] == 2


def test_install_patches_validated_archs_only(monkeypatch):
    import importlib

    sp._installed = False
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    monkeypatch.delenv("GMLX_SPARSE_ARCHS", raising=False)
    for _mod, _fn in _ENTRY_SEAMS.items():
        _mod.scaled_dot_product_attention = _fn
    assert sp.install_sparse_sdpa()
    assert getattr(llama_mod.scaled_dot_product_attention,
                   "_gmlx_sparse_route", False)
    gt = importlib.import_module("mlx_lm.models.gemma4_text")
    # gemma-4 failed its KLD gate (globals not top-k sparse): never patched
    assert not getattr(gt.scaled_dot_product_attention,
                       "_gmlx_sparse_route", False)
    # explicit env extension patches it (for running a new arch's gate)
    sp._installed = False
    monkeypatch.setenv("GMLX_SPARSE_ARCHS", "gemma4_text")
    assert sp.install_sparse_sdpa()
    assert getattr(gt.scaled_dot_product_attention,
                   "_gmlx_sparse_route", False)


def test_install_opt_in_and_killable(monkeypatch):
    sp._installed = False
    monkeypatch.delenv("GMLX_SPARSE_ATTN", raising=False)
    assert not sp.install_sparse_sdpa()  # default OFF: the route is lossy
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "0")
    assert not sp.install_sparse_sdpa()
    monkeypatch.setenv("GMLX_SPARSE_ATTN", "1")
    assert sp.install_sparse_sdpa()
    route = llama_mod.scaled_dot_product_attention
    assert sp.install_sparse_sdpa()  # idempotent
    assert llama_mod.scaled_dot_product_attention is route
