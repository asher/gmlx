"""Shared-prefix cascade decode route: claim gates, fused-kernel numerics
vs the masked reference on stamped caches, ragged left pads, stamp
lifecycle (warm-multi detection, extend drop, filter survival), and
install hygiene.

Numerics require the mlx_kquant fused op (GPU); those tests skip on
platforms where the op is unavailable. Stamp-detection tests are pure
host logic and always run."""

import mlx.core as mx
import pytest

import importlib

from mlx_lm.models import llama as llama_mod

from gmlx import cascade_sdpa as cs

_orig_llama = llama_mod.scaled_dot_product_attention
_ORIG_SEAMS = []
for _pkg, _name in cs._MODULES:
    try:
        _mod = importlib.import_module(f"{_pkg}.{_name}")
    except ImportError:
        continue
    _ORIG_SEAMS.append((_mod, getattr(_mod, "scaled_dot_product_attention",
                                      None)))


def _has_cascade_op():
    try:
        from mlx_kquant import sdpa_decode_gqa_cascade  # noqa: F401
    except ImportError:
        return False
    return mx.default_device() == mx.Device(mx.gpu)


def teardown_module(module):
    for _mod, _fn in _ORIG_SEAMS:
        if _fn is not None:
            _mod.scaled_dot_product_attention = _fn
    cs._installed_route = False
    cs._installed_stamp = False


class _FakeBatchCache:
    def __init__(self, pads, P=None):
        self.left_padding = mx.array(pads)
        self._right_padding = None
        if P is not None:
            self._gmlx_cascade = {"P": P}

    def extend(self, other):
        return "extended"


def _install():
    assert cs.install_cascade_sdpa()


def _shared_batch(B, hq, hkv, P, sp, pads, d=128, dtype=mx.float16):
    """[B, hkv, maxpad+P+sp, d] buffer whose rows carry one shared prefix
    copy at [pads[b], pads[b]+P) and private tails after it."""
    mx.random.seed(11)
    L = max(pads) + P + sp
    kb = mx.random.normal((B, hkv, L, d)).astype(dtype)
    vb = mx.random.normal((B, hkv, L, d)).astype(dtype)
    pk = kb[0:1, :, pads[0]:pads[0] + P]
    pv = vb[0:1, :, pads[0]:pads[0] + P]
    rk, rv = [], []
    for b in range(B):
        rk.append(mx.concatenate(
            [kb[b:b + 1, :, :pads[b]], pk, kb[b:b + 1, :, pads[b] + P:]],
            axis=2))
        rv.append(mx.concatenate(
            [vb[b:b + 1, :, :pads[b]], pv, vb[b:b + 1, :, pads[b] + P:]],
            axis=2))
    k = mx.concatenate(rk, axis=0)
    v = mx.concatenate(rv, axis=0)
    q = mx.random.normal((B, hq, 1, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _masked_ref(q, k, v, pads, scale):
    B, hkv, L, d = k.shape
    hq = q.shape[1]
    kr = mx.repeat(k, hq // hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, hq // hkv, axis=1).astype(mx.float32)
    s = (q.astype(mx.float32) * scale) @ kr.swapaxes(-1, -2)
    pos = mx.arange(L)[None, :]
    keep = pos >= mx.array(pads)[:, None]
    bias = mx.where(keep, mx.zeros(keep.shape),
                    mx.full(keep.shape, -mx.inf))[:, None, None, :]
    return mx.softmax(s + bias, axis=-1) @ vr


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
@pytest.mark.parametrize("pads", [[0, 0, 0, 0], [0, 64, 128, 32]])
def test_route_matches_masked_reference(monkeypatch, pads):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P, sp = 2048, 96
    q, k, v = _shared_batch(4, 16, 8, P, sp, pads)
    cache = _FakeBatchCache(pads, P=P)
    n0 = cs.claims()
    got = llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cs.claims() == n0 + 1
    ref = _masked_ref(q, k, v, pads, 0.125)
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"cascade vs masked ref err={err}"


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_non_claims_fall_through(monkeypatch):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P = 1024
    q, k, v = _shared_batch(2, 16, 8, P, 64, [0, 0])
    n0 = cs.claims()

    # no stamp
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache([0, 0]), scale=0.125, mask=None)
    # B == 1
    llama_mod.scaled_dot_product_attention(
        q[:1], k[:1], v[:1], cache=_FakeBatchCache([0], P=P), scale=0.125,
        mask=None)
    # quantized cache (checked via _claim: the stock fallback would then
    # route the fake into the real quantized path and fail on its own)
    qc = _FakeBatchCache([0, 0], P=P)
    qc.bits = 8
    assert cs._claim(q, k, v, qc, 0.125, None, None) is None
    # P below MIN_P
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache([0, 0], P=128), scale=0.125, mask=None)
    # verify width (qL > 1)
    q2 = mx.concatenate([q, q], axis=2)
    llama_mod.scaled_dot_product_attention(
        q2, k, v, cache=_FakeBatchCache([0, 0], P=P), scale=0.125, mask=None)
    assert cs.claims() == n0


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
@pytest.mark.parametrize("qL,pads", [(4, [0, 0, 0]), (8, [0, 64, 32])])
def test_route_verify_width_matches_masked_reference(monkeypatch, qL, pads):
    """qL 2..8 claims: end-aligned causal on each row's private tail plus
    full shared visibility, ragged left pads included."""
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P, sp = 2048, 96
    B, hq, hkv = 3, 8, 4
    mx.random.seed(17)
    L = max(pads) + P + sp
    kb = mx.random.normal((B, hkv, L, 128)).astype(mx.float16)
    vb = mx.random.normal((B, hkv, L, 128)).astype(mx.float16)
    pk = kb[0:1, :, pads[0]:pads[0] + P]
    pv = vb[0:1, :, pads[0]:pads[0] + P]
    rk, rv = [], []
    for b in range(B):
        rk.append(mx.concatenate(
            [kb[b:b + 1, :, :pads[b]], pk, kb[b:b + 1, :, pads[b] + P:]],
            axis=2))
        rv.append(mx.concatenate(
            [vb[b:b + 1, :, :pads[b]], pv, vb[b:b + 1, :, pads[b] + P:]],
            axis=2))
    k = mx.concatenate(rk, axis=0)
    v = mx.concatenate(rv, axis=0)
    q = mx.random.normal((B, hq, qL, 128)).astype(mx.float16)
    cache = _FakeBatchCache(pads, P=P)
    n0 = cs.claims()
    got = llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.088, mask="causal")
    assert cs.claims() == n0 + 1

    kr = mx.repeat(k, hq // hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, hq // hkv, axis=1).astype(mx.float32)
    sc = (q.astype(mx.float32) * 0.088) @ kr.swapaxes(-1, -2)
    pos = mx.arange(L)[None, None, None, :]
    end = (L - qL) + mx.arange(qL)[None, None, :, None]
    pad = mx.array(pads)[:, None, None, None]
    keep = (pos >= pad) & (pos <= end)
    ref = mx.softmax(
        mx.where(keep, sc, mx.array(-mx.inf)), axis=-1) @ vr
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"verify-width cascade err={err}"

    # None mask at verify width = unmasked block: never claimed
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=_FakeBatchCache(pads, P=P), scale=0.088, mask=None)
    assert cs.claims() == n0 + 1
    # over the folded-row cap (3 * 2 * 8 = 48 ok; force with wider gqa)
    qw = mx.random.normal((B, 32, 8, 128)).astype(mx.float16)
    llama_mod.scaled_dot_product_attention(
        qw, k, v, cache=_FakeBatchCache(pads, P=P), scale=0.088,
        mask="causal")
    assert cs.claims() == n0 + 1


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_qwen_owned_path_claims(monkeypatch):
    """The owned qwen left-padded decode resolver routes stamped B>1
    qL==1 calls through the cascade claim before the ragged kernels."""
    from gmlx import qwen35_attn as qa

    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P, pads = 1024, [0, 64, 32]
    q, k, v = _shared_batch(3, 16, 8, P, 96, pads)
    cache = _FakeBatchCache(pads, P=P)
    n0 = cs.claims()
    got = qa._left_padded_attention(q, k, v, cache=cache, scale=0.125,
                                    mask=None)
    assert cs.claims() == n0 + 1
    ref = _masked_ref(q, k, v, pads, 0.125)
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qwen cascade vs masked ref err={err}"

    # kill switch: claim skipped, ragged path answers instead
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    cache2 = _FakeBatchCache(pads, P=P)
    got2 = qa._left_padded_attention(q, k, v, cache=cache2, scale=0.125,
                                     mask=None)
    assert cs.claims() == n0 + 1
    assert got2 is not None
    err2 = mx.abs(got2.astype(mx.float32) - ref).max().item()
    assert err2 < 2e-2

    # unstamped cache: falls through to ragged, no claim
    cache3 = _FakeBatchCache(pads)
    monkeypatch.delenv("GMLX_CASCADE_SDPA", raising=False)
    qa._left_padded_attention(q, k, v, cache=cache3, scale=0.125, mask=None)
    assert cs.claims() == n0 + 1


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_qwen_owned_verify_width_claims(monkeypatch):
    """The owned qwen verify resolver routes stamped B>1 qL>1 calls
    through the cascade claim ahead of the padded masked-sdpa branch."""
    from gmlx import qwen35_attn as qa

    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P, sp, qL = 1024, 96, 4
    B, hq, hkv = 2, 16, 8
    pads = [0, 48]
    _, k, v = _shared_batch(B, hq, hkv, P, sp, pads)
    mx.random.seed(23)
    q = mx.random.normal((B, hq, qL, 128)).astype(mx.float16)
    mx.eval(q)
    L = k.shape[-2]

    def _cache(stamped=True):
        c = _FakeBatchCache(pads, P=P if stamped else None)
        c._qwen3_5_decode_left_padding = list(pads)
        return c

    n0 = cs.claims()
    got = qa._verify_attention(q, k, v, cache=_cache(), scale=0.088,
                               mask=None)
    assert cs.claims() == n0 + 1

    # padded verify reference: left-pad visibility + end-aligned causal
    kr = mx.repeat(k, hq // hkv, axis=1).astype(mx.float32)
    vr = mx.repeat(v, hq // hkv, axis=1).astype(mx.float32)
    sc = (q.astype(mx.float32) * 0.088) @ kr.swapaxes(-1, -2)
    pos = mx.arange(L)[None, None, None, :]
    end = (L - qL) + mx.arange(qL)[None, None, :, None]
    pad = mx.array(pads)[:, None, None, None]
    keep = (pos >= pad) & (pos <= end)
    ref = mx.softmax(
        mx.where(keep, sc, mx.array(-mx.inf)), axis=-1) @ vr
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qwen verify-width cascade err={err}"

    # kill switch: masked-sdpa branch answers instead, no claim
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    got2 = qa._verify_attention(q, k, v, cache=_cache(), scale=0.088,
                                mask=None)
    assert cs.claims() == n0 + 1
    err2 = mx.abs(got2.astype(mx.float32) - ref).max().item()
    assert err2 < 2e-2
    monkeypatch.delenv("GMLX_CASCADE_SDPA", raising=False)

    # unstamped cache: no claim, masked branch still answers
    got3 = qa._verify_attention(q, k, v, cache=_cache(stamped=False),
                                scale=0.088, mask=None)
    assert cs.claims() == n0 + 1
    err3 = mx.abs(got3.astype(mx.float32) - ref).max().item()
    assert err3 < 2e-2

    # width-matched array masks claim too: this tree's masks are always
    # cache-derived pad+causal, and the claim re-derives from pads
    am = mx.zeros((B, 1, qL, L), dtype=q.dtype)
    got4 = qa._verify_attention(q, k, v, cache=_cache(), scale=0.088,
                                mask=am)
    assert cs.claims() == n0 + 2
    err4 = mx.abs(got4.astype(mx.float32) - ref).max().item()
    assert err4 < 2e-2

    # right padding: claim declines, masked branch answers
    c5 = _cache()
    c5._right_padding = [0, 1]
    qa._verify_attention(q, k, v, cache=c5, scale=0.088, mask=None)
    assert cs.claims() == n0 + 2


@pytest.mark.skipif(not _has_cascade_op(), reason="fused cascade op needs GPU")
def test_starts_memoized_on_pad_identity(monkeypatch):
    _install()
    monkeypatch.setenv("GMLX_CASCADE_MIN_P", "256")
    P = 1024
    pads = [0, 32]
    q, k, v = _shared_batch(2, 16, 8, P, 64, pads)
    cache = _FakeBatchCache(pads, P=P)
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    memo = cache._gmlx_casc_starts
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cache._gmlx_casc_starts is memo
    cache.left_padding = mx.array(pads)  # rebinding invalidates
    llama_mod.scaled_dot_product_attention(
        q, k, v, cache=cache, scale=0.125, mask=None)
    assert cache._gmlx_casc_starts is not memo


def test_install_idempotent_and_killable(monkeypatch):
    _install()
    route = llama_mod.scaled_dot_product_attention
    assert cs.install_cascade_sdpa()
    assert llama_mod.scaled_dot_product_attention is route
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    cs._installed_route = False
    assert not cs.install_cascade_sdpa()


# ---------------------------------------------------------------------------
# Stamp v3: token-prefix currency (host-only)
# ---------------------------------------------------------------------------


def _rb(ids):
    return cs._row_bytes(ids)


def test_carry_stamp():
    """Owned prefill merge rebuilds cache objects; carry_stamp moves the
    formation stamp onto the merged cache and re-arms the extend hook."""
    src = _FakeBatchCache([0, 0], P=512)
    dst = _FakeBatchCache([0, 0])
    cs.carry_stamp(src, dst)
    assert dst._gmlx_cascade == src._gmlx_cascade
    assert getattr(dst.extend, "_gmlx_cascade_merge", False)
    # unstamped src is a no-op
    dst2 = _FakeBatchCache([0, 0])
    cs.carry_stamp(_FakeBatchCache([0, 0]), dst2)
    assert "_gmlx_cascade" not in dst2.__dict__


def test_lcp_rows():
    a = list(range(100))
    assert cs._lcp_rows([_rb(a), _rb(a)]) == 100
    assert cs._lcp_rows([_rb(a), _rb(a[:50] + [999] + a[51:])]) == 50
    assert cs._lcp_rows([_rb(a), _rb(a[:30])]) == 30
    assert cs._lcp_rows([_rb(a)]) == 100
    assert cs._lcp_rows([]) == 0
    assert cs._lcp_rows([_rb(a), b""]) == 0


def test_stamp_caches_and_merge():
    pre = list(range(64))
    r1 = _rb(pre + [7, 8])
    r2 = _rb(pre + [9])
    cache = _FakeBatchCache([0, 0])
    assert cs._stamp_caches([cache], [r1, r2])
    assert cache._gmlx_cascade["P"] == 64

    # same-prefix admission keeps cascading at the common P
    other = _FakeBatchCache([0])
    cs._stamp_caches([other], [_rb(pre[:40] + [999])])
    assert cache.extend(other) == "extended"
    assert cache._gmlx_cascade["P"] == 40
    assert len(cache._gmlx_cascade["tok"]) == 3

    # unstamped admission clears
    assert cache.extend(_FakeBatchCache([0])) == "extended"
    assert not hasattr(cache, "_gmlx_cascade")


def test_stamp_divergent_clears():
    cache = _FakeBatchCache([0])
    cs._stamp_caches([cache], [_rb([1, 2, 3])])
    other = _FakeBatchCache([0])
    cs._stamp_caches([other], [_rb([9, 9, 9])])
    cache.extend(other)
    assert not hasattr(cache, "_gmlx_cascade")


def test_stamp_no_common_prefix_no_stamp():
    cache = _FakeBatchCache([0, 0])
    assert not cs._stamp_caches([cache], [_rb([1, 2]), _rb([3, 4])])
    assert not hasattr(cache, "_gmlx_cascade")


class _FilterCache(_FakeBatchCache):
    def filter(self, keep):
        return "filtered"


def test_stamp_filter_masks_rows():
    pre = list(range(32))
    rows = [_rb(pre + [i]) for i in range(3)]
    cache = _FilterCache([0, 0, 0])
    cs._stamp_caches([cache], rows)
    assert cache.filter([True, False, True]) == "filtered"
    info = cache._gmlx_cascade
    assert info["P"] == 32 and len(info["tok"]) == 2
    assert cache.filter([False, False]) == "filtered"
    assert not hasattr(cache, "_gmlx_cascade")


class _FakePromptBatch:
    def __init__(self, rows, apc_meta=None, rp=None):
        import mlx.core as mx

        self.prompt_cache = [_FakeBatchCache([0] * len(rows))]
        self._apc_meta = apc_meta
        L = max(len(r) for r in rows)
        self._left_padding_per_row = [L - len(r) for r in rows]
        self._right_pad_per_row = rp or [0] * len(rows)
        self._input_ids = mx.array(
            [[0] * (L - len(r)) + list(r) for r in rows])


def test_rows_from_prompt_batch_cold():
    pre = list(range(48))
    pb = _FakePromptBatch([pre + [7], pre])
    rows = cs._rows_from_prompt_batch(pb)
    assert rows == [_rb(pre + [7]), _rb(pre)]


def test_rows_from_prompt_batch_apc_meta_wins():
    full = list(range(64))
    pb = _FakePromptBatch([[1, 2], [3, 4]],
                          apc_meta=[{"full_input_ids": full},
                                    {"full_input_ids": full + [5]}])
    rows = cs._rows_from_prompt_batch(pb)
    assert rows == [_rb(full), _rb(full + [5])]
    # meta present but incomplete: bail rather than mix currencies
    pb2 = _FakePromptBatch([[1, 2], [3, 4]],
                           apc_meta=[{"full_input_ids": full}, {}])
    assert cs._rows_from_prompt_batch(pb2) is None


def test_stamped_ppb_init_wrapper():
    pre = list(range(40))

    def fake_init(self, rows=None):
        _FakePromptBatch.__init__(self, rows)

    wrapped = cs._make_stamped_ppb_init(fake_init)
    pb = _FakePromptBatch.__new__(_FakePromptBatch)
    wrapped(pb, rows=[pre + [1], pre + [2]])
    assert pb.prompt_cache[0]._gmlx_cascade["P"] == 40


def test_extend_cache_lift_carries_stamp():
    pre = list(range(50))

    class _Plain:
        pass

    def fake_extend_cache(a, b):
        out = []
        for ca, cb in zip(a, b):
            oc = _Plain()  # simulates the merge([ca]) lift: attrs dropped
            out.append(oc)
        return out

    wrapped = cs._make_stamped_extend_cache(fake_extend_cache)
    ca, cb = _FakeBatchCache([0]), _FakeBatchCache([0])
    cs._stamp_caches([ca], [_rb(pre + [1])])
    cs._stamp_caches([cb], [_rb(pre + [2])])
    out = wrapped([ca], [cb])
    assert out[0]._gmlx_cascade["P"] == 50
    assert len(out[0]._gmlx_cascade["tok"]) == 2
    # unstamped side clears
    out2 = wrapped([ca], [_FakeBatchCache([0])])
    assert not hasattr(out2[0], "_gmlx_cascade")


def test_stamp_install_killable(monkeypatch):
    monkeypatch.setenv("GMLX_CASCADE_SDPA", "0")
    cs._installed_stamp = False
    assert not cs.install_cascade_stamp()


def test_stamp_install_real_seams(monkeypatch):
    monkeypatch.delenv("GMLX_CASCADE_SDPA", raising=False)
    cs._installed_stamp = False
    from mlx_vlm.generate import ar

    orig_init = ar.PromptProcessingBatch.__init__
    orig_ext = ar._extend_cache
    try:
        assert cs.install_cascade_stamp()
        assert getattr(ar.PromptProcessingBatch.__init__,
                       "_gmlx_cascade_stamp", False)
        assert getattr(ar._extend_cache, "_gmlx_cascade_stamp", False)
        assert cs.install_cascade_stamp()  # idempotent
    finally:
        ar.PromptProcessingBatch.__init__ = orig_init
        ar._extend_cache = orig_ext
        cs._installed_stamp = False
