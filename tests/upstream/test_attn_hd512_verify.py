"""hd512/hd256 speculative-verify routes: numerics vs stock SDPA + gating."""

import os

import mlx.core as mx
import pytest

import gmlx.upstream.attn_hd512 as attn_hd512

HQ, HKV, D = 32, 4, 512
KL = 4096
SCALE = D**-0.5



def _stock_sdpa():
    """The stock kernel, even when an earlier test left the hd512 wrapper
    installed on mx.fast: pinning _orig_sdpa to the wrapper itself makes
    the fallback recurse into the wrapper (an exponential hang under the
    chunked-prefill tile loop)."""
    fn = mx.fast.scaled_dot_product_attention
    return getattr(fn, "_gmlx_orig_sdpa", fn)

def _rand(qL, kL=KL, hq=HQ, hkv=HKV, d=D):
    # keyed by the shape, so a draw does not depend on which tests ran first
    kq, kk, kv = mx.random.split(mx.random.key(qL * 7919 + kL * 31 + d), 3)
    q = mx.random.normal((1, hq, qL, d), key=kq).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, kL, d), key=kk).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, kL, d), key=kv).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v


def _ref(q, k, v, causal, scale=SCALE):
    # f32 materialized reference with bottom-right causal alignment
    g = q.shape[1] // k.shape[1]
    qL, kL = q.shape[2], k.shape[2]
    kf = mx.repeat(k.astype(mx.float32), g, axis=1)
    vf = mx.repeat(v.astype(mx.float32), g, axis=1)
    s = (q.astype(mx.float32) * scale) @ kf.swapaxes(-1, -2)
    if causal:
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        s = mx.where(mx.arange(kL).reshape(1, kL) <= rows, s, -mx.inf)
    return mx.softmax(s, axis=-1) @ vf


@pytest.mark.parametrize("qL,kL", [(96, 96), (96, 224), (100, 228)])
def test_chunked_prefill_causal_with_cached_prefix(qL, kL, monkeypatch):
    # kL > qL is chunk 2+ of a chunked prefill into an accumulating cache:
    # every query row's causal horizon is offset by the cached prefix.
    monkeypatch.setattr(
        attn_hd512, "_orig_sdpa", _stock_sdpa())
    q, k, v = _rand(qL, kL=kL)
    out = attn_hd512._chunked_prefill(q, k, v, SCALE, "causal", tile=32)
    ref = _ref(q, k, v, True)
    assert out.shape == (1, HQ, qL, D)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qL={qL} kL={kL} err={err}"


@pytest.mark.parametrize("d", [96, 512])
@pytest.mark.parametrize("qL", [32, 96])
def test_chunked_prefill_unmasked_stays_unmasked(d, qL, monkeypatch):
    # mask=None is *unmasked*, not causal. A bidirectional encoder (the
    # muse-glimmer ViT, hd 96) attends to every key from every query row;
    # treating None as "causal" silently halved its receptive field.
    monkeypatch.setattr(
        attn_hd512, "_orig_sdpa", _stock_sdpa())
    scale = d**-0.5
    q, k, v = _rand(qL, kL=qL, hq=16, hkv=16, d=d)
    out = attn_hd512._chunked_prefill(q, k, v, scale, None, tile=32)
    err = mx.abs(out.astype(mx.float32)
                 - _ref(q, k, v, False, scale=scale)).max().item()
    assert err < 2e-2, f"d={d} qL={qL} err={err}"
    if qL > 32:
        causal_err = mx.abs(out.astype(mx.float32)
                            - _ref(q, k, v, True, scale=scale)).max().item()
        assert causal_err > 1e-2, "unmasked output collapsed onto the causal one"


def test_chunked_prefill_block_diagonal_mask(monkeypatch):
    # the ViT's window attention: a non-causal array mask, sliced per tile
    monkeypatch.setattr(
        attn_hd512, "_orig_sdpa", _stock_sdpa())
    qL, d = 96, 96
    q, k, v = _rand(qL, kL=qL, hq=16, hkv=16, d=d)
    seg = mx.arange(qL) // 32
    mask = (seg[:, None] == seg[None, :])[None, None]
    out = attn_hd512._chunked_prefill(q, k, v, d**-0.5, mask, tile=32)
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=d**-0.5, mask=mask)
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert err < 2e-2, f"err={err}"


@pytest.mark.parametrize("qL", [3, 4, 6])
@pytest.mark.parametrize("causal", [True, False])
def test_verify_gemm_matches_reference(qL, causal):
    q, k, v = _rand(qL)
    out = attn_hd512._verify_gemm(q, k, v, SCALE, causal)
    ref = _ref(q, k, v, causal)
    assert out.shape == (1, HQ, qL, D)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qL={qL} causal={causal} err={err}"


@pytest.mark.parametrize("qL", [3, 4])
def test_verify_gemm_hkv1(qL):
    # gemma-4-12b global layers (Hkv=1): numerically fine but never routed,
    # since stock's broadcast matmul is the same GEMM at one KV head
    q, k, v = _rand(qL, hq=16, hkv=1)
    out = attn_hd512._verify_gemm(q, k, v, SCALE, True)
    err = mx.abs(out.astype(mx.float32) - _ref(q, k, v, True)).max().item()
    assert err < 2e-2
    assert not attn_hd512._verify_gemm_eligible(q, k, v, "causal")


def test_verify_gemm_eligibility():
    ok = _rand(4)
    assert attn_hd512._verify_gemm_eligible(*ok, "causal")
    assert attn_hd512._verify_gemm_eligible(*ok, None)
    # every verify width at 4 KV heads, never decode or past qL 8
    assert attn_hd512._verify_gemm_eligible(*_rand(2), "causal")
    assert attn_hd512._verify_gemm_eligible(*_rand(8), "causal")
    assert not attn_hd512._verify_gemm_eligible(*_rand(1), "causal")
    assert not attn_hd512._verify_gemm_eligible(*_rand(9), "causal")
    # shallow KV stays on stock
    assert not attn_hd512._verify_gemm_eligible(*_rand(4, kL=256), "causal")
    # non-hd512 and array masks fall through
    assert not attn_hd512._verify_gemm_eligible(*_rand(4, d=256), "causal")
    q, k, v = ok
    arr_mask = mx.zeros((4, KL), dtype=mx.bool_)
    assert not attn_hd512._verify_gemm_eligible(q, k, v, arr_mask)


def test_fa_chunks():
    # split sizing: smallest kv-major split with (g/n)*qL <= tile cap
    assert attn_hd512._fa_chunks(4, 4, cap=32) == 1
    assert attn_hd512._fa_chunks(8, 4, cap=32) == 1
    assert attn_hd512._fa_chunks(16, 4, cap=32) == 2   # 122b on 32-row kernel
    assert attn_hd512._fa_chunks(16, 4, cap=64) == 1   # 122b on 64-row kernel
    assert attn_hd512._fa_chunks(16, 5, cap=32) == 4
    assert attn_hd512._fa_chunks(16, 5, cap=64) == 2
    assert attn_hd512._fa_chunks(12, 5, cap=32) == 2
    assert attn_hd512._fa_chunks(14, 5, cap=32) is None  # 14 % {2,4} != 0
    # live cap must be a real tile size when the kernel is present
    if attn_hd512._HAS_FA_VERIFY:
        assert attn_hd512._FA_MAX_ROWS in (32, 64)


def _routed_sdpa(q, k, v, scale):
    # causal call through the installed wrapper, with the routes it bumped
    attn_hd512.install_hd512_sdpa()
    before = attn_hd512.route_counts()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                               mask="causal")
    after = attn_hd512.route_counts()
    return out, [key[0] for key in after if after[key] > before.get(key, 0)]


_NEEDS_FA = pytest.mark.skipif(
    not attn_hd512._HAS_FA_VERIFY or os.environ.get("KQUANT_FORCE_CPU"),
    reason="mlx_kquant.sdpa_fa_verify unavailable")


@_NEEDS_FA
@pytest.mark.parametrize("hq,hkv,qL", [
    (24, 4, 4),   # in-tile fold (rows 24), the pre-existing single-call path
    (32, 2, 4),   # qwen3.5-122b geometry: gqa16 -> 64 rows -> 2 chunks
    (16, 2, 5),   # gqa8 x 5 = 40 rows -> 2 chunks
    (32, 2, 5),   # gqa16 x 5 = 80 rows -> 4 chunks
])
def test_fa_verify_fold_matches_reference(hq, hkv, qL):
    scale = 256**-0.5
    q, k, v = _rand(qL, kL=KL, hq=hq, hkv=hkv, d=256)
    assert attn_hd512._fa_verify_eligible(q, k, v, "causal")
    attn_hd512.install_hd512_sdpa()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                               mask="causal")
    ref = _ref(q, k, v, True, scale=scale)
    assert out.shape == (1, hq, qL, 256)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"hq/hkv={hq}/{hkv} qL={qL} err={err}"


@_NEEDS_FA
@pytest.mark.parametrize("hq,hkv,qL", [
    (16, 2, 4),   # gemma-4-26b-a4b globals: g8 x qL4 = one 32-row tile
    (16, 2, 3),   # 24 rows
    (8, 2, 6),    # gemma-4-e4b globals: g4 x qL6 = 24 rows
    (8, 2, 8),    # 32 rows
])
def test_fa_verify_hd512_fold_matches_reference(hq, hkv, qL):
    q, k, v = _rand(qL, kL=KL, hq=hq, hkv=hkv)
    assert attn_hd512._fa_verify_eligible(q, k, v, "causal")
    out, grew = _routed_sdpa(q, k, v, SCALE)
    assert grew and all(r.startswith("fa_verify") for r in grew), grew
    ref = _ref(q, k, v, True)
    assert out.shape == (1, hq, qL, D)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"hq/hkv={hq}/{hkv} qL={qL} err={err}"


@pytest.mark.parametrize("hq,hkv,qL,kL,want", [
    (32, 4, 1, 1023, None),     # 2+ KV heads: decode from 1024 keys
    (32, 4, 1, 1024, "vector"),
    (16, 2, 1, 1023, None),
    (16, 2, 1, 1024, "vector"),
    (8, 1, 1, 383, None),       # one KV head: every route starts at 384
    (8, 1, 1, 384, "vector"),
    (8, 1, 1, 16384, "vector"),  # 8 rows: vector to 16384 keys, decode too
    (8, 1, 1, 16385, None),
    (16, 1, 1, 8192, "vector"),  # 16 rows: to 8192
    (16, 1, 1, 8193, None),
    (8, 1, 2, 383, None),
    (8, 1, 2, 384, "vector"),
    (8, 1, 2, 8192, "vector"),
    (8, 1, 2, 8193, None),
    (8, 1, 3, 4096, "vector"),  # 24 rows: to 4096
    (8, 1, 3, 4097, None),
    (16, 1, 2, 1024, "vector"),  # 32 rows: to 1024
    (16, 1, 2, 1025, None),
    (16, 1, 3, 1024, None),     # 48 rows: stock at any depth
    (32, 4, 2, 383, None),
    (32, 4, 2, 384, "gemm"),    # 4 KV heads: the GEMM at every verify width
    (32, 4, 8, 65536, "gemm"),
    (16, 2, 2, 384, "vector"),  # 2 KV heads, up to 16 rows: vector
    (8, 2, 5, 384, "vector"),   # 20 rows: vector below 65536 keys, fa from there
    (8, 2, 5, 65535, "vector"),
    (8, 2, 5, 65536, "fa"),
    (16, 2, 3, 1023, None),     # 21 to 31 rows: fa from 1024
    (16, 2, 3, 1024, "fa"),
    (8, 2, 7, 1024, "fa"),
    (8, 2, 8, 512, "gemm"),     # 32 rows: GEMM below 4096, fa from there
    (16, 2, 4, 4095, "gemm"),
    (16, 2, 4, 4096, "fa"),
    (16, 2, 5, 512, "gemm"),    # over 32 rows: GEMM
    (16, 2, 9, 4096, None),     # past qL 8
    (12, 8, 2, 4096, None),     # query heads not a multiple of KV heads
    (8, 0, 2, 4096, None),      # no KV heads
])
def test_hd512_route_table(hq, hkv, qL, kL, want):
    # the table reads shapes only, so the operands stay unevaluated
    q = mx.zeros((1, hq, qL, 512), dtype=mx.bfloat16)
    k = mx.zeros((1, hkv, kL, 512), dtype=mx.bfloat16)
    assert attn_hd512._hd512_route(q, k) == want


def test_hd512_route_max_ql(monkeypatch):
    # GMLX_HD512_MAXQL caps kq.sdpa_vector; a vector shape past it goes to
    # stock, and the other routes do not move
    def route(hq, hkv, qL, kL):
        return attn_hd512._hd512_route(
            mx.zeros((1, hq, qL, 512), dtype=mx.bfloat16),
            mx.zeros((1, hkv, kL, 512), dtype=mx.bfloat16))

    monkeypatch.setattr(attn_hd512, "_MAX_QL", 3)
    assert route(8, 2, 3, 2048) == "vector"
    assert route(8, 2, 4, 2048) is None
    assert route(8, 1, 4, 2048) is None
    assert route(8, 2, 7, 2048) == "fa"
    assert route(32, 4, 4, 2048) == "gemm"


def test_gqa_decode_hd512_eligibility():
    # head_dim 512 decode takes kq.sdpa_decode_gqa from 32768 keys at 4 or
    # more KV heads or a group of 4 or less; 16/2 and one KV head stay on
    # _hd512_route
    def elig(hq, hkv, kL):
        return attn_hd512._gqa_decode_eligible(
            mx.zeros((1, hq, 1, 512), dtype=mx.bfloat16),
            mx.zeros((1, hkv, kL, 512), dtype=mx.bfloat16),
            mx.zeros((1, hkv, kL, 512), dtype=mx.bfloat16), None)

    if attn_hd512._GQA_MIN_KV_512 != 32768:
        pytest.skip("GMLX_GQA_SDPA_MINKV512 is set")
    assert elig(32, 4, 32768)
    assert elig(8, 2, 32768)
    assert not elig(8, 2, 32767)
    assert not elig(16, 2, 32768)
    assert not elig(8, 1, 32768)
    assert not elig(16, 1, 65536)


@_NEEDS_FA
@pytest.mark.parametrize("hq,hkv,qL,kL,route", [
    (8, 2, 4, 2048, "sdpa_vector"),
    (8, 2, 8, 4096, "fa_verify"),
    (32, 4, 8, 1024, "verify_gemm"),
    (16, 2, 6, 1024, "verify_gemm"),
    (16, 1, 2, 2048, "stock"),
    (8, 1, 1, 512, "sdpa_vector"),
    (16, 1, 1, 8193, "stock"),
    (8, 2, 1, 2048, "sdpa_vector"),
])
def test_hd512_routes_match_reference(hq, hkv, qL, kL, route):
    # each route of the table, through the installed wrapper
    q, k, v = _rand(qL, kL=kL, hq=hq, hkv=hkv)
    out, grew = _routed_sdpa(q, k, v, SCALE)
    assert grew and all(r.startswith(route) for r in grew), grew
    ref = _ref(q, k, v, qL > 1)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"{hq}/{hkv} qL={qL} kL={kL} err={err}"


def test_hd512_fa_shape_unmasked_is_stock():
    # fa is causal only: an unmasked call on a shape the table gives fa
    # lands on stock and attends to every key
    q, k, v = _rand(8, kL=KL, hq=8, hkv=2)
    assert attn_hd512._hd512_route(q, k) == "fa"
    attn_hd512.install_hd512_sdpa()
    before = attn_hd512.route_counts()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=None)
    after = attn_hd512.route_counts()
    grew = [key[0] for key in after if after[key] > before.get(key, 0)]
    assert grew == ["stock"], grew
    err = mx.abs(out.astype(mx.float32) - _ref(q, k, v, False)).max().item()
    assert err < 2e-2, f"err={err}"


def test_fa_verify_hd512_eligibility():
    # d-split tile is 32 rows regardless of the probed hd256 cap
    assert attn_hd512._fa_row_cap(512) == 32
    # Hkv==1 (gemma-4-12b globals) stays off, same as the verify_gemm gate
    assert not attn_hd512._fa_verify_eligible(
        *_rand(4, hq=16, hkv=1), "causal")
    # 4 KV heads belong to the GEMM, and fa takes causal masks only
    assert not attn_hd512._fa_verify_eligible(*_rand(2), "causal")
    fa_shape = _rand(4, hq=16, hkv=2)
    assert attn_hd512._fa_verify_eligible(*fa_shape, "causal")
    assert not attn_hd512._fa_verify_eligible(*fa_shape, None)


@_NEEDS_FA
def test_fa_verify_chunked_matches_single_call():
    # the 2-chunk fold must agree with the kernel's own in-tile answer on a
    # shape both can run (gqa8 x qL4 = 32 rows)
    scale = 256**-0.5
    q, k, v = _rand(4, kL=KL, hq=16, hkv=2, d=256)
    import mlx_kquant
    single = mlx_kquant.sdpa_fa_verify(
        q.reshape(1, 2, 32, 256), k, v, scale, 4).reshape(1, 16, 4, 256)
    qc = q.reshape(1, 2, 2, 16, 256)
    chunked = mx.concatenate(
        [mlx_kquant.sdpa_fa_verify(mx.contiguous(qc[:, :, i]), k, v, scale, 4)
         for i in range(2)], axis=2).reshape(1, 16, 4, 256)
    err = mx.abs(single.astype(mx.float32)
                 - chunked.astype(mx.float32)).max().item()
    assert err < 1e-3


@_NEEDS_FA
def test_fa_decode_matches_reference():
    # 122b decode shape: gqa16 (32/2) hd256 qL=1 routes to the fa kernel as a
    # 1-query fold; numerics vs the f32 reference
    scale = 256**-0.5
    q, k, v = _rand(1, kL=32768, hq=32, hkv=2, d=256)
    assert attn_hd512._fa_decode_eligible(q, k, v, None)
    attn_hd512.install_hd512_sdpa()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
    ref = _ref(q, k, v, False, scale=scale)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"err={err}"


def test_fa_decode_eligibility():
    # ratio gate: only wide groups (> 8) leave the stock/gqa-kernel path
    ok = _rand(1, kL=32768, hq=32, hkv=2, d=256)
    assert attn_hd512._fa_decode_eligible(*ok, None)
    assert attn_hd512._fa_decode_eligible(*ok, "causal")
    # ratio 8 stays put (stock hd256 vector is bandwidth-healthy there)
    assert not attn_hd512._fa_decode_eligible(
        *_rand(1, kL=32768, hq=16, hkv=2, d=256), None)
    # shallow KV stays on stock
    assert not attn_hd512._fa_decode_eligible(
        *_rand(1, kL=8192, hq=32, hkv=2, d=256), None)
    # verify width is the verify route's business
    assert not attn_hd512._fa_decode_eligible(
        *_rand(4, kL=32768, hq=32, hkv=2, d=256), None)
    # array masks fall through
    q, k, v = ok
    assert not attn_hd512._fa_decode_eligible(
        q, k, v, mx.zeros((1, 32768), dtype=mx.bool_))


def test_fa_verify_eligibility_oversized_fold():
    # gqa16 x qL4 (122b) is now eligible; indivisible folds stay out
    q, k, v = _rand(4, hq=32, hkv=2, d=256)
    assert attn_hd512._fa_verify_eligible(q, k, v, "causal")
    assert not attn_hd512._fa_verify_eligible(q, k, v, None)
    # g=27: odd and >64 rows unfolded -- no valid split at either tile cap
    q2, k2, v2 = _rand(4, hq=27, hkv=1, d=256)
    assert not attn_hd512._fa_verify_eligible(q2, k2, v2, "causal")


@_NEEDS_FA
@pytest.mark.parametrize("qL,kL", [(6, 512), (8, 512), (7, 4096), (8, 16384)])
def test_fa_verify_wide_fold_matches_reference(qL, kL):
    # qwen3.x full attention (24/4) at DFlash block widths: a fold over 32
    # rows is stock's materialized fallback, so fa claims it from 512 keys
    scale = 256**-0.5
    q, k, v = _rand(qL, kL=kL, hq=24, hkv=4, d=256)
    assert attn_hd512._fa_verify_eligible(q, k, v, "causal")
    out, grew = _routed_sdpa(q, k, v, scale)
    assert grew and all(r.startswith("fa_verify") for r in grew), grew
    ref = _ref(q, k, v, True, scale=scale)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qL={qL} kL={kL} err={err}"


@_NEEDS_FA
def test_fa_verify_qL2_wide_group_matches_reference():
    # qwen3.5-122b (32/2) MTP verifying 2 tokens: a 32-row fold, fa from 1024
    scale = 256**-0.5
    q, k, v = _rand(2, kL=1024, hq=32, hkv=2, d=256)
    out, grew = _routed_sdpa(q, k, v, scale)
    assert grew and all(r.startswith("fa_verify") for r in grew), grew
    ref = _ref(q, k, v, True, scale=scale)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"err={err}"


@_NEEDS_FA
@pytest.mark.parametrize("qL", [3, 4])
def test_narrow_fold_at_depth_routes_sdpa_vector(qL):
    # qwen3.5-9b (16/4) MTP: folds of 12 and 16 rows, where fa loses, stay
    # on kq.sdpa_vector at depth
    scale = 256**-0.5
    q, k, v = _rand(qL, kL=8192, hq=16, hkv=4, d=256)
    out, grew = _routed_sdpa(q, k, v, scale)
    assert grew == ["sdpa_vector"], grew
    ref = _ref(q, k, v, True, scale=scale)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qL={qL} err={err}"


@pytest.mark.parametrize("qL,hq,hkv,below,from_kv", [
    (8, 24, 4, 511, 512),     # 48 rows, one tile
    (5, 32, 2, 1023, 1024),   # 80 rows, two chunks
    (8, 16, 4, 511, 512),     # 32 rows at 4 KV heads
    (4, 16, 2, 1023, 1024),   # 32 rows at 2 KV heads
    (2, 32, 2, 1023, 1024),   # qL 2 with a 16-wide group
    (3, 24, 4, 1023, 1024),   # 18 rows at 4 KV heads
    (5, 8, 2, 8191, 8192),    # 20 rows at 2 KV heads
])
def test_fa_verify_eligibility_by_fold(qL, hq, hkv, below, from_kv):
    def elig(kL):
        return attn_hd512._fa_verify_eligible(
            *_rand(qL, kL=kL, hq=hq, hkv=hkv, d=256), "causal")

    assert not elig(below)
    assert elig(from_kv)


def test_fa_verify_eligibility_narrow_fold():
    def elig(qL, kL, hq, hkv, d=256):
        return attn_hd512._fa_verify_eligible(
            *_rand(qL, kL=kL, hq=hq, hkv=hkv, d=d), "causal")

    # folds under 18 rows stay on stock or kq.sdpa_vector at any depth
    assert not elig(4, 32768, 16, 4)
    assert not elig(8, 16384, 16, 8)
    assert not elig(2, 16384, 24, 4)
    # past the DFlash block width, and hd512 at 4 KV heads (the GEMM's),
    # fall through
    assert not elig(9, 512, 24, 4)
    assert not elig(6, KL, 32, 4, d=512)


@_NEEDS_FA
def test_wrapped_sdpa_routes_verify(monkeypatch):
    # the wrapper produces the GEMM result at verify width, and the kill
    # switches leave the GEMM's and fa's shapes to stock
    q, k, v = _rand(4)
    out, grew = _routed_sdpa(q, k, v, SCALE)
    assert grew == ["verify_gemm"], grew
    ref = _ref(q, k, v, True)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2
    monkeypatch.setattr(attn_hd512, "_VERIFY_GEMM", False)
    out2, grew = _routed_sdpa(q, k, v, SCALE)
    assert grew == ["stock"], grew
    err2 = mx.abs(out2.astype(mx.float32) - ref).max().item()
    assert err2 < 2e-2
    q3, k3, v3 = _rand(4, hq=16, hkv=2)
    monkeypatch.setattr(attn_hd512, "_VERIFY_FA", False)
    out3, grew = _routed_sdpa(q3, k3, v3, SCALE)
    assert grew == ["stock"], grew
    err3 = mx.abs(out3.astype(mx.float32) - _ref(q3, k3, v3, True)).max().item()
    assert err3 < 2e-2


@_NEEDS_FA
def test_route_counts_and_stock_warning(capsys, monkeypatch):
    attn_hd512.install_hd512_sdpa()
    before = attn_hd512.route_counts()
    q, k, v = _rand(4)
    mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE,
                                                 mask="causal"))
    after = attn_hd512.route_counts()
    grew = [key for key in after if after[key] > before.get(key, 0)]
    assert grew, "verify call did not bump any route counter"
    assert all(key[0] != "stock" for key in grew), f"verify landed stock: {grew}"

    # stock-at-depth tripwire: verify-shaped causal on stock warns ONCE per shape
    attn_hd512._STOCK_WARNED.clear()
    q2, k2, v2 = _rand(4, kL=16384, d=256, hq=32, hkv=2)
    attn_hd512._stock_depth_warning(q2, k2, "causal", None)
    attn_hd512._stock_depth_warning(q2, k2, "causal", None)
    err = capsys.readouterr().err
    assert err.count("stock materialized") == 1
    # decode (qL=1) and shallow KV never warn
    attn_hd512._STOCK_WARNED.clear()
    q3, k3, v3 = _rand(1, kL=16384, d=256, hq=32, hkv=2)
    attn_hd512._stock_depth_warning(q3, k3, "causal", None)
    q4, k4, v4 = _rand(4, kL=4096, d=256, hq=32, hkv=2)
    attn_hd512._stock_depth_warning(q4, k4, "causal", None)
    # head_dim 512: the table's stock pick (one KV head) stays quiet, a
    # shape the table routes elsewhere warns
    q5, k5, v5 = _rand(4, kL=16384, d=512, hq=8, hkv=1)
    assert attn_hd512._hd512_route(q5, k5) is None
    attn_hd512._stock_depth_warning(q5, k5, "causal", None)
    assert "stock materialized" not in capsys.readouterr().err
    q6, k6, v6 = _rand(4, kL=16384, d=512, hq=16, hkv=2)
    assert attn_hd512._hd512_route(q6, k6) == "fa"
    attn_hd512._stock_depth_warning(q6, k6, "causal", None)
    assert capsys.readouterr().err.count("stock materialized") == 1
    # a kill switch that sends its route's shapes to stock stays quiet
    attn_hd512._STOCK_WARNED.clear()
    monkeypatch.setattr(attn_hd512, "_VERIFY_FA", False)
    attn_hd512._stock_depth_warning(q6, k6, "causal", None)
    q7, k7, v7 = _rand(4, kL=16384, d=512, hq=32, hkv=4)
    assert attn_hd512._hd512_route(q7, k7) == "gemm"
    monkeypatch.setattr(attn_hd512, "_VERIFY_GEMM", False)
    attn_hd512._stock_depth_warning(q7, k7, "causal", None)
    assert "stock materialized" not in capsys.readouterr().err


def _stock_sdpa():
    # unwrap a prior install's wrapper to the true stock function
    fn = mx.fast.scaled_dot_product_attention
    return getattr(fn, "_gmlx_orig_sdpa", fn)


@pytest.mark.parametrize("arr_mask", [False, True])
def test_chunked_prefill_forwards_sinks(arr_mask, monkeypatch):
    stock = _stock_sdpa()
    monkeypatch.setattr(attn_hd512, "_orig_sdpa", stock)
    qL, kL = 96, 224
    q, k, v = _rand(qL, kL=kL)
    sinks = mx.random.normal((HQ,)).astype(mx.bfloat16)
    mx.eval(sinks)
    if arr_mask:
        rows = mx.arange(kL - qL, kL).reshape(qL, 1)
        keep = mx.arange(kL).reshape(1, kL) <= rows
        mask = mx.where(keep, 0.0, -mx.inf).astype(mx.bfloat16)[None, None]
    else:
        mask = "causal"
    tiled = attn_hd512._chunked_prefill(
        q, k, v, SCALE, mask, tile=32, sinks=sinks)
    ref = stock(q, k, v, scale=SCALE, mask=mask, sinks=sinks)
    err = mx.abs(tiled.astype(mx.float32)
                 - ref.astype(mx.float32)).max().item()
    assert err < 2e-2, f"arr_mask={arr_mask} err={err}"
    # and the result must differ from a sink-less pass (the old silent drop)
    dropped = stock(q, k, v, scale=SCALE, mask=mask)
    delta = mx.abs(ref.astype(mx.float32)
                   - dropped.astype(mx.float32)).max().item()
    assert delta > 1e-3


def test_wrapped_sdpa_prefill_route_forwards_sinks(monkeypatch):
    calls = []

    def rec(q, k, v, *, scale=1.0, mask=None, **kw):
        calls.append(kw)
        return q

    monkeypatch.setattr(attn_hd512, "_orig_sdpa", rec)
    q, k, v = _rand(100, kL=100)
    sinks = mx.zeros((HQ,), dtype=mx.bfloat16)
    attn_hd512._wrapped_sdpa(q, k, v, scale=SCALE, mask="causal", sinks=sinks)
    assert calls and all(c.get("sinks") is sinks for c in calls)
