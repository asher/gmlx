"""hd512/hd256 speculative-verify routes: numerics vs stock SDPA + gating."""

import os

import mlx.core as mx
import pytest

from gmlx import attn_hd512

HQ, HKV, D = 32, 4, 512
KL = 4096
SCALE = D**-0.5


def _rand(qL, kL=KL, hq=HQ, hkv=HKV, d=D):
    q = mx.random.normal((1, hq, qL, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, kL, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, kL, d)).astype(mx.bfloat16)
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
        attn_hd512, "_orig_sdpa", mx.fast.scaled_dot_product_attention)
    q, k, v = _rand(qL, kL=kL)
    out = attn_hd512._chunked_prefill(q, k, v, SCALE, "causal", tile=32)
    ref = _ref(q, k, v, True)
    assert out.shape == (1, HQ, qL, D)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"qL={qL} kL={kL} err={err}"


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
    # gemma-4-12b global layers (Hkv=1): numerically fine but gated OFF --
    # no GQA amplification to remove, stock is within +-8% everywhere
    q, k, v = _rand(qL, hq=16, hkv=1)
    out = attn_hd512._verify_gemm(q, k, v, SCALE, True)
    err = mx.abs(out.astype(mx.float32) - _ref(q, k, v, True)).max().item()
    assert err < 2e-2
    assert not attn_hd512._verify_gemm_eligible(q, k, v, "causal")


def test_verify_gemm_eligibility():
    ok = _rand(4)
    assert attn_hd512._verify_gemm_eligible(*ok, "causal")
    assert attn_hd512._verify_gemm_eligible(*ok, None)
    # qL gates: 2 stays on the vector route, decode (1) never routes here
    assert not attn_hd512._verify_gemm_eligible(*_rand(2), "causal")
    assert not attn_hd512._verify_gemm_eligible(*_rand(1), "causal")
    assert not attn_hd512._verify_gemm_eligible(*_rand(7), "causal")
    # shallow KV stays on stock
    assert not attn_hd512._verify_gemm_eligible(*_rand(4, kL=2048), "causal")
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
    (16, 2, 4),   # gemma-4-26b-moe globals: g8 x qL4 = one 32-row tile
    (32, 4, 4),   # gemma-4-31b globals: same fold
    (32, 4, 5),   # g8 x qL5 = 40 rows -> 2 chunks on the 32-row d-split tile
])
def test_fa_verify_hd512_fold_matches_reference(hq, hkv, qL):
    q, k, v = _rand(qL, kL=KL, hq=hq, hkv=hkv)
    assert attn_hd512._fa_verify_eligible(q, k, v, "causal")
    attn_hd512.install_hd512_sdpa()
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE,
                                               mask="causal")
    ref = _ref(q, k, v, True)
    assert out.shape == (1, hq, qL, D)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2, f"hq/hkv={hq}/{hkv} qL={qL} err={err}"


def test_fa_verify_hd512_eligibility():
    # d-split tile is 32 rows regardless of the probed hd256 cap
    assert attn_hd512._fa_row_cap(512) == 32
    # Hkv==1 (gemma-4-12b globals) stays off, same as the verify_gemm gate
    assert not attn_hd512._fa_verify_eligible(
        *_rand(4, hq=16, hkv=1), "causal")
    # decode/qL2 widths and non-causal fall through
    assert not attn_hd512._fa_verify_eligible(*_rand(2), "causal")
    assert not attn_hd512._fa_verify_eligible(*_rand(4), None)


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
def test_wrapped_sdpa_routes_verify(monkeypatch):
    # the wrapper must produce the GEMM result at verify width...
    attn_hd512.install_hd512_sdpa()
    q, k, v = _rand(4)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE,
                                               mask="causal")
    ref = _ref(q, k, v, True)
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err < 2e-2
    # ...and honor the kill switch (falls back to a non-GEMM route)
    monkeypatch.setattr(attn_hd512, "_VERIFY_GEMM", False)
    out2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE,
                                                mask="causal")
    err2 = mx.abs(out2.astype(mx.float32) - ref).max().item()
    assert err2 < 2e-2


@_NEEDS_FA
def test_route_counts_and_stock_warning(capsys):
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
