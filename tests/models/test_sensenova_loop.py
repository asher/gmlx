"""SenseNova Loop: the looped file layout, one cache per layer visit, and
the two-stream mHC routing against llama.cpp's arithmetic."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import mlx.core as mx

sys.path.insert(0, os.path.dirname(__file__))
import qwen35_tiny as T  # noqa: E402

from gmlx.load.config_synth import synthesize_config  # noqa: E402
from gmlx.load.gguf_layout import fold_sensenova_loop  # noqa: E402

ARCH = "sensenova_loop"
IDS = [[5, 9, 30, 41, 7, 66, 12, 3, 90, 100, 15, 22]]


def _reference_coef(logits: np.ndarray, iterations: int) -> np.ndarray:
    """llama.cpp build_route in float64: sigmoid pre and post gains, then
    Sinkhorn rounds of two-element log-softmax, rows then columns."""
    sig = 1.0 / (1.0 + np.exp(-logits[..., :4]))
    m = [logits[..., 4 + i].astype(np.float64) for i in range(4)]

    def norm(x, y):
        z = np.logaddexp(x, y)
        return x - z, y - z

    for _ in range(iterations):
        m[0], m[1] = norm(m[0], m[1])
        m[2], m[3] = norm(m[2], m[3])
        m[0], m[2] = norm(m[0], m[2])
        m[1], m[3] = norm(m[1], m[3])
    mix = np.stack([np.exp(x) for x in m], axis=-1)
    return np.concatenate([sig[..., :2], 2 * sig[..., 2:4], mix], axis=-1)


def test_the_router_coefficients_match_llama_cpp():
    from gmlx.models.sensenova_loop.mhc import _coef_fn

    logits = np.random.default_rng(0).standard_normal((3, 5, 8)) * 3
    got = np.array(_coef_fn(20, False)(mx.array(logits, dtype=mx.float32)))
    want = _reference_coef(logits, 20)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-6)
    # Each round ends on the columns.
    mix = got[..., 4:].reshape(3, 5, 2, 2)
    np.testing.assert_allclose(mix.sum(-2), 1.0, atol=1e-5)


def _shapes():
    return {n: list(reversed(a.shape))
            for n, a in T._tensors(ARCH, T.values(ARCH), True)}


def test_the_synth_reads_the_loop_and_the_mhc_range():
    c = synthesize_config(T.metadata(ARCH), tensor_shapes=_shapes())
    assert c["model_type"] == ARCH
    assert c["num_hidden_layers"] == T.LAYERS
    assert c["mtp_num_hidden_layers"] == 1
    assert c["loop"] == {"start": T.START, "end": T.END, "count": T.PASSES}
    assert c["mhc"]["start"] == T.START and c["mhc"]["end"] == T.END


def test_the_synth_refuses_a_visit_that_breaks_the_attention_pattern():
    meta = T.metadata(ARCH)
    pattern = list(meta[f"{ARCH}.attention.recurrent_layers"])
    pattern[T.END] = not pattern[T.END]
    meta[f"{ARCH}.attention.recurrent_layers"] = pattern
    with pytest.raises(ValueError, match="full-attention pattern"):
        synthesize_config(meta, tensor_shapes=_shapes())


def test_the_fold_puts_each_block_at_its_physical_index():
    visits = len(T.visits())
    after = list(range(visits - (T.LAYERS - T.END), visits))
    names = ([f"blk.{b}.attn_norm.weight" for b in range(T.END)]
             + [f"blk.{b}.attn_norm.weight" for b in after]
             + [f"blk.{visits}.nextn.eh_proj.weight"])
    arrays = {n: mx.zeros((1,)) for n in names}
    fold_sensenova_loop(arrays, {}, T.metadata(ARCH), {})
    assert sorted(arrays) == sorted(
        [f"blk.{b}.attn_norm.weight" for b in range(T.LAYERS)]
        + [f"blk.{T.LAYERS}.nextn.eh_proj.weight"])

    arrays = {f"blk.{T.END}.attn_norm.weight": mx.zeros((1,))}
    with pytest.raises(ValueError, match="repeated loop visit"):
        fold_sensenova_loop(arrays, {}, T.metadata(ARCH), {})


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    # Float32 activations: the random routers amplify 16-bit rounding far
    # past what a decode check can tell from a wrong cache.
    import gmlx

    path = tmp_path_factory.mktemp(ARCH) / "tiny.gguf"
    T.write(path, ARCH, T.values(ARCH), fused=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GMLX_ACTIVATION_DTYPE", "float32")
        return gmlx.load_model(str(path), verbose=False)[0]


def test_each_visit_has_its_own_cache_and_the_weights_are_physical(model):
    from mlx.utils import tree_flatten

    assert len(model.layers) == len(T.visits())
    assert len(model.make_cache()) == len(T.visits())
    blocks = {k.split(".")[2] for k, _ in tree_flatten(model.parameters())
              if k.startswith("model.layers.")}
    assert blocks == {str(i) for i in range(T.LAYERS)}
    routers = [(k, a.dtype) for k, a in tree_flatten(model.parameters())
               if ".mhc_" in k]
    assert len(routers) == 4 * (T.END - T.START)
    assert all(d == mx.float32 for _, d in routers)
    assert model.model.layers[0].input_layernorm.weight.dtype == mx.float32


def test_decode_on_the_cache_matches_the_full_forward(model):
    # On the CPU, where float32 products are exact.
    ids = mx.array(IDS)
    with mx.stream(mx.cpu):
        full = np.array(model(ids)[:, -3:])
        cache = model.make_cache()
        model(ids[:, :-3], cache=cache)
        steps = [model(ids[:, i:i + 1], cache=cache)
                 for i in range(ids.shape[1] - 3, ids.shape[1])]
        got = np.array(mx.concatenate(steps, axis=1))
    np.testing.assert_allclose(got, full, rtol=1e-4, atol=1e-4)


_GPU = pytest.mark.skipif(mx.default_device() != mx.gpu,
                          reason="the mHC kernels are Metal-only")


@_GPU
@pytest.mark.parametrize("settings", [
    {"iterations": 20},
    {"iterations": 3, "fixed_identity": True, "pre_scale": 0.5},
], ids=["sinkhorn", "identity-scaled"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("rows", [1, 5])
def test_the_mhc_kernels_give_the_ops_path_bits(monkeypatch, settings, dtype, rows):
    from gmlx.models.sensenova_loop import mhc

    hidden = 96
    router = mhc.MHCRouter(hidden, settings)
    n = router.weight.shape[0]
    rng = np.random.default_rng(1)
    router.weight = mx.array(rng.standard_normal((n, 2 * hidden)), mx.float32) * 2
    router.bias = mx.array(rng.standard_normal(n), mx.float32)
    s = (mx.array(rng.standard_normal((1, rows, 2 * hidden)), mx.float32) * 3).astype(dtype)
    y = mx.array(rng.standard_normal((1, rows, hidden)), mx.bfloat16)

    monkeypatch.setenv("GMLX_SENSENOVA_MHC_FUSED", "0")

    def run(kernel):
        monkeypatch.setenv("GMLX_SENSENOVA_MHC_KERNEL", kernel)
        x, coef = router(s, mx.bfloat16)
        out = mhc.merge(s, y, coef)
        assert out.dtype == dtype and x.dtype == mx.bfloat16
        return [np.array(v.astype(mx.float32)) for v in (x, coef, out)]

    for ops, fused in zip(run("0"), run("1")):
        np.testing.assert_array_equal(ops, fused)


@_GPU
def test_the_model_gives_the_same_logits_with_and_without_the_kernels(
        tmp_path, monkeypatch):
    import gmlx

    path = tmp_path / "tiny.gguf"
    T.write(path, ARCH, T.values(ARCH), fused=True)
    model = gmlx.load_model(str(path), verbose=False)[0]
    ids = mx.array(IDS)
    monkeypatch.setenv("GMLX_SENSENOVA_MHC_FUSED", "0")
    out = {}
    for kernel in ("0", "1"):
        monkeypatch.setenv("GMLX_SENSENOVA_MHC_KERNEL", kernel)
        out[kernel] = np.array(model(ids).astype(mx.float32))
    np.testing.assert_array_equal(out["0"], out["1"])


@pytest.mark.parametrize("rows", [2, 5])
def test_a_verify_width_router_gives_each_row_the_one_row_bits(rows):
    # A few-row float32 matmul runs a slow TF32 GEMM tile on M5; verify
    # widths run the router linear one matvec a row instead.
    from gmlx.models.sensenova_loop import mhc

    hidden = 512
    router = mhc.MHCRouter(hidden, {"iterations": 20})
    rng = np.random.default_rng(2)
    router.weight = mx.array(rng.standard_normal((8, 2 * hidden)), mx.float32) * 0.05
    router.bias = mx.array(rng.standard_normal(8), mx.float32)
    s = mx.array(rng.standard_normal((1, rows, 2 * hidden)), mx.float32)
    x, coef = router(s, mx.bfloat16)
    for i in range(rows):
        xi, ci = router(s[:, i:i + 1], mx.bfloat16)
        np.testing.assert_array_equal(np.array(coef[:, i:i + 1]), np.array(ci))
        np.testing.assert_array_equal(np.array(x[:, i:i + 1].astype(mx.float32)),
                                      np.array(xi.astype(mx.float32)))


@_GPU
@pytest.mark.parametrize("rows", [1, 7])
def test_the_one_dispatch_router_is_no_further_from_float64(monkeypatch, rows):
    # The one-dispatch router sums the norm and the linear in its own
    # order, so the gate is distance to a float64 reference.
    from gmlx.models.sensenova_loop import mhc

    hidden = 1024
    router = mhc.MHCRouter(hidden, {"iterations": 20})
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((8, 2 * hidden)) * 0.03).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    router.weight, router.bias = mx.array(w), mx.array(b)
    prev = rng.random((1, rows, 8)).astype(np.float32)
    s = rng.standard_normal((1, rows, 2 * hidden)).astype(np.float32)
    y = mx.array(rng.standard_normal((1, rows, hidden)), mx.bfloat16)

    out = {}
    for fused in ("0", "1"):
        monkeypatch.setenv("GMLX_SENSENOVA_MHC_FUSED", fused)
        out[fused] = [np.array(v.astype(mx.float32)) for v in mhc.route(
            router, mx.array(s), y, mx.array(prev), mx.bfloat16)]
    np.testing.assert_array_equal(out["0"][0], out["1"][0])  # merged streams
    merged = out["0"][0].astype(np.float64)
    ref = _reference_coef(
        merged / np.sqrt((merged ** 2).mean(-1, keepdims=True) + 1e-6)
        @ w.astype(np.float64).T + b, 20)
    err = {k: np.abs(v[2] - ref).max() for k, v in out.items()}
    assert err["1"] <= err["0"] * 1.5 + 1e-7, err


@_GPU
def test_the_one_dispatch_router_is_the_default(monkeypatch):
    from gmlx.models.sensenova_loop import mhc

    monkeypatch.delenv("GMLX_SENSENOVA_MHC_FUSED", raising=False)
    monkeypatch.delenv("GMLX_SENSENOVA_MHC_KERNEL", raising=False)
    assert mhc._use_fused()
    monkeypatch.setenv("GMLX_SENSENOVA_MHC_FUSED", "0")
    assert not mhc._use_fused()
