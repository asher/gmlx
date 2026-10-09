"""The resid and conv routes onto mlx-kquant: which calls each takes and
when the stock ops run, the Qwen-Image 2.1 block on the residual op, the
VAE conv path, and each kernel against a float64 reference beside the
stock ops. The decisions run on recording ops with the device probes
stood in, so they hold on any mlx-kquant and any default device. The
kernel tests skip by name on a build without the op or without tensor
ops."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq

from gmlx.image import memfit, ops
from gmlx.image import profile as prof
from gmlx.image.families.qwen_image21 import model as qm
from gmlx.image.ops import kq_routes
from gmlx.image.vae import conv as vconv
from image_fixtures import kq_leaf

HAS_RESID = hasattr(kq, "quantized_matmul_residual")
HAS_CONV = hasattr(kq, "conv2d_3x3")
RESID_CODECS = ("q4_k", "q5_k", "q6_k", "q8_0")


def _gpu() -> bool:
    return mx.default_device() == mx.gpu


def _nax() -> bool:
    probe = getattr(kq, "nax_available", None)
    return _gpu() and probe is not None and bool(probe())


def _x(shape, dtype=mx.bfloat16, seed: int = 0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    x = mx.array((rng.standard_normal(shape) * scale).astype(np.float32))
    x = x.astype(dtype)
    mx.eval(x)
    return x


def _f64(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32), np.float64)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls, self.args = fn, 0, None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.args = (args, kwargs)
        return self.fn(*args, **kwargs)


def _resid_composed(x, w, codec, residual, *, gate=None):
    s = mx.zeros((1,), mx.uint8)
    y = kq.quantized_matmul(x, w, s, codec, transpose=True)
    return residual + (y if gate is None else gate * y)


def _conv_composed(x, w, bias=None):
    y = vconv.conv2d_no_winograd(x, w, 1, 1, 1, 1)
    return y if bias is None else y + bias


@pytest.fixture
def spies(monkeypatch):
    """The resid and conv routes in force on recording ops, with tensor
    ops and a GPU default device reported present. A spy calls the build's
    op on a GPU and the composition elsewhere."""
    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(kq_routes._OPS))
    gpu = _gpu()
    resid = _Spy(gpu and kq_routes._kq_op("quantized_matmul_residual")
                 or _resid_composed)
    conv = _Spy(gpu and kq_routes._kq_op("conv2d_3x3") or _conv_composed)
    kq_routes._OPS.update(quantized_matmul_residual=resid, conv2d_3x3=conv,
                          nax=True)
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert kq_routes.install(("resid", "conv")) == {"resid", "conv"}
    return SimpleNamespace(resid=resid, conv=conv)


def _resid_args(codec="q4_k", rows=1024, k=256, n=256, batch=1,
                dtype=mx.bfloat16, gate_shape=None):
    leaf = kq_leaf(codec, n, k, 1)
    x = _x((batch, rows, k), dtype, 2)
    residual = _x((batch, rows, n), dtype, 3)
    gate = _x(gate_shape or (batch, 1, n), dtype, 4, 0.5)
    return leaf, x, residual, gate


# The switch and the feature check.

def test_the_switch_names_the_resid_and_conv_routes():
    w = kq_routes.wanted
    assert {"resid", "conv"} <= set(kq_routes.ROUTES)
    assert w("resid") == ("resid",) and w("conv") == ("conv",)
    assert w("conv,resid") == ("resid", "conv")


def test_a_build_without_the_ops_keeps_the_stock_ops(monkeypatch):
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(kq_routes._OPS))
    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    kq_routes._OPS.update(quantized_matmul_residual=None, conv2d_3x3=None,
                          nax=True)
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert kq_routes.install(("resid", "conv")) == frozenset()
    leaf, x, residual, gate = _resid_args()
    assert ops.resid_leaf(leaf, x, residual, gate) is None
    w = _x((32, 3, 3, 32), seed=5)
    assert kq_routes.conv3x3(_x((1, 32, 32, 32)), w, None, 1, 1, 1,
                             1) is None


def test_the_import_time_install_takes_the_ops_the_build_has():
    want = set(kq_routes._WANTED)
    assert ("resid" in kq_routes.active()) == (HAS_RESID and "resid" in want)
    assert ("conv" in kq_routes.active()) == (HAS_CONV and "conv" in want)


# The resid route.

@pytest.mark.parametrize("codec", RESID_CODECS)
def test_resid_takes_a_plain_kquant_leaf_in_a_kernel_codec(spies, codec):
    leaf, x, residual, gate = _resid_args(codec)
    assert ops.resid_leaf(leaf, x, residual, gate) is leaf
    y = ops.resid_fused(x, leaf, residual, gate)
    mx.eval(y)
    args, kwargs = spies.resid.args
    assert args[0] is x and args[1] is leaf["weight"] and args[2] == codec
    assert args[3] is residual and kwargs == {"gate": gate}
    want = residual + gate * leaf(x)
    assert _rel(_f64(y), _f64(want)) < 1e-2


def test_resid_takes_a_gate_of_one_row(spies):
    leaf, x, residual, _ = _resid_args()
    assert ops.resid_leaf(leaf, x, residual, _x((256,), seed=9)) is leaf


@pytest.mark.parametrize("case", [
    "route_off", "float16", "q3_k", "bias", "residual_f32", "gate_f32",
    "gate_shape", "residual_shape", "float_leaf", "no_nax", "cpu"])
def test_resid_declines_and_the_stock_ops_run(spies, monkeypatch, case):
    codec = "q3_k" if case == "q3_k" else "q4_k"
    dtype = mx.float16 if case == "float16" else mx.bfloat16
    leaf, x, residual, gate = _resid_args(codec, dtype=dtype)
    if case == "route_off":
        kq_routes.install(("conv",))
    elif case == "bias":
        leaf = kq_leaf("q4_k", 256, 256, 1, bias=True)
    elif case == "residual_f32":
        residual = residual.astype(mx.float32)
    elif case == "gate_f32":
        gate = gate.astype(mx.float32)
    elif case == "gate_shape":
        gate = _x((1, 2, 256), seed=4)
    elif case == "residual_shape":
        residual = _x((1, 1024, 128), seed=3)
    elif case == "float_leaf":
        leaf = nn.Linear(256, 256, bias=False)
    elif case == "no_nax":
        kq_routes._OPS["nax"] = False
    elif case == "cpu":
        monkeypatch.setattr(kq_routes, "_gpu", lambda: False)
    assert ops.resid_leaf(leaf, x, residual, gate) is None
    if case not in ("residual_shape", "gate_shape"):
        y = qm.gated_projection(residual, gate, leaf, x)
        mx.eval(y)
        assert spies.resid.calls == 0
        assert mx.array_equal(y, ops.gated_residual(residual, gate,
                                                    leaf(x))).item()


def _block(dim=256, ffn=256, heads=2):
    cfg = SimpleNamespace(dim=dim, ffn_dim=ffn, fused_mlp=True, eps=1e-6,
                          n_heads=heads, head_dim=dim // heads)
    blk = qm.Block(cfg)
    blk.attn.to_out[0] = kq_leaf("q5_k", dim, dim, 7)
    blk.img_mlp.out = kq_leaf("q4_k", dim, ffn, 8)
    for i, (path, m) in enumerate(blk.named_modules()):
        if type(m) is nn.Linear:
            m.weight = _x(m.weight.shape, seed=20 + i, scale=1 / 16)
    return blk


def _mod(dim=256, batch=1):
    return qm.Modulation(
        _x((batch, 1, dim), seed=30, scale=0.1),
        mx.tanh(_x((batch, 1, dim), seed=31)),
        _x((batch, 1, dim), seed=32, scale=0.1),
        mx.tanh(_x((batch, 1, dim), seed=33)),
        _x((batch, 1, dim), seed=34, scale=0.1))


def test_the_qwen_block_runs_both_residual_adds_on_the_op(spies):
    blk = _block()
    x = _x((1, 1024, 256), seed=40)
    o = _x((1, 2, 1024, 128), seed=41)
    y = blk._update(x, _mod(), o)
    mx.eval(y)
    assert spies.resid.calls == 2
    with kq_routes.only(()):
        stock = blk._update(x, _mod(), o)
    assert spies.resid.calls == 2
    assert _rel(_f64(y), _f64(stock)) < 1e-2


def test_the_profiler_books_the_residual_op_as_one_record(spies):
    blk = _block()
    x = _x((1, 1024, 256), seed=40)
    o = _x((1, 2, 1024, 128), seed=41)
    p = prof.StepProfiler(blk)
    with p:
        mx.eval(blk._update(x, _mod(), o))
    names = [(r.name, r.path) for r in p.records if r.kind == "seam"]
    assert ("resid_fused", "attn.to_out.0") in names
    assert ("resid_fused", "img_mlp.out") in names
    assert not any(n == "gated_residual" for n, _ in names)
    assert ops.resid_fused is kq_routes.resid_fused


needs_resid = pytest.mark.skipif(
    not (HAS_RESID and _nax()),
    reason="mlx-kquant has no quantized_matmul_residual or no tensor ops")


@needs_resid
@pytest.mark.parametrize("codec", RESID_CODECS)
def test_resid_kernel_against_the_float64_reference(codec):
    leaf, x, residual, gate = _resid_args(codec, k=512, n=256)
    got = kq_routes.resid_fused(x, leaf, residual, gate)
    stock = residual + gate * leaf(x)
    product = leaf(x.astype(mx.float32))
    ref = _f64(residual) + _f64(gate) * _f64(product)
    err, err_stock = _rel(_f64(got), ref), _rel(_f64(stock), ref)
    print(f"[resid] {codec} kernel {err:.3e} stock {err_stock:.3e}")
    assert err <= err_stock * 1.05


@needs_resid
def test_a_guided_batch_of_two_stays_within_a_rounding_of_each_row():
    """The op rounds once over the rows of the whole batch: two rows of
    512 run the kernel together and the composition apart, so they can
    differ in the last bit, never by more than a rounding."""
    leaf, x, residual, gate = _resid_args(rows=512, batch=2)
    both = kq_routes.resid_fused(x, leaf, residual, gate)
    for i in range(2):
        one = kq_routes.resid_fused(x[i:i + 1], leaf, residual[i:i + 1],
                                    gate[i:i + 1])
        diff = np.abs(_f64(both[i:i + 1]) - _f64(one))
        # The roundings are of the gated product and of the sum, so the
        # bound is a few bfloat16 steps of the larger of the two terms.
        terms = (np.abs(_f64(residual[i:i + 1]))
                 + np.abs(_f64(gate[i:i + 1] * leaf(x[i:i + 1]))))
        assert bool((diff <= 2 ** -6 * terms + 1e-6).all())


# The conv route.

def test_the_default_conv_path_is_kq():
    assert vconv.CONV == "kq" and vconv.CONV_PATHS[0] == "kq"
    assert vconv.conv_path(None) == "kq"


@pytest.mark.parametrize("bias", [False, True])
def test_conv_takes_a_bf16_3x3_stride_1_conv(spies, bias):
    m = vconv.Conv2d(32, 64, 3, padding=1, bias=bias)
    m.weight = _x(m.weight.shape, seed=6, scale=0.1)
    if bias:
        m.bias = _x((64,), seed=7)
    m.path = "kq"
    x = _x((1, 32, 32, 32), seed=8)
    y = m(x)
    mx.eval(y)
    assert spies.conv.calls == 1 and y.shape == (1, 32, 32, 64)
    args, kwargs = spies.conv.args
    assert args[0] is x and args[1] is m.weight
    assert kwargs == {"bias": m.bias if bias else None}
    m.path = "exact"
    want = m(x)
    assert _rel(_f64(y), _f64(want)) < 1e-2


@pytest.mark.parametrize("case", [
    "route_off", "float32", "stride_2", "pad_0", "one_by_one", "c_24",
    "exact_path", "winograd_path", "cpu"])
def test_conv_declines_and_the_path_of_before_runs(spies, monkeypatch, case):
    c = 24 if case == "c_24" else 32
    stride, pad, k = 1, 1, 3
    if case == "stride_2":
        stride = 2
    elif case == "pad_0":
        pad = 0
    elif case == "one_by_one":
        k, pad = 1, 0
    m = vconv.Conv2d(c, 32, k, stride=stride, padding=pad)
    dtype = mx.float32 if case == "float32" else mx.bfloat16
    m.weight = _x(m.weight.shape, dtype, seed=6, scale=0.1)
    m.bias = _x((32,), dtype, seed=7)
    x = _x((1, 32, 32, c), dtype, seed=8)
    m.path = {"exact_path": "exact", "winograd_path": "winograd"}.get(
        case, "kq")
    if case == "route_off":
        kq_routes.install(("resid",))
    elif case == "cpu":
        monkeypatch.setattr(kq_routes, "_gpu", lambda: False)
    y = m(x)
    mx.eval(y)
    assert spies.conv.calls == 0
    conv = mx.conv2d if m.path == "winograd" else vconv.conv2d_no_winograd
    want = conv(x, m.weight, m.stride, m.padding, m.dilation, m.groups) \
        + m.bias
    assert mx.array_equal(y, want).item()


def test_load_sets_each_conv_path_and_memfit_prices_kq_as_exact():
    blk = nn.Sequential(vconv.Conv2d(16, 16, 3, padding=1),
                        vconv.Conv2d(16, 16, 1))
    for path in vconv.CONV_PATHS:
        assert vconv.set_conv_path(blk, path) == path
        assert all(m.path == path for m in blk.layers)
    for op in ("decode", "encode"):
        assert memfit.vae_peak(op, 1024, 1024, family="z_image", itemsize=2,
                               conv="kq") == memfit.vae_peak(
            op, 1024, 1024, family="z_image", itemsize=2, conv="exact")
        assert memfit.vae_peak(op, 1024, 1024, family="qwen_image21",
                               itemsize=2, conv="kq") == memfit.vae_peak(
            op, 1024, 1024, family="qwen_image21", itemsize=2, conv="exact")


needs_conv = pytest.mark.skipif(
    not (HAS_CONV and _nax()),
    reason="mlx-kquant has no conv2d_3x3 or no tensor ops")


@needs_conv
@pytest.mark.parametrize("c,o,hw", [(32, 32, 32), (144, 144, 32),
                                    (128, 3, 64), (16, 64, 32)])
def test_conv_kernel_against_the_float64_reference(c, o, hw):
    x = _x((1, hw, hw, c), seed=10)
    w = _x((o, 3, 3, c), seed=11, scale=0.05)
    b = _x((o,), seed=12, scale=0.1)
    got = kq_routes.conv3x3(x, w, b, 1, 1, 1, 1)
    assert got is not None or "conv" not in kq_routes.active()
    if got is None:
        pytest.skip("the conv route is not in force")
    exact = _conv_composed(x, w, b)
    ref = _f64(_conv_composed(x.astype(mx.float32), w.astype(mx.float32),
                              b.astype(mx.float32)))
    err, err_exact = _rel(_f64(got), ref), _rel(_f64(exact), ref)
    print(f"[conv] c{c} o{o} {hw}px kernel {err:.3e} exact {err_exact:.3e}")
    assert err <= err_exact * 1.05
