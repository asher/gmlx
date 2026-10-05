"""The mlx-kquant route of the image ``qk_norm_rope`` seam: the seam
contract, when the route takes a call and when the stock seam runs, and
the kernel against a float64 reference beside the stock seam. The
decisions run on a recording op, so they hold on any mlx-kquant. The
kernel tests skip by name on a build without the op."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

from gmlx.image import ops
from gmlx.image.families.z_image.model import ZImageDiT
from gmlx.image.ops import attention, kq_routes
from image_fixtures import TINY_Z_CONFIG

_ROOT = Path(__file__).resolve().parent.parent.parent
SEAMS = ("sdpa", "qk_norm_rope", "swiglu", "modulate", "gated_residual")
DTYPES = [mx.bfloat16, mx.float16]
EPS = 1e-5
# Relative L2 to the float64 reference, measured on an M5 Max at these
# shapes: the kernel 1.7e-3 in bfloat16 and 2.1e-4 in float16, the stock
# seam 2.9e-3 and 3.6e-4. The gate is above the kernel's error and below
# the stock seam's own.
QK_TOL = {mx.bfloat16: 2.5e-3, mx.float16: 3.1e-4}
# A whole tiny step, the route on against the route off.
STEP_TOL = {mx.bfloat16: 7e-3, mx.float16: 9e-4}

HAS_QK = hasattr(kq, "qk_norm_rope")
needs_qk = pytest.mark.skipif(
    not HAS_QK, reason="mlx-kquant has no qk_norm_rope")

# Head dim 64, which has a kernel, and a 256-wide residual.
KQ_CONFIG = dict(TINY_Z_CONFIG, dim=256, n_heads=4, n_kv_heads=4,
                 head_dim=64, ffn_dim=64, axes_dims=[16, 24, 24])


def _gpu() -> bool:
    return mx.default_device() == mx.gpu


def _f64(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32), np.float64)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


@pytest.fixture
def routes(monkeypatch):
    """The routes module, with the seam, the op table and the routes in
    force put back after the test."""
    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(kq_routes._OPS))
    return kq_routes


@pytest.fixture
def spies(routes):
    """The route in force on a recording op, which calls the build's op,
    or the stock seam on a build without it."""
    qk = _Spy(kq_routes._kq_op("qk_norm_rope") or attention.qk_norm_rope)
    routes._OPS.update(qk_norm_rope=qk)
    assert routes.install(routes.ROUTES) == frozenset(routes.ROUTES)
    return SimpleNamespace(qk=qk)


def _leaf(codec: str, out: int, inp: int, seed: int = 0, bias=False):
    rng = np.random.default_rng(seed)
    w = mx.array((rng.standard_normal((out, inp)) / 16).astype(np.float32))
    m = KQuantLinear(inp, out, bias, codec)
    m.weight, m.scales = kq.quantize(w, codec)
    mx.eval(m.weight, m.scales)
    return m


def _x(shape, dtype, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(dtype)
    mx.eval(x)
    return x


def _qk_inputs(b, n, h, d, dtype, seed=0):
    """q and k as the attention block passes them, slices of one fused qkv
    output, with norm weights and RoPE tables."""
    rng = np.random.default_rng(seed)
    qkv = _x((b, n, 3 * h * d), dtype, seed).reshape(b, n, 3, h, d)
    w = (1.0 + 0.2 * rng.standard_normal((2, d))).astype(np.float32)
    ang = rng.uniform(-60.0, 60.0, (n, d // 2)).astype(np.float32)
    return (qkv[:, :, 0], qkv[:, :, 1], mx.array(w[0]).astype(dtype),
            mx.array(w[1]).astype(dtype), mx.array(np.cos(ang)),
            mx.array(np.sin(ang)))


def _qk_ref(x, w, cos, sin) -> np.ndarray:
    """float64 RMSNorm and rotation of interleaved pairs, (B, H, L, D)."""
    x, w = _f64(x), _f64(w)
    c = np.array(cos, np.float64)[None, :, None, :]
    s = np.array(sin, np.float64)[None, :, None, :]
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w
    o = np.empty_like(y)
    o[..., 0::2] = y[..., 0::2] * c - y[..., 1::2] * s
    o[..., 1::2] = y[..., 0::2] * s + y[..., 1::2] * c
    return o.transpose(0, 2, 1, 3)


# The seam contract.

def test_the_seams_are_five_plain_functions_and_install_assigns(routes):
    assert ops.SEAMS == SEAMS
    for name in SEAMS:
        assert type(getattr(ops, name)) is types.FunctionType, name
    assert ops.sdpa is attention.sdpa
    fake = _Spy(attention.qk_norm_rope)
    routes._OPS.update(qk_norm_rope=fake)
    assert routes.install(("qk",)) == {"qk"}
    assert routes.active() == {"qk"}
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope
    assert type(ops.qk_norm_rope) is types.FunctionType
    assert routes.install(()) == frozenset()
    assert ops.qk_norm_rope is attention.qk_norm_rope
    assert ops.SEAMS == SEAMS and fake.calls == 0


def test_the_import_time_install_follows_the_build():
    want = {"qk"} if HAS_QK else set()
    assert kq_routes.active() == want & set(kq_routes._WANTED)
    routed = "qk" in kq_routes.active()
    assert (ops.qk_norm_rope is kq_routes.qk_norm_rope) == routed
    assert (ops.qk_norm_rope is attention.qk_norm_rope) == (not routed)


def test_an_absent_op_keeps_the_stock_seam(routes):
    routes._OPS.update(qk_norm_rope=None)
    assert routes.install(routes.ROUTES) == frozenset()
    assert ops.qk_norm_rope is attention.qk_norm_rope
    # The routed function itself runs the stock seam on such a build.
    q, k, qw, kw, cos, sin = _qk_inputs(1, 8, 2, 64, mx.bfloat16)
    got = kq_routes.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    want = attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    for g, w in zip(got, want):
        assert mx.array_equal(g, w).item()


def test_the_switch_is_read_once_at_import():
    code = (
        "import json\n"
        "from gmlx.image import ops\n"
        "from gmlx.image.ops import attention, kq_routes as r\n"
        "print(json.dumps([list(r._WANTED), sorted(r.active()),\n"
        "    ops.qk_norm_rope is attention.qk_norm_rope]))\n")
    for value, want in ((None, ("qk",)), ("0", ()), ("sometimes", ("qk",))):
        env = {k: v for k, v in os.environ.items()
               if k != "GMLX_IMAGE_KQ_ROUTES"}
        if value is not None:
            env["GMLX_IMAGE_KQ_ROUTES"] = value
        r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        wanted, active, stock = json.loads(r.stdout.splitlines()[-1])
        assert tuple(wanted) == want, value
        assert active == (["qk"] if HAS_QK and want else []), value
        assert stock == ("qk" not in active), value


# When the qk route takes a call.

def test_qk_takes_half_precision_at_a_kernel_head_dim(spies):
    args = _qk_inputs(2, 8, 2, 64, mx.bfloat16)
    q, k = ops.qk_norm_rope(*args, EPS)
    mx.eval(q, k)
    assert spies.qk.calls == (1 if _gpu() else 0)
    assert q.shape == (2, 2, 8, 64) and q.dtype == mx.bfloat16


@pytest.mark.parametrize("case", ["float32", "head_dim_32", "head_dim_80",
                                  "k_dtype", "k_shape", "cpu_stream",
                                  "route_off"])
def test_qk_falls_back_to_the_stock_seam(spies, routes, case):
    d = {"head_dim_32": 32, "head_dim_80": 80}.get(case, 64)
    dtype = mx.float32 if case == "float32" else mx.bfloat16
    q, k, qw, kw, cos, sin = _qk_inputs(1, 8, 2, d, dtype)
    if case == "k_dtype":
        k = k.astype(mx.float16)
    if case == "k_shape":
        k = k[:, :, :1]
    if case == "route_off":
        routes.install(())
    with mx.stream(mx.cpu if case == "cpu_stream" else mx.default_device()):
        want = attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
        got = ops.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
        mx.eval(want, got)
    assert spies.qk.calls == 0
    for g, w in zip(got, want):
        assert mx.array_equal(g, w).item()


# The kernel against a float64 reference, beside the stock seam.

def _kq_dit(dtype=mx.bfloat16) -> ZImageDiT:
    """A small Z-Image DiT in ``dtype`` whose w1 and w3 are q4_k leaves."""
    from mlx.utils import tree_flatten

    m = ZImageDiT(KQ_CONFIG)
    rng = np.random.default_rng(0)
    params = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * 0.1
        if k.endswith("norm1.weight"):
            a += 1.0
        params.append((k, mx.array(a.astype(np.float32)).astype(dtype)))
    m.load_weights(params)
    for i, blk in enumerate((*m.noise_refiner, *m.context_refiner,
                             *m.layers)):
        ff = blk.feed_forward
        ff.w1 = _leaf("q4_k", 64, 256, 10 + i)
        ff.w3 = _leaf("q4_k", 64, 256, 20 + i)
    m.act_dtype = dtype
    m.eval()
    return m


@needs_qk
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("shape", [(2, 40, 2, 64), (1, 33, 3, 128),
                                   (2, 16896, 1, 64)],
                         ids=["hd64", "hd128", "rows_33792"])
def test_qk_kernel_against_the_float64_reference(routes, dtype, shape):
    if not _gpu():
        pytest.skip("the qk_norm_rope kernel runs on the GPU")
    assert routes.install(("qk",)) == {"qk"}
    q, k, qw, kw, cos, sin = _qk_inputs(*shape, dtype)
    got = ops.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    stock = attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    mx.eval(got, stock)
    b, n, h, d = shape
    for g, s, x, w in zip(got, stock, (q, k), (qw, kw)):
        assert g.shape == (b, h, n, d) and g.dtype == dtype
        ref = _qk_ref(x, w, cos, sin)
        err, err_stock = _rel(_f64(g), ref), _rel(_f64(s), ref)
        print(f"qk {shape} {dtype}: kernel {err:.3e} stock {err_stock:.3e}")
        assert np.isfinite(_f64(g)).all()
        assert err <= QK_TOL[dtype], err
        assert err <= 1.05 * err_stock, (err, err_stock)


@needs_qk
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
def test_a_routed_step_stays_near_the_stock_step(routes, dtype):
    if not _gpu():
        pytest.skip("the kernels run on the GPU")
    m = _kq_dit(dtype)
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 64, 64)).astype(np.float32))
    cap = mx.array(rng.standard_normal((1, 5, 16)).astype(np.float32))
    routes.install(())
    want = _f64(m(lat, 0.6, cap))
    on = routes.install(routes.ROUTES)
    assert "qk" in on
    got = _f64(m(lat, 0.6, cap))
    again = _f64(m(lat, 0.6, cap))
    assert np.isfinite(got).all() and np.array_equal(got, again)
    err = _rel(got, want)
    print(f"tiny step {dtype} routes {sorted(on)}: {err:.3e}")
    assert 0 < err <= STEP_TOL[dtype], err
