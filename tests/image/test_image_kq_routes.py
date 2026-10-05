"""The mlx-kquant routes of the image ops: the seam contract, when each
route takes a call and when the stock ops run, and each kernel against a
float64 reference beside the stock ops. The decisions run on recording
ops, so they hold on any mlx-kquant. The kernel tests skip by name on a
build without the op."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_kquant as kq
from mlx_kquant.nn import KQuantLinear

from gmlx.image import api, lora, ops
from gmlx.image import profile as prof
from gmlx.image.families.z_image.model import FeedForward, ZImageDiT
from gmlx.image.lora import install as lora_install
from gmlx.image.ops import attention, kq_routes
from gmlx.image.registry import get_family
from image_fixtures import TINY_Z_CONFIG

_ROOT = Path(__file__).resolve().parent.parent.parent
SEAMS = ("sdpa", "qk_norm_rope", "swiglu", "modulate", "gated_residual")
DTYPES = [mx.bfloat16, mx.float16]
EPS = 1e-5
# Relative L2 to the float64 reference, measured on an M5 Max at these
# shapes. qk_norm_rope: the kernel 1.7e-3 in bfloat16 and 2.1e-4 in
# float16, the stock seam 2.9e-3 and 3.6e-4. The GLU pair: the kernel
# 3.3e-3 and 4.2e-4, the stock ops 4.5e-3 and 5.8e-4. Each kernel gate is
# above the kernel's error and below the stock ops' own.
QK_TOL = {mx.bfloat16: 2.5e-3, mx.float16: 3.1e-4}
GLU_TOL = {mx.bfloat16: 4.4e-3, mx.float16: 5.6e-4}
# The GLU pair on either path.
PAIR_TOL = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# A whole tiny step, routes on against routes off: 2.2e-3 and 2.7e-4.
STEP_TOL = {mx.bfloat16: 7e-3, mx.float16: 9e-4}

HAS_QK = hasattr(kq, "qk_norm_rope")
HAS_GLU = hasattr(kq, "quantized_matmul_glu")
needs_qk = pytest.mark.skipif(
    not HAS_QK, reason="mlx-kquant has no qk_norm_rope")
needs_glu = pytest.mark.skipif(
    not HAS_GLU, reason="mlx-kquant has no quantized_matmul_glu")

# Head dim 64 and a 256-wide residual, so both kernels take the main
# layers at 1024 image rows.
KQ_CONFIG = dict(TINY_Z_CONFIG, dim=256, n_heads=4, n_kv_heads=4,
                 head_dim=64, ffn_dim=64, axes_dims=[16, 24, 24])
W1 = "layers.0.feed_forward.w1"
W3 = "layers.0.feed_forward.w3"


def _gpu() -> bool:
    return mx.default_device() == mx.gpu


def _fused_glu() -> bool:
    return _gpu() and kq.nax_available()


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


def _composed(x, w_gate, w_up, codec):
    s = mx.zeros((1,), mx.uint8)
    return ops.swiglu(
        kq.quantized_matmul(x, w_gate, s, codec, transpose=True),
        kq.quantized_matmul(x, w_up, s, codec, transpose=True))


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
    """Both routes in force on recording ops, with tensor ops reported
    present. A spy calls the build's op, or the stock ops on a build
    without it."""
    qk = _Spy(kq_routes._kq_op("qk_norm_rope") or attention.qk_norm_rope)
    glu = _Spy(kq_routes._kq_op("quantized_matmul_glu") or _composed)
    routes._OPS.update(qk_norm_rope=qk, quantized_matmul_glu=glu, nax=True)
    assert routes.install(routes.ROUTES) == frozenset(routes.ROUTES)
    return SimpleNamespace(qk=qk, glu=glu)


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


def _glu_ref(x, gate, up) -> np.ndarray:
    """float64 silu(x @ Wg^T) * (x @ Wu^T) on the dequantized weights."""
    wg, wu = (np.array(kq.dequantize(m.weight, m.scales, m.kquant_type,
                                     dtype=mx.float32), np.float64)
              for m in (gate, up))
    x = _f64(x)
    g = x @ wg.T
    return g / (1.0 + np.exp(-g)) * (x @ wu.T)


# The seam contract.

def test_the_seams_are_five_plain_functions_and_install_assigns(routes):
    assert ops.SEAMS == SEAMS
    for name in SEAMS:
        assert type(getattr(ops, name)) is types.FunctionType, name
    assert ops.sdpa is attention.sdpa
    fake = _Spy(attention.qk_norm_rope)
    routes._OPS.update(qk_norm_rope=fake, quantized_matmul_glu=fake)
    assert routes.install(("qk", "glu")) == {"qk", "glu"}
    assert routes.active() == {"qk", "glu"}
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope
    assert type(ops.qk_norm_rope) is types.FunctionType
    assert routes.install(("glu",)) == {"glu"}
    assert ops.qk_norm_rope is attention.qk_norm_rope
    assert routes.install(()) == frozenset()
    assert ops.SEAMS == SEAMS and fake.calls == 0


def test_the_import_time_install_follows_the_build():
    want = {r for r, op in (("qk", HAS_QK), ("glu", HAS_GLU)) if op}
    wanted = set(kq_routes._WANTED)
    assert kq_routes.active() == want & wanted
    routed = "qk" in kq_routes.active()
    assert (ops.qk_norm_rope is kq_routes.qk_norm_rope) == routed
    assert (ops.qk_norm_rope is attention.qk_norm_rope) == (not routed)


def test_an_absent_op_keeps_the_stock_ops(routes):
    routes._OPS.update(qk_norm_rope=None, quantized_matmul_glu=None, nax=True)
    assert routes.install(routes.ROUTES) == frozenset()
    assert ops.qk_norm_rope is attention.qk_norm_rope
    gate, up = _leaf("q4_k", 64, 256, 1), _leaf("q4_k", 64, 256, 2)
    assert ops.glu_pair(gate, up, _x((1024, 256), mx.bfloat16)) is None
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
    for value, want in ((None, ("qk", "glu")), ("0", ()), ("qk", ("qk",)),
                        ("GLU", ("glu",)), ("sometimes", ("qk", "glu"))):
        env = {k: v for k, v in os.environ.items()
               if k != "GMLX_IMAGE_KQ_ROUTES"}
        if value is not None:
            env["GMLX_IMAGE_KQ_ROUTES"] = value
        r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        wanted, active, stock = json.loads(r.stdout.splitlines()[-1])
        assert tuple(wanted) == want, value
        assert set(active) == {n for n, op in (("qk", HAS_QK),
                                               ("glu", HAS_GLU))
                               if op and n in want}, value
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
        routes.install(("glu",))
    with mx.stream(mx.cpu if case == "cpu_stream" else mx.default_device()):
        want = attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
        got = ops.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
        mx.eval(want, got)
    assert spies.qk.calls == 0
    for g, w in zip(got, want):
        assert mx.array_equal(g, w).item()


# When the GLU route takes a pair.

def _pair(codec="q4_k", inp=256, out=64):
    return _leaf(codec, out, inp, 1), _leaf(codec, out, inp, 2)


def _ff(gate, up, dtype=mx.bfloat16) -> FeedForward:
    out, inp = lora_install.linear_dims(gate)
    ff = FeedForward(SimpleNamespace(dim=inp, ffn_dim=out))
    ff.w1, ff.w3 = gate, up
    ff.w2.weight = _x((inp, out), dtype, 3) / 16
    return ff


def test_glu_takes_two_plain_kquant_leaves(spies):
    gate, up = _pair()
    x = _x((1, 1024, 256), mx.bfloat16)
    pair = ops.glu_pair(gate, up, x)
    if not _gpu():
        assert pair is None
        return
    assert pair is not None and pair[2] == "q4_k"
    assert pair[0] is gate["weight"] and pair[1] is up["weight"]
    ff = _ff(gate, up)
    y = ff(x)
    mx.eval(y)
    assert spies.glu.calls == 1 and y.shape == (1, 1024, 256)
    want = ff.w2(ops.swiglu(gate(x), up(x)))
    assert _rel(_f64(y), _f64(want)) <= 2 * PAIR_TOL[mx.bfloat16]


class _Scaled(KQuantLinear):
    def __call__(self, x, lora=None):
        return 2 * super().__call__(x, lora)


@pytest.mark.parametrize("case", [
    "route_off", "rows_1023", "codec_q6_k", "mixed_codecs", "bias",
    "float_leaf", "subclass", "float32_x", "one_dim_x", "k_off_the_tile",
    "shapes", "no_tensor_ops", "cpu_stream"])
def test_glu_falls_back_to_the_composition(spies, routes, monkeypatch, case):
    gate, up = _pair()
    x = _x((1, 1024, 256), mx.bfloat16)
    if case == "route_off":
        routes.install(("qk",))
    elif case == "rows_1023":
        x = x[:, :1023]
    elif case == "codec_q6_k":
        gate, up = _pair("q6_k")
    elif case == "mixed_codecs":
        up = _leaf("q5_k", 64, 256, 2)
    elif case == "bias":
        up = _leaf("q4_k", 64, 256, 2, bias=True)
    elif case == "float_leaf":
        up = nn.Linear(256, 64, bias=False)
        up.weight = up.weight.astype(mx.bfloat16)
    elif case == "subclass":
        gate.__class__ = _Scaled
    elif case == "float32_x":
        x = x.astype(mx.float32)
    elif case == "one_dim_x":
        x = x[0, 0]
    elif case == "k_off_the_tile":
        monkeypatch.setattr(kq_routes, "_GLU_CODECS", ("q4_0",))
        gate, up = _pair("q4_0", inp=160)
        x = _x((1, 1024, 160), mx.bfloat16)
    elif case == "shapes":
        up = _leaf("q4_k", 32, 256, 2)
    elif case == "no_tensor_ops":
        routes._OPS["nax"] = False
    if case == "cpu_stream":
        with mx.stream(mx.cpu):
            assert ops.glu_pair(gate, up, x) is None
    else:
        assert ops.glu_pair(gate, up, x) is None
    if case in ("shapes", "one_dim_x", "cpu_stream"):
        return
    ff = _ff(gate, up) if case != "k_off_the_tile" else None
    if ff is not None:
        y = ff(x)
        want = ff.w2(ops.swiglu(gate(x), up(x)))
        assert mx.array_equal(y, want).item()
    assert spies.glu.calls == 0


def test_glu_k_tile_fallback_has_a_taken_twin(spies, monkeypatch):
    """The K tile case above fails on its K alone: the same codec on the
    tile is taken."""
    monkeypatch.setattr(kq_routes, "_GLU_CODECS", ("q4_0",))
    gate, up = _pair("q4_0", inp=256)
    pair = ops.glu_pair(gate, up, _x((1, 1024, 256), mx.bfloat16))
    assert (pair is not None) == _gpu()


def test_glu_takes_rows_above_32767(spies):
    gate, up = _pair()
    # A batch of two at 2048x2048 flattens to 33792 rows.
    x = _x((2, 16896, 256), mx.bfloat16)
    if not _gpu():
        assert ops.glu_pair(gate, up, x) is None
        return
    y = ops.swiglu_fused(x, *ops.glu_pair(gate, up, x))
    mx.eval(y)
    assert spies.glu.calls == 1 and y.shape == (2, 16896, 64)
    err = _rel(_f64(y), _glu_ref(x, gate, up))
    assert err <= PAIR_TOL[mx.bfloat16], err


# A LoRA wrap.

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


def _pipe(dtype=mx.bfloat16) -> api.ImagePipeline:
    return api.ImagePipeline(dit=_kq_dit(dtype), family=get_family("z_image"),
                             variant="turbo")


def _table(pipe, path: str, strength: float = 1.0) -> str:
    rng = np.random.default_rng(5)
    down = (rng.standard_normal((4, 256)) * 0.2).astype(np.float32)
    up = (rng.standard_normal((64, 4)) * 0.2).astype(np.float32)
    key = lora.register_pair(pipe, down, up)
    return api.define_patch_table(pipe, [(path, None, [(key, strength)])])


@pytest.mark.parametrize("path", [W1, W3])
def test_a_live_lora_delta_takes_the_composition(spies, routes, path):
    pipe = _pipe()
    x = _x((1, 1024, 256), mx.bfloat16)
    ff = pipe.dit.layers[0].feed_forward
    other = pipe.dit.layers[1].feed_forward
    base = ff(x)
    mx.eval(base)
    taken = 1 if _gpu() else 0
    assert spies.glu.calls == taken
    tid = _table(pipe, path)
    assert isinstance(ff.w1, lora.ImageLoRALinear)
    # Wrapped with no table active: the base leaves, so the kernel.
    assert mx.array_equal(ff(x), base).item()
    assert spies.glu.calls == 2 * taken
    prev = lora_install.activate(pipe, tid)
    try:
        on = ff(x)
        mx.eval(on)
        assert spies.glu.calls == 2 * taken
        # A pair with no entry in the active table still takes the kernel.
        mx.eval(other(x))
        assert spies.glu.calls == 3 * taken
        routes.install(())
        off = ff(x)
        mx.eval(off)
    finally:
        lora_install.restore(pipe, prev)
    assert mx.array_equal(on, off).item()
    assert _rel(_f64(on), _f64(base)) > 1e-2
    assert spies.glu.calls == 3 * taken


def test_a_zero_strength_entry_keeps_the_kernel(spies):
    pipe = _pipe()
    x = _x((1, 1024, 256), mx.bfloat16)
    ff = pipe.dit.layers[0].feed_forward
    tid = _table(pipe, W1, 0.0)
    prev = lora_install.activate(pipe, tid)
    try:
        mx.eval(ff(x))
    finally:
        lora_install.restore(pipe, prev)
    assert spies.glu.calls == (1 if _gpu() else 0)


# The profiler.

def test_the_profiler_books_the_fused_pair_as_one_record(spies):
    if not _gpu():
        pytest.skip("the GLU route needs the GPU")
    m = _kq_dit()
    x = _x((1, 1024, 256), mx.bfloat16)
    ff = m.layers[0].feed_forward
    want = ff(x)
    mx.eval(want)
    with prof.StepProfiler(m) as p:
        assert type(ff.w1).__name__ == "TimedKQuantLinear"
        records, wall = p.step(lambda: ff(x))
    ran = [(r.kind, r.name) for r in records if r.kind != "glue"]
    assert ran == [("seam", "swiglu_fused"), ("module", "Linear")]
    fused = next(r for r in records if r.name == "swiglu_fused"
                 and r.kind == "seam")
    assert fused.in_shapes[0] == [1, 1024, 256]
    assert fused.out_shape == [1, 1024, 64]
    assert sum(r.seconds for r in records) == pytest.approx(wall, rel=1e-6,
                                                            abs=1e-6)
    assert type(ff.w1) is KQuantLinear
    assert ops.swiglu_fused is kq_routes.swiglu_fused


# The kernels against a float64 reference, beside the stock ops.

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


@needs_glu
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("lead", [(1024,), (1, 1056), (2, 16896)],
                         ids=["rows_1024", "rows_1056", "rows_33792"])
def test_glu_kernel_against_the_float64_reference(routes, dtype, lead):
    if not _fused_glu():
        pytest.skip("the GLU kernel runs on a GPU with tensor ops")
    assert routes.install(("glu",)) == {"glu"}
    gate, up = _pair()
    x = _x((*lead, 256), dtype)
    pair = ops.glu_pair(gate, up, x)
    assert pair is not None
    got = ops.swiglu_fused(x, *pair)
    stock = ops.swiglu(gate(x), up(x))
    mx.eval(got, stock)
    assert got.shape == (*lead, 64) and got.dtype == dtype
    ref = _glu_ref(x, gate, up)
    err, err_stock = _rel(_f64(got), ref), _rel(_f64(stock), ref)
    print(f"glu {lead} {dtype}: kernel {err:.3e} stock {err_stock:.3e}")
    assert np.isfinite(_f64(got)).all()
    assert err <= GLU_TOL[dtype], err
    assert err <= 1.1 * err_stock, (err, err_stock)


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
