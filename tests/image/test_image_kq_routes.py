"""The mlx-kquant routes of the image ops: the seam contract, when each
route takes a call and when the stock ops run, and each kernel against a
float64 reference beside the stock ops. The decisions run on recording
ops with the device probe stood in, so they hold on any mlx-kquant and on
any default device. The kernel tests skip by name on a build without the
op."""
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
from gmlx.image.families.z_image.model import FeedForward
from gmlx.image.lora import install as lora_install
from gmlx.image.ops import attention, kq_routes
from gmlx.image.registry import get_family
from image_fixtures import (glu_rounded_once, kq_leaf, kq_z_model,
                            lora_rounded_once, qk_rounded_once)

_ROOT = Path(__file__).resolve().parent.parent.parent
SEAMS = ("sdpa", "qk_norm_rope", "swiglu", "modulate", "gated_residual")
DTYPES = [mx.bfloat16, mx.float16]
EPS = 1e-5
# Relative L2 to the float64 reference. Each kernel gate is above the
# kernel's error and below the stock ops' own at these shapes.
QK_TOL = {mx.bfloat16: 2.5e-3, mx.float16: 3.1e-4}
GLU_TOL = {mx.bfloat16: 4.4e-3, mx.float16: 5.6e-4}
# The GLU pair on either path.
PAIR_TOL = {mx.bfloat16: 6e-3, mx.float16: 1e-3}
# A whole tiny step, routes on against routes off: 2.2e-3 and 2.7e-4.
STEP_TOL = {mx.bfloat16: 7e-3, mx.float16: 9e-4}

HAS_QK = hasattr(kq, "qk_norm_rope")
HAS_GLU = hasattr(kq, "quantized_matmul_glu")
HAS_LORA = hasattr(kq, "quantized_matmul_lora")
BUILT = (("qk", HAS_QK), ("glu", HAS_GLU), ("lora", HAS_LORA),
         ("resid", hasattr(kq, "quantized_matmul_residual")),
         ("conv", hasattr(kq, "conv2d_3x3")))
needs_qk = pytest.mark.skipif(
    not HAS_QK, reason="mlx-kquant has no qk_norm_rope")
needs_glu = pytest.mark.skipif(
    not HAS_GLU, reason="mlx-kquant has no quantized_matmul_glu")

# The codec and x dtype rows that the GLU route takes.
GLU_ROWS = [(c, d) for c in ("q4_k", "q5_k", "q6_k") for d in DTYPES]

W1 = "layers.0.feed_forward.w1"
W3 = "layers.0.feed_forward.w3"
DEVICE_PROBE = kq_routes._gpu


def _gpu() -> bool:
    return mx.default_device() == mx.gpu


def _fused_glu() -> bool:
    return _gpu() and kq.nax_available()


def _f64(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32), np.float64)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class _Spy:
    """Counts the calls of ``fn`` and keeps the arguments of the last."""

    def __init__(self, fn):
        self.fn, self.calls, self.args = fn, 0, None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.args = (args, kwargs)
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
def spies(routes, monkeypatch):
    """The qk and glu routes in force on recording ops, with tensor ops
    and a GPU default device reported present, so a decision reads the
    same on any device. A spy calls the build's op on a GPU, and the stock
    ops on the CPU or on a build without the op. The lora route has its
    own tests, in test_image_lora_route.py."""
    gpu = _gpu()
    qk = _Spy(gpu and kq_routes._kq_op("qk_norm_rope")
              or attention.qk_norm_rope)
    glu = _Spy(gpu and kq_routes._kq_op("quantized_matmul_glu") or _composed)
    routes._OPS.update(qk_norm_rope=qk, quantized_matmul_glu=glu, nax=True)
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert routes.install(("qk", "glu")) == {"qk", "glu"}
    return SimpleNamespace(qk=qk, glu=glu)


_leaf = kq_leaf


def _x(shape, dtype, seed: int = 0, scale: float = 1.0) -> mx.array:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    x = mx.array(x).astype(dtype)
    mx.eval(x)
    return x


def _qk_inputs(b, n, h, d, dtype, seed=0, scale: float = 1.0):
    """q and k as the attention block passes them, slices of one fused qkv
    output of standard deviation ``scale``, with norm weights and RoPE
    tables."""
    rng = np.random.default_rng(seed)
    qkv = _x((b, n, 3 * h * d), dtype, seed, scale).reshape(b, n, 3, h, d)
    w = (1.0 + 0.2 * rng.standard_normal((2, d))).astype(np.float32)
    ang = rng.uniform(-60.0, 60.0, (n, d // 2)).astype(np.float32)
    return (qkv[:, :, 0], qkv[:, :, 1], mx.array(w[0]).astype(dtype),
            mx.array(w[1]).astype(dtype), mx.array(np.cos(ang)),
            mx.array(np.sin(ang)))


def _qk_ref(x, w, cos, sin, eps: float = EPS) -> np.ndarray:
    """float64 RMSNorm and rotation of interleaved pairs, (B, H, L, D)."""
    x, w = _f64(x), _f64(w)
    c = np.array(cos, np.float64)[None, :, None, :]
    s = np.array(sin, np.float64)[None, :, None, :]
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
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


def test_only_puts_routes_in_force_for_a_block(spies):
    assert kq_routes.active() == {"qk", "glu"}
    with kq_routes.only(()) as on:
        assert on == frozenset() == kq_routes.active()
        assert ops.qk_norm_rope is attention.qk_norm_rope
        with kq_routes.only(("glu",)) as inner:
            assert inner == {"glu"} == kq_routes.active()
        assert kq_routes.active() == frozenset()
    assert kq_routes.active() == {"qk", "glu"}
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope
    with pytest.raises(KeyError), kq_routes.only(("qk",)):
        assert kq_routes.active() == {"qk"}
        raise KeyError("in the block")
    assert kq_routes.active() == {"qk", "glu"}
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope


@pytest.mark.parametrize("bad,error,names", [
    ("qk", TypeError, "the str 'qk'"), ("glu", TypeError, "the str 'glu'"),
    (None, TypeError, "routes is None"),
    (("glu", "bogus"), ValueError, "unknown route 'bogus'"),
    (("q", "k"), ValueError, "unknown route 'q'")],
    ids=["str_qk", "str_glu", "none", "unknown", "letters"])
def test_only_refuses_a_str_none_and_an_unknown_name(spies, bad, error,
                                                     names):
    """tuple("qk") is two unknown names, which would put no route in
    force. Each is refused by name before any change."""
    with kq_routes.only(("glu",)):
        with pytest.raises(error, match=names):
            with kq_routes.only(bad):
                pytest.fail("the block was entered")
        assert kq_routes.active() == {"glu"}
        assert ops.qk_norm_rope is attention.qk_norm_rope
    assert kq_routes.active() == {"qk", "glu"}
    # install takes None for the set of the switch, and refuses the rest.
    if bad is not None:
        with pytest.raises(error, match=names):
            kq_routes.install(bad)
        assert kq_routes.active() == {"qk", "glu"}
        assert ops.qk_norm_rope is kq_routes.qk_norm_rope


@pytest.mark.parametrize("raises", [False, True], ids=["clean", "error"])
def test_only_puts_the_earlier_routes_back_before_it_raises(spies, raises):
    """A profiler installed inside the block still holds the seam at the
    block's end. The exit puts the earlier routes back, drops the
    replacement and then raises, and the profiler's removal leaves the
    seam alone."""
    p = prof.StepProfiler(_kq_dit())
    with pytest.raises(RuntimeError, match="replaced inside the block") as e:
        with kq_routes.only(()):
            p.install()
            assert ops.qk_norm_rope is not attention.qk_norm_rope
            if raises:
                raise KeyError("in the block")
    assert isinstance(e.value.__context__, KeyError) == raises
    for _ in range(2):
        assert kq_routes.active() == {"qk", "glu"}
        assert ops.qk_norm_rope is kq_routes.qk_norm_rope
        p.remove()
    assert ops.glu_pair is kq_routes.glu_pair
    assert ops.swiglu_fused is kq_routes.swiglu_fused
    assert ops.sdpa is attention.sdpa


def test_install_refuses_a_replaced_seam(spies, monkeypatch):
    def other(*args):
        return attention.qk_norm_rope(*args)

    monkeypatch.setattr(ops, "qk_norm_rope", other)
    with pytest.raises(RuntimeError, match="qk_norm_rope is replaced"):
        kq_routes.install(())
    assert kq_routes.active() == {"qk", "glu"}
    assert ops.qk_norm_rope is other


def test_the_import_time_install_follows_the_build():
    want = {r for r, op in BUILT if op}
    wanted = set(kq_routes._WANTED)
    assert kq_routes.active() == want & wanted
    routed = "qk" in kq_routes.active()
    assert (ops.qk_norm_rope is kq_routes.qk_norm_rope) == routed
    assert (ops.qk_norm_rope is attention.qk_norm_rope) == (not routed)


def test_the_switch_takes_a_comma_list_of_routes():
    w = kq_routes.wanted
    assert w(None) == w("1") == w(" On ") == kq_routes.ROUTES
    assert w("0") == w("off") == w("") == ()
    assert w("lora") == ("lora",)
    assert w("glu,qk") == w(" qk , glu ") == ("qk", "glu")
    assert w("qk,glu,lora,resid,conv,qk") == kq_routes.ROUTES
    assert w("conv,resid") == ("resid", "conv")
    for bad in ("glu,nope", "qk;glu", "2"):
        with pytest.warns(UserWarning, match="comma list"):
            assert w(bad) == kq_routes.ROUTES


def test_an_absent_op_keeps_the_stock_ops(routes):
    routes._OPS.update(qk_norm_rope=None, quantized_matmul_glu=None,
                       quantized_matmul_lora=None,
                       quantized_matmul_residual=None, conv2d_3x3=None,
                       nax=True)
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
    every = kq_routes.ROUTES
    for value, want, warns in (
            (None, every, False), ("1", every, False), ("on", every, False),
            ("0", (), False), (" 0 ", (), False), ("off", (), False),
            ("False", (), False), ("no", (), False), ("", (), False),
            ("qk", ("qk",), False), ("GLU", ("glu",), False),
            ("lora", ("lora",), False), ("resid", ("resid",), False),
            ("conv,qk", ("qk", "conv"), False),
            ("sometimes", every, True), ("2", every, True)):
        env = {k: v for k, v in os.environ.items()
               if k != "GMLX_IMAGE_KQ_ROUTES"}
        if value is not None:
            env["GMLX_IMAGE_KQ_ROUTES"] = value
        r = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        # An unknown value keeps every route and says so, once.
        assert r.stderr.count("GMLX_IMAGE_KQ_ROUTES=") == warns, r.stderr
        wanted, active, stock = json.loads(r.stdout.splitlines()[-1])
        assert tuple(wanted) == want, value
        assert set(active) == {n for n, op in BUILT
                               if op and n in want}, value
        assert stock == ("qk" not in active), value


# When the qk route takes a call.

@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("d", [64, 96, 128, 256])
def test_qk_takes_half_precision_at_a_kernel_head_dim(spies, dtype, d):
    args = _qk_inputs(2, 8, 2, d, dtype)
    q, k = ops.qk_norm_rope(*args, EPS)
    want = attention.qk_norm_rope(*args, EPS)
    mx.eval(q, k, want)
    assert spies.qk.calls == 1
    # The kernel takes the seam's arguments, in the seam's order.
    got, kwargs = spies.qk.args
    assert len(got) == 7 and not kwargs and got[6] == EPS
    assert all(a is b for a, b in zip(got, args))
    assert q.shape == (2, 2, 8, d) and q.dtype == dtype
    for g, w in zip((q, k), want):
        assert _rel(_f64(g), _f64(w)) <= 2 * QK_TOL[dtype]


def test_a_cpu_default_device_keeps_the_stock_ops(spies, monkeypatch):
    """The device probe itself, which every other decision test stands
    in: on a CPU default device neither route hands a call to its
    kernel."""
    monkeypatch.setattr(kq_routes, "_gpu", DEVICE_PROBE)
    args = _qk_inputs(1, 8, 2, 64, mx.bfloat16)
    gate, up = _pair()
    x = _x((1, 1024, 256), mx.bfloat16)
    with mx.stream(mx.cpu):
        assert not kq_routes._gpu()
        want = attention.qk_norm_rope(*args, EPS)
        got = ops.qk_norm_rope(*args, EPS)
        mx.eval(want, got)
        assert ops.glu_pair(gate, up, x) is None
    assert spies.qk.calls == 0 and spies.glu.calls == 0
    for g, w in zip(got, want):
        assert mx.array_equal(g, w).item()
    # The probe reads the default device at each call.
    assert kq_routes._gpu() == _gpu()
    mx.eval(ops.qk_norm_rope(*args, EPS))
    assert spies.qk.calls == (1 if _gpu() else 0)
    assert (ops.glu_pair(gate, up, x) is not None) == _gpu()


@pytest.mark.parametrize("case", ["float32", "head_dim_32", "head_dim_80",
                                  "k_dtype", "k_shape", "cos_rows", "sin_rows",
                                  "route_off"])
def test_qk_falls_back_to_the_stock_seam(spies, routes, case):
    d = {"head_dim_32": 32, "head_dim_80": 80}.get(case, 64)
    dtype = mx.float32 if case == "float32" else mx.bfloat16
    q, k, qw, kw, cos, sin = _qk_inputs(1, 8, 2, d, dtype)
    if case == "k_dtype":
        k = k.astype(mx.float16)
    if case == "k_shape":
        k = k[:, :, :1]
    # The stock seam broadcasts a table of one row.
    if case == "cos_rows":
        cos = cos[:1]
    if case == "sin_rows":
        sin = sin[:1]
    if case == "route_off":
        routes.install(("glu",))
    want = attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    got = ops.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    mx.eval(want, got)
    assert spies.qk.calls == 0
    for g, w in zip(got, want):
        assert mx.array_equal(g, w).item()


@pytest.mark.parametrize("rank", [3, 5])
def test_qk_of_another_rank_raises_as_the_stock_seam(spies, rank):
    # A row count that also fits the tables when the dims are read as
    # (B, L, H, D), so the rank alone keeps the call off the kernel.
    args = _qk_inputs(1, 2 if rank == 3 else 1, 2, 64, mx.bfloat16)
    q, k = (a[0] if rank == 3 else a[None] for a in args[:2])
    with pytest.raises(ValueError) as stock:
        attention.qk_norm_rope(q, k, *args[2:], EPS)
    with pytest.raises(ValueError) as routed:
        ops.qk_norm_rope(q, k, *args[2:], EPS)
    assert str(routed.value) == str(stock.value)
    assert spies.qk.calls == 0


@pytest.mark.parametrize("which", ["q", "k"])
def test_qk_with_a_norm_weight_of_another_shape_raises_as_the_stock_seam(
        spies, which):
    q, k, qw, kw, cos, sin = _qk_inputs(1, 8, 2, 64, mx.bfloat16)
    if which == "q":
        qw = qw[None]
    else:
        kw = kw[None]
    with pytest.raises(ValueError) as stock:
        attention.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    with pytest.raises(ValueError) as routed:
        ops.qk_norm_rope(q, k, qw, kw, cos, sin, EPS)
    assert str(routed.value) == str(stock.value)
    assert spies.qk.calls == 0


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
    assert pair is not None and pair[2] == "q4_k"
    assert pair[0] is gate["weight"] and pair[1] is up["weight"]
    ff = _ff(gate, up)
    y = ff(x)
    mx.eval(y)
    assert spies.glu.calls == 1 and y.shape == (1, 1024, 256)
    # The kernel takes x, the gate weight, the up weight and the codec.
    got, kwargs = spies.glu.args
    assert len(got) == 4 and not kwargs and got[3] == "q4_k"
    assert got[0] is x and got[1] is gate["weight"]
    assert got[2] is up["weight"]
    want = ff.w2(ops.swiglu(gate(x), up(x)))
    assert _rel(_f64(y), _f64(want)) <= 2 * PAIR_TOL[mx.bfloat16]


class _Scaled(KQuantLinear):
    def __call__(self, x, lora=None):
        return 2 * super().__call__(x, lora)


@pytest.mark.parametrize("codec,dtype", GLU_ROWS + [
    ("q8_0", mx.bfloat16), ("q8_0", mx.float16), ("q4_0", mx.bfloat16),
    ("q3_k", mx.bfloat16)],
    ids=lambda v: v if isinstance(v, str) else str(v).split(".")[-1])
def test_glu_takes_the_codec_rows_of_the_table(spies, codec, dtype):
    """q8_0 takes the composition on either half dtype, on an inner dim
    that mlx-kquant's own q8_0 kernel takes."""
    taken = (codec, dtype) in GLU_ROWS
    gate, up = _pair(codec)
    x = _x((1, 1024, 256), dtype)
    pair = ops.glu_pair(gate, up, x)
    assert (pair is not None) == taken
    ff = _ff(gate, up, dtype)
    y = ff(x)
    want = ff.w2(ops.swiglu(gate(x), up(x)))
    mx.eval(y, want)
    assert spies.glu.calls == (1 if taken else 0)
    if taken:
        assert pair[2] == codec
        assert _rel(_f64(y), _f64(want)) <= 2 * PAIR_TOL[dtype]
    else:
        assert mx.array_equal(y, want).item()


@pytest.mark.parametrize("tile,taken", [(256, True), (512, False)])
def test_glu_takes_an_inner_dim_on_the_k_tile(spies, monkeypatch, tile,
                                              taken):
    monkeypatch.setitem(kq_routes._GLU_K_TILE, ("q4_k", mx.bfloat16), tile)
    gate, up = _pair()
    pair = ops.glu_pair(gate, up, _x((1, 1024, 256), mx.bfloat16))
    assert (pair is not None) == taken


@pytest.mark.parametrize("codec", ["q4_k", "q5_k", "q6_k"])
@pytest.mark.parametrize("k,taken", [(128, True), (192, False), (256, True),
                                     (320, False), (384, True)])
def test_glu_takes_an_inner_dim_that_is_a_multiple_of_128(spies, codec, k,
                                                          taken):
    """The K tile of each codec row is 128. An inner dim that is a
    multiple of 64 and not of 128 stays on the stock ops. The predicate
    reads the inner dim of x, so no product runs here."""
    gate, up = _pair(codec)
    pair = ops.glu_pair(gate, up, _x((1, 1024, k), mx.bfloat16))
    assert (pair is not None) == taken
    assert spies.glu.calls == 0


@pytest.mark.parametrize("case", [
    "route_off", "rows_1023", "mixed_codecs", "mixed_codecs_of_one_shape",
    "bias_up", "bias_gate", "float_leaf", "subclass", "float32_x",
    "one_dim_x", "zero_dim_x", "zero_k", "no_outputs", "shapes",
    "no_tensor_ops"])
def test_glu_falls_back_to_the_composition(spies, routes, case):
    gate, up = _pair()
    x = _x((1, 1024, 256), mx.bfloat16)
    if case == "route_off":
        routes.install(("qk",))
    elif case == "rows_1023":
        x = x[:, :1023]
    elif case == "mixed_codecs":
        up = _leaf("q5_k", 64, 256, 2)
    elif case == "mixed_codecs_of_one_shape":
        # q4_k and q4_0 both pack 256 weights into 144 bytes.
        up = _leaf("q4_0", 64, 256, 2)
        assert up.weight.shape == gate.weight.shape
    elif case == "bias_up":
        up = _leaf("q4_k", 64, 256, 2, bias=True)
    elif case == "bias_gate":
        gate = _leaf("q4_k", 64, 256, 1, bias=True)
    elif case == "float_leaf":
        up = nn.Linear(256, 64, bias=False)
        up.weight = up.weight.astype(mx.bfloat16)
    elif case == "subclass":
        gate.__class__ = _Scaled
    elif case == "float32_x":
        x = x.astype(mx.float32)
    elif case == "one_dim_x":
        x = x[0, 0]
    elif case == "zero_dim_x":
        x = x[0, 0, 0]
    elif case == "zero_k":
        x = x[..., :0]
    elif case == "no_outputs":
        gate.weight, up.weight = gate.weight[:0], up.weight[:0]
    elif case == "shapes":
        up = _leaf("q4_k", 32, 256, 2)
    elif case == "no_tensor_ops":
        routes._OPS["nax"] = False
    assert ops.glu_pair(gate, up, x) is None
    assert spies.glu.calls == 0
    # The cases left out have no feed-forward output on either path.
    if case in ("shapes", "one_dim_x", "zero_dim_x", "zero_k", "no_outputs"):
        return
    ff = _ff(gate, up)
    y = ff(x)
    want = ff.w2(ops.swiglu(gate(x), up(x)))
    assert mx.array_equal(y, want).item()
    assert spies.glu.calls == 0


def test_glu_takes_the_two_halves_of_one_fused_leaf(spies):
    """A gate and up pair stored as one matrix, as FLUX.3 klein's mlp_in:
    the first half of the rows is the gate, the second the up."""
    fused = _leaf("q4_k", 128, 256, 1)
    x = _x((1, 1024, 256), mx.bfloat16)
    pair = ops.glu_halves(fused, x)
    assert pair is not None and pair[2] == "q4_k"
    w = fused["weight"]
    assert mx.array_equal(pair[0], w[:64]).item()
    assert mx.array_equal(pair[1], w[64:]).item()
    y = ops.swiglu_fused(x, *pair)
    mx.eval(y)
    assert spies.glu.calls == 1 and y.shape == (1, 1024, 64)
    a, b = mx.split(fused(x), 2, axis=-1)
    want = ops.swiglu(a, b)
    assert _rel(_f64(y), _f64(want)) <= 2 * PAIR_TOL[mx.bfloat16]


@pytest.mark.parametrize("case", [
    "route_off", "rows_1023", "odd_rows", "bias", "float_leaf", "float32_x",
    "one_dim_x", "zero_k", "k_off_tile", "codec", "no_outputs",
    "no_tensor_ops", "cpu"])
def test_glu_halves_fall_back_to_the_composition(spies, routes, monkeypatch,
                                                 case):
    fused = _leaf("q4_k", 128, 256, 1)
    x = _x((1, 1024, 256), mx.bfloat16)
    if case == "route_off":
        routes.install(("qk",))
    elif case == "rows_1023":
        x = x[:, :1023]
    elif case == "odd_rows":
        fused.weight = fused.weight[:127]
    elif case == "bias":
        fused = _leaf("q4_k", 128, 256, 1, bias=True)
    elif case == "float_leaf":
        fused = nn.Linear(256, 128, bias=False)
    elif case == "float32_x":
        x = x.astype(mx.float32)
    elif case == "one_dim_x":
        x = x[0, 0]
    elif case == "zero_k":
        x = x[..., :0]
    elif case == "k_off_tile":
        # The predicate reads the inner dim of x: 192 is off the tile.
        x = _x((1, 1024, 192), mx.bfloat16)
    elif case == "codec":
        fused = _leaf("q8_0", 128, 256, 1)
    elif case == "no_outputs":
        fused.weight = fused.weight[:0]
    elif case == "no_tensor_ops":
        routes._OPS["nax"] = False
    elif case == "cpu":
        monkeypatch.setattr(kq_routes, "_gpu", lambda: False)
    assert ops.glu_halves(fused, x) is None
    assert spies.glu.calls == 0


def test_glu_takes_rows_above_32767(spies):
    gate, up = _pair()
    # A batch of two at 2048x2048 flattens to 33792 rows.
    x = _x((2, 16896, 256), mx.bfloat16)
    y = ops.swiglu_fused(x, *ops.glu_pair(gate, up, x))
    mx.eval(y)
    assert spies.glu.calls == 1 and y.shape == (2, 16896, 64)
    err = _rel(_f64(y), _glu_ref(x, gate, up))
    assert err <= PAIR_TOL[mx.bfloat16], err


# A LoRA wrap.

def _kq_dit(dtype=mx.bfloat16):
    """A small Z-Image DiT in ``dtype`` whose w1 and w3 are q4_k leaves."""
    return kq_z_model(dtype)


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
    assert spies.glu.calls == 1
    tid = _table(pipe, path)
    assert isinstance(ff.w1, lora.ImageLoRALinear)
    # Wrapped with no table active: the base leaves, so the kernel.
    assert mx.array_equal(ff(x), base).item()
    assert spies.glu.calls == 2
    prev = lora_install.activate(pipe, tid)
    try:
        on = ff(x)
        mx.eval(on)
        assert spies.glu.calls == 2
        # A pair with no entry in the active table still takes the kernel.
        mx.eval(other(x))
        assert spies.glu.calls == 3
        routes.install(())
        off = ff(x)
        mx.eval(off)
    finally:
        lora_install.restore(pipe, prev)
    assert mx.array_equal(on, off).item()
    assert _rel(_f64(on), _f64(base)) > 1e-2
    assert spies.glu.calls == 3


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
    assert spies.glu.calls == 1


# Kept results.

def test_kept_results_follow_the_routes_in_force(routes, monkeypatch):
    """The pipeline keeps the caption stream and the modulations of a
    step. A kernel rounds at other points than the stock ops, so a result
    kept under one set of routes is not served under another: after a
    change of routes a step is bit for bit the step of a fresh pipeline."""
    routes._OPS.update(qk_norm_rope=qk_rounded_once,
                       quantized_matmul_glu=glu_rounded_once,
                       quantized_matmul_lora=lora_rounded_once, nax=True)
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert routes.install(routes.ROUTES) == set(routes.ROUTES)
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((1, 4, 64, 64)).astype(np.float32)
    cond = api.Conditioning(mx.array(
        rng.standard_normal((5, 16)).astype(np.float32)), key="kept")

    def step(pipe) -> np.ndarray:
        return np.array(api.denoise_step(pipe, lat, 0.6, cond))

    kept = _pipe()
    on = step(kept)
    with kq_routes.only(()):
        off, fresh_off = step(kept), step(_pipe())
    again, fresh_on = step(kept), step(_pipe())
    assert not np.array_equal(on, off)
    assert np.array_equal(off, fresh_off)
    assert np.array_equal(again, on) and np.array_equal(on, fresh_on)
    # Each set of routes keeps its own caption stream and modulation.
    assert len(kept.captions) == len(kept.modulations) == 2


# The profiler.

def test_the_profiler_books_the_fused_pair_as_one_record(spies):
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
    assert ops.glu_pair is kq_routes.glu_pair


def test_the_fused_record_takes_the_path_of_the_swiglu_record(spies):
    m = _kq_dit()
    x = _x((1, 1024, 256), mx.bfloat16)
    ff = m.layers[1].feed_forward

    def paths():
        with prof.StepProfiler(m) as p:
            records, _ = p.step(lambda: ff(x))
        return {r.name: r.path for r in records if r.kind == "seam"}

    fused = paths()
    kq_routes.install(("qk",))
    composed = paths()
    # A composed pair books no fused record.
    assert sorted(fused) == ["swiglu_fused"] and sorted(composed) == ["swiglu"]
    assert fused["swiglu_fused"] == composed["swiglu"]
    assert fused["swiglu_fused"] == "layers.1.feed_forward"


def test_the_routes_cannot_change_under_a_profiler(spies):
    m = _kq_dit()
    with prof.StepProfiler(m):
        timed = ops.qk_norm_rope
        with pytest.raises(RuntimeError, match="StepProfiler"):
            kq_routes.install(())
        assert ops.qk_norm_rope is timed
        assert kq_routes.active() == {"qk", "glu"}
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope
    assert kq_routes.install(()) == frozenset()


# The kernels against a float64 reference, beside the stock ops.

@needs_qk
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f16"])
@pytest.mark.parametrize("shape,scale", [
    ((2, 40, 2, 64), 1.0), ((1, 33, 3, 128), 1.0), ((2, 16896, 1, 64), 1.0),
    ((1, 33, 2, 96), 1.0), ((1, 20, 2, 256), 1.0),
    ((2, 40, 2, 64), EPS ** 0.5)],
    ids=["hd64", "hd128", "rows_33792", "hd96", "hd256", "near_eps"])
def test_qk_kernel_against_the_float64_reference(routes, dtype, shape, scale):
    """The near_eps row has q and k of a mean square near eps, where the
    norm reads its eps: a kernel handed twice or half the eps is 18 or 15
    percent off the reference there, and unseen on the other rows."""
    if not _gpu():
        pytest.skip("the qk_norm_rope kernel runs on the GPU")
    assert routes.install(("qk",)) == {"qk"}
    q, k, qw, kw, cos, sin = _qk_inputs(*shape, dtype, scale=scale)
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
        if scale != 1.0:
            for wrong in (0.5, 2.0):
                off = _rel(_f64(g), _qk_ref(x, w, cos, sin, EPS * wrong))
                assert off > 20 * QK_TOL[dtype], (wrong, off)


@needs_glu
@pytest.mark.parametrize("codec,dtype", GLU_ROWS,
                         ids=lambda v: v if isinstance(v, str)
                         else str(v).split(".")[-1])
@pytest.mark.parametrize("lead", [(1024,), (1, 1056), (2, 16896)],
                         ids=["rows_1024", "rows_1056", "rows_33792"])
def test_glu_kernel_against_the_float64_reference(routes, codec, dtype, lead):
    if not _fused_glu():
        pytest.skip("the GLU kernel runs on a GPU with tensor ops")
    assert routes.install(("glu",)) == {"glu"}
    gate, up = _pair(codec)
    x = _x((*lead, 256), dtype)
    pair = ops.glu_pair(gate, up, x)
    assert pair is not None
    got = ops.swiglu_fused(x, *pair)
    stock = ops.swiglu(gate(x), up(x))
    mx.eval(got, stock)
    assert got.shape == (*lead, 64) and got.dtype == dtype
    ref = _glu_ref(x, gate, up)
    err, err_stock = _rel(_f64(got), ref), _rel(_f64(stock), ref)
    print(f"glu {codec} {lead} {dtype}: kernel {err:.3e} stock "
          f"{err_stock:.3e}")
    assert np.isfinite(_f64(got)).all()
    assert err <= GLU_TOL[dtype], err
    assert err <= 1.1 * err_stock, (err, err_stock)


@needs_glu
@pytest.mark.parametrize("lead", [(1, 4176), (1, 8272)],
                         ids=["t2i_1024", "edit_1024"])
def test_a_flux3_block_mlp_past_4096_rows_against_the_float64_reference(
        routes, lead):
    """FLUX.3 klein's ``Block.hidden`` on the route path, with its gate and
    up halves stored as one q4_k leaf, at the joint rows of a 1024 step:
    4096 image rows and 80 text rows, and those with 4096 reference rows.
    Every row is held, the ones past 4096 too."""
    from gmlx.image.families.flux3_klein import model as fm

    if not _fused_glu():
        pytest.skip("the GLU kernel runs on a GPU with tensor ops")
    assert routes.install(("glu",)) == {"glu"}
    fused = _leaf("q4_k", 128, 256, 1)
    x = _x((*lead, 256), mx.bfloat16)
    assert ops.glu_halves(fused, x) is not None
    block = SimpleNamespace(mlp_in=fused)
    got = fm.Block.hidden(block, x)
    a, b = mx.split(fused(x), 2, axis=-1)
    stock = ops.swiglu(a, b)
    mx.eval(got, stock)
    w = np.array(kq.dequantize(fused.weight, fused.scales, "q4_k",
                               dtype=mx.float32), np.float64)
    xf = _f64(x)
    g, u = xf @ w[:64].T, xf @ w[64:].T
    ref = g / (1.0 + np.exp(-g)) * u
    assert got.shape == (*lead, 64) and got.dtype == mx.bfloat16
    for rows, tag in ((slice(0, 4096), "first 4096"),
                      (slice(4096, None), "past 4096")):
        r, gr, sr = ref[:, rows], _f64(got)[:, rows], _f64(stock)[:, rows]
        err, err_stock = _rel(gr, r), _rel(sr, r)
        print(f"flux3 block glu {lead} {tag}: kernel {err:.3e} stock "
              f"{err_stock:.3e}")
        assert np.isfinite(gr).all()
        assert err <= GLU_TOL[mx.bfloat16], (tag, err)
        assert err <= 1.1 * err_stock, (tag, err, err_stock)


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
