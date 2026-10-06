"""The gates of ``block_gates`` on small models, with stand-ins for the
kernels of ``gmlx.image.ops.kq_routes``. The exact context: which products
it runs dense, and what ``assert_all_dense`` refuses, a kernel route left
in force among them. The site gate: the count each site is held to, the
row rule, the constants of the exact side, and what ``right_sites``
refuses. CPU, no model file."""
from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import block_gates
from block_gates import (KERNEL_ROUNDINGS, LEAVES, SITE_VELOCITY, Sites,
                         assert_all_dense, exact_products, kernel_calls,
                         plain_velocity, rerun_exact, rerun_sites,
                         right_sites, stock_ops_only)
from image_fixtures import glu_rounded_once, kq_z_model, qk_rounded_once

from gmlx.image import ops
from gmlx.image.ops import attention, kq_routes

kq = pytest.importorskip("mlx_kquant")

CODEC = "q4_k"
WIDTH = 256
EXACT_PROCESS = block_gates.exact_process


def _leaf(rng, out: int):
    from mlx_kquant.nn import KQuantLinear

    m = KQuantLinear(WIDTH, out, False, CODEC)
    m.weight, m.scales = kq.quantize(mx.array(
        (rng.standard_normal((out, WIDTH)) / 16).astype(np.float32)), CODEC)
    return m


class _Tiny(nn.Module):
    """Two K-quant linears in a row, with the activation dtype of a DiT."""

    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(0)
        self.a = _leaf(rng, WIDTH)
        self.b = _leaf(rng, 96)
        self.act_dtype = mx.bfloat16

    def __call__(self, x):
        return self.b(self.a(x))


@pytest.fixture(autouse=True)
def _exact(monkeypatch):
    # The tiny model runs no image op and no float32 GPU product, so any
    # process is an exact one for it.
    monkeypatch.setattr(block_gates, "exact_process", lambda: True)
    with mx.stream(mx.cpu):
        yield


@pytest.fixture
def tiny():
    return _Tiny()


def _x() -> mx.array:
    rng = np.random.default_rng(1)
    return mx.array(rng.standard_normal((3, WIDTH)).astype(np.float32))


def _dense(m, x: mx.array) -> mx.array:
    return x @ kq.dequantize(m.weight, m.scales, CODEC, mx.float32).T


def test_a_module_call_runs_as_a_dense_float32_product(tiny):
    x = _x()
    with exact_products(tiny) as used:
        assert tiny.act_dtype == mx.float32
        y = tiny(x)
        mx.eval(y)
    assert tiny.act_dtype == mx.bfloat16
    assert y.dtype == mx.float32
    assert mx.array_equal(y, _dense(tiny.b, _dense(tiny.a, x)))
    assert used == {id(tiny.a), id(tiny.b)}
    assert used.direct == 0 and used.kernel == []
    assert_all_dense(tiny, used)


def test_a_module_that_ran_no_dense_product_is_refused(tiny):
    with exact_products(tiny) as used:
        mx.eval(tiny.a(_x()))
    with pytest.raises(AssertionError, match=r"1 K-quant products ran off "
                                             r"the exact path, such as "
                                             r"\['b'\]"):
        assert_all_dense(tiny, used)


def test_band_products_past_the_module_run_dense(tiny):
    """The split placement calls ``kq.quantized_matmul`` with one band of
    a module's rows and the module's scales. In the exact context each
    band is the dense product on its rows, and the module counts as
    dense."""
    x = _x()
    real = kq.quantized_matmul
    bands = ((0, 32), (32, 96))
    with exact_products(tiny) as used:
        h = tiny.a(x)
        y = mx.concatenate([kq.quantized_matmul(
            h, mx.contiguous(tiny.b.weight[s:e]), tiny.b.scales, CODEC,
            transpose=True) for s, e in bands], axis=-1)
        mx.eval(y)
    assert kq.quantized_matmul is real
    assert used.direct == 2 and used.kernel == []
    assert_all_dense(tiny, used)
    # A band's rows dequantize to the rows of the whole weight, so the
    # band products are the whole product but for the order of a float32
    # sum.
    whole = kq.dequantize(tiny.b.weight, tiny.b.scales, CODEC, mx.float32)
    for s, e in bands:
        assert mx.array_equal(kq.dequantize(
            mx.contiguous(tiny.b.weight[s:e]), tiny.b.scales, CODEC,
            mx.float32), whole[s:e])
    want = np.array(_dense(tiny.b, _dense(tiny.a, x)), np.float64)
    got = np.array(y, np.float64)
    assert y.dtype == mx.float32
    assert np.linalg.norm(got - want) <= 1e-6 * np.linalg.norm(want)


def test_a_product_on_a_kernel_is_refused_beside_its_dense_one(
        tiny, monkeypatch):
    """Every module ran dense, and one product also went to a kernel: on
    a weight with scales of no module, or through another entry of
    mlx_kquant that took a module's weight. Each is refused."""
    x = _x()
    seen = []
    monkeypatch.setattr(kq, "gather_qmm",
                        lambda x, w, *a, **k: seen.append(w) or x)
    stand_in = kq.gather_qmm
    for name, call, record in (
            ("other scales", lambda: kq.quantized_matmul(
                x, tiny.a.weight, mx.array(tiny.a.scales), CODEC,
                transpose=True), ("quantized_matmul", CODEC)),
            ("another entry", lambda: kq.gather_qmm(
                x, tiny.b.weight, tiny.b.scales), ("gather_qmm", "b"))):
        with exact_products(tiny) as used:
            mx.eval(tiny(x))
            # The product is built and not run: the record is taken at
            # the call.
            call()
        assert used == {id(tiny.a), id(tiny.b)}, name
        assert used.kernel == [record], name
        with pytest.raises(AssertionError, match="1 K-quant products ran "
                                                 "on a kernel in the exact "
                                                 "process"):
            assert_all_dense(tiny, used)
        assert kq.gather_qmm is stand_in, name
    assert len(seen) == 1 and seen[0] is tiny.b.weight
    # An entry that takes no weight of the model is not a product of it.
    with exact_products(tiny) as used:
        mx.eval(tiny(x))
        kq.gather_qmm(x, mx.array(tiny.b.weight), tiny.b.scales)
    assert_all_dense(tiny, used)


def test_the_entries_are_put_back_when_the_run_raises(tiny):
    from mlx_kquant.nn import KQuantLinear

    before = (kq.quantized_matmul, kq.gather_qmm, KQuantLinear.__call__)
    with pytest.raises(RuntimeError, match="stop"):
        with exact_products(tiny):
            raise RuntimeError("stop")
    assert (kq.quantized_matmul, kq.gather_qmm,
            KQuantLinear.__call__) == before
    assert tiny.act_dtype == mx.bfloat16


# The kernel routes of gmlx.image.ops.

@pytest.fixture
def routes(monkeypatch):
    """Both routes in force on stand-in ops with a GPU default device
    reported, put back after the test."""
    calls = []

    def glu(x, w_gate, w_up, codec):
        calls.append("glu")
        return x

    def qk(q, k, *rest):
        calls.append("qk")
        return q, k

    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(
        kq_routes._OPS, qk_norm_rope=qk, quantized_matmul_glu=glu, nax=True))
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert kq_routes.install(kq_routes.ROUTES) == set(kq_routes.ROUTES)
    return calls


def _qk_args(dtype=mx.bfloat16) -> tuple:
    q = mx.zeros((1, 2, 1, 64), dtype)
    return (q, q, mx.ones((64,)), mx.ones((64,)), mx.ones((2, 32)),
            mx.zeros((2, 32)), 1e-5)


@pytest.mark.parametrize("route", ["glu", "qk"])
def test_a_kernel_route_left_in_force_is_refused(tiny, routes, route):
    """The routes keep their ops in a table of their own, past the names
    of mlx_kquant that the exact context watches. A route left in force
    takes a call in bfloat16, as of another model stepped in the exact
    process. The call is recorded by the name of its op, beside the dense
    products of every module, and refused."""
    x = _x()
    half = mx.zeros((1024, WIDTH), mx.bfloat16)
    table = dict(kq_routes._OPS)
    with exact_products(tiny) as used:
        mx.eval(tiny(x))
        if route == "glu":
            pair = ops.glu_pair(tiny.a, tiny.a, half)
            assert pair is not None
            ops.swiglu_fused(half, *pair)
        else:
            assert ops.qk_norm_rope is kq_routes.qk_norm_rope
            ops.qk_norm_rope(*_qk_args())
    assert routes == [route]
    assert used == {id(tiny.a), id(tiny.b)}
    assert used.kernel == [(block_gates.KERNEL_OPS[route], "kernel route")]
    with pytest.raises(AssertionError, match="1 K-quant products ran on a "
                                             "kernel in the exact process"):
        assert_all_dense(tiny, used)
    assert kq_routes._OPS == table


def test_the_exact_context_refuses_a_process_with_a_route_in_force(
        tiny, routes, monkeypatch):
    monkeypatch.setattr(block_gates, "tf32", lambda: False)
    monkeypatch.setattr(block_gates, "exact_process", EXACT_PROCESS)
    assert not block_gates.stock_ops()
    with pytest.raises(AssertionError, match="GMLX_IMAGE_KQ_ROUTES=0"):
        with exact_products(tiny):
            pass
    assert tiny.act_dtype == mx.bfloat16
    with kq_routes.only(()):
        assert block_gates.stock_ops()
        with exact_products(tiny) as used:
            mx.eval(tiny(_x()))
        assert_all_dense(tiny, used)
    assert not block_gates.stock_ops() and routes == []


def test_stock_ops_only_puts_the_routes_off_and_back(routes):
    args = _qk_args()
    q = args[0]
    table = dict(kq_routes._OPS)
    with stock_ops_only():
        assert kq_routes.active() == frozenset()
        assert ops.qk_norm_rope is attention.qk_norm_rope
        ops.qk_norm_rope(*args)
    assert routes == []
    assert kq_routes.active() == set(kq_routes.ROUTES)
    assert ops.qk_norm_rope is kq_routes.qk_norm_rope
    assert kq_routes._OPS == table
    # A call that still reaches a route's op in the block fails it.
    with pytest.raises(AssertionError, match="ran on the stock path"):
        with stock_ops_only():
            kq_routes.swiglu_fused(q, q, q, CODEC)
    assert routes == ["glu"]
    assert kq_routes.active() == set(kq_routes.ROUTES)
    assert kq_routes._OPS == table


def test_kernel_calls_names_each_call_a_route_hands_over(routes):
    seen = []
    table = dict(kq_routes._OPS)
    x = mx.zeros((1, 4), mx.bfloat16)
    with kernel_calls(seen.append) as has:
        assert has
        kq_routes.swiglu_fused(x, x, x, CODEC)
        ops.qk_norm_rope(*_qk_args())
        kq_routes.swiglu_fused(x, x, x, CODEC)
    assert seen == ["glu", "qk", "glu"] == routes
    assert kq_routes._OPS == table
    kq_routes.swiglu_fused(x, x, x, CODEC)
    assert seen == ["glu", "qk", "glu"]


@pytest.mark.parametrize("absent", [(), ("qk",), ("glu",), ("qk", "glu")])
def test_kernel_calls_yields_whether_the_build_has_a_route(routes,
                                                           monkeypatch,
                                                           absent):
    for route in absent:
        monkeypatch.setitem(kq_routes._OPS, block_gates.KERNEL_OPS[route],
                            None)
    table = dict(kq_routes._OPS)
    with kernel_calls(lambda route: None) as has:
        assert has == (len(absent) < 2)
    assert kq_routes._OPS == table


def test_kernel_calls_looks_a_cold_op_up_before_it_watches(routes,
                                                           monkeypatch):
    """A process with the switch at 0 has not looked its ops up. The
    watch does, so a route put in force later is still seen."""
    stand_in = kq_routes._OPS["quantized_matmul_glu"]
    monkeypatch.setattr(kq, "quantized_matmul_glu", stand_in, raising=False)
    monkeypatch.setattr(kq, "qk_norm_rope", None, raising=False)
    monkeypatch.setattr(kq_routes, "_OPS", {})
    seen = []
    x = mx.zeros((1, 4), mx.bfloat16)
    with kernel_calls(seen.append) as has:
        assert has
        kq_routes.swiglu_fused(x, x, x, CODEC)
    assert seen == ["glu"] == routes
    assert kq_routes._OPS == {"quantized_matmul_glu": stand_in,
                              "qk_norm_rope": None}


# The child processes of the gated tests.

CHILD = "GMLX_BLOCK_GATES_CHILD"


def test_the_exact_child_runs_with_tf32_and_the_routes_off(request,
                                                           monkeypatch):
    if os.environ.get(CHILD) != "1":
        monkeypatch.setenv(CHILD, "1")
        return rerun_exact(request, "exact child")
    assert os.environ.get("MLX_ENABLE_TF32") == "0" and not block_gates.tf32()
    assert os.environ.get("GMLX_IMAGE_KQ_ROUTES") == "0"
    assert kq_routes._WANTED == () and block_gates.stock_ops()


def test_the_site_child_keeps_the_routes_of_its_parent(request, monkeypatch):
    """The production side of a site is the path a request takes, so the
    child reads the switch as its parent did, and takes the parent's
    velocity."""
    if os.environ.get(CHILD) != "1":
        monkeypatch.setenv(CHILD, "1")
        return rerun_sites(request, "site child",
                           repr(os.environ.get("GMLX_IMAGE_KQ_ROUTES")))
    assert os.environ.get("MLX_ENABLE_TF32") == "0" and block_gates.site_process()
    assert os.environ[SITE_VELOCITY] == repr(
        os.environ.get("GMLX_IMAGE_KQ_ROUTES"))


# The site gate, on a small model with stand-ins for the kernels.

F32 = mx.float32
QK = ("noise_refiner.0.qk", "context_refiner.0.qk", "layers.0.qk",
      "layers.1.qk")
GLU = ("layers.0.glu", "layers.1.glu")
STOCK_GLU = ("noise_refiner.0.glu", "context_refiner.0.glu")
# The gates in numbers, so a gate or a unit that changes moves a reading
# out of its band: 2 n u at u = 2^-8.
GATE = {1: 2.0 ** -7, 3: 6 * 2.0 ** -8, 4: 2.0 ** -5, 5: 10 * 2.0 ** -8}
BAND = (1.3, 1.8)
# Scales of the delta of ``_gate_delta`` that change the glu output of its
# pair by half a gate of the kernel's count and by two.
PAIR_SMALL, PAIR_LARGE = 0.0006, 0.0025


@pytest.fixture(scope="module")
def small():
    """A small bfloat16 Z-Image DiT and the inputs of one step, 1024 image
    rows and 32 caption rows. The gate and up leaves of the two main
    layers are K-quant, so their pair takes the glu route. The refiners'
    leaves are float, and compose."""
    rng = np.random.default_rng(3)
    arrays = {"latents": rng.standard_normal((1, 4, 64, 64)).astype(
                  np.float32),
              "sigma": np.float32(0.6),
              "cap_feats": rng.standard_normal((5, 16)).astype(np.float32)}
    return kq_z_model(mx.bfloat16, groups=("layers",)), arrays


@pytest.fixture
def kernels(monkeypatch):
    """Both routes in force on stand-ins with the kernels' arithmetic and
    a GPU default device reported, put back after the test."""
    monkeypatch.setattr(ops, "qk_norm_rope", ops.qk_norm_rope)
    monkeypatch.setattr(kq_routes, "_active", kq_routes._active)
    monkeypatch.setattr(kq_routes, "_OPS", dict(
        kq_routes._OPS, qk_norm_rope=qk_rounded_once,
        quantized_matmul_glu=glu_rounded_once, nax=True))
    monkeypatch.setattr(kq_routes, "_gpu", lambda: True)
    assert kq_routes.install(kq_routes.ROUTES) == set(kq_routes.ROUTES)


def _read(small, fault=None) -> tuple:
    """The sites of one step of the small model under ``fault``, read
    against the right model, and the digest of the velocity."""
    model, arrays = small
    sites = Sites(model)
    with pytest.MonkeyPatch.context() as mp:
        if fault is not None:
            fault(mp, model)
        v = sites.run(arrays)
    return sites, v


def _scaled(a: mx.array, mask: np.ndarray) -> mx.array:
    return (a.astype(F32) * mx.array(mask.astype(np.float32))).astype(a.dtype)


def qk_kernel(change):
    """The qk kernel's result changed by ``change(q, k)``, both
    (B, H, L, D)."""
    def fault(mp, model):
        real = kq_routes._OPS["qk_norm_rope"]
        mp.setitem(kq_routes._OPS, "qk_norm_rope",
                   lambda *args: change(*real(*args)))
    return fault


def _head(a: mx.array, head: int, value: float, token) -> mx.array:
    """Head ``head`` of ``a`` (B, H, L, D) times ``value``, at ``token``
    where ``a`` has that row, or at every token for None."""
    m = np.ones((1, a.shape[1], a.shape[2], 1), np.float32)
    if token is None:
        m[0, head] = value
    elif token < a.shape[2]:
        m[0, head, token] = value
    return _scaled(a, m)


def head_of_q(head: int, value: float, token=None):
    return qk_kernel(lambda q, k: (_head(q, head, value, token), k))


def head_of_k(head: int, value: float, token=None):
    return qk_kernel(lambda q, k: (q, _head(k, head, value, token)))


def glu_rows(value: float, row=None):
    """The glu kernel's result of (B, L, N) times ``value``, at ``row`` or
    at every row."""
    def fault(mp, model):
        real = kq_routes._OPS["quantized_matmul_glu"]

        def op(x, *rest):
            out = real(x, *rest)
            m = np.ones((1, out.shape[1], 1), np.float32)
            m[0, slice(None) if row is None else row] = value
            return _scaled(out, m)
        mp.setitem(kq_routes._OPS, "quantized_matmul_glu", op)
    return fault


def test_the_right_small_model_passes_the_site_gate(small, kernels,
                                                    monkeypatch):
    model, arrays = small
    plain = plain_velocity(model, arrays)
    monkeypatch.setenv(SITE_VELOCITY, plain)
    sites, v = _read(small)
    assert v == plain
    right_sites(sites, v, "small")
    # Every block's qk site and the main layers' glu sites ran on a
    # kernel, and each is held to the kernel's count.
    assert sites.routed.keys() == set(QK) | set(GLU)
    assert KERNEL_ROUNDINGS == {"qk": 1, "glu": 4}
    assert {n: sites.sites[n][1] for n in QK + GLU + STOCK_GLU} == {
        **dict.fromkeys(QK, 1), **dict.fromkeys(GLU, 4),
        **dict.fromkeys(STOCK_GLU, 5)}
    for name in QK + GLU + STOCK_GLU:
        assert sites.gate(name) == GATE[sites.sites[name][1]], name
    # One rounding to nearest reads 0.43 u over all rows.
    reads, heads = sites.readings(), sites.worst_heads()
    assert heads.keys() == set(QK)
    for name in QK:
        assert 0.17 < reads[name]["all"] < 0.26, (name, reads[name])
        assert reads[name]["all"] < heads[name][0] <= 0.55, heads[name]
    for name in GLU:
        assert 0.04 < reads[name]["all"] < 0.07, (name, reads[name])
    # The step of another process is refused.
    monkeypatch.setenv(SITE_VELOCITY, plain[::-1])
    with pytest.raises(AssertionError, match="not a step of the parent"):
        right_sites(sites, v, "small")


def test_a_stock_site_keeps_the_stock_count(small, kernels, monkeypatch):
    """With the qk route off the qk sites run the stock seam, which
    rounds the norm, the weight product and the rotation."""
    model, arrays = small
    with kq_routes.only(("glu",)):
        sites, v = _read(small)
        monkeypatch.setenv(SITE_VELOCITY, plain_velocity(model, arrays))
        right_sites(sites, v, "small, qk on the stock seam")
        assert sites.routed.keys() == set(GLU)
        assert {sites.sites[n][1] for n in QK} == {3}
        assert all(sites.gate(n) == GATE[3] for n in QK)
        assert not sites.missed_rows()

        # 0.036 is 1.54 gates of three roundings, and under one gate of
        # six.
        def scaled(mp, model):
            real = ops.qk_norm_rope
            mp.setattr(ops, "qk_norm_rope", lambda *args: tuple(
                _scaled(o, np.float32(1.036)) for o in real(*args)))

        got, _ = _read(small, scaled)
        missed = got.missed()
        assert missed.keys() == set(QK), sorted(missed)
        assert all(BAND[0] < r < BAND[1] for r in missed.values()), missed
        # The row rule is of a site on a kernel.
        assert not got.missed_rows()
    assert kq_routes.active() == set(kq_routes.ROUTES)

    # The refiners' pairs compose: 0.06 is 1.54 gates of five roundings.
    def swiglu(mp, model):
        real = ops.swiglu
        mp.setattr(ops, "swiglu", lambda a, b: _scaled(
            real(a, b), np.float32(1.06)))

    got, v = _read(small, swiglu)
    missed = got.missed()
    assert {n for n in missed if n.endswith(".glu")} == set(STOCK_GLU), missed
    assert all(BAND[0] < missed[n] < BAND[1] for n in STOCK_GLU), missed
    assert not got.missed_rows()
    # A site off a kernel fails the site gate, through right_sites.
    monkeypatch.delenv(SITE_VELOCITY)
    with pytest.raises(AssertionError, match="site gate: relative L2"):
        right_sites(got, v, "small, swiglu scaled")


def test_a_kernel_site_is_held_to_the_count_of_the_kernel(small, kernels):
    """A fault of 1.54 gates of the kernel's count fails, where the count
    of the stock ops would pass it."""
    # Every row of the glu kernel times 1.048: 1.54 gates of four
    # roundings, 1.23 of five.
    got, v = _read(small, glu_rows(1.048))
    missed, rows = got.missed(), got.missed_rows()
    assert missed.keys() == rows.keys() == set(GLU), (missed, rows)
    for name in GLU:
        assert BAND[0] < missed[name] < BAND[1], missed
        assert BAND[0] < rows[name] < 2.0, rows
    with pytest.raises(AssertionError, match="site gate: relative L2"):
        right_sites(got, v, "small, the glu kernel scaled")

    # q and k of the qk kernel times 1.012: 1.54 gates of one rounding,
    # half a gate of three.
    got, _ = _read(small, qk_kernel(lambda q, k: (
        _scaled(q, np.float32(1.012)), _scaled(k, np.float32(1.012)))))
    missed = got.missed()
    assert missed.keys() == set(QK), sorted(missed)
    assert all(BAND[0] < r < BAND[1] for r in missed.values()), missed


def test_one_head_of_a_kernel_fails_the_head_row_alone(small, kernels):
    """One head of q times 1.012 is 1.54 gates on its head rows, and up
    to 2.1 on the worst of them with the rounding of the row. It is an
    eighth of a token row of the four heads of q and k, which passes."""
    got, v = _read(small, head_of_q(1, 1.012))
    assert not got.missed()
    rows, heads = got.worst_rows(), got.worst_heads()
    for name in QK:
        assert rows[name][0] < 1, (name, rows[name])
        assert BAND[0] < heads[name][0] < 2.6, (name, heads[name])
        assert heads[name][1][:2] == ("q", 1), heads[name]
    assert got.missed_rows().keys() == set(QK)
    with pytest.raises(AssertionError, match="row rule of a site on a kernel"):
        right_sites(got, v, "small, one head of q scaled")


@pytest.mark.parametrize("part", ["q", "k"])
def test_one_row_of_q_or_k_alone_fails_the_row_rule(small, kernels, part):
    """One head of one token of q, or of k, times 1.2: 25.6 gates on its
    head row, and that over the root of its share of the token row of q
    and k, an eighth for heads of one norm. Over all rows the reading
    does not move."""
    token = 700
    fault = (head_of_q if part == "q" else head_of_k)(2, 1.2, token)
    got, v = _read(small, fault)
    assert not got.missed()
    # The caption stream of the context refiner has no such row.
    hit = set(QK) - {"context_refiner.0.qk"}
    assert got.missed_rows().keys() == hit
    for name in hit:
        row, at = got.worst_rows()[name]
        assert at == token and 4 < row < 16, (name, row, at)
        head, where = got.worst_heads()[name]
        assert where == (part, 2, token) and 20 < head < 30, (head, where)
    with pytest.raises(AssertionError, match="row rule of a site on a kernel"):
        right_sites(got, v, f"small, one row of {part} scaled")


def test_one_row_of_the_glu_kernel_fails_the_row_rule(small, kernels):
    got, v = _read(small, glu_rows(1.2, 700))
    assert not got.missed()
    rows = got.missed_rows()
    assert rows.keys() == set(GLU), sorted(rows)
    for name in GLU:
        assert got.worst_rows()[name][1] == 700
        assert 5.5 < rows[name] < 7.5, rows
    with pytest.raises(AssertionError, match="row rule of a site on a kernel"):
        right_sites(got, v, "small, one row of the glu kernel scaled")


def test_a_row_that_is_not_finite_is_the_worst_row(small, kernels):
    """A row of NaN reads as the worst row of its site and fails the row
    rule and the site gate. The sites after it take NaN on every row, so
    the first site of the fault is the one read here."""
    got, _ = _read(small, glu_rows(np.nan, 700))
    v, row = got.worst_rows()["layers.0.glu"]
    assert row == 700 and not np.isfinite(v), (v, row)
    assert got.missed_rows()["layers.0.glu"] == np.inf
    assert got.missed()["layers.0.glu"] == np.inf
    got, _ = _read(small, head_of_k(3, np.nan, 5))
    for name in QK[:2]:
        v, where = got.worst_heads()[name]
        assert where == ("k", 3, 5) and not np.isfinite(v), (v, where)
        assert got.worst_rows()[name][1] == 5
        assert got.missed_rows()[name] == np.inf


def _gate_and_up_swapped(mp, model):
    ff = model.layers[0].feed_forward
    w1, w3 = ff.w1, ff.w3
    mp.setattr(ff, "w1", w3)
    mp.setattr(ff, "w3", w1)


def _qk_weights_swapped(mp, model):
    att = model.layers[0].attention
    qw, kw = att.q_norm.weight, att.k_norm.weight
    mp.setattr(att.q_norm, "weight", kw)
    mp.setattr(att.k_norm, "weight", qw)


def _rope_of_another_layout(mp, model):
    real = model.rope
    mp.setattr(model, "rope",
               lambda ht, wt, n_cap: real(ht // 2, 2 * wt, n_cap))


@pytest.mark.parametrize("fault,sites_missed", [
    (_gate_and_up_swapped, {"layers.0.glu"}),
    (_qk_weights_swapped, {"layers.0.qk"}),
    (lambda mp, model: mp.setattr(model.layers[0].attention, "eps", 1.0),
     {"layers.0.qk"}),
    (_rope_of_another_layout, {"noise_refiner.0.qk", "layers.0.qk",
                               "layers.1.qk"})],
    ids=["gate_and_up", "qk_norm_weights", "qk_eps", "rope_layout"])
def test_the_exact_side_takes_the_constants_of_the_right_model(
        small, kernels, fault, sites_missed):
    """A block that hands its kernel another constant fails at the site
    of the constant: the exact side takes the leaves and the norm weights
    the reader kept, the eps of the model's config and the model's own
    RoPE tables, and none of them from the call."""
    got, _ = _read(small, fault)
    hit = {n for n in got.missed() if n.endswith((".qk", ".glu"))}
    assert hit == sites_missed, sorted(got.missed())
    assert not got.broken, got.broken


def test_the_eps_of_the_exact_side_is_the_one_of_the_config(small, kernels,
                                                           monkeypatch):
    """A reader made of a model whose block already holds another eps
    fails at that block: the exact side takes the eps of the config,
    which no block holds."""
    model, arrays = small
    monkeypatch.setattr(model.layers[0].attention, "eps", 1.0)
    sites = Sites(model)
    assert sites.eps == model.cfg.norm_eps != 1.0
    sites.run(arrays)
    assert {n for n in sites.missed() if n.endswith(".qk")} == {"layers.0.qk"}


def _gate_delta(model, scale: float):
    """An adapter table of the reader with one rank 2 delta, on the gate
    leaf of the first main layer."""
    w1 = model.layers[0].feed_forward.w1
    n_out, n_in = w1.weight.shape[0], model.cfg.dim
    rng = np.random.default_rng(3)
    a_t = mx.array(rng.standard_normal((n_in, 2)).astype(np.float32))
    b_t = mx.array((scale * rng.standard_normal((2, n_out))
                    ).astype(np.float32))
    path = f"layers.0.{LEAVES['w1']}"
    return lambda p: (a_t, b_t) if p == path else None


def test_a_pair_without_its_deltas_reads_them_at_its_glu_site(small, kernels):
    """The reader holds an adapter table and the model holds none, as
    under a table that leaves a group out: the pair runs on the kernel
    without its delta. The glu site reads the change of the delta within
    the kernel's rounding, at the kernel's count."""
    model, arrays = small
    plain = Sites(model)
    plain.run(arrays)
    assert plain.pair_effect() == {}
    own = plain.readings()["layers.0.glu"]["all"]
    found = {}
    for scale in (PAIR_SMALL, PAIR_LARGE):
        got = Sites(model, _gate_delta(model, scale))
        got.run(arrays)
        assert got.routed.keys() == set(QK) | set(GLU)
        assert got.sites["layers.0.glu"][1] == 4
        pairs = got.pair_effect()
        assert pairs.keys() == {"layers.0.glu"}
        c, r = pairs["layers.0.glu"]["all"]
        read = got.readings()["layers.0.glu"]["all"]
        print(f"pair scale {scale}: c {c:.3f} r {r:.3f} reads {read:.3f}")
        # r is the kernel's own rounding, over the norm of the exact
        # output with the delta.
        assert r == pytest.approx(own, rel=0.1), (r, own)
        assert c - r - 1e-4 <= read <= c + r + 1e-4, (c, r, read)
        assert ("layers.0.glu" in got.missed()) == (read > 1)
        found[scale] = c
    # Under the gate the site passes, and over it the site misses.
    assert found[PAIR_SMALL] + own < 1 < found[PAIR_LARGE] - own, found
    assert found[PAIR_LARGE] / found[PAIR_SMALL] == pytest.approx(
        PAIR_LARGE / PAIR_SMALL, rel=0.2)


def test_right_sites_refuses_what_no_site_reads(small, kernels):
    """An op between two sites that is no site itself, and a kernel call
    in a block that has no site for it."""
    model, _ = small
    ff = model.noise_refiner[0].feed_forward
    real = type(ff).__call__

    def unread(mp, model):
        # One unit of rounding more: no site misses.
        def call(self, x):
            if self is not ff:
                return real(self, x)
            return self.w2(_scaled(ops.swiglu(self.w1(x), self.w3(x)),
                                   np.float32(1 + 2.0 ** -8)))
        mp.setattr(type(ff), "__call__", call)

    got, v = _read(small, unread)
    assert not got.missed() and not got.missed_rows()
    assert got.broken == ["noise_refiner.0: an op between swiglu and w2 is "
                          "not a site"]
    with pytest.raises(AssertionError, match="between swiglu and w2"):
        right_sites(got, v, "small, an op that is no site")

    got, v = _read(small)
    right_sites(got, v, "small")
    got.routed["layers.0.rope"] = "qk_norm_rope"
    with pytest.raises(AssertionError, match="kernel calls at no site"):
        right_sites(got, v, "small, a kernel call with no site")
