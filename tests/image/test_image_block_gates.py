"""The exact context of ``block_gates`` on a tiny K-quant model: which
products it runs dense, and what ``assert_all_dense`` refuses, a kernel
route left in force among them. CPU, no model file."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import block_gates
from block_gates import (assert_all_dense, exact_products, kernel_calls,
                         stock_ops_only)

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
    """Both routes in force on stand-in ops, put back after the test."""
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
    assert kq_routes.install(kq_routes.ROUTES) == set(kq_routes.ROUTES)
    return calls


def test_a_kernel_route_left_in_force_is_refused(tiny, routes):
    """The routes keep their ops in a table of their own, past the names
    of mlx_kquant that the exact context watches. A call a route hands to
    its op in the exact context is recorded, beside the dense products of
    every module, and refused."""
    x = _x()
    table = dict(kq_routes._OPS)
    with exact_products(tiny) as used:
        mx.eval(tiny(x))
        kq_routes.swiglu_fused(x, tiny.a.weight, tiny.a.weight, CODEC)
    assert routes == ["glu"]
    assert used == {id(tiny.a), id(tiny.b)}
    assert used.kernel == [("quantized_matmul_glu", "kernel route")]
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
    q = mx.zeros((1, 2, 1, 64), mx.bfloat16)
    args = (q, q, mx.ones((64,)), mx.ones((64,)), mx.ones((2, 32)),
            mx.zeros((2, 32)), 1e-5)
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
        kq_routes.swiglu_fused(x, x, x, CODEC)
    assert seen == ["glu", "glu"] == routes
    assert kq_routes._OPS == table
    kq_routes.swiglu_fused(x, x, x, CODEC)
    assert seen == ["glu", "glu"]
