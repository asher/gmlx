"""The hyper-connection Metal kernels are built at import. A process that
first imports the module under a CPU default device and later runs a GPU
forward builds them on first use."""
from __future__ import annotations

import os

import mlx.core as mx
import pytest

import gmlx.models.deepseek_v4.hyper_connection as hc_mod

if os.environ.get("KQUANT_FORCE_CPU") or not mx.metal.is_available():
    pytest.skip("the kernels need a real GPU and a GPU default device", allow_module_level=True)


def _hc(H, D):
    from types import SimpleNamespace

    from gmlx.models.deepseek_v4.hyper_connection import HyperConnection

    cfg = SimpleNamespace(hidden_size=D, hc_mult=H, hc_sinkhorn_iters=3, hc_eps=1e-6, rms_norm_eps=1e-6)
    hc = HyperConnection(cfg)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.02
    hc.base = mx.random.normal(hc.base.shape) * 0.1
    hc.eval()
    return hc


@pytest.mark.parametrize("name", ["_hc_sinkhorn_collapse_kernel", "_hc_expand_collapse_kernel"])
def test_a_kernel_missing_at_import_is_built_on_first_use(monkeypatch, name):
    from gmlx.models.deepseek_v4.hyper_connection import hc_expand_collapse

    B, L, H, D = 1, 2, 4, 64
    hc = _hc(H, D)
    x = mx.random.normal((B, L, H, D)).astype(mx.bfloat16)
    ref = hc(x) if name == "_hc_sinkhorn_collapse_kernel" else hc_expand_collapse(
        hc, x[:, :, 0], x, *hc(x)[1:])
    mx.eval(ref)
    monkeypatch.setattr(hc_mod, name, None)
    out = hc(x) if name == "_hc_sinkhorn_collapse_kernel" else hc_expand_collapse(
        hc, x[:, :, 0], x, *hc(x)[1:])
    mx.eval(out)
    assert getattr(hc_mod, name) is not None
    for a, b in zip(out, ref):
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("name", ["_hc_front_reduce_kernel", "_hc_front_expand_reduce_kernel",
                                  "_hc_sinkhorn_collapse_m1_kernel", "_hc_sinkhorn_collapse_lag_kernel",
                                  "_hc_expand_m1_kernel"])
def test_a_one_row_kernel_missing_at_import_is_built_on_first_use(monkeypatch, name):
    """The one-row decode route, the DeepSeek-V4.1 lag collapse included,
    on the in-repo kernels rather than the kquant ones."""
    monkeypatch.setattr(hc_mod, "_KQ_HC", False)
    B, L, H, D = 1, 1, 4, 1024
    hc = _hc(H, D)
    x = mx.random.normal((B, L, H, D)).astype(mx.bfloat16)
    norm_w = (1 + 0.1 * mx.random.normal((D,))).astype(mx.bfloat16)
    post = mx.random.uniform(shape=(B, L, H))
    comb = mx.random.uniform(shape=(B, L, H, H))
    pre_in = mx.random.uniform(shape=(B, L, H))
    assert hc.m1_fused_ok(x)

    def run():
        mixes, ssq = hc.front_m1(x)
        return {
            "_hc_front_reduce_kernel": lambda: (mixes, ssq),
            "_hc_front_expand_reduce_kernel": lambda: hc.front_expand_m1((x[:, :, 0], x, post, comb)),
            "_hc_sinkhorn_collapse_m1_kernel": lambda: hc._collapse_m1(x, mixes, ssq, norm_w),
            "_hc_sinkhorn_collapse_lag_kernel": lambda: hc.lag_collapse_m1(x, mixes, ssq, norm_w, pre_in),
            "_hc_expand_m1_kernel": lambda: (hc_mod.hc_expand_m1(x[:, :, 0], x, post, comb),),
        }[name]()

    ref = run()
    mx.eval(ref)
    monkeypatch.setattr(hc_mod, name, None)
    if name != "_hc_front_reduce_kernel":
        # the front of every other case still runs on its built kernel
        assert hc_mod._hc_front_reduce_kernel is not None
    out = run()
    mx.eval(out)
    assert getattr(hc_mod, name) is not None
    for a, b in zip(out, ref):
        assert mx.array_equal(a, b)


def test_the_one_row_route_is_offered_when_its_kernels_can_be_built(monkeypatch):
    names = ["_hc_front_reduce_kernel", "_hc_front_expand_reduce_kernel", "_hc_sinkhorn_collapse_m1_kernel",
             "_hc_sinkhorn_collapse_lag_kernel", "_hc_expand_m1_kernel"]
    for n in names:
        monkeypatch.setattr(hc_mod, n, None)
    hc = _hc(4, 1024)
    assert hc.m1_fused_ok(mx.zeros((1, 1, 4, 1024), dtype=mx.bfloat16))
    assert hc_mod._hc_sinkhorn_collapse_lag_kernel is not None
