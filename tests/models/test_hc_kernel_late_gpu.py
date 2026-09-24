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


@pytest.mark.parametrize("name", ["_hc_sinkhorn_collapse_kernel", "_hc_expand_collapse_kernel"])
def test_a_kernel_missing_at_import_is_built_on_first_use(monkeypatch, name):
    from types import SimpleNamespace

    from gmlx.models.deepseek_v4.hyper_connection import HyperConnection, hc_expand_collapse

    B, L, H, D = 1, 2, 4, 64
    cfg = SimpleNamespace(hidden_size=D, hc_mult=H, hc_sinkhorn_iters=3, hc_eps=1e-6, rms_norm_eps=1e-6)
    hc = HyperConnection(cfg)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.02
    hc.base = mx.random.normal(hc.base.shape) * 0.1
    hc.eval()
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
