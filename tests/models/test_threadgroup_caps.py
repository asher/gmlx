"""Pre-M3 threadgroup clamps for the wide custom kernels.

``mx.fast.metal_kernel`` emits no ``max_total_threads_per_threadgroup``
launch bound, so a pipeline's thread ceiling is register-allocation
dependent and lands below 512/1024 on pre-M3 GPUs (measured 448/640 on an
M1 Max). The fused GDN kernels clamp their simdgroup count there; the
ragged SDPA kernels (fixed 1024-thread groups) step aside so the caller's
per-pad-group mx.fast SDPA fallback runs instead. CPU-only.
"""

from __future__ import annotations

import pytest

import gmlx.load.dtypes as dtypes
import gmlx.upstream.gdn_patches as gp
from gmlx.models.qwen35 import attn


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GMLX_GDN_SG", raising=False)
    monkeypatch.delenv("GMLX_FORCE_RAGGED_SDPA", raising=False)
    attn._wide_threadgroups_ok.cache_clear()
    gp.gdn_sg.cache_clear()
    yield
    attn._wide_threadgroups_ok.cache_clear()
    gp.gdn_sg.cache_clear()


# ------------------------------------------------------------------ gdn_sg


def test_gdn_sg_m3_and_later_keeps_wide_shape(monkeypatch):
    for gen in (15, 16, 17):
        monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda g=gen: g)
        gp.gdn_sg.cache_clear()
        assert gp.gdn_sg(1) == 16
        assert gp.gdn_sg(4) == 32


def test_gdn_sg_pre_m3_clamps_to_8(monkeypatch):
    for gen in (13, 14):
        monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda g=gen: g)
        gp.gdn_sg.cache_clear()
        assert gp.gdn_sg(1) == 8
        assert gp.gdn_sg(4) == 8


def test_gdn_sg_unknown_arch_keeps_wide_shape(monkeypatch):
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 0)
    assert gp.gdn_sg(1) == 16
    assert gp.gdn_sg(2) == 32


def test_gdn_sg_env_override_wins(monkeypatch):
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 13)
    monkeypatch.setenv("GMLX_GDN_SG", "16")
    assert gp.gdn_sg(1) == 16
    assert gp.gdn_sg(4) == 16


# ------------------------------------------------------- ragged SDPA gate


def test_ragged_sdpa_gated_off_pre_m3(monkeypatch):
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 13)
    assert not attn._wide_threadgroups_ok()


def test_ragged_sdpa_on_for_m3_and_unknown(monkeypatch):
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 15)
    assert attn._wide_threadgroups_ok()
    attn._wide_threadgroups_ok.cache_clear()
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 0)
    assert attn._wide_threadgroups_ok()


def test_ragged_sdpa_force_flag(monkeypatch):
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 13)
    monkeypatch.setenv("GMLX_FORCE_RAGGED_SDPA", "1")
    assert attn._wide_threadgroups_ok()
