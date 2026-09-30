"""The copy's reads of the working set with the CPU as the default device.

``mx.device_info()`` answers for the default device, and the CPU device's
info has no ``max_recommended_working_set_size``. Where the copy raises the
wired limit, it skips that step when the key is absent, since the CPU
device has nothing to wire. Where it sizes a memory budget, it reads the
Metal device's working set, whatever the default device is. Metal is faked
present with a known working set, so the checks run without it too.
"""

from __future__ import annotations

import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

_WS = 3 << 30


class _Stop(Exception):
    """Ends a run right after its wired limit step."""


class _Args:
    """Training args whose first read, just after the wired limit step,
    ends the run."""

    @property
    def iters(self):
        raise _Stop


@pytest.fixture
def devices(monkeypatch):
    """The CPU as the default device, Metal present, ``device_info``
    answering per device, and the wired limit setter recorded. With
    ``state["default"] = "gpu"`` the default device's info is the GPU's."""
    state = {"default": "cpu", "wired": []}

    def device_info(device=None):
        gpu = (state["default"] == "gpu" if device is None
               else device == mx.gpu)
        if gpu:
            return {"architecture": "applegpu", "device_name": "gpu",
                    "max_recommended_working_set_size": _WS}
        return {"architecture": "arm", "device_name": "cpu"}

    saved = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", device_info)
    monkeypatch.setattr(
        mx, "set_wired_limit", lambda v: (state["wired"].append(v), 0)[1])
    yield state
    mx.set_default_device(saved)


def _batch_generator(monkeypatch):
    from gmlx._vendor.mlx_lm.generate import BatchGenerator

    BatchGenerator(nn.Module()).close()


def _server(monkeypatch):
    from gmlx._vendor.mlx_lm import server

    ran = []
    monkeypatch.setattr(sys, "argv", ["server"])
    monkeypatch.setattr(server, "ModelProvider", lambda args: None)
    monkeypatch.setattr(server, "run", lambda *a: ran.append(a))
    monkeypatch.setattr(server.logging, "basicConfig", lambda **k: None)
    server.main()
    assert ran


def _dwq(monkeypatch):
    from gmlx._vendor.mlx_lm.quant import dwq

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(sys, "argv", ["dwq", "--model", "m"])
    monkeypatch.setattr(dwq, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(dwq, "load_data", lambda *a, **k: ([], []))
    monkeypatch.setattr(dwq, "load", lambda *a, **k: (nn.Module(), None, {}))
    monkeypatch.setattr(dwq, "quantize_model", lambda m, c, **k: (m, c))
    monkeypatch.setattr(dwq, "dwq_quantize", stop)
    with pytest.raises(_Stop):
        dwq.main()


def _lora_train(monkeypatch):
    from gmlx._vendor.mlx_lm.tuner.trainer import train

    with pytest.raises(_Stop):
        train(nn.Module(), None, [], args=_Args())


def _orpo_train(monkeypatch):
    from gmlx._vendor.mlx_vlm.trainer.orpo_trainer import train_orpo

    with pytest.raises(_Stop):
        train_orpo(nn.Module(), None, [], args=_Args())


def _expert_cache(monkeypatch):
    from gmlx._vendor.mlx_vlm.moe_offload import _default_expert_cache_bytes

    assert _default_expert_cache_bytes() == int(0.8 * _WS)


def _apc_working_set(monkeypatch):
    from gmlx._vendor.mlx_vlm.apc import _metal_working_set_bytes

    assert _metal_working_set_bytes() == _WS


# Each site raises the wired limit.
_WIRING = {
    "mlx_lm.generate BatchGenerator": _batch_generator,
    "mlx_lm.server main": _server,
    "mlx_lm.quant.dwq main": _dwq,
    "mlx_lm.tuner.trainer train": _lora_train,
    "mlx_vlm.trainer.orpo_trainer train_orpo": _orpo_train,
}
# Each site sizes a budget from the working set.
_BUDGETS = {
    "mlx_vlm.moe_offload _default_expert_cache_bytes": _expert_cache,
    "mlx_vlm.apc _metal_working_set_bytes": _apc_working_set,
}


@pytest.mark.parametrize("site", [*_WIRING, *_BUDGETS])
def test_each_working_set_site_runs_with_the_cpu_as_default_device(
        devices, monkeypatch, site):
    """A wired limit step sets no limit, and a budget is the Metal
    device's."""
    {**_WIRING, **_BUDGETS}[site](monkeypatch)
    assert devices["wired"] == []


@pytest.mark.parametrize("site", list(_WIRING))
def test_a_wired_limit_step_still_wires_the_working_set_on_the_gpu(
        devices, monkeypatch, site):
    devices["default"] = "gpu"
    _WIRING[site](monkeypatch)
    assert devices["wired"][:1] == [_WS]
