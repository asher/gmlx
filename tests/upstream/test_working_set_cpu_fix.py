"""The copy's reads of the working set with the CPU as the default device.

``mx.device_info()`` answers for the default device, and the CPU device's
info has no ``max_recommended_working_set_size``. Where the copy raises the
wired limit, it skips that step on the CPU device, which has nothing to
wire. Where it sizes a memory budget, it reads the Metal device's working
set, whatever the default device is. Metal is faked present with a known
working set, so the checks run without it too. The generation contexts run
in a child that imports the copy alone, as a caller without gmlx's loader
does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

_WS = 3 << 30
_REPO = Path(__file__).resolve().parents[2]


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


# Each module that binds the copy's generation context; mlx-vlm's
# generators import it by name from common.
_CONTEXTS = (
    "gmlx._vendor.mlx_lm.generate",
    "gmlx._vendor.mlx_vlm.generate",
    "gmlx._vendor.mlx_vlm.generate.common",
    "gmlx._vendor.mlx_vlm.generate.ar",
    "gmlx._vendor.mlx_vlm.generate.dispatch",
    "gmlx._vendor.mlx_vlm.generate.diffusion",
    "gmlx._vendor.mlx_vlm.generate.audio",
)

_CHILD = textwrap.dedent("""
    import importlib, json, sys

    import mlx.core as mx

    default, names, ws = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
    if default == "cpu":
        mx.set_default_device(mx.cpu)
    contexts = {n: importlib.import_module(n).wired_limit for n in names}
    wired = []

    def device_info(device=None):
        gpu = default == "gpu" if device is None else device == mx.gpu
        if gpu:
            return {"max_recommended_working_set_size": ws}
        return {"architecture": "arm", "device_name": "cpu"}

    mx.metal.is_available = lambda: True
    mx.device_info = device_info
    mx.set_wired_limit = lambda v: (wired.append(v), 0)[1]
    if default == "gpu":
        mx.default_device = lambda: mx.Device(mx.gpu)
        mx.synchronize = lambda *a, **k: None
    stream = mx.new_stream(mx.cpu)
    model = {"w": mx.zeros((4,), dtype=mx.float32)}
    origins = {}
    for name, fn in contexts.items():
        origins[name] = fn.__module__
        with fn(model):
            pass
        with fn(model, [stream]):
            pass
    print(json.dumps({"wired": wired, "origins": origins,
                      "loader": "gmlx.load.loader" in sys.modules}))
""")


def _run_contexts(default: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(_REPO), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, default, json.dumps(_CONTEXTS),
         str(_WS)],
        capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    # The copy's own contexts, with no loader to wrap or replace them.
    assert not got["loader"]
    assert set(got["origins"].values()) == {
        "gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_vlm.generate.common"}
    return got


def test_the_copy_generation_contexts_run_with_the_cpu_as_default_device():
    assert _run_contexts("cpu")["wired"] == []


def test_the_copy_generation_contexts_wire_the_working_set_on_the_gpu():
    assert _run_contexts("gpu")["wired"] == [_WS, 0] * 2 * len(_CONTEXTS)
