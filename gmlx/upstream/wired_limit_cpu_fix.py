"""The wired limit context sets no limit on the CPU device.

mlx-lm's and mlx-vlm's ``wired_limit`` context raises the Metal wired limit
to the recommended working set for a generation. It checks that Metal is
present, then reads ``max_recommended_working_set_size`` from
``mx.device_info()``, which answers for the default device. The CPU
device's info has no such key, so on a Mac with the CPU as the default
device the context raises KeyError on entry, and so does every generation
and batch generator that enters it. A generation on the CPU device has
nothing to wire.

The guard wraps each stock binding. On the CPU default device it reads no
device info and sets no limit, and it synchronizes the streams on exit as
the stock exit does. On the GPU it runs the stock context. gmlx's own
contexts in ``gmlx.stream.wired_limit`` either make the same check or read
no device info, so the guard leaves them alone.

Install is idempotent and imports each module it guards.
"""

from __future__ import annotations

import contextlib
import importlib

import mlx.core as mx

from gmlx.device import cpu_default_device

# Each module that binds its own name for the context: mlx-vlm's
# generators import it by name from common.
_BINDINGS = (
    "gmlx._vendor.mlx_lm.generate",
    "gmlx._vendor.mlx_vlm.generate",
    "gmlx._vendor.mlx_vlm.generate.common",
    "gmlx._vendor.mlx_vlm.generate.ar",
    "gmlx._vendor.mlx_vlm.generate.dispatch",
    "gmlx._vendor.mlx_vlm.generate.diffusion",
    "gmlx._vendor.mlx_vlm.generate.audio",
)
# The copied contexts, by module prefix; gmlx's own live outside _vendor.
_COPIED = ("gmlx._vendor.mlx_lm.", "gmlx._vendor.mlx_vlm.")
_FLAG = "_gmlx_cpu_guard"


@contextlib.contextmanager
def cpu_wired_limit(streams=None):
    """The wired limit context on the CPU device: no limit, and the stock
    exit synchronize."""
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()


def _guard(stock):
    @contextlib.contextmanager
    def wired_limit(model, streams=None):
        if cpu_default_device():
            with cpu_wired_limit(streams):
                yield
        else:
            with stock(model, streams):
                yield

    setattr(wired_limit, _FLAG, stock)
    return wired_limit


def install_wired_limit_cpu_fix() -> bool:
    """Wrap the stock ``wired_limit`` in each module of ``_BINDINGS``.
    Returns whether any module binds one."""
    found = False
    for name in _BINDINGS:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        fn = getattr(mod, "wired_limit", None)
        if fn is None:
            continue
        found = True
        if getattr(fn, _FLAG, None) is not None:
            continue
        if (getattr(fn, "__module__", None) or "").startswith(_COPIED):
            mod.wired_limit = _guard(fn)
    return found
