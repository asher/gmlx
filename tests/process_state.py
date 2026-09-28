"""Process-wide state each test hands back as it found it.

CLI and library entry points that tests run change the whole process:
``gmlx distill`` and ``gmlx train`` turn MLX_ENABLE_TF32 off for their own
run, ``gmlx serve`` exports its settings, ``--stream-cpu`` lifts the MLX
command buffer caps, the umbrella replaces ``warnings.formatwarning``, and
library entry points set the MLX default device, seed the RNG, set the
memory, cache and wired limits and the mlx-kquant command buffer caps,
and rebind ``mx.set_wired_limit``.

``capture`` and ``restore`` bracket every test (tests/conftest.py). MLX
reads MLX_ENABLE_TF32 and the command buffer caps once, at the first
matmul and the first command buffer, so ``start_session`` also runs one
small GPU matmul: the process keeps the session's settings even when a
test changes the environment before its own first matmul.
"""

from __future__ import annotations

import os
import warnings

import mlx.core as mx

SESSION: dict = {}

# First value each setter replaced in the current test, and the RNG key
# before the test's first seed.
_prior: dict = {}
_setters: dict = {}

_LIMITS = ("set_memory_limit", "set_cache_limit", "set_wired_limit")
# pytest sets it per phase and removes it after each one.
_ENV_SKIP = ("PYTEST_CURRENT_TEST",)


def _gpu() -> bool:
    return not os.environ.get("KQUANT_FORCE_CPU") and mx.metal.is_available()


def _record(name: str, fn):
    def setter(*args, **kwargs):
        prev = fn(*args, **kwargs)
        _prior.setdefault(name, prev)
        return prev

    setter._gmlx_test_recorder = fn
    return setter


def _record_seed(fn):
    def seed(*args, **kwargs):
        _prior.setdefault("rng", mx.random.state[0])
        return fn(*args, **kwargs)

    seed._gmlx_test_recorder = fn
    return seed


def start_session() -> None:
    """Record the session-start environment and default device, wrap the
    MLX limit setters and the RNG seed so a test's changes can be undone,
    and pin MLX's once-per-process reads to the session environment."""
    SESSION["env"] = dict(os.environ)
    SESSION["device"] = str(mx.default_device())
    for name in _LIMITS:
        fn = getattr(mx, name)
        _setters[name] = fn
        setattr(mx, name, _record(name, fn))
    _setters["seed"] = mx.random.seed
    mx.random.seed = _record_seed(mx.random.seed)
    # Held here, since a test may stand a stub in for mlx_kquant.
    try:
        import mlx_kquant as kq

        _setters["get_cb_caps"] = kq.get_cb_caps
        _setters["set_cb_caps"] = kq.set_cb_caps
    except (ImportError, AttributeError):
        pass
    if _gpu():
        x = mx.ones((32, 32))
        mx.eval(mx.matmul(x, x, stream=mx.gpu))


def cb_caps():
    """The MLX command buffer caps mlx-kquant sets, or None without them."""
    get = _setters.get("get_cb_caps")
    if get is None or not _gpu():
        return None
    return tuple(get())


def capture() -> dict:
    _prior.clear()
    state = {
        "env": dict(os.environ),
        "formatwarning": warnings.formatwarning,
        "showwarning": warnings.showwarning,
        "device": mx.default_device(),
        "streams": {},
        "set_wired_limit": mx.set_wired_limit,
        "cb_caps": cb_caps(),
    }
    devices = [mx.cpu] + ([mx.gpu] if _gpu() else [])
    for d in devices:
        state["streams"][d] = mx.default_stream(d)
    return state


def _restore_env(saved: dict) -> None:
    for k in list(os.environ):
        if k not in saved and k not in _ENV_SKIP:
            del os.environ[k]
    for k, v in saved.items():
        if k not in _ENV_SKIP and os.environ.get(k) != v:
            os.environ[k] = v


def restore(state: dict) -> None:
    _restore_env(state["env"])
    warnings.formatwarning = state["formatwarning"]
    warnings.showwarning = state["showwarning"]
    mx.set_wired_limit = state["set_wired_limit"]
    for name in _LIMITS:
        if name in _prior:
            _setters[name](_prior[name])
    if "rng" in _prior:
        hi, lo = _prior["rng"].tolist()
        _setters["seed"]((int(hi) << 32) | int(lo))
    caps = state["cb_caps"]
    if caps is not None and cb_caps() != caps:
        _setters["set_cb_caps"](*caps)
    if mx.default_device() != state["device"]:
        mx.set_default_device(state["device"])
    for d, stream in state["streams"].items():
        if mx.default_stream(d) != stream:
            mx.set_default_stream(stream)
    _prior.clear()
