"""Process-wide state each test hands back as it found it.

CLI and library entry points that tests run change the whole process:
``gmlx distill`` and ``gmlx train`` turn MLX_ENABLE_TF32 off for their own
run, ``gmlx serve`` exports its settings, ``--stream-cpu`` lifts the MLX
command buffer caps, the umbrella replaces ``warnings.formatwarning``, and
library entry points set the MLX default device, seed the RNG, set the
memory, cache and wired limits and the mlx-kquant command buffer caps,
and rebind ``mx.set_wired_limit``. The caps come back with the phase
``gmlx.serve.cb_phase`` last flipped to, since a flip to the phase it
already names sets nothing.

Python-side copies of those settings come back with them: the
``wired_limit`` context in ``mlx_lm.generate`` and ``mlx_lm.utils``, which
the streaming and CPU device setups rebind, ``mx.device_info``, which
``--stream-cpu`` rebinds to the GPU's answer, the limits the serve
governor saved when it armed its throttle, and the cache limit the server
noted for the prefill decay. A governor state that outlives the test goes
back to the saved limits it held when the test started, and one the test
created goes back to unarmed, as the MLX limits go back to their values
from before the test.

Known exclusion: the rest of a governor state (its band, the halved
prefill step and the clamped width) stays as the test left it. The next
tick in a band that throttles arms the limits again, and the next tick in
the green band undoes the rest.

``capture`` and ``restore`` bracket every test (tests/conftest.py). MLX
reads MLX_ENABLE_TF32 and the command buffer caps once, at the first
matmul and the first command buffer, so ``start_session`` also runs one
small GPU matmul: the process keeps the session's settings even when a
test changes the environment before its own first matmul.

mlx-kquant reads most of its KQ_* keys once too, each at the first call
of the kernel route that reads it, and there is no one call that reaches
them all. ``note_kq_env`` records instead each such key a test sets,
changes or removes, and the end-of-run check fails on any: that test
decided the key for the rest of the run, or found it already decided.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
import weakref

import mlx.core as mx

SESSION: dict = {}

# First value each setter replaced in the current test, and the RNG key
# before the test's first seed.
_prior: dict = {}
_setters: dict = {}

_LIMITS = ("set_memory_limit", "set_cache_limit", "set_wired_limit")
# pytest sets it per phase and removes it after each one.
_ENV_SKIP = ("PYTEST_CURRENT_TEST",)
# The modules whose wired_limit context gmlx rebinds. Imported at session
# start, so restore holds the stock context even when the test that first
# imports one also rebinds it.
_WIRED_LIMIT_MODULES = ("mlx_lm.generate", "mlx_lm.utils")
_MISSING = object()
# Every serve governor state the session creates.
_governor_states: weakref.WeakSet = weakref.WeakSet()
# KQ_* keys mlx-kquant reads at each call. A key it reads once and keeps
# is any other, so a key a new test sets fails the end-of-run check until
# the kq source shows it is read at each call and it joins this set.
_KQ_READ_EACH_CALL = frozenset({"KQ_GQA_NAX", "KQ_SWITCH_GEMM_MIN_ROWS"})
# (test node id, key) for each other KQ_* key a test set, changed or
# removed.
KQ_TOUCHED: list = []


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


def _track_governor_states() -> None:
    from gmlx.serve import governor

    cls = governor._GovState
    init = cls.__init__
    if getattr(init, "_gmlx_test_recorder", None) is not None:
        return

    def tracked(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _governor_states.add(self)

    tracked._gmlx_test_recorder = init
    cls.__init__ = tracked


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
    for name in _WIRED_LIMIT_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    _track_governor_states()
    # The run keeps the session's MLX_ENABLE_TF32, TF32 on unless the
    # session sets it to 0. An in-process ``gmlx train`` or ``gmlx distill``
    # test runs under it too: the CLI sets MLX_ENABLE_TF32=0 after this
    # read, which changes nothing. Such a test checks numerics TF32 keeps,
    # and a test that needs exact float32 runs the CLI in a child process.
    if _gpu():
        x = mx.ones((32, 32))
        mx.eval(mx.matmul(x, x, stream=mx.gpu))


def cb_caps():
    """The MLX command buffer caps mlx-kquant sets, or None without them."""
    get = _setters.get("get_cb_caps")
    if get is None or not _gpu():
        return None
    return tuple(get())


def _cb_phase_state():
    """cb_phase's state dict, or None before the module is imported."""
    mod = sys.modules.get("gmlx.serve.cb_phase")
    return None if mod is None else mod._state


def _prefill_decay():
    return sys.modules.get("gmlx.gen.prefill_decay")


def capture() -> dict:
    _prior.clear()
    decay = _prefill_decay()
    state = {
        "env": dict(os.environ),
        "formatwarning": warnings.formatwarning,
        "showwarning": warnings.showwarning,
        "device": mx.default_device(),
        "streams": {},
        "set_wired_limit": mx.set_wired_limit,
        "device_info": mx.device_info,
        "cb_caps": cb_caps(),
        # None before the module is imported, as its first flip finds it.
        "cb_phase": (_cb_phase_state() or {}).get("phase"),
        "wired_limit": [
            (mod, getattr(mod, "wired_limit", _MISSING))
            for mod in map(sys.modules.get, _WIRED_LIMIT_MODULES) if mod],
        "governor": {st: (st.saved_mem_limit, st.saved_cache_limit)
                     for st in _governor_states},
        "noted_cache_limit": getattr(decay, "_NOTED_CACHE_LIMIT", None),
    }
    devices = [mx.cpu] + ([mx.gpu] if _gpu() else [])
    for d in devices:
        state["streams"][d] = mx.default_stream(d)
    return state


def note_kq_env(state: dict, nodeid: str) -> None:
    """Record each KQ_* key outside _KQ_READ_EACH_CALL whose value now
    differs from the one ``state`` captured."""
    saved = state["env"]
    for k in sorted(set(saved) | set(os.environ)):
        if (k.startswith("KQ_") and k not in _KQ_READ_EACH_CALL
                and saved.get(k) != os.environ.get(k)
                and (nodeid, k) not in KQ_TOUCHED):
            KQ_TOUCHED.append((nodeid, k))


def _restore_env(saved: dict, changed: list) -> None:
    for k in list(os.environ):
        if k not in saved and k not in _ENV_SKIP:
            changed.append(f"env {k}")
            del os.environ[k]
    for k, v in saved.items():
        if k not in _ENV_SKIP and os.environ.get(k) != v:
            changed.append(f"env {k}")
            os.environ[k] = v


def restore(state: dict) -> list:
    """Put back what ``state`` captured. Returns the name of each field
    that had changed."""
    changed: list = []
    _restore_env(state["env"], changed)
    for name, owner in (("formatwarning", warnings),
                        ("showwarning", warnings),
                        ("set_wired_limit", mx), ("device_info", mx)):
        if getattr(owner, name) is not state[name]:
            changed.append(f"{owner.__name__}.{name}")
            setattr(owner, name, state[name])
    for name in _LIMITS:
        if name in _prior and _setters[name](_prior[name]) != _prior[name]:
            changed.append(name.removeprefix("set_").replace("_", " "))
    if "rng" in _prior:
        changed.append("rng")
        hi, lo = _prior["rng"].tolist()
        _setters["seed"]((int(hi) << 32) | int(lo))
    caps = state["cb_caps"]
    if caps is not None and cb_caps() != caps:
        changed.append("cb_caps")
        _setters["set_cb_caps"](*caps)
    phase = _cb_phase_state()
    if phase is not None and phase["phase"] != state["cb_phase"]:
        changed.append("cb_phase")
        phase["phase"] = state["cb_phase"]
    for mod, fn in state["wired_limit"]:
        if getattr(mod, "wired_limit", _MISSING) is fn:
            continue
        changed.append(f"{mod.__name__}.wired_limit")
        if fn is _MISSING:
            del mod.wired_limit
        else:
            mod.wired_limit = fn
    governor = False
    for st in list(_governor_states):
        saved = state["governor"].get(st, (None, None))
        if (st.saved_mem_limit, st.saved_cache_limit) != saved:
            governor = True
            st.saved_mem_limit, st.saved_cache_limit = saved
    if governor:
        changed.append("governor saved limits")
    decay = _prefill_decay()
    if decay is not None and \
            decay._NOTED_CACHE_LIMIT != state["noted_cache_limit"]:
        changed.append("noted_cache_limit")
        decay._NOTED_CACHE_LIMIT = state["noted_cache_limit"]
    if mx.default_device() != state["device"]:
        changed.append("device")
        mx.set_default_device(state["device"])
    for d, stream in state["streams"].items():
        if mx.default_stream(d) != stream:
            changed.append(f"stream {getattr(d, 'name', d)}")
            mx.set_default_stream(stream)
    _prior.clear()
    return changed
