"""Process-wide state each test hands back as it found it.

CLI and library entry points that tests run change the whole process:
``gmlx distill`` and ``gmlx train`` turn MLX_ENABLE_TF32 off for their own
run, ``gmlx serve`` exports its settings, ``--stream-cpu`` lifts the MLX
command buffer caps, the umbrella replaces ``warnings.formatwarning``, and
library entry points set the MLX default device, seed the RNG, set the
memory, cache and wired limits and the mlx-kquant command buffer caps,
and rebind ``mx.set_wired_limit``. The caps come back with the phase and
the caps ``gmlx.serve.cb_phase`` last set, since a flip to the caps it
already holds sets nothing, and with its record of the CLI's caps lift.

Python-side copies of those settings come back with them: the
``wired_limit`` context in ``mlx_lm.generate`` and ``mlx_lm.utils``, which
the streaming and CPU device setups rebind, ``mx.device_info``, which
``--stream-cpu`` rebinds to the GPU's answer, the free-RAM read of the
copied ``apc``, which the APC manager build rebinds to the mach counters
(the simulated host reports those, the stock read sees the real one),
the render functions and input builders of the copied mlx-vlm, which
``load_vlm_model`` and the serve patches wrap (each test module hands
these back too, for the module fixtures that load a VLM), the limits the
serve governor saved when it armed its throttle, the cache limit the
server noted for the prefill decay, the table stream's latch and its
declared installs, the reading of the memory wired outside the process
that the first load takes for the wired cap, the MLX memory limit the
cap replaced, and the prefill decay's weight registries (untracked and
streamed bytes, their owners, the open build credits), which a build
leaves keyed to a model the next test never holds. A governor state
that outlives the test goes back to the saved limits it held when the
test started, and one the test created goes back to unarmed, as the MLX
limits go back to their values from before the test.

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
A class, module or session fixture sets up before the capture of any
test that uses it, so its setup and its teardown are recorded under the
fixture's name, and ``note_kq_env_left`` compares the end of the run with
the session start.
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
_WIRED_LIMIT_MODULES = ("gmlx._vendor.mlx_lm.generate", "gmlx._vendor.mlx_lm.utils")
# The module whose _free_ram_bytes the APC manager build rebinds, imported
# at session start for the same reason.
_APC_MODULE = "gmlx._vendor.mlx_vlm.apc"
# The copied mlx-vlm functions load_vlm_model and the serve patches wrap,
# as (module, name): the render functions, then the input builders the
# single-BOS fix wraps, in the modules of its _BINDINGS and
# _COUNT_BINDINGS. Their modules are imported at session start for the
# same reason.
_VLM_BINDINGS = (
    ("gmlx._vendor.mlx_vlm.prompt_utils", "get_chat_template"),
    ("gmlx._vendor.mlx_vlm.prompt_utils", "apply_chat_template"),
    ("gmlx._vendor.mlx_vlm.utils", "process_inputs"),
    ("gmlx._vendor.mlx_vlm.server.generation", "prepare_inputs"),
    ("gmlx._vendor.mlx_vlm.generate.dispatch", "prepare_inputs"),
    ("gmlx._vendor.mlx_vlm.server.anthropic", "prepare_inputs"),
    ("gmlx._vendor.mlx_vlm.server.openai", "prepare_inputs"),
)
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
    vlm = tuple(dict.fromkeys(m for m, _ in _VLM_BINDINGS))
    for name in _WIRED_LIMIT_MODULES + (_APC_MODULE,) + vlm:
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


# The cb_phase fields this module hands back, as its import leaves them.
_CB_PHASE_FRESH = {"phase": None, "caps": None, "lifted": False,
                   "pre_lift": None}


def _cb_phase_state():
    """cb_phase's state dict, or None before the module is imported."""
    mod = sys.modules.get("gmlx.serve.cb_phase")
    return None if mod is None else mod._state


def _prefill_decay():
    return sys.modules.get("gmlx.gen.prefill_decay")


def _table_stream():
    """gmlx.stream.table_stream, or None before it is imported."""
    return sys.modules.get("gmlx.stream.table_stream")


def _governor():
    """gmlx.serve.governor, or None before it is imported."""
    return sys.modules.get("gmlx.serve.governor")


def _capacity():
    """gmlx.serve.capacity, or None before it is imported."""
    return sys.modules.get("gmlx.serve.capacity")


def _wired_limit():
    """gmlx.stream.wired_limit, or None before it is imported."""
    return sys.modules.get("gmlx.stream.wired_limit")


# The prefill decay's weight registries: dicts handed back in place, so a
# reference a module took stays live.
_DECAY_REGISTRIES = ("_UNTRACKED_WEIGHTS", "_UNTRACKED_OWNERS",
                     "_STREAMED_TRACKED", "_STREAMED_CAP", "_BUILD_CREDIT")
_WIRED_LIMIT_FIELDS = {"_CAP_MEMORY_LIMIT": None, "_CLAMP_LOGGED": False,
                       "_ANNOUNCE": False}


# A governor state's throttle fields: the limits it saved and the ones it
# set in their place.
_THROTTLE_FIELDS = ("saved_mem_limit", "saved_cache_limit",
                    "throttle_mem_limit", "throttle_cache_limit")


def _throttle_fields(st) -> tuple:
    return tuple(getattr(st, name, None) for name in _THROTTLE_FIELDS)


def _decay_registries(decay) -> dict | None:
    """A copy of the weight registries, one level deep, and the weights
    owner, or None before the module is imported."""
    if decay is None:
        return None
    out = {name: {k: (set(v) if isinstance(v, set) else
                      dict(v) if isinstance(v, dict) else v)
                  for k, v in getattr(decay, name).items()}
           for name in _DECAY_REGISTRIES}
    out["_WEIGHTS_OWNER"] = decay._WEIGHTS_OWNER
    return out


def _put_back_decay_registries(decay, saved: dict | None) -> bool:
    """Hand the weight registries back as ``saved`` holds them (empty
    when the module was imported during the test). True when any had
    changed."""
    if decay is None:
        return False
    if saved is None:
        saved = {name: {} for name in _DECAY_REGISTRIES}
        saved["_WEIGHTS_OWNER"] = None
    changed = False
    with decay._UNTRACKED_LOCK:
        for name in _DECAY_REGISTRIES:
            reg = getattr(decay, name)
            if reg != saved[name]:
                changed = True
                reg.clear()
                reg.update(saved[name])
        if decay._WEIGHTS_OWNER is not saved["_WEIGHTS_OWNER"]:
            changed = True
            decay._WEIGHTS_OWNER = saved["_WEIGHTS_OWNER"]
    return changed


def vlm_bindings() -> list:
    """(module, name, function) for each of _VLM_BINDINGS in an imported
    module."""
    return [(mod, name, getattr(mod, name, _MISSING))
            for mod, name in ((sys.modules.get(m), n) for m, n in _VLM_BINDINGS)
            if mod is not None]


def put_back(bindings: list) -> list:
    """Rebind or remove each binding of ``bindings`` that changed. Returns
    the name of each."""
    changed = []
    for mod, name, fn in bindings:
        if getattr(mod, name, _MISSING) is fn:
            continue
        changed.append(f"{mod.__name__}.{name}")
        if fn is _MISSING:
            delattr(mod, name)
        else:
            setattr(mod, name, fn)
    return changed


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
        # Before the module is imported, the fields its first flip finds.
        "cb_phase": {k: (_cb_phase_state() or _CB_PHASE_FRESH)[k]
                     for k in _CB_PHASE_FRESH},
        "wired_limit": [
            (mod, getattr(mod, "wired_limit", _MISSING))
            for mod in map(sys.modules.get, _WIRED_LIMIT_MODULES) if mod],
        "free_ram_read": [
            (mod, getattr(mod, "_free_ram_bytes", _MISSING))
            for mod in [sys.modules.get(_APC_MODULE)] if mod],
        "vlm_bindings": vlm_bindings(),
        "governor": {st: _throttle_fields(st) for st in _governor_states},
        "throttle_holders": list(getattr(_governor(), "_HOLDERS", ())),
        "noted_cache_limit": getattr(decay, "_NOTED_CACHE_LIMIT", None),
        "table_stream_latch": getattr(_table_stream(), "_FORCED", False),
        "table_install_follows": set(getattr(_table_stream(),
                                             "_INSTALL_FOLLOWS", ())),
        "armed_floor": getattr(_governor(), "_ARMED_FLOOR", 0.0),
        "outside_wired": getattr(_capacity(), "_OUTSIDE_WIRED", None),
        "cap_engaged": getattr(_capacity(), "_CAP_ENGAGED", False),
        "cap_planned": set(getattr(_capacity(), "_CAP_PLANNED", ())),
        "wired_limit_fields": {
            name: getattr(_wired_limit(), name)
            for name in _WIRED_LIMIT_FIELDS if _wired_limit() is not None},
        "decay_registries": _decay_registries(decay),
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


def note_kq_env_left() -> None:
    """Record each KQ_* key outside _KQ_READ_EACH_CALL that differs from
    the session start at the end of the run, when no test or fixture has
    been named for it."""
    named = {k for _, k in KQ_TOUCHED}
    before = len(KQ_TOUCHED)
    note_kq_env(SESSION, "the end of the run")
    KQ_TOUCHED[before:] = [t for t in KQ_TOUCHED[before:]
                           if t[1] not in named]


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
    fields = state["cb_phase"]
    if phase is not None and any(phase[k] != v for k, v in fields.items()):
        changed.append("cb_phase")
        phase.update(fields)
    for mod, fn in state["wired_limit"]:
        if getattr(mod, "wired_limit", _MISSING) is fn:
            continue
        changed.append(f"{mod.__name__}.wired_limit")
        if fn is _MISSING:
            del mod.wired_limit
        else:
            mod.wired_limit = fn
    for mod, fn in state["free_ram_read"]:
        if getattr(mod, "_free_ram_bytes", _MISSING) is not fn:
            changed.append(f"{mod.__name__}._free_ram_bytes")
            mod._free_ram_bytes = fn
    changed += put_back(state["vlm_bindings"])
    governor = False
    for st in list(_governor_states):
        saved = state["governor"].get(st, (None,) * len(_THROTTLE_FIELDS))
        if _throttle_fields(st) != saved:
            governor = True
            for name, value in zip(_THROTTLE_FIELDS, saved):
                setattr(st, name, value)
    if governor:
        changed.append("governor saved limits")
    gov = _governor()
    if gov is not None and gov._HOLDERS != state["throttle_holders"]:
        changed.append("governor throttle holders")
        gov._HOLDERS[:] = state["throttle_holders"]
    if gov is not None and gov._ARMED_FLOOR != state["armed_floor"]:
        changed.append("governor armed floor")
        gov._ARMED_FLOOR = state["armed_floor"]
    decay = _prefill_decay()
    if decay is not None and \
            decay._NOTED_CACHE_LIMIT != state["noted_cache_limit"]:
        changed.append("noted_cache_limit")
        decay._NOTED_CACHE_LIMIT = state["noted_cache_limit"]
    if mx.default_device() != state["device"]:
        changed.append("device")
        mx.set_default_device(state["device"])
    tables = _table_stream()
    if tables is not None and tables._FORCED != state["table_stream_latch"]:
        changed.append("table stream latch")
        tables._FORCED = state["table_stream_latch"]
    if tables is not None and \
            tables._INSTALL_FOLLOWS != state["table_install_follows"]:
        changed.append("table install declarations")
        tables._INSTALL_FOLLOWS.clear()
        tables._INSTALL_FOLLOWS.update(state["table_install_follows"])
    capacity = _capacity()
    if capacity is not None and \
            capacity._OUTSIDE_WIRED != state["outside_wired"]:
        changed.append("outside wired reading")
        capacity._OUTSIDE_WIRED = state["outside_wired"]
    if capacity is not None and \
            capacity._CAP_ENGAGED != state["cap_engaged"]:
        changed.append("wired cap latch")
        capacity._CAP_ENGAGED = state["cap_engaged"]
    if capacity is not None and \
            capacity._CAP_PLANNED != state["cap_planned"]:
        changed.append("wired cap plans")
        capacity._CAP_PLANNED.clear()
        capacity._CAP_PLANNED.update(state["cap_planned"])
    wired = _wired_limit()
    if wired is not None:
        for name, fresh in _WIRED_LIMIT_FIELDS.items():
            want = state["wired_limit_fields"].get(name, fresh)
            if getattr(wired, name) != want:
                changed.append(f"wired_limit.{name}")
                setattr(wired, name, want)
    if _put_back_decay_registries(decay, state["decay_registries"]):
        changed.append("prefill decay weight registries")
    for d, stream in state["streams"].items():
        if mx.default_stream(d) != stream:
            changed.append(f"stream {getattr(d, 'name', d)}")
            mx.set_default_stream(stream)
    _prior.clear()
    return changed
