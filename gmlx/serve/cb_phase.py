"""Per-phase MLX command-buffer split caps.

Decode on a model in RAM runs on RAM_DECODE, 400 ops per buffer with no
size cap: at the device's 50 MB cap a step's encode splits into enough
command buffers that submission blocks on in-flight drain, so the engine's
async lookahead hides nothing. GMLX_CB_FINE_DECODE=1 decodes on
FINE_DECODE, 50 ops with no size cap, instead. Deep prefill in RAM needs
fine buffers: a giant buffer keeps every layer's transients live at once
and can exhaust GPU memory on multi-thousand-token chunks. A streamed
model decodes the other way round. Its decode syncs on
the host at every layer to read the routing, so a layer's buffers all
commit before the sync whatever the caps, and finer buffers let the GPU
start each layer while the host still encodes it. It prefills on the fine
caps too: a coarse buffer keeps a finished layer's buffers in flight
while the next layer allocates.

kq.set_cb_caps writes the live device fields, so the serve engine flips
at prompt steps (prefill_caps of the batch's model) and decode steps
(decode_caps), and `gmlx run` (its bench too) and `gmlx chat` wrap the
token-step generators the same way. GMLX_CB_PHASE=0, or the user's own
MLX_MAX_OPS_PER_BUFFER or MLX_MAX_MB_PER_BUFFER, keeps the caps as they
are in every phase. The engines of a residency pool share the device
fields, so a model in RAM and a streamed one decoding at once switch the
caps at each step.

The CLI lifts the caps to COARSE for the load of a streamed model, and
the streamed install ends the lift (end_lift), so no phase of a streamed
model runs on the lift, with GMLX_CB_PHASE=0 too.
"""

from __future__ import annotations

import os
import threading
import weakref

COARSE = (400, 100000)
FINE = (50, 50)  # the MLX device defaults on a Max or Ultra chip
# Decode caps of a model in RAM. FINE_DECODE, 50 ops with no size cap,
# decodes within spread of COARSE on M3 Max and M5 Max and 4 to 8 pct
# slower on M1 Ultra, so every chip decodes on COARSE.
RAM_DECODE = COARSE
FINE_DECODE = (50, 100000)
# Decode caps of a model whose experts stream from the drive. On DeepSeek
# V4.1 Flash served streamed, (50, 50) decoded 0.63 ms per token faster
# than (400, 100000), output identical.
STREAMED_DECODE = FINE
# Prefill caps of a model whose experts stream from the drive. On DeepSeek
# V4.1 Flash served streamed, an 8k prompt on (400, 100000) wired 13 GiB
# within a quarter second and left 5.4 GB of pages free.
STREAMED_PREFILL = FINE

_FLAG = "_gmlx_cb_phase"
_STREAMED_ATTR = "_gmlx_cb_streamed"
_ENV_CAPS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")
# pre_lift: the caps end_lift sets back, while a lift holds.
_UNSET = object()
# model: a weak reference to the model the caps were last set for, None
# for a flip without a model, _UNSET when the last set had no model.
_state: dict = {"phase": None, "caps": None, "kq": None, "lifted": False,
                "pre_lift": None, "model": _UNSET}
# Engines on separate threads share the device fields: the compare, the
# write and the record go together, so the record matches the device.
# flip's repeat check reads phase and model without the lock: a torn
# pair at worst skips one flip the other engine is overriding anyway.
_LOCK = threading.Lock()


def _kq():
    if _state["kq"] is None:
        try:
            import mlx_kquant as kq
            _state["kq"] = kq if hasattr(kq, "set_cb_caps") else False
        except ImportError:
            _state["kq"] = False
    return _state["kq"]


def snapshot() -> tuple:
    """The phase and caps set now, for :func:`restore`."""
    return _state["phase"], _state["caps"]


def restore(saved: tuple) -> None:
    """Set the caps a :func:`snapshot` recorded (nothing before a flip)."""
    ph, caps = saved
    if caps is not None:
        _set(caps, ph)


def _streamed(model) -> bool:
    """The install's streaming verdict for ``model``, read once per model."""
    if model is None:
        return False
    streamed = getattr(model, _STREAMED_ATTR, None)
    if streamed is not None:
        return streamed
    from gmlx.stream.expert_streaming import experts_streamed

    streamed = bool(experts_streamed(model))
    try:
        object.__setattr__(model, _STREAMED_ATTR, streamed)
    except (AttributeError, TypeError):
        pass
    return streamed


def decode_caps(model) -> tuple[int, int]:
    """The decode caps for ``model``: STREAMED_DECODE when its experts
    stream from the drive, RAM_DECODE otherwise (FINE_DECODE under
    GMLX_CB_FINE_DECODE=1)."""
    if _streamed(model):
        return STREAMED_DECODE
    return FINE_DECODE if os.environ.get("GMLX_CB_FINE_DECODE") == "1" else RAM_DECODE


def prefill_caps(model) -> tuple[int, int]:
    """The prefill caps for ``model``: STREAMED_PREFILL when its experts
    stream from the drive, FINE otherwise."""
    return STREAMED_PREFILL if _streamed(model) else FINE


def _ref(model):
    """A weak reference to ``model``, None for no model, _UNSET for an
    object that takes no weak reference (no repeat shortcut for it)."""
    if model is None:
        return None
    try:
        return weakref.ref(model)
    except TypeError:
        return _UNSET


def _same_model(model) -> bool:
    """``model`` is the model the caps were last set for."""
    ref = _state["model"]
    if ref is _UNSET:
        return False
    if ref is None:
        return model is None
    return model is not None and ref() is model


def _set(caps: tuple[int, int], ph: str | None,
         model=_UNSET) -> tuple[int, int] | None:
    """Set ``caps`` on the device. Dedupes on the caps set, so a phase
    change that keeps the caps costs nothing and a model change that
    moves them under one phase still applies. Records the phase and the
    model it was set for. Returns the caps the device held before, None
    when they are not known."""
    with _LOCK:
        prev = _state["caps"]
        if prev != caps:
            kq = _kq()
            if not kq:
                return None
            held = kq.set_cb_caps(*caps)
            if held is not None:
                prev = (int(held[0]), int(held[1]))
            _state["caps"] = caps
        _state["phase"] = ph
        _state["model"] = _UNSET if model is _UNSET else _ref(model)
        return prev


def env_caps_set() -> bool:
    """MLX_MAX_OPS_PER_BUFFER or MLX_MAX_MB_PER_BUFFER is set, by the
    user or by the CLI's streaming lift."""
    return any(os.environ.get(k) for k in _ENV_CAPS)


def _user_pinned() -> bool:
    return env_caps_set() and not _state["lifted"]


def _live() -> bool:
    """The phase caps apply: GMLX_CB_PHASE is not 0 and the user pinned
    no caps."""
    return os.environ.get("GMLX_CB_PHASE", "1") != "0" and not _user_pinned()


def flip(phase: str, model=None) -> None:
    """Set the caps for ``phase``: prefill_caps(model) for prefill,
    decode_caps(model) for decode. A repeat of the last phase for the
    last model returns at once: the engine flips at every step."""
    if _state["phase"] == phase and _same_model(model):
        return
    if not _live():
        return
    caps = decode_caps(model) if phase == "decode" else prefill_caps(model)
    _set(caps, phase, model)


def lift() -> None:
    """Set the coarse caps on the live device for the load of a streaming
    placement from config, which the argv lift does not cover. Through
    _set, so the next flip sees them. The caps the device held before are
    the ones end_lift sets back."""
    prev = _set(COARSE, None)
    if _state["pre_lift"] is None:
        _state["pre_lift"] = prev


def note_lift() -> None:
    """The CLI entry lifted the env caps for a streaming placement, with
    neither cap set before it. A cap set before that lift is the user's
    pin. MLX latches the lift at device init, so no caps from before it
    exist: end_lift sets FINE, the caps of every phase of a streamed model
    and the device defaults on a Max or Ultra chip."""
    _state["lifted"] = True
    _state["pre_lift"] = FINE


def end_lift() -> None:
    """End the lift at the end of a streamed install: set back the caps
    from before it. Nothing when no lift holds, as under the user's pin."""
    pre, _state["pre_lift"] = _state["pre_lift"], None
    if pre is not None:
        _set(pre, None)


def install_cb_phase_flips() -> bool:
    """Wrap the mlx-vlm engine's step entrypoints with phase flips."""
    if not _live():
        return False
    if not _kq():
        return False
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.upstream.patching import link, wrapped_with

    if wrapped_with(ar.GenerationBatch._step, _FLAG):
        return True

    def _wrap(cls, name, phase):
        orig = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            flip(phase, getattr(self, "model", None))
            return orig(self, *args, **kwargs)

        wrapped.__name__ = orig.__name__
        setattr(cls, name, link(wrapped, orig, _FLAG))

    _wrap(ar.GenerationBatch, "_step", "decode")
    _wrap(ar.SpeculativeGenerationBatch, "next", "decode")
    _wrap(ar.PromptProcessingBatch, "prompt_step", "prefill")
    # generate() runs the last prompt columns, and the whole prefill of a
    # prompt that never reaches prompt_step.
    _wrap(ar.PromptProcessingBatch, "generate", "prefill")
    return True


def phased_steps(gen, model=None):
    """Iterate a token-step generator on prefill_caps(model) through its
    prefill and decode_caps(model) from the first token on. The generator
    body runs on the first next(), so the prefill flip happens here, not
    at call. Closing this generator closes the wrapped one at once, so its
    own cleanup, such as a speculative loop's mid-round cache rollback,
    runs before the caller goes on."""
    flip("prefill", model)
    first = True
    try:
        for item in gen:
            if first:
                flip("decode", model)
                first = False
            yield item
    finally:
        close = getattr(gen, "close", None)
        if close is not None:
            close()


def _model_arg(args, kwargs):
    """The model a generate_step call drives: its first module argument
    (mlx-lm and mlx-vlm take the prompt first, gmlx's loop the model)."""
    import mlx.nn as nn

    for a in args:
        if isinstance(a, nn.Module):
            return a
    return kwargs.get("model")


def install_cb_phase_steps() -> bool:
    """Wrap the generate_step functions the CLI generate path drives
    (mlx-lm's plain and speculative loops, mlx-vlm's ar loop and the
    binding of it that stream_generate calls, and gmlx's own speculative
    loop, stream_speculative) in phased_steps."""
    if not _live():
        return False
    if not _kq():
        return False
    import importlib

    from gmlx.upstream.patching import wrapped_with

    targets = [("gmlx._vendor.mlx_lm.generate", "generate_step"),
               ("gmlx._vendor.mlx_lm.generate", "speculative_generate_step"),
               ("gmlx._vendor.mlx_vlm.generate.ar", "generate_step"),
               ("gmlx._vendor.mlx_vlm.generate.dispatch", "generate_step"),
               ("gmlx.spec.speculative", "stream_speculative")]
    for modname, name in targets:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        orig = getattr(mod, name, None)
        if orig is None or wrapped_with(orig, _FLAG):
            continue

        def wrapped(*args, _orig=orig, **kwargs):
            return phased_steps(_orig(*args, **kwargs),
                                _model_arg(args, kwargs))

        setattr(wrapped, _FLAG, True)
        wrapped.__name__ = orig.__name__
        wrapped.__doc__ = orig.__doc__
        setattr(wrapped, "__wrapped__", orig)
        setattr(mod, name, wrapped)
    return True
