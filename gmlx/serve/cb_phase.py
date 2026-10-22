"""Per-phase MLX command-buffer split caps.

Decode on a model in RAM wants coarse buffers: at the device defaults a
step's encode splits into enough command buffers that submission blocks
on in-flight drain, so the engine's async lookahead hides nothing. A
streamed decode is the other way round: it syncs on the host at every
layer to read the routing, so a layer's buffers all commit before the
sync whatever the caps, and finer buffers let the GPU start each layer
while the host still encodes it. Deep prefill needs fine buffers: a
giant buffer keeps every layer's transients live at once and can exhaust
GPU memory on multi-thousand-token chunks.

kq.set_cb_caps writes the live device fields, so the serve engine flips
at prompt steps (fine) and decode steps (decode_caps of the batch's
model), and `gmlx run` (its bench too) and `gmlx chat` wrap the
token-step generators the same way. GMLX_CB_PHASE=0 disables both
installs.
"""

from __future__ import annotations

import os

COARSE = (400, 100000)
FINE = (50, 50)  # mlx 0.31.2 device defaults, read back via get_cb_caps
# Decode caps of a model whose experts stream from the drive. On DeepSeek
# V4.1 Flash served streamed, (50, 50) and (100, 100) decoded 0.6 to 1.5
# ms per token faster than (200, 1000) and (400, 100000), output
# identical.
STREAMED_DECODE = FINE

_FLAG = "_gmlx_cb_phase"
_CAPS_ATTR = "_gmlx_decode_cb_caps"
_state: dict = {"phase": None, "caps": None, "kq": None, "lifted": False}


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


def decode_caps(model) -> tuple[int, int]:
    """The decode caps for ``model``: STREAMED_DECODE when its experts
    stream from the drive, COARSE otherwise. Read once per model."""
    if model is None:
        return COARSE
    caps = getattr(model, _CAPS_ATTR, None)
    if caps is not None:
        return caps
    from gmlx.stream.expert_streaming import experts_streamed

    caps = STREAMED_DECODE if experts_streamed(model) else COARSE
    try:
        object.__setattr__(model, _CAPS_ATTR, caps)
    except (AttributeError, TypeError):
        pass
    return caps


def _set(caps: tuple[int, int], ph: str | None) -> None:
    """Set ``caps`` on the device. Dedupes on the caps set, so a phase
    change that keeps the caps costs nothing and a model change that
    moves them under one phase still applies."""
    if _state["caps"] != caps:
        kq = _kq()
        if not kq:
            return
        kq.set_cb_caps(*caps)
        _state["caps"] = caps
    _state["phase"] = ph


def flip(phase: str, model=None) -> None:
    """Set the caps for ``phase``: FINE for prefill, decode_caps(model)
    for decode."""
    _set(decode_caps(model) if phase == "decode" else FINE, phase)


def lift() -> None:
    """Set the coarse caps on the live device for a streaming placement
    the argv lift did not cover (a config placement, or experts that
    stream for want of memory). Through _set, so the next flip sees them."""
    _set(COARSE, None)


def note_lift() -> None:
    """The CLI entry lifted the env caps for a streaming placement. An
    MLX_MAX_OPS_PER_BUFFER set before that lift is the user's pin."""
    _state["lifted"] = True


def _user_pinned() -> bool:
    return bool(os.environ.get("MLX_MAX_OPS_PER_BUFFER")) and not _state["lifted"]


def install_cb_phase_flips() -> bool:
    """Wrap the mlx-vlm engine's step entrypoints with phase flips."""
    if os.environ.get("GMLX_CB_PHASE", "1") == "0":
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


def phased_steps(gen, model=None, prefill: tuple[int, int] = FINE):
    """Iterate a token-step generator with ``prefill`` caps through its
    prefill and decode_caps(model) from the first token on. The generator
    body runs on the first next(), so the prefill flip happens here, not
    at call. Closing this generator closes the wrapped one at once, so its
    own cleanup, such as a speculative loop's mid-round cache rollback,
    runs before the caller goes on."""
    _set(prefill, "prefill")
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


def install_cb_phase_steps(streaming: bool = False) -> bool:
    """Wrap the generate_step functions the CLI generate path drives
    (mlx-lm's plain and speculative loops, mlx-vlm's ar loop, and gmlx's
    own speculative loop, stream_speculative) in phased_steps. A streaming
    placement prefills on its lifetime coarse caps (the argv lift); a
    user's MLX_MAX_OPS_PER_BUFFER pin skips the install."""
    if os.environ.get("GMLX_CB_PHASE", "1") == "0":
        return False
    if _user_pinned():
        return False
    if not _kq():
        return False
    import importlib

    from gmlx.upstream.patching import wrapped_with

    prefill = COARSE if streaming else FINE
    targets = [("gmlx._vendor.mlx_lm.generate", "generate_step"),
               ("gmlx._vendor.mlx_lm.generate", "speculative_generate_step"),
               ("gmlx._vendor.mlx_vlm.generate.ar", "generate_step"),
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
                                _model_arg(args, kwargs), prefill)

        setattr(wrapped, _FLAG, True)
        wrapped.__name__ = orig.__name__
        wrapped.__doc__ = orig.__doc__
        setattr(wrapped, "__wrapped__", orig)
        setattr(mod, name, wrapped)
    return True
