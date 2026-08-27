"""Per-phase MLX command-buffer split caps for serving.

Decode wants coarse buffers: at the device defaults a step's encode
splits into enough command buffers that submission blocks on in-flight
drain, so the engine's async lookahead hides nothing. Deep prefill
needs fine buffers: a giant buffer keeps every layer's transients live
at once and can exhaust GPU memory on multi-thousand-token chunks.
kq.set_cb_caps writes the live device fields, so the serve engine
flips coarse at decode steps and fine at prompt steps. GMLX_CB_PHASE=0
disables the install.
"""

from __future__ import annotations

import os

COARSE = (400, 100000)
FINE = (50, 50)  # mlx 0.31.2 device defaults, read back via get_cb_caps

_state: dict = {"phase": None, "kq": None}


def _kq():
    if _state["kq"] is None:
        try:
            import mlx_kquant as kq
            _state["kq"] = kq if hasattr(kq, "set_cb_caps") else False
        except ImportError:
            _state["kq"] = False
    return _state["kq"]


def flip(phase: str) -> None:
    """Set the caps for ``phase`` (decode or prefill). Dedupes repeats."""
    if _state["phase"] == phase:
        return
    kq = _kq()
    if not kq:
        return
    kq.set_cb_caps(*(COARSE if phase == "decode" else FINE))
    _state["phase"] = phase


def install_cb_phase_flips() -> bool:
    """Wrap the mlx-vlm engine's step entrypoints with phase flips."""
    if os.environ.get("GMLX_CB_PHASE", "1") == "0":
        return False
    if not _kq():
        return False
    from mlx_vlm.generate import ar

    if getattr(ar.GenerationBatch._step, "_gmlx_cb_phase", False):
        return True

    def _wrap(cls, name, phase):
        orig = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            flip(phase)
            return orig(self, *args, **kwargs)

        wrapped._gmlx_cb_phase = True
        wrapped.__name__ = orig.__name__
        setattr(cls, name, wrapped)

    _wrap(ar.GenerationBatch, "_step", "decode")
    _wrap(ar.SpeculativeGenerationBatch, "next", "decode")
    _wrap(ar.PromptProcessingBatch, "prompt_step", "prefill")
    # Prompts short enough to skip chunked processing never reach
    # prompt_step: generate() runs their whole prefill in one call.
    _wrap(ar.PromptProcessingBatch, "generate", "prefill")
    return True
