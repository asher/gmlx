"""Student loading for train and eval: a GGUF K-quant base with an optional
adapter, or an MLX checkpoint, behind one interface the trainer and the
evaluator share."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .format import read_json
from .tokens import llamacpp_bos_default

# ---------------------------------------------------------------------------
# student helpers
# ---------------------------------------------------------------------------

class adapter_disabled:
    """Context manager: LoRA contribution off in process (scale 0), the
    untouched-student reference without a second load."""

    def __init__(self, model):
        self.model = model
        self.saved: list[tuple[Any, float]] = []

    def __enter__(self):
        from mlx_lm.tuner.lora import LoRALinear

        def off(_k, m):
            if isinstance(m, LoRALinear) or hasattr(m, "lora_a") and hasattr(m, "scale"):
                self.saved.append((m, m.scale))
                m.scale = 0.0
                self._drop_tables(m)
        self.model.apply_to_modules(off)
        return self

    def __exit__(self, *exc):
        for m, s in self.saved:
            m.scale = s
            self._drop_tables(m)
        self.saved.clear()
        return False

    @staticmethod
    def _drop_tables(m):
        # gmlx's live LoRA module folds the scale into per-dtype factor
        # tables on first use; a scale change is invisible until they go
        t = getattr(m, "_kq_tables", None)
        if isinstance(t, dict):
            t.clear()


def load_gguf_student(gguf_path: str, adapter: str | None = None, hf_source: str | None = None):
    """(model, config, tokenizer) from a GGUF base plus an optional GGUF
    LoRA adapter applied live."""
    import gmlx.load.loadlog as loadlog
    from gmlx.load.loader import load_model
    with loadlog.load_ui(False, gguf_path):
        model, config, tokenizer = load_model(gguf_path, hf_source=hf_source)
    llamacpp_bos_default(tokenizer, gguf_path)
    if adapter:
        from gmlx.load.adapter import apply_gguf_adapter
        from gmlx.load.preflight import preflight
        raw = getattr(model, "language_model", model)
        apply_gguf_adapter(raw, config, adapter, base_arch=preflight(gguf_path).arch)
    return model, config, tokenizer


def load_mlx_student(path: str, adapter_path: str | None = None):
    from mlx_lm import load
    model, tokenizer = load(path, adapter_path=adapter_path)[:2]
    cfg = read_json(Path(path) / "config.json")
    return model, cfg, tokenizer


def trunk_hidden(model, inputs):
    """Final-norm hidden states [B, T, d] for an mlx-lm style model."""
    inner = getattr(model, "model", model)
    return inner(inputs)


