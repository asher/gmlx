"""Load an MLX safetensors sentence encoder and its tokenizer.

A checkpoint is a folder (or a Hugging Face repo id, resolved through the
local cache) with ``config.json``, ``*.safetensors`` and the tokenizer files.
Quantized checkpoints carry a ``quantization`` block in ``config.json``; the
layers that have ``scales`` in the weights are quantized to match. The
parameter names are the checkpoint's own, so the mlx-community conversions
load unchanged.

The sentence pooling comes from the sentence-transformers ``Pooling`` config
(``modules.json`` and its ``config.json``) when the folder has one, and
otherwise from the model type's default.
"""

from __future__ import annotations

import glob
import json
import os
from types import ModuleType

import mlx.core as mx
import mlx.nn as nn

from . import bert, gemma_embedding, modernbert, qwen3
from .base import POOLING_MODES

# config.json model_type -> encoder module.
MODEL_TYPES: dict[str, ModuleType] = {
    "gemma3_text": gemma_embedding,
    "bert": bert,
    "xlm-roberta": bert,
    "xlm_roberta": bert,
    "modernbert": modernbert,
    "qwen3": qwen3,
}

# Pooling when the checkpoint ships no sentence-transformers Pooling config
# (the mlx-community conversions ship modules.json without it). The
# catalogued XLM-RoBERTa embedders, BGE-M3 and Arctic-Embed-L v2.0, pool the
# first token in their upstream Pooling configs. EmbeddingGemma, the Nomic
# ModernBERT embedder and most BERT sentence embedders take the mean.
# Qwen3-Embedding pools the last token.
DEFAULT_POOLING = {
    "gemma3_text": "mean",
    "bert": "mean",
    "xlm-roberta": "cls",
    "xlm_roberta": "cls",
    "modernbert": "mean",
    "qwen3": "last",
}


class UnsupportedEncoderError(ValueError):
    """The checkpoint's model_type has no gmlx encoder."""


def resolve_path(model_ref: str) -> str:
    """A local folder as-is, else the local snapshot of a Hub repo id."""
    expanded = os.path.expanduser(model_ref)
    if os.path.isdir(expanded):
        return expanded
    from huggingface_hub import snapshot_download

    return snapshot_download(model_ref)


def _read_json(path: str) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def pooling_mode(path: str, model_type: str) -> str:
    """The sentence pooling for the checkpoint at ``path``."""
    modules = _read_json(os.path.join(path, "modules.json")) or []
    for m in modules if isinstance(modules, list) else []:
        if not str(m.get("type", "")).endswith(".Pooling"):
            continue
        cfg = _read_json(os.path.join(path, m.get("path", ""), "config.json"))
        if not isinstance(cfg, dict):
            break
        if cfg.get("pooling_mode_cls_token"):
            return "cls"
        if cfg.get("pooling_mode_mean_tokens"):
            return "mean"
        if cfg.get("pooling_mode_lasttoken"):
            return "last"
        enabled = sorted(k for k, v in cfg.items()
                         if k.startswith("pooling_mode_") and v is True)
        raise UnsupportedEncoderError(
            f"{path}: sentence pooling {enabled} is not supported "
            f"(supported: {', '.join(POOLING_MODES)})")
    return DEFAULT_POOLING[model_type]


def _load_weights(path: str) -> dict[str, mx.array]:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    weights: dict[str, mx.array] = {}
    for f in files:
        weights.update(mx.load(f))
    return weights


def _dense_dims(weights: dict) -> list[int]:
    dims = []
    while f"dense.{len(dims)}.weight" in weights:
        dims.append(int(weights[f"dense.{len(dims)}.weight"].shape[0]))
    return dims


def _quantize(model: nn.Module, config: dict, weights: dict) -> None:
    """Quantize the layers the checkpoint stores quantized. A per-layer
    entry in the ``quantization`` block overrides the global bits and group
    size for that layer."""
    quant = config.get("quantization") or config.get("quantization_config")
    if not quant:
        return

    def predicate(path: str, module: nn.Module):
        if f"{path}.scales" not in weights:
            return False
        override = quant.get(path)
        return override if isinstance(override, dict) else True

    nn.quantize(model, group_size=quant.get("group_size", 64),
                bits=quant.get("bits", 4), mode=quant.get("mode", "affine"),
                class_predicate=predicate)


def build(path: str) -> nn.Module:
    """Build the encoder at ``path`` and load its weights."""
    config = _read_json(os.path.join(path, "config.json"))
    if not isinstance(config, dict):
        raise FileNotFoundError(f"no config.json in {path}")
    model_type = config.get("model_type")
    module = MODEL_TYPES.get(str(model_type))
    if module is None:
        raise UnsupportedEncoderError(
            f"{path}: model_type {model_type!r} has no gmlx encoder "
            f"(supported: {', '.join(sorted(MODEL_TYPES))})")
    weights = module.Model.sanitize(_load_weights(path))
    args = dict(config, pooling=pooling_mode(path, str(model_type)))
    if module is gemma_embedding:
        args["dense_dims"] = _dense_dims(weights)
    model = module.Model(module.ModelArgs.from_dict(args))
    _quantize(model, config, weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def load(model_ref: str):
    """``(model, tokenizer)`` for an encoder folder or Hub repo id."""
    from transformers import AutoTokenizer

    path = resolve_path(model_ref)
    model = build(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
