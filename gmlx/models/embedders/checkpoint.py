"""Load an MLX safetensors sentence encoder and its tokenizer.

A checkpoint is a folder (or a Hugging Face repo id, resolved through the
local cache) with ``config.json``, ``*.safetensors`` and the tokenizer files.
Quantized checkpoints carry a ``quantization`` block in ``config.json``; the
layers that have ``scales`` in the weights are quantized to match. The
parameter names are the checkpoint's own, so the mlx-community conversions
load unchanged.

The sentence pooling comes from the sentence-transformers ``Pooling`` config
(``modules.json`` and its ``config.json``) when the folder has one. Without
one, a checkpoint of a known first-token family pools the first token, and
any other takes its model type's default.
"""

from __future__ import annotations

import glob
import json
import os
import posixpath
import re
import sys
from types import ModuleType

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from gmlx.device import cpu_default_device
from gmlx.load.dtypes import activation_dtype

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

# The model family each model_type serves, for messages.
FAMILIES = {
    "gemma3_text": "EmbeddingGemma",
    "bert": "BERT",
    "xlm-roberta": "XLM-RoBERTa",
    "xlm_roberta": "XLM-RoBERTa",
    "modernbert": "ModernBERT",
    "qwen3": "Qwen3-Embedding",
}

# Pooling when the checkpoint ships no sentence-transformers Pooling config
# (the mlx-community conversions list one in modules.json without shipping
# it). sentence-transformers takes the mean for a model without one, or the
# last token for a causal LM architecture such as Qwen3-Embedding's.
DEFAULT_POOLING = {
    "gemma3_text": "mean",
    "bert": "mean",
    "xlm-roberta": "mean",
    "xlm_roberta": "mean",
    "modernbert": "mean",
    "qwen3": "last",
}


def _name_pattern(prefix: str) -> re.Pattern:
    """Matches a model name that starts with ``prefix``, followed by its
    end or a ``-``, ``_`` or ``.``."""
    return re.compile(rf"(?:{prefix})(?:$|[-_.])")


# Families whose upstream sentence-transformers Pooling configs take the
# first token. Each pattern matches the last part of a model id the
# checkpoint names (see model_ids), lowercased.
FIRST_TOKEN_FAMILIES = {
    "BGE-M3": _name_pattern(r"bge-m3"),
    "Arctic-Embed": _name_pattern(r"snowflake-arctic-embed"),
    "BGE": _name_pattern(r"bge-(small|base|large)-(en|zh)"),
    "mxbai-embed-large": _name_pattern(r"mxbai-embed-large-v1"),
    "mxbai-embed-2d-large": _name_pattern(r"mxbai-embed-2d-large-v1"),
}

# Families whose upstream Pooling configs take the mean, which is also the
# default. A name here settles the pooling, so the server fetches no model
# card for it (see card_files_to_fetch). The mlx-community conversions of
# nomic-ai/modernbert-embed-base are named nomicai-modernbert-embed-base.
MEAN_FAMILIES = {
    "ModernBERT Embed": _name_pattern(r"(nomicai-)?modernbert-embed"),
}

# A model id of a Qwen3-Embedding checkpoint (see model_ids), such as
# Qwen3-Embedding-0.6B, qwen3_embedding or Qwen3-Embed-4B.
_QWEN3_EMBEDDING = _name_pattern(r"qwen3[-_]embed(ding)?")

_POOLING_NAMES = {"mean": "the mean", "cls": "the first token",
                  "last": "the last token"}

# Encoder model types that never run float16, each with the model type
# whose dtype rule it takes (gmlx.load.dtypes.NO_FLOAT16_MODEL_TYPES). The
# rule is applied here, not by model type in the GGUF loader, because every
# Gemma 3 chat model also has the model type gemma3_text.
NO_FLOAT16 = {"gemma3_text": "gemma_embedding"}


# Appended to a refusal of a Qwen checkpoint.
_QWEN3_HINT = (" For Qwen3-Embedding, set server.embeddings to the "
               "qwen3-embed-0.6b, qwen3-embed-4b or qwen3-embed-8b GGUF alias, "
               "or to a Qwen3-Embedding safetensors checkpoint such as "
               "mlx-community/Qwen3-Embedding-0.6B-8bit.")

# Appended to a refusal of a Gemma 3 checkpoint.
_GEMMA_HINT = (" For EmbeddingGemma, set server.embeddings to the "
               "embeddinggemma alias, the embeddinggemma-gguf GGUF alias, or "
               "an EmbeddingGemma safetensors checkpoint such as "
               "mlx-community/embeddinggemma-300m-8bit.")


class UnsupportedEncoderError(ValueError):
    """The checkpoint does not run as a gmlx encoder. ``reason`` says why
    without the local path, for a message to a client. ``transient`` marks a
    file that could not be read, which a later try may read."""

    def __init__(self, path: str, reason: str, *, transient: bool = False):
        super().__init__(f"{path}: {reason}")
        self.reason = reason
        self.transient = transient


def supported_models() -> str:
    """The supported model families, for messages."""
    names = list(dict.fromkeys(FAMILIES.values()))
    return ", ".join(names[:-1]) + " and " + names[-1]


def resolve_path(model_ref: str) -> str:
    """A local folder as-is, else the local snapshot of a Hub repo id."""
    expanded = os.path.expanduser(model_ref)
    if os.path.isdir(expanded):
        return expanded
    from huggingface_hub import snapshot_download

    return snapshot_download(model_ref)


def _read_json(path: str, name: str) -> dict | list | None:
    """The JSON file ``name`` in the checkpoint at ``path``, or None when it
    is absent. A file that cannot be read or parsed raises
    :class:`UnsupportedEncoderError` naming it."""
    try:
        with open(os.path.join(path, name), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except ValueError as exc:             # JSONDecodeError, UnicodeDecodeError
        raise UnsupportedEncoderError(
            path, f"{name} is not valid JSON ({exc})") from None
    except OSError as exc:
        raise UnsupportedEncoderError(
            path, f"{name} cannot be read "
                  f"({exc.strerror or type(exc).__name__})",
            transient=True) from None


def _modules(path: str) -> list:
    """The sentence-transformers modules the checkpoint lists, or []."""
    modules = _read_json(path, "modules.json")
    return [m for m in modules if isinstance(m, dict)] if isinstance(modules, list) else []


def _pooling_config_name(path: str) -> str | None:
    """The path, relative to the checkpoint, of the config of the Pooling
    module that modules.json lists, or None when it lists none. A module
    path that leaves the checkpoint folder counts as none."""
    for m in _modules(path):
        if not str(m.get("type", "")).endswith(".Pooling"):
            continue
        name = posixpath.normpath(
            posixpath.join(str(m.get("path") or ""), "config.json"))
        if name.startswith(("/", "../")) or "\\" in name:
            return None
        return name
    return None


def _card_front_matter(path: str) -> str:
    try:
        with open(os.path.join(path, "README.md"), encoding="utf-8") as f:
            text = f.read()
    except (OSError, ValueError):         # absent, unreadable or not UTF-8
        return ""
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    return text[3:end] if end > 0 else ""


def _path_ids(path: str) -> list[str]:
    """The model ids the checkpoint's location names: the repo of a Hugging
    Face cache snapshot (a ``models--org--name`` folder above it) and the
    folder name, where ``org__name`` stands for ``org/name``."""
    parts = os.path.abspath(path).split(os.sep)
    ids = [p.removeprefix("models--").replace("--", "/")
           for p in parts if p.startswith("models--")]
    return ids + [parts[-1].replace("__", "/")]


def model_ids(path: str, config: dict | None = None) -> list[str]:
    """The model ids a checkpoint names: ``_name_or_path`` in config.json,
    the ``base_model`` and ``model-index`` names in the front matter of its
    model card, and the repo or folder name of its location."""
    if config is None:
        config = _read_json(path, "config.json")
    ids = []
    if isinstance(config, dict) and config.get("_name_or_path"):
        ids.append(str(config["_name_or_path"]))
    front = _card_front_matter(path)
    ids += re.findall(r"^base_model:[ \t]*['\"]?([^\s'\"]+)", front, re.M)
    block = re.search(r"^base_model:[ \t]*\n((?:[ \t]*-[ \t]*\S+[ \t]*\n)+)",
                      front + "\n", re.M)
    if block:
        ids += re.findall(r"-[ \t]*['\"]?([^\s'\"]+)", block.group(1))
    ids += re.findall(r"^model-index:[ \t]*\n[ \t]*-[ \t]*name:[ \t]*['\"]?([^\s'\"]+)",
                      front, re.M)
    return ids + _path_ids(path)


def _names(ids: list[str]) -> list[str]:
    """The last part of each model id, lowercased."""
    return [i.rstrip("/").rsplit("/", 1)[-1].lower() for i in ids]


def _first_token_family(ids: list[str]) -> str | None:
    return _family(ids, FIRST_TOKEN_FAMILIES)


def _family(ids: list[str], families: dict) -> str | None:
    for name in _names(ids):
        for family, pattern in families.items():
            if pattern.match(name):
                return family
    return None


def _named_pooling(ids: list[str]) -> tuple[str, str] | None:
    """``(mode, family)`` for a model id in FIRST_TOKEN_FAMILIES or
    MEAN_FAMILIES, or None."""
    for mode, families in (("cls", FIRST_TOKEN_FAMILIES),
                           ("mean", MEAN_FAMILIES)):
        family = _family(ids, families)
        if family is not None:
            return mode, family
    return None


def pooling_mode(path: str, model_type: str) -> str:
    """The sentence pooling for the checkpoint at ``path``.

    Without a Pooling config, the mode is inferred and one line on stderr
    says which and why."""
    name = _pooling_config_name(path)
    cfg = _read_json(path, name) if name is not None else None
    if isinstance(cfg, dict):
        if cfg.get("pooling_mode_cls_token"):
            return "cls"
        if cfg.get("pooling_mode_mean_tokens"):
            return "mean"
        if cfg.get("pooling_mode_lasttoken"):
            return "last"
        enabled = sorted(k for k, v in cfg.items()
                         if k.startswith("pooling_mode_") and v is True)
        raise UnsupportedEncoderError(
            path, f"sentence pooling {enabled} is not supported "
                  f"(supported: {', '.join(POOLING_MODES)})")
    named = _named_pooling(model_ids(path))
    if named is not None:
        mode, family = named
        why = f"as the upstream {family} config does"
    else:
        mode = DEFAULT_POOLING[model_type]
        why = f"the default for {FAMILIES[model_type]}"
    print(f"[embeddings] {path} ships no sentence-transformers Pooling config; "
          f"pooling {_POOLING_NAMES[mode]}, {why}", file=sys.stderr)
    return mode


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


def _cast(model: nn.Module, dtypes: set, dtype: mx.Dtype) -> None:
    """Cast the parameters whose dtype is in ``dtypes`` to ``dtype``."""
    model.update(tree_map(lambda p: p.astype(dtype) if p.dtype in dtypes else p,
                          model.parameters()))


def _no_float16(model: nn.Module, model_type: str, quantized: bool = False) -> None:
    """Cast every 16-bit float parameter of a model type that never runs
    float16 to the dtype the chip policy gives it: bfloat16, or float32 where
    the policy picks float16. The activations follow the parameters.

    EmbeddingGemma is such a type. Its model card rules out float16
    activations, and the mlx-community conversions store their norms, dense
    weights and quantization scales in float16. A quantized checkpoint runs
    float32 on the CPU device (see :func:`build`)."""
    dtype = activation_dtype(NO_FLOAT16[model_type])
    if quantized and cpu_default_device():
        dtype = mx.float32
    _cast(model, {mx.float16, mx.bfloat16} - {dtype}, dtype)


def check(path: str, config: dict | None = None) -> str:
    """The checkpoint's model_type, once config.json and the
    sentence-transformers modules show an encoder gmlx runs. Reads no
    weights. Raises :class:`UnsupportedEncoderError` naming the reason.

    A gemma3_text checkpoint is EmbeddingGemma only with
    ``use_bidirectional_attention`` or the sentence-transformers Dense
    modules. A qwen3 checkpoint is Qwen3-Embedding only with the
    sentence-transformers Pooling module, or when a model id it names (see
    :func:`model_ids`) is a Qwen3-Embedding one, as for a conversion that
    left out modules.json. Otherwise each is a causal LM.
    """
    if config is None:
        config = _read_json(path, "config.json")
    if config is None and not os.path.exists(os.path.join(path, "config.json")):
        raise FileNotFoundError(f"no config.json in {path}")
    if not isinstance(config, dict):
        raise UnsupportedEncoderError(path, "config.json is not a JSON object")
    model_type = str(config.get("model_type"))
    if model_type not in MODEL_TYPES:
        hint = _QWEN3_HINT if model_type.lower().startswith("qwen") else ""
        raise UnsupportedEncoderError(
            path, f"model_type {config.get('model_type')!r} is not a supported "
                  f"embeddings model. Supported: {supported_models()}.{hint}")
    types = [str(m.get("type", "")) for m in _modules(path)]
    if model_type == "gemma3_text" and not (
            config.get("use_bidirectional_attention") is True
            or any(t.endswith(".Dense") for t in types)):
        raise UnsupportedEncoderError(
            path, "this gemma3_text checkpoint sets no "
                  "use_bidirectional_attention and lists no "
                  "sentence-transformers Dense module, so it is a Gemma 3 "
                  "causal LM, not EmbeddingGemma. Supported: "
                  f"{supported_models()}.{_GEMMA_HINT}")
    if model_type == "qwen3" and not (
            any(t.endswith(".Pooling") for t in types)
            or any(_QWEN3_EMBEDDING.match(n)
                   for n in _names(model_ids(path, config)))):
        raise UnsupportedEncoderError(
            path, "this qwen3 checkpoint lists no sentence-transformers "
                  "Pooling module and names no Qwen3-Embedding model, so it "
                  "is a Qwen3 causal LM, not Qwen3-Embedding. Supported: "
                  f"{supported_models()}.{_QWEN3_HINT}")
    return model_type


def card_files_to_fetch(path: str) -> list[str]:
    """The files of the checkpoint's Hub repo that the folder at ``path``
    lacks and that could settle what it is and how it pools, or [] when its
    own files and names settle that.

    The pooling is settled by a Pooling config in the folder, by a name in
    FIRST_TOKEN_FAMILIES or MEAN_FAMILIES (see :func:`model_ids`), or by the
    model type for EmbeddingGemma and Qwen3-Embedding. A gemma3_text or qwen3
    checkpoint must first pass :func:`check`, which modules.json can settle,
    and for qwen3 also the model card. A folder whose config.json is absent,
    unreadable or of another model type gets [], as no fetched file would
    load it. The server fetches these files (gmlx.serve.hf_cache)."""
    try:
        config = _read_json(path, "config.json")
        if not isinstance(config, dict):
            return []
        model_type = str(config.get("model_type"))
        if model_type not in MODEL_TYPES:
            return []
        pooling = _pooling_config_name(path)
        if pooling is not None and os.path.isfile(os.path.join(path, pooling)):
            return []
        has_modules = os.path.isfile(os.path.join(path, "modules.json"))
        has_card = os.path.isfile(os.path.join(path, "README.md"))
        if model_type in ("gemma3_text", "qwen3"):
            try:
                check(path, config)
                return []
            except UnsupportedEncoderError:
                wanted = [] if has_modules else ["modules.json"]
                if model_type == "qwen3" and not has_card:
                    wanted.append("README.md")
                return wanted
        if _named_pooling(model_ids(path, config)) is not None:
            return []
    except UnsupportedEncoderError:
        return []
    wanted = ["modules.json"] if not has_modules else []
    if pooling is not None:
        wanted.append(pooling)
    if not has_card:
        wanted.append("README.md")
    return wanted


def build(path: str) -> nn.Module:
    """Build the encoder at ``path`` and load its weights."""
    config = _read_json(path, "config.json")
    model_type = check(path, config)
    assert isinstance(config, dict)
    module = MODEL_TYPES[model_type]
    weights = module.Model.sanitize(_load_weights(path))
    args = dict(config, pooling=pooling_mode(path, model_type))
    if module is gemma_embedding:
        args["dense_dims"] = _dense_dims(weights)
    model = module.Model(module.ModelArgs.from_dict(args))
    _quantize(model, config, weights)
    model.load_weights(list(weights.items()), strict=True)
    # The stock CPU quantized matmul accumulates in the activation dtype, and
    # in bfloat16 its error is some 20 times the rounding of its output. On
    # the CPU device a quantized encoder runs float32 in place of bfloat16.
    quantized = any(k.endswith(".scales") for k in weights)
    if model_type in NO_FLOAT16:
        _no_float16(model, model_type, quantized)
    elif quantized and cpu_default_device():
        _cast(model, {mx.bfloat16}, mx.float32)
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
