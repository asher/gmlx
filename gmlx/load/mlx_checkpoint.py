"""MLX checkpoint loads with gmlx's model classes.

``load`` is the copied mlx-lm ``load`` with a K-quant route. Both build
the class the copy resolves, which for a Gemma 4 checkpoint is the owned
model. It keeps the signature of the copy's ``load``, so the mlx-lm server
bridge loads MLX checkpoints with it. ``resolve`` returns a local
directory as it is, and fetches anything else as a Hugging Face repo, with
the files mlx-lm fetches.

A K-quant checkpoint builds here, as ``mlx_kquant.loader.load`` builds
one but with the copy's resolver, or with the classes of the model file
its config names. The copy's ``load_model`` cannot read one. An adapter on a K-quant checkpoint wraps its layers through
``gmlx.tune.kquant_lora``, which the copy's LoRA setup calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

# The files mlx-lm fetches for a repo id.
_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def resolve(path_or_hf_repo: str | Path, revision: Optional[str] = None) -> Path:
    """The local directory of ``path_or_hf_repo``: the path itself when it
    exists, else the Hugging Face snapshot of that repo id, which a cache
    miss downloads."""
    path = Path(path_or_hf_repo)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    from gmlx.serve.hf_cache import network_fetch_allowed

    with network_fetch_allowed():
        return Path(snapshot_download(str(path_or_hf_repo), revision=revision,
                                      allow_patterns=_ALLOW_PATTERNS))


def load(
    path_or_hf_repo: str,
    tokenizer_config: Optional[dict[str, Any]] = None,
    model_config: Optional[dict[str, Any]] = None,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    return_config: bool = False,
    revision: Optional[str] = None,
) -> Any:
    """``(model, tokenizer)``, or ``(model, tokenizer, config)`` with
    ``return_config``, as ``mlx_lm.load`` returns them."""
    from gmlx._vendor.mlx_lm.utils import load_adapters, load_config, load_model, load_tokenizer

    model_path = resolve(path_or_hf_repo, revision)
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)
    per_tensor = _kquant_per_tensor(config, model_path)
    if per_tensor is None:
        model, config = load_model(model_path, lazy, model_config=model_config)
    else:
        # A model_file checkpoint names its own classes.
        model_file = config.get("model_file")
        model = _load_kquant(
            model_path, config, per_tensor, lazy,
            None if model_file is None
            else lambda _config: _model_file_classes(model_path / model_file))
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config,
                               eos_token_ids=config.get("eos_token_id", None))
    if return_config:
        return model, tokenizer, config
    return model, tokenizer


def _kquant_per_tensor(config: dict, model_path: Path) -> Optional[dict[str, str]]:
    """The ``{module path: codec}`` map of a K-quant checkpoint, from the
    first quantization block with mode kquant, at the top of the config
    or under ``text_config``; None for any other checkpoint."""
    for key in ("quantization", "quantization_config"):
        block = config.get(key)
        if isinstance(block, dict) and block.get("mode") == "kquant":
            if not block.get("per_tensor"):
                raise ValueError(f"{model_path}: kquant config has no "
                                 "'per_tensor' map")
            return block["per_tensor"]
    text = config.get("text_config")
    return _kquant_per_tensor(text, model_path) if isinstance(text, dict) else None


def _model_file_classes(model_file: Path) -> tuple[Any, Any]:
    """``(Model, ModelArgs)`` of a checkpoint's own model file, read as the
    copy's ``load_model`` reads it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("custom_model", model_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"{model_file}: not a Python module, and the "
                         "checkpoint's config names it as model_file")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    return arch.Model, arch.ModelArgs


def _load_kquant(model_path: Path, config: dict, per_tensor: dict[str, str],
                 lazy: bool, get_model_classes=None) -> Any:
    """The model of a K-quant checkpoint: the class of ``get_model_classes``,
    by default the copy's resolver, with a
    ``KQuant*`` module at each path the map names, and the wire bytes
    loaded into it. gmlx's module swap takes the switch classes of mlx-lm
    and of mlx-vlm, and an expert stack of a native float codec stays in
    its wire bytes, as the checkpoint holds it."""
    import mlx.core as mx

    from gmlx._vendor.mlx_lm.utils import _get_classes

    from gmlx.load.modules import install_kquant_modules

    model_class, args_class = (get_model_classes or _get_classes)(config)
    model = model_class(args_class.from_dict(config))
    files = sorted(model_path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights: dict[str, Any] = {}
    for f in files:
        weights.update(cast(dict[str, Any], mx.load(str(f))))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    install_kquant_modules(model, {f"{p}.weight": c for p, c in per_tensor.items()},
                           native_fp_wire=True)
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    return model
