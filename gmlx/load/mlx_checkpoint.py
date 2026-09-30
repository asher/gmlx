"""MLX checkpoint loads with gmlx's model classes.

``load`` is ``mlx_lm.load`` with the class resolver of
``gmlx.models.gemma4.text``: a Gemma 4 checkpoint builds the owned model,
and every other architecture builds mlx-lm's. It keeps the signature of
``mlx_lm.load``, so the mlx-lm server bridge loads MLX checkpoints with
it. ``resolve`` returns a local directory as it is, and fetches anything
else as a Hugging Face repo, with the files mlx-lm fetches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
    from gmlx._vendor.mlx_lm.utils import load_adapters, load_model, load_tokenizer

    from gmlx.models.gemma4.text import get_model_classes

    model_path = resolve(path_or_hf_repo, revision)
    model, config = load_model(model_path, lazy, model_config=model_config,
                               get_model_classes=get_model_classes)
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)
        model.eval()
    tokenizer = load_tokenizer(model_path, tokenizer_config,
                               eos_token_ids=config.get("eos_token_id", None))
    if return_config:
        return model, tokenizer, config
    return model, tokenizer
