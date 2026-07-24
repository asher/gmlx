"""gmlx - a local inference platform for Apple Silicon: run, chat with,
serve, and fine-tune community GGUF models natively on MLX.

``load_model`` loads any text-only GGUF (the K-quant, legacy, and IQ codec
families) straight off the file - no safetensors round-trip, no conversion -
into an mlx-lm-compatible ``class Model`` whose quantized leaves are
``KQuant*`` modules, gated only on gmlx recognizing the GGUF arch.

The K-quant ops/kernels come from the companion ``mlx-kquant`` project (a
runtime dependency). Vision-language loading adds the optional ``[vlm]``
extra.
"""

from __future__ import annotations

import importlib

# Exports are resolved lazily (PEP 562) so that importing the package - e.g.
# for `gmlx --help` / `validate` / `pull`, which never touch a model -
# doesn't pay the MLX + kernel-extension import, and so a broken runtime
# environment fails at first *use* with a message naming the missing piece.
_EXPORTS = {
    "load_model": "loader",
    "generate": "generation",
    "bench": "benchmarks",
    # The preflight function is deliberately not exported: it shares its name
    # with its submodule, and once anything imports gmlx.preflight (the
    # loader does) the submodule attribute shadows a lazy export - the name
    # would resolve to the function or the module depending on import order.
    # Use `from gmlx.preflight import preflight`.
    "UnsupportedCodecError": "preflight",
    "ARCH_TABLE": "arch_table",
    "UnsupportedArchError": "arch_table",
    "KQuantLinear": "modules",
    "KQuantEmbedding": "modules",
    "KQuantSwitchLinear": "modules",
    "KQuantMultiLinear": "modules",
    "install_kquant_modules": "modules",
    "install_gguf_bridge": "server_bridge_lm",
    # Tokenizer-only entry points: synthesize the HF tokenizer from GGUF
    # metadata without paying the model load. Used by eval tooling (mlx-kld)
    # that needs tokenizer parity checks before deciding to load weights.
    "detect_arch": "remap",
    "load_tokenizer_from_gguf": "tokenizer",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name == "__version__":
        from importlib import metadata
        try:
            version = metadata.version("gmlx")
        except metadata.PackageNotFoundError:  # uninstalled checkout
            version = "0.0.0+unknown"
        globals()[name] = version
        return version
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ImportError as exc:  # pragma: no cover - exercised without runtime deps
        raise ImportError(
            "gmlx requires mlx, mlx-kquant, mlx-lm, transformers, "
            "tokenizers, and gguf at runtime (Apple Silicon for the Metal "
            f"kernels). Importing gmlx.{module_name} failed - install "
            "with:  pip install gmlx"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | {"__version__"})
