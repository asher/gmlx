"""GGUF K-quant bridge into the mlx-lm server that gmlx carries
(``gmlx._vendor.mlx_lm.server``), on its sequential, non-batched path.

Installs a monkeypatch on that server's ``ModelProvider._load`` so that any
``--model *.gguf`` (or a request whose resolved model path ends in ``.gguf``)
is loaded through :func:`gmlx.load_model` instead of the copy's
``mlx_lm.utils.load``. Routing, the OpenAI surface, tool calls, structured
output and the prompt cache stay the copied server's. An installed stock
``mlx_lm.server`` is never patched. ``python -m gmlx.serve.bridge_lm`` runs
the patched server.

Non-``.gguf`` paths fall through to the original loader, so a single process
can mix GGUF K-quant and ordinary MLX checkpoints. That loader loads with
:func:`gmlx.load.mlx_checkpoint.load` in place of ``mlx_lm.utils.load``, so a
Gemma 4 checkpoint builds the owned model and every other architecture
builds mlx-lm's.

Batching note
-------------
The GGUF route sets ``is_batchable = False``, which pins every request to
``mlx_lm.server``'s validated *sequential* (``_serve_single``) path. The batched
``BatchGenerator`` path has not been validated against the ``KQuant*`` modules'
``gather_qmm`` / merged-cache behavior, so it is deliberately not enabled here.
Speculative decoding (``--draft-model``) is likewise ignored for GGUF models.
A K-quant MLX checkpoint holds the same modules, so its fallthrough load
sets ``is_batchable = False`` too.
``--adapter`` on a GGUF model raises (adapter apply is only wired in
``gmlx serve``).
"""

from __future__ import annotations

import sys

from gmlx.cache.kv_cache import make_prompt_cache
from gmlx.load.loader import load_model

_BRIDGE_FLAG = "_kq_gguf_bridge_installed"


def _is_gguf(path) -> bool:
    return isinstance(path, str) and path.endswith(".gguf")


def _has_kquant_modules(model) -> bool:
    import mlx.nn as nn
    from mlx_kquant.nn import (
        KQuantEmbedding,
        KQuantLinear,
        KQuantMultiLinear,
        KQuantSwitchLinear,
    )

    if not isinstance(model, nn.Module):
        return False
    kinds = (KQuantEmbedding, KQuantLinear, KQuantMultiLinear, KQuantSwitchLinear)
    return any(isinstance(m, kinds) for _, m in model.named_modules())


def install_gguf_bridge() -> None:
    """Idempotently patch ``ModelProvider._load`` to handle ``*.gguf`` models,
    and load MLX checkpoints with gmlx's model classes."""
    from gmlx._vendor.mlx_lm import server as _server

    from gmlx.load import mlx_checkpoint

    if getattr(_server.ModelProvider, _BRIDGE_FLAG, False):
        return

    # The fallthrough below, the stock _load, loads the model and any draft
    # model through the server module's ``load``.
    setattr(_server, "load", mlx_checkpoint.load)

    _orig_load = _server.ModelProvider._load

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if not _is_gguf(model_path):
            out = _orig_load(self, model_path, adapter_path, draft_model_path)
            if _has_kquant_modules(self.model):
                self.is_batchable = False
            return out

        if self.is_distributed:
            raise ValueError(
                "GGUF K-quant loading is not supported in distributed mode.")
        if adapter_path is not None:
            # Adapter apply is not wired on this route; raising beats silently
            # serving the bare base under the user's --adapter.
            raise ValueError(
                "adapter_path is not supported on the mlx-lm server bridge "
                "route - use `gmlx serve`")
        if draft_model_path is not None:
            # Speculative decoding against a GGUF target is unvalidated on this
            # route; warn instead of silently dropping it (the user asked for it).
            print(
                "warning: --draft-model is ignored for GGUF models on the "
                "mlx-lm server route (speculative decoding is unvalidated here) "
                "- use `gmlx serve` for MTP",
                file=sys.stderr)

        # Clear any previous model first (mirrors the upstream contract).
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None

        chat_template = self._tokenizer_config.get("chat_template")
        model, _config, tokenizer = load_model(
            model_path, chat_template=chat_template, verbose=True)

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        # Force the validated sequential path; see module docstring. The key
        # keeps the caller's original draft_model_path: load() recomputes it
        # per request, and a mismatch silently reloads the model every time.
        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = None
        self.is_batchable = False

        # Touch make_prompt_cache once so an unsupported cache shape surfaces
        # at load time rather than mid-request.
        make_prompt_cache(model)

    _server.ModelProvider._load = _load
    setattr(_server.ModelProvider, _BRIDGE_FLAG, True)


def main(argv: list[str] | None = None) -> int:
    """Install the GGUF bridge, then hand off to ``mlx_lm.server.main``."""
    install_gguf_bridge()
    from gmlx._vendor.mlx_lm import server as _server

    if argv is not None:
        sys.argv = ["gmlx serve", *argv]
    _server.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
