"""The errors of ``POST /v1/embeddings``, apart from the engine so that the
request parser imports no model code."""

from __future__ import annotations

from .subservice import SubserviceRequestError


class EmbeddingsRequestError(SubserviceRequestError):
    """A client-side problem with an embeddings request (HTTP 4xx)."""


class EmbeddingsModelError(EmbeddingsRequestError):
    """The configured model cannot load (HTTP 501). When no retry can fix
    that, the reason is cached and every request returns it."""

    err_type = "model_unsupported"

    def __init__(self, reason: str):
        super().__init__(
            501, f"the configured `server.embeddings` model cannot load: "
                 f"{reason}")
