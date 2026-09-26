"""A request body parsed for each decision backend.

DiffusionGemma reads answer slots on a denoise canvas (``diffusion``), and
any other text model answers with the letter readout (``letters``). The
server resolves the model after it reads the body, so the body is parsed
for both backends first and the model kind picks one parse later."""

from __future__ import annotations

from typing import Any

from . import letters
from .contract import jev_state
from .extensions import ignored_fields, request_schema
from .schema import Limits, SchemaError, parse_seed


class ParsedRequest:
    """``get("diffusion")`` is ``(schema, state, seed)`` and
    ``get("letters")`` is ``(schema, state)``. Each raises the SchemaError
    of a body its backend rejects."""

    def __init__(self, body, limits: Limits = Limits(), defaults: dict | None = None):
        self.body = body
        self._got: dict[str, Any] = {}
        self._failed: dict[str, SchemaError] = {}
        try:
            self._got["diffusion"] = (request_schema(body, limits, defaults),
                                      jev_state(body), parse_seed(body))
        except SchemaError as e:
            self._failed["diffusion"] = e
        try:
            self._got["letters"] = (letters.parse(body, limits), letters.state_text(body))
        except SchemaError as e:
            self._failed["letters"] = e

    @property
    def error(self) -> SchemaError | None:
        """The error to report when neither backend takes the body."""
        return None if self._got else self._failed["diffusion"]

    def get(self, backend: str):
        if backend in self._failed:
            raise self._failed[backend]
        return self._got[backend]

    def ignored(self, backend: str) -> set:
        """The fields the backend parses but does not use."""
        if backend == "letters":
            return letters.ignored(self.body)
        return ignored_fields(self.body, self._got["diffusion"][0])
