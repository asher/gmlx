# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Decision models: the header record that marks one, and the refusal
every path that generates gives it.

Liquid's d1 models are arch ``lfm3`` with ``lfm3.attention.causal``
false. Their full-attention layers read the prompt in both directions,
so the causal forward would answer wrongly with no error. They answer
structured questions on ``/v1/systemone`` and with ``gmlx systemone``.
Importing this module loads no MLX.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar

from .gguf_meta import read_bool, read_string

DECISION_ARCHES = frozenset({"lfm3"})
# The most tokens one question's whole prompt may hold, media included.
DECISION_STATE_CAP = 65536

_REFUSAL = (
    "{what} is a decision model: it answers structured questions on "
    "/v1/systemone and with `gmlx systemone`, and does not generate text")


class DecisionModelOnly(NotImplementedError):
    """A decision model reached a path that generates."""

    def __init__(self, what: str = "this model"):
        super().__init__(_REFUSAL.format(what=what))


def decision_header(meta) -> bool:
    """Whether a GGUF header (a ``GGUFReader`` or a key-value dict) is a
    decision model's. Keyed on the attention, not on a
    ``<arch>.decision.type`` key, which chat models carry too."""
    arch = read_string(meta, "general.architecture")
    if arch not in DECISION_ARCHES:
        return False
    return read_bool(meta, f"{arch}.attention.causal") is False


def is_decision_model(model) -> bool:
    """Whether a loaded model is a decision model: the ``decision`` field
    of its text config, found through any server or VLM wrapper."""
    seen = set()
    stack = [model]
    while stack:
        m = stack.pop()
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        for name in ("args", "config", "text_config"):
            cfg = getattr(m, name, None)
            if cfg is None:
                continue
            value = (cfg.get("decision") if isinstance(cfg, dict)
                     else getattr(cfg, "decision", None))
            if value:
                return True
            if name == "config":
                stack.append(cfg)
        stack.extend(getattr(m, n, None)
                     for n in ("language_model", "_model", "model"))
    return False


_SCOPE: ContextVar = ContextVar("gmlx_decision_scope", default=False)


@contextlib.contextmanager
def decision_scope():
    """The request may load a decision model: the systemone route and the
    server's startup preload run inside this."""
    token = _SCOPE.set(True)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def in_decision_scope() -> bool:
    return bool(_SCOPE.get())
