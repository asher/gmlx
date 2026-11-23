# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Inkling tool-call parser.

A tool call is one block of the model's turn, as in llama.cpp's inkling
parser (``common/parsers/inkling.cpp``):

    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

A call has no start marker of its own: ``<|message_model|>`` opens every
block. ``find_tool_calls`` finds the call spans, so ``tool_call_start`` and
``tool_call_end`` stay empty, as in the Beam parser.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

tool_call_start = ""
tool_call_end = ""

_CALL_RE = re.compile(
    r"<\|message_model\|>([^<]*)<\|content_invoke_tool_json\|>(.*?)"
    r"(?:<\|end_message\|>|$)", re.DOTALL)

_TEMPLATE_MARKERS = (("<|content_invoke_tool_json|>", "<|message_model|>"),)


def find_tool_calls(text: str) -> list[tuple[int, int]]:
    """The spans of the tool-call blocks in ``text``. A call cut off at the
    end of the text runs to the end."""
    return [m.span() for m in _CALL_RE.finditer(text)]


def parse_tool_call(text: str, tools: list[Any] | None = None) -> dict:
    """One call span -> ``{"name", "arguments"}``. The header names the
    tool, and the body's ``args`` object holds the arguments."""
    del tools
    m = _CALL_RE.search(text)
    if m is None:
        raise ValueError(f"not an Inkling tool call: {text!r}")
    name, body = m.group(1).strip(), m.group(2).strip()
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("args"), dict):
        raise ValueError(f"tool call body has no args object: {body!r}")
    # The header names the tool, as llama.cpp's grammar reads it.
    return {"name": name or payload.get("name") or "",
            "arguments": payload["args"]}


def ensure_registered() -> None:
    """Register this module as the ``inkling`` tool parser, ahead of any
    vendored parser of that name, and route Inkling templates to it.
    Idempotent."""
    from gmlx.upstream.tool_parsers import register_tool_parser

    register_tool_parser("inkling", sys.modules[__name__], _TEMPLATE_MARKERS)
