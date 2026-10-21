# SPDX-License-Identifier: Apache-2.0
"""Beam tool-call parser (Harmony headers with XML arguments).

Beam writes each call as its own assistant message. The recipient sits in
the header and the arguments are XML parameters::

    <|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>
    <rfl:parameter name="KEY">
    VALUE
    </rfl:parameter><|end|>

Parallel calls repeat the message, joined by ``<|end|>``. The last call ends
with ``<|call|>``, which is a stop token and never reaches the text. The
generation prompt ends at ``<|start|>assistant``, so the output can begin
with `` to=NAME``.

A call has no closing tag of its own, so the default extractor slicing
cannot find it. ``find_tool_calls`` finds the spans, and ``process_tool_calls``
uses it in place of ``tool_call_start``/``tool_call_end``. Both stay empty:
``ReasoningFilter`` hides the call from the stream, and an empty start makes
the stream suppressor pass content through.

Values follow llama.cpp's ``common/parsers/harmony-xml.cpp``. A value ends at
``\\n</rfl:parameter>``, or at a newline before the next parameter, ``<|end|>``
or the end of the text, so a missing close tag is tolerated. Values are
typed by the tool schema, as for K2 Horizon.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from gmlx.models.k2_horizon.tools import _decode, _param_types

tool_call_start = ""
tool_call_end = ""

_MESSAGE = "<|message|>"
_CONSTRAIN_XML = "<|constrain|>xml"
_PARAM_OPEN = '<rfl:parameter name="'
_PARAM_CLOSE = "\n</rfl:parameter>"
_NEXT_HEADER = "<|start|>assistant"
# Text that ends a call body. All but the next header belong to the call.
_CALL_ENDS = ("<|end|>", "<|call|>", "<|return|>", "<|endoftext|>",
              "<|im_end|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>")

_HEAD_RE = re.compile(r"(?:<\|start\|>assistant|\A)\s*to=([^\s<]+)")

# The template's tool instructions spell both. Both are needed so no other
# Harmony template that constrains to XML is claimed.
_TEMPLATE_MARKERS = ((_CONSTRAIN_XML, "rfl:parameter"),)


def _body_end(text: str, start: int) -> tuple[int, int]:
    """Where a call body that starts at ``start`` ends, and where its span
    ends. The span keeps a closing token and stops before a next header."""
    end, stop = len(text), len(text)
    i = text.find(_NEXT_HEADER, start)
    if i >= 0:
        end = stop = i
    for tok in _CALL_ENDS:
        i = text.find(tok, start, end)
        if i >= 0:
            end, stop = i, i + len(tok)
    return end, stop


def find_tool_calls(text: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` span of each call in ``text``, in order. A header
    with no ``<|message|>`` yet is no call."""
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _HEAD_RE.finditer(text):
        if m.start() < pos:
            continue
        msg = text.find(_MESSAGE, m.end())
        if msg < 0:
            break
        header = text[m.end():msg]
        if _CONSTRAIN_XML not in header or "<|start|>" in header or "<|end|>" in header:
            continue
        _, stop = _body_end(text, msg + len(_MESSAGE))
        spans.append((m.start(), stop))
        pos = stop
    return spans


def _value(raw: str) -> str:
    close = raw.find(_PARAM_CLOSE)
    if close >= 0:
        return raw[:close]
    return raw.removesuffix("\n")


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """One call span from ``find_tool_calls`` -> ``{"name", "arguments"}``."""
    m = _HEAD_RE.search(text)
    msg = text.find(_MESSAGE)
    if m is None or msg < 0:
        raise ValueError("no tool-call header")
    name = m.group(1)
    body_start = msg + len(_MESSAGE)
    end, _ = _body_end(text, body_start)
    types = _param_types(name, tools)
    arguments: dict[str, Any] = {}
    for chunk in text[body_start:end].split(_PARAM_OPEN)[1:]:
        key, sep, raw = chunk.partition('">')
        if not sep:
            continue
        value = _value(raw.removeprefix("\n"))
        arguments[key] = _decode(value, types.get(key, set()))
    return {"name": name, "arguments": arguments}


def ensure_registered() -> None:
    """Register this module as the ``beam`` tool parser and route templates
    carrying Beam's XML parameters to it. Idempotent."""
    from gmlx.upstream.tool_parsers import register_tool_parser

    register_tool_parser("beam", sys.modules[__name__], _TEMPLATE_MARKERS)
