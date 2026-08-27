"""Tencent Hy3 tool-call parser (the ``:opensource``-suffixed tag format).

Hy3's chat template emits glm47-shaped tool calls where every control tag
carries the ``:opensource`` suffix, the function name is terminated by a
separator tag, and calls sit inside a wrapper block:

    <tool_calls:opensource>
    <tool_call:opensource>{name}<tool_sep:opensource>
    <arg_key:opensource>{key}</arg_key:opensource>
    <arg_value:opensource>{value}</arg_value:opensource>
    ...
    </tool_call:opensource>
    </tool_calls:opensource>

mlx-vlm's parser registry knows none of this: its template-marker inference
(``mlx_vlm.tool_parsers._TEMPLATE_MARKERS``) matches bare ``<arg_key>`` etc.,
and the Hy3 template defines its tags via ``'<arg_key{}>'.format(HYTK)`` so no
stock marker fires. ``ensure_registered()`` grafts this module in as
``mlx_vlm.tool_parsers.hy_v3`` (upstream-first, same pattern as
``hy_v3_model``) and prepends the Hy3 markers so
``_infer_tool_parser_from_processor`` resolves it from the template.

Parser contract (see mlx-vlm's ``tool_parsers/__init__.py`` consumers): module
attributes ``tool_call_start`` / ``tool_call_end`` delimit the call block in
the generated text; ``parse_tool_call(text, tools)`` receives the inner text
and returns ``{"name", "arguments"}`` or a list of them. The delimiters are
the *wrapper* tags, kimi_k2-style: the server's streaming suppressor holds
deltas only while the tail is a prefix of ``tool_call_start``, so keying on
the inner ``<tool_call:opensource>`` tag would let the wrapper open tag leak
to the client as answer text (``<tool_calls`` diverges from ``<tool_call:`` at
the ``s``). ``parse_tool_call`` therefore splits the per-call segments itself.
Argument values keep their declared-string types from the tool schema;
everything else round-trips through JSON (matching the template, which
serializes non-string values with ``tojson``).
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

_HYTK = ":opensource"

tool_call_start = f"<tool_calls{_HYTK}>"
tool_call_end = f"</tool_calls{_HYTK}>"

# Per-call segments inside the wrapper block.
_CALL_RE = re.compile(
    rf"<tool_call{_HYTK}>(.*?)</tool_call{_HYTK}>", re.DOTALL
)

_TOOL_SEP = f"<tool_sep{_HYTK}>"
_ARG_RE = re.compile(
    rf"<arg_key{_HYTK}>(.*?)</arg_key{_HYTK}>\s*"
    rf"<arg_value{_HYTK}>(.*?)</arg_value{_HYTK}>",
    re.DOTALL,
)

# Template markers that identify a Hy3 template: the raw Jinja carries the
# unformatted '<tool_sep{}>' literal (tags are built via HYTK .format()); a
# template with pre-expanded tags carries the suffixed spelling.
_TEMPLATE_MARKERS = [
    (("<tool_sep{}>",), "hy_v3"),
    ((_TOOL_SEP,), "hy_v3"),
]


def _string_arg_names(tool_name: str, tools: list[Any] | None) -> set[str]:
    """Argument names declared ``type: string`` for ``tool_name`` (schema-
    typed values must not be JSON-coerced: '123' stays a string)."""
    for tool in tools or ():
        func = tool.get("function")
        if not func or func.get("name") != tool_name:
            continue
        properties = (func.get("parameters") or {}).get("properties") or {}
        return {n for n, s in properties.items() if s.get("type") == "string"}
    return set()


def _deserialize(value: str) -> Any:
    """Invert the template's ``tojson`` for non-string values; a value that
    doesn't parse stays literal text."""
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_single(body: str, tools: list[Any] | None):
    """One ``<tool_call:opensource>`` body -> ``{"name", "arguments"}``."""
    name, sep, rest = body.partition(_TOOL_SEP)
    if not sep:
        # No separator: tolerate a bare name (zero-arg call) but never guess
        # at freeform text - surface it for the caller's error path.
        stripped = body.strip()
        if stripped and "<" not in stripped and not re.search(r"\s", stripped):
            return {"name": stripped, "arguments": {}}
        return {"name": "unknown", "arguments": {"raw": stripped}}
    name = name.strip()
    string_args = _string_arg_names(name, tools)
    arguments: dict[str, Any] = {}
    for m in _ARG_RE.finditer(rest):
        key = m.group(1).strip()
        value = m.group(2).strip()
        arguments[key] = value if key in string_args else _deserialize(value)
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a wrapper block's inner text: a list of parsed calls when
    ``<tool_call:opensource>`` segments are present, else the text as one
    bare call body (bare-name / unknown fallback)."""
    bodies = _CALL_RE.findall(text)
    if bodies:
        return [_parse_single(b, tools) for b in bodies]
    return _parse_single(text, tools)


def ensure_registered() -> None:
    """Make ``mlx_vlm.tool_parsers.hy_v3`` resolve (upstream wins) and teach
    the template-marker inference the Hy3 spellings. Idempotent."""
    import importlib

    if "mlx_vlm.tool_parsers.hy_v3" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.tool_parsers.hy_v3")  # upstream wins
        except ImportError:
            sys.modules["mlx_vlm.tool_parsers.hy_v3"] = sys.modules[__name__]
    try:
        registry = importlib.import_module("mlx_vlm.tool_parsers")
    except ImportError:
        return
    markers = getattr(registry, "_TEMPLATE_MARKERS", None)
    if isinstance(markers, list):
        for entry in _TEMPLATE_MARKERS:
            if entry not in markers:
                markers.insert(0, entry)
