"""Muse Glimmer tool-call parser (the Onyx ATEM XML format).

Muse Glimmer's chat template emits Claude-shaped tool calls inside a wrapper
block, one ``<atem:invoke>`` segment per call:

    <atem:function_calls>
    <atem:invoke name="{name}">
    <atem:parameter name="{key}">{value}</atem:parameter>
    ...
    </atem:invoke>
    </atem:function_calls>

mlx-vlm's parser registry knows none of this, and its template-marker
inference (``mlx_vlm.tool_parsers._TEMPLATE_MARKERS``) matches no ATEM tag.
``ensure_registered()`` grafts this module in as
``mlx_vlm.tool_parsers.muse_glimmer`` (upstream-first, same pattern as
``hy_v3_tools``) and prepends the ATEM markers so
``_infer_tool_parser_from_processor`` resolves it from the template.

Parser contract (see mlx-vlm's ``tool_parsers/__init__.py`` consumers): module
attributes ``tool_call_start`` / ``tool_call_end`` delimit the call block in
the generated text; ``parse_tool_call(text, tools)`` receives the inner text
and returns ``{"name", "arguments"}`` or a list of them. The delimiters are
the *wrapper* tags: the server's streaming suppressor holds deltas only while
the tail is a prefix of ``tool_call_start``, so keying on the inner
``<atem:invoke`` tag would let the wrapper open tag leak to the client as
answer text. ``parse_tool_call`` therefore splits the per-call segments itself.

Values are coerced by the tool's declared JSON schema rather than by guessing,
because the template serializes by type: booleans as bare ``true``/``false``,
null as ``null``, objects and arrays through ``tojson``, and everything else
verbatim (so a string parameter must survive as ``"123"``, not become 123).
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

tool_call_start = "<atem:function_calls>"
tool_call_end = "</atem:function_calls>"

# Per-call segments inside the wrapper block.
_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="([^"]*)"\s*>(.*?)</atem:invoke>', re.DOTALL
)
_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="([^"]*)"\s*>(.*?)</atem:parameter>', re.DOTALL
)

# The raw Jinja carries these literals verbatim (the tags are not built by
# .format(), unlike Hy3's).
_TEMPLATE_MARKERS = [
    (("<atem:invoke name=",), "muse_glimmer"),
    ((tool_call_start,), "muse_glimmer"),
]


def _schema(tool_name: str, tools: list[Any] | None) -> dict:
    """The declared ``properties`` schema for ``tool_name``, or empty."""
    for tool in tools or ():
        func = tool.get("function") if isinstance(tool, dict) else None
        func = func or tool
        if not isinstance(func, dict) or func.get("name") != tool_name:
            continue
        return (func.get("parameters") or {}).get("properties") or {}
    return {}


def _coerce(value: str, spec: dict | None) -> Any:
    """Invert the template's per-type serialization for one parameter. With no
    schema entry the value stays literal text - guessing would turn a string
    argument that happens to look numeric into a number."""
    declared = (spec or {}).get("type")
    if declared == "string":
        return value
    stripped = value.strip()
    if declared == "boolean":
        return stripped == "true"
    if declared in ("number", "integer"):
        try:
            return json.loads(stripped)
        except ValueError:
            return value
    if declared in ("object", "array"):
        try:
            return json.loads(stripped)
        except ValueError:
            return value
    if declared == "null" or stripped == "null":
        return None
    return value


def _parse_single(name: str, body: str, tools: list[Any] | None) -> dict:
    """One ``<atem:invoke>`` body -> ``{"name", "arguments"}``."""
    properties = _schema(name, tools)
    arguments: dict[str, Any] = {}
    for m in _PARAM_RE.finditer(body):
        key = m.group(1)
        arguments[key] = _coerce(m.group(2), properties.get(key))
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a wrapper block's inner text: a list of parsed calls when
    ``<atem:invoke>`` segments are present, else an unknown-call envelope
    carrying the raw text (never a guess at freeform prose)."""
    calls = _INVOKE_RE.findall(text)
    if calls:
        return [_parse_single(name, body, tools) for name, body in calls]
    return {"name": "unknown", "arguments": {"raw": text.strip()}}


def ensure_registered() -> None:
    """Make ``mlx_vlm.tool_parsers.muse_glimmer`` resolve (upstream wins) and
    teach the template-marker inference the ATEM spellings. Idempotent."""
    import importlib

    if "mlx_vlm.tool_parsers.muse_glimmer" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.tool_parsers.muse_glimmer")
        except ImportError:
            sys.modules["mlx_vlm.tool_parsers.muse_glimmer"] = sys.modules[__name__]
    try:
        registry = importlib.import_module("mlx_vlm.tool_parsers")
    except ImportError:
        return
    markers = getattr(registry, "_TEMPLATE_MARKERS", None)
    if isinstance(markers, list):
        for entry in _TEMPLATE_MARKERS:
            if entry not in markers:
                markers.insert(0, entry)
