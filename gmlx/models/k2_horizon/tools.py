"""K2 Horizon tool-call parser (the ``<ifm|...>`` tag format).

The K2 Horizon chat template wraps one or more calls in a section block.
Each call is the function name on its own line, then key and value pairs
(``tool_call_format`` ``xml``, the default)::

    <ifm|tool_calls>
    <ifm|tool_call>{name}
    <ifm|arg_key>{key}</ifm|arg_key>
    <ifm|arg_value>{value}</ifm|arg_value>
    </ifm|tool_call>
    </ifm|tool_calls>

``xml_typed`` adds ``<ifm|arg_type>{type}</ifm|arg_type>`` after each key,
which the models often leave out. ``json`` writes
``<ifm|tool_call>{"name": ..., "arguments": {...}}</ifm|tool_call>``.

Parser contract (``mlx_vlm.tools.base.ToolParser``): ``tool_call_start`` and
``tool_call_end`` are the wrapper tags, so the streaming suppressor holds the
whole section, and ``parse_tool_call(text, tools)`` splits the per-call
segments of the inner text itself.

Argument values follow llama.cpp's ``common/parsers/k2-horizon.cpp``. The
template writes a string raw and any other value with ``tojson``. A value
whose schema type is only string stays raw text. A union with string takes
the JSON reading only when it parses as one of the union's other types. Any
other value is JSON, and stays text when it does not parse.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from gmlx.upstream.patching import link, wrapped_with

tool_call_start = "<ifm|tool_calls>"
tool_call_end = "</ifm|tool_calls>"

_CALL_RE = re.compile(r"<ifm\|tool_call>(.*?)</ifm\|tool_call>", re.DOTALL)
_ARG_KEY = "<ifm|arg_key>"
_ARG_RE = re.compile(
    r"<ifm\|arg_key>(.*?)</ifm\|arg_key>\s*"
    r"(?:<ifm\|arg_type>.*?</ifm\|arg_type>\s*)?"
    r"<ifm\|arg_value>(.*?)</ifm\|arg_value>",
    re.DOTALL,
)

# The template's tool-call instructions spell both tags. Both are needed so
# no other template that mentions one of them is claimed.
_TEMPLATE_MARKERS = ((tool_call_start, _ARG_KEY),)

# The reasoning fields the template reads from an assistant message. It
# raises when none is present.
_THINKING_FIELDS = ("think", "think_fast", "think_faster",
                    "reasoning_content", "reasoning")

_JSON_TYPES = {
    "object": (dict,),
    "array": (list,),
    "boolean": (bool,),
    "null": (type(None),),
}


def _schema_types(schema: Any) -> set[str]:
    """The JSON types a parameter schema allows. Empty means any type."""
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    out: set[str] = set()
    for key in ("anyOf", "oneOf"):
        for sub in schema.get(key) or ():
            sub_types = _schema_types(sub)
            if not sub_types:
                return set()
            out |= sub_types
    return out


def _param_types(tool_name: str, tools: list[Any] | None) -> dict[str, set[str]]:
    for tool in tools or ():
        func = tool.get("function") if isinstance(tool, dict) else None
        if not func or func.get("name") != tool_name:
            continue
        properties = (func.get("parameters") or {}).get("properties") or {}
        return {n: _schema_types(s) for n, s in properties.items()}
    return {}


def _is_type(value: Any, types: set[str]) -> bool:
    for t in types:
        if t in ("number", "integer"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        elif isinstance(value, _JSON_TYPES.get(t, ())):
            return True
    return False


def _decode(value: str, types: set[str]) -> Any:
    if types == {"string"}:
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    if "string" in types and not _is_type(parsed, types - {"string"}):
        return value
    return parsed


def _parse_json_call(body: str):
    try:
        call = json.loads(body)
    except ValueError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return {"name": call["name"], "arguments": args if args is not None else {}}


def _parse_single(body: str, tools: list[Any] | None):
    """One ``<ifm|tool_call>`` body -> ``{"name", "arguments"}``."""
    if body.lstrip().startswith("{"):
        call = _parse_json_call(body.strip())
        if call is not None:
            return call
    name, sep, rest = body.partition(_ARG_KEY)
    name = name.strip()
    if not sep:
        if name and "<" not in name and not re.search(r"\s", name):
            return {"name": name, "arguments": {}}
        return {"name": "unknown", "arguments": {"raw": name}}
    types = _param_types(name, tools)
    arguments: dict[str, Any] = {}
    for m in _ARG_RE.finditer(_ARG_KEY + rest):
        key = m.group(1).strip()
        arguments[key] = _decode(m.group(2), types.get(key, set()))
    return {"name": name, "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse a section's inner text: a list of calls when ``<ifm|tool_call>``
    segments are present, else the text as one bare call body."""
    bodies = _CALL_RE.findall(text)
    if bodies:
        return [_parse_single(b, tools) for b in bodies]
    return _parse_single(text, tools)


def normalize_messages(messages: Any) -> Any:
    """Give every assistant message a reasoning field. The template raises on
    an assistant turn that has none, as a replayed client history often is."""
    if not isinstance(messages, list):
        return messages
    out, changed = [], False
    for msg in messages:
        if (isinstance(msg, dict) and msg.get("role") == "assistant"
                and not any(f in msg for f in _THINKING_FIELDS)):
            msg = {**msg, "reasoning_content": ""}
            changed = True
        out.append(msg)
    return out if changed else messages


def install_message_normalizer(tokenizer) -> None:
    """Run ``normalize_messages`` ahead of every chat-template render on
    this tokenizer. Idempotent."""
    original = getattr(tokenizer, "apply_chat_template", None)
    if original is None or wrapped_with(original, "_k2_horizon_normalized"):
        return

    def apply_chat_template(conversation, *args, **kwargs):
        return original(normalize_messages(conversation), *args, **kwargs)

    apply_chat_template._k2_horizon_normalized = True
    tokenizer.apply_chat_template = link(apply_chat_template, original)


def ensure_registered() -> None:
    """Register this module as the ``k2_horizon`` tool parser, ahead of any
    vendored parser of that name, and route templates carrying the K2
    Horizon tags to it. Idempotent."""
    from gmlx.upstream.tool_parsers import register_tool_parser

    register_tool_parser("k2_horizon", sys.modules[__name__], _TEMPLATE_MARKERS)
