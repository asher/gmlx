"""DeepSeek-V4.1 DSML tool-call parser.

V4.1 spells its tags with a leading space inside the DSML token, which is
what separates it from V4::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

``｜DSML｜`` is one vocab token and carries no angle brackets, so every tag
above is plain text around it. The markers here are therefore decoded-text
markers, never a special-token stop.

``string="true"`` means the value is literal text; anything else is JSON,
which is how the reference ``decode_dsml_to_arguments`` reads it back.

Parser contract (mlx-vlm ``tool_parsers``): module attributes
``tool_call_start`` / ``tool_call_end`` delimit the block in generated text,
and ``parse_tool_call(text, tools)`` takes the inner text and returns
``{"name", "arguments"}`` or a list of them.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

_DSML = "｜DSML｜"

tool_call_start = f"<{_DSML} calls>"
tool_call_end = f"</{_DSML} calls>"

_INVOKE_RE = re.compile(
    rf"<{re.escape(_DSML)} invoke\s+name=\"(.*?)\">(.*?)</{re.escape(_DSML)} invoke>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    rf"<{re.escape(_DSML)} parameter\s+name=\"(.*?)\"\s+string=\"(true|false)\">"
    rf"(.*?)</{re.escape(_DSML)} parameter>",
    re.DOTALL,
)

# The spaced spelling is the V4.1 marker; V4's unspaced tags must not match.
_TEMPLATE_MARKERS = [((tool_call_start,), "deepseek_v41")]


def _decode(value: str, is_string: str) -> Any:
    if is_string == "true":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _parse_single(name: str, body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for key, is_string, value in _PARAM_RE.findall(body):
        arguments[key] = _decode(value, is_string)
    # A namespaced call keeps its "ns::name" spelling, which the chat
    # template splits again on the way back in.
    return {"name": name.strip(), "arguments": arguments}


def parse_tool_call(text: str, tools: list[Any] | None = None):
    """Parse the inner text of one calls block."""
    calls = [_parse_single(n, b) for n, b in _INVOKE_RE.findall(text)]
    if calls:
        return calls
    return {"name": "unknown", "arguments": {"raw": text.strip()}}


def normalize_messages(messages: Any) -> Any:
    """Tool-call arguments as dicts, which is what the template renders
    one DSML parameter per key from. OpenAI clients send a JSON string."""
    if not isinstance(messages, list):
        return messages
    out, changed = [], False
    for msg in messages:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not calls:
            out.append(msg)
            continue
        new_calls = []
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if not isinstance(args, str):
                new_calls.append(call)
                continue
            try:
                parsed = json.loads(args)
            except ValueError:
                new_calls.append(call)
                continue
            if not isinstance(parsed, dict):
                new_calls.append(call)
                continue
            new_calls.append({**call, "function": {**fn, "arguments": parsed}})
            changed = True
        out.append({**msg, "tool_calls": new_calls})
    return out if changed else messages


def install_message_normalizer(tokenizer) -> None:
    """Run ``normalize_messages`` ahead of every chat-template render on
    this tokenizer. Idempotent."""
    original = getattr(tokenizer, "apply_chat_template", None)
    if original is None or getattr(original, "_ds41_normalized", False):
        return

    def apply_chat_template(conversation, *args, **kwargs):
        return original(normalize_messages(conversation), *args, **kwargs)

    apply_chat_template._ds41_normalized = True
    tokenizer.apply_chat_template = apply_chat_template


def ensure_registered() -> None:
    """Make ``mlx_vlm.tool_parsers.deepseek_v41`` resolve (upstream wins)
    and teach the template-marker inference the V4.1 spelling."""
    import importlib

    if "mlx_vlm.tool_parsers.deepseek_v41" not in sys.modules:
        try:
            importlib.import_module("mlx_vlm.tool_parsers.deepseek_v41")
        except ImportError:
            sys.modules["mlx_vlm.tool_parsers.deepseek_v41"] = sys.modules[__name__]
    try:
        registry = importlib.import_module("mlx_vlm.tool_parsers")
    except ImportError:
        return
    markers = getattr(registry, "_TEMPLATE_MARKERS", None)
    if isinstance(markers, list):
        for entry in _TEMPLATE_MARKERS:
            if entry not in markers:
                markers.insert(0, entry)
