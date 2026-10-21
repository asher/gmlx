"""Extract structured tool calls from a model's text output.

Format-agnostic: given the parser module selected for the model (see
:mod:`mlx_vlm.tools.registry`), it slices the text between the parser's markers,
delegates per-format decoding to ``parse_tool_call``, and emits OpenAI-shaped
calls. A module that defines ``find_tool_calls(text)`` finds its own call
spans, for formats whose calls have no closing marker of their own.
"""

import json
import logging
import re
import uuid

from .base import ToolParser
from .types import ParseResult

logger = logging.getLogger("gmlx._vendor.mlx_vlm.server")


def process_tool_calls(
    model_output: str, tool_module: ToolParser, tools
) -> ParseResult:
    """Parse tool calls from model output using the given tool parser module."""
    called_tools = []
    remaining = model_output

    # gmlx: a parser can find its own call spans (Beam's harmony XML calls).
    find_spans = getattr(tool_module, "find_tool_calls", None)
    if find_spans is not None:
        spans = find_spans(model_output)
        for start, end in spans:
            _append_calls(called_tools, model_output[start:end], tool_module, tools)
        if spans:
            kept, pos = [], 0
            for start, end in spans:
                kept.append(model_output[pos:start])
                pos = end
            kept.append(model_output[pos:])
            remaining = " ".join(k.strip() for k in kept if k.strip())
        return ParseResult(calls=called_tools, remaining_text=remaining)

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                _append_calls(called_tools, call, tool_module, tools)
    return ParseResult(calls=called_tools, remaining_text=remaining)


def _append_calls(called_tools: list, call: str, tool_module: ToolParser, tools) -> None:
    try:
        parsed = tool_module.parse_tool_call(call, tools)
        parsed_calls = parsed if isinstance(parsed, list) else [parsed]
        for tool_call in parsed_calls:
            args = tool_call["arguments"]
            called_tools.append(
                {
                    "type": "function",
                    "index": len(called_tools),
                    "id": str(uuid.uuid4()),
                    "function": {
                        "name": tool_call["name"].strip(),
                        "arguments": (
                            args
                            if isinstance(args, str)
                            else json.dumps(args, ensure_ascii=False)
                        ),
                    },
                },
            )
    except Exception as exc:
        logger.warning("Invalid tool call %r: %s", call, exc)
