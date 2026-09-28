# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Apple Inc. (mlx-lm, MIT)
# Vendored from mlx-vlm 0.6.15, mlx_vlm/tool_parsers/function_gemma.py, which
# is mlx-lm 0.31.3, mlx_lm/tool_parsers/function_gemma.py.
"""FunctionGemma tool-call parser.

FunctionGemma, a gemma3 text model, emits a call as
``<start_function_call>call:name{key:value,...}<end_function_call>``, with
string values wrapped in ``<escape>``. mlx-vlm 0.7.3 no longer ships this
parser. ``ensure_registered()`` grafts this module in as
``mlx_vlm.tools.parsers.function_gemma`` and routes templates carrying
``<start_function_call>`` to it, as mlx-vlm 0.6.15 did. The parser body is
the 0.6.15 one, unchanged.
"""

import json
import sys
from typing import Any, Optional

import regex as re

_TEMPLATE_MARKERS = (("<start_function_call>",),)

_tool_call_regex = re.compile(r"call:(\w+)\{(.*?)\}", re.DOTALL)


def parse_tool_call(text: str, _: Optional[Any] = None):
    match = _tool_call_regex.findall(text)
    if not match:
        raise ValueError("No function provided.")
    func_name = match[0][0]
    args_str = match[0][1]
    arguments = {}
    escape = "<escape>"
    while args_str:
        split = args_str.index(":")
        key = args_str[:split]
        args_str = args_str[split + 1 :]
        # Parse a string
        if args_str.startswith(escape):
            args_str = args_str[len(escape) :]
            split = args_str.index(escape)
            arguments[key] = args_str[:split]
            args_str = args_str[split + len(escape) + 1 :]
            continue
        if "," in args_str:
            split = args_str.index(",")
        else:
            split = len(args_str)

        value = args_str[:split]
        args_str = args_str[split + 1 :]

        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value

    return dict(name=func_name, arguments=arguments)


tool_call_start = "<start_function_call>"
tool_call_end = "<end_function_call>"


def ensure_registered() -> None:
    """Make ``mlx_vlm.tools.parsers.function_gemma`` resolve to this
    module and route templates carrying ``<start_function_call>`` to it.
    Idempotent."""
    from gmlx.upstream.tool_parsers import register_tool_parser

    register_tool_parser("function_gemma", sys.modules[__name__],
                         _TEMPLATE_MARKERS)
