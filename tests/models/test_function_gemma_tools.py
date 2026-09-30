"""FunctionGemma tool-call parser (``gmlx.models.function_gemma``).

mlx-vlm 0.7.3 dropped the parser 0.6.15 shipped. A gemma3 text load
registers the vendored one, so a FunctionGemma template routes to it and a
call parses through the stock extractor. CPU-only, tiny model build.
"""

from __future__ import annotations

import json

_TEMPLATE = ("{% for m in messages %}<start_of_turn>{{m.role}}\n"
             "{{m.content}}<end_of_turn>{% endfor %}"
             "<start_function_call>call:x{}<end_function_call>")


def _tiny_gemma3_text():
    return dict(model_type="gemma3_text", hidden_size=16, num_hidden_layers=1,
                intermediate_size=32, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8, vocab_size=32,
                rms_norm_eps=1e-6, rope_theta=10000.0, sliding_window=16,
                sliding_window_pattern=2, query_pre_attn_scalar=8,
                max_position_embeddings=64)


def test_gemma3_text_load_routes_the_template_to_function_gemma():
    from gmlx._vendor.mlx_vlm.tools import registry

    from gmlx.load.loader import build_model

    build_model(_tiny_gemma3_text())
    assert registry._infer_tool_parser(_TEMPLATE) == "function_gemma"


def test_one_call_parses_through_the_stock_extractor():
    from gmlx._vendor.mlx_vlm.tools import process_tool_calls, registry

    from gmlx.load.loader import build_model

    build_model(_tiny_gemma3_text())
    module = registry.load_tool_module("function_gemma")
    out = ("<start_function_call>call:get_weather{city:<escape>Paris"
           "<escape>,days:3}<end_function_call>")
    result = process_tool_calls(out, module, None)
    assert len(result.calls) == 1
    function = result.calls[0]["function"]
    assert function["name"] == "get_weather"
    assert json.loads(function["arguments"]) == {"city": "Paris", "days": 3}
    assert result.remaining_text == ""
