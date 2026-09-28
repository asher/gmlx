"""DeepSeek-V4 and V4.1 renders with list message content.

mlx-vlm sends list content for deepseek_v4 (from 0.7.3) and for the
deepseek_v4_vl and deepseek_v41_vl rows gmlx adds. The templates read a
list only on a user turn, so the render hooks hand every other turn its
text as a string. Each render runs the stock mlx-vlm formatter."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mlx_vlm.prompt_utils as prompt_utils

from gmlx.load.tokenizer import bundled_chat_template, finish_gguf_tokenizer
from gmlx.load.vlm import DEEPSEEK4V_IMAGE_TOKEN
from gmlx.models.deepseek_v4.chat import string_content

from test_vlm_deepseek41v import _processor as _processor_v41
from test_vlm_deepseek4v import _processor as _processor_v4
from test_vlm_deepseek4v import _tokenizer as _tokenizer_v4

IMG = DEEPSEEK4V_IMAGE_TOKEN
U, A, END = "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"

# The V4 GGUF template's shape: system turns concatenate as strings, user
# turns go through a list-aware media macro, assistant turns print content.
V4_TEMPLATE = (
    "{%- macro media(c) -%}{%- if c is string -%}{{ c }}{%- else -%}"
    "{%- for p in c -%}{%- if not loop.first -%}{{ '\\n\\n' }}{%- endif -%}"
    "{%- if p['type'] in ['image', 'image_url'] -%}" + IMG
    + "{%- else -%}{{ p['text'] }}{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{%- endmacro -%}"
    "{%- set ns = namespace(sp='') -%}{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}{%- set ns.sp = ns.sp + (m['content'] or '') -%}"
    "{%- endif -%}{%- endfor -%}{{- ns.sp -}}"
    "{%- for m in messages -%}{%- if m['role'] == 'user' -%}" + U
    + "{{ media(m['content']) }}{%- elif m['role'] == 'assistant' -%}" + A
    + "</think>{{ m['content'] }}" + END
    + "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}" + A + "</think>{%- endif -%}")

TURNS = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "what is this"},
]


def _render(processor, model_type, num_images=0):
    return prompt_utils.apply_chat_template(
        processor, {"model_type": model_type}, [dict(m) for m in TURNS],
        add_generation_prompt=True, num_images=num_images)


def test_string_content_leaves_user_turns_and_media_as_sent():
    user = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    media = {"role": "system", "content": [{"type": "image"},
                                           {"type": "text", "text": "x"}]}
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "a"},
                                       {"type": "input_text", "text": "b"}]},
        user,
        {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
        {"role": "tool", "content": ["d"]},
        media,
    ]
    out = string_content(msgs)
    assert [m["content"] for m in out[::2]] == ["a\n\nb", "c", media["content"]]
    assert out[1] is user and out[3]["content"] == "d"
    plain = [{"role": "system", "content": "a"}, user]
    assert string_content(plain) is plain


def test_text_model_renders_system_and_history_as_text():
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    raw, _ = _tokenizer_v4()
    raw.chat_template = V4_TEMPLATE
    finish_gguf_tokenizer(raw, "deepseek_v4")
    out = _render(TokenizerWrapper(raw), "deepseek_v4")
    assert out == (f"be brief{U}hi{A}</think>hi there{END}"
                   f"{U}what is this{A}</think>")


def test_v4_vision_renders_system_and_history_as_text():
    proc, _ = _processor_v4()
    proc.chat_template = V4_TEMPLATE
    out = _render(proc, "deepseek_v4_vl", num_images=1)
    assert out == (f"be brief{U}hi{A}</think>hi there{END}"
                   f"{U}{IMG}\n\nwhat is this{A}</think>")


def test_v41_vision_renders_system_and_history_as_text():
    proc, _ = _processor_v41()
    proc.chat_template = bundled_chat_template("deepseek_v41")
    out = _render(proc, "deepseek_v41_vl", num_images=1)
    assert "'type'" not in out
    assert "be brief" in out and "hi there" + END in out
    assert out.endswith(f"{U}{IMG}\n\nwhat is this{A}</think>")
