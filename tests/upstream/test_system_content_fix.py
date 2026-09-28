"""system_content_fix: a text-only system or developer message reaches the
chat template as a string.

mlx-vlm builds the messages of a Gemma-4 request as lists of text parts,
and the Gemma-4 template prints the leading system or developer message
with ``| trim``, which renders a list as its Python repr. Each render goes
through the stock ``apply_chat_template`` and a fixture template.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

import mlx_vlm.prompt_utils as pu  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from gmlx.upstream.system_content_fix import install_system_content_fix  # noqa: E402

_FIXTURES = Path(__file__).parents[1] / "fixtures"
_TEMPLATES = {"gemma4": "gemma4_template.jinja",
              "qwen3_5": "qwen35_template.jinja"}
_EXPECTED = {
    "gemma4": ("<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHi.<turn|>\n"
               "<|turn>model\nHello.<turn|>\n<|turn>user\nBye.<turn|>\n"
               "<|turn>model\n"),
    "qwen3_5": ("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n"
                "Hi.<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n"
                "<|im_start|>user\nBye.<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n"),
}


class _Processor:
    """Renders a fixture template the way a tokenizer does."""

    def __init__(self, model_type):
        self.chat_template = (_FIXTURES / _TEMPLATES[model_type]).read_text()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, bos_token="<bos>",
            **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


@pytest.fixture(autouse=True)
def installed(monkeypatch):
    monkeypatch.setattr(pu, "get_chat_template", pu.get_chat_template)
    install_system_content_fix()
    install_system_content_fix()


def _conversation(role):
    return [{"role": role, "content": "Be brief."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye."}]


@pytest.mark.parametrize("model_type", sorted(_TEMPLATES))
@pytest.mark.parametrize("role", ["system", "developer"])
def test_system_and_history_render_as_text(model_type, role):
    # enable_thinking is explicit: another test may have installed the
    # template-default wrap, which changes the qwen3.5 generation prompt.
    out = pu.apply_chat_template(_Processor(model_type),
                                 {"model_type": model_type},
                                 _conversation(role),
                                 add_generation_prompt=True,
                                 enable_thinking=False)
    assert out == _EXPECTED[model_type]


def test_only_text_only_system_and_developer_content_is_joined(monkeypatch):
    seen = []

    def render(processor, messages, add_generation_prompt, tokenize=False,
               **kwargs):
        seen.append(messages)
        return ""

    monkeypatch.setattr(pu, "get_chat_template", render)
    install_system_content_fix()
    parts = [{"type": "text", "text": "A. "}, {"type": "input_text", "text": "B."}]
    image = [{"type": "image"}, {"type": "text", "text": "C."}]
    pu.get_chat_template(None, [
        {"role": "system", "content": parts, "name": "s"},
        {"role": "developer", "content": image},
        {"role": "user", "content": parts},
        {"role": "assistant", "content": parts},
    ], True)
    assert seen == [[
        {"role": "system", "content": "A. B.", "name": "s"},
        {"role": "developer", "content": image},
        {"role": "user", "content": parts},
        {"role": "assistant", "content": parts},
    ]]
