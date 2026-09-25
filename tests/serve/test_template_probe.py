"""The serve-load check for a chat template that drops list content.

For some model types the server sends message content to the template as a
list of parts. A template that renders only strings then gives the model
empty messages, so the load logs a warning that names the override."""

from __future__ import annotations

import logging

from jinja2.sandbox import ImmutableSandboxedEnvironment

import gmlx.serve.bridge_vlm as bridge

# Renders string content and drops anything else, like a text-only
# template shipped on a multimodal architecture.
_STRING_ONLY = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.content is string %}{{ m.content }}{% endif %}"
    "{% endfor %}<|assistant|>")

# Renders both forms, like the templates that ship with multimodal models.
_LIST_AWARE = (
    "{% for m in messages %}<|{{ m.role }}|>"
    "{% if m.content is string %}{{ m.content }}"
    "{% else %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endif %}"
    "{% endfor %}<|assistant|>")

_RAISES = "{{ raise_exception('no') }}"


class _Processor:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        env = ImmutableSandboxedEnvironment()

        def raise_exception(msg):
            raise ValueError(msg)

        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt)


def _warnings(caplog, template, model_type):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gmlx.serve.bridge_vlm"):
        bridge._warn_if_template_drops_text(
            "/models/finetune-Q4_K_M.gguf", _Processor(template),
            {"model_type": model_type})
    return [r.getMessage() for r in caplog.records]


def test_string_only_template_on_list_type_warns(caplog):
    [msg] = _warnings(caplog, _STRING_ONLY, "qwen3_5")
    assert "finetune-Q4_K_M.gguf" in msg
    assert "--chat-template" in msg
    assert "chat_template" in msg


def test_list_aware_template_is_quiet(caplog):
    assert _warnings(caplog, _LIST_AWARE, "qwen3_5") == []


def test_string_content_type_is_quiet(caplog):
    # Types outside MODEL_CONFIG get string content, so any template works.
    assert _warnings(caplog, _STRING_ONLY, "llama") == []


def test_render_error_is_quiet_and_does_not_raise(caplog):
    assert _warnings(caplog, _RAISES, "qwen3_5") == []
