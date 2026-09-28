# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Message content for the DeepSeek-V4 and V4.1 chat templates.

mlx-vlm sends message content as a list of parts for the model types in
its ``MODEL_CONFIG`` table: ``deepseek_v4`` from mlx-vlm 0.7.3, and the
``deepseek_v4_vl`` and ``deepseek_v41_vl`` rows gmlx adds so images reach
the placeholder. The V4 and V4.1 templates read a list only on a user turn.
A system turn concatenates its content as a string and raises on a list,
and an assistant turn renders the list itself into the prompt. The render
hooks here hand every other turn its text as a string first.
"""

from __future__ import annotations

from typing import Any

_TEXT_TYPES = ("text", "input_text")


def _text_of(parts: list) -> str | None:
    """The text of a list of text parts, or None when any part is not text.
    Parts join with a blank line, as both templates join user parts."""
    texts = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") in _TEXT_TYPES:
            texts.append(str(part.get("text") or ""))
        else:
            return None
    return "\n\n".join(texts)


def string_content(messages: Any) -> Any:
    """Messages with each non-user turn whose content is text parts only
    carrying that text as a string. User turns and media parts stay as
    sent. Returns the input object when nothing changes."""
    if not isinstance(messages, list):
        return messages
    out, changed = [], False
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list) and msg.get("role") != "user":
            text = _text_of(content)
            if text is not None:
                out.append({**msg, "content": text})
                changed = True
                continue
        out.append(msg)
    return out if changed else messages


def install_string_content(tokenizer) -> None:
    """Run ``string_content`` ahead of every chat-template render on this
    tokenizer. Idempotent."""
    original = getattr(tokenizer, "apply_chat_template", None)
    if original is None or getattr(original, "_ds4_string_content", False):
        return

    def apply_chat_template(conversation, *args, **kwargs):
        return original(string_content(conversation), *args, **kwargs)

    setattr(apply_chat_template, "_ds4_string_content", True)
    tokenizer.apply_chat_template = apply_chat_template
