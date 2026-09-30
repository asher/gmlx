"""Upstream-bug fix: a Gemma-4 system prompt renders as a Python list repr.

mlx-vlm's ``prompt_utils.apply_chat_template`` builds every message of a
model type in its ``MODEL_CONFIG`` table through ``get_message_json``. For
the list formats that includes system and developer messages, whose text
becomes a list of text parts. The Gemma-4 template prints a leading system
or developer message with ``messages[0]['content'] | trim``, which renders
the list as its Python repr, so the model reads
``[{'type': 'text', ...}]`` in place of the system prompt. The template
handles lists in every other message, assistant history included.

The fix wraps ``get_chat_template``, the render tail that
``apply_chat_template`` and the server's faithful-history wrap both call.
It applies to a processor enabled for a model type in ``_MODEL_TYPES``
(``enable_system_content_fix``), which the VLM loader does, and on the
server the load bridge for every model it loads. Other templates read
list content for every role (idefics3, SmolVLM and the llava-hf ones
iterate it), so their messages pass through unchanged. A system or
developer message whose content holds only text parts reaches the
template as one string. The parts are joined the way the Gemma-4
template renders the text parts of a user message: each part trimmed,
with no separator. Content with a media part and every other role pass
through unchanged. Idempotent.
"""

from __future__ import annotations

import importlib

from gmlx.upstream.patching import wrapped_with

_FLAG = "_gmlx_system_content_as_text"
# Set on a processor the loader built for one of _MODEL_TYPES.
_PROCESSOR_FLAG = "_gmlx_system_as_text"
# The model types whose template prints the leading system message as a
# string: Gemma-4 with its mmproj, the unified Gemma-4 container, and
# DiffusionGemma, whose template uses the Gemma-4 turns.
_MODEL_TYPES = ("gemma4", "gemma4_unified", "diffusion_gemma")
_ROLES = ("system", "developer")
_TEXT_PARTS = ("text", "input_text")


def _joined_text(content) -> str | None:
    """The text of a non-empty list of text parts, else None."""
    if not isinstance(content, list) or not content:
        return None
    texts = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in _TEXT_PARTS:
            return None
        text = item.get("text") or item.get("content") or ""
        if not isinstance(text, str):
            return None
        texts.append(text.strip())
    return "".join(texts)


def enable_system_content_fix(processor, model_type: str | None) -> bool:
    """Turn the fix on for ``processor`` when the loaded model's type is in
    ``_MODEL_TYPES``. Returns whether it did."""
    if model_type not in _MODEL_TYPES or processor is None:
        return False
    try:
        setattr(processor, _PROCESSOR_FLAG, True)
    except (AttributeError, TypeError):
        return False
    return True


def install_system_content_fix() -> None:
    """Wrap ``mlx_vlm.prompt_utils.get_chat_template``. Idempotent; the
    module global is what both render call sites resolve at call time."""
    pu = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
    original = pu.get_chat_template
    if wrapped_with(original, _FLAG):
        return

    def get_chat_template(processor, messages, add_generation_prompt,
                          tokenize=False, **kwargs):
        if (getattr(processor, _PROCESSOR_FLAG, False) is True
                and isinstance(messages, list)):
            out = []
            for message in messages:
                if (isinstance(message, dict)
                        and message.get("role") in _ROLES):
                    text = _joined_text(message.get("content"))
                    if text is not None:
                        message = {**message, "content": text}
                out.append(message)
            messages = out
        return original(processor, messages, add_generation_prompt,
                        tokenize=tokenize, **kwargs)

    get_chat_template.__dict__[_FLAG] = True
    get_chat_template.__wrapped__ = original  # type: ignore[attr-defined]
    pu.get_chat_template = get_chat_template
