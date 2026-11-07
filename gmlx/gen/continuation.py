"""Continue a final assistant message, as llama-server renders it.

A request that sets ``continue_final_message`` ends with a partial assistant
message, and the reply continues that message instead of opening a new turn.
llama.cpp picks a handler class from a string test on the chat template
source, renders ``messages[:-1]`` and appends a prefill that each class
writes its own way. This module ports the handler choice whole and the
prefill of three classes: Qwen3-Coder, gemma4 and the autoparser for the
templates a llama-server fixture covers. Every other class is refused.

Semantics follow llama.cpp at commit 8e5a58df87ab (MIT):
``common/chat.cpp`` (value parse, handler choice, mode resolution,
``render_content``, the generation prompt diff),
``common/parsers/qwen3-coder.cpp``, ``common/parsers/gemma4.cpp``,
``common/parsers/lfm2.cpp`` and ``common/chat-auto-parser-generator.cpp``.

Pure: no tokenizer, no MLX, no server imports.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import Callable, Literal

Requested = Literal["auto", "content", "reasoning_content"]
Mode = Literal["content", "reasoning_content"]

INVALID_VALUE = ('Invalid continue_final_message: expected a boolean, '
                 '"content" or "reasoning_content"')

# sha256 of each autoparser template a llama-server fixture covers
# (tests/fixtures/llama_continuation.json). Its rows show that the template
# has no reasoning markers, so the prefill is the generation prompt and the
# content.
AUTOPARSER_TEMPLATES = frozenset({
    # SmolLM2-135M-Instruct
    "872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788",
})


@dataclass(frozen=True)
class Continuation:
    """A resolved continuation of the final assistant message."""

    mode: Mode
    content: str
    reasoning: str
    handler: str


def parse_continue(value) -> Requested | None:
    """``continue_final_message`` as llama.cpp reads it: null and false
    give None, true gives "auto", and the two strings pass. Anything else,
    0 and 1 among them, raises ValueError."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "auto" if value else None
    if value in ("content", "reasoning_content"):
        return value
    raise ValueError(INVALID_VALUE)


def _get(message, key):
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def render_content(content) -> str:
    """The text of a message content: the string, or the text parts joined
    with a blank line. Other part types are skipped. No strip."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text = ""
    for part in content:
        if _get(part, "type") != "text":
            continue
        if text:
            text += "\n\n"
        text += _get(part, "text") or ""
    return text


def resolve(requested: Requested, message, handler: str = "") -> Continuation | None:
    """The continuation of ``message``, or None when it is empty (no
    content and no reasoning). "auto" picks reasoning only when the message
    has reasoning and no content: None, "" or an empty list. A list with any
    part continues the content, even with no text in it."""
    content = _get(message, "content")
    reasoning = _get(message, "reasoning_content") or ""
    no_content = content is None or content == "" or (
        isinstance(content, list) and not content)
    if no_content and not reasoning:
        return None
    if requested == "auto":
        mode: Mode = "reasoning_content" if (reasoning and no_content) else "content"
    else:
        mode = requested
    return Continuation(mode=mode, content=render_content(content),
                        reasoning=reasoning, handler=handler)


def is_lfm2_template(src: str) -> bool:
    return "<|tool_list_start|>" in src and "<|tool_list_end|>" in src


@functools.lru_cache(maxsize=32)
def handler_for(src: str) -> str:
    """llama.cpp's handler class for a template source. The first match
    wins, in llama.cpp's order; a template with no match takes the
    autoparser."""
    has = src.__contains__
    if (has("[SYSTEM_PROMPT]") and has("[TOOL_CALLS]") and has("[ARGS]")
            and not has("[CALL_ID]")):
        return "ministral-3"
    if has("chat_format=llm-jp-harmony-v1"):
        return "llm-jp-harmony"
    if has("<|channel|>") and has("<|constrain|>xml") and has("rfl:parameter"):
        return "harmony-xml"
    if has("<|channel|>"):
        return "gpt-oss"
    if has("to=commentary") and has("<|eom|>"):
        return "muse-spark"
    if has("<atem:function_calls>") and has("<|eom|>"):
        return "muse-glimmer"
    if has(">>>all") and has(">>>${recipient}"):
        return "functionary-v3-2"
    if has("<|tool_calls_section_begin|>") and has("<|tool_call_begin|>"):
        return "kimi-k2"
    if has("<|open|>") and has("<|close|>") and has("<|end_of_msg|>"):
        return "kimi-k3"
    if has("<|ifm|im_start|>") and has("<ifm|tool_calls>"):
        return "k2-horizon"
    if has("<role>ASSISTANT</role>") and has("<arg_key>"):
        return "ling3"
    if has("<|START_TEXT|>") and has("<|START_ACTION|>"):
        return "cohere2moe"
    if is_lfm2_template(src):
        return "lfm2"
    if has("List of tools: [") and not has("<|tool_list_start|>"):
        return "lfm2-5"
    if (has("<|role_sep|>") and has("<|message_sep|>")
            and not has("<|function_call|>")):
        return "gigachat-v3"
    if has("]<]minimax[>[") and has("<tool_call>") and has("<invoke name="):
        return "minimax-m3"
    if (has("dsml_token") and has("DSML")
            and (has("function_calls") or has("tool_calls"))):
        return "deepseek-v3-2"
    if has("'<|tool_call>call:'"):
        if not has("{#- OpenAI Chat Completions:"):
            return "gemma4-legacy"
        return "gemma4"
    if (has("Tool usage guidelines:") and has('<function name="')
            and has('<param name="')):
        return "minicpm5"
    if has("[source_lang_code]") and has("[target_lang_code]"):
        return "translate-gemma"
    if (has("<tool_call>") and has("<function=") and has("<parameter=")
            and not has("'<tool_call><function=' ~ tool_call.name ~ '>'")):
        return "qwen3-coder"
    return "autoparser"


def template_sha256(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def recipe_supported(handler: str, template: str, messages) -> bool:
    """Whether gmlx renders this handler's continuation. gemma4-legacy
    also rewrites tool responses, which gmlx does not port, so it is
    refused when the history holds a tool message."""
    if handler in ("qwen3-coder", "gemma4"):
        return True
    if handler == "gemma4-legacy":
        return not any(_get(m, "role") == "tool" for m in messages or ())
    if handler == "autoparser":
        return template_sha256(template) in AUTOPARSER_TEMPLATES
    return False


def _rendered(text) -> str:
    if not isinstance(text, str):
        raise TypeError(f"the render returned {type(text).__name__}, not a string")
    return text


def generation_prompt(render: Callable[[bool], str]) -> str:
    """The template's own generation prompt: the render with it, past the
    common prefix with the render without it."""
    off = _rendered(render(False))
    on = _rendered(render(True))
    n = 0
    limit = min(len(off), len(on))
    while n < limit and off[n] == on[n]:
        n += 1
    return on[n:]


def continued_prompt(cont: Continuation, template: str,
                     render: Callable[[bool], str]) -> str:
    """The prompt of ``messages[:-1]`` with the continuation appended.
    ``render(gen)`` renders ``messages[:-1]`` with or without the generation
    prompt. Raises TypeError when the render does not return a string, and
    ValueError for a handler with no recipe."""
    content_mode = cont.mode == "content"
    if cont.handler == "qwen3-coder":
        tail = "<|im_start|>assistant\n"
        if "<think>" in template:
            tail += "<think>\n" + cont.reasoning
            if content_mode:
                tail += "\n</think>\n\n"
        if content_mode:
            tail += cont.content
        return _rendered(render(False)) + tail
    if cont.handler in ("gemma4", "gemma4-legacy"):
        base = _rendered(render(False))
        tail = "<|turn>model\n" if base.endswith("<turn|>\n") else ""
        tail += "<|channel>thought\n" + cont.reasoning
        if content_mode:
            tail += "<channel|>" + cont.content
        return base + tail
    if cont.handler == "autoparser":
        base = _rendered(render(False))
        tail = generation_prompt(render)
        if content_mode:
            tail += cont.content
        return base + tail
    raise ValueError(f"no continuation recipe for the {cont.handler} handler")
