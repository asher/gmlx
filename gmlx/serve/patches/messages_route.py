"""The /v1/messages conversion of Anthropic content blocks to chat messages.

mlx-vlm's ``_anthropic_messages_to_internal`` reduces each user message to
its text joined into one string and collects every image block into the
request's image list, so the render places every image in the last user
message, ahead of that message's text. A turn's prompt then does not start
with the previous turn's, and the model reads earlier images beside the
latest question. The claim here keeps the stock conversion and hands the
render a user message that holds an image as its text runs and one
``{"type": "image"}`` marker per image, in block order, as the Responses
route builds a message. The render counts the markers per message and
keeps them where the client sent them (``render._images_in_place``).
Text-only messages, tool_use and tool_result blocks convert as before.

The stock conversion also emits a user message's text ahead of its tool
results, so the text renders between the assistant's tool call and the
tool response, and the Gemma-4 templates drop the tool response. The claim
emits the tool results first, in block order, then the text, as the
Messages API orders a user message. A message of tool results alone
converts as before.
"""

from __future__ import annotations

import importlib
import os

from gmlx.upstream.patching import link, wrapped_with

_FLAG = "_gmlx_messages_route"
# The user message block types the stock conversion reads no text from,
# besides images.
_NO_TEXT = ("tool_result", "thinking", "redacted_thinking")


def _placed(an, content) -> tuple[list, list] | None:
    """A user message's text and one image marker per image block, in block
    order, and the refs of those images, or None when it holds no image the
    stock conversion keeps.

    Text blocks between two images join with a newline, as the stock
    conversion joins a message's text.
    """
    parts: list[dict] = []
    run: list[str] = []
    refs: list = []
    for raw in content or ():
        item = an._as_plain_dict(raw)
        if not isinstance(item, dict):
            if item is not None:
                run.append(str(item))
            continue
        kind = item.get("type")
        if kind == "image":
            ref = an._anthropic_image_source_to_ref(item.get("source"))
            if ref:
                if run:
                    parts.append({"type": "text", "text": "\n".join(run)})
                    run = []
                parts.append({"type": "image"})
                refs.append(ref)
        elif kind not in _NO_TEXT and item.get("text"):
            run.append(str(item["text"]))
    if run:
        parts.append({"type": "text", "text": "\n".join(run)})
    return (parts, refs) if refs else None


def _results_first(new: list, own: list) -> list:
    """A message's image refs with those of its tool results first, as the
    message renders them. ``new`` holds them in block order and ``own``
    the message's own images among them."""
    rest = list(own)
    tool = []
    for ref in new:
        if ref in rest:
            rest.remove(ref)
        else:
            tool.append(ref)
    return tool + list(own)


def install_messages_route() -> None:
    """Claim the Messages route's conversion step. Idempotent.

    Kill: ``GMLX_FAITHFUL_HISTORY=0``, with the render wraps it relies on.
    """
    if os.environ.get("GMLX_FAITHFUL_HISTORY") == "0":
        return
    an = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
    stock = an._anthropic_messages_to_internal
    if wrapped_with(stock, _FLAG):
        return

    def _anthropic_messages_to_internal(request):
        # The stock body, with a user message that holds an image converted
        # to its text runs and image markers, and its tool results first.
        images: list = []
        out: list[dict] = []
        system_text = an._anthropic_system_text(request.system)
        if system_text:
            out.append({"role": "system", "content": system_text})
        for message in request.messages:
            start = len(images)
            text, calls, results = an._anthropic_content_blocks_to_text_and_tools(
                message.role, message.content, images)
            # Tool results come only from a user message, and render ahead
            # of its text.
            out.extend(results)
            placed = (_placed(an, message.content)
                      if message.role == "user"
                      and isinstance(message.content, list) else None)
            # A message of tool results and images, with no text, still
            # renders its images.
            if text or calls or not results or placed is not None:
                if placed is not None:
                    text, own = placed
                    images[start:] = _results_first(images[start:], own)
                msg: dict = {"role": message.role, "content": text}
                if calls:
                    msg["tool_calls"] = calls
                    if not text:
                        msg["content"] = ""
                out.append(msg)
        tools = an._anthropic_tools_to_openai(request.tools)
        tool_choice = an._anthropic_tool_choice_to_openai(request.tool_choice)
        out, tools, tool_choice = an._prepare_chat_tool_choice(
            out, tools, tool_choice)
        return out, images, tools, tool_choice

    an._anthropic_messages_to_internal = link(
        _anthropic_messages_to_internal, stock, _FLAG)
