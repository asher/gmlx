"""Faithful chat-history rendering for the OpenAI server routes.

mlx-vlm's ``prompt_utils.apply_chat_template`` rebuilds every non-tool
message for model types in its ``MODEL_CONFIG`` table from ``content`` and
``role`` alone, discarding all other keys the client sent -- most notably
``reasoning_content``, which silently defeats ``preserve_thinking`` on
every non-tool turn (the chat template never sees the reasoning it is
asked to preserve). Tool-metadata messages take a passthrough branch and
are unaffected, which is why tool chains replay verbatim while plain
turns lose their reasoning. Model types outside the table already take a
key-preserving path, so which behavior a text request gets today depends
on the model_type string alone.

The wrapper keeps the upstream rebuild (role mapping, media token
placement, multimodal text extraction) and restores only the keys the
rebuild did not produce: it calls the stock function with
``return_messages=True``, merges each input message's unconsumed keys
back into its rebuilt counterpart (rebuilt keys win), and renders the
merged list through the same tail the stock function uses. The chat
template stays the second gate -- with ``preserve_thinking`` unset it
drops prior-turn reasoning exactly as before, so this restores the
documented option rather than changing default output.

The rebuild also joins a message's text into one part and puts every
image of the message before it (after it on Kimi), whatever order the
client sent. For the model types in ``_IN_PLACE_IMAGES`` the wrapper puts
the images back between the text parts where the client sent them
(``_images_in_place``), and again after the rebuild the Gemma-4
processor runs inside the render (``_install_nested_in_place``).

The Messages route sends a tool result's images as image markers in the
tool message. The rebuild counts the markers of user messages only, so it
also adds each tool image to the last user message. For the model types
in ``_TOOL_IMAGES``, whose template renders an image in a tool turn, the
wrapper leaves the tool images out of the count, in its rebuild and in the
one the Gemma-4 processor runs. For other types it moves them to a user
message of their own right after the tool results, as the Responses route
places a tool output's images, and hands the template the tool text as a
string (``_tool_images_to_user``). Each tool image then renders in one
place, and in the same place in every later turn. A text-only model type
gets the tool text alone.

The Mistral chat templates reject a user turn right after a tool turn, while
the Messages API sends text and images beside the tool results in one user
message, and chat and Responses clients can send a user message right
after a tool message too. When a template rejects the conversation, the
wrapper folds each user message that follows a tool message into the text
of that tool message, each image as the processor's image token, and renders
again (``_user_after_tool_folded``). A fold whose render loses a system
message's text raises the template's error instead
(``_keeps_system_text``).
Kill: ``GMLX_FAITHFUL_HISTORY=0``.
"""

from __future__ import annotations

import contextvars
import importlib
import json
import os

from jinja2.exceptions import TemplateError

from gmlx.upstream.patching import link, wrapped_with

# Stock tail: these model types return only the last message when a
# rendered prompt is requested.
_LAST_MESSAGE_ONLY = ("paligemma", "molmo", "florence2", "falcon_ocr")

# The model types whose chat template renders an image item where the
# content list holds it, and whose processor expands the image markers of
# the prompt in order, one image each. Other types keep the stock order:
# the llava-hf template renders every image ahead of the text itself.
_IN_PLACE_IMAGES = frozenset((
    "gemma4", "gemma4_unified",
    "qwen3_5", "qwen3_5_moe", "qwen4_exp", "qwen3_vl", "qwen3_vl_moe",
    "qwen3_omni_moe",
    "kimi_k25", "deepseek_v4_vl", "deepseek_v41_vl", "muse_glimmer",
    "muse_spark", "glm5_next", "mistral3", "pixtral", "nemotron_h_omni",
))
# Image items as the client and the stock rebuild write them.
_IMAGE_ITEMS = ("image", "image_url", "input_image")
# The model types whose chat template renders an image item of a tool
# message in that tool turn.
_TOOL_IMAGES = frozenset((
    "qwen3_5", "qwen3_5_moe", "kimi_k25", "muse_glimmer", "muse_spark",
    "glm5_next", "gemma4_unified", "nemotron_h_omni",
))
# The model types whose chat template raises on a JSON-string
# ``tool_calls[*].function.arguments``.
_DICT_TOOL_ARGUMENTS = frozenset(("muse_glimmer", "muse_spark", "k2_horizon",
                                  "beam"))


def _tool_images(message) -> int:
    """The image markers of ``message`` when it is a tool message."""
    if not isinstance(message, dict) or message.get("role") != "tool":
        return 0
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(isinstance(part, dict) and part.get("type") in _IMAGE_ITEMS
               for part in content)


def _tool_images_to_user(items: list, images: bool = True) -> list:
    """``items`` with each tool message's content as its text, joined as
    the route joins a tool result's text, and the images of each run of tool
    messages in a user message of their own right after the run, as the
    Responses route places a tool output's images. Without ``images`` (a
    model type with no image path, whose rebuild turns that message into an
    empty turn), the images are left out."""
    out: list = []
    moved: list = []
    for message in items:
        if isinstance(message, dict) and message.get("role") == "tool":
            if _tool_images(message):
                parts = [p for p in message["content"] if isinstance(p, dict)]
                if images:
                    moved += [p for p in parts
                              if p.get("type") in _IMAGE_ITEMS]
                text = "\n".join(str(p["text"]) for p in parts
                                 if p.get("type") == "text" and p.get("text"))
                message = {**message, "content": text.strip()}
            out.append(message)
            continue
        if moved:
            out.append({"role": "user", "content": moved})
            moved = []
        out.append(message)
    if moved:
        out.append({"role": "user", "content": moved})
    return out


def _as_text(content, image_token: str | None) -> str | None:
    """``content`` as one string with each image as ``image_token``, or
    None when it holds an image and there is no token, or a part of
    another kind."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    out = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _IMAGE_ITEMS and image_token:
            out.append(image_token)
        elif kind in _TEXT_PARTS:
            out.append(str(part.get("text", "") or part.get("content", "")))
        else:
            return None
    return "".join(out)


def _user_after_tool_folded(msgs: list,
                            image_token: str | None) -> list | None:
    """``msgs`` with each user message that follows a tool message folded
    into the text of that tool message, or None when there is none or one
    does not fold."""
    out: list = []
    for message in msgs:
        prev = out[-1] if out else None
        if (isinstance(message, dict) and message.get("role") == "user"
                and isinstance(prev, dict) and prev.get("role") == "tool"):
            head = _as_text(prev.get("content"), image_token)
            tail = _as_text(message.get("content"), image_token)
            if head is None or tail is None:
                return None
            out[-1] = {**prev,
                       "content": "\n\n".join(t for t in (head, tail) if t)}
            continue
        out.append(message)
    return out if len(out) < len(msgs) else None


def _keeps_system_text(rendered, msgs: list) -> bool:
    """Whether ``rendered`` holds the text of every system message of
    ``msgs``. The Mistral 7B template writes the system text into the last
    message only when that is a user message, so a fold that ends the
    conversation on a tool message drops it."""
    texts = []
    for message in msgs:
        if not isinstance(message, dict) or message.get("role") not in (
                "system", "developer"):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text = part.get("text") if isinstance(part, dict) else part
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if not texts:
        return True
    return isinstance(rendered, str) and all(t in rendered for t in texts)


def _images_in_place(built: dict, sent: dict) -> dict:
    """``built`` with its text and images in the order ``sent`` holds them.

    Text parts between two images join as the stock text extraction joins
    them, and the message's leading and trailing space is stripped as
    before. Audio parts stay where the rebuild put them, after the text.
    Returns ``built`` unchanged when its order already matches, when its
    image count differs from the images ``sent`` holds (a legacy side
    channel image has no place), or when it holds a part of another kind.
    """
    content, parts = built.get("content"), sent.get("content")
    if not isinstance(content, list) or not isinstance(parts, list):
        return built
    images, texts, audio = [], [], []
    for item in content:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind in _IMAGE_ITEMS:
            images.append(item)
        elif kind == "text":
            texts.append(item)
        elif kind == "audio":
            audio.append(item)
        else:
            return built
    if not images or len(texts) != 1:
        return built
    # The client's order as (is_image, text) pairs, one per image and one
    # per run of text parts between images.
    seq: list[tuple[bool, str]] = []
    run: list[str] = []
    for item in parts:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind in _IMAGE_ITEMS:
            if run:
                seq.append((False, " ".join(run)))
                run = []
            seq.append((True, ""))
        elif kind in _TEXT_PARTS:
            text = item.get("text", "") or item.get("content", "")
            if text:
                run.append(text)
    if run:
        seq.append((False, " ".join(run)))
    if sum(is_image for is_image, _ in seq) != len(images):
        return built
    runs = [i for i, (is_image, _) in enumerate(seq) if not is_image]
    if runs:
        seq[runs[0]] = (False, seq[runs[0]][1].lstrip())
        seq[runs[-1]] = (False, seq[runs[-1]][1].rstrip())
    seq = [(is_image, text) for is_image, text in seq if is_image or text]
    stock = [item.get("type") != "text" for item in content
             if item.get("type") != "audio"
             and (item.get("type") != "text" or item.get("text"))]
    if [is_image for is_image, _ in seq] == stock:
        return built
    pics = iter(images)
    out = []
    for is_image, text in seq:
        if is_image:
            out.append(next(pics))
            continue
        part = {**texts[0], "text": text}
        if "content" in part:
            part["content"] = text
        out.append(part)
    return {**built, "content": out + audio}


def _decode_tool_arguments(msgs: list) -> None:
    """Decode ``tool_calls[*].function.arguments`` from the OpenAI wire form
    (a JSON string) into a mapping, in place.

    The ``_DICT_TOOL_ARGUMENTS`` templates (the Muse ATEM ones and K2
    Horizon) raise outright on a string, since the jinja sandbox cannot parse
    JSON, so a client replaying its own tool turn would 500 on render. Undecodable arguments become an empty
    mapping, the same fallback the retirement mirror uses
    (``retire_key.assistant_message_from_reply``).
    """
    for i, m in enumerate(msgs):
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not isinstance(calls, list):
            continue
        rebuilt, changed = [], False
        for call in calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if not isinstance(args, str):
                rebuilt.append(call)
                continue
            try:
                decoded = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                decoded = {}
            if not isinstance(decoded, dict):
                decoded = {}
            rebuilt.append({**call, "function": {**fn, "arguments": decoded}})
            changed = True
        if changed:
            msgs[i] = {**m, "tool_calls": rebuilt}


def install_faithful_history() -> None:
    """Wrap every captured ``apply_chat_template`` binding with the key
    merge: the openai and anthropic protocol modules plus
    ``app._protocol_deps`` each hold their own reference.

    Install before ``install_retire_render_capture`` so the retirement
    memo captures this wrapper as its render callable -- next-turn
    predictions must see the same faithful render the server produces.
    """
    if os.environ.get("GMLX_FAITHFUL_HISTORY") == "0":
        return
    from ._common import _render_target_modules
    prompt_utils = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")

    def _wrap(orig):
        def apply_chat_template(processor, config, prompt,
                                add_generation_prompt=True,
                                return_messages=False,
                                num_images=0, num_audios=0, **kwargs):
            cfg = config if isinstance(config, dict) else config.__dict__
            tool_images = (sum(_tool_images(m) for m in prompt)
                           if isinstance(prompt, list) else 0)
            keep_tool_images = (tool_images > 0
                                and cfg.get("model_type") in _TOOL_IMAGES)
            if keep_tool_images:
                num_images = max(num_images - tool_images, 0)
            elif tool_images:
                # A model type outside MODEL_CONFIG takes the stock text
                # path, which has no image path.
                prompt = _tool_images_to_user(
                    prompt, str(cfg.get("model_type") or "").lower()
                    in prompt_utils.MODEL_CONFIG)
            msgs = orig(processor, config, prompt,
                        add_generation_prompt=add_generation_prompt,
                        return_messages=True, num_images=num_images,
                        num_audios=num_audios, **kwargs)
            if not isinstance(msgs, list):
                # A wrapped render that ignores return_messages (test
                # fakes, exotic overrides) already produced final output.
                return msgs
            items = prompt if isinstance(prompt, list) else [prompt]
            in_place = cfg.get("model_type") in _IN_PLACE_IMAGES
            # The stock rebuild appends one message per input item; items
            # it cannot read (no role/content) are skipped, breaking the
            # 1:1 mapping -- fall back to the unmerged list in that case.
            if len(msgs) == len(items):
                for i, (built, sent) in enumerate(zip(msgs, items)):
                    if isinstance(built, dict) and isinstance(sent, dict):
                        extra = {k: v for k, v in sent.items()
                                 if k not in built}
                        if extra:
                            built = msgs[i] = {**extra, **built}
                        if in_place:
                            msgs[i] = _images_in_place(built, sent)
            if cfg.get("model_type") == "gpt_oss":
                # The harmony template reads prior-turn reasoning from a
                # ``thinking`` key; alias the OpenAI-style spelling so
                # preserve_thinking works for standard clients.
                for i, m in enumerate(msgs):
                    if (isinstance(m, dict) and "thinking" not in m
                            and m.get("reasoning_content")):
                        msgs[i] = {**m, "thinking": m["reasoning_content"]}
            elif cfg.get("model_type") in _DICT_TOOL_ARGUMENTS:
                _decode_tool_arguments(msgs)
            if return_messages:
                return msgs
            if cfg.get("model_type") in _LAST_MESSAGE_ONLY:
                return msgs[-1]
            token = _IN_PLACE_SENT.set(
                items if in_place and len(msgs) == len(items) else None)
            kept = _TOOL_IMAGES_KEPT.set(keep_tool_images)
            try:
                try:
                    return prompt_utils.get_chat_template(
                        processor, msgs, add_generation_prompt, **kwargs)
                except TemplateError as exc:
                    # The template's own raise_exception, not a template bug.
                    folded = (_user_after_tool_folded(
                        msgs, getattr(processor, "image_token", None))
                        if type(exc) is TemplateError else None)
                    if folded is None:
                        raise
                    rejected = exc
                rendered = prompt_utils.get_chat_template(
                    processor, folded, add_generation_prompt, **kwargs)
                if not _keeps_system_text(rendered, msgs):
                    raise rejected
                return rendered
            finally:
                _TOOL_IMAGES_KEPT.reset(kept)
                _IN_PLACE_SENT.reset(token)

        return link(apply_chat_template, orig, "_kq_faithful_history")

    for target in _render_target_modules():
        fn = getattr(target, "apply_chat_template", None)
        if fn is not None and not wrapped_with(fn, "_kq_faithful_history"):
            target.apply_chat_template = _wrap(fn)
    _install_nested_in_place(prompt_utils)


# The client's messages of the in-place render running in this context.
_IN_PLACE_SENT: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_gmlx_in_place_sent", default=None)
# Whether the render running in this context keeps its tool images in the
# tool messages.
_TOOL_IMAGES_KEPT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_gmlx_tool_images_kept", default=False)
_NESTED_FLAG = "_gmlx_in_place_nested"


def _install_nested_in_place(prompt_utils) -> None:
    """Keep the images in place through a rebuild the processor nests.

    ``Gemma4Processor.apply_chat_template``, which ``get_chat_template``
    calls to render, runs the stock rebuild once more and renders its
    output with the tokenizer, so each message's images move ahead of its
    text again. That rebuild renders through the module's
    ``get_chat_template``. Wrapped, it puts each message's images back
    where the client sent them while an in-place render runs, and passes
    every other render through. The rebuild itself, wrapped, leaves the
    tool images out of its count while a render that keeps them runs.
    Idempotent.
    """
    stock_apply = prompt_utils.apply_chat_template
    if not wrapped_with(stock_apply, _NESTED_FLAG):
        def apply_chat_template(processor, config, prompt,
                                add_generation_prompt=True,
                                return_messages=False, num_images=0,
                                num_audios=0, **kwargs):
            if _TOOL_IMAGES_KEPT.get() and isinstance(prompt, list):
                num_images = max(
                    num_images - sum(_tool_images(m) for m in prompt), 0)
            return stock_apply(processor, config, prompt,
                               add_generation_prompt=add_generation_prompt,
                               return_messages=return_messages,
                               num_images=num_images, num_audios=num_audios,
                               **kwargs)

        prompt_utils.apply_chat_template = link(apply_chat_template,
                                                stock_apply, _NESTED_FLAG)
    original = prompt_utils.get_chat_template
    if wrapped_with(original, _NESTED_FLAG):
        return

    def get_chat_template(processor, messages, add_generation_prompt,
                          tokenize=False, **kwargs):
        sent = _IN_PLACE_SENT.get()
        if (sent is not None and isinstance(messages, list)
                and len(messages) == len(sent)):
            messages = [
                _images_in_place(m, s)
                if isinstance(m, dict) and isinstance(s, dict) else m
                for m, s in zip(messages, sent)]
        return original(processor, messages, add_generation_prompt,
                        tokenize=tokenize, **kwargs)

    prompt_utils.get_chat_template = link(get_chat_template, original,
                                          _NESTED_FLAG)


# Content part types, as mlx-vlm's text extraction reads them.
_IMAGE_PARTS = ("image_url", "input_image")
_TEXT_PARTS = ("text", "input_text", "output_text")
_PLACEMENT_FLAG = "_kq_chat_image_placement"


class _PlacedImages(list):
    """A chat completions user message's content whose images render in
    that message."""


def _placed_content(content) -> list:
    """Text parts and one ``{"type": "image"}`` marker per image, in
    content order, as the Responses route builds a message. The stock
    text extraction reads the same text from it."""
    out = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in _IMAGE_PARTS:
            out.append({"type": "image"})
        elif kind in _TEXT_PARTS:
            text = item.get("text", "") or item.get("content", "")
            if text:
                out.append({"type": "text", "text": text})
    return out


def install_chat_image_placement() -> None:
    """Render each image of a chat completions request in the user message
    that sent it, as the Responses route does (``messages_route`` does the
    same for the Messages route).

    The stock route reduces every message to its text
    (``extract_text_from_content``) and passes only the image count, so
    ``apply_chat_template`` places every image in the last user message.
    A turn's prompt then does not start with the previous turn's, and the
    model reads earlier images beside the latest question. A route wrap
    marks each user message whose content holds an image, and the chat
    route's text extraction returns the marked content as text parts and
    image markers, which the render counts per message. Text-only content
    takes the stock extraction. Idempotent.
    """
    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    stock = openai.extract_text_from_content
    if not wrapped_with(stock, _PLACEMENT_FLAG):
        def extract_text_from_content(content):
            if isinstance(content, _PlacedImages):
                return _placed_content(content)
            return stock(content)

        extract_text_from_content.__wrapped__ = stock  # type: ignore[attr-defined]
        setattr(extract_text_from_content, _PLACEMENT_FLAG, True)
        openai.extract_text_from_content = extract_text_from_content

    from ._common import _CHAT_PATHS, _wrap_post_routes

    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app").app

    def _make(original):
        async def endpoint(request, http_request):
            for message in getattr(request, "messages", None) or ():
                content = getattr(message, "content", None)
                if (getattr(message, "role", None) == "user"
                        and isinstance(content, list)
                        and any(isinstance(item, dict)
                                and item.get("type") in _IMAGE_PARTS
                                for item in content)):
                    message.content = _PlacedImages(content)
            return await original(request, http_request)
        return endpoint

    _wrap_post_routes(app, _CHAT_PATHS, _PLACEMENT_FLAG, _make)
