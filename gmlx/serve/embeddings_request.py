"""The fields of a ``POST /v1/embeddings`` request, checked and grouped.

A request gives its inputs in ``input`` or ``messages``, in the forms
vLLM's embeddings API takes:

- ``input`` as a string: one vector.
- ``input`` as a part dict, such as ``{"type": "image_url", ...}``: one
  item, one vector.
- ``input`` as a list: a list of chat messages is one conversation; a list
  of such lists is a batch of conversations; any other list holds one item
  per element, a string, a part dict or a list of parts.
- ``messages``: one conversation, or a batch of them.

An item is a list of text strings and :class:`MediaRef` parts. A
conversation becomes one item as the model's chat template renders it:
the system messages' text first, then each other message's parts, and a
placeholder for each media part of a message whose text names none.

This module reads no media and loads no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embeddings_errors import EmbeddingsRequestError

IMAGE_PARTS = ("image_url", "input_image", "image")
AUDIO_PARTS = ("input_audio", "audio_url", "audio")
VIDEO_PARTS = ("video_url", "input_video", "video")
_PLACEHOLDERS = {"image": "<|image|>", "audio": "<|audio|>", "video": "<|video|>"}

ENCODING_FORMATS = ("float", "base64")


def _bad(message: str) -> EmbeddingsRequestError:
    return EmbeddingsRequestError(400, message)


@dataclass(frozen=True)
class MediaRef:
    """A media part as the request gives it: ``kind`` is ``image``,
    ``audio`` or ``video``; ``ref`` is a reference string (a data URI, a URL
    or a file in the media folder), or ``data`` the bytes of inline base64.
    ``path`` marks a reference that must be a file. ``field`` names the
    part in messages."""

    kind: str
    field: str
    ref: str | None = None
    data: bytes | None = None
    path: bool = False


@dataclass
class Item:
    """One input to embed: text strings and media, in order."""

    parts: list
    chat: bool = False

    @property
    def media(self) -> list[MediaRef]:
        return [p for p in self.parts if isinstance(p, MediaRef)]


@dataclass
class Request:
    items: list[Item]
    encoding_format: str = "float"
    dimensions: int | None = None
    prompt_name: str | None = None
    mm_processor_kwargs: dict | None = None
    add_special_tokens: bool = True
    chat: bool = False
    ignored: list = field(default_factory=list)

    @property
    def has_media(self) -> bool:
        return any(item.media for item in self.items)


# Parts

def _str(value, where: str) -> str:
    if not isinstance(value, str):
        raise _bad(f"{where} must be a string")
    return value


def _url(value, where: str) -> str:
    """``value``, or its ``url`` when it is an object, as a string."""
    if isinstance(value, dict):
        value = value.get("url")
        where += ".url"
    return _str(value, where)


def _base64(value, where: str) -> bytes:
    from .patches import media_gate as mg

    try:
        return mg.b64decode(_str(value, where), validate=True)
    except ValueError:
        raise _bad(f"{where} must be base64 data") from None


def _keyed(d: dict, kind: str, where: str, own: str | None = None) -> MediaRef:
    """A Hugging Face style part: its own key ``own``, or ``url``, ``path``
    or ``base64``. Exactly one must be given."""
    keys = [k for k in (own, "url", "path", "base64") if k and d.get(k) is not None]
    if len(keys) != 1:
        names = ", ".join(k for k in (own, "url", "path", "base64") if k)
        raise _bad(f"a {where} part takes one of {names}")
    key = keys[0]
    if key == "base64":
        return MediaRef(kind, f"{where}.base64", data=_base64(d[key], f"{where}.base64"))
    return MediaRef(kind, f"{where}.{key}", ref=_str(d[key], f"{where}.{key}"),
                    path=key == "path")


def _image_part(d: dict, kind: str) -> MediaRef:
    if kind in ("image_url", "input_image"):
        if d.get("file_id") is not None:
            raise _bad(f"{kind}.file_id is not supported. Send the image as a "
                       "data: URI.")
        return MediaRef("image", kind, ref=_url(d.get("image_url"), f"{kind}.image_url"))
    source = d.get("source")
    if isinstance(source, dict):
        if source.get("type") == "base64":
            return MediaRef("image", "image.source.data",
                            data=_base64(source.get("data"), "image.source.data"))
        if source.get("type") == "url":
            return MediaRef("image", "image.source.url",
                            ref=_str(source.get("url"), "image.source.url"))
        raise _bad(f"image.source.type {source.get('type')!r} is not supported")
    return _keyed(d, "image", "image", "image")


def _audio_part(d: dict, kind: str) -> MediaRef:
    if kind == "input_audio":
        audio = d.get("input_audio")
        if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
            raise _bad("an input_audio part needs input_audio.data")
        data = audio["data"].strip()
        if data.startswith("data:"):
            return MediaRef("audio", "input_audio.data", ref=data)
        return MediaRef("audio", "input_audio.data",
                        data=_base64(data, "input_audio.data"))
    if kind == "audio_url":
        return MediaRef("audio", "audio_url", ref=_url(d.get("audio_url"), "audio_url"))
    return _keyed(d, "audio", "audio", "audio")


def _video_part(d: dict, kind: str) -> MediaRef:
    if kind in ("video_url", "input_video") and d.get("video_url") is not None:
        return MediaRef("video", kind, ref=_url(d["video_url"], f"{kind}.video_url"))
    if kind == "input_video" and d.get("video") is not None:
        return MediaRef("video", kind, ref=_url(d["video"], f"{kind}.video"))
    return _keyed(d, "video", kind, "video")


def part(value):
    """A text string or :class:`MediaRef` from one content part."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise _bad("a content part must be a string or an object with a type")
    kind = value.get("type")
    if kind == "text":
        return _str(value.get("text"), "text")
    if kind in IMAGE_PARTS:
        return _image_part(value, kind)
    if kind in AUDIO_PARTS:
        return _audio_part(value, kind)
    if kind in VIDEO_PARTS:
        return _video_part(value, kind)
    known = ", ".join(("text",) + IMAGE_PARTS + AUDIO_PARTS + VIDEO_PARTS)
    raise _bad(f"content part type {kind!r} is not supported (supported: {known})")


# Conversations

def _is_message(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get("role"), str)


def _is_conversation(value) -> bool:
    return isinstance(value, list) and bool(value) and all(map(_is_message, value))


def _is_batch(value) -> bool:
    return isinstance(value, list) and bool(value) and all(map(_is_conversation, value))


def _content(message: dict, where: str) -> list:
    content = message.get("content")
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise _bad(f"{where}.content must be a string or a list of parts")
    return [part(p) for p in content]


def render(messages: list) -> Item:
    """The item the model's chat template makes of a conversation: the
    system messages' text, then the other messages' parts. A media part of
    a message whose text names no placeholder adds its own placeholder, so
    every media part pairs with a placeholder in order."""
    system: list = []
    rest: list = []
    media: list = []
    for i, message in enumerate(messages):
        where = f"messages[{i}]"
        parts = _content(message, where)
        if message["role"] == "system":
            system += [p for p in parts if isinstance(p, str)]
            continue
        text = "".join(p for p in parts if isinstance(p, str))
        manual = any(ph in text for ph in _PLACEHOLDERS.values())
        for p in parts:
            if isinstance(p, str):
                rest.append(p)
            else:
                media.append(p)
                if not manual:
                    rest.append(_PLACEHOLDERS[p.kind])
    return Item(["".join(system + rest)] + media, chat=True)


def items(body: dict) -> tuple[list[Item], bool]:
    """The items of a request body and whether they came from a chat form."""
    has_input = body.get("input") is not None
    has_messages = body.get("messages") is not None
    if has_input == has_messages:
        raise _bad("send exactly one of the fields 'input' and 'messages'")
    if has_messages:
        value = body["messages"]
        if _is_conversation(value):
            return [render(value)], True
        if _is_batch(value):
            return [render(c) for c in value], True
        raise _bad("field 'messages' must be a list of chat messages, or a list "
                   "of such lists")
    value = body["input"]
    if isinstance(value, str):
        if not value.strip():
            raise _bad("field 'input' must not be empty")
        return [Item([value])], False
    if isinstance(value, dict):
        if _is_message(value):
            raise _bad("a chat message in field 'input' must be in a list")
        return [Item([part(value)])], False
    if not isinstance(value, list):
        raise _bad("field 'input' must be a string, a content part or a list")
    if not value:
        raise _bad("field 'input' must not be an empty list")
    if _is_conversation(value):
        return [render(value)], True
    if _is_batch(value):
        return [render(c) for c in value], True
    out = []
    for i, entry in enumerate(value):
        if isinstance(entry, bool) or isinstance(entry, (int, float)) or (
                isinstance(entry, list) and entry
                and all(isinstance(x, int) for x in entry)):
            raise _bad("field 'input' holds token ids, which are not supported. "
                       "Send text.")
        if _is_message(entry) or _is_conversation(entry):
            raise _bad("field 'input' mixes chat messages with other inputs")
        if isinstance(entry, list):
            if not entry:
                raise _bad(f"input[{i}] is an empty list")
            if any(_is_message(x) for x in entry):
                raise _bad("field 'input' mixes chat messages with other inputs")
            out.append(Item([part(x) for x in entry]))
        else:
            out.append(Item([part(entry)]))
    return out, False


# Other fields

# vLLM's fields that gmlx honors for one value or a few: the field, the
# values gmlx takes (None is the field left out), what gmlx does, and why
# another value is refused.
_PINNED = {
    "embed_dtype": ((None, "float32"), "float32",
                    "the vector bytes would be decoded wrongly"),
    "endianness": ((None, "native", "little"), "native or little",
                   "the vector bytes would be decoded wrongly"),
    "use_activation": ((None, True), "true",
                       "gmlx always normalizes the vector"),
    "truncate_prompt_tokens": ((None, -1), "-1",
                               "gmlx cuts a long input at the model's context"),
    "chat_template": ((None,), "unset",
                      "gmlx renders with the model's own template"),
    "padding": ((None, "do_not_pad"), "do_not_pad",
                "padding changes the vector"),
    "truncation_side": ((None, "right"), "right",
                        "gmlx keeps the start of a long input"),
    "media_io_kwargs": ((None,), "unset",
                        "loader options change the vector"),
    "priority": ((None, 0), "0", "gmlx has no priority scheduling"),
}

# Fields read for bookkeeping only.
_IGNORED = ("request_id", "cache_salt", "user")


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def _bool(body: dict, name: str, default: bool) -> bool:
    value = body.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _bad(f"field {name!r} must be true or false")
    return value


def parse(body: dict) -> Request:
    """The :class:`Request` of a JSON body. Raises
    :class:`EmbeddingsRequestError` (400) for a field gmlx does not take."""
    if not isinstance(body, dict):
        raise _bad("the request body must be a JSON object")
    for name, (honored, shown, why) in _PINNED.items():
        value = body.get(name)
        if not any(_same(value, h) for h in honored):
            raise _bad(f"{name} {value!r} is not supported: gmlx takes {shown}, "
                       f"since {why}")
    fmt = str(body.get("encoding_format") or "float").strip().lower()
    if fmt not in ENCODING_FORMATS:
        raise _bad(f"unsupported encoding_format {fmt!r} "
                   f"(supported: {', '.join(ENCODING_FORMATS)})")
    dims = body.get("dimensions")
    if dims is not None and (isinstance(dims, bool) or not isinstance(dims, int)):
        raise _bad("field 'dimensions' must be an integer")
    prompt_name = body.get("prompt_name")
    if prompt_name is not None:
        prompt_name = _str(prompt_name, "prompt_name")
    kwargs = body.get("mm_processor_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise _bad("field 'mm_processor_kwargs' must be an object")
    for name in ("add_generation_prompt", "continue_final_message"):
        _bool(body, name, False)
    if body.get("chat_template_kwargs") is not None and not isinstance(
            body["chat_template_kwargs"], dict):
        raise _bad("field 'chat_template_kwargs' must be an object")
    found, chat = items(body)
    return Request(items=found, encoding_format=fmt, dimensions=dims,
                   prompt_name=prompt_name, mm_processor_kwargs=kwargs,
                   add_special_tokens=_bool(body, "add_special_tokens", True),
                   chat=chat,
                   ignored=[k for k in _IGNORED if body.get(k) is not None])
