"""Prompt upsampling: a chat model rewrites a typed prompt into the
caption a family's DiT was trained on, before ``encode_text`` reads it.

``chat_request`` builds the chat turn from the family's ``UpsampleParts``:
its system prompt, or the one in a file the user gives, the typed prompt
as the user turn, and its sampling. ``upsample`` sends it through
``chat``, the generation a surface supplies, and returns the caption the
family's parser takes from the reply. The rewrite runs outside the
pipeline, so the encoder, its conditioning cache and the DiT see a prompt
as they always do. It never runs on the image encoder, which has no
output head. The negative prompt is not rewritten. A prompt with
reference images runs as typed, except on a family whose upsampler sees
them (``UpsampleParts.pictures``): its turn holds each as a ``Picture``,
which the chat model reads through its vision tower.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any


class UpsampleError(RuntimeError):
    """The upsampler gave no caption, or a system prompt is missing or is
    not the one gmlx ships."""


@dataclass(frozen=True, eq=False)
class Picture:
    """A reference image the upsampler sees: ``pixels``, uint8 (H, W, 3)."""
    pixels: Any

    @classmethod
    def of(cls, image) -> "Picture":
        """The picture of a PIL image or a uint8 (H, W, 3 or 4) array."""
        import numpy as np

        if hasattr(image, "convert"):
            image = np.asarray(image.convert("RGB"))
        return cls(np.ascontiguousarray(np.asarray(image, np.uint8)[..., :3]))

    @property
    def size(self) -> tuple:
        """(width, height)."""
        return int(self.pixels.shape[1]), int(self.pixels.shape[0])

    @property
    def ratio(self) -> str:
        """Its shape as a reduced ``W:H``."""
        w, h = self.size
        g = math.gcd(w, h)
        return f"{w // g}:{h // g}"

    @functools.cached_property
    def digest(self) -> str:
        h = hashlib.sha256(repr(self.pixels.shape).encode())
        h.update(self.pixels.tobytes())
        return h.hexdigest()

    def image(self):
        """The picture as a PIL image, as the vision processor takes it."""
        from PIL import Image

        return Image.fromarray(self.pixels, "RGB")


@dataclass(frozen=True)
class ChatRequest:
    """One chat turn for the upsampler. ``messages`` are the system and
    user turns, ``thinking`` the value of the chat model's thinking
    switch. A turn that sees pictures has a user turn of content parts,
    text and ``{"type": "image"}`` in order, and ``pictures`` holds the
    images in that order. ``seed`` seeds a turn that samples, None for a
    greedy one or an unseeded one. ``thinking_budget`` closes the
    thinking after that many tokens, None for no budget."""
    messages: tuple
    max_tokens: int
    temperature: float
    presence_penalty: float
    thinking: str
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    pictures: tuple = field(default=())
    thinking_budget: int | None = None

    @property
    def system(self) -> str | None:
        """The system turn's text, None for a turn without one."""
        first = self.messages[0]
        return first["content"] if first["role"] == "system" else None

    @property
    def user(self):
        """The user turn: its text, or its content parts."""
        return self.messages[-1]["content"]


@functools.lru_cache(maxsize=None)
def _shipped(package: str, name: str, sha256: str, key: str | None) -> str:
    data = (resources.files(package) / name).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise UpsampleError(f"{name} of {package} has sha256 {got}, not "
                            f"the {sha256} gmlx ships; reinstall gmlx")
    return json.loads(data)[key] if key is not None else data.decode("utf-8")


def read_prompt_file(path: str, family=None) -> str:
    """The system prompt in the text file at ``path``, without the blank
    lines and spaces around it. For a family that ships none, the error
    a missing file raises says where to get one."""
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        needs = getattr(getattr(family, "upsample", None), "needs", "")
        raise UpsampleError(f"upsample prompt {path}: no such file"
                            + (f". {needs}" if needs else "")) from None
    except (OSError, UnicodeDecodeError) as e:
        raise UpsampleError(f"upsample prompt {path}: {e}") from None
    if not text:
        raise UpsampleError(f"upsample prompt {path} is empty")
    return text


def system_prompt(family, path: str | None = None) -> str:
    """The system prompt of ``family`` for a prompt without references:
    the one in the file at ``path``, else the one gmlx ships. A family
    that ships none needs ``path``. A family with its own reader
    (``UpsampleParts.read``) gets the text that reader makes of the file,
    which its turn is built from."""
    parts = family.upsample
    if path:
        if parts.read is not None:
            return parts.read(path, family)
        return read_prompt_file(path, family)
    if parts.prompts is None:
        raise UpsampleError(parts.needs)
    return _shipped(parts.package, parts.prompts, parts.sha256, parts.key)


def takes_pictures(family) -> bool:
    """Whether the upsampler of ``family`` sees a prompt's reference
    images."""
    parts = getattr(family, "upsample", None)
    return parts is not None and parts.pictures


def chat_request(family, prompt: str, system: str | None = None, *,
                 seed: int | None = None, pictures=(),
                 canvas: tuple | None = None) -> ChatRequest:
    """The chat turn that rewrites ``prompt`` for ``family`` under
    ``system``, by default the system prompt gmlx ships. ``seed`` seeds a
    turn that samples. ``pictures`` are the prompt's reference images, in
    order, for a family whose upsampler sees them, and ``canvas`` the
    (width, height) the request asks for, None when the family sizes the
    image by its references."""
    parts = family.upsample
    if parts is None:
        raise ValueError(f"{family.title} has no prompt upsampling")
    pictures = tuple(pictures)
    if pictures and not parts.pictures:
        raise ValueError(f"the {family.title} upsampler sees no reference "
                         "images")
    if system is None:
        system = system_prompt(family)
    if parts.turn is not None:
        messages = tuple(parts.turn(system, prompt, pictures, canvas))
    else:
        user = prompt
        if parts.slot and parts.slot in system:
            # The typed prompt goes in the system prompt, and the user turn
            # is the single space its source sends.
            system, user = system.replace(parts.slot, prompt), " "
        messages = ({"role": "system", "content": system},
                    {"role": "user", "content": user})
    return ChatRequest(
        messages=messages, max_tokens=parts.max_tokens,
        temperature=parts.temperature,
        presence_penalty=parts.presence_penalty, thinking=parts.thinking,
        top_p=parts.top_p, top_k=parts.top_k, min_p=parts.min_p,
        seed=seed if parts.temperature > 0 else None, pictures=pictures,
        thinking_budget=parts.thinking_budget)


def _spans(reply: str, thinking: bool):
    from gmlx.tui.reasoning import ReasoningFilter

    filt = ReasoningFilter(start_in_thinking=thinking)
    spans = filt.feed(reply) + filt.flush()
    return spans, filt.mode


def after_thinking(reply: str) -> str | None:
    """The answer of ``reply`` without its reasoning, as the chat terminal
    reads it: think tags in each spelling it knows, harmony analysis and
    final channels, Gemma's thought channel, and ATEM headers that start
    with ``<|start|>assistant``. A reply that closes a thinking block it
    never opened, as when the chat template opened it, is read from inside
    that block. None when the reply ends in its reasoning, as when it ran
    out of tokens there; a reply with no reasoning is all answer."""
    spans, mode = _spans(reply, thinking=True)
    if mode != "reason":
        return "".join(t for t, m in spans if m == "answer")
    spans, mode = _spans(reply, thinking=False)
    if mode == "reason":
        return None
    return "".join(t for t, m in spans if m == "answer")


def reply_lead(rendered, tokenizer=None) -> str:
    """What the rendered chat prompt opened that the reply goes on from
    without writing it, so ``after_thinking`` reads the reply whole: the
    ATEM header ``<|start|>assistant`` when the prompt ends at it, or the
    open think tag of a thinking block the prompt opens."""
    from gmlx.gen.thinking_budget import prompt_open_think_tag
    from gmlx.tui.reasoning import prompt_opens_header

    if prompt_opens_header(rendered):
        return "<|start|>assistant"
    return prompt_open_think_tag(rendered, tokenizer=tokenizer) or ""


def plain(reply: str) -> str | None:
    """The caption of a reply that is the caption."""
    answer = after_thinking(reply)
    return None if answer is None else answer.strip()


def paragraph(reply: str) -> str | None:
    """The last paragraph with letters in it that a reply writes after its
    thinking block. Lines before it, such as a heading or reasoning written
    out of the block, are not the caption, nor is a rule such as ``---``
    after it."""
    answer = after_thinking(reply)
    if answer is None:
        return None
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", answer)]
    paras = [p for p in paras if any(c.isalpha() for c in p)]
    return paras[-1] if paras else None


def _objects(text: str) -> list:
    """Each JSON object in ``text``, last first."""
    dec = json.JSONDecoder()
    found = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found[::-1]


_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


def _unescape(text: str) -> str:
    """``text`` with the JSON string escapes read, a surrogate pair as the
    one character it spells, and a backslash before any other character
    kept as it is."""
    def one(m):
        if m.group(2):
            hi, lo = int(m.group(1), 16), int(m.group(2), 16)
            return chr(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
        c = m.group(3)
        if c.startswith("u"):
            return chr(int(c[1:], 16))
        return _ESCAPES.get(c, m.group(0))
    return re.sub(r"\\u(d[89ab][0-9a-f]{2})\\u(d[c-f][0-9a-f]{2})"
                  r"|\\(u[0-9a-fA-F]{4}|.)", one, text, flags=re.S | re.I)


def _loose_field(answer: str, name: str) -> str | None:
    """The string ``name`` of the last object in ``answer`` that strict
    JSON does not read, as when the model leaves the quotes around text in
    the image unescaped: from the opening quote after the key and its
    colon to the quote before the next key, or with no key after it, to
    the last quote before the object's closing brace."""
    keys = list(re.finditer(r'"%s"\s*:\s*"' % re.escape(name), answer))
    if not keys:
        return None
    start = keys[-1].end()
    nxt = re.compile(r'"\s*,\s*"\w+"\s*:').search(answer, start)
    if nxt:
        return _unescape(answer[start:nxt.start()])
    close = answer.rfind("}", start)
    if close == -1:
        return None
    end = answer.rfind('"', start, close)
    if end == -1:
        return None
    return _unescape(answer[start:end])


def _field_values(answer: str, name: str):
    """The values ``name`` may have in ``answer``, best first: its string
    in each JSON object, last object first, then the string a loose read
    finds, then, for an answer with no object in it, the answer itself."""
    for obj in _objects(answer):
        yield obj.get(name)
    yield _loose_field(answer, name)
    if "{" not in answer:
        yield answer


def json_field(name: str):
    """A parser that takes the string ``name`` from the last JSON object
    after the thinking block that has one. An object that strict JSON
    does not read gives it by a loose read. A reply with no object in it
    is the caption as written, and one with an object but no string
    ``name`` in it has no caption."""
    def parse(reply: str) -> str | None:
        answer = after_thinking(reply)
        if answer is None:
            return None
        for value in _field_values(answer, name):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    return parse


def field_or_plain(name: str):
    """A parser that takes the string ``name`` from a JSON object of the
    reply, as ``json_field`` does, and the reply itself as the caption
    when no object has one."""
    field = json_field(name)

    def parse(reply: str) -> str | None:
        return field(reply) or plain(reply)
    return parse


def upsample(family, prompt: str, chat, *, name: str = "the upsampler",
             system: str | None = None, seed: int | None = None,
             pictures=(), canvas: tuple | None = None) -> str:
    """The caption ``chat(request)`` writes for ``prompt``, as the
    family's parser takes it from the reply. ``name`` names the chat
    model in the error a reply without a caption raises; ``seed``,
    ``pictures`` and ``canvas`` go to ``chat_request``."""
    parts = family.upsample
    reply = chat(chat_request(family, prompt, system, seed=seed,
                              pictures=pictures, canvas=canvas)) or ""
    caption = (parts.parse or plain)(reply)
    if not caption:
        if not reply.strip():
            what = "an empty caption"
        elif after_thinking(reply) is None:
            what = f"no caption within {parts.max_tokens} tokens"
        else:
            what = "a reply with no caption in it"
        raise UpsampleError(f"{name} wrote {what} for this prompt; run it "
                            "without prompt upsampling, or set another "
                            "upsampler")
    return caption


def cache_key(upsampler, family, prompt: str, system: str | None = None, *,
              seed: int | None = None, pictures=(),
              canvas: tuple | None = None) -> tuple | None:
    """What a caption depends on: the upsampler, as a hashable that tells
    one chat model from another, the family's turn and parser, the
    system prompt by its sha256, and the typed prompt. A turn that
    samples adds its seed, and one that sees pictures their digests and
    the canvas. A greedy turn leaves the seed out, since any seed writes
    the same caption. None for a turn that samples without a seed, which
    writes a new caption each time."""
    if system is None:
        system = system_prompt(family)
    digest = hashlib.sha256(system.encode("utf-8")).hexdigest()
    key = (upsampler, family.upsample, digest, prompt)
    if family.upsample.temperature > 0:
        if seed is None:
            return None
        key += (seed,)
    if pictures:
        key += (tuple(p.digest for p in pictures), canvas)
    return key
