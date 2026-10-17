"""Prompt upsampling: a chat model rewrites a typed prompt into the
caption a family's DiT was trained on, before ``encode_text`` reads it.

``chat_request`` builds the chat turn from the family's ``UpsampleParts``:
its system prompt, or the one in a file the user gives, the typed prompt
as the user turn, and its sampling. ``upsample`` sends it through
``chat``, the generation a surface supplies, and returns the caption the
family's parser takes from the reply. The rewrite runs outside the
pipeline, so the encoder, its conditioning cache and the DiT see a prompt
as they always do. It never runs on the image encoder, which has no
output head. The negative prompt is not rewritten, and a prompt with
reference images runs as typed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from importlib import resources


class UpsampleError(RuntimeError):
    """The upsampler gave no caption, or a system prompt is missing or is
    not the one gmlx ships."""


@dataclass(frozen=True)
class ChatRequest:
    """One chat turn for the upsampler. ``messages`` are the system and
    user turns, ``thinking`` the value of the chat model's thinking
    switch."""
    messages: tuple
    max_tokens: int
    temperature: float
    presence_penalty: float
    thinking: str

    @property
    def system(self) -> str:
        return self.messages[0]["content"]

    @property
    def user(self) -> str:
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
    that ships none needs ``path``."""
    parts = family.upsample
    if path:
        return read_prompt_file(path, family)
    if parts.prompts is None:
        raise UpsampleError(parts.needs)
    return _shipped(parts.package, parts.prompts, parts.sha256, parts.key)


def chat_request(family, prompt: str, system: str | None = None
                 ) -> ChatRequest:
    """The chat turn that rewrites ``prompt`` for ``family`` under
    ``system``, by default the system prompt gmlx ships."""
    parts = family.upsample
    if parts is None:
        raise ValueError(f"{family.title} has no prompt upsampling")
    if system is None:
        system = system_prompt(family)
    user = prompt
    if parts.slot and parts.slot in system:
        # The typed prompt goes in the system prompt, and the user turn is
        # the single space its source sends.
        system, user = system.replace(parts.slot, prompt), " "
    return ChatRequest(
        messages=({"role": "system", "content": system},
                  {"role": "user", "content": user}),
        max_tokens=parts.max_tokens, temperature=parts.temperature,
        presence_penalty=parts.presence_penalty, thinking=parts.thinking)


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
    that block. None when the reply ends in its reasoning with no answer,
    as when it ran out of tokens there; a reply with no reasoning is all
    answer."""
    spans, mode = _spans(reply, thinking=True)
    if mode != "reason":
        return "".join(t for t, m in spans if m == "answer")
    spans, mode = _spans(reply, thinking=False)
    answer = "".join(t for t, m in spans if m == "answer")
    if mode == "reason" and not answer.strip():
        return None
    return answer


def plain(reply: str) -> str | None:
    """The caption of a reply that is the caption."""
    answer = after_thinking(reply)
    return None if answer is None else answer.strip()


def paragraph(reply: str) -> str | None:
    """The paragraph a reply ends with after its thinking block. Lines
    before it, such as a heading or reasoning written out of the block,
    are not the caption."""
    answer = after_thinking(reply)
    if answer is None:
        return None
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", answer)]
    paras = [p for p in paras if p]
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
    """``text`` with the JSON string escapes read, and a backslash before
    any other character kept as it is."""
    def one(m):
        c = m.group(1)
        if c.startswith("u"):
            return chr(int(c[1:], 16))
        return _ESCAPES.get(c, m.group(0))
    return re.sub(r"\\(u[0-9a-fA-F]{4}|.)", one, text, flags=re.S)


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
             system: str | None = None) -> str:
    """The caption ``chat(request)`` writes for ``prompt``, as the
    family's parser takes it from the reply. ``name`` names the chat
    model in the error a reply without a caption raises."""
    parts = family.upsample
    reply = chat(chat_request(family, prompt, system)) or ""
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


def cache_key(upsampler, family, prompt: str, system: str | None = None
              ) -> tuple:
    """What a caption depends on: the upsampler, as a hashable that tells
    one chat model from another, the family's turn and parser, the
    system prompt by its sha256, and the typed prompt. Decoding is
    greedy, so the seed is not in it."""
    if system is None:
        system = system_prompt(family)
    digest = hashlib.sha256(system.encode("utf-8")).hexdigest()
    return (upsampler, family.upsample, digest, prompt)
