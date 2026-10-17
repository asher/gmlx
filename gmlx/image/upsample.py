"""Prompt upsampling: a chat model rewrites a typed prompt into the
caption a family's DiT was trained on, before ``encode_text`` reads it.

``chat_request`` builds the chat turn from the family's ``UpsampleParts``:
its system prompt, the typed prompt as the user turn, and its sampling.
``upsample`` sends it through ``chat``, the generation a surface supplies,
and returns the caption. The rewrite runs outside the pipeline, so the
encoder, its conditioning cache and the DiT see a prompt as they always
do. It never runs on the image encoder, which has no output head. The
negative prompt is not rewritten, and a prompt with reference images
runs as typed.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from importlib import resources


class UpsampleError(RuntimeError):
    """The upsampler gave no caption, or its prompt file is not the one
    gmlx ships."""


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
def _prompt_file(family_name: str, name: str, sha256: str) -> dict:
    data = (resources.files(f"gmlx.image.families.{family_name}") / name
            ).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise UpsampleError(f"{name} of {family_name} has sha256 {got}, not "
                            f"the {sha256} gmlx ships; reinstall gmlx")
    return json.loads(data)


def system_prompt(family) -> str:
    """The system prompt of ``family`` for a prompt without references."""
    parts = family.upsample
    return _prompt_file(family.name, parts.prompts, parts.sha256)[parts.key]


def chat_request(family, prompt: str) -> ChatRequest:
    """The chat turn that rewrites ``prompt`` for ``family``."""
    parts = family.upsample
    if parts is None:
        raise ValueError(f"{family.title} has no prompt upsampling")
    return ChatRequest(
        messages=({"role": "system", "content": system_prompt(family)},
                  {"role": "user", "content": prompt}),
        max_tokens=parts.max_tokens, temperature=parts.temperature,
        presence_penalty=parts.presence_penalty, thinking=parts.thinking)


def upsample(family, prompt: str, chat, *, name: str = "the upsampler") -> str:
    """The caption ``chat(request)`` writes for ``prompt``, stripped.
    ``name`` names the chat model in the error an empty reply raises."""
    caption = (chat(chat_request(family, prompt)) or "").strip()
    if not caption:
        raise UpsampleError(f"{name} wrote an empty caption for this prompt; "
                            "run it without prompt upsampling, or set "
                            "another upsampler")
    return caption


def cache_key(upsampler: str, family, prompt: str) -> tuple:
    """What a caption depends on: the upsampler, the system prompt and the
    typed prompt. Decoding is greedy, so the seed is not in it."""
    return (str(upsampler), family.upsample.sha256, family.upsample.key,
            prompt)
