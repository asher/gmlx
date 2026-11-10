# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Speaking rows in the batch engine.

A model with speech output fills each step of a reply with one slot: a
text token, or a frame of codes that its speech generator samples from
the hidden state. The engine holds one token per row per step, so a frame
slot reports the model's audio-out token, and the frame's codes and audio
ride in the row's record, :class:`SpeechRow`. The record is the ``speech``
field of the row's settings (:mod:`gmlx.serve.row_sampling`).

The model side is a hook on the loaded model:

- ``speech_start(last_token, mode, end_of_turn) -> (state, marker)``
- ``speech_logits(hidden, kinds)``: text logits with a mask per row
- ``speech_step(states, hidden, keys) -> (codes, embedding)``
- ``speech_audio(state, codes)`` and ``speech_flush(state)``

``state`` has ``take_token``, ``take_frame``, ``wants_frame``,
``finished`` and ``text_only``.

A batch with no speaking row does not come here. With one, the forward
evaluates each step before it builds the next: the kind of a slot depends
on the value of the slot before it.

A speaking row never requeues. A requeue replays a row from its prompt
ids and its delivered tokens, and a frame's token does not carry its
codes. So every path that ends a speaking row closes its record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import numpy as np

# Keeps a frame's key stream apart from the text draw at the same slot.
FRAME_SALT = 0x5EEC4A0D

MARKER, TEXT, FRAME, END = "marker", "text", "frame", "end"
MODES = ("audio", "interleaved")


@dataclass
class Slot:
    """What one token of a speaking row carries to the engine thread."""

    kind: str
    audio: Optional[np.ndarray] = None
    # No text: the forced marker, a frame, or a token that steers the reply.
    quiet: bool = False


class SpeechRow:
    """One speaking row's record: its reply state, the frame its last
    forward sampled, the codes of every frame, and the slots the engine
    thread has not read yet."""

    def __init__(self, model, state, marker: int, seed: int, *,
                 stop_at_end: bool = False, steering=()):
        self.model = model
        self.state = state
        # The mode marker, forced as the row's first token.
        self.marker = int(marker)
        self.seed = int(seed)
        # Text to speech ends at the end frame. A reply goes on in text.
        self.stop_at_end = bool(stop_at_end)
        self.steering = frozenset(int(t) for t in steering if t is not None)
        # The kind of the row's next input token.
        self.kind = MARKER
        self.codes_next: Any = None
        self.embedding: Any = None
        self.codes: list = []
        self.slots: deque = deque()
        self.closed = False

    def take(self, token: int) -> None:
        """Record the slot the row's input ``token`` stands for."""
        state, kind = self.state, self.kind
        if kind == MARKER:
            slot = Slot(MARKER, quiet=True)
        elif kind == FRAME:
            codes = np.asarray(self.codes_next).reshape(-1)
            if state.take_frame(codes):
                if self.stop_at_end:
                    state.finished = True
                slot = Slot(END, self.model.speech_flush(state), quiet=True)
            else:
                self.codes.append(codes)
                slot = Slot(FRAME, self.model.speech_audio(state, codes),
                            quiet=True)
        else:
            state.take_token(int(token))
            slot = Slot(TEXT, quiet=int(token) in self.steering)
        self.slots.append(slot)
        self.codes_next = None
        self.kind = TEXT

    def finish(self) -> Optional[np.ndarray]:
        """The samples the codec held back, when the reply ends with its
        audio open."""
        return self.model.speech_flush(self.state)

    def close(self) -> None:
        """Drop the codec stream, the kept frame and the unread slots.
        The codes stay until the row's settings go."""
        self.closed = True
        self.state.stream = None
        self.codes_next = self.embedding = None
        self.slots.clear()


def open_row(model, last_token: int, spec: dict, seed: Optional[int],
             end_of_turn=()) -> SpeechRow:
    """The record of a reply to a prompt that ends in ``last_token``.
    ``spec`` holds ``mode`` (``audio`` or ``interleaved``) and
    ``stop_at_end``. Raises ValueError for a model that does not speak,
    and for a prompt that already ends in a mode marker: the engine
    forces the marker as the row's first token."""
    capabilities = getattr(model, "speech_capabilities", lambda: None)()
    if capabilities is None:
        raise ValueError("this model has no speech output")
    mode = spec.get("mode")
    if mode not in MODES:
        raise ValueError(
            f"speech mode {mode!r} is not one of {', '.join(MODES)}")
    state, marker = model.speech_start(int(last_token), mode,
                                       end_of_turn=end_of_turn)
    if marker is None:
        raise ValueError("the prompt already ends in a speech marker")
    from gmlx._vendor.mlx_vlm.generate.dispatch import DEFAULT_SEED

    cfg = model.config.speech_config
    return SpeechRow(
        model, state, marker, DEFAULT_SEED if seed is None else seed,
        stop_at_end=bool(spec.get("stop_at_end")),
        steering=(cfg.audio_start_token_id, cfg.mixed_start_token_id,
                  cfg.text_end_token_id))


def record_of(settings) -> Optional[SpeechRow]:
    return getattr(settings, "speech", None)


def rows_of(batch) -> Optional[dict]:
    """``{row index: record}`` of the batch's open speaking rows, or None."""
    store = getattr(batch, "_row_settings", None)
    if not store:
        return None
    found = None
    for i, uid in enumerate(batch.uids):
        record = getattr(store.get(uid), "speech", None)
        if record is not None and not record.closed:
            if found is None:
                found = {}
            found[i] = record
    return found


def speaks(gen, uid) -> bool:
    """Whether ``uid`` is a speaking row of the generator ``gen``, open
    or closed."""
    store = getattr(gen, "_row_settings", None) or {}
    return getattr(store.get(uid), "speech", None) is not None


def close_row(gen, uid) -> None:
    store = getattr(gen, "_row_settings", None) or {}
    record = getattr(store.get(uid), "speech", None)
    if record is not None:
        record.close()


def forward(batch, inputs, fwd_kwargs: dict, speaking: dict):
    """The text model's step for a batch that holds speaking rows:
    ``(logits, hidden, quiet)``. ``logits`` is ``[rows, vocab]`` with each
    row's own mask, ``hidden`` is ``[rows, hidden]``, and ``quiet`` is the
    set of rows whose input was not a text slot.

    The inputs and the frames of the last step are evaluated first, and
    each speaking row's state takes the slot its input stands for."""
    lm = batch._language_model
    frames = [r.codes_next for r in speaking.values() if r.kind == FRAME]
    mx.eval(inputs, *frames)
    tokens = inputs.tolist()
    quiet, fed = set(), {}
    for i, record in speaking.items():
        if record.kind != TEXT:
            quiet.add(i)
        if record.kind == FRAME:
            fed[i] = record.embedding
        record.take(tokens[i])
        record.embedding = None

    embeddings = lm.model.embed_tokens(inputs)
    if fed:
        # A row whose last slot was a frame feeds that frame's embedding.
        embeddings = mx.concatenate([
            fed[i].astype(embeddings.dtype) if i in fed
            else embeddings[i:i + 1]
            for i in range(len(tokens))], axis=0)
    hidden = lm.model(inputs[:, None], batch.prompt_cache,
                      embeddings[:, None, :], **fwd_kwargs)[:, -1]
    model = next(iter(speaking.values())).model
    kinds = [speaking[i].state.text_only if i in speaking else None
             for i in range(len(tokens))]
    return model.speech_logits(hidden, kinds), hidden, quiet


def sample_frames(batch, sampled, hidden, speaking: dict):
    """A frame for each speaking row whose next slot is one. Returns
    ``sampled`` with the audio-out token in those rows. Each row draws
    from its own key, from its seed and its slot index."""
    want = [i for i, r in speaking.items() if r.state.wants_frame]
    if not want:
        return sampled
    from gmlx._vendor.mlx_vlm.generate.ar import _position_seed

    records = [speaking[i] for i in want]
    model = records[0].model
    keys = mx.stack([
        mx.random.key(_position_seed(
            (r.seed ^ FRAME_SALT) & 0xFFFFFFFF, 0, batch._num_tokens[i] + 1))
        for i, r in zip(want, records)])
    codes, embedding = model.speech_step(
        [r.state for r in records], hidden[mx.array(want)], keys)
    for j, record in enumerate(records):
        record.kind = FRAME
        record.codes_next = codes[j]
        record.embedding = embedding[j]
    mx.async_eval(codes, embedding)
    mask = np.zeros(sampled.shape[0], dtype=bool)
    mask[want] = True
    token = model.config.speech_config.audio_out_token_id
    return mx.where(mx.array(mask), mx.array(token, dtype=sampled.dtype),
                    sampled)


def finish_reason(record: SpeechRow, used: int, budget) -> Optional[str]:
    """Why a speaking row's reply ends at the token it just emitted. The
    stock stop test does not apply: a token of the model's end set can
    steer a spoken reply instead of ending it."""
    if record.state.finished:
        return "stop"
    if budget is not None and used >= budget:
        return "length"
    return None


def emit(record: SpeechRow, reason: Optional[str]) -> Slot:
    """The slot of the token a speaking row emits, with the codec's held
    samples when the reply ends there."""
    slot = record.slots.popleft() if record.slots else Slot(TEXT)
    if reason is not None:
        tail = record.finish()
        if tail is not None and tail.size:
            slot.audio = tail if slot.audio is None or not slot.audio.size \
                else np.concatenate([slot.audio, tail])
    return slot
