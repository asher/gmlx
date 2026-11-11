# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Spoken replies in ``gmlx chat``: the ``/speak`` command.

A spoken turn runs the model's interleaved mode on the conversation, one
slot a step. The text prints as a text reply does. The audio goes to a
:class:`Player`, which plays it on a thread of its own, and to the clip a
:class:`SpokenReply` saves.

The conversation keeps a spoken reply as its text, so a later turn reads
it as any text reply.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

USAGE = "/speak [on|off|save DIR]"


class SpeakUnavailable(RuntimeError):
    """Sound cannot be played here. The text says what to do."""


def pcm16(audio: np.ndarray) -> np.ndarray:
    """Float samples as 16-bit samples."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


class Player:
    """Plays the audio of a reply in the order it is fed, on its own
    thread, so the reply keeps generating while it plays.

    ``backend`` has ``play(pcm, rate, stop=event)``, which returns early
    once ``event`` is set (``gmlx.talk.audio.SoundDeviceBackend``)."""

    def __init__(self, backend):
        self._backend = backend
        self._queue: queue.Queue = queue.Queue()
        self._cut = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="chat-speech", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                pcm, rate, cut = item
                if not cut.is_set():
                    self._backend.play(pcm, rate, stop=cut)
            except Exception:  # noqa: BLE001  a device error ends the sound, not the chat
                self._cut.set()
            finally:
                self._queue.task_done()

    def begin(self) -> None:
        """Start a new reply: what the last one still has queued is
        dropped and its sound is cut."""
        self.stop()
        self._cut = threading.Event()

    def feed(self, audio: np.ndarray, rate: int) -> None:
        self._queue.put((pcm16(audio), int(rate), self._cut))

    def stop(self) -> None:
        """Cut the sound and drop what is queued."""
        self._cut.set()

    def drain(self) -> None:
        """Return once everything fed so far is played or dropped."""
        self._queue.join()

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        self._thread.join()


def open_player() -> Player:
    """A player on the default output device. Raises SpeakUnavailable
    without ``sounddevice``, which the talk extra installs."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        from gmlx.commands.extras import install_hint

        raise SpeakUnavailable(
            "playing sound needs the talk extra: "
            f"{install_hint('talk')}") from None
    from gmlx.talk.audio import SoundDeviceBackend

    try:
        return Player(SoundDeviceBackend())
    except Exception as e:  # noqa: BLE001  PortAudio reports its own error types
        raise SpeakUnavailable(f"no sound output: {e}") from None


@dataclass
class Chunk:
    """What the chat's reply printer reads of a streamed chunk."""

    text: str = ""
    generation_tokens: int = 0
    generation_tps: float = 0.0
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    finish_reason: str | None = None


@dataclass
class SpokenReply:
    """One spoken turn. :meth:`chunks` turns the model's speech events
    into the reply printer's chunks. Each piece of audio goes to
    ``player`` when there is one, and is kept for :meth:`save`."""

    rate: int
    player: Any = None
    clock: Any = time.perf_counter
    audio: list = field(default_factory=list)
    frames: int = 0

    def chunks(self, events) -> Iterator[Chunk]:
        if self.player is not None:
            self.player.begin()
        slots, start = 0, None
        for event in events:
            if event.audio is not None and event.audio.size:
                self.audio.append(event.audio)
                if self.player is not None:
                    self.player.feed(event.audio, self.rate)
            if event.finish_reason is not None:
                # The closing summary, with the reply's own counts.
                self.frames = int(event.frames)
                slots = int(event.text_tokens + event.frames)
                yield Chunk(
                    event.text, slots,
                    slots / max(event.generation_seconds, 1e-9),
                    int(event.prompt_tokens),
                    event.prompt_tokens / max(event.prompt_seconds, 1e-9),
                    event.finish_reason)
                return
            now = self.clock()
            start = now if start is None else start
            slots += 1
            yield Chunk(event.text, slots, slots / max(now - start, 1e-9))

    @property
    def seconds(self) -> float:
        return sum(a.size for a in self.audio) / self.rate

    def cancel(self) -> None:
        if self.player is not None:
            self.player.stop()

    def save(self, folder: str) -> str | None:
        """Write the reply's audio to the next free ``reply-NNN.wav`` in
        ``folder``. Returns the path, or None for a reply with no audio."""
        if not self.audio:
            return None
        from gmlx.commands.speech_run import write_wav

        os.makedirs(folder, exist_ok=True)
        n = 1
        while os.path.exists(path := os.path.join(folder, f"reply-{n:03d}.wav")):
            n += 1
        write_wav(path, np.concatenate(self.audio), self.rate)
        return path


def messages(history: list, system: str) -> list:
    """``history`` for a spoken turn: the model's own instruction to speak
    leads it when the conversation has no system message."""
    if any(isinstance(m, dict) and m.get("role") == "system" for m in history):
        return history
    return [{"role": "system", "content": system}, *history]


def command(arg: str, state, opener=None) -> None:
    """``/speak``: show or set whether replies are spoken. ``opener``
    makes the player, :func:`open_player` by default."""
    opener = opener or open_player
    words = arg.split(None, 1)
    verb = words[0].lower() if words else ""
    if state.speak_refusal:
        print(f"[chat] /speak: {state.speak_refusal}")
        return
    if verb == "on":
        if state.speak_player is None:
            try:
                state.speak_player = opener()
            except SpeakUnavailable as e:
                print(f"[chat] /speak: {e}")
                print("[chat] '/speak save DIR' writes each reply to a WAV "
                      "file instead")
                return
        state.speak_play = True
    elif verb == "save":
        if len(words) < 2:
            print(f"[chat] usage: {USAGE}")
            return
        state.speak_dir = os.path.abspath(os.path.expanduser(words[1].strip()))
    elif verb == "off":
        state.speak_play, state.speak_dir = False, None
        if state.speak_player is not None:
            state.speak_player.stop()
    elif verb:
        print(f"[chat] usage: {USAGE}")
        return
    print(f"[chat] {describe(state)}")


def describe(state) -> str:
    if not (state.speak_play or state.speak_dir):
        return "replies are text ('/speak on' has the model speak them)"
    parts = []
    if state.speak_play:
        parts.append("played")
    if state.speak_dir:
        parts.append(f"saved to {state.speak_dir}")
    return "replies are spoken and " + " and ".join(parts)
