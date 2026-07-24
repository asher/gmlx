"""``gmlx talk`` - hands-free voice chat against a gmlx server.

The loop: mic -> wake word -> VAD endpointing -> ``/v1/audio/transcriptions``
-> streamed ``/v1/chat/completions`` -> chunked ``/v1/audio/speech`` ->
speaker. The server does all model work (talk is a client, auto-starting a
background server exactly like ``gmlx launch``); this module owns the audio
threads, the state machine, and the terminal UX.

Barge-in: Space/Esc stops playback within one write slice (~150 ms), and in
wake mode the mic keeps scoring the wake word (only - never transcribing)
while the assistant thinks or speaks, so the wake phrase interrupts a reply
hands-free. The follow-up utterance is a stop phrase ("stop", "never mind",
...) to just cancel, or a normal turn that replaces the interrupted one.
VAD/PTT modes stay half-duplex (an open mic would transcribe the assistant's
own voice); upgrade path there is a voice-processing (AEC) capture backend -
the AudioBackend seam in talk_audio exists for exactly that.

Threads: sounddevice callback (copy frame -> queue) -> listener (wake/VAD/
endpointer -> events) -> main loop (state machine, keyboard, status line)
-> turn worker (STT + brain -> sentence chunks) -> TTS worker (merge the
sentences decoded so far, synthesize while earlier audio plays) -> playback.
All coordination flows through one events queue; the state machine
(:class:`TalkStateMachine`) is pure and tested against synthetic event
sequences.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

# States
IDLE = "idle"                    # armed: wake word (or key) starts a turn
LISTENING = "listening"          # mic open, waiting for speech onset
CAPTURING = "capturing"          # inside an utterance
TRANSCRIBING = "transcribing"    # utterance posted to STT
THINKING = "thinking"            # brain turn streaming, nothing spoken yet
SPEAKING = "speaking"            # audio queued/playing

_REST = {"wake": IDLE, "vad": LISTENING, "ptt": IDLE, "text": IDLE}


class TalkStateMachine:
    """Pure coordination core: events in, (state, actions) out.

    Action strings the loop interprets: ``chime_wake``, ``chime_idle``,
    ``transcribe``, ``start_turn``, ``stop_speaking``, ``cancel_turn``,
    ``force_end``. ``turn_active`` tracks the brain stream, ``speaking`` the
    audio pipeline - a turn finishes only when both are done."""

    def __init__(self, mode: str = "wake"):
        if mode not in _REST:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.muted = False
        self.state = _REST[mode]
        self.turn_active = False
        self.speaking = False
        self.barged = False       # last wake interrupted a live reply

    # -- what the mic feed should do right now --------------------------------
    @property
    def mic_role(self) -> str | None:
        """None (gated) | "wake" (score wake word only) | "vad" (endpoint)."""
        if self.muted or self.mode == "text":
            return None
        if self.state in (LISTENING, CAPTURING):
            return "vad"
        if self.mode == "wake":
            # Wake-word-only scoring stays live through a reply so the wake
            # phrase barges in; the phrase-specific spotter won't fire on the
            # assistant's own voice unless it speaks the wake phrase itself.
            return "wake"
        return None                      # vad/ptt: half-duplex, gated while busy

    # -- mic-side events -------------------------------------------------------
    def on_wake(self) -> list:
        if self.mode != "wake" or self.muted:
            return []
        if self.state == IDLE:
            self.state = LISTENING
            return ["chime_wake"]
        if self.state in (TRANSCRIBING, THINKING, SPEAKING):
            # Voice barge-in: kill the live reply, then listen for the
            # follow-up - a stop phrase to just cancel, or a new turn.
            self.turn_active = False
            self.speaking = False
            self.barged = True
            self.state = LISTENING
            return ["stop_speaking", "cancel_turn", "chime_wake"]
        return []

    def on_speech_start(self) -> list:
        if self.state == LISTENING:
            self.state = CAPTURING
        return []

    def on_utterance(self) -> list:
        if self.state != CAPTURING:
            return []
        self.state = TRANSCRIBING
        return ["transcribe"]

    def on_drop(self) -> list:
        if self.state != CAPTURING:
            return []
        return self._rest()

    # -- turn-side events ------------------------------------------------------
    def on_stt(self, text: str) -> list:
        # No start_turn action: the transcribe thread continues straight into
        # the brain itself, so this event only advances the state.
        if self.state != TRANSCRIBING:
            return []
        self.barged = False
        if not text.strip():
            return self._rest()
        self.state = THINKING
        self.turn_active = True
        return []

    def text_turn(self, text: str) -> list:
        """A typed line becomes a turn from any at-rest state."""
        if self.turn_active or self.state in (TRANSCRIBING, THINKING,
                                              SPEAKING):
            return []
        if not text.strip():
            return []
        self.state = THINKING
        self.turn_active = True
        return ["start_turn"]

    def on_say(self) -> list:
        if self.state == THINKING:
            self.state = SPEAKING
        self.speaking = True
        return []

    def on_turn_done(self) -> list:
        self.turn_active = False
        # Mic-open states too, not just rest: a stale event from the killed
        # turn (e.g. a _TURN_END that slipped past the cancel-time drain)
        # must not yank a post-barge-in LISTENING/CAPTURING back to rest.
        if self.speaking or self.state in (IDLE, LISTENING, CAPTURING):
            return []
        return self._rest()

    def on_playback_idle(self) -> list:
        self.speaking = False
        if self.turn_active or self.state in (IDLE, LISTENING, CAPTURING):
            return []
        return self._rest()

    def on_cancel(self) -> list:
        """Space/Esc while busy: stop audio + close the brain stream."""
        if self.state not in (TRANSCRIBING, THINKING, SPEAKING):
            return []
        self.turn_active = False
        self.speaking = False
        return ["stop_speaking", "cancel_turn"] + self._rest(chime=False)

    # -- user controls ---------------------------------------------------------
    def on_ptt(self) -> list:
        if self.mode != "ptt" or self.muted:
            return []
        if self.state == IDLE:
            self.state = LISTENING
            return ["chime_wake"]
        if self.state == LISTENING:
            self.state = IDLE
            return ["chime_idle"]
        if self.state == CAPTURING:
            return ["force_end"]         # loop flushes the endpointer
        return []

    def on_hotkey(self) -> list:
        """Global tap-to-talk (menu bar hotkey): drive toward an open mic in
        any mode. Unlike ``on_wake``, a physical tap while muted unmutes -
        pressing the key is explicit intent to talk."""
        if self.mode == "text":
            return []                    # no VAD/endpointer built
        self.muted = False
        if self.state == IDLE:
            self.state = LISTENING
            return ["chime_wake"]
        if self.state == LISTENING:
            return self._rest()          # tap again dismisses (vad: no-op)
        if self.state == CAPTURING:
            return ["force_end"]
        # Busy (transcribing/thinking/speaking): barge in, then listen.
        self.turn_active = False
        self.speaking = False
        self.barged = True
        self.state = LISTENING
        return ["stop_speaking", "cancel_turn", "chime_wake"]

    def set_mode(self, mode: str) -> list:
        if mode not in _REST:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        if self.state in (IDLE, LISTENING, CAPTURING):
            self.state = _REST[mode]
        return []

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        if self.muted and self.state in (LISTENING, CAPTURING):
            self.state = _REST[self.mode]
        return self.muted

    # -- helpers ---------------------------------------------------------------
    def _rest(self, chime: bool = True) -> list:
        self.state = _REST[self.mode]
        self.barged = False
        return (["chime_idle"] if chime and self.mode == "wake" and
                not self.muted else [])


# Whisper is known to hallucinate these on (near-)silence; a short, quiet
# utterance transcribing to one of them is treated as empty, never as a turn.
_STT_GHOSTS = frozenset({
    "thank you", "thanks", "you", "bye", "thank you for watching", "the",
    "so", "okay", "hmm", "mm-hmm", "uh",
})


def looks_hallucinated(text: str, utterance_ms: float,
                       max_ms: float = 1500.0) -> bool:
    """True when a short clip transcribed to a known Whisper silence-ghost."""
    if utterance_ms > max_ms:
        return False
    t = text.strip().strip(".!?,").lower()
    return t in _STT_GHOSTS


# What counts as "just stop" after a wake-phrase barge-in. Only consulted for
# the utterance that follows an interrupted reply, so "stop" spoken as a
# normal question from idle still reaches the model.
_STOP_PHRASES = frozenset({
    "stop", "stop it", "stop talking", "cancel", "cancel that", "quiet",
    "be quiet", "shut up", "never mind", "nevermind", "forget it", "enough",
    "that's enough", "okay stop", "ok stop", "no stop",
})


def is_stop_phrase(text: str) -> bool:
    """True when the utterance is a bare cancel command."""
    t = " ".join(w.strip(".!?,;:") for w in text.lower().split())
    return t in _STOP_PHRASES


def _age(seconds: float) -> str:
    """Compact age for the /memory listing: 45s, 12m, 3h, 2d."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"


# Terminal plumbing

class _Keys:
    """cbreak keyboard polling for the main loop (chat.py's _EscCancel
    pattern), with suspend/resume so line input can borrow the terminal."""

    def __init__(self):
        self.fd = None
        self._saved = None

    def __enter__(self):
        if sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def poll(self) -> str | None:
        """One pending key ('' decoded), or None. An escape sequence (arrow
        keys, function keys) collapses to a bare ESC: its tail would otherwise
        arrive as printable bytes and open a line-input prompt."""
        import select
        if self.fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = os.read(self.fd, 1)
            if ch == b"\x1b":
                # 10ms lets a tail split from its ESC (slow link) arrive; a
                # bare Esc pays it once. CSI/SS3 tails are short; cap the drain.
                for _ in range(8):
                    if not select.select([sys.stdin], [], [], 0.01)[0]:
                        break
                    os.read(self.fd, 1)
                return "\x1b"
            try:
                return ch.decode()
            except UnicodeDecodeError:
                return None
        return None

    def suspend(self):
        import termios
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def resume(self):
        import tty
        if self.fd is not None:
            tty.setcbreak(self.fd)

    def __exit__(self, *exc):
        self.suspend()


_GLYPHS = {IDLE: "○", LISTENING: "●", CAPTURING: "●", TRANSCRIBING: "⋯",
           THINKING: "⋯", SPEAKING: "▶"}
_VU = " ▁▂▃▄▅▆▇█"


class StatusLine:
    """One live, ``\\r``-repainted status line (color via reasoning.want_color).
    ``write`` is injectable for tests."""

    def __init__(self, write=None, color: bool | None = None):
        from .reasoning import want_color
        self._write = write or (lambda s: (sys.stdout.write(s),
                                           sys.stdout.flush()))
        self.color = want_color() if color is None else color
        self._visible = False

    def render(self, machine: TalkStateMachine, *, model: str, voice: str,
               level: float = 0.0, note: str = "") -> str:
        glyph = _GLYPHS.get(machine.state, "○")
        state = "muted" if machine.muted else machine.state
        parts = [f"{glyph} {state}"]
        if machine.mic_role == "vad" or machine.state == CAPTURING:
            bar = "".join(_VU[max(0, min(len(_VU) - 1,
                                         int(level * (len(_VU) - 1)) + (i - 2)))]
                          for i in range(5))
            parts.append(bar)
        parts.append(f"{model}, {voice}")
        if note:
            parts.append(note)
        line = " ,  ".join(parts)
        if self.color:
            line = f"\x1b[2m{line}\x1b[0m"
        return line

    def paint(self, line: str) -> None:
        self._write("\r\x1b[K" + line)
        self._visible = True

    def clear(self) -> None:
        if self._visible:
            self._write("\r\x1b[K")
            self._visible = False


_TURN_END = object()          # tts/playback queue sentinel: reply complete
_NO_ITEM = object()           # tts worker: no queue item carried over
_MERGE_CHARS = 400            # cap on one merged synthesis utterance


class TalkLoop:
    """Wires the threads around the state machine. Every dependency is a
    constructor argument so tests can inject fakes end-to-end."""

    def __init__(self, *, machine, backend, vad, wake, endpointer, brain,
                 base_url: str, api_key: str | None, voice: str | None,
                 speed: float, language: str | None, chime: bool,
                 model_label: str, once: bool = False, status: StatusLine,
                 transcribe=None, synthesize=None,
                 wake_threshold: float = 0.3, make_wake=None,
                 on_display=None):
        from . import talk_client
        self.on_display = on_display  # headless sink; None -> terminal TUI
        self.m = machine
        self.backend = backend
        self.vad = vad
        self.wake = wake
        self.wake_threshold = wake_threshold
        self._make_wake = make_wake              # seam; default resolved lazily
        self.ep = endpointer
        self.brain = brain
        self.base_url = base_url
        self.api_key = api_key
        self.voice = voice
        self.speed = speed
        self.language = language
        self.chime = chime
        self.model_label = model_label
        self.once = once
        self.status = status
        self._transcribe = transcribe or talk_client.transcribe_wav
        self._synthesize = synthesize or talk_client.synthesize

        self.events: queue.Queue = queue.Queue()
        self.frames: queue.Queue = queue.Queue(maxsize=64)
        self.tts_q: queue.Queue = queue.Queue()
        self.play_q: queue.Queue = queue.Queue()
        self.stop_speaking = threading.Event()
        self.cancel_turn = threading.Event()
        self.force_end = threading.Event()
        self.done = threading.Event()
        self.level = 0.0              # mic RMS 0..1 for the VU meter
        self.note = ""
        self._input_started = False
        self._turn_thread = None
        self._threads: list = []

    # -- capture side ----------------------------------------------------------
    def on_frame(self, frame) -> None:
        """sounddevice callback: copy-and-enqueue only (drop-oldest)."""
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.frames.put_nowait(frame)
            except queue.Empty:
                pass

    def _listener(self) -> None:
        from .talk_audio import rms_dbfs
        try:
            while not self.done.is_set():
                try:
                    frame = self.frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.level = max(0.0, min(1.0,
                                          (rms_dbfs(frame) + 60.0) / 60.0))
                role = self.m.mic_role
                if role == "vad":
                    if self.force_end.is_set():
                        self.force_end.clear()
                        ev = self.ep.flush()
                    else:
                        ev = self.ep.feed(frame, self.vad.prob(frame))
                    if ev:
                        self.events.put(("ep", ev))
                elif role == "wake":
                    if self.ep.capturing:
                        self.ep.reset()
                    if self.wake is not None and self.wake.feed(frame):
                        self.events.put(("wake", None))
                else:
                    if self.ep.capturing:  # gated mid-capture (mute/turn)
                        self.ep.reset()
        except Exception as e:  # noqa: BLE001 - a dead listener is a deaf
            # session that still looks alive; say so instead of going silent.
            self.events.put(("error", f"mic listener died: {e!r}"))

    # -- turn side --------------------------------------------------------------
    def _start_turn(self, source) -> None:
        # Fresh event per turn: a canceled turn may still be parked in the
        # HTTP stream, and it must keep seeing its own cancel flag. Clearing
        # a shared event here would resurrect it into the new turn (stale
        # speech, a premature turn_done, concurrent history writes).
        cancel = threading.Event()
        self.cancel_turn = cancel
        self.stop_speaking.clear()
        t = threading.Thread(target=self._run_turn, args=(source, cancel),
                             daemon=True, name="talk-turn")
        self._turn_thread = t
        t.start()

    def _run_turn(self, source, cancel) -> None:
        from .talk_audio import RATE
        from .talk_client import (SentenceChunker, TalkClientError,
                                  encode_wav)
        if isinstance(source, str):
            text = source
        else:
            ms = 1000.0 * len(source) / RATE
            try:
                text = self._transcribe(
                    self.base_url, encode_wav(source, RATE),
                    api_key=self.api_key, language=self.language)
            except TalkClientError as e:
                if not cancel.is_set():
                    self.events.put(("error", str(e)))
                    self.events.put(("stt", ""))
                return
            if cancel.is_set():
                return
            if looks_hallucinated(text, ms):
                text = ""
            if text.strip() and self.m.barged and is_stop_phrase(text):
                # Post-barge-in "stop": acknowledge, never start a turn.
                self.events.put(("stopcmd", text))
                return
            self.events.put(("stt", text))
            if not text.strip():
                return
        chunker = SentenceChunker()
        turn = self.brain.turn(text)
        try:
            for ev in turn:
                if cancel.is_set():
                    # Commit the brain's partial history now, on this
                    # thread, instead of at a GC-timed generator close that
                    # could interleave with the next turn's writes.
                    turn.close()
                    break
                kind = ev[0]
                if kind == "say":
                    for chunk in chunker.feed(ev[1]):
                        self.events.put(("say", chunk))
                        self.tts_q.put((cancel, chunk))
                elif kind == "status":
                    self.events.put(("status", ev[1]))
                elif kind == "done":
                    self.events.put(("stats", ev[1]))
        except Exception as e:            # noqa: BLE001 - surface, keep loop up
            if not cancel.is_set():
                self.events.put(("error", f"turn failed: {e}"))
        if cancel.is_set():
            # on_cancel already reset the state machine; a turn_done or
            # _TURN_END from this dead turn would flip the new turn's
            # turn_active/speaking mid-flight.
            return
        tail = chunker.flush()
        if tail:
            self.events.put(("say", tail))
            self.tts_q.put((cancel, tail))
        self.tts_q.put(_TURN_END)
        self.events.put(("turn_done", None))

    def _tts_worker(self) -> None:
        from .talk_client import TalkClientError
        carry = _NO_ITEM
        while not self.done.is_set():
            if carry is not _NO_ITEM:
                item, carry = carry, _NO_ITEM
            else:
                item = self.tts_q.get()
            if item is None:
                return
            if item is _TURN_END:
                self.play_q.put(_TURN_END)
                continue
            # Each chunk carries its turn's cancel event: a stale chunk that
            # slipped in after the cancel-time queue drain must not be
            # spoken into the next turn.
            cancel, text = item
            # Sentence-sized units only matter for the first audio of a turn;
            # after that, one-sentence synthesis makes the reply a series of
            # isolated utterances with a synthesis round trip at every
            # boundary. Fold whatever the turn has already decoded into one
            # utterance so the cadence follows the text, not the pipeline.
            while len(text) < _MERGE_CHARS:
                try:
                    nxt = self.tts_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None or nxt is _TURN_END or nxt[0] is not cancel:
                    carry = nxt
                    break
                text += " " + nxt[1]
            if cancel.is_set():
                continue
            try:
                pcm, rate = self._synthesize(
                    self.base_url, text, voice=self.voice, speed=self.speed,
                    api_key=self.api_key)
                self.play_q.put((cancel, text, pcm, rate))
            except TalkClientError as e:
                self.events.put(("error", str(e)))

    def _playback(self) -> None:
        while not self.done.is_set():
            item = self.play_q.get()
            if item is None:
                return
            if item is _TURN_END:
                self.events.put(("playback_idle", None))
                continue
            cancel, text, pcm, rate = item
            if text is not None:          # earcons pass text=None
                if self.stop_speaking.is_set() or (
                        cancel is not None and cancel.is_set()):
                    continue
                self.events.put(("chunk_start", text))
            try:
                self.backend.play(pcm, rate, stop=(self.stop_speaking
                                                   if text is not None
                                                   else None))
            except Exception as e:
                # A device error (e.g. PortAudioError on output-device sleep)
                # must not kill the thread: _TURN_END sentinels would pile up
                # and the loop would hang waiting on playback_idle forever.
                self.events.put(("error", f"playback: {e}"))

    def _chime(self, kind: str) -> None:
        if not self.chime:
            return
        from .talk_audio import earcon
        pcm, rate = earcon(kind)
        self.play_q.put((None, None, pcm, rate))

    # -- actions from the state machine -----------------------------------------
    def _apply(self, actions: list, payload=None) -> None:
        for act in actions:
            if act == "chime_wake":
                self._chime("wake")
            elif act == "chime_idle":
                self._chime("idle")
            elif act == "transcribe":
                self._start_turn(payload)
            elif act == "start_turn":
                self._start_turn(payload)
            elif act == "stop_speaking":
                self.stop_speaking.set()
                self._drain(self.play_q)
                self._drain(self.tts_q)
            elif act == "cancel_turn":
                self.cancel_turn.set()
            elif act == "force_end":
                self.force_end.set()

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # -- main loop ---------------------------------------------------------------
    def start_threads(self) -> None:
        for name, fn in (("talk-listener", self._listener),
                         ("talk-tts", self._tts_worker),
                         ("talk-playback", self._playback)):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def shutdown(self) -> None:
        self.done.set()
        self.cancel_turn.set()
        self.stop_speaking.set()
        self.tts_q.put(None)
        self.play_q.put(None)
        self.backend.close()

    def _print(self, text: str) -> None:
        if self.on_display is not None:
            self.on_display(text)
            return
        self.status.clear()
        print(text)

    def handle_event(self, kind: str, payload) -> str | None:
        """Apply one event; returns "quit" to leave the loop."""
        m = self.m
        if kind == "wake":
            self._apply(m.on_wake())
        elif kind == "ep":
            ev, data = payload
            if ev == "start":
                self._apply(m.on_speech_start())
            elif ev == "end":
                self._apply(m.on_utterance(), data)
            else:
                self.note = f"({data})"
                self._apply(m.on_drop())
        elif kind == "stt":
            if payload.strip():
                self._print(f"you: {payload}")
            else:
                self.note = "(heard nothing)"
            self._apply(m.on_stt(payload))
        elif kind == "stopcmd":
            self._print(f"you: {payload}")
            self.note = "(stopped)"
            self._apply(m.on_stt(""))     # rest + idle chime, no turn
        elif kind == "say":
            self._apply(m.on_say())
        elif kind == "chunk_start":
            self._print(f"  {payload}")
        elif kind == "status":
            self.note = payload
        elif kind == "stats":
            pass
        elif kind == "turn_done":
            self._apply(m.on_turn_done())
        elif kind == "playback_idle":
            self._apply(m.on_playback_idle())
            if self.once and not m.turn_active:
                return "quit"
            self.note = ""
        elif kind == "error":
            self._print(f"[talk] {payload}")
        return None

    def _ensure_input(self) -> None:
        """Open the mic stream and VAD lazily - at startup for voice modes,
        or on a /mode switch out of text mode (which builds neither)."""
        if self.m.mode == "text":
            return
        if self.vad is None:
            from .talk_audio import make_vad
            self.vad = make_vad()
        if not self._input_started:
            self.backend.start_input(self.on_frame)
            self._input_started = True

    def run(self) -> int:
        self.start_threads()
        self._ensure_input()
        hint = {"wake": f'say "{self.wake.name}"' if self.wake else "",
                "vad": "just speak", "ptt": "space = talk",
                "text": "type to chat"}.get(self.m.mode, "")
        self._print(f"[talk] {self.model_label}, voice {self.voice or 'default'}"
                    f", mode {self.m.mode}{', ' + hint if hint else ''}"
                    f"  (q quits, / commands)")
        try:
            with _Keys() as keys:
                while True:
                    try:
                        kind, payload = self.events.get(timeout=0.1)
                        if self.handle_event(kind, payload) == "quit":
                            return 0
                        continue        # drain events before repainting
                    except queue.Empty:
                        pass
                    key = keys.poll()
                    if key is not None:
                        out = self._handle_key(key, keys)
                        if out == "quit":
                            return 0
                    self.status.paint(self.status.render(
                        self.m, model=self.model_label,
                        voice=self.voice or "default", level=self.level,
                        note=self.note))
        except KeyboardInterrupt:
            return 130
        finally:
            self.status.clear()
            self.shutdown()

    def run_headless(self, control: "queue.Queue") -> int:
        """The voice loop with no terminal: no keyboard, no status line - the
        menu bar (or any host) reads state straight off ``self.m`` and text
        through the ``on_display`` seam, and steers via ``control``:
        ``"stop"`` (barge-in / cancel), ``"mute"``, ``"hotkey"`` (global
        tap-to-talk force-listen), ``"quit"``, and
        ``"memory [forget ID | clear yes]"`` (the /memory surface; output
        arrives via ``on_display``). Drives the same threads, state machine,
        and event handling as ``run``."""
        self.start_threads()
        self._ensure_input()
        try:
            while True:
                try:
                    kind, payload = self.events.get(timeout=0.1)
                    if self.handle_event(kind, payload) == "quit":
                        return 0
                    continue        # drain events before checking control
                except queue.Empty:
                    pass
                try:
                    cmd = control.get_nowait()
                except queue.Empty:
                    continue
                if cmd == "quit":
                    return 0
                if cmd == "stop" and self.m.state in (TRANSCRIBING, THINKING,
                                                      SPEAKING):
                    self._apply(self.m.on_cancel())
                    self._print("[talk] stopped")
                elif cmd == "mute":
                    self.note = "muted" if self.m.toggle_mute() else ""
                elif cmd == "hotkey":
                    self._apply(self.m.on_hotkey())
                elif cmd == "memory" or cmd.startswith("memory "):
                    self._memory_cmd(cmd[len("memory"):].strip())
        finally:
            self.shutdown()

    # -- keyboard / line input -----------------------------------------------
    def _handle_key(self, key: str, keys: _Keys) -> str | None:
        busy = self.m.state in (TRANSCRIBING, THINKING, SPEAKING)
        if key in ("\x1b", " ") and busy:
            self._apply(self.m.on_cancel())
            self._print("[talk] stopped")
            return None
        if key == " " and self.m.mode == "ptt":
            self._apply(self.m.on_ptt())
            return None
        if key in ("q", "\x04") and not busy:      # q / Ctrl-D
            return "quit"
        if key == "m":
            muted = self.m.toggle_mute()
            self.note = "muted" if muted else ""
            return None
        if key in ("\n", "\r", "\x1b", " "):
            return None
        if key.isprintable():
            return self._line_input(key, keys)
        return None

    def _line_input(self, first: str, keys: _Keys) -> str | None:
        keys.suspend()
        self.status.clear()
        try:
            try:
                line = (first + input(f"> {first}")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
        finally:
            keys.resume()
        if not line:
            return None
        if line.startswith("/"):
            return self._slash(line)
        self._print(f"you: {line}")
        self._apply(self.m.text_turn(line), line)
        return None

    def _slash(self, line: str) -> str | None:
        from .talk_client import list_voices
        parts = line.split(None, 1)
        cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1
                                      else "")
        if cmd in ("/quit", "/exit", "/q"):
            return "quit"
        if cmd == "/voice":
            if arg:
                self.voice = arg
                self._print(f"[talk] voice -> {arg}")
            else:
                voices = list_voices(self.base_url, self.api_key)
                self._print("[talk] voices: " + (", ".join(voices)
                            if voices else "(server has no voice listing; "
                            "pass a name)"))
        elif cmd == "/speed":
            try:
                self.speed = float(arg)
                self._print(f"[talk] speed -> {self.speed:g}")
            except ValueError:
                self._print("[talk] usage: /speed 1.2")
        elif cmd == "/mode":
            prev = self.m.mode
            try:
                self.m.set_mode(arg)
            except ValueError:
                self._print("[talk] modes: wake vad ptt text")
            else:
                try:
                    self._ensure_input()
                except Exception as e:
                    self.m.set_mode(prev)
                    self._print(f"[talk] cannot open microphone: {e}")
                    return None
                self._print(f"[talk] mode -> {arg}")
                if arg == "wake" and self.wake is None:
                    self._print("[talk] no wake detector yet - set a phrase "
                                "with /wake PHRASE")
        elif cmd == "/wake":
            if not arg:
                self._print(f'[talk] wake phrase: "{self.wake.name}"'
                            if self.wake else
                            "[talk] no wake detector - /wake PHRASE sets one")
            else:
                factory = self._make_wake
                if factory is None:
                    from .talk_audio import make_wake_detector as factory
                detector, hint = factory(arg, threshold=self.wake_threshold)
                if detector is None:              # keep the old detector
                    self._print(f"[talk] {hint}")
                else:
                    self.wake = detector
                    self._print(f'[talk] wake phrase -> "{arg}"'
                                + ("" if self.m.mode == "wake"
                                   else " (switch with /mode wake)"))
        elif cmd == "/mute":
            self.note = "muted" if self.m.toggle_mute() else ""
        elif cmd == "/reset":
            self.brain.reset()
            self._print("[talk] conversation reset")
        elif cmd == "/system":
            self.brain.system = arg or None
            self.brain.reset()
            self._print("[talk] system prompt " + ("set" if arg else
                                                   "cleared") + "; reset")
        elif cmd == "/memory":
            self._memory_cmd(arg)
        elif cmd == "/devices":
            self._print(self.backend.describe_devices())
        elif cmd == "/help":
            self._print(
                "  /voice [name]   list voices / switch voice\n"
                "  /speed X        speech speed (0.25-4)\n"
                "  /mode M         wake | vad | ptt | text\n"
                "  /wake [PHRASE]  show / change the wake phrase\n"
                "  /mute           toggle the mic\n"
                "  /reset          clear the conversation\n"
                "  /system TEXT    set the system prompt (empty clears)\n"
                "  /memory ...     list memories / forget ID / clear\n"
                "  /devices        list audio devices\n"
                "  /quit           leave (also: q)\n"
                "  keys: space/esc stop speech, m mute, any letter = type")
        else:
            self._print(f"[talk] unknown command {cmd} (try /help)")
        return None

    def _memory_cmd(self, arg: str) -> None:
        mem = getattr(self.brain, "memory", None)
        if mem is None:
            self._print("[talk] no memory store (assistant brain only - set "
                        "talk.brain: assistant)")
            return
        words = arg.split()
        if not words:
            total = mem.count()
            if total == 0:
                self._print("[talk] no memories stored")
                return
            rows = mem.list_all(limit=20)
            header = f"[talk] {total} memories"
            if total > len(rows):
                header += f" (newest {len(rows)} shown)"
            lines = [header]
            now = time.time()
            for r in rows:
                text = " ".join(str(r["text"]).split())
                if len(text) > 60:
                    text = text[:57] + "..."
                lines.append(f"  #{r['id']:<5} {_age(now - r['created']):>4}"
                             f"  x{r['recalled']:<3} {text}")
            self._print("\n".join(lines))
        elif words[0] == "forget" and len(words) == 2:
            try:
                mem_id = int(words[1].lstrip("#"))
            except ValueError:
                self._print("[talk] usage: /memory forget ID")
                return
            self._print(f"[talk] forgot #{mem_id}" if mem.delete(mem_id)
                        else f"[talk] no memory #{mem_id}")
        elif words[0] == "clear":
            if words[1:] == ["yes"]:
                self._print(f"[talk] cleared {mem.clear()} memories")
            else:
                self._print(f"[talk] this deletes {mem.count()} memories - "
                            "confirm with: /memory clear yes")
        else:
            self._print("[talk] usage: /memory | /memory forget ID | "
                        "/memory clear")


# The verb

def _build_parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Voice chat with a served model: wake word (or VAD / "
                    "push-to-talk), whisper STT, streamed replies spoken "
                    "as they decode. Starts a background server from "
                    "your config when none is running.")
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="Server config YAML (default: standard locations).")
    ap.add_argument("--model", default=None,
                    help="Chat model id[@profile] (default: talk.model, else "
                         "the server's default).")
    ap.add_argument("--voice", default=None,
                    help="TTS voice (default: talk.voice, else server default).")
    ap.add_argument("--speed", type=float, default=None,
                    help="Speech speed 0.25-4 (default 1.0).")
    ap.add_argument("--language", default=None,
                    help="Whisper language hint (default: auto).")
    ap.add_argument("--system", default=None, metavar="TEXT",
                    help="System prompt override.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Reply token cap (default 512).")
    ap.add_argument("--mode", choices=("wake", "vad", "ptt", "text"),
                    default=None,
                    help="Activation: wake word, always-on VAD, push-to-talk, "
                         "or typed-only (default wake).")
    ap.add_argument("--wake-word", default=None, metavar="PHRASE",
                    help='Wake phrase, any English text (default "hey '
                         'assistant").')
    ap.add_argument("--wake-threshold", type=float, default=None,
                    help="Wake sensitivity 0-1 (default 0.3).")
    ap.add_argument("--vad-threshold", type=float, default=None,
                    help="Speech-probability threshold (default 0.6).")
    ap.add_argument("--vad-silence-ms", type=float, default=None,
                    help="Trailing silence that ends an utterance (default 550).")
    ap.add_argument("--min-speech-ms", type=float, default=None,
                    help="Discard utterances shorter than this (default 300).")
    ap.add_argument("--input-device", default=None,
                    help="Mic: index or name substring (default: system).")
    ap.add_argument("--output-device", default=None,
                    help="Speaker: index or name substring (default: system).")
    ap.add_argument("--no-chime", action="store_true",
                    help="No earcons on wake/idle.")
    ap.add_argument("--brain", choices=("chat", "assistant"), default=None,
                    help="Turn engine: plain chat, or the assistant "
                         "(tools + memory; default from config).")
    ap.add_argument("--once", action="store_true",
                    help="One exchange (listen, reply), then exit.")
    ap.add_argument("--base-url", default=None,
                    help="Server OpenAI base URL (default http://HOST:PORT/v1).")
    ap.add_argument("--host", default=None, help="Server host.")
    ap.add_argument("--port", type=int, default=None, help="Server port.")
    ap.add_argument("--api-key", default=None, help="Server API key.")
    ap.add_argument("--no-start", action="store_true",
                    help="Never auto-start a server.")
    ap.add_argument("--start-timeout", type=float, default=180.0,
                    metavar="S",
                    help="Auto-start wait (default 180; 0 = wait forever).")
    ap.add_argument("--list-devices", action="store_true",
                    help="List audio devices and exit.")
    ap.add_argument("--list-voices", action="store_true",
                    help="List the server's TTS voices and exit.")
    return ap


def _load_talk_cfg(config_path: str | None):
    """(TalkCfg, why) from --config, else the first default-location config,
    else built-in defaults."""
    from . import config as cfgmod
    if config_path:
        return cfgmod.load_config(config_path).talk
    from .launch import _discover_config
    cfg, _path = _discover_config()
    return cfg.talk if cfg is not None else cfgmod.TalkCfg()


def _merged_settings(args, talk_cfg):
    """Flags (when passed) over the config's talk: block."""
    def pick(flag, cfg_val):
        return cfg_val if flag is None else flag
    vad = talk_cfg.vad
    return {
        "model": pick(args.model, talk_cfg.model),
        "voice": pick(args.voice, talk_cfg.voice),
        "speed": pick(args.speed, talk_cfg.speed),
        "language": pick(args.language, talk_cfg.language),
        "system": pick(args.system, talk_cfg.system),
        "max_tokens": pick(args.max_tokens, talk_cfg.max_tokens),
        "mode": pick(args.mode, talk_cfg.mode),
        "wake_word": pick(args.wake_word, talk_cfg.wake_word),
        "wake_threshold": pick(args.wake_threshold, talk_cfg.wake_threshold),
        "vad_threshold": pick(args.vad_threshold, vad.threshold),
        "silence_ms": pick(args.vad_silence_ms, vad.silence_ms),
        "min_speech_ms": pick(args.min_speech_ms, vad.min_speech_ms),
        "pre_roll_ms": vad.pre_roll_ms,
        "min_level_dbfs": -45.0,
        "input_device": pick(args.input_device, talk_cfg.input_device),
        "output_device": pick(args.output_device, talk_cfg.output_device),
        "chime": talk_cfg.chime and not args.no_chime,
        "brain": pick(getattr(args, "brain", None), talk_cfg.brain),
        "assistant": talk_cfg.assistant,
    }


def _pick_model(requested: str | None, caps: dict) -> str | None:
    """flag/config model > server default-marked > sole served id."""
    if requested:
        return requested
    if caps.get("default"):
        return caps["default"]
    ids = caps.get("chat_ids") or []
    return ids[0] if len(ids) == 1 else None


def _capability_guidance(caps: dict, *, needs_stt: bool, explicit_url: bool,
                         out=sys.stderr) -> bool:
    """Check stt/tts service markers; print fix-it guidance and return False
    when the server can't do voice."""
    missing = [name for name, need in (("stt", needs_stt), ("tts", True))
               if need and not caps.get(name)]
    if not missing:
        return True
    what = " + ".join(missing)
    print(f"[talk] the server is up but has no {what} service enabled",
          file=out)
    if explicit_url:
        print("  enable it on that server's config (server: {stt: true, "
              "tts: true}) and restart it", file=out)
    else:
        print(
            "  add to your config's server: block\n"
            "      stt: true      # whisper-turbo\n"
            "      tts: true      # kokoro\n"
            "  install the models' engines if needed:\n"
            "      pip install 'gmlx[talk]'\n"
            "  then restart:  gmlx restart   (and re-run gmlx talk)",
            file=out)
    return False


class TalkSetupError(Exception):
    """A fatal problem building the voice pipeline; str() is the user-facing
    message (the CLI prefixes ``error:``, the menu bar shows it verbatim)."""


def build_talk_loop(s: dict, *, base_url: str, api_key: str | None,
                    model: str, once: bool = False, on_display=None,
                    warn=None):
    """Construct the full voice pipeline - audio backend, wake/VAD, brain
    (incl. assistant tools + memory), :class:`TalkLoop` - from merged settings
    ``s``. Shared by the CLI verb and the menu bar's in-process session.
    Returns ``(loop, cleanup)``; call ``cleanup()`` after the loop exits.
    Fatal problems raise :class:`TalkSetupError`; non-fatal degradations
    (missing wake engine, energy-gate VAD, MCP servers down) go to ``warn``."""
    warn = warn or (lambda m: print(f"[talk] {m}", file=sys.stderr))
    from .talk_audio import (Endpointer, SoundDeviceBackend, TalkAudioError,
                             make_vad, make_wake_detector)
    from .talk_client import ServerChatBrain
    try:
        backend = SoundDeviceBackend(s["input_device"], s["output_device"])
    except (ImportError, TalkAudioError) as e:
        raise TalkSetupError(str(e))

    mode = s["mode"]
    wake = None
    if once and mode == "wake":
        mode = "vad"                              # one shot: skip the wake gate
    if mode == "wake":
        try:
            wake, hint = make_wake_detector(s["wake_word"],
                                            threshold=s["wake_threshold"])
        except TalkAudioError as e:
            raise TalkSetupError(str(e))
        if wake is None:
            warn(hint)
            mode = "vad"
    vad = make_vad(s["vad_threshold"]) if mode != "text" else None
    if vad is not None and vad.name == "energy" and mode != "text":
        warn("silero VAD unavailable (install 'gmlx[talk]'); using the "
             "energy gate - works best in a quiet room")

    mcp_host = None
    if s["brain"] == "assistant":
        from .assistant_brain import AssistantBrain
        from .talk_mcp import connect_servers
        a = s["assistant"]
        mcp_host, registry, mcp_warnings = connect_servers(
            a.mcp, call_timeout_s=a.tool_timeout_s)
        for w in mcp_warnings:
            warn(w)
        if a.mcp:
            warn(f"assistant tools: "
                 f"{', '.join(registry.names()) or '(none connected)'}")
        memory = None
        if a.memory.enabled:
            from .talk_memory import MemoryStore, make_extractor
            extractor = (make_extractor(base_url, model, api_key=api_key)
                         if a.memory.extract else None)
            memory = MemoryStore(base_url=base_url, api_key=api_key,
                                 path=a.memory.path, top_k=a.memory.top_k,
                                 extract=extractor,
                                 ttl_days=a.memory.ttl_days,
                                 max_items=a.memory.max_items)
            warn(f"assistant memory: {memory.path} "
                 f"({memory.count()} memories)")
        brain = AssistantBrain(
            base_url=base_url, model=model, api_key=api_key,
            system=s["system"], max_tokens=s["max_tokens"],
            tools=registry, max_tool_rounds=a.max_tool_rounds,
            tool_timeout_s=a.tool_timeout_s, memory=memory)
    else:
        brain = ServerChatBrain(base_url=base_url, model=model,
                                api_key=api_key, system=s["system"],
                                max_tokens=s["max_tokens"])
    loop = TalkLoop(
        machine=TalkStateMachine(mode), backend=backend, vad=vad, wake=wake,
        endpointer=Endpointer(silence_ms=s["silence_ms"],
                              min_speech_ms=s["min_speech_ms"],
                              pre_roll_ms=s["pre_roll_ms"],
                              min_level_dbfs=s["min_level_dbfs"]),
        brain=brain, base_url=base_url, api_key=api_key, voice=s["voice"],
        speed=s["speed"], language=s["language"], chime=s["chime"],
        model_label=model, once=once, status=StatusLine(),
        wake_threshold=s["wake_threshold"], on_display=on_display)

    # Hold the model resident (and warm it) for the session's lifetime: a
    # live mic with no model loaded is not a real mode of operation. Off the
    # startup path - a slow load must not delay the mic coming up.
    from .talk_client import keep_model
    threading.Thread(target=keep_model, args=(base_url, model),
                     kwargs={"api_key": api_key}, daemon=True,
                     name="talk-keep").start()

    def cleanup() -> None:
        keep_model(base_url, model, api_key=api_key, keep=False, timeout=3.0)
        if hasattr(brain, "close"):
            brain.close()                 # assistant brain closes its memory
        if mcp_host is not None:
            mcp_host.close()

    return loop, cleanup


def cmd_talk(argv: list | None = None, prog: str = "gmlx talk") -> int:
    args = _build_parser(prog).parse_args(
        sys.argv[1:] if argv is None else list(argv))

    from .talk_audio import TalkAudioError
    try:
        talk_cfg = _load_talk_cfg(args.config)
    except Exception as e:                      # ConfigError etc.
        print(f"error: {e}", file=sys.stderr)
        return 2
    s = _merged_settings(args, talk_cfg)

    if args.list_devices:
        from .talk_audio import SoundDeviceBackend
        try:
            print(SoundDeviceBackend().describe_devices())
        except (ImportError, TalkAudioError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0

    # Server: reuse launch's start-if-down machinery wholesale.
    from . import launch as launch_mod
    ns = argparse.Namespace(
        harness=None, rerun_label="talk", base_url=args.base_url,
        host=args.host, port=args.port, api_key=args.api_key,
        no_start=args.no_start, start_timeout=args.start_timeout,
        config_only=False)
    rc = launch_mod._ensure_server(ns)
    if rc is not None:
        return rc
    base_url, api_key = ns.base_url, ns.api_key

    from .talk_client import (TalkClientError, list_voices,
                              probe_capabilities)
    try:
        caps = probe_capabilities(base_url, api_key)
    except TalkClientError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.list_voices:
        voices = list_voices(base_url, api_key)
        print("\n".join(voices) if voices
              else "(the server has no /v1/audio/voices route)")
        return 0

    if not _capability_guidance(caps, needs_stt=s["mode"] != "text",
                                explicit_url=args.base_url is not None):
        return 1

    model = _pick_model(s["model"], caps)
    if not model:
        ids = ", ".join(caps.get("chat_ids") or []) or "(none)"
        print(f"error: no model selected and the server has no default - "
              f"pass --model one of: {ids}", file=sys.stderr)
        return 2

    try:
        loop, cleanup = build_talk_loop(s, base_url=base_url, api_key=api_key,
                                        model=model, once=args.once)
    except TalkSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        return loop.run()
    except TalkAudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        cleanup()
