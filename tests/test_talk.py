"""TalkStateMachine transition table (pure, synthetic event sequences)."""
from __future__ import annotations

import queue
import time

import numpy as np
import pytest

from gmlx.talk import (
    CAPTURING,
    IDLE,
    LISTENING,
    SPEAKING,
    THINKING,
    TRANSCRIBING,
    StatusLine,
    TalkLoop,
    TalkStateMachine,
    looks_hallucinated,
)
from gmlx.talk_audio import FRAME_SAMPLES, Endpointer, EnergyVAD


def test_wake_mode_full_turn():
    m = TalkStateMachine("wake")
    assert m.state == IDLE and m.mic_role == "wake"
    assert m.on_wake() == ["chime_wake"]
    assert m.state == LISTENING and m.mic_role == "vad"
    m.on_speech_start()
    assert m.state == CAPTURING
    assert m.on_utterance() == ["transcribe"]
    # wake-word-only scoring stays live through the reply (voice barge-in)
    assert m.state == TRANSCRIBING and m.mic_role == "wake"
    assert m.on_stt("hello there") == []   # transcribe thread runs the brain
    assert m.state == THINKING and m.turn_active
    m.on_say()
    assert m.state == SPEAKING and m.speaking and m.mic_role == "wake"
    assert m.on_turn_done() == []                            # audio still out
    assert m.on_playback_idle() == ["chime_idle"]
    assert m.state == IDLE and not m.turn_active and not m.speaking


def test_wake_barge_in_while_speaking():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("tell me a long story")
    m.on_say()
    assert m.state == SPEAKING

    acts = m.on_wake()
    assert acts == ["stop_speaking", "cancel_turn", "chime_wake"]
    assert m.state == LISTENING and m.barged
    assert not m.turn_active and not m.speaking
    # stale events from the killed turn must not close the open mic
    assert m.on_turn_done() == [] and m.state == LISTENING
    assert m.on_playback_idle() == [] and m.state == LISTENING
    # follow-up becomes a normal turn; barged clears
    m.on_speech_start()
    m.on_utterance()
    assert m.on_stt("make it shorter") == []
    assert m.state == THINKING and not m.barged


def test_wake_barge_in_stop_phrase_rests():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("tell me a long story")
    m.on_say()
    m.on_wake()                        # barge-in
    m.on_speech_start()
    m.on_utterance()
    # the loop maps a stop phrase to on_stt(""): rest + idle chime, no turn
    assert m.on_stt("") == ["chime_idle"]
    assert m.state == IDLE and not m.barged and not m.turn_active


def test_wake_barge_in_gated_when_muted():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("hello")
    m.on_say()
    m.muted = True
    assert m.mic_role is None
    assert m.on_wake() == []
    assert m.state == SPEAKING


def test_vad_mode_stays_half_duplex():
    m = TalkStateMachine("vad")
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("hello")
    m.on_say()
    assert m.state == SPEAKING and m.mic_role is None
    assert m.on_wake() == []


def test_cancel_chimes_once_not_three_times():
    """on_cancel rests silently, but the turn thread's `turn_done` and the
    pipeline's _TURN_END still arrive afterwards - each used to _rest() again
    and emit a second and third chime_idle after every barge-in."""
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("hello there")
    m.on_say()
    assert m.state == SPEAKING

    assert m.on_cancel() == ["stop_speaking", "cancel_turn"]   # no chime
    assert m.state == IDLE
    assert m.on_turn_done() == []                              # late event
    assert m.on_playback_idle() == []                          # later event
    assert m.state == IDLE


def test_vad_mode_rests_at_listening():
    m = TalkStateMachine("vad")
    assert m.state == LISTENING and m.mic_role == "vad"
    assert m.on_wake() == []                                 # no wake gating
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("hi")
    m.on_say()
    m.on_turn_done()
    assert m.on_playback_idle() == []                        # no idle chime
    assert m.state == LISTENING


def test_drop_and_empty_stt_return_to_rest():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    assert m.on_drop() == ["chime_idle"]
    assert m.state == IDLE
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    assert m.on_stt("   ") == ["chime_idle"]                 # whisper gave air
    assert m.state == IDLE and not m.turn_active


def test_cancel_stops_both_pipelines():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("x")
    m.on_say()
    acts = m.on_cancel()
    assert acts == ["stop_speaking", "cancel_turn"]
    assert m.state == IDLE and not m.turn_active and not m.speaking
    assert m.on_cancel() == []                               # idempotent


def test_playback_finishing_before_turn():
    m = TalkStateMachine("vad")
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("x")
    m.on_say()
    assert m.on_playback_idle() == []                        # brain still going
    assert m.state == SPEAKING and m.turn_active
    m.on_turn_done()
    assert m.state == LISTENING


def test_text_turn_from_rest_only():
    m = TalkStateMachine("wake")
    assert m.text_turn("  ") == []
    assert m.text_turn("hello") == ["start_turn"]
    assert m.state == THINKING
    assert m.text_turn("again") == []                        # busy


def test_ptt_toggle_cycle():
    m = TalkStateMachine("ptt")
    assert m.state == IDLE and m.mic_role is None            # armed by key
    assert m.on_ptt() == ["chime_wake"]
    assert m.state == LISTENING and m.mic_role == "vad"
    assert m.on_ptt() == ["chime_idle"]                      # nothing said
    assert m.state == IDLE
    m.on_ptt()
    m.on_speech_start()
    assert m.on_ptt() == ["force_end"]                       # key ends capture
    assert m.state == CAPTURING                              # loop flushes it


def test_hotkey_force_listen_across_states():
    m = TalkStateMachine("wake")
    assert m.on_hotkey() == ["chime_wake"]           # idle: open the mic
    assert m.state == LISTENING
    assert m.on_hotkey() == ["chime_idle"]           # tap again dismisses
    assert m.state == IDLE
    m.on_hotkey()
    m.on_speech_start()
    assert m.on_hotkey() == ["force_end"]            # capturing: flush now
    assert m.state == CAPTURING                      # loop flushes it
    m.on_utterance()
    acts = m.on_hotkey()                             # busy: barge in + listen
    assert acts == ["stop_speaking", "cancel_turn", "chime_wake"]
    assert m.state == LISTENING and m.barged
    assert not m.turn_active and not m.speaking


def test_hotkey_modes_and_mute():
    assert TalkStateMachine("text").on_hotkey() == []   # no endpointer built
    m = TalkStateMachine("vad")
    assert m.on_hotkey() == []                       # rest already listening
    assert m.state == LISTENING
    m.toggle_mute()
    assert m.on_hotkey() == [] and not m.muted       # tap unmutes (vs on_wake)
    assert m.mic_role == "vad"
    m = TalkStateMachine("wake")
    m.toggle_mute()
    assert m.on_hotkey() == ["chime_wake"]           # muted idle: unmute+listen
    assert not m.muted and m.state == LISTENING


def test_mute_gates_mic_and_aborts_capture():
    m = TalkStateMachine("wake")
    m.on_wake()
    m.on_speech_start()
    assert m.toggle_mute() is True
    assert m.state == IDLE and m.mic_role is None
    assert m.on_wake() == []                                 # muted: no wake
    assert m.toggle_mute() is False
    assert m.mic_role == "wake"


def test_set_mode_moves_rest_state_only():
    m = TalkStateMachine("wake")
    m.set_mode("vad")
    assert m.state == LISTENING
    m.on_speech_start()
    m.on_utterance()
    m.on_stt("x")
    m.set_mode("wake")                                       # mid-turn: state kept
    assert m.state == THINKING and m.mode == "wake"
    m.on_say()
    m.on_turn_done()
    assert m.on_playback_idle() == ["chime_idle"]            # new mode's rest
    assert m.state == IDLE


def test_stale_events_ignored():
    m = TalkStateMachine("wake")
    assert m.on_utterance() == []                            # not capturing
    assert m.on_stt("x") == []                               # not transcribing
    assert m.on_speech_start() == []
    assert m.state == IDLE


def test_text_mode_keeps_mic_off():
    m = TalkStateMachine("text")
    assert m.mic_role is None
    assert m.text_turn("hello") == ["start_turn"]


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        TalkStateMachine("telepathy")
    with pytest.raises(ValueError):
        TalkStateMachine("wake").set_mode("nope")


# ---------------------------------------------------------------------------
# TalkLoop pipeline with fakes (no sockets, no PortAudio, no models)


def test_looks_hallucinated():
    assert looks_hallucinated("Thank you.", 700.0)
    assert looks_hallucinated("you", 400.0)
    assert not looks_hallucinated("Thank you.", 4000.0)   # long clip: real
    assert not looks_hallucinated("what's the weather", 700.0)


class FakeBackend:
    def __init__(self):
        self.played = []           # (text-or-None marker via pcm length, rate)
        self.on_frame = None
        self.closed = False

    def start_input(self, on_frame, **kw):
        self.on_frame = on_frame

    def stop_input(self):
        pass

    def play(self, pcm, rate, stop=None, slice_ms=150.0):
        self.played.append((len(pcm), rate))
        return not (stop is not None and stop.is_set())

    def close(self):
        self.closed = True

    def describe_devices(self):
        return "  0  [in*,out*]  Fake Device"


class FakeWake:
    name = "hey test"

    def feed(self, frame):
        return int(frame[0]) == 12345

    def reset(self):
        pass


class FakeBrain:
    def __init__(self):
        self.turns = []
        self.system = None
        self.was_reset = False

    def turn(self, text):
        self.turns.append(text)
        yield ("status", "thinking")
        yield ("say", "Hello there. ")
        yield ("say", "Bye bye friend.")
        yield ("done", {"total_tokens": 3})

    def reset(self):
        self.was_reset = True


def _loud(marker=8000):
    return np.full(FRAME_SAMPLES, marker, dtype=np.int16)


def _quiet():
    return np.full(FRAME_SAMPLES, 3, dtype=np.int16)


def _wake_frame():
    f = _quiet()
    f[0] = 12345
    return f


def _make_loop(mode="wake", **kw):
    from gmlx.talk import TalkStateMachine
    backend = FakeBackend()
    said = []

    def fake_transcribe(base_url, wav_bytes, api_key=None, language=None):
        return "hello world"

    def fake_synth(base_url, text, voice=None, speed=1.0, api_key=None):
        said.append(text)
        return np.zeros(2400, dtype=np.int16), 24000

    loop = TalkLoop(
        machine=TalkStateMachine(mode), backend=backend, vad=EnergyVAD(),
        wake=FakeWake() if mode == "wake" else None,
        endpointer=Endpointer(silence_ms=240.0, min_speech_ms=160.0,
                              pre_roll_ms=160.0, min_level_dbfs=-60.0),
        brain=FakeBrain(), base_url="http://h:1/v1", api_key=None,
        voice="af_heart", speed=1.0, language=None, chime=False,
        model_label="m", status=StatusLine(write=lambda s: None, color=False),
        transcribe=fake_transcribe, synthesize=fake_synth, **kw)
    loop._said = said
    return loop


def _pump_until(loop, cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            kind, payload = loop.events.get(timeout=0.05)
        except queue.Empty:
            if cond():
                return
            continue
        loop.handle_event(kind, payload)
        if cond():
            return
    raise AssertionError(f"timeout; state={loop.m.state} "
                         f"turn={loop.m.turn_active} spk={loop.m.speaking}")


def test_loop_full_voice_turn(capsys):
    from gmlx.talk import IDLE, LISTENING
    loop = _make_loop("wake")
    loop.start_threads()
    try:
        loop.backend.start_input(loop.on_frame)
        loop.on_frame(_wake_frame())
        _pump_until(loop, lambda: loop.m.state == LISTENING)
        for _ in range(5):
            loop.on_frame(_loud())
        for _ in range(8):
            loop.on_frame(_quiet())
        _pump_until(loop, lambda: (loop.m.state == IDLE
                                   and not loop.m.turn_active
                                   and not loop.m.speaking))
    finally:
        loop.shutdown()
    assert loop.brain.turns == ["hello world"]
    # min_chars merges the short opener into one speakable chunk
    assert loop._said == ["Hello there. Bye bye friend."]
    assert len(loop.backend.played) >= 1
    out = capsys.readouterr().out
    assert "you: hello world" in out
    assert "Hello there. Bye bye friend." in out


def test_loop_typed_turn(capsys):
    from gmlx.talk import LISTENING
    loop = _make_loop("vad")
    loop.start_threads()
    try:
        loop._apply(loop.m.text_turn("typed question"), "typed question")
        _pump_until(loop, lambda: (loop.m.state == LISTENING
                                   and not loop.m.turn_active
                                   and not loop.m.speaking))
    finally:
        loop.shutdown()
    assert loop.brain.turns == ["typed question"]
    assert loop._said == ["Hello there. Bye bye friend."]


def test_loop_wake_barge_in_stop_phrase(capsys):
    """Wake phrase mid-reply stops it; a follow-up "Stop." never reaches the
    brain and the loop rests."""
    import threading
    from gmlx.talk import IDLE, LISTENING, SPEAKING

    class SlowBrain(FakeBrain):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def turn(self, text):
            self.turns.append(text)
            yield ("say", "Once upon a time, in a land far away. ")
            self.release.wait(5.0)          # park mid-turn until canceled
            yield ("say", "The end.")
            yield ("done", {})

    texts = iter(["tell me a story", "Stop."])
    loop = _make_loop("wake")
    loop.brain = SlowBrain()
    loop._transcribe = lambda *a, **k: next(texts)
    loop.start_threads()
    try:
        loop.on_frame(_wake_frame())
        _pump_until(loop, lambda: loop.m.state == LISTENING)
        for _ in range(5):
            loop.on_frame(_loud())
        for _ in range(8):
            loop.on_frame(_quiet())
        _pump_until(loop, lambda: loop.m.state == SPEAKING)

        loop.on_frame(_wake_frame())        # voice barge-in
        _pump_until(loop, lambda: loop.m.state == LISTENING and loop.m.barged)
        for _ in range(5):
            loop.on_frame(_loud())
        for _ in range(8):
            loop.on_frame(_quiet())
        _pump_until(loop, lambda: (loop.m.state == IDLE
                                   and not loop.m.turn_active))
    finally:
        loop.brain.release.set()
        loop.shutdown()
    assert loop.brain.turns == ["tell me a story"]   # "Stop." never a turn
    out = capsys.readouterr().out
    assert "you: Stop." in out


def test_loop_hallucination_never_becomes_turn():
    from gmlx.talk import LISTENING

    def ghost_transcribe(base_url, wav_bytes, api_key=None, language=None):
        return "Thank you."

    loop = _make_loop("vad")
    loop._transcribe = ghost_transcribe
    loop.start_threads()
    try:
        for _ in range(5):                        # ~400ms < the 1.5s ghost cap
            loop.on_frame(_loud())
        for _ in range(8):
            loop.on_frame(_quiet())
        _pump_until(loop, lambda: (loop.m.state == LISTENING
                                   and not loop.m.turn_active))
    finally:
        loop.shutdown()
    assert loop.brain.turns == []                 # ghost filtered, no turn


def _wait(cond, timeout=5.0):
    """Poll ``cond`` without touching loop.events (run_headless owns them)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_run_headless_full_turn_display_and_control():
    import threading
    lines = []
    loop = _make_loop("wake", on_display=lines.append)
    control = queue.Queue()
    t = threading.Thread(target=loop.run_headless, args=(control,),
                         daemon=True)
    t.start()
    try:
        loop.on_frame(_wake_frame())
        _wait(lambda: loop.m.state == LISTENING)
        for _ in range(5):
            loop.on_frame(_loud())
        for _ in range(8):
            loop.on_frame(_quiet())
        _wait(lambda: (loop.m.state == IDLE and not loop.m.turn_active
                       and not loop.m.speaking))
        control.put("mute")
        _wait(lambda: loop.m.muted)
        control.put("quit")
        t.join(3.0)
        assert not t.is_alive()
    finally:
        loop.shutdown()
    assert loop.brain.turns == ["hello world"]
    assert loop._said == ["Hello there. Bye bye friend."]
    assert any("you: hello world" in ln for ln in lines)  # display seam, not stdout


def test_run_headless_memory_control_command():
    """The menubar steers /memory through the control queue: "memory" lists
    to the display seam, "memory clear yes" wipes (confirmation happens in
    the host UI, so the queue command executes directly)."""
    import threading
    lines = []
    loop = _make_loop("vad", on_display=lines.append)

    class _Mem:
        cleared = False

        def count(self):
            return 1

        def list_all(self, limit=None):
            return [{"id": 7, "text": "likes green tea",
                     "created": time.time() - 30, "recalled": 2}]

        def clear(self):
            self.cleared = True
            return 1

    loop.brain.memory = _Mem()
    control = queue.Queue()
    t = threading.Thread(target=loop.run_headless, args=(control,),
                         daemon=True)
    t.start()
    try:
        control.put("memory")
        _wait(lambda: any("likes green tea" in ln for ln in lines))
        control.put("memory clear yes")
        _wait(lambda: any("cleared 1 memories" in ln for ln in lines))
        assert loop.brain.memory.cleared
        control.put("quit")
        t.join(3.0)
        assert not t.is_alive()
    finally:
        loop.shutdown()


def test_run_headless_stop_cancels_busy_turn():
    import threading

    class SlowBrain:
        def __init__(self):
            self.turns = []

        def turn(self, text):
            self.turns.append(text)
            yield ("status", "thinking")
            time.sleep(0.4)                       # cancellation window
            yield ("say", "too late")
            yield ("done", {})

        def reset(self):
            pass

    loop = _make_loop("vad")
    loop.brain = SlowBrain()
    control = queue.Queue()
    t = threading.Thread(target=loop.run_headless, args=(control,),
                         daemon=True)
    t.start()
    try:
        loop._apply(loop.m.text_turn("q"), "q")
        _wait(lambda: loop.m.state == THINKING)
        control.put("stop")
        _wait(lambda: not loop.m.turn_active and loop.m.state != THINKING)
        control.put("quit")
        t.join(3.0)
        assert not t.is_alive()
    finally:
        loop.shutdown()
    assert loop._said == []                       # cancelled before speech


def test_run_headless_hotkey_control_command():
    """Global tap-to-talk: "hotkey" opens the mic from idle, dismisses on a
    second tap, and barges in on a busy turn (always force-listen)."""
    import threading

    class SlowBrain:
        def __init__(self):
            self.turns = []

        def turn(self, text):
            self.turns.append(text)
            yield ("status", "thinking")
            time.sleep(0.4)                       # cancellation window
            yield ("say", "too late")
            yield ("done", {})

        def reset(self):
            pass

    loop = _make_loop("wake")
    loop.brain = SlowBrain()
    control = queue.Queue()
    t = threading.Thread(target=loop.run_headless, args=(control,),
                         daemon=True)
    t.start()
    try:
        control.put("hotkey")
        _wait(lambda: loop.m.state == LISTENING)
        control.put("hotkey")
        _wait(lambda: loop.m.state == IDLE)
        loop._apply(loop.m.text_turn("q"), "q")
        _wait(lambda: loop.m.state == THINKING)
        control.put("hotkey")                     # barge in, land listening
        _wait(lambda: loop.m.state == LISTENING and not loop.m.turn_active)
        control.put("quit")
        t.join(3.0)
        assert not t.is_alive()
    finally:
        loop.shutdown()
    assert loop._said == []                       # cancelled before speech


def test_canceled_turn_does_not_resurrect_into_next_turn():
    """Esc during a slow-first-token turn, then an immediate new question:
    the canceled turn thread (still parked in the HTTP stream) must not wake
    up, speak its stale reply into the new turn, or post a premature
    turn_done that flips the new turn's state."""
    import threading

    class BlockingBrain:
        def __init__(self):
            self.release = threading.Event()
            self.calls = 0

        def turn(self, text):
            self.calls += 1
            if self.calls == 1:
                yield ("status", "thinking")
                self.release.wait(5.0)      # parked awaiting first SSE delta
                yield ("say", "STALE REPLY. ")
                yield ("done", {})
            else:
                yield ("say", "Fresh reply for you. ")
                yield ("done", {})

        def reset(self):
            pass

    loop = _make_loop("vad")
    brain = BlockingBrain()
    loop.brain = brain
    loop.start_threads()
    try:
        loop._apply(loop.m.text_turn("one"), "one")
        _wait(lambda: brain.calls == 1)
        loop._apply(loop.m.on_cancel())               # Esc while parked
        loop._apply(loop.m.text_turn("two"), "two")   # immediate new turn
        _wait(lambda: brain.calls == 2)
        brain.release.set()                           # stale stream wakes up
        _pump_until(loop, lambda: (not loop.m.turn_active
                                   and not loop.m.speaking))
        time.sleep(0.2)     # window for the stale thread to misbehave
        while True:         # drain whatever it produced through the machine
            try:
                kind, payload = loop.events.get_nowait()
            except queue.Empty:
                break
            loop.handle_event(kind, payload)
    finally:
        loop.shutdown()
    assert loop._said == ["Fresh reply for you."]
    assert not loop.m.turn_active and not loop.m.speaking


def test_slash_commands():
    loop = _make_loop("vad")
    assert loop._slash("/voice bm_lewis") is None
    assert loop.voice == "bm_lewis"
    loop._slash("/speed 1.5")
    assert loop.speed == 1.5
    loop._slash("/mode wake")
    assert loop.m.mode == "wake"
    loop._slash("/reset")
    assert loop.brain.was_reset
    loop._slash("/system be terse")
    assert loop.brain.system == "be terse"
    assert loop._slash("/quit") == "quit"


def test_memory_slash(capsys):
    loop = _make_loop("vad")
    loop._slash("/memory")                        # FakeBrain has no .memory
    assert "no memory store" in capsys.readouterr().out

    class FakeMemory:
        def __init__(self):
            self.rows = [
                {"id": 2, "text": "likes green tea",
                 "created": time.time() - 90, "recalled": 3},
                {"id": 1, "text": "x" * 80,
                 "created": time.time() - 90000, "recalled": 0},
            ]
            self.cleared = False

        def count(self):
            return len(self.rows)

        def list_all(self, limit=None):
            return self.rows[:limit]

        def delete(self, mem_id):
            return mem_id == 2

        def clear(self):
            self.cleared = True
            return 2

    mem = FakeMemory()
    loop.brain.memory = mem
    loop._slash("/memory")
    out = capsys.readouterr().out
    assert "2 memories" in out and "#2" in out and "x3" in out
    assert "..." in out                           # long text truncated

    loop._slash("/memory forget 2")
    assert "forgot #2" in capsys.readouterr().out
    loop._slash("/memory forget 9")
    assert "no memory #9" in capsys.readouterr().out
    loop._slash("/memory forget nope")
    assert "usage: /memory forget ID" in capsys.readouterr().out

    loop._slash("/memory clear")
    assert "confirm with: /memory clear yes" in capsys.readouterr().out
    assert mem.cleared is False
    loop._slash("/memory clear yes")
    assert "cleared 2 memories" in capsys.readouterr().out
    assert mem.cleared is True

    loop._slash("/memory bogus")
    assert "usage: /memory" in capsys.readouterr().out


# cmd_talk plumbing (pure helpers)
def test_merged_settings_precedence():
    from gmlx.config import TalkCfg, TalkVad
    from gmlx.talk import _build_parser, _merged_settings
    cfg = TalkCfg(voice="am_adam", mode="vad",
                  vad=TalkVad(silence_ms=700.0))
    args = _build_parser("t").parse_args(["--voice", "af_bella"])
    s = _merged_settings(args, cfg)
    assert s["voice"] == "af_bella"               # flag wins
    assert s["mode"] == "vad"                     # config survives
    assert s["silence_ms"] == 700.0
    assert s["chime"] is True
    args = _build_parser("t").parse_args(["--no-chime"])
    assert _merged_settings(args, cfg)["chime"] is False


def test_pick_model_order():
    from gmlx.talk import _pick_model
    caps = {"default": "d", "chat_ids": ["a", "b", "d"]}
    assert _pick_model("x@coding", caps) == "x@coding"
    assert _pick_model(None, caps) == "d"
    assert _pick_model(None, {"chat_ids": ["only"]}) == "only"
    assert _pick_model(None, {"chat_ids": ["a", "b"]}) is None


def test_capability_guidance(capsys):
    from gmlx.talk import _capability_guidance
    ok = _capability_guidance({"stt": True, "tts": True}, needs_stt=True,
                              explicit_url=False, out=sys.stdout)
    assert ok
    ok = _capability_guidance({"stt": False, "tts": True}, needs_stt=True,
                              explicit_url=False, out=sys.stdout)
    assert not ok
    out = capsys.readouterr().out
    assert "stt" in out and "gmlx[talk]" in out
    # text mode does not need stt
    assert _capability_guidance({"stt": False, "tts": True}, needs_stt=False,
                                explicit_url=False, out=sys.stdout)


import sys  # noqa: E402  (used by the capability tests above)


def test_list_voices_explains_missing_tts(tmp_path, monkeypatch, capsys):
    """`--list-voices` against a server with no TTS configured points at the
    config fix, not at a missing route (which reads as a version mismatch)."""
    from gmlx import launch as launch_mod
    from gmlx import talk_client as tc
    from gmlx.talk import cmd_talk

    cfg = tmp_path / "gmlx.yaml"
    cfg.write_text("models: {}\n")
    monkeypatch.setattr(launch_mod, "_ensure_server", lambda ns: None)
    caps = {"stt": False, "tts": False, "chat_ids": ["m"], "default": "m"}
    monkeypatch.setattr(tc, "probe_capabilities", lambda *a, **k: caps)
    monkeypatch.setattr(
        tc, "list_voices",
        lambda *a, **k: pytest.fail("must not probe the route without tts"))

    assert cmd_talk(["--list-voices", "--config", str(cfg)]) == 1
    err = capsys.readouterr().err
    assert "no tts service enabled" in err
    assert "tts: true" in err                     # the fix-it guidance

    caps["tts"] = True
    monkeypatch.setattr(tc, "list_voices", lambda *a, **k: ["af_heart"])
    assert cmd_talk(["--list-voices", "--config", str(cfg)]) == 0
    assert capsys.readouterr().out.strip() == "af_heart"


def test_slash_wake_swaps_detector(capsys):
    calls = []

    def factory(phrase, *, threshold):
        calls.append((phrase, threshold))
        w = FakeWake()
        w.name = phrase
        return w, None

    loop = _make_loop("wake", wake_threshold=0.7, make_wake=factory)
    old = loop.wake
    loop._slash("/wake okay computer")
    assert calls == [("okay computer", 0.7)]
    assert loop.wake is not old and loop.wake.name == "okay computer"
    loop._slash("/wake")                          # bare: show current phrase
    out = capsys.readouterr().out
    assert 'wake phrase -> "okay computer"' in out
    assert 'wake phrase: "okay computer"' in out


def test_slash_wake_failure_keeps_old_detector(capsys):
    loop = _make_loop("wake",
                      make_wake=lambda p, *, threshold: (None, "can't spell"))
    old = loop.wake
    loop._slash("/wake qzx")
    assert loop.wake is old                       # rebuild failed -> unchanged
    assert "can't spell" in capsys.readouterr().out


def test_slash_wake_outside_wake_mode_hints(capsys):
    def factory(phrase, *, threshold):
        w = FakeWake()
        w.name = phrase
        return w, None

    loop = _make_loop("vad", make_wake=factory)
    assert loop.wake is None
    loop._slash("/mode wake")                     # no detector yet -> hint
    loop._slash("/mode vad")
    loop._slash("/wake computer")                 # set one while in vad
    out = capsys.readouterr().out
    assert "set a phrase with /wake" in out
    assert "switch with /mode wake" in out
    assert loop.wake is not None


def test_merged_settings_brain_and_assistant():
    from gmlx.config import AssistantCfg, TalkCfg
    from gmlx.talk import _build_parser, _merged_settings
    cfg = TalkCfg(brain="assistant",
                  assistant=AssistantCfg(max_tool_rounds=3))
    args = _build_parser("t").parse_args([])
    s = _merged_settings(args, cfg)
    assert s["brain"] == "assistant"              # config value
    assert s["assistant"].max_tool_rounds == 3
    args = _build_parser("t").parse_args(["--brain", "chat"])
    assert _merged_settings(args, cfg)["brain"] == "chat"   # flag wins


def test_tts_worker_merges_ready_sentences():
    """Sentences already decoded when the worker gets to them are folded into
    one synthesis call - the reply is spoken with the text's own cadence, not
    one isolated utterance (and one round trip) per sentence."""
    import threading
    from gmlx.talk import _TURN_END

    loop = _make_loop("vad")
    cancel = threading.Event()
    for s in ("One two three four five six seven.", "Eight nine ten.",
              "Eleven twelve."):
        loop.tts_q.put((cancel, s))
    loop.tts_q.put(_TURN_END)
    loop.tts_q.put(None)
    loop._tts_worker()
    merged = "One two three four five six seven. Eight nine ten. " \
             "Eleven twelve."
    assert loop._said == [merged]
    assert loop.play_q.get_nowait()[1] == merged
    assert loop.play_q.get_nowait() is _TURN_END


def test_tts_worker_merge_respects_cap_and_turns():
    """The merge stops at _MERGE_CHARS and never crosses into another turn's
    chunks; the boundary item is processed on its own, not dropped."""
    import threading
    from gmlx.talk import _MERGE_CHARS

    loop = _make_loop("vad")
    c1, c2 = threading.Event(), threading.Event()
    big = "a" * (_MERGE_CHARS + 1)
    loop.tts_q.put((c1, big))
    loop.tts_q.put((c1, "Same turn tail."))
    loop.tts_q.put((c2, "Next turn."))
    loop.tts_q.put(None)
    loop._tts_worker()
    assert loop._said == [big, "Same turn tail.", "Next turn."]


def test_tts_worker_merged_chunk_skipped_when_canceled():
    """A cancel that lands after the merge drain must still silence the
    whole merged utterance."""
    import threading

    loop = _make_loop("vad")
    cancel = threading.Event()
    loop.tts_q.put((cancel, "Never say this."))
    loop.tts_q.put((cancel, "Nor this."))
    cancel.set()
    loop.tts_q.put(None)
    loop._tts_worker()
    assert loop._said == []
    assert loop.play_q.empty()


def test_playback_survives_backend_play_error(capsys):
    from gmlx.talk import IDLE, LISTENING

    loop = _make_loop("wake")
    real_play = loop.backend.play
    fails = {"n": 0}

    def flaky_play(pcm, rate, stop=None, slice_ms=150.0):
        if fails["n"] == 0:
            fails["n"] += 1
            raise RuntimeError("PortAudioError: device gone")
        return real_play(pcm, rate, stop=stop, slice_ms=slice_ms)

    loop.backend.play = flaky_play
    loop.start_threads()
    try:
        for _ in range(2):
            loop.backend.start_input(loop.on_frame)
            loop.on_frame(_wake_frame())
            _pump_until(loop, lambda: loop.m.state == LISTENING)
            for _ in range(5):
                loop.on_frame(_loud())
            for _ in range(8):
                loop.on_frame(_quiet())
            # the thread must keep consuming _TURN_END so playback_idle
            # still fires and the turn completes
            _pump_until(loop, lambda: (loop.m.state == IDLE
                                       and not loop.m.turn_active
                                       and not loop.m.speaking))
    finally:
        loop.shutdown()
    assert fails["n"] == 1
    assert len(loop.backend.played) >= 1  # second turn actually played


def test_mode_switch_out_of_text_opens_mic():
    """A runtime /mode switch from text must lazily open the input stream
    and build a VAD (text-mode startup created neither)."""
    loop = _make_loop("text")
    loop.vad = None
    assert loop.backend.on_frame is None
    loop._slash("/mode vad")
    assert loop.m.mode == "vad"
    assert loop.backend.on_frame is not None  # mic opened
    assert loop.vad is not None


def test_mode_switch_reverts_when_mic_fails():
    loop = _make_loop("text")

    def broken_start(on_frame, **kw):
        raise RuntimeError("no input device")

    loop.backend.start_input = broken_start
    loop._slash("/mode vad")
    assert loop.m.mode == "text"


def test_keys_poll_collapses_an_escape_sequence(monkeypatch):
    """An arrow key is ESC [ A. poll() read one byte, so the '[' came back as a
    printable key and _handle_key opened a blocking line-input prompt."""
    import os as _os
    import sys as _sys

    from gmlx.talk import _Keys

    r_fd, w_fd = _os.pipe()
    try:
        keys = _Keys()
        keys.fd = r_fd
        monkeypatch.setattr(_sys, "stdin", _os.fdopen(r_fd, "rb", buffering=0))

        _os.write(w_fd, b"\x1b[A")             # up arrow
        assert keys.poll() == "\x1b"           # the whole sequence, once
        assert keys.poll() is None              # tail drained, nothing left

        _os.write(w_fd, b"q")
        assert keys.poll() == "q"

        _os.write(w_fd, b"\x1b")               # a bare Esc still cancels
        assert keys.poll() == "\x1b"
    finally:
        _os.close(w_fd)
