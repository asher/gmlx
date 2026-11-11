# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""``gmlx chat`` spoken replies: the player, the event adapter and the
``/speak`` command, with a scripted sound device and scripted events."""

from __future__ import annotations

import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from gmlx.tui import chat_speech as cs

RATE = 24000


class _Device:
    """A sound device that records what it plays. With ``gate`` it holds
    the first clip until the gate opens."""

    def __init__(self, gate=None, fails=False):
        self.played, self.gate, self.fails = [], gate, fails
        self.started = threading.Event()

    def play(self, pcm, rate, *, stop=None):
        self.started.set()
        if self.fails:
            raise RuntimeError("device lost")
        if self.gate is not None:
            self.gate.wait()
        if stop is not None and stop.is_set():
            return False
        self.played.append((pcm.copy(), rate))
        return True


def _clip(value: float, n: int = 4) -> np.ndarray:
    return np.full(n, value, np.float32)


def _event(text="", audio=None, **summary):
    base = dict(token=1, text=text, audio=audio, finish_reason=None,
                prompt_tokens=0, text_tokens=0, frames=0, prompt_seconds=0.0,
                generation_seconds=0.0)
    return SimpleNamespace(**{**base, **summary})


def _state(**kw):
    base = dict(speak_refusal=None, speak_play=False, speak_dir=None,
                speak_player=None)
    return SimpleNamespace(**{**base, **kw})


# -- the player ---------------------------------------------------------------


def test_the_player_plays_what_it_is_fed_in_order():
    device = _Device()
    player = cs.Player(device)
    player.begin()
    for value in (0.1, 0.2, 0.3):
        player.feed(_clip(value), RATE)
    player.drain()
    player.close()
    assert [int(p[0]) for p, _r in device.played] == [3276, 6553, 9830]
    assert {r for _p, r in device.played} == {RATE}
    assert device.played[0][0].dtype == np.int16


def test_samples_past_full_scale_are_clipped():
    assert cs.pcm16(np.array([2.0, -2.0, 0.0], np.float32)).tolist() == [
        32767, -32767, 0]


def test_stop_cuts_the_clip_that_plays_and_drops_the_rest():
    gate = threading.Event()
    device = _Device(gate)
    player = cs.Player(device)
    player.begin()
    player.feed(_clip(0.1), RATE)
    player.feed(_clip(0.2), RATE)
    device.started.wait()
    player.stop()
    gate.set()
    player.drain()
    player.close()
    assert device.played == []


def test_a_new_reply_drops_the_last_ones_tail_and_plays_its_own():
    gate = threading.Event()
    device = _Device(gate)
    player = cs.Player(device)
    player.begin()
    player.feed(_clip(0.1), RATE)
    player.feed(_clip(0.2), RATE)
    device.started.wait()
    player.begin()
    player.feed(_clip(0.5), RATE)
    gate.set()
    player.drain()
    player.close()
    assert [int(p[0]) for p, _r in device.played] == [16383]


def test_a_device_error_ends_the_sound_and_not_the_thread():
    device = _Device(fails=True)
    player = cs.Player(device)
    player.begin()
    player.feed(_clip(0.1), RATE)
    player.drain()
    # The thread still serves the queue: a later reply drains too.
    player.begin()
    player.feed(_clip(0.2), RATE)
    player.drain()
    player.close()
    assert device.played == []


def test_no_sounddevice_names_the_talk_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(cs.SpeakUnavailable, match="talk"):
        cs.open_player()


# -- the reply ----------------------------------------------------------------


def _reply_events():
    return [
        _event("Hel"),
        _event("lo."),
        _event(audio=_clip(0.1)),
        _event(audio=_clip(0.2)),
        _event(" Bye.", token=None, finish_reason="stop", prompt_tokens=30,
               text_tokens=2, frames=2, prompt_seconds=0.5,
               generation_seconds=2.0),
    ]


def test_a_spoken_reply_is_its_text_chunks_and_its_audio():
    ticks = iter([10.0, 10.5, 11.0, 12.0])
    device = _Device()
    player = cs.Player(device)
    reply = cs.SpokenReply(RATE, player, clock=lambda: next(ticks))
    chunks = list(reply.chunks(_reply_events()))
    player.drain()
    player.close()
    assert [c.text for c in chunks] == ["Hel", "lo.", "", "", " Bye."]
    # A frame is a slot of the reply, as a text token is.
    assert [c.generation_tokens for c in chunks] == [1, 2, 3, 4, 4]
    assert chunks[1].generation_tps == pytest.approx(4.0)
    last = chunks[-1]
    assert (last.finish_reason, last.prompt_tokens) == ("stop", 30)
    assert last.prompt_tps == pytest.approx(60.0)
    assert last.generation_tps == pytest.approx(2.0)
    assert reply.frames == 2
    assert reply.seconds == pytest.approx(8 / RATE)
    assert len(device.played) == 2


def test_a_reply_without_a_player_keeps_its_audio(tmp_path):
    reply = cs.SpokenReply(RATE)
    list(reply.chunks(_reply_events()))
    first = reply.save(str(tmp_path / "out"))
    second = reply.save(str(tmp_path / "out"))
    assert first.endswith("reply-001.wav") and second.endswith("reply-002.wav")
    with wave.open(first) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (
            RATE, 1, 8)


def test_a_reply_with_no_audio_writes_no_file(tmp_path):
    reply = cs.SpokenReply(RATE)
    list(reply.chunks([_event("Hi."), _event(token=None, finish_reason="stop")]))
    assert reply.save(str(tmp_path)) is None
    assert list(tmp_path.iterdir()) == []


def test_cancel_cuts_the_sound():
    gate = threading.Event()
    device = _Device(gate)
    player = cs.Player(device)
    reply = cs.SpokenReply(RATE, player)
    stream = reply.chunks(_reply_events())
    for _ in range(4):
        next(stream)
    device.started.wait()
    reply.cancel()
    gate.set()
    player.drain()
    player.close()
    assert device.played == []


# -- the conversation ---------------------------------------------------------


def test_the_instruction_to_speak_leads_a_conversation_with_no_system_message():
    history = [{"role": "user", "content": "hi"}]
    assert cs.messages(history, "SPEAK") == [
        {"role": "system", "content": "SPEAK"}, *history]
    assert history == [{"role": "user", "content": "hi"}]


def test_a_conversation_with_a_system_message_is_left_as_it_is():
    history = [{"role": "system", "content": "Be brief."},
               {"role": "user", "content": "hi"}]
    assert cs.messages(history, "SPEAK") is history


# -- /speak -------------------------------------------------------------------


def test_speak_on_opens_the_player_once(capsys):
    opened = []
    state = _state()

    def opener():
        opened.append(SimpleNamespace(stop=lambda: None))
        return opened[-1]

    cs.command("on", state, opener)
    cs.command("on", state, opener)
    assert state.speak_play and len(opened) == 1
    assert "replies are spoken and played" in capsys.readouterr().out


def test_speak_on_without_a_sound_device_points_at_save(capsys):
    state = _state()

    def opener():
        raise cs.SpeakUnavailable("playing sound needs the talk extra: X")

    cs.command("on", state, opener)
    out = capsys.readouterr().out
    assert not state.speak_play
    assert "needs the talk extra" in out and "/speak save DIR" in out


def test_speak_save_takes_a_folder(tmp_path, capsys):
    state = _state()
    cs.command(f"save {tmp_path}/clips", state)
    assert state.speak_dir == f"{tmp_path}/clips" and not state.speak_play
    assert f"saved to {tmp_path}/clips" in capsys.readouterr().out
    cs.command("save", state)
    assert "usage: /speak" in capsys.readouterr().out


def test_speak_off_stops_the_sound_and_both_outputs(capsys):
    stopped = []
    state = _state(speak_play=True, speak_dir="/tmp/x",
                   speak_player=SimpleNamespace(stop=lambda: stopped.append(1)))
    cs.command("off", state)
    assert (state.speak_play, state.speak_dir, stopped) == (False, None, [1])
    assert "replies are text" in capsys.readouterr().out


def test_speak_alone_shows_the_state_and_a_wrong_word_the_usage(capsys):
    state = _state(speak_dir="/tmp/x")
    cs.command("", state)
    assert "saved to /tmp/x" in capsys.readouterr().out
    cs.command("loud", state)
    assert "usage: /speak" in capsys.readouterr().out
    assert state.speak_dir == "/tmp/x"


def test_speak_is_refused_on_a_model_that_does_not_speak(capsys):
    state = _state(speak_refusal="this model has no speech output")
    cs.command("on", state, lambda: pytest.fail("opened a player"))
    assert not state.speak_play
    assert "no speech output" in capsys.readouterr().out
