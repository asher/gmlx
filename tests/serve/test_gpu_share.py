#!/usr/bin/env python3
"""Chat decode beside an image job: the chat step marks, the share
setting, and the pacer that lets a waiting chat step run between two
pieces of an image forward. Fake chat steps and a fake evaluation, so
nothing runs on the GPU."""
from __future__ import annotations

import threading
import time

import pytest

from gmlx.image import pacing
from gmlx.serve import gpu_share
from gmlx.serve.gpu_share import ChatActivity, ImagePacer


class _Clock:
    def __init__(self):
        self.t = 128.0

    def __call__(self):
        return self.t


def test_chat_is_active_while_a_step_runs_and_shortly_after():
    clock = _Clock()
    chat = ChatActivity(idle_s=0.25, clock=clock)
    assert not chat.active()
    chat.begin()
    clock.t += 5.0
    assert chat.active()
    chat.end()
    assert chat.ticks == 1 and chat.active()
    clock.t += 0.2
    assert chat.active()
    clock.t += 0.1
    assert not chat.active()


def test_a_point_without_a_pacer_does_nothing_and_a_pacer_is_scoped():
    seen = []
    pacing.point(1, 2)
    with pacing.paced(seen.append):
        pacing.point(1, 2)
        with pacing.paced(None):
            pacing.point(3)
        pacing.point()
    pacing.point(4)
    assert seen == [(1, 2), ()]


def test_the_pacer_does_nothing_with_no_chat_active():
    evals = []
    pacer = ImagePacer(ChatActivity(), share=0.5,
                       evaluate=lambda *a: evals.append(a))
    t0 = time.monotonic()
    for i in range(50):
        pacer((i,))
    assert evals == [] and pacer.pauses == 0
    assert time.monotonic() - t0 < 2


def _run(share: float, blocks: int, chat_delay: float):
    """An image forward of ``blocks`` blocks under a pacer, beside a chat
    engine whose steps each wait for the GPU while an image piece holds
    it, so each step can run only when the image pauses. Returns the
    order of the GPU work."""
    chat = ChatActivity()
    log, lock = [], threading.Lock()
    gpu_free = threading.Event()
    stop = threading.Event()

    def evaluate(*arrays):
        with lock:
            log.append(("piece", arrays[0]))
        gpu_free.set()

    def engine():
        while not stop.is_set():
            chat.begin()
            # The step's command buffers queue behind the image piece.
            if not gpu_free.wait(0.5):
                chat.end()
                continue
            gpu_free.clear()
            time.sleep(chat_delay)
            with lock:
                log.append(("chat",))
            chat.end()

    th = threading.Thread(target=engine, daemon=True)
    chat.begin()
    chat.end()                      # chat is active before the job starts
    th.start()
    pacer = ImagePacer(chat, share=share, evaluate=evaluate)
    with pacing.paced(pacer):
        for i in range(blocks):
            pacing.point(i)
    stop.set()
    th.join(2)
    return log, pacer


def test_a_waiting_chat_step_runs_before_the_next_image_piece():
    log, pacer = _run(share=0.0, blocks=4, chat_delay=0.02)
    pieces = [i for i, e in enumerate(log) if e[0] == "piece"]
    assert [log[i][1] for i in pieces] == [0, 1, 2, 3]
    # Between every two pieces a chat step ran.
    for a, b in zip(pieces, pieces[1:]):
        assert ("chat",) in log[a + 1:b], log
    assert pacer.pauses == 4


def test_a_larger_share_gives_chat_more_steps_per_pause():
    clock = _Clock()
    chat = ChatActivity(clock=clock)
    chat.begin()
    chat.end()
    steps = []

    def wait_tick(timeout):
        steps.append(1)
        clock.t += 1 / 128         # each chat step takes 7.8 ms
        return True

    chat.wait_tick = wait_tick
    pacer = ImagePacer(chat, share=0.5, clock=clock, evaluate=lambda *a: None)
    pacer(())                       # the first pause pays for no image time
    assert len(steps) == 1
    clock.t += 5 / 128              # 39 ms of image work
    steps.clear()
    pacer(())
    assert len(steps) == 5          # as much chat time as image time
    pacer.ratio = 0.5               # a share of 1/3
    clock.t += 5 / 128
    steps.clear()
    pacer(())
    assert len(steps) == 3


def _ticking(clock, steps):
    chat = ChatActivity(clock=clock)
    chat.begin()
    chat.end()

    def wait_tick(timeout):
        steps.append(1)
        clock.t += 1 / 128
        return True

    chat.wait_tick = wait_tick
    return chat


def test_one_pause_pays_for_at_most_the_longest_piece():
    clock, steps = _Clock(), []
    chat = _ticking(clock, steps)
    pacer = ImagePacer(chat, share=0.5, clock=clock, evaluate=lambda *a: None)
    pacer(())
    clock.t += 4.0                  # host work between two steps
    chat.last_end = clock.t         # chat stayed active through it
    steps.clear()
    pacer(())
    assert len(steps) == round(gpu_share._MAX_PIECE_S * 128)


def test_chat_that_goes_idle_starts_the_next_pause_afresh():
    clock, steps = _Clock(), []
    chat = _ticking(clock, steps)
    pacer = ImagePacer(chat, share=0.5, clock=clock, evaluate=lambda *a: None)
    pacer(())
    clock.t += 1.0                  # chat goes idle, the image runs alone
    pacer(())
    assert pacer.mark is None
    chat.last_end = clock.t         # chat comes back
    steps.clear()
    pacer(())
    # The image time while chat was idle is not paid for: one step.
    assert len(steps) == 1


def test_a_pause_stops_waiting_when_chat_goes_idle():
    chat = ChatActivity()
    chat.begin()
    chat.end()
    pacer = ImagePacer(chat, share=0.9, wait_s=0.01, evaluate=lambda *a: None)
    t0 = time.monotonic()
    pacer(())
    pacer.mark = time.monotonic() - 0.2
    pacer(())
    assert time.monotonic() - t0 < 1.2


@pytest.mark.parametrize("raw, want", [
    (None, gpu_share.DEFAULT_SHARE), ("", gpu_share.DEFAULT_SHARE),
    ("0", 0.0), ("0.25", 0.25), ("0.9", 0.9), ("off", None), ("OFF", None)])
def test_chat_share_values(monkeypatch, raw, want):
    monkeypatch.delenv("GMLX_IMAGE_CHAT_SHARE", raising=False)
    if raw is not None:
        monkeypatch.setenv("GMLX_IMAGE_CHAT_SHARE", raw)
    assert gpu_share.chat_share() == want
    assert (gpu_share.pacer() is None) == (want is None)


@pytest.mark.parametrize("raw", ["1", "-0.1", "lots", "nan"])
def test_a_bad_chat_share_warns_and_takes_the_default(raw):
    with pytest.warns(UserWarning, match="GMLX_IMAGE_CHAT_SHARE"):
        assert gpu_share.chat_share(raw) == gpu_share.DEFAULT_SHARE


def test_install_marks_the_engine_steps():
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar

    assert gpu_share.install_chat_ticks()
    assert gpu_share.install_chat_ticks()
    for cls, name in ((ar.GenerationBatch, "_step"),
                      (ar.SpeculativeGenerationBatch, "next"),
                      (ar.PromptProcessingBatch, "prompt_step"),
                      (ar.PromptProcessingBatch, "generate")):
        fn = getattr(cls, name)
        assert fn._gmlx_chat_tick, (cls, name)
        assert not getattr(fn.__wrapped__, "_gmlx_chat_tick", False)


def test_a_marked_step_counts_even_when_it_raises():
    chat = ChatActivity()
    orig = gpu_share.CHAT
    gpu_share.CHAT = chat
    try:
        class Batch:
            def _step(self):
                raise RuntimeError("boom")

        class Mod:
            GenerationBatch = Batch
            SpeculativeGenerationBatch = type("S", (), {"next": lambda s: 1})
            PromptProcessingBatch = type("P", (), {
                "prompt_step": lambda s: 2, "generate": lambda s: 3})

        import sys
        import types

        fake = types.ModuleType("gmlx._vendor.mlx_vlm.generate.ar")
        for k in ("GenerationBatch", "SpeculativeGenerationBatch",
                  "PromptProcessingBatch"):
            setattr(fake, k, getattr(Mod, k))
        pkg = sys.modules.get("gmlx._vendor.mlx_vlm.generate")
        saved = sys.modules.get("gmlx._vendor.mlx_vlm.generate.ar")
        done = dict(gpu_share._installed)
        gpu_share._installed["done"] = False
        sys.modules["gmlx._vendor.mlx_vlm.generate.ar"] = fake
        if pkg is not None:
            saved_attr = getattr(pkg, "ar", None)
            pkg.ar = fake
        try:
            gpu_share.install_chat_ticks()
            with pytest.raises(RuntimeError):
                Batch()._step()
            assert Mod.PromptProcessingBatch().generate() == 3
        finally:
            gpu_share._installed.update(done)
            if saved is not None:
                sys.modules["gmlx._vendor.mlx_vlm.generate.ar"] = saved
            else:
                sys.modules.pop("gmlx._vendor.mlx_vlm.generate.ar", None)
            if pkg is not None:
                pkg.ar = saved_attr
        assert chat.busy == 0 and chat.ticks == 2
    finally:
        gpu_share.CHAT = orig
