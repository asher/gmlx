"""The engine-thread job lane: a job queued on a diffusion model's request
queue runs in place of a generation on the real mlx-vlm engine loop. No
model is loaded."""

from __future__ import annotations

import importlib
import queue
import threading
import time

import pytest

pytest.importorskip("mlx_vlm")

from gmlx.serve.engine_jobs import (  # noqa: E402
    JobCancelled,
    JobTimeout,
    install_engine_jobs,
    run_on_engine,
)
from gmlx.systemone.reads import Cancelled  # noqa: E402

_GEN = importlib.import_module("mlx_vlm.server.generation")


@pytest.fixture
def engine(monkeypatch):
    """A ResponseGenerator shell with no model, its _run_diffusion loop on
    a thread, and the job lane installed over a recording original."""
    cls = _GEN.ResponseGenerator
    calls = []

    def original(self, uid, rqueue, raw_inputs, args, cancelled, log_state=None):
        calls.append(raw_inputs)
        rqueue.put("generated")

    monkeypatch.setattr(cls, "_generate_diffusion", original)
    install_engine_jobs()
    install_engine_jobs()
    rg = object.__new__(cls)
    rg._stop = False
    rg.requests = queue.Queue()
    rg._cancel_lock = threading.Lock()
    rg._cancelled = set()
    rg._ready = threading.Event()
    rg._ready.set()
    rg._load_error = None
    rg.model = rg.processor = rg.config = None
    rg.passthrough_calls = calls
    thread = threading.Thread(target=rg._run_diffusion, daemon=True)
    thread.start()
    yield rg
    rg._stop = True
    rg.requests.put(None)
    thread.join(timeout=5)


def _run(rg, job, *, stop=None, timeout_s=None, prompt_tokens=3):
    return run_on_engine(rg, job, request_id="t", prompt_tokens=prompt_tokens,
                         stop=stop or threading.Event(), timeout_s=timeout_s)


def _block(rg):
    """Occupy the engine with a job until the returned event is set."""
    started, release = threading.Event(), threading.Event()

    def job(_rg, _should_stop):
        started.set()
        release.wait(5)

    thread, _ = _in_thread(lambda: _run(rg, job))
    assert started.wait(5)
    return thread, release


def _in_thread(fn):
    out = {}

    def target():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the test
            out["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t, out


def test_a_normal_request_reaches_the_original(engine):
    rqueue = queue.Queue()
    engine.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rqueue, raw_inputs={"input_ids": None}, prompt_tokens=1,
        args=_GEN.GenerationArguments(max_tokens=1)))
    assert isinstance(rqueue.get(timeout=5), _GEN.GenerationContext)
    assert rqueue.get(timeout=5) == "generated"
    assert engine.passthrough_calls == [{"input_ids": None}]


def test_a_job_returns_its_value_and_sees_the_engine(engine):
    seen = {}

    def job(rg, should_stop):
        seen["rg"] = rg
        seen["stop"] = should_stop()
        seen["thread"] = threading.current_thread().name
        return 42

    assert _run(engine, job) == 42
    assert seen["rg"] is engine and seen["stop"] is False
    assert seen["thread"] != threading.current_thread().name
    assert engine.passthrough_calls == []


def test_the_prefill_line_carries_the_prompt_count(engine, caplog):
    caplog.set_level("INFO", logger=_GEN.logger.name)
    _run(engine, lambda rg, s: None, prompt_tokens=123)
    assert any("prompt_tokens=123" in r.getMessage() for r in caplog.records)


def test_a_job_whose_stop_is_set_while_queued_never_runs(engine):
    ran = []
    blocker, release = _block(engine)
    stop = threading.Event()
    waiter, out = _in_thread(lambda: _run(engine, lambda rg, s: ran.append(1),
                                          stop=stop))
    stop.set()
    release.set()
    waiter.join(5)
    blocker.join(5)
    assert isinstance(out.get("error"), JobCancelled)
    assert ran == []


def test_an_engine_cancel_reaches_a_running_job(engine):
    started = threading.Event()

    def job(rg, should_stop):
        started.set()
        while not should_stop():
            started.wait(0.005)
        raise Cancelled("stopped")

    waiter, out = _in_thread(lambda: _run(engine, job))
    assert started.wait(5)
    engine._cancel(1)
    waiter.join(5)
    assert isinstance(out.get("error"), JobCancelled)


def test_a_job_exception_is_raised_as_itself(engine):
    err = ValueError("bad schema")

    def job(rg, should_stop):
        raise err

    with pytest.raises(ValueError) as info:
        _run(engine, job)
    assert info.value is err


def test_a_job_past_its_deadline_times_out_and_is_told_to_stop(engine):
    stopped = threading.Event()

    def job(rg, should_stop):
        while not should_stop():
            stopped.wait(0.005)
        stopped.set()
        raise Cancelled("stopped")

    stop = threading.Event()
    with pytest.raises(JobTimeout):
        _run(engine, job, stop=stop, timeout_s=0.1)
    assert stop.is_set()
    assert stopped.wait(5)


def test_the_deadline_starts_when_the_job_is_dequeued(engine):
    blocker, release = _block(engine)
    started = time.perf_counter()
    waiter, out = _in_thread(lambda: _run(engine, lambda rg, s: "done",
                                          timeout_s=0.1))
    threading.Timer(0.4, release.set).start()
    waiter.join(5)
    blocker.join(5)
    assert out.get("value") == "done", out
    assert time.perf_counter() - started >= 0.3


def test_no_timeout_waits_for_a_slow_job(engine):
    def job(rg, should_stop):
        threading.Event().wait(0.3)
        return "slow"

    assert _run(engine, job, timeout_s=None) == "slow"
