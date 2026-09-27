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


# The step lane on an autoregressive engine. A fake batch loop calls
# _collect_pending_requests as _run_impl does, drains cancellations after
# each call, and records the chat items it gets.

class _Loop:
    def __init__(self, rg):
        self.rg = rg
        self.active = False
        self.capacity = None
        self.calls = 0
        self.items = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        rg = self.rg
        while not rg._stop:
            items, stop = rg._collect_pending_requests(
                active=self.active, capacity=self.capacity, idle_timeout=0.01)
            self.calls += 1
            rg._drain_cancellations()
            self.items += items
            if stop:
                return


@pytest.fixture
def ar_engine():
    install_engine_jobs()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop = False
    rg.requests = queue.Queue()
    rg._cancel_lock = threading.Lock()
    rg._cancelled = set()
    rg._ready = threading.Event()
    rg._ready.set()
    rg._load_error = None
    rg.model = rg.processor = rg.config = None
    loop = _Loop(rg)
    loop.thread.start()
    yield loop
    rg._stop = True
    rg.requests.put(None)
    loop.thread.join(timeout=5)


def _steps(n, value=None, seen=None, loop=None):
    def job(rg, should_stop):
        def step():
            for _ in range(n):
                if seen is not None:
                    seen.append(loop.calls if loop is not None else None)
                yield
            return value
        return step()
    return job


def _run_steps(rg, job, **kw):
    return run_on_engine(rg, job, request_id="t", prompt_tokens=3,
                         stop=kw.pop("stop", None) or threading.Event(),
                         timeout_s=kw.pop("timeout_s", 5), stepwise=True, **kw)


def _chat(rg):
    rq = queue.Queue()
    item = _GEN.QueuedGenerationRequest(rqueue=rq, raw_inputs={"input_ids": None},
                                        prompt_tokens=1,
                                        args=_GEN.GenerationArguments(max_tokens=1))
    rg.requests.put(item)
    return item


def test_a_step_job_returns_its_value(ar_engine):
    assert _run_steps(ar_engine.rg, _steps(3, value=42)) == 42


def test_one_step_runs_per_call_while_chat_is_active(ar_engine):
    ar_engine.active = True
    seen = []
    _run_steps(ar_engine.rg, _steps(4, seen=seen, loop=ar_engine))
    assert len(seen) == 4 and len(set(seen)) == 4


def _settle(loop):
    """Wait until the loop has started a call with its current settings."""
    start = loop.calls
    deadline = time.time() + 5
    while loop.calls < start + 2 and time.time() < deadline:
        time.sleep(0.005)


def test_a_job_runs_while_the_batch_is_full(ar_engine):
    ar_engine.active, ar_engine.capacity = True, 0
    _settle(ar_engine)
    chat = _chat(ar_engine.rg)
    assert _run_steps(ar_engine.rg, _steps(2, value="done")) == "done"
    assert ar_engine.items == [] and list(ar_engine.rg.requests.queue) == [chat]


def test_chat_items_reach_the_loop_and_jobs_do_not(ar_engine):
    chat = _chat(ar_engine.rg)
    assert _run_steps(ar_engine.rg, _steps(1, value=1)) == 1
    deadline = time.time() + 5
    while not ar_engine.items and time.time() < deadline:
        time.sleep(0.01)
    assert ar_engine.items == [chat]


def test_a_failing_step_is_forwarded_and_the_loop_goes_on(ar_engine):
    def job(rg, should_stop):
        def step():
            yield
            raise ValueError("bad read")
        return step()

    with pytest.raises(ValueError, match="bad read"):
        _run_steps(ar_engine.rg, job)
    assert _run_steps(ar_engine.rg, _steps(1, value=2)) == 2


def test_the_stop_event_cancels_a_running_job_and_an_engine_cancel_does_not():
    install_engine_jobs()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop, rg.requests, rg.model = False, queue.Queue(), None
    rg._cancel_lock, rg._cancelled = threading.Lock(), {7}
    stop = threading.Event()
    rq = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs={"_kq_engine_step_job": (_steps(10), stop)},
        prompt_tokens=1, args=_GEN.GenerationArguments(max_tokens=1)))
    rg._collect_pending_requests(active=True)
    ctx = rq.get_nowait()
    rg._cancelled.add(ctx.uid)
    rg._collect_pending_requests(active=True)
    assert rq.empty()                    # the uid cancel did nothing
    assert rg._cancelled == {7, ctx.uid}  # and chat's cancels are untouched
    stop.set()
    rg._collect_pending_requests(active=True)
    assert isinstance(rq.get_nowait().exc, JobCancelled)


def test_a_keyboard_interrupt_is_not_swallowed():
    install_engine_jobs()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop, rg.requests, rg.model = False, queue.Queue(), None

    def job(rg, should_stop):
        def step():
            raise KeyboardInterrupt
            yield
        return step()

    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=queue.Queue(), raw_inputs={"_kq_engine_step_job": (job, threading.Event())},
        prompt_tokens=1, args=_GEN.GenerationArguments(max_tokens=1)))
    rg._collect_pending_requests(active=True)
    with pytest.raises(KeyboardInterrupt):
        rg._collect_pending_requests(active=True)


def test_a_step_leaves_rng_rope_state_and_buffer_caps_as_they_were(monkeypatch):
    import types

    import mlx.core as mx

    from gmlx.serve import cb_phase

    install_engine_jobs()
    flips = []
    monkeypatch.setattr(cb_phase, "flip", lambda p: flips.append(p))
    monkeypatch.setattr(cb_phase, "phase", lambda: "decode")
    lm = types.SimpleNamespace(_rope_deltas="chat")
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop, rg.requests = False, queue.Queue()
    rg.model = types.SimpleNamespace(language_model=lm, _position_ids="chat-ids")
    rg.__dict__["_kq_engine_is_diffusion"] = False
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)

    def job(rg_, should_stop):
        def step():
            lm._rope_deltas = "job"
            rg_.model._position_ids = "job-ids"
            yield
        return step()

    mx.random.seed(3)
    before = mx.random.state[0]
    rq = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs={"_kq_engine_step_job": (job, threading.Event())},
        prompt_tokens=1, args=_GEN.GenerationArguments(max_tokens=1)))
    rg._collect_pending_requests(active=True)
    rg._collect_pending_requests(active=True)
    assert (lm._rope_deltas, rg.model._position_ids) == ("chat", "chat-ids")
    assert flips == ["prefill", "decode"]
    assert mx.array_equal(mx.random.state[0], before).item()


def test_a_diffusion_engine_passes_step_jobs_through():
    install_engine_jobs()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop, rg.requests, rg.model = False, queue.Queue(), None
    rg.__dict__["_kq_engine_is_diffusion"] = True
    item = _GEN.QueuedGenerationRequest(
        rqueue=queue.Queue(), raw_inputs={"_kq_engine_step_job": (_steps(1), threading.Event())},
        prompt_tokens=1, args=_GEN.GenerationArguments(max_tokens=1))
    rg.requests.put(item)
    pending, _ = rg._collect_pending_requests(active=True)
    assert pending == [item]


def test_a_stopping_engine_fails_its_pending_jobs():
    install_engine_jobs()
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._stop, rg.requests, rg.model = False, queue.Queue(), None
    rq = queue.Queue()
    rg.requests.put(_GEN.QueuedGenerationRequest(
        rqueue=rq, raw_inputs={"_kq_engine_step_job": (_steps(10), threading.Event())},
        prompt_tokens=1, args=_GEN.GenerationArguments(max_tokens=1)))
    rg._collect_pending_requests(active=True)
    rq.get_nowait()                          # the generation context
    rg._stop = True
    rg._collect_pending_requests(active=True)
    err = rq.get_nowait()
    assert isinstance(err.exc, RuntimeError) and "stopped" in str(err.exc)
