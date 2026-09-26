"""Run a job on a model's engine thread.

mlx-vlm serves each model from one engine thread. A job rides that
thread's request queue: ``run_on_engine`` queues a request whose inputs
carry the job.

On a diffusion model, which takes requests one at a time, the wrapped
``_generate_diffusion`` runs the job in place of a generation. The job owns
the model for its duration, and chat requests queue behind it.

On an autoregressive model the job is a step generator that yields after
each forward. The wrapped ``_collect_pending_requests``, which the batch
loop calls between decode steps, takes the job out of the queue and runs
one step per call, so chat keeps decoding while a job runs. The job must
leave the engine as it found it: it never drains cancellations, never
touches the stop tokens and never reseeds the RNG. Each step runs under
the wired limit and the prefill command-buffer caps, and restores any
rope position state it changed. An engine cancel for the job's uid never
reaches it, since the batch loop drains that uid and finds no row; the
job's stop event is its only cancel.

The caller waits for the engine to dequeue the job with no deadline, as
chat waits for its generation context. The deadline starts at dequeue and
bounds the job itself. A missed deadline sets the job's stop event, which
the job polls, and raises ``JobTimeout``."""

from __future__ import annotations

import contextlib
import importlib
import itertools
import logging
import queue
import threading
from typing import Any, Callable, Optional

import mlx.core as mx
from mlx_vlm.server import generation as _generation

from gmlx.systemone.reads import Cancelled

_JOB_KEY = "_kq_engine_job"
_STEP_JOB_KEY = "_kq_engine_step_job"
_INSTALLED_FLAG = "_kq_engine_jobs"
_JOBS_ATTR = "_kq_step_jobs"
_DIFFUSION_ATTR = "_kq_engine_is_diffusion"
_ROPE_STATE = ("_position_ids", "_rope_deltas")
_job_ids = itertools.count(1)

_log = logging.getLogger(__name__)


class JobCancelled(Cancelled):
    """The job's stop event was set, or the engine cancelled its uid."""


class JobTimeout(Exception):
    """The job did not finish within its deadline."""


class _JobResult:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _JobError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


def run_on_engine(rg, job: Callable[[Any, Callable[[], bool]], Any], *,
                  request_id: str, prompt_tokens: int,
                  stop: threading.Event, timeout_s: Optional[float],
                  stepwise: bool = False) -> Any:
    """Run ``job(response_generator, should_stop)`` on ``rg``'s engine
    thread and return its value. With ``stepwise`` the job returns a
    generator for an autoregressive model's engine, and its return value
    is the result. An exception from the job is raised here as itself.
    ``timeout_s`` of ``None`` waits for the job forever."""
    rg.wait_until_ready()
    rqueue: queue.Queue = queue.Queue()
    request = _generation.QueuedGenerationRequest(
        rqueue=rqueue,
        raw_inputs={_STEP_JOB_KEY if stepwise else _JOB_KEY: (job, stop)},
        prompt_tokens=int(prompt_tokens),
        args=_generation.GenerationArguments(max_tokens=1),
        request_id=request_id,
    )
    rg.requests.put(request)
    ctx = rqueue.get()
    if isinstance(ctx, BaseException):
        raise ctx
    try:
        item = rqueue.get(timeout=timeout_s)
    except queue.Empty:
        stop.set()
        raise JobTimeout(
            f"the decision did not finish within {timeout_s:g} s") from None
    if isinstance(item, _JobError):
        raise item.exc
    if isinstance(item, BaseException):
        raise item
    if isinstance(item, _JobResult):
        return item.value
    raise RuntimeError("the engine job ended without a result")


def install_engine_jobs() -> None:
    """Wrap ``ResponseGenerator._generate_diffusion`` so a queued job runs in
    place of a generation, and ``_collect_pending_requests`` so a step job
    runs between batch steps. Idempotent."""
    _install_diffusion_jobs()
    _install_step_jobs()


def _install_diffusion_jobs() -> None:
    cls = _generation.ResponseGenerator
    original = cls._generate_diffusion
    if getattr(original, _INSTALLED_FLAG, False):
        return

    def _generate_diffusion(self, uid, rqueue, raw_inputs, args, cancelled,
                            log_state=None):
        entry = raw_inputs.get(_JOB_KEY) if isinstance(raw_inputs, dict) else None
        if entry is None:
            return original(self, uid, rqueue, raw_inputs, args, cancelled,
                            log_state)
        job, stop = entry

        def should_stop() -> bool:
            if stop.is_set():
                return True
            cancelled.update(self._drain_cancellations())
            if uid in cancelled:
                cancelled.discard(uid)
                return True
            return False

        if stop.is_set():
            rqueue.put(_JobError(JobCancelled("cancelled before the job ran")))
            return None
        try:
            value = job(self, should_stop)
        except Cancelled as e:
            rqueue.put(_JobError(e if isinstance(e, JobCancelled)
                                 else JobCancelled(str(e))))
        except Exception as e:  # noqa: BLE001 - forwarded to the waiting caller
            rqueue.put(_JobError(e))
        else:
            rqueue.put(_JobResult(value))
        return None

    _generate_diffusion.__dict__[_INSTALLED_FLAG] = True
    _generate_diffusion.__wrapped__ = original  # type: ignore[attr-defined]
    cls._generate_diffusion = _generate_diffusion


class _StepJob:
    __slots__ = ("step", "stop", "rqueue")

    def __init__(self, step, stop: threading.Event, rqueue):
        self.step = step
        self.stop = stop
        self.rqueue = rqueue


def _is_step_job(item) -> bool:
    raw = getattr(item, "raw_inputs", None)
    return isinstance(raw, dict) and _STEP_JOB_KEY in raw


def _take_queued(rg) -> list:
    """Step jobs still in the request queue. The batch loop reads nothing
    from the queue while its batch is full, and a job must not wait for a
    free row."""
    q = rg.requests
    with q.mutex:
        found = [x for x in q.queue if _is_step_job(x)]
        for x in found:
            q.queue.remove(x)
    return found


@contextlib.contextmanager
def _step_scope(model):
    """One job step: the wired limit, the prefill command-buffer caps, and
    the rope position state of multimodal models put back afterwards."""
    from gmlx.serve import cb_phase

    holders = [model, getattr(model, "language_model", None)]
    saved = [(h, name, getattr(h, name)) for h in holders if h is not None
             for name in _ROPE_STATE if hasattr(h, name)]
    before = cb_phase.phase()
    cb_phase.flip("prefill")
    wired: Any = contextlib.nullcontext()
    if model is not None and mx.metal.is_available():
        wired = importlib.import_module("mlx_lm.generate").wired_limit(model)
    try:
        with wired:
            yield
    finally:
        for holder, name, value in saved:
            setattr(holder, name, value)
        if before is not None:
            cb_phase.flip(before)


def _start(rg, request) -> Optional[_StepJob]:
    job, stop = request.raw_inputs[_STEP_JOB_KEY]
    rqueue = request.rqueue
    rg._log_prefill_started(request, backend="decision")
    rqueue.put(_generation.GenerationContext(
        uid=f"job-{next(_job_ids)}", prompt_tokens=request.prompt_tokens))
    if stop.is_set():
        rqueue.put(_JobError(JobCancelled("cancelled before the job ran")))
        return None
    try:
        return _StepJob(job(rg, stop.is_set), stop, rqueue)
    except Exception as e:  # noqa: BLE001 - forwarded to the waiting caller
        rqueue.put(_JobError(e))
        return None


def _advance(rg, run: _StepJob) -> bool:
    """Run one step of ``run``. True when the job has finished."""
    try:
        if run.stop.is_set():
            run.step.close()
            raise JobCancelled("the decision was cancelled")
        with _step_scope(getattr(rg, "model", None)):
            next(run.step)
        return False
    except StopIteration as done:
        run.rqueue.put(_JobResult(done.value))
    except Cancelled as e:
        run.rqueue.put(_JobError(e if isinstance(e, JobCancelled)
                                 else JobCancelled(str(e))))
    except Exception as e:  # noqa: BLE001 - forwarded to the waiting caller
        run.rqueue.put(_JobError(e))
    return True


def _install_step_jobs() -> None:
    cls = _generation.ResponseGenerator
    original = cls._collect_pending_requests
    if getattr(original, _INSTALLED_FLAG, False):
        return
    from gmlx.gen.diffusion import is_diffusion_model

    def _collect_pending_requests(self, *, active: bool, capacity=None,
                                  idle_timeout: float = 0.1, coalesce_s: float = 0.0):
        diffusion = self.__dict__.get(_DIFFUSION_ATTR)
        if diffusion is None:
            diffusion = is_diffusion_model(getattr(self, "model", None))
            self.__dict__[_DIFFUSION_ATTR] = diffusion
        if diffusion:
            return original(self, active=active, capacity=capacity,
                            idle_timeout=idle_timeout, coalesce_s=coalesce_s)
        jobs: list = self.__dict__.setdefault(_JOBS_ATTR, [])
        arrived = _take_queued(self)
        if jobs:
            run = jobs.pop(0)
            if not _advance(self, run):
                jobs.append(run)
        pending, should_stop = original(
            self, active=active or bool(jobs) or bool(arrived),
            capacity=capacity, idle_timeout=idle_timeout, coalesce_s=coalesce_s)
        arrived += [x for x in pending if _is_step_job(x)]
        pending = [x for x in pending if not _is_step_job(x)]
        for request in arrived:
            run = _start(self, request)
            if run is not None:
                jobs.append(run)
        if self._stop and jobs:
            for run in jobs:
                run.step.close()
                run.rqueue.put(_JobError(RuntimeError("the model engine stopped")))
            jobs.clear()
        return pending, should_stop

    _collect_pending_requests.__dict__[_INSTALLED_FLAG] = True
    _collect_pending_requests.__wrapped__ = original  # type: ignore[attr-defined]
    cls._collect_pending_requests = _collect_pending_requests
