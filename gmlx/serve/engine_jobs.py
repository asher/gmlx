"""Run a callable on a diffusion model's engine thread.

mlx-vlm serves a diffusion model from one engine thread that takes queued
requests one at a time. A job rides that same queue: ``run_on_engine``
queues a request whose inputs carry the job, and the wrapped
``_generate_diffusion`` runs it in place of a generation. The job therefore
owns the model for its duration, and chat requests to the same model queue
behind it as they queue behind any diffusion generation.

The caller waits for the engine to dequeue the job with no deadline, as
chat waits for its generation context. The deadline starts at dequeue and
bounds the job itself. A missed deadline sets the job's stop event, which
the job polls, and raises ``JobTimeout``."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional

from mlx_vlm.server import generation as _generation

from gmlx.systemone.reads import Cancelled

_JOB_KEY = "_kq_engine_job"
_INSTALLED_FLAG = "_kq_engine_jobs"

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
                  stop: threading.Event, timeout_s: Optional[float]) -> Any:
    """Run ``job(response_generator, should_stop)`` on ``rg``'s engine
    thread and return its value. An exception from the job is raised here
    as itself. ``timeout_s`` of ``None`` waits for the job forever."""
    rg.wait_until_ready()
    rqueue: queue.Queue = queue.Queue()
    request = _generation.QueuedGenerationRequest(
        rqueue=rqueue,
        raw_inputs={_JOB_KEY: (job, stop)},
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
    place of a generation. Idempotent."""
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
