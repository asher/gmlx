"""The video jobs of POST /v1/videos: the job store and its driver.

A job is the OpenAI video object with a gmlx part, kept as
``jobs/<id>.json`` under the store's folder and written again, by a
rename, at each change of status. Its inputs are files under
``inputs/<id>/`` until it ends, and its clip is ``files/<id>.mp4``,
muxed as a ``.part`` and renamed once whole. A job lives for
``server.video.job_ttl_s`` from its creation, and once it ends for
``job_ttl_s`` from its end. Past ``store_max_bytes`` the oldest finished
clips go first.

The driver thread, ``video-jobs``, runs one job at a time in order and
does no MLX work: it reads the anchors' headers, has the upsampler
rewrite the prompt, acquires the video entry and hands the clip's work
to the video worker, waiting off the worker when memory is short for a
while. On the video worker the job encodes the prompt and the anchors,
takes its LoRA turn, samples and decodes, as the pack's steps do.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field

from .patches.jobs import JobBook
from .subservice import SingleWorker

_log = logging.getLogger(__name__)

QUEUED, RUNNING, DONE, FAILED, CANCELLED = (
    "queued", "in_progress", "completed", "failed", "cancelled")
ENDED = frozenset({DONE, FAILED, CANCELLED})
# Retry-After of a full queue with no clip timed yet.
FIRST_RETRY_S = 60
# The phase of the prompt encode, and the phases that can come before it.
ENCODING = "encoding the prompt"
_BEFORE_ENCODE = frozenset({"reading the inputs", "rewriting the prompt",
                            "loading the model", "waiting for memory"})
# The LoRA turn's subject in its refusals.
JOB = "the LoRA factors of this video job"
# A job's id, as :meth:`VideoJobs.new_id` makes it.
_JOB_ID = re.compile(r"video_[0-9a-f]{32}")


class JobFailed(Exception):
    """A job's failure: the HTTP status the request would answer, its
    error type and code, and, for a 503 that ends by itself, the seconds
    to wait before the work is tried again."""

    def __init__(self, status: int, message: str, *, code: str | None = None,
                 err_type: str | None = None, retry_after: int | None = None,
                 transient: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type or ("server_error" if status >= 500
                                     else "invalid_request_error")
        self.code = code
        self.retry_after = retry_after
        self.transient = transient


class QueueFull(Exception):
    def __init__(self, retry_after: int):
        super().__init__("queue full")
        self.retry_after = retry_after


@dataclass(eq=False)
class Job:
    id: str
    record: dict
    params: dict
    stop: threading.Event = field(default_factory=threading.Event)
    started: float | None = None
    # The encode's results, kept across a wait for memory.
    kept: dict = field(default_factory=dict)
    # Whether the prompt encode began: progress counts 5 from it.
    encoding: bool = False


def default_root(port=None) -> str:
    from .image_service import default_store_root

    return default_store_root(port, kind="videos")


def failure(e: BaseException) -> JobFailed:
    """The JobFailed of an error a job's work raised."""
    from fastapi import HTTPException

    from .image_service import ImageRequestError
    from .image_steps import StepError
    from .residency import LoraRefused

    if isinstance(e, JobFailed):
        return e
    if isinstance(e, StepError):
        return JobFailed(e.status, str(e), retry_after=e.retry_after,
                         transient=getattr(e, "transient", False))
    if isinstance(e, LoraRefused):
        return JobFailed(e.status, e.message, retry_after=e.retry_after,
                         transient=e.transient)
    if isinstance(e, ImageRequestError):
        retry = e.headers.get("Retry-After")
        return JobFailed(e.status, str(e), code=e.code, err_type=e.err_type,
                         retry_after=int(retry) if retry else None,
                         transient=e.status == 503 and retry is not None)
    if isinstance(e, HTTPException):
        detail = e.detail
        err = detail.get("error", {}) if isinstance(detail, dict) else {}
        message = err.get("message") if isinstance(err, dict) else None
        retry = (e.headers or {}).get("Retry-After")
        return JobFailed(e.status_code, str(message or detail),
                         code=err.get("code") if isinstance(err, dict)
                         else None,
                         err_type=err.get("type") if isinstance(err, dict)
                         else None,
                         retry_after=int(retry) if retry else None,
                         transient=e.status_code == 503 and retry is not None)
    from gmlx.image.memfit import ImageMemoryError
    from gmlx.video.fingerprint import VideoModelError
    from gmlx.video.io import FfmpegError

    if isinstance(e, (ValueError, FfmpegError, ImageMemoryError,
                      VideoModelError)):
        return JobFailed(400, str(e))
    _log.error("video job failed", exc_info=e)
    return JobFailed(500, f"the video job failed ({type(e).__name__}); see "
                     "the server log", code=type(e).__name__)


class VideoJobs:
    """The job store, its queue and its driver."""

    def __init__(self, root: str | None = None, *, time_fn=None,
                 mono_fn=None, run=None, driver=None):
        from gmlx.config import VideoServerCfg

        self.root = root
        self.settings = VideoServerCfg()
        self._time = time_fn or time.time
        self._mono = mono_fn or time.monotonic
        self._run = run
        self._driver = driver
        self.lock = threading.Lock()
        self.jobs: dict = {}
        # The bytes reserved by each POST whose inputs are still arriving,
        # by job id: the sweep leaves its inputs, and the room counts it.
        self.pending: dict = {}
        self.book = JobBook("completed", "failed", "cancelled")
        self._restored = False

    # Paths.

    def _dir(self, kind: str) -> str:
        assert self.root is not None
        path = os.path.join(self.root, kind)
        os.makedirs(path, mode=0o700, exist_ok=True)
        return path

    def inputs(self, jid: str) -> str:
        """The folder of job ``jid``'s inputs, made by no one here: a POST
        makes inputs/ at :meth:`begin`, past its refusals."""
        assert self.root is not None
        return os.path.join(self.root, "inputs", jid)

    def clip(self, jid: str) -> str:
        return os.path.join(self._dir("files"), f"{jid}.mp4")

    def _record_path(self, jid: str) -> str:
        return os.path.join(self._dir("jobs"), f"{jid}.json")

    def _write(self, job: Job) -> None:
        """Rename the job's record into place."""
        path = self._record_path(job.id)
        part = f"{path}.part"
        with open(part, "w", encoding="utf-8") as f:
            json.dump(job.record, f)
        os.replace(part, path)

    def _drop_inputs(self, jid: str) -> None:
        shutil.rmtree(self.inputs(jid), ignore_errors=True)

    def _forget(self, jid: str) -> None:
        """Remove a job's record, inputs and clip; under ``lock``."""
        self.jobs.pop(jid, None)
        self._drop_inputs(jid)
        for path in (self._record_path(jid), self.clip(jid),
                     f"{self.clip(jid)}.part"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    # Install, sweep and room.

    def configure(self, cfg) -> None:
        """Take ``cfg``'s server.video settings. The first call takes the
        records the store holds: a job a past server queued or ran fails
        as server_restarted, since no job resumes. Each call sweeps."""
        from gmlx.config import VideoServerCfg

        self.settings = getattr(cfg, "video", None) or VideoServerCfg()
        if self.root is None:
            self.root = default_root(getattr(cfg, "port", None))
        if not os.path.isdir(self.root):
            # A store no job has used holds nothing to take or sweep.
            self._restored = True
            return
        if not self._restored:
            self._restore()
            self._restored = True
        self.sweep()

    def _restore(self) -> None:
        with self.lock:
            folder = self._dir("jobs")
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                jid = name[:-len(".json")]
                record = None
                if name.endswith(".json") and _JOB_ID.fullmatch(jid):
                    try:
                        with open(path, encoding="utf-8") as f:
                            record = json.load(f)
                    except (OSError, ValueError):
                        pass
                # A job's files are named by its id: a record that is not
                # its own file's, such as a copy, is no job.
                if not isinstance(record, dict) or record.get("id") != jid:
                    with contextlib.suppress(OSError):
                        os.unlink(path)
                    continue
                job = Job(jid, record, {})
                self.jobs[jid] = job
                if record.get("status") not in ENDED:
                    self._end(job, FAILED, error={
                        "code": "server_restarted",
                        "message": "the server restarted before the job "
                        "finished; submit it again",
                        "type": "server_error", "status": 500})
                    # A clip renamed in before its record said completed
                    # is no job's: it would count to the job's expiry.
                    for clip in (self.clip(jid), f"{self.clip(jid)}.part"):
                        with contextlib.suppress(OSError):
                            os.unlink(clip)
            for name in os.listdir(self._dir("files")):
                if name.endswith(".part"):
                    with contextlib.suppress(OSError):
                        os.unlink(os.path.join(self._dir("files"), name))
            # The links of answers a past server was sending.
            shutil.rmtree(os.path.join(self.root, "serving"),
                          ignore_errors=True)

    def _bytes(self, path: str) -> int:
        try:
            if os.path.isdir(path):
                return sum(self._bytes(os.path.join(path, n))
                           for n in os.listdir(path))
            return os.stat(path).st_size
        except OSError:
            return 0

    def sweep(self, keep: str | None = None) -> int:
        """Remove the expired jobs, then the oldest finished clips past
        store_max_bytes with their jobs, never the clip ``keep``, a live
        job's inputs or a clip being written. Files no job names go too.
        Returns how many jobs went."""
        with self.lock:
            now = self._time()
            gone = 0
            for jid, job in list(self.jobs.items()):
                if job.record["status"] in ENDED and \
                        now >= job.record["expires_at"]:
                    self._forget(jid)
                    gone += 1
            for kind in ("inputs", "files"):
                for name in os.listdir(self._dir(kind)):
                    jid = name.split(".")[0]
                    if jid in self.pending:
                        continue
                    live = self.jobs.get(jid)
                    if live is None or (kind == "inputs"
                                        and live.record["status"] in ENDED):
                        path = os.path.join(self._dir(kind), name)
                        if kind == "inputs":
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            with contextlib.suppress(OSError):
                                os.unlink(path)
            total = self._held()
            done = sorted((j.record.get("completed_at") or 0, jid)
                          for jid, j in self.jobs.items()
                          if j.record["status"] == DONE)
            for _at, jid in done:
                if total <= self.settings.store_max_bytes:
                    break
                if self.clip(jid) == keep or self._sending(self.clip(jid)):
                    continue
                total -= self._bytes(self.clip(jid))
                self._forget(jid)
                gone += 1
            return gone

    def _held(self) -> int:
        """The bytes on disk and reserved: the clips, the inputs of the
        jobs, the clips a DELETE or the sweep removed that an answer still
        sends through its link, and the reservation of each POST still
        arriving in place of what it wrote so far. Under ``lock``."""
        inputs = self._dir("inputs")
        return (self._bytes(self._dir("files"))
                + sum(self._bytes(os.path.join(inputs, n))
                      for n in os.listdir(inputs) if n not in self.pending)
                + self._sent_bytes()
                + sum(self.pending.values()))

    @staticmethod
    def _sending(clip: str) -> bool:
        """Whether an answer still sends ``clip`` through a link: removing
        it then frees nothing until the answer ends."""
        try:
            return os.stat(clip).st_nlink > 1
        except OSError:
            return False

    def _sent_bytes(self) -> int:
        """The bytes of the clips that only links in serving/ hold: a clip
        still in files/ counts with files/. Each clip counts once, however
        many answers send it."""
        serving = os.path.join(self.root, "serving")
        links: dict[int, list] = {}
        with contextlib.suppress(FileNotFoundError):
            with os.scandir(serving) as it:
                for entry in it:
                    with contextlib.suppress(OSError):
                        st = entry.stat()
                        links.setdefault(st.st_ino, []).append(st)
        return sum(sts[0].st_size for sts in links.values()
                   if sts[0].st_nlink == len(sts))

    def room(self) -> int:
        """The bytes left in the store for a new job's inputs, after a
        sweep."""
        self.sweep()
        with self.lock:
            return max(0, self.settings.store_max_bytes - self._held())

    # A POST's reservation, from its first byte to its submit.

    def begin(self, jid: str) -> None:
        """Reserve the job id of a POST before its first byte: the sweep
        leaves its inputs until :meth:`submit` or :meth:`abandon`."""
        with self.lock:
            self._dir("inputs")
            self.pending[jid] = 0

    def take(self, jid: str, n: int | None) -> tuple[bool, int, int]:
        """Reserve ``n`` more bytes of the room for the POST ``jid``, or
        with ``n`` None the next step of a body with no length: BODY_STEP
        bytes, or the room left when less. Answers ``(True, bytes
        reserved, 0)``, or ``(False, room left, bytes the other POSTs
        hold)`` when they do not fit, reserving nothing. A POST that is
        not pending, abandoned or submitted, takes nothing."""
        from .image_steps import BODY_STEP

        self.sweep()
        with self.lock:
            if jid not in self.pending:
                return False, 0, 0
            left = max(0, self.settings.store_max_bytes - self._held())
            want = n if n is not None else min(BODY_STEP, left)
            if want > left or (n is None and not want):
                arriving = sum(b for j, b in self.pending.items() if j != jid)
                return False, left, arriving
            self.pending[jid] += want
            return True, want, 0

    def settle(self, jid: str) -> None:
        """Hold the POST ``jid`` to the bytes it wrote, once its form is
        read, so the rest of its reservation is room again."""
        with self.lock:
            if jid in self.pending:
                self.pending[jid] = self._bytes(
                    os.path.join(self._dir("inputs"), jid))

    def abandon(self, jid: str) -> None:
        """End a refused or failed POST: its inputs go with its
        reservation. A job submit registered keeps its inputs."""
        with self.lock:
            if jid in self.jobs:
                return
            self.pending.pop(jid, None)
            self._drop_inputs(jid)

    # Queue.

    def queued(self) -> int:
        return sum(1 for j in self.jobs.values()
                   if j.record["status"] == QUEUED)

    def retry_after(self) -> int:
        """Seconds until the queue has a place, as a 429 tells it: the
        median clip time of the recent jobs less the running job's time,
        at least 2."""
        with self.lock:
            return self._retry_after()

    def _retry_after(self) -> int:
        typical = self.book.typical()
        if typical is None:
            return FIRST_RETRY_S
        running = [j.started for j in self.jobs.values()
                   if j.record["status"] == RUNNING and j.started]
        elapsed = self._mono() - running[0] if running else 0.0
        return int(max(2, round(typical - elapsed)))

    def full(self) -> bool:
        with self.lock:
            return self.queued() >= self.settings.max_queue

    def new_id(self) -> str:
        return f"video_{secrets.token_hex(16)}"

    def submit(self, jid: str, record: dict, params: dict) -> Job:
        """Queue job ``jid`` and give it to the driver, or raise
        QueueFull. Its inputs are already under ``inputs(jid)``."""
        now = int(self._time())
        record = {"id": jid, "object": "video", "status": QUEUED,
                  "progress": 0, "created_at": now, "completed_at": None,
                  "expires_at": now + int(self.settings.job_ttl_s),
                  "error": None, **record}
        job = Job(jid, record, params)
        with self.lock:
            if self.queued() >= self.settings.max_queue:
                self.pending.pop(jid, None)
                self._drop_inputs(jid)
                raise QueueFull(self._retry_after())
            # The record goes to disk before the job is queued: a write
            # that fails queues nothing, and a restart finds every job.
            self._write(job)
            self.pending.pop(jid, None)
            self.jobs[jid] = job
        self._driver_submit(job)
        return job

    def _driver_submit(self, job: Job) -> None:
        driver = self._driver or DRIVER
        driver.submit(self.drive, job)

    # Reads.

    def get(self, jid: str) -> dict | None:
        """The record of job ``jid``, None when it is unknown or expired."""
        with self.lock:
            return self._record(jid)

    def _record(self, jid: str) -> dict | None:
        job = self.jobs.get(jid)
        if job is None or (job.record["status"] in ENDED and
                           self._time() >= job.record["expires_at"]):
            return None
        return json.loads(json.dumps(job.record))

    def content(self, jid: str) -> str | None:
        """The clip of a completed job, None otherwise."""
        rec = self.get(jid)
        if rec is None or rec["status"] != DONE:
            return None
        path = self.clip(jid)
        return path if os.path.isfile(path) else None

    def serve(self, jid: str) -> tuple[dict | None, str | None, bool]:
        """``(record, path, linked)`` of job ``jid`` in one hold: its
        record, None when unknown or expired, and for a completed job a
        link to its clip made for one answer (``linked``), which the
        answer removes once sent. A DELETE or a sweep meanwhile removes
        the clip, never the link. A store on a volume with no hard links
        answers with the clip itself."""
        with self.lock:
            rec = self._record(jid)
            if rec is None or rec["status"] != DONE:
                return rec, None, False
            clip = self.clip(jid)
            link = os.path.join(self._dir("serving"),
                                f"{secrets.token_hex(16)}.mp4")
            try:
                os.link(clip, link)
            except FileNotFoundError:
                return rec, None, False
            except OSError:
                return rec, clip, False
            return rec, link, True

    def delete(self, jid: str) -> dict | None:
        """Cancel a queued or running job, or remove a finished one with
        its clip. Returns the job's status after, None when unknown."""
        with self.lock:
            job = self.jobs.get(jid)
            if job is None or (job.record["status"] in ENDED and
                               self._time() >= job.record["expires_at"]):
                return None
            status = job.record["status"]
            if status in ENDED:
                self._forget(jid)
                return {"id": jid, "object": "video.deleted", "deleted": True,
                        "status": status}
            job.stop.set()
            self._end(job, CANCELLED)
            if status == QUEUED:
                self._drop_inputs(jid)
            return {"id": jid, "object": "video.deleted", "deleted": True,
                    "status": CANCELLED}

    # Transitions; under ``lock``.

    def _end(self, job: Job, status: str, error: dict | None = None) -> None:
        """End ``job`` with ``status``. Its record and clip last
        job_ttl_s from now. A job reads as completed once its record
        says so on disk: when that write fails, it raises and the job
        stays running. A failure or a cancel holds when its write fails,
        as a restart fails the job all the same."""
        old = job.record
        rec = copy.deepcopy(old)
        rec["status"] = status
        rec["completed_at"] = int(self._time())
        rec["expires_at"] = rec["completed_at"] + int(self.settings.job_ttl_s)
        rec["error"] = error
        rec.get("gmlx", {}).pop("eta_s", None)
        if status == DONE:
            rec["progress"] = 100
        job.record = rec
        try:
            self._write(job)
        except OSError:
            if status == DONE:
                job.record = old
                raise
            _log.warning("video job %s: its %s record was not written",
                         job.id, status, exc_info=True)
        self.book.counts[status] = self.book.counts.get(status, 0) + 1

    def progress(self, job: Job, *, phase: str | None = None,
                 step: tuple | None = None, frames: tuple | None = None
                 ) -> None:
        """Set the running job's phase, its step i of N and its frames k
        of F decoded: progress is 0 before the prompt encode, then
        5 + 85 i/N + 10 k/F."""
        with self.lock:
            rec = job.record
            if rec["status"] != RUNNING:
                return
            g = rec.setdefault("gmlx", {})
            if phase is not None:
                g["phase"] = phase
            job.encoding = job.encoding or step is not None \
                or frames is not None \
                or (phase is not None and phase not in _BEFORE_ENCODE)
            if not job.encoding:
                return
            if step is not None:
                g["step"], g["steps"] = step
            if frames is not None:
                g["frames_decoded"] = frames[0]
            i, n = g.get("step", 0), g.get("steps") or 1
            k, f = g.get("frames_decoded", 0), g.get("frames") or 1
            done = 5 + 85 * i / n + 10 * k / f
            rec["progress"] = int(min(99, done))
            if job.started is not None and done > 5:
                el = self._mono() - job.started
                g["eta_s"] = round(el * (100 - done) / done)

    def note(self, job: Job, **fields) -> None:
        with self.lock:
            job.record.setdefault("gmlx", {}).update(fields)

    def wait(self, job: Job, e: JobFailed) -> None:
        """Wait out a 503 that ends by itself, off the worker: the job
        stays in_progress, phase "waiting for memory". A wait that would
        pass the job's expiry fails it."""
        from .image_service import JobStopped

        retry = max(1, int(e.retry_after or 5))
        if self._time() + retry >= job.record["expires_at"]:
            raise JobFailed(e.status, f"the job waited for memory until it "
                            f"expired: {e.message}", code="timed_out")
        self.progress(job, phase="waiting for memory")
        self.note(job, waiting=e.message)
        if job.stop.wait(retry):
            raise JobStopped(timed_out=False)
        self.note(job, waiting=None)

    # The driver.

    def drive(self, job: Job) -> None:
        """Run ``job`` on the driver thread, unless DELETE ended it."""
        from .image_service import JobStopped

        with self.lock:
            if job.record["status"] != QUEUED:
                return
            if self._time() >= job.record["expires_at"]:
                # A job waits in the queue at most job_ttl_s from its
                # creation: it never runs, and its timed_out record lasts
                # job_ttl_s from now.
                self._end(job, FAILED, error={
                    "code": "timed_out",
                    "message": "the job was still queued when it expired; "
                    "submit it again, or raise server.video.job_ttl_s",
                    "type": "server_error", "status": 503})
                self._drop_inputs(job.id)
                job.kept.clear()
                return
            job.record["status"] = RUNNING
            job.record["progress"] = 0
            job.record.setdefault("gmlx", {})["phase"] = "starting"
            job.started = self._mono()
            self._write(job)
        keep = None
        try:
            (self._run or run_clip)(job, self)
            with self.lock:
                if job.record["status"] == RUNNING:
                    took = self._mono() - job.started
                    keep = self.clip(job.id)
                    self._end(job, DONE)
                    # A clip counts once its completed record is written.
                    self.book.timed(took)
        except JobStopped:
            pass
        except BaseException as e:  # noqa: BLE001 - the job's failure
            f = failure(e)
            with self.lock:
                if job.record["status"] == RUNNING:
                    self._end(job, FAILED, error={
                        "code": f.code or f.err_type, "message": f.message,
                        "type": f.err_type, "status": f.status})
        finally:
            # The encode's features and the anchors' latents: no ended job
            # needs them, and the job lives to its expiry.
            job.kept.clear()
            with self.lock:
                self._drop_inputs(job.id)
                if job.record["status"] != DONE:
                    for path in (self.clip(job.id),
                                 f"{self.clip(job.id)}.part"):
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(path)
            self.sweep(keep=keep)

    def metrics(self) -> dict:
        """The ``videos`` section of ``/v1/metrics``."""
        typical = self.book.typical()
        with self.lock:
            counts = dict(self.book.counts)
            running = sum(1 for j in self.jobs.values()
                          if j.record["status"] == RUNNING)
            return {"jobs_completed_total": counts.get(DONE, 0),
                    "jobs_failed_total": counts.get(FAILED, 0),
                    "jobs_cancelled_total": counts.get(CANCELLED, 0),
                    "queued": self.queued(), "running": running,
                    "max_queue": self.settings.max_queue,
                    "clip_s": round(typical, 1) if typical else None}


DRIVER = SingleWorker("video-jobs")
STORE = VideoJobs()


# The clip's work.

def run_clip(job: Job, store: VideoJobs) -> None:
    """The work of ``job``: anchors read, the prompt rewritten, the entry
    acquired, then the encodes and the clip on the video worker. A 503
    that ends by itself waits and tries the same part again."""
    from gmlx.video import run

    from . import bridge_vlm as serving
    from . import video_service as vs
    from .image_service import JobStopped
    from .image_upsample import revise
    from .patches.media_gate import MEDIA_MAX_PIXELS

    p = job.params
    _path, spec = serving.resolve_video_request(p["model"])
    from gmlx.video.loader import identify_dit

    fam = identify_dit(spec.path, variant=spec.variant).family
    s = p["settings"]
    store.progress(job, phase="reading the inputs")
    keyframes, todo = run.read_anchors(
        fam, s, first=p.get("first_frame"), last=p.get("last_frame"),
        guides=p.get("guides") or (),
        guide_no_audio=bool(p.get("guide_no_audio")), ffmpeg=p["ffmpeg"],
        names=("first_frame", "last_frame"), paths=False,
        max_pixels=MEDIA_MAX_PIXELS)
    caption = None
    store.progress(job, phase="rewriting the prompt")
    while True:
        try:
            caption = revise(p["model"], p["prompt"], p.get("upsampling"),
                             False, stop=job.stop,
                             resolve=serving.resolve_video_request,
                             frames=s["frames"],
                             first=bool(p.get("first_frame")),
                             last=bool(p.get("last_frame")))
            break
        except JobStopped:
            raise
        except Exception as e:  # noqa: BLE001 - mapped to the job's error
            f = failure(e)
            if not f.transient:
                raise f from None
            store.wait(job, f)
    if caption is None and p.get("rewrites"):
        raise JobFailed(500, "the job's prompt was to be rewritten, and the "
                        "upsampler gave no caption; see the server log",
                        code="upsample_failed")
    if caption is not None:
        store.note(job, revised_prompt=caption)
    text = caption or p["prompt"]
    out = store.clip(job.id)
    parts = ("encode", "clip")
    at = 0
    while at < len(parts):
        if job.stop.is_set():
            raise JobStopped(timed_out=False)
        store.progress(job, phase="loading the model")
        try:
            from .residency import acquire_video_model

            hold = acquire_video_model(p["model"])
        except Exception as e:  # noqa: BLE001
            f = failure(e)
            if not f.transient:
                raise f from None
            store.wait(job, f)
            continue
        try:
            if parts[at] == "encode":
                vs.WORKER.run(_encode_part, job, store, hold, text,
                              keyframes, todo)
            else:
                vs.WORKER.run(_clip_part, job, store, hold, out)
            at += 1
        except JobStopped:
            raise
        except Exception as e:  # noqa: BLE001
            f = failure(e)
            if not f.transient:
                raise f from None
            hold.release()
            hold = None
            store.wait(job, f)
        finally:
            if hold is not None:
                hold.release()


def _settle(hold, subject: str) -> None:
    from .patches.image_steps import _lora_pool

    pool = _lora_pool()
    if pool is not None:
        from .image_service import lora_settle

        lora_settle(pool, hold._entry, subject)


def _encode_part(job: Job, store: VideoJobs, hold, text: str, keyframes,
                 todo) -> None:
    """On the video worker: the prompt's encode and the anchors'."""
    from .image_steps import ENCODE, StepError
    from .patches.image_steps import _check_encode
    from . import video_service as vs

    served = hold._entry.pipeline
    why = vs.keyframe_refusal(served, keyframes) or vs.anchor_refusal(
        served, len(todo))
    if why:
        raise StepError(400, why)
    sizes = [(int(i.shape[1]), int(i.shape[0])) for i in keyframes]
    store.progress(job, phase=ENCODING)
    try:
        _check_encode(hold, served, text, sizes)
        feats, spans = vs.encode(served, text, keyframes)
    finally:
        _settle(hold, ENCODE)
    job.kept.update(feats=feats, spans=spans, rows=(), anchors=())
    if todo:
        store.progress(job, phase="encoding the anchors")
        job.kept.update(encode_anchors(job, hold, served, todo))


def encode_anchors(job: Job, hold, served, todo) -> dict:
    """The anchors' latents on the video worker: stills and sound priced
    as the anchors phase of the clip, a guide video read through ffmpeg a
    window of whole pieces at a time, each window priced before its
    read. Windows of whole pieces are the whole encode."""
    import dataclasses

    import mlx.core as mx
    import numpy as np

    from gmlx.video import anchors, io
    from gmlx.video.images import cover
    from gmlx.video.vae.reader import load_vae

    from .image_service import JobStopped
    from .image_steps import StepError
    from .patches.body_intake import _mib
    from .patches.image_steps import _check_clip, _window_fit

    s = job.params["settings"]
    w, h, frames = s["width"], s["height"], s["frames"]
    small = tuple(t.rows for t in todo if t.video is None)
    if small:
        _check_clip(hold, served, frames=frames, width=w, height=h, text=1,
                    phase="anchors", what="the anchors' encode",
                    anchors=small)
    # Each guide video's windows, priced before the VAE loads: the whole
    # clip in one window when it fits, else open windows of the most
    # whole pieces that fit.
    windows: dict = {}
    for i, t in enumerate(todo):
        n = t.placed.video_frames
        if not n or t.still is not None:
            continue
        fit = _window_fit(hold, served, n=n, width=w, height=h, level=1,
                          is_open=False)
        windows[i] = None
        if fit is not None and fit[0] > fit[1]:
            need, free, k = fit
            if n == 1 or not k:
                raise StepError(
                    400, f"the encode of {t.label}, a guide clip of {n} "
                    f"frames of {w}x{h}, does not fit in memory beside the "
                    "models now loaded: no window fits; the clip needs "
                    f"{_mib(need, True)}, and {_mib(max(free, 0))} is free")
            windows[i] = 17 * k
    video, audio = {}, {}
    if any(t.placed.video_frames for t in todo):
        vae = load_vae(os.path.expanduser(served.spec.vae), family="h3",
                       dtype=mx.float16)
        try:
            for i, t in enumerate(todo):
                n = t.placed.video_frames
                if not n:
                    continue
                if t.still is not None:
                    video[i] = np.array(anchors.encode(
                        dataclasses.replace(t.placed, audio_t=None),
                        frames=t.still[None], video_vae=vae).video)
                    continue
                window = windows[i]
                got: list = []
                buf: list = []

                def flush(last: bool, vae=vae, buf=buf, got=got):
                    if job.stop.is_set():
                        raise JobStopped(timed_out=False)
                    x = mx.array(np.stack(buf))
                    buf.clear()
                    z = vae.encode((x.astype(mx.float32) / 255.0)[None]
                                   * 2.0 - 1.0, drop=last)
                    mx.eval(z)
                    got.append(np.array(z))
                    del x, z
                    mx.clear_cache()

                read = [0]

                def each(f, n=n, window=window, buf=buf, flush=flush,
                         read=read):
                    buf.append(cover(f, w, h))
                    read[0] += 1
                    # An open window, while the clip's end is to come.
                    if window and len(buf) == window and n - read[0] >= 5:
                        flush(False)

                io.each_guide_frame(t.video, most=n, ffmpeg=t.ffmpeg,
                                    each=each)
                if buf:
                    flush(True)
                video[i] = np.concatenate(got, axis=2)
        finally:
            del vae
            mx.clear_cache()
    if any(t.wave is not None for t in todo):
        avae = load_vae(os.path.expanduser(served.spec.audio_vae),
                        family="h3_audio", dtype=mx.float32)
        try:
            for i, t in enumerate(todo):
                if t.wave is not None:
                    audio[i] = np.array(anchors.encode(
                        dataclasses.replace(t.placed, video_frames=0),
                        waveform=t.wave, audio_vae=avae).audio)
        finally:
            del avae
            mx.clear_cache()
    return {"rows": tuple(t.rows for t in todo),
            "anchors": tuple((t.placed.frame, video.get(i), audio.get(i))
                             for i, t in enumerate(todo))}


def job_pacer(job: Job):
    """The job's checks at each pause point: it stops the job once its stop
    is set and gives chat its share of the GPU. The sampler runs them on
    the row budget (``api.Budget``) and the decode between its pieces."""
    from .gpu_share import pacer
    from .image_service import JobStopped

    chat = pacer()

    def fn(arrays):
        if job.stop.is_set():
            raise JobStopped(timed_out=False)
        if chat is not None:
            chat(arrays)

    return fn


def _clip_part(job: Job, store: VideoJobs, hold, out: str) -> None:
    """On the video worker: the clip's fit with its decodes, the LoRA
    turn, the denoise and the decode into ``out``."""
    import mlx.core as mx

    from gmlx.image.pacing import paced
    from gmlx.video import anchors as anc
    from gmlx.video import api, pipeline, run
    from gmlx.video.vae.h3_audio import SAMPLE_RATE
    from gmlx.video.vae.reader import load_vae

    from . import video_service as vs
    from .patches.image_steps import _check_clip, _factor_turn, _lora_pool

    served = hold._entry.pipeline
    s = job.params["settings"]
    k = job.kept
    # The job's int8 setting, else the entry's; a job its POST recorded
    # before the field has none.
    int8 = job.params.get("int8")
    feats = mx.array(k["feats"])
    _check_clip(hold, served, frames=s["frames"], width=s["width"],
                height=s["height"], text=int(feats.shape[1]),
                phase="denoise", decodes=True, what="a clip",
                anchors=k["rows"], int8=int8)
    pool = _lora_pool()
    xv = xa = None
    try:
        store.progress(job, phase="loading the DiT")
        vs.ensure_dit(served)
        if pool is not None:
            _factor_turn(pool, hold._entry, served, None, subject=JOB)
        placed = tuple(anc.Anchor(f, None if v is None else mx.array(v),
                                  None if a is None else mx.array(a))
                       for f, v, a in k["anchors"])
        pace = job_pacer(job)
        vs.set_int8(served, int8)
        store.progress(job, phase="denoising", step=(0, s["steps"]))

        def on_step(i, n):
            store.progress(job, step=(i + 1, n))

        with paced(pace):
            xv, xa = pipeline.generate(
                served.pipe, feats, frames=s["frames"], width=s["width"],
                height=s["height"], steps=s["steps"], shifts=s["shifts"],
                seed=s["seed"], table=served.pipe.default_table,
                anchors=placed, spans=k["spans"], on_step=on_step,
                pacer=api.Budget(pace))
            mx.eval(xv, xa)
        del feats, placed
        store.progress(job, phase="decoding")

        def between(n):
            # The decode's pause point: the stop, and chat's share.
            pace(())
            store.progress(job, frames=(n, s["frames"]))

        from gmlx.video.families.h3 import family as hf

        run.decode_and_mux(
            xv, xa, frames=s["frames"], width=s["width"],
            height=s["height"], fps=hf.FPS, rate=SAMPLE_RATE, out=out,
            ffmpeg=job.params["ffmpeg"], between=between,
            audio_vae=lambda: load_vae(
                os.path.expanduser(served.spec.audio_vae),
                family="h3_audio", dtype=mx.float32),
            video_vae=lambda: load_vae(
                os.path.expanduser(served.spec.vae), family="h3",
                dtype=mx.float16))
    finally:
        if pool is not None:
            from .image_service import lora_settle

            lora_settle(pool, hold._entry, JOB)
        del xv, xa
        mx.clear_cache()
