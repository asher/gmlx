"""gmlx/serve/video_jobs.py and patches/videos.py: the video job store,
its driver and the /v1/videos routes, on a fake clock with stub work, so
no model loads."""
from __future__ import annotations

import base64
import io
import json
import math
import os
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.config import VideoServerCfg  # noqa: E402
from gmlx.serve import image_service, video_jobs as vj  # noqa: E402
from gmlx.serve.patches import body_intake, videos  # noqa: E402
from video_fixtures import ROW_REGIME_IDS, ROW_REGIMES  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class Driver:
    """Jobs handed to the driver, run when the test says."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run(self):
        while self.jobs:
            fn, args = self.jobs.pop(0)
            fn(*args)


def _store(tmp_path, run=None, **cfg):
    clock = Clock()
    store = vj.VideoJobs(str(tmp_path / "videos"), time_fn=clock,
                         mono_fn=clock, run=run, driver=Driver())
    store.configure(SimpleNamespace(video=VideoServerCfg(**cfg)))
    assert store.root is not None
    os.makedirs(store.root, exist_ok=True)
    return store, clock, store._driver


def _clip(job, store, data=b"mp4"):
    with open(store.clip(job.id), "wb") as f:
        f.write(data)


def _submit(store, **gmlx):
    jid = store.new_id()
    os.makedirs(store.inputs(jid), exist_ok=True)
    with open(os.path.join(store.inputs(jid), "first_frame"), "wb") as f:
        f.write(b"png")
    return store.submit(jid, {"model": "h3", "gmlx": {"frames": 22,
                                                     **gmlx}}, {})


def _on_disk(store, jid):
    with open(store._record_path(jid)) as f:
        return json.load(f)


def test_a_job_runs_from_queued_to_completed_with_its_record_on_disk(
        tmp_path):
    seen = []

    def run(job, store):
        seen.append(store.get(job.id)["status"])
        clock.now += 500
        _clip(job, store)

    store, clock, driver = _store(tmp_path, run)
    job = _submit(store)
    rec = store.get(job.id)
    assert (rec["status"], rec["object"], rec["progress"]) == (
        "queued", "video", 0)
    assert rec["expires_at"] == rec["created_at"] + 86400
    expires = rec["expires_at"]
    assert _on_disk(store, job.id)["status"] == "queued"
    driver.run()
    assert seen == ["in_progress"]
    rec = store.get(job.id)
    assert (rec["status"], rec["progress"], rec["error"]) == (
        "completed", 100, None)
    # Its record and clip last job_ttl_s from its end.
    assert rec["completed_at"] == rec["created_at"] + 500
    assert rec["expires_at"] == rec["completed_at"] + 86400 == expires + 500
    assert _on_disk(store, job.id)["status"] == "completed"
    assert store.content(job.id) == store.clip(job.id)
    # Its inputs go when it ends.
    assert not os.path.exists(store.inputs(job.id))


@pytest.mark.parametrize("raised, want", [
    (vj.JobFailed(400, "a clip does not fit"), (400, None, "a clip does not "
                                                "fit")),
    (KeyError("x"), (500, "KeyError", "the video job failed (KeyError); "
                     "see the server log")),
])
def test_a_job_that_raises_fails_with_its_status_and_no_clip(
        tmp_path, raised, want):
    def run(job, store):
        with open(store.clip(job.id) + ".part", "wb") as f:
            f.write(b"x")
        raise raised

    store, clock, driver = _store(tmp_path, run)
    job = _submit(store)
    driver.run()
    rec = store.get(job.id)
    err = rec["error"]
    assert rec["status"] == "failed"
    assert (err["status"], err["code"] if want[1] else None,
            err["message"]) == want
    assert store.content(job.id) is None
    assert os.listdir(os.path.join(str(store.root), "files")) == []
    assert store.delete(job.id) == {"id": job.id, "object": "video.deleted",
                                    "deleted": True, "status": "failed"}
    assert store.get(job.id) is None


def test_a_clip_whose_completed_record_cannot_be_written_fails(
        tmp_path, monkeypatch):
    """The disk fills as the clip lands: its completed record is never
    written, so the job does not read as completed, which a restart
    would contradict. It fails, and its clip goes."""
    full = []

    def run(job, store):
        _clip(job, store)
        clock.now += 60
        full.append(True)

    store, clock, driver = _store(tmp_path, run)
    job = _submit(store)
    write = store._write

    def write_unless_full(j):
        if full:
            raise OSError(28, "No space left on device")
        write(j)
    monkeypatch.setattr(store, "_write", write_unless_full)
    driver.run()
    rec = store.get(job.id)
    assert (rec["status"], rec["error"]["status"]) == ("failed", 500)
    assert store.content(job.id) is None
    assert os.listdir(os.path.join(str(store.root), "files")) == []
    assert _on_disk(store, job.id)["status"] == "in_progress"
    assert store.metrics()["jobs_completed_total"] == 0
    assert store.metrics()["clip_s"] is None


def test_a_clip_deleted_as_it_goes_out_counts_in_the_store_until_sent(
        tmp_path):
    """A DELETE removes a clip an answer still sends through its link:
    its bytes stay on disk, so the store counts them until the answer
    ends, and a new job cannot carry the store past its limit."""
    store, clock, driver = _store(
        tmp_path, lambda j, s: _clip(j, s, b"x" * 1000),
        store_max_bytes=5000)
    job = _submit(store)
    driver.run()
    room = store.room()
    _rec, link, linked = store.serve(job.id)
    assert linked
    store.delete(job.id)
    assert not os.path.exists(store.clip(job.id))
    assert store.room() == room
    os.unlink(link)
    assert store.room() == room + 1000


def test_a_clip_sent_twice_and_deleted_counts_once_until_both_end(
        tmp_path):
    """Two answers of one clip each hold a link to it: once a DELETE
    removes the clip, the store counts its bytes once, until the last
    answer ends."""
    store, clock, driver = _store(
        tmp_path, lambda j, s: _clip(j, s, b"x" * 1000),
        store_max_bytes=5000)
    job = _submit(store)
    driver.run()
    room = store.room()
    links = [store.serve(job.id)[1] for _ in range(2)]
    store.delete(job.id)
    assert store.room() == room
    os.unlink(links[0])
    assert store.room() == room
    os.unlink(links[1])
    assert store.room() == room + 1000


def test_the_sweep_leaves_a_clip_being_sent_and_frees_the_next(tmp_path):
    """Removing a clip an answer still sends frees nothing until the
    answer ends, so the sweep leaves it and removes the next oldest, and
    a new job's files fit."""
    def run(job, store):
        clock.now += 10
        _clip(job, store, b"x" * 2000)

    store, clock, driver = _store(tmp_path, run, store_max_bytes=5000,
                                  max_queue=3)
    jobs = [_submit(store) for _ in range(2)]
    driver.run()
    _rec, link, linked = store.serve(jobs[0].id)
    assert linked
    jobs.append(_submit(store))
    driver.run()
    jid = store.new_id()
    store.begin(jid)
    got = store.take(jid, 500)
    assert got == (True, 500, 0), got
    assert os.path.exists(store.clip(jobs[0].id))
    assert not os.path.exists(store.clip(jobs[1].id))
    os.unlink(link)


def test_a_queued_job_cancelled_never_runs_and_its_inputs_go(tmp_path):
    ran = []
    store, clock, driver = _store(tmp_path, lambda j, s: ran.append(j))
    job = _submit(store)
    got = store.delete(job.id)
    assert got == {"id": job.id, "object": "video.deleted", "deleted": True,
                   "status": "cancelled"}
    assert not os.path.exists(store.inputs(job.id))
    driver.run()
    assert ran == []
    assert store.get(job.id)["status"] == "cancelled"
    # A second DELETE removes the ended job.
    assert store.delete(job.id)["status"] == "cancelled"
    assert store.get(job.id) is None


def test_a_running_job_cancelled_stops_at_its_next_point(tmp_path):
    """DELETE sets the stop and answers at once; the job's next pause
    point raises and its part file goes."""
    from gmlx.serve.image_service import JobStopped

    store, clock, driver = _store(tmp_path)
    started, go = threading.Event(), threading.Event()

    def run(job, store):
        with open(store.clip(job.id) + ".part", "wb") as f:
            f.write(b"x")
        started.set()
        go.wait(10)
        if job.stop.is_set():
            raise JobStopped(timed_out=False)

    store._run = run
    job = _submit(store)
    t = threading.Thread(target=driver.run)
    t.start()
    assert started.wait(10)
    assert store.delete(job.id)["status"] == "cancelled"
    go.set()
    t.join(10)
    assert store.get(job.id)["status"] == "cancelled"
    assert os.listdir(os.path.join(str(store.root), "files")) == []
    assert not os.path.exists(store.inputs(job.id))


def test_a_restart_fails_the_jobs_a_past_server_queued_or_ran(tmp_path):
    store, clock, driver = _store(tmp_path, _clip)
    done = _submit(store)
    driver.run()
    queued = _submit(store)
    with open(store.clip(queued.id) + ".part", "wb") as f:
        f.write(b"x")
    # A crash after the clip's rename, before its record said completed.
    running = _submit(store)
    with store.lock:
        running.record["status"] = "in_progress"
        store._write(running)
    _clip(running, store)
    again = vj.VideoJobs(store.root, time_fn=clock, mono_fn=clock,
                         driver=Driver())
    again.configure(SimpleNamespace(video=VideoServerCfg()))
    for job in (queued, running):
        rec = again.get(job.id)
        assert rec["status"] == "failed"
        assert rec["error"] == {
            "code": "server_restarted", "message": "the server restarted "
            "before the job finished; submit it again",
            "type": "server_error", "status": 500}
        assert not os.path.exists(again.inputs(job.id))
    assert again.get(done.id)["status"] == "completed"
    assert os.listdir(os.path.join(str(store.root), "files")) == [
        f"{done.id}.mp4"]


def _restart(store, clock):
    again = vj.VideoJobs(store.root, time_fn=clock, mono_fn=clock,
                         driver=Driver())
    again.configure(SimpleNamespace(video=VideoServerCfg()))
    return again


def test_a_deleted_job_stays_gone_with_a_copy_of_its_record_beside(
        tmp_path):
    """A copy of a record, as the Finder names one, holds the job's id
    under another file name. The restart takes no job from it, so a job
    deleted after it is not back at the next restart."""
    import shutil

    store, clock, driver = _store(tmp_path, _clip)
    job = _submit(store)
    driver.run()
    path = store._record_path(job.id)
    shutil.copy(path, path[:-len(".json")] + " copy.json")
    again = _restart(store, clock)
    assert again.delete(job.id)["deleted"] is True
    assert _restart(store, clock).get(job.id) is None
    assert os.listdir(os.path.join(str(store.root), "jobs")) == []


def test_a_record_whose_id_leaves_the_store_removes_nothing_outside(
        tmp_path):
    """A record's id names its files: one that is not its file's own,
    such as a path out of the store, is no job, and its sweep removes
    nothing outside the store."""
    store, clock, _driver = _store(tmp_path)
    outside = os.path.join(str(store.root), "outside")
    os.makedirs(outside)
    jid = store.new_id()
    with open(store._record_path(jid), "w") as f:
        json.dump({"id": "../outside", "object": "video",
                   "status": "completed", "created_at": 0,
                   "completed_at": 0, "expires_at": 0, "error": None}, f)
    again = _restart(store, clock)
    assert os.path.isdir(outside)
    assert again.jobs == {}
    assert os.listdir(os.path.join(str(store.root), "jobs")) == []


def test_a_job_past_its_expiry_is_gone_before_any_sweep(tmp_path):
    store, clock, driver = _store(tmp_path, _clip, job_ttl_s=60)
    job = _submit(store)
    driver.run()
    clock.now += 59
    assert store.content(job.id) is not None
    clock.now += 1
    assert store.get(job.id) is None
    assert store.content(job.id) is None
    assert store.delete(job.id) is None
    # Its files stay until a sweep takes them.
    assert os.path.exists(store.clip(job.id))
    store.sweep()
    assert not os.path.exists(store.clip(job.id))
    assert not os.path.exists(store._record_path(job.id))


def test_a_job_that_runs_past_its_expiry_keeps_its_clip_from_its_end(
        tmp_path):
    """A clip that takes longer than job_ttl_s is kept job_ttl_s from
    when it finishes, not read as 404 and swept at once."""
    def run(job, store):
        clock.now += 100
        _clip(job, store)

    store, clock, driver = _store(tmp_path, run, job_ttl_s=60)
    job = _submit(store)
    driver.run()
    store.sweep()
    assert store.get(job.id)["status"] == "completed"
    assert store.content(job.id) == store.clip(job.id)
    clock.now += 59
    assert store.content(job.id) is not None
    clock.now += 1
    assert store.get(job.id) is None


def test_the_store_sweeps_at_install_and_after_each_job(tmp_path):
    store, clock, driver = _store(tmp_path, _clip, job_ttl_s=60)
    old = _submit(store)
    driver.run()
    clock.now += 61
    # After a job.
    _submit(store)
    driver.run()
    assert not os.path.exists(store.clip(old.id))
    # At install.
    again = vj.VideoJobs(store.root, time_fn=clock, mono_fn=clock,
                         driver=Driver())
    clock.now += 61
    again.configure(SimpleNamespace(video=VideoServerCfg(job_ttl_s=60)))
    assert os.listdir(os.path.join(str(store.root), "files")) == []


def test_past_the_bound_the_oldest_clip_goes_never_the_new_or_live_inputs(
        tmp_path):
    store, clock, driver = _store(tmp_path, store_max_bytes=2500)
    store._run = lambda job, s: _clip(job, s, b"x" * 1000)
    a = _submit(store)
    driver.run()
    clock.now += 1
    b = _submit(store)
    driver.run()
    clock.now += 1
    live = _submit(store)
    with open(os.path.join(store.inputs(live.id), "guide_0"), "wb") as f:
        f.write(b"g" * 400)
    store.sweep()
    assert store.get(a.id) is not None
    # 2000 clip bytes and the queued job's 403 input bytes.
    assert store.room() == 97
    assert os.path.exists(os.path.join(store.inputs(live.id), "guide_0"))
    store._run = lambda job, s: _clip(job, s, b"x" * 1000)
    driver.run()
    # 3000 clip bytes past 2500: the oldest goes, the one just written
    # stays.
    assert store.get(a.id) is None
    assert store.get(b.id)["status"] == "completed"
    assert store.get(live.id)["status"] == "completed"
    # A bound smaller than the new clip keeps it.
    store.settings = VideoServerCfg(store_max_bytes=10)
    c = _submit(store)
    store._run = lambda job, s: _clip(job, s, b"x" * 1000)
    driver.run()
    assert store.get(c.id)["status"] == "completed"
    # The next sweep holds the bound.
    assert store.room() == 10
    assert store.get(c.id) is None


def test_the_queue_refuses_at_max_queue_with_the_clip_median(tmp_path):
    store, clock, driver = _store(tmp_path, _clip, max_queue=2)
    _submit(store)
    _submit(store)
    assert store.full()
    with pytest.raises(vj.QueueFull) as ei:
        _submit(store)
    assert ei.value.retry_after == vj.FIRST_RETRY_S
    for s in (100, 300, 200):
        store.book.timed(s)
    clock.now += 0
    assert store.retry_after() == 200
    # Less the running job's time, at least 2.
    job = next(iter(store.jobs.values()))
    job.record["status"], job.started = "in_progress", clock.now - 150
    assert store.retry_after() == 50
    job.started = clock.now - 400
    assert store.retry_after() == 2
    # Long clips give a long wait: the median, with no cap.
    for s in (90_000,) * 5:
        store.book.timed(s)
    assert store.retry_after() == 90_000 - 400
    store.settings = VideoServerCfg(max_queue=0)
    assert store.full()


def test_progress_runs_over_the_steps_then_the_frames(tmp_path):
    store, clock, driver = _store(tmp_path)
    job = _submit(store, frames=100)
    job.record["status"] = "in_progress"
    job.started = clock.now
    store.progress(job, phase="denoising", step=(0, 20))
    assert job.record["progress"] == 5
    clock.now += 10
    store.progress(job, step=(10, 20))
    assert job.record["progress"] == 47
    assert job.record["gmlx"]["eta_s"] == round(10 * 52.5 / 47.5)
    store.progress(job, step=(20, 20), frames=(50, 100))
    assert job.record["progress"] == 95
    assert job.record["gmlx"]["phase"] == "denoising"


def test_progress_reads_0_until_the_prompt_encode(tmp_path):
    """The phases before the encode, the model's load among them, are no
    part of the clip's work: progress counts 5 from the encode on, a
    wait for memory after it included."""
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.record["status"] = "in_progress"
    job.started = clock.now
    seen = []
    for phase in ("reading the inputs", "rewriting the prompt",
                  "loading the model", "waiting for memory",
                  "encoding the prompt", "waiting for memory"):
        store.progress(job, phase=phase)
        seen.append(job.record["progress"])
    assert seen == [0, 0, 0, 0, 5, 5]


def test_a_wait_for_memory_holds_the_job_and_fails_it_past_expiry(
        tmp_path):
    from gmlx.serve.image_service import JobStopped

    store, clock, driver = _store(tmp_path, job_ttl_s=100)
    job = _submit(store)
    job.record["status"] = "in_progress"
    waits = []
    job.stop = SimpleNamespace(  # type: ignore[assignment]
        wait=lambda s: waits.append(s) or False, is_set=lambda: False)
    busy = vj.JobFailed(503, "models in use", retry_after=7, transient=True)
    store.wait(job, busy)
    assert waits == [7]
    assert job.record["gmlx"]["phase"] == "waiting for memory"
    clock.now += 95
    with pytest.raises(vj.JobFailed) as ei:
        store.wait(job, busy)
    assert ei.value.code == "timed_out"
    assert "models in use" in ei.value.message
    clock.now -= 95
    job.stop = SimpleNamespace(  # type: ignore[assignment]
        wait=lambda s: True, is_set=lambda: True)
    with pytest.raises(JobStopped):
        store.wait(job, busy)


# The driver's work, with stub parts.

def _run_clip_stubs(monkeypatch, outcomes):
    """run_clip with the read, the rewrite, the acquire and the worker
    parts stubbed. ``outcomes`` holds what each part raises, in order."""
    from gmlx.serve import residency, video_service
    from gmlx.serve import image_upsample
    from gmlx.video import loader, run

    calls = []
    monkeypatch.setattr(serving, "resolve_video_request",
                        lambda m: ("/abs/h3.gguf", SimpleNamespace(
                            path="/abs/h3.gguf", variant=None)))
    monkeypatch.setattr(loader, "identify_dit", lambda p, variant=None:
                        SimpleNamespace(family=None))
    monkeypatch.setattr(run, "read_anchors", lambda *a, **k: ([], []))
    monkeypatch.setattr(image_upsample, "revise", lambda *a, **k: None)

    class Hold:
        def release(self):
            calls.append("release")

    def acquire(model):
        calls.append("acquire")
        return Hold()

    monkeypatch.setattr(residency, "acquire_video_model", acquire)

    def worker(fn, *args):
        calls.append(fn.__name__)
        got = outcomes.pop(0) if outcomes else None
        if got is not None:
            raise got

    monkeypatch.setattr(video_service.WORKER, "run", worker)
    return calls


def test_a_transient_503_waits_and_runs_the_same_part_again(tmp_path,
                                                            monkeypatch):
    """The encode stays done: a 503 at the clip's turn waits, releases
    the entry, and tries the clip again."""
    calls = _run_clip_stubs(monkeypatch, [None, vj.JobFailed(
        503, "models in use", retry_after=3, transient=True)])
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat",
                  "settings": {"frames": 22}, "ffmpeg": "ffmpeg"}
    waits = []
    monkeypatch.setattr(store, "wait", lambda j, e: waits.append(e.message))
    vj.run_clip(job, store)
    assert waits == ["models in use"]
    assert calls == ["acquire", "_encode_part", "release", "acquire",
                     "_clip_part", "release", "acquire", "_clip_part",
                     "release"]


def test_a_503_that_needs_an_act_fails_the_job_with_its_text(tmp_path,
                                                             monkeypatch):
    calls = _run_clip_stubs(monkeypatch, [vj.JobFailed(
        503, "'zimage' is idle on the image worker; unload it (POST "
        "/unload) to make room", retry_after=5)])
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat",
                  "settings": {"frames": 22}, "ffmpeg": "ffmpeg"}
    with pytest.raises(vj.JobFailed) as ei:
        vj.run_clip(job, store)
    assert "unload it" in ei.value.message
    assert calls == ["acquire", "_encode_part", "release"]


def test_a_chat_model_in_use_holds_the_rewrite_to_its_release(
        tmp_path, monkeypatch):
    from gmlx.serve import image_upsample

    calls = _run_clip_stubs(monkeypatch, [])
    tries = []

    def revise(*a, **k):
        tries.append(k["frames"])
        if len(tries) == 1:
            raise image_service.ImageRequestError(
                503, "the chat model is loading", headers={
                    "Retry-After": "4"})
        return "the caption"

    monkeypatch.setattr(image_upsample, "revise", revise)
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat",
                  "settings": {"frames": 22}, "ffmpeg": "ffmpeg"}
    waits = []
    monkeypatch.setattr(store, "wait", lambda j, e: waits.append(
        e.retry_after))
    vj.run_clip(job, store)
    assert (tries, waits) == ([22, 22], [4])
    assert job.record["gmlx"]["revised_prompt"] == "the caption"
    assert calls[:2] == ["acquire", "_encode_part"]


def _rewrite_stubs(tmp_path, monkeypatch, chat=None):
    """run_clip's stubs with the real revise, on a video entry configured
    as the server reads it, whose header names MiniMax-H3, and a stand-in
    chat model; the system prompts it was sent."""
    from gmlx.config import VideoModelCfg
    from gmlx.serve import image_upsample
    from gmlx.video import loader
    from gmlx.video.families.h3 import family as hf

    revise = image_upsample.revise
    calls = _run_clip_stubs(monkeypatch, [])
    monkeypatch.setattr(image_upsample, "revise", revise)
    guide = tmp_path / "guide.md"
    guide.write_text("the guide")
    spec = VideoModelCfg(id="h3", path="/abs/h3.gguf", encoder="/abs/e.gguf",
                         vae="/abs/v.safetensors",
                         audio_vae="/abs/a.safetensors", upsampler="chat",
                         upsample_prompt=str(guide))
    monkeypatch.setattr(serving, "resolve_video_request",
                        lambda m: (spec.path, spec))

    def no_image_model(m):
        raise KeyError(m)
    monkeypatch.setattr(serving, "resolve_image_request", no_image_model)
    monkeypatch.setattr(loader, "identify_dit", lambda p, variant=None:
                        SimpleNamespace(family=hf.FAMILY, variant="fl2va"))
    monkeypatch.setattr(image_upsample, "_upsampler_identity",
                        lambda up: (up,))
    image_upsample.CAPTIONS.clear()
    systems = []
    reply = json.dumps({"instruction": "",
                        "integrated_multimodal_description": "[Shot 1] A cat.",
                        "overall_soundscape": "Purring.",
                        "non_diegetic_music": "N/A"})

    def stand_in(up, req, stop):
        systems.append(req.system)
        if chat is not None:
            chat(stop)
        return reply

    monkeypatch.setattr(image_upsample, "_chat", stand_in)
    return calls, systems


@pytest.mark.parametrize("first, last, task", [
    (False, False, "a T2VA video"), (True, False, "an I2VA video"),
    (True, True, "an FL2VA video"), (False, True, "an L2VA video")])
def test_a_jobs_prompt_is_rewritten_through_its_video_entry(
        tmp_path, monkeypatch, first, last, task):
    """The rewrite resolves the job's model among the video models, not
    the image models, reads its family from the entry's header, and its
    system prompt names the clip's length and its keyframes."""
    calls, systems = _rewrite_stubs(tmp_path, monkeypatch)
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat", "upsampling": None,
                  "rewrites": True, "settings": {"frames": 22},
                  "first_frame": "/s/first_frame" if first else None,
                  "last_frame": "/s/last_frame" if last else None,
                  "ffmpeg": "ffmpeg"}
    vj.run_clip(job, store)
    assert job.record["gmlx"]["revised_prompt"].startswith(
        "integrated_multimodal_description: [Shot 1] A cat.")
    assert f"{task} of 0.92 seconds" in systems[0]
    assert calls[:2] == ["acquire", "_encode_part"]


def test_a_job_sent_with_upsampling_off_runs_as_typed(tmp_path,
                                                      monkeypatch):
    calls, systems = _rewrite_stubs(tmp_path, monkeypatch)
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat", "upsampling": False,
                  "rewrites": False, "settings": {"frames": 22},
                  "ffmpeg": "ffmpeg"}
    vj.run_clip(job, store)
    assert systems == []
    assert "revised_prompt" not in job.record["gmlx"]
    assert calls[:2] == ["acquire", "_encode_part"]


def test_a_delete_during_the_rewrite_stops_the_job(tmp_path, monkeypatch):
    """The rewrite ends between tokens on the job's stop."""
    from gmlx.serve.image_service import JobStopped

    def chat(stop):
        if stop is job.stop:
            job.stop.set()
            raise JobStopped(timed_out=False)

    calls, systems = _rewrite_stubs(tmp_path, monkeypatch, chat)
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat", "upsampling": None,
                  "rewrites": True, "settings": {"frames": 22},
                  "ffmpeg": "ffmpeg"}
    with pytest.raises(JobStopped):
        vj.run_clip(job, store)
    assert len(systems) == 1 and calls == []


@pytest.mark.parametrize("rewrites", [True, False])
def test_a_job_whose_post_decided_a_rewrite_fails_when_none_comes(
        tmp_path, monkeypatch, rewrites):
    """revise answers None for an entry it cannot resolve or read: a job
    whose POST decided it rewrites fails with its reason, never runs as
    typed."""
    calls = _run_clip_stubs(monkeypatch, [])
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    job.params = {"model": "h3", "prompt": "a cat", "upsampling": None,
                  "rewrites": rewrites, "settings": {"frames": 22},
                  "ffmpeg": "ffmpeg"}
    if not rewrites:
        vj.run_clip(job, store)
        assert calls[:2] == ["acquire", "_encode_part"]
        return
    with pytest.raises(vj.JobFailed) as ei:
        vj.run_clip(job, store)
    assert (ei.value.status, ei.value.code) == (500, "upsample_failed")
    assert calls == []


def test_a_jobs_inputs_reach_the_read_as_the_post_kept_them(tmp_path,
                                                            monkeypatch):
    from gmlx.serve.patches.media_gate import MEDIA_MAX_PIXELS
    from gmlx.video import run

    _run_clip_stubs(monkeypatch, [])
    seen = {}
    monkeypatch.setattr(run, "read_anchors", lambda fam, s, **k: (
        seen.update(k), ([], []))[1])
    store, clock, driver = _store(tmp_path)
    job = _submit(store)
    guides = [("--guide", "/s/guide_0", 0, "guides[0] (guide_0)")]
    job.params = {"model": "h3", "prompt": "a cat", "settings": {
        "frames": 22}, "first_frame": "/s/first_frame",
        "last_frame": "/s/last_frame", "guides": guides,
        "guide_no_audio": True, "ffmpeg": "/opt/ffmpeg"}
    vj.run_clip(job, store)
    assert seen == {"first": "/s/first_frame", "last": "/s/last_frame",
                    "guides": guides, "guide_no_audio": True,
                    "ffmpeg": "/opt/ffmpeg",
                    "names": ("first_frame", "last_frame"), "paths": False,
                    "max_pixels": MEDIA_MAX_PIXELS}


# The routes.

def _png(w=32, h=32) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 90, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def routes(tmp_path, monkeypatch):
    """The video routes on an app behind the media gate, over a stub
    entry, a store on a fake clock whose driver runs when told, and no
    pool."""
    from gmlx.serve import media_programs
    from gmlx.serve.patches import media_gate

    store, clock, driver = _store(tmp_path, _clip)
    monkeypatch.setattr(videos, "_store", lambda: store)
    spec = SimpleNamespace(id="h3", defaults={}, upsampler=None,
                           upsample=True, encoder="/abs/enc.gguf",
                           load_signature=lambda: ())
    monkeypatch.setattr(serving, "video_model_ids", lambda: ["h3"])
    monkeypatch.setattr(serving, "resolve_video_request",
                        lambda m: ("/abs/h3.gguf", spec))
    monkeypatch.setattr(videos, "_entry_plan", lambda s: SimpleNamespace(
        family="h3", variant="fl2va", int8="off"))
    monkeypatch.setattr(media_programs, "find", lambda n: "/opt/ffmpeg")
    app = FastAPI()
    cfg = SimpleNamespace(video_models={"h3": {}}, video=VideoServerCfg())
    videos.install_video_routes(cfg, app=app)
    media_gate.install_media_gate(app=app)
    return SimpleNamespace(app=app, store=store, clock=clock, driver=driver,
                           client=TestClient(app), spec=spec)


def test_a_post_refused_before_its_body_leaves_no_store_behind(
        routes, monkeypatch):
    """A server with no ffmpeg refuses a POST before its body: the store
    no job has used stays unmade, as the server found it."""
    import shutil

    from gmlx.serve import media_programs

    shutil.rmtree(routes.store.root)
    monkeypatch.setattr(media_programs, "find", lambda n: None)
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 501, r.text
    assert not os.path.exists(routes.store.root)


def test_a_json_job_runs_to_its_mp4_through_the_routes(routes):
    c = routes.client
    r = c.post("/v1/videos", json={"prompt": "a cat", "seconds": "1",
                                   "size": "768x768", "seed": 7})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert (rec["status"], rec["model"], rec["size"], rec["seconds"]) == (
        "queued", "h3", "768x768", "1.62")
    assert rec["gmlx"]["frames"] == 39 and rec["gmlx"]["seed"] == 7
    assert routes.store.pending == {}
    jid = rec["id"]
    r = c.get(f"/v1/videos/{jid}/content")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "not_ready"
    routes.driver.run()
    assert c.get(f"/v1/videos/{jid}").json()["status"] == "completed"
    r = c.get(f"/v1/videos/{jid}/content")
    assert (r.status_code, r.content) == (200, b"mp4")
    assert r.headers["content-type"] == "video/mp4"
    r = c.delete(f"/v1/videos/{jid}")
    assert r.json() == {"id": jid, "object": "video.deleted",
                        "deleted": True, "status": "completed"}
    r = c.get(f"/v1/videos/{jid}")
    assert r.status_code == 404
    assert r.json()["error"]["message"] == (
        f"no video job {jid!r}; a job is gone once it expires or is "
        "deleted")


@pytest.mark.parametrize("end", ["delete", "expiry"])
def test_a_clip_removed_as_its_content_goes_out_is_sent_whole(routes, end):
    """GET content answers with the clip, then a DELETE, or the sweep at
    the job's expiry, removes it before a byte of it goes out: the
    answer is still the whole clip, never a 500."""
    import asyncio

    store, clock = routes.store, routes.clock
    job = _submit(store)
    routes.driver.run()
    response = asyncio.run(videos.content(job.id))
    if end == "delete":
        store.delete(job.id)
    else:
        clock.now += store.settings.job_ttl_s
        store.sweep()
    assert not os.path.exists(store.clip(job.id))
    sent = []

    async def receive():
        # A connected client sends nothing more: the response cancels this
        # wait once the clip has gone out.
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
    raised = None
    try:
        asyncio.run(response({"type": "http", "method": "GET",
                              "headers": []}, receive, send))
    except RuntimeError as e:       # the server answers it with a 500
        raised = e
    assert raised is None, raised
    assert sent[0]["status"] == 200
    assert b"".join(m.get("body", b"") for m in sent[1:]) == b"mp4"
    assert not os.path.exists(response.path)


def test_a_json_keyframe_is_a_data_url_written_to_the_jobs_inputs(routes):
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "input_reference": url})
    assert r.status_code == 200, r.text
    job = routes.store.jobs[r.json()["id"]]
    path = job.params["first_frame"]
    assert path == os.path.join(routes.store.inputs(job.id), "first_frame")
    with open(path, "rb") as f:
        assert f.read() == _png()
    for body, text in (
            ({"first_frame": "/etc/passwd"}, "first_frame is an image as a "
             "data URL"),
            ({"first_frame": url, "input_reference": url}, "both name the "
             "first frame"),
            ({"guides": [{"file": "g", "frame": 0}]}, "guides take files")):
        r = routes.client.post("/v1/videos", json={"prompt": "x", **body})
        assert r.status_code == 400, r.text
        assert text in r.json()["error"]["message"]


@pytest.mark.parametrize("body, status, text", [
    ({"prompt": "a cat", "references": [{"image": "x"}]}, 400,
     "references asks for MiniMax-H3's Ref2VA mode, which the server does "
     "not run; send the job without references"),
    ({"prompt": ""}, 400, "prompt is the text of the clip"),
    ({"prompt": "a cat", "seconds": "20"}, 400, "seconds takes 0.208333 to 15.0833, got '20'"),
    ({"prompt": "a cat", "size": "100x100"}, 400, "multiples of 32"),
    ({"prompt": "a cat", "seed": -1}, 400, "seed is a whole number"),
    ({"prompt": "a cat", "prompt_upsampling": True}, 400,
     "prompt_upsampling needs an upsampler"),
])
def test_a_job_the_server_cannot_run_is_refused_at_post(routes, body,
                                                       status, text):
    r = routes.client.post("/v1/videos", json=body)
    assert r.status_code == status, r.text
    assert text in r.json()["error"]["message"]
    assert routes.store.jobs == {}
    assert routes.store.pending == {}
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def _post(routes, how: str, fields: dict):
    if how == "json":
        return routes.client.post("/v1/videos", json=fields)
    body, ctype = _form(fields, {})
    return routes.client.post("/v1/videos", content=body,
                              headers={"content-type": ctype})


def _entry_at(monkeypatch, int8: str, variant: str = "fl2va") -> None:
    """The routes' entry plan at the int8 setting ``int8``."""
    monkeypatch.setattr(videos, "_entry_plan", lambda s: SimpleNamespace(
        family="h3", variant=variant, int8=int8))


@pytest.mark.parametrize("how", ["json", "form"])
@pytest.mark.parametrize("entry, sent, want", [
    ("off", None, "off"), ("off", "attention", "attention"),
    ("attention", None, "attention"), ("attention", "off", "off"),
    ("linears", None, "linears"), ("linears", "all", "all"),
    ("all", "linears", "linears")])
def test_a_jobs_int8_is_its_field_then_the_entrys(routes, monkeypatch, how,
                                                  entry, sent, want):
    """The job runs the int8 setting its POST names, else its entry's
    defaults.int8, and its record shows it as gmlx.int8."""
    from video_fixtures import int8_kernel

    int8_kernel(monkeypatch)
    _entry_at(monkeypatch, entry)
    fields = {"prompt": "a cat"} | ({"int8": sent} if sent else {})
    r = _post(routes, how, fields)
    assert r.status_code == 200, r.text
    assert r.json()["gmlx"]["int8"] == want
    assert routes.store.jobs[r.json()["id"]].params["int8"] == want


@pytest.mark.parametrize("entry, sent", [
    ("off", "linears"), ("off", "all"), ("attention", "linears"),
    ("linears", "off"), ("linears", "attention"), ("all", "off")])
def test_a_job_whose_linears_differ_from_its_entrys_is_a_400(
        routes, monkeypatch, entry, sent):
    """The linears are a property of the loaded DiT: a job names a value
    whose linears match its entry's, and any other is refused naming the
    entry's value and the setting that changes it, with no job."""
    from video_fixtures import int8_kernel

    int8_kernel(monkeypatch)
    _entry_at(monkeypatch, entry)
    allowed = (("off", "attention") if entry in ("off", "attention")
               else ("linears", "all"))
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "int8": sent})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (
        f"this model's entry loads its DiT at int8 {entry}, so a request "
        f"names int8 {allowed[0]} or {allowed[1]}, not {sent}; set the "
        f"entry's defaults.int8 to run {sent}", "int8")
    assert routes.store.jobs == {} and routes.store.pending == {}


@pytest.mark.parametrize("how", ["json", "form"])
def test_an_int8_value_off_the_list_is_a_400_naming_it(routes, how):
    r = _post(routes, how, {"prompt": "a cat", "int8": "int4"})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (
        "int8 is off, attention, linears or all, not 'int4'", "int8")
    assert routes.store.jobs == {} and routes.store.pending == {}


@pytest.mark.parametrize("sent", [["all"], {}, 8, True])
def test_an_int8_of_another_json_type_is_a_400_naming_it(routes, sent):
    """A JSON list, object, number or boolean in the int8 field is the
    field's 400, not a server error."""
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "int8": sent})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (
        f"int8 is off, attention, linears or all, not {sent!r}", "int8")
    assert routes.store.jobs == {} and routes.store.pending == {}


@pytest.mark.parametrize("case", ["sparse", "kq", "nax"])
def test_int8_this_server_cannot_run_is_a_400_before_the_job(
        routes, monkeypatch, case):
    """The variant with sparse attention, an mlx-kquant with no sdpa_qk8
    and a GPU with no NAX units are refused at the POST with the run's
    own text, naming the field, and leave no job."""
    import mlx_kquant as kq

    from video_fixtures import int8_kernel

    int8_kernel(monkeypatch, nax=case != "nax")
    variant = "fasth3" if case == "sparse" else "fl2va"
    _entry_at(monkeypatch, "off", variant)
    if case == "kq":
        monkeypatch.setattr(kq, "sdpa_qk8", None)
    what = f"MiniMax-H3 {variant} runs with int8 attention"
    want = {"sparse": "MiniMax-H3 fasth3 runs video sparse attention, and "
            "int8 attention applies to dense attention only",
            "kq": f"{what}, and the installed mlx-kquant {kq.__version__} "
            "has no sdpa_qk8; update mlx-kquant to 0.5.10 or later",
            "nax": f"{what}, whose kernel needs a GPU with NAX units (M5 or "
            "newer), and this Mac's GPU has none"}[case]
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "int8": "attention"})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (want, "int8")
    assert routes.store.jobs == {} and routes.store.pending == {}
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


@pytest.mark.parametrize("sent", [None, "linears", "all"])
def test_fasth3_takes_int8_linears_and_refuses_int8_attention(
        routes, monkeypatch, sent):
    """FastH3 on an entry at int8 linears takes a job at linears, and a
    job at all is a 400 with the attention text and no job."""
    from video_fixtures import int8_kernel, vsa_kernel

    int8_kernel(monkeypatch)
    vsa_kernel(monkeypatch)
    _entry_at(monkeypatch, "linears", "fasth3")
    fields = {"prompt": "a cat"} | ({"int8": sent} if sent else {})
    r = routes.client.post("/v1/videos", json=fields)
    if sent != "all":
        assert r.status_code == 200, r.text
        assert r.json()["gmlx"]["int8"] == "linears"
        return
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (
        "MiniMax-H3 fasth3 runs video sparse attention, and int8 attention "
        "applies to dense attention only", "int8")
    assert routes.store.jobs == {} and routes.store.pending == {}


def test_no_ffmpeg_is_a_501_with_the_missing_program_text(routes,
                                                         monkeypatch):
    from gmlx.serve import media_programs

    monkeypatch.setattr(media_programs, "find", lambda n: None)
    monkeypatch.setattr(media_programs, "problem", lambda n: (
        "The gmlx server finds no ffmpeg on its PATH. Install it with `brew install "
        "ffmpeg`, or start the server from a shell whose PATH holds your "
        "ffmpeg."))
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 501
    assert "brew install ffmpeg" in r.json()["error"]["message"]
    assert r.headers["connection"] == "close"


def test_a_full_queue_is_a_429_with_its_retry_after(routes):
    routes.store.settings = VideoServerCfg(max_queue=1)
    assert routes.client.post("/v1/videos",
                              json={"prompt": "a"}).status_code == 200
    r = routes.client.post("/v1/videos", json={"prompt": "b"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == str(vj.FIRST_RETRY_S)
    assert r.headers["connection"] == "close"
    assert routes.store.pending == {}


def test_post_takes_no_pool_lock_and_no_acquire(routes, monkeypatch):
    """The signed rule: the request thread books nothing; the job's
    memory is decided on the video worker."""
    from gmlx.serve import residency, video_service
    from gmlx.serve.patches import _common
    from gmlx.video import memfit

    def refuse(*a, **k):
        raise AssertionError("the request thread acquired")

    class Refuses:
        def __enter__(self):
            raise AssertionError("the request thread took a pool lock")

        def __exit__(self, *a):
            return False

        acquire = __enter__

    pool = SimpleNamespace(
        _lock=Refuses(), _build_lock=Refuses(),
        figures=SimpleNamespace(budget=100 << 30, resident=0, incoming=0,
                                entries=()))
    monkeypatch.setattr(residency, "acquire_video_model", refuse)
    monkeypatch.setattr(residency, "_installed_pool", lambda: pool)
    monkeypatch.setattr(_common, "_get_pool", lambda: pool)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(videos, "_tokenizer", lambda fam, spec: None)
    monkeypatch.setattr(memfit, "prompt_rows", lambda *a: 12)
    budgets = []
    monkeypatch.setattr(video_service, "plan_clip", lambda plan, **kw: (
        budgets.append(kw["budget"]), SimpleNamespace(over=[]))[1])
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 200, r.text
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": _png()})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    # The estimate ran on the pool's figures, read with no lock.
    assert budgets == [100 << 30] * 2


def test_the_estimate_refuses_a_clip_past_the_budget_beside_pinned_models(
        routes, monkeypatch):
    """The clip is priced at the job's int8 setting, which may run int8
    attention on an entry at off."""
    from gmlx.serve import residency, video_service
    from video_fixtures import int8_kernel

    int8_kernel(monkeypatch)

    pinned = SimpleNamespace(pinned=True, footprint=60, model_path="/abs/x")
    own = SimpleNamespace(pinned=True, footprint=1000, model_path="/abs/h3"
                          ".gguf")
    pool = SimpleNamespace(figures=SimpleNamespace(budget=100,
                                                   entries=(pinned, own)))
    monkeypatch.setattr(residency, "_installed_pool", lambda: pool)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    monkeypatch.setattr(videos, "_tokenizer", lambda fam, spec: None)
    from gmlx.video import memfit

    monkeypatch.setattr(memfit, "prompt_rows", lambda *a: 12)
    seen = {}

    def plan_clip(plan, **kw):
        seen.update(kw)
        return SimpleNamespace(over=["denoise"],
                               describe=lambda: "denoise 9 GiB",
                               hint=lambda: "use a smaller size")

    monkeypatch.setattr(video_service, "plan_clip", plan_clip)
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "size": "768x768",
                                               "int8": "attention"})
    assert r.status_code == 413, r.text
    assert r.json()["error"]["message"].endswith(
        "does not fit in memory: denoise 9 GiB; use a smaller size")
    assert (seen["budget"], seen["text"], seen["int8"]) == (40, 12,
                                                            "attention")


def test_the_gate_reads_no_video_body(routes, monkeypatch):
    from gmlx.serve.patches import media_gate

    async def read(*a, **k):
        raise AssertionError("the gate read the body")

    monkeypatch.setattr(media_gate, "_read_body", read)
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 200, r.text


def _form(fields: dict, files: dict) -> tuple[bytes, str]:
    b = "gmlxboundary"
    out = []
    for k, v in fields.items():
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"'
                   f"\r\n\r\n{v}\r\n".encode())
    for k, v in files.items():
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; '
                   f'filename="{k}.bin"\r\nContent-Type: application/'
                   f"octet-stream\r\n\r\n".encode() + v + b"\r\n")
    out.append(f"--{b}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={b}"


@pytest.mark.parametrize("how", ["json", "form"])
@pytest.mark.parametrize("field", ["first_frame", "input_reference",
                                   "last_frame"])
def test_an_empty_keyframe_field_is_a_400_naming_it(routes, how, field):
    """A client that sends a keyframe field empty gets a 400, not a clip
    without that keyframe."""
    if how == "form":
        body, ctype = _form({"prompt": "a cat", field: ""}, {})
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": ctype})
    else:
        r = routes.client.post("/v1/videos",
                               json={"prompt": "a cat", field: ""})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["param"], err["message"]) == (
        field, f"{field} is an image as a data URL "
        "(data:image/png;base64,...) in a JSON body, or a file in a "
        "multipart form")
    assert routes.store.jobs == {}
    assert routes.store.pending == {}


def test_a_form_streams_its_files_to_the_jobs_inputs(routes):
    body, ctype = _form({"prompt": "a cat", "guides": json.dumps(
        [{"file": "guide_0", "frame": 0}]), "guide_no_audio": "true"},
        {"first_frame": _png(), "guide_0": _png(64, 64)})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    job = routes.store.jobs[r.json()["id"]]
    folder = routes.store.inputs(job.id)
    assert sorted(os.listdir(folder)) == ["first_frame", "guide_0"]
    assert job.params["guides"] == [("--guide",
                                     os.path.join(folder, "guide_0"), 0,
                                     "guides[0] (guide_0)")]
    assert job.params["guide_no_audio"] is True


@pytest.mark.parametrize("guides, text", [
    (json.dumps([{"file": "guide_0", "kind": []}]),
     "guides[0].kind is video or audio"),
    (json.dumps([{"file": {}}]), "guides[0].file names a file field"),
    ("[" * 100_000, "guides is a JSON list")])
def test_a_malformed_guides_field_is_a_400(routes, guides, text):
    """A guide whose kind or file is not a string, or a guides field too
    deep to parse, is the field's 400, never a 500."""
    body, ctype = _form({"prompt": "a cat", "guides": guides},
                        {"guide_0": _png(64, 64)})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 400, r.text
    assert text in r.json()["error"]["message"]
    assert routes.store.jobs == {}


def test_a_guides_field_whose_parse_passes_the_memory_free_is_a_413(
        routes, monkeypatch):
    """A guides field of many small objects parses to far more than its
    text: it is priced as a JSON body is, and refused before its parse
    when that passes the memory free."""
    guides = "[" + ",".join(["{}"] * 50_000) + "]"
    body, ctype = _form({"prompt": "a cat", "guides": guides},
                        {"guide_0": _png(64, 64)})
    monkeypatch.setattr(body_intake, "_pool_free", lambda: 4 << 20)
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 413, r.text
    err = r.json()["error"]
    assert err["param"] == "guides"
    assert err["message"].startswith("guides is "), err["message"]
    assert routes.store.jobs == {}


@pytest.mark.parametrize("over", [False, True], ids=["budget", "overcommit"])
def test_a_prompt_with_a_lone_surrogate_is_a_400(routes, monkeypatch, over):
    """A lone surrogate escape parses from JSON to a str that encodes to
    no UTF-8: the tokenizer of the budget's estimate fails on it, and so
    does the record's JSON under overcommit."""
    from gmlx.serve import residency
    from gmlx.video import memfit

    pool = SimpleNamespace(figures=SimpleNamespace(budget=100 << 30,
                                                   entries=()))
    monkeypatch.setattr(residency, "_installed_pool", lambda: pool)
    monkeypatch.setenv("GMLX_OVERCOMMIT", "1" if over else "0")
    monkeypatch.setattr(videos, "_tokenizer", lambda fam, spec: None)
    # As a tokenizer, it encodes the prompt.
    monkeypatch.setattr(memfit, "prompt_rows",
                        lambda fam, text, tok, sizes: len(text.encode()))
    client = TestClient(routes.app, raise_server_exceptions=False)
    r = client.post("/v1/videos",
                    content=b'{"prompt": "a \\ud800 cat", "model": "h3"}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["message"] == (
        "prompt holds a lone surrogate escape")
    assert routes.store.jobs == {}


def test_a_seed_past_any_number_is_a_400(routes):
    """1e999 parses to inf, which int() refuses with OverflowError."""
    r = routes.client.post("/v1/videos",
                           content=b'{"prompt": "a cat", "seed": 1e999}',
                           headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert "seed is a whole number" in r.json()["error"]["message"]


def test_a_form_past_the_store_room_is_a_413_before_its_read(routes):
    """On a live socket: the Content-Length is past the room, so the
    answer comes before the body is sent, and the connection closes."""
    import http.client

    from live_socket import closed, peer, send, serving

    routes.store.settings = VideoServerCfg(store_max_bytes=1 << 20)
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": b"x" * 64})
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        other = peer(conn)
        r = send(conn, "/v1/videos", body[:64], length=8 << 20, ctype=ctype)
        assert r.status == 413, r.body
        assert json.loads(r.body)["error"]["message"] == (
            "the files of this job are 8.0 MiB, and the video job store has "
            "1.0 MiB of room left; send smaller files, delete finished "
            "jobs, or raise server.video.store_max_bytes")
        assert r.getheader("connection") == "close"
        assert closed(other)
        conn.close()
    assert routes.store.jobs == {}
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def test_a_form_with_no_length_stops_at_the_room_and_leaves_no_part(
        routes):
    routes.store.settings = VideoServerCfg(store_max_bytes=1 << 16)
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": b"x" * (
        1 << 18)})

    def chunks():
        for i in range(0, len(body), 4096):
            yield body[i:i + 4096]

    r = routes.client.post("/v1/videos", content=chunks(),
                           headers={"content-type": ctype})
    assert r.status_code == 413, r.text
    assert r.json()["error"]["message"].startswith(
        "the files of this job are over the 64.0 KiB of room the video "
        "job store has left;")
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def test_a_json_body_past_the_free_figure_is_a_413_with_no_byte_read(
        routes, monkeypatch):
    import http.client

    from live_socket import closed, peer, send, serving

    monkeypatch.setattr(body_intake, "_pool_free", lambda: 1 << 20)
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        other = peer(conn)
        r = send(conn, "/v1/videos", b"{", length=8 << 20)
        assert r.status == 413, r.body
        assert json.loads(r.body)["error"]["message"] == (
            "the request body is 8.0 MiB, its read and parse hold at least "
            "16.0 MiB, and 1.0 MiB of memory is free beside the models "
            "loaded; send a smaller body, or unload a model to make room")
        assert closed(other)
        conn.close()


@pytest.mark.parametrize("how", ["memory", "not_json"])
def test_a_video_body_refusal_is_in_the_server_log(routes, monkeypatch,
                                                   capsys, how):
    """The video route reads its own body, as the images routes do, and
    prints the body's refusal to the log as the media gate does."""
    from gmlx.serve.patches import hardening

    monkeypatch.setattr(hardening, "_refusal_logs", {})
    body = b'{"prompt": "a cat"'
    if how == "memory":
        monkeypatch.setattr(body_intake, "_pool_free", lambda: 1 << 10)
        body = json.dumps({"prompt": "a cat " * 1000}).encode()
    r = routes.client.post("/v1/videos", content=body, headers={
        "content-type": "application/json"})
    assert r.status_code in (400, 413), r.text
    message = r.json()["error"]["message"]
    assert (f"[server] refused a request with status {r.status_code}: "
            f"{message}" in capsys.readouterr().out.splitlines())


def test_a_form_that_stalls_frees_the_room_it_holds(routes, monkeypatch):
    """A form whose client stops sending holds the room of its whole
    Content-Length in the job store: past the stall bound, the server
    answers 408, ends the connection and frees the room and the file it
    began."""
    import http.client

    from live_socket import closed, peer, send, serving

    monkeypatch.setattr(body_intake, "STALL_S", 0.2)
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": _png()})
    head = body[:len(body) // 2]
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        other = peer(conn)
        r = send(conn, "/v1/videos", head, length=len(body) + (8 << 20),
                 ctype=ctype)
        assert r.status == 408, r.body
        assert json.loads(r.body)["error"]["message"].startswith(
            "the form sent nothing for 0.2 s with 8.")
        assert closed(other)
        conn.close()
    assert routes.store.pending == {}
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def test_a_video_json_body_is_priced_at_what_its_read_and_parse_hold(
        routes, monkeypatch):
    """A JSON body is priced at what its one read and parse hold, each str
    at the width of its own characters, so a prompt that is not ASCII
    leaves its keyframe at a byte a character: it fits only where that
    fits."""
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    body = json.dumps({"prompt": "un caf\u00e9 " + "a" * (256 << 10),
                       "first_frame": url}).encode()
    need = body_intake.json_price(body)
    for free, status in ((need - 1, 413), (need, 200)):
        monkeypatch.setattr(body_intake, "_pool_free", lambda f=free: f)
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": "application/json"})
        assert r.status_code == status, r.text
        if status == 413:
            assert " with the values it parses to, " in \
                r.json()["error"]["message"]
    assert len(routes.store.jobs) == 1
    assert routes.store.pending == {}


@pytest.mark.parametrize("wide", [False, True])
def test_a_video_forms_text_is_priced_at_its_own_width(routes, monkeypatch,
                                                       wide):
    """A form's text field is priced as it arrives, at its bytes and the
    str they decode to at the width of its own characters: it fits only
    where that fits."""
    text = "a" * (256 << 10) + ("\U0001F600" if wide else "")
    body, ctype = _form({"prompt": text}, {"first_frame": _png()})
    # The field's bytes with a bytearray's room to grow, and its str with
    # the narrower copy the decode widens from.
    n = len(text.encode())
    need = math.ceil(n * 1.125) + n * (6 if wide else 1) + body_intake._SLACK
    for free, status in ((need - 1, 413), (need, 200)):
        monkeypatch.setattr(body_intake, "_pool_free", lambda f=free: f)
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": ctype})
        assert r.status_code == status, r.text
        if status == 413:
            assert ", their read and parse hold " in \
                r.json()["error"]["message"]
    assert len(routes.store.jobs) == 1
    assert routes.store.pending == {}


def _photo(w: int, h: int) -> bytes:
    """A PNG of noise, which compresses as little as a photo does."""
    import numpy as np
    from PIL import Image

    pixels = np.random.default_rng(0).integers(0, 256, (h, w, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "PNG")
    return buf.getvalue()


def test_a_keyframe_decodes_in_the_parse_turn(routes, monkeypatch):
    """A JSON keyframe's decode is priced with its body, so it runs in the
    turn the parses take, never beside another body's parse. The estimate,
    which tokenizes the prompt, however long, runs out of the turn."""
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    body = json.dumps({"prompt": "a cat " * 10000,
                       "first_frame": url}).encode()
    held: dict = {}

    def turn_held():
        return any(t.locked() for t in body_intake._TURNS.values())

    def recorded(name, real):
        def run(*a):
            held.setdefault(name, []).append(turn_held())
            return real(*a)
        return run
    monkeypatch.setattr(videos, "_data_url",
                        recorded("decode", videos._data_url))
    monkeypatch.setattr(videos, "_pixels", recorded("pixels", videos._pixels))
    monkeypatch.setattr(videos, "estimate",
                        recorded("estimate", videos.estimate))
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    assert held == {"decode": [True], "pixels": [True], "estimate": [False]}


def test_keyframe_posts_past_the_executors_width_all_answer(routes):
    """Three keyframe jobs posted at once on a loop whose default executor
    has two threads all answer: no worker thread waits on the loop for the
    parse turn, so the turn's own thread always gets one."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    routes.store.configure(SimpleNamespace(video=VideoServerCfg(max_queue=3)))
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    executor = ThreadPoolExecutor(max_workers=2)

    async def main():
        asyncio.get_running_loop().set_default_executor(executor)
        transport = httpx.ASGITransport(app=routes.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            posts = [asyncio.create_task(client.post("/v1/videos", json={
                "prompt": f"a cat {i}", "first_frame": url}))
                for i in range(3)]
            done, pending = await asyncio.wait(posts, timeout=30)
            return [t.result().status_code for t in done], len(pending)

    loop = asyncio.new_event_loop()
    try:
        codes, stuck = loop.run_until_complete(main())
    finally:
        # A stuck post leaves threads that wait on the loop: cancel every
        # task, so they wake, before the loop closes.
        tasks = asyncio.all_tasks(loop)
        for t in tasks:
            t.cancel()
        if tasks:
            loop.run_until_complete(asyncio.wait(tasks))
        executor.shutdown(wait=False, cancel_futures=True)
        loop.close()
    assert stuck == 0, "a keyframe post never answered"
    assert codes == [200, 200, 200]


def test_a_keyframe_job_is_read_within_its_price(routes, monkeypatch,
                                                 tmp_path):
    """A job with a photo for its first frame and a prompt that is not
    ASCII fits at three times its body and a little more, where the copies
    of the body the read and the parse made, all at four bytes a character,
    took ten; and what its read, its parse and the keyframe's decode hold
    at their peak never passes the price: one byte less is refused."""
    import asyncio
    import tracemalloc

    from starlette.requests import Request

    url = "data:image/png;base64," + base64.b64encode(_photo(768, 768)) \
        .decode()
    body = json.dumps({"prompt": "un caf\u00e9 au soleil, la cam\u00e9ra "
                       "avance lentement", "first_frame": url},
                      ensure_ascii=False).encode()
    budget = 3 * len(body) + body_intake._SLACK
    monkeypatch.setattr(body_intake, "_pool_free", lambda: budget)
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text

    def run(free):
        chunks = [body[i:i + (64 << 10)]
                  for i in range(0, len(body), 64 << 10)]

        async def receive():
            if chunks:
                return {"type": "http.request", "body": chunks.pop(0),
                        "more_body": bool(chunks)}
            return {"type": "http.disconnect"}
        req = Request({"type": "http", "method": "POST", "path": "/v1/videos",
                       "query_string": b"", "headers": [
                           (b"content-type", b"application/json"),
                           (b"content-length", str(len(body)).encode())]},
                      receive)

        async def read():
            doc = await body_intake.read_json(req, free)
            await asyncio.to_thread(videos._data_url, doc["first_frame"],
                                    "first_frame", str(tmp_path), None)
        return read
    tracemalloc.start()
    try:
        asyncio.run(run(budget)())
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    with pytest.raises(body_intake.BodyRefused) as e:
        asyncio.run(run(peak - 1)())
    assert e.value.status == 413, e.value


def _shaped(shape: str, n: int = 64 << 10) -> bytes:
    """A JSON body of about ``n`` bytes whose parse the price must bound:
    lists of small values, a dict of short keys, a prompt with one emoji
    escaped or literal, or one plain string."""
    emoji = "\U0001F600"
    if shape in ("emoji_escaped", "emoji_literal", "string"):
        prompt = "a" * n + ("" if shape == "string" else emoji)
        return json.dumps({"prompt": prompt},
                          ensure_ascii=shape != "emoji_literal").encode()
    unit = {"objects": "{}", "floats": "0.5", "strs": '"ab"'}.get(shape)
    if unit is not None:
        x = "[" + ",".join([unit] * (n // (len(unit) + 1))) + "]"
    else:
        x = "{" + ",".join(f'"k{i}":0' for i in range(n // 8)) + "}"
    return f'{{"prompt": "a cat", "x": {x}}}'.encode()


SHAPES = ["objects", "floats", "strs", "pairs", "emoji_escaped",
          "emoji_literal", "string"]


@pytest.mark.parametrize("shape", SHAPES)
def test_a_json_body_is_priced_by_what_its_parse_makes(routes, monkeypatch,
                                                       shape):
    """Small values parse to far more than their text, and an emoji widens
    the str it is in: each shape is refused a byte under its price and fits
    at it."""
    body = _shaped(shape)
    need = body_intake.json_price(body)
    for free, status in ((need - 1, 413), (need, 200)):
        monkeypatch.setattr(body_intake, "_pool_free", lambda f=free: f)
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": "application/json"})
        assert r.status_code == status, r.text
        if status == 413:
            assert r.json()["error"]["message"].startswith(
                f"the request body is {body_intake._mib(len(body), True)}, "
                "its read and parse hold ")
    assert len(routes.store.jobs) == 1


def test_a_jobs_anchor_refusal_names_its_field_not_its_path(tmp_path):
    """A guide off the clip is named by its field in the job's error,
    never by its file under the store."""
    from gmlx.video import run
    from gmlx.video.registry import get_family

    path = tmp_path / "guide_0"
    path.write_bytes(_png())
    with pytest.raises(ValueError) as ei:
        run.read_anchors(get_family("h3"), {"frames": 22, "width": 32,
                                            "height": 32},
                         guides=[("--guide", str(path), 99,
                                  "guides[0] (guide_0)")],
                         names=("first_frame", "last_frame"), paths=False)
    assert str(ei.value).startswith("guides[0] (guide_0): ")
    assert str(tmp_path) not in str(ei.value)


def _clip_part_stubs(monkeypatch, order, generate):
    from gmlx.serve import gpu_share, video_service
    from gmlx.serve.patches import image_steps as pis
    from gmlx.video import pipeline, run

    pool = object()
    monkeypatch.setattr(pis, "_check_clip", lambda *a, **k: order.append(
        "check"))
    monkeypatch.setattr(pis, "_lora_pool", lambda: pool)

    def ensure_dit(s):
        # The job's encode let the DiT go: it loads again at the entry's
        # int8 setting.
        order.append("ensure_dit")
        if s.pipe.dit is None:
            s.pipe.dit = SimpleNamespace(attention=s.plan.attention)
    monkeypatch.setattr(video_service, "ensure_dit", ensure_dit)
    monkeypatch.setattr(pis, "_factor_turn", lambda p, e, s, core, subject:
                        order.append(("turn", core, subject)))
    monkeypatch.setattr(image_service, "lora_settle", lambda p, e, subject:
                        order.append(("settle", subject)))
    monkeypatch.setattr(pipeline, "generate", generate)
    monkeypatch.setattr(run, "decode_and_mux", lambda *a, **k: order.append(
        "decode"))
    monkeypatch.setattr(gpu_share, "pacer", lambda: None)
    served = SimpleNamespace(
        pipe=SimpleNamespace(default_table="t0", dit=None),
        spec=SimpleNamespace(vae="v", audio_vae="a"),
        plan=_Plan("off"))
    return SimpleNamespace(_entry=SimpleNamespace(pipeline=served))


class _Plan(SimpleNamespace):
    """An entry plan at an int8 setting, with the DiT attention of it."""

    def __init__(self, int8: str):
        super().__init__(int8=int8)

    @property
    def attention(self) -> str:
        from gmlx.video.registry import int8_parts

        return int8_parts(self.int8)[0]


def _clip_job(store):
    import numpy as np

    job = _submit(store)
    job.record["status"] = "in_progress"
    job.params = {"settings": {"frames": 22, "width": 64, "height": 64,
                               "steps": 2, "shifts": {}, "seed": 1},
                  "ffmpeg": "ffmpeg"}
    job.kept = {"feats": np.zeros((1, 4, 8), np.float32), "spans": (),
                "rows": (), "anchors": ()}
    return job


def test_the_jobs_turn_runs_after_the_dit_loads_and_settles_in_finally(
        tmp_path, monkeypatch):
    """The signed rule on the worker: the clip is checked, the DiT loads,
    the turn books the default table, and the settle runs when the
    denoise raises."""
    order = []

    def generate(*a, **k):
        order.append("generate")
        raise RuntimeError("boom")

    hold = _clip_part_stubs(monkeypatch, order, generate)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    with pytest.raises(RuntimeError):
        vj._clip_part(job, store, hold, store.clip(job.id))
    assert order == ["check", "ensure_dit", ("turn", None, vj.JOB),
                     "generate", ("settle", vj.JOB)]


@pytest.mark.parametrize("entry, left, sent, want", [
    ("off", None, "attention", "int8"), ("attention", "dense", None, "int8"),
    ("attention", None, "off", "dense"), ("off", "int8", None, "dense"),
    ("linears", "int8", None, "dense"), ("linears", None, "all", "int8")])
def test_the_clip_is_priced_and_runs_at_the_jobs_int8(
        tmp_path, monkeypatch, entry, left, sent, want):
    """The worker sets the DiT's attention before each clip from the
    job's int8 setting, else the entry's: after the DiT loads again at
    the entry's (``left`` None, as after the job's encode), or over
    whatever the last job or step left. The clip's check prices the
    job's."""
    import mlx.core as mx

    from gmlx.serve.patches import image_steps as pis

    ran, checked = [], []

    def generate(pipe, feats, **k):
        ran.append(pipe.dit.attention)
        return mx.zeros((1,)), mx.zeros((1,))

    hold = _clip_part_stubs(monkeypatch, [], generate)
    monkeypatch.setattr(pis, "_check_clip", lambda *a, **k: checked.append(
        k["int8"]))
    served = hold._entry.pipeline
    served.plan.int8 = entry
    if left is not None:
        served.pipe.dit = SimpleNamespace(attention=left)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    if sent:
        job.params["int8"] = sent
    vj._clip_part(job, store, hold, store.clip(job.id))
    assert (ran, checked) == ([want], [sent])


def test_the_jobs_pacer_reaches_the_sampler_and_stops_the_job(
        tmp_path, monkeypatch):
    from gmlx.serve.image_service import JobStopped

    order = []
    seen = {}

    def generate(pipe, feats, **k):
        seen["pacer"] = k["pacer"]
        k["pacer"](())
        k["on_step"](0, 2)
        job.stop.set()
        try:
            k["pacer"](())
        except JobStopped:
            seen["stopped"] = True
            raise
        pytest.fail("the pacer ran on past the stop")

    hold = _clip_part_stubs(monkeypatch, order, generate)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    with pytest.raises(JobStopped):
        vj._clip_part(job, store, hold, store.clip(job.id))
    assert seen.get("pacer") is not None and seen.get("stopped")
    assert job.record["gmlx"]["step"] == 1
    assert order[-1] == ("settle", vj.JOB)


class _Points:
    """A DiT whose forward marks 50 pause points and does nothing else,
    the array of each point its index."""

    def __init__(self, at=None):
        self.at = at
        self.last = -1

    def plan(self, layout, t_video, t_audio):
        return None

    def __call__(self, video, audio, *a, **k):
        import mlx.core as mx

        from gmlx.image import pacing

        for i in range(50):
            self.last = i
            pacing.point(mx.array(i))
            if self.at is not None:
                self.at(i)
        return mx.zeros(video.shape), mx.zeros(audio.shape)


def _sampling(monkeypatch, size, frames, text, dit, hit):
    """A generate that samples 2 steps of ``dit`` on the clip's layout
    through ``api.sample`` with the job's pacer, each evaluation of the
    row budget recorded in ``hit``. Returns (generate, the layout)."""
    import mlx.core as mx

    from gmlx.video import api, layout
    from gmlx.video.families.h3 import family as hf

    vs, aus = hf.latent_shapes(frames, *size)
    lay = layout.pack(hf.TIMELINE, text, *vs[2:], aus[3])

    class Budget(api.RowBudget):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._eval = lambda *x: hit.append(int(x[0].item()))

    monkeypatch.setattr(api, "RowBudget", Budget)

    def generate(pipe, feats, **k):
        return api.sample(dit, mx.zeros(vs), mx.zeros(aus),
                          mx.zeros((1, text, 8)), lay, steps=2,
                          shifts=hf.SHIFTS, on_step=k["on_step"],
                          pacer=k["pacer"])
    return generate, lay


@pytest.mark.parametrize("size, frames, text, rows, every", ROW_REGIMES,
                         ids=ROW_REGIME_IDS)
def test_the_jobs_sampler_evaluates_on_the_row_budget_with_its_checks(
        tmp_path, monkeypatch, size, frames, text, rows, every):
    """A job's sampler evaluates on the clip's rows as the CLI does,
    every ceil(32768 / rows) points and again from each step's start, and
    runs chat's share at every point."""
    from gmlx.serve import gpu_share

    hit, chat, order = [], [], []
    generate, lay = _sampling(monkeypatch, size, frames, text, _Points(),
                              hit)
    assert lay.rows == rows
    hold = _clip_part_stubs(monkeypatch, order, generate)
    monkeypatch.setattr(gpu_share, "pacer", lambda: chat.append)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    vj._clip_part(job, store, hold, store.clip(job.id))
    each = list(range(every - 1, 50, every))
    assert hit == each + each
    assert len(chat) == 2 * 50


def test_a_job_stops_at_its_next_point_between_two_evaluations(
        tmp_path, monkeypatch):
    """At 2054 rows the sampler evaluates every 16th point, and a stop
    set after point 3 stops the job at point 4."""
    from gmlx.serve.image_service import JobStopped

    hit, order = [], []
    dit = _Points(at=lambda i: i == 3 and job.stop.set())
    generate, lay = _sampling(monkeypatch, (512, 512), 22, 188, dit, hit)
    assert lay.rows == 2054
    hold = _clip_part_stubs(monkeypatch, order, generate)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    with pytest.raises(JobStopped):
        vj._clip_part(job, store, hold, store.clip(job.id))
    assert dit.last == 4 and hit == []


def test_the_decode_paces_and_stops_between_its_pieces(tmp_path,
                                                       monkeypatch):
    """Chat's share and the stop run after each decoded piece, and the
    progress counts the frames."""
    from gmlx.serve.image_service import JobStopped
    from gmlx.video import run

    order = []
    hold = _clip_part_stubs(monkeypatch, order, lambda *a, **k: (
        __import__("mlx.core").core.zeros((1,)),) * 2)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    seen = []

    def decode(*a, between, **k):
        between(11)
        seen.append(job.record["gmlx"]["frames_decoded"])
        job.stop.set()
        between(22)

    monkeypatch.setattr(run, "decode_and_mux", decode)
    with pytest.raises(JobStopped):
        vj._clip_part(job, store, hold, store.clip(job.id))
    assert seen == [11]
    assert order[-1] == ("settle", vj.JOB)


# An ended job's memory, and a job past its expiry.

@pytest.mark.parametrize("end", ["completed", "failed", "cancelled"])
def test_an_ended_job_keeps_no_features_or_latents(tmp_path, end):
    """The encode's features and the anchors' latents go when the job
    ends, however it ends: the job stays in the store to its expiry."""
    from gmlx.serve.image_service import JobStopped

    def run(job, store):
        job.kept.update(feats=b"f" * 64, anchors=((0, b"v", None),))
        if end == "failed":
            raise vj.JobFailed(400, "a clip does not fit")
        if end == "cancelled":
            job.stop.set()
            raise JobStopped(timed_out=False)
        _clip(job, store)

    store, clock, driver = _store(tmp_path, run)
    job = _submit(store)
    if end == "cancelled":
        store._run = lambda j, s: (store.delete(j.id), run(j, s))
    driver.run()
    assert store.get(job.id)["status"] == end
    assert job.kept == {}


def test_a_job_still_queued_at_its_expiry_never_runs(tmp_path):
    ran = []
    store, clock, driver = _store(tmp_path, lambda j, s: ran.append(j),
                                  job_ttl_s=60)
    job = _submit(store)
    clock.now += 60
    driver.run()
    assert ran == []
    rec = store.get(job.id)
    assert rec is not None and rec["status"] == "failed"
    assert rec["expires_at"] == rec["completed_at"] + 60
    assert rec["error"] == {
        "code": "timed_out", "message": "the job was still queued when it "
        "expired; submit it again, or raise server.video.job_ttl_s",
        "type": "server_error", "status": 503}
    assert not os.path.exists(store.inputs(job.id))


# A POST's reservation in the store.

def _start(conn, head: bytes, length: int, ctype: str) -> None:
    """Send the headers of a POST to /v1/videos and the first ``head``
    bytes of its body."""
    conn.putrequest("POST", "/v1/videos")
    conn.putheader("Content-Type", ctype)
    conn.putheader("Content-Length", str(length))
    conn.endheaders(head)


def _until(cond, timeout=10.0):
    import time

    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


def test_a_sweep_while_a_form_arrives_leaves_its_files(routes,
                                                      monkeypatch):
    """A sweep runs from every other POST and after every job: it leaves
    the inputs of a POST still arriving, before its job exists. Once read,
    the POST holds the bytes it wrote."""
    import http.client

    from live_socket import serving

    png = _png(48, 48)
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": png})
    cut = body.index(png) + len(png) // 2
    store = routes.store
    settled = []
    real = store.settle

    def settle(jid):
        real(jid)
        settled.append((store.pending[jid],
                        store._bytes(store.inputs(jid))))
    monkeypatch.setattr(store, "settle", settle)
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(conn, body[:cut], len(body), ctype)
        assert _until(lambda: store.pending and os.path.exists(os.path.join(
            store.inputs(next(iter(store.pending))), "first_frame.part")))
        jid = next(iter(store.pending))
        assert store.pending[jid] == len(body)
        store.sweep()
        conn.send(body[cut:])
        r = conn.getresponse()
        text = r.read()
        assert r.status == 200, text
        conn.close()
    assert json.loads(text)["id"] == jid
    with open(os.path.join(store.inputs(jid), "first_frame"), "rb") as f:
        assert f.read() == png
    assert settled == [(len(png), len(png))]
    assert store.pending == {}


def test_two_forms_that_each_fit_alone_cannot_pass_the_room_together(
        routes):
    """The first POST reserves its Content-Length before its first byte:
    the second, which fits alone, is refused beside it on its headers,
    before any of its body is sent."""
    import http.client

    from live_socket import send, serving

    store = routes.store
    store.settings = VideoServerCfg(store_max_bytes=1 << 20)
    pad = b"\0" * (600 << 10)
    a, ctype = _form({"prompt": "a cat"}, {"first_frame": _png() + pad})
    b, _ = _form({"prompt": "a dog"}, {"first_frame": _png() + pad})
    with serving(routes.app) as port:
        ca = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(ca, a[:64], len(a), ctype)
        assert _until(lambda: list(store.pending.values()) == [len(a)])
        cb = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        r = send(cb, "/v1/videos", b"", length=len(b), ctype=ctype)
        assert r.status == 413, r.body
        assert json.loads(r.body)["error"]["message"] == (
            f"the files of this job are {body_intake._mib(len(b), True)}, "
            "and the video job store has "
            f"{body_intake._mib((1 << 20) - len(a))} of room left, and "
            f"{body_intake._mib(len(a))} of it is held by uploads arriving "
            "now; send smaller files, delete finished jobs, or raise "
            "server.video.store_max_bytes")
        cb.close()
        ca.send(a[64:])
        r = ca.getresponse()
        assert r.status == 200, r.read()
        r.read()
        ca.close()
    assert store.pending == {}
    assert len(store.jobs) == 1
    assert store.room() == (1 << 20) - len(_png() + pad)


def _body_pool(monkeypatch, free: int) -> SimpleNamespace:
    """A pool with ``free`` bytes of its budget free, and none admitted."""
    from gmlx.serve.patches import _common

    pool = SimpleNamespace(figures=SimpleNamespace(
        budget=free, resident=0, incoming=0))
    monkeypatch.setattr(_common, "_get_pool", lambda: pool)
    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    return pool


def _json_job(prompt: str, pad: int) -> bytes:
    """A JSON job whose first frame, a data URL, is ``pad`` bytes long."""
    url = "data:image/png;base64," + base64.b64encode(
        _png() + b"\0" * pad).decode()
    return json.dumps({"prompt": prompt, "first_frame": url}).encode()


def test_two_json_jobs_that_each_fit_alone_cannot_hold_the_memory_together(
        routes, monkeypatch):
    """A JSON body counts what its read holds beside the other bodies
    being read from before its first byte: the second of two bodies that
    each fit the memory free alone, but not beside each other, is a 503
    on its headers, naming what the first holds. The pool is never
    booked."""
    import http.client

    from live_socket import send, serving

    monkeypatch.setattr(body_intake, "STALL_S", 5.0)
    a = _json_job("a cat", 3 << 20)
    b = _json_job("a dog", 3 << 20)
    # The first fits its parse; the second fits alone, not beside it.
    need = body_intake.json_price(a)
    assert body_intake.json_price(b) <= need < 4 * len(a)
    pool = _body_pool(monkeypatch, need)
    ctype = "application/json"
    with serving(routes.app) as port:
        ca = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(ca, a[:64], len(a), ctype)
        assert _until(lambda: bool(routes.store.pending))
        counted = _until(
            lambda: body_intake.free_bytes() == need - 2 * len(a))
        cb = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        r = send(cb, "/v1/videos", b"", length=len(b), ctype=ctype)
        cb.close()
        assert r.status == 503, r.body
        assert int(r.getheader("Retry-After")) > 0
        assert json.loads(r.body)["error"]["message"] == (
            f"the body of this request holds "
            f"{body_intake._mib(2 * len(b), True)} as it is read and "
            f"parsed, with {body_intake._mib(need)} of memory free beside "
            f"the models loaded, and {body_intake._mib(2 * len(a))} of it is "
            "held by request bodies arriving now; send it again once they "
            "are read")
        assert counted
        assert pool.figures.incoming == 0
        ca.send(a[64:])
        r = ca.getresponse()
        assert r.status == 200, r.read()
        r.read()
        ca.close()
        assert _until(lambda: body_intake.held_bytes() == 0)
        cb = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        r = send(cb, "/v1/videos", b)
        cb.close()
        assert r.status == 200, r.body
    assert body_intake.held_bytes() == 0
    assert pool.figures.incoming == 0
    assert len(routes.store.jobs) == 2


@pytest.mark.parametrize("ending", ["bad json", "empty first_frame",
                                    "client leaves", "stall"])
def test_a_json_bodys_count_ends_however_its_request_ends(
        routes, monkeypatch, ending):
    """The count a JSON body holds ends on every exit of its request: a
    parse refusal, the route's own 400, a client that leaves mid-body and
    a body that stops sending. The pool is never booked."""
    import http.client

    from live_socket import send, serving

    monkeypatch.setattr(body_intake, "STALL_S", 0.5)
    pool = _body_pool(monkeypatch, 64 << 20)
    body = {"bad json": b'{"prompt": "a cat",' + b" " * (1 << 20),
            "empty first_frame": json.dumps(
                {"prompt": "a cat", "first_frame": ""}).encode(),
            "client leaves": _json_job("a cat", 1 << 20),
            "stall": _json_job("a cat", 1 << 20)}[ending]
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        if ending in ("bad json", "empty first_frame"):
            r = send(conn, "/v1/videos", body)
            assert r.status == 400, r.body
        else:
            _start(conn, body[:64], len(body), "application/json")
            assert _until(
                lambda: body_intake.held_bytes() == 2 * len(body))
            assert pool.figures.incoming == 0
            if ending == "client leaves":
                conn.sock.close()
            else:
                r = conn.getresponse()
                assert r.status == 408, r.read()
        conn.close()
        assert _until(lambda: body_intake.held_bytes() == 0)
    assert pool.figures.incoming == 0
    assert routes.store.jobs == {}


def test_two_video_forms_whose_text_fits_alone_cannot_hold_it_together(
        routes, monkeypatch):
    """A form's text fields count what they hold beside the other bodies
    being read as they arrive: a second form whose text fits the memory
    free alone, but not beside the first's, is a 503 naming what the first
    holds, and runs once the first is read. The pool is never booked."""
    import http.client

    from live_socket import serving

    text = "a" * (1 << 20)
    a, ctype = _form({"prompt": text}, {"first_frame": _png()})
    b, _ = _form({"prompt": text}, {"first_frame": _png()})
    # What a text field holds once all of it arrived, before its end.
    held = math.ceil(len(text) * 1.125) + len(text) + body_intake._SLACK
    pool = _body_pool(monkeypatch, held + (64 << 10))
    head = a[:a.index(text.encode()) + len(text)]
    with serving(routes.app) as port:
        ca = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(ca, head, len(a), ctype)
        counted = _until(lambda: body_intake.held_bytes() >= held - 64)
        first = body_intake.held_bytes()
        r = routes.client.post("/v1/videos", content=b,
                               headers={"content-type": ctype})
        assert r.status_code == 503, r.text
        assert int(r.headers["retry-after"]) > 0
        message = r.json()["error"]["message"]
        assert message.startswith("the body of this request holds "), message
        assert message.endswith(
            f", and {body_intake._mib(first)} of it is held by request bodies "
            "arriving now; send it again once they are read"), message
        assert counted
        assert pool.figures.incoming == 0
        ca.send(a[len(head):])
        r = ca.getresponse()
        assert r.status == 200, r.read()
        r.read()
        ca.close()
        assert _until(lambda: body_intake.held_bytes() == 0)
    r = routes.client.post("/v1/videos", content=b,
                           headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    assert body_intake.held_bytes() == 0
    assert pool.figures.incoming == 0
    assert len(routes.store.jobs) == 2


@pytest.mark.parametrize("ending", ["bad form", "empty first_frame",
                                    "guides 413", "client leaves", "stall"])
def test_a_video_forms_count_ends_however_its_request_ends(
        routes, monkeypatch, ending):
    """The count a form's text holds ends on every exit of its request: a
    form that cannot be read, the route's own 400, a guides parse past the
    memory free, a client that leaves mid-text and a form that stops
    sending. The pool is never booked."""
    import http.client

    from live_socket import send, serving

    monkeypatch.setattr(body_intake, "STALL_S", 0.5)
    pool = _body_pool(monkeypatch, 4 << 20)
    text = "a" * (1 << 20)
    if ending == "bad form":
        body, ctype = b"--gmlxboundary\r\nno headers\r\n\r\n", \
            "multipart/form-data; boundary=gmlxboundary"
    elif ending == "empty first_frame":
        body, ctype = _form({"prompt": text, "first_frame": ""}, {})
    elif ending == "guides 413":
        body, ctype = _form({"prompt": "a cat", "guides": "[" + ",".join(
            ["{}"] * 50_000) + "]"}, {"guide_0": _png(64, 64)})
    else:
        body, ctype = _form({"prompt": text}, {"first_frame": _png()})
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        if ending in ("bad form", "empty first_frame", "guides 413"):
            r = send(conn, "/v1/videos", body, ctype=ctype)
            assert r.status == (413 if ending == "guides 413" else 400), \
                r.body
        else:
            head = body[:body.index(text.encode()) + len(text)]
            _start(conn, head, len(body), ctype)
            assert _until(lambda: body_intake.held_bytes() > len(text))
            assert pool.figures.incoming == 0
            if ending == "client leaves":
                conn.sock.close()
            else:
                r = conn.getresponse()
                assert r.status == 408, r.read()
        conn.close()
        assert _until(lambda: body_intake.held_bytes() == 0)
    assert pool.figures.incoming == 0
    assert routes.store.jobs == {}


def test_a_form_whose_client_leaves_frees_its_reservation(routes):
    import http.client

    from live_socket import serving

    store = routes.store
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": _png()})
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(conn, body[:64], len(body), ctype)
        assert _until(lambda: bool(store.pending))
        conn.sock.close()
        conn.close()
        assert _until(lambda: store.pending == {})
    assert store.jobs == {}
    assert os.listdir(os.path.join(str(store.root), "inputs")) == []


def _chunked(conn, ctype: str) -> None:
    """Send the headers of a POST to /v1/videos with no Content-Length."""
    conn.putrequest("POST", "/v1/videos")
    conn.putheader("Content-Type", ctype)
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()


def _chunk(conn, data: bytes) -> None:
    conn.send(b"%x\r\n" % len(data) + data + b"\r\n")


@pytest.mark.parametrize("pad_kib, fits", [(240, True), (560, False)])
def test_two_forms_with_no_length_reserve_the_room_as_they_arrive(
        routes, monkeypatch, pad_kib, fits):
    """A form with no Content-Length holds a step past the bytes it read,
    not the whole room: a second form beside it is refused only when the
    two together pass the room, and the refusal names what the first
    holds."""
    import http.client

    from live_socket import serving

    from gmlx.serve import image_steps

    step = 64 << 10
    monkeypatch.setattr(image_steps, "BODY_STEP", step)
    store = routes.store
    store.settings = VideoServerCfg(store_max_bytes=1 << 20)
    pad = b"\0" * (pad_kib << 10)
    a, ctype = _form({"prompt": "a cat"}, {"first_frame": _png() + pad})
    b, _ = _form({"prompt": "a dog"}, {"first_frame": _png() + pad})
    cut = len(a) - (16 << 10)
    with serving(routes.app) as port:
        ca = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _chunked(ca, ctype)
        _chunk(ca, a[:cut])
        assert _until(lambda: list(store.pending.values())
                      and next(iter(store.pending.values())) >= cut)
        held = next(iter(store.pending.values()))
        assert held < cut + step
        cb = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _chunked(cb, ctype)
        left = (1 << 20) - held
        if fits:
            _chunk(cb, b)
            cb.send(b"0\r\n\r\n")
        else:
            _chunk(cb, b[:left + step])
        r = cb.getresponse()
        text = r.read()
        cb.close()
        if fits:
            assert r.status == 200, text
        else:
            assert r.status == 413, text
            assert json.loads(text)["error"]["message"] == (
                f"the files of this job are over the "
                f"{body_intake._mib(left)} of room the video job store has "
                f"left, and {body_intake._mib(held)} of it is held by "
                "uploads arriving now; send smaller files, delete finished "
                "jobs, or raise server.video.store_max_bytes")
        _chunk(ca, a[cut:])
        ca.send(b"0\r\n\r\n")
        r = ca.getresponse()
        text = r.read()
        assert r.status == 200, text
        ca.close()
    assert store.pending == {}
    assert len(store.jobs) == (2 if fits else 1)


def _create(body: bytes, ctype: str = "application/json"):
    """POST /v1/videos as the route sees it, with no server."""
    import asyncio

    from starlette.requests import Request

    scope = {"type": "http", "method": "POST", "path": "/v1/videos",
             "query_string": b"", "headers": [
                 (b"content-type", ctype.encode()),
                 (b"content-length", str(len(body)).encode())]}
    sent = [False]

    async def receive():
        if not sent[0]:
            sent[0] = True
            return {"type": "http.request", "body": body,
                    "more_body": False}
        await asyncio.Event().wait()
    return videos.create(Request(scope, receive))


def _hold(store, monkeypatch, name):
    """Hold the store's ``name`` in its thread until the test lets it go:
    ``(entered, release)``."""
    entered, release = threading.Event(), threading.Event()
    real = getattr(store, name)

    def held(*a):
        entered.set()
        assert release.wait(10)
        return real(*a)
    monkeypatch.setattr(store, name, held)
    return entered, release


def _cancel_while_held(store, monkeypatch, name, body):
    """Cancel a POST while the store's ``name`` runs in its thread, then let
    it end, and wait for the abandon the cancel hands the executor."""
    import asyncio

    entered, release = _hold(store, monkeypatch, name)
    abandoned = threading.Event()
    real = store.abandon

    def abandon(jid):
        real(jid)
        abandoned.set()
    monkeypatch.setattr(store, "abandon", abandon)

    async def main():
        task = asyncio.ensure_future(_create(body))
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(abandoned.wait, 10)
    asyncio.run(main())


def test_a_cancel_while_submit_runs_keeps_the_job_it_queued(routes,
                                                             monkeypatch):
    """The cancel's abandon runs after the submit in its thread, and leaves
    the job it queued with its inputs."""
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    store = routes.store
    _cancel_while_held(store, monkeypatch, "submit", json.dumps(
        {"prompt": "a cat", "first_frame": url}).encode())
    (job,) = store.jobs.values()
    assert job.record["status"] == "queued"
    with open(job.params["first_frame"], "rb") as f:
        assert f.read() == _png()
    assert store.pending == {}


def test_a_cancel_while_begin_runs_leaves_nothing_pending(routes,
                                                          monkeypatch):
    """A begin still in its thread when the request was cancelled ends
    before the abandon, so nothing stays reserved."""
    store = routes.store
    _cancel_while_held(store, monkeypatch, "begin",
                       json.dumps({"prompt": "a cat"}).encode())
    assert store.pending == {}
    assert store.jobs == {}


def test_a_posts_store_calls_after_its_abandon_do_nothing(tmp_path):
    """Whatever order the executor runs them in, a begin, take, settle or
    submit that comes after a POST's abandon leaves nothing behind."""
    store, clock, driver = _store(tmp_path)
    post = videos._Post(store, store.new_id())
    post.abandon()
    post.begin()
    assert post.take(4096) == (False, 0, 0)
    post.settle()
    assert post.submit({"model": "h3", "gmlx": {"frames": 22}}, {}) is None
    assert (store.pending, store.jobs, driver.jobs) == ({}, {}, [])


def test_a_take_after_the_abandon_reserves_nothing(tmp_path):
    store, clock, driver = _store(tmp_path)
    jid = store.new_id()
    store.begin(jid)
    store.abandon(jid)
    assert store.take(jid, 4096) == (False, 0, 0)
    assert store.take(jid, None) == (False, 0, 0)
    assert store.pending == {}


def test_a_record_write_that_fails_queues_no_job(tmp_path, monkeypatch):
    store, clock, driver = _store(tmp_path)
    jid = store.new_id()
    store.begin(jid)

    def write(job):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(store, "_write", write)
    with pytest.raises(OSError):
        store.submit(jid, {"model": "h3", "gmlx": {"frames": 22}}, {})
    assert (store.jobs, store.queued(), driver.jobs) == ({}, 0, [])
    store.abandon(jid)
    assert store.pending == {}


# The route's refusals of what a form or a JSON body sends.

def _bomb_png(w=14000, h=14000) -> bytes:
    """The header of a 1-bit PNG of ``w`` x ``h``, a few dozen bytes,
    past Pillow's decompression bound."""
    import struct
    import zlib

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(
        ">IIBBBBB", w, h, 1, 0, 0, 0, 0)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("field", ["first_frame", "input_reference"])
@pytest.mark.parametrize("how", ["form", "json"])
def test_a_keyframe_past_the_decompression_bound_is_a_400(routes, how,
                                                          field):
    """The refusal names the field the client sent."""
    from gmlx.serve.patches.media_gate import MEDIA_MAX_PIXELS

    if how == "form":
        body, ctype = _form({"prompt": "a cat"}, {field: _bomb_png()})
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": ctype})
    else:
        url = "data:image/png;base64," + base64.b64encode(
            _bomb_png()).decode()
        r = routes.client.post("/v1/videos", json={
            "prompt": "a cat", field: url})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["param"], err["message"]) == (
        field, f"{field} is over the {MEDIA_MAX_PIXELS} pixels the server "
        "decodes")
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []
    assert routes.store.pending == {}


def test_a_guide_past_the_decompression_bound_names_its_file_field(routes):
    from gmlx.serve.patches.media_gate import MEDIA_MAX_PIXELS

    body, ctype = _form({"prompt": "a cat", "guides": json.dumps(
        [{"file": "guide_0", "frame": 0}])}, {"guide_0": _bomb_png()})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["param"], err["message"]) == (
        "guide_0", f"guide_0 is over the {MEDIA_MAX_PIXELS} pixels the "
        "server decodes")


@pytest.mark.parametrize("field", ["first_frame", "input_reference"])
def test_a_keyframe_in_a_format_off_the_list_is_no_image(routes, field):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, "PPM")
    body, ctype = _form({"prompt": "a cat"}, {field: buf.getvalue()})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["param"], err["message"]) == (
        field, f"{field} is no image the server reads")


def test_a_form_the_parser_cannot_read_is_a_400(routes, monkeypatch):
    """Starlette maps python-multipart's errors to its own; any other
    error of the parse is the same 400, never a 500."""
    from starlette.formparsers import MultiPartParser

    async def parse(self):
        raise ValueError("Invalid boundary")

    monkeypatch.setattr(MultiPartParser, "parse", parse)
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": _png()})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["message"] == (
        "the multipart form cannot be read: Invalid boundary")
    assert routes.store.pending == {}


@pytest.mark.parametrize("fields", [
    {"prompt": "a" * (400 << 10), "x1": "b" * (400 << 10),
     "x2": "c" * (400 << 10)},
    {"prompt": "a" * (2 << 20)}])
def test_a_forms_text_past_the_free_figure_is_a_413(routes, monkeypatch,
                                                     fields):
    """The text fields stay in RAM: their sum is priced against the
    memory free, and one field past it is the same 413."""
    monkeypatch.setattr(body_intake, "_pool_free", lambda: 1 << 20)
    body, ctype = _form(fields, {"first_frame": _png()})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 413, r.text
    msg = r.json()["error"]["message"]
    assert msg.startswith("the text fields of this form are over "), msg
    assert msg.endswith("and 1.0 MiB of memory is free beside the models "
                        "loaded; send shorter text fields, or unload a model "
                        "to make room")
    assert r.headers["connection"] == "close"
    assert routes.store.pending == {}


def _noise_png(side=64) -> bytes:
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(
        0, 256, (side, side, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_a_form_with_a_file_and_a_data_url_keyframe_runs(routes):
    """The keyframe takes its room beside the bytes the form wrote, not
    beside its whole length, whose text stays in RAM."""
    png = _noise_png()
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    body, ctype = _form({"prompt": "a cat", "last_frame": url},
                        {"first_frame": _png()})
    # The form fits, and the keyframe beside the form's length does not.
    routes.store.settings = VideoServerCfg(
        store_max_bytes=len(body) + len(png) // 2)
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    job = routes.store.jobs[r.json()["id"]]
    with open(job.params["last_frame"], "rb") as f:
        assert f.read() == png
    assert routes.store.room() == (len(body) + len(png) // 2
                                   - len(_png()) - len(png))


def test_a_json_keyframe_past_the_store_room_is_a_413(routes):
    png = _noise_png()
    routes.store.settings = VideoServerCfg(store_max_bytes=4096)
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "first_frame": url})
    assert r.status_code == 413, r.text
    err = r.json()["error"]
    assert (err["param"], err["message"]) == (
        "first_frame", f"the keyframe first_frame is "
        f"{body_intake._mib(len(png), True)}, and the video job store has "
        "4.0 KiB of room left; send a smaller image, delete finished jobs, "
        "or raise server.video.store_max_bytes")
    assert routes.store.pending == {}


def test_a_json_keyframe_refusal_names_the_uploads_arriving(routes):
    """Another POST still arriving holds part of the room: the keyframe's
    413 says how much."""
    png = _noise_png()
    store = routes.store
    store.settings = VideoServerCfg(store_max_bytes=len(png) + 4096)
    other = store.new_id()
    store.begin(other)
    assert store.take(other, len(png)) == (True, len(png), 0)
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    r = routes.client.post("/v1/videos", json={"prompt": "a cat",
                                               "first_frame": url})
    assert r.status_code == 413, r.text
    assert r.json()["error"]["message"] == (
        f"the keyframe first_frame is {body_intake._mib(len(png), True)}, "
        "and the video job store has 4.0 KiB of room left, and "
        f"{body_intake._mib(len(png))} of it is held by uploads arriving "
        "now; send a smaller image, delete finished jobs, or raise "
        "server.video.store_max_bytes")
    assert store.pending == {other: len(png)}


def test_a_form_keyframe_field_takes_no_path(routes, monkeypatch):
    """A form's text field where an image goes is a data URL: a path is
    refused with the field named, and never opened."""
    from PIL import Image

    opened = []
    real = Image.open
    monkeypatch.setattr(Image, "open", lambda fp, *a, **k: (
        opened.append(fp), real(fp, *a, **k))[1])
    body, ctype = _form({"prompt": "a cat", "first_frame": "/etc/passwd"},
                        {})
    r = routes.client.post("/v1/videos", content=body,
                           headers={"content-type": ctype})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["param"] == "first_frame"
    assert err["message"].startswith("first_frame is an image as a data URL")
    assert "/etc/passwd" not in opened


def test_the_post_names_the_fields_it_ignores(routes, caplog):
    import logging

    body, ctype = _form({"prompt": "a cat", "adapters": "[]"},
                        {"first_frame": _png(), "extra": b"x"})
    with caplog.at_level(logging.WARNING):
        r = routes.client.post("/v1/videos", content=body,
                               headers={"content-type": ctype})
        assert r.status_code == 200, r.text
        r = routes.client.post("/v1/videos", json={
            "prompt": "a cat", "adapters": [{"path": "x"}]})
        assert r.status_code == 200, r.text
    lines = [r.getMessage() for r in caplog.records
             if "ignoring unsupported" in r.getMessage()]
    assert lines == [
        "/v1/videos: ignoring unsupported parameter(s): adapters, extra "
        "(see the parameter support table in docs/api.md)",
        "/v1/videos: ignoring unsupported parameter(s): adapters "
        "(see the parameter support table in docs/api.md)"]


def test_the_store_lock_is_taken_off_the_event_loop(routes, monkeypatch):
    """The store's lock is held across a sweep's walk of the disk: every
    route takes it on a worker thread, never on the loop, a client that
    leaves included."""
    import asyncio
    import http.client

    from live_socket import serving

    store = routes.store
    on_loop = []

    def spy(name):
        real = getattr(store, name)

        def fn(*a, **k):
            try:
                asyncio.get_running_loop()
                on_loop.append(name)
            except RuntimeError:
                pass
            return real(*a, **k)
        monkeypatch.setattr(store, name, fn)

    for name in ("submit", "get", "content", "delete", "begin", "take",
                 "settle", "full", "retry_after", "abandon"):
        spy(name)
    c = routes.client
    body, ctype = _form({"prompt": "a cat"}, {"first_frame": _png()})
    r = c.post("/v1/videos", content=body, headers={"content-type": ctype})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    r = c.post("/v1/videos", json={"prompt": ""})
    assert r.status_code == 400, r.text
    routes.driver.run()
    assert c.get(f"/v1/videos/{jid}").json()["status"] == "completed"
    assert c.get(f"/v1/videos/{jid}/content").content == b"mp4"
    assert c.delete(f"/v1/videos/{jid}").json()["deleted"] is True
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        _start(conn, body[:64], len(body), ctype)
        assert _until(lambda: bool(store.pending))
        conn.sock.close()
        conn.close()
        assert _until(lambda: store.pending == {})
    assert on_loop == []


# The job's parts on the video worker.

def test_the_clip_samples_with_the_jobs_settings_and_anchors(
        tmp_path, monkeypatch):
    import mlx.core as mx
    import numpy as np

    order = []
    seen = {}

    def generate(pipe, feats, **k):
        seen.update(k)
        return mx.zeros((1,)), mx.zeros((1,))

    hold = _clip_part_stubs(monkeypatch, order, generate)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    video = np.ones((1, 4, 1, 2, 2), np.float32)
    job.kept.update(anchors=((0, video, None),), spans=((3, 9),))
    job.params["settings"].update(seed=1234, shifts={"video": 3.0})
    vj._clip_part(job, store, hold, store.clip(job.id))
    anchors = seen.pop("anchors")
    assert len(anchors) == 1, f"the sampler got anchors {anchors!r}"
    anchor = anchors[0]
    assert anchor.frame == 0 and anchor.audio is None
    np.testing.assert_array_equal(np.array(anchor.video), video)
    seen.pop("on_step"), seen.pop("pacer")
    assert seen == {"frames": 22, "width": 64, "height": 64, "steps": 2,
                    "shifts": {"video": 3.0}, "seed": 1234, "table": "t0",
                    "spans": ((3, 9),)}


def test_the_prompt_encode_settles_its_turn_when_it_raises(tmp_path,
                                                           monkeypatch):
    from gmlx.serve import image_steps, video_service
    from gmlx.serve.patches import image_steps as pis

    settled = []
    monkeypatch.setattr(pis, "_lora_pool", lambda: object())
    monkeypatch.setattr(image_service, "lora_settle", lambda p, e, subject:
                        settled.append(subject))
    monkeypatch.setattr(pis, "_check_encode", lambda *a: None)
    monkeypatch.setattr(video_service, "keyframe_refusal", lambda *a: None)
    monkeypatch.setattr(video_service, "anchor_refusal", lambda *a: None)

    def encode(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(video_service, "encode", encode)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    hold = SimpleNamespace(_entry=SimpleNamespace(pipeline=object()))
    with pytest.raises(RuntimeError):
        vj._encode_part(job, store, hold, "a cat", [], [])
    assert settled == [image_steps.ENCODE]


def test_a_guide_video_past_the_pixel_limit_is_refused_from_its_header(
        tmp_path, monkeypatch):
    """The frames of a guide video are read at their own size: one past
    the limit is refused before any of them is read."""
    from gmlx.serve.patches.media_gate import MEDIA_MAX_PIXELS
    from gmlx.video import io as vio
    from gmlx.video import run
    from gmlx.video.registry import get_family

    path = tmp_path / "guide_0"
    path.write_bytes(b"\0" * 64)

    def unread(*a, **k):
        raise AssertionError("a frame of the guide was read")

    monkeypatch.setattr(vio, "video_size", lambda p, ffmpeg: (9000, 9000))
    for name in ("guide_length", "guide_audio", "each_guide_frame"):
        monkeypatch.setattr(vio, name, unread)
    with pytest.raises(ValueError) as ei:
        run.read_anchors(get_family("h3"), {"frames": 22, "width": 32,
                                            "height": 32},
                         guides=[("--guide", str(path), 0,
                                  "guides[0] (guide_0)")],
                         ffmpeg="ffmpeg", paths=False,
                         max_pixels=MEDIA_MAX_PIXELS)
    assert str(ei.value) == (
        f"guides[0] (guide_0): its frames are 9000x9000, over the "
        f"{MEDIA_MAX_PIXELS} pixels the server decodes")


def test_a_jobs_keyframe_is_named_by_its_field(tmp_path):
    from gmlx.video import run
    from gmlx.video.registry import get_family

    path = tmp_path / "first_frame"
    path.write_bytes(_png())
    keyframes, todo = run.read_anchors(
        get_family("h3"), {"frames": 22, "width": 32, "height": 32},
        first=str(path), names=("first_frame", "last_frame"), paths=False)
    assert [t.label for t in todo] == ["first_frame"]
