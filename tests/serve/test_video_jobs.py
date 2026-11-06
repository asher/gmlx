"""gmlx/serve/video_jobs.py and patches/videos.py: the video job store,
its driver and the /v1/videos routes, on a fake clock with stub work, so
no model loads."""
from __future__ import annotations

import base64
import io
import json
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
        _clip(job, store)

    store, clock, driver = _store(tmp_path, run)
    job = _submit(store)
    rec = store.get(job.id)
    assert (rec["status"], rec["object"], rec["progress"]) == (
        "queued", "video", 0)
    assert rec["expires_at"] == rec["created_at"] + 86400
    assert _on_disk(store, job.id)["status"] == "queued"
    driver.run()
    assert seen == ["in_progress"]
    rec = store.get(job.id)
    assert (rec["status"], rec["progress"], rec["error"]) == (
        "completed", 100, None)
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
    again = vj.VideoJobs(store.root, time_fn=clock, mono_fn=clock,
                         driver=Driver())
    again.configure(SimpleNamespace(video=VideoServerCfg()))
    rec = again.get(queued.id)
    assert rec["status"] == "failed"
    assert rec["error"]["code"] == "server_restarted"
    assert again.get(done.id)["status"] == "completed"
    assert os.listdir(os.path.join(str(store.root), "files")) == [
        f"{done.id}.mp4"]
    assert not os.path.exists(again.inputs(queued.id))


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
        family="h3", variant="fl2va"))
    monkeypatch.setattr(media_programs, "find", lambda n: "/opt/ffmpeg")
    app = FastAPI()
    cfg = SimpleNamespace(video_models={"h3": {}}, video=VideoServerCfg())
    videos.install_video_routes(cfg, app=app)
    media_gate.install_media_gate(app=app)
    return SimpleNamespace(app=app, store=store, clock=clock, driver=driver,
                           client=TestClient(app), spec=spec)


def test_a_json_job_runs_to_its_mp4_through_the_routes(routes):
    c = routes.client
    r = c.post("/v1/videos", json={"prompt": "a cat", "seconds": "1",
                                   "size": "768x768", "seed": 7})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert (rec["status"], rec["model"], rec["size"], rec["seconds"]) == (
        "queued", "h3", "768x768", "1.62")
    assert rec["gmlx"]["frames"] == 39 and rec["gmlx"]["seed"] == 7
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
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def test_no_ffmpeg_is_a_501_with_the_missing_program_text(routes,
                                                         monkeypatch):
    from gmlx.serve import media_programs

    monkeypatch.setattr(media_programs, "find", lambda n: None)
    monkeypatch.setattr(media_programs, "problem", lambda n: (
        "the server cannot find ffmpeg. Install it with `brew install "
        "ffmpeg`, or start the server from a shell whose PATH holds your "
        "ffmpeg."))
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 501
    assert "brew install ffmpeg" in r.json()["error"]["message"]


def test_a_full_queue_is_a_429_with_its_retry_after(routes):
    routes.store.settings = VideoServerCfg(max_queue=1)
    assert routes.client.post("/v1/videos",
                              json={"prompt": "a"}).status_code == 200
    r = routes.client.post("/v1/videos", json={"prompt": "b"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == str(vj.FIRST_RETRY_S)


def test_post_takes_no_pool_lock_and_no_acquire(routes, monkeypatch):
    """The signed rule: the request thread books nothing; the job's
    memory is decided on the video worker."""
    from gmlx.serve import residency

    def refuse(*a, **k):
        raise AssertionError("the request thread acquired")

    monkeypatch.setattr(residency, "acquire_video_model", refuse)
    monkeypatch.setattr(residency, "_installed_pool", lambda: None)
    r = routes.client.post("/v1/videos", json={"prompt": "a cat"})
    assert r.status_code == 200, r.text


def test_the_estimate_refuses_a_clip_past_the_budget_beside_pinned_models(
        routes, monkeypatch):
    from gmlx.serve import residency, video_service

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
                                               "size": "768x768"})
    assert r.status_code == 413, r.text
    assert r.json()["error"]["message"].endswith(
        "does not fit in memory: denoise 9 GiB; use a smaller size")
    assert (seen["budget"], seen["text"]) == (40, 12)


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


def test_a_form_streams_its_files_to_the_jobs_inputs(routes):
    body, ctype = _form({"prompt": "a cat", "guides": json.dumps(
        [{"file": "guide_0", "frame": 0}])},
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
        "the files of this job are over 64.0 KiB")
    assert os.listdir(os.path.join(str(routes.store.root), "inputs")) == []


def test_a_json_body_past_the_free_figure_is_a_413_with_no_byte_read(
        routes, monkeypatch):
    import http.client

    from live_socket import closed, peer, send, serving

    monkeypatch.setattr(body_intake, "free_bytes", lambda: 1 << 20)
    with serving(routes.app) as port:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        other = peer(conn)
        r = send(conn, "/v1/videos", b"{", length=8 << 20)
        assert r.status == 413, r.body
        assert json.loads(r.body)["error"]["message"] == (
            "the request body is 8.0 MiB, and 1.0 MiB of memory is free "
            "beside the models loaded; send a smaller body, or unload a "
            "model to make room")
        assert closed(other)
        conn.close()


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
    monkeypatch.setattr(video_service, "ensure_dit", lambda s: order.append(
        "ensure_dit"))
    monkeypatch.setattr(pis, "_factor_turn", lambda p, e, s, core, subject:
                        order.append(("turn", core, subject)))
    monkeypatch.setattr(image_service, "lora_settle", lambda p, e, subject:
                        order.append(("settle", subject)))
    monkeypatch.setattr(pipeline, "generate", generate)
    monkeypatch.setattr(run, "decode_and_mux", lambda *a, **k: order.append(
        "decode"))
    monkeypatch.setattr(gpu_share, "pacer", lambda: None)
    served = SimpleNamespace(pipe=SimpleNamespace(default_table="t0"),
                             spec=SimpleNamespace(vae="v", audio_vae="a"))
    return SimpleNamespace(_entry=SimpleNamespace(pipeline=served))


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
        k["pacer"](())

    hold = _clip_part_stubs(monkeypatch, order, generate)
    store, clock, driver = _store(tmp_path)
    job = _clip_job(store)
    with pytest.raises(JobStopped):
        vj._clip_part(job, store, hold, store.clip(job.id))
    assert seen["pacer"] is not None
    assert job.record["gmlx"]["step"] == 1
    assert order[-1] == ("settle", vj.JOB)
