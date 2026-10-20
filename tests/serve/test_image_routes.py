#!/usr/bin/env python3
"""The OpenAI image routes: route replacement, field refusals, the key
exemption of the file route, URL and b64 answers, streaming frames, the
queue cap and the 499 and 504 answers. The model and the job are stubs, so
no model loads."""
from __future__ import annotations

import base64
import dataclasses
import importlib
import io
import json
import logging
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from fastapi.testclient import TestClient  # noqa: E402

import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.serve import image_service as svc  # noqa: E402
from gmlx.serve import residency  # noqa: E402
from gmlx.serve.patches import images  # noqa: E402

_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")

_DOC = {"image_models": {"zit": {"path": "/abs/zit.gguf",
                                 "encoder": "/abs/enc.gguf",
                                 "vae": "/abs/vae.gguf"}}}


def _png(w=32, h=32, alpha=False) -> bytes:
    import numpy as np
    from PIL import Image

    px = np.full((h, w, 4 if alpha else 3), 200, np.uint8)
    if alpha:
        px[: h // 2, :, 3] = 0
    buf = io.BytesIO()
    Image.fromarray(px, "RGBA" if alpha else "RGB").save(buf, format="PNG")
    return buf.getvalue()


class _Hold:
    def __init__(self, released):
        self._entry = SimpleNamespace(footprint=0, pipeline=_SERVED)
        self._pool = SimpleNamespace(stats=lambda: {"resident_bytes": 0})
        self.released = released

    def release(self):
        self.released.append(1)


_SERVED = SimpleNamespace(spec=SimpleNamespace(id="zit", encoder_model=None),
                          pipe=None)
_REAL_CHECK_FAMILY = svc.check_family


@pytest.fixture
def served(monkeypatch, tmp_path, request):
    """Image routes over a registered image model whose acquire and job
    are stubs. Returns a namespace with the recorded jobs and releases."""
    cfg = build_config(_DOC)
    serving.register_resolved_models(cfg)
    request.addfinalizer(serving.clear_resolved_models)
    rec = SimpleNamespace(jobs=[], released=[], job=None)

    def acquire(model):
        if model not in ("", "zit"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail={"error": {
                "type": "model_not_found", "message": "unknown"}})
        return _Hold(rec.released)

    def run_job(served, req, **kw):
        rec.jobs.append((req, kw))
        if rec.job is not None:
            return rec.job(req, kw)
        if kw["emit"] is not None:
            for i in range(req.partial_images):
                kw["emit"](i, b"partial%d" % i)
        return svc.ImageJobResult(
            images=[b"img%d" % i for i in range(req.n)], size=(64, 32),
            quality=req.quality, steps=4, seeds=list(range(req.n)),
            usage={"input_tokens": 3, "output_tokens": 8, "total_tokens": 11,
                   "input_tokens_details": {"text_tokens": 3,
                                            "image_tokens": 0}})

    monkeypatch.setattr(residency, "acquire_image_model", acquire)
    monkeypatch.setattr(svc, "run_job", run_job)
    monkeypatch.setattr(svc, "check_family", lambda req, s: None)
    monkeypatch.setattr(svc, "target_size", lambda req, s: (64, 32))
    monkeypatch.setattr(svc, "fit_check", lambda req, s, size, budget: None)
    monkeypatch.setattr(svc, "default_store_root", lambda port=None: str(tmp_path / "files"))
    monkeypatch.setattr(images, "_budget", lambda hold: None)
    images.install_image_routes(cfg)
    return rec


def _client():
    return TestClient(_APP.app, base_url="http://127.0.0.1")


def _endpoint_module(path, method="POST"):
    for r in _APP.app.router.routes:
        if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
            return r.endpoint.__module__
    return None


def test_the_stock_image_handlers_are_unreachable_without_an_image_model():
    images.install_image_routes(build_config({}))
    for route in images.ROUTES:
        for path in (f"/v1/images/{route}", f"/images/{route}"):
            assert _endpoint_module(path) == "gmlx.serve.patches.images", path
    r = _client().post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"
    assert "image_models" in r.json()["error"]["message"]


@pytest.mark.parametrize("body, param", [
    ({"prompt": "x", "output_path": "/tmp/x.png"}, "output_path"),
    ({"prompt": "x", "output_dir": "/tmp"}, "output_dir"),
    ({"prompt": "x", "response_format": "path"}, "response_format"),
    ({"prompt": "x", "stream": True, "n": 2}, "n"),
    ({"prompt": "x", "background": "transparent", "output_format": "jpeg"},
     "background"),
    ({"prompt": "x", "size": "big"}, "size"),
    ({"prompt": "x", "n": 11}, "n"),
    ({"prompt": "x", "partial_images": 4}, "partial_images"),
    ({"prompt": "x", "output_format": "gif"}, "output_format"),
    ({"prompt": "x", "adapters": [{"id": 3}]}, "adapters"),
    ({}, "prompt"),
])
def test_refused_fields_answer_400_naming_the_field(served, body, param):
    r = _client().post("/v1/images/generations", json=body)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == param
    assert served.jobs == []


def test_generations_answer_b64_by_default(served):
    r = _client().post("/images/generations",
                       json={"prompt": "a cat", "n": 2, "model": "dall-e-3",
                             "quality": "hd", "seed": 7})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [base64.b64decode(d["b64_json"]) for d in body["data"]] == [b"img0", b"img1"]
    assert body["size"] == "64x32" and body["quality"] == "high"
    assert body["usage"]["total_tokens"] == 11
    req, kw = served.jobs[0]
    assert req.model == "" and req.seed == 7 and kw["emit"] is None
    assert served.released == [1]


def test_url_answers_read_back_without_the_key_and_expire(served, monkeypatch):
    from gmlx.serve.patches import hardening

    hardening.install_api_key_auth("k")
    client = _client()
    key = {"Authorization": "Bearer k"}
    r = client.post("/v1/images/generations", headers=key,
                    json={"prompt": "a cat", "response_format": "url",
                          "output_format": "webp"})
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert url.startswith("http://127.0.0.1/v1/images/files/") and url.endswith(".webp")
    path = url.removeprefix("http://127.0.0.1")
    got = client.get(path)                       # no key
    assert got.status_code == 200 and got.content == b"img0"
    assert got.headers["content-type"] == "image/webp"
    assert client.get(path.removeprefix("/v1")).status_code == 200
    images.STATE.store.ttl_s = -1                # every file is past its time
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("method, path", [
    ("post", "/v1/images/generations"),
    ("get", "/v1/images/files/short.png"),
    ("get", "/v1/images/files/" + "a" * 43 + ".gif"),
    ("get", "/v1/images/files/" + "a" * 43 + ".png/x"),
    ("get", "/v1/images/filesx/" + "a" * 43 + ".png"),
    ("get", "/v1/images/files/../models"),
    ("get", "/v1/models"),
    ("post", "/v1/images/files/" + "a" * 43 + ".png"),
])
def test_the_key_exemption_covers_the_file_route_alone(served, method, path):
    from gmlx.serve.patches import hardening

    hardening.install_api_key_auth("k")
    r = getattr(_client(), method)(path)
    assert r.status_code == 401, (path, r.status_code)


def test_an_unknown_file_token_is_a_404_without_the_key(served):
    from gmlx.serve.patches import hardening

    hardening.install_api_key_auth("k")
    r = _client().get("/v1/images/files/" + "a" * 43 + ".png")
    assert r.status_code == 404


def test_streaming_sends_partials_then_the_completed_event(served):
    r = _client().post("/v1/images/generations",
                       json={"prompt": "a cat", "stream": True,
                             "partial_images": 2})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in r.text.split("\n\n") if f.strip()]
    events = [f.splitlines()[0].removeprefix("event: ") for f in frames]
    assert events == ["image_generation.partial_image"] * 2 + [
        "image_generation.completed"]
    datas = [json.loads(f.splitlines()[1].removeprefix("data: ")) for f in frames]
    assert [d["partial_image_index"] for d in datas[:2]] == [0, 1]
    assert all(d["type"] == e for d, e in zip(datas, events))
    assert base64.b64decode(datas[-1]["b64_json"]) == b"img0"
    assert datas[-1]["usage"]["output_tokens"] == 8
    assert "[DONE]" not in r.text


def test_an_edit_stream_uses_the_edit_events(served):
    r = _client().post("/v1/images/edits",
                       files={"image": ("a.png", _png(), "image/png")},
                       data={"prompt": "make it red", "stream": "true",
                             "partial_images": "1"})
    assert r.status_code == 200, r.text
    assert "event: image_edit.partial_image" in r.text
    assert "event: image_edit.completed" in r.text


def test_edits_take_several_multipart_images_and_a_mask(served):
    r = _client().post("/v1/images/edits", files=[
        ("image[]", ("a.png", _png(), "image/png")),
        ("image[]", ("b.png", _png(48, 16), "image/png")),
        ("mask", ("m.png", _png(alpha=True), "image/png"))],
        data={"prompt": "swap them", "n": "1", "size": "64x32"})
    assert r.status_code == 200, r.text
    req, _ = served.jobs[0]
    assert [i.shape for i in req.images] == [(32, 32, 3), (16, 48, 3)]
    assert req.mask.shape == (32, 32, 4) and req.size == (64, 32)


def test_a_json_edit_reads_data_urls_and_refuses_file_ids(served):
    url = "data:image/png;base64," + base64.b64encode(_png()).decode()
    r = _client().post("/v1/images/edits", json={
        "prompt": "x", "images": [{"image_url": url}]})
    assert r.status_code == 200, r.text
    assert served.jobs[0][0].images[0].shape == (32, 32, 3)
    r = _client().post("/v1/images/edits", json={
        "prompt": "x", "images": [{"file_id": "file-1"}]})
    assert r.status_code == 400 and "file_id" in r.text


def test_a_json_edit_never_reads_a_local_path(served, tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png())
    r = _client().post("/v1/images/edits", json={
        "prompt": "x", "images": [{"image_url": str(p)}]})
    assert r.status_code == 400 and served.jobs == []


def test_variations_need_no_prompt(served):
    r = _client().post("/v1/images/variations",
                       files={"image": ("a.png", _png(), "image/png")},
                       data={"n": "2", "response_format": "b64_json"})
    assert r.status_code == 200, r.text
    req, _ = served.jobs[0]
    assert req.route == "variations" and req.prompt == "" and req.n == 2


def test_a_generations_form_is_refused(served):
    r = _client().post("/v1/images/generations",
                       files={"image": ("a.png", _png(), "image/png")},
                       data={"prompt": "x"})
    assert r.status_code == 400


def _frames(text: str) -> list:
    out = []
    for f in (f for f in text.split("\n\n") if f.strip()):
        head, data = f.splitlines()[:2]
        out.append((head.removeprefix("event: "),
                    json.loads(data.removeprefix("data: "))))
    return out


@pytest.mark.parametrize("stream", [False, True])
def test_a_flux3_flow_map_with_guidance_is_a_400_before_any_stream(
        served, monkeypatch, stream):
    """check_family refuses guidance on a flow map DiT, so the streamed
    request gets the same 400 as the plain one, before the stream opens
    and before the job."""
    from gmlx.image.registry import get_family

    flux3 = SimpleNamespace(
        spec=SimpleNamespace(id="zit", encoder_model=None, defaults={}),
        pipe=SimpleNamespace(family=get_family("flux3_klein"),
                             variant="distilled",
                             dit=SimpleNamespace(cfg=SimpleNamespace(
                                 flow_map=True))),
        plan=SimpleNamespace(cfg=1.0))
    hold = _Hold(served.released)
    hold._entry = SimpleNamespace(footprint=0, pipeline=flux3)
    monkeypatch.setattr(residency, "acquire_image_model", lambda m: hold)
    monkeypatch.setattr(svc, "check_family", _REAL_CHECK_FAMILY)
    images.install_image_routes(build_config(_DOC))
    r = _client().post("/v1/images/generations", json={
        "prompt": "x", "guidance": 3.5, "stream": stream})
    assert r.status_code == 400, r.text
    assert r.headers["content-type"].startswith("application/json")
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "guidance"
    assert "flow map, which samples without guidance" in err["message"]
    assert served.jobs == [] and served.released == [1]


@pytest.mark.parametrize("raised, param", [
    ("refusal", None), ("adapter", None), ("memory", "size")])
def test_a_refusal_raised_in_a_streamed_job_is_the_plain_paths_400(
        served, monkeypatch, caplog, capsys, raised, param):
    """api.sample's own refusal, an adapter the request names that does not
    load, or a plan that does not fit, raised inside the job: the stream's
    error frame is the body the plain request gets as its 400, and the job
    is logged as refused, not failed."""
    from gmlx.image.lora.reader import UnmatchedKeysError
    from gmlx.image.memfit import ImageMemoryError
    from gmlx.image.refusal import RequestRefusal

    raised = {"refusal": RequestRefusal(
                  "the DiT is a flow map, which samples without guidance"),
              "adapter": UnmatchedKeysError("the adapter has 3 keys no "
                                            "layer of the DiT takes", ()),
              "memory": ImageMemoryError("the image needs 9 GiB of 4")}[raised]
    monkeypatch.setattr(images.STATE, "counts", dict.fromkeys(
        ("jobs", "failed", "cancelled", "images", "steps"), 0))
    served.job = lambda req, kw: (_ for _ in ()).throw(raised)
    plain = _client().post("/v1/images/generations", json={"prompt": "x"})
    assert plain.status_code == 400
    capsys.readouterr()
    with caplog.at_level(logging.INFO):
        r = _client().post("/v1/images/generations",
                           json={"prompt": "x", "stream": True})
    assert r.status_code == 200
    assert _frames(r.text) == [("error", plain.json())]
    assert plain.json()["error"] == {
        "type": "invalid_request_error", "message": str(raised),
        "param": param, "code": None}
    assert images.STATE.metrics()["jobs_failed_total"] == 0
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith("[req]")]
    assert len(line) == 1 and " refused " in line[0], line
    assert served.released == [1, 1]


def test_the_primitives_refuse_a_request_with_the_refusal_type():
    """The refusals a job can meet after the request checks are
    RequestRefusal, which the routes answer with a 400."""
    from gmlx.image import api, schedule
    from gmlx.image.refusal import RequestRefusal
    from gmlx.image.registry import get_family

    f3, f2 = get_family("flux3_klein"), get_family("flux2_klein")
    q = get_family("qwen_image21")
    for call in (lambda: api.check_sampling(f3, "distilled", True, 3.5),
                 lambda: api.check_sampling(f3, "distilled", True, 1.0,
                                            "uni_pc_bh2"),
                 lambda: api.check_guidance(f2, "distilled", 4.0),
                 lambda: api.check_steps(q, "turbo", 20),
                 lambda: schedule.qwen21_sigmas(1, 4096),
                 lambda: schedule.flux2_sigmas(0, 4096),
                 lambda: schedule.zimage_sigmas(0)):
        with pytest.raises(RequestRefusal):
            call()


def test_an_internal_value_error_in_a_job_is_a_500_with_its_trace(
        served, monkeypatch, caplog):
    """A ValueError that is no refusal, such as a shape mismatch inside a
    primitive, is the server's: a 500 on both paths, with its trace in the
    log, and the job counts as failed."""
    monkeypatch.setattr(images.STATE, "counts", dict.fromkeys(
        ("jobs", "failed", "cancelled", "images", "steps"), 0))
    served.job = lambda req, kw: (_ for _ in ()).throw(
        ValueError("latents must be (B, C, H, W), not (2, 3)"))
    with caplog.at_level(logging.INFO):
        plain = _client().post("/v1/images/generations", json={"prompt": "x"})
        r = _client().post("/v1/images/generations",
                           json={"prompt": "x", "stream": True})
    assert plain.status_code == 500 and r.status_code == 200
    assert plain.json()["error"]["type"] == "server_error"
    assert "ValueError" in plain.json()["error"]["message"]
    assert _frames(r.text) == [("error", plain.json())]
    assert "latents must be" not in plain.text + r.text
    traced = [rec for rec in caplog.records if rec.levelno >= logging.ERROR
              and rec.exc_info and rec.exc_info[0] is ValueError]
    assert len(traced) == 2, [rec.getMessage() for rec in caplog.records]
    assert images.STATE.metrics()["jobs_failed_total"] == 2


def test_a_failed_streamed_job_is_the_plain_paths_500(served, monkeypatch):
    monkeypatch.setattr(images.STATE, "counts", dict.fromkeys(
        ("jobs", "failed", "cancelled", "images", "steps"), 0))
    served.job = lambda req, kw: (_ for _ in ()).throw(RuntimeError("boom"))
    plain = _client().post("/v1/images/generations", json={"prompt": "x"})
    r = _client().post("/v1/images/generations",
                       json={"prompt": "x", "stream": True})
    assert plain.status_code == 500 and r.status_code == 200
    assert _frames(r.text) == [("error", plain.json())]
    assert plain.json()["error"]["type"] == "server_error"
    assert "boom" not in r.text
    assert images.STATE.metrics()["jobs_failed_total"] == 2


def test_a_request_error_raised_in_a_streamed_job_keeps_its_type(served):
    served.job = lambda req, kw: (_ for _ in ()).throw(svc._bad(
        "the shared encoder is not loaded", "model", err_type="server_error"))
    plain = _client().post("/v1/images/generations", json={"prompt": "x"})
    r = _client().post("/v1/images/generations",
                       json={"prompt": "x", "stream": True})
    assert _frames(r.text) == [("error", plain.json())]
    assert plain.json()["error"]["type"] == "server_error"
    assert plain.json()["error"]["param"] == "model"


def test_a_cancelled_job_answers_499_and_a_timeout_504(served):
    served.job = lambda req, kw: (_ for _ in ()).throw(svc.JobStopped(False))
    assert _client().post("/v1/images/generations",
                          json={"prompt": "x"}).status_code == 499
    served.job = lambda req, kw: (_ for _ in ()).throw(svc.JobStopped(True))
    r = _client().post("/v1/images/generations", json={"prompt": "x"})
    assert r.status_code == 504 and r.json()["error"]["type"] == "timeout"
    assert served.released == [1, 1]


def test_image_metrics_count_jobs_and_time_each_image_and_step(
        served, monkeypatch):
    monkeypatch.setattr(images.STATE, "counts", dict.fromkeys(
        ("jobs", "failed", "cancelled", "images", "steps"), 0))
    monkeypatch.setattr(images.STATE, "recent", [])

    def job(req, kw):
        return svc.ImageJobResult(
            images=[b"a", b"b"], size=(64, 32), quality="auto", usage={},
            seeds=[0, 1], steps=8, run_steps=8, image_s=6.0, step_s=4.0)

    served.job = job
    assert _client().post("/v1/images/generations",
                          json={"prompt": "x", "n": 2}).status_code == 200
    served.job = lambda req, kw: (_ for _ in ()).throw(svc.JobStopped(False))
    assert _client().post("/v1/images/generations",
                          json={"prompt": "x"}).status_code == 499
    served.job = lambda req, kw: (_ for _ in ()).throw(RuntimeError("boom"))
    assert _client().post("/v1/images/generations",
                          json={"prompt": "x"}).status_code == 500
    m = images.STATE.metrics()
    assert (m["jobs_total"], m["jobs_cancelled_total"], m["jobs_failed_total"],
            m["images_total"], m["steps_total"]) == (1, 1, 1, 2, 16)
    assert m["image_s"] == 3.0 and m["step_s"] == 0.25
    assert m["in_flight"] == 0 and m["waiting"] == 0
    assert m["max_queue"] == images.STATE.settings.max_queue


def test_the_queue_cap_answers_429_with_retry_after(served):
    gate, entered = threading.Event(), threading.Event()

    def slow(req, kw):
        entered.set()
        gate.wait(30)
        return svc.ImageJobResult(images=[b"x"], size=(64, 32), quality="auto",
                                  usage={}, seeds=[0], steps=1)

    served.job = slow
    images.STATE.settings = dataclasses.replace(images.STATE.settings,
                                                max_queue=0)
    client = _client()
    first = threading.Thread(target=lambda: client.post(
        "/v1/images/generations", json={"prompt": "x"}))
    first.start()
    assert entered.wait(30)
    r = client.post("/v1/images/generations", json={"prompt": "y"})
    gate.set()
    first.join(30)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 2


def _disconnecting_receive(body: bytes):
    """An ASGI receive that gives ``body``, then waits until ``leave()``,
    then answers every call with a disconnect, as uvicorn does.
    ``waiting`` is set once a call waits for the disconnect."""
    import asyncio

    state = {"sent": False, "event": asyncio.Event()}
    waiting = asyncio.Event()

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        waiting.set()
        await state["event"].wait()
        return {"type": "http.disconnect"}

    def leave():
        state["event"].set()

    return receive, leave, waiting


def test_a_disconnect_behind_the_http_middlewares_sets_the_stop_event():
    """The server stacks http middlewares, through which
    Request.is_disconnected never sees a disconnect; the watcher must."""
    import asyncio

    from fastapi import FastAPI
    from starlette.responses import JSONResponse

    app = FastAPI()

    @app.middleware("http")
    async def passthrough(request, call_next):
        return await call_next(request)

    seen = {}

    async def x(request):
        await request.body()
        stop = threading.Event()
        watcher = asyncio.create_task(images._watch_disconnect(request, stop))
        for _ in range(3000):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)
        watcher.cancel()
        seen["stop"] = stop.is_set()
        return JSONResponse({})

    app.add_route("/x", x, methods=["POST"])

    async def run():
        receive, leave, waiting = _disconnecting_receive(b"{}")
        sent = []

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"},
                 "http_version": "1.1", "method": "POST", "path": "/x",
                 "raw_path": b"/x", "query_string": b"", "root_path": "",
                 "headers": [(b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
                 "scheme": "http"}
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(waiting.wait(), 30)
        leave()
        await asyncio.wait_for(task, 30)

    asyncio.run(run())
    assert seen["stop"] is True


def test_each_request_logs_one_line_and_warns_on_ignored_fields(served, capsys, caplog):
    with caplog.at_level(logging.WARNING):
        r = _client().post("/v1/images/generations",
                           json={"prompt": "x", "user": "u", "style": "vivid",
                                 "frobnicate": 1})
    assert r.status_code == 200
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("[req]")]
    assert len(lines) == 1 and "/v1/images/generations zit" in lines[0]
    warned = [m for m in caplog.messages if "ignoring unsupported" in m]
    assert len(warned) == 1
    assert "frobnicate" in warned[0] and "style" in warned[0] and "user" in warned[0]


def test_the_access_log_drops_the_image_routes_and_file_tokens():
    from gmlx.serve.patches.observability import _AccessNoiseFilter

    f = _AccessNoiseFilter()

    def rec(path):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                                 '%s - "%s %s HTTP/%s" %d',
                                 ("127.0.0.1", "GET", path, "1.1", 200), None)

    assert not f.filter(rec("/v1/images/variations"))
    assert not f.filter(rec("/v1/images/files/" + "a" * 43 + ".png"))
    assert f.filter(rec("/v1/images/other"))


def test_the_media_gate_takes_image_forms_and_checks_json_edit_images():
    import asyncio

    from gmlx.serve.patches import media_gate as mg

    boundary = "b0"
    raw = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n"
           f"x\r\n--{boundary}--\r\n").encode()
    from starlette.datastructures import Headers

    headers = Headers({"content-type": f"multipart/form-data; boundary={boundary}"})
    for path in ("/v1/images/edits", "/images/variations"):
        asyncio.run(mg.check_request("POST", path, headers, raw))
    with pytest.raises(mg.MediaRefused):
        asyncio.run(mg.check_request("POST", "/v1/images/generations", headers, raw))
    body = json.dumps({"prompt": "x", "images": [{"image_url": "/etc/passwd"}]})
    with pytest.raises(mg.MediaRefused):
        asyncio.run(mg.check_request("POST", "/v1/images/edits",
                                     Headers({"content-type": "application/json"}),
                                     body.encode()))


def test_a_responses_image_generation_tool_warns_once(caplog):
    from gmlx.serve.patches import api_contract

    with caplog.at_level(logging.WARNING):
        api_contract._warn_image_tool("/v1/responses", {
            "tools": [{"type": "image_generation"}, {"type": "function"}]})
        api_contract._warn_image_tool("/v1/responses", {
            "tools": [{"type": "function"}]})
    assert sum("image_generation tool" in m for m in caplog.messages) == 1


def test_the_consumed_sets_name_every_documented_field():
    from pathlib import Path

    from gmlx.serve.patches import api_contract as c

    text = (Path(__file__).resolve().parents[2] / "docs"
            / "image-api.md").read_text()
    every = (c.IMAGES_GENERATIONS_CONSUMED | c.IMAGES_EDITS_CONSUMED
             | c.IMAGES_VARIATIONS_CONSUMED)
    missing = sorted(f for f in every if f"`{f}`" not in text)
    assert not missing, f"docs/image-api.md does not name {missing}"


def test_a_reinstall_leaves_one_handler_per_path():
    # The route install runs again on every reload.
    cfg = build_config(_DOC)
    images.install_image_routes(cfg)
    images.install_image_routes(cfg)
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    for route in images.ROUTES:
        assert paths.count(f"/v1/images/{route}") == 1
    assert paths.count("/v1/images/files/{name}") == 1


def test_a_shared_encoder_encodes_as_a_step_of_the_chat_engine(monkeypatch):
    """A chat model's engine runs a queued job only as steps between its
    decode steps; a plain job reaches its generation path and fails."""
    import importlib
    from types import SimpleNamespace

    import gmlx.serve.engine_jobs as engine_jobs
    from gmlx.image import api

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime")
    # A text-only GGUF is served inside the VLM-shaped wrapper.
    rg = SimpleNamespace(model=SimpleNamespace(
        _is_text_model=True,
        language_model=SimpleNamespace(_model="CHAT")), tokenizer="TOK")
    monkeypatch.setattr(app_mod, "get_cached_model", lambda *a, **k: None)
    monkeypatch.setattr(runtime_mod, "runtime",
                        SimpleNamespace(response_generator=rg))
    monkeypatch.setattr(api, "_borrowed", lambda pair, spec, gguf=None: SimpleNamespace(
        model=pair[0], tokenizer=pair[1]))
    import gmlx.image.text_encoder as text_encoder
    monkeypatch.setattr(text_encoder, "encoder_spec", lambda fam: "SPEC")
    calls = {}

    def run_on_engine(engine_rg, job, *, stepwise=False, **kw):
        calls["stepwise"] = stepwise
        gen = job(engine_rg, lambda: False)
        try:
            next(gen)
        except StopIteration as done:
            return done.value
        raise AssertionError("the encode took more than one step")

    monkeypatch.setattr(engine_jobs, "run_on_engine", run_on_engine)
    borrow, hold = images._borrow_for(SimpleNamespace(encoder_model="chat"))
    pipe = SimpleNamespace(family="FAM", encoder=None)
    seen = []
    out = borrow(pipe, lambda: seen.append(vars(pipe.encoder)) or "FEATURES",
                 threading.Event(), None)
    assert calls == {"stepwise": True} and out == "FEATURES"
    assert seen == [{"model": "CHAT", "tokenizer": "TOK"}]
    # What stays names the keys, and holds no chat model.
    assert pipe.encoder is None and vars(pipe.keys_encoder) == {
        "model": None, "tokenizer": "TOK"}


@pytest.mark.parametrize("share, paced", [(None, True), ("0.25", True),
                                          ("off", False)])
def test_the_job_runs_under_the_chat_pacer(served, monkeypatch, share, paced):
    from gmlx.image import pacing
    from gmlx.serve.gpu_share import ImagePacer

    if share is None:
        monkeypatch.delenv("GMLX_IMAGE_CHAT_SHARE", raising=False)
    else:
        monkeypatch.setenv("GMLX_IMAGE_CHAT_SHARE", share)
    seen = []

    def job(req, kw):
        seen.append(getattr(pacing._local, "pacer", None))
        served.job = None
        return svc.ImageJobResult(
            images=[b"img"], size=(64, 32), quality=req.quality, steps=4,
            seeds=[0], usage={"input_tokens": 1, "output_tokens": 1,
                              "total_tokens": 2, "input_tokens_details": {
                                  "text_tokens": 1, "image_tokens": 0}})

    served.job = job
    r = _client().post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    assert isinstance(seen[0], ImagePacer) == paced
    if paced and share is not None:
        assert seen[0].ratio == pytest.approx(0.25 / 0.75)
    # The worker thread holds no pacer after the job.
    assert svc.WORKER.submit(
        lambda: getattr(pacing._local, "pacer", None)).result() is None


# A request driven through the endpoint itself, with an ASGI receive that
# can leave, as a client that closes its connection does.

def _scope(path="/v1/images/generations"):
    return {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "path": path,
            "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json"),
                        (b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
            "scheme": "http"}


def _receive(body: dict, *, gone: bool):
    import asyncio

    raw = json.dumps(body).encode()
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": raw, "more_body": False}
        if not gone:
            await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    return receive


async def _job_ended():
    """Wait until the image worker has run what is queued on it: it is one
    thread, so the job and its done callbacks have run."""
    import asyncio

    await asyncio.wrap_future(svc.WORKER.submit(lambda: None))


def test_a_stream_closed_after_its_job_ended_frees_its_hold(served):
    import asyncio

    from starlette.requests import Request

    images.STATE.in_flight = 0

    async def run():
        ep = images.make_endpoint("generations")
        body = {"prompt": "x", "stream": True, "partial_images": 2}
        resp = await ep(Request(_scope(), _receive(body, gone=False)))
        it = resp.body_iterator
        await it.__anext__()               # the first partial
        await _job_ended()
        await it.aclose()                  # the client left

    asyncio.run(run())
    assert served.released == [1]
    assert images.STATE.in_flight == 0


@pytest.mark.parametrize("running", [False, True])
def test_a_stream_whose_body_never_starts_frees_its_hold(served, running):
    """A client gone before the response starts: the body task is cancelled
    before the generator runs, so the job's own end frees the slot."""
    import asyncio

    from starlette.requests import Request

    images.STATE.in_flight = 0
    if running:
        def job(req, kw):
            kw["stop"].wait(2)
            raise svc.JobStopped(False)

        served.job = job

    async def run():
        ep = images.make_endpoint("generations")
        body = {"prompt": "x", "stream": True, "partial_images": 1}
        receive = _receive(body, gone=True)
        resp = await ep(Request(_scope(), receive))

        async def send(message):
            pass

        await resp(_scope(), receive, send)
        await _job_ended()

    asyncio.run(run())
    assert served.released == [1]
    assert images.STATE.in_flight == 0


def test_a_left_stream_logs_no_unretrieved_future_or_pending_task(
        served, caplog):
    import asyncio
    import gc

    from starlette.requests import Request

    def job(req, kw):
        kw["emit"](0, b"p0")
        kw["stop"].wait(2)
        raise RuntimeError("the job failed after the client left")

    served.job = job

    async def run():
        ep = images.make_endpoint("generations")
        body = {"prompt": "x", "stream": True, "partial_images": 1}
        resp = await ep(Request(_scope(), _receive(body, gone=False)))
        it = resp.body_iterator
        await it.__anext__()
        await it.aclose()
        await _job_ended()                 # the job failed after the client left

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        asyncio.run(run())
        assert served.released == [1]
        gc.collect()
    bad = [m for m in caplog.messages
           if "never retrieved" in m or "Task was destroyed" in m]
    assert bad == []


def test_a_gone_client_stops_its_job(served):
    """The endpoint's disconnect watcher sets the job's stop event."""
    import asyncio

    from starlette.requests import Request

    seen = []

    def job(req, kw):
        seen.append(kw["stop"].wait(30))
        raise svc.JobStopped(False)

    served.job = job

    async def run():
        ep = images.make_endpoint("generations")
        return await ep(Request(_scope(), _receive({"prompt": "x"},
                                                   gone=True)))

    resp = asyncio.run(run())
    assert seen == [True] and resp.status_code == 499
    assert served.released == [1]


@pytest.mark.parametrize("exc, status", [("JobTimeout", 504),
                                         ("JobCancelled", 499)])
def test_a_shared_encoder_timeout_or_cancel_ends_the_job_as_a_stop(
        monkeypatch, exc, status):
    import gmlx.serve.engine_jobs as engine_jobs

    app_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    runtime_mod = importlib.import_module("gmlx._vendor.mlx_vlm.server.runtime")
    monkeypatch.setattr(app_mod, "get_cached_model", lambda *a, **k: None)
    monkeypatch.setattr(runtime_mod, "runtime", SimpleNamespace(
        response_generator=SimpleNamespace(model=None, tokenizer=None)))

    def run_on_engine(*a, **k):
        raise getattr(engine_jobs, exc)("the engine job ended")

    monkeypatch.setattr(engine_jobs, "run_on_engine", run_on_engine)
    borrow, _ = images._borrow_for(SimpleNamespace(encoder_model="chat"))
    with pytest.raises(svc.JobStopped) as e:
        borrow(SimpleNamespace(family="F", encoder=None), lambda: None,
               threading.Event(), 1.0)
    assert e.value.timed_out == (status == 504)


def test_the_budget_counts_the_other_models_not_this_one(monkeypatch):
    from gmlx.serve import capacity

    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100)
    hold = SimpleNamespace(
        _entry=SimpleNamespace(footprint=30),
        _pool=SimpleNamespace(stats=lambda: {"resident_bytes": 70}))
    assert images._budget(hold) == 60
    monkeypatch.setattr(capacity, "overcommit", lambda: True)
    assert images._budget(hold) is None


def test_the_responses_route_warns_on_the_image_generation_tool(caplog):
    import asyncio

    from starlette.requests import Request

    from gmlx.serve.patches import api_contract

    async def original(request):
        return "ANSWER"

    ep = api_contract._make_raw_endpoint(set(), "/v1/responses")(original)
    raw = json.dumps({"model": "m", "input": "x",
                      "tools": [{"type": "image_generation"}]}).encode()
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": raw, "more_body": False}

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(ep(Request(_scope("/v1/responses"), receive)))
    assert out == "ANSWER"
    assert sum("image_generation tool" in m for m in caplog.messages) == 1


@pytest.mark.parametrize("value, ok", [
    ("BASE64", True), ("data:image/png;base64,BASE64", True),
    ("/tmp/a.png", False), ("https://example.com/a.png", False),
    ("not base64!", False)])
def test_a_json_image_string_is_base64_or_a_data_url(served, value, ok):
    b64 = base64.b64encode(_png()).decode()
    r = _client().post("/v1/images/edits", json={
        "prompt": "x", "images": [{"image_url": value.replace("BASE64", b64)}]})
    assert (r.status_code == 200) == ok, r.text


def test_each_server_keeps_its_url_files_under_its_port(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    st = images._State()
    st.configure(build_config({**_DOC, "server": {"port": 8123}}))
    assert st.store.root == str(tmp_path / "gmlx" / "images" / "8123")
