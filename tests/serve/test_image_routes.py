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
    monkeypatch.setattr(svc, "default_store_root", lambda: str(tmp_path / "files"))
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


def test_a_cancelled_job_answers_499_and_a_timeout_504(served):
    served.job = lambda req, kw: (_ for _ in ()).throw(svc.JobStopped(False))
    assert _client().post("/v1/images/generations",
                          json={"prompt": "x"}).status_code == 499
    served.job = lambda req, kw: (_ for _ in ()).throw(svc.JobStopped(True))
    r = _client().post("/v1/images/generations", json={"prompt": "x"})
    assert r.status_code == 504 and r.json()["error"]["type"] == "timeout"
    assert served.released == [1, 1]


def test_the_queue_cap_answers_429_with_retry_after(served):
    gate, entered = threading.Event(), threading.Event()

    def slow(req, kw):
        entered.set()
        gate.wait(5)
        return svc.ImageJobResult(images=[b"x"], size=(64, 32), quality="auto",
                                  usage={}, seeds=[0], steps=1)

    served.job = slow
    images.STATE.settings = dataclasses.replace(images.STATE.settings,
                                                max_queue=0)
    client = _client()
    first = threading.Thread(target=lambda: client.post(
        "/v1/images/generations", json={"prompt": "x"}))
    first.start()
    assert entered.wait(5)
    r = client.post("/v1/images/generations", json={"prompt": "y"})
    gate.set()
    first.join(5)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 2


def _disconnecting_receive(body: bytes):
    """An ASGI receive that gives ``body``, then waits until ``leave()``,
    then answers every call with a disconnect, as uvicorn does."""
    import asyncio

    state = {"sent": False, "event": asyncio.Event()}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await state["event"].wait()
        return {"type": "http.disconnect"}

    def leave():
        state["event"].set()

    return receive, leave


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
        for _ in range(200):
            if stop.is_set():
                break
            await asyncio.sleep(0.01)
        watcher.cancel()
        seen["stop"] = stop.is_set()
        return JSONResponse({})

    app.add_route("/x", x, methods=["POST"])

    async def run():
        receive, leave = _disconnecting_receive(b"{}")
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
        await asyncio.sleep(0.1)
        leave()
        await asyncio.wait_for(task, 10)

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

    text = (Path(__file__).resolve().parents[2] / "docs" / "images.md").read_text()
    every = (c.IMAGES_GENERATIONS_CONSUMED | c.IMAGES_EDITS_CONSUMED
             | c.IMAGES_VARIATIONS_CONSUMED)
    missing = sorted(f for f in every if f"`{f}`" not in text)
    assert not missing, f"docs/images.md does not name {missing}"


def test_a_reinstall_leaves_one_handler_per_path():
    # The route install runs again on every reload.
    cfg = build_config(_DOC)
    images.install_image_routes(cfg)
    images.install_image_routes(cfg)
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    for route in images.ROUTES:
        assert paths.count(f"/v1/images/{route}") == 1
    assert paths.count("/v1/images/files/{name}") == 1
