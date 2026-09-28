"""gmlx/serve/patches/media_gate.py: a request names media only as inline
data, and a path or URL is refused with 400 before anything opens it."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from gmlx.serve.patches import media_gate as mg

PNG = io.BytesIO()
Image.new("RGB", (2, 2)).save(PNG, format="PNG")
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG.getvalue()).decode()
MP4_URI = "data:video/mp4;base64," + base64.b64encode(
    b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 16).decode()
WAV_B64 = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode()


@pytest.fixture
def opened(monkeypatch):
    """Every way the server could open or fetch a media reference, made to
    record the call instead."""
    calls = []

    def spy(name):
        def record(*a, **k):
            calls.append((name, a[:1]))
            raise AssertionError(f"{name} ran for a refused request")
        return record
    import cv2
    import requests
    import mlx_vlm.utils as mu
    monkeypatch.setattr(Image, "open", spy("Image.open"))
    monkeypatch.setattr(requests, "get", spy("requests.get"))
    monkeypatch.setattr(requests.Session, "request", spy("requests.Session.request"))
    monkeypatch.setattr(cv2, "VideoCapture", spy("cv2.VideoCapture"))
    for name in ("load_image", "load_audio", "load_video"):
        monkeypatch.setattr(mu, name, spy(name))
    return calls


def _client(allow_urls=False):
    """A stand-in server with the gate, whose routes record that they ran."""
    app = FastAPI()
    ran = []

    async def route(request: Request):
        ran.append(await request.json())
        return {"ok": True}
    for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages",
                 "/v1/images/generations", "/v1/images/edits", "/v1/audio/speech"):
        app.add_api_route(path, route, methods=["POST"])
    mg.install_media_gate(allow_urls, app=app)
    return TestClient(app), ran


def _chat(part):
    return {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"}, part]}]}


REFUSED = [
    ("/v1/chat/completions", _chat({"type": "image_url",
                                    "image_url": {"url": "/Users/me/.ssh/id_ed25519"}})),
    ("/v1/chat/completions", _chat({"type": "image_url",
                                    "image_url": {"url": "file:///etc/passwd"}})),
    ("/v1/chat/completions", _chat({"type": "image_url",
                                    "image_url": "~/Desktop/shot.png"})),
    ("/v1/chat/completions", _chat({"type": "image_url",
                                    "image_url": {"url": "data:text/plain;base64,QQ=="}})),
    ("/v1/chat/completions", _chat({"type": "input_image", "image_url": "../x.png"})),
    ("/v1/chat/completions", _chat({"type": "input_image", "file_id": "file-1"})),
    ("/v1/chat/completions", _chat({"type": "video_url",
                                    "video_url": {"url": "/Users/me/movie.mp4"}})),
    ("/v1/chat/completions", _chat({"type": "input_video", "video": "file:///m.mp4"})),
    ("/v1/chat/completions", _chat({"type": "video", "video":
                                    "data:video/mp4;base64," + base64.b64encode(
                                        b"#EXTM3U\nfile:///etc/passwd\n").decode()})),
    ("/v1/chat/completions", _chat({"type": "input_audio",
                                    "input_audio": {"data": "/Users/me/memo.wav"}})),
    ("/v1/chat/completions", _chat({"type": "input_audio",
                                    "input_audio": {"data": "./memo"}})),
    # Valid base64 that the audio reader would still take as a path.
    ("/v1/chat/completions", _chat({"type": "input_audio",
                                    "input_audio": {"data": "/Users/me/memo00"}})),
    ("/v1/responses", {"model": "m", "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "/private/etc/hosts"}]}]}),
    # A tool output nested in a responses input item.
    ("/v1/responses", {"model": "m", "input": [{"type": "function_call_output",
        "call_id": "c", "output": [{"type": "input_image",
                                    "image_url": "/Users/me/a.png"}]}]}),
    ("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "/Users/me/a.png"}}]}]}),
    ("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "text/x;a",
                                     "data": "QQ=="}}]}]}),
    ("/v1/images/generations", {"model": "m", "prompt": "p",
                                "output_path": "~/.zshrc"}),
    ("/v1/images/generations", {"model": "m", "prompt": "p", "response_format": "path"}),
    ("/v1/images/generations", {"model": "m", "prompt": "p",
                                "prompt_expansion_model": "/Users/me/model"}),
    ("/v1/images/edits", {"model": "m", "prompt": "p", "image": ["/Users/me/a.png"]}),
    ("/v1/audio/speech", {"model": "tts-1", "input": "hi",
                          "voice": "/Users/me/voice.safetensors"}),
    ("/v1/audio/speech", {"model": "tts-1", "input": "hi", "voice": "../../x"}),
    ("/v1/audio/speech", {"model": "tts-1", "input": "hi",
                          "ref_audio": "/Users/me/voice.wav"}),
]


@pytest.mark.parametrize("path, body", REFUSED)
def test_a_media_path_is_refused_before_anything_opens_it(opened, path, body):
    client, ran = _client()
    r = client.post(path, json=body)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert ran == [] and opened == []


URLS = [
    _chat({"type": "image_url", "image_url": {"url": "http://127.0.0.1:5432/"}}),
    _chat({"type": "input_image", "image_url": "https://example.com/a.png"}),
    _chat({"type": "input_audio", "input_audio": {"data": "https://example.com/a.wav"}}),
    _chat({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}),
    _chat({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}),
]


@pytest.mark.parametrize("body", URLS)
def test_a_url_is_refused_unless_the_server_allows_urls(opened, body):
    client, ran = _client()
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and "server.media_urls" in r.json()["error"]["message"]
    assert ran == [] and opened == []
    client, ran = _client(allow_urls=True)
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert ran == [body]


def test_allowing_urls_never_allows_a_path(opened):
    client, ran = _client(allow_urls=True)
    body = _chat({"type": "image_url", "image_url": {"url": "/Users/me/a.png"}})
    assert client.post("/v1/chat/completions", json=body).status_code == 400
    assert ran == [] and opened == []


ACCEPTED = [
    ("/v1/chat/completions", _chat({"type": "image_url", "image_url": {"url": PNG_URI}})),
    ("/v1/chat/completions", _chat({"type": "input_image", "image_url": PNG_URI})),
    ("/v1/chat/completions", _chat({"type": "input_audio",
                                    "input_audio": {"data": WAV_B64, "format": "wav"}})),
    ("/v1/chat/completions", _chat({"type": "input_audio", "input_audio": {
        "data": "data:audio/wav;base64," + WAV_B64}})),
    ("/v1/chat/completions", _chat({"type": "video_url", "video_url": {"url": MP4_URI}})),
    ("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": PNG_URI.split(",", 1)[1]}}]}]}),
    ("/v1/audio/speech", {"model": "tts-1", "input": "hi", "voice": "af_heart,af_bella"}),
    ("/v1/images/generations", {"model": "m", "prompt": "p",
                                "response_format": "b64_json"}),
    # A tool schema whose "type" keys are not media parts.
    ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user",
        "content": "hi"}], "tools": [{"type": "function", "function": {
            "name": "f", "parameters": {"type": "object", "properties": {
                "url": {"type": "string"}}}}}]}),
]


@pytest.mark.parametrize("path, body", ACCEPTED)
def test_inline_media_reaches_the_route_unchanged(opened, path, body):
    client, ran = _client()
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    assert ran == [body] and opened == []


def test_a_deeply_nested_body_is_walked_without_recursion():
    body: dict = {"type": "input_image", "image_url": "/Users/me/a.png"}
    for _ in range(50_000):
        body = {"content": [body]}
    with pytest.raises(mg.MediaRefused):
        mg.check_body(body, "/v1/chat/completions")


@pytest.mark.parametrize("images, audio, videos", [
    (["/Users/me/a.png"], None, None),
    (None, ["/Users/me/a.wav"], None),
    (None, ["https://example.com/a.wav"], None),
    (None, None, ["/Users/me/a.mp4"]),
])
def test_the_generation_path_refuses_what_the_walk_missed(images, audio, videos):
    from fastapi import HTTPException

    from gmlx.serve import mem_preflight as mp

    mg.install_media_gate(False, app=FastAPI())
    with pytest.raises(HTTPException) as e:
        mp._check_media(images, audio, videos)
    assert e.value.status_code == 400
    mp._check_media([PNG_URI, Image.new("RGB", (1, 1))], [io.BytesIO(b"x")], [MP4_URI])


@pytest.mark.parametrize("method", ["generate", "validate_context_budget"])
def test_the_installed_generation_wrappers_run_the_check(monkeypatch, method):
    from fastapi import HTTPException
    from mlx_vlm.server.generation import ResponseGenerator as RG

    from gmlx.serve import mem_preflight as mp

    reached = []
    monkeypatch.setattr(RG, method, lambda self, *a, **k: reached.append(a))
    monkeypatch.setattr(RG, "generate" if method != "generate"
                        else "validate_context_budget", lambda self, *a, **k: None)
    monkeypatch.setattr(RG.generate, mp._INSTALLED_FLAG, False, raising=False)
    mp.install_memory_preflight()
    mg.install_media_gate(False, app=FastAPI())
    with pytest.raises(HTTPException) as e:
        getattr(RG, method)(object(), "p", images=["/Users/me/a.png"])
    assert e.value.status_code == 400 and reached == []
