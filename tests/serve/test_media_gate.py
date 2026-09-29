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
    # Outside the gate, as in the server.
    from gmlx.serve.patches.hardening import install_json_content_type_tolerance
    install_json_content_type_tolerance(app=app)
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


@pytest.mark.parametrize("url, scheme", [("ftp://host/a.png", "ftp"),
                                         ("GOPHER://h/x", "gopher")])
def test_another_scheme_is_named_in_the_refusal(opened, url, scheme):
    client, ran = _client(allow_urls=True)
    body = _chat({"type": "image_url", "image_url": {"url": url}})
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert f"names a {scheme}:// URL" in r.json()["error"]["message"]
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


# Every body a route would parse

def _form_client():
    """The stand-in server plus the two upload routes, which read a form."""
    client, ran = _client()
    app = client.app

    async def upload(request: Request):
        form = await request.form()
        ran.append({k: (v if isinstance(v, str) else len(await v.read()))
                    for k, v in form.multi_items()})
        return {"ok": True}
    for path in ("/v1/audio/transcriptions", "/v1/audio/translations"):
        app.add_api_route(path, upload, methods=["POST"])
    app.middleware_stack = None
    return client, ran


WRITES = [
    {"model": "m", "prompt": "p", "output_path": "/Users/me/.zshrc"},
    {"model": "m", "prompt": "p", "output_dir": "/Users/me/Library/LaunchAgents"},
    {"model": "m", "prompt": "p", "response_format": "path"},
    {"model": "m", "prompt": "p", "prompt_expansion_model": "/Users/me/model"},
]
# Content types under which a route still reads the body as JSON.
CONTENT_TYPES = [
    "application/json",
    "Application/JSON; charset=utf-8",
    "application/vnd.api+json",
    "text/plain",
    "multipart/form-data; boundary=x",
    "MULTIPART/FORM-DATA",
    "application/x-www-form-urlencoded",
    None,
]


@pytest.mark.parametrize("ctype", CONTENT_TYPES)
@pytest.mark.parametrize("body", WRITES)
def test_a_file_write_is_refused_whatever_the_content_type(opened, body, ctype):
    import json
    client, ran = _client()
    headers = {} if ctype is None else {"content-type": ctype}
    r = client.post("/v1/images/generations", content=json.dumps(body).encode(),
                    headers=headers)
    assert r.status_code == 400, (ctype, r.text)
    assert ran == [] and opened == []


def test_a_chunked_body_is_checked(opened):
    import json
    client, ran = _client()

    def chunks():
        raw = json.dumps(WRITES[0]).encode()
        yield raw[:10]
        yield raw[10:]
    r = client.post("/v1/images/generations", content=chunks(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 400 and ran == []


@pytest.mark.parametrize("ctype, content", [
    ("multipart/form-data; boundary=b",
     b'--b\r\nContent-Disposition: form-data; name="output_path"\r\n\r\n/x\r\n--b--\r\n'),
    ("application/x-www-form-urlencoded", b"output_path=%2Fx&prompt=p"),
])
def test_a_form_body_reaches_only_the_upload_routes(opened, ctype, content):
    client, ran = _form_client()
    for path in ("/v1/images/generations", "/v1/chat/completions", "/v1/audio/speech"):
        r = client.post(path, content=content, headers={"content-type": ctype})
        message = r.json()["error"]["message"]
        assert r.status_code == 400 and ("JSON body" in message
                                         or "not valid JSON" in message), message
    assert ran == [] and opened == []


def test_an_upload_route_gets_its_form_whole(opened):
    client, ran = _form_client()
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1",
                                                      "language": "en"},
                    files={"file": ("a.wav", b"RIFF" + b"\x00" * 5000, "audio/wav")})
    assert r.status_code == 200, r.text
    assert ran == [{"model": "whisper-1", "language": "en", "file": 5004}]


def test_an_upload_route_refuses_an_unreadable_form(opened):
    client, ran = _form_client()
    r = client.post("/v1/audio/transcriptions", content=b"--x\r\nno headers",
                    headers={"content-type": "multipart/form-data"})
    assert r.status_code == 400 and ran == []


def test_the_upload_text_fields_go_through_the_same_check(monkeypatch):
    seen = []
    monkeypatch.setattr(mg, "check_body", lambda body, path: seen.append(body))
    client, ran = _form_client()
    client.post("/v1/audio/transcriptions", data={"model": "w", "prompt": "hi"},
                files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert {"model": "w"} in seen and {"prompt": "hi"} in seen
    assert not any("file" in b for b in seen if isinstance(b, dict))


# The image calls refuse file writes and paths themselves

def _Core(extra=None, image_paths=()):
    # The request mlx-vlm's edit route builds, a dataclass like the
    # generation request.
    from mlx_vlm.generate.edit_image import ImageEditRequest
    return ImageEditRequest(prompt="p", image_paths=tuple(image_paths), extra=extra or {})


def _guarded():
    import types
    called = []
    openai = types.SimpleNamespace(
        generate_image=lambda model, request, **kw: called.append(("gen", kw)) or "img",
        edit_image=lambda model, request, **kw: called.append(("edit", kw)) or "img")
    mg._refuse_image_writes(openai)
    return openai, called


@pytest.mark.parametrize("call", [
    lambda o: o.generate_image("m", _Core(), output_path="/Users/me/x.png"),
    lambda o: o.generate_image("m", _Core({"prompt_expansion_model": "/Users/me/m"})),
    lambda o: o.generate_image("m", _Core(), prompt_expansion_model="/Users/me/m"),
    lambda o: o.generate_image("m", "prompt", image_paths=["/Users/me/a.png"]),
    lambda o: o.edit_image("m", _Core(image_paths=["/Users/me/a.png"])),
    lambda o: o.edit_image("m", _Core(image_paths=[PNG_URI]), output_path="outputs/x.png"),
])
def test_the_image_calls_refuse_writes_and_paths(opened, call):
    from fastapi import HTTPException
    openai, called = _guarded()
    with pytest.raises(HTTPException) as e:
        call(openai)
    assert e.value.status_code == 400 and called == [] and opened == []


def test_the_image_calls_take_inline_images():
    openai, called = _guarded()
    assert openai.generate_image("m", _Core(), output_path=None) == "img"
    assert openai.edit_image("m", _Core(image_paths=[PNG_URI])) == "img"
    assert [c[0] for c in called] == ["gen", "edit"]
    first = openai.generate_image
    mg._refuse_image_writes(openai)                 # installed once
    assert openai.generate_image is first


def _file_state(path):
    import os
    import stat
    with Image.open(path) as image:
        size = image.size
    return (stat.S_IMODE(os.stat(path).st_mode),
            stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode), size)


def _recording(seen, fail=False):
    import types

    def call(model, request, **kw):
        paths = [*(kw.get("image_paths") or ()), *(getattr(request, "image_paths", ()))]
        seen.extend(paths)
        seen.extend(_file_state(p) for p in paths)
        if fail:
            raise RuntimeError("model failed")
        return "img"
    openai = types.SimpleNamespace(generate_image=call, edit_image=call)
    mg._refuse_image_writes(openai)
    return openai


@pytest.mark.parametrize("call", [
    lambda o: o.edit_image("m", _Core(image_paths=[PNG_URI, PNG_URI])),
    lambda o: o.edit_image("m", "prompt", image_paths=[PNG_URI, PNG_URI]),
    lambda o: o.generate_image("m", "prompt", task="edit", image_paths=[PNG_URI, PNG_URI]),
])
def test_an_inline_edit_image_reaches_the_call_as_a_private_file(call):
    import os
    seen = []
    assert call(_recording(seen)) == "img"
    paths, states = seen[:2], seen[2:]
    assert all(isinstance(p, str) and not p.startswith("data:") for p in paths)
    assert len(set(paths)) == 2
    assert states == [(0o600, 0o700, (2, 2))] * 2
    assert not any(os.path.exists(p) for p in paths)
    assert not os.path.exists(os.path.dirname(paths[0]))


def test_the_edit_image_files_are_removed_when_the_call_fails():
    import os
    seen = []
    with pytest.raises(RuntimeError):
        _recording(seen, fail=True).edit_image("m", _Core(image_paths=[PNG_URI]))
    assert seen and not os.path.exists(os.path.dirname(seen[0]))


def _uri(data: bytes, kind="png"):
    return f"data:image/{kind};base64," + base64.b64encode(data).decode()


def _image_bytes(fmt, size=(2, 2)):
    out = io.BytesIO()
    Image.new("RGB", size).save(out, format=fmt)
    return out.getvalue()


def _bad_checksum_png():
    # Opens as a 2x2 PNG, and only verify() finds the damaged chunk.
    data = bytearray(PNG.getvalue())
    data[data.index(b"IDAT") + 6] ^= 0xFF
    return bytes(data)


@pytest.mark.parametrize("value, limits", [
    ("/Users/me/a.png", {}),
    ("file:///Users/me/a.png", {}),
    (_uri(b"not an image"), {}),
    (_uri(_image_bytes("BMP"), "bmp"), {}),
    (_uri(b"%!PS-Adobe-3.0 EPSF-3.0\n", "eps"), {}),
    (_uri(PNG.getvalue()[:40]), {}),
    (_uri(_bad_checksum_png()), {}),
    (PNG_URI, {"MEDIA_MAX_BYTES": 16}),
    (_uri(_image_bytes("PNG", (8, 8))), {"MEDIA_MAX_PIXELS": 63}),
])
def test_an_edit_image_that_is_not_a_small_inline_image_is_refused(monkeypatch, value, limits):
    from fastapi import HTTPException
    for name, limit in limits.items():
        monkeypatch.setattr(mg, name, limit)
    folders = []
    monkeypatch.setattr(mg.tempfile, "mkdtemp", lambda **k: folders.append(k) or "/nonexistent")
    seen = []
    with pytest.raises(HTTPException) as e:
        _recording(seen).edit_image("m", _Core(image_paths=[PNG_URI, value]))
    assert e.value.status_code == 400 and seen == [] and folders == []


def test_websocket_routes_are_dropped():
    from starlette.routing import WebSocketRoute
    app = FastAPI()

    async def ws(websocket):
        await websocket.accept()
    app.add_api_websocket_route("/v1/realtime", ws)
    mg.install_media_gate(app=app)
    assert not [r for r in app.router.routes if isinstance(r, WebSocketRoute)]


# The media folder, the session rule and the size limits

@pytest.fixture
def media_root(tmp_path):
    root = str(tmp_path / "media")
    mg.set_media_root(root, (root,))
    yield root
    mg.set_media_root(None)


def _check(body, *, inline_only=False, path="/v1/chat/completions"):
    import asyncio
    import json
    return asyncio.run(mg.check_request(
        "POST", path, {"content-type": "application/json"},
        json.dumps(body).encode(), inline_only=inline_only))


def test_a_path_in_the_media_folder_passes_the_gate(media_root):
    for ref in (f"{media_root}/a.png", f"file://{media_root}/sub/a.png"):
        _check(_chat({"type": "image_url", "image_url": {"url": ref}}))
    _check(_chat({"type": "input_audio", "input_audio": {"data": f"{media_root}/a.wav"}}))


@pytest.mark.parametrize("ref", ["{root}/../secret.png", "{root}/./a.png", "{root}",
                                 "{root}x/a.png", "~/.cache/gmlx/media/a.png"])
def test_a_path_that_leaves_the_media_folder_is_refused(media_root, ref):
    with pytest.raises(mg.MediaRefused, match=f"cp -c FILE {media_root}/"):
        _check(_chat({"type": "image_url",
                      "image_url": {"url": ref.format(root=media_root)}}))


def test_a_session_socket_request_takes_media_only_inline(media_root):
    part = {"type": "image_url", "image_url": {"url": f"{media_root}/a.png"}}
    with pytest.raises(mg.MediaRefused, match="launch container session takes media "
                                              "only inline"):
        _check(_chat(part), inline_only=True)
    # The rule holds for this request only.
    _check(_chat(part))


def test_a_session_socket_request_may_not_name_a_url(monkeypatch):
    monkeypatch.setattr(mg, "_allow_urls", True)
    url = "https://example.com/a.png"
    for body in (_chat({"type": "image_url", "image_url": {"url": url}}),
                 _chat({"type": "input_audio", "input_audio": {"data": url}})):
        with pytest.raises(mg.MediaRefused, match="names a URL, and a launch container "
                                                  "session takes media only inline"):
            _check(body, inline_only=True)
        # server.media_urls still lets the TCP listener take it.
        _check(body)


def test_without_a_media_folder_every_path_is_refused():
    mg.set_media_root(None)
    with pytest.raises(mg.MediaRefused, match="does not open files"):
        _check(_chat({"type": "image_url", "image_url": {"url": "/tmp/gmlx/media/a.png"}}))


def test_inline_media_over_the_limit_is_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(mg, "MEDIA_MAX_BYTES", 8)
    decoded = []
    monkeypatch.setattr(mg.base64, "b64decode",
                        lambda *a, **k: decoded.append(1) or b"")
    for part in ({"type": "image_url", "image_url": {"url": PNG_URI}},
                 {"type": "input_audio", "input_audio": {"data": WAV_B64 * 4}},
                 {"type": "video_url", "video_url": {"url": MP4_URI}}):
        with pytest.raises(mg.MediaRefused, match="larger than the 0 MiB limit"):
            _check(_chat(part))
    assert decoded == []


def test_a_body_that_is_not_json_is_refused():
    import asyncio
    for ctype, match in (("application/json", "not valid JSON"),
                         ("application/x-www-form-urlencoded", "not a form"),
                         ("text/plain", "not valid JSON")):
        with pytest.raises(mg.MediaRefused, match=match):
            asyncio.run(mg.check_request("POST", "/v1/chat/completions",
                                         {"content-type": ctype}, b"a=b"))
    # curl -d sends JSON as a form, and the routes read it as JSON.
    asyncio.run(mg.check_request("POST", "/v1/chat/completions",
                                 {"content-type": "application/x-www-form-urlencoded"},
                                 b'{"model": "m"}'))
