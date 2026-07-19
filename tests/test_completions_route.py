#!/usr/bin/env python3
"""The gmlx /v1/completions route: request validation, non-stream and SSE
shapes, stop-sequence trimming, usage accounting, and the default max_tokens.
CPU-only - the engine is a stub ResponseGenerator, no model load."""
from __future__ import annotations

import importlib
import json
import types

import pytest

pytest.importorskip("mlx_vlm")

from gmlx import server_patches as sp  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_GEN = importlib.import_module("mlx_vlm.server.generation")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime


class _Tok:
    def __init__(self, text, finish=None):
        self.text = text
        self.finish_reason = finish
        self.token_count = 1


class _FakeIter:
    def __init__(self, toks):
        self._it = iter(toks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        self.closed = True


class _FakeRG:
    """Stub ResponseGenerator: records the prompt/args, yields fixed tokens."""

    def __init__(self, toks):
        self.toks = toks
        self.calls = []
        self.last_iter = None

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        self.calls.append((prompt, args))
        ctx = types.SimpleNamespace(prompt_tokens=7, uid=1)
        self.last_iter = _FakeIter(self.toks)
        return ctx, self.last_iter


@pytest.fixture(autouse=True)
def _restore():
    saved_routes = list(_APP.app.router.routes)
    saved_gcm = _APP.get_cached_model
    saved_rg = _RUNTIME.response_generator
    yield
    _APP.app.router.routes[:] = saved_routes
    _APP.get_cached_model = saved_gcm
    _RUNTIME.response_generator = saved_rg


def _client(toks=None):
    from fastapi.testclient import TestClient

    rg = _FakeRG(toks if toks is not None
                 else [_Tok("Hello"), _Tok(" wor"), _Tok("ld END tail",
                                                         "stop")])
    _APP.get_cached_model = lambda mid, *a, **k: (None, None, None)
    _RUNTIME.response_generator = rg
    sp.install_completions_route()
    return TestClient(_APP.app), rg


def test_route_registers_both_paths_idempotent():
    sp.install_completions_route()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert "/v1/completions" in paths and "/completions" in paths
    n = len(_APP.app.router.routes)
    sp.install_completions_route()
    assert len(_APP.app.router.routes) == n


def test_non_stream_text_completion_shape():
    client, rg = _client()
    r = client.post("/v1/completions",
                    json={"model": "m", "prompt": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    choice = body["choices"][0]
    assert choice["text"] == "Hello world END tail"
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3,
                             "total_tokens": 10}
    # raw prompt reaches the engine untouched - no chat template
    assert rg.calls[0][0] == "hi"
    assert rg.last_iter.closed


def test_non_stream_stop_trims_and_marks_stop():
    client, rg = _client()
    r = client.post("/v1/completions",
                    json={"model": "m", "prompt": "hi", "stop": "END"})
    choice = r.json()["choices"][0]
    assert choice["text"] == "Hello world "
    assert choice["finish_reason"] == "stop"


def test_non_stream_length_finish_reason():
    client, _ = _client([_Tok("a"), _Tok("b", "length")])
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    assert r.json()["choices"][0]["finish_reason"] == "length"


def test_default_max_tokens_matches_chat_default():
    client, rg = _client()
    client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    args = rg.calls[0][1]
    assert args.max_tokens == _GEN.get_server_max_tokens()
    # sampling params pass through _build_gen_args when set
    client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "max_tokens": 5, "temperature": 0.2,
        "top_p": 0.9, "seed": 42})
    args = rg.calls[1][1]
    assert (args.max_tokens, args.temperature, args.top_p, args.seed) == \
        (5, 0.2, 0.9, 42)
    # token logprobs are never collected on this route
    assert args.logprobs is False


@pytest.mark.parametrize("payload,fragment", [
    ({"prompt": ["a", "b"]}, "single non-empty string"),
    ({"prompt": [1, 2, 3]}, "single non-empty string"),
    ({"prompt": ""}, "single non-empty string"),
    ({"prompt": "p", "n": 2}, "'n' greater than 1"),
    ({"prompt": "p", "echo": True}, "'echo' is not supported"),
    ({"prompt": "p", "suffix": "tail"}, "'suffix'"),
    ({"prompt": "p", "best_of": 4}, "'best_of' greater than 1"),
])
def test_rejected_shapes_get_400(payload, fragment):
    client, _ = _client()
    r = client.post("/v1/completions", json={"model": "m", **payload})
    assert r.status_code == 400, r.text
    assert fragment in r.json()["detail"]


def test_tolerated_singular_values_pass():
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "n": 1, "best_of": 1, "echo": False,
        "suffix": ""})
    assert r.status_code == 200


def _sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        events.append("[DONE]" if payload == "[DONE]"
                      else json.loads(payload))
    return events


def test_stream_chunks_and_done():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    assert all(c["object"] == "text_completion" for c in chunks)
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == "Hello world END tail"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert rg.last_iter.closed


def test_stream_stop_trims_and_ends_early():
    client, rg = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True, "stop": ["END"]})
    events = _sse_events(r.text)
    assert events[-1] == "[DONE]"
    chunks = [e for e in events if e != "[DONE]"]
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == "Hello world "
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert rg.last_iter.closed


def test_stream_include_usage_emits_usage_chunk():
    client, _ = _client()
    r = client.post("/v1/completions", json={
        "model": "m", "prompt": "p", "stream": True,
        "stream_options": {"include_usage": True}})
    events = _sse_events(r.text)
    usage_chunks = [e for e in events
                    if e != "[DONE]" and e.get("usage")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert usage_chunks[0]["choices"] == []


def test_ignored_fields_warn_on_completions(caplog):
    import logging

    client, _ = _client()
    with caplog.at_level(logging.WARNING,
                         logger="gmlx.server_patches.api_contract"):
        r = client.post("/v1/completions", json={
            "model": "m", "prompt": "p", "logprobs": 3, "user": "u"})
    assert r.status_code == 200
    warns = [m for m in caplog.messages if "ignoring unsupported" in m]
    assert len(warns) == 1
    assert "logprobs" in warns[0] and "user" in warns[0]


def test_prompt_too_long_maps_to_400():
    from fastapi.testclient import TestClient

    class _Boom:
        def generate(self, prompt, images=None, audio=None, args=None,
                     videos=None):
            raise _GEN.PromptTooLongError("prompt too long")

    _APP.get_cached_model = lambda mid, *a, **k: (None, None, None)
    _RUNTIME.response_generator = _Boom()
    sp.install_completions_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/completions", json={"model": "m", "prompt": "p"})
    assert r.status_code == 400
    assert "too long" in r.json()["detail"]
