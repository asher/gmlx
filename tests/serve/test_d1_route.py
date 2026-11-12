#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The /v1/systemone route on a d1 decision model: the request reaches the
decision forward, the refusals are 400s, the memory preflight caps the
batch, and one decision holds the server's turn at a time.

A tiny random lfm3 model answers on the engine-thread shell of the route
tests, with a stand-in template, so no file loads."""

from __future__ import annotations

import logging
import threading

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.serve.mem_preflight as mem_preflight  # noqa: E402
from gmlx.serve import job_route  # noqa: E402
from gmlx.serve.patches import systemone as route  # noqa: E402
from gmlx.systemone import d1  # noqa: E402
from tests.serve.test_systemone_route import _GEN, app  # noqa: E402,F401
from tests.systemone.test_d1_format import _TEMPLATE, _encode  # noqa: E402


def _body(**extra):
    body = {"model": "d1", "state": "The build failed twice this morning.",
            "questions": {
                "broken": {"type": "noul", "instructions": "Is the build broken?"},
                "team": {"type": "choice", "instructions": "Who fixes it?",
                         "criteria": {"infra": None, "app": None}},
                "risk": {"type": "score", "instructions": "How risky?",
                         "criteria": ["low", "mid", "high"]}}}
    body.update(extra)
    return body


@pytest.fixture
def d1_app(app, monkeypatch):  # noqa: F811
    mx = pytest.importorskip("mlx.core")
    from tests.models.test_lfm3_decision_forward import _model

    fmt = d1.D1Format(_TEMPLATE, {"type": "lfm3-d1", "temperature": {}}, [],
                      lambda t: [i % 64 for i in _encode(t)])
    monkeypatch.setattr(route, "_d1_target", lambda *a, **k: True)
    monkeypatch.setattr(route, "_d1_for", lambda rg: fmt)
    monkeypatch.setattr(mem_preflight, "available_drained_bytes", lambda: 10 ** 12)
    client = app.use(diffusion_model=False)
    app.rg.model = _model(mx)
    app.client = client
    return app


def test_a_d1_request_answers_from_the_decision_forward(d1_app, monkeypatch):
    seen = []
    real = mem_preflight.preflight_decision

    def preflight(*a, **k):
        seen.append(job_route.decision_turn_held())
        return real(*a, **k)

    monkeypatch.setattr(mem_preflight, "preflight_decision", preflight)
    r = d1_app.client.post("/v1/systemone", json=_body())
    assert r.status_code == 200, r.text
    out = r.json()
    assert list(out["answers"]) == ["broken", "team", "risk"]
    assert out["answers"]["team"]["choice"] in ("infra", "app")
    assert out["answers"]["risk"]["score"] in (0.0, 1.0, 2.0)
    assert out["usage"]["output_tokens"] == 0
    assert out["diagnostics"]["readout"] == "d1"
    assert out["diagnostics"]["forwards"] == 1
    # The preflight ran while the request held the turn.
    assert seen == [True]
    assert not job_route.decision_turn_held()


def test_adapters_are_refused(d1_app):
    r = d1_app.client.post("/v1/systemone", json=_body(adapters={"x": 1.0}))
    assert r.status_code == 400
    assert "a d1 decision runs no adapters" in r.text


def test_a_question_llama_server_refuses_is_a_400(d1_app):
    body = _body()
    body["questions"]["risk"]["criteria"] = ["only"]
    r = d1_app.client.post("/v1/systemone", json=body)
    assert r.status_code == 400
    assert "2 to 10 levels" in r.text


def test_a_prompt_over_the_cap_is_a_400(d1_app, monkeypatch):
    import gmlx.load.decision as decision

    monkeypatch.setattr(decision, "DECISION_STATE_CAP", 20)
    r = d1_app.client.post("/v1/systemone", json=_body())
    assert r.status_code == 400
    assert "a d1 prompt holds at most 20" in r.text


def test_a_decision_that_does_not_fit_in_memory_is_a_400(d1_app, monkeypatch):
    monkeypatch.setattr(mem_preflight, "available_drained_bytes", lambda: 1)
    r = d1_app.client.post("/v1/systemone", json=_body())
    assert r.status_code == 400
    assert "memory ceiling" in r.text
    assert not job_route.decision_turn_held()


def test_unread_fields_are_named_in_the_warning(d1_app, caplog):
    with caplog.at_level(logging.WARNING):
        r = d1_app.client.post("/v1/systemone", json=_body(samples=4, seed=7))
    assert r.status_code == 200, r.text
    warned = [m for m in caplog.messages if "ignoring unsupported parameter" in m]
    assert warned and "samples" in warned[0] and "seed" in warned[0]


def test_the_row_cap_from_the_preflight_splits_the_batch(d1_app, monkeypatch):
    monkeypatch.setattr(mem_preflight, "preflight_decision",
                        lambda *a, **k: (lambda n: 1))
    r = d1_app.client.post("/v1/systemone", json=_body())
    assert r.status_code == 200, r.text
    assert r.json()["diagnostics"]["forwards"] == 3


def test_a_second_decision_waits_for_the_turn_and_past_its_bound_gets_a_503(
        d1_app, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = d1.decide_d1

    def held(*a, **k):
        started.set()
        assert release.wait(30)
        return (yield from real(*a, **k))

    monkeypatch.setattr(d1, "decide_d1", held)
    first = {}

    def send():
        first["r"] = d1_app.client.post("/v1/systemone", json=_body())

    thread = threading.Thread(target=send)
    thread.start()
    try:
        assert started.wait(30)
        assert job_route.decision_turn_held()
        timeout = _GEN.get_token_queue_timeout
        monkeypatch.setattr(_GEN, "get_token_queue_timeout", lambda: 0)
        busy = d1_app.client.post("/v1/systemone", json=_body())
        assert busy.status_code == 503
        assert busy.headers["Retry-After"] == str(route._BUSY_RETRY_S)
        assert busy.json()["error"]["type"] == "decision_busy"
        monkeypatch.setattr(_GEN, "get_token_queue_timeout", timeout)
    finally:
        release.set()
        thread.join(30)
    assert first["r"].status_code == 200, first["r"].text
    monkeypatch.setattr(d1, "decide_d1", real)
    after = d1_app.client.post("/v1/systemone", json=_body())
    assert after.status_code == 200, after.text


def test_the_preflight_caps_rows_by_the_price(monkeypatch):
    from types import SimpleNamespace

    from gmlx.models.lfm3.decision import price_bytes

    args = SimpleNamespace(hidden_size=2048, num_key_value_heads=4, head_dim=128,
                           intermediate_size=7680)
    rg = SimpleNamespace(model=SimpleNamespace(args=args))
    room = price_bytes(args, 3, 8192)
    monkeypatch.setattr(mem_preflight, "available_drained_bytes", lambda: room)
    cap = mem_preflight.preflight_decision(rg, [8192, 100])
    assert cap(8192) == 3
    assert cap(100) > 3
    # Features kept for the request leave less room.
    held = room - price_bytes(args, 1, 8192, embeds=True)
    assert mem_preflight.preflight_decision(
        rg, [8192], media_peak=1, media_held=held)(8192) == 1
    # A tower pass larger than the room is refused before it runs.
    with pytest.raises(_GEN.PromptTooLongError, match="memory ceiling"):
        mem_preflight.preflight_decision(rg, [100], media_peak=room + 1)
    # Without a memory probe, each batch holds one row.
    monkeypatch.setattr(mem_preflight, "available_drained_bytes", lambda: None)
    assert mem_preflight.preflight_decision(rg, [100])(100) == 1
