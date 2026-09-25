"""The /v1/systemone route: model policy, status codes, body shapes,
admission, logging and the engine-thread job. The reader is scripted and
the engine is a ResponseGenerator shell, so no model loads."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import math
import queue
import random
import re
import threading
import types
import zlib

import pytest

pytest.importorskip("mlx_vlm")

from fastapi.testclient import TestClient  # noqa: E402

import gmlx.gen.diffusion as diffusion  # noqa: E402
import gmlx.serve.bridge_vlm as serving  # noqa: E402
import gmlx.systemone.engine as engine_mod  # noqa: E402
from gmlx.config import SystemoneCfg, build_config  # noqa: E402
from gmlx.serve.engine_jobs import install_engine_jobs  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import systemone as route  # noqa: E402
from gmlx.serve.residency import _GenerationGuard  # noqa: E402
from gmlx.systemone.reads import Cancelled, ReadResult, SampleRead, SlotRead  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_GEN = importlib.import_module("mlx_vlm.server.generation")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime

_SPECIAL = {"<bos>": 2, "<|think|>": 98, "<|channel>": 100, "<channel|>": 101,
            "<|turn>": 105, "<turn|>": 106, "\n": 107}
_PIECE = re.compile(r"<[^<>\s]+>|\n| ?[^\s<]+| ")


class _Tok:
    """Word-level tokenizer: tags and newlines are single ids, a word with
    its leading space is one id."""

    bos_token = "<bos>"

    def __init__(self):
        self.by_id = {v: k for k, v in _SPECIAL.items()}

    def encode(self, text, add_special_tokens=True):
        ids = [2] if add_special_tokens else []
        for piece in _PIECE.findall(text):
            tid = _SPECIAL.get(piece)
            if tid is None:
                tid = 1000 + zlib.crc32(piece.encode()) % 50000
            self.by_id[tid] = piece
            ids.append(tid)
        return ids

    def decode(self, ids):
        return "".join(self.by_id.get(int(i), "?") for i in ids)

    def apply_chat_template(self, msgs, **kwargs):
        return _render(msgs, **kwargs)


def _render(msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False):
    assert all(isinstance(m["content"], str) for m in msgs)
    sys_text = msgs[0]["content"]
    user = msgs[1]["content"]
    think = "<|think|>\n" if enable_thinking else ""
    return (f"<bos><|turn>system\n{think}{sys_text}<turn|>\n"
            f"<|turn>user\n{user}<turn|>\n<|turn>model\n")


class _Cache:
    def __init__(self, ids):
        self.ids = list(ids)

    def extend(self, ids):
        self.ids += list(ids)


class _Reader:
    """Scripted reader: each slot's label distribution depends only on the
    seed and the slot position."""

    built = []
    mode = "ok"

    def __init__(self, model, *, prefill_step_size):
        self.calls = []
        _Reader.built.append(self)

    def prefill(self, ids):
        return _Cache(ids)

    def read(self, prompt, req, *, should_stop=None):
        if _Reader.mode == "slow":
            while not should_stop():
                threading.Event().wait(0.005)
            raise Cancelled("stopped")
        if _Reader.mode == "boom":
            raise RuntimeError("reader failed")
        self.calls.append(("read", tuple(req.seeds)))
        samples = []
        for seed in req.seeds:
            slots = []
            for slot in req.slots:
                rng = random.Random(seed * 1000 + slot.pos)
                logits = [rng.random() * 3 for _ in slot.label_ids]
                z = math.log(sum(math.exp(x) for x in logits))
                top = {lid: x - z for lid, x in zip(slot.label_ids, logits)}
                best = max(top, key=lambda k: top[k])
                slots.append(SlotRead(argmax_id=best, top=top))
            samples.append(SampleRead(seed=seed, canvas_in=(), slots=tuple(slots)))
        return ReadResult(samples=tuple(samples), prompt_len=len(prompt.ids),
                          timing_ms={})

    def think(self, prompt_ids, budget, *, stop_id, canvas_width, processor,
              backend, should_stop=None):
        self.calls.append(("think", budget, canvas_width))
        return [3000, 3001], {"tokens": 2, "closed": True, "ms": 1.0}


class _Metrics:
    def __init__(self):
        self.events = []

    def begin_request(self, **kw):
        self.events.append(("begin", kw))

    def record_success(self, envelope):
        self.events.append(("success", envelope))

    def record_failure(self, **kw):
        self.events.append(("failure", kw))


def _engine_shell(diffusion_model=True):
    cls = _GEN.ResponseGenerator
    rg = object.__new__(cls)
    rg._stop = False
    rg.requests = queue.Queue()
    rg._cancel_lock = threading.Lock()
    rg._cancelled = set()
    rg._ready = threading.Event()
    rg._ready.set()
    rg._load_error = None
    rg.model = types.SimpleNamespace(
        diffusion=diffusion_model,
        config=types.SimpleNamespace(canvas_length=256))
    rg.processor = _Tok()
    rg.config = types.SimpleNamespace(eos_token_id=1)
    rg.tokenizer = types.SimpleNamespace()
    rg._tokenizer_lock = threading.Lock()
    rg.prefill_step_size = 512
    thread = threading.Thread(target=rg._run_diffusion, daemon=True)
    thread.start()
    return rg, thread


@pytest.fixture
def app(monkeypatch):
    saved_routes = sp_common._snapshot_routes(_APP.app)
    saved = (_APP.get_cached_model, _RUNTIME.response_generator,
             _RUNTIME.metrics)
    cls = _GEN.ResponseGenerator
    monkeypatch.setattr(cls, "_generate_diffusion", cls._generate_diffusion)
    install_engine_jobs()
    monkeypatch.setattr(engine_mod, "StructuredReader", _Reader)
    monkeypatch.setattr(engine_mod, "engine_scope",
                        lambda model, seed: contextlib.nullcontext())
    monkeypatch.setattr(diffusion, "is_diffusion_model",
                        lambda m: bool(getattr(m, "diffusion", False)))
    _Reader.built.clear()
    _Reader.mode = "ok"
    engines = []
    state = types.SimpleNamespace(loaded=[], engines=engines)

    def use(rg_or_guard=None, *, diffusion_model=True, cfg=None):
        rg, thread = _engine_shell(diffusion_model)
        engines.append((rg, thread))
        _RUNTIME.response_generator = (rg if rg_or_guard is None
                                       else rg_or_guard(rg))
        state.rg = rg
        route.install_systemone_route(cfg or SystemoneCfg())
        return TestClient(_APP.app)

    def fake_get_cached_model(model, *a, **k):
        state.loaded.append(model)
        return None, None, None

    _APP.get_cached_model = fake_get_cached_model
    _RUNTIME.metrics = _Metrics()
    state.use = use
    state.metrics = _RUNTIME.metrics
    yield state
    for rg, thread in engines:
        rg._stop = True
        rg.requests.put(None)
        thread.join(timeout=5)
    sp_common._restore_routes(_APP.app, saved_routes)
    (_APP.get_cached_model, _RUNTIME.response_generator,
     _RUNTIME.metrics) = saved


def _ticket(**extra):
    body = {
        "model": "jev-latest",
        "state": {"ticket": "Everything is down and we have a demo at noon."},
        "questions": {
            "urgent": {"type": "noul",
                       "instructions": "Does the customer need a reply within the hour?"},
            "team": {"type": "choice",
                     "criteria": {"billing": "money", "infra": "outages"}},
            "severity": {"type": "score", "criteria": ["low", "mid", "high"]},
        },
    }
    body.update(extra)
    return body


# model policy

@pytest.fixture
def registry():
    serving.clear_resolved_models()

    def register(models, *, aliases=None, defaults=None, systemone=None):
        server = {"host": "127.0.0.1", "port": 8080}
        if defaults:
            server["defaults"] = {"model": defaults}
        if systemone:
            server["systemone"] = systemone
        cfg = build_config({
            "server": server,
            "profiles": {"fast": {"sampling": {"temperature": 0.1}}},
            "models": {m: {"path": f"/abs/{m}.gguf"} for m in models},
            "aliases": aliases or {},
        })
        serving.register_resolved_models(cfg)
    yield register
    serving.clear_resolved_models()


def test_without_a_config_the_model_field_passes_verbatim():
    assert route._pick_model("jev-latest", None, SystemoneCfg()) == "jev-latest"
    assert route._pick_model(None, None, SystemoneCfg()) == ""


def test_a_served_id_alias_or_profile_address_resolves(registry):
    registry(["dg", "other"], aliases={"diffuse": "dg@fast"})
    cfg = SystemoneCfg(model="other")
    assert route._pick_model("dg", None, cfg) == "dg"
    assert route._pick_model("diffuse", None, cfg) == "diffuse"
    assert route._pick_model("dg@fast", None, cfg) == "dg@fast"


def test_an_unknown_or_absent_model_falls_back_in_order(registry):
    registry(["dg", "other"], defaults="other")
    for field in ("jev-latest", None, ""):
        assert route._pick_model(field, None, SystemoneCfg(model="dg")) == "dg"
        assert route._pick_model(field, None, SystemoneCfg()) == ""
    serving.clear_resolved_models()
    registry(["dg"])
    assert route._pick_model("jev-latest", None, SystemoneCfg()) == ""


def test_the_body_profile_reaches_the_last_step(registry):
    registry(["dg"])
    with pytest.raises(serving.UnknownProfile):
        route._pick_model("jev-latest", "nope", SystemoneCfg())


def test_a_mistyped_name_keeps_its_404_and_an_absent_one_its_own_error(registry):
    registry(["dg", "other"])
    with pytest.raises(serving.ModelNotFound) as info:
        route._pick_model("dgg", None, SystemoneCfg())
    assert "dgg" in str(info.value)
    with pytest.raises(serving.NoModelSpecified):
        route._pick_model(None, None, SystemoneCfg())


def test_the_404_body_names_what_the_client_sent(app, registry):
    registry(["dg", "other"])
    client = app.use()
    r = client.post("/v1/systemone", json=_ticket(model="dgg"))
    assert r.status_code == 404, r.text
    assert "dgg" in r.text
    assert app.metrics.events[-1][0] == "failure"


def test_the_fallback_model_is_the_one_loaded_and_echoed(app, registry):
    registry(["dg", "other"], systemone={"model": "dg"})
    client = app.use()
    r = client.post("/v1/systemone", json=_ticket())
    assert r.status_code == 200, r.text
    assert app.loaded == ["dg"]
    assert r.json()["model"] == "dg"


def test_the_live_config_wins_over_the_installed_settings(app, registry):
    registry(["dg", "other"], systemone={"model": "dg", "max_questions": 2})
    client = app.use(cfg=SystemoneCfg(model="other"))
    r = client.post("/v1/systemone", json=_ticket())
    assert r.status_code == 422 and "at most 2" in r.text
    serving.clear_resolved_models()
    registry(["dg", "other"], systemone={"model": "other"})
    r = client.post("/v1/systemone", json=_ticket())
    assert r.status_code == 200 and app.loaded[-1] == "other"


# status codes

@pytest.mark.parametrize("kwargs,status,kind", [
    ({"content": b"{", "headers": {"content-type": "application/json"}},
     400, "invalid_request_error"),
    ({"json": [1, 2]}, 400, "invalid_request_error"),
    ({"json": _ticket(images=["data:image/png;base64,AA"])}, 400,
     "invalid_request_error"),
    ({"files": {"image": ("a.png", b"x")}}, 400, "invalid_request_error"),
    ({"json": {"state": "s", "questions": {}}}, 422, "validation_error"),
    ({"json": _ticket(seed="x")}, 422, "validation_error"),
    ({"json": {"questions": {"a": {"type": "noul"}}}}, 422, "validation_error"),
])
def test_bad_requests_get_their_status(app, kwargs, status, kind):
    client = app.use()
    r = client.post("/v1/systemone", **kwargs)
    assert r.status_code == status, r.text
    assert r.json()["error"]["type"] == kind
    assert [e[0] for e in app.metrics.events] == ["begin", "failure"]
    assert _Reader.built == []


def test_images_are_refused_by_name(app):
    r = app.use().post("/v1/systemone",
                       json=_ticket(images=["data:image/png;base64,AA"]))
    assert "text-only" in r.json()["error"]["message"]


def test_a_non_diffusion_model_is_a_400(app):
    client = app.use(diffusion_model=False)
    r = client.post("/v1/systemone", json=_ticket())
    assert r.status_code == 400
    assert "diffusion" in r.text


def test_a_decision_past_its_deadline_is_a_504(app, monkeypatch):
    monkeypatch.setattr(_GEN, "get_token_queue_timeout", lambda: 0.1)
    _Reader.mode = "slow"
    r = app.use().post("/v1/systemone", json=_ticket())
    assert r.status_code == 504
    assert r.json()["error"]["type"] == "timeout"


def test_an_engine_failure_is_a_500_naming_the_type(app):
    _Reader.mode = "boom"
    r = app.use().post("/v1/systemone", json=_ticket())
    assert r.status_code == 500
    body = r.json()["error"]
    assert body["type"] == "server_error" and "RuntimeError" in body["message"]
    assert app.metrics.events[-1][0] == "failure"


# body shapes

def test_answers_have_the_jev_shapes(app):
    r = app.use().post("/v1/systemone", json=_ticket())
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"model", "answers", "usage", "diagnostics"}
    urgent, team, severity = (body["answers"][k] for k in ("urgent", "team", "severity"))
    assert set(urgent) == {"type", "noul"} and 0.0 <= urgent["noul"] <= 1.0
    assert set(team) == {"type", "choice", "probabilities", "confidence"}
    assert abs(sum(team["probabilities"].values()) - 1.0) < 1e-6
    assert set(severity) == {"type", "score", "legend", "probabilities", "confidence"}
    assert severity["legend"] == {"0": "low", "1": "mid", "2": "high"}
    expect = sum(int(k) * p for k, p in severity["probabilities"].items())
    assert abs(severity["score"] - expect) < 1e-9
    assert body["diagnostics"]["engine"] == "gmlx"
    assert body["usage"]["input_tokens"] == body["diagnostics"]["prompt_tokens"]
    assert body["usage"]["output_tokens"] > 0


def test_a_skipped_question_answers_null(app):
    body = _ticket()
    body["questions"]["why"] = {"type": "noul", "ask_if": {"urgent": ["never"]}}
    r = app.use().post("/v1/systemone", json=body)
    assert r.status_code == 422
    body["questions"]["why"]["ask_if"] = {"team": ["billing"]}
    r = app.use().post("/v1/systemone", json=body)
    answers = r.json()["answers"]
    team_label = answers["team"]["choice"]
    assert (answers["why"] is None) == (team_label != "billing")


def test_auto_samples_report_the_extension(app):
    r = app.use().post("/v1/systemone", json=_ticket(auto_threshold=0.0))
    policy = r.json()["diagnostics"]["samples"]["policy"]
    assert policy["mode"] == "auto" and policy["extended"] is True
    r = app.use().post("/v1/systemone", json=_ticket(auto_threshold=99.0))
    assert r.json()["diagnostics"]["samples"]["policy"]["extended"] is False


def test_the_same_seed_gives_the_same_answers(app):
    client = app.use()
    a = client.post("/v1/systemone", json=_ticket(seed=7, samples=3)).json()
    b = client.post("/v1/systemone", json=_ticket(seed=7, samples=3)).json()
    c = client.post("/v1/systemone", json=_ticket(seed=8, samples=3)).json()
    assert a["answers"] == b["answers"]
    assert a["diagnostics"]["samples"]["tops"] == b["diagnostics"]["samples"]["tops"]
    assert a["diagnostics"]["samples"]["tops"] != c["diagnostics"]["samples"]["tops"]


def test_think_runs_at_the_served_canvas(app):
    client = app.use(cfg=SystemoneCfg(canvas=32))
    r = client.post("/v1/systemone", json=_ticket(think=16))
    assert r.status_code == 200, r.text
    thinks = [c for c in _Reader.built[0].calls if c[0] == "think"]
    assert thinks == [("think", 16, 32)]


# admission, logging, holds

def test_a_request_over_the_context_budget_is_refused_before_queueing(app, monkeypatch):
    monkeypatch.setattr(_GEN, "get_configured_context_limit", lambda: 400)
    client = app.use()
    r = client.post("/v1/systemone", json=_ticket())
    assert r.status_code == 200, r.text
    r = client.post("/v1/systemone", json=_ticket(think=4096))
    assert r.status_code == 400
    assert len(_Reader.built) == 1
    assert not any(c[0] == "think" for c in _Reader.built[0].calls)


def test_the_logs_carry_the_prompt_count_and_one_decision_line(app, caplog):
    caplog.set_level(logging.INFO)
    r = app.use().post("/v1/systemone", json=_ticket())
    assert r.status_code == 200
    prefill = [rec.getMessage() for rec in caplog.records
               if "Prefill started" in rec.getMessage()]
    assert len(prefill) == 1
    n = int(re.search(r"prompt_tokens=(\d+)", prefill[0]).group(1))
    assert n >= r.json()["usage"]["input_tokens"] > 0
    lines = [rec.getMessage() for rec in caplog.records
             if rec.getMessage().startswith("systemone: ")]
    assert len(lines) == 1 and "urgent=" in lines[0] and "reads=" in lines[0]
    kinds = [e[0] for e in app.metrics.events]
    assert kinds == ["begin", "success"]
    envelope = app.metrics.events[-1][1]
    assert envelope["endpoint"] == "/v1/systemone"
    from gmlx.serve.patches.observability import _format_timing_line

    assert _format_timing_line(envelope).startswith("[req] ")


class _Hold:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


def test_a_real_hold_is_released_and_a_missing_one_is_fine(app):
    hold = _Hold()
    r = app.use(lambda rg: _GenerationGuard(rg, hold)).post(
        "/v1/systemone", json=_ticket())
    assert r.status_code == 200, r.text
    assert hold.released == 1
    r = app.use(lambda rg: _GenerationGuard(rg, None)).post(
        "/v1/systemone", json=_ticket())
    assert r.status_code == 200, r.text


def test_the_route_registers_both_paths_idempotently(app):
    app.use()
    route.install_systemone_route(SystemoneCfg())
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/systemone") == 1 and paths.count("/systemone") == 1


def test_a_disconnect_sets_the_stop_event():
    class _Req:
        async def is_disconnected(self):
            return True

    stop = threading.Event()
    asyncio.run(route._watch_disconnect(_Req(), stop))
    assert stop.is_set()
