#!/usr/bin/env python3
"""Config-driven HTTP-surface patches: sampling-profile injection, /v1/models
override, HF gate, runtime-snapshot enrichment, pool-aware /unload, resolver
error handlers. CPU-only - the pure helpers run directly and the installs are
checked by their effect on the (snapshot/restored) mlx-vlm modules; no live
server, no model load."""
from __future__ import annotations

import importlib
import os
import sys
import time
import types

import pytest

pytest.importorskip("mlx_vlm")

from gmlx import server_patches as sp  # noqa: E402
from gmlx.server_patches import _common as sp_common  # noqa: E402
from gmlx.server_patches import chat_behavior as sp_chat  # noqa: E402
from gmlx.server_patches import hardening as sp_hardening  # noqa: E402
from gmlx.server_patches import request_flow as sp_flow  # noqa: E402
from gmlx.server_patches import routes as sp_routes  # noqa: E402
from gmlx.server_patches import sampling as sp_sampling  # noqa: E402
from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx.config import ResolvedModel, build_config  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_UTILS = importlib.import_module("mlx_vlm.utils")
_PKG = importlib.import_module("mlx_vlm.server")


@pytest.fixture(autouse=True)
def _restore_mlxvlm():
    """Snapshot every mlx-vlm seam these patches mutate, restore after each test."""
    fastapi_app = _APP.app
    saved = {
        "build_gen_args": _APP._build_gen_args,
        "snapshot": _APP._server_runtime_snapshot,
        "get_model_path": _UTILS.get_model_path,
        "routes": list(fastapi_app.router.routes),
        "handlers": dict(fastapi_app.exception_handlers),
        "deps": getattr(getattr(_APP, "_protocol_deps", None), "build_gen_args", None),
        "pool": getattr(_PKG, "_kq_residency_pool", None),
        # Middleware installs (auth / host guard) append to user_middleware and
        # set app.state flags; CORS hardening mutates a Middleware's kwargs in
        # place - snapshot all three or one test's auth poisons the rest.
        "middleware": list(fastapi_app.user_middleware),
        "mw_kwargs": [(m, dict(getattr(m, "kwargs", {}) or {}))
                      for m in fastapi_app.user_middleware],
    }
    openai = sys.modules.get("mlx_vlm.server.openai")
    anthropic = sys.modules.get("mlx_vlm.server.anthropic")
    saved["openai_bga"] = getattr(openai, "_build_gen_args", None)
    saved["anthropic_bga"] = getattr(anthropic, "_build_gen_args", None)
    apc = sys.modules.get("mlx_vlm.apc") or importlib.import_module("mlx_vlm.apc")
    saved["apc_harvest"] = apc.harvest_blocks_from_batch_cache
    saved["apc_lone_flag"] = getattr(apc, "_kq_lone_harvest", False)
    gen = importlib.import_module("mlx_vlm.server.generation")
    saved["to_template_kwargs"] = gen.GenerationArguments.to_template_kwargs
    saved["make_sampler"] = gen.ResponseGenerator._make_sampler
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    saved["stream_chunk_dump"] = schemas.ChatStreamChunk.model_dump_json
    saved["stopping_call"] = _UTILS.StoppingCriteria.__call__
    serving.clear_resolved_models()
    yield
    _UTILS.StoppingCriteria.__call__ = saved["stopping_call"]
    apc.harvest_blocks_from_batch_cache = saved["apc_harvest"]
    apc._kq_lone_harvest = saved["apc_lone_flag"]
    gen.GenerationArguments.to_template_kwargs = saved["to_template_kwargs"]
    gen.ResponseGenerator._make_sampler = saved["make_sampler"]
    schemas.ChatStreamChunk.model_dump_json = saved["stream_chunk_dump"]
    _APP._build_gen_args = saved["build_gen_args"]
    _APP._server_runtime_snapshot = saved["snapshot"]
    _UTILS.get_model_path = saved["get_model_path"]
    fastapi_app.router.routes[:] = saved["routes"]
    fastapi_app.exception_handlers.clear()
    fastapi_app.exception_handlers.update(saved["handlers"])
    if getattr(_APP, "_protocol_deps", None) is not None and saved["deps"] is not None:
        _APP._protocol_deps.build_gen_args = saved["deps"]
    if openai is not None:
        openai._build_gen_args = saved["openai_bga"]
    if anthropic is not None:
        anthropic._build_gen_args = saved["anthropic_bga"]
    if saved["pool"] is None:
        if hasattr(_PKG, "_kq_residency_pool"):
            delattr(_PKG, "_kq_residency_pool")
    else:
        _PKG._kq_residency_pool = saved["pool"]
    fastapi_app.user_middleware[:] = saved["middleware"]
    for m, kw in saved["mw_kwargs"]:
        if getattr(m, "kwargs", None) is not None:
            m.kwargs.clear()
            m.kwargs.update(kw)
    fastapi_app.middleware_stack = None       # force a rebuild from the restored list
    for flag in (sp_hardening._AUTH_FLAG, sp_hardening._HOST_GUARD_FLAG):
        if hasattr(fastapi_app.state, flag):
            delattr(fastapi_app.state, flag)
    serving.clear_resolved_models()


def _spec(**sampling):
    return ResolvedModel(id="m", path="/p", sampling=sampling, load={}, cache={},
                         system=None, speculative=False, mmproj=None,
                         draft_gguf=None, pin=False, ttl_s=None)


# 1. sampling injection (pure)
def test_inject_overrides_unset_keeps_client_set():
    args = types.SimpleNamespace(temperature=0.7, top_p=0.95, top_k=5, max_tokens=512)
    request = types.SimpleNamespace(model_fields_set={"top_k"})   # client set top_k
    spec = _spec(temperature=0.2, top_p=0.9, top_k=99, max_tokens=2048)
    sp_sampling._inject_profile_sampling(args, request, spec)
    assert args.temperature == 0.2       # injected (unset)
    assert args.top_p == 0.9             # injected (unset)
    assert args.top_k == 5               # kept (client set)
    assert args.max_tokens == 2048       # injected (unset)


def test_inject_max_tokens_alias_respects_max_output_tokens():
    args = types.SimpleNamespace(max_tokens=512)
    request = types.SimpleNamespace(model_fields_set={"max_output_tokens"})
    sp_sampling._inject_profile_sampling(args, request, _spec(max_tokens=2048))
    assert args.max_tokens == 512        # responses API set it -> not overridden


# 1b. ignore-eos: forced-length decode
def test_install_ignore_eos_suppresses_stop():
    crit = _UTILS.StoppingCriteria([7, 8])
    assert crit(7) is True               # baseline: 7 is an eos id -> stop
    sp.install_ignore_eos()
    assert crit(7) is False              # patched: EOS never stops decode
    assert crit(8) is False
    assert crit(123) is False
    sp.install_ignore_eos()              # idempotent
    assert crit(7) is False


def test_inject_noop_without_spec_or_sampling():
    args = types.SimpleNamespace(temperature=0.7)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, None)
    sp_sampling._inject_profile_sampling(args, request, _spec())          # empty sampling
    assert args.temperature == 0.7


def test_inject_skips_unknown_arg_attr():
    args = types.SimpleNamespace(temperature=0.7)               # no top_p attr
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(top_p=0.5))
    assert not hasattr(args, "top_p")


def test_inject_thinking_budget_from_profile():
    # off by default: GenerationArguments.thinking_budget is None; a profile/model
    # value seeds it when the client didn't ask.
    args = types.SimpleNamespace(thinking_budget=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(thinking_budget=1024))
    assert args.thinking_budget == 1024


def test_inject_thinking_budget_request_wins():
    args = types.SimpleNamespace(thinking_budget=256)           # client sent 256
    request = types.SimpleNamespace(model_fields_set={"thinking_budget"})
    sp_sampling._inject_profile_sampling(args, request, _spec(thinking_budget=1024))
    assert args.thinking_budget == 256                          # not clobbered


def test_inject_thinking_budget_off_by_default():
    args = types.SimpleNamespace(thinking_budget=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._inject_profile_sampling(args, request, _spec(temperature=0.2))  # no budget
    assert args.thinking_budget is None                         # stays off


# 1b. thinking_budget enforcement fix (generate-<think> models)
class _FakeThinkTok:
    """Encodes the three control strings the criteria resolves."""
    _MAP = {"<think>": [99], "</think>": [100], "\n": [10]}

    def encode(self, text, add_special_tokens=True):
        return self._MAP.get(text, [7])


def _drive(criteria, tokens):
    """Feed token ids; return the forced id (or None) the criteria emits each step."""
    return [criteria(t) for t in tokens]


def _armed(budget, prompt_open):
    cls = sp_chat._armed_thinking_budget_criteria_cls()
    return cls(tokenizer=_FakeThinkTok(), thinking_budget=budget,
               thinking_start_token="<think>", thinking_end_token="</think>",
               enable_thinking=True, prompt_open_thinking=prompt_open)


def test_armed_criteria_caps_generated_think():
    # prompt did NOT pre-fill <think> (Qwen3 case); the model generates it.
    c = _armed(3, prompt_open=False)
    assert c.in_thinking is False                       # not started in a block
    forced = _drive(c, [99, 1, 2, 3, 4, 5, 6])          # <think> then 6 words
    assert 10 in forced and 100 in forced              # forced \n then </think>
    assert forced.index(10) < forced.index(100)


def test_armed_criteria_caps_prefilled_think():
    # prompt pre-filled <think> (GLM-5.2 case): counting starts immediately.
    c = _armed(2, prompt_open=True)
    assert c.in_thinking is True
    forced = _drive(c, [1, 2, 3, 4, 5])
    assert 100 in forced                               # forced close


def test_armed_criteria_never_forces_non_thinking_answer():
    # thinking enabled + budget set, but the model never opens a <think> ->
    # no token is ever counted, so nothing is force-closed (no corruption).
    c = _armed(2, prompt_open=False)
    forced = _drive(c, [1, 2, 3, 4, 5, 6, 7, 8])
    assert all(f is None for f in forced)


def test_armed_criteria_reset_restores_prompt_seed():
    c = _armed(2, prompt_open=False)
    c.in_thinking = True
    c.reset_thinking_state()
    assert c.in_thinking is False                       # back to the prompt seed


def test_prompt_tail_opens_thinking_cases():
    pairs = (("<think>", "</think>"),)
    f = sp_chat._prompt_tail_opens_thinking
    assert f("", pairs) is False
    assert f("plain prompt, no markers", pairs) is False
    assert f("<|im_start|>assistant\n<think>\n", pairs) is True   # Qwen3.6 pre-fill
    assert f("x<think>\n\n</think>\n\n", pairs) is False          # thinking off
    assert f("a<think>x</think>b<think>", pairs) is True          # last pair open
    assert f(None, pairs) is False


def test_stream_thinking_seed_reseeds_from_prompt():
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    cls = rs.ThinkingStreamState
    original_init = cls.__init__
    try:
        sp.install_stream_thinking_seed()
        assert getattr(cls.__init__, sp_chat._STREAM_SEED_FLAG, False)
        patched = cls.__init__
        sp.install_stream_thinking_seed()                    # idempotent
        assert cls.__init__ is patched
        tok = sp_chat._LAST_RENDERED_PROMPT.set("rendered, no thinking scaffold")
        try:
            st = cls(True)               # enable_thinking forced True (7b)
            assert st.in_thinking is False   # gemma-4 default-off: content mode
            sp_chat._LAST_RENDERED_PROMPT.set("<|im_start|>assistant\n<think>\n")
            st = cls(True)
            assert st.in_thinking is True    # Qwen3.6 pre-fill: reasoning first
            st = cls(False)
            assert st.in_thinking is True    # prompt truth beats the flag
            sp_chat._LAST_RENDERED_PROMPT.set(None)
            st = cls(True)
            assert st.in_thinking is True    # no render seen -> stock seed
        finally:
            sp_chat._LAST_RENDERED_PROMPT.reset(tok)
    finally:
        cls.__init__ = original_init


def test_stream_thinking_seed_wraps_render_binding():
    openai_mod = importlib.import_module("mlx_vlm.server.openai")
    rs = importlib.import_module("mlx_vlm.server.responses_state")
    original_fn = openai_mod.apply_chat_template
    original_init = rs.ThinkingStreamState.__init__
    try:
        openai_mod.apply_chat_template = lambda *a, **kw: "tail <think>\n"
        sp.install_stream_thinking_seed()
        wrapped = openai_mod.apply_chat_template
        assert getattr(wrapped, sp_chat._STREAM_SEED_FLAG, False)
        out = wrapped("processor", "config", [])
        assert out == "tail <think>\n"
        assert sp_chat._LAST_RENDERED_PROMPT.get() == out         # stashed
        sp_chat._LAST_RENDERED_PROMPT.set(None)
    finally:
        openai_mod.apply_chat_template = original_fn
        rs.ThinkingStreamState.__init__ = original_init


def test_install_thinking_budget_fix_applies_and_idempotent():
    # Fail-loud guard: asserts the seam bound to the REAL mlx-vlm symbol, so a
    # rename of ResponseGenerator._make_thinking_budget_criteria turns into a CI
    # failure instead of a silent no-op.
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_thinking_budget_criteria
    try:
        sp.install_thinking_budget_fix()
        patched = cls._make_thinking_budget_criteria
        assert patched is not original                       # actually bound
        assert getattr(patched, sp_chat._TBUDGET_FLAG, False)
        sp.install_thinking_budget_fix()                     # idempotent
        assert cls._make_thinking_budget_criteria is patched
    finally:
        cls._make_thinking_budget_criteria = original


def test_make_criteria_honors_budget_when_enable_thinking_false():
    # A configured thinking_budget must arm even when enable_thinking is False:
    # a group/profile config may disable thinking, but the model can still emit
    # <think>, and an explicit budget must cap it. None budget still opts out.
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_thinking_budget_criteria
    try:
        sp.install_thinking_budget_fix()
        make = cls._make_thinking_budget_criteria
        me = types.SimpleNamespace(
            tokenizer=_FakeThinkTok(),
            _thinking_token_ids=lambda args: (99, 100),  # <think>=99, </think>=100
        )
        args = types.SimpleNamespace(
            thinking_budget=8, enable_thinking=False,
            thinking_start_token=None, thinking_end_token=None)
        # generate-style prompt (no open <think>): armed but not seeded in-block
        criteria = make(me, args, [1, 2, 3])
        assert criteria is not None                  # armed despite enable_thinking=False
        assert criteria.in_thinking is False
        # pre-fill prompt ending with an open <think> seeds in_thinking True, even
        # though enable_thinking is False - the cap must still fire (GLM-style).
        criteria2 = make(me, args, [1, 2, 99])
        assert criteria2 is not None and criteria2.in_thinking is True
        args.thinking_budget = None
        assert make(me, args, [1, 2, 3]) is None      # no budget -> still opts out
    finally:
        cls._make_thinking_budget_criteria = original


# 2. gen-args wrapper reference swap
def test_install_wraps_build_gen_args_and_injects():
    def stub(request, processor=None, tenant_id=None):
        return types.SimpleNamespace(temperature=0.7, top_p=0.95, max_tokens=512)

    _APP._build_gen_args = stub
    sp.install_gen_args_profile_injection()
    assert _APP._build_gen_args is not stub                     # wrapped

    spec = _spec(temperature=0.2, top_p=0.9, max_tokens=2048)
    tok = serving.set_active_spec(spec)
    try:
        req = types.SimpleNamespace(model_fields_set=set())
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    assert (args.temperature, args.top_p, args.max_tokens) == (0.2, 0.9, 2048)


def test_install_gen_args_idempotent():
    sp.install_gen_args_profile_injection()
    first = _APP._build_gen_args
    sp.install_gen_args_profile_injection()
    assert _APP._build_gen_args is first


def test_gen_arg_keys_exist_upstream():
    """Drift guard: every injection key must be a declared GenerationArguments
    field and every request-field alias a declared field on some mlx-vlm request
    schema - a rename upstream turns a silent injection no-op into a red X."""
    import dataclasses
    gen = importlib.import_module("mlx_vlm.server.generation")
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    ga_fields = {f.name for f in dataclasses.fields(gen.GenerationArguments)}
    request_fields = set()
    for cls in (schemas.ChatRequest, schemas.AnthropicRequest,
                schemas.OpenAIRequest):
        request_fields |= set(cls.model_fields)
    for key, req_names in sp_sampling._GEN_ARG_REQUEST_FIELDS.items():
        assert key in ga_fields, \
            f"{key!r} is not a GenerationArguments field anymore"
        for name in req_names:
            assert name in request_fields, \
                f"{name!r} is not declared on any mlx-vlm request schema"


def test_inject_profile_sampling_on_real_gen_args():
    """End-to-end over the REAL seam: a real ChatRequest through the real
    _build_gen_args, with the injection installed - profile values land on the
    real GenerationArguments and a client-set field still wins."""
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    gen = importlib.import_module("mlx_vlm.server.generation")
    sp.install_gen_args_profile_injection()
    spec = _spec(temperature=0.15, top_p=0.8, top_k=40, min_p=0.05,
                 max_tokens=777)
    req = schemas.ChatRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], top_k=7)
    tok = serving.set_active_spec(spec)
    try:
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    assert isinstance(args, gen.GenerationArguments)
    assert args.temperature == 0.15          # injected (unset on the request)
    assert args.top_p == 0.8
    assert args.min_p == 0.05
    assert args.max_tokens == 777
    assert args.top_k == 7                   # client set it -> request wins


# 3. /v1/models payload + route override
def _register(doc):
    serving.register_resolved_models(build_config(doc))


def test_models_payload_lists_configured_ids_not_hf():
    _register({"models": {
        "qwen": {"path": "/abs/qwen.gguf"},
        "gemma-vlm": {"path": "/abs/g.gguf", "mmproj": "/abs/mm.gguf"},
    }})
    payload = sp_routes._models_payload()
    ids = {m["id"] for m in payload["data"]}
    assert ids == {"qwen", "gemma-vlm"}
    vlm = next(m for m in payload["data"] if m["id"] == "gemma-vlm")
    assert vlm["vlm"] is True
    assert all(m["resident"] is False for m in payload["data"])  # no pool


def test_models_payload_marks_resident_from_pool():
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf", "pin": True}}})

    class _FakePool:
        def stats(self):
            return {"resident": [{"model_path": "/abs/qwen.gguf", "pinned": True,
                                  "footprint_bytes": 10, "idle_s": 3.0,
                                  "ttl_s": 900}]}

    _PKG._kq_residency_pool = _FakePool()
    m = sp_routes._models_payload()["data"][0]
    assert m["resident"] is True and m["pinned"] is True


def test_models_payload_lists_aliases_as_pickable_entries():
    _register({
        "profiles": {"coder": {"sampling": {"temperature": 0.2}}},
        "models": {"qwen": {"path": "/abs/qwen.gguf", "speculative": False}},
        "aliases": {"big": "qwen", "coder-preset": "qwen@coder"},
    })
    payload = sp_routes._models_payload()
    by_id = {m["id"]: m for m in payload["data"]}
    assert set(by_id) == {"qwen", "big", "coder-preset"}        # aliases listed
    assert by_id["big"]["alias_of"] == "qwen"
    assert by_id["coder-preset"]["alias_of"] == "qwen"
    assert by_id["coder-preset"]["profile"] == "coder"          # baked profile shown
    assert "alias_of" not in by_id["qwen"]                      # real model unmarked


def test_models_payload_marks_default():
    _register({
        "server": {"defaults": {"model": "qwen"}},
        "models": {"qwen": {"path": "/abs/qwen.gguf"},
                   "gemma": {"path": "/abs/g.gguf"}},
    })
    by_id = {m["id"]: m for m in sp_routes._models_payload()["data"]}
    assert by_id["qwen"]["default"] is True
    assert by_id["gemma"]["default"] is False


def test_models_override_registers_single_route():
    sp.install_models_endpoint_override()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/models") == 1
    sp.install_models_endpoint_override()                       # idempotent-ish
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/models") == 1


# 4. HF gate
def test_gate_allows_local_and_gguf(tmp_path):
    calls = []
    orig = lambda p, *a, **k: calls.append(p) or "OK"
    local = tmp_path / "f"
    local.write_text("x")
    assert sp_routes._gate_model_path(str(local), False, orig) == "OK"
    assert sp_routes._gate_model_path("/x/model.gguf", False, orig) == "OK"
    assert len(calls) == 2


def test_gate_blocks_hf_id_when_disabled():
    orig = lambda p, *a, **k: "OK"
    with pytest.raises(sp.HFAccessDisabled):
        sp_routes._gate_model_path("org/model", False, orig)


def test_gate_allows_hf_id_when_cache_on():
    calls = []
    orig = lambda p, *a, **k: calls.append(p) or "OK"
    assert sp_routes._gate_model_path("org/model", True, orig) == "OK"
    assert calls == ["org/model"]


def test_install_hf_gate_sets_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    sp.install_hf_download_gate(hf_cache=True)
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


# 5. runtime-snapshot enrichment
def test_snapshot_enrichment_adds_resident_models():
    _APP._server_runtime_snapshot = lambda: {"loaded_model": "x"}

    class _FakePool:
        def stats(self):
            return {"resident": [{"model_path": "/abs/qwen.gguf", "pinned": False,
                                  "busy": 3, "footprint_bytes": 99,
                                  "idle_s": 1.234, "ttl_s": 900}]}

    _PKG._kq_residency_pool = _FakePool()
    serving._PATH_TO_IDS["/abs/qwen.gguf"] = ["qwen"]
    try:
        sp.install_runtime_snapshot_enrichment()
        snap = _APP._server_runtime_snapshot()
    finally:
        serving._PATH_TO_IDS.pop("/abs/qwen.gguf", None)
    assert snap["loaded_model"] == "x"                          # base preserved
    assert snap["resident_models"][0]["ids"] == ["qwen"]
    assert snap["resident_models"][0]["idle_s"] == 1.2          # rounded
    assert snap["resident_models"][0]["busy"] == 3              # in-flight count


# 6. error handlers + reload + unload route
def test_error_content_dialect_shapes():
    # One condition, two envelopes: OpenAI-style everywhere, Anthropic's
    # {"type": "error", ...} with its fixed taxonomy on /v1/messages.
    openai = sp_common._error_content(
        "/v1/chat/completions", 404, "model_not_found", "no such model",
        available_models=["a"])
    assert openai == {"error": {"type": "model_not_found",
                                "message": "no such model",
                                "available_models": ["a"]}}
    anthropic = sp_common._error_content(
        "/v1/messages", 404, "model_not_found", "no such model")
    assert anthropic["type"] == "error"
    assert anthropic["error"] == {"type": "not_found_error",
                                  "message": "no such model"}
    assert sp_common._error_content("/v1/messages", 500, "server_error",
                                    "x")["error"]["type"] == "api_error"


def test_http_exception_envelope_unwrapped():
    # The residency resolver path raises HTTPException carrying the unified
    # {"error": {...}} detail; the app-level handler must serve that body
    # directly (no {"detail": ...} wrapper) and wrap plain-string details.
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    app = _APP.app
    if not any(getattr(r, "path", None) == "/test/raise-envelope"
               for r in app.router.routes):
        @app.get("/test/raise-envelope")
        async def _raise_envelope():
            raise HTTPException(status_code=404, detail={"error": {
                "type": "model_not_found", "message": "no such model",
                "available_models": ["a"]}})

        @app.get("/test/raise-string")
        async def _raise_string():
            raise HTTPException(status_code=500, detail="it broke")

    sp.install_resolver_error_handlers()
    client = TestClient(app)
    r = client.get("/test/raise-envelope")
    assert r.status_code == 404
    assert r.json() == {"error": {"type": "model_not_found",
                                  "message": "no such model",
                                  "available_models": ["a"]}}
    r2 = client.get("/test/raise-string")
    assert r2.status_code == 500
    assert r2.json() == {"error": {"type": "server_error",
                                   "message": "it broke"}}


def test_resolver_error_handlers_registered():
    sp.install_resolver_error_handlers()
    handlers = _APP.app.exception_handlers
    assert serving.ModelNotFound in handlers
    assert serving.ModelFileMissing in handlers
    assert serving.UnknownProfile in handlers
    assert sp.HFAccessDisabled in handlers


def test_unload_and_reload_routes_register():
    sp.install_pool_aware_unload()
    sp.install_reload_route(lambda: {"reloaded": 1})
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/unload") == 1
    assert paths.count("/v1/reload") == 1


def test_unload_accepts_body_and_empty_post():
    """Regression: the ``request: Request`` annotation must resolve at module level.
    Under ``from __future__ import annotations`` a locally-imported ``Request`` left
    FastAPI treating ``request`` as a required query param, so every POST 422'd before
    the body was read (caught only at e2e). A route-count check can't see this - POST
    it for real and assert the body is actually consumed."""
    from fastapi.testclient import TestClient

    sp.install_pool_aware_unload()
    client = TestClient(_APP.app)
    # no pool registered -> handler runs (no 422) and reports the absence
    # cleanly, as a 503: the unload cannot be honored without a pool
    r_body = client.post("/unload", json={"model": "m"})
    assert r_body.status_code == 503, r_body.text
    assert r_body.json() == {"status": "error", "message": "no residency pool"}
    r_empty = client.post("/unload")
    assert r_empty.status_code == 200, r_empty.text
    assert r_empty.json()["status"] == "no_model_loaded"


# 6b. /v1/keep - the keep tier (TTL-exempt, LRU-eligible)
class _FakeKeepPool:
    def __init__(self):
        self.kept = []

    def set_keep(self, path, keep):
        self.kept.append((path, keep))


def test_json_content_type_tolerance():
    # `curl -d '{...}'` (every doc example) sends form-encoded; the middleware
    # must rewrite it to application/json so pydantic parses the body instead
    # of 422ing. Multipart (audio uploads) must pass through untouched.
    from fastapi.testclient import TestClient

    app = _APP.app
    if not any(getattr(r, "path", None) == "/test/echo-ct"
               for r in app.router.routes):
        # `dict` (a builtin) survives this module's stringized annotations; a
        # test-local pydantic class would resolve as a query param instead.
        @app.post("/test/echo-ct")
        async def _echo_ct(body: dict):
            return {"model": body.get("model")}

    sp.install_json_content_type_tolerance()
    client = TestClient(app)
    r = client.post("/test/echo-ct", content=b'{"model": "m1"}',
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 200 and r.json() == {"model": "m1"}
    r = client.post("/test/echo-ct", content=b'{"model": "m2"}',
                    headers={"Content-Type": "text/plain"})
    assert r.status_code == 200 and r.json() == {"model": "m2"}
    r = client.post("/test/echo-ct", json={"model": "m3"})   # normal path intact
    assert r.status_code == 200
    r = client.post("/test/echo-ct", files={"file": ("a.txt", b"x")})
    assert r.status_code == 422                              # multipart not rewritten


def test_keep_route_registers():
    sp.install_keep_route()
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/keep") == 1


def test_keep_no_pool_reports_error():
    from fastapi.testclient import TestClient

    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "m"})
    assert r.status_code == 503, r.text
    assert r.json() == {"status": "error", "message": "no residency pool"}


def test_keep_marks_resolved_model(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: None)
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    pool = _FakeKeepPool()
    _PKG._kq_residency_pool = pool
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen", "warm": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "kept", "model": "qwen", "warming": False}
    assert pool.kept == [("/abs/qwen.gguf", True)]


def test_keep_warm_default_spawns_warm(monkeypatch):
    from fastapi.testclient import TestClient

    warmed = []
    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: warmed.append(model_id))
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen"})    # warm omitted -> default True
    assert r.json() == {"status": "kept", "model": "qwen", "warming": True}
    assert warmed == ["qwen"]


def test_keep_false_releases_without_evicting(monkeypatch):
    # A voice session ending releases its hold; the model stays resident
    # under normal LRU/TTL rather than being dumped.
    from fastapi.testclient import TestClient

    warmed = []
    monkeypatch.setattr(sp_routes, "_spawn_keep_warm", lambda model_id: warmed.append(model_id))
    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    pool = _FakeKeepPool()
    _PKG._kq_residency_pool = pool
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "qwen", "keep": False})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "released", "model": "qwen"}
    assert pool.kept == [("/abs/qwen.gguf", False)]
    assert warmed == []                                    # release never warms


# 7. vanilla streaming chunks (exclude_none) - the Open WebUI blank-render fix
def _stream_schemas():
    s = importlib.import_module("mlx_vlm.server.schemas")
    return s.ChatStreamChunk, s.ChatStreamChoice, s.ChatMessage, s.UsageStats, s.GenerationTimings


def test_stream_chunk_drops_null_fields_keeps_content():
    """mlx-vlm serialises content chunks with ``timings: null`` (+ other null
    fields). The patch must strip them so the chunk is byte-vanilla OpenAI while
    the actual delta content survives."""
    Chunk, Choice, Msg, _, _ = _stream_schemas()
    sp.install_vanilla_stream_chunks()
    out = Chunk(model="m", choices=[Choice(
        index=0, finish_reason=None, delta=Msg(role="assistant", content="Hi"))]
    ).model_dump_json()
    assert "null" not in out                       # no timings:null/usage:null/etc
    assert '"timings"' not in out and '"usage"' not in out
    assert '"reasoning"' not in out and '"tool_calls"' not in out
    assert '"content":"Hi"' in out and '"role":"assistant"' in out


def test_stream_chunk_keeps_real_usage_and_timings():
    """The final chunk's populated ``usage``/``timings`` dicts (and an empty
    ``choices: []``) are real values, not ``None`` - they must be preserved."""
    Chunk, _, _, Usage, Timings = _stream_schemas()
    sp.install_vanilla_stream_chunks()
    timings = Timings(**{f: 0.0 for f in Timings.model_fields})
    out = Chunk(model="m", choices=[],
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                timings=timings).model_dump_json()
    assert '"usage"' in out and '"timings"' in out and '"choices":[]' in out


def test_stream_chunk_unblocks_open_webui_usage_merge():
    """Reproduces the exact relay line that dropped every content chunk:
    ``raw_usage.update(chunk.get("timings", {}))`` raised ``TypeError`` on our
    ``timings: null``. After the patch the key is absent, so the merge is a
    no-op instead of a crash."""
    import json

    Chunk, Choice, Msg, _, _ = _stream_schemas()
    sp.install_vanilla_stream_chunks()
    chunk = json.loads(Chunk(model="m", choices=[Choice(
        index=0, delta=Msg(role="assistant", content="Hi"))]).model_dump_json())
    raw_usage = chunk.get("usage", {}) or {}
    raw_usage.update(chunk.get("timings", {}))     # was: {}.update(None) -> TypeError
    assert raw_usage == {}


def test_install_vanilla_stream_chunks_idempotent():
    Chunk, _, _, _, _ = _stream_schemas()
    sp.install_vanilla_stream_chunks()
    once = Chunk.model_dump_json
    sp.install_vanilla_stream_chunks()
    assert Chunk.model_dump_json is once           # not re-wrapped


# 8. aux-route request logging - audio/embeddings bypass the metrics funnel AND
# have their uvicorn access line filtered (they sit in _TIMED_PATHS), so without
# an explicit [req] line they go dark. These assert each emits one.
def test_log_aux_request_format(capsys):
    sp_routes._log_aux_request("/v1/audio/speech", "kokoro", time.monotonic() - 0.5,
                        voice="af_heart", chars=12, bytes=34000)
    line = capsys.readouterr().out.strip()
    assert line.startswith("[req] ")
    assert "/v1/audio/speech kokoro" in line
    assert "voice=af_heart" in line and "chars=12" in line and "bytes=34000" in line
    assert "total=" in line and line.endswith("s")


def test_log_aux_request_failed_status_and_skips_none(capsys):
    sp_routes._log_aux_request("/v1/embeddings", "qwen3-embed", time.monotonic(),
                        status="failed", inputs=None, error="RuntimeError")
    line = capsys.readouterr().out.strip()
    assert "/v1/embeddings qwen3-embed failed" in line
    assert "inputs=" not in line                    # None-valued fields dropped
    assert "error=RuntimeError" in line


def test_speech_route_emits_req_line(monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from gmlx import tts
    monkeypatch.setattr(tts, "run_synthesis",
                        lambda *a, **k: (b"AUDIODATA", "audio/mpeg"))
    sp.install_audio_speech_route("mlx-community/Kokoro-82M-bf16")
    client = TestClient(_APP.app)
    r = client.post("/v1/audio/speech",
                    json={"model": "tts-1", "voice": "af_heart",
                          "input": "hello there"})
    assert r.status_code == 200 and r.content == b"AUDIODATA"
    out = capsys.readouterr().out
    assert "[req]" in out and "/v1/audio/speech" in out
    assert "voice=af_heart" in out and "chars=11" in out and "bytes=9" in out


def test_embeddings_route_emits_req_line(monkeypatch, capsys):
    from fastapi.testclient import TestClient

    from gmlx import embeddings as emb
    payload = {"object": "list", "data": [{"embedding": [0.1, 0.2, 0.3]}],
               "model": "m", "usage": {"prompt_tokens": 7}}
    monkeypatch.setattr(emb, "run_embeddings", lambda *a, **k: payload)
    sp.install_embeddings_route("Qwen/Qwen3-Embedding-0.6B")
    client = TestClient(_APP.app)
    r = client.post("/v1/embeddings",
                    json={"model": "text-embedding-3-small", "input": "hi"})
    assert r.status_code == 200
    out = capsys.readouterr().out
    assert "/v1/embeddings" in out and "inputs=1" in out
    assert "dims=3" in out and "tokens=7" in out


def test_transcriptions_route_emits_req_line(monkeypatch, capsys):
    pytest.importorskip("multipart")               # python-multipart (stt extra)
    from fastapi.testclient import TestClient

    from gmlx import stt
    monkeypatch.setattr(stt, "run_transcription",
                        lambda *a, **k: ({"text": "hi"}, "application/json"))
    sp.install_audio_transcription_route("mlx-community/whisper-large-v3-turbo")
    client = TestClient(_APP.app)
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "whisper-1"})
    assert r.status_code == 200
    out = capsys.readouterr().out
    assert "/v1/audio/transcriptions" in out
    assert "file=a.wav" in out and "in_bytes=4" in out


def test_keep_unknown_model_graceful():
    from fastapi.testclient import TestClient

    _register({"models": {"qwen": {"path": "/abs/qwen.gguf"}}})
    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={"model": "nope"})
    # 404, not 200: a typo'd keep must not read as success (launch checks the
    # status code); the body still names the id for older/other clients.
    assert r.status_code == 404, r.text
    assert r.json() == {"status": "unknown_model", "model": "nope"}


def test_keep_missing_model_field():
    from fastapi.testclient import TestClient

    _PKG._kq_residency_pool = _FakeKeepPool()
    sp.install_keep_route()
    client = TestClient(_APP.app)
    r = client.post("/v1/keep", json={})
    assert r.status_code == 400, r.text
    assert r.json() == {"status": "error", "message": "missing 'model'"}


# 6c. /v1/audio/speech - TTS route (mlx-audio stubbed)
def test_speech_route_absent_without_tts():
    """cfg.tts None => 404 on the speech routes (clients feature-detect TTS
    by probing). mlx-vlm >= 0.6.4 registers its own per-request-model speech
    routes at app build, under both the /v1 path and a bare /audio/speech
    alias - the installer must drop both."""
    sp.install_audio_speech_route(None)
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert "/v1/audio/speech" not in paths
    assert "/audio/speech" not in paths


def test_speech_route_owns_both_aliases_with_tts():
    """With cfg.tts set, OUR endpoint is the only speech route: upstream's
    /audio/speech alias must not survive alongside it (it would serve a
    different, per-request-loaded model)."""
    sp.install_audio_speech_route("mlx-community/Kokoro-82M-bf16")
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert paths.count("/v1/audio/speech") == 1
    assert "/audio/speech" not in paths


def test_speech_route_synthesizes(monkeypatch):
    """POST a real request body (the ``request: Request`` 422 footgun): the
    route must consume JSON and return raw audio bytes with the codec's media
    type, not 422 it away as a query param."""
    from fastapi.testclient import TestClient

    from gmlx import tts

    captured = {}

    def fake_run(text, *, configured_model, model, voice, response_format, speed):
        captured.update(text=text, configured=configured_model, voice=voice,
                        fmt=response_format, speed=speed)
        return b"RIFFfake-wav", "audio/wav"

    monkeypatch.setattr(tts, "run_synthesis", fake_run)
    sp.install_audio_speech_route("mlx-community/Kokoro-82M-bf16")
    client = TestClient(_APP.app)
    r = client.post("/v1/audio/speech", json={
        "model": "tts-1", "input": "hello", "voice": "af_bella",
        "response_format": "wav", "speed": 1.25})
    assert r.status_code == 200, r.text
    assert r.content == b"RIFFfake-wav"
    assert r.headers["content-type"] == "audio/wav"
    assert captured == {"text": "hello", "configured": "mlx-community/Kokoro-82M-bf16",
                        "voice": "af_bella", "fmt": "wav", "speed": "1.25"}


def test_speech_route_maps_request_error_to_4xx(monkeypatch):
    from fastapi.testclient import TestClient

    from gmlx import tts

    def boom(text, **kw):
        raise tts.TTSRequestError(400, "field 'input' is required")

    monkeypatch.setattr(tts, "run_synthesis", boom)
    sp.install_audio_speech_route("mlx-community/Kokoro-82M-bf16")
    client = TestClient(_APP.app)
    r = client.post("/v1/audio/speech", json={"input": ""})
    assert r.status_code == 400
    assert "input" in r.json()["error"]["message"]


def test_models_payload_advertises_tts_when_configured():
    sp.install_models_endpoint_override(tts_model="mlx-community/Kokoro-82M-bf16")
    from fastapi.testclient import TestClient
    client = TestClient(_APP.app)
    data = client.get("/v1/models").json()["data"]
    tts1 = next(m for m in data if m["id"] == "tts-1")
    assert tts1["tts"] is True
    assert tts1["alias_of"] == "mlx-community/Kokoro-82M-bf16"


# embeddings route + models advertisement
EMB_REPO = "mlx-community/all-MiniLM-L6-v2-bf16"


def test_embeddings_route_absent_without_config():
    sp.install_embeddings_route(None)
    paths = [getattr(r, "path", None) for r in _APP.app.router.routes]
    assert "/v1/embeddings" not in paths


def test_embeddings_route_embeds(monkeypatch):
    """POST a real JSON body (the ``request: Request`` 422 footgun): the route
    must consume JSON and return the OpenAI embeddings payload."""
    from fastapi.testclient import TestClient

    from gmlx import embeddings as emb

    captured = {}

    def fake_run(inputs, *, configured_model, model, encoding_format):
        captured.update(inputs=inputs, configured=configured_model,
                        model=model, fmt=encoding_format)
        return {"object": "list",
                "data": [{"object": "embedding", "index": 0,
                          "embedding": [0.1, 0.2]}],
                "model": configured_model,
                "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    monkeypatch.setattr(emb, "run_embeddings", fake_run)
    sp.install_embeddings_route(EMB_REPO)
    client = TestClient(_APP.app)
    r = client.post("/v1/embeddings", json={
        "model": "text-embedding-3-small", "input": ["hi", "yo"],
        "encoding_format": "float"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"][0]["embedding"] == [0.1, 0.2]
    assert body["model"] == EMB_REPO
    assert captured == {"inputs": ["hi", "yo"], "configured": EMB_REPO,
                        "model": "text-embedding-3-small", "fmt": "float"}


def test_embeddings_route_maps_request_error_to_4xx(monkeypatch):
    from fastapi.testclient import TestClient

    from gmlx import embeddings as emb

    def boom(inputs, **kw):
        raise emb.EmbeddingsRequestError(400, "field 'input' is required")

    monkeypatch.setattr(emb, "run_embeddings", boom)
    sp.install_embeddings_route(EMB_REPO)
    client = TestClient(_APP.app)
    r = client.post("/v1/embeddings", json={})
    assert r.status_code == 400
    assert "input" in r.json()["error"]["message"]


def test_embeddings_route_missing_file_is_typed_404(monkeypatch):
    # A deleted embeddings GGUF is disk state, not a server error: the endpoint
    # must answer with a typed 404 naming the config key and the fix - never a
    # raw 500 errno.
    from fastapi.testclient import TestClient

    from gmlx import embeddings as emb

    def boom(inputs, **kw):
        raise FileNotFoundError("[Errno 2] No such file: '/lib/e.gguf'")

    monkeypatch.setattr(emb, "run_embeddings", boom)
    sp.install_embeddings_route(EMB_REPO)
    client = TestClient(_APP.app)
    r = client.post("/v1/embeddings", json={"input": "hi"})
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "model_file_missing"
    assert "server.embeddings" in err["message"]
    assert "sync-models" in err["message"]


def test_models_endpoint_delists_missing_gguf_service(monkeypatch, tmp_path):
    # A GGUF-backed embeddings service whose file is gone must vanish from
    # /v1/models like a chat entry would - and come back with the file.
    import asyncio

    gone = tmp_path / "emb.gguf"
    gone.write_bytes(b"GGUF")
    sp.install_models_endpoint_override(embeddings_model=str(gone))
    route = next(r for r in _APP.app.router.routes
                 if getattr(r, "path", None) == "/v1/models")
    ids = [m["id"] for m in asyncio.run(route.endpoint())["data"]]
    assert "text-embedding-3-small" in ids
    gone.unlink()
    ids = [m["id"] for m in asyncio.run(route.endpoint())["data"]]
    assert "text-embedding-3-small" not in ids


def test_rerank_route_reranks(monkeypatch):
    """POST a real JSON body and get the Cohere/Jina rerank payload back."""
    from fastapi.testclient import TestClient

    from gmlx import rerank as rr

    captured = {}

    def fake_run(query, documents, *, configured_model, model, top_n,
                 instruction, return_documents):
        captured.update(query=query, documents=documents, top_n=top_n)
        return {"model": model or configured_model,
                "results": [{"index": 0, "relevance_score": 0.9,
                             "document": {"text": documents[0]}}],
                "usage": {"total_tokens": 7}}

    monkeypatch.setattr(rr, "run_rerank", fake_run)
    sp.install_rerank_route("/models/rerank.gguf")
    client = TestClient(_APP.app)
    r = client.post("/v1/rerank", json={
        "model": "reranker", "query": "q", "documents": ["a", "b"], "top_n": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["index"] == 0
    assert captured == {"query": "q", "documents": ["a", "b"], "top_n": 1}


def test_rerank_top_k_alias(monkeypatch):
    """Cohere/Jina clients send `top_k`; the route must alias it onto `top_n`
    (and a real `top_n` still wins) so results get limited."""
    from fastapi.testclient import TestClient

    from gmlx import rerank as rr

    captured = {}

    def fake_run(query, documents, *, configured_model, model, top_n,
                 instruction, return_documents):
        captured["top_n"] = top_n
        docs = documents[: int(top_n)] if top_n else documents
        return {"model": model or configured_model,
                "results": [{"index": i, "relevance_score": 0.9}
                            for i in range(len(docs))],
                "usage": {"total_tokens": 7}}

    monkeypatch.setattr(rr, "run_rerank", fake_run)
    sp.install_rerank_route("/models/rerank.gguf")
    client = TestClient(_APP.app)
    r = client.post("/v1/rerank", json={
        "query": "q", "documents": ["a", "b", "c"], "top_k": 2})
    assert r.status_code == 200, r.text
    assert captured["top_n"] == 2
    assert len(r.json()["results"]) == 2
    r2 = client.post("/v1/rerank", json={
        "query": "q", "documents": ["a", "b", "c"], "top_n": 1, "top_k": 2})
    assert r2.status_code == 200 and captured["top_n"] == 1


def test_install_server_patches_wires_every_aux_route():
    """Regression: the per-route installers all work in isolation, but a route is
    only live if ``install_server_patches`` actually calls it - ``/v1/rerank`` was
    implemented yet never wired in, so it 404'd at runtime. Drive the aggregate
    installer and assert every configured aux endpoint is registered."""
    cfg = build_config({"server": {
        "embeddings": "/models/embed.gguf",
        "rerank": "/models/rerank.gguf",
        "stt": "whisper-turbo",
    }})
    sp.install_server_patches(cfg, reload_fn=lambda *a, **k: None)
    paths = {getattr(r, "path", None) for r in _APP.app.router.routes}
    assert {"/v1/embeddings", "/v1/rerank",
            "/v1/audio/transcriptions", "/v1/audio/translations"} <= paths


def test_models_payload_advertises_embeddings_when_configured():
    sp.install_models_endpoint_override(embeddings_model=EMB_REPO)
    from fastapi.testclient import TestClient
    client = TestClient(_APP.app)
    data = client.get("/v1/models").json()["data"]
    emb_entry = next(m for m in data if m["id"] == "text-embedding-3-small")
    assert emb_entry["embeddings"] is True
    assert emb_entry["alias_of"] == EMB_REPO


def test_models_service_entries_share_chat_schema_and_hide_paths(tmp_path):
    # Service entries must be indexable exactly like chat entries (naive
    # consumers do m["resident"]) and alias_of must never leak a local path.
    from fastapi.testclient import TestClient
    gguf = tmp_path / "Qwen3-Embedding-0.6B.Q6_K.gguf"
    gguf.write_bytes(b"GGUF")
    sp.install_models_endpoint_override(
        stt_model="mlx-community/whisper-turbo",
        embeddings_model=str(gguf))
    data = TestClient(_APP.app).get("/v1/models").json()["data"]
    emb = next(m for m in data if m["id"] == "text-embedding-3-small")
    stt = next(m for m in data if m["id"] == "whisper-1")
    for entry in (emb, stt):
        for key in ("resident", "pinned", "speculative", "vlm", "profile",
                    "family", "default", "created", "owned_by"):
            assert key in entry
    assert emb["alias_of"] == "Qwen3-Embedding-0.6B.Q6_K.gguf"  # basename only
    assert stt["alias_of"] == "mlx-community/whisper-turbo"     # repo id intact


def test_auto_docs_routes_removed():
    from fastapi.testclient import TestClient
    sp.install_auto_docs_removal()
    client = TestClient(_APP.app)
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_model_not_found_str_is_plain():
    # ModelNotFound subclasses KeyError, whose __str__ is repr() - that
    # double-quoted the message in every HTTP error body built from str(exc).
    from gmlx import server_bridge_vlm as serving
    msg = str(serving.ModelNotFound("default", ["a", "b"]))
    assert msg.startswith("unknown model id 'default'")
    assert not msg.startswith('"')


# 7. XTC sampling injection
def test_attach_xtc_noop_without_request_or_profile():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(model_fields_set=set())
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is None


def test_attach_xtc_appends_processor_from_request_extras():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability=1.0, xtc_threshold=0.2)
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is not None and len(args.logits_processors) == 1
    # functional: prob=1.0 always triggers; threshold 0.2 with probs ~[.6,.3,.1]
    # masks the top token, so argmax moves to the runner-up.
    import math

    import mlx.core as mx
    logits = mx.log(mx.array([[0.6, 0.3, 0.1]]))
    out = args.logits_processors[0](mx.array([0]), logits)
    assert int(mx.argmax(out, axis=-1).item()) == 1
    assert math.isinf(float(out[0, 0].item()))


def test_attach_xtc_profile_fallback_and_request_precedence():
    spec = _spec(xtc_probability=1.0, xtc_threshold=0.3)
    token = serving.set_active_spec(spec)
    try:
        args = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args, types.SimpleNamespace(), None)
        assert args.logits_processors and len(args.logits_processors) == 1
        # an explicit client 0.0 wins over the profile and disables XTC
        args2 = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args2, types.SimpleNamespace(xtc_probability=0.0), None)
        assert args2.logits_processors is None
    finally:
        serving.reset_active_spec(token)


def test_attach_xtc_string_zero_disables():
    # extra="allow" preserves raw JSON types: a client's "0" (string) is truthy,
    # but must still disable XTC after coercion - the live bug this pins down.
    for raw in ("0", "0.0", 0, 0.0):
        args = types.SimpleNamespace(logits_processors=None)
        sp_sampling._attach_xtc(args, types.SimpleNamespace(xtc_probability=raw), None)
        assert args.logits_processors is None, f"xtc_probability={raw!r}"


def test_attach_xtc_string_prob_attaches():
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability="0.5", xtc_threshold="0.2")
    sp_sampling._attach_xtc(args, request, None)
    assert args.logits_processors is not None and len(args.logits_processors) == 1


def test_attach_xtc_garbage_prob_rejects_400():
    # matches the neighboring coercion behavior (_sampling_float): typed 400,
    # never a 500 out of the handler, and args stay untouched.
    from fastapi import HTTPException
    args = types.SimpleNamespace(logits_processors=None)
    request = types.SimpleNamespace(xtc_probability="lots")
    with pytest.raises(HTTPException) as ei:
        sp_sampling._attach_xtc(args, request, None)
    assert ei.value.status_code == 400
    assert args.logits_processors is None


def test_xtc_special_tokens_dedup_and_defensive():
    class _Tok:
        eos_token_id = 7

        def encode(self, s, add_special_tokens=True):
            return [7]

    assert sp_sampling._xtc_special_tokens(types.SimpleNamespace(tokenizer=_Tok())) == [7]
    assert sp_sampling._xtc_special_tokens(None) == []

    class _IntEosTok(_Tok):
        # regression (live server): TokenizersBackend exposes eos_token_ids as
        # a bare int - iterating it raised "'int' object is not iterable"
        eos_token_ids = 9

    assert sp_sampling._xtc_special_tokens(
        types.SimpleNamespace(tokenizer=_IntEosTok())) == [7, 9]


def test_install_xtc_wraps_and_stacks_with_profile_injection():
    sp.install_gen_args_profile_injection()
    sp.install_xtc_sampling()
    fn = _APP._build_gen_args
    assert getattr(fn, sp_common._PATCH_FLAG, False)      # carried forward
    assert getattr(fn, sp_sampling._XTC_FLAG, False)
    sp.install_xtc_sampling()                      # idempotent
    assert _APP._build_gen_args is fn


# 7a2. top_k / min_p aware batch sampler (the historical dropped-top_k bug class)
def _kept_ids(sampler, probs):
    """Vocab ids surviving the sampler's filter for one row of probs, plus the
    masked [1, k] logits (sorted desc by prob)."""
    import mlx.core as mx
    logits = mx.log(mx.array([probs]))
    masked, part, order = sampler._filtered(logits)
    kept = []
    for j in range(masked.shape[-1]):
        if float(masked[0, j].item()) != float("-inf"):
            kept.append(int(part[0, int(order[0, j].item())].item()))
    return kept, masked


def test_fast_sampler_hierarchical_topk_matches_flat():
    # Large vocabs route _filtered's top-k through the hierarchical id
    # selector; the surviving id SET must equal the flat argpartition's
    # (order within the set is re-sorted downstream either way).
    import mlx.core as mx
    for v, seed in ((201088, 0), (200005, 1), (131072, 2)):
        lp = mx.random.normal((1, v), key=mx.random.key(seed))
        lp = lp.astype(mx.float32)
        mx.eval(lp)
        hier = set(sp_sampling._topk_ids(lp, 20)[0].tolist())
        flat = set(mx.argpartition(-lp, kth=19, axis=-1)[:, :20][0].tolist())
        assert hier == flat


def test_fast_sampler_masking():
    S = sp_sampling._FastPositionedSampler
    probs = [0.4, 0.3, 0.2, 0.1]
    # top_k=2: exactly the two most probable survive
    kept, _ = _kept_ids(S(temperature=1.0, top_k=2), probs)
    assert kept == [0, 1]
    # top_p=0.5: nucleus keeps ids 0,1 (mass-before 0.0 and 0.4 < 0.5)
    kept, _ = _kept_ids(S(temperature=1.0, top_p=0.5), probs)
    assert kept == [0, 1]
    # min_p=0.6: threshold 0.4*0.6=0.24 -> 0.3 stays, 0.2 pruned
    kept, _ = _kept_ids(S(temperature=1.0, min_p=0.6), probs)
    assert kept == [0, 1]
    # llama.cpp order: top_k FIRST, top_p over the k renormalized survivors.
    # [0.36, 0.34, 0.30] @ top_k=2 renorms to [0.514, 0.486]; top_p=0.45 then
    # drops the runner-up (mass-before 0.514 > 0.45). Vocab-order top_p would
    # have kept it (0.36 < 0.45).
    kept, _ = _kept_ids(S(temperature=1.0, top_k=2, top_p=0.45),
                        [0.36, 0.34, 0.30])
    assert kept == [0]
    # the argmax can never be filtered away (_MIN_KEEP)
    kept, _ = _kept_ids(S(temperature=1.0, top_p=1e-9), probs)
    assert kept == [0]
    # temperature scales the surviving logits (applied last)
    import math
    _, masked = _kept_ids(S(temperature=0.5, top_k=2), probs)
    assert math.isclose(float(masked[0, 0].item()), math.log(0.4) / 0.5,
                        rel_tol=1e-5)


def test_fast_sampler_call_shapes_and_determinism():
    import mlx.core as mx
    s = sp_sampling._FastPositionedSampler(temperature=0.7, top_k=1)
    logits = mx.log(mx.array([[0.1, 0.2, 0.6, 0.1]]))
    # top_k=1 leaves a single candidate -> always the argmax id
    assert int(s(logits).item()) == 2
    # a drafter's [B, 1, V] block keeps its leading shape
    assert s(logits[:, None, :]).shape == (1, 1)


def test_fast_sampler_install_lands():
    """Identity check on the REAL upstream class: an mlx-vlm rename of
    ResponseGenerator._make_sampler must fail here, not silently no-op."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    original = cls._make_sampler
    try:
        sp.install_fast_sampler()
        patched = cls._make_sampler
        assert patched is not original                      # actually swapped
        assert getattr(patched, sp_sampling._FAST_SAMPLER_FLAG, False)
        sp.install_fast_sampler()                           # idempotent
        assert cls._make_sampler is patched
        me = types.SimpleNamespace()
        # greedy keeps the batch engine's argmax fast path
        assert patched(me, types.SimpleNamespace(temperature=0)) is None
        s = patched(me, types.SimpleNamespace(temperature=0.6, top_p=0.9,
                                              top_k=40, min_p=0.05, seed=3))
        assert isinstance(s, sp_sampling._FastPositionedSampler)
        assert (s.top_k, s.min_p, s.seed) == (40, 0.05, 3)
    finally:
        cls._make_sampler = original


# 7b. chat_template_kwargs passthrough
def _spec_ctkw(**ctkw):
    return ResolvedModel(id="m", path="/p", sampling={}, load={}, cache={},
                         system=None, speculative=False, mmproj=None,
                         draft_gguf=None, pin=False, ttl_s=None,
                         chat_template_kwargs=ctkw)


def test_merged_template_kwargs_request_wins_over_profile():
    spec = _spec_ctkw(preserve_thinking=True, foo="profile")
    request = types.SimpleNamespace(chat_template_kwargs={"foo": "request"})
    merged = sp_chat._merged_template_kwargs(request, spec)
    assert merged == {"preserve_thinking": True, "foo": "request"}


def test_merged_template_kwargs_each_side_alone_and_empty():
    # request only (single-model mode: no active spec)
    req = types.SimpleNamespace(chat_template_kwargs={"preserve_thinking": True})
    assert sp_chat._merged_template_kwargs(req, None) == {"preserve_thinking": True}
    # profile only (request carries nothing)
    spec = _spec_ctkw(preserve_thinking=False)
    assert sp_chat._merged_template_kwargs(types.SimpleNamespace(), spec) == {
        "preserve_thinking": False}
    # neither => {}
    assert sp_chat._merged_template_kwargs(types.SimpleNamespace(), None) == {}


def test_install_chat_template_kwargs_forwards_into_to_template_kwargs():
    """End-to-end seam: the gen-args wrapper stashes the merged dict and the
    patched to_template_kwargs folds it into what mlx-vlm hands the template."""
    gen = importlib.import_module("mlx_vlm.server.generation")

    def stub(request, processor=None, tenant_id=None):
        return gen.GenerationArguments()

    _APP._build_gen_args = stub
    sp.install_gen_args_profile_injection()
    sp.install_chat_template_kwargs()
    fn = _APP._build_gen_args
    assert getattr(fn, sp_chat._CTKW_FLAG, False)        # stash carried on the chain

    spec = _spec_ctkw(preserve_thinking=True)
    tok = serving.set_active_spec(spec)
    try:
        req = types.SimpleNamespace(model_fields_set=set(),
                                    chat_template_kwargs={"foo": "bar"})
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    kw = args.to_template_kwargs()
    assert kw["preserve_thinking"] is True          # from the profile
    assert kw["foo"] == "bar"                        # from the request
    # enable_thinking was not explicit (request/spec/env) -> dropped from the
    # template kwargs so the chat template's own default governs (b90aa60),
    # while the args flag stays True for the generation path.
    assert "enable_thinking" not in kw
    assert args.enable_thinking is True

    # explicitly set on the request -> preserved verbatim
    spec = _spec_ctkw()
    tok = serving.set_active_spec(spec)
    try:
        req = types.SimpleNamespace(model_fields_set={"enable_thinking"},
                                    chat_template_kwargs=None,
                                    enable_thinking=False)
        args = _APP._build_gen_args(req)
    finally:
        serving.reset_active_spec(tok)
    assert "enable_thinking" in args.to_template_kwargs()


def test_install_chat_template_kwargs_idempotent_and_noop_default():
    sp.install_chat_template_kwargs()
    gen = importlib.import_module("mlx_vlm.server.generation")
    first = gen.GenerationArguments.to_template_kwargs
    sp.install_chat_template_kwargs()
    assert gen.GenerationArguments.to_template_kwargs is first
    # a request/spec with no kwargs leaves to_template_kwargs untouched (stock keys)
    assert gen.GenerationArguments().to_template_kwargs() == {
        "enable_thinking": gen.GenerationArguments().enable_thinking}


# 8. OpenAI stop sequences
def test_request_stop_sequences_normalizes():
    assert sp_chat._request_stop_sequences(types.SimpleNamespace(stop="END")) == ["END"]
    assert sp_chat._request_stop_sequences(
        types.SimpleNamespace(stop=["a", "", "b"])) == ["a", "b"]
    assert sp_chat._request_stop_sequences(types.SimpleNamespace(model="m")) == []


def test_request_stop_sequences_rejects_non_string():
    from fastapi import HTTPException
    for bad in (5, True, {"a": 1}, ["ok", 3]):
        with pytest.raises(HTTPException) as ei:
            sp_chat._request_stop_sequences(types.SimpleNamespace(stop=bad))
        assert ei.value.status_code == 400


def test_request_stop_sequences_profile_fallback():
    """A request that omits `stop` under an active server config inherits the
    resolved spec's sampling stop strings; a request `stop` still wins."""
    _register({
        "profiles": {"strict": {"sampling": {"stop": ["END", "<|eot|>"]}}},
        "models": {"m": {"path": "/abs/m.gguf", "profile": "strict"}},
    })
    assert serving.server_config() is not None
    req = types.SimpleNamespace(model="m")               # no stop attr
    assert sp_chat._request_stop_sequences(req) == ["END", "<|eot|>"]
    req2 = types.SimpleNamespace(model="m", stop="X")    # request wins
    assert sp_chat._request_stop_sequences(req2) == ["X"]


def test_trim_chat_response_cuts_earliest_and_marks_stop():
    msg = types.SimpleNamespace(content="hello STOP world END")
    choice = types.SimpleNamespace(message=msg, finish_reason="length")
    resp = types.SimpleNamespace(choices=[choice])
    sp_chat._trim_chat_response(resp, ["END", "STOP"])
    assert msg.content == "hello "
    assert choice.finish_reason == "stop"


def test_trim_chat_response_passthrough_without_hit():
    msg = types.SimpleNamespace(content="hello world")
    choice = types.SimpleNamespace(message=msg, finish_reason="length")
    sp_chat._trim_chat_response(types.SimpleNamespace(choices=[choice]), ["STOP"])
    assert msg.content == "hello world"
    assert choice.finish_reason == "length"


def _sse(obj):
    import json
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(content=None, finish=None, **meta):
    delta = {} if content is None else {"content": content}
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **meta}


def _run_sse_filter(events, stops):
    """Drive _stop_filter_sse over a fake upstream; returns (out_events, closed)."""
    import asyncio

    state = {"closed": False}

    async def upstream():
        try:
            for e in events:
                yield e
        finally:
            state["closed"] = True

    async def collect():
        out = []
        async for e in sp_chat._stop_filter_sse(upstream(), stops):
            out.append(e)
        return out

    return asyncio.run(collect()), state["closed"]


def _stream_text(out_events):
    import json
    text = ""
    for e in out_events:
        if not e.startswith("data: ") or "[DONE]" in e:
            continue
        obj = json.loads(e[len("data: "):])
        for ch in obj.get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
    return text


def test_sse_filter_stop_split_across_chunks():
    events = [_sse(_chunk(content="Hello ")), _sse(_chunk(content="wor")),
              _sse(_chunk(content="ld ST")), _sse(_chunk(content="OP more")),
              _sse(_chunk(finish="length")), "data: [DONE]\n\n"]
    out, closed = _run_sse_filter(events, ["STOP"])
    assert _stream_text(out) == "Hello world "
    assert closed                                     # upstream cancelled
    assert out[-1] == "data: [DONE]\n\n"
    import json
    fin = json.loads(out[-2][len("data: "):])
    assert fin["choices"][0]["finish_reason"] == "stop"
    assert fin["model"] == "m"                        # meta carried over


def test_sse_filter_no_hit_releases_holdback_at_finish():
    events = [_sse(_chunk(content="alpha ")), _sse(_chunk(content="beta")),
              _sse(_chunk(finish="length")), "data: [DONE]\n\n"]
    out, _ = _run_sse_filter(events, ["LONG-STOP-SEQ"])
    assert _stream_text(out) == "alpha beta"          # nothing swallowed
    assert out[-1] == "data: [DONE]\n\n"


# _keepalive_sse: comments injected while the upstream is silent, chunks pass
# through untouched, upstream errors propagate, and closing the wrapper closes
# the upstream (the client-disconnect -> token_iter.close -> batch-cancel path).
def test_keepalive_sse_injects_comment_on_silence():
    import asyncio

    async def upstream():
        yield "data: a\n\n"
        await asyncio.sleep(0.2)
        yield "data: b\n\n"

    async def collect():
        out = []
        async for e in sp_flow._keepalive_sse(upstream(), 0.05):
            out.append(e)
        return out

    out = asyncio.run(collect())
    assert [e for e in out if e.startswith("data: ")] == \
        ["data: a\n\n", "data: b\n\n"]
    ia, ib = out.index("data: a\n\n"), out.index("data: b\n\n")
    assert ": keepalive\n\n" in out[ia + 1:ib]


def test_keepalive_sse_quiet_when_upstream_is_fast():
    import asyncio

    async def upstream():
        yield "data: a\n\n"
        yield "data: b\n\n"

    async def collect():
        return [e async for e in sp_flow._keepalive_sse(upstream(), 5.0)]

    assert asyncio.run(collect()) == ["data: a\n\n", "data: b\n\n"]


def test_keepalive_sse_propagates_upstream_error():
    import asyncio

    async def upstream():
        yield "data: a\n\n"
        raise ValueError("boom")

    async def collect():
        async for _ in sp_flow._keepalive_sse(upstream(), 0.05):
            pass

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(collect())


def test_keepalive_sse_close_closes_upstream():
    import asyncio

    state = {"closed": False}

    async def upstream():
        try:
            yield "data: a\n\n"
            await asyncio.sleep(60)
            yield "data: b\n\n"
        finally:
            state["closed"] = True

    async def run():
        gen = sp_flow._keepalive_sse(upstream(), 0.05)
        assert await gen.__anext__() == "data: a\n\n"
        await gen.aclose()

    asyncio.run(run())
    assert state["closed"]


def test_sse_filter_passes_role_usage_and_done():
    role = _sse({"id": "c1", "object": "chat.completion.chunk", "created": 1,
                 "model": "m",
                 "choices": [{"index": 0, "delta": {"role": "assistant"},
                              "finish_reason": None}]})
    usage = _sse({"id": "c1", "choices": [], "usage": {"total_tokens": 3}})
    events = [role, _sse(_chunk(content="hi")), _sse(_chunk(finish="stop")),
              usage, "data: [DONE]\n\n"]
    out, _ = _run_sse_filter(events, ["ZZZ"])
    assert out[0] == role
    assert any("total_tokens" in e for e in out)
    assert out[-1] == "data: [DONE]\n\n"


def test_install_openai_stop_wraps_routes_idempotent():
    sp.install_openai_stop_sequences()
    routes = {getattr(r, "path", None): r for r in _APP.app.router.routes}
    for path in sp_common._CHAT_PATHS:
        ep = routes[path].endpoint
        assert getattr(ep, sp_chat._STOP_FLAG, False)
        # FastAPI must still see the original ChatRequest body annotation
        import inspect
        params = list(inspect.signature(ep).parameters.values())
        assert params[0].annotation.__name__ == "ChatRequest"
    n = len(_APP.app.router.routes)
    sp.install_openai_stop_sequences()                # idempotent
    assert len(_APP.app.router.routes) == n


def test_openai_stop_endpoint_e2e_with_stub():
    """POST through FastAPI with a stub original handler: proves the signature
    propagation parses the body and the wrapper trims non-stream responses."""
    from fastapi.testclient import TestClient

    import inspect

    from fastapi import Request as _Request

    schemas = importlib.import_module("mlx_vlm.server.schemas")

    async def stub(request, http_request):
        return schemas.ChatResponse(choices=[schemas.ChatChoice(
            finish_reason="length",
            message=schemas.ChatMessage(role="assistant",
                                        content="hello STOP world"))])

    # Real-class annotations (the module's future-annotations would stringize
    # inline ones, and FastAPI couldn't resolve them against test locals).
    stub.__signature__ = inspect.Signature([
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=schemas.ChatRequest),
        inspect.Parameter("http_request",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_Request),
    ])

    sp_common._remove_routes(_APP.app, *sp_common._CHAT_PATHS)
    _APP.app.add_api_route("/v1/chat/completions", stub, methods=["POST"])
    sp.install_openai_stop_sequences()
    client = TestClient(_APP.app)
    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "stop": "STOP"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello "
    assert body["choices"][0]["finish_reason"] == "stop"
    # without stop: untouched
    r2 = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert r2.json()["choices"][0]["message"]["content"] == "hello STOP world"


# 9-10. API-key auth, host guard, CORS hardening, /health trim
def _auth_client():
    """Auth + a known authed route (/v1/reload) and the exempt /health."""
    from fastapi.testclient import TestClient

    sp.install_api_key_auth("sekrit")
    sp.install_reload_route(lambda: {"reloaded": 1})
    sp.install_health_liveness_override()
    return TestClient(_APP.app)


def test_auth_noop_without_key():
    before = len(_APP.app.user_middleware)
    sp.install_api_key_auth(None)
    sp.install_api_key_auth("")
    assert len(_APP.app.user_middleware) == before
    assert not getattr(_APP.app.state, sp_hardening._AUTH_FLAG, False)


def test_auth_401_without_or_wrong_key():
    client = _auth_client()
    r = client.post("/v1/reload")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    r2 = client.post("/v1/reload", headers={"Authorization": "Bearer wrong"})
    assert r2.status_code == 401
    r3 = client.post("/v1/reload", headers={"x-api-key": "wrong"})
    assert r3.status_code == 401


def test_auth_200_with_bearer_or_x_api_key():
    client = _auth_client()
    r = client.post("/v1/reload", headers={"Authorization": "Bearer sekrit"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "success"
    r2 = client.post("/v1/reload", headers={"x-api-key": "sekrit"})
    assert r2.status_code == 200, r2.text


def test_auth_health_exempt():
    client = _auth_client()
    r = client.get("/health")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "healthy"


def test_auth_options_preflight_exempt():
    """CORS preflight is credential-less by spec - the auth middleware must let
    OPTIONS through so CORSMiddleware can answer it for browser clients."""
    client = _auth_client()
    r = client.options("/v1/reload", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST"})
    assert r.status_code != 401, r.text


def test_auth_install_idempotent():
    before = len(_APP.app.user_middleware)
    sp.install_api_key_auth("sekrit")
    sp.install_api_key_auth("sekrit")
    assert len(_APP.app.user_middleware) == before + 1


def test_auth_non_ascii_key_no_500():
    """A non-ASCII configured key must not 500 (str compare_digest raises
    TypeError on it); the matching key arrives utf-8 on the wire."""
    from fastapi.testclient import TestClient

    sp.install_api_key_auth("klücz-ключ")
    sp.install_reload_route(lambda: {"reloaded": 1})
    client = TestClient(_APP.app)
    r = client.post("/v1/reload",
                    headers={b"x-api-key": "klücz-ключ".encode()})
    assert r.status_code == 200, r.text
    r2 = client.post("/v1/reload", headers={"x-api-key": "wrong"})
    assert r2.status_code == 401


def test_host_guard_rejects_foreign_host():
    from fastapi.testclient import TestClient

    sp.install_loopback_host_guard("127.0.0.1")
    sp.install_health_liveness_override()
    client = TestClient(_APP.app)
    r = client.get("/health", headers={"Host": "evil.example.com"})
    assert r.status_code == 403
    assert r.json()["error"]["type"] == "invalid_host"
    for ok in ("127.0.0.1:8080", "localhost:8080", "localhost", "[::1]:8080",
               "Localhost:8080", "LOCALHOST"):  # Host compares case-insensitively
        assert client.get("/health", headers={"Host": ok}).status_code == 200, ok


def test_host_guard_idempotent():
    before = len(_APP.app.user_middleware)
    sp.install_loopback_host_guard("127.0.0.1")
    sp.install_loopback_host_guard("127.0.0.1")
    assert len(_APP.app.user_middleware) == before + 1


def test_host_header_name_parsing():
    assert sp_hardening._host_header_name("127.0.0.1:8080") == "127.0.0.1"
    assert sp_hardening._host_header_name("localhost") == "localhost"
    assert sp_hardening._host_header_name("[::1]:8080") == "::1"
    assert sp_hardening._host_header_name("evil.com:80") == "evil.com"
    assert sp_hardening._host_header_name("LocalHost:8080") == "localhost"
    assert sp_hardening._host_header_name("::1") == "::1"  # bare IPv6, no port strip


def test_disable_credentialed_cors():
    from fastapi.middleware.cors import CORSMiddleware

    cors = [m for m in _APP.app.user_middleware
            if getattr(m, "cls", None) is CORSMiddleware]
    assert cors, "stock mlx-vlm app should carry CORSMiddleware"
    assert cors[0].kwargs.get("allow_credentials") is True   # stock hazard
    sp.disable_credentialed_cors()
    assert cors[0].kwargs.get("allow_credentials") is False


def test_health_liveness_no_paths():
    """/health (the unauthenticated route) must not leak model/adapter paths."""
    from fastapi.testclient import TestClient

    sp.install_health_liveness_override()
    client = TestClient(_APP.app)
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["pid"] == os.getpid()      # lets the CLI verify process identity
    assert not any(isinstance(v, str) and "/" in v for v in body.values())


# APC lone-request harvest (the single-stream SSD-store fix)
class _FakeKVCache:
    """A plain KVCache stand-in: exposes keys/values/offset, no _idx."""
    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class _FakeBatchKVCache:
    """A batched cache stand-in: carries _idx + left_padding like BatchKVCache."""
    def __init__(self, keys, values, idx, left_padding):
        self.keys = keys
        self.values = values
        self._idx = idx
        self.left_padding = left_padding


class _RecordingAPC:
    def __init__(self):
        self.calls = []

    def store_kv_blocks(self, token_ids, layer_keys, layer_values, *,
                        extra_hash=0, skip_first_n_tokens=0):
        self.calls.append({
            "n_tokens": len(token_ids),
            "layer_shapes": [k.shape for k in layer_keys],
            "skip": skip_first_n_tokens,
        })
        return ["blk"]


def test_lone_harvest_stores_plain_kvcache():
    """After install, a plain KVCache (offset, no _idx) harvests its
    [0, offset) row. Stock declined it through mlx-vlm 0.6.3 (the gap the
    patch fills); 0.6.4 absorbed the fallback but not the quantized guard
    (see test_lone_harvest_skips_unsupported_cache), so the replacement
    installs on every version and no stock baseline is asserted here."""
    import mlx.core as mx
    apc = importlib.import_module("mlx_vlm.apc")

    keys = mx.zeros((1, 2, 40, 8))
    values = mx.zeros((1, 2, 40, 8))
    cache = [_FakeKVCache(keys, values, 40)]

    sp.install_apc_lone_harvest()
    mgr2 = _RecordingAPC()
    out = apc.harvest_blocks_from_batch_cache(mgr2, cache, 0, list(range(40)))
    assert out == ["blk"]
    assert len(mgr2.calls) == 1
    assert mgr2.calls[0]["n_tokens"] == 40
    assert mgr2.calls[0]["layer_shapes"] == [(1, 2, 40, 8)]


def test_lone_harvest_preserves_batched_path():
    """The patch must not change behaviour for a real batched cache (_idx present):
    it still slices [left_padding, _idx) for the requested row."""
    import mlx.core as mx
    apc = importlib.import_module("mlx_vlm.apc")
    sp.install_apc_lone_harvest()

    keys = mx.zeros((2, 2, 48, 8))
    values = mx.zeros((2, 2, 48, 8))
    cache = [_FakeBatchKVCache(keys, values, idx=48, left_padding=mx.array([0, 16]))]
    mgr = _RecordingAPC()
    # row 1 has 16 left-pad -> harvested span is 48-16 = 32
    out = apc.harvest_blocks_from_batch_cache(mgr, cache, 1, list(range(48)))
    assert out == ["blk"]
    assert mgr.calls[0]["layer_shapes"] == [(1, 2, 32, 8)]


def test_lone_harvest_skips_unsupported_cache():
    """A cache with neither _idx nor offset (or quantized tuple keys) is declined,
    not crashed."""
    import mlx.core as mx
    apc = importlib.import_module("mlx_vlm.apc")
    sp.install_apc_lone_harvest()
    mgr = _RecordingAPC()

    # no _idx, no offset
    nocache = [_FakeKVCache(mx.zeros((1, 2, 8, 8)), mx.zeros((1, 2, 8, 8)), None)]
    assert apc.harvest_blocks_from_batch_cache(mgr, nocache, 0, list(range(8))) == []

    # quantized: keys is a tuple, not an mx.array
    quant = [_FakeKVCache((mx.zeros((1, 2, 8, 4)),), (mx.zeros((1, 2, 8, 4)),), 8)]
    assert apc.harvest_blocks_from_batch_cache(mgr, quant, 0, list(range(8))) == []
    assert mgr.calls == []


def test_lone_harvest_install_idempotent():
    apc = importlib.import_module("mlx_vlm.apc")
    sp.install_apc_lone_harvest()
    first = apc.harvest_blocks_from_batch_cache
    sp.install_apc_lone_harvest()
    assert apc.harvest_blocks_from_batch_cache is first
    assert apc._kq_lone_harvest is True


# 8b. off-loop model load (install_chat_load_offload) - keeps /health responsive
# during a cold load/swap. The route wrapper pre-warms the model on a worker
# thread, then the stock handler's get_cached_model is a cache hit.
import asyncio  # noqa: E402


def _register_fake_chat(handler):
    """Register a fake POST /v1/chat/completions whose endpoint is ``handler``
    (no annotations -> FastAPI treats the params as query params, fine for the
    direct-invocation tests below)."""
    app = _APP.app
    sp_common._remove_routes(app, "/v1/chat/completions")
    app.add_api_route("/v1/chat/completions", handler, methods=["POST"],
                      include_in_schema=False)


def _chat_endpoint():
    return next(r.endpoint for r in _APP.app.router.routes
                if getattr(r, "path", None) == "/v1/chat/completions"
                and "POST" in (getattr(r, "methods", None) or ()))


class _FakeReq:
    def __init__(self, model="m", adapter=None, has_adapter=False):
        self.model = model
        self.model_fields_set = {"adapter_path"} if has_adapter else set()
        if has_adapter:
            self.adapter_path = adapter


def test_chat_load_offload_warms_off_loop_before_handler(monkeypatch):
    order, seen = [], []

    async def fake_handler(request, http_request):
        order.append(("handler", getattr(request, "model", None)))
        return {"ok": True}

    def fake_warm(model_id, adapter):
        import threading
        seen.append((model_id, adapter, threading.current_thread().name))
        order.append(("warm", model_id))

    _register_fake_chat(fake_handler)
    monkeypatch.setattr(sp_routes, "_warm_and_release", fake_warm)
    sp.install_chat_load_offload()

    res = asyncio.run(_chat_endpoint()(_FakeReq(model="m"), object()))
    assert res == {"ok": True}
    assert order == [("warm", "m"), ("handler", "m")]      # warm strictly first
    assert seen[0][1] is _APP._INHERIT_ADAPTER             # inherit (no adapter set)
    assert seen[0][2] != "MainThread"                      # ran off the event loop


def test_chat_load_offload_passes_explicit_adapter(monkeypatch):
    seen = []

    async def fake_handler(request, http_request):
        return {"ok": True}

    monkeypatch.setattr(sp_routes, "_warm_and_release",
                        lambda m, a: seen.append((m, a)))
    _register_fake_chat(fake_handler)
    sp.install_chat_load_offload()

    asyncio.run(_chat_endpoint()(
        _FakeReq(model="m", adapter="/lora", has_adapter=True), object()))
    assert seen == [("m", "/lora")]


def test_chat_load_offload_swallows_warm_error(monkeypatch):
    ran = []

    async def fake_handler(request, http_request):
        ran.append(request.model)
        return {"ok": True}

    def boom(model_id, adapter):
        raise RuntimeError("load failed")

    _register_fake_chat(fake_handler)
    monkeypatch.setattr(sp_routes, "_warm_and_release", boom)
    sp.install_chat_load_offload()

    res = asyncio.run(_chat_endpoint()(_FakeReq(model="m"), object()))
    assert res == {"ok": True}      # handler still ran; warm error swallowed
    assert ran == ["m"]


def test_chat_load_offload_skips_when_no_model(monkeypatch):
    seen = []

    async def fake_handler(request, http_request):
        return {"ok": True}

    monkeypatch.setattr(sp_routes, "_warm_and_release",
                        lambda m, a: seen.append(m))
    _register_fake_chat(fake_handler)
    sp.install_chat_load_offload()

    asyncio.run(_chat_endpoint()(_FakeReq(model=None), object()))
    assert seen == []               # nothing to warm -> no off-loop call


def test_chat_load_offload_idempotent():
    async def fake_handler(request, http_request):
        return {"ok": True}

    _register_fake_chat(fake_handler)
    sp.install_chat_load_offload()
    first = _chat_endpoint()
    sp.install_chat_load_offload()
    assert _chat_endpoint() is first     # second install is a no-op


class _FakeRawReq:
    """Duck-typed starlette Request (the anthropic/responses routes take the
    raw request): ``.receive`` + async ``.json()``."""

    def __init__(self, body):
        self._body = body
        self.receive = object()

    async def json(self):
        return self._body


def test_load_offload_covers_raw_request_routes(monkeypatch):
    order, seen = [], []

    async def fake_handler(request):
        order.append("handler")
        return {"ok": True}

    app = _APP.app
    sp_common._remove_routes(app, "/v1/messages")
    app.add_api_route("/v1/messages", fake_handler, methods=["POST"],
                      include_in_schema=False)
    monkeypatch.setattr(sp_routes, "_warm_and_release",
                        lambda m, a: (order.append("warm"), seen.append((m, a))))
    sp.install_chat_load_offload()

    ep = next(r.endpoint for r in app.router.routes
              if getattr(r, "path", None) == "/v1/messages"
              and "POST" in (getattr(r, "methods", None) or ()))
    res = asyncio.run(ep(_FakeRawReq({"model": "raw-m", "max_tokens": 8})))
    assert res == {"ok": True}
    assert order == ["warm", "handler"]                 # warm strictly first
    assert seen == [("raw-m", _APP._INHERIT_ADAPTER)]

    res = asyncio.run(ep(_FakeRawReq({"max_tokens": 8})))
    assert res == {"ok": True}
    assert len(seen) == 1               # no model in the body -> no warm


def test_warm_and_release_drops_hold(monkeypatch):
    from gmlx import residency
    released = []

    class FakeHold:
        def release(self):
            released.append(True)

    def fake_gcm(model_id, *a, **k):
        residency._active_hold.set(FakeHold())
        return (object(), None, None)

    monkeypatch.setattr(_APP, "get_cached_model", fake_gcm)
    residency._active_hold.set(None)
    sp_routes._warm_and_release("m")
    assert released == [True]            # transient warm drops its busy ref


def test_spawn_preload_warm_retains_hold(monkeypatch):
    from gmlx import residency

    class FakeHold:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    def fake_gcm(model_id, *a, **k):
        residency._active_hold.set(FakeHold())
        return (object(), None, None)

    monkeypatch.setattr(_APP, "get_cached_model", fake_gcm)
    sp_routes._PRELOAD_HOLDS.clear()
    try:
        sp.spawn_preload_warm("m").join(timeout=5)
        assert len(sp_routes._PRELOAD_HOLDS) == 1
        assert sp_routes._PRELOAD_HOLDS[0].released is False   # retained -> eviction-proof
    finally:
        sp_routes._PRELOAD_HOLDS.clear()


def test_spawn_preload_warm_extras_after_primary(monkeypatch):
    order = []

    monkeypatch.setattr(sp_routes, "_load_resident",
                        lambda m, *a, **k: order.append(("primary", m)))
    monkeypatch.setattr(sp_routes, "_warm_and_release",
                        lambda m, *a, **k: order.append(("extra", m)))
    monkeypatch.setattr(sp_routes, "_preload_extra_over_budget",
                        lambda m: m == "huge")
    sp.spawn_preload_warm("m", ["e1", "huge", "e2"]).join(timeout=5)
    # primary first, extras in order, the over-budget streaming one skipped
    assert order == [("primary", "m"), ("extra", "e1"), ("extra", "e2")]


def test_spawn_preload_warm_extras_without_primary(monkeypatch):
    order = []

    monkeypatch.setattr(sp_routes, "_load_resident",
                        lambda m, *a, **k: order.append(("primary", m)))
    monkeypatch.setattr(sp_routes, "_warm_and_release",
                        lambda m, *a, **k: order.append(("extra", m)))
    monkeypatch.setattr(sp_routes, "_preload_extra_over_budget", lambda m: False)
    sp.spawn_preload_warm(None, ["e1"]).join(timeout=5)
    assert order == [("extra", "e1")]   # no pin/default primary: extras only


def test_optional_request_model_defaults_blank():
    # Three surfaces promise "fallback model when a request omits `model`";
    # the schemas must therefore accept the omission (the resolver already
    # treats "" as the default).
    schemas = importlib.import_module("mlx_vlm.server.schemas")
    sp.install_optional_request_model()
    req = schemas.ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.model == ""
    ant = schemas.AnthropicRequest(messages=[{"role": "user", "content": "hi"}],
                                   max_tokens=5)
    assert ant.model == ""


def test_profile_capture_rewrites_blank_model_to_default(monkeypatch):
    # An omitted/empty model is rewritten to the resolved default id at the
    # route seam so the response echoes the model that actually served it.
    seen = {}

    async def fake_handler(request, http_request):
        seen["model"] = request.model
        return {"ok": True}

    _register_fake_chat(fake_handler)
    sp.install_request_profile_capture()
    monkeypatch.setattr(serving, "_default_model_id", lambda: "the-default")

    class _Req:
        model_fields_set = set()
        model = ""

    asyncio.run(_chat_endpoint()(_Req(), object()))
    assert seen["model"] == "the-default"

    class _Req2:
        model_fields_set = set()
        model = "explicit"

    asyncio.run(_chat_endpoint()(_Req2(), object()))
    assert seen["model"] == "explicit"                # explicit ids untouched


# 8c. request-body `profile` capture
def test_profile_capture_binds_contextvar_around_handler():
    seen = {}

    async def fake_handler(request, http_request):
        seen["profile"] = serving.get_request_profile()
        return {"ok": True}

    _register_fake_chat(fake_handler)
    sp.install_request_profile_capture()

    class _Req:
        model_fields_set = set()
        profile = "coding"

    res = asyncio.run(_chat_endpoint()(_Req(), object()))
    assert res == {"ok": True}
    assert seen["profile"] == "coding"
    assert serving.get_request_profile() is None      # reset after the request


def test_profile_capture_none_when_absent():
    seen = {}

    async def fake_handler(request, http_request):
        seen["profile"] = serving.get_request_profile()
        return {"ok": True}

    _register_fake_chat(fake_handler)
    sp.install_request_profile_capture()

    class _Req:
        model_fields_set = set()

    asyncio.run(_chat_endpoint()(_Req(), object()))
    assert seen["profile"] is None


def test_profile_capture_reads_raw_request_json():
    """The anthropic / responses routes receive a raw starlette Request - the
    capture falls back to the (cached) JSON body."""

    class _RawReq:
        def receive(self):                       # marks it request-like
            pass

        async def json(self):
            return {"model": "m", "profile": "creative"}

    async def probe():
        return await sp_flow._extract_request_profile([_RawReq()])

    assert asyncio.run(probe()) == "creative"

    class _BadReq(_RawReq):
        async def json(self):
            raise ValueError("no body")

    async def probe_bad():
        return await sp_flow._extract_request_profile([_BadReq()])

    assert asyncio.run(probe_bad()) is None


def test_profile_capture_parsed_body_skips_raw_read():
    """When a parsed request object is present (chat), the raw Request must NOT
    be consulted - no needless body reads on every chat request."""

    class _Parsed:
        model_fields_set = set()

    class _Trap:
        def receive(self):
            pass

        async def json(self):
            raise AssertionError("raw body read despite parsed request")

    async def probe():
        return await sp_flow._extract_request_profile([_Parsed(), _Trap()])

    assert asyncio.run(probe()) is None


def test_profile_capture_reset_on_handler_error():
    async def boom(request, http_request):
        raise RuntimeError("handler failed")

    _register_fake_chat(boom)
    sp.install_request_profile_capture()

    class _Req:
        model_fields_set = set()
        profile = "coding"

    with pytest.raises(RuntimeError):
        asyncio.run(_chat_endpoint()(_Req(), object()))
    assert serving.get_request_profile() is None


def test_profile_capture_idempotent():
    async def fake_handler(request, http_request):
        return {"ok": True}

    _register_fake_chat(fake_handler)
    sp.install_request_profile_capture()
    n = len(_APP.app.router.routes)
    sp.install_request_profile_capture()
    assert len(_APP.app.router.routes) == n


def test_body_profile_shapes_resolution_like_residency():
    """Mirror the residency seam's exact call: with the ContextVar bound, the
    resolved spec carries the body profile's sampling (the original bug: the
    body field reached only the stop resolver)."""
    _register({
        "server": {"model_dirs": ["/models"]},
        "profiles": {"slow": {"sampling": {"temperature": 0.01}}},
        "models": {"m": {"path": "/abs/m.gguf"}},
    })
    tok = serving.set_request_profile("slow")
    try:
        _path, spec = serving.resolve_request_model(
            "m", profile_field=serving.get_request_profile())
    finally:
        serving.reset_request_profile(tok)
    assert spec.sampling["temperature"] == 0.01
    assert spec.profile_name == "slow"


def test_voices_route_lists_and_404s_when_unconfigured(monkeypatch):
    from fastapi.testclient import TestClient

    from gmlx import tts
    monkeypatch.setattr(tts, "available_voices",
                        lambda m: ["af_heart", "am_adam"])
    sp.install_audio_voices_route("mlx-community/Kokoro-82M-bf16")
    client = TestClient(_APP.app)
    r = client.get("/v1/audio/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["voices"] == ["af_heart", "am_adam"]
    assert body["default"] == tts.DEFAULT_VOICE
    # unconfigured server: no route is added at all
    sp_common._remove_routes(_APP.app, "/v1/audio/voices")
    sp.install_audio_voices_route(None)
    assert TestClient(_APP.app).get("/v1/audio/voices").status_code == 404


def test_request_stop_sequences_caps_request_count():
    from fastapi import HTTPException
    many = [f"s{i}" for i in range(sp_chat._MAX_REQUEST_STOPS + 1)]
    with pytest.raises(HTTPException) as ei:
        sp_chat._request_stop_sequences(types.SimpleNamespace(stop=many))
    assert ei.value.status_code == 400
    ok = sp_chat._request_stop_sequences(
        types.SimpleNamespace(stop=[f"s{i}" for i in range(sp_chat._MAX_REQUEST_STOPS)]))
    assert len(ok) == sp_chat._MAX_REQUEST_STOPS


def test_sampling_float_rejects_non_numeric():
    from fastapi import HTTPException
    assert sp_sampling._sampling_float("xtc_probability", "0.5") == 0.5
    for bad in ("abc", [0.5], {}):
        with pytest.raises(HTTPException) as ei:
            sp_sampling._sampling_float("xtc_probability", bad)
        assert ei.value.status_code == 400
