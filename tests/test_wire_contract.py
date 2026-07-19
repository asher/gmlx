#!/usr/bin/env python3
"""Wire contract over the REAL mlx-vlm routes with generation stubbed.

Both wire bugs that shipped (timings:null breaking Open WebUI; top_k/min_p
silently unhonored) lived in the seam between gmlx's patches and the stock
mlx-vlm protocol handlers - a seam no other test crosses (they call patched
functions directly or stub the whole handler). This module mounts the GENUINE
chat/completions/messages/responses handlers, applies install_server_patches
exactly as `gmlx serve` does, and stubs only the generation seam the routes
already abstract over: ``runtime.response_generator`` (the continuous-batching
engine every handler drives) plus the model loader / chat-template / structured
-processor bindings the handlers resolve late through the ``mlx_vlm.server``
package (``_server_package_attr``). Requests flow through FastAPI TestClient,
so real pydantic parsing, serialization, and SSE framing are exercised.
CPU-only; no model load."""
from __future__ import annotations

import importlib
import json
import time

import pytest

pytest.importorskip("mlx_vlm")

from fastapi.testclient import TestClient  # noqa: E402

from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx import server_patches as sp  # noqa: E402
from gmlx.server_patches import hardening as sp_hardening  # noqa: E402
from gmlx.config import build_config  # noqa: E402
from gmlx.residency import _http_from_resolver_error  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_PKG = importlib.import_module("mlx_vlm.server")
_GEN = importlib.import_module("mlx_vlm.server.generation")
_SCHEMAS = importlib.import_module("mlx_vlm.server.schemas")
_UTILS = importlib.import_module("mlx_vlm.utils")
_RUNTIME = importlib.import_module("mlx_vlm.server.runtime").runtime

MODEL_ID = "stub"
MODEL_PATH = "/abs/wire-stub.gguf"  # absolute path: resolver skips the stat
DEFAULT_TEXT = "Hello world."


# -- generation-seam fakes -----------------------------------------------------

class _WireTokenizer:
    # both gemma4 markers so the REAL parser inference picks a tool module
    chat_template = "wire template <|tool_call> ... <tool_call|>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return [1]

    def decode(self, ids):
        return "x"


class _WireProcessor:
    tokenizer = _WireTokenizer()


class _WireConfig:
    model_type = "stub"

    class text_config:
        max_position_embeddings = 8192


class _FakeResponseGenerator:
    """Duck-typed stand-in for mlx-vlm's ResponseGenerator: the one seam every
    protocol handler drives. Yields real StreamingToken objects."""

    def __init__(self):
        self.tokenizer = _WireTokenizer()
        self.reset()

    def reset(self):
        self.calls = []
        self.script = None          # list[(text, finish_reason)] override
        self.prompt_tokens = 7
        self.prefill_delay_s = 0.0  # models prefill: first next() blocks
        self.iter_closed = False

    def _tokens(self):
        return self.script or [("Hello", None), (" world", None), (".", "stop")]

    def validate_context_budget(self, prompt, images, audio, args):
        return None

    def generate(self, prompt=None, images=None, audio=None, args=None):
        self.calls.append({"prompt": prompt, "args": args})
        ctx = _GEN.GenerationContext(uid=1, prompt_tokens=self.prompt_tokens)

        def _iter():
            try:
                if self.prefill_delay_s:
                    # runs under asyncio.to_thread, so the event loop stays
                    # free to emit keepalives while this blocks
                    time.sleep(self.prefill_delay_s)
                for i, (text, fin) in enumerate(self._tokens()):
                    yield _GEN.StreamingToken(
                        text=text, token=100 + i, logprobs=-0.1,
                        finish_reason=fin, prompt_tps=250.0)
            finally:
                self.iter_closed = True

        return ctx, _iter()


def _fake_get_cached_model(model_path, adapter_path=None, *, model_kind="auto"):
    # Same resolution step as residency's pooled_get_cached_model: friendly id
    # (maybe id@profile) -> spec, bound for the gen-args seam. Load stubbed.
    load_path = model_path
    if serving.server_config() is not None:
        try:
            load_path, spec = serving.resolve_request_model(
                str(model_path), profile_field=serving.get_request_profile())
        except (serving.ModelNotFound, serving.ModelFileMissing,
                serving.UnknownProfile, serving.NoModelSpecified) as e:
            raise _http_from_resolver_error(e) from e
        serving.set_active_spec(spec)
    entry = {"model_path": load_path, "processor": _PROCESSOR,
             "config": _CONFIG}
    mc = _RUNTIME.model_cache
    if hasattr(mc, "ensure_kind"):  # ModelCacheRegistry (mlx-vlm >= 0.6.4)
        mc.ensure_kind().update(entry)
    else:  # plain dict (<= 0.6.3)
        mc.update(entry)
    return object(), _PROCESSOR, _CONFIG


def _fake_apply_chat_template(processor, config, messages, **kwargs):
    return "user: " + json.dumps(messages, default=str)


def _fake_json_schema_processor(tokenizer, schema):
    def _noop(tokens, logits):
        return logits
    return _noop


_PROCESSOR = _WireProcessor()
_CONFIG = _WireConfig()


# -- fixture: real routes + full patch set over the fakes ----------------------

@pytest.fixture(scope="module")
def wire_app():
    fastapi_app = _APP.app
    apc = importlib.import_module("mlx_vlm.apc")
    openai_mod = importlib.import_module("mlx_vlm.server.openai")
    anthropic_mod = importlib.import_module("mlx_vlm.server.anthropic")
    deps = getattr(_APP, "_protocol_deps", None)
    saved = {
        "build_gen_args": _APP._build_gen_args,
        "snapshot": _APP._server_runtime_snapshot,
        "get_model_path": _UTILS.get_model_path,
        "routes": list(fastapi_app.router.routes),
        "handlers": dict(fastapi_app.exception_handlers),
        "middleware": list(fastapi_app.user_middleware),
        "mw_kwargs": [(m, dict(getattr(m, "kwargs", {}) or {}))
                      for m in fastapi_app.user_middleware],
        "deps_bga": getattr(deps, "build_gen_args", None),
        "deps_act": getattr(deps, "apply_chat_template", None),
        "openai_bga": getattr(openai_mod, "_build_gen_args", None),
        "anthropic_bga": getattr(anthropic_mod, "_build_gen_args", None),
        "openai_act": getattr(openai_mod, "apply_chat_template", None),
        "anthropic_act": getattr(anthropic_mod, "apply_chat_template", None),
        "to_template_kwargs": _GEN.GenerationArguments.to_template_kwargs,
        "make_sampler": _GEN.ResponseGenerator._make_sampler,
        "make_tb_criteria": _GEN.ResponseGenerator._make_thinking_budget_criteria,
        "metrics_success": _GEN.ServerMetricsStore.record_success,
        "metrics_failure": _GEN.ServerMetricsStore.record_failure,
        "tss_init": importlib.import_module(
            "mlx_vlm.server.responses_state").ThinkingStreamState.__init__,
        "stream_chunk_dump": _SCHEMAS.ChatStreamChunk.model_dump_json,
        "apc_harvest": apc.harvest_blocks_from_batch_cache,
        "apc_lone_flag": getattr(apc, "_kq_lone_harvest", False),
        "apc_store": apc.APCManager.store_kv_blocks,
        "apc_store_flag": getattr(apc, "_kq_batched_store_eval", False),
        "pkg_gcm": _PKG.get_cached_model,
        "app_gcm": _APP.get_cached_model,
        "pkg_act": _PKG.apply_chat_template,
        "pkg_jsp": _PKG.build_json_schema_logits_processor,
        "rt_rg": _RUNTIME.response_generator,
        "rt_mc": _RUNTIME.model_cache,
        "rt_metrics": _RUNTIME.metrics,
        "pool": getattr(_PKG, "_kq_residency_pool", None),
        "model_defaults": {
            name: getattr(_SCHEMAS, name).model_fields["model"].default
            for name in ("VLMRequest", "GenerationRequest", "ChatRequest",
                         "AnthropicRequest", "OpenAIRequest")},
    }

    serving.clear_resolved_models()
    cfg = build_config({
        "server": {"host": "127.0.0.1", "port": 8080,
                   "stt": "mlx-community/whisper-large-v3-turbo"},
        "profiles": {"fast": {"sampling": {"temperature": 0.15}}},
        "models": {MODEL_ID: {"path": MODEL_PATH}},
    })
    serving.register_resolved_models(cfg)
    sp.install_server_patches(cfg, reload_fn=None)   # exactly the serve path

    gen = _FakeResponseGenerator()
    _RUNTIME.response_generator = gen
    _RUNTIME.model_cache = {"model_path": MODEL_PATH, "processor": _PROCESSOR,
                            "config": _CONFIG}
    _RUNTIME.metrics = _GEN.ServerMetricsStore()
    for target in (_PKG, _APP):
        target.get_cached_model = _fake_get_cached_model
    _PKG.apply_chat_template = _fake_apply_chat_template
    _PKG.build_json_schema_logits_processor = _fake_json_schema_processor

    # base_url must be loopback: the installed host guard 403s "testserver"
    client = TestClient(fastapi_app, base_url="http://127.0.0.1")

    class _Wire:
        pass

    w = _Wire()
    w.client = client
    w.gen = gen
    yield w

    for target in (_PKG, _APP):
        target.get_cached_model = saved["pkg_gcm"] if target is _PKG \
            else saved["app_gcm"]
    _PKG.apply_chat_template = saved["pkg_act"]
    _PKG.build_json_schema_logits_processor = saved["pkg_jsp"]
    _RUNTIME.response_generator = saved["rt_rg"]
    _RUNTIME.model_cache = saved["rt_mc"]
    _RUNTIME.metrics = saved["rt_metrics"]
    _APP._build_gen_args = saved["build_gen_args"]
    _APP._server_runtime_snapshot = saved["snapshot"]
    _UTILS.get_model_path = saved["get_model_path"]
    _GEN.GenerationArguments.to_template_kwargs = saved["to_template_kwargs"]
    _GEN.ResponseGenerator._make_sampler = saved["make_sampler"]
    _GEN.ResponseGenerator._make_thinking_budget_criteria = \
        saved["make_tb_criteria"]
    _GEN.ServerMetricsStore.record_success = saved["metrics_success"]
    _GEN.ServerMetricsStore.record_failure = saved["metrics_failure"]
    importlib.import_module(
        "mlx_vlm.server.responses_state").ThinkingStreamState.__init__ = \
        saved["tss_init"]
    _SCHEMAS.ChatStreamChunk.model_dump_json = saved["stream_chunk_dump"]
    apc.harvest_blocks_from_batch_cache = saved["apc_harvest"]
    apc._kq_lone_harvest = saved["apc_lone_flag"]
    apc.APCManager.store_kv_blocks = saved["apc_store"]
    apc._kq_batched_store_eval = saved["apc_store_flag"]
    if deps is not None:
        if saved["deps_bga"] is not None:
            deps.build_gen_args = saved["deps_bga"]
        if saved["deps_act"] is not None:
            deps.apply_chat_template = saved["deps_act"]
    openai_mod._build_gen_args = saved["openai_bga"]
    anthropic_mod._build_gen_args = saved["anthropic_bga"]
    openai_mod.apply_chat_template = saved["openai_act"]
    anthropic_mod.apply_chat_template = saved["anthropic_act"]
    for name, default in saved["model_defaults"].items():
        cls = getattr(_SCHEMAS, name)
        cls.model_fields["model"].default = default
        cls.model_rebuild(force=True)
    fastapi_app.router.routes[:] = saved["routes"]
    fastapi_app.exception_handlers.clear()
    fastapi_app.exception_handlers.update(saved["handlers"])
    fastapi_app.user_middleware[:] = saved["middleware"]
    for m, kw in saved["mw_kwargs"]:
        if getattr(m, "kwargs", None) is not None:
            m.kwargs.clear()
            m.kwargs.update(kw)
    fastapi_app.middleware_stack = None
    for flag in (sp_hardening._AUTH_FLAG, sp_hardening._HOST_GUARD_FLAG, sp_hardening._JSON_CT_FLAG):
        if hasattr(fastapi_app.state, flag):
            delattr(fastapi_app.state, flag)
    if saved["pool"] is None:
        if hasattr(_PKG, "_kq_residency_pool"):
            delattr(_PKG, "_kq_residency_pool")
    else:
        _PKG._kq_residency_pool = saved["pool"]
    serving.clear_resolved_models()


@pytest.fixture()
def wire(wire_app):
    wire_app.gen.reset()
    _RUNTIME.metrics.reset()
    yield wire_app
    if hasattr(_PKG, "_kq_residency_pool"):
        delattr(_PKG, "_kq_residency_pool")


# -- helpers -------------------------------------------------------------------

def _chat_body(**kw):
    body = {"model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


def _sse_data_lines(text):
    return [ln[len("data: "):] for ln in text.splitlines()
            if ln.startswith("data: ")]


def _sse_event_names(text):
    return [ln[len("event: "):] for ln in text.splitlines()
            if ln.startswith("event: ")]


def _assert_no_nulls(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert v is not None, f"null value at {path}.{k}"
            _assert_no_nulls(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nulls(v, f"{path}[{i}]")


# 1. streaming chunks are byte-vanilla (the Open WebUI timings:null regression;
#    contract: server_patches.install_vanilla_stream_chunks, held through the
#    REAL streaming route, not just the schema method)
def test_stream_chunks_have_no_null_fields(wire):
    r = wire.client.post("/v1/chat/completions",
                         json=_chat_body(stream=True))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = _sse_data_lines(r.text)
    assert lines, "no SSE data lines"
    assert lines[-1] == "[DONE]"                    # terminal marker
    for payload in lines[:-1]:
        _assert_no_nulls(json.loads(payload))       # no timings:null etc.
    said = ""
    for payload in lines[:-1]:
        for ch in json.loads(payload).get("choices", []):
            said += ch.get("delta", {}).get("content", "") or ""
    assert said == DEFAULT_TEXT
    fin = [json.loads(p) for p in lines[:-1]
           if any(c.get("finish_reason") for c in json.loads(p)["choices"])]
    assert fin and fin[-1]["choices"][0]["finish_reason"] == "stop"


# 2. non-stream chat wire shape (stock ChatResponse through FastAPI encoding)
def test_chat_nonstream_wire_shape(wire):
    r = wire.client.post("/v1/chat/completions", json=_chat_body())
    assert r.status_code == 200, r.text
    body = r.json()
    # pin the stock top-level key set (timings is mlx-vlm's llama.cpp-style extra)
    assert set(body) == {"id", "object", "created", "model", "choices",
                         "usage", "timings"}
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["model"] == MODEL_ID
    choice = body["choices"][0]
    assert choice["message"]["content"] == DEFAULT_TEXT
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 10
    assert all(isinstance(usage[k], int) for k in
               ("prompt_tokens", "completion_tokens", "total_tokens"))
    # the exclude_none patch is stream-only by design (server_patches/chat_behavior.py,
    # install_vanilla_stream_chunks docstring: "The non-streaming ChatResponse
    # is a different class and is untouched"), so the non-stream body still
    # carries stock null-valued optionals - pin that, and require the value
    # subtrees clients consume to be null-free.
    assert choice["logprobs"] is None
    assert choice["message"]["reasoning"] is None
    _assert_no_nulls(usage, "$.usage")
    _assert_no_nulls(body["timings"], "$.timings")


# 3. Anthropic surface (docs/serving-architecture.md:138 - ANTHROPIC_BASE_URL
#    / Claude Code target /v1/messages)
def test_messages_wire_shape(wire):
    r = wire.client.post("/v1/messages", json={
        "model": MODEL_ID, "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == DEFAULT_TEXT
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["output_tokens"] == 3


def test_messages_stop_sequences_honored(wire):
    wire.gen.script = [("alpha ", None), ("STOP", None), (" beta", "stop")]
    r = wire.client.post("/v1/messages", json={
        "model": MODEL_ID, "max_tokens": 32, "stop_sequences": ["STOP"],
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"][0]["text"] == "alpha"
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "STOP"


def test_messages_streaming_event_types(wire):
    r = wire.client.post("/v1/messages", json={
        "model": MODEL_ID, "max_tokens": 32, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    names = _sse_event_names(r.text)
    assert "message_start" in names
    assert "content_block_delta" in names
    assert names[-1] == "message_stop"
    deltas = [json.loads(p) for p in _sse_data_lines(r.text)
              if json.loads(p).get("type") == "content_block_delta"]
    said = "".join(d["delta"].get("text", "") for d in deltas
                   if d["delta"].get("type") == "text_delta")
    assert said == DEFAULT_TEXT


# 4. /v1/completions - the gmlx-installed minimal text route (single string
#    prompt, n=1), driving the same engine seam without a chat template.
def test_completions_route_shape(wire):
    r = wire.client.post("/v1/completions", json={
        "model": MODEL_ID, "prompt": "hi", "max_tokens": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert body["choices"][0]["text"] == DEFAULT_TEXT
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 7
    assert body["usage"]["completion_tokens"] == 3
    # the raw prompt reached the engine untouched - no chat template applied
    assert wire.gen.calls[-1]["prompt"] == "hi"


def test_completions_stream_shape(wire):
    r = wire.client.post("/v1/completions", json={
        "model": MODEL_ID, "prompt": "hi", "stream": True})
    assert r.status_code == 200, r.text
    lines = _sse_data_lines(r.text)
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(p) for p in lines[:-1]]
    assert all(c["object"] == "text_completion" for c in chunks)
    text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
    assert text == DEFAULT_TEXT
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_completions_rejects_out_of_scope(wire):
    r = wire.client.post("/v1/completions", json={
        "model": MODEL_ID, "prompt": ["a", "b"]})
    assert r.status_code == 400
    r = wire.client.post("/v1/completions", json={
        "model": MODEL_ID, "prompt": "hi", "n": 2})
    assert r.status_code == 400


# 4b. tool_choice "none" enforcement (api_contract): tools are dropped before
#     the handler, so the tools-present skip_special_tokens flip never fires.
def test_chat_tool_choice_none_drops_tools_end_to_end(wire):
    body = _chat_body(tools=[{"type": "function", "function": {
        "name": "t", "parameters": {"type": "object"}}}])
    r = wire.client.post("/v1/chat/completions",
                         json={**body, "tool_choice": "none"})
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].skip_special_tokens is True
    # without tool_choice none the tools reach the handler and flip it
    r = wire.client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].skip_special_tokens is False


def test_anthropic_tool_choice_none_drops_tools_end_to_end(wire):
    body = {"model": MODEL_ID, "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}]}
    r = wire.client.post("/v1/messages",
                         json={**body, "tool_choice": {"type": "none"}})
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].skip_special_tokens is True
    r = wire.client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].skip_special_tokens is False


# 5. /v1/responses (README / getting-started promise the Responses API)
def test_responses_route_shape(wire):
    r = wire.client.post("/v1/responses",
                         json={"model": MODEL_ID, "input": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == DEFAULT_TEXT
    msg = next(o for o in body["output"] if o.get("type") == "message")
    assert msg["content"][0]["text"] == DEFAULT_TEXT
    assert body["usage"]["input_tokens"] == 7
    assert body["usage"]["output_tokens"] == 3


# 6. stream_options.include_usage on the stock streaming path
def test_stream_options_include_usage_on_stock_path(wire):
    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        stream=True, stream_options={"include_usage": True}))
    assert r.status_code == 200, r.text
    lines = _sse_data_lines(r.text)
    assert lines[-1] == "[DONE]"
    usage_chunks = [json.loads(p) for p in lines[:-1]
                    if "usage" in json.loads(p)]
    assert len(usage_chunks) == 1
    chunk = usage_chunks[0]
    assert chunk["choices"] == []
    assert chunk["usage"]["prompt_tokens"] == 7
    assert chunk["usage"]["completion_tokens"] == 3
    _assert_no_nulls(chunk)


# 6b. SSE keepalive (server_patches.install_sse_keepalive): comments flow while
#     the engine is silent (deep-prefill model: first next() blocks), the data
#     payloads are untouched, and 0 disables. Interval is read per request from
#     GMLX_SSE_KEEPALIVE_S, so monkeypatch.setenv works under the
#     module-scoped fixture.
def _keepalive_offsets(text):
    ka = text.find(": keepalive")
    data = text.find("data: ")
    return ka, data


def test_stream_keepalive_before_first_token(wire, monkeypatch):
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "0.05")
    wire.gen.prefill_delay_s = 0.3
    r = wire.client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert r.status_code == 200, r.text
    ka, data = _keepalive_offsets(r.text)
    assert ka != -1, "no keepalive comment emitted during slow prefill"
    assert ka < data, "keepalive did not precede the first data chunk"
    lines = _sse_data_lines(r.text)
    assert lines[-1] == "[DONE]"
    said = "".join(
        json.loads(p)["choices"][0]["delta"].get("content") or ""
        for p in lines[:-1] if json.loads(p).get("choices"))
    assert said == DEFAULT_TEXT


def test_stream_no_keepalive_when_fast(wire, monkeypatch):
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "5")
    r = wire.client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert r.status_code == 200, r.text
    assert ": keepalive" not in r.text


def test_keepalive_zero_disables(wire, monkeypatch):
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "0")
    wire.gen.prefill_delay_s = 0.3
    r = wire.client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert r.status_code == 200, r.text
    assert ": keepalive" not in r.text
    assert _sse_data_lines(r.text)[-1] == "[DONE]"


def test_keepalive_on_responses_route(wire, monkeypatch):
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "0.05")
    wire.gen.prefill_delay_s = 0.3
    r = wire.client.post("/v1/responses", json={
        "model": MODEL_ID, "input": "hi", "stream": True})
    assert r.status_code == 200, r.text
    # the route yields response.created immediately; the silent window is
    # between it and the first output delta
    ka = r.text.find(": keepalive")
    delta = r.text.find("response.output_text.delta")
    assert ka != -1, "no keepalive comment emitted during slow prefill"
    assert delta != -1 and ka < delta


def test_keepalive_on_messages_route(wire, monkeypatch):
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "0.05")
    wire.gen.prefill_delay_s = 0.3
    r = wire.client.post("/v1/messages", json={
        "model": MODEL_ID, "max_tokens": 32, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    ka, data = _keepalive_offsets(r.text)
    assert ka != -1 and ka < data
    names = _sse_event_names(r.text)
    assert "message_start" in names and names[-1] == "message_stop"


def test_stream_content_identical_with_keepalive(wire, monkeypatch):
    r_plain = wire.client.post("/v1/chat/completions",
                               json=_chat_body(stream=True))
    monkeypatch.setenv("GMLX_SSE_KEEPALIVE_S", "0.05")
    wire.gen.prefill_delay_s = 0.3
    r_ka = wire.client.post("/v1/chat/completions",
                            json=_chat_body(stream=True))
    assert r_plain.status_code == r_ka.status_code == 200

    def _deltas(text):
        return [json.loads(p)["choices"][0]["delta"]
                for p in _sse_data_lines(text)[:-1]
                if json.loads(p).get("choices")]

    assert _deltas(r_plain.text) == _deltas(r_ka.text)


# 7. tool-call extraction (docs/serving-architecture.md:121 "Tool calls are
#    extracted"; docs/server-config.md:912-918). The parser is inferred from
#    the processor chat template (gemma4 markers here), and the REAL
#    process_tool_calls parses the generated markup.
def test_chat_tool_calls_shape(wire):
    wire.gen.script = [
        ('<|tool_call>call:get_time{tz:<|"|>UTC<|"|>}<tool_call|>', "stop")]
    tools = [{"type": "function", "function": {
        "name": "get_time",
        "parameters": {"type": "object",
                       "properties": {"tz": {"type": "string"}}}}}]
    r = wire.client.post("/v1/chat/completions",
                         json=_chat_body(tools=tools))
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["id"]
    assert call["function"]["name"] == "get_time"
    assert json.loads(call["function"]["arguments"]) == {"tz": "UTC"}


# 8. response_format (docs/server-config.md:939-948). json_schema -> 200.
#    DOC MISMATCH: docs/server-config.md:946 says `"json_object"` "is rejected
#    with `Unsupported response_format type`", but mlx-vlm now maps it to a
#    permissive object schema (mlx_vlm/server/app.py:250-251,
#    _extract_response_format_schema: `("json_object", "object") ->
#    {"type": "object"}`) - it is ACCEPTED and grammar-constrained. Only an
#    unknown type gets the documented 400.
def test_response_format_wire(wire):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        response_format={"type": "json_schema",
                         "json_schema": {"schema": schema}}))
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].logits_processors  # constraint attached

    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        response_format={"type": "json_object"}))
    assert r.status_code == 200, r.text                  # accepted (see above)
    assert wire.gen.calls[-1]["args"].logits_processors

    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        response_format={"type": "grammar"}))
    assert r.status_code == 400, r.text                  # not 5xx, not ignored
    assert "Unsupported response_format type" in r.text

    # json_schema without a schema field: clean 400, never a 500
    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        response_format={"type": "json_schema", "json_schema": {}}))
    assert r.status_code == 400, r.text


# 9. unknown OpenAI SDK params ride the extra="allow" schemas - never a 422
def test_unknown_openai_params_tolerated(wire):
    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        n=1, logprobs=False, top_logprobs=0, user="u1",
        parallel_tool_calls=True, service_tier="auto"))
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == DEFAULT_TEXT


# 10. /v1/metrics: resident_models[] enrichment (docs/server-config.md:870) +
#     per-request timing fields in the envelope after a completed request
def test_metrics_resident_models_and_timings(wire):
    class _FakePool:
        def stats(self):
            return {"resident": [{"model_path": MODEL_PATH, "pinned": True,
                                  "busy": 0, "footprint_bytes": 123,
                                  "idle_s": 4.56, "ttl_s": 900}]}

    _PKG._kq_residency_pool = _FakePool()
    r = wire.client.post("/v1/chat/completions", json=_chat_body())
    assert r.status_code == 200, r.text

    m = wire.client.get("/v1/metrics")
    assert m.status_code == 200, m.text
    payload = m.json()
    entry = payload["server"]["resident_models"][0]
    assert entry["ids"] == [MODEL_ID]
    assert entry["pinned"] is True
    assert entry["idle_s"] == 4.6                    # rounded to 0.1
    assert entry["ttl_s"] == 900
    assert entry["footprint_bytes"] == 123
    latest = payload["latest"]
    assert latest["endpoint"] == "/chat/completions"
    assert latest["ttft_s"] is not None and latest["ttft_s"] >= 0.0
    assert latest["prefill_tok_s"] == 250.0
    assert latest["decode_tok_s"] is not None and latest["decode_tok_s"] > 0
    assert latest["prompt_tokens"] == 7
    assert latest["generated_tokens"] == 3
    assert payload["summary"]["requests_completed"] >= 1


# 11. /v1/models over HTTP (the installed override, not _models_payload direct)
def test_models_endpoint_over_http(wire):
    r = wire.client.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list"
    by_id = {m["id"]: m for m in body["data"]}
    stub = by_id[MODEL_ID]
    for key in ("resident", "pinned", "speculative", "vlm", "profile",
                "default", "owned_by"):
        assert key in stub
    whisper = by_id["whisper-1"]                     # configured stt service
    assert whisper["stt"] is True


# 12. @profile sampling reaches generation over the wire (server-config.md
#     `id@profile` addressing + install_gen_args_profile_injection)
def test_profile_sampling_reaches_generation_over_wire(wire):
    r = wire.client.post("/v1/chat/completions",
                         json=_chat_body(model=f"{MODEL_ID}@fast"))
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].temperature == 0.15
    # explicit client value beats the profile
    r = wire.client.post("/v1/chat/completions", json=_chat_body(
        model=f"{MODEL_ID}@fast", temperature=0.9))
    assert r.status_code == 200, r.text
    assert wire.gen.calls[-1]["args"].temperature == 0.9
    # unknown profile is a clean 400, matching the production seam
    r = wire.client.post("/v1/chat/completions",
                         json=_chat_body(model=f"{MODEL_ID}@nope"))
    assert r.status_code == 400, r.text


# 13. runtime.model_cache shape contract. mlx-vlm >= 0.6.4 replaces the
#     runtime's model_cache dict with a ModelCacheRegistry; residency reads
#     pooled entries duck-typed (entry.model_cache["model"], .get(...)), and
#     upstream's _model_cache_registry() converts a plain dict in place. Pin
#     the two operations residency performs against the registry so an
#     upstream reshape of the registry names itself here rather than as a
#     500 deep inside a pooled request (found live: this file's fake loader
#     called dict-.update() on a registry).
def test_model_cache_registry_supports_residency_reads():
    runtime_mod = importlib.import_module("mlx_vlm.server.runtime")
    registry_cls = getattr(runtime_mod, "ModelCacheRegistry", None)
    if registry_cls is None:
        pytest.skip("mlx-vlm <= 0.6.3: runtime.model_cache is a plain dict")
    model, proc, cfg = object(), object(), object()
    reg = registry_cls()
    # empty registry reads like an empty cache (the scratch-load contract)
    assert not reg.for_kind("text_generation")
    assert reg.get("model") is None
    reg.ensure_kind().update({"model": model, "processor": proc,
                              "config": cfg})
    # residency.py's entry reads: mc["model"], mc.get("processor"/"config")
    assert reg["model"] is model
    assert reg.get("processor") is proc
    assert reg.get("config") is cfg
