#!/usr/bin/env python3
"""API-contract route wrapper: the per-request unread-parameter warning,
tool_choice "none" enforcement, the unfulfilled-tool_choice warning, and the
drift guards that keep the consumed-field allowlists in sync with what the
upstream handlers and gmlx patches actually read. CPU-only, no model load."""
from __future__ import annotations

import importlib
import inspect
import logging
import re
from pathlib import Path

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from fastapi import Request  # noqa: E402

import gmlx.serve.patches as sp  # noqa: E402
from gmlx.serve.patches import _common as sp_common  # noqa: E402
from gmlx.serve.patches import api_contract as sp_api  # noqa: E402

_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_SCHEMAS = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "api.md"


@pytest.fixture(autouse=True)
def _restore_routes():
    saved = sp_common._snapshot_routes(_APP.app)
    yield
    sp_common._restore_routes(_APP.app, saved)


# -- allowlist drift guards ------------------------------------------------
# These fail loudly when a consumed field appears that the allowlists don't
# know about (which would make the warning fire on a field that actually
# works, or stay silent on one that doesn't). To update: re-read what the
# handler / _build_gen_args / gmlx patches read off the request, adjust the
# sets in gmlx/serve/patches/api_contract.py, the docs/api.md
# "Parameter support" table, and the hand lists below - together.

def _scrape_request_reads(*funcs) -> set:
    names = set()
    for fn in funcs:
        src = inspect.getsource(fn)
        names |= set(re.findall(r'getattr\(request,\s*"(\w+)"', src))
        names |= set(re.findall(
            r'_request_field_or_default\(\s*request,\s*"(\w+)"', src))
    return names


def test_gen_args_allowlist_matches_upstream_source():
    """Every field _build_gen_args / the structured-output extractor reads is
    in _GEN_ARGS_CONSUMED - and nothing more. An upstream mlx-vlm bump that
    adds or drops a gen-arg field must fail here."""
    # 0.6.15 moved the real body to request_normalization; app._build_gen_args
    # is a delegator, so scrape both.
    import gmlx._vendor.mlx_vlm.server.request_normalization as _rn
    scraped = _scrape_request_reads(
        _APP._build_gen_args, _rn._build_gen_args,
        _rn._standard_reasoning_control,
        _rn._extract_response_format_schema)
    # _request_field_is_set reads without getattr; assert the two standard
    # reasoning fields it consumes explicitly.
    scraped |= {"reasoning", "reasoning_effort"}
    # enable_thinking rides through _request_field_or_default with the name
    # built at the call site; assert it separately if the regex misses it.
    scraped.add("enable_thinking")
    assert scraped == set(sp_api._GEN_ARGS_CONSUMED), (
        "gen-args consumed-field drift: "
        f"missing={scraped - set(sp_api._GEN_ARGS_CONSUMED)} "
        f"stale={set(sp_api._GEN_ARGS_CONSUMED) - scraped}")


# Declared schema fields the handlers deliberately never read. Everything
# else declared on the schema must be in the dialect's consumed set.
_KNOWN_UNREAD = {
    "ChatRequest": set(),
    "OpenAIRequest": set(),
    "AnthropicRequest": {"metadata"},   # declared, never read by the handler
}


@pytest.mark.parametrize("schema_name,consumed", [
    ("ChatRequest", sp_api.CHAT_CONSUMED),
    ("OpenAIRequest", sp_api.RESPONSES_CONSUMED),
    ("AnthropicRequest", sp_api.ANTHROPIC_CONSUMED),
])
def test_declared_schema_fields_all_classified(schema_name, consumed):
    """Every declared upstream request field is either consumed (allowlisted,
    never warned about) or known-unread (warned about). A new upstream field
    lands here unclassified and fails loudly."""
    cls = getattr(_SCHEMAS, schema_name)
    declared = set(cls.model_fields)
    unclassified = declared - consumed - _KNOWN_UNREAD[schema_name]
    assert not unclassified, (
        f"{schema_name} fields not classified consumed/unread: "
        f"{sorted(unclassified)} - upstream added a field? update "
        "api_contract.py, the docs table, and _KNOWN_UNREAD")
    # a known-unread field must not also be allowlisted
    assert not (_KNOWN_UNREAD[schema_name] & consumed)


def test_gmlx_extras_allowlist_matches_patch_reads():
    """The gmlx-patch extras every dialect consumes. Hand-maintained: update
    when a server patch starts reading a new request field."""
    assert set(sp_api._GMLX_CONSUMED) == {
        "profile",              # request_flow profile capture + stop resolver
        "chat_template_kwargs",  # chat_behavior template-kwargs passthrough
        "xtc_probability",      # sampling XTC
        "xtc_threshold",
        "thinking",             # chat_behavior: z.ai {"type": en/disabled} spelling
        "adapters",             # lora_request: per-request adapter strengths
        "lora",                 # lora_request: llama-server spelling, by index
        "thinking_budget_message",   # sampling: the thinking-budget fields
        "reasoning_budget_tokens",
        "reasoning_budget_message",
    }


def test_known_ignored_params_do_warn():
    """The headline ignored params draw the warning on their dialect."""
    for name in ("n", "user", "parallel_tool_calls"):
        assert name not in sp_api.CHAT_CONSUMED
    assert "metadata" not in sp_api.ANTHROPIC_CONSUMED
    # stop is chat-only: the stop filter wraps only the chat routes
    assert "stop" in sp_api.CHAT_CONSUMED
    assert "stop" not in sp_api.RESPONSES_CONSUMED
    # output_config IS read (json_schema -> response_format): never warn
    assert "output_config" in sp_api.ANTHROPIC_CONSUMED


def test_prefill_fields_are_chat_only():
    """The chat route reads the prefill fields; Responses and Messages warn
    on them."""
    for name in ("continue_final_message", "add_generation_prompt"):
        assert name in sp_api.CHAT_CONSUMED
        assert name not in sp_api.RESPONSES_CONSUMED
        assert name not in sp_api.ANTHROPIC_CONSUMED
        assert f"| `{name}` | Honored | Ignored | Ignored |" in _DOCS.read_text()


def test_doc_table_rows_match_allowlists():
    """Cross-check the docs/api.md "Parameter support" table
    against the consumed sets: an `ignored` cell must be off the allowlist,
    a `honored` cell on it. Cells reading template/rejected/n-a are prose."""
    text = _DOCS.read_text()
    m = re.search(r"### Parameter support\n(.*?)\n\n(?:###|##) ", text,
                  re.DOTALL)
    assert m, "api.md lost its '### Parameter support' table"
    consumed_by_col = [sp_api.CHAT_CONSUMED, sp_api.RESPONSES_CONSUMED,
                       sp_api.ANTHROPIC_CONSUMED]
    rows = 0
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].startswith("`"):
            continue
        rows += 1
        param = cells[0].strip("`")
        for col, consumed in enumerate(consumed_by_col, start=1):
            cell = cells[col].lower()
            if cell == "honored":
                assert param in consumed, \
                    f"doc says honored, allowlist disagrees: {param} col={col}"
            elif cell == "ignored":
                assert param not in consumed, \
                    f"doc says ignored, allowlist disagrees: {param} col={col}"
    assert rows >= 10, "parameter table went missing or lost its rows"


def test_systemone_allowlist_covers_extensions_and_docs():
    """SYSTEMONE_CONSUMED holds every schema extension, and the decisions
    guide or its reference names each consumed field in backticks."""
    from gmlx.systemone.schema import JEV_EXTENSIONS

    assert set(JEV_EXTENSIONS) <= sp_api.SYSTEMONE_CONSUMED
    text = "\n".join((_DOCS.parent / name).read_text()
                     for name in ("decisions.md", "decisions-reference.md"))
    missing = sorted(f for f in sp_api.SYSTEMONE_CONSUMED if f"`{f}`" not in text)
    assert not missing, f"decisions.md and decisions-reference.md do not name {missing}"


# -- wrapper behavior --------------------------------------------------------

def _capture_warnings():
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logger = logging.getLogger("gmlx.serve.patches.api_contract")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    return records, handler, logger


def _chat_stub(seen, tool_calls=None, finish="stop"):
    async def stub(request, http_request):
        seen["tools"] = getattr(request, "tools", None)
        return _SCHEMAS.ChatResponse(choices=[_SCHEMAS.ChatChoice(
            finish_reason=finish,
            message=_SCHEMAS.ChatMessage(role="assistant", content="hi",
                                         tool_calls=tool_calls))])
    stub.__signature__ = inspect.Signature([
        inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=_SCHEMAS.ChatRequest),
        inspect.Parameter("http_request",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=Request),
    ])
    return stub


def _raw_stub(seen):
    async def stub(http_request):
        seen["body"] = await http_request.json()
        return {"ok": True}
    stub.__signature__ = inspect.Signature([
        inspect.Parameter("http_request",
                          inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          annotation=Request),
    ])
    return stub


def _client_with(path, stub):
    from fastapi.testclient import TestClient
    sp_common._remove_routes(_APP.app, path)
    _APP.app.add_api_route(path, stub, methods=["POST"])
    sp.install_api_contract()
    return TestClient(_APP.app)


def test_chat_warns_once_listing_ignored_fields():
    records, handler, logger = _capture_warnings()
    try:
        seen = {}
        client = _client_with("/v1/chat/completions", _chat_stub(seen))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "n": 3, "user": "u", "parallel_tool_calls": False})
        assert r.status_code == 200
        warns = [m for m in records if "ignoring unsupported" in m]
        assert len(warns) == 1
        assert "n, parallel_tool_calls, user" in warns[0]
        assert "/v1/chat/completions" in warns[0]
    finally:
        logger.removeHandler(handler)


def test_chat_consumed_fields_never_warn():
    records, handler, logger = _capture_warnings()
    try:
        seen = {}
        client = _client_with("/v1/chat/completions", _chat_stub(seen))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.5, "stop": "END", "profile": "fast",
            "xtc_probability": 0, "chat_template_kwargs": {},
            "tool_choice": "auto", "seed": 3})
        assert r.status_code == 200
        assert not [m for m in records if "ignoring unsupported" in m]
    finally:
        logger.removeHandler(handler)


def test_return_token_ids_is_read_on_chat_only():
    assert "return_token_ids" in sp_api.CHAT_CONSUMED
    for consumed in (sp_api.COMPLETIONS_CONSUMED, sp_api.RESPONSES_CONSUMED,
                     sp_api.ANTHROPIC_CONSUMED, sp_api.SYSTEMONE_CONSUMED):
        assert "return_token_ids" not in consumed


def test_return_tokens_as_token_ids_is_read_on_chat_and_completions():
    assert "return_tokens_as_token_ids" in sp_api.CHAT_CONSUMED
    assert "return_tokens_as_token_ids" in sp_api.COMPLETIONS_CONSUMED
    for consumed in (sp_api.RESPONSES_CONSUMED, sp_api.ANTHROPIC_CONSUMED,
                     sp_api.SYSTEMONE_CONSUMED):
        assert "return_tokens_as_token_ids" not in consumed
    records, handler, logger = _capture_warnings()
    try:
        client = _client_with("/v1/chat/completions", _chat_stub({}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "logprobs": True, "return_tokens_as_token_ids": True})
        assert r.status_code == 200
        assert not [m for m in records if "ignoring unsupported" in m]
    finally:
        logger.removeHandler(handler)


def test_top_logprobs_above_cap_warns(monkeypatch):
    monkeypatch.delenv("TOP_LOGPROBS_K", raising=False)
    records, handler, logger = _capture_warnings()
    try:
        client = _client_with("/v1/chat/completions", _chat_stub({}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "logprobs": True, "top_logprobs": 25})
        assert r.status_code == 200
        warns = [m for m in records if "TOP_LOGPROBS_K" in m]
        assert len(warns) == 1
        assert "top_logprobs=25" in warns[0]
    finally:
        logger.removeHandler(handler)


def test_the_default_cap_is_20(monkeypatch):
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    monkeypatch.delenv("TOP_LOGPROBS_K", raising=False)
    assert gen.get_top_logprobs_k() == 20
    records, handler, logger = _capture_warnings()
    try:
        client = _client_with("/v1/chat/completions", _chat_stub({}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "logprobs": True, "top_logprobs": 5})
        assert r.status_code == 200
        assert not [m for m in records if "TOP_LOGPROBS_K" in m]
    finally:
        logger.removeHandler(handler)


def test_top_logprobs_within_cap_does_not_warn(monkeypatch):
    monkeypatch.setenv("TOP_LOGPROBS_K", "10")
    records, handler, logger = _capture_warnings()
    try:
        client = _client_with("/v1/chat/completions", _chat_stub({}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "logprobs": True, "top_logprobs": 5})
        assert r.status_code == 200
        assert not [m for m in records if "TOP_LOGPROBS_K" in m]
    finally:
        logger.removeHandler(handler)


def test_tool_choice_none_drops_tools_on_chat():
    seen = {}
    client = _client_with("/v1/chat/completions", _chat_stub(seen))
    tools = [{"type": "function", "function": {"name": "f"}}]
    client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "tools": tools, "tool_choice": "none"})
    assert seen["tools"] is None
    # without tool_choice none, tools pass through untouched
    client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "tools": tools})
    assert seen["tools"] == tools


def test_tool_choice_none_scrubs_raw_body_on_messages():
    seen = {}
    client = _client_with("/v1/messages", _raw_stub(seen))
    client.post("/v1/messages", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "t", "input_schema": {}}],
        "tool_choice": {"type": "none"}})
    assert "tools" not in seen["body"]
    assert "tool_choice" not in seen["body"]
    # any other tool_choice leaves the body alone
    client.post("/v1/messages", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "t", "input_schema": {}}],
        "tool_choice": {"type": "auto"}})
    assert "tools" in seen["body"]


def test_tool_choice_none_scrubs_raw_body_on_responses():
    seen = {}
    client = _client_with("/v1/responses", _raw_stub(seen))
    client.post("/v1/responses", json={
        "model": "m", "input": "x",
        "tools": [{"type": "function", "name": "f"}],
        "tool_choice": "none"})
    assert "tools" not in seen["body"]


def test_unfulfilled_required_tool_choice_warns():
    records, handler, logger = _capture_warnings()
    try:
        seen = {}
        client = _client_with("/v1/chat/completions", _chat_stub(seen))
        client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required"})
        assert [m for m in records if "forced tool call" in m]
    finally:
        logger.removeHandler(handler)


def test_fulfilled_tool_choice_does_not_warn():
    records, handler, logger = _capture_warnings()
    try:
        seen = {}
        stub = _chat_stub(seen, tool_calls=[{"id": "1", "type": "function"}],
                          finish="tool_calls")
        client = _client_with("/v1/chat/completions", stub)
        client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required"})
        assert not [m for m in records if "forced tool call" in m]
    finally:
        logger.removeHandler(handler)


def test_result_tool_calls_shapes():
    import types

    # anthropic non-stream: pydantic-shaped with stop_reason + content blocks
    msg = types.SimpleNamespace(stop_reason="tool_use", content=[])
    assert sp_api._result_tool_calls(msg) is True
    msg = types.SimpleNamespace(
        stop_reason="end_turn", content=[{"type": "text", "text": "x"}])
    assert sp_api._result_tool_calls(msg) is False
    msg = types.SimpleNamespace(
        stop_reason="end_turn", content=[{"type": "tool_use", "name": "t"}])
    assert sp_api._result_tool_calls(msg) is True
    # responses API: output items
    resp = types.SimpleNamespace(
        output=[{"type": "message"}, {"type": "function_call"}])
    assert sp_api._result_tool_calls(resp) is True
    resp = types.SimpleNamespace(output=[{"type": "message"}])
    assert sp_api._result_tool_calls(resp) is False
    # unknown shapes (streams, error bodies) stay undeterminable
    assert sp_api._result_tool_calls(object()) is None
    assert sp_api._result_tool_calls({"detail": "err"}) is None


def test_tool_choice_helpers():
    assert sp_api._is_tool_choice_none("none")
    assert sp_api._is_tool_choice_none({"type": "none"})
    assert not sp_api._is_tool_choice_none("auto")
    assert not sp_api._is_tool_choice_none(None)
    assert sp_api._tool_choice_wants_calls("required")
    assert sp_api._tool_choice_wants_calls({"type": "any"})
    assert sp_api._tool_choice_wants_calls({"type": "tool", "name": "t"})
    assert sp_api._tool_choice_wants_calls(
        {"type": "function", "function": {"name": "f"}})
    assert not sp_api._tool_choice_wants_calls("auto")
    assert not sp_api._tool_choice_wants_calls(None)


def test_install_api_contract_idempotent():
    sp.install_api_contract()
    n = len(_APP.app.router.routes)
    sp.install_api_contract()
    assert len(_APP.app.router.routes) == n
    routes = {getattr(r, "path", None): r for r in _APP.app.router.routes}
    for path in sp_common._CHAT_PATHS + sp_api._RESPONSES_PATHS \
            + sp_api._MESSAGES_PATHS:
        if path in routes:
            assert getattr(routes[path].endpoint,
                           sp_api._API_CONTRACT_FLAG, False)


def test_context_overflow_check_keeps_the_rule(monkeypatch):
    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    for mod in (gen, pkg):
        for name in ("_check_configured_context_budget",
                     "get_configured_context_limit"):
            monkeypatch.setattr(mod, name, getattr(mod, name))
    anthropic = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
    monkeypatch.setattr(anthropic, "_preflight_stream_context_budget",
                        anthropic._preflight_stream_context_budget)
    sp.install_context_overflow_wording()
    sp.install_context_overflow_wording()
    check = gen._check_configured_context_budget
    assert check is sp_api._check_context_budget
    assert pkg._check_configured_context_budget is check

    monkeypatch.setattr(gen, "get_configured_context_limit", lambda: 100)
    check(80, 20)
    check(100, None)
    with pytest.raises(gen.PromptTooLongError,
                       match="101 tokens > 100 maximum"):
        check(81, 20)
    monkeypatch.setattr(gen, "get_configured_context_limit", lambda: None)
    check(10**6, 10**6)


def test_context_limit_follows_the_request_model(monkeypatch):
    # The load keys reach the environment only while a model loads, so the
    # request check reads a per-model max_kv_size off the active spec.
    import types

    import gmlx.serve.bridge_vlm as serving

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    anthropic = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
    for mod in (gen, pkg):
        for name in ("_check_configured_context_budget",
                     "get_configured_context_limit"):
            monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(anthropic, "_preflight_stream_context_budget",
                        anthropic._preflight_stream_context_budget)
    monkeypatch.setattr(gen.runtime.config, "max_kv_size", None)
    monkeypatch.delenv("MAX_KV_SIZE", raising=False)
    sp.install_context_overflow_wording()
    sp.install_context_overflow_wording()
    limit = gen.get_configured_context_limit
    flag = sp_api._CONTEXT_LIMIT_FLAG
    assert getattr(limit, flag)
    assert not getattr(limit.__wrapped__, flag, False)  # wrapped once
    assert pkg.get_configured_context_limit is limit

    assert limit() is None
    monkeypatch.setenv("MAX_KV_SIZE", "8192")
    assert limit() == 8192
    cases = ((types.SimpleNamespace(load={"max_kv_size": 4096}), 4096),
             (types.SimpleNamespace(load={}), 8192),
             (types.SimpleNamespace(load={"max_kv_size": 0}), None))
    for spec, want in cases:
        assert sp_api.model_context_limit(spec) == want
        token = serving.set_active_spec(spec)
        try:
            assert limit() == want
        finally:
            serving.reset_active_spec(token)

    token = serving.set_active_spec(cases[0][0])
    try:
        with pytest.raises(gen.PromptTooLongError,
                           match="4097 tokens > 4096 maximum"):
            gen._check_configured_context_budget(4000, 97)
    finally:
        serving.reset_active_spec(token)


def _stream_response(chunks):
    """A minimal StreamingResponse stand-in: just a ``body_iterator``."""
    import types

    async def gen():
        for c in chunks:
            yield c

    return types.SimpleNamespace(body_iterator=gen())


async def _drain(resp):
    out = []
    async for chunk in resp.body_iterator:
        out.append(chunk)
    return out


def test_stream_forced_tool_choice_warns_at_clean_end(caplog):
    import asyncio

    resp = _stream_response([
        'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
        "data: [DONE]\n\n",
    ])
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_unfulfilled_tool_choice(
            "/v1/chat/completions", "required", resp)
        chunks = asyncio.run(_drain(resp))
    assert len(chunks) == 2                       # chunks pass through intact
    assert any("forced tool call" in r.message for r in caplog.records)


def test_stream_forced_tool_choice_silent_when_calls_seen(caplog):
    import asyncio

    resp = _stream_response([
        'data: {"choices": [{"delta": {"tool_calls": [{"id": "1"}]}}]}\n\n',
        "data: [DONE]\n\n",
    ])
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_unfulfilled_tool_choice(
            "/v1/chat/completions", "required", resp)
        asyncio.run(_drain(resp))
    assert not any("forced tool call" in r.message for r in caplog.records)


def test_stream_forced_tool_choice_silent_on_early_close(caplog):
    import asyncio

    resp = _stream_response(['data: {"choices": []}\n\n'] * 3)
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_unfulfilled_tool_choice(
            "/v1/chat/completions", "required", resp)

        async def read_one_and_close():
            it = resp.body_iterator
            await it.__anext__()
            await it.aclose()          # client disconnect: absence proves
        asyncio.run(read_one_and_close())          # nothing, so no warning
    assert not any("forced tool call" in r.message for r in caplog.records)


def test_stream_watch_skips_non_forced_and_non_streams():
    # auto/None tool_choice leaves the response untouched
    resp = _stream_response(["data: x\n\n"])
    it = resp.body_iterator
    sp_api._maybe_warn_unfulfilled_tool_choice(
        "/v1/chat/completions", "auto", resp)
    assert resp.body_iterator is it


def test_top_logprobs_cap_clamped_to_engine_max(caplog, monkeypatch):
    import types

    monkeypatch.setenv("TOP_LOGPROBS_K", "500")
    req = types.SimpleNamespace(top_logprobs=150)
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_top_logprobs("/v1/chat/completions", req)
    # the engine clamps the effective cap to 100, so 150 must warn even
    # though the raw env value says 500
    assert any("top_logprobs=150 exceeds the server cap 100" in r.message
               for r in caplog.records)
    caplog.clear()
    req = types.SimpleNamespace(top_logprobs=50)
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_top_logprobs("/v1/chat/completions", req)
    assert not caplog.records                     # within the effective cap


def test_context_limit_reads_under_a_later_wrap(monkeypatch):
    """A functools.wraps wrap on top of the per-model limit copies its
    flag. The process limit then unwrapped one level, to the per-model
    limit itself, and read the model of the request in flight."""
    import functools
    import types

    import gmlx.serve.bridge_vlm as serving

    gen = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    anthropic = importlib.import_module("gmlx._vendor.mlx_vlm.server.anthropic")
    for mod in (gen, pkg):
        for name in ("_check_configured_context_budget",
                     "get_configured_context_limit"):
            monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(anthropic, "_preflight_stream_context_budget",
                        anthropic._preflight_stream_context_budget)
    monkeypatch.setattr(gen.runtime.config, "max_kv_size", None)
    monkeypatch.setenv("MAX_KV_SIZE", "8192")
    sp.install_context_overflow_wording()
    limit = gen.get_configured_context_limit

    @functools.wraps(limit)
    def later():
        return limit()

    gen.get_configured_context_limit = later
    token = serving.set_active_spec(
        types.SimpleNamespace(load={"max_kv_size": 4096}))
    try:
        assert limit() == 4096 and later() == 4096
        assert sp_api.model_context_limit(
            types.SimpleNamespace(load={})) == 8192
        sp.install_context_overflow_wording()
        assert gen.get_configured_context_limit is later
    finally:
        serving.reset_active_spec(token)


# -- a lone surrogate escape -------------------------------------------------

class _QwenProcessor:
    """Renders the Qwen3.5 fixture template as its tokenizer does."""

    chat_template = (Path(__file__).resolve().parents[1] / "fixtures"
                     / "qwen35_template.jinja").read_text()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        from transformers.utils.chat_template_utils import (
            render_jinja_template)

        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt, **kw)
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class _OpensThinking(_QwenProcessor):
    """The fixture template, which opens its thinking block with the start
    token the request names, as the route hands it to the template."""

    chat_template = _QwenProcessor.chat_template + (
        "{%- if thinking_start_token is defined %}{{ thinking_start_token }}"
        "{%- endif %}")


class _Tokenizing:
    """The engine as far as its tokenizer, which encodes the prompt."""

    def generate(self, prompt, images=None, audio=None, args=None,
                 videos=None):
        import types

        prompt.encode("utf-8")

        class Tok:
            text, finish_reason, token_count = "ok", "stop", 1
        return types.SimpleNamespace(prompt_tokens=1, uid=1), iter([Tok()])


@pytest.mark.parametrize("body, field", [
    ({"messages": [{"role": "user", "content": "a \ud800 cat"}]},
     "messages[0].content"),
    ({"messages": [{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": [
                       {"type": "text", "text": "a \udc00 cat"}]}]},
     "messages[1].content[0].text"),
    ({"model": "m\ud800", "messages": [{"role": "user", "content": "a"}]},
     "model"),
    ({"messages": [{"role": "user", "content": "a"}],
      "tool_choice": {"type": "function", "function": {"name": "f\udc80"}},
      "tools": [{"type": "function", "function": {"name": "f",
                                                  "parameters": {}}}]},
     "tool_choice.function.name"),
    ({"messages": [{"role": "user", "content": "a"}],
      "response_format": {"type": "json_schema", "json_schema": {
          "name": "s", "schema": {"type": "object",
                                  "description": "a \udc80"}}}},
     "response_format.json_schema.schema.description"),
    ({"messages": [{"role": "user", "content": "a"}],
      "thinking_start_token": "<think\udc80>"},
     "thinking_start_token"),
    ({"messages": [{"role": "user", "content": "a"}],
      "instructions": "Be \udc80 brief."},
     "instructions")])
def test_a_chat_string_with_a_lone_surrogate_is_a_400(monkeypatch, body,
                                                      field):
    """JSON parses a lone surrogate escape to a str that encodes to no
    UTF-8, which the tokenizer then fails on: the chat route refuses it
    with the 400 the video route gives, naming the field, in any field of
    the body."""
    import json

    from fastapi.testclient import TestClient

    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    runtime = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.runtime").runtime
    monkeypatch.setattr(openai, "get_cached_model", lambda *a, **k: (
        None, _OpensThinking(), {"model_type": "qwen3_5"}))
    monkeypatch.setattr(runtime, "response_generator", _Tokenizing(),
                        raising=False)
    # The server's error handlers, on this test's copy of the app's, take
    # effect in the next middleware stack the app builds.
    monkeypatch.setattr(_APP.app, "exception_handlers",
                        dict(_APP.app.exception_handlers))
    monkeypatch.setattr(_APP.app, "middleware_stack", None)
    sp.install_resolver_error_handlers()
    sp.install_api_contract()
    client = TestClient(_APP.app, raise_server_exceptions=False)
    r = client.post("/v1/chat/completions",
                    content=json.dumps({"model": "m", "max_tokens": 1,
                                        **body}).encode(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert (err["message"], err["param"]) == (
        f"{field} holds a lone surrogate escape", field)


@pytest.mark.parametrize("path, body, field", [
    ("/v1/responses", {"input": "a \ud800 cat"}, "input"),
    ("/v1/responses", {"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "a \udc00 cat"}]}]},
     "input[0].content[0].text"),
    ("/v1/responses", {"input": "a", "instructions": "Be \udc80 brief."},
     "instructions"),
    ("/v1/messages", {"messages": [{"role": "user",
                                    "content": "a \ud800 cat"}]},
     "messages[0].content"),
    ("/v1/messages", {"system": "Be \udc80 brief.",
                      "messages": [{"role": "user", "content": "a"}]},
     "system")])
def test_a_responses_or_messages_string_with_a_lone_surrogate_is_a_400(
        monkeypatch, path, body, field):
    """The Responses and Messages routes parse their own body, and a lone
    surrogate escape in it reached the tokenizer's UTF-8 encode as a 500:
    each refuses it with the chat route's 400, in its own error shape."""
    import json

    from fastapi.testclient import TestClient

    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    anthropic = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.anthropic")
    runtime = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.runtime").runtime
    cached = lambda *a, **k: (  # noqa: E731
        None, _QwenProcessor(), {"model_type": "qwen3_5"})
    monkeypatch.setattr(openai, "get_cached_model", cached)
    monkeypatch.setattr(anthropic, "get_cached_model", cached)
    monkeypatch.setattr(runtime, "response_generator", _Tokenizing(),
                        raising=False)
    monkeypatch.setattr(_APP.app, "exception_handlers",
                        dict(_APP.app.exception_handlers))
    monkeypatch.setattr(_APP.app, "middleware_stack", None)
    sp.install_resolver_error_handlers()
    sp.install_api_contract()
    client = TestClient(_APP.app, raise_server_exceptions=False)
    r = client.post(path, content=json.dumps(
        {"model": "m", "max_tokens": 1, "max_output_tokens": 16,
         **body}).encode(), headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    message = f"{field} holds a lone surrogate escape"
    if path == "/v1/messages":
        assert r.json() == {"type": "error", "error": {
            "type": "invalid_request_error", "message": message}}
    else:
        err = r.json()["error"]
        assert (err["type"], err["message"], err["param"]) == (
            "invalid_request_error", message, field)


class _CountsTokens(_Tokenizing):
    """The engine's preprocess, which tokenizes the prompt to count it."""

    def _cpu_preprocess(self, prompt, images=None, audio=None):
        prompt.encode("utf-8")
        return {"input_ids": [1, 2, 3]}


@pytest.mark.parametrize("path, body, field", [
    ("/v1/responses/input_tokens", {"input": "a \ud800 cat"}, "input"),
    ("/v1/messages/count_tokens", {"messages": [
        {"role": "user", "content": "a \ud800 cat"}]},
     "messages[0].content")])
def test_a_token_count_of_a_string_with_a_lone_surrogate_is_a_400(
        monkeypatch, path, body, field):
    """The token counting routes of Responses and Messages tokenize the
    prompt as their generation routes do: a lone surrogate escape gets
    the same 400, in the route's own error shape."""
    import json

    from fastapi.testclient import TestClient

    openai = importlib.import_module("gmlx._vendor.mlx_vlm.server.openai")
    anthropic = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.anthropic")
    runtime = importlib.import_module(
        "gmlx._vendor.mlx_vlm.server.runtime").runtime
    cached = lambda *a, **k: (  # noqa: E731
        None, _QwenProcessor(), {"model_type": "qwen3_5"})
    monkeypatch.setattr(openai, "get_cached_model", cached)
    monkeypatch.setattr(anthropic, "get_cached_model", cached)
    monkeypatch.setattr(runtime, "response_generator", _CountsTokens(),
                        raising=False)
    monkeypatch.setattr(_APP.app, "exception_handlers",
                        dict(_APP.app.exception_handlers))
    monkeypatch.setattr(_APP.app, "middleware_stack", None)
    sp.install_resolver_error_handlers()
    sp.install_api_contract()
    client = TestClient(_APP.app, raise_server_exceptions=False)
    r = client.post(path, content=json.dumps(
        {"model": "m", "max_tokens": 1, **body}).encode(),
        headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    message = f"{field} holds a lone surrogate escape"
    if path.startswith("/v1/messages"):
        assert r.json() == {"type": "error", "error": {
            "type": "invalid_request_error", "message": message}}
    else:
        err = r.json()["error"]
        assert (err["type"], err["message"], err["param"]) == (
            "invalid_request_error", message, field)
