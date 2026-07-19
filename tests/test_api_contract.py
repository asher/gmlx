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

pytest.importorskip("mlx_vlm")

from fastapi import Request  # noqa: E402

from gmlx import server_patches as sp  # noqa: E402
from gmlx.server_patches import _common as sp_common  # noqa: E402
from gmlx.server_patches import api_contract as sp_api  # noqa: E402

_APP = importlib.import_module("mlx_vlm.server.app")
_SCHEMAS = importlib.import_module("mlx_vlm.server.schemas")

_DOCS = Path(__file__).resolve().parents[1] / "docs" / "server-config.md"


@pytest.fixture(autouse=True)
def _restore_routes():
    saved = list(_APP.app.router.routes)
    yield
    _APP.app.router.routes[:] = saved


# -- allowlist drift guards ------------------------------------------------
# These fail loudly when a consumed field appears that the allowlists don't
# know about (which would make the warning fire on a field that actually
# works, or stay silent on one that doesn't). To update: re-read what the
# handler / _build_gen_args / gmlx patches read off the request, adjust the
# sets in gmlx/server_patches/api_contract.py, the docs/server-config.md
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
    scraped = _scrape_request_reads(
        _APP._build_gen_args, _APP._extract_response_format_schema)
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


def test_doc_table_rows_match_allowlists():
    """Cross-check the docs/server-config.md "Parameter support" table
    against the consumed sets: an `ignored` cell must be off the allowlist,
    a `honored` cell on it. Cells reading template/rejected/n-a are prose."""
    text = _DOCS.read_text()
    m = re.search(r"### Parameter support\n(.*?)\n\n(?:###|##) ", text,
                  re.DOTALL)
    assert m, "server-config.md lost its '### Parameter support' table"
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


# -- wrapper behavior --------------------------------------------------------

def _capture_warnings():
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logger = logging.getLogger("gmlx.server_patches.api_contract")
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


def test_top_logprobs_above_cap_warns(monkeypatch):
    monkeypatch.delenv("TOP_LOGPROBS_K", raising=False)
    records, handler, logger = _capture_warnings()
    try:
        client = _client_with("/v1/chat/completions", _chat_stub({}))
        r = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "logprobs": True, "top_logprobs": 5})
        assert r.status_code == 200
        warns = [m for m in records if "TOP_LOGPROBS_K" in m]
        assert len(warns) == 1
        assert "top_logprobs=5" in warns[0]
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

    monkeypatch.setenv("TOP_LOGPROBS_K", "50")
    req = types.SimpleNamespace(top_logprobs=30)
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_top_logprobs("/v1/chat/completions", req)
    # the engine clamps the effective cap to 20, so 30 must warn even though
    # the raw env value says 50
    assert any("top_logprobs=30 exceeds the server cap 20" in r.message
               for r in caplog.records)
    caplog.clear()
    req = types.SimpleNamespace(top_logprobs=15)
    with caplog.at_level(logging.WARNING, logger=sp_api.__name__):
        sp_api._maybe_warn_top_logprobs("/v1/chat/completions", req)
    assert not caplog.records                     # within the effective cap
