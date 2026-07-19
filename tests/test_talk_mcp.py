#!/usr/bin/env python3
"""McpToolHost / connect_servers tests through the ``open_session`` seam - a
fake async session, no ``mcp`` SDK, no subprocesses. Exercises the sync<->
asyncio bridge itself (the host's loop thread runs for real)."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from types import SimpleNamespace

import pytest

from gmlx import talk_mcp
from gmlx.config import McpServerCfg
from gmlx.talk_mcp import (ASSISTANT_EXTRA_HINT, McpToolHost,
                               TalkMcpError, _result_text, connect_servers)


def _tooldef(name, desc="a tool"):
    return SimpleNamespace(name=name, description=desc,
                           inputSchema={"type": "object", "properties": {}})


class FakeSession:
    def __init__(self, tools):
        self.tools = tools
        self.initialized = False
        self.calls = []

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"{name} ok")],
            isError=False)


def _fake_open(sessions):
    """open_session seam: pop a FakeSession (or raise an Exception instance)
    per server, recording which server asked."""
    opened = []

    @contextlib.asynccontextmanager
    async def open_session(server):
        opened.append(server.name)
        item = sessions[server.name]
        if isinstance(item, Exception):
            raise item
        yield item

    open_session.opened = opened
    return open_session


def _srv(name):
    return McpServerCfg(name=name, command=["fake-server"])


def test_connect_servers_builds_registry_and_calls_bridge():
    session = FakeSession([_tooldef("get_time"), _tooldef("get_weather")])
    host, reg, warnings = connect_servers(
        [_srv("clock")], open_session=_fake_open({"clock": session}))
    try:
        assert warnings == [] and len(reg) == 2
        assert session.initialized
        tool = reg.get("get_time")
        assert tool.spec()["function"]["description"] == "a tool"
        assert tool.call({"tz": "UTC"}) == "get_time ok"   # sync -> asyncio
        assert session.calls == [("get_time", {"tz": "UTC"})]
    finally:
        host.close()


def test_connect_servers_degrades_per_server():
    ok = FakeSession([_tooldef("read_file")])
    host, reg, warnings = connect_servers(
        [_srv("bad"), _srv("files")],
        open_session=_fake_open({"bad": RuntimeError("boom"),
                                 "files": ok}))
    try:
        assert len(reg) == 1 and reg.get("read_file")
        assert len(warnings) == 1 and "'bad'" in warnings[0]
        assert "boom" in warnings[0]
    finally:
        host.close()


def test_tool_name_collision_gets_server_prefix():
    a, b = FakeSession([_tooldef("search")]), FakeSession([_tooldef("search")])
    host, reg, warnings = connect_servers(
        [_srv("web"), _srv("docs")],
        open_session=_fake_open({"web": a, "docs": b}))
    try:
        assert sorted(reg.names()) == ["docs_search", "search"]
        # the prefixed registry name still calls the server's ORIGINAL name
        assert reg.get("docs_search").call({}) == "search ok"
        assert b.calls == [("search", {})]
    finally:
        host.close()


def test_no_servers_and_missing_sdk_paths(monkeypatch):
    host, reg, warnings = connect_servers([])
    assert host is None and not reg and warnings == []

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: None if n == "mcp" else real(n, *a))
    host, reg, warnings = connect_servers([_srv("clock")])
    assert host is None and not reg and warnings == [ASSISTANT_EXTRA_HINT]


def test_connect_timeout_raises():
    @contextlib.asynccontextmanager
    async def slow_open(server):
        await asyncio.sleep(0.3)                 # ready never set in time
        yield FakeSession([])

    host = McpToolHost(connect_timeout_s=0.05, open_session=slow_open)
    try:
        with pytest.raises(TalkMcpError, match="no response"):
            host.connect(_srv("slow"))
    finally:
        host.close()


def test_result_text_shapes():
    ok = SimpleNamespace(content=[SimpleNamespace(type="text", text="a"),
                                  SimpleNamespace(type="image")],
                         isError=False)
    assert _result_text(ok) == "a\n[image content]"
    err = SimpleNamespace(content=[SimpleNamespace(type="text", text="nope")],
                          isError=True)
    assert _result_text(err) == "error: nope"
    assert _result_text(SimpleNamespace(content=[], isError=True)) == \
        "error: tool failed"


def test_extras_table_has_assistant():
    from gmlx import extras
    assert extras.extra_packages("assistant") == ["mcp"]
    assert isinstance(extras.extra_installed("assistant"), bool)


def test_stderr_log_path_sanitizes_and_lands_in_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    p = talk_mcp.stderr_log_path("my tools/v2!")
    assert p == tmp_path / "gmlx" / "mcp-my-tools-v2.log"
    assert talk_mcp.stderr_log_path("///") .name == "mcp-server.log"


def test_stderr_log_opens_append_and_degrades(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with talk_mcp._stderr_log("clock") as f:
        f.write("noise\n")
    with talk_mcp._stderr_log("clock") as f:      # append, not truncate
        f.write("more\n")
    assert (tmp_path / "gmlx" / "mcp-clock.log").read_text() == \
        "noise\nmore\n"
    # unwritable sink -> parent stderr, server still comes up
    import sys as _sys
    monkeypatch.setattr(talk_mcp, "stderr_log_path",
                        lambda name: tmp_path / "absent" / "x.log")
    with talk_mcp._stderr_log("clock") as f:
        assert f is _sys.stderr


def test_stdio_connect_failure_names_stderr_log(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    host, reg, warnings = connect_servers(
        [_srv("bad")], open_session=_fake_open({"bad": RuntimeError("boom")}))
    try:
        assert len(warnings) == 1 and "boom" in warnings[0]
        assert "server stderr:" in warnings[0]
        assert "mcp-bad.log" in warnings[0]
    finally:
        host.close()


def test_stdio_env_is_additive_over_the_sdk_default(monkeypatch, tmp_path):
    """`env:` must not switch the child to the parent's os.environ: a tool
    server is third-party code and must never see this process's secrets."""
    pytest.importorskip("mcp")           # the [assistant] extra owns the SDK
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "super-secret")
    from mcp.client import stdio as mcp_stdio

    captured: dict = {}

    class _Stop(Exception):
        pass

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        captured["env"] = params.env
        raise _Stop
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)
    srv = McpServerCfg(name="brave", command=["fake-server"],
                       env={"BRAVE_API_KEY": "k"})

    async def _go():
        async with talk_mcp._open_session(srv):
            pass  # pragma: no cover

    with pytest.raises(_Stop):
        asyncio.run(_go())

    env = captured["env"]
    assert env["BRAVE_API_KEY"] == "k"       # the configured var is passed
    assert "PATH" in env                     # the SDK default is the base
    assert "HF_TOKEN" not in env             # the parent environment is not
