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

import gmlx.assistant.mcp as talk_mcp
from gmlx.config import McpServerCfg
from gmlx.assistant.mcp import (McpToolHost, assistant_extra_hint,
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
    assert host is None and not reg and warnings == [assistant_extra_hint()]


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
    import gmlx.commands.extras as extras
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
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "fake-server").write_text("#!/bin/sh\n")
    (tools / "fake-server").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin")
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


def _stdio_spawns(monkeypatch) -> list:
    """Stand in for the SDK's stdio_client. Records the parameters of each
    spawn, and stops before a process starts."""
    pytest.importorskip("mcp")           # the [assistant] extra owns the SDK
    from mcp.client import stdio as mcp_stdio

    spawns: list = []

    @contextlib.asynccontextmanager
    async def fake_stdio_client(params, errlog=None):
        spawns.append(params)
        raise RuntimeError("stopped before the spawn")
        yield  # pragma: no cover - unreachable, keeps this an async generator

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)
    return spawns


def _open(srv) -> None:
    async def _go():
        async with talk_mcp._open_session(srv):
            pass  # pragma: no cover
    asyncio.run(_go())


def _shared(*folders) -> None:
    import json

    from gmlx.container.state import data_path
    from gmlx.safe_path import canonical
    data_path().mkdir(parents=True, exist_ok=True)
    (data_path() / "shared.json").write_text(
        json.dumps({"shared": [canonical(f) for f in folders]}))


def _tool(folder, name="mcp-tool"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("#!/bin/sh\n")
    (folder / name).chmod(0o755)
    return folder / name


def test_a_tool_server_never_runs_from_a_folder_a_container_client_can_write(
        monkeypatch, tmp_path):
    """A tool server command on PATH in a shared .venv/bin is skipped, and
    the tool server's own PATH leaves out that folder and every relative
    entry, so `#!/usr/bin/env node` cannot find a client's node either."""
    from gmlx.container import settings
    spawns = _stdio_spawns(monkeypatch)
    monkeypatch.setattr(settings, "SYSTEM_PATH", "/usr/bin:/bin")
    share, tools = tmp_path / "proj", tmp_path / "tools"
    _tool(share / ".venv" / "bin")
    _tool(tools)
    _shared(share)
    monkeypatch.setenv("PATH", f"{share}/.venv/bin:.::{tools}:/usr/bin:/bin")
    with pytest.raises(RuntimeError, match="stopped before the spawn"):
        _open(McpServerCfg(name="t", command=["mcp-tool", "--x"]))
    (params,) = spawns
    assert params.command == str(tools / "mcp-tool")
    assert params.args == ["--x"]
    assert params.env["PATH"] == f"{tools}:/usr/bin:/bin"


def test_a_tool_server_named_by_its_path_in_a_share_is_refused(monkeypatch, tmp_path):
    spawns = _stdio_spawns(monkeypatch)
    share = tmp_path / "proj"
    program = _tool(share / "bin")
    _shared(share)
    with pytest.raises(Exception, match=(
            f"gmlx will not run {program}, because it lies in .*proj, a folder that a "
            "container session shared read-write. A container client could have written "
            "that file. Install the tool server in a folder that no container session "
            "shares, and give that path as its command in the config's mcp list.")):
        _open(McpServerCfg(name="t", command=[str(program)]))
    assert spawns == []


def test_a_tool_server_through_a_link_into_a_share_is_refused(monkeypatch, tmp_path):
    spawns = _stdio_spawns(monkeypatch)
    share, links = tmp_path / "proj", tmp_path / "bin"
    _tool(share / "tools")
    links.mkdir()
    (links / "mcp-tool").symlink_to(share / "tools" / "mcp-tool")
    _shared(share)
    monkeypatch.setenv("PATH", f"{links}:/usr/bin:/bin")
    with pytest.raises(Exception, match=f"gmlx will not run {links}/mcp-tool, because it "
                                        "leads to .*proj/tools/mcp-tool"):
        _open(McpServerCfg(name="t", command=["mcp-tool"]))
    assert spawns == []


def test_connect_servers_gives_the_refusal_as_a_warning(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    share = tmp_path / "proj"
    program = _tool(share / "bin")
    _shared(share)
    host, registry, warnings = connect_servers(
        [McpServerCfg(name="t", command=[str(program)])])
    try:
        assert registry.names() == []
        assert len(warnings) == 1 and "gmlx will not run" in warnings[0]
    finally:
        if host is not None:
            host.close()


def test_close_closes_the_event_loop():
    """A loop that close() leaves open gives a ResourceWarning when it is
    collected, such as in the middle of the config check of gmlx doctor."""
    host = McpToolHost(open_session=_fake_open([]))
    host.close()
    assert host._loop.is_closed()
    host.close()                        # a second close does nothing
