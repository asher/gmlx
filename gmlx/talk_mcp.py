"""MCP tool source for the built-in assistant brain.

Connects the ``assistant.mcp`` servers (stdio ``command`` or streamable-HTTP
``url``) via the official ``mcp`` SDK ([assistant] extra) and exposes each
server's tools as :class:`~gmlx.assistant_brain.Tool` entries in a
:class:`~gmlx.assistant_brain.ToolRegistry` - the assistant brain never
knows tools came from MCP.

The SDK is asyncio-native and its sessions are async context managers; the
talk loop is threads + queues. :class:`McpToolHost` bridges: one daemon
thread runs an asyncio event loop that owns every session (each held open by
a task parked on a shutdown event), and each ``Tool.call`` submits
``call_tool`` to that loop with ``run_coroutine_threadsafe`` and blocks on
the result. The whole SDK surface is behind the ``open_session`` seam, so
tests drive the host with a fake async session and no SDK installed.

Per-server connection failures degrade to a warning line (the loop runs with
the tools that did come up); only a missing SDK when servers are configured
is a hard hint to install the extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import re
import sys
import threading

from .assistant_brain import Tool, ToolRegistry


class TalkMcpError(RuntimeError):
    """An MCP server could not be reached / initialized."""


ASSISTANT_EXTRA_HINT = ("MCP tools need the assistant extra: "
                        "pip install 'gmlx[assistant]'")


def _result_text(result) -> str:
    """A ``CallToolResult`` -> the string the model sees. Text content joins;
    non-text items are named; ``isError`` becomes an error: prefix."""
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(f"[{getattr(item, 'type', 'non-text')} content]")
    out = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"error: {out or 'tool failed'}"
    return out


def stderr_log_path(name: str):
    """Where a stdio server's stderr lands - a per-server file in the cache
    dir, so tool-server logging never interleaves with the REPL / voice UI."""
    from .lifecycle import runtime_dir
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "server"
    return runtime_dir() / f"mcp-{safe}.log"


@contextlib.contextmanager
def _stderr_log(name: str):
    """Open the per-server stderr sink (append); an unwritable cache dir
    degrades to the parent's stderr rather than losing the server."""
    try:
        f = open(stderr_log_path(name), "a", encoding="utf-8",
                 errors="replace")
    except OSError:
        yield sys.stderr
        return
    with f:
        yield f


@contextlib.asynccontextmanager
async def _open_session(server):
    """Default ``open_session``: yield an initialized-capable ClientSession
    for one :class:`~gmlx.config.McpServerCfg` (imports the SDK here so
    the module stays importable without the [assistant] extra)."""
    from mcp import ClientSession
    if server.url:
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(server.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                yield session
    else:
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client
        # `env:` is additive over the SDK's minimal default (HOME/PATH/...), not
        # over os.environ: a tool server is third-party code and must not inherit
        # this process's HF tokens and API keys just because one var was set.
        params = StdioServerParameters(
            command=server.command[0], args=list(server.command[1:]),
            env={**get_default_environment(), **server.env}
            if server.env else None)
        with _stderr_log(server.name) as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session


class McpToolHost:
    """Owns the event-loop thread and every open MCP session (see module
    docstring). ``connect`` returns one server's tools; ``close`` shuts the
    sessions and the loop down."""

    def __init__(self, *, call_timeout_s: float = 60.0,
                 connect_timeout_s: float = 20.0, open_session=None):
        self.call_timeout_s = call_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._open_session = open_session or _open_session
        self._shutdowns: list = []               # asyncio.Event per session
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="talk-mcp")
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout)

    async def _serve(self, server, box: dict, ready: threading.Event,
                     shutdown) -> None:
        """The per-server task: enter the session, list tools, park until
        shutdown. Session lifetime == task lifetime, as the SDK requires."""
        try:
            async with self._open_session(server) as session:
                await session.initialize()
                listing = await session.list_tools()
                box["session"] = session
                box["tools"] = list(getattr(listing, "tools", None) or [])
                ready.set()
                await shutdown.wait()
        except Exception as e:                    # noqa: BLE001 - to warning
            box["error"] = e
            ready.set()

    def connect(self, server) -> list:
        """Open ``server`` and return its tools as :class:`Tool` entries.
        Raises :class:`TalkMcpError` on failure/timeout."""
        box: dict = {}
        ready = threading.Event()

        async def start():
            shutdown = asyncio.Event()
            self._shutdowns.append(shutdown)
            asyncio.ensure_future(self._serve(server, box, ready, shutdown))

        self._submit(start(), 5.0)
        if not ready.wait(self.connect_timeout_s):
            raise TalkMcpError(
                f"mcp server {server.name!r}: no response within "
                f"{self.connect_timeout_s:g}s")
        if "error" in box:
            raise TalkMcpError(f"mcp server {server.name!r}: {box['error']}")
        session = box["session"]
        tools = []
        for t in box["tools"]:
            tools.append(self._wrap(session, t))
        return tools

    def _wrap(self, session, t) -> Tool:
        name = t.name

        def call(args: dict) -> str:
            result = self._submit(session.call_tool(name, args or {}),
                                  self.call_timeout_s)
            return _result_text(result)

        return Tool(name=name, description=t.description or "",
                    parameters=dict(getattr(t, "inputSchema", None) or {}),
                    call=call)

    def close(self) -> None:
        if not self._loop.is_running():
            return

        async def stop():
            for evt in self._shutdowns:
                evt.set()
            # Give the parked _serve tasks loop time to unwind their session
            # stacks (stdio transports terminate child processes here).
            tasks = [t for t in asyncio.all_tasks()
                     if t is not asyncio.current_task() and not t.done()]
            if tasks:
                await asyncio.wait(tasks, timeout=3.0)
            # Still pending = stuck before shutdown.wait() (say, a server
            # hanging in initialize()). Cancel so the session stack unwinds
            # and terminates its child process - destroying the task with
            # the loop would leak it.
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.wait(pending, timeout=2.0)

        with contextlib.suppress(Exception):
            self._submit(stop(), 7.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


def connect_servers(servers, *, call_timeout_s: float = 60.0,
                    open_session=None):
    """Build the assistant's tool registry from the configured MCP servers.

    Returns ``(host | None, registry, warnings)``: per-server failures become
    warning lines and the rest still connect; a missing SDK (and no injected
    seam) yields no host and the install hint. Tool-name collisions across
    servers are disambiguated with a ``<server>_`` prefix. The caller owns
    ``host.close()``."""
    registry = ToolRegistry()
    servers = list(servers or [])
    if not servers:
        return None, registry, []
    if open_session is None and importlib.util.find_spec("mcp") is None:
        return None, registry, [ASSISTANT_EXTRA_HINT]
    host = McpToolHost(call_timeout_s=call_timeout_s,
                       open_session=open_session)
    warnings: list = []
    for server in servers:
        try:
            tools = host.connect(server)
        except TalkMcpError as e:
            msg = str(e)
            if server.command:
                msg += f" (server stderr: {stderr_log_path(server.name)})"
            warnings.append(msg)
            continue
        if not tools:
            warnings.append(f"mcp server {server.name!r}: no tools")
        for tool in tools:
            if registry.get(tool.name) is not None:
                tool.name = f"{server.name}_{tool.name}"
            try:
                registry.add(tool)
            except ValueError:
                # The prefixed name can still collide (another server literally
                # named `<name>_<tool>`); degrade to a warning rather than let it
                # abort the whole assistant startup.
                warnings.append(
                    f"mcp server {server.name!r}: tool {tool.name!r} still "
                    f"collides after prefixing; skipped")
    return host, registry, warnings
