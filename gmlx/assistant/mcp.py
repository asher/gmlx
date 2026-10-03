"""MCP tool source for the built-in assistant brain.

Connects the ``assistant.mcp`` servers (stdio ``command`` or streamable-HTTP
``url``) via the official ``mcp`` SDK ([assistant] extra) and exposes each
server's tools as :class:`~gmlx.assistant.brain.Tool` entries in a
:class:`~gmlx.assistant.brain.ToolRegistry` - the assistant brain never
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

A stdio tool server never runs from a folder that a container client can
write, and its PATH leaves out such folders (see :mod:`gmlx.serve.programs`).
It never starts in such a folder either, because a tool server can load
code from the folder that it runs in, as npx and ``python -m`` do. A
container session can share a folder after the tool server starts, so the
host checks the program, the working folder and the PATH of the tool
server again before each tool call. When a share now holds one of them, the
host stops the tool server. It starts it again with a PATH without the
shared folder, or refuses the call when the share holds the program or the
working folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import re
import sys
import threading

from .brain import Tool, ToolRegistry


class TalkMcpError(RuntimeError):
    """An MCP server could not be reached / initialized."""


def unsupported_sdk() -> str | None:
    """The warning for an installed ``mcp`` release that gmlx cannot use,
    or None. mcp 2 renamed the streamable HTTP client and moved its types
    to snake_case fields, so a tool server gets no parameters and its
    errors read as results. An install that kept mcp 2 from an earlier
    gmlx keeps it until the extra is installed again."""
    from importlib import metadata

    try:
        version = metadata.version("mcp")
    except metadata.PackageNotFoundError:
        return None
    if not isinstance(version, str):           # a dist-info without its metadata
        return None
    major = version.split(".", 1)[0]
    if not major.isdigit() or int(major) < 2:
        return None
    from gmlx.commands.extras import install_hint
    return (f"MCP tools need an mcp release before 2.0, and mcp {version} is installed: "
            f"{install_hint('assistant')}")


def assistant_extra_hint() -> str:
    """The missing-assistant-extra warning. A function, not a constant: the
    install command depends on how gmlx itself was installed."""
    from gmlx.commands.extras import install_hint
    return f"MCP tools need the assistant extra: {install_hint('assistant')}"


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
    from gmlx.serve.lifecycle import runtime_dir
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


def tool_step(lookup) -> str:
    """The next step when gmlx will not or cannot run the tool server
    program of ``lookup``, a :class:`gmlx.serve.programs.Lookup`."""
    if lookup.refusal is not None:
        return ("Install the tool server in a folder that no container session shares, and "
                "give that path as its command in the config's mcp list.")
    return (f"Install {lookup.command}, or give its full path as the command of the tool "
            "server in the config's mcp list.")


def _note(server, text: str) -> None:
    """Write a line from gmlx to the stderr log of the tool server."""
    with _stderr_log(server.name) as f:
        print(f"[gmlx] {text}", file=f, flush=True)


def _working_folder() -> str | None:
    """The folder that a stdio tool server runs in, the working folder of
    gmlx, or None when that folder is gone, so that no file can be added
    to it. A tool server can load code from the folder that it runs in, as
    npx does with the node_modules folder of a project and ``python -m``
    does with a module file. Raises
    :class:`gmlx.serve.programs.ProgramRefused` when a container client can
    write that folder."""
    from gmlx.serve import programs
    try:
        cwd = os.getcwd()
    except OSError:
        return None
    hit = programs.why(cwd)
    if hit is not None:
        reason, step = hit
        raise programs.ProgramRefused(" ".join(filter(None, [
            f"gmlx will not start the tool server in {programs.tilde(cwd)}, the folder that "
            f"gmlx runs in, because that folder {reason}. A tool server can load code from the "
            "folder that it runs in, as npx and python -m do. Start gmlx in a folder that no "
            "container session shared, such as your home folder.",
            step])))
    return cwd


class _ToolServer:
    """The session of a stdio tool server, with the program, the working
    folder and the PATH that the tool server started with. Every other
    attribute is the session's."""

    def __init__(self, session, program: str, folders: tuple[str, ...],
                 cwd: str | None = None):
        self._session = session
        self.program = program
        self.folders = folders
        self.cwd = cwd

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    def changed(self) -> str | None:
        """Why the tool server must start again, or None. A session that
        starts after the tool server can share a folder on its PATH, the
        folder of its program or the folder that it runs in. A tool server
        can run a program by name at each call, such as git, and it must
        not find one that a container client wrote."""
        from gmlx.serve import programs
        why = programs.refusal(self.program)
        if why is not None:
            return f"its program {programs.tilde(self.program)} {why}"
        if self.cwd is not None:
            why = programs.refusal(self.cwd)
            if why is not None:
                return f"the folder {programs.tilde(self.cwd)} that it runs in {why}"
        now = programs.skipped_now(self.folders)
        if now:
            entry, why = now[0]
            return f"the PATH entry {programs.tilde(entry)} of the tool server {why}"
        return None


@contextlib.asynccontextmanager
async def _open_session(server):
    """Default ``open_session``: yield an initialized-capable ClientSession
    for one :class:`~gmlx.config.McpServerCfg` (imports the SDK here so
    the module stays importable without the [assistant] extra). The
    session of a stdio server comes in a :class:`_ToolServer`."""
    from mcp import ClientSession
    if server.url:
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(server.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                yield session
    else:
        from mcp import StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        from gmlx.serve import programs
        # `env:` is additive over the SDK's minimal default (HOME/PATH/...), not
        # over os.environ: a tool server is third-party code and must not inherit
        # this process's HF tokens and API keys just because one var was set.
        env = {**get_default_environment(), **server.env}
        lookup = programs.look_up(server.command[0], env.get("PATH", os.defpath))
        program = programs.checked(lookup, tool_step(lookup), "gmlx")
        # A tool server can run programs by name too, such as the node that
        # `#!/usr/bin/env node` names, so its PATH has the same folders.
        env["PATH"] = os.pathsep.join(program.search.folders)
        path = program.path or server.command[0]
        cwd = _working_folder()
        params = StdioServerParameters(command=path, args=list(server.command[1:]), env=env,
                                       cwd=cwd)
        with _stderr_log(server.name) as errlog:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    yield _ToolServer(session, path, program.search.folders, cwd)


class _Link:
    """One server of the host: its config, and the session and the task
    that hold it open. The host can replace the session (see
    :meth:`McpToolHost._live`), and the tools of the server keep this link."""

    def __init__(self, server):
        self.server = server
        self.session = None
        self.shutdown: asyncio.Event | None = None
        self.task: asyncio.Future | None = None
        self.lock = threading.Lock()


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

    def _start(self, link: _Link) -> list:
        """Open the server of ``link``, and return its tool listing.
        Raises :class:`TalkMcpError` on failure/timeout."""
        server = link.server
        box: dict = {}
        ready = threading.Event()

        async def start():
            shutdown = asyncio.Event()
            self._shutdowns.append(shutdown)
            box["shutdown"] = shutdown
            box["task"] = asyncio.ensure_future(self._serve(server, box, ready, shutdown))

        self._submit(start(), 5.0)
        if not ready.wait(self.connect_timeout_s):
            raise TalkMcpError(
                f"mcp server {server.name!r}: no response within "
                f"{self.connect_timeout_s:g}s")
        if "error" in box:
            raise TalkMcpError(f"mcp server {server.name!r}: {box['error']}")
        link.session, link.shutdown, link.task = box["session"], box["shutdown"], box["task"]
        return box["tools"]

    def _stop(self, link: _Link) -> None:
        """End the session of ``link``, which stops its tool server."""
        shutdown, task = link.shutdown, link.task
        link.session = link.shutdown = link.task = None
        if shutdown is None or task is None:
            return

        async def stop():
            shutdown.set()
            await asyncio.wait([task], timeout=3.0)
            if not task.done():
                task.cancel()
                await asyncio.wait([task], timeout=2.0)

        with contextlib.suppress(Exception):
            self._submit(stop(), 7.0)

    def _live(self, link: _Link):
        """The session of ``link`` for a tool call. When the session gives
        a reason to start again (see :meth:`_ToolServer.changed`), or when
        the last start failed, the host starts the tool server again first.
        Raises :class:`TalkMcpError` when that start fails. The log of the
        tool server gets each stop, and the result of each start."""
        with link.lock:
            changed = getattr(link.session, "changed", None)
            why = changed() if callable(changed) else None
            if link.session is not None and why is None:
                return link.session
            if why is not None:
                _note(link.server, f"gmlx stops the tool server, because {why}.")
                self._stop(link)
            try:
                self._start(link)
            except TalkMcpError as e:
                reason = str(e).removeprefix(f"mcp server {link.server.name!r}: ")
                _note(link.server, f"gmlx did not start the tool server again: {reason}")
                raise
            _note(link.server, "gmlx started the tool server again.")
            return link.session

    def connect(self, server) -> list:
        """Open ``server`` and return its tools as :class:`Tool` entries.
        Raises :class:`TalkMcpError` on failure/timeout."""
        link = _Link(server)
        return [self._wrap(link, t) for t in self._start(link)]

    def _wrap(self, link: _Link, t) -> Tool:
        name = t.name

        def call(args: dict) -> str:
            result = self._submit(self._live(link).call_tool(name, args or {}),
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
        # A loop that is not closed gives a ResourceWarning when it is
        # collected, which can land in any later warning check.
        if not self._thread.is_alive():
            self._loop.close()


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
        return None, registry, [assistant_extra_hint()]
    if open_session is None and (newer := unsupported_sdk()):
        return None, registry, [newer]
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
