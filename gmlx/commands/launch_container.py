"""``gmlx launch <client> --container``: the launch order in container mode.

The client's handler in ``launch.py`` still resolves the server, writes the
client's configuration and names the command, but it does so with ``HOME``
pointed at the client's private home, and its finish helper hands the
command back here instead of replacing the process. This module decides
container mode, takes the session lock, resolves the shares and the image,
and runs the supervisor in ``gmlx.container.session``.
"""

from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import os
import shlex
import signal
import socket
import sys
import threading
import urllib.error
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Callable

from gmlx.config import (LAUNCH_CLIENTS, ConfigError, LaunchCfg, launch_block_enables,
                         load_launch_settings)
from gmlx.container import cli, confine, images, runtime, session, settings
from gmlx.container.cli import ContainerError
from gmlx.container.settings import Mount, SettingsError
from gmlx.container.text import printable, printable_lines

# Flags that only mean something in container mode, by argparse dest.
CONTAINER_FLAGS = {"mount": "--mount", "mount_cwd": "--mount-cwd", "image": "--image",
                   "rebuild": "--rebuild", "reseed": "--reseed", "network": "--network",
                   "shell": "--shell"}
# Flags an attaching --shell ignores, since the running session already has
# its server and model, and the flags it refuses, which shape a new session.
_ATTACH_IGNORED = {
    "model": None, "base_url": None, "host": None, "port": None, "api_key": None,
    "no_start": False, "start_timeout": 0.0, "no_keep": False,
}
_ATTACH_REFUSED = {
    "provider_id": "gmlx", "config_path": None, "config_only": False, "dsh_profile": None,
    "mount": [], "mount_cwd": None, "image": None, "rebuild": False, "reseed": False,
    "network": None,
}
# Printable ASCII up to the end of the line, so a URL with a terminal
# control in it opens nothing.
_DSH_URL_LINE = r"dsh web: ([\x21-\x7e]+)(?=\s)"


def _say(line: str) -> None:
    print(printable(line), flush=True)


def _flag_name(dest: str, value) -> str:
    """The flag as the user typed it. ``--no-mount-cwd`` sets False."""
    if dest == "mount_cwd" and value is False:
        return "--no-mount-cwd"
    return CONTAINER_FLAGS.get(dest) or "--" + dest.replace("_", "-")


def _listed(words: list[str]) -> str:
    """``a``, ``a and b`` or ``a, b and c``."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _flag_set(a, dest: str) -> bool:
    value = getattr(a, dest, None)
    if dest == "mount_cwd":           # --mount-cwd and --no-mount-cwd both count
        return value is not None
    return value not in (None, False, [])


def container_mode(a, ap) -> tuple[bool, LaunchCfg]:
    """Whether this launch runs in a container, from the flags and the
    ``enabled`` keys of the user-level config, and those settings. A broken
    ``launch`` block stops a launch that asks for container mode on the
    command line, and a launch of a client the block may turn on. Only
    ``--no-container``, or a block that clearly leaves this client off, runs
    the client on the Mac, with one notice."""
    implied = [_flag_name(dest, getattr(a, dest, None))
               for dest in CONTAINER_FLAGS if _flag_set(a, dest)]
    if a.container is False and implied:
        ap.error(f"{implied[0]} applies only in container mode, so it cannot go with "
                 "--no-container")
    try:
        launch_cfg = load_launch_settings()
    except ConfigError as e:
        if a.container or implied:
            raise
        if a.container is None:
            # The block may be broken in a key that has nothing to do with
            # this client, but a block that turns container mode on must
            # never run the client on the Mac without the sandbox.
            on, path = launch_block_enables(a.harness)
            if on is not False:
                verb = "turns" if on else "may turn"
                first = str(e).rstrip()
                # A YAML error ends on the parser's location lines, so the
                # next sentence starts a line of its own.
                sep = "\n" if "\n" in first else " "
                if sep == " " and first[-1:] not in ".?!":
                    first += "."
                subject = "That file" if str(path) in first else str(path)
                raise ConfigError(
                    f"{first}{sep}{subject} {verb} container mode on for "
                    f"{a.harness}, so launch stops until the launch block is fixed. Pass "
                    "--no-container to run it on the Mac instead.") from None
        print(printable_lines(f"[launch] ignoring the launch settings, so {a.harness} runs "
                              f"on the Mac: {e}"), file=sys.stderr)
        return False, LaunchCfg()
    if a.container is not None:
        return a.container, launch_cfg
    if implied:
        return True, launch_cfg
    return bool(launch_cfg.container.for_client(a.harness).enabled), launch_cfg


@contextlib.contextmanager
def guest_home(home: Path):
    """Point ``HOME`` at the private home while a handler runs, and hide the
    variables that would send it to the user's own files."""
    hidden = ("DSH_HOME", "HERMES_HOME")
    saved = {k: os.environ.get(k) for k in ("HOME", *hidden)}
    os.environ["HOME"] = str(home)
    for key in hidden:
        os.environ.pop(key, None)
    try:
        # The guest can plant links in the private home, so every file the
        # handler touches is checked against them.
        with confine.confined(home):
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _server_endpoint(a) -> tuple[str, int]:
    """The server host and port this launch will target, before the server
    check runs."""
    from gmlx.serve import lifecycle

    if a.base_url:
        split = urllib.parse.urlsplit(a.base_url)
        try:
            port = split.port
        except ValueError:
            port = None
        return a.host or split.hostname or "127.0.0.1", int(
            a.port or port or (443 if split.scheme == "https" else 80))
    if a.host or a.port:
        return a.host or "127.0.0.1", int(a.port or 8080)
    host, port = lifecycle.auto_target(None, None)
    return host, int(port)


class _Signalled(BaseException):
    """A SIGTERM or SIGHUP arrived during step 8. It is not an Exception, so
    no ``except Exception`` on the way can stop the launch from exiting."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


@contextlib.contextmanager
def _signals_raise():
    """Turn SIGTERM and SIGHUP into an exception while the block runs, so
    every ``finally`` in it runs. The exit code is 128 plus the signal."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield                          # only the main thread can set handlers
        return

    def raise_it(signum, _frame):
        raise _Signalled(signum)
    saved = {sig: signal.signal(sig, raise_it) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def _cwd() -> str:
    try:
        return os.getcwd()
    except FileNotFoundError:
        raise SettingsError("the current folder no longer exists. Change to a folder "
                            "that exists, then launch again.") from None


class _ServerCheck:
    """What :func:`_server_precheck` found. ``rc`` stops the launch with
    that exit code. ``missing`` is the line a dry run prints when no server
    answers. ``offered`` is the answer of the session probe of ``base``."""

    def __init__(self, rc: int | None = None, missing: str | None = None,
                 base: str | None = None, offered: bool | None = None):
        self.rc, self.missing, self.base, self.offered = rc, missing, base, offered


def _why_unreachable(e: BaseException) -> str:
    """The reason a request failed, without Python's error number."""
    reason = getattr(e, "reason", None)
    if isinstance(e, urllib.error.HTTPError):
        return f"status {e.code}"
    if reason is not None and not isinstance(reason, str):
        e = reason
    elif reason:
        return str(reason)
    return getattr(e, "strerror", None) or str(e) or type(e).__name__


def _server_precheck(a, dry: bool) -> _ServerCheck:
    """Stop before any build or download when the server check in step 9
    would stop anyway: no server answers and there is none to start, or
    the server that answers offers no session sockets. This check starts
    nothing. A dry run goes on without a server and says what it cannot
    show."""
    from gmlx.commands import launch as L
    from gmlx.serve import lifecycle
    from gmlx.talk.client import ensure_v1_base

    if a.base_url:                    # the server check never starts a server for it
        base = ensure_v1_base(a.base_url)
        if L._server_ready(base, a.api_key):
            return _probe_sessions(a, base, dry)
        try:
            L._http_get_json(L._server_root(base) + "/health", timeout=5.0)
            why = "it does not answer as a gmlx server"
        except (urllib.error.URLError, OSError, ValueError) as e:
            why = _why_unreachable(e)
        if dry:
            return _ServerCheck(missing=(
                f"[launch] cannot reach the server at {base} ({why}), so the dry run shows "
                "no client configuration and no command. Check the URL, or start that "
                "server."))
        raise L.LaunchError(f"cannot reach the server at {base} ({why}). Check the URL, "
                            "or start that server.")
    if a.host or a.port:
        host, port = a.host or L._DEFAULT_HOST, int(a.port or L._DEFAULT_PORT)
    else:
        host, port = lifecycle.auto_target(None, None)
    base = f"http://{host}:{port}/v1"
    if L._server_ready(base, a.api_key):
        return _probe_sessions(a, base, dry)
    cfg, cfg_path = L._discover_config()
    if cfg_path is None or cfg is None or a.no_start:
        if not dry:
            # These paths of the server check print their guidance and start
            # nothing.
            return _ServerCheck(rc=L._ensure_server(a))
        why = ("no config was found to start one from" if cfg_path is None
               else f"its config {cfg_path} does not load" if cfg is None
               else "--no-start keeps launch from starting one")
        return _ServerCheck(missing=(
            f"[launch] no server answers at {base}, and {why}, so the dry run shows no "
            "client configuration and no command."))
    return _ServerCheck()


def _probe_sessions(a, base: str, dry: bool) -> _ServerCheck:
    """Check a server that answers before the image steps. A local server
    must offer session sockets, the server must have a model, and
    ``--model`` or the default model the client needs must be one of them."""
    from gmlx.commands import launch as L

    try:
        _, api_port, targets = guest_url(base)
    except SettingsError:
        return _ServerCheck()             # step 9 reports it
    key = a.api_key
    if key is None and not a.base_url and L._auth_required(base):
        split = urllib.parse.urlsplit(base)
        key = L._server_key(split.hostname or L._DEFAULT_HOST, split.port)
    check = _ServerCheck()
    if api_port is not None and uses_session(base, targets):
        offered = sessions_offered(base, key)
        if not offered and not dry:
            raise _old_server(base)
        check = _ServerCheck(base=base, offered=offered)
    L.check_model_choice(a.harness, L.probe_models(base, key, a.harness), a.model)
    return check


def guest_url(base_url: str) -> tuple[str, int | None, list]:
    """``(guest URL, API port, relay targets)`` for the server URL. An https
    URL passes through unchanged with no relay."""
    from gmlx.container.relay import resolve_targets

    split = urllib.parse.urlsplit(base_url)
    if split.scheme == "https":
        return base_url, None, []
    host = split.hostname or "127.0.0.1"
    try:
        port = split.port or 80
    except ValueError:
        raise SettingsError(f"the server URL {base_url} has a port that is not a number "
                            "from 0 to 65535.") from None
    if host in ("127.0.0.1", "0.0.0.0"):
        targets = [("127.0.0.1", port)]
    elif host in ("::1", "::"):
        targets = [("::1", port)]
    else:
        try:
            targets = resolve_targets(host, port)
        except OSError as e:
            raise SettingsError(f"cannot resolve the server host {host} "
                                f"({e.strerror or e}). Check the server URL.") from None
        if not targets:
            raise SettingsError(f"the server host {host} has no IPv4 or IPv6 address.")
    url = urllib.parse.urlunsplit(("http", f"127.0.0.1:{port}", split.path, split.query,
                                   split.fragment))
    return url, port, targets


# The session socket of the server

# The key a client config gets when the client reaches the server through a
# session socket, which needs no key.
SESSION_KEY = "gmlx-container-session"
_SESSIONS_PATH = "/launch/sessions"
_SESSION_TIMEOUT = 5.0


def _own_address(addr: str) -> bool:
    """Whether ``addr`` is a loopback address or an address of one of this
    Mac's interfaces, which only a local socket can bind."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return True
    if ip.is_unspecified or ip.is_multicast:
        return False
    family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.bind((addr, 0))
    except OSError:
        return False
    return True


def uses_session(base_url: str, targets: list) -> bool:
    """Whether the client reaches the server through a session socket: a
    plain http server whose every address, as ``guest_url`` resolved it, is
    this Mac's own. The spelling of the host does not matter, so
    ``127.1``, ``localhost.`` and the Mac's LAN address all count."""
    split = urllib.parse.urlsplit(base_url)
    return (split.scheme == "http" and bool(targets)
            and all(_own_address(host) for host, _ in targets))


def full_api_line(base_url: str, client: str, api_key: str | None) -> str:
    """The line launch prints when the client reaches a server it cannot
    limit to a session socket."""
    what = ("an https server" if urllib.parse.urlsplit(base_url).scheme == "https"
            else "not a server on this Mac")
    gets = ("gets the key you passed and every route the server offers" if api_key
            else "reaches every route the server offers")
    return (f"[launch] {base_url} is {what}, so launch cannot limit it to a session "
            f"socket, and {client} {gets}.")


def _sessions_url(base_url: str) -> str:
    return base_url.rstrip("/") + _SESSIONS_PATH


def _old_server(base_url: str) -> Exception:
    from gmlx.commands import launch as L

    return L.LaunchError(
        f"the server at {base_url} does not offer session sockets, which container "
        "mode needs to limit what the client can reach. When it is a gmlx server, "
        "restart it with gmlx restart so that it runs the installed version.")


# A server from before session sockets has no such route.
_NO_ROUTE = (404, 405)


def _refusal(base_url: str, e: urllib.error.HTTPError) -> Exception:
    from gmlx.commands import launch as L

    if e.code in (401, 403):
        return L.LaunchError(f"the server at {base_url} refused the API key ({e.code}). "
                             "Pass the server's key with --api-key.")
    if e.code in _NO_ROUTE:
        return _old_server(base_url)
    message = _error_message(e).rstrip(".")
    return L.LaunchError(f"the server at {base_url} could not open a session socket "
                         f"({e.code}): {message}. Its log may say more: gmlx logs")


def _error_message(e: urllib.error.HTTPError) -> str:
    """The message in an error reply, else the HTTP reason."""
    try:
        body = json.loads(e.read(64 * 1024) or b"null")
        message = body["error"]["message"]
        if isinstance(message, str) and message:
            return message[:500]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return str(e.reason)


def _unreachable(base_url: str, e: Exception) -> Exception:
    from gmlx.commands import launch as L

    return L.LaunchError(f"cannot reach the server at {base_url} "
                         f"({_why_unreachable(e)}). Check that it runs with: gmlx status")


def _request_refusal(e: urllib.error.HTTPError) -> bool:
    """Whether an error reply is the one a gmlx server gives a request body
    it refuses."""
    try:
        body = json.loads(e.read(64 * 1024) or b"null")
        return body["error"]["type"] == "invalid_request_error"
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def sessions_offered(base_url: str, api_key: str | None) -> bool:
    """Whether the server offers session sockets. The request has a body the
    endpoint refuses with a 400 invalid_request_error, so it creates no
    session, and only that answer counts. A server with no such route
    answers 404 or 405, and any other answer, a success included, comes
    from something that is not a gmlx server. A refused key or a server
    error raises, since every later request would fail the same way."""
    from gmlx.commands import launch as L

    try:
        L._http_post_json(_sessions_url(base_url), {"probe": True}, api_key=api_key,
                          timeout=_SESSION_TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return _request_refusal(e)
        if e.code in _NO_ROUTE:
            return False
        raise _refusal(base_url, e) from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise _unreachable(base_url, e) from None
    return False


class ServerSession:
    """One session socket of the gmlx server for one container session. The
    server serves the client's requests on it with a limited set of routes
    and only the assistants listed in ``assistants``. :meth:`open` and
    :meth:`close` run in the supervisor's thread, and :meth:`renew` in a
    thread of the API relay, so the session id is kept under a lock."""

    def __init__(self, base_url: str, api_key: str | None, client: str,
                 assistants: list[str], web_ports: list[int] | None = None):
        self.base_url = base_url
        self.port = urllib.parse.urlsplit(base_url).port or 80
        self.api_key = api_key
        self.client = client
        self.assistants = list(assistants)
        # The ports of the browser app's pages. The server refuses those
        # pages on its TCP port while the session is open.
        self.web_ports = list(web_ports or ())
        self.id: str | None = None
        self.socket: str | None = None
        self.allowed: dict[str, list[str]] = {}
        self.unknown: list[str] = []
        self.closed = False
        # Why the last renewal got no socket, or None. The supervisor prints
        # it after the client exits, and ``log`` gets it at once.
        self.refused: str | None = None
        self.log: Callable[[str], None] = lambda line: None
        self._lock = threading.Lock()

    def _post(self) -> dict:
        from gmlx.commands import launch as L

        body = {"client": self.client, "assistants": self.assistants}
        if self.web_ports:
            body["web_ports"] = self.web_ports
        reply = L._http_post_json(_sessions_url(self.base_url), body,
                                  api_key=self.api_key, timeout=_SESSION_TIMEOUT)
        return _session_reply(reply, self.base_url, self.port)

    def open(self) -> str:
        """Ask the server for a session socket and return its path."""
        from gmlx.commands import launch as L

        try:
            reply = self._post()
        except urllib.error.HTTPError as e:
            raise _refusal(self.base_url, e) from None
        except _BadReply as e:
            raise L.LaunchError(str(e)) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise _unreachable(self.base_url, e) from None
        with self._lock:
            self.id, self.socket = reply["id"], reply["socket"]
            self.allowed, self.unknown = reply["assistants"], reply["unknown"]
            if self.closed:
                stale, self.id = self.id, None
            else:
                stale = None
        if stale:
            self._delete(stale)
        return reply["socket"]

    def renew(self) -> str | None:
        """Ask again after the socket stopped answering, such as after a
        server restart, with the same list. Returns the new path, or None
        with the reason in :attr:`refused` and the log."""
        try:
            reply = self._post()
        except urllib.error.HTTPError as e:
            return self._renew_refused(str(_refusal(self.base_url, e)))
        except _BadReply as e:
            return self._renew_refused(str(e))
        except (urllib.error.URLError, OSError, ValueError) as e:
            return self._renew_refused(str(_unreachable(self.base_url, e)))
        with self._lock:
            if self.closed:
                old, path = reply["id"], None
            else:
                old, self.id = self.id, reply["id"]
                path = self.socket = reply["socket"]
                self.refused = None
        if old:
            self._delete_later(old)
        return path

    def _renew_refused(self, why: str) -> None:
        with self._lock:
            if self.closed:
                return None
            self.refused = why
        self.log(f"gmlx api: the server gave no new session socket, so the client's "
                 f"requests fail ({why})")
        return None

    def close(self) -> None:
        """End the session on the server. The server also removes its
        sockets when it starts and stops, so a failure here is only logged."""
        with self._lock:
            self.closed = True
            old, self.id = self.id, None
        if old:
            self._delete(old)

    def _delete_later(self, session_id: str) -> None:
        # The connections that wait for the new path need not wait for the
        # end of the old session too.
        threading.Thread(target=self._delete, args=(session_id,), daemon=True).start()

    def _delete(self, session_id: str) -> None:
        from gmlx.commands import launch as L

        url = f"{_sessions_url(self.base_url)}/{urllib.parse.quote(session_id, safe='')}"
        try:
            L._http_delete(url, api_key=self.api_key, timeout=_SESSION_TIMEOUT)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def lines(self) -> list[str]:
        """One line for each assistant the client can use, and a warning for
        each listed one the server does not have."""
        out = []
        for alias, tools in self.allowed.items():
            if tools:
                out.append(f"[launch] {self.client} can use assistant {alias}, whose tools "
                           f"run on the Mac: {', '.join(tools)}")
            else:
                out.append(f"[launch] {self.client} can use assistant {alias}, which has "
                           "no tools")
        for alias in self.unknown:
            out.append(f"[launch] warning: the server has no assistant {alias}, which "
                       f"launch.container.clients.{self.client}.assistants lists")
        return out


class _BadReply(ValueError):
    pass


def _session_reply(reply, base_url: str, port: int) -> dict:
    """The parts of a session reply launch uses, checked for shape. The
    socket must be a private socket of this user in a session folder of a
    server on ``port``, since the relay hands the client whatever socket
    this names."""
    try:
        sid, path = reply["id"], reply["socket"]
        allowed, unknown = reply.get("assistants", {}), reply.get("unknown", [])
        if not (isinstance(sid, str) and sid and isinstance(path, str)
                and os.path.isabs(path) and isinstance(allowed, dict)
                and isinstance(unknown, list)):
            raise TypeError
        tools = {}
        for alias, entry in allowed.items():
            names = entry.get("tools", [])
            if not isinstance(names, list):
                raise TypeError
            tools[str(alias)] = [str(t) for t in names]
    except (KeyError, TypeError, AttributeError):
        raise _BadReply(f"the server's session reply has an unexpected form: "
                        f"{printable(str(reply)[:200])}") from None
    from gmlx.serve.session_paths import socket_refusal

    why = socket_refusal(path, port)
    if why:
        raise _BadReply(f"the server at {base_url} named {printable(path)} as the session "
                        f"socket, which launch will not use, because {why}.")
    return {"id": sid, "socket": path, "assistants": tools,
            "unknown": [str(u) for u in unknown]}


# Step 3: the container command, the service and the guest entry

class _Prereqs:
    def __init__(self):
        self.binary = cli.find()
        self.version = None
        self.running = False
        self.entry = runtime.entry_path()
        if self.binary:
            try:
                self.version = cli.version()
                self.running = cli.system_running()
            except ContainerError:
                pass

    def report(self) -> list[str]:
        lines = []
        if not self.binary:
            lines.append(f"[launch] Apple container is not installed. {cli.INSTALL_HINT}")
        else:
            v = ".".join(map(str, self.version)) if self.version else "unknown version"
            state = "running" if self.running else "stopped"
            lines.append(f"[launch] container {v}, service {state}")
            if self.version and self.version < cli.CONTAINER_MIN:
                lines.append(f"[launch] container {v} is older than the "
                             f"{'.'.join(map(str, cli.CONTAINER_MIN))} this mode needs")
        if not self.entry.is_file():
            lines.append(f"[launch] the guest entry {self.entry} is not built. Build it "
                         f"with: {runtime.BUILD_HINT}")
        return lines

    def require_installed(self) -> None:
        """Refuse what cannot run. This check downloads nothing."""
        from gmlx.commands.launch import LaunchError

        if not self.binary:
            raise LaunchError(f"container mode needs Apple container, which is not "
                              f"installed. {cli.INSTALL_HINT}")
        if self.version is None or self.version < cli.CONTAINER_MIN:
            need = ".".join(map(str, cli.CONTAINER_MIN))
            have = ".".join(map(str, self.version)) if self.version else "an unknown version"
            raise LaunchError(f"container mode needs Apple container {need} or newer, and "
                              f"this Mac has {have}. Upgrade with: brew upgrade container")
        if not self.entry.is_file():
            raise LaunchError(f"the guest entry {self.entry} is not built. In a git "
                              f"checkout, build it with: {runtime.BUILD_HINT}")

    def start_service(self, say) -> bool:
        """Start a stopped service. Returns True when it had to start, which
        marks a first run."""
        from gmlx.commands.launch import LaunchError

        if self.running:
            return False
        if not session.stdin_is_tty():
            raise LaunchError("the container service is not running. Start it with: "
                              "container system start")
        say(f"[launch] step 1: start the container service. The first start asks to "
            f"install a Linux kernel and downloads about {cli.KERNEL_DOWNLOAD_MB} MB.")
        cli.system_start()
        self.running = True
        return True


# --shell into a running session

def _attach(a, exec_fn, say) -> int:
    from gmlx.commands.launch import LaunchError

    for dest, default in _ATTACH_REFUSED.items():
        value = getattr(a, dest, default)
        if value != default:
            flag = _flag_name(dest, value)
            raise LaunchError(f"a {a.harness} session is already running, so --shell attaches "
                              f"to it, and {flag} applies only to a new session.")
    ignored = [_flag_name(dest, getattr(a, dest, default))
               for dest, default in _ATTACH_IGNORED.items()
               if getattr(a, dest, default) != default]
    containers = [c for c in cli.list_launch_containers() if c.state == "running"
                  and c.labels.get("gmlx.launch.client") == a.harness]
    try:
        record = session.read_record(a.harness)
    except SettingsError as e:
        names = [c.name for c in containers]
        stop = (f"Stop it with: container stop {names[0]}" if len(names) == 1
                else "Quit the client in that session")
        raise LaunchError(f"{e} End that session and launch again. {stop}") from None
    name = (record or {}).get("name")
    if not record or not any(c.name == name for c in containers):
        raise LaunchError(f"the {a.harness} session is still starting. Try again in a moment.")
    if ignored:
        say(f"[launch] {_listed(ignored)} {'applies' if len(ignored) == 1 else 'apply'} only "
            "to a new session, so the shell ignores "
            f"{'it' if len(ignored) == 1 else 'them'}.")
    shares = [Mount(s["host"], s["guest"], bool(s.get("readonly"))) for s in record["shares"]]
    cwd = settings.guest_path(os.path.realpath(_cwd()), shares)
    say(f"[launch] attaching to {name} (working folder {record['workdir']})")
    if cwd is None:
        say("[launch] the current folder is not shared with this session, so the shell "
            "opens in its working folder.")
    command = [runtime.GUEST_ENTRY, *(["--clipboard"] if record.get("clipboard") else []),
               "--shell", "--", *a.passthrough]
    argv = cli.exec_argv(name, command, tty=session.stdin_is_tty(), cwd=cwd)
    return exec_fn(argv[0], argv, dict(os.environ))


# The launch order

def _client_env(client: str, plan, ready, command_cfg, web_port: int | None) -> dict:
    """Guest variables launch sets by value for one client."""
    env: dict[str, str] = {}
    if client == "claude-code":
        env.update({"IS_SANDBOX": "1", "DISABLE_AUTOUPDATER": "1"})
        if plan.network == "none":
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if web_port is not None:
        env.update({"HOST": "127.0.0.1", "PORT": str(web_port)})
    if client == "open-webui" and command_cfg == "image":
        env["WEBUI_SECRET_KEY_FILE"] = str(plan.home / ".webui_secret_key")
    if client == "omp" and plan.clipboard == "images":
        # omp looks for a display before it runs a clipboard tool.
        env["WAYLAND_DISPLAY"] = "wayland-0"
    return env


def _split_env(entries: list[str]) -> tuple[list[str], dict[str, str]]:
    names, values = [], {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        names.append(name)
        if sep:
            values[name] = value
    return names, values


def _dsh_profile_is_web(a) -> bool:
    """Whether a dsh profile runs the web app in a container, by its name
    only. The profile's manifest lies in the private home, which the guest
    writes, so it never decides whether the Mac opens a port and a browser."""
    from gmlx.commands import launch as L

    profile = L._DSH_PROFILE if a.dsh_profile is None else a.dsh_profile
    return profile in (L._DSH_PROFILE, L._DSH_TEMPLATE)


def _image_state(image_plan, rebuild: bool, running: bool
                 ) -> tuple[str, str, images.ReadyImage | None]:
    """For the dry run: the reference the command would name, the lines
    about the image, and the image when it is in the store. Nothing is
    built or pulled."""
    kind, client = image_plan.kind, image_plan.client
    if not running:
        tag = (image_plan.ref if kind == "image"
               else images.shipped_tag(client, image_plan.packages)
               if kind == "shipped" else images.build_repo(client))
        return tag, f"[launch] image {tag}: its state is unknown until the service runs", None
    lines = []
    bases_ready = True
    digests: dict[str, str] = {}
    for base in sorted(image_plan.bases) if kind == "build" else []:
        info = cli.image_info(images.shipped_tag(base, image_plan.base_packages.get(base, [])))
        if info is not None:
            digests[base] = info.digest
        bases_ready = bases_ready and info is not None
        base_state = ("would be rebuilt" if rebuild else "present" if info
                      else "would be built first")
        lines.append(f"[launch] base {images.base_ref(base)}: {base_state}")
    if kind == "image":
        tag = image_plan.ref
        verb = "pulled again"
    elif kind == "shipped":
        tag = images.shipped_tag(client, image_plan.packages)
        verb = "rebuilt"
    elif bases_ready and not rebuild:
        # The tag hashes the build context and the base digests, which
        # reading only can work out.
        tag = f"{images.build_repo(client)}:{images.build_hash(image_plan, digests, _say)}"
        verb = "rebuilt"
    else:
        tag = images.build_repo(client)
        lines.insert(0, f"[launch] image {tag}: would be "
                        f"{'rebuilt' if rebuild else 'built after its bases'}")
        return tag, "\n".join(lines), None
    info = cli.image_info(tag)
    if rebuild:
        state = f"would be {verb}"
    else:
        state = "present" if info else ("would be pulled" if kind == "image"
                                        else "would be built")
    lines.insert(0, f"[launch] image {tag}: {state}")
    if info is None:
        return tag, "\n".join(lines), None
    ref = f"{images.repository_of(info.name or tag)}@{info.digest}"
    return ref, "\n".join(lines), images.ReadyImage(kind, tag, info, ref, "found", client)


def run_container(a, launch_cfg: LaunchCfg, *, exec_fn) -> int:
    from gmlx.commands import launch as L

    say = _say
    client = a.harness
    cfg = launch_cfg.container.for_client(client)
    dry = bool(a.config_only)
    a.container_mode = True
    try:
        if a.config_path:
            raise L.LaunchError("--config-path does not apply in container mode, where the "
                                "client's configuration goes in its private home.")
        # Step 3. A stopped service starts only after the refusals of step 6.
        prereqs = _Prereqs()
        # Step 4
        lock = session.try_session_lock(client)
        if lock is None:
            if a.shell:                   # attaching refuses --config-only itself
                return _attach(a, exec_fn, say)
            try:
                record = session.read_record(client) or {}
            except SettingsError:
                record = {}
            where = f" in {record['name']}" if record.get("name") else ""
            raise L.LaunchError(
                f"a {client} session is already running{where}, and one session of a "
                "client runs at a time. End that session to launch another, or open a "
                f"shell in it with: gmlx launch {client} --shell")
        held = [lock]
        try:
            if dry:
                say("[launch] container dry run: no image is built or pulled, and no "
                    "container is started.")
                for line in prereqs.report():
                    say(line)
            else:
                prereqs.require_installed()
            return _run_locked(a, launch_cfg, cfg, prereqs, held, exec_fn, say)
        finally:
            for item in reversed(held):
                item.release()
    except (L.LaunchError, SettingsError, ContainerError, ConfigError,
            confine.ConfinedError) as e:
        sys.stdout.flush()
        print(printable_lines(f"[launch] {e}"), file=sys.stderr)
        return 1
    except OSError as e:
        # Such as a launch data folder that is a file, or a full disk.
        why = f"cannot use {e.filename} ({e.strerror})." if e.filename and e.strerror else e
        sys.stdout.flush()
        print(printable_lines(f"[launch] {why}"), file=sys.stderr)
        return 1
    except _Signalled as e:
        print(f"[launch] stopped by signal {e.signum} while the image was prepared",
              file=sys.stderr)
        return 128 + e.signum


def _run_locked(a, launch_cfg, cfg, prereqs, held, exec_fn, say) -> int:
    from gmlx.commands import launch as L

    client = a.harness
    dry = bool(a.config_only)
    # Step 5
    session.remove_record(client)
    # Step 6
    # The shares and the image settings are checked here, before the image
    # steps, so a mistake in them never waits behind a download or a build.
    # A busy web port (when the supervisor binds it), a problem in the image
    # itself and a link in the private home (step 11) stop the launch only
    # after the image steps.
    if client == "dsh" and a.dsh_profile in L._DSH_STDIO:
        raise L.LaunchError(f"the {a.dsh_profile} profile serves another program over stdio, "
                            "which a container session cannot hand over. Use it on the Mac "
                            "with --no-container.")
    host, port = _server_endpoint(a)
    _, api_port, _ = guest_url(a.base_url or f"http://{host}:{port}/v1")
    web = client == "open-webui" or (client == "dsh" and _dsh_profile_is_web(a))
    web_port = L.web_port_for(client, port) if web else None
    # A read-write share of any client's build: folder would let this
    # client change what that image runs.
    builds = {c: launch_cfg.container.for_client(c).build for c in LAUNCH_CLIENTS}
    plan = settings.resolve_plan(client, cfg, cwd=_cwd(), mount_cwd=a.mount_cwd,
                                 cli_mounts=a.mount, network=a.network, api_port=api_port,
                                 web_port=web_port,
                                 build_folders={c: b for c, b in builds.items() if b})
    if api_port is None and plan.network == "none":
        raise L.LaunchError(f"network: none cannot reach {a.base_url}, which is not a local "
                            "http server. Use the default network for this server.")
    # A build: folder the client can write would run its code at the next build.
    writable = [m.source for m in plan.mounts if not m.readonly and m.kind != "volume"]
    image_plan = images.resolve_image(client, cfg, launch_cfg.container,
                                      image_override=a.image, writable=writable)
    config_notes: list[str] = []
    config_path = settings.server_config_path(host, port,
                                              autostart=not (a.base_url or a.no_start),
                                              notes=config_notes)
    for line in [*plan.warnings, *plan.notes, *image_plan.notices, *config_notes,
                 *settings.server_config_warnings(config_path, plan.shares),
                 *settings.pythonpath_warnings(plan.shares)]:
        say(line)
    for line in settings.seed_home(plan.home, plan.seed, reseed=getattr(a, "reseed", False),
                                   writable=settings.seed_writable(plan, _cwd())):
        say(line)
    if getattr(a, "reseed", False) and not plan.seed:
        say(f"[launch] --reseed has nothing to copy, because no seed is configured for "
            f"{client}.")
    if not dry:
        settings.record_shares(plan)
    check = _server_precheck(a, dry)
    if check.rc is not None:
        return check.rc
    # The service start and its kernel download come after every refusal.
    first_run = False if dry else prereqs.start_service(say)
    running = prereqs.running
    ready = None
    steps = int(first_run)
    runtime_dir = runtime.runtime_root() / (
        runtime.entry_digest() if prereqs.entry.is_file() else "<sha256>")
    if check.missing:
        # A dry run with no server: the handler needs the server's models,
        # so the plan is all it can show.
        _, image_line, _ = _image_state(image_plan, a.rebuild, running)
        _print_dry_plan(runtime_dir, plan, image_line,
                        _summary_lines(plan, None, a.shell, client, plan.workdir), running,
                        say)
        say(check.missing)
        return 0
    if not dry:
        # Step 7
        session.cleanup_stale(client, keep_runtime=runtime_dir.name, say=say)
        containers = cli.containers()
        for line in session.orphan_notices(client, containers):
            say(line)
        pending = images.pending_work(image_plan, a.rebuild)
        # A build is about to use the builder, so an owed stop waits for it.
        # A pull does not use the builder, and nothing stops it after a pull.
        notice = images.builder_notice(say=say, settle=pending != "build")
        if notice:
            say(notice)
        # Step 8. A closed terminal tab during a long first build must still
        # run the build's clean-up, which records the builder's owed stop.
        with _signals_raise():
            runtime_dir, runtime_lock = runtime.acquire_runtime()
            held.append(runtime_lock)
            held.extend(session.lock_volumes(plan.volumes))
            session.check_volumes_free(plan.volumes, containers)
            session.ensure_volumes(plan.volumes, say)
            # The steps are numbered when this launch starts the service or
            # builds or pulls an image, and only the steps that run get a
            # number.
            if pending:
                steps += 1
            ready = images.ensure_image(
                image_plan, rebuild=a.rebuild, say=say,
                step=f"step {steps}" if steps > int(first_run) else None)
            try:
                word = (cfg.command[0] if isinstance(cfg.command, list)
                        else images.image_command(ready, "image", [], a.passthrough)[0][0]
                        if cfg.command == "image" else images.CLIENT_BINARY[client])
            except images.ImageError as e:
                if not a.shell:            # a shell is how you look into such an image
                    raise
                say(f"[launch] warning: {e}")
            else:
                images.check_command(ready, word, str(runtime_dir), shell=a.shell,
                                     say=say)
    # Step 9
    rc = L._ensure_server(a)
    if rc is not None:
        return rc
    base = a.base_url or f"http://{a.host}:{a.port}/v1"
    a.guest_base_url, api_port, api_targets = guest_url(base)
    if api_port is None and plan.network == "none":
        raise L.LaunchError(f"network: none cannot reach {base}, which is not a local http "
                            "server. Use the default network for this server.")
    if int(a.port) != port:
        # The server check found the server on another port than step 6
        # assumed, so the ports that depend on it are worked out again.
        web_port = L.web_port_for(client, int(a.port)) if web else None
        plan.forward = settings.forward_ports(plan.forward, api_port=api_port,
                                              web_port=web_port)
    server_session, session_line = None, None
    if api_port is not None and uses_session(base, api_targets):
        offered = (check.offered if check.base == base and check.offered is not None
                   else sessions_offered(base, a.api_key))
        if not offered and not dry:
            raise _old_server(base)
        if dry:
            session_line = (f"[launch] the server at {base} offers session sockets"
                            if offered else f"[launch] warning: {_old_server(base)}")
        # The probe and the session request use the server's key, and the
        # client only the socket.
        a.client_api_key = SESSION_KEY
        server_session = ServerSession(base, a.api_key, client, cfg.assistants,
                                       [web_port] if web_port is not None else [])
    full_api = None if server_session else full_api_line(base, client, a.api_key)
    # Step 10
    if a.model and not a.no_keep and not dry:
        L._pick_default(L.probe_models(base, a.api_key, client), a.model)
        L._keep_model(a)
    # Step 11
    captured: dict = {}

    def sink(argv, pairs, extra):
        captured.update(argv=argv, pairs=pairs, extra=extra)
        return 0
    a.container_sink = sink
    # Under --shell the client does not start, so its summary and notes
    # would describe a program that is not running.
    quiet = contextlib.redirect_stdout(io.StringIO()) if a.shell else contextlib.nullcontext()
    with guest_home(plan.home), quiet:
        rc = L._HARNESSES[client](a, exec_fn=exec_fn)
    sys.stdout.flush()
    if rc != 0 or not captured:
        return rc
    if client == "open-webui":
        # The official image does not create its data folder, and SQLite
        # cannot open a database in a folder that does not exist.
        with confine.confined(plan.home):
            confine.mkdirs(Path(captured["pairs"]["DATA_DIR"]))
    # Step 12
    passthrough = captured["extra"]
    image_line = ""
    if dry:
        image_ref, image_line, ready = _image_state(image_plan, a.rebuild, prereqs.running)
    if a.shell:
        command, image_workdir = list(passthrough), None
    elif ready is not None:
        command, image_workdir = images.image_command(ready, cfg.command, captured["argv"],
                                                      passthrough)
    elif cfg.command == "image":            # the dry run, with the image not in the store
        # The arguments after -- replace CMD, so only ENTRYPOINT stays.
        command = (["<ENTRYPOINT of the image>", *passthrough] if passthrough
                   else ["<ENTRYPOINT and CMD of the image>"])
        image_workdir = None
    else:
        command = ([*cfg.command, *passthrough] if isinstance(cfg.command, list)
                   else [*captured["argv"], *passthrough])
        image_workdir = None
    names, values = _split_env(plan.env)
    env_values = {**settings.guest_env(plan.home),
                  **_client_env(client, plan, ready, cfg.command, web_port)}
    pair_names = [n for n in captured["pairs"] if n not in env_values]
    env_names = list(dict.fromkeys([*pair_names, *(n for n in names if n not in env_values)]))
    child_env = {**captured["pairs"], **values}
    # dsh prints its URL with a per-process login token, which the Mac
    # browser needs, so launch reads it from the client's output.
    token_url = client == "dsh" and web_port is not None and not a.shell
    if dry:
        sess = session.Session(client, "xxxxxx", Path("<session folder>"))
    else:
        sess = session.new_session(client, plan.forward)
        image_ref = ready.run_ref
    spec = session.RunSpec(
        session=sess, plan=plan, image_ref=image_ref, runtime_dir=runtime_dir,
        command=command, workdir=image_workdir or plan.workdir, env_values=env_values,
        env_names=env_names, child_env=child_env, api_port=api_port, web_port=web_port,
        tty=session.stdin_is_tty() and not token_url, interactive=not token_url,
        shell=a.shell, url_pattern=_DSH_URL_LINE if token_url else None,
        labels={"gmlx.launch.runtime": runtime_dir.name})
    summary = _summary_lines(plan, None if dry else ready, a.shell, client, spec.workdir)
    if full_api:
        summary.append(full_api)
    if dry:
        if session_line:
            summary.append(session_line)
        return _print_dry_run(spec, plan, image_line, summary, cfg, captured, running, say)
    record = {"name": sess.name, "workdir": spec.workdir, "clipboard": plan.clipboard == "images",
              "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                         for m in plan.shares]}
    if steps:
        summary.insert(0, f"[launch] step {steps + 1}: start {client}")
    # Under --shell the app is not running yet, so there is nothing to open.
    opener = webbrowser.open if (web_port and plan.open_browser and not a.shell) else None
    return session.supervise(spec, api_targets=api_targets, record=record, say=say,
                             opener=opener, summary=summary, server_session=server_session)


def _summary_lines(plan, ready, shell: bool, client: str, workdir: str) -> list[str]:
    lines = [images.describe(ready)] if ready is not None else []
    if ready is not None:
        note = images.image_age_note(ready)
        if note:
            lines.append(note)
    holder = None
    if plan.cwd_shared:
        guest = [m for m in plan.shares if m.target == plan.workdir
                 or plan.workdir.startswith(m.target.rstrip("/") + "/")]
        holder = max(guest, key=lambda m: len(m.target), default=None)
    for m in plan.shares:
        mode = "read-only" if m.readonly else "read-write"
        where = "" if m.target == m.source else f" at {m.target}"
        extra = ", working folder" if m is holder else ""
        if m.kind == "git":
            extra = f", {m.note}"
        lines.append(f"[launch] sharing {settings._tilde(m.source)}{where} ({mode}{extra})")
    if ready is not None:
        lines += session.volume_lines(plan.volumes)
    for port in plan.forward:
        lines.append(f"[launch] forwarding the guest's 127.0.0.1:{port} to Mac port {port}")
    if not plan.cwd_shared and client not in settings.NO_CWD_CLIENTS:
        where = "its private home" if workdir == str(plan.home) else workdir
        who = "the shell opens" if shell else f"{client} starts"
        lines.append(f"[launch] the current folder is not shared, so {who} in {where}")
    if shell:
        lines.append(f"[launch] opening a shell instead of {client}")
    return lines


def _print_dry_plan(runtime_dir, plan, image_line, summary, running, say) -> None:
    for line in image_line.split("\n"):
        say(line)
    say(f"[launch] runtime folder {runtime_dir}")
    if plan.volumes:
        existing = {v.name for v in cli.volume_list()} if running else None
        for v in plan.volumes:
            if existing is None:
                state = "unknown until the service runs"
            else:
                state = "exists" if v.source in existing else "would be created"
            say(f"[launch] volume {v.source}: {state}")
    for line in summary:
        say(line)


def _print_dry_run(spec, plan, image_line, summary, cfg, captured, running, say) -> int:
    _print_dry_plan(spec.runtime_dir, plan, image_line, summary, running, say)
    if isinstance(cfg.command, list) or cfg.command == "image":
        say("[launch] the command: setting replaces the client's own command, "
            f"{shlex.join(captured['argv'])}")
    argv = session.compose_run_argv(spec)
    say("[launch] the command, which needs the API socket that only a running launch "
        "provides:")
    print(printable(shlex.join(argv)))
    return 0
