"""``gmlx launch <client> --container``: the launch order in container mode.

The client's handler in ``launch.py`` still resolves the server, writes the
client's configuration and names the command, but it does so with ``HOME``
pointed at the private home of the client's project, and its finish helper
hands the command back here instead of replacing the process. This module
decides container mode and the project, takes the session lock or joins the
session that holds it, resolves the shares and the image, and runs the
supervisor in ``gmlx.container.session``.
"""

from __future__ import annotations

import contextlib
import io
import ipaddress
import json
import os
import secrets
import select
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Callable, Collection

from gmlx.config import (AGENT_DEPS_TARGET, ConfigError, LaunchCfg, agent_deps_volume,
                         agent_name, config_key, launch_block_enables, load_launch_settings,
                         normal_guest_target, parse_volume_spec, target_label)
from gmlx.container import cli, confine, images, notices, runtime, session, settings, web_ports
from gmlx.container.cli import ContainerError
from gmlx.container.settings import Mount, SettingsError
from gmlx.container.text import printable, printable_lines

# Flags that only mean something in container mode, by argparse dest.
CONTAINER_FLAGS = {"mount": "--mount", "mount_cwd": "--mount-cwd", "image": "--image",
                   "rebuild": "--rebuild", "reseed": "--reseed", "network": "--network",
                   "shell": "--shell", "remove_home": "--remove-home", "detach": "--detach",
                   "stop": "--stop"}
# Flags a launch that joins a running session ignores, since that session
# already has its server and model, and the flags it refuses, which shape a
# new session. --mount-cwd counts as ignored only when the session does not
# share the current folder. --no-mount-cwd chose the project the launch
# joins, so it never counts. --mount counts only when it names a share that
# the session does not have. A dsh profile must match the running one.
_JOIN_IGNORED = {
    "model": None, "base_url": None, "host": None, "port": None, "api_key": None,
    "no_start": False, "start_timeout": 0.0, "no_keep": False, "mount_cwd": None,
}
_JOIN_REFUSED = {
    "provider_id": "gmlx", "config_path": None, "config_only": False,
    "mount": [], "image": None, "rebuild": False, "reseed": False, "network": None,
}
# Printable ASCII up to the end of the line, so a URL with a terminal
# control in it opens nothing.
_DSH_URL_LINE = r"dsh web: ([\x21-\x7e]+)(?=\s)"
# The dsh flag that makes a profile from a template. dsh refuses it for a
# profile that exists.
_DSH_FROM_DEFAULT = "--from-default-profile"
# The variable that gives a launch that --detach started the pipe on which
# it reports the start of its session to the launch that started it.
DETACH_FD_ENV = "GMLX_LAUNCH_DETACH_FD"
# The variable that names the descriptor of the session lock that the
# launch with --detach hands to the launch it starts.
DETACH_LOCK_ENV = "GMLX_LAUNCH_DETACH_LOCK_FD"
# How long --detach waits for the container of an agent with no web app to
# run, and after the start for a web app to answer, and --stop for a
# session to end.
DETACH_RUN_WAIT = 120.0
DETACH_ANSWER_WAIT = session.OPEN_TIMEOUT + 30.0
STOP_WAIT = 60.0
# --list and gmlx status wait this long for the container service, as
# doctor does, so a service that does not answer costs seconds.
LIST_QUERY_TIMEOUT = 5.0


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


def _parses(path) -> bool:
    """Whether the config file at ``path`` reads as a YAML mapping, or as
    nothing."""
    import yaml
    try:
        with open(path) as f:
            return isinstance(yaml.safe_load(f), (dict, type(None)))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError, ValueError):
        return False


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
                named = str(path) in first
                if not _parses(path):
                    # A file that does not parse may have no launch block at all.
                    raise ConfigError(
                        f"{first}{sep}gmlx launch cannot read "
                        f"{'that file' if named else path}, so it cannot tell whether "
                        f"container mode is on for {a.harness}. Fix the file, or pass "
                        f"--no-container to run {a.harness} on the Mac.") from None
                subject = "That file" if named else str(path)
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
    variables that would send it to the user's own files. gmlx's own cache
    and data folders stay on the Mac."""
    hidden = ("DSH_HOME", "HERMES_HOME")
    pinned = {"XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME") or "~/.cache",
              "XDG_DATA_HOME": os.environ.get("XDG_DATA_HOME") or "~/.local/share"}
    saved = {k: os.environ.get(k) for k in ("HOME", *pinned, *hidden)}
    # gmlx's runfiles and launch records follow these folders, which default
    # to folders in HOME. The guest can plant links in the private home, so
    # they keep their Mac values while HOME moves.
    for key, value in pinned.items():
        os.environ[key] = os.path.expanduser(value)
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
    no ``except Exception`` on the way can stop the launch from exiting.
    ``ends_cleanup`` is true for a signal after the ignored one, which also
    ends the clean-ups that are still to start, such as the stop of the
    image builder."""

    def __init__(self, signum: int, ends_cleanup: bool = False):
        super().__init__(signum)
        self.signum = signum
        self.ends_cleanup = ends_cleanup


class _Interrupted(KeyboardInterrupt):
    """A Ctrl-C during step 8, with ``ends_cleanup`` as for
    :class:`_Signalled`."""

    def __init__(self, ends_cleanup: bool = False):
        super().__init__()
        self.ends_cleanup = ends_cleanup


@contextlib.contextmanager
def _signals_raise():
    """Turn SIGTERM and SIGHUP into an exception while the block runs, so
    every ``finally`` in it runs. The exit code is 128 plus the signal.
    Ctrl-C raises KeyboardInterrupt, as it does outside the block.

    The first signal raises. A closed window sends launch a second SIGHUP
    while those clean-ups run, and a user can press Ctrl-C two times. Thus
    the second signal of any of the three is ignored, and the clean-ups,
    such as the stop of the image builder, can finish. The third signal and
    each signal after it raise again with ``ends_cleanup`` set, so a
    clean-up that waits for a container service that does not answer stops,
    and the builder clean-up does not start. A signal that was ignored when
    launch started, as nohup ignores SIGHUP, stays ignored."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield                          # only the main thread can set handlers
        return
    count = 0

    def raise_it(signum, _frame):
        nonlocal count
        count += 1
        if count == 2:
            # A closed window has no reader for SIGHUP. The line goes straight
            # to the descriptor, since a handler that enters a buffered
            # stream again raises RuntimeError in place of the interrupt.
            if signum == signal.SIGINT:
                with contextlib.suppress(OSError):
                    os.write(2, b"\n[launch] launch stops when its clean-up ends. "
                                b"Press Ctrl-C again to stop at once.\n")
            return
        if signum == signal.SIGINT:
            raise _Interrupted(count > 2)
        raise _Signalled(signum, count > 2)
    saved = {sig: signal.signal(sig, raise_it)
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
             if signal.getsignal(sig) != signal.SIG_IGN}
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
    answers. ``offered`` is the answer of the session probe of ``base``.
    ``start`` means that no server answers and launch starts one from the
    config, which a launch that is not a dry run does before its first
    write."""

    def __init__(self, rc: int | None = None, missing: str | None = None,
                 base: str | None = None, offered: bool | None = None,
                 start: bool = False):
        self.rc, self.missing, self.base, self.offered = rc, missing, base, offered
        self.start = start


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
                            "or start that server.", L.EXIT_UNAVAILABLE)
    if a.host or a.port:
        host, port = a.host or L._DEFAULT_HOST, int(a.port or L._DEFAULT_PORT)
    else:
        host, port = lifecycle.auto_target(None, None)
    base = L._base_url(host, port)
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
    return _ServerCheck(start=not dry)


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
    L.check_model_choice(a.harness, L.probe_models(base, key, a.harness), L.requested_model(a),
                         origin=L.model_origin(a))
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


def loopback_host(host: str) -> bool:
    """Whether ``host`` is a loopback address, or the name localhost."""
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def open_bind(base_url: str, targets: list) -> bool:
    """Whether the server of ``base_url`` listens on more than a loopback
    address, such as on every address or on the Mac's LAN address. The
    container then reaches every route of the server at the Mac's address
    on the container network, past the session socket. ``targets`` are the
    addresses that :func:`guest_url` found for a host name. A server bound
    to every address also answers at 127.0.0.1, so the runfiles of the
    servers on the URL's port count too."""
    split = urllib.parse.urlsplit(base_url)
    host = split.hostname or ""
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        named = any(not loopback_host(addr) for addr, _ in targets)
    else:
        named = not loopback_host(host)
    try:
        port = split.port or 80
    except ValueError:
        return named
    return named or _open_server_on(port)


def _open_server_on(port: int) -> bool:
    """Whether a running server that gmlx started on ``port`` listens on
    more than a loopback address, as its runfile records the bind."""
    import gmlx.serve.lifecycle as lifecycle

    for run in lifecycle.classify_runs()[0]:
        try:
            same = int(run.get("port") or 0) == port
        except (TypeError, ValueError):
            continue
        if same and not loopback_host(str(run.get("host") or "127.0.0.1")):
            return True
    return False


def open_bind_line(base_url: str) -> str:
    """The warning for a server that :func:`open_bind` finds open and that
    asks for no key."""
    return (f"[launch] warning: the server at {base_url} listens on more than the loopback "
            "address and needs no key. The container can reach every route of the server "
            "at the Mac's address on the container network. Set server.api_key in the "
            "server's config and restart the server.")


def full_api_line(base_url: str, client: str, api_key: str | None) -> str:
    """The line launch prints when the client reaches a server it cannot
    limit to a session socket."""
    what = ("an https server" if urllib.parse.urlsplit(base_url).scheme == "https"
            else "not a server on this Mac")
    gets = ("gets the key you passed and every route the server offers" if api_key
            else "reaches every route the server offers")
    return (f"[launch] {base_url} is {what}, so launch cannot limit it to a session "
            f"socket, and {target_label(client)} {gets}.")


def _sessions_url(base_url: str) -> str:
    return base_url.rstrip("/") + _SESSIONS_PATH


def _old_server(base_url: str) -> Exception:
    from gmlx.commands import launch as L

    return L.LaunchError(
        f"the server at {base_url} does not offer session sockets, which container "
        "mode needs. Restart a gmlx server with gmlx restart. For another server, run "
        "the client on the Mac with --no-container.", L.EXIT_UNAVAILABLE)


# A server from before session sockets has no such route.
_NO_ROUTE = (404, 405)


def _refusal(base_url: str, e: urllib.error.HTTPError) -> Exception:
    from gmlx.commands import launch as L

    if e.code in (401, 403):
        return L.LaunchError(f"the server at {base_url} refused the API key ({e.code}). "
                             "Pass the server's key with --api-key.")
    if e.code in _NO_ROUTE:
        return _old_server(base_url)
    kind, message = _error_reply(e)
    message = message.rstrip(".")
    if e.code == 503 and kind == "server_overloaded":
        # The server holds its most sessions, and the message says what to do.
        return L.LaunchError(f"the server at {base_url} could not open a session "
                             f"({e.code}): {message}.", L.EXIT_TEMPFAIL)
    return L.LaunchError(f"the server at {base_url} could not open a session socket "
                         f"({e.code}): {message}. Its log may say more: gmlx logs")


def _error_reply(e: urllib.error.HTTPError) -> tuple[str | None, str]:
    """The type and the message of an error reply. The message is the HTTP
    reason when the reply gives none."""
    kind = message = None
    try:
        error = json.loads(e.read(64 * 1024) or b"null")["error"]
        kind, message = error.get("type"), error.get("message")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    if not isinstance(message, str) or not message:
        message = str(e.reason)
    return (kind if isinstance(kind, str) else None), message[:500]


def _unreachable(base_url: str, e: Exception) -> Exception:
    from gmlx.commands import launch as L

    return L.LaunchError(f"cannot reach the server at {base_url} "
                         f"({_why_unreachable(e)}). Check that it runs with: gmlx status",
                         L.EXIT_UNAVAILABLE)


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
                 assistants: list[str], web_ports: list[int] | None = None,
                 project: str | None = None):
        self.base_url = base_url
        self.port = urllib.parse.urlsplit(base_url).port or 80
        self.api_key = api_key
        self.client = client
        self.assistants = list(assistants)
        # The ports of the browser app's pages. The server refuses those
        # pages on its TCP port while the session is open, and for a grace
        # after it ends.
        self.web_ports = list(web_ports or ())
        # The server keeps the prompts of a client and project apart.
        self.project = project
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

    def _post(self, replaces: str | None = None) -> dict:
        from gmlx.commands import launch as L

        body = {"client": self.client, "assistants": self.assistants}
        if self.web_ports:
            body["web_ports"] = self.web_ports
        if self.project is not None:
            body["project"] = self.project
        if replaces is not None:
            body["replaces"] = replaces
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
        server restart, with the same list. The request names the old
        session, so a server that closed it to make room closes no other
        session for this one. Returns the new path, or None with the reason
        in :attr:`refused` and the log."""
        with self._lock:
            replaces = self.id
        try:
            reply = self._post(replaces)
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
        label = target_label(self.client)
        for alias, tools in self.allowed.items():
            if tools:
                out.append(f"[launch] {label} can use assistant {alias}, whose tools "
                           f"run on the Mac: {', '.join(tools)}")
            else:
                out.append(f"[launch] {label} can use assistant {alias}, which has "
                           "no tools")
        for alias in self.unknown:
            out.append(f"[launch] warning: the server has no assistant {alias}, which "
                       f"{config_key(self.client, 'assistants')} lists")
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

def _checked_program() -> str | None:
    """The container program, found once so that every later container
    call of this command runs the same file. A program that a client could
    have replaced is refused, also for a command that starts no session."""
    binary = cli.pin()
    settings.check_program(binary)
    return binary


class _Prereqs:
    def __init__(self):
        self.binary = _checked_program()
        self.version = None
        self.running = False
        # The folder where the running service keeps its data, when it names
        # one. None means the folder that a start by launch uses.
        self.app_root: Path | None = None
        self.entry = runtime.entry_path()
        if self.binary:
            try:
                self.version = cli.version()
                found = cli.service()
                self.running, self.app_root = found.running, found.app_root
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
            # A launch refuses these two cases in require_installed.
            need = ".".join(map(str, cli.CONTAINER_MIN))
            shown = settings._tilde(self.binary)
            if self.version is None:
                lines.append(f"[launch] {shown}, the first container command on PATH, "
                             f"gives no version number, and this mode needs {need} or "
                             f"newer. {cli.upgrade_steps(self.binary)[0]}")
            elif self.version < cli.CONTAINER_MIN:
                lines.append(f"[launch] container {v} at {shown} is older than the "
                             f"{need} this mode needs. {cli.upgrade_steps(self.binary)[0]}")
        if not self.entry.is_file():
            lines.append(f"[launch] the guest entry {settings._tilde(str(self.entry))} is not "
                         f"built. Build it with: {runtime.BUILD_HINT}")
        return lines

    def require_installed(self) -> None:
        """Refuse what cannot run. This check downloads nothing."""
        from gmlx.commands.launch import EXIT_UNAVAILABLE, LaunchError

        if not self.binary:
            raise LaunchError(f"container mode needs Apple container, which is not "
                              f"installed. {cli.INSTALL_HINT}", EXIT_UNAVAILABLE)
        if self.version is None or self.version < cli.CONTAINER_MIN:
            # Another container program, such as an older install in
            # /usr/local/bin, can come before Homebrew's on PATH.
            need = ".".join(map(str, cli.CONTAINER_MIN))
            have = (f"is version {'.'.join(map(str, self.version))}" if self.version
                    else "gives no version number")
            raise LaunchError(f"container mode needs Apple container {need} or newer, and "
                              f"{settings._tilde(self.binary)}, the first container command "
                              f"on PATH, {have}. {cli.upgrade_steps(self.binary)[0]}",
                              EXIT_UNAVAILABLE)
        if not self.entry.is_file():
            raise LaunchError(f"the guest entry {settings._tilde(str(self.entry))} is not "
                              f"built. In a git checkout, build it with: "
                              f"{runtime.BUILD_HINT}", EXIT_UNAVAILABLE)

    def start_service(self, say, step: str) -> bool:
        """Start a stopped service. Returns True for its first start, which
        asks whether to install the Linux kernel and marks a first run, and
        prints ``step`` on its line. A later start, such as after a Mac
        restart, asks nothing, so it runs without a terminal too."""
        from gmlx.commands.launch import EXIT_UNAVAILABLE, LaunchError

        if self.running:
            return False
        if cli.kernel_installed():
            say("[launch] starting the container service")
            cli.system_start(install_kernel=False)
            self.running = True
            return False
        if not session.stdin_is_tty():
            raise LaunchError("the container service is not running, and its first start "
                              "asks whether to install a Linux kernel. Run it once in a "
                              "terminal with: container system start", EXIT_UNAVAILABLE)
        say(f"[launch] {step}: starting the container service. Its first start asks to "
            f"install a Linux kernel, which downloads about {cli.KERNEL_DOWNLOAD_MB} MB once.")
        try:
            cli.system_start()
        except ContainerError as e:
            # The kernel command needs a service that answers, and a start
            # can fail before the service answers.
            try:
                found = cli.service()
            except ContainerError:
                found = cli.Service(False)
            if not found.running:
                raise LaunchError(f"{e} The container service does not answer. Read its "
                                  "log with: container system logs", EXIT_UNAVAILABLE) from None
            if cli.kernel_installed(found.app_root):
                raise
            raise LaunchError(f"{e} {cli.NO_KERNEL}", EXIT_UNAVAILABLE) from None
        self.running = True
        return True

    def require_kernel(self) -> None:
        """Refuse a running service with no Linux kernel, as
        :data:`cli.NO_KERNEL` explains."""
        from gmlx.commands.launch import EXIT_UNAVAILABLE, LaunchError

        if self.running and not cli.kernel_installed(self.app_root):
            raise LaunchError(cli.NO_KERNEL, EXIT_UNAVAILABLE)


# The project a launch keys, and joining its running session

def _is_web(a) -> bool:
    """Whether the launch runs a web app, which the Mac reaches on a port of
    the project's own: Open WebUI, a dsh web profile, or an agent with
    web_port."""
    if agent_name(a.harness) is not None:
        return a.agent_cfg.web_port is not None
    return a.harness == "open-webui" or (a.harness == "dsh" and _dsh_profile_is_web(a))


def _guest_web_port(a, mac_port: int | None, api_port: int | None) -> int | None:
    """The port of this launch's web app in the guest, or None. A client's
    app listens on its Mac port. An agent's listens on the port that its
    web_port names, which is refused when the guest reaches the gmlx server
    on it, since the user chose it."""
    from gmlx.commands import launch as L

    if mac_port is None or agent_name(a.harness) is None:
        return mac_port
    port = a.agent_cfg.web_port
    if port == api_port:
        raise L.LaunchError(f"{config_key(a.harness, 'web_port')} is {port}, the gmlx server's "
                            f"port. Choose another port for {target_label(a.harness)}'s web app.")
    return port


def _session_key(a, cfg) -> tuple[str, str | None]:
    """The project id this launch keys, and the folder it keys, or None.
    A session that shares the current folder keys its real path, and one
    that shares it through a --mount or mounts: entry keys the folder of
    that share. A session that shares only other folders, or none, keys
    the default id. Open WebUI keys the default id whatever it shares,
    since it keeps one data store."""
    if a.harness == "open-webui":
        return settings.PROJECT_DEFAULT, None
    if settings.shares_cwd(a.harness, a.mount_cwd, cfg):
        folder = settings.canonical(_cwd())
        settings.check_cwd_share(folder)
        return settings.project_id(folder), folder
    folder = _cwd_share(cfg, a.mount)
    return settings.project_id(folder), folder


def _cwd_share(cfg, mounts: list[str]) -> str | None:
    """The real path of the explicit share that holds the current folder,
    the longest when several do, or None. resolve_plan names the mistake
    in a share that is not in the correct form."""
    cwd = settings.canonical(_cwd())
    best = None
    for spec in [*cfg.mounts, *mounts]:
        try:
            source, _, _ = settings.parse_mount_spec(spec)
        except SettingsError:
            continue
        real = settings.canonical(source)
        if settings._inside(cwd, real) and (best is None or len(real) > len(best)):
            best = real
    return best


def _join_folder(a, folder: str | None) -> str | None:
    """The folder whose running session can take this launch: the folder
    that the launch keys, or the current folder when the launch shares no
    folder only because launch.container.mount_cwd or the client's default
    leaves it out. Such a launch then joins the session that a launch with
    ``--mount .`` started there. ``--no-mount-cwd`` and a ``--mount`` entry
    ask for a session of the default project, and Open WebUI has one store."""
    if folder is not None:
        return folder
    if a.mount_cwd is not None or a.mount or a.harness == "open-webui":
        return None
    return settings.canonical(_cwd())


def _scope(folder: str | None) -> str:
    """Names a session by its project folder, or by nothing for the session
    that shares none."""
    return f" for {settings._tilde(folder)}" if folder else ""


def _busy(client: str, folder: str | None, state: str) -> Exception:
    """The refusal for a launch that meets a session that is not running
    yet, or not any more."""
    from gmlx.commands import launch as L

    scope, label = _scope(folder), target_label(client)
    if state == "ending":
        return L.LaunchError(f"the {label} session{scope} is ending. Launch again once it "
                             "has stopped.", L.EXIT_TEMPFAIL)
    return L.LaunchError(f"the {label} session{scope} is still starting. Try again in a "
                         "moment.", L.EXIT_TEMPFAIL)


def _holding_sessions(client: str, project: str, folder: str) -> list[tuple[str, dict]]:
    """The session records of the client's other projects that hold
    ``folder`` in their project folder or in a read-write share, by whole
    path components, as (project id, record) pairs, the longest share
    first. A read-only share alone does not count, because that session
    cannot change the files."""
    found = []
    for other, record in session.records(client):
        roots = [s["host"] for s in record["shares"] if not s.get("readonly")]
        if record.get("project"):
            roots.append(record["project"])
        hold = [len(root) for root in roots if settings._inside(folder, root)]
        if hold and other != project:
            found.append((max(hold), other, record))
    return [(other, record) for _, other, record in sorted(found, key=lambda f: -f[0])]


def _enclosing_session(client: str, project: str, folder: str) -> tuple[str, dict] | None:
    """The running session of another project that holds ``folder``, as
    :func:`_holding_sessions` finds it, as its project id and record. A
    session whose launch is gone is left out, and step 7 reports its
    container. A session that is starting or ending stops this launch,
    since it shares the files too. While the container query fails, a
    session of a live launch stops it as well."""
    found = _holding_sessions(client, project, folder)
    if not found:
        return None
    failed = None
    try:
        containers = cli.list_launch_containers()
    except ContainerError as e:
        # No session runs while the service is down, but a launch that
        # starts the service is starting its session. Only the starting
        # and ending marks tell the state then.
        containers, failed = [], e
    for other, record in found:
        state = session.session_state(client, other, record, containers)
        if state == "running":
            return other, record
        if state is None:
            continue
        if failed is not None and not (record.get("starting") or record.get("ending")):
            raise _unknown(client, record.get("project"), failed)
        raise _busy(client, record.get("project"), state)
    return None


def _unknown(client: str, folder: str | None, failed: ContainerError) -> Exception:
    """The refusal for a launch that meets the session record of a live
    launch while the container query fails, so the session can run, boot
    or end."""
    from gmlx.commands import launch as L

    reason = failed.reason if isinstance(failed, cli.Stuck) else str(failed).rstrip(".")
    text = (f"the {target_label(client)} session{_scope(folder)} shares this folder, and "
            f"launch cannot tell whether it runs, because {reason}. Try again once "
            "`container ls` works.")
    if isinstance(failed, cli.Stuck):
        # The restart of the service stops every container.
        text += f" A restart of the service also stops that session. {cli.RESTART_HINT}"
    return L.LaunchError(text, L.exit_code(failed))


def _overlap_line(client: str, project: str, plan) -> str | None:
    """A warning when a session of any launch target shares a folder that
    holds or lies inside a folder this launch shares, since two virtual
    machines then change the same files. A session counts while its container runs, and
    while its launch starts it: the image step, the start of the container
    service and the boot. A session whose container has stopped does not
    count, also while its launch ends."""
    mine = [m.source for m in plan.shares if m.kind == "share"]
    found = []
    for other_client in settings.launch_targets_on_disk():
        for other, record in session.records(other_client):
            if (other_client, other) != (client, project) and any(
                    settings._inside(s["host"], m) or settings._inside(m, s["host"])
                    for s in record["shares"] for m in mine):
                found.append((other_client, other, record))
    if not found:
        return None
    failed = False
    try:
        containers = cli.list_launch_containers()
    except ContainerError:
        # No session runs while the service is down, but a launch that
        # starts the service is starting its session.
        containers, failed = [], True
    names = []
    for c, o, r in found:
        if session.record_runs(c, o, r, containers):
            state = "running "
        elif session.session_state(c, o, r, containers) != "starting":
            continue
        elif failed and not r.get("starting"):
            state = ""                   # a live launch whose container can run or boot
        else:
            state = "starting "
        names.append(f"the {state}{target_label(c)} session{_scope(r.get('project'))}")
    if not names:
        return None
    return (f"[launch] {_listed(names)} {'shares' if len(names) == 1 else 'share'} files with "
            "this session. File locks do not reach from one virtual machine to another, so "
            "do not let two clients change the same file at once.")


def _project_volumes(launch_cfg: LaunchCfg, client: str, project: str) -> list[str]:
    """The volume entries that get the project's own name: those of the
    target's own view that are not listed for every target, which includes
    a runtime agent's dependency volume. The default project keeps the
    configured names."""
    if project == settings.PROJECT_DEFAULT:
        return []
    shared = launch_cfg.container.volumes
    return [v for v in launch_cfg.for_target(client).volumes if v not in shared]


def _session_command(ready, cfg, captured
                     ) -> tuple[list[str] | None, list[str] | None, str | None]:
    """The client's command without the arguments after --, which a copy
    that joins the session runs with its own, and for command: image the
    ENTRYPOINT that such arguments follow in place of CMD and the image's
    working folder, where the command runs. A shell session records the
    client's command too, and it starts in another folder."""
    folder = None
    try:
        if ready is not None:
            command, folder = images.image_command(ready, cfg.command, captured["argv"], [])
        elif isinstance(cfg.command, list):
            command = list(cfg.command)
        elif cfg.command == "image":
            return None, None, None
        else:
            command = list(captured["argv"])
    except images.ImageError:
        return None, None, None
    if cfg.command == "image" and ready is not None:
        return command, list(ready.info.entrypoint or []), folder
    return command, None, None


def _join(a, cfg, project: str, folder: str | None, say) -> int:
    """Run another copy of the client, or a shell under --shell, in the
    running session of ``project``, whose folder is ``folder``, or open a
    web app that runs."""
    from gmlx.commands import launch as L

    client, scope = a.harness, _scope(folder)
    label = target_label(client)

    def refused(dest: str, value, record: dict | None = None) -> Exception:
        flag = _flag_name(dest, value)
        if dest == "mount":
            held = _share_specs(record or {})
            join = (f", or give --mount only shares that the session has, as it has them: "
                    f"{_listed(held)}" if held else "")
            step = (f"To join the session, leave out --mount{join}. To change the shares, "
                    "end the session, then launch again.")
        elif dest == "config_only":
            step = f"End the session, then launch again with {flag}."
        else:
            step = (f"To join the session, leave out {flag}. To use {flag}, end the "
                    "session, then launch again.")
        return L.LaunchError(f"a {label} session is already running{scope}, and {flag} "
                             f"applies only to a new session. {step}")
    # The shares of the running session tell if --mount applies, so the
    # check for --mount comes after the session is known to run.
    for dest, default in _JOIN_REFUSED.items():
        value = getattr(a, dest, default)
        if value != default and dest != "mount":
            raise refused(dest, value)
    containers = [c for c in cli.list_launch_containers() if c.state == "running"
                  and c.labels.get("gmlx.launch.client") == client
                  and c.labels.get("gmlx.launch.project") == project]
    try:
        record = session.read_record(client, project)
    except SettingsError as e:
        names = [c.name for c in containers]
        stop = (f"Stop it with: container stop {names[0]}" if len(names) == 1
                else "Quit the client in that session")
        raise L.LaunchError(f"{e} End that session and launch again. {stop}") from None
    name = (record or {}).get("name")
    if record and record.get("ending"):
        raise _busy(client, folder, "ending")
    if not record or record.get("starting") or not any(c.name == name for c in containers):
        raise _busy(client, folder, "starting")
    _refuse_second_detach(a, folder)
    if a.mount and not _shares_held(a.mount, record):
        raise refused("mount", a.mount, record)
    if client == "dsh" and not a.shell:
        want, have = a.dsh_profile or L._DSH_PROFILE, record.get("profile")
        if have is not None and want != have:
            raise L.LaunchError(f"a dsh session with the {have} profile is already "
                                f"running{scope}, and a project runs one session at a time. "
                                f"End it to start the {want} profile.")
    shares = [Mount(s["host"], s["guest"], bool(s.get("readonly"))) for s in record["shares"]]
    cwd = settings.guest_path(os.path.realpath(_cwd()), shares)
    web = record.get("web") and not a.shell
    ignored = [_flag_name(dest, getattr(a, dest, default))
               for dest, default in _JOIN_IGNORED.items()
               if getattr(a, dest, default) != default
               and (dest != "mount_cwd" or (a.mount_cwd and cwd is None))]
    if ignored:
        who = "the shell" if a.shell else "this launch" if web else "this copy"
        say(f"[launch] {_listed(ignored)} {'applies' if len(ignored) == 1 else 'apply'} only "
            f"to a new session, so {who} ignores {'it' if len(ignored) == 1 else 'them'}.")
    if web:
        return _web_again(client, project, cfg, record, say,
                          _unshared_line(record) if cwd is None and shares else None)
    copy_id = secrets.token_hex(8)
    entry = [runtime.GUEST_ENTRY, *(["--clipboard"] if record.get("clipboard") else []),
             "--join", "--copy-id", copy_id]
    if a.shell:
        say(f"[launch] opening a shell in the running {label} session{scope} ({name})")
        command = [*entry, "--shell", "--", *a.passthrough]
    else:
        base, entrypoint = record.get("command"), record.get("entrypoint")
        run = ([*entrypoint, *a.passthrough] if entrypoint is not None and a.passthrough
               else [*(base or []), *a.passthrough])
        if not run:
            raise L.LaunchError(f"the running {label} session{scope} does not record the "
                                "command it runs, so no copy can join it. Open a shell in it "
                                f"with: gmlx launch {label} --shell")
        say(f"[launch] joining the running {label} session{scope}")
        command = [*entry, "--", *run]
    if cwd is None and (shares or a.shell):
        what = "the shell opens" if a.shell else f"{label} starts"
        say(_unshared_line(record, f", so {what} in its working folder {record['workdir']}"))
    argv = cli.exec_argv(name, command, tty=session.stdin_is_tty(), cwd=cwd)
    return session.run_copy(argv, dict(os.environ), name=name, copy_id=copy_id)


def _shares_held(mounts: list[str], record: dict) -> bool:
    """Whether the running session of ``record`` already has each share
    that ``mounts`` names, with the same folder, container path and mode.
    The command that started a session, such as one with ``--mount .``,
    then joins it when you type it again."""
    held = [(s["host"], normal_guest_target(s["guest"]), bool(s.get("readonly")))
            for s in record["shares"]]
    for spec in mounts:
        try:
            source, target, readonly = settings.parse_mount_spec(spec)
        except SettingsError:
            return False
        real = settings.canonical(source)
        # A new session refuses a share through a link, so a join does too.
        if not settings._same(os.path.abspath(source), real):
            return False
        guest = normal_guest_target(target or real)
        if not any(settings._same(real, host) and guest == at and readonly == ro
                   for host, at, ro in held):
            return False
    return True


def _share_specs(record: dict) -> list[str]:
    """The shares of the running session of ``record``, each as --mount
    names it: the folder, then the container path when it is not the
    folder, then ``:ro`` for a read-only share."""
    out = []
    for s in record.get("shares") or []:
        guest = normal_guest_target(s["guest"])
        at = "" if guest == normal_guest_target(s["host"]) else f":{guest}"
        out.append(settings._tilde(s["host"]) + at + (":ro" if s.get("readonly") else ""))
    return out


def _unshared_line(record: dict, then: str = "") -> str:
    """The line for a launch that joins a session from a folder that none
    of its shares holds. It names the folders the session shares, and ends
    with ``then``."""
    shown = [settings._tilde(s["host"]) + (" (read-only)" if s.get("readonly") else "")
             for s in record["shares"]]
    which = f", which shares {_listed(shown)}" if shown else ""
    return f"[launch] the current folder is not shared with this session{which}{then}."


def _web_again(client: str, project: str, cfg, record: dict, say,
               unshared: str | None = None) -> int:
    """A second launch of a running web app says where it answers and opens
    it. dsh's address holds a login token, which the session records once
    dsh prints it. ``unshared`` is the line for a current folder that the
    session does not share. A session that runs a shell has no app to open
    until you start it there. A session on a port that the pages of another
    project used is not opened, as the launch that started it did not open
    it, so you can clear the site data of that address first."""
    port = record.get("web_port")
    label = target_label(client)
    url = record.get("url") if client == "dsh" else f"{session.web_origin(port)}/"
    ready = bool(port and url and url.startswith(f"{session.web_origin(port)}/")
                 and url.isprintable() and not record.get("shell"))
    reused = record.get("reused") is True
    opens = cfg.open_browser is not False and not reused
    if record.get("shell"):
        say(f"[launch] the running {label} session runs a shell. To open another shell in "
            f"the session, run: gmlx launch {label} --shell")
        say(f"[launch] {label} answers at {session.web_origin(port)}/ once you start it in "
            f"that shell{session.shell_start(_shell_record(client, project, record))}"
            f"{session.token_step(client)}")
    elif ready:
        say(f"[launch] {label} is already running at {url}")
    else:
        say(f"[launch] {label} is already running, and its web app has not printed its "
            f"address yet. The launch that started it {'opens' if opens else 'shows'} the "
            "address once it is ready.")
    if reused and port:
        say(f"[launch] the pages of another project or app used port {port} before this "
            f"session. {_reused_advice(port, dry=False)}")
    if unshared:
        say(unshared)
    if ready and url and opens:
        session.open_in_browser(url)
    return 0


def _shell_record(client: str, project: str, record: dict) -> dict:
    """The record whose command the shell line of a running session names.
    dsh's --from-default-profile makes the profile on the first start and
    refuses a profile that exists. So once the private home holds the
    profile's manifest, the command leaves out that flag and its template,
    as the first launch does when it finds the manifest."""
    command, profile = record.get("command"), record.get("profile")
    if (client != "dsh" or not isinstance(command, list) or not isinstance(profile, str)
            or _DSH_FROM_DEFAULT not in command[:-1]):
        return record
    home = settings.private_home_path(client, project)
    try:
        with confine.confined(home):
            made = confine.exists(home / ".dsh" / "profiles" / profile / "package.json")
    except (OSError, confine.ConfinedError):
        # A link that the guest put in the private home: the line keeps the
        # recorded command.
        return record
    if not made:
        return record
    at = command.index(_DSH_FROM_DEFAULT)
    return {**record, "command": command[:at] + command[at + 2:]}


def _remove_home(a, launch_cfg: LaunchCfg, project: str, folder: str | None,
                 say) -> int:
    """Remove the private home of this launch's project, and the records
    beside it, after a question on the terminal. For a runtime agent the
    same question names the project's dependency volume, which a yes
    deletes too. --mount-cwd, --no-mount-cwd and --mount choose the project,
    so they can go with --remove-home."""
    import shlex
    import shutil

    from gmlx.commands import launch as L
    from gmlx.commands.doctor import _WALK_CAP, _folder_bytes

    client, label = a.harness, target_label(a.harness)
    others = [_flag_name(dest, getattr(a, dest, None)) for dest in CONTAINER_FLAGS
              if dest not in ("remove_home", "mount_cwd", "mount") and _flag_set(a, dest)]
    if others or a.passthrough or a.config_only:
        what = others[0] if others else "--config-only" if a.config_only else "arguments after --"
        raise L.LaunchError(f"--remove-home removes a home and starts nothing, so it cannot "
                            f"go with {what}.")
    where = f" for {settings._tilde(folder)}" if folder else " for the default project"
    target = settings.project_dir_path(client, project)
    home = target / "home"
    have_home = home.is_dir() and not home.is_symlink()
    volume = _deps_volume_offered(launch_cfg, client, project)
    # The volume is looked for only when the service runs, since a stopped
    # service cannot delete it either.
    info, unchecked = None, None
    if volume is not None:
        if not _checked_program():
            unchecked = (f"[launch] Apple container is not installed, so the dependency volume "
                         f"{volume} was not looked for. Install it with: brew install container. "
                         "Then run --remove-home again.")
        elif cli.service().running:
            info = next((v for v in cli.volume_list() if v.name == volume), None)
        else:
            unchecked = (f"[launch] the container service is stopped, so the dependency volume "
                         f"{volume} was not looked for. Start it with: container system start. "
                         f"Then run --remove-home again, or delete the volume with: container "
                         f"volume delete {volume}")
    if not have_home and info is None:
        say(f"[launch] {label} has no private home{where}, so nothing was removed.")
        if unchecked:
            say(unchecked)
        # The user can delete the folder by hand. What the pages of the
        # project left in the browser stays, so launch names the addresses.
        _site_data_line(web_ports.release(client, project, unless_running=True), say)
        return 0
    lock = session.try_session_lock(client, project)
    if lock is None:
        raise L.LaunchError(f"the {label} session{where} is running. End it, then remove "
                            "its home.", L.EXIT_TEMPFAIL)
    held = [lock]
    try:
        if info is not None:
            # Taken as a launch takes it, so a session that mounts the volume
            # refuses the question, and nothing is removed.
            mount = Mount(info.name, AGENT_DEPS_TARGET, kind="volume")
            held.extend(session.lock_volumes([mount]))
            session.check_volumes_free([mount], cli.containers())
        rm_home = f"rm -rf {shlex.quote(str(target))}"
        rm_volume = f"container volume delete {volume}"
        if not session.stdin_is_terminal():
            if have_home and info is not None:
                yourself = (f"Remove the home yourself with: {rm_home}\n  and delete the "
                            f"volume with: {rm_volume}")
            elif have_home:
                yourself = f"Remove the home yourself with: {rm_home}"
            else:
                yourself = f"Delete the volume yourself with: {rm_volume}"
            raise L.LaunchError(f"--remove-home asks before it removes anything, and there is "
                                f"no terminal to ask on. {yourself}")
        if have_home:
            budget = [_WALK_CAP]
            size = session.gb(_folder_bytes(home, budget))
            more = "at least " if budget[0] <= 0 else ""
            question = (f"remove the private home of {label}{where}, {more}{size} at "
                        f"{settings._tilde(str(home))}, with its settings and history")
            if info is not None:
                used = session.gb(session.allocated_bytes(info.source) if info.source else 0)
                question += f", and its dependency volume {volume}, {used} on the Mac"
        else:
            used = session.gb(session.allocated_bytes(info.source) if info.source else 0)
            question = (f"{label} has no private home{where}. Delete its dependency volume "
                        f"{volume}, {used} on the Mac")
        try:
            answer = input(f"[launch] {question}? [y/N] ")
        except EOFError:                  # Ctrl-D answers no
            print()
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            say("[launch] nothing was removed.")
            return 1
        # The start mark goes with the folder, and it tells whether the
        # project's port served pages.
        started = session.started_path(client, project).exists()
        if have_home:
            # The guest can put links in the home, so no link is followed.
            with confine.confined(target):
                confine.remove_tree(home)
            shutil.rmtree(target, ignore_errors=True)
            say(f"[launch] removed {settings._tilde(str(target))}")
        _site_data_line(web_ports.release(client, project, started=started), say)
        if unchecked:
            say(unchecked)
        if info is not None:
            try:
                cli.volume_delete(info.name)
            except ContainerError as e:
                say(f"[launch] the volume {volume} was not deleted: {e} Delete it with: "
                    f"{rm_volume}")
                return 1
            say(f"[launch] deleted the volume {volume}")
        return 0
    finally:
        if not have_home:
            # The lock made the project's folder, which holds nothing else.
            session.drop_unused_project(client, project, lock)
        for item in reversed(held):
            item.release()
        settings.drop_empty_target(client)


def _site_data_line(ports: list[int], say) -> None:
    """Name the addresses that the web app of a removed project used, so
    you can clear what its pages left in the browser. A page that is still
    open can store data again after a clear, so its tabs close first."""
    if not ports:
        return
    these = "that address" if len(ports) == 1 else "these addresses"
    say(f"[launch] the web app of this project used "
        f"{_listed([session.web_origin(p) for p in ports])}. Its pages possibly left a "
        "service worker and stored data there, and a page that is still open keeps running. "
        f"Close each browser tab and window of {these}, and each window that the app's pages "
        f"opened, or quit the browser. Then clear the site data of {these} in your browser.")


# Sessions in the background: --detach, --stop and --list

class _DetachEvents:
    """The pipe on which a launch that --detach started tells the launch
    that started it how far its session got: ``started`` once ``container
    run`` runs, and ``answers`` with the address once the web app answers.
    That launch exits once it has what it waits for, so a later write
    fails, and the failure is dropped."""

    def __init__(self, fd: int):
        self._fd: int | None = fd
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> _DetachEvents | None:
        """The pipe that :data:`DETACH_FD_ENV` names, or None. A value that
        names no pipe is not used, so a stray variable writes nowhere."""
        value = os.environ.pop(DETACH_FD_ENV, None)
        try:
            fd = int(value or "")
            if not stat.S_ISFIFO(os.fstat(fd).st_mode):
                return None
        except (ValueError, OSError):
            return None
        os.set_inheritable(fd, False)
        return cls(fd)

    def send(self, event: str, **fields) -> None:
        line = (json.dumps({"event": event, **fields}) + "\n").encode()
        with self._lock:
            if self._fd is None:
                return
            try:
                os.write(self._fd, line)
            except OSError:
                self._close()

    def answered(self, url: str) -> None:
        self.send("answers", url=url)
        self.close()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None


def _refuse_second_detach(a, folder: str | None) -> None:
    """--detach starts a new session. When one runs, a web app's address
    prints as for any second launch. Any other target is refused, since a
    copy that joins the session needs a terminal of its own."""
    from gmlx.commands import launch as L

    detached = getattr(a, "detach", False) or getattr(a, "detach_events", None) is not None
    if detached and not _is_web(a):
        label = target_label(a.harness)
        raise L.LaunchError(f"{label} already runs{_scope(folder)}, and --detach starts only "
                            "a new session. gmlx launch --list shows the sessions, and "
                            f"{_stop_command(a)} in this folder ends this one.")


def _stop_command(a) -> str:
    """The --stop command that ends this launch's session from the current
    folder. It repeats the flags of this launch that choose the project."""
    words = ["gmlx", "launch", target_label(a.harness), "--stop"]
    if a.harness != "open-webui":
        if a.mount_cwd is not None:
            words.append("--mount-cwd" if a.mount_cwd else "--no-mount-cwd")
        for spec in a.mount or []:
            words += ["--mount", shlex.quote(spec)]
    return " ".join(words)


def _row_scope(row: session.SessionRow) -> str:
    if row.folder:
        return _scope(row.folder)
    if row.project == settings.PROJECT_DEFAULT and row.client != "open-webui":
        return " in the default project"
    return ""


def _end_line(row: session.SessionRow, configured: Callable[[str], bool]) -> str | None:
    """The line that tells how to end the session of ``row``. The command
    keys the row's project whatever launch.container.mount_cwd says, as the
    --remove-home step of a project does: --mount . keys the project
    folder, and --no-mount-cwd keys the default project in /, which no
    share holds. A leftover container, the session of an agent that is not
    in launch.agents, and a project whose folder launch does not know are
    stopped by their container."""
    label, where = target_label(row.client), _row_scope(row)
    if row.state == "leftover":
        return (f"[launch] {row.name} is left over from a launch that is gone. Stop it with: "
                f"container stop {row.name}") if row.name else None
    if not configured(row.client):
        return (f"[launch] {label} is not in launch.agents, so gmlx launch cannot stop its "
                f"session{where}. Stop it with: container stop {row.name}") if row.name else None
    if row.client == "open-webui":
        step = f"gmlx launch {label} --stop"
    elif row.project == settings.PROJECT_DEFAULT:
        step = f"gmlx launch {label} --stop --no-mount-cwd in /"
    elif row.folder:
        step = f"gmlx launch {label} --stop --mount . in {settings._tilde(row.folder)}"
    elif row.name:
        step = f"container stop {row.name}"
    else:
        return None
    return f"[launch] to end the {label} session{where}, run {step}"


def _detach(a, project: str, folder: str | None, lock, let_go, say) -> int:
    """Start this launch again without --detach, in a session of its own
    with no terminal and with its output in the project's output file, and
    follow that output here until the session runs. The new launch takes
    over the session ``lock``, so no other launch of the project starts in
    between, and only the launch that holds the lock writes the output
    file. ``let_go`` releases the lock when the new launch does not start."""
    from gmlx.commands import launch as L
    from gmlx.serve import procname

    client = a.harness
    path = session.output_path(client, project)
    try:
        try:
            out = session.open_output(path)
        except OSError as e:
            raise L.LaunchError(f"cannot write the output file {settings._tilde(str(path))} "
                                f"({e.strerror or e}).") from None
        argv = list(getattr(a, "argv_given", None) or [])
        cut = argv.index("--") if "--" in argv else len(argv)
        # Container mode can come from --detach alone, so the launch in the
        # background gets --container in its place.
        argv = [*(x for x in argv[:cut] if x != "--detach"), "--container", *argv[cut:]]
        read_end, write_end = os.pipe()
        env = procname.child_env()
        env[DETACH_FD_ENV] = str(write_end)
        env[DETACH_LOCK_ENV] = str(lock.fd)
        try:
            proc = subprocess.Popen(
                [*procname.gmlx_argv(procname.stable_executable()), "launch", *argv],
                stdin=subprocess.DEVNULL, stdout=out, stderr=out, env=env,
                pass_fds=(write_end, lock.fd), start_new_session=True)
        except OSError as e:
            os.close(read_end)
            raise L.LaunchError(f"cannot start the launch in the background "
                                f"({e.strerror or e}).") from None
        finally:
            os.close(write_end)
            os.close(out)
    except BaseException:
        let_go()
        raise
    # The launch in the background holds the lock through its own copy of
    # the descriptor.
    lock.release()
    return _follow(proc, read_end, path, client, project, folder, _is_web(a),
                   _stop_command(a), say)


def _adopted_lock(client: str, project: str):
    """The session lock that the launch with --detach handed to this one,
    or None."""
    value = os.environ.pop(DETACH_LOCK_ENV, None)
    try:
        fd = int(value or "")
    except ValueError:
        return None
    return session.adopt_session_lock(client, project, fd)


def _session_runs(client: str, project: str) -> bool:
    try:
        record = session.read_record(client, project)
        containers = cli.list_launch_containers()
    except (SettingsError, ContainerError):
        return False
    return (record is not None
            and session.session_state(client, project, record, containers) == "running")


def _follow(proc: subprocess.Popen, events_fd: int, path: Path, client: str, project: str,
            folder: str | None, web: bool, stop: str, say) -> int:
    """Copy the output of the launch in the background here until its
    container runs, and for a web app until the app answers. Then name the
    output file and the commands that list and end the session, ``stop``
    among them. A launch that exits before that passes on its exit code. A
    Ctrl-C ends only the wait, and the session goes on."""
    import codecs

    label, scope, shown = target_label(client), _scope(folder), settings._tilde(str(path))
    decode = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending, url, started, checked = b"", None, None, 0.0
    waiting = True
    events: int | None = events_fd

    def pump(output, final: bool = False) -> None:
        text = decode.decode(output.read(), final)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
    try:
        with open(path, "rb") as output:
            try:
                while waiting:
                    pump(output)
                    rc = proc.poll()
                    if rc is not None:
                        pump(output, final=True)
                        if rc < 0:
                            say(f"[launch] signal {-rc} ended the launch of {label} in the "
                                f"background{scope} before its session ran. Its output is in "
                                f"{shown}.")
                            return 128 - rc
                        return rc
                    ready, _, _ = select.select([events] if events is not None else [], [],
                                                [], 0.25)
                    if ready and events is not None:
                        chunk = os.read(events, 4096)
                        if not chunk:
                            os.close(events)
                            events = None
                        pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("event") == "started" and started is None:
                            started = time.monotonic()
                        elif event.get("event") == "answers" and isinstance(event.get("url"),
                                                                             str):
                            url = event["url"]
                    now = time.monotonic()
                    if web and url is not None:
                        waiting = False
                    elif not web and started is not None and now - checked >= 1.0:
                        checked = now
                        waiting = not _session_runs(client, project)
                    if waiting and started is not None and now - started > (
                            DETACH_ANSWER_WAIT if web else DETACH_RUN_WAIT):
                        break
                pump(output, final=True)
            except KeyboardInterrupt:
                pump(output, final=True)
                say(f"[launch] {label} goes on starting in the background{scope}, and its "
                    f"output goes to {shown}. gmlx launch --list shows it, and {stop} in this "
                    "folder ends it.")
                return 130
    finally:
        if events is not None:
            with contextlib.suppress(OSError):
                os.close(events)
        # The client's start output can hold terminal queries, and the
        # shell would read their answers as typed input.
        session._flush_terminal_input()
    if waiting:
        what = "its web app has not answered yet" if web else "its container is still starting"
        say(f"[launch] {label} runs in the background{scope}, and {what}.")
    else:
        say(f"[launch] {label} runs in the background{scope}"
            + (f" at {url}." if url else "."))
    say(f"[launch] its output goes to {shown}. gmlx launch --list shows the running "
        f"sessions, and {stop} in this folder ends this one.")
    return 0


def _session_to_stop(a, client: str, project: str, folder: str | None,
                     containers: list[cli.Container]) -> tuple[str, dict, str] | None:
    """The session that --stop ends, as (project id, record, state): this
    project's, else the running session of another project that holds this
    folder, which a launch from here would join. A running container whose
    launch is gone has the state leftover, and so does a running container
    of this project with no record that launch can read."""
    def state_of(key: str, record: dict) -> str | None:
        state = session.session_state(client, key, record, containers)
        if state is None and session.record_runs(client, key, record, containers):
            return "leftover"
        return state
    try:
        record = session.read_record(client, project)
    except SettingsError:
        record = None
    if record is not None:
        state = state_of(project, record)
        if state is not None:
            return project, record, state
    here = _join_folder(a, folder)
    if here:
        for other, held in _holding_sessions(client, project, here):
            state = state_of(other, held)
            if state is not None:
                return other, held, state
    for c in containers:
        if c.state == "running" and c.labels.get("gmlx.launch.client") == client \
                and c.labels.get("gmlx.launch.project") == project:
            return project, {"name": c.name, "project": folder}, "leftover"
    return None


def _stop(a, project: str, folder: str | None, say) -> int:
    """End the session that --stop names: SIGTERM to the launch that runs
    it, as when its window closes, and a wait until that launch is gone. A
    session from a launch that recorded no start time, and a container
    whose launch is gone, are stopped by their container. The project keeps
    its home, volumes and port."""
    from gmlx.commands import launch as L

    client = a.harness
    label = target_label(client)
    _checked_program()
    try:
        containers = cli.list_launch_containers()
    except ContainerError:
        # No container runs while the service is down, and a launch that
        # starts the service has written its record already.
        containers = []
    found = _session_to_stop(a, client, project, folder, containers)
    if found is None:
        say(f"[launch] no {label} session runs{_scope(folder)}.")
        # The other sessions of the client, each with the command that
        # ends it, such as one of the default project.
        for row in session.session_rows([client])[0]:
            line = _end_line(row, lambda _: True)
            if line:
                say(line)
        return 0
    key, record, state = found
    scope, name = _scope(record.get("project")), record.get("name")
    owner = record.get("pid") if record.get("pid_start") is not None else None
    if state == "leftover" or owner is None or not session._launch_alive(record):
        if not name:
            say(f"[launch] the {label} session{scope} has no container to stop.")
            return 0
        cli.stop(name, timeout=session.STOP_GRACE)
        cli.delete(name)
        try:
            left = any(c.name == name for c in cli.list_launch_containers())
        except ContainerError:
            left = False                   # a service that is down runs no container
        if left:
            raise L.LaunchError(f"the container {name} of the {label} session{scope} did not "
                                f"stop. Stop it with: container stop {name}", L.EXIT_TEMPFAIL)
        say(f"[launch] stopped the container {name} of the {label} session{scope}.")
        return 0
    if state != "ending":
        with contextlib.suppress(ProcessLookupError):
            os.kill(owner, signal.SIGTERM)
    deadline = time.monotonic() + STOP_WAIT
    while session._launch_alive(record):
        if time.monotonic() > deadline:
            # A launch that ignores a second SIGTERM while it cleans up
            # ends only with SIGKILL.
            hint = (f"Stop its container with: container stop {name}" if name else
                    f"It has no container yet. End its launch with: kill -KILL {owner}")
            raise L.LaunchError(f"the {label} session{scope} has not ended after "
                                f"{STOP_WAIT:.0f} s. {hint}", L.EXIT_TEMPFAIL)
        time.sleep(0.2)
    say(f"[launch] stopped the {label} session{scope}.")
    return 0


def _project_cell(row: session.SessionRow) -> str:
    if row.folder:
        return settings._tilde(row.folder)
    return "(default project)" if row.project == settings.PROJECT_DEFAULT else row.project


def _started_cell(started: int | None) -> str:
    if not started:
        return "-"
    when = time.localtime(started / 1e6)
    today = time.strftime("%Y-%m-%d") == time.strftime("%Y-%m-%d", when)
    return time.strftime("%H:%M" if today else "%Y-%m-%d %H:%M", when)


def _shown_url(url: str | None) -> str | None:
    """The address of a web app without its query, which holds dsh's login
    token. A second launch of the app opens the whole address."""
    if not url:
        return None
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def list_sessions(client: str | None, agents: Collection[str] | None = None) -> int:
    """Print the sessions of ``client``, or of every launch target, that
    start, run or end, and the containers left over from a launch that is
    gone, as ``gmlx launch --list`` shows them, each with the command that
    ends it. ``agents`` names the agents in launch.agents, and the session
    of any other agent is stopped by its container."""
    known = set(agents or ())

    def configured(target: str) -> bool:
        name = agent_name(target)
        return name is None or name in known
    try:
        with cli.query_timeout(LIST_QUERY_TIMEOUT):
            rows, error = session.session_rows([client] if client else None)
    except SettingsError as e:
        sys.stdout.flush()
        print(printable_lines(f"[launch] {e}"), file=sys.stderr)
        return 1
    unlisted = (f"[launch] the container list is not available, so a session can show as "
                f"unknown, and no leftover container is listed: {error}" if error else None)
    if not rows:
        _say(f"[launch] no {target_label(client)} session runs." if client
             else "[launch] no launch session runs.")
        if unlisted:
            _say(unlisted)
        return 0
    table = [("TARGET", "PROJECT", "STATE", "LAUNCH", "ADDRESS", "STARTED")]
    for r in rows:
        how = "-" if r.state == "leftover" else "detached" if r.detached else "foreground"
        table.append(tuple(printable(cell) for cell in (
            target_label(r.client), _project_cell(r), r.state, how, _shown_url(r.url) or "-",
            _started_cell(r.started))))
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]) - 1)]
    for row in table:
        _say("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) + "  " + row[-1])
    for r in rows:
        if r.output:
            _say(f"[launch] the {target_label(r.client)} session{_row_scope(r)} writes its "
                 f"output to {settings._tilde(r.output)}")
    for r in rows:
        line = _end_line(r, configured)
        if line:
            _say(line)
    if unlisted:
        _say(unlisted)
    return 0


def status_lines() -> list[str]:
    """The lines of ``gmlx status`` about launch sessions, or none. They
    ask the container service only when a session record exists."""
    if sys.platform != "darwin":
        return []
    try:
        with cli.query_timeout(LIST_QUERY_TIMEOUT):
            rows, error = session.session_rows(records_only=True)
    except OSError:
        return []
    except SettingsError as e:
        return [printable(f"launch sessions not listed: {e}")]
    out = []
    for r in rows:
        url = _shown_url(r.url)
        parts = [r.state, *(["detached"] if r.detached else []), *([url] if url else [])]
        out.append(printable(f"launch session {target_label(r.client)}{_row_scope(r)}: "
                             f"{', '.join(parts)}"))
    if error:
        out.append(printable(f"launch sessions: the container list is not available: {error}"))
    if rows:
        out.append(f"  {len(rows)} launch session{'s' if len(rows) != 1 else ''} - `gmlx "
                   "launch --list` lists them with the command that ends each one")
    return out


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


def _agent_env(plan) -> dict:
    """The uv variables of a runtime agent, by value: the project folder in
    the guest, the environment, the cache and the managed Pythons on the
    dependency volume, the lock rule of a read-only source, and offline
    mode without a network, so an environment that was never synced fails
    at once with uv's message. A client plan sets none."""
    if plan.source_guest is None:
        return {}
    env = {"UV_PROJECT": plan.source_guest,
           "UV_PROJECT_ENVIRONMENT": f"{AGENT_DEPS_TARGET}/venv",
           "UV_CACHE_DIR": f"{AGENT_DEPS_TARGET}/cache",
           "UV_PYTHON_INSTALL_DIR": f"{AGENT_DEPS_TARGET}/python"}
    if plan.source_readonly:
        env["UV_LOCKED"] = "1"
    if plan.network == "none":
        env["UV_OFFLINE"] = "1"
    return env


def _runtime_agent(launch_cfg: LaunchCfg, client: str):
    """The agent block of a runtime agent's target key, or None for a client
    or an agent that brings its own image."""
    try:
        agent = launch_cfg.agent(client)
    except KeyError:
        return None
    return agent if agent.runtime else None


def _deps_volume_offered(launch_cfg: LaunchCfg, client: str, project: str) -> str | None:
    """The name of the dependency volume that launch created for a runtime
    agent's project, which --remove-home offers to delete. None for a client,
    an agent with its own image, or an agent that configured a volume at the
    dependency folder, which is the user's."""
    agent = _runtime_agent(launch_cfg, client)
    if agent is None or any(normal_guest_target(parse_volume_spec(v)[1]) == AGENT_DEPS_TARGET
                            for v in agent.volumes):
        return None
    name = parse_volume_spec(agent_deps_volume(client))[0]
    return settings.project_volume_name(name, project) if project != settings.PROJECT_DEFAULT \
        else name


def _ignored_env_lines(launch_cfg: LaunchCfg, client: str, names: list[str],
                       env_values: dict[str, str]) -> list[str]:
    """One line for each env entry whose name launch sets in the container
    itself, so the entry has no effect. An entry of the target's own block
    should go, and its line prints at every launch. An entry of
    launch.container.env can serve other targets, so its line prints once
    and asks for nothing."""
    try:
        own_block = launch_cfg.agent(client)
    except KeyError:
        own_block = launch_cfg.container.clients.get(client)
    own = set(_split_env(own_block.env)[0]) if own_block is not None else set()
    label = target_label(client)
    lines: list[str] = []
    for name in dict.fromkeys(n for n in names if n in env_values):
        if name in own:
            lines.append(f"[launch] the entry {name} in {config_key(client, 'env')} has no "
                         f"effect, because launch sets {name} in the container for {label}. "
                         "Remove the entry.")
        else:
            lines.append(notices.Once(
                f"[launch] the entry {name} in launch.container.env has no effect for "
                f"{label}, because launch sets {name} in its container.",
                f"env-ignored:{client}:{name}"))
    return lines


def _split_env(entries: list[str]) -> tuple[list[str], dict[str, str]]:
    names, values = [], {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        names.append(name)
        if sep:
            values[name] = value
    return names, values


def _env_entry_value(entries: list[str], name: str) -> str | None:
    """The value that the env entries give ``name``: the entry's own with
    ``NAME=VALUE``, the Mac's with ``NAME`` alone, and None with no entry."""
    names, values = _split_env(entries)
    if name in values:
        return values[name]
    return os.environ.get(name) if name in names else None


def _dsh_profile_is_web(a) -> bool:
    """Whether a dsh profile runs the web app in a container, by its name
    only. The profile's manifest lies in the private home, which the guest
    writes, so it never decides whether the Mac opens a port and a browser."""
    from gmlx.commands import launch as L

    profile = L._DSH_PROFILE if a.dsh_profile is None else a.dsh_profile
    return profile in (L._DSH_PROFILE, L._DSH_TEMPLATE)


def _web_port(client: str, project: str, cfg, server_port: int, dry: bool,
              say, targets: Collection[str], moved_line: bool = True) -> web_ports.Choice:
    """The Mac port of the project's web app, from the range of
    :mod:`gmlx.container.web_ports`. The gmlx server's port and the
    forwarded ports are never used. The dry run records nothing. Without
    ``moved_line`` a move from the recorded port prints no line.
    ``targets`` are the configured launch targets, which gmlx launch can
    name in the step that frees a port."""
    choice = web_ports.choose(client, project, avoid={int(server_port), *cfg.forward},
                              record=not dry, configured=set(targets).__contains__)
    port = choice.port
    if choice.moved and moved_line:
        verb = "would move" if dry else "moves"
        say(f"[launch] port {choice.before} of the {target_label(client)} web app of this "
            f"project is not free, so the app {verb} to port {port}. The browser keeps "
            "sign-ins and saved data by address, so the app can ask you to sign in again.")
    if choice.reused:
        verb = "would take" if dry else "takes"
        say(f"[launch] the {target_label(client)} web app of this project {verb} port {port}, "
            f"which the pages of another project or app used. {_reused_advice(port, dry)}")
    return choice


def _reused_advice(port: int, dry: bool) -> str:
    """What to do before you open a web app on a port that the pages of
    another project or app used. A launch does not open such a port. A page
    of that project that is still open would be the same origin as the
    app, so its tabs close before the site data is cleared."""
    then = "" if dry else " This launch does not open the browser, so you can do that first."
    return (f"They possibly left a service worker and stored data at "
            f"{session.web_origin(port)}, and a page that is still open keeps running and can "
            "store data again. So close each browser tab and window of that address, and each "
            "window that its pages opened, or quit the browser. Then clear the site data of "
            f"that address before you open the app.{then}")


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


# The checks of the shares and the seeds in one launch share their tables
# and their answer for each folder.
@settings.launch_memo()
def run_container(a, launch_cfg: LaunchCfg, *, exec_fn) -> int:
    from gmlx.commands import launch as L

    say = _say
    client = a.harness
    cfg = launch_cfg.for_target(client)
    dry = bool(a.config_only)
    a.container_mode = True
    try:
        if a.config_path:
            raise L.LaunchError("--config-path does not apply in container mode, where the "
                                "client's configuration goes in its private home.")
        project, folder = _session_key(a, cfg)
        if getattr(a, "remove_home", False):
            return _remove_home(a, launch_cfg, project, folder, say)
        if getattr(a, "stop", False):
            return _stop(a, project, folder, say)
        if getattr(a, "detach", False) and agent_name(client) is None and not _is_web(a):
            what = (f"the dsh profile {a.dsh_profile}" if client == "dsh"
                    else target_label(client))
            raise L.LaunchError(f"--detach runs Open WebUI, a dsh web profile or a custom "
                                f"agent in the background, and {what} needs a terminal. "
                                "Launch it without --detach.")
        # A launch that --detach started reports its start on this pipe.
        a.detach_events = _DetachEvents.from_env()
        # Step 3. A stopped service starts only after the refusals of step 6.
        prereqs = _Prereqs()
        # Step 4. A launch that --detach started holds the lock that the
        # launch which started it took.
        lock = _adopted_lock(client, project) or session.try_session_lock(client, project)

        def let_go() -> None:
            if lock is not None:
                session.drop_unused_project(client, project, lock)
                lock.release()
        # A session that shares a folder holding this one takes this launch
        # too, since a second virtual machine would share the same files.
        # Another launch that holds this project's lock but has written no
        # record yet may be about to join that session, so this launch
        # looks there first.
        enclosing = None
        here = _join_folder(a, folder)
        if here and (lock is not None or not session.record_path(client, project).exists()):
            try:
                enclosing = _enclosing_session(client, project, here)
            except L.LaunchError:
                let_go()
                raise
        if enclosing is not None:
            let_go()
            other, record = enclosing
            return _join(a, cfg, other, record.get("project"), say)
        if lock is None:                  # joining refuses --config-only itself
            return _join(a, cfg, project, folder, say)
        if getattr(a, "detach", False):
            try:
                prereqs.require_installed()
                if not prereqs.running and not cli.kernel_installed():
                    # The first start of the service asks a question, and the
                    # session in the background has no terminal for it.
                    raise L.LaunchError("the container service is not running, and its first "
                                        "start asks whether to install a Linux kernel, which "
                                        "--detach has no terminal for. Run it once with: "
                                        "container system start. Then launch again.",
                                        L.EXIT_UNAVAILABLE)
            except BaseException:
                let_go()
                raise
            return _detach(a, project, folder, lock, let_go, say)
        held = [lock]
        try:
            if dry:
                say("[launch] container dry run: no image is built or pulled, and no "
                    "container is started.")
                for line in prereqs.report():
                    say(line)
            else:
                prereqs.require_installed()
            # Until the session starts, a repeated container query reuses
            # its first answer.
            with cli.memoized():
                return _run_locked(a, launch_cfg, cfg, prereqs, held, exec_fn, say,
                                   project, folder)
        finally:
            if not dry:
                # The starting record of a launch that stopped before its
                # session ran.
                with contextlib.suppress(OSError):
                    session.remove_record(client, project)
            for item in reversed(held):
                item.release()
    except (L.LaunchError, SettingsError, ContainerError, ConfigError,
            confine.ConfinedError) as e:
        sys.stdout.flush()
        print(printable_lines(f"[launch] {e}"), file=sys.stderr)
        return L.exit_code(e)
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


def _run_locked(a, launch_cfg, cfg, prereqs, held, exec_fn, say, project: str,
                folder: str | None) -> int:
    from gmlx.commands import launch as L
    from gmlx.serve import procname

    client = a.harness
    dry = bool(a.config_only)
    # Step 5
    session.remove_record(client, project)
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
    _, api_port, _ = guest_url(a.base_url or L._base_url(host, port))
    # A refused --model, a missing server or an old server stops the launch
    # before it writes a home, a record or a once-notice. A server that
    # launch starts is checked after the other refusals of this step.
    check = _server_precheck(a, dry)
    if check.rc is not None:
        return check.rc
    web = _is_web(a)
    web_choice = (_web_port(client, project, cfg, port, dry, say, launch_cfg.targets())
                  if web else None)
    web_port = web_choice.port if web_choice else None
    guest_web_port = _guest_web_port(a, web_port, api_port)
    # Launch does not open a port that the pages of another project used,
    # so you can clear its site data first.
    reused = bool(web_choice and web_choice.reused)
    # A read-write share of any target's build: folder would let this
    # client change what that image runs.
    builds = {c: launch_cfg.for_target(c).build for c in launch_cfg.targets()}
    agent = _runtime_agent(launch_cfg, client)
    plan = settings.resolve_plan(client, cfg, cwd=_cwd(), mount_cwd=a.mount_cwd,
                                 cli_mounts=a.mount, network=a.network, api_port=api_port,
                                 web_port=guest_web_port,
                                 build_folders={c: b for c, b in builds.items() if b},
                                 project=project,
                                 project_volumes=_project_volumes(launch_cfg, client, project),
                                 source=agent.source if agent else None,
                                 runtime=agent is not None)
    settings.check_program(prereqs.binary, [m.source for m in plan.shares if not m.readonly])
    # --list names the output file of a detached session from the start.
    detached = ({"detached": True, "output": str(session.output_path(client, project))}
                if getattr(a, "detach_events", None) is not None else {})
    if not dry:
        # A launch from a folder this session will share waits for it,
        # instead of starting a second virtual machine on the same files.
        session.write_record(client, project, {
            "name": "", "workdir": plan.workdir, "starting": True, **session.launch_owner(),
            "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                       for m in plan.shares],
            "project": folder, "web": web, "web_port": web_port, **detached})
    # The line prints until a session of the project reaches container run.
    if plan.new_home or not session.started_path(client, project).exists():
        plan.notes.insert(0, settings.new_home_line(client, project))
    overlap = _overlap_line(client, project, plan)
    if overlap:
        say(overlap)
    if api_port is None and plan.network == "none":
        raise L.LaunchError(f"network: none cannot reach {a.base_url}, which is not a local "
                            "http server. Use the default network for this server.")
    # A build: folder the client can write would run its code at the next build.
    writable = [m.source for m in plan.mounts if not m.readonly and m.kind != "volume"]
    image_plan = images.resolve_image(client, cfg, launch_cfg.container,
                                      stage=images.stage_for(launch_cfg, client),
                                      image_override=a.image, writable=writable)
    config_notes: list[str] = []
    config_path = settings.server_config_path(host, port,
                                              autostart=not (a.base_url or a.no_start),
                                              notes=config_notes)
    started = check.start
    # The server and the menu bar that launch starts run programs by name,
    # so their PATH leaves out each folder that a client can write.
    server_path = settings.server_path(plan.mounts)
    if started:
        with procname.child_path(server_path):
            rc = L._ensure_server(a)
        if rc is not None:
            return rc
        check = _probe_sessions(a, a.base_url, dry)
    # The lines that print once are recorded only when the session starts,
    # so a launch that stops before it shows them again.
    shown = notices.due([*plan.warnings, *plan.notes, *image_plan.notices, *config_notes,
                         *settings.server_config_warnings(config_path, plan.shares),
                         *settings.pythonpath_warnings(plan.shares)], record=False)
    for line in shown:
        say(line)
    agent_line = settings.agent_key_line(plan)
    if agent_line:
        say(agent_line)
    # The plan only names the private home, so a refused launch makes none.
    settings.private_home(client, project)
    # A dry run copies no seed again, since that replaces the client's edits.
    reseed = getattr(a, "reseed", False)
    for line in settings.seed_home(plan.home, plan.seed, reseed=reseed and not dry,
                                   writable=settings.seed_writable(plan, _cwd())
                                   if plan.seed else ()):
        say(line)
    settings.ready_home(client, plan.home)
    settings.write_project_record(client, project, folder)
    if reseed and not plan.seed:
        say(f"[launch] --reseed has nothing to copy, because no seed is configured for "
            f"{target_label(client)}.")
    elif reseed and dry:
        say(f"[launch] the dry run copies no seed again. A launch with --reseed copies "
            f"{_listed(plan.seed)} again, in place of the copies in the private home.")
    if not dry:
        settings.record_shares(plan)
    # The service start and its kernel download come after every refusal.
    # The first start of the service is the first of three steps, with the
    # image and the client after it.
    first_run = False if dry else prereqs.start_service(say, "step 1 of 3")
    prereqs.require_kernel()
    running = prereqs.running
    ready = None
    steps = 3 if first_run else 0
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
        session.cleanup_stale(client, project, keep_runtime=runtime_dir.name, say=say)
        containers = cli.containers()
        for line in session.orphan_notices(client, project, containers):
            say(line)
        memory = session.memory_line(containers, plan.memory)
        if memory:
            say(memory)
        pending = images.pending_work(image_plan, a.rebuild)
        # A build is about to use the builder, so an owed stop waits for it.
        # A pull does not use the builder, and nothing stops it after a pull.
        notice = images.builder_notice(say=say, settle=pending != "build")
        if notice:
            say(notice)
        # Step 8. A closed terminal tab or a second Ctrl-C during a long
        # first build must still let the build's clean-up run, which records
        # the builder's owed stop.
        with _signals_raise():
            runtime_dir, runtime_lock = runtime.acquire_runtime()
            held.append(runtime_lock)
            held.extend(session.lock_volumes(plan.volumes))
            session.check_volumes_free(plan.volumes, containers)
            session.ensure_volumes(plan.volumes, say)
            # The steps are numbered when this launch starts the service for
            # the first time or builds or pulls an image.
            if pending and not first_run:
                steps = 2
            ready = images.ensure_image(
                image_plan, rebuild=a.rebuild, say=say,
                step=f"step {steps - 1} of {steps}" if pending else None)
            if first_run and not pending:
                say(f"[launch] step 2 of 3: found {ready.tag} in the image store")
            images.forget_unnamed(launch_cfg, say)
            try:
                # A runtime agent's command is a shell script that runs uv.
                words = (["uv", "sh"] if agent is not None
                         else [cfg.command[0]] if isinstance(cfg.command, list)
                         else [images.image_command(ready, "image", [], a.passthrough)[0][0]]
                         if cfg.command == "image" else [images.CLIENT_BINARY[client]])
            except images.ImageError as e:
                if not a.shell:            # a shell is how you look into such an image
                    raise
                say(f"[launch] warning: {e}")
            else:
                for word in words:
                    images.check_command(ready, word, str(runtime_dir), shell=a.shell,
                                         say=say, runtime=agent is not None)
    # Step 9. A server that step 6 started is ready.
    if not started:
        with procname.child_path(server_path):
            rc = L._ensure_server(a)
        if rc is not None:
            return rc
    base = a.base_url or L._base_url(a.host, a.port)
    a.guest_base_url, api_port, api_targets = guest_url(base)
    if api_port is None and plan.network == "none":
        raise L.LaunchError(f"network: none cannot reach {base}, which is not a local http "
                            "server. Use the default network for this server.")
    from gmlx.container.localhost_domains import launch_warning
    if plan.network != "none" and (note := launch_warning()) is not None:
        say(note)
    if int(a.port) != port:
        # The server check found the server on another port than step 6
        # assumed, so the ports that depend on it are worked out again.
        if web_choice is not None and web_port == int(a.port):
            # Step 6 recorded the port of a first launch a moment ago. The
            # app had no address before, so the move needs no line.
            web_choice = _web_port(client, project, cfg, int(a.port), dry, say,
                                   launch_cfg.targets(),
                                   moved_line=web_choice.before is not None)
            web_port = web_choice.port
            reused = reused or web_choice.reused
        guest_web_port = _guest_web_port(a, web_port, api_port)
        plan.forward = settings.forward_ports(plan.forward, api_port=api_port,
                                              web_port=guest_web_port)
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
                                       [web_port] if web_port is not None else [],
                                       project)
    full_api = None if server_session else full_api_line(base, client, a.api_key)
    if server_session and open_bind(base, api_targets) and not L._auth_required(base):
        full_api = open_bind_line(base)
    # Step 10
    if L.requested_model(a) and not a.no_keep and not dry:
        L._pick_default(L.probe_models(base, a.api_key, client), L.requested_model(a),
                        origin=L.model_origin(a))
        L._keep_model(a)
    # Step 11. The last step line prints before the client's own lines.
    if steps:
        say(f"[launch] step {steps} of {steps}: starting "
            f"{'a shell' if a.shell else target_label(client)}")
    captured: dict = {}

    def sink(argv, pairs, extra):
        captured.update(argv=argv, pairs=pairs, extra=extra)
        return 0
    a.container_sink = sink
    a.container_web_port = web_port
    a.container_context_tokens = _env_entry_value(plan.env, L.CONTEXT_TOKENS)
    # Under --shell the client does not start, so its summary and notes
    # would describe a program that is not running.
    quiet = contextlib.redirect_stdout(io.StringIO()) if a.shell else contextlib.nullcontext()
    # The handler runs with HOME in the private home, which the guest
    # writes, so the facts it needs from gmlx's state on the Mac are read
    # here. Only the handlers of Claude Code and of an agent with api:
    # anthropic use the server's config, for the context window.
    anthropic = agent_name(client) is not None and a.agent_cfg.api == "anthropic"
    a.served_config = (L._served_config(a.host, a.port)
                       if client == "claude-code" or anthropic else None)
    a.no_models_text = L.no_models_message(L._server_root(base))
    handler = L._HARNESSES.get(client, L._launch_agent)
    with guest_home(plan.home), quiet:
        rc = handler(a, exec_fn=exec_fn)
    sys.stdout.flush()
    if rc != 0 or not captured:
        return rc
    if client == "aichat" and not a.shell:
        for line in notices.due([L.aichat_notice()], record=not dry):
            print(line)
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
                  **_client_env(client, plan, ready, cfg.command, guest_web_port),
                  **_agent_env(plan)}
    pair_names = [n for n in captured["pairs"] if n not in env_values]
    env_names = list(dict.fromkeys([*pair_names, *(n for n in names if n not in env_values)]))
    for line in notices.due(_ignored_env_lines(launch_cfg, client, names, env_values),
                            record=not dry):
        say(line)
    child_env = {**captured["pairs"], **values}
    if plan.ssh_socket:
        # container run forwards the agent that its own SSH_AUTH_SOCK names.
        child_env["SSH_AUTH_SOCK"] = plan.ssh_socket
    if L.CONTEXT_TOKENS in captured["pairs"]:
        # The launcher has already weighed the entry's value against the window.
        child_env[L.CONTEXT_TOKENS] = captured["pairs"][L.CONTEXT_TOKENS]
    # dsh prints its URL with a per-process login token, which the Mac
    # browser needs, so launch reads it from the client's output.
    token_url = client == "dsh" and web_port is not None and not a.shell
    if dry:
        sess = session.Session(client, "xxxxxx", Path("<session folder>"), project)
    else:
        sess = session.new_session(client, project, plan.forward)
        image_ref = ready.run_ref
    spec = session.RunSpec(
        session=sess, plan=plan, image_ref=image_ref, runtime_dir=runtime_dir,
        command=command, workdir=image_workdir or plan.workdir, env_values=env_values,
        env_names=env_names, child_env=child_env, api_port=api_port, web_port=web_port,
        web_guest_port=guest_web_port, tty=session.stdin_is_tty() and not token_url,
        interactive=not token_url,
        shell=a.shell,
        url_pattern=_DSH_URL_LINE if token_url else None,
        labels={"gmlx.launch.runtime": runtime_dir.name})
    summary = _summary_lines(plan, None if dry else ready, a.shell, client, spec.workdir)
    if full_api:
        summary.append(full_api)
    if dry:
        if session_line:
            summary.append(session_line)
        return _print_dry_run(spec, plan, image_line, summary, cfg, captured, running, say)
    command_base, entrypoint, command_workdir = _session_command(ready, cfg, captured)
    record = {"name": sess.name, "workdir": spec.workdir, "clipboard": plan.clipboard == "images",
              "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                         for m in plan.shares],
              "command": command_base, "entrypoint": entrypoint,
              "command_workdir": command_workdir, "project": folder,
              "web": web, "web_port": web_port, "shell": bool(a.shell),
              "profile": (a.dsh_profile or L._DSH_PROFILE) if client == "dsh" else None,
              # A second launch does not open a port that this launch did not open.
              "reused": reused,
              # A launch that finds this record while the container boots
              # sees from the live launch that the session is starting.
              **session.launch_owner()}
    events = getattr(a, "detach_events", None)
    record.update(detached)
    # Under --shell the app is not running yet, so there is nothing to open.
    opener = (session.open_in_browser
              if (web_port and plan.open_browser and not a.shell and not reused) else None)
    start = command_base if client == "dsh" and a.shell and command_base else []
    if _DSH_FROM_DEFAULT in start[:-1]:
        # The shell line names this command, and dsh refuses the flag once
        # the profile exists.
        template = start[start.index(_DSH_FROM_DEFAULT) + 1]
        summary.append(f"[launch] the first start of dsh in the shell makes its "
                       f"{record['profile']} profile from the {template} template. To start "
                       f"dsh again after that, leave out {_DSH_FROM_DEFAULT} {template}.")
    cli.end_memo()

    def started() -> None:
        if web_port is not None:
            # From now on, another project gets this port only when no
            # other port is free.
            web_ports.mark_served(client, project, web_port)
        notices.record(shown)
        session.mark_started(client, project)
        if events is not None:
            events.send("started")
            if web_port is None:
                events.close()
    return session.supervise(spec, api_targets=api_targets, record=record, say=say,
                             opener=opener, summary=summary, server_session=server_session,
                             on_start=started,
                             on_answer=events.answered if events is not None else None,
                             output_max=session.OUTPUT_MAX if events is not None else None)


def _summary_lines(plan, ready, shell: bool, client: str, workdir: str) -> list[str]:
    lines = [images.describe(ready)] if ready is not None else []
    if ready is not None:
        note = images.image_age_note(ready)
        if note:
            lines += notices.due([note])
    holder = None
    if plan.cwd_shared:
        guest = [m for m in plan.shares if m.target == plan.workdir
                 or plan.workdir.startswith(m.target.rstrip("/") + "/")]
        holder = max(guest, key=lambda m: len(m.target), default=None)
    for m in plan.shares:
        mode = "read-only" if m.readonly else "read-write"
        where = "" if m.target == m.source else f" at {m.target}"
        extra = ", working folder" if m is holder else ""
        if m.kind == "git" or m.note == "source folder":
            extra = f", {m.note}"
        lines.append(f"[launch] sharing {settings._tilde(m.source)}{where} ({mode}{extra})")
    if ready is not None:
        lines += session.volume_lines(plan.volumes)
    for port in plan.forward:
        lines.append(f"[launch] forwarding the container's 127.0.0.1:{port} to Mac port "
                     f"{port}")
    if plan.network == "none":
        lines.append("[launch] with network none, the client reaches only the gmlx server and "
                     "the forwarded ports, and a download such as npm install fails")
    if not plan.cwd_shared and client not in settings.NO_CWD_CLIENTS:
        where = "its private home" if workdir == str(plan.home) else workdir
        who = "the shell opens" if shell else f"{target_label(client)} starts"
        lines.append(f"[launch] the current folder is not shared, so {who} in {where}")
    if shell:
        lines.append(f"[launch] opening a shell instead of {target_label(client)}")
    return lines


def _print_dry_plan(runtime_dir, plan, image_line, summary, running, say) -> None:
    for line in image_line.split("\n"):
        say(line)
    say(f"[launch] /opt/gmlx in the container holds launch's program from {runtime_dir}")
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
    # An agent has no command of its own that the setting could replace.
    if (isinstance(cfg.command, list) or cfg.command == "image") \
            and agent_name(spec.session.client) is None:
        say("[launch] the command: setting replaces the client's own command, "
            f"{shlex.join(captured['argv'])}")
    argv = session.compose_run_argv(spec)
    say("[launch] the command, which needs the API socket that only a running launch "
        "provides:")
    print(printable(shlex.join(argv)))
    return 0
