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
from gmlx.container import cli, confine, images, notices, runtime, session, settings
from gmlx.container.cli import ContainerError
from gmlx.container.settings import Mount, SettingsError
from gmlx.container.text import printable, printable_lines

# Flags that only mean something in container mode, by argparse dest.
CONTAINER_FLAGS = {"mount": "--mount", "mount_cwd": "--mount-cwd", "image": "--image",
                   "rebuild": "--rebuild", "reseed": "--reseed", "network": "--network",
                   "shell": "--shell", "remove_home": "--remove-home"}
# Flags a launch that joins a running session ignores, since that session
# already has its server and model, and the flags it refuses, which shape a
# new session. --mount-cwd counts as ignored only when the session does not
# share the current folder. --no-mount-cwd chose the project the launch
# joins, so it never counts. A dsh profile must match the running one.
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
    no ``except Exception`` on the way can stop the launch from exiting."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


@contextlib.contextmanager
def _signals_raise():
    """Turn SIGTERM and SIGHUP into an exception while the block runs, so
    every ``finally`` in it runs. The exit code is 128 plus the signal.
    Ctrl-C raises KeyboardInterrupt, as it does outside the block.

    The first signal raises. A closed window sends launch a second SIGHUP
    while those clean-ups run, and a user can press Ctrl-C two times. Thus
    the second signal of any of the three is ignored, and the clean-up that
    records the builder's owed stop can finish. The third signal and each
    signal after it raise again, so a clean-up that waits for a container
    service that does not answer stops. A signal that was ignored when
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
            return
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise _Signalled(signum)
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
    addresses that :func:`guest_url` found for a host name."""
    host = urllib.parse.urlsplit(base_url).hostname or ""
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return any(not loopback_host(addr) for addr, _ in targets)
    return not loopback_host(host)


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
            f"socket, and {client} {gets}.")


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
        # Every later container call of this launch runs this one file.
        self.binary = cli.pin()
        settings.check_program(self.binary)
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
            if self.version is None:
                lines.append(f"[launch] {self.binary}, the first container program on PATH, "
                             f"gives no version number, and this mode needs {need} or "
                             f"newer. {cli.UPGRADE_HINT}")
            elif self.version < cli.CONTAINER_MIN:
                lines.append(f"[launch] container {v} at {self.binary} is older than the "
                             f"{need} this mode needs. {cli.UPGRADE_HINT}")
        if not self.entry.is_file():
            lines.append(f"[launch] the container program {self.entry} is not built. Build "
                         f"it with: {runtime.BUILD_HINT}")
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
                              f"{self.binary}, the first container program on PATH, "
                              f"{have}. {cli.UPGRADE_HINT}", EXIT_UNAVAILABLE)
        if not self.entry.is_file():
            raise LaunchError(f"the container program {self.entry} is not built. In a git "
                              f"checkout, build it with: {runtime.BUILD_HINT}",
                              EXIT_UNAVAILABLE)

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
                raise LaunchError(f"{e} The container service does not answer, so no "
                                  "kernel can be installed yet. Read its log with: "
                                  "container system logs", EXIT_UNAVAILABLE) from None
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
    """Whether the launch runs a web app, which the Mac reaches on one port
    per client."""
    return a.harness == "open-webui" or (a.harness == "dsh" and _dsh_profile_is_web(a))


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


def _scope(folder: str | None) -> str:
    """Names a session by its project folder, or by nothing for the session
    that shares none."""
    return f" for {settings._tilde(folder)}" if folder else ""


def _busy(client: str, folder: str | None, state: str) -> Exception:
    """The refusal for a launch that meets a session that is not running
    yet, or not any more."""
    from gmlx.commands import launch as L

    scope = _scope(folder)
    if state == "ending":
        return L.LaunchError(f"the {client} session{scope} is ending. Launch again once it "
                             "has stopped.", L.EXIT_TEMPFAIL)
    return L.LaunchError(f"the {client} session{scope} is still starting. Try again in a "
                         "moment.", L.EXIT_TEMPFAIL)


def _enclosing_session(client: str, project: str, folder: str) -> tuple[str, dict] | None:
    """The running session of another project that holds ``folder`` in its
    project folder or in a read-write share, by whole path components, as
    its project id and record. When several do, the one with the longest
    share wins. A read-only share alone does not count, because that
    session cannot change the files. A session whose launch is gone is left
    out, and step 7 reports its container. A session that is starting or
    ending stops this launch, since it shares the files too."""
    found = []
    for other, record in session.records(client):
        roots = [s["host"] for s in record["shares"] if not s.get("readonly")]
        if record.get("project"):
            roots.append(record["project"])
        hold = [len(root) for root in roots if settings._inside(folder, root)]
        if hold and other != project:
            found.append((max(hold), other, record))
    if not found:
        return None
    try:
        containers = cli.list_launch_containers()
    except ContainerError:
        return None                      # no session runs while the service is down
    for _, other, record in sorted(found, key=lambda f: -f[0]):
        state = session.session_state(client, other, record, containers)
        if state == "running":
            return other, record
        if state is not None:
            raise _busy(client, record.get("project"), state)
    return None


def _one_web_session(client: str, project: str) -> None:
    """Refuse a new web app session while a session of the client in
    another project starts, runs or ends, since the web app has one port on
    the Mac. A session whose launch is gone holds no port."""
    from gmlx.commands import launch as L

    others = [(o, r) for o, r in session.records(client) if o != project and r.get("web")]
    if not others:
        return
    try:
        containers = cli.list_launch_containers()
    except ContainerError:
        containers = []                  # no session runs while the service is down
    for other, record in others:
        state = session.session_state(client, other, record, containers)
        if state is None:
            continue
        folder = record.get("project")
        if state == "ending":
            then = " Launch again once it has stopped."
        elif folder:
            then = (f" To open it, launch {client} from {settings._tilde(folder)}. To start "
                    "one here, end it first.")
        else:
            then = " To start one here, end it first."
        raise L.LaunchError(f"the {client} session{_scope(folder)} is {state}, and {client} "
                            "runs one web session at a time, because its web app has one port "
                            f"on the Mac.{then}", L.EXIT_TEMPFAIL)


def _overlap_line(client: str, project: str, plan) -> str | None:
    """A warning when the running session of any client shares a folder
    that holds or lies inside a folder this launch shares, since two
    virtual machines then change the same files."""
    mine = [m.source for m in plan.shares if m.kind == "share"]
    found = []
    for other_client in LAUNCH_CLIENTS:
        for other, record in session.records(other_client):
            if (other_client, other) != (client, project) and any(
                    settings._inside(s["host"], m) or settings._inside(m, s["host"])
                    for s in record["shares"] for m in mine):
                found.append((other_client, other, record))
    if not found:
        return None
    try:
        containers = cli.list_launch_containers()
    except ContainerError:
        return None                      # no session runs while the service is down
    names = [f"the running {c} session{_scope(r.get('project'))}" for c, o, r in found
             if session.record_runs(c, o, r, containers)]
    if not names:
        return None
    return (f"[launch] {_listed(names)} {'shares' if len(names) == 1 else 'share'} files with "
            "this session. File locks do not reach from one virtual machine to another, so "
            "do not let two clients change the same file at once.")


def _project_volumes(launch_cfg: LaunchCfg, client: str, project: str) -> list[str]:
    """The volume entries that get the project's own name: those listed
    under the client and not for every client. The default project keeps
    the configured names."""
    if project == settings.PROJECT_DEFAULT:
        return []
    box = launch_cfg.container
    own = box.clients[client].volumes if client in box.clients else []
    return [v for v in own if v not in box.volumes]


def _session_command(ready, cfg, captured) -> tuple[list[str] | None, list[str] | None]:
    """The client's command without the arguments after --, which a copy
    that joins the session runs with its own, and for command: image the
    ENTRYPOINT that such arguments follow in place of CMD. A shell session
    records the client's command too."""
    try:
        if ready is not None:
            command, _ = images.image_command(ready, cfg.command, captured["argv"], [])
        elif isinstance(cfg.command, list):
            command = list(cfg.command)
        elif cfg.command == "image":
            return None, None
        else:
            command = list(captured["argv"])
    except images.ImageError:
        return None, None
    if cfg.command == "image" and ready is not None:
        return command, list(ready.info.entrypoint or [])
    return command, None


def _join(a, cfg, project: str, folder: str | None, say) -> int:
    """Run another copy of the client, or a shell under --shell, in the
    running session of ``project``, whose folder is ``folder``, or open a
    web app that runs."""
    from gmlx.commands import launch as L

    client, scope = a.harness, _scope(folder)
    for dest, default in _JOIN_REFUSED.items():
        value = getattr(a, dest, default)
        if value != default:
            flag = _flag_name(dest, value)
            raise L.LaunchError(f"a {client} session is already running{scope}, so this "
                                f"launch joins it, and {flag} applies only to a new session.")
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
        return _web_again(client, cfg, record, say,
                          _unshared_line(record) if cwd is None and shares else None)
    copy_id = secrets.token_hex(8)
    entry = [runtime.GUEST_ENTRY, *(["--clipboard"] if record.get("clipboard") else []),
             "--join", "--copy-id", copy_id]
    if a.shell:
        say(f"[launch] opening a shell in the running {client} session{scope} ({name})")
        command = [*entry, "--shell", "--", *a.passthrough]
    else:
        base, entrypoint = record.get("command"), record.get("entrypoint")
        run = ([*entrypoint, *a.passthrough] if entrypoint is not None and a.passthrough
               else [*(base or []), *a.passthrough])
        if not run:
            raise L.LaunchError(f"the running {client} session{scope} does not record the "
                                "command it runs, so no copy can join it. Open a shell in it "
                                f"with: gmlx launch {client} --shell")
        say(f"[launch] joining the running {client} session{scope}")
        command = [*entry, "--", *run]
    if cwd is None and (shares or a.shell):
        what = "the shell opens" if a.shell else f"{client} starts"
        say(_unshared_line(record, f", so {what} in its working folder {record['workdir']}"))
    argv = cli.exec_argv(name, command, tty=session.stdin_is_tty(), cwd=cwd)
    return session.run_copy(argv, dict(os.environ), name=name, copy_id=copy_id)


def _unshared_line(record: dict, then: str = "") -> str:
    """The line for a launch that joins a session from a folder that none
    of its shares holds. It names the folders the session shares, and ends
    with ``then``."""
    shown = [settings._tilde(s["host"]) + (" (read-only)" if s.get("readonly") else "")
             for s in record["shares"]]
    which = f", which shares {_listed(shown)}" if shown else ""
    return f"[launch] the current folder is not shared with this session{which}{then}."


def _web_again(client: str, cfg, record: dict, say, unshared: str | None = None) -> int:
    """A second launch of a running web app says where it answers and opens
    it. dsh's address holds a login token, which the session records once
    dsh prints it. ``unshared`` is the line for a current folder that the
    session does not share. A session that runs a shell has no app to open
    until you start it there."""
    port = record.get("web_port")
    url = record.get("url") if client == "dsh" else f"http://127.0.0.1:{port}/"
    ready = bool(port and url and url.startswith(f"http://127.0.0.1:{port}/")
                 and url.isprintable() and not record.get("shell"))
    if record.get("shell"):
        where = "at the address it prints" if client == "dsh" else f"at {url}"
        say(f"[launch] the running {client} session runs a shell, so {client} answers only "
            f"after you start it in that shell, {where}. To open another shell in the "
            f"session, run: gmlx launch {client} --shell")
    elif ready:
        say(f"[launch] {client} is already running at {url}")
    else:
        say(f"[launch] {client} is already running, and its web app has not printed its "
            "address yet. The launch that started it opens the address once it is ready.")
    if unshared:
        say(unshared)
    if ready and url and cfg.open_browser is not False:
        webbrowser.open(url)
    return 0


def _remove_home(a, project: str, folder: str | None, say) -> int:
    """Remove the private home of this launch's project, and the records
    beside it, after a question on the terminal. --mount-cwd, --no-mount-cwd
    and --mount choose the project, so they can go with --remove-home."""
    import shlex
    import shutil

    from gmlx.commands import launch as L
    from gmlx.commands.doctor import _WALK_CAP, _folder_bytes

    client = a.harness
    others = [_flag_name(dest, getattr(a, dest, None)) for dest in CONTAINER_FLAGS
              if dest not in ("remove_home", "mount_cwd", "mount") and _flag_set(a, dest)]
    if others or a.passthrough or a.config_only:
        what = others[0] if others else "--config-only" if a.config_only else "arguments after --"
        raise L.LaunchError(f"--remove-home removes a home and starts nothing, so it cannot "
                            f"go with {what}.")
    where = f" for {settings._tilde(folder)}" if folder else " for the default project"
    target = settings.project_dir_path(client, project)
    home = target / "home"
    if not home.is_dir() or home.is_symlink():
        say(f"[launch] {client} has no private home{where}, so nothing was removed.")
        return 0
    lock = session.try_session_lock(client, project)
    if lock is None:
        raise L.LaunchError(f"the {client} session{where} is running. End it, then remove "
                            "its home.", L.EXIT_TEMPFAIL)
    try:
        if not session.stdin_is_terminal():
            raise L.LaunchError(f"--remove-home asks before it removes anything, and there is "
                                f"no terminal to ask on. Remove the home yourself with: rm -rf "
                                f"{shlex.quote(str(target))}")
        budget = [_WALK_CAP]
        size = session.gb(_folder_bytes(home, budget))
        more = "at least " if budget[0] <= 0 else ""
        try:
            answer = input(f"[launch] remove the private home of {client}{where}, {more}"
                           f"{size} at {settings._tilde(str(home))}, with its settings and "
                           "history? [y/N] ")
        except EOFError:                  # Ctrl-D answers no
            print()
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            say("[launch] nothing was removed.")
            return 1
        # The guest can put links in the home, so no link is followed.
        with confine.confined(target):
            confine.remove_tree(home)
        shutil.rmtree(target, ignore_errors=True)
        say(f"[launch] removed {settings._tilde(str(target))}")
        return 0
    finally:
        lock.release()


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
        project, folder = _session_key(a, cfg)
        if getattr(a, "remove_home", False):
            return _remove_home(a, project, folder, say)
        # Step 3. A stopped service starts only after the refusals of step 6.
        prereqs = _Prereqs()
        # Step 4
        lock = session.try_session_lock(client, project)

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
        if folder and (lock is not None or not session.record_path(client, project).exists()):
            try:
                enclosing = _enclosing_session(client, project, folder)
            except L.LaunchError:
                let_go()
                raise
        if enclosing is not None:
            let_go()
            other, record = enclosing
            return _join(a, cfg, other, record.get("project"), say)
        if lock is None:                  # joining refuses --config-only itself
            return _join(a, cfg, project, folder, say)
        if _is_web(a) and not dry:
            try:
                _one_web_session(client, project)
            except L.LaunchError:
                let_go()
                raise
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
    web_port = L.web_port_for(client, port) if web else None
    # A read-write share of any client's build: folder would let this
    # client change what that image runs.
    builds = {c: launch_cfg.container.for_client(c).build for c in LAUNCH_CLIENTS}
    plan = settings.resolve_plan(client, cfg, cwd=_cwd(), mount_cwd=a.mount_cwd,
                                 cli_mounts=a.mount, network=a.network, api_port=api_port,
                                 web_port=web_port,
                                 build_folders={c: b for c, b in builds.items() if b},
                                 project=project,
                                 project_volumes=_project_volumes(launch_cfg, client, project))
    settings.check_program(prereqs.binary, [m.source for m in plan.shares if not m.readonly])
    if not dry:
        # A launch from a folder this session will share waits for it,
        # instead of starting a second virtual machine on the same files.
        session.write_record(client, project, {
            "name": "", "workdir": plan.workdir, "starting": True, "pid": os.getpid(),
            "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                       for m in plan.shares],
            "project": folder, "web": web, "web_port": web_port})
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
                                      image_override=a.image, writable=writable)
    config_notes: list[str] = []
    config_path = settings.server_config_path(host, port,
                                              autostart=not (a.base_url or a.no_start),
                                              notes=config_notes)
    started = check.start
    if started:
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
                                   writable=settings.seed_writable(plan, _cwd())):
        say(line)
    settings.ready_home(client, plan.home)
    settings.write_project_record(client, project, folder)
    if reseed and not plan.seed:
        say(f"[launch] --reseed has nothing to copy, because no seed is configured for "
            f"{client}.")
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
            images.forget_unnamed(launch_cfg.container, say)
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
    # Step 9. A server that step 6 started is ready.
    if not started:
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
                                       [web_port] if web_port is not None else [],
                                       project)
    full_api = None if server_session else full_api_line(base, client, a.api_key)
    if server_session and open_bind(base, api_targets) and not L._auth_required(base):
        full_api = open_bind_line(base)
    # Step 10
    if a.model and not a.no_keep and not dry:
        L._pick_default(L.probe_models(base, a.api_key, client), a.model)
        L._keep_model(a)
    # Step 11. The last step line prints before the client's own lines.
    if steps:
        say(f"[launch] step {steps} of {steps}: starting "
            f"{'a shell' if a.shell else client}")
    captured: dict = {}

    def sink(argv, pairs, extra):
        captured.update(argv=argv, pairs=pairs, extra=extra)
        return 0
    a.container_sink = sink
    a.container_context_tokens = _env_entry_value(plan.env, L.CONTEXT_TOKENS)
    # Under --shell the client does not start, so its summary and notes
    # would describe a program that is not running.
    quiet = contextlib.redirect_stdout(io.StringIO()) if a.shell else contextlib.nullcontext()
    # The handler runs with HOME in the private home, which the guest
    # writes, so the facts it needs from gmlx's state on the Mac are read
    # here.
    a.served_config = L._served_config(a.host, a.port)
    a.no_models_text = L.no_models_message(L._server_root(base))
    with guest_home(plan.home), quiet:
        rc = L._HARNESSES[client](a, exec_fn=exec_fn)
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
                  **_client_env(client, plan, ready, cfg.command, web_port)}
    pair_names = [n for n in captured["pairs"] if n not in env_values]
    env_names = list(dict.fromkeys([*pair_names, *(n for n in names if n not in env_values)]))
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
    command_base, entrypoint = _session_command(ready, cfg, captured)
    record = {"name": sess.name, "workdir": spec.workdir, "clipboard": plan.clipboard == "images",
              "shares": [{"host": m.source, "guest": m.target, "readonly": m.readonly}
                         for m in plan.shares],
              "command": command_base, "entrypoint": entrypoint, "project": folder,
              "web": web, "web_port": web_port, "shell": bool(a.shell),
              "profile": (a.dsh_profile or L._DSH_PROFILE) if client == "dsh" else None}
    # Under --shell the app is not running yet, so there is nothing to open.
    opener = webbrowser.open if (web_port and plan.open_browser and not a.shell) else None
    cli.end_memo()

    def started() -> None:
        notices.record(shown)
        session.mark_started(client, project)
    return session.supervise(spec, api_targets=api_targets, record=record, say=say,
                             opener=opener, summary=summary, server_session=server_session,
                             on_start=started)


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
        if m.kind == "git":
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
        who = "the shell opens" if shell else f"{client} starts"
        lines.append(f"[launch] the current folder is not shared, so {who} in {where}")
    if shell:
        lines.append(f"[launch] opening a shell instead of {client}")
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
    if isinstance(cfg.command, list) or cfg.command == "image":
        say("[launch] the command: setting replaces the client's own command, "
            f"{shlex.join(captured['argv'])}")
    argv = session.compose_run_argv(spec)
    say("[launch] the command, which needs the API socket that only a running launch "
        "provides:")
    print(printable(shlex.join(argv)))
    return 0
