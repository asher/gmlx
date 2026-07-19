"""Background-server process layer for ``gmlx serve``.

Primitives only - no model code, no server imports - so this stays light and unit
testable. The CLI glue (argument parsing, the ``serve --background`` divert) lives in
``server.py``; the menu bar (``menubar.py``) calls these same primitives.

A managed server is tracked by a **runfile** under the cache dir, keyed by
``<host>-<port>`` so distinct binds never collide. The runfile records the pid (and
process-group), the exact relaunch ``argv``, the resolved config path, and how it's
managed (a detached child of ours, or a launchd agent). Stop/restart verify the live
pid's identity (cmdline) before signalling, so a recycled PID is never mistaken for the
server, and signal the whole process **group** so uvicorn workers don't orphan.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .textfmt import plural_s
from . import procname

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is POSIX-only
    fcntl = None


# State dir + runfile / log paths (keyed by host+port)

def runtime_dir() -> Path:
    """Cache dir for runfiles + logs (matches the `chat` history convention)."""
    d = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "gmlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(host: str, port) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{host}-{port}").strip("-")


def run_path(host: str, port) -> Path:
    return runtime_dir() / f"run-{_key(host, port)}.json"


def _run_lock(host: str, port) -> Path:
    return runtime_dir() / f"run-{_key(host, port)}.lock"


@contextlib.contextmanager
def _spawn_guard_lock(host: str, port, on_wait=None):
    """Serialize the check-then-spawn-then-write critical section in
    :func:`_spawn_detached` so two concurrent serves on the same bind can't both
    pass the refuse-guard and double-spawn (the loser dies on the port bind, but
    its parent may still overwrite the runfile with the dead pid). flock is
    advisory and kernel-released on holder death - no stale-lock hazard - and a
    no-op where ``fcntl`` is unavailable. ``on_wait`` fires once if the lock is
    contended (stop() can hold it through its kill wait, up to ~20s) so the
    caller can say why it stalled instead of blocking silently."""
    if fcntl is None:
        yield
        return
    with open(_run_lock(host, port), "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait is not None:
                on_wait()
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def log_path(host: str, port) -> Path:
    return runtime_dir() / f"server-{_key(host, port)}.log"


def _coerce_run(run):
    """A hand-edited runfile must degrade to stale, never crash ``os.kill``:
    pid becomes int-or-None, non-dict documents become None."""
    if not isinstance(run, dict):
        return None
    try:
        run["pid"] = int(run.get("pid"))
    except (TypeError, ValueError):
        run["pid"] = None
    return run


def read_run(host: str, port) -> dict | None:
    try:
        return _coerce_run(json.loads(run_path(host, port).read_text()))
    except (OSError, ValueError):
        return None


def write_run(host: str, port, data: dict) -> None:
    p = run_path(host, port)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)


def _remove_run(host: str, port) -> None:
    """Drop the runfile. The spawn-guard lock file is deliberately left behind:
    unlinking it while a `serve` holds flock on that inode lets the next `serve`
    create a fresh lock file and enter the check->spawn->write window
    concurrently. It is an empty file, and flock is released on holder death."""
    try:
        run_path(host, port).unlink()
    except OSError:
        pass


def _remove_run_if_pid(host: str, port, pid: int | None) -> None:
    """Remove the runfile only while it still names ``pid``. A `serve` that won
    the spawn guard while we were killing the old process has already written
    its own runfile; deleting that would orphan a live server from
    status/stop/logs. Call under :func:`_spawn_guard_lock`."""
    run = read_run(host, port)
    if run is None or (pid is not None and run.get("pid") != pid):
        return
    _remove_run(host, port)


def list_runs() -> list:
    """All current runfiles (parsed dicts), skipping unreadable ones."""
    out = []
    for p in sorted(runtime_dir().glob("run-*.json")):
        try:
            run = _coerce_run(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
        if run is not None:
            out.append(run)
    return out


def _config_target() -> tuple | None:
    """(host, port) from the default-location config, when one loads cleanly.
    Best-effort by design: the lifecycle verbs must keep working against a
    corrupt or absent config, so any problem here just means no opinion."""
    try:
        from .config import default_config_paths, load_config
        for p in default_config_paths():
            if p.exists():
                cfg = load_config(p)
                return cfg.host or "127.0.0.1", int(cfg.port or 8080)
    except Exception:
        return None
    return None


def auto_target(host: str | None, port: int | None) -> tuple:
    """Resolve (host, port) for stop/status/logs/restart/ps/launch. Explicit
    flags win; else the single managed server; else the default config's
    host/port; else 127.0.0.1:8080. Every consumer of a bare lifecycle verb
    shares this resolution so they all talk about the same server."""
    if host is None and port is None:
        runs = list_runs()
        if len(runs) == 1:
            r = runs[0]
            return r.get("host") or "127.0.0.1", int(r.get("port") or 8080)
        cfg_target = _config_target()
        if cfg_target:
            return cfg_target
    return host or "127.0.0.1", int(port or 8080)


# Process identity (guards against a recycled PID - B1)

def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, TypeError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _proc_cmdline(pid: int) -> str:
    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def identity_ok(run: dict | None) -> bool:
    """True only if the runfile's pid is alive and looks like *our* server - the
    cmdline mentions gmlx and the bound port. A dead or mismatched pid (PID reuse,
    stale runfile) returns False so callers never signal an unrelated process."""
    if not run:
        return False
    pid = run.get("pid")
    if not pid_alive(pid):
        return False
    cmd = _proc_cmdline(int(pid))
    port = str(run.get("port") or "")
    looks_like_ours = "gmlx" in cmd
    return looks_like_ours and (port in cmd if port else True)


# Child invocation (absolute interpreter so launchd's minimal PATH still works - B4).
# On macOS the daemon runs through the `gmlx`-named stub copy (procname.py) so
# ps / Activity Monitor don't show it as "Python"; spawn sites pass
# procname.child_env() so the renamed stub still resolves this venv.

def child_argv(serve_args: list) -> list:
    exe = procname.named_python() or os.path.abspath(sys.executable)
    return [exe, "-m", "gmlx", "serve", *serve_args]


def _agent_path() -> str:
    """A PATH for a launchd agent: the running interpreter's bin dir + system dirs."""
    venv_bin = os.path.dirname(os.path.abspath(sys.executable))
    return ":".join([venv_bin, "/usr/local/bin", "/usr/bin", "/bin",
                     "/usr/sbin", "/sbin"])


# Log handling

def rotate_log(path) -> None:
    """Rename a non-empty existing log to ``<path>.1`` (keep one prior run)."""
    p = Path(path)
    try:
        if p.exists() and p.stat().st_size > 0:
            p.replace(Path(str(p) + ".1"))
    except OSError:
        pass


def rotate_log_if_large(path, max_bytes: int = 10 * 1024 * 1024) -> None:
    """Size-capped variant for append-mode logs (launchd agents, the menu
    bar): rotate to ``<path>.1`` only past ``max_bytes``, so the file never
    grows unbounded but a small log keeps its history across restarts."""
    p = Path(path)
    try:
        if p.exists() and p.stat().st_size > max_bytes:
            p.replace(Path(str(p) + ".1"))
    except OSError:
        pass


def _log_tail(path, n: int = 40) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def human_gb(n_bytes: int, decimals: int = 1) -> str:
    """Binary-GB size string (``16.8 GB``), degrading to MB/KB/B below each
    0.1 threshold. The one user-facing byte formatter (manage/menubar/launch),
    so GB never silently flips between 1e9 and 1024**3 across surfaces."""
    gb = n_bytes / 1024**3
    if gb >= 0.1:
        return f"{gb:.{decimals}f} GB"
    mb = n_bytes / 1024**2
    if mb >= 0.1:
        return f"{mb:.0f} MB"
    kb = n_bytes / 1024
    return f"{kb:.0f} KB" if kb >= 0.1 else f"{int(n_bytes)} B"


# Local-server HTTP-JSON helpers (shared by menubar/launch and the probes below)

def server_root(base_url: str) -> str:
    """The server root for ``/health`` (strip a trailing ``/v1``)."""
    b = base_url.rstrip("/")
    return b[:-3].rstrip("/") if b.endswith("/v1") else b


def get_json(url: str, *, api_key: str | None = None, timeout: float = 5.0,
             headers: dict | None = None):
    """GET ``url`` and parse JSON (Bearer auth via ``api_key``, or raw ``headers``)."""
    hdrs = dict(headers or {})
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
        return json.loads(r.read())


def post_json(url: str, payload: dict | None = None, *,
              api_key: str | None = None, timeout: float = 5.0):
    """POST ``payload`` as JSON and parse the reply (``{}`` on an empty body)."""
    data = json.dumps(payload or {}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local server)
        body = r.read()
    return json.loads(body) if body else {}


# Readiness / health probes

def _health_ok(host: str, port, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health",
                                    timeout=timeout) as r:           # noqa: S310
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _health_pid(host: str, port, timeout: float = 1.5) -> int | None:
    """The pid the /health body reports, or None (down, or a pre-pid server)."""
    try:
        body = get_json(f"http://{host}:{port}/health", timeout=timeout)
        return int(body["pid"]) if isinstance(body, dict) and body.get("pid") else None
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        return None


def _ready(host: str, port, api_key: str | None = None,
           expect_pid: int | None = None) -> bool:
    """Up *and* serving: /health responds and /v1/models answers with the OpenAI
    list shape (a 401 also proves the app is up + auth-gating). Gating on
    /v1/models - not bare /health - avoids declaring a foreign or half-bound
    server ready (B3). The ``data`` list may legitimately be empty (a config
    with no models yet); requiring it non-empty made those boots stall the full
    start timeout and report "may still be loading" about a healthy server.

    ``expect_pid`` pins readiness to the process we spawned: a foreign server
    already holding the port answers the probes with its own pid, and without
    the pin our child dies on bind while serve reports its pid as up."""
    if not _health_ok(host, port):
        return False
    if expect_pid is not None and _health_pid(host, port) != int(expect_pid):
        return False
    try:
        payload = get_json(f"http://{host}:{port}/v1/models",
                           api_key=api_key, timeout=1.5)
        return isinstance(payload, dict) and isinstance(payload.get("data"), list)
    except urllib.error.HTTPError as e:
        return e.code == 401
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Start / stop / restart / status / logs

def _spawn_detached(child: list, *, host: str, port: int,
                    config_abspath: str | None = None, log=None,
                    api_key: str | None = None, api_key_set: bool = False):
    """Spawn ``child`` as a detached background server and write its ``starting``
    runfile. Returns ``(proc, log_path)``, or ``None`` if an identity-OK server is
    already healthy at this bind (nothing spawned). The caller owns the readiness
    wait - :func:`launch_detached` blocks on it; ``launch`` polls with a spinner."""
    host = host or "127.0.0.1"
    port = int(port or 8080)
    # Hold the lock across the whole check->spawn->write window: a concurrent serve
    # blocks here, then re-reads the runfile we just wrote and refuses below.
    with _spawn_guard_lock(host, port, on_wait=lambda: print(
            f"waiting for a concurrent gmlx stop/serve on {host}:{port} "
            "to finish ...", file=sys.stderr)):
        existing = read_run(host, port)
        # Refuse on identity alone (a live, ours, correct-port process), not on
        # health: a server still in its preload window legitimately holds this
        # bind. Replacing a hung-but-live server is `gmlx restart` / `stop`.
        if existing and existing.get("managed_by") != "launchd" \
                and identity_ok(existing):
            print(f"a server already holds http://{host}:{port} "
                  f"(pid {existing.get('pid')}) - `gmlx status`, or "
                  f"`gmlx restart` / `gmlx stop` to replace it",
                  file=sys.stderr)
            return None

        lp = Path(os.path.expanduser(log)) if log else log_path(host, port)
        lp.parent.mkdir(parents=True, exist_ok=True)
        rotate_log(lp)
        logf = open(lp, "w")
        try:
            proc = subprocess.Popen(child, stdout=logf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    env=procname.child_env())
        finally:
            logf.close()                      # the child holds its own dup of the fd

        run = {
            "pid": proc.pid, "pgid": proc.pid,  # start_new_session => group leader
            "host": host, "port": port, "url": f"http://{host}:{port}",
            "config_abspath": config_abspath, "argv": list(child), "log": str(lp),
            "started_at": time.time(), "managed_by": "detach",
            "api_key_set": bool(api_key_set or api_key), "status": "starting",
        }
        write_run(host, port, run)
    return proc, lp


def report_port_in_use(tail: str, host: str, port: int, tag: str = "error:") -> bool:
    """When a dead child's log ``tail`` shows a bind failure, print a targeted
    headline plus next commands and return True (callers then skip the raw tail)."""
    if "address already in use" not in tail.lower():
        return False
    print(f"{tag} port {port} on {host} is already in use - another process "
          f"is listening there", file=sys.stderr)
    print(f"  if it's a gmlx server: gmlx status --port {port} / "
          f"gmlx stop --port {port}", file=sys.stderr)
    print("  otherwise pick another port: gmlx serve --port <N>", file=sys.stderr)
    return True


def _warn_missing_models(host, port, api_key, config_abspath) -> None:
    """After a ready background start, compare the config's ``models:`` ids with
    what ``/v1/models`` actually serves and warn when entries were skipped
    (file missing, model_dirs root gone). The child logs each skip to the log
    file only; without this, the parent's "server up" reads as all-good even
    when 0 of N models loaded. Best-effort: any failure stays silent."""
    if not config_abspath:
        return
    try:
        import yaml
        doc = yaml.safe_load(Path(config_abspath).read_text()) or {}
        configured = set((doc.get("models") or {}).keys())
        if not configured:
            return
        payload = get_json(f"http://{host}:{port}/v1/models",
                           api_key=api_key, timeout=3)
        data = payload.get("data") or []
        served = {m.get("id") for m in data if not m.get("alias_of")}
        n = len(configured & served)
        if n < len(configured):
            print(f"  note: {n} of {len(configured)} configured "
                  f"model{plural_s(len(configured))} "
                  f"available - see `gmlx logs` for what was skipped",
                  file=sys.stderr)
    except Exception:
        pass


def launch_detached(child: list, *, host: str, port: int,
                    config_abspath: str | None = None, log=None,
                    start_timeout: float = 40.0, api_key: str | None = None,
                    api_key_set: bool = False) -> int:
    """Spawn ``child`` as a detached background server, wait for readiness, and record
    a runfile. Returns 0 on ready, non-zero on early child death."""
    host = host or "127.0.0.1"
    port = int(port or 8080)
    spawned = _spawn_detached(child, host=host, port=port,
                              config_abspath=config_abspath, log=log,
                              api_key=api_key, api_key_set=api_key_set)
    if spawned is None:
        return 1
    proc, lp = spawned
    run = read_run(host, port) or {}

    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:           # child died - surface the log tail (B3)
            tail = _log_tail(lp, 40).rstrip()
            _remove_run_if_pid(host, port, proc.pid)
            if report_port_in_use(tail, host, port):
                return 1
            print(f"error: server exited (code {proc.returncode}) before it was ready",
                  file=sys.stderr)
            if tail and tail != "(no log)":
                print(tail, file=sys.stderr)
            return 1
        if _ready(host, port, api_key, expect_pid=proc.pid):
            run["status"] = "running"
            write_run(host, port, run)
            print(f"server up at http://{host}:{port}  (pid {proc.pid})")
            print(f"  try:  gmlx launch <harness>   or   "
                  f"curl http://{host}:{port}/v1/models")
            print(f"  logs: {lp}")
            tgt = "" if (host, port) == ("127.0.0.1", 8080) else f" --port {port}"
            print(f"  stop: gmlx stop{tgt}   status: gmlx status{tgt}")
            _warn_missing_models(host, port, api_key, config_abspath)
            return 0
        time.sleep(0.4)

    print(f"server starting (pid {proc.pid}) but not ready after "
          f"{start_timeout:.0f}s - it may still be loading; check: gmlx logs",
          file=sys.stderr)
    return 0


def start_background(serve_args: list, *, host: str, port: int,
                     config_abspath: str | None = None, log=None,
                     start_timeout: float = 40.0,
                     api_key: str | None = None) -> int:
    """Build the child argv (with host/port baked in) and launch it detached. The
    child runs ``--foreground`` so it serves in place rather than re-detaching."""
    host = host or "127.0.0.1"
    port = int(port or 8080)
    child = child_argv([*serve_args, "--host", host, "--port", str(port),
                        "--foreground"])
    return launch_detached(child, host=host, port=port,
                           config_abspath=config_abspath, log=log,
                           start_timeout=start_timeout, api_key=api_key)


def start_background_nowait(serve_args: list, *, host: str, port: int,
                            config_abspath: str | None = None, log=None,
                            api_key: str | None = None):
    """Like :func:`start_background`, but return ``(proc, log_path)`` (or ``None`` if a
    server already holds the bind) the instant the child is spawned - no readiness
    wait. The caller polls for readiness itself (``launch`` does, with a spinner)."""
    host = host or "127.0.0.1"
    port = int(port or 8080)
    child = child_argv([*serve_args, "--host", host, "--port", str(port),
                        "--foreground"])
    return _spawn_detached(child, host=host, port=port,
                           config_abspath=config_abspath, log=log, api_key=api_key)


# Menu-bar companion (macOS) - one machine-wide monitor, raised alongside a background
# server and tracked by a single pidfile so a second `serve` (on any port) or a manual
# `launch menubar` is a no-op. It tracks the primary server (auto_target), not whichever
# server happened to spawn it, so multiple servers never accumulate multiple menu bars.

def gui_session_available() -> bool:
    """Heuristic for 'a macOS Aqua session the menu bar can attach to': macOS, and not
    an SSH login (which has no GUI session). Best-effort - the worst case is a menu-bar
    child that exits at once, which self-heals its pidfile."""
    if sys.platform != "darwin":
        return False
    return not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def menubar_run_path() -> Path:
    """The single machine-wide menu-bar pidfile (not per server)."""
    return runtime_dir() / "menubar.json"


def menubar_log_path() -> Path:
    """The menu-bar process's stdout/stderr log (one machine-wide file)."""
    return runtime_dir() / "menubar.log"


def menubar_alive() -> bool:
    """True if the one menu-bar process is recorded and still alive + ours."""
    try:
        run = json.loads(menubar_run_path().read_text())
    except (OSError, ValueError):
        return False
    pid = run.get("pid") if isinstance(run, dict) else None
    if not pid_alive(pid):
        return False
    cmd = _proc_cmdline(int(pid))
    return "gmlx" in cmd and "menubar" in cmd


def write_menubar_run(pid: int, *, auto: bool = False) -> None:
    """Record the menu-bar pid. ``auto`` marks a bar raised as a side effect of
    serve/launch (vs an explicit `launch menubar`); an auto bar is taken down
    again when the last managed server is stopped."""
    menubar_run_path().write_text(
        json.dumps({"pid": pid, "started_at": time.time(), "auto": bool(auto)}))


def menubar_is_auto() -> bool:
    """True when the recorded menu bar was auto-raised by serve/launch."""
    try:
        run = json.loads(menubar_run_path().read_text())
    except (OSError, ValueError):
        return False
    return bool(isinstance(run, dict) and run.get("auto"))


def remove_menubar_run() -> None:
    try:
        menubar_run_path().unlink()
    except OSError:
        pass


def start_menubar(*, extra: list | None = None, auto: bool = False) -> int:
    """Spawn the single machine-wide menu-bar monitor detached, recording a pidfile so a
    later call - from a second `serve` on any port, or a manual `launch menubar` - is a
    no-op. By default it tracks the primary server (``auto_target``); ``extra`` may pin
    an explicit ``--url``/``--host``/``--port``. Idempotent and best-effort: returns 0
    whether it spawned or found one already running, and never raises into the serve
    path. The GUI loop child resolves the api-key from the server's own config."""
    if menubar_alive():
        return 0
    # The gmlx.app-bundled stub makes the notification-permission prompt (and
    # ps / Activity Monitor) read "gmlx" instead of "Python".
    exe = procname.menubar_bundle() or os.path.abspath(sys.executable)
    argv = [exe, "-m", "gmlx", "launch", "menubar",
            "--foreground", *(["--auto-raised"] if auto else []), *(extra or [])]
    # stdout/stderr to a log (crashed GUI threads used to vanish into DEVNULL);
    # "Open logs" in the bar surfaces it next to the server log.
    lp = menubar_log_path()
    rotate_log_if_large(lp, max_bytes=1_000_000)
    env = procname.child_env()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        log_f = open(lp, "a")
    except OSError:
        log_f = subprocess.DEVNULL
    try:
        proc = subprocess.Popen(argv, stdout=log_f, stderr=log_f,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
    except OSError:
        return 1
    finally:
        if log_f is not subprocess.DEVNULL:
            log_f.close()
    write_menubar_run(proc.pid, auto=auto)
    return 0


def stop_menubar() -> bool:
    """Terminate the machine-wide menu-bar monitor if it is ours and alive;
    always clears the pidfile. True when a live process was signalled."""
    alive = menubar_alive()
    if alive:
        try:
            run = json.loads(menubar_run_path().read_text())
            os.kill(int(run["pid"]), signal.SIGTERM)
        except (OSError, ValueError, KeyError, TypeError):
            alive = False
    remove_menubar_run()
    return alive


def _maybe_stop_auto_menubar() -> None:
    """After a stop leaves no managed servers, take an auto-raised menu bar down
    with them; a manually launched bar stays up (its owner asked for it)."""
    if list_runs() or not menubar_is_auto() or not menubar_alive():
        return
    if stop_menubar():
        print("stopped the auto-raised menu bar (no servers left)")


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    return not pid_alive(pid)


def stop(host: str, port, *, timeout: float = 15.0) -> int:
    # Hold the spawn guard across read->kill->remove: a `serve` that started
    # during the kill window would otherwise have its fresh runfile deleted
    # below, leaving a live server invisible to status/stop/restart.
    with _spawn_guard_lock(host, port, on_wait=lambda: print(
            f"waiting for a concurrent gmlx stop/serve on {host}:{port} "
            "to finish ...", file=sys.stderr)):
        rc = _stop_locked(host, port, timeout)
    if rc == 0:
        _maybe_stop_auto_menubar()
    return rc


def _stop_locked(host: str, port, timeout: float) -> int:
    run = read_run(host, port)
    if run is None:
        print(f"no managed server at http://{host}:{port} - start one with "
              f"`gmlx serve`", file=sys.stderr)
        return 1
    if run.get("managed_by") == "launchd":
        print("this server is managed by launchd - stop it with "
              "`gmlx service uninstall`", file=sys.stderr)
        return 1
    if not identity_ok(run):
        _remove_run(host, port)
        print(f"server for {host}:{port} is not running (removed a stale runfile)")
        return 0
    pgid = int(run.get("pgid") or run["pid"])
    pid = int(run["pid"])
    try:
        os.killpg(pgid, signal.SIGTERM)       # whole group, never SIGHUP (B2)
    except (ProcessLookupError, PermissionError):
        pass
    if not _wait_gone(pid, timeout):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        _wait_gone(pid, 5.0)
    _remove_run_if_pid(host, port, pid)
    print(f"stopped server {host}:{port}  (pid {pid})")
    return 0


def restart(host: str, port, *, timeout: float = 15.0,
            start_timeout: float = 40.0) -> int:
    run = read_run(host, port)
    if run is None:
        print(f"no managed server at http://{host}:{port} - start one with "
              f"`gmlx serve`", file=sys.stderr)
        return 1
    if run.get("managed_by") == "launchd":
        label = run.get("label", _label(host, port))
        print("this server is managed by launchd - restart it with: "
              f"launchctl kickstart -k gui/{os.getuid()}/{label}", file=sys.stderr)
        return 1
    argv = run.get("argv")
    if not argv:
        print(f"runfile for {host}:{port} has no argv to relaunch", file=sys.stderr)
        return 1
    config_abspath = run.get("config_abspath")
    # stop() tears down an auto-raised menu bar when this was the last server;
    # remember to re-raise it once the relaunch succeeds.
    was_auto_bar = menubar_alive() and menubar_is_auto()
    stop(host, port, timeout=timeout)
    rc = launch_detached(list(argv), host=host, port=port,
                         config_abspath=config_abspath,
                         start_timeout=start_timeout,
                         api_key_set=bool(run.get("api_key_set")))
    if rc == 0 and was_auto_bar and gui_session_available():
        start_menubar(auto=True)
    return rc


def reload_config(config_abspath: str) -> list:
    """SIGHUP every running server started from ``config_abspath`` so it re-reads the
    (just-edited) config and re-registers its model set - no restart, residency kept.

    Matching on the recorded config path is the safety gate: only servers launched with
    ``--config`` carry a ``config_abspath`` *and* install the SIGHUP reload handler, so
    we never signal a single-model server (no handler) that would die on the default
    SIGHUP disposition. ``identity_ok`` further guards against a recycled pid. Returns
    ``[(host, port, pid), ...]`` for each server signalled (best-effort)."""
    if not hasattr(signal, "SIGHUP"):
        return []
    target = os.path.realpath(os.path.expanduser(config_abspath))
    signalled = []
    for run in list_runs():
        ca = run.get("config_abspath")
        if not ca or os.path.realpath(os.path.expanduser(ca)) != target:
            continue
        if not identity_ok(run):
            continue
        pid = int(run["pid"])
        try:
            os.kill(pid, signal.SIGHUP)
        except (ProcessLookupError, PermissionError):
            continue
        signalled.append((run.get("host"), run.get("port"), pid))
    return signalled


def _human_dur(s) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def status_info(host: str, port) -> dict | None:
    """The status dict for one managed server, or None without a runfile."""
    run = read_run(host, port)
    if run is None:
        return None
    managed = run.get("managed_by", "detach")
    alive = identity_ok(run) if managed != "launchd" else pid_alive(run.get("pid"))
    healthy = _health_ok(run.get("host", host), run.get("port", port))
    # An HTTP responder alone does not prove our server is up: a foreign process
    # can hold the port while the runfile's pid is long dead. For detach-managed
    # runs, liveness is the process identity; launchd runs may lack a usable pid,
    # so the probe still counts there.
    running = bool(alive or healthy) if managed == "launchd" else alive
    return {
        "running": running, "healthy": healthy,
        "pid": run.get("pid"), "host": run.get("host", host),
        "port": run.get("port", port), "url": run.get("url"),
        "managed_by": managed, "log": run.get("log"),
        "uptime_s": int(time.time() - run.get("started_at", time.time())),
        "api_key_set": bool(run.get("api_key_set")),
    }


def status(host: str, port, *, as_json: bool = False) -> int:
    """Process-layer status - needs no API key (uses the runfile + auth-exempt
    /health). Resident-model detail lives in `ps` (which handles the key)."""
    info = status_info(host, port)
    if info is None:
        # No readable runfile - but the port may still be answering (a garbage
        # runfile must not claim "nothing running" while a server serves).
        answering = _health_ok(host, port)
        if as_json:
            print(json.dumps({"running": False, "host": host, "port": port,
                              "answering": answering}))
        else:
            msg = f"no managed server at http://{host}:{port}"
            if answering:
                msg += (f"; note: a process IS answering on "
                        f"http://{host}:{port} (unreadable or missing runfile)")
            else:
                msg += " - start one with `gmlx serve`"
            print(msg)
        return 3
    managed, healthy = info["managed_by"], info["healthy"]
    if as_json:
        print(json.dumps(info, indent=2))
        return 0 if info["running"] else 3
    if not info["running"]:
        msg = (f"server {host}:{port}: not running (stale runfile - `gmlx stop` "
               f"to clear)")
        if healthy:
            msg += (f"; note: a different process is answering on "
                    f"http://{info['host']}:{info['port']}")
        print(msg)
        return 3
    where = f"up {_human_dur(info['uptime_s'])}" if managed == "detach" else "via launchd"
    health = "healthy" if healthy else "starting/unhealthy"
    pid = f"pid {info['pid']}" if info["pid"] else "launchd-managed"
    url = info["url"] or f"http://{info['host']}:{info['port']}"
    print(f"{url}: up ({health}) - {pid}, {where}")
    if info["log"]:
        print(f"  logs: {info['log']}  (gmlx logs)")
    if info["api_key_set"]:
        print("  auth: api-key - `gmlx ps --api-key KEY` for resident models")
    return 0


def tail_log(host: str, port, *, n: int = 40, follow: bool = False,
             clear: bool = False) -> int:
    run = read_run(host, port)
    lp = Path(run["log"]) if run and run.get("log") else log_path(host, port)
    if clear:
        try:
            with open(lp, "r+") as f:                  # truncate, never unlink (B/log)
                f.truncate(0)
        except OSError:
            try:
                open(lp, "w").close()
            except OSError:
                pass
        print(f"cleared {lp}")
        return 0
    if not lp.exists():
        print(f"no log at {lp}", file=sys.stderr)
        return 1
    with open(lp) as f:
        lines = f.readlines()
    sys.stdout.write("".join(lines[-n:]))
    if follow:
        _follow(lp)
    return 0


def _follow(path) -> None:
    try:
        with open(path) as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        pass


# launchd LaunchAgent (macOS only)

def _require_macos(what: str) -> int:
    if sys.platform != "darwin":
        print(f"error: {what} uses launchd, which is macOS-only", file=sys.stderr)
        return 2
    return 0


def _label(host: str, port) -> str:
    return f"com.gmlx.server.{_key(host, port)}"


def _plist_path(host: str, port) -> Path:
    return (Path("~/Library/LaunchAgents").expanduser()
            / f"{_label(host, port)}.plist")


MENUBAR_AGENT_LABEL = "com.gmlx.menubar"


def _menubar_agent_plist_path() -> Path:
    return (Path("~/Library/LaunchAgents").expanduser()
            / f"{MENUBAR_AGENT_LABEL}.plist")


def boot_time() -> str:
    """kern.boottime seconds as a string, '' when unavailable. The login
    stamp for run-once-per-boot autostart (see menubar._autostart_server_once):
    a menu bar respawned by KeepAlive mid-session must not re-run autostart
    and resurrect a server the user deliberately stopped."""
    try:
        r = subprocess.run(["sysctl", "-n", "kern.boottime"],
                           capture_output=True, text=True, timeout=5)
        m = re.search(r"sec\s*=\s*(\d+)", r.stdout or "")
        return m.group(1) if m else ""
    except Exception:
        return ""


def autostart_stamp_path() -> Path:
    return runtime_dir() / "menubar-autostart.stamp"


def agent_loaded(label: str) -> bool:
    """True when a LaunchAgent ``label`` is loaded in the gui domain."""
    if sys.platform != "darwin":
        return False
    r = subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
        capture_output=True)
    return r.returncode == 0


def menubar_agent_loaded() -> bool:
    """True when the com.gmlx.menubar LaunchAgent is loaded - killing
    that bar's pid just makes KeepAlive respawn it; it is quit from its own
    menu (clean exit) or removed with `service uninstall`."""
    return agent_loaded(MENUBAR_AGENT_LABEL)


def _agent_entry() -> list:
    """The plist ProgramArguments prefix: the gmlx.app trampoline (Login
    Items then attribute the agent to gmlx, not "Python"), falling back to
    the venv interpreter. Either way the entry point re-execs through a
    refreshed stub - see procname.launchd_reexec."""
    tramp = procname.agent_trampoline()
    if tramp:
        return [tramp]
    return [os.path.abspath(sys.executable), "-m", "gmlx"]


def _load_agent(label: str, pp: Path) -> str | None:
    """bootout (idempotent) + bootstrap, retried: bootout unwinds its old
    instance asynchronously, and a bootstrap that lands mid-unwind fails.
    The legacy `load -w` fallback is verified with `launchctl print` rather
    than trusted - on current macOS it can return 0 without loading.
    Returns an error message, or None on success."""
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{label}"],
                   capture_output=True)             # drop any old instance
    err = ""
    for attempt in range(5):
        if attempt:
            time.sleep(0.5)
        r = subprocess.run(["launchctl", "bootstrap", domain, str(pp)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return None
        err = (r.stderr or "").strip()
    subprocess.run(["launchctl", "load", "-w", str(pp)], capture_output=True)
    if subprocess.run(["launchctl", "print", f"{domain}/{label}"],
                      capture_output=True).returncode == 0:
        return None
    return err or "unknown launchctl error"


def render_plist(label: str, program_args: list, log, *, env: dict | None = None,
                 keepalive: bool = True, run_at_load: bool = True) -> bytes:
    import plistlib
    pl = {
        "Label": label,
        "ProgramArguments": list(program_args),
        "RunAtLoad": run_at_load,
        # restart on a crash, but not on a clean stop, and back off between respawns
        "KeepAlive": {"SuccessfulExit": False} if keepalive else False,
        "ThrottleInterval": 10,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
        "EnvironmentVariables": dict(env or {}),
    }
    return plistlib.dumps(pl)


def service_install(serve_args: list, *, host: str, port: int,
                    config_abspath: str | None = None, log=None,
                    keepalive: bool = True, api_key_set: bool = False) -> int:
    """The --headless mode: a per-port LaunchAgent that runs `serve` itself
    (no menu bar, works for SSH-only boxes). The plist execs the venv
    interpreter and `serve --launchd` re-execs through a refreshed stub -
    see procname.launchd_reexec."""
    rc = _require_macos("service install")
    if rc:
        return rc
    host = host or "127.0.0.1"
    port = int(port or 8080)
    # A detach (`serve --background`) server on this host:port would fight the
    # launchd agent for the bind (crash-looping the agent) and lose its
    # runfile. Make the takeover explicit.
    existing = read_run(host, port)
    if existing and existing.get("managed_by") != "launchd" \
            and identity_ok(existing):
        print(f"a background server already holds http://{host}:{port} "
              f"(pid {existing.get('pid')}) - `gmlx stop` it first, then "
              f"re-run service install", file=sys.stderr)
        return 2
    lp = Path(os.path.expanduser(log)) if log else log_path(host, port)
    lp.parent.mkdir(parents=True, exist_ok=True)
    # launchd appends to StandardOutPath across every respawn and login;
    # cap it here (the one point gmlx owns before launchd opens the file).
    rotate_log_if_large(lp)
    child = [*_agent_entry(), "serve",
             *serve_args, "--host", host, "--port", str(port),
             "--foreground", "--launchd"]        # launchd detaches; serve in place
    label = _label(host, port)
    plist = render_plist(label, child, lp, env={"PATH": _agent_path()},
                         keepalive=keepalive)
    pp = _plist_path(host, port)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_bytes(plist)

    err = _load_agent(label, pp)
    if err:
        print(f"error: launchctl could not load the agent: {err}",
              file=sys.stderr)
        return 1

    write_run(host, port, {
        "pid": None, "pgid": None, "host": host, "port": port,
        "url": f"http://{host}:{port}", "config_abspath": config_abspath,
        "argv": list(child), "log": str(lp), "started_at": time.time(),
        "managed_by": "launchd", "label": label, "plist": str(pp),
        "api_key_set": bool(api_key_set),
    })
    tgt = "" if (host, port) == ("127.0.0.1", 8080) else f" --port {port}"
    print(f"installed launchd agent {label}")
    print(f"  plist: {pp}")
    print(f"  logs:  {lp}")
    print("  starts now and at login (restarts on crash); "
          f"remove with: gmlx service uninstall{tgt}")
    return 0


def service_install_menubar(serve_args: list, *, host: str, port: int,
                            config_abspath: str | None = None, log=None,
                            autostart: bool = True, start_timeout: float = 40.0,
                            api_key: str | None = None) -> int:
    """The default install mode: one LaunchAgent for the menu bar (launchd
    parentage makes TCC prompts attribute to gmlx), which optionally starts
    the recorded server at login when it isn't already up. The server itself
    stays an ordinary detach child - crash recovery is the bar's down
    notification and one-click Start, not a supervisor."""
    rc = _require_macos("service install")
    if rc:
        return rc
    host = host or "127.0.0.1"
    port = int(port or 8080)
    # A headless server agent on the same bind would race the menu bar's
    # autostart for the port at every login.
    if _plist_path(host, port).exists():
        print(f"a headless launchd server agent ({_label(host, port)}) "
              f"already exists for http://{host}:{port} - remove it first "
              f"(gmlx service uninstall{'' if port == 8080 else f' --port {port}'}) "
              "or re-run with --headless to update it", file=sys.stderr)
        return 2
    # Bring the server up now (detached) unless something already holds the
    # bind; its runfile argv becomes the durable autostart record.
    existing = read_run(host, port)
    if not (existing and identity_ok(existing)):
        rc = start_background(serve_args, host=host, port=port,
                              config_abspath=config_abspath, log=log,
                              start_timeout=start_timeout, api_key=api_key)
        if rc != 0:
            return rc
    run = read_run(host, port) or {}
    from . import menubar as _mb
    settings = _mb.load_menubar_settings()
    settings["autostart"] = ({
        "argv": [str(x) for x in (run.get("argv") or [])],
        "host": host, "port": port,
        "config_abspath": run.get("config_abspath") or config_abspath,
        "api_key_set": bool(run.get("api_key_set")),
    } if autostart and run.get("argv") else None)
    _mb.save_menubar_settings(settings)

    if not menubar_agent_loaded():
        stop_menubar()                # the agent instance replaces a detached
                                      # bar; an agent bar is recycled by the
                                      # bootout in _load_agent instead
    lp = menubar_log_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    rotate_log_if_large(lp, max_bytes=1_000_000)
    plist = render_plist(
        MENUBAR_AGENT_LABEL,
        [*_agent_entry(), "launch", "menubar", "--foreground", "--launchd"],
        lp, env={"PATH": _agent_path()})
    pp = _menubar_agent_plist_path()
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_bytes(plist)
    err = _load_agent(MENUBAR_AGENT_LABEL, pp)
    if err:
        print(f"error: launchctl could not load the agent: {err}",
              file=sys.stderr)
        return 1
    print(f"installed launchd agent {MENUBAR_AGENT_LABEL} (menu bar at login)")
    print(f"  plist: {pp}")
    print(f"  logs:  {lp}")
    # Report the record actually persisted: with no runfile argv to replay
    # (nothing to autostart), claiming "on" would be a lie.
    if settings["autostart"]:
        print(f"  server autostart: on - at login the menu bar starts "
              f"http://{host}:{port} when it isn't already up")
    else:
        print("  server autostart: off - start the server from the menu bar")
    print("  remove with: gmlx service uninstall")
    return 0


def _remove_agent(label: str, pp: Path) -> bool:
    """bootout + delete the plist; True when a plist was removed."""
    domain = f"gui/{os.getuid()}"
    r = subprocess.run(["launchctl", "bootout", f"{domain}/{label}"],
                       capture_output=True, text=True)
    if r.returncode != 0 and pp.exists():
        subprocess.run(["launchctl", "unload", "-w", str(pp)], capture_output=True)
    if pp.exists():
        try:
            pp.unlink()
            return True
        except OSError:
            pass
    return False


def service_uninstall(host: str, port) -> int:
    rc = _require_macos("service uninstall")
    if rc:
        return rc
    host = host or "127.0.0.1"
    port = int(port or 8080)

    # The menu-bar agent (default install mode) + its autostart record.
    mb_removed = _remove_agent(MENUBAR_AGENT_LABEL, _menubar_agent_plist_path())
    if mb_removed:
        try:
            from . import menubar as _mb
            settings = _mb.load_menubar_settings()
            settings["autostart"] = None
            _mb.save_menubar_settings(settings)
        except Exception:
            pass
        print(f"uninstalled launchd agent {MENUBAR_AGENT_LABEL} "
              "(menu bar login item; a running server is left up - "
              "`gmlx stop` for that)")

    # The per-port --headless server agent.
    label = _label(host, port)
    removed = _remove_agent(label, _plist_path(host, port))
    # Only clear a runfile this agent owns. A `serve --background` (detach)
    # server on the same host:port is not launchd-managed; deleting its
    # runfile would orphan the still-running process from stop/status/logs.
    run = read_run(host, port) or {}
    if removed or run.get("managed_by") == "launchd":
        _remove_run(host, port)
    if removed:
        print(f"uninstalled launchd agent {label} (plist removed)")
    elif mb_removed:
        pass
    elif run.get("managed_by") == "detach":
        print(f"no launchd agent for {host}:{port}; a background server is "
              f"running there (pid {run.get('pid')}) - use `gmlx stop` for it")
    else:
        print(f"no launchd agent for {host}:{port} (nothing to remove)")
    return 0


def service_status(host: str, port) -> int:
    rc = _require_macos("service status")
    if rc:
        return rc
    host = host or "127.0.0.1"
    port = int(port or 8080)
    domain = f"gui/{os.getuid()}"

    # The menu-bar agent (default install mode) first.
    r_mb = subprocess.run(["launchctl", "print",
                           f"{domain}/{MENUBAR_AGENT_LABEL}"],
                          capture_output=True, text=True)
    mb_loaded = r_mb.returncode == 0
    if mb_loaded or _menubar_agent_plist_path().exists():
        auto = None
        try:
            from . import menubar as _mb
            auto = _mb.load_menubar_settings().get("autostart")
        except Exception:
            pass
        state = "loaded" if mb_loaded else "installed but not loaded"
        extra = (f"; server autostart on ({auto.get('host')}:{auto.get('port')})"
                 if auto else "; server autostart off")
        print(f"launchd agent {MENUBAR_AGENT_LABEL}: {state}{extra}")

    label = _label(host, port)
    r = subprocess.run(["launchctl", "print", f"{domain}/{label}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"launchd agent {label}: not loaded")
        others = sorted(p.stem for p in
                        Path("~/Library/LaunchAgents").expanduser()
                        .glob("com.gmlx.server.*.plist") if p.stem != label)
        if others:
            print(f"  installed agent(s): {', '.join(others)} - target one "
                  f"with --host/--port")
        return 0 if mb_loaded else 3
    # launchctl print repeats keys across subsections - report each once.
    summary, seen = [], set()
    for ln in (s.strip() for s in r.stdout.splitlines()):
        key = next((k for k in ("state =", "pid =", "last exit")
                    if ln.startswith(k)), None)
        if key and key not in seen:
            seen.add(key)
            summary.append(ln)
    print(f"launchd agent {label}:")
    for ln in summary or ["(loaded)"]:
        print(f"  {ln}")
    return 0
