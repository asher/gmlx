"""One container session: its locks, its volumes, the ``container run``
command and the supervisor that runs it.

The supervisor is ``gmlx launch`` itself, which stays alive while the client
runs. It serves the socket relays in one thread, starts ``container run``,
forwards signals the CLI does not, and cleans up when the client exits. A
session holds an exclusive ``flock`` on its client's session lock, so one
session per client runs at a time, and a lock that is free means every
container and session folder of that client is left over from a killed
launch.
"""

from __future__ import annotations

import errno
import json
import os
import re
import resource
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gmlx.config import parse_size_bytes

from . import cli, runtime
from .clipboard import ClipboardServer
from .relay import Address, Relay, RelayLoop, loopback_targets
from .settings import ContainerPlan, Mount, SettingsError
from .state import FileLock, LockHeld, cache_dir, data_dir

HOST_SERVICES = "/var/host-services"
API_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-api.sock"
WEB_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-web.sock"
CLIP_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-clip.sock"
# macOS caps a socket path at 104 bytes.
SOCKET_PATH_MAX = 100
OPEN_TIMEOUT = 300.0
STOP_GRACE = 10
# The supervisor raises its open-file limit to this, or to the hard limit
# when that is lower, so guest connections cannot use up a soft limit of 256.
NOFILE_TARGET = 10240
LOG_MAX = 1 << 20
# How long a signal that arrives before the container exists waits for it.
PENDING_SIGNAL_WAIT = 60.0

Say = Callable[[str], None]


def _say(line: str) -> None:
    print(line, flush=True)


def fwd_guest_sock(port: int) -> str:
    return f"{HOST_SERVICES}/gmlx-fwd-{port}.sock"


# Session locks and the session record

def client_dir(client: str) -> Path:
    d = data_dir() / client
    d.mkdir(parents=True, exist_ok=True)
    return d


def try_session_lock(client: str) -> FileLock | None:
    """The client's session lock, or None when another session holds it."""
    try:
        return FileLock(client_dir(client) / "session.lock", blocking=False)
    except LockHeld:
        return None


def record_path(client: str) -> Path:
    return client_dir(client) / "session.json"


def write_record(client: str, record: dict) -> None:
    path = record_path(client)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, path)


def read_record(client: str) -> dict | None:
    try:
        record = json.loads(record_path(client).read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def remove_record(client: str) -> None:
    try:
        record_path(client).unlink()
    except FileNotFoundError:
        pass


# The session folder

@dataclass
class Session:
    client: str
    token: str
    dir: Path

    @property
    def name(self) -> str:
        return f"gmlx-{self.client}-{self.token}"

    def sock(self, name: str) -> Path:
        return self.dir / name


def session_dir_candidates(client: str) -> list[Path]:
    """Every session folder of ``client`` in both places sessions use."""
    found = []
    for root, prefix in ((cache_dir(), ""), (Path(_tmpdir()), "gmlx-launch-")):
        pattern = re.compile(rf"^{prefix}{re.escape(client)}-[0-9a-f]{{6}}$")
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        found += [e for e in entries if pattern.match(e.name) and e.is_dir()]
    return found


def _tmpdir() -> str:
    return os.environ.get("TMPDIR") or "/tmp"


def new_session(client: str, forward: list[int]) -> Session:
    """A fresh session folder, mode 0700. It moves to ``$TMPDIR`` when its
    longest socket path would pass the macOS limit, or when the cache path
    holds a ``:``, which ends the Mac side of a ``-v`` socket relay."""
    token = secrets.token_hex(3)
    longest = max([len("api.sock"), len("web.sock"), len("clip.sock")]
                  + [len(f"fwd-{p}.sock") for p in forward])
    folder = cache_dir() / f"{client}-{token}"
    if len(str(folder)) + 1 + longest > SOCKET_PATH_MAX or ":" in str(folder):
        folder = Path(_tmpdir()) / f"gmlx-launch-{client}-{token}"
    if ":" in str(folder):
        raise SettingsError(f"the session folder {folder} contains ':', which "
                            "`container run -v` cannot take. Set XDG_CACHE_HOME or TMPDIR "
                            "to a path without one.")
    folder.mkdir(mode=0o700, parents=True)
    os.chmod(folder, 0o700)
    return Session(client, token, folder)


# Volumes

def lock_volumes(volumes: list[Mount]) -> list[FileLock]:
    """An exclusive lock per volume, held until the session ends."""
    held: list[FileLock] = []
    try:
        for v in volumes:
            try:
                held.append(FileLock(data_dir() / "volumes" / f"{v.source}.lock",
                                     blocking=False))
            except LockHeld:
                raise SettingsError(
                    f"the volume {v.source} is in use by another launch session. Two "
                    "containers cannot attach one volume, so stop that session first.") from None
    except BaseException:
        for lock in held:
            lock.release()
        raise
    return held


def check_volumes_free(volumes: list[Mount], containers: list[cli.Container]) -> None:
    """Refuse a volume that a running container outside this session mounts."""
    wanted = {v.source for v in volumes}
    for c in containers:
        if c.state != "running":
            continue
        for name in c.volumes:
            if name in wanted:
                raise SettingsError(f"the volume {name} is mounted by the running container "
                                    f"{c.name}. Stop it first: container stop {c.name}")


def ensure_volumes(volumes: list[Mount], say: Say = _say) -> None:
    """Create the missing volumes with the launch label and their size, and
    say when an existing one has another size."""
    if not volumes:
        return
    existing = {v.name: v for v in cli.volume_list()}
    for v in volumes:
        have = existing.get(v.source)
        if have is None:
            cli.volume_create(v.source, size=v.size or "32G")
            continue
        want = parse_size_bytes(v.size or "32G")
        if have.size_bytes is None:
            say(f"[launch] the volume {v.source} was created without a size, so it has "
                "Apple's 512 GB default. To give it the configured size, delete it with "
                f"`container volume delete {v.source}`, which loses its data.")
        elif want is not None and have.size_bytes != want:
            say(f"[launch] the volume {v.source} has {gb(have.size_bytes)}, not the "
                f"configured {v.size}. The size applies only when a volume is created: "
                f"`container volume delete {v.source}` recreates it, and loses its data.")


def gb(n: int) -> str:
    """A size as G, or as M below one gibibyte."""
    if n < 1 << 30:
        return f"{n / (1 << 20):.0f}M"
    return f"{n / (1 << 30):.1f}G".replace(".0G", "G")


def allocated_bytes(path: str) -> int:
    """The disk space a file takes on the Mac: its allocated blocks, not its
    apparent size, since a volume's disk image is sparse."""
    try:
        return os.stat(path).st_blocks * 512
    except OSError:
        return 0


def volume_lines(volumes: list[Mount]) -> list[str]:
    """One summary line per volume, with its limit and its space on the Mac,
    and a warning when the Mac disk cannot hold the unused limits."""
    if not volumes:
        return []
    by_name = {v.name: v for v in cli.volume_list()}
    lines, unused, disk = [], 0, None
    for v in volumes:
        info = by_name.get(v.source)
        used = 0
        if info is not None and info.source and os.path.exists(info.source):
            used = allocated_bytes(info.source)
            disk = os.path.dirname(info.source)
        limit = parse_size_bytes(v.size or "32G") or 0
        unused += max(0, limit - used)
        lines.append(f"[launch] volume {v.source} at {v.target} ({v.size} limit, "
                     f"{gb(used)} used on the Mac)")
    if disk is not None:
        free = shutil.disk_usage(disk).free
        if free < unused:
            lines.append(f"[launch] warning: the Mac disk has {gb(free)} free, less than the "
                         f"{gb(unused)} these volumes may still grow into.")
    return lines


# The run command

@dataclass
class RunSpec:
    """Everything ``container run`` needs for one session."""
    session: Session
    plan: ContainerPlan
    image_ref: str
    runtime_dir: Path
    command: list[str]
    workdir: str
    env_values: dict[str, str]
    env_names: list[str]
    child_env: dict[str, str] = field(default_factory=dict)
    api_port: int | None = None
    web_port: int | None = None
    tty: bool = False
    interactive: bool = True           # -i: the client reads the terminal
    shell: bool = False
    # A line of the client's output that carries the URL to open, for a web
    # app whose URL holds a per-process login token.
    url_pattern: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


def _mount_arg(m: Mount) -> list[str]:
    if m.kind == "volume":
        return ["--mount", f"type=volume,source={m.source},target={m.target}"]
    return ["--mount", f"type=bind,source={m.source},target={m.target}"
            + (",readonly" if m.readonly else "")]


def compose_run_argv(spec: RunSpec, binary: str = "container") -> list[str]:
    s, plan = spec.session, spec.plan
    argv = [binary, "run", "--rm", "--init", "--progress", "none", "--name", s.name]
    labels = {"gmlx.launch": "1", "gmlx.launch.client": s.client,
              "gmlx.launch.pid": str(os.getpid()), **spec.labels}
    for key, value in labels.items():
        argv += ["--label", f"{key}={value}"]
    if spec.interactive:
        argv.append("-i")
    if spec.tty:
        argv.append("-t")
    argv += ["--uid", "0", "--gid", "0", "--cpus", str(plan.cpus), "--memory", plan.memory]
    if plan.network == "none":
        argv += ["--network", "none"]
    if plan.ssh_agent:
        argv.append("--ssh")
    argv += ["--workdir", spec.workdir]
    for key, value in spec.env_values.items():
        argv += ["-e", f"{key}={value}"]
    for name in spec.env_names:
        argv += ["-e", name]
    argv += ["--entrypoint", runtime.GUEST_ENTRY]
    runtime_mount = Mount(str(spec.runtime_dir), runtime.GUEST_MOUNT, readonly=True,
                          kind="runtime")
    for m in sorted([*plan.mounts, runtime_mount],
                    key=lambda m: (m.target.rstrip("/").count("/"), m.target)):
        argv += _mount_arg(m)
    if spec.api_port is not None:
        argv += ["-v", f"{s.sock('api.sock')}:{API_GUEST_SOCK}"]
    for port in plan.forward:
        argv += ["-v", f"{s.sock(f'fwd-{port}.sock')}:{fwd_guest_sock(port)}"]
    if plan.clipboard == "images":
        argv += ["-v", f"{s.sock('clip.sock')}:{CLIP_GUEST_SOCK}"]
    if spec.web_port is not None:
        argv += ["--publish-socket", f"{s.sock('web.sock')}:{WEB_GUEST_SOCK}"]
    argv.append(spec.image_ref)
    if spec.api_port is not None:
        argv += ["--tcp", f"{spec.api_port}={API_GUEST_SOCK}"]
    for port in plan.forward:
        argv += ["--tcp", f"{port}={fwd_guest_sock(port)}"]
    if spec.web_port is not None:
        argv += ["--unix", f"{WEB_GUEST_SOCK}={spec.web_port}"]
    if plan.clipboard == "images":
        argv.append("--clipboard")
    if spec.shell:
        argv.append("--shell")
    return [*argv, "--", *spec.command]


# Cleanup

def cleanup_stale(client: str, *, keep_runtime: str | None, say: Say = _say) -> None:
    """Remove what a killed session of ``client`` left behind, with one line
    per container. The caller holds the client's session lock, so every
    labelled container and session folder of this client is stale. Other
    clients are never touched."""
    for c in cli.list_launch_containers():
        if c.labels.get("gmlx.launch.client") != client:
            continue
        if c.state == "running":
            cli.stop(c.name, timeout=5)
        cli.delete(c.name)
        say(f"[launch] removed the leftover container {c.name} of an earlier session")
    for folder in session_dir_candidates(client):
        shutil.rmtree(folder, ignore_errors=True)
    runtime.cleanup_runtime(keep=keep_runtime)


def _pid_alive(pid: str | None) -> bool:
    try:
        os.kill(int(pid or ""), 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def leftover_containers(containers: list[cli.Container],
                        skip_client: str | None = None) -> list[cli.Container]:
    """Running launch containers whose ``gmlx launch`` process is gone. A
    reused process ID can only hide one, never mark a live one."""
    return [c for c in containers
            if c.labels.get("gmlx.launch") == "1" and c.state == "running"
            and c.labels.get("gmlx.launch.client") != skip_client
            and not _pid_alive(c.labels.get("gmlx.launch.pid"))]


def orphan_notices(client: str, containers: list[cli.Container]) -> list[str]:
    """Lines for other clients' launch containers whose launch is gone. Only
    reports: it probes no lock and deletes nothing."""
    out = []
    for c in leftover_containers(containers, skip_client=client):
        other = c.labels.get("gmlx.launch.client")
        memory = f" and holds {gb(c.memory_bytes)} of memory" if c.memory_bytes else ""
        out.append(f"[launch] {c.name} from an earlier {other} launch is still running{memory}. "
                   f"Stop it with: container stop {c.name}")
    return out


# The supervisor

def open_when_ready(port: int, opener: Callable[[str], object], stop: threading.Event,
                    say: Say = _say, timeout: float = OPEN_TIMEOUT) -> None:
    """Open the browser once the app answers an HTTP request through the
    relay. A bare connection proves nothing, since the relay accepts at once."""
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                conn.sendall(f"GET / HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
                if conn.recv(5) == b"HTTP/":
                    opener(url)
                    return
        except OSError:
            pass
        stop.wait(0.5)
    if not stop.is_set():
        say(f"[launch] nothing answered at {url} after {timeout:.0f} s, so the browser was "
            "not opened. A custom command must listen on 127.0.0.1:$PORT.")


class _Signals:
    """Signal handling while ``container run`` runs.

    SIGINT never stops launch itself. Without a terminal the CLI cannot pass
    SIGINT to the guest, so launch sends it with ``container kill``. A first
    SIGTERM or SIGHUP stops the container, a second kills it, and a third
    kills the ``container run`` process, for a runtime that no longer
    answers. The commands run in threads, and their errors go to the session
    log, never to the client's screen.
    """

    def __init__(self, name: str, tty: bool, log: Callable[[str], None] = lambda line: None):
        self.name, self.tty, self.log = name, tty, log
        self.child: subprocess.Popen | None = None
        self.count = 0
        self.saved: dict[int, object] = {}
        # Set when the child exits, so a pending stop gives up.
        self.done = threading.Event()

    def install(self) -> None:
        for sig, handler in ((signal.SIGINT, self._on_int), (signal.SIGTERM, self._on_term),
                             (signal.SIGHUP, self._on_term)):
            self.saved[sig] = signal.signal(sig, handler)

    def restore(self) -> None:
        for sig, handler in self.saved.items():
            signal.signal(sig, handler)

    def _bg(self, fn, *args, **kw) -> None:
        def run() -> None:
            try:
                fn(*args, **kw)
            except Exception as e:  # noqa: BLE001 - a signal thread must never raise
                self.log(f"signal: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _listed(self) -> bool:
        return any(c.name == self.name for c in cli.containers())

    def _when_listed(self, fn, *args, **kw) -> None:
        """Run ``fn`` once the container exists. A signal can arrive before
        ``container run`` has created it, and a stop of a missing name does
        nothing. When the child exits first, nothing is left to stop."""
        deadline = time.monotonic() + PENDING_SIGNAL_WAIT
        while not self._listed():
            if self.done.is_set() or time.monotonic() > deadline:
                return
            self.done.wait(0.2)
        fn(*args, **kw)

    def _on_int(self, signum, frame) -> None:
        if not self.tty:
            self._bg(cli.kill, self.name, signal="SIGINT")

    def _on_term(self, signum, frame) -> None:
        self.count += 1
        if self.count == 1:
            self._bg(self._when_listed, cli.stop, self.name, timeout=STOP_GRACE)
        elif self.count == 2:
            self._bg(self._kill)
        elif self.child is not None:
            self.child.kill()

    def _kill(self) -> None:
        if self._listed():
            cli.kill(self.name)
        elif self.child is not None:
            self.child.kill()             # the container does not exist yet


def supervise(spec: RunSpec, *, api_targets: list | None, record: dict,
              say: Say = _say, opener: Callable[[str], object] | None = None,
              summary: list[str] = ()) -> int:
    """Run the session and return the client's exit code."""
    s = spec.session
    raise_nofile_limit()
    log = _SessionLog(cache_dir() / f"last-{s.client}.log")
    loop = RelayLoop(log)
    loop.start()
    relays: list[Relay | ClipboardServer] = []
    child: subprocess.Popen | None = None
    signals: _Signals | None = None
    reader: threading.Thread | None = None
    stop_open = threading.Event()
    try:
        if spec.api_port is not None and api_targets:
            relays.append(_listen(lambda a: Relay(loop, a, api_targets, name="gmlx api"),
                                  str(s.sock("api.sock")), "the gmlx API"))
        for port in spec.plan.forward:
            relays.append(_listen(lambda a, p=port: Relay(loop, a, loopback_targets(p),
                                                          name=f"port {p}"),
                                  str(s.sock(f"fwd-{port}.sock")), f"forwarded port {port}"))
        if spec.web_port is not None:
            relays.append(_listen(lambda a: Relay(loop, a, str(s.sock("web.sock")), name="web"),
                                  ("127.0.0.1", spec.web_port), "the web app"))
        if spec.plan.clipboard == "images":
            relays.append(_listen(lambda a: ClipboardServer(loop, a),
                                  str(s.sock("clip.sock")), "the clipboard"))
        write_record(s.client, record)
        for line in summary:
            say(line)
        if spec.web_port is not None and spec.url_pattern is None:
            say(f"[launch] open http://127.0.0.1:{spec.web_port}/ in a browser")
            if opener is not None:
                threading.Thread(target=open_when_ready,
                                 args=(spec.web_port, opener, stop_open, say),
                                 daemon=True).start()
        argv = compose_run_argv(spec, cli.find() or "container")
        # A child that reads the terminal stays in the foreground group, or its
        # first read stops it with SIGTTIN. Any other child gets its own group,
        # so a Ctrl-C reaches only the supervisor, which forwards it.
        foreground = spec.tty or (spec.interactive and stdin_is_terminal())
        # Installed before the child starts. A handler resets across exec, so
        # the child starts with the default dispositions.
        signals = _Signals(s.name, spec.tty, log)
        signals.install()
        child = subprocess.Popen(
            argv, env={**os.environ, **spec.child_env},
            process_group=None if foreground else 0,
            stdin=None if spec.interactive else subprocess.DEVNULL,
            stdout=subprocess.PIPE if spec.url_pattern else None)
        signals.child = child
        if spec.url_pattern and child.stdout is not None:
            reader = threading.Thread(target=_tee_for_url, daemon=True, args=(
                child.stdout, spec.url_pattern, spec.web_port, opener, log))
            reader.start()
        rc = child.wait()
        signals.done.set()
        if reader is not None:
            reader.join(2)                # the last output reaches the terminal
        return rc if rc >= 0 else 128 - rc
    finally:
        # Each step runs even when one before it fails, and the signal
        # handlers stay until the end, so a Ctrl-C here cannot stop the
        # cleanup halfway.
        try:
            if signals is not None:
                signals.done.set()
            stop_open.set()
            _step(log, "remove the session record", remove_record, s.client)
            for relay in relays:
                _step(log, "close a relay", relay.close)
            _step(log, "stop the relay loop", loop.stop)
            if child is not None:
                _step(log, "remove the container", _remove_container, s.name,
                      stop=signals is None or signals.count < 3, log=log)
        finally:
            try:
                shutil.rmtree(s.dir, ignore_errors=True)
            finally:
                if signals is not None:
                    signals.restore()
                log.close()


def _step(log: Callable[[str], None], what: str, fn, /, *args, **kw) -> None:
    try:
        fn(*args, **kw)
    except Exception as e:  # noqa: BLE001 - one failed cleanup step must not skip the rest
        log(f"cleanup: cannot {what} ({type(e).__name__}: {e})")


def _listen(make: Callable[[Address], object], addr: Address, what: str):
    """Make one session listener, and name its address when that fails."""
    try:
        return make(addr)
    except OSError as e:
        where = addr if isinstance(addr, str) else f"{addr[0]}:{addr[1]}"
        reason = e.strerror or str(e)
        hint = (" Stop that program first." if e.errno == errno.EADDRINUSE
                and not isinstance(addr, str) else "")
        raise SettingsError(f"cannot listen on {where} for {what}: {reason}.{hint}") from None


def raise_nofile_limit(target: int = NOFILE_TARGET) -> None:
    """Raise the soft limit on open files toward ``target``, never past the
    hard limit. A failure leaves the limit as it was."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ValueError, OSError):
        pass


class _SessionLog:
    """The host log ``last-<client>.log``, replaced per session and capped at
    :data:`LOG_MAX` bytes. Writes are thread-safe and never raise."""

    def __init__(self, path: Path, limit: int = LOG_MAX):
        self.limit = limit
        self.size = 0
        self.full = False
        self._lock = threading.Lock()
        try:
            self._file = open(path, "w", buffering=1)
        except OSError:
            self._file = None

    def __call__(self, line: str) -> None:
        text = f"{time.strftime('%H:%M:%S')} {line}\n"
        with self._lock:
            if self._file is None or self.full:
                return
            if self.size + len(text) > self.limit:
                self.full = True
                text = f"{time.strftime('%H:%M:%S')} the log reached its size limit\n"
            try:
                self._file.write(text)
                self.size += len(text)
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


def _remove_container(name: str, *, stop: bool, log: Callable[[str], None]) -> None:
    """Stop and delete the session's container when it is still listed. After
    a third signal the runtime did not answer ``container stop``, so the
    container goes straight to ``container delete --force``. Errors go to the
    session log, so the client's exit code stands."""
    try:
        if not any(c.name == name for c in _safe_containers()):
            return
        if stop:
            cli.stop(name, timeout=5)
        cli.delete(name)
    except (cli.ContainerError, OSError) as e:
        log(f"cleanup: {e}")


def _tee_for_url(stream, pattern: str, web_port: int | None,
                 opener: Callable[[str], object] | None,
                 log: Callable[[str], None] = lambda line: None) -> None:
    """Copy the client's output to the terminal, and open the first URL the
    pattern finds. Only a URL of the session's own web port opens, so the
    guest cannot make the Mac open anything else. The copy goes on whatever
    the opener does, or the client would block on a full pipe."""
    regex = re.compile(pattern)
    opened = False
    out = sys.stdout.buffer
    for raw in iter(stream.readline, b""):
        try:
            out.write(raw)
            out.flush()
        except (OSError, ValueError):
            pass
        if opened or opener is None:
            continue
        m = regex.search(raw.decode("utf-8", "replace"))
        if m and m.group(1).startswith(f"http://127.0.0.1:{web_port}/"):
            opened = True
            try:
                opener(m.group(1))
            except Exception as e:  # noqa: BLE001 - see the docstring
                log(f"cannot open the browser ({type(e).__name__}: {e})")


def _safe_containers() -> list[cli.Container]:
    try:
        return cli.containers()
    except (cli.ContainerError, OSError):
        return []


def stdin_is_tty() -> bool:
    """Whether the session gets a terminal (``-t``): stdin and stdout both."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False
