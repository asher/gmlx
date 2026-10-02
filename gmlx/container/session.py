"""One container session: its locks, its volumes, the ``container run``
command and the supervisor that runs it.

The supervisor is ``gmlx launch`` itself, which stays alive while the client
runs. It serves the socket relays in one thread, starts ``container run``,
forwards signals the CLI does not, and cleans up when the container exits.
A session is keyed by its client and project, and holds an exclusive
``flock`` on that key's session lock, so one session per client and project
runs at a time. A lock that is free means every container and session
folder of that key is left over from a killed launch.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import termios
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gmlx.config import parse_size_bytes
from gmlx.rlimit import low_limit_warning, raise_nofile_limit
from gmlx.serve.session_paths import SESSION_CONNECTIONS_MAX

from . import cli, notices, runtime, settings
from .clipboard import ClipboardServer
from .relay import (CONNECTIONS_MAX, TARGET_CHECK_GAP, Address, Relay, RelayLoop,
                    loopback_targets)
from .settings import ContainerPlan, Mount, SettingsError
from .state import FileLock, LockHeld, cache_dir, data_dir
from .text import printable

HOST_SERVICES = "/var/host-services"
API_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-api.sock"
WEB_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-web.sock"
CLIP_GUEST_SOCK = f"{HOST_SERVICES}/gmlx-clip.sock"
# macOS caps a socket path at 104 bytes.
SOCKET_PATH_MAX = 100
OPEN_TIMEOUT = 300.0
STOP_GRACE = 10
LOG_MAX = 1 << 20
# The most connections a forwarded port holds at a time. A Mac service such
# as Postgres serves a fixed number of clients, so the container cannot hold
# all of them.
FORWARD_CONNECTIONS_MAX = 32
# The part of the log that only launch's own lines, such as cleanup and
# signals, may fill, so lines the guest causes cannot crowd them out.
LOG_OWN_RESERVE = 64 << 10
# A line the guest causes is logged once per this many seconds for each
# kind, with a count of the ones in between.
GUEST_LOG_EVERY = 60.0
GUEST_LOG_KINDS_MAX = 256
# How long a signal that arrives before the container exists waits for it.
PENDING_SIGNAL_WAIT = 60.0
# Each query of the session cleanup waits at most this long.
TEARDOWN_QUERY_TIMEOUT = 5.0
# ``container delete --force`` stops the VM first, which takes longer.
TEARDOWN_DELETE_TIMEOUT = 30.0
# The dsh output reader copies at most this much at a time, and looks for
# the URL in the last this many bytes.
TEE_CHUNK = 1 << 16
TEE_WINDOW = 4096

Say = Callable[[str], None]


def _say(line: str) -> None:
    print(printable(line), flush=True)


def fwd_guest_sock(port: int) -> str:
    return f"{HOST_SERVICES}/gmlx-fwd-{port}.sock"


# Session locks and the session record

# struct timeval on macOS, and room for one struct kinfo_proc (648 bytes
# on arm64).
_TIMEVAL = struct.Struct("@qi")
_KINFO_PROC_MAX = 1024
# How many times a launch opens its session lock when another launch keeps
# removing the empty project folder before the open.
_LOCK_TRIES = 5


def try_session_lock(client: str, project: str) -> FileLock | None:
    """The session lock of a client's project, or None when another session
    holds it. A lock on a file that another launch removed after this one
    opened it is taken again on the new file, and so is a lock whose folder
    another launch removed before this one opened the file."""
    misses = 0
    while True:
        try:
            lock = FileLock(settings.project_dir(client, project) / "session.lock",
                            blocking=False)
        except LockHeld:
            return None
        except FileNotFoundError:
            # drop_unused_project of a joining launch removed the folder
            # between the mkdir and the open.
            misses += 1
            if misses >= _LOCK_TRIES:
                raise
            continue
        if lock.still_current():
            return lock
        lock.release()


def drop_unused_project(client: str, project: str, lock: FileLock) -> None:
    """Remove the folder of a project that holds nothing but its session
    lock, such as the one a launch that joins another project's session
    leaves. The caller holds ``lock``."""
    folder = settings.project_dir_path(client, project)
    try:
        if os.listdir(folder) != ["session.lock"] or not lock.still_current():
            return
        (folder / "session.lock").unlink()
        folder.rmdir()
    except OSError:
        pass                     # another launch of the project wrote there meanwhile


def record_path(client: str, project: str) -> Path:
    return settings.project_dir_path(client, project) / "session.json"


def write_record(client: str, project: str, record: dict) -> None:
    path = record_path(client, project)
    try:
        settings.project_dir(client, project)
        write_private(path, json.dumps(record, indent=1).encode())
    except OSError as e:
        raise SettingsError(f"cannot write the session file {path} "
                            f"({e.strerror or e}).") from None


def started_path(client: str, project: str) -> Path:
    """The mark that a session of the project reached ``container run``.
    Until it exists, a launch prints the line for a new private home."""
    return settings.project_dir_path(client, project) / "started"


def mark_started(client: str, project: str) -> None:
    try:
        write_private(started_path(client, project), b"")
    except OSError:
        pass                     # costs only a repeat of the new-home line


def write_private(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data``, readable only by you. The temporary
    file is new, with a name no other writer uses, and never a link."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600)
    try:
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_record(client: str, project: str) -> dict | None:
    """The running session's record, or None when there is none. A record
    that is not the shape the supervisor writes is a SettingsError."""
    path = record_path(client, project)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError):
        record = None
    if not _record_ok(record):
        raise SettingsError(f"the session file {path} is damaged, so this launch cannot "
                            "join the running session.")
    return record


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _record_ok(record) -> bool:
    if not isinstance(record, dict):
        return False
    shares = record.get("shares")
    optional = {"command": _strings, "entrypoint": _strings,
                "project": lambda v: isinstance(v, str), "profile": lambda v: isinstance(v, str),
                "url": lambda v: isinstance(v, str),
                "web_port": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "pid": lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0,
                "pid_start": lambda v: isinstance(v, int) and not isinstance(v, bool)}
    return (isinstance(record.get("name"), str) and isinstance(record.get("workdir"), str)
            and all(isinstance(record.get(key, False), bool)
                    for key in ("clipboard", "web", "shell", "starting", "ending"))
            and all(record.get(key) is None or ok(record[key]) for key, ok in optional.items())
            and isinstance(shares, list)
            and all(isinstance(m, dict) and isinstance(m.get("host"), str)
                    and isinstance(m.get("guest"), str)
                    and isinstance(m.get("readonly", False), bool) for m in shares))


def remove_record(client: str, project: str) -> None:
    try:
        record_path(client, project).unlink()
    except FileNotFoundError:
        pass


def records(client: str) -> list[tuple[str, dict]]:
    """The session record of each of a client's projects that has one, as
    (project id, record) pairs. A damaged record is left out. A record
    stays after its session is killed, so :func:`record_runs` decides
    whether its session runs."""
    root = settings.data_path() / client / "projects"
    try:
        projects = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for project in projects:
        try:
            record = read_record(client, project)
        except SettingsError:
            continue
        if record is not None:
            out.append((project, record))
    return out


def record_runs(client: str, project: str, record: dict,
                containers: list[cli.Container]) -> bool:
    """Whether the container a session record names runs, with the labels
    of the record's client and project."""
    return any(c.state == "running" and c.name == record.get("name")
               and c.labels.get("gmlx.launch.client") == client
               and c.labels.get("gmlx.launch.project") == project for c in containers)


def session_state(client: str, project: str, record: dict,
                  containers: list[cli.Container]) -> str | None:
    """``starting`` or ``ending`` while the launch that marked the record so
    lives, and ``running`` when :func:`record_runs` is true and the launch
    that the record names lives, checked by its process ID and start time.
    A record from an older launch, with no process ID, is checked by the
    container's gmlx.launch.pid label. A record whose launch lives while
    its container does not run yet is ``starting``, since the launch writes
    it before ``container run`` boots the virtual machine. Else None. A
    container whose launch is gone is a leftover, which
    :func:`orphan_notices` reports."""
    for mark in ("starting", "ending"):
        if record.get(mark):
            return mark if _launch_alive(record) else None
    for c in containers:
        if (c.state == "running" and c.name == record.get("name")
                and _key(c) == (client, project)):
            alive = (_launch_alive(record) if record.get("pid")
                     else _pid_alive(c.labels.get("gmlx.launch.pid")))
            return "running" if alive else None
    return "starting" if record.get("pid") and _launch_alive(record) else None


def launch_owner() -> dict:
    """The fields of a session record that name this launch: its process ID
    and the time the process started. The system can give the ID of a
    launch that was killed to another process, and the start time tells
    the two apart."""
    return {"pid": os.getpid(), "pid_start": _process_start(os.getpid())}


def _launch_alive(record: dict) -> bool:
    """Whether the launch that :func:`launch_owner` names in ``record``
    still runs. A record without a start time is checked by its ID only.
    The record is in your own data folder, so a process of another user,
    such as launchd's 1, is never the launch that wrote it."""
    pid = record.get("pid")
    try:
        os.kill(int(pid or ""), 0)
    except (ValueError, OverflowError, OSError):
        return False
    start = record.get("pid_start")
    if start is None:
        return True
    now = _process_start(int(pid or 0))
    return now is None or now == start


def _process_start(pid: int) -> int | None:
    """When process ``pid`` started, in microseconds since the epoch, from
    the kernel's process table. None when there is no such process or the
    system cannot tell."""
    import ctypes

    try:
        sysctl = ctypes.CDLL(None, use_errno=True).sysctl
        # CTL_KERN, KERN_PROC, KERN_PROC_PID: one struct kinfo_proc, which
        # starts with the process start time as a struct timeval.
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)
        buf = ctypes.create_string_buffer(_KINFO_PROC_MAX)
        size = ctypes.c_size_t(len(buf))
        if sysctl(mib, 4, buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
    except (OSError, AttributeError, TypeError, OverflowError):
        return None
    if size.value < _TIMEVAL.size:
        return None                     # no such process
    seconds, micros = _TIMEVAL.unpack_from(buf.raw)
    return seconds * 1_000_000 + micros


# The session folder

@dataclass
class Session:
    client: str
    token: str
    dir: Path
    project: str = settings.PROJECT_DEFAULT

    @property
    def name(self) -> str:
        return f"gmlx-{self.client}-{self.token}"

    def sock(self, name: str) -> Path:
        return self.dir / name


def _project_tag(project: str) -> str:
    """Six hex digits of the project id, which session folder names carry."""
    import hashlib

    return hashlib.sha256(project.encode()).hexdigest()[:6]


def session_dir_candidates(client: str, project: str) -> list[Path]:
    """Every session folder of a client's project in both places sessions
    use."""
    tag = _project_tag(project) + "-"
    found = []
    for root, prefix in ((cache_dir(), ""), (Path(_tmpdir()), "gmlx-launch-")):
        pattern = re.compile(rf"^{prefix}{re.escape(client)}-{tag}[0-9a-f]{{6}}$")
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        found += [e for e in entries if pattern.match(e.name) and e.is_dir()]
    return found


def _tmpdir() -> str:
    return os.environ.get("TMPDIR") or "/tmp"


def new_session(client: str, project: str, forward: list[int]) -> Session:
    """A fresh session folder, mode 0700, whose name carries the project's
    tag. It moves to ``$TMPDIR`` when its longest socket path would pass the
    macOS limit, or when the cache path holds a ``:``, which ends the Mac
    side of a ``-v`` socket relay."""
    token = secrets.token_hex(3)
    name = f"{client}-{_project_tag(project)}-{token}"
    longest = max([len("api.sock"), len("web.sock"), len("clip.sock")]
                  + [len(f"fwd-{p}.sock") for p in forward])
    folder = cache_dir() / name
    if len(str(folder)) + 1 + longest > SOCKET_PATH_MAX or ":" in str(folder):
        folder = Path(_tmpdir()) / f"gmlx-launch-{name}"
    if ":" in str(folder):
        raise SettingsError(f"the session folder {folder} contains a colon, which Apple "
                            "container cannot take in a socket path. Set XDG_CACHE_HOME or "
                            "TMPDIR to a path without one.")
    try:
        folder.mkdir(mode=0o700, parents=True)
        os.chmod(folder, 0o700)
    except OSError as e:
        raise SettingsError(f"cannot create the session folder {folder} "
                            f"({e.strerror or e}).") from None
    return Session(client, token, folder, project)


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
                raise settings.Busy(
                    f"the volume {v.source} is in use by another launch session, and two "
                    "containers cannot use one volume at once. End that session first.") from None
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
                raise settings.Busy(f"the running container {c.name} uses the volume {name}. "
                                    f"Stop it first with: container stop {c.name}")


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
        # Its only fix deletes the data, so each warning prints once for the
        # volume and the two sizes.
        key = f"volume-size:{v.source}:{have.size_bytes}:{want}"
        if have.size_bytes is None:
            warn = (f"[launch] warning: the volume {v.source} was created without a size, so "
                    "it has Apple's 512 GB default. Deleting it loses its data, and the next "
                    "launch creates it with the configured size. Delete it with: container "
                    f"volume delete {v.source}")
        elif want is not None and have.size_bytes != want:
            warn = (f"[launch] warning: the volume {v.source} has {gb(have.size_bytes)}, not "
                    f"the configured {v.size or '32G'}, because a size applies only when a "
                    "volume is created. Deleting it loses its data, and the next launch "
                    f"creates it with the configured size. Delete it with: container volume "
                    f"delete {v.source}")
        else:
            continue
        for line in notices.due([notices.Once(warn, key)]):
            say(line)


def gb(n: int) -> str:
    """A size as G, or as M below one gibibyte, where a size of a few
    kibibytes reads as under 1M rather than 0M."""
    if 0 < n <= 1 << 19:
        return "under 1M"
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
    """One summary line per volume, with the limit it has, which a volume
    created earlier keeps whatever the config says, and its space on the
    Mac, and a warning when the Mac disk cannot hold the unused limits."""
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
        if info is None:
            limit = parse_size_bytes(v.size or "32G") or 0
        else:
            limit = info.size_bytes or cli.VOLUME_DEFAULT_BYTES
        unused += max(0, limit - used)
        lines.append(f"[launch] volume {v.source} at {v.target} ({gb(limit)} limit, "
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


def recheck_sources(spec: RunSpec) -> None:
    """Check the shared folders and the runtime folder again just before
    ``container run``. A client of another session can swap a shared folder
    for a link after the plan was made, and ``container run`` would follow
    it."""
    settings.recheck_sources(spec.plan)
    try:
        st = os.lstat(spec.runtime_dir)
    except OSError:
        st = None
    if st is None or not stat.S_ISDIR(st.st_mode) or not runtime._complete(Path(spec.runtime_dir)):
        raise SettingsError(f"{spec.runtime_dir}, which holds launch's program for the "
                            "container, changed after launch checked it. Launch again.")


def compose_run_argv(spec: RunSpec, binary: str = "container") -> list[str]:
    s, plan = spec.session, spec.plan
    argv = [binary, "run", "--rm", "--init", "--progress", "none", "--name", s.name]
    labels = {"gmlx.launch": "1", "gmlx.launch.client": s.client,
              "gmlx.launch.project": s.project, "gmlx.launch.pid": str(os.getpid()),
              **spec.labels}
    for key, value in labels.items():
        argv += ["--label", f"{key}={value}"]
    if spec.interactive:
        argv.append("-i")
    if spec.tty:
        argv.append("-t")
    argv += ["--uid", "0", "--gid", "0", "--cpus", str(plan.cpus), "--memory", plan.memory]
    if plan.network == "none":
        argv += ["--network", "none"]
    if settings.forwarded_agent(plan):
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

def cleanup_stale(client: str, project: str, *, keep_runtime: str | None,
                  say: Say = _say) -> None:
    """Remove what a killed session of a client's project left behind, with
    one line per container. The caller holds that project's session lock,
    so every container and session folder labelled with it is stale. Other
    projects and clients are never touched, and their locks never probed.
    A container that is still listed after its delete gets the command that
    removes it, since it keeps its memory."""
    stale = [c for c in cli.list_launch_containers()
             if c.labels.get("gmlx.launch.client") == client
             and c.labels.get("gmlx.launch.project") == project]
    for c in stale:
        if c.state == "running":
            cli.stop(c.name, timeout=5)
        cli.delete(c.name)
    left = {c.name for c in cli.containers()} if stale else set()
    for c in stale:
        say(f"[launch] the leftover container {c.name} of an earlier session is still there. "
            f"Remove it with: container delete --force {c.name}" if c.name in left else
            f"[launch] removed the leftover container {c.name} of an earlier session")
    for folder in session_dir_candidates(client, project):
        shutil.rmtree(folder, ignore_errors=True)
    runtime.cleanup_runtime(keep=keep_runtime)


def _pid_alive(pid: str | None) -> bool:
    try:
        os.kill(int(pid or ""), 0)
    except (ValueError, OverflowError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _key(c: cli.Container) -> tuple[str | None, str | None]:
    return c.labels.get("gmlx.launch.client"), c.labels.get("gmlx.launch.project")


def leftover_containers(containers: list[cli.Container],
                        skip: tuple[str, str] | None = None) -> list[cli.Container]:
    """Running launch containers whose ``gmlx launch`` process is gone,
    except the ones of the client and project ``skip`` names. A reused
    process ID can only hide one, never mark a live one."""
    return [c for c in containers
            if c.labels.get("gmlx.launch") == "1" and c.state == "running"
            and (skip is None or _key(c) != skip)
            and not _pid_alive(c.labels.get("gmlx.launch.pid"))]


def orphan_notices(client: str, project: str, containers: list[cli.Container]) -> list[str]:
    """Lines for other sessions' launch containers whose launch is gone.
    Only reports: it probes no lock and deletes nothing."""
    out = []
    for c in leftover_containers(containers, skip=(client, project)):
        other = c.labels.get("gmlx.launch.client")
        # An image check names no client.
        whose = f"an earlier {other} launch" if other else "the image check of an earlier launch"
        memory = f" and holds {gb(c.memory_bytes)} of memory" if c.memory_bytes else ""
        out.append(f"[launch] {c.name} from {whose} is still running{memory}. "
                   f"Stop it with: container stop {c.name}")
    return out


def mac_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


def memory_line(containers: list[cli.Container], memory: str) -> str | None:
    """The memory every running launch container and this session will
    hold, against the Mac's, when another launch container runs. Each
    virtual machine holds its container's memory and
    :data:`settings.VM_MEMORY_OVERHEAD`."""
    running = [c for c in containers
               if c.labels.get("gmlx.launch") == "1" and c.state == "running"]
    own = parse_size_bytes(memory)
    total = mac_memory_bytes()
    if not running or own is None or not total:
        return None
    held = (own + sum(c.memory_bytes or 0 for c in running)
            + settings.VM_MEMORY_OVERHEAD * (len(running) + 1))
    others = f"{len(running)} other launch container{'s' if len(running) != 1 else ''}"
    return (f"[launch] with {others} running, launch containers will hold {gb(held)} of "
            f"the Mac's {gb(total)} of memory, which the model server cannot use.")


# The supervisor

# The program that opens an address in the Mac's browser, by its full path.
# A guest can put a program in a shared folder on PATH, such as the bin
# folder of a project's virtual environment, and Python's webbrowser module
# runs osascript from PATH.
OPEN_PROGRAM = "/usr/bin/open"
OPEN_PROGRAM_TIMEOUT = 30.0


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the Mac's default browser with :data:`OPEN_PROGRAM`,
    and return whether it did. PATH stays as it is, because launch keys
    the container program it found on the PATH value."""
    try:
        done = subprocess.run([OPEN_PROGRAM, url], stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=OPEN_PROGRAM_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def open_when_ready(port: int, opener: Callable[[str], object], stop: threading.Event,
                    say: Say = _say, timeout: float = OPEN_TIMEOUT, *,
                    browser: bool = True) -> None:
    """Call ``opener`` with the app's address once the app answers an HTTP
    request through the relay. A bare connection proves nothing, since the
    relay accepts at once. ``browser`` says whether ``opener`` opens a
    browser, which the timeout line mentions."""
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
        say(f"[launch] nothing answered at {url} after {timeout:.0f} s"
            + (", so the browser was not opened" if browser else "")
            + ". A custom command must listen on 127.0.0.1:$PORT.")


class TeardownAbandoned(BaseException):
    """Raised by a third signal during the cleanup of a session. It is not
    an Exception, so a cleanup step that catches errors lets it through and
    the remaining steps are skipped."""


class _Signals:
    """Signal handling while ``container run`` runs.

    SIGINT never stops launch itself. Without a terminal the CLI cannot pass
    SIGINT to the guest, so launch sends it with ``container kill``. A first
    SIGTERM or SIGHUP stops the container, a second kills it, and a third
    kills the ``container run`` process, for a runtime that no longer
    answers. A signal that arrives before the container runs waits for it.
    The commands run in threads, and their errors go to the session log,
    never to the client's screen.

    The handlers stay installed while the session is cleaned up. A third
    signal of any kind during the cleanup abandons the step that waits, for
    a container service that no longer answers.

    A signal that was ignored when launch started, as nohup ignores SIGHUP,
    stays ignored, and ``container run`` inherits that.
    """

    def __init__(self, name: str, tty: bool, log: Callable[[str], None] = lambda line: None):
        self.name, self.tty, self.log = name, tty, log
        self.child: subprocess.Popen | None = None
        self.count = 0
        self.saved: dict[int, object] = {}
        # Set when the child exits, so a pending stop gives up.
        self.done = threading.Event()
        # Set while the session is cleaned up; counts the signals since.
        self.tearing_down = False
        self.teardown_count = 0

    def install(self) -> None:
        for sig, handler in ((signal.SIGINT, self._on_int), (signal.SIGTERM, self._on_term),
                             (signal.SIGHUP, self._on_term)):
            if signal.getsignal(sig) != signal.SIG_IGN:
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

    def _state(self) -> str | None:
        """The state that ``container ls`` gives the container, or None
        when it is not listed."""
        return next((c.state for c in cli.containers(own_group=True) if c.name == self.name),
                    None)

    def _listed(self) -> bool:
        return self._state() is not None

    def _when_running(self, fn, *args, **kw) -> None:
        """Run ``fn`` once the container runs. A signal can arrive before
        ``container run`` has created the container, or while its virtual
        machine starts. Apple container lists the container from its create
        on, but until the machine has started, a stop does nothing and a
        kill fails. When the child exits first, nothing is left to stop."""
        deadline = time.monotonic() + PENDING_SIGNAL_WAIT
        while self._state() not in ("running", "stopping"):
            if self.done.is_set() or time.monotonic() > deadline:
                return
            self.done.wait(0.2)
        fn(*args, **kw)

    def _abandon_teardown(self) -> bool:
        """Count a signal during the cleanup, and abandon the cleanup at the
        third. Returns True when the signal belongs to the cleanup."""
        if not self.tearing_down:
            return False
        self.teardown_count += 1
        if self.teardown_count >= 3:
            self.tearing_down = False
            raise TeardownAbandoned
        return True

    def _on_int(self, signum, frame) -> None:
        if self._abandon_teardown():
            return
        if not self.tty:
            self._bg(self._when_running, cli.kill, self.name, signal="SIGINT")

    def _on_term(self, signum, frame) -> None:
        if self._abandon_teardown():
            return
        self.count += 1
        if self.count == 1:
            self._bg(self._when_running, cli.stop, self.name, timeout=STOP_GRACE)
        elif self.count == 2:
            self._bg(self._kill)
        elif self.child is not None:
            self.child.kill()

    def _kill(self) -> None:
        if self._listed():
            self._when_running(cli.kill, self.name)
        elif self.child is not None:
            self.child.kill()             # the container does not exist yet


def shell_start(record: dict) -> str:
    """The end of the line that tells how to start a web app from a shell
    in its session. The app's own default port is not the session's port,
    so the line names the recorded command, which holds the session's port.
    command: image runs in the image's working folder, and a shell starts in
    another folder, so the line changes to that folder first."""
    start, folder = record.get("command"), record.get("command_workdir")
    if not start or not _strings(start):
        return ", where it must listen on 127.0.0.1:$PORT"
    text = shlex.join(start)
    if isinstance(folder, str) and folder:
        text = f"cd {shlex.quote(folder)} && {text}"
    return f" with: {text}"


def supervise(spec: RunSpec, *, api_targets: list | None, record: dict,
              say: Say = _say, opener: Callable[[str], object] | None = None,
              summary: list[str] = (), server_session=None,
              on_start: Callable[[], None] | None = None) -> int:
    """Run the session and return the client's exit code. With a
    ``server_session``, the API relay goes to the session socket it opens
    instead of ``api_targets``, and asks it for a new socket when that one
    stops answering, such as after a server restart. ``on_start`` runs once
    ``container run`` has started."""
    s = spec.session
    nofile = raise_nofile_limit()
    log = _SessionLog(cache_dir() / f"last-{s.client}-{s.project}.log")
    low = low_limit_warning(nofile, "the launch supervisor")
    if low:
        log(f"warning: {low}")
    loop = RelayLoop(log.guest, event=log.guest_event)
    loop.start()
    relays: list[Relay | ClipboardServer] = []
    child: subprocess.Popen | None = None
    signals: _Signals | None = None
    reader: threading.Thread | None = None
    stop_open = threading.Event()
    # The record as last written, which teardown marks as ending.
    recorded: dict | None = None
    record_lock = threading.Lock()

    def mark_ending() -> None:
        """Mark the record as ending, once. A launch that would join the
        session then gets the ending line until the container is gone."""
        nonlocal recorded
        with record_lock:
            stop_open.set()
            if recorded is None or recorded.get("ending"):
                return
            recorded = {**recorded, "ending": True, "pid": os.getpid()}
        _step(log, "mark the session record as ending", write_record, s.client, s.project,
              recorded)
    try:
        renew = None
        if server_session is not None:
            server_session.log = log.guest
            api_targets, renew = [server_session.open()], server_session.renew
        if spec.api_port is not None and api_targets:
            # Each API connection holds one of the shared server's
            # descriptors, so it must send a whole request head in time.
            # A session socket answers 503 once more connections than this
            # cap are open, so more clients wait in the relay's listen queue.
            # The relay also asks for a new session socket soon after a
            # server restart, so the server refuses the app's pages again.
            relays.append(_listen(lambda a: Relay(loop, a, api_targets, name="gmlx api",
                                                  idle_until_head=True, renew=renew,
                                                  check_every=TARGET_CHECK_GAP,
                                                  max_connections=SESSION_CONNECTIONS_MAX
                                                  if server_session is not None
                                                  else CONNECTIONS_MAX),
                                  str(s.sock("api.sock")), "the gmlx API"))
        for port in spec.plan.forward:
            relays.append(_listen(lambda a, p=port: Relay(
                loop, a, loopback_targets(p), name=f"port {p}",
                max_connections=FORWARD_CONNECTIONS_MAX),
                                  str(s.sock(f"fwd-{port}.sock")), f"forwarded port {port}"))
        if spec.web_port is not None:
            relays.append(_listen(lambda a: Relay(loop, a, str(s.sock("web.sock")), name="web"),
                                  ("127.0.0.1", spec.web_port), "the web app"))
        if spec.plan.clipboard == "images":
            relays.append(_listen(lambda a: ClipboardServer(loop, a),
                                  str(s.sock("clip.sock")), "the clipboard"))
        write_record(s.client, s.project, record)
        recorded = record
        for line in [*summary, *(server_session.lines() if server_session else [])]:
            say(line)
        if spec.web_port is not None and spec.shell:
            say(f"[launch] the web app answers at http://127.0.0.1:{spec.web_port}/ "
                f"once you start it from the shell{shell_start(record)}")
        elif spec.web_port is not None and spec.url_pattern is None:
            if opener is not None:
                say(f"[launch] opening http://127.0.0.1:{spec.web_port}/ in your browser "
                    "once the app answers")
            # Without a browser the address prints once the app answers, since
            # an app can take minutes to start.
            ready = opener or (lambda url: say(f"[launch] the web app answers at {url}"))
            threading.Thread(target=open_when_ready,
                             args=(spec.web_port, ready, stop_open, say),
                             kwargs={"browser": opener is not None}, daemon=True).start()
        recheck_sources(spec)
        argv = compose_run_argv(spec, cli.find() or "container")
        # A child that reads the terminal stays in the foreground group, or its
        # first read stops it with SIGTTIN. Any other child gets its own group,
        # so a Ctrl-C reaches only the supervisor, which forwards it.
        foreground = spec.tty or (spec.interactive and stdin_is_terminal())
        # Installed before the child starts. A handler resets across exec, so
        # the child starts with the default dispositions.
        signals = _Signals(s.name, spec.tty, log)
        signals.install()
        # A killed ``container run -t`` leaves the terminal in the raw mode
        # it set, so launch then puts back the settings from before the start.
        mode = _terminal_mode() if spec.tty else None
        try:
            child = subprocess.Popen(
                argv, env={**os.environ, **spec.child_env},
                process_group=None if foreground else 0,
                stdin=None if spec.interactive else subprocess.DEVNULL,
                stdout=subprocess.PIPE if spec.url_pattern else None)
        except OSError as e:
            raise cli.ContainerError(f"cannot start `container run` "
                                     f"({e.strerror or e}).") from None
        signals.child = child
        if on_start is not None:
            try:
                on_start()
            except Exception as e:  # noqa: BLE001 - a record that fails must not end the session
                log(f"cannot record the session start ({type(e).__name__}: {e})")
        if spec.url_pattern and child.stdout is not None:
            def found(url: str) -> None:
                nonlocal recorded
                with record_lock:
                    if stop_open.is_set():
                        return            # teardown marks the record
                    # A second launch of the web app opens the recorded URL.
                    recorded = {**record, "url": url}
                    write_record(s.client, s.project, recorded)
            reader = threading.Thread(target=_tee_for_url, daemon=True, args=(
                child.stdout, spec.url_pattern, spec.web_port, opener, log, found))
            reader.start()
        rc = child.wait()
        signals.done.set()
        # The container has stopped, so the session ends, also while the
        # last output reaches the terminal.
        mark_ending()
        if rc < 0 and mode is not None:
            _restore_terminal(mode)
        if spec.tty:
            _flush_terminal_input()
        if reader is not None:
            reader.join(2)                # the last output reaches the terminal
        refused = getattr(server_session, "refused", None)
        if refused:
            # The client owned the terminal until now.
            say("[launch] the server stopped answering on the session socket and gave no "
                "new one, so the client could not reach it after that.")
            say(f"[launch] {refused}")
        return rc if rc >= 0 else 128 - rc
    finally:
        # Each step runs even when one before it fails, and the signal
        # handlers stay until the end, so a Ctrl-C here cannot stop the
        # cleanup halfway. The queries get a short timeout, and a third
        # signal abandons a step that still waits.
        try:
            if signals is not None:
                signals.done.set()
                signals.tearing_down = True
            mark_ending()
            for relay in relays:
                _step(log, "close a relay", relay.close)
            _step(log, "stop the relay loop", loop.stop)
            if server_session is not None:
                _step(log, "end the server session", server_session.close)
            if child is not None:
                with cli.query_timeout(TEARDOWN_QUERY_TIMEOUT):
                    _step(log, "remove the container", _remove_container, s.name,
                          stop=signals is None or signals.count < 3, log=log)
                    _step(log, "check the container is gone", _report_leftover, s.name,
                          log=log)
        except TeardownAbandoned:
            log("cleanup: abandoned after a third signal")
        finally:
            # A third signal from here on no longer abandons anything.
            if signals is not None:
                signals.tearing_down = False
            try:
                _step(log, "remove the session record", remove_record, s.client, s.project)
                shutil.rmtree(s.dir, ignore_errors=True)
            finally:
                if signals is not None:
                    signals.restore()
                log.close()
                if child is not None and spec.tty:
                    # The answers to the client's last queries can arrive
                    # while the session is cleaned up.
                    _flush_terminal_input()


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
        busy = e.errno == errno.EADDRINUSE and not isinstance(addr, str)
        error = settings.Busy if busy else SettingsError
        raise error(f"cannot listen on {where} for {what} ({reason})."
                    + (" Stop that program first." if busy else "")) from None


class _SessionLog:
    """The host log ``last-<client>-<project>.log``, replaced per session and capped at
    :data:`LOG_MAX` bytes. Writes are thread-safe and never raise.

    Calling the log writes launch's own lines. :meth:`guest` writes a line
    the guest causes, such as a refused connection, at most once a minute
    for each kind of line, and :meth:`guest_event` writes one for every
    event, such as each image the clipboard sends. Guest lines stop
    :data:`LOG_OWN_RESERVE` bytes before the cap, so launch's own lines
    still fit after a guest has filled its part."""

    def __init__(self, path: Path, limit: int = LOG_MAX,
                 reserve: int = LOG_OWN_RESERVE, every: float = GUEST_LOG_EVERY):
        self.limit = limit
        self.guest_limit = max(0, limit - reserve)
        self.every = every
        self.size = 0
        self.full = False
        self.guest_full = False
        self._seen: dict[str, list] = {}      # kind -> [last write, lines skipped]
        self._lock = threading.Lock()
        self._file = _open_log(path)

    def __call__(self, line: str) -> None:
        self._write(line, guest=False)

    def guest_event(self, line: str) -> None:
        self._write(line, guest=True)

    def guest(self, line: str) -> None:
        kind = re.sub(r"\d+", "#", line.split(" (", 1)[0])
        now = time.monotonic()
        with self._lock:
            if kind not in self._seen and len(self._seen) >= GUEST_LOG_KINDS_MAX:
                kind = "other"
            seen = self._seen.setdefault(kind, [None, 0])
            if seen[0] is not None and now - seen[0] < self.every:
                seen[1] += 1
                return
            skipped, seen[0], seen[1] = seen[1], now, 0
        if skipped:
            line = f"{line} (and {skipped} more like it since the last one logged)"
        self._write(line, guest=True)

    def _write(self, line: str, *, guest: bool) -> None:
        text = f"{time.strftime('%H:%M:%S')} {printable(line)}\n"
        with self._lock:
            if self._file is None or self.full or (guest and self.guest_full):
                return
            if guest and self.size + len(text) > self.guest_limit:
                self.guest_full = True
                text = (f"{time.strftime('%H:%M:%S')} the log reached its size limit "
                        "for lines the container causes\n")
            elif self.size + len(text) > self.limit:
                self.full = True
                text = f"{time.strftime('%H:%M:%S')} the log reached its size limit\n"
            try:
                self._file.write(text)
                self.size += len(text)
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        with self._lock:
            skipped = sorted((k, v[1]) for k, v in self._seen.items() if v[1])
        for kind, n in skipped:
            self(f"{kind}: {n} more like it were not logged")
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None


def _open_log(path: Path):
    """The session log, emptied, readable only by you. A link or anything
    other than a regular file at the path is not opened."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
                     | os.O_CLOEXEC, 0o600)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        return os.fdopen(fd, "w", buffering=1)
    except OSError:
        os.close(fd)
        return None


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
        with cli.query_timeout(TEARDOWN_DELETE_TIMEOUT):
            cli.delete(name)
    except (cli.ContainerError, OSError) as e:
        log(f"cleanup: {e}")


def _report_leftover(name: str, *, log: Callable[[str], None]) -> None:
    """Print one line when the session's container is still listed after the
    cleanup, since it keeps its memory until it is removed."""
    if any(c.name == name for c in _safe_containers()):
        line = (f"[launch] the container {name} is still there after the session. Remove "
                f"it with: container delete --force {name}")
        log(line)
        print(printable(line), file=sys.stderr, flush=True)


def _tee_for_url(stream, pattern: str, web_port: int | None,
                 opener: Callable[[str], object] | None,
                 log: Callable[[str], None] = lambda line: None,
                 found: Callable[[str], object] | None = None) -> None:
    """Copy the client's output to the terminal, and open the first URL the
    pattern finds and pass it to ``found``. Only a URL of the session's own
    web port counts, so the guest cannot make the Mac open anything else.
    The copy goes on whatever the opener does, or the client would block on
    a full pipe."""
    regex = re.compile(pattern)
    opened = False
    out = sys.stdout.buffer
    tail = b""
    # read1 returns what the pipe holds, up to the limit, so output with no
    # newline never piles up in memory.
    while chunk := stream.read1(TEE_CHUNK):
        try:
            out.write(chunk)
            out.flush()
        except (OSError, ValueError):
            pass
        if opened or (opener is None and found is None):
            continue
        window = tail + chunk
        tail = window[-TEE_WINDOW:]
        text = window.decode("latin-1")
        for m in regex.finditer(text):
            # A match that runs to the end of the window may be cut short;
            # the next window holds the rest of it.
            if m.end() == len(text):
                break
            url = m.group(1)
            parts = urllib.parse.urlsplit(url)
            if (url.startswith(f"http://127.0.0.1:{web_port}/") and url.isprintable()
                    and parts.scheme == "http" and parts.netloc == f"127.0.0.1:{web_port}"):
                opened = True
                for call, what in ((found, "record the address"), (opener, "open the browser")):
                    if call is None:
                        continue
                    try:
                        call(url)
                    except Exception as e:  # noqa: BLE001 - see the docstring
                        log(f"cannot {what} ({type(e).__name__}: {e})")
                break


def _safe_containers() -> list[cli.Container]:
    try:
        return cli.containers(own_group=True)
    except (cli.ContainerError, OSError):
        return []


def run_copy(argv: list[str], env: dict, *, name: str, copy_id: str) -> int:
    """Run the ``container exec`` of a joined copy and return its exit code.

    The CLI passes no SIGHUP to the guest, and with a terminal no SIGTERM
    either, so a closed window would leave the copy running with no
    terminal. Launch therefore stays the parent of ``container exec``. A
    first SIGHUP or SIGTERM sends the copy SIGHUP through
    ``gmlx-entry --hangup``, and a second one kills ``container exec``.
    The handlers stay until that hangup ends, since a closed window sends
    SIGHUP twice. SIGINT never stops launch itself, so it reaches the CLI
    alone.

    A signal that was ignored when launch started, as nohup ignores SIGHUP,
    stays ignored, and ``container exec`` inherits that. A killed
    ``container exec`` leaves the terminal in the raw mode it set, so launch
    then puts back the terminal settings from before the start. With a
    terminal, the input that waits when the copy ends is dropped, as
    :func:`_flush_terminal_input` explains."""
    child: list[subprocess.Popen] = []
    hangups: list[threading.Thread] = []
    killed = threading.Event()

    def hang_up() -> None:
        with contextlib.suppress(cli.ContainerError, OSError):
            cli.hangup_copy(name, runtime.GUEST_ENTRY, copy_id)

    def kill() -> None:
        killed.set()
        child[0].kill()

    def on_end(signum, frame) -> None:
        hangups.append(threading.Thread(target=hang_up, daemon=True))
        if len(hangups) == 1:
            hangups[0].start()
        elif child:
            kill()

    saved = {sig: signal.signal(sig, handler) for sig, handler in
             ((signal.SIGINT, lambda signum, frame: None), (signal.SIGTERM, on_end),
              (signal.SIGHUP, on_end)) if signal.getsignal(sig) != signal.SIG_IGN}
    mode = _terminal_mode()
    tty = stdin_is_tty()
    try:
        child.append(subprocess.Popen(argv, env=env))
        if len(hangups) > 1:              # a second signal came during the start
            kill()
        code = child[0].wait()
        if tty:
            _flush_terminal_input()
        # A closed window ends ``container exec`` too, so the hangup can
        # still be on its way.
        if hangups:
            hangups[0].join(cli.HANGUP_TIMEOUT + 5)
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
    if killed.is_set() and mode is not None:
        _restore_terminal(mode)
    if tty:
        _flush_terminal_input()
    return code if code >= 0 else 128 - code


def _terminal_mode() -> list | None:
    """The settings of the terminal on stdin, or None without one."""
    try:
        return termios.tcgetattr(0)
    except (termios.error, OSError):
        return None


def _restore_terminal(mode: list) -> None:
    """Put back the terminal settings ``mode`` while launch runs in the
    foreground of that terminal, and drop the input that waits, as
    :func:`_flush_terminal_input` does. A closed terminal takes none."""
    with contextlib.suppress(termios.error, OSError):
        if os.tcgetpgrp(0) == os.getpgrp():
            termios.tcsetattr(0, termios.TCSAFLUSH, mode)


def _flush_terminal_input() -> None:
    """Drop the input that waits on the terminal while launch runs in its
    foreground. A client can send the terminal a query as it quits, and the
    answer arrives after the client stopped reading. The shell would then
    read that answer as typed input."""
    with contextlib.suppress(termios.error, OSError):
        if os.tcgetpgrp(0) == os.getpgrp():
            termios.tcflush(0, termios.TCIFLUSH)


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
