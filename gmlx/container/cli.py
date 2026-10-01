"""The ``container`` command, the one way gmlx talks to Apple container.

Every call goes through :func:`_run`, or :func:`_run_watched` for a build,
so tests replace the binary with a fake script on ``PATH``. Queries capture
their output and time out, because a wedged container service must not hang
a launch. Builds, pulls and the service start pass their output through to
the terminal, since they show progress and can ask the user a question.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pty
import re
import secrets
import select
import shutil
import signal
import subprocess
import sys
import termios
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CONTAINER_MIN = (1, 4, 0)
# Download sizes for the first-run steps, in MB.
KERNEL_DOWNLOAD_MB = 700
NODE_BASE_DOWNLOAD_MB = 80
QUERY_TIMEOUT = 60.0
DELETE_TIMEOUT = 600.0
INSTALL_HINT = ("Apple also publishes a signed installer at "
                "https://github.com/apple/container/releases. Install it with: brew "
                "install container")
LAUNCH_LABEL = "gmlx.launch"
# What npm, curl, git, apt and pip print when they cannot look up a host
# name. In a failed build on a Mac that is most often a VPN that routes all
# traffic, which leaves containers without a network. A bare "Could not
# resolve" would also match Maven and Gradle dependency errors.
NO_NETWORK_WORDS = ("EAI_AGAIN", "ENOTFOUND", "Could not resolve host",
                    "Could not resolve '", "Temporary failure resolving",
                    "Temporary failure in name resolution")
NO_NETWORK_HINT = ("the image build could not reach the network from the container. "
                   "A VPN that routes all traffic blocks that network, so disconnect the "
                   "VPN, or allow local network access in its settings, and launch again.")
# How much of a build's output launch keeps to look for those words.
_WATCH_TAIL = 256 << 10
# The size Apple container gives a volume created without one.
VOLUME_DEFAULT_BYTES = 512 << 30


class ContainerError(RuntimeError):
    """A ``container`` command failed. The message names the command."""


class BuildFailed(ContainerError):
    """A step of ``container build`` failed for a reason other than the
    network. The build's output on the terminal shows the step."""

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode


def find() -> str | None:
    return shutil.which("container")


# Set by query_timeout. None means QUERY_TIMEOUT.
_query_timeout: float | None = None
_QUERY = object()


@contextlib.contextmanager
def query_timeout(seconds: float):
    """Give every query in the block ``seconds`` instead of
    :data:`QUERY_TIMEOUT`, as ``gmlx doctor`` does so it never waits long."""
    global _query_timeout
    saved, _query_timeout = _query_timeout, seconds
    try:
        yield
    finally:
        _query_timeout = saved


class _Memo:
    """The answers :func:`memoized` keeps: the container list, the volume
    list and each image found by reference. Only the thread that started
    the block reads them."""

    def __init__(self):
        self.owner = threading.get_ident()
        self.containers: list | None = None
        self.volumes: list | None = None
        self.images: dict[str, ImageInfo] = {}


_memo: _Memo | None = None


@contextlib.contextmanager
def memoized():
    """Answer a repeated query in the block from its first answer: the
    container list, the volume list, and ``image inspect`` of an image that
    exists. A command that changes what a query reports, such as a build, a
    tag or a stop, drops the answers it affects. A missing image is always
    looked up again, so a build that another launch finishes meanwhile is
    found. :func:`end_memo` ends the block early."""
    global _memo
    _memo = _Memo()
    try:
        yield
    finally:
        _memo = None


def end_memo() -> None:
    """End :func:`memoized` before the session starts, whose own queries
    need current answers."""
    global _memo
    _memo = None


def _memo_here() -> _Memo | None:
    memo = _memo
    return memo if memo is not None and memo.owner == threading.get_ident() else None


def _forget(*, images: bool = False, containers: bool = False,
            volumes: bool = False) -> None:
    memo = _memo_here()
    if memo is None:
        return
    if images:
        memo.images.clear()
    if containers:
        memo.containers = None
    if volumes:
        memo.volumes = None


def _run(args: list[str], *, capture: bool = True, timeout=_QUERY,
         check: bool = True, env: dict | None = None,
         keep_cr: bool = False) -> subprocess.CompletedProcess:
    """Run ``container ARGS``. With ``capture`` the output is returned as
    text, else it goes to the terminal. ``check`` raises
    :class:`ContainerError` on a nonzero exit. A query without a timeout of
    its own gets :data:`QUERY_TIMEOUT`, or the one :func:`query_timeout`
    sets. ``keep_cr`` keeps each carriage return in the text, which text
    mode would turn into a newline."""
    if timeout is _QUERY:
        timeout = _query_timeout if _query_timeout is not None else QUERY_TIMEOUT
    binary = find()
    if binary is None:
        raise ContainerError(f"Apple container is not installed. {INSTALL_HINT}")
    argv = [binary, *args]
    try:
        proc = subprocess.run(argv, capture_output=capture, text=not keep_cr,
                              timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL if capture else None)
    except subprocess.TimeoutExpired:
        raise ContainerError(
            f"`container {' '.join(args[:3])}` gave no answer in {timeout:.0f} s. "
            "The container service may be stuck. Restart it with: container system stop "
            "&& container system start") from None
    except OSError as e:
        # Such as too many open files, or a binary that went away.
        raise ContainerError(f"cannot run `container {' '.join(args[:3])}`: {e}") from None
    if keep_cr and capture:
        proc.stdout = proc.stdout.decode(errors="replace")
        proc.stderr = proc.stderr.decode(errors="replace")
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise ContainerError(
            f"`container {' '.join(args[:3])}` failed (exit {proc.returncode})"
            + (f": {detail.splitlines()[-1]}" if detail else "."))
    return proc


def _run_watched(args: list[str], *, env: dict | None = None) -> None:
    """Run ``container ARGS`` with its output on the terminal, as
    ``_run(capture=False)`` does, and raise :class:`ContainerError` on a
    nonzero exit. ``container build`` draws its progress on standard error,
    so launch reads that stream to recognize a build without a network. On
    a terminal the stream goes through a pseudo-terminal of the same size,
    which follows a resize, so the progress display looks as it does
    without launch."""
    binary = find()
    if binary is None:
        raise ContainerError(f"Apple container is not installed. {INSTALL_HINT}")
    out = sys.stderr
    master = slave = None
    if out.isatty():
        master, slave = pty.openpty()
        _copy_size(out, slave)
        # The terminal turns each newline into CR LF already.
        with contextlib.suppress(termios.error):
            attrs = termios.tcgetattr(slave)
            attrs[1] &= ~termios.ONLCR
            termios.tcsetattr(slave, termios.TCSANOW, attrs)
    try:
        proc = subprocess.Popen([binary, *args], env=env,
                                stderr=slave if slave is not None else subprocess.PIPE)
    except OSError as e:
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)
        raise ContainerError(f"cannot run `container {' '.join(args[:3])}`: {e}") from None
    resize_handler: list = []            # the handler to put back, once installed
    if slave is not None:
        os.close(slave)

        def forward_resize(_signum, _frame):
            # Setting the size on the master still works with the slave closed.
            _copy_size(out, master)
            with contextlib.suppress(OSError):
                proc.send_signal(signal.SIGWINCH)
        with contextlib.suppress(ValueError):       # only the main thread sets handlers
            resize_handler.append(signal.signal(signal.SIGWINCH, forward_resize))
    source = master if master is not None else proc.stderr.fileno()
    tail = b""
    try:
        while True:
            ready, _, _ = select.select([source], [], [], 0.5)
            if not ready:
                # A helper the build started can keep the stream open.
                if proc.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(source, 65536)
            except OSError:              # the pseudo-terminal closed
                chunk = b""
            if not chunk:
                break
            _write_through(out, chunk)
            tail = (tail + chunk)[-_WATCH_TAIL:]
        proc.wait()
    except BaseException:
        # As subprocess.run does, so a Ctrl-C leaves no build behind. The
        # build has its own Ctrl-C, so it gets a moment to end by itself.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=1.0)
        proc.kill()
        proc.wait()
        raise
    finally:
        # Before the master closes, so the handler never sizes a reused fd.
        if resize_handler:
            signal.signal(signal.SIGWINCH, resize_handler[0] or signal.SIG_DFL)
        if master is not None:
            os.close(master)
        elif proc.stderr is not None:
            proc.stderr.close()
    if proc.returncode != 0:
        text = tail.decode(errors="replace")
        if any(word in text for word in NO_NETWORK_WORDS):
            raise ContainerError(NO_NETWORK_HINT)
        raise BuildFailed(
            f"`container {' '.join(args[:3])}` failed (exit {proc.returncode}).",
            proc.returncode)


def _copy_size(out, fd: int) -> None:
    """Give the pseudo-terminal at ``fd`` the size of the terminal ``out``."""
    with contextlib.suppress(OSError, ValueError):
        size = fcntl.ioctl(out.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


def _write_through(out, chunk: bytes) -> None:
    out.flush()
    buffer = getattr(out, "buffer", None)
    if buffer is not None:
        buffer.write(chunk)
        buffer.flush()
    else:
        out.write(chunk.decode(errors="replace"))
        out.flush()


def _json(args: list[str]):
    out = _run(args).stdout
    try:
        return json.loads(out or "null")
    except json.JSONDecodeError:
        raise ContainerError(
            f"`container {' '.join(args[:3])}` printed output that is not JSON.") from None


def version() -> tuple[int, int, int] | None:
    """The CLI version, or None when the output has none."""
    out = _run(["--version"]).stdout
    m = re.search(r"version (\d+)\.(\d+)\.(\d+)", out)
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def system_running() -> bool:
    proc = _run(["system", "status"], check=False)
    return proc.returncode == 0 and bool(
        re.search(r"^status\s+running\s*$", proc.stdout, re.M))


def app_root() -> Path:
    """The folder where Apple container keeps its data, which
    ``CONTAINER_APP_ROOT`` moves. Apple container finds the Application
    Support folder from the account's home, not from ``HOME``."""
    root = os.environ.get("CONTAINER_APP_ROOT")
    if root:
        return Path(os.path.abspath(root))
    return account_home() / "Library" / "Application Support" / "com.apple.container"


def account_home() -> Path:
    """The home folder of the user database entry, or ``HOME`` without one."""
    import pwd

    try:
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except KeyError:
        return Path.home()


def kernel_installed() -> bool:
    """Whether the default Linux kernel is installed. Without it, ``container
    system start`` asks whether to install one and waits for the answer."""
    return (app_root() / "kernels" / "default.kernel-arm64").is_file()


def system_start(*, install_kernel: bool = True) -> None:
    """Start the service attached to the terminal, so its kernel install
    question reaches the user. Without ``install_kernel`` the start never
    asks, which a start with no terminal needs."""
    _forget(images=True, containers=True, volumes=True)
    _run(["system", "start", *([] if install_kernel else ["--disable-kernel-install"])],
         capture=False, timeout=None)


def _parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", text)
    if not m:
        return None
    return datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


@dataclass
class ImageInfo:
    """What launch needs from ``container image inspect``. The command
    fields come from the linux/arm64 variant, and stay None without one."""
    name: str
    digest: str                       # sha256:<hex> of the stored descriptor
    architectures: list[str]          # "os/arch" of every runnable variant
    arm64: bool
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None
    workdir: str | None = None
    created: datetime | None = None


def _image_info(entry: dict) -> ImageInfo:
    conf = entry.get("configuration") or {}
    arches, arm = [], None
    for variant in entry.get("variants") or []:
        plat = variant.get("platform") or {}
        os_, arch = plat.get("os"), plat.get("architecture")
        if not os_ or os_ == "unknown":
            continue                  # attestation manifests
        arches.append(f"{os_}/{arch}")
        if os_ == "linux" and arch == "arm64" and arm is None:
            arm = variant
    info = ImageInfo(name=conf.get("name", ""),
                     digest=(conf.get("descriptor") or {}).get("digest", ""),
                     architectures=arches, arm64=arm is not None)
    if arm is not None:
        cfg = (arm.get("config") or {})
        inner = cfg.get("config") or {}
        info.entrypoint = inner.get("Entrypoint") or None
        info.cmd = inner.get("Cmd") or None
        info.workdir = inner.get("WorkingDir") or None
        info.created = _parse_time(cfg.get("created"))
    return info


# What `container image inspect` prints for a reference the store does not hold.
_IMAGE_NOT_FOUND = re.compile(r"\bimage not found:")


def image_info(ref: str) -> ImageInfo | None:
    """The image stored under ``ref``, or None when the store has none. Any
    other failure raises, so a service error never looks like a missing
    image that launch would pull again."""
    memo = _memo_here()
    if memo is not None and ref in memo.images:
        return memo.images[ref]
    info = _inspect(ref)
    if memo is not None and info is not None:
        memo.images[ref] = info
    return info


def _inspect(ref: str) -> ImageInfo | None:
    proc = _run(["image", "inspect", ref], check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if _IMAGE_NOT_FOUND.search(detail):
            return None
        raise ContainerError(
            f"`container image inspect {ref}` failed (exit {proc.returncode})"
            + (f": {detail.splitlines()[-1]}" if detail else "."))
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        raise ContainerError("`container image inspect` printed output that is not JSON.") from None
    return _image_info(data[0]) if data else None


@dataclass
class StoredImage:
    """One reference in the image store."""
    name: str
    digest: str
    size: int                         # the bytes of its runnable variants


def image_list() -> list[StoredImage]:
    """Every reference in the image store."""
    out = []
    for row in _json(["image", "list", "--format", "json"]) or []:
        conf = row.get("configuration") or {}
        out.append(StoredImage(
            name=conf.get("name", ""), digest=(conf.get("descriptor") or {}).get("digest", ""),
            size=sum(v.get("size") or 0 for v in row.get("variants") or []
                     if (v.get("platform") or {}).get("os") != "unknown")))
    return out


def image_names() -> list[tuple[str, str]]:
    """Every reference in the image store with its digest."""
    return [(image.name, image.digest) for image in image_list()]


@dataclass
class Builder:
    """Apple container's image builder, a virtual machine of its own, from
    ``container builder status``."""
    state: str                        # "running", "stopping", "stopped" and so on
    cpus: int | None = None
    memory_bytes: int | None = None
    ssh: bool = False
    # When the builder last started, as whole-second UTC text. A new start
    # gives a new value, so it identifies one start of the builder.
    started: str | None = None
    # The builder's BUILDKIT_COLORS and NO_COLOR entries, which 1.4.1
    # compares with the build command's own environment.
    colors: tuple[str, ...] = ()


_BUILDER_ENV = ("BUILDKIT_COLORS=", "NO_COLOR=")


def builder() -> Builder | None:
    """The image builder, or None when there is none."""
    rows = _json(["builder", "status", "--format", "json"]) or []
    for row in rows:
        conf = row.get("configuration") or {}
        res = conf.get("resources") or {}
        status = row.get("status") or {}
        env = (conf.get("initProcess") or {}).get("environment") or []
        state = status.get("state", "") if isinstance(status, dict) else status
        started = status.get("startedDate") if isinstance(status, dict) else None
        return Builder(state=str(state), cpus=res.get("cpus"),
                       memory_bytes=res.get("memoryInBytes"), ssh=bool(conf.get("ssh")),
                       started=str(started) if started else None,
                       colors=tuple(sorted(e for e in env if isinstance(e, str)
                                           and e.startswith(_BUILDER_ENV))))
    return None


def builder_build_args(running: Builder) -> list[str]:
    """The ``container build`` options that match a running builder. A build
    with other settings makes 1.4.1 stop, delete and create the builder
    again, which ends any build that runs on it. ``--ssh`` is never passed,
    since it would give the Containerfile every key in the Mac's SSH agent,
    so a builder that forwards the agent cannot be matched and launch
    refuses to build on it."""
    args = []
    if running.cpus:
        args += ["--cpus", str(running.cpus)]
    if running.memory_bytes:
        # The builder takes memory in whole MiB.
        args += ["--memory", f"{running.memory_bytes >> 20}M"]
    return args


def builder_build_env(running: Builder) -> dict[str, str]:
    """The environment for ``container build`` that matches a running
    builder. 1.4.1 gives the builder ``BUILDKIT_COLORS`` from the build
    command's environment, and ``NO_COLOR=true`` when that environment sets
    ``NO_COLOR``, and it creates the builder again when these differ."""
    env = {k: v for k, v in os.environ.items() if k not in ("BUILDKIT_COLORS", "NO_COLOR")}
    for entry in running.colors:
        name, _, value = entry.partition("=")
        env[name] = value
    return env


def builder_stop() -> None:
    _run(["builder", "stop"])


def build(context: str, *, file: str, tags: list[str], build_args: dict[str, str] | None = None,
          labels: dict[str, str] | None = None, no_cache: bool = False,
          pull: bool = False, builder_args: list[str] | None = None,
          env: dict[str, str] | None = None) -> None:
    args = ["build", *(builder_args or []), "--file", file]
    for tag in tags:
        args += ["--tag", tag]
    for key, value in (build_args or {}).items():
        args += ["--build-arg", f"{key}={value}"]
    for key, value in (labels or {}).items():
        args += ["--label", f"{key}={value}"]
    if no_cache:
        args.append("--no-cache")
    if pull:
        args.append("--pull")
    # A build takes minutes, in which other launches start containers.
    _forget(images=True, containers=True)
    _run_watched([*args, context], env=env)


def pull(ref: str) -> None:
    _forget(images=True)
    _run(["image", "pull", ref], capture=False, timeout=None)


def tag(source: str, target: str) -> None:
    _forget(images=True)
    _run(["image", "tag", source, target])


def image_delete(refs: list[str]) -> None:
    # A delete collects unreferenced content store-wide, which can take
    # minutes when it frees gigabytes.
    if refs:
        _forget(images=True)
        _run(["image", "delete", *refs], check=False, timeout=DELETE_TIMEOUT)


@dataclass
class Container:
    """One row of ``container ls --all``."""
    name: str
    state: str
    labels: dict[str, str]
    image: str                        # the reference the container was run with
    image_digest: str
    volumes: list[str] = field(default_factory=list)   # named volumes it mounts
    memory_bytes: int | None = None


def _container(row: dict) -> Container:
    conf = row.get("configuration") or {}
    image = conf.get("image") or {}
    volumes = []
    for mount in conf.get("mounts") or []:
        kind = mount.get("type") or {}
        if isinstance(kind, dict) and "volume" in kind:
            volumes.append((kind["volume"] or {}).get("name", ""))
    return Container(
        name=row.get("id") or conf.get("id", ""),
        state=(row.get("status") or {}).get("state", "") if isinstance(
            row.get("status"), dict) else str(row.get("status", "")),
        labels=dict(conf.get("labels") or {}),
        image=image.get("reference", ""),
        image_digest=(image.get("descriptor") or {}).get("digest", ""),
        volumes=volumes,
        memory_bytes=(conf.get("resources") or {}).get("memoryInBytes"))


def containers() -> list[Container]:
    memo = _memo_here()
    if memo is not None and memo.containers is not None:
        return list(memo.containers)
    found = [_container(r) for r in _json(["ls", "--all", "--format", "json"]) or []]
    if memo is not None:
        memo.containers = found
    return list(found)


def list_launch_containers() -> list[Container]:
    """The containers ``gmlx launch`` started, running or not."""
    return [c for c in containers() if c.labels.get(LAUNCH_LABEL) == "1"]


def stop(name: str, *, timeout: int = 10) -> None:
    _forget(containers=True)
    # The grace time plus the query time, which query_timeout() shortens.
    _run(["stop", "--time", str(timeout), name], check=False,
         timeout=timeout + (_query_timeout or QUERY_TIMEOUT))


def kill(name: str, *, signal: str | None = None) -> None:
    _forget(containers=True)
    _run(["kill", *(["--signal", signal] if signal else []), name], check=False)


def delete(name: str) -> None:
    _forget(containers=True)
    _run(["delete", "--force", name], check=False)


CHECK_TIMEOUT = 120.0


def run_entry_check(ref: str, runtime_dir: str, word: str) -> tuple[int, str]:
    """Run ``gmlx-entry --check WORD`` in the image with no network, and
    return its exit code and its last line of output. The container has a
    name, so a check that gives no answer can be removed."""
    name = f"gmlx-check-{secrets.token_hex(3)}"
    _forget(containers=True)
    try:
        proc = _run(["run", "--rm", "--name", name, "--progress", "none",
                     "--network", "none", "--entrypoint", "/opt/gmlx/gmlx-entry",
                     "--mount", f"type=bind,source={runtime_dir},target=/opt/gmlx,readonly",
                     ref, "--check", word], check=False, timeout=CHECK_TIMEOUT,
                    keep_cr=True)
    except ContainerError:
        try:
            delete(name)
        except ContainerError:
            pass
        raise
    # Split on newlines only, so a carriage return inside a message, such
    # as one in a command name, keeps the line whole.
    text = proc.stderr if proc.stderr.strip() else proc.stdout
    lines = [line for line in text.split("\n") if line.strip()]
    return proc.returncode, lines[-1] if lines else ""


def exec_argv(name: str, command: list[str], *, tty: bool,
              cwd: str | None = None) -> list[str]:
    """The argv that replaces launch with a process in a running container."""
    binary = find() or "container"
    return [binary, "exec", "-i", *(["-t"] if tty else []),
            *(["--cwd", cwd] if cwd else []), name, *command]


@dataclass
class Volume:
    name: str
    labels: dict[str, str]
    size_bytes: int | None            # None when created without a size
    source: str                       # the disk image on the Mac


def volume_list() -> list[Volume]:
    memo = _memo_here()
    if memo is not None and memo.volumes is not None:
        return list(memo.volumes)
    out = []
    for row in _json(["volume", "list", "--format", "json"]) or []:
        conf = row.get("configuration") or row
        out.append(Volume(name=conf.get("name", ""), labels=dict(conf.get("labels") or {}),
                          size_bytes=conf.get("sizeInBytes") if (conf.get("options") or {}).get("size") else None,
                          source=conf.get("source", "")))
    if memo is not None:
        memo.volumes = out
    return list(out)


def volume_create(name: str, *, size: str) -> None:
    """Create a named volume with the launch label and a size limit. Nothing
    in gmlx deletes a volume."""
    _forget(volumes=True)
    _run(["volume", "create", "--label", f"{LAUNCH_LABEL}=1", "-s", size, name])
