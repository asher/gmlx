"""The ``container`` command, the one way gmlx talks to Apple container.

Every call goes through :func:`_run`, so tests replace the binary with a
fake script on ``PATH``. Queries capture their output and time out, because
a wedged container service must not hang a launch. Builds, pulls and the
service start pass their output through to the terminal, since they show
progress and can ask the user a question.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

CONTAINER_MIN = (1, 4, 0)
# Download sizes for the first-run steps, in MB.
KERNEL_DOWNLOAD_MB = 700
NODE_BASE_DOWNLOAD_MB = 80
QUERY_TIMEOUT = 60.0
DELETE_TIMEOUT = 600.0
INSTALL_HINT = ("Install Apple container with `brew install container`, or the "
                "signed installer from https://github.com/apple/container/releases.")
LAUNCH_LABEL = "gmlx.launch"


class ContainerError(RuntimeError):
    """A ``container`` command failed. The message names the command."""


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
        raise ContainerError(f"container is not on PATH. {INSTALL_HINT}")
    argv = [binary, *args]
    try:
        proc = subprocess.run(argv, capture_output=capture, text=not keep_cr,
                              timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL if capture else None)
    except subprocess.TimeoutExpired:
        raise ContainerError(
            f"`container {' '.join(args[:3])}` gave no answer in {timeout:.0f} s. "
            "The container service may be stuck: try `container system stop` "
            "and `container system start`.") from None
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


def system_start() -> None:
    """Start the service attached to the terminal, so its kernel install
    question reaches the user."""
    _run(["system", "start"], capture=False, timeout=None)


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


def image_names() -> list[tuple[str, str]]:
    """Every reference in the image store with its digest."""
    rows = _json(["image", "list", "--format", "json"]) or []
    return [((r.get("configuration") or {}).get("name", ""),
             ((r.get("configuration") or {}).get("descriptor") or {}).get("digest", ""))
            for r in rows]


def launch_images() -> tuple[int, int]:
    """The number of images under the reserved launch domain and the bytes
    of their layers, counting each image once."""
    sizes: dict[str, int] = {}
    for row in _json(["image", "list", "--format", "json"]) or []:
        conf = row.get("configuration") or {}
        if not conf.get("name", "").startswith("gmlx.invalid/"):
            continue
        digest = (conf.get("descriptor") or {}).get("digest", "")
        sizes[digest] = sum(v.get("size") or 0 for v in row.get("variants") or []
                            if (v.get("platform") or {}).get("os") != "unknown")
    return len(sizes), sum(sizes.values())


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
    _run([*args, context], capture=False, timeout=None, env=env)


def pull(ref: str) -> None:
    _run(["image", "pull", ref], capture=False, timeout=None)


def tag(source: str, target: str) -> None:
    _run(["image", "tag", source, target])


def image_delete(refs: list[str]) -> None:
    # A delete collects unreferenced content store-wide, which can take
    # minutes when it frees gigabytes.
    if refs:
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
    return [_container(r) for r in _json(["ls", "--all", "--format", "json"]) or []]


def list_launch_containers() -> list[Container]:
    """The containers ``gmlx launch`` started, running or not."""
    return [c for c in containers() if c.labels.get(LAUNCH_LABEL) == "1"]


def stop(name: str, *, timeout: int = 10) -> None:
    # The grace time plus the query time, which query_timeout() shortens.
    _run(["stop", "--time", str(timeout), name], check=False,
         timeout=timeout + (_query_timeout or QUERY_TIMEOUT))


def kill(name: str, *, signal: str | None = None) -> None:
    _run(["kill", *(["--signal", signal] if signal else []), name], check=False)


def delete(name: str) -> None:
    _run(["delete", "--force", name], check=False)


CHECK_TIMEOUT = 120.0


def run_entry_check(ref: str, runtime_dir: str, word: str) -> tuple[int, str]:
    """Run ``gmlx-entry --check WORD`` in the image with no network, and
    return its exit code and its last line of output. The container has a
    name, so a check that gives no answer can be removed."""
    name = f"gmlx-check-{secrets.token_hex(3)}"
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
    out = []
    for row in _json(["volume", "list", "--format", "json"]) or []:
        conf = row.get("configuration") or row
        out.append(Volume(name=conf.get("name", ""), labels=dict(conf.get("labels") or {}),
                          size_bytes=conf.get("sizeInBytes") if (conf.get("options") or {}).get("size") else None,
                          source=conf.get("source", "")))
    return out


def volume_create(name: str, *, size: str) -> None:
    """Create a named volume with the launch label and a size limit. Nothing
    in gmlx deletes a volume."""
    _run(["volume", "create", "--label", f"{LAUNCH_LABEL}=1", "-s", size, name])
