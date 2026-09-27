"""The ``container`` command, the one way gmlx talks to Apple container.

Every call goes through :func:`_run`, so tests replace the binary with a
fake script on ``PATH``. Queries capture their output and time out, because
a wedged container service must not hang a launch. Builds, pulls and the
service start pass their output through to the terminal, since they show
progress and can ask the user a question.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

CONTAINER_MIN = (1, 4, 0)
# Download sizes for the first-run steps, in MB.
KERNEL_DOWNLOAD_MB = 700
NODE_BASE_DOWNLOAD_MB = 80
QUERY_TIMEOUT = 60.0
INSTALL_HINT = ("Install Apple container with `brew install container`, or the "
                "signed installer from https://github.com/apple/container/releases.")
LAUNCH_LABEL = "gmlx.launch"


class ContainerError(RuntimeError):
    """A ``container`` command failed. The message names the command."""


def find() -> str | None:
    return shutil.which("container")


def _run(args: list[str], *, capture: bool = True, timeout: float | None = QUERY_TIMEOUT,
         check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``container ARGS``. With ``capture`` the output is returned as
    text, else it goes to the terminal. ``check`` raises
    :class:`ContainerError` on a nonzero exit."""
    binary = find()
    if binary is None:
        raise ContainerError(f"container is not on PATH. {INSTALL_HINT}")
    argv = [binary, *args]
    try:
        proc = subprocess.run(argv, capture_output=capture, text=True, timeout=timeout,
                              env=env, stdin=subprocess.DEVNULL if capture else None)
    except subprocess.TimeoutExpired:
        raise ContainerError(
            f"`container {' '.join(args[:3])}` gave no answer in {timeout:.0f} s. "
            "The container service may be stuck: try `container system stop` "
            "and `container system start`.") from None
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


def image_info(ref: str) -> ImageInfo | None:
    """The image stored under ``ref``, or None when the store has none."""
    proc = _run(["image", "inspect", ref], check=False)
    if proc.returncode != 0:
        return None
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


def builder_running() -> bool:
    """Whether Apple container's image builder, a virtual machine of its
    own, is running."""
    proc = _run(["builder", "status", "--format", "json"], check=False)
    if proc.returncode != 0:
        return False
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return False
    return any((r.get("status") or {}).get("state") == "running" for r in rows)


def builder_stop() -> None:
    _run(["builder", "stop"], check=False)


def build(context: str, *, file: str, tags: list[str], build_args: dict[str, str] | None = None,
          labels: dict[str, str] | None = None, no_cache: bool = False,
          pull: bool = False) -> None:
    args = ["build", "--file", file]
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
    _run([*args, context], capture=False, timeout=None)


def pull(ref: str) -> None:
    _run(["image", "pull", ref], capture=False, timeout=None)


def tag(source: str, target: str) -> None:
    _run(["image", "tag", source, target])


def image_delete(refs: list[str]) -> None:
    if refs:
        _run(["image", "delete", *refs], check=False)


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
    _run(["stop", "--time", str(timeout), name], check=False,
         timeout=timeout + QUERY_TIMEOUT)


def kill(name: str, *, signal: str | None = None) -> None:
    _run(["kill", *(["--signal", signal] if signal else []), name], check=False)


def delete(name: str) -> None:
    _run(["delete", "--force", name], check=False)


def run_entry_check(ref: str, runtime_dir: str, word: str) -> tuple[int, str]:
    """Run ``gmlx-entry --check WORD`` in the image with no network, and
    return its exit code and its last line of output."""
    proc = _run(["run", "--rm", "--progress", "none", "--network", "none",
                 "--entrypoint", "/opt/gmlx/gmlx-entry",
                 "--mount", f"type=bind,source={runtime_dir},target=/opt/gmlx,readonly",
                 ref, "--check", word], check=False, timeout=120.0)
    lines = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
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
