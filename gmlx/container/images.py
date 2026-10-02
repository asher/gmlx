"""The image a container launch runs: which one, building or pulling it, and
the checks before its first run.

A client runs, in order of precedence, the ``--image`` reference, its
``image:`` reference, an image built from its ``build:`` Containerfile, or
the shipped image built from ``files/Containerfile``. gmlx tags every image
it builds under the reserved ``gmlx.invalid`` domain, so a missing tag can
never be pulled from a registry under a squatted name. Every run names its
image by content digest, which :func:`ensure_image` adds to the local store
as a reference, so the run needs no network and runs exactly the image the
checks approved.

Builds and tag changes hold an exclusive ``flock`` on the repository's lock
file. A ``build:`` image built on a shipped ``:base`` holds those base
locks shared while it builds, so no base moves under it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gmlx import DOCS_URL
from gmlx.config import (LAUNCH_CLIENTS, LaunchCfg, LaunchClientCfg, LaunchContainerCfg,
                         agent_name, config_key, target_label)

from . import cli, ignore, notices
from .cli import ContainerError, ImageInfo
from .state import FileLock, LockHeld, canonical, data_path, images_dir, path_inside
from .text import printable

DOMAIN = "gmlx.invalid"
SHIPPED_CONTAINERFILE = Path(__file__).parent / "files" / "Containerfile"
# `container build` refuses a Containerfile of this size or more.
CONTAINERFILE_MAX = 16 * 1024
DEFAULT_CONTAINERFILES = ("Containerfile", "Dockerfile")
AGE_NOTE_DAYS = 30
# The most context files a rebuild reason can name from its record, and the
# most it names in its line.
MANIFEST_MAX = 20_000
REASON_NAMES = 3
LAUNCH_LABELS = {cli.LAUNCH_LABEL: "1"}

# The command each shipped image installs for its client.
CLIENT_BINARY = {
    "opencode": "opencode", "pi": "pi", "omp": "omp", "hermes": "hermes",
    "goose": "goose", "claude-code": "claude", "aichat": "aichat",
    "elia": "elia", "open-webui": "open-webui", "dsh": "dsh",
}
# The shipped stage that installs the tool of each agent runtime, the tool
# it installs, and the name messages give the image. Every runtime agent
# shares the stage's image, so the stage, not the agent, keys its claims.
RUNTIME_STAGES = {"python": "runtime-python"}
RUNTIME_BINARY = {"runtime-python": "uv"}
RUNTIME_LABEL = {"runtime-python": "Python runtime"}


def stage_for(launch_cfg: LaunchCfg, key: str) -> str | None:
    """The shipped stage a launch target builds on: the client's own stage,
    the runtime stage of a runtime agent, or None for an agent that brings
    its own image."""
    if key in LAUNCH_CLIENTS:
        return key
    runtime = launch_cfg.agent(key).runtime
    return RUNTIME_STAGES[runtime] if runtime else None


def _label(client: str) -> str:
    """The name messages give a target or a shipped stage."""
    return RUNTIME_LABEL.get(client) or target_label(client)

Say = Callable[[str], None]


def _shown(path: str | os.PathLike) -> str:
    """A path for a message: ``~`` for the home folder, and any character
    :func:`printable` escapes written as an escape, since a file name can
    hold one."""
    text = str(path)
    home = os.path.expanduser("~")
    if text == home or text.startswith(home.rstrip("/") + "/"):
        text = "~" + text[len(home.rstrip("/")):]
    return printable(text)


def _read_regular(path: Path, limit: int) -> bytes:
    """A regular file of less than ``limit`` bytes. Never waits on a named
    pipe or a device."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise ImageError(f"cannot read {_shown(path)} ({e.strerror}).") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ImageError(f"{_shown(path)} is not a regular file.")
        data = b""
        while len(data) < limit:
            chunk = os.read(fd, limit - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) >= limit:
        raise ImageError(f"{_shown(path)} is {limit} bytes or more, which Apple container "
                         "cannot build. Move some of its steps into a script it copies.")
    return data


def _write_private(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` through a new temporary file beside it,
    created with ``O_EXCL`` and ``O_NOFOLLOW`` at mode 0600, so a link
    planted at either name is never written through. The temporary file is
    removed when the write fails."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _say(line: str) -> None:
    print(printable(line), flush=True)  # before the build's own output


class ImageError(ContainerError):
    """The configured image cannot run. The message says what to change."""


def recipe_repo(client: str) -> str:
    return f"{DOMAIN}/launch-{client}"


def base_ref(client: str) -> str:
    return f"{recipe_repo(client)}:base"


def build_repo(client: str) -> str:
    return f"{DOMAIN}/launch-{client}-build"


def repository_of(ref: str) -> str:
    """``ref`` without its digest and tag."""
    ref = ref.split("@", 1)[0]
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    return ref[:colon] if colon > slash else ref


_BASE_SCAN = re.compile(r"gmlx\.invalid/[^\s\"'\\]*")
_BASE_FORM = re.compile(r"gmlx\.invalid/launch-([a-z0-9-]+):base")


def base_refs_in(text: str) -> list[str]:
    """The clients and runtime stages whose ``:base`` a Containerfile names,
    in first-use order. Any other ``gmlx.invalid`` reference is refused,
    because launch deletes every other tag there when a newer build
    replaces it."""
    clients: list[str] = []
    for ref in _BASE_SCAN.findall(text):
        m = _BASE_FORM.fullmatch(ref)
        if m is None or (m[1] not in LAUNCH_CLIENTS and m[1] not in RUNTIME_STAGES.values()):
            raise ImageError(
                f"the Containerfile names {ref}, but launch deletes that tag when it builds "
                "again. Name the base of a client or of a runtime, such as "
                "gmlx.invalid/launch-claude-code:base or "
                "gmlx.invalid/launch-runtime-python:base.")
        if m[1] not in clients:
            clients.append(m[1])
    return clients


@dataclass
class ImagePlan:
    """The image one launch needs, before anything is built or pulled."""
    kind: str                         # "shipped", "build" or "image"
    client: str
    ref: str | None = None            # the image: or --image reference
    packages: list[str] = field(default_factory=list)
    containerfile: Path | None = None
    context: Path | None = None
    bases: list[str] = field(default_factory=list)   # clients whose :base it names
    base_packages: dict[str, list[str]] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


def _containerfile(build: str, client: str) -> tuple[Path, Path]:
    path = Path(build).expanduser()
    if not path.is_absolute():
        raise ImageError(
            f"{config_key(client, 'build')} is {build!r}. Give an absolute "
            "path or one that starts with ~, since launch runs from many folders.")
    # Anything at the name counts, so a named pipe reaches the regular-file
    # check instead of passing for a missing file.
    if path.is_dir():
        for name in DEFAULT_CONTAINERFILES:
            if os.path.lexists(path / name) and not (path / name).is_dir():
                return path / name, path
        raise ImageError(f"{path} holds no Containerfile or Dockerfile.")
    if os.path.lexists(path):
        return path, path.parent
    raise ImageError(f"{config_key(client, 'build')} names {path}, which does not exist.")


def resolve_image(client: str, cfg: LaunchClientCfg, container: LaunchContainerCfg, *,
                  stage: str | None = None, image_override: str | None = None,
                  writable: Sequence[str] = ()) -> ImagePlan:
    """Decide which image the target ``client`` runs, and refuse a ``build:``
    Containerfile that cannot build. ``stage`` is the shipped stage the
    target builds on when it names no image, which is the client's own
    stage by default (:func:`stage_for`). ``writable`` holds the Mac
    folders the session shares read-write. A ``build:`` folder that
    overlaps one, a folder an earlier launch shared read-write, or the
    private homes is refused, because a client could change what the next
    build runs, and a build has the network and the builder."""
    if stage is None and client in LAUNCH_CLIENTS:
        stage = client
    if image_override:
        notices = []
        if cfg.packages:
            notices.append(f"[launch] --image replaces the {_label(client)} image, so its "
                           "packages list is not used.")
        return ImagePlan("image", client, ref=image_override, notices=notices)
    if cfg.image:
        return ImagePlan("image", client, ref=cfg.image)
    if cfg.build:
        file, context = _containerfile(cfg.build, client)
        _refuse_writable_build(client, file, context, writable)
        text = _read_regular(file, CONTAINERFILE_MAX).decode("utf-8", errors="replace")
        bases = base_refs_in(text)
        if cfg.packages and client not in bases:
            raise ImageError(
                f"the {_label(client)} packages list reaches a build: image only through its "
                f"own base. Start {file} with FROM {base_ref(client)}, or move the packages "
                "into the Containerfile.")
        # A runtime base installs no packages list.
        return ImagePlan("build", client, containerfile=file, context=context,
                         bases=bases, packages=list(cfg.packages),
                         base_packages={b: (container.for_client(b).packages
                                            if b in LAUNCH_CLIENTS else []) for b in bases})
    if stage is None:
        raise ImageError(f"{_label(client)} names no image to run. Set "
                         f"{config_key(client, 'image')}, {config_key(client, 'build')} or "
                         f"{config_key(client, 'runtime')}.")
    return ImagePlan("shipped", stage, packages=list(cfg.packages))


def _refuse_writable_build(client: str, file: Path, context: Path,
                           writable: Sequence[str]) -> None:
    from .settings import shared_history

    # The form macOS gives a path, so a /System/Volumes/Data alias of a share
    # or of the build folder still compares equal.
    real_file, real_context = canonical(file), canonical(context)

    def overlaps(folder: str) -> bool:
        return (path_inside(real_file, folder) or path_inside(real_context, folder)
                or path_inside(folder, real_context))
    for share in (canonical(w) for w in writable):
        if overlaps(share):
            raise ImageError(
                f"the client could change the {_label(client)} build: folder "
                f"{_shown(real_context)} through the read-write share {_shown(share)}. "
                "Move the build folder out of the share, or share it read-only.")
    data = canonical(data_path())
    if overlaps(data):
        raise ImageError(
            f"the {_label(client)} build: folder {_shown(real_context)} overlaps "
            f"{_shown(data)}, "
            "where the clients' private homes are, so a client could change it. Move "
            "the build folder out of it.")
    for share in (canonical(w) for w in shared_history()):
        if overlaps(share):
            raise ImageError(
                f"an earlier launch shared {_shown(share)} read-write, so a client may have "
                f"changed the {_label(client)} build: folder {_shown(real_context)}. Move "
                "the build "
                "folder to a folder no launch has shared read-write.")


_FROM_LINE = re.compile(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", re.IGNORECASE)


def _stages(text: str) -> tuple[list[str], dict[str, tuple[str, list[str]]], list[str]]:
    """The shipped Containerfile in parts: the lines before the first
    stage, each named stage with the stage or image it starts from and its
    lines, and the lines of the last stage, which the CLIENT build argument
    points at its client's stage. Comments and blank lines are left out, so
    they never change a hash."""
    preamble: list[str] = []
    named: dict[str, tuple[str, list[str]]] = {}
    last: list[str] = []
    current = preamble
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _FROM_LINE.match(line)
        if m:
            current = [line]
            if m[2]:
                named[m[2]] = (m[1], current)
            else:
                last = current
            continue
        current.append(line)
    return preamble, named, last


def shipped_recipe(client: str) -> dict[str, str]:
    """The sha256 of the two parts of a client's recipe: ``common``, the
    lines outside the client stages, which every client builds on, and
    ``client``, the client's own stage."""
    preamble, named, last = _stages(SHIPPED_CONTAINERFILE.read_text())
    chain: list[str] = []
    stage = client
    while stage in named and stage not in chain:
        chain.append(stage)
        stage = named[stage][0]
    shared = [*preamble, *(line for s in reversed(chain[1:]) for line in named[s][1]), *last]
    own = named[client][1] if client in named else []
    return {"common": hashlib.sha256("\n".join(shared).encode()).hexdigest(),
            "client": hashlib.sha256("\n".join(own).encode()).hexdigest()}


def shipped_version(client: str) -> str | None:
    """The client version the shipped recipe pins, from the VERSION at the
    top of the client's stage."""
    _, named, _ = _stages(SHIPPED_CONTAINERFILE.read_text())
    for line in named.get(client, ("", []))[1]:
        m = re.match(r"ARG\s+VERSION=(\S+)\s*$", line)
        if m:
            return m[1]
    return None


def shipped_hash(client: str, packages: list[str]) -> str:
    """The tag of a shipped image: the client's recipe and the build
    arguments, so a change to another client's stage leaves it alone."""
    h = hashlib.sha256(json.dumps(shipped_recipe(client), sort_keys=True).encode())
    h.update(json.dumps(_shipped_args(client, packages), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _shipped_args(client: str, packages: list[str]) -> dict[str, str]:
    # Sorted, so the same packages in another order use the same image.
    return {"CLIENT": client, "EXTRA_PACKAGES": " ".join(sorted(set(packages)))}


def _node_base() -> str:
    """The base image the shipped Containerfile names."""
    m = re.search(r"^FROM\s+(\S+)", SHIPPED_CONTAINERFILE.read_text(), re.M)
    return m[1] if m else ""


def _shown_base(ref: str) -> str:
    """A base image as people write it, such as node:22-bookworm-slim,
    without the registry of Docker Hub and the digest."""
    name = ref.split("@", 1)[0]
    for prefix in ("docker.io/library/", "docker.io/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _registry_of(ref: str) -> str:
    """The registry server of an image reference, docker.io when it names
    none."""
    first, sep, _ = ref.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        return first
    return "docker.io"


def shipped_tag(client: str, packages: list[str]) -> str:
    return f"{recipe_repo(client)}:{shipped_hash(client, packages)}"


def context_files(context: Path, matcher: ignore.Matcher | None):
    """``(relative path, lstat)`` of every file a build of ``context`` can
    see, sorted. The ``.git`` folder at the root is left out. A folder or a
    file that cannot be read is listed with a stat of None, so the hash
    still sees it and the build itself reports the error."""
    found: list = []

    def unreadable(error: OSError) -> None:
        if error.filename:
            rel = os.path.relpath(error.filename, context)
            if matcher is None or not matcher.excluded(rel):
                found.append((rel, None))

    for root, dirs, files in os.walk(context, onerror=unreadable):
        rel_root = os.path.relpath(root, context)
        rel_root = "" if rel_root == "." else rel_root + "/"
        if not rel_root:
            dirs[:] = [d for d in dirs if d != ".git"]
        if matcher is not None:
            dirs[:] = [d for d in dirs if not matcher.excludes_all_below(rel_root + d)]
        dirs.sort()
        for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            rel = rel_root + name
            if matcher is not None and matcher.excluded(rel):
                continue
            try:
                found.append((rel, os.lstat(os.path.join(root, name))))
            except FileNotFoundError:
                continue
            except OSError:
                found.append((rel, None))
    found.sort(key=lambda item: item[0])
    return found


def _context_entries(plan: ImagePlan, say: Say) -> dict[str, str]:
    """What the hash records of each context file the build sees: its size,
    modification time, mode and link target, in path order."""
    assert plan.containerfile is not None and plan.context is not None
    matcher, notice = ignore.load(plan.containerfile, plan.context)
    if notice:
        say(notice)
    try:
        files = context_files(plan.context, matcher)
    except ignore.TooMuchWork as e:
        # Every file counts then, so a rebuild is never missed.
        say(f"[launch] {_shown(ignore.ignore_file(plan.containerfile, plan.context))}: "
            f"{e}, so every context file counts toward the rebuild check.")
        files = context_files(plan.context, None)
    entries: dict[str, str] = {}
    for rel, st in files:
        if st is None:
            entries[rel] = "unreadable"
            continue
        link = os.readlink(plan.context / rel) if stat.S_ISLNK(st.st_mode) else ""
        entries[rel] = f"{st.st_size}\0{st.st_mtime_ns}\0{st.st_mode}\0{link}"
    return entries


def _hash_of(containerfile: bytes, base_digests: dict[str, str],
             entries: dict[str, str]) -> str:
    h = hashlib.sha256(containerfile)
    h.update(json.dumps(sorted(base_digests.items())).encode())
    for rel, entry in entries.items():
        h.update(f"{rel}\0{entry}\n".encode())
    return h.hexdigest()[:16]


def build_hash(plan: ImagePlan, base_digests: dict[str, str], say: Say) -> str:
    """The tag of a ``build:`` image: its Containerfile, the path, size and
    modification time of every context file the build sees, and the digest
    of each base it names."""
    assert plan.containerfile is not None
    return _hash_of(_read_regular(plan.containerfile, CONTAINERFILE_MAX), base_digests,
                    _context_entries(plan, say))


def _manifest_path(repo: str, kind: str = "context") -> Path:
    """The record of what the last build of ``repo`` used: its build
    context, or for a shipped image its recipe."""
    return images_dir() / (re.sub(r"[^A-Za-z0-9_.-]+", "_", repo) + f".{kind}.json")


def _node_base_path() -> Path:
    """Holds the base image of the last shipped build that completed."""
    return images_dir() / "node-base"


def _recipe_manifest(client: str, packages: list[str]) -> dict:
    return {**shipped_recipe(client), "version": shipped_version(client),
            "packages": sorted(set(packages))}


def _recipe_reason(repo: str, client: str, new: dict) -> str | None:
    """The line that says why the shipped image of ``client`` is built
    again: its recipe or its packages changed since the last build, or that
    image is gone from the image store. None for a first build."""
    try:
        old = json.loads(_manifest_path(repo, "recipe").read_text())
    except (OSError, ValueError, RecursionError):
        old = None
    if not isinstance(old, dict):
        # An earlier gmlx kept no recipe, but its pins are on record.
        try:
            records = json.loads(_records_path().read_text())
        except (OSError, ValueError, RecursionError):
            records = {}
        if isinstance(records, dict) and any(repository_of(ref) == repo for ref in records):
            return f"[launch] rebuilding because gmlx updated the {_label(client)} recipe"
        return None
    changed: list[str] = []
    moved = RUNTIME_BINARY.get(client, _label(client))
    if old.get("client") != new["client"]:
        if old.get("version") not in (None, new["version"]):
            changed.append(f"gmlx moved {moved} from {old['version']} to {new['version']}")
        else:
            changed.append(f"gmlx updated the {_label(client)} recipe")
    if old.get("common") != new["common"]:
        changed.append(f"gmlx updated the layers that {_label(client)} shares with other "
                       "clients")
    if old.get("packages") != new["packages"]:
        changed.append("the packages list changed")
    if not changed:
        return (f"[launch] rebuilding because the {_label(client)} image is no longer in the "
                "image store")
    return f"[launch] rebuilding because {_join(changed)}"


def _manifest(plan: ImagePlan, containerfile: bytes, base_digests: dict[str, str],
              entries: dict[str, str]) -> dict:
    return {"containerfile": str(plan.containerfile), "context": str(plan.context),
            "containerfile_sha256": hashlib.sha256(containerfile).hexdigest(),
            "bases": dict(sorted(base_digests.items())),
            "files": entries if len(entries) <= MANIFEST_MAX else None}


def _join(names: list[str]) -> str:
    if len(names) > REASON_NAMES:
        return f"{', '.join(names[:REASON_NAMES])} and {len(names) - REASON_NAMES} more"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def rebuild_reason(old: dict | None, new: dict) -> str | None:
    """The line that names what changed since the last build of the
    repository, or None when the record of that build is missing or shows
    no change, such as after the image was deleted."""
    if not isinstance(old, dict):
        return None
    if (old.get("containerfile"), old.get("context")) != (new["containerfile"], new["context"]):
        return f"[launch] rebuilding because build: now names {_shown(new['containerfile'])}"
    changed: list[str] = []
    if old.get("containerfile_sha256") != new["containerfile_sha256"]:
        changed.append(_shown(new["containerfile"]))
    old_files, new_files = old.get("files"), new["files"]
    context = Path(new["context"])
    if isinstance(old_files, dict) and new_files is not None:
        for rel in sorted(set(old_files) | set(new_files)):
            if old_files.get(rel) != new_files.get(rel):
                path = _shown(context / rel)
                if path not in changed:
                    changed.append(path)
    elif old_files != new_files:
        changed.append(f"files in {_shown(context)}")
    old_bases = old.get("bases") if isinstance(old.get("bases"), dict) else {}
    for base in sorted(set(old_bases) | set(new["bases"])):
        if old_bases.get(base) != new["bases"].get(base):
            changed.append(f"the {base} base image")
    if not changed:
        return None
    return f"[launch] rebuilding because {_join(changed)} changed"


@dataclass
class ReadyImage:
    """An image present in the local store, pinned by digest."""
    kind: str
    tag: str                          # the readable reference
    info: ImageInfo
    run_ref: str                      # <repository>@sha256:<digest>
    action: str                       # "built", "pulled" or "found"
    client: str = ""
    fetched: datetime | None = None   # when launch last built or pulled it


def _lock_path(repo: str) -> Path:
    return images_dir() / (re.sub(r"[^A-Za-z0-9_.-]+", "_", repo) + ".lock")


def repo_lock(repo: str, *, shared: bool = False, say: Say | None = None) -> FileLock:
    def waiting():
        if say is not None:
            say(f"[launch] waiting for another launch to finish with {repo}")
    return FileLock(_lock_path(repo), shared=shared, on_wait=waiting)


def _records_path() -> Path:
    return images_dir() / "references.json"


def _update_records(fn: Callable[[dict], None]) -> dict:
    """Read, change and write the record of the references launch added,
    under its own lock. Each entry maps a reference to the clients that use
    it and the process IDs of the launches that pinned it."""
    with FileLock(images_dir() / "references.lock"):
        path = _records_path()
        try:
            records = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            records = {}
        if not isinstance(records, dict):
            records = {}
        for ref, entry in list(records.items()):
            if isinstance(entry, list):   # the older form: only the clients
                records[ref] = {"clients": sorted(str(c) for c in entry), "pids": []}
            elif not isinstance(entry, dict):
                del records[ref]
        fn(records)
        _write_private(path, json.dumps(records, indent=1, sort_keys=True))
        return records


def _owners(entry) -> set[str]:
    if isinstance(entry, dict):
        return set(entry.get("clients") or [])
    return set()


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pids(entry) -> set[int]:
    """The launches that pinned the reference and still run."""
    raw = entry.get("pids") if isinstance(entry, dict) else None
    return {p for p in raw or [] if _alive(p)}


def _pin(source: str, repo: str, info: ImageInfo, client: str, *, fetched: bool,
         pulled: bool = False) -> str:
    """Add ``<repo>@<digest>`` to the local store, and note it as the digest
    reference of ``source``, with ``fetched`` when launch built or pulled it
    just now, and ``pulled`` when launch pulled it. The reference is
    recorded as launch's only when launch added it, so a reference you
    added yourself is never deleted."""
    run_ref = f"{repo}@{info.digest}"
    present = cli.image_info(run_ref)
    added = present is None or present.digest != info.digest
    if added:
        cli.tag(source, run_ref)
        # The tag copies what ``source`` names now. A pull outside launch
        # can move ``source`` after its inspect, and the new name would then
        # hold an image other than the digest it names.
        tagged = cli.image_info(run_ref)
        if tagged is None or tagged.digest != info.digest:
            if tagged is not None:
                cli.image_delete([run_ref])
            raise ImageError(f"{source} changed while launch added its digest reference, "
                             "so launch did not use it. Launch again.")
    _claim(run_ref, client, added=added)
    _write_pin(source, run_ref, fetched=fetched, pulled=pulled)
    return run_ref


def _claim(run_ref: str, client: str, *, added: bool = False) -> bool:
    """Add ``client`` and this launch's process to the record of
    ``run_ref``, so no cleanup deletes the reference while the launch runs.
    A reference launch did not add gets no record. Returns whether the
    records hold another reference of the repository that a cleanup for
    ``client`` would delete, such as one that a session kept until it ended."""
    repo = repository_of(run_ref)
    due = False

    def add(records):
        nonlocal due
        entry = records.get(run_ref)
        if entry is not None or added:
            records[run_ref] = {"clients": sorted(_owners(entry) | {client}),
                                "pids": sorted(_pids(entry) | {os.getpid()})}
        due = any(ref != run_ref and repository_of(ref) == repo
                  and _owners(other) <= {client} and not _pids(other) - {os.getpid()}
                  for ref, other in records.items())
    _update_records(add)
    return due


def _pins_path() -> Path:
    return images_dir() / "pins.json"


def _read_pins() -> dict[str, dict]:
    """The notes of the references launch pinned: for each readable
    reference, its digest reference and when launch last built or pulled
    the image, in seconds since the epoch."""
    try:
        pins = json.loads(_pins_path().read_text())
    except (OSError, ValueError, RecursionError):
        return {}
    if not isinstance(pins, dict):
        return {}
    return {ref: note for ref, note in pins.items()
            if isinstance(note, dict) and isinstance(note.get("pin"), str)
            and isinstance(note.get("at"), (int, float))}


def _write_pin(ref: str, run_ref: str, *, fetched: bool, pulled: bool = False) -> None:
    """Note ``run_ref`` as the digest reference that ``ref`` names. The time
    is now when launch built or pulled the image just now, and otherwise
    stays as noted for the same reference. An image launch finds with no
    note, such as one you pulled yourself, counts from now. ``pulled``
    marks a reference that launch pulled, which launch may delete once no
    setting names it."""
    with FileLock(images_dir() / "pins.lock"):
        pins = _read_pins()
        old = pins.get(ref)
        if not fetched and old is not None and old["pin"] == run_ref:
            return
        pulled = pulled or bool(old is not None and old.get("pulled"))
        pins[ref] = {"pin": run_ref, "at": time.time(), **({"pulled": True} if pulled else {})}
        _write_private(_pins_path(), json.dumps(pins, indent=1, sort_keys=True))


def _fetched(ref: str) -> datetime | None:
    """When launch last built or pulled the image ``ref`` names."""
    note = _read_pins().get(ref)
    return None if note is None else datetime.fromtimestamp(note["at"], timezone.utc)


def _drop_pins(deleted: set[str]) -> None:
    """Forget the notes of deleted references. A note whose digest
    reference alone was deleted stays, since it still says whether launch
    pulled the reference, and :func:`_find` checks the digest reference."""
    if not deleted:
        return
    with FileLock(images_dir() / "pins.lock"):
        pins = _read_pins()
        kept = {k: v for k, v in pins.items() if k not in deleted}
        if kept != pins:
            _write_private(_pins_path(), json.dumps(kept, indent=1, sort_keys=True))


def _find(ref: str) -> tuple[ImageInfo | None, str | None]:
    """The image ``ref`` names, and the digest reference launch noted for
    ``ref`` when the image came from it. That reference is looked up first,
    which also confirms it is still in the store, and ``ref`` itself only
    when it is gone, such as after ``container image prune``."""
    note = _read_pins().get(ref)
    pinned = note["pin"] if note is not None else None
    if pinned is not None and "@" in pinned:
        info = cli.image_info(pinned)
        if info is not None and pinned.endswith(f"@{info.digest}"):
            return info, pinned
    return cli.image_info(ref), None


def _in_use() -> tuple[set[str], set[str]]:
    """The references and digests that running launch containers use."""
    refs, digests = set(), set()
    for c in cli.list_launch_containers():
        if c.state == "running":
            refs.add(c.image)
            if c.image_digest:
                digests.add(c.image_digest)
    return refs, digests


def _cleanup(repo: str, keep: set[str], client: str, *, tags: bool, say: Say) -> None:
    """Delete the older references in ``repo``: its tags when ``tags`` is
    set, and the digest references launch recorded. ``client`` stops using
    each recorded reference it does not keep, and a reference is deleted
    once no client uses it. ``:base``, any reference a running launch
    container uses, and any reference a running launch pinned stay. A
    failure here only warns, because the image the launch needs is ready."""
    try:
        _cleanup_or_raise(repo, keep, client, tags=tags)
    except ContainerError as e:
        say(f"[launch] warning: could not delete older images of {repo}: {e}")


def _cleanup_or_raise(repo: str, keep: set[str], client: str, *, tags: bool) -> None:
    used_refs, used_digests = _in_use()
    doomed: list[str] = []
    names: list[tuple[str, str]] = []

    def choose(records):
        # The store is read while the record lock is held, so a reference
        # that another launch added before it waited for the lock is in it.
        # Only this repository is swept, which its own lock guards.
        names[:] = cli.image_names()
        store = {name for name, _digest in names}
        for ref in [r for r in records if repository_of(r) == repo and r not in store]:
            del records[ref]              # the image is gone
        for ref, entry in records.items():
            if isinstance(entry, dict):
                entry["pids"] = sorted(_pids(entry))
        for name, digest in names:
            if name in keep or repository_of(name) != repo or name.endswith(":base"):
                continue
            if "@" in name:
                entry = records.get(name)
                if not isinstance(entry, dict):
                    continue              # not added by launch
                owners = _owners(entry) - {client}
                entry["clients"] = sorted(owners)
                if owners or _pids(entry) - {os.getpid()}:
                    continue
            elif not tags:
                continue
            if name in used_refs or digest in used_digests:
                continue
            doomed.append(name)
    _update_records(choose)
    if not doomed:
        return
    cli.image_delete(doomed)
    # A reference the store refused to delete stays recorded with no
    # client, so a later cleanup tries it again.
    left = {name for name, _digest in cli.image_names()}

    def drop(records):
        for name in doomed:
            if name not in left:
                records.pop(name, None)
    _update_records(drop)
    _drop_pins(set(doomed) - left)


def check_arch(info: ImageInfo, ref: str) -> None:
    if not info.arm64:
        found = ", ".join(info.architectures) or "no runnable platform"
        raise ImageError(f"{ref} has no linux/arm64 variant ({found}). Container mode "
                         "runs Linux arm64 images only.")


# The longest wait for a builder that is stopping. 1.4.1 refuses to build
# while the builder stops.
BUILDER_STOP_WAIT = 60.0


def _owed_path() -> Path:
    """Holds the start date of a builder that a launch started and that no
    launch stopped yet."""
    return images_dir() / "builder-owed"


def _noticed_path() -> Path:
    """Holds the start date of the running builder that launch last reported."""
    return images_dir() / "builder-noticed"


def _read_date(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _write_date(path: Path, value: str) -> None:
    _write_private(path, value + "\n")


def _current_builder() -> cli.Builder | None:
    """The builder, after waiting while it stops."""
    deadline = time.monotonic() + BUILDER_STOP_WAIT
    while True:
        current = cli.builder()
        if current is None or current.state != "stopping" or time.monotonic() > deadline:
            return current
        time.sleep(0.5)


def _build(context: str, *, say: Say, announce: "_Announce", **kw) -> None:
    """Run ``container build``. The image builder is a virtual machine of
    its own that keeps its memory until it stops. When the build starts it,
    its start date is recorded as a stop that launch owes, and
    :func:`_settle_builder` stops it once no build uses it. A builder that
    was already running gets its own settings and colour variables passed
    back, so the build never makes 1.4.1 create it again."""
    using = FileLock(images_dir() / "builder.lock", shared=True)
    # Earlier versions kept a marker file here.
    (images_dir() / "builder-started").unlink(missing_ok=True)
    announce.built = True
    started_here = False
    try:
        current = _current_builder()
        if current is not None and current.state == "running":
            if current.ssh:
                raise ImageError(
                    "the running image builder forwards your SSH agent, which a "
                    "Containerfile could use. Stop it before launch builds with: "
                    "container builder stop")
            kw["builder_args"] = cli.builder_build_args(current)
            kw["env"] = cli.builder_build_env(current)
        else:
            started_here = True
        cli.build(context, **kw)
    finally:
        # Nothing here may replace the build's own error. A closed window
        # sends launch a second SIGHUP, so the query has a process group of
        # its own.
        try:
            after = cli.builder(own_group=True) if started_here else None
            if after is not None and after.state == "running" and after.started:
                _write_date(_owed_path(), after.started)
        except ContainerError:
            pass
        except OSError as e:
            say(f"[launch] warning: could not record that launch started the image "
                f"builder ({e}). Stop it when the build ends with: container builder stop")
        finally:
            using.release()


def _other_builds() -> bool:
    """Whether a ``container build`` or ``container builder`` command runs
    on the Mac, such as a build you started yourself."""
    try:
        out = subprocess.run(["/bin/ps", "-Ao", "command="], capture_output=True, text=True,
                             timeout=10, stdin=subprocess.DEVNULL, process_group=0).stdout
    except (OSError, subprocess.SubprocessError):
        return True                       # unknown, so keep the builder
    return any(_BUILD_COMMAND.search(line) for line in out.splitlines())


# Flags, such as --debug, can come before the subcommand.
_BUILD_COMMAND = re.compile(r"(^|/)container(\s+-\S+)*\s+(build|builder)(\s|$)")


def _settle_builder(say: Say) -> None:
    """Stop the builder when launch owes the stop: the builder that runs
    now is the one a launch started, no launch holds the builder lock, and
    no other build runs. The recorded start date identifies that one start,
    so a builder you started later is never stopped. A failure only warns,
    since the image is ready. The calls have a process group of their own,
    since they also run after a signal ends the build."""
    try:
        last = FileLock(images_dir() / "builder.lock", blocking=False)
    except LockHeld:
        return                            # another launch is still building
    try:
        owed = _read_date(_owed_path())
        if owed is None:
            return
        current = cli.builder(own_group=True)
        if current is None or current.state != "running" or current.started != owed:
            _owed_path().unlink(missing_ok=True)
            return
        if _other_builds():
            return
        cli.builder_stop(own_group=True)
        _owed_path().unlink(missing_ok=True)
    except ContainerError as e:
        say(f"[launch] warning: could not stop the image builder ({e}). Stop it with: "
            "container builder stop")
    finally:
        last.release()


def _idle_builder() -> tuple[cli.Builder, bool] | None:
    """The running builder when no build uses it, and whether launch owes
    its stop."""
    current = cli.builder()
    if current is None or current.state != "running":
        return None
    try:
        idle = FileLock(images_dir() / "builder.lock", blocking=False)
    except LockHeld:
        return None                       # a launch is building
    try:
        if _other_builds():
            return None
    finally:
        idle.release()
    owed = current.started is not None and _read_date(_owed_path()) == current.started
    return current, owed


def _builder_line(current: cli.Builder) -> str:
    size = (f" and holds {current.memory_bytes / (1 << 30):.0f} GB of memory"
            if current.memory_bytes else "")
    return (f"[launch] the image builder is running{size}. Stop it with: "
            "container builder stop")


def builder_report() -> tuple[str, bool] | None:
    """For ``gmlx doctor``: a line about a running builder that no build
    uses, and whether launch owes its stop, or None. It never raises."""
    try:
        found = _idle_builder()
    except (ContainerError, OSError):
        return None
    if found is None:
        return None
    current, owed = found
    return _builder_line(current), owed


def builder_notice(say: Say = _say, *, settle: bool = True) -> str | None:
    """For a launch: stop a builder whose stop launch owes, else one line
    about a running builder that no build uses, once for each start of the
    builder. None otherwise. It never raises.

    With ``settle`` false an owed builder is left running, for a launch
    that builds next and would only start the builder again."""
    try:
        found = _idle_builder()
        if found is None:
            return None
        current, owed = found
        if owed:
            if settle:
                _settle_builder(say)
            return None
        if current.started and _read_date(_noticed_path()) == current.started:
            return None
        if current.started:
            _write_date(_noticed_path(), current.started)
        return _builder_line(current)
    except (ContainerError, OSError):
        return None


class _Announce:
    """Prints the build and pull lines, and puts the first-run step number
    on the first of them only. It also notes whether a build ran, so the
    builder is settled once for the whole image."""

    def __init__(self, say: Say, step: str | None):
        self.say, self.step = say, step
        self.built = False

    def __call__(self, text: str) -> None:
        prefix = f"{self.step}: " if self.step else ""
        self.step = None
        self.say(f"[launch] {prefix}{text}")


BUILD_FAILED_URL = f"{DOCS_URL}troubleshooting.html#the-image-build-fails"


def _shipped_build_failure(client: str, packages: list[str], returncode: int) -> str:
    """The next step after a failed build of a shipped image."""
    fix = (" When it names a package from packages, fix that entry. Otherwise launch again "
           "with --rebuild." if packages else " Launch again with --rebuild.")
    return (f"the build of the {_label(client)} image failed (exit {returncode}). The build "
            "output "
            f"above shows the failing step.{fix}\nSee {BUILD_FAILED_URL}")


def _ensure_shipped(client: str, packages: list[str], *, rebuild: bool,
                    say: Say, announce: _Announce | None = None) -> ReadyImage:
    announce = announce or _Announce(say, None)
    repo, tag, base = recipe_repo(client), shipped_tag(client, packages), base_ref(client)
    with repo_lock(repo, say=say):
        info, pinned = (None, None) if rebuild else _find(tag)
        action = "found"
        if info is None:
            node = _node_base()
            recipe = _recipe_manifest(client, packages)
            if not rebuild:
                reason = _recipe_reason(repo, client, recipe)
                if reason:
                    say(reason)
            announce(f"building the {_label(client)} image, which takes a few minutes. Later "
                     "launches reuse it.")
            # The builder keeps the base it downloaded, which never reaches
            # the image store, so a completed build records it.
            if (node and _read_date(_node_base_path()) != node
                    and cli.image_info(node) is None):
                say(f"[launch] the build first downloads about {cli.NODE_BASE_DOWNLOAD_MB} MB "
                    f"for the {_shown_base(node)} base image")
            try:
                _build(str(SHIPPED_CONTAINERFILE.parent), say=say, announce=announce,
                       file=str(SHIPPED_CONTAINERFILE),
                       tags=[tag, base], build_args=_shipped_args(client, packages),
                       labels=LAUNCH_LABELS, no_cache=rebuild, pull=rebuild)
            except cli.BuildFailed as e:
                raise ImageError(_shipped_build_failure(client, packages, e.returncode)) from None
            info, pinned = cli.image_info(tag), None
            if info is None:
                raise ImageError(f"the build finished but {tag} is not in the image store. "
                                 "Launch again with --rebuild.")
            action = "built"
            try:
                _write_private(_manifest_path(repo, "recipe"), json.dumps(recipe, sort_keys=True))
                _write_private(_node_base_path(), node + "\n")
            except OSError as e:
                say(f"[launch] warning: could not record the recipe of the build ({e}).")
        else:
            current = cli.image_info(base)
            if current is None or current.digest != info.digest:
                cli.tag(tag, base)
        run_ref = _pin_and_clean(tag, repo, info, client, pinned=pinned, keep={tag, base},
                                 tags=True, say=say, fetched=action != "found")
    return ReadyImage("shipped", tag, info, run_ref, action, client, _fetched(tag))


def _ensure_build(plan: ImagePlan, *, rebuild: bool, say: Say,
                  announce: _Announce) -> ReadyImage:
    bases = sorted(plan.bases)
    rebuilt: set[str] = set()
    while True:
        for b in bases:
            _ensure_shipped(b, plan.base_packages.get(b, []),
                            rebuild=rebuild and b not in rebuilt, say=say, announce=announce)
            rebuilt.add(b)
        held: list[FileLock] = []
        try:
            for b in bases:
                held.append(repo_lock(recipe_repo(b), shared=True, say=say))
            digests: dict[str, str] = {}
            for b in bases:
                info, _ = _find(shipped_tag(b, plan.base_packages.get(b, [])))
                current = cli.image_info(base_ref(b))
                if info is None or current is None or current.digest != info.digest:
                    break
                digests[b] = info.digest
            else:
                return _build_user_image(plan, digests, rebuild=rebuild, say=say,
                                         announce=announce)
        finally:
            for lock in held:
                lock.release()


def _build_user_image(plan: ImagePlan, digests: dict[str, str], *, rebuild: bool,
                      say: Say, announce: _Announce) -> ReadyImage:
    assert plan.containerfile is not None and plan.context is not None
    repo = build_repo(plan.client)
    containerfile = _read_regular(plan.containerfile, CONTAINERFILE_MAX)
    entries = _context_entries(plan, say)
    tag = f"{repo}:{_hash_of(containerfile, digests, entries)}"
    with repo_lock(repo, say=say):
        info, pinned = (None, None) if rebuild else _find(tag)
        action = "found"
        if info is None:
            manifest = _manifest(plan, containerfile, digests, entries)
            if not rebuild:
                try:
                    old = json.loads(_manifest_path(repo).read_text())
                except (OSError, ValueError, RecursionError):
                    old = None
                reason = rebuild_reason(old, manifest)
                if reason:
                    say(reason)
            announce(f"building {plan.containerfile}")
            try:
                _build(str(plan.context), say=say, announce=announce,
                       file=str(plan.containerfile), tags=[tag],
                       labels=LAUNCH_LABELS, no_cache=rebuild, pull=rebuild and not plan.bases)
            except cli.BuildFailed as e:
                raise ImageError(
                    f"the build of {plan.containerfile} failed (exit {e.returncode}). The build "
                    "output above shows the failing step. Fix that step, or launch again with "
                    f"--rebuild.\nSee {BUILD_FAILED_URL}") from None
            info, pinned = cli.image_info(tag), None
            if info is None:
                raise ImageError(f"the build finished but {tag} is not in the image store. "
                                 "Launch again with --rebuild.")
            action = "built"
            try:
                _write_private(_manifest_path(repo), json.dumps(manifest, sort_keys=True))
            except OSError as e:
                say(f"[launch] warning: could not record the build context ({e}).")
        check_arch(info, tag)
        run_ref = _pin_and_clean(tag, repo, info, plan.client, pinned=pinned, keep={tag},
                                 tags=True, say=say, fetched=action != "found")
    return ReadyImage("build", tag, info, run_ref, action, plan.client, _fetched(tag))


def _ensure_pulled(plan: ImagePlan, *, rebuild: bool, say: Say,
                   announce: _Announce) -> ReadyImage:
    assert plan.ref is not None
    ref = plan.ref
    # The lock covers the lookup, the pull and the pin. Another launch then
    # cannot pull the reference again, or delete the pin that the lookup
    # found, before this launch pins and claims it.
    locked = repository_of(normalized(ref))
    with repo_lock(locked, say=say):
        info, pinned = (None, None) if rebuild else _find(ref)
        action, pulled = "found", False
        if info is None:
            # A reference that is in the store before the pull is yours, also
            # under --rebuild, so launch never deletes it later.
            pulled = not rebuild or cli.image_info(ref) is None
            announce(f"pulling {ref}")
            try:
                cli.pull(ref)
            except ContainerError as e:
                raise ImageError(f"{e} Check the image reference. When the image is private, "
                                 f"sign in to its registry with: container registry login "
                                 f"{_registry_of(ref)}") from None
            info, pinned = cli.image_info(ref), None
            if info is None:
                raise ImageError(f"the pull finished but {ref} is not in the image store. "
                                 "Launch again with --rebuild.")
            action = "pulled"
        check_arch(info, ref)
        repo = repository_of(pinned or info.name or ref)
        with repo_lock(repo, say=say) if repo != locked else contextlib.nullcontext():
            run_ref = _pin_and_clean(ref, repo, info, plan.client, pinned=pinned, keep=set(),
                                     tags=False, say=say, fetched=action != "found",
                                     pulled=pulled)
    return ReadyImage("image", ref, info, run_ref, action, plan.client, _fetched(ref))


def _pin_and_clean(ref: str, repo: str, info: ImageInfo, client: str, *, pinned: str | None,
                   keep: set[str], tags: bool, say: Say, fetched: bool,
                   pulled: bool = False) -> str:
    """The digest reference to run. An image found through the reference
    launch noted for ``ref`` needs no new pin, and it needs a cleanup only
    when the records hold an older reference to delete, so a launch that
    changes nothing runs no command here."""
    if pinned is not None:
        if _claim(pinned, client):
            _cleanup(repo, {*keep, pinned}, client, tags=tags, say=say)
        return pinned
    run_ref = _pin(ref, repo, info, client, fetched=fetched, pulled=pulled)
    _cleanup(repo, {*keep, run_ref}, client, tags=tags, say=say)
    return run_ref


# What the config no longer names

def normalized(ref: str) -> str:
    """``ref`` as the image store names it: a Docker Hub short name gets
    ``docker.io/library/``, and a name with no tag or digest ``:latest``."""
    first, sep, _ = ref.partition("/")
    if not (sep and ("." in first or ":" in first or first == "localhost")):
        ref = "docker.io/" + (ref if sep else "library/" + ref)
    name = ref.split("@", 1)[0]
    if "@" not in ref and name.rfind(":") <= name.rfind("/"):
        ref += ":latest"
    return ref


def _named(launch_cfg: LaunchCfg) -> dict:
    """What the config names for each launch target: an image: reference,
    or a build: folder, whose images live in the target's build
    repository."""
    named: dict[str, str] = {}
    builds: list[str] = []
    for client in launch_cfg.targets():
        cfg = launch_cfg.for_target(client)
        if cfg.image:
            named[client] = cfg.image
        elif cfg.build:
            builds.append(client)
    return {"images": named, "builds": builds}


def _named_path() -> Path:
    """Holds what the config named at the last launch."""
    return images_dir() / "named.json"


def _was_target(key) -> bool:
    """Whether a key read from a record is a client or an agent key, so a
    removed agent's images are cleaned too."""
    return isinstance(key, str) and (key in LAUNCH_CLIENTS or agent_name(key) is not None)


def forget_unnamed(launch_cfg: LaunchCfg, say: Say = _say) -> None:
    """Delete what the config named at the last launch and names no more:
    the tags and digest references of the build repository of a target
    that no longer sets build:, or is no longer configured, and an image:
    reference that launch pulled and no target names, with launch's digest
    references of it. An image that a running container uses stays. A
    launch whose config is unchanged runs no command here. A failure only
    warns."""
    new = _named(launch_cfg)
    try:
        old = json.loads(_named_path().read_text())
    except (OSError, ValueError, RecursionError):
        old = None
    if old == new:
        return
    try:
        _write_private(_named_path(), json.dumps(new, sort_keys=True))
    except OSError as e:
        say(f"[launch] warning: could not record the image settings ({e}).")
        return
    if not isinstance(old, dict):
        return                            # nothing to compare with yet
    old_builds = old.get("builds") if isinstance(old.get("builds"), list) else []
    old_images = old.get("images") if isinstance(old.get("images"), dict) else {}
    try:
        for client in sorted(set(old_builds) - set(new["builds"])):
            if _was_target(client):
                repo = build_repo(client)
                with repo_lock(repo, say=say):
                    _cleanup_or_raise(repo, set(), client, tags=True)
        for client, ref in sorted(old_images.items()):
            if _was_target(client) and isinstance(ref, str) \
                    and new["images"].get(client) != ref:
                _forget_image(client, ref, new["images"])
    except (ContainerError, OSError) as e:
        say(f"[launch] warning: could not delete images that no setting uses: {e}")


def _forget_image(client: str, ref: str, named: dict[str, str]) -> None:
    """Drop ``client`` from launch's digest references of ``ref``, and
    delete ``ref`` itself when launch pulled it, no client names it and no
    running container uses it."""
    repo = repository_of(normalized(ref))
    notes = _read_pins()
    keep = {notes[r]["pin"] for r in named.values() if r in notes}
    with repo_lock(repo):
        _cleanup_or_raise(repo, keep, client, tags=False)
        note = notes.get(ref)
        if ref in named.values() or note is None or not note.get("pulled"):
            return
        stored = normalized(ref)
        info = cli.image_info(stored)
        running = [c for c in cli.containers() if c.state == "running"]
        if info is None or any(c.image in (stored, ref) or c.image_digest == info.digest
                               for c in running):
            return
        cli.image_delete([stored])
        if cli.image_info(stored) is None:
            _drop_pins({ref})


def disk_report(launch_cfg: LaunchCfg | None) -> tuple[int, int, list[str]]:
    """For ``gmlx doctor``: the number of images launch keeps, the bytes of
    their layers, each image counted once, and the references among them
    that no current setting uses and no running container needs. Launch's
    images are its own ``gmlx.invalid`` references, the references it
    pinned or pulled, and the image: references of the config. Without a
    config, nothing counts as unused."""
    stored = cli.image_list()
    notes = _read_pins()
    try:
        records = json.loads(_records_path().read_text())
    except (OSError, ValueError, RecursionError):
        records = {}
    named = _named(launch_cfg)["images"] if launch_cfg is not None else {}
    ours = ({normalized(r) for r in notes} | {n["pin"] for n in notes.values()}
            | set(records if isinstance(records, dict) else ())
            | {normalized(r) for r in named.values()})
    mine = [s for s in stored if s.name.startswith(f"{DOMAIN}/") or s.name in ours]
    sizes = {s.digest: s.size for s in mine}
    if launch_cfg is None:
        return len(sizes), sum(sizes.values()), []
    used = _used_names(launch_cfg, [s.name for s in mine])
    running = [c for c in cli.containers() if c.state == "running"]
    used |= {c.image for c in running}
    used_digests = {s.digest for s in mine if s.name in used} | {
        c.image_digest for c in running}
    unused = sorted(s.name for s in mine if s.name not in used and s.digest not in used_digests)
    return len(sizes), sum(sizes.values()), unused


def _used_names(launch_cfg: LaunchCfg, names: list[str]) -> set[str]:
    """The references that the current settings of the launch targets use.
    A setting launch cannot read keeps every image of that target, and the
    runtime image of a runtime agent."""
    used: set[str] = set()
    for client in launch_cfg.targets():
        cfg = launch_cfg.for_target(client)
        stage = stage_for(launch_cfg, client)
        try:
            plan = resolve_image(client, cfg, launch_cfg.container, stage=stage)
        except (ImageError, OSError):
            repos = {recipe_repo(client), build_repo(client)}
            if stage is not None:
                repos.add(recipe_repo(stage))
            used |= {n for n in names if repository_of(n) in repos}
            continue
        if plan.kind == "image":
            assert plan.ref is not None
            used.add(normalized(plan.ref))
            continue
        # A shipped plan's client is its stage, so every runtime agent
        # counts the shared runtime image as used.
        shipped = [plan.client] if plan.kind == "shipped" else plan.bases
        for b in shipped:
            packages = plan.packages if b == plan.client else plan.base_packages.get(b, [])
            used |= {shipped_tag(b, packages), base_ref(b)}
        if plan.kind == "build":
            used |= {n for n in names if repository_of(n) == build_repo(client)}
    return used


def ensure_image(plan: ImagePlan, *, rebuild: bool = False, say: Say = _say,
                 step: str | None = None) -> ReadyImage:
    """Build, pull or find the planned image and pin it by digest. ``step``,
    such as ``"step 2 of 3"``, goes on the first build or pull line."""
    announce = _Announce(say, step)
    try:
        if plan.kind == "shipped":
            return _ensure_shipped(plan.client, plan.packages, rebuild=rebuild, say=say,
                                   announce=announce)
        if plan.kind == "build":
            return _ensure_build(plan, rebuild=rebuild, say=say, announce=announce)
        return _ensure_pulled(plan, rebuild=rebuild, say=say, announce=announce)
    finally:
        # Once for the whole image, so a base build and a user build do not
        # stop and start the builder between them.
        if announce.built:
            _settle_builder(say)


def pending_work(plan: ImagePlan, rebuild: bool) -> str | None:
    """``"build"`` or ``"pull"`` when :func:`ensure_image` would build or
    pull, None when the image is ready. It only reads the image store."""
    if plan.kind == "image":
        assert plan.ref is not None
        return "pull" if rebuild or _find(plan.ref)[0] is None else None
    if rebuild:
        return "build"
    if plan.kind == "shipped":
        return "build" if _find(shipped_tag(plan.client, plan.packages))[0] is None else None
    digests: dict[str, str] = {}
    for b in plan.bases:
        info, _ = _find(shipped_tag(b, plan.base_packages.get(b, [])))
        if info is None:
            return "build"
        digests[b] = info.digest
    tag = f"{build_repo(plan.client)}:{build_hash(plan, digests, lambda _line: None)}"
    return "build" if _find(tag)[0] is None else None


def _checks_path() -> Path:
    return images_dir() / "checks.json"


# The guest entry knows no config key, so its hint for a missing command
# ends with this placeholder, which the Mac fills in.
_GENERIC_COMMAND_HINT = "or set launch.container.clients.<client>.command."


def _command_hint(line: str, client: str, *, runtime: bool) -> str:
    """The guest's message with the config key of this target in place of
    the placeholder. A runtime agent's uv comes from its runtime, not from
    its command, so the hint names the runtime key."""
    if not line.endswith(_GENERIC_COMMAND_HINT):
        return line
    head = line[:-len(_GENERIC_COMMAND_HINT)]
    if runtime:
        return f"{head}or remove {config_key(client, 'runtime')}, so the command runs without uv."
    return f"{head}or set {config_key(client, 'command')}."


def check_command(ready: ReadyImage, word: str, runtime_dir: str, *, shell: bool,
                  say: Say = _say, runtime: bool = False) -> None:
    """Confirm once per image and command that the command exists in the
    image and can run, by running ``gmlx-entry --check`` in it with no
    network. A shipped image running its own client skips the check. Under
    ``--shell`` a missing command, or one without the execute bit, only
    warns. Only a passed check is remembered. ``runtime`` marks a runtime
    agent, whose missing command is uv."""
    if ready.kind == "shipped" and word in (CLIENT_BINARY.get(ready.client),
                                            RUNTIME_BINARY.get(ready.client)):
        return
    key = f"{ready.info.digest} {word}"
    with FileLock(images_dir() / "checks.lock"):
        try:
            if key in json.loads(_checks_path().read_text()):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    rc, line = cli.run_entry_check(ready.run_ref, runtime_dir, word)
    line = _command_hint(line, ready.client, runtime=runtime)
    if rc == 0:
        with FileLock(images_dir() / "checks.lock"):
            try:
                seen = json.loads(_checks_path().read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                seen = {}
            seen[key] = True
            _write_private(_checks_path(), json.dumps(seen, indent=1, sort_keys=True))
        return
    if rc not in (126, 127):
        raise ImageError(f"the check of {ready.tag} for {word} failed (exit {rc})"
                         + (f": {line}" if line else "."))
    if shell:
        say(f"[launch] warning: {line}")
        return
    raise ImageError(line or f"{word} is not in {ready.tag}.")


def image_command(ready: ReadyImage, command: list[str] | str | None,
                  handler_argv: list[str], passthrough: list[str]
                  ) -> tuple[list[str], str | None]:
    """The argv the container runs and the working folder the image asks for.

    With no ``command`` it is the handler's argv plus the passthrough. A list
    replaces the handler's argv. ``image`` runs the image's ENTRYPOINT and
    CMD, where the passthrough replaces CMD, and uses the image's WorkingDir.
    """
    if command == "image":
        argv = [*(ready.info.entrypoint or []),
                *(passthrough if passthrough else (ready.info.cmd or []))]
        if not argv:
            raise ImageError(f"{ready.tag} sets no ENTRYPOINT or CMD, so command: image "
                             f"has nothing to run. Set {config_key(ready.client, 'command')} "
                             "to the command to run.")
        return argv, ready.info.workdir
    base = list(command) if isinstance(command, list) else list(handler_argv)
    return [*base, *passthrough], None


def _age_days(ready: ReadyImage, now: datetime | None) -> int | None:
    """Days since launch last built or pulled the image. An image's own
    creation date would not do: a registry image keeps it however recently
    launch pulled it."""
    if ready.fetched is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0, (now - ready.fetched).days)


def _ago(days: int) -> str:
    if days == 0:
        return "today"
    return f"{days} day{'s' if days != 1 else ''} ago"


def _verb(ready: ReadyImage) -> str:
    return "pulled" if ready.kind == "image" else "built"


def describe(ready: ReadyImage, now: datetime | None = None) -> str:
    """The summary line that names the image by its readable reference,
    and the client version a shipped image installs, or the tool version of
    a runtime image, such as ``with uv 0.12.22``."""
    version = shipped_version(ready.client) if ready.kind == "shipped" else None
    tool = RUNTIME_BINARY.get(ready.client, ready.client)
    line = f"[launch] image {ready.tag}" + (f" with {tool} {version}" if version else "")
    days = _age_days(ready, now)
    return line if days is None else f"{line}, {_verb(ready)} {_ago(days)}"


def image_age_note(ready: ReadyImage, now: datetime | None = None) -> str | None:
    """A note, once a day, when launch built or pulled the image more than
    :data:`AGE_NOTE_DAYS` ago. ``--rebuild`` renews the image and ends it."""
    days = _age_days(ready, now)
    if days is None or days <= AGE_NOTE_DAYS:
        return None
    again = ("pulls it again" if ready.kind == "image"
             else "builds it again with current packages")
    return notices.Once(f"[launch] launch {_verb(ready)} this image {days} days ago. "
                        f"--rebuild {again}, which ends this note.",
                        f"age:{ready.tag}", every=notices.DAY)
