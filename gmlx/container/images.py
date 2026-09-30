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

from gmlx.config import LAUNCH_CLIENTS, LaunchClientCfg, LaunchContainerCfg

from . import cli, ignore
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

Say = Callable[[str], None]


def _shown(path: str | os.PathLike) -> str:
    """A path for a message: ``~`` for the home folder, and any control
    character written as an escape, since a file name can hold one."""
    text = str(path)
    home = os.path.expanduser("~")
    if text == home or text.startswith(home.rstrip("/") + "/"):
        text = "~" + text[len(home.rstrip("/")):]
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)


def _read_regular(path: Path, limit: int) -> bytes:
    """A regular file of less than ``limit`` bytes. Never waits on a named
    pipe or a device."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise ImageError(f"cannot read {_shown(path)}: {e.strerror}") from None
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
        raise ImageError(f"{_shown(path)} is {limit} bytes or more. `container build` "
                         f"refuses a Containerfile of {limit} bytes or more.")
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
    """The clients whose ``:base`` a Containerfile names, in first-use order.
    Any other ``gmlx.invalid`` reference is refused, because launch deletes
    every other tag there when a newer build replaces it."""
    clients: list[str] = []
    for ref in _BASE_SCAN.findall(text):
        m = _BASE_FORM.fullmatch(ref)
        if m is None or m[1] not in LAUNCH_CLIENTS:
            raise ImageError(
                f"the Containerfile names {ref}. A Containerfile may name only the "
                "stable base of a client, such as gmlx.invalid/launch-claude-code:base, "
                "because launch deletes the other gmlx.invalid tags when it builds again.")
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
            f"launch.container.clients.{client}.build is {build!r}. Give an absolute "
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
    raise ImageError(f"launch.container.clients.{client}.build names {path}, which does not exist.")


def resolve_image(client: str, cfg: LaunchClientCfg, container: LaunchContainerCfg, *,
                  image_override: str | None = None,
                  writable: Sequence[str] = ()) -> ImagePlan:
    """Decide which image ``client`` runs, and refuse a ``build:``
    Containerfile that cannot build. ``writable`` holds the Mac folders the
    session shares read-write. A ``build:`` folder that overlaps one, a
    folder an earlier launch shared read-write, or the private homes is
    refused, because a client could change what the next build runs, and a
    build has the network and the builder."""
    if image_override:
        notices = []
        if cfg.packages:
            notices.append(f"[launch] --image replaces the {client} image, so the "
                           "packages: list is not used.")
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
                f"the {client} packages: list reaches a build: image only through its "
                f"own base: start {file} with FROM {base_ref(client)}, or move the "
                "packages into the Containerfile.")
        return ImagePlan("build", client, containerfile=file, context=context,
                         bases=bases, packages=list(cfg.packages),
                         base_packages={b: container.for_client(b).packages for b in bases})
    return ImagePlan("shipped", client, packages=list(cfg.packages))


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
                f"the {client} build: folder {_shown(real_context)} overlaps "
                f"{_shown(share)}, which this launch shares read-write, so the client "
                "could change what the next build runs. Move the build folder out of "
                "the share, or share that folder read-only.")
    data = canonical(data_path())
    if overlaps(data):
        raise ImageError(
            f"the {client} build: folder {_shown(real_context)} overlaps {_shown(data)}, "
            "which holds the private homes of the clients, so a client could change "
            "what the next build runs. Move the build folder out of it.")
    for share in (canonical(w) for w in shared_history()):
        if overlaps(share):
            raise ImageError(
                f"the {client} build: folder {_shown(real_context)} overlaps "
                f"{_shown(share)}, which an earlier launch shared read-write, so a client "
                "may have changed what the next build runs. Move the build folder to a "
                "folder that no launch has shared read-write.")


def shipped_hash(client: str, packages: list[str]) -> str:
    h = hashlib.sha256(SHIPPED_CONTAINERFILE.read_bytes())
    h.update(json.dumps(_shipped_args(client, packages), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _shipped_args(client: str, packages: list[str]) -> dict[str, str]:
    # Sorted, so the same packages in another order use the same image.
    return {"CLIENT": client, "EXTRA_PACKAGES": " ".join(sorted(set(packages)))}


def _node_base() -> str:
    """The base image the shipped Containerfile names."""
    m = re.search(r"^FROM\s+(\S+)", SHIPPED_CONTAINERFILE.read_text(), re.M)
    return m[1] if m else ""


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


def _manifest_path(repo: str) -> Path:
    return images_dir() / (re.sub(r"[^A-Za-z0-9_.-]+", "_", repo) + ".context.json")


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


def _pin(source: str, repo: str, info: ImageInfo, client: str) -> str:
    """Add ``<repo>@<digest>`` to the local store. The reference is recorded
    as launch's only when launch added it, so a reference you added yourself
    is never deleted."""
    run_ref = f"{repo}@{info.digest}"
    present = cli.image_info(run_ref)
    added = present is None or present.digest != info.digest
    if added:
        cli.tag(source, run_ref)

    def add(records):
        entry = records.get(run_ref)
        if entry is None and not added:
            return
        records[run_ref] = {"clients": sorted(_owners(entry) | {client}),
                            "pids": sorted(_pids(entry) | {os.getpid()})}
    _update_records(add)
    return run_ref


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
                    "the image builder that runs now forwards your SSH agent, so a "
                    "Containerfile could use every key in it. Launch does not build on "
                    "it until you stop it with container builder stop")
            kw["builder_args"] = cli.builder_build_args(current)
            kw["env"] = cli.builder_build_env(current)
        else:
            started_here = True
        cli.build(context, **kw)
    finally:
        # Nothing here may replace the build's own error.
        try:
            after = cli.builder() if started_here else None
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
        out = subprocess.run(["ps", "-Ao", "command="], capture_output=True, text=True,
                             timeout=10).stdout
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
    since the image is ready."""
    try:
        last = FileLock(images_dir() / "builder.lock", blocking=False)
    except LockHeld:
        return                            # another launch is still building
    try:
        owed = _read_date(_owed_path())
        if owed is None:
            return
        current = cli.builder()
        if current is None or current.state != "running" or current.started != owed:
            _owed_path().unlink(missing_ok=True)
            return
        if _other_builds():
            return
        cli.builder_stop()
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


def _ensure_shipped(client: str, packages: list[str], *, rebuild: bool,
                    say: Say, announce: _Announce | None = None) -> ReadyImage:
    announce = announce or _Announce(say, None)
    repo, tag, base = recipe_repo(client), shipped_tag(client, packages), base_ref(client)
    with repo_lock(repo, say=say):
        info = None if rebuild else cli.image_info(tag)
        action = "found"
        if info is None:
            node = _node_base()
            download = rebuild or (node and cli.image_info(node) is None)
            announce(f"building the {client} image" + (
                f", which first downloads about {cli.NODE_BASE_DOWNLOAD_MB} MB for {node}"
                if download else ""))
            _build(str(SHIPPED_CONTAINERFILE.parent), say=say, announce=announce,
                   file=str(SHIPPED_CONTAINERFILE),
                   tags=[tag, base], build_args=_shipped_args(client, packages),
                   labels=LAUNCH_LABELS, no_cache=rebuild, pull=rebuild)
            info = cli.image_info(tag)
            if info is None:
                raise ImageError(f"the build finished but {tag} is not in the image store.")
            action = "built"
        else:
            current = cli.image_info(base)
            if current is None or current.digest != info.digest:
                cli.tag(tag, base)
        run_ref = _pin(tag, repo, info, client)
        _cleanup(repo, {tag, base, run_ref}, client, tags=True, say=say)
    return ReadyImage("shipped", tag, info, run_ref, action, client)


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
                info = cli.image_info(shipped_tag(b, plan.base_packages.get(b, [])))
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
        info = None if rebuild else cli.image_info(tag)
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
            _build(str(plan.context), say=say, announce=announce,
                   file=str(plan.containerfile), tags=[tag],
                   labels=LAUNCH_LABELS, no_cache=rebuild, pull=rebuild and not plan.bases)
            info = cli.image_info(tag)
            if info is None:
                raise ImageError(f"the build finished but {tag} is not in the image store.")
            action = "built"
            try:
                _write_private(_manifest_path(repo), json.dumps(manifest, sort_keys=True))
            except OSError as e:
                say(f"[launch] warning: could not record the build context ({e}).")
        check_arch(info, tag)
        run_ref = _pin(tag, repo, info, plan.client)
        _cleanup(repo, {tag, run_ref}, plan.client, tags=True, say=say)
    return ReadyImage("build", tag, info, run_ref, action, plan.client)


def _ensure_pulled(plan: ImagePlan, *, rebuild: bool, say: Say,
                   announce: _Announce) -> ReadyImage:
    assert plan.ref is not None
    ref = plan.ref
    info = None if rebuild else cli.image_info(ref)
    action = "found"
    if info is None:
        announce(f"pulling {ref}")
        cli.pull(ref)
        info = cli.image_info(ref)
        if info is None:
            raise ImageError(f"the pull finished but {ref} is not in the image store.")
        action = "pulled"
    check_arch(info, ref)
    repo = repository_of(info.name or ref)
    with repo_lock(repo, say=say):
        run_ref = _pin(ref, repo, info, plan.client)
        _cleanup(repo, {run_ref}, plan.client, tags=False, say=say)
    return ReadyImage("image", ref, info, run_ref, action, plan.client)


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
        return "pull" if rebuild or cli.image_info(plan.ref) is None else None
    if rebuild:
        return "build"
    if plan.kind == "shipped":
        return "build" if cli.image_info(shipped_tag(plan.client, plan.packages)) is None else None
    digests: dict[str, str] = {}
    for b in plan.bases:
        info = cli.image_info(shipped_tag(b, plan.base_packages.get(b, [])))
        if info is None:
            return "build"
        digests[b] = info.digest
    tag = f"{build_repo(plan.client)}:{build_hash(plan, digests, lambda _line: None)}"
    return "build" if cli.image_info(tag) is None else None


def _checks_path() -> Path:
    return images_dir() / "checks.json"


def check_command(ready: ReadyImage, word: str, runtime_dir: str, *, shell: bool,
                  say: Say = _say) -> None:
    """Confirm once per image and command that the command exists in the
    image and can run, by running ``gmlx-entry --check`` in it with no
    network. A shipped image running its own client skips the check. Under
    ``--shell`` a missing command, or one without the execute bit, only
    warns. Only a passed check is remembered."""
    if ready.kind == "shipped" and word == CLIENT_BINARY.get(ready.client):
        return
    key = f"{ready.info.digest} {word}"
    with FileLock(images_dir() / "checks.lock"):
        try:
            if key in json.loads(_checks_path().read_text()):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    rc, line = cli.run_entry_check(ready.run_ref, runtime_dir, word)
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
                             "has nothing to run.")
        return argv, ready.info.workdir
    base = list(command) if isinstance(command, list) else list(handler_argv)
    return [*base, *passthrough], None


def _age_days(ready: ReadyImage, now: datetime | None) -> int | None:
    if ready.info.created is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0, (now - ready.info.created).days)


def _ago(days: int) -> str:
    if days == 0:
        return "today"
    return f"{days} day{'s' if days != 1 else ''} ago"


def describe(ready: ReadyImage, now: datetime | None = None) -> str:
    """The summary line that names the image by its readable reference."""
    days = _age_days(ready, now)
    if days is None:
        return f"[launch] image {ready.tag}"
    verb = "created" if ready.kind == "image" else "built"
    return f"[launch] image {ready.tag}, {verb} {_ago(days)}"


def image_age_note(ready: ReadyImage, now: datetime | None = None) -> str | None:
    days = _age_days(ready, now)
    if days is None or days <= AGE_NOTE_DAYS:
        return None
    action = "pulls it again" if ready.kind == "image" else "rebuilds it with current packages"
    return f"[launch] the image is {days} days old, and --rebuild {action}."
