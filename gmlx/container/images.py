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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gmlx.config import LAUNCH_CLIENTS, LaunchClientCfg, LaunchContainerCfg

from . import cli, ignore
from .cli import ContainerError, ImageInfo
from .state import FileLock, images_dir

DOMAIN = "gmlx.invalid"
SHIPPED_CONTAINERFILE = Path(__file__).parent / "files" / "Containerfile"
# `container build` refuses a Containerfile of this size or more.
CONTAINERFILE_MAX = 16 * 1024
DEFAULT_CONTAINERFILES = ("Containerfile", "Dockerfile")
AGE_NOTE_DAYS = 30
LAUNCH_LABELS = {cli.LAUNCH_LABEL: "1"}

# The command each shipped image installs for its client.
CLIENT_BINARY = {
    "opencode": "opencode", "pi": "pi", "omp": "omp", "hermes": "hermes",
    "goose": "goose", "claude-code": "claude", "aichat": "aichat",
    "elia": "elia", "open-webui": "open-webui", "dsh": "dsh",
}

Say = Callable[[str], None]


def _say(line: str) -> None:
    print(line, flush=True)             # before the build's own output


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
    if path.is_dir():
        for name in DEFAULT_CONTAINERFILES:
            if (path / name).is_file():
                return path / name, path
        raise ImageError(f"{path} holds no Containerfile or Dockerfile.")
    if path.is_file():
        return path, path.parent
    raise ImageError(f"launch.container.clients.{client}.build names {path}, which does not exist.")


def resolve_image(client: str, cfg: LaunchClientCfg, container: LaunchContainerCfg, *,
                  image_override: str | None = None) -> ImagePlan:
    """Decide which image ``client`` runs, and refuse a ``build:``
    Containerfile that cannot build."""
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
        size = file.stat().st_size
        if size >= CONTAINERFILE_MAX:
            raise ImageError(f"{file} is {size} bytes. `container build` refuses a "
                             f"Containerfile of {CONTAINERFILE_MAX} bytes or more.")
        bases = base_refs_in(file.read_text(encoding="utf-8", errors="replace"))
        if cfg.packages and client not in bases:
            raise ImageError(
                f"the {client} packages: list reaches a build: image only through its "
                f"own base: start {file} with FROM {base_ref(client)}, or move the "
                "packages into the Containerfile.")
        return ImagePlan("build", client, containerfile=file, context=context,
                         bases=bases, packages=list(cfg.packages),
                         base_packages={b: container.for_client(b).packages for b in bases})
    return ImagePlan("shipped", client, packages=list(cfg.packages))


def shipped_hash(client: str, packages: list[str]) -> str:
    h = hashlib.sha256(SHIPPED_CONTAINERFILE.read_bytes())
    h.update(json.dumps(_shipped_args(client, packages), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _shipped_args(client: str, packages: list[str]) -> dict[str, str]:
    return {"CLIENT": client, "EXTRA_PACKAGES": " ".join(packages)}


def shipped_tag(client: str, packages: list[str]) -> str:
    return f"{recipe_repo(client)}:{shipped_hash(client, packages)}"


def context_files(context: Path, matcher: ignore.Matcher | None):
    """``(relative path, lstat)`` of every file a build of ``context`` can
    see, sorted. The ``.git`` folder at the root is left out."""
    found = []
    prune = matcher is not None and not matcher.has_exclusions
    for root, dirs, files in os.walk(context):
        rel_root = os.path.relpath(root, context)
        rel_root = "" if rel_root == "." else rel_root + "/"
        if not rel_root:
            dirs[:] = [d for d in dirs if d != ".git"]
        if prune:
            dirs[:] = [d for d in dirs if not matcher.excluded(rel_root + d)]
        dirs.sort()
        for name in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            rel = rel_root + name
            if matcher is not None and matcher.excluded(rel):
                continue
            try:
                found.append((rel, os.lstat(os.path.join(root, name))))
            except FileNotFoundError:
                continue
    found.sort(key=lambda item: item[0])
    return found


def build_hash(plan: ImagePlan, base_digests: dict[str, str], say: Say) -> str:
    """The tag of a ``build:`` image: its Containerfile, the path, size and
    modification time of every context file the build sees, and the digest
    of each base it names."""
    assert plan.containerfile is not None and plan.context is not None
    matcher, notice = ignore.load(plan.containerfile, plan.context)
    if notice:
        say(notice)
    h = hashlib.sha256(plan.containerfile.read_bytes())
    h.update(json.dumps(sorted(base_digests.items())).encode())
    for rel, st in context_files(plan.context, matcher):
        link = ""
        if os.path.islink(plan.context / rel):
            link = os.readlink(plan.context / rel)
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_mode}\0{link}\n".encode())
    return h.hexdigest()[:16]


@dataclass
class ReadyImage:
    """An image present in the local store, pinned by digest."""
    kind: str
    tag: str                          # the readable reference
    info: ImageInfo
    run_ref: str                      # <repository>@sha256:<digest>
    action: str                       # "built", "pulled" or "found"


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
    under its own lock. Each entry maps a reference to the clients using it."""
    with FileLock(images_dir() / "references.lock"):
        path = _records_path()
        try:
            records = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            records = {}
        fn(records)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=1, sort_keys=True))
        os.replace(tmp, path)
        return records


def _pin(source: str, repo: str, info: ImageInfo, client: str) -> str:
    """Add ``<repo>@<digest>`` to the local store and record it."""
    run_ref = f"{repo}@{info.digest}"
    present = cli.image_info(run_ref)
    if present is None or present.digest != info.digest:
        cli.tag(source, run_ref)

    def add(records):
        clients = set(records.get(run_ref, []))
        clients.add(client)
        records[run_ref] = sorted(clients)
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


def _cleanup(repo: str, keep: set[str], client: str, *, tags: bool) -> None:
    """Delete the older references in ``repo``: its tags when ``tags`` is
    set, and the digest references launch recorded for ``client``. ``:base``
    and any reference a running launch container uses stay."""
    used_refs, used_digests = _in_use()
    doomed = []
    records_seen: dict = {}
    _update_records(lambda r: records_seen.update(r))
    for name, digest in cli.image_names():
        if name in keep or repository_of(name) != repo or name.endswith(":base"):
            continue
        if name in used_refs or digest in used_digests:
            continue
        if "@" in name:
            owners = set(records_seen.get(name, []))
            if client not in owners or owners - {client}:
                continue
        elif not tags:
            continue
        doomed.append(name)
    if not doomed:
        return
    cli.image_delete(doomed)

    def drop(records):
        for name in doomed:
            records.pop(name, None)
    _update_records(drop)


def check_arch(info: ImageInfo, ref: str) -> None:
    if not info.arm64:
        found = ", ".join(info.architectures) or "no runnable platform"
        raise ImageError(f"{ref} has no linux/arm64 variant ({found}). Container mode "
                         "runs Linux arm64 images only.")


def _ensure_shipped(client: str, packages: list[str], *, rebuild: bool,
                    say: Say) -> ReadyImage:
    repo, tag, base = recipe_repo(client), shipped_tag(client, packages), base_ref(client)
    with repo_lock(repo, say=say):
        info = None if rebuild else cli.image_info(tag)
        action = "found"
        if info is None:
            say(f"[launch] building the {client} image")
            cli.build(str(SHIPPED_CONTAINERFILE.parent), file=str(SHIPPED_CONTAINERFILE),
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
        _cleanup(repo, {tag, base, run_ref}, client, tags=True)
    return ReadyImage("shipped", tag, info, run_ref, action)


def _ensure_build(plan: ImagePlan, *, rebuild: bool, say: Say) -> ReadyImage:
    bases = sorted(plan.bases)
    rebuilt: set[str] = set()
    while True:
        for b in bases:
            _ensure_shipped(b, plan.base_packages.get(b, []),
                            rebuild=rebuild and b not in rebuilt, say=say)
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
                return _build_user_image(plan, digests, rebuild=rebuild, say=say)
        finally:
            for lock in held:
                lock.release()


def _build_user_image(plan: ImagePlan, digests: dict[str, str], *, rebuild: bool,
                      say: Say) -> ReadyImage:
    assert plan.containerfile is not None and plan.context is not None
    repo = build_repo(plan.client)
    tag = f"{repo}:{build_hash(plan, digests, say)}"
    with repo_lock(repo, say=say):
        info = None if rebuild else cli.image_info(tag)
        action = "found"
        if info is None:
            say(f"[launch] building {plan.containerfile}")
            cli.build(str(plan.context), file=str(plan.containerfile), tags=[tag],
                      labels=LAUNCH_LABELS, no_cache=rebuild,
                      pull=rebuild and not plan.bases)
            info = cli.image_info(tag)
            if info is None:
                raise ImageError(f"the build finished but {tag} is not in the image store.")
            action = "built"
        check_arch(info, tag)
        run_ref = _pin(tag, repo, info, plan.client)
        _cleanup(repo, {tag, run_ref}, plan.client, tags=True)
    return ReadyImage("build", tag, info, run_ref, action)


def _ensure_pulled(plan: ImagePlan, *, rebuild: bool, say: Say) -> ReadyImage:
    assert plan.ref is not None
    ref = plan.ref
    info = None if rebuild else cli.image_info(ref)
    action = "found"
    if info is None:
        say(f"[launch] pulling {ref}")
        cli.pull(ref)
        info = cli.image_info(ref)
        if info is None:
            raise ImageError(f"the pull finished but {ref} is not in the image store.")
        action = "pulled"
    check_arch(info, ref)
    repo = repository_of(info.name or ref)
    with repo_lock(repo, say=say):
        run_ref = _pin(ref, repo, info, plan.client)
        _cleanup(repo, {run_ref}, plan.client, tags=False)
    return ReadyImage("image", ref, info, run_ref, action)


def ensure_image(plan: ImagePlan, *, rebuild: bool = False,
                 say: Say = _say) -> ReadyImage:
    """Build, pull or find the planned image and pin it by digest."""
    if plan.kind == "shipped":
        return _ensure_shipped(plan.client, plan.packages, rebuild=rebuild, say=say)
    if plan.kind == "build":
        return _ensure_build(plan, rebuild=rebuild, say=say)
    return _ensure_pulled(plan, rebuild=rebuild, say=say)


def _checks_path() -> Path:
    return images_dir() / "checks.json"


def check_command(ready: ReadyImage, word: str, runtime_dir: str, *, shell: bool,
                  say: Say = _say) -> None:
    """Confirm once per image and command that the command exists in the
    image, by running ``gmlx-entry --check`` in it with no network. Shipped
    images skip the check. Under ``--shell`` a missing command only warns."""
    if ready.kind == "shipped":
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
            _checks_path().write_text(json.dumps(seen, indent=1, sort_keys=True))
        return
    if rc != 127:
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
