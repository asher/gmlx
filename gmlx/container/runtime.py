"""The guest entry binary and its runtime folder.

The wheel ships ``guest/gmlx-entry``, a static Linux arm64 program. Launch
copies it once to ``runtime/<sha256>/`` under the launch data folder and
shares that folder read-only at ``/opt/gmlx``, so the mount depends neither on
the install's file modes nor on a site-packages path that ``--mount`` cannot
take. The folder's ``bin/`` holds the clipboard stand-ins, links named
``xclip``, ``xsel`` and ``wl-paste`` to the same binary. Each launch holds a
shared lock on the folder until its session ends, and :func:`cleanup_runtime`
removes the folders no launch holds.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
import uuid
from pathlib import Path

from .state import FileLock, LockHeld, data_dir

GUEST_MOUNT = "/opt/gmlx"
GUEST_ENTRY = f"{GUEST_MOUNT}/gmlx-entry"
GUEST_BIN = f"{GUEST_MOUNT}/bin"
BUILD_HINT = "python scripts/build_guest_entry.py"
CLIP_TOOLS = ("xclip", "xsel", "wl-paste")
# An install folder older than this belongs to a launch that was killed,
# since an install takes a few seconds.
TMP_STALE = 600.0


def entry_path() -> Path:
    """The packaged guest entry. It may be missing in a git checkout."""
    return Path(__file__).parent / "guest" / "gmlx-entry"


def entry_digest(path: Path | None = None) -> str:
    return hashlib.sha256((path or entry_path()).read_bytes()).hexdigest()


def runtime_root() -> Path:
    return data_dir() / "runtime"


def _complete(folder: Path) -> bool:
    """Whether the folder holds the entry its name promises, as a regular
    file whose digest is the folder name, and the clipboard links. A
    session that shared the launch data folder could have replaced it."""
    entry = folder / "gmlx-entry"
    try:
        st = os.lstat(entry)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode) or not (folder / ".lock").exists():
        return False
    bin_dir = folder / "bin"
    if not all((bin_dir / tool).is_symlink() and os.readlink(bin_dir / tool) == "../gmlx-entry"
               for tool in CLIP_TOOLS):
        return False
    try:
        return entry_digest(entry) == folder.name
    except OSError:
        return False


def _install(folder: Path, source: Path) -> None:
    """Build the folder beside its final path and rename it into place, so no
    launch ever sees half a folder."""
    tmp = folder.parent / f".tmp-{uuid.uuid4().hex[:8]}"
    folder.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp.mkdir(mode=0o700)
    try:
        shutil.copyfile(source, tmp / "gmlx-entry")
        os.chmod(tmp / "gmlx-entry", 0o755)
        (tmp / "bin").mkdir()
        for tool in CLIP_TOOLS:
            os.symlink("../gmlx-entry", tmp / "bin" / tool)
        (tmp / ".lock").touch()
        try:
            os.rename(tmp, folder)
        except OSError:
            if not _complete(folder):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def acquire_runtime(source: Path | None = None) -> tuple[Path, FileLock]:
    """The runtime folder of the packaged entry, with a shared lock on it that
    the caller holds until its session ends."""
    from .settings import check_mount_chars

    source = source or entry_path()
    folder = runtime_root() / entry_digest(source)
    check_mount_chars(str(folder), "the runtime folder")
    for _ in range(5):
        if not _complete(folder):
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            _install(folder, source)
        lock = FileLock(folder / ".lock", shared=True)
        if lock.still_current() and _complete(folder):
            return folder, lock
        lock.release()                # a cleanup removed it between the two steps
    raise RuntimeError(f"cannot prepare the runtime folder {folder}")


def cleanup_runtime(keep: str | None = None) -> list[Path]:
    """Remove the runtime folders no launch holds, except ``keep``. Returns
    the folders removed."""
    root = runtime_root()
    removed = []
    if not root.is_dir():
        return removed
    for folder in sorted(root.iterdir()):
        if folder.name == keep or not folder.is_dir():
            continue
        if folder.name.startswith(".tmp-"):
            # A recent one is another launch's install in progress.
            try:
                age = time.time() - folder.stat().st_mtime
            except OSError:
                continue
            if age > TMP_STALE:
                shutil.rmtree(folder, ignore_errors=True)
                removed.append(folder)
            continue
        lock_file = folder / ".lock"
        if not lock_file.exists():
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
            continue
        try:
            lock = FileLock(lock_file, blocking=False)
        except LockHeld:
            continue
        try:
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
        finally:
            lock.release()
    return removed
