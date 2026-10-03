"""Load an stt, tts or embeddings model from the local Hugging Face cache.

The audio and embeddings backends (mlx-audio, mlx-whisper, the gmlx
encoders) each take a repo id and call ``snapshot_download`` themselves.
With ``local_files_only=False``, the default, that asks the Hub for the
repo's file list on every load, so each server start needs the network: it
stalls or fails offline, and it prints a ``Fetching N files`` bar although
nothing is downloaded.

The backends cannot take the snapshot folder in place of the repo id. Some
read the model from the repo id: mlx-audio reads Kokoro's type from the
"Kokoro" in it, and a snapshot folder is named by its commit hash. So
:func:`offline_resolve` passes the repo id on, makes sure the repo is in the
cache, and holds ``HF_HUB_OFFLINE`` on while the backend loads, so that the
backend's own ``snapshot_download`` reads the cache.

A repo that is not cached is downloaded once, and its snapshot in the cache
marks it as downloaded. A cached repo loads with no network access, except
that a caller may name files the cached snapshot lacks: :func:`fetch_pinned`
fetches those at the snapshot's commit, until the Hub has answered once.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import threading
import time
from urllib.parse import quote

# Serializes the offline-flag window: the flag is process-global, so a download
# (which needs it off) must not run while another load has flipped it on.
# Reentrant so a fetch nested inside the same thread's offline window degrades
# to huggingface_hub's OfflineModeIsEnabled error instead of deadlocking.
_OFFLINE_LOCK = threading.RLock()

# Seconds that one call of fetch_pinned may take in all: to connect, and to
# read each file it requests.
FETCH_TIMEOUT = 5.0
# A file larger than this is not fetched. Model cards and module configs
# are a few kB.
_FETCH_MAX_BYTES = 4 << 20
# The folder, in a repo's cache folder, that holds one file per snapshot
# commit for which fetch_pinned has its answer.
FETCHED_DIR = ".gmlx-fetched"

# What one request of fetch_pinned ends in. GOT: the file is stored, or the
# commit lacks it (200 or 404). FINAL: the Hub answered and the fetch ends,
# or it timed out with no request of the fetch answered; neither changes on
# a retry soon after. RETRY: the connection failed, the Hub gave another
# status, or it timed out after it answered a request, as a slow Hub does,
# and the next start tries again.
_GOT, _FINAL, _RETRY = "got", "final", "retry"

_COMMIT = re.compile(r"[0-9a-f]{40}")
_ETAG = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _offline_now(hf) -> bool:
    return bool(hf.constants.HF_HUB_OFFLINE)


def fetch_pinned(hf, model_ref: str, snapshot: str, wanted, *,
                 offline: bool) -> None:
    """Download into ``snapshot``, a cache snapshot folder of the repo
    ``model_ref``, the files that ``wanted(snapshot)`` names and the folder
    lacks.

    Each request names the snapshot's commit, so it gets that commit's file
    and moves no ref of the cache. Each is one GET with no retry, and the
    whole call gives up after about :data:`FETCH_TIMEOUT` seconds. A file
    the commit lacks is skipped, and any other failure ends the fetch.
    ``wanted`` is asked again after each round, since a fetched file may
    name another.

    The fetch runs once per snapshot once it has its answer. It then leaves
    a file named by the commit in the repo's :data:`FETCHED_DIR` folder,
    and with that file in place it requests nothing. The answer is a 200 or
    404 for each file, a 401 or 403, or a timeout before any request is
    answered. After a failed connection, another status, or a timeout after
    a request was answered, it leaves no mark, so the next start tries
    again. It keeps the files it stored, and asks then only for the others.
    With ``offline`` set, as by ``HF_HUB_OFFLINE``, or a cache folder it
    cannot write, it requests nothing and leaves no mark. The caller holds
    no lock."""
    snapshot = os.path.normpath(snapshot)
    commit = os.path.basename(snapshot)
    store = os.path.dirname(os.path.dirname(snapshot))
    if offline or not _COMMIT.fullmatch(commit):
        return
    mark = os.path.join(store, FETCHED_DIR, commit)
    if os.path.exists(mark):
        return
    names = _missing(snapshot, wanted)
    if not names:
        return                            # settled by the cached files
    if not _writable(store, snapshot):
        print(f"[server] {model_ref}: {', '.join(names)} not fetched, since "
              f"the HF cache folder {store} is read-only",
              file=sys.stderr, flush=True)
        return
    import httpx
    from huggingface_hub.utils import build_hf_headers

    deadline = time.monotonic() + FETCH_TIMEOUT
    tried: set = set()
    outcome = _GOT
    answered = False
    with httpx.Client(follow_redirects=True,
                      headers=build_hf_headers(library_name="gmlx")) as client:
        while names and outcome == _GOT:
            for name in names:
                tried.add(name)
                outcome = _fetch_one(hf, client, model_ref, commit, store,
                                     snapshot, name, deadline, mark, answered)
                if outcome != _GOT:
                    break
                answered = True
            names = [n for n in _missing(snapshot, wanted) if n not in tried]
    if outcome != _RETRY and not _touch(mark):
        print(f"[server] {model_ref}: {mark} could not be written, so the "
              f"next start tries again", file=sys.stderr, flush=True)


def _missing(snapshot: str, wanted) -> list[str]:
    try:
        names = list(wanted(snapshot))
    except Exception:                     # noqa: BLE001 (the load reports it)
        return []
    return [n for n in names if not os.path.isfile(_inside(snapshot, n) or "")]


def _inside(folder: str, name: str) -> str | None:
    """``name``, a path relative to ``folder`` with ``/`` separators, joined
    to it, or None when it is not such a path."""
    parts = name.split("/")
    if "\\" in name or any(p in ("", ".", "..") for p in parts):
        return None
    return os.path.join(folder, *parts)


def _fetch_one(hf, client, model_ref, commit, store, snapshot, name,
               deadline, mark, answered) -> str:
    """One GET of ``name`` at ``commit``, given up at ``deadline`` (a
    ``time.monotonic`` value). ``answered``: an earlier request of the fetch
    was answered. Returns what it ended in: _GOT, _FINAL or _RETRY."""
    import httpx

    dest = _inside(snapshot, name)
    if dest is None:
        return _GOT
    url = (f"{hf.constants.ENDPOINT.rstrip('/')}/{model_ref}/resolve/"
           f"{commit}/{quote(name)}")
    outcome = _FINAL
    try:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("the fetch ran out of time")
        with client.stream("GET", url, timeout=left) as r:
            if r.status_code == 404:
                return _GOT               # the commit has no such file
            if r.status_code not in (200, 401, 403):
                outcome = _RETRY
            if r.status_code != 200:
                raise OSError(f"HTTP {r.status_code}")
            body = bytearray()
            for chunk in r.iter_bytes():
                body += chunk
                if len(body) > _FETCH_MAX_BYTES:
                    raise OSError("the file is too large")
                if time.monotonic() > deadline:
                    raise TimeoutError("the read timed out")
            etag = _etag(r)
        outcome = _RETRY                  # a failure to store it
        _store(store, dest, bytes(body), etag)
    except Exception as exc:              # noqa: BLE001 (best effort)
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            outcome = _RETRY if answered else _FINAL
        elif isinstance(exc, httpx.TransportError):
            outcome = _RETRY              # no connection, or it broke
        then = ("the next start tries again" if outcome == _RETRY
                else f"delete {mark} to try again")
        print(f"[server] {model_ref}: {name} not fetched "
              f"({type(exc).__name__}: {exc}); {then}",
              file=sys.stderr, flush=True)
        return outcome
    print(f"[server] {model_ref}: fetched {name} at {commit[:12]}",
          file=sys.stderr, flush=True)
    return _GOT


def _etag(response) -> str | None:
    """The blob name the Hub gives the file: its git or LFS hash."""
    for r in (*response.history, response):
        for key in ("X-Linked-Etag", "ETag"):
            value = r.headers.get(key)
            if value:
                value = value.removeprefix("W/").strip('"')
                return value if _ETAG.fullmatch(value) else None
    return None


def _store(store: str, dest: str, body: bytes, etag: str | None) -> None:
    """Place a fetched file as huggingface_hub does: the content in
    ``blobs/<etag>``, linked from the snapshot. Without an etag the file
    itself goes in the snapshot."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if etag is None:
        _write_atomic(dest, body)
        return
    blob = os.path.join(store, "blobs", etag)
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    if not os.path.isfile(blob):
        _write_atomic(blob, body)
    link = f"{dest}.{os.getpid()}.{threading.get_ident()}.tmp"
    with contextlib.suppress(FileNotFoundError):
        os.remove(link)
    os.symlink(os.path.relpath(blob, os.path.dirname(dest)), link)
    os.replace(link, dest)


def _write_atomic(path: str, body: bytes) -> None:
    """Write ``path`` through a temporary file beside it. The file gets the
    mode of a new file under the umask, as huggingface_hub's files do."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _writable(store: str, snapshot: str) -> bool:
    """Whether a fetch can store its files and its mark: the repo's cache
    folder, the snapshot and the folders in the repo's that it writes to
    are writable."""
    folders = [store, snapshot] + [
        p for p in (os.path.join(store, "blobs"),
                    os.path.join(store, FETCHED_DIR)) if os.path.isdir(p)]
    return all(os.access(p, os.W_OK) for p in folders)


def _touch(path: str) -> bool:
    """Create the empty file ``path`` and its folder. False when it could
    not."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a"):
            pass
    except OSError:
        return False
    return True


@contextlib.contextmanager
def offline_resolve(model_ref: str, wanted=None):
    """Context manager: within the block, a backend that resolves
    ``model_ref`` (a repo id) reads the local cache and makes no Hub
    request.

    A local path, or an empty ref, is a no-op, as the backend has nothing to
    fetch. For a repo id, a cache lookup that makes no request comes first,
    and a repo that is not cached is downloaded once, with a log line. Then
    ``HF_HUB_OFFLINE`` is forced on, and progress bars off, for the load,
    and both are restored afterwards. The lookup, the download and the load
    hold one lock, so a download never runs while another load has forced
    the flag on.

    ``wanted``, a function of the snapshot folder, names files that a cached
    snapshot may lack, such as an encoder's model card. Between the lookup
    and the load, with the lock released, :func:`fetch_pinned` fetches them
    at the snapshot's commit. A repo downloaded here holds every file of
    its commit, so nothing more is fetched.
    """
    if not model_ref or os.path.exists(model_ref):
        yield
        return

    import huggingface_hub as hf
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import LocalEntryNotFoundError
    from huggingface_hub.utils import (disable_progress_bars,
                                       enable_progress_bars)
    try:
        from huggingface_hub.utils import are_progress_bars_disabled
    except ImportError:  # older huggingface_hub, which has no state query
        are_progress_bars_disabled = None

    with _OFFLINE_LOCK:
        # 1. The cached snapshot, or a download on a miss. No window is open
        # while this thread holds the lock, so the flag is the user's own.
        offline = _offline_now(hf)
        try:
            snapshot = hf.snapshot_download(model_ref, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"[server] {model_ref}: not in the HF cache, downloading "
                  f"once...", flush=True)
            hf.snapshot_download(model_ref)
            snapshot = None
    # 2. The files the cached snapshot lacks, with no lock held.
    if wanted is not None and snapshot is not None:
        fetch_pinned(hf, model_ref, snapshot, wanted, offline=offline)
    with _OFFLINE_LOCK:
        # 3. Load offline and quiet; the backend's snapshot_download stays
        # local.
        prev = hf.constants.HF_HUB_OFFLINE
        prev_bars_disabled = (are_progress_bars_disabled()
                              if are_progress_bars_disabled else False)
        hf.constants.HF_HUB_OFFLINE = True
        disable_progress_bars()
        try:
            yield
        finally:
            hf.constants.HF_HUB_OFFLINE = prev
            # Restore the caller's progress-bar state, as for the offline
            # flag, and never enable bars that were disabled before.
            if prev_bars_disabled:
                disable_progress_bars()
            else:
                enable_progress_bars()


@contextlib.contextmanager
def network_fetch_allowed():
    """Guard for load-path Hub fetches (hf_source config and processor
    pulls): holds the offline-window lock, so the fetch cannot run while
    another thread's :func:`offline_resolve` has forced ``HF_HUB_OFFLINE``
    on.

    Nesting this inside the same thread's :func:`offline_resolve` window is
    a caller bug, and the fetch then fails offline. The lock is reentrant,
    so that failure is a clear huggingface_hub error, not a deadlock.
    """
    with _OFFLINE_LOCK:
        yield
