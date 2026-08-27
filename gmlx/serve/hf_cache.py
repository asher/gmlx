"""Load an stt/tts/embeddings model without a Hugging Face Hub round-trip when
it is already cached.

The optional audio/embeddings backends (mlx-audio, mlx-whisper, mlx-embeddings)
each take a repo id and call ``snapshot_download`` themselves. With
``local_files_only=False`` (the default) that issues a live ``api.repo_info``
request to huggingface.co on *every* load - a network dependency on each server
start that stalls/fails offline and prints a ``Fetching N files`` bar, even
though nothing is downloaded.

We can't hand the backend the resolved snapshot directory instead: several
backends infer the architecture from the *path name* (e.g. mlx-audio reads
Kokoro's type from the "Kokoro" in the repo id; the commit-hash snapshot dir
defeats that). So :func:`offline_resolve` keeps the repo id flowing to the
backend and instead (1) makes sure the repo is in the local cache - downloading
it once, online, only on a genuine miss - then (2) forces hf_hub offline for the
duration of the load so the backend's own ``snapshot_download`` resolves purely
from cache. The HF cache snapshot is the persistent "already downloaded" marker;
no separate marker file is kept.
"""
from __future__ import annotations

import contextlib
import os
import threading

# Serializes the offline-flag window: the flag is process-global, so a download
# (which needs it off) must not run while another load has flipped it on.
# Reentrant so a fetch nested inside the same thread's offline window degrades
# to huggingface_hub's OfflineModeIsEnabled error instead of deadlocking.
_OFFLINE_LOCK = threading.RLock()


@contextlib.contextmanager
def offline_resolve(model_ref: str):
    """Context manager: within the block, a backend that resolves ``model_ref``
    (a repo id) does so from the local cache only - no Hub round-trip.

    A local path (or empty ref) is a no-op: the backend already has nothing to
    fetch. For a repo id, the cache is ensured first (a one-time, logged
    ``snapshot_download`` on a true miss; a no-network probe otherwise), then
    ``HF_HUB_OFFLINE`` is forced on (and progress bars off) for the load and
    restored afterwards. Serialized so the ensure-cache download of one model
    can't be starved offline by another model's load flipping the global flag.
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
    except ImportError:  # older huggingface_hub - no state query
        are_progress_bars_disabled = None

    with _OFFLINE_LOCK:
        # 1. Ensure cached - network only on a genuine miss (flag still off here).
        try:
            hf.snapshot_download(model_ref, local_files_only=True)
        except LocalEntryNotFoundError:
            print(f"[server] {model_ref}: not in HF cache - downloading once...",
                  flush=True)
            hf.snapshot_download(model_ref)
        # 2. Load offline + quiet; the backend's snapshot_download stays local.
        prev = hf.constants.HF_HUB_OFFLINE
        prev_bars_disabled = (are_progress_bars_disabled()
                              if are_progress_bars_disabled else False)
        hf.constants.HF_HUB_OFFLINE = True
        disable_progress_bars()
        try:
            yield
        finally:
            hf.constants.HF_HUB_OFFLINE = prev
            # Restore the caller's prior progress-bar state (mirror the offline
            # flag); don't force-enable over a globally-disabled setting.
            if prev_bars_disabled:
                disable_progress_bars()
            else:
                enable_progress_bars()


@contextlib.contextmanager
def network_fetch_allowed():
    """Guard for load-path Hub fetches (hf_source config/processor pulls):
    holds the offline-window lock so the fetch can't run while another
    thread's :func:`offline_resolve` has forced ``HF_HUB_OFFLINE`` on.

    Nesting this inside the same thread's :func:`offline_resolve` window is
    a caller bug (the fetch will fail offline); the reentrant lock keeps
    that failure a clear huggingface_hub error rather than a deadlock.
    """
    with _OFFLINE_LOCK:
        yield
