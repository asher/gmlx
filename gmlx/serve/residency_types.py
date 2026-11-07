"""The residency pool's data: entries, ledgers, figures, kinds and refusals."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

# One logger for every residency module.
_log = logging.getLogger("gmlx.serve.residency")


# Scratch target for an in-flight stock load/teardown, isolated to the building
# context so it never leaks into concurrent resident-model requests.
_build_scratch: ContextVar["_Scratch | None"] = ContextVar(
    "_kq_build_scratch", default=None
)


class _Scratch:
    """Mutable stand-in for the three pooled runtime fields during a stock
    load or teardown. Empty ``model_cache`` makes stock ``get_cached_model``
    take the cache-miss path (no spurious unload of a resident model)."""

    __slots__ = ("response_generator", "model_cache", "apc_manager")

    def __init__(self):
        self.response_generator: Any = None
        # A dict, or the ModelCacheRegistry stock load turns it into.
        self.model_cache: Any = {}
        self.apc_manager: Any = None


class ModelBusyError(RuntimeError):
    """An explicit unload targeted a model with in-flight requests."""

    def __init__(self, model_path, in_flight: int):
        self.model_path = model_path
        self.in_flight = in_flight
        super().__init__(
            f"{model_path} has {in_flight} in-flight request(s)")


def profile_label(cache_key: tuple) -> str:
    """A short, stable name for what distinguishes one resident entry
    from another on the same GGUF: the adapter's basename and/or a hash
    of the load signature (kv bits, mmproj, drafter ...), ``default``
    when neither applies. Unique per entry (the pool is keyed on
    ``cache_key``) and unchanged for the entry's lifetime and across
    reloads of the same config - a metrics label, unlike ``seq``, which
    is the LRU clock and re-stamps on every acquire. Every key component
    but the path contributes: adapter, model kind (when not ``auto``),
    load-signature extras."""
    import hashlib

    parts = []
    adapter = cache_key[1] if len(cache_key) > 1 else None
    if adapter:
        parts.append(os.path.basename(str(adapter)))
    # model_kind is part of the key too: an --embeddings / --rerank / audio
    # route pointed at a GGUF that also serves chat is a second entry with
    # the same adapter and extras. Only ``auto`` (what gmlx's own loads
    # pass) collapses into ``default``; upstream's ``text_generation`` is a
    # different key and labels itself, so the label stays unique even if
    # the stock lifespan preload ever fires alongside ours.
    kind = cache_key[2] if len(cache_key) > 2 else None
    if kind and str(kind) != "auto":
        parts.append(str(kind))
    extras = tuple(cache_key[3:])
    if extras:
        parts.append(hashlib.blake2b(repr(extras).encode(),
                                     digest_size=4).hexdigest())
    return "+".join(parts) or "default"


# The cache-key kind of an image model entry. mlx-vlm's own
# ``get_cached_model`` kinds ``image_generation`` and ``image_edit`` load
# its safetensors pipelines; no gmlx route passes either one.
IMAGE_KIND = "gmlx_image"


# The cache-key kind of a video model entry. Its pipeline runs on the
# video worker, so both kinds are worker entries, each on its own worker.
VIDEO_KIND = "gmlx_video"


_WORKER_KINDS = frozenset({IMAGE_KIND, VIDEO_KIND})


# The workers in the order a thread that holds both takes them.
_WORKER_ORDER = (IMAGE_KIND, VIDEO_KIND)


_WORKER_NAMES = {IMAGE_KIND: "image-worker", VIDEO_KIND: "video-worker"}


_WORKER_WORDS = {IMAGE_KIND: "image", VIDEO_KIND: "video"}


_STOCK_IMAGE_KINDS = frozenset({"image_generation", "image_edit"})


class LoraRefused(Exception):
    """LoRA bytes a turn or a route cannot grant: ``status`` 413 when no
    unload makes room, 503 when one can, with ``retry_after`` seconds.
    ``transient`` marks a 503 that ends by itself, as models in use
    finish, a load lands or a worker frees; one that needs an act, as an
    unload or other apps closed, is not."""

    def __init__(self, status: int, message: str,
                 retry_after: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after
        self.transient = transient


def gb(n: float, up: bool) -> str:
    """``n`` bytes in decimal GB to one decimal, rounded up or down."""
    import math

    x = max(0.0, float(n)) / 1e9 * 10
    return f"{(math.ceil(x) if up else math.floor(x)) / 10:.1f}"


class _Store:
    """The step routes' upload store as the owner of a turn. It keeps no
    ledger: the store counts its own bytes."""

    lora = None

    def __repr__(self) -> str:
        return "uploads"


STORE = _Store()


class _Ledger:
    """The LoRA bytes of an image or video entry past its footprint: the
    request keys its caches hold (``keys``), the excess of its own keys
    over the footprint's own term, and the bytes its job booked at its
    turn and has not allocated yet. Set from the caches on the worker,
    under the pool's lock."""

    __slots__ = ("keys", "own_term", "own", "excess", "pending", "base")

    def __init__(self, own_term: int = 0):
        self.keys: dict = {}
        self.own_term = int(own_term)
        self.own = 0
        self.excess = 0
        self.pending = 0
        # What the ledger held when the job booked ``pending``.
        self.base = 0

    @property
    def held(self) -> int:
        return sum(self.keys.values()) + self.excess

    def set(self, figures) -> None:
        keys, own = figures
        self.keys = {k: int(n) for k, n in keys.items()}
        self.own = int(own)
        self.excess = max(0, self.own - self.own_term)


@dataclass(frozen=True)
class _Figures:
    """What the pool holds, as of the last ``_lock`` hold that changed it.
    Routes and uploads read it with no lock."""
    budget: int = 0
    entries: tuple = ()
    held: tuple = ()            # each entry's ledger bytes, as ``entries``
    footprints: int = 0
    lora: int = 0               # the ledgers and their pending bytes
    building: int = 0
    tearing: int = 0            # the victims whose teardown runs
    incoming: int = 0           # the upload bodies admitted, not put

    @property
    def resident(self) -> int:
        return self.footprints + self.lora + self.building + self.tearing


@dataclass
class _Entry:
    cache_key: tuple
    model_path: str
    # A chat entry's stock load state; an image entry has none of the three.
    model_cache: Any = None     # dict or ModelCacheRegistry (see _Scratch)
    response_generator: Any = None
    apc_manager: Any = None
    pinned: bool = False
    seq: int = 0
    footprint: int = 0          # resident weight bytes (file size; a streaming
                                # entry: every-token weights + decode arena)
    ttl: float | None = None  # idle auto-unload seconds (None/0 => never)
    last_access: float = 0.0    # monotonic time of the last acquire/touch
    busy: int = 0               # in-flight refcount; busy entries are never LRU-evicted
    retained: int = 0           # of ``busy``: process-lifetime holds (preload), not requests
    loaded_as: str | None = None  # the configured ``id[@profile]`` whose request built it
    kv_policy: Any = None       # ServeKvPolicy (None when kv-bits is off)
    kind: str = "chat"          # "chat", IMAGE_KIND or VIDEO_KIND
    pipeline: Any = None        # an image or video entry's served pipeline
    priced_up: bool = False     # the install streamed less than predicted
    lora: _Ledger | None = None  # an image or video entry's LoRA bytes
    tearing: int = 0            # counted until its teardown or release ends

    def __post_init__(self):
        if self.model_cache is None:
            self.model_cache = {}

    @property
    def is_image(self) -> bool:
        """Whether the entry is a pipeline on a worker: an image model on
        the image worker or a video model on the video worker."""
        return self.kind in _WORKER_KINDS


def _owner_kind(owner) -> str:
    """The worker kind of a turn's owner: the upload store's is the image
    worker's, where its store turn runs."""
    kind = getattr(owner, "kind", None)
    return kind if kind in _WORKER_KINDS else IMAGE_KIND


def _mine(e, kind: str) -> bool:
    """Whether a turn on the worker of ``kind`` may shed or evict entry
    ``e``: a chat entry, or one of its own worker."""
    return not e.is_image or e.kind == kind
