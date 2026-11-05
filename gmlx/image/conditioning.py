"""Conditionings, their cache keys, and the byte-bounded caches of a
pipeline.

A conditioning is one encoded prompt. Its key is a sha256 over the family,
the encoder identity, the text adapter state, the template, the token ids,
the reference images, the tap layers and the dtype, so equal keys mean
equal features. The latents of the reference images are no part of the
key: a state the DiT derives from a conditioning is keyed on their hashes
too. The pipeline keeps conditionings, the caption state the
DiT derives from them, and the per-sigma modulation in ``ByteLRU`` caches,
each bounded in bytes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True, eq=False)
class Reference:
    """One reference image at the size its family runs it at: ``pixels``
    is (H, W, 4) uint8 with straight alpha, and ``sha256`` names its
    bytes and size. A ``sha256`` given that is not theirs is refused."""
    pixels: np.ndarray
    sha256: str = ""

    def __post_init__(self):
        px = np.ascontiguousarray(self.pixels)
        if px.dtype != np.uint8 or px.ndim != 3 or px.shape[2] != 4:
            raise ValueError("reference pixels must be (H, W, 4) uint8, "
                             f"not {px.dtype} {px.shape}")
        px.setflags(write=False)
        object.__setattr__(self, "pixels", px)
        h = hashlib.sha256(f"{px.shape[0]}x{px.shape[1]}:".encode())
        h.update(px.tobytes())
        if self.sha256 and self.sha256 != h.hexdigest():
            raise ValueError("the sha256 given is not that of the "
                             "reference pixels")
        object.__setattr__(self, "sha256", h.hexdigest())

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def identity(self) -> tuple:
        """What a conditioning key holds of the image."""
        return (self.sha256, self.height, self.width)


def latents_hash(latents: mx.array) -> str:
    """sha256 of latents (C, h, w): their shape and float32 bits."""
    a = np.ascontiguousarray(np.array(latents.astype(mx.float32)))
    h = hashlib.sha256(("x".join(str(d) for d in a.shape) + ":").encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Conditioning:
    """One encoded prompt: the encoder tap ``(L, D)`` of its tokens,
    unpadded, and its cache key. An empty key is never cached. An edit
    also holds ``image_slots``, the indices of the rows of ``features``
    that the encoder gave to image tokens, ascending, and ``refs``, the
    ``Reference`` of each image in prompt order. On a family whose
    encoder reads the prompt alone it has no image slots, and its key is
    that of the text. ``ref_latents`` are the latents ``(C, h, w)`` of
    each reference in model space, which the VAE adds after the encoder
    ran, and ``ref_hashes`` name their content: the state a DiT derives
    from the conditioning depends on them, and the key does not hold
    them."""
    features: mx.array
    key: str = ""
    ref_latents: tuple = ()
    image_slots: tuple = ()
    refs: tuple = ()
    ref_hashes: tuple = ()

    def __post_init__(self):
        if self.ref_latents and not self.ref_hashes:
            object.__setattr__(self, "ref_hashes", tuple(
                latents_hash(z) for z in self.ref_latents))
        if len(self.ref_hashes) != len(self.ref_latents):
            raise ValueError(f"{len(self.ref_hashes)} hashes for "
                             f"{len(self.ref_latents)} reference latents")

    @property
    def pending(self) -> bool:
        """Whether the conditioning has reference images, in image slots
        or beside the text, and no reference latents yet."""
        return bool(self.image_slots or self.refs) and not self.ref_latents


def conditioning_key(*, family: str, encoder: str, adapters: str = "",
                     template: str, ids, tap_layers, dtype: str,
                     images=()) -> str:
    """sha256 of the canonical JSON of everything a conditioning depends
    on. ``images`` holds (sha256, height, width) per reference image, in
    prompt order."""
    doc = {"family": family, "encoder": encoder, "adapters": adapters,
           "template": template, "ids": [int(i) for i in ids],
           "tap_layers": [int(n) for n in tap_layers], "dtype": dtype}
    # A text prompt keeps the key it had before reference images.
    if images:
        doc["images"] = [list(i) for i in images]
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def array_bytes(arrays) -> int:
    return sum(int(a.nbytes) for a in arrays)


# The clock of the uses of every ByteLRU, so that the uses of two caches
# compare.
_USES = itertools.count()


class ByteLRU:
    """A least recently used map that holds at most ``max_bytes`` of
    values. An entry larger than the bound is not kept. A caller with a
    rule of its own puts with ``shed`` off, which keeps the entry whatever
    its bytes and sheds nothing, and sheds through ``shed_others``. A put
    never sheds the entries of the keys given to ``keep``, and an entry
    that does not fit beside them is not kept and sheds nothing."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict = OrderedDict()
        self._used: dict = {}
        self._bytes = 0
        self._kept: frozenset = frozenset()
        self._lock = threading.Lock()

    def keep(self, keys) -> None:
        """The keys whose entries a put never sheds, in place of any given
        before. Their entries still count toward the bound."""
        with self._lock:
            self._kept = frozenset(keys)

    @property
    def kept(self) -> frozenset:
        """The keys given to ``keep``."""
        return self._kept

    def __len__(self) -> int:
        return len(self._items)

    @property
    def bytes(self) -> int:
        return self._bytes

    def get(self, key):
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            self._items.move_to_end(key)
            self._used[key] = next(_USES)
            return hit[0]

    def peek(self, key):
        """The value at ``key``, or None, without a use: the order of use
        does not change."""
        with self._lock:
            hit = self._items.get(key)
            return None if hit is None else hit[0]

    def used(self, key) -> int:
        """When the entry at ``key`` was last put or got, on a clock that
        every ByteLRU shares, or -1 without one."""
        with self._lock:
            return self._used.get(key, -1)

    def nbytes(self, key) -> int:
        """Bytes of the entry at ``key``, 0 without one. The order of use
        does not change."""
        with self._lock:
            hit = self._items.get(key)
            return 0 if hit is None else hit[1]

    def sizes(self) -> dict:
        """{key: bytes} of every entry, read in one hold of the cache's
        lock. The order of use does not change."""
        with self._lock:
            return {k: n for k, (_v, n) in self._items.items()}

    def keys(self, match) -> list:
        """The keys that ``match`` takes, the least recently used first.
        The order of use does not change."""
        with self._lock:
            return [key for key in self._items if match(key)]

    def newest(self, match):
        """The most recently used key that ``match`` takes, or None. The
        order of use does not change."""
        with self._lock:
            for key in reversed(self._items):
                if match(key):
                    return key
        return None

    def _pop(self, key) -> int:
        """Remove the entry at ``key`` and return its bytes."""
        n = self._items.pop(key)[1]
        del self._used[key]
        self._bytes -= n
        return n

    def put(self, key, value, nbytes: int, shed: bool = True) -> None:
        with self._lock:
            if key in self._items:
                self._pop(key)
            nbytes = int(nbytes)
            if shed and self._bytes + nbytes > self.max_bytes:
                kept = sum(n for k, (_, n) in self._items.items()
                           if k in self._kept) if self._kept else 0
                if nbytes > self.max_bytes - kept:
                    return
                for k in [k for k in self._items if k not in self._kept]:
                    if self._bytes + nbytes <= self.max_bytes:
                        break
                    self._pop(k)
            self._items[key] = (value, nbytes)
            self._used[key] = next(_USES)
            self._bytes += nbytes

    def shed_for(self, nbytes: int, pinned, before: int | None = None) -> bool:
        """Shed the entries whose key is not in ``pinned``, the least
        recently used first, until ``nbytes`` more fit, and return
        whether they fit. With ``before``, a use of the clock of ``used``,
        only the entries last used before it go."""
        with self._lock:
            for k in list(self._items):
                if self._bytes + nbytes <= self.max_bytes:
                    break
                if before is not None and self._used[k] > before:
                    break
                if k not in pinned:
                    self._pop(k)
            return self._bytes + nbytes <= self.max_bytes

    def shed_others(self, own, room: int) -> None:
        """Shed the entries whose key is not in ``own``, the least
        recently used first, until they hold at most ``room`` bytes. The
        entries of ``own`` stay whatever their bytes."""
        with self._lock:
            others = [k for k in self._items if k not in own]
            held = sum(self._items[k][1] for k in others)
            for k in others:
                if held <= room:
                    break
                held -= self._pop(k)

    def shed_by(self, nbytes: int, rank) -> int:
        """Drop entries until ``nbytes`` are gone, the lowest ``rank`` of
        their key first and the least recently used first within a rank.
        Returns the bytes dropped."""
        with self._lock:
            gone = 0
            for k in sorted(self._items, key=rank):
                if gone >= nbytes:
                    break
                gone += self._pop(k)
            return gone

    def drop_where(self, match) -> None:
        """Drop the entries whose value ``match`` takes."""
        with self._lock:
            for k in [k for k, v in self._items.items() if match(v[0])]:
                self._pop(k)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._used.clear()
            self._bytes = 0
