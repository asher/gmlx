"""Conditionings, their cache keys, and the byte-bounded caches of a
pipeline.

A conditioning is one encoded prompt. Its key is a sha256 over the family,
the encoder identity, the text adapter state, the template, the token ids,
the reference images, the tap layers and the dtype, so equal keys mean
equal features. The pipeline keeps conditionings, the caption state the
DiT derives from them, and the per-sigma modulation in ``ByteLRU`` caches,
each bounded in bytes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Conditioning:
    """One encoded prompt: the encoder tap ``(L, D)`` of its tokens,
    unpadded, and its cache key. An empty key is never cached. An edit
    also holds ``ref_latents``, the latents ``(C, h, w)`` of each
    reference image in model space and in prompt order, and
    ``image_slots``, the indices of the rows of ``features`` that the
    encoder gave to image tokens, ascending."""
    features: mx.array
    key: str = ""
    ref_latents: tuple = ()
    image_slots: tuple = ()


def conditioning_key(*, family: str, encoder: str, adapters: str = "",
                     template: str, ids, tap_layers, dtype: str,
                     images=()) -> str:
    """sha256 of the canonical JSON of everything a conditioning depends
    on. ``images`` holds (sha256, height, width) per reference image."""
    doc = {"family": family, "encoder": encoder, "adapters": adapters,
           "template": template, "ids": [int(i) for i in ids],
           "tap_layers": [int(n) for n in tap_layers], "dtype": dtype,
           "images": [list(i) for i in images]}
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def array_bytes(arrays) -> int:
    return sum(int(a.nbytes) for a in arrays)


class ByteLRU:
    """A least recently used map that holds at most ``max_bytes`` of
    values. An entry larger than the bound is not kept. A caller with a
    rule of its own puts with ``shed`` off, which keeps the entry whatever
    its bytes and sheds nothing, and sheds through ``shed_others``."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

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
            return hit[0]

    def nbytes(self, key) -> int:
        """Bytes of the entry at ``key``, 0 without one. The order of use
        does not change."""
        with self._lock:
            hit = self._items.get(key)
            return 0 if hit is None else hit[1]

    def put(self, key, value, nbytes: int, shed: bool = True) -> None:
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            if shed and nbytes > self.max_bytes:
                return
            self._items[key] = (value, int(nbytes))
            self._bytes += int(nbytes)
            while shed and self._bytes > self.max_bytes:
                _, (_, n) = self._items.popitem(last=False)
                self._bytes -= n

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
                n = self._items.pop(k)[1]
                held -= n
                self._bytes -= n

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
