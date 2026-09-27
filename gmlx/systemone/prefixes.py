"""State prefixes kept between letter decisions.

A letter decision prefills its state prefix once: the chat header and the
state. The server's APC manager keeps that prefix for the next decision on
the same state, in the checkpoint tier when the model has recurrent layers
and in the exact tier when every layer is plain attention. The records
carry their own salt, so a chat request never adopts one and a decision
never adopts a chat record."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Optional

from gmlx import lora_rows

from . import ar_reader


def prefix_salt(scales=None) -> int:
    """The APC salt of a decision prefix: the reader version, the forward
    size, and the request's adapter scales when any is set."""
    key = repr(("gmlx-letters", ar_reader.READER_VERSION, ar_reader.FORWARD_TOKENS))
    salt = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
    if scales and any(scales):
        salt ^= lora_rows.lora_salt(scales)
    return salt


class Prefixes:
    """No reuse: every decision computes its prefix. ``tier`` names where
    prefixes are kept, for the diagnostics."""

    tier = "off"

    def lookup(self, ids: list[int]) -> Optional[list]:
        return None

    def store(self, ids: list[int], cache) -> bool:
        return False


class ApcPrefixes(Prefixes):
    """Prefixes in an APC manager. A lookup takes only a record of exactly
    the prefix, never a shorter one."""

    def __init__(self, manager, salt: int, make_cache: Callable[[], Any]):
        from gmlx.cache.compat import cache_types
        from gmlx.cache.snapshot import ckpt_layout

        self.manager = manager
        self.salt = int(salt)
        probe = make_cache()
        self.layout = ckpt_layout(probe, int(manager.block_size))
        kv = cache_types("KVCache")
        if self.layout is not None:
            self.tier = "ckpt"
        elif probe and all(type(c) in kv for c in probe):
            self.tier = "exact"
        else:
            self.tier = "unsupported"

    def lookup(self, ids):
        from gmlx.cache.snapshot import ckpt_lookup

        n = len(ids)
        if n < 2:
            return None
        # One token past the prefix, so only a record at exactly n matches.
        query = list(ids) + [0]
        if self.tier == "ckpt":
            warm, p = ckpt_lookup(self.manager, query, extra_hash=self.salt,
                                  min_prefix_tokens=n - 1, layout=tuple(self.layout or ()))
        elif self.tier == "exact":
            warm, p = self.manager.lookup_exact_cache(
                query, extra_hash=self.salt, min_prefix_tokens=n - 1)
        else:
            return None
        return warm if warm is not None and p == n else None

    def store(self, ids, cache):
        from gmlx.cache.snapshot import ckpt_store

        if len(ids) < 2:
            return False
        if self.tier == "ckpt":
            return ckpt_store(self.manager, ids, cache, extra_hash=self.salt,
                              skeleton_disk=False, kind="decision") > 0
        if self.tier == "exact":
            return bool(self.manager.store_exact_cache(ids, cache, extra_hash=self.salt))
        return False
