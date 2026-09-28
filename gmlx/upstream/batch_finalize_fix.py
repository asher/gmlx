"""Ragged right-trim rebinds a batch cache's left padding.

``finalize`` on the batch KV caches (``BatchKVCache``,
``BatchRotatingKVCache``, ``BatchQuantizedKVCache``, both origins) rolls a
right-padded tail to the front of each row and adds it to ``left_padding``
with ``+=``, which updates the array in place. The rotating cache takes the
tail from ``prepare(lengths=...)``, the others from ``right_padding``.

The host-side pad memos key on the identity of ``cache.left_padding``: the
qwen3.5 pad and decode memos, verify_fold, cascade and sparse SDPA, and
gemma4 batched SDPA. After the trim they keep serving the old pads, so a
row's rolled-in rejected keys stay attended. A batched speculative rollback
with different accept counts per row runs this trim every round; the next
decode of the row that accepted less then diverged from single-stream.

The fix rebinds ``left_padding`` to a copy before a finalize that carries a
tail, so the stock ``+=`` lands on the copy. Each memo misses once and
rereads the values, and an earlier holder of the old array, such as a
snapshot, keeps the old values.
"""

from __future__ import annotations

import mlx.core as mx

_PATCH_FLAG = "_gmlx_finalize_rebinds_left_padding"
_KINDS = ("BatchKVCache", "BatchRotatingKVCache", "BatchQuantizedKVCache")


def install_batch_finalize_fix() -> bool:
    """Wrap ``finalize`` on every loaded origin of each batch KV kind.

    Idempotent per class (the patch flag) and re-scanned on every call, so
    an origin module imported after the first install is patched by the
    next.
    """
    from gmlx.cache.compat import cache_types

    for kind in _KINDS:
        try:
            classes = cache_types(kind)
        except AttributeError:
            continue
        for cls in classes:
            orig = cls.__dict__.get("finalize")
            if orig is None or getattr(orig, _PATCH_FLAG, False):
                continue

            def finalize(self, _orig=orig):
                lp = getattr(self, "left_padding", None)
                padded = (getattr(self, "_right_padding", None) is not None
                          or getattr(self, "_lengths", None) is not None)
                if padded and isinstance(lp, mx.array):
                    self.left_padding = mx.array(lp)
                _orig(self)

            finalize.__dict__[_PATCH_FLAG] = True
            cls.finalize = finalize
    return True
