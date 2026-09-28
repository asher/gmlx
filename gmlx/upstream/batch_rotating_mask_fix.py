"""A rotated batch rotating cache masks a wider update with the trim that
update applies.

``BatchRotatingKVCache`` (both origins) writes a one-token update in place.
Once the window is full the buffer rotates, and ``_idx`` becomes the write
position in the ring. A wider update goes through ``_update_concat``,
which first puts the buffer in temporal order, so ``_idx`` becomes the
buffer length, and then trims ``_idx - max_size + 1`` keys and takes the
same count off ``left_padding``. Stock ``make_mask`` runs before the update
and computes that trim from the ring position, so after a rotation its
mask keeps one more column of left padding than the update leaves. A
left-padded row whose first real key is still inside the window loses
that key for every query of the update.

Any batch that writes a one-token update after its window fills and then a
wider one reaches this: a prefill chunk of one column followed by a wider
chunk (a block that starts one column past a chunk boundary), or the
one-column final prefill forward or a plain decode step followed by a
speculative verify block.

The wrap runs the stock body with ``_idx`` at the buffer length whenever
the buffer is rotated and the update takes the concat path, which is the
value ``_temporal_order`` sets before the trim.
"""

from __future__ import annotations

from collections.abc import Callable

_PATCH_FLAG = "_gmlx_mask_matches_concat_trim"


def _takes_concat(cache, n: int) -> bool:
    """True when an ``n``-token update of ``cache`` goes through
    ``_update_concat``: a wider update, or a one-token update while
    right-padding lengths are set (the mlx-lm origin raises on that one in
    ``_update_in_place``, so its mask is never used)."""
    return n > 1 or getattr(cache, "_lengths", None) is not None


def _wrap(orig: Callable) -> Callable:
    def make_mask(self, N, window_size=None, return_array=False):
        keys = getattr(self, "keys", None)
        if not (getattr(self, "rotated", False) and keys is not None
                and _takes_concat(self, N)):
            return orig(self, N, window_size=window_size,
                        return_array=return_array)
        idx = self._idx
        self._idx = int(keys.shape[2])
        try:
            return orig(self, N, window_size=window_size,
                        return_array=return_array)
        finally:
            self._idx = idx

    make_mask.__dict__[_PATCH_FLAG] = True
    make_mask.__dict__["_gmlx_orig"] = orig
    return make_mask


def install_batch_rotating_mask_fix() -> bool:
    """Wrap ``make_mask`` on every loaded origin of ``BatchRotatingKVCache``.

    Idempotent per class (the patch flag) and re-scanned on every call, so
    an origin module imported after the first install is patched by the
    next.
    """
    from gmlx.cache.compat import cache_types

    try:
        classes = cache_types("BatchRotatingKVCache")
    except AttributeError:
        return False
    for cls in classes:
        orig = cls.__dict__.get("make_mask")
        if orig is None or getattr(orig, _PATCH_FLAG, False):
            continue
        cls.make_mask = _wrap(orig)
    return True
