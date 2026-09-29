"""A trim of a rotating cache in temporal order drops the trimmed columns.

``RotatingKVCache`` and ``BatchRotatingKVCache`` (both origins) trim by
moving ``_idx`` and the offsets back, and keep the trimmed columns in the
buffer. A wider update leaves the buffer in temporal order with ``_idx``
at its length, and past a full window the buffer then holds
``max_size - 1`` older keys plus the update. When a speculative rollback
trims the rejected drafts of such an update, the next wider update drops
them: the batch ``_update_concat`` slices the buffer to ``_idx``, and the
single-row ``_temporal_order`` moves them to the front, where the window
trim takes them. A one-token update does not. ``_update_in_place`` trims
``keys.shape[2] - max_size`` of the oldest keys and rotates, so the
rejected drafts stay in the ring and are attended while valid keys are
lost in their place, and ``make_mask``, which reads its trim from
``_idx``, matches neither.

A batch reaches this when a request joins a speculating batch after a
rejected draft: the stock ``extend`` keeps the trimmed columns, and the
width gate turns the next rounds into one-token steps.

The wrap slices the keys and values to ``_idx`` after a trim that only
moved ``_idx`` back through a buffer in temporal order: a batch buffer that
is not rotated, or a single-row buffer whose ``_idx`` was at its length
before the trim. It leaves a buffer below ``max_size`` alone when the
buffer no longer starts at the first token, since ``_update_in_place``
would then grow it from an offset past the window.

A trim deeper than ``_idx`` raises ValueError and leaves the cache as it
was. It can only reach a buffer that has wrapped or dropped tokens off its
window, where the stock trim leaves ``_idx`` negative: the tokens it would
keep are no longer in the buffer. ``is_trimmable`` reports such a cache as
not trimmable, and no rollback trims one that deep.

Install is idempotent; GMLX_ROTATING_TRIM_FIX=0 disables it.
"""

from __future__ import annotations

import os
from collections.abc import Callable

_PATCH_FLAG = "_gmlx_trim_drops_tail"


def _in_temporal_order(cache) -> bool:
    keys = cache.keys
    if hasattr(cache, "rotated"):
        return not cache.rotated
    return cache._idx == keys.shape[2]


def _total(cache) -> int:
    return cache._offset if hasattr(cache, "rotated") else cache.offset


def _can_slice(cache) -> bool:
    """Whether ``_update_in_place`` still works on a buffer cut to
    ``_idx``: at least a full window, or a buffer that starts at the first
    token, which it grows from the offset."""
    if cache._idx >= cache.max_size:
        return True
    return _total(cache) == cache._idx


def _wrap(orig: Callable) -> Callable:
    def trim(self, n):
        keys = getattr(self, "keys", None)
        idx = getattr(self, "_idx", None)
        # An armed DeepSeek-V4 undo log restores its snapshot in place of
        # the stock trim, so the depth check leaves it alone.
        if (keys is not None and getattr(self, "_mtp_undo", None) is None
                and min(_total(self), n) > idx):
            raise ValueError(
                f"{type(self).__name__}.trim({n}) goes past the write "
                f"position {idx} of a buffer that has wrapped or dropped "
                f"tokens off its window, so the trimmed state cannot be "
                f"held")
        in_order = keys is not None and _in_temporal_order(self)
        got = orig(self, n)
        if (in_order and got and self.keys is keys
                and self._idx == idx - got and keys.shape[2] > self._idx
                and _can_slice(self)):
            self.keys = keys[..., : self._idx, :]
            self.values = self.values[..., : self._idx, :]
        return got

    trim.__dict__[_PATCH_FLAG] = True
    trim.__dict__["_gmlx_orig"] = orig
    return trim


def install_rotating_trim_fix() -> bool:
    """Wrap ``trim`` on every loaded origin of ``RotatingKVCache`` and
    ``BatchRotatingKVCache``.

    Idempotent per class (a class flag, since the DeepSeek-V4 undo log may
    wrap ``trim`` again on top) and re-scanned on every call, so an origin
    module imported after the first install is patched by the next. The
    wrap composes with the undo log in either order: it slices only after
    a trim that kept the same buffer.
    """
    if os.environ.get("GMLX_ROTATING_TRIM_FIX") == "0":
        return False
    from gmlx.cache.compat import cache_types

    found = False
    for kind in ("RotatingKVCache", "BatchRotatingKVCache"):
        try:
            classes = cache_types(kind)
        except AttributeError:
            continue
        for cls in classes:
            orig = cls.__dict__.get("trim")
            if orig is None:
                continue
            found = True
            if cls.__dict__.get(_PATCH_FLAG, False):
                continue
            cls.trim = _wrap(orig)
            setattr(cls, _PATCH_FLAG, True)
    return found
