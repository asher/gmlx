"""batch_rotating_mask_fix: a rotated BatchRotatingKVCache masks a wider
update with the trim that update applies.

A one-token update after the window fills rotates the buffer. The next
wider update puts the buffer in temporal order and trims it, and the mask
built before that update must hide exactly the left padding the update
leaves. Each key carries its column, so the expected mask follows from
the columns the update returns: causal, inside the window, and not
padding. CPU, both class origins.
"""

from __future__ import annotations

import mlx.core as mx
import mlx_vlm.models.cache  # noqa: F401  (both origins for cache_types)
import numpy as np
import pytest

from gmlx.cache.compat import cache_types
from gmlx.upstream.batch_rotating_mask_fix import (
    install_batch_rotating_mask_fix,
)

_W = 8


@pytest.fixture(autouse=True)
def installed():
    install_batch_rotating_mask_fix()


def _keys(start: int, n: int, rows: int) -> mx.array:
    """Keys whose first feature is the column plus one (0 marks a slot the
    cache has not written)."""
    cols = mx.arange(start + 1, start + n + 1, dtype=mx.float32)
    return mx.broadcast_to(cols[None, None, :, None], (rows, 1, n, 2))


def _expected(cols: np.ndarray, first: int, n: int, pads) -> np.ndarray:
    """(B, n, K) visibility for queries at columns first..first+n-1."""
    q = np.arange(first, first + n)[None, :, None]
    k = cols[:, None, :] - 1
    written = cols[:, None, :] > 0
    real = k >= np.asarray(pads)[:, None, None]
    return written & real & (k <= q) & (q - k < _W)


def _run(cls, widths, pads):
    """Mismatches between each update's mask and the expected one, over
    the query rows of real tokens, per update: [(width, rows, cells)]."""
    c = cls(max_size=_W, left_padding=list(pads))
    done, out = 0, []
    for n in widths:
        mask = c.make_mask(n)
        k = _keys(done, n, len(pads))
        keys, _ = c.update_and_fetch(k, k)
        mask = np.array(mx.broadcast_to(mask, (len(pads), 1, n, keys.shape[2])))[:, 0]
        want = _expected(np.array(keys[:, 0, :, 0]).astype(np.int64), done, n,
                         pads)
        bad = []
        for b, pad in enumerate(pads):
            rows = np.arange(done, done + n) >= pad
            diff = (mask[b] != want[b])[rows]
            if diff.any():
                bad.append((b, int(diff.sum())))
        out.append((n, bad))
        done += n
    return out


@pytest.mark.parametrize("cls", cache_types("BatchRotatingKVCache"),
                         ids=lambda c: c.__module__)
@pytest.mark.parametrize("widths,pads", [
    ([12, 1, 3], [0, 7]),      # a one-token update past the window, a block
    ([10, 1, 1, 4], [0, 6]),   # two rotating steps before the block
    ([8, 1, 3], [0, 3]),       # the window exactly full, then one token
    ([3, 1, 4], [0, 2]),       # the window never fills: stock is right
])
def test_every_mask_hides_exactly_the_padding_the_update_leaves(
        cls, widths, pads):
    assert all(bad == [] for _n, bad in _run(cls, widths, pads))


@pytest.mark.parametrize("cls", cache_types("BatchRotatingKVCache"),
                         ids=lambda c: c.__module__)
def test_decode_steps_then_a_verify_block(cls):
    """The final one-column prefill forward and two decode steps rotate the
    buffer, and a four-token verify block follows."""
    assert all(bad == [] for _n, bad in _run(cls, [11, 1, 1, 1, 4], [0, 8]))


@pytest.mark.parametrize("cls", cache_types("BatchRotatingKVCache"),
                         ids=lambda c: c.__module__)
def test_the_wrap_never_writes_the_cache(cls):
    writes = []

    class Spy(cls):
        def __setattr__(self, name, value):
            if getattr(self, "_watch", False):
                writes.append(name)
            super().__setattr__(name, value)

    c = Spy(max_size=_W, left_padding=[0, 7])
    c.update_and_fetch(_keys(0, 12, 2), _keys(0, 12, 2))
    c.update_and_fetch(_keys(12, 1, 2), _keys(12, 1, 2))
    assert c.rotated and c._idx == 1
    object.__setattr__(c, "_watch", True)
    c.make_mask(3)
    assert writes == [] and c.rotated and c._idx == 1


def test_install_is_idempotent():
    fns = [cls.make_mask for cls in cache_types("BatchRotatingKVCache")]
    install_batch_rotating_mask_fix()
    assert [cls.make_mask for cls in cache_types("BatchRotatingKVCache")] == fns
