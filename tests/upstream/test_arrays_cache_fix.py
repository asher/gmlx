"""arrays_cache_fix: ragged right-padded prefill must mask ArraysCache state.

Upstream flow (mlx-lm BatchGenerator on every ragged insert; mlx-vlm
PromptProcessingBatch on mixed warm/cold APC prefill): the batch cache is
seeded with ``left_padding = [0] * B`` (ArraysCache.merge all-empty branch /
the engine's _make_cache), rows are right-padded, and the layout is declared
via ``prepare(lengths=..., right_padding=...)``. Stock ``prepare`` leaves the
stale left_padding in place and ``make_mask`` prefers it over ``lengths``,
so the SSM/conv mask is all-True and pad garbage enters recurrent state
(falcon-h1: shortest row confidently wrong at its first decode token). The
fix makes a lengths-bearing ``prepare`` clear ``left_padding``.

CPU-only; exercises both class origins the way the engines drive them.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.upstream.arrays_cache_fix import install_arrays_cache_fix
from gmlx.cache.compat import cache_types


@pytest.fixture(autouse=True)
def installed():
    install_arrays_cache_fix()


def _origins():
    return cache_types("ArraysCache")


@pytest.mark.parametrize("cls", _origins(), ids=lambda c: c.__module__)
def test_prepare_lengths_overrides_stale_left_padding(cls):
    # BatchGenerator flow: merge of fresh caches seeds left_padding zeros.
    merged = cls.merge([cls(1), cls(1)])
    assert merged.left_padding is not None
    merged.prepare(lengths=[2, 5], right_padding=[3, 0])
    mask = merged.make_mask(5)
    assert mask is not None
    expect = mx.array([[True, True, False, False, False],
                       [True, True, True, True, True]])
    assert mx.array_equal(mask, expect)


@pytest.mark.parametrize("cls", _origins(), ids=lambda c: c.__module__)
def test_prepare_after_engine_seeded_left_padding(cls):
    # mlx-vlm mixed warm/cold flow: _make_cache assigns left_padding zeros
    # directly before prepare(right_padding=..., lengths=...) runs.
    c = cls(1)
    c.left_padding = mx.array([0, 0])
    c.prepare(right_padding=[0, 4], lengths=[6, 2])
    mask = c.make_mask(6)
    assert mx.array_equal(mask[0], mx.ones((6,), dtype=mx.bool_))
    assert mx.array_equal(mask[1],
                          mx.array([True, True, False, False, False, False]))


@pytest.mark.parametrize("cls", _origins(), ids=lambda c: c.__module__)
def test_chunked_advance_keeps_lengths_mask(cls):
    c = cls(1)
    c.left_padding = mx.array([0, 0])
    c.prepare(lengths=[3, 6], right_padding=[3, 0])
    c.advance(4)
    mask = c.make_mask(2)
    # Row 0 exhausted after the first 4-token chunk; row 1 has 2 left.
    assert mx.array_equal(mask, mx.array([[False, False], [True, True]]))
    c.finalize()
    assert c.make_mask(1) is None


@pytest.mark.parametrize("cls", _origins(), ids=lambda c: c.__module__)
def test_left_padded_layout_untouched(cls):
    # The serve engine's plain ragged path left-pads and never calls
    # prepare; its mask source must survive the patch.
    c = cls(1)
    c.left_padding = mx.array([2, 0])
    mask = c.make_mask(3)
    assert mx.array_equal(mask, mx.array([[False, False, True],
                                          [True, True, True]]))


def test_install_idempotent():
    install_arrays_cache_fix()
    before = [cls.prepare for cls in _origins()]
    install_arrays_cache_fix()
    assert [cls.prepare for cls in _origins()] == before
