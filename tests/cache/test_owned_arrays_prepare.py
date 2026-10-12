"""A ragged right-padded prefill masks ArraysCache state.

Both batch engines right-pad ragged prefill rows and declare the layout
with ``prepare(lengths=...)``: mlx-lm's BatchGenerator on every ragged
insert, and mlx-vlm's PromptProcessingBatch on mixed warm and cold APC
prefill. By then a merge of fresh caches, or the engine's ``_make_cache``,
has seeded ``left_padding`` with zeros, and ``make_mask`` prefers
``left_padding`` over ``lengths``, so the stock mask is all True and pad
keys enter the conv window and recurrent state of every short row. The
owned ``prepare`` clears ``left_padding`` when it takes lengths. Each case
also runs the stock prepare body, which must fail the right-padded ones.
CPU.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from gmlx.models.qwen35.cache import Qwen35ArraysCache
from stock_twin import stock_method


@pytest.fixture(params=["owned", "stock"])
def body(request, monkeypatch):
    if request.param == "stock":
        monkeypatch.setattr(owned.ArraysCache, "prepare",
                            stock_method(owned.ArraysCache, "prepare"))
    return request.param


_CLASSES = pytest.mark.parametrize(
    "cls", [owned.ArraysCache, Qwen35ArraysCache], ids=lambda c: c.__name__)


@_CLASSES
def test_prepare_lengths_overrides_stale_left_padding(cls, body):
    # BatchGenerator flow: merge of fresh caches seeds left_padding zeros.
    merged = cls.merge([cls(1), cls(1)])
    assert merged.left_padding is not None
    merged.prepare(lengths=[2, 5], right_padding=[3, 0])
    mask = merged.make_mask(5)
    expect = mx.array([[True, True, False, False, False],
                       [True, True, True, True, True]])
    assert mx.array_equal(mask, expect).item() is (body == "owned")


@_CLASSES
def test_prepare_after_engine_seeded_left_padding(cls, body):
    # mlx-vlm mixed warm/cold flow: _make_cache assigns left_padding zeros
    # directly before prepare(right_padding=..., lengths=...) runs.
    c = cls(1)
    c.left_padding = mx.array([0, 0])
    c.prepare(right_padding=[0, 4], lengths=[6, 2])
    mask = c.make_mask(6)
    assert mx.array_equal(mask[0], mx.ones((6,), dtype=mx.bool_)).item()
    ok = mx.array_equal(mask[1], mx.array([True, True, False, False, False,
                                           False])).item()
    assert ok is (body == "owned")


@_CLASSES
def test_chunked_advance_keeps_lengths_mask(cls, body):
    c = cls(1)
    c.left_padding = mx.array([0, 0])
    c.prepare(lengths=[3, 6], right_padding=[3, 0])
    c.advance(4)
    mask = c.make_mask(2)
    # Row 0 exhausted after the first 4-token chunk; row 1 has 2 left.
    ok = mx.array_equal(mask, mx.array([[False, False], [True, True]])).item()
    assert ok is (body == "owned")
    c.finalize()
    assert c.make_mask(1) is None


@_CLASSES
def test_left_padded_layout_untouched(cls, body):
    # The serve engine's plain ragged path left-pads and never calls
    # prepare; its mask source is left_padding on both bodies.
    c = cls(1)
    c.left_padding = mx.array([2, 0])
    mask = c.make_mask(3)
    assert mx.array_equal(mask, mx.array([[False, False, True],
                                          [True, True, True]])).item()
