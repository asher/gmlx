#!/usr/bin/env python3
"""The no-op mask guard on the fused gated-delta decode path.

``BatchGenerator`` hands every batched decode step an ssm mask derived from the
cache's left padding. At S=1 that mask admits every row whose padding is
behind (left padding is history, already folded into the recurrent state, not
part of this step's input), so it usually excludes nothing, yet the old
``mask is not None`` bail dropped all gated-delta layers onto the unfused chain
whenever it was present.

``_mask_excludes_nothing`` is what makes skipping it safe, and it must not
read the mask back on a decode step. The cache answers from a host-side
bound, which holds in one direction without exception: a mask that excludes
an entry is never admitted, so a row whose padding is still ahead keeps the
stock path, which honors the mask. The other direction is loose by design:
after a ragged speculative commit the bound can trail the true padding by up
to the spread of the block's accepts, adding up across ragged commits, and
keep the stock path that much longer.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.upstream.gdn_patches as patches
from gmlx._vendor.mlx_lm.models.qwen3_5 import GatedDeltaNet
from gmlx.cache.kv_cache import ArraysCache
from gmlx.models.qwen35.cache import Qwen35ArraysCache
from gmlx.upstream.gdn_patches import _mask_excludes_nothing


def _reset_memo():
    import gmlx.upstream.gdn_patches as g

    g._noop_mask_memo = (None, False)


def test_all_true_bool_mask_excludes_nothing():
    _reset_memo()
    assert _mask_excludes_nothing(mx.ones((3, 1), dtype=mx.bool_))


def test_single_false_entry_is_refused():
    _reset_memo()
    m = mx.array([[True], [False], [True]])
    assert not _mask_excludes_nothing(m)


def test_all_false_mask_is_refused():
    _reset_memo()
    assert not _mask_excludes_nothing(mx.zeros((2, 1), dtype=mx.bool_))


def test_additive_mask_with_negative_entries_is_refused():
    """Float/additive masks carry -inf (or large negatives) for excluded
    positions; only an all-zero additive mask excludes nothing."""
    _reset_memo()
    blocked = mx.array([[0.0], [-float("inf")]])
    assert not _mask_excludes_nothing(blocked)


def test_additive_all_zero_mask_excludes_nothing():
    _reset_memo()
    assert _mask_excludes_nothing(mx.zeros((2, 1)))


def test_memo_is_keyed_on_identity_not_value():
    """The memo exists so one host sync covers every layer in a step. A
    different array with a different verdict must not inherit the cached one."""
    _reset_memo()
    empty = mx.ones((2, 1), dtype=mx.bool_)
    blocking = mx.array([[True], [False]])
    assert _mask_excludes_nothing(empty)
    assert not _mask_excludes_nothing(blocking)
    assert _mask_excludes_nothing(empty)


def test_repeated_calls_on_same_object_are_stable():
    _reset_memo()
    m = mx.array([[True], [False]])
    assert not _mask_excludes_nothing(m)
    assert not _mask_excludes_nothing(m)


_SYNCS = ("item", "tolist", "__int__", "__bool__", "__float__")


def _forbid_host_sync(monkeypatch):
    """Every blocking read of an array raises, so a decode step that waits
    for the GPU fails the test instead of slowing it."""

    def boom(*args, **kwargs):
        raise AssertionError("host sync inside the GDN decode route")

    for name in _SYNCS:
        monkeypatch.setattr(mx.array, name, boom)
    monkeypatch.setattr(mx, "eval", boom)


def _decode_route(monkeypatch):
    """A bare GatedDeltaNet that `_gdn_fused_decode_call` routes, with the
    fused body and the stock call replaced by recorders, so the route
    itself is under test on any device."""
    taken = []
    monkeypatch.setattr(patches, "_gdn_fused_decode_kernel", object())
    monkeypatch.setattr(
        patches, "_gdn_fused_decode_body",
        lambda self, inputs, cache: taken.append("fused"))
    monkeypatch.setattr(
        patches._FUSED_DECODE_PATCH, "stock",
        lambda self, inputs, mask, cache: taken.append("stock"))
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn._gdn_fused = True
    gdn.head_v_dim = 128
    gdn.head_k_dim = 128
    return gdn, taken


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_decode_mask_check_without_host_sync(monkeypatch, cls):
    """A batch cache (any batch under --kv-bits, or prompts of different
    lengths served together) hands every decode step a fresh all-true mask.
    Deciding that it excludes nothing must not read it back: the read waits
    behind the step in flight, which cost a 30-layer hybrid a quarter of its
    decode rate at kv-bits 8."""
    _reset_memo()
    gdn, taken = _decode_route(monkeypatch)
    cache = cls(size=2, left_padding=[2, 0])
    cache[1] = mx.zeros((2, 4))
    cache.advance(3)  # the padded prefill: no row's padding is still ahead
    x = mx.zeros((2, 1, 8))
    patches._gdn_fused_decode_call(gdn, x, cache.make_mask(1), cache)
    assert taken == ["fused"]
    _forbid_host_sync(monkeypatch)
    for _ in range(3):
        cache.advance(1)
        patches._gdn_fused_decode_call(gdn, x, cache.make_mask(1), cache)
    assert taken == ["fused"] * 4


def test_fully_padded_row_keeps_the_stock_path(monkeypatch):
    """A row whose padding is still ahead of this step contributes no real
    token, so its mask entry is False and the fused kernel, which ignores
    the mask, must not run. Once every row's padding is behind, the fused
    path resumes from the advanced bound alone."""
    _reset_memo()
    gdn, taken = _decode_route(monkeypatch)
    cache = ArraysCache(size=2, left_padding=[1, 0])
    cache[1] = mx.zeros((2, 4))
    x = mx.zeros((2, 1, 8))
    mask = cache.make_mask(1)
    assert mask.tolist() == [[False], [True]]
    patches._gdn_fused_decode_call(gdn, x, mask, cache)
    assert taken == ["stock"]
    _forbid_host_sync(monkeypatch)
    cache.advance(1)
    patches._gdn_fused_decode_call(gdn, x, cache.make_mask(1), cache)
    assert taken == ["stock", "fused"]


def test_cache_bounds_follow_the_batch():
    """The host bound is read again only when the padding array is rebound:
    a row joining or leaving, or a lengths layout replacing it."""
    cache = ArraysCache(size=2, left_padding=[2, 0])
    cache.advance(3)
    assert cache.mask_excludes_nothing(1)
    other = ArraysCache(size=2, left_padding=[4])
    cache.extend(other)  # the new row's padding is still ahead
    assert not cache.mask_excludes_nothing(1)
    cache.filter(mx.array([0, 1]))
    assert cache.mask_excludes_nothing(1)
    cache.prepare(lengths=[3, 1])
    assert cache.mask_excludes_nothing(1)
    assert not cache.mask_excludes_nothing(2)
    cache.advance(1)
    assert not cache.mask_excludes_nothing(1)
    cache.finalize()
    assert cache.mask_excludes_nothing(1)


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_ragged_speculative_commit_keeps_the_bound(monkeypatch, cls):
    """A drafter block that rows accept unevenly rebinds the padding array
    on commit. The bound follows it as a bound, so the next block's first
    step reads nothing, and a row whose padding is still ahead keeps the
    stock path."""
    cache = cls(size=2, left_padding=[2, 0])
    cache.advance(3)
    assert cache.mask_excludes_nothing(1)
    cache.start_speculation(3)
    cache.advance(3)  # the block's verify step
    _forbid_host_sync(monkeypatch)
    cache.commit_speculation([1, 3])
    assert cache.mask_excludes_nothing(1)
    monkeypatch.undo()
    ahead = cls(size=2, left_padding=[5, 0])
    ahead.advance(3)
    assert not ahead.mask_excludes_nothing(1)
    ahead.start_speculation(3)
    ahead.advance(3)
    _forbid_host_sync(monkeypatch)
    ahead.commit_speculation([1, 3])  # row 0 still has one pad ahead
    assert not ahead.mask_excludes_nothing(1)
    monkeypatch.undo()
    assert ahead.make_mask(1).tolist() == [[False], [True]]


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_even_commit_and_abort_keep_the_bound(monkeypatch, cls):
    """An even accept moves the block's count onto the padding the block
    started from, an abort restores that padding as it was. Either way the
    bound stays with it and the padded row keeps the stock path until its
    padding is behind."""
    cache = cls(size=2, left_padding=[3, 0])
    cache.advance(1)
    assert not cache.mask_excludes_nothing(1)
    cache.start_speculation(2)
    cache.advance(2)
    _forbid_host_sync(monkeypatch)
    cache.abort_speculation()  # row 0 has two pads ahead again
    assert not cache.mask_excludes_nothing(1)
    cache.start_speculation(2)
    cache.advance(2)
    cache.commit_speculation([1, 1])  # one pad ahead
    assert not cache.mask_excludes_nothing(1)
    cache.advance(1)
    assert cache.mask_excludes_nothing(1)
    monkeypatch.undo()
    assert cache.make_mask(1).tolist() == [[True], [True]]


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_a_bound_first_read_inside_a_block_carries(monkeypatch, cls):
    """With a drafter every decode step runs inside a block, so the first
    read happens there, before the block's advance. The restore still
    carries it, and the next block reads nothing."""
    cache = cls(size=2, left_padding=[2, 0])
    cache.advance(3)
    cache.start_speculation(3)
    assert cache.mask_excludes_nothing(1)  # the only read
    cache.advance(3)
    _forbid_host_sync(monkeypatch)
    cache.commit_speculation([1, 3])
    cache.start_speculation(3)
    assert cache.mask_excludes_nothing(1)
    cache.advance(3)
    cache.commit_speculation([2, 2])
    assert cache.mask_excludes_nothing(1)


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_a_bound_first_read_after_an_advance_carries(monkeypatch, cls):
    """A long block runs several forwards, and its first one-token step,
    the first to ask, can come after an advance. The end of the block
    still reaches the saved array from there, on an abort as on a
    commit."""
    cache = cls(size=2, left_padding=[2, 0])
    cache.advance(3)
    cache.start_speculation(3)
    cache.advance(2)  # a wider forward first
    assert cache.mask_excludes_nothing(1)  # the only read
    cache.advance(1)
    _forbid_host_sync(monkeypatch)
    cache.abort_speculation()
    assert cache.mask_excludes_nothing(1)
    cache.start_speculation(3)
    cache.advance(3)
    cache.commit_speculation([1, 3])
    assert cache.mask_excludes_nothing(1)


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_settles_within_a_block_add_up(monkeypatch, cls):
    """A block with two forwards, the first ask between them, moves the
    array twice where the cache settles its advance. The end of the block
    reaches the saved array by the sum of those moves, so a row with
    padding still ahead afterwards keeps the stock path."""
    cache = cls(size=2, left_padding=[5, 0])
    cache.advance(1)
    cache.start_speculation(3)
    cache.advance(1)
    assert not cache.mask_excludes_nothing(1)  # the only read
    _forbid_host_sync(monkeypatch)
    cache.advance(2)
    cache.abort_speculation()  # row 0 has four pads ahead again
    cache.advance(3)
    assert not cache.mask_excludes_nothing(1)  # one still ahead
    monkeypatch.undo()
    assert cache.make_mask(1).tolist() == [[False], [True]]


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_a_rebound_array_is_read_before_it_is_trusted(cls):
    """A row joining the batch rebinds the padding array, a new lengths
    layout the lengths. The bound kept for the old array must not move
    onto the new one with the next advance: the first ask after that
    reads the new array."""
    cache = cls(size=2, left_padding=[2, 0])
    cache.advance(3)
    assert cache.mask_excludes_nothing(1)
    cache.extend(cls(size=2, left_padding=[4]))
    cache.advance(1)
    assert not cache.mask_excludes_nothing(1)
    assert cache.make_mask(1).tolist() == [[True], [True], [False]]
    cache.prepare(lengths=[4, 3, 2])
    assert cache.mask_excludes_nothing(2)
    cache.prepare(lengths=[1, 1, 1])
    cache.advance(1)
    assert not cache.mask_excludes_nothing(1)
    assert cache.make_mask(1).tolist() == [[False], [False], [False]]


def test_a_loose_bound_only_delays_the_fused_path(monkeypatch):
    """After a ragged commit the bound can trail the true padding: it
    keeps the stock path while the mask already admits every row, and
    catches up as the batch advances. It never admits a padded row."""
    cache = ArraysCache(size=2, left_padding=[5, 0])
    cache.advance(3)
    assert not cache.mask_excludes_nothing(1)
    cache.start_speculation(3)
    cache.advance(3)
    _forbid_host_sync(monkeypatch)
    cache.commit_speculation([3, 1])  # row 0 accepted past its padding
    early = cache.make_mask(1)
    assert not cache.mask_excludes_nothing(1)  # the bound, 1, trails -1
    cache.advance(1)
    assert cache.mask_excludes_nothing(1)
    monkeypatch.undo()
    assert early.tolist() == [[True], [True]]


@pytest.mark.parametrize("cls", [ArraysCache, Qwen35ArraysCache])
def test_ragged_commit_carries_the_lengths_bound(monkeypatch, cls):
    """A lengths layout carries its smallest length the same way: a bound
    that admits a one-token step without a read and keeps the stock path
    for a wider one it cannot vouch for."""
    cache = cls(size=2)
    cache.prepare(lengths=[4, 3])
    assert cache.mask_excludes_nothing(2)
    cache.start_speculation(2)
    cache.advance(2)
    _forbid_host_sync(monkeypatch)
    cache.commit_speculation([2, 1])  # two tokens left in each row
    assert cache.mask_excludes_nothing(1)
    assert not cache.mask_excludes_nothing(2)  # the bound, 1, trails 2
    monkeypatch.undo()
    assert cache.make_mask(2).tolist() == [[True, True], [True, True]]

