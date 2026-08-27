#!/usr/bin/env python3
"""The no-op mask guard on the fused gated-delta decode path.

``BatchGenerator`` hands every batched decode step an ssm mask derived from the
cache's left padding. At S=1 that mask admits every row -- left padding is
history, already folded into the recurrent state, not part of this step's input
-- so it excludes nothing, yet the old ``mask is not None`` bail dropped all
gated-delta layers onto the unfused chain whenever it was present.

``_mask_excludes_nothing`` is what makes skipping it safe, so it has to be exact
in both directions: admit genuinely empty masks, and refuse anything that
excludes even one entry (a fully padded row must keep the stock path, which
honors the mask).
"""

from __future__ import annotations

import mlx.core as mx

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
