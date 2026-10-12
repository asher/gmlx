"""Qwen35ArraysCache: the vendored qwen3.5 model's recurrent cache keeps
0.6.15 metadata semantics (advance rebinds left_padding and lengths) on
the mlx-vlm 0.7.3 ArraysCache base. CPU, no model."""

from types import SimpleNamespace

import mlx.core as mx
from gmlx._vendor.mlx_vlm import apc_adapters
from gmlx._vendor.mlx_vlm.models.cache import ArraysCache

from gmlx.models.qwen35.cache import Qwen35ArraysCache
from gmlx.models.qwen35.language import (
    LanguageModel,
    _qwen3_5_left_padding_info,
    _qwen3_5_lengths_info,
)


def _cache(lp=(3, 0), lengths=None):
    c = Qwen35ArraysCache(size=2, left_padding=list(lp))
    if lengths is not None:
        c.lengths = mx.array(lengths)
    c[0] = mx.zeros((len(lp), 3, 8))
    c[1] = mx.ones((len(lp), 2, 4, 4))
    return c


def test_advance_rebinds_without_mutating_the_old_arrays():
    c = _cache(lengths=(5, 2))
    lp, ln = c.left_padding, c.lengths
    c.advance(2)
    assert c.left_padding is not lp and c.lengths is not ln
    assert c.left_padding.tolist() == [1, -2]
    assert c.lengths.tolist() == [3, 0]
    assert lp.tolist() == [3, 0] and ln.tolist() == [5, 2]
    assert c._left_padding_advance == 0 and c._lengths_advance == 0
    # Reads between advances return one object, so identity memos hit.
    assert c.left_padding is c.left_padding
    assert c.make_mask(3).tolist() == [[False, True, True], [True, True, True]]


def test_memo_helpers_track_every_advance():
    c = _cache(lengths=(5, 2))
    assert _qwen3_5_left_padding_info(c) == ((3, 0), 3)
    assert _qwen3_5_lengths_info(c) == 2
    c.advance(1)
    assert _qwen3_5_left_padding_info(c) == ((2, -1), 2)
    assert _qwen3_5_lengths_info(c) == 1


def test_make_cache_builds_the_gmlx_class():
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    caches = LanguageModel.make_cache(SimpleNamespace(layers=layers))
    assert type(caches[0]) is Qwen35ArraysCache
    assert type(caches[1]).__name__ == "KVCache"


def test_abort_restores_the_saved_metadata():
    c = _cache(lengths=(5, 2))
    gen = c.start_speculation(2)
    c.advance(2)
    c.abort_speculation(gen)
    assert c.left_padding.tolist() == [3, 0]
    assert c.lengths.tolist() == [5, 2]


def test_commit_settles_the_advance_offset():
    c = _cache()
    gen = c.start_speculation(2)
    c.update_recurrent(1, 2, lambda s, steps: (None, s + 1, (s + 0.5)[:, None]))
    c.update_window(0, mx.zeros((2, 5, 8)), 3)
    c.advance(2)
    c.commit_speculation([1, 1], gen)
    assert c._left_padding_advance == 0
    assert c.left_padding.tolist() == [2, -1]


def test_apc_contract_carries_over_from_the_base():
    c = _cache()
    assert apc_adapters.resolve_capability(c) is apc_adapters.Capability.CHECKPOINT
    assert apc_adapters.apc_exact_eligible(c)
    targets = []
    clone = apc_adapters.clone_cache_entry(
        c, min_capacity_tokens=0, eval_targets=targets)
    assert type(clone) is Qwen35ArraysCache
    assert mx.array_equal(clone[1], c[1])


def test_extract_and_merge_keep_the_class():
    c = _cache()
    row = c.extract(1)
    assert type(row) is Qwen35ArraysCache
    merged = Qwen35ArraysCache.merge([row, c.extract(0)])
    assert type(merged) is Qwen35ArraysCache
    assert isinstance(merged, ArraysCache)
