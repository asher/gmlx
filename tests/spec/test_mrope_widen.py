"""Regression for the spec prompt-batch mrope width crash (2026-07-23 bench:
mtp@3 d4k c4 lost 16/16 requests on '[broadcast_shapes] Shapes (4) and (3)
cannot be broadcast'). Continuous-batch admission grew the prompt batch 3->4
while the target's cached text mrope deltas kept width 3; upstream qwen3_5
only slices deltas down, never widens. _widen_prompt_rope_state zero-pads
both delta sources (prompt kwargs + model cache) to the live width, which is
exact for text rows (delta 0). Decode-loop twin: speculative.py injection
guard (4e27fb2)."""

from types import SimpleNamespace

import mlx.core as mx

from gmlx.spec.engine import _widen_prompt_rope_state


def _batch(width, model):
    return SimpleNamespace(_input_ids=mx.zeros((width, 5), dtype=mx.int32),
                           model=model)


def _lm(deltas):
    m = SimpleNamespace()
    if deltas is not None:
        m._rope_deltas = deltas
    return m


def test_model_cache_widened_and_values_kept():
    lm = _lm(mx.array([[1.0], [2.0], [3.0]]))
    out = _widen_prompt_rope_state(_batch(4, lm), {})
    assert out == {}
    assert lm._rope_deltas.shape == (4, 1)
    assert lm._rope_deltas[:3].tolist() == [[1.0], [2.0], [3.0]]
    assert lm._rope_deltas[3].tolist() == [0.0]


def test_kwargs_widened_without_mutating_caller_dict():
    lm = _lm(None)
    kwargs = {"rope_deltas": mx.array([[5.0], [6.0], [7.0]])}
    out = _widen_prompt_rope_state(_batch(4, lm), kwargs)
    assert out is not kwargs
    assert kwargs["rope_deltas"].shape == (3, 1)
    assert out["rope_deltas"].shape == (4, 1)
    assert out["rope_deltas"].tolist() == [[5.0], [6.0], [7.0], [0.0]]


def test_equal_width_is_a_noop():
    rd = mx.array([[1.0], [2.0], [3.0]])
    lm = _lm(rd)
    kwargs = {"rope_deltas": rd}
    out = _widen_prompt_rope_state(_batch(3, lm), kwargs)
    assert out is kwargs
    assert lm._rope_deltas is rd


def test_wider_cache_left_for_upstream_slice_down():
    lm = _lm(mx.zeros((5, 1)))
    _widen_prompt_rope_state(_batch(3, lm), {})
    assert lm._rope_deltas.shape == (5, 1)


def test_model_without_cached_deltas_untouched():
    lm = _lm(None)
    out = _widen_prompt_rope_state(_batch(4, lm), {})
    assert out == {}
    assert not hasattr(lm, "_rope_deltas")


def test_language_model_nesting_resolved():
    inner = _lm(mx.zeros((2, 1)))
    outer = SimpleNamespace(language_model=inner)
    _widen_prompt_rope_state(_batch(4, outer), {})
    assert inner._rope_deltas.shape == (4, 1)


def test_int_deltas_keep_dtype():
    lm = _lm(mx.array([[3], [4]], dtype=mx.int64))
    _widen_prompt_rope_state(_batch(3, lm), {})
    assert lm._rope_deltas.dtype == mx.int64
    assert lm._rope_deltas.tolist() == [[3], [4], [0]]
