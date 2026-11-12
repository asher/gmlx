#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The d1 decision forward against references outside its module.

A tiny random lfm3 model with the real window (256), 7 conv taps and the
d1 layer pattern cut to 8 layers, in float32 on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

LAYERS = ["conv", "conv", "full_attention", "conv", "sliding_attention",
          "conv", "full_attention", "conv"]


@pytest.fixture
def mx():
    mx = pytest.importorskip("mlx.core")
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(prev)


def _model(mx, layer_types=LAYERS, decision=False):
    from gmlx.models.lfm3.model import Model, ModelArgs

    args = ModelArgs(model_type="lfm3", hidden_size=32,
                     num_hidden_layers=len(layer_types), intermediate_size=64,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     vocab_size=64, layer_types=list(layer_types),
                     sliding_window=256, decision=decision)
    mx.random.seed(0)
    model = Model(args)
    for layer in model.layers:
        if layer.layer_type == "conv":
            # The taps start at zero, which would make every conv a no-op.
            layer.conv.kernel = 0.3 * mx.random.normal(layer.conv.kernel.shape)
    mx.eval(model.parameters())
    return model


def _ids(mx, n, seed=1):
    return mx.array(np.random.default_rng(seed).integers(0, 64, (1, n)))


def _forward(model, ids, lengths=None, **kwargs):
    from gmlx.models.lfm3.decision import run_decision_forward

    lengths = lengths or [ids.shape[1]] * ids.shape[0]
    return np.asarray(run_decision_forward(model, ids, lengths, **kwargs))


def _close(got, want, rel):
    scale = float(np.abs(want).max())
    assert float(np.abs(got - want).max()) <= rel * scale, (
        float(np.abs(got - want).max()), scale)


def _reference(mx, model, ids, causal=False):
    """Each layer once over the whole sequence, with no cache: a full layer
    is bidirectional (offset 0, its own keys, no mask) or causal; a sliding
    layer takes the mask ``Lfm3Model.__call__`` builds."""
    from gmlx.cache.attention import create_attention_mask

    trunk = model.model
    h = trunk.embed_tokens(ids)
    for layer in trunk.layers:
        if layer.layer_type == "full_attention":
            h = layer(h, "causal" if causal else None, None)
        elif layer.layer_type == "sliding_attention":
            h = layer(h, create_attention_mask(
                h, None, window_size=trunk.sliding_window), None)
        else:
            h = layer(h, None, None)
    out = model.head(trunk.norm(h[:, -1].astype(mx.float32)), suppress=False)
    return np.asarray(out)


def test_the_forward_equals_a_bidirectional_reference(mx):
    model = _model(mx, decision=True)
    ids = _ids(mx, 700)
    with pytest.raises(NotImplementedError):
        model(ids)                      # the causal path stays refused
    got = _forward(model, ids, block=64)
    assert got.dtype == np.float32
    _close(got, _reference(mx, model, ids), 1e-3)


def test_padded_rows_each_equal_the_reference_on_that_row_alone(mx):
    model = _model(mx)
    lengths = [700, 523, 301]
    rows = [_ids(mx, n, seed=n) for n in lengths]
    padded = mx.concatenate([
        mx.pad(r, [(0, 0), (0, 700 - r.shape[1])]) for r in rows])
    got = _forward(model, padded, lengths, block=64)
    for i, r in enumerate(rows):
        _close(got[i:i + 1], _reference(mx, model, r), 1e-3)
        _close(got[i:i + 1], _forward(model, r, block=64), 1e-4)


def test_causal_mode_equals_the_chat_forward(mx):
    model = _model(mx)
    ids = _ids(mx, 700)
    chat = np.asarray(model(ids)[:, -1].astype(mx.float32))
    got = _forward(model, ids, block=64, causal=True)
    _close(got, chat, 1e-3)
    _close(got, _reference(mx, model, ids, causal=True), 1e-3)


def test_the_block_size_does_not_change_the_answer(mx):
    model = _model(mx)
    ids = _ids(mx, 3000)
    _close(_forward(model, ids, block=64), _forward(model, ids, block=3000),
           1e-4)


def test_only_the_full_layers_read_later_tokens(mx):
    ids = _ids(mx, 300)
    changed = mx.concatenate([ids[:, :-1], (ids[:, -1:] + 1) % 64], axis=1)
    with_full = _model(mx)
    a = _forward(with_full, ids, positions=[0], block=64)
    b = _forward(with_full, changed, positions=[0], block=64)
    assert np.abs(a - b).max() > 1e-4
    causal_only = _model(mx, ["conv", "sliding_attention", "conv"])
    a = _forward(causal_only, ids, positions=[0], block=64)
    b = _forward(causal_only, changed, positions=[0], block=64)
    np.testing.assert_array_equal(a, b)


def test_lengths_and_positions_are_checked(mx):
    model = _model(mx)
    ids = _ids(mx, 10)
    with pytest.raises(ValueError):
        _forward(model, ids, [11])
    with pytest.raises(ValueError):
        _forward(model, ids, [5], positions=[5])
