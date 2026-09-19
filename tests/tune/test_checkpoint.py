"""checkpoint_layers: gradients equal the plain backward, the rewrite is
idempotent and finds layers under a language-model wrapper."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_lm.models import llama
from mlx_lm.models.llama import ModelArgs

from gmlx.tune.checkpoint import checkpoint_layers



@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _model(seed=0):
    mx.random.seed(seed)
    args = ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=4,
                     num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=64)
    return llama.Model(args)


def _loss(model, ids):
    logits = model(ids[:, :-1])
    return nn.losses.cross_entropy(logits, ids[:, 1:]).mean()


def test_gradients_match_and_rewrite_is_idempotent():
    ids = mx.array([[1, 5, 9, 2, 7, 3], [4, 4, 8, 1, 2, 6]])
    model = _model()
    g0 = nn.value_and_grad(model, _loss)(model, ids)[1]
    mx.eval(g0)
    cls = type(model.layers[0])
    orig = cls.__call__
    try:
        assert checkpoint_layers(model) == 1
        assert checkpoint_layers(model) == 0
        model.train()
        g1 = nn.value_and_grad(model, _loss)(model, ids)[1]
        mx.eval(g1)
    finally:
        cls.__call__ = orig
    from mlx.utils import tree_flatten
    a = dict(tree_flatten(g0))
    b = dict(tree_flatten(g1))
    assert a.keys() == b.keys()
    for k in a:
        assert np.allclose(np.array(a[k]), np.array(b[k]), atol=1e-5), k


def test_finds_layers_under_a_language_model_wrapper():
    class Wrapper(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.language_model = lm

    model = _model()
    cls = type(model.layers[0])
    orig = cls.__call__
    try:
        assert checkpoint_layers(Wrapper(model)) == 1
    finally:
        cls.__call__ = orig
