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


def test_checkpointed_layers_replay_the_dropout_mask(monkeypatch):
    """A dropout inside a checkpointed layer draws from the global stream,
    so the backward recompute would see a fresh mask; the layer's seed is
    drawn once and replayed, and the gradient equals a plain backward whose
    layers draw the same masks."""
    import gmlx.tune.checkpoint as ck
    from mlx_lm.tuner.lora import LoRALinear

    def build():
        model = _model(3)
        for layer in model.layers:
            q = LoRALinear.from_base(layer.self_attn.q_proj, r=4, dropout=0.5)
            q.lora_b = mx.random.normal(q.lora_b.shape)
            layer.self_attn.q_proj = q
        model.train()
        return model

    ids = mx.array([[1, 5, 9, 2, 7, 3], [4, 4, 8, 1, 2, 6]])
    ref = build()
    cls = type(ref.layers[0])
    orig = cls.__call__

    def seeded(self, *args, **kwargs):
        mx.random.seed(7)
        return orig(self, *args, **kwargs)

    try:
        cls.__call__ = seeded
        g0 = nn.value_and_grad(ref, _loss)(ref, ids)[1]
        mx.eval(g0)
        cls.__call__ = orig
        model = build()
        monkeypatch.setattr(ck, "layer_seed", lambda: 7)
        assert checkpoint_layers(model, replay_dropout=True) == 1
        g1 = nn.value_and_grad(model, _loss)(model, ids)[1]
        mx.eval(g1)
    finally:
        cls.__call__ = orig
    from mlx.utils import tree_flatten
    a = dict(tree_flatten(g0))
    b = dict(tree_flatten(g1))
    assert a.keys() == b.keys() and any("lora_b" in k for k in a)
    for k in a:
        assert np.allclose(np.array(a[k]), np.array(b[k]), atol=1e-5), k
    assert not ck.draws_random(ref.eval()) and ck.draws_random(ref.train())


def test_checkpointed_layers_run_under_a_compiled_step():
    """Without replay_dropout a checkpointed layer draws nothing eagerly,
    so a compiled training step (mlx-lm's train loop) runs through it even
    when the layer holds dropout."""
    from functools import partial

    from mlx_lm.tuner.lora import LoRALinear

    model = _model(5)
    for layer in model.layers:
        q = LoRALinear.from_base(layer.self_attn.q_proj, r=4, dropout=0.5)
        q.lora_b = mx.random.normal(q.lora_b.shape)
        layer.self_attn.q_proj = q
    model.train()
    ids = mx.array([[1, 5, 9, 2, 7, 3], [4, 4, 8, 1, 2, 6]])
    cls = type(model.layers[0])
    orig = cls.__call__
    try:
        assert checkpoint_layers(model) == 1
        lvg = nn.value_and_grad(model, _loss)
        state = [model.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(batch):
            loss, grads = lvg(model, batch)
            return loss, grads
        loss, grads = step(ids)
        mx.eval(loss, grads)
    finally:
        cls.__call__ = orig
    assert np.isfinite(float(loss))


def test_checkpoint_layers_takes_the_latest_replay_setting(monkeypatch):
    """A rewritten class follows the replay setting of the latest call,
    so a trainer that wants replay after another installed without it
    gets it."""
    import gmlx.tune.checkpoint as ck

    model = _model()
    for layer in model.layers:
        layer.self_attn.q_proj = nn.Sequential(layer.self_attn.q_proj, nn.Dropout(0.5))
    model.train()
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    cls = type(model.layers[0])
    orig = cls.__call__
    draws = []
    monkeypatch.setattr(ck, "layer_seed", lambda: draws.append(1) or 7)
    try:
        assert checkpoint_layers(model) == 1
        mx.eval(model(ids))
        assert not draws
        assert checkpoint_layers(model, replay_dropout=True) == 0
        mx.eval(model(ids))
        assert len(draws) == 2
        assert checkpoint_layers(model, replay_dropout=False) == 0
        mx.eval(model(ids))
        assert len(draws) == 2
    finally:
        cls.__call__ = orig
