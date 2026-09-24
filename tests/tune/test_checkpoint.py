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


def test_checkpoint_layers_marks_the_instances_not_the_class(monkeypatch):
    """Checkpointing asked for one model leaves a later model of the same
    layer class on its original forward: no seed draw, so a compiled
    step with dropout runs, and no checkpoint at all."""
    from functools import partial

    import gmlx.tune.checkpoint as ck

    a = _model()
    b = _model(1)
    for layer in b.layers:
        layer.self_attn.q_proj = nn.Sequential(layer.self_attn.q_proj, nn.Dropout(0.5))
    b.train()
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    cls = type(a.layers[0])
    orig = cls.__call__
    draws = []
    monkeypatch.setattr(ck, "layer_seed", lambda: draws.append(1) or 7)
    try:
        assert checkpoint_layers(a, replay_dropout=True) == 1
        assert all(getattr(layer, "_gmlx_ckpt", False) for layer in a.layers)
        assert not any(getattr(layer, "_gmlx_ckpt", False) for layer in b.layers)
        lvg = nn.value_and_grad(b, _loss)
        state = [b.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(batch):
            return lvg(b, batch)
        loss, grads = step(ids)
        mx.eval(loss, grads)
        assert not draws
        for layer in a.layers:
            layer.self_attn.q_proj = nn.Sequential(layer.self_attn.q_proj, nn.Dropout(0.5))
        a.train()
        mx.eval(a(ids))
        assert len(draws) == 2
    finally:
        cls.__call__ = orig
    assert np.isfinite(float(loss))


class _Mixed(nn.Module):
    """A layer whose arguments mix what crosses the checkpoint (the hidden
    state, a tuple of floating arrays) with what stays in the closure (an
    integer gather index, a frozen module, a string)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def __call__(self, x, idx, pair, frozen, tag="mix"):
        h = mx.take_along_axis(self.proj(x) + frozen(x), idx, axis=-1)
        return h * pair[0] + pair[1] if tag == "mix" else h


class _MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(8, 8)
        self.frozen = nn.Linear(8, 8)
        self.frozen.freeze()
        self.layers = [_Mixed(), _Mixed()]

    def __call__(self, x, idx):
        h = self.emb(x)
        pair = (mx.sin(h), mx.tanh(h))
        h = self.layers[0](h, idx, pair, self.frozen, tag="mix")
        # a keyword argument crosses the checkpoint the same way
        return self.layers[1](h, idx, pair=pair, frozen=self.frozen, tag="mix")


def test_arguments_left_in_the_closure_keep_the_plain_gradient():
    """An integer index crossing the checkpoint would fail the gather's
    backward, and a frozen module or a string cannot cross at all. They
    stay in the closure while the tuple of floating arrays crosses, and
    every gradient equals the plain backward."""
    from mlx.utils import tree_flatten

    mx.random.seed(4)
    model = _MixedModel()
    x = mx.random.normal((3, 8))
    idx = mx.array(np.stack([np.random.default_rng(i).permutation(8) for i in range(3)]).astype(np.int32))

    def loss(m, x, idx):
        return (m(x, idx) ** 2).sum()

    g0 = dict(tree_flatten(nn.value_and_grad(model, loss)(model, x, idx)[1]))
    orig = _Mixed.__call__
    try:
        assert checkpoint_layers(model) == 1
        g1 = dict(tree_flatten(nn.value_and_grad(model, loss)(model, x, idx)[1]))
    finally:
        _Mixed.__call__ = orig
    assert g0.keys() == g1.keys() and "emb.weight" in g0 and "frozen.weight" not in g0
    for k in g0:
        assert np.allclose(np.array(g0[k]), np.array(g1[k]), atol=1e-5), k


def test_an_argument_the_recompute_cannot_carry_is_refused_at_call_time():
    """A dict can hold state the layer changes, a module with trainable
    parameters would train outside the checkpoint's inputs, and a
    floating array beside an integer one in a tuple would lose its
    gradient. Each makes the checkpointed call raise and names the
    argument."""

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def __call__(self, x, extra=None, state=None):
            return self.proj(x)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Layer()]

    model = _Model()
    x = mx.zeros((2, 8))
    orig = _Layer.__call__
    try:
        checkpoint_layers(model)
        layer = model.layers[0]
        with pytest.raises(ValueError, match=r"cannot run _Layer: its argument 'state' is a dict, which the "
                                             r"backward recompute would neither differentiate nor restore"):
            layer(x, state={})
        with pytest.raises(ValueError, match=r"its argument 1 is a Linear, which the backward recompute"):
            layer(x, nn.Linear(8, 8))
        with pytest.raises(ValueError, match=r"its argument 1 is a tuple holding floating arrays beside other "
                                             r"values, which would get no gradient"):
            layer(x, (x, mx.zeros((2,), dtype=mx.int32)))
        assert layer(x, (mx.zeros((2,), dtype=mx.int32), "a", None, 3)).shape == (2, 8)
    finally:
        _Layer.__call__ = orig


def test_a_class_refusal_raises_before_any_class_is_rewritten():
    """A layer class that passes state between layers declares it, and
    checkpoint_layers raises on it before it rewrites or marks a layer of
    any class."""

    class _Refused(nn.Module):
        _gmlx_checkpoint_refusal = "its layers share a bank"

        def __call__(self, x):
            return x

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Mixed(), _Refused()]

    model = _Model()
    orig = _Mixed.__call__
    try:
        with pytest.raises(ValueError, match="per-layer checkpointing cannot run _Refused: its layers share a bank"):
            checkpoint_layers(model)
        assert _Mixed.__call__ is orig
        assert not any(getattr(ly, "_gmlx_ckpt", False) for ly in model.layers)
    finally:
        _Mixed.__call__ = orig
