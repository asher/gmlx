"""layer_capture and block_outputs on minted models of the shapes the hook
must handle: plain layers, tuple returns, a fused subclass that calls its
base through super() and hands the next layer its output, and a wide
hyper-connection residual. Each forward stops after the trunk."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.models.vlm_text_only import TextOnlyModel
from gmlx.tune.capture import block_outputs, capture_layers, layer_capture

_D, _VOCAB = 16, 32
_IDS = [3, 1, 4, 1, 5, 9, 2]


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _weight(seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((_D, _D)).astype(np.float32) * 0.2)


class _Block(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        self.w = _weight(seed)

    def __call__(self, x, mask=None, cache=None):
        return x + mx.tanh(x @ self.w)


class _TupleBlock(_Block):
    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=0):
        h = super().__call__(x)
        return h, (h * 2, h * 3), offset + x.shape[1]


class _Fused(_Block):
    """Calls its base through super(), and reuses what the previous fused
    layer handed it only when the handed array is its very input."""

    def __call__(self, x, mask=None, cache=None):
        ho = self.__dict__.pop("_handoff", None)
        if ho is not None and ho[0] is x:
            self.__dict__["_hits"].append(1)
        out = super().__call__(x)
        nxt = self.__dict__.get("_next")
        if nxt is not None:
            nxt.__dict__["_handoff"] = (out, None)
        return out


class _Wide(_Block):
    """A layer on the [B, T, streams, D] residual of a hyper-connection
    model."""

    def __call__(self, x, mask=None, cache=None):
        mix = mx.arange(1, x.shape[2] + 1, dtype=mx.float32)[None, None, :, None]
        return x + mx.tanh(x @ self.w) * mix


class _Trunk(nn.Module):
    def __init__(self, layers, streams: int = 0, run: int | None = None):
        super().__init__()
        rng = np.random.default_rng(7)
        self.embed = mx.array(rng.standard_normal((_VOCAB, _D)).astype(np.float32))
        self.layers = layers
        self.streams = streams
        self.run = len(layers) if run is None else run
        self.seen: list = []

    def __call__(self, inputs, cache=None, keep_last=0):
        self.seen.append(keep_last)
        h = self.embed[inputs]
        if self.streams:
            h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], self.streams, _D))
        for layer in self.layers[:self.run]:
            out = layer(h)
            if isinstance(out, tuple):
                assert len(out) == 3
                out = out[0]
            h = out
        return h


class _LM(nn.Module):
    def __init__(self, trunk):
        super().__init__()
        self.model = trunk

    def __call__(self, inputs, cache=None):
        self.model(inputs, cache=cache)
        raise AssertionError("the head ran")

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [None] * len(self.layers)


class _Nested(nn.Module):
    """The qwen3.5 shape: the text model sits under ``language_model``."""

    def __init__(self, trunk):
        super().__init__()
        self.language_model = _LM(trunk)

    def __call__(self, inputs, cache=None):
        return self.language_model(inputs, cache=cache)

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()


def _wrap(trunk, nested=False):
    inner = _Nested(trunk) if nested else _LM(trunk)
    return TextOnlyModel(inner, config={"model_type": "toy"})


def _reference(trunk, positions: int) -> np.ndarray:
    """Each layer's output, the streams averaged, over the last
    ``positions`` columns, from a plain loop with no hook."""
    h = trunk.embed[mx.array([_IDS])]
    if trunk.streams:
        h = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], trunk.streams, _D))
    rows = []
    for layer in trunk.layers[:trunk.run]:
        out = layer(h)
        h = out[0] if isinstance(out, tuple) else out
        a = np.array(h, dtype=np.float32)
        if a.ndim == 4:
            a = a.mean(axis=2)
        rows.append(a[0, -positions:].mean(axis=0))
    return np.stack(rows)


def _capture(model, positions: int, n_layer=None) -> np.ndarray:
    layers = capture_layers(model, n_layer)
    with layer_capture(layers, positions) as rec:
        out = block_outputs(model, _IDS, rec)
    return np.array(out)


@pytest.mark.parametrize("positions, nested", [(1, False), (3, False), (50, False), (2, True)])
def test_plain_layers_match_the_loop_and_skip_the_head(positions, nested):
    trunk = _Trunk([_Block(i) for i in range(3)])
    got = _capture(_wrap(trunk, nested), positions)
    assert got.shape == (3, _D)
    np.testing.assert_allclose(got, _reference(trunk, positions), rtol=1e-5, atol=1e-6)
    # Every layer runs every column.
    assert trunk.seen == [None]


def test_tuple_returns_record_the_hidden_state():
    trunk = _Trunk([_TupleBlock(i) for i in range(3)])
    got = _capture(_wrap(trunk), 2)
    np.testing.assert_allclose(got, _reference(trunk, 2), rtol=1e-5, atol=1e-6)


def test_fused_layers_record_once_and_keep_the_handoff():
    hits: list = []
    kinds = [_Fused, _Fused, _Block, _Fused, _Fused]
    layers = [k(i) for i, k in enumerate(kinds)]
    for a, b in zip(layers, layers[1:]):
        if isinstance(a, _Fused):
            a.__dict__["_hits"] = hits
            if isinstance(b, _Fused):
                a.__dict__["_next"] = b
    layers[-1].__dict__["_hits"] = hits
    trunk = _Trunk(layers)
    want = _reference(trunk, 2)
    assert len(hits) == 2
    hits.clear()
    fused_call, block_call = _Fused.__call__, _Block.__call__
    got = _capture(_wrap(trunk), 2)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # The wrapper hands back the layer's own array, so each reuse still fires.
    assert len(hits) == 2
    assert _Fused.__call__ is fused_call and _Block.__call__ is block_call


def test_inherited_call_is_restored_as_inherited():
    class _Plain(_Block):
        pass

    trunk = _Trunk([_Plain(0), _Plain(1)])
    _capture(_wrap(trunk), 1)
    assert "__call__" not in _Plain.__dict__


def test_streams_are_averaged():
    trunk = _Trunk([_Wide(i) for i in range(2)], streams=4)
    got = _capture(_wrap(trunk), 2)
    np.testing.assert_allclose(got, _reference(trunk, 2), rtol=1e-5, atol=1e-6)


def test_trailing_layers_past_the_trunk_are_left_out():
    # Two blocks after the trunk's layers, as an MTP or nextn block sits.
    trunk = _Trunk([_Block(i) for i in range(5)], run=3)
    got = _capture(_wrap(trunk), 1, n_layer=3)
    np.testing.assert_allclose(got, _reference(trunk, 1), rtol=1e-5, atol=1e-6)


def test_a_layer_that_never_ran_is_an_error():
    trunk = _Trunk([_Block(i) for i in range(3)], run=2)
    with pytest.raises(RuntimeError, match=r"layers \[2\] ran \[0\] times"):
        _capture(_wrap(trunk), 1)


def test_layers_run_plain_outside_the_block():
    layer = _Block(0)
    trunk = _Trunk([layer])
    model = _wrap(trunk)
    with layer_capture(capture_layers(model), 1):
        pass
    assert "_gmlx_capture_idx" not in layer.__dict__
    x = mx.ones((1, 2, _D))
    np.testing.assert_allclose(np.array(layer(x)), np.array(x + mx.tanh(x @ layer.w)))
