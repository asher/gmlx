"""Per-layer activation checkpointing for training.

``checkpoint_layers`` rewrites the ``__call__`` of every decoder-layer
class in a model so the forward runs under ``mx.checkpoint``: the gradient
tape keeps each layer's inputs and recomputes its activations in the
backward. The tape then holds one set of layer inputs per layer instead of
every intermediate of every layer, which is what bounds the tokens a step
can hold. mlx-lm's ``grad_checkpoint`` does this for the class of
``model.layers[0]``; this version covers models whose layers are of more
than one class and models that keep their layers under a language model
wrapper.
"""
from __future__ import annotations

import mlx.core as mx


def _layer_list(model):
    for owner in (model, getattr(model, "language_model", None),
                  getattr(model, "model", None)):
        if owner is None:
            continue
        layers = getattr(owner, "layers", None)
        if layers is None:
            inner = getattr(owner, "model", None)
            layers = getattr(inner, "layers", None)
        if layers:
            return layers
    return []


def checkpoint_layers(model) -> int:
    """Install ``mx.checkpoint`` around every decoder-layer class of
    ``model``. Returns the number of classes rewritten; a class already
    rewritten is left alone, so the call is idempotent."""
    n = 0
    for layer in _layer_list(model):
        cls = type(layer)
        if getattr(cls.__call__, "_gmlx_checkpointed", False):
            continue
        fn = cls.__call__

        def make(fn):
            def checkpointed(self, *args, **kwargs):
                def inner(params, *args, **kwargs):
                    self.update(params)
                    return fn(self, *args, **kwargs)
                return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)
            checkpointed._gmlx_checkpointed = True
            checkpointed._gmlx_orig = fn
            return checkpointed

        cls.__call__ = make(fn)
        n += 1
    return n
