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
import mlx.nn as nn


def draws_random(module) -> bool:
    """True when a training forward of ``module`` draws from the global
    random stream: a dropout with p > 0 somewhere inside it."""
    return bool(module.training) and any(isinstance(m, nn.Dropout) and float(m._p_1) < 1.0
                                          for m in module.modules())


def layer_seed() -> int:
    """One seed drawn from the global stream, for a recompute to replay."""
    return int(mx.random.randint(0, 2 ** 31 - 1, shape=()).item())


def layer_list(model):
    """The decoder layers of ``model``, looking under a language-model
    wrapper and an inner ``model`` attribute as mlx-lm and mlx-vlm lay
    them out. Empty when none are found."""
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


def checkpoint_layers(model, replay_dropout: bool = False) -> int:
    """Install ``mx.checkpoint`` around every decoder-layer class of
    ``model`` and mark its layers to take it. Returns the number of
    classes rewritten; a class already rewritten is left alone, so the
    call is idempotent. The mark lives on the layer instances: a layer
    of the same class in a model this was never asked for runs its
    original forward, and a marked layer takes the ``replay_dropout`` of
    the latest call for its model. With ``replay_dropout`` a layer that
    draws dropout masks draws one seed per call and replays it in the
    backward recompute. The draw evaluates an array, so it serves eager
    training loops only: a step under ``mx.compile`` cannot use it."""
    n = 0
    for layer in layer_list(model):
        layer._gmlx_ckpt = True
        layer._gmlx_replay_dropout = bool(replay_dropout)
        cls = type(layer)
        if getattr(cls.__call__, "_gmlx_checkpointed", False):
            continue
        fn = cls.__call__

        def make(fn):
            def checkpointed(self, *args, **kwargs):
                if not getattr(self, "_gmlx_ckpt", False):
                    return fn(self, *args, **kwargs)
                # the backward recomputes the layer, and a dropout inside it
                # would draw fresh masks from the global stream by then: the
                # layer's seed is drawn once here and replayed in the recompute
                replay = getattr(self, "_gmlx_replay_dropout", False)
                seed = layer_seed() if (replay and draws_random(self)) else None

                def inner(params, *args, **kwargs):
                    if seed is not None:
                        mx.random.seed(seed)
                    self.update(params)
                    return fn(self, *args, **kwargs)
                return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)
            checkpointed._gmlx_checkpointed = True
            checkpointed._gmlx_orig = fn
            return checkpointed

        cls.__call__ = make(fn)
        n += 1
    return n
