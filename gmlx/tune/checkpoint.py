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

Only floating arrays, alone or in a tuple or list of floating arrays,
enter the checkpoint and get a gradient. The rest of a layer's arguments
are read from the closure: integer arrays (a gather has no gradient with
respect to its indices), strings, numbers, None and modules without
trainable parameters. Any other argument makes the wrapped call raise. A
floating array in a container beside other values would lose its
gradient, and an object can hold state the layer reads or changes, which
the recompute would neither differentiate nor restore. A layer class that
passes such state declares it in ``_gmlx_checkpoint_refusal``, and
``checkpoint_layers`` raises before it rewrites anything.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


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


def _floating(a) -> bool:
    return isinstance(a, mx.array) and mx.issubdtype(a.dtype, mx.floating)


def _crosses(a) -> bool:
    """An argument the checkpoint takes as an input: a floating array, or
    a non-empty tuple or list of them (the key and value arrays a gemma-4
    layer hands a later one)."""
    if isinstance(a, (tuple, list)):
        return len(a) > 0 and all(_floating(x) for x in a)
    return _floating(a)


def _closable(a) -> bool:
    """An argument the recompute may read from the closure: it carries no
    gradient and the layer cannot change it."""
    if isinstance(a, (tuple, list)):
        return all(_closable(x) for x in a)
    if a is None or isinstance(a, (str, int, float, bool)):
        return True
    if isinstance(a, mx.array):
        return not _floating(a)
    if isinstance(a, nn.Module):
        return not tree_flatten(a.trainable_parameters())
    return False


def _refusal(a) -> str:
    if isinstance(a, (tuple, list)) and any(_floating(x) for x in a):
        return f"a {type(a).__name__} holding floating arrays beside other values, which would get no gradient"
    return f"a {type(a).__name__}, which the backward recompute would neither differentiate nor restore"


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
    training loops only: a step under ``mx.compile`` cannot use it.
    Raises ValueError, with nothing rewritten, when a layer class declares
    a ``_gmlx_checkpoint_refusal``."""
    for layer in layer_list(model):
        why = getattr(type(layer), "_gmlx_checkpoint_refusal", None)
        if why:
            raise ValueError(f"per-layer checkpointing cannot run {type(layer).__name__}: {why}")
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
                named = list(enumerate(args)) + list(kwargs.items())
                cross = [key for key, a in named if _crosses(a)]
                for key, a in named:
                    if key not in cross and not _closable(a):
                        what = f"argument {key}" if isinstance(key, int) else f"argument {key!r}"
                        raise ValueError(
                            f"per-layer checkpointing cannot run {type(self).__name__}: its {what} is "
                            f"{_refusal(a)}")

                def inner(params, *vals):
                    if seed is not None:
                        mx.random.seed(seed)
                    self.update(params)
                    a2, k2 = list(args), dict(kwargs)
                    for key, v in zip(cross, vals):
                        if isinstance(key, int):
                            a2[key] = v
                        else:
                            k2[key] = v
                    return fn(self, *a2, **k2)
                return mx.checkpoint(inner)(self.trainable_parameters(),
                                            *[args[k] if isinstance(k, int) else kwargs[k] for k in cross])
            checkpointed._gmlx_checkpointed = True
            checkpointed._gmlx_orig = fn
            return checkpointed

        cls.__call__ = make(fn)
        n += 1
    return n
