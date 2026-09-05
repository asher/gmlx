"""Shared scaffolding for the kvarn cache tests.

``needs_kvarn_ops`` skips when the kvarn kernel surface is unusable: CPU
default device, mlx-kquant absent, or a build without the ops (the same
probe the scheme resolver consults). A plain GPU check would let a GPU
box with an op-less wheel fail on AttributeError instead of skipping.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from gmlx.cache.kvarn_cache import KVarNKVCache

needs_kvarn_ops = pytest.mark.needs_kvarn_ops

H = 2
D = 128


def tokens(n, seed=0, d=D):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    v = mx.array(rng.standard_normal((1, H, n, d)).astype(np.float16))
    return k, v


def filled(n, tail=384, seed=0, d=D, **kw):
    c = KVarNKVCache(tail_tokens=tail, **kw)
    c.update_and_fetch(*tokens(n, seed, d=d))
    return c


class Args:
    """A model args stand-in: llama-shaped, head_dim 128 unless overridden."""

    def __init__(self, **kw):
        self.model_type = kw.pop("model_type", "llama")
        self.head_dim = kw.pop("head_dim", 128)
        for k, v in kw.items():
            setattr(self, k, v)


class FakeLM:
    """A language model whose make_cache builds plain KV layers, or the
    given cache factories."""

    def __init__(self, n_layers=2, stack=None, **kw):
        self.args = Args(**kw)
        self._stack = stack
        self.layers = [object()] * (len(stack) if stack is not None else n_layers)

    def make_cache(self):
        if self._stack is not None:
            return [f() for f in self._stack]
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(len(self.layers))]
