"""Selection ids without a gradient, for training.

A MoE router or a sparse-attention indexer picks ids with
``mx.argpartition``, ``mx.argsort``, ``mx.argmax`` or ``mx.argmin`` over
scores that depend on the trained weights, then gathers at those ids.
MLX refuses the backward of a gather with respect to its indices unless
the ids pass through ``mx.stop_gradient``, and most mlx-lm and mlx-vlm
MoE blocks do not wrap them, so their training forward has no gradient.
``install_index_stop_gradient`` replaces the four functions in
``mlx.core`` with versions that return ``mx.stop_gradient`` of their
result, for the length of a training run. Integer ids carry no gradient
in any case, so no value changes.
"""
from __future__ import annotations

import mlx.core as mx

_NAMES = ("argpartition", "argsort", "argmax", "argmin")


class _Detached:
    """``fn`` with its result passed through ``mx.stop_gradient``."""

    _gmlx_index_stop_gradient = True

    def __init__(self, fn):
        self._gmlx_orig = fn
        self.__name__ = getattr(fn, "__name__", "op")
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, *args, **kwargs):
        return mx.stop_gradient(self._gmlx_orig(*args, **kwargs))


class _Restore:
    """Puts back the functions an install replaced. ``count`` is how many
    it replaced."""

    def __init__(self, patched):
        self._patched = patched
        self.count = len(patched)

    def __call__(self):
        for name, fn in self._patched:
            setattr(mx, name, fn)
        self._patched.clear()


def install_index_stop_gradient():
    """Replace ``mx.argpartition``, ``mx.argsort``, ``mx.argmax`` and
    ``mx.argmin`` with versions whose result passes through
    ``mx.stop_gradient``, and return a callable that puts the originals
    back. The replacement is process-wide. A second install while one is
    in place changes nothing, and its restore leaves the first in place."""
    patched = []
    for name in _NAMES:
        fn = getattr(mx, name)
        if getattr(fn, "_gmlx_index_stop_gradient", False):
            continue
        setattr(mx, name, _Detached(fn))
        patched.append((name, fn))
    return _Restore(patched)
