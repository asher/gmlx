"""Run-time rotation for Hadamard-folded projections and embeddings.

``install_hadamard_modules`` swaps the class of every ``KQuantLinear`` and
``KQuantEmbedding`` that ``hadamard.resolve_hadamard_targets`` named, so
the rotation lives where the S=1 decode routes and the drafter binders
call the projection directly. The rotation itself is one of two forms:
``kq.hadamard_rotate`` when the installed mlx-kquant has it and the GPU is
the default device, else MLX ops (an f32 upcast, the sign multiply and
``mx.hadamard_transform`` per block), which is also the CPU path.

Switches: ``GMLX_HADAMARD_KERNEL=0`` forces the MLX-op form,
``GMLX_HADAMARD_TRACE=1`` counts rotations (``rotation_count``), and
``GMLX_HADAMARD_ROTATE=0`` skips the rotation entirely, which makes the
model produce garbage and measures the rotation's price.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_kquant.codec_geometry import bytes_per_row
from mlx_kquant.nn import KQuantEmbedding, KQuantLinear

from .hadamard import FoldTarget

_count = 0


def rotation_count() -> int:
    """Rotations applied since the last reset (``GMLX_HADAMARD_TRACE=1``)."""
    return _count


def reset_rotation_count() -> None:
    global _count
    _count = 0


class _Fold:
    """Per-module rotation state, attached outside the parameter tree."""

    __slots__ = ("width", "block", "signs", "perm", "inverse", "key")

    def __init__(self, target: FoldTarget, signs: mx.array | None):
        self.width = target.width
        self.block = target.block
        self.signs = signs
        self.perm = target.perm
        self.inverse = target.inverse
        # Rotations with the same key can be shared across modules.
        self.key = (target.width, target.block, id(signs), target.perm)


# Block widths the kq kernel instantiates; any other width takes the
# MLX-op form.
_KERNEL_BLOCKS = frozenset({256, 512, 1024, 2048, 4096})


def _kernel(block: int):
    if block not in _KERNEL_BLOCKS:
        return None
    if os.environ.get("GMLX_HADAMARD_KERNEL", "1") == "0":
        return None
    if mx.default_device() != mx.gpu:
        return None
    import mlx_kquant as kq
    return getattr(kq, "hadamard_rotate", None)


def _blocked_transform(xf: mx.array, block: int) -> mx.array:
    lead = xf.shape[:-1]
    y = mx.hadamard_transform(
        xf.reshape(*lead, -1, block), scale=block ** -0.5)
    return y.reshape(*lead, -1)


def rotate(x: mx.array, fold: _Fold, kernel: bool = True) -> mx.array:
    """The forward fold of a projection input: permute, sign, transform.
    Returns ``x.dtype``. ``kernel`` False keeps to the MLX ops, which have
    a backward: the kq kernel has none."""
    global _count
    if os.environ.get("GMLX_HADAMARD_ROTATE") == "0":
        return x
    if os.environ.get("GMLX_HADAMARD_TRACE") == "1":
        _count += 1
    kq_rotate = _kernel(fold.block) if kernel else None
    if kq_rotate is not None:
        return kq_rotate(x, fold.signs, block=fold.block, perm=fold.perm)
    if fold.perm is not None:
        rep, nk, hd = fold.perm
        lead = x.shape[:-1]
        x = x.reshape(*lead, rep, nk, hd).swapaxes(-3, -2).reshape(*lead, -1)
    xf = x.astype(mx.float32)
    if fold.signs is not None:
        xf = xf * fold.signs
    return _blocked_transform(xf, fold.block).astype(x.dtype)


def rotate_inverse(rows: mx.array, fold: _Fold) -> mx.array:
    """The embedding un-rotation: transform, then sign. Returns
    ``rows.dtype``."""
    global _count
    if os.environ.get("GMLX_HADAMARD_ROTATE") == "0":
        return rows
    if os.environ.get("GMLX_HADAMARD_TRACE") == "1":
        _count += 1
    kernel = _kernel(fold.block)
    if kernel is not None:
        out = kernel(rows, None, block=fold.block)
        if fold.signs is not None:
            out = out * fold.signs.astype(rows.dtype)
        return out
    xf = _blocked_transform(rows.astype(mx.float32), fold.block)
    if fold.signs is not None:
        xf = xf * fold.signs
    return xf.astype(rows.dtype)


class HadamardKQuantLinear(KQuantLinear):
    """A folded projection: rotates its input unless the caller already
    did (``pre_rotated=True``, the shared-rotation sites)."""

    def __call__(self, x, lora=None, *, pre_rotated=False):
        if not pre_rotated:
            x = rotate(x, self._hadamard, kernel=not self.training)
        return super().__call__(x, lora=lora)


class HadamardKQuantEmbedding(KQuantEmbedding):
    """A folded embedding table: un-rotates the gathered rows, and rotates
    the input of the tied-head projection."""

    def __call__(self, x):
        return rotate_inverse(super().__call__(x), self._hadamard)

    def as_linear(self, x):
        return super().as_linear(rotate(x, self._hadamard, kernel=not self.training))


def is_folded(module) -> bool:
    return getattr(module, "_hadamard", None) is not None


def shared_linears(modules, x: mx.array) -> tuple:
    """Call each module on ``x``, rotating once for the members that share
    a fold (same width, block and signs, no permute). Members without a
    fold, with a permute, or whose fold no other member shares are called
    plainly. This is the sharing site for the projection groups that read
    one activation: qkv+z, q+k+v and gate+up."""
    folds = [getattr(m, "_hadamard", None) for m in modules]
    if all(f is None for f in folds):
        return tuple(m(x) for m in modules)
    keys = [
        f.key if f is not None and f.perm is None and not f.inverse else None
        for f in folds
    ]
    rotated: dict = {}
    outs = []
    for m, f, key in zip(modules, folds, keys):
        if key is None or keys.count(key) < 2:
            outs.append(m(x))
            continue
        xr = rotated.get(key)
        if xr is None:
            xr = rotate(x, f, kernel=not m.training)
            rotated[key] = xr
        outs.append(m(xr, pre_rotated=True))
    return tuple(outs)


def _match(path: str, targets: dict[str, FoldTarget]) -> str | None:
    for key in targets:
        if path == key or path.endswith("." + key):
            return key
    return None


def install_hadamard_modules(model: nn.Module,
                             targets: dict[str, FoldTarget]) -> int:
    """Swap every targeted ``KQuantLinear`` / ``KQuantEmbedding`` for its
    Hadamard subclass and attach the fold. Runs after
    ``install_kquant_modules`` and before any fusion that reads the
    projection classes. Idempotent; a target with no module, or a module
    of the wrong class or width, raises. Returns the swap count."""
    if not targets:
        return 0
    signs_by_width: dict[int, mx.array] = {}

    def signs_for(t: FoldTarget) -> mx.array | None:
        if t.signs is None:
            return None
        arr = signs_by_width.get(t.width)
        if arr is None:
            arr = mx.array(np.asarray(t.signs, dtype=np.float32))
            signs_by_width[t.width] = arr
        return arr

    matched: set[str] = set()
    n = 0
    for path, m in model.named_modules():
        key = _match(path, targets)
        if key is None:
            continue
        t = targets[key]
        base, wrapped = ((KQuantEmbedding, HadamardKQuantEmbedding) if t.inverse
                         else (KQuantLinear, HadamardKQuantLinear))
        if type(m) is wrapped:
            matched.add(key)
            continue
        if type(m) is not base:
            raise TypeError(
                f"Hadamard-folded weight at {path!r} sits on "
                f"{type(m).__module__}.{type(m).__name__}, not {base.__name__}")
        expected = bytes_per_row(m.kquant_type, t.width)
        if int(m.weight.shape[-1]) != expected:
            raise ValueError(
                f"Hadamard-folded weight at {path!r}: row width "
                f"{int(m.weight.shape[-1])} bytes, expected {expected} for "
                f"input width {t.width} ({m.kquant_type})")
        m.__class__ = wrapped
        object.__setattr__(m, "_hadamard", _Fold(t, signs_for(t)))
        matched.add(key)
        n += 1
    missing = sorted(set(targets) - matched)
    if missing:
        head = ", ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        raise ValueError(
            f"{len(missing)} Hadamard-folded weights have no module to "
            f"attach to: {head}{more}")
    return n
