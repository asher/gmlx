# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Routes of image ops onto mlx-kquant kernels.

``install`` replaces the ``qk_norm_rope`` seam with ``qk_norm_rope`` here
when the installed mlx-kquant has the op. A call that the kernel does not
take runs the stock seam, and so does every call on an mlx-kquant without
the op.

The kernel has no gradient, so ``mx.grad`` through a routed call raises.
The image pipeline is inference only.

GMLX_IMAGE_KQ_ROUTES=0 keeps the seam on stock ops. It is read once, at
import.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from gmlx.envflags import env_choice

from . import attention

ROUTES = ("qk",)
_CHOICE = env_choice("GMLX_IMAGE_KQ_ROUTES", "1", ("1", "0") + ROUTES)
_WANTED = {"1": ROUTES, "0": ()}.get(_CHOICE, (_CHOICE,))

_HALF = (mx.float16, mx.bfloat16)
# Head dims that have a qk_norm_rope kernel.
_QK_HEAD_DIMS = (64, 96, 128, 256)

_OPS: dict = {}
_active: frozenset = frozenset()


def _kq_op(name: str):
    """mlx_kquant's ``name`` when the build carries it, else None."""
    if name not in _OPS:
        op = None
        try:
            import mlx_kquant as kq
            op = getattr(kq, name, None)
        except ImportError:
            pass
        _OPS[name] = op
    return _OPS[name]


def active() -> frozenset:
    """Names of the routes in force."""
    return _active


def install(routes=None) -> frozenset:
    """Put in force the routes of GMLX_IMAGE_KQ_ROUTES, or of ``routes``,
    that the installed mlx-kquant has an op for, and return their names.
    The ``qk_norm_rope`` seam of ``gmlx.image.ops`` is replaced by
    assignment, or set back to the stock function when its route is not in
    force. Runs at import of ``gmlx.image.ops``."""
    global _active
    want = _WANTED if routes is None else tuple(routes)
    on = set()
    if "qk" in want and _kq_op("qk_norm_rope") is not None:
        on.add("qk")
    setattr(sys.modules[__package__], "qk_norm_rope",
            qk_norm_rope if "qk" in on else attention.qk_norm_rope)
    _active = frozenset(on)
    return _active


def qk_norm_rope(q: mx.array, k: mx.array, q_w: mx.array, k_w: mx.array,
                 cos: mx.array, sin: mx.array, eps: float
                 ) -> tuple[mx.array, mx.array]:
    """The ``qk_norm_rope`` seam on mlx-kquant's kernel, which norms and
    rotates in fp32 and rounds once, where the stock seam also rounds the
    normalized value. The kernel takes float16 or bfloat16 q and k of one
    shape, on the GPU, at a head dim it is built for. Any other call runs
    the stock seam."""
    op = _kq_op("qk_norm_rope")
    if (op is not None and q.dtype in _HALF and k.dtype == q.dtype
            and q.ndim == 4 and k.shape == q.shape
            and q.shape[-1] in _QK_HEAD_DIMS
            and mx.default_device() == mx.gpu):
        return op(q, k, q_w, k_w, cos, sin, eps)
    return attention.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps)
