# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Routes of image ops onto mlx-kquant kernels.

``install`` replaces the ``qk_norm_rope`` seam with ``qk_norm_rope`` here
when the installed mlx-kquant has the op. ``glu_pair`` and ``swiglu_fused``
run the gate and up products of a feed-forward and their swiglu as one
kernel over two K-quant weights. A call that a kernel does not take runs
the stock ops, and so does every call on an mlx-kquant without the op.

The kernels have no gradient, so ``mx.grad`` through a routed call raises.
The image pipeline is inference only.

GMLX_IMAGE_KQ_ROUTES=0 keeps both routes on stock ops, and ``qk`` or
``glu`` keeps that route alone. It is read once, at import.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from gmlx.envflags import env_choice

from . import attention

ROUTES = ("qk", "glu")
_CHOICE = env_choice("GMLX_IMAGE_KQ_ROUTES", "1", ("1", "0") + ROUTES)
_WANTED = {"1": ROUTES, "0": ()}.get(_CHOICE, (_CHOICE,))

_HALF = (mx.float16, mx.bfloat16)
# Head dims that have a qk_norm_rope kernel.
_QK_HEAD_DIMS = (64, 96, 128, 256)
# Codecs that have a GLU kernel, its K tile and its fewest rows. The
# kernel runs on tensor-op hardware only.
_GLU_CODECS = ("q4_k",)
_GLU_K_TILE = 128
_GLU_MIN_ROWS = 1024

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


def _nax() -> bool:
    if "nax" not in _OPS:
        probe = _kq_op("nax_available")
        _OPS["nax"] = bool(probe is not None and probe())
    return _OPS["nax"]


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
    if "glu" in want and _kq_op("quantized_matmul_glu") is not None:
        on.add("glu")
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


def _kquant_leaf(m, dtype):
    """The KQuantLinear whose product is all of ``m(x)`` for x in
    ``dtype``, or None. That is a plain KQuantLinear, or the base of an
    ImageLoRALinear that has no delta for its path under the active patch
    table. A leaf that StepProfiler timed counts as its own class."""
    from mlx_kquant.nn import KQuantLinear

    from gmlx.image.lora.install import ImageLoRALinear

    if isinstance(m, ImageLoRALinear):
        if m._state.factors(m._path, dtype) is not None:
            return None
        m = m.base
    cls = type(m)
    if getattr(cls, "_prof_base", cls) is not KQuantLinear:
        return None
    return m


def glu_pair(gate, up, x: mx.array):
    """``(w_gate, w_up, codec)`` when ``swiglu_fused`` takes
    ``silu(gate(x)) * up(x)``, else None, and the caller then composes the
    two products and ``swiglu``. The kernel takes two K-quant leaves of one
    codec and shape with no bias and no live LoRA delta, a float16 or
    bfloat16 x with an inner dim on the K tile and at least 1024 rows over
    its leading dims, on a GPU with tensor ops."""
    if "glu" not in _active or x.dtype not in _HALF or x.ndim < 2:
        return None
    k = x.shape[-1]
    if k == 0 or k % _GLU_K_TILE or x.size // k < _GLU_MIN_ROWS:
        return None
    if mx.default_device() != mx.gpu or not _nax():
        return None
    g, u = _kquant_leaf(gate, x.dtype), _kquant_leaf(up, x.dtype)
    if g is None or u is None:
        return None
    codec = g.kquant_type
    if codec not in _GLU_CODECS or u.kquant_type != codec:
        return None
    if "bias" in g or "bias" in u:
        return None
    w_gate, w_up = g["weight"], u["weight"]
    if w_gate.shape != w_up.shape:
        return None
    return w_gate, w_up, codec


def swiglu_fused(x: mx.array, w_gate: mx.array, w_up: mx.array,
                 codec: str) -> mx.array:
    """``silu(x @ w_gate^T) * (x @ w_up^T)`` over the wire bytes of two
    K-quant weights in one kernel, for a pair that ``glu_pair`` returned.
    Both products stay in fp32 until the one store, where the stock ops
    round each product first. No gradient."""
    return _kq_op("quantized_matmul_glu")(x, w_gate, w_up, codec)
