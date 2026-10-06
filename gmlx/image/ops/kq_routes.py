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
``glu`` keeps that route alone. It is read once, at import. ``only`` puts
a set of routes in force for a block, for a check that reads the stock
ops and the kernels in one process.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import mlx.core as mx

from gmlx.envflags import env_switch

from . import attention

ROUTES = ("qk", "glu")
_CHOICE = env_switch("GMLX_IMAGE_KQ_ROUTES", ROUTES)
_WANTED = {"1": ROUTES, "0": ()}.get(_CHOICE, (_CHOICE,))

_HALF = (mx.float16, mx.bfloat16)
# Head dims that have a qk_norm_rope kernel.
_QK_HEAD_DIMS = (64, 96, 128, 256)
# K tile of the GLU kernel by codec and x dtype, and its fewest rows. The
# rows follow the dispatch of mlx-kquant's quantized_matmul_glu, which
# also needs tensor-op hardware. Outside its dispatch the op composes with
# one more output buffer than the stock ops, so such a pair stays on the
# stock ops. The q8_0 kernel has no row: a q8_0 pair is no faster on it
# than on the stock ops.
_GLU_K_TILE = {("q4_k", mx.float16): 128, ("q4_k", mx.bfloat16): 128,
               ("q5_k", mx.float16): 128, ("q5_k", mx.bfloat16): 128,
               ("q6_k", mx.float16): 128, ("q6_k", mx.bfloat16): 128}
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
    force. Runs at import of ``gmlx.image.ops``. Raises while another
    function holds the seam, as under an installed ``StepProfiler``, whose
    removal would put back the function it saved."""
    global _active
    ns = sys.modules[__package__]
    held = ns.qk_norm_rope
    if held is not qk_norm_rope and held is not attention.qk_norm_rope:
        raise RuntimeError(
            "gmlx.image.ops.qk_norm_rope is replaced, as by an installed "
            "StepProfiler: remove the replacement before changing routes")
    want = _WANTED if routes is None else tuple(routes)
    on = set()
    if "qk" in want and _kq_op("qk_norm_rope") is not None:
        on.add("qk")
    if "glu" in want and _kq_op("quantized_matmul_glu") is not None:
        on.add("glu")
    ns.qk_norm_rope = qk_norm_rope if "qk" in on else attention.qk_norm_rope
    _active = frozenset(on)
    return _active


@contextmanager
def only(routes):
    """Put ``routes`` in force for the block, and after it the routes that
    were in force before. Yields the names in force inside the block. The
    switch is not read again. A graph built in the block keeps the ops it
    was built with."""
    before = _active
    on = install(routes)
    try:
        yield on
    finally:
        install(before)


def qk_norm_rope(q: mx.array, k: mx.array, q_w: mx.array, k_w: mx.array,
                 cos: mx.array, sin: mx.array, eps: float
                 ) -> tuple[mx.array, mx.array]:
    """The ``qk_norm_rope`` seam on mlx-kquant's kernel, which norms and
    rotates in fp32 and rounds once, where the stock seam also rounds the
    normalized value. The kernel takes float16 or bfloat16 q and k of one
    (B, L, H, D) shape at a head dim it is built for, norm weights of
    shape (D,) and tables of shape (L, D // 2), on the GPU. Any other call
    runs the stock seam, which also broadcasts the tables."""
    op = _kq_op("qk_norm_rope")
    if (op is not None and q.ndim == 4 and q.dtype in _HALF
            and k.dtype == q.dtype and k.shape == q.shape
            and q.shape[-1] in _QK_HEAD_DIMS
            and q_w.shape == k_w.shape == q.shape[-1:]
            and cos.shape == sin.shape == (q.shape[1], q.shape[-1] // 2)
            and mx.default_device() == mx.gpu):
        return op(q, k, q_w, k_w, cos, sin, eps)
    return attention.qk_norm_rope(q, k, q_w, k_w, cos, sin, eps)


def kquant_leaf(m, dtype):
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
    codec and shape with no bias and no live LoRA delta, in a codec it is
    built for at the dtype of x, an x with an inner dim on the codec's K
    tile and at least 1024 rows over its leading dims, on a GPU with
    tensor ops."""
    if "glu" not in _active or x.ndim < 2:
        return None
    k = x.shape[-1]
    if k == 0 or x.size // k < _GLU_MIN_ROWS:
        return None
    if mx.default_device() != mx.gpu or not _nax():
        return None
    g, u = kquant_leaf(gate, x.dtype), kquant_leaf(up, x.dtype)
    if g is None or u is None:
        return None
    codec = g.kquant_type
    tile = _GLU_K_TILE.get((codec, x.dtype))
    if tile is None or k % tile or u.kquant_type != codec:
        return None
    if "bias" in g or "bias" in u:
        return None
    w_gate, w_up = g["weight"], u["weight"]
    if w_gate.shape != w_up.shape or w_gate.shape[0] == 0:
        return None
    return w_gate, w_up, codec


def swiglu_fused(x: mx.array, w_gate: mx.array, w_up: mx.array,
                 codec: str) -> mx.array:
    """``silu(x @ w_gate^T) * (x @ w_up^T)`` over the wire bytes of two
    K-quant weights in one kernel, for a pair that ``glu_pair`` returned.
    The kernel takes both products in fp32, rounds the activation once and
    its product with the up product once. The stock ops also round each
    matmul product and the sigmoid. No gradient."""
    return _kq_op("quantized_matmul_glu")(x, w_gate, w_up, codec)
