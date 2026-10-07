# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Routes of image ops onto mlx-kquant kernels.

``install`` replaces the ``qk_norm_rope`` seam with ``qk_norm_rope`` here
when the installed mlx-kquant has the op. ``glu_pair`` and ``swiglu_fused``
run the gate and up products of a feed-forward and their swiglu as one
kernel over two K-quant weights. ``lora_leaf`` and ``lora_fused`` run the
product of a K-quant leaf and the low-rank term of its adapter as one op.
A call that a kernel does not take runs the stock ops, and so does every
call on an mlx-kquant without the op.

No kernel has a gradient with respect to a weight, and the qk kernel has
none at all, so ``mx.grad`` through a routed qk call raises. The image
pipeline is inference only.

GMLX_IMAGE_KQ_ROUTES=0 keeps every route on stock ops, and ``qk``,
``glu`` or ``lora`` keeps that route alone. It is read once, at import.
``only`` puts a set of routes in force for a block, for a check that reads
the stock ops and the kernels in one process. The routes in force are
state of the process, and the image pipeline keys the results it keeps by
them.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import mlx.core as mx

from gmlx.envflags import env_switch

from . import attention

ROUTES = ("qk", "glu", "lora")
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
# The calls the lora route hands to mlx-kquant's quantized_matmul_lora. They
# are inside the op's own kernel dispatch, which also needs tensor-op
# hardware: outside it the op composes from its unfused ops, a step slower
# than the stock ops with the strength in B, so such a call stays on the
# stock ops. The rows are the codecs and x dtype of the measured steps, an
# inner dim of at most 4096, at least 8192 outputs, where the kernel runs
# from 1024 rows at every rank it takes, at most 256 ranks on a column and
# at most 16 bands that start on a multiple of 64 columns and end on a
# multiple of 32 or at the last.
_LORA_ROWS = (("q4_k", mx.bfloat16), ("q5_k", mx.bfloat16),
              ("q6_k", mx.bfloat16))
_LORA_MAX_K = 4096
_LORA_MIN_N = 8192
_LORA_MIN_ROWS = 1024
_LORA_MAX_RANKS = 256
_LORA_MAX_BANDS = 16

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


def _gpu() -> bool:
    """Whether the default device is the GPU, where the kernels run."""
    return mx.default_device() == mx.gpu


def active() -> frozenset:
    """Names of the routes in force."""
    return _active


def _named(routes) -> tuple:
    """``routes`` as a tuple of names of ROUTES. A str, None and an unknown
    name raise, each by name."""
    if routes is None:
        raise TypeError(f"routes is None: pass a tuple of names of {ROUTES}, "
                        f"or () for no route")
    if isinstance(routes, str):
        raise TypeError(f"routes is the str {routes!r}: pass a tuple of "
                        f"names of {ROUTES}, such as ({routes!r},)")
    want = tuple(routes)
    for name in want:
        if name not in ROUTES:
            raise ValueError(f"unknown route {name!r}: the routes are "
                             f"{ROUTES}")
    return want


def install(routes=None) -> frozenset:
    """Put in force the routes of GMLX_IMAGE_KQ_ROUTES, or of ``routes``,
    that the installed mlx-kquant has an op for, and return their names.
    The ``qk_norm_rope`` seam of ``gmlx.image.ops`` is replaced by
    assignment, or set back to the stock function when its route is not in
    force. Runs at import of ``gmlx.image.ops``. Raises while another
    function holds the seam, as under an installed ``StepProfiler``, and
    for a ``routes`` that is a str or holds an unknown name, before any
    change."""
    global _active
    ns = sys.modules[__package__]
    held = ns.qk_norm_rope
    if held is not qk_norm_rope and held is not attention.qk_norm_rope:
        raise RuntimeError(
            "gmlx.image.ops.qk_norm_rope is replaced, as by an installed "
            "StepProfiler: remove the replacement before changing routes")
    want = _WANTED if routes is None else _named(routes)
    on = set()
    if "qk" in want and _kq_op("qk_norm_rope") is not None:
        on.add("qk")
    if "glu" in want and _kq_op("quantized_matmul_glu") is not None:
        on.add("glu")
    if "lora" in want and _kq_op("quantized_matmul_lora") is not None:
        on.add("lora")
    ns.qk_norm_rope = qk_norm_rope if "qk" in on else attention.qk_norm_rope
    _active = frozenset(on)
    return _active


@contextmanager
def only(routes):
    """Put ``routes`` in force for the block, and after it the routes that
    were in force before. Yields the names in force inside the block. The
    switch is not read again. A graph built in the block keeps the ops it
    was built with.

    ``routes`` is a tuple of names of ROUTES, or () for none. A str, None
    and an unknown name raise before any change. The routes in force are
    state of the process, not of a thread or a pipeline: a forward on
    another thread runs on the block's routes while the block is open.
    The block must not be suspended. A generator that yields inside it
    leaves its routes in force until the generator is resumed or closed,
    and its exit then puts back the routes of before the block over any
    set installed since. The exit puts the earlier routes back first and
    raises after when a function that replaced the seam inside the block
    still holds it, as a ``StepProfiler`` installed in the block does.
    The replacement is dropped."""
    want = _named(routes)
    before = _active
    on = install(want)
    try:
        yield on
    finally:
        ns = sys.modules[__package__]
        held = ns.qk_norm_rope
        replaced = held is not qk_norm_rope \
            and held is not attention.qk_norm_rope
        if replaced:
            ns.qk_norm_rope = attention.qk_norm_rope
        install(before)
        if replaced:
            raise RuntimeError(
                "gmlx.image.ops.qk_norm_rope was replaced inside the block, "
                "as by a StepProfiler installed in it: the replacement is "
                "dropped and the routes of before the block are in force")


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
            and _gpu()):
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
    if not _gpu() or not _nax():
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
    matmul product and the sigmoid. No gradient with respect to the
    weights."""
    return _kq_op("quantized_matmul_glu")(x, w_gate, w_up, codec)


def lora_static(base, dtype, ranks: int, bands) -> bool:
    """Whether ``lora_leaf`` can take a call of ``base`` on an x in
    ``dtype`` with a low-rank term of ``ranks`` ranks and the band table
    ``bands``, at some row count: a plain KQuantLinear with no bias whose
    codec, dims, ranks and bands are on the route's table. A leaf that
    StepProfiler timed counts as its own class."""
    from mlx_kquant.codec_geometry import in_features
    from mlx_kquant.nn import KQuantLinear

    cls = type(base)
    if getattr(cls, "_prof_base", cls) is not KQuantLinear or "bias" in base:
        return False
    codec = base.kquant_type
    if (codec, dtype) not in _LORA_ROWS:
        return False
    n, packed = base["weight"].shape
    if in_features(codec, int(packed)) > _LORA_MAX_K or n < _LORA_MIN_N:
        return False
    most = ranks
    if bands is not None:
        if not 0 < len(bands) <= _LORA_MAX_BANDS:
            return False
        if any(c0 % 64 or (c1 % 32 and c1 != n) for c0, c1, _, _ in bands):
            return False
        most = max(r for _, _, _, r in bands)
    return 0 < most <= _LORA_MAX_RANKS


def lora_leaf(base, x: mx.array):
    """``base`` when ``lora_fused`` takes its product on ``x`` with a
    low-rank term, else None, and the caller then runs the base product
    and adds the term on stock ops. ``base`` is a leaf that
    ``lora_static`` took with its term. The route takes an x of at least
    1024 rows over its leading dims, on a GPU with tensor ops."""
    if "lora" not in _active or x.ndim < 2:
        return None
    k = x.shape[-1]
    if k == 0 or x.size // k < _LORA_MIN_ROWS:
        return None
    if not _gpu() or not _nax():
        return None
    return base


def lora_fused(x: mx.array, base, a_t: mx.array, b_t: mx.array,
               strength: mx.array, bands) -> mx.array:
    """``x @ W^T + (strength * (x @ a_t)) @ b_t`` over the wire bytes of
    the K-quant leaf ``base``, for a leaf that ``lora_leaf`` returned.
    ``a_t`` is (in, rank) and ``b_t`` (rank, out), ``strength`` holds one
    factor per rank, and ``bands`` is None or rows of (first column, end
    column, first rank, ranks): the columns of a band take that block of
    ranks alone. The kernel adds both products in fp32 and rounds once.
    The stock ops round the base product before the add. No gradient
    with respect to the weight."""
    return _kq_op("quantized_matmul_lora")(
        x, base["weight"], base.kquant_type, a_t, b_t, strength=strength,
        bands=bands)
