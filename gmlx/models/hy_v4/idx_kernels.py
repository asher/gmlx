# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""mlx-kquant route for the HY4 DSA indexer key selection.

The ops path materializes the [B, 32, L, S] per-head score tensor before
the head reduction: 2.1 GB at a 4096-token chunk against 4096 keys, on a
model that is already streaming its experts through what memory is left.
``kq.dsa_indexer_scores`` reduces the heads inside a steel GEMM and emits
[B, 1, L, S]; ``kq.dsa_topk_indices`` runs a radix select over it.

The selected set is not identical to ``argpartition``'s. Both admit ties at
the threshold in scan order, and the kernel reduces the heads at the score
dtype where the ops path promotes to the fp32 head weights, so the two can
part on keys that sit on the boundary. ``GMLX_HY4_IDX_KERNEL=0`` restores
the ops path.
"""

import os

import mlx.core as mx

try:
    import mlx_kquant as _kq

    _HAVE = hasattr(_kq, "dsa_indexer_scores") and hasattr(
        _kq, "dsa_topk_indices")
except ImportError:  # pragma: no cover - kq is a hard dependency
    _kq = None
    _HAVE = False

# dsa_topk_indices carries a radix select for these widths only.
_TOPK = (512, 2048)
_HEADS = (32, 64)
_TILE = 64


def _enabled() -> bool:
    return os.environ.get("GMLX_HY4_IDX_KERNEL", "1") != "0"


def select(q, k, weights, topk, offset, mask):
    """Top-``topk`` key indices per query row, or None to use the ops path.

    ``q`` [B, H, L, 128], ``k`` [B, 1, S, 128], ``weights`` [B, L, H].
    ``mask`` is HY4's plain causal array; the kernels rebuild it from
    ``offset``, so a mask of any other shape declines the route.
    """
    if not (_HAVE and _enabled() and mx.default_device() == mx.gpu):
        return None
    B, H, L, hd = q.shape
    S = k.shape[2]
    if (topk not in _TOPK or H not in _HEADS or hd != 128
            or q.dtype not in (mx.float16, mx.bfloat16)
            or S % _TILE or S < topk):
        return None
    if mask is not None and mask.shape[-2:] != (L, S):
        return None

    w = weights.astype(q.dtype)
    if L == 1:
        if H != 64:
            pad = mx.zeros((B, 64 - H, 1, hd), q.dtype)
            q = mx.concatenate([q, pad], axis=1)
            w = mx.concatenate([w, mx.zeros((B, 1, 64 - H), w.dtype)],
                               axis=-1)
        scores = _kq.dsa_indexer_score_decode(
            q, k.reshape(B, S, hd), w, offset, 1)
        return _kq.dsa_topk_indices(scores, topk)

    if L % _TILE:
        return None
    scores = _kq.dsa_indexer_scores(q, k, w, True, causal_q_offset=offset)
    return _kq.dsa_topk_indices(scores, topk, causal_valid_prefix=True)
