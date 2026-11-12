"""Row ends: the native head batch's uneven rollback without a roll.

After a speculative round each row keeps its own count of the verified
tokens. An armed BatchKVCache lowers each row's end instead of rolling the
shorter rows right (a gather of every row's whole KV), and its next append
writes each row at its own end. The rejected tail of a row stays in the
buffer past the row's end, hidden by it, until the row's next append writes
over it. Attention of an uneven cache reads each row's start and end off
the cache. Attention is permutation invariant over keys and the rows never
move, so the left padding, the shared prefix stamp and the stores keep their
meaning.

Armed only by the batch rounds of the native Qwen head on a dense qwen3_5
target (``row_ends_active``); a trip to plain decode right-justifies the
rows once and disarms. GMLX_ROW_ENDS=0 keeps the roll.
"""

from __future__ import annotations

import mlx.core as mx

from gmlx.envflags import env_bool

_KQ_HEAD_DIMS = (64, 128, 256, 512)


def row_ends_active(drafter, lm, mixed) -> bool:
    """Whether a batch of this drafter and target keeps row ends."""
    if mixed or not env_bool("GMLX_ROW_ENDS", True):
        return False
    if not (getattr(lm, "row_ends_rollback", False)
            and getattr(drafter, "row_ends_rollback", False)):
        return False
    from gmlx.cache.kvarn_sdpa import kvarn_row_ends_ok

    return kvarn_row_ends_ok()


def _plain_batch_caches(caches):
    from gmlx.cache.kv_cache import BatchKVCache

    for c in caches or ():
        members = getattr(c, "caches", None)
        if isinstance(members, (tuple, list)):
            yield from _plain_batch_caches(members)
        elif type(c) is BatchKVCache:
            yield c


def arm(caches) -> None:
    """Arm the fp16 batch caches of a stack: their rollback keeps row ends."""
    for c in _plain_batch_caches(caches):
        c._row_ends_armed = True


def disarm(caches) -> None:
    """Right-justify and disarm the batch caches of a stack, one layer's
    roll evaluated at a time."""
    from gmlx.cache.kv_append import eval_cache

    for c in _plain_batch_caches(caches):
        if c._row_ends is not None:
            c.justify_rows()
            eval_cache(c)
        c.__dict__.pop("_row_ends_armed", None)  # back to the class default
        c.__dict__.pop("_row_ends_mx", None)


def _ends_array(cache, ends) -> mx.array:
    memo = cache.__dict__.get("_row_ends_mx")
    key = tuple(ends)
    if memo is None or memo[0] != key:
        memo = (key, mx.array(key, dtype=mx.int32))
        cache._row_ends_mx = memo
    return memo[1]


def _kq_route_ok(queries, keys) -> bool:
    return (
        not isinstance(keys, (tuple, list))
        and mx.default_device() == mx.gpu
        and queries.ndim == 4
        and 1 <= queries.shape[2] <= 4
        and queries.shape[-1] in _KQ_HEAD_DIMS
        and queries.dtype in (mx.float16, mx.bfloat16)
        and keys.dtype == queries.dtype
    )


_MAX_QL = 8


def row_ends_attention(queries, keys, values, cache, scale):
    """Attention for a cache with uneven row ends, else None.

    Row b attends keys ``[left_padding[b], end_b)``, its query block causal
    at its own end. ``keys`` and ``values`` are what the cache's append
    returned, up to the largest end. One kq call reads every row in place
    at qL 1 to 4; other shapes take one fused causal call per row on the
    row's own keys (the per-row verify of gmlx/models/qwen35/attn.py)."""
    ends = getattr(cache, "_row_ends", None)
    if ends is None:
        return None
    qL = queries.shape[2]
    if qL > _MAX_QL:
        # The rounds append at most a verify block to an uneven cache; a
        # wider forward would read rows the route was not built for.
        raise ValueError(
            f"a {qL}-token forward reached a batch whose rows end apart; "
            f"row ends serve at most {_MAX_QL}")
    if _kq_route_ok(queries, keys):
        import mlx_kquant as kq

        return kq.sdpa_decode_gqa(
            queries, keys, values, scale,
            starts=cache.left_padding.astype(mx.int32),
            ends=_ends_array(cache, ends))
    from gmlx.models.qwen35.attn import _per_row_verify

    spans = list(zip(cache.left_padding.tolist(), ends))
    return _per_row_verify(queries, keys, values, spans, cache=cache, scale=scale)
