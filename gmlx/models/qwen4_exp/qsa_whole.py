"""One record for a whole attention layer of a verify.

The split form (model._fused_qsa_verify) records the two ends of the layer
and builds the middle in Python on every call, because the rope offset,
the cache append offsets, the block count and the base length change with
depth. Here those values are slots of a parameter block (an int32 array
that is an input of the record, see mlx_kquant.fuse_graph), so one record
serves every depth at which the cache keeps its layout:

  slot 0  cache offset before the append (rope, queries, scores, ends)
  slot 1  rows live in the decode tail before the append
  slot 2  rows in the base (offset - tail rows)
  slot 3  block keys live in the block tail
  slot 4  finished blocks before the call

What stays in the key: the row count, the dtypes, the shapes of the base
segments, and the number of block keys in the base. A fold of the tail, a
grown base and a rebuilt block base make a new layout.

Bits are those of Attention._attend: same kernels on the same operands.
The block key tail is kept at a fixed capacity; rows past the live count
are not visible to the scorer (visibility follows the offset), and the
top-k over the longer row selects the same blocks in the same order.
"""

from __future__ import annotations

import mlx.core as mx

P_OFFSET, P_TAIL, P_BASE, P_BTAIL, P_BLOCKS = range(5)
# One row only: the offset less one, the scorer's query offset there.
P_QOFF = 5
# Blocks one call finishes at most (8 rows at ratio 4), and the rows of
# the block key tail past the live count a call may write.
MAX_NEW = 2

_KERNELS: dict = {}


def _kernels():
    """The parameter forms of the indexer norm and rope kernel: block
    finish reading the raw keys from the base and the tail, and queries
    at an offset from the block."""
    if "finish" not in _KERNELS:
        from . import model as M

        src = M._norm_rope_source()
        head = ("    const int m = meta[0];\n"
                "    const int start = meta[1];\n")
        assert src.count(head) == 1
        q_src = src.replace(
            head, "    const int m = MQ;\n    const int start = meta[SLOT];\n")
        _KERNELS["queries"] = mx.fast.metal_kernel(
            name="q4_norm_rope_pq", input_names=["raw", "w", "meta", "fp"],
            output_names=["out"], source=q_src, ensure_row_contiguous=True)
        loop = ("          for (int rr = 0; rr < R; rr++) {\n"
                "            s = s + static_cast<float>(src[rr * D + e]);\n"
                "          }\n")
        assert src.count(loop) == 1
        f_src = src.replace(
            head,
            "    const int m = MQ;\n    const int start = meta[SLOT];\n"
            "    const int bl = meta[SLOT_BL];\n"
            "    const int tcap = rawt_shape[1];\n").replace(
            loop,
            "          for (int rr = 0; rr < R; rr++) {\n"
            "            const int ra = ((int)(row % m) + start) * R + rr;\n"
            "            const int rt = ra - bl < tcap ? ra - bl : tcap - 1;\n"
            "            const T_IN xr = ra < bl\n"
            "                ? raw[(size_t)ra * D + e]\n"
            "                : rawt[(size_t)rt * D + e];\n"
            "            s = s + static_cast<float>(xr);\n"
            "          }\n")
        _KERNELS["finish"] = mx.fast.metal_kernel(
            name="q4_norm_rope_pf",
            input_names=["raw", "rawt", "w", "meta", "fp"],
            output_names=["out"], source=f_src, ensure_row_contiguous=True)
    return _KERNELS


def finish_blocks_p(idx, ib, it, p):
    """The MAX_NEW block keys from block p[P_BLOCKS] on, as
    QSAIndexer._fused_finish computes them, the raw keys read from the
    base ``ib`` below row p[P_BASE] and from the tail ``it`` above."""
    D = ib.shape[-1]
    w = idx.k_norm.weight
    out_t = mx.float32 if mx.float32 in (ib.dtype, w.dtype) else ib.dtype
    tg = 32 * (((D + 3) // 4 + 31) // 32)
    return _kernels()["finish"](
        inputs=[ib, it, w.astype(out_t), p, idx._finish_fp],
        template=[("T_IN", ib.dtype), ("T", out_t), ("T_OUT", out_t),
                  ("D", D), ("DIMS", idx.rotary_dims), ("R", idx.ratio),
                  ("NH", 0), ("MQ", MAX_NEW), ("SLOT", P_BLOCKS),
                  ("SLOT_BL", P_BASE)],
        grid=(tg * MAX_NEW, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(1, MAX_NEW, D)],
        output_dtypes=[out_t],
    )[0]


def queries_p(idx, x, p, dtype):
    """QSAIndexer._fused_queries with the offset read from p[P_OFFSET]."""
    from . import model as M

    B, L, _ = x.shape
    D, H = idx.head_dim, idx.n_heads
    w = idx.q_norm.weight
    q = M._indexer_proj(idx.q_proj, x)
    t = mx.float32 if mx.float32 in (q.dtype, w.dtype) else q.dtype
    tg = 32 * (((D + 3) // 4 + 31) // 32)
    return _kernels()["queries"](
        inputs=[q, w.astype(t), p, idx._query_fp],
        template=[("T_IN", q.dtype), ("T", t), ("T_OUT", dtype), ("D", D),
                  ("DIMS", idx.rotary_dims), ("R", 1), ("NH", H),
                  ("MQ", L), ("SLOT", P_OFFSET)],
        grid=(tg * B * L * H, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[dtype],
    )[0]


def block_tail_cap(cache) -> int:
    """Rows of the fixed-capacity block key tail."""
    return cache._tail_rows() // cache.ratio + MAX_NEW


def state(attn, cache, L: int):
    """The parameter values of a call of ``L`` rows on ``cache``, or None
    when the whole-layer record does not cover the call: the plain branch
    it mirrors is the paged rows attention over a base and a tail, with
    the fused scorer, every query sparse and the new block keys in the
    block tail."""
    st = _state(attn, cache, L)
    return None if isinstance(st, str) else st


def _state(attn, cache, L: int):
    """``state``, or a word for why not (the stats of verify_bound)."""
    idx = attn.indexer
    r, topk = attn.ratio, idx.block_topk
    off, tl = cache.offset, cache._tl
    kb, kt = cache._kb, cache._kt
    cap = cache._tail_rows()
    if kb is None or kt is None or cache._ib is None:
        return "no tail"
    if (cache.pos is not None or not 1 <= L <= 8
            or L > cap or kt.shape[0] != 1 or kt.dtype != kb.dtype):
        return "form"
    if tl + L > cap:
        return "tail full"
    if cache.blocks is None:
        return "no blocks"
    if (off + 1) // r <= topk or topk != 512:
        return "dense"
    nbb, nb_old = cache._nbb, cache.n_blocks
    nb_new = (off + L) // r
    nt = nb_old - nbb
    m = nb_new - nb_old
    if nbb < 1 or nbb != cache.blocks.shape[1]:
        return "block base"
    if nt < 0 or m < 0 or m > MAX_NEW or nt + m > cap // r:
        return "block tail"
    if cache._bt is None and nt:
        return "no block tail"
    if L == 1:
        return (off, tl, off - tl, nt, nb_old, off - 1), nb_new
    return (off, tl, off - tl, nt, nb_old), nb_new


def ranges(attn, cache, L: int):
    """The values each slot may take under one record."""
    r, topk = attn.ratio, attn.indexer.block_topk
    cap = cache._tail_rows()
    big = (1 << 30) - 1
    out = [(P_OFFSET, r * (topk + 1) - 1, big), (P_TAIL, 0, cap - L),
           (P_BASE, 0, int(cache._kb.shape[2])), (P_BTAIL, 0, cap // r),
           (P_BLOCKS, 0, big)]
    if L == 1:
        out.append((P_QOFF, r * (topk + 1) - 2, big))
    return out


def _mark(cache) -> None:
    cache._w_mark = (id(cache), id(cache._kt), id(cache._vt),
                     id(cache._it), id(cache._bt))


def owned(cache) -> bool:
    """The tails are this cache's alone (the mark holds): own() would copy
    nothing."""
    return cache._w_mark == (id(cache), id(cache._kt), id(cache._vt),
                             id(cache._it), id(cache._bt))


def own(cache) -> None:
    """Make the tails the record appends into this cache's alone. The
    record writes rows in place; the plain call writes a new array. A
    copy of the cache (a prompt cache entry and its deep copy) shares the
    arrays, and after a trim the two disagree about which rows are live.
    The mark names this cache object and the tail arrays the record last
    left in it. A copy of the cache or a restored state changes one of
    them and a trim (the rollback of a verify included) clears the mark;
    the tails are then copied, inside the graph, before the next in-place
    write."""
    cur = (id(cache), id(cache._kt), id(cache._vt), id(cache._it),
           id(cache._bt))
    if cache._w_mark == cur:
        return

    def copy(a):
        return None if a is None else mx.concatenate(
            [a[:, :1], a[:, 1:]], axis=1)

    cache._kt, cache._vt = copy(cache._kt), copy(cache._vt)
    cache._it, cache._bt = copy(cache._it), copy(cache._bt)
    _mark(cache)


def ensure_block_tail(cache, nt: int) -> None:
    """Give the cache its block key tail at the fixed capacity (the live
    rows kept, zeros after them)."""
    cap = block_tail_cap(cache)
    bt = cache._bt
    if bt is not None and bt.shape[1] == cap:
        return
    blocks = cache.blocks
    shape = (blocks.shape[0], cap - nt, blocks.shape[2])
    parts = [mx.zeros(shape, blocks.dtype)]
    if nt:
        parts.insert(0, bt[:, :nt])
    cache._bt = mx.concatenate(parts, axis=1)
    cache._bt_fixed = True


def graph(layer, h, cache, p, vals):
    """The layer on ``h`` with its changing values read from ``p``:
    (layer output, key tail, value tail, indexer key tail, block key
    tail), or None where a kernel of the plain branch does not apply."""
    import mlx_kquant as kq

    from . import model as M

    attn = layer.self_attn
    idx = attn.indexer
    off, tl, bl, nt = vals[:4]
    B, L = h.shape[0], h.shape[1]
    r = attn.ratio
    mixed, inject = layer.hc_attn(h)
    q, gate, k, v, ik = attn._project(mixed)
    half = (mx.float16, mx.bfloat16)
    float_types = (mx.float32,) + half
    w_k, w_q = idx.k_norm.weight, idx.q_norm.weight
    D = idx.head_dim
    blocks = cache.blocks
    sdt = blocks.dtype if blocks.dtype in half else (
        mixed.dtype if mixed.dtype in half else None)
    ib, it0 = cache._ib, cache._it
    if (attn._rope_scaled is not None or idx._rope_scaled is not None
            or sdt is None or M._kq_topk() is None
            or not M._kq_score_fits(idx.n_heads, D)
            or not M._kq_score_wide() or M._norm_rope() is None
            or D % 4 or idx.rotary_dims % 2 or idx.rotary_dims > D
            or mixed.dtype not in float_types or w_q.dtype not in float_types
            or w_k.dtype not in float_types or ib.dtype not in float_types
            or it0.dtype != ib.dtype or ik is None
            or (L > 1
                and attn._rows_kernel(q, None, None, cache, True, L) is None)
            or (L == 1 and not _one_row_ok(attn, q, cache))):
        return None
    # The float constants the plain kernels keep per indexer.
    if getattr(idx, "_finish_fp", None) is None:
        idx._finish_fp = mx.array(
            [idx.k_norm.eps, float(r), M._log2f(idx.rope_theta), 1.0 / r],
            dtype=mx.float32)
    if getattr(idx, "_query_fp", None) is None:
        idx._query_fp = mx.array(
            [idx.q_norm.eps, 1.0, M._log2f(idx.rope_theta), 1.0],
            dtype=mx.float32)
    # a slice and a squeeze: int indexing is a Gather, not replayable
    p_off = p[P_OFFSET:P_OFFSET + 1].squeeze()
    q = mx.fast.rope(q, attn.rotary_dims, traditional=False,
                     base=attn.rope_theta, scale=1.0, offset=p_off)
    k = mx.fast.rope(k, attn.rotary_dims, traditional=False,
                     base=attn.rope_theta, scale=1.0, offset=p_off)
    # The cache append, in place at the tail offset.
    kt = kq.kv_append(cache._kt, k.astype(cache._kt.dtype), tl,
                      params=p, offset_slot=P_TAIL)
    vt = kq.kv_append(cache._vt, v.astype(cache._vt.dtype), tl,
                      params=p, offset_slot=P_TAIL)
    it = kq.kv_append(
        it0.reshape(1, 1, *it0.shape[1:]),
        ik.astype(it0.dtype).reshape(1, 1, L, -1), tl,
        params=p, offset_slot=P_TAIL).reshape(it0.shape)
    # The block keys the call completes, into the block tail.
    new = finish_blocks_p(idx, ib, it, p)
    bt0 = cache._bt
    bt = kq.kv_append(
        bt0.reshape(1, 1, *bt0.shape[1:]),
        new.astype(bt0.dtype).reshape(1, 1, MAX_NEW, -1), nt,
        params=p, offset_slot=P_BTAIL).reshape(bt0.shape)
    # Selection: fused queries, fused scores over the base and the tail,
    # radix top-k.
    qi = queries_p(idx, mixed, p, sdt)
    if L == 1:
        core = _one_row(attn, idx, q, qi, blocks.astype(sdt), bt.astype(sdt),
                        cache, kt, vt, p, off, sdt)
    else:
        w = M._kq_score_weights(B, L, idx.n_heads, D, sdt)
        s16 = kq.dsa_indexer_score_decode(
            qi, blocks.astype(sdt), w, off, r, keys_tail=bt.astype(sdt),
            params=p, q_offset_slot=P_OFFSET)
        sel = M._kq_topk()(s16, idx.block_topk, True)[:, 0].view(mx.int32)
        ends = mx.broadcast_to(
            (p[P_OFFSET:P_OFFSET + 1] + _steps(L))[None], (B, L))
        core = M._kq_paged_rows()(
            q, cache._kb, cache._vb, attn.scale, sel, ends, kt, vt, bl,
            page_rows=r, tail_page=True, params=p, base_len_slot=P_BASE)
    out = core.transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = attn.o_proj(out * mx.sigmoid(gate))
    h = M._hc_combine(h, out, inject, kern=True)
    mixed, inj = layer.hc_ffn(h)
    out = M._hc_combine(h, layer.mlp(mixed), inj, kern=True)
    return out, kt, vt, it, bt


_STEPS: dict = {}


def _steps(L: int) -> mx.array:
    """1..L as an evaluated constant: an Arange node inside the record
    would keep it from a direct replay."""
    a = _STEPS.get(L)
    if a is None:
        a = _STEPS[L] = 1 + mx.arange(L, dtype=mx.int32)
        mx.eval(a)
    return a


def _one_row_ok(attn, q, cache) -> bool:
    """The plain branch of one row is the paged decode kernel over a
    two-segment gather (Attention._segmented_attention)."""
    from . import model as M

    kb, vb, kt, vt = cache._kb, cache._vb, cache._kt, cache._vt
    return (attn.ratio == 4 and attn.head_dim == 256
            and q.dtype in (mx.bfloat16, mx.float16)
            and M._kq_paged() is not None and M._kv_gather() is not None
            and kb.shape[3] % 16 == 0 and vb.shape == kb.shape
            and kt.dtype == kb.dtype and vt.dtype == vb.dtype)


def _arange(n: int, add: int = 0) -> mx.array:
    a = _STEPS.get(("a", n, add))
    if a is None:
        a = _STEPS[("a", n, add)] = mx.arange(n, dtype=mx.int32) + add
        mx.eval(a)
    return a


def _one_row(attn, idx, q, qi, blocks, bt, cache, kt, vt, p, off, sdt):
    """Selection and attention of a one-row call. The scorer shows every
    key row to a single query, so it runs on the query twice with its
    offset one lower: row 1 of that call sees the blocks the single query
    may see and no row of the block tail past them."""
    import mlx_kquant as kq

    from . import model as M

    B = q.shape[0]
    r = attn.ratio
    Hkv, D = cache._kb.shape[1], cache._kb.shape[3]
    w = M._kq_score_weights(B, 2, idx.n_heads, idx.head_dim, sdt)
    s16 = kq.dsa_indexer_score_decode(
        mx.concatenate([qi, qi], axis=2), blocks, w, off - 1, r,
        keys_tail=bt, params=p, q_offset_slot=P_QOFF)[:, :, 1:2]
    sel = M._kq_topk()(s16, idx.block_topk, True)[:, 0].view(mx.int32)
    rows = (sel[:, 0, :, None] * r + _arange(r)).reshape(B, -1)
    rem = (off + 1) % r
    if rem:
        tail = p[P_OFFSET:P_OFFSET + 1] + _arange(rem, 1 - rem)
        rows = mx.concatenate([rows, tail[None]], axis=-1)
    n = rows.shape[1]
    total = B * Hkv * n * (D // 16)
    k_sel, v_sel = M._kv_gather()(
        inputs=[cache._kb, kt, cache._vb, vt, rows, p[P_BASE:P_BASE + 1]],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[(B, Hkv, n, D), (B, Hkv, n, D)],
        output_dtypes=[cache._kb.dtype, cache._vb.dtype])
    n_pages = -(-n // r)
    pages = mx.broadcast_to(_arange(n_pages)[None, None], (B, Hkv, n_pages))
    return M._kq_paged()(q, k_sel, v_sel, attn.scale, pages, tile_c=4)


def commit(cache, outs, L: int, nb_new: int):
    """Leave the cache as Attention._attend leaves it."""
    out, kt, vt, it, bt = outs
    cache._kt, cache._vt, cache._it, cache._bt = kt, vt, it, bt
    _mark(cache)
    cache._tl += L
    cache.offset += L
    cache.n_blocks = nb_new
    return out


def params(vals):
    return mx.array(list(vals), dtype=mx.int32)


# Dense depth: every block still fits the top-k, the plain call runs
# mx.fast.scaled_dot_product_attention over all keys with a causal mask.
# Its key count is a shape. The record keeps the keys at a fixed count
# (base rows, then tail rows, copied by a kernel that reads the base length
# from the parameter block, up to a cap) and hides the rows past each
# query's end with a bool mask built from the parameter block. The mlx
# kernels walk the keys in strides that do not depend on the count and
# skip a hidden key, so the bits are those of the exact count as long as
# the cap and the count pick the same kernel: mlx switches at 1024 keys,
# so the cap is 1023 and a call that reaches 1024 keys takes the split
# form (over 1024 keys a capped form differs from the exact count in
# rounding on Qwen3.8-Flash-Next). Above 5 rows at 6 query heads a key
# head mlx builds the unfused graph, whose sums follow the key count: not
# covered.

_DENSE_CUT = 1024
_CONSTS: dict = {}


def _const(kind: str, n: int) -> mx.array:
    a = _CONSTS.get((kind, n))
    if a is None:
        a = mx.arange(n, dtype=mx.int32)
        a = a.reshape(n, 1) if kind == "rows" else a.reshape(1, n)
        mx.eval(a)
        _CONSTS[(kind, n)] = a
    return a


def _kv_cat():
    """K and V of the first CAP rows as one array each: base rows below
    meta[SLOT], tail rows after them (rows past the tail read its last
    row; the mask hides them). Plain copies."""
    k = _KERNELS.get("cat")
    if k is None:
        k = _KERNELS["cat"] = mx.fast.metal_kernel(
            name="q4_kv_cat_p",
            input_names=["kb", "kt", "vb", "vt", "meta"],
            output_names=["ko", "vo"],
            source="""
    const uint g = thread_position_in_grid.x;
    constexpr int DC = D / 16;
    const int c = g % DC;
    const int r = (g / DC) % CAP;
    const int hh = g / (DC * CAP);
    const int bl = meta[SLOT];
    const int nb = kb_shape[2];
    const int nt = kt_shape[2];
    const size_t o = ((size_t)hh * CAP + r) * D + c * 16;
    if (r < bl) {
      const size_t s = ((size_t)hh * nb + r) * D + c * 16;
      for (int e = 0; e < 16; e++) {
        ko[o + e] = kb[s + e];
        vo[o + e] = vb[s + e];
      }
    } else {
      const int t = r - bl < nt ? r - bl : nt - 1;
      const size_t s = ((size_t)hh * nt + t) * D + c * 16;
      for (int e = 0; e < 16; e++) {
        ko[o + e] = kt[s + e];
        vo[o + e] = vt[s + e];
      }
    }
""", ensure_row_contiguous=True)
    return k


def dense_state(attn, cache, L: int):
    """((offset, tail rows, base rows, 0, 0), key cap, (offset lo, hi))
    of a dense call of ``L`` rows, or None where the dense record does
    not cover it."""
    r, topk = attn.ratio, attn.indexer.block_topk
    off, tl = cache.offset, cache._tl
    kb, kt, vb, vt = cache._kb, cache._kt, cache._vb, cache._vt
    tcap = cache._tail_rows()
    H, Hkv, D = (attn.num_attention_heads, attn.num_key_value_heads,
                 attn.head_dim)
    if (kb is None or kt is None or cache._ib is None or cache._it is None
            or cache.pos is not None or not 1 <= L <= 8
            or kt.shape[0] != 1 or kt.dtype != kb.dtype
            or vt.dtype != vb.dtype or vb.shape != kb.shape
            or vt.shape != kt.shape or kt.shape[2] != tcap
            or tl + L > tcap or D not in (64, 96, 128, 256)
            or L * (H // Hkv) > 32 or H % Hkv):
        return None
    n = off + L
    bl = off - tl
    top = r * (topk + 1) - 1          # the last count select() leaves dense
    if n > top or n >= _DENSE_CUT or bl < 0 or bl > kb.shape[2]:
        return None
    cap = _DENSE_CUT - 1
    if not L <= n <= cap:
        return None
    return (off, tl, bl, 0, 0), cap, (0, cap - L)


def dense_ranges(cache, L: int, span):
    return [(P_OFFSET, span[0], span[1]),
            (P_TAIL, 0, cache._tail_rows() - L),
            (P_BASE, 0, int(cache._kb.shape[2]))]


def dense_graph(layer, h, cache, p, vals, cap: int):
    """The layer at dense depth with its offsets read from ``p``: (layer
    output, key tail, value tail, indexer key tail)."""
    import mlx_kquant as kq

    from . import model as M

    attn = layer.self_attn
    _, tl, _, _, _ = vals
    L = h.shape[1]
    B = h.shape[0]
    mixed, inject = layer.hc_attn(h)
    q, gate, k, v, ik = attn._project(mixed)
    it0 = cache._it
    if attn._rope_scaled is not None or ik is None:
        return None
    p_off = p[P_OFFSET:P_OFFSET + 1].squeeze()
    q = mx.fast.rope(q, attn.rotary_dims, traditional=False,
                     base=attn.rope_theta, scale=1.0, offset=p_off)
    k = mx.fast.rope(k, attn.rotary_dims, traditional=False,
                     base=attn.rope_theta, scale=1.0, offset=p_off)
    kt = kq.kv_append(cache._kt, k.astype(cache._kt.dtype), tl,
                      params=p, offset_slot=P_TAIL)
    vt = kq.kv_append(cache._vt, v.astype(cache._vt.dtype), tl,
                      params=p, offset_slot=P_TAIL)
    it = kq.kv_append(
        it0.reshape(1, 1, *it0.shape[1:]),
        ik.astype(it0.dtype).reshape(1, 1, L, -1), tl,
        params=p, offset_slot=P_TAIL).reshape(it0.shape)
    kb, vb = cache._kb, cache._vb
    Hkv, D = kb.shape[1], kb.shape[3]
    keys, vals_ = _kv_cat()(
        inputs=[kb, kt, vb, vt, p],
        template=[("D", D), ("CAP", cap), ("SLOT", P_BASE)],
        grid=(Hkv * cap * (D // 16), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, Hkv, cap, D), (1, Hkv, cap, D)],
        output_dtypes=[kb.dtype, vb.dtype])
    # key i is visible to query row j when i <= offset + j
    ends = p[P_OFFSET:P_OFFSET + 1].reshape(1, 1) + _const("rows", L)
    mask = _const("keys", cap) <= ends
    core = mx.fast.scaled_dot_product_attention(
        q, keys, vals_, scale=attn.scale, mask=mask)
    out = core.transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = attn.o_proj(out * mx.sigmoid(gate))
    h = M._hc_combine(h, out, inject, kern=True)
    mixed, inj = layer.hc_ffn(h)
    out = M._hc_combine(h, layer.mlp(mixed), inj, kern=True)
    return out, kt, vt, it


def dense_commit(cache, outs, L: int):
    """Leave the cache as Attention._attend leaves it at dense depth."""
    out, kt, vt, it = outs
    cache._kt, cache._vt, cache._it = kt, vt, it
    _mark(cache)
    cache._tl += L
    cache.offset += L
    return out
