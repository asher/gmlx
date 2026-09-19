"""Attention for training with bounded memory.

Under gradient tracing, MLX runs ``mx.fast.scaled_dot_product_attention``
through its unfused ops for both the forward and the backward, and the
backward needs the full ``[B, H, T, T]`` softmax of every layer, which
stays on the gradient tape until the step evaluates. ``blocked_attention``
computes the same attention over blocks of query rows inside an
``mx.custom_function``: the forward keeps the output and one float32
log-sum-exp per row, and the backward recomputes each block's
probabilities from those. The largest transient is one ``[B, H, block, T]``
block and nothing quadratic in T stays on the tape.

Numerics follow the unfused path: scores in the input dtype, softmax in
float32, probabilities cast back to the input dtype before the value
matmul. Grouped-query heads are handled without tiling keys or values.
Masks: ``None``, ``"causal"`` (bottom-right aligned when the key length
exceeds the query length), a boolean array or an additive array whose
last two axes are ``[T_q, T_k]`` with leading axes that broadcast.
Attention sinks are not supported here and fall through to MLX.
"""
from __future__ import annotations

import mlx.core as mx

_BLOCK = 256


def _prep_mask(mask, Tq: int, Tk: int):
    """Return (causal, mask_array) with the array's last two axes [Tq, Tk]."""
    if mask is None:
        return False, None
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"blocked_attention: unsupported mask {mask!r}")
        return True, None
    if mask.shape[-2] != Tq or mask.shape[-1] != Tk:
        raise ValueError(
            f"blocked_attention: mask shape {mask.shape} does not end in [{Tq}, {Tk}]")
    return False, mask


def _scores(qb, kb, scale: float, causal: bool, i0: int, i1: int, kend: int,
            offset: int, mask, dtype):
    """Masked float32 scores of one query block against keys [0, kend)."""
    s = mx.matmul(qb, kb.swapaxes(-1, -2)).astype(mx.float32)
    s = s * scale
    low = mx.finfo(mx.float32).min
    if causal and kend > i0 + offset + 1:
        rows = mx.arange(i0, i1)[:, None] + offset
        cols = mx.arange(kend)[None, :]
        s = mx.where(cols <= rows, s, low)
    if mask is not None:
        mb = mask[..., i0:i1, :kend]
        if mask.ndim == 4 and qb.ndim == 5:
            # [B, Hq or 1, bq, kend] -> [B, Hkv or 1, rep or 1, bq, kend]
            if mb.shape[1] == 1:
                mb = mb[:, :, None]
            else:
                mb = mb.reshape(mb.shape[0], qb.shape[1], qb.shape[2], *mb.shape[2:])
        if mb.dtype == mx.bool_:
            s = mx.where(mb, s, low)
        else:
            s = s + mb.astype(mx.float32)
    return s


def blocked_attention(queries, keys, values, *, scale: float, mask=None,
                      block: int = _BLOCK):
    """``mx.fast.scaled_dot_product_attention`` for training: same inputs,
    same output, a backward whose transient memory is linear in T."""
    B, Hq, Tq, D = queries.shape
    Hkv, Tk = keys.shape[1], keys.shape[2]
    Dv = values.shape[-1]
    rep = Hq // Hkv
    causal, mask_arr = _prep_mask(mask, Tq, Tk)
    offset = Tk - Tq
    dtype = queries.dtype
    block = max(1, min(block, Tq))

    def q5(q):
        return q.reshape(B, Hkv, rep, Tq, D)

    def kend_of(i1):
        return min(Tk, i1 + offset) if causal else Tk

    @mx.custom_function
    def attn(q, k, v):
        q = q5(q)
        k = k[:, :, None]
        v = v[:, :, None]
        outs, lses = [], []
        for i0 in range(0, Tq, block):
            i1 = min(Tq, i0 + block)
            kend = kend_of(i1)
            s = _scores(q[:, :, :, i0:i1], k[:, :, :, :kend], scale, causal,
                        i0, i1, kend, offset, mask_arr, dtype)
            lse = mx.logsumexp(s, axis=-1, keepdims=True)
            p = mx.exp(s - lse).astype(dtype)
            outs.append(mx.matmul(p, v[:, :, :, :kend]))
            lses.append(lse)
        out = mx.concatenate(outs, axis=3).reshape(B, Hq, Tq, Dv)
        lse = mx.concatenate(lses, axis=3).reshape(B, Hq, Tq, 1)
        return out, lse

    @attn.vjp
    def attn_vjp(primals, cotangents, outputs):
        q, k, v = primals
        dout = cotangents[0]
        out, lse = outputs
        q = q5(q)
        k = k[:, :, None]
        v = v[:, :, None]
        dout = dout.reshape(B, Hkv, rep, Tq, Dv)
        out = out.reshape(B, Hkv, rep, Tq, Dv)
        lse = lse.reshape(B, Hkv, rep, Tq, 1)
        delta = mx.sum(dout.astype(mx.float32) * out.astype(mx.float32), axis=-1,
                       keepdims=True)
        dq_blocks = []
        dk = mx.zeros((B, Hkv, Tk, D), dtype=mx.float32)
        dv = mx.zeros((B, Hkv, Tk, Dv), dtype=mx.float32)
        for i0 in range(0, Tq, block):
            i1 = min(Tq, i0 + block)
            kend = kend_of(i1)
            qb = q[:, :, :, i0:i1]
            kb = k[:, :, :, :kend]
            vb = v[:, :, :, :kend]
            if dq_blocks:
                # Evaluate the blocks one after another: without this edge
                # the three gradient chains can run block by block out of
                # step and every block's probabilities stay alive until the
                # last chain consumes them.
                qb, kb, vb = mx.depends([qb, kb, vb], [dq_blocks[-1], dk, dv])
            s = _scores(qb, kb, scale, causal, i0, i1, kend, offset, mask_arr, dtype)
            p = mx.exp(s - lse[:, :, :, i0:i1])                      # f32
            db = dout[:, :, :, i0:i1]
            dp = mx.matmul(db, vb.swapaxes(-1, -2)).astype(mx.float32)
            ds = p * (dp - delta[:, :, :, i0:i1]) * scale            # f32
            pd = p.astype(dtype)
            dsd = ds.astype(dtype)
            dv_b = mx.sum(mx.matmul(pd.swapaxes(-1, -2), db), axis=2)
            dk_b = mx.sum(mx.matmul(dsd.swapaxes(-1, -2), qb), axis=2)
            dq_blocks.append(mx.matmul(dsd, kb))
            dv = dv.at[:, :, :kend].add(dv_b.astype(mx.float32))
            dk = dk.at[:, :, :kend].add(dk_b.astype(mx.float32))
        dq = mx.concatenate(dq_blocks, axis=3).reshape(B, Hq, Tq, D)
        return dq.astype(dtype), dk.astype(dtype), dv.astype(dtype)

    out, _lse = attn(queries, keys, values)
    return out


def install_training_attention(model, *, block: int = _BLOCK):
    """Route every attention call of ``model`` to ``blocked_attention``
    while ``model.training`` is set. Patches the ``scaled_dot_product_attention``
    global in each module that defines one of the model's layer classes
    (the mlx-lm and mlx-vlm seam every stock attention calls) and returns
    a function that restores them. Calls with a cache or attention sinks,
    or made while the model is in eval mode, take the original path."""
    import sys

    patched = []
    seen = set()
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        orig = getattr(mod, "scaled_dot_product_attention", None)
        if orig is None or getattr(orig, "_gmlx_training_attention", False):
            continue

        def make(orig):
            def sdpa(queries, keys, values, cache=None, scale=None, mask=None,
                     sinks=None, **kw):
                if (model.training and cache is None and sinks is None
                        and not kw and scale is not None
                        and isinstance(keys, mx.array)):
                    return blocked_attention(queries, keys, values, scale=scale,
                                             mask=mask, block=block)
                return orig(queries, keys, values, cache, scale, mask,
                            *(() if sinks is None else (sinks,)), **kw)
            sdpa._gmlx_training_attention = True
            sdpa._gmlx_orig = orig
            return sdpa

        setattr(mod, "scaled_dot_product_attention", make(orig))
        patched.append((mod, orig))

    def restore():
        for mod, orig in patched:
            setattr(mod, "scaled_dot_product_attention", orig)
        patched.clear()

    restore.count = len(patched)
    return restore
