# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The decision forward of a d1 model: the whole prompt in one pass, with
no cache, read at chosen rows.

d1's full-attention layers read the prompt in both directions; its short
convs and sliding layers stay causal, as llama.cpp runs them
(``LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY``). The forward runs layer by layer
over row blocks:

- a conv or sliding layer runs its own ``DecoderLayer`` code with a cache
  of its own, block after block, as chat's chunked prefill does;
- a full layer first computes the keys and values of every row, then runs
  each block of queries against all of them.

Rows of one batch are right-padded. Pads come after the real rows, so the
causal layers never carry them back, and the full layers mask them as
keys.
"""

from __future__ import annotations

from typing import Generator, Optional, Sequence

import mlx.core as mx

from gmlx.cache.attention import create_attention_mask, create_ssm_mask

from .model import FULL, SLIDING

DEFAULT_BLOCK = 2048
# The fewest positions in a block, however many rows a batch has.
MIN_BLOCK = 16
# Head rows decoded to float32 at a time: 8,192 x 2,048 x 4 bytes, 64 MiB.
HEAD_CHUNK = 8192
# What a forward holds beside its rows and blocks: the head rows, masks
# and the allocator's slack.
_FIXED_BYTES = 128 << 20


def block_for(rows: int, block: int = DEFAULT_BLOCK) -> int:
    """The positions in one block of a batch of ``rows``: a block holds
    about ``block`` rows x positions, whatever the batch width."""
    return max(MIN_BLOCK, block // max(1, rows))


def price_bytes(args, rows: int, n: int, *, embeds: bool = False,
                block: int = DEFAULT_BLOCK) -> int:
    """The bytes a forward of ``rows`` rows of ``n`` positions holds at its
    peak, beside the weights.

    - Held for the whole forward: every row's hidden state, and one full
      layer's keys and values, twice while its blocks are joined. With
      input embeddings, the stacked inputs too.
    - Held for one block: its output, the attention projections and the
      FFN's gate, up and product, with a quarter more for the slack.
    """
    hidden = args.hidden_size
    kv_width = 2 * args.num_key_value_heads * args.head_dim
    held = rows * n * (hidden * 2 + kv_width * 2 * 2)
    if embeds:
        held += rows * n * hidden * 2
    tokens = rows * min(n, block_for(rows, block))
    per_block = tokens * (hidden * 2 * 4 + args.intermediate_size * 2 * 4)
    return int(held + per_block * 5 // 4 + _FIXED_BYTES)


class _FrozenKeys:
    """Stands in for a full layer's cache: ``Attention`` rotates its
    queries at ``offset`` and attends to the keys and values of every row,
    computed before. It has no ``bits`` and no ``make_mask``, so attention
    goes straight to ``mx.fast.scaled_dot_product_attention``."""

    def __init__(self, keys: mx.array, values: mx.array, offset: int,
                 end: Optional[int] = None):
        self.offset = offset
        self._keys = keys if end is None else keys[:, :, :end]
        self._values = values if end is None else values[:, :, :end]

    def update_and_fetch(self, keys, values):
        return self._keys, self._values


def _keys_values(attn, x: mx.array, offset: int):
    """A block's keys and values, as ``Attention.__call__`` makes them."""
    b, length, _ = x.shape
    keys = attn.k_proj(x).reshape(b, length, attn.n_kv_heads, -1)
    values = attn.v_proj(x).reshape(b, length, attn.n_kv_heads, -1)
    keys = attn._rotate(attn.k_norm(keys.transpose(0, 2, 1, 3)), offset)
    return keys, values.transpose(0, 2, 1, 3)


def _all_keys_values(layer, blocks: list, starts: list):
    """A full layer's keys and values for every row, block by block,
    yielding after each block."""
    attn = layer.self_attn
    parts_k, parts_v = [], []
    for start, h in zip(starts, blocks):
        k, v = _keys_values(attn, layer.input_layernorm(h), start)
        mx.eval(k, v)
        parts_k.append(k)
        parts_v.append(v)
        yield
    keys = mx.concatenate(parts_k, axis=2)
    values = mx.concatenate(parts_v, axis=2)
    del parts_k, parts_v
    mx.eval(keys, values)
    return keys, values


def _head_rows(text, index) -> mx.array:
    """The tied head's rows at ``index``, a slice or an array of ids,
    decoded to float32. A Q8_0 value is an int8 times a float16 scale, so
    float32 holds it exactly."""
    if not text.args.tie_word_embeddings:
        raise ValueError("decision forward: an lfm3 head must be tied")
    embed = text.model.embed_tokens
    rows = embed["weight"][index]
    codec = getattr(embed, "kquant_type", None)
    if codec is not None:
        import mlx_kquant as kq

        # GGUF scales live inside the rows; the scales argument is unused.
        out = kq.dequantize(rows, embed["scales"], codec, dtype=mx.float32)
        return out.reshape(rows.shape[0], -1)
    if "scales" in embed:
        biases = embed.get("biases")
        return mx.dequantize(rows, embed["scales"][index],
                             None if biases is None else biases[index],
                             group_size=embed.group_size, bits=embed.bits,
                             mode=getattr(embed, "mode", "affine"),
                             dtype=mx.float32)
    return rows.astype(mx.float32)


def _float32_logits(text, h: mx.array,
                    ids: Optional[Sequence[int]] = None) -> mx.array:
    """The unmasked head in float32, as llama.cpp computes it: over the
    whole vocabulary, or over ``ids`` only. The quantized matmul of the
    chat head returns 16-bit logits."""
    if ids is not None:
        return h @ _head_rows(text, mx.array([int(i) for i in ids])).T
    vocab = text.args.vocab_size
    parts = []
    for start in range(0, vocab, HEAD_CHUNK):
        rows = _head_rows(text, slice(start, min(start + HEAD_CHUNK, vocab)))
        part = h @ rows.T
        mx.eval(part)
        parts.append(part)
    return mx.concatenate(parts, axis=-1)


def _key_mask(lengths: Sequence[int], n: int) -> Optional[mx.array]:
    """``[rows, 1, 1, n]``, true at each row's real keys. None when no row
    is padded. It broadcasts over heads and queries, and stays on the
    fused attention kernel (tests/gen/test_prefill_fused_rule.py)."""
    if all(length == n for length in lengths):
        return None
    real = mx.arange(n)[None] < mx.array(list(lengths))[:, None]
    return real[:, None, None, :]


def _split(text, inputs: mx.array, block: int) -> list:
    """The input rows as evaluated blocks of ``block`` rows. Ids are
    embedded block by block. Embeddings are copied, so the caller's array
    can go before layer 0."""
    embed = text.model.embed_tokens
    blocks = []
    for start in range(0, inputs.shape[1], block):
        part = inputs[:, start:start + block]
        part = embed(part) if inputs.ndim == 2 else part * 1
        mx.eval(part)
        blocks.append(part)
    return blocks


def decision_forward(
    text,
    inputs: mx.array,
    lengths: Sequence[int],
    *,
    positions: Optional[Sequence[int]] = None,
    block: int = DEFAULT_BLOCK,
    causal: bool = False,
    head_ids: Optional[Sequence[int]] = None,
) -> Generator[None, None, mx.array]:
    """Run a d1 text model over ``inputs`` and return float32 logits
    ``[rows, vocab]`` from the unmasked head, computed in float32, or
    ``[rows, len(head_ids)]`` for the ``head_ids`` columns only.

    ``text`` is the lfm3 text ``Model``. ``inputs`` is ids ``[rows, n]``
    or input embeddings ``[rows, n, hidden]``, right-padded; row r holds
    ``lengths[r]`` real rows and is read at ``positions[r]``, by default
    its last real row. A generator: it yields after each block, and its
    return value is the logits. ``causal`` runs the full layers causally,
    for tests.
    """
    rows, n = inputs.shape[0], inputs.shape[1]
    lengths = [int(length) for length in lengths]
    if len(lengths) != rows or not all(0 < x <= n for x in lengths):
        raise ValueError(f"decision forward: lengths {lengths} for {rows} "
                         f"rows of {n}")
    if positions is None:
        positions = [length - 1 for length in lengths]
    positions = [int(p) for p in positions]
    if len(positions) != rows or not all(
            0 <= p < length for p, length in zip(positions, lengths)):
        raise ValueError(f"decision forward: positions {positions} outside "
                         f"lengths {lengths}")

    model = text.model
    starts = list(range(0, n, block))
    blocks = _split(text, inputs, block)
    del inputs
    full_mask = "causal" if causal else _key_mask(lengths, n)
    caches = text.make_cache()

    for index, layer in enumerate(model.layers):
        cache = caches[index]
        caches[index] = None
        kv = None
        if layer.layer_type == FULL:
            kv = yield from _all_keys_values(layer, blocks, starts)
        for i, (start, h) in enumerate(zip(starts, blocks)):
            if kv is not None:
                end = start + h.shape[1] if causal else None
                out = layer(h, full_mask, _FrozenKeys(*kv, start, end))
                mx.eval(out)
            elif layer.layer_type == SLIDING:
                mask = create_attention_mask(
                    h, cache, window_size=model.sliding_window)
                out = layer(h, mask, cache)
                mx.eval(out)
            else:
                out = layer(h, create_ssm_mask(h, cache), cache)
                # The conv tail is evaluated with the block, so it does
                # not hold the block's window until the next block.
                mx.eval(out, cache[0])
            blocks[i] = out
            yield
        del cache, kv

    picked = [blocks[p // block][r, p % block] for r, p in enumerate(positions)]
    h = mx.stack(picked).astype(mx.float32)
    logits = _float32_logits(text, model.norm(h).astype(mx.float32), head_ids)
    mx.eval(logits)
    return logits


def run_decision_forward(text, inputs, lengths, **kwargs) -> mx.array:
    """:func:`decision_forward` run to the end."""
    gen = decision_forward(text, inputs, lengths, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


__all__ = ["DEFAULT_BLOCK", "block_for", "decision_forward", "price_bytes",
           "run_decision_forward"]
