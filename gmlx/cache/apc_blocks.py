"""Join prompt cache blocks back into one key and one value array per layer.

A block holds its arrays stacked (KVBlockHandle). The join writes each
group of blocks into per-layer arrays and frees the group before the next,
so a restore holds one group beside its output. Joining a view of every
block per layer would make a graph node of each view, which costs several
times the copy on a long prefix.
"""

from __future__ import annotations

import mlx.core as mx

# Bytes of blocks joined at once. It keeps each group's join far under the
# 2^31 elements past which stock concatenate misplaces its inputs.
GROUP_BYTES = 1 << 30


def block_tokens(blocks) -> int:
    """Tokens the blocks hold, in total."""
    return sum(b.kv_handle().tokens for b in blocks)


def block_layers(block) -> int:
    """Layers a block holds, or 0 for a block without KV."""
    h = block.kv_handle()
    return h.n_layers if h is not None else 0


def _groups(handles):
    out, cur, size = [], [], 0
    for h in handles:
        n = h.resident_bytes()
        if cur and size + n > GROUP_BYTES:
            out.append(cur)
            cur, size = [], 0
        cur.append(h)
        size += n
    if cur:
        out.append(cur)
    return out


def join_blocks(blocks) -> list[tuple[mx.array, mx.array]]:
    """(keys, values) per layer of the blocks' tokens, in block order.

    Every block must hold the same layers in the same shapes. With more
    than one group the output is evaluated group by group."""
    from gmlx.eval_guard import guard

    handles = [b.kv_handle() for b in blocks]
    first = handles[0]
    layout = [p.shape[:-2] + p.shape[-1:] for p in first.parts]
    for h in handles[1:]:
        if h.index != first.index or [
                p.shape[:-2] + p.shape[-1:] for p in h.parts] != layout:
            raise ValueError("join_blocks: blocks of different layouts")
    total = sum(h.tokens for h in handles)
    outs = []
    for c, _row in first.index:
        shape = list(first.parts[c].shape[1:])
        shape[-2] = total
        outs.append(mx.zeros(shape, dtype=first.parts[c].dtype))
    groups = _groups(handles)
    start = 0
    for grp in groups:
        n = sum(h.tokens for h in grp)
        joined = [mx.concatenate([h.parts[c] for h in grp], axis=-2)
                  for c in range(len(first.parts))]
        for i, (c, row) in enumerate(first.index):
            outs[i][..., start:start + n, :] = joined[c][row]
        del joined
        if len(groups) > 1:
            guard.eval(*outs, site="apc-block-join", owner="scratch")
        start += n
    n_layers = first.n_layers
    return [(outs[i], outs[n_layers + i]) for i in range(n_layers)]
