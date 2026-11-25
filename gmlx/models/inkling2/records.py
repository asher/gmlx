# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Recorded layer visits at decode and verify widths.

A visit runs as two kq records around a plain middle. The first record
holds the attention inputs (the depth mix, the controllers, the norms,
the projections, the K and V convolutions, the clips), the second the
tail (the gate, the output projection, the FFN depth mix and MoE, the
other two convolutions). Both depend on the row count alone. The cache
append and the attention core, which depend on the cache offset, are
built as the plain call builds them. A record holds the layer's arrays as
constants; every array of the request it reads (the stream, the blocks,
the messages, the convolution states, the positions) is an input.

A layout runs plain on its first use, records on its second and replays
after that. A record replays its Metal calls, so a layout whose record kq
refuses to replay that way (``kq.fused_direct_info``) stays plain. Bits
equal the plain call's.

Off by default: on an M5 Max both steps are GPU-bound, and the replay's
barriers cost more GPU time than the host build it saves.
``GMLX_INKLING2_RECORDS=1`` turns it on where the host build bounds the
step.
"""

import mlx.core as mx

from gmlx.envflags import env_bool, env_int

# Visits wider than this run plain: a prefill chunk's shapes rarely repeat.
MAX_ROWS = 16

_ON: list = [None]
# Bumped by invalidate: records of an older epoch are dropped at their
# next use. Blocked for the process once weights are swapped around a
# layer call.
_EPOCH = [0]
_BLOCKED = [False]
# Handles held by the process, and the most it may hold.
_HELD = [0]
_MAX: list = [None]
# Records made and replays run, for probes and tests.
STATS = {"recorded": 0, "replayed": 0, "plain": 0}
# A record kq will not replay as Metal calls runs its primitives one by
# one, slower than the plain call: the layout runs plain. Tests turn this
# off to check the records on any kq.
DIRECT_ONLY = [True]


def enabled() -> bool:
    if _ON[0] is None:
        on = env_bool("GMLX_INKLING2_RECORDS", False)
        if on:
            import mlx_kquant as kq

            on = hasattr(kq, "fuse_graph") and hasattr(kq, "env_cache")
        _ON[0] = on
    return _ON[0] and not _BLOCKED[0]


def invalidate(block: bool = False) -> None:
    """Drop every record: a module was replaced or wrapped (a LoRA
    install). ``block`` also turns records off for the process (weights
    swapped around each layer call)."""
    _EPOCH[0] += 1
    if block:
        _BLOCKED[0] = True


def _max_handles() -> int:
    if _MAX[0] is None:
        _MAX[0] = max(2, env_int("GMLX_INKLING2_RECORDS_MAX", 2048))
    return _MAX[0]


def _leaves(node, out: list) -> list:
    """``(container, key, value)`` for every array and module under
    ``node``: what a record froze."""
    import mlx.nn as nn

    items = (node.items() if isinstance(node, dict)
             else enumerate(node) if isinstance(node, (list, tuple)) else ())
    for k, v in items:
        if isinstance(v, mx.array):
            out.append((node, k, v))
        elif isinstance(v, (dict, list, tuple)):
            if isinstance(v, nn.Module):
                out.append((node, k, v))
            _leaves(v, out)
    return out


def _has_lora(layer) -> bool:
    from gmlx.load.modules import LoRAKQuantEmbedding, LoRAKQuantLinear

    for m in layer.modules():
        if isinstance(m, (LoRAKQuantLinear, LoRAKQuantEmbedding)):
            return True
        if getattr(m, "_kq_lora", None) is not None:
            return True
    return False


def _release(table: dict) -> None:
    import mlx_kquant as kq

    for rec in table.values():
        if isinstance(rec, tuple):
            for handle in rec[:2]:
                kq.fuse_release(handle)
                _HELD[0] -= 1


def _table(layer) -> dict:
    """The layer's records by layout, emptied when the epoch moved or one
    of the layer's arrays or modules was replaced."""
    recs = vars(layer).get("_ink_records")
    if (recs is None or recs[0] != _EPOCH[0]
            or not all(d[k] is v for d, k, v in recs[2])):
        if recs is not None:
            _release(recs[1])
        recs = (_EPOCH[0], {}, _leaves(layer, []))
        object.__setattr__(layer, "_ink_records", recs)
    return recs[1]


def _usable(layer, h, conv_cache, kv_cache, pad_mask) -> bool:
    if (kv_cache is None or conv_cache is None or pad_mask is not None
            or layer.training or h.shape[0] != 1 or h.shape[1] > MAX_ROWS
            or not enabled()):
        return False
    if conv_cache.lengths is not None or conv_cache.history_capacity:
        return False
    slots = list(conv_cache.cache)
    if len(slots) != 4 or not all(isinstance(a, mx.array) for a in slots):
        return False
    from gmlx.stream.moe_experts import expert_controls_active

    return not expert_controls_active(layer.mlp)


def _record(inputs: list, outputs: list):
    """(handle, picked inputs, output refs): the graph from ``inputs`` to
    ``outputs`` with the inputs it does not read left out. An output that
    is an input is passed through. None where kq refuses the graph."""
    import mlx_kquant as kq

    ins = {id(a): i for i, a in enumerate(inputs)}
    if len(ins) != len(inputs):
        return None
    outs: list = []
    refs = []
    for a in outputs:
        if id(a) in ins:
            refs.append((0, ins[id(a)]))
            continue
        j = next((j for j, o in enumerate(outs) if o is a), len(outs))
        if j == len(outs):
            outs.append(a)
        refs.append((1, j))
    # Direct replay needs the cache of KQ_* reads.
    kq.env_cache(True)
    try:
        probe = kq.fuse_graph(inputs, outs)
        used = kq.fused_info(probe)[2]
        kq.fuse_release(probe)
        pick = [i for i, u in enumerate(used) if u or i == 0]
        handle = kq.fuse_graph([inputs[i] for i in pick], outs, direct=True)
    except (ValueError, RuntimeError, TypeError):
        return None
    if (DIRECT_ONLY[0]
            and kq.fused_direct_info(handle).get("state") == "refused"):
        kq.fuse_release(handle)
        return None
    return handle, pick, refs


def _replay(rec, inputs: list) -> list:
    import mlx_kquant as kq

    handle, pick, refs = rec
    res = kq.fused_call(handle, [inputs[i] for i in pick])
    return [(inputs if kind == 0 else res)[i] for kind, i in refs]


def _windows(sink, n: int) -> list:
    return [] if sink is None else [w for _, _, w in sink[-n:]]


def visit(layer, h, ip, blocks, conv_cache, kv_cache, pad_mask, positions,
          messages, conv_sink):
    """The visit's output through its records, or None where the plain
    call must run. Leaves the caches, ``blocks`` and ``conv_sink`` as the
    plain call leaves them, except the convolution offset, which the
    caller advances."""
    if not _usable(layer, h, conv_cache, kv_cache, pad_mask):
        return None
    table = _table(layer)
    msgs = [] if messages is None else list(messages)
    full = not layer.is_sliding
    a_in = [h, *blocks, *msgs[:3], conv_cache[0], conv_cache[1]]
    if full:
        a_in.append(positions)
    key = (conv_sink is not None, len(blocks), len(msgs),
           tuple((a.shape, a.dtype) for a in a_in))
    rec = table.get(key)
    if rec is None:
        table[key] = "plain" if _has_lora(layer) else False
        STATS["plain"] += 1
        return None
    if rec == "plain":
        STATS["plain"] += 1
        return None
    windows = conv_sink is not None
    n_blocks = len(blocks)
    if rec is False:
        table[key] = "plain"
        STATS["plain"] += 1
        if _HELD[0] + 2 > _max_handles():
            return None
        # The plain call, kept apart so its graphs can be recorded.
        sink_a = [] if windows else None
        q, k, v, r, sink, gate = layer.attn_inputs(
            h, blocks, conv_cache, None, positions, messages, sink_a)
        a_out = [q, k, v, r, sink, gate, conv_cache[0], conv_cache[1],
                 *_windows(sink_a, 2)]
        att = layer.attn_core(q, k, v, r, sink, kv_cache)
        b_in = [h, att, gate, *blocks, *msgs[3:], conv_cache[2],
                conv_cache[3]]
        sink_b = [] if windows else None
        out = layer.tail(h, ip, blocks, att, gate, conv_cache, None,
                         messages, sink_b)
        pushed = blocks[n_blocks:]
        b_out = [out, conv_cache[2], conv_cache[3], *pushed,
                 *_windows(sink_b, 2)]
        if windows:
            conv_sink.extend(sink_a)
            conv_sink.extend(sink_b)
        ra = _record(a_in, a_out)
        rb = None if ra is None else _record(b_in, b_out)
        if rb is None:
            if ra is not None:
                import mlx_kquant as kq

                kq.fuse_release(ra[0])
            return out
        table[key] = (ra[0], rb[0], ra, rb, len(pushed),
                      (att.shape, att.dtype))
        _HELD[0] += 2
        STATS["recorded"] += 1
        return out
    _, _, ra, rb, n_pushed, att_key = rec
    STATS["replayed"] += 1
    a = _replay(ra, a_in)
    q, k, v, r, sink, gate, c0, c1 = a[:8]
    conv_cache[0], conv_cache[1] = c0, c1
    if windows:
        conv_sink.extend([(conv_cache, 0, a[8]), (conv_cache, 1, a[9])])
    att = layer.attn_core(q, k, v, r, sink, kv_cache)
    if (att.shape, att.dtype) != att_key:
        sink_b = [] if windows else None
        out = layer.tail(h, ip, blocks, att, gate, conv_cache, None,
                         messages, sink_b)
        if windows:
            conv_sink.extend(sink_b)
        return out
    b_in = [h, att, gate, *blocks, *msgs[3:], conv_cache[2], conv_cache[3]]
    b = _replay(rb, b_in)
    out, c2, c3 = b[:3]
    conv_cache[2], conv_cache[3] = c2, c3
    blocks.extend(b[3:3 + n_pushed])
    if windows:
        w = b[3 + n_pushed:]
        conv_sink.extend([(conv_cache, 2, w[0]), (conv_cache, 3, w[1])])
    return out


def stats(model) -> dict:
    """Counts of the records under ``model`` and what direct replay made
    of them: states and the reasons kq refused a record."""
    import mlx_kquant as kq

    out = {"records": 0, "states": {}, "refused": {}, **STATS}
    for m in model.modules():
        recs = vars(m).get("_ink_records")
        if recs is None:
            continue
        for rec in recs[1].values():
            if not isinstance(rec, tuple):
                continue
            for handle in rec[:2]:
                out["records"] += 1
                info = kq.fused_direct_info(handle)
                st = info.get("state", "")
                out["states"][st] = out["states"].get(st, 0) + 1
                if st == "refused":
                    why = (info.get("reason", ""),
                           tuple(info.get("primitives", ()) or ()))
                    out["refused"][why] = out["refused"].get(why, 0) + 1
    return out
