"""The site gate of a MiniMax-H3 step, video or image, on the blocks
both share (``gmlx.image.h3.blocks``); it never loads gmlx.video.

Each op of a block on the path a request takes is read against the
stock op in float32 on the same production input, within 2 n u.
The float32 side runs on the CPU, whose float32 products are exact. A
reading is (block, "step", op, call number) for an op of a block, with
``qg.HEAD`` after it for the worst head row of the q and k site, and
(module path, call number) for a linear leaf, so the helpers of
qwen_gates.py read it. n counts the outputs the stock ops of a site
write in the activation dtype, and a site that a kernel route ran is
held to the kernel's own count, ``bg.KERNEL_ROUNDINGS``.

The sites of an H3 block: the two norms; a modulation and a gated
residual add per segment at each of the two branches; q and k through
the q and k seam, in the pair order of ``rope.pair_halves``; sdpa per
chunk of query rows; the feed-forward from its input to the rows of fc2,
swiglu, which rounds over the rounding of fc1 on the stock path; and
every linear leaf. A refiner block also norms q and k as modules, has no
RoPE and no modulation, and adds its residuals in line. The shift of
each modulation is added in line, and the exact gate holds it. Under a
patch table each adapted leaf is read whole, its delta included, against
the stock product with the table's delta in float32, and held to two
more roundings, or to the low-rank op's own count where that op ran.
"""
from __future__ import annotations

import math
from collections import Counter
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import block_gates as bg
import qwen_gates as qg

from gmlx.image.h3 import blocks


OP_ROUNDINGS = {
    "modulate": 2,        # 1 + scale, then the product
    "sdpa": 1,
    "swiglu": 3,          # sigmoid(a), a times it, then the product with b
    "gated_residual_once": 1,  # in float32, rounded once
}


def block_ops(kind: str, rows: int, segments: int, chunk: int) -> dict:
    """The calls of each op in one call of a block of ``kind`` over
    ``rows`` rows in ``segments`` segments, attention ``chunk`` query rows
    a call."""
    if kind == "block":
        return {"norm": 2, "modulate": 2 * segments, "qk": 1,
                "sdpa": math.ceil(rows / chunk), "swiglu": 1,
                "gated_residual_once": 2 * segments}
    return {"norm": 4, "sdpa": 1, "swiglu": 1}


def block_names(model) -> dict:
    """id of each block of ``model`` to (name, kind)."""
    out = {id(b): (f"blocks.{i}", "block")
           for i, b in enumerate(model.blocks)}
    for i, b in enumerate(model.token_refiner.blocks):
        out[id(b)] = (f"token_refiner.blocks.{i}", "refiner")
    return out


def qk_stock_n(model) -> int:
    """The stock q and k op's rounding count: the norm, then the cast
    after RoPE."""
    w = model.blocks[0].attn.q_norm.weight
    return bg.norm_roundings(w, model.act_dtype) + 1


@contextmanager
def step_sites(model, sites: dict, routed: dict | None = None,
               heads: dict | None = None, skipped: set | None = None,
               by_row: dict | None = None, built=None):
    """Fill ``sites`` with a reading for every op of every block and every
    call of a linear leaf while steps run: key to (relative L2, gate).
    ``routed`` takes the mlx-kquant op of each site a kernel route ran,
    ``heads`` the place of the worst head row of each q and k site, as
    (0, batch, head, token). ``skipped`` takes the path of a leaf called
    on a float32 input, which runs in float32 and has no site. ``by_row``
    takes the worst token row of each site on the GLU kernel, at the
    site's gate (block_gates.ROW_RULE). ``built`` is the built table of
    the patch table the step runs under, from ``install.build_table`` in
    the block form, and gives the exact side each leaf's delta."""
    import mlx.nn as nn
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    from gmlx.image import ops
    from gmlx.image.lora import install
    from gmlx.image.ops import attention, epilogues, norms

    f = mx.float32
    act = model.act_dtype
    names = block_names(model)
    cur: list = []
    counts: dict = {}
    calls: dict = {}
    ran: list = []
    wrapped = any(isinstance(m, install.ImageLoRALinear)
                  for m in install.modules(model).values())
    paths = {id(m): p for p, m in install.modules(model).items()
             if install.linear_dims(m) is not None}

    def factors(path):
        g = None if built is None else built.get(path)
        return None if g is None else (g.a_t.astype(f), g.b_t.astype(f))

    def exact_leaf(mod, x32):
        """The leaf's product in float32 on its dequantized weight, with
        its delta."""
        base = getattr(mod, "base", mod)
        codec = getattr(base, "kquant_type", None)
        w = base["weight"].astype(f) if codec is None else \
            kq.dequantize(base["weight"], base["scales"], codec, f)
        y = x32 @ w.T
        if "bias" in base:
            y = y + base["bias"].astype(f)
        fac = factors(paths[id(mod)])
        if fac is not None:
            y = y + (x32 @ fac[0]) @ fac[1]
        return y

    def leaf_rounds(mod) -> int:
        base = getattr(mod, "base", mod)
        quant = getattr(base, "kquant_type", None) is not None
        n = 2 if quant and "bias" in base else 1
        return n + (2 if factors(paths[id(mod)]) is not None else 0)

    def read(op, got, want, rounds, by_head=False, rows=False):
        key = (cur[-1], "step", op)
        n = counts[key] = counts.get(key, 0) + 1
        outs = got if isinstance(got, tuple) else (got,)
        same = all(o.dtype == act for o in outs)
        gate = 2 * rounds * qg.SITE_UNIT * same
        with mx.stream(mx.cpu):
            exact = want()
            rel_l2 = qg._rel_l2(got, exact)
            if by_head:
                e2, r2 = bg._head_sums(outs, exact if isinstance(
                    exact, tuple) else (exact,))
                mx.eval(e2, r2)
            if rows:
                t2, w2 = bg._row_sums(got, exact)
                mx.eval(t2, w2)
        sites[(*key, n)] = (rel_l2, gate)
        if rows and by_row is not None:
            by_row[(*key, n)] = (bg._worst(np.array(t2), np.array(w2))[0],
                                 gate)
        if by_head:
            worst, at = bg._worst(np.array(e2), np.array(r2))
            sites[(*key, n, qg.HEAD)] = (worst, gate)
            if heads is not None:
                heads[(*key, n)] = at
        return (*key, n)

    def within(orig):
        def run(blk, *a, **k):
            hit = names.get(id(blk))
            if hit is None:
                return orig(blk, *a, **k)
            cur.append(hit[0])
            try:
                return orig(blk, *a, **k)
            finally:
                cur.pop()
        return run

    def norm(orig):
        def run(nrm, x):
            y = orig(nrm, x)
            if cur:
                read("norm", y, lambda: mx.fast.rms_norm(
                    x.astype(f), nrm.weight.astype(f), nrm.eps),
                    bg.norm_roundings(nrm.weight, x.dtype))
            return y
        return run

    def modulate(orig):
        def run(x, scale):
            y = orig(x, scale)
            if cur:
                read("modulate", y, lambda: epilogues.modulate(
                    x.astype(f), scale.astype(f)), OP_ROUNDINGS["modulate"])
            return y
        return run

    def qk(orig):
        def run(q, k, q_w, k_w, cos, sin, eps):
            before = len(ran)
            out = orig(q, k, q_w, k_w, cos, sin, eps)
            if not cur:
                return out
            kernel = "qk" in ran[before:]
            key = read("qk", tuple(out), lambda: attention.qk_norm_rope(
                q.astype(f), k.astype(f), q_w.astype(f), k_w.astype(f), cos,
                sin, eps),
                bg.KERNEL_ROUNDINGS["qk"] if kernel else
                bg.norm_roundings(q_w, q.dtype) + 1,  # the cast after RoPE
                by_head=True)
            if kernel and routed is not None:
                routed[key] = bg.KERNEL_OPS["qk"]
            return out
        return run

    def sdpa(orig):
        def run(q, k, v, mask=None):
            out = orig(q, k, v, mask)
            if cur:
                read("sdpa", out, lambda: attention.sdpa(
                    q.astype(f), k.astype(f), v.astype(f), mask),
                    OP_ROUNDINGS["sdpa"])
            return out
        return run

    def hidden(orig):
        def run(blk, h):
            before = len(ran)
            y = orig(blk, h)
            if not cur:
                return y
            fc1 = blk.mlp.fc1
            kernel = "glu" in ran[before:]

            def want():
                a, b = mx.split(exact_leaf(fc1, h.astype(f)), 2, axis=-1)
                return epilogues.swiglu(a, b)

            rounds = bg.KERNEL_ROUNDINGS["glu"] if kernel else \
                OP_ROUNDINGS["swiglu"] + leaf_rounds(fc1)
            key = read("swiglu", y, want, rounds, rows=kernel)
            if kernel and routed is not None:
                routed[key] = bg.KERNEL_OPS["glu"]
            return y
        return run

    def gated_residual_once(orig):
        def run(x, gate, y):
            out = orig(x, gate, y)
            if cur:
                read("gated_residual_once", out,
                     lambda: epilogues.gated_residual(
                         x.astype(f), gate.astype(f), y.astype(f)),
                     OP_ROUNDINGS["gated_residual_once"])
            return out
        return run

    def leaf(orig):
        def run(mod, x, *a, **k):
            before = len(ran)
            y = orig(mod, x, *a, **k)
            path = paths.get(id(mod))
            if path is None:
                return y
            if x.dtype != act:
                if skipped is not None:
                    skipped.add(path)
                return y
            with mx.stream(mx.cpu):
                rel_l2 = qg._rel_l2(y, exact_leaf(mod, x.astype(f)))
            kernel = "lora" in ran[before:]
            rounds = bg.KERNEL_ROUNDINGS["lora"] if kernel else \
                leaf_rounds(mod)
            n = calls[path] = calls.get(path, 0) + 1
            sites[(path, n)] = (rel_l2, 2 * rounds * qg.SITE_UNIT)
            if kernel and routed is not None:
                routed[(path, n)] = bg.KERNEL_OPS["lora"]
            return y
        return run

    leaf_types = {type(m) for m in install.linear_leaves(model).values()}
    assert leaf_types <= {nn.Linear, KQuantLinear}, leaf_types
    if wrapped:
        leaf_types = {install.ImageLoRALinear}
    hooks = [(blocks.DiTBlock, "__call__", within),
             (blocks.RefinerBlock, "__call__", within),
             (norms.RMSNorm, "__call__", norm),
             (blocks._Block, "hidden", hidden),
             (ops, "modulate", modulate), (ops, "qk_norm_rope", qk),
             (ops, "sdpa", sdpa),
             (ops, "gated_residual_once", gated_residual_once)]
    hooks += [(cls, "__call__", leaf) for cls in leaf_types]
    saved = []
    try:
        for obj, attr, hook in hooks:
            saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, hook(saved[-1][2]))
        with bg.kernel_calls(ran.append):
            yield sites
    finally:
        for obj, attr, orig in reversed(saved):
            setattr(obj, attr, orig)


def missing_ops(model, sites: dict, lay, routed: dict, skipped,
                chunk: int) -> list:
    """What one step over ``lay`` leaves unread: each block calls every op
    of ``block_ops`` for its kind at attention ``chunk``, and every linear
    leaf but those of ``skipped`` is called once, fc1 once per
    ``blocks.MLP_ROWS`` rows unless its block's swiglu ran on the GLU
    kernel, which reads fc1's weight past the leaf."""
    from gmlx.image.lora import install

    got: Counter = Counter((block, op)
                           for block, _, op, _ in qg.op_sites(sites))
    text = lay.span("text").rows
    nseg = sum(1 for s in lay.segments if s.rows)
    want = {}
    for name, kind in block_names(model).values():
        rows = lay.rows if kind == "block" else text
        for op, n in block_ops(kind, rows, nseg, chunk).items():
            want[(name, op)] = n
    out = sorted(f"{k}: {got.get(k, 0)} of {n}" for k, n in want.items()
                 if got.get(k, 0) != n)
    out += sorted(f"{k}: not a known op call" for k in got if k not in want)
    glu = {k[0] for k, v in routed.items() if v == bg.KERNEL_OPS["glu"]}
    leaves = {}
    for p in install.linear_leaves(model):
        if p in skipped:
            continue
        if p.endswith(".mlp.fc1"):
            block = p.removesuffix(".mlp.fc1")
            rows = text if block.startswith("token_refiner.") else lay.rows
            leaves[p] = 0 if block in glu else math.ceil(rows / blocks.MLP_ROWS)
        else:
            leaves[p] = 1
    read: Counter = Counter(k[0] for k in sites if len(k) == 2)
    out += sorted(f"leaf {p}: {read.get(p, 0)} of {n}"
                  for p, n in leaves.items() if read.get(p, 0) != n)
    out += sorted(f"leaf {p}: not a leaf a step calls" for p in read
                  if p not in leaves)
    return out
