#!/usr/bin/env python
"""Check the owned DFlash 2 conv and selector against the z-lab reference on
real drafter weights, and record the reference outputs as a test fixture.

    python scripts/dflash2_reference_check.py DRAFTER.gguf \
        --reference /path/to/z-lab/dflash/model_mlx.py \
        --fixture tests/fixtures/dflash2_reference_qwen38.npz

The reference module is not vendored: its conv and selector classes are
extracted from the given file (or the installed ``dflash`` package) and run
as-is. Layer 0's two convolutions and one block of the selector are compared;
the fixture stores the inputs and the reference outputs in float32 so the
pinned test needs only the drafter GGUF, not the reference.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import numpy as np

from gmlx.dflash_drafter import CandidateSelector, GroupedDynamicConv, greedy_walk

_REF_NAMES = ("_sampling_probs", "_sample_probs", "_grouped_dynamic_convolve",
              "GroupedDynamicCausalConv", "CandidateSelector")
_TOL = 1e-2


def load_reference(path: str | None) -> dict:
    if path is None:
        spec = importlib.util.find_spec("dflash.model_mlx")
        if spec is None or not spec.origin:
            sys.exit("no --reference path and the dflash package is not installed")
        path = spec.origin
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _REF_NAMES]
    missing = set(_REF_NAMES) - {n.name for n in keep}
    if missing:
        sys.exit(f"{path}: reference is missing {sorted(missing)}")
    ns = {"mx": mx, "nn": nn}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def dequant(arrays: dict, codecs: dict, name: str, dtype=mx.bfloat16) -> mx.array:
    codec = codecs.get(name)
    if codec is None:
        return arrays[name].astype(dtype)
    prefix = name[:-len(".weight")] if name.endswith(".weight") else name
    return kq.dequantize(arrays[name], arrays[prefix + ".scales"], codec, dtype=dtype)


def rel_diff(a: mx.array, b: mx.array) -> tuple[float, float]:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    mx.eval(a32, b32)
    diff = float(mx.abs(a32 - b32).max().item())
    scale = float(mx.abs(b32).max().item()) or 1.0
    return diff, diff / scale


def canonical(cands: mx.array, first: mx.array, edges: mx.array):
    """Reorder every position's candidates ascending by token id so two
    argpartition orders compare equal. Returns numpy float32 arrays."""
    c = np.array(cands.tolist())
    f = np.array(first.astype(mx.float32).tolist())
    e = np.array(edges.astype(mx.float32).tolist())
    order = np.argsort(c, axis=-1)
    c2 = np.take_along_axis(c, order, axis=-1)
    f2 = f[order[0]]
    e2 = np.empty_like(e)
    for p in range(e.shape[0]):
        e2[p] = e[p][order[p]][:, order[p + 1]]
    return c2, f2, e2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    ref = load_reference(args.reference)
    arrays, codecs, meta, _shapes = kq.load_gguf(args.gguf)
    H = int(meta["dflash.embedding_length"])
    K = int(meta["dflash.conv_kernel_size"])
    G = int(meta["dflash.conv_group_size"])
    rank = int(meta["dflash.selector_rank"])
    top_k = int(meta["dflash.selector_top_k"])
    block = args.block or int(meta["dflash.block_size"])
    L = block - 1
    pred_w = dequant(arrays, codecs, "selector_predecessor.weight")
    V = int(pred_w.shape[0])
    print(f"hidden={H} kernel={K} group={G} rank={rank} top_k={top_k} "
          f"block={block} vocab={V}")

    mx.random.seed(args.seed)
    failures = []
    out = {"block": block, "temp": args.temp, "seed": args.seed}

    def check(label, own, theirs):
        d, r = rel_diff(own, theirs)
        flag = "" if r <= _TOL else "  <-- FAIL"
        print(f"  {label:28s} max_abs={d:.3e} rel={r:.3e}{flag}")
        if r > _TOL:
            failures.append(label)

    for tag in ("attn", "ffn"):
        base = dequant(arrays, codecs, f"blk.0.{tag}_conv_base")
        proj = dequant(arrays, codecs, f"blk.0.{tag}_conv_proj.weight")
        theirs = ref["GroupedDynamicCausalConv"](H, K, G)
        theirs.base_kernel = base
        theirs.kernel_projection.weight = proj
        own = GroupedDynamicConv(H, K, G)
        own.base_kernel = base
        own.kernel_projection.weight = proj
        x = mx.random.normal((1, block, H)).astype(mx.bfloat16)
        z = mx.random.normal((1, block, H)).astype(mx.bfloat16)
        y_t, dyn_t = theirs.prepare(x)
        y_o, dyn_o = own.prepare(x)
        fin_t = theirs.finish(z, dyn_t)
        fin_o = own.finish(z, dyn_o)
        print(f"{tag}_conv (layer 0):")
        check("prepare output", y_o, y_t)
        check("prepare dynamic kernel", dyn_o, dyn_t)
        check("finish output", fin_o, fin_t)
        for k, v in (("x", x), ("y", y_t), ("dyn", dyn_t), ("z", z), ("fin", fin_t)):
            out[f"{tag}_{k}"] = np.array(v.astype(mx.float32).tolist(), dtype=np.float32)

    cfg = SimpleNamespace(selector_top_k=top_k, vocab_size=V,
                          selector_rank=rank, hidden_size=H)
    theirs = ref["CandidateSelector"](cfg)
    own = CandidateSelector(H, V, rank, top_k)
    succ_w = dequant(arrays, codecs, "selector_successor.weight")
    hid_w = dequant(arrays, codecs, "selector_hidden.weight")
    for m in (theirs, own):
        m.predecessor_codebook.weight = pred_w
        m.successor_codebook.weight = succ_w
        m.hidden_projection.weight = hid_w
    hidden = mx.random.normal((1, L, H)).astype(mx.bfloat16)
    logits = (mx.random.normal((1, L, V)) * 4.0)
    anchor = mx.random.randint(0, V, (1,))
    cands, first, edges = own.lattice(hidden[0], logits[0], anchor[0])
    path_o = greedy_walk(cands, first, edges)
    path_t, cands_t, _ = theirs.select(hidden, logits, anchor, 0.0)
    mx.eval(path_o, path_t, cands_t)
    print("selector (greedy):")
    same = path_o.tolist() == path_t[0].tolist()
    print(f"  greedy path {'identical' if same else 'DIFFERS'}: "
          f"own={path_o.tolist()} ref={path_t[0].tolist()}")
    if not same:
        failures.append("greedy path")
    c_own = sorted(cands.tolist()[0])
    c_ref = sorted(cands_t[0].tolist()[0])
    if c_own != c_ref:
        failures.append("candidates")
        print("  candidate sets differ at position 0")

    mx.random.seed(args.seed + 1)
    path_s, cands_s, q_rows = theirs.select(hidden, logits, anchor, args.temp)
    mx.eval(path_s, cands_s, q_rows)
    print(f"selector (sampled, temp {args.temp}), q along the reference path:")
    q_own = []
    prev = None
    for p in range(L):
        if p == 0:
            row = first
        else:
            i = cands[p - 1].tolist().index(prev)
            row = edges[p - 1][i]
        q_own.append(ref["_sampling_probs"](row[None], args.temp)[0])
        prev = int(path_s[0, p].item())
    q_own = mx.stack(q_own)
    # the reference's q rows are in its own candidate order; map to ours
    q_ref = []
    for p in range(L):
        ours = cands[p].tolist()
        ref_order = cands_s[0, p].tolist()
        perm = [ref_order.index(t) for t in ours]
        q_ref.append(q_rows[0, p][mx.array(perm)])
    q_ref = mx.stack(q_ref)
    check("q rows", q_own, q_ref)

    c2, f2, e2 = canonical(cands, first, edges)
    unary = np.array(mx.take_along_axis(logits[0], cands, axis=-1).tolist(),
                     dtype=np.float32)
    unary = np.take_along_axis(unary, np.argsort(np.array(cands.tolist()), -1), -1)
    out.update({
        "sel_hidden": np.array(hidden[0].astype(mx.float32).tolist(), dtype=np.float32),
        "sel_cands": c2.astype(np.int64), "sel_unary": unary,
        "sel_anchor": int(anchor[0].item()),
        "sel_first": f2, "sel_edges": e2,
        "sel_greedy_path": np.array(path_t[0].tolist(), dtype=np.int64),
        "sel_sampled_path": np.array(path_s[0].tolist(), dtype=np.int64),
        "sel_sampled_q": np.array(q_ref.astype(mx.float32).tolist(), dtype=np.float32),
    })
    if args.fixture:
        np.savez_compressed(args.fixture, **out)
        print(f"fixture written: {args.fixture}")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("all checks within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
