"""The rows of a MOSS-Image step against a check 3 fixture of
scripts/image_oracle/check3_moss_image.py: free running from the
fixture's inputs, and teacher-forced for each text block, each block and
the head from the fixture's float32 input to it."""
from __future__ import annotations

import math
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import block_gates as bg
import qwen_gates as qg

from gmlx.image.families.moss_image import model as mm

TIME_ROWS = ("time_embedding", "shared_block_modulation",
             "video_head_modulation")
HEAD = "final_linear_video"


def tiny_model(weights: dict, config: dict, act=mx.float32):
    """The DiT of float32 weights by name, in torch order."""
    m = mm.MossImageDiT(config)
    m.load_weights([(k, mx.array(np.asarray(v, np.float32)))
                    for k, v in weights.items()])
    m.act_dtype = act
    m.eval()
    return m


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def rel(got, ref) -> float:
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


def _inputs(arrays: dict):
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    sigma = np.reshape(np.float32(arrays["sigma"]), (1,))
    return lat, feats, sigma


def free_rows(model, arrays: dict, mod=None) -> dict:
    """Every row of one step from the fixture's inputs, and the
    velocity."""
    lat, feats, sigma = _inputs(arrays)
    taps: dict = {}
    out = dict(zip(TIME_ROWS, model.time_vectors(sigma, 1)))
    vel = model(lat, sigma, feats, taps, mod=mod)
    out.update(taps)
    got = {k: _np(v) for k, v in out.items()}
    got["velocity"] = _np(vel)
    return got


def fixture_modulation(model, arrays: dict) -> mm.StepModulation:
    """The step's modulation from the fixture's two modulation rows, as
    the reference formed them."""
    act = model.act_dtype

    def chunks(rows, n):
        a = mx.array(rows).astype(act)[:, None, :]
        return tuple(mx.split(a, n, axis=-1))

    shift, scale = chunks(arrays["video_head_modulation"], 2)
    return mm.StepModulation(
        blocks=chunks(arrays["shared_block_modulation"], 6), shift=shift,
        scale=scale)


def forced_rows(model, arrays: dict) -> dict:
    """Each text block, each block and the head on the fixture's float32
    input to it, with the fixture's modulation."""
    act = model.act_dtype
    mod = fixture_modulation(model, arrays)
    n_txt = int(arrays["features"].shape[0])
    _, _, h, w = arrays["latents"].shape
    out = {}
    cos, sin = model.text_rope(n_txt)
    for i, blk in enumerate(model.text_embedder.blocks):
        src = (arrays["text_embedder.proj"] if i == 0
               else arrays[f"text_embedder.blocks.{i - 1}"])
        y = blk(mx.array(src).astype(act), cos, sin)
        out[f"text_embedder.blocks.{i}"] = _np(y)
    cos, sin = model.rope_tables(h, w, n_txt)
    for i, blk in enumerate(model.layers):
        src = arrays["in.layers.0"] if i == 0 else arrays[f"layers.{i - 1}"]
        y = blk(mx.array(src).astype(act), mod.blocks, cos, sin)
        out[f"layers.{i}"] = _np(y)
    y = model.head(mx.array(arrays["in.final_norm_video"]).astype(act),
                   mod.shift, mod.scale)
    out[HEAD] = _np(y)
    return out


def measures(model, arrays: dict) -> dict:
    """Relative L2 of every row against the fixture: teacher-forced for
    the text blocks, the blocks and the head, free running for the rest,
    and ``free <row>`` for each teacher-forced row run free."""
    free = free_rows(model, arrays)
    forced = forced_rows(model, arrays)
    got = {}
    for k, v in free.items():
        if k == "velocity":
            got[k] = rel(v, arrays["velocity"])
        elif k in forced:
            got[k] = rel(forced[k], arrays[k])
            got[f"free {k}"] = rel(v, arrays[k])
        elif k in arrays:
            got[k] = rel(v, arrays[k])
    return got


def missed(got: dict, tolerances: dict) -> dict:
    """The measures over their tolerance, by name. A ``free`` measure
    has no tolerance of its own and is not gated."""
    out = {}
    for k, v in got.items():
        if k.startswith("free "):
            continue
        tol = tolerances[k]["rel_l2"]
        if not v <= tol:
            out[k] = (v, tol)
    return out


# The exact gate: the step in the oracle's fp32 arithmetic.

def exact_floats(model, path: str) -> int:
    """Every float tensor of the file into the model in float32, where the
    load keeps the activation dtype. Run it before a patch table wraps
    the leaves, so the names are the file's."""
    import gguf
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    reader = gguf.GGUFReader(path)
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    pairs = []
    for t in reader.tensors:
        if t.tensor_type not in floats or t.name not in params:
            continue
        a = np.array(t.data, np.float32).reshape(params[t.name].shape)
        pairs.append((t.name, mx.array(a)))
    model.load_weights(pairs, strict=False)
    return len(pairs)


@contextmanager
def _exact(model):
    """The oracle's arithmetic: every K-quant product dense on its
    dequantized weight, each one checked at the exit."""
    with bg.exact_products(model) as used:
        yield
        bg.assert_all_dense(model, used)


def exact_measures(model, arrays: dict) -> dict:
    """``measures`` in the oracle's arithmetic."""
    with _exact(model):
        return measures(model, arrays)


def forced_row(name: str) -> bool:
    return (name.startswith(("layers.", "text_embedder.blocks."))
            or name == HEAD)


def free_row(name: str) -> bool:
    """A free-running row of ``measures``: every measure but the
    teacher-forced rows and the velocity."""
    return not forced_row(name) and name != "velocity"


# The exact gate's free-running rows and the velocity carry fp32 rounding
# that no other summation order reproduces. Each fixture records gmlx's
# own fp32 floor on the free rows (the worst of them) and on the
# velocity: the relative L2 between the exact step on the GPU and the
# same step on the CPU stream. The free rows and the velocity each take
# the larger of EXACT_TOL and twice their floor. The teacher-forced rows
# keep EXACT_TOL.

def exact_free_floor(model, arrays: dict) -> dict:
    """Relative L2 of each free row of ``measures`` and of the velocity
    between the exact step on the GPU and on the CPU stream."""
    with _exact(model):
        gpu = free_rows(model, arrays)
        with mx.stream(mx.cpu):
            cpu = free_rows(model, arrays)
    out = {}
    for k, v in gpu.items():
        name = f"free {k}" if forced_row(k) else k
        if k in arrays and not forced_row(name):
            out[name] = rel(v, cpu[k])
    return out


# The kinds of measure with a floor: manifest result and tolerance keys.
FLOORS = {"free": ("exact_free_floor", "exact_free"),
          "velocity": ("exact_velocity_floor", "exact_velocity")}


def exact_tols(m: dict) -> dict:
    """The free-row and velocity tolerances a manifest records, each of
    which must follow the rule from the floor beside it."""
    out = {}
    for kind, (res, key) in FLOORS.items():
        floor = m["results"][res]["rel_l2"]
        tol = m["tolerances"][key]["rel_l2"]
        assert tol == max(bg.EXACT_TOL, 2 * floor), (kind, tol, floor)
        out[kind] = tol
    return out


def check_floor(m: dict, floor: dict, label: str) -> dict:
    """The floors measured now are the floors the manifest records.
    Returns the manifest's tolerances."""
    free = {k: v for k, v in floor.items() if free_row(k)}
    top = max(free, key=free.get)
    now = {"free": free[top], "velocity": floor["velocity"]}
    print(f"{label}: fp32 floor of the free rows {now['free']!r} at {top}, "
          f"of the velocity {now['velocity']!r}")
    for kind, (res, _) in FLOORS.items():
        assert res in m["results"], f"{m['name']} records no {res}"
    tols = exact_tols(m)
    for kind, (res, _) in FLOORS.items():
        rec = m["results"][res]["rel_l2"]
        print(f"{label}: recorded {kind} floor {rec:.2e}, gate "
              f"{tols[kind]:.2e}")
        assert now[kind] <= 1.25 * rec, (kind, now[kind], rec)
    return tols


def exact_gate(name: str, tols: dict) -> float:
    if name == "velocity":
        return tols["velocity"]
    return tols["free"] if free_row(name) else bg.EXACT_TOL


def exact_over(got: dict, tols: dict) -> dict:
    return {k: (v, exact_gate(k, tols)) for k, v in got.items()
            if not v <= exact_gate(k, tols)}


def exact_line(label: str, got: dict, tols: dict) -> str:
    def worst(keys):
        k = max(keys, key=got.get)
        return f"{got[k]:.2e} at {k}"
    forced = [k for k in got if forced_row(k)]
    free = [k for k in got if free_row(k)]
    return (f"{label}: teacher-forced rows within {worst(forced)}, gate "
            f"{bg.EXACT_TOL:.0e}; velocity {got['velocity']:.2e}, gate "
            f"{tols['velocity']:.2e}; free rows within {worst(free)}, gate "
            f"{tols['free']:.2e}")


def fault_line(label: str, bad: dict, tols: dict) -> tuple[str, float]:
    """Where a wrong model misses the exact gate, by kind of row, and the
    largest ratio of a measure to its gate."""
    parts = []
    for kind, keep in (("teacher-forced", forced_row),
                       ("velocity", lambda k: k == "velocity"),
                       ("free", free_row)):
        r = {k: bad[k] / exact_gate(k, tols) for k in bad if keep(k)}
        miss = [k for k, v in r.items() if v > 1]
        if miss:
            top = max(miss, key=r.get)
            parts.append(f"{kind} {len(miss)} rows, up to {r[top]:.0f} of "
                         f"the gate at {top}")
    top = max(bad[k] / exact_gate(k, tols) for k in bad)
    return (f"{label}: misses through " + ("; ".join(parts) or "nothing"),
            top)


# The site gate: each op of a block on the path a request takes, against
# the stock op in float32 on the same production input, within 2 n u.
# The float32 side runs on the CPU, whose float32 products are exact. A
# reading is (block, "step", op, call number) for an op of a block, with
# ``qg.HEAD`` after it for the worst head row of the q and k site, and
# (module path, call number) for a linear leaf, so the helpers of
# qwen_gates.py read it. n counts the outputs the stock ops of a site
# write in the activation dtype, and a site that a kernel route ran is
# held to the kernel's own count, ``bg.KERNEL_ROUNDINGS``.
#
# The sites of a MOSS-Image block: the two norms, which round the
# normalized value and then the weight product; the two modulations; q
# and k through the q and k seam, in the dims of ``mm.interleave``; sdpa;
# the feed-forward from its input to the rows of its down leaf, situ,
# which rounds once over the rounding of its leaf; the two gated residual
# adds, or the residual op that also takes the leaf before it; and every
# linear leaf. A text block has the norms, the q and k seam, sdpa and
# situ, and adds its residuals in line. The shift of each modulation is
# added in line, and the exact gate holds it.
OP_ROUNDINGS = {
    "modulate": 2,        # 1 + scale, then the product
    "qk": 2,              # the norm, then the cast after the fp32 rotation
    "sdpa": 1,
    "situ": 1,            # float32, rounded once
    "gated_residual": 2,  # gate * y, then the sum
}
# The calls of each op in one call of a block, by kind of block.
BLOCK_OPS = {"block": {"norm": 2, "modulate": 2, "qk": 1, "sdpa": 1,
                       "situ": 1, "gated_residual": 2},
             "text": {"norm": 2, "qk": 1, "sdpa": 1, "situ": 1}}
# The leaves a step reads past their call: the time vectors run as
# float32 products on the dequantized weight (``ops.products.precise``).
NOT_CALLED = ("time_embedding.0", "time_embedding.2",
              "shared_block_modulation", "video_head_modulation")


def block_names(model) -> dict:
    """id of each block of ``model`` to (name, kind)."""
    out = {id(b): (f"layers.{i}", "block")
           for i, b in enumerate(model.layers)}
    for i, b in enumerate(model.text_embedder.blocks):
        out[id(b)] = (f"text_embedder.blocks.{i}", "text")
    return out


@contextmanager
def step_sites(model, sites: dict, routed: dict | None = None,
               heads: dict | None = None, built=None,
               skipped: set | None = None):
    """Fill ``sites`` with a reading for every op of every block and every
    call of a linear leaf while steps run: key to (relative L2, gate).
    ``routed`` takes the mlx-kquant op of each site a kernel route ran,
    ``heads`` the place of the worst head row of each q and k site, as
    (0, batch, head, token). ``built`` is the built table of the patch
    table the step runs under, from ``install.build_table`` in the block
    form, and gives the exact side each leaf's delta. ``skipped`` takes
    the path of a leaf called on a float32 input, which runs in float32
    and has no site."""
    import mlx_kquant as kq

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
    bases = {id(m): p for p, m in install.linear_leaves(model).items()}

    def factors(path):
        g = None if built is None else built.get(path)
        return None if g is None else (g.a_t.astype(f), g.b_t.astype(f))

    def exact_leaf(mod, x32, path):
        """The leaf's product in float32 on its dequantized weight, with
        its delta."""
        base = getattr(mod, "base", mod)
        codec = getattr(base, "kquant_type", None)
        w = base["weight"].astype(f) if codec is None else \
            kq.dequantize(base["weight"], base["scales"], codec, f)
        y = x32 @ w.T
        if "bias" in base:
            y = y + base["bias"].astype(f)
        fac = factors(path)
        if fac is not None:
            y = y + (x32 @ fac[0]) @ fac[1]
        return y

    def leaf_rounds(mod, path) -> int:
        base = getattr(mod, "base", mod)
        n = 2 if getattr(base, "kquant_type", None) is not None \
            and "bias" in base else 1
        return n + (2 if factors(path) is not None else 0)

    def read(op, got, want, rounds, by_head=False):
        key = (cur[-1], "step", op)
        n = counts[key] = counts.get(key, 0) + 1
        outs = got if isinstance(got, tuple) else (got,)
        same = all(o.dtype == act for o in outs)
        gate = 2 * rounds * qg.SITE_UNIT * same
        with mx.stream(mx.cpu):
            exact = want()
            rel = qg._rel_l2(got, exact)
            if by_head:
                e2, r2 = bg._head_sums(outs, exact if isinstance(
                    exact, tuple) else (exact,))
                mx.eval(e2, r2)
        sites[(*key, n)] = (rel, gate)
        if by_head:
            worst, at = bg._worst(np.array(e2), np.array(r2))
            sites[(*key, n, qg.HEAD)] = (worst, gate)
            if heads is not None:
                heads[(*key, n)] = at
        return (*key, n)

    def leaf_site(path, rel, gate):
        n = calls[path] = calls.get(path, 0) + 1
        sites[(path, n)] = (rel, gate)

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
                bg.KERNEL_ROUNDINGS["qk"] if kernel else OP_ROUNDINGS["qk"],
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

    def block_hidden(orig):
        def run(blk, h):
            y = orig(blk, h)
            if not cur:
                return y
            lin = blk.proj["ffn_up_gate"]
            path = paths[id(lin)]

            def want():
                hid, gate = mx.split(exact_leaf(lin, h.astype(f), path), 2,
                                     axis=-1)
                return mm.situ(hid, gate)

            read("situ", y, want,
                 OP_ROUNDINGS["situ"] + leaf_rounds(lin, path))
            return y
        return run

    def text_hidden(orig):
        def run(blk, h):
            y = orig(blk, h)
            if not cur:
                return y
            pu, pg = paths[id(blk.ffn_up)], paths[id(blk.ffn_gate)]

            def want():
                x32 = h.astype(f)
                return mm.situ(exact_leaf(blk.ffn_up, x32, pu),
                               exact_leaf(blk.ffn_gate, x32, pg))

            read("situ", y, want, OP_ROUNDINGS["situ"]
                 + leaf_rounds(blk.ffn_up, pu)
                 + leaf_rounds(blk.ffn_gate, pg))
            return y
        return run

    def gated_residual(orig):
        def run(x, gate, y):
            out = orig(x, gate, y)
            if cur:
                read("gated_residual", out, lambda: epilogues.gated_residual(
                    x.astype(f), gate.astype(f), y.astype(f)),
                    OP_ROUNDINGS["gated_residual"])
            return out
        return run

    def resid(orig):
        # The residual route's op, ``residual + gate * (x @ W^T)`` in one
        # call: the block's gated residual add and the call of its leaf.
        # From 1024 rows the kernel rounds once, below that as the stock
        # product and gated add do.
        def run(x, leaf, residual, gate):
            out = orig(x, leaf, residual, gate)
            if not cur:
                return out
            rows = math.prod(x.shape[:-1])
            rounds = 1 if rows >= qg.RESID_ONE_ROUNDING_ROWS else 3

            def want():
                w = kq.dequantize(leaf["weight"], leaf["scales"],
                                  leaf.kquant_type, f)
                return epilogues.gated_residual(
                    residual.astype(f), gate.astype(f), x.astype(f) @ w.T)

            key = read("gated_residual", out, want, rounds)
            if routed is not None:
                routed[key] = "quantized_matmul_residual"
            leaf_site(bases[id(leaf)], *sites[key])
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
                rel = qg._rel_l2(y, exact_leaf(mod, x.astype(f), path))
            rounds = leaf_rounds(mod, path)
            kernel = "lora" in ran[before:]
            if kernel:
                rounds = bg.KERNEL_ROUNDINGS["lora"]
            leaf_site(path, rel, 2 * rounds * qg.SITE_UNIT)
            if kernel and routed is not None:
                routed[(path, calls[path])] = bg.KERNEL_OPS["lora"]
            return y
        return run

    leaf_types = ({install.ImageLoRALinear} if wrapped else
                  {type(m) for m in install.linear_leaves(model).values()})
    hooks = [(mm.Block, "__call__", within),
             (mm.TextBlock, "__call__", within),
             (norms.RMSNorm, "__call__", norm),
             (mm.Block, "hidden", block_hidden),
             (mm.TextBlock, "hidden", text_hidden),
             (ops, "modulate", modulate), (ops, "qk_norm_rope", qk),
             (ops, "sdpa", sdpa), (ops, "gated_residual", gated_residual),
             (ops, "resid_fused", resid)]
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


def missing_ops(model, sites: dict, steps: int = 1, skipped=()) -> list:
    """What ``steps`` steps leave unread: each block calls every op of
    ``BLOCK_OPS`` for its kind, and every linear leaf but those of
    ``NOT_CALLED`` and ``skipped`` is called once a step."""
    from gmlx.image.lora import install

    got: dict = {}
    for block, _, op, _ in qg.op_sites(sites):
        got[(block, op)] = got.get((block, op), 0) + 1
    want = {}
    for name, kind in block_names(model).values():
        for op, n in BLOCK_OPS[kind].items():
            want[(name, op)] = n * steps
    out = sorted(f"{k}: {got.get(k, 0)} of {n}" for k, n in want.items()
                 if got.get(k, 0) != n)
    out += sorted(f"{k}: not a known op call" for k in got if k not in want)
    leaves = {p for p in install.linear_leaves(model)
              if p not in NOT_CALLED and p not in skipped}
    read: dict = {}
    for k in sites:
        if len(k) == 2:
            read[k[0]] = read.get(k[0], 0) + 1
    out += sorted(f"leaf {p}: {read.get(p, 0)} of {steps}" for p in leaves
                  if read.get(p, 0) != steps)
    out += sorted(f"leaf {p}: not a leaf a step calls" for p in read
                  if p not in leaves)
    return out
