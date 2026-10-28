"""The rows of a MiniMax-H3 step against a check 3 fixture of
scripts/video_oracle/check3_h3.py, under three gates.

The product gate runs the step as it ships: free running from the
fixture's inputs, and teacher-forced for each refiner block, the
refiner's norm, each block and the heads from the fixture's float32 input
to them, each row within twice the oracle's own gap. The exact gate runs
it in the oracle's fp32 arithmetic. The site gate reads each op of every
block and each linear leaf against the stock op in float32 on the same
input. The step's layout, angles, segment rows, times and curve rows are
held to the oracle's own."""
from __future__ import annotations

import json
import math
from collections import Counter
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import block_gates as bg
import qwen_gates as qg
from block_gates import tf32

from gmlx.image import pacing
from gmlx.video import api, layout, rope, schedule
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm

VELOCITIES = ("velocity_video", "velocity_audio")
HEADS = ("final_layer.video_out", "final_layer.audio_out")
# Rows outside the blocks that every fixture holds.
TOP_ROWS = ("condition_proj", "video_patch_proj", "audio_patch_proj",
            "token_refiner.final_norm", "final_layer.adaln_proj.linear",
            *HEADS)
# The oracle runs the patch projections and the head's modulation in
# float32 in both modes, so the manifest gives those rows a float32
# tolerance. Where MLX multiplies float32 at TF32 precision, its default
# on chips that have it, they take TF32_ROWS: the bound on one product's
# relative error with both operands cut to a 10 bit mantissa. A dtype
# test holds those leaves in float32, which this bound alone would not.
TF32_ROWS = 2.0 ** -9


def keeps_fp32(name: str) -> bool:
    """Whether the loader keeps parameter or leaf ``name`` in float32."""
    return any(s in name + "." for s in hf.FAMILY.fp32_keep)


def tiny_model(weights: dict, config: dict, act=mx.float32):
    """The DiT of float32 weights by name, in torch order, cast as the
    loader casts a file's float tensors: float32 under the family's
    fp32_keep, the activation dtype for the rest."""
    m = hm.H3Transformer(config)

    def cast(k, v):
        a = mx.array(np.asarray(v, np.float32))
        return a if keeps_fp32(k) else a.astype(act)

    m.load_weights([(k, cast(k, v)) for k, v in weights.items()])
    m.act_dtype = act
    m.eval()
    return m


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def rel(got, ref) -> float:
    r = np.asarray(ref, np.float64)
    g = np.asarray(got, np.float64).reshape(r.shape)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


def cosine(got, ref) -> float:
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    return float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r)))


def times_at(sigma) -> tuple[float, float]:
    """gmlx's video and audio times at a video sigma."""
    s = float(np.float32(sigma))
    return (api.stream_time(s), api.stream_time(schedule.time_shift(
        s, hf.SHIFTS["video"], hf.SHIFTS["audio"])))


def stream_times(arrays: dict) -> tuple[float, float]:
    """gmlx's video and audio times at the fixture's video sigma."""
    return times_at(arrays["sigma"])


def case(arrays: dict):
    """The fixture's step: latents, text features, gmlx's stream times and
    the layout gmlx builds for them."""
    video = mx.array(arrays["video"])
    audio = mx.array(arrays["audio"])
    feats = mx.array(arrays["features"])[None]
    _, _, t, h, w = video.shape
    lay = layout.pack(hf.TIMELINE, feats.shape[1], t, h, w, audio.shape[-1])
    return (video, audio, feats, *stream_times(arrays), lay)


def layout_misses(model, arrays: dict, m: dict) -> list[str]:
    """Where the step's times, layout, angles, segment rows or curve rows
    differ from the oracle's."""
    *_, tv, ta, lay = case(arrays)
    out = []
    if m["inputs"]["shifts"] != hf.SHIFTS:
        out.append(f"shifts {m['inputs']['shifts']}")
    if (tv, ta) != (m["inputs"]["t_video"], m["inputs"]["t_audio"]):
        out.append(f"stream times {(tv, ta)}")
    plan = model.plan(lay, tv, ta)
    kinds = [s.kind for s in lay.segments if s.rows]
    if kinds != m["inputs"]["segment_kinds"]:
        out.append(f"segment kinds {kinds}")
    if not np.array_equal(lay.positions, arrays["positions"]):
        out.append("positions")
    # The oracle's angles repeat its rotated half: rotate-half pairs dims
    # i and i + 48 at one angle.
    ang = np.array(rope.angles(lay.positions, model.rope.inv_freq))
    ref = arrays["rope.angles"]
    w = ang.shape[1]
    if not (ref.shape == (ang.shape[0], 2 * w)
            and np.array_equal(ref[:, :w], ang)
            and np.array_equal(ref[:, w:], ang)):
        out.append("rope angles")
    if plan.segs != tuple(tuple(r) for r in arrays["segments"].tolist()):
        out.append(f"segments {plan.segs}")
    if plan.times != tuple(arrays["t_values"].tolist()):
        out.append(f"times {plan.times}")
    t_emb = np.array(model.curve(plan.times))
    if not np.allclose(t_emb, arrays["t_emb"], rtol=1e-6, atol=0):
        out.append("curve rows")
    return out


def _eval_blocks(arrays) -> None:
    mx.eval(*arrays)


def free_rows(model, arrays: dict) -> dict:
    """Every tapped row of one step from the fixture's inputs, and the
    velocities. Each block evaluates before the next, as a run does."""
    video, audio, feats, tv, ta, lay = case(arrays)
    taps: dict = {}
    with pacing.paced(_eval_blocks):
        text = model.caption(feats, taps)
        vv, va = model(video, audio, tv, ta, text, lay, taps=taps)
    got = {k: _np(v) for k, v in taps.items()}
    got["velocity_video"], got["velocity_audio"] = _np(vv), _np(va)
    return got


def forced_rows(model, arrays: dict) -> dict:
    """Each refiner block, the refiner's norm, each block the fixture
    holds and the heads on the fixture's float32 input to them."""
    act = model.act_dtype
    *_, tv, ta, lay = case(arrays)

    def fed(name):
        return mx.array(arrays[name])[None].astype(act)

    out = {}
    tr = model.token_refiner
    src = "condition_proj"
    for i, blk in enumerate(tr.blocks):
        out[f"token_refiner.blocks.{i}"] = _np(blk(fed(src)))
        src = f"token_refiner.blocks.{i}"
    out["token_refiner.final_norm"] = _np(tr.final_norm(fed(src)))
    plan = model.plan(lay, tv, ta)
    t_emb = model.curve(plan.times)
    src = "in.blocks.0"
    for i, blk in enumerate(model.blocks):
        y = blk(fed(src), blk.modulation(t_emb, act), plan.segs,
                plan.cos, plan.sin)
        out[f"blocks.{i}"] = _np(y)
        src = f"blocks.{i}"
    v, a = model.head(fed("in.final_layer"), t_emb, plan, lay)
    out[HEADS[0]], out[HEADS[1]] = _np(v), _np(a)
    return out


def forced_row(name: str) -> bool:
    return (name.startswith(("blocks.", "token_refiner."))
            or name in HEADS)


def measures(model, arrays: dict, m: dict) -> dict:
    """Relative L2 of every row the manifest holds against the fixture:
    teacher-forced where the oracle forced it, free running for the rest,
    and ``free <row>`` for each teacher-forced row run free. ``cosine
    <velocity>`` for each velocity."""
    free = free_rows(model, arrays)
    forced = forced_rows(model, arrays)
    tols = m["tolerances"]
    got = {}
    for k, v in free.items():
        if k not in tols or k not in arrays:
            continue
        if forced_row(k):
            got[k] = rel(forced[k], arrays[k])
            got[f"free {k}"] = rel(v, arrays[k])
        else:
            got[k] = rel(v, arrays[k])
        if k in VELOCITIES:
            got[f"cosine {k}"] = cosine(v, arrays[k])
    return got


def gate(tol: dict) -> float:
    """A row's gate in this process."""
    if tol.get("float32") and tf32():
        return TF32_ROWS
    return tol["rel_l2"]


def missed(got: dict, tolerances: dict) -> dict:
    """The measures over their tolerance, by name."""
    out = {}
    for k, v in got.items():
        if k.startswith("cosine "):
            floor = tolerances[k.removeprefix("cosine ")]["cosine_min"]
            if not v >= floor:
                out[k] = (v, floor)
            continue
        tol = gate(tolerances[k])
        if not v <= tol:
            out[k] = (v, tol)
    return out


def expected_rows(m: dict) -> set:
    """The rows a check 3 fixture of ``m``'s depth holds a tolerance for:
    the rows outside the blocks, each refiner block and block, ``free``
    for each teacher-forced row, and the velocities."""
    config = m["inputs"].get("config")
    if config is None:
        depth, refiner = m["inputs"]["blocks"], hf._REFINER
    else:
        depth, refiner = config["num_layers"], config["num_refiner_layers"]
    rows = {*TOP_ROWS, *(f"token_refiner.blocks.{i}" for i in range(refiner)),
            *(f"blocks.{i}" for i in range(depth))}
    return rows | {f"free {k}" for k in rows if forced_row(k)} | \
        set(VELOCITIES)


def gated_rows(m: dict) -> set:
    """The measures a manifest gates: each row tolerance, and the cosine
    of each velocity."""
    rows = {k for k in m["tolerances"]
            if k != "rule" and not k.startswith("exact_")}
    return rows | {f"cosine {k}" for k in VELOCITIES}


def unmeasured(got: dict, m: dict) -> list[str]:
    want = gated_rows(m)
    return sorted([f"{k}: not measured" for k in want - set(got)]
                  + [f"{k}: no tolerance" for k in set(got) - want])


def worst_line(label: str, got: dict, m: dict) -> str:
    tols = m["tolerances"]
    ratio = {k: v / gate(tols[k]) for k, v in got.items()
             if not k.startswith("cosine ")}
    top = max(ratio, key=ratio.get)
    free = {k: r for k, r in ratio.items() if k.startswith("free ")}
    worst_free = max(free, key=free.get)
    vel = ", ".join(f"{k} {got[k]:.2e} of {tols[k]['rel_l2']:.2e}"
                    for k in VELOCITIES if k in got)
    return (f"{label}: {len(ratio)} rows, worst {got[top]:.2e} at {top}, "
            f"{ratio[top]:.2f} of its gate; worst free row {worst_free} "
            f"{free[worst_free]:.2f} of its gate; {vel}")


def kernel_census(model, lay, ran: list) -> tuple[Counter, Counter]:
    """The kernel calls one ``measures`` made, and the calls the routes in
    force must make: the q and k kernel in every block of the free and the
    forced pass, and the GLU kernel there too from its row floor on a GPU
    with tensor ops. The refiner's rows are below the floor."""
    from gmlx.image.ops import kq_routes

    on = kq_routes.active()
    n = 2 * len(model.blocks)
    want: Counter = Counter()
    if "qk" in on:
        want["qk"] = n
    if ("glu" in on and kq_routes._nax()
            and lay.rows >= kq_routes._GLU_MIN_ROWS):
        want["glu"] = n
    return Counter(ran), want


# The exact gate: the step in the oracle's fp32 arithmetic.

@contextmanager
def exact_floats(model):
    """Every float parameter outside the K-quant leaves in float32 while
    the block runs. The loader holds each in bfloat16 or float32 as the
    file stores it, so the cast is exact. Yields how many it cast."""
    from mlx.utils import tree_flatten

    quant = set(bg.kquant_leaves(model).values())
    saved = [(k, a) for k, a in tree_flatten(model.parameters())
             if k.rsplit(".", 1)[0] not in quant
             and mx.issubdtype(a.dtype, mx.floating)]
    model.load_weights([(k, a.astype(mx.float32)) for k, a in saved],
                       strict=False)
    try:
        yield sum(a.dtype != mx.float32 for _, a in saved)
    finally:
        model.load_weights(saved, strict=False)


@contextmanager
def _exact(model):
    """The oracle's arithmetic: every K-quant product dense on its
    dequantized weight, each one checked at the exit."""
    with bg.exact_products(model) as used:
        yield
        bg.assert_all_dense(model, used)


def exact_measures(model, arrays: dict, m: dict) -> dict:
    """``measures`` in the oracle's arithmetic, without the cosines."""
    with _exact(model):
        got = measures(model, arrays, m)
    return {k: v for k, v in got.items() if not k.startswith("cosine ")}


def free_row(name: str) -> bool:
    """A free-running row of ``measures`` other than a velocity."""
    return not forced_row(name) and name not in VELOCITIES


# The exact gate's free-running rows and the velocities carry fp32
# rounding, which the blocks amplify: two exact steps that differ only in
# rounding drift apart by an amount that varies severalfold from one
# rounding to the next. Each fixture records gmlx's own floor on the free
# rows (the worst of them) and on each velocity: the largest relative L2
# between the exact step on the GPU and the same step on the CPU stream
# or on inputs nudged by one float32 step (NUDGE_SEEDS). Each takes the
# larger of EXACT_TOL and twice its floor. The teacher-forced rows keep
# EXACT_TOL.
FLOORS = {"free": ("exact_free_floor", "exact_free"),
          **{v: (f"exact_{v}_floor", f"exact_{v}") for v in VELOCITIES}}
NUDGE_SEEDS = (0, 1, 2, 3)
# Set to 1 to record the floors measured now in the manifests, for new
# fixtures or another machine.
RECORD = "GMLX_VIDEO_RECORD_FLOORS"


def nudged(arrays: dict, seed: int) -> dict:
    """The fixture's inputs with each element moved one float32 step up,
    down or not at all, at random."""
    rng = np.random.default_rng(seed)
    out = dict(arrays)
    for k in ("video", "audio", "features"):
        a = np.asarray(arrays[k], np.float32)
        step = rng.integers(-1, 2, a.shape)
        out[k] = np.where(step == 0, a, np.nextafter(
            a, np.where(step > 0, np.inf, -np.inf).astype(np.float32)))
    return out


def exact_free_floor(model, arrays: dict, m: dict) -> dict:
    """The largest relative L2 of each free row of ``measures`` and of each
    velocity between the exact step on the GPU and the same step on the
    CPU stream or on nudged inputs."""
    out: dict = {}

    def take(other):
        for k, v in gpu.items():
            if k in m["tolerances"]:
                name = f"free {k}" if forced_row(k) else k
                out[name] = max(out.get(name, 0.0), rel(v, other[k]))

    with _exact(model):
        gpu = free_rows(model, arrays)
        for s in NUDGE_SEEDS:
            take(free_rows(model, nudged(arrays, s)))
        with mx.stream(mx.cpu):
            take(free_rows(model, arrays))
    return out


def exact_tols(m: dict) -> dict:
    """The free-row and velocity tolerances a manifest records, each of
    which must follow the rule from the floor beside it."""
    out = {}
    for kind, (res, key) in FLOORS.items():
        assert res in m["results"], (f"{m['name']} records no {res}: run "
                                     f"the exact gate with {RECORD}=1")
        floor = m["results"][res]["rel_l2"]
        tol = m["tolerances"][key]["rel_l2"]
        assert tol == max(bg.EXACT_TOL, 2 * floor), (kind, tol, floor)
        out[kind] = tol
    return out


def floors_now(floor: dict) -> dict:
    """The worst free-row floor, where it is, and each velocity's."""
    free = {k: v for k, v in floor.items() if free_row(k)}
    top = max(free, key=free.get)
    return {"free": free[top], "free_at": top,
            **{v: floor[v] for v in VELOCITIES}}


def check_floor(m: dict, floor: dict, label: str) -> dict:
    """The floors measured now are the floors the manifest records.
    Returns the manifest's tolerances."""
    now = floors_now(floor)
    print(f"{label}: fp32 floor of the free rows {now['free']!r} at "
          f"{now['free_at']}, of the velocities "
          + ", ".join(f"{v} {now[v]!r}" for v in VELOCITIES))
    tols = exact_tols(m)
    for kind, (res, _) in FLOORS.items():
        rec = m["results"][res]["rel_l2"]
        print(f"{label}: recorded {kind} floor {rec:.2e}, gate "
              f"{tols[kind]:.2e}")
        assert rec / 1.25 <= now[kind] <= 1.25 * rec, (
            f"{kind} floor {now[kind]:.3e} is not within 1.25x of the "
            f"recorded {rec:.3e}; on another machine, or after an mlx, "
            f"mlx-kquant or macOS update, run the exact gate with "
            f"{RECORD}=1")
    return tols


def record_floors(m: dict, floor: dict, paths) -> dict:
    """Write the floors measured now, and the tolerances the rule gives
    them, into each manifest at ``paths``. Returns the new manifest."""
    now = floors_now(floor)
    how = (f"the largest relative L2 between gmlx's exact step on the GPU "
           f"and the same step on the CPU stream or on inputs nudged by "
           f"one float32 step ({len(NUDGE_SEEDS)} seeds), {{}}; "
           f"tests/video/h3_gates.py exact_free_floor on "
           f"{m['files']['dit']['name']} {m['files']['dit']['sha256'][:8]}, "
           f"{mx.device_info()['device_name']}")
    res = {"exact_free_floor": {"rel_l2": now["free"], "row": now["free_at"],
                                "measured": how.format("worst free row")}}
    for v in VELOCITIES:
        res[f"exact_{v}_floor"] = {"rel_l2": now[v],
                                   "measured": how.format(v)}
    tols = {key: {"rel_l2": max(bg.EXACT_TOL, 2 * res[r]["rel_l2"])}
            for r, key in FLOORS.values()}
    for path in paths:
        doc = json.loads(path.read_text())
        assert doc["npz"] == m["npz"], path
        doc["results"].update(res)
        doc["tolerances"].update(tols)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(path)
    return {**m, "results": {**m["results"], **res},
            "tolerances": {**m["tolerances"], **tols}}


def exact_gate(name: str, tols: dict) -> float:
    if name in VELOCITIES:
        return tols[name]
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
    vel = "; ".join(f"{v} {got[v]:.2e}, gate {tols[v]:.2e}"
                    for v in VELOCITIES)
    return (f"{label}: teacher-forced rows within {worst(forced)}, gate "
            f"{bg.EXACT_TOL:.0e}; {vel}; free rows within {worst(free)}, "
            f"gate {tols['free']:.2e}")


def fault_line(label: str, bad: dict, tols: dict) -> tuple[str, float]:
    """Where a wrong model misses the exact gate, by kind of row, and the
    largest ratio of a measure to its gate."""
    parts = []
    for kind, keep in (("teacher-forced", forced_row),
                       ("velocity", lambda k: k in VELOCITIES),
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
# The sites of an H3 block: the two norms; a modulation and a gated
# residual add per segment at each of the two branches; q and k through
# the q and k seam, in the pair order of ``rope.pair_halves``; sdpa per
# chunk of query rows; the feed-forward from its input to the rows of fc2,
# swiglu, which rounds over the rounding of fc1 on the stock path; and
# every linear leaf. A refiner block also norms q and k as modules, has no
# RoPE and no modulation, and adds its residuals in line. The shift of
# each modulation is added in line, and the exact gate holds it.
OP_ROUNDINGS = {
    "modulate": 2,        # 1 + scale, then the product
    "sdpa": 1,
    "swiglu": 3,          # sigmoid(a), a times it, then the product with b
    "gated_residual": 2,  # gate * y, then the sum
}


def block_ops(kind: str, rows: int, segments: int) -> dict:
    """The calls of each op in one call of a block of ``kind`` over
    ``rows`` rows in ``segments`` segments."""
    if kind == "block":
        return {"norm": 2, "modulate": 2 * segments, "qk": 1,
                "sdpa": math.ceil(rows / hf.ATTENTION_CHUNK), "swiglu": 1,
                "gated_residual": 2 * segments}
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
               by_row: dict | None = None):
    """Fill ``sites`` with a reading for every op of every block and every
    call of a linear leaf while steps run: key to (relative L2, gate).
    ``routed`` takes the mlx-kquant op of each site a kernel route ran,
    ``heads`` the place of the worst head row of each q and k site, as
    (0, batch, head, token). ``skipped`` takes the path of a leaf called
    on a float32 input, which runs in float32 and has no site. ``by_row``
    takes the worst token row of each site on the GLU kernel, at the
    site's gate (block_gates.ROW_RULE)."""
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
    paths = {id(m): p for p, m in install.linear_leaves(model).items()}

    def exact_leaf(mod, x32):
        """The leaf's product in float32 on its dequantized weight."""
        codec = getattr(mod, "kquant_type", None)
        w = mod["weight"].astype(f) if codec is None else \
            kq.dequantize(mod["weight"], mod["scales"], codec, f)
        y = x32 @ w.T
        if "bias" in mod:
            y = y + mod["bias"].astype(f)
        return y

    def leaf_rounds(mod) -> int:
        quant = getattr(mod, "kquant_type", None) is not None
        return 2 if quant and "bias" in mod else 1

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

    def gated_residual(orig):
        def run(x, gate, y):
            out = orig(x, gate, y)
            if cur:
                read("gated_residual", out, lambda: epilogues.gated_residual(
                    x.astype(f), gate.astype(f), y.astype(f)),
                    OP_ROUNDINGS["gated_residual"])
            return out
        return run

    def leaf(orig):
        def run(mod, x, *a, **k):
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
            n = calls[path] = calls.get(path, 0) + 1
            sites[(path, n)] = (rel_l2, 2 * leaf_rounds(mod) * qg.SITE_UNIT)
            return y
        return run

    leaf_types = {type(m) for m in install.linear_leaves(model).values()}
    assert leaf_types <= {nn.Linear, KQuantLinear}, leaf_types
    hooks = [(hm.DiTBlock, "__call__", within),
             (hm.RefinerBlock, "__call__", within),
             (norms.RMSNorm, "__call__", norm),
             (hm._Block, "hidden", hidden),
             (ops, "modulate", modulate), (ops, "qk_norm_rope", qk),
             (ops, "sdpa", sdpa), (ops, "gated_residual", gated_residual)]
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


def missing_ops(model, sites: dict, lay, routed: dict,
                skipped=()) -> list:
    """What one step over ``lay`` leaves unread: each block calls every op
    of ``block_ops`` for its kind, and every linear leaf but those of
    ``skipped`` is called once, fc1 once per ``hm.MLP_ROWS`` rows unless
    its block's swiglu ran on the GLU kernel, which reads fc1's weight
    past the leaf."""
    from gmlx.image.lora import install

    got: Counter = Counter((block, op)
                           for block, _, op, _ in qg.op_sites(sites))
    text = lay.span("text").rows
    nseg = sum(1 for s in lay.segments if s.rows)
    want = {}
    for name, kind in block_names(model).values():
        rows = lay.rows if kind == "block" else text
        for op, n in block_ops(kind, rows, nseg).items():
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
            leaves[p] = 0 if block in glu else math.ceil(rows / hm.MLP_ROWS)
        else:
            leaves[p] = 1
    read: Counter = Counter(k[0] for k in sites if len(k) == 2)
    out += sorted(f"leaf {p}: {read.get(p, 0)} of {n}"
                  for p, n in leaves.items() if read.get(p, 0) != n)
    out += sorted(f"leaf {p}: not a leaf a step calls" for p in read
                  if p not in leaves)
    return out
