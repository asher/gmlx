"""The rows of a MiniMax-H3 step against a check 3 fixture of
scripts/video_oracle/check3_h3.py, under three gates.

The product gate runs the step as it ships: free running from the
fixture's inputs, and teacher-forced for each refiner block, the
refiner's norm, each block and the heads from the fixture's float32 input
to them, each row within twice the oracle's own gap. The exact gate runs
it in the oracle's fp32 arithmetic. The site gate reads each op of every
block and each linear leaf against the stock op in float32 on the same
input. The step's layout, angles, segment rows, times and curve rows are
held to the oracle's own. An anchor case's fixture holds each anchor's
latents, the noise the oracle mixed into its video rows and the prompt's
vision spans; the step takes them as a request does."""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np

import block_gates as bg
import h3_sites as hs
from block_gates import tf32

from gmlx.image import pacing
from gmlx.video import anchors as va
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


def anchored(arrays: dict) -> tuple:
    """The fixture's placed anchors, the oracle's noise of their video rows
    in order and the prompt's vision spans; none for a text to video
    step."""
    out, noise, i = [], [], 0
    while f"anchor.{i}.frame" in arrays:
        v, a = arrays.get(f"anchor.{i}.video"), arrays.get(f"anchor.{i}.audio")
        out.append(va.Anchor(int(arrays[f"anchor.{i}.frame"]),
                             None if v is None else mx.array(v),
                             None if a is None else mx.array(a)))
        if v is not None:
            noise.append(mx.array(arrays[f"anchor_noise.{len(noise)}"]))
        i += 1
    spans = tuple(tuple(r) for r in arrays["spans"].tolist()) \
        if "spans" in arrays else ()
    return tuple(out), noise, spans


def cond(model, arrays: dict) -> tuple:
    """(cond_video, cond_audio): the fixture's anchor rows as the step
    mixes them with the oracle's noise."""
    anchors, noise, _ = anchored(arrays)
    if not anchors:
        return (), ()
    return model.cond_rows(anchors, int(arrays["anchor_seed"]), noise=noise)


def case(arrays: dict):
    """The fixture's step: latents, text features, gmlx's stream times and
    the layout gmlx builds for them, with its anchors and vision spans."""
    video = mx.array(arrays["video"])
    audio = mx.array(arrays["audio"])
    feats = mx.array(arrays["features"])[None]
    _, _, t, h, w = video.shape
    anchors, _, spans = anchored(arrays)
    lay = layout.pack(hf.TIMELINE, feats.shape[1], t, h, w, audio.shape[-1],
                      anchors=va.layout_rows(anchors), spans=spans)
    return (video, audio, feats, *stream_times(arrays), lay)


def segment_kinds(lay) -> list:
    """The kinds of a layout's segments as the oracle's layout has them:
    the oracle's text segment holds the vision rows, which its tags
    split."""
    out: list = []
    for s in lay.segments:
        if not s.rows:
            continue
        k = "text" if s.kind == "vision" else s.kind
        if not (out and out[-1] == k == "text"):
            out.append(k)
    return out


def want_kinds(m: dict) -> list:
    """The segment kinds of a fixture's step: text, each anchor's cond and
    cond_audio rows in turn, then the targets."""
    kinds = ["text"]
    for a in m["inputs"].get("anchors", ()):
        kinds += ["cond"] * bool(a["video_t"]) + \
            ["cond_audio"] * bool(a["audio_t"])
    return kinds + ["audio", "video"]


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
    kinds = segment_kinds(lay)
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
    cv, ca = cond(model, arrays)
    taps: dict = {}
    with pacing.paced(_eval_blocks):
        text = model.caption(feats, taps)
        vv, vaud = model(video, audio, tv, ta, text, lay, cond_video=cv,
                         cond_audio=ca, taps=taps)
    got = {k: _np(v) for k, v in taps.items()}
    got["velocity_video"], got["velocity_audio"] = _np(vv), _np(vaud)
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


def over(v: float, name: str, tolerances: dict) -> float:
    """How many times its gate a measure is. A cosine's distance from 1
    goes as the square of a relative L2, so its ratio to its floor's is
    taken under a square root."""
    if name.startswith("cosine "):
        floor = tolerances[name.removeprefix("cosine ")]["cosine_min"]
        return math.sqrt(max(1 - v, 0.0) / (1 - floor))
    return v / gate(tolerances[name])


# The rows the oracle runs in float32 in both modes.
FLOAT32_ROWS = {"video_patch_proj", "audio_patch_proj",
                "final_layer.adaln_proj.linear"}

# The restarts a free block row's restart spread is the largest gap over
# (scripts/video_oracle/check3_h3_spread.py): the oracle's bf16 blocks
# from the fixture's state at each of these blocks, as is and nudged with
# each seed. The counts are fixed, so the largest gap does not grow with
# more runs.
RESTART_STARTS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30)
RESTART_SEEDS = (0, 1, 2)


def restart_spread(m: dict) -> dict:
    """The restart spread the manifest records for each free block row, by
    row; none when it records no spread. A spread covers the fixed
    restarts and every block row from its first, past the last start, to
    the last block."""
    rec = m["results"].get("free_restart_spread")
    if rec is None:
        return {}
    assert tuple(rec["starts"]) == RESTART_STARTS
    assert tuple(rec["seeds"]) == RESTART_SEEDS
    depth = m["inputs"]["blocks"]
    assert RESTART_STARTS[-1] < rec["first"] < depth
    assert set(rec["rows"]) == {f"blocks.{b}"
                                for b in range(rec["first"], depth)}
    runs = {f"blocks.{k}" for k in RESTART_STARTS} | {
        f"blocks.{k} nudge{s}" for k in RESTART_STARTS
        for s in RESTART_SEEDS}
    for k, r in rec["rows"].items():
        assert r["run"] in runs and r["rel_l2"] > 0, k
    return {k: r["rel_l2"] for k, r in rec["rows"].items()}


def assert_manifest_rule(m: dict) -> None:
    """Every row tolerance is twice the gap beside it, or the float32
    floor for a row the oracle runs in float32 in both modes, each
    teacher-forced row run free is twice its free gap, or the larger of
    that and its restart spread where the manifest records one, the
    velocity floors follow their gaps, the fixture holds every row of its
    depth with a gap for each, and gmlx's stream times are the oracle's.
    The restart spread widens free block rows alone."""
    tols, res = m["tolerances"], m["results"]
    for stream in ("video", "audio"):
        gap = res["velocity_bf16_vs_fp32"][stream]
        tol = tols[f"velocity_{stream}"]
        assert tol["rel_l2"] == 2 * gap["rel_l2"]
        assert math.isclose(tol["cosine_min"], 1 - 4 * (1 - gap["cosine"]),
                            rel_tol=0, abs_tol=1e-12)
    float32 = set()
    for k, g in res["rows_bf16_vs_fp32"].items():
        if g["rel_l2"] == 0:
            assert tols[k] == {"rel_l2": 1e-5, "float32": True}, k
            float32.add(k)
            continue
        assert tols[k]["rel_l2"] == 2 * g["rel_l2"], k
        assert g["teacher_forced"] == tols[k]["teacher_forced"] == \
            forced_row(k), k
    assert float32 == FLOAT32_ROWS
    assert all(keeps_fp32(k) for k in float32)
    forced = {k for k in res["rows_bf16_vs_fp32"] if forced_row(k)}
    assert set(res["free_bf16_vs_fp32"]) == forced
    spread = restart_spread(m)
    assert set(spread) <= set(res["free_bf16_vs_fp32"])
    for k, g in res["free_bf16_vs_fp32"].items():
        want = max(2 * g["rel_l2"], spread.get(k, 0.0))
        assert tols[f"free {k}"] == {"rel_l2": want}, k
    assert gated_rows(m) - {f"cosine {v}" for v in VELOCITIES} == \
        expected_rows(m)
    assert set(res["rows_bf16_vs_fp32"]) == {
        k for k in expected_rows(m)
        if not k.startswith("free ") and k not in VELOCITIES}
    assert m["inputs"]["segment_kinds"] == want_kinds(m)
    assert times_at(m["inputs"]["sigma_v"]) == \
        (m["inputs"]["t_video"], m["inputs"]["t_audio"])


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


def exact_forced(model, arrays: dict, m: dict) -> dict:
    """Relative L2 of each teacher-forced row the manifest holds, in the
    oracle's arithmetic."""
    with _exact(model):
        got = forced_rows(model, arrays)
    return {k: rel(v, arrays[k]) for k, v in got.items()
            if k in m["tolerances"]}


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
# Set to a directory to record the floors measured now, for new fixtures
# or another machine: each manifest the exact gate reads is written there
# with them, to copy over the committed one under
# tests/fixtures/video/oracle/ and the one under GMLX_VIDEO_FIXTURES. A run
# never rewrites either.
RECORD = "GMLX_VIDEO_RECORD_FLOORS"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def record_dir() -> Path | None:
    """The directory RECORD names, or None while it is unset."""
    value = os.environ.get(RECORD)
    if not value:
        return None
    out = Path(value).expanduser().resolve()
    assert out.is_dir(), (f"{RECORD}={value} is not a directory: set it to "
                          "the directory for the recorded manifests")
    kept = [_FIXTURES.resolve()]
    if os.environ.get("GMLX_VIDEO_FIXTURES"):
        kept.append(Path(os.environ["GMLX_VIDEO_FIXTURES"]).expanduser()
                    .resolve())
    for d in kept:
        assert not out.is_relative_to(d), (
            f"{RECORD}={value} is under {d}, whose manifests a run reads: "
            "name another directory and copy the recorded manifests over")
    return out


def write_recorded(m: dict, paths, edit) -> None:
    """The manifest at each of ``paths`` with ``edit`` applied to it,
    written under the same name in the RECORD directory."""
    out = record_dir()
    assert out is not None, f"{RECORD} is not set"
    for path in paths:
        doc = json.loads(Path(path).read_text())
        assert doc["npz"] == m["npz"], path
        edit(doc)
        dst = out / Path(path).name
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(dst)


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
                                     f"the exact gate with {RECORD}=<dir>")
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
            f"{RECORD}=<dir>")
    return tols


def record_floors(m: dict, floor: dict, paths) -> dict:
    """Write the floors measured now, and the tolerances the rule gives
    them, into a copy of each manifest at ``paths`` in the RECORD
    directory. Returns the new manifest."""
    record_dir()
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
    def edit(doc):
        doc["results"].update(res)
        doc["tolerances"].update(tols)
    write_recorded(m, paths, edit)
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


# The site gate (tests/image/h3_sites.py) at the video's attention chunk.
qk_stock_n = hs.qk_stock_n
step_sites = hs.step_sites


def missing_ops(model, sites: dict, lay, routed: dict,
                skipped=()) -> list:
    return hs.missing_ops(model, sites, lay, routed, skipped,
                          hf.ATTENTION_CHUNK)
