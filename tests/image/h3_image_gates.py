"""The rows of a MiniMax-H3-Image step against a check 3 fixture of
scripts/image_oracle/check3_h3_image.py: free running from the fixture's
inputs, and teacher-forced for each refiner block, the refiner's norm,
each block and the head from the fixture's float32 input to it. The
exact gate's floors follow tests/video/h3_gates.py."""
from __future__ import annotations

import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np

import block_gates as bg
from block_gates import tf32

from gmlx.image import pacing
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him
from gmlx.image.h3 import rope

VELOCITY = "velocity"
HEAD = "final_layer"
# Rows outside the blocks that every fixture holds.
TOP_ROWS = ("condition_proj", "video_patch_proj", "token_refiner.final_norm",
            "final_layer.adaln_proj.linear", HEAD)
# The rows the oracle runs in float32 in both modes. Where MLX multiplies
# float32 at TF32 precision they take TF32_ROWS, as in h3_gates.py.
FLOAT32_ROWS = {"video_patch_proj", "final_layer.adaln_proj.linear"}
TF32_ROWS = 2.0 ** -9


def keeps_fp32(name: str) -> bool:
    """Whether the loader keeps parameter or leaf ``name`` in float32."""
    return any(s in name + "." for s in hi.FAMILY.fp32_keep)


def tiny_model(weights: dict, config: dict, act=mx.float32):
    """The DiT of float32 weights by name, in torch order, cast as the
    loader casts a file's float tensors."""
    m = him.H3ImageDiT(config)

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


def case(arrays: dict):
    """The fixture's step: latents (1, C, H, W), features (1, L, text dim)
    and sigma."""
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    return lat, feats, float(arrays["sigma"])


def plan_of(model, arrays: dict):
    lat, feats, sigma = case(arrays)
    _, _, h, w = lat.shape
    return model.modulation(sigma, 1), model.plan(feats.shape[1], h, w)


def layout_misses(model, arrays: dict, m: dict) -> list[str]:
    """Where the step's layout, angles, segment rows, time, curve rows or
    head segment differ from the oracle's."""
    mod, plan = plan_of(model, arrays)
    lay = plan.layout
    out = []
    kinds = [s.kind for s in lay.segments if s.rows]
    if kinds != m["inputs"]["segment_kinds"]:
        out.append(f"segment kinds {kinds}")
    if not np.array_equal(lay.positions, arrays["positions"]):
        out.append("positions")
    # The oracle's angles repeat its rotated half.
    ang = np.array(rope.angles(lay.positions, model.rope.inv_freq))
    ref = arrays["rope.angles"]
    w = ang.shape[1]
    if not (ref.shape == (ang.shape[0], 2 * w)
            and np.array_equal(ref[:, :w], ang)
            and np.array_equal(ref[:, w:], ang)):
        out.append("rope angles")
    if plan.segs != tuple(tuple(r) for r in arrays["segments"].tolist()):
        out.append(f"segments {plan.segs}")
    if tuple(arrays["t_values"].tolist()) != mod.times[:1]:
        out.append(f"times {mod.times}")
    t_emb = np.array(mod.t_emb[:1])
    if not np.allclose(t_emb, arrays["t_emb"], rtol=1e-6, atol=0):
        out.append("curve rows")
    s = plan.target
    if tuple(arrays["head_segment"].tolist()) != (s.start, s.stop, 0):
        out.append(f"head segment {(s.start, s.stop)}")
    if hi.model_sigma(arrays["sigma"]) != float(arrays["head_sigma"]):
        out.append("head sigma")
    return out


def _eval_blocks(arrays) -> None:
    mx.eval(*arrays)


def free_rows(model, arrays: dict) -> dict:
    """Every tapped row of one step from the fixture's inputs, and the
    velocity. Each block evaluates before the next, as a run does."""
    lat, feats, sigma = case(arrays)
    taps: dict = {}
    with pacing.paced(_eval_blocks):
        vel = model(lat, sigma, feats, taps)
    got = {k: _np(v) for k, v in taps.items()}
    got[VELOCITY] = _np(vel)
    return got


def forced_rows(model, arrays: dict) -> dict:
    """Each refiner block, the refiner's norm, each block the fixture
    holds and the head on the fixture's float32 input to them."""
    act = model.act_dtype

    def fed(name):
        return mx.array(arrays[name])[None].astype(act)

    out = {}
    tr = model.token_refiner
    src = "condition_proj"
    for i, blk in enumerate(tr.blocks):
        out[f"token_refiner.blocks.{i}"] = _np(blk(fed(src)))
        src = f"token_refiner.blocks.{i}"
    out["token_refiner.final_norm"] = _np(tr.final_norm(fed(src)))
    mod, plan = plan_of(model, arrays)
    cos, sin = rope.tables(plan.layout.positions, model.rope.inv_freq,
                           model.cfg.block.head_dim)
    src = "in.blocks.0"
    for i, blk in enumerate(model.blocks):
        y = blk(fed(src), blk.modulation(mod.t_emb, act), plan.segs, cos, sin)
        out[f"blocks.{i}"] = _np(y)
        src = f"blocks.{i}"
    out[HEAD] = _np(model.head(fed("in.final_layer"), mod, plan))
    return out


def forced_row(name: str) -> bool:
    return name.startswith(("blocks.", "token_refiner.")) or name == HEAD


def measures(model, arrays: dict, m: dict) -> dict:
    """Relative L2 of every row the manifest holds against the fixture:
    teacher-forced where the oracle forced it, free running for the rest,
    ``free <row>`` for each teacher-forced row run free, and the cosine of
    the velocity."""
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
    got[f"cosine {VELOCITY}"] = cosine(free[VELOCITY], arrays[VELOCITY])
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
    """The rows a check 3 fixture holds a tolerance for: the rows outside
    the blocks, each refiner block and block, ``free`` for each
    teacher-forced row, and the velocity."""
    config = m["inputs"].get("config")
    if config is None:
        depth, refiner = m["inputs"]["blocks"], hi._REFINER
    else:
        depth = config["num_layers"]
        refiner = config["token_refiner_num_layers"]
    rows = {*TOP_ROWS, *(f"token_refiner.blocks.{i}" for i in range(refiner)),
            *(f"blocks.{i}" for i in range(depth))}
    return rows | {f"free {k}" for k in rows if forced_row(k)} | {VELOCITY}


def gated_rows(m: dict) -> set:
    rows = {k for k in m["tolerances"]
            if k != "rule" and not k.startswith("exact_")}
    return rows | {f"cosine {VELOCITY}"}


def assert_manifest_rule(m: dict) -> None:
    """Every row tolerance is twice the gap beside it, or the float32
    floor for a row the oracle runs in float32 in both modes, each
    teacher-forced row run free is twice its free gap, the velocity
    floors follow its gap, and the fixture holds every row of its depth
    with a gap for each."""
    tols, res = m["tolerances"], m["results"]
    gap = res["velocity_bf16_vs_fp32"]
    assert tols[VELOCITY]["rel_l2"] == 2 * gap["rel_l2"]
    assert abs(tols[VELOCITY]["cosine_min"]
               - (1 - 4 * (1 - gap["cosine"]))) <= 1e-12
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
    for k, g in res["free_bf16_vs_fp32"].items():
        assert tols[f"free {k}"] == {"rel_l2": 2 * g["rel_l2"]}, k
    assert gated_rows(m) - {f"cosine {VELOCITY}"} == expected_rows(m)
    assert set(res["rows_bf16_vs_fp32"]) == {
        k for k in expected_rows(m)
        if not k.startswith("free ") and k != VELOCITY}
    assert m["inputs"]["segment_kinds"] == ["text", "video"]
    assert m["inputs"]["sample_sigmas"] == [
        float(np.float32(s)) for s in hi.PDD_SIGMAS]


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
    return (f"{label}: {len(ratio)} rows, worst {got[top]:.2e} at {top}, "
            f"{ratio[top]:.2f} of its gate; worst free row {worst_free} "
            f"{free[worst_free]:.2f} of its gate; velocity "
            f"{got[VELOCITY]:.2e} of {tols[VELOCITY]['rel_l2']:.2e}")


def kernel_census(model, rows: int, ran: list) -> tuple[Counter, Counter]:
    """The kernel calls one ``measures`` made, and the calls the routes in
    force must make: the q and k kernel in every block of the free and the
    forced pass, and the GLU kernel there too from its row floor on a GPU
    with tensor ops."""
    from gmlx.image.ops import kq_routes

    on = kq_routes.active()
    n = 2 * len(model.blocks)
    want: Counter = Counter()
    if "qk" in on:
        want["qk"] = n
    if ("glu" in on and kq_routes._nax()
            and rows >= kq_routes._GLU_MIN_ROWS):
        want["glu"] = n
    return Counter(ran), want


# The exact gate: the step in the oracle's fp32 arithmetic.

@contextmanager
def exact_floats(model):
    """Every float parameter outside the K-quant leaves in float32 while
    the block runs. Yields how many it cast."""
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
    with bg.exact_products(model) as used:
        yield
        bg.assert_all_dense(model, used)


def exact_measures(model, arrays: dict, m: dict) -> dict:
    """``measures`` in the oracle's arithmetic, without the cosine."""
    with _exact(model):
        got = measures(model, arrays, m)
    return {k: v for k, v in got.items() if not k.startswith("cosine ")}


def free_row(name: str) -> bool:
    return not forced_row(name) and name != VELOCITY


# Each fixture records gmlx's own fp32 floor on the free rows (the worst
# of them) and on the velocity: the largest relative L2 between the exact
# step on the GPU and the same step on the CPU stream or on inputs nudged
# by one float32 step. Each takes the larger of EXACT_TOL and twice its
# floor; the teacher-forced rows keep EXACT_TOL.
FLOORS = {"free": ("exact_free_floor", "exact_free"),
          VELOCITY: ("exact_velocity_floor", "exact_velocity")}
NUDGE_SEEDS = (0, 1, 2, 3)
# A directory to record the floors measured now: each manifest the exact
# gate reads is written there with them, to copy over the committed one
# and the one under GMLX_IMAGE_FIXTURES. A run never rewrites either.
RECORD = "GMLX_IMAGE_RECORD_FLOORS"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def record_dir() -> Path | None:
    value = os.environ.get(RECORD)
    if not value:
        return None
    out = Path(value).expanduser().resolve()
    assert out.is_dir(), f"{RECORD}={value} is not a directory"
    kept = [_FIXTURES.resolve()]
    if os.environ.get("GMLX_IMAGE_FIXTURES"):
        kept.append(Path(os.environ["GMLX_IMAGE_FIXTURES"]).expanduser()
                    .resolve())
    for d in kept:
        assert not out.is_relative_to(d), (
            f"{RECORD}={value} is under {d}, whose manifests a run reads")
    return out


def nudged(arrays: dict, seed: int) -> dict:
    """The fixture's inputs with each element moved one float32 step up,
    down or not at all, at random."""
    rng = np.random.default_rng(seed)
    out = dict(arrays)
    for k in ("latents", "features"):
        a = np.asarray(arrays[k], np.float32)
        step = rng.integers(-1, 2, a.shape)
        out[k] = np.where(step == 0, a, np.nextafter(
            a, np.where(step > 0, np.inf, -np.inf).astype(np.float32)))
    return out


def exact_free_floor(model, arrays: dict, m: dict) -> dict:
    """The largest relative L2 of each free row of ``measures`` and of the
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
    free = {k: v for k, v in floor.items() if free_row(k)}
    top = max(free, key=free.get)
    return {"free": free[top], "free_at": top, VELOCITY: floor[VELOCITY]}


def check_floor(m: dict, floor: dict, label: str) -> dict:
    """The floors measured now are within 1.25x of the floors the manifest
    records. Returns the manifest's tolerances."""
    now = floors_now(floor)
    print(f"{label}: fp32 floor of the free rows {now['free']!r} at "
          f"{now['free_at']}, of the velocity {now[VELOCITY]!r}")
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
    out = record_dir()
    assert out is not None, f"{RECORD} is not set"
    now = floors_now(floor)
    how = (f"the largest relative L2 between gmlx's exact step on the GPU "
           f"and the same step on the CPU stream or on inputs nudged by "
           f"one float32 step ({len(NUDGE_SEEDS)} seeds), {{}}; "
           f"tests/image/h3_image_gates.py exact_free_floor on "
           f"{m['files']['dit']['name']} {m['files']['dit']['sha256'][:8]}, "
           f"{mx.device_info()['device_name']}")
    res = {"exact_free_floor": {"rel_l2": now["free"], "row": now["free_at"],
                                "measured": how.format("worst free row")},
           "exact_velocity_floor": {"rel_l2": now[VELOCITY],
                                    "measured": how.format(VELOCITY)}}
    tols = {key: {"rel_l2": max(bg.EXACT_TOL, 2 * res[r]["rel_l2"])}
            for r, key in FLOORS.values()}
    for path in paths:
        doc = json.loads(Path(path).read_text())
        assert doc["npz"] == m["npz"], path
        doc["results"].update(res)
        doc["tolerances"].update(tols)
        dst = out / Path(path).name
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(dst)
    return {**m, "results": {**m["results"], **res},
            "tolerances": {**m["tolerances"], **tols}}


def exact_gate(name: str, tols: dict) -> float:
    if name == VELOCITY:
        return tols[VELOCITY]
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
            f"{bg.EXACT_TOL:.0e}; velocity {got[VELOCITY]:.2e}, gate "
            f"{tols[VELOCITY]:.2e}; free rows within {worst(free)}, gate "
            f"{tols['free']:.2e}")


def fault_line(label: str, bad: dict, tols: dict) -> tuple[str, float]:
    """Where a wrong model misses the exact gate, by kind of row, and the
    largest ratio of a measure to its gate."""
    parts = []
    for kind, keep in (("teacher-forced", forced_row),
                       ("velocity", lambda k: k == VELOCITY),
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
