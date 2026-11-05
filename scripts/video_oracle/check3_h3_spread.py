#!/usr/bin/env python3
"""The restart spread of a check 3 fixture's free block rows, from
ComfyUI's own bf16 blocks.

A free block row of check 3 (``free blocks.<k>``) is gated at twice the
oracle's own free bf16 against fp32 gap. Past a few dozen blocks a free
bf16 run can amplify a rounding sized change to its state far past that
gap, in the oracle's blocks as in gmlx's. Where gmlx's free row then
misses twice the gap while the fixture's teacher-forced rows, velocities
and exact gates pass, the row takes the restart spread as well (RULE):
the oracle's own largest gap over restarts from fixed states of the
fixture.

Each restart starts from the fixture's float32 output of one block of
STARTS, cast to bf16, as is (the base) and nudged as the tests nudge an
input (one float32 step up, down or not at all at random, seeded by each
of SEEDS) before the cast, and runs ComfyUI's bf16 blocks one by one
through the last block, on the step's own time rows, segment table and
rotation table (a one-block call of the step). The spread of each block
row from --first on is the largest relative L2 to the fixture over the
4 x len(STARTS) runs. The counts are fixed so the largest value does not
grow with more runs.

Each run's reads go to --work as the run ends, and a rerun resumes after
the runs already there. --write puts the spread and the gates it gives
into the fixture's manifest under GMLX_VIDEO_FIXTURES and into each
--copy of it; the repository keeps its copy under
tests/fixtures/video/oracle/. --write needs no ComfyUI.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 8 check3_h3_spread.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf \\
      --name check3_h3_clip17_512x512x22 --first 36 --work <runs.json>
  python scripts/video_oracle/check3_h3_spread.py --write \\
      --name check3_h3_clip17_512x512x22 --first 36 --work <runs.json> \\
      --copy tests/fixtures/video/oracle/check3_h3_clip17_512x512x22.manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import fixtures

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

SCRIPT = "scripts/video_oracle/check3_h3_spread.py"
STARTS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30)
SEEDS = (0, 1, 2)
RULE = ("a free block row from the first one gmlx's free run misses at "
        "twice the oracle's free gap, while the teacher-forced rows, the "
        "velocities and the exact gates pass, to the last block: "
        "tolerance = the larger of 2 x the oracle's free bf16 against fp32 "
        "relative L2 and the restart spread, the oracle's own largest "
        "relative L2 at that row over its bf16 blocks restarted from the "
        "fixture's state at each of 13 blocks (4 to 30), as is and with 3 "
        "nudges of one float32 step; measured once per fixture by " + SCRIPT)


def nudge(a: np.ndarray, seed: int) -> np.ndarray:
    """One float32 step up, down or none at random per element, as
    tests/video/h3_gates.py nudged() moves an input."""
    a = np.asarray(a, np.float32)
    step = np.random.default_rng(seed).integers(-1, 2, a.shape)
    return np.where(step == 0, a, np.nextafter(
        a, np.where(step > 0, np.inf, -np.inf).astype(np.float32)))


def bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round to nearest even of float32 to bfloat16, as bits."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def runs() -> list[tuple[str, int, int | None]]:
    """(tag, start, seed) of every restart: the bases first, then each
    start's nudges."""
    out = [(f"blocks.{k}", k, None) for k in STARTS]
    out += [(f"blocks.{k} nudge{s}", k, s) for k in STARTS for s in SEEDS]
    return out


def measure(a) -> int:
    import check3_h3 as c3
    import oracle

    torch = oracle.setup()
    head = oracle.start_comfy(a.comfy_dir)
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    z, m = fixtures.load(a.name)
    want = m["files"]["dit"]
    got = fixtures.file_identity(a.dit)
    if (got["name"], got["size"], got["sha256"]) != \
            (want["name"], want["size"], want["sha256"]):
        raise SystemExit(f"error: {a.dit} is not the fixture's DiT")
    inp = m["inputs"]
    case = inp.get("case")
    payload = None
    if case is not None:
        keyframes, tags, extra, _ = c3.anchor_case(
            case, tuple(inp["shape"]), c3.ANCHOR_SEED)
        for k, v in extra.items():
            if not np.array_equal(np.asarray(v), z[k]):
                raise SystemExit(f"error: {k} is not the fixture's")
        payload = c3.payload_of(keyframes, tags, c3.ANCHOR_SEED)
    _, dm = c3.load_gguf_dit(torch, a.dit.resolve(), None)
    last = len(dm.blocks) - 1
    if last != inp["blocks"] - 1 or a.first > last:
        raise SystemExit(f"error: {len(dm.blocks)} blocks, the fixture "
                         f"{inp['blocks']}, --first {a.first}")
    work = json.loads(a.work.read_text()) if a.work.exists() else {}
    if work and (work.get("name"), work.get("first")) != (a.name, a.first):
        raise SystemExit(f"error: {a.work} holds another measurement")
    work.update(name=a.name, first=a.first, last=last, comfy=head,
                comfy_gguf=gguf_head, threads=torch.get_num_threads(),
                starts=list(STARTS), seeds=list(SEEDS))
    work.setdefault("runs", {})
    oracle.emulate_bf16_matmul(True)
    try:
        full = dm.blocks
        dm.blocks = full[:1]
        try:
            _, _, cap, _ = c3.call(torch, dm, z["video"], z["audio"],
                                   z["features"], float(z["sigma"]), "bf16",
                                   payload)
        finally:
            dm.blocks = full
        x, t_emb, msegs, table = cap.inputs["blocks.0"][:4]
        shape, dtype = x.shape, x.dtype
        del cap, x
        for tag, k, seed in runs():
            if tag in work["runs"]:
                continue
            state = z[f"blocks.{k}"]
            if seed is not None:
                state = nudge(state, seed)
            h = torch.from_numpy(np.array(state, np.float32)).reshape(
                shape).to(dtype)
            cast = oracle.to_numpy(h).copy()
            if not np.array_equal(bf16_bits(cast),
                                  bf16_bits(state).reshape(cast.shape)):
                raise SystemExit(f"error: {tag}: torch's cast to bf16 is "
                                 f"not round to nearest even")
            flips = int(np.count_nonzero(
                bf16_bits(state) != bf16_bits(z[f"blocks.{k}"])))
            reads, t0 = {}, time.perf_counter()
            for b in range(k + 1, last + 1):
                with torch.no_grad():
                    h = dm.blocks[b](h, t_emb, msegs, table,
                                     transformer_options={})
                ref = z[f"blocks.{b}"]
                reads[f"blocks.{b}"] = fixtures.rel_l2(
                    oracle.to_numpy(h).reshape(ref.shape), ref)
            work["runs"][tag] = {
                "start": k, "seed": seed, "bf16_flips": flips,
                "digest": hashlib.sha256(bf16_bits(state).tobytes())
                .hexdigest()[:16],
                "reads": reads, "wall_s": round(time.perf_counter() - t0, 1)}
            tmp = a.work.with_name(a.work.name + ".part")
            tmp.write_text(json.dumps(work, indent=1) + "\n")
            os.replace(tmp, a.work)
            print(f"[spread] {tag}: blocks.{last} "
                  f"{reads[f'blocks.{last}']:.4f}, "
                  + " ".join(f"{b} {reads[f'blocks.{b}']:.4f}"
                             for b in range(a.first, min(a.first + 3,
                                                         last + 1))),
                  flush=True)
    finally:
        oracle.emulate_bf16_matmul(False)
    return 0


def spread(work: dict) -> dict:
    """The largest read at each row from ``first`` on, and its run."""
    want = [t for t, _, _ in runs()]
    missing = [t for t in want if t not in work["runs"]]
    if missing or set(work["runs"]) - set(want):
        raise SystemExit(f"error: runs missing {missing} or extra "
                         f"{sorted(set(work['runs']) - set(want))}")
    out = {}
    for b in range(work["first"], work["last"] + 1):
        key = f"blocks.{b}"
        tag = max(want, key=lambda t: work["runs"][t]["reads"][key])
        out[key] = {"rel_l2": work["runs"][tag]["reads"][key], "run": tag}
    return out


def write(a) -> int:
    work = json.loads(a.work.read_text())
    if (work["name"], work["first"]) != (a.name, a.first):
        raise SystemExit(f"error: {a.work} holds another measurement")
    rows = spread(work)
    paths = [fixtures.fixture_root() / f"{a.name}.manifest.json", *a.copy]
    docs = [json.loads(p.read_text()) for p in paths]
    for p, d in zip(paths, docs):
        if d["npz"] != docs[0]["npz"] or \
                d["tensors"] != docs[0]["tensors"]:
            raise SystemExit(f"error: {p} is not the fixture's manifest")
    free = docs[0]["results"]["free_bf16_vs_fp32"]
    rec = {"rule": RULE, "starts": work["starts"], "seeds": work["seeds"],
           "first": work["first"], "rows": rows,
           "measured": (f"ComfyUI {work['comfy']}, ComfyUI-GGUF "
                        f"{work['comfy_gguf']}, {work['threads']} CPU "
                        f"threads, bf16 matmul emulation; {SCRIPT}")}
    tols = {f"free {k}": {"rel_l2": max(2 * free[k]["rel_l2"],
                                        r["rel_l2"])}
            for k, r in rows.items()}
    for p, d in zip(paths, docs):
        d["results"]["free_restart_spread"] = rec
        d["tolerances"].update(tols)
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(d, indent=1) + "\n")
        os.replace(tmp, p)
    for k, r in rows.items():
        print(f"[spread] {a.name} free {k}: spread {r['rel_l2']:.4f} "
              f"({r['run']}), twice the free gap "
              f"{2 * free[k]['rel_l2']:.4f}, gate "
              f"{tols[f'free {k}']['rel_l2']:.4f}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="check 3 fixture name")
    ap.add_argument("--first", type=int, required=True,
                    help="the first block row the spread gates")
    ap.add_argument("--work", type=Path, required=True,
                    help="JSON of every run's reads, resumed on a rerun")
    ap.add_argument("--comfy-dir", type=Path,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, help="MiniMax-H3 DiT GGUF")
    ap.add_argument("--write", action="store_true",
                    help="write the spread from --work into the manifest")
    ap.add_argument("--copy", type=Path, nargs="*", default=[],
                    help="committed copies of the manifest to update too")
    a = ap.parse_args(argv)
    if a.write:
        return write(a)
    if a.comfy_dir is None or a.dit is None:
        ap.error("--comfy-dir and --dit are required to measure")
    a.dit = a.dit.expanduser().resolve()
    return measure(a)


if __name__ == "__main__":
    sys.exit(main())
