#!/usr/bin/env python3
"""The nudge spread of a check 6 fixture's shipping steps, from ComfyUI's
own bf16 step.

A shipping step of check 6 is gated at twice the oracle's own bf16
against fp32 relative L2 at that step, one sample of it. Where gmlx's
shipping step misses that gate while the case's exact gates pass, the
step takes the nudge spread as well (RULE): the oracle's own largest bf16
against fp32 gap over its base and the same bf16 step from inputs nudged
as the tests nudge an input (one float32 step up, down or not at all at
random on the step's input latents and features, seeded by each of
SEEDS). Sigmas are not nudged.

Each run is one bf16 forward as check6_h3.py's bf16_forced pass runs it:
ComfyUI's MiniMaxH3Model on the fixture's DiT and LoRA, the bf16 matmul
emulation, the fixture's fp32 latents before the step, check 2's bf16
tap of the prompt, the payload of the fixture's anchors and seed. The
base reruns the fixture's own bf16 step as a check of the inputs, and the
spread takes the fixture's recorded gap as the base sample.

Each run's gaps go to --work as the run ends, and a rerun resumes after
the runs already there. --steps limits a process to those steps, so two
processes with their own --work files share a fixture. --write takes
every --work file, refuses a run in two of them, and puts the spread and
the gates it gives into the fixture's manifest under GMLX_VIDEO_FIXTURES
and into each --copy of it. --gmlx names gmlx's shipping reading of each
step and stream the spread gates, recorded beside it; a step and stream
gmlx's reading does not miss the one-sample gate at is refused. --write
needs no ComfyUI.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 8 check6_h3_spread.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf --lora <8-step LoRA> \\
      --features check2_h3_keyframes.npz \\
      --name check6_h3_lora_last_512x512x22 --steps 2 3 8 --work last.json
  python scripts/video_oracle/check6_h3_spread.py \\
      --name check6_h3_lora_last_512x512x22 --work last.json --write \\
      --gmlx 2:audio:0.0359 3:audio:0.0323 8:audio:0.0374 --copy <manifest>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import fixtures

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

SCRIPT = "scripts/video_oracle/check6_h3_spread.py"
SEEDS = (0, 1, 2)
STREAMS = ("video", "audio")
RULE = ("a shipping step and stream whose gmlx reading misses twice the "
        "oracle's one-sample bf16 against fp32 gap while the case's exact "
        "gates pass: tolerance = the larger of that and the nudge spread, "
        "the oracle's own largest bf16 against fp32 relative L2 at the step "
        "over its recorded base and the same bf16 step from the step's "
        "input latents and features nudged one float32 step (seeds 0 to "
        "2), sigmas not nudged; measured once per fixture by " + SCRIPT)


def nudged(inputs: dict, seed: int) -> dict:
    """``inputs`` (video, audio, features) each moved one float32 step up,
    down or not at all at random, as tests/video/h3_gates.py nudged()
    moves a step's inputs."""
    rng = np.random.default_rng(seed)
    out = dict(inputs)
    for k in ("video", "audio", "features"):
        a = np.asarray(inputs[k], np.float32)
        step = rng.integers(-1, 2, a.shape)
        out[k] = np.where(step == 0, a, np.nextafter(
            a, np.where(step > 0, np.inf, -np.inf).astype(np.float32)))
    return out


def runs(steps) -> list[tuple[str, int, int | None]]:
    """(tag, step, seed) of every run of ``steps``: each step's base, then
    its nudges."""
    out = []
    for i in steps:
        out.append((f"step{i} base", i, None))
        out += [(f"step{i} nudge{s}", i, s) for s in SEEDS]
    return out


def _check_file(path: Path, want: dict, role: str) -> Path:
    path = path.expanduser().resolve()
    got = fixtures.file_identity(path)
    if (got["name"], got["size"], got["sha256"]) != \
            (want["name"], want["size"], want["sha256"]):
        raise SystemExit(f"error: {path} is not the fixture's {role}")
    return path


def _payload(torch, z, inp):
    """The fixture's anchors as ComfyUI's keyframes and their payload,
    with the tags of a keyframe prompt; None for a text to video case."""
    import check3_h3 as c3

    if inp.get("case") is None:
        return None
    keyframes = []
    for i in range(len(inp["anchors"])):
        kf = {"resolved_frame_index": int(z[f"anchor.{i}.frame"])}
        for k, key in (("latent", "video"), ("audio_latent", "audio")):
            if f"anchor.{i}.{key}" in z:
                kf[k] = torch.from_numpy(np.array(z[f"anchor.{i}.{key}"]))
        keyframes.append(kf)
    tags = (torch.from_numpy(z["text_token_tags"].astype(np.int64))
            if "text_token_tags" in z else None)
    return c3.payload_of(keyframes, tags, int(z["anchor_seed"]))


def measure(a) -> int:
    import check3_h3 as c3
    import check6_h3 as c6
    import oracle

    torch = oracle.setup()
    head = oracle.start_comfy(a.comfy_dir)
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    z, m = fixtures.load(a.name)
    inp = m["inputs"]
    dit = _check_file(a.dit, m["files"]["dit"], "DiT")
    feats = _check_file(a.features, m["files"]["features"], "features")
    lora = None
    if inp["lora"] is not None:
        if a.lora is None:
            raise SystemExit("error: the fixture ran a LoRA: give --lora")
        lora = _check_file(a.lora, m["files"]["lora"], "LoRA")
        if inp["lora_merge"] == "float32":
            c6.merge_in_float32(torch)
    zf = np.load(feats)
    prompt = inp["prompt"]
    if not np.array_equal(zf[f"tap_{prompt}"].astype(np.float32),
                          z["features"]):
        raise SystemExit(f"error: {feats.name} tap_{prompt} is not the "
                         "fixture's features")
    bf16_feats = zf[f"tap_bf16_{prompt}"].astype(np.float32)
    n = inp["steps"]
    only = sorted(set(a.steps))
    if not set(only) <= set(range(1, n + 1)):
        raise SystemExit(f"error: --steps {only} are not steps of 1 to {n}")
    (path,) = a.work
    work = json.loads(path.read_text()) if path.exists() else {}
    if work and work.get("name") != a.name:
        raise SystemExit(f"error: {path} holds another fixture")
    extra = set(work.get("runs", {})) - {t for t, _, _ in runs(only)}
    if extra:
        raise SystemExit(f"error: {path} holds runs of other steps: "
                         f"{sorted(extra)}")
    patcher, dm = c3.load_gguf_dit(torch, dit, None)
    if lora is not None:
        patcher, dm, _ = c6.add_lora(lora, patcher, inp["lora_strength"])
    payload = _payload(torch, z, inp)
    if payload is not None:
        drawn = c3.anchor_noise(torch, dm, payload)
        if len(drawn) != sum(k.startswith("anchor_noise.") for k in z) \
                or not all(np.array_equal(d, z[f"anchor_noise.{i}"])
                           for i, d in enumerate(drawn)):
            raise SystemExit("error: the rebuilt payload draws another "
                             "anchor noise than the fixture's")
    work.update(name=a.name, comfy=head, comfy_gguf=gguf_head,
                threads=torch.get_num_threads(), seeds=list(SEEDS))
    work.setdefault("runs", {})
    sv, shifts = z["sigmas_video"], inp["shifts"]
    opts = {"minimax_h3_sigma_shift_video": shifts["video"],
            "minimax_h3_sigma_shift_audio": shifts["audio"]}
    oracle.emulate_bf16_matmul(True)
    try:
        for tag, i, seed in runs(only):
            if tag in work["runs"]:
                continue
            src = "noise_{}" if i == 1 else f"{{}}_step{i - 1}"
            x = {"video": z[src.format("video")],
                 "audio": z[src.format("audio")], "features": bf16_feats}
            if seed is not None:
                x = nudged(x, seed)
            ctx = torch.from_numpy(np.array(x["features"])[None]).to(
                torch.bfloat16)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = dm([torch.from_numpy(np.array(x["video"])),
                          torch.from_numpy(np.array(x["audio"]))],
                         c3.timestep(torch, float(sv[i - 1])), ctx,
                         transformer_options=opts, minimax_payload=payload)
            vel = {s: oracle.to_numpy(o).astype(np.float32)
                   for s, o in zip(STREAMS, out)}
            gaps = {s: fixtures.rel_l2(vel[s], z[f"velocity_{s}_step{i}"])
                    for s in STREAMS}
            rec = {"step": i, "seed": seed, "gaps": gaps,
                   "wall_s": round(time.perf_counter() - t0, 1)}
            if seed is None:
                rec["vs_recorded_bf16"] = {
                    s: fixtures.rel_l2(vel[s],
                                       z[f"velocity_bf16_{s}_step{i}"])
                    for s in STREAMS}
            work["runs"][tag] = rec
            tmp = path.with_name(path.name + ".part")
            tmp.write_text(json.dumps(work, indent=1) + "\n")
            os.replace(tmp, path)
            print(f"[spread6] {a.name} {tag}: video {gaps['video']:.4e} "
                  f"audio {gaps['audio']:.4e}"
                  + (f", to the recorded bf16 step "
                     f"{rec['vs_recorded_bf16']['video']:.2e} / "
                     f"{rec['vs_recorded_bf16']['audio']:.2e}"
                     if seed is None else "")
                  + f", {rec['wall_s']} s", flush=True)
    finally:
        oracle.emulate_bf16_matmul(False)
    return 0


def merged(paths: list[Path]) -> dict:
    """The runs of every work file, one measurement: the same fixture,
    oracle and thread count in each, and no run in two files."""
    keys = ("name", "comfy", "comfy_gguf", "threads", "seeds")
    out: dict = {}
    for p in paths:
        w = json.loads(p.read_text())
        if out and any(w[k] != out[k] for k in keys):
            raise SystemExit(f"error: {p} is another measurement")
        out.update({k: w[k] for k in keys})
        both = set(out.get("runs", {})) & set(w["runs"])
        if both:
            raise SystemExit(f"error: {p} repeats runs {sorted(both)}")
        out.setdefault("runs", {}).update(w["runs"])
    return out


def write(a) -> int:
    work = merged(a.work)
    if work["name"] != a.name:
        raise SystemExit(f"error: {a.work} holds another fixture")
    paths = [fixtures.fixture_root() / f"{a.name}.manifest.json", *a.copy]
    docs = [json.loads(p.read_text()) for p in paths]
    for p, d in zip(paths, docs):
        if d["npz"] != docs[0]["npz"] or \
                d["tensors"] != docs[0]["tensors"]:
            raise SystemExit(f"error: {p} is not the fixture's manifest")
    gaps = docs[0]["results"]["step_bf16_vs_fp32"]
    tols = [dict(t) for t in docs[0]["tolerances"]["steps"]]
    steps: dict = {}
    for spec in a.gmlx:
        i, s, v = spec.split(":")
        i, v = int(i), float(v)
        one = 2 * gaps[i - 1][s]["rel_l2"]
        if s not in STREAMS or not v > one:
            raise SystemExit(f"error: gmlx's step {i} {s} reading {v} does "
                             f"not miss the one-sample gate {one}")
        tags = [t for t, j, _ in runs([i])]
        missing = [t for t in tags if t not in work["runs"]]
        if missing:
            raise SystemExit(f"error: runs missing {missing}")
        reads = {"recorded": gaps[i - 1][s]["rel_l2"],
                 **{t.split()[1]: work["runs"][t]["gaps"][s]
                    for t in tags if work["runs"][t]["seed"] is not None}}
        top = max(reads, key=reads.get)
        steps.setdefault(str(i), {})[s] = {
            "rel_l2": reads[top], "run": top, "runs": reads, "gmlx": v,
            "base_rerun": work["runs"][tags[0]]["gaps"][s]}
        tols[i - 1][s] = {"rel_l2": max(one, reads[top])}
    rec = {"rule": RULE, "seeds": work["seeds"], "steps": steps,
           "measured": (f"ComfyUI {work['comfy']}, ComfyUI-GGUF "
                        f"{work['comfy_gguf']}, {work['threads']} CPU "
                        f"threads, bf16 matmul emulation; {SCRIPT}")}
    for p, d in zip(paths, docs):
        d["results"]["step_nudge_spread"] = rec
        d["tolerances"]["steps"] = tols
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(d, indent=1) + "\n")
        os.replace(tmp, p)
    for i, v in steps.items():
        for s, r in v.items():
            print(f"[spread6] {a.name} step {i} {s}: spread "
                  f"{r['rel_l2']:.4e} ({r['run']}), one-sample gate "
                  f"{2 * gaps[int(i) - 1][s]['rel_l2']:.4e}, gmlx "
                  f"{r['gmlx']:.4e}, gate {tols[int(i) - 1][s]['rel_l2']:.4e}",
                  flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="check 6 fixture name")
    ap.add_argument("--work", type=Path, nargs="+", required=True,
                    help="JSON of the runs' gaps, resumed on a rerun; "
                         "--write takes every process's file")
    ap.add_argument("--steps", type=int, nargs="+",
                    help="measure these steps (1-based)")
    ap.add_argument("--comfy-dir", type=Path,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, help="MiniMax-H3 DiT GGUF")
    ap.add_argument("--lora", type=Path, help="the fixture's LoRA")
    ap.add_argument("--features", type=Path,
                    help="the check 2 fixture .npz the fixture's features "
                         "came from")
    ap.add_argument("--write", action="store_true",
                    help="write the spread from --work into the manifest")
    ap.add_argument("--gmlx", nargs="+", default=[],
                    help="STEP:STREAM:READING of each gated step, gmlx's "
                         "shipping reading")
    ap.add_argument("--copy", type=Path, nargs="*", default=[],
                    help="committed copies of the manifest to update too")
    a = ap.parse_args(argv)
    if a.write:
        if not a.gmlx:
            ap.error("--write needs --gmlx")
        return write(a)
    if a.comfy_dir is None or a.dit is None or a.features is None \
            or a.steps is None:
        ap.error("--comfy-dir, --dit, --features and --steps are required "
                 "to measure")
    return measure(a)


if __name__ == "__main__":
    sys.exit(main())
