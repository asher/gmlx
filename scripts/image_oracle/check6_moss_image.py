#!/usr/bin/env python3
"""Check 6 oracle for MOSS-Image: a whole guided sampling run on injected
noise, to the final latents.

The run is MOSSImagePipeline.__call__ itself, from the release's
pipeline_moss_image.py loaded by path, with the DiT of --dit built as in
check3_moss_image and the release's MOSSImageScheduler on its official
config. The call takes the encoder features of --features as
prompt_embeds and negative_prompt_embeds, each with a mask that keeps
every row: the check 2 oracle's float32 features of the plain prompt and
of the empty prompt, which is the release's default negative prompt. The
two have different lengths, and the call runs each pass at its own. It
runs at the release's defaults: 50 steps, guidance 3, block 22 left out
of the negative pass, the latents in float32 and output_type latent. A
wrapper on the scheduler's step keeps the latents after every step. The
noise is numpy's PCG64 standard normal for the seed, NCHW, which is how
gmlx makes noise from a seed.

The fp32 run has every op in float32. The bf16 run is the same call on
the DiT in bfloat16, the features cast to bfloat16, products as one
float32 GEMM over bf16 operands with one rounding. The release turns each
velocity to float32 before the guidance, so the guidance and the Euler
steps run in float32 in both modes.

One fixture per size holds the noise, both features, the schedule, the
fp32 latents after the steps of --keep and after the last one, and with
a bf16 run the bf16 latents at the same steps and the bf16 against fp32
gap after every step. At GATE_SIZE the tolerance of gmlx's bf16 latents
after a step is twice that gap, so that size needs a bf16 run. At every
other size gmlx's exact fp32 run is held to EXACT_REL_L2 after each kept
step and the last, and its bf16 run is read against the GATE_SIZE gates.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_moss_image.py --pipeline <dir>/pipeline_moss_image.py \\
      --dit MOSS-Image-SFT-Q4_K_M.gguf --config <dir>/transformer/config.json \\
      --scheduler-config <dir>/scheduler/scheduler_config.json \\
      --features check2_moss_image_q8_0.npz --sizes 512 1024 --bf16 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import check3_moss_image as c3
import fixtures
import oracle

NAME = "check6_moss_image"
SCRIPT = "scripts/image_oracle/check6_moss_image.py"
SEED = 6
STEPS = 50
GUIDANCE = 3.0
NEG_SKIP = (22,)
KEEP = (1, 2, 5, 10, 25)
GATE_SIZE = 512
EXACT_REL_L2 = 1e-3
RULE = (f"latents after a kept step and after the last. At {GATE_SIZE}: "
        "gmlx's bf16 run at relative L2 twice the oracle's own bf16 "
        "against fp32 relative L2 after that step. At any other size: "
        "gmlx's exact fp32 run against the fp32 latents at exact_rel_l2, "
        f"and its bf16 run read as a share of the {GATE_SIZE} gate of the "
        "step, not gated")


def scheduler(rel, config: dict):
    return rel.MOSSImageScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        shift=config["shift"], training_shift=config["training_shift"])


def run(torch, rel, model, sched_cfg: dict, noise: np.ndarray, pos, neg,
        mode: str, steps: int):
    """The release's call. Returns the latents after every step, the
    sigmas and timesteps of the schedule and the seconds of every
    step."""
    dtype = oracle.torch_dtype(mode)
    sched = scheduler(rel, sched_cfg)
    pipe = rel.MOSSImagePipeline(transformer=model, scheduler=sched)
    pipe.set_progress_bar_config(disable=True)
    after, marks = [], [time.perf_counter()]
    step = sched.step

    def keep(*args, **kw):
        out = step(*args, **kw)
        after.append(oracle.to_numpy(out[0]))
        marks.append(time.perf_counter())
        return out

    sched.step = keep

    def embeds(f):
        return torch.from_numpy(f[None]).to(dtype)

    _, _, h, w = noise.shape
    with torch.no_grad():
        (lat,) = pipe(prompt_embeds=embeds(pos),
                      negative_prompt_embeds=embeds(neg),
                      height=h * 16, width=w * 16, num_inference_steps=steps,
                      guidance_scale=GUIDANCE,
                      uncond_skip_layers=list(NEG_SKIP),
                      latents=torch.from_numpy(noise), output_type="latent",
                      return_dict=False)
    final = oracle.to_numpy(lat)
    if not after or not np.array_equal(after[-1], final):
        raise SystemExit(f"error: {mode}: the kept steps end apart from "
                         "the call's latents")
    return (after, oracle.to_numpy(sched.sigmas).astype(np.float32),
            oracle.to_numpy(sched.timesteps).astype(np.float32),
            np.diff(marks))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipeline", type=Path, required=True,
                    help="the release's pipeline_moss_image.py")
    ap.add_argument("--dit", type=Path, required=True,
                    help="MOSS-Image DiT GGUF")
    ap.add_argument("--config", type=Path, required=True,
                    help="the release's transformer/config.json")
    ap.add_argument("--scheduler-config", type=Path, required=True,
                    help="the release's scheduler/scheduler_config.json")
    ap.add_argument("--features", type=Path, required=True,
                    help=".npz with the features of both prompts")
    ap.add_argument("--positive-key", default="features_plain")
    ap.add_argument("--negative-key", default="features_empty")
    ap.add_argument("--features-source",
                    default="check2_moss_image_q8_0 fp32 features")
    ap.add_argument("--blocks", type=int, help="keep the first N blocks")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--bf16", type=int, nargs="*", default=[GATE_SIZE],
                    help="sizes that also run in bf16")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--keep", type=int, nargs="*", default=list(KEEP),
                    help="steps whose latents the fixture keeps")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if any(s % 16 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 16, and --bf16 among --sizes")
    if GATE_SIZE in a.sizes and GATE_SIZE not in a.bf16:
        ap.error(f"size {GATE_SIZE} sets the gates and needs a bf16 run")
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    rel = c3.release(a.pipeline)
    cfg = {k: v for k, v in json.loads(a.config.read_text()).items()
           if not k.startswith("_")}
    sched_cfg = {k: v for k, v in
                 json.loads(a.scheduler_config.read_text()).items()
                 if not k.startswith("_")}
    t0 = time.perf_counter()
    state, census, _ = c3.gguf_state(torch, a.dit, a.blocks)
    if a.blocks is not None:
        cfg["num_layers"] = a.blocks
    if any(i >= cfg["num_layers"] for i in NEG_SKIP):
        ap.error(f"--blocks must keep block {max(NEG_SKIP)}")
    models = {m: c3.build(torch, rel, state, cfg, m)
              for m in (("fp32", "bf16") if a.bf16 else ("fp32",))}
    print(f"[check6] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    with np.load(a.features) as z:
        pos = z[a.positive_key].astype(np.float32)
        neg = z[a.negative_key].astype(np.float32)
    print(f"[check6] text rows: positive {pos.shape[0]}, negative "
          f"{neg.shape[0]}", flush=True)

    for size in a.sizes:
        n = size // 16
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, cfg["latent_channels"], n, n),
                                    dtype=np.float32)
        name = f"{a.name}_{size}"
        fx = fixtures.FixtureWriter(
            name, check=6, family="moss_image", script=SCRIPT,
            argv=[Path(__file__).name, "--dit", a.dit.name, "--features",
                  a.features.name, "--sizes", str(size),
                  *(["--bf16", str(size)] if size in a.bf16 else [])])
        for role, path in (("dit", a.dit), ("pipeline", a.pipeline),
                           ("config", a.config),
                           ("scheduler_config", a.scheduler_config),
                           ("features", a.features)):
            fx.add_file(role, path)
        after, sigmas, timesteps, secs = run(
            torch, rel, models["fp32"], sched_cfg, noise, pos, neg, "fp32",
            a.steps)
        kept = [k for k in a.keep if k < len(after)]
        fx.add("noise", noise)
        fx.add("features_positive", pos)
        fx.add("features_negative", neg)
        fx.add("sigmas", sigmas)
        fx.add("timesteps", timesteps)
        fx.add("latents", after[-1])
        for k in kept:
            fx.add(f"latents_step{k}", after[k - 1])
        row = {"steps_run": len(after),
               "fp32_step_s": [round(float(s), 1) for s in secs],
               "fp32_s": round(float(secs.sum()), 1)}
        print(f"[check6] {size}: fp32 {row['fp32_s']} s, {len(after)} "
              f"steps, median step {np.median(secs):.1f} s", flush=True)
        tol = {}
        if size in a.bf16:
            oracle.emulate_bf16_matmul(True)
            try:
                after16, _, _, secs16 = run(
                    torch, rel, models["bf16"], sched_cfg, noise, pos, neg,
                    "bf16", a.steps)
            finally:
                oracle.emulate_bf16_matmul(False)
            fx.add("latents_bf16", after16[-1])
            for k in kept:
                fx.add(f"latents_bf16_step{k}", after16[k - 1])
            by_step = [fixtures.rel_l2(x, y) for x, y in zip(after16, after)]
            gap = fixtures.compare(after16[-1], after[-1])
            row.update(bf16_s=round(float(secs16.sum()), 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=by_step)
            tol = {"latents_rel_l2": 2 * gap["rel_l2"],
                   **{f"step{k}_rel_l2": 2 * by_step[k - 1] for k in kept}}
            print(f"[check6] {size}: bf16 {row['bf16_s']} s, final latents "
                  f"rel L2 {gap['rel_l2']:.3e}, cosine {gap['cosine']:.6f}",
                  flush=True)
        if size != GATE_SIZE:
            tol = {**tol, "exact_rel_l2": EXACT_REL_L2}
        fx.inputs.update(
            size=size, steps=a.steps, steps_run=len(after),
            blocks=cfg["num_layers"], guidance_scale=GUIDANCE,
            uncond_skip_layers=list(NEG_SKIP), negative_prompt="",
            scheduler=sched_cfg, seed=a.seed, features=a.features_source,
            positive_key=a.positive_key, negative_key=a.negative_key,
            text_rows={"positive": int(pos.shape[0]),
                       "negative": int(neg.shape[0])},
            kept_steps=kept, gate_size=GATE_SIZE,
            noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
                  "dtype=float32), NCHW, shape (1, 128, size / 16, "
                  "size / 16)",
            latents="(1, 128, size / 16, size / 16) float32 in model "
                    "space, after the step", gguf_codecs=census)
        fx.results.update(
            oracle={"device": "cpu", "threads": torch.get_num_threads(),
                    "dequant": oracle.dequant_mode(),
                    "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                    "rounding", "timestep": "sigma x 1000 in float32 in "
                    "both modes", "guidance": "float32 velocities in both "
                    "modes, as the release casts them"},
            **row)
        fx.tolerances.update(rule=RULE, **tol)
        path = fx.write(a.out)
        print(f"[check6] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
