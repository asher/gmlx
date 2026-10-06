#!/usr/bin/env python3
"""Check 6 oracle for Qwen-Image 2.1: a whole sampling run on injected
noise, to the final latents.

The fp32 run is QwenImage21Pipeline.__call__ itself at its defaults for a
text prompt: the encoder features of the check 2 fixture as
prompt_embeds, true_cfg_scale 1, the key and value cache on, the noise as
latents, and output_type latent. The noise is numpy's PCG64 standard
normal for the seed, which is how gmlx makes noise from a seed.

The bf16 run follows the same loop with the DiT in bfloat16: the first
step extracts the cache, the later steps read it, and the latents and
the scheduler's Euler update stay in float32. The script runs that loop
in fp32 too for --tie steps and holds it, bit for bit, to the latents the
pipeline's own call has after each of them. In bf16 the linear products
run as float32 products over bfloat16 operands with one bfloat16
rounding, and the sinusoid reads the float32 timestep.

For each size the fixture holds the noise, the features, the fp32
latents after the steps of --keep and after the last one, and for a size
in --bf16 the final bf16 latents. The manifest records the bf16 against
fp32 gap of the final latents and sets their tolerance at twice that gap.
A size without a bf16 run takes the tolerance of the largest size that
has one as its starting value. The fixture holds no image: the decode is
check 5's.

--sigmas runs a shorter schedule, passed to the pipeline as it is, and
the manifest names it.

  python scripts/image_oracle/run.py --python <oracle python> --threads 12 \\
      check6_qwen_image21.py --dit qwen-image-2.1-Q4_K_M.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check3_qwen_image21 as c3
import fixtures
import oracle
from check2_qwen_image21 import text_processor

NAME = "check6_qwen_image21"
CHECK2 = "check2_qwen_image21"
SIZES = (512, 1024)
STEPS = 40
SEED = 6
KEEP = (1, 2, 10, 20, 30)
STARTING = {"latents_rel_l2": 5e-2}
RULE = ("final latents: relative L2 at twice the oracle's own bf16 against "
        "fp32 relative L2 at that size; a size without a bf16 run starts "
        "from the tolerance of the largest size that has one")


def pack(torch, z: np.ndarray):
    """(1, C, h, w) as the pipeline packs it, (1, h * w, C)."""
    return torch.from_numpy(z).reshape(1, z.shape[1], -1).transpose(1, 2)


def unpack(t, like: np.ndarray) -> np.ndarray:
    return oracle.to_numpy(t.transpose(1, 2).reshape(like.shape))


def scheduler(torch, a, tokens: int):
    """The Euler scheduler as QwenImage21Pipeline sets it."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_shift, retrieve_timesteps)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    c = sched.config
    mu = calculate_shift(tokens, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    sigmas = (np.linspace(1.0, 1 / a.steps, a.steps) if a.sigmas is None
              else a.sigmas)
    timesteps, _ = retrieve_timesteps(sched, len(sigmas),
                                      torch.device("cpu"), sigmas=sigmas,
                                      mu=mu)
    sched.set_begin_index(0)
    return sched, timesteps


def pipeline_run(torch, a, pipe, noise: np.ndarray, feats: np.ndarray):
    """The pipeline's own call. Returns the final latents, the latents
    after every step and the seconds of every step."""
    after, marks = [], [time.perf_counter()]

    def keep(_pipe, _i, _t, kw):
        after.append(unpack(kw["latents"], noise))
        marks.append(time.perf_counter())
        return {}

    size = noise.shape[2] * 16, noise.shape[3] * 16
    steps = a.steps if a.sigmas is None else len(a.sigmas)
    (lat,) = pipe(prompt_embeds=torch.from_numpy(feats)[None],
                  height=size[0], width=size[1], num_inference_steps=steps,
                  sigmas=a.sigmas, latents=pack(torch, noise),
                  output_type="latent", return_dict=False,
                  callback_on_step_end=keep)
    return unpack(lat, noise), after, np.diff(marks)


def loop(torch, a, model, noise: np.ndarray, cond: dict, mode: str,
         stop: int | None = None):
    """The pipeline's denoise loop on ``model`` with float32 latents.
    Returns the latents after every step and the seconds of every step."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache)

    sched, timesteps = scheduler(torch, a, noise.shape[2] * noise.shape[3])
    cache = QwenImage21KVCache(len(model.transformer_blocks))
    lat, after, secs = noise.copy(), [], []
    for i, t in enumerate(timesteps[:stop]):
        t0 = time.perf_counter()
        out = c3.call(torch, model, cond, lat, t, mode, a, cache,
                      "extract" if i == 0 else "cached")
        v = c3.velocity(out, lat)
        nxt = sched.step(pack(torch, v), t, pack(torch, lat),
                         return_dict=False)[0]
        lat = unpack(nxt, noise)
        after.append(lat)
        secs.append(time.perf_counter() - t0)
    return after, np.array(secs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", required=True, type=Path,
                    help="Qwen-Image 2.1 DiT GGUF")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--check2", default=CHECK2,
                    help="check 2 fixture that holds the features")
    ap.add_argument("--prompt-key", default="plain")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--bf16", type=int, nargs="*", default=[512],
                    help="sizes that also run in bf16")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    help="an explicit schedule in place of --steps")
    ap.add_argument("--keep", type=int, nargs="*", default=list(KEEP),
                    help="steps whose fp32 latents the fixture keeps")
    ap.add_argument("--tie", type=int, default=2,
                    help="steps of the loop held to the pipeline in fp32")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bf16-timestep", choices=("fp32", "stock"),
                    default="fp32")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if any(s % 32 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 32, and --bf16 among --sizes")
    torch = oracle.setup()
    from diffusers import QwenImage21Pipeline

    root = fixtures.fixture_root(a.out)
    m2 = fixtures.read_manifest(a.check2, root=root)
    with np.load(root / m2["npz"], allow_pickle=False) as z:
        feats = z[f"tap_{a.prompt_key}"].astype(np.float32)
    if fixtures.array_sha256(feats) != \
            m2["tensors"][f"tap_{a.prompt_key}"]["sha256"]:
        raise SystemExit(f"error: tap_{a.prompt_key} of {a.check2} differs "
                         "from its manifest")
    cond = {"features": feats, "img_mask": np.zeros(len(feats), bool),
            "ref_shapes": [], "ref_latents": []}
    models = {m: c3.load_model(torch, a.dit, a.config, m, a.dequant)
              for m in (("fp32", "bf16") if a.bf16 else ("fp32",))}
    c_in = models["fp32"].config.in_channels
    sched, _ = scheduler(torch, a, 1024)
    pipe = QwenImage21Pipeline(scheduler=sched, vae=None, text_encoder=None,
                               processor=text_processor(a.config),
                               transformer=models["fp32"])
    pipe.set_progress_bar_config(disable=True)
    steps = a.steps if a.sigmas is None else len(a.sigmas)

    fx = fixtures.FixtureWriter(
        a.name, check=6, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--dit", a.dit.name, "--config",
              a.config.name, "--sizes", *map(str, a.sizes), "--bf16",
              *map(str, a.bf16)])
    fx.add_file("dit", a.dit)
    fx.add("features", feats)
    results, tol, sigmas = {}, {}, {}
    for size in a.sizes:
        side = size // 16
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, c_in, side, side), dtype=np.float32)
        final, after, secs = pipeline_run(torch, a, pipe, noise, feats)
        sigmas[str(size)] = [float(s) for s in pipe.scheduler.sigmas]
        if len(after) != steps or not np.array_equal(after[-1], final):
            raise SystemExit(f"error: {size}: {len(after)} steps kept")
        tied, _ = loop(torch, a, models["fp32"], noise, cond, "fp32",
                       a.tie)
        same = [bool(np.array_equal(x, y)) for x, y in zip(tied, after)]
        if not all(same):
            raise SystemExit(f"error: {size}: the loop parts from the "
                             f"pipeline in fp32: {same}")
        fx.add(f"noise_{size}", noise)
        fx.add(f"latents_{size}", final)
        # The schedule of the pipeline's own call at this token count.
        fx.add(f"sigmas_{size}", oracle.to_numpy(
            pipe.scheduler.sigmas).astype(np.float32))
        fx.add(f"timesteps_{size}", oracle.to_numpy(
            pipe.scheduler.timesteps).astype(np.float32))
        for k in a.keep:
            if k < steps:
                fx.add(f"latents_{size}_step{k}", after[k - 1])
        row = {"fp32_step_s": [round(float(s), 1) for s in secs],
               "fp32_s": round(float(secs.sum()), 1),
               "loop_equals_pipeline_steps": len(same)}
        print(f"[check6] {size}: fp32 {row['fp32_s']} s, first step "
              f"{secs[0]:.1f} s, later steps {np.median(secs[1:]):.1f} s",
              flush=True)
        if size in a.bf16:
            oracle.emulate_bf16_matmul(True)
            t0 = time.perf_counter()
            after16, secs16 = loop(torch, a, models["bf16"], noise, cond,
                                   "bf16")
            oracle.emulate_bf16_matmul(False)
            fx.add(f"latents_{size}_bf16", after16[-1])
            gap = fixtures.compare(after16[-1], final)
            row.update(bf16_s=round(time.perf_counter() - t0, 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=[
                           fixtures.rel_l2(x, y)
                           for x, y in zip(after16, after)])
            tol[str(size)] = {"latents_rel_l2": 2 * gap["rel_l2"]}
            print(f"[check6] {size}: bf16 {row['bf16_s']} s, final latents "
                  f"rel L2 {gap['rel_l2']:.3e}, cosine {gap['cosine']:.6f}",
                  flush=True)
        results[str(size)] = row
    if tol:
        base = tol[str(max(int(s) for s in tol))]
        for size in a.sizes:
            tol.setdefault(str(size), {**base, "from": "another size"})

    fx.inputs.update(
        sizes=list(a.sizes), bf16_sizes=list(a.bf16), steps=steps,
        schedule=("the pipeline's default, np.linspace(1, 1 / steps, steps)"
                  if a.sigmas is None else "explicit sigmas"),
        sigmas_passed=a.sigmas, sigmas=sigmas, seed=a.seed,
        prompt=m2["inputs"]["prompts"][a.prompt_key],
        features=f"tap_{a.prompt_key} of {a.check2}", guidance=1.0,
        kv_cache=True, kept_steps=[k for k in a.keep if k < steps],
        config_revision=a.config_revision,
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, shape (1, 64, size / 16, size / 16)",
        latents="(1, 64, size / 16, size / 16) float32 in model space, "
                "after the last step")
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "dequant": oracle.dequant_mode(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                "rounding", "bf16_timestep": a.bf16_timestep,
                "plain_decoded": list(models["fp32"].plain_decoded)},
        sizes=results)
    fx.tolerances.update(rule=RULE, starting=STARTING, sizes=tol)
    path = fx.write(a.out)
    print(f"[check6] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
