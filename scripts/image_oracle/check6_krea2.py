#!/usr/bin/env python3
"""Check 6 oracle for Krea 2: a whole Turbo sampling run on injected
noise, to the final latents.

The fp32 run is Krea2Pipeline.__call__ itself as the Turbo model card
runs it: is_distilled, so mu 1.15, guidance_scale 0, which is no
guidance, 8 steps, the encoder features of --features as prompt_embeds
with a mask that keeps every row, the noise as packed latents, and
output_type latent. The scheduler is check4_krea2's: the config
Krea2Pipeline documents, since the official one is gated. The noise is
numpy's PCG64 standard normal for the seed, which is how gmlx makes noise
from a seed.

The bf16 run follows the same loop with the DiT in bfloat16 and the
latents, the timestep and the scheduler's Euler update in float32. The
script runs that loop in fp32 too for --tie steps and holds it, bit for
bit, to the latents the pipeline's own call has after each of them. In
bf16 the linear products run as float32 products over bfloat16 operands
with one bfloat16 rounding.

For each size the fixture holds the noise, the features, the schedule
of the pipeline's call, the fp32 latents after the steps of --keep and
after the last one, and for a size in --bf16 the final bf16 latents. The
manifest records the bf16 against fp32 gap after every step and sets the
tolerance of the final latents at twice that gap. A size without a bf16
run takes the tolerance of the largest size that has one.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_krea2.py --dit Krea-2-Turbo-Q4_K_M.gguf \\
      --features check2_krea2_plain_features.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check3_krea2 as c3
import fixtures
import oracle
from check4_krea2 import CONFIG

NAME = "check6_krea2"
SIZES = (512,)
STEPS = 8
SEED = 6
KEEP = (1, 2, 4)
TURBO_MU = 1.15
RULE = ("final latents and kept steps: relative L2 at twice the oracle's "
        "own bf16 against fp32 relative L2 after that step at that size; "
        "a size without a bf16 run takes the tolerances of the largest "
        "size that has one")


def scheduler(torch, steps: int):
    """The Euler scheduler and its timesteps as Krea2Pipeline sets them
    for the distilled checkpoint."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

    sched = FlowMatchEulerDiscreteScheduler(**CONFIG)
    timesteps, _ = retrieve_timesteps(
        sched, steps, torch.device("cpu"),
        sigmas=np.linspace(1.0, 1 / steps, steps), mu=TURBO_MU)
    sched.set_begin_index(0)
    return sched, timesteps


def embeds(torch, feats: np.ndarray, layers: int, dtype):
    n = feats.shape[0]
    e = torch.from_numpy(feats.reshape(1, n, layers, -1)).to(dtype)
    return e, torch.ones((1, n), dtype=torch.bool)


def pipeline_run(torch, pipe, noise: np.ndarray, feats: np.ndarray,
                 layers: int, steps: int):
    """The pipeline's own call. Returns the final latents, the latents
    after every step and the seconds of every step."""
    _, _, h, w = noise.shape
    after, marks = [], [time.perf_counter()]

    def keep(_pipe, _i, _t, kw):
        after.append(c3.unpack(oracle.to_numpy(kw["latents"]), h, w))
        marks.append(time.perf_counter())
        return {}

    e, mask = embeds(torch, feats, layers, torch.float32)
    (lat,) = pipe(prompt_embeds=e, prompt_embeds_mask=mask,
                  height=h * 8, width=w * 8, num_inference_steps=steps,
                  guidance_scale=0.0,
                  latents=torch.from_numpy(c3.pack(noise)),
                  output_type="latent", return_dict=False,
                  callback_on_step_end=keep)
    return c3.unpack(oracle.to_numpy(lat), h, w), after, np.diff(marks)


def loop(torch, model, noise: np.ndarray, feats: np.ndarray, layers: int,
         mode: str, steps: int, stop: int | None = None):
    """The pipeline's denoise loop on ``model`` with float32 latents.
    Returns the latents after every step and the seconds of every
    step."""
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    _, _, h, w = noise.shape
    dtype = oracle.torch_dtype(mode)
    sched, timesteps = scheduler(torch, steps)
    e, mask = embeds(torch, feats, layers, dtype)
    ids = Krea2Pipeline.prepare_position_ids(feats.shape[0], h // 2, w // 2,
                                             "cpu")
    lat = torch.from_numpy(c3.pack(noise))
    after, secs = [], []
    for t in timesteps[:stop]:
        t0 = time.perf_counter()
        ts = (t / 1000).expand(1).to(torch.float32)
        with torch.no_grad():
            v = model(hidden_states=lat.to(dtype), encoder_hidden_states=e,
                      timestep=ts, position_ids=ids,
                      encoder_attention_mask=mask, return_dict=False)[0]
        lat = sched.step(v.float(), t, lat, return_dict=False)[0]
        after.append(c3.unpack(oracle.to_numpy(lat), h, w))
        secs.append(time.perf_counter() - t0)
    return after, np.array(secs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", required=True, type=Path,
                    help="Krea 2 Turbo DiT GGUF")
    ap.add_argument("--features", required=True, type=Path,
                    help=".npz with features (L, layers * dim)")
    ap.add_argument("--features-source",
                    default="check2_krea2 fp32 features of the plain prompt")
    ap.add_argument("--blocks", type=int, help="keep the first N blocks")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--bf16", type=int, nargs="*", default=[512],
                    help="sizes that also run in bf16")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--keep", type=int, nargs="*", default=list(KEEP),
                    help="steps whose fp32 latents the fixture keeps")
    ap.add_argument("--tie", type=int, default=2,
                    help="steps of the loop held to the pipeline in fp32")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if any(s % 16 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 16, and --bf16 among --sizes")
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    import diffusers
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    t0 = time.perf_counter()
    state, census = c3.gguf_state(torch, a.dit, a.blocks)
    cfg = dict(c3.inspect_defaults(diffusers))
    if a.blocks is not None:
        cfg["num_layers"] = a.blocks
    models = {m: c3.build(torch, state, cfg, m)
              for m in (("fp32", "bf16") if a.bf16 else ("fp32",))}
    print(f"[check6] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    layers = cfg["num_text_layers"]
    c_lat = cfg["in_channels"] // 4
    feats = np.load(a.features)["features"].astype(np.float32)
    sched, _ = scheduler(torch, a.steps)
    pipe = Krea2Pipeline(scheduler=sched, vae=None, text_encoder=None,
                         tokenizer=None, transformer=models["fp32"],
                         is_distilled=True)
    pipe.set_progress_bar_config(disable=True)

    fx = fixtures.FixtureWriter(
        a.name, check=6, family="krea2",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--dit", a.dit.name, "--features",
              a.features.name, "--sizes", *map(str, a.sizes), "--bf16",
              *map(str, a.bf16)])
    fx.add_file("dit", a.dit)
    fx.add_file("features", a.features)
    fx.add("features", feats)
    results, tol, sigmas = {}, {}, {}
    for size in a.sizes:
        side = size // 8
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, c_lat, side, side), dtype=np.float32)
        with torch.no_grad():
            final, after, secs = pipeline_run(torch, pipe, noise, feats,
                                              layers, a.steps)
        sigmas[str(size)] = [float(s) for s in pipe.scheduler.sigmas]
        if len(after) != a.steps or not np.array_equal(after[-1], final):
            raise SystemExit(f"error: {size}: {len(after)} steps kept")
        tied, _ = loop(torch, models["fp32"], noise, feats, layers, "fp32",
                       a.steps, a.tie)
        same = [bool(np.array_equal(x, y)) for x, y in zip(tied, after)]
        if not all(same):
            raise SystemExit(f"error: {size}: the loop parts from the "
                             f"pipeline in fp32: {same}")
        fx.add(f"noise_{size}", noise)
        fx.add(f"latents_{size}", final)
        fx.add(f"sigmas_{size}", oracle.to_numpy(
            pipe.scheduler.sigmas).astype(np.float32))
        fx.add(f"timesteps_{size}", oracle.to_numpy(
            pipe.scheduler.timesteps).astype(np.float32))
        for k in a.keep:
            if k < a.steps:
                fx.add(f"latents_{size}_step{k}", after[k - 1])
        row = {"fp32_step_s": [round(float(s), 1) for s in secs],
               "fp32_s": round(float(secs.sum()), 1),
               "loop_equals_pipeline_steps": len(same)}
        print(f"[check6] {size}: fp32 {row['fp32_s']} s, median step "
              f"{np.median(secs):.1f} s", flush=True)
        if size in a.bf16:
            oracle.emulate_bf16_matmul(True)
            t1 = time.perf_counter()
            after16, _ = loop(torch, models["bf16"], noise, feats, layers,
                              "bf16", a.steps)
            oracle.emulate_bf16_matmul(False)
            fx.add(f"latents_{size}_bf16", after16[-1])
            gap = fixtures.compare(after16[-1], final)
            row.update(bf16_s=round(time.perf_counter() - t1, 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=[
                           fixtures.rel_l2(x, y)
                           for x, y in zip(after16, after)])
            tol[str(size)] = {"latents_rel_l2": 2 * gap["rel_l2"]}
            print(f"[check6] {size}: bf16 {row['bf16_s']} s, final latents "
                  f"rel L2 {gap['rel_l2']:.3e}, cosine {gap['cosine']:.6f}, "
                  f"by step {[f'{v:.2e}' for v in row['bf16_vs_fp32_by_step']]}",
                  flush=True)
        results[str(size)] = row
    if tol:
        base = tol[str(max(int(s) for s in tol))]
        for size in a.sizes:
            tol.setdefault(str(size), {**base, "from": "another size"})

    fx.inputs.update(
        sizes=list(a.sizes), bf16_sizes=list(a.bf16), steps=a.steps,
        blocks=cfg["num_layers"], variant="turbo", mu=TURBO_MU,
        guidance_scale=0.0, scheduler=CONFIG,
        schedule="the pipeline's default, np.linspace(1, 1 / steps, steps)",
        sigmas=sigmas, seed=a.seed, features=a.features_source,
        text_rows=int(feats.shape[0]), kept_steps=[k for k in a.keep
                                                   if k < a.steps],
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, shape (1, 16, size / 8, size / 8)",
        latents="(1, 16, size / 8, size / 8) float32 in model space, "
                "after the last step", gguf_codecs=census)
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "dequant": oracle.dequant_mode(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                "rounding", "timestep": "float32 in both modes"},
        sizes=results)
    fx.tolerances.update(rule=RULE, sizes=tol)
    fx.note("The official scheduler config is gated. The scheduler is the "
            "one Krea2Pipeline's docstring asks for, as in check4_krea2.")
    path = fx.write(a.out)
    print(f"[check6] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
