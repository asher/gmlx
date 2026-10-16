#!/usr/bin/env python3
"""Check 6 oracle for FLUX.2 klein: a whole sampling run on injected
noise, to the final latents, for a distilled checkpoint and for Base.

The run is the text to image path of BFL's scripts/cli.py on the
reference's commit, read from --reference (its src directory), with the
DiT of --dit built as in check3_flux2_klein: the prompt's features, and
for Base the empty prompt's before them, through batched_prc_txt, the
noise in bf16 through batched_prc_img, get_schedule(steps, image tokens),
and sampling.denoise for a distilled checkpoint or sampling.denoise_cfg
for Base. Steps and guidance are the defaults of FLUX2_MODEL_INFO in
src/flux2/util.py for the variant of --variant at the file's size. No
reference images.

The features are those of --features, the check 2 oracle's float32
features of the plain prompt and of the empty prompt, rounded to bf16 as
the CLI rounds the encoder's output. The noise is numpy's PCG64 standard
normal for the seed, NCHW, which is how gmlx makes noise from a seed,
rounded to bf16 where the CLI draws it in bf16.

The bf16 run is the CLI's: the DiT, the latents and every op of the loop
in bfloat16, products as one float32 GEMM over bf16 operands with one
rounding (oracle.emulate_bf16_matmul). The fp32 run keeps the latents and
every op in float32. In both the time goes into the DiT in bf16, as the
CLI fills it from its bf16 latents, and its embedding comes back in bf16.

One fixture per variant and size holds the noise, the features, the
schedule, the time the DiT saw at each step, the fp32 latents after the
steps of --keep and after the last one, and with a bf16 run the final
bf16 latents and the bf16 against fp32 gap after every step. At GATE_SIZE
the tolerance of gmlx's bf16 latents after a step is twice that gap, so
that size needs a bf16 run. At every other size gmlx's exact fp32 run is
held to EXACT_REL_L2 after each kept step and the last, and its bf16 run
is read against the GATE_SIZE gates.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_flux2_klein.py --reference <flux2>/src \\
      --reference-commit <sha> --dit flux-2-klein-4b-Q4_K_M.gguf \\
      --variant distilled --features check2_flux2_klein.npz \\
      --sizes 512 1024 --bf16 512

--self-test runs the same path on a stub DiT whose velocity is a table of
the step plus the first text feature less the latents, rounded to bf16,
and keeps the latents after every step, so the gmlx loop can be held to
the reference's arithmetic bit for bit.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

import check3_flux2_klein as c3
import fixtures
import oracle

NAME = "check6_flux2_klein"
SCRIPT = "scripts/image_oracle/check6_flux2_klein.py"
SEED = 6
KEEP = {"distilled": (1, 2, 3), "base": (1, 2, 5, 10, 25)}
PROMPT = "plain"
GATE_SIZE = 512
EXACT_REL_L2 = 1e-3
RULE = (f"latents after a kept step and after the last. At {GATE_SIZE}: "
        "gmlx's bf16 run at relative L2 twice the oracle's own bf16 "
        "against fp32 relative L2 after that step. At any other size: "
        "gmlx's exact fp32 run against the fp32 latents at exact_rel_l2, "
        f"and its bf16 run read as a share of the {GATE_SIZE} gate of the "
        "step, not gated")
# The stub run of --self-test: the latent grid and channels, the text
# rows and their width, and the steps of each variant.
STUB = {"latent": (4, 6), "channels": 8, "text": 5, "text_dim": 4,
        "steps": {"distilled": 4, "base": 6}}


def reference(path: Path):
    """The reference's model, sampling and util modules. sampling imports
    torchvision for its image preparation, which the run does not use and
    the oracle environment lacks, so an empty module stands in while they
    import. diffusers and transformers look torchvision up by its spec,
    which the stand-in lacks, so the names they give import first, and the
    stand-in leaves sys.modules after."""
    from diffusers.quantizers.gguf.utils import GGUFParameter  # noqa: F401
    from transformers import (AutoModelForCausalLM, AutoProcessor,  # noqa: F401
                              AutoTokenizer, Mistral3ForConditionalGeneration,
                              pipeline)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    if "torchvision" in sys.modules:
        raise SystemExit("error: torchvision is imported")
    sys.modules["torchvision"] = types.ModuleType("torchvision")
    try:
        from flux2 import model, sampling, util
    finally:
        del sys.modules["torchvision"]
    return model, sampling, util


def model_info(util, fm, state, variant: str) -> dict:
    """The FLUX2_MODEL_INFO row of the file's size and the variant."""
    size = "4b" if c3.params_of(fm, state).hidden_size == 3072 else "9b"
    key = (f"flux.2-klein-{size}" if variant == "distilled"
           else f"flux.2-klein-base-{size}")
    row = util.FLUX2_MODEL_INFO[key]
    if row["guidance_distilled"] != (variant == "distilled"):
        raise SystemExit(f"error: {key} is not {variant}")
    return {"key": key, **row["defaults"]}


class Recorded:
    """The DiT as denoise calls it. At each call it keeps sample 0's image
    latents, the time, the rows and the moment, and passes the time in
    bf16, as the CLI fills it from its bf16 latents."""

    def __init__(self, torch, model, n_img: int):
        self.torch, self.model, self.n_img = torch, model, n_img
        self.latents, self.times, self.rows, self.marks = [], [], [], []

    def __call__(self, **kw):
        t = kw["timesteps"].to(self.torch.bfloat16)
        kw["timesteps"] = t
        self.latents.append(oracle.to_numpy(kw["x"][0, :self.n_img]))
        self.times.append(float(t[0].float()))
        self.rows.append(int(kw["x"].shape[0]))
        self.marks.append(time.perf_counter())
        return self.model(**kw)


def to_tokens(lat: np.ndarray) -> np.ndarray:
    _, c, h, w = lat.shape
    return np.ascontiguousarray(lat[0].reshape(c, h * w).T)


def to_latents(tok: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.ascontiguousarray(tok.T.reshape(1, tok.shape[1], h, w))


def generate(torch, sampling, model, variant: str, feats: dict,
             noise: np.ndarray, steps: int, guidance: float, dtype):
    """The CLI's text to image path on ``model``, the latents and the
    loop in ``dtype``. Returns the latents after every step, the times,
    the rows of each call, the seconds of every step and the schedule."""
    _, _, h, w = noise.shape
    prompt = torch.from_numpy(feats[PROMPT][None].copy()).to(torch.bfloat16)
    if variant == "distilled":
        ctx = prompt
    else:
        empty = torch.from_numpy(feats[""][None].copy()).to(torch.bfloat16)
        ctx = torch.cat([empty, prompt], dim=0)
    ctx, ctx_ids = sampling.batched_prc_txt(ctx)
    randn = torch.from_numpy(noise.copy()).to(torch.bfloat16)
    x, x_ids = sampling.batched_prc_img(randn)
    timesteps = sampling.get_schedule(steps, x.shape[1])
    rec = Recorded(torch, model, x.shape[1])
    run = sampling.denoise if variant == "distilled" else sampling.denoise_cfg
    t0 = time.perf_counter()
    out = run(rec, x.to(dtype), x_ids, ctx.to(dtype), ctx_ids,
              timesteps=timesteps, guidance=guidance)
    end = time.perf_counter()
    if len(rec.latents) != steps:
        raise SystemExit(f"error: {len(rec.latents)} DiT calls for {steps} "
                         f"steps")
    want = oracle.to_numpy(randn)
    if not np.array_equal(to_latents(rec.latents[0], h, w), want):
        raise SystemExit("error: the first latents are not the noise "
                         "rounded to bf16")
    after = [to_latents(t, h, w)
             for t in rec.latents[1:] + [oracle.to_numpy(out[0])]]
    secs = np.diff(np.array([t0, *rec.marks[1:], end]))
    return (after, np.float32(rec.times), rec.rows, secs,
            np.asarray(timesteps, np.float32))


def check_times(torch, schedule: np.ndarray, times) -> None:
    """The times the DiT saw: each the bf16 of its schedule entry."""
    want = oracle.to_numpy(torch.from_numpy(schedule[:-1]).to(
        torch.bfloat16))
    if not np.array_equal(times, want):
        raise SystemExit("error: the DiT saw times off the schedule")


def load_features(torch, path: Path) -> dict:
    z = np.load(path)
    return {PROMPT: c3.bf16(torch, z["features_plain"]),
            "": c3.bf16(torch, z["features_empty"])}


# The self-test.

class StubDiT:
    """The velocity of sample s at call k: (table[k] + ctx[s, 0, 0]) less
    the latents, rounded to bf16."""

    def __init__(self, torch, table: np.ndarray):
        self.torch, self.step = torch, 0
        self.table = torch.from_numpy(np.stack([to_tokens(t[None])
                                                for t in table]))

    def __call__(self, *, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        torch = self.torch
        out = [((self.table[self.step] + ctx[s, 0, 0].float()) - x[s])
               .to(torch.bfloat16) for s in range(x.shape[0])]
        self.step += 1
        return torch.stack(out)


def self_test(torch, sampling, util, commit: str, a) -> int:
    rng = np.random.default_rng(a.seed)
    h, w = STUB["latent"]
    c = STUB["channels"]
    noise = rng.standard_normal((1, c, h, w), dtype=np.float32)
    feats = {PROMPT: c3.bf16(torch, rng.standard_normal(
                 (STUB["text"], STUB["text_dim"]))),
             "": c3.bf16(torch, rng.standard_normal(
                 (STUB["text"], STUB["text_dim"])))}
    fx = fixtures.FixtureWriter("selftest_flux2_klein_loop", check=6,
                                family="flux2_klein", script=SCRIPT)
    fx.add("noise", noise)
    fx.add("features", feats[PROMPT])
    fx.add("features_empty", feats[""])
    guidance = {}
    for variant in ("distilled", "base"):
        key = ("flux.2-klein-4b" if variant == "distilled"
               else "flux.2-klein-base-4b")
        g = util.FLUX2_MODEL_INFO[key]["defaults"]["guidance"]
        steps = STUB["steps"][variant]
        table = rng.standard_normal((steps, c, h, w), dtype=np.float32)
        after, times, rows, _, schedule = generate(
            torch, sampling, StubDiT(torch, table), variant, feats, noise,
            steps, g, torch.bfloat16)
        check_times(torch, schedule, times)
        fx.add(f"{variant}_table", table)
        fx.add(f"{variant}_schedule", schedule)
        fx.add(f"{variant}_times", times)
        fx.add(f"{variant}_latents", np.concatenate(after))
        fx.results[variant] = {"steps": steps, "guidance": g, "rows": rows,
                               "model_info": key}
        guidance[variant] = g
    fx.inputs.update(stub=STUB, seed=a.seed, reference_commit=commit,
                     tokens=h * w,
                     velocity="bf16((table[step] + ctx[s, 0, 0]) - latents) "
                              "per sample s, its own text",
                     latents="after every step, (steps, C, h, w) float32")
    fx.note("the CLI's text to image path on the stub DiT, the latents in "
            "bf16, sampling.denoise for distilled and denoise_cfg for Base "
            "with the empty prompt's text in row 0")
    path = fx.write(a.out)
    print(f"[check6] self-test wrote {path}; guidance {guidance}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux2")
    ap.add_argument("--reference-commit", required=True,
                    help="the commit of --reference, for the manifest")
    ap.add_argument("--dit", type=Path, help="FLUX.2 klein DiT GGUF")
    ap.add_argument("--variant", choices=("distilled", "base"),
                    help="the checkpoint --dit holds")
    ap.add_argument("--features", type=Path,
                    help="check 2 .npz with features_plain and "
                         "features_empty")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--bf16", type=int, nargs="*", default=[512],
                    help="sizes that also run in bf16")
    ap.add_argument("--steps", type=int,
                    help="steps, FLUX2_MODEL_INFO's for the variant when "
                         "unset")
    ap.add_argument("--keep", type=int, nargs="*",
                    help="steps whose fp32 latents the fixture keeps")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", help="fixture name before the size, "
                    f"{NAME} or {NAME}_base by the variant when unset")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    fm, sampling, util = reference(a.reference.expanduser())
    commit = a.reference_commit
    if a.self_test:
        return self_test(torch, sampling, util, commit, a)
    if a.dit is None or a.features is None or a.variant is None:
        ap.error("--dit, --variant and --features are required")
    if any(s % 16 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 16, and --bf16 among --sizes")
    if GATE_SIZE in a.sizes and GATE_SIZE not in a.bf16:
        ap.error(f"the {GATE_SIZE} gate is the bf16 run's gap: pass --bf16 "
                 f"{GATE_SIZE}")
    variant = a.variant
    t0 = time.perf_counter()
    state, census = c3.gguf_state(torch, a.dit)
    info = model_info(util, fm, state, variant)
    modes = ("fp32", "bf16") if a.bf16 else ("fp32",)
    models = {m: c3.build(torch, fm, state, m) for m in modes}
    print(f"[check6] loaded {a.dit.name} ({variant}, {info['key']}) in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    steps = a.steps or info["num_steps"]
    guidance = info["guidance"]
    keep = [k for k in (KEEP[variant] if a.keep is None else a.keep)
            if k < steps]
    feats = load_features(torch, a.features)
    name = a.name or (NAME if variant == "distilled" else f"{NAME}_base")
    c = c3.params_of(fm, state).in_channels
    for size in a.sizes:
        n = size // 16
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, c, n, n), dtype=np.float32)
        fx = fixtures.FixtureWriter(f"{name}_{size}", check=6,
                                    family="flux2_klein", script=SCRIPT)
        fx.add_file("dit", a.dit)
        fx.add_file("features", a.features)
        fx.add("noise", noise)
        fx.add("features", feats[PROMPT])
        if variant == "base":
            fx.add("features_empty", feats[""])
        oracle.emulate_bf16_matmul(False)
        after, times, rows, secs, schedule = generate(
            torch, sampling, models["fp32"], variant, feats, noise, steps,
            guidance, torch.float32)
        check_times(torch, schedule, times)
        fx.add("schedule", schedule)
        fx.add("times", times)
        fx.add("latents", after[-1])
        for k in keep:
            fx.add(f"latents_step{k}", after[k - 1])
        row = {"fp32_s": round(float(secs.sum()), 1),
               "fp32_median_step_s": round(float(np.median(secs)), 2),
               "rows": sorted(set(rows))}
        print(f"[check6] {variant} {size}: fp32 {row['fp32_s']} s, rows "
              f"{row['rows']}", flush=True)
        if size in a.bf16:
            oracle.emulate_bf16_matmul(True)
            try:
                after16, times16, _, secs16, _ = generate(
                    torch, sampling, models["bf16"], variant, feats, noise,
                    steps, guidance, torch.bfloat16)
            finally:
                oracle.emulate_bf16_matmul(False)
            if not np.array_equal(times16, times):
                raise SystemExit("error: the bf16 run saw other times")
            fx.add("latents_bf16", after16[-1])
            gap = fixtures.compare(after16[-1], after[-1])
            by_step = [fixtures.rel_l2(x, y) for x, y in zip(after16, after)]
            row.update(bf16_s=round(float(secs16.sum()), 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=by_step)
            if size == GATE_SIZE:
                fx.tolerances["latents_rel_l2"] = 2 * gap["rel_l2"]
                fx.tolerances["by_step_rel_l2"] = [2 * g for g in by_step]
            print(f"[check6] {variant} {size}: bf16 {row['bf16_s']} s, final "
                  f"rel L2 {gap['rel_l2']:.3e}, by step "
                  f"{[f'{v:.2e}' for v in by_step]}", flush=True)
        fx.results.update(
            oracle={"device": "cpu", "threads": torch.get_num_threads(),
                    "dequant": oracle.dequant_mode(),
                    "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                                   "rounding"},
            run=row)
        fx.inputs.update(
            size=size, variant=variant, model_info=info["key"], steps=steps,
            guidance=guidance, seed=a.seed, bf16=size in a.bf16,
            kept_steps=keep, refs=0, tokens=n * n,
            noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
                  "dtype=float32), NCHW, shape (1, 128, size / 16, "
                  "size / 16); the CLI's noise is bf16, so it is rounded",
            features="check2_flux2_klein float32 features_plain, and "
                     "features_empty for Base, rounded to bf16",
            latents="(1, 128, size / 16, size / 16) float32 in the gmlx "
                    "layout, after the step",
            times="the bf16 time the DiT saw at each step, as float32",
            reference_commit=commit, gguf_codecs=census)
        if size != GATE_SIZE:
            fx.tolerances["exact_rel_l2"] = EXACT_REL_L2
        fx.tolerances["rule"] = RULE
        path = fx.write(a.out)
        print(f"[check6] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
