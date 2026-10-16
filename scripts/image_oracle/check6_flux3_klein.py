#!/usr/bin/env python3
"""Check 6 oracle for FLUX.3 klein: a whole sampling run on injected
noise, to the final latents, for the distilled flow map and for Base.

The run is BFL's own Pipeline.generate on the reference's commit, read
from --reference (its src directory), with the DiT of --dit built as in
check3_flux3_klein. generate packs the samples, makes the schedule
sampling.timesteps(steps, alpha) at its own alpha default, rounds the
noise to bf16 and keeps the latents float32, and runs
sampling.denoise_flow_map on one sample of the prompt for the flow map,
or sampling.denoise on two for Base: the unconditional sample of the
empty prompt, then the prompt's, each with the image, at the guidance of
pipeline.MODELS. Steps and guidance are those of pipeline.MODELS for the
variant the file's tensors name. No reference images.

Three parts of generate stand in for what the run does not load. The
text encoder returns the features of --features, the check 2 oracle's
float32 features of the plain prompt and of the empty prompt, rounded to
bf16 as the encoder returns them, and records the prompts it was asked
for. The autoencoder's decode keeps the latents it is given. torch.randn
returns numpy's PCG64 standard normal for the seed, NCHW, which is how
gmlx makes noise from a seed, as the (h w, 128) tokens generate asks for.

The fp32 run has every op in float32. The bf16 run has the DiT in
bfloat16, products as one float32 GEMM over bf16 operands with one
rounding, so the velocity, the guidance and the product of each Euler
step come out in bf16 as under the reference's autocast, while the
latents stay float32. In both, pre-hooks cast the tokens to the model's
dtype at the first product, as autocast does.

One fixture per variant and size holds the noise, the features, the
schedule, the time and target the DiT saw at each step, the fp32
latents after the steps of --keep and after the last one, and with a
bf16 run the final bf16 latents and the bf16 against fp32 gap after
every step. At GATE_SIZE the tolerance of gmlx's bf16 latents after a
step is twice that gap, so that size needs a bf16 run. At every other
size gmlx's exact fp32 run is held to EXACT_REL_L2 after each kept step
and the last, and its bf16 run is read against the GATE_SIZE gates.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_flux3_klein.py --reference ~/src/flux-3-image-klein/src \\
      --dit flux3_image_klein_sd-Q4_K_M.gguf \\
      --features check2_flux3_klein.npz --sizes 512 --bf16 512

--self-test runs generate the same way on a stub DiT whose velocity is a
table of the step less the latents plus the first text feature, rounded
to bf16, and keeps the latents after every step, so the gmlx loop can be
held to the reference's arithmetic bit for bit.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import sys
import time
from pathlib import Path

import numpy as np

import check3_flux3_klein as c3
import fixtures
import oracle

NAME = "check6_flux3_klein"
SCRIPT = "scripts/image_oracle/check6_flux3_klein.py"
SEED = 6
KEEP = {"distilled": (1, 2, 4, 8), "base": (1, 2, 5, 10, 25)}
PROMPT = "plain"
GATE_SIZE = 512
EXACT_REL_L2 = 1e-3
RULE = (f"latents after a kept step and after the last. At {GATE_SIZE}: "
        "gmlx's bf16 run at relative L2 twice the oracle's own bf16 "
        "against fp32 relative L2 after that step. At any other size: "
        "gmlx's exact fp32 run against the fp32 latents at exact_rel_l2, "
        f"and its bf16 run read as a share of the {GATE_SIZE} gate of the "
        "step, not gated")
# The stub run of --self-test: the latent grid and channels, and the
# text rows of the prompt and of the empty prompt.
STUB = {"latent": (4, 6), "channels": 8, "text": (3, 2), "text_dim": 4}


class TextEncoder:
    """Stands in for the reference's encoder: the features of a prompt as
    bf16 tokens at the reference's text ids."""

    def __init__(self, torch, feats: dict):
        self.torch, self.feats, self.asked = torch, feats, []

    def encode(self, prompt: str):
        torch = self.torch
        self.asked.append(prompt)
        f = self.feats["" if prompt == "" else PROMPT]
        pos = torch.arange(f.shape[0])
        ids = torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1)
        return torch.from_numpy(f.copy()).to(torch.bfloat16), ids


class AutoEncoder:
    """Keeps the latent tokens generate decodes."""

    def __init__(self, torch):
        self.torch, self.tokens = torch, None

    def decode(self, tokens, ids):
        self.tokens = tokens
        return self.torch.zeros((3, 16, 16))


class Steps:
    """A pre-hook on the DiT that keeps, at each call, sample 0's latents,
    the bf16 time of the image and the flow map's target."""

    def __init__(self, model):
        self.latents, self.times, self.targets, self.marks = [], [], [], []
        self.handle = model.register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, _m, args, kwargs):
        packed = args[0] if args else kwargs["packed"]
        a, b = packed.mode_indices["image"][0]
        self.latents.append(oracle.to_numpy(packed.data["image"][a:b]))
        self.times.append(float(packed.timesteps["image"][a].float()))
        ref = kwargs.get("timesteps_ref")
        self.targets.append(None if ref is None else float(ref[0]))
        self.marks.append(time.perf_counter())


@contextlib.contextmanager
def injected(torch, tokens: np.ndarray):
    """torch.randn returns ``tokens`` for the one call generate makes."""
    real = torch.randn

    def randn(size, *args, **kw):
        if tuple(size) != tokens.shape:
            raise SystemExit(f"error: generate asked for noise {tuple(size)}"
                             f", not {tokens.shape}")
        return torch.from_numpy(tokens.copy())

    torch.randn = randn
    try:
        yield
    finally:
        torch.randn = real


def to_tokens(lat: np.ndarray) -> np.ndarray:
    _, c, h, w = lat.shape
    return np.ascontiguousarray(lat[0].reshape(c, h * w).T)


def to_latents(tok: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.ascontiguousarray(tok.T.reshape(1, tok.shape[1], h, w))


def cast_inputs(torch, model, dtype) -> None:
    """Tokens in the model's dtype at the first product, as autocast casts
    them."""
    for mod in (model.emb_in["image"], model.emb_in["image_cond"],
                model.txt_in):
        mod.register_forward_pre_hook(
            lambda _m, args, dt=dtype: tuple(a.to(dt) for a in args))


def generate(torch, pipeline, model, variant: str, feats: dict,
             noise: np.ndarray, steps: int):
    """Pipeline.generate on ``model`` with the stand-ins. Returns the
    latents after every step, the times, the targets, the seconds of every
    step and the prompts encoded."""
    _, _, h, w = noise.shape
    p = object.__new__(pipeline.Pipeline)
    p.device, p.offload, p.dits = "cpu", False, {variant: model}
    p.text_encoder = TextEncoder(torch, feats)
    p.autoencoder = AutoEncoder(torch)
    rec = Steps(model)
    t0 = time.perf_counter()
    try:
        with injected(torch, to_tokens(noise)):
            p.generate(variant, PROMPT, [], w * 16, h * 16, steps,
                       pipeline.MODELS[variant]["guidance"], SEED)
    finally:
        rec.handle.remove()
    end = time.perf_counter()
    if len(rec.latents) != steps:
        raise SystemExit(f"error: {len(rec.latents)} DiT calls for {steps} "
                         f"steps")
    first = rec.latents[0]
    want = oracle.to_numpy(torch.from_numpy(to_tokens(noise)).to(
        torch.bfloat16))
    if not np.array_equal(first, want):
        raise SystemExit("error: the first latents are not the noise "
                         "rounded to bf16")
    final = oracle.to_numpy(p.autoencoder.tokens)
    after = [to_latents(x, h, w) for x in rec.latents[1:] + [final]]
    secs = np.diff(np.array([t0, *rec.marks[1:], end]))
    return (after, np.float32(rec.times), rec.targets, secs,
            list(p.text_encoder.asked))


def alpha_default(pipeline) -> float:
    return float(inspect.signature(pipeline.Pipeline.generate)
                 .parameters["alpha"].default)


def check_times(sampling, pipeline, variant: str, steps: int, times,
                targets) -> np.ndarray:
    """The schedule generate ran, held to the times and targets the DiT
    saw: each time the bf16 of its entry, each target its next entry."""
    import torch

    s = sampling.timesteps(steps, alpha_default(pipeline))
    want = oracle.to_numpy(s[:-1].to(torch.bfloat16))
    if not np.array_equal(times, want):
        raise SystemExit("error: the DiT saw times off the schedule")
    if variant == "distilled":
        if [float(np.float32(t)) for t in targets] != \
                [float(v) for v in oracle.to_numpy(s[1:])]:
            raise SystemExit("error: the targets are not the next times")
    elif any(t is not None for t in targets):
        raise SystemExit("error: Base got a target")
    return oracle.to_numpy(s)


def load_features(torch, path: Path) -> dict:
    z = np.load(path)
    return {PROMPT: c3.bf16(torch, z["features_plain"]),
            "": c3.bf16(torch, z["features_empty"])}


# The self-test.

class StubDiT:
    """The velocity of sample s at step k: (table[k] + text_s[0, 0]) less
    the latents, rounded to bf16."""

    def __init__(self, torch, flow_map: bool, table: np.ndarray):
        self.torch, self.flow_map, self.step = torch, flow_map, 0
        self.table = torch.from_numpy(np.stack([to_tokens(t[None])
                                                for t in table]))
        self.hooks = []

    def register_forward_pre_hook(self, fn, with_kwargs=False):
        self.hooks.append(fn)
        stub = self

        class Handle:
            def remove(self):
                stub.hooks.remove(fn)
        return Handle()

    def __call__(self, packed, timesteps_ref=None):
        torch = self.torch
        for fn in self.hooks:
            fn(self, (packed,), {"timesteps_ref": timesteps_ref})
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a "
                             "flow-map DiT")
        out = []
        for s in range(packed.num_samples):
            a, b = packed.mode_indices["image"][s]
            ta, _ = packed.mode_indices["text"][s]
            x = packed.data["image"][a:b]
            t0 = packed.data["text"][ta, 0].float()
            out.append(((self.table[self.step] + t0) - x).to(torch.bfloat16))
        self.step += 1
        return torch.cat(out)


def self_test(torch, dit, pipeline, sampling, commit: str, a) -> int:
    rng = np.random.default_rng(a.seed)
    h, w = STUB["latent"]
    c = STUB["channels"]
    # pipeline holds the same dict.
    dit.IN_CHANNELS["image"] = c
    noise = rng.standard_normal((1, c, h, w), dtype=np.float32)
    n_plain, n_empty = STUB["text"]
    feats = {PROMPT: c3.bf16(torch, rng.standard_normal(
                 (n_plain, STUB["text_dim"]))),
             "": c3.bf16(torch, rng.standard_normal(
                 (n_empty, STUB["text_dim"])))}
    fx = fixtures.FixtureWriter("selftest_flux3_klein_loop", check=6,
                                family="flux3_klein", script=SCRIPT)
    fx.add("noise", noise)
    fx.add("features", feats[PROMPT])
    fx.add("features_empty", feats[""])
    for variant in ("distilled", "base"):
        steps = pipeline.MODELS[variant]["steps"]
        table = rng.standard_normal((steps, c, h, w), dtype=np.float32)
        stub = StubDiT(torch, variant == "distilled", table)
        after, times, targets, _, asked = generate(
            torch, pipeline, stub, variant, feats, noise, steps)
        schedule = check_times(sampling, pipeline, variant, steps, times,
                               targets)
        fx.add(f"{variant}_table", table)
        fx.add(f"{variant}_schedule", schedule)
        fx.add(f"{variant}_times", times)
        fx.add(f"{variant}_latents", np.concatenate(after))
        fx.results[variant] = {"steps": steps, "prompts": asked,
                               "guidance": pipeline.MODELS[variant][
                                   "guidance"]}
    fx.inputs.update(stub=STUB, seed=a.seed, reference_commit=commit,
                     alpha=alpha_default(pipeline),
                     velocity="bf16((table[step] + text[0, 0]) - latents) "
                              "per sample, the text of that sample",
                     latents="after every step, (steps, C, h, w) float32")
    fx.note("Pipeline.generate on the stub DiT with the stand-in encoder, "
            "autoencoder and noise of the real run")
    path = fx.write(a.out)
    print(f"[check6] self-test wrote {path}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--dit", type=Path, help="FLUX.3 klein DiT GGUF")
    ap.add_argument("--features", type=Path,
                    help="check 2 .npz with features_plain and "
                         "features_empty")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512])
    ap.add_argument("--bf16", type=int, nargs="*", default=[512],
                    help="sizes that also run in bf16")
    ap.add_argument("--steps", type=int,
                    help="steps, pipeline.MODELS' for the variant when unset")
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
    dit, pipeline, sampling, commit = c3.reference(a.reference.expanduser())
    if a.self_test:
        return self_test(torch, dit, pipeline, sampling, commit, a)
    if a.dit is None or a.features is None:
        ap.error("--dit and --features are required")
    if any(s % 16 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 16, and --bf16 among --sizes")
    if GATE_SIZE in a.sizes and GATE_SIZE not in a.bf16:
        ap.error(f"the {GATE_SIZE} gate is the bf16 run's gap: pass --bf16 "
                 f"{GATE_SIZE}")
    t0 = time.perf_counter()
    state, census, flow_map = c3.gguf_state(torch, a.dit)
    variant = "distilled" if flow_map else "base"
    modes = ("fp32", "bf16") if a.bf16 else ("fp32",)
    models = {}
    for m in modes:
        models[m] = c3.build(torch, dit, state, flow_map, m)
        cast_inputs(torch, models[m], oracle.torch_dtype(m))
    print(f"[check6] loaded {a.dit.name} ({variant}) in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    steps = a.steps or pipeline.MODELS[variant]["steps"]
    keep = [k for k in (KEEP[variant] if a.keep is None else a.keep)
            if k < steps]
    feats = load_features(torch, a.features)
    name = a.name or (NAME if flow_map else f"{NAME}_base")
    c = dit.IN_CHANNELS["image"]
    for size in a.sizes:
        n = size // 16
        rng = np.random.Generator(np.random.PCG64(a.seed))
        noise = rng.standard_normal((1, c, n, n), dtype=np.float32)
        fx = fixtures.FixtureWriter(f"{name}_{size}", check=6,
                                    family="flux3_klein", script=SCRIPT)
        fx.add_file("dit", a.dit)
        fx.add_file("features", a.features)
        fx.add("noise", noise)
        fx.add("features", feats[PROMPT])
        if variant == "base":
            fx.add("features_empty", feats[""])
        after, times, targets, secs, asked = generate(
            torch, pipeline, models["fp32"], variant, feats, noise, steps)
        schedule = check_times(sampling, pipeline, variant, steps, times,
                               targets)
        fx.add("schedule", schedule)
        fx.add("times", times)
        fx.add("latents", after[-1])
        for k in keep:
            fx.add(f"latents_step{k}", after[k - 1])
        row = {"fp32_s": round(float(secs.sum()), 1),
               "fp32_median_step_s": round(float(np.median(secs)), 2),
               "prompts": asked}
        print(f"[check6] {variant} {size}: fp32 {row['fp32_s']} s, prompts "
              f"{asked}", flush=True)
        if size in a.bf16:
            oracle.emulate_bf16_matmul(True)
            try:
                after16, times16, _, secs16, _ = generate(
                    torch, pipeline, models["bf16"], variant, feats, noise,
                    steps)
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
            size=size, variant=variant, flow_map=flow_map, steps=steps,
            guidance=pipeline.MODELS[variant]["guidance"],
            alpha=alpha_default(pipeline), seed=a.seed,
            bf16=size in a.bf16, kept_steps=keep, refs=0,
            noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
                  "dtype=float32), NCHW, shape (1, 128, size / 16, "
                  "size / 16); generate rounds it to bf16",
            features="check2_flux3_klein float32 features_plain, and "
                     "features_empty for Base, rounded to bf16",
            latents="(1, 128, size / 16, size / 16) float32 tokens in the "
                    "gmlx layout, after the step",
            times="the bf16 image time the DiT saw at each step, as float32",
            reference_commit=commit, gguf_codecs=census)
        if size != GATE_SIZE:
            fx.tolerances["exact_rel_l2"] = EXACT_REL_L2
        fx.tolerances["rule"] = RULE
        path = fx.write(a.out)
        print(f"[check6] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
