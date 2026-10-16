#!/usr/bin/env python3
"""Check 6 oracle for FLUX.3 klein edits: whole sampling runs with one
reference image and with two, on injected noise, to the final latents,
and the preparation of reference images of several sizes.

The references are prepared as the reference's generate.py prepares
them, from pipeline.py: one by prepare_reference, whose box crops the
output when the request gives no size, each of several by
prepare_multi_reference, and the size by default_size. The source
images are synthetic, made from --seed. The preparations of the sources
of PREPARE are kept beside those of the runs, so the sizing rules are
held at a resize, a reflect pad, a symmetric pad on a least side of 16
pixels, the least side of a canvas and the greatest side.

A run is BFL's own Pipeline.generate as in check6_flux3_klein, on the
same stand-ins for the text encoder and the noise, with BFL's
AutoEncoder in float32 from --autoencoder for the references: its
encode gives the reference tokens, and generate gives them the scaled
ids of their index. A pre-hook keeps the image_cond tokens and ids the
DiT read. Steps and guidance are pipeline.MODELS' for the variant of the
file. The fp32 and bf16 runs and the gates are those of
check6_flux3_klein, and the reference encode is float32 in both.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_flux3_klein_edit.py --reference ~/src/flux-3-image-klein/src \\
      --dit flux3_image_klein_sd-Q4_K_M.gguf --autoencoder ae.safetensors \\
      --features check2_flux3_klein.npz --cases one two
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import check3_flux3_klein as c3
import check6_flux3_klein as c6
import fixtures
import oracle

NAME = "check6_flux3_klein_edit"
SCRIPT = "scripts/image_oracle/check6_flux3_klein_edit.py"
SEED = 6
# Synthetic sources, (width, height, channels).
SOURCES = {"wide": (600, 400, 3), "square": (400, 400, 4),
           "large": (2400, 1800, 3), "small": (40, 24, 3),
           "long": (5000, 200, 3), "many": (1600, 1000, 3),
           "tiny": (20, 16, 3)}
# The edits the DiT runs: the sources of each, in order.
CASES = {"one": ("wide",), "two": ("wide", "square")}
# Sources prepared alone, as one reference, and as one of several.
PREPARE = {"single": ("wide", "square", "large", "small", "long", "tiny"),
           "multi": ("wide", "square", "large", "many")}
# The gate of a run, at the default size of its references.
RULE = ("latents after a kept step and after the last: relative L2 at "
        "twice the oracle's own bf16 against fp32 relative L2 after that "
        "step, on the run's own size")


def source(rng, width: int, height: int, channels: int) -> np.ndarray:
    """A picture of gradients, rectangles and a little noise, (H, W, C)
    uint8."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.stack([x / max(1, width - 1) * 255,
                    y / max(1, height - 1) * 255,
                    (x + y) / max(1, width + height - 2) * 255], axis=-1)
    for _ in range(12):
        x0, x1 = sorted(rng.integers(0, width, 2))
        y0, y1 = sorted(rng.integers(0, height, 2))
        img[y0:y1 + 1, x0:x1 + 1] = rng.integers(0, 256, 3)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    if channels == 4:
        alpha = rng.integers(0, 256, (height, width, 1), dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)
    return img


def pil(a: np.ndarray):
    from PIL import Image

    return Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")


def prepared(pipeline, images: list) -> tuple:
    """generate.py's preparation of ``images``: the prepared references,
    the crop box or None, and the default size (width, height)."""
    if len(images) > 1:
        refs, crop = [pipeline.prepare_multi_reference(im)
                      for im in images], None
    else:
        pairs = [pipeline.prepare_reference(im) for im in images]
        refs = [im for im, _ in pairs]
        crop = pairs[0][1] if pairs else None
    return refs, crop, pipeline.default_size(refs)


class AutoEncoder:
    """BFL's AutoEncoder in float32 for the references; its decode keeps
    the latent tokens generate decodes."""

    def __init__(self, torch, ae):
        self.torch, self.ae, self.tokens, self.refs = torch, ae, None, []

    def encode(self, image):
        t, i = self.ae.encode(image)
        self.refs.append(oracle.to_numpy(t.float()))
        return t, i

    def decode(self, tokens, ids):
        self.tokens = tokens
        return self.torch.zeros((3, 16, 16))


class Cond:
    """A pre-hook that keeps the image_cond tokens and ids of sample 0 at
    the first call."""

    def __init__(self, model):
        self.tokens = self.ids = None
        self.handle = model.register_forward_pre_hook(self, with_kwargs=True)

    def __call__(self, _m, args, kwargs):
        if self.tokens is not None:
            return
        packed = args[0] if args else kwargs["packed"]
        a, b = packed.mode_indices["image_cond"][0]
        self.tokens = oracle.to_numpy(packed.data["image_cond"][a:b].float())
        self.ids = oracle.to_numpy(packed.ids["image_cond"][a:b])


def generate(torch, pipeline, model, ae, variant: str, feats: dict,
             noise: np.ndarray, refs: list, steps: int):
    """Pipeline.generate with references on ``model``. Returns what
    check6_flux3_klein.generate does, the reference tokens of the
    encoder and the image_cond tokens and ids the DiT read."""
    _, _, h, w = noise.shape
    p = object.__new__(pipeline.Pipeline)
    p.device, p.offload, p.dits = "cpu", False, {variant: model}
    p.text_encoder = c6.TextEncoder(torch, feats)
    p.autoencoder = AutoEncoder(torch, ae)
    rec, cond = c6.Steps(model), Cond(model)
    t0 = time.perf_counter()
    try:
        with c6.injected(torch, c6.to_tokens(noise)):
            p.generate(variant, c6.PROMPT, refs, w * 16, h * 16, steps,
                       pipeline.MODELS[variant]["guidance"], SEED)
    finally:
        rec.handle.remove()
        cond.handle.remove()
    end = time.perf_counter()
    if len(rec.latents) != steps:
        raise SystemExit(f"error: {len(rec.latents)} DiT calls for {steps} "
                         f"steps")
    final = oracle.to_numpy(p.autoencoder.tokens)
    after = [c6.to_latents(x, h, w) for x in rec.latents[1:] + [final]]
    secs = np.diff(np.array([t0, *rec.marks[1:], end]))
    return (after, np.float32(rec.times), rec.targets, secs,
            list(p.text_encoder.asked), p.autoencoder.refs, cond)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--dit", type=Path, required=True,
                    help="FLUX.3 klein DiT GGUF")
    ap.add_argument("--autoencoder", type=Path, required=True,
                    help="BFL's ae.safetensors")
    ap.add_argument("--features", type=Path, required=True,
                    help="check 2 .npz with features_plain and "
                         "features_empty")
    ap.add_argument("--cases", nargs="*", default=list(CASES),
                    choices=list(CASES))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--steps", type=int,
                    help="steps, pipeline.MODELS' for the variant when unset")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    dit, pipeline, sampling, commit = c3.reference(a.reference.expanduser())
    from flux3 import autoencoder

    from safetensors.torch import load_file

    t0 = time.perf_counter()
    ae = autoencoder.AutoEncoder()
    ae.load_state_dict(load_file(str(a.autoencoder)), strict=True)
    ae = ae.float().eval()
    state, census, flow_map = c3.gguf_state(torch, a.dit)
    variant = "distilled" if flow_map else "base"
    modes = ("fp32",) if a.no_bf16 else ("fp32", "bf16")
    models = {}
    for m in modes:
        models[m] = c3.build(torch, dit, state, flow_map, m)
        c6.cast_inputs(torch, models[m], oracle.torch_dtype(m))
    print(f"[edit] loaded {a.dit.name} ({variant}) and the autoencoder in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    steps = a.steps or pipeline.MODELS[variant]["steps"]
    keep = [k for k in c6.KEEP[variant] if k < steps]
    feats = c6.load_features(torch, a.features)

    fx = fixtures.FixtureWriter(a.name, check=6, family="flux3_klein",
                                script=SCRIPT)
    fx.add_file("dit", a.dit)
    fx.add_file("autoencoder", a.autoencoder)
    fx.add_file("features", a.features)
    fx.add("features", feats[c6.PROMPT])
    if variant == "base":
        fx.add("features_empty", feats[""])
    rng = np.random.Generator(np.random.PCG64(a.seed))
    srcs = {k: source(rng, *v) for k, v in SOURCES.items()}
    for k, img in srcs.items():
        fx.add(f"source_{k}", img)
    preps = {}
    for rule, names in PREPARE.items():
        for k in names:
            if rule == "single":
                refs, crop, size = prepared(pipeline, [pil(srcs[k])])
            else:
                refs = [pipeline.prepare_multi_reference(pil(srcs[k]))]
                crop, size = None, refs[0].size
            fx.add(f"prepared_{rule}_{k}", np.asarray(refs[0]))
            preps[f"{rule}_{k}"] = {"size": list(size), "crop": crop and
                                    list(crop)}
    fx.inputs["prepared"] = preps
    runs = {}
    for case in a.cases:
        names = CASES[case]
        refs, crop, (width, height) = prepared(
            pipeline, [pil(srcs[k]) for k in names])
        h, w = height // 16, width // 16
        c = dit.IN_CHANNELS["image"]
        noise = np.random.Generator(np.random.PCG64(a.seed)).standard_normal(
            (1, c, h, w), dtype=np.float32)
        fx.add(f"noise_{case}", noise)
        after, times, targets, secs, asked, ref_tokens, cond = generate(
            torch, pipeline, models["fp32"], ae, variant, feats, noise,
            refs, steps)
        schedule = c6.check_times(sampling, pipeline, variant, steps, times,
                                  targets)
        if len(ref_tokens) != len(names):
            raise SystemExit(f"error: {len(ref_tokens)} reference encodes "
                             f"for {len(names)} references")
        lats = []
        for i, (r, t) in enumerate(zip(refs, ref_tokens)):
            lat = c6.to_latents(t, r.size[1] // 16, r.size[0] // 16)
            fx.add(f"ref_latents_{case}_{i}", lat)
            lats.append(lat)
        joined = np.concatenate([c6.to_tokens(x) for x in lats])
        bf = oracle.to_numpy(torch.from_numpy(joined).to(torch.bfloat16)
                             .float())
        if not np.array_equal(cond.tokens, bf):
            raise SystemExit("error: the DiT read other reference tokens")
        fx.add(f"ref_ids_{case}", cond.ids)
        fx.add(f"schedule_{case}", schedule)
        fx.add(f"times_{case}", times)
        fx.add(f"latents_{case}", after[-1])
        for k in keep:
            fx.add(f"latents_{case}_step{k}", after[k - 1])
        row = {"sources": list(names), "size": [width, height],
               "crop": crop and list(crop), "prompts": asked,
               "fp32_s": round(float(secs.sum()), 1)}
        print(f"[edit] {case}: {width}x{height}, crop {crop}, fp32 "
              f"{row['fp32_s']} s, prompts {asked}", flush=True)
        if "bf16" in models:
            oracle.emulate_bf16_matmul(True)
            try:
                after16, times16, _, secs16, _, refs16, _ = generate(
                    torch, pipeline, models["bf16"], ae, variant, feats,
                    noise, refs, steps)
            finally:
                oracle.emulate_bf16_matmul(False)
            if not np.array_equal(times16, times):
                raise SystemExit("error: the bf16 run saw other times")
            if not all(np.array_equal(x, y) for x, y in zip(refs16,
                                                            ref_tokens)):
                raise SystemExit("error: the bf16 run encoded other "
                                 "references")
            fx.add(f"latents_bf16_{case}", after16[-1])
            gap = fixtures.compare(after16[-1], after[-1])
            by_step = [fixtures.rel_l2(x, y) for x, y in zip(after16, after)]
            row.update(bf16_s=round(float(secs16.sum()), 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=by_step)
            fx.tolerances[f"by_step_rel_l2_{case}"] = [2 * g for g in by_step]
            print(f"[edit] {case}: bf16 {row['bf16_s']} s, final rel L2 "
                  f"{gap['rel_l2']:.3e}, by step "
                  f"{[f'{v:.2e}' for v in by_step]}", flush=True)
        runs[case] = row
    fx.results.update(
        oracle={"device": "cpu", "threads": torch.get_num_threads(),
                "dequant": oracle.dequant_mode(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                               "rounding",
                "autoencoder": "float32"},
        runs=runs)
    fx.inputs.update(
        variant=variant, flow_map=flow_map, steps=steps,
        guidance=pipeline.MODELS[variant]["guidance"],
        alpha=c6.alpha_default(pipeline), seed=a.seed, kept_steps=keep,
        cases={k: list(CASES[k]) for k in a.cases}, sources=SOURCES,
        bf16="bf16" in models,
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, (1, 128, height / 16, width / 16) "
              "of the default size; generate rounds it to bf16",
        sources_rule="numpy Generator(PCG64(seed)) in SOURCES order: "
                     "gradients, 12 rectangles, normal noise of 6, and a "
                     "random alpha for 4 channels",
        ref_latents="float32 tokens of BFL's float32 encode, (1, 128, h, "
                    "w); the DiT reads them in bf16",
        ref_ids="the image_cond ids the DiT read, (L, 4) float32",
        reference_commit=commit, gguf_codecs=census)
    fx.tolerances["rule"] = RULE
    path = fx.write(a.out)
    print(f"[edit] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
