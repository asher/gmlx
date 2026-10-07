#!/usr/bin/env python3
"""Check 6 oracle for Qwen-Image 2.1 with reference images: a guided edit
run on injected noise, to the final latents.

The fp32 run of a case is QwenImage21Pipeline.__call__ itself: a prompt,
its reference images, a negative prompt, true_cfg_scale above 1, the key
and value cache on, the noise as latents and output_type latent. The
pipeline resizes the images, encodes them with the VAE in float32 and
lays them out before the target itself. Its encode_prompt alone is
replaced: the call cannot take prompt_embeds with an image, so the
replacement returns the float32 features and the image rows that the
check 2 fixture with images holds for the prompt, after it has held the
resized images the pipeline passes it to the fixture's, bit for bit.

The bf16 run of a case follows the same loop with the DiT in bfloat16,
as check6_qwen_image21.py describes it, with the two velocities of a
step combined in float32 as the pipeline combines them. The script runs
that loop in fp32 too for --tie steps and holds it, bit for bit, to the
latents the pipeline's own call has after each of them.

The fixture holds the noise, the reference latents of each image in
model space as the pipeline made them, (64, h / 16, w / 16), and per case
the fp32 latents after the steps of --keep and after the last one, and
for a case in --bf16 the final bf16 latents. The tolerance of a case is
twice its own bf16 against fp32 gap, and a case without a bf16 run takes
the tolerance of the first case that has one.

The stages run in order and each keeps its result under --work, so a
run that stops, by itself at --budget-s or otherwise, goes on from there
when it is started again: refs (the VAE encodes; it also writes
cond_<case>.npz, the conditioning of a case as check3_qwen_image21.py
--conditioning reads it), fp32:<case>, bf16:<case> (which goes on from
its last finished step) and write.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check6_qwen_image21_edit.py --dit qwen-image-2.1-Q4_K_M.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import check3_qwen_image21 as c3
import check6_qwen_image21 as c6
import fixtures
import oracle
from check2_qwen_image21 import text_processor

NAME = "check6_qwen_image21_edit"
CHECK2 = "check2_qwen_image21_edit"
# case: its prompt and its negative prompt, as cases of the check 2 fixture.
CASES = {"one": ("one", "one_neg"), "two": ("two", "two_neg")}
SIZE = 512
STEPS = 40
SEED = 6
SCALE = 4.0
KEEP = (1, 2, 10, 20, 30)
STARTING = {"latents_rel_l2": 5e-2}
RULE = ("final latents: relative L2 at twice the oracle's own bf16 against "
        "fp32 relative L2 of the case; a case without a bf16 run starts "
        "from the tolerance of the first case that has one")


def load_check2(a, root: Path) -> tuple[dict, dict]:
    m2 = fixtures.read_manifest(a.check2, root=root)
    with np.load(root / m2["npz"], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files
                  if k.startswith(("source_", "resized_", "slots_"))
                  or (k.startswith("tap_") and not k.startswith("tap_bf16"))}
    for k, v in arrays.items():
        if fixtures.array_sha256(v) != m2["tensors"][k]["sha256"]:
            raise SystemExit(f"error: {k} of {a.check2} differs from its "
                             "manifest")
    return arrays, m2


def image_mask(arrays: dict, key: str) -> np.ndarray:
    mask = np.zeros(arrays[f"tap_{key}"].shape[0], bool)
    mask[arrays[f"slots_{key}"]] = True
    return mask


def vae_pipe(torch, a):
    """The pipeline with its VAE in float32 and no encoder."""
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline

    vae = AutoencoderKLQwenImage21.from_pretrained(
        str(a.config / "vae"), torch_dtype=torch.float32,
        local_files_only=True).eval()
    sched, _ = c6.scheduler(torch, a, 1024)
    pipe = QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=None,
                               processor=text_processor(a.config),
                               transformer=None)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def reference_latents(torch, a, pipe, src: np.ndarray):
    """The resized image and its latents (C, h, w) in model space, by the
    pipeline's own resize, preprocess and VAE encode."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_dimensions)
    from PIL import Image

    img = Image.fromarray(src)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    iw, ih, _ = calculate_dimensions(a.resolution * a.resolution, w / h)
    resized = pipe.image_processor.resize(img, width=iw, height=ih)
    x = pipe.image_processor.preprocess(img, width=iw,
                                        height=ih).unsqueeze(2)
    with torch.no_grad():
        z = pipe._encode_vae_image(x.to(torch.float32), None)
    return np.asarray(resized), oracle.to_numpy(z[0, :, 0])


def stage_refs(torch, a, arrays: dict, m2: dict, work: Path) -> None:
    out = work / "refs.npz"
    if out.is_file():
        return
    pipe = vae_pipe(torch, a)
    refs, secs = {}, {}
    for name in m2["inputs"]["images"]:
        t0 = time.perf_counter()
        resized, z = reference_latents(torch, a, pipe, arrays[f"source_{name}"])
        if not np.array_equal(resized, arrays[f"resized_{name}"]):
            raise SystemExit(f"error: image {name} resizes to other pixels "
                             f"than {a.check2} holds")
        refs[f"ref_latents_{name}"] = z
        secs[name] = round(time.perf_counter() - t0, 1)
        print(f"[check6e] reference {name}: latents {z.shape}, "
              f"{secs[name]} s", flush=True)
    for case in a.cases:
        images = m2["inputs"]["cases"][CASES[case][0]]["images"]
        for key in CASES[case]:
            cond = {"features": arrays[f"tap_{key}"].astype(np.float32),
                    "img_mask": image_mask(arrays, key)}
            cond.update({f"ref_latents.{i}": refs[f"ref_latents_{n}"]
                         for i, n in enumerate(images)})
            np.savez(work / f"cond_{key}.npz", **cond)
    np.savez(out, **refs)


def conditioning(arrays: dict, m2: dict, refs: dict, key: str) -> dict:
    images = m2["inputs"]["cases"][key]["images"]
    lat = [refs[f"ref_latents_{n}"] for n in images]
    return {"features": arrays[f"tap_{key}"].astype(np.float32),
            "img_mask": image_mask(arrays, key),
            "ref_shapes": [tuple(z.shape[-2:]) for z in lat],
            "ref_latents": lat}


def noise_of(a, c_in: int) -> np.ndarray:
    side = a.size // 16
    rng = np.random.Generator(np.random.PCG64(a.seed))
    return rng.standard_normal((1, c_in, side, side), dtype=np.float32)


def replay(torch, arrays: dict, m2: dict, keys: tuple):
    """encode_prompt of the pipeline, from the check 2 fixture."""
    by_prompt = {m2["inputs"]["cases"][k]["prompt"]: k for k in keys}
    used = []

    def encode_prompt(prompt=None, image=None, **kw):
        if kw.get("prompt_embeds") is not None or not isinstance(prompt, str):
            raise SystemExit("error: the replay takes one prompt string")
        key = by_prompt[prompt]
        names = m2["inputs"]["cases"][key]["images"]
        got = [np.asarray(i) for i in (image or [])]
        if len(got) != len(names) or not all(
                np.array_equal(g, arrays[f"resized_{n}"])
                for g, n in zip(got, names)):
            raise SystemExit(f"error: the pipeline resized the images of "
                             f"{key} to other pixels than the fixture's")
        used.append(key)
        return (torch.from_numpy(arrays[f"tap_{key}"].astype(np.float32))[None],
                None, torch.from_numpy(image_mask(arrays, key))[None])

    return encode_prompt, used


def pipeline_run(torch, a, pipe, arrays, m2, refs, case: str,
                 noise: np.ndarray):
    """The pipeline's own call. Returns the final latents, the latents
    after every step and the seconds of every step."""
    from PIL import Image

    pos, neg = CASES[case]
    rows = m2["inputs"]["cases"]
    after, marks, seen = [], [time.perf_counter()], {}

    def keep(_pipe, _i, _t, kw):
        after.append(c6.unpack(kw["latents"], noise))
        marks.append(time.perf_counter())
        return {}

    prepare = pipe.prepare_latents

    def prepared(*args, **kw):
        lat, img = prepare(*args, **kw)
        seen["image_latents"] = img
        return lat, img

    pipe.prepare_latents = prepared
    pipe.encode_prompt, used = replay(torch, arrays, m2, (pos, neg))
    try:
        (lat,) = pipe(
            prompt=rows[pos]["prompt"],
            image=[Image.fromarray(arrays[f"source_{n}"])
                   for n in rows[pos]["images"]],
            negative_prompt=rows[neg]["prompt"], true_cfg_scale=a.scale,
            height=a.size, width=a.size, num_inference_steps=a.steps,
            latents=c6.pack(torch, noise), output_type="latent",
            return_dict=False, callback_on_step_end=keep,
            output_resolution=a.resolution)
    finally:
        del pipe.prepare_latents, pipe.encode_prompt
    if used != [pos, neg]:
        raise SystemExit(f"error: {case}: the pipeline encoded {used}")
    # The pipeline's own reference latents are those of the refs stage.
    want = torch.cat([c6.pack(torch, refs[f"ref_latents_{n}"][None])
                      for n in rows[pos]["images"]], dim=1)
    if not torch.equal(seen["image_latents"], want):
        raise SystemExit(f"error: {case}: the pipeline's reference latents "
                         "differ from the refs stage")
    return c6.unpack(lat, noise), after, np.diff(marks)


def cache_digest(caches) -> str:
    h = hashlib.sha256()
    for cache in caches:
        for layer in cache.layer_caches:
            for t in (layer.k, layer.v):
                h.update(oracle.to_numpy(t.float()).tobytes())
    return h.hexdigest()


def loop(torch, a, model, noise: np.ndarray, conds: list, mode: str, *,
         stop: int | None = None, state: Path | None = None, clock=None):
    """The pipeline's guided denoise loop on ``model`` with float32
    latents: the positive and the negative conditioning each with its own
    cache. Returns the latents after every step done and the seconds of
    each. With ``state`` the loop keeps its latents there after every
    step and goes on from them, and stops when ``clock`` says so."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache)

    sched, timesteps = c6.scheduler(torch, a, noise.shape[2] * noise.shape[3])
    tokens = noise.shape[2] * noise.shape[3]
    lat, after, secs, digest = noise.copy(), [], [], None
    if state is not None and state.is_file():
        with np.load(state, allow_pickle=False) as z:
            after = list(z["after"])
            secs = list(z["secs"])
            digest = str(z["cache"])
        lat = after[-1]
        sched.set_begin_index(len(after))
    caches = [QwenImage21KVCache(len(model.transformer_blocks))
              for _ in conds]

    def step(i, t, kv_mode):
        outs = [c3.call(torch, model, c, lat, t, mode, a, cache,
                        kv_mode)[:, -tokens:].float()
                for c, cache in zip(conds, caches)]
        pred = outs[1] + a.scale * (outs[0] - outs[1])
        return c6.unpack(sched.step(pred, t, c6.pack(torch, lat),
                                    return_dict=False)[0], noise)

    start = len(after)
    if start:
        # The cache is a function of the conditioning alone: one extract
        # call builds it again, and it must be the cache of the first run.
        for c, cache in zip(conds, caches):
            c3.call(torch, model, c, lat, timesteps[start], mode, a, cache,
                    "extract")
        if cache_digest(caches) != digest:
            raise SystemExit("error: the cache built again differs from "
                             "the cache of the first step")
    for i in range(start, len(timesteps[:stop])):
        if clock is not None and clock():
            break
        t0 = time.perf_counter()
        lat = step(i, timesteps[i], "extract" if i == 0 else "cached")
        after.append(lat)
        secs.append(time.perf_counter() - t0)
        if i == 0 and state is not None:
            digest = cache_digest(caches)
        if state is not None:
            tmp = state.with_suffix(".tmp.npz")
            np.savez(tmp, after=np.stack(after), secs=np.array(secs),
                     cache=np.array(digest))
            tmp.replace(state)
    return after, np.array(secs)


def stage_fp32(torch, a, ctx: dict, case: str, work: Path) -> None:
    out = work / f"fp32_{case}.npz"
    if out.is_file():
        return
    model, pipe = ctx["model"]("fp32"), ctx["pipe"]()
    arrays, m2, refs = ctx["arrays"], ctx["m2"], ctx["refs"]()
    noise = noise_of(a, model.config.in_channels)
    final, after, secs = pipeline_run(torch, a, pipe, arrays, m2, refs,
                                      case, noise)
    if len(after) != a.steps or not np.array_equal(after[-1], final):
        raise SystemExit(f"error: {case}: {len(after)} steps kept")
    conds = [conditioning(arrays, m2, refs, k) for k in CASES[case]]
    tied, _ = loop(torch, a, model, noise, conds, "fp32", stop=a.tie)
    same = [bool(np.array_equal(x, y)) for x, y in zip(tied, after)]
    if not all(same):
        raise SystemExit(f"error: {case}: the loop parts from the pipeline "
                         f"in fp32: {same}")
    np.savez(out, noise=noise, after=np.stack(after), secs=secs,
             tied=np.array(len(same)),
             sigmas=oracle.to_numpy(pipe.scheduler.sigmas).astype(np.float32),
             timesteps=oracle.to_numpy(
                 pipe.scheduler.timesteps).astype(np.float32))
    print(f"[check6e] {case}: fp32 {secs.sum():.1f} s, first step "
          f"{secs[0]:.1f} s, later steps {np.median(secs[1:]):.1f} s",
          flush=True)


def stage_bf16(torch, a, ctx: dict, case: str, work: Path, clock) -> bool:
    """True when the run of the case is whole."""
    state = work / f"bf16_{case}.npz"
    model = ctx["model"]("bf16")
    arrays, m2, refs = ctx["arrays"], ctx["m2"], ctx["refs"]()
    noise = noise_of(a, model.config.in_channels)
    conds = [conditioning(arrays, m2, refs, k) for k in CASES[case]]
    oracle.emulate_bf16_matmul(True)
    try:
        after, secs = loop(torch, a, model, noise, conds, "bf16",
                           state=state, clock=clock)
    finally:
        oracle.emulate_bf16_matmul(False)
    print(f"[check6e] {case}: bf16 {len(after)} of {a.steps} steps, "
          f"{secs.sum():.1f} s", flush=True)
    return len(after) == a.steps


def stage_write(a, ctx: dict, work: Path, root: Path) -> Path:
    arrays, m2 = ctx["arrays"], ctx["m2"]
    with np.load(work / "refs.npz", allow_pickle=False) as z:
        refs = {k: z[k] for k in z.files}
    fx = fixtures.FixtureWriter(
        a.name, check=6, family="qwen_image21",
        script=f"scripts/image_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--dit", a.dit.name, "--config",
              a.config.name, "--cases", *a.cases, "--bf16", *a.bf16])
    fx.add_file("dit", a.dit)
    fx.add_file("vae", a.config / "vae" / "diffusion_pytorch_model.safetensors")
    fx.add_file("vae_config", a.config / "vae" / "config.json")
    results, tol, cases, used = {}, {}, {}, set()
    for case in a.cases:
        pos, neg = CASES[case]
        with np.load(work / f"fp32_{case}.npz", allow_pickle=False) as z:
            f32 = {k: z[k] for k in z.files}
        after = f32["after"]
        if "noise" not in fx.arrays:
            fx.add("noise", f32["noise"])
            fx.add("sigmas", f32["sigmas"])
            fx.add("timesteps", f32["timesteps"])
        elif not np.array_equal(fx.arrays["noise"], f32["noise"]):
            raise SystemExit("error: the cases ran from other noise")
        fx.add(f"latents_{case}", after[-1])
        for k in a.keep:
            if k < a.steps:
                fx.add(f"latents_{case}_step{k}", after[k - 1])
        for key in (pos, neg):
            fx.add(f"features_{key}", arrays[f"tap_{key}"].astype(np.float32))
            fx.add(f"slots_{key}", arrays[f"slots_{key}"])
        images = m2["inputs"]["cases"][pos]["images"]
        used.update(images)
        row = {"fp32_step_s": [round(float(s), 1) for s in f32["secs"]],
               "fp32_s": round(float(f32["secs"].sum()), 1),
               "loop_equals_pipeline_steps": int(f32["tied"])}
        if case in a.bf16:
            with np.load(work / f"bf16_{case}.npz", allow_pickle=False) as z:
                b16, secs = z["after"], z["secs"]
            if len(b16) != a.steps:
                raise SystemExit(f"error: the bf16 run of {case} holds "
                                 f"{len(b16)} of {a.steps} steps")
            fx.add(f"latents_{case}_bf16", b16[-1])
            gap = fixtures.compare(b16[-1], after[-1])
            row.update(bf16_s=round(float(secs.sum()), 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=[fixtures.rel_l2(x, y)
                                             for x, y in zip(b16, after)])
            tol[case] = {"latents_rel_l2": 2 * gap["rel_l2"]}
        results[case] = row
        cases[case] = {
            "prompt": m2["inputs"]["cases"][pos]["prompt"],
            "negative_prompt": m2["inputs"]["cases"][neg]["prompt"],
            "images": images, "features": f"tap_{pos} of {a.check2}",
            "negative_features": f"tap_{neg} of {a.check2}",
            "reference_latents": [list(refs[f"ref_latents_{n}"].shape)
                                  for n in images]}
    for name in sorted(used):
        fx.add(f"ref_latents_{name}", refs[f"ref_latents_{name}"])
        fx.add(f"resized_{name}", arrays[f"resized_{name}"])
        fx.add(f"source_{name}", arrays[f"source_{name}"])
    if tol:
        first = next(c for c in a.cases if c in tol)
        for case in a.cases:
            tol.setdefault(case, {**tol[first], "from": first})
    fx.inputs.update(
        cases=cases, size=a.size, reference_resolution=a.resolution,
        steps=a.steps, seed=a.seed, guidance=a.scale, kv_cache=True,
        bf16_cases=list(a.bf16),
        kept_steps=[k for k in a.keep if k < a.steps],
        schedule="the pipeline's default, np.linspace(1, 1 / steps, steps)",
        config_revision=a.config_revision,
        noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
              "dtype=float32), NCHW, shape (1, 64, size / 16, size / 16)",
        latents="(1, 64, size / 16, size / 16) float32 in model space, "
                "after the last step",
        reference_latents="(64, h / 16, w / 16) float32 in model space: "
                          "the mode of the VAE in float32 on the resized "
                          "RGBA image, less latents_mean over latents_std",
        guidance_rule="negative + guidance x (positive - negative), in "
                      "float32 on the two velocities of a step",
        encode_prompt=f"replaced by the features and image rows of "
                      f"{a.check2}, after the resized images the pipeline "
                      "passes are held to that fixture's in bits")
    plain = work / "plain_decoded_fp32.json"
    if not plain.is_file():
        ctx["model"]("fp32")
    fx.results.update(
        oracle={"device": "cpu", "dequant": oracle.dequant_mode(),
                "bf16_matmul": "bf16 operands, float32 GEMM, one bf16 "
                "rounding", "bf16_timestep": a.bf16_timestep,
                "plain_decoded": json.loads(plain.read_text())},
        cases=results)
    fx.tolerances.update(rule=RULE, starting=STARTING, cases=tol)
    return fx.write(root)


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
                    help="check 2 fixture with images that holds the "
                         "features")
    ap.add_argument("--cases", nargs="+", default=list(CASES),
                    choices=list(CASES))
    ap.add_argument("--bf16", nargs="*", default=["one"],
                    help="cases that also run in bf16")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--resolution", type=int, default=512,
                    help="output_resolution of the call, which sizes the "
                         "references; the check 2 fixture's")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--scale", type=float, default=SCALE,
                    help="true_cfg_scale")
    ap.add_argument("--keep", type=int, nargs="*", default=list(KEEP),
                    help="steps whose fp32 latents the fixture keeps")
    ap.add_argument("--tie", type=int, default=2,
                    help="steps of the loop held to the pipeline in fp32")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--bf16-timestep", choices=("fp32", "stock"),
                    default="fp32")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--stages", nargs="+",
                    help="stages to run, in order: refs, fp32:<case>, "
                         "bf16:<case>, write (default: all)")
    ap.add_argument("--budget-s", type=float,
                    help="start no stage and no bf16 step after this many "
                         "seconds")
    ap.add_argument("--work", type=Path,
                    help="where the stages keep their results (default "
                         "<fixtures>/<name>.work)")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    a.sigmas = None
    if a.size % 32 or set(a.bf16) - set(a.cases):
        ap.error("--size must be a multiple of 32, and --bf16 among --cases")
    torch = oracle.setup()
    t0 = time.perf_counter()
    root = fixtures.fixture_root(a.out)
    work = a.work or root / f"{a.name}.work"
    work.mkdir(parents=True, exist_ok=True)
    arrays, m2 = load_check2(a, root)
    if m2["inputs"]["output_resolution"] != a.resolution:
        ap.error(f"{a.check2} holds references at "
                 f"{m2['inputs']['output_resolution']}")
    stages = a.stages or ["refs", *[f"fp32:{c}" for c in a.cases],
                          *[f"bf16:{c}" for c in a.bf16], "write"]
    held: dict = {}

    def model(mode: str):
        if mode not in held:
            held[mode] = c3.load_model(torch, a.dit, a.config, mode,
                                       a.dequant)
            # What the load decoded outside a linear layer, for the
            # manifest.
            (work / f"plain_decoded_{mode}.json").write_text(
                json.dumps(list(held[mode].plain_decoded)))
        return held[mode]

    def refs() -> dict:
        with np.load(work / "refs.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def pipe():
        if "pipe" not in held:
            p = vae_pipe(torch, a)
            p.register_modules(transformer=model("fp32"))
            held["pipe"] = p
        return held["pipe"]

    def clock() -> bool:
        return (a.budget_s is not None
                and time.perf_counter() - t0 > a.budget_s)

    ctx = {"arrays": arrays, "m2": m2, "model": model, "refs": refs,
           "pipe": pipe}
    for stage in stages:
        if clock():
            print(f"[check6e] stopped before {stage}: start again to go on",
                  flush=True)
            return 0
        kind, _, case = stage.partition(":")
        if kind == "refs":
            stage_refs(torch, a, arrays, m2, work)
        elif kind == "fp32":
            stage_fp32(torch, a, ctx, case, work)
        elif kind == "bf16":
            if not stage_bf16(torch, a, ctx, case, work, clock):
                print(f"[check6e] stopped in {stage}: start again to go on",
                      flush=True)
                return 0
        elif kind == "write":
            print(f"[check6e] wrote {stage_write(a, ctx, work, root)}")
        else:
            ap.error(f"unknown stage {stage!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
