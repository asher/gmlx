#!/usr/bin/env python3
"""Check 6 oracle for MiniMax-H3-Image: a whole text to image run of
ComfyUI's MiniMaxH3ImageModel on seeded noise, to the final latents.

The DiT loads as check3_h3_image.py loads it: ComfyUI-GGUF's
UnetLoaderGGUF with dequant_dtype float32 on the copy of --source with
the config key, torch's SDPA. The run is the six steps of the PDD grid,
the schedule gmlx samples with: each step one forward as ComfyUI's
sampler makes it, latents (1, 24, 1, H, W) at the step's sigma, the
timestep sigma x 1000 in float32 and the grid as the sampler's sigmas,
then an Euler step in float32, x + (sigma_next - sigma) v. Guidance is 1,
so ComfyUI runs no negative pass. The text features are check 2's
float32 tap of one prompt (--features, --prompt). The noise is numpy's
PCG64 standard normal for the seed, NCHW, which is how gmlx makes noise
from a seed.

The fp32 run has every op in float32. The bf16 run (--bf16 sizes) is the
same run with the features cast to bfloat16 and each linear a float32
GEMM over bfloat16 operands rounded once; the latents stay float32 at the
model boundary. At GATE_SIZE the manifest sets the tolerance of gmlx's
bf16 latents after each step to twice the oracle's own bf16 against fp32
gap after that step, the image ladder's rule, so that size needs a bf16
run. At every other size gmlx's exact fp32 run is held to EXACT_REL_L2,
or to twice gmlx's own exact floor where that is larger, which the gmlx
test records.

At the sizes of --sampler-check the fp32 run also goes through ComfyUI's
own sampler (comfy.sample.sample_custom, euler, on the grid), and its
final latents against the loop's are a report column that gates nothing.

--edit runs an edit at GATE_SIZE instead, under the same rule. Its npz
holds the pictures, the prompt's features with their vision rows and the
vision spans, from gmlx's encode of the prompt with its pictures, which
both sides take as given. Each picture gets its two copies from the
helpers of ComfyUI's MiniMaxH3ImageConditioning, which must equal the
copies the npz holds, since the features read its vision copies. Picture
1's VAE copy must be GATE_SIZE square, the size the node gives the edit.
ComfyUI's MiniMaxH3ImageVAE (--vae) encodes each VAE copy in fp32 with its
own draws, seed 42 over the whole latent, and every step carries the
minimax_payload MiniMaxH3ImageConditioning and model_base give it, as
check3_h3_image.py's edit step does, at the run seed. The fixture also
holds the pictures, both copies, the draws and latents of each encode and
the noise the forward mixes into each reference's rows. It runs no
sampler check.

One fixture per size holds the noise, the features, the schedule, the
fp32 latents and velocity of every step, and with a bf16 run its latents
after every step. Each step logs its time and the process's peak
footprint, so the times of the smaller sizes price a larger one.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check6_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --dit <copy with the config key> --source <release GGUF> \\
      --features check2_h3_image.npz --sizes 1024 --bf16 1024
  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check6_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --dit <copy with the config key> --source <release GGUF> \\
      --edit edit-inputs.npz --vae minimax_h3_image_vae_fp16.safetensors
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

import check3_h3_image as c3
import fixtures
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen", "safetensors")

NAME = "check6_h3_image"
SCRIPT = "scripts/image_oracle/check6_h3_image.py"
SEED = 6
GATE_SIZE = 1024
EXACT_REL_L2 = 1e-3
RULE = (f"latents after every step. At {GATE_SIZE}: gmlx's bf16 run at "
        "relative L2 twice the oracle's own bf16 against fp32 relative L2 "
        "after that step. At any other size: gmlx's exact fp32 run against "
        "the fp32 latents at the larger of exact_rel_l2 and twice gmlx's "
        f"own exact floor, and its bf16 run read as a share of the "
        f"{GATE_SIZE} gate of the step, not gated")
EULER = "x + (sigma_next - sigma) v in float32, the velocity in float32"
_F32 = np.float32


def _max_rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def noise_from_seed(seed: int, size: int) -> np.ndarray:
    n = size // 16
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.standard_normal((1, 24, n, n), dtype=np.float32)


def run(torch, dm, noise, feats, mode: str, tag: str, steps: int,
        payload: dict | None = None) -> tuple:
    """The latents after each step, each step's velocity and seconds.
    ``payload`` is an edit's minimax_payload, given to every step."""
    grid = torch.tensor(dm.pdd_sigmas, dtype=torch.float32)
    sig = np.asarray(dm.pdd_sigmas, _F32)
    ctx = torch.from_numpy(feats[None]).to(oracle.torch_dtype(mode))
    x = noise.copy()
    lat, vel, secs = [], [], []
    kw = {} if payload is None else {"minimax_payload": payload}
    oracle.emulate_bf16_matmul(mode == "bf16")
    try:
        for i in range(steps):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = dm(torch.from_numpy(x[:, :, None]),
                         c3.timestep(torch, float(sig[i])), ctx,
                         transformer_options={"sample_sigmas": grid},
                         **{k: dict(v) for k, v in kw.items()})
            v = oracle.to_numpy(out[:, :, 0]).astype(_F32)
            x = (x + (sig[i + 1] - sig[i]) * v).astype(_F32)
            lat.append(x.copy())
            vel.append(v)
            secs.append(time.perf_counter() - t0)
            print(f"[check6] {tag} {mode} step {i + 1}/{steps} "
                  f"{secs[-1]:.1f} s, max rss {_max_rss():.1f} GiB",
                  flush=True)
    finally:
        oracle.emulate_bf16_matmul(False)
    return lat, vel, secs


def edit_refs(torch, path: Path, vae_path: Path, seed: int) -> dict:
    """The arrays of the edit in ``path`` as ComfyUI makes them: each
    picture's two copies by MiniMaxH3ImageConditioning's helpers, which
    must equal the npz's, each VAE copy's encode by MiniMaxH3ImageVAE in
    fp32 with its seed 42 draws, and the minimax_payload of the steps at
    ``seed``. Also the share of each encode its draws make."""
    import comfy_extras.nodes_minimax_h3 as nm
    from comfy.ldm.minimax import vae as cv

    import check5_h3_image_vae as c5

    with np.load(path) as z:
        given = {k: z[k] for k in z.files}
    count = sum(k.startswith("picture_") for k in given)
    low, high = nm.IMAGE_REF_SHORT_EDGE, nm.IMAGE_REF_VAE_MAX_SHORT_EDGE
    vae = c5.load(torch, c5._build(False), vae_path, "fp32")
    real, held = vae._adaptive_encode, {}

    def spy(x):
        held["moments"] = real(x)
        return held["moments"]
    vae._adaptive_encode = spy
    stats = [getattr(vae, k).view(1, -1, 1, 1, 1) for k in c5.STATS]
    out: dict = {"features": given["features"].astype(_F32),
                 "spans": [tuple(int(v) for v in sp)
                           for sp in given["spans"]],
                 "prompt": str(given["prompt"])}
    refs, share = [], []
    for i in range(1, count + 1):
        pic = given[f"picture_{i}"]
        out[f"picture_{i}"] = pic
        img = torch.from_numpy(pic.astype(_F32) / 255)[None]
        h, w = pic.shape[:2]
        for kind, size in (("vae", nm._size_short_edge(w, h, low, high)),
                           ("vision", nm._size_short_edge(w, h, low, low))):
            c = nm._resize_to(img, size)[0].numpy()
            u8 = np.rint(c * 255).astype(np.uint8)
            if not (np.array_equal(u8.astype(_F32) / 255, c)
                    and np.array_equal(u8, given[f"{kind}_copy_{i}"])):
                raise SystemExit(f"error: ComfyUI's {kind} copy of picture "
                                 f"{i} differs from the one {path.name} "
                                 "holds")
            out[f"{kind}_copy_{i}"] = u8
        x = torch.from_numpy(out[f"vae_copy_{i}"].astype(_F32) / 255)
        x = x[None].movedim(-1, 1).unsqueeze(2) * 2 - 1
        with torch.no_grad():
            lat = vae.encode(x)
        eps = torch.randn(lat.shape, generator=torch.Generator().manual_seed(
            cv.IMAGE_ENCODE_SEED))
        mean, logvar = held.pop("moments").float().chunk(2, dim=1)
        again = (torch.addcmul(mean, torch.exp(0.5 * logvar.clamp(
            *c5.LOGVAR_RANGE)), eps) - stats[0]) / stats[1]
        if not torch.equal(again, lat):
            raise SystemExit("error: the class's encode is not its moments' "
                             "sample with the seed 42 draws")
        mode = ((mean - stats[0]) / stats[1])[:, :, 0].numpy()
        z = lat[:, :, 0].float().numpy()
        out[f"ref_vae_noise_{i - 1}"] = eps[0, :, 0].numpy()
        out[f"ref_latents_{i - 1}"] = z
        share.append(fixtures.rel_l2(z, mode))
        refs.append(z)
    del vae
    payload = c3.edit_payload(torch, refs, out["features"].shape[0],
                              out["spans"], seed)
    for i, n in enumerate(c3.ref_noise(torch, refs, seed)):
        out[f"ref_noise_{i}"] = n
    out["text_tags"] = payload["text_token_tags"].numpy()
    return {"arrays": out, "payload": payload, "count": count,
            "draw_share": share}


def comfy_sampler(torch, patcher, dm, noise, feats, steps: int):
    """The fp32 run through ComfyUI's own sampler: sample_custom with
    euler on the grid, guidance 1, an empty latent and the noise given.
    Returns the final latents in model space and the seconds."""
    import comfy.sample
    import comfy.samplers

    sig = torch.tensor(dm.pdd_sigmas[:steps + 1], dtype=torch.float32)
    pos = [[torch.from_numpy(feats[None]).float(), {}]]
    neg = [[torch.zeros_like(pos[0][0]), {}]]
    lat = torch.zeros((1, 24, 1) + noise.shape[2:], dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = comfy.sample.sample_custom(
            patcher, torch.from_numpy(noise[:, :, None]), 1.0,
            comfy.samplers.sampler_object("euler"), sig, pos, neg, lat,
            disable_pbar=True, seed=0)
        out = patcher.model.process_latent_in(out)
    return oracle.to_numpy(out[:, :, 0]).astype(_F32), \
        time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, required=True,
                    help="H3-Image DiT GGUF with the config key")
    ap.add_argument("--source", type=Path, required=True,
                    help="the release GGUF that --dit copies")
    ap.add_argument("--features", type=Path,
                    help="check 2 fixture .npz with tap_<prompt>")
    ap.add_argument("--prompt", default="plain",
                    help="prompt key of --features")
    ap.add_argument("--sizes", type=int, nargs="+", default=[GATE_SIZE])
    ap.add_argument("--bf16", type=int, nargs="*", default=[GATE_SIZE],
                    help="sizes that also run in bf16")
    ap.add_argument("--sampler-check", type=int, nargs="*",
                    help="sizes whose fp32 run also goes through ComfyUI's "
                         "sampler, a report column (default the gate "
                         "size, none with --edit)")
    ap.add_argument("--edit", type=Path,
                    help="npz of an edit's pictures, features and vision "
                         "spans: run that edit at the gate size")
    ap.add_argument("--vae", type=Path,
                    help="minimax_h3_image_vae_fp16.safetensors, for --edit")
    ap.add_argument("--steps", type=int, default=len(c3.PDD_SIGMAS) - 1,
                    help="steps to run, fewer for a probe")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_IMAGE_FIXTURES)")
    a = ap.parse_args(argv)
    if a.edit is not None:
        if a.vae is None or a.features is not None or a.sampler_check:
            ap.error("--edit takes --vae, and no --features or "
                     "--sampler-check")
        if a.sizes != [GATE_SIZE]:
            ap.error(f"--edit runs at the gate size, {GATE_SIZE}")
    elif a.features is None:
        ap.error("--features is required without --edit")
    if a.sampler_check is None:
        a.sampler_check = [] if a.edit else [GATE_SIZE]
    if any(s % 32 for s in a.sizes) or set(a.bf16) - set(a.sizes):
        ap.error("sizes must be multiples of 32, and --bf16 among --sizes")
    if GATE_SIZE in a.sizes and GATE_SIZE not in a.bf16:
        ap.error(f"size {GATE_SIZE} sets the gates and needs a bf16 run")
    torch = oracle.setup()
    co = c3.comfy_oracle()
    comfy = co.start_comfy(a.comfy_dir)
    a.dit = a.dit.expanduser().resolve()
    comfy_gguf = co.load_comfy_gguf(a.comfy_dir)
    import comfy.ldm.modules.attention as cattn

    t0 = time.perf_counter()
    patcher, dm = c3.load_gguf_dit(torch, a.dit)
    print(f"[check6] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s"
          f", {len(dm.blocks)} blocks", flush=True)
    edit = None
    if a.edit is not None:
        a.edit, a.vae = a.edit.resolve(), a.vae.expanduser().resolve()
        edit = edit_refs(torch, a.edit, a.vae, a.seed)
        feats = edit["arrays"]["features"]
        a.prompt = "edit"
        w, h = edit["arrays"]["vae_copy_1"].shape[1::-1]
        if (w, h) != (GATE_SIZE, GATE_SIZE):
            raise SystemExit(f"error: picture 1's VAE copy is {w}x{h}, and "
                             f"the edit runs at {GATE_SIZE}x{GATE_SIZE}")
        print(f"[check6] edit of {edit['count']} pictures, spans "
              f"{edit['arrays']['spans']}, the draws' share of each encode "
              f"{[f'{v:.3e}' for v in edit['draw_share']]}", flush=True)
    else:
        with np.load(a.features) as z:
            feats = z[f"tap_{a.prompt}"].astype(_F32)
    print(f"[check6] {feats.shape[0]} text rows of {a.prompt!r}",
          flush=True)
    sig = np.asarray(dm.pdd_sigmas, _F32)
    steps = min(a.steps, len(sig) - 1)
    timesteps = np.array([c3.timestep(torch, float(s)).item()
                          for s in sig[:steps]], _F32)

    for size in a.sizes:
        noise = noise_from_seed(a.seed, size)
        name = f"{a.name}_{'edit_' if edit else ''}{size}"
        given = (["--edit", a.edit.name, "--vae", a.vae.name] if edit
                 else ["--features", a.features.name, "--prompt", a.prompt])
        fx = fixtures.FixtureWriter(
            name, check=6, family=c3.FAMILY, script=SCRIPT,
            argv=[Path(__file__).name, "--dit", a.dit.name, "--source",
                  a.source.name, *given, "--sizes", str(size),
                  *(["--bf16", str(size)] if size in a.bf16 else [])])
        files = ((("edit_inputs", a.edit), ("vae", a.vae)) if edit
                 else (("features", a.features),))
        for role, path in (("dit", a.source), ("oracle_dit", a.dit),
                           *files):
            fx.add_file(role, path)
        fx.note(f"oracle: ComfyUI {comfy}, ComfyUI-GGUF {comfy_gguf}, "
                "UnetLoaderGGUF with dequant_dtype float32")
        fx.note("oracle_dit is dit with the config key {\"transformer\": "
                "{\"pdd_steps\": 6, \"pdd_sigmas\": [...]}} and every "
                "tensor unchanged")
        payload = None if edit is None else edit["payload"]
        lat, vel, secs = run(torch, dm, noise, feats, "fp32", name, steps,
                             payload)
        fx.add("noise", noise)
        fx.add("features", feats)
        fx.add("sigmas", sig[:steps + 1])
        fx.add("timesteps", timesteps)
        fx.add("latents", lat[-1])
        for k, (x, v) in enumerate(zip(lat, vel), 1):
            fx.add(f"latents_step{k}", x)
            fx.add(f"velocity_step{k}", v)
        row = {"steps_run": len(lat),
               "fp32_step_s": [round(s, 1) for s in secs],
               "fp32_s": round(sum(secs), 1)}
        print(f"[check6] {size}: fp32 {row['fp32_s']} s, median step "
              f"{np.median(secs):.1f} s", flush=True)
        tol = {}
        if size in a.bf16:
            lat16, _, secs16 = run(torch, dm, noise, feats, "bf16", name,
                                   steps, payload)
            fx.add("latents_bf16", lat16[-1])
            for k, x in enumerate(lat16, 1):
                fx.add(f"latents_bf16_step{k}", x)
            by_step = [fixtures.rel_l2(x, y) for x, y in zip(lat16, lat)]
            gap = fixtures.compare(lat16[-1], lat[-1])
            row.update(bf16_step_s=[round(s, 1) for s in secs16],
                       bf16_s=round(sum(secs16), 1),
                       latents_bf16_vs_fp32=gap,
                       bf16_vs_fp32_by_step=by_step)
            tol = {"latents_rel_l2": 2 * gap["rel_l2"],
                   **{f"step{k}_rel_l2": 2 * g
                      for k, g in enumerate(by_step, 1)}}
            print(f"[check6] {size}: bf16 {row['bf16_s']} s, by step "
                  f"{[f'{g:.3e}' for g in by_step]}", flush=True)
        if size != GATE_SIZE:
            tol["exact_rel_l2"] = EXACT_REL_L2
        if size in a.sampler_check:
            got, s = comfy_sampler(torch, patcher, dm, noise, feats, steps)
            row["comfyui_sampler"] = {
                **fixtures.compare(got, lat[-1]), "seconds": round(s, 1),
                "what": "comfy.sample.sample_custom, euler on the grid, "
                "guidance 1, fp32, final latents against the loop's"}
            print(f"[check6] {size}: ComfyUI's sampler against the loop, "
                  f"rel L2 {row['comfyui_sampler']['rel_l2']:.3e}",
                  flush=True)
        row["max_rss_gib"] = round(_max_rss(), 2)
        if edit is not None:
            ar = edit["arrays"]
            for k, v in ar.items():
                if k not in ("features", "spans", "prompt"):
                    fx.add(k, v)
            refs = [list(ar[f"vae_copy_{i}"].shape[1::-1])
                    for i in range(1, edit["count"] + 1)]
            fx.inputs.update(
                edit=True, refs=refs, spans=[list(v) for v in ar["spans"]],
                ref_seed=a.seed, edit_prompt=ar["prompt"],
                pictures=[list(ar[f"picture_{i}"].shape[1::-1])
                          for i in range(1, edit["count"] + 1)],
                copies="MiniMaxH3ImageConditioning's _size_short_edge and "
                       "_resize_to: the VAE copy's short edge into 1024 to "
                       "2048, the vision copy's to 1024, uint8",
                vae_encode="MiniMaxH3ImageVAE.encode in fp32 on (VAE copy "
                           "/ 255) x 2 - 1: the posterior sample with "
                           "torch.Generator().manual_seed(42) draws over "
                           "the whole latent (ref_vae_noise_i, C x h x w), "
                           "normalized (ref_latents_i)",
                ref_noise="check3_h3_image.ref_noise at the run seed",
                text_tags="minimax_token_tags: 1, 0 over each vision span")
            row["edit"] = {"copies_equal_the_npz": True,
                           "draw_share_of_encode": edit["draw_share"]}
            fx.note("the edit's features are gmlx's: the tap of the "
                    "Qwen3-VL-32B Q4_K_M encoder, tower in float32, on the "
                    "prompt with the vision copies of its pictures; both "
                    "sides take them as given")
        fx.inputs.update(
            size=size, steps=steps, guidance=1.0, seed=a.seed,
            prompt=a.prompt,
            features=("gmlx's encode of the edit prompt with its pictures"
                      if edit else f"check 2 fp32 tap of {a.prompt!r}"),
            text_rows=int(feats.shape[0]), gate_size=GATE_SIZE,
            sigmas="the PDD grid of pdd_config.json, the sampler's sigmas "
                   "and the step sigmas",
            timestep="sigma x 1000 in float32", euler=EULER,
            noise="numpy Generator(PCG64(seed)).standard_normal(shape, "
                  "dtype=float32), NCHW, shape (1, 24, size / 16, "
                  "size / 16)",
            latents="(1, 24, size / 16, size / 16) float32 in model space, "
                    "after the step",
            dequant=c3.DEQUANT, matmul_bf16=c3.MATMUL,
            attention=cattn.optimized_attention.__name__)
        fx.results.update(
            oracle={"device": "cpu", "threads": torch.get_num_threads(),
                    "bf16_features": "the fp32 tap cast to bfloat16"},
            **row)
        fx.tolerances.update(rule=RULE, **tol)
        path = fx.write(a.out)
        print(f"[check6] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
