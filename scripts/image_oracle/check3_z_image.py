#!/usr/bin/env python3
"""Check 3 oracle for Z-Image: one denoise step, block by block.

The DiT runs one step in fp32 and in bf16 on the same latents, sigma and
caption features, through ZImageTransformer2DModel loaded from the GGUF
with from_single_file. The step follows ZImagePipeline.__call__: the
model timestep is (1000 - t) / 1000 for the scheduler's t, and the
velocity is the negated, squeezed model output. Each block of the bf16
model is also run teacher-forced on the fp32 model's input to that block.

For a size in --blocks the fixture holds, whole in float32, the latents,
the caption features, the fp32 input of the first block of each chain,
the fp32 output of every block, the final layer output and the velocity.
For the other sizes it holds the latents, the caption features and the
velocity. The fp32 block outputs serve both gates, since the fp32 model's
input to each block is its own previous output. The manifest records the
bf16 against fp32 gap of every row and sets each tolerance to twice that
gap: teacher-forced per block, free running per block, and free running
for the velocity.

Caption features come from the pinned pipeline's encode_prompt with the
text encoder loaded from its GGUF by transformers, which dequantizes with
gguf-py's float32 reference. --caption random replaces them with seeded
normal features, which lack the outliers of real ones.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_z_image.py --dit z-image-turbo-Q4_K_M.gguf \\
      --encoder Qwen3-4B-Q4_K_M.gguf --config <official repo dir>

--self-test writes a tiny random model in GGUF tensor names and its
fixture instead, so the gmlx side can check its conventions on the CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

PROMPT = ("A red bicycle leaning against a weathered brick wall at sunset, "
          "a cat asleep in its basket, warm light, 35mm photograph")
STARTING = {"block_rel_l2": 5e-3, "velocity_rel_l2": 2e-2}
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each row, free running for each block of the "
        "free-running pass and for the velocity. The velocity cosine floor "
        "is 1 - 2 x the oracle's own cosine distance. Cosine distance "
        "grows with the square of the angle, so the floor bounds the angle "
        "to the fp32 velocity at sqrt(2) x the oracle's own bf16 angle")


def chain(model) -> list[tuple[str, object]]:
    """(row name, module) in execution order."""
    rows = [(f"noise_refiner.{i}", m) for i, m in
            enumerate(model.noise_refiner)]
    rows += [(f"context_refiner.{i}", m) for i, m in
             enumerate(model.context_refiner)]
    rows += [(f"layers.{i}", m) for i, m in enumerate(model.layers)]
    rows.append(("final_layer", model.all_final_layer["2-1"]))
    return rows


class Capture:
    """Inputs and outputs of the timestep embedder and every block."""

    def __init__(self, model):
        self.calls: dict[str, dict] = {}
        self._hooks = []
        mods = [("t_embedder", model.t_embedder), *chain(model)]
        for name, mod in mods:
            self._hooks.append(mod.register_forward_pre_hook(
                self._pre(name), with_kwargs=True))
            self._hooks.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(_mod, args, kwargs):
            self.calls[name] = {"args": args, "kwargs": dict(kwargs)}
        return hook

    def _post(self, name):
        def hook(_mod, _args, out):
            self.calls[name]["out"] = out
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()

    def out(self, name):
        return self.calls[name]["out"]

    def first_arg(self, name):
        return self.calls[name]["args"][0]


def schedule(torch, config_dir: Path, steps: int, image_seq_len: int):
    """Sigmas and timesteps as ZImagePipeline sets them for ``steps``."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.z_image.pipeline_z_image import (
        calculate_shift, get_default_z_image_sigmas, retrieve_timesteps)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config_dir), subfolder="scheduler", local_files_only=True)
    c = sched.config
    mu = calculate_shift(image_seq_len, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(
        sched, steps, torch.device("cpu"),
        sigmas=get_default_z_image_sigmas(steps), mu=mu)
    return sched.sigmas.clone(), timesteps.clone()


def encode_caption(torch, a) -> tuple[np.ndarray, np.ndarray, dict]:
    """Caption features (L, 2560) fp32 and the prompt's token ids."""
    if a.caption == "random":
        rng = np.random.default_rng(a.seed + 1)
        feats = rng.standard_normal((a.random_len, 2560)).astype(np.float32)
        return feats, np.zeros(0, np.int64), {
            "caption": "random", "generator": "numpy default_rng(seed + 1)"
            " standard_normal float32", "length": a.random_len}

    from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import AutoConfig, AutoTokenizer, Qwen3Model

    tok = AutoTokenizer.from_pretrained(str(a.config / "tokenizer"),
                                        local_files_only=True)
    cfg = AutoConfig.from_pretrained(str(a.config / "text_encoder"),
                                     local_files_only=True)
    t0 = time.perf_counter()
    # Without a device_map, transformers places GGUF weights on MPS when
    # torch has it. The oracle computes on the CPU.
    te = Qwen3Model.from_pretrained(
        str(a.encoder.parent), gguf_file=a.encoder.name, config=cfg,
        torch_dtype=torch.float32, device_map="cpu",
        local_files_only=True).eval()
    load_s = time.perf_counter() - t0
    where = {str(p.device) for p in te.parameters()}
    kinds = {p.dtype for p in te.parameters()}
    if where != {"cpu"} or kinds != {torch.float32}:
        raise SystemExit(f"error: text encoder on {where} in {kinds}")
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(a.config), subfolder="scheduler", local_files_only=True)
    pipe = ZImagePipeline(scheduler=sched, vae=None, text_encoder=te,
                          tokenizer=tok, transformer=None)
    t0 = time.perf_counter()
    embeds, _ = pipe.encode_prompt(a.prompt, device=torch.device("cpu"),
                                   do_classifier_free_guidance=False)
    enc_s = time.perf_counter() - t0
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    ids = np.asarray(tok(text)["input_ids"], np.int64)
    feats = oracle.to_numpy(embeds[0])
    if len(ids) != len(feats):
        raise SystemExit(f"error: {len(ids)} prompt tokens but {len(feats)} "
                         "caption rows")
    del te, pipe
    return feats, ids, {
        "caption": "prompt", "encoder": "transformers Qwen3Model from GGUF "
        "through its legacy loader, gguf-py float32 dequantization, fp32 "
        "activations on the CPU",
        "hidden_state": "hidden_states[-2], masked rows",
        "max_sequence_length": 512, "length": int(len(feats)),
        "load_seconds": round(load_s, 1), "encode_seconds": round(enc_s, 1)}


def step(torch, model, lat: np.ndarray, t_model, cap: np.ndarray, mode: str):
    """One transformer call as ZImagePipeline makes it, at batch 1."""
    dt = oracle.torch_dtype(mode)
    x = [torch.from_numpy(lat[0]).to(dt).unsqueeze(1)]
    caps = [torch.from_numpy(cap).to(dt)]
    out = model(x, t_model, caps, return_dict=False)[0]
    return -torch.stack([o.float() for o in out], dim=0).squeeze(2)


def _bf16(torch, v):
    if torch.is_tensor(v) and v.is_floating_point():
        return v.to(torch.bfloat16)
    return v


def teacher_forced(torch, model16, cap32: Capture) -> dict:
    """Output of every bf16 block fed the fp32 model's input to it."""
    outs = {}
    t_args = cap32.calls["t_embedder"]["args"]
    outs["t_embedder"] = model16.t_embedder(*t_args)
    for name, mod in chain(model16):
        call = cap32.calls[name]
        args = tuple(_bf16(torch, v) for v in call["args"])
        kwargs = {k: _bf16(torch, v) for k, v in call["kwargs"].items()}
        outs[name] = mod(*args, **kwargs)
    return outs


ROW_SOURCES = (("t_embedder", "t_embedder"),
               ("x_embed", "noise_refiner.0"),
               ("cap_embed", "context_refiner.0"))


def run_size(torch, a, models, size: int, cap: np.ndarray, ids, cap_meta,
             blocks: bool, prefix: str) -> Path:
    h = w = size // 8
    rng = np.random.default_rng(a.seed)
    c_in = models["fp32"].config.in_channels
    lat = rng.standard_normal((1, c_in, h, w)).astype(np.float32)
    sigmas, timesteps = schedule(torch, a.config, a.steps,
                                 (h // 2) * (w // 2))
    t = timesteps[a.step].expand(1)
    t_model = (1000 - t) / 1000
    sigma = sigmas[a.step]

    fx = fixtures.FixtureWriter(f"{prefix}_{size}", check=3,
                                family="z_image",
                                script="scripts/image_oracle/check3_z_image.py")
    for role, p in (("dit", a.dit), ("encoder", a.encoder)):
        if p is not None and not (role == "encoder" and a.caption == "random"):
            fx.add_file(role, p)
    fx.inputs.update(
        size=[size, size], latent_shape=list(lat.shape), noise_seed=a.seed,
        noise="numpy default_rng(seed) standard_normal float32, NCHW",
        prompt=a.prompt if a.caption == "prompt" else None, **cap_meta,
        steps=a.steps, step=a.step, sigmas=[float(s) for s in sigmas],
        timesteps=[float(s) for s in timesteps], sigma=float(sigma),
        t_model=float(t_model[0]), config_revision=a.config_revision)
    fx.results["oracle"] = {
        "device": "cpu", "threads": torch.get_num_threads(),
        "dequant": oracle.dequant_mode(), "matmul": {}}
    fx.add("latents", lat)
    fx.add("cap_feats", cap)
    fx.add("prompt_ids", ids)
    fx.add("sigma", oracle.to_numpy(sigma))
    fx.add("t_model", oracle.to_numpy(t_model))

    times = {}
    v = {}
    caps = {}
    for mode in oracle.MODES:
        oracle.emulate_bf16_matmul(mode == "bf16")
        fx.results["oracle"]["matmul"][mode] = oracle.matmul_mode()
        c = Capture(models[mode]) if blocks else None
        t0 = time.perf_counter()
        v[mode] = oracle.to_numpy(step(torch, models[mode], lat, t_model,
                                       cap, mode))
        times[f"{mode}_step_s"] = round(time.perf_counter() - t0, 1)
        if c is not None:
            c.remove()
            caps[mode] = c
    fx.add("velocity", v["fp32"])
    fx.add("velocity_bf16", v["bf16"])
    gap_v = fixtures.compare(v["bf16"], v["fp32"])
    fx.results["velocity_bf16_vs_fp32"] = gap_v
    fx.tolerances.update(rule=RULE, starting=STARTING,
                         velocity_rel_l2=2 * gap_v["rel_l2"],
                         velocity_cosine_min=1 - 2 * (1 - gap_v["cosine"]))

    if blocks:
        c32, c16 = caps["fp32"], caps["bf16"]
        # Each block's input is the previous block's output.
        names = [n for n, _ in chain(models["fp32"])]
        for prev, nxt in zip(names, names[1:]):
            if nxt in ("context_refiner.0", "layers.0"):
                continue
            assert c32.first_arg(nxt) is c32.out(prev), (prev, nxt)
        fx.add("t_embedder.in",
               oracle.to_numpy(c32.first_arg("t_embedder")))
        fx.add("t_embedder.out", oracle.to_numpy(c32.out("t_embedder")))
        for n in ("noise_refiner.0", "context_refiner.0", "layers.0"):
            fx.add(f"{n}.in", oracle.to_numpy(c32.first_arg(n)))
        for n in names:
            fx.add(f"{n}.out", oracle.to_numpy(c32.out(n)))

        oracle.emulate_bf16_matmul(True)
        t0 = time.perf_counter()
        tf = teacher_forced(torch, models["bf16"], c32)
        times["bf16_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
        oracle.emulate_bf16_matmul(False)
        rows, free = {}, {}
        ref = {"t_embedder": c32.out("t_embedder")}
        got = {"t_embedder": tf["t_embedder"]}
        for row, src in ROW_SOURCES[1:]:
            ref[row] = c32.first_arg(src)
            got[row] = c16.first_arg(src)
        for n in names:
            ref[n] = c32.out(n)
            got[n] = tf[n]
            free[n] = fixtures.compare(oracle.to_numpy(c16.out(n)),
                                       oracle.to_numpy(c32.out(n)))
        for row in ref:
            rows[row] = fixtures.compare(oracle.to_numpy(got[row]),
                                         oracle.to_numpy(ref[row]))
        fx.results["rows_bf16_vs_fp32_teacher_forced"] = rows
        fx.results["blocks_bf16_vs_fp32_free"] = free
        fx.tolerances["rows_rel_l2"] = {r: 2 * g["rel_l2"]
                                        for r, g in rows.items()}
        fx.tolerances["free_rel_l2"] = {n: 2 * g["rel_l2"]
                                        for n, g in free.items()}
        fx.inputs["rows"] = list(rows)
        fx.inputs["free_rows"] = list(free)
    else:
        oracle.emulate_bf16_matmul(False)
    fx.results["seconds"] = times
    path = fx.write()
    print(f"[check3] {size}: velocity bf16 vs fp32 rel L2 "
          f"{gap_v['rel_l2']:.3e} cos {gap_v['cosine']:.6f}, {times}")
    return path


# Self-test: a tiny random model, written in GGUF tensor names.

TINY = {"all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 4,
        "dim": 64, "n_layers": 2, "n_refiner_layers": 1, "n_heads": 2,
        "n_kv_heads": 2, "norm_eps": 1e-5, "qk_norm": True,
        "cap_feat_dim": 16, "rope_theta": 256.0, "t_scale": 1000.0,
        "axes_dims": [8, 12, 12], "axes_lens": [128, 32, 32]}
_RENAME = (("all_x_embedder.2-1.", "x_embedder."),
           ("all_final_layer.2-1.", "final_layer."),
           (".attention.to_out.0.", ".attention.out."),
           (".attention.norm_q.", ".attention.q_norm."),
           (".attention.norm_k.", ".attention.k_norm."))


def gguf_names(state: dict) -> dict:
    """diffusers parameter names and values in GGUF names, qkv fused."""
    out = {}
    for k, v in state.items():
        if ".attention.to_k." in k or ".attention.to_v." in k:
            continue
        if ".attention.to_q." in k:
            qkv = [state[k.replace("to_q", p)] for p in ("to_q", "to_k",
                                                          "to_v")]
            out[k.replace("to_q", "qkv")] = np.concatenate(qkv, 0)
            continue
        for old, new in _RENAME:
            k = k.replace(old, new)
        out[k] = v
    return out


def self_test(torch, a) -> int:
    from diffusers import ZImageTransformer2DModel

    torch.manual_seed(a.seed)
    m32 = ZImageTransformer2DModel(**TINY).eval()
    gen = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        for name, p in m32.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2
                    + (1.0 if name.endswith("norm1.weight") else 0.0))
    m16 = ZImageTransformer2DModel(**TINY).eval()
    m16.load_state_dict(m32.state_dict())
    m16 = m16.to(torch.bfloat16)
    state = {k: oracle.to_numpy(v) for k, v in m32.state_dict().items()}
    weights = fixtures.FixtureWriter(
        "selftest_z_image_tiny_weights", check=3, family="z_image",
        script="scripts/image_oracle/check3_z_image.py")
    weights.inputs.update(config=TINY, seed=a.seed,
                          init="randn * 0.2, plus 1 on norm1 weights")
    for k, v in sorted(gguf_names(state).items()):
        weights.add(k, v)
    weights.write()
    a.config_revision = None
    a.dit = a.encoder = None
    rng = np.random.default_rng(a.seed + 1)
    cap = rng.standard_normal((5, TINY["cap_feat_dim"])).astype(np.float32)
    models = {"fp32": m32, "bf16": m16}
    meta = {"caption": "random", "length": 5}
    orig_schedule = globals()["schedule"]

    def tiny_schedule(torch, _config, steps, _seq):
        s = torch.linspace(1.0, 1 / steps, steps)
        return s, s * 1000

    globals()["schedule"] = tiny_schedule
    try:
        run_size(torch, a, models, 48, cap, np.zeros(0, np.int64), meta,
                 True, "selftest_z_image_tiny")
    finally:
        globals()["schedule"] = orig_schedule
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", type=Path, help="Z-Image DiT GGUF")
    ap.add_argument("--encoder", type=Path, help="Qwen3-4B text GGUF")
    ap.add_argument("--config", type=Path,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo (configs, "
                         "tokenizer and scheduler)")
    ap.add_argument("--config-revision",
                    default="f332072aa78be7aecdf3ee76d5c247082da564a6",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--caption", choices=("prompt", "random"),
                    default="prompt")
    ap.add_argument("--random-len", type=int, default=64)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--blocks", type=int, nargs="*", default=[512],
                    help="sizes whose fixture keeps every block tensor")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--step", type=int, default=3,
                    help="index of the step whose sigma is used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check3_z_image_turbo",
                    help="fixture name prefix; the size is appended")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny random model fixture and stop")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.self_test:
        return self_test(torch, a)
    if not (a.dit and a.config and (a.encoder or a.caption == "random")):
        ap.error("--dit, --config and --encoder are required")
    if any(s % 16 for s in a.sizes):
        ap.error("sizes must be multiples of 16")
    t0 = time.perf_counter()
    cap, ids, cap_meta = encode_caption(torch, a)
    print(f"[check3] caption {cap_meta['caption']}: {cap.shape[0]} rows in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    models = {m: oracle.load_dit(a.dit, a.config, m, dequant=a.dequant)
              for m in oracle.MODES}
    for m, model in models.items():
        where = {str(p.device) for p in model.parameters()}
        if where != {"cpu"}:
            raise SystemExit(f"error: {m} DiT on {where}")
    for size in a.sizes:
        path = run_size(torch, a, models, size, cap, ids, cap_meta,
                        size in a.blocks, a.name)
        print(f"[check3] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

