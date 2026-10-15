#!/usr/bin/env python3
"""Check 3 oracle for FLUX.3 klein: one denoise step of BFL's reference
DiT, block by block, on the weights of a DiT GGUF.

The reference is BFL's flux-3-image-klein repository, read from
--reference (its src directory), at the commit the manifest records.
Its DiT class has no GGUF loader, and the GGUF tensor names are its state
dict keys, so the script builds the class on the meta device, puts each
quantized and BF16 matrix in a diffusers GGUFLinear over the file's own
bytes, and sets the F32 and F16 tensors in the mode's dtype. Each
GGUFLinear dequantizes its weight on every call in float32 arithmetic
(oracle ``dequant="exact"``), which is ggml's.

One step is one call as the reference pipeline makes it for one sample:
pipeline.pack of the text, the reference images, if any, and the image
latents, the image's time filled in the bf16 timesteps of the pack, and
for the distilled flow map the time the step goes to, the next entry of
sampling.timesteps(16, 6.93). The reference runs under bf16 autocast on
CUDA. Here the fp32 model runs every op in float32 and the bf16 model
every op in bfloat16, products as one float32 GEMM over the bf16 operands
with one rounding (oracle.emulate_bf16_matmul). The time embedding is
the reference's own on its bf16 timesteps, which it returns in bf16, in
both modes; a pre-hook casts it to the fp32 model's dtype. The token data
are cast to the mode's dtype, as autocast casts them at the first
product.

The latents are seeded normal rounded to bf16, as the pipeline's noise
is, at the first step, time 1 (--sigma). The text features are seeded
normal with --feature-std, rounded to bf16, in --text-len tokens, at
position ids (0, 0, 0, i). With --refs N, N reference images of
--ref-size pixels each give seeded normal tokens, bf16, at the
reference's ids: (0, row / 2, column / 2, -1 - index).

For a size in --rows the fixture holds, in float32: the inputs, the time
vector of each mode, the early, single and final modulations, the token
embeddings, the output of every block, the input of the first single
block and of the final layer, the final layer and the velocity. The bf16
model runs the same call, and each of its blocks and its final layer
also run teacher-forced on the fp32 model's input to them, with the fp32
model's modulations. The manifest sets each row's tolerance to twice its
bf16 against fp32 gap: teacher-forced for the blocks and the final
layer, free running for the rest and for the velocity. For the other
sizes the fixture holds the inputs and the velocities.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_flux3_klein.py --reference ~/src/flux-3-image-klein/src \\
      --dit flux3_image_klein_sd-Q4_K_M.gguf --sizes 512 --refs 1

--self-test writes a tiny model of the reference class, its sizes set
small, on random weights, so the gmlx side can check its conventions on
the CPU.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

SCRIPT = "scripts/image_oracle/check3_flux3_klein.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each block and the final layer, free running "
        "for the time vectors, modulations and embeddings and for the "
        "velocity")
MODES = ("image", "image_cond", "text")
MOD_KEYS = {"image": "image", "image_cond": "image_cond", "text": "txt"}
STEPS, ALPHA = 16, 6.93

# The tiny model of --self-test: the reference's module constants scaled
# down, with two reference images of different grids.
TINY = {"HIDDEN": 64, "HEADS": 4, "DEPTH": 2, "DEPTH_SINGLE": 2,
        "AXES_DIM": (4, 4, 4, 4), "MLP_RATIO": 3.0,
        "IN_CHANNELS": {"image": 8, "image_cond": 8, "text": 24}}
TINY_CASE = {"latent": (4, 6), "refs": ((2, 3), (2, 2)), "text": 10,
             "sigma": 0.6875, "target": 0.5}


def reference(path: Path):
    """The reference's dit, pipeline and sampling modules, and its git
    commit."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    from flux3 import dit, pipeline, sampling

    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return dit, pipeline, sampling, commit


def gguf_state(torch, path: Path) -> tuple[dict, dict, bool]:
    """The file's tensors by name, quantized and BF16 ones as
    GGUFParameter, with the codec census and whether it is a flow map."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    reader = gguf.GGUFReader(str(path))
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    out, census = {}, {}
    for t in reader.tensors:
        name = re.sub(r"^(model\.)?diffusion_model\.", "", t.name)
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
        data = torch.from_numpy(np.array(t.data))
        if t.tensor_type in floats:
            out[name] = data
        else:
            out[name] = GGUFParameter(data, quant_type=t.tensor_type)
    return out, census, "ref_time_in.in_layer.weight" in out


def build(torch, dit, state: dict, flow_map: bool, mode: str):
    """The reference DiT on ``state``, quantized leaves as GGUFLinear."""
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 _replace_with_gguf_linear)

    dtype = oracle.torch_dtype(mode)
    with torch.device("meta"):
        model = dit.DiT(flow_map)
    _replace_with_gguf_linear(model, dtype, state)
    names = {n for n, _ in model.named_parameters()}
    if names != set(state):
        raise SystemExit(f"names: missing {sorted(names - set(state))[:4]}, "
                         f"extra {sorted(set(state) - names)[:4]}")
    for name, v in state.items():
        owner, _, leaf = name.rpartition(".")
        mod = model.get_submodule(owner)
        if not isinstance(v, GGUFParameter):
            v = torch.nn.Parameter(v.to(dtype), requires_grad=False)
        mod._parameters[leaf] = v
    assert not any(p.is_meta for p in model.parameters())
    assert not list(model.buffers())
    for emb in ("time_in", "ref_time_in"):
        if hasattr(model, emb):
            # The embedding is bf16, as the reference returns it; the fp32
            # model takes it in its own dtype.
            getattr(model, emb).register_forward_pre_hook(
                lambda _m, args, dt=dtype: tuple(a.to(dt) for a in args))
    return model.eval()


def bf16(torch, a: np.ndarray) -> np.ndarray:
    return oracle.to_numpy(torch.from_numpy(np.asarray(a, np.float32)).to(
        torch.bfloat16))


class Capture:
    """Hooks that keep the rows of one call by gmlx module path."""

    def __init__(self, model, modes: tuple):
        self.rows: dict = {}
        self.inputs: dict = {}
        self.handles = []
        reg = self._reg
        for m in modes:
            k = MOD_KEYS[m]
            reg(f"early_stream_modulations.{k}",
                model.early_stream_modulations[k], first=True, vec=m)
            reg(f"single_stream_modulations.{k}",
                model.single_stream_modulations[k], first=True)
            if m == "text":
                reg("txt_in", model.txt_in)
                blocks = [(f"txt_mode_blocks.{i}", b)
                          for i, b in enumerate(model.txt_mode_blocks)]
            else:
                reg(f"emb_in.{m}", model.emb_in[m])
                blocks = [(f"content_mode_blocks.{m}.{i}", b) for i, b in
                          enumerate(model.content_mode_blocks[m])]
            for name, b in blocks:
                reg(name, b)
        for i, b in enumerate(model.single_blocks):
            reg(f"single_blocks.{i}", b)
        fl = model.final_layer["image"]
        reg("final_layer.image.adaLN_modulation", fl.adaLN_modulation,
            first=True)
        reg("final_layer.image", fl)

    def _reg(self, name, mod, first=False, vec=None):
        def keep(_m, args, out):
            if isinstance(out, tuple):
                out = __import__("torch").cat(out, dim=-1)
            self.rows[name] = out[:1] if first else out
            if vec is not None:
                self.rows[f"vec.{vec}"] = args[0][:1]

        def feed(_m, args):
            self.inputs[name] = args
        self.handles.append(mod.register_forward_hook(keep))
        self.handles.append(mod.register_forward_pre_hook(feed))

    def remove(self):
        for h in self.handles:
            h.remove()


def case_inputs(torch, pipeline, lat: np.ndarray, feats: np.ndarray,
                refs: list, mode: str):
    """The pack of one sample and its image ids, in the mode's dtype."""
    from flux3.autoencoder import latent_ids

    dtype = oracle.torch_dtype(mode)
    _, c, h, w = lat.shape
    img = torch.from_numpy(lat[0].reshape(c, h * w).T.copy())
    image = (img, pipeline.scaled_ids(latent_ids(h, w)))
    n = feats.shape[0]
    pos = torch.arange(n)
    text_ids = torch.stack([torch.zeros_like(pos)] * 3 + [pos], dim=-1)
    text = (torch.from_numpy(feats).to(torch.bfloat16), text_ids)
    cond = None
    if refs:
        toks, ids = [], []
        for index, r in enumerate(refs):
            _, rc, rh, rw = r.shape
            toks.append(torch.from_numpy(r[0].reshape(rc, rh * rw).T.copy())
                        .to(torch.bfloat16))
            i = pipeline.scaled_ids(latent_ids(rh, rw))
            i[:, 3] = -(1 + index)
            ids.append(i)
        cond = (torch.cat(toks), torch.cat(ids))
    packed = pipeline.pack([[text, cond, image]])
    packed.data = {m: v.to(dtype) for m, v in packed.data.items()}
    return packed


def call(torch, pipeline, model, lat, feats, refs, sigma, target, mode,
         rows: bool):
    packed = case_inputs(torch, pipeline, lat, feats, refs, mode)
    packed.timesteps["image"].fill_(sigma)
    modes = tuple(m for m in MODES if m != "image_cond" or refs)
    cap = Capture(model, modes) if rows else None
    ref = None
    if target is not None:
        ref = torch.tensor([target], dtype=torch.float32)
    try:
        out = model(packed, timesteps_ref=ref)
    finally:
        if cap is not None:
            cap.remove()
    _, c, h, w = lat.shape
    vel = oracle.to_numpy(out).T.reshape(1, c, h, w)
    return vel, cap, packed


def teacher_forced(torch, model16, cap32: Capture) -> dict:
    """The bf16 output of each block and of the final layer fed the fp32
    model's inputs and modulations. The rotation tables stay float32, as
    the reference forms them."""
    def cast(a):
        if isinstance(a, tuple):
            return tuple(cast(x) for x in a)
        if isinstance(a, torch.Tensor) and a.is_floating_point() \
                and a.dim() != 6:
            return a.to(torch.bfloat16)
        return a

    out = {}
    for name, args in cap32.inputs.items():
        if name.startswith(("content_mode_blocks.", "txt_mode_blocks.",
                            "single_blocks.")) or name == "final_layer.image":
            mod = model16.get_submodule(name)
            out[name] = oracle.to_numpy(mod(*(cast(a) for a in args)))
    return out


def rows_of(cap: Capture) -> dict:
    return {k: oracle.to_numpy(v) for k, v in cap.rows.items()}


def run_case(torch, pipeline, models: dict, lat, feats, refs, sigma,
             target, rows: bool, name: str, inputs: dict, files: dict,
             notes: list, *, script: str = SCRIPT) -> dict:
    """One step in fp32 and bf16 written as fixture ``name``. Returns the
    velocity of each mode."""
    fx = fixtures.FixtureWriter(name, check=3, family="flux3_klein",
                                script=script)
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in notes:
        fx.note(n)
    fx.add("latents", lat)
    fx.add("features", feats)
    for i, r in enumerate(refs):
        fx.add(f"refs.{i}", r)
    fx.add("sigma", np.float32(sigma))
    if target is not None:
        fx.add("target", np.float32(target))
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        vel[mode], caps[mode], packed = call(
            torch, pipeline, models[mode], lat, feats, refs, sigma, target,
            mode, rows)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    oracle.emulate_bf16_matmul(False)
    for m in MODES:
        if packed.data[m].shape[0]:
            fx.add(f"ids.{m}", oracle.to_numpy(packed.ids[m]))
    fx.add("velocity", vel["fp32"])
    fx.add("velocity_bf16", vel["bf16"])
    gap = fixtures.compare(vel["bf16"], vel["fp32"])
    fx.results["velocity_bf16_vs_fp32"] = gap
    fx.tolerances["velocity"] = {"rel_l2": 2 * gap["rel_l2"],
                                 "cosine_min": 1 - 4 * (1 - gap["cosine"])}
    if rows:
        r32, r16 = rows_of(caps["fp32"]), rows_of(caps["bf16"])
        oracle.emulate_bf16_matmul(True)
        t0 = time.perf_counter()
        tf = teacher_forced(torch, models["bf16"], caps["fp32"])
        times["bf16_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
        oracle.emulate_bf16_matmul(False)
        ins = caps["fp32"].inputs
        fx.add("in.single_blocks.0", oracle.to_numpy(
            ins["single_blocks.0"][0]))
        fx.add("in.final_layer.image", oracle.to_numpy(
            ins["final_layer.image"][0]))
        gaps = {}
        for k, a in r32.items():
            fx.add(k, a)
            forced = k in tf
            gaps[k] = fixtures.compare(tf[k] if forced else r16[k], a)
            gaps[k]["teacher_forced"] = forced
            fx.tolerances[k] = {"rel_l2": 2 * gaps[k]["rel_l2"],
                                "teacher_forced": forced}
        fx.results["rows_bf16_vs_fp32"] = gaps
        fx.results["free_bf16_vs_fp32"] = {
            k: fixtures.compare(r16[k], r32[k])["rel_l2"] for k in tf}
    fx.results["times"] = times
    fx.tolerances["rule"] = RULE
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 "
          f"{gap['rel_l2']:.3e}, {times}, wrote {path}", flush=True)
    return vel


def seeded(torch, rng, shape, std=1.0) -> np.ndarray:
    return bf16(torch, rng.standard_normal(shape) * std)


# The self-test.

def tiny_state(torch, dit, flow_map: bool, seed: int) -> dict:
    """Random tiny weights by name, float32 numpy in torch order: norm
    scales near one, matrices at 1 / sqrt(fan in)."""
    with torch.device("meta"):
        model = dit.DiT(flow_map)
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if name.endswith(".scale"):
            v = 1 + rng.standard_normal(shape) * 0.2
        else:
            v = rng.standard_normal(shape) / np.sqrt(shape[-1])
        out[name] = v.astype(np.float32)
    return out


def tiny_config(dit) -> dict:
    """The tiny model in the keys of the official transformer config."""
    head = TINY["HIDDEN"] // TINY["HEADS"]
    return {"attention_head_dim": head,
            "axes_dims_rope": list(TINY["AXES_DIM"]),
            "cond_in_channels": TINY["IN_CHANNELS"]["image_cond"],
            "eps": 1e-06, "flow_map": True,
            "in_channels": TINY["IN_CHANNELS"]["image"],
            "joint_attention_dim": TINY["IN_CHANNELS"]["text"],
            "mlp_ratio": TINY["MLP_RATIO"],
            "num_attention_heads": TINY["HEADS"],
            "num_layers": TINY["DEPTH"],
            "num_single_layers": TINY["DEPTH_SINGLE"],
            "rope_theta": dit.THETA, "timestep_channels": 256,
            "dit": {"image_rope_scale": 0.5}}


def self_test(torch, dit, pipeline, commit: str, a) -> int:
    for k, v in TINY.items():
        if isinstance(v, dict):
            # pipeline holds the same dict.
            getattr(dit, k).clear()
            getattr(dit, k).update(v)
        else:
            setattr(dit, k, v)
    state = tiny_state(torch, dit, True, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    c = TINY["IN_CHANNELS"]["image"]
    h, w = TINY_CASE["latent"]
    lat = seeded(torch, rng, (1, c, h, w))
    feats = seeded(torch, rng, (TINY_CASE["text"],
                                TINY["IN_CHANNELS"]["text"]), 2.0)
    refs = [seeded(torch, rng, (1, c, rh, rw))
            for rh, rw in TINY_CASE["refs"]]
    cfg = tiny_config(dit)
    fx_inputs = {"config": cfg, "case": TINY_CASE, "seed": a.seed,
                 "reference_commit": commit}
    for flow_map, cond, name in ((True, True, "selftest_flux3_klein_tiny"),
                                 (False, False,
                                  "selftest_flux3_klein_tiny_base")):
        st = {k: torch.from_numpy(v) for k, v in state.items()
              if flow_map or not k.startswith("ref_time_in.")}
        models = {m: plain_build(torch, dit, st, flow_map, m)
                  for m in ("fp32", "bf16")}
        run_case(torch, pipeline, models, lat, feats, refs if cond else [],
                 TINY_CASE["sigma"], TINY_CASE["target"] if flow_map
                 else None, True, name,
                 dict(fx_inputs, flow_map=flow_map, refs=len(refs) if cond
                      else 0), {},
                 ["the weights are in selftest_flux3_klein_tiny_weights by "
                  "GGUF name in torch order"])
    wx = fixtures.FixtureWriter("selftest_flux3_klein_tiny_weights", check=3,
                                family="flux3_klein", script=SCRIPT)
    wx.inputs.update(fx_inputs)
    for g, v in state.items():
        wx.add(g, v)
    wx.write()
    return 0


def plain_build(torch, dit, state: dict, flow_map: bool, mode: str):
    """The reference DiT on float tensors in the mode's dtype."""
    dtype = oracle.torch_dtype(mode)
    model = dit.DiT(flow_map)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()},
                          strict=True)
    model = model.to(dtype)
    for emb in ("time_in", "ref_time_in"):
        if hasattr(model, emb):
            getattr(model, emb).register_forward_pre_hook(
                lambda _m, args, dt=dtype: tuple(x.to(dt) for x in args))
    return model.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--dit", type=Path, help="FLUX.3 klein DiT GGUF")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--rows", type=int, nargs="*", default=[512],
                    help="sizes whose fixture holds every row")
    ap.add_argument("--refs", type=int, default=0,
                    help="reference images, each of --ref-size")
    ap.add_argument("--ref-size", type=int, default=0,
                    help="pixels of a reference side, 0 for the size")
    ap.add_argument("--text-len", type=int, default=80)
    ap.add_argument("--feature-std", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=0,
                    help="step of the 16-step schedule whose next entry "
                         "is the flow map's target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check3_flux3_klein")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    dit, pipeline, sampling, commit = reference(a.reference.expanduser())
    if a.self_test:
        return self_test(torch, dit, pipeline, commit, a)
    if a.dit is None:
        ap.error("--dit is required")
    t0 = time.perf_counter()
    state, census, flow_map = gguf_state(torch, a.dit)
    models = {m: build(torch, dit, state, flow_map, m)
              for m in ("fp32", "bf16")}
    print(f"[check3] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}, flow map "
          f"{flow_map}", flush=True)
    target = None
    if flow_map:
        schedule = sampling.timesteps(STEPS, ALPHA)
        target = float(schedule[a.step + 1])
    rng = np.random.default_rng(a.seed)
    feats = seeded(torch, rng, (a.text_len, dit.IN_CHANNELS["text"]),
                   a.feature_std)
    c = dit.IN_CHANNELS["image"]
    for size in a.sizes:
        n = size // 16
        srng = np.random.default_rng(a.seed + size)
        lat = seeded(torch, srng, (1, c, n, n))
        rn = (a.ref_size or size) // 16
        refs = [seeded(torch, srng, (1, c, rn, rn)) for _ in range(a.refs)]
        inputs = {"size": size, "sigma": a.sigma, "target": target,
                  "step": a.step, "schedule": f"timesteps({STEPS}, "
                  f"{ALPHA})", "flow_map": flow_map, "seed": a.seed + size,
                  "latents": "seeded normal, rounded to bf16",
                  "features": f"seeded normal, std {a.feature_std}, "
                              "rounded to bf16",
                  "text_rows": int(feats.shape[0]), "refs": a.refs,
                  "ref_size": rn * 16 if a.refs else 0,
                  "dequant": oracle.dequant_mode(),
                  "matmul_bf16": "bf16 operands, float32 GEMM, one bf16 "
                                 "rounding",
                  "timestep": "bf16 as the reference pipeline packs it, in "
                              "both modes",
                  "reference_commit": commit, "gguf_codecs": census}
        tag = f"_r{a.refs}" if a.refs else ""
        run_case(torch, pipeline, models, lat, feats, refs, a.sigma, target,
                 size in a.rows, f"{a.name}{tag}_{size}", inputs,
                 {"dit": a.dit}, [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
