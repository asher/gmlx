#!/usr/bin/env python3
"""Check 3 oracle for MiniMax-H3-Image: one denoise step of ComfyUI's
MiniMaxH3ImageModel, block by block, on the weights of the DiT GGUF.

The H3 video method (scripts/video_oracle/check3_h3.py) on the image
model. The script starts the ComfyUI checkout of --comfy-dir on the CPU
and loads the GGUF through its ComfyUI-GGUF node (UnetLoaderGGUF) with
dequant_dtype float32. ComfyUI reads the PDD head count and grid from the
file's `config` key, so --dit is a copy of the release file with that key
(results/image/h3-image/b1/gguf-config-copy.py in the lab), which holds
every tensor of --source unchanged; the manifest names --source as the
DiT, the file gmlx loads. The fp32 pass takes float32 text features, the
bf16 pass bfloat16 ones and runs each linear as a float32 GEMM over
bfloat16 operands. Attention runs through torch's
scaled_dot_product_attention.

One step is one forward as ComfyUI's sampler makes it for a text to image
prompt: latents (1, 24, 1, H, W), the timestep sigma x 1000 in float32,
the text features (1, L, 5120) and the sampler's sigmas, here the PDD
grid, which picks the step's head. An edit step (--refs) adds the
minimax_payload that MiniMaxH3ImageConditioning and model_base give it:
each reference's latents (seeded normal, one frame) as minimax_refs and
cond_video_latents, the run seed, and the prompt's token tags with 0 over
each vision span (--spans). Its fixture also holds each reference's
latents (1, 24, h, w), the noise ComfyUI's forward mixes into its rows (a
CPU torch generator restarted at the seed, drawn in the rows' shape), the
tags and the seed. Each fixture holds, in float32 under
gmlx's module paths: the text projection, each refiner block and the
refiner's norm, the patch projection, the curve rows, the input of the
first block and the output of every block, the head's input and
modulation, the step's head rows (``final_layer``) and the velocity, with
the layout's positions, the RoPE angles, the segment table and the
distinct times.

The bf16 pass runs the same call, and each refiner block, the refiner's
norm, each block and the head also run teacher-forced on the fp32 pass's
input to them. The manifest sets each row's tolerance to twice its bf16
against fp32 gap, as check3_h3.py does.

  python scripts/image_oracle/run.py --python <ComfyUI venv python> \\
      --threads 16 check3_h3_image.py --comfy-dir <ComfyUI checkout> \\
      --dit <copy with the config key> --source <release GGUF> \\
      --cases 512x512:1.0 512x512:0.75 [--refs 512x512 384x640 \\
      --spans 3:20 24:41 --text-len 48]

--self-test writes a tiny model of the class on random weights, with the
weights by name, and its step at sigma 1.0 and 0.75, each as text to image
and as an edit, and at sigma 1.0 the edit's prompt without its references
(the node without its VAE), so the gmlx side checks the whole forward on
the CPU. Its fixtures are written compressed, to be committed beside their
manifests.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

FAMILY = "h3_image"
SCRIPT = "scripts/image_oracle/check3_h3_image.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each refiner block, the refiner's norm, each "
        "block and the head, free running for the other rows and for the "
        "velocity, and free running for each teacher-forced row as "
        "'free <row>'; rows in float32 in both modes take a float32 "
        "tolerance")
DEQUANT = ("ComfyUI-GGUF dequantize_tensor with dequant_dtype float32: "
           "K-quant block math in float32, on every call")
MATMUL = "bf16 operands, float32 GEMM, one bf16 rounding"
FP32_ROWS = 1e-5
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
              0.5327739119529724, 0.23200887441635132, 0.0)

# The tiny model of --self-test in ComfyUI's constructor arguments, and
# the same model in the config names gmlx reads.
TINY = {"hidden_size": 32, "num_layers": 2, "token_refiner_num_layers": 1,
        "num_attention_heads": 2, "attention_head_dim": 64,
        "ffn_hidden_size": 48, "latents_dim": 24, "patch_size": [1, 2, 2],
        "text_dim": 40, "time_embed_dim": 8, "rope_inv_freq_len": 8,
        "adaln_curve_grid": 1025, "norm_eps": 1e-5, "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5, "pdd_steps": 6,
        "pdd_sigmas": list(PDD_SIGMAS)}
_GMLX = {"adaln_curve_grid": "adaln_grid", "time_embed_dim": "adaln_basis"}
TINY_CASE = {"shape": "96x64", "text": 8, "sigmas": (1.0, 0.75)}
# The tiny edit: two references of other sizes than the target and a
# vision span in the prompt.
TINY_EDIT = {"refs": ((64, 64), (96, 32)), "spans": ((1, 5),),
             "ref_seed": 7}
# The tiny edit's prompt without its references, as
# MiniMaxH3ImageConditioning gives it without its VAE: the pictures
# condition the text encoder alone.
TINY_VISION = {"refs": (), "spans": TINY_EDIT["spans"],
               "ref_seed": TINY_EDIT["ref_seed"]}


def comfy_oracle():
    """scripts/video_oracle/oracle.py: ComfyUI and ComfyUI-GGUF on the
    CPU."""
    path = Path(__file__).resolve().parent.parent / "video_oracle" / \
        "oracle.py"
    spec = importlib.util.spec_from_file_location("_h3_comfy_oracle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_case(s: str) -> tuple[int, int, float]:
    try:
        size, sigma = s.split(":")
        w, h = (int(v) for v in size.lower().split("x"))
        sigma = float(sigma)
    except ValueError:
        raise SystemExit(f"error: case {s!r} is not WxH:SIGMA") from None
    if w % 32 or h % 32:
        raise SystemExit(f"error: {w}x{h} is not a multiple of 32")
    return w, h, sigma


def latents(w: int, h: int, seed: int) -> np.ndarray:
    """Seeded normal latents (1, 24, H / 16, W / 16), one frame."""
    rng = np.random.default_rng([seed, w, h])
    return rng.standard_normal((1, 24, h // 16, w // 16)).astype(np.float32)


def timestep(torch, sigma: float):
    return torch.tensor([np.float32(sigma) * np.float32(1000)],
                        dtype=torch.float32)


def _clone(a):
    return a.detach().clone() if hasattr(a, "detach") else a


class Capture:
    """Hooks that keep copies of rows of one forward by gmlx module
    path."""

    def __init__(self, dm):
        self.rows: dict = {}
        self.inputs: dict = {}
        self.handles = []
        fl, tr = dm.final_layer, dm.token_refiner
        keep = [("condition_proj", dm.condition_proj),
                ("video_patch_proj", dm.video_patch_proj),
                ("token_refiner.final_norm", tr.final_norm),
                ("final_layer.adaln_proj.linear", fl.adaln_proj.linear),
                ("final_layer", fl)]
        keep += [(f"token_refiner.blocks.{i}", b)
                 for i, b in enumerate(tr.blocks)]
        keep += [(f"blocks.{i}", b) for i, b in enumerate(dm.blocks)]
        for name, mod in keep:
            self.handles.append(mod.register_forward_hook(self._keep(name)))
        for name, mod in (("blocks.0", dm.blocks[0]), ("final_layer", fl)):
            self.handles.append(
                mod.register_forward_pre_hook(self._feed(name)))

    def _keep(self, name):
        def hook(_mod, _args, out):
            self.rows[name] = _clone(out[0] if name == "final_layer"
                                     else out)
        return hook

    def _feed(self, name):
        def hook(_mod, args):
            self.inputs[name] = tuple(_clone(a) for a in args)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def ref_latents(sizes, seed: int) -> list:
    """Seeded normal latents (1, 24, h / 16, w / 16) of each reference
    size (w, h), one frame each."""
    out = []
    for i, (w, h) in enumerate(sizes):
        rng = np.random.default_rng([seed, 1000 + i, w, h])
        out.append(rng.standard_normal((1, 24, h // 16, w // 16)).astype(
            np.float32))
    return out


def ref_noise(torch, refs, seed: int) -> list:
    """The noise ComfyUI's forward mixes into each reference's rows
    (comfy model.py:567-580): a CPU generator restarted at the seed for
    each one, drawn in the shape of its patchified rows."""
    out = []
    for z in refs:
        _, c, h, w = z.shape
        gen = torch.Generator("cpu").manual_seed(seed)
        out.append(torch.randn((h // 2 * (w // 2), c * 4), generator=gen,
                               dtype=torch.float32).numpy())
    return out


def edit_payload(torch, refs, text_len: int, spans, seed: int) -> dict:
    """The minimax_payload of an edit step: the references as
    MiniMaxH3ImageConditioning gives them, the run seed and the prompt's
    token tags, 0 over each vision span. Without references it holds no
    refs, as the node gives it without its VAE."""
    tags = torch.ones(text_len, dtype=torch.long)
    for a, b in spans:
        tags[a:b] = 0
    payload = {"seed": seed, "text_token_tags": tags}
    if refs:
        blocks = [{"kind": "image", "latent_h": z.shape[2],
                   "latent_w": z.shape[3],
                   "latent": torch.from_numpy(z[:, :, None])} for z in refs]
        payload.update(refs=blocks, cond_video_latents=[b["latent"]
                                                        for b in blocks])
    return payload


def call(torch, dm, lat, feats, sigma, mode, grid, payload=None):
    """One forward; the velocity (1, 24, H, W), the hooks' rows and the
    layout."""
    ctx = torch.from_numpy(feats[None]).to(oracle.torch_dtype(mode))
    opts: dict = {"sample_sigmas": grid}
    cap = Capture(dm)
    try:
        out = dm(torch.from_numpy(lat[:, :, None]), timestep(torch, sigma),
                 ctx, transformer_options=opts,
                 minimax_payload=None if payload is None else dict(payload))
    finally:
        cap.remove()
    return oracle.to_numpy(out[:, :, 0]), cap, opts["minimax_h3_layout"]


def teacher_forced(torch, dm, cap32: Capture) -> dict:
    """The bf16 output of each refiner block, the refiner's norm, each
    block and the head, fed the fp32 pass's input to them rounded to
    bfloat16."""
    def bf(a):
        return a.to(torch.bfloat16)

    out = {}
    tr = dm.token_refiner
    x = cap32.rows["condition_proj"]
    for i, blk in enumerate(tr.blocks):
        out[f"token_refiner.blocks.{i}"] = oracle.to_numpy(
            blk(bf(x), transformer_options={}))
        x = cap32.rows[f"token_refiner.blocks.{i}"]
    out["token_refiner.final_norm"] = oracle.to_numpy(tr.final_norm(bf(x)))
    x, t_emb, segs, table = cap32.inputs["blocks.0"][:4]
    table = table.to(torch.bfloat16)
    for i, blk in enumerate(dm.blocks):
        out[f"blocks.{i}"] = oracle.to_numpy(
            blk(bf(x), t_emb, segs, table, transformer_options={}))
        x = cap32.rows[f"blocks.{i}"]
    args = list(cap32.inputs["final_layer"])
    args[0] = bf(args[0])
    out["final_layer"] = oracle.to_numpy(dm.final_layer(*args)[0])
    return out


def layout_rows(torch, dm, cap: Capture, layout, sigma) -> dict:
    """The layout's positions, angles and segment table, the distinct
    times and the curve rows, checked against what the forward used."""
    x, t_emb, segs = cap.inputs["blocks.0"][:3]
    s = float(timestep(torch, sigma).flatten()[0]) / 1000.0
    t = 1.0 - s
    # Reference rows run at max(t, 0.999), the distinct times ascending
    # (comfy model.py:871-875).
    refs = any(k == "ref_img" for _, _, k in layout.segments)
    times = sorted({t, max(t, 0.999)} if refs else {t})
    if t_emb.shape[0] != len(times):
        raise SystemExit(f"error: the forward used {t_emb.shape[0]} times")
    table = dm.adaln_t_table.float()
    g = table.shape[0] - 1
    pos = torch.tensor(times, dtype=torch.float32).clamp(0, 1) * g
    i0 = pos.floor().long().clamp(max=g - 1)
    want = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    if not torch.equal(want, t_emb):
        raise SystemExit("error: the curve rows differ from the table blend")
    fl = cap.inputs["final_layer"]
    return {
        "positions": layout.position_ids.numpy().astype(np.float64),
        "rope.angles": oracle.to_numpy(dm.rope_freqs(layout.position_ids,
                                                     "cpu")),
        "segments": np.array([(a, b, int(r)) for a, b, r in segs],
                             np.int64),
        "t_values": np.array(times, np.float64),
        "t_emb": oracle.to_numpy(t_emb),
        "in.blocks.0": oracle.to_numpy(x),
        "head_segment": np.array(fl[2], np.int64),
        "head_sigma": np.float64(fl[4]),
    }


def run_case(torch, dm, lat, feats, sigma, name, inputs, files, notes, *,
             compressed: bool = False, edit: dict | None = None,
             check: int = 3, script: str = SCRIPT,
             sample_sigmas=None, step_head: int | None = None) -> None:
    """One step in fp32 and bf16, written as fixture ``name`` of check
    ``check`` by ``script``. ``edit`` holds the references' latents, their
    sizes' spans of the prompt and the seed of an edit step.
    ``sample_sigmas`` is the sampler's schedule the step is part of, the
    grid by default, from which ComfyUI picks the heads the step reads.
    ``step_head`` records the head a step that reads its head by index
    reads, which the caller has the model read."""
    payload = None
    fx = fixtures.FixtureWriter(name, check=check, family=FAMILY,
                                script=script)
    if edit is not None:
        refs = edit["latents"]
        payload = edit_payload(torch, refs, feats.shape[0], edit["spans"],
                               edit["seed"])
        for i, (z, n) in enumerate(zip(refs, ref_noise(torch, refs,
                                                       edit["seed"]))):
            fx.add(f"ref_latents_{i}", z)
            fx.add(f"ref_noise_{i}", n)
        fx.add("text_tags", payload["text_token_tags"].numpy())
        fx.add("ref_seed", np.int64(edit["seed"]))
        fx.inputs.update(refs=[[int(z.shape[3]) * 16, int(z.shape[2]) * 16]
                               for z in refs],
                         spans=[list(v) for v in edit["spans"]],
                         ref_seed=edit["seed"])
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in notes:
        fx.note(n)
    grid = torch.tensor(dm.pdd_sigmas if sample_sigmas is None
                        else sample_sigmas, dtype=torch.float32)
    fx.inputs.update(sigma=float(np.float32(sigma)),
                     sample_sigmas=[float(v) for v in grid])
    fx.add("latents", lat)
    fx.add("features", feats)
    fx.add("sigma", np.float32(sigma))
    fx.add("timestep", oracle.to_numpy(timestep(torch, sigma)))
    fx.add("sample_sigmas", grid.numpy())
    if step_head is not None:
        fx.inputs.update(heads="in_order", step_head=int(step_head))
        fx.add("step_head", np.int64(step_head))
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        vel[mode], caps[mode], layout = call(torch, dm, lat, feats, sigma,
                                             mode, grid, payload)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    oracle.emulate_bf16_matmul(False)
    fx.inputs["segment_kinds"] = [k for _, _, k in layout.segments]
    for k, arr in layout_rows(torch, dm, caps["fp32"], layout,
                              sigma).items():
        fx.add(k, arr)
    fx.add("velocity", vel["fp32"])
    fx.add("velocity_bf16", vel["bf16"])
    g = fixtures.compare(vel["bf16"], vel["fp32"])
    fx.tolerances["velocity"] = {"rel_l2": 2 * g["rel_l2"],
                                 "cosine_min": 1 - 4 * (1 - g["cosine"])}
    fx.results["velocity_bf16_vs_fp32"] = g
    r32, r16 = caps["fp32"].rows, caps["bf16"].rows
    oracle.emulate_bf16_matmul(True)
    t0 = time.perf_counter()
    tf = teacher_forced(torch, dm, caps["fp32"])
    times["bf16_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
    oracle.emulate_bf16_matmul(False)
    fx.add("in.final_layer",
           oracle.to_numpy(caps["fp32"].inputs["final_layer"][0]))
    row_gaps, free_gaps = {}, {}
    for k, t in r32.items():
        a = oracle.to_numpy(t)
        fx.add(k, a)
        free = fixtures.compare(oracle.to_numpy(r16[k]), a)
        forced = k in tf
        gap = fixtures.compare(tf[k], a) if forced else free
        gap = {**gap, "teacher_forced": forced}
        row_gaps[k] = gap
        fx.tolerances[k] = (
            {"rel_l2": 2 * gap["rel_l2"], "teacher_forced": forced}
            if gap["rel_l2"] > 0 else {"rel_l2": FP32_ROWS, "float32": True})
        if forced:
            free_gaps[k] = free
            fx.tolerances[f"free {k}"] = {"rel_l2": 2 * free["rel_l2"]}
    fx.results["rows_bf16_vs_fp32"] = row_gaps
    fx.results["free_bf16_vs_fp32"] = free_gaps
    fx.results["times"] = times
    fx.tolerances["rule"] = RULE
    path = fx.write(compressed=compressed)
    print(f"[check{check}] {name}: velocity bf16 vs fp32 rel L2 "
          f"{g['rel_l2']:.3e}, {times}, wrote {path}", flush=True)


def load_gguf_dit(torch, path: Path):
    """The DiT through the checkout's UnetLoaderGGUF node with
    dequant_dtype float32, loaded as the sampler loads it."""
    import comfy.model_management
    import folder_paths
    from ComfyUI_GGUF.nodes import UnetLoaderGGUF

    folder_paths.add_model_folder_path("unet", str(path.parent),
                                       is_default=True)
    found = folder_paths.get_full_path("unet", path.name)
    if found is None or not Path(found).samefile(path):
        raise SystemExit(f"error: ComfyUI resolves {path.name} to {found}")
    patcher = UnetLoaderGGUF().load_unet(path.name,
                                         dequant_dtype="float32")[0]
    dm = patcher.model.diffusion_model
    if type(dm).__name__ != "MiniMaxH3ImageModel":
        raise SystemExit(f"error: ComfyUI loaded {type(dm).__name__}")
    if dm.final_layer.pdd_steps != len(PDD_SIGMAS) - 1 or \
            tuple(dm.pdd_sigmas) != PDD_SIGMAS:
        raise SystemExit("error: ComfyUI built another PDD head bank; is "
                         "the config key in the file?")
    comfy.model_management.load_models_gpu([patcher])
    return patcher, dm


def tiny_model(torch, seed: int):
    """ComfyUI's class at the tiny size on random float32 weights, and the
    weights and buffers by name in torch order."""
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3ImageModel

    kw = dict(TINY, patch_size=tuple(TINY["patch_size"]))
    dm = MiniMaxH3ImageModel(**kw, dtype=torch.float32, device="cpu",
                             operations=comfy.ops.manual_cast)
    rng = np.random.default_rng(seed)
    state = {}
    named = dict(dm.named_parameters()) | dict(dm.named_buffers())
    for name, p in named.items():
        shape = tuple(p.shape)
        owner = name.rsplit(".", 1)[0]
        if name == "rope.inv_freq":
            v = 1.0 / 10000 ** (np.arange(shape[0]) / shape[0])
        elif owner.endswith("norm") or owner.endswith("norm1") \
                or owner.endswith("norm2"):
            v = 1 + 0.2 * rng.standard_normal(shape)
        elif name == "adaln_t_table" or name.endswith(".bias"):
            v = 0.5 * rng.standard_normal(shape)
        else:
            v = rng.standard_normal(shape) / np.sqrt(shape[-1])
        state[name] = v.astype(np.float32)
    dm.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()},
                       strict=True)
    return dm.eval().requires_grad_(False), state


def self_test(torch, a) -> int:
    dm, state = tiny_model(torch, a.seed)
    w, h = (int(v) for v in TINY_CASE["shape"].split("x"))
    lat = latents(w, h, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    feats = rng.standard_normal((TINY_CASE["text"], TINY["text_dim"])).astype(
        np.float32)
    name = f"selftest_{FAMILY}_tiny"
    config = {_GMLX.get(k, k): v for k, v in TINY.items()}
    inputs = {"comfy_config": TINY, "config": config, "seed": a.seed,
              "shape": [w, h], "text_rows": TINY_CASE["text"],
              "matmul_bf16": MATMUL, "attention": "attention_pytorch",
              "comfy": a.comfy_head}
    note = f"the weights are in {name}_weights by name in torch order"
    for sigma in TINY_CASE["sigmas"]:
        run_case(torch, dm, lat, feats, sigma,
                 f"{name}_s{PDD_SIGMAS.index(sigma)}", inputs, {}, [note],
                 compressed=True)
    edit = {"latents": ref_latents(TINY_EDIT["refs"], a.seed),
            "spans": TINY_EDIT["spans"], "seed": TINY_EDIT["ref_seed"]}
    for sigma in TINY_CASE["sigmas"]:
        run_case(torch, dm, lat, feats, sigma,
                 f"{name}_edit_s{PDD_SIGMAS.index(sigma)}", inputs, {},
                 [note], compressed=True, edit=edit)
    vision = {"latents": [], "spans": TINY_VISION["spans"],
              "seed": TINY_VISION["ref_seed"]}
    sigma = TINY_CASE["sigmas"][0]
    run_case(torch, dm, lat, feats, sigma,
             f"{name}_vision_s{PDD_SIGMAS.index(sigma)}", inputs, {},
             [note], compressed=True, edit=vision)
    wx = fixtures.FixtureWriter(name + "_weights", check=3, family=FAMILY,
                                script=SCRIPT)
    wx.inputs.update(inputs)
    for k, v in state.items():
        wx.add(k, v)
    wx.write(compressed=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path,
                    help="H3-Image DiT GGUF with the config key")
    ap.add_argument("--source", type=Path,
                    help="the release GGUF that --dit copies")
    ap.add_argument("--cases", nargs="+", default=["512x512:1.0",
                                                   "512x512:0.75"],
                    help="steps as WxH:SIGMA")
    ap.add_argument("--refs", nargs="*", default=[],
                    help="reference sizes WxH: each case runs as an edit "
                         "on seeded latents of these sizes")
    ap.add_argument("--spans", nargs="*", default=[],
                    help="vision spans A:B of the text rows, tag 0 in the "
                         "edit's token tags")
    ap.add_argument("--ref-seed", type=int, default=7,
                    help="the edit's run seed, which draws the noise of "
                         "its reference rows")
    ap.add_argument("--text-len", type=int, default=8)
    ap.add_argument("--feature-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="check3_h3_image")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    co = comfy_oracle()
    a.comfy_head = co.start_comfy(a.comfy_dir)
    if a.self_test:
        return self_test(torch, a)
    if a.dit is None or a.source is None:
        ap.error("--dit and --source are required")
    a.dit = a.dit.expanduser().resolve()
    gguf_head = co.load_comfy_gguf(a.comfy_dir)
    cases = [parse_case(c) for c in a.cases]
    t0 = time.perf_counter()
    _, dm = load_gguf_dit(torch, a.dit)
    print(f"[check3] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s"
          f", {len(dm.blocks)} blocks", flush=True)
    import comfy.ldm.modules.attention as cattn

    rng = np.random.default_rng(a.seed)
    feats = (rng.standard_normal((a.text_len, 5120))
             * a.feature_std).astype(np.float32)
    files = {"dit": a.source, "oracle_dit": a.dit}
    notes = [f"oracle: ComfyUI {a.comfy_head}, ComfyUI-GGUF {gguf_head}, "
             "UnetLoaderGGUF with dequant_dtype float32",
             "oracle_dit is dit with the config key {\"transformer\": "
             "{\"pdd_steps\": 6, \"pdd_sigmas\": [...]}} and every tensor "
             "unchanged"]
    edit = None
    if a.refs:
        sizes = [parse_case(f"{r}:1.0")[:2] for r in a.refs]
        spans = [tuple(int(v) for v in sp.split(":")) for sp in a.spans]
        if any(not 0 < x < y <= a.text_len for x, y in spans):
            raise SystemExit(f"error: spans {a.spans} are not inside "
                             f"{a.text_len} text rows")
        edit = {"latents": ref_latents(sizes, a.seed), "spans": spans,
                "seed": a.ref_seed}
        notes.append("an edit step: references of seeded normal latents "
                     "and token tags 0 over the spans, as "
                     "MiniMaxH3ImageConditioning gives them")
    for w, h, sigma in cases:
        lat = latents(w, h, a.seed)
        k = min(range(len(PDD_SIGMAS)),
                key=lambda i: abs(PDD_SIGMAS[i] - sigma))
        inputs = {"shape": [w, h], "blocks": len(dm.blocks),
                  "seed": a.seed,
                  "features": f"seeded normal, std {a.feature_std}",
                  "text_rows": int(feats.shape[0]),
                  "latents_shape": list(lat.shape), "dequant": DEQUANT,
                  "matmul_bf16": MATMUL,
                  "timestep": "sigma x 1000 in float32 in both modes",
                  "attention": cattn.optimized_attention.__name__,
                  "comfy": a.comfy_head, "comfy_gguf": gguf_head}
        tag = "_edit" if edit else ""
        run_case(torch, dm, lat, feats, sigma,
                 f"{a.name}{tag}_{w}x{h}_s{k}", inputs, files, notes,
                 edit=edit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
