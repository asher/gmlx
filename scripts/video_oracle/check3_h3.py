#!/usr/bin/env python3
"""Check 3 oracle for MiniMax-H3: one denoise step of ComfyUI's
MiniMaxH3Model, block by block, on the weights of the DiT GGUF.

ComfyUI is the oracle since it is the reference that runs the GGUF's
curve-table form. The script starts the ComfyUI checkout of --comfy-dir
on the CPU and loads the GGUF through the checkout's ComfyUI-GGUF node
(UnetLoaderGGUF) with dequant_dtype float32: each quantized matrix stays
in the file's bytes and dequantizes on every call with its block math in
float32, as ggml does. Each linear casts its weight to the dtype of its
input, so one model runs both modes. The fp32 pass takes float32 text
features, the bf16 pass bfloat16 ones and runs each linear as a float32
GEMM over bfloat16 operands. Attention runs through torch's
scaled_dot_product_attention (oracle.start_comfy).

One step is one forward as ComfyUI's sampler makes it for a text to video
prompt: video latents (1, 24, T, H, W), audio latents (1, 32, 2, T), the
timestep sigma x 1000 in float32 and the text features (1, L, 5120), with
no payload, so the audio carry is 1. The model derives the video time
1 - sigma and the audio time from the audio shift, and returns the
negated heads as the velocity of each stream.

Each fixture holds, in float32 under gmlx's module
paths: the text projection, each refiner block and the refiner's norm,
the two patch projections, the curve rows, the input of the first block
and the output of every block, the head's input and modulation and both
heads, and the velocities. Every fixture holds the inputs, the layout's
positions, the RoPE angles, the segment table with each segment's
modulation row, the distinct times and the velocities.

The bf16 pass runs the same call, and each refiner block, the refiner's
norm, each block and the head also run teacher-forced on the fp32 pass's
input to them. The manifest sets each row's tolerance to twice its bf16
against fp32 gap: teacher-forced for those rows, free running for the
others and for the velocities. Each teacher-forced row also has a free
running tolerance, ``free <row>``, twice its gap in the free bf16 pass.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      --threads 16 check3_h3.py --comfy-dir <ComfyUI checkout> \\
      --dit minimax_h3_fl2va_pruned-Q4_K.gguf \\
      --shapes 64x64x5 512x512x22 --name check3_h3

--self-test writes a tiny model of the class on random weights, with the
weights by name, so the gmlx side checks the whole forward on the CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "gguf", "einops", "comfy_kitchen")

FAMILY = "h3"
SCRIPT = "scripts/video_oracle/check3_h3.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each refiner block, the refiner's norm, each "
        "block and the heads, free running for the other rows and for the "
        "velocities, and free running for each teacher-forced row as "
        "'free <row>'; rows in float32 in both modes take a float32 "
        "tolerance")
DEQUANT = ("ComfyUI-GGUF dequantize_tensor with dequant_dtype float32: "
           "K-quant block math in float32, on every call")
MATMUL = "bf16 operands, float32 GEMM, one bf16 rounding"
# Rows the model computes in float32 in both modes (the patch projections
# and the head's modulation) have no bf16 gap; their tolerance is this
# float32 relative L2 instead.
FP32_ROWS = 1e-5

# The tiny model of --self-test, in ComfyUI's constructor arguments, and
# the same model in the official config's names that gmlx reads.
TINY = {"hidden_size": 64, "num_layers": 2, "token_refiner_num_layers": 1,
        "num_attention_heads": 2, "attention_head_dim": 128,
        "ffn_hidden_size": 48, "latents_dim": 24, "audio_latents_dim": 32,
        "patch_size": [1, 2, 2], "text_dim": 40, "time_embed_dim": 8,
        "rope_inv_freq_len": 16, "adaln_curve_grid": 1025,
        "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5}
_OFFICIAL = {"hidden_size": "hidden_size", "num_layers": "num_layers",
             "token_refiner_num_layers": "num_refiner_layers",
             "num_attention_heads": "num_attention_heads",
             "attention_head_dim": "attention_head_dim",
             "ffn_hidden_size": "ffn_dim", "latents_dim": "in_channels",
             "audio_latents_dim": "audio_in_channels",
             "patch_size": "patch_size", "text_dim": "text_dim",
             "rope_inv_freq_len": "rope_freq_dim",
             "adaln_curve_grid": "adaln_grid",
             "time_embed_dim": "adaln_basis", "norm_eps": "norm_eps",
             "qk_norm_eps": "qk_norm_eps",
             "final_norm_eps": "final_norm_eps"}
TINY_CASE = {"shape": "96x64x22", "text": 8, "sigma": 0.6875}


def parse_shape(s: str) -> tuple[int, int, int]:
    try:
        w, h, f = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise SystemExit(f"error: shape {s!r} is not WxHxFRAMES") from None
    if w % 32 or h % 32 or f < 5 or (f - 5) % 17:
        raise SystemExit(f"error: shape {s!r}: W and H must be multiples "
                         "of 32 and FRAMES on the 17 k + 5 grid")
    return w, h, f


def latents(shape: tuple, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded normal video and audio latents in ComfyUI's empty AV latent
    shapes for a WxHxFRAMES clip."""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    lat, _ = _empty_av_latent(*shape)
    v, a = lat["samples"].unbind()
    rng = np.random.default_rng([seed, *shape])
    return (rng.standard_normal(tuple(v.shape)).astype(np.float32),
            rng.standard_normal(tuple(a.shape)).astype(np.float32))


def timestep(torch, sigma: float):
    """The sampler's timestep: sigma times 1000, in float32."""
    return torch.tensor([np.float32(sigma) * np.float32(1000)],
                        dtype=torch.float32)


def stream_times(torch, ts, dm) -> tuple[float, float, float]:
    """sigma, the video time and the audio time, as the forward derives
    them from the timestep."""
    from comfy.ldm.minimax.model import time_shift_sigma

    sigma_v = (ts.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, dm.sigma_shift_video,
                                       dm.sigma_shift_audio))
    return float(sigma_v), t_v, t_a


def _clone(a):
    return a.detach().clone() if hasattr(a, "detach") else a


class Capture:
    """Hooks that keep copies of rows of one forward by gmlx module path.
    The blocks update the hidden state in place, so every row is copied
    when its hook runs."""

    def __init__(self, dm):
        self.rows: dict = {}
        self.inputs: dict = {}
        self.handles = []
        fl, tr = dm.final_layer, dm.token_refiner
        keep = [("condition_proj", dm.condition_proj),
                ("video_patch_proj", dm.video_patch_proj),
                ("audio_patch_proj", dm.audio_patch_proj),
                ("token_refiner.final_norm", tr.final_norm),
                ("final_layer.adaln_proj.linear", fl.adaln_proj.linear),
                ("final_layer.video_out", fl.video_out),
                ("final_layer.audio_out", fl.audio_out)]
        keep += [(f"token_refiner.blocks.{i}", b)
                 for i, b in enumerate(tr.blocks)]
        keep += [(f"blocks.{i}", b) for i, b in enumerate(dm.blocks)]
        feed = [("blocks.0", dm.blocks[0]), ("final_layer", fl)]
        for name, mod in keep:
            self.handles.append(mod.register_forward_hook(self._keep(name)))
        for name, mod in feed:
            self.handles.append(
                mod.register_forward_pre_hook(self._feed(name)))

    def _keep(self, name):
        def hook(_mod, _args, out):
            self.rows[name] = _clone(out)
        return hook

    def _feed(self, name):
        def hook(_mod, args):
            self.inputs[name] = tuple(_clone(a) for a in args)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def call(torch, dm, video, audio, feats, sigma, mode):
    """One forward; the velocities, the hooks' rows and the layout."""
    dtype = oracle.torch_dtype(mode)
    ctx = torch.from_numpy(feats[None]).to(dtype)
    opts: dict = {}
    cap = Capture(dm)
    try:
        out = dm([torch.from_numpy(video), torch.from_numpy(audio)],
                 timestep(torch, sigma), ctx, transformer_options=opts)
    finally:
        cap.remove()
    return (oracle.to_numpy(out[0]), oracle.to_numpy(out[1]), cap,
            opts["minimax_h3_layout"])


def teacher_forced(torch, dm, cap32: Capture) -> dict:
    """The bf16 output of each refiner block, the refiner's norm, each
    block and the heads, fed the fp32 pass's input to them rounded to
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
    v, a = dm.final_layer(*args)
    out["final_layer.video_out"] = oracle.to_numpy(v)
    out["final_layer.audio_out"] = oracle.to_numpy(a)
    return out


def layout_rows(torch, dm, cap: Capture, layout, sigma_v, t_v, t_a):
    """The layout's positions, angles and segment table, the distinct
    times and the curve rows, checked against what the forward used."""
    x, t_emb, segs = cap.inputs["blocks.0"][:3]
    times = sorted({t_v, t_a})
    if t_emb.shape[0] != len(times):
        raise SystemExit(f"error: the forward used {t_emb.shape[0]} times, "
                         f"a text to video step has {len(times)}")
    table = dm.adaln_t_table.float()
    g = table.shape[0] - 1
    pos = torch.tensor(times, dtype=torch.float32).clamp(0, 1) * g
    i0 = pos.floor().long().clamp(max=g - 1)
    want = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    if not torch.equal(want, t_emb):
        raise SystemExit("error: the curve rows differ from the table blend "
                         "at the derived times")
    return {
        "positions": layout.position_ids.numpy().astype(np.float64),
        "rope.angles": oracle.to_numpy(dm.rope_freqs(layout.position_ids,
                                                     "cpu")),
        "segments": np.array([(a, b, int(r)) for a, b, r in segs],
                             np.int64),
        "t_values": np.array(times, np.float64),
        "t_emb": oracle.to_numpy(t_emb),
        "in.blocks.0": oracle.to_numpy(x),
    }


def run_case(torch, dm, video, audio, feats, sigma, name, inputs, files,
             notes) -> None:
    """One step in fp32 and bf16, written as fixture ``name``."""
    fx = fixtures.FixtureWriter(name, check=3, family=FAMILY, script=SCRIPT)
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in notes:
        fx.note(n)
    ts = timestep(torch, sigma)
    sigma_v, t_v, t_a = stream_times(torch, ts, dm)
    fx.inputs.update(sigma_v=sigma_v, t_video=t_v, t_audio=t_a,
                     shifts={"video": dm.sigma_shift_video,
                             "audio": dm.sigma_shift_audio})
    fx.add("video", video)
    fx.add("audio", audio)
    fx.add("features", feats)
    fx.add("sigma", np.float32(sigma))
    fx.add("timestep", oracle.to_numpy(ts))
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        v, a, caps[mode], layout = call(torch, dm, video, audio, feats,
                                        sigma, mode)
        vel[mode] = {"video": v, "audio": a}
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    oracle.emulate_bf16_matmul(False)
    fx.inputs["segment_kinds"] = [k for _, _, k in layout.segments]
    for k, arr in layout_rows(torch, dm, caps["fp32"], layout, sigma_v, t_v,
                              t_a).items():
        fx.add(k, arr)
    gaps = {}
    for stream in ("video", "audio"):
        fx.add(f"velocity_{stream}", vel["fp32"][stream])
        fx.add(f"velocity_{stream}_bf16", vel["bf16"][stream])
        g = fixtures.compare(vel["bf16"][stream], vel["fp32"][stream])
        gaps[stream] = g
        fx.tolerances[f"velocity_{stream}"] = {
            "rel_l2": 2 * g["rel_l2"], "cosine_min": 1 - 4 * (1 - g["cosine"])}
    fx.results["velocity_bf16_vs_fp32"] = gaps
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
        g = fixtures.compare(tf[k], a) if forced else free
        g = {**g, "teacher_forced": forced}
        row_gaps[k] = g
        fx.tolerances[k] = (
            {"rel_l2": 2 * g["rel_l2"], "teacher_forced": forced}
            if g["rel_l2"] > 0 else {"rel_l2": FP32_ROWS, "float32": True})
        if forced:
            free_gaps[k] = free
            fx.tolerances[f"free {k}"] = {"rel_l2": 2 * free["rel_l2"]}
    fx.results["rows_bf16_vs_fp32"] = row_gaps
    fx.results["free_bf16_vs_fp32"] = free_gaps
    fx.results["times"] = times
    fx.tolerances["rule"] = RULE
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 video "
          f"{gaps['video']['rel_l2']:.3e} audio {gaps['audio']['rel_l2']:.3e}"
          f", {times}, wrote {path}", flush=True)


def load_gguf_dit(torch, path: Path, blocks: int | None):
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
    if type(dm).__name__ != "MiniMaxH3Model":
        raise SystemExit(f"error: ComfyUI loaded {type(dm).__name__}")
    if blocks is not None:
        dm.blocks = dm.blocks[:blocks]
    comfy.model_management.load_models_gpu([patcher])
    return patcher, dm


def tiny_model(torch, seed: int):
    """ComfyUI's class at the tiny size on random float32 weights, and the
    weights and buffers by name in torch order."""
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    kw = dict(TINY, patch_size=tuple(TINY["patch_size"]))
    dm = MiniMaxH3Model(**kw, dtype=torch.float32, device="cpu",
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
    shape = parse_shape(TINY_CASE["shape"])
    video, audio = latents(shape, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    feats = rng.standard_normal((TINY_CASE["text"], TINY["text_dim"])).astype(
        np.float32)
    name = f"selftest_{FAMILY}_tiny"
    config = {_OFFICIAL[k]: v for k, v in TINY.items()}
    inputs = {"comfy_config": TINY, "config": config, "case": TINY_CASE,
              "seed": a.seed, "shape": list(shape), "text_rows":
              TINY_CASE["text"], "matmul_bf16": MATMUL,
              "attention": "attention_pytorch", "comfy": a.comfy_head}
    run_case(torch, dm, video, audio, feats, TINY_CASE["sigma"], name,
             inputs, {}, [
                 f"the weights are in {name}_weights by name in torch order"])
    wx = fixtures.FixtureWriter(name + "_weights", check=3, family=FAMILY,
                                script=SCRIPT)
    wx.inputs.update(inputs)
    for k, v in state.items():
        wx.add(k, v)
    wx.write()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout with custom_nodes/ComfyUI-GGUF")
    ap.add_argument("--dit", type=Path, help="MiniMax-H3 DiT GGUF")
    ap.add_argument("--blocks", type=int, help="keep the first N blocks")
    ap.add_argument("--shapes", nargs="+", default=["64x64x5", "512x512x22"],
                    help="clip shapes WxHxFRAMES")
    ap.add_argument("--features", type=Path,
                    help=".npz with text features (L, 5120)")
    ap.add_argument("--features-key", default="features",
                    help="array of --features to read")
    ap.add_argument("--features-source", default="",
                    help="what the features are, for the manifest")
    ap.add_argument("--text-len", type=int, default=8)
    ap.add_argument("--feature-std", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="check3_h3")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    a.comfy_head = oracle.start_comfy(a.comfy_dir)
    if a.self_test:
        return self_test(torch, a)
    if a.dit is None:
        ap.error("--dit is required")
    a.dit = a.dit.expanduser().resolve()
    gguf_head = oracle.load_comfy_gguf(a.comfy_dir)
    shapes = [parse_shape(s) for s in a.shapes]
    t0 = time.perf_counter()
    _patcher, dm = load_gguf_dit(torch, a.dit, a.blocks)
    depth = len(dm.blocks)
    print(f"[check3] loaded {a.dit.name} in {time.perf_counter() - t0:.1f}s"
          f", {depth} blocks", flush=True)
    import comfy.ldm.modules.attention as cattn

    if a.features is not None:
        feats = np.load(a.features)[a.features_key].astype(np.float32)
        source = a.features_source or a.features.name
    else:
        rng = np.random.default_rng(a.seed)
        feats = (rng.standard_normal((a.text_len, 5120))
                 * a.feature_std).astype(np.float32)
        source = f"seeded normal, std {a.feature_std}"
    files = {"dit": a.dit}
    if a.features is not None:
        files["features"] = a.features
    notes = [f"oracle: ComfyUI {a.comfy_head}, ComfyUI-GGUF {gguf_head}, "
             "UnetLoaderGGUF with dequant_dtype float32"]
    for shape in shapes:
        video, audio = latents(shape, a.seed)
        inputs = {"shape": list(shape), "blocks": depth, "sigma": a.sigma,
                  "seed": a.seed, "features": source,
                  "text_rows": int(feats.shape[0]),
                  "video_shape": list(video.shape),
                  "audio_shape": list(audio.shape), "dequant": DEQUANT,
                  "matmul_bf16": MATMUL,
                  "timestep": "sigma x 1000 in float32 in both modes",
                  "attention": cattn.optimized_attention.__name__,
                  "comfy": a.comfy_head, "comfy_gguf": gguf_head}
        tag = f"_{a.blocks}b" if a.blocks is not None else ""
        w, h, f = shape
        run_case(torch, dm, video, audio, feats, a.sigma,
                 f"{a.name}{tag}_{w}x{h}x{f}", inputs, files, notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
