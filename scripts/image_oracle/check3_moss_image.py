#!/usr/bin/env python3
"""Check 3 oracle for MOSS-Image: one denoise step of the release's
MOSSImageTransformer, block by block, on the weights of a DiT GGUF.

The class lives in the release's pipeline_moss_image.py, not in
diffusers, so the script loads that file by path from --pipeline and
builds the model itself. The GGUF tensor names are the class's own
parameter names, so no name map is needed. Each quantized matrix goes in
a GGUFLinear over the file's own bytes, which dequantizes on every call
in float32 arithmetic (oracle ``dequant="exact"``), and each float tensor
is set by name in the mode's dtype: the release loads every weight in
bfloat16. The file's rope.bands is checked against the class's own
buffer, which the class rebuilds and never loads. --blocks N keeps the
first N blocks, for a memory ladder or a short fixture.

One step is one call as MOSSImagePipeline makes it for one prompt:
latents (1, 128, H / 16, W / 16), the unpadded encoder features of the
prompt, and the scheduler's timestep, sigma times 1000 in float32 in both
modes. The velocity is the output, with no sign flip.

For a size in --rows the fixture holds, in float32: the inputs, the
timestep embedding and the two modulation outputs, the text projection
and the output of each text block, the image embedding, the input of the
first block and the output of every block, the head and the velocity.
The bf16 model runs the same call, and each text block, each block and
the head also run teacher-forced on the fp32 model's input to them. The
manifest sets each row's tolerance to twice its bf16 against fp32 gap:
teacher-forced for those rows, free running for the others and for the
velocity. For the other sizes the fixture holds the inputs and the
velocities.

The features are --features, an .npz with --features-key (L, 4096), as
check 2 writes them, or seeded normal features of std --feature-std.
The latents are seeded normal at --sigma, or with --latents and --step a
point on a sampling trajectory (an .npz with ``xs`` (steps, 1, 128, h, w)
and ``sigmas``).

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_moss_image.py --pipeline <dir>/pipeline_moss_image.py \\
      --dit MOSS-Image-SFT-Q4_K_M.gguf --sizes 512 --name check3_moss_image

--self-test writes a tiny model of the class on random weights, so the
gmlx side can check its conventions on the CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

CLS = "MOSSImageTransformer"
FAMILY = "moss_image"
SCRIPT = "scripts/image_oracle/check3_moss_image.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each text block, each block and the head, free "
        "running for the other rows and for the velocity")
BANDS = "rope.bands"

# The tiny model of --self-test: every width of the real one scaled down.
TINY = {"dim": 64, "ffn_dim": 48, "num_heads": 2, "num_layers": 2,
        "text_dim": 24, "freq_dim": 16, "latent_channels": 8,
        "rope_base": 10000.0}
TINY_CASE = {"latent": (3, 4), "text": 5, "sigma": 0.6875}


def release(path: Path):
    """The release's pipeline module, loaded by path."""
    spec = importlib.util.spec_from_file_location("moss_release", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gguf_state(torch, path: Path, blocks: int | None):
    """The file's tensors by name, quantized ones as GGUFParameter, with
    the codec census and the stored bands. Blocks from ``blocks`` on are
    left out."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    reader = gguf.GGUFReader(str(path))
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
              gguf.GGMLQuantizationType.BF16)
    out, census, bands = {}, {}, None
    for t in reader.tensors:
        name = t.name
        for p in ("model.diffusion_model.", "diffusion_model."):
            name = name.removeprefix(p)
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
        if name == BANDS:
            bands = np.array(t.data, np.float32)
            continue
        m = re.match(r"layers\.(\d+)\.", name)
        if m and blocks is not None and int(m[1]) >= blocks:
            continue
        data = np.array(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = (data.view(np.uint16).astype(np.uint32) << 16).view(
                np.float32)
        if t.tensor_type in floats:
            out[name] = torch.from_numpy(np.ascontiguousarray(data))
        else:
            out[name] = GGUFParameter(torch.from_numpy(data),
                                      quant_type=t.tensor_type)
    return out, census, bands


_TYPED: dict = {}


def typed(dtype):
    """A GGUFParameter whose ``dtype`` reads as the compute dtype. The
    class casts each block's input to ``proj["q"].weight.dtype``, which on
    the stored bytes would be uint8."""
    from diffusers.quantizers.gguf.utils import GGUFParameter

    if dtype not in _TYPED:
        _TYPED[dtype] = type(f"GGUFParameter_{dtype}".replace(".", "_"),
                             (GGUFParameter,),
                             {"dtype": property(lambda self: dtype)})
    return _TYPED[dtype]


def build(torch, rel, state: dict, config: dict, mode: str):
    """The class on ``state``, quantized leaves as GGUFLinear over a view
    of the stored bytes typed for the mode."""
    from accelerate import init_empty_weights
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 _replace_with_gguf_linear)

    dtype = oracle.torch_dtype(mode)
    with init_empty_weights():
        model = getattr(rel, CLS)(**config)
    _replace_with_gguf_linear(model, dtype, state)
    names = {n for n, _ in model.named_parameters()}
    if names != set(state):
        raise SystemExit(f"names: missing {sorted(names - set(state))[:4]}, "
                         f"extra {sorted(set(state) - names)[:4]}")
    for name, v in state.items():
        owner, _, leaf = name.rpartition(".")
        mod = model.get_submodule(owner)
        if isinstance(v, GGUFParameter):
            v = typed(dtype)(v.as_tensor(), quant_type=v.quant_type)
        else:
            v = torch.nn.Parameter(v.to(dtype), requires_grad=False)
        mod._parameters[leaf] = v
    assert not any(p.is_meta for p in model.parameters())
    bufs = dict(model.named_buffers())
    assert set(bufs) == {BANDS}, sorted(bufs)
    assert bufs[BANDS].dtype == torch.float32
    return model.eval()


class Capture:
    """Forward hooks that keep the rows of one call by gmlx module path."""

    def __init__(self, model, blocks: bool):
        self.rows: dict = {}
        self.inputs: dict = {}
        self.handles = []
        te = model.text_embedder
        mods = [("time_embedding", model.time_embedding),
                ("shared_block_modulation", model.shared_block_modulation),
                ("video_head_modulation", model.video_head_modulation),
                ("text_embedder.proj", te.proj),
                ("video_embedder", model.video_embedder),
                ("final_norm_video", model.final_norm_video),
                ("final_linear_video", model.final_linear_video)]
        mods += [(f"text_embedder.blocks.{i}", b)
                 for i, b in enumerate(te.blocks)]
        if blocks:
            mods += [(f"layers.{i}", b) for i, b in enumerate(model.layers)]
        for name, mod in mods:
            self.handles.append(mod.register_forward_hook(self._keep(name)))
            self.handles.append(
                mod.register_forward_pre_hook(self._feed(name)))

    def _keep(self, name):
        def hook(_mod, _args, out):
            self.rows[name] = out.detach()
        return hook

    def _feed(self, name):
        def hook(_mod, args):
            self.inputs[name] = tuple(a.detach() if hasattr(a, "detach")
                                      else a for a in args)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def timestep(torch, sigma: float):
    """The scheduler's timestep: sigma times 1000, in float32."""
    s = np.float32(sigma) * np.float32(1000)
    return torch.tensor([s], dtype=torch.float32)


def call(torch, model, lat: np.ndarray, feats: np.ndarray, sigma: float,
         mode: str, blocks: bool):
    dtype = oracle.torch_dtype(mode)
    hs = torch.from_numpy(lat)
    ehs = torch.from_numpy(feats[None]).to(dtype)
    cap = Capture(model, blocks)
    try:
        out = model(hs, timestep(torch, sigma), ehs, return_dict=False)[0]
    finally:
        cap.remove()
    return oracle.to_numpy(out), cap


def head(torch, model, x, mod):
    """The release's output head on image rows ``x`` with the head
    modulation ``mod`` (1, 2 dim), as its forward runs it."""
    m = mod.view(1, 2, -1)
    y = model.final_norm_video(x.to(model.final_norm_video.weight.dtype))
    y = y * (1 + m[:, 1, None]) + m[:, 0, None]
    return model.final_linear_video(y.to(model.final_linear_video.weight.dtype))


def teacher_forced(torch, model16, cap32: Capture) -> dict:
    """The bf16 output of each text block, each block and the head fed
    the fp32 model's inputs to them, rounded to bfloat16 as the bf16 run
    holds them."""
    def bf(a):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            return a.to(torch.bfloat16)
        return a

    out = {}
    for name, args in cap32.inputs.items():
        if name.startswith("layers."):
            mod = model16.layers[int(name.split(".")[1])]
        elif name.startswith("text_embedder.blocks."):
            mod = model16.text_embedder.blocks[int(name.split(".")[2])]
        else:
            continue
        out[name] = oracle.to_numpy(mod(*(bf(a) for a in args)))
    x = cap32.inputs["final_norm_video"][0]
    mod = cap32.rows["video_head_modulation"]
    out["final_linear_video"] = oracle.to_numpy(
        head(torch, model16, bf(x), bf(mod)))
    return out


def run_case(torch, models: dict, lat, feats, sigma, rows: bool, name: str,
             inputs: dict, files: dict, notes: list) -> dict:
    """One step in fp32 and bf16 written as fixture ``name``."""
    fx = fixtures.FixtureWriter(name, check=3, family=FAMILY, script=SCRIPT)
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in notes:
        fx.note(n)
    fx.add("latents", lat)
    fx.add("features", feats)
    fx.add("sigma", np.float32(sigma))
    fx.add("timestep", oracle.to_numpy(timestep(torch, sigma)))
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        vel[mode], caps[mode] = call(torch, models[mode], lat, feats, sigma,
                                     mode, rows)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    fx.add("velocity", vel["fp32"])
    fx.add("velocity_bf16", vel["bf16"])
    gap = fixtures.compare(vel["bf16"], vel["fp32"])
    fx.results["velocity_bf16_vs_fp32"] = gap
    fx.tolerances["velocity"] = {"rel_l2": 2 * gap["rel_l2"],
                                 "cosine_min": 1 - 4 * (1 - gap["cosine"])}
    if rows:
        r32 = {k: oracle.to_numpy(v) for k, v in caps["fp32"].rows.items()
               if k != "final_norm_video"}
        r16 = {k: oracle.to_numpy(v) for k, v in caps["bf16"].rows.items()
               if k != "final_norm_video"}
        oracle.emulate_bf16_matmul(True)
        t0 = time.perf_counter()
        tf = teacher_forced(torch, models["bf16"], caps["fp32"])
        times["bf16_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
        oracle.emulate_bf16_matmul(False)
        fx.add("in.layers.0",
               oracle.to_numpy(caps["fp32"].inputs["layers.0"][0]))
        fx.add("in.final_norm_video",
               oracle.to_numpy(caps["fp32"].inputs["final_norm_video"][0]))
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
            k: fixtures.compare(r16[k], r32[k])["rel_l2"]
            for k in r32 if k.startswith("layers.")}
    fx.results["times"] = times
    fx.tolerances["rule"] = RULE
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 "
          f"{gap['rel_l2']:.3e}, {times}, wrote {path}", flush=True)
    return vel


def tiny_state(torch, rel, seed: int) -> tuple[dict, np.ndarray]:
    """Random tiny weights by name, float32 numpy arrays in torch order,
    norm weights near one, and the class's bands."""
    torch.manual_seed(seed)
    model = getattr(rel, CLS)(**TINY)
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if "norm" in name.split(".")[-2]:
            v = 1 + rng.standard_normal(shape) * 0.2
        else:
            fan = shape[-1] if len(shape) > 1 else 16
            v = rng.standard_normal(shape) / np.sqrt(fan)
        out[name] = v.astype(np.float32)
    return out, oracle.to_numpy(model.rope.bands)


def self_test(torch, rel, a) -> int:
    """The tiny model, fp32 and bf16, on random weights and inputs."""
    state, bands = tiny_state(torch, rel, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    h, w = TINY_CASE["latent"]
    lat = rng.standard_normal((1, TINY["latent_channels"], h, w)).astype(
        np.float32)
    feats = rng.standard_normal((TINY_CASE["text"], TINY["text_dim"])).astype(
        np.float32)
    torch_state = {g: torch.from_numpy(v) for g, v in state.items()}
    models = {m: build(torch, rel, torch_state, TINY, m)
              for m in ("fp32", "bf16")}
    name = f"selftest_{FAMILY}_tiny"
    fx_inputs = {"config": TINY, "case": TINY_CASE, "seed": a.seed}
    run_case(torch, models, lat, feats, TINY_CASE["sigma"], True, name,
             fx_inputs, {"pipeline": a.pipeline}, [
                 f"the weights are in {name}_weights by name in torch "
                 "order, with the class's rope.bands buffer"])
    wx = fixtures.FixtureWriter(name + "_weights", check=3, family=FAMILY,
                                script=SCRIPT)
    wx.inputs.update(fx_inputs)
    for g, v in state.items():
        wx.add(g, v)
    wx.add(BANDS, bands)
    wx.write()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipeline", type=Path, required=True,
                    help="the release's pipeline_moss_image.py")
    ap.add_argument("--dit", type=Path, help="MOSS-Image DiT GGUF")
    ap.add_argument("--config", type=Path,
                    help="the release's transformer/config.json")
    ap.add_argument("--blocks", type=int, help="keep the first N blocks")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--rows", type=int, nargs="*", default=[512],
                    help="sizes whose fixture holds every row")
    ap.add_argument("--features", type=Path,
                    help=".npz with features (L, 4096)")
    ap.add_argument("--features-key", default="features",
                    help="array of --features to read")
    ap.add_argument("--features-source", default="",
                    help="what the features are, for the manifest")
    ap.add_argument("--latents", type=Path,
                    help=".npz with xs and sigmas of a sampling trajectory")
    ap.add_argument("--step", type=int,
                    help="step of --latents whose x and sigma the step takes")
    ap.add_argument("--latents-source", default="",
                    help="what --latents is, for the manifest")
    ap.add_argument("--text-len", type=int, default=48)
    ap.add_argument("--feature-std", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check3_moss_image")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    rel = release(a.pipeline)
    if a.self_test:
        return self_test(torch, rel, a)
    if a.dit is None or a.config is None:
        ap.error("--dit and --config are required")
    import json

    cfg = {k: v for k, v in json.loads(a.config.read_text()).items()
           if not k.startswith("_")}
    t0 = time.perf_counter()
    state, census, stored = gguf_state(torch, a.dit, a.blocks)
    if a.blocks is not None:
        cfg["num_layers"] = a.blocks
    models = {m: build(torch, rel, state, cfg, m) for m in ("fp32", "bf16")}
    own = oracle.to_numpy(models["fp32"].rope.bands)
    if stored is None or stored.tobytes() != own.tobytes():
        raise SystemExit("error: the file's rope.bands differ from the "
                         "class's")
    print(f"[check3] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    rng = np.random.default_rng(a.seed)
    if a.features is not None:
        feats = np.load(a.features)[a.features_key].astype(np.float32)
        source = a.features_source or a.features.name
    else:
        feats = (rng.standard_normal((a.text_len, cfg["text_dim"]))
                 * a.feature_std).astype(np.float32)
        source = f"seeded normal, std {a.feature_std}"
    files = {"dit": a.dit, "pipeline": a.pipeline, "config": a.config}
    if a.features is not None:
        files["features"] = a.features
    traj = None
    if a.latents is not None:
        if a.step is None or len(a.sizes) != 1:
            ap.error("--latents takes --step and one size in --sizes")
        files["latents"] = a.latents
        with np.load(a.latents) as z:
            x = z["xs"][a.step].astype(np.float32)
            traj = (x.reshape(1, x.shape[1], *x.shape[-2:]),
                    float(z["sigmas"][a.step]))
        if traj[0].shape[-1] * 16 != a.sizes[0]:
            ap.error(f"--latents holds {traj[0].shape}, not size "
                     f"{a.sizes[0]}")
    for size in a.sizes:
        n = size // 16
        if traj is None:
            lat = np.random.default_rng(a.seed + size).standard_normal(
                (1, cfg["latent_channels"], n, n)).astype(np.float32)
            sigma, origin = a.sigma, {"seed": a.seed + size}
        else:
            lat, sigma = traj
            origin = {"latents": a.latents_source or a.latents.name,
                      "step": a.step}
        inputs = {"size": size, "blocks": cfg["num_layers"],
                  "sigma": sigma, **origin, "features": source,
                  "text_rows": int(feats.shape[0]),
                  "dequant": oracle.dequant_mode(),
                  "matmul_bf16": "bf16 operands, float32 GEMM, one bf16 "
                                 "rounding",
                  "timestep": "sigma x 1000 in float32 in both modes",
                  "gguf_codecs": census}
        tag = f"_{a.blocks}b" if a.blocks is not None else ""
        run_case(torch, models, lat, feats, sigma, size in a.rows,
                 f"{a.name}{tag}_{size}", inputs, files, [])
    return 0


if __name__ == "__main__":
    sys.exit(main())
