#!/usr/bin/env python3
"""Check 3 oracle for Krea 2: one denoise step of Krea2Transformer2DModel,
block by block, on the weights of a DiT GGUF.

diffusers has no single-file loader for Krea 2, so the script builds the
model itself: it maps each GGUF tensor name to the diffusers name, puts
each quantized matrix in a GGUFLinear over the file's own bytes, and sets
the float tensors by name in the dtype that from_single_file would give
them, the norm weights in float32 as the class asks. Each GGUFLinear
dequantizes its weight on every call in float32 arithmetic (oracle
``dequant="exact"``), which is ggml's. The two final layer matrices that
one converter writes, last.up and last.down, are not read by the class
and are left out. --blocks N keeps the first N blocks, for a memory
ladder or a short fixture.

One step is one call as Krea2Pipeline makes it at batch 1 for one
unpadded prompt: packed latents, the stacked encoder features of the
prompt, the timestep sigma, and the rotary positions of the pipeline,
text at zero and image rows at (0, h, w). The velocity is the output
unpacked, with no sign flip. The pipeline rounds the timestep to the
latent dtype. The script passes it in float32 in both modes, so that the
bf16 against fp32 gap measures arithmetic alone.

For a size in --rows the fixture holds, in float32: the inputs, the
timestep embedding and the shared modulation vectors, the output of
every text fusion block, of the projector and of the text projection,
the image embedding, the input of the first block and the output of
every block, the final layer and the velocity. The bf16 model runs the
same call, and each of its blocks also runs teacher-forced on the fp32
model's input to it. The manifest sets each row's tolerance to twice its
bf16 against fp32 gap: teacher-forced for the blocks, free running for
the rows before them and for the velocity. For the other sizes the
fixture holds the inputs and the velocities.

The features are --features, an .npz with ``features`` (L, layers *
dim), or seeded normal features with the spread of each encoder layer
given by --feature-std.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_krea2.py --dit Krea-2-Turbo-Q4_K_M.gguf --sizes 512 1024

--self-test writes a tiny model in GGUF tensor names with diffusers' own
class on random weights, so the gmlx side can check its conventions on
the CPU.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

CLS = "Krea2Transformer2DModel"
SCRIPT = "scripts/image_oracle/check3_krea2.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each block and the final layer, free running "
        "for the rows before the first block and for the velocity")
UNREAD = ("last.up.weight", "last.down.weight")
_GROUPS = {"blocks": "transformer_blocks",
           "txtfusion.layerwise_blocks": "text_fusion.layerwise_blocks",
           "txtfusion.refiner_blocks": "text_fusion.refiner_blocks"}
_INNER = (("prenorm.scale", "norm1.weight"),
          ("postnorm.scale", "norm2.weight"),
          ("attn.qknorm.qnorm.scale", "attn.norm_q.weight"),
          ("attn.qknorm.knorm.scale", "attn.norm_k.weight"),
          ("attn.wq.", "attn.to_q."), ("attn.wk.", "attn.to_k."),
          ("attn.wv.", "attn.to_v."), ("attn.gate.", "attn.to_gate."),
          ("attn.wo.", "attn.to_out.0."), ("mlp.", "ff."),
          ("mod.lin", "scale_shift_table"))
_TOP = (("first.", "img_in."), ("tmlp.0.", "time_embed.linear_1."),
        ("tmlp.2.", "time_embed.linear_2."),
        ("tproj.1.", "time_mod_proj."),
        ("txtmlp.0.scale", "txt_in.norm.weight"),
        ("txtmlp.1.", "txt_in.linear_1."),
        ("txtmlp.3.", "txt_in.linear_2."),
        ("last.norm.scale", "final_layer.norm.weight"),
        ("last.linear.", "final_layer.linear."),
        ("last.modulation.lin", "final_layer.scale_shift_table"),
        ("txtfusion.projector.weight", "text_fusion.projector.weight"))
_BLOCK = re.compile(r"(blocks|txtfusion\.layerwise_blocks|"
                    r"txtfusion\.refiner_blocks)\.(\d+)\.(.*)")

# The tiny model of --self-test: every width of the real one scaled down,
# with grouped query heads in the blocks and three encoder layers.
TINY = {"in_channels": 16, "num_layers": 2, "attention_head_dim": 16,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 96, "timestep_embed_dim": 32,
        "text_hidden_dim": 32, "num_text_layers": 3,
        "text_num_attention_heads": 2, "text_num_key_value_heads": 2,
        "text_intermediate_size": 48, "num_layerwise_text_blocks": 2,
        "num_refiner_text_blocks": 2, "axes_dims_rope": [4, 6, 6],
        "rope_theta": 1000.0, "norm_eps": 1e-5}
TINY_CASE = {"latent": (4, 6), "text": 5, "sigma": 0.6875}


def diffusers_name(name: str) -> str | None:
    """The diffusers name of a GGUF tensor, None for one the class does
    not read."""
    if name in UNREAD:
        return None
    m = _BLOCK.match(name)
    if m:
        rest = m[3]
        for a, b in _INNER:
            if rest.startswith(a):
                rest = b + rest[len(a):]
                break
        return f"{_GROUPS[m[1]]}.{m[2]}.{rest}"
    for a, b in _TOP:
        if name.startswith(a):
            return b + name[len(a):]
    raise KeyError(f"no diffusers name for {name}")


def gguf_state(torch, path: Path, blocks: int | None) -> tuple[dict, dict]:
    """The file's tensors by diffusers name, quantized ones as
    GGUFParameter, with the codec census. Blocks from ``blocks`` on are
    left out."""
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    reader = gguf.GGUFReader(str(path))
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    out, census = {}, {}
    for t in reader.tensors:
        name = diffusers_name(t.name)
        if name is None:
            continue
        m = re.match(r"transformer_blocks\.(\d+)\.", name)
        if m and blocks is not None and int(m[1]) >= blocks:
            continue
        census[t.tensor_type.name] = census.get(t.tensor_type.name, 0) + 1
        data = torch.from_numpy(np.array(t.data))
        if t.tensor_type in floats:
            if name.endswith("scale_shift_table") and data.ndim == 1:
                data = data.reshape(6, -1)
            if name == "text_fusion.projector.weight" and data.ndim == 1:
                data = data.reshape(1, -1)
            out[name] = data
        else:
            out[name] = GGUFParameter(data, quant_type=t.tensor_type)
    return out, census


def build(torch, state: dict, config: dict, mode: str):
    """The class on ``state``, quantized leaves as GGUFLinear."""
    import diffusers
    from accelerate import init_empty_weights
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 _replace_with_gguf_linear)

    dtype = oracle.torch_dtype(mode)
    with init_empty_weights():
        model = getattr(diffusers, CLS)(**config)
    _replace_with_gguf_linear(model, dtype, state)
    keep = set(model._keep_in_fp32_modules)
    names = {n for n, _ in model.named_parameters()}
    if names != set(state):
        raise SystemExit(f"name map: missing {sorted(names - set(state))[:4]}"
                         f", extra {sorted(set(state) - names)[:4]}")
    for name, v in state.items():
        owner, _, leaf = name.rpartition(".")
        mod = model.get_submodule(owner)
        if not isinstance(v, GGUFParameter):
            fp32 = any(p in keep for p in owner.split("."))
            v = torch.nn.Parameter(v.to(torch.float32 if fp32 else dtype),
                                   requires_grad=False)
        mod._parameters[leaf] = v
    assert not any(p.is_meta for p in model.parameters())
    assert not list(model.buffers())
    return model.eval()


def pack(latents: np.ndarray, p: int = 2) -> np.ndarray:
    b, c, h, w = latents.shape
    x = latents.reshape(b, c, h // p, p, w // p, p).transpose(0, 2, 4, 1, 3,
                                                              5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def unpack(x: np.ndarray, h: int, w: int, p: int = 2) -> np.ndarray:
    b, _, ch = x.shape
    c = ch // (p * p)
    x = x.reshape(b, h // p, w // p, c, p, p).transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h, w)


class Capture:
    """Forward hooks that keep the rows of one call by gmlx module path."""

    def __init__(self, torch, model, blocks: bool):
        self.rows: dict = {}
        self.inputs: dict = {}
        self.handles = []
        tf = model.text_fusion
        mods = [("tmlp", model.time_embed), ("tproj", model.time_mod_proj),
                ("txtfusion.projector", tf.projector),
                ("txtmlp", model.txt_in), ("first", model.img_in),
                ("last", model.final_layer)]
        mods += [(f"txtfusion.layerwise_blocks.{i}", b)
                 for i, b in enumerate(tf.layerwise_blocks)]
        mods += [(f"txtfusion.refiner_blocks.{i}", b)
                 for i, b in enumerate(tf.refiner_blocks)]
        if blocks:
            mods += [(f"blocks.{i}", b)
                     for i, b in enumerate(model.transformer_blocks)]
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


def call(torch, model, lat: np.ndarray, feats: np.ndarray, sigma: float,
         mode: str, layers: int, blocks: bool):
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    dtype = oracle.torch_dtype(mode)
    _, _, h, w = lat.shape
    n = feats.shape[0]
    hs = torch.from_numpy(pack(lat)).to(dtype)
    ehs = torch.from_numpy(feats.reshape(1, n, layers, -1)).to(dtype)
    ids = Krea2Pipeline.prepare_position_ids(n, h // 2, w // 2, "cpu")
    t = torch.tensor([sigma], dtype=torch.float32)
    cap = Capture(torch, model, blocks)
    try:
        out = model(hidden_states=hs, encoder_hidden_states=ehs, timestep=t,
                    position_ids=ids, return_dict=False)[0]
    finally:
        cap.remove()
    vel = unpack(oracle.to_numpy(out), h, w)
    return vel, cap, ids


def teacher_forced(torch, model16, cap32: Capture) -> dict:
    """The bf16 output of each block and of the final layer fed the fp32
    model's inputs to it."""
    out = {}
    names = [k for k in cap32.inputs if k.startswith("blocks.")] + ["last"]
    for name in names:
        mod = (model16.final_layer if name == "last" else
               model16.transformer_blocks[int(name.split(".")[1])])
        args = []
        for a in cap32.inputs[name]:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                a = a.to(torch.bfloat16)
            elif isinstance(a, tuple):
                a = tuple(x.float() for x in a)
            args.append(a)
        out[name] = oracle.to_numpy(mod(*args))
    return out


def rows_of(cap: Capture) -> dict:
    out = {}
    for k, v in cap.rows.items():
        a = oracle.to_numpy(v)
        if k == "txtfusion.projector":
            a = a[..., 0]
        out[k] = a
    return out


def run_case(torch, models: dict, lat, feats, sigma, layers, rows: bool,
             name: str, inputs: dict, files: dict, notes: list) -> None:
    fx = fixtures.FixtureWriter(name, check=3, family="krea2", script=SCRIPT)
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    for n in notes:
        fx.note(n)
    fx.add("latents", lat)
    fx.add("features", feats)
    fx.add("sigma", np.float32(sigma))
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        vel[mode], caps[mode], ids = call(torch, models[mode], lat, feats,
                                          sigma, mode, layers, rows)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    fx.add("position_ids", oracle.to_numpy(ids, np.int32))
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
        fx.add("in.blocks.0", oracle.to_numpy(caps["fp32"].inputs["blocks.0"][0]))
        fx.add("in.last", oracle.to_numpy(caps["fp32"].inputs["last"][0]))
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
            for k in r32 if k.startswith("blocks.")}
    fx.results["times"] = times
    fx.tolerances["rule"] = RULE
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 "
          f"{gap['rel_l2']:.3e}, {times}, wrote {path}", flush=True)


def tiny_state(torch, seed: int) -> dict:
    """Random tiny weights by GGUF name, as float32 numpy arrays in torch
    order, zero-centered norm scales near zero."""
    import diffusers

    torch.manual_seed(seed)
    model = getattr(diffusers, CLS)(**TINY)
    inv = {diffusers_name(g): g for g in tiny_gguf_names()}
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if name.endswith(("norm1.weight", "norm2.weight", "norm.weight",
                          "norm_q.weight", "norm_k.weight")):
            v = rng.standard_normal(shape) * 0.2
        elif name.endswith("scale_shift_table"):
            v = rng.standard_normal(shape) * 0.3
        else:
            fan = shape[-1] if len(shape) > 1 else 16
            v = rng.standard_normal(shape) / np.sqrt(fan)
        out[inv[name]] = v.astype(np.float32)
    return out


def tiny_gguf_names() -> list[str]:
    c = TINY
    inner = ["prenorm.scale", "postnorm.scale", "attn.qknorm.qnorm.scale",
             "attn.qknorm.knorm.scale", "attn.wq.weight", "attn.wk.weight",
             "attn.wv.weight", "attn.gate.weight", "attn.wo.weight",
             "mlp.gate.weight", "mlp.up.weight", "mlp.down.weight"]
    names = [f"blocks.{i}.{n}" for i in range(c["num_layers"])
             for n in inner + ["mod.lin"]]
    for g, k in (("txtfusion.layerwise_blocks", "num_layerwise_text_blocks"),
                 ("txtfusion.refiner_blocks", "num_refiner_text_blocks")):
        names += [f"{g}.{i}.{n}" for i in range(c[k]) for n in inner]
    names += ["first.weight", "first.bias", "tmlp.0.weight", "tmlp.0.bias",
              "tmlp.2.weight", "tmlp.2.bias", "tproj.1.weight",
              "tproj.1.bias", "txtmlp.0.scale", "txtmlp.1.weight",
              "txtmlp.1.bias", "txtmlp.3.weight", "txtmlp.3.bias",
              "txtfusion.projector.weight", "last.norm.scale",
              "last.linear.weight", "last.linear.bias",
              "last.modulation.lin"]
    return names


def self_test(torch, a) -> int:
    """The tiny model, fp32 and bf16, on random weights and inputs."""
    state = tiny_state(torch, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    h, w = (2 * s for s in TINY_CASE["latent"])
    lat = rng.standard_normal((1, 4, h, w)).astype(np.float32)
    feats = rng.standard_normal(
        (TINY_CASE["text"], TINY["num_text_layers"]
         * TINY["text_hidden_dim"])).astype(np.float32)
    torch_state = {diffusers_name(g): torch.from_numpy(v)
                   for g, v in state.items()}
    models = {m: build(torch, torch_state, TINY, m) for m in ("fp32",
                                                             "bf16")}
    name = "selftest_krea2_tiny"
    fx_inputs = {"config": TINY, "case": TINY_CASE, "seed": a.seed}
    run_case(torch, models, lat, feats, TINY_CASE["sigma"],
             TINY["num_text_layers"], True, name, fx_inputs, {}, [
                 "the weights are in selftest_krea2_tiny_weights by GGUF "
                 "name in torch order; mod.lin is the (6, dim) table of the "
                 "class"])
    # The weights go in a second fixture, so the first holds rows alone.
    wx = fixtures.FixtureWriter(name + "_weights", check=3, family="krea2",
                                script=SCRIPT)
    wx.inputs.update(fx_inputs)
    for g, v in state.items():
        wx.add(g, v)
    wx.write()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", type=Path, help="Krea 2 DiT GGUF")
    ap.add_argument("--blocks", type=int, help="keep the first N blocks")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--rows", type=int, nargs="*", default=[512],
                    help="sizes whose fixture holds every row")
    ap.add_argument("--features", type=Path,
                    help=".npz with features (L, layers * dim)")
    ap.add_argument("--features-source", default="",
                    help="what the features are, for the manifest")
    ap.add_argument("--text-len", type=int, default=48)
    ap.add_argument("--feature-std", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check3_krea2")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    if a.self_test:
        return self_test(torch, a)
    if a.dit is None:
        ap.error("--dit is required")
    t0 = time.perf_counter()
    state, census = gguf_state(torch, a.dit, a.blocks)
    import diffusers
    cfg = dict(inspect_defaults(diffusers))
    if a.blocks is not None:
        cfg["num_layers"] = a.blocks
    models = {m: build(torch, state, cfg, m) for m in ("fp32", "bf16")}
    print(f"[check3] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}", flush=True)
    layers = cfg["num_text_layers"]
    rng = np.random.default_rng(a.seed)
    if a.features is not None:
        feats = np.load(a.features)["features"].astype(np.float32)
        source = a.features_source or a.features.name
    else:
        feats = (rng.standard_normal(
            (a.text_len, layers * cfg["text_hidden_dim"])) * a.feature_std
        ).astype(np.float32)
        source = f"seeded normal, std {a.feature_std}"
    files = {"dit": a.dit}
    if a.features is not None:
        files["features"] = a.features
    for size in a.sizes:
        n = size // 8
        lat = np.random.default_rng(a.seed + size).standard_normal(
            (1, 16, n, n)).astype(np.float32)
        inputs = {"size": size, "blocks": cfg["num_layers"],
                  "sigma": a.sigma, "seed": a.seed + size,
                  "features": source, "text_rows": int(feats.shape[0]),
                  "dequant": oracle.dequant_mode(),
                  "matmul_bf16": "bf16 operands, float32 GEMM, one bf16 "
                                 "rounding", "timestep": "float32 in both "
                                 "modes", "gguf_codecs": census}
        tag = f"_{a.blocks}b" if a.blocks is not None else ""
        run_case(torch, models, lat, feats, a.sigma, layers,
                 size in a.rows, f"{a.name}{tag}_{size}", inputs, files, [])
    return 0


def inspect_defaults(diffusers) -> dict:
    import inspect

    sig = inspect.signature(getattr(diffusers, CLS).__init__)
    return {k: (list(v.default) if isinstance(v.default, tuple)
                else v.default)
            for k, v in sig.parameters.items() if k != "self"}


if __name__ == "__main__":
    sys.exit(main())
