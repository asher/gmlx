#!/usr/bin/env python3
"""Check 3 oracle for FLUX.2 klein: one denoise step of BFL's reference
DiT, block by block, on the weights of a DiT GGUF.

The reference is BFL's flux2 repository, read from --reference (its src
directory), at the commit --reference-commit names, which the manifest
records. Its Flux2 class has no GGUF loader, and the GGUF tensor names are
its state dict keys, so the script builds the class on the meta device,
puts each quantized and BF16 matrix in a diffusers GGUFLinear over the
file's own bytes, and sets the F32 and F16 tensors in the mode's dtype.
Each GGUFLinear dequantizes its weight on every call in float32
arithmetic (oracle ``dequant="exact"``), which is ggml's.

One step is one call as the reference's denoise makes it for one sample:
the image tokens of batched_prc_img, then the tokens of the reference
images, if any, at the time offsets of encode_image_refs, the text tokens
of batched_prc_txt, and the step's time in bf16, as denoise fills it in
the dtype of the bf16 latents. The reference casts the model to bf16.
Here the fp32 model runs every op in float32 and the bf16 model every op
in bfloat16, products as one float32 GEMM over the bf16 operands with one
rounding (oracle.emulate_bf16_matmul). The time embedding is the
reference's own on the bf16 time, which it returns in bf16, in both modes;
a pre-hook casts it to the fp32 model's dtype.

The latents are seeded normal rounded to bf16, as the reference's noise
is, at the time of --step in get_schedule(--steps, image tokens), which is
1 at the first step, or at --sigma. The text features are seeded normal
with --feature-std, rounded to bf16, in --text-len rows, which the
reference's encoder always pads to 512. With --refs N, N reference images
of --ref-size pixels each give seeded normal tokens, bf16.

Every fixture holds the reference's time embedding as time_in takes it.
For a size in --rows the fixture holds, in float32: the inputs, the time
vector, the modulations, the token embeddings, the image and text output
of every double block, the output of every single block, the input of the
first single block and of the final layer, the final layer and the
velocity. The bf16 model runs the same call, and each of its blocks and
its final layer also run teacher-forced on the fp32 model's input to them,
with the fp32 model's modulations. The manifest sets each row's tolerance
to twice its bf16 against fp32 gap: teacher-forced for the blocks and the
final layer, free running for the rest and for the velocity. For the
other sizes the fixture holds the inputs and the velocities.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_flux2_klein.py --reference <flux2>/src \\
      --reference-commit <sha> --dit flux-2-klein-4b-Q4_K_M.gguf \\
      --sizes 512 --refs 1

With --kv the file is FLUX.2 klein 9B KV. A size without references is
the step above. With --refs N a fixture holds the reference's cached
pass, denoise_cached: forward_kv_extract with the references at the time
of --step, then forward_kv_cached on other seeded latents at the time of
the next step, each velocity in fp32 and bf16 with the gap between them,
and, for a size in --rows, the reference K and V of the blocks of
KV_BLOCKS in float32 with the gap of each.

--self-test writes a tiny model of the reference class on random weights,
so the gmlx side can check its conventions on the CPU.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

import fixtures
import oracle

SCRIPT = "scripts/image_oracle/check3_flux2_klein.py"
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each block and the final layer, free running "
        "for the time vector, modulations and embeddings and for the "
        "velocity")
MODS = ("double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation")
# The time offsets of encode_image_refs: reference i at REF_T * (1 + i).
REF_T = 10

# The tiny model of --self-test: Flux2Params scaled down, with two
# reference images of different grids.
TINY = {"in_channels": 8, "context_in_dim": 24, "hidden_size": 64,
        "num_heads": 4, "depth": 2, "depth_single_blocks": 2,
        "axes_dim": [4, 4, 4, 4], "theta": 2000, "mlp_ratio": 3.0,
        "use_guidance_embed": False}
TINY_CASE = {"latent": (4, 6), "refs": ((2, 3), (2, 2)), "text": 10,
             "sigma": 0.73}
# The time of the cached step of the tiny model's cached pass.
KV_SIGMA2 = 0.41
# The registry family of a KV file and the tag its fixture names carry.
KV_FAMILY = ("flux2_klein_9b_kv", "_9b_kv")
# The blocks whose reference K and V a KV fixture holds, in the order of
# the blocks: the first and the last double block and single block of 9B.
KV_BLOCKS = (0, 7, 8, 31)
KV_RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2 "
           "of each velocity of the cached pass and of each block's "
           "reference K and V")


def reference(path: Path):
    """The reference's model and sampling modules. sampling imports
    torchvision for its image preparation, which a step does not use and
    the oracle environment lacks, so an empty module stands in."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    from flux2 import model, sampling

    return model, sampling


def gguf_state(torch, path: Path) -> tuple[dict, dict]:
    """The file's tensors by name, quantized and BF16 ones as
    GGUFParameter, with the codec census."""
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
    return out, census


def params_of(fm, state: dict):
    """Klein4BParams or Klein9BParams, by the width of img_in."""
    width = state["img_in.weight"].shape
    for cls in (fm.Klein4BParams, fm.Klein9BParams):
        p = cls()
        if p.hidden_size in width:
            return p
    raise SystemExit(f"img_in.weight {tuple(width)} is no FLUX.2 klein size")


# The registry family of each FLUX.2 klein width, and the tag its fixture
# names carry after the family's name.
FAMILIES = {3072: ("flux2_klein", ""), 4096: ("flux2_klein_9b", "_9b")}


def family_of(params) -> tuple[str, str]:
    """(registry family, fixture name tag) of Klein4BParams or
    Klein9BParams."""
    return FAMILIES[params.hidden_size]


def _embedding_in_dtype(torch, model, dtype):
    # The embedding is bf16, as the reference returns it for a bf16 time;
    # the fp32 model takes it in its own dtype.
    model.time_in.register_forward_pre_hook(
        lambda _m, args, dt=dtype: tuple(a.to(dt) for a in args))


def build(torch, fm, state: dict, mode: str):
    """The reference DiT on ``state``, quantized leaves as GGUFLinear."""
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 _replace_with_gguf_linear)

    dtype = oracle.torch_dtype(mode)
    with torch.device("meta"):
        model = fm.Flux2(params_of(fm, state))
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
    _embedding_in_dtype(torch, model, dtype)
    return model.eval()


def plain_build(torch, fm, params, state: dict, mode: str):
    """The reference DiT on float tensors in the mode's dtype."""
    dtype = oracle.torch_dtype(mode)
    model = fm.Flux2(params)
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()},
                          strict=True)
    model = model.to(dtype)
    _embedding_in_dtype(torch, model, dtype)
    return model.eval()


def bf16(torch, a: np.ndarray) -> np.ndarray:
    return oracle.to_numpy(torch.from_numpy(np.asarray(a, np.float32)).to(
        torch.bfloat16))


def _cat(torch, out):
    """A modulation's chunks side by side, (B, n * dim)."""
    flat = []

    def walk(o):
        if isinstance(o, (tuple, list)):
            for x in o:
                walk(x)
        elif o is not None:
            flat.append(o)
    walk(out)
    y = torch.cat(flat, dim=-1)
    return y[:, 0] if y.dim() == 3 else y


class Capture:
    """The rows of one call by gmlx module path. The forward calls each
    block's forward_kv_extract, past module hooks, so each block's method
    is wrapped on its instance."""

    def __init__(self, torch, model, n_img: int):
        self.torch = torch
        self.rows: dict = {}
        self.inputs: dict = {}
        self.kwargs: dict = {}
        self.undo = []
        self.n_img = n_img
        hook = self._hook
        hook("vec", model.time_in)
        for name in MODS:
            hook(name, getattr(model, name), chunks=True)
        hook("final_layer.adaLN_modulation",
             model.final_layer.adaLN_modulation, chunks=True)
        hook("img_in", model.img_in)
        hook("txt_in", model.txt_in)
        hook("final_layer", model.final_layer, image=True)
        for i, b in enumerate(model.double_blocks):
            self._wrap(f"double_blocks.{i}", b, double=True)
        for i, b in enumerate(model.single_blocks):
            self._wrap(f"single_blocks.{i}", b, double=False)

    def _hook(self, name, mod, *, chunks=False, image=False):
        def keep(_m, args, out):
            if chunks:
                out = _cat(self.torch, out)
            if image:
                out = out[:, :self.n_img]
                self.inputs[name] = tuple(
                    a[:, :self.n_img] if i == 0 else a
                    for i, a in enumerate(args))
            self.rows[name] = out[0] if out.dim() == 3 else out
        h = mod.register_forward_hook(keep)
        self.undo.append(h.remove)

    def _wrap(self, name, blk, double: bool):
        orig = blk.forward_kv_extract

        def run(*args, **kw):
            self.inputs[name] = args
            self.kwargs[name] = kw
            out = orig(*args, **kw)
            if double:
                self.rows[f"{name}.img"] = out[0][0]
                self.rows[f"{name}.txt"] = out[1][0]
            else:
                self.rows[name] = out[0][0]
            return out
        blk.forward_kv_extract = run
        self.undo.append(lambda: delattr(blk, "forward_kv_extract"))

    def remove(self):
        for u in self.undo:
            u()


def case_inputs(torch, sampling, lat: np.ndarray, feats: np.ndarray,
                refs: list, mode: str) -> dict:
    """The tokens and ids of one sample, as the reference's denoise takes
    them, in the mode's dtype."""
    dtype = oracle.torch_dtype(mode)
    x, x_ids = sampling.batched_prc_img(torch.from_numpy(lat).to(
        torch.bfloat16))
    n_img = x.shape[1]
    if refs:
        t_off = [REF_T + REF_T * t for t in torch.arange(0, len(refs))]
        t_off = [t.view(-1) for t in t_off]
        toks, ids = sampling.listed_prc_img(
            [torch.from_numpy(r[0]).to(torch.bfloat16) for r in refs],
            t_coord=t_off)
        x = torch.cat([x, torch.cat(toks)[None]], dim=1)
        x_ids = torch.cat([x_ids, torch.cat(ids)[None]], dim=1)
    ctx, ctx_ids = sampling.batched_prc_txt(
        torch.from_numpy(feats[None]).to(torch.bfloat16))
    return {"x": x.to(dtype), "x_ids": x_ids, "ctx": ctx.to(dtype),
            "ctx_ids": ctx_ids, "n_img": n_img}


def call(torch, sampling, model, lat, feats, refs, sigma, mode, rows: bool):
    ins = case_inputs(torch, sampling, lat, feats, refs, mode)
    n_img = ins["n_img"]
    # denoise fills the time in the dtype of its bf16 latents.
    t_vec = torch.full((1,), sigma, dtype=torch.bfloat16)
    cap = Capture(torch, model, n_img) if rows else None
    emb = []
    h_emb = model.time_in.register_forward_pre_hook(
        lambda _m, args: emb.append(args[0]))
    try:
        out = model(x=ins["x"], x_ids=ins["x_ids"], timesteps=t_vec,
                    ctx=ins["ctx"], ctx_ids=ins["ctx_ids"], guidance=None)
    finally:
        h_emb.remove()
        if cap is not None:
            cap.remove()
    ins["time_in"] = emb[0]
    _, c, h, w = lat.shape
    vel = oracle.to_numpy(out[0, :n_img]).T.reshape(1, c, h, w)
    return vel, cap, ins


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
        if name.startswith("double_blocks."):
            blk = model16.get_submodule(name)
            img, txt, _ = blk.forward_kv_extract(
                *(cast(a) for a in args), **cap32.kwargs[name])
            out[f"{name}.img"] = oracle.to_numpy(img[0])
            out[f"{name}.txt"] = oracle.to_numpy(txt[0])
        elif name.startswith("single_blocks."):
            blk = model16.get_submodule(name)
            y, _ = blk.forward_kv_extract(*(cast(a) for a in args),
                                          **cap32.kwargs[name])
            out[name] = oracle.to_numpy(y[0])
    # The final layer forms its shift and scale from the time vector, so
    # it runs here on the fp32 model's, rounded to bf16, as the blocks
    # take theirs.
    fl = model16.final_layer
    x = cast(cap32.inputs["final_layer"][0])
    mod = cap32.rows["final_layer.adaLN_modulation"].to(torch.bfloat16)
    shift, scale = mod[:, None].chunk(2, dim=-1)
    y = fl.linear((1 + scale) * fl.norm_final(x) + shift)
    out["final_layer"] = oracle.to_numpy(y[0])
    return out


def rows_of(cap: Capture) -> dict:
    return {k: oracle.to_numpy(v) for k, v in cap.rows.items()}


def run_case(torch, sampling, models: dict, lat, feats, refs, sigma,
             rows: bool, name: str, inputs: dict, files: dict, notes: list,
             *, check: int = 3, script: str = SCRIPT, more=None,
             family: str = "flux2_klein") -> dict:
    """One step in fp32 and bf16 written as fixture ``name`` of
    ``family``. ``more`` takes the writer and the fp32 velocity before the
    write. Returns the velocity of each mode."""
    fx = fixtures.FixtureWriter(name, check=check, family=family,
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
    vel, caps, times = {}, {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        vel[mode], caps[mode], ins = call(
            torch, sampling, models[mode], lat, feats, refs, sigma, mode,
            rows)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
    oracle.emulate_bf16_matmul(False)
    fx.add("ids.image", oracle.to_numpy(ins["x_ids"][0]))
    fx.add("ids.text", oracle.to_numpy(ins["ctx_ids"][0]))
    # The reference's embedding of the bf16 time: its fp32 exp can be an
    # ulp off the correctly rounded one, which can move one bf16 entry.
    fx.add("in.time_in", oracle.to_numpy(ins["time_in"]))
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
        ins32 = caps["fp32"].inputs
        fx.add("in.single_blocks.0", oracle.to_numpy(
            ins32["single_blocks.0"][0][0]))
        fx.add("in.final_layer", oracle.to_numpy(ins32["final_layer"][0][0]))
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
    if more is not None:
        more(fx, vel["fp32"])
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 "
          f"{gap['rel_l2']:.3e}, {times}, wrote {path}", flush=True)
    return vel


def seeded(torch, rng, shape, std=1.0) -> np.ndarray:
    return bf16(torch, rng.standard_normal(shape) * std)


# The self-test.

def tiny_state(torch, fm, params, seed: int) -> dict:
    """Random tiny weights by name, float32 numpy in torch order: norm
    scales near one, matrices at 1 / sqrt(fan in). The key norm scales
    are then 1.5 times their draw at even indices and 1 / 1.5 times at odd
    ones. A swap of the query and key scales changes the logits only
    through the cross terms of each rotary pair, so the key scales differ
    from the query scales inside each pair."""
    with torch.device("meta"):
        model = fm.Flux2(params)
    rng = np.random.default_rng(seed)
    out = {}
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if name.endswith(".scale"):
            v = 1 + rng.standard_normal(shape) * 0.2
            if name.endswith("key_norm.scale"):
                v = v * np.where(np.arange(shape[-1]) % 2 == 0, 1.5, 1 / 1.5)
        else:
            v = rng.standard_normal(shape) / np.sqrt(shape[-1])
        out[name] = v.astype(np.float32)
    return out


def tiny_config() -> dict:
    """The tiny model in the keys of the official transformer config."""
    t = TINY
    return {"attention_head_dim": t["hidden_size"] // t["num_heads"],
            "axes_dims_rope": list(t["axes_dim"]), "eps": 1e-06,
            "guidance_embeds": False, "in_channels": t["in_channels"],
            "joint_attention_dim": t["context_in_dim"],
            "mlp_ratio": t["mlp_ratio"], "num_attention_heads": t["num_heads"],
            "num_layers": t["depth"],
            "num_single_layers": t["depth_single_blocks"],
            "out_channels": None, "patch_size": 1, "rope_theta": t["theta"],
            "timestep_guidance_channels": 256}


def self_test(torch, fm, sampling, commit: str, a) -> int:
    params = fm.Flux2Params(**TINY)
    state = tiny_state(torch, fm, params, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    c = TINY["in_channels"]
    h, w = TINY_CASE["latent"]
    lat = seeded(torch, rng, (1, c, h, w))
    feats = seeded(torch, rng, (TINY_CASE["text"], TINY["context_in_dim"]),
                   2.0)
    refs = [seeded(torch, rng, (1, c, rh, rw))
            for rh, rw in TINY_CASE["refs"]]
    fx_inputs = {"config": tiny_config(), "case": TINY_CASE, "seed": a.seed,
                 "reference_commit": commit}
    st = {k: torch.from_numpy(v) for k, v in state.items()}
    models = {m: plain_build(torch, fm, params, st, m)
              for m in ("fp32", "bf16")}
    for with_refs, name in ((True, "selftest_flux2_klein_tiny"),
                            (False, "selftest_flux2_klein_tiny_norefs")):
        run_case(torch, sampling, models, lat, feats,
                 refs if with_refs else [], TINY_CASE["sigma"], True, name,
                 dict(fx_inputs, refs=len(refs) if with_refs else 0), {},
                 ["the weights are in selftest_flux2_klein_tiny_weights by "
                  "GGUF name in torch order"])
    wx = fixtures.FixtureWriter("selftest_flux2_klein_tiny_weights", check=3,
                                family="flux2_klein", script=SCRIPT)
    wx.inputs.update(fx_inputs)
    for g, v in state.items():
        wx.add(g, v)
    wx.write()
    rope_fixture(torch, fm, commit)
    return 0


def kv_call(torch, sampling, model, lat, feats, refs, sigma, mode,
            cache=None) -> tuple:
    """One step of the reference's denoise_cached: forward_kv_extract with
    the reference tokens, which returns the cache, or forward_kv_cached on
    ``cache``. Returns the velocity, the cache and the inputs."""
    dtype = oracle.torch_dtype(mode)
    ins = case_inputs(torch, sampling, lat, feats, [], mode)
    t_vec = torch.full((1,), sigma, dtype=torch.bfloat16)
    common = dict(x=ins["x"], x_ids=ins["x_ids"], timesteps=t_vec,
                  ctx=ins["ctx"], ctx_ids=ins["ctx_ids"], guidance=None)
    # The first embedding time_in takes is the step's, the second, in the
    # extract, that of the references' time zero.
    emb = []
    hook = model.time_in.register_forward_pre_hook(
        lambda _m, args: emb.append(args[0]))
    try:
        if cache is None:
            t_off = [(REF_T + REF_T * t).view(-1)
                     for t in torch.arange(0, len(refs))]
            toks, ids = sampling.listed_prc_img(
                [torch.from_numpy(r[0]).to(torch.bfloat16) for r in refs],
                t_coord=t_off)
            out, cache = model.forward_kv_extract(
                **common, x_seq_concat=torch.cat(toks)[None].to(dtype),
                x_seq_concat_ids=torch.cat(ids)[None])
        else:
            out = model.forward_kv_cached(**common, kv_cache=cache)
    finally:
        hook.remove()
    ins["time_in"] = emb[0]
    _, c, h, w = lat.shape
    return oracle.to_numpy(out[0]).T.reshape(1, c, h, w), cache, ins


def kv_cache_arrays(cache: dict, keep=None) -> dict:
    """The cache's K and V by block, in the order of the blocks:
    ``kv.<i>.k`` and ``kv.<i>.v``, (1, heads, R, head dim), of the blocks
    in ``keep``, or of every block."""
    blocks = [*cache["double_blocks"], *cache["single_blocks"]]
    out = {}
    for i, b in enumerate(blocks):
        if keep is not None and i not in keep:
            continue
        out[f"kv.{i}.k"] = oracle.to_numpy(b["k_ref"])
        out[f"kv.{i}.v"] = oracle.to_numpy(b["v_ref"])
    return out


def kv_case(torch, sampling, models: dict, lat, lat2, feats, refs, sigma,
            sigma2, rows: bool, name: str, inputs: dict, files: dict,
            family: str) -> None:
    """The reference's cached pass in fp32 and bf16 written as fixture
    ``name`` of ``family``: the extract step at ``sigma`` on ``lat`` with
    the references, then a cached step at ``sigma2`` on ``lat2``. With
    ``rows`` the fixture holds the reference K and V of KV_BLOCKS."""
    fx = fixtures.FixtureWriter(name, check=3, family=family, script=SCRIPT)
    fx.inputs.update(inputs)
    for role, path in files.items():
        fx.add_file(role, path)
    fx.add("latents", lat)
    fx.add("latents2", lat2)
    fx.add("features", feats)
    for i, r in enumerate(refs):
        fx.add(f"refs.{i}", r)
    fx.add("sigma", np.float32(sigma))
    fx.add("sigma2", np.float32(sigma2))
    kv, times = {}, {}
    for mode in ("fp32", "bf16"):
        t0 = time.perf_counter()
        oracle.emulate_bf16_matmul(mode == "bf16")
        try:
            v0, cache, ins0 = kv_call(torch, sampling, models[mode], lat,
                                      feats, refs, sigma, mode)
            v1, _, ins1 = kv_call(torch, sampling, models[mode], lat2,
                                  feats, refs, sigma2, mode, cache)
        finally:
            oracle.emulate_bf16_matmul(False)
        times[f"{mode}_s"] = round(time.perf_counter() - t0, 1)
        tag = "" if mode == "fp32" else "_bf16"
        fx.add(f"velocity_extract{tag}", v0)
        fx.add(f"velocity_cached{tag}", v1)
        if rows:
            kv[mode] = kv_cache_arrays(cache, KV_BLOCKS)
        if mode == "fp32":
            fx.add("in.time_in", oracle.to_numpy(ins0["time_in"]))
            fx.add("in.time_in2", oracle.to_numpy(ins1["time_in"]))
        del cache
    gaps = {}
    for k, a in kv.get("fp32", {}).items():
        fx.add(k, a)
        gaps[k] = fixtures.compare(kv["bf16"][k], a)
        fx.tolerances[k] = {"rel_l2": 2 * gaps[k]["rel_l2"]}
    if gaps:
        fx.results["kv_bf16_vs_fp32"] = gaps
    for step in ("extract", "cached"):
        gap = fixtures.compare(fx.arrays[f"velocity_{step}_bf16"],
                               fx.arrays[f"velocity_{step}"])
        fx.results[f"velocity_{step}_bf16_vs_fp32"] = gap
        fx.tolerances[f"velocity_{step}"] = {
            "rel_l2": 2 * gap["rel_l2"],
            "cosine_min": 1 - 4 * (1 - gap["cosine"])}
    fx.results["times"] = times
    fx.tolerances["rule"] = KV_RULE
    path = fx.write()
    print(f"[check3] {name}: velocity bf16 vs fp32 rel L2 extract "
          f"{fx.results['velocity_extract_bf16_vs_fp32']['rel_l2']:.3e}, "
          f"cached {fx.results['velocity_cached_bf16_vs_fp32']['rel_l2']:.3e}"
          f", {times}, wrote {path}", flush=True)


def kv_self_test(torch, fm, sampling, commit: str, a) -> int:
    """The tiny model of --self-test through the reference's cached pass:
    the extract step at the case's time with both reference images, and a
    cached step on other latents at another time, in fp32 and bf16."""
    params = fm.Flux2Params(**TINY)
    state = tiny_state(torch, fm, params, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    c = TINY["in_channels"]
    h, w = TINY_CASE["latent"]
    lat = seeded(torch, rng, (1, c, h, w))
    feats = seeded(torch, rng, (TINY_CASE["text"], TINY["context_in_dim"]),
                   2.0)
    refs = [seeded(torch, rng, (1, c, rh, rw))
            for rh, rw in TINY_CASE["refs"]]
    lat2 = seeded(torch, np.random.default_rng(a.seed + 2), (1, c, h, w))
    sigma, sigma2 = TINY_CASE["sigma"], KV_SIGMA2
    st = {k: torch.from_numpy(v) for k, v in state.items()}
    fx = fixtures.FixtureWriter("selftest_flux2_klein_tiny_kv", check=3,
                                family="flux2_klein", script=SCRIPT)
    fx.inputs.update({"config": tiny_config(), "case": TINY_CASE,
                      "seed": a.seed, "reference_commit": commit,
                      "refs": len(refs), "sigma2": sigma2})
    fx.note("the weights are in selftest_flux2_klein_tiny_weights by GGUF "
            "name in torch order; the latents, features and references are "
            "those of selftest_flux2_klein_tiny")
    fx.add("latents", lat)
    fx.add("latents2", lat2)
    fx.add("features", feats)
    for i, r in enumerate(refs):
        fx.add(f"refs.{i}", r)
    fx.add("sigma", np.float32(sigma))
    fx.add("sigma2", np.float32(sigma2))
    for mode in ("fp32", "bf16"):
        model = plain_build(torch, fm, params, st, mode)
        oracle.emulate_bf16_matmul(mode == "bf16")
        v0, cache, ins0 = kv_call(torch, sampling, model, lat, feats, refs,
                                  sigma, mode)
        v1, _, ins1 = kv_call(torch, sampling, model, lat2, feats, refs,
                              sigma2, mode, cache)
        oracle.emulate_bf16_matmul(False)
        tag = "" if mode == "fp32" else "_bf16"
        fx.add(f"velocity_extract{tag}", v0)
        fx.add(f"velocity_cached{tag}", v1)
        if mode == "fp32":
            fx.add("in.time_in", oracle.to_numpy(ins0["time_in"]))
            fx.add("in.time_in2", oracle.to_numpy(ins1["time_in"]))
            for k, v in kv_cache_arrays(cache).items():
                fx.add(k, v)
    for name in ("extract", "cached"):
        gap = fixtures.compare(fx.arrays[f"velocity_{name}_bf16"],
                               fx.arrays[f"velocity_{name}"])
        fx.results[f"velocity_{name}_bf16_vs_fp32"] = gap
        fx.tolerances[f"velocity_{name}"] = {"rel_l2": 2 * gap["rel_l2"]}
    fx.tolerances["rule"] = RULE
    print(f"[check3] wrote {fx.write()}", flush=True)
    return 0


def rope_fixture(torch, fm, commit: str) -> None:
    """The reference's RoPE tables of a klein head, four axes of 32 at
    theta 2000, at the positions of small grids and up to 4095, where an
    angle formed in float64 differs from the reference's fp32 one."""
    pos = np.concatenate([np.arange(64), [127, 128, 255, 256, 511, 1000,
                                          1023, 2047, 2048, 3000, 4095]])
    ids = np.stack([pos, pos[::-1], np.roll(pos, 7), pos], axis=1)
    emb = fm.EmbedND(dim=128, theta=2000, axes_dim=[32] * 4)(
        torch.from_numpy(ids.astype(np.int64))[None])
    fx = fixtures.FixtureWriter("selftest_flux2_klein_rope", check=3,
                                family="flux2_klein", script=SCRIPT)
    fx.inputs.update({"axes_dims": [32] * 4, "theta": 2000,
                      "reference_commit": commit,
                      "ids": "int64, as prc_img and prc_txt give them"})
    fx.add("ids", ids.astype(np.int64))
    fx.add("cos", oracle.to_numpy(emb[0, 0, :, :, 0, 0]))
    fx.add("sin", oracle.to_numpy(emb[0, 0, :, :, 1, 0]))
    fx.write()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux2")
    ap.add_argument("--reference-commit", required=True,
                    help="the commit of --reference, for the manifest")
    ap.add_argument("--dit", type=Path, help="FLUX.2 klein DiT GGUF")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--rows", type=int, nargs="*", default=[512],
                    help="sizes whose fixture holds every row")
    ap.add_argument("--refs", type=int, default=0,
                    help="reference images, each of --ref-size")
    ap.add_argument("--ref-size", type=int, default=0,
                    help="pixels of a reference side, 0 for the size")
    ap.add_argument("--text-len", type=int, default=512)
    ap.add_argument("--feature-std", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=4,
                    help="steps of the schedule --step reads")
    ap.add_argument("--step", type=int, default=0,
                    help="step of the schedule whose time is the image's")
    ap.add_argument("--sigma", type=float,
                    help="the image's time, the time of --step when unset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", help="fixture name before the tags, "
                    "check3_flux2_klein or check3_flux2_klein_9b by the "
                    "width when unset")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--kv", action="store_true",
                    help="--dit is FLUX.2 klein 9B KV, whose cases with "
                    "references run the reference's cached pass, "
                    "denoise_cached; with --self-test, that pass of the "
                    "tiny model alone")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    fm, sampling = reference(a.reference.expanduser())
    commit = a.reference_commit
    if a.self_test and a.kv:
        return kv_self_test(torch, fm, sampling, commit, a)
    if a.self_test:
        return self_test(torch, fm, sampling, commit, a)
    if a.dit is None:
        ap.error("--dit is required")
    t0 = time.perf_counter()
    state, census = gguf_state(torch, a.dit)
    models = {m: build(torch, fm, state, m) for m in ("fp32", "bf16")}
    params = params_of(fm, state)
    family, size_tag = family_of(params)
    if a.kv:
        if size_tag != "_9b":
            raise SystemExit("--kv takes a FLUX.2 klein 9B KV file")
        family, size_tag = KV_FAMILY
    base_name = a.name or f"check3_flux2_klein{size_tag}"
    print(f"[check3] loaded {a.dit.name} in "
          f"{time.perf_counter() - t0:.1f}s, codecs {census}, width "
          f"{params.hidden_size}", flush=True)
    rng = np.random.default_rng(a.seed)
    feats = seeded(torch, rng, (a.text_len, params.context_in_dim),
                   a.feature_std)
    c = params.in_channels
    for size in a.sizes:
        n = size // 16
        schedule = sampling.get_schedule(a.steps, n * n)
        sigma = float(schedule[a.step]) if a.sigma is None else a.sigma
        srng = np.random.default_rng(a.seed + size)
        lat = seeded(torch, srng, (1, c, n, n))
        rn = (a.ref_size or size) // 16
        refs = [seeded(torch, srng, (1, c, rn, rn)) for _ in range(a.refs)]
        inputs = {"size": size, "sigma": sigma, "step": a.step,
                  "schedule": f"get_schedule({a.steps}, {n * n})",
                  "seed": a.seed + size,
                  "latents": "seeded normal, rounded to bf16",
                  "features": f"seeded normal, std {a.feature_std}, "
                              "rounded to bf16",
                  "text_rows": int(feats.shape[0]), "refs": a.refs,
                  "ref_size": rn * 16 if a.refs else 0,
                  "dequant": oracle.dequant_mode(),
                  "matmul_bf16": "bf16 operands, float32 GEMM, one bf16 "
                                 "rounding",
                  "timestep": "bf16 as the reference's denoise fills it, in "
                              "both modes",
                  "reference_commit": commit, "gguf_codecs": census}
        tag = f"_r{a.refs}" if a.refs else ""
        if a.kv and refs:
            # The cached step reads other latents at the next step's time.
            sigma2 = float(schedule[a.step + 1])
            lat2 = seeded(torch, srng, (1, c, n, n))
            inputs.update(
                sigma2=sigma2, step2=a.step + 1,
                latents2="seeded normal after the references, rounded to "
                         "bf16",
                kv_blocks=list(KV_BLOCKS) if size in a.rows else [],
                kv_pass="denoise_cached: forward_kv_extract at sigma on "
                        "latents with the references, then "
                        "forward_kv_cached at sigma2 on latents2")
            kv_case(torch, sampling, models, lat, lat2, feats, refs, sigma,
                    sigma2, size in a.rows, f"{base_name}{tag}_{size}",
                    inputs, {"dit": a.dit}, family)
            continue
        run_case(torch, sampling, models, lat, feats, refs, sigma,
                 size in a.rows, f"{base_name}{tag}_{size}", inputs,
                 {"dit": a.dit}, [], family=family)
    return 0


if __name__ == "__main__":
    sys.exit(main())
