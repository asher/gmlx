#!/usr/bin/env python3
"""Check 7 oracle for MOSS-Image, step half: check 3's step at 512 with a
synthetic LoRA applied.

No community MOSS-Image adapter is known, so the script writes one from
a seed: one pair on every linear layer of three blocks and of a text
refiner block, the fused ``proj.ffn_up_gate`` and ``qkv`` among them,
and on six layers outside the blocks, at ranks 4 to 16, each with an
alpha of half its rank and random factors scaled so that its delta has
--ratio of the Frobenius norm of its dequantized weight. The default
moves a leaf by about 8 percent, as a published LoRA moves its leaves by
5 to 15: one fixed spread of the factors would move the head, whose
weights are small, by most of its output. The release's MOSSImageTransformer
names its layers as the GGUF does, so the pairs carry those names. PEFT
is not in the oracle environment, so the script applies each pair as
PEFT does when it leaves the weights unmerged: a forward hook on the
target layer adds

    strength * (alpha / rank) * linear(linear(x, down), up)

to the layer's output, in the mode's dtype. In bf16 the two products run
as the base layers do, as float32 products over bfloat16 operands with
one bfloat16 rounding.

The step is check 3's at 512: the latents, the features and the sigma of
the --source fixture, every row in fp32 and the bf16 gaps that set the
tolerances, with check 3's rule, so the gmlx side reads the model under a
patch table with the rows it reads the base model with. The fixture
holds the pairs as ``down.<target>`` and ``up.<target>``, so the gmlx
side writes the same adapter in each dialect, and the relative L2 of
each fp32 row against the same row of the source fixture.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check7_moss_image_lora.py --pipeline <dir>/pipeline_moss_image.py \\
      --dit MOSS-Image-SFT-Q4_K_M.gguf --config <dir>/transformer/config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import check3_moss_image as c3
import fixtures
import oracle

NAME = "check7_moss_image_synthetic_512"
SOURCE = "check3_moss_image_512"
SCRIPT = "scripts/image_oracle/check7_moss_image_lora.py"
# The layers of the synthetic adapter: every linear layer of these
# blocks and text refiner blocks, and these layers outside the blocks.
BLOCKS = (0, 21, 41)
REFINERS = (1,)
TOP = ("video_embedder", "text_embedder.proj", "time_embedding.2",
       "shared_block_modulation", "video_head_modulation",
       "final_linear_video")
_BLOCK = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate",
          "proj.ffn_down")
_REFINER = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")


def synthetic_pairs(torch, model, state: dict, seed: int,
                    ratio: float) -> dict:
    """Module name to (down, up, alpha) of the synthetic adapter, in
    float32, each delta at ``ratio`` of its weight's Frobenius norm."""
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 dequantize_gguf_tensor)

    names = [f"layers.{i}.{leaf}" for i in BLOCKS for leaf in _BLOCK]
    names += [f"text_embedder.blocks.{i}.{leaf}" for i in REFINERS
              for leaf in _REFINER]
    names += list(TOP)
    have = dict(model.named_modules())
    rng = np.random.default_rng(seed)
    out = {}
    for j, name in enumerate(names):
        lin = have[name]
        rank = 4 * (1 + j % 4)
        alpha = rank / 2
        w = state[f"{name}.weight"]
        if isinstance(w, GGUFParameter):
            w = dequantize_gguf_tensor(w)
        norm = float(w.to(torch.float64).norm())
        down = rng.standard_normal((rank, lin.in_features)) \
            / np.sqrt(lin.in_features)
        up = rng.standard_normal((lin.out_features, rank)) / np.sqrt(rank)
        delta = np.linalg.norm(alpha / rank * (up @ down))
        up *= ratio * norm / delta
        out[name] = (torch.from_numpy(down.astype(np.float32)),
                     torch.from_numpy(up.astype(np.float32)), alpha)
    return out


def install(torch, model, pairs: dict, strength: float, mode: str) -> list:
    """A forward hook per target that adds its pair's product."""
    import torch.nn.functional as F

    dtype = oracle.torch_dtype(mode)
    handles = []
    for name, (down, up, alpha) in pairs.items():
        mod = model.get_submodule(name)
        scale = strength * alpha / down.shape[0]
        d, u = down.to(dtype), up.to(dtype)

        def hook(_mod, args, out, d=d, u=u, scale=scale):
            x = args[0].to(dtype)
            return out + (F.linear(F.linear(x, d), u) * scale).to(out.dtype)

        handles.append(mod.register_forward_hook(hook))
    return handles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipeline", type=Path, required=True,
                    help="the release's pipeline_moss_image.py")
    ap.add_argument("--dit", type=Path, required=True,
                    help="MOSS-Image DiT GGUF of the source fixture")
    ap.add_argument("--config", type=Path, required=True,
                    help="the release's transformer/config.json")
    ap.add_argument("--seed", type=int, default=7,
                    help="seed of the synthetic adapter")
    ap.add_argument("--ratio", type=float, default=0.08,
                    help="Frobenius norm of each delta over its weight's")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--source", default=SOURCE,
                    help="check 3 fixture whose inputs the step reads")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    rel = c3.release(a.pipeline)
    root = fixtures.fixture_root()
    m3 = fixtures.read_manifest(a.source, root=root)
    if m3["files"]["dit"]["name"] != a.dit.name:
        raise SystemExit(f"error: {a.source} read {m3['files']['dit']}")
    with np.load(root / m3["npz"], allow_pickle=False) as z:
        src = {k: z[k] for k in z.files}
    cfg = {k: v for k, v in json.loads(a.config.read_text()).items()
           if not k.startswith("_")}
    state, census, _ = c3.gguf_state(torch, a.dit, None)
    models = {m: c3.build(torch, rel, state, cfg, m)
              for m in ("fp32", "bf16")}
    pairs = synthetic_pairs(torch, models["fp32"], state, a.seed, a.ratio)
    for mode, model in models.items():
        install(torch, model, pairs, a.strength, mode)
    inputs = dict(m3["inputs"])
    inputs.update(
        source=a.source, strength=a.strength, lora_pairs=len(pairs),
        lora_ranks=sorted({int(d.shape[0]) for d, _, _ in pairs.values()}),
        lora_alphas=sorted({al for _, _, al in pairs.values()}),
        lora_source=f"synthetic, seed {a.seed}, each delta at {a.ratio} of "
                    "its weight's Frobenius norm, ranks 4 to 16, alpha "
                    "half the rank",
        targets=sorted(pairs),
        lora_pair_alphas={n: al for n, (_, _, al) in pairs.items()},
        application="unmerged: strength * (alpha / rank) * up(down(x)) "
                    "added to each target's output in the mode's dtype")

    def more(fx, _vel):
        for name, (down, up, _) in pairs.items():
            fx.add(f"down.{name}", down.numpy())
            fx.add(f"up.{name}", up.numpy())
        fx.results["adapter_effect"] = {
            k: fixtures.rel_l2(fx.arrays[k], src[k])
            for k in fx.arrays if k in src and k not in (
                "latents", "features", "sigma", "timestep",
                "velocity_bf16") and not k.startswith("in.")}

    c3.run_case(torch, models, src["latents"], src["features"],
                float(src["sigma"]), True, a.name, inputs,
                {"dit": a.dit, "pipeline": a.pipeline, "config": a.config},
                ["the adapter is applied unmerged as PEFT applies it; the "
                 "gmlx side runs it under a patch table"],
                check=7, script=SCRIPT, more=more)
    return 0


if __name__ == "__main__":
    sys.exit(main())
