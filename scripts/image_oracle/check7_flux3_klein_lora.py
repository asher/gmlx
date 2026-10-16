#!/usr/bin/env python3
"""Check 7 oracle for FLUX.3 klein, step half: check 3's step at 512 with
a synthetic LoRA adapter applied to BFL's reference DiT.

The adapter is written here from --synthetic SEED: one pair on every
linear layer of a few blocks of each mode and of the single stream, and
on a few layers outside the blocks, the time, target time and modulation
layers among them, at ranks 4 to 16, each with an alpha of half its
rank, its down rows at unit spread over the input and its up rows at
--gain over the rank. The default gain moves a leaf's output by a few
percent: a delta under a leaf's bf16 rounding is one the site gate
cannot see. The names are BFL's module names, which the GGUF keeps, and
the fixture holds the pairs as ``down.<target>`` and ``up.<target>``, so
the gmlx side writes the same adapter in each dialect.

Each pair is applied as PEFT applies it when it leaves the weights
unmerged: a forward hook on the target layer adds

    strength * (alpha / rank) * linear(linear(x, down), up)

to the layer's output, in the mode's dtype. In bf16 the two products run
as the base layers do, as float32 products over bfloat16 operands with
one bfloat16 rounding.

The step is check 3's at 512: the latents, the features, the reference
latents, the sigma and the target of the --source fixture, every row in
fp32 and the bf16 gaps that set the tolerances, with check 3's rule, so
the gmlx side reads the model under a patch table with the rows it reads
the base model with. The manifest also records the adapter's effect:
the relative L2 of each fp32 row with the adapter against the same row
of the source fixture.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check7_flux3_klein_lora.py --reference ~/src/flux-3-image-klein/src \\
      --dit flux3_image_klein_sd-Q4_K_M.gguf --synthetic 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import check3_flux3_klein as c3
import fixtures
import oracle

NAME = "check7_flux3_klein_synthetic_512"
SOURCE = "check3_flux3_klein_r1_512"
SCRIPT = "scripts/image_oracle/check7_flux3_klein_lora.py"

# The layers of the synthetic adapter: every linear layer of these
# blocks, and these layers outside the blocks.
SYNTHETIC_BLOCKS = {"content_mode_blocks.image": (0, 4),
                    "content_mode_blocks.image_cond": (2,),
                    "txt_mode_blocks": (0, 4),
                    "single_blocks": (0, 13, 27)}
SYNTHETIC_TOP = ("emb_in.image", "emb_in.image_cond", "txt_in",
                 "time_in.out_layer", "ref_time_in.in_layer",
                 "early_stream_modulations.image.lin",
                 "single_stream_modulations.txt.lin",
                 "final_layer.image.adaLN_modulation.1",
                 "final_layer.image.linear")
LEAVES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")


def synthetic_pairs(torch, model, seed: int, gain: float) -> dict:
    """Module name to (down, up, alpha) of the synthetic adapter, in
    float32 on the layers that the model has."""
    names = [f"{g}.{i}.{leaf}" for g, idx in SYNTHETIC_BLOCKS.items()
             for i in idx for leaf in LEAVES]
    names += list(SYNTHETIC_TOP)
    have = dict(model.named_modules())
    rng = np.random.default_rng(seed)
    out = {}
    for j, name in enumerate(names):
        if name not in have:
            continue
        lin = have[name]
        rank = 4 * (1 + j % 4)
        down = rng.standard_normal((rank, lin.in_features)) \
            / np.sqrt(lin.in_features)
        up = rng.standard_normal((lin.out_features, rank)) * gain \
            / np.sqrt(rank)
        out[name] = (torch.from_numpy(down.astype(np.float32)),
                     torch.from_numpy(up.astype(np.float32)), rank / 2)
    return out


def install(torch, model, pairs: dict, strength: float, mode: str) -> list:
    """A forward hook per target that adds its pair's product."""
    dtype = oracle.torch_dtype(mode)
    handles = []
    for name, (down, up, alpha) in pairs.items():
        mod = model.get_submodule(name)
        scale = strength * (1.0 if alpha is None else alpha / down.shape[0])
        d, u = down.to(dtype), up.to(dtype)

        def hook(_mod, args, out, d=d, u=u, scale=scale):
            import torch.nn.functional as F

            x = args[0].to(dtype)
            return out + (F.linear(F.linear(x, d), u) * scale).to(out.dtype)

        handles.append(mod.register_forward_hook(hook))
    return handles


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux-3-image-klein")
    ap.add_argument("--dit", required=True, type=Path,
                    help="FLUX.3 klein DiT GGUF of the source fixture")
    ap.add_argument("--synthetic", type=int, metavar="SEED", required=True,
                    help="seed of the synthetic adapter")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="spread of the synthetic up rows over the rank")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--source", default=SOURCE,
                    help="check 3 fixture whose inputs the step reads")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    dit, pipeline, _, commit = c3.reference(a.reference.expanduser())
    root = fixtures.fixture_root()
    m3 = fixtures.read_manifest(a.source, root=root)
    if m3["files"]["dit"]["name"] != a.dit.name:
        raise SystemExit(f"error: {a.source} read {m3['files']['dit']}")
    if m3["inputs"]["reference_commit"] != commit:
        raise SystemExit(f"error: {a.source} read the reference at "
                         f"{m3['inputs']['reference_commit']}, not {commit}")
    with np.load(root / m3["npz"], allow_pickle=False) as z:
        src = {k: z[k] for k in z.files}
    state, census, flow_map = c3.gguf_state(torch, a.dit)
    models = {m: c3.build(torch, dit, state, flow_map, m)
              for m in ("fp32", "bf16")}
    pairs = synthetic_pairs(torch, models["fp32"], a.synthetic, a.gain)
    for mode, model in models.items():
        install(torch, model, pairs, a.strength, mode)
    refs = [src[f"refs.{i}"] for i in range(m3["inputs"]["refs"])]
    target = float(src["target"]) if "target" in src else None
    ranks = sorted({int(d.shape[0]) for d, _, _ in pairs.values()})
    inputs = dict(m3["inputs"])
    inputs.update(source=a.source, strength=a.strength,
                  lora_pairs=len(pairs), lora_ranks=ranks,
                  lora_alphas=sorted({al for _, _, al in pairs.values()}),
                  lora_source=f"synthetic, seed {a.synthetic}, gain "
                  f"{a.gain}, ranks 4 to 16, alpha half the rank",
                  targets=sorted(pairs),
                  application="unmerged: strength * (alpha / rank) * "
                  "up(down(x)) added to each target's output in the "
                  "mode's dtype")

    def effect(fx, _vel):
        for name, (down, up, _) in pairs.items():
            fx.add(f"down.{name}", down.numpy())
            fx.add(f"up.{name}", up.numpy())
        fx.inputs["lora_pair_alphas"] = {
            name: al for name, (_, _, al) in pairs.items()}
        fx.results["adapter_effect"] = {
            k: fixtures.rel_l2(fx.arrays[k], src[k])
            for k in fx.arrays if k in src and k not in c3_inputs(src)}

    c3.run_case(torch, pipeline, models, src["latents"], src["features"],
                refs, float(src["sigma"]), target, True, a.name, inputs,
                {"dit": a.dit}, [
                    "the adapter is applied unmerged as PEFT applies it; "
                    "the gmlx side runs it under a patch table"],
                check=7, script=SCRIPT, more=effect)
    return 0


def c3_inputs(src: dict) -> set:
    """The arrays of the source fixture that are inputs of the step."""
    return {k for k in src if k in ("latents", "features", "sigma",
                                    "target", "velocity_bf16")
            or k.startswith(("refs.", "ids.", "in."))}


if __name__ == "__main__":
    sys.exit(main())
