#!/usr/bin/env python3
"""Check 7 oracle for Krea 2, step half: check 3's step at 512 with a LoRA
file applied.

Krea2Pipeline's own lora_state_dict reads the file, which turns the
reference trainer's and AI-Toolkit's names into the diffusers names of
Krea2Transformer2DModel and keeps a file already in those names. PEFT is
not in the oracle environment, so the script applies each pair as PEFT
does when it leaves the weights unmerged: a forward hook on the target
layer adds

    strength * (alpha / rank) * linear(linear(x, down), up)

to the layer's output, in the mode's dtype, where a missing alpha gives
the factor 1.0. In bf16 the two products run as the base layers do, as
float32 products over bfloat16 operands with one bfloat16 rounding.

The step is check 3's at 512: the latents, the features and the sigma of
the --source fixture, every row in fp32 and the bf16 gaps that set the
tolerances, with check 3's rule, so the gmlx side reads the model under a
patch table with the rows it reads the base model with. The manifest also
records the adapter's effect: the relative L2 of each fp32 row with the
adapter against the same row of the source fixture.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check7_krea2_lora.py --dit Krea-2-Turbo-Q4_K_M.gguf \\
      --lora krea2_kidsdrawing.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import check3_krea2 as c3
import fixtures
import oracle

NAME = "check7_krea2_lora_512"
SOURCE = "check3_krea2_512"
SCRIPT = "scripts/image_oracle/check7_krea2_lora.py"


def read_pairs(torch, path: Path) -> dict:
    """Module name to (down, up, alpha) in float32, as the pipeline's
    loader names the modules."""
    from diffusers import Krea2Pipeline

    sd = Krea2Pipeline.lora_state_dict(str(path))
    pairs: dict = {}
    for k, v in sd.items():
        name = k.removeprefix("transformer.")
        for part, suffix in (("down", ".lora_A.weight"),
                             ("up", ".lora_B.weight"), ("alpha", ".alpha")):
            if name.endswith(suffix):
                pairs.setdefault(name[:-len(suffix)], {})[part] = \
                    v.to(torch.float32)
                break
        else:
            raise SystemExit(f"error: {k} is not a LoRA pair key")
    out = {}
    for m, p in pairs.items():
        if set(p) - {"alpha"} != {"down", "up"}:
            raise SystemExit(f"error: {m} has {sorted(p)}")
        alpha = p.get("alpha")
        out[m] = (p["down"], p["up"],
                  None if alpha is None else float(alpha))
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
    ap.add_argument("--dit", required=True, type=Path,
                    help="Krea 2 DiT GGUF of the source fixture")
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA safetensors file")
    ap.add_argument("--lora-source", default="",
                    help="repository and revision of --lora, for the "
                    "manifest")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--source", default=SOURCE,
                    help="check 3 fixture whose inputs the step reads")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    import diffusers

    root = fixtures.fixture_root()
    m3 = fixtures.read_manifest(a.source, root=root)
    if m3["files"]["dit"]["name"] != a.dit.name:
        raise SystemExit(f"error: {a.source} read {m3['files']['dit']}")
    with np.load(root / m3["npz"], allow_pickle=False) as z:
        src = {k: z[k] for k in z.files}
    state, census = c3.gguf_state(torch, a.dit, None)
    cfg = dict(c3.inspect_defaults(diffusers))
    models = {m: c3.build(torch, state, cfg, m) for m in ("fp32", "bf16")}
    pairs = read_pairs(torch, a.lora)
    for mode, model in models.items():
        install(torch, model, pairs, a.strength, mode)
    ranks = sorted({int(d.shape[0]) for d, _, _ in pairs.values()})
    alphas = sorted({al for _, _, al in pairs.values() if al is not None})
    inputs = {k: v for k, v in m3["inputs"].items()}
    inputs.update(source=a.source, strength=a.strength,
                  lora_pairs=len(pairs), lora_ranks=ranks,
                  lora_alphas=alphas, lora_source=a.lora_source,
                  targets=sorted(pairs),
                  application="unmerged: strength * (alpha / rank) * "
                  "up(down(x)) added to each target's output in the "
                  "mode's dtype, alpha absent gives 1.0")

    def effect(fx, _vel):
        fx.results["adapter_effect"] = {
            k: fixtures.rel_l2(fx.arrays[k], src[k])
            for k in fx.arrays if k in src and k not in (
                "latents", "features", "sigma", "position_ids",
                "velocity_bf16")
            and not k.startswith("in.")}

    c3.run_case(torch, models, src["latents"], src["features"],
                float(src["sigma"]), cfg["num_text_layers"], True, a.name,
                inputs, {"dit": a.dit, "lora": a.lora}, [
                    "the adapter is applied unmerged as PEFT applies it; "
                    "the gmlx side runs it under a patch table"],
                check=7, script=SCRIPT, more=effect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
