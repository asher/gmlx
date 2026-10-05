#!/usr/bin/env python3
"""Check 7 oracle for Z-Image, step half: one denoise step with a LoRA
merged into dequantized dense weights.

The step is check 3's at 512: its latents, caption features and timestep
are read from that fixture, so no text encoder loads. The adapter is
synthetic. Each target is a linear layer in the diffusers layout, where q,
k and v are separate weights, so the merge places no band and the gmlx
side's band placement on the fused GGUF weight is checked against it. The
pair of a target is seeded normal noise, scaled so that the Frobenius
norm of up @ down is --gain times that of the target's dequantized weight.

A variant fixes the alpha and the strength. For each variant and each
mode the DiT is loaded from the GGUF, every target's weight is
dequantized to float32, and

    W' = W + strength * (alpha / rank) * (up @ down)

replaces it as a dense linear layer in the mode's dtype, where a missing
alpha gives the factor 1.0. The fp32 velocity is the reference, and the
tolerance of a variant is twice its own bf16 against fp32 gap, which is
check 3's rule.

The fixture holds the step inputs, every pair, the base velocity and the
two velocities of each variant. The gmlx test writes the pairs in each
dialect's key spelling and runs the step with them applied live.

  python scripts/image_oracle/run.py --python <oracle python> --threads 12 \\
      check7_z_image_lora.py --dit z-image-turbo-Q4_K_M.gguf \\
      --config <official repo dir>
"""

from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

SOURCE = "check3_z_image_turbo_512"
STEP_KEYS = ("latents", "cap_feats", "sigma", "t_model", "velocity")
LEAVES = ("attention.to_q", "attention.to_k", "attention.to_v",
          "attention.to_out.0", "feed_forward.w1", "feed_forward.w2",
          "feed_forward.w3")
# name -> (alpha, strength, dialects whose files carry this arithmetic)
VARIANTS = {
    "alpha": (4.0, 0.8, ("kohya", "onetrainer")),
    "plain": (None, 0.5, ("diffusers", "peft")),
}
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2 of "
        "the velocity with the adapter merged, and the cosine floor is 1 - "
        "2 x its own cosine distance: check 3's rule on the merged model")


def rss_gib() -> float:
    """Peak resident set of this process so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 30 if sys.platform == "darwin" else 1 << 20)


def targets(model, blocks: int | None) -> list[str]:
    """Target names in the diffusers layout. ``blocks`` keeps that many
    blocks, spread over the three groups, or all of them."""
    names = []
    for group in ("noise_refiner", "context_refiner", "layers"):
        names += [f"{group}.{i}" for i in range(len(getattr(model, group)))]
    if blocks is not None and blocks < len(names):
        idx = np.unique(np.round(np.linspace(0, len(names) - 1, blocks))
                        .astype(int))
        names = [names[i] for i in idx]
    return [f"{b}.{leaf}" for b in names for leaf in LEAVES]


def module_at(model, name: str):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def dense_weight(torch, lin):
    """A linear layer's weight, dequantized to float32."""
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    return dequantize_gguf_tensor(lin.weight).to(torch.float32)


def make_pairs(torch, model, names: list, rank: int, gain: float,
               seed: int) -> dict:
    """name -> (down (rank, in), up (out, rank)) float32, with the norm
    of up @ down at ``gain`` times the norm of the target's weight."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for name in names:
        parent, leaf = module_at(model, name)
        lin = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        w = dense_weight(torch, lin)
        out, inp = w.shape
        down = rng.standard_normal((rank, inp)).astype(np.float32)
        up = rng.standard_normal((out, rank)).astype(np.float32)
        d64, u64 = down.astype(np.float64), up.astype(np.float64)
        prod = float(np.sqrt(np.sum((u64.T @ u64) * (d64 @ d64.T))))
        target = gain * float(torch.linalg.norm(w.double()))
        pairs[name] = (down, (up * np.float32(target / prod)))
        del w
    return pairs


def merge(torch, model, pairs: dict, alpha, strength: float, mode: str):
    """Replace every target with a dense linear layer of the merged
    weight in the mode's dtype."""
    dt = oracle.torch_dtype(mode)
    for name, (down, up) in pairs.items():
        parent, leaf = module_at(model, name)
        lin = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        w = dense_weight(torch, lin)
        rank = down.shape[0]
        f = strength * (1.0 if alpha is None else alpha / rank)
        w += f * (torch.from_numpy(up) @ torch.from_numpy(down))
        new = torch.nn.Linear(w.shape[1], w.shape[0],
                              bias=lin.bias is not None)
        new.weight = torch.nn.Parameter(w.to(dt), requires_grad=False)
        if lin.bias is not None:
            new.bias = torch.nn.Parameter(lin.bias.detach().to(dt),
                                          requires_grad=False)
        if leaf.isdigit():
            parent[int(leaf)] = new
        else:
            setattr(parent, leaf, new)
        del w, lin
    return model.eval()


def step(torch, model, lat: np.ndarray, t_model, cap: np.ndarray, mode: str):
    """One transformer call as ZImagePipeline makes it, at batch 1."""
    dt = oracle.torch_dtype(mode)
    x = [torch.from_numpy(lat[0]).to(dt).unsqueeze(1)]
    caps = [torch.from_numpy(cap).to(dt)]
    out = model(x, t_model, caps, return_dict=False)[0]
    return -torch.stack([o.float() for o in out], dim=0).squeeze(2)


def read_step(root: Path) -> tuple[dict, dict]:
    """The step inputs and base velocity of check 3's fixture, each
    checked against its manifest."""
    m = fixtures.read_manifest(SOURCE, root=root)
    arrays = {}
    with np.load(root / m["npz"], allow_pickle=False) as z:
        for k in STEP_KEYS:
            a = z[k]
            if fixtures.array_sha256(a) != m["tensors"][k]["sha256"]:
                raise SystemExit(f"error: {k} of {SOURCE} differs from its "
                                 "manifest")
            arrays[k] = a
    return arrays, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", type=Path, required=True,
                    help="Z-Image DiT GGUF")
    ap.add_argument("--config", type=Path, required=True,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--gain", type=float, default=0.15,
                    help="norm of each delta over the norm of its weight. "
                         "At 0.15 the adapter moves the velocity by a few "
                         "times the tolerance, so the base model fails the "
                         "gate")
    ap.add_argument("--blocks", type=int, default=None,
                    help="adapt this many blocks, spread over the model "
                         "(default: all)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check7_z_image_turbo_lora_512")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the fixture here instead of the fixture "
                         "directory, for a trial run")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    root = fixtures.fixture_root()
    base, src = read_step(root)
    if fixtures.file_identity(a.dit)["sha256"] != \
            src["files"]["dit"]["sha256"]:
        raise SystemExit(f"error: {a.dit} is not the DiT {SOURCE} read")
    lat, cap = base["latents"], base["cap_feats"]
    t_model = torch.from_numpy(base["t_model"])

    fx = fixtures.FixtureWriter(
        a.name, check=7, family="z_image",
        script="scripts/image_oracle/check7_z_image_lora.py")
    fx.add_file("dit", a.dit)
    for k in STEP_KEYS[:-1]:
        fx.add(k, base[k])
    fx.add("velocity_base", base["velocity"])

    t0 = time.perf_counter()
    model = oracle.load_dit(a.dit, a.config, "fp32", dequant=a.dequant)
    names = targets(model, a.blocks)
    pairs = make_pairs(torch, model, names, a.rank, a.gain, a.seed)
    del model
    gc.collect()
    print(f"[check7] {len(pairs)} pairs of rank {a.rank} in "
          f"{time.perf_counter() - t0:.0f} s, peak {rss_gib():.1f} GiB",
          flush=True)
    for name, (down, up) in pairs.items():
        fx.add(f"down.{name}", down)
        fx.add(f"up.{name}", up)

    fx.inputs.update(
        source=SOURCE, size=src["inputs"]["size"],
        sigma=src["inputs"]["sigma"], t_model=src["inputs"]["t_model"],
        prompt=src["inputs"]["prompt"], targets=names, rank=a.rank,
        gain=a.gain, pair_seed=a.seed, blocks=a.blocks,
        pairs="numpy default_rng(seed) standard_normal float32, down then "
              "up per target in order, up scaled so that the norm of up @ "
              "down is gain x the norm of the dequantized weight",
        merge="W + strength * (alpha / rank) * (up @ down) in float32, "
              "then cast to the mode's dtype; a missing alpha gives 1.0",
        variants={v: {"alpha": VARIANTS[v][0], "strength": VARIANTS[v][1],
                      "dialects": list(VARIANTS[v][2])}
                  for v in a.variants})
    fx.results["oracle"] = {"device": "cpu",
                            "threads": torch.get_num_threads(),
                            "dequant": a.dequant}
    fx.tolerances.update(rule=RULE, check3_512={
        k: src["tolerances"][k] for k in ("velocity_rel_l2",
                                          "velocity_cosine_min")})
    times = {}
    for variant in a.variants:
        alpha, strength, _ = VARIANTS[variant]
        v = {}
        for mode in oracle.MODES:
            t0 = time.perf_counter()
            model = merge(torch, oracle.load_dit(a.dit, a.config, mode,
                                                 dequant=a.dequant),
                          pairs, alpha, strength, mode)
            t1 = time.perf_counter()
            oracle.emulate_bf16_matmul(mode == "bf16")
            v[mode] = oracle.to_numpy(step(torch, model, lat, t_model, cap,
                                           mode))
            oracle.emulate_bf16_matmul(False)
            times[f"{variant}_{mode}"] = {
                "merge_s": round(t1 - t0, 1),
                "step_s": round(time.perf_counter() - t1, 1)}
            del model
            gc.collect()
            print(f"[check7] {variant} {mode}: {times[f'{variant}_{mode}']}, "
                  f"peak {rss_gib():.1f} GiB", flush=True)
        fx.add(f"velocity.{variant}", v["fp32"])
        fx.add(f"velocity_bf16.{variant}", v["bf16"])
        gap = fixtures.compare(v["bf16"], v["fp32"])
        effect = fixtures.compare(v["fp32"], base["velocity"])
        fx.results[variant] = {"velocity_bf16_vs_fp32": gap,
                               "adapter_vs_base_fp32": effect}
        fx.tolerances[variant] = {
            "velocity_rel_l2": 2 * gap["rel_l2"],
            "velocity_cosine_min": 1 - 2 * (1 - gap["cosine"])}
        print(f"[check7] {variant}: bf16 vs fp32 rel L2 "
              f"{gap['rel_l2']:.3e} cos {gap['cosine']:.6f}; adapter vs "
              f"base rel L2 {effect['rel_l2']:.3e}", flush=True)
    fx.results["seconds"] = times
    fx.results["peak_rss_gib"] = round(rss_gib(), 2)
    path = fx.write(a.out)
    print(f"[check7] wrote {path}, peak {rss_gib():.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
