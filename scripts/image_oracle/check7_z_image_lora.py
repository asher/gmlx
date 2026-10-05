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

The step also runs block by block, as check 3's does on the base model:
every block of the merged bf16 model is fed the merged fp32 model's input
to it, and each row's tolerance is twice the bf16 against fp32 gap of
that row. A velocity sums 34 blocks, so an adapter that is missing from
one block, or is a tenth too strong, moves it less than the velocity
tolerance. A block row sees only its own block, where the same fault is
several times its tolerance. The manifest also records the adapter's
effect on every block, as the base model's block fed the merged model's
input against the merged model's output, which says which faults a block
row can see.

The fixture holds the step inputs, every pair, the base velocity and the
two velocities of each variant. A second fixture per variant, named
<name>_<variant>_blocks, holds what check 3's 512 fixture holds, for the
merged model: the fp32 input of the first block of each chain, the fp32
output of every block and of its attention and feed-forward modules, and
the tolerance of every row. The gmlx test writes the pairs in each
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

import check3_z_image as c3
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
BLOCK_RULE = (RULE + ". Each row, branch row and free-running block has 2 x "
              "the oracle's own bf16 against fp32 relative L2 at that "
              "point, with the bf16 block teacher-forced on the merged "
              "fp32 model's input for a row and a branch row")


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


def block_case(torch, build, build_base, lat: np.ndarray, t_model,
               cap: np.ndarray) -> dict:
    """One step of a merged model in fp32 and in bf16, block by block, as
    check 3 runs the base model. ``build(mode)`` gives the merged model
    in that mode, ``build_base()`` the fp32 model with no adapter, and
    one model is alive at a time. Returns the fp32 arrays of a block
    fixture, the two velocities, the bf16 against fp32 gap of every row,
    branch row and free-running block, and the adapter's effect on every
    block: the base model's block fed the merged model's input, against
    the merged model's output."""
    to_np = oracle.to_numpy
    times = {}
    t0 = time.perf_counter()
    m32 = build("fp32")
    t1 = time.perf_counter()
    oracle.emulate_bf16_matmul(False)
    c32, b32 = c3.Capture(m32), c3.Branches(m32)
    v32 = to_np(step(torch, m32, lat, t_model, cap, "fp32"))
    c32.remove()
    b32.remove()
    times["fp32"] = {"merge_s": round(t1 - t0, 1),
                     "step_s": round(time.perf_counter() - t1, 1)}
    names = [n for n, _ in c3.chain(m32)]
    # Each block's input is the previous block's output.
    for prev, nxt in zip(names, names[1:]):
        if nxt not in ("context_refiner.0", "layers.0"):
            assert c32.first_arg(nxt) is c32.out(prev), (prev, nxt)
    del m32
    gc.collect()
    arrays = {"t_embedder.in": to_np(c32.first_arg("t_embedder")),
              "t_embedder.out": to_np(c32.out("t_embedder"))}
    for n in ("noise_refiner.0", "context_refiner.0", "layers.0"):
        arrays[f"{n}.in"] = to_np(c32.first_arg(n))
    for n in names:
        arrays[f"{n}.out"] = to_np(c32.out(n))
    for k, v in b32.out.items():
        arrays[f"{k}.out"] = to_np(v)

    # What the adapter changes in each block, which says which faults a
    # block row can see: the base block on the merged model's input.
    t0 = time.perf_counter()
    plain = build_base()
    bp = c3.Branches(plain)
    effect = {}
    for name, mod in c3.chain(plain):
        call = c32.calls[name]
        out = mod(*call["args"], **call["kwargs"])
        effect[name] = fixtures.rel_l2(to_np(out), arrays[f"{name}.out"])
    bp.remove()
    for k, v in bp.out.items():
        effect[k] = fixtures.rel_l2(to_np(v), arrays[f"{k}.out"])
    times["base_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
    del plain, bp, out
    gc.collect()

    t0 = time.perf_counter()
    m16 = build("bf16")
    t1 = time.perf_counter()
    oracle.emulate_bf16_matmul(True)
    c16 = c3.Capture(m16)
    v16 = to_np(step(torch, m16, lat, t_model, cap, "bf16"))
    c16.remove()
    t2 = time.perf_counter()
    b16 = c3.Branches(m16)
    forced = c3.teacher_forced(torch, m16, c32)
    b16.remove()
    oracle.emulate_bf16_matmul(False)
    times["bf16"] = {"merge_s": round(t1 - t0, 1),
                     "step_s": round(t2 - t1, 1),
                     "teacher_forced_s": round(time.perf_counter() - t2, 1)}
    del m16
    gc.collect()
    rows = {"t_embedder": fixtures.compare(to_np(forced["t_embedder"]),
                                           arrays["t_embedder.out"])}
    for row, src in c3.ROW_SOURCES[1:]:
        rows[row] = fixtures.compare(to_np(c16.first_arg(src)),
                                     arrays[f"{src}.in"])
    free = {}
    for n in names:
        rows[n] = fixtures.compare(to_np(forced[n]), arrays[f"{n}.out"])
        free[n] = fixtures.compare(to_np(c16.out(n)), arrays[f"{n}.out"])
    branches = {k: fixtures.compare(to_np(b16.out[k]), arrays[f"{k}.out"])
                for k in b32.out}
    return {"arrays": arrays, "fp32": v32, "bf16": v16, "rows": rows,
            "free": free, "branches": branches, "effect": effect,
            "seconds": times}


def write_blocks(name: str, script: str, dit: Path, base: dict, src: dict,
                 case: dict, inputs: dict, out) -> Path:
    """Write the block fixture of one merged model, in the layout of
    check 3's 512 fixture, so the gmlx side reads both the same way."""
    fx = fixtures.FixtureWriter(name, check=7, family="z_image",
                                script=script)
    fx.add_file("dit", dit)
    for k in STEP_KEYS[:-1]:
        fx.add(k, base[k])
    for k, v in case["arrays"].items():
        fx.add(k, v)
    fx.add("velocity", case["fp32"])
    gap = fixtures.compare(case["bf16"], case["fp32"])
    fx.inputs.update(
        source=SOURCE, size=src["inputs"]["size"],
        sigma=src["inputs"]["sigma"], t_model=src["inputs"]["t_model"],
        prompt=src["inputs"]["prompt"], **inputs,
        rows=list(case["rows"]), free_rows=list(case["free"]),
        branches=list(case["branches"]))
    fx.results.update(
        oracle={"device": "cpu", "dequant": oracle.dequant_mode()},
        velocity_bf16_vs_fp32=gap,
        rows_bf16_vs_fp32_teacher_forced=case["rows"],
        branches_bf16_vs_fp32_teacher_forced=case["branches"],
        blocks_bf16_vs_fp32_free=case["free"],
        adapter_effect_teacher_forced=case["effect"],
        seconds=case["seconds"])
    fx.tolerances.update(
        rule=BLOCK_RULE, velocity_rel_l2=2 * gap["rel_l2"],
        velocity_cosine_min=1 - 2 * (1 - gap["cosine"]),
        rows_rel_l2={r: 2 * g["rel_l2"] for r, g in case["rows"].items()},
        free_rel_l2={n: 2 * g["rel_l2"] for n, g in case["free"].items()},
        branch_rel_l2={k: 2 * g["rel_l2"]
                       for k, g in case["branches"].items()})
    return fx.write(out)


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
        alpha, strength, dialects = VARIANTS[variant]

        def build(mode, alpha=alpha, strength=strength):
            return merge(torch, oracle.load_dit(a.dit, a.config, mode,
                                                dequant=a.dequant),
                         pairs, alpha, strength, mode)

        case = block_case(torch, build, lambda: oracle.load_dit(
            a.dit, a.config, "fp32", dequant=a.dequant), lat, t_model, cap)
        v = {m: case[m] for m in oracle.MODES}
        times[variant] = case["seconds"]
        print(f"[check7] {variant}: {case['seconds']}, peak "
              f"{rss_gib():.1f} GiB", flush=True)
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
        blocks = write_blocks(
            f"{a.name}_{variant}_blocks", fx.manifest["script"], a.dit,
            base, src, case, {
                "pairs_fixture": a.name, "variant": variant, "alpha": alpha,
                "strength": strength, "dialects": list(dialects),
                "targets": names, "rank": a.rank, "gain": a.gain}, a.out)
        worst = max(case["rows"].items(), key=lambda kv: kv[1]["rel_l2"])
        print(f"[check7] wrote {blocks}; worst teacher-forced row "
              f"{worst[0]} {worst[1]['rel_l2']:.3e}", flush=True)
        del case
        gc.collect()
    fx.results["seconds"] = times
    fx.results["peak_rss_gib"] = round(rss_gib(), 2)
    path = fx.write(a.out)
    print(f"[check7] wrote {path}, peak {rss_gib():.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
