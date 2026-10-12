#!/usr/bin/env python3
"""Check 7 oracle for Z-Image on community LoRA files, step half.

The step is check 3's at 512, as in check7_z_image_lora.py: its latents,
caption features and timestep are read from that fixture. Each adapter is
a LoRA file from a public repository. The oracle reads its keys with the
small parser below, which knows the two spellings these files use:

    lora_unet_<path with underscores>.lora_down.weight, .lora_up.weight,
        .alpha
    diffusion_model.<path>.lora_A.weight, .lora_B.weight

A path is a linear layer of the diffusers model, where q, k and v are
separate weights, so the merge places no band. A key that names no such
layer stops the run. For each adapter and each mode the DiT is loaded
from the GGUF, every target's weight is dequantized to float32, and

    W' = W + strength * (alpha / rank) * (up @ down)

replaces it as a dense linear layer in the mode's dtype, where a missing
alpha gives the factor 1.0. That is ComfyUI's arithmetic for a LoRA patch.
The fp32 velocity is the reference, and the tolerance of an adapter is
twice its own bf16 against fp32 gap, which is check 3's rule. The step
also runs block by block on the merged model, as in check7_z_image_lora.py,
and a fixture per adapter named <name>_<id>_blocks holds the fp32 output
of every block and the tolerance of every row.

The key mapping and the alpha factor are also compared with diffusers'
own Z-Image LoRA converter: for every target, the product of the factors
it returns against the oracle's scaled product, in float64.

The adapters are listed in a JSON file given with --spec:

    {"adapters": [{"id": "name", "path": "file.safetensors",
                   "repo": "owner/repo", "revision": "sha",
                   "license": "apache-2.0", "strength": 1.0}]}

A relative path is taken from the spec file's directory. The fixture holds
the step inputs, the base velocity and the two velocities of each adapter.
The adapter files stay outside the fixture: the manifest records each
file's name, size, sha256, repository, revision and license, and the gmlx
test finds the file under GMLX_TEST_LORA_DIR.

  python scripts/image_oracle/run.py --python <oracle python> --threads 12 \\
      check7_z_image_community.py --dit z-image-turbo-Q4_K_M.gguf \\
      --config <official repo dir> --spec adapters.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import check7_z_image_lora as synth
import fixtures
import oracle

SUFFIXES = {".lora_down.weight": "down", ".lora_up.weight": "up",
            ".lora_A.weight": "down", ".lora_B.weight": "up",
            ".alpha": "alpha"}
PREFIXES = {"lora_unet_": "kohya", "diffusion_model.": "dotted"}


def linear_names(model) -> set:
    """Names of the model's linear layers, float or GGUF."""
    return {n for n, m in model.named_modules()
            if getattr(getattr(m, "weight", None), "ndim", 0) == 2}


def read_pairs(torch, path: Path, model) -> tuple[dict, dict]:
    """target name -> (down (rank, in), up (out, rank), alpha or None) as
    float32 arrays, and the facts of the file. Stops on a key that names
    no linear layer of the model or is not part of a plain pair."""
    from safetensors import safe_open

    names = linear_names(model)
    flat = {n.replace(".", "_"): n for n in names}
    parts: dict = {}
    spelled, bad, dtypes = set(), [], set()
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            role = next((r for s, r in SUFFIXES.items() if key.endswith(s)),
                        None)
            stem = key if role is None else key[:-len(
                next(s for s in SUFFIXES if key.endswith(s)))]
            prefix = next((p for p in PREFIXES if stem.startswith(p)), None)
            name = None
            if role is not None and prefix is not None:
                rest = stem[len(prefix):]
                name = flat.get(rest) if PREFIXES[prefix] == "kohya" else rest
            if name not in names:
                bad.append(key)
                continue
            spelled.add(PREFIXES[prefix])
            t = f.get_tensor(key)
            dtypes.add(str(t.dtype).rsplit(".", 1)[-1])
            parts.setdefault(name, {})[role] = (
                float(t) if role == "alpha"
                else t.to(torch.float32).numpy())
    if bad:
        raise SystemExit(f"error: {path.name}: {len(bad)} keys name no "
                         f"linear layer or pair part: {bad[:8]}")
    pairs = {}
    for name, p in sorted(parts.items()):
        if "down" not in p or "up" not in p:
            raise SystemExit(f"error: {path.name}: {name} lacks a down or "
                             "an up weight")
        if p["down"].ndim != 2 or p["up"].ndim != 2:
            raise SystemExit(f"error: {path.name}: {name} is not a plain "
                             "pair of matrices")
        pairs[name] = (p["down"], p["up"], p.get("alpha"))
    ranks = sorted({int(d.shape[0]) for d, _, _ in pairs.values()})
    alphas = sorted({a for _, _, a in pairs.values() if a is not None})
    leaves = sorted({n.split(".", 2)[2] for n in pairs})
    groups = {g: len({n.split(".")[1] for n in pairs if n.startswith(g)})
              for g in ("layers", "noise_refiner", "context_refiner")}
    facts = {"pairs": len(pairs), "spelling": sorted(spelled),
             "ranks": ranks, "alphas": alphas or None,
             "tensor_dtypes": sorted(dtypes), "leaves": leaves,
             "blocks": groups}
    return pairs, facts


def scale_of(alpha, rank: int, strength: float) -> float:
    return strength * (1.0 if alpha is None else alpha / rank)


def merge_pairs(torch, model, pairs: dict, strength: float, mode: str):
    """Replace every target with a dense linear layer of the merged
    weight in the mode's dtype."""
    dt = oracle.torch_dtype(mode)
    for name, (down, up, alpha) in pairs.items():
        parent, leaf = synth.module_at(model, name)
        lin = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        w = synth.dense_weight(torch, lin)
        w += scale_of(alpha, down.shape[0], strength) * (
            torch.from_numpy(up) @ torch.from_numpy(down))
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


def converter_check(torch, path: Path, pairs: dict) -> dict:
    """The oracle's scaled products against those of diffusers' Z-Image
    LoRA converter, per target, as the worst relative L2 in float64."""
    try:
        from diffusers import ZImagePipeline

        sd = ZImagePipeline.lora_state_dict(str(path))
    except Exception as e:  # noqa: BLE001 - a missing converter is a note
        return {"ran": False, "why": f"{type(e).__name__}: {e}"[:300]}
    theirs = {}
    for key in sd:
        if key.endswith(".lora_A.weight"):
            theirs[key[:-len(".lora_A.weight")].removeprefix(
                "transformer.")] = key
    worst, missing = 0.0, sorted(set(pairs) ^ set(theirs))
    for name in sorted(set(pairs) & set(theirs)):
        down, up, alpha = pairs[name]
        mine = scale_of(alpha, down.shape[0], 1.0) * (
            torch.from_numpy(up).double() @ torch.from_numpy(down).double())
        a = sd[theirs[name]].double()
        b = sd[theirs[name].replace(".lora_A.", ".lora_B.")].double()
        diff = torch.linalg.norm(b @ a - mine) / torch.linalg.norm(mine)
        worst = max(worst, float(diff))
    return {"ran": True, "targets": len(theirs),
            "targets_not_in_both": missing[:8],
            "count_not_in_both": len(missing),
            "worst_product_rel_l2": worst}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", type=Path, required=True,
                    help="Z-Image DiT GGUF")
    ap.add_argument("--config", type=Path, required=True,
                    help="local copy of Tongyi-MAI/Z-Image-Turbo")
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON list of the adapters, see the module text")
    ap.add_argument("--only", nargs="+", default=None, metavar="ID",
                    help="run these adapters of the spec only")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check7_z_image_turbo_community_512")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the fixture here instead of the fixture "
                         "directory, for a trial run")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    root = fixtures.fixture_root()
    base, src = synth.read_step(root)
    if fixtures.file_identity(a.dit)["sha256"] != \
            src["files"]["dit"]["sha256"]:
        raise SystemExit(f"error: {a.dit} is not the DiT {synth.SOURCE} "
                         "read")
    spec = json.loads(a.spec.read_text())["adapters"]
    if a.only:
        spec = [s for s in spec if s["id"] in a.only]
    lat, cap = base["latents"], base["cap_feats"]
    t_model = torch.from_numpy(base["t_model"])

    fx = fixtures.FixtureWriter(
        a.name, check=7, family="z_image",
        script="scripts/image_oracle/check7_z_image_community.py")
    fx.add_file("dit", a.dit)
    for k in synth.STEP_KEYS[:-1]:
        fx.add(k, base[k])
    fx.add("velocity_base", base["velocity"])
    fx.inputs.update(
        source=synth.SOURCE, size=src["inputs"]["size"],
        sigma=src["inputs"]["sigma"], t_model=src["inputs"]["t_model"],
        prompt=src["inputs"]["prompt"],
        merge="W + strength * (alpha / rank) * (up @ down) in float32, "
              "then cast to the mode's dtype; a missing alpha gives 1.0",
        adapters={})
    fx.results["oracle"] = {"device": "cpu",
                            "threads": torch.get_num_threads(),
                            "dequant": a.dequant}
    fx.tolerances.update(rule=synth.RULE, check3_512={
        k: src["tolerances"][k] for k in ("velocity_rel_l2",
                                          "velocity_cosine_min")})
    times = {}
    for row in spec:
        aid, strength = row["id"], float(row.get("strength", 1.0))
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            path = a.spec.parent / path
        ident = fx.add_file(f"adapter.{aid}", path)
        model = oracle.load_dit(a.dit, a.config, "fp32", dequant=a.dequant)
        pairs, facts = read_pairs(torch, path, model)
        del model
        gc.collect()
        check = converter_check(torch, path, pairs)
        print(f"[check7] {aid}: {facts['pairs']} pairs, ranks "
              f"{facts['ranks']}, alphas {facts['alphas']}, spelling "
              f"{facts['spelling']}; diffusers converter {check}",
              flush=True)
        fx.inputs["adapters"][aid] = {
            "file": ident, "repo": row["repo"], "revision": row["revision"],
            "license": row["license"], "strength": strength, **facts,
            "targets": sorted(pairs)}

        def build(mode, pairs=pairs, strength=strength):
            return merge_pairs(torch, oracle.load_dit(
                a.dit, a.config, mode, dequant=a.dequant), pairs, strength,
                mode)

        case = synth.block_case(torch, build, lambda: oracle.load_dit(
            a.dit, a.config, "fp32", dequant=a.dequant), lat, t_model, cap)
        v = {m: case[m] for m in oracle.MODES}
        times[aid] = case["seconds"]
        print(f"[check7] {aid}: {case['seconds']}, peak "
              f"{synth.rss_gib():.1f} GiB", flush=True)
        del pairs, build
        gc.collect()
        blocks = synth.write_blocks(
            f"{a.name}_{aid}_blocks", fx.manifest["script"], a.dit, base,
            src, case, {"adapters_fixture": a.name, "adapter": aid,
                        "file": ident, "strength": strength}, a.out)
        worst = max(case["rows"].items(), key=lambda kv: kv[1]["rel_l2"])
        print(f"[check7] wrote {blocks}; worst teacher-forced row "
              f"{worst[0]} {worst[1]['rel_l2']:.3e}", flush=True)
        del case
        gc.collect()
        fx.add(f"velocity.{aid}", v["fp32"])
        fx.add(f"velocity_bf16.{aid}", v["bf16"])
        gap = fixtures.compare(v["bf16"], v["fp32"])
        effect = fixtures.compare(v["fp32"], base["velocity"])
        fx.results[aid] = {"velocity_bf16_vs_fp32": gap,
                           "adapter_vs_base_fp32": effect,
                           "diffusers_converter": check}
        fx.tolerances[aid] = {
            "velocity_rel_l2": 2 * gap["rel_l2"],
            "velocity_cosine_min": 1 - 4 * (1 - gap["cosine"])}
        print(f"[check7] {aid}: bf16 vs fp32 rel L2 {gap['rel_l2']:.3e} cos "
              f"{gap['cosine']:.6f}; adapter vs base rel L2 "
              f"{effect['rel_l2']:.3e}", flush=True)
    fx.results["seconds"] = times
    fx.results["peak_rss_gib"] = round(synth.rss_gib(), 2)
    out = fx.write(a.out)
    print(f"[check7] wrote {out}, peak {synth.rss_gib():.1f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
