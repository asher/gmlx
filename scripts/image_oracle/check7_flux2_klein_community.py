#!/usr/bin/env python3
"""Check 7 oracle for FLUX.2 klein on community LoRA files, step half:
check 3's step at 512 with an adapter file applied to BFL's reference DiT.

Each adapter is a LoRA file from a public repository, listed in --spec
with its repository, revision, license, strength, the DiT GGUF it is for
and the check 3 fixture whose step it takes. The oracle reads the file's
keys with the parser below, which knows the three spellings these files
use:

    transformer.<diffusers name>.lora_A.weight, .lora_B.weight
        (diffusers' save_lora_weights, with a PEFT config in the header)
    lora_unet_<reference name with underscores>.lora_down.weight,
        .lora_up.weight, .alpha (kohya)
    diffusion_model.<reference name>.lora_A.weight, .lora_B.weight
        (AI-Toolkit)

A reference name is a linear layer of the reference's Flux2 class, which
the GGUF names. A diffusers name is a layer of diffusers'
Flux2Transformer2DModel, and ``diffusers_table`` maps it to the reference
layer and the rows of that layer it is: to_q, to_k and to_v are the
thirds of a double block's img_attn.qkv, add_q_proj, add_k_proj and
add_v_proj those of txt_attn.qkv, and norm_out.linear is
final_layer.adaLN_modulation.1 with its two halves in the other order. A
key that names no such layer stops the run. The table is checked against
diffusers' own converter of the reference names, which the manifest
records.

The scale of a pair is ``strength * scale``. For a file with a PEFT config
in its header, ``scale`` is PEFT's own: the config goes to peft's
LoraConfig, the adapter is injected into a stand-in tree of the file's
module names, and the ``scaling`` of each LoRA layer is read, so r,
lora_alpha, rank_pattern, alpha_pattern and use_rslora count as PEFT
counts them. Otherwise ``scale`` is the alpha over the rank with an alpha
tensor, and 1 without. The manifest records peft's version.

Each pair is applied unmerged, as PEFT applies it: a forward hook on the
reference layer adds ``strength * scale * linear(linear(x, down), up)`` to
the layer's rows of the pair in the mode's dtype. In bf16 the two products
run as the base layers do, as float32 products over bfloat16 operands with
one bfloat16 rounding.

The step is the source fixture's: its latents, features, reference
latents and sigma, every row in fp32 and the bf16 gaps that set the
tolerances, with check 3's rule. The manifest also records the adapter's
effect, the relative L2 of each fp32 row against the same row of the
source fixture, and the products of the pairs that diffusers' FLUX.2 LoRA
loader gives against the oracle's. The adapter file stays outside the
fixture: the manifest records its name, size, sha256, repository, revision
and license, and the gmlx test finds it under GMLX_TEST_LORA_DIR.

The spec is a JSON file:

    {"adapters": [{"id": "name", "path": "file.safetensors",
                   "repo": "owner/repo", "revision": "sha",
                   "license": "apache-2.0", "strength": 1.0,
                   "dit": "flux-2-klein-4b-Q4_K_M.gguf",
                   "source": "check3_flux2_klein_512"}]}

A relative path is taken from the spec file's directory. Each adapter
writes the fixture <name>_<id>_512.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check7_flux2_klein_community.py --reference <flux2>/src \\
      --spec adapters.json --only pixel_art
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

# Under -I the script's directory is not on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import check3_flux2_klein as c3  # noqa: E402
import fixtures  # noqa: E402
import oracle  # noqa: E402

NAME = "check7_flux2_klein"
SCRIPT = "scripts/image_oracle/check7_flux2_klein_community.py"
SUFFIXES = {".lora_down.weight": "down", ".lora_up.weight": "up",
            ".lora_A.weight": "down", ".lora_B.weight": "up",
            ".alpha": "alpha"}
PREFIXES = {"transformer.": "diffusers", "lora_unet_": "kohya",
            "diffusion_model.": "reference"}
HEADER_ENTRY = "lora_adapter_metadata"


def diffusers_table(model) -> dict:
    """diffusers name -> (reference layer, third of its rows or None,
    whether the two halves of its rows are in the other order)."""
    out = {}
    for i in range(len(model.double_blocks)):
        d, b = f"transformer_blocks.{i}", f"double_blocks.{i}"
        for j, n in enumerate(("to_q", "to_k", "to_v")):
            out[f"{d}.attn.{n}"] = (f"{b}.img_attn.qkv", j, False)
        for j, n in enumerate(("add_q_proj", "add_k_proj", "add_v_proj")):
            out[f"{d}.attn.{n}"] = (f"{b}.txt_attn.qkv", j, False)
        out[f"{d}.attn.to_out.0"] = (f"{b}.img_attn.proj", None, False)
        out[f"{d}.attn.to_add_out"] = (f"{b}.txt_attn.proj", None, False)
        for ff, mlp in (("ff", "img_mlp"), ("ff_context", "txt_mlp")):
            out[f"{d}.{ff}.linear_in"] = (f"{b}.{mlp}.0", None, False)
            out[f"{d}.{ff}.linear_out"] = (f"{b}.{mlp}.2", None, False)
    for i in range(len(model.single_blocks)):
        d, b = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
        out[f"{d}.attn.to_qkv_mlp_proj"] = (f"{b}.linear1", None, False)
        out[f"{d}.attn.to_out"] = (f"{b}.linear2", None, False)
    emb = "time_guidance_embed.timestep_embedder"
    out.update({
        "x_embedder": ("img_in", None, False),
        "context_embedder": ("txt_in", None, False),
        f"{emb}.linear_1": ("time_in.in_layer", None, False),
        f"{emb}.linear_2": ("time_in.out_layer", None, False),
        "proj_out": ("final_layer.linear", None, False),
        "norm_out.linear": ("final_layer.adaLN_modulation.1", None, True)})
    for m in c3.MODS:
        out[f"{m}.linear"] = (f"{m}.lin", None, False)
    return out


def linear_layers(model) -> dict:
    """Reference name -> (out, in) of each linear layer."""
    return {n: (m.out_features, m.in_features)
            for n, m in model.named_modules()
            if hasattr(m, "out_features") and hasattr(m, "in_features")}


def swap_halves(a: np.ndarray) -> np.ndarray:
    h = a.shape[0] // 2
    return np.concatenate([a[h:], a[:h]], axis=0)


def peft_scales(torch, metadata: dict, shapes: dict) -> tuple[dict, str]:
    """diffusers module name -> PEFT's scaling of its LoRA layer, from the
    header's transformer config, and peft's version."""
    import peft
    from peft import LoraConfig, inject_adapter_in_model

    doc = json.loads(metadata[HEADER_ENTRY])
    fields = {k.split(".", 1)[1]: v for k, v in doc.items()
              if k.startswith("transformer.")}
    if set(doc) != {f"transformer.{k}" for k in fields}:
        raise SystemExit("error: the adapter header has a LoRA config for "
                         "a component other than transformer")
    known = set(LoraConfig.__dataclass_fields__)
    extra = sorted(set(fields) - known)
    if extra:
        raise SystemExit(f"error: peft {peft.__version__} LoraConfig has no "
                         f"field {extra}")
    cfg = LoraConfig(**fields)
    root = torch.nn.Module()
    for name in shapes:
        parent = root
        *path, leaf = name.split(".")
        for part in path:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        parent.add_module(leaf, torch.nn.Linear(1, 1))
    inject_adapter_in_model(cfg, root)
    out = {}
    for name in shapes:
        layer = root.get_submodule(name)
        scaling = getattr(layer, "scaling", None)
        if not scaling:
            raise SystemExit(f"error: the header's config puts no LoRA "
                             f"layer on {name}")
        out[name] = float(scaling["default"])
    return out, peft.__version__


def read_pairs(torch, path: Path, model) -> tuple[dict, dict]:
    """Pair key -> {module, band, down, up, scale, stem} in float32, the
    up rows in the reference's order, and the facts of the file. Stops on
    a key that names no linear layer or is not part of a plain pair."""
    from safetensors import safe_open

    layers = linear_layers(model)
    table = diffusers_table(model)
    flat = {n.replace(".", "_"): n for n in layers}
    parts: dict = {}
    bad, spelled, dtypes = [], set(), set()
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            suffix = next((s for s in SUFFIXES if key.endswith(s)), None)
            stem = key[:-len(suffix)] if suffix else key
            prefix = next((p for p in PREFIXES if stem.startswith(p)), None)
            name = None
            if suffix and prefix:
                rest = stem[len(prefix):]
                kind = PREFIXES[prefix]
                name = (rest if kind == "diffusers" and rest in table
                        else flat.get(rest) if kind == "kohya"
                        else rest if kind == "reference" and rest in layers
                        else None)
            if name is None:
                bad.append(key)
                continue
            spelled.add(PREFIXES[prefix])
            t = f.get_tensor(key)
            dtypes.add(str(t.dtype).rsplit(".", 1)[-1])
            parts.setdefault((PREFIXES[prefix], name), {"stem": stem})[
                SUFFIXES[suffix]] = (float(t) if suffix == ".alpha" else
                                     t.to(torch.float32).numpy())
    if bad:
        raise SystemExit(f"error: {path.name}: {len(bad)} keys name no "
                         f"linear layer or pair part: {bad[:8]}")
    if len(spelled) != 1:
        raise SystemExit(f"error: {path.name} mixes spellings {spelled}")
    kind = spelled.pop()
    for (_, name), p in parts.items():
        if "down" not in p or "up" not in p or p["down"].ndim != 2 \
                or p["up"].ndim != 2:
            raise SystemExit(f"error: {path.name}: {name} is not a plain "
                             "pair of matrices")
    scales, peft_version = {}, None
    if HEADER_ENTRY in metadata:
        if kind != "diffusers":
            raise SystemExit(f"error: {path.name} has a PEFT config and "
                             f"{kind} names")
        scales, peft_version = peft_scales(torch, metadata, {
            name: p["up"].shape for (_, name), p in parts.items()})
    pairs = {}
    for (_, name), p in sorted(parts.items()):
        down, up = p["down"], p["up"]
        rank = down.shape[0]
        if kind == "diffusers":
            module, third, swapped = table[name]
        else:
            module, third, swapped = name, None, False
        out, inp = layers[module]
        band = None
        if third is not None:
            band = (third * out // 3, out // 3)
        if swapped:
            up = swap_halves(up)
        if down.shape[1] != inp or up.shape != (
                band[1] if band else out, rank):
            raise SystemExit(f"error: {path.name}: {name} has down "
                             f"{down.shape} and up {up.shape} for "
                             f"{module} of {out} x {inp}, rows {band}")
        if name in scales:
            scale = scales[name]
        elif "alpha" in p:
            scale = p["alpha"] / rank
        else:
            scale = 1.0
        pairs[name] = {"module": module, "band": band, "down": down,
                       "up": up, "scale": scale, "stem": p["stem"],
                       "alpha": p.get("alpha")}
    facts = {"pairs": len(pairs), "spelling": kind,
             "ranks": sorted({int(p["down"].shape[0])
                              for p in pairs.values()}),
             "alphas": sorted({p["alpha"] for p in pairs.values()
                               if p["alpha"] is not None}) or None,
             "scales": sorted({p["scale"] for p in pairs.values()}),
             "scale_source": (f"peft {peft_version} LoraConfig from the "
                              "header" if peft_version else
                              "alpha over rank, 1 without an alpha"),
             "peft": peft_version,
             "tensor_dtypes": sorted(dtypes),
             "header_config": (json.loads(metadata[HEADER_ENTRY])
                               if HEADER_ENTRY in metadata else None)}
    return pairs, facts


def install(torch, model, pairs: dict, strength: float, mode: str) -> list:
    """A forward hook per pair that adds its product to its rows."""
    dtype = oracle.torch_dtype(mode)
    handles = []
    for p in pairs.values():
        mod = model.get_submodule(p["module"])
        d = torch.from_numpy(p["down"]).to(dtype)
        u = torch.from_numpy(p["up"]).to(dtype)
        scale = strength * p["scale"]
        start = p["band"][0] if p["band"] else None

        def hook(_mod, args, out, d=d, u=u, scale=scale, start=start):
            import torch.nn.functional as F

            x = args[0].to(dtype)
            delta = (F.linear(F.linear(x, d), u) * scale).to(out.dtype)
            if start is None:
                return out + delta
            out = out.clone()
            out[..., start:start + delta.shape[-1]] += delta
            return out

        handles.append(mod.register_forward_hook(hook))
    return handles


def table_check(torch, model) -> dict:
    """diffusers' converter of the reference names on one pair per linear
    layer against ``diffusers_table``: every name it gives is in the
    table, on the same layer and rows. A layer whose rows the table holds
    in the other order is recorded when the converter leaves them as they
    are."""
    from diffusers.loaders.lora_conversion_utils import (
        _convert_non_diffusers_flux2_lora_to_diffusers)

    layers = linear_layers(model)
    names = sorted(layers)
    sd = {}
    for i, name in enumerate(names):
        out, inp = layers[name]
        sd[f"diffusion_model.{name}.lora_A.weight"] = torch.full(
            (1, inp), float(i + 1))
        sd[f"diffusion_model.{name}.lora_B.weight"] = torch.arange(
            out, dtype=torch.float32)[:, None] + 1000 * i
    conv = _convert_non_diffusers_flux2_lora_to_diffusers(dict(sd))
    table = diffusers_table(model)
    bad, unswapped, seen = [], [], set()
    for key, b in sorted(conv.items()):
        if not key.endswith(".lora_B.weight"):
            continue
        name = key.removeprefix("transformer.")[:-len(".lora_B.weight")]
        if name not in table:
            bad.append(f"{name} is not in the table")
            continue
        module, third, swapped = table[name]
        i = int(conv[key.replace(".lora_B.", ".lora_A.")][0, 0].item()) - 1
        out = layers[module][0]
        rows = torch.arange(out, dtype=torch.float32) + 1000 * i
        if third is not None:
            rows = rows[third * out // 3:(third + 1) * out // 3]
        got = b[:, 0]
        flipped = torch.from_numpy(swap_halves(rows.numpy()))
        if names[i] != module:
            bad.append(f"{name}: the converter puts {names[i]} there")
        elif swapped and torch.equal(got, rows):
            unswapped.append(name)
        elif not torch.equal(got, flipped if swapped else rows):
            bad.append(f"{name}: other rows of {module}")
        seen.add(module)
    dropped = sorted(set(layers) - seen)
    if bad or dropped:
        raise SystemExit(f"error: diffusers_table and diffusers' converter "
                         f"differ: {bad[:6]}, layers it drops {dropped[:4]}")
    return {"layers": len(layers),
            "converter_names": sum(k.endswith(".lora_B.weight")
                                   for k in conv),
            "same_layer_and_rows": True,
            "halves_the_converter_leaves_unswapped": unswapped}


def lowrank_rel(torch, b1, a1, b2, a2) -> float:
    """Relative L2 of b1 @ a1 against b2 @ a2 from the factors, in the
    precision of the factors: the square norm of a product of rank r is
    the trace of an r by r product."""
    def sq(b, a):
        return float(torch.trace((a @ a.T) @ (b.T @ b)))
    num = sq(torch.cat([b1, -b2], 1), torch.cat([a1, a2], 0))
    return (max(num, 0.0) / sq(b2, a2)) ** 0.5


def converter_check(torch, path: Path, pairs: dict, model) -> dict:
    """The products of diffusers' FLUX.2 LoRA loader against the oracle's,
    per diffusers name, as the worst relative L2 in float64. The loader
    folds a kohya alpha into the factors and leaves a PEFT scale to PEFT,
    which the oracle's scale stands in for."""
    try:
        from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin

        sd = Flux2LoraLoaderMixin.lora_state_dict(str(path))
    except Exception as e:  # noqa: BLE001 - a loader that fails is a note
        return {"ran": False, "why": f"{type(e).__name__}: {e}"[:300]}
    table = diffusers_table(model)
    by_rows = {(p["module"], p["band"]): p for p in pairs.values()}
    out_of = {m: o for m, (o, _) in linear_layers(model).items()}
    worst, missing, folded = 0.0, [], None
    for key in sd:
        if not key.endswith(".lora_A.weight"):
            continue
        name = key.removeprefix("transformer.")[:-len(".lora_A.weight")]
        module, third, swapped = table[name]
        o = out_of[module]
        band = None if third is None else (third * o // 3, o // 3)
        p = by_rows.get((module, band))
        whole = by_rows.get((module, None))
        if p is None and whole is not None and band is not None:
            up = whole["up"][band[0]:band[0] + band[1]]
            p = dict(whole, up=up)
        if p is None:
            missing.append(name)
            continue
        a = sd[key].double()
        b = sd[key.replace(".lora_A.", ".lora_B.")].double()
        up = p["up"] if not swapped else swap_halves(p["up"])
        kohya = p["alpha"] is not None
        if not kohya and p["scale"] != 1.0:
            folded = False
        mine_b = torch.from_numpy(up).double() * (
            p["scale"] if kohya else 1.0)
        worst = max(worst, lowrank_rel(
            torch, b, a, mine_b, torch.from_numpy(p["down"]).double()))
    return {"ran": True, "names": len([k for k in sd
                                       if k.endswith(".lora_A.weight")]),
            "names_without_an_oracle_pair": missing[:8],
            "count_without_an_oracle_pair": len(missing),
            "worst_product_rel_l2": worst,
            "peft_scale_left_to_peft": folded is False}


def c3_inputs(src: dict) -> set:
    return {k for k in src if k in ("latents", "features", "sigma",
                                    "velocity_bf16")
            or k.startswith(("refs.", "ids.", "in."))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True,
                    help="src directory of BFL's flux2")
    ap.add_argument("--spec", type=Path, required=True,
                    help="JSON list of the adapters, see the module text")
    ap.add_argument("--gguf", type=Path, required=True, nargs="+",
                    help="the DiT GGUFs the spec names")
    ap.add_argument("--only", nargs="+", default=None, metavar="ID",
                    help="run these adapters of the spec only")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args(argv)
    torch = oracle.setup()
    oracle.set_dequant(a.dequant)
    fm, sampling = c3.reference(a.reference.expanduser())
    root = fixtures.fixture_root()
    dits = {p.name: p for p in a.gguf}
    spec = json.loads(a.spec.read_text())["adapters"]
    if a.only:
        spec = [s for s in spec if s["id"] in a.only]
    for row in spec:
        aid, strength = row["id"], float(row.get("strength", 1.0))
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            path = a.spec.parent / path
        m3 = fixtures.read_manifest(row["source"], root=root)
        dit = dits[row["dit"]]
        want = m3["files"]["dit"]
        if fixtures.file_identity(dit) != want:
            raise SystemExit(f"error: {dit} is not the DiT {row['source']} "
                             "read")
        with np.load(root / m3["npz"], allow_pickle=False) as z:
            src = {k: z[k] for k in z.files}
        state, census = c3.gguf_state(torch, dit)
        models = {m: c3.build(torch, fm, state, m) for m in ("fp32", "bf16")}
        pairs, facts = read_pairs(torch, path, models["fp32"])
        tcheck = table_check(torch, models["fp32"])
        ccheck = converter_check(torch, path, pairs, models["fp32"])
        print(f"[check7] {aid}: {facts['pairs']} pairs, {facts['spelling']}"
              f", ranks {facts['ranks']}, scales {facts['scales']} "
              f"({facts['scale_source']}); table {tcheck}; diffusers "
              f"loader {ccheck}", flush=True)
        for mode, model in models.items():
            install(torch, model, pairs, strength, mode)
        refs = [src[f"refs.{i}"] for i in range(m3["inputs"]["refs"])]
        inputs = dict(m3["inputs"])
        inputs.update(
            source=row["source"], strength=strength,
            adapter={"id": aid, "repo": row["repo"],
                     "revision": row["revision"], "license": row["license"],
                     **facts},
            lora_pairs=len(pairs),
            targets=sorted([p["module"], list(p["band"]) if p["band"]
                            else None] for p in pairs.values()),
            application="unmerged: strength * scale * up(down(x)) added "
                        "to each pair's rows of its layer's output in the "
                        "mode's dtype")

        def more(fx, _vel, src=src, tcheck=tcheck, ccheck=ccheck):
            fx.results["adapter_effect"] = {
                k: fixtures.rel_l2(fx.arrays[k], src[k])
                for k in fx.arrays if k in src and k not in c3_inputs(src)}
            fx.results["key_table"] = tcheck
            fx.results["diffusers_loader"] = ccheck

        c3.run_case(torch, sampling, models, src["latents"], src["features"],
                    refs, float(src["sigma"]), True, f"{a.name}_{aid}_512",
                    inputs, {"dit": dit, "lora": path}, [
                        "the adapter is applied unmerged as PEFT applies "
                        "it; the gmlx side runs it under a patch table"],
                    check=7, script=SCRIPT, more=more)
        del models, state, pairs, src
        gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
