#!/usr/bin/env python3
"""Check 7 oracle for Qwen-Image 2.1, step half: check 3's calls on a
model with a LoRA merged into dense weights.

An adapter targets linear layers of the reference transformer, where the
gate and the projection of the image MLP are separate weights. The merge
therefore places no band, and the gmlx side's bands on the fused GGUF
weight are checked against it. Every target's weight is dequantized to
float32, and

    W' = W + sum over adapters of strength * (alpha / rank) * (up @ down)

replaces it as a dense linear layer in the mode's dtype, where a missing
alpha gives the factor 1.0. The merged models then run check 3's three
calls: the joint forward, the extract call and the cached decode. The
fixture of a variant has the layout of a check 3 fixture and its
tolerance rule, on the merged model, so the gmlx side reads the model
under a patch table with the rows it reads the base model with.

On the real file the step is check 3's at 512: the conditioning, the
latents and the two steps are read from that fixture. The pairs are
seeded normal noise on the seven linear layers of --blocks blocks, scaled
so that the Frobenius norm of up @ down is --gain times that of the
target's dequantized weight. A variant fixes the alpha and the strength.
The fixture <name> holds the pairs and <name>_<variant> the calls of a
variant. An adapted block holds 0.81 GiB of dense float32 weights in the
fp32 model and half of that in the bf16 model, and both models are alive
at once.

  python scripts/image_oracle/run.py --python <oracle python> --threads 12 \\
      check7_qwen_image21_lora.py --dit qwen-image-2.1-Q4_K_M.gguf \\
      --config <official repo dir>

--self-test merges the adapters of qwen_image21_tiny.py into the tiny
model and writes one fixture per variant, small enough to commit. The
first one also records the linear layers of the reference class and, for
each key spelling of an adapter file, the layer that diffusers' own Qwen
LoRA loader gives the key.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

import check3_qwen_image21 as c3
import fixtures
import oracle
import qwen_image21_tiny as tiny

SCRIPT = "scripts/image_oracle/check7_qwen_image21_lora.py"
SOURCE = "check3_qwen_image21_512"
LEAVES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
          "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")
# name -> (alpha, strength, dialects whose files carry this arithmetic)
VARIANTS = {
    "alpha": (4.0, 0.8, ("kohya", "onetrainer")),
    "plain": (None, 0.5, ("diffusers", "peft")),
}
MERGE = ("W + the sum over adapters of strength * (alpha / rank) * (up @ "
         "down) in float32, then cast to the mode's dtype; a missing alpha "
         "gives 1.0")
# Key spellings of an adapter file, as name to (prefix, underscore form,
# down suffix, up suffix, stores an alpha). The first four are the
# dialects that tests/image/lora_fixtures.py writes.
SPELLINGS = {
    "kohya": ("lora_unet_", True, ".lora_down.weight", ".lora_up.weight",
              True),
    "diffusers": ("transformer.", False, ".lora_A.weight", ".lora_B.weight",
                  False),
    "peft": ("base_model.model.", False, ".lora_A.default.weight",
             ".lora_B.default.weight", False),
    "onetrainer": ("transformer.", False, ".lora_down.weight",
                   ".lora_up.weight", True),
    "dotted": ("diffusion_model.", False, ".lora_down.weight",
               ".lora_up.weight", True),
}


def linear_modules(torch, model) -> list[str]:
    """The name of every linear layer of the reference class."""
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)]


def block_targets(model, blocks: int | None) -> list[str]:
    """The linear layers of ``blocks`` blocks, spread over the model, or
    of all of them."""
    idx = list(range(len(model.transformer_blocks)))
    if blocks is not None and blocks < len(idx):
        idx = np.unique(np.round(np.linspace(0, len(idx) - 1, blocks))
                        .astype(int)).tolist()
    return [f"transformer_blocks.{i}.{leaf}" for i in idx for leaf in LEAVES]


def leaf_at(model, name: str):
    """(parent, key, layer) of the linear layer ``name``."""
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    key = parts[-1]
    return parent, key, (parent[int(key)] if key.isdigit()
                         else getattr(parent, key))


def dense_weight(torch, lin):
    """A linear layer's weight as a float32 copy, dequantized when the
    layer holds GGUF blocks."""
    w = lin.weight
    if hasattr(w, "quant_type"):
        from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

        w = dequantize_gguf_tensor(w)
    return w.detach().to(torch.float32).clone()


def make_pairs(torch, model, names: list, rank: int, gain: float,
               seed: int) -> dict:
    """name -> (down (rank, in), up (out, rank)) float32, with the norm
    of up @ down at ``gain`` times the norm of the target's weight."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for name in names:
        w = dense_weight(torch, leaf_at(model, name)[2])
        out, inp = w.shape
        down = rng.standard_normal((rank, inp)).astype(np.float32)
        up = rng.standard_normal((out, rank)).astype(np.float32)
        d64, u64 = down.astype(np.float64), up.astype(np.float64)
        prod = float(np.sqrt(np.sum((u64.T @ u64) * (d64 @ d64.T))))
        target = gain * float(torch.linalg.norm(w.double()))
        pairs[name] = (down, (up * np.float32(target / prod)))
        del w
    return pairs


def merge(torch, model, rows: list, mode: str):
    """Replace every target of ``rows``, one (pairs, alpha, strength) per
    adapter, with a dense linear layer of the merged weight in the
    mode's dtype."""
    dt = oracle.torch_dtype(mode)
    names = list(dict.fromkeys(n for pairs, _, _ in rows for n in pairs))
    for name in names:
        parent, key, lin = leaf_at(model, name)
        w = dense_weight(torch, lin)
        for pairs, alpha, strength in rows:
            if name not in pairs:
                continue
            down, up = pairs[name]
            f = strength * (1.0 if alpha is None else alpha / down.shape[0])
            w += f * (torch.from_numpy(up) @ torch.from_numpy(down))
        if lin.bias is not None:
            raise SystemExit(f"error: {name} has a bias")
        new = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        new.weight = torch.nn.Parameter(w.to(dt), requires_grad=False)
        if key.isdigit():
            parent[int(key)] = new
        else:
            setattr(parent, key, new)
        del w, lin
    return model.eval()


def spellings(torch, names: list) -> dict:
    """Spelling to key stem to the layer name that diffusers' Qwen LoRA
    loader gives a pair on that key, one layer and one key at a time. A
    name outside ``names`` is one the loader got wrong, and an exception
    is recorded by its type."""
    import diffusers
    from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin

    if not issubclass(diffusers.QwenImage21Pipeline,
                      QwenImageLoraLoaderMixin):
        raise SystemExit("error: QwenImage21Pipeline has another LoRA "
                         "loader")
    out = {}
    for label, (prefix, under, down_s, up_s, alpha) in SPELLINGS.items():
        rows = {}
        for name in names:
            stem = prefix + (name.replace(".", "_") if under else name)
            state = {stem + down_s: torch.zeros(1, 2),
                     stem + up_s: torch.zeros(2, 1)}
            if alpha:
                state[stem + ".alpha"] = torch.tensor(1.0)
            try:
                got = QwenImageLoraLoaderMixin.lora_state_dict(state)
            except Exception as e:  # noqa: BLE001
                rows[stem] = f"error: {type(e).__name__}"
                continue
            gave = sorted({k.removeprefix("transformer.").split(".lora_")[0]
                           for k in got})
            rows[stem] = gave[0] if len(gave) == 1 else "error: " + \
                ", ".join(gave)
        out[label] = rows
    return out


def read_source(root: Path, name: str) -> tuple[dict, dict]:
    """The conditioning, latents and sigmas of a check 3 fixture, each
    checked against its manifest."""
    m = fixtures.read_manifest(name, root=root)
    arrays = {}
    with np.load(root / m["npz"], allow_pickle=False) as z:
        keys = ["features", "img_mask", "latents_a", "latents_b", "sigma_a",
                "sigma_b", *(k for k in z.files
                             if k.startswith("ref_latents."))]
        for k in keys:
            a = z[k]
            if fixtures.array_sha256(a) != m["tensors"][k]["sha256"]:
                raise SystemExit(f"error: {k} of {name} differs from its "
                                 "manifest")
            arrays[k] = a
    return arrays, m


def source_cond(arrays: dict) -> dict:
    refs = [arrays[k] for k in sorted(
        (k for k in arrays if k.startswith("ref_latents.")),
        key=lambda k: int(k.rsplit(".", 1)[1]))]
    return {"features": arrays["features"],
            "img_mask": arrays["img_mask"].astype(bool),
            "ref_shapes": [tuple(int(v) for v in r.shape[-2:]) for r in refs],
            "ref_latents": refs,
            "target": tuple(int(v) for v in arrays["latents_a"].shape[-2:]),
            "latents_a": arrays["latents_a"],
            "latents_b": arrays["latents_b"]}


def same_steps(a, arrays: dict, sigmas, source: str) -> None:
    """Refuse a schedule whose two steps are not the ones ``source`` ran."""
    for tag, i in (("a", a.step), ("b", a.step_b)):
        if not np.array_equal(oracle.to_numpy(sigmas[i]),
                              arrays[f"sigma_{tag}"]):
            raise SystemExit(f"error: sigma {i} of this schedule is not "
                             f"sigma_{tag} of {source}")


def self_test(torch, a) -> int:
    root = fixtures.fixture_root(create=True)
    base, weights = c3.tiny_models(torch, root)
    work = root / "selftest_qwen_image21_tiny_model"
    names = linear_modules(torch, base["fp32"])
    want = sorted(k[:-len(".weight")]
                  for k, v in tiny.unfuse(weights).items() if v.ndim == 2)
    if sorted(names) != want:
        raise SystemExit("error: the reference class has other linear "
                         "layers than the tiny rule")
    keys = {"linear_modules": names,
            "loader": "QwenImageLoraLoaderMixin.lora_state_dict",
            "spellings": {k: list(v) for k, v in SPELLINGS.items()},
            "resolved": spellings(torch, names)}
    # The pairs of a real run, on the tiny model: the delta of each has
    # the norm the gain asks for.
    scaled = make_pairs(torch, base["fp32"], block_targets(base["fp32"], 1),
                        2, 0.15, 7)
    for name, (down, up) in scaled.items():
        w = dense_weight(torch, leaf_at(base["fp32"], name)[2])
        ratio = float(np.linalg.norm(up.astype(np.float64) @ down)
                      / torch.linalg.norm(w.double()))
        if abs(ratio - 0.15) > 1e-6:
            raise SystemExit(f"error: the pair on {name} has gain {ratio}")
    del base
    a.dit = a.conditioning = a.config_revision = None
    sig = torch.linspace(1.0, 1 / a.steps, a.steps)
    committed = Path(__file__).resolve().parents[2] / "tests" / "fixtures" \
        / "image" / "oracle"
    for variant, (case, stack) in tiny.LORA_VARIANTS.items():
        target, blocks = c3.SELF_CASES[case]
        cond = c3.tiny_cond(case, target)
        # The base model's fixture of this case holds the same inputs.
        src, _ = read_source(committed, f"selftest_qwen_image21_tiny_{case}")
        same_steps(a, src, sig, f"selftest_qwen_image21_tiny_{case}")
        read = source_cond(src)
        for k, v in cond.items():
            same = (all(np.array_equal(x, y) for x, y in zip(v, read[k]))
                    and len(v) == len(read[k])) if isinstance(v, list) \
                else np.array_equal(np.asarray(v), np.asarray(read[k]))
            if not same:
                raise SystemExit(f"error: {k} of {case} differs from the "
                                 "tiny rule")
        rows = [(tiny.lora_pairs(n), alpha, s) for n, alpha, s in stack]
        models = {m: merge(torch, oracle.load_dit(
            work / "tiny.gguf", work, m, cls_name=c3.CLS), rows, m)
            for m in oracle.MODES}
        note = {"config": tiny.TINY, "weights": "qwen_image21_tiny.weights()",
                "weights_sha256": tiny.sha256(weights),
                "inputs": "qwen_image21_tiny features, ref_latents and "
                          "latents",
                "loaded": "from a GGUF in GGUF tensor names through "
                          "from_single_file",
                "case": case, "variant": variant,
                "pairs": "qwen_image21_tiny.lora_pairs", "merge": MERGE,
                "adapters": [{"name": n, "alpha": alpha, "strength": s,
                              "targets": list(tiny.LORA[n]),
                              "pairs_sha256": tiny.lora_sha256(n)}
                             for n, alpha, s in stack]}
        if keys is not None:
            note["lora_keys"], keys = keys, None
        path = c3.run_case(torch, a, models, read, sig, sig * 1000, blocks,
                           f"selftest_qwen_image21_tiny_lora_{variant}",
                           note, check=7, script=SCRIPT)
        print(f"[check7] wrote {path}", flush=True)
        del models
        gc.collect()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", type=Path, help="Qwen-Image 2.1 DiT GGUF")
    ap.add_argument("--config", type=Path,
                    help="local copy of Qwen/Qwen-Image-2.1 (transformer "
                         "and scheduler configs)")
    ap.add_argument("--config-revision",
                    default="790c92633540aa0cb11d9abf19eb46d861714758",
                    help="revision of the --config copy, for the manifest")
    ap.add_argument("--source", default=SOURCE,
                    help="check 3 fixture whose step this runs")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--gain", type=float, default=0.15,
                    help="norm of each delta over the norm of its weight")
    ap.add_argument("--blocks", type=int, default=8,
                    help="adapt this many blocks, spread over the model. "
                         "Each holds 0.81 GiB of dense float32 weights "
                         "and half of that in bf16, both at once")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bf16-timestep", choices=("fp32", "stock"),
                    default="fp32")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check7_qwen_image21_lora_512")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny model's fixtures and stop")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.self_test:
        a.steps, a.step, a.step_b = 8, 3, 5
        return self_test(torch, a)
    if not (a.dit and a.config):
        ap.error("--dit and --config are required")
    root = fixtures.fixture_root()
    src, sm = read_source(root, a.source)
    if fixtures.file_identity(a.dit)["sha256"] != \
            sm["files"]["dit"]["sha256"]:
        raise SystemExit(f"error: {a.dit} is not the DiT {a.source} read")
    ins = sm["inputs"]
    a.steps, a.step, a.step_b = ins["steps"], ins["step_a"], ins["step_b"]
    a.conditioning = None
    cond = source_cond(src)
    h, w = cond["target"]
    sigmas, timesteps = c3.schedule(torch, a.config, a.steps, h * w)
    same_steps(a, src, sigmas, a.source)

    def load(mode):
        return oracle.load_dit(a.dit, a.config, mode, cls_name=c3.CLS,
                               dequant=a.dequant)

    t0 = time.perf_counter()
    model = load("fp32")
    names = block_targets(model, a.blocks)
    pairs = make_pairs(torch, model, names, a.rank, a.gain, a.seed)
    del model
    gc.collect()
    print(f"[check7] {len(pairs)} pairs of rank {a.rank} in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    fx = fixtures.FixtureWriter(a.name, check=7, family="qwen_image21",
                                script=SCRIPT)
    fx.add_file("dit", a.dit)
    for name, (down, up) in pairs.items():
        fx.add(f"down.{name}", down)
        fx.add(f"up.{name}", up)
    fx.inputs.update(
        source=a.source, targets=names, rank=a.rank, gain=a.gain,
        pair_seed=a.seed, blocks=a.blocks,
        pairs="numpy default_rng(seed) standard_normal float32, down then "
              "up per target in order, up scaled so that the norm of up @ "
              "down is gain x the norm of the dequantized weight",
        merge=MERGE,
        variants={v: {"alpha": VARIANTS[v][0], "strength": VARIANTS[v][1],
                      "dialects": list(VARIANTS[v][2]),
                      "fixture": f"{a.name}_{v}"} for v in a.variants})
    fx.results["oracle"] = {"device": "cpu",
                            "threads": torch.get_num_threads(),
                            "dequant": a.dequant}
    print(f"[check7] wrote {fx.write()}", flush=True)
    for variant in a.variants:
        alpha, strength, dialects = VARIANTS[variant]
        models = {m: merge(torch, load(m), [(pairs, alpha, strength)], m)
                  for m in oracle.MODES}
        note = {"source": a.source, "size": ins.get("size"),
                "variant": variant, "pairs_fixture": a.name, "merge": MERGE,
                "adapters": [{"name": a.name, "alpha": alpha,
                              "strength": strength, "targets": names}],
                "dialects": list(dialects)}
        path = c3.run_case(torch, a, models, cond, sigmas, timesteps, True,
                           f"{a.name}_{variant}", note, check=7,
                           script=SCRIPT)
        print(f"[check7] wrote {path}", flush=True)
        del models
        gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
