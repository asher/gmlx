#!/usr/bin/env python3
"""Check 3 oracle for Qwen-Image 2.1: one denoise step in both of the
reference's modes, block by block.

QwenImage21Transformer2DModel runs one conditioning through three calls,
each as QwenImage21Pipeline makes it at batch 1 for one unpadded prompt:

- joint: kv_cache_mode=None, one forward of the whole joint sequence,
  prefix and target, on latents a at sigma a;
- extract: the same call with a cache, as the pipeline's first step, which
  keeps each block's K and V of the prefix;
- cached: latents b at sigma b over that cache, the target rows alone.

The model timestep is t / 1000 for the scheduler's t, and the velocity is
the model output on the target rows, unpacked, with no sign flip.

For a size in --blocks the fixture holds, whole in float32: the inputs;
the embedder outputs; for the joint and for the cached call the input of
the first block and the output of every block, of its attention and of
its MLP, and the final projection; the cached K and V of every block; and
both velocities. For the other sizes it holds the inputs and the
velocities. Every fixture holds the timestep embedding and the shared
modulation of both sigmas, each with its t = 0 row, and the token layout
read back from the reference's own outputs: the position of each token
from its row of the rotary table, the source of each token from its row
of the first block's input, the image block ids, the target mask, the
modulation row each token reads and the prefill segments.

The bf16 model runs the same calls, and each of its blocks also runs
teacher-forced on the fp32 model's input to it. The manifest records the
bf16 against fp32 gap of every row and sets each tolerance to twice that
gap. In bf16 the reference rounds the timestep itself to bfloat16 before
it embeds it. --bf16-timestep fp32, the default, gives the sinusoid the
float32 timestep instead, so that the gap measures arithmetic alone.

The conditioning is --conditioning, an .npz of encoder features, their
image slot mask and reference latents, or with --caption random, seeded
normal features and reference latents, which lack the outliers of real
ones.

  python scripts/image_oracle/run.py --python <oracle python> --threads 16 \\
      check3_qwen_image21.py --dit qwen-image-2.1-Q4_K_M.gguf \\
      --config <official repo dir> --conditioning <npz>

--self-test writes a tiny model in GGUF tensor names, loads it back
through the same single-file path, and writes its fixtures and the
layout fixture, so the gmlx side can check its conventions on the CPU.
qwen_image21_tiny.py holds the tiny model's rule.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle
import qwen_image21_tiny as tiny

CLS = "QwenImage21Transformer2DModel"
SCRIPT = "scripts/image_oracle/check3_qwen_image21.py"
STARTING = {"block_rel_l2": 5e-3, "velocity_rel_l2": 2e-2}
RULE = ("tolerance = 2 x the oracle's own bf16 against fp32 relative L2: "
        "teacher-forced for each row, free running for each block of each "
        "call and for each velocity. A velocity's cosine floor is 1 - 4 x "
        "the oracle's own cosine distance. Each block also has two "
        "teacher-forced branch rows, the output of its attention and of "
        "its MLP. Their fp32 arrays serve a gate that runs the model in "
        "fp32, and 2 x the oracle's own gap on a branch row is recorded "
        "and is not a gate. A gate that runs the model in the oracle's "
        "fp32 arithmetic holds every row to a fixed bound instead")
CALLS = ("joint", "cached")
SINGLE = ("txt_in", "img_in", "time_text_embed", "modulation", "pos_embed",
          "proj_out")
BRANCHES = ("attn", "img_mlp")


def chain(model) -> list[tuple[str, object]]:
    return [(f"transformer_blocks.{i}", b)
            for i, b in enumerate(model.transformer_blocks)]


class Capture:
    """Outputs of the embedders, the timestep path, the rotary table and
    the final projection, and the keyword arguments and the output of
    every block, with the outputs of its attention and its MLP."""

    def __init__(self, model):
        self.out: dict[str, object] = {}
        self.kwargs: dict[str, dict] = {}
        self._hooks = []
        for name in SINGLE:
            self._hooks.append(getattr(model, name).register_forward_hook(
                self._post(name)))
        for name, blk in chain(model):
            self._hooks.append(blk.register_forward_pre_hook(
                self._pre(name), with_kwargs=True))
            self._hooks.append(blk.register_forward_hook(self._post(name)))
            for sub in BRANCHES:
                self._hooks.append(getattr(blk, sub).register_forward_hook(
                    self._post(f"{name}.{sub}")))

    def _pre(self, name):
        def hook(_mod, args, kwargs):
            if args:
                raise SystemExit(f"error: {name} took positional arguments")
            self.kwargs[name] = dict(kwargs)
        return hook

    def _post(self, name):
        def hook(_mod, _args, out):
            self.out[name] = out
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()


@contextlib.contextmanager
def float32_timestep(torch, model, t32):
    """Give the sinusoid the float32 timestep ``t32`` (B,), with its
    t = 0 row, whatever dtype the forward cast the timestep to."""
    rows = torch.cat([t32.float(), t32.new_zeros(1).float()], dim=0)
    hook = model.time_text_embed.time_proj.register_forward_pre_hook(
        lambda _mod, _args: (rows,))
    try:
        yield
    finally:
        hook.remove()


def call(torch, model, cond: dict, lat: np.ndarray, t, mode: str, a,
         cache=None, kv_mode=None):
    """One transformer call as QwenImage21Pipeline makes it, at batch 1.
    Returns the output rows the call computes, (1, S or T, C)."""
    dt = oracle.torch_dtype(mode)

    def pack(z):
        return torch.from_numpy(z).reshape(z.shape[0], -1).transpose(0, 1)

    hidden = torch.cat([pack(z) for z in (*cond["ref_latents"], lat[0])],
                       dim=0)[None].to(dt)
    feats = torch.from_numpy(cond["features"])[None].to(dt)
    tokens = lat.shape[2] * lat.shape[3]
    mask = torch.from_numpy(cond["img_mask"])[None]
    mask = torch.cat([mask, mask.new_ones(1, tokens // 4)], dim=1)
    shapes = [[*[(1, h, w) for h, w in cond["ref_shapes"]],
               (1, lat.shape[2], lat.shape[3])]]
    t32 = t.expand(1).float() / 1000
    timestep = t.expand(1).to(dt) / 1000
    exact = mode == "bf16" and a.bf16_timestep == "fp32"
    with (float32_timestep(torch, model, t32) if exact
          else contextlib.nullcontext()):
        return model(hidden_states=hidden, timestep=timestep,
                     encoder_hidden_states=feats, img_shapes=shapes,
                     img_mask=mask, kv_cache=cache, kv_cache_mode=kv_mode,
                     return_dict=False)[0]


def velocity(out, lat: np.ndarray) -> np.ndarray:
    """The target rows of a call's output as (1, C, H, W) float32."""
    _, c, h, w = lat.shape
    v = out[:, -h * w:].float()
    return oracle.to_numpy(v.transpose(1, 2).reshape(1, c, h, w))


def run_calls(torch, model, cond, lat_a, lat_b, t_a, t_b, mode, a,
              capture: bool):
    """The three calls on one model. Returns the two velocities, the cache
    and, with ``capture``, the capture of the joint and the cached call."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache)

    caps = {}
    c = Capture(model) if capture else None
    joint = call(torch, model, cond, lat_a, t_a, mode, a)
    if c is not None:
        c.remove()
        caps["joint"] = c
    cache = QwenImage21KVCache(len(model.transformer_blocks))
    extract = call(torch, model, cond, lat_a, t_a, mode, a, cache, "extract")
    gap = float((extract.float() - joint.float()).abs().max())
    c = Capture(model) if capture else None
    cached = call(torch, model, cond, lat_b, t_b, mode, a, cache, "cached")
    if c is not None:
        c.remove()
        caps["cached"] = c
    return (velocity(joint, lat_a), velocity(cached, lat_b), cache, caps,
            gap)


def _rows_of(table) -> dict:
    """Row bytes to row index, for a table whose rows are distinct."""
    keys = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(table))}
    if len(keys) != len(table):
        raise SystemExit("error: a lookup table has equal rows")
    return keys


def token_layout(torch, model, cap: Capture, cond: dict) -> dict:
    """The joint sequence's layout, read back from the reference's
    outputs of the joint call."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        _select_modulation_rows)

    first = cap.kwargs["transformer_blocks.0"]
    rot = first["rotary_emb"].numpy()
    at = 0
    pos = np.zeros((rot.shape[0], 3), np.int64)
    for axis, table in enumerate(model.pos_embed.freqs):
        table = table.numpy()
        keys = _rows_of(table)
        width = table.shape[1]
        part = np.ascontiguousarray(rot[:, at:at + width])
        idx = np.asarray([keys[r.tobytes()] for r in part])
        # The table holds positions 0 to 8191 and then -1024 to -1.
        pos[:, axis] = np.where(idx < 8192, idx, idx - table.shape[0])
        at += width
    txt = oracle.to_numpy(cap.out["txt_in"])[0]
    img = oracle.to_numpy(cap.out["img_in"])[0]
    keys = _rows_of(np.concatenate([txt, img], axis=0))
    joint_in = oracle.to_numpy(first["hidden_states"])[0]
    source = np.asarray([keys[np.ascontiguousarray(r).tobytes()]
                         for r in joint_in])
    image = source >= txt.shape[0]
    shapes = [*[(1, h, w) for h, w in cond["ref_shapes"]],
              (1, *cond["target"])]
    ids, target = model.build_token_metadata(torch.from_numpy(image), shapes)
    seen = first["target_token_mask"]
    if not torch.equal(seen, target):
        raise SystemExit("error: the target mask of the call differs")
    row = _select_modulation_rows(torch.tensor([[0.0], [1.0]]), target)
    return {
        "positions": pos.astype(np.int32),
        "image_pad_mask": image,
        "joint_source": source.astype(np.int32),
        "image_ids": ids.numpy().astype(np.int32),
        "target_token_mask": target.numpy(),
        "modulation_row": row.numpy().reshape(-1).astype(np.int8),
        "segments": np.asarray(first["segments"], np.int32).reshape(-1, 3),
    }


def _bf16(torch, v):
    if torch.is_tensor(v) and v.is_floating_point():
        return v.to(torch.bfloat16)
    return v


def teacher_forced(torch, model16, cap32: Capture, cache32) -> dict:
    """Output of every bf16 block fed the fp32 model's input to it, with
    the outputs of its attention and MLP. A cached block reads a bf16
    copy of the fp32 K and V."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVLayerCache)

    outs = {}
    for i, (name, blk) in enumerate(chain(model16)):
        kwargs = {k: _bf16(torch, v)
                  for k, v in cap32.kwargs[name].items()}
        if kwargs.get("layer_cache") is not None:
            k, v = cache32.get_layer(i).get()
            layer = QwenImage21KVLayerCache()
            layer.store(k.to(torch.bfloat16), v.to(torch.bfloat16))
            kwargs["layer_cache"] = layer
        got = {}
        hooks = [getattr(blk, sub).register_forward_hook(
            lambda _m, _a, out, key=f"{name}.{sub}": got.update({key: out}))
            for sub in BRANCHES]
        outs[name] = blk(**kwargs)
        for h in hooks:
            h.remove()
        outs.update(got)
    return outs


def run_case(torch, a, models: dict, cond: dict, sigmas, timesteps,
             blocks: bool, name: str, family_note: dict, check: int = 3,
             script: str = SCRIPT) -> Path:
    """Run the calls on ``models``, fp32 and bf16, and write the fixture.
    Check 7 passes models with an adapter merged and its own script."""
    h, w = cond["target"]
    c_in = models["fp32"].config.in_channels
    rng = np.random.default_rng(a.seed)
    lat_a = cond.get("latents_a")
    lat_b = cond.get("latents_b")
    if lat_a is None:
        lat_a = rng.standard_normal((1, c_in, h, w)).astype(np.float32)
        lat_b = rng.standard_normal((1, c_in, h, w)).astype(np.float32)
    t_a, t_b = timesteps[a.step], timesteps[a.step_b]

    fx = fixtures.FixtureWriter(name, check=check, family="qwen_image21",
                                script=script)
    for role, p in (("dit", a.dit), ("conditioning", a.conditioning)):
        if p is not None:
            fx.add_file(role, p)
    fx.inputs.update(
        target_tokens=[h, w], latent_shape=list(lat_a.shape),
        ref_shapes=[list(s) for s in cond["ref_shapes"]],
        image_slots=np.flatnonzero(cond["img_mask"]).tolist(),
        noise_seed=a.seed, steps=a.steps, step_a=a.step, step_b=a.step_b,
        sigmas=[float(s) for s in sigmas],
        sigma_a=float(sigmas[a.step]), sigma_b=float(sigmas[a.step_b]),
        blocks=blocks, config_revision=a.config_revision, **family_note)
    fx.results["oracle"] = {
        "device": "cpu", "threads": torch.get_num_threads(),
        "dequant": oracle.dequant_mode(), "matmul": {},
        "bf16_timestep": a.bf16_timestep,
        "encoder_hidden_states_mask": None}
    fx.add("features", cond["features"])
    fx.add("img_mask", cond["img_mask"])
    for i, z in enumerate(cond["ref_latents"]):
        fx.add(f"ref_latents.{i}", z)
    fx.add("latents_a", lat_a)
    fx.add("latents_b", lat_b)
    for tag, t, s in (("a", t_a, sigmas[a.step]), ("b", t_b,
                                                   sigmas[a.step_b])):
        fx.add(f"sigma_{tag}", oracle.to_numpy(s))
        passed = t.expand(1).float() / 1000
        fx.add(f"t_passed_{tag}", oracle.to_numpy(passed))
        fx.add(f"t_embedded_{tag}", oracle.to_numpy(1000.0 * passed))

    times, vel, caps, caches = {}, {}, {}, {}
    for mode in oracle.MODES:
        oracle.emulate_bf16_matmul(mode == "bf16")
        fx.results["oracle"]["matmul"][mode] = oracle.matmul_mode()
        t0 = time.perf_counter()
        va, vb, caches[mode], caps[mode], gap = run_calls(
            torch, models[mode], cond, lat_a, lat_b, t_a, t_b, mode, a,
            capture=blocks or mode == "fp32")
        times[f"{mode}_calls_s"] = round(time.perf_counter() - t0, 1)
        vel[mode] = {"a": va, "b": vb}
        fx.results[f"extract_against_joint_max_abs_{mode}"] = gap
    oracle.emulate_bf16_matmul(False)

    c32 = caps["fp32"]
    for key, arr in token_layout(torch, models["fp32"], c32["joint"],
                                 cond).items():
        fx.add(key, arr)
    for tag in ("a", "b"):
        fx.add(f"velocity_{tag}", vel["fp32"][tag])
        fx.add(f"velocity_{tag}_bf16", vel["bf16"][tag])
        gap_v = fixtures.compare(vel["bf16"][tag], vel["fp32"][tag])
        fx.results[f"velocity_{tag}_bf16_vs_fp32"] = gap_v
        fx.tolerances[f"velocity_{tag}_rel_l2"] = 2 * gap_v["rel_l2"]
        fx.tolerances[f"velocity_{tag}_cosine_min"] = (
            1 - 4 * (1 - gap_v["cosine"]))
    fx.tolerances.update(rule=RULE, starting=STARTING)
    for tag, which in (("a", "joint"), ("b", "cached")):
        fx.add(f"temb_{tag}", oracle.to_numpy(
            c32[which].out["time_text_embed"]))
        fx.add(f"modulation_{tag}", oracle.to_numpy(
            c32[which].out["modulation"]))

    if blocks:
        names = [n for n, _ in chain(models["fp32"])]
        rot = c32["joint"].out["pos_embed"]
        fx.add("rope_cos", oracle.to_numpy(rot.real))
        fx.add("rope_sin", oracle.to_numpy(rot.imag))
        fx.add("txt_in.out", oracle.to_numpy(c32["joint"].out["txt_in"]))
        fx.add("img_in.out", oracle.to_numpy(c32["joint"].out["img_in"]))
        for which in CALLS:
            c = c32[which]
            for prev, nxt in zip(names, names[1:]):
                assert c.kwargs[nxt]["hidden_states"] is c.out[prev], nxt
            fx.add(f"{which}.in", oracle.to_numpy(
                c.kwargs[names[0]]["hidden_states"]))
            for n in names:
                fx.add(f"{which}.{n}.out", oracle.to_numpy(c.out[n]))
                for sub in BRANCHES:
                    fx.add(f"{which}.{n}.{sub}.out",
                           oracle.to_numpy(c.out[f"{n}.{sub}"]))
            fx.add(f"{which}.proj_out.out",
                   oracle.to_numpy(c.out["proj_out"]))
        for i in range(len(names)):
            k, v = caches["fp32"].get_layer(i).get()
            fx.add(f"cache.k.{i}", oracle.to_numpy(k))
            fx.add(f"cache.v.{i}", oracle.to_numpy(v))

        oracle.emulate_bf16_matmul(True)
        t0 = time.perf_counter()
        rows, free, branches = {}, {}, {}
        c16 = caps["bf16"]
        for which in CALLS:
            tf = teacher_forced(torch, models["bf16"], c32[which],
                                caches["fp32"])
            for n in names:
                rows[f"{which}.{n}"] = fixtures.compare(
                    oracle.to_numpy(tf[n]), oracle.to_numpy(c32[which].out[n]))
                free[f"{which}.{n}"] = fixtures.compare(
                    oracle.to_numpy(c16[which].out[n]),
                    oracle.to_numpy(c32[which].out[n]))
                for sub in BRANCHES:
                    key = f"{n}.{sub}"
                    branches[f"{which}.{key}"] = fixtures.compare(
                        oracle.to_numpy(tf[key]),
                        oracle.to_numpy(c32[which].out[key]))
            for single in ("time_text_embed", "modulation", "proj_out"):
                free[f"{which}.{single}"] = fixtures.compare(
                    oracle.to_numpy(c16[which].out[single]),
                    oracle.to_numpy(c32[which].out[single]))
        for single in ("txt_in", "img_in"):
            rows[single] = fixtures.compare(
                oracle.to_numpy(c16["joint"].out[single]),
                oracle.to_numpy(c32["joint"].out[single]))
        for i in range(len(names)):
            for part, j in (("k", 0), ("v", 1)):
                free[f"cache.{part}.{i}"] = fixtures.compare(
                    oracle.to_numpy(caches["bf16"].get_layer(i).get()[j]),
                    oracle.to_numpy(caches["fp32"].get_layer(i).get()[j]))
        oracle.emulate_bf16_matmul(False)
        times["bf16_teacher_forced_s"] = round(time.perf_counter() - t0, 1)
        fx.results["rows_bf16_vs_fp32_teacher_forced"] = rows
        fx.results["branches_bf16_vs_fp32_teacher_forced"] = branches
        fx.results["free_bf16_vs_fp32"] = free
        fx.tolerances["rows_rel_l2"] = {r: 2 * g["rel_l2"]
                                        for r, g in rows.items()}
        fx.tolerances["free_rel_l2"] = {r: 2 * g["rel_l2"]
                                        for r, g in free.items()}
        fx.tolerances["branch_rel_l2"] = {r: 2 * g["rel_l2"]
                                          for r, g in branches.items()}
        fx.inputs["rows"] = list(rows)
        fx.inputs["free_rows"] = list(free)
        fx.inputs["branches"] = list(branches)
    fx.results["seconds"] = times
    fx.note("The pipeline passes a prompt mask of ones for one unpadded "
            "prompt. These calls pass none, which gives the same values.")
    path = fx.write()
    print(f"[check{check}] {name}: velocity bf16 vs fp32 rel L2 "
          f"a {fx.results['velocity_a_bf16_vs_fp32']['rel_l2']:.3e} "
          f"b {fx.results['velocity_b_bf16_vs_fp32']['rel_l2']:.3e}, "
          f"extract against joint max abs "
          f"{fx.results['extract_against_joint_max_abs_fp32']:.1e}, {times}")
    return path


def schedule(torch, config_dir: Path, steps: int, tokens: int):
    """Sigmas and timesteps as QwenImage21Pipeline sets them."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_shift, retrieve_timesteps)

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(config_dir), subfolder="scheduler", local_files_only=True)
    c = sched.config
    mu = calculate_shift(tokens, c.get("base_image_seq_len", 256),
                         c.get("max_image_seq_len", 4096),
                         c.get("base_shift", 0.5), c.get("max_shift", 1.15))
    timesteps, _ = retrieve_timesteps(
        sched, steps, torch.device("cpu"),
        sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
    return sched.sigmas.clone(), timesteps.clone()


def read_conditioning(a, c_in: int, ctx: int) -> tuple[dict, dict]:
    """The conditioning of a run, and what the manifest says of it."""
    if a.caption == "random":
        rng = np.random.default_rng(a.seed + 1)
        refs = [tuple(int(v) for v in r.lower().split("x"))
                for r in a.random_refs]
        slots = [h * w // 4 for h, w in refs]
        n_rows = a.random_len + sum(slots)
        mask = np.zeros(n_rows, bool)
        at = a.random_len // 2
        for n in slots:
            mask[at:at + n] = True
            at += n + 1
        return ({
            "features": rng.standard_normal((n_rows, ctx)).astype(np.float32),
            "img_mask": mask, "ref_shapes": refs,
            "ref_latents": [rng.standard_normal((c_in, h, w)).astype(
                np.float32) for h, w in refs]},
            {"conditioning": "random", "generator": "numpy default_rng("
             "seed + 1) standard_normal float32", "rows": n_rows})
    with np.load(a.conditioning, allow_pickle=False) as z:
        refs = [z[k] for k in sorted(
            (k for k in z.files if k.startswith("ref_latents.")),
            key=lambda k: int(k.rsplit(".", 1)[1]))]
        cond = {"features": z["features"].astype(np.float32),
                "img_mask": z["img_mask"].astype(bool),
                "ref_shapes": [tuple(r.shape[-2:]) for r in refs],
                "ref_latents": [r.astype(np.float32) for r in refs]}
    return cond, {"conditioning": "file",
                  "rows": int(cond["features"].shape[0])}


# Self-test: the tiny model of qwen_image21_tiny.py.

def write_gguf(path: Path, tensors: dict) -> None:
    import gguf

    w = gguf.GGUFWriter(str(path), "qwen_image")
    for name, arr in tensors.items():
        w.add_tensor(name, np.ascontiguousarray(arr, np.float32),
                     raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tiny_models(torch, root: Path) -> tuple[dict, dict]:
    """The tiny model in fp32 and bf16, loaded from a GGUF in GGUF tensor
    names through the single-file path, and its weights by name."""
    import diffusers

    work = root / "selftest_qwen_image21_tiny_model"
    (work / "transformer").mkdir(parents=True, exist_ok=True)
    weights = tiny.weights()
    write_gguf(work / "tiny.gguf",
               {tiny.PREFIX + k: v for k, v in weights.items()})
    config = {"_class_name": CLS,
              "_diffusers_version": diffusers.__version__, **tiny.TINY}
    (work / "transformer" / "config.json").write_text(
        json.dumps(config, indent=1) + "\n")
    models = {m: oracle.load_dit(work / "tiny.gguf", work, m, cls_name=CLS)
              for m in oracle.MODES}
    # The loaded model equals the rule's weights under the reference's own
    # names, with each fused gate_up split gate rows first.
    want = tiny.unfuse(weights)
    state = {k: oracle.to_numpy(v)
             for k, v in models["fp32"].state_dict().items()}
    if sorted(state) != sorted(want):
        raise SystemExit("error: the loaded tiny model has other names")
    for k, v in want.items():
        if not np.array_equal(state[k], v):
            raise SystemExit(f"error: {k} differs after the load")
    return models, weights


def tiny_cond(case: str, target) -> dict:
    return {"features": tiny.features(case), "img_mask": tiny.image_mask(case),
            "ref_shapes": [tuple(s) for s in tiny.CASES[case][2]],
            "ref_latents": list(tiny.ref_latents(case)),
            "target": tuple(target),
            "latents_a": tiny.latents(0, 1, target),
            "latents_b": tiny.latents(1, 1, target)}


# case: (target tokens, whether the fixture keeps every block tensor)
SELF_CASES = {"text": (tiny.TARGET, False), "ref1": (tiny.TARGET, False),
              "ref2": ((2, 4), True)}
LAYOUT_CASES = {"text": tiny.TARGET, "ref1": tiny.TARGET, "ref2": (2, 4),
                "adjacent": tiny.TARGET, "wide": (12, 20)}
FREQ_DIMS = (4, 6, 16, 56)


def self_test(torch, a) -> int:
    root = fixtures.fixture_root(create=True)
    models, weights = tiny_models(torch, root)
    a.dit = a.conditioning = a.config_revision = None
    sig = torch.linspace(1.0, 1 / a.steps, a.steps)
    note = {"config": tiny.TINY, "weights": "qwen_image21_tiny.weights()",
            "weights_sha256": tiny.sha256(weights),
            "inputs": "qwen_image21_tiny features, ref_latents and latents",
            "loaded": "from a GGUF in GGUF tensor names through "
                      "from_single_file"}
    for case, (target, blocks) in SELF_CASES.items():
        path = run_case(torch, a, models, tiny_cond(case, target), sig,
                        sig * 1000, blocks,
                        f"selftest_qwen_image21_tiny_{case}",
                        {**note, "case": case})
        print(f"[check3] wrote {path}", flush=True)

    # The layout of more cases, and the rotary frequencies.
    fx = fixtures.FixtureWriter("selftest_qwen_image21_layout", check=3,
                                family="qwen_image21", script=SCRIPT)
    oracle.emulate_bf16_matmul(False)
    for case, target in LAYOUT_CASES.items():
        cond = tiny_cond(case, target)
        cap = Capture(models["fp32"])
        call(torch, models["fp32"], cond, cond["latents_a"], sig[1] * 1000,
             "fp32", a)
        cap.remove()
        for key, arr in token_layout(torch, models["fp32"], cap,
                                     cond).items():
            fx.add(f"{case}.{key}", arr)
    rope = models["fp32"].pos_embed
    for d in FREQ_DIMS:
        fx.add(f"rope_freqs.{d}", oracle.to_numpy(1.0 / torch.pow(
            rope.theta, torch.arange(0, d, 2).to(torch.float32).div(d))))
    fx.inputs.update(
        cases={c: {"rows": tiny.CASES[c][0],
                   "image_slots": list(tiny.CASES[c][1]),
                   "ref_shapes": [list(s) for s in tiny.CASES[c][2]],
                   "target_tokens": list(t)}
               for c, t in LAYOUT_CASES.items()},
        theta=rope.theta, freq_dims=list(FREQ_DIMS), config=tiny.TINY)
    fx.tolerances.update(layout="exact", rope_freqs="one ulp")
    fx.note("positions come from each token's row of the reference's "
            "rotary table, joint_source from each token's row of the first "
            "block's input, image_ids and target_token_mask from "
            "build_token_metadata, and modulation_row from "
            "_select_modulation_rows: 0 reads the sampled timestep's row "
            "and 1 the t = 0 row. joint_source indexes the encoder rows "
            "followed by the image tokens, references first.")
    print(f"[check3] wrote {fx.write()}", flush=True)
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
    ap.add_argument("--conditioning", type=Path,
                    help=".npz of features (L, D), img_mask (L,) and "
                         "ref_latents.N (C, h, w)")
    ap.add_argument("--caption", choices=("file", "random"), default="file")
    ap.add_argument("--random-len", type=int, default=64,
                    help="text rows of a random conditioning")
    ap.add_argument("--random-refs", nargs="*", default=[],
                    help="reference shapes of a random conditioning in "
                         "latent tokens, as HxW")
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--blocks", type=int, nargs="*", default=[512],
                    help="sizes whose fixture keeps every block tensor")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step", type=int, default=12,
                    help="index of the step of the joint call's sigma")
    ap.add_argument("--step-b", type=int, default=13,
                    help="index of the step of the cached call's sigma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16-timestep", choices=("fp32", "stock"),
                    default="fp32")
    ap.add_argument("--dequant", choices=oracle.DEQUANT, default="exact")
    ap.add_argument("--name", default="check3_qwen_image21",
                    help="fixture name prefix; the size is appended")
    ap.add_argument("--self-test", action="store_true",
                    help="write the tiny model's fixtures and stop")
    a = ap.parse_args(argv)
    torch = oracle.setup()
    if a.self_test:
        a.steps, a.step, a.step_b = 8, 3, 5
        return self_test(torch, a)
    if not (a.dit and a.config and (a.conditioning
                                    or a.caption == "random")):
        ap.error("--dit, --config and --conditioning are required")
    if any(s % 32 for s in a.sizes):
        ap.error("sizes must be multiples of 32")
    models = {m: oracle.load_dit(a.dit, a.config, m, cls_name=CLS,
                                 dequant=a.dequant) for m in oracle.MODES}
    for m, model in models.items():
        where = {str(p.device) for p in model.parameters()}
        if where != {"cpu"}:
            raise SystemExit(f"error: {m} DiT on {where}")
    cfg = models["fp32"].config
    cond, meta = read_conditioning(a, cfg.in_channels, cfg.context_in_dim)
    for size in a.sizes:
        side = size // 16
        cond["target"] = (side, side)
        sigmas, timesteps = schedule(torch, a.config, a.steps, side * side)
        path = run_case(torch, a, models, cond, sigmas, timesteps,
                        size in a.blocks, f"{a.name}_{size}",
                        {**meta, "size": [size, size]})
        print(f"[check3] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
