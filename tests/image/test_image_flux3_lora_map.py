"""The FLUX.3 klein LoRA key map and the delta half of acceptance check 7:
every spelling of every target, in the GGUF names and in the diffusers
names, resolves to its leaf, the placed factors equal a float64
reference per dialect, both halves of ``mlp_in`` among them and the
final modulation of the diffusers layout with its halves swapped back,
and a step of the tiny model under a patch table equals the step on
merged weights. CPU."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import flux3_gates as fg
from image_fixtures import fixture_view, load_json
from lora_fixtures import (DIALECTS, dialect_keys, make_pair,
                           reference_delta, rel_l2, write_adapter,
                           write_keys)
from oracle_fixture import committed_fixture

from gmlx.image import api, lora
from gmlx.image import fingerprint as fp
from gmlx.image.families.flux3_klein.lora_map import lora_map
from gmlx.image.families.flux3_klein.model import Flux3Config, Flux3KleinDiT
from gmlx.image.lora import install, plan, reader
from gmlx.image.registry import get_family

DELTA_TOL = 1e-6
# The live step against merged fp32 weights on the CPU, relative L2 of
# each row of the step.
STEP_TOL = 2e-5
HEADERS = {"flux3_klein_q4_k_m.header.json": True,
           "flux3_klein_abiray_q4_k_m.header.json": True,
           "flux3_klein_base_q4_k_m.header.json": False}
# The tensor names of Flux3Transformer2DModel, from the official index.
DIFFUSERS_NAMES = "flux3_klein_diffusers_names.json"
# The GGUF names under the model prefixes that ComfyUI and the diffusers
# loader strip, and under the underscore prefixes of kohya and LyCORIS.
SPELLINGS = {
    "comfy": ("diffusion_model.", False, ".lora_A.weight", ".lora_B.weight",
              False),
    "comfy_model": ("model.diffusion_model.", False, ".lora_down.weight",
                    ".lora_up.weight", True),
    "lycoris": ("lycoris_", True, ".lora_down.weight", ".lora_up.weight",
                True),
    "transformer_kohya": ("lora_transformer_", True, ".lora_down.weight",
                          ".lora_up.weight", True),
}
TINY = "selftest_flux3_klein_tiny"
ADALN = fg.ADALN
diffusers_name = fg.diffusers_name
swap_halves = fg.swap_halves


def tiny_config(flow_map: bool = True) -> dict:
    _, wm = committed_fixture(TINY + "_weights")
    return dict(wm["inputs"]["config"], flow_map=flow_map)


def dims_of(cfg: dict) -> dict:
    """Module path to (out, in) of every linear leaf of the built
    model."""
    return {p: tuple(m.weight.shape)
            for p, m in Flux3KleinDiT(cfg).named_modules()
            if isinstance(m, nn.Linear)}


def mlp_dim(cfg: dict) -> int:
    return Flux3Config.from_dict(cfg).mlp_dim


def gguf_delta(name: str, delta: np.ndarray) -> np.ndarray:
    """A delta of the layer that ``name`` names, in the GGUF's row order:
    the merged diffusers weight turned back by the converter's swap, for
    the diffusers final modulation."""
    return swap_halves(delta) if name == "norm_out.linear" else delta


@pytest.mark.parametrize("flow_map", [True, False])
def test_the_key_map_covers_every_linear_leaf_and_only_those(flow_map):
    cfg = tiny_config(flow_map)
    km = lora_map(cfg)
    dims = dims_of(cfg)
    assert km.family == "flux3_klein" and km.title == "FLUX.3 klein"
    assert {m for m, _ in km.targets.values()} == set(dims)
    out = [p for p in dims if p.endswith(".attn_out")]
    for path in dims:
        assert km.targets[path] == (path, None)
        assert km.targets[diffusers_name(path)] == (path, None)
    for path in out:
        assert km.targets[diffusers_name(path)[:-len(".0")]] == (path, None)
    assert len(km.targets) == 2 * len(dims) + len(out)
    # The blocks of two content modes, the text mode and the single
    # stream, of six leaves each, and the leaves outside the blocks.
    blocks = 3 * cfg["num_layers"] + cfg["num_single_layers"]
    assert len(dims) == 6 * blocks + (15 if flow_map else 13)
    assert km.diffusers_only == frozenset(
        [diffusers_name(p) for p in dims]
        + [diffusers_name(p)[:-len(".0")] for p in out])
    assert km.swapped == frozenset({"norm_out.linear"})
    ref = {"ref_time_in.in_layer", "ref_time_in.out_layer"}
    qkv = {p[:-len(".q_proj")] + ".qkv" for p in dims
           if p.endswith(".q_proj")}
    assert len(qkv) == blocks
    if flow_map:
        assert ref <= set(km.targets) and set(km.absent) == qkv
    else:
        refs = ref | {diffusers_name(p) for p in ref}
        assert not refs & set(km.targets)
        assert set(km.absent) == qkv | refs
    built = dataclasses.asdict(Flux3Config.from_dict(cfg))
    assert lora_map(built) == km
    assert get_family("flux3_klein").lora_map(cfg) == km
    assert len({n.replace(".", "_") for n in km.targets}) == len(km.targets)
    assert km.text_names == () and km.text_targets == ()


@pytest.mark.parametrize("name", sorted(HEADERS))
def test_the_key_map_names_every_linear_weight_of_the_real_header(name):
    ident = fp.identify(fixture_view(name))
    km = lora_map(ident.config)
    assert ident.family.name == "flux3_klein"
    linear = {n[:-len(".weight")] for n, t in ident.view.tensors.items()
              if n.endswith(".weight") and len(t.shape) == 2}
    assert linear == {m for m, _ in km.targets.values()}
    # 5 blocks in each of three modes and 28 single blocks of 6 leaves,
    # and 15 others, 13 in Base.
    assert len(linear) == 6 * (3 * 5 + 28) + (15 if HEADERS[name] else 13)
    assert km.targets["single_blocks.27.mlp_in"] == (
        "single_blocks.27.mlp_in", None)
    assert km.targets["single_transformer_blocks.27.ff.linear_in"] == (
        "single_blocks.27.mlp_in", None)
    assert "single_blocks.28.mlp_in" not in km.targets
    assert "content_mode_blocks.image_cond.5.q_proj" not in km.targets


def test_the_key_map_names_every_linear_weight_of_the_diffusers_model():
    """Every weight of Flux3Transformer2DModel that is a matrix, which is
    every one but the q and k norms, names a target, one per module of
    the distilled GGUF."""
    names = load_json(DIFFUSERS_NAMES)["names"]
    assert len(names) == 359
    ident = fp.identify(fixture_view("flux3_klein_q4_k_m.header.json"))
    km = lora_map(ident.config)
    norms = [n for n in names if n.endswith((".norm_q.weight",
                                             ".norm_k.weight"))]
    linear = [n[:-len(".weight")] for n in names if n not in norms]
    assert len(norms) == 2 * (3 * 5 + 28)
    assert all(n.endswith(".weight") for n in names)
    assert all(n in km.targets for n in linear), \
        [n for n in linear if n not in km.targets][:4]
    modules = [km.targets[n][0] for n in linear]
    gguf = {n[:-len(".weight")] for n, t in ident.view.tensors.items()
            if n.endswith(".weight") and len(t.shape) == 2}
    assert len(set(modules)) == len(modules) and set(modules) == gguf
    assert {diffusers_name(m) for m in modules} == set(linear)


def _stem(row: tuple, target: str) -> str:
    prefix, under, _, _, _ = row
    return prefix + (target.replace(".", "_") if under else target)


SPELLED = [*(f"{d} diffusers" for d in sorted(DIALECTS)),
           *(f"{d} gguf" for d in sorted(DIALECTS)), *sorted(SPELLINGS)]


@pytest.mark.parametrize("case", SPELLED)
def test_every_spelling_of_every_target_resolves(tmp_path, case):
    """One pair on every linear layer, in the diffusers names or the GGUF
    names as each dialect spells them, lands on its leaf. The pair on the
    diffusers final modulation lands with its up halves swapped."""
    dialect, _, names = case.partition(" ")
    names = names or "gguf"
    row = DIALECTS.get(dialect) or SPELLINGS[dialect]
    cfg = tiny_config()
    rng = np.random.default_rng(4)
    dims = dims_of(cfg)
    by_path = {p: make_pair(rng, o, i, 2, 4.0 if row[4] else None)
               for p, (o, i) in dims.items()}
    named = {(diffusers_name(p) if names == "diffusers" else p): v
             for p, v in by_path.items()}
    _, _, down_s, up_s, _ = row
    keys = {}
    for target, (down, up, alpha) in named.items():
        stem = _stem(row, target)
        keys[stem + down_s], keys[stem + up_s] = down, up
        if alpha is not None:
            keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    if dialect in DIALECTS:
        assert keys.keys() == dialect_keys(dialect, named).keys()
    res = reader.resolve(reader.read_lora(
        write_keys(tmp_path / "all.safetensors", keys)), lora_map(cfg))
    assert res.text == () and res.unmatched == ()
    if names == "diffusers" and dialect == "kohya":
        # ComfyUI reads no diffusers name of FLUX.3.
        assert len(res.warnings) == 1
        assert "ComfyUI drops them" in res.warnings[0]
    else:
        assert res.warnings == ()
    assert len(res.dit) == len(dims)
    for t in res.dit:
        down, up, alpha = by_path[t.module]
        assert t.band is None
        want = swap_halves(up) if (names == "diffusers"
                                   and t.module == ADALN) else up
        assert np.array_equal(np.array(t.pair.down), down)
        assert np.array_equal(np.array(t.pair.up), want)
        assert t.pair.alpha == alpha


def _adapter(cfg: dict, seed: int, alpha, paths, names: str) -> dict:
    """Pairs on ``paths`` at ranks 2 to 4, by diffusers or GGUF name."""
    dims = dims_of(cfg)
    rng = np.random.default_rng(seed)
    name = diffusers_name if names == "diffusers" else (lambda p: p)
    return {name(p): make_pair(rng, *dims[p], 2 + i % 3, alpha)
            for i, p in enumerate(paths)}


def _placed(km, dims: dict, stack) -> tuple[dict, dict]:
    """The placed delta of each module of the (path, pairs, strength)
    rows of ``stack``, and its float64 reference in the GGUF's order."""
    got: dict = {}
    want: dict = {}
    for path, pairs, strength in stack:
        res = reader.resolve(reader.read_lora(str(path)), km)
        for t in res.dit:
            got.setdefault(t.module, []).append(plan.Part(
                t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
        for target, (down, up, alpha) in pairs.items():
            module, band = km.targets[target]
            ref = reference_delta([(down, up, alpha, band, strength)],
                                  dims[module])
            want[module] = want.get(module, 0) + gguf_delta(target, ref)
    placed = {}
    for module, parts in got.items():
        lm = plan.block_diagonal(module, parts, dims[module])
        placed[module] = (np.array(lm.b, np.float64)
                          @ np.array(lm.a, np.float64))
    return placed, want


@pytest.mark.parametrize("names", ["diffusers", "gguf"])
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_delta_half_per_dialect(tmp_path, dialect, names):
    """Strengths 0.5 and 1.0 and a stacked pair with a negative strength,
    against float64, over the whole weight and over each half of every
    ``mlp_in``: the gate rows, then the up rows. A dialect that stores no
    alpha takes the factor 1.0. The diffusers final modulation is the
    merged diffusers weight turned back to the GGUF's order."""
    cfg = tiny_config()
    km, dims, half = lora_map(cfg), dims_of(cfg), mlp_dim(cfg)
    has_alpha = DIALECTS[dialect][4]
    paths = sorted(dims)
    # Every mlp_in and the final modulation take a pair of both adapters.
    both = {p for p in paths if p.endswith(".mlp_in")} | {ADALN}
    one = _adapter(cfg, 1, 3.0 if has_alpha else None,
                   sorted(set(paths[::2]) | both), names)
    two = _adapter(cfg, 2, 1.0 if has_alpha else None,
                   sorted(set(paths[::3]) | both), names)
    p1 = write_adapter(tmp_path / "one.safetensors", dialect, one)
    p2 = write_adapter(tmp_path / "two.safetensors", dialect, two)
    cases = [([(p1, one, 0.5)], "0.5"), ([(p1, one, 1.0)], "1.0"),
             ([(p1, one, 0.75), (p2, two, -0.4)], "stacked")]
    worst = 0.0
    for stack, label in cases:
        placed, want = _placed(km, dims, stack)
        assert set(placed) == set(want), label
        for module, delta in placed.items():
            ref = want[module]
            errs = [rel_l2(delta, ref)]
            if module.endswith(".mlp_in"):
                errs += [rel_l2(delta[:half], ref[:half]),
                         rel_l2(delta[half:], ref[half:])]
            worst = max(worst, *errs)
            assert max(errs) <= DELTA_TOL, (dialect, label, module, errs)
    print(f"flux3 check7 delta {dialect} {names}: worst relative L2 "
          f"{worst:.3e}")


def test_the_diffusers_final_modulation_is_swapped_back(tmp_path,
                                                        monkeypatch):
    """A pair on norm_out.linear that moves only its first half of rows,
    the scale rows in diffusers, moves only the scale rows of the GGUF's
    adaLN_modulation.1, its second half. The weight it gives is the one
    the diffusers loader gives its own model, turned back by the
    converter's swap. Without the swap it misses the delta gate."""
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    out, inp = dims[ADALN]
    h = out // 2
    rng = np.random.default_rng(9)
    w = rng.standard_normal((out, inp))
    down, up, _ = make_pair(rng, out, inp, 4)
    up[h:] = 0
    path = write_adapter(tmp_path / "n.safetensors", "diffusers",
                         {"norm_out.linear": (down, up, None)})

    def merged(keymap):
        placed, _ = _placed(keymap, dims, [(path, {}, 1.0)])
        return w + placed[ADALN]

    # The diffusers side: its converter's weight, the PEFT delta of the
    # pair on it, and the converter's swap back.
    diffusers = swap_halves(swap_halves(w) + up.astype(np.float64)
                            @ down.astype(np.float64))
    got = merged(km)
    assert rel_l2(got - w, diffusers - w) <= DELTA_TOL
    assert np.array_equal(got[:h], w[:h]) and not np.allclose(got[h:],
                                                              w[h:])
    unswapped = dataclasses.replace(km, swapped=frozenset())
    assert rel_l2(merged(unswapped) - w, diffusers - w) > 1.0


def _tiny_pipe(flow_map: bool, merged=None) -> api.ImagePipeline:
    """The tiny DiT in fp32, with (module, delta) rows added to its
    weights in float64 for ``merged``."""
    weights, _ = committed_fixture(TINY + "_weights")
    weights = dict(weights)
    for module, delta in (merged or {}).items():
        key = f"{module}.weight"
        weights[key] = (np.asarray(weights[key], np.float64)
                        + delta).astype(np.float32)
    model = fg.tiny_model(weights, tiny_config(flow_map), flow_map=flow_map)
    return api.ImagePipeline(dit=model, family=get_family("flux3_klein"),
                             variant="distilled" if flow_map else "base")


def _request(pipe, arrays: dict, tid) -> np.ndarray:
    refs = tuple(mx.array(arrays[f"refs.{i}"])[0]
                 for i in range(fg.n_refs(arrays)))
    # A key, so that the pipeline keeps the caption rows.
    cond = api.Conditioning(mx.array(arrays["features"]), "tiny",
                            ref_latents=refs)
    return np.array(api.denoise_step(
        pipe, mx.array(arrays["latents"]), arrays["sigma"], cond,
        patch_table=tid, target=arrays.get("target")))


@pytest.mark.parametrize("case", ["distilled", "base"])
def test_a_step_under_a_table_equals_the_step_on_merged_weights(tmp_path,
                                                                case):
    """A stacked pair of adapters over every linear leaf, one in the
    diffusers names and one in the GGUF names, applied live on the tiny
    model: every row of the step and the velocity, with two reference
    images on the distilled flow map and text alone on Base, and the
    request path, whose time vectors, modulations and caption rows a
    pipeline keeps per table."""
    flow_map = case == "distilled"
    arrays, _ = committed_fixture(TINY if flow_map else TINY + "_base")
    cfg = tiny_config(flow_map)
    km, dims = lora_map(cfg), dims_of(cfg)
    paths = sorted(dims)
    one = _adapter(cfg, 11, 2.0, paths, "diffusers")
    two = _adapter(cfg, 12, None, paths[1::2], "gguf")
    pipe = _tiny_pipe(flow_map)
    rows, stack = [], []
    for i, (pairs, strength, dialect) in enumerate(
            [(one, 0.6, "kohya"), (two, -0.3, "diffusers")]):
        path = write_adapter(tmp_path / f"a{i}.safetensors", dialect, pairs)
        rows.append((lora.load_adapter(pipe, path), strength))
        stack.append((path, pairs, strength))
    _, want = _placed(km, dims, stack)
    tid = api.define_patch_table(pipe, lora.table_entries(rows))
    merged = _tiny_pipe(flow_map, want)
    with mx.stream(mx.cpu):
        prev = install.activate(pipe, tid)
        try:
            got = fg.free_rows(pipe.dit, arrays)
        finally:
            install.restore(pipe, prev)
        ref = fg.free_rows(merged.dit, arrays)
        base = fg.free_rows(pipe.dit, arrays)
        # The base step first, so a memo that ignored the table would
        # serve the base rows to the next.
        first = _request(pipe, arrays, None)
        live = _request(pipe, arrays, tid)
        again = _request(pipe, arrays, None)
        whole = _request(merged, arrays, None)
    assert got.keys() == ref.keys()
    errs = {k: fg.rel(got[k], ref[k]) for k in ref}
    errs["request velocity"] = fg.rel(live, whole)
    moved = fg.rel(base["velocity"], ref["velocity"])
    top = max(errs, key=errs.get)
    print(f"flux3 check7 tiny {case}: {len(errs)} rows, worst "
          f"{errs[top]:.2e} at {top}; the adapter moves the velocity "
          f"{moved:.2e}")
    assert errs[top] <= STEP_TOL, (top, errs[top])
    assert moved > 100 * STEP_TOL
    assert np.array_equal(first, again)
    assert fg.rel(first, base["velocity"]) <= STEP_TOL


def test_a_key_that_names_no_module_is_refused(tmp_path):
    """A block past the last, a norm scale in both layouts and a layer
    that FLUX.2 has."""
    cfg = tiny_config()
    km = lora_map(cfg)
    rng = np.random.default_rng(5)
    good = _adapter(cfg, 6, None, ["single_blocks.0.q_proj"], "gguf")
    for i, name in enumerate(("single_blocks.2.q_proj",
                              "single_transformer_blocks.2.attn.to_q",
                              "txt_mode_blocks.0.norm.query_norm",
                              "txt_mode_blocks.0.attn.norm_q",
                              "double_blocks.0.img_attn.qkv")):
        mixed = {**good, name: make_pair(rng, 64, 64, 2)}
        path = write_adapter(tmp_path / f"s{i}.safetensors", "diffusers",
                             mixed)
        with pytest.raises(reader.UnmatchedKeysError,
                           match="1 of 2 targets name no FLUX.3 klein "
                           "module"):
            reader.resolve(reader.read_lora(path), km)


def _vet(path: str, name: str) -> int:
    ident = fp.identify(fixture_view(name))
    keymap = get_family("flux3_klein").lora_map(ident.config)
    modules = {n[:-len(".weight")] for n in ident.view.tensors
               if n.endswith(".weight")}
    return reader.vet_header(path, keymap=keymap, modules=modules)


@pytest.mark.parametrize("names", ["diffusers", "gguf"])
def test_a_target_time_key_on_base_is_refused_by_the_distilled_name(
        tmp_path, names):
    """The header check that ``generate`` runs before a model loads: a
    pair on the target time embedder resolves on the distilled file and
    is refused on Base by the name of the distilled DiT's layer."""
    rng = np.random.default_rng(7)
    target = ("target_timestep_embedder.linear_2" if names == "diffusers"
              else "ref_time_in.out_layer")
    pairs = {target: make_pair(rng, 3072, 3072, 2, 2.0),
             "single_blocks.0.q_proj": make_pair(rng, 3072, 3072, 2, 2.0)}
    path = write_adapter(tmp_path / "t.safetensors", "onetrainer", pairs)
    for name, flow_map in HEADERS.items():
        if flow_map:
            assert _vet(path, name) == 2
            continue
        with pytest.raises(reader.AdapterError,
                           match="target time embedder of the distilled "
                           "FLUX.3 klein DiT"):
            _vet(path, name)


def test_a_fused_qkv_key_is_refused_by_the_name_of_its_layout(tmp_path):
    rng = np.random.default_rng(8)
    path = write_adapter(tmp_path / "q.safetensors", "kohya", {
        "single_blocks.3.qkv": make_pair(rng, 3 * 3072, 3072, 2, 2.0),
        "single_blocks.3.mlp_out": make_pair(rng, 3072, 9216, 2, 2.0)})
    with pytest.raises(reader.AdapterError,
                       match="fused q, k and v weight of ComfyUI's FLUX.3 "
                       "layout"):
        _vet(path, "flux3_klein_q4_k_m.header.json")
