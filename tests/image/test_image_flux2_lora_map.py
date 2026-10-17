"""The FLUX.2 klein LoRA key map and the delta half of acceptance check 7:
every spelling of every target, in the GGUF names and in the diffusers
names, resolves to its leaf or to its band of a fused q, k and v weight,
the placed factors equal a float64 reference per dialect, the final
modulation of the diffusers layout lands with its halves swapped back,
the ComfyUI warning names each spelling that ComfyUI v0.41.1 drops, and
a step of the tiny model under a patch table equals the step on merged
weights. CPU."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import flux2_gates as fg
from image_fixtures import fixture_view, load_json
from lora_fixtures import (DIALECTS, dialect_keys, make_pair,
                           reference_delta, rel_l2, write_adapter,
                           write_keys)
from oracle_fixture import committed_fixture

from gmlx.image import api, lora
from gmlx.image import fingerprint as fp
from gmlx.image.families.flux2_klein.lora_map import NORM_OUT, lora_map
from gmlx.image.families.flux2_klein.model import Flux2KleinDiT
from gmlx.image.lora import install, plan, reader
from gmlx.image.registry import get_family

DELTA_TOL = 1e-6
# The live step against merged fp32 weights on the CPU, relative L2 of
# each row of the step.
STEP_TOL = 2e-5
HEADERS = ("flux2_klein_4b_q4_k_m.header.json",
           "flux2_klein_4b_f16.header.json")
# The tensor names and shapes of Flux2Transformer2DModel, from the header
# of BFL's diffusers transformer file.
DIFFUSERS_NAMES = "flux2_klein_diffusers_names.json"
ADALN = "final_layer.adaLN_modulation.1"
# The GGUF names under the model prefixes that ComfyUI and the diffusers
# loader strip, and under the underscore prefixes of kohya and LyCORIS.
SPELLINGS = {
    "comfy": ("diffusion_model.", False, ".lora_A.weight", ".lora_B.weight",
              False),
    "comfy_model": ("model.diffusion_model.", False, ".lora_down.weight",
                    ".lora_up.weight", True),
    "bare": ("", False, ".lora_A.weight", ".lora_B.weight", False),
    "lycoris": ("lycoris_", True, ".lora_down.weight", ".lora_up.weight",
                True),
    "transformer_kohya": ("lora_transformer_", True, ".lora_down.weight",
                          ".lora_up.weight", True),
}
TINY = "selftest_flux2_klein_tiny"
NOREFS = "selftest_flux2_klein_tiny_norefs"


def tiny_config() -> dict:
    _, wm = committed_fixture(TINY + "_weights")
    return dict(wm["inputs"]["config"])


def dims_of(cfg: dict) -> dict:
    """Module path to (out, in) of every linear leaf of the built
    model."""
    return {p: tuple(m.weight.shape)
            for p, m in Flux2KleinDiT(cfg).named_modules()
            if isinstance(m, nn.Linear)}


def swap_halves(a: np.ndarray) -> np.ndarray:
    h = a.shape[0] // 2
    return np.concatenate([a[h:], a[:h]], axis=0)


def diffusers_targets(km) -> list:
    """The diffusers names of the map, one per module or band: every
    diffusers name but the spelling of a double block's attn.to_out
    without its .0."""
    return sorted(n for n in km.diffusers_only
                  if not (n.startswith("transformer_blocks.")
                          and n.endswith(".attn.to_out")))


def target_dims(km, dims: dict, name: str) -> tuple:
    """(out, in) of the pair a target takes: its band's rows, or the
    module's."""
    module, band = km.targets[name]
    out, inp = dims[module]
    return (band[2] if band else out, inp)


def comfy_reads(prefix: str, name: str, km) -> bool:
    """Whether ComfyUI v0.41.1 loads the stem of target ``name`` under
    ``prefix``, by the table below."""
    if name.endswith(".linear1_qkv"):
        return prefix == "diffusion_model."
    if name in km.diffusers_only:
        return (prefix in COMFY_DIFFUSERS and not name.endswith(NOT_IN_TABLE)
                and not (name.startswith("transformer_blocks.")
                         and name.endswith(".attn.to_out")))
    return prefix in COMFY_GGUF


# What ComfyUI v0.41.1 loads of a FLUX.2 klein adapter, from the key map
# of comfy.lora.model_lora_keys_unet on a weightless FLUX.2 klein 4B
# model built from the header fixture: the GGUF names under two
# prefixes, the q, k and v rows of linear1 under diffusion_model., and
# the diffusers names of its FLUX table under four prefixes. The table
# has no name for the three modulations or the time embedder of FLUX.2,
# and none for a double block's attn.to_out without its .0.
COMFY_GGUF = ("diffusion_model.", "lora_unet_")
COMFY_DIFFUSERS = ("transformer.", "", "lycoris_", "lora_transformer_")
NOT_IN_TABLE = ("_modulation_img.linear", "_modulation_txt.linear",
                "single_stream_modulation.linear",
                "timestep_embedder.linear_1", "timestep_embedder.linear_2")


def test_the_key_map_covers_every_linear_leaf_and_only_those():
    cfg = tiny_config()
    km = lora_map(cfg)
    dims = dims_of(cfg)
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    assert km.family == "flux2_klein" and km.title == "FLUX.2 klein"
    assert {m for m, _ in km.targets.values()} == set(dims)
    for path in dims:
        assert km.targets[path] == (path, None)
    # 8 leaves in a double block, 2 in a single one and 9 outside.
    assert len(dims) == (8 * cfg["num_layers"]
                         + 2 * cfg["num_single_layers"] + 9)
    for i in range(cfg["num_layers"]):
        x, d = f"double_blocks.{i}", f"transformer_blocks.{i}.attn"
        for j, (q, k, v) in enumerate([("to_q", "to_k", "to_v"),
                                       ("add_q_proj", "add_k_proj",
                                        "add_v_proj")]):
            qkv = f"{x}.{('img_attn', 'txt_attn')[j]}.qkv"
            for band, n in enumerate((q, k, v)):
                assert km.targets[f"{d}.{n}"] == (qkv, (0, band * dim, dim))
        assert km.targets[f"{d}.to_out.0"] == km.targets[f"{d}.to_out"] == (
            f"{x}.img_attn.proj", None)
    for i in range(cfg["num_single_layers"]):
        assert km.targets[f"single_blocks.{i}.linear1_qkv"] == (
            f"single_blocks.{i}.linear1", (0, 0, 3 * dim))
    whole = [n for n in diffusers_targets(km) if km.targets[n][1] is None]
    assert sorted(km.targets[n][0] for n in whole) == sorted(
        p for p in dims if not p.endswith(".qkv"))
    assert km.swapped == frozenset({NORM_OUT})
    assert km.targets[NORM_OUT] == (ADALN, None)
    assert km.absent == {} and km.comfy_prefixes == ()
    assert km.text_names == () and km.text_targets == ()
    built = dataclasses.asdict(Flux2KleinDiT(cfg).cfg)
    assert lora_map(built) == km
    assert get_family("flux2_klein").lora_map(cfg) == km
    assert len({n.replace(".", "_") for n in km.targets}) == len(km.targets)


@pytest.mark.parametrize("name", HEADERS)
def test_the_key_map_names_every_linear_weight_of_the_real_header(name):
    ident = fp.identify(fixture_view(name))
    km = lora_map(ident.config)
    assert ident.family.name == "flux2_klein"
    linear = {n[:-len(".weight")] for n, t in ident.view.tensors.items()
              if n.endswith(".weight") and len(t.shape) == 2}
    assert linear == {m for m, _ in km.targets.values()}
    assert len(linear) == 8 * 5 + 2 * 20 + 9
    assert km.targets["transformer_blocks.4.attn.add_v_proj"] == (
        "double_blocks.4.txt_attn.qkv", (0, 6144, 3072))
    assert km.targets["single_blocks.19.linear1_qkv"] == (
        "single_blocks.19.linear1", (0, 0, 9216))
    assert "double_blocks.5.img_attn.qkv" not in km.targets
    assert "single_transformer_blocks.20.attn.to_out" not in km.targets


def test_the_key_map_names_every_linear_weight_of_the_diffusers_model():
    """Every weight of Flux2Transformer2DModel that is a matrix, which is
    every one but the q and k norms, names a target of its shape: a band
    of a fused q, k and v weight for the split projections, else the
    module of the same weight, one per module of the GGUF."""
    fx = load_json(DIFFUSERS_NAMES)
    names, shapes = fx["names"], fx["shapes"]
    assert len(names) == 169 and fx["class"] == "Flux2Transformer2DModel"
    ident = fp.identify(fixture_view(HEADERS[0]))
    km = lora_map(ident.config)
    norms = [n for n in names if ".norm_" in n]
    linear = [n[:-len(".weight")] for n in names if n not in norms]
    assert len(norms) == 4 * 5 + 2 * 20
    assert sorted(linear) == diffusers_targets(km)
    gguf = {n[:-len(".weight")]: tuple(reversed(t.shape))
            for n, t in ident.view.tensors.items()
            if n.endswith(".weight") and len(t.shape) == 2}
    placed = {}
    for n in linear:
        module, band = km.targets[n]
        out, inp = gguf[module]
        assert tuple(shapes[n + ".weight"]) == (
            band[2] if band else out, inp), n
        placed.setdefault(module, []).append(band)
    assert set(placed) == set(gguf)
    for module, bands in placed.items():
        assert bands == [None] or sorted(b[1] for b in bands) == [
            0, 3072, 6144], module


@pytest.mark.parametrize("source", ["tiny", HEADERS[0]])
def test_each_diffusers_name_lands_where_diffusers_converts_it_from(source):
    """Every diffusers name of the map is the one diffusers' converter
    gives its GGUF module, at the same third of a fused q, k and v."""
    cfg = (tiny_config() if source == "tiny"
           else fp.identify(fixture_view(source)).config)
    km = lora_map(cfg)
    depth, single = cfg["num_layers"], cfg["num_single_layers"]
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    want = {dname: (module, None if third is None
                    else (0, third * dim, dim))
            for module, names in fg.diffusers_names(depth, single).items()
            for dname, third in names}
    assert set(diffusers_targets(km)) == set(want)
    for dname, target in want.items():
        assert km.targets[dname] == target, dname


def test_a_diffusers_converted_adapter_places_the_deltas_of_its_source(
        tmp_path):
    """An adapter in the GGUF names and its diffusers conversion, as
    diffusers' converter splits the fused q, k and v pairs, give the same
    delta on every module. The pair on norm_out.linear holds its up rows
    in the order of the diffusers model's weight, the halves swapped."""
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    table = fg.diffusers_names(cfg["num_layers"], cfg["num_single_layers"])
    assert set(table) == set(dims)
    src = _adapter(km, dims, 6, None, sorted(dims))
    conv = {}
    for module, (down, up, alpha) in src.items():
        for dname, third in table[module]:
            rows = up if third is None else up[third * dim:(third + 1) * dim]
            if dname == NORM_OUT:
                rows = swap_halves(rows)
            conv[dname] = (down, rows, alpha)

    def deltas(path):
        res = reader.resolve(reader.read_lora(str(path)), km)
        parts: dict = {}
        for t in res.dit:
            parts.setdefault(t.module, []).append(plan.Part(
                t.pair.down, t.pair.up, t.pair.alpha, t.band, 1.0))
        out = {}
        for module, ps in parts.items():
            lm = plan.block_diagonal(module, ps, dims[module])
            out[module] = (np.array(lm.b, np.float64)
                           @ np.array(lm.a, np.float64))
        return out

    a = deltas(write_adapter(tmp_path / "gguf.safetensors", "diffusers", src))
    b = deltas(write_adapter(tmp_path / "conv.safetensors", "diffusers", conv))
    assert set(a) == set(b) == set(dims)
    for module in dims:
        assert rel_l2(b[module], a[module]) <= DELTA_TOL, module


def _stem(row: tuple, target: str) -> str:
    prefix, under, _, _, _ = row
    return prefix + (target.replace(".", "_") if under else target)


SPELLED = [*(f"{d} diffusers" for d in sorted(DIALECTS)),
           *(f"{d} gguf" for d in sorted(DIALECTS)),
           *(f"{s} {n}" for s in sorted(SPELLINGS)
             for n in ("gguf", "diffusers"))]


@pytest.mark.parametrize("case", SPELLED)
def test_every_spelling_of_every_target_resolves(tmp_path, case):
    """One pair on every linear layer, in the diffusers names with a band
    for each split projection or in the GGUF names, as each dialect spells
    them, lands on its leaf and band. The pair on the diffusers final
    modulation lands with its up halves swapped. The warning counts the
    stems that ComfyUI drops."""
    dialect, names = case.split(" ")
    row = DIALECTS.get(dialect) or SPELLINGS[dialect]
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    rng = np.random.default_rng(4)
    targets = diffusers_targets(km) if names == "diffusers" else sorted(dims)
    named = {t: make_pair(rng, *target_dims(km, dims, t), 2,
                          4.0 if row[4] else None) for t in targets}
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
        write_keys(tmp_path / "all.safetensors", keys)), km)
    assert res.text == () and res.unmatched == ()
    dropped = sum(not comfy_reads(row[0], t, km) for t in targets)
    if dropped:
        assert len(res.warnings) == 1, res.warnings
        assert "ComfyUI drops them" in res.warnings[0]
        assert f": {dropped} keys " in res.warnings[0]
    else:
        assert res.warnings == ()
    assert len(res.dit) == len(targets)
    by_stem = {_stem(row, t): t for t in targets}
    for t in res.dit:
        target = by_stem[t.pair.stem]
        down, up, alpha = named[target]
        assert (t.module, t.band) == km.targets[target]
        want = swap_halves(up) if target == NORM_OUT else up
        assert np.array_equal(np.array(t.pair.down), down)
        assert np.array_equal(np.array(t.pair.up), want)
        assert t.pair.alpha == alpha


# Four targets by GGUF name and by diffusers name. The stem of each under
# each prefix is loaded by ComfyUI v0.41.1 or not as COMFY_LOADS says.
COMFY_TARGETS = {
    "norm_out": (ADALN, NORM_OUT),
    "q band": ("double_blocks.1.img_attn.qkv",
               "transformer_blocks.1.attn.to_q"),
    "text v band": ("double_blocks.0.txt_attn.qkv",
                    "transformer_blocks.0.attn.add_v_proj"),
    "linear1": ("single_blocks.1.linear1",
                "single_transformer_blocks.1.attn.to_qkv_mlp_proj"),
    "modulation": ("double_stream_modulation_txt.lin",
                   "double_stream_modulation_txt.linear"),
    "time": ("time_in.out_layer",
             "time_guidance_embed.timestep_embedder.linear_2"),
}
PREFIXES = ("lora_unet_", "lycoris_", "lora_transformer_",
            "diffusion_model.", "model.diffusion_model.", "transformer.",
            "", "unet.", "base_model.model.")


def _loads(prefix: str, which: str, label: str) -> bool:
    if which == "gguf":
        return prefix in COMFY_GGUF
    return prefix in COMFY_DIFFUSERS and label not in ("modulation", "time")


def test_the_comfyui_warning_names_every_spelling_comfyui_drops():
    km = lora_map(fp.identify(fixture_view(HEADERS[0])).config)
    zeros = reader.RawPair("", mx.zeros((1, 4)), mx.zeros((4, 1)), None)
    seen = 0
    for label, names in COMFY_TARGETS.items():
        for which, name in zip(("gguf", "diffusers"), names):
            for prefix in PREFIXES:
                stem = prefix + (name.replace(".", "_")
                                 if prefix.endswith("_") else name)
                raw = reader.RawLora(
                    "a.safetensors",
                    {stem: dataclasses.replace(zeros, stem=stem)}, {})
                res = reader.resolve(raw, km)
                assert [(t.module, t.band) for t in res.dit] == [
                    km.targets[name]], stem
                warned = [w for w in res.warnings if "ComfyUI drops them"
                          in w]
                loads = _loads(prefix, which, label)
                assert len(warned) == (0 if loads else 1), stem
                assert loads == comfy_reads(prefix, name, km), stem
                if warned:
                    assert stem in warned[0]
                    assert "ComfyUI reads the GGUF names under " \
                        "diffusion_model. or lora_unet_" in warned[0]
                seen += 1
    assert seen == len(COMFY_TARGETS) * 2 * len(PREFIXES)
    for prefix, loads in (("diffusion_model.", True), ("lora_unet_", False),
                          ("", False)):
        stem = prefix + ("single_blocks_0_linear1_qkv" if prefix.endswith(
            "_") else "single_blocks.0.linear1_qkv")
        res = reader.resolve(reader.RawLora(
            "a.safetensors", {stem: dataclasses.replace(zeros, stem=stem)},
            {}), km)
        assert [(t.module, t.band) for t in res.dit] == [
            ("single_blocks.0.linear1", (0, 0, 9216))]
        assert bool(res.warnings) == (not loads), stem


# The header of Limbicnation/pixel-art-lora: diffusers names under
# transformer., rank 64, and a rank-stabilized PEFT config of alpha 128.
PIXEL_ART = "flux2_klein_pixel_art_lora.header.json"
PIXEL_ART_MODULATIONS = ("transformer.double_stream_modulation_img.linear",
                         "transformer.double_stream_modulation_txt.linear",
                         "transformer.single_stream_modulation.linear")


def header_pairs(name: str) -> reader.RawLora:
    """The pairs of a header fixture with their stems and ranks and empty
    rows, and the header's metadata."""
    doc = load_json(name)
    parts: dict = {}
    for key, shape, _ in doc["tensors"]:
        stem, _, role = key.rpartition(".lora_")
        parts.setdefault(stem, {})[role] = shape
    pairs = {stem: reader.RawPair(stem, mx.zeros((p["A.weight"][0], 1)),
                                  mx.zeros((1, p["B.weight"][1])), None)
             for stem, p in parts.items()}
    return reader.RawLora(name, pairs, doc["metadata"])


def test_the_pixel_art_adapter_warns_of_its_modulations_and_its_scale():
    """The pixel-art adapter resolves whole. Its three modulation pairs
    have diffusers names that ComfyUI's FLUX table lacks, and the warning
    names all three and no other. Its rank-stabilized header scales each
    pair by 128 over the root of 64, 16 times the scale 1 that ComfyUI
    gives a pair without an alpha tensor, and the header warning says
    so."""
    km = lora_map(fp.identify(fixture_view(HEADERS[0])).config)
    raw = header_pairs(PIXEL_ART)
    res = reader.resolve(raw, km)
    assert len(res.dit) == len(raw.pairs) == 86 and res.unmatched == ()
    assert {t.pair.alpha / t.pair.rank for t in res.dit} == {16.0}
    dropped = [w for w in res.warnings if "ComfyUI drops them" in w]
    assert len(dropped) == 1
    assert (f"3 keys have a spelling that ComfyUI does not read for FLUX.2 "
            f"klein ({', '.join(PIXEL_ART_MODULATIONS)})") in dropped[0]
    scaled = [w for w in res.warnings if "rank-stabilized" in w]
    assert len(scaled) == 1 and len(res.warnings) == 2
    assert "86 pairs take an alpha from the adapter header" in scaled[0]
    assert ("The header makes 86 of them rank-stabilized, scaled by the "
            "alpha over the square root of the rank, and gmlx applies those "
            "at 16 times the strength that ComfyUI gives them") in scaled[0]
    # The modulations land on the GGUF modules.
    mods = {t.pair.stem: t.module for t in res.dit
            if t.pair.stem in PIXEL_ART_MODULATIONS}
    assert mods == {s: s.removeprefix("transformer.").replace(
        ".linear", ".lin") for s in PIXEL_ART_MODULATIONS}


def _adapter(km, dims: dict, seed: int, alpha, targets) -> dict:
    """Pairs on ``targets`` at ranks 2 to 4."""
    rng = np.random.default_rng(seed)
    return {t: make_pair(rng, *target_dims(km, dims, t), 2 + i % 3, alpha)
            for i, t in enumerate(targets)}


def _placed(km, dims: dict, stack) -> tuple[dict, dict]:
    """The placed delta of each module of the (path, pairs, strength)
    rows of ``stack``, and its float64 reference in the GGUF's order: a
    band's rows of its module, and the diffusers final modulation's
    halves turned back."""
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
            if target == NORM_OUT:
                ref = swap_halves(ref)
            want[module] = want.get(module, 0) + ref
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
    against float64, over the whole weight and over each third of every
    fused q, k and v weight. A dialect that stores no alpha takes the
    factor 1.0. Every q, k and v band and the final modulation take a pair
    of both adapters."""
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    has_alpha = DIALECTS[dialect][4]
    targets = diffusers_targets(km) if names == "diffusers" else sorted(dims)
    both = {t for t in targets if km.targets[t][0].endswith(".qkv")} | {
        NORM_OUT if names == "diffusers" else ADALN}
    one = _adapter(km, dims, 1, 3.0 if has_alpha else None,
                   sorted(set(targets[::2]) | both))
    two = _adapter(km, dims, 2, 1.0 if has_alpha else None,
                   sorted(set(targets[::3]) | both))
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
            if module.endswith(".qkv"):
                errs += [rel_l2(delta[i * dim:(i + 1) * dim],
                                ref[i * dim:(i + 1) * dim])
                         for i in range(3)]
            worst = max(worst, *errs)
            assert max(errs) <= DELTA_TOL, (dialect, label, module, errs)
    print(f"flux2 check7 delta {dialect} {names}: worst relative L2 "
          f"{worst:.3e}")


def test_the_diffusers_final_modulation_is_swapped_back(tmp_path):
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
                         {NORM_OUT: (down, up, None)})

    def merged(keymap):
        res = reader.resolve(reader.read_lora(path), keymap)
        (t,) = res.dit
        lm = plan.block_diagonal(t.module, [plan.Part(
            t.pair.down, t.pair.up, t.pair.alpha, t.band, 1.0)], (out, inp))
        return w + np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)

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


def _tiny_pipe(merged=None) -> api.ImagePipeline:
    """The tiny DiT in fp32, with (module, delta) rows added to its
    weights in float64 for ``merged``."""
    weights, _ = committed_fixture(TINY + "_weights")
    weights = dict(weights)
    for module, delta in (merged or {}).items():
        key = f"{module}.weight"
        weights[key] = (np.asarray(weights[key], np.float64)
                        + delta).astype(np.float32)
    model = fg.tiny_model(weights, tiny_config())
    return api.ImagePipeline(dit=model, family=get_family("flux2_klein"),
                             variant="distilled")


def _request(pipe, arrays: dict, tid) -> np.ndarray:
    refs = tuple(mx.array(arrays[f"refs.{i}"])[0]
                 for i in range(fg.n_refs(arrays)))
    # A key, so that the pipeline keeps the caption rows.
    cond = api.Conditioning(mx.array(arrays["features"]), "tiny",
                            ref_latents=refs)
    return np.array(api.denoise_step(
        pipe, mx.array(arrays["latents"]), float(arrays["sigma"]), cond,
        patch_table=tid))


@pytest.mark.parametrize("name", [TINY, NOREFS])
def test_a_step_under_a_table_equals_the_step_on_merged_weights(tmp_path,
                                                                name):
    """A stacked pair of adapters over every linear leaf, one in the
    diffusers names with the split projections and one in the GGUF names,
    applied live on the tiny model: every row of the step and the
    velocity, with two reference images and with text alone, and the
    request path, whose time vectors, modulations and caption rows a
    pipeline keeps per table."""
    arrays, _ = committed_fixture(name)
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    one = _adapter(km, dims, 11, 2.0, diffusers_targets(km))
    two = _adapter(km, dims, 12, None, sorted(dims)[1::2])
    pipe = _tiny_pipe()
    rows, stack = [], []
    for i, (pairs, strength, dialect) in enumerate(
            [(one, 0.6, "kohya"), (two, -0.3, "diffusers")]):
        path = write_adapter(tmp_path / f"a{i}.safetensors", dialect, pairs)
        rows.append((lora.load_adapter(pipe, path), strength))
        stack.append((path, pairs, strength))
    _, want = _placed(km, dims, stack)
    tid = api.define_patch_table(pipe, lora.table_entries(rows))
    merged = _tiny_pipe(want)
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
    print(f"flux2 check7 tiny {name}: {len(errs)} rows, worst "
          f"{errs[top]:.2e} at {top}; the adapter moves the velocity "
          f"{moved:.2e}")
    assert errs[top] <= STEP_TOL, (top, errs[top])
    assert moved > 100 * STEP_TOL
    assert np.array_equal(first, again)
    assert fg.rel(first, base["velocity"]) <= STEP_TOL


def test_a_key_that_names_no_module_is_refused(tmp_path):
    """A block past the last, a norm scale in both layouts, a layer that
    FLUX.3 klein has, and a FLUX.1 diffusers name that ComfyUI's table
    also holds."""
    cfg = tiny_config()
    km = lora_map(cfg)
    rng = np.random.default_rng(5)
    good = {"single_blocks.0.linear2": make_pair(rng, 64, 256, 2)}
    for i, name in enumerate(("single_blocks.2.linear1",
                              "transformer_blocks.2.attn.to_q",
                              "double_blocks.0.img_attn.norm.query_norm",
                              "transformer_blocks.0.attn.norm_q",
                              "single_blocks.0.mlp_in",
                              "transformer_blocks.0.ff.net.0.proj")):
        mixed = {**good, name: make_pair(rng, 64, 64, 2)}
        path = write_adapter(tmp_path / f"s{i}.safetensors", "diffusers",
                             mixed)
        with pytest.raises(reader.UnmatchedKeysError,
                           match="1 of 2 targets name no FLUX.2 klein "
                           "module"):
            reader.resolve(reader.read_lora(path), km)


def test_a_split_projection_and_its_fused_weight_both_apply(tmp_path):
    """A pair on to_q is the first band of img_attn.qkv and a pair on the
    whole qkv is the whole weight, so a file with both applies both, and
    two spellings of one band are refused."""
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    rng = np.random.default_rng(6)
    pairs = _adapter(km, dims, 7, None, ["transformer_blocks.0.attn.to_q",
                                         "double_blocks.0.img_attn.qkv"])
    path = write_adapter(tmp_path / "b.safetensors", "diffusers", pairs)
    placed, want = _placed(km, dims, [(path, pairs, 1.0)])
    qkv = "double_blocks.0.img_attn.qkv"
    assert rel_l2(placed[qkv], want[qkv]) <= DELTA_TOL
    keys = {**dialect_keys("diffusers", pairs),
            **dialect_keys("kohya", {"transformer_blocks.0.attn.to_q":
                                     make_pair(rng, 64, 64, 2, 2.0)})}
    with pytest.raises(reader.AdapterError, match="both target "
                       "double_blocks.0.img_attn.qkv band"):
        reader.resolve(reader.read_lora(
            write_keys(tmp_path / "two.safetensors", keys)), km)
