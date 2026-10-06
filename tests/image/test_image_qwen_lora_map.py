"""The Qwen-Image 2.1 LoRA key map, the bands of the fused image MLP
weight, and the delta half of acceptance check 7: the placed factors
against a float64 reference, per dialect and on both layouts. CPU."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import qwen_tiny as qt
from image_fixtures import Q_HEADER, fixture_view
from lora_fixtures import (DIALECTS, dialect_keys, make_pair, reference_delta,
                           rel_l2, tiny_z_adapter, write_adapter, write_keys)

from gmlx.image import fingerprint as fp
from gmlx.image.families.qwen_image21.lora_map import lora_map
from gmlx.image.families.qwen_image21.model import (QwenImage21Config,
                                                    QwenImage21DiT)
from gmlx.image.families.z_image.lora_map import lora_map as z_lora_map
from gmlx.image.lora import plan, reader
from gmlx.image.registry import get_family

# Check 7 delta half: relative L2 against the float64 reference.
DELTA_TOL = 1e-6
LAYOUTS = ("fused", "unfused")
DIM = qt.TINY["num_attention_heads"] * qt.TINY["attention_head_dim"]
FFN = qt.TINY["mlp_ratio"] * DIM
B0, B1 = "transformer_blocks.0.", "transformer_blocks.1."
# A fifth spelling that the reference converter reads: dotted names under
# the model prefix, with kohya's suffixes and an alpha.
DOTTED = ("diffusion_model.", False, ".lora_down.weight", ".lora_up.weight",
          True)


def config(layout: str) -> dict:
    """The config the family synthesizes for the tiny weights."""
    return qt.config_of(qt.weights(fused=layout == "fused"))


def dims_of(cfg: dict) -> dict:
    """Module path to (out, in) of every linear leaf of the built model."""
    return {p: tuple(m.weight.shape)
            for p, m in QwenImage21DiT(cfg).named_modules()
            if isinstance(m, nn.Linear)}


def adapter(name: str, alpha=None) -> dict:
    """An adapter of the tiny rule as target to (down, up, alpha)."""
    return {t: (d, u, alpha) for t, (d, u) in qt.rule.lora_pairs(name).items()}


def placed(target: str, layout: str) -> tuple:
    """Where the tiny model of ``layout`` holds a reference target."""
    for band, leaf in enumerate(("img_mlp.gate_layer", "img_mlp.proj")):
        if layout == "fused" and target.endswith(leaf):
            return (target[:-len(leaf)] + "img_mlp.gate_up",
                    (0, band * FFN, FFN))
    return target, None


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_key_map_covers_every_linear_leaf_and_only_those(layout):
    cfg = config(layout)
    km = lora_map(cfg)
    dims = dims_of(cfg)
    assert km.family == "qwen_image21" and km.title == "Qwen-Image 2.1"
    assert {m for m, _ in km.targets.values()} == set(dims)
    for name, (module, band) in km.targets.items():
        if band is not None:
            assert band[0] == 0 and band[1] + band[2] <= dims[module][0], name
    # A name of the file maps to itself on the whole weight.
    for path in dims:
        assert km.targets[path] == (path, None)
    # Every reference name is a target, placed where the layout holds it.
    reference = dims_of(config("unfused"))
    assert len(reference) == 8 + 7 * qt.TINY["num_layers"]
    for target in reference:
        assert km.targets[target] == placed(target, layout)
    assert len(km.targets) == len(reference) + (
        qt.TINY["num_layers"] if layout == "fused" else 0)
    # The header's config and the built model's give one map.
    built = dataclasses.asdict(QwenImage21Config.from_dict(cfg))
    assert lora_map(built) == km
    assert get_family("qwen_image21").lora_map(cfg) == km
    # No two targets share an underscore spelling, and no key reaches a
    # text encoder or another program's key map.
    assert len({n.replace(".", "_") for n in km.targets}) == len(km.targets)
    assert km.text_names == () and km.text_targets == ()
    assert km.diffusers_only == frozenset()


def test_gate_layer_and_proj_are_bands_0_and_1_of_the_fused_weight():
    km = lora_map(config("fused"))
    for b in (B0, B1):
        gate_up = b + "img_mlp.gate_up"
        assert km.targets[b + "img_mlp.gate_layer"] == (gate_up, (0, 0, FFN))
        assert km.targets[b + "img_mlp.proj"] == (gate_up, (0, FFN, FFN))
        assert km.targets[gate_up] == (gate_up, None)
    apart = lora_map(config("unfused"))
    for leaf in ("gate_layer", "proj"):
        assert apart.targets[f"{B0}img_mlp.{leaf}"] == (
            f"{B0}img_mlp.{leaf}", None)
    assert B0 + "img_mlp.gate_up" not in apart.targets
    assert "transformer_blocks.2.attn.to_q" not in km.targets


def test_the_key_map_names_every_linear_weight_of_the_real_header():
    ident = fp.identify(fixture_view(Q_HEADER))
    view = ident.view
    km = lora_map(ident.config)
    assert ident.family.name == "qwen_image21" and ident.layout == "fused"
    linear = {n[:-len(".weight")] for n, t in view.tensors.items()
              if n.endswith(".weight") and len(t.shape) == 2}
    assert len(linear) == 8 + 6 * 32
    assert linear == {m for m, _ in km.targets.values()}
    assert len([n for n in linear if n.endswith("img_mlp.gate_up")]) == 32
    assert view.shape(B0 + "img_mlp.gate_up.weight") == (4096, 24576)
    for i in (0, 31):
        x = f"transformer_blocks.{i}.img_mlp"
        assert km.targets[f"{x}.gate_layer"] == (f"{x}.gate_up",
                                                 (0, 0, 12288))
        assert km.targets[f"{x}.proj"] == (f"{x}.gate_up",
                                           (0, 12288, 12288))
        assert km.targets[f"{x}.out"] == (f"{x}.out", None)
    assert "transformer_blocks.32.attn.to_q" not in km.targets
    assert len(km.targets) == 8 + 8 * 32


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dialect", [*sorted(DIALECTS), "dotted"])
def test_every_spelling_of_every_reference_target_resolves(tmp_path, dialect,
                                                           layout):
    """One pair on every linear layer of the reference model, spelled as
    each dialect spells it, lands on the leaf and band of the layout."""
    row = DIALECTS.get(dialect, DOTTED)
    rng = np.random.default_rng(4)
    reference = dims_of(config("unfused"))
    pairs = {t: make_pair(rng, out, inp, 2, 4.0 if row[4] else None)
             for t, (out, inp) in reference.items()}
    prefix, under, down_s, up_s, _ = row
    keys = {}
    for target, (down, up, alpha) in pairs.items():
        stem = prefix + (target.replace(".", "_") if under else target)
        keys[stem + down_s], keys[stem + up_s] = down, up
        if alpha is not None:
            keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    if dialect in DIALECTS:
        assert keys.keys() == dialect_keys(dialect, pairs).keys()
    km = lora_map(config(layout))
    res = reader.resolve(reader.read_lora(
        write_keys(tmp_path / "all.safetensors", keys)), km)
    assert res.text == () and res.unmatched == () and res.warnings == ()
    want = {placed(t, layout): pairs[t] for t in pairs}
    assert len(res.dit) == len(want) == len(reference)
    for t in res.dit:
        down, up, alpha = want[(t.module, t.band)]
        assert np.array_equal(np.array(t.pair.down), down)
        assert np.array_equal(np.array(t.pair.up), up)
        assert t.pair.alpha == alpha
    # The flat names that the reference converter reads for the layers
    # outside the blocks are the underscore forms of their paths.
    if under:
        flat = {s[len(prefix):-len(down_s)] for s in keys
                if s.endswith(down_s)}
        assert {"img_in", "proj_out", "norm_out_linear",
                "time_text_embed_timestep_embedder_linear_1",
                "time_text_embed_timestep_embedder_linear_2"} <= flat


def _parts(path, strength, keymap) -> dict:
    """Module to plan parts of an adapter file."""
    res = reader.resolve(reader.read_lora(str(path)), keymap)
    out: dict = {}
    for t in res.dit:
        out.setdefault(t.module, []).append(plan.Part(
            t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
    return out


def _reference(pairs: dict, strength: float, keymap) -> dict:
    out: dict = {}
    for target, (down, up, alpha) in pairs.items():
        module, band = keymap.targets[target]
        out.setdefault(module, []).append((down, up, alpha, band, strength))
    return out


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_delta_half_per_dialect(tmp_path, dialect, layout):
    """Strengths 0.5 and 1.0 and a stacked pair with a negative strength,
    against float64, on the fused weight and on plain leaves. A dialect
    that stores no alpha takes the factor 1.0."""
    cfg = config(layout)
    km, dims = lora_map(cfg), dims_of(cfg)
    has_alpha = DIALECTS[dialect][4]
    one = adapter("one", 3.0 if has_alpha else None)
    two = adapter("two", 1.0 if has_alpha else None)
    p1 = write_adapter(tmp_path / "one.safetensors", dialect, one)
    p2 = write_adapter(tmp_path / "two.safetensors", dialect, two)
    cases = [([(p1, one, 0.5)], "0.5"), ([(p1, one, 1.0)], "1.0"),
             ([(p1, one, 0.75), (p2, two, -0.4)], "stacked")]
    fused = layout == "fused"
    worst = 0.0
    for stack, label in cases:
        got: dict = {}
        want: dict = {}
        for path, pairs, strength in stack:
            for m, parts in _parts(path, strength, km).items():
                got.setdefault(m, []).extend(parts)
            for m, rows in _reference(pairs, strength, km).items():
                want.setdefault(m, []).extend(rows)
        # One has 18 targets and two adds 2 new ones. A fused block holds
        # its gate and its projection target in one module.
        targets = 18 + 2 * (len(stack) - 1)
        assert set(got) == set(want)
        assert len(got) == targets - (len(stack) if fused else 0), label
        for module, parts in got.items():
            lm = plan.block_diagonal(module, parts, dims[module])
            delta = np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)
            ref = reference_delta(want[module], dims[module])
            err = rel_l2(delta, ref)
            worst = max(worst, err)
            assert err <= DELTA_TOL, (dialect, layout, label, module, err)
    stacked = {}
    for path, _, strength in cases[2][0]:
        for m, parts in _parts(path, strength, km).items():
            stacked.setdefault(m, []).extend(parts)
    if fused:
        # Block 0: the gate at ranks 4 and 3 and the projection at rank 2
        # share one pair of rank 9. Block 1: one names the gate alone, so
        # its delta leaves the projection rows at zero.
        gate_up = B0 + "img_mlp.gate_up"
        lm = plan.block_diagonal(gate_up, stacked[gate_up], dims[gate_up])
        assert lm.a.shape == (9, DIM) and lm.b.shape == (2 * FFN, 9)
        gate_up = B1 + "img_mlp.gate_up"
        alone = reference_delta(_reference(one, 1.0, km)[gate_up],
                                dims[gate_up])
        assert alone[:FFN].any() and not alone[FFN:].any()
        both = reference_delta(
            [*_reference(one, 0.75, km)[gate_up],
             *_reference(two, -0.4, km)[gate_up]], dims[gate_up])
        assert both[:FFN].any() and both[FFN:].any()
    else:
        gate = B0 + "img_mlp.gate_layer"
        lm = plan.block_diagonal(gate, stacked[gate], dims[gate])
        assert lm.a.shape == (7, DIM) and lm.b.shape == (FFN, 7)
        assert B1 + "img_mlp.proj" in stacked
    print(f"qwen check7 delta {dialect} {layout}: worst relative L2 "
          f"{worst:.3e}")


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_an_adapter_of_the_other_family_is_refused_by_name(tmp_path,
                                                           dialect):
    from image_fixtures import TINY_Z_CONFIG

    alpha = 4.0 if DIALECTS[dialect][4] else None
    z = write_adapter(tmp_path / "z.safetensors", dialect,
                      tiny_z_adapter(1, alpha, wide=True))
    q = write_adapter(tmp_path / "q.safetensors", dialect,
                      adapter("one", alpha))
    for layout in LAYOUTS:
        km = lora_map(config(layout))
        with pytest.raises(reader.AdapterFamilyError,
                           match="z.safetensors is not a Qwen-Image 2.1 "
                                 "adapter: none of its 13 targets"):
            reader.resolve(reader.read_lora(z), km)
        with pytest.raises(reader.AdapterFamilyError,
                           match="not a Qwen-Image 2.1 adapter"):
            reader.vet_header(z, keymap=km)
        assert len(reader.resolve(reader.read_lora(q), km).dit) == 18
    with pytest.raises(reader.AdapterFamilyError,
                       match="q.safetensors is not a Z-Image adapter"):
        reader.resolve(reader.read_lora(q), z_lora_map(TINY_Z_CONFIG))


# Names that are no linear layer of this family: a block past the last
# one, the layers of the older Qwen-Image transformer, a text encoder
# layer, and a norm weight.
STRANGERS = ("transformer_blocks.2.attn.to_q",
             "transformer_blocks.0.attn.add_q_proj",
             "transformer_blocks.0.img_mlp.net.0.proj",
             "transformer_blocks.0.txt_mlp.net.2",
             "transformer_blocks.0.img_mod.1",
             "transformer_blocks.0.attn.norm_q",
             "txt_in", "txt_norm")


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_a_key_that_names_no_module_is_refused(tmp_path, dialect):
    rng = np.random.default_rng(5)
    alpha = 4.0 if DIALECTS[dialect][4] else None
    good = adapter("one", alpha)
    km = lora_map(config("fused"))
    for i, name in enumerate(STRANGERS):
        mixed = {**good, name: make_pair(rng, DIM, DIM, 2, alpha)}
        path = write_adapter(tmp_path / f"s{i}.safetensors", dialect, mixed)
        with pytest.raises(reader.UnmatchedKeysError,
                           match="1 of 19 targets name no Qwen-Image 2.1 "
                                 "module") as e:
            reader.resolve(reader.read_lora(path), km)
        assert len(e.value.keys) == 1
        with pytest.raises(reader.UnmatchedKeysError):
            reader.vet_header(path, keymap=km)
        res = reader.resolve(reader.read_lora(path), km,
                             allow_unmatched=True)
        assert len(res.dit) == 18 and len(res.unmatched) == 1
        assert "They are skipped" in res.warnings[0]
    # The fused name is a module of the fused file alone.
    fused = {**good, B1 + "img_mlp.gate_up": make_pair(rng, 2 * FFN, DIM, 2,
                                                       alpha)}
    path = write_adapter(tmp_path / "gate_up.safetensors", dialect, fused)
    assert len(reader.resolve(reader.read_lora(path), km).dit) == 19
    with pytest.raises(reader.UnmatchedKeysError, match="gate_up"):
        reader.resolve(reader.read_lora(path), lora_map(config("unfused")))


def test_a_text_encoder_key_is_refused(tmp_path):
    rng = np.random.default_rng(6)
    km = lora_map(config("fused"))
    keys = dialect_keys("kohya", adapter("one", 3.0))
    stem = "lora_te_model_layers_0_self_attn_q_proj"
    keys[stem + ".lora_down.weight"], keys[stem + ".lora_up.weight"], _ = \
        make_pair(rng, DIM, DIM, 2)
    path = write_keys(tmp_path / "te.safetensors", keys)
    with pytest.raises(reader.UnmatchedKeysError, match=stem):
        reader.resolve(reader.read_lora(path), km)


def test_two_keys_on_one_band_are_refused(tmp_path):
    """A file that names the fused weight's gate rows twice, once by the
    reference name and once under another prefix, is not summed."""
    rng = np.random.default_rng(7)
    km = lora_map(config("fused"))
    pair = make_pair(rng, FFN, DIM, 2)
    keys = dialect_keys("diffusers", {B0 + "img_mlp.gate_layer": pair})
    keys.update({k.replace("transformer.", "diffusion_model.", 1): v
                 for k, v in keys.items()})
    path = write_keys(tmp_path / "twice.safetensors", keys)
    with pytest.raises(reader.AdapterError, match="both target "
                       "transformer_blocks.0.img_mlp.gate_up band"):
        reader.resolve(reader.read_lora(path), km)


def test_an_adapter_is_checked_against_the_real_header(tmp_path):
    """What the image command does before a model loads: the key map of
    the header's config and the modules the file holds."""
    ident = fp.identify(fixture_view(Q_HEADER))
    km = ident.family.lora_map(ident.config)
    held = {n[:-len(".weight")] for n in ident.view.tensors
            if n.endswith(".weight")}
    one = write_adapter(tmp_path / "one.safetensors", "kohya",
                        adapter("one", 3.0))
    assert reader.vet_header(one, keymap=km, modules=held) == 18
    z = write_adapter(tmp_path / "z.safetensors", "kohya",
                      tiny_z_adapter(1, 4.0))
    with pytest.raises(reader.AdapterFamilyError,
                       match="not a Qwen-Image 2.1 adapter"):
        reader.vet_header(z, keymap=km, modules=held)
    # A file with the pair apart holds no fused weight for the bands.
    with pytest.raises(reader.AdapterError,
                       match="img_mlp.gate_up, which this DiT does not have"):
        reader.vet_header(one, keymap=km, modules={
            m for m in held if not m.endswith("gate_up")})


def test_the_rule_adapters_follow_from_the_tiny_rule():
    one, two = qt.rule.lora_pairs("one"), qt.rule.lora_pairs("two")
    assert len(one) == 18 and len(two) == 7 and len(set(one) & set(two)) == 5
    reference = dims_of(config("unfused"))
    for pairs in (one, two):
        for target, (down, up) in pairs.items():
            out, inp = reference[target]
            assert down.shape[1] == inp and up.shape[0] == out
            assert down.shape[0] == up.shape[1]
            for a in (down, up):
                assert a.dtype == np.float32 and np.abs(a).max() <= 0.25
                # bfloat16 holds every value.
                assert np.array_equal(
                    np.array(mx.array(a).astype(mx.bfloat16)
                             .astype(mx.float32)), a)
    # Every kind of linear layer has a pair.
    kinds = {t.split(".", 2)[2] if t.startswith("transformer_blocks.") else t
             for t in one}
    assert kinds == {p.split(".", 2)[2] if p.startswith("transformer_blocks.")
                     else p for p in reference}
