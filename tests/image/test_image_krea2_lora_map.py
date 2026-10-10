"""The Krea 2 LoRA key map and the delta half of acceptance check 7: every
spelling of every target resolves to its leaf, and the placed factors
equal a float64 reference per dialect. CPU."""
from __future__ import annotations

import dataclasses

import mlx.nn as nn
import numpy as np
import pytest

import krea2_gates as kg
from image_fixtures import fixture_view
from lora_fixtures import (DIALECTS, dialect_keys, make_pair,
                           reference_delta, rel_l2, write_adapter,
                           write_keys)
from oracle_fixture import committed_fixture

from gmlx.image import fingerprint as fp
from gmlx.image.families.krea2.lora_map import lora_map
from gmlx.image.families.krea2.model import Krea2Config, Krea2DiT
from gmlx.image.lora import plan, reader
from gmlx.image.registry import get_family

DELTA_TOL = 1e-6
HEADERS = ("krea2_turbo_q4_k_m.header.json", "krea2_raw_q4_k_m.header.json")
UNREAD = ("last.up", "last.down")
# The reference trainer's and ComfyUI's spelling: the GGUF names under the
# model prefix, as diffusers' Krea 2 converter reads them.
# The kohya spelling of the GGUF names, and the one underscore prefix
# under which ComfyUI reads the diffusers names.
SPELLINGS = {
    "native": ("diffusion_model.", False, ".lora_A.weight",
               ".lora_B.weight", False),
    "native_kohya": ("lora_unet_", True, ".lora_down.weight",
                     ".lora_up.weight", True),
    "lycoris": ("lycoris_", True, ".lora_down.weight", ".lora_up.weight",
                True),
}
# The diffusers names of the layers outside the blocks.
_TOP = {"first": "img_in", "tmlp.0": "time_embed.linear_1",
        "tmlp.2": "time_embed.linear_2", "tproj.1": "time_mod_proj",
        "txtmlp.1": "txt_in.linear_1", "txtmlp.3": "txt_in.linear_2",
        "txtfusion.projector": "text_fusion.projector",
        "last.linear": "final_layer.linear"}
_LEAF = {"attn.wq": "attn.to_q", "attn.wk": "attn.to_k",
         "attn.wv": "attn.to_v", "attn.gate": "attn.to_gate",
         "attn.wo": "attn.to_out.0", "mlp.gate": "ff.gate",
         "mlp.up": "ff.up", "mlp.down": "ff.down"}
_GROUP = {"blocks": "transformer_blocks",
          "txtfusion.layerwise_blocks": "text_fusion.layerwise_blocks",
          "txtfusion.refiner_blocks": "text_fusion.refiner_blocks"}


def tiny_config(last_mlp: bool = False) -> dict:
    _, wm = committed_fixture("selftest_krea2_tiny_weights")
    return dict(kg.config_of(wm["inputs"]["config"]), last_mlp=last_mlp)


def dims_of(cfg: dict) -> dict:
    """Module path to (out, in) of every linear leaf of the built
    model."""
    return {p: tuple(m.weight.shape)
            for p, m in Krea2DiT(cfg).named_modules()
            if isinstance(m, nn.Linear)}


def diffusers_name(path: str) -> str:
    """The Krea2Transformer2DModel name of a gmlx module path."""
    if path in _TOP:
        return _TOP[path]
    for native, d in _GROUP.items():
        if path.startswith(native + "."):
            i, leaf = path[len(native) + 1:].split(".", 1)
            return f"{d}.{i}.{_LEAF[leaf]}"
    raise KeyError(path)


def targets_of(cfg: dict) -> dict:
    return {p: d for p, d in dims_of(cfg).items()
            if not p.startswith(UNREAD)}


@pytest.mark.parametrize("last_mlp", [False, True])
def test_the_key_map_covers_every_read_linear_leaf_and_only_those(last_mlp):
    cfg = tiny_config(last_mlp)
    km = lora_map(cfg)
    dims = dims_of(cfg)
    assert km.family == "krea2" and km.title == "Krea 2"
    want = targets_of(cfg)
    assert {m for m, _ in km.targets.values()} == set(want)
    assert set(dims) - set(want) == ({"last.up", "last.down"} if last_mlp
                                     else set())
    for path in want:
        assert km.targets[path] == (path, None)
        assert km.targets[diffusers_name(path)] == (path, None)
    wo = [p for p in want if p.endswith(".attn.wo")]
    for path in wo:
        assert km.targets[diffusers_name(path)[:-len(".0")]] == (path, None)
    assert len(km.targets) == 2 * len(want) + len(wo)
    assert km.diffusers_only == frozenset(
        [diffusers_name(p) for p in want]
        + [diffusers_name(p)[:-len(".0")] for p in wo])
    built = dataclasses.asdict(Krea2Config.from_dict(cfg))
    assert lora_map(built) == km
    assert get_family("krea2").lora_map(cfg) == km
    assert len({n.replace(".", "_") for n in km.targets}) == len(km.targets)
    assert km.text_names == () and km.text_targets == ()


@pytest.mark.parametrize("name", HEADERS)
def test_the_key_map_names_every_read_linear_weight_of_the_real_header(name):
    ident = fp.identify(fixture_view(name))
    km = lora_map(ident.config)
    assert ident.family.name == "krea2"
    # One converter stores the projector flat, with its shape in a KV.
    linear = {n[:-len(".weight")] for n, t in ident.view.tensors.items()
              if n.endswith(".weight") and not n.startswith(
                  ("last.up.", "last.down."))
              and (len(t.shape) == 2 or n == "txtfusion.projector.weight")}
    assert linear == {m for m, _ in km.targets.values()}
    # 28 blocks and 2 + 2 text fusion blocks of 8 leaves, and 8 others.
    assert len(linear) == 8 * 32 + 8
    assert km.targets["transformer_blocks.27.attn.to_out.0"] == (
        "blocks.27.attn.wo", None)
    assert "transformer_blocks.28.attn.to_q" not in km.targets


@pytest.mark.parametrize("dialect", [*sorted(DIALECTS), "native",
                                     "native_kohya", "lycoris"])
def test_every_spelling_of_every_target_resolves(tmp_path, dialect):
    """One pair on every read linear layer, in the diffusers names or,
    for the native spellings, the GGUF names, as each dialect spells
    them, lands on its leaf."""
    row = DIALECTS.get(dialect) or SPELLINGS[dialect]
    cfg = tiny_config()
    rng = np.random.default_rng(4)
    want = targets_of(cfg)
    by_path = {p: make_pair(rng, o, i, 2, 4.0 if row[4] else None)
               for p, (o, i) in want.items()}
    prefix, under, down_s, up_s, _ = row
    keys = {}
    for path, (down, up, alpha) in by_path.items():
        target = (path if dialect.startswith("native")
                  else diffusers_name(path))
        stem = prefix + (target.replace(".", "_") if under else target)
        keys[stem + down_s], keys[stem + up_s] = down, up
        if alpha is not None:
            keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    if dialect in DIALECTS:
        named = {diffusers_name(p): v for p, v in by_path.items()}
        assert keys.keys() == dialect_keys(dialect, named).keys()
    res = reader.resolve(reader.read_lora(
        write_keys(tmp_path / "all.safetensors", keys)), lora_map(cfg))
    assert res.text == () and res.unmatched == ()
    if dialect == "kohya":
        # ComfyUI reads the diffusers names under lycoris_ alone.
        assert len(res.warnings) == 1
        assert "ComfyUI drops them" in res.warnings[0]
    else:
        assert res.warnings == ()
    assert len(res.dit) == len(want)
    for t in res.dit:
        down, up, alpha = by_path[t.module]
        assert t.band is None
        assert np.array_equal(np.array(t.pair.down), down)
        assert np.array_equal(np.array(t.pair.up), up)
        assert t.pair.alpha == alpha


def _adapter(cfg: dict, seed: int, alpha, paths) -> dict:
    """Pairs in the diffusers names on ``paths`` at ranks 2 to 4."""
    dims = dims_of(cfg)
    rng = np.random.default_rng(seed)
    return {diffusers_name(p): make_pair(rng, *dims[p], 2 + i % 3, alpha)
            for i, p in enumerate(paths)}


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_delta_half_per_dialect(tmp_path, dialect):
    """Strengths 0.5 and 1.0 and a stacked pair with a negative strength,
    against float64. A dialect that stores no alpha takes the factor
    1.0."""
    cfg = tiny_config()
    km, dims = lora_map(cfg), dims_of(cfg)
    has_alpha = DIALECTS[dialect][4]
    paths = sorted(targets_of(cfg))
    one = _adapter(cfg, 1, 3.0 if has_alpha else None, paths[::2])
    two = _adapter(cfg, 2, 1.0 if has_alpha else None, paths[::3])
    p1 = write_adapter(tmp_path / "one.safetensors", dialect, one)
    p2 = write_adapter(tmp_path / "two.safetensors", dialect, two)
    cases = [([(p1, one, 0.5)], "0.5"), ([(p1, one, 1.0)], "1.0"),
             ([(p1, one, 0.75), (p2, two, -0.4)], "stacked")]
    worst = 0.0
    for stack, label in cases:
        got: dict = {}
        want: dict = {}
        for path, pairs, strength in stack:
            res = reader.resolve(reader.read_lora(str(path)), km)
            for t in res.dit:
                got.setdefault(t.module, []).append(plan.Part(
                    t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
            for target, (down, up, alpha) in pairs.items():
                module, band = km.targets[target]
                want.setdefault(module, []).append(
                    (down, up, alpha, band, strength))
        assert set(got) == set(want), label
        for module, parts in got.items():
            lm = plan.block_diagonal(module, parts, dims[module])
            delta = np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)
            err = rel_l2(delta, reference_delta(want[module], dims[module]))
            worst = max(worst, err)
            assert err <= DELTA_TOL, (dialect, label, module, err)
    print(f"krea2 check7 delta {dialect}: worst relative L2 {worst:.3e}")


def test_a_key_that_names_no_module_is_refused(tmp_path):
    """A block past the last, a norm, the modulation table and the final
    layer matrices the model does not read."""
    cfg = tiny_config(last_mlp=True)
    km = lora_map(cfg)
    rng = np.random.default_rng(5)
    good = _adapter(cfg, 6, None, ["blocks.0.attn.wq"])
    for i, name in enumerate(("transformer_blocks.2.attn.to_q",
                              "transformer_blocks.0.norm1",
                              "transformer_blocks.0.scale_shift_table",
                              "last.up", "last.down")):
        mixed = {**good, name: make_pair(rng, 64, 64, 2)}
        path = write_adapter(tmp_path / f"s{i}.safetensors", "diffusers",
                             mixed)
        with pytest.raises(reader.UnmatchedKeysError,
                           match="1 of 2 targets name no Krea 2 module"):
            reader.resolve(reader.read_lora(path), km)
