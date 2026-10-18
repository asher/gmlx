"""The MOSS-Image LoRA key map and the delta half of acceptance check 7:
every spelling of every target resolves to its leaf, the placed factors
equal a float64 reference per dialect, and a step of the tiny DiT under
a patch table equals the step of the DiT with the deltas merged into its
weights. CPU."""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from image_fixtures import fixture_view
from lora_fixtures import (DIALECTS, dialect_keys, make_pair,
                           reference_delta, rel_l2, write_adapter,
                           write_keys)

from gmlx.image import fingerprint as fp
from gmlx.image.families.moss_image import model as mm
from gmlx.image.families.moss_image.lora_map import lora_map
from gmlx.image.lora import plan, reader
from gmlx.image.registry import get_family

DELTA_TOL = 1e-6
HEADER = "moss_image_sft_q4_k_m.header.json"
TINY = {"dim": 64, "ffn_dim": 48, "freq_dim": 16, "latent_channels": 8,
        "num_heads": 2, "num_layers": 2, "rope_base": 10000.0,
        "text_dim": 24, "num_text_blocks": 2}
H, W = 3, 4
# The GGUF names under the prefixes ComfyUI reads, and two that it does
# not: the diffusers prefix and lycoris_.
SPELLINGS = {
    "native": ("diffusion_model.", False, ".lora_A.weight",
               ".lora_B.weight", False),
    "native_kohya": ("lora_unet_", True, ".lora_down.weight",
                     ".lora_up.weight", True),
    "lycoris": ("lycoris_", True, ".lora_down.weight", ".lora_up.weight",
                True),
}
COMFY_READS = ("native", "native_kohya", "kohya")


def dims_of(cfg: dict = TINY) -> dict:
    """Module path to (out, in) of every linear leaf of the built
    model."""
    return {p: tuple(m.weight.shape)
            for p, m in mm.MossImageDiT(cfg).named_modules()
            if isinstance(m, nn.Linear)}


def test_the_key_map_covers_every_linear_leaf_and_only_those():
    km = lora_map(TINY)
    dims = dims_of()
    assert km.family == "moss_image" and km.title == "MOSS-Image"
    assert set(km.targets) == set(dims)
    assert all(km.targets[p] == (p, None) for p in dims)
    # 2 blocks of 6 leaves, 2 refiner blocks of 5, and 7 others.
    assert len(dims) == 2 * 6 + 2 * 5 + 7
    assert km.diffusers_only == frozenset()
    assert km.comfy_prefixes == ("lora_unet_", "diffusion_model.")
    built = dataclasses.asdict(mm.MossConfig.from_dict(TINY))
    assert lora_map(built) == km
    assert get_family("moss_image").lora_map(TINY) == km
    assert len({n.replace(".", "_") for n in km.targets}) == len(km.targets)
    assert km.text_names == () and km.text_targets == ()


def test_the_key_map_names_every_linear_weight_of_the_real_header():
    ident = fp.identify(fixture_view(HEADER))
    km = lora_map(ident.config)
    assert ident.family.name == "moss_image"
    linear = {n[:-len(".weight")] for n, t in ident.view.tensors.items()
              if n.endswith(".weight") and len(t.shape) == 2}
    assert linear == set(km.targets)
    assert len(linear) == 42 * 6 + 2 * 5 + 7
    assert km.targets["layers.41.proj.ffn_up_gate"] == (
        "layers.41.proj.ffn_up_gate", None)
    assert "layers.42.proj.q" not in km.targets


@pytest.mark.parametrize("dialect", [*sorted(DIALECTS), *SPELLINGS])
def test_every_spelling_of_every_target_resolves(tmp_path, dialect):
    """One pair on every linear layer, in the GGUF names, as each dialect
    spells them, lands on its leaf. A spelling ComfyUI does not read for
    MOSS-Image is applied with a warning."""
    row = DIALECTS.get(dialect) or SPELLINGS[dialect]
    rng = np.random.default_rng(4)
    want = dims_of()
    by_path = {p: make_pair(rng, o, i, 2, 4.0 if row[4] else None)
               for p, (o, i) in want.items()}
    prefix, under, down_s, up_s, _ = row
    keys = {}
    for path, (down, up, alpha) in by_path.items():
        stem = prefix + (path.replace(".", "_") if under else path)
        keys[stem + down_s], keys[stem + up_s] = down, up
        if alpha is not None:
            keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    if dialect in DIALECTS:
        assert keys.keys() == dialect_keys(dialect, by_path).keys()
    res = reader.resolve(reader.read_lora(
        write_keys(tmp_path / "all.safetensors", keys)), lora_map(TINY))
    assert res.text == () and res.unmatched == ()
    if dialect in COMFY_READS:
        assert res.warnings == ()
    else:
        assert len(res.warnings) == 1
        assert "ComfyUI drops them" in res.warnings[0]
    assert len(res.dit) == len(want)
    for t in res.dit:
        down, up, alpha = by_path[t.module]
        assert t.band is None
        assert np.array_equal(np.array(t.pair.down), down)
        assert np.array_equal(np.array(t.pair.up), up)
        assert t.pair.alpha == alpha


def _adapter(seed: int, alpha, paths) -> dict:
    dims = dims_of()
    rng = np.random.default_rng(seed)
    return {p: make_pair(rng, *dims[p], 2 + i % 3, alpha)
            for i, p in enumerate(paths)}


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_delta_half_per_dialect(tmp_path, dialect):
    """Strengths 0.5 and 1.0 and a stacked pair with a negative strength,
    against float64. A dialect that stores no alpha takes the factor
    1.0."""
    km, dims = lora_map(TINY), dims_of()
    has_alpha = DIALECTS[dialect][4]
    paths = sorted(dims)
    one = _adapter(1, 3.0 if has_alpha else None, paths[::2])
    two = _adapter(2, 1.0 if has_alpha else None, paths[::3])
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
    print(f"moss check7 delta {dialect}: worst relative L2 {worst:.3e}")


def test_a_key_that_names_no_module_is_refused(tmp_path):
    """A block past the last, a refiner past the last, a norm and the
    RoPE bands."""
    km = lora_map(TINY)
    rng = np.random.default_rng(5)
    good = _adapter(6, None, ["layers.0.proj.q"])
    for i, (name, o, n) in enumerate((
            ("layers.2.proj.q", 64, 64),
            ("text_embedder.blocks.2.qkv", 192, 64),
            ("layers.0.norm1", 64, 64), ("rope.bands", 4, 4))):
        mixed = {**good, name: make_pair(rng, o, n, 2)}
        path = write_adapter(tmp_path / f"s{i}.safetensors", "diffusers",
                             mixed)
        with pytest.raises(reader.UnmatchedKeysError,
                           match="1 of 2 targets name no MOSS-Image module"):
            reader.resolve(reader.read_lora(path), km)


def _tiny(seed: int = 0):
    from mlx.utils import tree_flatten

    m = mm.MossImageDiT(TINY)
    rng = np.random.default_rng(seed)
    params = []
    for k, v in tree_flatten(m.parameters()):
        a = rng.standard_normal(v.shape) * 0.3
        if k.endswith("norm1.weight") or k.endswith("norm2.weight"):
            a += 1.0
        params.append((k, mx.array(a.astype(np.float32))))
    m.load_weights(params)
    m.act_dtype = mx.float32
    m.eval()
    return m, params


# One pair on a leaf of every kind: each block leaf, the fused weights,
# the refiner and the layers that feed the modulation and caption memos.
APPLIED = ("layers.0.proj.q", "layers.1.proj.ffn_up_gate",
           "layers.1.proj.o", "text_embedder.blocks.0.qkv",
           "text_embedder.blocks.1.ffn_up", "text_embedder.proj",
           "time_embedding.0", "shared_block_modulation",
           "video_head_modulation", "final_linear_video", "video_embedder")


def test_a_step_under_a_patch_table_is_the_step_of_the_merged_weights(
        tmp_path):
    """Each pass of a guided step under the table, the negative one with
    its skip set, equals the DiT whose weights hold the deltas, and the
    memos of the base step do not leak into it, nor its into a later base
    step."""
    from gmlx.image import api, lora

    strength = 0.7
    pairs = _adapter(7, 2.0, APPLIED)
    path = write_adapter(tmp_path / "a.safetensors", "kohya", pairs)
    model, params = _tiny()
    pipe = api.ImagePipeline(dit=model, family=get_family("moss_image"),
                             variant="sft")
    merged, _ = _tiny()
    held = dict(params)
    for p, (down, up, alpha) in pairs.items():
        w = np.array(held[f"{p}.weight"], np.float64)
        w += strength * alpha / down.shape[0] * (
            up.astype(np.float64) @ down.astype(np.float64))
        held[f"{p}.weight"] = mx.array(w.astype(np.float32))
    merged.load_weights(list(held.items()))
    mpipe = api.ImagePipeline(dit=merged, family=get_family("moss_image"),
                              variant="sft")
    rng = np.random.default_rng(8)
    x = mx.array(rng.standard_normal((1, 8, H, W)).astype(np.float32))
    cond = api.Conditioning(mx.array(
        rng.standard_normal((5, 24)).astype(np.float32)), "moss-lora-pos")
    sigma = np.float32(0.6875)

    def step(p, tid=None, skip=()):
        return np.array(api.denoise_step(p, x, sigma, cond,
                                         patch_table=tid, skip=skip),
                        np.float64)

    with mx.stream(mx.cpu):
        base = step(pipe)
        ad = lora.load_adapter(pipe, path)
        assert len(ad.dit) == len(APPLIED)
        tid = api.define_patch_table(
            pipe, lora.table_entries([(ad, strength)]))
        for skip in ((), (1,)):
            got, want = step(pipe, tid, skip), step(mpipe, skip=skip)
            assert rel_l2(got, want) < 1e-5, (skip, rel_l2(got, want))
            assert rel_l2(got, step(pipe, skip=skip)) > 1e-2
        assert np.array_equal(step(pipe), base)
