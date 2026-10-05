"""The Z-Image LoRA key map, band placement on the fused qkv weight, and
the delta half of acceptance check 7: the placed factors against a float64
reference, per dialect. CPU."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.families.z_image.lora_map import lora_map
from gmlx.image.lora import plan, reader
from gmlx.image.registry import get_family
from image_fixtures import (TINY_Z_CONFIG, Z_HEADER, fixture_view, load_json,
                            official_config)
from lora_fixtures import (DIALECTS, make_pair, reference_delta, rel_l2,
                           tiny_z_adapter, write_adapter)

DIM = TINY_Z_CONFIG["dim"]
QKV = "layers.1.attention.qkv"
# Check 7 delta half: relative L2 against the float64 reference.
DELTA_TOL = 1e-6


def _dims(config: dict) -> dict:
    """Module path to (out, in) of every linear leaf of the built model."""
    import mlx.nn as nn

    from gmlx.image.families.z_image.model import ZImageDiT

    return {p: tuple(m.weight.shape)
            for p, m in ZImageDiT(config).named_modules()
            if isinstance(m, nn.Linear)}


def test_the_key_map_covers_every_linear_leaf_and_only_those():
    km = lora_map(TINY_Z_CONFIG)
    dims = _dims(TINY_Z_CONFIG)
    assert km.family == "z_image" and get_family("z_image").lora_map
    assert {m for m, _ in km.targets.values()} == set(dims)
    for name, (module, band) in km.targets.items():
        out, inp = dims[module]
        if band is not None:
            assert band[0] == 0 and band[1] + band[2] <= out, name
    # Native names map to themselves on the whole weight.
    for path in dims:
        assert km.targets[path] == (path, None)


def test_the_key_map_names_every_linear_weight_of_the_real_header():
    cfg = get_family("z_image").synth_config(fixture_view(Z_HEADER))
    km = lora_map(cfg)
    assert official_config("z_image_turbo_transformer_config.json")[
        "dim"] == cfg["dim"] == 3840
    linear = {n[:-len(".weight")] for n, shape, _ in
              load_json(Z_HEADER)["tensors"]
              if n.endswith(".weight") and len(shape) == 2}
    assert linear and linear == {m for m, _ in km.targets.values()}
    assert len([n for n in linear if n.endswith("attention.qkv")]) == 34


def test_q_k_and_v_are_bands_0_1_and_2_of_the_fused_weight():
    cfg = get_family("z_image").synth_config(fixture_view(Z_HEADER))
    km = lora_map(cfg)
    for group, n in (("layers", 30), ("noise_refiner", 2),
                     ("context_refiner", 2)):
        for i in (0, n - 1):
            x = f"{group}.{i}.attention"
            assert km.targets[f"{x}.to_q"] == (f"{x}.qkv", (0, 0, 3840))
            assert km.targets[f"{x}.to_k"] == (f"{x}.qkv", (0, 3840, 3840))
            assert km.targets[f"{x}.to_v"] == (f"{x}.qkv", (0, 7680, 3840))
            assert km.targets[f"{x}.to_out.0"] == (f"{x}.out", None)
        assert f"{group}.{n}.attention.to_q" not in km.targets
    assert km.targets["all_x_embedder.2-1"] == ("x_embedder", None)
    assert km.targets["all_final_layer.2-1.linear"] == (
        "final_layer.linear", None)
    assert "context_refiner.0.adaLN_modulation.0" not in km.targets
    assert "layers.0.attention.to_q" in km.diffusers_only
    assert "layers.0.attention.qkv" not in km.diffusers_only
    assert km.text_names == ("qwen3_4b",)
    assert "model.layers.35.mlp.down_proj" in km.text_targets


def test_the_block_diagonal_pair_places_each_band_in_its_rows():
    rng = np.random.default_rng(0)
    q = make_pair(rng, 4, 5, 2, 6.0)
    v = make_pair(rng, 4, 5, 3, None)
    whole = make_pair(rng, 12, 5, 1, 2.0)
    parts = [plan.Part(mx.array(q[0]), mx.array(q[1]), 6.0, (0, 0, 4), 0.5),
             plan.Part(mx.array(v[0]), mx.array(v[1]), None, (0, 8, 4), -1.5),
             plan.Part(mx.array(whole[0]), mx.array(whole[1]), 2.0, None,
                       1.0)]
    lm = plan.block_diagonal("m", parts, (12, 5))
    a, b = np.array(lm.a), np.array(lm.b)
    assert a.shape == (6, 5) and b.shape == (12, 6)
    assert lm.rank == 6 and lm.scale == 1.0 and lm.a.dtype == mx.float32
    assert np.array_equal(a[:2], q[0]) and np.array_equal(a[2:5], v[0])
    # Each band owns its rank columns, and B is zero outside the band rows.
    assert not b[4:, :2].any() and not b[:8, 2:5].any()
    assert np.array_equal(b[:4, :2], q[1] * np.float32(0.5 * 6.0 / 2))
    assert np.array_equal(b[8:, 2:5], v[1] * np.float32(-1.5))
    assert np.array_equal(b[:, 5:], whole[1] * np.float32(2.0))
    ref = reference_delta([(q[0], q[1], 6.0, (0, 0, 4), 0.5),
                           (v[0], v[1], None, (0, 8, 4), -1.5),
                           (whole[0], whole[1], 2.0, None, 1.0)], (12, 5))
    assert rel_l2(b.astype(np.float64) @ a.astype(np.float64),
                  ref) <= DELTA_TOL
    assert ref[4:8].any()                 # only the whole pair is there
    assert plan.pair_scale(None, 7) == 1.0 and plan.pair_scale(3.0, 4) == 0.75


def test_an_input_band_pads_the_columns_of_a():
    rng = np.random.default_rng(1)
    d, u, _ = make_pair(rng, 6, 3, 2)
    lm = plan.block_diagonal(
        "m", [plan.Part(mx.array(d), mx.array(u), 4.0, (1, 5, 3), 1.0)],
        (6, 9))
    a, b = np.array(lm.a, np.float64), np.array(lm.b, np.float64)
    assert a.shape == (2, 9) and not a[:, :5].any() and not a[:, 8:].any()
    assert rel_l2(b @ a, reference_delta([(d, u, 4.0, (1, 5, 3), 1.0)],
                                         (6, 9))) <= DELTA_TOL


def test_banded_keeps_one_pair_per_band_or_declines():
    rng = np.random.default_rng(2)
    P = lambda out, rank, band, s=1.0: plan.Part(   # noqa: E731
        *(mx.array(x) for x in make_pair(rng, out, 5, rank)[:2]), None, band,
        s)
    rows = plan.banded("m", [P(4, 2, (0, 8, 4)), P(4, 3, (0, 0, 4)),
                             P(4, 1, (0, 8, 4), 0.5)], (12, 5))
    assert [(s, e, lm.rank) for s, e, lm in rows] == [(0, 4, 3), (8, 12, 3)]
    assert rows[1][2].b.shape == (4, 3) and rows[1][2].a.shape == (3, 5)
    assert plan.banded("m", [P(12, 2, None), P(4, 2, (0, 0, 4))],
                       (12, 5)) is None
    assert plan.banded("m", [P(12, 2, (1, 0, 5))], (12, 5)) is None
    whole = plan.banded("m", [P(12, 2, None), P(12, 1, None)], (12, 5))
    assert [(s, e, lm.rank) for s, e, lm in whole] == [(0, 12, 3)]


def test_fit_names_the_mismatch():
    assert plan.fit((4, 2), (2, 5), (0, 4, 4), (12, 5)) is None
    assert plan.fit((12, 2), (2, 5), None, (12, 5)) is None
    assert "differ in rank" in plan.fit((4, 2), (3, 5), None, (4, 5))
    assert "4x5 (out x in) and the target is 12x5" in plan.fit(
        (4, 2), (2, 5), None, (12, 5))
    assert "target is 4x5" in plan.fit((12, 2), (2, 5), (0, 0, 4), (12, 5))
    assert "target is 12x3" in plan.fit((12, 2), (2, 5), (1, 0, 3), (12, 5))


def _parts(path, strength, keymap):
    """Module and band to plan parts of an adapter file."""
    res = reader.resolve(reader.read_lora(str(path)), keymap)
    out: dict = {}
    for t in res.dit:
        out.setdefault(t.module, []).append(plan.Part(
            t.pair.down, t.pair.up, t.pair.alpha, t.band, strength))
    return out


def _reference(adapter: dict, strength: float, keymap) -> dict:
    out: dict = {}
    for target, (down, up, alpha) in adapter.items():
        module, band = keymap.targets[target]
        out.setdefault(module, []).append((down, up, alpha, band, strength))
    return out


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_delta_half_per_dialect(tmp_path, dialect):
    """Strengths 0.5 and 1.0 and a stacked pair, against float64."""
    km = lora_map(TINY_Z_CONFIG)
    dims = _dims(TINY_Z_CONFIG)
    has_alpha = DIALECTS[dialect][4]
    one = tiny_z_adapter(1, 6.0 if has_alpha else None)
    two = tiny_z_adapter(2, 1.0 if has_alpha else None)
    p1 = write_adapter(tmp_path / "one.safetensors", dialect, one)
    p2 = write_adapter(tmp_path / "two.safetensors", dialect, two)
    cases = [([(p1, one, 0.5)], "0.5"), ([(p1, one, 1.0)], "1.0"),
             ([(p1, one, 0.75), (p2, two, -0.4)], "stacked")]
    worst = 0.0
    for stack, label in cases:
        got: dict = {}
        want: dict = {}
        for path, adapter, strength in stack:
            for m, parts in _parts(path, strength, km).items():
                got.setdefault(m, []).extend(parts)
            for m, rows in _reference(adapter, strength, km).items():
                want.setdefault(m, []).extend(rows)
        assert set(got) == set(want) and len(got) == 6
        for module, parts in got.items():
            lm = plan.block_diagonal(module, parts, dims[module])
            delta = np.array(lm.b, np.float64) @ np.array(lm.a, np.float64)
            ref = reference_delta(want[module], dims[module])
            err = rel_l2(delta, ref)
            worst = max(worst, err)
            assert err <= DELTA_TOL, (dialect, label, module, err)
    # q, k and v at ranks 4, 2 and 8 share one pair of rank 14.
    lm = plan.block_diagonal(QKV, _parts(p1, 1.0, km)[QKV], dims[QKV])
    assert lm.a.shape == (14, DIM) and lm.b.shape == (3 * DIM, 14)
    print(f"check7 delta {dialect}: worst relative L2 {worst:.3e}")


def test_a_pair_that_spells_another_family_resolves_nowhere(tmp_path):
    km = lora_map(TINY_Z_CONFIG)
    rng = np.random.default_rng(3)
    flux = {"transformer_blocks.0.attn.to_q": make_pair(rng, 8, 8, 2),
            "single_transformer_blocks.3.proj_out": make_pair(rng, 8, 8, 2)}
    for dialect in ("kohya", "diffusers"):
        p = write_adapter(tmp_path / f"{dialect}.safetensors", dialect, flux)
        with pytest.raises(reader.AdapterFamilyError,
                           match="not a Z-Image adapter"):
            reader.resolve(reader.read_lora(p), km)
