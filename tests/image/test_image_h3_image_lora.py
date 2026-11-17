"""MiniMax-H3-Image LoRA files: the key map against the release file's
names in each form of config, a LoRA run live on the tiny DiT against the
merged weights, with a pair on the head patching head 0 of the bank as
ComfyUI places it, and the files gmlx refuses by name, at the load and in
the CLI before a model loads. CPU."""
from __future__ import annotations

import dataclasses
import re
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import h3_image_gates as hg
from image_fixtures import fixture_view
from lora_fixtures import reference_delta, rel_l2, write_keys
from oracle_fixture import committed_fixture

from gmlx.image import api
from gmlx.image.conditioning import Conditioning
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him
from gmlx.image.h3.lora_map import lora_map
from gmlx.image.lora import UnsupportedAdapterError, install

HEADER = "h3_image_pruned_ud_q4_k_xl.header.json"
TINY = "selftest_h3_image_tiny"
HEAD = "final_layer.video_out"
# The live step against the merged float32 weights, relative L2.
RUN_TOL = 1e-5
# One pair on a leaf of every kind, the head among them.
APPLIED = ("blocks.0.attn.qkv_proj", "blocks.1.attn.out_proj",
           "blocks.0.mlp.fc1", "blocks.1.mlp.fc2",
           "blocks.1.adaln_proj.linear", "token_refiner.blocks.0.attn.qkv_proj",
           "token_refiner.blocks.0.mlp.fc2", "condition_proj",
           "video_patch_proj", "final_layer.adaln_proj.linear", HEAD)


@pytest.fixture(scope="module")
def tiny():
    weights, wm = committed_fixture(TINY + "_weights")
    # At sigma 1.0 the step reads head 0 of the bank alone.
    arrays, _ = committed_fixture(TINY + "_s0")
    weights = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    return weights, wm["inputs"]["config"], arrays


def _pipe(weights: dict, config: dict) -> api.ImagePipeline:
    # The tiny model runs at 64x64, under the family's least pixels.
    return api.ImagePipeline(
        dit=hg.tiny_model(weights, config),
        family=dataclasses.replace(hi.FAMILY, min_pixels=None),
        variant="pruned")


def _head_rows(config: dict) -> int:
    return config["latents_dim"] * 4


def _adapter(pipe, config: dict, seed: int) -> dict:
    """Target -> (down, up, alpha) for each of APPLIED, rank 4, alpha 2;
    the head's pair on one head's rows."""
    rng = np.random.default_rng(seed)
    leaves = install.linear_leaves(pipe.dit)
    out = {}
    for p in APPLIED:
        rows, inp = install.linear_dims(leaves[p])
        if p == HEAD:
            rows = _head_rows(config)
        out[p] = ((rng.standard_normal((4, inp)) * 0.2).astype(np.float32),
                  (rng.standard_normal((rows, 4)) * 0.2).astype(np.float32),
                  2.0)
    return out


def _comfy_keys(adapter: dict) -> dict:
    out = {}
    for target, (down, up, alpha) in adapter.items():
        stem = f"diffusion_model.{target}"
        out[f"{stem}.lora_A.weight"] = down
        out[f"{stem}.lora_B.weight"] = up
        out[f"{stem}.alpha"] = np.asarray(alpha, np.float32)
    return out


def _merged(weights: dict, adapter: dict, strength: float,
            head_rows: int) -> dict:
    """The weights with each pair's delta added, the head's on head 0."""
    out = dict(weights)
    for target, (down, up, alpha) in adapter.items():
        w = out[f"{target}.weight"]
        band = (0, 0, head_rows) if target == HEAD else None
        d = reference_delta([(down, up, alpha, band, strength)], w.shape)
        out[f"{target}.weight"] = (w + d).astype(np.float32)
    return out


def _step(pipe, arrays: dict, cond, tid=None) -> np.ndarray:
    lat, _, sigma, target = hg.case(arrays)
    return np.array(api.denoise_step(pipe, lat, np.float32(sigma), cond,
                                     patch_table=tid,
                                     target=np.float32(target)), np.float64)


def test_the_key_map_names_each_linear_of_the_release_file():
    """Every linear weight maps to itself, the head on head 0 of its bank
    of six, and nothing of an audio stream."""
    view = fixture_view(HEADER)
    km = hi.FAMILY.lora_map(hi.synth_config(view))
    names = {n.removesuffix(".weight") for n, t in view.tensors.items()
             if n.endswith(".weight") and len(t.shape) == 2}
    assert set(km.targets) == names
    assert km.targets[HEAD] == (HEAD, (0, 0, 96))
    assert all(km.targets[p] == (p, None) for p in names - {HEAD})
    assert (km.family, km.title) == ("h3_image", hi.TITLE)


def test_the_built_models_config_maps_as_the_header_does():
    config = hi.synth_config(fixture_view(HEADER))
    built = dataclasses.asdict(him.H3ImageConfig.from_dict(config))
    assert "block" in built and "num_layers" not in built
    assert lora_map(built).targets == lora_map(config).targets


def test_a_lora_runs_live_as_the_merged_weights(tiny, tmp_path):
    """A step under the table equals the step of the DiT whose weights
    hold the deltas, the head's pair on head 0, the one head the step
    reads; the caption the base step keeps does not leak into it, nor its
    into a later base step."""
    from gmlx.image import lora

    weights, config, arrays = tiny
    strength = 0.7
    pipe = _pipe(weights, config)
    adapter = _adapter(pipe, config, 1)
    path = write_keys(tmp_path / "a.safetensors", _comfy_keys(adapter))
    merged = _pipe(_merged(weights, adapter, strength, _head_rows(config)),
                   config)
    cond = Conditioning(mx.array(arrays["features"]), "h3-lora")
    with mx.stream(mx.cpu):
        base = _step(pipe, arrays, cond)
        ad = lora.load_adapter(pipe, path)
        assert len(ad.dit) == len(APPLIED)
        tid = api.define_patch_table(pipe,
                                     lora.table_entries([(ad, strength)]))
        got = _step(pipe, arrays, cond, tid)
        want = _step(merged, arrays, cond)
        assert rel_l2(got, want) < RUN_TOL, rel_l2(got, want)
        assert rel_l2(base, want) > 1e-2
        assert np.array_equal(_step(pipe, arrays, cond), base)


_D = 32


def _keys(stem: str, down_shape, up_shape, *, a="lora_A", b="lora_B"):
    return {f"{stem}.{a}.weight": np.zeros(down_shape, np.float32),
            f"{stem}.{b}.weight": np.zeros(up_shape, np.float32)}


# A pair over the whole bank of six heads, as a PDD LoRA writes it.
_BANK = _keys(f"diffusion_model.{HEAD}", (4, _D), (576, 4))
_DIFFSYNTH = _keys("diffusion_model.blocks.0.attn.qkv_proj", (4, _D),
                   (384, 4), a="lora_A.default", b="lora_B.default")
_CONFIG = {"diffusion_model.blocks.0.attn.comfy_attention.config":
           np.frombuffer(b'{"attention": "comfy_kitchen_sol"}', np.uint8)}
_GATE = _keys("diffusion_model.blocks.1.attn.to_gate_compress", (4, _D),
              (128, 4))


@pytest.mark.parametrize("extra, kind, names", [
    (_BANK, "PDD", "head 0 of the model's bank of 6 heads"),
    (_DIFFSYNTH, "DiffSynth", ".default."),
    (_CONFIG, "config", "blocks.0.attn.comfy_attention.config"),
    (_GATE, "FastH3 gate", "blocks.1.attn.to_gate_compress"),
], ids=["bank", "diffsynth", "config", "gate"])
def test_a_lora_gmlx_does_not_run_is_refused_by_name(tiny, tmp_path, extra,
                                                     kind, names):
    """At the load, alone and second after a file that loads, before
    either file's pairs are read."""
    from gmlx.image import lora

    weights, config, _ = tiny
    pipe = _pipe(weights, config)
    good = write_keys(tmp_path / "good.safetensors",
                      _comfy_keys(_adapter(pipe, config, 2)))
    path = write_keys(tmp_path / "lora.safetensors", dict(extra))
    with pytest.raises(UnsupportedAdapterError,
                       match=re.escape(f"{path}: gmlx does not run this LoRA "
                                       f"on {hi.TITLE}: ")) as e:
        lora.load_adapter(pipe, path)
    assert e.value.kind == kind and names in str(e.value)
    assert not pipe.adapter_files
    lora.load_adapter(pipe, good)
    with pytest.raises(UnsupportedAdapterError):
        lora.load_adapter(pipe, path)
    assert len(pipe.adapter_files) == 1


def test_the_cli_refuses_a_bank_pair_before_a_model_loads(tmp_path):
    from gmlx.commands import image as cmd

    view = fixture_view(HEADER)
    ident = SimpleNamespace(family=hi.FAMILY, config=hi.synth_config(view),
                            view=view)
    bank = _keys(f"diffusion_model.{HEAD}", (4, 5376), (576, 4))
    path = write_keys(tmp_path / "bank.safetensors", bank)
    a = SimpleNamespace(adapter=[(path, 1.0)])
    cmd._vet_adapters(a)
    with pytest.raises(UnsupportedAdapterError, match="bank of 6 heads"):
        cmd._vet_adapters(a, ident)
    head = _keys(f"diffusion_model.{HEAD}", (4, 5376), (96, 4))
    a.adapter = [(write_keys(tmp_path / "head.safetensors", head), 1.0)]
    cmd._vet_adapters(a, ident)
