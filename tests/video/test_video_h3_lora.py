"""MiniMax-H3 LoRA files: the key map against the GGUF's names, a LoRA in
ComfyUI's form run live on the tiny H3 DiT against the same run on merged
weights, the spellings ComfyUI drops, and the files gmlx refuses. CPU."""
from __future__ import annotations

import os
import re

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.lora import AdapterError, UnsupportedAdapterError
from gmlx.video import memfit, pipeline, schedule
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import model as hm
from gmlx.video.families.h3.lora_map import lora_map
from lora_fixtures import reference_delta, rel_l2, write_keys
from test_video_h3_model import TINY, tiny_weights
from video_fixtures import H3_HEADER, fixture_view

# The live run against merged float32 weights, relative L2 of what the
# run moves the latents by.
RUN_TOL = 2e-5
INNER = TINY["num_attention_heads"] * TINY["attention_head_dim"]


def _params() -> dict:
    return {k: np.array(v) for k, v in tiny_weights().items()}


def _pipe(params: dict) -> pipeline.VideoPipeline:
    m = hm.H3Transformer(TINY)
    m.load_weights([(k, mx.array(v)) for k, v in params.items()])
    m.act_dtype = mx.float32
    m.eval()
    return pipeline.VideoPipeline(dit=m, family=hf.FAMILY,  # type: ignore
                                  variant="fl2va")


def _adapter(seed: int) -> dict:
    """Target -> (down, up, alpha) on the tiny DiT. Each qkv pair is one
    pair over the whole fused weight at three times the rank, with B zero
    outside each projection's rows, as lightx2v's files write it."""
    rng = np.random.default_rng(seed)

    def pair(out, inp, rank=4, alpha=2.0):
        return ((rng.standard_normal((rank, inp)) * 0.2).astype(np.float32),
                (rng.standard_normal((out, rank)) * 0.2).astype(np.float32),
                alpha)

    def qkv(rank=4):
        down, up, _ = pair(3 * INNER, TINY["hidden_size"], 3 * rank)
        mask = np.zeros_like(up)
        for i in range(3):
            mask[i * INNER:(i + 1) * INNER, i * rank:(i + 1) * rank] = 1
        return down, up * mask, 3 * 2.0

    dim, ffn = TINY["hidden_size"], TINY["ffn_dim"]
    basis = TINY["adaln_basis"]
    return {
        "blocks.0.attn.qkv_proj": qkv(),
        "blocks.1.attn.out_proj": pair(dim, INNER),
        "blocks.0.mlp.fc1": pair(2 * ffn, dim),
        "blocks.1.mlp.fc2": pair(dim, ffn),
        "token_refiner.blocks.0.attn.qkv_proj": qkv(),
        "token_refiner.blocks.0.mlp.fc2": pair(dim, ffn),
        "condition_proj": pair(dim, TINY["text_dim"]),
        "video_patch_proj": pair(dim, 96),
        "audio_patch_proj": pair(dim, TINY["audio_in_channels"]),
        "blocks.1.adaln_proj.linear": pair(18 * dim, basis),
        "final_layer.adaln_proj.linear": pair(2 * dim, basis),
        "final_layer.video_out": pair(96, dim),
        # Rank twice the head's rows: a plain pair, not a PDD bank.
        "final_layer.audio_out": pair(TINY["audio_in_channels"], dim,
                                      2 * TINY["audio_in_channels"]),
    }


def _comfy_keys(adapter: dict) -> dict:
    out = {}
    for target, (down, up, alpha) in adapter.items():
        stem = f"diffusion_model.{target}"
        out[f"{stem}.lora_A.weight"] = down
        out[f"{stem}.lora_B.weight"] = up
        out[f"{stem}.alpha"] = np.asarray(alpha, np.float32)
    return out


def _merged(params: dict, stack) -> dict:
    out = dict(params)
    for adapter, strength in stack:
        for target, (down, up, alpha) in adapter.items():
            w = out[f"{target}.weight"]
            d = reference_delta([(down, up, alpha, None, strength)], w.shape)
            out[f"{target}.weight"] = (w + d).astype(np.float32)
    return out


def _run(pipe, table=None) -> list:
    """What two steps move the video and audio latents by."""
    rng = np.random.default_rng(3)
    vs, aus = hf.latent_shapes(5, 64, 64)
    start = tuple(mx.array(rng.standard_normal(s).astype(np.float32))
                  for s in (vs, aus))
    feats = mx.array(rng.standard_normal((1, 8, TINY["text_dim"]))
                     .astype(np.float32))
    # The CPU stream keeps the float32 products exact.
    with mx.stream(mx.cpu):
        got = pipeline.generate(pipe, feats, frames=5, width=64, height=64,
                                steps=2, shifts=hf.SHIFTS, start=start,
                                table=table)
        return [np.array(g - s, np.float64) for g, s in zip(got, start)]


def test_the_key_map_names_each_linear_of_the_gguf():
    view = fixture_view(H3_HEADER)
    km = lora_map(hf.FAMILY.synth_config(view))
    names = {n.removesuffix(".weight") for n, t in view.tensors.items()
             if n.endswith(".weight") and len(t.shape) == 2}
    assert set(km.targets) == names
    assert all(km.targets[p] == (p, None) for p in km.targets)
    # The 208 pairs of each lightx2v file.
    assert len([p for p in km.targets if "blocks." in p
                and "adaln" not in p]) == 208


_GATE = "attn.to_gate_compress"


@pytest.mark.parametrize("form", ["header", "model"])
def test_fasth3s_key_map_adds_the_gate_of_each_block(form):
    """FastH3's config, as the header gives it or as the built model holds
    it, adds the VSA gate of each block to the key map in each spelling
    ComfyUI reads, and h3's config does not."""
    import dataclasses

    from gmlx.image.lora.reader import _dit_target, _underscore_index

    config = hf.FAMILY.synth_config(fixture_view(H3_HEADER))
    if form == "model":
        config = dataclasses.asdict(hm.H3Config.from_dict(config))
    h3 = lora_map(config)
    fast = lora_map({**config, "vsa_sparsity": hf.VSA_SPARSITY})
    depth = int(config.get("depth", config.get("num_layers")))
    gates = {f"blocks.{i}.{_GATE}" for i in range(depth)}
    assert not gates & set(h3.targets)
    assert set(fast.targets) == set(h3.targets) | gates
    under = _underscore_index(fast.targets)
    assert len(under) == len(fast.targets)
    for g in gates:
        for stem in (g, f"diffusion_model.{g}",
                     "lora_unet_" + g.replace(".", "_")):
            assert stem in fast.comfy_stems
            assert _dit_target(stem, fast.targets, under)[0] == g


def test_a_comfyui_lora_runs_live_as_the_merged_weights(tmp_path):
    params = _params()
    one, two = _adapter(1), _adapter(2)
    files = {id(one): write_keys(tmp_path / "one.safetensors",
                                 _comfy_keys(one)),
             id(two): write_keys(tmp_path / "two.safetensors",
                                 _comfy_keys(two))}
    pipe = _pipe(params)
    base = _run(pipe)
    worst = 0.0
    for stack in ([(one, 1.0)], [(one, 0.5), (two, -0.4)]):
        tid = pipeline.adapter_table(pipe, [(files[id(a)], s)
                                            for a, s in stack])
        got = _run(pipe, tid)
        want = _run(_pipe(_merged(params, stack)))
        for stream, g, w, b in zip(("video", "audio"), got, want, base):
            err = rel_l2(g, w)
            worst = max(worst, err)
            assert err <= RUN_TOL, (stream, stack[0][1], err)
            # The adapter moves the run far more than the tolerance.
            assert rel_l2(b, w) > 1e-2, stream
    assert all(np.array_equal(a, b) for a, b in zip(_run(pipe), base))
    print(f"tiny H3 LoRA run: worst relative L2 {worst:.3e}")


def test_each_pair_of_the_tiny_lora_moves_the_run():
    """Without any one pair, the merged run misses the live tolerance by
    far, so the live run above sees each pair's place."""
    params, adapter = _params(), _adapter(1)
    want = _run(_pipe(_merged(params, [(adapter, 1.0)])))
    for target in adapter:
        rest = {k: v for k, v in adapter.items() if k != target}
        got = _run(_pipe(_merged(params, [(rest, 1.0)])))
        moved = max(rel_l2(g, w) for g, w in zip(got, want))
        assert moved > 100 * RUN_TOL, (target, moved)


_DIFFSYNTH = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.default.weight":
              np.zeros((4, 64), np.float32),
              "diffusion_model.blocks.0.attn.qkv_proj.lora_B.default.weight":
              np.zeros((768, 4), np.float32)}
_CONFIG = {"diffusion_model.blocks.0.attn.comfy_attention.config":
           np.frombuffer(b'{"attention": "comfy_kitchen_sol"}', np.uint8)}
# A PDD LoRA's head pair spans a bank of two heads.
_PDD = {"diffusion_model.final_layer.video_out.lora_A.weight":
        np.zeros((4, 64), np.float32),
        "diffusion_model.final_layer.video_out.lora_B.weight":
        np.zeros((192, 4), np.float32)}
_GATE_PAIR = {"diffusion_model.blocks.1.attn.to_gate_compress.lora_A.weight":
              np.zeros((4, TINY["hidden_size"]), np.float32),
              "diffusion_model.blocks.1.attn.to_gate_compress.lora_B.weight":
              np.zeros((INNER, 4), np.float32)}
_PDD_AUDIO = {"lora_unet_final_layer_audio_out.lora_down.weight":
              np.zeros((4, 64), np.float32),
              "lora_unet_final_layer_audio_out.lora_up.weight":
              np.zeros((2 * TINY["audio_in_channels"], 4), np.float32)}


@pytest.mark.parametrize("extra, kind, names", [
    (_DIFFSYNTH, "DiffSynth", ".default."),
    (_CONFIG, "config", "blocks.0.attn.comfy_attention.config"),
    (_PDD, "PDD", "final_layer.video_out"),
    (_PDD_AUDIO, "PDD", "final_layer_audio_out"),
    (_GATE_PAIR, "FastH3 gate", "blocks.1.attn.to_gate_compress"),
], ids=["diffsynth", "config", "pdd", "pdd-audio-underscore", "gate"])
def test_a_lora_gmlx_does_not_run_is_refused_by_name(tmp_path, extra, kind,
                                                     names):
    keys = dict(extra) if kind == "DiffSynth" else \
        {**_comfy_keys({k: v for k, v in _adapter(1).items()
                        if "_out" not in k}), **extra}
    path = write_keys(tmp_path / "lora.safetensors", keys)
    good = write_keys(tmp_path / "good.safetensors",
                      _comfy_keys(_adapter(2)))
    pipe = _pipe(_params())
    # Alone, and second in a stack: no file of the stack loads.
    for stack in ([(path, 1.0)], [(good, 1.0), (path, 1.0)]):
        with pytest.raises(UnsupportedAdapterError,
                           match=re.escape(f"{path}: ")) as e:
            pipeline.adapter_table(pipe, stack)
        assert e.value.kind == kind
        assert names in str(e.value)
        assert not pipe.adapter_files and len(pipe.patch_tables) == 1


@pytest.mark.parametrize("stem, down, up", [
    ("diffusion_model.blocks.1.attn.to_gate_compress", ".lora_A.weight",
     ".lora_B.weight"),
    ("blocks.1.attn.to_gate_compress", ".lora_A.weight", ".lora_B.weight"),
    ("lora_unet_blocks_1_attn_to_gate_compress", ".lora_down.weight",
     ".lora_up.weight"),
], ids=["diffusion-model", "bare", "lora-unet"])
def test_a_lora_on_gates_alone_is_refused_by_name_on_h3(tmp_path, stem,
                                                        down, up):
    """A file whose only pairs are on FastH3's gates names the gate on a
    model without it, rather than another model, in each spelling ComfyUI
    reads. The message names the pair's down key, not its alpha."""
    path = write_keys(tmp_path / "gate.safetensors", {
        stem + ".alpha": np.asarray(4.0, np.float32),
        stem + down: np.zeros((4, TINY["hidden_size"]), np.float32),
        stem + up: np.zeros((INNER, 4), np.float32)})
    pipe = _pipe(_params())
    with pytest.raises(UnsupportedAdapterError, match=re.escape(
            f"its pair {stem}{down} targets FastH3's attention gate, which "
            "the model's file does not have")) as e:
        pipeline.adapter_table(pipe, [(path, 1.0)])
    assert e.value.kind == "FastH3 gate"
    assert not pipe.adapter_files and len(pipe.patch_tables) == 1


def test_another_models_lora_in_a_stack_loads_nothing(tmp_path):
    good = write_keys(tmp_path / "good.safetensors",
                      _comfy_keys(_adapter(1)))
    stem = "diffusion_model.double_blocks.0.img_attn.qkv"
    flux = write_keys(tmp_path / "flux.safetensors",
                      {f"{stem}.lora_A.weight": np.zeros((4, 64), np.float32),
                       f"{stem}.lora_B.weight": np.zeros((192, 4),
                                                         np.float32)})
    pipe = _pipe(_params())
    with pytest.raises(AdapterError, match=re.escape(str(flux))):
        pipeline.adapter_table(pipe, [(good, 1.0), (flux, 1.0)])
    assert not pipe.adapter_files and len(pipe.patch_tables) == 1


def test_a_seed_is_one_draw_of_the_start_noise():
    pipe = _pipe(_params())
    feats = mx.zeros((1, 8, TINY["text_dim"]))
    kw = dict(frames=5, width=64, height=64, steps=1, shifts=hf.SHIFTS)
    with mx.stream(mx.cpu):
        nv, na = pipeline.noise(pipe, 5, 64, 64, seed=7)
        assert (nv.shape, na.shape) == hf.latent_shapes(5, 64, 64)
        got = pipeline.generate(pipe, feats, seed=7, **kw)
        want = pipeline.generate(pipe, feats, start=(nv, na), **kw)
        other = pipeline.generate(pipe, feats, seed=8, **kw)
    assert all(np.array_equal(g, w) for g, w in zip(got, want))
    assert not np.array_equal(got[0], other[0])


def test_a_pacer_runs_inside_each_forward_and_can_stop_the_run():
    """A job's pacer reaches the pause points inside the forwards in
    place of the default, which evaluates there, so the latents are the
    default's; one that raises ends the run in its first forward."""
    pipe = _pipe(_params())
    feats = mx.zeros((1, 8, TINY["text_dim"]))
    kw = dict(frames=5, width=64, height=64, steps=2, shifts=hf.SHIFTS,
              seed=7)
    seen = []

    def pacer(arrays):
        seen.append(len(arrays))
        mx.eval(*arrays)

    class Stopped(Exception):
        pass

    def stop(arrays):
        raise Stopped

    steps = []
    with mx.stream(mx.cpu):
        want = pipeline.generate(pipe, feats, **kw)
        got = pipeline.generate(pipe, feats, pacer=pacer, **kw)
        with pytest.raises(Stopped):
            pipeline.generate(pipe, feats, pacer=stop,
                              on_step=lambda i, n: steps.append(i), **kw)
    assert len(seen) >= 2 * TINY["num_layers"]
    assert all(np.array_equal(g, w) for g, w in zip(got, want))
    assert steps == []


@pytest.mark.parametrize("stem, dropped", [
    ("lora_unet_blocks_1_mlp_fc2", False),
    ("diffusion_model.blocks.1.mlp.fc2", False),
    ("blocks.1.mlp.fc2", False),
    ("transformer.blocks.1.mlp.fc2", True),
    ("lycoris_blocks_1_mlp_fc2", True),
], ids=["lora_unet", "diffusion_model", "bare", "transformer",
         "lycoris"])
def test_a_spelling_comfyui_drops_loads_with_a_warning(tmp_path, stem,
                                                      dropped):
    down, up, alpha = _adapter(1)["blocks.1.mlp.fc2"]
    path = write_keys(tmp_path / "lora.safetensors",
                      {f"{stem}.lora_A.weight": down,
                       f"{stem}.lora_B.weight": up,
                       f"{stem}.alpha": np.asarray(alpha, np.float32)})
    logged: list = []
    pipe = _pipe(_params())
    pipe.log = logged.append
    pipeline.adapter_table(pipe, [(path, 1.0)])
    assert bool([w for w in logged if "ComfyUI" in w]) == dropped, logged


def test_given_sigma_tables_replace_the_shifts():
    """An oracle's tables run as the schedule of the shifts that make
    them."""
    pipe = _pipe(_params())
    feats = mx.ones((1, 8, TINY["text_dim"]))
    kw = dict(frames=5, width=64, height=64, steps=2, seed=3)
    other = {"video": 6.0, "audio": 2.0}
    got = pipeline.generate(pipe, feats, shifts=hf.SHIFTS,
                            sigmas=schedule.stream_sigmas(2, other), **kw)
    want = pipeline.generate(pipe, feats, shifts=other, **kw)
    for g, w in zip(got, want):
        assert np.array_equal(np.array(g), np.array(w))
    with pytest.raises(ValueError, match="3 entries"):
        pipeline.generate(pipe, feats, shifts=hf.SHIFTS,
                          sigmas=schedule.stream_sigmas(3, other), **kw)


@pytest.mark.parametrize("variant,steps,rungs", [
    (hf.FASTH3, 8, hf.FASTH3_RUNGS), (hf.FASTH3, 2, ()), ("fl2va", 8, ())])
def test_a_variant_at_its_trained_count_samples_its_timesteps(
        monkeypatch, variant, steps, rungs):
    """FastH3 at 8 steps runs the sigmas of its 8 rungs, and at another
    count, as every other variant, the even grid of its shifts."""
    from gmlx.video import api

    got: list = []
    monkeypatch.setattr(api, "sample", lambda *a, **k: got.append(k) or a[1:3])
    pipe = _pipe(_params())
    pipe.variant = variant
    shifts = {"video": 10.0, "audio": 3.0}
    pipeline.generate(pipe, mx.ones((1, 8, TINY["text_dim"])), frames=5,
                      width=64, height=64, steps=steps, shifts=shifts, seed=3)
    sig = got[0]["sigmas"]
    if not rungs:
        assert sig is None
        return
    want = schedule.stream_sigmas(steps, shifts, rungs)
    assert {k: v.tolist() for k, v in sig.items()} == {
        k: v.tolist() for k, v in want.items()}
    assert sig["video"][0] != schedule.sigmas(steps, 10.0)[0]


def test_a_lora_over_the_adapter_cache_still_builds_its_table(tmp_path):
    """The memory plan prices every file a run names, so the cache holds
    them all, whatever its bound."""
    path = write_keys(tmp_path / "lora.safetensors", _comfy_keys(_adapter(1)))
    pipe = _pipe(_params())
    pipe.adapters.max_bytes = os.path.getsize(path) // 2
    tid = pipeline.adapter_table(pipe, [(path, 1.0)])
    assert pipe.patch_tables[tid]["entries"]


def test_a_lone_bf16_lora_holds_its_file_once(tmp_path):
    """On a bf16 DiT the patch table holds no copy of a lone bf16 LoRA,
    on the leaves H3 runs in float32 too (the adaLN linears, the patch
    projections and the heads), as memfit prices it."""
    keys = {k: mx.array(v).astype(mx.bfloat16) if v.ndim == 2
            else mx.array(v) for k, v in _comfy_keys(_adapter(1)).items()}
    path = str(tmp_path / "lora.safetensors")
    mx.save_safetensors(path, keys, metadata={"format": "pt"})
    params = _params()
    m = hm.H3Transformer(TINY)
    keep = ("rope.inv_freq", "adaln_t_table")
    m.load_weights([(k, mx.array(v) if k in keep
                     else mx.array(v).astype(mx.bfloat16))
                    for k, v in params.items()])
    m.act_dtype = mx.bfloat16
    m.eval()
    pipe = pipeline.VideoPipeline(dit=m, family=hf.FAMILY,  # type: ignore
                                  variant="fl2va")
    tid = pipeline.adapter_table(pipe, [(path, 1.0)])
    rng = np.random.default_rng(3)
    vs, aus = hf.latent_shapes(5, 64, 64)
    start = tuple(mx.array(rng.standard_normal(s).astype(np.float32))
                  for s in (vs, aus))
    feats = mx.array(rng.standard_normal((1, 8, TINY["text_dim"]))
                     .astype(np.float32))
    mx.eval(pipeline.generate(pipe, feats, frames=5, width=64, height=64,
                              steps=1, shifts=hf.SHIFTS, start=start,
                              table=tid))
    assert memfit.lora_dtype(path) == "BF16"
    assert pipe.adapters.bytes <= os.path.getsize(path)
    assert pipe.tables.bytes == 0


def test_a_home_path_loads_and_a_missing_file_is_refused(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write_keys(tmp_path / "lora.safetensors", _comfy_keys(_adapter(1)))
    pipe = _pipe(_params())
    pipeline.adapter_table(pipe, [("~/lora.safetensors", 1.0)])
    with pytest.raises(AdapterError, match="nope.safetensors"):
        pipeline.adapter_table(pipe, [(tmp_path / "nope.safetensors", 1.0)])


_D = TINY["text_dim"]


@pytest.mark.parametrize("shape", [(8, _D), (1, 0, _D), (1, 8, _D + 1),
                                   (2, 8, _D)])
def test_features_that_are_not_one_prompt_are_refused(shape):
    with pytest.raises(ValueError, match=re.escape(f"(1, L, {_D})")):
        pipeline.generate(_pipe(_params()), mx.zeros(shape), frames=5,
                          width=64, height=64, steps=1, shifts=hf.SHIFTS,
                          seed=0)
