"""The rope swap on class-built models: every attention rope takes the
resolved table and llama.cpp's magnitude, sliding layers take the sliding
scaling, and a plain rope leaves the model as it was. Tiny synthesized
models, no files."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.load import rope_scaling as rs
from gmlx.load import rope_swap
from gmlx.load.config_synth import synthesize_config
from gmlx.load.loader import build_model
from gmlx.load.rope_scaling import RopeOverride
from test_config_synth import (
    VOCAB,
    _MINIMAX_SHAPES,
    _base_meta,
    _glm4moe_meta,
    _gpt_oss_meta,
    _meta_for,
    _minimax_meta,
    _qwen3next_meta,
)

YARN4 = RopeOverride(rope_scaling="yarn", rope_scale=4.0)


def _moe(arch, ffn=64):
    m = _base_meta(arch)
    m[f"{arch}.expert_count"] = 4
    m[f"{arch}.expert_used_count"] = 2
    m[f"{arch}.expert_feed_forward_length"] = ffn
    return m


# arch -> (meta, tensor shapes)
ARCHES = {
    "llama": lambda: (_meta_for("llama"), {}),
    "qwen2": lambda: (_meta_for("qwen2"), {}),
    "qwen3": lambda: (_meta_for("qwen3"), {}),
    "glm4": lambda: (_meta_for("glm4"), {}),
    "seed_oss": lambda: (_meta_for("seed_oss"), {}),
    "smollm3": lambda: (_meta_for("smollm3"), {}),
    "granite": lambda: (_meta_for("granite"), {}),
    "qwen3moe": lambda: (_moe("qwen3moe"), {}),
    "qwen2moe": lambda: (_moe("qwen2moe"),
                         {"blk.0.ffn_gate_exps.weight": [64, 48, 4]}),
    "glm4moe": lambda: (_glm4moe_meta(), {"output.weight": [64, VOCAB]}),
    "gpt-oss": lambda: (_gpt_oss_meta(), {"output.weight": [64, VOCAB]}),
    "minimax-m2": lambda: (_minimax_meta(), _MINIMAX_SHAPES),
    "qwen3next": lambda: (_qwen3next_meta(), {}),
    "mistral3": lambda: (_meta_for("mistral3"), {}),
}


def _build(arch, override=None, meta_edit=None):
    meta, shapes = ARCHES[arch]()
    if meta_edit:
        meta_edit(meta)
    cfg = synthesize_config(meta, shapes, rope_override=override)
    model, cfg = build_model(cfg)
    n = rope_swap.apply(model, cfg)
    return model, cfg, n, meta


def _ropes(model):
    return [(name, m.rope) for name, m in model.named_modules()
            if isinstance(getattr(m, "rope", None), nn.Module)]


def _ref_rotate(x, pos, dims, inv, mscale, traditional):
    """float64 rope of the first ``dims`` of ``x [..., D]`` at one position."""
    x = np.asarray(x.astype(mx.float32), dtype=np.float64)
    ang = (np.float32(pos) * inv.astype(np.float32)).astype(np.float64)
    c, s = np.cos(ang) * mscale, np.sin(ang) * mscale
    out = x.copy()
    if traditional:
        a, b = x[..., 0:dims:2], x[..., 1:dims:2]
        out[..., 0:dims:2] = a * c - b * s
        out[..., 1:dims:2] = b * c + a * s
    else:
        h = dims // 2
        a, b = x[..., :h], x[..., h:dims]
        out[..., :h] = a * c - b * s
        out[..., h:dims] = b * c + a * s
    return out


def _check_rope(r, scaling, base, pos=300):
    """Rotate a probe with module ``r`` and compare with the reference
    rotation from the resolver's table."""
    mx.random.seed(0)
    x = mx.random.normal((1, 1, 1, r.dims + 4))
    inv = rs.inv_freq(r.dims, base, scaling).astype(np.float64)
    got = np.asarray(r(x, offset=pos))
    want = _ref_rotate(x, pos, r.dims, inv, scaling.mscale, r.traditional)
    np.testing.assert_allclose(got, want, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("arch", sorted(ARCHES))
def test_options_reach_every_rope(arch):
    assert arch in rs.SWAP_ARCHES and arch in rs.WIRED_ARCHES
    model, cfg, n, meta = _build(arch, YARN4)
    ropes = _ropes(model)
    assert n > 0 and ropes
    assert all(isinstance(r, rope_swap.ScaledRoPE) for _, r in ropes
               if type(r).__name__ != "NoPE")
    s = rs.from_config(cfg)
    assert s.factor == 4.0 and s.type == "yarn"
    base = meta[f"{arch}.rope.freq_base"]
    sliding = {i for i, t in enumerate(cfg.get("layer_types") or [])
               if t == "sliding_attention"}
    for name, r in ropes:
        if not isinstance(r, rope_swap.ScaledRoPE):
            continue
        layer = int(name.split("layers.")[1].split(".")[0])
        _check_rope(r, s.for_sliding() if layer in sliding else s, base)
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[-1] == VOCAB
    # A second pass changes nothing.
    assert rope_swap.apply(model, cfg) == 0


@pytest.mark.parametrize("arch", sorted(set(ARCHES) - {"gpt-oss"}))
def test_plain_rope_leaves_the_model_as_built(arch):
    model, cfg, n, _ = _build(arch)
    assert n == 0 and rs.CONFIG_KEY not in cfg
    assert not any(isinstance(r, rope_swap.ScaledRoPE)
                   for _, r in _ropes(model))


def _header_yarn(arch, factor=4.0, orig=None):
    def edit(meta):
        meta[f"{arch}.rope.scaling.type"] = "yarn"
        meta[f"{arch}.rope.scaling.factor"] = factor
        if orig is not None:
            meta[f"{arch}.rope.scaling.original_context_length"] = orig
    return edit


@pytest.mark.parametrize("arch", ["qwen3moe", "glm4moe", "qwen2moe"])
def test_header_yarn_reaches_classes_that_ignored_it(arch):
    """These classes build their rope from the base alone, so a GGUF with
    yarn keys ran with a plain rope."""
    model, cfg, n, meta = _build(arch, meta_edit=_header_yarn(arch, 4.0, 256))
    s = rs.from_config(cfg)
    assert n > 0 and s.factor == 4.0 and s.original_context == 256
    _, r = _ropes(model)[0]
    _check_rope(r, s, meta[f"{arch}.rope.freq_base"])
    assert r.mscale == pytest.approx(1 + 0.1 * math.log(4.0), rel=1e-6)


@pytest.mark.parametrize("arch", ["qwen3", "seed_oss", "granite", "smollm3"])
def test_header_yarn_without_original_context_uses_the_trained_one(arch):
    """llama.cpp falls back to the trained context (1024 here); the class
    defaults put the ramp at 4096."""
    model, cfg, n, meta = _build(arch, meta_edit=_header_yarn(arch))
    s = rs.from_config(cfg)
    assert s.original_context == 1024
    for _, r in _ropes(model):
        if isinstance(r, rope_swap.ScaledRoPE):
            _check_rope(r, s, meta[f"{arch}.rope.freq_base"])


def test_gpt_oss_sliding_layers_keep_the_trained_scale():
    """gpt-oss sets rope_freq_scale_train_swa to the trained scale, so its
    sliding layers keep factor 32 under a factor 4 option while the full
    layers take 4 (llama.cpp openai-moe loader)."""
    model, cfg, n, meta = _build("gpt-oss", YARN4)
    s = rs.from_config(cfg)
    assert s.factor == 4.0 and s.for_sliding().factor == 32.0
    layers = cfg["layer_types"]
    seen = set()
    for name, r in _ropes(model):
        i = int(name.split("layers.")[1].split(".")[0])
        want = s.for_sliding() if layers[i] == "sliding_attention" else s
        _check_rope(r, want, meta["gpt-oss.rope.freq_base"])
        seen.add(layers[i])
    assert seen == {"sliding_attention", "full_attention"}
    # Header only: every layer is the header's factor 32 table.
    model, cfg, n, meta = _build("gpt-oss")
    s = rs.from_config(cfg)
    assert n > 0 and s.factor == 32.0
    for _, r in _ropes(model):
        _check_rope(r, s, meta["gpt-oss.rope.freq_base"])


def test_header_yarn_through_the_swap_matches_the_class_rope():
    """On a class that read the header correctly (gpt-oss), the swapped
    rope rotates as the class's own yarn rope did."""
    meta, shapes = ARCHES["gpt-oss"]()
    cfg = synthesize_config(meta, shapes)
    before, _ = build_model(cfg)
    after, cfg2 = build_model(cfg)
    assert rope_swap.apply(after, cfg2) > 0
    mx.random.seed(1)
    x = mx.random.normal((1, 2, 3, 16))
    for (_, a), (_, b) in zip(_ropes(before), _ropes(after)):
        np.testing.assert_allclose(np.asarray(a(x, offset=5000)),
                                   np.asarray(b(x, offset=5000)),
                                   atol=2e-4, rtol=1e-4)


def test_freq_base_option_moves_full_layers_only():
    ov = RopeOverride(rope_scaling="yarn", rope_scale=4.0,
                      rope_freq_base=500000.0)
    model, cfg, n, meta = _build("gpt-oss", ov)
    s = rs.from_config(cfg)
    for name, r in _ropes(model):
        i = int(name.split("layers.")[1].split(".")[0])
        if cfg["layer_types"][i] == "sliding_attention":
            _check_rope(r, s.for_sliding(), meta["gpt-oss.rope.freq_base"])
        else:
            _check_rope(r, s, 500000.0)


def test_factored_rope_composes_with_the_scaling():
    """llama rope_freqs factors divide the angle before the yarn blend."""
    from gmlx.load.loader import _patch_rope_factors

    meta, shapes = ARCHES["llama"]()
    cfg = synthesize_config(meta, shapes, rope_override=YARN4)
    model, cfg = build_model(cfg)
    factors = mx.linspace(1.0, 8.0, 8)
    _patch_rope_factors(model, factors)
    assert rope_swap.apply(model, cfg) > 0
    s = rs.from_config(cfg)
    _, r = _ropes(model)[0]
    inv = rs.inv_freq(16, 10000.0, s) / np.asarray(factors, dtype=np.float64)
    np.testing.assert_allclose(1.0 / np.asarray(r._freqs), inv, rtol=1e-6)


def test_unswappable_rope_refuses():
    class Odd(nn.Module):
        def __call__(self, x, offset=0):
            return x

    model, cfg, n, _ = _build("qwen3", YARN4)
    model.model.layers[0].self_attn.rope = type("OddRoPE", (Odd,), {})()
    with pytest.raises(NotImplementedError, match="no scaled form"):
        rope_swap.apply(model, cfg)


def test_scaled_rope_partial_dims_pass_through():
    r = rope_swap.ScaledRoPE(8, False, mx.full((4,), 100.0), 1.25)
    x = mx.random.normal((1, 1, 2, 12))
    out = r(x, offset=3)
    np.testing.assert_array_equal(np.asarray(out[..., 8:]),
                                  np.asarray(x[..., 8:]))


@pytest.mark.parametrize("family", ["llava", "pixtral"])
def test_vlm_text_tower_builds_and_swaps_under_yarn(family):
    """The llava and pixtral text configs raised on a yarn type; they now
    build a plain rope that the swap rebuilds."""
    import importlib

    meta = _meta_for("llama")
    meta["llama.rope.scaling.type"] = "yarn"
    meta["llama.rope.scaling.factor"] = 4.0
    meta["llama.rope.scaling.original_context_length"] = 256
    cfg = synthesize_config(meta, {})
    assert cfg["rope_scaling"]["type"] == "yarn"
    mod = importlib.import_module(f"gmlx._vendor.mlx_vlm.models.{family}")
    tc = mod.TextConfig.from_dict({**cfg, "model_type": "llama"})
    lm = mod.LanguageModel(tc)
    assert rope_swap.apply(lm, cfg) > 0
    s = rs.from_config(cfg)
    for _, r in _ropes(lm):
        _check_rope(r, s, 10000.0)


# More class-built archs: the same swap, checked on the
# built rope modules (their forwards need caches or kernels the tiny
# synthesized models skip).
def _later_arches():
    import test_config_synth as t

    return {
        "hy_v3": (t._hy_v3_meta, t._HY_V3_SHAPES),
        "hyv4": (t._hyv4_meta, t._HYV4_SHAPES),
        "muse-glimmer": (t._muse_glimmer_meta, t._MUSE_GLIMMER_SHAPES),
        "muse-spark": (t._muse_spark_meta, t._MUSE_SPARK_SHAPES),
        "k2-horizon": (t._k2_horizon_mova_meta, t._K2_HORIZON_MOVA_SHAPES),
        "minimax-m3": (t._minimax_m3_meta, t._MINIMAX_M3_SHAPES),
        "falcon-h1": (t._falcon_h1_meta, t._FALCON_H1_SHAPES),
        "granitehybrid": (t._granitehybrid_meta, {}),
        "ernie4_5-moe": (t._ernie_meta, {}),
    }


@pytest.mark.parametrize("arch", ["hy_v3", "hyv4", "muse-glimmer",
                                  "muse-spark", "k2-horizon", "minimax-m3",
                                  "falcon-h1", "granitehybrid", "ernie4_5-moe"])
def test_later_class_built_arches_take_the_options(arch):
    assert arch in rs.SWAP_ARCHES and arch in rs.WIRED_ARCHES
    meta_fn, shapes = _later_arches()[arch]
    meta = meta_fn()
    plain = synthesize_config(meta, dict(shapes))
    before, _ = build_model(plain)
    bases = {name: float(r.base) for name, r in _ropes(before)
             if hasattr(r, "base")}
    cfg = synthesize_config(meta, dict(shapes), rope_override=YARN4)
    model, cfg = build_model(cfg)
    assert rope_swap.apply(model, cfg) > 0
    s = rs.from_config(cfg)
    assert s.factor == 4.0
    sliding = {i for i, t in enumerate(cfg.get("layer_types") or [])
               if t in ("sliding_attention", "sliding")}
    for name, r in _ropes(model):
        assert isinstance(r, rope_swap.ScaledRoPE), name
        layer = name.split("layers.")
        i = int(layer[1].split(".")[0]) if len(layer) > 1 else -1
        base = bases.get(name) or meta.get(f"{arch}.rope.freq_base", 10000.0)
        _check_rope(r, s.for_sliding() if i in sliding else s, base)


# Gemma families. llama.cpp lays sliding layers out with load_swa_pattern:
# period 2 for gemma2, 6 for gemma3, the last layer of each period full.
_GEMMA_PERIOD = {"gemma": None, "gemma2": 2, "gemma3": 6}


def _gemma_build(arch, override=None, meta_edit=None):
    meta = _meta_for(arch)
    if arch == "gemma3":
        meta["gemma3.rope.freq_base"] = 1000000.0
    if meta_edit:
        meta_edit(meta)
    before, _ = build_model(synthesize_config(meta, {}))
    cfg = synthesize_config(meta, {}, rope_override=override)
    model, cfg = build_model(cfg)
    n = rope_swap.apply(model, cfg)
    return before, model, cfg, n


def _layer_ropes(model):
    return [(int(name.split("layers.")[1].split(".")[0]), r)
            for name, r in _ropes(model) if "layers." in name]


@pytest.mark.parametrize("arch", ["gemma", "gemma2", "gemma3"])
def test_gemma_families_take_the_options(arch):
    assert arch in rs.SWAP_ARCHES and arch in rs.WIRED_ARCHES
    before, model, cfg, n = _gemma_build(arch, YARN4)
    s = rs.from_config(cfg)
    assert n > 0 and s.factor == 4.0
    bases = {i: float(r.base) for i, r in _layer_ropes(before)}
    period = _GEMMA_PERIOD[arch]
    kinds = set()
    for i, r in _layer_ropes(model):
        assert isinstance(r, rope_swap.ScaledRoPE)
        swa = period is not None and i % period < period - 1
        kinds.add(swa)
        _check_rope(r, s.for_sliding() if swa else s, bases[i])
    assert kinds == ({False} if period is None else {True, False})
    mx.eval(model(mx.array([[1, 2, 3]])))
    assert rope_swap.apply(model, cfg) == 0


def test_gemma2_sliding_layers_keep_the_trained_scale():
    """gemma2.cpp sets rope_freq_scale_train_swa to the trained scale."""
    def edit(meta):
        meta["gemma2.rope.scaling.type"] = "linear"
        meta["gemma2.rope.scaling.factor"] = 2.0
    _, model, cfg, _ = _gemma_build("gemma2", YARN4, edit)
    s = rs.from_config(cfg)
    assert s.factor == 4.0 and s.for_sliding().factor == 2.0
    for i, r in _layer_ropes(model):
        _check_rope(r, s.for_sliding() if i % 2 == 0 else s, 10000.0)


def test_gemma3_full_layers_take_the_header_linear_factor():
    """Gemma 3 4B and up carry rope.scaling linear 8 for the full-attention
    layers. The synthesized config never passed it to the class, so those
    layers rotated unscaled. Sliding layers keep scale 1 (gemma3.cpp)."""
    def edit(meta):
        meta["gemma3.rope.scaling.type"] = "linear"
        meta["gemma3.rope.scaling.factor"] = 8.0
    _, model, cfg, n = _gemma_build("gemma3", meta_edit=edit)
    s = rs.from_config(cfg)
    assert n == 6 and s.factor == 8.0 and s.mscale == 1.0
    mx.random.seed(0)
    x = mx.random.normal((1, 1, 1, 16))
    for i, r in _layer_ropes(model):
        if i % 6 == 5:
            want = mx.fast.rope(x, 16, traditional=False, base=1000000.0,
                                scale=1 / 8, offset=800)
            np.testing.assert_allclose(np.asarray(r(x, offset=800)),
                                       np.asarray(want), atol=3e-5)
        else:
            _check_rope(r, s.for_sliding(), 10000.0)
            assert s.for_sliding().freq_scale == 1.0


def _gemma4_text(**kw):
    return {
        "model_type": "gemma4_text", "hidden_size": 64,
        "num_hidden_layers": 4, "intermediate_size": 128,
        "num_attention_heads": 4, "head_dim": 16, "global_head_dim": 16,
        "num_key_value_heads": 2, "num_global_key_value_heads": 2,
        "num_kv_shared_layers": 0, "vocab_size": 128,
        "vocab_size_per_layer_input": 128,
        "hidden_size_per_layer_input": 0, "sliding_window": 64,
        "sliding_window_pattern": 2, "attention_k_eq_v": False,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
        "max_position_embeddings": 1024,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25,
                               "rope_theta": 1000000.0,
                               "rope_type": "proportional"},
            "sliding_attention": {"partial_rotary_factor": 1.0,
                                  "rope_theta": 10000.0,
                                  "rope_type": "default"}},
        **kw}


def _gemma4_scaling(override=YARN4):
    meta = {"gemma4.rope.freq_base": 1000000.0, "gemma4.context_length": 1024}
    return rs.resolve(meta, "gemma4", override)


def _check_proportional(r, s, base, rotated):
    """llama.cpp's full gemma4 layers: rope_freqs 1 on the rotated pairs and
    1e30 on the rest, so those pairs keep angle 0 but take the magnitude."""
    assert r.dims == 16
    inv = rs.inv_freq(16, base, s).astype(np.float64)
    inv[rotated:] = 0.0
    mx.random.seed(0)
    x = mx.random.normal((1, 1, 1, 16))
    want = _ref_rotate(x, 700, 16, inv, s.mscale, False)
    np.testing.assert_allclose(np.asarray(r(x, offset=700)), want,
                               atol=3e-5, rtol=3e-5)


def test_gemma4_full_layers_rotate_their_proportional_pairs_scaled():
    cfg = _gemma4_text()
    s = _gemma4_scaling()
    rs.to_config(cfg, s)
    model, cfg = build_model(cfg)
    assert rope_swap.apply(model, cfg) == 4
    for i, r in _layer_ropes(model):
        assert isinstance(r, rope_swap.ScaledRoPE)
        if i % 2 == 0:
            _check_rope(r, s.for_sliding(), 10000.0)
        else:
            _check_proportional(r, s, 1000000.0, rotated=2)
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(getattr(out, "logits", out))


def test_proportional_rope_swap_keeps_the_class_rotation():
    """With a plain table the swapped proportional rope rotates as the
    class did."""
    from gmlx._vendor.mlx_vlm.models.rope_utils import ProportionalRoPE

    r = ProportionalRoPE(16, base=1000000.0,
                         scaling_config={"partial_rotary_factor": 0.25})
    s = _gemma4_scaling(RopeOverride(rope_freq_base=1000000.0))
    assert s.mscale == 1.0 and not s.scales_frequencies
    swapped = rope_swap._scaled(r, s, 1000000.0)
    mx.random.seed(2)
    x = mx.random.normal((1, 2, 3, 16))
    np.testing.assert_allclose(np.asarray(swapped(x, offset=5000)),
                               np.asarray(r(x, offset=5000)),
                               atol=2e-5, rtol=1e-5)


def test_gemma4_assistant_rotates_as_its_target():
    import importlib

    import gmlx.spec.mtp_load as mtp_load

    dcfg = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.gemma4_assistant.config")
    dmod = importlib.import_module(
        "gmlx._vendor.mlx_vlm.speculative.drafters.gemma4_assistant."
        "gemma4_assistant")
    tc = _gemma4_text(num_hidden_layers=2)
    drafter = dmod.Gemma4AssistantDraftModel(
        dcfg.Gemma4AssistantConfig.from_dict(
            {"backbone_hidden_size": 64, "text_config": tc}))
    assert mtp_load._rotate_as_target(drafter, tc, {"model_type": "x"}) == 0
    target = dict(tc)
    s = _gemma4_scaling()
    rs.to_config(target, s)
    assert mtp_load._rotate_as_target(drafter, tc, target) == 2
    for i, r in _layer_ropes(drafter):
        if i % 2 == 0:
            _check_rope(r, s.for_sliding(), 10000.0)
        else:
            _check_proportional(r, s, 1000000.0, rotated=2)
