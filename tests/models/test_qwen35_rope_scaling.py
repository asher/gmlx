"""Rope scaling on the Qwen 3.5 family: the multi-axis rope takes the
resolved table in its inv_freq buffer and the magnitude as its attention
scaling, on the fused Metal apply as on the cos/sin path, in the text
class, the MTP target and the VLM tower."""

import math
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx._vendor.mlx_vlm.models.rope_utils import MRoPERotaryEmbedding
from gmlx.load import rope_scaling as rs
from gmlx.load import rope_swap
from gmlx.load.config_synth import synthesize_config
from gmlx.load.loader import build_model
from gmlx.load.rope_scaling import RopeOverride
from gmlx.models.qwen35.language import Qwen3_5RotaryEmbedding
import gmlx.models.qwen35.rope as rp
from test_config_synth import _base_meta

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the fused MRoPE apply is Metal-only")

YARN4 = RopeOverride(rope_scaling="yarn", rope_scale=4.0)


def _meta(arch="qwen35"):
    m = _base_meta(arch)
    m[f"{arch}.block_count"] = 4
    m[f"{arch}.full_attention_interval"] = 4
    m[f"{arch}.ssm.inner_size"] = 64
    m[f"{arch}.ssm.time_step_rank"] = 4
    m[f"{arch}.ssm.group_count"] = 2
    m[f"{arch}.ssm.conv_kernel"] = 4
    m[f"{arch}.rope.freq_base"] = 10000000.0
    if arch == "qwen35moe":
        m[f"{arch}.expert_count"] = 4
        m[f"{arch}.expert_used_count"] = 2
        m[f"{arch}.expert_feed_forward_length"] = 32
        m[f"{arch}.expert_shared_feed_forward_length"] = 32
    return m


_SHAPES = {"blk.0.ssm_conv1d.weight": [4, 128]}


def _ref(x, pos, dims, inv, mscale):
    """float64 rotate-half rope of the first ``dims`` of ``x [B, H, L, D]``
    at positions ``pos [L]`` (the same on every axis)."""
    x = np.asarray(x.astype(mx.float32), dtype=np.float64)
    ang = (np.asarray(pos, np.float32)[:, None]
           * inv.astype(np.float32)[None, :]).astype(np.float64)
    c, s = np.cos(ang) * mscale, np.sin(ang) * mscale
    h = dims // 2
    out = x.copy()
    a, b = x[..., :h], x[..., h:dims]
    out[..., :h] = a * c - b * s
    out[..., h:dims] = b * c + a * s
    return out


def _rotary(scale=1.3):
    r = Qwen3_5RotaryEmbedding(16, base=10000000.0, mrope_section=[3, 3, 2])
    r.attention_scaling = scale
    return r


def _qk(L=5):
    mx.random.seed(0)
    return (mx.random.normal((1, 4, L, 64)).astype(mx.bfloat16),
            mx.random.normal((1, 2, L, 64)).astype(mx.bfloat16))


def _pos(off, L=5):
    p = mx.arange(off, off + L)[None]
    return mx.broadcast_to(p[None], (3, 1, L))


@_NEEDS_GPU
@pytest.mark.parametrize("owned", [False, True])
def test_fused_apply_takes_the_attention_scaling(owned):
    """The kernel path dropped attention_scaling while the cos/sin path
    applied it, so the two routes disagreed under any scaling."""
    r = _rotary()
    q, k = _qk()
    pos = _pos(40)
    assert rp.fused_rope_active(r)
    if owned:
        fq, fk = rp.rope_apply_rotary(r, q, k, pos)
    else:
        fq, fk = r.apply_rotary(q, k, pos)
    cos, sin = r(k, pos)
    from gmlx._vendor.mlx_vlm.models.rope_utils import (
        apply_multimodal_rotary_pos_emb)
    cq, ck = apply_multimodal_rotary_pos_emb(
        q, k, cos, sin, mrope_section=r.mrope_section, style=r.style)
    for a, b in ((fq, cq), (fk, ck)):
        np.testing.assert_allclose(np.asarray(a.astype(mx.float32)),
                                   np.asarray(b.astype(mx.float32)),
                                   atol=3e-2, rtol=2e-2)
    want = _ref(q, np.arange(40, 45), 16,
                rs.inv_freq(16, 10000000.0), 1.3)
    np.testing.assert_allclose(np.asarray(fq.astype(mx.float32)), want,
                               atol=3e-2, rtol=2e-2)


def _config(arch="qwen35", override=YARN4):
    return synthesize_config(_meta(arch), _SHAPES, rope_override=override)


def _check_mrope(r, scaling, base=10000000.0):
    inv = rs.inv_freq(r.dim, base, scaling)
    np.testing.assert_array_equal(np.asarray(r.inv_freq), inv)
    assert r.attention_scaling == pytest.approx(scaling.mscale)
    q, k = _qk(3)
    pos = _pos(1000, 3)
    got, _ = rp.rope_apply_rotary(r, q, k, pos)
    want = _ref(q, np.arange(1000, 1003), r.dim, inv, scaling.mscale)
    np.testing.assert_allclose(np.asarray(got.astype(mx.float32)), want,
                               atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe"])
def test_text_class_takes_the_options(arch):
    cfg = _config(arch)
    model, cfg = build_model(cfg)
    assert rope_swap.apply(model, cfg) > 0
    s = rs.from_config(cfg)
    assert s.factor == 4.0 and s.mscale == pytest.approx(1 + 0.1 * math.log(4))
    ropes = [m.rope for _, m in model.named_modules()
             if isinstance(getattr(m, "rope", None), nn.Module)]
    assert ropes and all(isinstance(r, rope_swap.ScaledRoPE) for r in ropes)
    inv = rs.inv_freq(ropes[0].dims, 10000000.0, s)
    np.testing.assert_allclose(1.0 / np.asarray(ropes[0]._freqs), inv,
                               rtol=1e-6)


def _mropes(model):
    return [m.rotary_emb for _, m in model.named_modules()
            if isinstance(getattr(m, "rotary_emb", None),
                          MRoPERotaryEmbedding)]


@_NEEDS_GPU
def test_mtp_target_takes_the_options():
    cfg = _config()
    model, cfg = build_model(cfg, mtp=True)
    assert rope_swap.apply(model, cfg) > 0
    s = rs.from_config(cfg)
    ropes = _mropes(model)
    assert ropes
    for r in ropes:
        _check_mrope(r, s)
    # A second pass changes nothing.
    assert rope_swap.apply(model, cfg) == 0


def test_plain_rope_keeps_the_class_table():
    cfg = _config(override=None)
    model, cfg = build_model(cfg, mtp=True)
    assert rope_swap.apply(model, cfg) == 0
    for r in _mropes(model):
        assert r.attention_scaling == 1.0
        np.testing.assert_array_equal(
            np.asarray(r.inv_freq),
            np.asarray(rp.compute_inv_freq(r.dim, 10000000.0)))


@_NEEDS_GPU
def test_vlm_tower_takes_the_options():
    """The VLM text tower is the owned language model with the same
    rotary class; the swap reaches it through the VLM container's
    language_model with the nested text config."""
    from gmlx._vendor.mlx_vlm.models.qwen3_5.config import (
        TextConfig as Q35TextConfig)
    from gmlx.models.qwen35.language import LanguageModel

    text = _config()
    tc = Q35TextConfig.from_dict({**text, "model_type": "qwen3_5"})
    lm = LanguageModel(tc)
    container = {"model_type": "qwen3_5", "text_config": text}
    assert rope_swap.apply(lm, container) > 0
    s = rs.from_config(text)
    for r in _mropes(lm):
        _check_mrope(r, s)


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe", "qwen3vl",
                                  "qwen3vlmoe"])
def test_family_is_wired(arch):
    assert arch in rs.SWAP_ARCHES and arch in rs.WIRED_ARCHES
