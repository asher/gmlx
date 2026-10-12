"""qwen4exp rope scaling: every rope site (attention, the QSA indexer's
queries and block keys, the multi-axis cos/sin path, the MTP head) rotates
with the resolved YaRN table and llama.cpp's magnitude, and a load without
options keeps the plain rope."""

import mlx.core as mx
import numpy as np
import pytest

from gmlx.load import rope_scaling as rs
from gmlx.load.config_synth import synthesize_config
from gmlx.load.rope_scaling import RopeOverride
from gmlx.models.qwen4_exp import model as q4
from gmlx.models.qwen4_exp.mtp import Qwen4ExpMTPConfig, Qwen4ExpMTPDrafter
from test_config_synth import _QWEN4EXP_SHAPES, _qwen4exp_meta

CARD = RopeOverride(rope_scaling="yarn", rope_scale=4.0, yarn_orig_ctx=262144)
BASE = 10000000.0


def _meta():
    m = _qwen4exp_meta(True, False)
    m["qwen4exp.context_length"] = 262144
    return m


def _args(override=None):
    q4.ensure_registered()
    cfg = synthesize_config(_meta(), _QWEN4EXP_SHAPES, rope_override=override)
    return q4.ModelArgs.from_dict(cfg), cfg


def _ref_rope(x, positions, dims, inv, mscale):
    """Rotate-half rope of the first ``dims`` of ``x [..., L, D]`` at
    ``positions [L]``, cos and sin times ``mscale``. The angle is the
    float32 product both runtimes form; the rest is float64."""
    x = np.asarray(x.astype(mx.float32), dtype=np.float64)
    half = dims // 2
    ang = (np.asarray(positions, dtype=np.float32)[:, None]
           * inv.astype(np.float32)[None, :]).astype(np.float64)
    c, s = np.cos(ang) * mscale, np.sin(ang) * mscale
    x1, x2 = x[..., :half], x[..., half:dims]
    out = x.copy()
    out[..., :half] = x1 * c - x2 * s
    out[..., half:dims] = x2 * c + x1 * s
    return out


def _full_attention(args):
    return q4.Attention(args, 3)


def test_config_carries_the_scaling_and_context():
    args, cfg = _args(CARD)
    assert cfg["max_position_embeddings"] == 1048576
    s = rs.RopeScaling(**args.gmlx_rope_scaling)
    assert s.factor == 4.0 and s.original_context == 262144
    plain, pcfg = _args()
    assert plain.gmlx_rope_scaling is None
    assert pcfg["max_position_embeddings"] == 262144


def test_options_resolve_for_qwen4exp_only_after_wiring():
    assert "qwen4exp" in rs.WIRED_ARCHES


def test_attention_rope_matches_the_reference():
    args, _ = _args(CARD)
    attn = _full_attention(args)
    d = attn.rotary_dims
    s = rs.RopeScaling(**args.gmlx_rope_scaling)
    inv = rs.inv_freq(d, BASE, s).astype(np.float64)
    mx.random.seed(0)
    x = mx.random.normal((1, 2, 5, attn.head_dim))
    for offset in (0, 1000, 300000):
        got = np.asarray(attn._rope(x, offset))
        want = _ref_rope(x, np.arange(offset, offset + 5), d, inv, s.mscale)
        np.testing.assert_allclose(got, want, atol=2e-4, rtol=1e-4)
    # The magnitude is llama.cpp's 0.1 ln 4 + 1, and the dims past the
    # rotary width pass through untouched.
    assert s.mscale == pytest.approx(1 + 0.1 * np.log(4), rel=1e-6)
    np.testing.assert_array_equal(np.asarray(attn._rope(x, 7))[..., d:],
                                  np.asarray(x)[..., d:])


def test_indexer_block_rope_composes_with_the_block_scale():
    args, _ = _args(CARD)
    attn = _full_attention(args)
    idx = attn.indexer
    d, r = idx.rotary_dims, idx.ratio
    s = rs.RopeScaling(**args.gmlx_rope_scaling)
    inv = rs.inv_freq(d, BASE, s).astype(np.float64)
    mx.random.seed(1)
    x = mx.random.normal((1, 1, 3, idx.head_dim))
    start = 50
    got = np.asarray(idx._rope(x, start, scale=float(r)))
    want = _ref_rope(x, (start + np.arange(3)) * r, d, inv, s.mscale)
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=1e-4)


def test_multi_axis_path_agrees_with_the_text_path():
    """With the same position on all three axes, the cos/sin path (VLM
    loads, indexer block keys at cached positions) rotates as the text
    path does, table and magnitude included."""
    args, _ = _args(CARD)
    attn = _full_attention(args)
    d = attn.rotary_dims
    L, off = 4, 1234
    pos = mx.broadcast_to(mx.arange(off, off + L)[None, None, :], (3, 1, L))
    cos, sin = q4._mrope_cos_sin(pos, d, BASE, attn._mrope_selector,
                                 attn._rope_scaled)
    mx.random.seed(2)
    x = mx.random.normal((1, 2, L, attn.head_dim))
    a = q4._apply_rope_cos_sin(x, cos, sin, d)
    b = attn._rope(x, off)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-4)
    # The indexer's explicit-position block keys take the same table.
    idx = attn.indexer
    raw = mx.random.normal((1, 2 * idx.ratio, idx.head_dim))
    pos_all = mx.broadcast_to(mx.arange(0, 2 * idx.ratio)[None, None, :],
                              (3, 1, 2 * idx.ratio))
    np.testing.assert_allclose(
        np.asarray(idx.finish_blocks_at(raw, 0, pos_all, attn._mrope_selector)),
        np.asarray(idx.finish_blocks(raw, 0)), atol=2e-4)


def test_mtp_head_takes_the_target_scaling():
    args, _ = _args(CARD)
    drafter = Qwen4ExpMTPDrafter(Qwen4ExpMTPConfig(text=args, block_size=3))
    attns = [m for _, m in drafter.named_modules()
             if isinstance(m, q4.Attention)]
    assert attns and all(a._rope_scaled is not None for a in attns)
    assert attns[0]._rope_scaled.mscale == pytest.approx(1 + 0.1 * np.log(4))


def test_no_options_keep_the_plain_rope():
    args, _ = _args()
    attn = _full_attention(args)
    assert attn._rope_scaled is None and attn.indexer._rope_scaled is None
    mx.random.seed(3)
    x = mx.random.normal((1, 2, 5, attn.head_dim))
    plain = mx.fast.rope(x, attn.rotary_dims, traditional=False, base=BASE,
                         scale=1.0, offset=9)
    assert mx.array_equal(attn._rope(x, 9), plain)
