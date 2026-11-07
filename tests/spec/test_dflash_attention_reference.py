"""DFlash draft attention against a float64 port of SGLang's draft forward
with the Ripple vendor patch (inference/server/model.py).

The reference keeps every context row at its absolute position and masks
by SGLang's rules:

- a sliding layer keeps context key k for block row q while
  q - k <= window - 1 (``window_left``, dflash_utils.py);
- a full layer with a context window sees the last ``C`` committed rows,
  the same span for every block row (worker_v2.py, the compact draft
  window);
- a layer in ``no_rope_layers`` has an identity rotary (the vendor's
  ``_patch_draft_attention``).

gmlx stores rows in rings that compact and inserts a long seed only up to
each layer's span, so its positions are relative. RoPE depends only on
q - k, so the outputs must agree.
"""

import mlx.core as mx
import numpy as np
import pytest

from gmlx.spec.dflash_drafter import DFlashConfig, DFlashDrafter

DIM, HEADS, KV, HD = 32, 4, 2, 8
THETA = 10000.0
WINDOW, CONTEXT, BLOCK = 8, 12, 4
EPS = 1e-6


def _config(layer_types, rope_layers):
    return DFlashConfig(
        hidden_size=DIM, intermediate_size=64,
        num_hidden_layers=len(layer_types), num_attention_heads=HEADS,
        num_key_value_heads=KV, head_dim=HD, rms_norm_eps=EPS,
        vocab_size=64, max_position_embeddings=4096, rope_theta=THETA,
        block_size=BLOCK, mask_token_id=7, target_layer_ids=[0],
        num_target_layers=2, layer_types=layer_types, sliding_window=WINDOW,
        rope_layers=rope_layers, context_window=CONTEXT)


def _rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def _rope(x, pos):
    half = HD // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / HD)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def _reference(attn, ctx, block, *, sliding, roped):
    """SGLang's draft attention for one layer, float64, absolute positions."""
    w = {k: np.array(v, dtype=np.float64) for k, v in (
        ("q", attn.q_proj.weight), ("k", attn.k_proj.weight),
        ("v", attn.v_proj.weight), ("o", attn.o_proj.weight),
        ("qn", attn.q_norm.weight), ("kn", attn.k_norm.weight))}
    n, length = ctx.shape[0], block.shape[0]
    rows = np.concatenate([ctx, block])
    pos = np.arange(n + length, dtype=np.float64)
    q = _rms((block @ w["q"].T).reshape(length, HEADS, HD), w["qn"])
    k = _rms((rows @ w["k"].T).reshape(n + length, KV, HD), w["kn"])
    v = (rows @ w["v"].T).reshape(n + length, KV, HD)
    if roped:
        q = _rope(q.transpose(1, 0, 2), pos[n:]).transpose(1, 0, 2)
        k = _rope(k.transpose(1, 0, 2), pos).transpose(1, 0, 2)
    qpos = pos[n:][:, None]
    kpos = pos[None, :]
    visible = kpos >= n  # the block itself, non-causal
    if sliding:
        in_ctx = (kpos < n) & (qpos - kpos <= WINDOW - 1)
    else:
        in_ctx = (kpos < n) & (kpos >= n - CONTEXT)
    visible = visible | in_ctx
    out = np.zeros((length, HEADS, HD))
    group = HEADS // KV
    for h in range(HEADS):
        s = q[:, h] @ k[:, h // group].T * HD ** -0.5
        s = np.where(visible, s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[:, h] = p @ v[:, h // group]
    return out.reshape(length, -1) @ w["o"].T


@pytest.mark.parametrize("n_ctx", [10, 150])
@pytest.mark.parametrize("seed_rows", [3, 40])
def test_draft_attention_matches_the_reference(n_ctx, seed_rows):
    layer_types = ["sliding_attention", "full_attention", "full_attention"]
    rope_layers = [True, False, True]
    drafter = DFlashDrafter(_config(layer_types, rope_layers))
    mx.random.seed(0)
    for layer in drafter.layers:
        attn = layer.self_attn
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            lin.weight = mx.random.normal(lin.weight.shape) * 0.3
        attn.q_norm.weight = 1 + 0.1 * mx.random.normal((HD,))
        attn.k_norm.weight = 1 + 0.1 * mx.random.normal((HD,))
    caches = drafter.make_cache()
    rng = np.random.default_rng(1)
    ctx = rng.standard_normal((n_ctx, DIM))
    block = rng.standard_normal((BLOCK, DIM))
    # A long seed first, as a prefill does, then round-sized commits.
    first = min(seed_rows, n_ctx)
    chunks = [ctx[:first]] + [ctx[i:i + 3] for i in range(first, n_ctx, 3)]
    for layer, c in zip(drafter.layers, caches):
        for part in chunks:
            layer.self_attn.inject(
                mx.array(part[None], dtype=mx.float32), drafter.rope, c)
    for i, (layer, c) in enumerate(zip(drafter.layers, caches)):
        got = layer.self_attn.draft(
            mx.array(block[None], dtype=mx.float32), drafter.rope, c)
        mx.eval(got)
        want = _reference(layer.self_attn, ctx, block,
                          sliding=layer_types[i] == "sliding_attention",
                          roped=rope_layers[i])
        err = np.abs(np.array(got[0], dtype=np.float64) - want).max()
        assert err < 1e-4, f"layer {i}: max error {err:.2e}"
