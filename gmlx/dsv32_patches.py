"""Runtime correctness patches for DeepSeek-V3.2 / glm-dsa.

Mask-path decode attention, indexer RoPE convention, fp32 indexer and MoE
router selection, and the dense-by-default DSA indexer. Each installer is
install-once and kill-switchable via its GMLX_DSV32_* env flag.
"""

from __future__ import annotations

import mlx.core as mx

import mlx_kquant as kq

from . import loadlog
from .envflags import env_bool, env_int
from .patching import ClassPatch


# DeepSeek-V3.2 / glm-dsa: mask-path decode attention (correctness patch)
#
# mlx-lm's DeepseekV32Attention applies the DSA "lightning indexer" top-k by
# gathering the selected keys on the L==1 decode step, but by masking full keys
# on the L>1 prefill step. The gather is mathematically equivalent to the mask
# (verified ~bit-for-bit at small scale) yet corrupts the decode attention
# *distribution* once context exceeds index_topk (~2048): the argmax stays right
# (greedy looks fine) while the tail is wrong, so temperature/top-p sampling
# degenerates to token-garbage at depth. llama.cpp and mlx-lm's own prefill both
# take the mask path and stay coherent.
#
# This routes the L==1 decode step through that same mask path. The body is the
# exact upstream forward with only the `if topk_indices is not None` block changed
# to always mask (never gather). Trade-off: decode attention becomes O(context)
# rather than O(index_topk) - negligible for over-RAM streaming (decode is
# disk-bound) but it forfeits DSA's sparse-decode speedup for long-context in-RAM
# use. Kill with GMLX_DSV32_MASK_DECODE=0.
_MASK_DECODE_PATCH = ClassPatch()


def _dsv32_mask_decode_call(self, x, mask=None, cache=None):
    """Patched DeepseekV32Attention.__call__ applying the indexer top-k as a mask
    over full keys for every L (incl. L==1 decode) instead of gathering at L==1.
    No-op fallback to the stock forward for instances without the per-instance
    ``_dsv32_mask_decode`` flag, so unrelated loads in the same process are
    untouched."""
    if not getattr(self, "_dsv32_mask_decode", False):
        return _MASK_DECODE_PATCH.stock(self, x, mask, cache)

    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, D = x.shape

    qr = self.q_a_layernorm(self.q_a_proj(x))
    q = self.q_b_proj(qr)

    q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
    compressed_kv = self.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = self.kv_a_layernorm(compressed_kv)

    offset = cache[0].offset if cache is not None else 0
    q_pe = self.rope(q_pe, offset)
    k_pe = self.rope(k_pe, offset)

    kv_latent = mx.expand_dims(kv_latent, axis=1)

    if cache is not None:
        kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
    else:
        cache = [None] * 2

    topk_indices = self.indexer(x, qr, mask, cache=cache[1])
    if topk_indices is not None:
        # Fix: mask over full keys for every L (incl. L==1 decode), never gather.
        shape = list(topk_indices.shape)
        shape[-1] = kv_latent.shape[2]
        sparse_mask = mx.zeros(shape, dtype=mx.bool_)
        sparse_mask = mx.put_along_axis(
            sparse_mask, topk_indices, mx.array(True), axis=-1
        )
        if mask is not None:
            sparse_mask = sparse_mask & mask
        mask = sparse_mask
    # Ensure the indexer cache is evaluated even if topk_indices is unused, to
    # keep the graph from getting too large.
    if cache is not None and cache[0] is not None:
        cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    if mask is not None:
        pe_scores = mx.where(
            mask,
            pe_scores,
            mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
        )

    if L == 1:
        q_nope = self.embed_q(q_nope)
        k = v = kv_latent
    else:
        k = self.embed_q(kv_latent, transpose=False)
        v = self.unembed_out(kv_latent)

    output = scaled_dot_product_attention(
        q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
    )
    if L == 1:
        output = self.unembed_out(output)

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def _patch_dsv32_mask_decode(model) -> None:
    """Route DeepseekV32Attention's L==1 decode through the mask path on every
    attention module in ``model``. Installs the class-level dispatch once (a no-op
    for any instance without the per-instance ``_dsv32_mask_decode`` flag, so
    unrelated loads in the same process are untouched) and flags this model's
    instances. Kill with GMLX_DSV32_MASK_DECODE=0."""
    if not env_bool("GMLX_DSV32_MASK_DECODE", True):
        return
    from mlx_lm.models.deepseek_v32 import DeepseekV32Attention

    _MASK_DECODE_PATCH.install(
        DeepseekV32Attention, "__call__", _dsv32_mask_decode_call)
    n = 0
    for m in model.modules():
        if isinstance(m, DeepseekV32Attention):
            m._dsv32_mask_decode = True
            n += 1
    loadlog.verbose_print(f"[patch] dsv32: mask-path decode attention enabled on {n} layers")


# DeepSeek-V3.2 / glm-dsa: indexer RoPE convention + k_norm eps (correctness patch)
#
# The DSA "lightning indexer" ropes its query/key with the same convention as the
# main DeepSeek attention: interleaved / GPT-J (mlx traditional=True). That is stock
# mlx-lm for the whole DeepSeek family (v2/v3/v3.2, main and indexer). The confusion
# was reading HF's literal apply_rotary_pos_emb, which calls rotate_half (looks
# half-split / NeoX) - but it first deinterleaves (view(d//2,2).transpose.reshape).
# Deinterleave-then-rotate_half yields the interleaved-rope values in a permuted
# order, and the attention score q_pe*k_pe is invariant to permuting both operands,
# so the HF score == interleaved (traditional=True) score. An earlier build of this
# patch instead rebuilt the indexer rope as half-split (traditional=False) and applied
# it directly (no deinterleave); that makes q_pe*k_pe diverge from HF by an amount
# that GROWS with relative key distance (dg-dsv32-rope-convention.py: delta up to ~11 at
# dist 128 vs a ~7e-5 floor for traditional=True; exactly 0 at dist 0). The indexer is
# dormant at ctx<=index_topk (returns None -> dense), so the wrong scores only bite
# past ~2048: coherent at first, then token-garbage at depth (the observed signature).
# Fix: keep the indexer rope interleaved (traditional=True), matching the main
# attention + stock. We also set k_norm eps to 1e-6 (HF's value; mlx-lm inherits
# nn.LayerNorm's 1e-5). Kill with GMLX_DSV32_INDEXER_ROPE=0.
def _patch_dsv32_indexer_rope(model) -> None:
    """Pin the DSA indexer's geometry on every Indexer in ``model`` to match HF +
    stock mlx-lm: interleaved (traditional=True) RoPE - the same convention as the
    main DeepSeek attention - and k_norm eps=1e-6 (HF's value; mlx-lm inherits
    nn.LayerNorm's 1e-5 default). Idempotent; kill with
    GMLX_DSV32_INDEXER_ROPE=0."""
    if not env_bool("GMLX_DSV32_INDEXER_ROPE", True):
        return
    from mlx_lm.models.deepseek_v32 import DeepseekV32Attention, Indexer
    from mlx_lm.models.rope_utils import initialize_rope

    cfg = None
    for m in model.modules():
        if isinstance(m, DeepseekV32Attention):
            cfg = m.config
            break
    if cfg is None:
        return
    n = 0
    for m in model.modules():
        if isinstance(m, Indexer) and not getattr(
            m, "_dsv32_indexer_rope_fixed", False
        ):
            m.rope = initialize_rope(
                dims=cfg.qk_rope_head_dim,
                base=cfg.rope_theta,
                traditional=True,  # interleaved/GPT-J: HF deinterleave+rotate_half ==
                                   # interleaved for q*k; matches main attn + stock
                max_position_embeddings=cfg.max_position_embeddings,
                scaling_config=cfg.rope_scaling,
            )
            # HF builds the indexer k_norm with eps=1e-6; mlx-lm uses nn.LayerNorm's
            # 1e-5 default. Negligible numerically but a real divergence - match HF.
            m.k_norm.eps = 1e-6
            m._dsv32_indexer_rope_fixed = True
            n += 1
    if n:
        loadlog.verbose_print(
            f"[patch] dsv32: indexer interleaved rope + k_norm eps=1e-6 on {n} layers"
        )


# DeepSeek-V3.2 / glm-dsa: fp32 indexer selection (correctness patch)
#
# The DSA indexer's top-k *selection* is an argpartition over per-head-weighted
# relu(q*k) sums; once context exceeds index_topk the keys near the rank-index_topk
# boundary are packed tightly, so a sub-percent perturbation of the scores flips
# which keys make the cut. Dropping a key the model needs is the depth degradation:
# forcing the indexer dormant (dense attention) on the slipping prompt produces
# clean output, so the residual is the *selection*, not the rest of the pipeline.
#
# llama.cpp (the flawless reference, incl. Unsloth's IQ1 one-shots) accumulates the
# q8 indexer matmuls in fp32 (ggml dequantizes the q8 weight and accumulates in
# fp32). mlx-lm runs them in the model dtype (bf16); worse, our KQuantLinear forward
# (kq.quantized_matmul) *always returns bf16* regardless of activation dtype, so the
# projections q/k are bf16-rounded before the score even forms - measured ~0.7% of
# the score magnitude, enough to flip borderline picks. Just upcasting the score
# matmul (q.astype(f32) @ k.astype(f32)) is therefore insufficient: q/k are already
# bf16. This patch dequantizes the indexer projection weights to fp32 and runs the
# whole selection path (wq_b / wk / k_norm / rope / score / weights_proj /
# argpartition) in fp32, matching llama.cpp. The dequant is per-call and transient
# (no resident fp32 weights, so the over-RAM page cache is undisturbed); the indexer
# is a tiny fraction of FLOPs so the cost is negligible. Pairs with the NeoX-rope/eps
# fix above. Kill with GMLX_DSV32_INDEXER_FP32=0.
_INDEXER_FP32_PATCH = ClassPatch()


def _dsv32_dense_f32(linear, x):
    """fp32 projection ``x @ W^T (+bias)`` for an indexer linear, dequantizing a
    KQuantLinear weight to fp32 (kq.quantized_matmul forces bf16 output, which
    rounds the top-k scores) or upcasting a float nn.Linear. ``x`` must be fp32."""
    from .modules import KQuantLinear

    if isinstance(linear, KQuantLinear):
        w = kq.dequantize(linear["weight"], linear["scales"], linear.kquant_type)
    else:
        w = linear["weight"]
    y = x @ w.astype(mx.float32).T
    if "bias" in linear:
        y = y + linear["bias"].astype(mx.float32)
    return y


def _dsv32_ln_f32(k, ln):
    """fp32 affine LayerNorm matching the indexer k_norm (biased variance + eps)."""
    kf = k.astype(mx.float32)
    mu = kf.mean(-1, keepdims=True)
    var = mx.mean(mx.square(kf - mu), axis=-1, keepdims=True)
    out = (kf - mu) * mx.rsqrt(var + ln.eps)
    if "weight" in ln:
        out = out * ln.weight.astype(mx.float32)
    if "bias" in ln:
        out = out + ln.bias.astype(mx.float32)
    return out


def _dsv32_indexer_fp32_call(self, x, qr, mask, cache=None):
    """Patched Indexer.__call__ computing the whole top-k selection path in fp32
    (matching llama.cpp). No-op fallback to the stock forward for instances without
    the per-instance ``_dsv32_indexer_fp32`` flag."""
    if not getattr(self, "_dsv32_indexer_fp32", False):
        return _INDEXER_FP32_PATCH.stock(self, x, qr, mask, cache)

    b, s, _ = x.shape
    xf = x.astype(mx.float32)
    qrf = qr.astype(mx.float32)
    # Fix: fp32 projections (dequant q8 -> fp32 matmul), matching llama.cpp's fp32
    # indexer accumulation. KQuantLinear's bf16-only output otherwise rounds q/k.
    q = _dsv32_dense_f32(self.wq_b, qrf)
    q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
    k = _dsv32_dense_f32(self.wk, xf)
    k = _dsv32_ln_f32(k, self.k_norm)
    k = mx.reshape(k, (b, 1, s, self.head_dim))

    offset = cache.offset if cache is not None else 0
    q = self.rope(q, offset=offset)
    k = self.rope(k, offset=offset)

    if cache is not None:
        k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
    if k.shape[2] <= self.index_topk:
        return None
    scores = q @ k.swapaxes(-1, -2)  # both fp32
    scores = mx.maximum(scores, 0)
    weights = _dsv32_dense_f32(self.weights_proj, xf) * (
        self.n_heads**-0.5 * self.softmax_scale
    )
    weights = weights.swapaxes(-1, -2)[..., None]
    scores = scores * weights
    scores = scores.sum(axis=1, keepdims=True)
    if mask is not None:
        scores = mx.where(mask, scores, -float("inf"))
    # Force-keep the attention-sink (first `sink` keys) + a recent local window
    # (last `local` keys, query-relative) by giving them +inf score, so the top-k
    # always retains them while the budget stays exactly index_topk. The DSA indexer
    # otherwise scores the BOS sink / most-recent keys very negative in ~17/78 layers
    # and drops them - yet the main attention parks 0.5-0.99 of its weight on exactly
    # those keys (StreamingLLM); dropping them is the residual decode degradation that
    # survives the rope/fp32/mask fixes. Disable with GMLX_DSV32_SINK/_LOCAL=0.
    sink = getattr(self, "_dsv32_sink", 0)
    local = getattr(self, "_dsv32_local", 0)
    if sink or local:
        S = scores.shape[-1]
        L = scores.shape[-2]
        idx = mx.arange(S)
        qpos = offset + mx.arange(L)
        force = (idx[None, :] < sink) | (
            (idx[None, :] > qpos[:, None] - local) & (idx[None, :] <= qpos[:, None])
        )
        force = force.reshape(1, 1, L, S)
        if mask is not None:
            force = force & mask
        scores = mx.where(force, mx.array(float("inf"), scores.dtype), scores)
    return mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
        ..., -self.index_topk :
    ]


def _patch_dsv32_indexer_fp32(model) -> None:
    """Compute the DSA indexer's top-k selection in fp32 (matching HF) on every
    Indexer in ``model``. Installs the class-level dispatch once (a no-op for any
    instance without the per-instance ``_dsv32_indexer_fp32`` flag) and flags this
    model's instances. Kill with GMLX_DSV32_INDEXER_FP32=0."""
    if not env_bool("GMLX_DSV32_INDEXER_FP32", True):
        return
    from mlx_lm.models.deepseek_v32 import Indexer

    _INDEXER_FP32_PATCH.install(Indexer, "__call__", _dsv32_indexer_fp32_call)
    # Force-keep budget for the sparse path: retain the attention-sink + a recent
    # local window the indexer otherwise drops (see _dsv32_indexer_fp32_call). Only
    # consulted in sparse mode (dense default returns None before the top-k).
    sink = env_int("GMLX_DSV32_SINK", 4)
    local = env_int("GMLX_DSV32_LOCAL", 128)
    n = 0
    for m in model.modules():
        if isinstance(m, Indexer):
            m._dsv32_indexer_fp32 = True
            m._dsv32_sink = sink
            m._dsv32_local = local
            n += 1
    if n:
        msg = f"[patch] dsv32: fp32 indexer selection on {n} layers"
        if env_bool("GMLX_DSV32_SPARSE", False) and (sink or local):
            msg += f" (force-keep sink={sink}+local={local})"
        loadlog.verbose_print(msg)


# DeepSeek-V3.2 / glm-dsa: fp32 MoE router selection (correctness patch)
#
# HF computes the MoE router entirely in fp32: router_logits =
# F.linear(hidden.float(), weight.float()) (line 495), with the correction bias
# kept in fp32 (_keep_in_fp32_modules_strict). mlx-lm computes x @ weight.T in the
# model dtype (bf16) and only upcasts inside group_expert_select's sigmoid - so the
# 256 router logits are already bf16-rounded before the sigmoid / group-top-k /
# top-k selection. With 256 experts and grouped top-k, bf16 rounding flips
# borderline expert picks; a wrong expert is a wrong FFN for that token, which
# shows up as occasional local token slips at depth. This upcasts the router matmul
# and the correction bias to fp32 to match HF. Kill with GMLX_DSV32_GATE_FP32=0.
_GATE_FP32_PATCH = ClassPatch()


def _dsv32_moe_gate_fp32_call(self, x):
    """Patched MoEGate.__call__ computing the router matmul + correction bias in
    fp32 (matching HF). No-op fallback to the stock forward for instances without
    the per-instance ``_dsv32_gate_fp32`` flag."""
    if not getattr(self, "_dsv32_gate_fp32", False):
        return _GATE_FP32_PATCH.stock(self, x)

    from mlx_lm.models.deepseek_v32 import group_expert_select

    return group_expert_select(
        x.astype(mx.float32) @ self.weight.T.astype(mx.float32),
        self.e_score_correction_bias.astype(mx.float32),
        self.top_k,
        self.n_group,
        self.topk_group,
        self.routed_scaling_factor,
        self.norm_topk_prob,
    )


def _patch_dsv32_moe_gate_fp32(model) -> None:
    """Compute the DSA MoE router selection in fp32 (matching HF) on every MoEGate
    in ``model``. Installs the class-level dispatch once (a no-op for any instance
    without the per-instance ``_dsv32_gate_fp32`` flag) and flags this model's
    instances. Kill with GMLX_DSV32_GATE_FP32=0."""
    if not env_bool("GMLX_DSV32_GATE_FP32", True):
        return
    from mlx_lm.models.deepseek_v32 import MoEGate

    _GATE_FP32_PATCH.install(MoEGate, "__call__", _dsv32_moe_gate_fp32_call)
    n = 0
    for m in model.modules():
        if isinstance(m, MoEGate):
            m._dsv32_gate_fp32 = True
            n += 1
    if n:
        loadlog.verbose_print(f"[patch] dsv32: fp32 MoE router selection on {n} layers")


# DeepSeek-V3.2 / glm-dsa: hand routing scores to the expert call (mix seam)
#
# The stock DeepseekV32MoE mixes python-side: y = switch_mlp(x, inds) then a
# scores-weighted sum. A swapped switch_mlp that accepts scores (the fused
# kquant mix seam, or the streaming offload wrapper's scores sink feeding
# --moe-miss-shed) never sees them through that call, so the miss-shed lever
# is inert on this family. This forwards the scores when the module
# advertises either seam; an unmixed (ndim + 1) return keeps the stock
# python-side sum, so behavior without the levers is unchanged. Kill with
# GMLX_DSV32_MOE_MIX=0.
_MOE_SCORES_PATCH = ClassPatch()


def _dsv32_moe_scores_call(self, x):
    """Patched DeepseekV32MoE.__call__ passing gate scores to a scores-taking
    switch_mlp. No-op fallback to the stock forward for instances without the
    per-instance ``_dsv32_moe_scores`` flag."""
    if not getattr(self, "_dsv32_moe_scores", False):
        return _MOE_SCORES_PATCH.stock(self, x)

    from mlx_lm.models.deepseek_v32 import sum_gradients

    if self.sharding_group is not None:
        x = sum_gradients(self.sharding_group)(x)

    inds, scores = self.gate(x)
    sw = self.switch_mlp
    if (getattr(sw, "_kq_mix_scores", False)
            or getattr(sw, "_kq_scores_sink", False)):
        y = sw(x, inds, scores.astype(x.dtype))
    else:
        y = sw(x, inds)
    if y.ndim == scores.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if self.config.n_shared_experts is not None:
        y = y + self.shared_experts(x)

    if self.sharding_group is not None:
        y = mx.distributed.all_sum(y, group=self.sharding_group)

    return y


def _patch_dsv32_moe_scores(model) -> None:
    """Hand MoE routing scores to scores-taking expert modules on every
    DeepseekV32MoE in ``model``. Installs the class-level dispatch once (a
    no-op for any instance without the per-instance ``_dsv32_moe_scores``
    flag) and flags this model's instances. Kill with GMLX_DSV32_MOE_MIX=0."""
    if not env_bool("GMLX_DSV32_MOE_MIX", True):
        return
    from mlx_lm.models.deepseek_v32 import DeepseekV32MoE

    _MOE_SCORES_PATCH.install(
        DeepseekV32MoE, "__call__", _dsv32_moe_scores_call)
    n = 0
    for m in model.modules():
        if isinstance(m, DeepseekV32MoE):
            m._dsv32_moe_scores = True
            n += 1
    if n:
        loadlog.verbose_print(
            f"[patch] dsv32: scores-taking MoE expert call on {n} layers")


# DeepSeek-V3.2 / glm-dsa: default the DSA indexer to dormant (dense attention)
#
# The DSA "lightning indexer" picks a sparse top-k of keys for the main attention.
# From q8 indexer weights its selection drops keys the main attention needs - a
# drop-count-dependent token slip at depth (clean shallow, degrades as the dropped
# set grows past ~index_topk). Dense attention over all keys is exact, and for a
# disk-bound MoE the extra attention is free (measured *faster* than the sparse
# gather/mask). So default the indexer dormant. Opt into the model's native sparse
# DSA - its designed behavior, lossy at depth on this quant - with
# GMLX_DSV32_SPARSE=1.
def _patch_dsv32_dense_default(model) -> None:
    """Default the glm-dsa / deepseek_v32 DSA indexer to dormant on every Indexer
    (index_topk set huge -> always returns None -> dense attention). Opt back into
    the native sparse DSA path with GMLX_DSV32_SPARSE=1."""
    if env_bool("GMLX_DSV32_SPARSE", False):
        loadlog.warn(
            "[experimental] dsv32 DSA: native SPARSE attention enabled. The fp32 "
            "indexer + sink/local force-keep make decode coherent, but selection is "
            "NOT bit-identical to llama.cpp (an unresolved indexer score divergence "
            "remains). Dense (default, exact) - unset GMLX_DSV32_SPARSE to use it."
        )
        return
    from mlx_lm.models.deepseek_v32 import Indexer

    n = 0
    for m in model.modules():
        if isinstance(m, Indexer):
            m.index_topk = 1 << 30
            n += 1
    if n:
        loadlog.verbose_print(
            f"[patch] dsv32: dense attention default (DSA indexer dormant) on {n} "
            f"layers - exact + faster over-RAM. GMLX_DSV32_SPARSE=1 -> native "
            f"sparse DSA (experimental)."
        )
