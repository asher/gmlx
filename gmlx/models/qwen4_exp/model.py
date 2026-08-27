# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2024 Apple Inc. (mlx-lm qwen3_next/qwen3_5 skeleton, MIT)
"""Vendored mlx-lm-style model for Qwen3.8-Flash-Next (GGUF arch ``qwen4exp``).

Neither pinned mlx-lm nor pinned mlx-vlm ships a qwen4_exp class; this module
is the runtime for llama.cpp's ``LLM_ARCH_QWEN4EXP`` conversions. It is the
qwen3.5 hybrid (gated DeltaNet on three of every four layers, gated full
attention on the fourth, a 512-expert top-10 MoE with a sigmoid-gated shared
expert on every layer) plus three mechanisms of its own:

  1. Hyper-connections. The residual is four parallel streams ``[T, 4, D]``
     initialised from the token embedding. Each sub-layer reads one mixed
     ``[T, D]`` row (grouped RMSNorm per stream, low-rank silu down / sigmoid
     up gate, mean over the streams) and writes back ``out * inject``, where
     ``inject = 2 * sigmoid(W xn / 4)`` is one scalar per stream. A final
     mixer without inject replaces output_norm.
  2. QSA sparse attention. A 4-head indexer scores mean-pooled blocks of
     ``compress_ratio`` keys (k_norm + rope at the block start position,
     ``sum_h relu(q_h . k_b)``), keeps the ``budget / ratio`` best complete
     blocks plus the incomplete tail, and attention is masked to that set.
     Dense while at most ``budget / ratio`` complete blocks exist.
  3. PLE n-gram hash embeddings on one layer: 2-gram and 3-gram hashes of the
     token history (EOS resets the context) index a huge quantized table; the
     rows gate into all four streams plus a dilated depthwise conv branch.

Wire facts the forward relies on: every norm weight is already ``(1 + w)``
baked (plain RMSNorm); the GDN output gate is sigmoid; GDN V heads are tiled
(V head ``hv`` reads K head ``hv % Hk``), so q/k are tiled explicitly before
the scan; attention ``q_proj`` carries ``[q | gate]`` per head; rotary is 64
of 256 dims with interleaved mrope sections, which for text-only positions is
plain NEOX rope.
"""

import importlib
import math
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.switch_layers import SwitchGLU


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.qwen4_exp`` resolve, preferring upstream,
    and expose ``QSAKVCache`` on the cache modules (prompt-cache save/load
    resolves cache classes by name there; mlx-vlm >= 0.6.4 vendors its own
    models/cache.py, so register on both when it is loaded)."""
    import mlx_lm.models.cache as _mlx_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (_mlx_cache, vlm_cache):
        if mod is not None and not hasattr(mod, "QSAKVCache"):
            mod.QSAKVCache = QSAKVCache
    if "mlx_lm.models.qwen4_exp" in sys.modules:
        return
    try:
        importlib.import_module("mlx_lm.models.qwen4_exp")  # upstream wins
    except ImportError:
        sys.modules["mlx_lm.models.qwen4_exp"] = sys.modules[__name__]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    max_position_embeddings: int = 262144
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    hc_count: int = 4
    hc_lowrank: int = 320
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    compress_ratios: Optional[List[int]] = None
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_ngram_size: int = 3
    ple_heads_per_ngram: int = 8
    ple_conv_kernel: int = 4
    ple_eos_token_id: int = 0
    ple_image_token_id: Optional[int] = None
    ple_embed_dim: int = 160
    ple_table_rows: int = 0
    ple_layer_multipliers: List[int] = field(default_factory=list)
    ple_head_offsets: List[int] = field(default_factory=list)
    ple_head_vocab_sizes: List[int] = field(default_factory=list)
    tie_word_embeddings: bool = False
    kv_head_layout: str = "tiled"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.compress_ratios is None:
            self.compress_ratios = [0] * self.num_hidden_layers


# Norms


def _grouped_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm over the last axis of ``[..., hc, D]`` with a ``[hc * D]``
    gamma: one statistic per stream, one gain per element."""
    hc, d = x.shape[-2], x.shape[-1]
    return mx.fast.rms_norm(x, None, eps) * weight.reshape(hc, d)


class GroupedRMSNorm(nn.Module):
    """RMSNorm applied per residual stream, gamma over all streams."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _grouped_rms_norm(x, self.weight, self.eps)


class RMSNormGatedSigmoid(nn.Module):
    """``rms_norm(x) * sigmoid(gate)``: the GDN output norm (qwen3.5 uses
    silu here; this is the one numerical difference in the GDN)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps) * mx.sigmoid(gate)


# Hyper-connections


class HyperConnection(nn.Module):
    """Low-rank sigmoid-gated stream mixer (+ optional per-stream inject)."""

    def __init__(self, hidden: int, hc: int, lowrank: int, eps: float,
                 inject: bool = True):
        super().__init__()
        self.hc = hc
        self.hidden = hidden
        self.norm = GroupedRMSNorm(hc * hidden, eps)
        self.down = nn.Linear(hc * hidden, lowrank, bias=False)
        self.up = nn.Linear(lowrank, hc * hidden, bias=False)
        if inject:
            self.inject = nn.Linear(hc * hidden, hc, bias=False)

    def __call__(self, h: mx.array):
        B, T, hc, D = h.shape
        if B * T <= 8 and self._hclr_ok(h.dtype):
            norm, front, epi = _kq_hc()
            xn = norm(h, self.norm.weight, self.norm.eps)
            lo, inj = front(xn, self.down.weight, self.inject.weight,
                            self.norm.weight.dtype)
            return epi(lo, self.up.weight, xn), inj
        xn = self.norm(h)
        xf = xn.reshape(B, T, hc * D)
        lo = nn.silu(self.down(xf) * (1.0 / hc))
        gate = mx.sigmoid(self.up(lo)).reshape(B, T, hc, D)
        mixed = (gate * xn).mean(axis=2)
        if "inject" not in self:
            return mixed
        inj = 2.0 * mx.sigmoid(self.inject(xf) * (1.0 / hc))
        return mixed, inj

    def _hclr_ok(self, h_dtype) -> bool:
        """Fused-path eligibility: kq hc_lowrank ops present, q8_0 down/up
        wire, f32 inject, half norm gain, kernel-aligned shapes. Cached per
        module after the first call (weights are frozen post-load)."""
        ok = self.__dict__.get("_hclr_cache")
        if ok is None:
            ok = (
                self.hc == 4
                and "inject" in self
                and getattr(self.down, "kquant_type", None) == "q8_0"
                and getattr(self.up, "kquant_type", None) == "q8_0"
                and self.inject.weight.dtype == mx.float32
                and self.norm.weight.dtype in (mx.float16, mx.bfloat16)
                and self.hidden % 64 == 0
                and self.down.weight.shape[0] % 32 == 0
                and self.down.weight.shape[0] <= 512
            )
            self.__dict__["_hclr_cache"] = ok
        return (ok and _kq_hc() is not None
                and (h_dtype == mx.float32
                     or h_dtype == self.norm.weight.dtype))


def _hc_combine(h: mx.array, out: mx.array, inject: mx.array) -> mx.array:
    return h + out[:, :, None, :] * inject[..., None]


# Gated DeltaNet


class GatedDeltaNet(nn.Module):
    """qwen3.5 gated DeltaNet with a sigmoid output gate and the GGUF tiled
    K->V head pairing applied explicitly. Attribute and cache layout match
    mlx-lm's ``GatedDeltaNet`` so the fused S=1 decode kernel body in
    ``gdn_patches`` (conv -> silu -> q/k norm -> scan -> gated norm in one
    launch) runs unchanged; ``gdn_gate_sigmoid`` selects its gate."""

    gdn_gate_sigmoid = True

    def __init__(self, args: ModelArgs):
        super().__init__()
        self._fused_decode = False
        self._fused_verify = False
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.eps = args.rms_norm_eps

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = RMSNormGatedSigmoid(self.head_v_dim, eps=self.eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _fused_decode_ok(self, B, S, mask, cache) -> bool:
        if not (self._fused_decode and S == 1 and cache is not None
                and cache[1] is not None and cache[1].shape[0] == B):
            return False
        if mask is not None and isinstance(mask, mx.array):
            return False
        import gmlx.upstream.gdn_patches as _gp

        return (
            _gp._gdn_fused_decode_kernel is not None and _gp.gpu_active()
            and self.head_v_dim % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        )

    def _fused_verify_ok(self, B, mask, cache) -> bool:
        if not (self._fused_verify and cache is not None):
            return False
        if mask is not None and not (
            isinstance(mask, mx.array) and mask.ndim == 2 and mask.shape[0] == B
        ):
            return False
        import gmlx.upstream.gdn_patches as _gp

        return (
            _gp._gdn_fused_verify_kernel is not None and _gp.gpu_active()
            and self.head_v_dim % _gp.gdn_sg(B) == 0
            and self.head_k_dim % 32 == 0
        )

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None,
                 cache=None, gdn_sink=None) -> mx.array:
        """``gdn_sink`` (a list) marks the MTP verify forward: every call
        appends the record ``rollback_verify_sink`` needs to rewind this
        layer's cache to a shorter accepted prefix."""
        B, S, _ = inputs.shape
        if self._fused_decode_ok(B, S, mask, cache):
            import gmlx.upstream.gdn_patches as _gp

            return _gp._gdn_fused_decode_body(self, inputs, cache)
        pre = (cache[0], cache[1]) if cache is not None else (None, None)
        if gdn_sink is not None and S > 1 and self._fused_verify_ok(B, mask, cache):
            import gmlx.upstream.gdn_patches as _gp

            rec: list = []
            out = _gp._gdn_fused_verify_body(self, inputs, mask, cache, rec)
            gdn_sink.append({
                "kind": "gdn", "layer": self, "cache": cache, "pre": pre,
                "inputs": inputs, "mask": mask, "conv_input": rec[-1][9],
                "inter": rec[-1][11], "K": self.conv_kernel_size,
            })
            return out
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        n_keep = self.conv_kernel_size - 1
        conv_state = None
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
            if conv_state.shape[0] != B:
                conv_state = None
        if conv_state is None:
            conv_state = mx.zeros((B, n_keep, self.conv_dim), dtype=inputs.dtype)

        if mask is not None:
            if mask.shape[0] != B:
                mask = None
            else:
                qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            lengths = getattr(cache, "lengths", None)
            if lengths is not None:
                ends = mx.clip(lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        if gdn_sink is not None:
            gdn_sink.append({
                "kind": "gdn", "layer": self, "cache": cache, "pre": pre,
                "inputs": inputs, "mask": mask, "conv_input": conv_input,
                "inter": None, "K": self.conv_kernel_size,
            })
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                (self.num_k_heads, self.num_k_heads, self.num_v_heads),
                (self.head_k_dim, self.head_k_dim, self.head_v_dim),
            )
        ]
        inv_scale = self.head_k_dim ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, self.eps)
        k = inv_scale * mx.fast.rms_norm(k, None, self.eps)
        if self.num_v_heads != self.num_k_heads:
            # GGUF tiled V layout: V head hv pairs with K head hv % Hk.
            r = self.num_v_heads // self.num_k_heads
            q = mx.tile(q, [1, 1, r, 1])
            k = mx.tile(k, [1, 1, r, 1])

        state = cache[1] if cache is not None else None
        if state is not None and state.shape[0] != B:
            state = None
        out, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            if hasattr(cache, "advance"):
                cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


_KQ_TOPK_UNSET = object()
_kq_topk_fn = _KQ_TOPK_UNSET
_kq_score_fn = _KQ_TOPK_UNSET
_kq_hc_fns = _KQ_TOPK_UNSET
_kq_paged_fn = _KQ_TOPK_UNSET
_kq_bs_fn = _KQ_TOPK_UNSET


def _kq_hc():
    """kq fused low-rank hyper-connection (norm, front, epilogue) ops, or
    None (eager op chain). ``GMLX_Q4_HC_FUSED=0`` disables."""
    global _kq_hc_fns
    if _kq_hc_fns is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fns = None
        if env_bool("GMLX_Q4_HC_FUSED", True):
            try:
                import mlx_kquant as _kq
                norm = getattr(_kq, "hc_lowrank_norm", None)
                front = getattr(_kq, "hc_lowrank_front", None)
                epi = getattr(_kq, "hc_lowrank_epilogue", None)
                if (norm is not None and front is not None
                        and epi is not None
                        and mx.default_device().type == mx.DeviceType.gpu):
                    fns = (norm, front, epi)
            except ImportError:
                pass
        _kq_hc_fns = fns
    return _kq_hc_fns


def _kq_paged():
    """kq page-gather decode sdpa with 4-row pages at head_dim 256, or
    None (gathered-KV eager path). ``GMLX_Q4_QSA_PAGED_SDPA=0`` disables.
    Requires a kq build whose sdpa_decode_gqa_paged accepts tile_c=4
    (probed once with a dry call)."""
    global _kq_paged_fn
    if _kq_paged_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_PAGED_SDPA", True):
            try:
                import mlx_kquant as _kq
                cand = getattr(_kq, "sdpa_decode_gqa_paged", None)
                if cand is not None and mx.default_device().type == mx.DeviceType.gpu:
                    try:
                        cand(mx.zeros((1, 2, 1, 256), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 8, 256), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 8, 256), dtype=mx.bfloat16),
                             1.0, mx.zeros((1, 1, 2), dtype=mx.int32),
                             tile_c=4)
                        fn = cand
                    except (TypeError, ValueError):
                        fn = None  # older kq: fixed 16-row pages
            except ImportError:
                pass
        _kq_paged_fn = fn
    return _kq_paged_fn


def _kq_bs_prefill():
    """kq block-sparse FA prefill over QSA-selected 4-row pages, or None
    (dense-masked stock FA). ``GMLX_Q4_QSA_BS_PREFILL=0`` disables."""
    global _kq_bs_fn
    if _kq_bs_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_BS_PREFILL", True):
            try:
                import mlx_kquant as _kq
                if mx.default_device().type == mx.DeviceType.gpu:
                    fn = getattr(_kq, "sdpa_prefill_block_sparse", None)
            except ImportError:
                pass
        _kq_bs_fn = fn
    return _kq_bs_fn


def _kq_topk():
    """kq radix top-k for the QSA block selection, or None (stock
    argpartition). ``GMLX_Q4_QSA_KQ_TOPK=0`` disables."""
    global _kq_topk_fn
    if _kq_topk_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_KQ_TOPK", True):
            try:
                import mlx_kquant as _kq
                if mx.default_device().type == mx.DeviceType.gpu:
                    fn = getattr(_kq, "dsa_topk_indices", None)
            except ImportError:
                pass
        _kq_topk_fn = fn
    return _kq_topk_fn


def _kq_score():
    """kq fused decode-width indexer score (4-head band), or None.
    ``GMLX_Q4_QSA_KQ_SCORE=0`` disables. Requires a kq build whose
    dsa_indexer_score_decode accepts 4 heads (probed once with a dry
    shape check on the host validator)."""
    global _kq_score_fn
    if _kq_score_fn is _KQ_TOPK_UNSET:
        from gmlx.envflags import env_bool
        fn = None
        if env_bool("GMLX_Q4_QSA_KQ_SCORE", True):
            try:
                import mlx_kquant as _kq
                cand = getattr(_kq, "dsa_indexer_score_decode", None)
                if cand is not None and mx.default_device().type == mx.DeviceType.gpu:
                    try:
                        cand(mx.zeros((1, 4, 1, 128), dtype=mx.bfloat16),
                             mx.zeros((1, 8, 128), dtype=mx.bfloat16),
                             mx.zeros((1, 1, 4), dtype=mx.bfloat16), 0, 4)
                        fn = cand
                    except ValueError:
                        fn = None  # older kq: 64-head band only
            except ImportError:
                pass
        _kq_score_fn = fn
    return _kq_score_fn


# mrope (vision positions). Text-only calls pass positions=None and take the
# scalar-offset mx.fast.rope path; the two agree exactly when the three
# position streams are equal (interleaved sections tile the same frequency
# ladder). The VLM wrapper passes [3, B, L] t/h/w position ids.


def _mrope_selector(sections: List[int], half: int) -> mx.array:
    """``[half]`` stream index per frequency: freq ``j`` reads stream
    ``j % 3`` while ``j < sections[j % 3] * 3`` (HF interleaved layout)."""
    sel = []
    for j in range(half):
        src = j % 3
        assert j < sections[src] * 3, (j, sections)
        sel.append(src)
    return mx.array(sel, dtype=mx.int32)


def _mrope_cos_sin(positions: mx.array, rotary_dims: int, base: float,
                   selector: mx.array):
    """``positions [3, B, L]`` -> ``(cos, sin)`` each ``[B, L, rotary_dims]``
    in the non-traditional (rotate-half) layout."""
    half = rotary_dims // 2
    inv = mx.power(base, -mx.arange(0, half, dtype=mx.float32) / half)
    freqs = positions.astype(mx.float32)[..., None] * inv  # [3, B, L, half]
    _, B, L, _ = freqs.shape
    idx = mx.broadcast_to(selector[None, None, None, :], (1, B, L, half))
    freqs = mx.take_along_axis(freqs, idx.astype(mx.uint32), axis=0)[0]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_rope_cos_sin(x: mx.array, cos: mx.array, sin: mx.array,
                        rotary_dims: int) -> mx.array:
    """Rotate the first ``rotary_dims`` of ``x [B, H, L, D]`` with
    ``cos/sin [B, L, rotary_dims]`` (rotate-half pairing)."""
    half = rotary_dims // 2
    xr, xp = x[..., :rotary_dims], x[..., rotary_dims:]
    rot = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    c, sn = cos[:, None].astype(x.dtype), sin[:, None].astype(x.dtype)
    return mx.concatenate([xr * c + rot * sn, xp], axis=-1)


# QSA: cache, indexer, attention


class QSAKVCache(KVCache):
    """KVCache extended with the QSA indexer key stream.

    ``ik`` holds one raw (pre-norm, pre-rope) indexer key per cached token,
    appended in lockstep with K/V so ``offset`` covers all three streams.
    ``blocks`` caches the finished (mean-pooled, normed, roped) block keys
    derived from ``ik``; it is a pure function of the raw stream, so trim and
    state restore just invalidate it. Quantizing would drop the stream, so
    ``to_quantized`` is refused.
    """

    kv_quant_unsupported = True

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.ratio = ratio
        self.ik = None       # [B, cap, index_dim]
        self.pos = None      # [3, B, cap] mrope positions (VLM loads only)
        self.blocks = None   # [B, n_blocks, index_dim]
        self.n_blocks = 0

    def update_and_fetch_qsa(self, keys, values, ik, pos=None):
        prev = self.offset
        k, v = super().update_and_fetch(keys, values)
        n = ik.shape[1]
        if self.ik is None or (prev + n) > self.ik.shape[1]:
            B, _, idx_dim = ik.shape
            n_steps = (self.step + n - 1) // self.step
            new = mx.zeros((B, n_steps * self.step, idx_dim), ik.dtype)
            if self.ik is not None:
                if prev % self.step != 0:
                    self.ik = self.ik[:, :prev, :]
                self.ik = mx.concatenate([self.ik, new], axis=1)
            else:
                self.ik = new
            if pos is not None:
                newp = mx.zeros((3, B, self.ik.shape[1] - (
                    0 if self.pos is None else self.pos.shape[2])), mx.int32)
                self.pos = (newp if self.pos is None else
                            mx.concatenate([self.pos[:, :, :prev] if prev %
                                            self.step else self.pos, newp],
                                           axis=2))
        self.ik[:, prev:prev + n, :] = ik
        if pos is not None:
            self.pos[:, :, prev:prev + n] = pos.astype(mx.int32)
        return k, v, self.ik[:, :self.offset, :]

    def finished_blocks(self, n_blocks: int, finish):
        """Block keys ``[B, n_blocks, D]``; ``finish(raw, start_block)`` turns
        ``[B, m * ratio, D]`` raw keys into ``[B, m, D]`` finished ones."""
        r = self.ratio
        if self.blocks is not None and self.blocks.shape[0] != self.ik.shape[0]:
            self.blocks, self.n_blocks = None, 0
        if n_blocks > self.n_blocks:
            raw = self.ik[:, self.n_blocks * r:n_blocks * r, :]
            new = finish(raw, self.n_blocks)
            self.blocks = new if self.blocks is None else mx.concatenate(
                [self.blocks[:, :self.n_blocks], new], axis=1)
            self.n_blocks = n_blocks
        return self.blocks[:, :n_blocks]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            base = (self.keys, self.values, self.ik[:, :self.offset])
            return base if self.pos is None else base + (
                self.pos[:, :, :self.offset],)
        tail = () if self.pos is None else (self.pos[:, :, :self.offset],)
        return (
            self.keys[..., :self.offset, :],
            self.values[..., :self.offset, :],
            self.ik[:, :self.offset],
        ) + tail

    @state.setter
    def state(self, v):
        if len(v) == 4:
            self.keys, self.values, self.ik, self.pos = v
        else:
            self.keys, self.values, self.ik = v
        self.offset = self.keys.shape[2]
        self.blocks, self.n_blocks = None, 0

    @property
    def meta_state(self):
        return str(self.ratio)

    @meta_state.setter
    def meta_state(self, v):
        if v:
            self.ratio = int(v)

    def trim(self, n):
        n = super().trim(n)
        self.n_blocks = min(self.n_blocks, self.offset // self.ratio)
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        raise NotImplementedError(
            "QSAKVCache cannot quantize: the QSA indexer key stream has no "
            "quantized form (drop --kv-bits for qwen4exp)")

    @property
    def nbytes(self):
        n = super().nbytes
        if self.ik is not None:
            n += self.ik.nbytes
        if self.blocks is not None:
            n += self.blocks.nbytes
        return n


class QSAIndexer(nn.Module):
    """Block selector for one attention layer."""

    def __init__(self, args: ModelArgs, ratio: int, rotary_dims: int):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.ratio = ratio
        self.block_topk = args.indexer_budget // ratio
        self.rotary_dims = rotary_dims
        self.rope_theta = args.rope_theta
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def _rope(self, x, offset, scale=1.0):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=scale, offset=offset)

    def _pooled(self, raw: mx.array):
        B, n, D = raw.shape
        m = n // self.ratio
        pooled = raw[:, :m * self.ratio].reshape(B, m, self.ratio, D)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw.dtype)
        return self.k_norm(pooled)

    def finish_blocks(self, raw: mx.array, start_block: int) -> mx.array:
        """Mean-pool ``[B, m * r, D]`` raw keys into ``[B, m, D]`` block keys,
        k_norm, then rope at the block start positions (``scale=r`` makes
        position ``i`` of the call land on token ``(start_block + i) * r``)."""
        return self._rope(self._pooled(raw)[:, None], start_block,
                          scale=float(self.ratio))[:, 0]

    def finish_blocks_at(self, raw: mx.array, start_block: int,
                         pos_all: mx.array, selector: mx.array) -> mx.array:
        """``finish_blocks`` with explicit mrope positions: block ``i`` is
        roped at the cached position of its start token
        (``pos_all[:, :, (start_block + i) * r]``), matching the reference's
        ``full_cos.index_select(group_starts)``."""
        pooled = self._pooled(raw)
        m = pooled.shape[1]
        starts = (start_block + mx.arange(m, dtype=mx.int32)) * self.ratio
        pos = mx.take(pos_all, starts.astype(mx.uint32), axis=2)  # [3, B, m]
        cos, sin = _mrope_cos_sin(pos, self.rotary_dims, self.rope_theta,
                                  selector)
        return _apply_rope_cos_sin(pooled[:, None], cos, sin,
                                   self.rotary_dims)[:, 0]

    def _queries(self, x: mx.array, offset: int, cos, sin) -> mx.array:
        """Normed, roped indexer queries ``[B, H, L, D]``."""
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        if cos is not None:
            return _apply_rope_cos_sin(q, cos, sin, self.rotary_dims)
        return self._rope(q, offset)

    def scores(self, x: mx.array, blocks: mx.array, offset: int,
               cos=None, sin=None) -> mx.array:
        """``[B, L, n_blocks]`` f32 block scores for the queries in ``x``."""
        q = self._queries(x, offset, cos, sin)
        s = q.astype(mx.float32) @ blocks.astype(mx.float32)[:, None].transpose(0, 1, 3, 2)
        s = mx.maximum(s, 0).sum(axis=1)
        return s * (1.0 / math.sqrt(self.head_dim))

    def select(self, x: mx.array, ik_all: mx.array, cache, offset: int,
               cos=None, sin=None, pos_all=None, selector=None):
        """Top-k complete blocks per query, or None when every query is
        still dense. Returns ``(blocks [B, L, topk], complete_counts [L])``."""
        B, L, _ = x.shape
        key_len = ik_all.shape[1]
        r = self.ratio
        n_blocks = key_len // r
        if n_blocks <= self.block_topk:
            return None
        if pos_all is not None:
            def finish(raw, start_block):
                return self.finish_blocks_at(raw, start_block, pos_all,
                                             selector)
        else:
            finish = self.finish_blocks
        if cache is not None:
            blocks = cache.finished_blocks(n_blocks, finish)
        else:
            blocks = finish(ik_all[:, :n_blocks * r], 0)
        query_ends = offset + mx.arange(L) + 1
        complete = query_ends // r
        k = self.block_topk
        topk = _kq_topk()
        kq_ok = (topk is not None and k == 512 and n_blocks >= k
                 and x.dtype in (mx.float16, mx.bfloat16))
        if kq_ok and L <= 4 and _kq_score() is not None:
            # Fused decode/verify path: one kernel scores every pooled block
            # (relu dots summed over the 4 heads, per-row visibility baked as
            # finite_min) and the radix top-k consumes its 16-bit rows
            # directly. Replaces the astype/matmul/relu/sum/where chain.
            q = self._queries(x, offset, cos, sin).astype(x.dtype)
            w = mx.full((B, L, self.n_heads),
                        1.0 / math.sqrt(self.head_dim), dtype=x.dtype)
            s16 = _kq_score()(q, blocks.astype(x.dtype), w, offset, self.ratio)
            sel = topk(s16, k, True)[:, 0].astype(mx.int64)
            return sel, complete
        s = self.scores(x, blocks, offset, cos=cos, sin=sin)
        valid = mx.arange(n_blocks)[None, None, :] < complete[None, :, None]
        s = mx.where(valid, s, -mx.inf)
        if kq_ok:
            # kq radix top-k (one threadgroup per row) replaces the
            # sort-based argpartition. Selection is set-equivalent up to
            # ties at the threshold, and the mask/gather consumers are
            # order-insensitive. Scores narrow to the activation dtype for
            # the kernel's 16-bit wire.
            sel = topk(s.astype(x.dtype)[:, None], k, True)[:, 0]
            sel = sel.astype(mx.int64)
        else:
            sel = mx.argpartition(s, kth=-k, axis=-1)[..., -k:]
        return sel, complete


def _qsa_token_mask(sel, complete, offset, L, key_len, ratio, topk):
    """Boolean ``[B, 1, L, key_len]`` attention mask from the block selection:
    selected block members plus the incomplete tail, or plain causal for
    queries with at most ``topk`` complete blocks."""
    B = sel.shape[0]
    n_blocks = key_len // ratio
    blk = mx.zeros((B, L, n_blocks + 1), dtype=mx.bool_)
    blk = mx.put_along_axis(blk, sel, mx.array(True), axis=-1)[..., :n_blocks]
    tok_sel = mx.repeat(blk, ratio, axis=-1)
    pad = key_len - n_blocks * ratio
    if pad:
        tok_sel = mx.concatenate(
            [tok_sel, mx.zeros((B, L, pad), dtype=mx.bool_)], axis=-1)
    tok = mx.arange(key_len)[None, :]
    query_ends = (offset + mx.arange(L) + 1)[:, None]
    tail_start = (complete * ratio)[:, None]
    causal = tok < query_ends
    tail = (tok >= tail_start) & causal
    use_sparse = (complete > topk)[:, None]
    m = mx.where(use_sparse[None], tok_sel | tail[None], causal[None])
    return m[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        self.q_proj = nn.Linear(args.hidden_size, H * D * 2, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, Hkv * D, bias=False)
        self.o_proj = nn.Linear(H * D, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(D, eps=args.rms_norm_eps)
        self.ratio = int(args.compress_ratios[layer_idx])
        self._mrope_selector = _mrope_selector(
            list(args.mrope_section), self.rotary_dims // 2)
        if self.ratio > 0:
            self.indexer = QSAIndexer(args, self.ratio, self.rotary_dims)

    def _rope(self, x, offset):
        return mx.fast.rope(x, self.rotary_dims, traditional=False,
                            base=self.rope_theta, scale=1.0, offset=offset)

    def _gathered_attention(self, q, k, v, sel, complete, offset, L):
        """Decode / verify: gather the selected keys per query and run one
        ``B * L``-batched qL=1 sdpa over ``topk * r + r`` keys."""
        B, H, _, D = q.shape
        Hkv = k.shape[1]
        r, topk = self.ratio, self.indexer.block_topk
        members = (sel[..., None] * r + mx.arange(r)).reshape(B, L, topk * r)
        tail_start = (complete * r)[None, :, None]
        tail = tail_start + mx.arange(r)[None, None, :]
        query_ends = (offset + mx.arange(L) + 1)[None, :, None]
        tail_ok = tail < query_ends
        idx = mx.concatenate([members, mx.minimum(tail, query_ends - 1)], axis=-1)
        W = idx.shape[-1]
        bias = mx.concatenate(
            [mx.ones((B, L, topk * r), dtype=mx.bool_), tail_ok], axis=-1)
        flat = mx.broadcast_to(idx.reshape(B, 1, L * W, 1), (B, Hkv, L * W, 1))
        k_sel = mx.take_along_axis(k, flat, axis=2).reshape(B, Hkv, L, W, D)
        v_sel = mx.take_along_axis(v, flat, axis=2).reshape(B, Hkv, L, W, D)
        k_sel = k_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        v_sel = v_sel.transpose(0, 2, 1, 3, 4).reshape(B * L, Hkv, W, D)
        q_l = q.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D)
        out = mx.fast.scaled_dot_product_attention(
            q_l, k_sel, v_sel, scale=self.scale,
            mask=bias.reshape(B * L, 1, 1, W))
        return out.reshape(B, L, H, D).transpose(0, 2, 1, 3)

    def _block_sparse_prefill(self, q, k, v, sel, offset, key_len, bs):
        """Prefill chunk with every query sparse: fold queries into 4-wide
        windows and walk each window's own page list (union of its queries'
        selected blocks plus the window's tail span) through the kq
        block-sparse FA kernel. One host sync reads the widest union;
        prefill graphs rebuild per chunk anyway."""
        L = q.shape[2]
        r = self.ratio
        n_qt = L // r
        nb_total = (key_len + r - 1) // r
        topk = self.indexer.block_topk
        g = sel[0].astype(mx.int32).reshape(n_qt, r * topk)
        w4 = offset + mx.arange(n_qt, dtype=mx.int32) * r
        span = mx.stack(
            [w4 // r, mx.minimum((w4 + r) // r, nb_total - 1)], axis=1)
        srt = mx.sort(mx.concatenate([g, span], axis=1), axis=1)
        newv = mx.concatenate(
            [mx.ones((n_qt, 1), dtype=mx.bool_), srt[:, 1:] != srt[:, :-1]],
            axis=1)
        counts = newv.sum(axis=1).astype(mx.int32)
        max_p = int(counts.max())  # host sync
        slot = mx.cumsum(newv.astype(mx.int32), axis=1) - 1
        pages = mx.put_along_axis(
            mx.full((n_qt, max_p), -1, dtype=mx.int32), slot, srt, axis=1)
        lut = mx.put_along_axis(
            mx.zeros((n_qt, nb_total), dtype=mx.int32), srt, slot, axis=1)
        pm = mx.zeros((n_qt, max_p), dtype=mx.uint16)
        selq = sel[0].astype(mx.int32).reshape(n_qt, r, topk)
        for qi in range(r):
            slots_q = mx.take_along_axis(lut, selq[:, qi], axis=1)
            pm = pm | mx.put_along_axis(
                mx.zeros((n_qt, max_p), dtype=mx.uint16), slots_q,
                mx.array(1 << qi, dtype=mx.uint16), axis=1)
        return bs(q, k, v, self.scale, pages, pm, counts, offset)

    def _paged_attention(self, q, k, v, sel, key_len, paged):
        """Decode (L=1): kq page-gather sdpa straight over the KV cache
        with the selected 4-row blocks as pages -- no per-token K/V copy.
        The partial tail block is appended when present; a full tail
        contributes nothing in the gathered path and is omitted here."""
        B = q.shape[0]
        Hkv = k.shape[1]
        pages = sel[:, 0, :].astype(mx.int32)
        if key_len % self.ratio:
            tail = mx.full((B, 1), key_len // self.ratio, dtype=mx.int32)
            pages = mx.concatenate([pages, tail], axis=-1)
        pages = mx.broadcast_to(pages[:, None, :], (B, Hkv, pages.shape[-1]))
        return paged(q, k, v, self.scale, pages, tile_c=4)

    def __call__(self, x: mx.array, mask=None, cache=None,
                 positions=None) -> mx.array:
        """``positions`` is the ``[3, B, L]`` mrope position block for this
        window (VLM loads); ``None`` takes the scalar-offset text path."""
        B, L, _ = x.shape
        H, Hkv, D = self.num_attention_heads, self.num_key_value_heads, self.head_dim
        qg = self.q_proj(x).reshape(B, L, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, L, Hkv, D)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, Hkv, D).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        cos = sin = None
        if positions is not None:
            cos, sin = _mrope_cos_sin(positions, self.rotary_dims,
                                      self.rope_theta, self._mrope_selector)
            q = _apply_rope_cos_sin(q, cos, sin, self.rotary_dims)
            k = _apply_rope_cos_sin(k, cos, sin, self.rotary_dims)
        else:
            q = self._rope(q, offset)
            k = self._rope(k, offset)

        selection = None
        if "indexer" in self:
            ik = self.indexer.k_proj(x)
            if cache is not None and hasattr(cache, "update_and_fetch_qsa"):
                k, v, ik_all = cache.update_and_fetch_qsa(k, v, ik,
                                                          pos=positions)
                selection = self.indexer.select(
                    x, ik_all, cache, offset, cos=cos, sin=sin,
                    pos_all=cache.pos, selector=self._mrope_selector)
            elif cache is not None:
                k, v = cache.update_and_fetch(k, v)
            else:
                selection = self.indexer.select(
                    x, ik, None, 0, cos=cos, sin=sin,
                    pos_all=(positions.astype(mx.int32)
                             if positions is not None else None),
                    selector=self._mrope_selector)
        elif cache is not None:
            k, v = cache.update_and_fetch(k, v)

        if selection is None:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask)
        else:
            sel, complete = selection
            key_len = k.shape[2]
            all_sparse = (offset + 1) // self.ratio > self.indexer.block_topk
            paged = _kq_paged() if (
                L == 1 and self.ratio == 4 and D == 256
                and q.dtype in (mx.bfloat16, mx.float16)) else None
            if paged is not None and all_sparse and not isinstance(mask, mx.array):
                out = self._paged_attention(q, k, v, sel, key_len, paged)
            elif L <= 8 and all_sparse and not isinstance(mask, mx.array):
                out = self._gathered_attention(q, k, v, sel, complete, offset, L)
            elif (B == 1 and all_sparse and not isinstance(mask, mx.array)
                  and L % 4 == 0 and self.ratio == 4 and D == 256
                  and H == 12 * Hkv and q.dtype in (mx.bfloat16, mx.float16)
                  and (bs := _kq_bs_prefill()) is not None):
                out = self._block_sparse_prefill(q, k, v, sel, offset,
                                                 key_len, bs)
            else:
                qsa = _qsa_token_mask(sel, complete, offset, L, key_len,
                                      self.ratio, self.indexer.block_topk)
                if isinstance(mask, mx.array):
                    if mask.dtype == mx.bool_:
                        qsa = qsa & mask
                    else:
                        qsa = mask + mx.where(qsa, 0.0, -mx.inf).astype(mask.dtype)
                out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=qsa)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# MoE (qwen3-next shape: router + SwitchGLU + sigmoid-gated shared expert)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, self.num_experts)
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


# PLE n-gram hash embeddings


def _make_ple_hash_kernel():
    """One-dispatch n-gram row hash: thread (b, t, head) recomputes the
    eager chain (shift-with-EOS-cut, 64-bit mix, mod + offset) for its
    head. Integer ALU only. hist is [B, T + CTX] int64; out [B, T, NH]
    int32 (table rows stay below 2^31)."""
    source = """
        const uint gid = thread_position_in_grid.x;
        const int T = hist_shape[1] - CTX;
        const int nh = gid % NH;
        const int t = (gid / NH) % T;
        const int b = gid / (uint)(NH * T);
        const size_t hbase = (size_t)b * (T + CTX);
        const int p = CTX + t;
        const int n = 2 + nh / HPN;
        ulong mixed = (ulong)hist[hbase + p] * mults[0];
        bool eos_seen = false;
        for (int sh = 1; sh < n; sh++) {
            const long tok = hist[hbase + p - sh];
            eos_seen = eos_seen || (tok == EOS);
            mixed ^= (eos_seen ? (ulong)EOS : (ulong)tok) * mults[sh];
        }
        out[gid] = (int)(mixed % sizes[nh] + offsets[nh]);
    """
    return mx.fast.metal_kernel(
        name="ple_hash_rows",
        input_names=["hist", "mults", "sizes", "offsets"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_ple_hash_kernel = None


def _ple_hash():
    """The fused PLE row-hash kernel, or None (eager op chain).
    ``GMLX_Q4_PLE_FUSED_HASH=0`` disables."""
    global _ple_hash_kernel
    if _ple_hash_kernel is None:
        from gmlx.envflags import env_bool
        if (env_bool("GMLX_Q4_PLE_FUSED_HASH", True)
                and mx.default_device().type == mx.DeviceType.gpu):
            _ple_hash_kernel = _make_ple_hash_kernel()
        else:
            _ple_hash_kernel = False
    return _ple_hash_kernel or None


class PLEEmbedding(nn.Module):
    """Hash the token history into per-head row ids and gather the rows.

    Head ``h`` of n-gram order ``n`` (2 <= n <= ngram_size) hashes
    ``ctx[0] * m[0] ^ ... ^ ctx[n-1] * m[n-1]`` (uint64 wraparound) into
    ``mixed % vocab[h] + offset[h]``. ``ctx[s]`` is the token ``s`` positions
    back, or EOS when an EOS sits anywhere in between (the token's own EOS
    does not cut its context). The multipliers are 45-bit and token ids
    18-bit, so the products never reach the sign bit and signed (HF) and
    unsigned (llama.cpp) arithmetic agree.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ngram_size = args.ple_ngram_size
        self.heads_per_ngram = args.ple_heads_per_ngram
        self.context_len = self.ngram_size - 1
        self.n_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = args.ple_eos_token_id
        self.embed_dim = args.ple_embed_dim
        self._mults = [int(m) for m in args.ple_layer_multipliers]
        self._mults_u64 = mx.array(self._mults, dtype=mx.uint64)
        self._sizes = mx.array([int(s) for s in args.ple_head_vocab_sizes],
                               dtype=mx.uint64)
        self._offsets = mx.array([int(o) for o in args.ple_head_offsets],
                                 dtype=mx.uint64)

    def _shift_right_ignore_eos(self, tokens: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return tokens
        B, T = tokens.shape
        positions = mx.arange(T, dtype=mx.int64)
        eos_pos = mx.where(tokens == self.eos_token_id, positions[None], -1)
        prev_eos_incl = mx.cummax(eos_pos, axis=1)
        prev_eos = mx.concatenate(
            [mx.full((B, 1), -1, dtype=mx.int64), prev_eos_incl[:, :-1]], axis=1)
        src = positions - shift
        gathered = mx.take_along_axis(
            tokens, mx.broadcast_to(mx.maximum(src, 0)[None], (B, T)), axis=1)
        valid = (src[None] > prev_eos) & (src[None] >= 0)
        return mx.where(valid, gathered, self.eos_token_id)

    def prev_history(self, cache, B: int) -> mx.array:
        """The ``[B, context_len]`` token history before this call (EOS
        filled when the cache is fresh or was built for another batch)."""
        if cache is not None and cache[3] is not None and cache[3].shape[0] == B:
            return cache[3]
        return mx.full((B, self.context_len), self.eos_token_id, dtype=mx.int64)

    def row_ids(self, input_ids: mx.array, cache, prev=None) -> mx.array:
        """``[B, T, n_heads]`` table rows for the tokens in ``input_ids``."""
        ids = input_ids.astype(mx.int64)
        B, T = ids.shape
        if prev is None:
            prev = self.prev_history(cache, B)
        hist = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(hist[:, -self.context_len:])
        kern = _ple_hash() if self.ngram_size <= 3 else None
        if kern is not None:
            nh = self.n_heads
            return kern(
                inputs=[hist, self._mults_u64, self._sizes, self._offsets],
                template=[("CTX", self.context_len),
                          ("HPN", self.heads_per_ngram),
                          ("NH", nh),
                          ("EOS", self.eos_token_id)],
                grid=(B * T * nh, 1, 1),
                threadgroup=(min(256, B * T * nh), 1, 1),
                output_shapes=[(B, T, nh)],
                output_dtypes=[mx.int32],
            )[0]
        shifted = [
            self._shift_right_ignore_eos(hist, s).astype(mx.uint64)
            for s in range(self.ngram_size)
        ]
        mults = [mx.array(m, dtype=mx.uint64) for m in self._mults]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            start = (n - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * mults[0]
            for p in range(1, n):
                mixed = mx.bitwise_xor(mixed, shifted[p] * mults[p])
            rows = mixed[..., None] % self._sizes[start:end] + self._offsets[start:end]
            blocks.append(rows)
        rows = mx.concatenate(blocks, axis=-1)[:, -T:]
        return rows.astype(mx.int32)

    def __call__(self, input_ids: mx.array, cache, table, prev=None) -> mx.array:
        """``table`` is the model-level row table (``model.ple_embed``); it is
        passed in rather than owned so the 320M-row weight has one parameter
        path independent of which layer hosts the PLE block."""
        rows = self.row_ids(input_ids, cache, prev=prev)
        emb = table(rows)
        return emb.reshape(*emb.shape[:-2], -1)


class PLELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc = args.hc_count
        hc_dim = self.hidden_size * self.hc
        eps = args.rms_norm_eps
        self.embedding = PLEEmbedding(args)
        emb_dim = self.embedding.n_heads * args.ple_embed_dim
        self.key_proj = nn.Linear(emb_dim, hc_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, self.hidden_size, bias=False)
        self.norm_key = GroupedRMSNorm(hc_dim, eps)
        self.norm_query = GroupedRMSNorm(hc_dim, eps)
        self.norm_conv = GroupedRMSNorm(hc_dim, eps)
        self.conv_dilation = args.ple_ngram_size
        self.conv_kernel = args.ple_conv_kernel
        self.conv_state_len = (self.conv_kernel - 1) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_dim, hc_dim, kernel_size=self.conv_kernel,
            dilation=self.conv_dilation, groups=hc_dim, bias=False)

    def _conv(self, x: mx.array, cache) -> mx.array:
        B = x.shape[0]
        state = None
        if cache is not None and cache[2] is not None:
            state = cache[2]
            if state.shape[0] != B:
                state = None
        if state is None:
            state = mx.zeros((B, self.conv_state_len, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.conv_state_len:])
        return nn.silu(self.conv1d(conv_input)), conv_input

    def __call__(self, h: mx.array, input_ids: mx.array, cache, table,
                 mask=None, gdn_sink=None) -> mx.array:
        B, T, hc, D = h.shape
        prev = self.embedding.prev_history(cache, B)
        emb = self.embedding(input_ids, cache, table, prev=prev)
        keys = self.norm_key(self.key_proj(emb).reshape(B, T, hc, D))
        values = self.value_proj(emb)
        queries = self.norm_query(h)
        gate = (keys.astype(mx.float32) * queries.astype(mx.float32)).sum(
            axis=-1, keepdims=True) * (1.0 / math.sqrt(D))
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gate = mx.sigmoid(gate).astype(h.dtype)
        gated = gate * values[:, :, None, :]
        normed = self.norm_conv(gated).reshape(B, T, hc * D)
        gated = gated.reshape(B, T, hc * D)
        if isinstance(mask, mx.array) and mask.ndim == 2 and mask.shape[0] == B:
            gated = mx.where(mask[..., None], gated, 0)
            normed = mx.where(mask[..., None], normed, 0)
        conv, conv_input = self._conv(normed, cache)
        if gdn_sink is not None:
            gdn_sink.append({
                "kind": "ple", "cache": cache, "prev": prev, "ids": input_ids,
                "ctx": self.embedding.context_len, "conv_input": conv_input,
                "L": self.conv_state_len,
            })
        return h + (gated + conv).reshape(B, T, hc, D)


def prepare_runtime(model) -> dict:
    """Arm the owned kernel routes after the weights are loaded: the fused
    S=1 GDN decode kernel (``GMLX_FUSED_GDN=0`` disables) and the
    concatenated b/a decay matvec it consumes. Returns counts for the load
    log."""
    import gmlx.upstream.gdn_patches as _gp
    from gmlx.envflags import env_bool

    counts = {"gdn_fused": 0, "gdn_ba_cat": 0, "gdn_fused_verify": 0}
    want = env_bool("GMLX_FUSED_GDN", True)
    fused = want and _gp._gdn_fused_decode_kernel is not None
    fused_verify = want and _gp._gdn_fused_verify_kernel is not None
    cat_ba = env_bool("GMLX_GDN_BA_CAT", True)
    for m in model.modules():
        if isinstance(m, GatedDeltaNet):
            m._fused_decode = fused
            m._fused_verify = fused_verify
            counts["gdn_fused"] += int(fused)
            counts["gdn_fused_verify"] += int(fused_verify)
            if fused and cat_ba and _gp._gdn_try_cat_ba(m):
                counts["gdn_ba_cat"] += 1
    return counts


def rollback_verify_sink(sink, n: int) -> None:
    """Rewind the recurrent caches after an MTP verify forward over ``S``
    positions to the state after its first ``n`` (the accepted prefix).

    GDN: the conv state is a window of the recorded conv input and the scan
    state is the fused verify kernel's per-position intermediate; without
    intermediates (unfused path) the layer is re-run over the prefix from
    its pre-verify state. PLE: both the token history and the conv state are
    windows of recorded arrays. KV caches are trimmed by the caller.
    """
    for e in sink:
        cache = e["cache"]
        if cache is None:
            continue
        if e["kind"] == "gdn":
            K = e["K"]
            if e["inter"] is not None:
                cache[0] = mx.contiguous(e["conv_input"][:, n:n + K - 1, :])
                cache[1] = mx.contiguous(e["inter"][:, n - 1])
                continue
            cache[0], cache[1] = e["pre"]
            mask = e["mask"]
            if isinstance(mask, mx.array):
                mask = mask[:, :n]
            e["layer"](e["inputs"][:, :n], mask=mask, cache=cache)
        elif e["kind"] == "ple":
            hist = mx.concatenate([e["prev"], e["ids"][:, :n].astype(mx.int64)],
                                  axis=1)
            cache[3] = mx.contiguous(hist[:, -e["ctx"]:])
            cache[2] = mx.contiguous(e["conv_input"][:, n:n + e["L"], :])


# Layers and model


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        eps = args.rms_norm_eps
        self.hc_attn = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, eps)
        self.hc_ffn = HyperConnection(args.hidden_size, args.hc_count,
                                      args.hc_lowrank, eps)
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args, layer_idx)
        self.mlp = SparseMoeBlock(args)
        if layer_idx in args.ple_layer_ids:
            self.ple = PLELayer(args)

    def __call__(self, h: mx.array, input_ids: mx.array, mask=None, cache=None,
                 ple_table=None, gdn_sink=None, positions=None):
        if "ple" in self:
            h = self.ple(h, input_ids, cache, ple_table, mask, gdn_sink=gdn_sink)
        mixed, inject = self.hc_attn(h)
        if self.is_linear:
            out = self.linear_attn(mixed, mask=mask, cache=cache, gdn_sink=gdn_sink)
        else:
            out = self.self_attn(mixed, mask=mask, cache=cache,
                                 positions=positions)
        h = _hc_combine(h, out, inject)
        mixed, inject = self.hc_ffn(h)
        out = self.mlp(mixed)
        return _hc_combine(h, out, inject)


class Qwen4ExpModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.hc = args.hc_count
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperConnection(args.hidden_size, args.hc_count,
                                       args.hc_lowrank, args.rms_norm_eps,
                                       inject=False)
        if args.ple_layer_ids:
            # The PLE row table (wire bytes at load; KQuantEmbedding gathers
            # and dequantizes only the touched rows).
            self.ple_embed = nn.Embedding(max(int(args.ple_table_rows), 1),
                                          args.ple_embed_dim)
        self.ssm_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "linear_attention"), 0)
        self.fa_idx = next(
            (i for i, t in enumerate(args.layer_types) if t == "full_attention"), 0)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None,
                 return_streams: bool = False, gdn_sink=None, position_ids=None):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
        B, T, D = h.shape
        h = mx.broadcast_to(h[:, :, None, :], (B, T, self.hc, D))
        if cache is None:
            cache = [None] * len(self.layers)
        probe = h[:, :, 0, :]
        fa_mask = create_attention_mask(probe, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(probe, cache[self.ssm_idx])
        table = self.ple_embed if "ple_embed" in self else None
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, inputs, mask=mask, cache=c, ple_table=table,
                      gdn_sink=gdn_sink, positions=position_ids)
        out = self.hc_head(h)
        if return_streams:
            return out, h
        return out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches: List[Any] = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            elif layer.self_attn.ratio > 0:
                caches.append(QSAKVCache(layer.self_attn.ratio))
            else:
                caches.append(KVCache())
        return caches
