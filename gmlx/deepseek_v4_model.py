# Copyright (c) 2026 Apple Inc.
#
# DeepSeek V4 (Flash) model class, VENDORED from unmerged mlx-lm PR #1192
# (Blaizzy/mlx-lm branch pc/add-deepseekv4flash-model, head 5c10538) by way
# of omlx's patches/deepseek_v4/deepseek_v4_model.py port, pending upstream
# merge. Registered into the mlx_lm.models namespace at load time by
# ensure_registered() below (upstream wins if a real
# mlx_lm.models.deepseek_v4 exists). Delete this file once the installed
# mlx-lm ships models/deepseek_v4.py.
#
# Deviations from PR 1192:
#  - omlx custom-kernel dispatch (glm_moe_dsa sparse-attention/indexer
#    natives) stripped; the inline pure-MLX paths are the only paths.
#  - SwitchGLU comes from installed mlx-lm (no `scores=` fusion kwarg).
#  - QAT simulation round-trips ADDED to match the official graph (and the
#    dwarfstar C reference, ds4.c): the model was trained with simulated
#    quantization, so these are semantic:
#      * KV latent rows after RoPE: non-RoPE 448 dims through a per-64-block
#        FP8-E4M3 round-trip, then the whole 512-dim row through an FP16
#        round-trip (ds4.c dsv4_fp8_kv_quantize_row / f16_round).
#      * Compressed pool rows after norm+RoPE: same FP8 round-trip
#        (attention compressor) or Hadamard-128 + per-32-block FP4-E2M1
#        round-trip (indexer compressor).
#      * Indexer queries after RoPE: Hadamard-128 + FP4 round-trip.
#    Without these the lightning indexer's top-k selection diverges from
#    the model's training-time graph.

import importlib
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_flatten

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.pipeline import PipelineMixin
from mlx_lm.models.switch_layers import SwitchGLU

from gmlx import prefill_decay as _prefill_decay
from gmlx.deepseek_v4_cache import BatchPoolingCache, PoolingCache
from gmlx.deepseek_v4_hyper_connection import (
    HyperConnection,
    HyperHead,
    hc_expand,
)


# Native DSA kernels (mlx-kquant builds that ship dsa_sparse_attention /
# dsa_indexer_scores / dsa_topk_indices). Probed lazily; env-gated
# (GMLX_DSA_SPARSE / GMLX_DSA_INDEXER, default on when the symbols
# exist); any runtime failure permanently disables the failing path for the
# process and the inline pure-MLX code takes over (omlx dispatch pattern).
_dsa_state: Dict[str, Optional[bool]] = {
    "sparse": None,
    "indexer": None,
    "indexer_q": None,
    "window": None,
    "qat": None,
    "kv_qat": None,
}
_DSA_ENV = {
    "sparse": "GMLX_DSA_SPARSE",
    "indexer": "GMLX_DSA_INDEXER",
    "indexer_q": "GMLX_DSA_INDEXER_Q",
    "window": "GMLX_DSA_WINDOW",
    "qat": "GMLX_DSA_QAT",
    "kv_qat": "GMLX_DSA_KV_QAT",
}
_DSA_SYMS = {
    "sparse": ("dsa_sparse_attention",),
    "indexer": ("dsa_indexer_scores", "dsa_topk_indices"),
    "indexer_q": (
        "dsa_indexer_qat_quant",
        "dsa_indexer_qat_pack",
        "dsa_indexer_scores_q",
        "dsa_topk_indices",
    ),
    "window": ("dsa_sparse_attention",),
    "qat": ("dsa_indexer_qat",),
    "kv_qat": ("dsa_kv_qat",),
}


def _dsa_probe(path: str) -> bool:
    on = _dsa_state[path]
    if on is None:
        try:
            import mlx_kquant as kq

            has = all(hasattr(kq, sym) for sym in _DSA_SYMS[path])
        except Exception:
            has = False
        on = has and os.environ.get(_DSA_ENV[path], "1") != "0"
        _dsa_state[path] = on
    return on


def _dsa_disable(path: str, exc: Exception) -> None:
    _dsa_state[path] = False
    print(
        f"[dsa] native {path} kernel disabled for this process after error "
        f"(inline fallback active): {exc!r}",
        file=sys.stderr,
    )


# Arch-default prefill chunk once the flash profile is armed (4096 measured
# +8% prefill over 2048 at 200k depth on V4-Flash, decode unchanged; the
# armed transient stays ~0.8 GB there). GMLX_V4_PREFILL_STEP overrides in
# either direction; an explicit PREFILL_STEP_SIZE (flag/config/env) always
# wins over both.
_V4_BASE_STEP: Optional[int] = 4096


def _v4_base_step() -> Optional[int]:
    env = os.environ.get("GMLX_V4_PREFILL_STEP")
    if env:
        try:
            n = int(env)
            return n if n > 0 else None
        except ValueError:
            return None
    return _V4_BASE_STEP


# Score-transient profile for prefill_decay. With every native DSA kernel
# armed, single-sequence prefill never materializes the dense [heads, step, S]
# scores; the transient peaks at the indexer's [1, step, S/ratio] fp32 block
# (heads collapse in-kernel). Quantized pools keep the profile: at prefill
# (L > 4) the pooled property dequantizes on read and the kernels run
# unchanged, and the per-layer fp16 pool copy is step-independent, so the fit
# test correctly ignores it (dense-modeled decay would floor the chunk to 256
# and multiply that same dequant across ~16x more chunks). Disarms (dense
# decay stays authoritative) on kernels off, batched caches, or a quantized
# local KV.
_prefill_score_profile = _prefill_decay.build_score_profile(
    profile=lambda: _prefill_decay.ScoreTransientProfile(
        heads=1, bytes_per_elem=4, depth_divisor=4,
        base_step=_v4_base_step()),
    kernels_armed=lambda: all(
        _dsa_probe(p) for p in ("window", "sparse", "indexer")),
    require_cache=PoolingCache,
    disarm_cache=BatchPoolingCache,
    allow_quantized_pools=True,
)


_prefill_decay.register_score_profile("deepseek_v4", _prefill_score_profile)


def _cacheless_pool_mask(n_pooled: int, L: int, offset, ratio: int):
    """Causal pooled-visibility mask for the cache-less (training/eval)
    forward, mirroring PoolingCache.make_mask: query at absolute position
    offset + j sees pooled row i iff i < (offset + j) // ratio."""
    if n_pooled == 0 or L == 1 or not isinstance(offset, int):
        return None
    pool_idx = mx.arange(n_pooled)
    query_idx = mx.arange(offset + 1, offset + L + 1)
    return pool_idx < query_idx[:, None] // ratio


def _pair_split(fused_call, n_a: int, x: mx.array):
    y = fused_call(x)
    return y[..., :n_a], y[..., n_a:]


def _fuse_kquant_pair(a, b, in_dims: int):
    """One KQuantLinear serving a's rows then b's; None when ineligible."""
    try:
        from mlx_kquant.nn import KQuantLinear
    except Exception:
        return None
    if not (isinstance(a, KQuantLinear) and isinstance(b, KQuantLinear)):
        return None
    if (
        a.kquant_type != b.kquant_type
        or "bias" in a
        or "bias" in b
        or a.weight.ndim != 2
        or b.weight.ndim != 2
        or a.weight.shape[1] != b.weight.shape[1]
    ):
        return None
    n_a, n_b = a.weight.shape[0], b.weight.shape[0]
    fused = KQuantLinear(in_dims, n_a + n_b, bias=False, codec=a.kquant_type)
    fused.weight = mx.concatenate([a.weight, b.weight], axis=0)
    fused.scales = a.scales
    mx.eval(fused.weight)
    # Rebind the pair as views of the fused buffer; the originals release.
    a.weight = fused.weight[:n_a]
    b.weight = fused.weight[n_a:]
    return fused


def _fuse_float_pair(a, b):
    """One nn.Linear serving a's rows then b's; None when ineligible."""
    if type(a) is not nn.Linear or type(b) is not nn.Linear:
        return None
    if "bias" in a or "bias" in b:
        return None
    wa, wb = a.weight, b.weight
    if wa.dtype != wb.dtype or wa.ndim != 2 or wa.shape[1] != wb.shape[1]:
        return None
    n_a = wa.shape[0]
    fused = nn.Linear(wa.shape[1], n_a + wb.shape[0], bias=False)
    fused.weight = mx.concatenate([wa, wb], axis=0)
    mx.eval(fused.weight)
    a.weight = fused.weight[:n_a]
    b.weight = fused.weight[n_a:]
    return fused


def install_gemv_row_fusion(model, max_float_rows: int = 1024) -> int:
    """Fuse same-input projection pairs into single GEMV dispatches.

    Each pair's weight rows are concatenated into one buffer post-load and
    the originals rebound as slice views, so one matmul serves both outputs;
    row results are exact (independent dot products per output row). Pairs:
    attention wq_a+wkv on every layer, and compressor wkv+wgate when the
    concat stays within ``max_float_rows`` (wider float concats lose to two
    dispatches under MLX's gemv kernel selection, so the ratio-4 attention
    compressor stays split). The shared expert pair is skipped: fusing it
    materializes its wire bytes for negligible return. Returns the number
    of pairs fused. GMLX_GEMV_FUSE=0 disables."""
    if os.environ.get("GMLX_GEMV_FUSE", "1") == "0":
        return 0
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    n = 0
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue
        wq_a = getattr(attn, "wq_a", None)
        wkv = getattr(attn, "wkv", None)
        if wq_a is not None and wkv is not None:
            fused = _fuse_kquant_pair(wq_a, wkv, attn.hidden_size)
            if fused is None and (
                getattr(wq_a.weight, "ndim", 0) == 2
                and getattr(wkv.weight, "ndim", 0) == 2
                and wq_a.weight.shape[0] + wkv.weight.shape[0]
                <= max_float_rows
            ):
                # Float-weight targets (tiny test models): same fusion via
                # nn.Linear, under the float row cap.
                fused = _fuse_float_pair(wq_a, wkv)
            if fused is not None:
                attn._qa_kv_fused = fused
                attn._qa_rows = wq_a.weight.shape[0]
                n += 1
        for comp in (
            getattr(attn, "compressor", None),
            getattr(getattr(attn, "indexer", None), "compressor", None),
        ):
            if comp is None:
                continue
            if 2 * comp.out_dim > max_float_rows:
                continue
            fused = _fuse_float_pair(comp.wkv, comp.wgate)
            if fused is not None:
                comp._kv_gate_fused = fused
                n += 1
    return n


def ensure_registered() -> None:
    """Make ``import mlx_lm.models.deepseek_v4`` resolve, preferring
    upstream, and expose the pooling caches on ``mlx_lm.models.cache``
    (name-based cache-class resolution and isinstance checks in shared
    code paths look them up there; since mlx-vlm 0.6.4 vendored its own
    models/cache.py, its shared paths resolve there instead, so register
    on both when it is loaded)."""
    import mlx_lm.models.cache as _mlx_cache

    vlm_cache = sys.modules.get("mlx_vlm.models.cache")
    for mod in (_mlx_cache, vlm_cache):
        if mod is not None and not hasattr(mod, "PoolingCache"):  # upstream wins
            mod.PoolingCache = PoolingCache
            mod.BatchPoolingCache = BatchPoolingCache
    if "mlx_lm.models.deepseek_v4" in sys.modules:
        return
    try:
        importlib.import_module("mlx_lm.models.deepseek_v4")  # upstream wins
    except ImportError:
        sys.modules["mlx_lm.models.deepseek_v4"] = sys.modules[__name__]


_CACHE_EVAL_EVERY = int(os.environ.get("GMLX_CACHE_EVAL_EVERY", "1"))
_cache_eval_step = 0


def _materialize_cache_arrays(cache: Optional[Any]) -> None:
    """Detach DeepSeek-V4 cache update graphs from prior decode steps."""
    if cache is None:
        return

    global _cache_eval_step
    _cache_eval_step += 1
    if _CACHE_EVAL_EVERY == 0 or _cache_eval_step % _CACHE_EVAL_EVERY:
        return

    cache_arrays = []
    for layer_cache in cache:
        if layer_cache is None:
            continue
        leaves = getattr(layer_cache, "caches", None) or (layer_cache,)
        for leaf in leaves:
            if leaf is None:
                continue
            for value in vars(leaf).values():
                if isinstance(value, mx.array):
                    cache_arrays.append(value)

    if cache_arrays:
        mx.async_eval(*cache_arrays)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if not self.compress_ratios:
            n = self.num_hidden_layers
            self.compress_ratios = (
                [0]
                + [4 if i % 2 else 128 for i in range(max(n - 2, 0))]
                + ([0] if n >= 2 else [])
            )
        self.compress_ratios = list(self.compress_ratios[: self.num_hidden_layers])
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError(
                "`compress_ratios` must have one entry per hidden layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers."
            )
        bad = [r for r in self.compress_ratios if r not in (0, 4, 128)]
        if bad:
            raise ValueError(f"Unsupported DeepSeek-V4 compress ratios: {bad}")


def make_quantization_config(model):
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8 for k, _ in flat_modules if ".attn.w" in k or ".attn.indexer.wq" in k
    }
    # MTP fusion projections (oMLX MTP patch attaches them as mtp.<i>.e_proj /
    # mtp.<i>.h_proj). The fp8 checkpoint ships them as e4m3 weight + e8m0
    # block scale, i.e. mxfp8 after sanitize. Without an explicit entry they
    # fall through to the affine default, whose QuantizedLinear expects a
    # .biases tensor the checkpoint doesn't have, and strict load fails.
    mtp_projs = {
        k: mxfp8
        for k, _ in flat_modules
        if k.startswith("mtp.") and (k.endswith(".e_proj") or k.endswith(".h_proj"))
    }

    return {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
        **mtp_projs,
    }


# QAT simulation round-trips (deviation from PR 1192; ds4.c:2440-2620).
#
# All helpers work on |values| and reapply the sign, mirroring the C
# reference. Rounding is round-half-to-even in VALUE-INDEX space, which
# mx.round provides directly because representable values within an octave
# are uniformly spaced and even index parity == even quotient parity.
# Scales are exact powers of two looked up from a table (every entry is an
# exactly-representable float32, so this matches the original ldexpf
# exponent-bit construction bit-for-bit -- pinned by the bit-identity tests
# in test_deepseek_v4_qat.py -- while staying mx.compile-traceable; the
# .view() form it replaces was untraceable and kept the whole per-step
# fp8/fp4 chain out of mx.compile).

_EXP2_TABLE = mx.array([2.0**i for i in range(-126, 128)], dtype=mx.float32)


def _exp2i(e: mx.array) -> mx.array:
    """2**e for integer-valued float e in [-126, 127], exactly."""
    return mx.take(_EXP2_TABLE, e.astype(mx.int32) + 126)


def _e4m3_round(v: mx.array) -> mx.array:
    """Round v (|v| <= 448) to the nearest FP8-E4M3FN value, ties to even
    mantissa. Subnormal step 2**-9 below 2**-6; normal step 2**(e-3)."""
    s = mx.sign(v)
    a = mx.abs(v)
    e = mx.floor(mx.log2(mx.maximum(a, 2.0**-9)))
    e = mx.clip(e, -6.0, 8.0)
    q = _exp2i(e - 3.0)
    return s * mx.round(a / q) * q


@partial(mx.compile, shapeless=True)
def _e2m1_round(v: mx.array) -> mx.array:
    """Round v (|v| <= 6) to the nearest FP4-E2M1 value
    {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even value index
    (ds4.c dsv4_e2m1fn_dequant tie-break)."""
    s = mx.sign(v)
    a = mx.abs(v)
    q = mx.where(
        a <= 0.25, 0.0,
        mx.where(a < 0.75, 0.5,
        mx.where(a <= 1.25, 1.0,
        mx.where(a < 1.75, 1.5,
        mx.where(a <= 2.5, 2.0,
        mx.where(a < 3.5, 3.0,
        mx.where(a <= 5.0, 4.0, 6.0)))))))
    return s * q


@partial(mx.compile, shapeless=True)
def _fp8_block_core(v: mx.array) -> mx.array:
    """Compiled body of the fp8 round-trip on the [..., block] view (axis=-1
    reduction + elementwise chain: shape-polymorphic, fuses per step)."""
    amax = mx.maximum(mx.max(mx.abs(v), axis=-1, keepdims=True), 1e-4)
    scale = _exp2i(mx.ceil(mx.log2(amax / 448.0)))
    return _e4m3_round(mx.clip(v / scale, -448.0, 448.0)) * scale


@partial(mx.compile, shapeless=True)
def _fp4_block_core(v: mx.array) -> mx.array:
    """Compiled body of the fp4 round-trip on the [..., block] view."""
    amax = mx.maximum(
        mx.max(mx.abs(v), axis=-1, keepdims=True), 7.052966104933725e-38
    )
    scale = _exp2i(mx.ceil(mx.log2(amax / 6.0)))
    return _e2m1_round(mx.clip(v / scale, -6.0, 6.0)) * scale


def _fp8_e4m3_roundtrip(x: mx.array, block: int = 64) -> mx.array:
    """Per-block E4M3 round-trip: scale = 2**ceil(log2(amax/448)) with an
    amax floor of 1e-4, clamp to +-448, nearest-even quantize, rescale.
    (ds4.c dsv4_fp8_kv_quantize_row_inplace_cpu)"""
    orig_dtype = x.dtype
    if x.shape[-1] % block:
        block = x.shape[-1]  # tiny test configs; real rows are 64-aligned
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, block))
    return mx.flatten(_fp8_block_core(v), -2).astype(orig_dtype)


def _fp4_e2m1_roundtrip(x: mx.array, block: int = 32) -> mx.array:
    """Per-block E2M1 round-trip: scale = 2**ceil(log2(amax/6)) with an
    amax floor of FLT_MIN*6 (2**-126 scale floor), clamp to +-6.
    (ds4.c dsv4_fp4_act_quantize_row_inplace_cpu)"""
    orig_dtype = x.dtype
    if x.shape[-1] % block:
        block = x.shape[-1]  # tiny test configs; real rows are 32-aligned
    v = mx.unflatten(x.astype(mx.float32), -1, (-1, block))
    return mx.flatten(_fp4_block_core(v), -2).astype(orig_dtype)


def _kv_qat_roundtrip(kv: mx.array, n_rot: int) -> mx.array:
    """Main-attention KV row round-trip, applied after RoPE and before the
    cache update: FP8 on the leading non-RoPE dims, FP16 on the whole row.
    (ds4.c forward site: dsv4_fp8_kv_quantize_row + f16_round)"""
    if (kv.shape[-1] - n_rot) % 64 == 0 and kv.shape[-1] > n_rot and _dsa_probe(
        "kv_qat"
    ):
        # Fused kernel, bit-identical to the chain below (pinned by
        # mlx-kquant's test_dsa_kv_qat bit-identity suite).
        try:
            import mlx_kquant as kq

            return kq.dsa_kv_qat(kv, n_rot)
        except Exception as exc:  # noqa: BLE001 - permanent fallback
            _dsa_disable("kv_qat", exc)
    orig_dtype = kv.dtype
    nope, rot = mx.split(kv, [kv.shape[-1] - n_rot], axis=-1)
    kv = mx.concatenate([_fp8_e4m3_roundtrip(nope), rot], axis=-1)
    return kv.astype(mx.float16).astype(orig_dtype)


def _indexer_qat_roundtrip(x: mx.array) -> mx.array:
    """Indexer activation round-trip: 128-wide Hadamard (scale 1/sqrt(128),
    mx.hadamard_transform's default) then the FP4 round-trip. Applies to
    indexer queries and indexer compressed-pool rows; the top-k selection
    is not the model's graph without it. (ds4.c dsv4_indexer_qat_row)"""
    if x.shape[-1] == 128 and _dsa_probe("qat"):
        # Fused kernel, bit-identical to the chain below (pinned by
        # mlx-kquant's test_dsa_qat bit-identity suite).
        try:
            import mlx_kquant as kq

            return kq.dsa_indexer_qat(x)
        except Exception as exc:  # noqa: BLE001 - permanent fallback
            _dsa_disable("qat", exc)
    orig_dtype = x.dtype
    v = mx.hadamard_transform(x.astype(mx.float32))
    return _fp4_e2m1_roundtrip(v).astype(orig_dtype)


_pool_grid_certified = False


def _pool_rows_on_grid(pooled: mx.array) -> bool:
    """One-time (per process) certificate that cached indexer pool rows are
    a fixed point of the FP4 round-trip, i.e. exactly expressible as
    codes+scales. The quantized-operand score path is lossless ONLY under
    this precondition; it holds by construction for deepseek4's trained
    indexer QAT, and this check guards a future family member whose QAT
    differs -- on failure the path disarms permanently and the fp16 kernel
    takes over. Value-level compare (signed zeros compare equal)."""
    global _pool_grid_certified
    if _pool_grid_certified:
        return True
    ok = bool(mx.array_equal(_fp4_e2m1_roundtrip(pooled), pooled))
    _pool_grid_certified = ok
    return ok


def warm_kernel_pipelines() -> int:
    """Dispatch each armed indexer kernel once at load, off the request path.

    Moves every one-time first-dispatch cost (Metal pipeline builds,
    ~15 ms measured per process) out of the first request's TTFT, and
    gives cold-start A/Bs a control: GMLX_DSA_WARM=0 restores
    first-dispatch-on-request on an identical build. Pipeline identity is
    dtype+flag-keyed, not shape-keyed, so tiny dummy operands compile
    exactly the runtime variants (fp16 operands, causal=False, bucketed
    top-k). Failures are swallowed: the warm is an optimization and the
    per-call probes stay the arming authority. Returns the number of
    warmed dispatch groups."""
    if os.environ.get("GMLX_DSA_WARM", "1") == "0":
        return 0
    if not _dsa_probe("indexer"):
        return 0
    import mlx_kquant as kq

    n = 0

    def fire(fn):
        nonlocal n
        try:
            out = fn()
            mx.eval(*(out if isinstance(out, (tuple, list)) else (out,)))
            n += 1
        except Exception:  # noqa: BLE001 - warm only, probes stay live
            pass

    B, H, L, D, P = 1, 64, 64, 128, 1024
    q = mx.zeros((B, H, L, D), dtype=mx.float16)
    keys = mx.zeros((B, 1, P, D), dtype=mx.float16)
    w = mx.zeros((B, L, H), dtype=mx.float16)
    if _dsa_probe("qat"):
        fire(lambda: kq.dsa_indexer_qat(q))
    fire(lambda: kq.dsa_topk_indices(
        kq.dsa_indexer_scores(q, keys, w, causal=False), 512, bucketed=True))
    if hasattr(kq, "dsa_indexer_score_decode"):
        fire(lambda: kq.dsa_indexer_score_decode(
            mx.zeros((B, H, 1, D), dtype=mx.float16),
            mx.zeros((B, P, D), dtype=mx.float16),
            mx.zeros((B, 1, H), dtype=mx.float16),
            P * 4,
            4,
        ))
    if _dsa_probe("indexer_q"):
        def quant_chain():
            qc, qs = kq.dsa_indexer_qat_quant(q)
            kc, ks = kq.dsa_indexer_qat_pack(keys)
            return kq.dsa_indexer_scores_q(qc, qs, kc, ks, w, causal=False)

        fire(quant_chain)
    return n


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported DeepSeek-V4 scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _hash_expert_select(
    input_ids: mx.array,
    logits: mx.array,
    tid2eid: mx.array,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    inds = tid2eid[input_ids]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV4RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale

        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")

        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001

            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")

        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(
        self,
        x: mx.array,
        offset: Any = 0,
        inverse: bool = False,
    ) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


def _apply_score_mask(scores: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return scores
    if mask.dtype == mx.bool_:
        return mx.where(mask, scores, mx.finfo(scores.dtype).min)
    return scores + mask.astype(scores.dtype)


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    if mask is None:
        return None

    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape

    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))

    full_mask = mx.concatenate([mask, pool_mask], axis=-1)

    return full_mask


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, ape, head_dim):
    weights = mx.softmax(gate.astype(mx.float32) + ape, axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)


@mx.compile
def _overlap_compress_kv(kv, gate, ape, head_dim):
    B, L, R, D = kv.shape

    gate = gate + ape.astype(gate.dtype)

    kv_0 = mx.zeros((B, 1, R, D // 2), dtype=kv.dtype)
    kv_a, kv_b = mx.split(kv, 2, axis=-1)
    kv_a = mx.concatenate([kv_0, kv_a[:, :-1]], axis=1)
    kv = mx.concatenate([kv_a, kv_b], axis=2)

    gate_0 = mx.full((B, 1, R, D // 2), -mx.inf, dtype=kv.dtype)
    gate_a, gate_b = mx.split(gate, 2, axis=-1)
    gate_a = mx.concatenate([gate_0, gate_a[:, :-1]], axis=1)
    gate = mx.concatenate([gate_a, gate_b], axis=2)

    weights = mx.softmax(gate, axis=-2, precise=True)
    return (kv * weights).sum(axis=-2)


@partial(mx.compile, shapeless=True)
def _split_softmax(log_normalizer, logits_a, logits_b, sinks=None):
    if sinks is not None:
        log_normalizer = mx.logaddexp(log_normalizer, sinks)
    weights_a = mx.exp(logits_a - log_normalizer)
    weights_b = mx.exp(logits_b - log_normalizer)
    return weights_a, weights_b


def _sparse_topk_gather(pooled: mx.array, topk: mx.array, L: int, D: int):
    B = pooled.shape[0]
    idx = topk[:, None, :, :, None]
    return mx.take_along_axis(
        mx.broadcast_to(pooled[:, None, None], (B, 1, L, pooled.shape[1], D)),
        mx.broadcast_to(idx, idx.shape[:-1] + (D,)),
        axis=3,
    )


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    L, D = q.shape[2], q.shape[3]
    pooled = _sparse_topk_gather(pooled, topk, L, D)
    return _sparse_gathered_attention(
        q, local_kv, pooled, local_mask, pooled_mask, scale, sinks
    )


def _sparse_gathered_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    local_mask: Optional[mx.array],
    pooled_mask: Optional[mx.array],
    scale: float,
    sinks: Optional[mx.array],
) -> mx.array:
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = _apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    pooled_sq = pooled.squeeze(1)
    q_bl = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = q_bl @ pooled_sq.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = _apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer, mx.logsumexp(pooled_scores, -1, keepdims=True)
    )

    local_weights, pooled_weights = _split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )

    out = local_weights @ local_kv
    pw_bl = pooled_weights.transpose(0, 2, 1, 3)
    out = out + (pw_bl @ pooled_sq).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


# Compiled decode variant: collapses the ~40-node matmul/logsumexp chain
# into one walk node per layer (the C++ graph walk+encode is the decode
# pacer). Decode-only: the ring-full local window and the fixed index_topk
# keep shapes steady. The top-k gather stays eager - it is the only op
# that sees the raw pooled length, which steps every compress_ratio tokens
# and would force a ~5 ms retrace every 4th token at ratio 4.
_sparse_gathered_attention_c = mx.compile(_sparse_gathered_attention)
_COMPILE_SPARSE = os.environ.get("GMLX_COMPILE_SPARSE", "1") != "0"


def _dense_sinks_attention(q, kv, sinks, scale):
    return mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=scale, mask=None, sinks=sinks
    )


def _dense_sinks_attention_pooled(q, kv, pooled, sinks, scale):
    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
    return mx.fast.scaled_dot_product_attention(
        q, full_kv, full_kv, scale=scale, mask=None, sinks=sinks
    )


def _dense_sinks_attention_masked(q, kv, mask, sinks, scale):
    return mx.fast.scaled_dot_product_attention(
        q, kv, kv, scale=scale, mask=mask, sinks=sinks
    )


def _dense_sinks_attention_pooled_masked(q, kv, pooled, mask, sinks, scale):
    full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
    return mx.fast.scaled_dot_product_attention(
        q, full_kv, full_kv, scale=scale, mask=mask, sinks=sinks
    )


# Compiled decode variants for the window (+ pooled) layers: hd512 + sinks
# routes to mlx's fallback graph (matmul / concat sinks / softmax / slice),
# whose movement glue folds into the compiled segments. Tracing the stock
# call keeps the op recipe identical. Gated on a full ring (constant S);
# the pooled length steps every compress_ratio tokens, so those retraces
# are shared across layers via these module-level fns. The masked variants
# cover the MTP verify widths (qL 2-4): the mask is an array argument, not
# a trace key, so each L adds one trace at the ring-full S.
_dense_sinks_attention_c = mx.compile(_dense_sinks_attention)
_dense_sinks_attention_pooled_c = mx.compile(_dense_sinks_attention_pooled)
_dense_sinks_attention_masked_c = mx.compile(_dense_sinks_attention_masked)
_dense_sinks_attention_pooled_masked_c = mx.compile(
    _dense_sinks_attention_pooled_masked
)
_COMPILE_DENSE = os.environ.get("GMLX_COMPILE_DENSE", "1") != "0"
_MOE_MIX_SCORES = os.environ.get("GMLX_V4_MOE_MIX", "1") != "0"


def _kernel_window_attention(module, q, kv, pooled, sinks, offset, ratio):
    """Sliding-window (+ causally visible pooled) attention via the DSA
    sparse kernel (mlx-kquant), replacing masked-dense SDPA that pays full
    L x S quadratic work for a sliding_window-wide live band.

    With an identity top-k over the pooled rows, the kernel's causal clamp
    (offset + pos + 1) // ratio reproduces PoolingCache.make_mask exactly,
    so no mask is materialized. pooled=None serves window-only layers: one
    dummy pooled row that a 2**30 ratio keeps invisible at any reachable
    offset. Returns [B, 64, qL, 512], or None to fall back."""
    B, H, L, D = q.shape
    if H != 64 or D != 512 or kv.shape[1] != 1 or kv.shape[2] < L:
        return None
    # One threadgroup per query position underfills the GPU at decode
    # widths, same as the sparse-branch kernel: prefill only.
    min_l = int(os.environ.get("GMLX_DSA_WINDOW_MIN_L", "64"))
    if L < min_l:
        return None
    try:
        import mlx_kquant as kq

        if pooled is None:
            pooled = mx.zeros((B, 1, D), dtype=q.dtype)
            topk = mx.zeros((B, 1, L, 1), dtype=mx.uint32)
            ratio = 1 << 30
        else:
            P = pooled.shape[1]
            topk = mx.broadcast_to(
                mx.arange(P, dtype=mx.uint32)[None, None, None],
                (B, 1, L, P),
            )
        return kq.dsa_sparse_attention(
            q,
            kv,
            pooled,
            topk,
            sinks,
            module.scale,
            offset,
            ratio,
            module.config.sliding_window,
        ).astype(q.dtype)
    except Exception as exc:  # noqa: BLE001 - permanent fallback
        _dsa_disable("window", exc)
        return None


_KQ_ROUTER_STATE = {"ok": None}


def _kq_router_available() -> bool:
    """One-time probe for the mlx-kquant router kernel's sqrtsoftplus arm
    (feature-sniffed from the docstring so older kquant builds fall back)."""
    ok = _KQ_ROUTER_STATE["ok"]
    if ok is None:
        ok = os.environ.get("GMLX_V4_KQ_ROUTER", "1") != "0"
        if ok:
            try:
                import mlx_kquant as kq

                ok = mx.metal.is_available() and "sqrtsoftplus" in (
                    getattr(kq.moe_router_topk, "__doc__", "") or ""
                )
            except Exception:  # noqa: BLE001 - optional dependency
                ok = False
        _KQ_ROUTER_STATE["ok"] = ok
    return ok


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.hash = layer_idx < config.num_hash_layers
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        if self.hash:
            self.tid2eid = mx.zeros((config.vocab_size, self.top_k), dtype=mx.int32)
        else:
            self.e_score_correction_bias = mx.zeros(
                (self.num_experts,), dtype=mx.float32
            )

    def __call__(self, x: mx.array, input_ids: Optional[mx.array] = None):
        logits = x @ self.weight.T

        if self.hash:
            if input_ids is None:
                raise ValueError("DeepSeek-V4 hash routing requires input_ids.")
            inds, weights = _hash_expert_select(
                input_ids,
                logits,
                self.tid2eid,
                self.routed_scaling_factor,
                self.norm_topk_prob,
                self.scoring_func,
            )
        else:
            t = logits.size // self.num_experts
            if (
                self.scoring_func == "sqrtsoftplus"
                and self.norm_topk_prob
                and t * self.top_k < 64
                and not self.training
                and _kq_router_available()
            ):
                import mlx_kquant as kq

                inds, weights = kq.moe_router_topk(
                    logits.reshape(t, self.num_experts),
                    self.top_k,
                    True,
                    shared_gate=False,
                    bias=self.e_score_correction_bias,
                    scoring="sqrtsoftplus",
                    scale=self.routed_scaling_factor,
                )
                shp = logits.shape[:-1]
                inds = inds.reshape(*shp, self.top_k)
                weights = weights.reshape(*shp, self.top_k)
            else:
                inds, weights = _expert_select(
                    logits,
                    self.e_score_correction_bias,
                    self.top_k,
                    self.routed_scaling_factor,
                    self.norm_topk_prob,
                    self.scoring_func,
                )

        return inds, weights


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV4MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
        )
        self.sharding_group = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x, input_ids)
        if _MOE_MIX_SCORES and getattr(self.switch_mlp, "_kq_mix_scores", False):
            # Fused arm folds the score-weighted sum into the down gather
            # (one dispatch, no [..., k, H] intermediate); the wrapper
            # applies the sum itself whenever the fused path is ineligible.
            y = self.switch_mlp(x, inds, scores)
        else:
            y = self.switch_mlp(x, inds)
            if y.ndim == scores.ndim + 1:
                y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Compressor(nn.Module):

    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self._kv_gate_fused = None
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )
        # QAT round-trip for freshly pooled rows (ds4.c compressor site):
        # attention compressor rows get the FP8 treatment on the non-RoPE
        # dims; indexer compressor rows get Hadamard-128 + FP4 on the whole
        # row. Dispatch on head_dim, mirroring the C reference.
        if head_dim == config.head_dim:
            self._qat = "fp8"
        elif head_dim == config.index_head_dim:
            self._qat = "fp4"
        else:
            self._qat = None

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
        fetch: bool = True,
    ) -> mx.array:
        B, _, _ = x.shape
        if self._kv_gate_fused is not None:
            kv, gate = _pair_split(self._kv_gate_fused, self.out_dim, x)
        else:
            kv = self.wkv(x)
            gate = self.wgate(x)
        if pool_cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            compress_func = (
                _overlap_compress_kv if self.overlap else _simple_compress_kv
            )
            kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
            new_pooled = compress_func(kv, gate, self.ape, self.head_dim)
            new_pooled = self.norm(new_pooled)
            new_pooled = self.rope(
                new_pooled[:, None],
                offset=pool_base,
            ).squeeze(1)
            if self._qat == "fp8":
                new_pooled = mx.concatenate(
                    [
                        _fp8_e4m3_roundtrip(
                            new_pooled[..., : self.head_dim - self.rope_head_dim]
                        ),
                        new_pooled[..., self.head_dim - self.rope_head_dim :],
                    ],
                    axis=-1,
                )
            elif self._qat == "fp4":
                new_pooled = _indexer_qat_roundtrip(new_pooled)
            if self.overlap and pool_cache is not None:
                # accumulate_windows prepended the previous completed window
                # so the kernel's cross-window shift links this call's first
                # window to its true predecessor; the prepended window's own
                # pooled row is a recompute of one already in the pool.
                new_pooled = new_pooled[:, 1:]

        if pool_cache is not None:
            if fetch:
                new_pooled = pool_cache.update_and_fetch(new_pooled)
            else:
                # Sparse decode under quantized storage: append only; the
                # caller reads back its top-k rows via gather_pooled instead
                # of a full-pool dequantize.
                if new_pooled.shape[1] > 0:
                    pool_cache.append_pooled(new_pooled)
                new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)

        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config: ModelArgs, compress_ratio: int):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        q_residual: mx.array,
        position_rope: DeepseekV4RoPE,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
    ):
        B, L, _ = x.shape
        pooled = self.compressor(x, pool_cache, offset)
        if pooled.shape[1] == 0:
            return None

        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = position_rope(q, offset)
        # Quantized-operand prefill arm: emit codes+scales from the same QAT
        # core before the fp16 round-trip (bit-consistent by kernel contract:
        # qat_quant(x) == qat_pack(qat(x))). Prefill widths only; decode and
        # every fallback path keep consuming the round-tripped fp16 q.
        q_quant = None
        if L > 4 and _dsa_probe("indexer") and _dsa_probe("indexer_q"):
            try:
                import mlx_kquant as kq

                q_quant = kq.dsa_indexer_qat_quant(q)
            except Exception as exc:  # noqa: BLE001 - permanent fallback
                _dsa_disable("indexer_q", exc)
        q = _indexer_qat_roundtrip(q)

        pmask = (
            pool_cache.make_mask(L, offset)
            if pool_cache is not None
            else _cacheless_pool_mask(
                pooled.shape[1], L, offset, self.compressor.compress_ratio
            )
        )
        k = min(self.index_topk, pooled.shape[1])

        if k == self.index_topk and _dsa_probe("indexer"):
            topk = self._kernel_topk(
                x, q, pooled, pmask, k,
                offset if pool_cache is not None else None,
                q_quant=q_quant,
            )
            if topk is not None:
                return topk

        scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(
            mx.float32
        )
        scores = mx.maximum(scores, 0) * self.scale
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
        if pmask is not None:
            scores = mx.where(
                pmask if pmask.ndim == 3 else pmask[None],
                scores,
                mx.finfo(scores.dtype).min,
            )
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]

    def _kernel_topk(
        self,
        x: mx.array,
        q: mx.array,
        pooled: mx.array,
        pmask: Optional[mx.array],
        k: int,
        offset: Optional[int] = None,
        q_quant: Optional[tuple] = None,
    ) -> Optional[mx.array]:
        """Native indexer scores + radix top-k (mlx-kquant).

        Selection-equivalent to the inline path: self.scale is dropped
        (top-k is invariant under a positive factor) and n_heads**-0.5
        folds into the per-head weights. Decode widths (L <= 4) take the
        fused direct-score kernel, which reproduces make_mask(L, offset)
        from (offset, ratio) instead of applying pmask; offset is None
        when there is no pool cache, keeping that arm off. Prefill widths
        pad pooled keys -- and, for unaligned chunk tails, the queries and
        per-token weights -- to a 64 multiple for the GEMM and slice the
        scores back to [L, P] before selection. When ``q_quant`` carries the
        QAT codes+scales and the indexer_q path is armed, scoring runs the
        int8 tensor-op kernel on quantized operands (bit-equal scores,
        certified by the pool grid check); any failure falls back to the
        fp16 kernel, then inline. Returns [B, L, k] uint32, or None to fall
        back."""
        B, H, L, D = q.shape
        P = pooled.shape[1]
        if H not in (32, 64) or D != 128 or k not in (512, 2048) or P < k:
            return None
        if (
            L <= 4
            and H == 64
            and isinstance(offset, int)
            # qL >= 3 is occupancy-starved on small pools and measured
            # slower than the inline path below 2048 rows.
            and (L <= 2 or P >= 2048)
            and os.environ.get("GMLX_DSA_INDEXER_DECODE", "1") != "0"
        ):
            try:
                import mlx_kquant as kq

                if hasattr(kq, "dsa_indexer_score_decode"):
                    weights = self.weights_proj(x) * (self.n_heads**-0.5)
                    scores = kq.dsa_indexer_score_decode(
                        q,
                        pooled,
                        weights,
                        offset,
                        self.compressor.compress_ratio,
                    )
                    return kq.dsa_topk_indices(scores, k, bucketed=True)[:, 0]
            except Exception as exc:  # noqa: BLE001 - permanent fallback
                _dsa_disable("indexer", exc)
                return None
        # Prefill widths only below this point; the decode arm above covers
        # L <= 4 and pad-to-64 measured slower than inline at those widths.
        # Unaligned widths pad the queries to the kernel's 64 multiple and
        # slice back: on a deep pool the inline fallback materializes a
        # [heads, L, P] fp32 score block, so at P >= 2048 the kernel arms
        # for any width above the decode arm; below that the inline path is
        # cheap and keeps its measured edge under min_l.
        min_l = int(os.environ.get("GMLX_DSA_INDEXER_MIN_L", "64"))
        if L <= 4 or (L < min_l and P < 2048):
            return None
        try:
            import mlx_kquant as kq

            weights = self.weights_proj(x) * (self.n_heads**-0.5)
            keys = pooled[:, None]
            pad_n = (-P) % 64
            if pad_n:
                keys = mx.concatenate(
                    [keys, mx.zeros((B, 1, pad_n, D), dtype=keys.dtype)],
                    axis=2,
                )
            pad_l = (-L) % 64
            q_in = q
            if pad_l:
                q_in = mx.concatenate(
                    [q, mx.zeros((B, H, pad_l, D), dtype=q.dtype)], axis=2
                )
                weights = mx.concatenate(
                    [weights,
                     mx.zeros((B, pad_l, H), dtype=weights.dtype)],
                    axis=1,
                )
            scores = None
            if q_quant is not None and _dsa_probe("indexer_q"):
                # Quantized-operand arm: exact int8 tensor-op MMA over the
                # codes the model's own QAT already produced. Lossless only
                # while pool rows sit on the FP4 grid -- certified once per
                # process, disarmed permanently otherwise. Zero-padding is
                # exact (zero codes score zero).
                try:
                    if not _pool_rows_on_grid(pooled):
                        raise ValueError(
                            "indexer pool rows are not FP4-grid fixed points"
                        )
                    qc, qs = q_quant
                    if pad_l:
                        qc = mx.concatenate(
                            [qc, mx.zeros((B, H, pad_l, D), dtype=qc.dtype)],
                            axis=2,
                        )
                        qs = mx.concatenate(
                            [qs,
                             mx.zeros((B, H, pad_l, qs.shape[-1]),
                                      dtype=qs.dtype)],
                            axis=2,
                        )
                    kc, ks = kq.dsa_indexer_qat_pack(keys)
                    scores = kq.dsa_indexer_scores_q(
                        qc, qs, kc, ks, weights, causal=False
                    )
                except Exception as exc:  # noqa: BLE001 - permanent fallback
                    _dsa_disable("indexer_q", exc)
                    scores = None
            if scores is None:
                scores = kq.dsa_indexer_scores(q_in, keys, weights,
                                               causal=False)
            if pad_n or pad_l:
                scores = scores[..., :L, :P]
            if pmask is not None:
                pm = pmask if pmask.ndim == 3 else pmask[None]
                scores = mx.where(
                    pm[:, None], scores, mx.finfo(scores.dtype).min
                )
            return kq.dsa_topk_indices(scores, k, bucketed=True)[:, 0]
        except Exception as exc:  # noqa: BLE001 - permanent fallback
            _dsa_disable("indexer", exc)
            return None


class LocalAttention(nn.Module):
    """DeepSeek V4 attention with no KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = 0
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._qa_kv_fused = None
        self._qa_rows = 0
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.rope_theta,
            None,
            config.max_position_embeddings,
        )

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        if self._qa_kv_fused is not None:
            qa_out, kv_out = _pair_split(self._qa_kv_fused, self._qa_rows, x)
        else:
            qa_out, kv_out = self.wq_a(x), self.wkv(x)
        q = self.wq_b(self.q_norm(qa_out))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(kv_out).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = _kv_qat_roundtrip(kv, self.config.qk_rope_head_dim)
        if cache is not None:
            kv, _ = cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        sinks = self.attn_sink.astype(q.dtype)
        out = None
        if _dsa_probe("window") and isinstance(offset, int):
            out = _kernel_window_attention(self, q, kv, None, sinks, offset, 0)
        if out is None and _COMPILE_DENSE and L <= 4:
            if mask is None and kv.shape[2] == self.config.sliding_window:
                out = _dense_sinks_attention_c(q, kv, sinks, self.scale)
            elif (
                mask is not None
                and kv.shape[2] == self.config.sliding_window + L - 1
            ):
                # Verify widths: ring-full concat fetch (constant S per L).
                out = _dense_sinks_attention_masked_c(
                    q, kv, mask, sinks, self.scale
                )
        if out is None:
            out = scaled_dot_product_attention(
                q,
                kv,
                kv,
                cache=cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class CompressedAttention(nn.Module):
    """DeepSeek V4 attention with pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._qa_kv_fused = None
        self._qa_rows = 0
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        # Compressed layers use Yarn-scaled RoPE
        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        pool_cache = cache[1] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        if self._qa_kv_fused is not None:
            qa_out, kv_out = _pair_split(self._qa_kv_fused, self._qa_rows, x)
        else:
            qa_out, kv_out = self.wq_a(x), self.wkv(x)
        q = self.wq_b(self.q_norm(qa_out))
        q = q.reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(kv_out).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = _kv_qat_roundtrip(kv, self.config.qk_rope_head_dim)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        # Pool tokens into compressed KV and concatenate with local KV
        pooled = self.compressor(x, pool_cache, offset)
        sinks = self.attn_sink.astype(q.dtype)

        out = None
        if _dsa_probe("window") and isinstance(offset, int):
            out = _kernel_window_attention(
                self,
                q,
                kv,
                pooled if pooled.shape[1] > 0 else None,
                sinks,
                offset,
                self.compress_ratio,
            )
        if out is None and _COMPILE_DENSE and L <= 4:
            if mask is None and kv.shape[2] == self.config.sliding_window:
                # mask None means the pooled visibility mask is dead
                # (_extend_mask returns None), so the compiled route skips
                # building it.
                out = (
                    _dense_sinks_attention_pooled_c(
                        q, kv, pooled, sinks, self.scale
                    )
                    if pooled.shape[1] > 0
                    else _dense_sinks_attention_c(q, kv, sinks, self.scale)
                )
            elif (
                mask is not None
                and kv.shape[2] == self.config.sliding_window + L - 1
            ):
                # Verify widths: masks stay eager (the pooled visibility
                # mask sees the raw pooled length); the sdpa glue compiles.
                if pooled.shape[1] > 0:
                    pooled_mask = (
                        pool_cache.make_mask(L, offset)
                        if pool_cache is not None
                        else None
                    )
                    mask_ext = _extend_mask(
                        mask, pooled_mask, kv.shape[2] + pooled.shape[1]
                    )
                    out = _dense_sinks_attention_pooled_masked_c(
                        q, kv, pooled, mask_ext, sinks, self.scale
                    )
                else:
                    out = _dense_sinks_attention_masked_c(
                        q, kv, mask, sinks, self.scale
                    )
        if out is None:
            pooled_mask = None
            if pooled.shape[1] > 0:
                pooled_mask = (
                    pool_cache.make_mask(L, offset)
                    if pool_cache is not None
                    else None
                )
                kv = mx.concatenate([kv, pooled[:, None]], axis=2)

            mask = _extend_mask(mask, pooled_mask, kv.shape[2])

            out = scaled_dot_product_attention(
                q,
                kv,
                kv,
                cache=local_cache,
                scale=self.scale,
                mask=mask,
                sinks=sinks,
            )
        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class SparseCompressedAttention(nn.Module):
    """DeepSeek V4 attention with sparse indexed pooled KV compression."""

    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.scale = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._qa_kv_fused = None
        self._qa_rows = 0
        self.wo_a = MultiLinear(
            self.n_heads * self.head_dim // config.o_groups,
            config.o_lora_rank,
            config.o_groups,
        )
        self.wo_b = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)

        self.rope = DeepseekV4RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        self.indexer = Indexer(config, self.compress_ratio)

        self.sharding_group = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        local_cache = cache[0] if cache is not None else None
        comp_cache = cache[1] if cache is not None else None
        idx_cache = cache[2] if cache is not None else None
        offset = local_cache.offset if local_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        if self._qa_kv_fused is not None:
            qa_out, kv_out = _pair_split(self._qa_kv_fused, self._qa_rows, x)
        else:
            qa_out, kv_out = self.wq_a(x), self.wkv(x)
        q_residual = self.q_norm(qa_out)
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)
        q = self.rope(q, offset)

        kv = self.kv_norm(kv_out).reshape(B, 1, L, self.head_dim)
        kv = self.rope(kv, offset)
        kv = _kv_qat_roundtrip(kv, self.config.qk_rope_head_dim)
        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, mx.zeros((B, 1, L, 0)))

        # Quantized pooled storage (--kv-bits): decode-width steps append
        # without a dense fetch and gather-dequantize only their top-k rows;
        # prefill widths still fetch the full pool (kernel path input).
        fetch = not (
            comp_cache is not None and comp_cache.is_quantized and L <= 4
        )
        pooled = self.compressor(x, comp_cache, offset, fetch=fetch)
        plen = pooled.shape[1] if fetch else comp_cache.size()
        pmask = (
            comp_cache.make_mask(L, offset)
            if comp_cache is not None
            else _cacheless_pool_mask(plen, L, offset, self.compress_ratio)
        )
        topk = self.indexer(x, q_residual, self.rope, idx_cache, offset)
        sinks = self.attn_sink.astype(q.dtype)

        # Local attention
        if plen == 0:
            out = None
            if _dsa_probe("window") and isinstance(offset, int):
                out = _kernel_window_attention(
                    self, q, kv, None, sinks, offset, 0
                )
            if out is None:
                out = scaled_dot_product_attention(
                    q,
                    kv,
                    kv,
                    cache=local_cache,
                    scale=self.scale,
                    mask=mask,
                    sinks=sinks,
                )

        # Compressed attention
        elif plen <= self.indexer.index_topk:
            if not fetch:
                pooled = comp_cache.pooled
            out = None
            if _dsa_probe("window") and isinstance(offset, int):
                out = _kernel_window_attention(
                    self, q, kv, pooled, sinks, offset, self.compress_ratio
                )
            if out is None:
                full_kv = mx.concatenate([kv, pooled[:, None]], axis=2)
                mask = _extend_mask(mask, pmask, full_kv.shape[2])
                out = scaled_dot_product_attention(
                    q,
                    full_kv,
                    full_kv,
                    cache=local_cache,
                    scale=self.scale,
                    mask=mask,
                    sinks=sinks,
                )

        # Sparse compressed attention
        else:
            out = None
            if fetch and _dsa_probe("sparse") and isinstance(offset, int):
                out = self._kernel_sparse_attention(
                    q, kv, pooled, topk, sinks, offset
                )
            if out is None:
                sparse_mask = None
                if pmask is not None:
                    sparse_mask = mx.take_along_axis(
                        pmask[None] if pmask.ndim == 2 else pmask,
                        topk,
                        axis=2,
                    )[:, None]
                if L <= 4 and (_COMPILE_SPARSE or not fetch):
                    # Offset stays out of the compiled call (a python-int
                    # arg would retrace every step); the gather runs eager
                    # so the core's trace never sees the growing pooled
                    # length. The window mask (verify widths) is an array
                    # argument -- one extra trace per L, not per step.
                    gathered = (
                        comp_cache.gather_pooled(topk)
                        if not fetch
                        else _sparse_topk_gather(pooled, topk, L, self.head_dim)
                    )
                    attn_fn = (
                        _sparse_gathered_attention_c
                        if _COMPILE_SPARSE
                        else _sparse_gathered_attention
                    )
                    out = attn_fn(
                        q, kv, gathered, mask, sparse_mask, self.scale, sinks
                    )
                else:
                    out = _sparse_pooled_attention(
                        q,
                        kv,
                        pooled,
                        topk,
                        mask,
                        sparse_mask,
                        self.scale,
                        sinks,
                    )

        out = self.rope(out, offset, inverse=True)

        out = out.reshape(B, self.o_groups, -1, L, self.head_dim)
        out = out.transpose(0, 1, 3, 2, 4).flatten(-2)
        out = self.wo_a(out)
        out = out.transpose(0, 2, 1, 3).flatten(-2)
        out = self.wo_b(out)

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    def _kernel_sparse_attention(
        self,
        q: mx.array,
        kv: mx.array,
        pooled: mx.array,
        topk: mx.array,
        sinks: mx.array,
        offset: int,
    ) -> Optional[mx.array]:
        """Native window + gathered-pooled + sinks attention (mlx-kquant).

        One dispatch replaces _sparse_pooled_attention plus both masks: the
        kernel derives the local window from (localL, qL, local_window) and
        its causal pooled clamp matches the pool cache's visibility mask.
        Returns [B, 64, qL, 512], or None to fall back."""
        B, H, L, D = q.shape
        if H != 64 or D != 512 or kv.shape[1] != 1 or kv.shape[2] < L:
            return None
        # The kernel launches one threadgroup per (query position, batch),
        # so narrow qL underfills the GPU; decode measured slower than the
        # inline path. Prefill widths only until a KV-split decode variant.
        min_l = int(os.environ.get("GMLX_DSA_SPARSE_MIN_L", "64"))
        if L < min_l:
            return None
        try:
            import mlx_kquant as kq

            return kq.dsa_sparse_attention(
                q,
                kv,
                pooled,
                topk[:, None].astype(mx.uint32),
                sinks,
                self.scale,
                offset,
                self.compress_ratio,
                self.config.sliding_window,
            ).astype(q.dtype)
        except Exception as exc:  # noqa: BLE001 - permanent fallback
            _dsa_disable("sparse", exc)
            return None


def v4_attention_factory(config: ModelArgs, layer_idx: int) -> nn.Module:
    """Instantiate the appropriate attention module for a given layer."""
    ratio = config.compress_ratios[layer_idx]
    if ratio == 0:
        return LocalAttention(config, layer_idx)
    if ratio == 128:
        return CompressedAttention(config, layer_idx)
    return SparseCompressedAttention(config, layer_idx)


class DeepseekV4Block(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = v4_attention_factory(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        input_ids: mx.array,
    ) -> mx.array:
        residual = h
        x, post, comb = self.attn_hc(h)
        x = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        h = hc_expand(x, residual, post, comb)

        residual = h
        x, post, comb = self.ffn_hc(h)
        x = self.ffn(self.ffn_norm(x), input_ids)
        return hc_expand(x, residual, post, comb)


class DeepseekV4Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV4Block(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        return_raw_hidden: bool = False,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        from .cache_compat import cache_types

        cache_list_types = cache_types("CacheList")
        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, cache_list_types)
            else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        _materialize_cache_arrays(cache)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, cache_list_types):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # MTP seam (omlx mlx_lm_mtp parity): the drafter head fuses on the raw
        # pre-hc_head 4D hidden (B, L, hc_mult, hidden), so the verify path
        # needs it alongside the collapsed+normed output.
        if return_raw_hidden:
            return self.norm(self.hc_head(h)), h
        return self.norm(self.hc_head(h))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None):
        return self.lm_head(self.model(inputs, cache))

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        # Plain caches take the consuming stack's class identities: under
        # mlx-vlm serve its apc/batch machinery isinstance-gates on its own
        # (vendored since 0.6.4) classes; under the pure mlx-lm CLI, on
        # mlx-lm's.
        from .cache_compat import construction_cache_module

        cmod = construction_cache_module()
        caches = []
        for layer in self.layers:
            ratio = layer.attn.compress_ratio
            if ratio == 0:
                caches.append(
                    cmod.RotatingKVCache(max_size=self.args.sliding_window))
            elif isinstance(layer.attn, SparseCompressedAttention):
                # local + compressor pool + indexer pool. The indexer pool
                # opts out of --kv-bits packing: the score kernel reads it
                # in full every step (see PoolingCache.quantizable).
                idx_pool = PoolingCache(ratio)
                idx_pool.quantizable = False
                caches.append(
                    cmod.CacheList(
                        cmod.RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                        idx_pool,
                    )
                )
            else:
                # local + compressor pool
                caches.append(
                    cmod.CacheList(
                        cmod.RotatingKVCache(max_size=self.args.sliding_window),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers

        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        for key, value in list(weights.items()):
            if (
                ".ffn.switch_mlp." not in key
                or not key.endswith((".scales", ".biases"))
                or value.dtype != mx.bfloat16
            ):
                continue
            stem = key.rsplit(".", 1)[0]
            if (
                stem + ".weight" in weights
                and stem + ".scales" in weights
                and stem + ".biases" in weights
                and weights[stem + ".weight"].dtype == mx.uint32
            ):
                weights[key] = value.astype(mx.float16)

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all layers
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.attn.sharding_group = group
            layer.attn.wq_b = shard_linear(
                layer.attn.wq_b,
                "all-to-sharded",
                segments=self.args.o_groups,
                group=group,
            )
            shard_inplace(layer.attn.wo_a, "sharded-to-all", group=group)
            layer.attn.attn_sink = mx.split(layer.attn.attn_sink, N)[rank]
            layer.attn.n_heads //= N

            layer.ffn.sharding_group = group
            shard_inplace(
                layer.ffn.shared_experts.gate_proj, "all-to-sharded", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.down_proj, "sharded-to-all", group=group
            )
            shard_inplace(
                layer.ffn.shared_experts.up_proj, "all-to-sharded", group=group
            )
            shard_inplace(layer.ffn.switch_mlp.gate_proj, "all-to-sharded", group=group)
            shard_inplace(layer.ffn.switch_mlp.down_proj, "sharded-to-all", group=group)
            shard_inplace(layer.ffn.switch_mlp.up_proj, "all-to-sharded", group=group)
