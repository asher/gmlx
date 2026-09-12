"""GGUF K-quant -> in-memory mlx-lm model load pipeline.

``load_model`` is the public entry point: it loads any text-only GGUF (the
K-quant, legacy, and IQ codec families) directly into a stock mlx-lm ``class
Model`` with the
quantized leaves swapped for ``KQuant*`` modules, gated only on the GGUF arch
having an ``mlx_lm/models/<arch>.py``. No safetensors round-trip, no conversion.

The 8-step flow (preflight+load -> remap -> build -> patch -> sanitize ->
swap -> load_weights -> tokenizer) keeps per-arch behaviour entirely in the
``remap``/``config_synth``/``tokenizer`` modules, so this file stays
arch-generic.
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import mlx_kquant as kq

from . import loadlog
from .dtypes import activation_dtype, activation_dtype_name
from gmlx.envflags import env_bool, env_choice, env_int
from gmlx.upstream.attn_hd512 import install_hd512_sdpa
from gmlx.gen.prefill_decay import install_prefill_decay, note_untracked_weights
import gmlx.upstream.gpt_oss_prefill as gpt_oss_prefill  # noqa: F401  (registers gpt_oss score profile)
from .modules import install_fused_moe_glu, install_hyv3_shexp_fold
from gmlx.upstream.occupancy_fuse import install_occupancy_fuse
from gmlx.upstream.qkv_fuse import install_fused_qkv
from gmlx.models.gemma4.batched_sdpa import install_gemma4_batched_sdpa
from gmlx.upstream.cascade_sdpa import install_cascade_sdpa, install_cascade_stamp
from gmlx.upstream.sparse_sdpa import install_sparse_sdpa
from gmlx.upstream.arrays_cache_fix import install_arrays_cache_fix
from gmlx.models.gemma4.sync import install_gemma4_nosync
from gmlx.upstream.softcap_f32 import install_gemma4_softcap_f32
from gmlx.upstream.quantized_cache_pack_fix import (
    install_quantized_cache_pack_fix)
from gmlx.upstream.quantized_sdpa_fix import install_quantized_sdpa_mask_fix
from gmlx.upstream.rope_batch_fix import install_rope_batch_fix
from gmlx.upstream.rotating_cache_fix import install_rotating_cache_fix
from .modules import (
    KQuantEmbedding,
    dequantize_unattachable_leaves,
    install_kquant_modules,
)
from .populate import (
    maybe_populate_for_load,
    start_populate,
    wait_for as wait_for_populate,
)
from .preflight import preflight
from gmlx.upstream.dsv32_patches import (
    _patch_dsv32_dense_default,
    _patch_dsv32_indexer_fp32,
    _patch_dsv32_indexer_rope,
    _patch_dsv32_mask_decode,
    _patch_dsv32_moe_gate_fp32,
    _patch_dsv32_moe_scores,
)
from gmlx.upstream.gdn_patches import (
    _needs_tiled_v_patch,
    _patch_gated_delta_fused_decode,
    _patch_gated_delta_tiled_v,
    _patch_qwen3next_split_gdn,
    _tiled_v_patch_applied,
)
from .gguf_meta import first_nonzero_int, read_int
from .mtp_target import _build_mtp_target
from .transforms import coalesce_split_experts, fuse_shexp_gate_up
from .wire import load_gguf_wire_bytes, remap_arrays, strip_nextn_trunk_overflow




# Model construction (bypassing nn.quantize)


def build_model(config_dict: dict, *, mtp: bool = False):
    """Instantiate the mlx_lm model class without running ``nn.quantize()``.

    The caller supplies a flat HF-shaped config dict with ``model_type`` set to
    the canonical text-model class. The ``_UNWRAP_TO_TEXT`` shortcut builds the
    inner ``TextModel`` directly for wrapper models (e.g. qwen3_5_moe), avoiding
    a ``language_model.`` prefix mismatch.

    When ``mtp=True`` the target is built on the mlx-vlm ``LanguageModel`` class
    (which carries the ``speculative_*`` hooks) wrapped in ``MTPTextTarget``; the
    stock mlx-lm text path is left byte-identical for every non-MTP load.
    """
    if mtp:
        return _build_mtp_target(config_dict)

    from mlx_lm.utils import _get_classes

    config = dict(config_dict)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    _UNWRAP_TO_TEXT = {"qwen3_5_moe", "qwen3_5_moe_text"}
    mt = config.get("model_type", "")
    if mt in _UNWRAP_TO_TEXT:
        import importlib

        mod = importlib.import_module("mlx_lm.models.qwen3_5")
        TextModel = mod.TextModel
        TextModelArgs = mod.TextModelArgs
        model_args = TextModelArgs.from_dict(config)
        model = TextModel(model_args)
        loadlog.verbose_print(
            f"[build] unwrap {mt} -> qwen3_5.TextModel (avoid language_model. prefix)"
        )
        return model, config
    if mt == "gemma3n":
        # gemma-3n's mlx_lm text tower is a standalone `LanguageModel` taking a
        # `TextConfig`. Build it directly (the synthesized config nests the text
        # fields under `text_config`) so the weight keys are the unprefixed
        # `LanguageModel` attribute paths the remap targets - no `model.` /
        # `language_model.` wrapper prefix. LanguageModel.__call__ already
        # produces logits (tied embed_tokens.as_linear + softcap) and carries
        # make_cache, so no outer Model wrapper is needed.
        import importlib

        mod = importlib.import_module("mlx_lm.models.gemma3n")
        text_config = mod.TextConfig.from_dict(config["text_config"])
        model = mod.LanguageModel(text_config)
        loadlog.verbose_print(f"[build] {mt} -> gemma3n.LanguageModel (text tower, no prefix)")
        return model, config
    if mt == "diffusion_gemma":
        # DiffusionGemma's model class lives in mlx-vlm, not mlx-lm: it's a
        # non-autoregressive encoder-decoder block-diffusion model. The synthesized
        # config nests the Gemma-4 MoE backbone under `text_config`; the full
        # `Model` (encoder + decoder + self-conditioning) builds from the nested
        # ModelConfig and exposes `config.canvas_length`, which is what mlx-vlm's
        # generate dispatch keys on to route into the diffusion denoiser. The remap
        # already targets the `model.decoder.*` / `model.encoder.*` tree this class
        # expects, so no target prefix is applied.
        import importlib

        dg = importlib.import_module("mlx_vlm.models.diffusion_gemma")
        model = dg.Model(dg.ModelConfig.from_dict(config))
        loadlog.verbose_print(
            f"[build] {mt} -> mlx-vlm diffusion_gemma.Model "
            "(block-diffusion encoder-decoder)"
        )
        return model, config
    if mt == "gemma_embedding":
        # EmbeddingGemma: a gemma3 text backbone run as a bidirectional sentence
        # encoder. The model class lives in mlx-embeddings, not mlx-lm; its
        # ModelArgs is mlx-lm gemma3_text's (reused). The remap already targets the
        # encoder Model's tree (model.* backbone + dense.0/dense.1), so no prefix.
        import importlib

        ge = importlib.import_module("mlx_embeddings.models.gemma3_text")
        model = ge.Model(ge.ModelArgs.from_dict(config))
        loadlog.verbose_print(
            f"[build] {mt} -> mlx-embeddings gemma3_text.Model "
            "(bidirectional encoder)"
        )
        return model, config
    if mt == "minimax_m3":
        # mlx-lm ships no minimax_m3 module yet (PR #1401 unmerged); register
        # the vendored copy into the mlx_lm.models namespace so _get_classes
        # (and every other importer) resolves it. Upstream wins if present.
        import gmlx.models.minimax_m3 as minimax_m3_model

        minimax_m3_model.ensure_registered()
    if mt == "deepseek_v4":
        # mlx-lm ships no deepseek_v4 module yet (PR #1192 unmerged); same
        # vendored-registration pattern as minimax_m3, plus PoolingCache
        # injection into mlx_lm.models.cache.
        import gmlx.models.deepseek_v4.model as deepseek_v4_model

        deepseek_v4_model.ensure_registered()
    if mt == "hy_v3":
        # mlx-lm ships no hy_v3 module yet (PR #1485 unmerged); same vendored-
        # registration pattern as minimax_m3. The tool parser registers with
        # the model so a later serve template-inference resolves it.
        import gmlx.models.hy_v3.model as hy_v3_model
        import gmlx.models.hy_v3.tools as hy_v3_tools

        hy_v3_model.ensure_registered()
        hy_v3_tools.ensure_registered()
    if mt == "kimi_k3":
        # mlx-lm ships no kimi_k3 module (llama.cpp PR #26185 arch); same
        # vendored-registration pattern as minimax_m3.
        import gmlx.models.kimi_k3 as kimi_k3_model

        kimi_k3_model.ensure_registered()
    if mt == "glm5_next":
        # mlx-lm ships no glm5_next module (llama.cpp PR #27754 arch); same
        # vendored-registration pattern as kimi_k3, plus the deepseek_v4
        # PoolingCache injection its hybrid cache depends on.
        import gmlx.models.glm5_next.model as glm5_next_model

        glm5_next_model.ensure_registered()
    if mt == "qwen4_exp":
        # Neither pinned mlx-lm nor mlx-vlm ships qwen4_exp (llama.cpp PR
        # #27742); same vendored-registration pattern as deepseek_v4, plus
        # QSAKVCache injection into the cache modules.
        import gmlx.models.qwen4_exp.model as qwen4_exp_model

        qwen4_exp_model.ensure_registered()
    if mt == "muse_glimmer":
        # mlx-lm ships no muse_glimmer module (afmoe is the nearest relative);
        # same vendored-registration pattern as kimi_k3. The tool parser
        # registers with the model so a later serve template-inference
        # resolves it.
        import gmlx.models.muse_glimmer.model as muse_glimmer_model
        import gmlx.models.muse_glimmer.tools as muse_glimmer_tools

        muse_glimmer_model.ensure_registered()
        muse_glimmer_tools.ensure_registered()
    if mt == "hy_v4":
        # mlx-lm ships no hy_v4 module (llama.cpp LLM_ARCH_HYV4); same
        # vendored-registration pattern as glm5_next. The tool parser
        # registers with the model so a later serve template-inference
        # resolves it.
        import gmlx.models.hy_v4.model as hy_v4_model
        import gmlx.models.hy_v4.tools as hy_v4_tools

        hy_v4_model.ensure_registered()
        hy_v4_tools.ensure_registered()
    Model, ModelArgs = _get_classes(config)
    model_args = ModelArgs.from_dict(config)
    model = Model(model_args)
    return model, config


def _patch_hunyuan_norm_topk(model) -> None:
    """Renormalize Hunyuan's top-k router scores (``norm_topk_prob``).

    The HF Hunyuan reference and llama.cpp's hunyuan-moe graph
    (``build_moe_ffn(..., norm_topk_prob=true, ...)``) both rescale the
    selected top-k softmax probabilities to sum to 1 before weighting the
    expert outputs. mlx-lm's ``hunyuan.MoeBlock`` skips that rescale -
    upstream gap - which under-weights the routed branch on every MoE layer
    and degenerates the model outright (single-token spam from the first
    step on Hunyuan-A13B, 64 experts top-8).

    Per-instance ``__class__`` swap (no mlx-lm globals touched), forward
    byte-identical to stock apart from the one-line renormalization.
    """
    from mlx_lm.models.hunyuan import MoeBlock

    class _NormTopKMoE(MoeBlock):
        def __call__(self, x):
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            k = self.top_k
            inds = mx.stop_gradient(
                mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
            )
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            if (
                getattr(self, "_kq_expert_mass", None) is not None
                or getattr(self, "_kq_expert_probe", None) is not None
            ):
                from gmlx.stream.moe_experts import _apply_expert_controls

                inds, scores = _apply_expert_controls(self, inds, scores)
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
            if self.use_shared_mlp:
                y = y + self.shared_mlp(x)
            return y

    n = 0
    for m in model.modules():
        if type(m) is MoeBlock:
            m.__class__ = _NormTopKMoE
            n += 1
    if n:
        loadlog.verbose_print(f"[patch] hunyuan: norm_topk_prob router rescale on {n} MoE layers")



# Tokens-per-call at or above which an offloaded expert forward runs on the
# GPU stream (prefill regime). Decode calls (1-few tokens) stay on CPU.
_STREAM_GPU_TOKENS_DEFAULT = 32
# Streaming-mode expert calls at or above this many tokens are treated as
# prefill by the sequential-prefetch hook (see gmlx.stream.prefetch).
_STREAM_PREFETCH_MIN_TOKENS = 32


def _stream_gpu_tokens(default: int = _STREAM_GPU_TOKENS_DEFAULT) -> int:
    return env_int("GMLX_STREAM_GPU_TOKENS", default)


def _arena_stage_max_tokens() -> int:
    """Largest expert call served router-aware (decode-feeder arena, or
    partial ring staging). Above this, a chunk routes to nearly every expert
    and whole-layer staging wins back its pipelining; below it, reading only
    the routed slices is the smaller IO."""
    return env_int("GMLX_ARENA_STAGE_MAX_TOKENS", 64)


def _arena_split_max_tokens() -> int:
    """Largest expert call the arena serves by token-splitting when its
    routed union exceeds the arena's slots (a turn-transition prefill or a
    wide verify batch). Halves recurse until each piece's union fits, so
    every read stays on the wired arena's read pool instead of the CPU
    page-cache gather. 0 disables. Above the cap, the whole-layer prefill
    paths win back their sequential pipelining."""
    return env_int("GMLX_ARENA_SPLIT_MAX_TOKENS", 256)










def _resolve_feeder_defaults(
    feeder_prefill: bool | None, feeder_decode: bool | None
) -> tuple[bool, bool]:
    """Feeder policy for streaming models. Explicit caller intent (a CLI
    flag) wins; then the ``GMLX_FEEDER_*`` env vars (A/B levers); then the
    defaults: prefill feeder on everywhere it can exist, decode feeder on
    only when the every-token layers are on the GPU (``--stream-experts``) -
    under ``--stream-cpu`` there is no GPU work for the arena gathers to
    join, so it is not even attempted. config.ResolvedModel.load_signature
    mirrors this resolution to canonicalize the residency cache key; keep
    the two in step."""
    gpu_resident = "gpu" in str(mx.default_device()).lower()
    if feeder_prefill is None:
        feeder_prefill = env_bool("GMLX_FEEDER_PREFILL", True)
    if feeder_decode is None:
        feeder_decode = env_bool("GMLX_FEEDER_DECODE", gpu_resident)
    return feeder_prefill, feeder_decode




def _kq_expert_gpu_ok(module) -> bool:
    """False when any expert projection's codec lacks Metal matmul kernels
    (``kq.codec_has_matmul``): its gathers must stay off the GPU stream, so
    the module is excluded from feeder/arena coverage and GPU prefill
    routing. Older mlx-kquant builds without the capability query only ship
    GPU-capable codecs - default True."""
    has = getattr(kq, "codec_has_matmul", None)
    if has is None:
        return True
    for name in ("gate_proj", "up_proj", "down_proj"):
        codec = getattr(getattr(module, name, None), "kquant_type", None)
        if codec is not None and not has(codec):
            return False
    return True


# Decode phase stats (GMLX_DECODE_PHASE_STATS=1): wall seconds per wrapper
# phase, per decode token, dumped at exit. Buckets: ev = the router eval
# (GPU segment for the previous layer's gather plus this layer's every-token
# work, plus the host sync); la = lookahead router replica; stage_wait =
# demand-read join inside stage(); stage_book = stage() minus the join;
# prestage = speculative-read submission; build = gather graph build + slot
# upload. resid = per-token wall (first-MoE-layer to first-MoE-layer) minus
# the buckets: head matmul eval, sampling, detokenize, serve-loop glue.
# First token after any non-decode call is skipped (prefill contamination);
# its buckets still land in the sums, a <=1/N skew.
_PHASE_KEYS = ("ev", "la", "stage_wait", "stage_book", "prestage", "build")
_PHASE = (
    {k: 0.0 for k in _PHASE_KEYS}
    | {"tokens": 0, "wall": 0.0, "last": None, "first_li": None, "dirty": True}
    if env_bool("GMLX_DECODE_PHASE_STATS", False)
    else None
)


def _phase_token(ph, li, n_tokens):
    """Token-boundary bookkeeping: a decode call on the first covered MoE
    layer opens a new token; the previous boundary-to-boundary wall lands in
    the per-token average unless a non-decode call dirtied the window."""
    now = time.perf_counter()
    if n_tokens != 1:
        ph["dirty"] = True
        return
    if li is None:
        return
    if ph["first_li"] is None:
        ph["first_li"] = li
    if li != ph["first_li"]:
        return
    last = ph["last"]
    ph["last"] = now
    if last is not None and not ph["dirty"]:
        ph["tokens"] += 1
        ph["wall"] += now - last
    ph["dirty"] = False


def _phase_dump():
    ph = _PHASE
    if not ph or not ph["tokens"]:
        return
    n = ph["tokens"]
    ms = {k: 1e3 * ph[k] / n for k in _PHASE_KEYS}
    tot = 1e3 * ph["wall"] / n
    print(
        f"[phase] decode per-token ms over {n} tokens: total {tot:.1f} | "
        + " ".join(f"{k} {v:.1f}" for k, v in ms.items())
        + f" | resid {tot - sum(ms.values()):.1f}",
        flush=True,
    )
    try:
        from gmlx.stream.lookahead import _LA_PHASE as lap
    except Exception:
        lap = None
    if lap is not None:
        b, s = 1e3 * lap["build"] / n, 1e3 * lap["sync"] / n
        print(
            f"[phase] la split: build {b:.1f} | sync {s:.1f} | "
            f"post {ms['la'] - b - s:.1f}",
            flush=True,
        )


if _PHASE is not None:
    import atexit

    atexit.register(_phase_dump)


# Families where lookahead prestage defaults OFF: the replica router's
# per-layer sync tax measured above its stall savings there (see the
# la_default comment in install_expert_streaming).
_LA_DEFAULT_OFF_FAMILIES = frozenset({"glm_moe_dsa", "deepseek_v32"})


def _lookahead_default(model) -> bool:
    """Family default for GMLX_DECODE_LOOKAHEAD when the env is unset."""
    return getattr(
        model, "model_type", None) not in _LA_DEFAULT_OFF_FAMILIES






# Default prefill chunk width on streaming-mode (over-wired-budget) models.
# Every prefill chunk re-streams ~the whole expert lane from disk (no
# page-cache retention between chunks at realistic sizes), so prefill wall
# time ~= compute + n_chunks x lane-stream and bigger chunks win almost
# linearly until the compute floor: 8192 measured ~2.2x over the stock 2048
# at 8k/16k prompts on a 162 GB MoE (M5 Max 128 GB), at ~+1.6 GB transient
# memory. 16384 adds ~8% at 16k+ prompts for another ~+2 GB of transients -
# worth passing explicitly on big-RAM boxes, too tight to default on small
# ones. In-RAM models keep mlx-lm's own 2048 default, which is fastest there
# (see the _PREFILL_CHUNK note: the two prefill engines tune differently).
_STREAMING_PREFILL_STEP = 8192

# Streaming models whose per-chunk activation footprint argues for a
# narrower default than _STREAMING_PREFILL_STEP.
#   hy_v4 - iHC carries hc_mult (4) parallel residual streams and the
#   reference computes the collapse in fp32, so one [1, step, 4, 6144]
#   fp32 transient is 805 MB at 8192 and the front builds several per
#   layer across 78 layers. Half the step halves every one of them. Raise
#   it back with --prefill-step-size on a large-RAM box.
_STREAMING_PREFILL_STEP_BY_MODEL_TYPE: dict[str, int] = {
    "hy_v4": 4096,
}


def moe_streaming_active(model) -> bool:
    """True when ``install_expert_streaming`` put this model in streaming mode
    (weights exceed the GPU wired budget; expert bytes stream from disk)."""
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None) or ()
    return any(
        getattr(m, "_kq_cpu_only", False) for layer in layers for m in layer.modules()
    )




def _switch_num_experts(glu) -> int:
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(glu, name, None)
        if proj is None:
            continue
        n = getattr(proj, "num_experts", None)
        if n is not None:
            return int(n)
        w = getattr(proj, "weight", None)
        if w is not None:
            return int(w.shape[0])
    return 0


def _decoder_layers(model):
    """Transformer layers of a text model or a VLM wrapper, or ()."""
    for owner in (model, getattr(model, "language_model", None)):
        if owner is None:
            continue
        layers = getattr(owner, "layers", None)
        if layers is None:
            layers = getattr(getattr(owner, "model", None), "layers", None)
        if layers:
            return layers
    return ()


def model_is_moe(model) -> bool:
    """True when the model routes tokens through stacked expert MLPs.

    Structural on purpose. The synthesized config spells the expert count four
    ways across arches (``num_experts``, ``num_local_experts``,
    ``moe_num_experts``, ``n_routed_experts``), all from one GGUF field, so a
    key lookup fails OPEN on the next arch that picks a fifth: it would report
    an MoE target as dense. Every routed block instead reaches its forward
    through a switch GLU whose projections carry a stacked ``[experts, out, in]``
    weight, or a ``num_experts`` attribute on the quantized/K-quant variants.

    ``_switch_num_experts`` is NOT a usable test here: on a dense MLP it returns
    ``weight.shape[0]``, the output width, not 0.
    """
    for layer in _decoder_layers(model):
        for m in layer.modules():
            # fc1/fc2: mlx-lm's plain SwitchMLP naming (nemotron_h_moe).
            for name in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
                proj = getattr(m, name, None)
                if proj is None:
                    continue
                if getattr(proj, "num_experts", None) is not None:
                    return True
                w = getattr(proj, "weight", None)
                if w is not None and getattr(w, "ndim", 0) == 3:
                    return True
    return False




class _FactoredRoPE(nn.Module):
    """RoPE with per-dimension frequency factors from a GGUF ``rope_freqs``
    tensor. llama.cpp rotates at angle ``pos / (base^(2i/d) * factor_i)``,
    which is ``mx.fast.rope`` with ``freqs = base^(2i/d) * factors``."""

    def __init__(self, dims: int, base: float, traditional: bool,
                 scale: float, factors: mx.array):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.scale = scale
        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = freqs * factors.astype(mx.float32)

    def __call__(self, x, offset=0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=self.scale,
            offset=offset,
            freqs=self._freqs,
        )


# llama.cpp converts Llama-3.1/3.2's "llama3" rope scaling into a per-dim
# factors tensor (rope_freqs.weight) instead of scaling KV, so the synthesized
# config carries no rope_scaling and the stock rope is unscaled - coherent
# below the ~8k original context, degenerate beyond it. Rebuild each attention
# rope from the factors; exact by construction for any factors tensor.
# Kill with GMLX_ROPE_FACTORS=0.
def _patch_rope_factors(model, factors: mx.array) -> None:
    """Swap every plain ``nn.RoPE`` whose width matches ``factors`` for a
    ``_FactoredRoPE`` built from the same dims/base/scale plus the factors."""
    if not env_bool("GMLX_ROPE_FACTORS", True):
        return
    if bool(mx.all(factors == 1.0)):
        return
    n = 0
    for m in model.modules():
        r = getattr(m, "rope", None)
        if type(r) is nn.RoPE and r.dims == 2 * factors.size:
            m.rope = _FactoredRoPE(
                r.dims, r.base, r.traditional, r.scale, factors)
            n += 1
    if n:
        loadlog.verbose_print(
            f"[patch] rope_freqs factors applied on {n} layers")


# Reporting


def print_inventory(
    arch: str,
    kquant_meta: dict[str, str],
    hf_kquant_meta: dict[str, str],
    stats: dict[str, int],
) -> None:
    from collections import Counter

    print(f"[gmlx] load plan: arch={arch!r}")
    print(f"  quantized tensors (GGUF names)  : {len(kquant_meta)}")
    print(f"  quantized tensors (model names) : {len(hf_kquant_meta)}")
    remapped = ", ".join(f"{k}={v}" for k, v in stats.items() if v)
    print(f"  name remap: {remapped or 'none'}")
    hist = Counter(hf_kquant_meta.values())
    print("  codec histogram (after remap):")
    for codec, n in sorted(hist.items()):
        print(f"    {codec:5s} {n}")


# Shared swap+load back-half (steps 5-7), reused by the two-GGUF VLM loader


def _verify_zero_copy_views(model, no_alias, log) -> None:
    """Post-load donation tripwire over the zero-copy GGUF mapping.

    Any param whose buffer sits inside a live mapping must keep its wire dtype
    (integer reinterprets allowed), and arithmetic-transform results
    (``no_alias``) must own their buffers. A violation is the signature of MLX
    buffer donation into the file mapping: the donated write is dropped on
    read-only shared maps, leaving stale wire bytes typed as the new dtype
    (garbage weights from token 0). Metadata-only and O(#params), so it runs
    on every load; GMLX_VERIFY_VIEWS=0 disables.
    """
    if os.environ.get("GMLX_VERIFY_VIEWS", "1") == "0":
        return
    verify = getattr(kq, "verify_zero_copy_views", None)
    if verify is None or kq.zero_copy_view_count() == 0:
        return
    named = tree_flatten(model.parameters())
    mx.eval([a for _, a in named])
    problems = verify(named, sorted(no_alias or ()))
    if problems:
        shown = "\n  ".join(problems[:12])
        more = f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""
        raise RuntimeError(
            f"[verify] zero-copy view integrity check failed for "
            f"{len(problems)} params (buffer donation through the mapping?):"
            f"\n  {shown}{more}"
        )
    log(f"[verify] zero-copy views OK ({len(named)} params)")


def materialize_module_arrays(*modules) -> None:
    """Evaluate every array in the module trees, including the non-parameter
    attributes ``parameters()`` skips (precomputed RoPE ``_freqs`` and the
    like). MLX default streams are per-thread: an array a load thread leaves
    lazy is bound to a stream no other thread can evaluate, and the first
    forward elsewhere dies with "There is no Stream(gpu, N) in current
    thread". Loader entry points call this last so a model may load on one
    thread and generate on another (chat background load, server
    preload/keep-warm). Cheap: weights are already evaluated, only small
    stragglers remain, and zero-copy mmap views evaluate without paging."""
    mx.eval(list(modules))


def _warm_touch_threshold_bytes() -> int:
    """Size above which the first forward risks the Metal watchdog and the
    eager GPU touch pass runs. The watchdog is a time limit - how many mmap
    bytes wire before it fires depends on SSD/memory bandwidth and RAM, all
    hardware-dependent. Observed on a 128 GB M3 Max: death at 61 GB, survival
    at ~30 GB. Scale with the machine (a third of the recommended working
    set), capped at the 32 GB validated here - touching too eagerly costs
    seconds of load, not touching when needed hangs the GPU, so bias low.
    GMLX_WARM_TOUCH_GB overrides.
    """
    gb = os.environ.get("GMLX_WARM_TOUCH_GB", "")
    if gb:
        try:
            return int(float(gb) * (1 << 30))
        except ValueError:
            pass
    cap = 32 << 30
    try:
        wss = int(mx.device_info()["max_recommended_working_set_size"])
        return min(cap, wss // 3)
    except Exception:
        return cap


def _active_now() -> float | None:
    """MLX-tracked active bytes, None off-device (baseline for the
    untracked-weights split in _warm_mmap_residency)."""
    try:
        return float(mx.get_active_memory())
    except Exception:
        return None


def weights_source_key(*paths: str) -> tuple | None:
    """Identity of a load's weight bytes (absolute file paths) for the
    untracked-headroom registry: reloads of the same file count once (the
    first registration per key wins; see note_untracked_weights)."""
    return tuple(os.path.abspath(p) for p in paths) or None


def _warm_mmap_residency(
    model, *, log=print, paths: list[str] | None = None,
    batch_bytes: int = 4 << 30, threshold_bytes: int | None = None,
    active_before: float | None = None, source_key: tuple | None = None,
) -> None:
    """Pre-wire GPU residency of mmap-backed weights in small batches.

    Zero-copy loads leave every weight a view over the GGUF mmap; the first
    forward then faults + wires the whole file inside a single command
    buffer, which blows the Metal watchdog once the file outgrows ~50 GB
    (gpt-oss-120b at 61 GB dies; 35B-class at ~30 GB survives). Touch each
    weight with a throwaway reduction and eval every few GB so the wiring
    spreads across many short command buffers.

    Scope: models over the wired budget stream (their expert bytes are never
    GPU-referenced), so touching them here read the whole file only for the
    head to be evicted by the tail - a 162 GB model spent ~50 s of load on
    it, trashing every other model's cache on the way. Skipped. Below
    ``threshold_bytes`` the watchdog is not at risk and lazy wiring during
    the first forward is near-free (measured: an eager touch costs seconds,
    the lazy path nothing), so the GPU touch is skipped there too and only
    the page-cache populate runs. GMLX_RESIDENCY_WARM=0 disables
    everything, =1 forces the GPU touch regardless of size.
    """
    pairs = tree_flatten(model.parameters())
    total = sum(a.nbytes for _, a in pairs)
    # Streamable-table exclusion: a table the selection ladder will stream
    # must not be GPU-touched here - the touch would wire the whole buffer
    # before install_expert_streaming ever runs. Only the touch skips it;
    # ``total`` keeps the full sum so the untracked-weights registration
    # below stays whole-model.
    try:
        _tbudget = int(
            0.9 * mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        _tbudget = None
    from gmlx.stream.table_stream import warm_touch_exclusions

    _tskip = warm_touch_exclusions(model, total, _tbudget)
    arrays = [v for _, v in pairs if id(v) not in _tskip]
    try:
        _warm_touch_pass(arrays, total, log=log, paths=paths,
                         batch_bytes=batch_bytes,
                         threshold_bytes=threshold_bytes)
    finally:
        # Register on every exit path: the headroom estimate needs weight
        # bytes counted whether or not the touch pass ran. Only bytes
        # invisible to mx.get_active_memory may be registered: weights a
        # load materializes (owned copies, repacked buffers) are tracked
        # already, and noting the full total for such a load counts them
        # twice, driving the headroom estimate negative. The tracked
        # portion is the active-memory delta across the load; the touch
        # pass evaluates any still-lazy materialized weights first, so
        # the delta is settled by this point.
        tracked = 0.0
        if active_before is not None:
            try:
                tracked = max(0.0, mx.get_active_memory() - active_before)
            except Exception:
                tracked = 0.0
        # Keyed by shard paths so a drafter reloading the target's GGUF
        # cannot register the same pages twice (first registration wins).
        key = source_key or (weights_source_key(*paths) if paths else None)
        if key is not None:
            # install_expert_streaming reads this to deduct streamed-out
            # expert bytes from the same registration.
            object.__setattr__(model, "_kq_weights_key", key)
        note_untracked_weights(max(0.0, total - min(tracked, total)), key=key)


def _warm_touch_pass(
    arrays, total, *, log, paths, batch_bytes, threshold_bytes,
) -> None:
    mode = os.environ.get("GMLX_RESIDENCY_WARM", "")
    if mode == "0":
        return
    if threshold_bytes is None:
        threshold_bytes = _warm_touch_threshold_bytes()
    try:
        budget = int(0.9 * mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        budget = None
    if mode != "1" and budget is not None and total > budget:
        return  # streaming-bound: bytes are page cache, never wired
    if paths:
        start_populate(paths, log=log)
    if mode != "1" and total < threshold_bytes:
        return
    t0 = time.time()
    pending, acc = [], 0
    for a in arrays:
        pending.append(a.sum())
        acc += a.nbytes
        if acc >= batch_bytes:
            mx.eval(pending)
            pending, acc = [], 0
    if pending:
        mx.eval(pending)
    log(f"[load_weights] residency warm: {total / 1e9:.1f} GB wired "
        f"in {time.time() - t0:.1f}s")


# Per-model_type target-name substrings whose params must stay float32
# through the blanket bf16 cast in the load path. deepseek_v4: the vendored
# hyper-connection Metal kernel casts its mixes/base inputs to
# `device float*`/`float4*` (bf16 bits reinterpreted as f32 = garbage), and
# the QAT-parity params (per-head sinks, compressor ape tables, router +
# selection bias) are semantically fp32 - ds4, the parity reference, computes
# all of them in fp32, and casting them measurably breaks logit parity.
# Norms and matrix weights still cast normally.
_FP32_KEEP_BY_MODEL_TYPE: dict[str, tuple[str, ...]] = {
    "deepseek_v4": ("_hc.", "hc_head.", ".attn_sink", ".ape",
                    ".e_score_correction_bias", ".gate.weight"),
    # deepseek_v4_vl: the Vision-Exp container; the text tower's set applies
    # under language_model.*, the ViT and aligner cast normally (the tower
    # keeps its norms and rope tables fp32 internally).
    "deepseek_v4_vl": ("_hc.", "hc_head.", ".attn_sink", ".ape",
                       ".e_score_correction_bias", ".gate.weight"),
    # hy_v3 routing is semantically fp32 (F32 wire; llama.cpp routes in fp32,
    # and the vendored class's cast_predicate exempts expert_bias): sigmoid
    # gate + selection bias decide top-8 of 192, where bf16 rounding flips
    # near-tie selections.
    "hy_v3": (".mlp.router.gate.weight", ".mlp.router.expert_bias"),
    # kimi_k3: routing is fp32 (sigmoid top-16-of-896 + correction bias is
    # near-tie-heavy); the KDA decay/state path and the res-mix scores are
    # computed fp32 (the vendored cast_predicate pins the same set).
    "kimi_k3": (".mlp.gate.weight", ".e_score_correction_bias",
                ".a_folded", ".dt_bias", "_res_score"),
    # qwen4_exp: softmax top-10-of-512 routing is fp32 in llama.cpp (F32
    # wire); the GDN decay params feed the fp32 scan; the per-stream inject
    # scalars scale the residual streams directly.
    "qwen4_exp": (".mlp.gate.weight", ".A_log", ".dt_bias", ".inject.weight"),
    # glm5_next: hyper-connection mixers are fp32 like deepseek_v4; sigmoid
    # top-8-of-288 routing + correction bias is near-tie-heavy; the KDA
    # decay params feed the fp32 recurrence; the indexer head-weights GEMM
    # and ape table are fp32 per llama.cpp PR 27754.
    "glm5_next": ("_hc.", ".mlp.gate.weight", ".e_score_correction_bias",
                  ".a_folded", ".dt_bias", ".ape", ".weights_proj."),
    # hy_v4: the iHC mixers and the collapse head are fp32 in the reference
    # and accumulate over 78 layers x 2 sublayers; the per-head sinks join
    # the softmax normalizer; sigmoid top-8-of-256 routing with a correction
    # bias is near-tie-heavy; the indexer head weights decide near-tied key
    # rankings. Kept in step with the model class's cast_predicate, which
    # tests/models/test_hy_v4_model.py cross-checks against this row.
    "hy_v4": ("_hc.", "hc_head.", ".sinks", ".e_score_correction_bias",
              ".mlp.gate.weight", ".weights_proj."),
}

# Params kept at their native f16 through the bf16 cast (no upcast). MLX
# promotes an f16-weight matmul against f32 activations to f32, so these read
# half the bytes of an fp32 pin while computing the same values.
_F16_KEEP_BY_MODEL_TYPE: dict[str, tuple[str, ...]] = {
    # muse_glimmer's mmproj is native F16 and llama.cpp runs the tower with f32
    # activations. 50 residual layers with large outliers (features span +-76)
    # compound bf16 rounding into ~10% relative RMS on the projected embeddings
    # against an f32 run. The tower entry casts its input to f32, so activations
    # ride fp32 promotion while the weights stay F16 - the oracle's own layout.
    # Vision only - the text tower's bf16 holds 16k parity.
    "muse_glimmer": ("vision_tower.", "vision_adapter.", "vision_projection."),
    "kimi_k25": ("vision_tower.", "mm_projector."),
}


def preset_native_fp_wire_env(args) -> None:
    """Pre-set wire mode when a streaming placement is coming.

    Placement (``_apply_placement``) runs only after ``load_model`` returns,
    but the native-fp repack-vs-wire decision happens inside the load - so
    the CLI surfaces call this before loading. ``setdefault`` keeps an
    explicit ``GMLX_NATIVE_FP`` override in charge.
    """
    if (getattr(args, "stream_cpu", False)
            or getattr(args, "stream_experts", False)):
        os.environ.setdefault("GMLX_NATIVE_FP", "wire")


def _resolve_native_fp_wire(hf_weights, hf_kquant_meta, log) -> bool:
    """Decide wire vs packed handling for native-fp (mxfp4/nvfp4) tensors.

    Wire mode keeps them as zero-copy GGUF wire bytes dispatched through the
    kq kernels like every k-quant codec (streamable, instant load); packed
    mode eagerly de-interleaves into MLX's packed layout for the stock
    ``mx.gather_qmm(mode=...)`` kernels (materializes every native-fp tensor).

    ``GMLX_NATIVE_FP`` = ``wire`` | ``packed`` | ``auto`` (default). Auto
    picks wire when the kq build carries the codecs and the model does not
    fit the wired working set (a fitting model keeps today's packed kernels
    until the wire path passes the perf gate); the CLI pre-sets ``wire`` for
    CPU/hybrid placements via ``preset_native_fp_wire_env``.
    """
    from .native_fp import NATIVE_FP_CODECS

    codecs = {c for c in hf_kquant_meta.values() if c in NATIVE_FP_CODECS}
    if not codecs:
        return False
    mode = env_choice("GMLX_NATIVE_FP", "auto", ("wire", "packed", "auto"))
    supported = codecs <= set(kq.codecs())
    if mode == "wire":
        if not supported:
            missing = sorted(codecs - set(kq.codecs()))
            raise RuntimeError(
                f"GMLX_NATIVE_FP=wire but this mlx-kquant build lacks "
                f"codec(s) {missing}; upgrade mlx-kquant or use packed")
        log("[native-fp] wire mode (env): mxfp4/nvfp4 stay zero-copy wire bytes")
        return True
    if mode == "packed":
        return False
    if not supported:
        return False
    total = sum(v.nbytes for v in hf_weights.values())
    info = mx.device_info()
    wss = int(info.get("max_recommended_working_set_size", 0))
    if wss and total > 0.9 * wss:
        log(
            f"[native-fp] wire mode (auto): {total / 2**30:.0f} GiB exceeds "
            f"the wired budget ({0.9 * wss / 2**30:.0f} GiB) - native-fp "
            f"tensors stay zero-copy wire bytes")
        return True
    return False


def _native_fp_multilinear_keys(model, hf_kquant_meta) -> set[str]:
    """Weight names of native-fp tensors destined for a MultiLinear leaf.

    These stay ggml wire bytes even in packed mode (kq.gather_qmm dispatch;
    no packed MultiLinear module exists) - the repack must skip them.
    """
    from .modules import MultiLinear
    from .native_fp import NATIVE_FP_CODECS

    if MultiLinear is None:
        return set()
    keys = set()
    for path, mod in tree_flatten(model.leaf_modules(),
                                  is_leaf=nn.Module.is_module):
        wk = f"{path}.weight"
        if (isinstance(mod, MultiLinear)
                and hf_kquant_meta.get(wk) in NATIVE_FP_CODECS):
            keys.add(wk)
    return keys


def _gemma4_target(model) -> bool:
    """True when the loaded model runs the gemma4 classes the gemma-4
    patches target. The installs themselves stay unconditional (they are
    module-level and inert elsewhere); this only keeps the [install] log
    lines from claiming gemma-4 levers on unrelated archs."""
    lm = getattr(model, "language_model", model)
    return ("gemma4" in type(model).__module__
            or "gemma4" in type(lm).__module__)


def _install_and_load(
    model,
    hf_weights,
    hf_kquant_meta,
    *,
    log,
    sanitize: bool = True,
    no_alias: set[str] | None = None,
    fp32_keep: tuple[str, ...] = (),
    f16_keep: tuple[str, ...] = (),
    source_key: tuple | None = None,
    active_before: float | None = None,
) -> None:
    """Sanitize -> de-interleave native-fp -> swap kquant leaves -> cast -> load.

    The back half of ``load_model`` (steps 5-7), factored so the two-GGUF VLM
    loader (``gguf/vlm.py``) drives the exact same swap+load behaviour. ``model``
    is mutated in place; ``hf_weights`` keys with no matching model parameter are
    dropped (``strict=False``). ``log`` is a ``print``-like callable.

    ``sanitize=False`` skips ``model.sanitize`` for callers that already produce
    final parameter names: an mlx-vlm Model's sanitize remaps ``language_model.X``
    -> ``language_model.model.X``, which would double-prefix VLM text keys the
    remap has already placed there.

    ``no_alias``: post-remap names of transform results that must own their
    buffers (donation tripwire); tracked through ``model.sanitize`` renames by
    the same suffix match used for the kquant meta.

    ``fp32_keep``: target-name substrings pinned to float32 through the bf16
    cast (see ``_FP32_KEEP_BY_MODEL_TYPE``). ``f16_keep``: substrings kept at
    their native f16 instead (see ``_F16_KEEP_BY_MODEL_TYPE``).

    ``active_before``: active-memory baseline for the untracked-weights split.
    Callers that read wire bytes before installing must pass the pre-read
    value; wire reads can grow active memory, and a post-read baseline makes
    those tracked bytes register as untracked on top of it.
    """
    loadlog.stage("loading weights")
    if active_before is None:
        active_before = _active_now()
    # 5. sanitize first - model.sanitize may rename keys; rebuild meta.
    # Codec'd tensors can land on non-``.weight`` raw leaves (hc ``fn``),
    # so only the vestigial wire siblings are excluded from the rematch.
    if sanitize and hasattr(model, "sanitize"):
        hf_weights = model.sanitize(hf_weights)
        new_meta: dict[str, str] = {}
        unmatched_meta = set(hf_kquant_meta)
        for new_k in hf_weights:
            if new_k.endswith(".scales") or new_k.endswith(".biases"):
                continue
            for old_k in list(unmatched_meta):
                if new_k == old_k or new_k.endswith("." + old_k):
                    new_meta[new_k] = hf_kquant_meta[old_k]
                    unmatched_meta.discard(old_k)
                    break
        hf_kquant_meta = new_meta
        if no_alias:
            no_alias = {
                new_k
                for new_k in hf_weights
                for old_k in no_alias
                if new_k == old_k or new_k.endswith("." + old_k)
            }

    # 5b. native-fp codecs (mxfp4/nvfp4): keep as zero-copy wire bytes (wire
    # mode) or de-interleave into MLX's packed layout (packed mode).
    native_fp_wire = _resolve_native_fp_wire(hf_weights, hf_kquant_meta, log)
    if not native_fp_wire:
        from .native_fp import repack_native_fp_weights

        n_fp = repack_native_fp_weights(
            hf_weights, hf_kquant_meta,
            skip=_native_fp_multilinear_keys(model, hf_kquant_meta))
        if n_fp:
            log(
                f"[native-fp] de-interleaved {n_fp} mxfp4/nvfp4 tensors -> MLX packed layout"
            )

    # 6. swap leaves with kquant equivalents.
    dequant = dequantize_unattachable_leaves(model, hf_weights, hf_kquant_meta)
    if dequant:
        log(f"[install] dequantized {len(dequant)} raw-array leaves to f32: "
            + ", ".join(dequant[:3])
            + (f" (+{len(dequant) - 3} more)" if len(dequant) > 3 else ""))
    n_replaced = install_kquant_modules(
        model, hf_kquant_meta, native_fp_wire=native_fp_wire)
    log(f"[install] replaced {n_replaced} leaves with kquant modules")

    if install_hd512_sdpa():
        log("[install] head_dim-512 fused SDPA active")
    if install_quantized_sdpa_mask_fix():
        log("[install] quantized-KV SDPA batch-mask fix active")
    if install_quantized_cache_pack_fix():
        log("[install] quantized-KV pack-width fix active")
    if install_rope_batch_fix():
        log("[install] rope int-offset batch fix active")
    # Inside the decay wrap (installed first): media blocks stay whole.
    from gmlx.gen.media_spans import install_span_aware_prompt_step
    install_span_aware_prompt_step()
    if install_prefill_decay():
        log("[install] depth-decay prefill chunking active")
    if install_gemma4_nosync() and _gemma4_target(model):
        log("[install] gemma-4 host-sync-free masks/rope offsets active")
    if install_gemma4_batched_sdpa() and _gemma4_target(model):
        log("[install] gemma-4 hd512 batched-decode row route active")
    if install_gemma4_softcap_f32() and _gemma4_target(model):
        log("[install] gemma-4 float32 logit softcap active")
    if install_cascade_sdpa() and install_cascade_stamp():
        log("[install] shared-prefix cascade decode route active")
    if install_sparse_sdpa():
        log("[install] sparse top-k page decode route active (lossy, opt-in)")
    n_fused_moe = install_fused_moe_glu(model)
    if n_fused_moe:
        log(f"[install] fused mxfp4 MoE GLU decode on {n_fused_moe} layers")
    n_shexp = install_hyv3_shexp_fold(model)
    if n_shexp:
        log(f"[install] hy3 shared-expert fold on {n_shexp} MoE layers")
    n_fused_qkv = install_fused_qkv(model)
    if n_fused_qkv:
        log(f"[install] fused QKV decode projection on {n_fused_qkv} layers")
    n_occ = install_occupancy_fuse(model)
    if n_occ:
        log(f"[install] occupancy fusion (qkv + gate/up decode) on "
            f"{n_occ} modules")
    install_rotating_cache_fix()
    install_arrays_cache_fix()

    # 7. partition by what the constructed model actually defines + load.
    model.eval()
    model_params = {p for p, _ in tree_flatten(model.parameters())}
    loadable = {k: v for k, v in hf_weights.items() if k in model_params}
    redundant = sorted(set(hf_weights.keys()) - set(loadable.keys()))
    if redundant:
        log(
            f"[load_weights] dropping {len(redundant)} redundant tensors "
            f"(no model slot): {redundant[:3]}..."
        )

    act_dtype = activation_dtype()
    act_name = activation_dtype_name()
    n_cast = 0
    for k in list(loadable):
        v = loadable[k]
        if v.dtype in (mx.float32, mx.float16) and k not in hf_kquant_meta:
            if fp32_keep and any(s in k for s in fp32_keep):
                if v.dtype != mx.float32:      # e.g. F16 ape tables
                    loadable[k] = v.astype(mx.float32)
                continue
            if f16_keep and any(s in k for s in f16_keep):
                continue
            if v.dtype == act_dtype:
                continue                       # already the activation dtype
            if v.dtype == mx.float16:
                # Same-itemsize f16->bf16 gets buffer-donated into the source
                # view -- a write through the zero-copy file mapping (dropped
                # on read-only maps, leaving f16 bits typed as bf16). The f32
                # hop makes both steps size-changing, so neither can donate.
                v = v.astype(mx.float32)
            loadable[k] = v.astype(act_dtype)
            n_cast += 1
    if n_cast:
        log(f"[dtype] cast {n_cast} float params (norms etc.) to {act_name}")

    model.load_weights(list(loadable.items()), strict=False)
    log(f"[load_weights] loaded {len(loadable)} / {len(model_params)} model parameters")
    _warm_mmap_residency(model, log=log, active_before=active_before,
                         source_key=source_key)

    missing = sorted(model_params - set(loadable.keys()))
    if missing:
        loadlog.warn(
            f"WARNING: {len(missing)} model params not loaded: {missing[:5]}..."
        )

    _verify_zero_copy_views(model, no_alias, log)


def _dequantize_diffusion_embedding(model, log) -> None:
    """Replace a DiffusionGemma kquant ``embed_tokens`` with a bf16 embedding.

    The denoiser mixes embedding rows weighted by token probabilities
    (``probs @ embed_tokens.weight``) - in both the model's self-conditioning
    path and the engine's soft-embedding step. That needs a dense float table;
    kquant wire bytes feed neither, and the stock fast path only special-cases
    ``nn.QuantizedEmbedding``. So dequantize the table to a plain
    ``nn.Embedding`` at the activation dtype (a layout the model handles
    natively), in row chunks to stay under the single-dispatch grid limit. The
    tied ``as_linear`` logits then run at that dtype too. Other quantized
    leaves are untouched.
    """
    emb = model.model.decoder.embed_tokens
    if not isinstance(emb, KQuantEmbedding):
        return
    n, dims, codec = emb.num_embeddings, emb.dims, emb.kquant_type
    packed, scales = emb["weight"], emb["scales"]
    chunk = 16384
    act_dtype = activation_dtype()
    rows = [
        kq.dequantize(
            packed[i : i + chunk].reshape(-1, packed.shape[-1]), scales, codec
        )
        .reshape(min(chunk, n - i), dims)
        .astype(act_dtype)
        for i in range(0, n, chunk)
    ]
    table = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    mx.eval(table)
    new_emb = nn.Embedding(n, dims)
    new_emb.weight = table
    new_emb.freeze()
    model.model.decoder.embed_tokens = new_emb
    log(
        f"[diffusion] dequantized embed_tokens {codec}->"
        f"{activation_dtype_name()} ({n}x{dims})"
    )


# Archs whose sparse-attention indexer tensors may arrive via a companion
# sidecar GGUF when the model file itself was converted without them.
_INDEXER_SIDECAR_ARCHS = frozenset({"minimax-m3"})


def _resolve_indexer_sidecar(
    gguf_path: str, arch: str | None, tensor_shapes: dict
) -> str | None:
    """Path of the MSA indexer sidecar GGUF to merge, or None.

    Only fires for ``_INDEXER_SIDECAR_ARCHS`` models whose GGUF lacks the
    indexer tensors (either spelling). ``GMLX_INDEXER_SIDECAR`` overrides
    discovery - a path, or ``0``/``off``/``none`` to disable; otherwise the
    model's directory is scanned for ``*indexer*.gguf``. A sidecar-less
    indexless model loads dense with a one-time quality warning.
    ``GMLX_MSA_DISABLE=1`` skips the sidecar (and the warning) entirely.
    """
    if arch not in _INDEXER_SIDECAR_ARCHS:
        return None
    if os.environ.get("GMLX_MSA_DISABLE", "") == "1":
        return None  # MSA off: sanitize would drop the merged tensors anyway
    if any(".indexer." in n or ".index_q_proj." in n for n in tensor_shapes):
        return None  # native MSA GGUF - nothing to merge
    env = os.environ.get("GMLX_INDEXER_SIDECAR")
    if env is not None:
        if env.strip().lower() in ("", "0", "off", "none"):
            return None
        p = os.path.abspath(os.path.expanduser(env))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"GMLX_INDEXER_SIDECAR not found: {p}")
        return p
    import glob

    model_dir = os.path.dirname(os.path.abspath(gguf_path))
    cand = sorted(glob.glob(os.path.join(model_dir, "*indexer*.gguf")))
    if not cand:
        loadlog.warn(
            f"{arch}: no MSA indexer tensors in the GGUF and no "
            "*indexer*.gguf sidecar next to it - running DENSE attention. "
            "The model is trained with sparse attention; dense output "
            "degrades at long context (reasoning loops). Use an MSA-converted "
            "GGUF or place an indexer sidecar beside the model "
            "(GMLX_INDEXER_SIDECAR overrides discovery)."
        )
        return None
    if len(cand) > 1:
        loadlog.warn(
            f"multiple indexer sidecars in {model_dir}; using {cand[0]} "
            "(set GMLX_INDEXER_SIDECAR to pick)"
        )
    return cand[0]


# Public load entry point


@loadlog.seeds
def load_model(
    gguf_path: str,
    *,
    arch: str | None = None,
    hf_source: str | None = None,
    chat_template: str | None = None,
    target_prefix: str = "",
    no_remap: bool = False,
    fail_on_unknown: bool = False,
    zero_copy: bool = True,
    verbose: bool = False,
):
    """Load a text-only GGUF K-quant file into an mlx-lm model.

    Returns ``(model, config, tokenizer)``. The model is a stock mlx-lm
    ``class Model`` with quantized leaves swapped for ``KQuant*`` modules; it
    drives normally under ``mlx_lm.generate`` / ``stream_generate``.

    Args:
        gguf_path: path to a (possibly sharded) GGUF file.
        arch: override ``general.architecture`` detection.
        hf_source: load the model config from this local dir's ``config.json``
            or HF repo id instead of synthesizing it from GGUF metadata. Also
            unlocks arches that have an mlx-lm model class but no config
            synthesizer yet, and fixes variants whose synthesized constants
            are wrong (e.g. gemma-2/3 27B ``query_pre_attn_scalar``).
        chat_template: inline Jinja string or path to a ``.jinja``/``.txt`` file
            that replaces the GGUF's chat template (threaded into the
            tokenizer synthesizer).
        target_prefix: prepend to all remapped tensor names.
        no_remap: skip GGUF->HF name remap (raw GGUF names).
        fail_on_unknown: hard-fail on any tensor with no remap entry.
        zero_copy: load tensors as no-copy mmap views (default) vs memcpy.
        verbose: print load diagnostics (default quiet; the CLI drives
            its own spinner/summary session instead).
    """

    _log = loadlog.verbose_print
    active_before = _active_now()

    # 0. preflight - discover shards, classify codecs (IQ / unsupported types
    #    refuse here, naming the codec, before kq.load_gguf's cryptic
    #    "unsupported type N"), and gate on the architecture. Reads only the
    #    GGUF header via gguf-py, so it stays cheap on multi-GB files.
    loadlog.stage("reading gguf metadata")
    pf = preflight(gguf_path, arch=arch, hf_source=hf_source)
    arch = pf.arch
    loadlog.fact("arch", arch)
    loadlog.fact_file_size(pf.shards)
    _log(f"[arch] {arch}")

    # Kick off the page-cache populate as early as the shard list exists so
    # the disk stream overlaps the whole CPU-bound remainder of load (the
    # phase-7 residency warm dedupes via the populate registry).
    maybe_populate_for_load(pf.shards, log=_log)

    # Larger-than-RAM shards leave a stale cache remnant that taxes the next
    # process's fault path; sweep it back to the free list at exit.
    from gmlx.stream.pagecache import register_streaming_release
    register_streaming_release(pf.shards)

    # 1. load wire bytes via kq.load_gguf (now known IQ-free; shards reused).
    loadlog.stage("reading tensors")
    t0 = time.perf_counter()
    arrays, kquant_meta, _arch_meta, meta, tensor_shapes = load_gguf_wire_bytes(
        gguf_path, zero_copy=zero_copy, shards=pf.shards
    )
    _log(
        f"[gguf] {len(arrays)} arrays, {len(kquant_meta)} kquant "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    # 1a. MSA indexer sidecar (minimax-m3): community GGUFs converted by the
    #     dense-only port ship without the blk.N.indexer.* tensors the sparse
    #     attention path needs. A small companion GGUF holding just those
    #     tensors (+ the attention.indexer.* KVs) can sit next to the model;
    #     its contents merge into the wire set ahead of remap/synth so they
    #     flow exactly like a native MSA GGUF's.
    sidecar = _resolve_indexer_sidecar(gguf_path, arch, tensor_shapes)
    if sidecar:
        s_arrays, s_kmeta, _s_arch, s_meta, s_shapes = load_gguf_wire_bytes(
            sidecar, zero_copy=zero_copy, expect_quant=False
        )
        arrays.update(s_arrays)
        kquant_meta.update(s_kmeta)
        tensor_shapes.update(s_shapes)
        meta.update(
            {k: v for k, v in s_meta.items() if ".attention.indexer." in k}
        )
        loadlog.fact("indexer-sidecar", os.path.basename(sidecar))
        _log(
            f"[sidecar] merged {len(s_arrays)} indexer tensors from "
            f"{os.path.basename(sidecar)}"
        )

    # 1b. coalesce legacy per-expert MoE weights (old Mixtral split format) into
    #     the stacked `_exps` form the remap + SwitchGLU expect. No-op otherwise.
    arrays, kquant_meta, n_coalesced = coalesce_split_experts(arrays, kquant_meta)
    if n_coalesced:
        _log(f"[gguf] coalesced {n_coalesced} split-expert groups -> stacked _exps")

    # 1c. granitemoehybrid: fuse each layer's shared-expert gate/up pair into
    #     the single fused tensor mlx-lm's shared_mlp.input_linear expects.
    #     Arch-gated - every other shexp arch keeps the halves separate.
    if arch == "granitehybrid":
        arrays, kquant_meta, n_fused = fuse_shexp_gate_up(arrays, kquant_meta)
        if n_fused:
            _log(
                f"[gguf] fused {n_fused} shared-expert gate/up pairs "
                f"-> ffn_gate_up_shexp"
            )

    # 2. remap names + layout. Head counts (for the llama.cpp Q/K permute) come
    # from the decoded GGUF KV; head_count_kv may be a per-layer array.
    loadlog.stage("remapping tensors")
    n_head = read_int(meta, f"{arch}.attention.head_count")
    n_head_kv = first_nonzero_int(meta, f"{arch}.attention.head_count_kv")

    # llama.cpp bakes Llama-3.x "llama3" rope scaling into a per-dim factors
    # tensor (no scaling KV survives conversion); copy it out before remap
    # drops it so the rope patch below can restore long-context fidelity.
    rope_factors = arrays.get("rope_freqs.weight")
    if rope_factors is not None:
        rope_factors = mx.array(np.asarray(rope_factors, dtype=np.float32))

    owned_names: set[str] = set()
    hf_weights, hf_kquant_meta, stats = remap_arrays(
        arrays,
        kquant_meta,
        arch,
        no_remap=no_remap,
        target_prefix=target_prefix,
        fail_on_unknown=fail_on_unknown,
        n_head=n_head,
        n_head_kv=n_head_kv,
        owned_names=owned_names,
    )
    n_nextn_dropped = strip_nextn_trunk_overflow(hf_weights, hf_kquant_meta, meta, arch)
    if n_nextn_dropped:
        _log(f"[gguf] dropped {n_nextn_dropped} NextN/MTP-block trunk entries")
    # hf_weights now holds the only ref to each wire view; drop arrays so the
    # native-fp repack below can free each view as it packs it (caps 120B peak).
    del arrays

    from collections import Counter

    loadlog.fact("codecs", Counter(hf_kquant_meta.values()))
    if loadlog.is_verbose():
        print_inventory(arch, kquant_meta, hf_kquant_meta, stats)

    # 3. build unquantized model from synthesized (or supplied) config.
    loadlog.stage("building model")
    if hf_source is not None:
        config_dict = _load_config_from_source(hf_source)
    else:
        from .config_synth import synthesize_config

        config_dict = synthesize_config(meta, tensor_shapes)
    model, config = build_model(config_dict)
    loadlog.fact("model_type", config.get("model_type"))
    if config.get("use_sparse_attention"):
        loadlog.fact(
            "attn",
            "dense (GMLX_MSA_DISABLE)"
            if os.environ.get("GMLX_MSA_DISABLE", "") == "1"
            else "msa",
        )

    # 4. runtime patches.
    # 4a. qwen3next split-GDN wire layout (current llama.cpp converts split
    #     in_proj_qkvz -> attn_qkv + attn_gate): restructure the GDN modules so
    #     the remapped split weights land directly. Must precede sanitize and
    #     the kquant leaf swap.
    if config.get("model_type") == "qwen3_next" and config.get("gdn_split_layout"):
        _patch_qwen3next_split_gdn(model)

    # 4a'. hunyuan MoE router: mlx-lm omits the norm_topk_prob rescale that
    #      the HF reference and llama.cpp both apply - without it A13B
    #      degenerates from the first token.
    if config.get("model_type") == "hunyuan":
        _patch_hunyuan_norm_topk(model)

    # 4a''. llama-3.x long-context rope: restore the "llama3" scaling that
    #       llama.cpp baked into rope_freqs.weight (see _patch_rope_factors).
    if rope_factors is not None and config.get("model_type") == "llama":
        _patch_rope_factors(model, rope_factors)

    # 4b. GGUF V-head tiling fixup for asymmetric K/V heads.
    if _needs_tiled_v_patch(config):
        _patch_gated_delta_tiled_v()
    elif config.get("model_type") == "qwen3_next" and _tiled_v_patch_applied():
        # The tiled-V patch rewrites mlx_lm.models.gated_delta module globals,
        # and qwen3_next's gated_delta_update goes through that same module -
        # once a qwen3.5/3.6 hybrid has been loaded in this process, a
        # subsequent qwen3_next load would silently run the wrong (tiled) K->V
        # mapping. Fail loudly instead. (kimi_k3 also dispatches through
        # gated_delta but needs no guard: its K/V heads are symmetric, and
        # with Hk == Hv the tiled and grouped K->V mappings are identical.)
        raise RuntimeError(
            "cannot load a qwen3next GGUF after a qwen3.5/3.6 hybrid in the "
            "same process: the qwen3.5 tiled-V runtime patch (already applied) "
            "would corrupt qwen3_next's grouped gated-delta K->V mapping. "
            "Load the qwen3next model in a fresh process."
        )

    # 4c. DeepSeek-V3.2 / glm-dsa decode correctness. Fixes only visible past
    #     index_topk (~2048), so generation "starts strong then degrades at depth":
    #       (rope) the indexer must rope its q/k with DeepSeek's INTERLEAVED
    #         convention (mlx traditional=True) - the same as the main attention and
    #         stock mlx-lm. HF's apply_rotary_pos_emb deinterleaves then rotate_half,
    #         which equals interleaved for the score q*k. Also k_norm eps=1e-6 + fp32
    #         selection. Kill with GMLX_DSV32_INDEXER_ROPE=0 /
    #         GMLX_DSV32_INDEXER_FP32=0.
    #       (routing) the MoE router runs in bf16 vs HF's fp32, flipping borderline
    #         expert picks. Kill with GMLX_DSV32_GATE_FP32=0.
    #       (sink/local) even with the above, the indexer scores the BOS attention-
    #         sink + most-recent keys very negative in ~17/78 layers and drops them,
    #         though the main attention parks 0.5-0.99 of its weight there
    #         (StreamingLLM) - the residual degradation. The fp32 indexer force-keeps
    #         sink+local (GMLX_DSV32_SINK/_LOCAL). Why our score diverges from
    #         llama (which keeps them via score alone) is unresolved; sparse stays
    #         experimental and dense is the default.
    #       (secondary, retired default) the L==1 gather decode was suspected of
    #         corrupting the sampling tail on an early stack; re-tested clean, so
    #         stock gather is the default. GMLX_DSV32_MASK_DECODE=1 re-arms the
    #         mask-path mitigation.
    if config.get("model_type") in ("glm_moe_dsa", "deepseek_v32"):
        _patch_dsv32_indexer_rope(model)
        _patch_dsv32_indexer_fp32(model)
        _patch_dsv32_moe_gate_fp32(model)
        _patch_dsv32_moe_scores(model)
        _patch_dsv32_mask_decode(model)
        _patch_dsv32_dense_default(model)  # exact default; GMLX_DSV32_SPARSE=1 -> sparse (experimental)

    # 5. sanitize first - model.sanitize may rename keys; rebuild meta.
    # Codec'd tensors can land on non-``.weight`` raw leaves (hc ``fn``),
    # so only the vestigial wire siblings are excluded from the rematch.
    if hasattr(model, "sanitize"):
        hf_weights = model.sanitize(hf_weights)
        new_meta: dict[str, str] = {}
        unmatched_meta = set(hf_kquant_meta)
        for new_k in hf_weights:
            if new_k.endswith(".scales") or new_k.endswith(".biases"):
                continue
            for old_k in list(unmatched_meta):
                if new_k == old_k or new_k.endswith("." + old_k):
                    new_meta[new_k] = hf_kquant_meta[old_k]
                    unmatched_meta.discard(old_k)
                    break
        hf_kquant_meta = new_meta
        if owned_names:
            owned_names = {
                new_k
                for new_k in hf_weights
                for old_k in owned_names
                if new_k == old_k or new_k.endswith("." + old_k)
            }

    # 5b. native-fp codecs (mxfp4/nvfp4): wire mode keeps them as zero-copy
    #     ggml wire bytes on the kq kernels (like every k-quant codec, so they
    #     stream); packed mode de-interleaves into MLX's packed (uint32 weight
    #     + uint8 scales) layout, in place - a pure byte/nibble shuffle (no
    #     dequant) that drives mx.gather_qmm(mode=<codec>) but materializes
    #     every native-fp tensor.
    native_fp_wire = _resolve_native_fp_wire(hf_weights, hf_kquant_meta, _log)
    if not native_fp_wire:
        from .native_fp import repack_native_fp_weights

        n_fp = repack_native_fp_weights(
            hf_weights, hf_kquant_meta,
            skip=_native_fp_multilinear_keys(model, hf_kquant_meta))
        if n_fp:
            _log(
                f"[native-fp] de-interleaved {n_fp} mxfp4/nvfp4 tensors -> MLX packed layout"
            )

    # 6. swap leaves with kquant equivalents.
    loadlog.stage("installing quantized weights")
    dequant = dequantize_unattachable_leaves(model, hf_weights, hf_kquant_meta)
    if dequant:
        _log(f"[install] dequantized {len(dequant)} raw-array leaves to f32: "
             + ", ".join(dequant[:3])
             + (f" (+{len(dequant) - 3} more)" if len(dequant) > 3 else ""))
    n_replaced = install_kquant_modules(
        model, hf_kquant_meta, native_fp_wire=native_fp_wire)
    _log(f"[install] replaced {n_replaced} leaves with kquant modules")

    if install_hd512_sdpa():
        _log("[install] head_dim-512 fused SDPA active")
    if install_quantized_sdpa_mask_fix():
        _log("[install] quantized-KV SDPA batch-mask fix active")
    if install_quantized_cache_pack_fix():
        _log("[install] quantized-KV pack-width fix active")
    if install_rope_batch_fix():
        _log("[install] rope int-offset batch fix active")
    # Inside the decay wrap (installed first): media blocks stay whole.
    from gmlx.gen.media_spans import install_span_aware_prompt_step
    install_span_aware_prompt_step()
    if install_prefill_decay():
        _log("[install] depth-decay prefill chunking active")
    if install_gemma4_nosync() and _gemma4_target(model):
        _log("[install] gemma-4 host-sync-free masks/rope offsets active")
    if install_gemma4_batched_sdpa() and _gemma4_target(model):
        _log("[install] gemma-4 hd512 batched-decode row route active")
    if install_gemma4_softcap_f32() and _gemma4_target(model):
        _log("[install] gemma-4 float32 logit softcap active")
    if install_cascade_sdpa() and install_cascade_stamp():
        _log("[install] shared-prefix cascade decode route active")
    if install_sparse_sdpa():
        _log("[install] sparse top-k page decode route active (lossy, opt-in)")
    n_fused_moe = install_fused_moe_glu(model)
    if n_fused_moe:
        _log(f"[install] fused mxfp4 MoE GLU decode on {n_fused_moe} layers")
    n_shexp = install_hyv3_shexp_fold(model)
    if n_shexp:
        _log(f"[install] hy3 shared-expert fold on {n_shexp} MoE layers")
    n_fused_qkv = install_fused_qkv(model)
    if n_fused_qkv:
        _log(f"[install] fused QKV decode projection on {n_fused_qkv} layers")
    n_occ = install_occupancy_fuse(model)
    if n_occ:
        _log(f"[install] occupancy fusion (qkv + gate/up decode) on "
             f"{n_occ} modules")
    install_rotating_cache_fix()
    install_arrays_cache_fix()

    # 7. partition by what the constructed model actually defines + load.
    loadlog.stage("loading weights")
    model.eval()
    model_params = {p for p, _ in tree_flatten(model.parameters())}
    loadable = {k: v for k, v in hf_weights.items() if k in model_params}
    redundant = sorted(set(hf_weights.keys()) - set(loadable.keys()))
    if redundant:
        _log(
            f"[load_weights] dropping {len(redundant)} redundant tensors "
            f"(no model slot): {redundant[:3]}..."
        )

    # Cast non-quantized float params (norms, SSM weights, and the F16 matrix
    # weights some conversions ship - e.g. gemma-3n's AltUp/LAuReL/per-layer
    # projections) to the activation dtype so activations flow at one width,
    # avoiding float32 kernel promotion (and mixed-width dtype mismatches) in
    # quantized/regular matmul.
    # Model-types in _FP32_KEEP_BY_MODEL_TYPE pin listed params to float32.
    fp32_keep = _FP32_KEEP_BY_MODEL_TYPE.get(config.get("model_type"), ())
    act_dtype = activation_dtype()
    act_name = activation_dtype_name()
    n_cast = 0
    for k in list(loadable):
        v = loadable[k]
        if v.dtype in (mx.float32, mx.float16) and k not in hf_kquant_meta:
            if fp32_keep and any(s in k for s in fp32_keep):
                if v.dtype != mx.float32:      # e.g. F16 ape tables
                    loadable[k] = v.astype(mx.float32)
                continue
            if v.dtype == act_dtype:
                continue                       # already the activation dtype
            if v.dtype == mx.float16:
                # Same-itemsize f16->bf16 gets buffer-donated into the source
                # view -- a write through the zero-copy file mapping (dropped
                # on read-only maps, leaving f16 bits typed as bf16). The f32
                # hop makes both steps size-changing, so neither can donate.
                v = v.astype(mx.float32)
            loadable[k] = v.astype(act_dtype)
            n_cast += 1
    if n_cast:
        _log(f"[dtype] cast {n_cast} float params (norms etc.) to {act_name}")

    model.load_weights(list(loadable.items()), strict=False)
    _log(
        f"[load_weights] loaded {len(loadable)} / {len(model_params)} model parameters"
    )
    _warm_mmap_residency(model, log=_log, paths=pf.shards,
                         active_before=active_before)

    # DiffusionGemma's denoiser needs a dense float embedding table for its
    # probability-weighted soft-embedding step; dequantize it post-load.
    if config.get("model_type") == "diffusion_gemma":
        _dequantize_diffusion_embedding(model, _log)

    missing = sorted(model_params - set(loadable.keys()))
    if missing:
        loadlog.warn(
            f"WARNING: {len(missing)} model params not loaded: {missing[:5]}..."
        )

    _verify_zero_copy_views(model, owned_names, _log)

    # Fused gated-delta decode kernel - opt-in on MoE, where the sparse
    # per-layer matmuls leave the chain's launch latency exposed. Dense
    # hybrids win too (+1.5% sync decode on qwen3.6-27b; the old dense
    # "wash" was measured under the served async pipeline, which hides
    # launch-valley wins). Fused output is closer to f32 truth than the
    # stock bf16 chain (per-layer max-err ~2x lower), so greedy token
    # drift vs stock is benign tie-flips. Must run after load_weights and
    # the kquant leaf swap: the z/b/a merge snapshots the loaded weights,
    # and its quantized-z guard must see the swapped leaves.
    if config.get("model_type") in (
        "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"
    ):
        _patch_gated_delta_fused_decode(model)

    if config.get("model_type") == "qwen4_exp":
        from gmlx.models.qwen4_exp.model import prepare_runtime

        counts = prepare_runtime(model)
        loadlog.verbose_print(
            f"[patch] qwen4_exp: fused GDN decode on {counts['gdn_fused']} "
            f"layers, b/a matvecs concatenated on {counts['gdn_ba_cat']}")

    if config.get("model_type") in ("deepseek_v4", "deepseek_v4_vl"):
        from gmlx.models.deepseek_v4.model import (
            install_gemv_row_fusion,
            warm_kernel_pipelines,
        )

        n_fused_gemv = install_gemv_row_fusion(
            getattr(model, "language_model", model))
        if n_fused_gemv:
            _log(f"[install] gemv row fusion on {n_fused_gemv} projection pairs")
        t_warm = time.perf_counter()
        n_warm = warm_kernel_pipelines()
        if n_warm:
            _log(
                f"[warm] {n_warm} dsa indexer kernel pipelines compiled "
                f"({time.perf_counter() - t_warm:.2f}s)"
            )

    # 8. tokenizer (synthesized from GGUF metadata), wrapped with the full EOS
    #    set so generation stops on turn-end tokens (e.g. gemma-4 <turn|>).
    loadlog.stage("building tokenizer")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from .tokenizer import load_tokenizer_from_gguf

    template_override = _resolve_chat_template(chat_template)
    # The override is threaded *into* the synthesizer so it's set on the fast
    # tokenizer before turn-end-EOS inference (multi-EOS detection must see the
    # override, not the GGUF template).
    raw_tokenizer = load_tokenizer_from_gguf(
        meta, arch, chat_template_override=template_override
    )
    eos_ids = getattr(raw_tokenizer, "_gguf_eos_token_ids", None)
    tokenizer = TokenizerWrapper(raw_tokenizer, eos_token_ids=eos_ids)
    _detect_xtml_thinking(tokenizer, raw_tokenizer, _log)

    materialize_module_arrays(model)
    wait_for_populate(pf.shards, log=_log)

    # Resident generation re-enters mlx-lm's wired_limit() every turn, and
    # its near-budget warning prints on every entry; cap it at one.
    from gmlx.stream.wired_limit import _install_wired_limit_warn_once

    _install_wired_limit_warn_once()

    return model, config, tokenizer


def _detect_xtml_thinking(tokenizer, raw_tokenizer, log) -> None:
    """Complete mlx-lm's thinking detection for XTML-channel templates.

    TokenizerWrapper infers thinking support from vocab token pairs like
    <think>/</think>. Kimi-K3 gates thinking on an XTML channel
    (<|open|>think<|sep|> ... <|close|>think<|sep|>) where "think" is
    plain text between structural tokens, so the inference misses it and
    apply_chat_template injects enable_thinking=False into a template
    whose own default is thinking on. The model then opens the response
    channel immediately and never thinks. Detect the gate in the template
    text and set the wrapper's think markers so has_thinking flips True
    and the injected default becomes enable_thinking=True.

    Detection renders the template's own default generation prompt (raw
    tokenizer, no wrapper injection) and checks it ends with the XTML
    think-open. Matching on rendered output rather than template source
    covers both spellings in the wild (otag('think') in GGUF-embedded
    templates, open_tag('think') in the llama.cpp jinja)."""
    if tokenizer.has_thinking:
        return
    if not getattr(raw_tokenizer, "chat_template", None):
        return
    start, end = "<|open|>think<|sep|>", "<|close|>think<|sep|>"
    try:
        rendered = raw_tokenizer.apply_chat_template(
            [{"role": "user", "content": "probe"}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception:
        return
    if not (isinstance(rendered, str) and rendered.endswith(start)):
        return
    try:
        start_ids = raw_tokenizer.encode(start, add_special_tokens=False)
        end_ids = raw_tokenizer.encode(end, add_special_tokens=False)
    except Exception:
        return
    tokenizer._think_start = start
    tokenizer._think_end = end
    tokenizer._think_start_tokens = tuple(start_ids)
    tokenizer._think_end_tokens = tuple(end_ids)
    log(
        "[tokenizer] XTML think channel detected; "
        "enable_thinking defaults on"
    )


def _resolve_chat_template(chat_template: str | None) -> str | None:
    """Accept an inline Jinja string or a path to a ``.jinja``/``.txt`` file.

    Fails loudly on the two silent-garbage paths: a mistyped file path would
    otherwise be rendered as the literal "template", and a malformed template
    would surface only at generation time as a raw Jinja traceback."""
    if chat_template is None:
        return None
    if os.path.isfile(chat_template):
        with open(chat_template, "r") as f:
            chat_template = f.read()
    elif chat_template.endswith((".jinja", ".txt")) and "{" not in chat_template:
        raise ValueError(f"chat template file not found: {chat_template!r}")
    try:
        import jinja2
    except ImportError:  # validated later by apply_chat_template instead
        return chat_template
    try:
        jinja2.Environment().parse(chat_template)
    except jinja2.TemplateSyntaxError as e:
        raise ValueError(
            f"chat template is not valid Jinja (line {e.lineno}: {e.message})"
        ) from e
    return chat_template


def _load_config_from_source(hf_source: str) -> dict:
    """Load a ``config.json`` from a local dir or HF id (override path)."""
    import json

    cfg_path = os.path.join(hf_source, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    from gmlx.serve.hf_cache import network_fetch_allowed
    with network_fetch_allowed():
        path = hf_hub_download(hf_source, "config.json")
    with open(path, "r") as f:
        return json.load(f)
