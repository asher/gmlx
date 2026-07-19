"""Synthesize an HF-equivalent config dict from GGUF metadata.

`synthesize_config(meta, tensor_shapes)` returns a dict ready to feed into
mlx_lm's `_get_classes(config)` + `ModelArgs.from_dict(config)` - i.e. the same
shape that `load_config()` produces from a real `config.json`. Caller's only job
is to map the GGUF arch string ("gemma4", "qwen35", ...) to mlx_lm's
`model_type` ("gemma4_text", "qwen3_5", ...); that mapping lives here.

Inputs come from `mlx_kquant.load_gguf` (the C++ GGUF loader):
  - `meta`: decoded GGUF KV metadata, `key -> int | float | bool | str | list`.
  - `tensor_shapes`: `tensor name -> logical shape list` in GGUF native
    (innermost-first) order - the same order gguf-py's ReaderTensor.shape uses.

Supported arches (have per-arch field synthesis):
  - gemma4  -> gemma4_text
  - qwen3   -> qwen3
  - qwen35  -> qwen3_5
  - qwen35moe -> qwen3_5_moe   (paired with `_UNWRAP_TO_TEXT` in the caller)
  - mistral3 -> ministral3
  - nemotron_h_moe -> nemotron_h
  - llama   -> llama           (Llama-2/3, Mistral-7B-as-llama, Vicuna, ...)

Other archs raise NotImplementedError; the universal fields alone are
generally not enough.

This module is intentionally numpy/mlx-free: it operates on plain Python
dicts. Mirror of the design constraints in `remap.py`.
"""

from __future__ import annotations

import functools
import math
from typing import Any

from . import loadlog

# The dual-mode GGUF KV readers (decoded KV dict or gguf-py GGUFReader) live
# in gguf_meta; aliased to keep the per-arch synthesizers' call sites short.
from .gguf_meta import (
    array_len as _array_len,
    is_reader as _is_reader,
    read_bool as _read_bool,
    read_bool_array as _read_bool_array,
    read_float as _read_float,
    read_float_array as _read_float_array,
    read_int as _read_int,
    read_int_array as _read_int_array,
    read_string as _read_string,
    scalar as _scalar,
)


GGUF_ARCH_TO_MODEL_TYPE = {
    "gemma4": "gemma4_text",
    # gemma-3n (E2B/E4B) text tower; the loader builds gemma3n.LanguageModel
    # directly from a nested `text_config`, so model_type is the bare arch.
    "gemma3n": "gemma3n",
    "gemma3": "gemma3_text",
    # EmbeddingGemma: gemma3 backbone run as a bidirectional encoder; built into
    # the mlx-embeddings gemma3_text.Model by build_model's gemma_embedding branch.
    "gemma-embedding": "gemma_embedding",
    "qwen35": "qwen3_5",
    "qwen35moe": "qwen3_5_moe",
    "qwen3": "qwen3",
    "qwen3moe": "qwen3_moe",
    # Qwen3-VL / Qwen3-Omni text tower. Structurally a Qwen3-MoE (identical
    # tensor layout: qk-norm + stacked routed experts, no shared expert) plus
    # M-RoPE position sections. Text-only generation degenerates M-RoPE to plain
    # 1-D RoPE, so it loads as qwen3_moe; the vision/audio towers (a separate
    # mmproj) ride the VLM path.
    "qwen3vlmoe": "qwen3_moe",
    "qwen2": "qwen2",
    "qwen2moe": "qwen2_moe",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "phi3": "phi3",
    "glm4": "glm4",
    "llama": "llama",
    # SmolLM3 (HF): Llama backbone with NoPE on every 4th layer.
    "smollm3": "smollm3",
    # IBM Granite 3.x/4.x dense: Llama backbone + runtime scalar multipliers.
    "granite": "granite",
    # llama.cpp 'mistral3' arch covers both Mistral-Small-3.1 and Ministral-3.
    # In mlx_lm both deserialize as model_type='ministral3' (LlamaModel with
    # yarn rope + llama-4-style attention temperature scaling).
    "mistral3": "ministral3",
    "nemotron_h_moe": "nemotron_h",
    # llama.cpp's 'deepseek2' arch covers DeepSeek-V2/V3/R1 and the GLM-4.x MLA
    # conversions (e.g. GLM-4.7-Flash). We target the V3-style variant.
    "deepseek2": "deepseek_v3",
    # GLM-5.2 (llama.cpp 'glm-dsa'): DeepSeek-V3.2 - MLA + fine-grained MoE + a DSA
    # "lightning indexer" + an MTP/nextn layer. mlx-lm's glm_moe_dsa subclasses
    # deepseek_v32 (which adds the Indexer to the deepseek_v3 backbone).
    "glm-dsa": "glm_moe_dsa",
    # DeepSeek V4 Flash (dwarfstar/antirez GGUF; not a llama.cpp arch): single
    # shared KV latent + grouped low-rank output proj, per-layer local/
    # compressed/sparse-indexed attention (compress_ratios), hyper-connections,
    # sqrt-softplus MoE with leading hash-routed layers, QAT simulation
    # round-trips. Model class vendored from mlx-lm PR #1192; the MTP layer
    # ships in a separate companion GGUF (arch deepseek4_mtp_support).
    "deepseek4": "deepseek_v4",
    # GLM-4.5 / 4.6 (incl. GLM-4.5-Air): MHA + deepseek-V3-style fine-grained MoE.
    "glm4moe": "glm4_moe",
    # OpenAI gpt-oss (20B / 120B): MoE with attention sinks, sliding/full
    # alternating attention, YaRN rope, and MXFP4 experts.
    "gpt-oss": "gpt_oss",
    # ByteDance Seed-OSS 36B: plain Llama-shaped dense, explicit head_dim.
    "seed_oss": "seed_oss",
    # Baidu ERNIE-4.5-MoE (21B-A3B): fine-grained MoE + shared expert behind
    # leading dense layers; softmax or sigmoid (aux-free) gating.
    "ernie4_5-moe": "ernie4_5_moe",
    # MiniMax-M2 (230B-A10B): every-layer fine-grained sigmoid-gated MoE (no
    # dense layers, no shared expert), full attention + full-width qk-norm,
    # partial rotary.
    "minimax-m2": "minimax",
    # MiniMax-M3 (428B-A23B): M2's GQA base plus gemma +1 norms (unbaked on
    # load), per-head qk-norm, DeepSeek-V3-style MoE (leading dense layers,
    # shared expert, sigmoid + correction bias, routed scaling) and SwiGLU-OAI.
    # MSA sparse attention runs as dense (GGUFs ship no indexer tensors; same
    # fallback as llama.cpp). Model class vendored from mlx-lm PR #1401.
    "minimax-m3": "minimax_m3",
    # Tencent Hunyuan-A13B: softmax-gated fine-grained MoE + per-layer shared
    # expert, per-head qk-norm, NTK-alpha rope.
    "hunyuan-moe": "hunyuan",
    # Tencent Hy3 (299B-A21B): sigmoid-gated fine-grained MoE + selection-only
    # expert bias + ungated shared expert behind one leading dense layer,
    # per-head qk-norm, plain rope. Native MTP/NextN block past the trunk.
    # Model class vendored from mlx-lm PR #1485.
    "hy_v3": "hy_v3",
    # IBM Granite 4.x hybrid (H-Micro/H-Tiny/H-Small): alternating Mamba2 +
    # attention layers, softmax MoE + fused-input shared MLP, granite multipliers.
    "granitehybrid": "granitemoehybrid",
    # TII Falcon-H1: parallel attention + Mamba2 in every layer, dense MLP,
    # muP multipliers folded into the weights at convert (synth pins neutral).
    "falcon-h1": "falcon_h1",
    # Qwen3-Next 80B-A3B: gated-DeltaNet linear attention (3 of 4 layers) +
    # gated full attention, every-layer 512-expert MoE + shared expert.
    # Legacy fused-ssm_in GGUFs only (split attn_qkv layout rejected loudly).
    "qwen3next": "qwen3_next",
    # DiffusionGemma (26B-A4B): an encoder-decoder block-diffusion model on the
    # Gemma-4 MoE stack. Non-autoregressive - the decoder denoises a fixed-length
    # canvas over reverse-diffusion steps. The model class lives in mlx-vlm
    # (diffusion_gemma); the text config nests under `text_config`.
    "diffusion-gemma": "diffusion_gemma",
}

# Arches that synthesize_config() actually produces complete configs for.
# Derived from the `_SYNTH` dispatch table at the bottom of this module, so an
# arch can never be "supported" without a synthesizer (or vice versa). Other
# archs in the table above are accepted as model_type targets but the
# synthesizer raises NotImplementedError - they need a per-arch extension.
def _supported() -> frozenset:
    return frozenset(_SYNTH)


# Arches with a complete synthesizer + mlx-lm model + remap that the load gate
# deliberately refuses, because no correctly-converted GGUF is known to exist.
# They stay fully wired (in the `_SYNTH` dispatch and remap) so they re-enable
# the instant a fixed file appears; the value is the reason the gate reports.
# `supported_arches()` subtracts them, so coverage and the no-override load
# path never count them as usable.
DISABLED_ARCHES: dict[str, str] = {
    "gemma3n": (
        "no correctly-converted GGUF is known to exist: the laurel low-rank "
        "residual weights are written as a degenerate +/-2.0 constant during "
        "conversion (laurel is never quantized, so every quant level is "
        "affected), which makes the model emit NaN / garbage output under any "
        "runtime, including llama.cpp itself. The arch is otherwise fully "
        "implemented and re-enables automatically once a correctly-converted "
        "file is available."
    ),
}


def supported_arches() -> frozenset:
    """GGUF arch strings the loader builds a model for with no ``hf_source``
    override - those with a synthesizer that aren't gate-disabled.

    Single source of truth for "loadable today"; coverage and the gate derive
    from this rather than hand-keeping a parallel flag. ``DISABLED_ARCHES`` are
    subtracted: they have a synthesizer but no known-good GGUF, so they aren't
    loadable even though the code is present.
    """
    return _supported() - DISABLED_ARCHES.keys()


# Tensor-inventory probes (for fields not in KV metadata)

def _tensor_shapes(reader) -> dict[str, list[int]]:
    """Map tensor name -> integer shape list (GGUF native order). Used only for
    the legacy gguf-py-reader path; the dict path passes tensor_shapes in."""
    return {t.name: [int(x) for x in t.shape] for t in reader.tensors}


def _has_tensor(shapes: dict[str, list[int]], name: str) -> bool:
    return name in shapes


# Public entry point

def _zero_count_guard(fn):
    """A corrupt GGUF declaring a zero head/expert/layer count reaches a
    division somewhere in per-arch synthesis; surface a named refusal
    instead of a ZeroDivisionError traceback."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ZeroDivisionError as e:
            raise ValueError(
                "GGUF metadata declares a zero head/expert/dim count - "
                "corrupt or hand-edited file?") from e
    return wrapped


@_zero_count_guard
def synthesize_config(meta, tensor_shapes=None) -> dict[str, Any]:
    """Build a config dict from GGUF metadata.

    Args:
        meta: either a decoded GGUF KV metadata dict (from
            mlx_kquant.load_gguf) or a gguf-py GGUFReader (legacy callers).
        tensor_shapes: tensor name -> logical shape list (GGUF native order).
            Optional when `meta` is a reader - derived from `meta.tensors`.

    Returns a dict shaped like a HuggingFace `config.json` for the relevant
    text-model class, ready to feed into mlx_lm's `_get_classes(config)` +
    `ModelArgs.from_dict(config)`.
    """
    if tensor_shapes is None:
        if not _is_reader(meta):
            raise ValueError(
                "tensor_shapes is required when meta is a decoded dict")
        tensor_shapes = _tensor_shapes(meta)

    arch = _read_string(meta, "general.architecture")
    if arch is None:
        raise ValueError("GGUF missing 'general.architecture' KV field")

    model_type = GGUF_ARCH_TO_MODEL_TYPE.get(arch)
    if model_type is None:
        raise ValueError(
            f"unsupported GGUF arch {arch!r}; extend GGUF_ARCH_TO_MODEL_TYPE")

    synth = _SYNTH.get(arch)
    if synth is None:
        raise NotImplementedError(
            f"config synthesis for arch {arch!r} not implemented "
            f"(GGUF_ARCH_TO_MODEL_TYPE has it, but no per-arch extension exists)")

    shapes = tensor_shapes
    config: dict[str, Any] = {"model_type": model_type}

    _add_universal_fields(meta, shapes, config, arch)
    synth(meta, shapes, config)

    _print_summary(config, arch)
    return config


# Universal field extraction

def _require(value, *, arch: str, gguf_field: str):
    if value is None:
        raise ValueError(
            f"config synth: missing GGUF field {gguf_field!r} for arch {arch!r}")
    return value


def _first_or_scalar(meta, key: str) -> int | None:
    """Read a field that may be a scalar or a per-layer array; return the
    first element (which should be uniform across layers for the targets
    we support - verified for gemma-4 dense `feed_forward_length` and
    `head_count_kv`)."""
    s = _scalar(meta, key)
    return None if s is None else int(s)


def _add_universal_fields(meta, shapes, config: dict, arch: str) -> None:
    config["hidden_size"] = _require(
        _read_int(meta, f"{arch}.embedding_length"),
        arch=arch, gguf_field=f"{arch}.embedding_length")
    block_count = _require(
        _read_int(meta, f"{arch}.block_count"),
        arch=arch, gguf_field=f"{arch}.block_count")
    nextn_layers = _read_int(meta, f"{arch}.nextn_predict_layers") or 0
    config["num_hidden_layers"] = block_count - nextn_layers
    # Expose the native MTP (multi-token-prediction) head depth for the
    # speculative-decoding drafter build. Only set when present so the text
    # config stays unchanged for non-MTP GGUFs; the text model class ignores it.
    if nextn_layers > 0:
        config["mtp_num_hidden_layers"] = nextn_layers
    config["num_attention_heads"] = _require(
        _read_int(meta, f"{arch}.attention.head_count"),
        arch=arch, gguf_field=f"{arch}.attention.head_count")
    # head_count_kv is per-layer on some MoE configs; first element is uniform.
    config["num_key_value_heads"] = _require(
        _first_or_scalar(meta, f"{arch}.attention.head_count_kv"),
        arch=arch, gguf_field=f"{arch}.attention.head_count_kv")

    # feed_forward_length: scalar on most arches, per-layer array on gemma-4.
    intermediate = _first_or_scalar(meta, f"{arch}.feed_forward_length")
    if intermediate is not None:
        config["intermediate_size"] = intermediate

    ctx = _read_int(meta, f"{arch}.context_length")
    if ctx is not None:
        config["max_position_embeddings"] = ctx

    eps = _read_float(meta, f"{arch}.attention.layer_norm_rms_epsilon")
    if eps is not None:
        config["rms_norm_eps"] = eps

    # Default head_dim from key_length; arch-specific code may override
    # (gemma-4 uses different full-attn vs sliding head_dims).
    key_length = _read_int(meta, f"{arch}.attention.key_length")
    if key_length is not None:
        config["head_dim"] = key_length

    # Default rope_theta; arch-specific code may build a richer
    # `rope_parameters` dict on top.
    freq_base = _read_float(meta, f"{arch}.rope.freq_base")
    if freq_base is not None:
        config["rope_theta"] = freq_base

    # tied embeddings: present when the GGUF has no separate output.weight.
    config["tie_word_embeddings"] = not _has_tensor(shapes, "output.weight")

    # vocab size
    vocab_size = _array_len(meta, "tokenizer.ggml.tokens")
    if vocab_size is not None:
        config["vocab_size"] = vocab_size


# gemma4

def _synth_gemma4(meta, shapes, config: dict, arch: str = "gemma4") -> None:
    # `arch` is the GGUF metadata-key prefix. It is "gemma4" for a plain Gemma-4
    # checkpoint, but DiffusionGemma reuses this same Gemma-4 MoE backbone synth
    # under its own "diffusion-gemma" metadata prefix.

    # Asymmetric K/V head dims (full vs SWA).
    full_kdim = _require(
        _read_int(meta, f"{arch}.attention.key_length"),
        arch=arch, gguf_field=f"{arch}.attention.key_length")
    swa_kdim = _read_int(meta, f"{arch}.attention.key_length_swa")
    config["global_head_dim"] = full_kdim
    if swa_kdim is not None:
        # mlx_lm gemma4_text uses `head_dim` as the sliding-attention dim
        # and `global_head_dim` for full-attention layers.
        config["head_dim"] = swa_kdim

    config["num_kv_shared_layers"] = _require(
        _read_int(meta, f"{arch}.attention.shared_kv_layers"),
        arch=arch, gguf_field=f"{arch}.attention.shared_kv_layers")

    sliding = _read_int(meta, f"{arch}.attention.sliding_window")
    if sliding is not None:
        config["sliding_window"] = sliding

    softcap = _read_float(meta, f"{arch}.final_logit_softcapping")
    if softcap is not None:
        config["final_logit_softcapping"] = softcap

    hspl = _read_int(meta, f"{arch}.embedding_length_per_layer_input")
    if hspl is not None:
        config["hidden_size_per_layer_input"] = hspl

    # layer_types from the per-layer bool array.
    pattern = _read_bool_array(meta, f"{arch}.attention.sliding_window_pattern")
    if pattern is not None:
        config["layer_types"] = [
            "sliding_attention" if v else "full_attention" for v in pattern]

    # K-eq-V detection: 26B/31B-class models drop the V projection on
    # full-attention layers and reuse K as V (mlx_lm gemma4_text.Attention
    # gates this on `attention_k_eq_v`). Detect by tensor inventory: if any
    # full-attention layer lacks `attn_v.weight`, the model is K-eq-V and
    # the surviving `attn_k.weight` carries n_global_kv_heads x head_dim.
    if pattern is not None:
        full_indices = [i for i, v in enumerate(pattern) if not v]
        if full_indices:
            i = full_indices[0]
            has_v = _has_tensor(shapes, f"blk.{i}.attn_v.weight")
            if not has_v:
                config["attention_k_eq_v"] = True
                k_shape = shapes.get(f"blk.{i}.attn_k.weight")
                if k_shape is None:
                    raise ValueError(
                        f"gemma4 synth: full-attn layer {i} missing both "
                        f"attn_v and attn_k tensors")
                if k_shape[1] % full_kdim != 0:
                    raise ValueError(
                        f"gemma4 synth: full-attn k_proj cols {k_shape[1]} "
                        f"not divisible by global_head_dim {full_kdim}")
                config["num_global_key_value_heads"] = k_shape[1] // full_kdim

    # rope_parameters: full-attn + sliding-attn sub-dicts.
    full_freq = _read_float(meta, f"{arch}.rope.freq_base")
    swa_freq = _read_float(meta, f"{arch}.rope.freq_base_swa")
    if full_freq is not None or swa_freq is not None:
        rp: dict[str, Any] = {}
        if full_freq is not None:
            rp["full_attention"] = {
                "partial_rotary_factor": 0.25,
                "rope_theta": full_freq,
                "rope_type": "proportional",
            }
        if swa_freq is not None:
            rp["sliding_attention"] = {
                "partial_rotary_factor": 1.0,
                "rope_theta": swa_freq,
                "rope_type": "default",
            }
        config["rope_parameters"] = rp

    # use_double_wide_mlp: True when kv-shared layers' MLP is 2x the base
    # intermediate size. With num_kv_shared_layers=0, no kv-shared layers
    # exist and the flag is functionally inert - match HF's convention of
    # `False` in that case.
    n_layers = config["num_hidden_layers"]
    n_shared = config["num_kv_shared_layers"]
    if n_shared > 0:
        first_shared = n_layers - n_shared
        base_shape = shapes.get("blk.0.ffn_gate.weight")
        shared_shape = shapes.get(f"blk.{first_shared}.ffn_gate.weight")
        if base_shape is None or shared_shape is None:
            raise ValueError(
                f"gemma4 synth: missing ffn_gate tensors needed to detect "
                f"use_double_wide_mlp (base={base_shape}, "
                f"shared={shared_shape})")
        config["use_double_wide_mlp"] = shared_shape[1] == 2 * base_shape[1]
    else:
        config["use_double_wide_mlp"] = False

    # MoE: enable iff expert_count is present.
    expert_count = _read_int(meta, f"{arch}.expert_count")
    if expert_count is not None:
        config["enable_moe_block"] = True
        config["num_experts"] = expert_count
        config["top_k_experts"] = _require(
            _read_int(meta, f"{arch}.expert_used_count"),
            arch=arch, gguf_field=f"{arch}.expert_used_count")
        config["moe_intermediate_size"] = _require(
            _read_int(meta, f"{arch}.expert_feed_forward_length"),
            arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")


# DiffusionGemma (block-diffusion on the Gemma-4 MoE backbone)

def _synth_diffusion_gemma(meta, shapes, config: dict) -> None:
    """Synthesize a DiffusionGemma config and nest the backbone under
    ``text_config`` for the mlx-vlm ``diffusion_gemma`` Model.

    The decoder backbone is a Gemma-4 MoE (each layer runs a dense MLP *and*
    routed experts in parallel), so the Gemma-4 synth produces the text fields
    verbatim - only the GGUF metadata-key prefix differs (``diffusion-gemma.*``).
    On top of the text config this adds the diffusion knobs the mlx-vlm denoising
    engine reads off ``model.config``: the canvas length and an entropy-bound
    sampler ``generation_config`` (steps / temperature schedule / acceptance and
    stop thresholds), each defaulted from GGUF metadata when present.
    """
    arch = "diffusion-gemma"

    # The universal pass + Gemma-4 MoE synth have already populated `config`
    # flat, reading every key under the `diffusion-gemma.*` prefix.
    _synth_gemma4(meta, shapes, config, arch=arch)

    text_config = dict(config)
    text_config["model_type"] = "diffusion_gemma_text"
    # mlx-lm-only knobs the mlx-vlm TextConfig doesn't declare; harmless if left
    # (unknown keys are dropped at from_dict) but cleaner to omit.
    text_config.pop("enable_moe_block", None)
    text_config.pop("mtp_num_hidden_layers", None)

    # Entropy-bound denoiser sampler config. The upstream model ships these in
    # `generation_config.json`, but the llama.cpp GGUF conversion drops them
    # (it carries only `diffusion.canvas_length` + the scalar eos), so each is
    # read from `diffusion.eb_*` when present and otherwise filled from the
    # canonical DiffusionGemma default below. Baking the defaults is load-bearing
    # for both reported behaviours: without `diffusion_stopping_config` the
    # denoiser never early-stops and runs the full `max_denoising_steps` schedule
    # every canvas (~3-4x slower decode), and without the turn-end token in the
    # eos set the canvas never stops on `<end_of_turn>` and runs to max_tokens.
    gen_cfg: dict[str, Any] = {}

    # DiffusionGemma stops a turn on <eos> (1), <end_of_turn> (106) and
    # <end_of_tool_response> (50) - fixed ids in the Gemma-4 tokenizer it ships
    # with. The GGUF gives only the scalar eos, so the turn-end ids are added
    # here; the mlx-vlm engine's `add_eos_token_ids` folds this list into the
    # stop set on every path (run/chat and the server diffusion lane).
    eos = _read_int(meta, "tokenizer.ggml.eos_token_id")
    eos_ids = [1, 106, 50]
    if eos is not None and eos not in eos_ids:
        eos_ids.insert(0, eos)
    gen_cfg["eos_token_id"] = eos_ids

    max_steps = _read_int(meta, "diffusion.eb_max_steps")
    gen_cfg["max_denoising_steps"] = max_steps if max_steps is not None else 48

    t_min = _read_float(meta, "diffusion.eb_t_min")
    t_max = _read_float(meta, "diffusion.eb_t_max")
    gen_cfg["linear_temperature_schedule_config"] = {
        "t_min": t_min if t_min is not None else 0.4,
        "t_max": t_max if t_max is not None else 0.8,
    }

    entropy_bound = _read_float(meta, "diffusion.eb_entropy_bound")
    gen_cfg["sampler_config"] = {
        "_cls_name": "EntropyBoundSamplerConfig",
        "entropy_bound": entropy_bound if entropy_bound is not None else 0.1,
    }

    conf = _read_float(meta, "diffusion.eb_confidence_threshold")
    stab = _read_int(meta, "diffusion.eb_stability_threshold")
    gen_cfg["diffusion_stopping_config"] = {
        "confidence_threshold": conf if conf is not None else 0.005,
        "stability_threshold": stab if stab is not None else 1,
    }

    config.clear()
    config["model_type"] = "diffusion_gemma"
    config["text_config"] = text_config
    # Top-level eos as well as in generation_config: mlx-vlm's diffusion server
    # lane resets the stop criteria off ``model.config.eos_token_id`` directly.
    config["eos_token_id"] = eos_ids
    canvas_length = _read_int(meta, "diffusion.canvas_length")
    if canvas_length is not None:
        config["canvas_length"] = canvas_length
    config["generation_config"] = gen_cfg


# gemma4 assistant drafter (two-GGUF MTP companion)

# The assistant drafter is its own small GGUF (arch ``gemma4_mtp`` for a MoE
# target, ``gemma4-assistant`` for a dense target). Both carry a tiny dense
# gemma4 stack (block_count layers, own ``embedding_length`` hidden) plus two
# projections bridging the drafter hidden to the target's backbone hidden, and a
# shared output norm. The drafter reuses the target's K/V (no attn_k/attn_v in
# the GGUF) and the target's input embedding; its own token_embd is the tied
# output head. mlx-vlm builds it from a single ``gemma4_assistant`` class for
# both target kinds - the dense/MoE difference is just ``backbone_hidden_size``.

@_zero_count_guard
def synthesize_gemma4_assistant_config(meta, tensor_shapes=None) -> dict[str, Any]:
    """Build the ``Gemma4AssistantConfig`` dict for the mlx-vlm gemma4 drafter
    from a drafter GGUF's own metadata. Data-driven, so the same path serves the
    E2B / E4B / 12B (dense) and 26B-A4B (MoE) assistants - only the dims differ.
    """
    if tensor_shapes is None:
        if not _is_reader(meta):
            raise ValueError("tensor_shapes is required when meta is a dict")
        tensor_shapes = _tensor_shapes(meta)

    arch = _read_string(meta, "general.architecture")
    if arch is None:
        raise ValueError("drafter GGUF missing 'general.architecture'")

    def gi(suffix):
        return _read_int(meta, f"{arch}.{suffix}")

    def gf(suffix):
        return _read_float(meta, f"{arch}.{suffix}")

    hidden = _require(gi("embedding_length"), arch=arch,
                      gguf_field=f"{arch}.embedding_length")
    n_layers = _require(gi("block_count"), arch=arch,
                        gguf_field=f"{arch}.block_count")
    ffn = _require(gi("feed_forward_length"), arch=arch,
                   gguf_field=f"{arch}.feed_forward_length")
    n_head = _require(gi("attention.head_count"), arch=arch,
                      gguf_field=f"{arch}.attention.head_count")
    # Target backbone hidden - three names across the assistant variants:
    # gemma4_mtp=backbone_embedding_length, gemma4-assistant=embedding_length_out,
    # gemma4_assistant (E2B/E4B)=n_embd_backbone.
    backbone = (gi("backbone_embedding_length")
                or gi("embedding_length_out")
                or gi("n_embd_backbone"))
    if backbone is None:
        raise ValueError(
            f"{arch}: missing target hidden size (backbone_embedding_length / "
            f"embedding_length_out / n_embd_backbone)")

    pattern = _read_bool_array(meta, f"{arch}.attention.sliding_window_pattern")
    kv = _read_int_array(meta, f"{arch}.attention.head_count_kv") or [n_head]
    if pattern is not None:
        # layer_types is per-layer and must be derived from the pattern whether or
        # not head_count_kv is per-layer: the last drafter layer is global
        # (full_attention) and needs global_head_dim, which the attention selects
        # via layer_types[idx]. head_count_kv is sometimes a single scalar while
        # the pattern is a full per-layer bool list, so gating layer_types on
        # len(kv)==len(pattern) would drop it and force the global layer onto the
        # sliding head_dim - a crash when key_length != key_length_swa.
        lt = ["sliding_attention" if v else "full_attention" for v in pattern]
        layer_types = (lt * (n_layers // len(lt) + 1))[:n_layers]
        if len(kv) == len(pattern):
            sliding_kv = next((kv[i] for i, v in enumerate(pattern) if v), kv[0])
            global_kv = next((kv[i] for i, v in enumerate(pattern) if not v),
                             kv[-1])
        else:
            sliding_kv = global_kv = kv[0]
    else:
        sliding_kv = global_kv = kv[0]
        layer_types = None

    te = tensor_shapes.get("token_embd.weight")
    vocab = int(te[1]) if te else None

    text_config: dict[str, Any] = {
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "intermediate_size": ffn,
        "num_attention_heads": n_head,
        "num_key_value_heads": sliding_kv,
        "num_global_key_value_heads": global_kv,
        "head_dim": gi("attention.key_length_swa") or 256,
        "global_head_dim": gi("attention.key_length") or 512,
        # The drafter reuses the target's K/V for every layer.
        "num_kv_shared_layers": gi("attention.shared_kv_layers") or n_layers,
        "sliding_window": gi("attention.sliding_window"),
        "rms_norm_eps": gf("attention.layer_norm_rms_epsilon") or 1e-6,
        # The assistant has no per-layer-input embeddings (PLE off) - leaving the
        # default on would build per_layer_* params the GGUF doesn't carry.
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
        "tie_word_embeddings": True,
    }
    if vocab is not None:
        text_config["vocab_size"] = vocab
    if layer_types is not None:
        text_config["layer_types"] = layer_types
    softcap = gf("final_logit_softcapping")
    if softcap is not None:
        text_config["final_logit_softcapping"] = softcap

    full_freq = gf("rope.freq_base")
    swa_freq = gf("rope.freq_base_swa")
    if full_freq is not None or swa_freq is not None:
        rp: dict[str, Any] = {}
        if full_freq is not None:
            rp["full_attention"] = {"partial_rotary_factor": 0.25,
                                    "rope_theta": full_freq,
                                    "rope_type": "proportional"}
        if swa_freq is not None:
            rp["sliding_attention"] = {"partial_rotary_factor": 1.0,
                                       "rope_theta": swa_freq,
                                       "rope_type": "default"}
        text_config["rope_parameters"] = rp

    block_size = gi("nextn_predict_layers") or n_layers
    cfg = {
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": backbone,
        "block_size": block_size,
        "tie_word_embeddings": True,
        "text_config": text_config,
    }
    # Ordered-embeddings (centroid-routed sparse head) variant - E2B/E4B. The
    # drafter then builds a MaskedEmbedder (mtp.centroids / mtp.token_ordering)
    # instead of a tied head; num_centroids comes from the centroids tensor.
    if _read_bool(meta, f"{arch}.use_ordered_embeddings"):
        cfg["use_ordered_embeddings"] = True
        cen = tensor_shapes.get("mtp.centroids.weight")
        if cen is not None:
            cfg["num_centroids"] = int(cen[1])
    return cfg


# qwen3

def _synth_qwen3(meta, shapes, config: dict) -> None:
    """Synthesize a qwen3 config. Universal fields cover everything; this
    function handles rope_scaling passthrough so mlx_lm's initialize_rope
    doesn't see unexpected keys."""
    arch = "qwen3"
    rope_scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None and rope_scaling_type != "none":
        rp: dict[str, Any] = {"type": rope_scaling_type, "rope_type": rope_scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp


# qwen3moe (Qwen3-MoE: Qwen3-30B-A3B, Qwen3-235B-A22B)

def _synth_qwen3moe(meta, shapes, config: dict, arch: str = "qwen3moe") -> None:
    """Synthesize a qwen3_moe config. Backbone (hidden/layers/heads/kv/eps/vocab/
    head_dim-from-key_length/rope_theta/ctx/tie) comes from the universal fields;
    this adds the rope-scaling passthrough (same as qwen3) and the MoE block.

    GGUF doesn't carry ``decoder_sparse_step`` / ``mlp_only_layers``; Qwen3-MoE is
    all-MoE (step 1, no dense-only layers), which is the default emitted here. The
    routed experts are stored already-stacked (``ffn_*_exps``) and remap straight
    onto mlx-lm's ``mlp.switch_mlp.*`` (no shared expert, unlike qwen35moe).

    ``arch`` selects the GGUF metadata key prefix: ``qwen3moe`` for Qwen3-MoE, or
    ``qwen3vlmoe`` for the Qwen3-VL / Qwen3-Omni text tower (same MoE layout, read
    as text-only - its M-RoPE sections collapse to 1-D RoPE without an image)."""
    rope_scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None and rope_scaling_type != "none":
        rp: dict[str, Any] = {"type": rope_scaling_type,
                              "rope_type": rope_scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp

    config["num_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    config["moe_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
    # All layers MoE on the shipped Qwen3-MoE checkpoints.
    config["decoder_sparse_step"] = 1
    config["mlp_only_layers"] = []
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm


# qwen2moe (Qwen1.5-MoE-A2.7B)

def _synth_qwen2moe(meta, shapes, config: dict) -> None:
    """Synthesize a qwen2_moe config (Qwen1.5-MoE). Universal fields cover the
    backbone (hidden/layers/heads/kv/intermediate/eps/vocab/rope_theta/ctx/tie);
    head_dim is derived inside mlx-lm as hidden//heads (no config key), and q/k/v
    are always built with bias=True (the GGUF's QKV bias tensors remap onto them).

    All layers are MoE (no dense-only layers, no decoder_sparse_step concept in
    mlx-lm's qwen2_moe) with both routed experts (switch_mlp) and a single shared
    expert. Two sizes need care because the public Qwen1.5-MoE GGUFs don't write
    the expert-width KV keys:

      - moe_intermediate_size: prefer ``expert_feed_forward_length`` if present,
        else derive from the stacked ``ffn_gate_exps`` tensor - gguf-py shape is
        ``[hidden, moe_intermediate, n_experts]``, so the middle dim is it.
      - shared_expert_intermediate_size: prefer
        ``expert_shared_feed_forward_length``, else fall back to the dense
        ``feed_forward_length`` (the shared expert's MLP width, e.g. 5632)."""
    arch = "qwen2moe"

    config["num_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")

    moe_inter = _read_int(meta, f"{arch}.expert_feed_forward_length")
    if moe_inter is None:
        gate_exps = shapes.get("blk.0.ffn_gate_exps.weight")
        if gate_exps is None or len(gate_exps) != 3:
            raise ValueError(
                f"qwen2moe synth: missing {arch}.expert_feed_forward_length and "
                f"can't derive moe_intermediate_size - need a stacked 3-D "
                f"blk.0.ffn_gate_exps.weight (got {gate_exps}). Pass hf_source.")
        # gguf-py shape order: [hidden, moe_intermediate, n_experts].
        moe_inter = gate_exps[1]
    config["moe_intermediate_size"] = moe_inter

    shared = _read_int(meta, f"{arch}.expert_shared_feed_forward_length")
    if shared is None:
        shared = _read_int(meta, f"{arch}.feed_forward_length")
    config["shared_expert_intermediate_size"] = _require(
        shared, arch=arch,
        gguf_field=f"{arch}.expert_shared_feed_forward_length / feed_forward_length")


# qwen2

def _synth_qwen2(meta, shapes, config: dict) -> None:
    """Synthesize a qwen2 config. The universal fields cover the backbone
    (hidden/layers/heads/kv_heads/intermediate/eps/vocab/rope_theta/ctx/tie).
    mlx_lm's qwen2 derives head_dim as hidden_size // num_attention_heads and
    always builds q/k/v projections with bias=True, so neither needs a config
    key - the GGUF's QKV bias tensors are remapped straight onto them. This
    adds only the optional yarn/linear rope-scaling passthrough (present on the
    long-context 7B/72B variants; absent on 0.5B/1.5B)."""
    arch = "qwen2"
    rope_scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None and rope_scaling_type != "none":
        rp: dict[str, Any] = {"type": rope_scaling_type,
                              "rope_type": rope_scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        orig_ctx = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig_ctx is not None:
            rp["original_max_position_embeddings"] = orig_ctx
        config["rope_scaling"] = rp


# gemma3 (gemma-3 text models: 1B / 4B / 12B / 27B text tower)

def _synth_gemma3(meta, shapes, config: dict, arch: str = "gemma3") -> None:
    """Synthesize a gemma3_text config from a 'gemma3' GGUF.

    Universal fields cover hidden/layers/heads/kv/intermediate/eps/vocab, plus
    head_dim from `attention.key_length` and the global rope_theta from
    `rope.freq_base`. This adds the gemma3-specific attention scaling, the
    sliding-window interleave, and the local (sliding-layer) rope base.

    ``arch`` is the GGUF metadata-key prefix: "gemma3" for a plain Gemma-3 GGUF,
    "gemma-embedding" when reused for the EmbeddingGemma encoder backbone (same
    gemma3 tensor/KV layout under a different prefix).
    """
    # query_pre_attn_scalar is not written to GGUF KV (attention scale =
    # query_pre_attn_scalar**-0.5). gemma_pytorch / llama.cpp gemma3.cpp rule:
    # 27B (62 layers) uses hidden//heads (5376/32 = 168, head_dim 128); every
    # other size (1B/4B/12B) uses head_dim (256).
    if config.get("head_dim") is not None:
        config["query_pre_attn_scalar"] = (
            config["hidden_size"] // config["num_attention_heads"]
            if config["num_hidden_layers"] == 62 else config["head_dim"])

    sw = _read_int(meta, f"{arch}.attention.sliding_window")
    if sw is not None:
        config["sliding_window"] = sw
    # 1 global layer every `sliding_window_pattern` layers. Omitted at the
    # default (6) in some conversions; mlx_lm's gemma3_text default is also 6.
    swp = _read_int(meta, f"{arch}.attention.sliding_window_pattern")
    if swp is not None:
        config["sliding_window_pattern"] = swp

    # Sliding layers use a separate (smaller) rope base. llama.cpp writes it as
    # `rope.freq_base_swa` when non-default; mlx_lm defaults rope_local_base_freq
    # to 10000 (gemma-3's actual local base), so only override when present.
    local_freq = _read_float(meta, f"{arch}.rope.freq_base_swa")
    if local_freq is not None:
        config["rope_local_base_freq"] = local_freq


# EmbeddingGemma (gemma-embedding: gemma3 backbone as a bidirectional encoder)

def _synth_gemma_embedding(meta, shapes, config: dict) -> None:
    """Synthesize a config for an EmbeddingGemma GGUF (general.architecture
    'gemma-embedding').

    The backbone is a gemma3 text model, so reuse ``_synth_gemma3`` against the
    'gemma-embedding' key prefix (same query scaling / sliding-window interleave /
    local rope). ``model_type`` stays 'gemma_embedding' (set by synthesize_config
    from GGUF_ARCH_TO_MODEL_TYPE) so build_model routes the encoder branch; the
    backbone ModelArgs is mlx-lm gemma3_text's. The mlx-embeddings Model derives
    its mean-pool + 2-layer dense head from hidden_size, so no head dims are
    emitted here. ``pooling_type`` / ``attention.causal`` are surfaced for the
    record (gemma3_text.ModelArgs ignores them); the encoder always mean-pools
    under a full bidirectional mask.
    """
    arch = "gemma-embedding"
    _synth_gemma3(meta, shapes, config, arch=arch)
    pooling = _read_int(meta, f"{arch}.pooling_type")
    if pooling is not None:
        config["pooling_type"] = pooling
    causal = _read_bool(meta, f"{arch}.attention.causal")
    if causal is not None:
        config["attention_causal"] = causal


# gemma3n (gemma-3n E2B / E4B text tower: MatFormer + AltUp + LAuReL)

def _synth_gemma3n(meta, shapes, config: dict) -> None:
    """Synthesize a gemma-3n config and nest it under ``text_config``.

    mlx_lm's gemma3n.LanguageModel takes a flat ``TextConfig`` (~25 fields); the
    loader builds it directly, so this collects the universal backbone fields
    (already on ``config``) plus the gemma-3n-specific knobs into a
    ``text_config`` dict and replaces ``config`` with ``{model_type, text_config}``.

    Two GGUF<->HF conversions are load-bearing:
      - ``sliding_window_pattern`` (1=sliding, 0=full) -> ``layer_types``.
      - ``activation_sparsity_scale`` stores the *precomputed* gelu-topk std
        multiplier m = sqrt(2)*erfinv(2p-1); mlx_lm wants the raw sparsity fraction
        p, so invert it: p = (1 + erf(m/sqrt(2))) / 2 (non-finite/<=0 -> 0.0).

    Fields the GGUF doesn't carry use the architecture defaults:
      - ``rope_local_base_freq`` = 10000.0 (sliding-layer rope base; llama.cpp's
        ``rope_freq_base_train_swa`` default, applied to the 24 sliding layers).
      - ``altup_coef_clip`` = 120.0 (coef clamp; O(1) coefs never reach it).
      - ``final_logit_softcapping`` = 30.0 (the gemma3n constant - llama.cpp's
        hparams default, applied unconditionally in the gemma3n graph, and the
        HF config value; the GGUF omits the key, so default it rather than
        leaving the cap off. Monotonic, so greedy tokens are unchanged, but it
        keeps logits/logprobs faithful to the reference model).
    """
    arch = "gemma3n"

    n_layers = config["num_hidden_layers"]

    # vocab sizes: derive from the embedding tensors so the tied output head and
    # the per-layer-input embedding exactly match the loaded weights (GGUF
    # native shape is [n_embd, n_vocab], so the vocab dim is index 1).
    tok = shapes.get("token_embd.weight")
    vocab_size = tok[1] if tok and len(tok) == 2 else config.get("vocab_size")
    ple = shapes.get("per_layer_token_embd.weight")
    if ple is None or len(ple) != 2:
        raise ValueError(
            "gemma3n synth: missing per_layer_token_embd.weight - needed for "
            "vocab_size_per_layer_input")
    vocab_per_layer = ple[1]

    laurel_l = shapes.get("blk.0.laurel_l.weight")
    if laurel_l is None or len(laurel_l) != 2:
        raise ValueError("gemma3n synth: missing blk.0.laurel_l.weight "
                         "(needed for laurel_rank)")
    laurel_rank = laurel_l[1]

    # layer_types from the sliding-window pattern (1 -> sliding, 0 -> full).
    pattern = _read_int_array(meta, f"{arch}.attention.sliding_window_pattern")
    if pattern is None or len(pattern) != n_layers:
        raise ValueError(
            f"gemma3n synth: {arch}.attention.sliding_window_pattern length "
            f"{None if pattern is None else len(pattern)} != num_hidden_layers "
            f"{n_layers}")
    layer_types = ["sliding_attention" if v else "full_attention"
                   for v in pattern]

    # activation_sparsity_pattern: invert the precomputed std multiplier.
    scales = _read_float_array(meta, f"{arch}.activation_sparsity_scale")
    if scales is not None:
        sparsity = [
            (1.0 + math.erf(m / math.sqrt(2.0))) / 2.0
            if math.isfinite(m) and m > 0 else 0.0
            for m in scales
        ]
    else:
        sparsity = [0.0] * n_layers

    shared_kv = _require(
        _read_int(meta, f"{arch}.attention.shared_kv_layers"),
        arch=arch, gguf_field=f"{arch}.attention.shared_kv_layers")

    # feed_forward_length is per-layer (MatFormer): keep the full list when it
    # varies (e.g. E4B), else the scalar the universal pass already derived
    # (E2B is uniform 8192). mlx_lm's MLP indexes a list or uses a scalar
    # transparently, so both shapes are valid.
    ff = _read_int_array(meta, f"{arch}.feed_forward_length")
    intermediate_size = (ff if ff and len(set(ff)) > 1
                         else config["intermediate_size"])

    text_config: dict[str, Any] = {
        "model_type": "gemma3n_text",
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": n_layers,
        "intermediate_size": intermediate_size,
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],            # key_length == value_length
        "rms_norm_eps": config["rms_norm_eps"],
        "vocab_size": vocab_size,
        "vocab_size_per_layer_input": vocab_per_layer,
        "num_kv_shared_layers": shared_kv,
        "sliding_window": _require(
            _read_int(meta, f"{arch}.attention.sliding_window"),
            arch=arch, gguf_field=f"{arch}.attention.sliding_window"),
        "max_position_embeddings": config.get("max_position_embeddings", 32768),
        "rope_theta": config.get("rope_theta", 1000000.0),
        "rope_local_base_freq": _read_float(meta, f"{arch}.rope.freq_base_swa")
                                or 10000.0,
        "final_logit_softcapping": _read_float(
            meta, f"{arch}.final_logit_softcapping") or 30.0,
        "layer_types": layer_types,
        "activation_sparsity_pattern": sparsity,
        "hidden_size_per_layer_input": _require(
            _read_int(meta, f"{arch}.embedding_length_per_layer_input"),
            arch=arch, gguf_field=f"{arch}.embedding_length_per_layer_input"),
        "altup_num_inputs": _require(
            _read_int(meta, f"{arch}.altup.num_inputs"),
            arch=arch, gguf_field=f"{arch}.altup.num_inputs"),
        "altup_active_idx": _read_int(meta, f"{arch}.altup.active_idx") or 0,
        "altup_coef_clip": 120.0,
        # correct_output_scale tensors are present per layer -> scale is applied.
        "altup_correct_scale": _has_tensor(
            shapes, "blk.0.altup_correct_scale.weight"),
        "laurel_rank": laurel_rank,
        "rope_scaling": None,
    }

    config.clear()
    config["model_type"] = "gemma3n"
    config["text_config"] = text_config


# gemma2 (gemma-2 2B / 9B / 27B)

def _synth_gemma2(meta, shapes, config: dict) -> None:
    """Synthesize a gemma2 config. Universal fields cover hidden/layers/heads/kv/
    intermediate/eps/vocab, plus head_dim from `attention.key_length` (256).
    gemma-2 is always tied (mlx_lm uses embed_tokens.as_linear). This adds the
    attention/final logit softcaps, the sliding window, and query_pre_attn_scalar
    (= head_dim for 2B/9B; hidden//heads for the 27B)."""
    arch = "gemma2"

    # Not in GGUF KV. gemma_pytorch / llama.cpp gemma2.cpp rule: 27B (46 layers)
    # uses hidden//heads (4608/32 = 144, head_dim 128); 2B/9B use head_dim (256).
    if config.get("head_dim") is not None:
        config["query_pre_attn_scalar"] = (
            config["hidden_size"] // config["num_attention_heads"]
            if config["num_hidden_layers"] == 46 else config["head_dim"])

    asc = _read_float(meta, f"{arch}.attn_logit_softcapping")
    if asc is not None:
        config["attn_logit_softcapping"] = asc
    fsc = _read_float(meta, f"{arch}.final_logit_softcapping")
    if fsc is not None:
        config["final_logit_softcapping"] = fsc

    sw = _read_int(meta, f"{arch}.attention.sliding_window")
    if sw is not None:
        config["sliding_window"] = sw


# gemma (Gemma-1 2B / 7B)

def _synth_gemma(meta, shapes, config: dict) -> None:
    """Synthesize a gemma (Gemma-1) config. The universal fields cover the whole
    backbone - mlx_lm's gemma ModelArgs needs only hidden/layers/heads/kv/
    intermediate/eps/vocab/rope_theta plus head_dim, all of which the universal
    extractor produces. Gemma is always tied (no output.weight tensor) and its
    RMSNorm bakes +1 (undone at remap via gemma_norm_minus_one).

    The one load-bearing detail is head_dim: Gemma-1 7B has head_dim=256 while
    hidden_size // num_attention_heads = 192, so head_dim must come from
    `attention.key_length` (which the universal extractor reads) - never the
    hidden//heads fallback. Guard it explicitly."""
    arch = "gemma"
    if config.get("head_dim") is None:
        raise ValueError(
            f"gemma synth: missing {arch}.attention.key_length (head_dim); "
            f"hidden//heads is wrong for Gemma-1 7B (256 vs 192). "
            f"Pass hf_source for a config with an explicit head_dim.")


# phi3 (Phi-3 mini / small / medium; fused qkv + fused gate_up)

def _synth_phi3(meta, shapes, config: dict) -> None:
    """Synthesize a phi3 config. Universal fields cover the backbone; mlx_lm's
    phi3 derives head_dim as hidden_size // num_attention_heads (== this GGUF's
    rope.dimension_count) and the fused `attn_qkv` / `ffn_up` tensors are remapped
    onto qkv_proj / gate_up_proj. The 4K variants need nothing further. The 128K
    long-context variants use su/longrope factor arrays that are not reconstructed
    here - pass hf_source for those."""
    arch = "phi3"
    orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
    if orig is not None:
        config["original_max_position_embeddings"] = orig


# glm4 (GLM-4 9B / 32B 0414 dense)

def _synth_glm4(meta, shapes, config: dict) -> None:
    """Synthesize a glm4 (GLM-4 dense) config. Universal fields cover hidden/
    layers/heads/kv/intermediate/eps/vocab/head_dim (from key_length 128)/
    rope_theta/max_pos. mlx_lm glm4 has no tie_word_embeddings field - it always
    builds a separate lm_head (this GGUF is untied: output.weight present). Two
    arch-specific fields remain:

      - attention_bias: GLM-4 has q/k/v bias (o_proj has none); inferred from the
        presence of the bias tensor so it's right whether or not a variant ships it.
      - partial_rotary_factor: GLM-4 rotates only the first rope.dimension_count
        (64) of head_dim (128), i.e. 0.5; mlx_lm requires it (no default)."""
    arch = "glm4"
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q.bias")
    rope_dim = _read_int(meta, f"{arch}.rope.dimension_count")
    head_dim = config.get("head_dim")
    if rope_dim is None or not head_dim:
        raise ValueError(
            f"glm4 synth: need {arch}.rope.dimension_count and "
            f"{arch}.attention.key_length to derive partial_rotary_factor "
            f"(rope_dim/head_dim). Pass hf_source for a config that supplies them.")
    config["partial_rotary_factor"] = rope_dim / head_dim


# llama (Llama-2/3, Mistral-7B-as-llama, Vicuna, LLaVA text tower, ...)

def _synth_llama(meta, shapes, config: dict) -> None:
    """Synthesize a llama config from a 'llama'-arch GGUF.

    The universal fields already cover a plain Llama/Mistral-as-llama backbone
    (hidden/layers/heads/kv_heads/intermediate/eps/vocab/rope_theta). This adds
    the two llama-specific details: head_dim source and rope-scaling passthrough.
    """
    arch = "llama"

    # llama-family GGUFs carry head_dim as `rope.dimension_count`, not
    # `attention.key_length` (which the universal extractor reads, so it left
    # head_dim unset). mlx_lm's llama.Attention falls back to hidden//heads when
    # head_dim is None, but setting it explicitly is correct for models whose
    # head_dim != hidden_size // num_attention_heads.
    rope_dim = _read_int(meta, f"{arch}.rope.dimension_count")
    if rope_dim is not None:
        config["head_dim"] = rope_dim

    # rope-scaling passthrough (linear / yarn), keyed for mlx_lm's
    # initialize_rope. Note: Llama-3's "llama3" rope is baked by llama.cpp into
    # a `rope_freqs.weight` tensor with no scaling KV - nothing to synthesize
    # here; the loader consumes that tensor directly (_patch_rope_factors).
    # Llama-2 / Mistral-7B carry no scaling KV and need nothing.
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"type": scaling_type, "rope_type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        orig_ctx = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig_ctx is not None:
            rp["original_max_position_embeddings"] = orig_ctx
        if scaling_type == "yarn":
            beta_fast = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_fast")
            if beta_fast is not None:
                rp["beta_fast"] = beta_fast
            beta_slow = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_slow")
            if beta_slow is not None:
                rp["beta_slow"] = beta_slow
            log_mul = _read_float(meta, f"{arch}.rope.scaling.yarn_log_multiplier")
            if log_mul is not None:
                rp["mscale_all_dim"] = log_mul
        config["rope_scaling"] = rp

    # Optional sliding-window attention (some Mistral-as-llama conversions).
    # Only present when the GGUF carries it; mlx_lm.llama honors sliding_window.
    sliding = _read_int(meta, f"{arch}.attention.sliding_window")
    if sliding is not None and sliding > 0:
        config["sliding_window"] = sliding

    # Mixtral (and other llama-arch sparse MoE): llama.cpp ships Mixtral under
    # general.architecture='llama' + an expert count. mlx-lm models it as a
    # distinct `mixtral` model_type (block_sparse_moe + SwitchGLU), so retarget
    # and add the two MoE fields. The remaining backbone (attention, norms, rope)
    # is the shared llama layout, so the universal + above fields already cover
    # it. The per-expert split weights are coalesced + routed in the loader/remap.
    n_experts = _read_int(meta, f"{arch}.expert_count") or 0
    if n_experts > 0:
        config["model_type"] = "mixtral"
        config["num_local_experts"] = n_experts
        config["num_experts_per_tok"] = _require(
            _read_int(meta, f"{arch}.expert_used_count"),
            arch=arch, gguf_field=f"{arch}.expert_used_count")


# seed_oss (ByteDance Seed-OSS 36B)

def _synth_seed_oss(meta, shapes, config: dict) -> None:
    """Synthesize a seed_oss config from a 'seed_oss'-arch GGUF.

    Plain Llama-shaped dense; the universal fields cover the backbone. mlx-lm's
    ``seed_oss.ModelArgs`` requires ``head_dim`` (no default), and Seed-OSS-36B's
    head_dim (128) is *not* hidden//heads - it rides in GGUF as
    ``attention.key_length`` (read by the universal extractor). Keep a
    hidden//heads fallback for any conversion that omits it. Seed-OSS-36B
    carries q/k/v attention biases (no output bias) - GGUF has no KV flag for
    them, so both bias flags are derived from tensor presence.
    tie_word_embeddings is derived universally from the presence of
    ``output.weight``. rope-scaling is passed through for long-context
    variants.
    """
    arch = "seed_oss"
    config.setdefault(
        "head_dim", config["hidden_size"] // config["num_attention_heads"])
    config["attention_bias"] = "blk.0.attn_q.bias" in shapes
    config["attention_out_bias"] = "blk.0.attn_output.bias" in shapes

    rope_scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None and rope_scaling_type != "none":
        rp: dict[str, Any] = {"type": rope_scaling_type,
                              "rope_type": rope_scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp


# smollm3 (HF SmolLM3 - Llama backbone with NoPE)

def _synth_smollm3(meta, shapes, config: dict) -> None:
    """Synthesize a smollm3 config from a 'smollm3'-arch GGUF.

    SmolLM3 is a Llama subclass; the universal fields + the llama head_dim source
    (``rope.dimension_count``) cover the backbone. The only SmolLM3-specific knob
    is NoPE - rotary embeddings are disabled on every Nth layer. llama.cpp
    hardcodes ``n_no_rope_layer_step = 4`` (no GGUF KV; ``use_rope = (il+1) % 4 !=
    0``), which is exactly mlx-lm's ``no_rope_layer_interval = 4`` default and
    formula - so the default is correct and nothing needs emitting. Kept explicit
    here so a future variant that does carry the step can override it.
    """
    arch = "smollm3"
    rope_dim = _read_int(meta, f"{arch}.rope.dimension_count")
    if rope_dim is not None:
        config["head_dim"] = rope_dim

    # NoPE interval: llama.cpp ships no KV for it (hardcoded 4); read one only if
    # a future conversion adds it, else mlx-lm's default-4 matches llama.cpp.
    step = _read_int(meta, f"{arch}.attention.n_no_rope_layer_step")
    if step is not None:
        config["no_rope_layer_interval"] = step

    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"type": scaling_type, "rope_type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp


# granite (IBM Granite 3.x/4.x dense)

def _synth_granite(meta, shapes, config: dict) -> None:
    """Synthesize a granite config from a 'granite'-arch GGUF.

    Granite is a Llama backbone (reuses the LLAMA remap alias: NORM rope =>
    qk_permute, ffn_norm pin) plus four scalar multipliers that mlx-lm's
    granite.ModelArgs requires (no defaults) and applies at *runtime* - they are
    not folded into the weights, so they must reach the config exactly. Attention
    uses head_dim = hidden//heads (no head_dim arg). Biases are inferred from
    tensor presence (Granite dense ships none). max_position_embeddings /
    rope_theta come from the universal fields.
    """
    arch = "granite"
    config["embedding_multiplier"] = _require(
        _read_float(meta, f"{arch}.embedding_scale"),
        arch=arch, gguf_field=f"{arch}.embedding_scale")
    config["residual_multiplier"] = _require(
        _read_float(meta, f"{arch}.residual_scale"),
        arch=arch, gguf_field=f"{arch}.residual_scale")
    config["attention_multiplier"] = _require(
        _read_float(meta, f"{arch}.attention.scale"),
        arch=arch, gguf_field=f"{arch}.attention.scale")
    config["logits_scaling"] = _require(
        _read_float(meta, f"{arch}.logit_scale"),
        arch=arch, gguf_field=f"{arch}.logit_scale")
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q.bias")
    config["mlp_bias"] = _has_tensor(shapes, "blk.0.ffn_gate.bias")

    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"type": scaling_type, "rope_type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp


# ernie4_5-moe (Baidu ERNIE-4.5-MoE, e.g. 21B-A3B)

def _synth_ernie4_5_moe(meta, shapes, config: dict) -> None:
    """Synthesize an ernie4_5_moe config from an 'ernie4_5-moe'-arch GGUF.

    Fine-grained MoE (stacked routed experts -> SwitchGLU) with a shared expert,
    behind ``moe_layer_start_index`` leading dense layers; MoE fires on layers
    where ``(layer+1) % moe_layer_interval == 0`` at/after that index. The
    universal fields cover the backbone (hidden/layers/heads/kv/intermediate/eps/
    vocab/rope_theta/ctx/tie + head_dim from key_length); mlx-lm's ernie4_5_moe
    derives head_dim = hidden//heads when absent (correct for the 21B: 2560/20 =
    128). This adds the MoE block + the bias flag.

    mlx-lm's ernie4_5_moe builds *every* Linear/SwitchGLU with ``bias=use_bias``
    and computes the shared-expert width as moe_intermediate_size *
    moe_num_shared_experts - so only the count is emitted, not the width. The
    aux-free routing correction bias (``exp_probs_b`` in GGUF ->
    e_score_correction_bias) is *dropped* by the model's sanitize (and skipped in
    remap), so sigmoid-gated variants route without it.
    """
    arch = "ernie4_5-moe"

    config["use_bias"] = _has_tensor(shapes, "blk.0.attn_q.bias")

    config["moe_num_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["moe_k"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")

    # moe_intermediate_size: prefer the KV, else derive from the stacked expert
    # tensor (gguf-py order [hidden, moe_intermediate, n_experts] -> middle dim).
    moe_inter = _read_int(meta, f"{arch}.expert_feed_forward_length")
    if moe_inter is None:
        gate_exps = shapes.get("blk.0.ffn_gate_exps.weight")
        if gate_exps is None or len(gate_exps) != 3:
            raise ValueError(
                f"ernie4_5-moe synth: missing {arch}.expert_feed_forward_length "
                f"and can't derive moe_intermediate_size - need a stacked 3-D "
                f"blk.0.ffn_gate_exps.weight (got {gate_exps}). Pass hf_source.")
        moe_inter = gate_exps[1]
    config["moe_intermediate_size"] = moe_inter

    config["moe_num_shared_experts"] = (
        _read_int(meta, f"{arch}.expert_shared_count") or 0)
    config["moe_layer_start_index"] = (
        _read_int(meta, f"{arch}.leading_dense_block_count") or 0)
    config["moe_layer_interval"] = (
        _read_int(meta, f"{arch}.interleave_moe_layer_step") or 1)

    # expert_gating_func: 1 = softmax, 2 = sigmoid (aux-free). mlx-lm selects the
    # gate activation by name; default to softmax when the KV is absent.
    gating = _read_int(meta, f"{arch}.expert_gating_func")
    config["moe_gate_act"] = "sigmoid" if gating == 2 else "softmax"
    config["moe_use_aux_free"] = gating == 2


# Shared V3 fine-grained-MoE metadata read (MiniMax-M2, Hunyuan, ...)

def _read_v3_moe(meta, shapes, arch: str) -> dict[str, Any]:
    """Read the GGUF metadata common to the new sigmoid-gated fine-grained MoE
    archs: expert count, top-k, the expert FFN width (KV or, failing that,
    derived from the stacked ``ffn_gate_exps`` tensor's middle dim), the shared-
    expert count, and whether the gate is sigmoid (vs softmax).

    Returns a plain dict; each ``_synth_X`` maps these onto its own ModelArgs
    field names - the V3-MoE config *schemas* (num_local_experts vs
    n_routed_experts, block_sparse_moe vs mlp, shared-expert handling) diverge too
    much to share the field-*writing*, but the GGUF read is identical.
    """
    n_experts = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    n_used = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")

    moe_ffn = _read_int(meta, f"{arch}.expert_feed_forward_length")
    if moe_ffn is None:
        gate_exps = shapes.get("blk.0.ffn_gate_exps.weight")
        if gate_exps is None or len(gate_exps) != 3:
            raise ValueError(
                f"{arch} synth: missing {arch}.expert_feed_forward_length and "
                f"can't derive the expert FFN width - need a stacked 3-D "
                f"blk.0.ffn_gate_exps.weight (got {gate_exps}). Pass hf_source.")
        # gguf-py order: [hidden, expert_intermediate, n_experts].
        moe_ffn = gate_exps[1]

    gating = _read_int(meta, f"{arch}.expert_gating_func")
    return {
        "n_experts": n_experts,
        "n_used": n_used,
        "moe_ffn": moe_ffn,
        "n_shared": _read_int(meta, f"{arch}.expert_shared_count") or 0,
        "sigmoid": gating == 2,
    }


# minimax-m2 (MiniMax-M2 230B-A10B)

def _synth_minimax(meta, shapes, config: dict) -> None:
    """Synthesize a minimax config from a 'minimax-m2'-arch GGUF.

    MiniMax-M2 is an every-layer fine-grained sigmoid-gated MoE (no dense layers,
    no shared expert) with standard full attention, full-width qk-norm, and
    partial rotary. The universal fields cover hidden/layers/heads/kv/eps/
    rope_theta/ctx/vocab/tie + head_dim (from key_length - critical: head_dim
    (128) != hidden//heads (64), and the full-width qk-norm shape depends on it).
    This adds the MoE block + rotary_dim + the qk-norm flag.

    mlx-lm's minimax SwitchGLU uses ``intermediate_size`` for the experts (there
    are no dense layers), so intermediate_size is the expert FFN width - override
    the universal feed_forward_length with it. ``shared_intermediate_size`` is a
    required ModelArgs field but is unused in the forward (no shared expert), so
    it's supplied as 0 just to satisfy from_dict. The sigmoid correction bias
    (exp_probs_b -> block_sparse_moe.e_score_correction_bias) is kept (not dropped
    like ERNIE).
    """
    arch = "minimax-m2"
    moe = _read_v3_moe(meta, shapes, arch)
    config["num_local_experts"] = moe["n_experts"]
    config["num_experts_per_tok"] = moe["n_used"]
    config["intermediate_size"] = moe["moe_ffn"]   # expert FFN width
    config["shared_intermediate_size"] = 0          # required-but-unused
    config["scoring_func"] = "sigmoid"              # minimax always sigmoid-gates

    # Full-width qk-norm (RMSNorm over head_dim*n_heads): present on every layer.
    config["use_qk_norm"] = _has_tensor(shapes, "blk.0.attn_q_norm.weight")

    # Partial rotary: rotary_dim = rope.dimension_count (64 < head_dim 128).
    config["rotary_dim"] = _require(
        _read_int(meta, f"{arch}.rope.dimension_count"),
        arch=arch, gguf_field=f"{arch}.rope.dimension_count")

    # head_dim (128) != hidden//heads (64), so it must have come from key_length;
    # the full-width qk-norm shape (head_dim*n_heads) is wrong without it.
    if "head_dim" not in config:
        raise ValueError(
            "minimax synth: need attention.key_length (head_dim != hidden//heads "
            "for MiniMax-M2). Pass hf_source.")


# minimax-m3 (MiniMax-M3 428B-A23B)

def _synth_minimax_m3(meta, shapes, config: dict) -> None:
    """Synthesize a minimax_m3 config from a 'minimax-m3'-arch GGUF.

    M3 keeps M2's GQA base (partial rotary, head_dim from key_length) but moves
    to a DeepSeek-V3-shaped MoE: ``leading_dense_block_count`` dense layers
    (dense FFN width = the arch feed_forward_length), then sigmoid-gated routed
    experts + a shared expert per layer, with routed weights renormalized and
    scaled by ``expert_weights_scale``. qk-norm is per-head ([head_dim], vs
    M2's full-width) but the flag reads identically from tensor presence. The
    norms arrive gemma-+1-baked (remap unbakes); activation is SwiGLU-OAI
    (ModelArgs defaults alpha=1.702, limit=7.0 - no GGUF KV carries them).

    Field mapping vs the universal read: the universal ``intermediate_size``
    (from feed_forward_length) is the dense width on this arch - it moves to
    ``dense_intermediate_size`` and ``intermediate_size`` becomes the expert
    FFN width, matching mlx-lm minimax_m3.ModelArgs. The shared-expert width
    has no KV; it's read off the first ``ffn_gate_shexp`` tensor.
    """
    arch = "minimax-m3"
    moe = _read_v3_moe(meta, shapes, arch)
    config["num_local_experts"] = moe["n_experts"]
    config["num_experts_per_tok"] = moe["n_used"]
    config["dense_intermediate_size"] = _require(
        config.get("intermediate_size"),
        arch=arch, gguf_field=f"{arch}.feed_forward_length")
    config["intermediate_size"] = moe["moe_ffn"]   # expert FFN width
    config["scoring_func"] = "sigmoid"             # minimax always sigmoid-gates

    # Shared expert width: no KV exists for it - read the first shexp tensor
    # (gguf-py dims order [hidden, shared_intermediate]).
    shexp = next((s for n, s in shapes.items()
                  if n.startswith("blk.") and n.endswith(".ffn_gate_shexp.weight")),
                 None)
    if moe["n_shared"] > 0:
        if shexp is None or len(shexp) != 2:
            raise ValueError(
                f"{arch} synth: expert_shared_count={moe['n_shared']} but no 2-D "
                f"ffn_gate_shexp tensor to size the shared expert (got {shexp}). "
                "Pass hf_source.")
        config["shared_intermediate_size"] = shexp[1]
    else:
        config["shared_intermediate_size"] = 0     # required-but-unused

    # Routed weights are renormalized over the top-k then scaled; the model
    # class hardcodes the renorm (expert_weights_norm=true on every known M3).
    scale = _read_float(meta, f"{arch}.expert_weights_scale")
    if scale is not None:
        config["routed_scaling_factor"] = scale

    # Leading dense layers -> the per-layer MLP dispatch list.
    n_dense = _read_int(meta, f"{arch}.leading_dense_block_count") or 0
    n_layers = config["num_hidden_layers"]
    config["mlp_layer_types"] = (
        ["dense"] * n_dense + ["sparse"] * (n_layers - n_dense))

    # Per-head qk-norm: present on every layer.
    config["use_qk_norm"] = _has_tensor(shapes, "blk.0.attn_q_norm.weight")

    # Partial rotary: rotary_dim = rope.dimension_count (64 < head_dim 128).
    config["rotary_dim"] = _require(
        _read_int(meta, f"{arch}.rope.dimension_count"),
        arch=arch, gguf_field=f"{arch}.rope.dimension_count")

    # head_dim (128) != hidden//heads (96), so it must have come from key_length.
    if "head_dim" not in config:
        raise ValueError(
            "minimax_m3 synth: need attention.key_length (head_dim != "
            "hidden//heads for MiniMax-M3). Pass hf_source.")

    # MiniMax Sparse Attention: armed by indexer-tensor presence (a native
    # MSA GGUF per llama.cpp PR #24908, or the loader's sidecar merge - both
    # spellings accepted). The `attention.indexer.*` KVs are the PR metadata;
    # a GGUF with inline indexer tensors but no KVs falls back to the
    # released model's constants (4 heads, dim 128, top-16 of 128-blocks,
    # 1 local block).
    has_indexer = any(
        ".indexer.q_proj." in n or ".index_q_proj." in n for n in shapes)
    config["use_sparse_attention"] = has_indexer
    if has_indexer:
        idx = f"{arch}.attention.indexer"
        config["sparse_num_index_heads"] = (
            _read_int(meta, f"{idx}.head_count") or 4)
        config["sparse_index_dim"] = _read_int(meta, f"{idx}.key_length") or 128
        config["sparse_topk_blocks"] = _read_int(meta, f"{idx}.top_k") or 16
        config["sparse_block_size"] = _read_int(meta, f"{idx}.block_size") or 128
        config["sparse_local_block"] = (
            _read_int(meta, f"{idx}.local_blocks") or 1)


# hunyuan-moe (Tencent Hunyuan-A13B)

def _synth_hunyuan(meta, shapes, config: dict) -> None:
    """Synthesize a hunyuan config from a 'hunyuan-moe'-arch GGUF.

    Softmax-gated fine-grained MoE (stacked routed experts -> SwitchGLU) plus a
    per-layer shared expert, with per-head qk-norm and NTK-alpha rope. The
    universal fields cover hidden/layers/heads/kv/eps/rope_theta/ctx/vocab/tie;
    mlx-lm's hunyuan derives head_dim = hidden//heads (no config field). This adds
    the MoE block, the shared expert, the qk-norm/cla/bias flags, and rope_scaling.

    Width bookkeeping (mlx-lm hunyuan): the routed experts use
    ``moe_intermediate_size`` and the shared MLP uses
    ``intermediate_size * num_shared_expert`` - but only the *product* matters for
    the shared MLP, so the whole shared width is folded into intermediate_size
    with num_shared_expert = 1 (the GGUF stores the width, not a count).

    CLA: the llama.cpp hunyuan-moe GGUF materializes k/v on every layer (no
    cross-layer KV sharing in the wire format), so use_cla=False makes mlx-lm
    build k_proj/v_proj on every layer to match. cla_share_factor is a required
    ModelArgs field but is dead when use_cla=False.
    """
    arch = "hunyuan-moe"
    moe = _read_v3_moe(meta, shapes, arch)
    config["num_experts"] = moe["n_experts"]
    config["moe_topk"] = moe["n_used"]
    config["moe_intermediate_size"] = moe["moe_ffn"]   # routed-expert FFN width

    # Shared-expert width -> folded into intermediate_size (num_shared_expert = 1).
    shared_w = _read_int(meta, f"{arch}.expert_shared_feed_forward_length")
    if shared_w is None:
        shexp = shapes.get("blk.0.ffn_gate_shexp.weight")
        shared_w = (shexp[1] if shexp and len(shexp) == 2 else moe["moe_ffn"])
    config["intermediate_size"] = shared_w
    config["num_shared_expert"] = 1
    config["use_mixed_mlp_moe"] = _has_tensor(shapes, "blk.0.ffn_gate_shexp.weight")

    # Per-head qk-norm (RMSNorm over head_dim) + attention bias: tensor-derived.
    config["use_qk_norm"] = _has_tensor(shapes, "blk.0.attn_q_norm.weight")
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q.bias")

    # No cross-layer attention in the GGUF (k/v on every layer) - see docstring.
    config["use_cla"] = False
    config["cla_share_factor"] = 2

    # DynamicNTKAlphaRoPE needs rope_scaling["alpha"]; mlx-lm's effective base is
    # rope_theta * alpha**(d/(d-2)). llama.cpp folds any NTK scaling into freq_base
    # (-> rope_theta) and applies no alpha, so alpha defaults to 1.0 (a no-op that
    # matches the reference); a GGUF carrying an explicit alpha overrides it.
    # __post_init__ also requires factor + type keys. (alpha is the #1 16k-parity
    # check for hunyuan.)
    alpha = _read_float(meta, f"{arch}.rope.scaling.alpha")
    config["rope_scaling"] = {
        "alpha": alpha if alpha is not None else 1.0,
        "factor": 1.0,
        "type": "dynamic",
    }


# hy_v3 (Tencent Hy3 299B-A21B)

def _synth_hy_v3(meta, shapes, config: dict) -> None:
    """Synthesize a hy_v3 config from a 'hy_v3'-arch GGUF (llama.cpp PR #25395).

    Sigmoid-gated fine-grained MoE (192 experts top-8) with a selection-only
    expert bias, top-k renorm x router_scaling_factor, one ungated shared
    expert, and a single leading dense layer. The universal fields cover
    hidden/layers/heads/kv/eps/ctx/vocab/tie (num_hidden_layers already
    excludes the NextN/MTP block via nextn_predict_layers), and the universal
    ``intermediate_size`` (feed_forward_length) is the dense-layer width -
    matching hy_v3.ModelArgs, where the routed/shared expert width is the
    separate ``expert_hidden_dim``.

    first_k_dense_replace has no GGUF KV: derived from tensor presence (the
    leading blocks with no ffn_gate_inp router). num_shared_experts likewise
    has no count KV: expert_shared_feed_forward_length / expert width (the
    converter writes width x count; every known Hy3 config uses 1).

    enable_lm_head_fp32 (true in the HF config) is pinned off: the kquant
    matmul kernels take bf16/f16 activations, and llama.cpp - the 16k-parity
    oracle for this family - also runs its LM head in the compute dtype.
    """
    arch = "hy_v3"
    moe = _read_v3_moe(meta, shapes, arch)
    config["num_experts"] = moe["n_experts"]
    config["num_experts_per_tok"] = moe["n_used"]
    config["expert_hidden_dim"] = moe["moe_ffn"]

    _require(config.get("intermediate_size"),
             arch=arch, gguf_field=f"{arch}.feed_forward_length")

    # head_dim (128) != hidden//heads (64): key_length must have been present.
    if "head_dim" not in config:
        raise ValueError(
            "hy_v3 synth: need attention.key_length (head_dim != "
            "hidden//heads for Hy3). Pass hf_source.")

    # Shared experts: width KV / expert width -> count (no count KV).
    shared_w = _read_int(meta, f"{arch}.expert_shared_feed_forward_length")
    if shared_w is None:
        shexp = next((s for n, s in shapes.items()
                      if n.endswith(".ffn_gate_shexp.weight")), None)
        shared_w = shexp[1] if shexp is not None and len(shexp) == 2 else 0
    config["num_shared_experts"] = (
        shared_w // moe["moe_ffn"] if shared_w else 0)

    # Leading dense layers: no KV - the first block with a router ends the run.
    n_layers = config["num_hidden_layers"]
    n_dense = 0
    while (n_dense < n_layers
           and not _has_tensor(shapes, f"blk.{n_dense}.ffn_gate_inp.weight")):
        n_dense += 1
    config["first_k_dense_replace"] = n_dense

    config["router_scaling_factor"] = (
        _read_float(meta, f"{arch}.expert_weights_scale") or 1.0)
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["route_norm"] = True if norm is None else norm
    # expert_gating_func: 1 = softmax, 2 = sigmoid; absent means sigmoid for
    # this arch (llama.cpp defaults it; the converter writes SIGMOID).
    gating = _read_int(meta, f"{arch}.expert_gating_func")
    config["moe_router_use_sigmoid"] = gating != 1
    config["qk_norm"] = _has_tensor(shapes, "blk.0.attn_q_norm.weight")
    # The selection bias is stored suffix-less on Hy3 GGUFs (see remap block).
    config["moe_router_enable_expert_bias"] = (
        _has_tensor(shapes, f"blk.{n_dense}.exp_probs_b")
        or _has_tensor(shapes, f"blk.{n_dense}.exp_probs_b.bias"))

    # NextN depth (universal fields put it in mtp_num_hidden_layers); the
    # vendored sanitize strips model.layers.{trunk+i} using this count.
    config["num_nextn_predict_layers"] = config.get("mtp_num_hidden_layers", 0)
    config["enable_lm_head_fp32"] = False

    # Plain rope (theta 11.16M on Hy3). A yarn/linear rope.scaling.type in the
    # KV is translated for initialize_rope - long-context (1M) conversions may
    # carry one.
    rope_parameters = {
        "rope_theta": _require(config.get("rope_theta"),
                               arch=arch, gguf_field=f"{arch}.rope.freq_base"),
        "rope_type": "default",
    }
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type in ("yarn", "linear"):
        rope_parameters["rope_type"] = scaling_type
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor:
            rope_parameters["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig:
            rope_parameters["original_max_position_embeddings"] = orig
    config["rope_parameters"] = rope_parameters


# granitehybrid (IBM Granite 4.x hybrid: H-Micro / H-Tiny / H-Small)

def _synth_granite_hybrid(meta, shapes, config: dict) -> None:
    """Synthesize a granitemoehybrid config from a 'granitehybrid'-arch GGUF.

    Alternating Mamba2 + attention layers - a layer is recurrent iff its entry
    in the per-layer ``attention.head_count_kv`` array is 0 (the same rule
    llama.cpp uses) - each followed by softmax MoE + a fused-input shared MLP,
    or by a plain dense MLP on non-MoE variants. The universal fields cover the
    backbone; this adds ``layer_types``, the four granite multipliers, the
    Mamba2 geometry, the MoE block, and NoPE.

    Multipliers: optional in the GGUF - llama.cpp treats absent/0.0 as "off"
    (x1), so default each to neutral. ``attention_multiplier`` is mlx-lm's SDPA
    scale outright, so its fallback is 1/sqrt(head_dim) (head_dim is always
    hidden//heads in mlx-lm's granitemoehybrid).

    NoPE: Granite 4.x hybrids use no positional embeddings; llama.cpp keys this
    off ``rope.scaling.finetuned`` (default true => rope).
    """
    arch = "granitehybrid"

    # granite multipliers (neutral when absent/0.0, see docstring)
    config["embedding_multiplier"] = (
        _read_float(meta, f"{arch}.embedding_scale") or 1.0)
    config["residual_multiplier"] = (
        _read_float(meta, f"{arch}.residual_scale") or 1.0)
    config["logits_scaling"] = _read_float(meta, f"{arch}.logit_scale") or 1.0
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    config["attention_multiplier"] = (
        _read_float(meta, f"{arch}.attention.scale") or head_dim ** -0.5)

    # layer_types from the per-layer head_count_kv (0 => recurrent)
    n_layers = config["num_hidden_layers"]
    kv_arr = _read_int_array(meta, f"{arch}.attention.head_count_kv")
    if kv_arr is None:
        config["layer_types"] = ["attention"] * n_layers
    else:
        config["layer_types"] = [
            "mamba" if v == 0 else "attention" for v in kv_arr[:n_layers]]
        # The universal extractor took element 0 of the array, which is 0
        # whenever layer 0 is recurrent - fix to the first attention layer's.
        nonzero = [v for v in kv_arr if v]
        if nonzero:
            config["num_key_value_heads"] = nonzero[0]

    config["attention_bias"] = any(
        n.endswith(".attn_q.bias") for n in shapes)

    # Mamba2 geometry (all from KV; d_head derives from inner//heads)
    d_inner = _require(
        _read_int(meta, f"{arch}.ssm.inner_size"),
        arch=arch, gguf_field=f"{arch}.ssm.inner_size")
    config["mamba_n_heads"] = _require(
        _read_int(meta, f"{arch}.ssm.time_step_rank"),
        arch=arch, gguf_field=f"{arch}.ssm.time_step_rank")
    config["mamba_d_head"] = d_inner // config["mamba_n_heads"]
    config["mamba_d_state"] = _require(
        _read_int(meta, f"{arch}.ssm.state_size"),
        arch=arch, gguf_field=f"{arch}.ssm.state_size")
    config["mamba_d_conv"] = _require(
        _read_int(meta, f"{arch}.ssm.conv_kernel"),
        arch=arch, gguf_field=f"{arch}.ssm.conv_kernel")
    config["mamba_n_groups"] = _read_int(meta, f"{arch}.ssm.group_count") or 1
    config["mamba_conv_bias"] = any(
        n.endswith(".ssm_conv1d.bias") for n in shapes)
    config["mamba_proj_bias"] = False

    # MoE (absent expert_count => dense-MLP variant; fields stay unset)
    n_experts = _read_int(meta, f"{arch}.expert_count") or 0
    if n_experts:
        config["num_local_experts"] = n_experts
        config["num_experts_per_tok"] = _require(
            _read_int(meta, f"{arch}.expert_used_count"),
            arch=arch, gguf_field=f"{arch}.expert_used_count")
        # mlx-lm builds the fused shared MLP unconditionally in MoE mode, so
        # the shared width is required (granite hybrids always ship it).
        config["shared_intermediate_size"] = _require(
            _read_int(meta, f"{arch}.expert_shared_feed_forward_length"),
            arch=arch,
            gguf_field=f"{arch}.expert_shared_feed_forward_length")

    # NoPE switch
    finetuned = _read_bool(meta, f"{arch}.rope.scaling.finetuned")
    config["position_embedding_type"] = "nope" if finetuned is False else "rope"
    # rope_theta is a required ModelArgs field but NoPE GGUFs may omit freq_base.
    config.setdefault("rope_theta", 10000.0)


# falcon-h1 (TII Falcon-H1: 0.5B / 1.5B / 1.5B-Deep / 3B / 7B / 34B)

def _synth_falcon_h1(meta, shapes, config: dict) -> None:
    """Synthesize a falcon_h1 config from a 'falcon-h1'-arch GGUF.

    Every layer runs attention and a Mamba2 mixer in parallel (input_layernorm
    feeds both, outputs summed), then a dense gated MLP. The universal fields
    cover the backbone - including head_dim from attention.key_length, which is
    required here (128 != hidden//heads on every released size).

    Multipliers: Falcon-H1's muP zoo (embedding / attention in+out / key /
    lm_head / mlp / ssm in+out + the 5-segment ssm vector) is folded into the
    wire weights by llama.cpp's converter, and the GGUF carries no multiplier
    KVs at all. So every ModelArgs multiplier is pinned neutral here: mlx-lm's
    own fold (sanitize) is layout-gated on the raw-HF conv1d shape and
    early-exits on our already-MLX-layout conv1d, but the class defaults are
    non-neutral (they encode one HF checkpoint), and the tied-embed lm_head
    path applies lm_head_multiplier/embedding_multiplier at RUNTIME - neutral
    values make both a no-op.

    Mamba2 geometry mirrors granitehybrid (same llama.cpp builder):
    n_heads = ssm.time_step_rank, d_head = inner//heads. norm_before_gate is
    pinned False - llama.cpp's shared mamba2 builder applies swiglu(z, y)
    before the grouped RMS norm (mamba-base.cpp), and no GGUF KV exists for it.
    """
    arch = "falcon-h1"

    # neutral muP multipliers (folded at convert; see docstring)
    config["embedding_multiplier"] = 1.0
    config["attention_in_multiplier"] = 1.0
    config["attention_out_multiplier"] = 1.0
    config["key_multiplier"] = 1.0
    config["lm_head_multiplier"] = 1.0
    config["mlp_multipliers"] = [1.0, 1.0]
    config["ssm_in_multiplier"] = 1.0
    config["ssm_out_multiplier"] = 1.0
    config["ssm_multipliers"] = [1.0] * 5

    # Mamba2 geometry (all from KV; d_head derives from inner//heads)
    d_inner = _require(
        _read_int(meta, f"{arch}.ssm.inner_size"),
        arch=arch, gguf_field=f"{arch}.ssm.inner_size")
    config["mamba_d_ssm"] = d_inner
    config["mamba_n_heads"] = _require(
        _read_int(meta, f"{arch}.ssm.time_step_rank"),
        arch=arch, gguf_field=f"{arch}.ssm.time_step_rank")
    config["mamba_d_head"] = d_inner // config["mamba_n_heads"]
    config["mamba_d_state"] = _require(
        _read_int(meta, f"{arch}.ssm.state_size"),
        arch=arch, gguf_field=f"{arch}.ssm.state_size")
    config["mamba_d_conv"] = _require(
        _read_int(meta, f"{arch}.ssm.conv_kernel"),
        arch=arch, gguf_field=f"{arch}.ssm.conv_kernel")
    config["mamba_n_groups"] = _read_int(meta, f"{arch}.ssm.group_count") or 1
    config["mamba_norm_before_gate"] = False
    config["mamba_rms_norm"] = any(
        n.endswith(".ssm_norm.weight") for n in shapes)
    config["mamba_conv_bias"] = any(
        n.endswith(".ssm_conv1d.bias") for n in shapes)
    config["mamba_proj_bias"] = any(
        n.endswith(".ssm_in.bias") for n in shapes)
    config["projectors_bias"] = any(
        n.endswith(".ssm_out.bias") for n in shapes)

    # optional biases on the attention / MLP linears
    config["attention_bias"] = any(
        n.endswith(".attn_q.bias") for n in shapes)
    config["mlp_bias"] = any(
        n.endswith(".ffn_gate.bias") for n in shapes)


# qwen3next (Qwen3-Next-80B-A3B)

def _synth_qwen3next(meta, shapes, config: dict) -> None:
    """Synthesize a qwen3_next config from a 'qwen3next'-arch GGUF.

    Gated-DeltaNet linear attention on every layer whose (idx+1) is not a
    multiple of ``full_attention_interval`` (the rest are gated full attention
    - the output gate rides fused inside attn_q on both sides), and an
    every-layer fine-grained MoE + shared expert. The universal fields cover
    the backbone; head_dim (attention.key_length) is required - 256 on the
    80B, not hidden//heads.

    GDN geometry inverts the converter's KV mapping: linear_num_value_heads =
    ssm.time_step_rank, linear_num_key_heads = ssm.group_count,
    linear_key_head_dim = ssm.state_size, value head dim = inner // v_heads.

    Both GDN input layouts load. The legacy layout stores the raw HF
    per-k-head-interleaved fused tensor as ssm_in - exactly what mlx-lm's
    in_proj_qkvz consumes. The current llama.cpp converter instead splits it
    into de-interleaved attn_qkv (q|k|v, flat head-major) + attn_gate (z);
    re-fusing those is a row re-interleave on quantized blocks, so instead
    the loader restructures the model: ``gdn_split_layout=True`` here makes
    ``gdn_patches._patch_qwen3next_split_gdn`` swap each GDN's in_proj_qkvz for
    split in_proj_qkv/in_proj_z Linears and skip the runtime reorder (the
    split tensor is the mixed_qkv the stock forward rebuilds).

    V heads stay HF-grouped on the wire - the loader's qwen3.5 tiled-V patch
    is explicitly not applied (see ``_needs_tiled_v_patch``); the config
    carries ``kv_head_layout="grouped"`` to document that.
    """
    arch = "qwen3next"

    if any(n.endswith(".attn_qkv.weight") for n in shapes):
        # Split GDN layout (current llama.cpp converts): the loader patches
        # the model modules to match; remap already routes attn_qkv/attn_gate
        # to linear_attn.in_proj_qkv / in_proj_z via the canonical map.
        config["gdn_split_layout"] = True

    _require(config.get("head_dim"),
             arch=arch, gguf_field=f"{arch}.attention.key_length")

    # rope-scaling passthrough (same shape as qwen3/qwen3moe).
    rope_scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if rope_scaling_type is not None and rope_scaling_type != "none":
        rp: dict[str, Any] = {"type": rope_scaling_type,
                              "rope_type": rope_scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        config["rope_scaling"] = rp

    # Partial rotary: the converter writes rope.dimension_count =
    # head_dim * partial_rotary_factor (0.25 on every released checkpoint).
    rot = _read_int(meta, f"{arch}.rope.dimension_count")
    config["partial_rotary_factor"] = (
        rot / config["head_dim"] if rot is not None else 0.25)

    # gated-DeltaNet geometry (converter KV mapping inverted)
    inner = _require(
        _read_int(meta, f"{arch}.ssm.inner_size"),
        arch=arch, gguf_field=f"{arch}.ssm.inner_size")
    num_v = _require(
        _read_int(meta, f"{arch}.ssm.time_step_rank"),
        arch=arch, gguf_field=f"{arch}.ssm.time_step_rank")
    num_k = _require(
        _read_int(meta, f"{arch}.ssm.group_count"),
        arch=arch, gguf_field=f"{arch}.ssm.group_count")
    if inner % num_v != 0:
        raise ValueError(
            f"qwen3next synth: ssm.inner_size={inner} not divisible by "
            f"time_step_rank={num_v}")
    config["linear_num_value_heads"] = num_v
    config["linear_num_key_heads"] = num_k
    config["linear_key_head_dim"] = _require(
        _read_int(meta, f"{arch}.ssm.state_size"),
        arch=arch, gguf_field=f"{arch}.ssm.state_size")
    config["linear_value_head_dim"] = inner // num_v
    config["linear_conv_kernel_dim"] = _require(
        _read_int(meta, f"{arch}.ssm.conv_kernel"),
        arch=arch, gguf_field=f"{arch}.ssm.conv_kernel")
    config["kv_head_layout"] = "grouped"

    # layer schedule: mlx-lm only models the interval pattern
    interval = _read_int(meta, f"{arch}.full_attention_interval") or 4
    config["full_attention_interval"] = interval
    recr = (_read_bool_array(meta, f"{arch}.attention.recurrent_layers")
            or _read_int_array(meta, f"{arch}.attention.recurrent_layers"))
    if recr is not None:
        expected = [(i + 1) % interval != 0
                    for i in range(config["num_hidden_layers"])]
        if [bool(v) for v in recr[:len(expected)]] != expected:
            raise ValueError(
                "qwen3next synth: attention.recurrent_layers does not match "
                f"the full_attention_interval={interval} pattern; mlx-lm's "
                "qwen3_next only supports the interval schedule")

    # MoE (every layer; llama.cpp also hard-errors on zero experts)
    config["num_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    config["moe_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
    config["shared_expert_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_shared_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_shared_feed_forward_length")
    config["decoder_sparse_step"] = 1
    config["mlp_only_layers"] = []
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm


# qwen35 / qwen35moe

def _synth_qwen35(meta, shapes, config: dict, arch: str) -> None:
    config["full_attention_interval"] = _require(
        _read_int(meta, f"{arch}.full_attention_interval"),
        arch=arch, gguf_field=f"{arch}.full_attention_interval")

    inner = _require(
        _read_int(meta, f"{arch}.ssm.inner_size"),
        arch=arch, gguf_field=f"{arch}.ssm.inner_size")
    num_v = _require(
        _read_int(meta, f"{arch}.ssm.time_step_rank"),
        arch=arch, gguf_field=f"{arch}.ssm.time_step_rank")
    num_k = _require(
        _read_int(meta, f"{arch}.ssm.group_count"),
        arch=arch, gguf_field=f"{arch}.ssm.group_count")
    conv_kernel = _require(
        _read_int(meta, f"{arch}.ssm.conv_kernel"),
        arch=arch, gguf_field=f"{arch}.ssm.conv_kernel")

    if inner % num_v != 0:
        raise ValueError(
            f"qwen35 synth: ssm.inner_size={inner} not divisible by "
            f"time_step_rank={num_v}")
    head_v_dim = inner // num_v

    # Cross-check: ssm_norm.shape == [head_v_dim], ssm_a.shape == [num_v]
    ssm_norm_shape = shapes.get("blk.0.ssm_norm.weight")
    if ssm_norm_shape is not None and ssm_norm_shape[0] != head_v_dim:
        raise ValueError(
            f"qwen35 synth: derived head_v_dim={head_v_dim} but "
            f"blk.0.ssm_norm.weight shape={ssm_norm_shape}")
    ssm_a_shape = shapes.get("blk.0.ssm_a")
    if ssm_a_shape is not None and ssm_a_shape[0] != num_v:
        raise ValueError(
            f"qwen35 synth: derived num_v_heads={num_v} but "
            f"blk.0.ssm_a shape={ssm_a_shape}")
    # ssm_alpha is Linear(hidden, num_v_heads) - same num_v, not num_k.
    # No tensor directly exposes num_k_heads; the conv1d shape derivation
    # below + the metadata field `<arch>.ssm.group_count` are the sources.

    # head_k_dim is genuinely absent from KV metadata. Derive from
    # ssm_conv1d shape: it's [kernel, inner + 2 * num_k * head_k_dim].
    conv_shape = shapes.get("blk.0.ssm_conv1d.weight")
    if conv_shape is None:
        raise ValueError(
            "qwen35 synth: missing tensor blk.0.ssm_conv1d.weight - "
            "needed to derive linear_key_head_dim")
    conv_in = conv_shape[1]
    if (conv_in - inner) % (2 * num_k) != 0:
        raise ValueError(
            f"qwen35 synth: ssm_conv1d shape={conv_shape} doesn't match "
            f"inner_size={inner} + 2 * num_k_heads={num_k} * head_k_dim")
    head_k_dim = (conv_in - inner) // (2 * num_k)

    config["linear_num_value_heads"] = num_v
    config["linear_num_key_heads"] = num_k
    config["linear_value_head_dim"] = head_v_dim
    config["linear_key_head_dim"] = head_k_dim
    config["linear_conv_kernel_dim"] = conv_kernel

    # GGUF V-indexed tensors are in tiled order (convert_hf_to_gguf reorders).
    config["kv_head_layout"] = "tiled"

    # Build rope_parameters so __post_init__ doesn't override rope_theta
    # with the default (100000). mrope_section comes from the GGUF dimension
    # sections array (last element is padding/zero and gets dropped).
    freq_base = config.get("rope_theta", 10000000.0)
    dim_sections = _read_int_array(meta, f"{arch}.rope.dimension_sections")
    mrope_section = [s for s in (dim_sections or [11, 11, 10, 0]) if s > 0]
    config["rope_parameters"] = {
        "type": "default",
        "rope_theta": freq_base,
        "mrope_section": mrope_section,
        "partial_rotary_factor": 0.25,
    }

    # MoE fields (only on qwen35moe).
    if arch == "qwen35moe":
        config["num_experts"] = _require(
            _read_int(meta, f"{arch}.expert_count"),
            arch=arch, gguf_field=f"{arch}.expert_count")
        config["num_experts_per_tok"] = _require(
            _read_int(meta, f"{arch}.expert_used_count"),
            arch=arch, gguf_field=f"{arch}.expert_used_count")
        config["moe_intermediate_size"] = _require(
            _read_int(meta, f"{arch}.expert_feed_forward_length"),
            arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
        shared_ffn = _read_int(meta, f"{arch}.expert_shared_feed_forward_length")
        if shared_ffn is not None:
            config["shared_expert_intermediate_size"] = shared_ffn


# mistral3 (Ministral-3 / Mistral-Small-3.1)

def _synth_mistral3(meta, shapes, config: dict) -> None:
    """Synthesize a ministral3 config from a 'mistral3' GGUF.

    The model is a LlamaModel variant with yarn RoPE scaling and a
    llama-4-style attention temperature scale. mlx_lm's ministral3 packs
    all of that into `rope_parameters` (it serves as both the rope scaling
    config and the bag of llama4 attention-scale knobs).
    """
    arch = "mistral3"

    # head_dim already set from key_length in universal fields.

    # rope_parameters bag. Reads the yarn scaling fields llama.cpp writes
    # for ministral3, plus the llama-4 attention temperature scale fields.
    # Emitting type=yarn without `factor` KeyErrors in mlx-lm's yarn init;
    # without the scaling KV the model is plain RoPE. The bag itself is
    # mandatory (ministral3 indexes rope_theta / llama_4_scaling_beta /
    # original_max_position_embeddings unconditionally).
    factor = _read_float(meta, f"{arch}.rope.scaling.factor")
    if factor is not None:
        rp: dict[str, Any] = {"rope_type": "yarn", "type": "yarn",
                              "factor": factor}
    else:
        rp = {"rope_type": "default", "type": "default"}

    freq_base = _read_float(meta, f"{arch}.rope.freq_base")
    if freq_base is not None:
        rp["rope_theta"] = freq_base
    rp.setdefault("rope_theta", config.get("rope_theta", 10000.0))

    orig_ctx = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
    if orig_ctx is not None:
        rp["original_max_position_embeddings"] = orig_ctx

    beta_fast = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_fast")
    if beta_fast is not None:
        rp["beta_fast"] = beta_fast
    beta_slow = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_slow")
    if beta_slow is not None:
        rp["beta_slow"] = beta_slow

    # llama.cpp's `add_rope_scaling_yarn_log_mul` is the HF
    # `rope_parameters.mscale_all_dim` value.
    log_mul = _read_float(meta, f"{arch}.rope.scaling.yarn_log_multiplier")
    if log_mul is not None:
        rp["mscale_all_dim"] = log_mul

    # llama.cpp's `add_attn_temperature_scale` is HF
    # `rope_parameters.llama_4_scaling_beta`. ministral3.LanguageModel
    # reads this off rope_parameters when computing the per-position
    # attention scale.
    temp_scale = _read_float(meta, f"{arch}.attention.temperature_scale")
    if temp_scale is not None:
        rp["llama_4_scaling_beta"] = temp_scale
    else:
        # Field is mandatory for ministral3 forward; default to 0.0 (no
        # scaling) so non-ministral3 mistral3 variants still load.
        rp["llama_4_scaling_beta"] = 0.0

    config["rope_parameters"] = rp

    # ministral3.LanguageModel reads
    # rope_parameters["original_max_position_embeddings"] for the attn-scale
    # divisor; if we didn't find it, fall back to max_position_embeddings.
    if "original_max_position_embeddings" not in rp:
        rp["original_max_position_embeddings"] = config.get(
            "max_position_embeddings", 4096)

    # ministral3 is dense, single layer type. (Mistral-Small-3.1 has SWA
    # but the public GGUF conversions don't expose a sliding pattern KV,
    # so default to all full_attention until we see a counter-example.)
    config.setdefault(
        "layer_types", ["full_attention"] * config["num_hidden_layers"])


# nemotron_h_moe (Nemotron-3-Super, hybrid Mamba2+Attention+MoE)

def _synth_nemotron_h_moe(meta, shapes, config: dict) -> None:
    arch = "nemotron_h_moe"

    # nemotron_h model uses layer_norm_epsilon (not rms_norm_eps)
    eps = _read_float(meta, f"{arch}.attention.layer_norm_rms_epsilon")
    if eps is not None:
        config["layer_norm_epsilon"] = eps
        config.pop("rms_norm_eps", None)

    # Mamba2 SSM parameters
    ssm_inner = _require(
        _read_int(meta, f"{arch}.ssm.inner_size"),
        arch=arch, gguf_field=f"{arch}.ssm.inner_size")
    ssm_time_step_rank = _require(
        _read_int(meta, f"{arch}.ssm.time_step_rank"),
        arch=arch, gguf_field=f"{arch}.ssm.time_step_rank")
    ssm_state_size = _require(
        _read_int(meta, f"{arch}.ssm.state_size"),
        arch=arch, gguf_field=f"{arch}.ssm.state_size")
    conv_kernel = _require(
        _read_int(meta, f"{arch}.ssm.conv_kernel"),
        arch=arch, gguf_field=f"{arch}.ssm.conv_kernel")
    n_groups = _require(
        _read_int(meta, f"{arch}.ssm.group_count"),
        arch=arch, gguf_field=f"{arch}.ssm.group_count")

    config["mamba_num_heads"] = ssm_time_step_rank
    config["mamba_head_dim"] = ssm_inner // ssm_time_step_rank
    config["ssm_state_size"] = ssm_state_size
    config["conv_kernel"] = conv_kernel
    config["n_groups"] = n_groups

    # Bias flags. use_conv_bias=True is architectural for NemotronH
    # (Mamba2 always has conv bias). Shape-based detection fails on
    # split GGUFs where shard 1 has no tensors.
    config["use_conv_bias"] = True
    config["mamba_proj_bias"] = False
    config["attention_bias"] = False
    config["mlp_bias"] = False
    config["use_bias"] = False
    # NemotronH always has a separate lm_head (never tied).
    config["tie_word_embeddings"] = False

    # head_count_kv is per-layer: extract the non-zero value for attention
    # layers (all attention layers share the same GQA group count).
    kv_array = _read_int_array(meta, f"{arch}.attention.head_count_kv")
    if kv_array is not None:
        non_zero = [v for v in kv_array if v > 0]
        if non_zero:
            config["num_key_value_heads"] = non_zero[0]

    # MoE parameters
    n_experts = _read_int(meta, f"{arch}.expert_count")
    if n_experts is not None:
        config["n_routed_experts"] = n_experts
        config["num_experts_per_tok"] = _require(
            _read_int(meta, f"{arch}.expert_used_count"),
            arch=arch, gguf_field=f"{arch}.expert_used_count")
        config["moe_intermediate_size"] = _require(
            _read_int(meta, f"{arch}.expert_feed_forward_length"),
            arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")

        shared_ffn = _read_int(meta, f"{arch}.expert_shared_feed_forward_length")
        if shared_ffn is not None:
            config["moe_shared_expert_intermediate_size"] = shared_ffn

        shared_count = _read_int(meta, f"{arch}.expert_shared_count")
        if shared_count is not None:
            config["n_shared_experts"] = shared_count

        latent_size = _read_int(meta, f"{arch}.moe_latent_size")
        if latent_size is not None:
            config["moe_latent_size"] = latent_size

        config["n_group"] = _read_int(meta, f"{arch}.expert_group_count") or 1
        config["topk_group"] = _read_int(meta, f"{arch}.expert_group_used_count") or 1

        norm_topk = _read_int(meta, f"{arch}.expert_weights_norm")
        config["norm_topk_prob"] = bool(norm_topk) if norm_topk is not None else True

        scale = _read_float(meta, f"{arch}.expert_weights_scale")
        if scale is None:
            scale_int = _read_int(meta, f"{arch}.expert_weights_scale")
            scale = float(scale_int) if scale_int is not None else 1.0
        config["routed_scaling_factor"] = scale

    # intermediate_size for dense MLP layers (type "-"). Use
    # feed_forward_length if uniform, else default to moe_intermediate_size.
    ff_array = _read_int_array(meta, f"{arch}.feed_forward_length")
    if ff_array is not None:
        non_zero_ff = [v for v in ff_array if v > 0]
        if non_zero_ff:
            config["intermediate_size"] = non_zero_ff[0]
    if "intermediate_size" not in config:
        config["intermediate_size"] = config.get("moe_intermediate_size", 0)

    # Derive hybrid_override_pattern from per-layer metadata arrays.
    # head_count_kv > 0 -> attention (*); feed_forward_length > 0 and kv==0
    # -> MoE (E) when experts exist, else dense MLP (-); otherwise -> Mamba (M).
    if kv_array is not None and ff_array is not None:
        n_layers = len(kv_array)
        pattern = []
        for i in range(n_layers):
            if kv_array[i] > 0:
                pattern.append("*")
            elif ff_array[i] > 0:
                pattern.append("E" if n_experts else "-")
            else:
                pattern.append("M")
        config["hybrid_override_pattern"] = pattern
        config["num_hidden_layers"] = n_layers


# deepseek2 (DeepSeek-V2/V3 + GLM-4.x MLA conversions; MLA + fine-grained MoE)

def _synth_deepseek2(meta, shapes, config: dict) -> None:
    """Synthesize a deepseek_v3 config from a 'deepseek2'-arch GGUF.

    llama.cpp emits DeepSeek-V2/V3/R1 *and* the GLM-4.x MLA conversions under
    the single 'deepseek2' arch. We target the V3-style variant - sigmoid
    expert gating with a correction bias + group routing - which mlx_lm's
    deepseek_v3 implements and which the shipped GLM-4.7-Flash / DeepSeek-V3 /
    R1 GGUFs are. A softmax-gated V2 GGUF is rejected with an actionable error
    (mlx_lm's deepseek_v2 + a non-absorbed kv_b remap would be needed).

    Universal fields cover hidden/layers/heads(20)/kv_heads(1, MLA)/eps/
    rope_theta/ctx; intermediate_size = feed_forward_length is the leading
    dense block's MLP width. This adds the MLA head-dim decomposition and the
    V3 MoE block. The MLA up-projections (attn_k_b/attn_v_b) load straight onto
    deepseek_v3's MultiLinear embed_q/unembed_out via KQuantMultiLinear.
    """
    arch = "deepseek2"

    # V2 vs V3: expert_gating_func 1=softmax (V2), 2=sigmoid (V3); the V3
    # correction-bias tensor (exp_probs_b) is the other tell.
    gating = _read_int(meta, f"{arch}.expert_gating_func")
    has_corr = any(_has_tensor(shapes, f"blk.{i}.exp_probs_b.bias")
                   for i in (1, 2, 3))
    if not (gating == 2 or has_corr):
        raise NotImplementedError(
            "deepseek2 synth: this looks like a softmax-gated DeepSeek-V2 "
            "(expert_gating_func != 2, no exp_probs_b correction bias). Only "
            "the V3-style sigmoid-gated variant (deepseek_v3) is supported; "
            "pass hf_source for a V2 config.")
    config["model_type"] = "deepseek_v3"

    # mlx_lm deepseek_v3 has no `head_dim` field; the universal default set it
    # to key_length (= kv_lora + qk_rope), which is not a head dim. Drop it.
    config.pop("head_dim", None)

    num_heads = config["num_attention_heads"]

    # MLA head-dim decomposition
    config["q_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.q_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.q_lora_rank")
    config["kv_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.kv_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.kv_lora_rank")
    qk_rope = _require(
        _read_int(meta, f"{arch}.rope.dimension_count"),
        arch=arch, gguf_field=f"{arch}.rope.dimension_count")
    config["qk_rope_head_dim"] = qk_rope

    # qk_nope_head_dim and v_head_dim from the per-head MLA tensor shapes
    # (robust across llama.cpp metadata-key variations). GGUF-native order:
    #   attn_q_b: [q_lora, num_heads * q_head_dim]   (q_head_dim = nope + rope)
    #   attn_v_b: [kv_lora, v_head_dim, num_heads]   (per-head stacked)
    q_b = shapes.get("blk.0.attn_q_b.weight")
    v_b = shapes.get("blk.0.attn_v_b.weight")
    if q_b is None or v_b is None:
        raise ValueError(
            "deepseek2 synth: need blk.0.attn_q_b.weight + blk.0.attn_v_b.weight "
            "to derive the MLA head dims. Pass hf_source.")
    q_head_dim = q_b[1] // num_heads
    config["qk_nope_head_dim"] = q_head_dim - qk_rope
    config["v_head_dim"] = v_b[1]
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q_a.bias")

    # V3 MoE block
    config["n_routed_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    config["moe_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
    config["n_shared_experts"] = (
        _read_int(meta, f"{arch}.expert_shared_count") or 1)
    config["first_k_dense_replace"] = (
        _read_int(meta, f"{arch}.leading_dense_block_count") or 0)
    config["n_group"] = _read_int(meta, f"{arch}.expert_group_count") or 1
    config["topk_group"] = (
        _read_int(meta, f"{arch}.expert_group_used_count") or 1)
    scale = _read_float(meta, f"{arch}.expert_weights_scale")
    config["routed_scaling_factor"] = scale if scale is not None else 1.0
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm
    config["scoring_func"] = "sigmoid"
    config["topk_method"] = "noaux_tc"
    config["moe_layer_freq"] = 1

    # vocab_size: prefer the explicit GGUF field (the tokenizer token list can
    # be padded relative to the embedding rows).
    vocab = _read_int(meta, f"{arch}.vocab_size")
    if vocab is not None:
        config["vocab_size"] = vocab

    # rope scaling (yarn) passthrough when the GGUF carries it. mlx_lm
    # deepseek_v3 reads `factor` / `mscale_all_dim` off rope_scaling for the
    # attention-scale correction. Absent on non-yarn conversions.
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"type": scaling_type, "rope_type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig is not None:
            rp["original_max_position_embeddings"] = orig
        ymul = _read_float(meta, f"{arch}.rope.scaling.yarn_log_multiplier")
        if ymul is not None:
            rp["mscale_all_dim"] = ymul
        config["rope_scaling"] = rp


# glm-dsa (GLM-5.2; DeepSeek-V3.2 - MLA + fine-grained MoE + DSA indexer + MTP)

def _synth_glm_moe_dsa(meta, shapes, config: dict) -> None:
    """Synthesize a glm_moe_dsa config from a 'glm-dsa'-arch GGUF (GLM-5.2).

    GLM-5.2 is the DeepSeek-V3.2 architecture: the deepseek2 MLA + fine-grained
    sigmoid-gated MoE block, plus a per-layer DSA "lightning indexer" (a cheap
    Q/K scorer that picks the top-k keys for sparse attention) and an MTP/nextn
    layer. mlx-lm's glm_moe_dsa.Model subclasses deepseek_v32.Model (deepseek_v3
    backbone + the Indexer), so this mirrors _synth_deepseek2 and adds the three
    indexer dims + the `rope_parameters` dict glm_moe_dsa.ModelArgs requires. The
    MTP layer is dropped by the universal `block_count - nextn_predict_layers`.
    """
    arch = "glm-dsa"

    # Sigmoid (V3-style) gating tell: expert_gating_func 2 and/or the exp_probs_b
    # correction bias. mlx_lm deepseek_v32 assert-gates topk_method == noaux_tc.
    gating = _read_int(meta, f"{arch}.expert_gating_func")
    has_corr = any(_has_tensor(shapes, f"blk.{i}.exp_probs_b.bias")
                   for i in (1, 2, 3))
    if not (gating == 2 or has_corr):
        raise NotImplementedError(
            "glm-dsa synth: expected a sigmoid-gated (V3-style) MoE "
            "(expert_gating_func == 2 or an exp_probs_b correction bias). "
            "Pass hf_source for an alternate gating variant.")
    config["model_type"] = "glm_moe_dsa"

    # mlx_lm glm_moe_dsa has no `head_dim` field; the universal default set it to
    # key_length (= kv_lora + qk_rope), which is not a head dim. Drop it.
    config.pop("head_dim", None)

    num_heads = config["num_attention_heads"]

    # MLA head-dim decomposition (identical to deepseek2)
    config["q_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.q_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.q_lora_rank")
    config["kv_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.kv_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.kv_lora_rank")
    qk_rope = _require(
        _read_int(meta, f"{arch}.rope.dimension_count"),
        arch=arch, gguf_field=f"{arch}.rope.dimension_count")
    config["qk_rope_head_dim"] = qk_rope

    q_b = shapes.get("blk.0.attn_q_b.weight")
    v_b = shapes.get("blk.0.attn_v_b.weight")
    if q_b is None or v_b is None:
        raise ValueError(
            "glm-dsa synth: need blk.0.attn_q_b.weight + blk.0.attn_v_b.weight "
            "to derive the MLA head dims. Pass hf_source.")
    q_head_dim = q_b[1] // num_heads
    config["qk_nope_head_dim"] = q_head_dim - qk_rope
    config["v_head_dim"] = v_b[1]
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q_a.bias")

    # V3 MoE block (identical to deepseek2)
    config["n_routed_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    config["moe_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
    config["n_shared_experts"] = (
        _read_int(meta, f"{arch}.expert_shared_count") or 1)
    config["first_k_dense_replace"] = (
        _read_int(meta, f"{arch}.leading_dense_block_count") or 0)
    config["n_group"] = _read_int(meta, f"{arch}.expert_group_count") or 1
    config["topk_group"] = (
        _read_int(meta, f"{arch}.expert_group_used_count") or 1)
    scale = _read_float(meta, f"{arch}.expert_weights_scale")
    config["routed_scaling_factor"] = scale if scale is not None else 1.0
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm
    config["scoring_func"] = "sigmoid"
    config["topk_method"] = "noaux_tc"
    config["moe_layer_freq"] = 1

    # DSA "lightning indexer" dims (glm-dsa only). mlx_lm's Indexer reads
    # index_n_heads / index_head_dim / index_topk; q_lora_rank + qk_rope are shared.
    config["index_n_heads"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.head_count"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.head_count")
    config["index_head_dim"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.key_length"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.key_length")
    config["index_topk"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.top_k"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.top_k")

    # vocab_size: prefer the explicit GGUF field.
    vocab = _read_int(meta, f"{arch}.vocab_size")
    if vocab is not None:
        config["vocab_size"] = vocab

    # glm_moe_dsa.ModelArgs.__post_init__ requires a `rope_parameters` dict and
    # reads rope_theta off it (then mirrors it into rope_scaling). Build it from
    # the universal rope_theta and fold in yarn scaling if the GGUF carries it.
    rope_params: dict[str, Any] = {}
    rope_theta = config.get("rope_theta")
    if rope_theta is not None:
        rope_params["rope_theta"] = rope_theta
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rope_params["type"] = scaling_type
        rope_params["rope_type"] = scaling_type
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rope_params["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig is not None:
            rope_params["original_max_position_embeddings"] = orig
        ymul = _read_float(meta, f"{arch}.rope.scaling.yarn_log_multiplier")
        if ymul is not None:
            rope_params["mscale_all_dim"] = ymul
    config["rope_parameters"] = rope_params


# deepseek4 (DeepSeek V4 Flash 256x8.4B)

def _synth_deepseek4(meta, shapes, config: dict) -> None:
    """Synthesize a deepseek_v4 config from a 'deepseek4'-arch GGUF.

    DeepSeek V4 Flash (dwarfstar/antirez conversion; llama.cpp has no support
    for this arch - the GGUF layout is defined by the ds4 reference
    engine's loader). The
    model class is vendored from mlx-lm PR #1192 (gmlx/deepseek_v4_model).

    Shape: single shared 512-dim KV latent (head_count_kv=1, K=V), low-rank q
    and grouped low-rank output projections, per-layer compress_ratios
    selecting local (0) / compressed (128) / sparse-indexed (4) attention,
    hyper-connections (4 streams), every-layer MoE with sqrt-softplus gating
    (expert_gating_func=4), hash routing on the leading ``hash_layer_count``
    layers, and one shared expert.

    Layer-count gotcha: ``nextn_predict_layers=1`` is set in the metadata but
    the MTP layer ships in a separate GGUF (arch deepseek4_mtp_support), so
    ``block_count`` (43) already excludes it. The universal
    ``num_hidden_layers = block_count - nextn`` subtraction is wrong here and
    is reset below. ``compress_ratios`` has ``block_count + nextn`` entries
    (the tail entry is the MTP layer's ratio) and is truncated.
    """
    arch = "deepseek4"
    block_count = _require(
        _read_int(meta, f"{arch}.block_count"),
        arch=arch, gguf_field=f"{arch}.block_count")
    config["num_hidden_layers"] = block_count  # undo the universal nextn cut

    gating = _read_int(meta, f"{arch}.expert_gating_func")
    if gating != 4:
        raise ValueError(
            f"deepseek4 synth: expert_gating_func={gating!r}, but the "
            "deepseek_v4 model class implements the V4-Flash sqrt-softplus "
            "gate (func 4) only.")
    config["scoring_func"] = "sqrtsoftplus"

    moe = _read_v3_moe(meta, shapes, arch)
    config["n_routed_experts"] = moe["n_experts"]
    config["num_experts_per_tok"] = moe["n_used"]
    config["moe_intermediate_size"] = moe["moe_ffn"]
    config["n_shared_experts"] = moe["n_shared"] or 1
    scale = _read_float(meta, f"{arch}.expert_weights_scale")
    config["routed_scaling_factor"] = scale if scale is not None else 1.0
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm
    config["num_hash_layers"] = _read_int(meta, f"{arch}.hash_layer_count") or 0

    # Low-rank attention decomposition. head_dim (512) came from the
    # universal attention.key_length read; kv is a single shared latent.
    config["q_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.q_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.q_lora_rank")
    config["o_lora_rank"] = _require(
        _read_int(meta, f"{arch}.attention.output_lora_rank"),
        arch=arch, gguf_field=f"{arch}.attention.output_lora_rank")
    config["o_groups"] = _require(
        _read_int(meta, f"{arch}.attention.output_group_count"),
        arch=arch, gguf_field=f"{arch}.attention.output_group_count")
    config["qk_rope_head_dim"] = _require(
        _read_int(meta, f"{arch}.rope.dimension_count"),
        arch=arch, gguf_field=f"{arch}.rope.dimension_count")
    config["sliding_window"] = _require(
        _read_int(meta, f"{arch}.attention.sliding_window"),
        arch=arch, gguf_field=f"{arch}.attention.sliding_window")

    # Lightning-indexer dims (ratio-4 layers).
    config["index_n_heads"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.head_count"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.head_count")
    config["index_head_dim"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.key_length"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.key_length")
    config["index_topk"] = _require(
        _read_int(meta, f"{arch}.attention.indexer.top_k"),
        arch=arch, gguf_field=f"{arch}.attention.indexer.top_k")

    # Per-layer attention-variant selection; entries beyond block_count
    # describe MTP layers living in the companion GGUF.
    ratios = _require(
        _read_int_array(meta, f"{arch}.attention.compress_ratios"),
        arch=arch, gguf_field=f"{arch}.attention.compress_ratios")
    if len(ratios) < block_count:
        raise ValueError(
            f"deepseek4 synth: compress_ratios has {len(ratios)} entries for "
            f"{block_count} layers.")
    config["compress_ratios"] = [int(r) for r in ratios[:block_count]]
    compress_theta = _read_float(meta, f"{arch}.attention.compress_rope_freq_base")
    if compress_theta is not None:
        config["compress_rope_theta"] = compress_theta

    # Yarn rope scaling applies to the compressed/sparse layers only (the
    # model class wires it there); local layers use the plain rope_theta.
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rope_scaling: dict[str, Any] = {"type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rope_scaling["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig is not None:
            rope_scaling["original_max_position_embeddings"] = orig
        beta_fast = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_fast")
        if beta_fast is not None:
            rope_scaling["beta_fast"] = beta_fast
        beta_slow = _read_float(meta, f"{arch}.rope.scaling.yarn_beta_slow")
        if beta_slow is not None:
            rope_scaling["beta_slow"] = beta_slow
        config["rope_scaling"] = rope_scaling

    # Hyper-connections (4 parallel residual streams, sinkhorn mixing).
    config["hc_mult"] = _require(
        _read_int(meta, f"{arch}.hyper_connection.count"),
        arch=arch, gguf_field=f"{arch}.hyper_connection.count")
    hc_iters = _read_int(meta, f"{arch}.hyper_connection.sinkhorn_iterations")
    if hc_iters is not None:
        config["hc_sinkhorn_iters"] = hc_iters
    hc_eps = _read_float(meta, f"{arch}.hyper_connection.epsilon")
    if hc_eps is not None:
        config["hc_eps"] = hc_eps

    # Clipped SwiGLU: per-layer array in the GGUF, uniform on every known
    # V4 conversion; the model class takes one scalar.
    clamps = _read_float_array(meta, f"{arch}.swiglu_clamp_exp")
    if clamps:
        uniq = set(clamps[:block_count])
        if len(uniq) != 1:
            raise ValueError(
                f"deepseek4 synth: non-uniform swiglu_clamp_exp {sorted(uniq)} "
                "- the deepseek_v4 model class supports one global limit.")
        config["swiglu_limit"] = float(clamps[0])

    nextn = _read_int(meta, f"{arch}.nextn_predict_layers")
    if nextn is not None:
        config["num_nextn_predict_layers"] = nextn

    # vocab_size: prefer the explicit GGUF field (the universal read derives
    # it from the token array, which header-only scans truncate).
    vocab = _read_int(meta, f"{arch}.vocab_size")
    if vocab is not None:
        config["vocab_size"] = vocab


# glm4moe (GLM-4.5 / 4.6; MHA + DeepSeek-V3-style fine-grained MoE)

def _synth_glm4moe(meta, shapes, config: dict) -> None:
    """Synthesize a glm4_moe config from a 'glm4moe'-arch GGUF.

    Same fine-grained V3 MoE block as deepseek2 (sigmoid gating + correction
    bias + group routing + a shared expert behind a leading dense block), but
    with a *standard* MHA attention (q/k/v/o, optional qk-norm, partial rotary)
    instead of MLA - so the universal fields already cover the backbone
    (hidden/layers/heads/kv_heads/intermediate/eps/head_dim/rope_theta/ctx/vocab)
    and this only adds the attention details + the MoE block. The MTP/NextN layer
    is excluded from num_hidden_layers via the universal nextn_predict_layers
    subtraction; its tensors are skipped/dropped on load.
    """
    arch = "glm4moe"

    # Partial rotary: only rope.dimension_count of head_dim is rotated. mlx_lm
    # glm4_moe requires both head_dim (set by the universal extractor from
    # key_length) and partial_rotary_factor.
    head_dim = config.get("head_dim")
    if not head_dim:
        nh = config["num_attention_heads"]
        head_dim = config["hidden_size"] // nh
        config["head_dim"] = head_dim
    rope_dim = _read_int(meta, f"{arch}.rope.dimension_count")
    config["partial_rotary_factor"] = (rope_dim / head_dim) if rope_dim else 1.0

    # qk-norm and attention bias from the tensor inventory (GLM-4.5 uses qk-norm;
    # the bias presence varies by checkpoint).
    config["use_qk_norm"] = _has_tensor(shapes, "blk.0.attn_q_norm.weight")
    config["attention_bias"] = _has_tensor(shapes, "blk.0.attn_q.bias")

    # V3-style MoE block
    config["n_routed_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    config["moe_intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")
    config["n_shared_experts"] = (
        _read_int(meta, f"{arch}.expert_shared_count") or 1)
    config["first_k_dense_replace"] = (
        _read_int(meta, f"{arch}.leading_dense_block_count") or 0)
    config["n_group"] = _read_int(meta, f"{arch}.expert_group_count") or 1
    config["topk_group"] = (
        _read_int(meta, f"{arch}.expert_group_used_count") or 1)
    scale = _read_float(meta, f"{arch}.expert_weights_scale")
    config["routed_scaling_factor"] = scale if scale is not None else 1.0
    norm = _read_bool(meta, f"{arch}.expert_weights_norm")
    config["norm_topk_prob"] = True if norm is None else norm
    config["scoring_func"] = "sigmoid"
    config["topk_method"] = "noaux_tc"

    # rope scaling (yarn) passthrough when present. glm4_moe's ModelArgs requires
    # the field, so default it to None for the (common) non-yarn case.
    config["rope_scaling"] = None
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"type": scaling_type, "rope_type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig is not None:
            rp["original_max_position_embeddings"] = orig
        config["rope_scaling"] = rp


# gpt-oss (OpenAI gpt-oss 20B / 120B; MoE + attention sinks + sliding/full + MXFP4)

def _synth_gpt_oss(meta, shapes, config: dict) -> None:
    """Synthesize a gpt_oss config from a 'gpt-oss'-arch GGUF.

    Universal fields cover hidden/layers/heads(64)/kv_heads(8)/eps/head_dim
    (key_length 64)/rope_theta(freq_base)/ctx/vocab and tie (untied: output.weight
    present). This adds the MoE block, the sliding window, the YaRN rope scaling,
    and the per-layer sliding/full attention pattern.

    The attention has per-head learned sinks and alternates sliding-window and
    full attention (mlx_lm builds `self.sinks` and reads `layer_types`); the
    experts are MXFP4 SwitchGLU (handled by the loader's native-fp repack +
    module, not here). No `query_pre_attn_scalar` / softcap - gpt-oss uses a
    plain 1/sqrt(head_dim) attention scale.
    """
    arch = "gpt-oss"

    config["num_local_experts"] = _require(
        _read_int(meta, f"{arch}.expert_count"),
        arch=arch, gguf_field=f"{arch}.expert_count")
    config["num_experts_per_tok"] = _require(
        _read_int(meta, f"{arch}.expert_used_count"),
        arch=arch, gguf_field=f"{arch}.expert_used_count")
    # Expert MLP width - gpt-oss has no dense MLP; the SwitchGLU hidden dim is
    # `intermediate_size`, which is the expert (not the absent dense) width.
    config["intermediate_size"] = _require(
        _read_int(meta, f"{arch}.expert_feed_forward_length"),
        arch=arch, gguf_field=f"{arch}.expert_feed_forward_length")

    sliding = _read_int(meta, f"{arch}.attention.sliding_window")
    if sliding is not None:
        config["sliding_window"] = sliding

    # YaRN rope: GGUF carries type/factor/original_context_length; the betas are
    # gpt-oss's standard yarn defaults (mlx_lm's initialize_rope supplies them).
    scaling_type = _read_string(meta, f"{arch}.rope.scaling.type")
    if scaling_type and scaling_type != "none":
        rp: dict[str, Any] = {"rope_type": scaling_type, "type": scaling_type}
        factor = _read_float(meta, f"{arch}.rope.scaling.factor")
        if factor is not None:
            rp["factor"] = factor
        orig = _read_int(meta, f"{arch}.rope.scaling.original_context_length")
        if orig is not None:
            rp["original_max_position_embeddings"] = orig
        config["rope_scaling"] = rp

    # Per-layer attention pattern: gpt-oss alternates sliding/full starting with
    # sliding (layer 0). GGUF carries no explicit pattern array, so synthesize the
    # canonical alternation; mlx_lm's gpt_oss default is identical, but set it
    # explicitly so make_cache / mask selection is unambiguous.
    n_layers = config["num_hidden_layers"]
    config["layer_types"] = [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(n_layers)]

    # vocab_size from the lm_head rows (output.weight), authoritative vs a
    # possibly-padded tokenizer token list.
    out_shape = shapes.get("output.weight")
    if out_shape is not None and len(out_shape) == 2:
        config["vocab_size"] = out_shape[1]   # GGUF native [hidden, vocab]


# Reporting

def _print_summary(config: dict, arch: str) -> None:
    # Wrapper configs (e.g. gemma3n) nest the backbone under `text_config`.
    inner = config.get("text_config", config)
    bits = [
        f"model_type={config.get('model_type')}",
        f"hidden={inner.get('hidden_size')}",
        f"layers={inner.get('num_hidden_layers')}",
        f"heads={inner.get('num_attention_heads')}",
        f"kv_heads={inner.get('num_key_value_heads')}",
        f"vocab={inner.get('vocab_size')}",
    ]
    if "num_kv_shared_layers" in inner:
        bits.append(f"kv_shared={inner['num_kv_shared_layers']}")
    if inner.get("enable_moe_block") or inner.get("num_experts"):
        bits.append(f"experts={inner.get('num_experts')}")
    loadlog.verbose_print("[config] synthesized: " + " ".join(bits))


# The per-arch dispatch: GGUF arch string -> synthesizer. Single source of
# truth for which arches synthesize_config() completes (`_supported()` is its
# key set). Multi-arch synthesizers bind their arch via lambda so every entry
# is callable as fn(meta, shapes, config).
_SYNTH = {
    "gemma4": _synth_gemma4,
    "qwen3": _synth_qwen3,
    "qwen3moe": lambda m, s, c: _synth_qwen3moe(m, s, c, "qwen3moe"),
    "qwen3vlmoe": lambda m, s, c: _synth_qwen3moe(m, s, c, "qwen3vlmoe"),
    "qwen2": _synth_qwen2,
    "qwen2moe": _synth_qwen2moe,
    "gemma2": _synth_gemma2,
    "gemma": _synth_gemma,
    "phi3": _synth_phi3,
    "glm4": _synth_glm4,
    "qwen35": lambda m, s, c: _synth_qwen35(m, s, c, "qwen35"),
    "qwen35moe": lambda m, s, c: _synth_qwen35(m, s, c, "qwen35moe"),
    "mistral3": _synth_mistral3,
    "nemotron_h_moe": _synth_nemotron_h_moe,
    "deepseek2": _synth_deepseek2,
    "glm-dsa": _synth_glm_moe_dsa,
    "deepseek4": _synth_deepseek4,
    "glm4moe": _synth_glm4moe,
    "gpt-oss": _synth_gpt_oss,
    "llama": _synth_llama,
    "seed_oss": _synth_seed_oss,
    "smollm3": _synth_smollm3,
    "granite": _synth_granite,
    "ernie4_5-moe": _synth_ernie4_5_moe,
    "minimax-m2": _synth_minimax,
    "minimax-m3": _synth_minimax_m3,
    "hunyuan-moe": _synth_hunyuan,
    "hy_v3": _synth_hy_v3,
    "granitehybrid": _synth_granite_hybrid,
    "falcon-h1": _synth_falcon_h1,
    "qwen3next": _synth_qwen3next,
    "gemma3": _synth_gemma3,
    "gemma-embedding": _synth_gemma_embedding,
    "gemma3n": _synth_gemma3n,
    "diffusion-gemma": _synth_diffusion_gemma,
}
