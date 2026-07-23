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
import random
import re
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import mlx_kquant as kq

from . import loadlog
from .envflags import env_bool, env_choice, env_float, env_int
from .attn_hd512 import install_hd512_sdpa
from .prefill_decay import install_prefill_decay, note_untracked_weights
from . import gpt_oss_prefill  # noqa: F401  (registers gpt_oss score profile)
from .modules import install_fused_moe_glu, install_hyv3_shexp_fold
from .qkv_fuse import install_fused_qkv
from .qwen35_verify_fold import install_qwen35_verify_fold
from .rotating_cache_fix import install_rotating_cache_fix
from .modules import KQuantEmbedding, install_kquant_modules
from .populate import (
    maybe_populate_for_load,
    start_populate,
    wait_for as wait_for_populate,
)
from .preflight import find_split_shards, preflight
from .dsv32_patches import (
    _patch_dsv32_dense_default,
    _patch_dsv32_indexer_fp32,
    _patch_dsv32_indexer_rope,
    _patch_dsv32_mask_decode,
    _patch_dsv32_moe_gate_fp32,
    _patch_dsv32_moe_scores,
)
from .gdn_patches import (
    _needs_tiled_v_patch,
    _patch_gated_delta_fused_decode,
    _patch_gated_delta_tiled_v,
    _patch_qwen3next_split_gdn,
    _tiled_v_patch_applied,
)
from .gguf_meta import first_nonzero_int, read_int
from .native_fp import _strip_weight
from .remap import RemapDecision, parse_gguf_name
from .transforms import (
    coalesce_split_experts,
    fuse_shexp_gate_up,
    qk_permute_wire,
    retarget,
    split_fused_gate_up_kquant,
)


# GGUF wire-byte loading


def load_gguf_wire_bytes(
    gguf_path: str,
    zero_copy: bool = True,
    shards: list[str] | None = None,
    expect_quant: bool = True,
) -> tuple[dict[str, mx.array], dict[str, str], str | None, dict, dict]:
    """Load GGUF tensors as raw kquant wire bytes via the C++ ``kq.load_gguf``.

    ``kq.load_gguf`` reads every supported quant codec (K-quant, legacy, IQ)
    as uint8 wire
    bytes with a vestigial ``<prefix>.scales`` placeholder, and F32/F16/BF16/
    I8/I16/I32 tensors with their native dtype. By default (``zero_copy=True``)
    each tensor is a no-copy view over gguflib's mmap; ``zero_copy=False``
    memcpy's every tensor out of the mmap in C++. It also decodes all GGUF KV
    metadata, so no gguf-py GGUFReader is opened in the load path.

    Returns ``(arrays, kquant_meta, arch, meta, tensor_shapes)``:
      - ``arch`` is ``general.architecture`` from the first shard's metadata, or
        None if absent (caller may override).
      - ``meta`` is the decoded GGUF KV dict (key -> int/float/bool/str/list).
      - ``tensor_shapes`` is tensor name -> logical shape (GGUF native order).

    Handles split GGUFs by loading all shards and merging; metadata +
    tensor_shapes come from the first shard. ``shards`` may be passed (e.g. from
    a prior preflight pass) to skip re-discovery.
    """
    if shards is None:
        shards = find_split_shards(gguf_path)
    arrays: dict[str, mx.array] = {}
    kquant_meta: dict[str, str] = {}
    meta: dict = {}
    tensor_shapes: dict = {}
    for i, shard in enumerate(shards):
        s_arrays, s_codecs, s_meta, s_shapes = kq.load_gguf(shard, zero_copy)
        arrays.update(s_arrays)
        kquant_meta.update(s_codecs)
        tensor_shapes.update(s_shapes)
        if i == 0:
            meta = s_meta
    if len(shards) > 1:
        loadlog.verbose_print(
            f"[gguf] loaded {len(shards)} shards, {len(arrays)} total tensors"
        )

    if expect_quant and not kquant_meta:
        loadlog.warn(
            "WARNING: no quantized tensors found - is this actually a K-quant GGUF?"
        )

    arch = meta.get("general.architecture")
    return arrays, kquant_meta, arch, meta, tensor_shapes


# Tensor-name remap + layout transforms


class _RemapDict(dict):
    """Weight sink that refuses silent clobbers: two GGUF tensors remapping to
    the same target name is a table bug, never a legitimate overwrite."""

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(
                f"tensor remap collision: two source tensors map to {key!r}")
        dict.__setitem__(self, key, value)


def _own(arr: mx.array) -> mx.array:
    """Return an owned copy of ``arr`` decoupled from the source GGUF mapping.

    With zero-copy loading, native (non-quantized) tensors are views over a
    file-backed shared mapping. An in-place elementwise transform on such a view
    can be fused by the array library's buffer-donation optimization into a
    write *through* the mapping, mutating the file on disk. Copying the data out
    to host first breaks that aliasing, so the transform result is computed in a
    private buffer and the source file is never touched. Used only by the small
    arithmetic transforms (RMSNorm-unbake, SSM ``A``), where the cost is
    negligible; bulk quantized tensors stay zero-copy.
    """
    if arr.dtype == mx.bfloat16:
        # numpy has no bf16 buffer format; both transform call sites compute
        # in f32 anyway. astype allocates a fresh buffer, never the mapping.
        arr = arr.astype(mx.float32)
    return mx.array(np.array(arr))


def remap_arrays(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
    arch: str,
    *,
    no_remap: bool = False,
    target_prefix: str = "",
    fail_on_unknown: bool = False,
    n_head: int | None = None,
    n_head_kv: int | None = None,
    owned_names: set[str] | None = None,
) -> tuple[dict[str, mx.array], dict[str, str], dict[str, int]]:
    """Apply name remap + layout transforms to GGUF arrays.

    Returns ``(hf_weights, hf_kquant_meta, stats)`` where ``hf_kquant_meta``
    maps the post-remap tensor name to its codec string.

    ``n_head`` / ``n_head_kv`` are required when any tensor needs the LLAMA Q/K
    permute applied. When omitted, the qk_permute transform falls back to a
    pass-through with a warning (the resulting model mis-attends).

    ``owned_names``, when given, collects the post-remap names of arithmetic
    transform results (qk_permute, SSM A, gemma norm-unbake): arrays that must
    own their buffers, never alias the source mapping (donation tripwire; see
    ``_verify_zero_copy_views``). Shape-op transforms legitimately alias and
    are not collected.
    """
    hf_weights: dict[str, mx.array] = _RemapDict()
    hf_kquant_meta: dict[str, str] = {}
    stats = {
        "mapped": 0,
        "skipped": 0,
        "split": 0,
        "failed": 0,
        "passthrough": 0,
        "qk_permute_applied": 0,
        "qk_permute_skipped": 0,
        "conv1d_unsqueeze": 0,
        "gemma_norm_minus_one": 0,
    }

    # We process weight tensors; .scales sibling placeholders produced by the
    # wire-byte loader get re-emitted alongside their weight under the HF name.
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        codec = kquant_meta.get(name)

        if no_remap:
            hf_name = name
            transform = "passthrough"
        else:
            dec = parse_gguf_name(arch, name)
            if dec.kind == RemapDecision.KIND_SKIP:
                stats["skipped"] += 1
                continue
            if dec.kind == RemapDecision.KIND_FAIL:
                if fail_on_unknown:
                    raise RuntimeError(f"unmapped tensor {name!r}: {dec.reason}")
                loadlog.warn(
                    f"WARNING: skipping unmapped tensor {name!r}: {dec.reason}"
                )
                stats["failed"] += 1
                continue
            hf_name = retarget(dec.hf_name, target_prefix)
            transform = dec.transform

        if transform == "passthrough":
            hf_weights[hf_name] = arr
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["passthrough"] += 1
            stats["mapped"] += 1

        elif transform == "moe_split_gate_up":
            base = hf_name[: -len("gate_up_proj.weight")].rstrip(".")
            gate_name = f"{base}.gate_proj.weight"
            up_name = f"{base}.up_proj.weight"
            gate, up = split_fused_gate_up_kquant(arr)
            hf_weights[gate_name] = gate
            hf_weights[up_name] = up
            if codec is not None:
                # Both halves get a vestigial scales entry under their own name.
                hf_weights[_strip_weight(gate_name) + ".scales"] = mx.zeros(
                    (1,), dtype=mx.uint8
                )
                hf_weights[_strip_weight(up_name) + ".scales"] = mx.zeros(
                    (1,), dtype=mx.uint8
                )
                hf_kquant_meta[gate_name] = codec
                hf_kquant_meta[up_name] = codec
            stats["split"] += 1
            stats["mapped"] += 2

        elif transform == "altup_split":
            # gemma-3n stores the AltUp (un)projections as one stacked 3-D
            # tensor; the MLX-native layout (GGUF dims reversed) is
            # (altup_num_inputs-1, out, in). mlx_lm wants a list of separate
            # Linears, so emit `{base}.{i}.weight` per stack slice. These are
            # plain F16 tensors (not kquant), so a pure array slice suffices.
            base = _strip_weight(hf_name)
            for i in range(arr.shape[0]):
                hf_weights[f"{base}.{i}.weight"] = arr[i]
                stats["mapped"] += 1
            stats["split"] += 1

        elif transform == "qk_permute":
            # llama.cpp's convert_hf_to_gguf::LlamaModel.permute reorders Q/K
            # rows so ggml's interleaved-pairs RoPE matches HF's concat-half
            # RoPE. mlx-lm's llama/mistral3 attention uses the HF layout, so we
            # undo the permute when loading from GGUF directly.
            is_k = hf_name.endswith("k_proj.weight")
            n_heads_for = n_head_kv if (is_k and n_head_kv is not None) else n_head
            if n_heads_for is None:
                loadlog.warn(
                    f"WARNING: qk_permute requested for {hf_name!r} but "
                    f"n_head/n_head_kv not provided; loading without "
                    f"permute (attention will be wrong)."
                )
                hf_weights[hf_name] = arr
                stats["qk_permute_skipped"] += 1
            else:
                hf_weights[hf_name] = qk_permute_wire(arr, n_heads_for)
                stats["qk_permute_applied"] += 1
                if owned_names is not None:
                    owned_names.add(hf_name)
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["mapped"] += 1

        elif transform == "conv1d_unsqueeze":
            # Pure shape op; works on any dtype. Mamba conv weights are
            # F32/BF16 (not kquant), so codec is None here.
            hf_weights[hf_name] = arr[..., None]
            stats["conv1d_unsqueeze"] += 1
            stats["mapped"] += 1

        elif transform == "ssm_a_to_a_log":
            # GGUF stores SSM_A as -exp(A_log); invert to recover A_log.
            # Squeeze extra leading dim (nemotron_h stores as [1, N]).
            # _own() first: the negate/log would otherwise be donated into the
            # source mapping (see _own docstring).
            out = mx.log(-_own(arr).astype(mx.float32))
            hf_weights[hf_name] = out.reshape(-1) if out.ndim > 1 else out
            stats["mapped"] += 1
            if owned_names is not None:
                owned_names.add(hf_name)

        elif transform == "flatten":
            # Reshape multi-dim tensor to 1D (e.g. nemotron_h ssm_norm stored as
            # [n_groups, group_size], ssm_d stored as [1, N]).
            hf_weights[hf_name] = arr.reshape(-1)
            stats["mapped"] += 1

        elif transform == "gate_1d_unsqueeze":
            # Shared expert gate: GGUF stores 1D [hidden_size], but
            # nn.Linear(hidden_size, 1, bias=False) has weight [1, hidden_size].
            hf_weights[hf_name] = arr.reshape(1, -1) if arr.ndim == 1 else arr
            stats["mapped"] += 1

        elif transform == "gemma_norm_minus_one":
            # llama.cpp bakes +1 into gemma RMSNorm weights at conversion (the
            # GGUF stores hf_weight + 1, used directly by ggml). mlx_lm's
            # gemma/gemma2/gemma3 RMSNorm computes rms_norm(x, 1.0 + weight),
            # i.e. it expects the *raw* HF weight - so undo the bake here.
            # (gemma4_text uses its norm weight directly and is not tagged.)
            # _own() first so the subtract isn't donated back into the source
            # mapping (see _own docstring).
            hf_weights[hf_name] = _own(arr).astype(mx.float32) - 1.0
            stats["gemma_norm_minus_one"] += 1
            stats["mapped"] += 1
            if owned_names is not None:
                owned_names.add(hf_name)

        else:
            raise RuntimeError(f"unknown transform {transform!r} for {name!r}")

    # Hand back a plain dict: the anti-clobber guard applies to remap
    # population only. Later stages (native-fp repack, transforms)
    # legitimately replace entries in place.
    return dict(hf_weights), hf_kquant_meta, stats


# MTP / "nextn" drafter remap (native-head: the drafter weights live in the
# GGUF's own MTP block, i.e. block index >= num_hidden_layers)

# The four ``nextn.*`` extras -> the mlx-vlm ``Qwen3_5MTPDraftModel`` param tree.
# The MTP block's *standard* decoder tensors (attn_*, ffn_*, the two block norms)
# reuse the canonical text remap (``parse_gguf_name``) with ``model.layers.{N}.``
# rewritten to the drafter's ``layers.{i}.``. The embed table + LM head are not
# here - the drafter binds the target's at runtime (qwen3.5/3.6 GGUFs carry no
# ``nextn.embed_tokens`` / ``nextn.shared_head_head``).
_MTP_NEXTN_MAP = {
    "eh_proj": "fc.weight",
    "enorm": "pre_fc_norm_embedding.weight",
    "hnorm": "pre_fc_norm_hidden.weight",
    "shared_head_norm": "norm.weight",
}


def remap_mtp_arrays(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
    arch: str,
    *,
    first_mtp_block: int,
    num_mtp_layers: int = 1,
    n_head: int | None = None,
    n_head_kv: int | None = None,
) -> tuple[dict[str, mx.array], dict[str, str], dict[str, int]]:
    """Remap a GGUF's native MTP block(s) onto the drafter's ``mtp.*`` tree.

    ``first_mtp_block`` is the GGUF block index of the first MTP block (equals
    the target's ``num_hidden_layers``); block ``first_mtp_block + i`` maps to the
    drafter's ``layers.{i}``. Returns drafter-relative names (no ``model.``
    prefix); the caller builds the drafter and ``load_weights`` these onto it.

    Self-contained (does not touch the text-path ``remap_arrays``): it reuses
    ``parse_gguf_name`` for the standard decoder tensors' name+transform decision
    and the shared standalone transforms for emit.
    """
    hf_weights: dict[str, mx.array] = _RemapDict()
    hf_kquant_meta: dict[str, str] = {}
    stats = {
        "mapped": 0,
        "skipped": 0,
        "split": 0,
        "passthrough": 0,
        "qk_permute_applied": 0,
        "qk_permute_skipped": 0,
        "conv1d_unsqueeze": 0,
    }

    def _emit(hf_name: str, transform: str, arr, codec, src_name: str) -> None:
        if transform == "passthrough":
            hf_weights[hf_name] = arr
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(src_name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["passthrough"] += 1
            stats["mapped"] += 1
        elif transform == "moe_split_gate_up":
            base = hf_name[: -len("gate_up_proj.weight")].rstrip(".")
            gate_name = f"{base}.gate_proj.weight"
            up_name = f"{base}.up_proj.weight"
            gate, up = split_fused_gate_up_kquant(arr)
            hf_weights[gate_name] = gate
            hf_weights[up_name] = up
            if codec is not None:
                hf_weights[_strip_weight(gate_name) + ".scales"] = mx.zeros(
                    (1,), dtype=mx.uint8
                )
                hf_weights[_strip_weight(up_name) + ".scales"] = mx.zeros(
                    (1,), dtype=mx.uint8
                )
                hf_kquant_meta[gate_name] = codec
                hf_kquant_meta[up_name] = codec
            stats["split"] += 1
            stats["mapped"] += 2
        elif transform == "gate_1d_unsqueeze":
            hf_weights[hf_name] = arr.reshape(1, -1) if arr.ndim == 1 else arr
            stats["mapped"] += 1
        elif transform == "flatten":
            hf_weights[hf_name] = arr.reshape(-1)
            stats["mapped"] += 1
        elif transform == "qk_permute":
            is_k = hf_name.endswith("k_proj.weight")
            nh = n_head_kv if (is_k and n_head_kv is not None) else n_head
            if nh is None:
                loadlog.warn(
                    f"WARNING: qk_permute requested for {hf_name!r} but "
                    f"n_head/n_head_kv not provided; loading without "
                    f"permute (attention will be wrong)."
                )
                hf_weights[hf_name] = arr
                stats["qk_permute_skipped"] += 1
            else:
                hf_weights[hf_name] = qk_permute_wire(arr, nh)
                stats["qk_permute_applied"] += 1
            if codec is not None:
                hf_weights[_strip_weight(hf_name) + ".scales"] = arrays[
                    _strip_weight(src_name) + ".scales"
                ]
                hf_kquant_meta[hf_name] = codec
            stats["mapped"] += 1
        elif transform == "conv1d_unsqueeze":
            hf_weights[hf_name] = arr[..., None]
            stats["conv1d_unsqueeze"] += 1
            stats["mapped"] += 1
        else:
            raise RuntimeError(
                f"MTP remap: unsupported transform {transform!r} for {src_name!r}"
            )

    mtp_blocks = {first_mtp_block + i: i for i in range(num_mtp_layers)}
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        m = re.match(r"^blk\.(\d+)\.(.+)$", name)
        if not m:
            continue
        blk = int(m.group(1))
        if blk not in mtp_blocks:
            continue
        layer_i = mtp_blocks[blk]
        rest = m.group(2)
        codec = kquant_meta.get(name)
        if rest.startswith("nextn."):
            key = rest[len("nextn.") :]
            base = key[: -len(".weight")] if key.endswith(".weight") else key
            target = _MTP_NEXTN_MAP.get(base)
            if target is None:
                # e.g. nextn.embed_tokens / shared_head_head - shared from target.
                stats["skipped"] += 1
                continue
            _emit(target, "passthrough", arr, codec, name)
        else:
            dec = parse_gguf_name(arch, name)
            if dec.kind != RemapDecision.KIND_MAP:
                stats["skipped"] += 1
                continue
            marker = f"model.layers.{blk}."
            if marker not in dec.hf_name:
                stats["skipped"] += 1
                continue
            inner = dec.hf_name.split(marker, 1)[1]
            _emit(f"layers.{layer_i}.{inner}", dec.transform, arr, codec, name)
    # Hand back a plain dict: the anti-clobber guard applies to remap
    # population only. Later stages (native-fp repack, transforms)
    # legitimately replace entries in place.
    return dict(hf_weights), hf_kquant_meta, stats


def remap_gemma4_assistant_arrays(arrays: dict, kquant_meta: dict):
    """Remap a gemma4 assistant-drafter GGUF onto the mlx-vlm
    ``Gemma4AssistantDraftModel`` param tree.

    The standard decoder / embed / norm tensors reuse the canonical gemma4 remap
    (``parse_gguf_name`` already emits the exact ``model.*`` names the drafter
    uses, including ``layer_output_scale -> layers.N.layer_scalar`` and
    ``output_norm -> model.norm``); only the two bridge projections need renaming
    and ``rope_freqs`` is dropped (it's computed, not a param). Every gemma4
    tensor maps as a passthrough (no qk-permute), so the emit is direct.
    """
    hf_weights: dict[str, mx.array] = _RemapDict()
    hf_kquant_meta: dict[str, str] = {}
    stats = {"mapped": 0, "skipped": 0}
    for name, arr in arrays.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        if base.endswith("pre_proj") or base.endswith("pre_projection"):
            hf = "pre_projection.weight"
        elif base.endswith("post_proj") or base.endswith("post_projection"):
            hf = "post_projection.weight"
        elif base.endswith("centroids"):
            # ordered-embeddings sparse head (E2B/E4B); Q8_0, swapped by kquant.
            hf = "masked_embedding.centroids.weight"
        elif base.endswith("token_ordering"):
            # I32 index vector, no .weight suffix on the param, never quantized.
            hf_weights["masked_embedding.token_ordering"] = arr.astype(mx.int32)
            stats["mapped"] += 1
            continue
        elif base == "rope_freqs":
            stats["skipped"] += 1
            continue
        else:
            dec = parse_gguf_name("gemma4", name)
            if dec.kind != RemapDecision.KIND_MAP:
                stats["skipped"] += 1
                continue
            if dec.transform != "passthrough":
                raise RuntimeError(
                    f"gemma4 assistant remap: unexpected transform "
                    f"{dec.transform!r} for {name!r}"
                )
            hf = dec.hf_name
        codec = kquant_meta.get(name)
        hf_weights[hf] = arr
        if codec is not None:
            hf_weights[_strip_weight(hf) + ".scales"] = arrays[
                _strip_weight(name) + ".scales"
            ]
            hf_kquant_meta[hf] = codec
        stats["mapped"] += 1
    # Hand back a plain dict: the anti-clobber guard applies to remap
    # population only. Later stages (native-fp repack, transforms)
    # legitimately replace entries in place.
    return dict(hf_weights), hf_kquant_meta, stats


# MTP target wrapper + capability resolver

# The hooks the mlx-vlm MTP engine probes on the *target*'s ``language_model``.
# Only ``rollback_speculative_cache`` is hard-required by the engine; the rest are
# pinned as version tripwires (a mlx-vlm bump that renames/drops one fails the
# hook-contract smoke loudly instead of corrupting decode). qwen3.5/3.6 expose
# the full verify_* set; gemma4 has a leaner set (it drafts via an assistant
# model + ``speculative_draft_hidden`` rather than the verify_* hooks).
_MTP_TARGET_HOOKS = (
    "rollback_speculative_cache",
    "speculative_logits_from_hidden",
    "speculative_argmax_from_hidden",
    "speculative_verify_logits",
    "speculative_verify_hidden",
)
_MTP_TARGET_HOOKS_BY_TYPE = {
    "gemma4_text": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_draft_hidden",
    ),
    # DeepseekV4SpecLM (vendored mlx-lm class, not mlx-vlm): no
    # speculative_verify_logits -- verify goes through verify_hidden and the
    # walk computes logits/argmax from the raw 4D hidden.
    "deepseek_v4": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
    # HyV3SpecLM (vendored mlx-lm class): same lean set as deepseek_v4.
    "hy_v3": (
        "rollback_speculative_cache",
        "speculative_logits_from_hidden",
        "speculative_argmax_from_hidden",
        "speculative_verify_hidden",
    ),
}


class MTPTextTarget(nn.Module):
    """Expose an mlx-vlm text ``LanguageModel`` as ``.language_model``.

    The MTP engine reaches the target through ``model.language_model`` (for the
    ``speculative_*`` hooks + ``hidden_states``), and the drafter's ``bind``
    walks ``.language_model.model.embed_tokens``. This is deliberately not the
    serving ``TextOnlyModel`` wrapper, whose ``.language_model`` is a logits-only
    adapter with none of those hooks.
    """

    def __init__(self, language_model, config: dict):
        super().__init__()
        self.language_model = language_model
        self.config = config

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        """Text-only embedding lookup the MTP engine calls on the top-level
        model (``mlx_vlm.generate.ar.generate_step``). Mirrors the qwen3.5 VLM
        ``Model``'s text-only branch - a GGUF text target has no vision tower -
        returning an ``InputEmbeddingsFeatures`` whose ``inputs_embeds`` is the
        token embedding. Clears ``_position_ids`` so mrope falls back to the
        plain text positions."""
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        self.language_model._position_ids = None
        embeds = self.language_model.model.embed_tokens(input_ids)
        # gemma4 scales token embeddings by sqrt(hidden) in its input_ids path
        # (Gemma4Model.__call__), but the inputs_embeds path does not - so a
        # target fed embeds must get them pre-scaled. qwen has no such scale.
        if self.config.get("model_type") == "gemma4_text":
            embeds = embeds * self.language_model.model.embed_scale
        return InputEmbeddingsFeatures(inputs_embeds=embeds)

    def __call__(self, *args, **kwargs):
        out = self.language_model(*args, **kwargs)
        if isinstance(out, mx.array):
            # mlx-lm-style SpecLMs (deepseek_v4) return raw logits on the
            # plain path; mlx-vlm's AR engine expects ``outputs.logits``.
            from mlx_vlm.models.base import LanguageModelOutput

            return LanguageModelOutput(logits=out)
        return out


def _mtp_target_classes(model_type: str):
    """Capability-resolver row ``(arch, speculative) -> (LanguageModel, build)``.

    ``build(config_dict) -> language_model`` encapsulates the per-arch
    constructor signature: qwen3.5/3.6 take ``(TextConfig, ModelConfig)`` (the
    mrope ``get_rope_index`` reads ``vision_config`` even for text, so we pass a
    default one - the text forward never touches the tower); gemma4 takes a bare
    ``TextConfig``. The stock mlx-lm classes gmlx builds for the plain text
    capability ship none of the ``speculative_*`` hooks, which is why MTP
    escalates to mlx-vlm here. Extend with new rows (deepseek_v4, ...) as drafters
    land.
    """
    import importlib

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        sub = model_type
        lang = importlib.import_module(f"mlx_vlm.models.{sub}.language")
        cfg = importlib.import_module(f"mlx_vlm.models.{sub}.config")

        def build(config):
            text_config = cfg.TextConfig.from_dict(config)
            model_config = cfg.ModelConfig.from_dict(
                {
                    "model_type": sub,
                    "text_config": dict(config),
                    "vision_config": {},
                    "vocab_size": config.get("vocab_size"),
                }
            )
            return lang.LanguageModel(text_config, model_config)

        return lang.LanguageModel, build
    if model_type == "gemma4_text":
        lang = importlib.import_module("mlx_vlm.models.gemma4.language")
        cfg = importlib.import_module("mlx_vlm.models.gemma4.config")

        def build(config):
            return lang.LanguageModel(cfg.TextConfig.from_dict(config))

        return lang.LanguageModel, build
    if model_type == "deepseek_v4":
        # Vendored mlx-lm-class target (no mlx-vlm counterpart): the SpecLM
        # subclass carries the speculative_* hooks + rotating-undo arming.
        from . import deepseek_v4_mtp
        from .deepseek_v4_model import ModelArgs, ensure_registered

        ensure_registered()

        def build(config):
            return deepseek_v4_mtp.DeepseekV4SpecLM(ModelArgs.from_dict(config))

        return deepseek_v4_mtp.DeepseekV4SpecLM, build
    if model_type == "hy_v3":
        from . import hy_v3_mtp, hy_v3_tools
        from .hy_v3_model import ModelArgs, ensure_registered

        ensure_registered()
        hy_v3_tools.ensure_registered()

        def build(config):
            return hy_v3_mtp.HyV3SpecLM(ModelArgs.from_dict(config))

        return hy_v3_mtp.HyV3SpecLM, build
    raise NotImplementedError(
        f"MTP target class for model_type {model_type!r} not wired "
        f"(supported: qwen3_5 / qwen3_5_moe / gemma4_text / deepseek_v4 / "
        f"hy_v3)"
    )


def _build_mtp_target(config_dict: dict):
    """Build the MTP target as an mlx-vlm text ``LanguageModel`` (seam 1)."""
    config = dict(config_dict)
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    model_type = config.get("model_type", "")
    LanguageModel, build = _mtp_target_classes(model_type)
    hooks = _MTP_TARGET_HOOKS_BY_TYPE.get(model_type, _MTP_TARGET_HOOKS)
    missing = [h for h in hooks if not hasattr(LanguageModel, h)]
    if missing:
        raise RuntimeError(
            f"mlx-vlm {model_type} LanguageModel missing MTP hooks {missing} "
            f"- version drift; pin mlx-vlm or update the hook set"
        )
    language_model = build(config)
    # Batched greedy verify walk: under greedy the engine takes the per-position
    # deferred walk (one CPU<->GPU sync per draft position) unless the target
    # exposes speculative_argmax_from_hidden, which lets it argmax all block+1
    # verify positions in a single op (zero per-position syncs -> _speculative_walk).
    # gemma4's LanguageModel ships only speculative_logits_from_hidden, so
    # synthesize the argmax wrapper from it - lossless (same tokens), just fewer
    # syncs (~+8% decode on a small target whose round is sync-bound).
    if not hasattr(language_model, "speculative_argmax_from_hidden") and hasattr(
        language_model, "speculative_logits_from_hidden"
    ):
        _lm = language_model
        _lm.speculative_argmax_from_hidden = lambda hidden: mx.argmax(
            _lm.speculative_logits_from_hidden(hidden), axis=-1
        )
    wrapper = MTPTextTarget(language_model, config)
    loadlog.verbose_print(f"[build] {model_type} -> mlx-vlm LanguageModel (MTP target wrapper)")
    return wrapper, config


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
        from . import minimax_m3_model

        minimax_m3_model.ensure_registered()
    if mt == "deepseek_v4":
        # mlx-lm ships no deepseek_v4 module yet (PR #1192 unmerged); same
        # vendored-registration pattern as minimax_m3, plus PoolingCache
        # injection into mlx_lm.models.cache.
        from . import deepseek_v4_model

        deepseek_v4_model.ensure_registered()
    if mt == "hy_v3":
        # mlx-lm ships no hy_v3 module yet (PR #1485 unmerged); same vendored-
        # registration pattern as minimax_m3. The tool parser registers with
        # the model so a later serve template-inference resolves it.
        from . import hy_v3_model, hy_v3_tools

        hy_v3_model.ensure_registered()
        hy_v3_tools.ensure_registered()
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
                from .moe_experts import _apply_expert_controls

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


# MoE expert CPU offload (hybrid GPU+CPU inference)
#
# On unified memory the GPU constraint is the wired limit, not a separate
# VRAM pool: Metal-resident buffers must be wired, while CPU-consumed mmap
# pages ride the page cache (evictable, can exceed RAM). For fine-grained MoE
# the routed expert stacks are ~90-95% of the bytes but each expert is read
# with probability top_k/n_experts per token, while the every-token layers
# (attention, norms, routers, shared experts, embeddings, KV cache) are read
# every token.
# Running the SwitchGLU expert containers on the CPU stream therefore keeps
# those hot layers + KV on GPU while the expert wire bytes stay file-backed in
# the page cache when the GPU is idle, and the kquant gather op executes on
# its threaded CPU path. MLX's cross-stream dependency tracking handles the
# GPU->CPU->GPU handoff inside each MoE layer (zero-copy - same pages).
#
# Residency (measured): Metal wires only what GPU work references or what
# sits in MLX's residency set - unreferenced file-backed buffers stay
# evictable page cache even under full memory pressure. The one hazard is
# mlx-lm's generation-time wired-limit bump (see
# _neutralize_wired_limit_sweep): MLX services a raised wired limit by
# sweeping every live buffer into the residency set, offloaded experts
# included. Models larger than the wired budget therefore run in streaming
# mode: the sweep is neutralized and GPU prefill routing is forced off, so
# the GPU never references (and never wires) expert bytes, and the page
# cache streams them from disk.
#
# Prefill staging: at decode each expert sees ~top_k/n_experts of one token,
# but a prefill chunk makes every expert hot with tens of rows each - a GEMM
# workload where the CPU (~1.5 TFLOP/s) is the wrong device. Calls with at
# least GMLX_STREAM_GPU_TOKENS tokens therefore run on the default (GPU)
# stream against the same zero-copy buffers - no copies, no staging; the
# driver wires the touched expert bytes for the duration of the work and
# releases them when the GPU goes idle. Threshold 0 disables GPU routing
# (pure CPU experts, the conservative choice when the model is far larger
# than RAM and prefill-wiring every expert is undesirable).
#
# Cost model (measured): offloaded decode pays a per-layer surcharge of
# genuine CPU dot compute plus per-layer stream fences and CPU-pool
# wake-from-idle (3 wakes per layer, one per gather; the wake cost grows
# when the pool sits idle between layers while the GPU runs the
# every-token layers).
# Routing every call to the GPU stream instead (GMLX_STREAM_GPU_TOKENS=1,
# no CPU hop) runs ~3.7x faster on a fits-in-RAM MoE, so in-RAM the CPU
# offload is for the over-budget regime, not the fast path.

_CPU_OFFLOAD_CLASS_CACHE: dict = {}

# Tokens-per-call at or above which an offloaded expert forward runs on the
# GPU stream (prefill regime). Decode calls (1-few tokens) stay on CPU.
_STREAM_GPU_TOKENS_DEFAULT = 32
# Streaming-mode expert calls at or above this many tokens are treated as
# prefill by the sequential-prefetch hook (see gmlx.prefetch).
_STREAM_PREFETCH_MIN_TOKENS = 32


def _stream_gpu_tokens(default: int = _STREAM_GPU_TOKENS_DEFAULT) -> int:
    return env_int("GMLX_STREAM_GPU_TOKENS", default)


def _arena_stage_max_tokens() -> int:
    """Largest expert call served router-aware (decode-feeder arena, or
    partial ring staging). Above this, a chunk routes to nearly every expert
    and whole-layer staging wins back its pipelining; below it, reading only
    the routed slices is the smaller IO."""
    return env_int("GMLX_ARENA_STAGE_MAX_TOKENS", 64)


_DECODE_ARENA_RAM_FRAC_DEFAULT = 0.6


def _available_ram_bytes(include_inactive: bool = True) -> int | None:
    """RAM this process can take: free + inactive + purgeable + speculative
    pages (macOS ``vm_stat``). A load-time snapshot of the machine's offer -
    a machine already half-occupied by other workloads offers the arena
    half a machine, whatever the hardware total says.

    ``include_inactive=False`` is the stricter no-victims set: inactive
    pages include other processes' anonymous memory, reclaimable only by
    swapping it. Counting them is the right optimistic call at load time
    (inactive is usually stale page cache from earlier runs), and the wrong
    call when the decode feeder regrows its wired arena into a *running*
    system - measured, that pushed double-digit GB of anon to swap."""
    import subprocess

    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return None
    m = re.search(r"page size of (\d+)", out)
    if not m:
        return None
    keys = ["free", "purgeable", "speculative"]
    if include_inactive:
        keys.append("inactive")
    pages = 0
    found = False
    for key in keys:
        mm = re.search(rf"Pages {key}:\s+(\d+)\.", out)
        if mm:
            pages += int(mm.group(1))
            found = True
    return pages * int(m.group(1)) if found else None


def _ram_floor_bytes(ram: int | None) -> int:
    """Breathing margin left to the system whenever the arena takes
    reclaimable RAM - at sizing time and again on every pressure-driven
    regrow (``GMLX_DECODE_RAM_FLOOR_GB`` overrides). On top of the base
    margin, ``GMLX_DECODE_PAGECACHE_GB`` reserves room for the page cache
    specifically: the prefill feeder and the CPU-mmap fallback read
    through it, and starving it collapses buffered pread throughput far
    below the SSD's sequential rate."""
    gb = float(
        os.environ.get("GMLX_DECODE_RAM_FLOOR_GB", "")
        or max(4.0, 0.05 * (ram or 0) / (1 << 30))
    )
    gb += env_float("GMLX_DECODE_PAGECACHE_GB", 2.5)
    return int(gb * (1 << 30))


def _decode_arena_bytes(total_bytes: int, offsets, budget: int | None) -> int:
    """Arena budget for the decode feeder, under two hardware-derived
    ceilings: the GPU working-set budget, and a fraction of physical RAM
    (``GMLX_DECODE_ARENA_RAM_FRAC``). The second exists because the arena is
    host anonymous memory - under pressure the kernel pages its cold slots
    out to swap rather than shrink other demand, and an arena hit that
    faults from swap is slower than reading the GGUF; the fraction leaves
    room for the non-expert weights' mmap, KV, the page cache and the rest
    of the system on any RAM size. Both ceilings then pay for the
    non-expert weights (everything that is not a routed-expert stack, which
    GPU work wires on its own) and a KV/runtime reserve, and the result is
    capped at the
    expert bytes themselves - a model whose experts fit goes fully
    resident. ``GMLX_DECODE_ARENA_GB`` overrides the ceilings but is still
    clamped to what is reclaimable minus the floor - an arena wired past
    that starves the page cache every buffered read path depends on
    (``GMLX_DECODE_ARENA_FORCE=1`` restores the unclamped behavior)."""
    env = os.environ.get("GMLX_DECODE_ARENA_GB")
    if env:
        want = int(float(env) * (1 << 30))
        if env_bool("GMLX_DECODE_ARENA_FORCE", False):
            return want
        avail = _available_ram_bytes()
        if avail is None:
            return want
        try:
            ram = int(mx.device_info()["memory_size"])
        except Exception:
            ram = avail
        cap = max(0, avail - _ram_floor_bytes(ram))
        if want > cap:
            print(
                f"[stream] GMLX_DECODE_ARENA_GB={env} exceeds reclaimable"
                f" RAM minus the floor; clamping the arena to"
                f" {cap / (1 << 30):.1f}GB (GMLX_DECODE_ARENA_FORCE=1"
                f" overrides)"
            )
            return cap
        return want
    if budget is None:
        return 0
    ceiling = budget
    ram = None
    try:
        ram = int(mx.device_info()["memory_size"])
        frac = float(
            os.environ.get("GMLX_DECODE_ARENA_RAM_FRAC", "")
            or _DECODE_ARENA_RAM_FRAC_DEFAULT
        )
        ceiling = min(ceiling, int(frac * ram))
    except Exception:
        pass
    # Third ceiling: what is reclaimable right now. The fraction assumes an
    # otherwise idle machine; co-resident workloads shrink the offer, and a
    # wired arena sized past it would evict them to swap. The floor keeps a
    # breathing margin for the system.
    avail = _available_ram_bytes()
    if avail is not None:
        ceiling = min(ceiling, avail - _ram_floor_bytes(ram or avail))
    expert_bytes = sum(r[2] for ranges in offsets.values() for r in ranges)
    non_expert_bytes = max(0, total_bytes - expert_bytes)
    reserve = int(
        float(os.environ.get("GMLX_DECODE_KV_RESERVE_GB", "8") or 8) * (1 << 30)
    )
    return min(max(0, ceiling - non_expert_bytes - reserve), expert_bytes)


def _neutralize_wired_limit_sweep():
    """Pin the MLX wired limit at its default for the rest of the process.

    mlx-lm wraps generation in a context manager that raises the wired limit
    to the device's max recommended working set. MLX services that by adding
    every live buffer - file-backed zero-copy weight views included - to its
    Metal residency set, which wires them all. For a model larger than the
    wired budget that sweep exhausts wired memory within seconds of the
    first GPU command (hard-panic territory). There is no per-buffer
    opt-out, so streaming mode no-ops ``mx.set_wired_limit`` instead: Metal
    then wires only what GPU work actually references (the every-token
    layers + KV), and
    expert pages stay plain evictable page cache. Covers every caller
    (generate, server batch path, trainer) since all resolve the function
    through ``mx.`` at call time. Idempotent.

    mlx-lm's ``wired_limit()`` context manager also prints a per-generation
    large-model warning sized against the limit this function just pinned -
    meaningless in streaming mode, and noisy (once per chat turn). Swap it
    for a quiet context that keeps the exit synchronize (the original syncs
    the generation stream before restoring the limit; callers may rely on
    that barrier at generator teardown). NB: patched via importlib -
    ``import mlx_lm.generate`` binds the function mlx_lm re-exports in
    ``__init__``, not the submodule.
    """
    if getattr(mx.set_wired_limit, "_kq_no_sweep", False):
        return

    def _no_sweep(*_a, **_k):
        return 0

    _no_sweep._kq_no_sweep = True
    mx.set_wired_limit = _no_sweep

    import contextlib
    import importlib

    @contextlib.contextmanager
    def _quiet_wired_limit(model, streams=None):
        try:
            yield
        finally:
            if streams is not None:
                for s in streams:
                    mx.synchronize(s)
            else:
                mx.synchronize()

    _quiet_wired_limit._kq_no_sweep = True
    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mod.wired_limit = _quiet_wired_limit


def configure_cpu_device():
    """Run everything on the CPU device (``--stream-cpu``): mmap-streamed weights.

    Besides setting the default device this (a) keeps the graph
    single-device - the fused-GDN runtime patch dispatches Metal kernels
    regardless of the default device - and (b) no-ops mlx-lm's
    ``wired_limit`` context: it reads
    ``device_info()["max_recommended_working_set_size"]``, absent on the
    CPU device, and wiring is meaningless on CPU. NB: patched via importlib
    - ``import mlx_lm.generate`` binds the function mlx_lm re-exports in
    ``__init__``, not the submodule.
    """
    import contextlib
    import importlib

    mx.set_default_device(mx.cpu)
    os.environ.setdefault("GMLX_FUSED_GDN", "0")

    @contextlib.contextmanager
    def _wired_noop(model, streams=None):
        yield

    for mod_name in ("mlx_lm.generate", "mlx_lm.utils"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "wired_limit"):
            mod.wired_limit = _wired_noop
    print("[device] cpu (mmap-streamed weights; fused-GDN Metal patch off)")


def _resolve_feeder_defaults(
    feeder_prefill: bool | None, feeder_decode: bool | None
) -> tuple[bool, bool]:
    """Feeder policy for streaming models. Explicit caller intent (a CLI
    flag) wins; then the ``GMLX_FEEDER_*`` env vars (A/B levers); then the
    defaults: prefill feeder on everywhere it can exist, decode feeder on
    only when the every-token layers are on the GPU (``--stream-experts``) -
    under ``--stream-cpu`` there is no GPU work for the arena gathers to
    join, so it is not even attempted."""
    gpu_resident = "gpu" in str(mx.default_device()).lower()
    if feeder_prefill is None:
        feeder_prefill = env_bool("GMLX_FEEDER_PREFILL", True)
    if feeder_decode is None:
        feeder_decode = env_bool("GMLX_FEEDER_DECODE", gpu_resident)
    return feeder_prefill, feeder_decode


def configure_stream_cpu(
    model,
    gguf_path: str | None = None,
    feeder_prefill: bool | None = None,
    feeder_decode: bool | None = None,
):
    """Whole-model CPU streaming (``--stream-cpu``): run the model on the CPU
    device with the streaming-expert machinery always engaged.

    ``--stream-cpu`` is an explicit opt-in into the CPU/over-RAM path, so it
    forces streaming (``force_stream=True``) regardless of model size - experts
    run on the CPU stream whether or not the model fits the wired budget (a
    fits-in-RAM model is then served from the page cache rather than faulting
    from disk; for the faster all-GPU path on a model that fits, omit
    ``--stream-cpu``). The GPU
    working-set budget is still captured before switching the default device to
    CPU so the over-/under-budget log line stays accurate (the CPU device would
    otherwise report a budget that hides the condition). Returns
    ``(n_wrapped, offloaded_bytes)``.
    """
    try:
        gpu_info = dict(mx.device_info())
    except Exception:
        gpu_info = None
    configure_cpu_device()
    if gpu_info and "max_recommended_working_set_size" in gpu_info:
        mx.device_info = lambda: gpu_info
    return install_expert_streaming(
        model,
        gguf_path=gguf_path,
        force_stream=True,
        feeder_prefill=feeder_prefill,
        feeder_decode=feeder_decode,
    )


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
        from .lookahead import _LA_PHASE as lap
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


def install_expert_streaming(
    model,
    n_layers: int | None = None,
    gguf_path: str | None = None,
    force_stream: bool = False,
    feeder_prefill: bool | None = None,
    feeder_decode: bool | None = None,
    stats_verbose: bool | None = None,
):
    """Run routed-expert stacks (SwitchGLU) on the CPU stream.

    Wraps each ``SwitchGLU`` in the first ``n_layers`` decoder layers (all
    layers when None) so its forward - the expert gather matmuls - executes
    under ``mx.stream(mx.cpu)`` at decode shapes, and on the default (GPU)
    stream for prefill-sized calls (see the staging note above). Per-instance
    ``__class__`` swap; routers, shared experts, attention, and the KV cache
    stay on the default (GPU) stream. Returns ``(n_wrapped, offloaded_bytes)``.

    ``gguf_path`` (the loaded checkpoint) enables sequential expert prefetch
    for streaming-mode models - see ``gmlx.prefetch``. Without it,
    over-budget prefill demand-faults expert bytes at random-read bandwidth.
    """
    from .modules import switch_layer_types

    _, glu_types = switch_layer_types()

    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers

    # Streaming mode: neutralize the generation-time residency sweep (which
    # would otherwise wire the whole model - see _neutralize_wired_limit_sweep)
    # and run every expert call on the CPU stream. Engaged when the model is
    # over the wired budget (it must stream) or when force_stream is set:
    # --stream-cpu (configure_stream_cpu) passes force_stream so the flag does
    # what it says - experts on CPU regardless of model size; on a fits-in-RAM
    # model the page cache then serves those bytes from RAM rather than faulting
    # from disk. --stream-experts keeps the budget-keyed decision, so below the
    # budget it still routes prefill-sized calls to the GPU stream
    # (GMLX_STREAM_GPU_TOKENS) - the fast path in-RAM.
    params = getattr(model, "parameters", None)
    total_bytes = sum(a.nbytes for _, a in tree_flatten(params())) if params else 0
    try:
        budget = int(0.9 * mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        budget = None
    over_budget = budget is not None and total_bytes > budget
    streaming = force_stream or over_budget
    prefetcher = None
    if streaming:
        _neutralize_wired_limit_sweep()
        from .prefetch import maybe_make_prefetcher

        prefetcher = maybe_make_prefetcher(gguf_path)
        if prefetcher is not None:
            object.__setattr__(model, "_kq_prefetcher", prefetcher)

    def _wrapped_class(cls):
        sub = _CPU_OFFLOAD_CLASS_CACHE.get(cls)
        if sub is None:
            # A fused base consumes routing scores itself (mix seam); a
            # stock base (unrecognized activation, e.g. minimax-m3's
            # SwiGLUOAI) takes (x, indices) only. The wrapper still
            # advertises _kq_scores_sink so blocks hand scores over for
            # miss-shed; it strips them before forwarding and applies
            # the shed mix python-side.
            _fwd_scores = bool(getattr(cls, "_kq_mix_scores", False))

            class _CPUOffload(cls):
                _kq_scores_sink = True

                def __call__(self, x, indices, *args, **kwargs):
                    # Extra args pass through untouched (e.g. deepseek-v4
                    # hands the fused SwitchGLU its routing scores). A base
                    # without the mix seam takes (x, indices) only: keep the
                    # scores for the miss-shed hook and strip them from what
                    # gets forwarded.
                    scores_arg = args[0] if args else None
                    if args and not _fwd_scores:
                        args = args[1:]
                    # Threshold read per call (cheap; once per MoE layer per
                    # forward) so env changes A/B without a reload. Streaming
                    # mode pins everything to CPU: a GPU expert call would
                    # wire the buffers it references, which an over-budget
                    # model cannot afford.
                    cpu_only = getattr(self, "_kq_cpu_only", False)
                    gpu_tokens = _stream_gpu_tokens(
                        getattr(
                            self, "_kq_gpu_tokens_default", _STREAM_GPU_TOKENS_DEFAULT
                        )
                    )
                    n_tokens = indices.size // indices.shape[-1]
                    pf = getattr(self, "_kq_prefetcher", None)
                    fdr = getattr(self, "_kq_feeder", None)
                    dfr = getattr(self, "_kq_decode_feeder", None)
                    small = n_tokens <= _arena_stage_max_tokens()
                    la = getattr(self, "_kq_lookahead", None)
                    la_pred = None
                    ph = _PHASE
                    if ph is not None:
                        _phase_token(
                            ph, getattr(self, "_kq_li", None), n_tokens)
                        if n_tokens != 1:
                            ph = None
                    lsp = getattr(self, "_kq_layer_shed", None)
                    if lsp is not None and cpu_only and n_tokens == 1:
                        rng = getattr(self, "_kq_shed_rng", None)
                        if rng is None:
                            # per-layer seed: reproducible shed pattern
                            rng = random.Random(
                                0x5EED ^ (getattr(self, "_kq_li", 0) or 0))
                            object.__setattr__(self, "_kq_shed_rng", rng)
                        if rng.random() < lsp:
                            # Skip the routed path entirely (gather, stage
                            # and this layer's eval fence). The unmixed
                            # zeros return makes the block mix nothing and
                            # still add its shared expert.
                            if dfr is not None:
                                dfr._layer_shed_n += 1
                            return mx.zeros(
                                (*x.shape[:-1], indices.shape[-1],
                                 x.shape[-1]), dtype=x.dtype)
                    gt = getattr(self, "_kq_gpu_token", None)
                    if (
                        gt is not None
                        and gt._route_shed is not None
                        and cpu_only
                        and n_tokens == 1
                        and scores_arg is not None
                        and dfr is not None
                        and dfr.covers(self._kq_li)
                        and not dfr.wedged_at(self._kq_li)
                    ):
                        # GPU-autonomous token (gpu-dispatch Tier 2): no
                        # per-layer eval. route_shed remaps ids to arena
                        # slots and sheds non-resident experts on the GPU;
                        # the token flushes once at the logits, and the
                        # host consumes recorded misses at the boundary
                        # (popularity + prestage + fresh slot tables) - see
                        # gpu_token.py for the fence argument.
                        gt.on_layer_entry(
                            self._kq_li,
                            getattr(self, "_kq_miss_shed", None))
                        tbl = gt.table(self._kq_li)
                        self._kq_cpu_only = False
                        try:
                            with dfr.swapped(self._kq_li):
                                with mx.stream(mx.gpu):
                                    sc_f32 = scores_arg.astype(mx.float32)
                                    slots, mix, m_ids, m_sc = (
                                        gt._route_shed(
                                            indices.astype(mx.uint32),
                                            sc_f32, tbl))
                                    mix_c = mix.astype(x.dtype)
                                    if _fwd_scores:
                                        y = super().__call__(
                                            x, slots, mix_c,
                                            *args[1:], **kwargs)
                                    else:
                                        y = super().__call__(
                                            x, slots, *args, **kwargs)
                                        if y.ndim == x.ndim + 1:
                                            y = (y * mix_c[..., None]).sum(
                                                axis=-2)
                                    gt.record(
                                        self._kq_li, indices, sc_f32,
                                        m_ids, m_sc, y)
                            return y
                        finally:
                            self._kq_cpu_only = True
                    if la is not None and cpu_only and small:
                        # Lookahead: run the NEXT MoE layer's router on this
                        # layer's input and evaluate it together with the
                        # router read below (one sync either way). The
                        # prediction feeds nothing downstream - it only
                        # records recall (probe) or drives prestage reads.
                        if ph is not None:
                            t_la = time.perf_counter()
                            la_pred = la.on_call(x, indices)
                            ph["la"] += time.perf_counter() - t_la
                        else:
                            la_pred = la.on_call(x, indices)
                    if (
                        dfr is not None
                        and cpu_only
                        and small
                        and dfr.covers(self._kq_li)
                    ):
                        # Decode feeder: the routed experts are served from
                        # this layer's wired GPU arena; misses are pread from
                        # the GGUF into evicted slots first. Small prefill
                        # chunks take this path too when their routed set
                        # fits - the arena persists across requests, which is
                        # what makes repeat short-prompt TTFT cheap. The eval
                        # is both the router read and the arena-overwrite
                        # safety fence (see decode_feeder.py). ``stage``
                        # returns None when the call routes to more distinct
                        # experts than the arena has slots - fall through.
                        t0 = time.perf_counter() if ph is not None else 0.0
                        ms = getattr(self, "_kq_miss_shed", None)
                        sc_f32 = None
                        if (ms is not None and scores_arg is not None
                                and n_tokens == 1):
                            # Shed reads the scores host-side; fold them into
                            # the router eval so the hook adds a small D2H
                            # copy, not a second per-layer graph flush.
                            sc_f32 = scores_arg.astype(mx.float32)
                            mx.eval(indices, sc_f32)
                        else:
                            mx.eval(indices)
                        if ph is not None:
                            t1 = time.perf_counter()
                            ph["ev"] += t1 - t0
                            wait0 = getattr(dfr, "_t_demand", 0.0)
                        ids = np.array(indices)
                        shed_args = None
                        shed_mix = None
                        if sc_f32 is not None:
                            sc = np.asarray(sc_f32).reshape(-1)
                            keep = dfr.shed_misses(
                                self._kq_li, ids.reshape(-1), sc, ms)
                            if keep is not None:
                                # Arena-path only: the overflow fallback
                                # below keeps the original routed set.
                                kept = ids.reshape(-1)[keep]
                                shp = ids.shape[:-1] + (kept.size,)
                                ids = np.ascontiguousarray(kept.reshape(shp))
                                scn = sc[keep]
                                # survivors keep the token's full mass
                                scn = scn * (sc.sum() / max(scn.sum(), 1e-20))
                                sc_mx = mx.array(scn.reshape(shp)).astype(
                                    scores_arg.dtype)
                                if _fwd_scores:
                                    shed_args = (sc_mx,) + args[1:]
                                else:
                                    # Stock base returns per-expert outputs;
                                    # the block's weights still cover the
                                    # full routed set, so mix the shed
                                    # survivors here instead.
                                    shed_mix = sc_mx
                        slots = dfr.stage(self._kq_li, ids)
                        if ph is not None:
                            t2 = time.perf_counter()
                            w = getattr(dfr, "_t_demand", 0.0) - wait0
                            ph["stage_wait"] += w
                            ph["stage_book"] += (t2 - t1) - w
                        if la_pred is not None:
                            # This layer's demand misses have joined
                            # (stage returned); the next layer's predicted
                            # misses now read in the background while this
                            # layer's gather and the next layer's every-token
                            # work compute - speculation never competes with
                            # demand traffic for the SSD.
                            dfr.prestage(la.predictor.dst_li, la_pred)
                            if ph is not None:
                                ph["prestage"] += time.perf_counter() - t2
                        if slots is not None:
                            # arena call: weights are wired GPU views for
                            # this scope, so lift the streaming CPU pin
                            # and let the fused kq kernels run
                            if shed_args is not None:
                                args = shed_args
                            self._kq_cpu_only = False
                            try:
                                t3 = (time.perf_counter()
                                      if ph is not None else 0.0)
                                with dfr.swapped(self._kq_li):
                                    with mx.stream(mx.gpu):
                                        y = super().__call__(
                                            x, mx.array(slots),
                                            *args, **kwargs)
                                        if (shed_mix is not None
                                                and y.ndim == x.ndim + 1):
                                            y = (y * shed_mix[..., None]).sum(
                                                axis=-2)
                                if ph is not None:
                                    ph["build"] += time.perf_counter() - t3
                                return y
                            finally:
                                self._kq_cpu_only = True
                    wedged = dfr is not None and dfr.wedged_at(self._kq_li)
                    if wedged and dfr.has_dead(self._kq_li):
                        # A wedged read poisoned part of this layer's file
                        # range: no fallback below (mmap gather, advisory
                        # prefetch, prefill staging) may touch a dead
                        # expert's bytes - rewrite the routing ids first.
                        mx.eval(indices)
                        indices = mx.array(dfr.redirect_dead(
                            self._kq_li, np.array(indices)))
                    if (
                        fdr is not None
                        and not wedged
                        and small
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                        and fdr.covers(self._kq_li)
                    ):
                        # Router-aware partial staging: a short chunk routes
                        # to a fraction of the experts, so stage only those
                        # slices into the ring slot instead of the whole
                        # layer (see feeder.prefill_partial_call).
                        mx.eval(indices)
                        ids = np.unique(np.array(indices)).tolist()
                        with fdr.prefill_partial_call(self, self._kq_li, ids):
                            with mx.stream(mx.gpu):
                                return super().__call__(
                                    x, indices, *args, **kwargs)
                    if (
                        fdr is not None
                        and not wedged
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                        and fdr.covers(self._kq_li)
                    ):
                        # Feeder prefill: this layer's expert stacks are
                        # staged straight from the GGUF into GPU-visible
                        # ring slots and the GEMM runs on the GPU stream
                        # from the slot - the page cache never sees the
                        # bytes. The eval is the ring protocol's slot-free
                        # proof (previous layer's compute has finished);
                        # see feeder.py. Wedged layers skip this (and the
                        # whole-layer advisory below): both sweep the full
                        # expert range, poisoned bytes included.
                        mx.eval(x)
                        with fdr.prefill_call(self, self._kq_li):
                            with mx.stream(mx.gpu):
                                return super().__call__(
                                    x, indices, *args, **kwargs)
                    if (
                        pf is not None
                        and not wedged
                        and pf.enabled
                        and n_tokens >= _STREAM_PREFETCH_MIN_TOKENS
                    ):
                        # Streaming prefill: materialize the lazy graph up to
                        # this layer so the advisory window advances at
                        # execution pace. Build-time would fire every layer's
                        # advisory at once, and an over-RAM advisory storm
                        # evicts its own earlier reads.
                        mx.eval(x)
                        pf.on_layer(self._kq_li)
                    elif (
                        pf is not None
                        and pf.enabled
                        and cpu_only
                        and env_bool("GMLX_DECODE_PREFETCH", True)
                    ):
                        # Streaming decode: the router's top-k is tiny and
                        # the gather needs it anyway - evaluate it now and
                        # pull the selected experts' slices into the page
                        # cache at queue depth (on_decode) instead of
                        # demand-faulting 16 KB clusters from inside the
                        # gemv. GMLX_DECODE_PREFETCH=0 disables.
                        mx.eval(indices)
                        pf.on_decode(
                            self._kq_li,
                            np.unique(np.array(indices)).tolist(),
                        )
                    if gpu_tokens > 0 and n_tokens >= gpu_tokens and not cpu_only:
                        # Prefill regime: GEMM on the GPU stream, same
                        # zero-copy buffers.
                        return super().__call__(x, indices, *args, **kwargs)
                    with mx.stream(mx.cpu):
                        return super().__call__(x, indices, *args, **kwargs)

            _CPUOffload.__name__ = cls.__name__ + "_CPUOffload"
            _CPU_OFFLOAD_CLASS_CACHE[cls] = sub = _CPUOffload
        return sub

    n_wrapped = 0
    offloaded = 0
    n_cpu_only_codec = 0
    moe_modules: dict[int, list] = {}
    for li, layer in enumerate(layers):
        if n_layers is not None and li >= n_layers:
            break
        for m in layer.modules():
            if not isinstance(m, glu_types):
                continue
            if m.__class__ in _CPU_OFFLOAD_CLASS_CACHE.values():
                continue  # already wrapped (idempotent)
            gpu_ok = _kq_expert_gpu_ok(m)
            if not gpu_ok:
                n_cpu_only_codec += 1
            m.__class__ = _wrapped_class(m.__class__)
            if streaming:
                m._kq_cpu_only = True
                object.__setattr__(m, "_kq_li", li)
                if gpu_ok:
                    moe_modules.setdefault(li, []).append(m)
                if prefetcher is not None:
                    object.__setattr__(m, "_kq_prefetcher", prefetcher)
            elif gpu_ok:
                # All-GPU auto-policy: in-RAM, the residency sweep wires the
                # whole model regardless of where expert calls run, so the
                # CPU hop has no memory benefit and a large decode cost
                # (measured ~4-5x). Route every call to the GPU stream; an
                # explicit GMLX_STREAM_GPU_TOKENS (e.g. 0) overrides.
                object.__setattr__(m, "_kq_gpu_tokens_default", 1)
            else:
                m._kq_cpu_only = True
            offloaded += sum(a.nbytes for _, a in tree_flatten(m.parameters()))
            n_wrapped += 1
    if n_cpu_only_codec:
        print(
            f"[stream] {n_cpu_only_codec} expert stacks use a CPU-only codec "
            "(no Metal matmul kernels yet): feeder/arena staging and GPU "
            "prefill routing off - every expert call runs on the CPU stream"
        )
    # Non-expert weights + KV run on the default device: CPU for --stream-cpu
    # (configure_stream_cpu sets the default to CPU before this call), GPU for
    # --stream-experts.
    base_dev = "CPU" if "cpu" in str(mx.default_device()).lower() else "GPU"
    if streaming:
        head = (
            f"model {total_bytes / 1e9:.0f} GB > ~{budget / 1e9:.0f} GB "
            "wired budget"
            if over_budget
            else f"model {total_bytes / 1e9:.0f} GB, streaming forced"
        )
        print(
            f"[stream] streaming: {head} - {n_wrapped} MoE layers' experts "
            f"({offloaded / 1e9:.1f} GB) stay file-backed; rest of the model "
            f"+ KV on {base_dev}"
        )
    feeder_prefill, feeder_decode = _resolve_feeder_defaults(
        feeder_prefill, feeder_decode
    )
    feeder = None
    dfeeder = None
    if (
        streaming
        and prefetcher is not None
        and moe_modules
        and feeder_prefill
    ):
        from .feeder import maybe_make_prefill_feeder

        feeder = maybe_make_prefill_feeder(prefetcher.offsets, moe_modules)
        if feeder is not None:
            n_cov = sum(feeder.covers(li) for li in moe_modules)
            for li, mods in moe_modules.items():
                if feeder.covers(li):
                    for m in mods:
                        object.__setattr__(m, "_kq_feeder", feeder)
            object.__setattr__(model, "_kq_feeder", feeder)
            cov = (
                "" if n_cov == len(moe_modules)
                else f" on {n_cov}/{len(moe_modules)} layers"
            )
            print(
                "[stream] feeder prefill: expert stacks staged straight "
                f"from GGUF through 2 x {feeder.slot_bytes / 1e9:.1f} GB "
                f"GPU-visible ring slots{cov} (--no-prefill-feeder disables)"
            )
    if (
        streaming
        and prefetcher is not None
        and moe_modules
        and feeder_decode
    ):
        from .decode_feeder import maybe_make_decode_feeder

        arena = _decode_arena_bytes(total_bytes, prefetcher.offsets, budget)
        dfeeder = maybe_make_decode_feeder(
            prefetcher.offsets, moe_modules, arena, stats_verbose)
        if dfeeder is not None:
            n_cov = sum(dfeeder.covers(li) for li in moe_modules)
            for li, mods in moe_modules.items():
                if dfeeder.covers(li):
                    for m in mods:
                        object.__setattr__(m, "_kq_decode_feeder", dfeeder)
            object.__setattr__(model, "_kq_decode_feeder", dfeeder)
            wired = (
                "fully wired"
                if dfeeder.locked_bytes >= dfeeder.arena_bytes
                else f"{dfeeder.locked_bytes / 1e9:.1f} GB wired"
            )
            cov = (
                "" if n_cov == len(moe_modules)
                else f" on {n_cov}/{len(moe_modules)} layers"
            )
            print(
                f"[stream] decode feeder: {dfeeder.arena_bytes / 1e9:.1f} GB "
                f"popularity-managed expert arena ({wired}){cov} "
                "(--no-decode-feeder disables, GMLX_DECODE_ARENA_GB sizes)"
            )
    if streaming and dfeeder is not None:
        from . import gpu_token

        if gpu_token.autonomous_enabled():
            if gpu_token.route_shed_op() is None:
                print(
                    "[stream] gpu-autonomous: requested but the installed "
                    "mlx_kquant has no route_shed op; falling back to "
                    "per-layer staging"
                )
            else:
                gt = gpu_token.GpuTokenState(dfeeder)
                gpu_token.register_exit_stats(gt)
                for li, mods in moe_modules.items():
                    if dfeeder.covers(li):
                        for m in mods:
                            object.__setattr__(m, "_kq_gpu_token", gt)
                object.__setattr__(model, "_kq_gpu_token", gt)
                print(
                    "[stream] gpu-autonomous token: route_shed remaps + "
                    "sheds on GPU, one flush per token, misses prestage at "
                    "token boundaries (lossy at low hit rates; "
                    "GMLX_GPU_AUTONOMOUS=1 enables)"
                )
    if streaming and dfeeder is not None and env_bool(
            "GMLX_GPU_KEEPWARM", False):
        from . import keepwarm

        keepwarm.start()
        print(
            "[stream] gpu keep-warm: background heartbeat holds GPU "
            "clocks between per-layer decode bursts (lossless, costs "
            "power; --gpu-keepwarm / GMLX_GPU_KEEPWARM=1 enables)"
        )
    la_probe = env_bool("GMLX_DECODE_LOOKAHEAD_PROBE", False)
    # Lookahead's replica router folds into the per-layer sync; whether its
    # stall savings cover that tax is a per-family measurement. On
    # glm_moe_dsa (GLM-5.2, 75 layers, top-8) it measured net negative
    # (~40ms/tok sync for ~18ms of stalls), so those families default off.
    # An explicit GMLX_DECODE_LOOKAHEAD always wins.
    la_default = _lookahead_default(model)
    la_prefetch = (
        env_bool("GMLX_DECODE_LOOKAHEAD", la_default) and dfeeder is not None)
    if (streaming and dfeeder is not None and not la_default
            and "GMLX_DECODE_LOOKAHEAD" not in os.environ):
        print(
            "[stream] lookahead prestage: off by family default (replica-"
            "router sync tax measured above its stall savings; "
            "GMLX_DECODE_LOOKAHEAD=1 enables)"
        )
    if streaming and (la_probe or la_prefetch):
        from .lookahead import install_lookahead

        n_la = install_lookahead(
            model, layers, probe=la_probe, prefetch=la_prefetch,
            stats_verbose=stats_verbose,
        )
        if n_la and la_prefetch:
            print(
                f"[stream] lookahead prestage: next-layer router "
                f"predictions pre-read arena misses on {n_la} MoE layer "
                "pairs (lossless; GMLX_DECODE_LOOKAHEAD=0 disables)"
            )
        if n_la and la_probe:
            print(
                f"[stream] lookahead probe: recording next-layer router "
                f"recall on {n_la} MoE layer pairs (lossless; table at exit)"
            )
    if streaming:
        # The context line printed above and the feeder lines cover the
        # normal story; what remains is the fallback mechanics for
        # whatever the feeders don't handle.
        fallback = []
        if dfeeder is None:
            fallback.append(
                "decode streams expert bytes from disk through the page "
                "cache (disk-bound)"
                if over_budget
                else "decode reads experts through the page cache on the "
                "CPU stream"
            )
        if feeder is None:
            fallback.append(
                "prefill uses sequential page-cache prefetch"
                if prefetcher is not None
                else "prefill demand-faults (no gguf_path)"
            )
        if fallback:
            print(f"[stream] {'; '.join(fallback)}")
        if not over_budget:
            b = f"~{budget / 1e9:.0f} GB" if budget else "unknown"
            print(
                f"[stream] --stream-cpu streams experts even though the "
                f"{total_bytes / 1e9:.0f} GB model fits the wired budget "
                f"({b}) - omit --stream-cpu for the faster all-GPU path on "
                "a model that fits"
            )
    else:
        gpu_tokens = _stream_gpu_tokens(1)
        if gpu_tokens == 1:
            staging = (
                "model fits the wired budget - decode auto-routed to the "
                "GPU stream (GMLX_STREAM_GPU_TOKENS=0 forces CPU decode)"
            )
        elif gpu_tokens > 0:
            staging = f"prefill calls >={gpu_tokens} tokens routed to GPU"
        else:
            staging = "GPU prefill routing disabled"
        print(
            f"[stream] routed experts -> CPU stream on {n_wrapped} layers "
            f"({offloaded / 1e9:.1f} GB stays file-backed; rest of the model "
            f"+ KV on {base_dev}; {staging})"
        )
    return n_wrapped, offloaded


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


def moe_streaming_active(model) -> bool:
    """True when ``install_expert_streaming`` put this model in streaming mode
    (weights exceed the GPU wired budget; expert bytes stream from disk)."""
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "model", None), "layers", None) or ()
    return any(
        getattr(m, "_kq_cpu_only", False) for layer in layers for m in layer.modules()
    )


def _resolve_prefill_step(model, requested: int | None) -> tuple[int | None, bool]:
    """Pick the prefill chunk width: an explicit request always wins; a
    streaming-mode model defaults to ``_STREAMING_PREFILL_STEP``; everything
    else keeps mlx-lm's own default. Returns ``(step_or_none, defaulted)``."""
    if requested is not None or not moe_streaming_active(model):
        return requested, False
    return _STREAMING_PREFILL_STEP, True


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


def install_moe_experts_override(model, k: int) -> int:
    """Experiment, lossy: route every token to ``k`` experts instead of the
    trained top-k, on MoE blocks whose experts ``install_expert_streaming``
    wrapped (and only those - the knob exists to probe how router fan-out
    shapes offloaded prefill/decode traffic, not as a general sampler).

    The override rewrites the router's own top-k attribute (``top_k`` /
    ``num_experts_per_tok`` on the block, and on a DeepSeek-style gate
    submodule when present - named ``gate``, or ``router`` on hy_v3), so
    expert selection and the arch's weight renormalization run unchanged
    at the new k. Outputs differ from the trained model by design; parity
    gates will fail. Returns the number of MoE blocks overridden; raises
    on k < 1 or k > the expert count.
    """
    if k < 1:
        raise ValueError(f"MoE top-k override must be >= 1, got {k}")
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers
    overridden = 0
    trained_k = None
    for layer in layers:
        for owner in layer.modules():
            glu = None
            for child in owner.children().values():
                candidates = child if isinstance(child, (list, tuple)) else [child]
                for c in candidates:
                    if type(c).__name__.endswith("_CPUOffload"):
                        glu = c
                        break
                if glu is not None:
                    break
            if glu is None:
                continue
            n_experts = _switch_num_experts(glu)
            if n_experts and k > n_experts:
                raise ValueError(
                    f"MoE top-k override {k} exceeds the {n_experts}-expert "
                    "stack on an offloaded layer"
                )
            hit = False
            for target in (
                owner,
                getattr(owner, "gate", None),
                getattr(owner, "router", None),
            ):
                if target is None:
                    continue
                for attr in ("top_k", "num_experts_per_tok"):
                    current = getattr(target, attr, None)
                    if isinstance(current, int):
                        if trained_k is None:
                            trained_k = current
                        setattr(target, attr, k)
                        hit = True
            if hit:
                overridden += 1
    if overridden:
        print(
            f"[stream] MoE top-k override: {trained_k}->{k} experts/token "
            f"on {overridden} offloaded MoE layers (lossy - outputs differ "
            "from the trained router)"
        )
    else:
        print(
            "[stream] MoE top-k override found no offloaded MoE block "
            "with a router top-k attribute - no effect"
        )
    return overridden


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


def _warm_mmap_residency(
    model, *, log=print, paths: list[str] | None = None,
    batch_bytes: int = 4 << 30, threshold_bytes: int | None = None,
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
    arrays = [v for _, v in tree_flatten(model.parameters())]
    total = sum(a.nbytes for a in arrays)
    # Register before any early return: the MTP seed-cap headroom estimate
    # needs these bytes counted whether or not the touch pass runs.
    note_untracked_weights(total)
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
    # hy_v3 routing is semantically fp32 (F32 wire; llama.cpp routes in fp32,
    # and the vendored class's cast_predicate exempts expert_bias): sigmoid
    # gate + selection bias decide top-8 of 192, where bf16 rounding flips
    # near-tie selections.
    "hy_v3": (".mlp.router.gate.weight", ".mlp.router.expert_bias"),
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


def _install_and_load(
    model,
    hf_weights,
    hf_kquant_meta,
    *,
    log,
    sanitize: bool = True,
    no_alias: set[str] | None = None,
    fp32_keep: tuple[str, ...] = (),
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
    cast (see ``_FP32_KEEP_BY_MODEL_TYPE``).
    """
    loadlog.stage("loading weights")
    # 5. sanitize first - model.sanitize may rename keys; rebuild meta.
    if sanitize and hasattr(model, "sanitize"):
        hf_weights = model.sanitize(hf_weights)
        new_meta: dict[str, str] = {}
        unmatched_meta = set(hf_kquant_meta)
        for new_k in hf_weights:
            if not new_k.endswith(".weight"):
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

        n_fp = repack_native_fp_weights(hf_weights, hf_kquant_meta)
        if n_fp:
            log(
                f"[native-fp] de-interleaved {n_fp} mxfp4/nvfp4 tensors -> MLX packed layout"
            )

    # 6. swap leaves with kquant equivalents.
    n_replaced = install_kquant_modules(
        model, hf_kquant_meta, native_fp_wire=native_fp_wire)
    log(f"[install] replaced {n_replaced} leaves with kquant modules")

    if install_hd512_sdpa():
        log("[install] head_dim-512 fused SDPA active")
    if install_prefill_decay():
        log("[install] depth-decay prefill chunking active")
    if install_qwen35_verify_fold():
        log("[install] qwen3.5 folded verify attention active")
    n_fused_moe = install_fused_moe_glu(model)
    if n_fused_moe:
        log(f"[install] fused mxfp4 MoE GLU decode on {n_fused_moe} layers")
    n_shexp = install_hyv3_shexp_fold(model)
    if n_shexp:
        log(f"[install] hy3 shared-expert fold on {n_shexp} MoE layers")
    n_fused_qkv = install_fused_qkv(model)
    if n_fused_qkv:
        log(f"[install] fused QKV decode projection on {n_fused_qkv} layers")
    install_rotating_cache_fix()

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

    n_cast = 0
    for k in list(loadable):
        v = loadable[k]
        if v.dtype in (mx.float32, mx.float16) and k not in hf_kquant_meta:
            if fp32_keep and any(s in k for s in fp32_keep):
                if v.dtype != mx.float32:      # e.g. F16 ape tables
                    loadable[k] = v.astype(mx.float32)
                continue
            if v.dtype == mx.float16:
                # Same-itemsize f16->bf16 gets buffer-donated into the source
                # view -- a write through the zero-copy file mapping (dropped
                # on read-only maps, leaving f16 bits typed as bf16). The f32
                # hop makes both steps size-changing, so neither can donate.
                v = v.astype(mx.float32)
            loadable[k] = v.astype(mx.bfloat16)
            n_cast += 1
    if n_cast:
        log(f"[dtype] cast {n_cast} float params (norms etc.) to bf16")

    model.load_weights(list(loadable.items()), strict=False)
    log(f"[load_weights] loaded {len(loadable)} / {len(model_params)} model parameters")
    _warm_mmap_residency(model, log=log)

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
    ``nn.QuantizedEmbedding``. So dequantize the table to a plain bf16
    ``nn.Embedding`` (a layout the model handles natively), in row chunks to stay
    under the single-dispatch grid limit. The tied ``as_linear`` logits then run
    in bf16. Other quantized leaves are untouched.
    """
    emb = model.model.decoder.embed_tokens
    if not isinstance(emb, KQuantEmbedding):
        return
    n, dims, codec = emb.num_embeddings, emb.dims, emb.kquant_type
    packed, scales = emb["weight"], emb["scales"]
    chunk = 16384
    rows = [
        kq.dequantize(
            packed[i : i + chunk].reshape(-1, packed.shape[-1]), scales, codec
        )
        .reshape(min(chunk, n - i), dims)
        .astype(mx.bfloat16)
        for i in range(0, n, chunk)
    ]
    table = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
    mx.eval(table)
    new_emb = nn.Embedding(n, dims)
    new_emb.weight = table
    new_emb.freeze()
    model.model.decoder.embed_tokens = new_emb
    log(f"[diffusion] dequantized embed_tokens {codec}->bf16 ({n}x{dims})")


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

    # 0. preflight - discover shards, classify codecs (IQ / unsupported types
    #    refuse here, naming the codec, before kq.load_gguf's cryptic
    #    "unsupported type N"), and gate on the architecture. Reads only the
    #    GGUF header via gguf-py, so it stays cheap on multi-GB files.
    loadlog.stage("reading gguf metadata")
    pf = preflight(gguf_path, arch=arch, hf_source=hf_source)
    arch = pf.arch
    loadlog.fact("arch", arch)
    _log(f"[arch] {arch}")

    # Kick off the page-cache populate as early as the shard list exists so
    # the disk stream overlaps the whole CPU-bound remainder of load (the
    # phase-7 residency warm dedupes via the populate registry).
    maybe_populate_for_load(pf.shards, log=_log)

    # Larger-than-RAM shards leave a stale cache remnant that taxes the next
    # process's fault path; sweep it back to the free list at exit.
    from .pagecache import register_streaming_release
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
        # mapping. Fail loudly instead.
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
    #       (secondary) the L==1 decode applies the top-k by gathering keys rather
    #         than masking, corrupting the sampling tail. Kill with
    #         GMLX_DSV32_MASK_DECODE=0.
    if config.get("model_type") in ("glm_moe_dsa", "deepseek_v32"):
        _patch_dsv32_indexer_rope(model)
        _patch_dsv32_indexer_fp32(model)
        _patch_dsv32_moe_gate_fp32(model)
        _patch_dsv32_moe_scores(model)
        _patch_dsv32_mask_decode(model)
        _patch_dsv32_dense_default(model)  # exact default; GMLX_DSV32_SPARSE=1 -> sparse (experimental)

    # 5. sanitize first - model.sanitize may rename keys; rebuild meta.
    if hasattr(model, "sanitize"):
        hf_weights = model.sanitize(hf_weights)
        new_meta: dict[str, str] = {}
        unmatched_meta = set(hf_kquant_meta)
        for new_k in hf_weights:
            if not new_k.endswith(".weight"):
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

        n_fp = repack_native_fp_weights(hf_weights, hf_kquant_meta)
        if n_fp:
            _log(
                f"[native-fp] de-interleaved {n_fp} mxfp4/nvfp4 tensors -> MLX packed layout"
            )

    # 6. swap leaves with kquant equivalents.
    loadlog.stage("installing quantized weights")
    n_replaced = install_kquant_modules(
        model, hf_kquant_meta, native_fp_wire=native_fp_wire)
    _log(f"[install] replaced {n_replaced} leaves with kquant modules")

    if install_hd512_sdpa():
        _log("[install] head_dim-512 fused SDPA active")
    if install_prefill_decay():
        _log("[install] depth-decay prefill chunking active")
    if install_qwen35_verify_fold():
        _log("[install] qwen3.5 folded verify attention active")
    n_fused_moe = install_fused_moe_glu(model)
    if n_fused_moe:
        _log(f"[install] fused mxfp4 MoE GLU decode on {n_fused_moe} layers")
    n_shexp = install_hyv3_shexp_fold(model)
    if n_shexp:
        _log(f"[install] hy3 shared-expert fold on {n_shexp} MoE layers")
    n_fused_qkv = install_fused_qkv(model)
    if n_fused_qkv:
        _log(f"[install] fused QKV decode projection on {n_fused_qkv} layers")
    install_rotating_cache_fix()

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
    # projections) to bf16 so activations flow bf16, avoiding float32 kernel
    # promotion (and bf16xf16 dtype mismatches) in quantized/regular matmul.
    # Model-types in _FP32_KEEP_BY_MODEL_TYPE pin listed params to float32.
    fp32_keep = _FP32_KEEP_BY_MODEL_TYPE.get(config.get("model_type"), ())
    n_cast = 0
    for k in list(loadable):
        v = loadable[k]
        if v.dtype in (mx.float32, mx.float16) and k not in hf_kquant_meta:
            if fp32_keep and any(s in k for s in fp32_keep):
                if v.dtype != mx.float32:      # e.g. F16 ape tables
                    loadable[k] = v.astype(mx.float32)
                continue
            if v.dtype == mx.float16:
                # Same-itemsize f16->bf16 gets buffer-donated into the source
                # view -- a write through the zero-copy file mapping (dropped
                # on read-only maps, leaving f16 bits typed as bf16). The f32
                # hop makes both steps size-changing, so neither can donate.
                v = v.astype(mx.float32)
            loadable[k] = v.astype(mx.bfloat16)
            n_cast += 1
    if n_cast:
        _log(f"[dtype] cast {n_cast} float params (norms etc.) to bf16")

    model.load_weights(list(loadable.items()), strict=False)
    _log(
        f"[load_weights] loaded {len(loadable)} / {len(model_params)} model parameters"
    )
    _warm_mmap_residency(model, log=_log, paths=pf.shards)

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

    if config.get("model_type") == "deepseek_v4":
        from .deepseek_v4_model import install_gemv_row_fusion, warm_kernel_pipelines

        n_fused_gemv = install_gemv_row_fusion(model)
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

    wait_for_populate(pf.shards, log=_log)

    return model, config, tokenizer


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

    from .hf_cache import network_fetch_allowed
    with network_fetch_allowed():
        path = hf_hub_download(hf_source, "config.json")
    with open(path, "r") as f:
        return json.load(f)
