"""GGUF wire-byte loading and array remapping.

Reads a GGUF's raw kquant wire bytes (``load_gguf_wire_bytes``) and remaps
GGUF tensor names onto the mlx-lm/mlx-vlm parameter tree: the text path
(``remap_arrays``), MTP extras (``remap_mtp_arrays``), and the gemma4
assistant head (``remap_gemma4_assistant_arrays``).
"""
from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

import mlx_kquant as kq

from . import loadlog
from .gguf_meta import read_int
from .native_fp import _strip_weight
from .preflight import find_split_shards
from .remap import RemapDecision, parse_gguf_name
from .transforms import qk_permute_wire, retarget, split_fused_gate_up_kquant


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
        "kda_conv_weight": 0,
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

        elif transform == "kda_conv_weight":
            # Kimi-K3 KDA depthwise short conv: GGUF ships (1, d_inner, 1,
            # d_conv) numpy order, or (d_inner, 1, d_conv) when quantization
            # drops the trailing 1. conv_step varies fastest in both, so a
            # pure reshape to (d_inner, d_conv) is exact; mlx Conv1d wants
            # (out_channels=d_inner, kernel=d_conv, in/groups=1).
            hf_weights[hf_name] = arr.reshape(-1, arr.shape[-1])[..., None]
            stats["kda_conv_weight"] += 1
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


def strip_nextn_trunk_overflow(
    hf_weights: dict, hf_kquant_meta: dict, meta, arch: str
) -> int:
    """Drop remapped weights of trailing NextN/MTP block(s) from the trunk tree.

    nemotron_h_moe GGUFs carry the MTP layer as ``blk.{block_count - 1}`` with
    the same tensor names as trunk blocks, so the trunk remap emits
    ``backbone.layers.{N}.*`` entries for a layer index the trunk model does not
    have (llama.cpp likewise excludes nextn layers from the trunk graph; the
    stock nemotron_h ``sanitize`` only strips HF-named ``mtp.*`` keys). The MTP
    drafter loads that block separately. Returns the number of entries dropped.
    """
    if arch != "nemotron_h_moe":
        return 0
    nextn = read_int(meta, f"{arch}.nextn_predict_layers") or 0
    block_count = read_int(meta, f"{arch}.block_count") or 0
    if nextn <= 0 or block_count <= nextn:
        return 0
    trunk = block_count - nextn
    # backbone.*: the NEMOTRON_H_MOE override table; model.*: MTP-block
    # tensors the override table does not claim (post_attention_norm) fall
    # through to the canonical map's model.layers.{N}.* naming.
    pat = re.compile(r"^(?:backbone|model)\.layers\.(\d+)\.")
    dropped = 0
    for name in list(hf_weights):
        m = pat.match(name)
        if m and int(m.group(1)) >= trunk:
            del hf_weights[name]
            hf_kquant_meta.pop(name, None)
            dropped += 1
    return dropped


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
