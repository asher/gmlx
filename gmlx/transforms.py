"""Wire-byte layout transforms applied during the GGUF -> HF name remap.

Each transform is tagged on a ``RemapDecision`` (see ``remap.py``) and applied
by ``loader.remap_arrays``. They operate on raw kquant ``uint8`` wire bytes (or
native dtypes for non-quantized tensors), preserving per-block byte alignment.

The simpler transforms (``conv1d_unsqueeze`` / ``ssm_a_to_a_log`` / ``flatten``
/ ``gate_1d_unsqueeze``) are one-liners inlined in ``remap_arrays``; the
non-trivial ones live here.
"""

from __future__ import annotations

import re

import mlx.core as mx


# Legacy per-expert MoE tensor: blk.{layer}.ffn_{gate,up,down}.{expert}.weight.
_SPLIT_EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)\.(\d+)\.weight$")


def coalesce_split_experts(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
) -> tuple[dict[str, mx.array], dict[str, str], int]:
    """Stack legacy per-expert MoE weights into the modern stacked ``_exps`` form.

    Old llama.cpp Mixtral (and similar) GGUFs store each expert's projection as
    its own tensor ``blk.{L}.ffn_{gate,up,down}.{E}.weight`` rather than the
    stacked ``blk.{L}.ffn_{gate,up,down}_exps.weight`` that newer conversions -
    and every other MoE arch handled here - use. mlx-lm's ``SwitchGLU`` wants the
    stacked ``(n_experts, out, in)`` layout, so collapse each per-expert group by
    stacking its wire bytes along a new leading expert axis. The kquant wire bytes
    stack losslessly (per-block byte rows are the trailing axis) - a pure repack,
    no dequant. No-op when a GGUF already uses the stacked form.

    Returns ``(arrays, kquant_meta, n_groups)`` with each per-expert group
    replaced by its single stacked tensor (plus a vestigial ``.scales``).
    """
    groups: dict[tuple[int, str], dict[int, str]] = {}
    for name in arrays:
        m = _SPLIT_EXPERT_RE.match(name)
        if m:
            bid, proj, xid = int(m.group(1)), m.group(2), int(m.group(3))
            groups.setdefault((bid, proj), {})[xid] = name
    if not groups:
        return arrays, kquant_meta, 0

    arrays = dict(arrays)
    kquant_meta = dict(kquant_meta)
    for (bid, proj), by_xid in groups.items():
        n_exp = len(by_xid)
        # Experts are always indexed 0..n_exp-1; KeyError here is a loud, correct
        # failure on a malformed (non-contiguous) expert range.
        members = [by_xid[e] for e in range(n_exp)]
        stacked = mx.stack([arrays[n] for n in members], axis=0)
        codec = kquant_meta.get(members[0])
        for n in members:
            arrays.pop(n, None)
            arrays.pop(n[: -len(".weight")] + ".scales", None)
            kquant_meta.pop(n, None)
        new_name = f"blk.{bid}.ffn_{proj}_exps.weight"
        arrays[new_name] = stacked
        arrays[new_name[: -len(".weight")] + ".scales"] = mx.zeros(
            (1,), dtype=mx.uint8)
        if codec is not None:
            kquant_meta[new_name] = codec
    return arrays, kquant_meta, len(groups)


_SHEXP_FUSE_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up)_shexp\.weight$")


def fuse_shexp_gate_up(
    arrays: dict[str, mx.array],
    kquant_meta: dict[str, str],
) -> tuple[dict[str, mx.array], dict[str, str], int]:
    """Fuse each layer's shared-expert ``ffn_gate_shexp`` + ``ffn_up_shexp``
    into a single ``blk.{L}.ffn_gate_up_shexp.weight``.

    granitemoehybrid's shared MLP is a *fused* projection - mlx-lm builds one
    ``input_linear`` of width ``2*shared_intermediate`` and splits its output in
    half at forward (gate first, then up); its ``sanitize`` has no path that
    accepts the two halves separately. The GGUF stores them as two tensors, so
    concatenate the wire bytes along axis 0 (the output-rows axis - gate rows
    first to match the forward split). Each kquant row is an independent block
    row, so the concat is lossless - a pure repack, no dequant (the same
    invariant ``coalesce_split_experts`` relies on for stacking).

    Arch-gated at the call site (only granitemoehybrid wants the fusion -
    qwen2moe/glm4moe/deepseek2/ernie/hunyuan keep their shexp halves separate).
    Returns ``(arrays, kquant_meta, n_fused)``.
    """
    pairs: dict[int, dict[str, str]] = {}
    for name in arrays:
        m = _SHEXP_FUSE_RE.match(name)
        if m:
            pairs.setdefault(int(m.group(1)), {})[m.group(2)] = name
    if not pairs:
        return arrays, kquant_meta, 0

    arrays = dict(arrays)
    kquant_meta = dict(kquant_meta)
    for bid, by_proj in sorted(pairs.items()):
        if set(by_proj) != {"gate", "up"}:
            raise ValueError(
                f"fuse_shexp_gate_up: layer {bid} has {sorted(by_proj)} but "
                f"needs both gate and up shared-expert tensors")
        gate_n, up_n = by_proj["gate"], by_proj["up"]
        gate, up = arrays[gate_n], arrays[up_n]
        codec_g, codec_u = kquant_meta.get(gate_n), kquant_meta.get(up_n)
        if codec_g != codec_u or gate.shape[1:] != up.shape[1:]:
            raise ValueError(
                f"fuse_shexp_gate_up: layer {bid} gate/up shexp codecs or row "
                f"widths differ ({codec_g} {tuple(gate.shape)} vs {codec_u} "
                f"{tuple(up.shape)}) - cannot fuse losslessly")
        fused = mx.concatenate([gate, up], axis=0)
        for n in (gate_n, up_n):
            arrays.pop(n, None)
            arrays.pop(n[: -len(".weight")] + ".scales", None)
            kquant_meta.pop(n, None)
        new_name = f"blk.{bid}.ffn_gate_up_shexp.weight"
        arrays[new_name] = fused
        arrays[new_name[: -len(".weight")] + ".scales"] = mx.zeros(
            (1,), dtype=mx.uint8)
        if codec_g is not None:
            kquant_meta[new_name] = codec_g
    return arrays, kquant_meta, len(pairs)


def split_fused_gate_up_kquant(w: mx.array) -> tuple[mx.array, mx.array]:
    """Split a fused MoE gate-up wire-byte tensor along the byte-axis midpoint.

    Input shape (after GGUF axis reversal): ``(n_experts, 2 * intermediate,
    bytes_per_row)``. Block boundaries are along the last axis, so splitting
    ``axis=-2`` preserves per-block alignment.

    Both halves are materialized ROW-CONTIGUOUS at load. A bare slice
    ``w[..., :half, :]`` keeps the fused expert (leading) stride ``2*half``, so
    it's matrix-contiguous but not row_contiguous - which forces the prefill
    ``gather_qmm_rhs_nax`` leaf to copy the whole ~142 MB half to dense packing
    on every gather (~0.6 ms each, ~40 ms / 512-tok prefill forward across
    gate+up x 30 layers). Paying one contiguous copy here instead removes that
    per-gather prefill cost (and the per-call strided-access overhead at
    decode). Memory-neutral: the two halves sum to the fused tensor's size, and
    the fused array is freed after the split. (Load-bearing for prefill perf.)
    """
    if w.ndim != 3:
        raise ValueError(f"expected 3D fused expert tensor, got shape {w.shape}")
    half = w.shape[-2] // 2
    return (mx.contiguous(w[..., :half, :]),
            mx.contiguous(w[..., half:, :]))


def qk_permute_wire(w: mx.array, n_head: int) -> mx.array:
    """Invert llama.cpp's ``LlamaModel.permute`` on Q/K weight rows.

    The forward permute is ``reshape(N, 2, D)`` -> ``swapaxes(1, 2)``; the
    inverse is ``reshape(N, D, 2)`` -> ``swapaxes(1, 2)``. Not self-inverse when
    ``D != 2``. mlx-lm's llama/mistral3 attention uses
    ``mx.fast.rope(traditional=False)`` (the HF concat-half layout), so we must
    undo the permute when loading from a GGUF that convert_hf_to_gguf permuted.
    """
    n_out = w.shape[0]
    head_dim = n_out // n_head
    return (w.reshape(n_head, head_dim // 2, 2, *w.shape[1:])
              .swapaxes(1, 2)
              .reshape(n_out, *w.shape[1:]))


def qk_permute_wire_inverse(w: mx.array, n_head: int) -> mx.array:
    """Apply llama.cpp's ``LlamaModel.permute`` to Q/K rows - the inverse of
    :func:`qk_permute_wire`, used when *emitting* a GGUF (e.g. a trained LoRA
    adapter's Q/K ``lora_b``). The forward permute is ``reshape(N, 2, D)`` ->
    ``swapaxes(1, 2)``, so de-permuting then re-permuting recovers the input:
    ``qk_permute_wire(qk_permute_wire_inverse(w)) == w``.
    """
    n_out = w.shape[0]
    head_dim = n_out // n_head
    return (w.reshape(n_head, 2, head_dim // 2, *w.shape[1:])
              .swapaxes(1, 2)
              .reshape(n_out, *w.shape[1:]))


def retarget(name: str, target_prefix: str) -> str:
    """Prepend ``target_prefix`` to ``name`` (no-op when prefix is empty)."""
    if not target_prefix:
        return name
    already = name == target_prefix or name.startswith(target_prefix + ".")
    return name if already else f"{target_prefix}.{name}"
