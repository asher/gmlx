"""Native floating-point GGUF codecs (MXFP4, NVFP4) -> MLX's packed layout.

MLX ships dedicated kernels for the micro-scaling float formats
(``mx.quantized_matmul`` / ``mx.gather_qmm`` with ``mode="mxfp4"`` / ``"nvfp4"``).
GGML stores the same formats in a block layout that (a) interleaves each group's
scale byte(s) with the packed nibbles and (b) orders the 4-bit E2M1 codes in
ggml's "two halves" split. MLX wants the scales split into their own array and
the nibbles packed sequentially (8 per ``uint32``, element 0 in the low 4 bits).

The ``*_deinterleave`` repack bridges the two. It is a **pure byte/nibble
shuffle**: the E2M1 nibble codes and the per-group scale bytes are copied
unchanged - only their positions move. There is no dequantization, no
requantization, and no precision change; ``mx.dequantize`` of the repacked
arrays is bit-for-bit the values ggml stored.

Codec facts (ggml-common.h):
  * MXFP4  - ``block_mxfp4 {u8 e; u8 qs[16];}`` = 17 B / 32 vals. One E8M0 scale
    byte per 32-value group. ``qs[j]&0xF`` -> value ``j`` (j<16);
    ``qs[j]>>4`` -> value ``j+16``.
  * NVFP4  - ``block_nvfp4 {u8 d[4]; u8 qs[32];}`` = 36 B / 64 vals = four
    16-value groups, each with its own UE4M3 scale byte; the same two-halves
    split applies within each 16-value sub-block. No per-tensor global scale.

mxfp8 is intentionally absent: MLX has a kernel for it, but GGML defines no
mxfp8 wire type, so it can never appear in a GGUF (listed in
``NATIVE_FP_MLX_ONLY`` for documentation only).
"""

from __future__ import annotations

import numpy as np

# codec -> (group_size, bits, ggml_bytes_per_block, ggml_vals_per_block)
# group_size/bits are MLX's quantization params; the ggml block geometry is what
# the C++ loader reads as raw wire bytes (see kquant_gguf.cpp gguf_type_to_fp_codec).
NATIVE_FP_GEOMETRY: dict[str, tuple[int, int, int, int]] = {
    "mxfp4": (32, 4, 17, 32),
    "nvfp4": (16, 4, 36, 64),
}
NATIVE_FP_CODECS = frozenset(NATIVE_FP_GEOMETRY)

# MLX has a native kernel for these, but GGML defines no wire type, so a GGUF
# can't carry them - kept only so callers can reason about full MLX coverage.
NATIVE_FP_MLX_ONLY = frozenset({"mxfp8"})

# OCP E2M1 code (raw 4-bit nibble) -> value. Shared by mxfp4 and nvfp4; matches
# ggml's kvalues_mxfp4 (which stores 2x these and compensates with a half-scale)
# and MLX's fp4_e2m1 decode. Provided for reference/validation, not used by the
# repack itself (which never looks at the values).
E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32)


def _repack_groups_sequential(data_bytes: np.ndarray) -> np.ndarray:
    """ggml two-halves nibble groups -> MLX sequential packed uint32.

    ``data_bytes``: ``(..., n_groups, G//2)`` uint8, where each group's ``G//2``
    bytes hold ``G`` E2M1 codes in ggml order - low nibble of byte ``b`` is
    value ``b``, high nibble is value ``b + G//2``. Returns
    ``(..., n_groups * G//8)`` uint32 with the codes packed sequentially
    (8 per uint32, value 0 in the low 4 bits), matching MLX's fp4 packing.
    """
    *lead, ngrp, half = data_bytes.shape            # half = G // 2
    lo = data_bytes & 0x0F                           # values 0 .. half-1
    hi = data_bytes >> 4                             # values half .. G-1
    v = np.concatenate([lo, hi], axis=-1)            # (..., ngrp, G) sequential
    out = (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)   # (..., ngrp, G//2)
    out = np.ascontiguousarray(out)
    u32 = out.reshape(-1, half).view(np.uint32)      # (M, G//8); little-endian
    return u32.reshape(*lead, ngrp * (half // 4))


def mxfp4_deinterleave(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml block_mxfp4 wire bytes -> (packed uint32, scales uint8 E8M0).

    ``raw`` last axis = whole 17-byte blocks. Returns ``packed`` (last axis =
    n_blocks*4 uint32) and ``scales`` (last axis = n_blocks uint8).
    """
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    *lead, last = raw.shape
    blk = 17
    if last % blk != 0:
        raise ValueError(f"mxfp4: last dim {last} not a multiple of {blk}")
    nblk = last // blk
    blocks = raw.reshape(*lead, nblk, blk)
    scales = np.ascontiguousarray(blocks[..., 0])             # (..., nblk)
    data = blocks[..., 1:]                                     # (..., nblk, 16)
    packed = _repack_groups_sequential(data)                  # 1 group / block
    return packed, scales


def nvfp4_deinterleave(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml block_nvfp4 wire bytes -> (packed uint32, scales uint8 UE4M3).

    ``raw`` last axis = whole 36-byte blocks (4 UE4M3 sub-scales + 32 nibble
    bytes = four 16-value groups). Returns ``packed`` (last axis = n_blocks*8
    uint32) and ``scales`` (last axis = n_blocks*4 uint8, one per 16-value group).
    """
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    *lead, last = raw.shape
    blk, n_sub, g = 36, 4, 16
    if last % blk != 0:
        raise ValueError(f"nvfp4: last dim {last} not a multiple of {blk}")
    nblk = last // blk
    blocks = raw.reshape(*lead, nblk, blk)
    scales = np.ascontiguousarray(
        blocks[..., :n_sub]).reshape(*lead, nblk * n_sub)     # (..., nblk*4)
    data = blocks[..., n_sub:].reshape(*lead, nblk * n_sub, g // 2)
    packed = _repack_groups_sequential(data)                  # 4 groups / block
    return packed, scales


_DEINTERLEAVE = {
    "mxfp4": mxfp4_deinterleave,
    "nvfp4": nvfp4_deinterleave,
}


def _strip_weight(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def repack_native_fp_weights(hf_weights: dict, hf_codec_meta: dict) -> int:
    """Repack every native-fp tensor in ``hf_weights`` from raw ggml wire bytes
    into MLX's native ``(packed uint32, scales uint8)`` form, in place.

    ``hf_codec_meta`` maps post-remap tensor name -> codec. For each whose codec
    is a native-fp codec, ``<name>`` (the ``.weight``) is replaced with the
    packed uint32 array and its sibling ``<base>.scales`` placeholder with the
    real per-group scale bytes. Returns the number of tensors repacked. Imports
    ``mlx`` lazily so the de-interleave helpers stay importable numpy-only (for
    unit tests) without pulling in mlx.
    """
    import mlx.core as mx

    n = 0
    for name, codec in list(hf_codec_meta.items()):
        if codec not in NATIVE_FP_CODECS:
            continue
        raw = hf_weights.get(name)
        if raw is None:
            continue
        packed_np, scales_np = _DEINTERLEAVE[codec](np.asarray(raw))
        # Free each wire view + temporaries as we go so peak stays ~packed + one
        # tensor, not wire + packed for the whole model (fits 120B).
        scales_key = _strip_weight(name) + ".scales"
        hf_weights[name] = mx.array(packed_np)
        hf_weights[scales_key] = mx.array(scales_np)
        mx.eval(hf_weights[name], hf_weights[scales_key])
        del raw, packed_np, scales_np
        mx.clear_cache()
        n += 1
    return n
