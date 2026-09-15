# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The engram table's row codec.

The llama.cpp conversion stores the two 384M-row tables as a normal GGUF
quant, so the row is ``key_length`` values wide and mlx-kquant decodes it.
The ds4 conversion stores raw bytes under GGML's ``I8`` type and names the
layout in ``deepseek41.engram.encoding``, e.g. ``e4m3_e8m0_32_row264``:
256 fp8-E4M3 values followed by 8 E8M0 exponent bytes, one per 32 values.
Decoding is a 256-entry lookup and a per-block scale, so it needs no
kernel; the gather that feeds it reads 24 rows per token per table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_ENCODING_RE = re.compile(
    r"^(?P<value>[a-z0-9]+)_(?P<scale>[a-z0-9]+)_(?P<block>\d+)_row(?P<row>\d+)$")

__all__ = ["RowCodec", "decode_rows", "parse_row_encoding", "row_width"]


@dataclass(frozen=True)
class RowCodec:
    value: str
    scale: str
    block: int
    row_bytes: int


def parse_row_encoding(encoding: str | None) -> RowCodec | None:
    """The named byte layout, or None for a native GGUF quant."""
    if not encoding:
        return None
    m = _ENCODING_RE.match(encoding)
    if m is None or m.group("value") != "e4m3" or m.group("scale") != "e8m0":
        raise ValueError(f"unsupported engram row encoding {encoding!r}")
    return RowCodec(value=m.group("value"), scale=m.group("scale"),
                    block=int(m.group("block")), row_bytes=int(m.group("row")))


def row_width(head_dim: int, encoding: str | None) -> int:
    codec = parse_row_encoding(encoding)
    if codec is None:
        return int(head_dim)
    want = head_dim + -(-head_dim // codec.block)
    if codec.row_bytes != want:
        raise ValueError(
            f"engram row encoding {encoding!r} claims {codec.row_bytes} bytes "
            f"but {head_dim} values in blocks of {codec.block} need {want}")
    return codec.row_bytes


def _e4m3_table() -> np.ndarray:
    """Every E4M3 byte as a float32. NaN (exponent 15, mantissa 7) stays NaN,
    which the caller treats as a corrupt row rather than a value."""
    code = np.arange(256, dtype=np.uint32)
    exp = (code >> 3) & 15
    man = code & 7
    out = np.where(exp > 0,
                   (8 + man).astype(np.float32) * np.exp2(
                       exp.astype(np.float32) - 10),
                   man.astype(np.float32) * np.float32(2.0 ** -9))
    out = np.where((code & 127) == 127, np.float32("nan"), out)
    return np.where(code & 128, -out, out).astype(np.float32)


def _e8m0_table() -> np.ndarray:
    """Every E8M0 byte as a float32 power of two; 255 is the NaN code."""
    code = np.arange(256, dtype=np.int32)
    with np.errstate(over="ignore"):
        out = np.exp2((code - 127).astype(np.float64)).astype(np.float32)
    out[255] = np.float32("nan")
    return out


def tables():
    """(value, scale) lookup tables, float32 [256]."""
    return _e4m3_table(), _e8m0_table()


def decode_rows(raw, codec: RowCodec, head_dim: int, value_lut, scale_lut):
    """Raw rows ``[..., row_bytes]`` uint8 -> ``[..., head_dim]``.

    ``value_lut`` / ``scale_lut`` are the arrays :func:`tables` returns, as
    mx arrays on the caller's device. A corrupt row decodes to NaN, which
    the reference runtime treats as an error; the graph cannot branch, so
    it propagates instead.
    """
    import mlx.core as mx

    # GGML's I8 loads signed, so mask back to the byte value.
    codes = raw[..., :head_dim].astype(mx.int32) & 0xFF
    scales = raw[..., head_dim:].astype(mx.int32) & 0xFF
    values = value_lut[codes]
    return values * mx.repeat(scale_lut[scales], codec.block, axis=-1)
