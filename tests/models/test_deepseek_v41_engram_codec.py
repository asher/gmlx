#!/usr/bin/env python3
"""The ds4 engram row codec (gmlx.models.deepseek_v41.engram_codec) against
a port of the reference decoder, ds4_engram.c:140-181."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from gmlx.models.deepseek_v41.engram_codec import (
    decode_rows,
    parse_row_encoding,
    row_width,
    tables,
)

DIM, BLOCK, ROW_BYTES = 256, 32, 264
ENCODING = f"e4m3_e8m0_{BLOCK}_row{ROW_BYTES}"


def _ref_e4m3(byte: int) -> float:
    """ds4_engram.c:140-146."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    value = (math.ldexp(8 + mantissa, exponent - 10) if exponent
             else math.ldexp(mantissa, -9))
    return -value if byte & 128 else value


def _ref_bf16(value: float) -> float:
    """The round-to-nearest-even truncation ds4_engram.c:170-173 does."""
    bits = int(np.float32(value).view(np.uint32))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return float(np.uint32(bits).view(np.float32))


def _ref_row(raw: np.ndarray) -> np.ndarray:
    """ds4_engram.c:160-180 for one row; raises on the NaN codes."""
    out = np.empty(DIM, dtype=np.float32)
    for j in range(DIM):
        code, scale = int(raw[j]), int(raw[DIM + j // BLOCK])
        if (code & 127) == 127 or scale == 255:
            raise ValueError("corrupt row")
        out[j] = _ref_bf16(math.ldexp(_ref_e4m3(code), scale - 127))
    return out


def _clean_row(rng) -> np.ndarray:
    """A row the reference accepts: no NaN codes, and scales in the range
    the real tables use (the file samples at 119-120), so no product
    overflows float32 - the reference rejects those too."""
    raw = rng.integers(0, 256, size=ROW_BYTES, dtype=np.uint8)
    raw[:DIM][raw[:DIM] & 127 == 127] = 3
    raw[DIM:] = rng.integers(100, 140, size=ROW_BYTES - DIM, dtype=np.uint8)
    return raw


# --- the lookup tables ----------------------------------------------------


def test_e4m3_table_matches_the_reference():
    value, _scale = tables()
    for code in range(256):
        if (code & 127) == 127:
            assert math.isnan(value[code]), code
            continue
        assert value[code] == np.float32(_ref_e4m3(code)), code


def test_e8m0_table_is_a_power_of_two_with_a_nan_code():
    _value, scale = tables()
    assert math.isnan(scale[255])
    for code in (0, 1, 127, 150, 254):
        assert scale[code] == np.float32(math.ldexp(1.0, code - 127)), code


# --- whole rows -----------------------------------------------------------


def test_decoded_rows_match_the_reference():
    rng = np.random.default_rng(11)
    raw = np.stack([_clean_row(rng) for _ in range(8)])
    want = np.stack([_ref_row(r) for r in raw])

    codec = parse_row_encoding(ENCODING)
    value, scale = tables()
    got = decode_rows(mx.array(raw), codec, DIM,
                      mx.array(value), mx.array(scale))
    # The model emits bf16; the reference rounds the same way.
    got = np.array(got.astype(mx.bfloat16).astype(mx.float32))
    assert np.array_equal(got, want)


def test_a_signed_row_decodes_the_same():
    """GGML's I8 loads signed, so the gather may hand over int8."""
    rng = np.random.default_rng(12)
    raw = _clean_row(rng)[None]
    codec = parse_row_encoding(ENCODING)
    value, scale = tables()
    args = (codec, DIM, mx.array(value), mx.array(scale))
    assert mx.array_equal(decode_rows(mx.array(raw), *args),
                          decode_rows(mx.array(raw.view(np.int8)), *args))


@pytest.mark.parametrize("spoil", ["code", "scale"])
def test_the_nan_codes_propagate(spoil):
    """The reference errors; a graph cannot branch, so it carries NaN."""
    rng = np.random.default_rng(13)
    raw = _clean_row(rng)[None]
    raw[0, 5 if spoil == "code" else DIM] = 127 if spoil == "code" else 255
    with pytest.raises(ValueError):
        _ref_row(raw[0])
    value, scale = tables()
    got = decode_rows(mx.array(raw), parse_row_encoding(ENCODING), DIM,
                      mx.array(value), mx.array(scale))
    hit = np.array(mx.isnan(got))[0]
    assert hit[5] if spoil == "code" else hit[:BLOCK].all()


# --- the encoding string --------------------------------------------------


def test_row_width_checks_the_named_width():
    assert row_width(DIM, ENCODING) == ROW_BYTES
    assert row_width(DIM, None) == DIM
    with pytest.raises(ValueError, match="claims 263 bytes"):
        row_width(DIM, f"e4m3_e8m0_{BLOCK}_row263")


def test_unknown_encodings_are_refused():
    assert parse_row_encoding(None) is None
    assert parse_row_encoding("") is None
    for bad in ("e5m2_e8m0_32_row264", "e4m3_ue8m0_32_row264", "e4m3_32_264"):
        with pytest.raises(ValueError, match="unsupported engram row"):
            parse_row_encoding(bad)


def test_an_overflowing_scale_carries_infinity():
    """The reference rejects a row whose product leaves float32; the graph
    carries the infinity instead, same as for the NaN codes."""
    raw = np.full((1, ROW_BYTES), 0x78, dtype=np.uint8)   # E4M3 max, 448
    raw[0, DIM:] = 254                                    # 2**127
    value, scale = tables()
    got = decode_rows(mx.array(raw), parse_row_encoding(ENCODING), DIM,
                      mx.array(value), mx.array(scale))
    assert not bool(mx.all(mx.isfinite(got)))
