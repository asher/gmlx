#!/usr/bin/env python3
"""The wire load's handling of a tensor past the device buffer ceiling:
it is skipped (no MTLBuffer can hold it) and the model reads its rows
from the GGUF. See gmlx.stream.table_pread."""

from __future__ import annotations

import numpy as np
import pytest

import gmlx.load.wire as wire

gguf = pytest.importorskip("gguf")

ROWS, ROW_BYTES = 500, 33


@pytest.fixture
def table_gguf(tmp_path):
    path = str(tmp_path / "t.gguf")
    w = gguf.GGUFWriter(path, "deepseek41")
    w.add_tensor("blk.1.engram_embd.weight",
                 np.ones((ROWS, ROW_BYTES), np.int8),
                 raw_dtype=gguf.GGMLQuantizationType.I8)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_names_the_tables_past_the_ceiling(table_gguf, monkeypatch):
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "1000")
    assert wire._oversize_table_names([table_gguf], "deepseek41") == [
        "blk.1.engram_embd.weight"]
    assert wire._oversize_table_names([table_gguf], None) == [
        "blk.1.engram_embd.weight"]        # arch read from the header
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", str(1 << 40))
    assert wire._oversize_table_names([table_gguf], "deepseek41") == []


def test_skips_the_table_and_loads_the_rest(table_gguf, monkeypatch):
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "1000")
    arrays, _codecs, arch, _meta, shapes = wire.load_gguf_wire_bytes(
        table_gguf, shards=[table_gguf], expect_quant=False)
    assert arch == "deepseek41"
    assert "blk.1.engram_embd.weight" not in arrays
    assert "token_embd.weight" in arrays
    # The geometry survives; only the array is gone.
    assert list(shapes["blk.1.engram_embd.weight"]) == [ROW_BYTES, ROWS]


def test_an_mlx_kquant_without_skip_is_named(table_gguf, monkeypatch):
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "1000")

    def old_load_gguf(path, zero_copy=True):
        raise AssertionError("unreachable")

    monkeypatch.setattr(wire.kq, "load_gguf", old_load_gguf)
    with pytest.raises(RuntimeError, match="past the device buffer ceiling"):
        wire.load_gguf_wire_bytes(table_gguf, shards=[table_gguf],
                                  expect_quant=False)
