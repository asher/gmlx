#!/usr/bin/env python3
"""The wire load's handling of a tensor past the device buffer ceiling:
it is skipped (no MTLBuffer can hold it) and the model reads its rows
from the GGUF. See gmlx.stream.table_pread."""

from __future__ import annotations

from types import SimpleNamespace

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
    assert wire._deferred_table_names([table_gguf], "deepseek41") == [
        "blk.1.engram_embd.weight"]
    assert wire._deferred_table_names([table_gguf], None) == [
        "blk.1.engram_embd.weight"]        # arch read from the header
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", str(1 << 40))
    assert wire._deferred_table_names([table_gguf], "deepseek41") == []


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


def test_another_tensor_past_the_ceiling_is_named(table_gguf, monkeypatch):
    """Only lookup tables can be read from the file. Any other tensor that
    no buffer can hold fails the load with its name, not Metal's bare
    allocation error."""
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "100")

    def load_gguf(path, zero_copy=True, skip=()):
        raise RuntimeError(
            "[metal::malloc] Attempting to allocate 128 bytes which is greater "
            "than the maximum allowed buffer size of 100 bytes.")

    monkeypatch.setattr(wire.kq, "load_gguf", load_gguf)
    with pytest.raises(RuntimeError, match="token_embd.weight is larger"):
        wire.load_gguf_wire_bytes(table_gguf, shards=[table_gguf],
                                  expect_quant=False)


@pytest.fixture
def mtp_table_gguf(tmp_path):
    """The table and one Q8_0 matrix, so the wire load finds a codec."""
    path = str(tmp_path / "m.gguf")
    w = gguf.GGUFWriter(path, "deepseek41")
    w.add_tensor("blk.1.engram_embd.weight",
                 np.ones((ROWS, ROW_BYTES), np.int8),
                 raw_dtype=gguf.GGMLQuantizationType.I8)
    w.add_tensor("blk.0.attn_q.weight", np.zeros((4, 34), np.uint8),
                 raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_the_mtp_target_install_gets_the_skipped_tables(mtp_table_gguf,
                                                        monkeypatch):
    """load_mtp_model hands the install the tables its wire load skipped,
    so the model reads their rows from the file. Without them the fill
    check stops the load: the table's parameter has no tensor."""
    import gmlx.load.config_synth as config_synth
    import gmlx.spec.mtp_load as ml

    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "1000")
    pf = SimpleNamespace(arch="deepseek41", shards=[mtp_table_gguf])
    got = {}

    class Installed(Exception):
        pass

    def install(model, hf_weights, hf_kquant_meta, **kwargs):
        got.update(kwargs)
        raise Installed

    # Everything between the wire load and the install is the model's own.
    monkeypatch.setattr(ml, "preflight", lambda path, arch=None: pf)
    monkeypatch.setattr(ml, "maybe_populate_for_load", lambda *a, **k: None)
    monkeypatch.setattr(config_synth, "synthesize_config", lambda meta, shapes: {
        "model_type": "deepseek_v41", "mtp_num_hidden_layers": 1})
    monkeypatch.setattr(ml, "_resolve_companion_drafter", lambda *a, **k: None)
    monkeypatch.setattr(ml, "require_native_head_tensors", lambda *a: None)
    monkeypatch.setattr(ml, "read_int", lambda *a: 1)
    monkeypatch.setattr(ml, "first_nonzero_int", lambda *a: 1)
    monkeypatch.setattr(ml, "remap_arrays", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(ml, "build_model", lambda cfg, mtp=False: (
        SimpleNamespace(language_model=None), cfg))
    monkeypatch.setattr(ml.rope_swap, "apply", lambda *a: 0)
    monkeypatch.setattr(ml, "_install_and_load", install)

    with pytest.raises(Installed):
        ml.load_mtp_model(mtp_table_gguf)
    tables = got.get("deferred_tables") or {}
    assert list(tables) == ["blk.1.engram_embd.weight"]
    assert tables["blk.1.engram_embd.weight"].rows == ROWS



@pytest.mark.parametrize("table_codec", ["stq1_0", "mxfp8"])
def test_an_mxfp8_tensor_read_as_stq1_0_is_refused(tmp_path, monkeypatch,
                                                   table_codec):
    """An mlx-kquant from before MXFP8 reads every type-43 tensor as STQ1_0.
    MXFP8 is gather-only, so a token embedding may use it too; one loaded
    as an array must not decode as ternary weights. A build that names it
    MXFP8 loads it."""
    from helpers import mint_type43_gguf

    path = str(tmp_path / "t.gguf")
    mint_type43_gguf(path, [
        ("blk.0.ffn_up.weight", (256, 4), "STQ1_0", None),
        ("token_embd.weight", (256, 4), "MXFP8", None)])
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", str(1 << 40))
    codecs = {"blk.0.ffn_up.weight": "stq1_0",
              "token_embd.weight": table_codec}
    monkeypatch.setattr(wire.kq, "load_gguf", lambda p, zero_copy=True, skip=(): (
        {}, codecs, {"general.architecture": "deepseek41"}, {}))
    if table_codec == "stq1_0":
        with pytest.raises(RuntimeError, match="is an MXFP8 tensor"):
            wire.load_gguf_wire_bytes(path, shards=[path], expect_quant=False)
    else:
        wire.load_gguf_wire_bytes(path, shards=[path], expect_quant=False)


@pytest.mark.parametrize("with_weight", [False, True])
def test_a_file_read_mxfp8_table_loads(tmp_path, monkeypatch, with_weight):
    """The MXFP8 table is skipped and read from the file, so the load goes
    on whatever codec the installed mlx-kquant names for it. The STQ1_0
    weight beside it still loads as one."""
    from helpers import mint_type43_gguf

    path = str(tmp_path / "t.gguf")
    weight = [("blk.0.ffn_up.weight", (256, 4), "STQ1_0", None)]
    table = [("per_layer_token_embd.weight", (256, 4), "MXFP8", None)]
    mint_type43_gguf(path, (weight if with_weight else []) + table)
    monkeypatch.setenv("GMLX_TABLE_MAX_BUFFER", "500")
    arrays, codecs, arch, _meta, _shapes = wire.load_gguf_wire_bytes(
        path, shards=[path], expect_quant=False)
    assert arch == "deepseek41"
    assert "per_layer_token_embd.weight" not in arrays
    if with_weight:
        assert codecs["blk.0.ffn_up.weight"] == "stq1_0"
