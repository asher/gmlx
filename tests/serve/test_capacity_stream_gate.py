"""Preload gate discounts the bytes a streaming build serves from disk.

The U4 preload gate refuses a model whose weights exceed the working
budget; ``stream: experts`` serves the routed-expert stacks from disk,
plus any lookup table the arch declares streamable, so the gate must
judge only the resident share (the 153 GB antirez DSv4 GGUF regression:
refused on full size after the gate shipped)."""

import numpy as np

from gmlx.serve.capacity import (
    preload_gate_bytes,
    streamed_expert_bytes,
    streamed_off_disk_bytes,
    streamed_table_bytes,
)


def _mint_moe(path):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 1)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
    w.add_tensor("blk.0.ffn_gate_exps.weight",
                 np.zeros((8, 64, 64), dtype=np.float16))
    w.add_tensor("blk.0.ffn_down_exps.weight",
                 np.zeros((8, 64, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_streamed_expert_bytes_sums_exps_tensors(tmp_path):
    p = _mint_moe(tmp_path / "moe.gguf")
    got = streamed_expert_bytes(p)
    assert got == 2 * 8 * 64 * 64 * 2          # the two _exps stacks, f16


def test_streamed_expert_bytes_unreadable_is_zero(tmp_path):
    assert streamed_expert_bytes(str(tmp_path / "missing.gguf")) == 0


def test_preload_gate_bytes_discounts_only_expert_streaming():
    assert preload_gate_bytes(100, "experts", 70) == 30
    assert preload_gate_bytes(100, "experts", 0) == 100    # header unreadable
    assert preload_gate_bytes(100, "cpu", 70) == 100       # unified RAM, no discount
    assert preload_gate_bytes(100, None, 70) == 100
    assert preload_gate_bytes(50, "experts", 70) == 0      # clamped, never negative


def _mint_engram(path):
    """A deepseek41-shaped header: two engram tables plus one expert stack."""
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "deepseek41")
    w.add_uint32("deepseek41.block_count", 2)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
    w.add_tensor("blk.0.ffn_gate_exps.weight",
                 np.zeros((8, 64, 64), dtype=np.float16))
    for i in (1, 14):
        w.add_tensor(f"blk.{i}.engram_embd.weight",
                     np.zeros((128, 64), dtype=np.float16))
        w.add_tensor(f"blk.{i}.engram_q.weight",
                     np.zeros((4, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_streamed_table_bytes_sums_declared_tables(tmp_path):
    p = _mint_engram(tmp_path / "ds41.gguf")
    assert streamed_table_bytes(p) == 2 * 128 * 64 * 2    # engram_q excluded
    assert streamed_expert_bytes(p) == 8 * 64 * 64 * 2
    assert streamed_off_disk_bytes(p) == (
        streamed_table_bytes(p) + streamed_expert_bytes(p))


def test_streamed_table_bytes_zero_without_a_declared_arch(tmp_path):
    p = _mint_moe(tmp_path / "moe.gguf")               # llama: no tables
    assert streamed_table_bytes(p) == 0
    assert streamed_off_disk_bytes(p) == streamed_expert_bytes(p)


def test_streamed_table_bytes_unreadable_is_zero(tmp_path):
    assert streamed_table_bytes(str(tmp_path / "missing.gguf")) == 0
