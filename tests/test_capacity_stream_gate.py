"""Preload gate discounts streamed routed-expert bytes.

The U4 preload gate refuses a model whose weights exceed the working
budget; ``stream: experts`` serves the routed-expert stacks from disk,
so the gate must judge only the resident share (the 153 GB antirez
DSv4 GGUF regression: refused on full size after the gate shipped)."""

import numpy as np

from gmlx.capacity import preload_gate_bytes, streamed_expert_bytes


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
