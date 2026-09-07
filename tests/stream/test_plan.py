"""The streaming planner: header-only pricing of a MoE file and its fit
on a box (every-token weights + KV room under the ceiling, the rest is
the decode arena)."""

import json

import numpy as np
import pytest

from gmlx.stream import plan as sp
from gmlx.stream.budget import KvRoom

GiB = 1 << 30
F16 = 2


def _mint_moe(path, *, layers=2, experts=8, used=2, expert_count_kv=True,
              big_layer=None):
    """A llama-shaped MoE file. ``big_layer`` doubles that layer's expert
    stacks so the ring (twice the largest layer) is distinguishable."""
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", layers)
    w.add_uint32("llama.context_length", 4096)
    w.add_uint32("llama.embedding_length", 64)
    w.add_uint32("llama.feed_forward_length", 128)
    w.add_uint32("llama.attention.head_count", 4)
    w.add_uint32("llama.attention.head_count_kv", 2)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32("llama.rope.dimension_count", 16)
    if expert_count_kv:
        w.add_uint32("llama.expert_count", experts)
        w.add_uint32("llama.expert_used_count", used)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float16))
    w.add_tensor("output.weight", np.zeros((32, 64), dtype=np.float16))
    for i in range(layers):
        rows = 128 if i == big_layer else 64
        w.add_tensor(f"blk.{i}.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
        w.add_tensor(f"blk.{i}.ffn_gate_inp.weight",
                     np.zeros((experts, 64), dtype=np.float16))
        w.add_tensor(f"blk.{i}.ffn_gate_shexp.weight",
                     np.zeros((16, 64), dtype=np.float16))
        for kind in ("gate", "up", "down"):
            w.add_tensor(f"blk.{i}.ffn_{kind}_exps.weight",
                         np.zeros((experts, rows, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _mint_dense(path):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "llama")
    w.add_uint32("llama.block_count", 1)
    w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float16))
    w.add_tensor("blk.0.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
    w.add_tensor("blk.0.ffn_up.weight", np.zeros((64, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_group_of_names_every_token_groups():
    assert sp.group_of("blk.3.ffn_gate_exps.weight") is None
    assert sp.group_of("blk.3.ffn_gate_up_exps.weight") is None
    # The tier selects with fullmatch; a longer name stays resident.
    assert sp.group_of("blk.3.ffn_gate_exps.weight.scale") == "ffn"
    assert sp.group_of("blk.3.ffn_up_shexp.weight") == "shared_experts"
    assert sp.group_of("blk.3.attn_kv_a_mqa.weight") == "attention"
    assert sp.group_of("blk.3.ssm_conv1d_q.weight") == "recurrent"
    assert sp.group_of("blk.3.ffn_gate_inp.weight") == "ffn"
    assert sp.group_of("blk.3.ffn_routed_up.weight") == "ffn"
    assert sp.group_of("token_embd.weight") == "embedding"
    assert sp.group_of("output_norm.weight") == "output"
    assert sp.group_of("rope_freqs.weight") == "other"


def test_model_plan_prices_experts_ring_and_reads(tmp_path):
    p = _mint_moe(tmp_path / "moe.gguf", layers=2, experts=8, used=2, big_layer=1)
    m = sp.model_plan(sp.scan_path(p))
    small = 3 * 8 * 64 * 64 * F16
    big = 3 * 8 * 128 * 64 * F16
    assert m.expert_bytes == small + big
    assert m.moe_layers == 2
    assert m.n_experts == 8 and m.experts_per_token == 2
    assert m.ring_bytes == 2 * big
    assert m.per_token_read_bytes == (small + big) * 2 // 8
    every = (2 * 32 * 64 + 2 * (64 * 64 + 8 * 64 + 16 * 64)) * F16
    assert m.every_token_bytes == every
    assert m.total_bytes == every + m.expert_bytes
    assert m.groups["shared_experts"] == 2 * 16 * 64 * F16
    assert m.groups["embedding"] == 32 * 64 * F16
    assert m.streamable
    assert m.kv_costs is not None and m.kv_bytes(1) > 0
    assert m.trained_ctx == 4096


def test_model_plan_expert_count_from_shape(tmp_path):
    p = _mint_moe(tmp_path / "moe.gguf", experts=8, expert_count_kv=False)
    m = sp.model_plan(sp.scan_path(p))
    assert m.n_experts == 8
    assert m.experts_per_token is None
    assert m.per_token_read_bytes is None


def test_plan_path_none_for_dense(tmp_path):
    assert sp.plan_path(_mint_dense(tmp_path / "dense.gguf")) is None
    assert sp.plan_path(str(tmp_path / "missing.gguf")) is None
    got = sp.plan_path(_mint_moe(tmp_path / "moe.gguf"))
    assert got is not None and got[0].streamable


def test_ceiling_for_margin_and_reserve(monkeypatch):
    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.delenv("GMLX_GOV_RESERVE_GB", raising=False)
    # 128 GiB box, 3/4 working set: the margin binds.
    ram = 128 * GiB
    assert sp.ceiling_for(ram, ram * 0.75) == pytest.approx(ram * 0.75 * 0.95)
    # 16 GiB box, 2/3 working set: the 8 GB reserve binds.
    ram = 16 * GiB
    assert sp.ceiling_for(ram, ram * 2 / 3) == pytest.approx(ram - 8e9)


def _model(*, total, expert, ring=0):
    return sp.ModelPlan(arch="x", total_bytes=total, expert_bytes=expert,
                        groups={}, moe_layers=4, n_experts=64,
                        experts_per_token=8, ring_bytes=ring, kv_costs=None,
                        trained_ctx=None)


@pytest.fixture
def flat_room(monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "1")
    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.delenv("GMLX_GOV_RESERVE_GB", raising=False)


def test_box_plan_verdicts(flat_room):
    ram, ws = 60e9, 45e9
    ceiling = 45e9 * 0.95
    room = 1 * GiB
    b = sp.box_plan(_model(total=10e9, expert=8e9), ram_bytes=ram, ws_bytes=ws)
    assert b.verdict == sp.VERDICT_RESIDENT
    assert b.room == KvRoom(room, 32768, 1, 0, 0, 0, priced=False)
    b = sp.box_plan(_model(total=250e9, expert=200e9), ram_bytes=ram, ws_bytes=ws)
    assert b.verdict == sp.VERDICT_TOO_BIG
    assert b.short_bytes == pytest.approx(50e9 + room - ceiling, abs=2)
    assert b.arena_bytes == 0
    b = sp.box_plan(_model(total=241.5e9, expert=200e9), ram_bytes=ram, ws_bytes=ws)
    assert b.verdict == sp.VERDICT_PAGE_CACHE
    assert 0 < b.arena_bytes < GiB
    b = sp.box_plan(_model(total=220e9, expert=200e9, ring=5e9),
                    ram_bytes=ram, ws_bytes=ws)
    assert b.verdict == sp.VERDICT_STREAMS
    assert b.arena_bytes == pytest.approx(ceiling - 20e9 - room, abs=2)
    assert b.arena_share == pytest.approx(b.arena_bytes / 200e9)
    assert b.ring_fits
    b = sp.box_plan(_model(total=220e9, expert=200e9, ring=30e9),
                    ram_bytes=ram, ws_bytes=ws)
    assert b.verdict == sp.VERDICT_STREAMS and not b.ring_fits


def test_box_plan_arena_capped_at_experts(flat_room, monkeypatch):
    # A MoE file over RAM whose experts still fit under the ceiling.
    monkeypatch.setenv("GMLX_GOV_RESERVE_GB", "1")
    b = sp.box_plan(_model(total=52e9, expert=20e9), ram_bytes=60e9, ws_bytes=60e9)
    assert b.verdict == sp.VERDICT_STREAMS
    assert b.arena_bytes == 20e9 and b.arena_share == 1.0


def test_box_plan_none_without_a_box(monkeypatch):
    import gmlx.load.memfit as memfit
    import gmlx.serve.capacity as cap
    monkeypatch.setattr(cap, "working_set_bytes", lambda: None)
    monkeypatch.setattr(memfit, "total_ram_bytes", lambda: None)
    assert sp.box_plan(_model(total=10, expert=5)) is None


def test_lines_and_dict(flat_room):
    ram, ws = 60e9, 45e9
    m = _model(total=220e9, expert=200e9, ring=30e9)
    b = sp.box_plan(m, ram_bytes=ram, ws_bytes=ws)
    lines = sp.box_lines(m, b)
    assert lines[0].startswith("this Mac: 56 GB RAM, ceiling 42.8 GB, KV room 1.1 GB (flat")
    assert "decode arena" in lines[1] and "a cold token reads about 25.0 GB" in lines[1]
    assert lines[2].startswith("prefill ring 30.0 GB exceeds the arena")
    assert lines[-1].startswith("=> streams with --stream-experts")
    assert "cannot stream" in sp.box_lines(
        m, sp.box_plan(_model(total=250e9, expert=200e9), ram_bytes=ram, ws_bytes=ws))[-1]
    assert "page cache" in sp.box_lines(
        m, sp.box_plan(_model(total=241.5e9, expert=200e9), ram_bytes=ram, ws_bytes=ws))[-1]
    assert "streaming is optional" in sp.box_lines(
        m, sp.box_plan(_model(total=10e9, expert=8e9), ram_bytes=ram, ws_bytes=ws))[-1]
    assert sp.model_line(m).startswith("every-token weights 20.0 GB, routed experts 200.0 GB (4 layers, 64 experts, 8 per token), prefill ring 30.0 GB")
    assert sp.share_text(0.004) == " (under 1% of the experts)"
    assert sp.share_text(0.23) == " (23% of the experts)"
    d = sp.to_dict(m, b)
    json.dumps(d)
    assert d["every_token_bytes"] == 20e9 and d["box"]["verdict"] == "streams"
    assert d["box"]["room"]["priced"] is False
    assert sp.to_dict(m, None)["box"] is None


def _scan(path, tensors, kv=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        path=path, kv=kv or {"general.architecture": "llama"},
        tensors=[SimpleNamespace(name=n, nbytes=b, shape=(1, 1, 8))
                 for n, b in tensors])


def test_ring_is_the_runtime_formula():
    # Per-kind maxima across layers, over layers with all three kinds:
    # the same rule prefill_feeder.ring_bytes applies to the offsets.
    m = sp.model_plan([_scan("a", [
        ("blk.0.ffn_gate_exps.weight", 10), ("blk.0.ffn_up_exps.weight", 10),
        ("blk.0.ffn_down_exps.weight", 50),
        ("blk.1.ffn_gate_exps.weight", 40), ("blk.1.ffn_up_exps.weight", 40),
        ("blk.1.ffn_down_exps.weight", 10),
        ("blk.2.ffn_gate_exps.weight", 90),            # incomplete layer
        ("blk.0.attn_q.weight", 7)])])
    assert m.ring_bytes == 2 * (40 + 40 + 50)
    assert m.moe_layers == 3 and m.expert_bytes == 250
    assert m.every_token_bytes == 7 and m.n_experts == 8
