"""The streaming memory budget: the KV room the arena leaves under the
governor ceiling, and the governor-shifted headroom the regrow gate reads."""

from types import SimpleNamespace

import pytest

import gmlx.serve.capacity as cap
import gmlx.stream.budget as budget

GB = 1e9

CFG = {
    "num_hidden_layers": 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "max_position_embeddings": 65536,
}
BPT = 2 * 8 * 64 * 2 * 10  # 20480 B/token across layers


@pytest.fixture
def header(monkeypatch, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    import gmlx.commands.tool_preflight as tp
    import gmlx.gen.prefill_decay as pd
    import gmlx.serve.memory as sm

    for k in ("GMLX_STREAM_KV_CTX", "GMLX_STREAM_KV_WIDTH",
              "GMLX_DECODE_KV_RESERVE_GB", "GMLX_GOV_MARGIN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
    monkeypatch.setattr(tp, "_synth_config", lambda p: dict(CFG))
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 100.0 * GB)
    monkeypatch.setattr(sm, "admit_reserve_bytes", lambda ws, gen=None: 1.0 * GB)
    monkeypatch.setattr(pd, "_cap_bytes", lambda: 3.0 * GB)
    return str(f)


def test_room_prices_ctx_width_transient_reserve(header, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_KV_CTX", "4096")
    monkeypatch.setenv("GMLX_STREAM_KV_WIDTH", "2")
    room = budget.kv_room_bytes(header)
    assert room.priced
    assert (room.depth, room.width) == (4096, 2)
    assert room.kv_bytes == BPT * 4096 * 2
    assert room.transient_bytes == int(3.0 * GB)
    assert room.reserve_bytes == int(1.0 * GB)
    assert room.bytes == room.kv_bytes + room.transient_bytes + room.reserve_bytes


def test_room_depth_clamped_to_trained_context(header, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_KV_CTX", str(1 << 20))
    room = budget.kv_room_bytes(header)
    assert room.depth == 65536
    assert room.kv_bytes == BPT * 65536


def test_room_default_depth(header):
    room = budget.kv_room_bytes(header)
    assert room.depth == 32768 and room.width == 1


def test_flat_room_on_env_override(header, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "3")
    room = budget.kv_room_bytes(header)
    assert not room.priced
    assert room.bytes == 3 << 30


def test_flat_room_when_header_unpriceable(header, monkeypatch):
    import gmlx.commands.tool_preflight as tp

    def _boom(p):
        raise OSError("no header")

    monkeypatch.setattr(tp, "_synth_config", _boom)
    room = budget.kv_room_bytes(header)
    assert not room.priced
    assert room.bytes == 8 << 30
    assert budget.kv_room_bytes(None).bytes == 8 << 30


def test_ceiling_and_governor_headroom(monkeypatch):
    import mlx.core as mx
    import gmlx.gen.prefill_decay as pd

    monkeypatch.delenv("GMLX_GOV_MARGIN", raising=False)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: 100.0 * GB)
    monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 128.0 * GB})
    # 0.95 x ws binds below RAM minus the 10% reserve (115.2 GB).
    assert budget.ceiling_bytes() == pytest.approx(95.0 * GB)
    monkeypatch.setattr(pd, "headroom_bytes", lambda: 30.0 * GB)
    assert budget.governor_headroom_bytes() == pytest.approx(25.0 * GB)
    monkeypatch.setattr(cap, "working_set_bytes", lambda: None)
    assert budget.ceiling_bytes() is None
    assert budget.governor_headroom_bytes() is None


def test_kv_room_dataclass_is_frozen():
    room = budget.KvRoom(1, 2, 3, 4, 5, 6, priced=True)
    with pytest.raises(Exception):
        room.bytes = 7
    assert isinstance(SimpleNamespace(**room.__dict__).bytes, int)


def test_legacy_room_ignores_a_malformed_reserve(monkeypatch):
    from gmlx.stream.budget import legacy_room_bytes

    for bad in ("abc", "1e999", "nan"):
        monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", bad)
        assert legacy_room_bytes() == 8 << 30
    monkeypatch.setenv("GMLX_DECODE_KV_RESERVE_GB", "2")
    assert legacy_room_bytes() == 2 << 30


def test_transient_bytes_follows_the_decay_cap(monkeypatch):
    from gmlx.stream.budget import transient_bytes

    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB", raising=False)
    assert transient_bytes(100e9) == 5e9
    assert transient_bytes(10e9) == 2e9
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "3")
    assert transient_bytes(100e9) == 3e9
