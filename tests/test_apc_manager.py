#!/usr/bin/env python3
"""gmlx-owned APC manager: build gate + from_env knob mirror, the bridge
channel, and the residency wiring that leaves apc.from_env unpatched and
unused (pinned APC_ENABLED=0 around the stock load; the effective enablement
crosses as GMLX_APC_ENABLED). CPU-only, fake stock loads, no model files."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

apc = pytest.importorskip("mlx_vlm.apc")

from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx.apc_manager import GmlxAPCManager, build_apc_manager  # noqa: E402
from gmlx.residency import _RuntimeProxy, _ResidencyPool, _active_entry  # noqa: E402

GB = 1024**3


# -- build gate + knobs --

def test_build_gate_absent_or_off(monkeypatch):
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    assert build_apc_manager() is None
    monkeypatch.setenv("GMLX_APC_ENABLED", "0")
    assert build_apc_manager() is None


def test_build_reads_the_env_from_env_reads(monkeypatch):
    monkeypatch.setenv("GMLX_APC_ENABLED", "1")
    monkeypatch.setenv("APC_BLOCK_SIZE", "32")
    monkeypatch.setenv("APC_NUM_BLOCKS", "8")
    monkeypatch.delenv("APC_DISK_PATH", raising=False)
    m = build_apc_manager(model_namespace="/m/a.gguf")
    assert isinstance(m, GmlxAPCManager)
    assert m.block_size == 32
    assert len(m.pool) == 8
    assert m.disk is None


def test_build_defaults_match_stock(monkeypatch):
    monkeypatch.setenv("GMLX_APC_ENABLED", "1")
    for k in ("APC_BLOCK_SIZE", "APC_NUM_BLOCKS", "APC_DISK_PATH"):
        monkeypatch.delenv(k, raising=False)
    m = build_apc_manager()
    assert m.block_size == apc.DEFAULT_BLOCK_SIZE
    assert len(m.pool) == apc.DEFAULT_NUM_BLOCKS


def test_from_env_is_dead_under_the_pin(monkeypatch):
    monkeypatch.setenv("APC_ENABLED", "0")
    assert apc.from_env(model_namespace="/m/a.gguf") is None


# -- bridge channel --

def test_channel_pop_returns_and_clears():
    serving.publish_built_apc_manager("MGR")
    assert serving.pop_built_apc_manager() == "MGR"
    assert serving.pop_built_apc_manager() is None


# -- residency wiring --

def make_pool(*, enabled_windows=True, fail=False):
    """Pool whose fake stock load plays the bridge's part: records the APC env
    it sees and publishes a manager on the channel when the gmlx gate is on."""
    proxy = _RuntimeProxy(SimpleNamespace(metrics=None))
    seen = []
    built = []

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        seen.append({k: os.environ.get(k)
                     for k in ("APC_ENABLED", "GMLX_APC_ENABLED")})
        proxy.model_cache = {"model_path": model_path, "apc_manager": None}
        proxy.response_generator = SimpleNamespace(apc_manager=None)
        proxy.apc_manager = None                    # stock: from_env returned None
        if os.environ.get("GMLX_APC_ENABLED") == "1":
            mgr = f"GMLX-APC:{model_path}"
            built.append(mgr)
            serving.publish_built_apc_manager(mgr)
        if fail:
            raise RuntimeError("load failed")

    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, 200 * GB, (),
        footprint_fn=lambda p: GB, in_flight_fn=lambda: 0)
    return proxy, pool, seen, built


def _acq(pool, path, **kw):
    e = pool.acquire(path, None, "auto", **kw)
    _active_entry.set(e)
    return e


def test_manager_wired_into_entry_cache_and_rg(monkeypatch):
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, seen, built = make_pool()
    e = _acq(pool, "/m/a.gguf", env={"APC_ENABLED": "1"})
    assert built == ["GMLX-APC:/m/a.gguf"]
    assert e.apc_manager == built[0]                       # proxy target
    assert e.model_cache["apc_manager"] == built[0]        # unload/reuse dict
    assert e.response_generator.apc_manager == built[0]    # lazy BatchGenerator read
    assert serving.pop_built_apc_manager() is None         # channel drained
    assert os.environ.get("APC_ENABLED") is None           # window restored
    assert os.environ.get("GMLX_APC_ENABLED") is None


def test_ambient_env_enables_without_config(monkeypatch):
    # APC_ENABLED=1 in the shell (no config cache block) still enables the
    # cache - the documented plain-env flow the disk e2e exercises.
    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, seen, built = make_pool()
    e = _acq(pool, "/m/a.gguf", env=None)
    assert seen[-1] == {"APC_ENABLED": "0", "GMLX_APC_ENABLED": "1"}
    assert e.apc_manager == built[0]
    assert os.environ["APC_ENABLED"] == "1"                # ambient value restored


def test_config_off_beats_ambient_on(monkeypatch):
    # An explicit cache.enabled: false window pins the gate off even when the
    # shell exports APC_ENABLED=1 - the stock window-over-ambient precedence.
    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, seen, built = make_pool()
    e = _acq(pool, "/m/a.gguf", env={"APC_ENABLED": "0"})
    assert seen[-1]["GMLX_APC_ENABLED"] == "0"
    assert built == []
    assert e.apc_manager is None


def test_disabled_build_wires_nothing(monkeypatch):
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, _seen, built = make_pool()
    e = _acq(pool, "/m/a.gguf", env=None)
    assert built == []
    assert e.apc_manager is None
    assert e.response_generator.apc_manager is None


def test_failed_load_drains_the_channel(monkeypatch):
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    _proxy, pool, _seen, built = make_pool(fail=True)
    with pytest.raises(RuntimeError):
        _acq(pool, "/m/a.gguf", env={"APC_ENABLED": "1"})
    assert built                                            # bridge did build
    assert serving.pop_built_apc_manager() is None          # but never leaks
    assert os.environ.get("APC_ENABLED") is None
    assert os.environ.get("GMLX_APC_ENABLED") is None


# -- pool autosize --

def _fake_model(layers=10, n_kv=4, head_dim=64):
    cfg = SimpleNamespace(num_hidden_layers=layers, num_attention_heads=8,
                          num_key_value_heads=n_kv, head_dim=head_dim)
    return SimpleNamespace(config=cfg)


def _autosize_rig(monkeypatch, budget):
    import mlx.core as mx

    from gmlx import capacity, prefill_decay

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(prefill_decay, "untracked_weight_bytes", lambda: 0.0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)


def test_autosize_extends_pool_to_budget_share(monkeypatch):
    monkeypatch.delenv("APC_NUM_BLOCKS", raising=False)
    _autosize_rig(monkeypatch, budget=100 * GB)
    m = GmlxAPCManager(num_blocks=2048, block_size=16)
    m.autosize(_fake_model())
    per_tok = 10 * 2 * 4 * 64 * 2.0  # layers x 2KV x kv_heads x hd x fp16
    want = min(int(100 * GB * 0.2 // (16 * per_tok)),
               450000 // (2 * 10))
    assert m.num_blocks == want > 2048
    assert len(m.pool) == want
    assert m.pool[-1].block_id == want - 1


def test_autosize_env_and_floor_win(monkeypatch):
    _autosize_rig(monkeypatch, budget=100 * GB)
    m = GmlxAPCManager(num_blocks=2048, block_size=16)
    monkeypatch.setenv("APC_NUM_BLOCKS", "2048")
    m.autosize(_fake_model())
    assert m.num_blocks == 2048
    monkeypatch.delenv("APC_NUM_BLOCKS", raising=False)
    _autosize_rig(monkeypatch, budget=1 * GB)  # target below current cap
    m.autosize(_fake_model(layers=100))
    assert m.num_blocks == 2048  # never shrinks
    m.autosize(None)
    assert m.num_blocks == 2048


def test_auto_block_size_covers_budget(monkeypatch, tmp_path):
    from gmlx import apc_manager as am
    from gmlx import capacity, tool_preflight as tp

    f = tmp_path / "m.gguf"
    f.write_bytes(b"\0" * 1024)
    monkeypatch.setattr(tp, "_shards", lambda p: [str(f)])
    monkeypatch.setattr(tp, "_synth_config", lambda p: {"ok": True})
    layers = 48
    monkeypatch.setattr(tp, "_kv_costs",
                        lambda cfg: [(None, 90e3 / layers)] * layers)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 114e9)
    monkeypatch.delenv("APC_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("APC_MAX_POOL_TENSORS", raising=False)
    bs = am._auto_block_size(str(f))
    # budget tokens ~ (114e9 * 0.2) / 90e3 = 253k; need = 96*253k/450k = 54
    assert bs == 64
    # tiny model: 16 suffices, keep stock default
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 8e9)
    assert am._auto_block_size(str(f)) is None
    # non-derivable header keeps default
    monkeypatch.setattr(tp, "_synth_config", lambda p: None)
    assert am._auto_block_size(str(f)) is None
