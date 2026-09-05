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

import gmlx.serve.bridge_vlm as serving  # noqa: E402
from gmlx.cache.apc_manager import GmlxAPCManager, build_apc_manager  # noqa: E402
from gmlx.serve.residency import _RuntimeProxy, _ResidencyPool, _active_entry  # noqa: E402

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

    import gmlx.serve.capacity as capacity
    import gmlx.gen.prefill_decay as prefill_decay

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(prefill_decay, "untracked_weight_bytes", lambda: 0.0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)


def test_autosize_extends_pool_to_budget_share(monkeypatch):
    monkeypatch.delenv("APC_NUM_BLOCKS", raising=False)
    _autosize_rig(monkeypatch, budget=100 * GB)
    m = GmlxAPCManager(num_blocks=2048, block_size=16)
    m.autosize(_fake_model())
    per_tok = 10 * 2 * 4 * 64 * 2.0  # layers x 2KV x kv_heads x hd x fp16
    want = min(int(100 * GB * 0.5 // (16 * per_tok)),
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
    import gmlx.cache.apc_manager as am
    import gmlx.serve.capacity as capacity
    import gmlx.commands.tool_preflight as tp

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
    # budget tokens ~ (114e9 * 0.5) / 90e3 = 633k; need = 96*633k/450k = 135
    assert bs == 256
    # tiny model: 16 suffices, keep stock default
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 8e9)
    assert am._auto_block_size(str(f)) is None
    # non-derivable header keeps default
    monkeypatch.setattr(tp, "_synth_config", lambda p: None)
    assert am._auto_block_size(str(f)) is None


def test_exact_tier_byte_budget(monkeypatch):
    import mlx.core as mx

    from mlx_vlm.apc import APCExactCacheEntry

    import gmlx.serve.capacity as capacity
    import gmlx.gen.prefill_decay as prefill_decay

    monkeypatch.delenv("APC_EXACT_CACHE_ENTRIES", raising=False)
    monkeypatch.delenv("APC_NUM_BLOCKS", raising=False)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 100 * GB)
    monkeypatch.setattr(prefill_decay, "untracked_weight_bytes", lambda: 0.0)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 0)
    m = GmlxAPCManager(num_blocks=8, block_size=16)
    m.autosize(_fake_model())
    assert m._exact_cache_max == 64

    class FakeKVClone:
        def __init__(self):
            self.keys = mx.zeros((1, 4, 256, 64), dtype=mx.float16)

    m._exact_budget_bytes = 3 * FakeKVClone().keys.nbytes  # fits 3 entries
    for i in range(5):
        m._exact_cache[i] = APCExactCacheEntry(
            token_ids=(i,), extra_hash=0,
            prompt_cache=[FakeKVClone()], last_used=float(i))
    m._trim_exact_to_budget()
    assert set(m._exact_cache) == {2, 3, 4}  # oldest two evicted

    # explicit env keeps stock count semantics: no budget stashed
    m2 = GmlxAPCManager(num_blocks=8, block_size=16)
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "2")
    m2.autosize(_fake_model())
    assert getattr(m2, "_exact_budget_bytes", None) is None


def test_exact_bytes_reaches_stats_snapshot():
    # Three silent gaps found on the DSv4-Flash cert, locked here:
    # the byte gauge must survive trim -> stats attr -> stats_snapshot
    # (upstream's APCStats.snapshot is a fixed whitelist that drops
    # dynamically-set attrs).
    import mlx.core as mx

    from mlx_vlm.apc import APCExactCacheEntry

    class FakeKVClone:
        def __init__(self):
            self.keys = mx.zeros((1, 4, 64, 64), dtype=mx.float16)

    m = GmlxAPCManager(num_blocks=8, block_size=16)
    m._exact_budget_bytes = 10 * FakeKVClone().keys.nbytes
    m._exact_cache[1] = APCExactCacheEntry(
        token_ids=(1,), extra_hash=0,
        prompt_cache=[FakeKVClone()], last_used=1.0)
    m._trim_exact_to_budget()
    snap = m.stats_snapshot()
    assert snap.get("exact_bytes") == FakeKVClone().keys.nbytes


def test_exact_bytes_counts_cachelist_wrapped_layers():
    # deepseek_v4 make_cache wraps each sparse layer's caches in a
    # CacheList; walking the wrapper's vars sees cache objects, not
    # arrays, so the gauge read ~0 on the DSv4-Flash cert. entry_bytes
    # must unwrap one CacheList level and descend tuple-packed pools.
    import mlx.core as mx

    from mlx_vlm.apc import APCExactCacheEntry

    class FakePool:
        def __init__(self):
            self._pbuf = (mx.zeros((1, 32, 8), dtype=mx.float16),
                          mx.zeros((1, 32, 8), dtype=mx.float16))

    class FakeList:
        def __init__(self):
            self.caches = [FakePool()]

    m = GmlxAPCManager(num_blocks=8, block_size=16)
    m._exact_budget_bytes = 10 ** 9
    m._exact_cache[1] = APCExactCacheEntry(
        token_ids=(1,), extra_hash=0,
        prompt_cache=[FakeList()], last_used=1.0)
    m._trim_exact_to_budget()
    expect = sum(a.nbytes for a in FakePool()._pbuf)
    assert m.stats.exact_bytes == expect


def test_inline_layer_major_store_enforces_budget():
    # store_kv_blocks' inline exact branch bypassed the budget wrappers
    # (count cap only); the trim must fire there too.
    import mlx.core as mx

    m = GmlxAPCManager(num_blocks=4, block_size=16)
    m._exact_cache_max = 64
    m._exact_budget_bytes = 1e12
    m._layer_major_memory_min_tokens = 32
    ks = [mx.zeros((1, 2, 160, 8)) for _ in range(2)]
    vs = [mx.zeros((1, 2, 160, 8)) for _ in range(2)]
    m.store_kv_blocks(list(range(160)), ks, vs)
    assert m.stats.exact_stores == 1
    one = m.stats.exact_bytes
    assert one > 0
    m._exact_budget_bytes = int(1.5 * one)  # fits one entry, not two
    for i in range(3):
        m.store_kv_blocks(list(range(1000 + i, 1160 + i)), ks, vs)
    assert len(m._exact_cache) == 1
    assert m.stats.exact_bytes <= m._exact_budget_bytes
    assert m.stats_snapshot().get("exact_bytes") == m.stats.exact_bytes


def test_store_stops_at_kernel_floor(monkeypatch):
    # A block store is one call between governed ticks; it must stop
    # wiring pages itself when the kernel is below the floor.
    import mlx.core as mx
    import gmlx.serve.kernel_vm as kv
    from gmlx.cache.apc_manager import GmlxAPCManager

    import gmlx.serve.governor as gov

    monkeypatch.delenv("GMLX_APC_PAGED", raising=False)
    monkeypatch.setattr(gov, "_ARMED_FLOOR", 8e9)   # as install_governor arms it
    man = GmlxAPCManager(num_blocks=16, block_size=16)
    ids = list(range(96))
    lk = [mx.random.normal((1, 2, 96, 8)).astype(mx.float16)]
    lv = [mx.random.normal((1, 2, 96, 8)).astype(mx.float16)]
    calls = []

    def scripted():
        calls.append(1)
        return 100e9 if len(calls) <= 2 else 1e9   # floor hit at block 3

    monkeypatch.setattr(kv, "reclaimable_bytes", scripted)
    blocks = man.store_kv_blocks(ids, lk, lv, extra_hash=0)
    man.release(blocks)
    assert len(blocks) == 2
    assert man.stats.rejects_by_reason.get("kernel_floor") == 1

    monkeypatch.setattr(gov, "_ARMED_FLOOR", 0.0)   # no governor installed
    man2 = GmlxAPCManager(num_blocks=16, block_size=16)
    blocks = man2.store_kv_blocks(ids, lk, lv, extra_hash=0)
    man2.release(blocks)
    assert len(blocks) == 6
    assert "kernel_floor" not in man2.stats.rejects_by_reason


def test_kvarn_salt_applied_from_the_stamped_model(monkeypatch):
    """The wire salt is read off the response generator's model, the
    object the load-time policy stamps. The runtime's model cache is a
    registry, not a dict, so it cannot be the only route to the model."""
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("GMLX_APC_ENABLED", raising=False)
    import gmlx.cache.kvarn_apc as kvarn_apc
    import gmlx.serve.kv_policy as skv
    from gmlx.cache.kvarn_apc import kvarn_entry_salt

    class Registry:  # mlx-vlm's ModelCacheRegistry shape: get(), no dict
        def __init__(self, d):
            self._d = d

        def get(self, key, default=None, *, kind=None):
            return self._d.get(key, default)

    pol = SimpleNamespace(single=SimpleNamespace(
        scheme="kvarn", bits=6, value_bits=None, tail_tokens=1024))
    model = SimpleNamespace(_gmlx_kv_policy=pol, _kq_apc_mode="exact")
    manager = SimpleNamespace(_exact_extra_salt=0, autosize=lambda m: None)
    proxy = _RuntimeProxy(SimpleNamespace(metrics=None))

    def fake_stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.model_cache = Registry({"model": model})
        proxy.response_generator = SimpleNamespace(apc_manager=None, model=model)
        proxy.apc_manager = None
        serving.publish_built_apc_manager(manager)

    monkeypatch.setattr(skv, "resolve_for_load", lambda rg, mid: pol)
    monkeypatch.setattr(kvarn_apc, "kvarn_model_converts", lambda m: True)
    pool = _ResidencyPool(
        proxy, fake_stock_get, lambda: True, 200 * GB, (),
        footprint_fn=lambda p: GB, in_flight_fn=lambda: 0)
    e = _acq(pool, "/m/a.gguf", env={"APC_ENABLED": "1"})
    assert e.apc_manager is manager
    assert manager._exact_extra_salt == kvarn_entry_salt(model) != 0
