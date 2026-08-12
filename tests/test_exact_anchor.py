"""Exact-tier anchor: whole-prefix clone at the sibling divergence.

Store and lookup live in a gmlx-owned side LRU; the upstream exact LRU
is count-capped and every request writes its guard-column entry there,
so sibling fan-out churns out the early shared-prefix entry. The armed
prefill pauses twice: the anchor hook stores at the divergence and
hands the checkpoint column back to the stock guard, whose store and
checkpoint_done latch fire exactly as unarmed.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.apc import APCManager
from mlx_vlm.models.cache import CacheList, KVCache

import gmlx.cache_snapshot as cs
import gmlx.spec_engine as se
from gmlx.cache_snapshot import anchor_exact_lookup, anchor_exact_store


def make_kv_cache(p, layers=2, seed=0):
    out = []
    for i in range(layers):
        c = KVCache()
        if p > 0:
            k = mx.full((1, 2, p, 8), float(seed + i), dtype=mx.float16)
            v = mx.full((1, 2, p, 8), float(seed + i + 1), dtype=mx.float16)
            c.update_and_fetch(k, v)
        out.append(c)
    return out


IDS = list(range(500, 500 + 120))


# -- store/lookup roundtrip on the side LRU --

def test_anchor_roundtrip_and_decoupling():
    man = APCManager(num_blocks=8, block_size=16)
    src = make_kv_cache(64)
    assert anchor_exact_store(man, IDS[:64], src)
    # The stored entry is a clone: mutating the source afterwards must
    # not leak into what siblings restore.
    src[0].update_and_fetch(
        mx.zeros((1, 2, 8, 8), dtype=mx.float16),
        mx.zeros((1, 2, 8, 8), dtype=mx.float16))
    warm, p = anchor_exact_lookup(man, IDS)
    assert p == 64 and all(int(c.offset) == 64 for c in warm)
    # The warm result is a clone too: advancing it must not corrupt the
    # entry the next sibling gets.
    warm[0].update_and_fetch(
        mx.zeros((1, 2, 8, 8), dtype=mx.float16),
        mx.zeros((1, 2, 8, 8), dtype=mx.float16))
    warm2, p2 = anchor_exact_lookup(man, IDS)
    assert p2 == 64 and all(int(c.offset) == 64 for c in warm2)


def test_anchor_lookup_gates():
    man = APCManager(num_blocks=8, block_size=16)
    anchor_exact_store(man, IDS[:64], make_kv_cache(64))
    # Different chain, wrong extra_hash, equal-length query, min-prefix
    # at or past the anchor: all misses.
    assert anchor_exact_lookup(man, [1, 2, 3] + IDS[3:]) == (None, 0)
    assert anchor_exact_lookup(man, IDS, extra_hash=9) == (None, 0)
    assert anchor_exact_lookup(man, IDS[:64]) == (None, 0)
    assert anchor_exact_lookup(man, IDS, min_prefix_tokens=64) == (None, 0)


def test_anchor_longest_prefix_wins():
    man = APCManager(num_blocks=8, block_size=16)
    anchor_exact_store(man, IDS[:32], make_kv_cache(32))
    anchor_exact_store(man, IDS[:64], make_kv_cache(64))
    _, p = anchor_exact_lookup(man, IDS)
    assert p == 64


def test_anchor_entries_cap_lru(monkeypatch):
    monkeypatch.setattr(cs, "_ANCHOR_ENTRIES", 2)
    man = APCManager(num_blocks=8, block_size=16)
    a, b, c = ([i] + IDS for i in (1, 2, 3))
    anchor_exact_store(man, a[:64], make_kv_cache(64))
    anchor_exact_store(man, b[:64], make_kv_cache(64))
    _, p = anchor_exact_lookup(man, a)      # refresh a's LRU position
    assert p == 64
    anchor_exact_store(man, c[:64], make_kv_cache(64))
    assert anchor_exact_lookup(man, a)[1] == 64
    assert anchor_exact_lookup(man, b) == (None, 0)
    assert anchor_exact_lookup(man, c)[1] == 64


def test_anchor_byte_budget_newest_survives(monkeypatch):
    monkeypatch.setattr(cs, "_ANCHOR_BUDGET_BYTES", 1)
    man = APCManager(num_blocks=8, block_size=16)
    a, b = ([i] + IDS for i in (1, 2))
    anchor_exact_store(man, a[:64], make_kv_cache(64))
    anchor_exact_store(man, b[:64], make_kv_cache(64))
    assert anchor_exact_lookup(man, a) == (None, 0)
    assert anchor_exact_lookup(man, b)[1] == 64


def test_anchor_cleared_by_ckpt_reset():
    man = APCManager(num_blocks=8, block_size=16)
    anchor_exact_store(man, IDS[:64], make_kv_cache(64))
    cs.ckpt_reset(man)
    assert anchor_exact_lookup(man, IDS) == (None, 0)


# -- boundary: ungridded divergence, clamped to the stock guard --

def _stub_sys(monkeypatch, lcp):
    from gmlx import retire_key
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda ids: {"messages": ()})
    monkeypatch.setattr(retire_key, "system_prefix_lcp",
                        lambda ctx, ids: lcp)


def _bmeta(n=5000):
    return SimpleNamespace(), {"full_input_ids": list(range(n))}


def test_anchor_boundary_ungridded(monkeypatch):
    _stub_sys(monkeypatch, 2900)
    batch, meta = _bmeta()
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) == 2900


def test_anchor_boundary_clamps_to_guard(monkeypatch):
    _stub_sys(monkeypatch, 4500)
    batch, meta = _bmeta()
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) == 4000
    # Guard 0 (stock checkpoint disabled): the divergence stands alone.
    assert se._exact_anchor_boundary(batch, meta, 0, 0) == 4500


def test_anchor_boundary_floor_kill_restored(monkeypatch):
    _stub_sys(monkeypatch, 200)
    batch, meta = _bmeta()
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) is None
    monkeypatch.setenv("GMLX_APC_CKPT_SYS_MIN", "100")
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) == 200
    assert se._exact_anchor_boundary(batch, meta, 4000, 200) is None
    monkeypatch.setenv("GMLX_APC_CKPT_SYS", "0")
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) is None


def test_anchor_boundary_no_render_ctx(monkeypatch):
    from gmlx import retire_key
    monkeypatch.setattr(retire_key, "lookup_render_ctx", lambda ids: None)
    batch, meta = _bmeta()
    assert se._exact_anchor_boundary(batch, meta, 4000, 0) is None


# -- two-stop schedule: anchor store, then the untouched stock guard --

def _armed_exact_batch(man, guard, monkeypatch, lcp):
    _stub_sys(monkeypatch, lcp)
    monkeypatch.setenv("GMLX_APC_CKPT_SYS_MIN", "16")
    meta = {"full_input_ids": IDS, "prefix_len": 0, "extra_hash": 7,
            "checkpoint_len": guard}
    batch = SimpleNamespace(
        _apc_manager=man, _apc_meta=[meta], _apc_mode="exact",
        prompt_cache=None)
    batch._apc_prompt_cache_for_store = lambda idx: batch.prompt_cache
    se._exact_anchor_arm(batch, meta, guard, 0)
    return batch, meta


def test_anchor_two_stop_schedule(monkeypatch):
    from mlx_vlm.generate import ar

    man = APCManager(num_blocks=8, block_size=16)
    calls = []
    orig = man.store_exact_cache
    man.store_exact_cache = (
        lambda *a, **k: (calls.append(len(a[0])), orig(*a, **k))[1])
    batch, meta = _armed_exact_batch(man, guard=96, monkeypatch=monkeypatch,
                                     lcp=64)
    assert batch._kq_anchor_armed and meta["checkpoint_len"] == 64
    stock = ar.PromptProcessingBatch._store_apc_exact_checkpoints

    # Anchor stop: the hook stores, hands the column back to the guard,
    # and the stock body (running right after, as in the wrap) skips.
    batch.prompt_cache = make_kv_cache(64)
    batch._row_real_tokens_processed = lambda idx: 64
    se._exact_anchor_store(batch)
    stock(batch)
    assert meta["anchor_done"] and meta["checkpoint_len"] == 96
    assert not meta.get("checkpoint_done") and calls == []
    assert anchor_exact_lookup(man, IDS, extra_hash=7)[1] == 64

    # Guard stop: the hook is spent; the stock store fires and latches.
    batch.prompt_cache = make_kv_cache(96)
    batch._row_real_tokens_processed = lambda idx: 96
    se._exact_anchor_store(batch)
    stock(batch)
    assert calls == [96] and meta.get("checkpoint_done")


def test_anchor_at_guard_single_stop(monkeypatch):
    from mlx_vlm.generate import ar

    man = APCManager(num_blocks=8, block_size=16)
    calls = []
    orig = man.store_exact_cache
    man.store_exact_cache = (
        lambda *a, **k: (calls.append(len(a[0])), orig(*a, **k))[1])
    # Divergence past the guard clamps onto it: one pause, both stores.
    batch, meta = _armed_exact_batch(man, guard=96, monkeypatch=monkeypatch,
                                     lcp=110)
    assert meta["anchor_len"] == 96 and meta["checkpoint_len"] == 96
    batch.prompt_cache = make_kv_cache(96)
    batch._row_real_tokens_processed = lambda idx: 96
    se._exact_anchor_store(batch)
    ar.PromptProcessingBatch._store_apc_exact_checkpoints(batch)
    assert anchor_exact_lookup(man, IDS, extra_hash=7)[1] == 96
    assert calls == [96] and meta.get("checkpoint_done")


# -- stock-path init: lookup, in-place trim, no re-arm at the anchor --

def test_plain_anchor_init_restores_and_trims(monkeypatch):
    from gmlx.deepseek_v4_cache import PoolingCache

    se._bind_l1_view()
    man = APCManager(num_blocks=8, block_size=16)
    anchor_exact_store(man, IDS[:64], make_kv_cache(64))
    _stub_sys(monkeypatch, 64)
    monkeypatch.setenv("GMLX_APC_CKPT_SYS_MIN", "16")

    n = len(IDS)
    # A pooling stack resolves exact and is not ckpt-tier (the layout
    # probe rejects unknown cache classes) -- the ds4 shape.
    model = SimpleNamespace(
        _kq_apc_manager=man, _kq_apc_mode="exact",
        config=SimpleNamespace(),
        make_cache=lambda: [KVCache(),
                            CacheList(KVCache(), PoolingCache(4))])
    meta = {"full_input_ids": list(IDS), "prefix_len": 0, "extra_hash": 0,
            "checkpoint_len": 96}
    batch = SimpleNamespace(
        model=model, uids=["u1"], _right_pad_per_row=None,
        _apc_manager=man, _apc_mode="exact", _apc_meta=[meta],
        _input_ids=mx.array([IDS]), _inputs_embeds=mx.zeros((1, n, 4)),
        _prompt_kwargs={}, _prompt_length_aware_keys=[],
        prompt_cache=None)
    se._plain_anchor_init(batch)
    assert batch._processed_prompt_columns == 64
    assert batch._input_ids.shape[1] == n - 64
    assert all(int(c.offset) == 64 for c in batch.prompt_cache)
    # Restored exactly at the divergence: nothing below it to store, so
    # the anchor hook stays unarmed and the stock guard runs alone.
    assert not getattr(batch, "_kq_anchor_armed", False)
    assert meta["checkpoint_len"] == 96
