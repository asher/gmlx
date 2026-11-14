"""Checkpoint records take their room from the pool's share.

A static records-only budget beside a pool sized from free room dropped a
conversation's deepest record at width 4 while the pool sat mostly idle:
the conversation looked up last was the coldest one, so the budget took
its record as the other three stored theirs. Records and the pool's
committed blocks now share one figure; a chain of blocks is useless
without its record, so blocks no record holds go first, and a record goes
with the blocks only it held. Beside it: a retirement no lookup can use
stores nothing, and the request's N-1 replay stands in for the p=N
post-prefill store.
"""

import logging
from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.cache.apc_manager as am
import gmlx.cache.retire_key as retire_key
import gmlx.cache.snapshot as cs
import gmlx.spec.ckpt as ckpt
import gmlx.spec.engine as engine
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.snapshot import ckpt_lookup, ckpt_store
from test_ckpt_plain_path import _plain_batch
from test_ckpt_tier import (
    D,
    H,
    LAYOUT,
    _qsa_hybrid_cache,
    _trim_kv,
    make_hybrid_cache,
)

pytestmark = pytest.mark.host_gb(16)

# A pool block of make_hybrid_cache: two KV layers, K and V of
# (1, 2, 16, 8) float32.
_BLOCK_BYTES = 2 * 2 * 2 * 16 * 8 * 4


def _ids(chain, p):
    return list(range(1000 * (chain + 1), 1000 * (chain + 1) + p))


def _records_bytes(man):
    return sum(r.nbytes for r in cs._ckpt_records(man).values())


def _block_bytes(man):
    return sum(int(b.resident_bytes()) for b in man.pool)


def _idle_bytes(man):
    return sum(int(b.resident_bytes()) for b in man.pool if b.ref_cnt == 0)


def _big(p, seed):
    """make_hybrid_cache with each recurrent layer's matrix state at
    512 KiB, so a record holds about 1.5 MiB and the MiB counter moves."""
    caches = make_hybrid_cache(p, seed=seed)
    mx.random.seed(seed)
    for c, kind in zip(caches, LAYOUT):
        if kind == "arr":
            c.cache = [c.cache[0], mx.random.normal((1, H, 256, 256))]
    return caches


def _store(man, chain, p=64):
    return ckpt_store(man, _ids(chain, p), _big(p, chain), extra_hash=chain)


def _chains(man):
    return [r.extra_hash for r in cs._ckpt_records(man).values()]


def _budget_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("APC ckpt budget")]


def test_round_robin_keeps_the_coldest_conversations_deep_record(monkeypatch):
    """Four conversations, each an anchor and a deep record. Three look up
    and store a deeper record; the fourth, looked up last, still restores
    its deep record. The records-only budget stands at what the four held
    before the stores, the proportion of the width 4 run (4096 MB against
    a 5737 MB peak); the pool's share holds them all."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(4):
        assert ckpt_store(man, _ids(c, 32), make_hybrid_cache(32, seed=c),
                          extra_hash=c)
        assert ckpt_store(man, _ids(c, 112),
                          make_hybrid_cache(112, seed=10 + c), extra_hash=c)
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", _records_bytes(man) + 1)
    # The share autosize stashes: the pool's cap on this manager.
    man._kq_ckpt_share_bytes = 64 * _BLOCK_BYTES
    for c in range(3):
        _, got = ckpt_lookup(man, _ids(c, 112) + [7], extra_hash=c)
        assert got == 112
    for c in range(3):
        assert ckpt_store(man, _ids(c, 128),
                          make_hybrid_cache(128, seed=20 + c), extra_hash=c)
    _, got = ckpt_lookup(man, _ids(3, 112) + [7], extra_hash=3)
    assert got == 112


def _round_robin(man):
    """Four conversations, each an anchor and a deep record; three are
    served in turn and each stores a deeper record."""
    for c in range(4):
        assert ckpt_store(man, _ids(c, 32), make_hybrid_cache(32, seed=c),
                          extra_hash=c)
        assert ckpt_store(man, _ids(c, 112),
                          make_hybrid_cache(112, seed=10 + c), extra_hash=c)
    for c in range(3):
        _, got = ckpt_lookup(man, _ids(c, 112) + [7], extra_hash=c)
        assert got == 112
        assert ckpt_store(man, _ids(c, 128),
                          make_hybrid_cache(128, seed=20 + c), extra_hash=c)


def test_room_keeps_each_conversations_newest_record():
    """The room binds on the last store. The conversation served longest
    ago keeps the record its next turn restores; least recently used
    would drop exactly that record, since its conversation is due next.
    A record its own conversation superseded goes instead."""
    free = GmlxAPCManager(num_blocks=64, block_size=16)
    _round_robin(free)
    room = _records_bytes(free) + _block_bytes(free)
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    man._kq_ckpt_share_bytes = room - 1
    _round_robin(man)
    _, got = ckpt_lookup(man, _ids(3, 112) + [7], extra_hash=3)
    assert got == 112
    kept = {(r.extra_hash, r.p) for r in cs._ckpt_records(man).values()}
    assert (0, 112) not in kept
    assert man.stats_snapshot()["ckpt_budget_evictions"] == 1


def test_over_the_share_idle_blocks_go_first_then_records_with_theirs(
        caplog):
    """Past the share, blocks no record holds are freed first and no
    record goes while they cover the excess. Past that, the least
    recently used record goes with the blocks only it held: no block is
    left committed without a record, both counters move, and one line
    names the victim."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(3):
        assert _store(man, c)
    # A record another path dropped leaves its blocks idle: hashed,
    # unreferenced, at the tail of the free queue.
    assert _store(man, 9)
    idx = cs._ckpt_records(man)
    cs._release_record(man, idx.pop(next(
        k for k, r in idx.items() if r.extra_hash == 9)))
    idle = _idle_bytes(man)
    assert idle > 0
    # What one more conversation adds, its record and its blocks.
    twin = GmlxAPCManager(num_blocks=64, block_size=16)
    assert _store(twin, 3)
    add = _records_bytes(twin) + _block_bytes(twin)

    total = _records_bytes(man) + _block_bytes(man)
    share = total + add - idle // 2
    man._kq_ckpt_share_bytes = share
    with caplog.at_level(logging.INFO, logger="gmlx.cache.snapshot"):
        assert _store(man, 3)
    assert _chains(man) == [0, 1, 2, 3]
    assert _idle_bytes(man) < idle
    assert _records_bytes(man) + _block_bytes(man) <= share
    st = man.stats_snapshot()
    assert st["ckpt_budget_evictions"] == 0
    assert st["ckpt_budget_evicted_mb"] == 0
    assert _budget_lines(caplog) == []

    left = _idle_bytes(man)
    share = _records_bytes(man) + _block_bytes(man) + add - left - add // 2
    man._kq_ckpt_share_bytes = share
    with caplog.at_level(logging.INFO, logger="gmlx.cache.snapshot"):
        assert _store(man, 4)
    assert _chains(man) == [1, 2, 3, 4]
    assert _idle_bytes(man) == 0
    assert all(b.ref_cnt > 0 for b in man.pool if b.resident_bytes())
    assert _records_bytes(man) + _block_bytes(man) <= share
    _, got = ckpt_lookup(man, _ids(0, 64) + [7], extra_hash=0)
    assert got == 0
    st = man.stats_snapshot()
    assert st["ckpt_budget_evictions"] == 1
    assert st["ckpt_budget_evicted_mb"] >= 1
    lines = _budget_lines(caplog)
    assert len(lines) == 1
    assert "dropped 1 record(s)" in lines[0]
    assert "p=64 anchor" in lines[0]
    assert "records and pool blocks" in lines[0]


def test_the_override_is_a_records_only_budget_and_wins(monkeypatch):
    """GMLX_APC_CKPT_BUDGET_MB set: records alone count against it, and
    a share the blocks already fill drops nothing."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(2):
        assert _store(man, c)
    one = _records_bytes(man) // 2
    man._kq_ckpt_share_bytes = 1
    monkeypatch.setattr(cs, "_CKPT_BUDGET_SET", True)
    monkeypatch.setattr(cs, "_CKPT_BUDGET_BYTES", 3 * one + 1)
    assert cs._record_room(man) == (3 * one + 1, False)
    assert _store(man, 2)
    assert _chains(man) == [0, 1, 2]
    assert man.stats_snapshot()["ckpt_budget_evictions"] == 0
    assert _store(man, 3)
    assert _chains(man) == [1, 2, 3]
    assert man.stats_snapshot()["ckpt_budget_evictions"] == 1


def test_autosize_stashes_the_pools_share(monkeypatch):
    """autosize stashes the share it sizes the pool from, the pool's own
    bytes when APC_NUM_BLOCKS sets it, and never less than the pool's
    cap. A manager autosize never sized keeps the records-only default."""
    from test_apc_manager import GB, _autosize_rig, _fake_model

    for k in ("APC_NUM_BLOCKS", "APC_EXACT_CACHE_ENTRIES",
              "APC_MEMORY_MAX_GB"):
        monkeypatch.delenv(k, raising=False)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    assert cs._record_room(m) == (cs._CKPT_BUDGET_BYTES, False)

    _autosize_rig(monkeypatch, budget=100 * GB)
    m.autosize(_fake_model())
    pool_cap = m.num_blocks * m.block_size * m._pool_per_token_bytes
    share = 100 * GB * am._POOL_BUDGET_FRACTION
    assert m._kq_ckpt_share_bytes == int(max(share, pool_cap))
    assert share <= m._kq_ckpt_share_bytes < share + 16 * m._pool_per_token_bytes
    assert cs._record_room(m) == (m._kq_ckpt_share_bytes, True)

    # A pool already past the share: the share is the pool's cap.
    _autosize_rig(monkeypatch, budget=8 << 20)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    m.autosize(_fake_model())
    assert m._kq_ckpt_share_bytes == int(64 * 16 * m._pool_per_token_bytes)

    monkeypatch.setenv("APC_NUM_BLOCKS", "64")
    _autosize_rig(monkeypatch, budget=100 * GB)
    m = GmlxAPCManager(num_blocks=64, block_size=16)
    m.autosize(_fake_model())
    assert m.num_blocks == 64
    assert m._kq_ckpt_share_bytes == int(64 * 16 * m._pool_per_token_bytes)


def test_a_retirement_no_lookup_can_use_stores_nothing():
    """The arming recorded the layout the request's lookup signed with. A
    retirement whose cache tags another layout (QSA layers out of a merge
    without their indexer keys tag "kv") stores nothing and counts a
    retire_layout decline; one that matches stores."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = _ids(0, 52)
    cache = make_hybrid_cache(52, seed=4)
    live = cs.ckpt_layout(cache, 16)
    signed = tuple("qsa:4" if t == "kv" else t for t in live)
    assert cs.retirement_store(man, "ckpt", ids, cache, layout=signed) == 0
    assert len(cs._ckpt_records(man)) == 0
    assert man.stats_snapshot()["ckpt_declines"] == {"retire_layout": 1}
    assert cs.retirement_store(man, "ckpt", ids, cache, layout=live) == 52
    assert [r.p for r in cs._ckpt_records(man).values()] == [52]
    # No signature recorded (an unsupported live cache at arming): no check.
    assert cs.retirement_store(man, "ckpt", _ids(1, 52), cache,
                               layout=None) == 52


@pytest.mark.parametrize("max_len,refused", [(70, 2), (None, 3)],
                         ids=["diverged", "whole"])
def test_a_retirement_counts_its_refused_layouts_once(caplog, max_len,
                                                      refused):
    """A QSA row that took decode snapshots alone, then joined a batch,
    leaves as plain KV, and so does every snapshot assembled for it. Its
    retirement counts one decline and logs one line however many
    candidates it refused, and says nothing of missing snapshots."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    cache = make_hybrid_cache(72, seed=5)
    signed = tuple("qsa:4" if t == "kv" else t
                   for t in cs.ckpt_layout(cache, 16))
    snaps = [(p, [c for c in make_hybrid_cache(p, seed=p)
                  if not hasattr(c, "keys")]) for p in (48, 64)]
    with caplog.at_level(logging.INFO, logger="gmlx.cache.snapshot"):
        assert cs.retirement_store(man, "ckpt", _ids(0, 72), cache,
                                   max_len=max_len, decode_snaps=snaps,
                                   layout=signed) == 0
    lines = [r.getMessage() for r in caplog.records
             if "matches no lookup" in r.getMessage()]
    assert len(lines) == 1 and f"{refused} candidate(s)" in lines[0]
    assert not [r for r in caplog.records
                if "no decode snapshot" in r.getMessage()]
    assert man.stats_snapshot()["ckpt_declines"] == {"retire_layout": 1}
    assert len(cs._ckpt_records(man)) == 0


def test_the_replay_stands_in_for_the_post_prefill_store():
    """With the request's N-1 replay landed, the p=N store is skipped. On
    a recurrent layout the replay becomes a boundary and serves both the
    identical resend and an extension, as the p=N record would; with no
    replay the p=N store runs."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    n = 52
    ids = _ids(0, n)
    cache = make_hybrid_cache(n, seed=3)
    assert not cs.ckpt_replay_stands_in(man, ids, cache)
    assert ckpt_store(man, ids[:32], _trim_kv(cache, 32))
    assert ckpt_store(man, ids[:n - 1], _trim_kv(cache, n - 1),
                      kind="replay")
    # The replay gate leaves an extension the anchor below.
    _, got = ckpt_lookup(man, ids + [7])
    assert got == 32
    assert cs.ckpt_replay_stands_in(man, ids, cache)
    recs = [(r.p, r.kind) for r in cs._ckpt_records(man).values()]
    assert recs == [(32, "anchor"), (n - 1, "boundary")]
    _, got = ckpt_lookup(man, ids)
    assert got == n - 1
    _, got = ckpt_lookup(man, ids + [7, 8])
    assert got == n - 1
    # Another prompt on the chain has no replay at its N-1.
    assert not cs.ckpt_replay_stands_in(man, ids + [7, 8], cache)


def _plain_armed(man, ids, make, seed):
    """A stock prompt batch armed by _plain_ckpt_init over a live cache
    from ``make``. The stash it leaves on the cache is the request's."""
    engine._bind_l1_view()
    b = _plain_batch(man, ids)
    b.prompt_cache = make(len(ids), seed=seed)
    ckpt._plain_ckpt_init(b)
    return b


def _mtp_armed(man, ids, make, seed):
    """An MTP prompt batch armed by _mtp_prefill_init, and the retirement
    context the B=1 rounds pop off its cache."""
    from gmlx.spec import mtp_prefill
    from gmlx.spec.speculative import _pop_retire_ctx

    engine._bind_l1_view()
    model = SimpleNamespace(_kq_apc_manager=man, _kq_apc_mode="exact",
                            config=SimpleNamespace(),
                            make_cache=lambda: make(0))
    cache = make(len(ids), seed=seed)
    batch = SimpleNamespace(
        model=model, _input_ids=mx.array([ids]),
        _inputs_embeds=mx.zeros((1, len(ids), 4)), prompt_cache=cache,
        _prompt_kwargs={}, prefill_step_size=16,
        _prompt_length_aware_keys=())
    mtp_prefill._mtp_prefill_init(batch)
    return model, cache, _pop_retire_ctx(cache)


def _replay(man, stash, cache):
    """The N-1 replay record the prefill stores for the stash's prompt."""
    ids = stash["full_ids"]
    assert ckpt_store(man, ids[:-1], _trim_kv(cache, len(ids) - 1),
                      extra_hash=stash["extra_hash"], kind="replay")


def _plain_post_prefill(man, ids, seed, replay):
    from gmlx.spec.mtp_prefill import _ckpt_plain_post_prefill

    b = _plain_armed(man, ids, make_hybrid_cache, seed)
    if replay:
        _replay(man, b.prompt_cache[0]._kq_apc_retire, b.prompt_cache)
    _ckpt_plain_post_prefill(b)
    return b._apc_meta[0]


def _mtp_post_prefill(man, ids, seed, replay):
    from gmlx.spec.speculative import _ckpt_post_prefill

    model, cache, ctx = _mtp_armed(man, ids, make_hybrid_cache, seed)
    if replay:
        _replay(man, ctx, cache)
    _ckpt_post_prefill(model, cache, ctx)
    return ctx["apc_meta"]


@pytest.mark.parametrize("post", [_plain_post_prefill, _mtp_post_prefill],
                         ids=["plain", "mtp"])
def test_the_post_prefill_skips_the_store_its_replay_covers(post):
    """The p=N store of the stock path and of the B=1 MTP rounds, each
    driven from the stash its arming left: it runs for a prompt with no
    replay and is skipped for one whose replay landed, which stands in."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    n = 52
    a, b = _ids(1, n), _ids(2, n)
    assert post(man, a, 1, replay=False)["ckpt_stored_boundaries"] == [n]
    assert post(man, b, 2, replay=True)["ckpt_stored_boundaries"] == []
    # The first restorable boundary on a chain is its anchor.
    recs = {r.ids[0]: (r.p, r.kind) for r in cs._ckpt_records(man).values()}
    assert recs == {a[0]: (n, "anchor"), b[0]: (n - 1, "anchor")}
    assert man.stats_snapshot()["ckpt_stores"] == 2


def _decode_qsa(cache, tok):
    """One decode step on a QSA hybrid row: a key, value and indexer key
    row on each QSA layer, new recurrent states on the rest."""
    mx.random.seed(tok)
    for c in cache:
        if hasattr(c, "update_and_fetch_qsa"):
            c.update_and_fetch_qsa(mx.random.normal((1, H, 1, D)),
                                   mx.random.normal((1, H, 1, D)),
                                   mx.random.normal((1, 1, D)))
        else:
            c.cache = [mx.random.normal(a.shape) for a in c.cache]


def _render(monkeypatch, final):
    """A server render context at arming. The next-turn LCP does not
    diverge while decoding and is ``final(seq)`` at retirement."""
    monkeypatch.setenv("GMLX_APC_DECODE_CKPT", "16")
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda ids: {"stub": True})
    monkeypatch.setattr(
        retire_key, "next_turn_lcp",
        lambda ctx, seq, gen, partial=False:
            len(seq) if partial else final(seq))


def _plain_turn(man, ids, steps):
    """Arm, decode and retire a stock-path row: the step tick takes the
    decode snapshots and the batch filter retires through _plain_retire."""
    from gmlx._vendor.mlx_vlm.generate.ar import GenerationBatch

    ckpt._install_plain_ckpt_decode()
    b = _plain_armed(man, ids, _qsa_hybrid_cache, ids[0])
    gb = GenerationBatch.empty(model=None, sampler=None, stop_criteria=None)
    gb.uids = [ids[0]]
    gb.prompt_cache = b.prompt_cache
    for tok in steps:
        _decode_qsa(gb.prompt_cache, tok)
        ckpt._plain_step_tick(gb, ([tok], None, None, None))
    stash = gb._kq_apc_retire_rows[ids[0]]
    gb.filter([])
    return stash


def _mtp_turn(man, ids, steps):
    """The same through the B=1 MTP rounds: their decode tick, then
    _retire_b1."""
    from gmlx.cache.snapshot import decode_ckpt_tick
    from gmlx.spec.speculative import _retire_b1

    model, cache, ctx = _mtp_armed(man, ids, _qsa_hybrid_cache, ids[0])
    gen = []
    for tok in steps:
        _decode_qsa(cache, tok)
        gen.append(tok)
        decode_ckpt_tick(ctx, cache, gen)
    _retire_b1(model, cache, gen, ctx)
    return ctx


@pytest.mark.parametrize("turn", [_plain_turn, _mtp_turn],
                         ids=["plain", "mtp"])
def test_a_retirement_checks_the_layout_its_arming_signed(monkeypatch, turn):
    """Driven from the arming: it signs the live QSA layout, a decode
    snapshot lands at 48, and the row retires at 56. The whole sequence
    stores. A render that diverges at 52 falls to the snapshot, whose QSA
    layers come back as plain KV: one retire_layout decline, no record."""
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    steps = range(9000, 9024)
    _render(monkeypatch, len)
    stash = turn(man, _ids(0, 32), steps)
    assert [p for p, _ in stash["snaps"]] == [48]
    assert [r.p for r in cs._ckpt_records(man).values()] == [56]
    _render(monkeypatch, lambda seq: 52)
    stash = turn(man, _ids(1, 32), steps)
    assert [p for p, _ in stash["snaps"]] == [48]
    assert [r.p for r in cs._ckpt_records(man).values()] == [56]
    assert man.stats_snapshot()["ckpt_declines"] == {"retire_layout": 1}


def test_the_counters_reach_the_stats_and_the_metrics_payload(monkeypatch):
    """Both counters sit in stats_snapshot(), which /v1/cache/stats
    returns, and in the server snapshot /v1/metrics carries as "server":
    server.apc.ckpt_budget_evictions and server.apc.ckpt_budget_evicted_mb.
    The stock snapshot is read under any wrap the session installed."""
    import importlib

    # The package exports the FastAPI app under the module's name.
    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")

    man = GmlxAPCManager(num_blocks=64, block_size=16)
    snap = man.stats_snapshot()
    assert snap["ckpt_budget_evictions"] == 0
    assert snap["ckpt_budget_evicted_mb"] == 0
    assert "_budget_evicted_bytes" not in snap

    cs._ckpt_bump(man, "ckpt_budget_evictions", 2)
    cs._ckpt_bump(man, "ckpt_budget_evicted_mb", 3)
    stock = app._server_runtime_snapshot
    while getattr(stock, "_gmlx_orig", None) is not None:
        stock = stock._gmlx_orig
    for name, value in (("apc_manager", man), ("model_cache", {}),
                        ("response_generator", None), ("audio_queue", None)):
        monkeypatch.setattr(app.runtime, name, value)
    apc = stock()["apc"]
    assert apc["enabled"] is True
    assert apc["ckpt_budget_evictions"] == 2
    assert apc["ckpt_budget_evicted_mb"] == 3


def _held_by_none(man):
    return [b for b in man.pool if b.resident_bytes() and b.ref_cnt == 0]


def test_the_count_bound_frees_the_blocks_only_its_victim_held(monkeypatch):
    """A record the count bound drops goes with the blocks only it held,
    as a room victim does: no block stays committed without a record."""
    monkeypatch.setattr(cs, "_CKPT_RECORD_ENTRIES", 2)
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    for c in range(3):
        assert ckpt_store(man, _ids(c, 64), make_hybrid_cache(64, seed=c),
                          extra_hash=c)
    assert _chains(man) == [1, 2]
    assert _held_by_none(man) == []


def test_strip_on_extend_frees_the_window_chain_its_victim_held():
    """A record strip-on-extend drops shares its main chain with the
    deeper record, but its window chain is salted by its own length: those
    blocks go with it."""
    from test_ckpt_tier import make_swa_cache

    man = GmlxAPCManager(num_blocks=64, block_size=16)
    ids = _ids(0, 128)
    for p in (32, 64, 96, 128):
        assert ckpt_store(man, ids[:p], make_swa_cache(p, seed=1))
    assert sorted(r.p for r in cs._ckpt_records(man).values()) == [32, 96, 128]
    assert _held_by_none(man) == []


def test_a_forked_parents_record_is_kept_like_a_head(monkeypatch):
    """A client forks a parent's context into two children, so each child's
    record continues the parent's. The parent's record is a fork point and
    stays like a head, for the parent's next turn; the anchor below it,
    on one line with it, goes first."""
    monkeypatch.setattr(cs, "_CKPT_RECORD_ENTRIES", 5)
    man = GmlxAPCManager(num_blocks=64, block_size=16)
    parent = _ids(0, 64)
    assert ckpt_store(man, parent[:32], make_hybrid_cache(32, seed=1))
    assert ckpt_store(man, parent, make_hybrid_cache(64, seed=2))
    for k in range(2):
        child = parent + list(range(5000 + 100 * k, 5032 + 100 * k))
        assert ckpt_store(man, child, make_hybrid_cache(96, seed=3 + k))
    assert cs._chain_heads(cs._ckpt_records(man)) == {
        (r.ids, r.extra_hash) for r in cs._ckpt_records(man).values()
        if r.p != 32}
    for c in (1, 2):
        assert ckpt_store(man, _ids(c, 64), make_hybrid_cache(64, seed=9 + c),
                          extra_hash=c)
    kept = {(r.extra_hash, r.p) for r in cs._ckpt_records(man).values()}
    assert (0, 64) in kept and (0, 32) not in kept
    _, got = ckpt_lookup(man, parent + [7])
    assert got == 64


def test_blockless_records_compare_the_last_id_first():
    """Kvarn records carry no chain, so the heads scan compares ids. Two
    different prompts differ at the parent's last id in one compare, with
    no slice of the whole prompt."""
    from types import SimpleNamespace

    class NoSlice(tuple):
        def __getitem__(self, i):
            if isinstance(i, slice):
                raise AssertionError("sliced the prompt")
            return tuple.__getitem__(self, i)

    r = SimpleNamespace(p=10, ids=tuple(range(10)), main_blocks=(), b_full=0)
    other = SimpleNamespace(p=20, ids=NoSlice([*range(9), 99, *range(10)]),
                            main_blocks=(), b_full=0)
    assert not cs._ckpt_extends(other, r)
    same = SimpleNamespace(p=20, ids=tuple(range(20)), main_blocks=(),
                           b_full=0)
    assert cs._ckpt_extends(same, r)
