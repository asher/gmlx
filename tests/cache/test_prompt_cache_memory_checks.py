"""How often a request makes room in the prompt cache.

Each ``prepare_prefill`` sums the resident bytes of the whole cache. A
cold text request makes room once when its prompt batch is built, and
once after each prefill chunk. A media request also makes room when it
arrives, before its encoder allocates. A warm batch plans its memory in
``merge_rows`` with the rows' full lengths. An exact lookup still sees a
queued disk write, because the batch build flushes the writer before it
looks up.

The model is a tiny random gemma-4 language model, whose sliding window
layers put it on the exact tier.
"""

from __future__ import annotations

import importlib
import queue
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest
from gmlx._vendor.mlx_vlm import apc as _apc
from gmlx._vendor.mlx_vlm.apc_coordinator import APCCoordinator
from gmlx._vendor.mlx_vlm.models.cache import KVCache
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria

import gmlx.load.vlm as gvlm
import gmlx.spec.ckpt as ckpt
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.apc_manager import GmlxAPCManager

from test_apc_media import _HIDDEN, _gemma4_config
from test_generation_loop import _GEN, _Batch, _Inputs, _response_generator, _stop

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")

_V = 128
_BS = 16
_RIG = {}


@pytest.fixture(autouse=True)
def _stock_tier(monkeypatch):
    """These tests read the stock exact tier. The checkpoint tier, which
    another test in this process may have installed, stays off."""
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(ckpt, "_SPEC_APC_DISABLED", True)


def _rig():
    if not _RIG:
        mx.random.seed(3)
        model, _ = gvlm.build_vlm_model(_gemma4_config("gemma4"))
        lm = model.language_model
        mx.eval(lm.parameters())
        table = mx.random.normal((_V, _HIDDEN), key=mx.random.key(1)) * 0.5
        mx.eval(table)
        _RIG.update(lm=lm, table=table)
    return SimpleNamespace(**_RIG)


def _kw(ids):
    return {"inputs_embeds": _rig().table[mx.array(ids)][None],
            "_apc_semantic_hash": 0}


def _ids(n, seed=1):
    return [1 + (i * 7 + seed) % 50 for i in range(n)]


def _manager(**kw):
    m = GmlxAPCManager(num_blocks=64, block_size=_BS, **kw)
    m._exact_cache_max = 8
    m._disk_min_free_ram_bytes = 0
    return m


class _Counted:
    """Records each call of a bound method and passes it on."""

    def __init__(self, obj, name):
        self.calls = []
        orig = getattr(obj, name)

        def call(*a, **k):
            self.calls.append((a, k))
            return orig(*a, **k)

        setattr(obj, name, call)


def _generator(manager):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    return ar.BatchGenerator(_rig().lm, proc,
                             sampler=lambda x: mx.argmax(x, axis=-1),
                             max_tokens=1, apc_manager=manager)


def _run(gen, rows):
    out = {u: [] for u in gen.insert(rows, [1] * len(rows),
                                     prompt_kwargs=[_kw(r) for r in rows])}
    while gen.has_work:
        _p, resp = gen.next()
        for r in resp:
            out[r.uid].append(int(r.token))
    gen.close()
    return list(out.values())


def _matched(manager):
    return int(manager.stats_snapshot().get("matched_tokens", 0) or 0)


# -- the batch engine -------------------------------------------------------

def test_a_cold_text_admission_makes_room_once_before_prefill():
    m = _manager()
    made = _Counted(m, "prepare_prefill")
    gen = _generator(m)
    planned = _Counted(gen.apc, "prepare_prefill")
    # 12 tokens is under the checkpoint guard, so one prefill chunk.
    _run(gen, [_ids(12)])
    assert [a for a, _k in planned.calls] == [([12],)]
    assert planned.calls[0][1]["prefix_lengths"] == [0]
    # The prompt batch's own call, then the one after its chunk.
    assert len(made.calls) == 2


def test_warm_rows_plan_memory_with_their_full_lengths():
    m = _manager()
    _run(_generator(m), [_ids(40)])
    gen = _generator(m)
    planned = _Counted(gen.apc, "prepare_prefill")
    merged = _Counted(gen.apc, "merge_rows")
    warm, cold = _ids(40) + [7, 8, 9], [3] * 20
    before = _matched(m)
    _run(gen, [warm, cold])
    restored = _matched(m) - before
    assert restored > 0
    plans = [(list(a[0]), list(k["prefix_lengths"])) for a, k in planned.calls]
    merges = [(list(k["lengths"]), list(a[1])) for a, k in merged.calls]
    # The rows may share a batch or not; the warm row's batch merges.
    (lengths, prefixes), = [m for m in merges if len(warm) in m[0]]
    assert prefixes[lengths.index(len(warm))] == restored
    assert set(lengths) <= {len(warm), len(cold)}
    # merge_rows plans before it allocates, the prompt batch after.
    for merge in merges:
        assert plans.count(merge) == 2


def test_an_exact_lookup_sees_a_queued_disk_write(tmp_path):
    """The entry's disk write is still queued when the next request comes,
    and the writer holds it until the first flush. The batch build
    flushes before its lookup, so the lookup finds the entry."""
    source = _manager()
    _run(_generator(source), [_ids(40)])
    entries = list(source._exact_cache.items())
    assert entries

    disk = _apc.DiskBlockStore(tmp_path, namespace="t")
    release = threading.Event()
    try:
        m = _manager(disk=disk)
        m._exact_cache_max = 0       # every exact hit comes from disk
        write, flush = disk._write_payload, disk.flush

        def held_write(*a):
            release.wait(timeout=60)
            return write(*a)

        def releasing_flush():
            release.set()
            flush()

        disk._write_payload = held_write
        disk.flush = releasing_flush
        for key, e in entries:
            assert disk.save_exact_cache(key, e.token_ids, e.extra_hash,
                                         e.prompt_cache)
        assert disk.pending_bytes > 0 and not release.is_set()

        gen = _generator(m)
        picks = []
        pick = gen._apc_pick_for

        def recorded_pick(seq):
            picks.append(release.is_set())
            return pick(seq)

        gen._apc_pick_for = recorded_pick
        _run(gen, [_ids(40) + [7, 8, 9]])
        assert _matched(m) > 0
        assert picks == [True]
    finally:
        release.set()
        disk.close()



def test_a_lookup_plans_memory_for_its_own_rows():
    """A restore inside a lookup sizes its room from the memory plan, so
    the plan holds this admission's rows, not the last admission's."""
    m = _manager()
    _run(_generator(m), [_ids(40)])
    assert m.memory_plan.lengths == [40]

    gen = _generator(m)
    seen = []
    pick = gen._apc_pick_for

    def recorded_pick(seq):
        seen.append(list(m.memory_plan.lengths))
        return pick(seq)

    gen._apc_pick_for = recorded_pick
    _run(gen, [_ids(24, seed=5)])
    assert seen == [[24]]

# -- eviction under scripted memory -----------------------------------------

def _row(n, seed):
    c = KVCache()
    k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed))
    c.update_and_fetch(k, k + 1)
    mx.eval(c.keys, c.values)
    return [c]


def _scripted(budget):
    """A manager under autosize rules (no reserves) whose headroom is
    ``budget[0]`` less what it holds, as if its entries were all of the
    memory in use."""
    m = _manager()
    m.memory_reserve_bytes = 0
    m._gmlx_no_prefill_reserve = True
    m._prefill_reserve_bytes = 0
    m.memory_max_bytes = 1 << 40
    m._memory_headroom = lambda: budget[0] - m._resident_bytes_locked()
    return m


def _fill(m, rows):
    for i, n in enumerate(rows):
        assert m.store_exact_cache(_ids(n, seed=20 + i), _row(n, i),
                                   extra_hash=99)


def test_cold_admissions_evict_the_same_entries_as_an_arrival_check():
    """The same stores and cold admissions, with and without the room
    check that admission made before the lookup, leave the same entries."""
    rows = [32, 48, 64, 32, 48]
    total = sum(_apc._cache_nbytes(_row(n, 0)) for n in rows)
    kept = []
    for early in (True, False):
        budget = [1 << 40]
        m = _scripted(budget)
        _fill(m, rows)
        budget[0] = total // 2
        for n in (12, 40, 24):
            gen = _generator(m)
            prompt = _ids(n, seed=n)
            if early:
                gen.apc.prepare_prefill(
                    [n], prefill_step_size=gen.prefill_step_size)
            _run(gen, [prompt])
        assert m.stats.memory_evictions > 0
        kept.append((list(m._exact_cache), m.stats.memory_evictions))
    assert kept[0] == kept[1]


def test_a_warm_request_keeps_its_entry_under_pressure():
    """The request's entry is the oldest, and the budget is short by a
    little more than its size. A room check before the lookup evicted it,
    so the request ran cold. The lookup now runs first and marks the entry
    recent, so the room its restore makes comes from a newer entry."""
    source = _manager()
    _run(_generator(source), [_ids(40)])
    entry = max(source._exact_cache.values(), key=lambda e: len(e.token_ids))
    restored = []
    for early in (True, False):
        budget = [1 << 40]
        m = _scripted(budget)
        assert m.store_exact_cache(entry.token_ids, entry.prompt_cache,
                                   extra_hash=entry.extra_hash)
        prompt = _ids(40) + [7, 8, 9]
        own = m._resident_bytes_locked()
        restore = m.memory_plan.restore_bytes(
            entry.prompt_cache, len(entry.token_ids), len(prompt))
        # A stored filler row holds 128 bytes a token. Each is larger than
        # the entry and its restored copy together.
        _fill(m, [(own + restore) // 128 + 16] * 2)
        sizes = [_apc._cache_nbytes(e.prompt_cache)
                 for e in m._exact_cache.values()]
        assert min(sizes[1:]) > own + restore
        budget[0] = m._resident_bytes_locked() - own - 1
        gen = _generator(m)
        if early:
            gen.apc.prepare_prefill([len(prompt)],
                                    prefill_step_size=gen.prefill_step_size)
        before = _matched(m)
        _run(gen, [prompt])
        assert m.stats.memory_evictions > 0
        restored.append(_matched(m) - before)
    assert restored[0] == 0 and restored[1] > 0


def test_a_text_request_makes_no_room_at_arrival(monkeypatch):
    """A text request the gates defer never reaches its prompt batch, so
    it evicts nothing. A media request still makes room before its
    encoder runs."""
    budget = [1 << 40]
    m = _scripted(budget)
    _fill(m, [32, 48, 64])
    budget[0] = 0
    coordinator = APCCoordinator(m, _rig().lm)

    class _ApcBatch(_Batch):
        apc = coordinator

    rg = _response_generator(monkeypatch)
    monkeypatch.setattr(_GEN, "BatchGenerator", _ApcBatch)
    planned = _Counted(coordinator, "prepare_prefill")
    thread = threading.Thread(target=rg._run_impl, daemon=True)
    thread.start()

    def request(**media):
        rq: queue.Queue = queue.Queue()
        rg.requests.put(_GEN.QueuedGenerationRequest(
            rqueue=rq, raw_inputs={"input_ids": None, "audio_clips": []},
            prompt_tokens=2, args=_GEN.GenerationArguments(max_tokens=1),
            **media))
        assert isinstance(rq.get(timeout=5), _GEN.GenerationContext)
        assert rq.get(timeout=5) is None

    try:
        request()
        assert planned.calls == [] and m.stats.memory_evictions == 0
        assert len(m._exact_cache) == 3
        request(images=[_Inputs()])
        assert len(planned.calls) == 1 and m.stats.memory_evictions == 3
        request(audio=[_Inputs()])
        assert len(planned.calls) == 2
    finally:
        _stop(rg, thread)
