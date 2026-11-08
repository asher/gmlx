"""A request that ends early (a stop string, a disconnect, a queue timeout)
leaves the tick-guard ledger and the batch. A row ``remove`` cannot take
out of a shared prefill is retried until it is gone, and each new engine
numbers its rows on from the last, so a late cancel names no new request.

Real ``BatchGenerator`` over a tiny llama, with the tick guard installed
and prompt kwargs built as the server's prompt embedding builds them. The
server loop is not started: a ``ResponseGenerator`` shell runs its cancel
branch and builds its engines."""

from __future__ import annotations

import importlib
import queue
import threading
import types

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

import gmlx.serve.tick_guard as tg  # noqa: E402
from gmlx._vendor.mlx_lm.models import llama  # noqa: E402
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria  # noqa: E402
from gmlx.models.vlm_text_only import TextOnlyModel  # noqa: E402
from gmlx.serve.row_sampling import live_uids  # noqa: E402

_GEN = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")

_SHORT = [[3, 9, 4, 7, 2, 5], [5, 1, 8, 2, 6, 11, 13], [4, 4, 9, 1, 3]]
# Six prefill chunks of 16, so two of these share a prompt batch.
_LONG = [[(3 * j) % 60 + 1 for j in range(96)],
         [(5 * j) % 60 + 1 for j in range(96)]]


@pytest.fixture
def server(monkeypatch):
    """A shell with what ``_new_batch_gen`` reads; the retry set and the
    uid counter are left unset."""
    monkeypatch.delenv("GMLX_TICK_GUARD", raising=False)
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    assert tg.install_tick_guard() is True
    mx.random.seed(0)
    rg = object.__new__(_GEN.ResponseGenerator)
    rg._cancel_lock = threading.Lock()
    rg._cancelled = set()
    rg.model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=64)))
    rg.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    rg.stop_tokens = None
    rg.kv_bits = None
    rg.kv_group_size = 64
    rg.kv_quant_scheme = "uniform"
    rg.quantized_kv_start = 5000
    rg.top_logprobs_k = 0
    rg.apc_manager = rg.draft_model = rg.draft_kind = None
    rg.prefill_step_size = 16
    return rg


def _engine(rg):
    return rg._new_batch_gen(_GEN.GenerationArguments(temperature=0.0),
                             mx.default_stream(mx.default_device()))


def _sharing_engine(rg):
    """An engine whose prompt batches take several rows. Other tests'
    installs can cap the decode width, which serializes prefill."""
    gen = _engine(rg)
    gen.completion_batch_size = 32
    gen.prefill_batch_size = 8
    return gen


def _insert(rg, gen, active, prompt, max_tokens):
    """Insert as the serve loop does, and save the uid counter after it."""
    feats = rg.model.get_input_embeddings(mx.array([prompt]), None).to_dict()
    kw = {k: v for k, v in feats.items() if v is not None}
    if rg.apc_manager is not None:
        kw["_apc_semantic_hash"] = 0
    (uid,) = gen.insert([list(prompt)], max_tokens, prompt_kwargs=[kw])
    rg._kq_next_uid = gen.uid_count
    active[uid] = {"rqueue": queue.Queue(), "request_id": f"req-{uid}"}
    return uid


def _tick(rg, gen, active, got, finish):
    """One serve-loop iteration: the cancel branch, then one engine step.
    Responses for a uid not in ``active`` are dropped, as the loop does."""
    assert rg._drop_cancelled(gen, active) is gen
    for r in gen.next()[1]:
        if r.uid not in active:
            continue
        got.setdefault(r.uid, []).append(r.token)
        if r.finish_reason:
            finish[r.uid] = r.finish_reason
            del active[r.uid]


def _run(rg, gen, active, got=None, finish=None):
    got = {} if got is None else got
    finish = {} if finish is None else finish
    while gen.has_work:
        _tick(rg, gen, active, got, finish)
    return got, finish


def _cancelled(info):
    return info["rqueue"].get_nowait() is None


def test_cancelled_decoding_and_queued_rows_leave_the_ledger(server):
    rg = server
    gen = _engine(rg)
    active: dict = {}
    a = _insert(rg, gen, active, _SHORT[0], 24)
    b = _insert(rg, gen, active, _SHORT[1], 8)
    got: dict = {}
    finish: dict = {}
    while gen._prompt_batch is not None or a not in gen._generation_batch.uids:
        _tick(rg, gen, active, got, finish)
    q = _insert(rg, gen, active, _SHORT[2], 8)
    st = tg._state(gen)
    tg._harvest_pending(gen, st)        # what the next tick does first
    assert {a, b, q} <= set(st.ledger)
    infos = {u: active[u] for u in (a, q)}
    rg._cancel(a)
    rg._cancel(q)
    rg._drop_cancelled(gen, active)
    assert not {a, q} & set(st.ledger)
    assert not {a, q} & live_uids(gen)
    assert not {a, q} & set(active)
    assert all(_cancelled(i) for i in infos.values())
    assert rg._kq_unremoved == set()
    _run(rg, gen, active, got, finish)
    assert finish == {b: "length"}
    assert len(got[b]) == 8
    assert st.ledger == {}
    gen.close()


def test_a_row_cancelled_in_a_shared_prefill_stops_after_it(server):
    rg = server
    gen = _sharing_engine(rg)
    active: dict = {}
    a = _insert(rg, gen, active, _LONG[0], 40)
    b = _insert(rg, gen, active, _LONG[1], 40)
    got: dict = {}
    finish: dict = {}
    _tick(rg, gen, active, got, finish)
    assert sorted(gen._prompt_batch.uids) == sorted([a, b])
    rg._cancel(a)
    rg._drop_cancelled(gen, active)
    assert rg._kq_unremoved == {a}
    assert a not in tg._state(gen).ledger
    while gen._prompt_batch is not None:
        gen.next()
    assert a in live_uids(gen)
    rg._drop_cancelled(gen, active)
    assert a not in live_uids(gen)
    assert rg._kq_unremoved == set()
    _run(rg, gen, active, got, finish)
    assert finish == {b: "length"}
    gen.close()


def _prefix_stored(rg, monkeypatch):
    """A prefix-cache engine that has stored the first 64 tokens of
    ``_LONG[0]``."""
    from gmlx._vendor.mlx_vlm.apc import APCManager

    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")
    rg.apc_manager = APCManager(num_blocks=128, block_size=16)
    gen = _sharing_engine(rg)
    active: dict = {}
    _insert(rg, gen, active, _LONG[0][:64], 2)
    _run(rg, gen, active)
    return gen, active


def test_a_row_cancelled_in_a_prefill_alone_gives_back_its_blocks(
        server, monkeypatch):
    """A row that prefills alone is removed at once, and gives back the
    prefix blocks its prefill holds."""
    rg = server
    gen, active = _prefix_stored(rg, monkeypatch)
    a = _insert(rg, gen, active, _LONG[0], 8)
    _tick(rg, gen, active, {}, {})
    assert gen._prompt_batch.uids == [a]
    assert any(blk.ref_cnt > 0 for blk in rg.apc_manager.pool)
    rg._cancel(a)
    assert rg._drop_cancelled(gen, active) is gen
    assert rg._kq_unremoved == set()
    assert all(blk.ref_cnt == 0 for blk in rg.apc_manager.pool)
    gen.close()


def test_a_prefill_every_sharer_left_is_dropped_with_the_engine(
        server, monkeypatch):
    """The loop steps the engine only while a request is active, so a
    prefill no request reads would hold its memory until the next
    request. Its prefix blocks go back to the cache."""
    rg = server
    gen, active = _prefix_stored(rg, monkeypatch)
    a = _insert(rg, gen, active, _LONG[0][:64] + _LONG[0][64:], 8)
    b = _insert(rg, gen, active, _LONG[0][:64] + _LONG[1][64:], 8)
    _tick(rg, gen, active, {}, {})
    assert sorted(gen._prompt_batch.uids) == sorted([a, b])
    assert any(blk.ref_cnt > 0 for blk in rg.apc_manager.pool)
    rg._cancel(a)
    rg._cancel(b)
    assert rg._drop_cancelled(gen, active) is None
    assert live_uids(gen) == set()
    assert all(blk.ref_cnt == 0 for blk in rg.apc_manager.pool)
    assert tg._state(gen).ledger == {}
    gen = _engine(rg)
    assert rg._kq_unremoved == set()
    c = _insert(rg, gen, active, _SHORT[0], 6)
    _, finish = _run(rg, gen, active)
    assert finish == {c: "length"}
    gen.close()


def test_an_unremoved_row_that_finishes_first_leaves_the_retry_set(server):
    rg = server
    gen = _sharing_engine(rg)
    active: dict = {}
    a = _insert(rg, gen, active, _LONG[0], 1)
    b = _insert(rg, gen, active, _LONG[1], 4)
    gen.next()
    rg._cancel(a)
    rg._drop_cancelled(gen, active)
    assert rg._kq_unremoved == {a}
    # The engine steps with no cancel branch until the row's one token
    # ends it.
    for _ in range(20):
        if a not in live_uids(gen):
            break
        gen.next()
    assert a not in live_uids(gen)
    assert rg._kq_unremoved == {a}
    rg._drop_cancelled(gen, active)
    assert rg._kq_unremoved == set()
    _, finish = _run(rg, gen, active)
    assert finish == {b: "length"}
    gen.close()


def test_a_cancelled_row_is_not_requeued(server):
    rg = server
    gen = _engine(rg)
    active: dict = {}
    a = _insert(rg, gen, active, _SHORT[0], 12)
    b = _insert(rg, gen, active, _SHORT[1], 12)
    got: dict = {}
    finish: dict = {}
    while not {a, b} <= set(gen._generation_batch.uids):
        _tick(rg, gen, active, got, finish)
    rg._cancel(a)
    rg._drop_cancelled(gen, active)
    # A later requeue names both rows, as a contained failure would.
    st = tg._state(gen)
    assert gen.remove(b)
    assert tg._requeue(gen, st, [a, b]) == [b]
    assert [s[0] for s in gen._unprocessed_sequences] == [b]
    _run(rg, gen, active, got, finish)
    assert finish == {b: "length"}
    assert len(got[b]) == 12
    assert a not in got or len(got[a]) < 12
    gen.close()


def test_a_new_engine_starts_with_an_empty_retry_set(server):
    rg = server
    rg._drop_cancelled(None, {})            # creates the set
    assert rg._kq_unremoved == set()
    first = _engine(rg)
    assert first.uid_count == 0             # the counter's default
    first.close()
    rg._kq_unremoved.add(0)                 # a uid the old engine left
    gen = _engine(rg)
    assert rg._kq_unremoved == set()
    active: dict = {}
    a = _insert(rg, gen, active, _SHORT[0], 6)
    _, finish = _run(rg, gen, active)
    assert finish == {a: "length"}
    gen.close()


def test_a_late_cancel_names_no_request_of_the_next_engine(server):
    rg = server
    old = _engine(rg)
    active: dict = {}
    olds = [_insert(rg, old, active, p, 3) for p in _SHORT[:2]]
    _run(rg, old, active)
    old.close()
    gen = _engine(rg)
    new = _insert(rg, gen, active, _SHORT[2], 6)
    assert new == max(olds) + 1
    # The client of the first old request stops reading after its finish.
    rg._cancel(olds[0])
    got, finish = _run(rg, gen, active)
    assert finish == {new: "length"}
    assert len(got[new]) == 6
    gen.close()
