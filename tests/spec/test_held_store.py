"""A finished request's prompt cache store, held for an idle worker
(GMLX_SPEC_STORE_WHEN_IDLE): the requests that run it first, the free RAM
it may hold, the governor's release, the drafter state it reads, the
worker that runs it and the switch that stores before the last token."""

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

import gmlx.spec.speculative as spec
from gmlx.spec.dflash_drafter import HiddenTail


@pytest.fixture
def held(monkeypatch):
    import gmlx.serve.governor as gov
    import gmlx.serve.kernel_vm as kvm

    monkeypatch.setattr(spec, "_STORE_WHEN_IDLE", True)
    monkeypatch.setattr(spec, "_RETIRE_AFTER_FINAL", True)
    monkeypatch.setattr(spec, "_held_stores", [])
    monkeypatch.setattr(gov, "_REG", dict(gov._REG))
    free = {"bytes": 8 << 30}
    monkeypatch.setattr(kvm, "available_bytes", lambda: free["bytes"])
    stored = []
    held_lcp = []

    def fake_retire(model, prompt_cache, generated, retire, drafter=None,
                    sidecar_ctx=None, lcp=None):
        n = len(retire["full_ids"])
        tail = (sidecar_ctx or {}).get("tail")
        rows = None if tail is None else tail.take(n, n + len(generated))
        stored.append((list(generated), rows))
        held_lcp.append(lcp)

    monkeypatch.setattr(spec, "_retire_b1", fake_retire)
    return SimpleNamespace(free=free, stored=stored, lcp=held_lcp)


def _finish(generated, sidecar_ctx=None, snaps=None, render_ctx=None):
    """Close a 10-token prompt's request on a 4-token block grid, with a
    256-byte cache."""
    model = SimpleNamespace(_kq_apc_manager=SimpleNamespace(block_size=4))
    retire = {"mode": "ckpt", "full_ids": list(range(10)), "snaps": snaps,
              "render_ctx": render_ctx}
    spec._close_and_retire(SimpleNamespace(close=lambda: None), model,
                           [mx.zeros((1, 64))], list(generated), retire,
                           None, sidecar_ctx)


def test_a_held_store_runs_first_only_for_a_request_that_can_use_it(held):
    gen = list(range(10, 16))
    _finish(gen)
    assert held.stored == [] and len(spec._held_stores) == 1
    # Another conversation on the same system prompt, and the same prompt
    # sent again, share less than the sequence end's block: they go first.
    spec.run_held_stores_for(list(range(9)) + [77] * 20)
    spec.run_held_stores_for(list(range(10)))
    assert held.stored == []
    # The next turn of this one shares the whole sequence.
    spec.run_held_stores_for(list(range(16)) + [5, 6])
    assert held.stored == [(gen, None)] and spec._held_stores == []


def test_a_decode_snapshot_below_the_sequence_end_moves_the_cut(held):
    # The store may write the snapshot at 11, which a request sharing the
    # first 11 tokens restores.
    _finish(list(range(10, 16)), snaps=[(11, None)])
    spec.run_held_stores_for(list(range(11)) + [77])
    assert len(held.stored) == 1


def test_a_store_over_its_share_of_free_ram_runs_at_once(held):
    held.free["bytes"] = 1000
    _finish([10, 11])
    assert held.stored == [([10, 11], None)] and spec._held_stores == []


def test_the_governor_drops_held_stores(held):
    import gmlx.serve.governor as gov

    _finish([10, 11])
    _finish([12])
    held_bytes, evict = gov._REG["spec_held_store"]
    assert held_bytes() == 512
    assert evict(1.0) == 512
    spec.run_held_stores("idle")
    assert held.stored == []


def test_a_held_store_reads_the_drafter_tail_as_it_was_at_the_close(held):
    tail = HiddenTail(16)
    tail.prompt(mx.ones((1, 10, 2)))
    tail.accept(mx.full((1, 3, 2), 2.0), 2)
    _finish([10, 11], {"tail": tail})
    # The drafter's next request resets the same tail.
    tail.prompt(mx.zeros((1, 4, 2)))
    spec.run_held_stores("decode")
    (generated, rows), = held.stored
    assert rows.shape == (1, 12, 2) and rows.sum().item() == 28.0


def test_each_worker_runs_only_the_stores_it_held(held):
    # Two engines in one process: B holds a store on its worker while A's
    # worker goes idle. A's pass must not write B's prompt cache.
    def engine(gen, holding, go, done):
        _finish(gen)
        holding.set()
        go.wait(5)
        spec.run_held_stores("idle")
        done.set()

    ev = {k: threading.Event() for k in
          ("a_held", "b_held", "a_go", "b_go", "a_done", "b_done")}
    b = threading.Thread(target=engine, args=(
        [20, 21], ev["b_held"], ev["b_go"], ev["b_done"]))
    a = threading.Thread(target=engine, args=(
        [10, 11], ev["a_held"], ev["a_go"], ev["a_done"]))
    b.start()
    a.start()
    try:
        assert ev["a_held"].wait(5) and ev["b_held"].wait(5)
        assert held.stored == [] and len(spec._held_stores) == 2
        assert not spec.has_held_stores()
        ev["a_go"].set()
        assert ev["a_done"].wait(5)
        assert held.stored == [([10, 11], None)]
        assert len(spec._held_stores) == 1
        ev["b_go"].set()
        assert ev["b_done"].wait(5)
        assert held.stored == [([10, 11], None), ([20, 21], None)]
        assert spec._held_stores == []
    finally:
        ev["a_go"].set()
        ev["b_go"].set()
        a.join(5)
        b.join(5)


def test_a_render_that_diverges_keys_the_hold_on_the_divergence(
        held, monkeypatch):
    # The next turn re-renders the reply (thinking stripped): it shares the
    # sequence only up to 11, where the retire stores its record.
    import gmlx.cache.retire_key as rk

    renders = []

    def lcp(ctx, seq, generated, **kw):
        renders.append(len(seq))
        return 11

    monkeypatch.setattr(rk, "next_turn_lcp", lcp)
    _finish(list(range(10, 16)), render_ctx={"messages": []})
    spec.run_held_stores_for(list(range(11)) + [77, 78])
    assert len(held.stored) == 1
    # The render ran once, at the hold, and the store keys on it.
    assert renders == [16] and held.lcp == [11]


def test_retire_after_final_off_stores_before_the_last_token(
        held, monkeypatch):
    monkeypatch.setattr(spec, "_RETIRE_AFTER_FINAL", False)
    _finish([10, 11])
    assert held.stored == [([10, 11], None)] and spec._held_stores == []
