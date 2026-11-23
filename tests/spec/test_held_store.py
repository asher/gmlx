"""A finished request's prompt cache store, held for an idle worker
(GMLX_SPEC_STORE_WHEN_IDLE): the requests that run it first, the free RAM
it may hold, the governor's release, and the drafter state it reads."""

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
    monkeypatch.setattr(spec, "_held_stores", [])
    monkeypatch.setattr(gov, "_REG", dict(gov._REG))
    free = {"bytes": 8 << 30}
    monkeypatch.setattr(kvm, "available_bytes", lambda: free["bytes"])
    stored = []

    def fake_retire(model, prompt_cache, generated, retire, drafter=None,
                    sidecar_ctx=None):
        n = len(retire["full_ids"])
        tail = (sidecar_ctx or {}).get("tail")
        rows = None if tail is None else tail.take(n, n + len(generated))
        stored.append((list(generated), rows))

    monkeypatch.setattr(spec, "_retire_b1", fake_retire)
    return SimpleNamespace(free=free, stored=stored)


def _finish(generated, sidecar_ctx=None, snaps=None):
    """Close a 10-token prompt's request on a 4-token block grid, with a
    256-byte cache."""
    model = SimpleNamespace(_kq_apc_manager=SimpleNamespace(block_size=4))
    retire = {"mode": "ckpt", "full_ids": list(range(10)), "snaps": snaps}
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
