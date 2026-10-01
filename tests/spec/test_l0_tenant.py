"""The L0 prefix cache of the speculative path keeps the entries of each APC
tenant apart. A launch session sets its own tenant, so it neither restores
nor times another client's prompt. CPU only: no model loads."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import gmlx.cache.prefix_cache as pc
from gmlx.cache.prefix_cache import SpecPrefixCache

pytest.importorskip("mlx_vlm")


def _kv(total: int) -> KVCache:
    c = KVCache()
    k = mx.zeros((1, 2, total, 4), dtype=mx.float32)
    c.update_and_fetch(k, k)
    return c


def test_an_entry_hits_only_a_lookup_with_its_salt():
    cache = SpecPrefixCache()
    ids = mx.arange(64)[None]
    longer = mx.arange(65)[None]
    cache.store(ids, [_kv(64)], mx.zeros((1, 1, 4)), salt=7)
    assert cache.lookup(longer) is None
    assert cache.lookup(longer, salt=9) is None
    hit = cache.lookup(longer, salt=7)
    assert hit is not None and hit[0] == 64 and hit[1].salt == 7
    cache.store(ids, [_kv(64)], mx.zeros((1, 1, 4)))
    assert len(cache) == 2
    assert cache.lookup(longer)[1].salt == 0
    cache.store(ids, [_kv(64)], mx.zeros((1, 1, 4)), salt=7)
    assert len(cache) == 2


def test_the_row_salt_is_the_semantic_hash_else_the_tenant():
    assert pc.row_salt(None) == pc.row_salt({}) == 0
    assert pc.row_salt({"_apc_semantic_hash": 5, "_apc_tenant": "a"}) == 5
    a, b = pc.row_salt({"_apc_tenant": "a"}), pc.row_salt({"_apc_tenant": "b"})
    assert a != 0 and b != 0 and a != b
    assert a == pc.row_salt({"_apc_tenant": "a"})


def test_insert_notes_each_row_salt_for_its_prompt_batch(monkeypatch):
    from mlx_vlm.generate import ar

    from gmlx.spec import mtp_prefill

    model = object()
    monkeypatch.setattr(ar.BatchGenerator, "insert",
                        lambda self, prompts, *a, **k: [11 + i for i in range(len(prompts))])
    monkeypatch.setattr(pc, "_ROW_SALTS", type(pc._ROW_SALTS)())
    mtp_prefill._install_l0_row_salts()
    gen = SimpleNamespace(model=model)
    kws = [{"_apc_tenant": "launch-a"}, {}]
    assert ar.BatchGenerator.insert(gen, [[1], [2]], 5, kws) == [11, 12]
    assert pc.take_row_salts(model, [11, 12]) == [pc.row_salt(kws[0]), 0]
    assert pc.take_row_salts(model, [11]) == [0]           # taken once
    ar.BatchGenerator.insert(gen, [[1]], prompt_kwargs=[{"_apc_semantic_hash": 3}])
    assert pc.take_row_salts(object(), [11]) == [0]        # another model's row
    assert pc.take_row_salts(model, [11]) == [3]


def _batch(model, salt: int):
    return SimpleNamespace(
        model=model, _input_ids=mx.arange(70)[None], _inputs_embeds=mx.zeros((1, 70, 4)),
        prompt_cache=[KVCache()], _apc_meta=None, _gmlx_l0_salt=salt,
        _prompt_kwargs={}, _prompt_length_aware_keys=[])


def test_the_prefill_restores_only_an_entry_of_its_own_salt(monkeypatch):
    from gmlx.spec import mtp_prefill

    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_RETIRE_DISABLED", True)
    cache = SpecPrefixCache()
    cache.store(mx.arange(64)[None], [_kv(64)], mx.zeros((1, 64, 4)), salt=7)
    model = SimpleNamespace(_spec_prefix_cache=cache)
    other = _batch(model, 0)
    mtp_prefill._mtp_prefill_init(other)
    assert other._input_ids.shape[1] == 70 and other._mtp_chunk_hiddens == []
    own = _batch(model, 7)
    mtp_prefill._mtp_prefill_init(own)
    assert own._input_ids.shape[1] == 6 and own._mtp_apc_prefix_len == 64
