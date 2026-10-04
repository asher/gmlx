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
    monkeypatch.setattr(ar.BatchGenerator, "remove", ar.BatchGenerator.remove)
    monkeypatch.setattr(pc, "_ROW_SALTS", type(pc._ROW_SALTS)())
    mtp_prefill._install_l0_row_salts()
    gen = SimpleNamespace(model=model)
    kws = [{"_apc_tenant": "launch-a"}, {}]
    assert ar.BatchGenerator.insert(gen, [[1], [2]], 5, kws) == [11, 12]
    assert pc.take_row_salts(model, [11, 12]) == [pc.row_salt(kws[0]), 0]
    assert pc.take_row_salts(model, [11]) == [None]        # taken once
    ar.BatchGenerator.insert(gen, [[1]], prompt_kwargs=[{"_apc_semantic_hash": 3}])
    assert pc.take_row_salts(object(), [11]) == [None]     # another model's row
    assert pc.take_row_salts(model, [11]) == [3]


def _generator(model):
    """A batch generator with only the state that insert and remove use."""
    from mlx_vlm.generate import ar

    gen = object.__new__(ar.BatchGenerator)
    gen.model = model
    gen.uid_count = 0
    gen.max_tokens = 16
    gen.logits_processors = []
    gen._unprocessed_sequences = []
    gen._stream = mx.default_stream(mx.cpu)
    gen._prompt_batch = None
    gen._generation_batch = SimpleNamespace(uids=[])
    gen._wire_stack = None
    return gen


def _real_insert_and_remove(monkeypatch):
    from mlx_vlm.generate import ar

    from gmlx.spec import mtp_prefill

    monkeypatch.setattr(ar.BatchGenerator, "insert", ar.BatchGenerator.insert)
    monkeypatch.setattr(ar.BatchGenerator, "remove", ar.BatchGenerator.remove)
    monkeypatch.setattr(pc, "_ROW_SALTS", type(pc._ROW_SALTS)())
    mtp_prefill._install_l0_row_salts()
    return ar.BatchGenerator


def test_a_row_removed_while_it_waits_leaves_no_salt(monkeypatch):
    gen_cls = _real_insert_and_remove(monkeypatch)
    model = object()
    gen = _generator(model)
    (busy,) = gen_cls.insert(gen, [[1, 2, 3]], prompt_kwargs=[{}])
    (queued,) = gen_cls.insert(gen, [[4, 5, 6]],
                               prompt_kwargs=[{"_apc_semantic_hash": 0x5E55}])
    assert gen_cls.remove(gen, queued)
    assert pc.take_row_salts(model, [busy]) == [0]
    assert dict(pc._ROW_SALTS) == {}


def test_a_row_at_a_reused_uid_takes_its_own_salt(monkeypatch):
    """The server makes a new batch generator after an idle gap, and its
    rows start again at uid 0 on the same model. A session row that never
    reached a prompt batch can leave its note at such a uid."""
    gen_cls = _real_insert_and_remove(monkeypatch)
    model = object()
    pc.note_row_salts(model, [1], [0x5E55])
    gen = _generator(model)
    rows = [gen_cls.insert(gen, [[7, 8, 9]], prompt_kwargs=[{}])[0] for _ in range(2)]
    assert rows == [0, 1]
    assert pc.take_row_salts(model, rows) == [0, 0]
    pc.note_row_salts(model, [0], [0x5E55])
    gen_cls.insert(_generator(model), [[1]], prompt_kwargs=None)
    assert pc.take_row_salts(model, [0]) == [0]


def test_a_replaced_note_leaves_last(monkeypatch):
    monkeypatch.setattr(pc, "_ROW_SALTS", type(pc._ROW_SALTS)())
    monkeypatch.setattr(pc, "_ROW_SALTS_MAX", 2)
    model = object()
    pc.note_row_salts(model, [0, 1], [5, 6])
    pc.note_row_salts(model, [0], [7])
    pc.note_row_salts(model, [2], [8])
    assert pc.take_row_salts(model, [0, 1, 2]) == [7, None, 8]


@pytest.mark.parametrize("manager", [object(), None], ids=["l1", "l0-only"])
def test_the_salt_of_an_adapted_row_holds_its_lora_scales(monkeypatch, manager):
    """The server installs the LoRA channel's insert wrapper first, and that
    wrapper takes the request's scales."""
    from mlx_vlm.generate import ar
    from mlx_vlm.server.generation import ResponseGenerator

    import gmlx.lora_rows as lora_rows
    from gmlx.spec import mtp_prefill

    for owner, name in ((ResponseGenerator, "_make_logits_processors"),
                        (ar.GenerationBatch, "_step"),
                        (ar.PromptProcessingBatch, "generate"),
                        (ar.PromptProcessingBatch, "prompt_step"),
                        (ar.BatchGenerator, "remove")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(ar.BatchGenerator, "insert",
                        lambda self, prompts, *a, **k: [0 for _ in prompts])
    monkeypatch.setattr(lora_rows, "_channel_installed", False)
    monkeypatch.setattr(pc, "_ROW_SALTS", type(pc._ROW_SALTS)())
    lora_rows.install_row_channel()
    mtp_prefill._install_l0_row_salts()
    model = object()
    gen = SimpleNamespace(model=model, apc_manager=manager,
                          _apc_extra_hash=lambda kw: 0)
    salts = {}
    for scales in ((), (0.5,), (0.25,)):
        lora_rows._stash_pending(scales)
        ar.BatchGenerator.insert(gen, [[1, 2, 3]], prompt_kwargs=[{"_apc_semantic_hash": 5}])
        salts[scales] = pc.take_row_salts(model, [0])[0]
    assert salts[()] == 5
    assert salts[(0.5,)] == 5 ^ lora_rows.lora_salt((0.5,))
    assert len(set(salts.values())) == 3


def _batch(model, salt: int | None):
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


def test_a_row_with_no_salt_note_skips_l0(monkeypatch):
    """The tick guard queues a row again after a repeated memory error, and
    that bypasses insert, so the row reaches its next prompt batch with no
    note. A note can also leave past the most notes. Such a row must not
    read or write the entries of the clients with salt 0."""
    from gmlx.serve import tick_guard
    from gmlx.spec import mtp_prefill

    gen_cls = _real_insert_and_remove(monkeypatch)
    model = object()
    gen = _generator(model)
    kw = {"_apc_tenant": "launch-a"}
    (uid,) = gen_cls.insert(gen, [[1, 2, 3]], prompt_kwargs=[kw])
    st = tick_guard._state(gen)
    tick_guard._harvest_pending(gen, st)
    assert pc.take_row_salts(model, [uid]) == [pc.row_salt(kw)]
    gen._unprocessed_sequences.clear()
    gen._generation_batch = SimpleNamespace(uids=[uid], filter=lambda keep: None)
    tick_guard._rebuild_row(gen, st, uid)
    assert gen._unprocessed_sequences[0][0] == uid
    assert pc.take_row_salts(model, [uid]) == [None]

    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_RETIRE_DISABLED", True)
    cache = SpecPrefixCache()
    cache.store(mx.arange(64)[None], [_kv(64)], mx.zeros((1, 64, 4)))
    model = SimpleNamespace(_spec_prefix_cache=cache)
    unknown = _batch(model, None)
    assert mtp_prefill._l0_cache(unknown) is None
    mtp_prefill._mtp_prefill_init(unknown)
    assert unknown._input_ids.shape[1] == 70 and unknown._mtp_chunk_hiddens == []
    known = _batch(model, 0)
    assert mtp_prefill._l0_cache(known) is cache
    mtp_prefill._mtp_prefill_init(known)
    assert known._input_ids.shape[1] == 6
