"""Quantized-KV + APC composition.

Exact snapshots keep a quantized row in its packed form
(snapshot_prompt_cache_row), and the warm merge re-quantizes a float
row to the live policy when kv_quant_config is passed. With no live
config both MTP warm paths, the owned L1 merge and stock admission,
recover packed rows to float. These tests pin the seams gmlx rides so
both paths stay aligned.
"""

import mlx.core as mx
import pytest

import gmlx.spec.engine as engine


def _stamped_model(bits=8, group=32):
    from types import SimpleNamespace

    from mlx_lm.models.cache import KVCache

    from gmlx.cache.kv_policy import resolve_kv_quant_policy
    from gmlx.serve.kv_policy import ServeKvPolicy

    single = resolve_kv_quant_policy([KVCache()], kv_bits=bits,
                                     kv_group_size=group, mode="single")
    batched = resolve_kv_quant_policy([KVCache()], kv_bits=bits,
                                      kv_group_size=group, mode="batched")
    m = SimpleNamespace()
    m._gmlx_kv_policy = ServeKvPolicy(single, batched)
    return m


def test_live_kv_quant_config_off_without_stamp(monkeypatch):
    # env alone never decides the warm merge
    monkeypatch.setenv("KV_BITS", "8")
    assert engine._live_kv_quant_config() is None
    assert engine._live_kv_quant_config(object()) is None


def test_live_kv_quant_config_reads_stamped_policy():
    cfg = engine._live_kv_quant_config(_stamped_model(8, 32))
    assert cfg is not None
    assert float(cfg["bits"]) == 8.0 and int(cfg["group_size"]) == 32


def test_warm_merge_requantizes_float_row():
    # mlx_vlm cache classes: the serve path's batch merge dispatches on
    # their merge() signature (mlx_lm KVCache has an incompatible one
    # and never reaches this seam).
    from mlx_vlm import apc
    from mlx_vlm.models.cache import KVCache

    row = []
    for _ in range(3):
        c = KVCache()
        k = mx.random.normal((1, 2, 64, 64))
        c.update_and_fetch(k, k)
        row.append(c)
    cfg = engine._live_kv_quant_config(_stamped_model(8, 32))
    warm, n = apc.make_warm_batch_exact_cache_multi(
        [row], prefix_lens=[64], kv_quant_config=cfg)
    assert warm is not None and n == 64
    names = [type(c).__name__ for c in warm]
    assert any("Quantized" in nm for nm in names), names
    # Without the config the merge stays float (the pre-fix behavior:
    # a float row joining a quantized live batch).
    warm2, _ = apc.make_warm_batch_exact_cache_multi(
        [row], prefix_lens=[64])
    assert all("Quantized" not in type(c).__name__ for c in warm2)


def test_store_seam_keeps_quantized_rows_packed():
    # gmlx's anchor store goes through _apc_prompt_cache_for_store ->
    # snapshot_prompt_cache_row; a quantized single-row cache comes back
    # packed, and the warm merge keeps it packed, so a hit joins the
    # quantized live batch it was stored from.
    from mlx_vlm import apc
    from mlx_vlm.models.cache import QuantizedKVCache

    c = QuantizedKVCache(group_size=32, bits=8)
    k = mx.random.normal((1, 2, 64, 64))
    c.update_and_fetch(k, k)
    snap = apc.snapshot_prompt_cache_row([c])
    assert snap is not None
    assert type(snap[0]) is QuantizedKVCache
    assert int(snap[0].offset) == 64
    warm, n = apc.make_warm_batch_exact_cache_multi([snap], prefix_lens=[64])
    assert warm is not None and n == 64
    assert "Quantized" in type(warm[0]).__name__


def test_owned_l1_recovers_packed_rows_when_mtp_batches_run_fp16():
    # Under uniform KV_BITS with MTP the B=1 round packs its KV and its
    # retirement stores packed exact rows, while batches run fp16 (the
    # batched verdict drops, so the live merge config is None). An L1 hit
    # must come back as fp16 batch rows, or the row cannot join a batch.
    from types import SimpleNamespace

    from mlx_vlm.apc import APCManager
    from mlx_vlm.models.cache import BatchKVCache, KVCache, QuantizedKVCache

    import gmlx.spec.mtp_prefill as mtp_prefill
    from gmlx.cache.kv_policy import dropped_policy, resolve_kv_quant_policy
    from gmlx.serve.kv_policy import ServeKvPolicy
    from gmlx.spec.speculative import _lift_injected_cache

    engine._bind_l1_view()
    man = APCManager(num_blocks=64, block_size=16)
    single = resolve_kv_quant_policy([KVCache()], kv_bits=8,
                                     kv_group_size=32, mode="single")
    model = SimpleNamespace(
        _kq_apc_manager=man, _kq_apc_mode="exact", config=SimpleNamespace(),
        make_cache=lambda: [KVCache() for _ in range(3)],
        _gmlx_kv_policy=ServeKvPolicy(
            single, dropped_policy("mtp fp16 when batched", 8, 32,
                                   "batched")))
    assert engine._live_kv_quant_config(model) is None

    prefix = list(range(100, 140))
    stored = []
    for _ in range(3):
        c = QuantizedKVCache(group_size=32, bits=8)
        k = mx.random.normal((1, 2, len(prefix), 64))
        c.update_and_fetch(k, k)
        stored.append(c)
    assert man.store_exact_cache(prefix, stored)

    ids = prefix + list(range(10, 18))
    batch = SimpleNamespace(
        model=model, _input_ids=mx.array([ids]),
        _inputs_embeds=mx.zeros((1, len(ids), 4)),
        prompt_cache=[QuantizedKVCache(group_size=32, bits=8)
                      for _ in range(3)],
        _prompt_kwargs={}, prefill_step_size=32,
        _prompt_length_aware_keys=(),
        _apc_meta=[{"full_input_ids": list(ids), "prefix_len": 0,
                    "extra_hash": 0, "apc_blocks": [],
                    "checkpoint_len": 0}])
    mtp_prefill._mtp_prefill_init(batch)
    assert batch._mtp_l1_prefix_len == len(prefix)
    assert [type(c) for c in batch.prompt_cache] == [BatchKVCache] * 3
    want = mx.dequantize(*(t[..., :len(prefix), :] for t in stored[0].keys),
                         group_size=32, bits=8)
    got = batch.prompt_cache[0].keys[..., :len(prefix), :]
    assert mx.array_equal(got, want.astype(got.dtype))

    host = BatchKVCache([0])
    k = mx.random.normal((1, 2, 24, 64))
    host.update_and_fetch(k, k)
    row = batch.prompt_cache[0]
    host.extend(_lift_injected_cache(host, row))
    q = mx.random.normal((2, 2, 1, 64))
    host.update_and_fetch(q, q)
    assert host.offset.tolist() == [25, len(prefix) + 1]


@pytest.mark.parametrize("strategy", ["block", "checkpoint"])
def test_stock_admission_recovers_packed_rows_when_mtp_batches_run_fp16(
        strategy):
    # Stock admission merges a warm hit to the generator's kv_bits layer
    # types. Under uniform KV_BITS with MTP, live batches run fp16, so a
    # packed warm row could neither join a decoding batch nor take a
    # cold row once it decodes itself.
    from types import SimpleNamespace

    from mlx_vlm import apc
    from mlx_vlm.apc_coordinator import APCCoordinator
    from mlx_vlm.generate import ar
    from mlx_vlm.models.cache import (ArraysCache, BatchKVCache, KVCache,
                                      QuantizedKVCache)

    import gmlx.spec.mtp_prefill as mtp_prefill
    from gmlx.cache.kv_policy import dropped_policy, resolve_kv_quant_policy
    from gmlx.serve.kv_policy import ServeKvPolicy
    from gmlx.spec.speculative import _lift_injected_cache

    class _Generator(ar.BatchGenerator):
        # The stock builder on a bare generator: only its fields are set.
        def __init__(self):
            pass

        def close(self):
            pass

    mtp_prefill.install_full_prompt_mtp_prefill()
    shape = (1, 2, 32, 64)
    if strategy == "block":
        # dense layers page through the block pool as float blocks
        def make_cache():
            return [KVCache() for _ in range(3)]
    else:
        # a recurrent layer makes a checkpoint model, whose exact entries
        # the B=1 round stores packed
        def make_cache():
            return [KVCache(), KVCache(), ArraysCache(size=2)]
    single = resolve_kv_quant_policy([KVCache()], kv_bits=4,
                                     kv_group_size=64, mode="single")
    model = SimpleNamespace(
        config=SimpleNamespace(), make_cache=make_cache,
        _gmlx_kv_policy=ServeKvPolicy(
            single, dropped_policy("mtp fp16 when batched", 4, 64,
                                   "batched")))
    man = apc.APCManager(num_blocks=64, block_size=16)
    gen = _Generator()
    gen.model, gen.apc_manager = model, man
    gen.apc = APCCoordinator(man, model)
    assert gen.apc.strategy == strategy
    gen.apc_mode = gen.apc.legacy_mode
    gen.kv_bits, gen.kv_quant_scheme, gen.kv_group_size = 4, "uniform", 64
    gen.prefill_step_size = 32
    gen.draft_model, gen.draft_kind = object(), "mtp"

    prefix = list(range(100, 132))
    ks = [mx.random.normal(shape) for _ in range(2)]
    if strategy == "block":
        man.release(man.store_kv_blocks(prefix, ks + ks[:1], ks + ks[:1]))
        want = [mx.dequantize(*mx.quantize(k, group_size=64, bits=4),
                              group_size=64, bits=4) for k in ks]
    else:
        row = []
        for k in ks:
            c = QuantizedKVCache(group_size=64, bits=4)
            c.update_and_fetch(k, k)
            row.append(c)
        state = ArraysCache(size=2)
        state[0], state[1] = mx.zeros((1, 3, 8)), mx.zeros((1, 2, 4, 4))
        assert man.store_exact_cache(prefix, row + [state])
        want = [mx.dequantize(*(t[..., :32, :] for t in c.keys),
                              group_size=64, bits=4) for c in row]

    def seq(uid, ids):
        kw = {"inputs_embeds": mx.zeros((1, len(ids), 8)),
              "_apc_semantic_hash": 0}
        return (uid, ids, 4, kw, None, None)

    batch = gen._build_mixed_prompt_batch(
        [seq(0, prefix + [7, 8, 9]), seq(1, list(range(40, 50)))])
    assert batch._apc_meta[0]["prefix_len"] == len(prefix)
    kv = batch.prompt_cache[:2]
    assert [type(c) for c in kv] == [BatchKVCache] * 2
    for c, w in zip(kv, want):
        assert mx.array_equal(c.keys[:1, :, :32, :], w.astype(c.keys.dtype))

    # the suffix prefill: right-padded rows, rolled into left padding
    for c in kv:
        k = mx.random.normal((2, 2, 10, 64))
        c.update_and_fetch(k, k)
        c.finalize()
    assert kv[0].offset.tolist() == [35, 10]

    # the warm batch joins an fp16 batch that is already decoding
    host = BatchKVCache([0])
    k = mx.random.normal((1, 2, 24, 64))
    host.update_and_fetch(k, k)
    host.extend(_lift_injected_cache(host, kv[0]))
    # and, decoding itself, takes a cold row the B=1 round packed
    cold = QuantizedKVCache(group_size=64, bits=4)
    k = mx.random.normal((1, 2, 12, 64))
    cold.update_and_fetch(k, k)
    kv[1].extend(_lift_injected_cache(kv[1], cold))
    for c in (host, kv[1]):
        q = mx.random.normal((c.offset.shape[0], 2, 1, 64))
        c.update_and_fetch(q, q)
    assert host.offset.tolist() == [25, 36, 11]
    assert kv[1].offset.tolist() == [36, 11, 13]


def test_exact_row_recovery_reaches_cache_list_members():
    from mlx_vlm.models.cache import (CacheList, KVCache, QuantizedKVCache,
                                      RotatingKVCache)

    from gmlx.spec.kv_quant import dequantize_exact_row

    q = QuantizedKVCache(group_size=32, bits=8)
    k = mx.random.normal((1, 2, 16, 64))
    q.update_and_fetch(k, k)
    rot = RotatingKVCache(max_size=8)
    plain = KVCache()
    out = dequantize_exact_row([CacheList(q, rot), plain])
    assert out[1] is plain
    assert type(out[0]) is CacheList and out[0].caches[1] is rot
    assert type(out[0].caches[0]) is KVCache
    assert out[0].caches[0].offset == 16
