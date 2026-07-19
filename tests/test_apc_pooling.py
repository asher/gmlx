"""APC participation for PoolingCache stacks + the rotating-safe KV
quantizer (serve KV_BITS).

Gates the merge blockers: a deepseek4 row snapshot must clone (in-memory
exact store), round-trip through a disk shard bit-identically (packed pools
stored packed), and the per-step maybe_quantize_kv_cache replacement must
pack pools / skip rotating caches instead of raising the upstream
NotImplementedError mid-generation."""

import mlx.core as mx
import pytest

from gmlx.apc_pooling import (
    install_pooling_apc_support,
    install_safe_kv_quantization,
)
from gmlx.deepseek_v4_cache import PoolingCache

D = 64


@pytest.fixture(scope="module", autouse=True)
def _installed():
    install_pooling_apc_support()
    install_safe_kv_quantization()


def _pool(quantized=False, rows=9, remainder=2):
    c = PoolingCache(4)
    if quantized:
        c.quantize_storage(group_size=32, bits=8)
    if rows:
        c.update_and_fetch(mx.random.normal((1, rows, D)).astype(mx.float16))
    if remainder:
        c.accumulate_windows(
            mx.random.normal((1, remainder, D)).astype(mx.float16),
            mx.random.normal((1, remainder, 2)).astype(mx.float16),
            0,
        )
        c._undo = None
    if rows:
        # Ratio-4 overlap lookback: the last completed window's raw rows.
        c._prev_kv = mx.random.normal((1, 4, D)).astype(mx.float16)
        c._prev_gate = mx.random.normal((1, 4, 2)).astype(mx.float16)
    return c


def _same_pool(a, b):
    assert a.ratio == b.ratio
    assert a.remainder == b.remainder
    assert a.size() == b.size()
    assert a._qbits == b._qbits
    if a.remainder:
        assert mx.array_equal(
            a.buf_kv[:, : a.remainder], b.buf_kv[:, : b.remainder]
        )
        assert mx.array_equal(
            a.buf_gate[:, : a.remainder], b.buf_gate[:, : b.remainder]
        )
    if a.size():
        assert mx.array_equal(a.pooled, b.pooled)
    assert (a._prev_kv is None) == (b._prev_kv is None)
    if a._prev_kv is not None:
        assert mx.array_equal(a._prev_kv, b._prev_kv)
        assert mx.array_equal(a._prev_gate, b._prev_gate)


@pytest.mark.parametrize("quantized", [False, True])
def test_clone_roundtrip(quantized):
    from mlx_vlm import apc

    mx.random.seed(2)
    src = _pool(quantized=quantized)
    targets = []
    clone = apc._clone_cache_entry_for_apc(
        src, min_capacity_tokens=None, eval_targets=targets
    )
    if targets:
        mx.eval(*targets)
    _same_pool(src, clone)
    assert clone.is_quantized == quantized  # packed entries stay packed
    # Decoupled: appending to the source must not leak into the clone.
    n = clone.size()
    src.update_and_fetch(mx.random.normal((1, 1, D)).astype(mx.float16))
    assert clone.size() == n


def test_model_apc_mode_resolves_exact_for_pooling_stack():
    # Gate ahead of every other arm: if the support predicate rejects
    # PoolingCache, model_apc_mode is None and serve-side APC never engages.
    from gmlx.cache_compat import construction_cache_module
    _c = construction_cache_module()
    CacheList, RotatingKVCache = _c.CacheList, _c.RotatingKVCache

    from mlx_vlm import apc

    stack = [CacheList(RotatingKVCache(max_size=8), _pool(rows=0, remainder=0),
                       _pool(rows=0, remainder=0))]

    class _LM:
        def make_cache(self):
            return stack

    assert apc._cache_entry_supports_exact_apc(stack[0])
    assert apc.model_apc_mode(_LM()) == "exact"


def test_apc_engages_for_mlx_lm_origin_model():
    """The serve engagement gate: BatchGenerator drops the APC manager
    outright when model_apc_mode(model) is None. mlx-lm-arch text models
    build mlx-lm-origin caches, which mlx-vlm >= 0.6.4's isinstance gates
    reject - found live as gemma-4 serving with apc_enabled=true but zero
    lookups/stores. ensure_runtime_origin_make_cache must keep the mode
    resolving on every supported mlx-vlm version."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_vlm import apc

    from gmlx.cache_compat import (
        ensure_runtime_origin_make_cache,
        runtime_cache_module,
    )

    class _GemmaLike:  # sliding-window + global attention mix
        def make_cache(self):
            return [RotatingKVCache(max_size=8), KVCache()]

    class _DenseLike:
        def make_cache(self):
            return [KVCache(), KVCache()]

    gemma = ensure_runtime_origin_make_cache(_GemmaLike())
    dense = ensure_runtime_origin_make_cache(_DenseLike())
    assert apc.model_apc_mode(gemma) == "exact"
    assert apc.model_apc_mode(dense) == "block"
    vlm = runtime_cache_module()
    assert all(type(c) is vlm.KVCache for c in dense.make_cache())

    # Idempotent: re-wrapping must not stack another layer.
    wrapped = gemma.make_cache
    ensure_runtime_origin_make_cache(gemma)
    assert gemma.make_cache is wrapped


def test_spec_apc_kill_switch_strips_manager_from_stock(monkeypatch):
    """GMLX_SPEC_APC=0 must keep the manager away from stock ar.py too:
    since mlx-vlm 0.6.4 its post-prefill exact store handles B=1 MTP caches
    (earlier versions silently declined them), so a stashed-but-disabled
    manager still collects stores. Found live by test_l1_kill_switch (which
    needs the real MTP model); this pins the stash wrapper's behavior
    model-free by wrapping a probe class."""
    import importlib
    from types import SimpleNamespace

    import gmlx.spec_engine as spec_engine

    ar = importlib.import_module("mlx_vlm.generate.ar")
    seen = {}

    class _ProbeBG:
        def __init__(self, model, processor, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(ar, "BatchGenerator", _ProbeBG)
    spec_engine._install_apc_manager_stash()
    mgr = object()

    monkeypatch.setattr(spec_engine, "_SPEC_APC_DISABLED", True)
    model = SimpleNamespace()
    ar.BatchGenerator(model, None, draft_model=object(), apc_manager=mgr)
    assert seen["apc_manager"] is None
    assert model._kq_apc_manager is None

    monkeypatch.setattr(spec_engine, "_SPEC_APC_DISABLED", False)
    seen.clear()
    model = SimpleNamespace()
    ar.BatchGenerator(model, None, draft_model=object(), apc_manager=mgr)
    assert seen["apc_manager"] is mgr
    assert model._kq_apc_manager is mgr

    # Non-speculative batches are outside the spec kill switch's scope.
    monkeypatch.setattr(spec_engine, "_SPEC_APC_DISABLED", True)
    seen.clear()
    ar.BatchGenerator(SimpleNamespace(), None, apc_manager=mgr)
    assert seen["apc_manager"] is mgr


def test_rebind_to_runtime_origin_recurses_and_skips_ours():
    from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache

    from gmlx.cache_compat import (
        rebind_to_runtime_origin,
        runtime_cache_module,
    )
    from gmlx.deepseek_v4_cache import PoolingCache

    vlm = runtime_cache_module()
    pool = _pool(rows=0, remainder=0)
    stack = [CacheList(RotatingKVCache(max_size=8), pool), KVCache()]
    rebind_to_runtime_origin(stack)
    assert type(stack[0]) is vlm.CacheList
    assert type(stack[0].caches[0]) is vlm.RotatingKVCache
    assert type(stack[0].caches[1]) is PoolingCache  # ours, untouched
    assert type(stack[1]) is vlm.KVCache


def test_warm_batch_adoption_merges_pooling_stack():
    # After an exact hit the engine merges the restored row into batch
    # caches via _merge_exact_cache_entries; a None there silently falls
    # back to cold prefill, so the hit must survive the merge.
    from gmlx.cache_compat import construction_cache_module
    _c = construction_cache_module()
    CacheList, RotatingKVCache = _c.CacheList, _c.RotatingKVCache

    from mlx_vlm import apc

    mx.random.seed(7)
    warm = [
        RotatingKVCache(max_size=8),
        CacheList(RotatingKVCache(max_size=8), _pool(rows=8, remainder=0),
                  _pool(rows=8, remainder=0)),
    ]
    for _ in range(6):
        warm[0].update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
        warm[1].caches[0].update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
    batch, prefix = apc.make_warm_batch_exact_cache_multi([warm], [6])
    assert batch is not None
    assert prefix == 6
    merged_pool = batch[1].caches[1]
    assert mx.array_equal(
        merged_pool.pooled[0][: warm[1].caches[1].size()],
        warm[1].caches[1].pooled[0],
    )


def test_row_snapshot_v4_stack():
    # The merge blocker: a deepseek4-shaped cache row must snapshot whole.
    from gmlx.cache_compat import construction_cache_module
    _c = construction_cache_module()
    CacheList, RotatingKVCache = _c.CacheList, _c.RotatingKVCache

    from gmlx.cache_snapshot import row_snapshot

    mx.random.seed(3)
    rot = RotatingKVCache(max_size=8)
    for _ in range(3):
        rot.update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
    idx_pool = _pool(rows=5, remainder=1)
    idx_pool.quantizable = False
    stack = [rot, CacheList(rot, _pool(rows=5, remainder=1), idx_pool)]
    snaps = row_snapshot(stack)
    assert snaps is not None and len(snaps) == 2
    _same_pool(stack[1].caches[1], snaps[1].caches[1])


@pytest.mark.parametrize("quantized", [False, True])
def test_disk_exact_roundtrip(tmp_path, quantized):
    from mlx_vlm import apc

    mx.random.seed(4)
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        src = _pool(quantized=quantized)
        snapshot = apc._DiskExactCacheSnapshot(
            cache_hash=42,
            token_ids=(1, 2, 3),
            extra_hash=0,
            prompt_cache=[src],
        )
        path = store.dir / f"exact_{42:032x}{store.SUFFIX}"
        store._write_exact_cache_snapshot(path, snapshot)
        loaded = store.load_exact_cache(42)
        assert loaded is not None
        token_ids, _extra, caches = loaded
        assert token_ids == (1, 2, 3)
        _same_pool(src, caches[0])
        assert caches[0].is_quantized == quantized
    finally:
        stop = getattr(store, "close", None) or getattr(store, "stop", None)
        if callable(stop):
            stop()


def test_disk_roundtrip_zero_width_values(tmp_path):
    # deepseek4's DSA local caches store keys with zero-width values;
    # mx.save_safetensors rejects zero-size arrays, so the shard writer
    # spills them to metadata and the reader synthesizes zeros.
    from gmlx.cache_compat import construction_cache_module
    _c = construction_cache_module()
    CacheList, RotatingKVCache = _c.CacheList, _c.RotatingKVCache

    from mlx_vlm import apc

    mx.random.seed(8)
    rot = RotatingKVCache(max_size=8)
    for _ in range(5):
        rot.update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
    stack = [rot, CacheList(RotatingKVCache(max_size=8), _pool(), _pool())]
    for _ in range(5):
        stack[1].caches[0].update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
    store = apc.DiskBlockStore(tmp_path, namespace="t")
    try:
        snapshot = apc._DiskExactCacheSnapshot(
            cache_hash=7, token_ids=(1, 2, 3), extra_hash=0,
            prompt_cache=stack,
        )
        path = store.dir / f"exact_{7:032x}{store.SUFFIX}"
        store._write_exact_cache_snapshot(path, snapshot)
        assert path.stat().st_size > 0
        loaded = store.load_exact_cache(7)
        assert loaded is not None
        _ids, _extra, caches = loaded
        assert mx.array_equal(caches[0].keys[..., :5, :], rot.keys[..., :5, :])
        assert caches[0].values.shape[-1] == 0
        assert caches[1].caches[0].values.shape[-1] == 0
        _same_pool(stack[1].caches[1], caches[1].caches[1])
    finally:
        stop = getattr(store, "close", None) or getattr(store, "stop", None)
        if callable(stop):
            stop()


def test_safe_kv_quantization_packs_pools_and_skips_rotating():
    from mlx_lm.generate import maybe_quantize_kv_cache
    from gmlx.cache_compat import construction_cache_module
    _c = construction_cache_module()
    CacheList, KVCache, RotatingKVCache = _c.CacheList, _c.KVCache, _c.RotatingKVCache

    mx.random.seed(5)
    rot = RotatingKVCache(max_size=4)
    for _ in range(6):  # rotated: stock to_quantized would raise NYI
        rot.update_and_fetch(
            mx.random.normal((1, 1, 1, 4)), mx.zeros((1, 1, 1, 0))
        )
    kv = KVCache()
    kv.update_and_fetch(
        mx.random.normal((1, 1, 3, 32)), mx.random.normal((1, 1, 3, 32))
    )
    comp, idx = _pool(rows=5, remainder=0), _pool(rows=5, remainder=0)
    idx.quantizable = False
    pc = [rot, kv, CacheList(RotatingKVCache(max_size=4), comp, idx)]

    maybe_quantize_kv_cache(pc, 0, 32, 8)

    assert pc[0] is rot  # rotating: skipped, not crashed
    assert type(pc[1]).__name__ == "QuantizedKVCache"  # stock conversion kept
    assert comp.is_quantized  # pool packed in place (had rows already)
    assert not idx.is_quantized  # indexer pool opted out
    assert comp.size() == 5


def test_quantize_storage_converts_existing_rows():
    mx.random.seed(6)
    rows = mx.random.normal((1, 9, D)).astype(mx.float16)
    ref = PoolingCache(4)
    ref.update_and_fetch(rows)
    conv = PoolingCache(4)
    conv.update_and_fetch(rows)
    conv.quantize_storage(group_size=32, bits=8)
    assert conv.is_quantized and conv.size() == 9
    err = mx.abs(conv.pooled - ref.pooled).max().item()
    assert err < 0.05
    conv.quantize_storage(group_size=32, bits=8)  # idempotent
    assert conv.size() == 9


# --- pooled kv_bits at prompt-batch construction (serve) --------------------
#
# install_pooled_prompt_kv_quant wraps PromptProcessingBatch.__init__; the
# tests swap in a minimal stand-in class so the wrap is exercised without a
# real model/engine, and monkeypatch restores the module attr afterwards.

from types import SimpleNamespace  # noqa: E402


class _V4ish:
    def __init__(self):
        self.comp = _pool(rows=0, remainder=0)
        self.idx = _pool(rows=0, remainder=0)
        self.idx.quantizable = False
        self._caches = [SimpleNamespace(caches=[self.comp, self.idx])]

    def make_cache(self):
        return self._caches


def _install_on_fake(monkeypatch):
    from mlx_vlm.generate import ar

    class _FakePPB:
        def __init__(self, *a, **k):
            self.init_kv_bits = k.get("kv_bits")
            self.prompt_cache = k["model"].make_cache()

    monkeypatch.setattr(ar, "PromptProcessingBatch", _FakePPB)
    from gmlx.apc_pooling import install_pooled_prompt_kv_quant

    install_pooled_prompt_kv_quant()
    return _FakePPB


def test_pooled_prompt_kv_quant_arms_b1(monkeypatch, capsys):
    ppb = _install_on_fake(monkeypatch)
    m = _V4ish()
    b = ppb(model=m, input_ids=[[1, 2, 3]], kv_bits=8, kv_group_size=32)
    assert b.init_kv_bits is None  # stripped before the stock init
    assert m.comp.is_quantized and m.comp._qgroup == 32
    assert not m.idx.is_quantized  # indexer pool opted out
    assert "[kv] 8-bit pooled" in capsys.readouterr().out


def test_pooled_prompt_kv_quant_batch_and_mtp_stay_fp16(monkeypatch):
    ppb = _install_on_fake(monkeypatch)
    m = _V4ish()
    b = ppb(model=m, input_ids=[[1], [2]], kv_bits=8)
    assert b.init_kv_bits is None  # still stripped: batch init must not crash
    assert not m.comp.is_quantized
    m2 = _V4ish()
    ppb(model=m2, input_ids=[[1]], kv_bits=8, draft_kind="mtp")
    assert not m2.comp.is_quantized


def test_pooled_prompt_kv_quant_non_pooled_untouched(monkeypatch):
    ppb = _install_on_fake(monkeypatch)
    m = SimpleNamespace(make_cache=lambda: [SimpleNamespace(offset=0)])
    b = ppb(model=m, input_ids=[[1]], kv_bits=8)
    assert b.init_kv_bits == 8  # passes through to the stock path


def test_pooled_prompt_kv_quant_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_POOLED_KV_QUANT", "0")
    ppb = _install_on_fake(monkeypatch)
    assert not getattr(ppb, "_kq_pooled_prompt_kv", False)
    m = _V4ish()
    b = ppb(model=m, input_ids=[[1]], kv_bits=8)
    assert b.init_kv_bits == 8 and not m.comp.is_quantized


def test_pooled_prompt_kv_quant_idempotent(monkeypatch):
    ppb = _install_on_fake(monkeypatch)
    wrapped = ppb.__init__
    from gmlx.apc_pooling import install_pooled_prompt_kv_quant

    install_pooled_prompt_kv_quant()
    assert ppb.__init__ is wrapped
