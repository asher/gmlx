"""B=1 MTP KV_BITS: construction conversion, trim rollback, quantized verify."""

import os

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.generate.ar")
q35l = pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_lm.models.cache import KVCache, QuantizedKVCache  # noqa: E402
from mlx_vlm.generate import ar  # noqa: E402
from mlx_vlm.server import generation as gen  # noqa: E402
from mlx_vlm.speculative import utils as su  # noqa: E402

import gmlx.models.qwen35.verify_fold as qwen35_verify_fold  # noqa: E402
import gmlx.spec.engine as spec_engine  # noqa: E402


class _SSMCache:
    """ArraysCache stand-in: not a KVCache, no to_quantized."""

    def is_trimmable(self):
        return False


class _FakeLM:
    def make_cache(self):
        return [KVCache(), _SSMCache(), KVCache()]


@pytest.fixture
def restorable(monkeypatch):
    # Start from the stock function even when an earlier module installed
    # the wrap (it is once per process, with the boot env it saw), and
    # undo this test's install at teardown.
    for mod in (su, ar, gen):
        fn = mod.make_speculative_prompt_cache
        monkeypatch.setattr(mod, "make_speculative_prompt_cache",
                            getattr(fn, "_gmlx_orig", fn))
    return monkeypatch


def _mk(batch_size=1, make_cache=None, lm=None):
    return ar.make_speculative_prompt_cache(
        lm or _FakeLM(),
        draft_kind="mtp",
        batch_size=batch_size,
        left_padding=[0] * batch_size,
        make_cache=make_cache or (lambda lm, lp: pytest.fail(
            "B=1 mtp bypass must not call make_cache")),
    )


def test_b1_mtp_converts(restorable):
    restorable.setenv("KV_BITS", "4")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert isinstance(caches[0], QuantizedKVCache)
    # The last layer of a deep stack stays fp16. The MTP arm
    # conforms to the batch-path policy.
    assert type(caches[2]) is KVCache
    assert isinstance(caches[1], _SSMCache)
    assert caches[0].bits == 4 and caches[0].group_size == 64
    assert caches[0].offset == 0 and caches[0].is_trimmable()
    # idempotent: second install keeps the same wrapper
    wrapped = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is wrapped


def test_group_size_env(restorable):
    restorable.setenv("KV_BITS", "8")
    restorable.setenv("KV_GROUP_SIZE", "32")
    spec_engine.install_spec_kv_quant()
    caches = _mk()
    assert caches[0].bits == 8 and caches[0].group_size == 32


def test_no_env_installs_and_stays_fp16(restorable):
    # The wrap installs regardless: a later-loaded model may carry a
    # stamped policy even when the boot env asked for nothing.
    restorable.delenv("KV_BITS", raising=False)
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is not before
    assert all(type(c) in (KVCache, _SSMCache) for c in _mk())


def _stamp(lm, single):
    from gmlx.serve.kv_policy import ServeKvPolicy

    lm._gmlx_kv_policy = ServeKvPolicy(single, single)
    return lm


def test_stamped_policy_rules_the_boot_env(restorable):
    # Serve's per-model env window is closed by request time; the stamp
    # left by resolve_for_load is what this model was loaded at.
    from gmlx.cache.kv_policy import off_policy, resolve_kv_quant_policy

    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    fp16 = _stamp(_FakeLM(), off_policy("single"))
    assert all(type(c) in (KVCache, _SSMCache) for c in _mk(lm=fp16))
    narrow = _stamp(_FakeLM(), resolve_kv_quant_policy(
        [KVCache()], kv_bits=4, kv_group_size=32, mode="single"))
    caches = _mk(lm=narrow)
    assert isinstance(caches[0], QuantizedKVCache)
    assert caches[0].bits == 4 and caches[0].group_size == 32
    # unstamped: the boot env still applies
    assert _mk().__getitem__(0).bits == 8


def test_kill_switch(restorable):
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("GMLX_SPEC_KV_QUANT", "0")
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


@pytest.mark.parametrize(
    "bits,scheme", [("4", "turboquant"), ("1.6", "uniform")]
)
def test_non_affine_stays_fp16(restorable, bits, scheme):
    restorable.setenv("KV_BITS", bits)
    restorable.setenv("KV_QUANT_SCHEME", scheme)
    spec_engine.install_spec_kv_quant()
    assert all(type(c) in (KVCache, _SSMCache) for c in _mk())


def test_batch_passthrough(restorable):
    restorable.setenv("KV_BITS", "4")
    spec_engine.install_spec_kv_quant()
    sentinel = ["stock"]
    out = _mk(batch_size=2, make_cache=lambda lm, lp: sentinel)
    assert out is sentinel


def test_batch_forces_fp16(restorable):
    # B>1 MTP swaps stock quantized batch caches for BatchKVCache:
    # the stock rollback misfiles BatchQuantizedKVCache and never
    # trims rejected drafts.
    from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache

    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    lp = [0, 0]
    stock = [BatchQuantizedKVCache(lp, group_size=64, bits=8),
             _SSMCache(),
             BatchQuantizedKVCache(lp, group_size=64, bits=8)]
    out = _mk(batch_size=2, make_cache=lambda lm, _: stock)
    assert type(out[0]) is BatchKVCache
    assert type(out[2]) is BatchKVCache
    assert isinstance(out[1], _SSMCache)


def test_batch_forces_fp16_nested(restorable):
    # to_batch_cache's CacheList arm quantizes nested subcaches with the
    # default quantize=True, so the swap must walk into CacheList entries.
    from mlx_vlm.models.cache import (BatchKVCache, BatchQuantizedKVCache,
                                      CacheList)

    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    lp = [0, 0]
    stock = [CacheList(BatchQuantizedKVCache(lp, group_size=64, bits=8),
                       _SSMCache()),
             BatchQuantizedKVCache(lp, group_size=64, bits=8)]
    out = _mk(batch_size=2, make_cache=lambda lm, _: stock)
    assert type(out[0]) is CacheList
    assert type(out[0].caches[0]) is BatchKVCache
    assert isinstance(out[0].caches[1], _SSMCache)
    assert type(out[1]) is BatchKVCache


def test_dequantize_lift_cache():
    # B=1 quantized caches lift by dequantize, so the batch rebuild
    # proceeds instead of a drain-wait.
    from mlx_vlm.models.cache import BatchKVCache

    mx.random.seed(3)
    k = mx.random.normal((1, 2, 41, 64)).astype(mx.float16)
    v = mx.random.normal((1, 2, 41, 64)).astype(mx.float16)
    q = QuantizedKVCache(group_size=64, bits=8)
    q.update_and_fetch(k, v)
    q._gmlx_cascade = "stamp"
    lifted = spec_engine.dequantize_lift_cache(q)
    assert type(lifted) is BatchKVCache
    assert lifted.offset == 41
    assert lifted._gmlx_cascade == "stamp"
    assert hasattr(lifted, "filter") and hasattr(lifted, "extend")
    with mx.stream(mx.cpu):
        ref = mx.dequantize(*(mx.contiguous(t[..., :41, :])
                              for t in q.keys),
                            group_size=64, bits=8)
    got = lifted.keys[..., :41, :]
    assert mx.abs(got.astype(mx.float32)
                  - ref.astype(mx.float32)).max().item() < 1e-2


def _fill(c, parts):
    for p in parts:
        c.update_and_fetch(p, p)


def test_quantized_trim_rollback_exact():
    # The rollback premise: suffix trim on packed KV is a pure offset move
    # (packing is per-token along head_dim), so trim + re-append lands
    # bit-identically to a straight-line fill.
    mx.random.seed(7)
    k1 = mx.random.normal((1, 2, 40, 64)).astype(mx.bfloat16)
    blk = mx.random.normal((1, 2, 4, 64)).astype(mx.bfloat16)
    k3 = mx.random.normal((1, 2, 3, 64)).astype(mx.bfloat16)
    a = QuantizedKVCache(group_size=64, bits=4)
    _fill(a, [k1, blk])
    assert a.trim(3) == 3 and a.offset == 41
    a.update_and_fetch(k3, k3)
    assert a.offset == 44
    b = QuantizedKVCache(group_size=64, bits=4)
    _fill(b, [k1, blk[:, :, :1, :], k3])
    for xa, xb in zip(a.state, b.state):
        for pa, pb in zip(xa, xb):
            assert mx.array_equal(pa, pb).item()


def test_prefix_cache_quantized_roundtrip():
    # The spec prefix cache must snapshot quantized targets: state entries
    # are (packed, scales, biases) triples, and the old array-only path
    # crashed every MTP store under KV_BITS. Store, mutate the live cache,
    # restore into a fresh one, and the pre-mutation triples must match.
    from gmlx.cache.prefix_cache import SpecPrefixCache

    mx.random.seed(11)
    a = QuantizedKVCache(group_size=64, bits=8)
    _fill(a, [mx.random.normal((1, 2, 40, 64)).astype(mx.bfloat16)])
    ref = [tuple(mx.contiguous(x) for x in side) for side in a.state]
    cache = SpecPrefixCache()
    cache.store(mx.array(list(range(40))), [a], mx.zeros((1, 1, 8)))
    _fill(a, [mx.random.normal((1, 2, 5, 64)).astype(mx.bfloat16)])

    hit = cache.lookup(mx.array(list(range(41))))
    assert hit is not None and hit[0] == 40
    b = QuantizedKVCache(group_size=64, bits=8)
    cache.restore(hit[1], [b])
    assert b.offset == 40
    for got, want in zip((b.keys, b.values), ref):
        for pg, pw in zip(got, want):
            assert mx.array_equal(pg, pw).item()
    # restored cache must keep decoding past the snapshot point
    b.update_and_fetch(*2 * (mx.random.normal((1, 2, 1, 64)).astype(mx.bfloat16),))
    assert b.offset == 41


def _dequant_ref(q, qc, scale):
    # upstream per-token verify loop on dequantized KV
    keys = mx.dequantize(*qc.keys, group_size=qc.group_size, bits=qc.bits)
    values = mx.dequantize(*qc.values, group_size=qc.group_size, bits=qc.bits)
    keys = keys[..., : qc.offset, :].astype(q.dtype)
    values = values[..., : qc.offset, :].astype(q.dtype)
    L = q.shape[2]
    prefix = qc.offset - L
    return mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                q[:, :, i : i + 1, :],
                keys[:, :, : prefix + i + 1, :],
                values[:, :, : prefix + i + 1, :],
                scale=scale,
                mask=None,
            )
            for i in range(L)
        ],
        axis=2,
    )


def test_fold_claims_quantized_b1():
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn = q35l._target_verify_left_padded_attention
    mx.random.seed(3)
    scale = 64**-0.5
    qc = QuantizedKVCache(group_size=64, bits=8)
    prefix = mx.random.normal((1, 2, 512, 64)).astype(mx.bfloat16)
    qc.update_and_fetch(prefix, prefix)
    blk = mx.random.normal((1, 2, 4, 64)).astype(mx.bfloat16)
    keys, values = qc.update_and_fetch(blk, blk)
    q = mx.random.normal((1, 8, 4, 64)).astype(mx.bfloat16)
    # ref FIRST: upstream quantized SDPA scales its queries arg in place
    # (mlx-lm `queries *= scale`), so q is unusable after the fold call.
    ref = _dequant_ref(q, qc, scale)
    mx.eval(ref)
    out = fn(q, keys, values, cache=qc, scale=scale, mask="causal")
    assert out is not None and out.shape == ref.shape
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    # The CPU backend's quantized SDPA accumulates in a different order:
    # 0.027 max err on this seed vs <=0.02 on Metal. Same quantization, both
    # within q8 noise; keep the Metal bound tight.
    tol = 3e-2 if os.environ.get("KQUANT_FORCE_CPU") else 2e-2
    assert err < tol, f"quantized fold err={err}"


def test_batch_sdpa_tuple_defers():
    assert qwen35_verify_fold.install_qwen35_verify_fold()
    fn = q35l.scaled_dot_product_attention
    orig = fn._gmlx_orig
    mx.random.seed(5)
    qc = QuantizedKVCache(group_size=64, bits=8)
    pref = mx.random.normal((2, 2, 128, 64)).astype(mx.bfloat16)
    keys, values = qc.update_and_fetch(pref, pref)
    q = mx.random.normal((2, 8, 1, 64)).astype(mx.bfloat16)
    # fresh copy per call: quantized SDPA scales queries in place upstream
    out = fn(q + 0, keys, values, cache=qc, scale=64**-0.5, mask=None)
    ref = orig(q + 0, keys, values, cache=qc, scale=64**-0.5, mask=None)
    assert mx.array_equal(out, ref).item()


class _GdnFakeLM(_FakeLM):
    model_type = "qwen3_5"


class _GdnConfigFakeLM(_FakeLM):
    # model_type only on config: the loader wrapper shape
    class config:
        model_type = "qwen3_5_moe"


@pytest.mark.parametrize("lm_cls", [_GdnFakeLM, _GdnConfigFakeLM])
def test_owned_off_gdn_declines_quantization(restorable, lm_cls):
    # The stock fallback cannot verify on quantized KV tuples. The
    # guard keys on model_type, direct or config fallback.
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("GMLX_QWEN_OWNED", "0")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        lm_cls(), draft_kind="mtp", batch_size=1, left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail(
            "B=1 mtp bypass must not call make_cache"),
    )
    assert not any(isinstance(c, QuantizedKVCache) for c in caches)


def test_owned_on_gdn_still_converts(restorable):
    restorable.setenv("KV_BITS", "4")
    restorable.setenv("GMLX_QWEN_OWNED", "1")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _GdnFakeLM(), draft_kind="mtp", batch_size=1, left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail(
            "B=1 mtp bypass must not call make_cache"),
    )
    assert isinstance(caches[0], QuantizedKVCache)


class _Stamp:
    pass


def test_warm_merge_config_follows_batched_policy(restorable):
    # The warm merge follows the batched verdict stamped on the
    # model, never the environment. MTP models drop kv when batched.
    from gmlx.cache.kv_policy import dropped_policy, resolve_kv_quant_policy
    from gmlx.serve.kv_policy import ServeKvPolicy

    restorable.setenv("KV_BITS", "8")   # env says quantize; stamp wins

    model = _Stamp()
    single = resolve_kv_quant_policy([KVCache()], kv_bits=8,
                                     kv_group_size=64, mode="single")
    batched_drop = dropped_policy("mtp fp16 when batched", 8, 64, "batched")
    model._gmlx_kv_policy = ServeKvPolicy(single, batched_drop)
    assert spec_engine._live_kv_quant_config(model) is None

    batched_full = resolve_kv_quant_policy([KVCache()], kv_bits=8,
                                           kv_group_size=64, mode="batched")
    model._gmlx_kv_policy = ServeKvPolicy(single, batched_full)
    assert spec_engine._live_kv_quant_config(model) is not None

    # no stamp: fail-safe None, never the environment
    assert spec_engine._live_kv_quant_config(_Stamp()) is None
    assert spec_engine._live_kv_quant_config(None) is None


# Hybrid arch shapes on the B=1 arm. CacheList members, opt-outs, nested
# windows, and top-level windows must all follow the shared policy.


class _OptOutKVCache(KVCache):
    kv_quant_unsupported = True


def _cache_list(*inner):
    from mlx_vlm.models.cache import CacheList

    return CacheList(*inner)


class _ListFakeLM:
    """glm5_next's shape: CacheList(KVCache, opted-out pool) per layer."""

    def __init__(self, n=3):
        self.n = n

    def make_cache(self):
        return [_cache_list(KVCache(), _SSMCache()) for _ in range(self.n)]


def test_b1_mtp_quantizes_the_kv_member_of_a_cache_list(restorable):
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _ListFakeLM(), draft_kind="mtp", batch_size=1, left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    # The last layer of a deep stack stays fp16, as everywhere else.
    inner = caches[0].caches[0]
    assert isinstance(inner, QuantizedKVCache), (
        "the KV member of a CacheList layer stayed fp16 while the shared "
        "policy reports the layer as quantized")
    assert inner.bits == 8
    assert isinstance(caches[0].caches[1], _SSMCache)


class _OptOutFakeLM:
    def make_cache(self):
        return [_OptOutKVCache(), KVCache(), KVCache()]


def test_b1_mtp_honors_kv_quant_unsupported(restorable):
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _OptOutFakeLM(), draft_kind="mtp", batch_size=1, left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    assert not isinstance(caches[0], QuantizedKVCache), (
        "a cache declaring kv_quant_unsupported must never be converted")
    assert isinstance(caches[1], QuantizedKVCache)


class _NestedWindowFakeLM:
    def make_cache(self):
        from mlx_lm.models.cache import RotatingKVCache

        return [_cache_list(RotatingKVCache(max_size=16), _SSMCache())
                for _ in range(3)]


def test_b1_mtp_sees_a_window_nested_in_a_cache_list(restorable):
    # A window-plus-state list classifies as state. Nothing converts.
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _NestedWindowFakeLM(), draft_kind="mtp", batch_size=1,
        left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    for c in caches:
        assert not isinstance(c.caches[0], QuantizedKVCache)


class _Glm5ShapeFakeLM:
    """glm5_next's real shape: CacheList(KVCache, PoolingCache) per layer."""

    def make_cache(self):
        from gmlx.models.deepseek_v4.cache import PoolingCache

        return [_cache_list(KVCache(), PoolingCache(4)) for _ in range(3)]


def test_b1_mtp_arms_the_pool_beside_the_kv_member(restorable, caplog):
    # A kv member rules the list, so pool arming must not key off the
    # layer kind. Every layer's pool packs, the held last layer included.
    import logging

    caplog.set_level(logging.INFO, logger="gmlx.spec.engine")
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _Glm5ShapeFakeLM(), draft_kind="mtp", batch_size=1,
        left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    assert isinstance(caches[0].caches[0], QuantizedKVCache)
    assert type(caches[2].caches[0]) is KVCache
    for c in caches:
        assert c.caches[1].is_quantized, (
            "the pool member of a kv-ruled CacheList stayed fp16")
    assert ("[kv] MTP spec path: kv_bits=8 group=64 -> 3 pooled at rest; "
            "quantized 2/3 attn layers (1 held fp16)") in caplog.text


class _PoolOnlyFakeLM:
    """No growing KV at all: CacheList(state, PoolingCache) per layer."""

    def make_cache(self):
        from gmlx.models.deepseek_v4.cache import PoolingCache

        return [_cache_list(_SSMCache(), PoolingCache(4)) for _ in range(2)]


def test_b1_mtp_notes_a_pool_only_engagement(restorable, caplog):
    # Nothing converts, so the note must key off the pools armed.
    import logging

    caplog.set_level(logging.INFO, logger="gmlx.spec.engine")
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _PoolOnlyFakeLM(), draft_kind="mtp", batch_size=1,
        left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    for c in caches:
        assert c.caches[1].is_quantized
    assert "2 pooled at rest; quantized 0/2 attn layers" in caplog.text


class _HybridFakeLM:
    def make_cache(self):
        from mlx_lm.models.cache import RotatingKVCache

        return [RotatingKVCache(max_size=16), KVCache(),
                RotatingKVCache(max_size=16), KVCache()]


def test_b1_mtp_hybrid_stack_quantizes_full_attn_layers(restorable):
    # Top-level windows must not drop the whole stack: the verdict is
    # partial, windows stay fp16, full-attention layers convert.
    restorable.setenv("KV_BITS", "8")
    spec_engine.install_spec_kv_quant()
    caches = ar.make_speculative_prompt_cache(
        _HybridFakeLM(), draft_kind="mtp", batch_size=1, left_padding=[0],
        make_cache=lambda lm, lp: pytest.fail("B=1 mtp bypass"),
    )
    assert isinstance(caches[1], QuantizedKVCache)
    assert caches[1].bits == 8
    for i in (0, 2):
        assert not isinstance(caches[i], QuantizedKVCache)
    # the last layer of a deep stack stays fp16
    assert type(caches[3]) is KVCache


def test_mtp_kv_decline_is_shared_by_serve_run_and_chat(restorable):
    """serve, run and chat resolve the same predicate. The CLI dropped
    kv_bits under MTP while serve quantized the same stack; the two
    stock verify walks are the only real declines."""
    restorable.setenv("GMLX_QWEN_OWNED", "1")
    assert spec_engine.mtp_kv_decline(_FakeLM()) is None
    assert spec_engine.mtp_kv_decline(_GdnFakeLM()) is None

    restorable.setenv("GMLX_QWEN_OWNED", "0")
    assert "stock fallback" in spec_engine.mtp_kv_decline(_GdnFakeLM())
    assert "stock fallback" in spec_engine.mtp_kv_decline(_GdnConfigFakeLM())
    assert spec_engine.mtp_kv_decline(_FakeLM()) is None

    restorable.setenv("GMLX_QWEN_OWNED", "1")
    reason = spec_engine.mtp_kv_decline(_FakeLM(), owned_round=False)
    assert "GMLX_OWNED_ROUND=0" in reason


def test_injected_quantized_row_lifts_to_fp16_batch():
    """A B=1 affine row admitted into a live MTP batch lifts by
    dequantize; QuantizedKVCache has no merge, so the class-merge lift
    raised on every concurrent admission under kv-bits."""
    from mlx_vlm.models.cache import BatchKVCache

    from gmlx.spec.speculative import _lift_injected_cache, _lift_live_cache

    mx.random.seed(5)
    k = mx.random.normal((1, 2, 40, 64)).astype(mx.float16)
    live = BatchKVCache.merge([_filled_kv(k)])
    q = QuantizedKVCache(group_size=64, bits=8)
    q.update_and_fetch(k, k)
    lifted = _lift_injected_cache(live, q)
    assert type(lifted) is BatchKVCache
    live.extend(lifted)
    assert live.keys.shape[0] == 2
    assert live.offset.tolist() == [40, 40]
    # the live side too: a batch formed from one packed row
    q2 = QuantizedKVCache(group_size=64, bits=8)
    q2.update_and_fetch(k, k)
    assert type(_lift_live_cache(q2)) is BatchKVCache


def _filled_kv(k):
    c = KVCache()
    c.update_and_fetch(k, k)
    return c
