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
    # Identity-setattr records the current attrs so the install's direct
    # module assignment is undone at teardown.
    for mod in (su, ar, gen):
        monkeypatch.setattr(
            mod, "make_speculative_prompt_cache",
            mod.make_speculative_prompt_cache)
    return monkeypatch


def _mk(batch_size=1, make_cache=None):
    return ar.make_speculative_prompt_cache(
        _FakeLM(),
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
    # Serve carve-out: the last layer of a deep stack stays fp16
    # (should_quantize_kv_layer); the MTP arm conforms to the same
    # policy as the batch path.
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


def test_no_env_no_patch(restorable):
    restorable.delenv("KV_BITS", raising=False)
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


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
    before = su.make_speculative_prompt_cache
    spec_engine.install_spec_kv_quant()
    assert su.make_speculative_prompt_cache is before


def test_batch_passthrough(restorable):
    restorable.setenv("KV_BITS", "4")
    spec_engine.install_spec_kv_quant()
    sentinel = ["stock"]
    out = _mk(batch_size=2, make_cache=lambda lm, lp: sentinel)
    assert out is sentinel


def test_batch_forces_fp16(restorable):
    # B>1 MTP: quantized batch caches from the stock builder are swapped
    # for BatchKVCache. The stock rollback misfiles BatchQuantizedKVCache
    # as an SSM cache (not trimmable, no zero_row_tail): rejected drafts
    # are never trimmed and the state pairing shifts.
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
    # Preempt/injection lift for B=1 quantized caches: same fp16
    # conversion the B>1 swap performs, so the batch rebuild proceeds
    # instead of declining into a drain-wait.
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
    # GMLX_QWEN_OWNED=0 stock fallback cannot verify on quantized KV
    # tuples (issue #104 second symptom); the guard keys on model_type
    # (direct or config fallback), not the module name.
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
    # The APC warm path joins a batch: its merge config must follow the
    # BATCHED policy verdict stamped on the model, not the environment.
    # MTP models drop kv when batched; a quantized warm merge would
    # rebuild the misfiled-SSM rollback crash.
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

    # no stamp: fail-safe None. The sole caller is the MTP prefill
    # init, where fp16 is the only correct merge; an env fallback here
    # would quantize exactly the merge the policy exists to protect.
    assert spec_engine._live_kv_quant_config(_Stamp()) is None
    assert spec_engine._live_kv_quant_config(None) is None
