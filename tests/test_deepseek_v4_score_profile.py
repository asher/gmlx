"""deepseek_v4 prefill score profile: arming conditions + decay wiring."""

import sys
import types
from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx import deepseek_v4_model as dsv4
from gmlx import prefill_decay as pd
from gmlx.deepseek_v4_cache import BatchPoolingCache, PoolingCache


@pytest.fixture
def armed(monkeypatch):
    for path in ("window", "sparse", "indexer"):
        monkeypatch.setitem(dsv4._dsa_state, path, True)


def _cache(*layer_caches):
    # deepseek4 make_cache shape: bare RotatingKVCache for local layers,
    # CacheList(local, pool[, idx_pool]) for the rest.
    return list(layer_caches)


def _sparse_layer(ratio=4):
    local = SimpleNamespace(offset=1000)
    return SimpleNamespace(caches=[local, PoolingCache(ratio), PoolingCache(ratio)])


def test_registered_on_import():
    assert pd._SCORE_PROFILES["deepseek_v4"] is dsv4._prefill_score_profile


def test_armed_single_sequence_profile(armed):
    prof = dsv4._prefill_score_profile(
        None, _cache(SimpleNamespace(offset=1000), _sparse_layer()))
    assert prof == pd.ScoreTransientProfile(
        heads=1, bytes_per_elem=4, depth_divisor=4, base_step=4096)


def test_kernels_disarmed_returns_none(armed, monkeypatch):
    caches = _cache(_sparse_layer())
    monkeypatch.setitem(dsv4._dsa_state, "indexer", False)
    assert dsv4._prefill_score_profile(None, caches) is None
    monkeypatch.setitem(dsv4._dsa_state, "indexer", True)
    monkeypatch.setitem(dsv4._dsa_state, "window", False)
    assert dsv4._prefill_score_profile(None, caches) is None


def test_batched_caches_return_none(armed):
    batched = SimpleNamespace(
        caches=[SimpleNamespace(offset=0), BatchPoolingCache(4, [0, 0])])
    assert dsv4._prefill_score_profile(None, _cache(batched)) is None


def test_quantized_pool_keeps_profile(armed):
    # prefill dequantizes pooled rows on read and runs the same kernels; the
    # per-layer fp16 pool copy is step-independent, so the profile stays
    p = PoolingCache(4)
    p.quantize_storage(64, 8)
    layer = SimpleNamespace(caches=[SimpleNamespace(offset=0), p])
    assert dsv4._prefill_score_profile(None, _cache(layer)) == \
        pd.ScoreTransientProfile(heads=1, bytes_per_elem=4, depth_divisor=4,
                                 base_step=4096)


def test_quantized_comp_pool_fp16_indexer_pool_arms(armed):
    # the real quantize_pooled_caches outcome: comp pool packed, indexer pool
    # (quantizable=False) and local window fp16
    comp = PoolingCache(4)
    comp.quantize_storage(64, 8)
    idx = PoolingCache(4)
    layer = SimpleNamespace(caches=[SimpleNamespace(offset=0), comp, idx])
    prof = dsv4._prefill_score_profile(None, _cache(layer))
    assert prof is not None and prof.depth_divisor == 4


def test_quantized_local_returns_none(armed):
    layer = SimpleNamespace(
        caches=[SimpleNamespace(offset=0, bits=8), PoolingCache(4)])
    assert dsv4._prefill_score_profile(None, _cache(layer)) is None


def test_no_pools_returns_none(armed):
    assert dsv4._prefill_score_profile(
        None, _cache(SimpleNamespace(offset=1000))) is None
    assert dsv4._prefill_score_profile(None, None) is None


def test_resolves_through_decay_for_model_type(armed, monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "5.0")
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    batch = SimpleNamespace(
        prefill_step_size=2048,
        prompt_cache=_cache(SimpleNamespace(offset=200_000), _sparse_layer()),
        model=SimpleNamespace(config=SimpleNamespace(
            model_type="deepseek_v4", num_attention_heads=64)),
    )
    # dense modeling would floor this to 256; the armed profile holds full
    # chunks and swaps in the arch-default 4096 base
    assert pd.decayed_for_batch(batch) == 4096
    # an explicit step keeps the stock base authoritative
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert pd.decayed_for_batch(batch) == 2048


def test_base_step_env_override(armed, monkeypatch):
    caches = _cache(_sparse_layer())
    # arch default (probe-certified 4096) flows through when env unset
    assert dsv4._prefill_score_profile(None, caches).base_step == 4096
    monkeypatch.setenv("GMLX_V4_PREFILL_STEP", "2048")
    assert dsv4._prefill_score_profile(None, caches).base_step == 2048
    monkeypatch.setenv("GMLX_V4_PREFILL_STEP", "0")
    assert dsv4._prefill_score_profile(None, caches).base_step is None
    monkeypatch.setenv("GMLX_V4_PREFILL_STEP", "banana")
    assert dsv4._prefill_score_profile(None, caches).base_step is None
    monkeypatch.delenv("GMLX_V4_PREFILL_STEP")
    monkeypatch.setattr(dsv4, "_V4_BASE_STEP", None)
    assert dsv4._prefill_score_profile(None, caches).base_step is None


# indexer kernel arm: unaligned-width padding


def _stub_kq(rec):
    stub = types.ModuleType("mlx_kquant")

    def dsa_indexer_scores(q, keys, weights, causal=False):
        rec["q"] = q.shape
        rec["keys"] = keys.shape
        rec["weights"] = weights.shape
        return mx.zeros(
            (q.shape[0], 1, q.shape[2], keys.shape[2]), dtype=mx.float32)

    def dsa_topk_indices(scores, k, bucketed=False):
        rec["scores"] = scores.shape
        return mx.zeros(
            scores.shape[:-1] + (k,), dtype=mx.uint32)

    stub.dsa_indexer_scores = dsa_indexer_scores
    stub.dsa_topk_indices = dsa_topk_indices
    return stub


def _indexer_self(h=64, d=128):
    return SimpleNamespace(
        n_heads=h,
        weights_proj=lambda x: mx.zeros(
            (x.shape[0], x.shape[1], h), dtype=x.dtype),
        compressor=SimpleNamespace(compress_ratio=4),
    )


def _topk(monkeypatch, L, P, k=512, pmask=True, offset=None):
    rec = {}
    monkeypatch.setitem(sys.modules, "mlx_kquant", _stub_kq(rec))
    q = mx.zeros((1, 64, L, 128), dtype=mx.bfloat16)
    x = mx.zeros((1, L, 8), dtype=mx.bfloat16)
    pooled = mx.zeros((1, P, 128), dtype=mx.bfloat16)
    pm = mx.ones((L, P), dtype=mx.bool_) if pmask else None
    out = dsv4.Indexer._kernel_topk(
        _indexer_self(), x, q, pooled, pm, k, offset)
    return out, rec


def test_kernel_topk_pads_unaligned_width(monkeypatch):
    out, rec = _topk(monkeypatch, L=1500, P=4096)
    assert rec["q"] == (1, 64, 1536, 128)  # padded to %64
    assert rec["weights"] == (1, 1536, 64)
    assert rec["keys"] == (1, 1, 4096, 128)
    assert rec["scores"] == (1, 1, 1500, 4096)  # sliced back before top-k
    assert out.shape == (1, 1500, 512)


def test_kernel_topk_aligned_width_unpadded(monkeypatch):
    out, rec = _topk(monkeypatch, L=128, P=4096)
    assert rec["q"] == (1, 64, 128, 128)  # no padding: identical call
    assert rec["weights"] == (1, 128, 64)
    assert out.shape == (1, 128, 512)


def test_kernel_topk_small_tail_arms_on_deep_pool(monkeypatch):
    out, rec = _topk(monkeypatch, L=63, P=4096)
    assert rec["q"] == (1, 64, 64, 128)
    assert out.shape == (1, 63, 512)


def test_kernel_topk_small_tail_shallow_pool_inline(monkeypatch):
    out, _ = _topk(monkeypatch, L=63, P=1024)
    assert out is None


def test_kernel_topk_decode_widths_stay_inline(monkeypatch):
    # L <= 4 belongs to the decode arm / inline path, never the padded GEMM
    out, _ = _topk(monkeypatch, L=4, P=4096)
    assert out is None
