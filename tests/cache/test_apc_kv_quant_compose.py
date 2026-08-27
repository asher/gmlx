"""Quantized-KV + APC composition (0.6.15 alignment retest).

Upstream 0.6.15 lifted the old mutual exclusion: exact snapshots store
float (snapshot_prompt_cache_row dequantizes), and the warm merge
re-quantizes to the live policy when kv_quant_config is passed. These
tests pin the seams gmlx rides so the owned L1 merge stays aligned.
"""

import mlx.core as mx

import gmlx.spec.engine as spec_engine


def test_live_kv_quant_config_off_without_env(monkeypatch):
    monkeypatch.delenv("KV_BITS", raising=False)
    assert spec_engine._live_kv_quant_config() is None
    monkeypatch.setenv("KV_BITS", "not-a-number")
    assert spec_engine._live_kv_quant_config() is None
    monkeypatch.setenv("KV_BITS", "0")
    assert spec_engine._live_kv_quant_config() is None


def test_live_kv_quant_config_reads_serve_env(monkeypatch):
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.setenv("KV_GROUP_SIZE", "32")
    monkeypatch.delenv("KV_QUANT_SCHEME", raising=False)
    cfg = spec_engine._live_kv_quant_config()
    assert cfg is not None
    assert float(cfg["bits"]) == 8.0 and int(cfg["group_size"]) == 32


def test_warm_merge_requantizes_float_row(monkeypatch):
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
    monkeypatch.setenv("KV_BITS", "8")
    monkeypatch.setenv("KV_GROUP_SIZE", "32")
    cfg = spec_engine._live_kv_quant_config()
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


def test_store_seam_dequantizes_quantized_rows():
    # gmlx's anchor store goes through _apc_prompt_cache_for_store ->
    # snapshot_prompt_cache_row; a quantized single-row cache must come
    # back as a float snapshot (stored entries stay float by contract).
    from mlx_vlm import apc
    from mlx_vlm.models.cache import QuantizedKVCache

    c = QuantizedKVCache(group_size=32, bits=8)
    k = mx.random.normal((1, 2, 64, 64))
    c.update_and_fetch(k, k)
    snap = apc.snapshot_prompt_cache_row([c])
    assert snap is not None
    nm = type(snap[0]).__name__
    assert "Quantized" not in nm, nm
    assert int(snap[0].offset) == 64
