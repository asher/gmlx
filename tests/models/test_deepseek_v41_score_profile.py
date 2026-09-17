"""deepseek_v41 prefill score profile: arming conditions, base step, decay."""

from types import SimpleNamespace

import pytest

import gmlx.gen.prefill_decay as pd
import gmlx.models.deepseek_v4.model as dsv4
import gmlx.models.deepseek_v41.model as dsv41
from gmlx.models.deepseek_v4.cache import BatchPoolingCache, PoolingCache


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setitem(dsv4._dsa_state, "indexer", True)
    monkeypatch.setitem(dsv4._SPARSE_KERNEL, "on", True)
    monkeypatch.delenv("GMLX_DS41_PREFILL_STEP", raising=False)


def _sparse_layer(ratio=1):
    # make_cache shape on a kv source layer: rotating window, latent pool,
    # indexer pool
    local = SimpleNamespace(offset=1000)
    return SimpleNamespace(
        caches=[local, PoolingCache(ratio), PoolingCache(ratio)])


def _resident(streamed: bool):
    """A model whose residency moe_streaming_active reports as `streamed`."""
    inner = SimpleNamespace(modules=lambda: [
        SimpleNamespace(_kq_cpu_only=streamed)])
    return SimpleNamespace(layers=[inner])


def test_registered_on_import():
    assert pd._SCORE_PROFILES["deepseek_v41"] is dsv41._prefill_score_profile


def test_armed_in_ram_profile(armed):
    prof = dsv41._prefill_score_profile(_resident(False), [_sparse_layer()])
    assert prof == pd.ScoreTransientProfile(
        heads=1, bytes_per_elem=4, depth_divisor=1, base_step=4096)


def test_streamed_model_takes_the_wider_base(armed):
    prof = dsv41._prefill_score_profile(_resident(True), [_sparse_layer()])
    assert prof.base_step == 8192
    # only the base step moves; the transient shape is residency-independent
    assert (prof.heads, prof.bytes_per_elem, prof.depth_divisor) == (1, 4, 1)


def test_indexer_disarmed_returns_none(armed, monkeypatch):
    monkeypatch.setitem(dsv4._dsa_state, "indexer", False)
    assert dsv41._prefill_score_profile(
        _resident(True), [_sparse_layer()]) is None


def test_sparse_kernel_disarmed_returns_none(armed, monkeypatch):
    monkeypatch.setitem(dsv4._SPARSE_KERNEL, "on", False)
    assert dsv41._prefill_score_profile(
        _resident(True), [_sparse_layer()]) is None


def test_batched_caches_return_none(armed):
    batched = SimpleNamespace(
        caches=[SimpleNamespace(offset=0), BatchPoolingCache(1, [0, 0])])
    assert dsv41._prefill_score_profile(_resident(True), [batched]) is None


def test_quantized_pool_keeps_profile(armed):
    pool = PoolingCache(1)
    pool.quantize_storage(64, 8)
    layer = SimpleNamespace(caches=[SimpleNamespace(offset=0), pool])
    assert dsv41._prefill_score_profile(
        _resident(False), [layer]).base_step == 4096


def test_quantized_local_window_returns_none(armed):
    layer = SimpleNamespace(
        caches=[SimpleNamespace(offset=0, bits=8), PoolingCache(1)])
    assert dsv41._prefill_score_profile(_resident(True), [layer]) is None


def test_no_pools_returns_none(armed):
    assert dsv41._prefill_score_profile(
        _resident(True), [SimpleNamespace(offset=1000)]) is None
    assert dsv41._prefill_score_profile(_resident(True), None) is None


def test_base_step_env_override(armed, monkeypatch):
    caches = [_sparse_layer()]
    monkeypatch.setenv("GMLX_DS41_PREFILL_STEP", "2048")
    assert dsv41._prefill_score_profile(
        _resident(True), caches).base_step == 2048
    monkeypatch.setenv("GMLX_DS41_PREFILL_STEP", "0")
    assert dsv41._prefill_score_profile(
        _resident(True), caches).base_step is None
    monkeypatch.setenv("GMLX_DS41_PREFILL_STEP", "banana")
    assert dsv41._prefill_score_profile(
        _resident(True), caches).base_step is None


def _batch(depth, streamed=True):
    return SimpleNamespace(
        prefill_step_size=2048,
        prompt_cache=[SimpleNamespace(offset=depth), _sparse_layer()],
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41",
                                   num_attention_heads=64),
            layers=_resident(streamed).layers),
    )


def test_streamed_base_reaches_the_batch(armed, monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "6.0")
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    # dense modeling at 64 heads would floor this chunk; the armed profile
    # holds the streaming base at ladder depths
    assert pd.decayed_for_batch(_batch(8192)) == 8192
    assert pd.decayed_for_batch(_batch(67_000)) == 8192


def test_deep_prompt_decays_the_streamed_base(armed, monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "6.0")
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    # the guard: the transient at 8192 outgrows the cap past ~200k, so the
    # chunk halves rather than the box running out of room
    assert pd.decayed_for_batch(_batch(200_000)) == 4096
    assert pd.decayed_for_batch(_batch(500_000)) == 2048


def test_explicit_step_stays_authoritative(armed, monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_SCORE_CAP_GB", "6.0")
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2048")
    assert pd.decayed_for_batch(_batch(8192)) == 2048
