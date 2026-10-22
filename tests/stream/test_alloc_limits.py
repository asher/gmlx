"""The MLX allocator limits a streamed model sets: the memory limit past the
tracked file-backed bytes, the buffer cache capped by GMLX_STREAM_CACHE_GB,
and nothing under GMLX_STREAM_ALLOC_LIMITS=0."""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx.stream import expert_streaming as es


@pytest.fixture
def limits(monkeypatch):
    calls = {}
    monkeypatch.setattr(mx, "set_memory_limit", lambda n: calls.setdefault("memory", n))
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: calls.setdefault("cache", n))
    return calls


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal allocator only")
def test_limits_follow_the_tracked_bytes_and_the_cache_env(limits, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_CACHE_GB", "2")
    es._set_allocator_limits(300 * 10**9)
    assert es._physical_ram() > 2**30
    assert limits["memory"] == 300 * 10**9 + 2 * es._physical_ram()
    assert limits["cache"] == 2 * 2**30


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal allocator only")
def test_cache_is_the_pool_the_kv_room_prices(limits, monkeypatch):
    monkeypatch.delenv("GMLX_STREAM_CACHE_GB", raising=False)
    es._set_allocator_limits(10**9, pool_bytes=11 * 10**9)
    assert limits["cache"] == 11 * 10**9
    limits.clear()
    es._set_allocator_limits(10**9)   # no priced room: the flat 4 GB
    assert limits["cache"] == 4 * 2**30


def test_switch_keeps_the_defaults(limits, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_ALLOC_LIMITS", "0")
    es._set_allocator_limits(10**9)
    assert limits == {}


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal allocator only")
@pytest.mark.parametrize("raw", ["2g", "-1"])
def test_a_bad_cache_env_sets_the_pool_the_room_reads(limits, monkeypatch, raw):
    """The room and the allocator read GMLX_STREAM_CACHE_GB through one
    parser: a value it cannot use leaves the install the pool the room
    priced, not a traceback at load."""
    from gmlx.stream import budget

    monkeypatch.setenv("GMLX_STREAM_CACHE_GB", raw)
    pool = budget.pool_bytes(8 << 30)
    es._set_allocator_limits(10**9, pool_bytes=pool)
    assert limits["cache"] == pool
    limits.clear()
    es._set_allocator_limits(10**9)   # no room
    assert limits["cache"] == budget.pool_bytes(0)
