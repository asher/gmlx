"""The MLX allocator limits a streamed model sets: the gc limit past the
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


def test_switch_keeps_the_defaults(limits, monkeypatch):
    monkeypatch.setenv("GMLX_STREAM_ALLOC_LIMITS", "0")
    es._set_allocator_limits(10**9)
    assert limits == {}
