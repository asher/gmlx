"""Quantized-KV first-allocation pack width (upstream bits-3/6 bug)."""

from __future__ import annotations

import pytest

import mlx.core as mx

pytest.importorskip("mlx_vlm")

from mlx_vlm.models.cache import (  # noqa: E402
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
)

from gmlx.upstream.quantized_cache_pack_fix import (  # noqa: E402
    install_quantized_cache_pack_fix,
)

# The loader installs it; import order in the suite is not guaranteed.
install_quantized_cache_pack_fix()

WIDTHS = (2, 3, 4, 6, 8)
DIMS = (64, 128, 256, 512)


def test_install_is_idempotent():
    assert install_quantized_cache_pack_fix() is False


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_batch_cache_appends_from_empty(dim, bits):
    """Serve builds the batch cache empty, so the first append allocates.
    At bits 3 and 6 the stock formula sized it too wide and raised."""
    c = BatchQuantizedKVCache([0], group_size=64, bits=bits)
    k = mx.zeros((1, 4, 8, dim), mx.float16)
    for _ in range(4):  # past the step boundary, into the grow path
        c.update_and_fetch(k, k)
    mx.eval(c.keys[0])
    assert c.keys[0].shape[-2] >= c._idx


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("bits", WIDTHS)
def test_seeded_codes_match_the_mid_stream_conversion(dim, bits):
    """The seeded buffers must hold the same codes the always-working
    populated-then-converted path produces."""
    mx.random.seed(0)
    k = mx.random.normal((1, 4, 300, dim)).astype(mx.float16)

    empty = KVCache().to_quantized(group_size=64, bits=bits)
    seeded, _ = empty.update_and_fetch(k, k)

    full = KVCache()
    full.update_and_fetch(k, k)
    mid = full.to_quantized(group_size=64, bits=bits).keys

    for a, b in zip(seeded, mid):
        assert mx.array_equal(a[..., :300, :], b[..., :300, :]).item()


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_QKV_PACK_FIX", "0")
    assert install_quantized_cache_pack_fix() is False


def test_wrapped_single_cache_still_grows():
    q = QuantizedKVCache(group_size=64, bits=6)
    k = mx.zeros((1, 4, 300, 256), mx.float16)
    q.update_and_fetch(k, k)
    q.update_and_fetch(k, k)
    mx.eval(q.keys[0])
    assert q.offset == 600
