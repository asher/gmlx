"""A quantized KV cache that starts empty takes its first append at every
bit width.

``mx.quantize`` packs a head of ``dim`` values into dim * bits / 32 uint32
words. The upstream first allocation sized the code buffer as
dim // (32 // bits), which is wider at bits 3, 5 and 6 for most head
widths, so the first append into an empty cache failed to broadcast. Serve
builds BatchQuantizedKVCache empty, and to_quantized on an empty KVCache
gives an empty QuantizedKVCache. A cache converted after it holds tokens
takes its width from the conversion, so only the empty start failed. Each
case also runs the stock update_and_fetch body, which must fail exactly
where the two widths differ. CPU.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import gmlx.cache.kv_cache as owned
from stock_twin import stock_method

WIDTHS = (2, 3, 4, 5, 6, 8)
# (head_dim, group_size)
SHAPES = ((64, 64), (128, 64), (256, 64), (512, 64), (96, 32))


def _empty(cls, group, bits):
    if cls is owned.BatchQuantizedKVCache:
        return cls([0], group_size=group, bits=bits)
    return cls(group_size=group, bits=bits)


def _appends_from_empty(cls, dim, group, bits) -> bool:
    c = _empty(cls, group, bits)
    k = mx.zeros((1, 4, 8, dim), mx.float16)
    try:
        for _ in range(40):  # past the step boundary, into the grow path
            c.update_and_fetch(k, k)
        mx.eval(c.keys[0])
    except ValueError:
        return False
    return c.keys[0].shape[-2] >= 320


@pytest.mark.parametrize("body", ["owned", "stock"])
@pytest.mark.parametrize("cls", [owned.QuantizedKVCache,
                                 owned.BatchQuantizedKVCache],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize(("dim", "group"), SHAPES)
@pytest.mark.parametrize("bits", WIDTHS)
def test_an_empty_cache_takes_its_first_append(bits, dim, group, cls, body,
                                               monkeypatch):
    if body == "stock":
        monkeypatch.setattr(cls, "update_and_fetch",
                            stock_method(cls, "update_and_fetch"))
    stock_width_holds = dim // (32 // bits) == dim * bits // 32
    ok = _appends_from_empty(cls, dim, group, bits)
    assert ok is (body == "owned" or stock_width_holds)


def test_some_case_fails_on_the_stock_width():
    assert any(dim // (32 // bits) != dim * bits // 32
               for bits in WIDTHS for dim, _ in SHAPES)


@pytest.mark.parametrize(("dim", "group"), SHAPES)
@pytest.mark.parametrize("bits", WIDTHS)
def test_codes_from_empty_match_the_conversion_of_a_filled_cache(dim, group,
                                                                 bits):
    mx.random.seed(0)
    k = mx.random.normal((1, 4, 300, dim)).astype(mx.float16)

    empty = owned.KVCache().to_quantized(group_size=group, bits=bits)
    from_empty, _ = empty.update_and_fetch(k, k)

    full = owned.KVCache()
    full.update_and_fetch(k, k)
    converted = full.to_quantized(group_size=group, bits=bits).keys

    for a, b in zip(from_empty, converted):
        assert a.shape[-1] == b.shape[-1]
        assert mx.array_equal(a[..., :300, :], b[..., :300, :]).item()


def test_a_converted_cache_grows_at_its_width():
    full = owned.KVCache()
    k = mx.zeros((1, 4, 300, 256), mx.float16)
    full.update_and_fetch(k, k)
    q = full.to_quantized(group_size=64, bits=6)
    q.update_and_fetch(k, k)
    mx.eval(q.keys[0])
    assert q.offset == 600 and q.keys[0].shape[-1] == 256 * 6 // 32
