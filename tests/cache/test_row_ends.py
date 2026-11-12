"""Row ends: an armed batch cache keeps each row's end through an uneven
rollback (gmlx/cache/row_ends.py). Nothing moves: the rollback lowers the
ends, the next append writes each row at its own end, and attention reads
each row's keys up to its end.

Each test builds a batch through the cache's own calls (a left-padded
prefill, a verify block, the rollback the qwen3.5 target runs) and checks a
row against the same row's tokens alone. The memory tests need the Metal
device and skip under KQUANT_FORCE_CPU."""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from gmlx.cache import kv_cache as owned
from gmlx.cache import row_ends
from gmlx.models.qwen35.language import rollback_attention_caches

metal = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="allocation and the kq route run on the Metal device")

H, D = 2, 64
PADS = [5, 0, 2, 7]          # four rows of different lengths
T = 40                       # prefill width, pads included
BS = 3                       # verify block


def _kv(n, seed, b=len(PADS), dtype=mx.float16):
    k = mx.random.normal((b, H, n, D), key=mx.random.key(seed)).astype(dtype)
    v = mx.random.normal((b, H, n, D), key=mx.random.key(seed + 1)).astype(dtype)
    mx.eval(k, v)
    return k, v


def _batch(accepted, armed=True, dtype=mx.float16):
    """A batch after a prefill, one verify block and its uneven rollback.
    Returns the cache and each row's valid keys and values in order."""
    c = owned.BatchKVCache(PADS)
    c._row_ends_armed = armed
    pk, pv = _kv(T, 0, dtype=dtype)
    c.update_and_fetch(pk, pv)
    vk, vv = _kv(BS, 10, dtype=dtype)
    c.update_and_fetch(vk, vv)
    rollback_attention_caches([c], list(accepted), BS)
    rows = []
    for i, (p, a) in enumerate(zip(PADS, accepted)):
        rows.append((mx.concatenate([pk[i:i + 1, :, p:], vk[i:i + 1, :, :a + 1]], axis=2),
                     mx.concatenate([pv[i:i + 1, :, p:], vv[i:i + 1, :, :a + 1]], axis=2)))
    return c, rows


def _row_keys(c, i):
    """Row ``i``'s valid keys and values as the cache holds them."""
    start = int(c.left_padding[i].item())
    end = c._idx if c._row_ends is None else c._row_ends[i]
    return c.keys[i:i + 1, :, start:end], c.values[i:i + 1, :, start:end]


def _reference(q, k, v, scale):
    """One row's attention over its own keys, the query block causal at the
    end."""
    qL, n = q.shape[2], k.shape[2]
    mask = mx.arange(n)[None, :] <= (n - qL + mx.arange(qL))[:, None]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def test_an_armed_rollback_lowers_each_rows_end_and_moves_nothing():
    c, rows = _batch([2, 0, 1, 2])
    before = np.asarray(c.keys)
    assert c._row_ends is not None
    assert c._idx == T + BS
    assert c._row_ends == [T + BS, T + 1, T + 2, T + BS]
    assert c.offset.tolist() == [e - p for e, p in zip(c._row_ends, PADS)]
    assert c.left_padding.tolist() == PADS
    for i, (k, v) in enumerate(rows):
        got_k, got_v = _row_keys(c, i)
        np.testing.assert_array_equal(np.asarray(got_k), np.asarray(k))
        np.testing.assert_array_equal(np.asarray(got_v), np.asarray(v))
    np.testing.assert_array_equal(np.asarray(c.keys), before)


def test_an_unarmed_rollback_still_rolls_the_rows_right():
    c, rows = _batch([2, 0, 1, 2], armed=False)
    assert c._row_ends is None
    assert c.left_padding.tolist() == [5, 2, 3, 7]
    for i, (k, _) in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(_row_keys(c, i)[0]), np.asarray(k))


def test_the_next_append_writes_each_row_at_its_own_end():
    c, rows = _batch([2, 0, 1, 2])
    nk, nv = _kv(BS, 20)
    c.update_and_fetch(nk, nv)
    assert c._row_ends == [T + 2 * BS, T + 1 + BS, T + 2 + BS, T + 2 * BS]
    for i, (k, v) in enumerate(rows):
        got_k, got_v = _row_keys(c, i)
        np.testing.assert_array_equal(
            np.asarray(got_k), np.asarray(mx.concatenate([k, nk[i:i + 1]], axis=2)))
        np.testing.assert_array_equal(
            np.asarray(got_v), np.asarray(mx.concatenate([v, nv[i:i + 1]], axis=2)))


def test_equal_ends_go_back_to_right_justified_rows():
    c, _ = _batch([1, 1, 1, 1])
    assert c._row_ends is None
    assert c._idx == T + 2


def test_filter_join_and_extract_keep_each_rows_tokens():
    c, rows = _batch([2, 0, 1, 2])
    c.filter(mx.array([1, 2], dtype=mx.int32))
    rows = [rows[1], rows[2]]
    assert c.left_padding.tolist() == [0, 2]
    for i, (k, _) in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(_row_keys(c, i)[0]), np.asarray(k))
    # a joiner longer than the batch: the batch's rows move right with
    # their ends
    j = owned.BatchKVCache([0])
    jk, jv = _kv(T + 9, 30, b=1)
    j.update_and_fetch(jk, jv)
    c.extend(j)
    rows.append((jk, jv))
    assert c._row_ends is not None and c._idx == T + 9
    for i, (k, v) in enumerate(rows):
        np.testing.assert_array_equal(np.asarray(_row_keys(c, i)[0]), np.asarray(k))
        out = c.extract(i)
        np.testing.assert_array_equal(np.asarray(out.keys), np.asarray(k))
        np.testing.assert_array_equal(np.asarray(out.values), np.asarray(v))


def test_justify_rows_gives_the_unarmed_layout():
    armed, _ = _batch([2, 0, 1, 2])
    plain, _ = _batch([2, 0, 1, 2], armed=False)
    armed.justify_rows()
    assert armed._row_ends is None
    assert armed.left_padding.tolist() == plain.left_padding.tolist()
    n = plain._idx
    for i, p in enumerate(plain.left_padding.tolist()):
        np.testing.assert_array_equal(np.asarray(armed.keys[i, :, p:n]),
                                      np.asarray(plain.keys[i, :, p:n]))


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("qL", [1, BS, 6])
def test_each_row_attends_its_own_keys(qL, dtype):
    """qL 1 to 4 in half precision take the kq call, the rest one call per
    row."""
    if dtype != mx.float32 and not mx.metal.is_available():
        pytest.skip("the kq route runs on the Metal device")
    c, rows = _batch([2, 0, 1, 2], dtype=dtype)
    nk, nv = _kv(qL, 40, dtype=dtype)
    keys, values = c.update_and_fetch(nk, nv)
    q = mx.random.normal((len(PADS), 4 * H, qL, D), key=mx.random.key(50)).astype(dtype)
    scale = D ** -0.5
    got = row_ends.row_ends_attention(q, keys, values, c, scale)
    tol = 1e-5 if dtype == mx.float32 else 2e-2
    for i, (k, v) in enumerate(rows):
        k = mx.concatenate([k, nk[i:i + 1]], axis=2)
        v = mx.concatenate([v, nv[i:i + 1]], axis=2)
        ref = _reference(q[i:i + 1], k, v, scale)
        np.testing.assert_allclose(np.asarray(got[i:i + 1].astype(mx.float32)),
                                   np.asarray(ref.astype(mx.float32)),
                                   atol=tol, rtol=tol)


def test_a_forward_wider_than_a_verify_block_is_refused():
    c, _ = _batch([2, 0, 1, 2])
    nk, nv = _kv(9, 40)
    keys, values = c.update_and_fetch(nk, nv)
    q = mx.zeros((len(PADS), 4 * H, 9, D), mx.float16)
    with pytest.raises(ValueError, match="end apart"):
        row_ends.row_ends_attention(q, keys, values, c, 1.0)


def test_an_even_cache_takes_no_route():
    c, _ = _batch([1, 1, 1, 1])
    q = mx.zeros((len(PADS), 4 * H, 1, D), mx.float16)
    assert row_ends.row_ends_attention(q, c.keys, c.values, c, 1.0) is None


@metal
def test_the_rollback_and_the_next_append_copy_no_row_of_the_batch():
    """A deep batch: the uneven rollback allocates nothing, and the next
    append writes into the buffer it has, not into a copy of it."""
    c = owned.BatchKVCache(PADS)
    c._row_ends_armed = True
    depth = 1 << 14
    k = mx.zeros((len(PADS), H, depth, D), mx.float16)
    c.update_and_fetch(k, k)
    del k
    mx.eval(c.keys, c.values)
    layer_row = c.keys.nbytes // len(PADS)       # one row of K in one layer
    vk, vv = _kv(BS, 10)
    c.update_and_fetch(vk, vv)
    mx.eval(c.keys, c.values)
    mx.synchronize()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    rollback_attention_caches([c], [2, 0, 1, 2], BS)
    mx.eval(c.keys, c.values)
    mx.synchronize()
    assert mx.get_peak_memory() - base < layer_row // 8
    nk, nv = _kv(BS, 20)
    mx.synchronize()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    c.update_and_fetch(nk, nv)
    mx.eval(c.keys, c.values)
    mx.synchronize()
    assert c._row_ends is not None
    assert mx.get_peak_memory() - base < layer_row // 8
