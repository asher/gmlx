"""Prompt cache blocks store their arrays stacked and join back per layer."""
from __future__ import annotations

import mlx.core as mx
import pytest

from gmlx._vendor.mlx_vlm.apc_storage import APCNode, KVBlockHandle
from gmlx.cache import apc_blocks
from gmlx.cache.apc_blocks import block_layers, block_tokens, join_blocks
from gmlx.cache.apc_manager import GmlxAPCManager


class _Node(APCNode):
    def __init__(self):
        self.components = {}


def _kv(shapes, tokens, seed):
    mx.random.seed(seed)
    keys = [mx.random.normal((1, h, tokens, d)).astype(mx.bfloat16)
            for h, d in shapes]
    values = [mx.random.normal((1, h, tokens, d)).astype(mx.bfloat16)
              for h, d in shapes]
    mx.eval(keys + values)
    return keys, values


def _blocks(shapes, sizes, seed=0):
    """Blocks of the given token counts, and the per-layer source arrays."""
    keys, values = _kv(shapes, sum(sizes), seed)
    blocks, start = [], 0
    for n in sizes:
        b = _Node()
        b.set_kv([k[..., start:start + n, :] for k in keys],
                 [v[..., start:start + n, :] for v in values])
        blocks.append(b)
        start += n
    return blocks, keys, values


def test_layers_of_one_shape_stack_into_one_buffer():
    blocks, _, _ = _blocks([(2, 8)] * 4, [16])
    h = blocks[0].kv_handle()
    assert h.buffers() == 1
    assert h.parts[0].shape == (8, 1, 2, 16, 8)
    assert h.tokens == 16 and h.n_layers == 4
    assert h.resident_bytes() == 8 * 2 * 16 * 8 * 2
    assert not hasattr(blocks[0], "keys") and not hasattr(blocks[0], "values")


def test_mixed_shapes_stack_one_part_per_shape():
    shapes = [(2, 8), (1, 16), (2, 8), (1, 16), (4, 8)]
    blocks, _, _ = _blocks(shapes, [16])
    assert blocks[0].kv_handle().buffers() == 3
    k, v = _kv(shapes, 16, 1)
    assert KVBlockHandle.count_parts(k, v) == 3


@pytest.mark.parametrize("v_dim, layers", [(8, 4), (16, 1)])
def test_parts_do_not_alias_the_source(v_dim, layers):
    # A block cut from a long prompt must not keep the prompt's buffers
    # alive once the prompt is gone. Keys and values of different head
    # dims store each array alone, the path that copies before stacking.
    import gc

    mx.clear_cache()
    base = mx.get_active_memory()
    mx.random.seed(2)
    keys = [mx.random.normal((1, 2, 65536, 8)).astype(mx.bfloat16)
            for _ in range(layers)]
    values = [mx.random.normal((1, 2, 65536, v_dim)).astype(mx.bfloat16)
              for _ in range(layers)]
    mx.eval(keys + values)
    source = sum(a.nbytes for a in keys + values)
    b = _Node()
    b.set_kv([k[..., :16, :] for k in keys], [v[..., :16, :] for v in values])
    mx.eval(b.kv_handle().parts)
    del keys, values
    gc.collect()
    mx.clear_cache()
    assert b.kv_handle().resident_bytes() == layers * 2 * 16 * (8 + v_dim) * 2
    assert mx.get_active_memory() - base < source // 8


@pytest.mark.parametrize("group_bytes", [1 << 30, 1])
@pytest.mark.parametrize("shapes", [[(2, 8)] * 3, [(2, 8), (1, 16), (2, 8)]])
def test_join_equals_the_source(monkeypatch, shapes, group_bytes):
    # group_bytes 1 puts each block in its own group, so the join runs
    # once per block with an eval between.
    monkeypatch.setattr(apc_blocks, "GROUP_BYTES", group_bytes)
    sizes = [16, 16, 16, 8]
    blocks, keys, values = _blocks(shapes, sizes, seed=3)
    assert block_tokens(blocks) == 56 and block_layers(blocks[0]) == 3
    joined = join_blocks(blocks)
    assert len(joined) == len(shapes)
    for (k, v), wk, wv in zip(joined, keys, values):
        assert k.dtype == wk.dtype and k.shape == wk.shape
        assert mx.array_equal(k, wk) and mx.array_equal(v, wv)


def test_join_refuses_blocks_of_different_layouts():
    a, _, _ = _blocks([(2, 8)] * 2, [16])
    b, _, _ = _blocks([(2, 8), (1, 16)], [16])
    with pytest.raises(ValueError, match="different layouts"):
        join_blocks(a + b)


def test_released_block_holds_no_kv():
    blocks, _, _ = _blocks([(2, 8)] * 2, [16])
    b = blocks[0]
    b.release_components()
    assert b.kv_handle() is None and block_layers(b) == 0


def test_store_counts_one_buffer_a_block():
    # The pool's buffer guard counts stacked parts, not layers.
    keys, values = _kv([(2, 8)] * 6, 64, 4)
    m = GmlxAPCManager(num_blocks=8, block_size=16)
    stored = m.store_kv_blocks(list(range(64)), keys, values)
    assert len(stored) == 4
    assert {b.kv_handle().buffers() for b in stored} == {1}
    for i, (k, v) in enumerate(join_blocks(stored)):
        assert mx.array_equal(k, keys[i]) and mx.array_equal(v, values[i])
    m.release(stored)


def test_resident_bytes_reads_the_stacked_parts():
    blocks, _, _ = _blocks([(2, 8), (1, 16)], [16])
    node = blocks[0]
    assert node.resident_bytes() == sum(
        p.nbytes for p in node.kv_handle().parts)
