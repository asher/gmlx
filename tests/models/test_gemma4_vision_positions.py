"""The Gemma 4 vision tower's position lookup.

The tower adds one row of each axis table per patch. gmlx gathers those
rows, where upstream multiplied a one-hot by the table. In bfloat16 both
are exact, so they agree bit for bit. In float32 the gather equals the
table rows, which the one-hot matmul does not promise under TF32.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_vlm.models.gemma4.config import VisionConfig
from gmlx._vendor.mlx_vlm.models.gemma4.vision import VisionPatchEmbedder

_SIZE = 64


def _embedder(dtype, seed=0):
    emb = VisionPatchEmbedder(VisionConfig(hidden_size=32,
                                           position_embedding_size=_SIZE))
    rng = np.random.default_rng(seed)
    emb.position_embedding_table = mx.array(
        rng.standard_normal((2, _SIZE, 32)).astype(np.float32)).astype(dtype)
    return emb


def _positions(seed=1):
    """Two images' patch positions, the second padded with (-1, -1)."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, _SIZE, (2, 40, 2))
    pos[1, 30:] = -1
    return mx.array(pos.astype(np.int32)), mx.array(np.all(pos == -1, axis=-1))


def _one_hot_lookup(table, positions, padding):
    """The upstream lookup: a one-hot matmul, then the sum of both axes."""
    oh = (positions[..., None] == mx.arange(_SIZE)).astype(table.dtype)
    out = (oh.transpose(0, 2, 1, 3) @ table).sum(axis=1)
    return mx.where(padding[..., None], 0.0, out)


def test_bfloat16_gather_equals_the_one_hot_lookup():
    emb = _embedder(mx.bfloat16)
    pos, pad = _positions()
    got = emb._position_embeddings(pos, pad)
    want = _one_hot_lookup(emb.position_embedding_table, pos, pad)
    assert got.dtype == want.dtype == mx.bfloat16
    assert mx.array_equal(got, want)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_gather_equals_the_table_rows(dtype):
    emb = _embedder(dtype)
    pos, pad = _positions()
    got = np.array(emb._position_embeddings(pos, pad).astype(mx.float32))
    table = np.array(emb.position_embedding_table.astype(mx.float32))
    p = np.array(pos)
    want = table[0][p[..., 0]] + table[1][p[..., 1]]
    if dtype == mx.bfloat16:
        want = np.array(mx.array(want).astype(mx.bfloat16).astype(mx.float32))
    want[np.array(pad)] = 0.0
    np.testing.assert_array_equal(got, want)
