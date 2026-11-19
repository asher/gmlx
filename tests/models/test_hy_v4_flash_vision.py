"""Hy4-flash vision tower and projector on a tiny random config: the
position grid resample, the two convolutions run as Linears, and the row
layout an image takes in the prompt. CPU-only."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.models.hy_v4_flash.vision import (
    VisionConfig,
    VisionModel,
    resampled_positions,
)


@pytest.fixture(autouse=True)
def _on_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _tower(**over):
    cfg = dict(depth=2, hidden_size=16, num_heads=2, intermediate_size=24,
               patch_size=2, in_channels=3, spatial_merge_size=2,
               merge_hidden_size=20, out_hidden_size=12, position_grid=8)
    cfg.update(over)
    model = VisionModel(VisionConfig(**cfg))
    mx.random.seed(4)
    model.load_weights(
        [(k, mx.random.normal(v.shape, scale=0.3).astype(v.dtype))
         for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    return model


def _reference_resample(table, edge, n_h, n_w):
    """torch F.interpolate(bilinear, align_corners=False) at an explicit
    scale factor of (n + 0.1) / edge, written out per output cell."""
    grid = np.asarray(table, dtype=np.float64).reshape(edge, edge, -1)

    def taps(n):
        out = []
        for d in range(n):
            src = max((d + 0.5) * edge / (n + 0.1) - 0.5, 0.0)
            lo = min(int(np.floor(src)), edge - 1)
            hi = min(lo + 1, edge - 1)
            out.append((lo, hi, src - lo))
        return out

    ref = np.zeros((n_h, n_w, grid.shape[-1]))
    for i, (r0, r1, fr) in enumerate(taps(n_h)):
        for j, (c0, c1, fc) in enumerate(taps(n_w)):
            top = grid[r0, c0] * (1 - fc) + grid[r0, c1] * fc
            bot = grid[r1, c0] * (1 - fc) + grid[r1, c1] * fc
            ref[i, j] = top * (1 - fr) + bot * fr
    return ref.reshape(n_h * n_w, -1)


# Smaller than the table, its size, and larger: the last reads past the
# table's last row, which holds.
@pytest.mark.parametrize("n_h,n_w", [(2, 6), (8, 8), (4, 14)])
def test_the_position_grid_resamples_at_the_reference_scale(n_h, n_w):
    edge = 8
    mx.random.seed(2)
    table = mx.random.normal((edge * edge, 5))
    got = np.array(resampled_positions(table, edge, n_h, n_w))
    assert got.shape == (n_h * n_w, 5)
    assert np.allclose(got, _reference_resample(table, edge, n_h, n_w),
                       atol=1e-5)
    # Not the table itself at the table's own size: the scale is
    # (n + 0.1) / edge, never 1.
    if (n_h, n_w) == (edge, edge):
        assert not np.allclose(got, np.array(table), atol=1e-4)


def test_the_projector_is_the_stride_2_convolution_then_the_1x1():
    model = _tower()
    proj = model.projector
    n_h, n_w, dim = 4, 6, 16
    mx.random.seed(9)
    x = mx.random.normal((n_h * n_w, dim))
    got = np.array(proj(x, n_h, n_w))

    normed = proj.pre_norm(x).reshape(1, n_h, n_w, dim)
    # The file's kernel is [out, c, kh, kw]; MLX takes [out, kh, kw, c].
    kernel = proj.merge.weight.reshape(-1, dim, 2, 2).transpose(0, 2, 3, 1)
    y = mx.conv2d(normed, kernel, stride=2) + proj.merge.bias
    y = mx.conv2d(nn.gelu(y), proj.out.weight[:, None, None, :])
    y = y + proj.out.bias
    assert got.shape == (n_h // 2, n_w // 2, 12)
    assert np.allclose(got, np.array(y[0]), atol=1e-5)


def test_an_image_gives_its_rows_then_a_newline_each():
    model = _tower()
    n_h, n_w = 4, 6                       # merged 2 x 3
    mx.random.seed(1)
    patches = mx.random.normal((n_h * n_w, 3 * 2 * 2))
    rows = model(patches, n_h, n_w)
    assert rows.shape == (2 * (3 + 1), 12)
    got = np.array(rows).reshape(2, 4, 12)
    assert np.allclose(got[:, 3], np.array(model.image_newline)[None],
                       atol=1e-6)

    x = model.patch_embed(patches, n_h, n_w)
    for block in model.blocks:
        x = block(x)
    grid = np.array(model.projector(x, n_h, n_w))
    assert np.allclose(got[:, :3], grid, atol=1e-6)


def test_a_grid_that_does_not_merge_is_refused():
    model = _tower()
    with pytest.raises(ValueError, match="merged 2 x 2"):
        model(mx.zeros((3 * 4, 12)), 3, 4)
    with pytest.raises(ValueError, match="patches"):
        model(mx.zeros((7, 12)), 2, 4)
