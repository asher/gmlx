"""The VAE's exact conv path skips MLX's Winograd path, and the Winograd
path stays selectable. In bf16 Winograd holds its transformed tiles in
bf16, so its error is several times the rounding of the output. The reference is the same conv in fp32 on the CPU stream,
over the same bf16 operands, rounded once to bf16. The structural test runs
anywhere, and the precision tests need Metal, where the Winograd path is."""
from __future__ import annotations

import os

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from gmlx.image.vae import flux1_ae
from gmlx.image.vae import conv as conv_mod
from gmlx.image.vae.conv import Conv2d, conv2d_no_winograd

metal = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="the Winograd path is Metal only")

# Relative L2 error of a bf16 3x3 conv against fp32 products rounded once,
# measured on an M5 Max at (1, 64, 64, 128) -> 128: 3.4e-5 for the implicit
# GEMM path and 5.8e-2 for stock nn.Conv2d, which takes the Winograd path.
LIMIT = 1e-3

# Shapes that meet every condition of MLX's Winograd route: 3x3, stride 1,
# C and O multiples of 32, N * H * W >= 4096 and C + O >= 256.
WINOGRAD_SHAPES = [(1, 64, 64, 128, 128), (2, 48, 48, 256, 128),
                   (1, 64, 64, 512, 512), (1, 128, 128, 128, 256)]


def _operands(n, h, w, c, o, seed=0):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((n, h, w, c)).astype(np.float32))
    k = rng.standard_normal((o, 3, 3, c)) / np.sqrt(9 * c)
    return x.astype(mx.bfloat16), mx.array(k.astype(np.float32)).astype(
        mx.bfloat16)


def _rel_error(y: mx.array, x: mx.array, k: mx.array) -> float:
    ref = mx.conv2d(x.astype(mx.float32), k.astype(mx.float32), padding=1,
                    stream=mx.cpu).astype(mx.bfloat16)
    y = np.array(y.astype(mx.float32), dtype=np.float64)
    r = np.array(ref.astype(mx.float32), dtype=np.float64)
    return float(np.linalg.norm(y - r) / np.linalg.norm(r))


def test_every_vae_conv_is_the_guarded_conv():
    cfg = flux1_ae.Flux1AEConfig(
        in_channels=3, out_channels=3, latent_channels=16,
        block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=32,
        eps=1e-6)
    convs = [m for _, m in flux1_ae.Flux1AE(cfg).named_modules()
             if isinstance(m, nn.Conv2d)]
    assert convs and all(isinstance(m, Conv2d) for m in convs)


@pytest.mark.parametrize("path", ["kq", "exact", "winograd"])
def test_every_conv_path_equals_stock_conv_in_fp32(path):
    rng = np.random.default_rng(1)
    for stride, k, pad in ((1, 3, 1), (2, 3, 0), (1, 1, 0)):
        conv = Conv2d(32, 64, k, stride=stride, padding=pad)
        conv.path = path
        x = mx.array(rng.standard_normal((1, 9, 9, 32)).astype(np.float32))
        want = nn.Conv2d.__call__(conv, x)
        got = conv(x)
        np.testing.assert_allclose(np.array(got), np.array(want),
                                   rtol=1e-4, atol=1e-4)


def test_the_default_path_is_kq_and_a_path_is_checked():
    assert conv_mod.CONV == "kq" and Conv2d.path == "exact"
    assert conv_mod.conv_path() == conv_mod.CONV
    with pytest.raises(ValueError, match="conv must be one of"):
        conv_mod.conv_path("fast")
    cfg = flux1_ae.Flux1AEConfig(
        in_channels=3, out_channels=3, latent_channels=16,
        block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=32,
        eps=1e-6)
    vae = flux1_ae.Flux1AE(cfg)
    convs = [m for _, m in vae.named_modules() if isinstance(m, Conv2d)]
    assert conv_mod.set_conv_path(vae, "winograd") == "winograd"
    assert all(m.path == "winograd" for m in convs)
    assert conv_mod.set_conv_path(vae) == "kq"
    assert all(m.path == "kq" for m in convs)


@metal
def test_guarded_conv_rounds_once_at_a_winograd_shape():
    x, k = _operands(1, 64, 64, 128, 128)
    conv = Conv2d(128, 128, 3, padding=1, bias=False)
    conv.weight = k
    assert _rel_error(conv(x), x, k) < LIMIT


@metal
@pytest.mark.parametrize("shape", WINOGRAD_SHAPES)
def test_flipped_conv_stays_off_the_winograd_path(shape):
    x, k = _operands(*shape)
    assert _rel_error(conv2d_no_winograd(x, k, padding=1), x, k) < LIMIT



@metal
def test_the_winograd_path_is_stock_conv2d():
    x, k = _operands(1, 64, 64, 128, 128)
    conv = Conv2d(128, 128, 3, padding=1, bias=False)
    conv.weight = k
    conv.path = "winograd"
    assert mx.array_equal(conv(x), mx.conv2d(x, k, padding=1)).item()
