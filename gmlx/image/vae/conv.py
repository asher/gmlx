# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""VAE convs on MLX's exact path or its Winograd path.

On Metal, MLX computes a 3x3 conv with stride 1, no dilation, input and
output channels that are multiples of 32, 4096 or more input pixels and
256 or more channels in and out together by Winograd F(6x6, 3x3). The
transformed tiles are held in the input dtype, so in bf16 the output
carries several roundings of large intermediate sums, and a FLUX.1 decode
loses 9 dB of PSNR against products rounded once. The Winograd route is
taken only when ``flip`` is false. The same conv written as the flipped
kernel with ``flip=True`` goes to the implicit GEMM kernels, which
accumulate in fp32 and round once.

The exact path costs time: a bf16 FLUX.1 decode at 1024 pixels takes 1.48
times as long on an M5 Max. The "kq" path runs a 3x3 stride 1 padding 1
conv on mlx-kquant's ``conv2d_3x3``, which sums in fp32, adds the bias in
fp32 and rounds once, when its route is in force and the conv is bf16 with
input channels on a multiple of 16; every other conv on that path runs as
on the exact path. ``load_vae(conv=...)`` picks the path for one VAE, and
``CONV`` is the default.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

CONV_PATHS = ("kq", "exact", "winograd")
CONV = "kq"


def _pair(v) -> tuple[int, int]:
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def conv2d_no_winograd(x: mx.array, weight: mx.array, stride=1, padding=0,
                       dilation=1, groups: int = 1) -> mx.array:
    """``mx.conv2d`` for a channels-last ``x`` and an (O, kH, kW, C)
    ``weight``. A 3x3 stride 1 undilated conv runs on the flipped kernel
    with ``flip=True``, and every other conv runs as ``mx.conv2d``."""
    if (weight.shape[1:3] == (3, 3) and _pair(stride) == (1, 1)
            and _pair(dilation) == (1, 1)):
        return mx.conv_general(x, weight[:, ::-1, ::-1, :], stride=stride,
                               padding=padding, kernel_dilation=dilation,
                               groups=groups, flip=True)
    return mx.conv2d(x, weight, stride=stride, padding=padding,
                     dilation=dilation, groups=groups)


class Conv2d(nn.Conv2d):
    """``nn.Conv2d`` on the conv path ``path``: "kq" takes the kernel
    where ``kq_routes.conv3x3`` does, "exact" computes by
    ``conv2d_no_winograd`` and "winograd" by ``mx.conv2d``."""

    path = "exact"

    def __call__(self, x: mx.array) -> mx.array:
        bias = self.bias if "bias" in self else None
        if self.path == "kq":
            from gmlx.image.ops import kq_routes

            y = kq_routes.conv3x3(x, self.weight, bias, self.stride,
                                  self.padding, self.dilation, self.groups)
            if y is not None:
                return y
        conv = mx.conv2d if self.path == "winograd" else conv2d_no_winograd
        y = conv(x, self.weight, self.stride, self.padding, self.dilation,
                 self.groups)
        if bias is not None:
            y = y + bias
        return y


def conv_path(conv: str | None = None) -> str:
    """``conv`` checked against ``CONV_PATHS``, or ``CONV`` when None."""
    conv = CONV if conv is None else conv
    if conv not in CONV_PATHS:
        raise ValueError(f"conv must be one of {CONV_PATHS}, not {conv!r}")
    return conv


def set_conv_path(module: nn.Module, conv: str | None = None) -> str:
    """Put every ``Conv2d`` in ``module`` on ``conv_path(conv)`` and
    return that path."""
    conv = conv_path(conv)
    for _, m in module.named_modules():
        if isinstance(m, Conv2d):
            m.path = conv
    return conv
