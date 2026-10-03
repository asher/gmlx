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
times as long on an M5 Max. ``load_vae(conv=...)`` picks the path for one
VAE, and ``CONV`` is the default.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

CONV_PATHS = ("exact", "winograd")
CONV = "exact"


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
    """``nn.Conv2d`` computed by ``conv2d_no_winograd`` when ``exact`` is
    true, and by ``mx.conv2d`` otherwise."""

    exact = True

    def __call__(self, x: mx.array) -> mx.array:
        conv = conv2d_no_winograd if self.exact else mx.conv2d
        y = conv(x, self.weight, self.stride, self.padding, self.dilation,
                 self.groups)
        if "bias" in self:
            y = y + self.bias
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
            m.exact = conv == "exact"
    return conv
