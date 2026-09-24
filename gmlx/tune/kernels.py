"""Kernel choices for a training run.

kq sends a sorted expert call of at least ``KQ_SWITCH_GEMM_MIN_ROWS``
routed rows (512 by default) to a per-expert segment GEMM on the GPU,
which has no backward. A MoE training step routes more rows than that,
so its backward fails. ``install_training_switch_gemm`` sets the
threshold to 0 for the length of a training run, so every expert call
takes kq's gather matmul, whose backward gives the gradient with respect
to the input. kq reads the variable on every call.
"""
from __future__ import annotations

import os

_SWITCH_GEMM = "KQ_SWITCH_GEMM_MIN_ROWS"


def install_training_switch_gemm():
    """Set ``KQ_SWITCH_GEMM_MIN_ROWS`` to 0 and return a callable that
    puts back the value it had."""
    old = os.environ.get(_SWITCH_GEMM)
    os.environ[_SWITCH_GEMM] = "0"

    def restore():
        if old is None:
            os.environ.pop(_SWITCH_GEMM, None)
        else:
            os.environ[_SWITCH_GEMM] = old

    return restore
