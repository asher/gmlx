# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 TSAIL Team and The HuggingFace Team
# Ported from diffusers' src/diffusers/schedulers/scheduling_unipc_multistep.py
# (the bh2 predictor and corrector with flow sigmas and x0 prediction) at
# e0abab83, and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""UniPC of order 2 with the bh2 variant over flow sigmas, ComfyUI's
``uni_pc_bh2``.

The model's velocity v gives the clean estimate ``x - sigma * v``. Each
step first corrects the latents it is given with that estimate (UniC),
then predicts the next latents (UniP). The first step and the last run at
order 1, as diffusers runs them with ``lower_order_final``. A first sigma
within 1e-6 of 1 is lowered by 1e-6, as diffusers lowers it, so its log
stays finite. The coefficients are float32 on the host, as diffusers
computes them in float32 tensors, and the latents stay float32.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .samplers import Sampler

_F32 = np.float32


def flow_sigmas(sigmas) -> np.ndarray:
    """The schedule as diffusers' flow UniPC holds it: a first sigma within
    1e-6 of 1 lowered by 1e-6, and the final sigma kept."""
    s = np.asarray(sigmas, _F32).copy()
    head = s[:-1]
    if np.fabs(head[0] - 1) < 1e-6:
        head[0] -= 1e-6
    return np.concatenate([head, s[-1:]]).astype(_F32)


def _lambda(sigma: np.float32) -> np.float32:
    with np.errstate(divide="ignore"):
        return _F32(np.log(_F32(1) - sigma) - np.log(sigma))


def _phis(h: np.float32, order: int):
    """h_phi_1, B(h) and the b vector of orders 1 to ``order``."""
    hh = -h
    with np.errstate(divide="ignore", invalid="ignore"):
        h_phi_1 = _F32(np.expm1(hh))
        h_phi_k = _F32(h_phi_1 / hh - 1)
        b_h = _F32(np.expm1(hh))
        b, fact = [], 1
        for i in range(1, order + 1):
            b.append(_F32(h_phi_k * fact / b_h))
            fact *= i + 1
            h_phi_k = _F32(h_phi_k / hh - _F32(1) / fact)
    return h_phi_1, b_h, np.array(b, _F32)


class UniPC(Sampler):
    name = "uni_pc_bh2"
    order = 2

    def __init__(self, sigmas):
        super().__init__(sigmas)
        self.sigmas = flow_sigmas(self.sigmas)
        self._x0: list = []                 # clean estimates, oldest first
        self._last: mx.array | None = None  # latents before the predictor
        self._order = 0                     # order of the last predictor
        self._lower = 0

    def _scal(self, a: float) -> mx.array:
        return mx.array(_F32(a), mx.float32)

    def step(self, x: mx.array, v: mx.array, i: int) -> mx.array:
        if not 0 <= i < self.steps:
            raise IndexError(f"step {i} outside 0..{self.steps - 1}")
        s = self.sigmas
        x = x.astype(mx.float32)
        m = x - self._scal(s[i]) * v.astype(mx.float32)
        if i > 0 and self._last is not None:
            x = self._correct(m, i)
        self._x0 = (self._x0 + [m])[-self.order:]
        order = min(self.order, self.steps - i, self._lower + 1)
        self._order = order
        self._last = x
        out = self._predict(x, i, order)
        self._lower = min(self._lower + 1, self.order)
        return out

    def _predict(self, x: mx.array, i: int, order: int) -> mx.array:
        s = self.sigmas
        s0, st = s[i], s[i + 1]
        m0 = self._x0[-1]
        lam0 = _lambda(s0)
        h = _F32(_lambda(st) - lam0)
        h_phi_1, b_h, _ = _phis(h, order)
        alpha_t = _F32(1) - st
        x_t = self._scal(st / s0) * x - self._scal(alpha_t * h_phi_1) * m0
        if order == 2:
            rk = _F32((_lambda(s[i - 1]) - lam0) / h)
            d1 = (self._x0[-2] - m0) / self._scal(rk)
            x_t = x_t - self._scal(alpha_t * b_h * _F32(0.5)) * d1
        return x_t

    def _correct(self, m_t: mx.array, i: int) -> mx.array:
        s = self.sigmas
        order = self._order
        st, s0 = s[i], s[i - 1]
        m0 = self._x0[-1]
        lam0 = _lambda(s0)
        h = _F32(_lambda(st) - lam0)
        h_phi_1, b_h, b = _phis(h, order)
        alpha_t = _F32(1) - st
        x_t = (self._scal(st / s0) * self._last
               - self._scal(alpha_t * h_phi_1) * m0)
        if order == 1:
            rhos = np.array([0.5], _F32)
            corr = None
        else:
            rk = _F32((_lambda(s[i - 2]) - lam0) / h)
            r = np.array([[1, 1], [rk, 1]], _F32)
            rhos = np.linalg.solve(r, b).astype(_F32)
            corr = self._scal(rhos[0]) * ((self._x0[-2] - m0)
                                          / self._scal(rk))
        res = self._scal(rhos[-1]) * (m_t - m0)
        if corr is not None:
            res = corr + res
        return x_t - self._scal(alpha_t * b_h) * res
