# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2025 Black Forest Labs, Inc.
# Ported from BFL's FLUX.2 reference at commit 50fe516: denoise_cached in
# src/flux2/sampling.py, and modified for gmlx (Apache-2.0; see
# licenses/flux2-LICENSE).
"""The step of FLUX.2 klein 9B KV in the parts a pipeline caches.

A conditioning's state is its text rows after ``txt_in`` and, for an
edit, each block's K and V of its reference tokens, from the reference
pass. Every step reads that state in place of the reference tokens. A
step without reference images is the plain forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from gmlx.image.families.flux2_klein.model import Caption
from gmlx.image.ref_cache import RefCache, cache_bytes


@dataclass(frozen=True)
class KVState:
    """The text rows of a conditioning and the cache of its references,
    None for a conditioning without reference images."""
    caption: Caption
    refs: RefCache | None = None

    def arrays(self) -> list:
        return [*self.caption.arrays(),
                *(self.refs.arrays() if self.refs is not None else ())]


def modulation(dit, sigma):
    return dit.modulation(np.reshape(np.float32(sigma), (1,)), 1)


def conditioning(dit, cond) -> KVState:
    caption = dit.caption(cond.features[None])
    if not cond.ref_latents:
        return KVState(caption)
    refs = dit.references([mx.array(z)[None] for z in cond.ref_latents])
    return KVState(caption, dit.reference_pass(refs))


def forward(dit, latents, sigma, cond, state: KVState, mod) -> mx.array:
    return dit(latents, np.reshape(np.float32(sigma), (1,)), None, mod=mod,
               caption=state.caption, cache=state.refs)


def ref_tokens(cond, patch: int = 1) -> int:
    """Tokens of a conditioning's reference latents (C, h, w)."""
    return sum((int(z.shape[-2]) // patch) * (int(z.shape[-1]) // patch)
               for z in cond.ref_latents)


def state_bytes(dit, cond) -> int:
    """Bytes of the state of ``cond`` before it is built: its text rows
    and the K and V of its reference tokens in every block."""
    cfg = dit.cfg
    item = dit.act_dtype.size
    text = int(cond.features.shape[0]) * cfg.dim * item
    blocks = cfg.depth + cfg.depth_single
    return text + cache_bytes(blocks, ref_tokens(cond, cfg.patch), cfg.dim,
                              item)
