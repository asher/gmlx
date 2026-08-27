"""Float32 final-logit softcap for gemma-4.

Upstream computes tanh(x/cap)*cap at the activation dtype. Each 16-bit
logit then rounds by ~cap*eps: ~0.12 nats at bfloat16 with cap 30, enough
to flip near-tie top-1 picks. Compute in float32 and emit float32 logits,
matching the muse-glimmer softcap path. Covers the mlx-lm gemma4_text and
mlx-vlm gemma4 seams; install-once, GMLX_G4_SOFTCAP_F32=0 disables.
"""
from __future__ import annotations

import mlx.core as mx

from gmlx.envflags import env_bool

_installed = False


def _logit_softcap_f32(softcap, x):
    return mx.tanh(x.astype(mx.float32) / softcap) * softcap


def install_gemma4_softcap_f32() -> bool:
    global _installed
    if not env_bool("GMLX_G4_SOFTCAP_F32", True):
        return False
    if _installed:
        return True
    from mlx_lm.models import gemma4_text as _lm
    _lm.logit_softcap = _logit_softcap_f32
    try:
        from mlx_vlm.models.gemma4 import language as _vlm
        _vlm.logit_softcap = _logit_softcap_f32
    except ImportError:
        pass
    _installed = True
    return True
