# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Kernel seams and numeric helpers for image DiTs.

Model code calls the five seams through this namespace, as ``ops.sdpa(...)``,
so a kernel install or ``StepProfiler`` replaces one by assignment here.
The kernel install is ``kq_routes.install``. It runs at import and puts in
force the mlx-kquant routes that the installed build has an op for.
"""

from . import kq_routes
from .attention import qk_norm_rope, sdpa
from .embeds import rope_tables, timestep_embedding
from .epilogues import gated_residual, modulate, swiglu
from .kq_routes import glu_pair, lora_fused, swiglu_fused
from .norms import RMSNorm, layer_norm, rms_norm

SEAMS = ("sdpa", "qk_norm_rope", "swiglu", "modulate", "gated_residual")

__all__ = [
    "SEAMS", "RMSNorm", "gated_residual", "glu_pair", "kq_routes",
    "layer_norm", "lora_fused", "modulate", "qk_norm_rope", "rms_norm",
    "rope_tables", "sdpa", "swiglu", "swiglu_fused", "timestep_embedding",
]

kq_routes.install()
