# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Kernel seams and numeric helpers for image DiTs.

Model code calls the five seams through this namespace, as ``ops.sdpa(...)``,
so a kernel install or ``StepProfiler`` replaces one by assignment here.
"""

from .attention import qk_norm_rope, sdpa
from .embeds import rope_tables, timestep_embedding
from .epilogues import gated_residual, modulate, swiglu
from .norms import RMSNorm, layer_norm, rms_norm

SEAMS = ("sdpa", "qk_norm_rope", "swiglu", "modulate", "gated_residual")

__all__ = [
    "SEAMS", "RMSNorm", "gated_residual", "layer_norm", "modulate",
    "qk_norm_rope", "rms_norm", "rope_tables", "sdpa", "swiglu",
    "timestep_embedding",
]
