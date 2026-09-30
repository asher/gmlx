# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Apple Inc. (mlx-lm, MIT)
# Vendored from mlx-lm 0.31.3, mlx_lm/models/gemma4_text.py.
# Source sha256: 77f46bc3f162a0b9513157dade4be2c381d4df3295262c69034a53d46111370f
# Extract: Model.sanitize, lines 610 to 639.
# Extract sha256: 0a3aa7a1f83b1ef8180cb17864387f1526c3d61d7778631f7d6c92cd177c93a1
"""mlx-lm 0.31.3 Gemma 4 text weight sanitize, owned by gmlx.

``Model.sanitize`` from ``mlx_lm/models/gemma4_text.py`` as a module
function, dedented four spaces, with the import it needs. It drops rotary
tables and clip ranges and splits the fused expert projections.
"""

import mlx.core as mx


def sanitize(self, weights):
    sanitized = {}
    for k, v in weights.items():
        if any(
            s in k
            for s in (
                "self_attn.rotary_emb",
                "input_max",
                "input_min",
                "output_max",
                "output_min",
            )
        ):
            continue

        if k.endswith(".experts.gate_up_proj"):
            base = k.removesuffix(".gate_up_proj")
            gate, up = map(mx.contiguous, mx.split(v, 2, axis=-2))
            sanitized[f"{base}.switch_glu.gate_proj.weight"] = gate
            sanitized[f"{base}.switch_glu.up_proj.weight"] = up
            continue

        if k.endswith(".experts.down_proj"):
            base = k.removesuffix(".down_proj")
            sanitized[f"{base}.switch_glu.down_proj.weight"] = v
            continue

        sanitized[k] = v

    return sanitized
