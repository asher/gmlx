# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Upstream import names that gmlx binds to its own modules.

mlx-lm and mlx-vlm build a model by importing ``<package>.models.<model_type>``,
and mlx-vlm finds a tool parser by importing ``mlx_vlm.tool_parsers.<name>``.
For every name in :data:`OWNED_MODULES`, the GGUF path needs the gmlx module:
its parameter names match the GGUF remap, and its classes carry the K-quant
layout and the speculative hooks. An upstream class of the same name has none
of these, so it builds a model that fails to load an image, refuses a drafter
or decodes from mismatched weights.

:func:`install` binds the name to the gmlx module, replacing any upstream
module of that name, so an upstream release that adds one changes nothing.
Each owning module calls it from its ``ensure_registered()``. gmlx never builds
a model from upstream weights, so no load path needs the replaced module.
"""

from __future__ import annotations

import importlib
import sys

# Upstream import name -> the gmlx module that owns it.
OWNED_MODULES = {
    # Text backbones in the mlx_lm.models namespace, which mlx-lm's
    # _get_classes imports. gmlx.load.arch_table derives its model_type map
    # from these rows.
    # MiniMax-M3, from mlx-lm PR #1401 plus the MSA path.
    "mlx_lm.models.minimax_m3": "gmlx.models.minimax_m3",
    # Qwen3.8-Flash-Next (llama.cpp PR #27742).
    "mlx_lm.models.qwen4_exp": "gmlx.models.qwen4_exp.model",
    # DeepSeek-V4-Flash, from mlx-lm PR #1192, with the hyper_connection and
    # PoolingCache companions that ensure_registered injects.
    "mlx_lm.models.deepseek_v4": "gmlx.models.deepseek_v4.model",
    # DeepSeek-V4.1-Flash. Reuses the deepseek_v4 companions.
    "mlx_lm.models.deepseek_v41": "gmlx.models.deepseek_v41.model",
    # Tencent Hy3, from mlx-lm PR #1485, which supersedes #1211.
    "mlx_lm.models.hy_v3": "gmlx.models.hy_v3.model",
    # Kimi-K3 (llama.cpp PR #26185). kimi_linear is the nearest relative and
    # lacks the five K3-only mechanisms.
    "mlx_lm.models.kimi_k3": "gmlx.models.kimi_k3",
    # Meta Muse Glimmer (llama.cpp LLM_ARCH_MUSE_GLIMMER). afmoe is the
    # nearest relative and has neither the NoPE/RoPE inversion nor the second
    # norm epsilon.
    "mlx_lm.models.muse_glimmer": "gmlx.models.muse_glimmer.model",
    # GLM-5.3-Flash (llama.cpp PR #27754). kimi_k3 is the nearest relative
    # and has none of the indexer, hyper-connection or clamped-swiglu
    # mechanisms.
    "mlx_lm.models.glm5_next": "gmlx.models.glm5_next.model",
    # Tencent HY4-preview (llama.cpp LLM_ARCH_HYV4). deepseek_v32 is the
    # nearest relative and carries neither the independent hyper-connections
    # nor the sinks and the attention gate.
    "mlx_lm.models.hy_v4": "gmlx.models.hy_v4.model",
    # Vision-language models in the mlx_vlm.models namespace, which mlx-vlm's
    # get_model_and_args imports.
    "mlx_vlm.models.muse_glimmer": "gmlx.models.muse_glimmer.vlm_model",
    "mlx_vlm.models.qwen4_exp": "gmlx.models.qwen4_exp.vlm_model",
    "mlx_vlm.models.glm5_next": "gmlx.models.glm5_next.vlm_model",
    "mlx_vlm.models.deepseek_v4_vl": "gmlx.models.deepseek_v4.vlm_model",
    "mlx_vlm.models.deepseek_v41_vl": "gmlx.models.deepseek_v41.vlm_model",
    # Tool parsers, which serve template inference resolves by name.
    "mlx_vlm.tool_parsers.deepseek_v41": "gmlx.models.deepseek_v41.tools",
    "mlx_vlm.tool_parsers.hy_v3": "gmlx.models.hy_v3.tools",
    "mlx_vlm.tool_parsers.hy_v4": "gmlx.models.hy_v4.tools",
    "mlx_vlm.tool_parsers.muse_glimmer": "gmlx.models.muse_glimmer.tools",
}


def install(name: str, owner: str) -> None:
    """Bind ``import <name>`` to the already-imported module ``owner``.

    Sets both ``sys.modules[name]`` and the attribute on the parent package,
    because ``from <parent> import <leaf>`` reads the attribute first. Raises
    ``KeyError`` when ``name`` is not owned by ``owner`` in
    :data:`OWNED_MODULES`. Idempotent.
    """
    if OWNED_MODULES.get(name) != owner:
        raise KeyError(f"{name} is not owned by {owner} in OWNED_MODULES")
    module = sys.modules[owner]
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    setattr(importlib.import_module(parent), leaf, module)
