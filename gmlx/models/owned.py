# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""The model types gmlx owns, and the module that builds each.

The vendored loaders look a model type up here before their own
packages: ``_get_classes`` in the vendored mlx-lm ``utils.py`` and
``get_model_and_args`` in the vendored mlx-vlm ``utils.py``. A gmlx model
never enters a vendored package namespace. The vendored mlx-lm 0.31.3
ships none of the mlx-lm types. The vendored mlx-vlm 0.7.3 ships
qwen4_exp, glm5_next and muse_glimmer packages of its own, with parameter
trees the GGUF remap does not target, and the copy leaves them out.
"""

# mlx-lm model_type -> the gmlx module that defines Model and ModelArgs.
OWNED_LM_MODELS = {
    # mlx-lm PR #1401 (MiniMax-M3 text backbone).
    "minimax_m3": "gmlx.models.minimax_m3",
    # llama.cpp PR #27742 (Qwen3.8-Flash-Next).
    "qwen4_exp": "gmlx.models.qwen4_exp.model",
    # mlx-lm PR #1192 (DeepSeek V4 Flash), with the hyper_connection and
    # DS4PoolingCache/DS4BatchPoolingCache companions.
    "deepseek_v4": "gmlx.models.deepseek_v4.model",
    # DeepSeek-V4.1-Flash. Reuses the deepseek_v4 hyper_connection and
    # DS4PoolingCache companions.
    "deepseek_v41": "gmlx.models.deepseek_v41.model",
    # mlx-lm PR #1485 (Tencent Hy3, supersedes #1211).
    "hy_v3": "gmlx.models.hy_v3.model",
    # llama.cpp PR #26185 (Kimi-K3); kimi_linear is the nearest relative
    # and lacks the five K3-only mechanisms.
    "kimi_k3": "gmlx.models.kimi_k3",
    # llama.cpp LLM_ARCH_MUSE_GLIMMER; afmoe is the nearest relative and has
    # neither the NoPE/RoPE inversion nor the second norm epsilon.
    "muse_glimmer": "gmlx.models.muse_glimmer.model",
    # llama.cpp LLM_ARCH_MUSE_SPARK; muse_glimmer is the nearest relative
    # and has neither the latent MoE nor the norm-preserving residuals.
    "muse_spark": "gmlx.models.muse_spark.model",
    # llama.cpp PR #27754 (GLM-5.3-Flash); kimi_k3 is the nearest relative
    # and has none of the indexer, hyper-connection or clamped-swiglu
    # mechanisms.
    "glm5_next": "gmlx.models.glm5_next.model",
    # llama.cpp LLM_ARCH_HYV4 (HY4-preview); deepseek_v32 is the nearest
    # relative and carries neither the independent hyper-connections nor
    # the sinks and the attention gate.
    "hy_v4": "gmlx.models.hy_v4.model",
    # llama.cpp LLM_ARCH_K2_HORIZON; the mlx-vlm class is a plain Llama
    # with none of the group norm, the output gate, MoE or MoVA.
    "k2_horizon": "gmlx.models.k2_horizon.model",
    # llama.cpp LLM_ARCH_BEAM; neither mlx-lm nor transformers has a class.
    "beam": "gmlx.models.beam.model",
    # llama.cpp LLM_ARCH_FLINT; the converter subclasses qwen2moe, whose
    # class has none of the sinks, the NoPE layers or the sigmoid router.
    "flint": "gmlx.models.flint.model",
    # mlx-lm nemotron_h, with per-layer MoE sizes (Nemotron-3.5-Bolt) and
    # llama.cpp's fp32 routing.
    "nemotron_h": "gmlx.models.nemotron_h.model",
}

# mlx-vlm model_type -> the gmlx module that stands for the model package
# (Model, ModelConfig, TextConfig, VisionConfig, LanguageModel).
OWNED_VLM_MODELS = {
    "deepseek_v4_vl": "gmlx.models.deepseek_v4.vlm_model",
    "deepseek_v41_vl": "gmlx.models.deepseek_v41.vlm_model",
    "glm5_next": "gmlx.models.glm5_next.vlm_model",
    "minicpmv4_6": "gmlx.models.minicpmv.vlm_model",
    "minicpmv4_7": "gmlx.models.minicpmv.vlm_model",
    "muse_glimmer": "gmlx.models.muse_glimmer.vlm_model",
    "muse_spark": "gmlx.models.muse_spark.vlm_model",
    "qwen4_exp": "gmlx.models.qwen4_exp.vlm_model",
    "nemotron_h_omni": "gmlx.models.nemotron_h.vlm_model",
}
