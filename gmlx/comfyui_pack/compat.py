# SPDX-License-Identifier: Apache-2.0
"""The ComfyUI internals this pack depends on, probed at import.

A ComfyUI release can rename or drop an internal the pack touches. The
probe names the first one missing together with the ComfyUI version, so the
pack loads its stand-in nodes (stub.py), whose error says so, instead of
failing during a run. This module imports no ComfyUI module at its top.
"""

from __future__ import annotations

import importlib
import inspect

# (module, dotted attribute) pairs the pack reads, calls or overrides.
NAMES = (
    ("comfy.model_patcher", "ModelPatcher"),
    ("comfy.model_patcher", "ModelPatcher.clone"),
    ("comfy.model_patcher", "ModelPatcher.add_patches"),
    ("comfy.model_patcher", "ModelPatcher.load"),
    ("comfy.model_patcher", "ModelPatcher.partially_load"),
    ("comfy.model_patcher", "ModelPatcher.partially_unload"),
    ("comfy.model_patcher", "ModelPatcher.patch_weight_to_device"),
    ("comfy.model_patcher", "ModelPatcher.unpatch_model"),
    ("comfy.model_patcher", "ModelPatcher.detach"),
    ("comfy.model_patcher", "ModelPatcher.pre_run"),
    ("comfy.model_patcher", "ModelPatcher.cleanup"),
    ("comfy.model_patcher", "ModelPatcher.model_size"),
    ("comfy.model_patcher", "ModelPatcher.loaded_size"),
    ("comfy.model_patcher", "ModelPatcher.get_free_memory"),
    ("comfy.model_patcher", "ModelPatcher.is_dynamic"),
    ("comfy.model_patcher", "ModelPatcher.use_ejected"),
    ("comfy.model_patcher", "ModelPatcher.unpatch_hooks"),
    ("comfy.model_patcher", "ModelPatcher.apply_hooks"),
    ("comfy.model_patcher", "ModelPatcher.get_all_callbacks"),
    ("comfy.model_patcher", "ModelPatcher.set_model_patch"),
    ("comfy.model_patcher", "ModelPatcher.set_model_patch_replace"),
    ("comfy.model_patcher", "ModelPatcher.set_model_rope_options"),
    ("comfy.model_patcher", "ModelPatcher.set_model_optimized_attention"),
    ("comfy.model_patcher", "ModelPatcher.add_weight_wrapper"),
    ("comfy.model_patcher", "ModelPatcher.add_object_patch"),
    ("comfy.model_patcher", "ModelPatcher.add_wrapper_with_key"),
    ("comfy.model_patcher", "ModelPatcher.add_hook_patches"),
    ("comfy.patcher_extension", "CallbacksMP.ON_LOAD"),
    ("comfy.patcher_extension", "WrappersMP.DIFFUSION_MODEL"),
    ("comfy.patcher_extension", "get_all_wrappers"),
    ("comfy.model_base", "BaseModel._apply_model"),
    ("comfy.model_base", "BaseModel.extra_conds"),
    ("comfy.model_base", "ModelType.FLOW"),
    ("comfy.conds", "CONDRegular"),
    ("comfy.latent_formats", "Flux2"),
    ("comfy.model_detection", "model_config_from_unet"),
    ("comfy.model_detection", "unet_prefix_from_state_dict"),
    ("comfy.supported_models", "ZImage"),
    ("comfy.supported_models", "QwenImage21"),
    ("comfy.supported_models_base", "BASE.set_inference_dtype"),
    ("comfy.supported_models_base", "BASE.get_model"),
    ("comfy.model_management", "load_models_gpu"),
    ("comfy.model_management", "processing_interrupted"),
    ("comfy.model_management", "throw_exception_if_processing_interrupted"),
    ("comfy.samplers", "calculate_sigmas"),
    ("comfy.sd", "CLIP"),
    ("comfy.sd", "CLIP.encode_from_tokens_scheduled"),
    ("comfy.sd", "VAE"),
    ("comfy.sd", "VAE.vae_encode_crop_pixels"),
    ("comfy.sd", "load_lora_for_models"),
    ("comfy.utils", "state_dict_prefix_replace"),
    ("comfy.utils", "set_attr"),
    ("comfy.weight_adapter", "LoRAAdapter"),
)

# Arguments the pack passes by name to a ComfyUI signature.
SIGNATURES = (
    ("comfy.model_patcher", "ModelPatcher.__init__",
     ("model", "load_device", "offload_device", "size", "weight_inplace_update", "fast_disk")),
    ("comfy.sd", "CLIP.__init__", ("no_init",)),
)


def comfyui_version() -> str:
    try:
        import comfyui_version
        return str(comfyui_version.__version__)
    except Exception:  # noqa: BLE001: an unknown version is reported as such
        return "unknown"


def _resolve(module: str, dotted: str):
    obj = importlib.import_module(module)
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def first_missing() -> str | None:
    """The first internal this ComfyUI lacks, as ``module.name``, or None."""
    for module, dotted in NAMES:
        try:
            _resolve(module, dotted)
        except (ImportError, AttributeError):
            return f"{module}.{dotted}"
    for module, dotted, params in SIGNATURES:
        try:
            sig = inspect.signature(_resolve(module, dotted))
        except (ImportError, AttributeError, TypeError, ValueError):
            return f"{module}.{dotted}"
        for p in params:
            if p not in sig.parameters:
                return f"{module}.{dotted}({p}=...)"
    return None


def check() -> None:
    """Raise ImportError naming the first missing internal and the version."""
    missing = first_missing()
    if missing is not None:
        raise ImportError(f"the gmlx nodes need {missing}, which ComfyUI {comfyui_version()} "
                          "does not have")
