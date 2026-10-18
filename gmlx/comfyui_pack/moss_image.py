# SPDX-License-Identifier: Apache-2.0
"""MOSS-Image as a ComfyUI model, which ComfyUI has no class of its own for.

The pack registers ``MOSSImage`` with the settings of the OpenMOSS plugin:
a flow model on the FLUX.2 latent format, sampling shift 1.0 and a
timestep of sigma times 1000. Its DiT holds the manifest's tensors on the
meta device under their own names, so ComfyUI's generic LoRA key map
reads them, as it reads a model of the plugin. A conditioning keeps its
own length: rows of other lengths never join one call.
"""

from __future__ import annotations

import torch

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base

from .client import GmlxError, GmlxRefusal

# Tensors the release model and the GGUF both hold, checked before a build.
REQUIRED = ("video_embedder.weight", "text_embedder.proj.weight",
            "shared_block_modulation.weight", "final_linear_video.weight",
            "layers.0.proj.ffn_up_gate.weight", "text_embedder.blocks.0.qkv.weight")


class WeightlessDiT(torch.nn.Module):
    """The DiT's tensors on the meta device under the manifest's names.
    The pack replaces its forward with the step call."""

    def __init__(self, tensors=(), dtype=None, device=None, operations=None, **_):
        super().__init__()
        self.dtype = dtype or torch.float32
        for name, shape in tensors:
            *path, leaf = name.split(".")
            mod = self
            for part in path:
                child = mod._modules.get(part)
                if child is None:
                    child = torch.nn.Module()
                    mod.add_module(part, child)
                mod = child
            mod.register_parameter(leaf, torch.nn.Parameter(
                torch.empty(tuple(shape), dtype=self.dtype, device="meta"),
                requires_grad=False))
        self.blocks = sum(1 for n, _ in tensors
                          if n.startswith("layers.") and n.endswith(".proj.q.weight"))

    def forward(self, *args, **kwargs):
        raise GmlxError("gmlx: the MOSS-Image DiT runs on the server")


class MOSSImage(comfy.model_base.BaseModel):
    def __init__(self, model_config, device=None):
        super().__init__(model_config, comfy.model_base.ModelType.FLOW, device=device,
                         unet_model=WeightlessDiT)

    def extra_conds(self, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is not None and not bool(torch.all(mask != 0)):
            raise GmlxRefusal("gmlx: this conditioning masks some of its text rows, which a "
                              "gmlx MOSS-Image model reads all of; encode it with the gmlx CLIP")
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn")
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)
        return out


class MOSSImageConfig(comfy.supported_models_base.BASE):
    sampling_settings = {"shift": 1.0, "multiplier": 1000}
    latent_format = comfy.latent_formats.Flux2
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 4.0

    def get_model(self, state_dict, prefix="", device=None):
        return MOSSImage(self, device=device)


def model_config(manifest: dict) -> MOSSImageConfig:
    """The config of the weightless model of a MOSS-Image manifest."""
    names = {t["name"] for t in manifest["tensors"]}
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        raise GmlxError(f"gmlx: the server's manifest of {manifest.get('model')!r} is not "
                        f"a MOSS-Image DiT: it has no {', '.join(missing)}")
    return MOSSImageConfig({"tensors": tuple((t["name"], tuple(t["shape"]))
                                             for t in manifest["tensors"])})


# The models the pack builds itself, by the class name the server gives.
PACK_MODELS = {"MOSSImage": model_config}
