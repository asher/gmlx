# SPDX-License-Identifier: Apache-2.0
"""The nodes of ComfyUI-gmlx.

The loaders give a stock MODEL, CLIP and VAE whose work runs on a gmlx
server, so stock sampler, LoRA, conditioning and image nodes compose with
them. ``GmlxSigmas`` gives the schedule a model variant was distilled to,
for the custom sampler nodes.
"""

from __future__ import annotations

import os

import torch

import comfy.samplers

from .client import GmlxError, GmlxRefusal, get_client
from .clip import GmlxCLIP
from .model import load_model
from .vae import GmlxVAE

CATEGORY = "gmlx"


def _default_model() -> str:
    return os.environ.get("GMLX_DEFAULT_IMAGE_MODEL") or "z-image-turbo"


def _common():
    return {"model_name": ("STRING", {"default": _default_model(),
                                      "tooltip": "The image model id in the gmlx server's config."}),
            "server_url": ("STRING", {"default": "",
                                      "tooltip": "Empty uses GMLX_BASE_URL, the settings file, "
                                                 "gmlx status, then http://127.0.0.1:8080."}),
            "start_server": ("BOOLEAN", {"default": True,
                                         "tooltip": "Run gmlx serve when no server answers."})}


class GmlxModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common()}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "A diffusion model whose steps run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client = get_client(server_url, start_server=start_server)
        return (load_model(client, model_name.strip()),)


class GmlxCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common()}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "The text encoder of a gmlx image model, run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client = get_client(server_url, start_server=start_server)
        status = client.status(model_name.strip())
        return (GmlxCLIP(client, model_name.strip(), status.get("family")),)


class GmlxVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common()}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "The VAE of a gmlx image model, run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client = get_client(server_url, start_server=start_server)
        status = client.status(model_name.strip())
        return (GmlxVAE(client, model_name.strip(), status),)


def variant_sigmas(model, steps: int = 0, denoise: float = 1.0) -> torch.Tensor:
    """The sigmas of a gmlx model's variant: the fixed schedule it was
    distilled to, or ComfyUI's simple schedule over the variant's steps."""
    b = getattr(model, "gmlx", None)
    if b is None:
        raise GmlxRefusal("gmlx: GmlxSigmas takes a MODEL from the gmlx model loader")
    sampling = b.status.get("sampling") or {}
    fixed = sampling.get("sigmas")
    if fixed:
        sig = torch.tensor([float(s) for s in fixed], dtype=torch.float32)
        n = len(sig) - 1
        if steps and steps != n:
            raise GmlxRefusal(f"gmlx: {b.model_id!r} samples on a fixed schedule of {n} steps; "
                              "set steps to 0 or to that count")
        keep = n if denoise >= 1.0 else max(1, round(n * denoise))
        return sig[-(keep + 1):]
    steps = steps or int(sampling.get("steps") or 20)
    total = steps if denoise >= 1.0 else int(steps / max(denoise, 1e-4))
    sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple",
                                          total).cpu()
    return sig[-(steps + 1):]


class GmlxSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 0, "min": 0, "max": 1000,
                              "tooltip": "0 uses the variant's own step count."}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})}}

    RETURN_TYPES = ("SIGMAS", "FLOAT")
    RETURN_NAMES = ("sigmas", "cfg")
    FUNCTION = "get_sigmas"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The schedule of a gmlx model's variant, for SamplerCustom. A variant "
                   "distilled to fixed sigmas samples on those.")

    def get_sigmas(self, model, steps=0, denoise=1.0):
        b = getattr(model, "gmlx", None)
        cfg = float(((b.status.get("sampling") or {}).get("cfg")) or 1.0) if b else 1.0
        return (variant_sigmas(model, steps, denoise), cfg)


class GmlxServerStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common()}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "status"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "The gmlx server's view of an image model: residency and memory."

    def status(self, model_name, server_url="", start_server=False):
        try:
            client = get_client(server_url, start_server=start_server)
            s = client.status(model_name.strip())
        except GmlxError as e:
            text = str(e)
        else:
            mem = s.get("memory") or {}
            gib = {k: round(v / (1 << 30), 2) for k, v in mem.items() if isinstance(v, (int, float))}
            text = (f"{s.get('model')}: {s.get('family')} {s.get('variant') or ''}, "
                    f"resident {s.get('resident')}, memory GiB {gib}")
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "GmlxModelLoader": GmlxModelLoader,
    "GmlxCLIPLoader": GmlxCLIPLoader,
    "GmlxVAELoader": GmlxVAELoader,
    "GmlxSigmas": GmlxSigmas,
    "GmlxServerStatus": GmlxServerStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GmlxModelLoader": "gmlx Model Loader",
    "GmlxCLIPLoader": "gmlx CLIP Loader",
    "GmlxVAELoader": "gmlx VAE Loader",
    "GmlxSigmas": "gmlx Sigmas",
    "GmlxServerStatus": "gmlx Server Status",
}
