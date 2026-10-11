# SPDX-License-Identifier: Apache-2.0
"""The gmlx nodes for ComfyUI.

The loaders give a stock MODEL, CLIP and VAE whose work runs on a gmlx
server, so stock sampler, LoRA, conditioning and image nodes compose with
them. ``GmlxSigmas`` gives the schedule a model variant was distilled to,
for the custom sampler nodes.
"""

from __future__ import annotations

import torch

import comfy.samplers

from . import discovery
from .client import GmlxError, GmlxRefusal, get_client
from .clip import GmlxCLIP
from .model import load_model
from .vae import GmlxVAE
from .version import __version__

CATEGORY = "gmlx"

MANAGED = (f"These gmlx nodes are from gmlx {__version__}. gmlx manages them and updates "
           "them with itself. Comfy Desktop does not list them, and ComfyUI-Manager lists "
           "them as an unknown pack that it cannot update.")


def _common():
    return discovery.loader_inputs()["required"]


def connect(model_name, server_url="", start_server=True):
    """The client of the node's server and the model name, after checking
    that the server has that image model. A name it lacks is refused with
    the `gmlx pull` that adds it."""
    name = str(model_name).strip()
    client = get_client(server_url, start_server=start_server)
    ids = [str(m.get("model")) for m in client.models() if isinstance(m, dict)]
    if name not in ids:
        raise GmlxRefusal(discovery.not_on_server(name, client.base_url, ids))
    return client, name


class _Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _common()}

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        return discovery.validate(model_name)


class GmlxModelLoader(_Loader):
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "A diffusion model whose steps run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client, name = connect(model_name, server_url, start_server)
        return (load_model(client, name),)


class GmlxCLIPLoader(_Loader):
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "The text encoder of a gmlx image model, run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client, name = connect(model_name, server_url, start_server)
        status = client.status(name)
        return (GmlxCLIP(client, name, status.get("family")),)


class GmlxVAELoader(_Loader):
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "The VAE of a gmlx image model, run on the gmlx server."

    def load(self, model_name, server_url="", start_server=True):
        client, name = connect(model_name, server_url, start_server)
        status = client.status(name)
        return (GmlxVAE(client, name, status),)


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


class GmlxServerStatus(_Loader):
    RETURN_TYPES = ("STRING",)
    FUNCTION = "status"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("The gmlx server's view of an image model: residency and memory. The "
                   "gmlx nodes are managed by gmlx and update with it.")

    def status(self, model_name, server_url="", start_server=False):
        note = None
        try:
            client, name = connect(model_name, server_url, start_server)
            s = client.status(name)
            note = client.version_note
        except GmlxError as e:
            text = str(e)
        else:
            mem = s.get("memory") or {}
            gib = {k: round(v / (1 << 30), 2) for k, v in mem.items() if isinstance(v, (int, float))}
            text = (f"{s.get('model')}: {s.get('family')} {s.get('variant') or ''}, "
                    f"resident {s.get('resident')}, memory GiB {gib}")
        lines = [text] + ([f"gmlx: {note}"] if note else []) + [MANAGED]
        text = "\n".join(lines)
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
