# SPDX-License-Identifier: Apache-2.0
"""The gmlx nodes for ComfyUI.

The loaders give a stock MODEL, CLIP and VAE whose work runs on a gmlx
server, so stock sampler, LoRA, conditioning and image nodes compose with
them. MiniMax-H3 has a video VAE and an audio VAE, each with its own
loader, and ``GmlxH3ImageToVideo`` runs ComfyUI's MiniMax-H3 prompt node
with the clip's length and keyframes told to the server. ``GmlxSigmas``
gives the schedule a model variant was distilled to, or the schedule of
the image size for a variant whose schedule follows it, for the custom
sampler nodes.
"""

from __future__ import annotations

import torch

import comfy.samplers

from . import discovery, schedules
from .client import GmlxError, GmlxRefusal, get_client
from .clip import GMLX_NODE, GmlxCLIP
from .model import load_model
from .vae import GmlxAudioVAE, GmlxVAE, GmlxVideoVAE
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
    models = [m for m in client.models() if isinstance(m, dict)]
    if name not in [str(m.get("model")) for m in models]:
        msg = discovery.not_on_server(name, client.base_url, models)
        note = client.redirect_note
        raise GmlxRefusal(f"{msg}\nNote: {note}." if note else msg)
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

    @classmethod
    def INPUT_TYPES(cls):
        return discovery.model_loader_inputs()

    def load(self, model_name, server_url="", start_server=True, int8="entry"):
        client, name = connect(model_name, server_url, start_server)
        return (load_model(client, name, int8),)


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
        if status.get("audio"):
            raise GmlxRefusal(f"gmlx: {name!r} has a video VAE and an audio VAE; load them "
                              f"with the {VIDEO_VAE} and the {AUDIO_VAE}")
        return (GmlxVAE(client, name, status),)


VIDEO_VAE, AUDIO_VAE = "gmlx Video VAE Loader", "gmlx Audio VAE Loader"


def _video_status(client, name: str) -> dict:
    status = client.status(name)
    if not status.get("audio"):
        raise GmlxRefusal(f"gmlx: {name!r} is an image model; the {VIDEO_VAE} and the "
                          f"{AUDIO_VAE} take a MiniMax-H3 model, and the gmlx VAE Loader "
                          "takes this one")
    return status


class GmlxVideoVAELoader(_Loader):
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The video VAE of a gmlx MiniMax-H3 model, run on the gmlx server, for "
                   "VAE Decode.")

    def load(self, model_name, server_url="", start_server=True):
        client, name = connect(model_name, server_url, start_server)
        return (GmlxVideoVAE(client, name, _video_status(client, name)),)


class GmlxAudioVAELoader(_Loader):
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The audio VAE of a gmlx MiniMax-H3 model, run on the gmlx server, for "
                   "VAE Decode Audio.")

    def load(self, model_name, server_url="", start_server=True):
        client, name = connect(model_name, server_url, start_server)
        return (GmlxAudioVAE(client, name, _video_status(client, name)),)


def variant_sigmas(model, steps: int = 0, denoise: float = 1.0, width: int = 0,
                   height: int = 0) -> torch.Tensor:
    """The sigmas of a gmlx model's variant: the fixed schedule it was
    distilled to, the schedule of a ``width`` by ``height`` image for a
    variant whose schedule follows the image size, or ComfyUI's simple
    schedule over the variant's steps."""
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
    try:
        sized = schedules.size_sigmas(b.status, b.model_id, steps, denoise,
                                      int(width or 0), int(height or 0))
    except ValueError as e:
        raise GmlxRefusal(str(e) if str(e).startswith("gmlx:")
                          else f"gmlx: {b.model_id!r}: {e}") from None
    if sized is not None:
        return torch.from_numpy(sized.copy())
    total = steps if denoise >= 1.0 else int(steps / max(denoise, 1e-4))
    sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), "simple",
                                          total).cpu()
    return sig[-(steps + 1):]


_SIZE_TIP = ("The image's width or height in pixels, which the schedule of FLUX.2 klein "
             "and Qwen-Image 2.1 follows. 0 gives none.")


class GmlxSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "steps": ("INT", {"default": 0, "min": 0, "max": 1000,
                              "tooltip": "0 uses the variant's own step count."}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})},
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                  "tooltip": _SIZE_TIP}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                   "tooltip": _SIZE_TIP})}}

    RETURN_TYPES = ("SIGMAS", "FLOAT")
    RETURN_NAMES = ("sigmas", "cfg")
    FUNCTION = "get_sigmas"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The schedule of a gmlx model's variant, for SamplerCustom. A variant "
                   "distilled to fixed sigmas samples on those, and one whose schedule "
                   "follows the image size, such as FLUX.2 klein, on that of width and "
                   "height.")

    def get_sigmas(self, model, steps=0, denoise=1.0, width=0, height=0):
        b = getattr(model, "gmlx", None)
        cfg = float(((b.status.get("sampling") or {}).get("cfg")) or 1.0) if b else 1.0
        return (variant_sigmas(model, steps, denoise, width, height), cfg)


_FRAMES_TIP = ("Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; "
               "trained range is ~124-362, longer is untested)")


class GmlxH3ImageToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        size = {"min": 32, "max": 16384, "step": 32}
        return {"required": {
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "width": ("INT", {"default": 1344, **size}),
            "height": ("INT", {"default": 768, **size}),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17,
                               "tooltip": _FRAMES_TIP})},
            "optional": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)}}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "LATENT")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("ComfyUI's MiniMax H3 Image to Video, which also tells the gmlx server the "
                   "clip's length and its first and last frames, so an upsampler on the "
                   "model's entry rewrites the prompt as POST /v1/videos does.")

    def encode(self, clip, vae, prompt, width, height, length, first_frame=None,
               last_frame=None):
        from . import compat

        missing = compat.first_missing(compat.H3_NODE_NAMES)
        if missing:
            from .version import COMFYUI_FLOOR, COMFYUI_TESTED

            raise GmlxError(f"gmlx: ComfyUI {compat.comfyui_version()} has no {missing}, which "
                            f"{GMLX_NODE} runs; the gmlx nodes are tested on ComfyUI "
                            f"{COMFYUI_FLOOR} to {COMFYUI_TESTED}")
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, temporal_shape

        if isinstance(clip, GmlxCLIP) and clip.gmlx_client is not None:
            # The stock node shows the encoder the first frame, then the last.
            keyframes = (["first"] if first_frame is not None else []) + (
                ["last"] if last_frame is not None else [])
            clip = clip.clone()
            clip.gmlx_clip = {"frames": int(temporal_shape(length)[0]), "keyframes": keyframes}
        out = MiniMaxH3ImageToVideo.execute(clip, vae, prompt, width, height, length,
                                            first_frame=first_frame, last_frame=last_frame)
        return tuple(out.result if hasattr(out, "result") else out)


class GmlxServerStatus(_Loader):
    RETURN_TYPES = ("STRING",)
    FUNCTION = "status"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("The gmlx server's view of an image or video model: residency and memory. "
                   "The gmlx nodes are managed by gmlx and update with it.")

    def status(self, model_name, server_url="", start_server=False):
        note = redirect = None
        try:
            client, name = connect(model_name, server_url, start_server)
            s = client.status(name)
            note = client.version_note
            redirect = client.redirect_note
        except GmlxError as e:
            text = str(e)
        else:
            mem = s.get("memory") or {}
            gib = {k: round(v / (1 << 30), 2) for k, v in mem.items() if isinstance(v, (int, float))}
            text = (f"{s.get('model')}: {s.get('family')} {s.get('variant') or ''}, "
                    f"resident {s.get('resident')}, memory GiB {gib}")
        lines = ([text] + ([f"gmlx: {redirect}."] if redirect else [])
                 + ([f"gmlx: {note}"] if note else []) + [MANAGED])
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "GmlxModelLoader": GmlxModelLoader,
    "GmlxCLIPLoader": GmlxCLIPLoader,
    "GmlxVAELoader": GmlxVAELoader,
    "GmlxVideoVAELoader": GmlxVideoVAELoader,
    "GmlxAudioVAELoader": GmlxAudioVAELoader,
    "GmlxSigmas": GmlxSigmas,
    "GmlxH3ImageToVideo": GmlxH3ImageToVideo,
    "GmlxServerStatus": GmlxServerStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GmlxModelLoader": "gmlx Model Loader",
    "GmlxCLIPLoader": "gmlx CLIP Loader",
    "GmlxVAELoader": "gmlx VAE Loader",
    "GmlxVideoVAELoader": VIDEO_VAE,
    "GmlxAudioVAELoader": AUDIO_VAE,
    "GmlxSigmas": "gmlx Sigmas",
    "GmlxH3ImageToVideo": GMLX_NODE,
    "GmlxServerStatus": "gmlx Server Status",
}
