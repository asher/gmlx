# SPDX-License-Identifier: Apache-2.0
"""A ComfyUI VAE that encodes and decodes on the gmlx server.

It sets the attributes stock nodes read and sends latents in the space
ComfyUI's LATENT carries, the raw VAE space or, for FLUX.2 and FLUX.3, the
DiT's own model space, and images as (B, H, W, C) floats in [0, 1]. The server tiles
large images itself, so the tiled calls are the plain ones.
"""

from __future__ import annotations

import torch

import comfy.model_patcher
import comfy.sd

from .client import GmlxRefusal, StepClient
from .patcher import CPU


class _StubDecoder(torch.nn.Module):
    pass


class GmlxVAE(comfy.sd.VAE):
    def __init__(self, client: StepClient, model_id: str, status: dict):
        lat = status.get("latent") or {}
        vae = status.get("vae") or {}
        self.gmlx_client, self.gmlx_model = client, model_id
        self.latent_channels = int(lat.get("channels", 16))
        stride = int(vae.get("stride", lat.get("stride", 8)))
        self.downscale_ratio = stride
        self.upscale_ratio = stride
        self.output_channels = int(vae.get("channels_out", 3))
        self.pad_channel_value = 1.0 if self.output_channels == 4 else None
        self.latent_dim = 2
        self.crop_input = True
        self.not_video = False
        self.extra_1d_channel = None
        self.handles_tiling = False
        self.format_encoded = None
        self.disable_offload = True
        self.working_dtypes = [torch.float32]
        self.vae_dtype = torch.float32
        self.device = CPU
        self.output_device = CPU
        self.process_input = lambda image: image
        self.process_output = lambda image: image
        self.memory_used_encode = lambda shape, dtype: 1
        self.memory_used_decode = lambda shape, dtype: 1
        self.first_stage_model = _StubDecoder()
        self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=CPU,
                                                        offload_device=CPU, size=1)

    def throw_exception_if_invalid(self):
        return None

    def vae_output_dtype(self):
        return torch.float32

    def is_dynamic(self):
        return False

    def decode(self, samples_in, vae_options={}):
        if samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]
        img = self.gmlx_client.vae_decode(self.gmlx_model, samples_in.detach().float().cpu(),
                                          alpha=self.output_channels == 4)
        return img.float()

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None,
                     overlap_t=None):
        return self.decode(samples)

    def encode(self, pixel_samples):
        pixels = self.vae_encode_crop_pixels(pixel_samples.detach().float().cpu())
        return self.gmlx_client.vae_encode(self.gmlx_model, pixels).float()

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None,
                     overlap_t=None):
        return self.encode(pixel_samples)

    def get_sd(self):
        raise GmlxRefusal("gmlx: a gmlx VAE has no weights in ComfyUI to save")
