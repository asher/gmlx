# SPDX-License-Identifier: Apache-2.0
"""ComfyUI VAEs that encode and decode on the gmlx server.

They set the attributes stock nodes read and send latents in the space
ComfyUI's LATENT carries, the raw VAE space or, for FLUX.2, FLUX.3 and
MiniMax-H3, the DiT's own model space, and images as (B, H, W, C) floats in
[0, 1]. The server tiles large images itself, so the tiled calls are the
plain ones. MiniMax-H3 has a video VAE and an audio VAE, one node each.
"""

from __future__ import annotations

import torch

import comfy.model_patcher
import comfy.sd

from .client import GmlxRefusal, StepClient
from .patcher import CPU, H3_TEXT_ONLY


class _StubDecoder(torch.nn.Module):
    pass


class _ServerVAE(comfy.sd.VAE):
    """What every gmlx VAE sets: no weights, the cpu device, f32."""

    def _bind(self, client: StepClient, model_id: str) -> None:
        self.gmlx_client, self.gmlx_model = client, model_id
        self.pad_channel_value = None
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

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None,
                     overlap_t=None):
        return self.decode(samples)

    def encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None,
                     overlap_t=None):
        return self.encode(pixel_samples)

    def get_sd(self):
        raise GmlxRefusal("gmlx: a gmlx VAE has no weights in ComfyUI to save")


class GmlxVAE(_ServerVAE):
    def __init__(self, client: StepClient, model_id: str, status: dict):
        lat = status.get("latent") or {}
        vae = status.get("vae") or {}
        self._bind(client, model_id)
        self.latent_channels = int(lat.get("channels", 16))
        stride = int(vae.get("stride", lat.get("stride", 8)))
        self.downscale_ratio = stride
        self.upscale_ratio = stride
        self.output_channels = int(vae.get("channels_out", 3))
        self.pad_channel_value = 1.0 if self.output_channels == 4 else None
        self.latent_dim = 2

    def decode(self, samples_in, vae_options={}):
        if samples_in.ndim == 5:
            samples_in = samples_in[:, :, 0]
        img = self.gmlx_client.vae_decode(self.gmlx_model, samples_in.detach().float().cpu(),
                                          alpha=self.output_channels == 4)
        return img.float()

    def encode(self, pixel_samples):
        pixels = self.vae_encode_crop_pixels(pixel_samples.detach().float().cpu())
        return self.gmlx_client.vae_encode(self.gmlx_model, pixels).float()


def _no_encode(kind: str):
    raise GmlxRefusal(f"gmlx: a gmlx {kind} VAE decodes and does not encode; {H3_TEXT_ONLY}")


class GmlxVideoVAE(_ServerVAE):
    """MiniMax-H3's video VAE: model-space latents (1, 24, T, h, w) to
    frames, at 17 k + 5 frames for 5 k + 2 latents and 16x in space."""

    def __init__(self, client: StepClient, model_id: str, status: dict):
        lat = status.get("latent") or {}
        vae = status.get("vae") or {}
        self._bind(client, model_id)
        stride = int(vae.get("stride", lat.get("stride", 16)))
        self.latent_channels = int(lat.get("channels", 24))
        self.output_channels = int(vae.get("channels_out", 3))
        self.latent_dim = 3
        self.upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), stride, stride)
        self.downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1,
                                stride, stride)
        self.upscale_index_formula = (4, stride, stride)
        self.downscale_index_formula = (4, stride, stride)
        self.handles_tiling = True

    def decode(self, samples_in, vae_options={}):
        frames = self.gmlx_client.video_decode(self.gmlx_model,
                                               samples_in.detach().float().cpu())
        return frames.float()[None]

    def encode(self, pixel_samples):
        _no_encode("video")


class GmlxAudioVAE(_ServerVAE):
    """MiniMax-H3's audio VAE: model-space latents (1, 32, 2, T) to a stereo
    waveform at 32 kHz, 800 samples a latent."""

    def __init__(self, client: StepClient, model_id: str, status: dict):
        audio = status.get("audio") or {}
        self._bind(client, model_id)
        self.latent_channels = int(audio.get("channels", 32))
        self.output_channels = int(audio.get("stereo", 2))
        self.audio_sample_rate = int(audio.get("sample_rate", 32000))
        self.upscale_ratio = self.audio_sample_rate // int(audio.get("latents_per_second", 40))
        self.downscale_ratio = self.upscale_ratio
        self.latent_dim = 2

    def decode(self, samples_in, vae_options={}):
        wave = self.gmlx_client.audio_decode(self.gmlx_model, samples_in.detach().float().cpu())
        # (1, 2, samples) to the (B, samples, channels) a stock decode gives.
        return wave.float().movedim(1, -1)

    def encode(self, pixel_samples):
        _no_encode("audio")
