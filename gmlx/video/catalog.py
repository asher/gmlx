"""The video models gmlx can pull.

``RECIPES`` names each pullable video model: its DiT, text encoder and
vision tower, video VAE, audio VAE, preview decoder and LoRA, each pinned
by repository revision and sha256, with the steps, shift and clip the
recipe runs and the licenses. Every file is used as published.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# The memory classes of the video recipes, in GB of RAM. A Mac takes the
# largest class at or under its RAM.
VIDEO_MEMORY_CLASSES = (24, 32, 64, 128)


@dataclass(frozen=True)
class Pin:
    """One published file: where it is and what its bytes are. ``url``
    is set for a file outside Hugging Face."""
    repo: str
    revision: str
    file: str
    sha256: str
    size: int
    url: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.file)


@dataclass(frozen=True)
class VideoRecipe:
    name: str
    title: str
    family: str
    variant: str
    dit: Pin
    encoder: Pin
    mmproj: Pin
    vae: Pin
    audio_vae: Pin
    tiny: Pin
    lora: Pin | None
    steps: int
    shift: float                     # the video shift; audio keeps the family's
    size: str                        # WxH
    seconds: int
    license: str
    license_url: str
    license_note: str
    encoder_license: str
    vae_license: str
    tiny_license: str
    lora_license: str = ""           # "" when its card states none
    aliases: tuple = ()
    use: str = ""                    # gmlx video pull's GOOD FOR cell
    blurb: str = ""

    def pins(self) -> dict:
        """Role -> pin of every file the recipe pulls."""
        out = {"dit": self.dit, "encoder": self.encoder,
               "mmproj": self.mmproj, "vae": self.vae,
               "audio_vae": self.audio_vae, "tiny": self.tiny}
        if self.lora is not None:
            out["lora"] = self.lora
        return out


_UNSLOTH = ("unsloth/MiniMax-H3-GGUF",
            "babc040eff0084465460ead58fdb8ec44296f8dd")
_QWEN = ("unsloth/Qwen3-VL-32B-Instruct-GGUF",
         "b9262a359f54dead8e2609f6146e2fc3398fd0d9")
_COMFY = ("Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6")
_TURBO = ("lightx2v/Minimax-h3-Turbo",
          "30f485ed874aa44c8183348ad361c5eabad318da")
_TAEHV = ("madebyollin/taehv", "62f7591f59dfbb4c3c02b7a621d180a9eeaba26c")

_H3_FILES = dict(
    dit=Pin(*_UNSLOTH, "minimax_h3_fl2va_pruned-Q4_K.gguf",
            "dd948e08ad0ba3c71bd42f368e283dd82e790f5122a63b276e22a3e0283d0c10",
            11_420_663_904),
    encoder=Pin(*_QWEN, "Qwen3-VL-32B-Instruct-Q4_K_M.gguf",
                "92d605566f8661b296251c535ed028ecf81c32e14e06948a3d8bef829e96a804",
                19_762_151_200),
    mmproj=Pin(*_QWEN, "mmproj-BF16.gguf",
               "f42400deb87085f1e76159a92aedd276050c665c72423597413d341c36c18c71",
               1_200_334_496),
    vae=Pin(*_COMFY, "vae/minimax_h3_video_vae_fp16.safetensors",
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
            5_207_808_496),
    audio_vae=Pin(*_COMFY, "vae/minimax_h3_audio_vae_fp32.safetensors",
                  "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
                  605_254_808),
    tiny=Pin(*_TAEHV, "safetensors/taeh3.safetensors",
             "4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13",
             22_709_752,
             url="https://github.com/madebyollin/taehv/raw/"
                 f"{_TAEHV[1]}/safetensors/taeh3.safetensors"),
)

_TURBO_8 = Pin(*_TURBO,
               "minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors",
               "fb4c00361c94bad47f878d5e096c6cab2f45063b0968bd5756dcb7da6936f53e",
               1_956_193_000)
_TURBO_4 = Pin(*_TURBO,
               "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors",
               "70103fd0d80a947ed4e7fe6f88915863d54ef22c5cab03bd5d2fd649385ffe1e",
               1_956_193_000)

_H3 = dict(
    family="h3", variant="fl2va", **_H3_FILES, size="1344x768", seconds=5,
    license="minimax-h3-community-license-agreement",
    license_url="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/"
                "LICENSE",
    license_note=("no grant in the US, EU, UK or South Korea without "
                  "MiniMax's separate license; written authorization above "
                  "$20M yearly revenue; a commercial UI must show MiniMax H3; "
                  "hosted services need user terms and safeguards; outputs "
                  "posted publicly must be marked machine-generated"),
    encoder_license="apache-2.0",
    vae_license="minimax-h3-community-license-agreement",
    tiny_license="mit")

RECIPES = {r.name: r for r in (
    VideoRecipe(name="h3", title="MiniMax-H3 Turbo 8-step", lora=_TURBO_8,
                steps=8, shift=6.0, use="video with sound",
                blurb="Clips with sound from a prompt, in 8 steps.", **_H3),
    VideoRecipe(name="h3-fast", title="MiniMax-H3 Turbo 4-step",
                lora=_TURBO_4, steps=4, shift=6.0, use="fast drafts",
                blurb="Clips with sound in 4 steps, with softer motion.",
                **_H3),
    VideoRecipe(name="h3-base", title="MiniMax-H3", lora=None, steps=20,
                shift=12.0, use="base model, slow",
                blurb="The base model at 20 steps, no LoRA.", **_H3),
)}


def recipe(name: str) -> VideoRecipe | None:
    r = RECIPES.get(name)
    if r is not None:
        return r
    return next((r for r in RECIPES.values() if name in r.aliases), None)


def memory_class(ram_bytes: int | None) -> int:
    """The video memory class of a Mac with ``ram_bytes`` of RAM."""
    if not ram_bytes:
        return VIDEO_MEMORY_CLASSES[0]
    gb = ram_bytes / (1 << 30)
    fit = [c for c in VIDEO_MEMORY_CLASSES if c <= gb + 0.5]
    return fit[-1] if fit else VIDEO_MEMORY_CLASSES[0]
