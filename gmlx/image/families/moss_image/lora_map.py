"""MOSS-Image LoRA key map.

The GGUF keeps the names of the release's MOSSImageTransformer, which
the release trains under, the OpenMOSS ComfyUI plugin loads and PEFT
names under ``transformer.``. In each block those are ``proj.q``,
``proj.k``, ``proj.v``, ``proj.o``, ``proj.ffn_up_gate`` and
``proj.ffn_down``; in each text refiner block of ``text_embedder`` they
are ``qkv``, ``proj``, ``ffn_gate``, ``ffn_up`` and ``ffn_down``; and
outside the blocks they are ``video_embedder``, ``text_embedder.proj``,
``time_embedding.0``, ``time_embedding.2``, ``shared_block_modulation``,
``video_head_modulation`` and ``final_linear_video``. Each name maps to
itself. The fused weights take adapters under their own names: every
known layout holds ``proj.ffn_up_gate`` and the refiner ``qkv`` as one
module, so no target is a band. No text encoder key resolves: the map
has no encoder targets.

ComfyUI reads the adapter of a model it has no class of its own for
under ``diffusion_model.`` and ``lora_unet_``, in these names only.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

_BLOCK = ("proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate",
          "proj.ffn_down")
_REFINER = ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")
_TOP = ("video_embedder", "text_embedder.proj", "time_embedding.0",
        "time_embedding.2", "shared_block_modulation",
        "video_head_modulation", "final_linear_video")
# The prefixes under which ComfyUI's key map finds a MOSS-Image adapter.
COMFY_PREFIXES = ("lora_unet_", "diffusion_model.")


def _counts(config: dict) -> tuple[int, int]:
    """Blocks and text refiner blocks, from the config synthesized from a
    header or the fields of a built model's config."""
    if "depth" in config:
        return int(config["depth"]), int(config["text_depth"])
    return int(config["num_layers"]), int(config.get("num_text_blocks", 2))


def lora_map(config: dict) -> LoraKeyMap:
    blocks, refiners = _counts(config)
    names = [f"layers.{i}.{leaf}" for i in range(blocks) for leaf in _BLOCK]
    names += [f"text_embedder.blocks.{i}.{leaf}" for i in range(refiners)
              for leaf in _REFINER]
    names += list(_TOP)
    return LoraKeyMap(family="moss_image", title="MOSS-Image",
                      targets={n: (n, None) for n in names},
                      comfy_prefixes=COMFY_PREFIXES)
