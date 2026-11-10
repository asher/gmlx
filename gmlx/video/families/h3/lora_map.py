"""MiniMax-H3 LoRA key map.

The GGUF keeps ComfyUI's module names, which ComfyUI's generic LoRA files
target, such as lightx2v's Turbo LoRAs. Every linear leaf of the DiT maps
to itself: in each block and each token refiner block ``attn.qkv_proj``,
``attn.out_proj``, ``mlp.fc1`` and ``mlp.fc2``, in each block
``adaln_proj.linear``, and outside the blocks ``condition_proj``, the two
patch projections, the two output heads and the final adaLN linear.
FastH3 (config ``vsa_sparsity``) adds ``attn.to_gate_compress`` in each
block, its VSA gate, and only FastH3's files have that module. A
pair on the fused qkv covers the whole weight, as those files write it:
one A over the three ranks, and a B that is zero outside each
projection's rows. ComfyUI reads these names bare, under
``diffusion_model.`` and as ``lora_unet_`` underscores, and drops every
other spelling.

``refusal`` names the files gmlx does not run: a LoRA in DiffSynth's form
(keys with ``.default.``), whose qkv order and fc1 halves gmlx has not
read; one with ``.config`` keys, which ComfyUI reads as a module's config,
such as int8 or sparse attention in a block; one whose head pairs have a
multiple of the head's rows, as a PDD LoRA grows each head into a bank of
heads; and on a file without FastH3's gate, one with a pair on it.
"""

from __future__ import annotations

from gmlx.image.lora.reader import (LoraKeyMap, _dit_target, _split,
                                    _underscore_index)

_LEAVES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_BLOCK_ONLY = ("adaln_proj.linear",)
_GATE = "attn.to_gate_compress"
_HEADS = ("final_layer.video_out", "final_layer.audio_out")
_TOP = ("condition_proj", "video_patch_proj", "audio_patch_proj",
        "final_layer.adaln_proj.linear") + _HEADS
_COMFY = ("", "diffusion_model.", "lora_unet_")
COMFY_READS = ("the GGUF names bare, under diffusion_model. or as "
               "lora_unet_ underscores")


def _counts(config: dict) -> tuple[int, int]:
    """Blocks and refiner blocks, from the config synthesized from a
    header or the fields of a built model's config."""
    if "depth" in config:
        return int(config["depth"]), int(config["refiner_depth"])
    return int(config["num_layers"]), int(config["num_refiner_layers"])


def _head_rows(config: dict) -> dict:
    """The output rows of each head, from either form of config."""
    if "latent" in config:
        latent, audio, patch = (config["latent"], config["audio"],
                                config["patch"])
    else:
        latent, audio, patch = (config["in_channels"],
                                config["audio_in_channels"],
                                config["patch_size"])
    rows = int(latent)
    for p in patch:
        rows *= int(p)
    return {_HEADS[0]: rows, _HEADS[1]: int(audio)}


def _spell(prefix: str, name: str) -> str:
    return prefix + (name.replace(".", "_") if prefix.endswith("_")
                     else name)


def _gates(config: dict) -> tuple:
    """The VSA gate of each block, as FastH3's config has them."""
    blocks, _ = _counts(config)
    return tuple(f"blocks.{i}.{_GATE}" for i in range(blocks))


def lora_map(config: dict) -> LoraKeyMap:
    blocks, refiner = _counts(config)
    names = list(_TOP)
    block = _LEAVES + _BLOCK_ONLY
    if config.get("vsa_sparsity"):
        block += (_GATE,)
    for group, count, leaves in (("blocks", blocks, block),
                                 ("token_refiner.blocks", refiner, _LEAVES)):
        names += [f"{group}.{i}.{leaf}" for i in range(count)
                  for leaf in leaves]
    return LoraKeyMap(
        family="h3", title="MiniMax-H3",
        targets={n: (n, None) for n in names},
        comfy_stems=frozenset(_spell(p, n) for n in names for p in _COMFY),
        comfy_reads=COMFY_READS)


def _head_ups(header: dict):
    """(head, key, (rows, rank)) of each up matrix on an output head, in
    every spelling the LoRA reader resolves."""
    heads = {h: (h, None) for h in _HEADS}
    under = _underscore_index(heads)
    for k, info in header.items():
        split = _split(k)
        if split is None or split[1] != "up":
            continue
        hit = _dit_target(split[0], heads, under)
        shape = info.get("shape") if isinstance(info, dict) else None
        if hit is not None and isinstance(shape, list) and len(shape) >= 2 \
                and all(type(d) is int for d in shape):
            # A 1x1 kernel reads as its matrix, as the LoRA reader takes it.
            yield hit[0], k, tuple(shape[:2])


def refusal(header: dict, config: dict) -> tuple[str, str] | None:
    """Why gmlx does not run a LoRA file with this safetensors header, and
    the kind of file, or None."""
    keys = [k for k in header if k != "__metadata__"]
    if any(".default." in k for k in keys):
        return ("its keys are in DiffSynth's form (.default.), whose qkv "
                "order and fc1 halves gmlx has not read", "DiffSynth")
    cfg = [k for k in keys if k.endswith(".config")]
    if cfg:
        return (f"it sets a module config ({cfg[0]}), such as the attention "
                "method of a block, which gmlx does not apply", "config")
    if not config.get("vsa_sparsity"):
        gates = {g: (g, None) for g in _gates(config)}
        under = _underscore_index(gates)
        for k in keys:
            split = _split(k)
            if split is not None and _dit_target(split[0], gates, under):
                return (f"its pair {k} targets FastH3's attention gate, "
                        "which the model's file does not have",
                        "FastH3 gate")
    rows = _head_rows(config)
    for head, k, shape in _head_ups(header):
        n = rows[head]
        if shape[0] > n and shape[0] % n == 0:
            return (f"its pair {k} has {shape[0]} rows, {shape[0] // n} "
                    f"heads of {n}, as a PDD LoRA grows each head into a "
                    "bank of heads, and gmlx runs one plain head per "
                    "stream", "PDD")
    return None
