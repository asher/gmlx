"""Qwen-Image 2.1 LoRA key map.

An adapter names the linear layers of the reference transformer: in each
block ``attn.to_q``, ``attn.to_k``, ``attn.to_v``, ``attn.to_out.0``,
``img_mlp.gate_layer``, ``img_mlp.proj`` and ``img_mlp.out``, and outside
the blocks the two embedders, the timestep embedder, the modulation and
the final layers. The GGUF has the same names but for the image MLP,
which it keeps as one fused ``img_mlp.gate_up`` weight whose output rows
are the gate rows, then the projection rows. On that layout
``gate_layer`` and ``proj`` map to bands 0 and 1 of ``gate_up``, each of
the MLP width in rows, and ``gate_up`` itself names the whole weight. A
file that stores the pair apart builds the two leaves, each name maps to
its own, and a ``gate_up`` key is refused with the name of that layout.

The reader spells every name here with underscores too, which covers the
flat names that the reference LoRA converter reads, such as
``norm_out_linear``. No text encoder key resolves: the map has no encoder
targets. It names no target as one that only the reference layout has,
so the reader gives no warning about another program's key map.

An ``alpha`` tensor scales its pair by alpha over rank in every spelling
of the pair's keys, as ComfyUI reads it. The reference loader folds that
factor into a pair spelled ``lora_down`` and ``lora_up`` and drops the
alpha of a pair spelled ``lora_A`` and ``lora_B``, which it runs at scale
1. A file with those keys and alpha tensors therefore gives another image
in the reference pipeline than in gmlx and in ComfyUI.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

_ATTN = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")
_TOP = ("img_in", "txt_in.in_layer", "txt_in.out_layer",
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2", "modulation.1",
        "norm_out.linear", "proj_out")
# The pair of the image MLP in the order of its rows in the fused weight.
_GATE_UP = ("img_mlp.gate_layer", "img_mlp.proj")
# What a gate_up key names on a file that stores the pair apart.
_APART = ("the fused img_mlp.gate_up weight, and this DiT file stores the "
          "gate and the projection apart, as img_mlp.gate_layer and "
          "img_mlp.proj")


def _shape(config: dict) -> tuple[int, int, bool]:
    """Blocks, MLP width and whether the MLP is fused. ``config`` is the
    config synthesized from a header, which a file is checked with before
    a model loads, or the fields of a built model's config."""
    if "mlp_layout" in config:
        return (int(config["num_layers"]), int(config["ffn_dim"]),
                config["mlp_layout"] == "fused")
    return (int(config["n_layers"]), int(config["ffn_dim"]),
            bool(config["fused_mlp"]))


def lora_map(config: dict) -> LoraKeyMap:
    blocks, ffn, fused = _shape(config)
    targets: dict = {}
    absent: dict = {}
    for i in range(blocks):
        x = f"transformer_blocks.{i}"
        for leaf in (*_ATTN, "img_mlp.out"):
            targets[f"{x}.{leaf}"] = (f"{x}.{leaf}", None)
        if fused:
            targets[f"{x}.img_mlp.gate_up"] = (f"{x}.img_mlp.gate_up", None)
        else:
            absent[f"{x}.img_mlp.gate_up"] = _APART
        for band, leaf in enumerate(_GATE_UP):
            key = f"{x}.{leaf}"
            targets[key] = ((f"{x}.img_mlp.gate_up", (0, band * ffn, ffn))
                            if fused else (key, None))
    for name in _TOP:
        targets[name] = (name, None)
    return LoraKeyMap(family="qwen_image21", title="Qwen-Image 2.1",
                      targets=targets, absent=absent)
