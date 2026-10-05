"""Z-Image LoRA key map.

The GGUF keeps q, k and v as one fused ``attention.qkv`` weight, whose
output rows are q, then k, then v, each heads times head width rows. An adapter trained
on the diffusers layout targets ``to_q``, ``to_k`` and ``to_v``, which map
to bands 0, 1 and 2 of ``qkv``, and ``to_out.0``, which is ``out``. Native
names map to themselves, and the diffusers spellings of the patch embedder
and the final layer map to theirs.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

_BLOCK_NATIVE = ("attention.qkv", "attention.out", "feed_forward.w1",
                 "feed_forward.w2", "feed_forward.w3")
_TOP = ("x_embedder", "cap_embedder.1", "t_embedder.mlp.0",
        "t_embedder.mlp.2", "final_layer.linear",
        "final_layer.adaLN_modulation.1")
_DIFFUSERS_TOP = (("all_x_embedder.2-1", "x_embedder"),
                  ("all_final_layer.2-1.linear", "final_layer.linear"),
                  ("all_final_layer.2-1.adaLN_modulation.1",
                   "final_layer.adaLN_modulation.1"))
# The Qwen3-4B encoder as ComfyUI names it, and its linear leaves.
_TEXT_NAMES = ("qwen3_4b",)
_TEXT_LEAVES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                "mlp.down_proj")


def lora_map(config: dict) -> LoraKeyMap:
    inner = int(config["n_heads"]) * int(config["head_dim"])
    targets: dict = {}
    diffusers_only = set()
    groups = (("noise_refiner", config["n_refiner_layers"], True),
              ("context_refiner", config["n_refiner_layers"], False),
              ("layers", config["n_layers"], True))
    for group, count, modulated in groups:
        for i in range(int(count)):
            x = f"{group}.{i}"
            for leaf in _BLOCK_NATIVE:
                targets[f"{x}.{leaf}"] = (f"{x}.{leaf}", None)
            if modulated:
                targets[f"{x}.adaLN_modulation.0"] = (
                    f"{x}.adaLN_modulation.0", None)
            for band, name in enumerate(("to_q", "to_k", "to_v")):
                key = f"{x}.attention.{name}"
                targets[key] = (f"{x}.attention.qkv", (0, band * inner, inner))
                diffusers_only.add(key)
            targets[f"{x}.attention.to_out.0"] = (f"{x}.attention.out", None)
            diffusers_only.add(f"{x}.attention.to_out.0")
    for name in _TOP:
        targets[name] = (name, None)
    for name, module in _DIFFUSERS_TOP:
        targets[name] = (module, None)
        diffusers_only.add(name)
    from gmlx.image.text_encoder import Z_IMAGE

    text = tuple(f"model.layers.{i}.{leaf}" for i in range(Z_IMAGE.layers)
                 for leaf in _TEXT_LEAVES)
    return LoraKeyMap(family="z_image", title="Z-Image", targets=targets,
                      diffusers_only=frozenset(diffusers_only),
                      text_names=_TEXT_NAMES, text_targets=text)
