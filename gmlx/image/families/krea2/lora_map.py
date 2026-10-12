"""Krea 2 LoRA key map.

The GGUF keeps the names of the reference transformer, which the
reference trainer, ComfyUI and AI-Toolkit adapters target: in each block
and each text fusion block ``attn.wq``, ``attn.wk``, ``attn.wv``,
``attn.gate``, ``attn.wo``, ``mlp.gate``, ``mlp.up`` and ``mlp.down``,
and outside the blocks ``first``, ``tmlp.0``, ``tmlp.2``, ``tproj.1``,
``txtmlp.1``, ``txtmlp.3``, ``txtfusion.projector`` and ``last.linear``.
Those names map to themselves. Every projection is its own weight in the
file, so no target is a band.

An adapter in the diffusers layout, such as the Comfy-Org Krea 2 LoRAs,
names the same layers as Krea2Transformer2DModel does: ``to_q``,
``to_k``, ``to_v``, ``to_gate`` and ``to_out.0`` for the attention,
``ff.gate``, ``ff.up`` and ``ff.down`` for the MLP, under
``transformer_blocks`` and ``text_fusion``, and the diffusers names of
the layers outside the blocks. Each maps to the module of the same
weight. Some tools write ``to_out`` without the ``.0``, and that name
maps to ``attn.wo`` too. No text encoder key resolves: the map has no
encoder targets. The two final layer matrices that one converter writes
and the model does not read are no target.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

_ATTN = (("wq", "to_q"), ("wk", "to_k"), ("wv", "to_v"),
         ("gate", "to_gate"), ("wo", "to_out.0"))
_MLP = (("gate", "ff.gate"), ("up", "ff.up"), ("down", "ff.down"))
_TOP = (("first", "img_in"), ("tmlp.0", "time_embed.linear_1"),
        ("tmlp.2", "time_embed.linear_2"), ("tproj.1", "time_mod_proj"),
        ("txtmlp.1", "txt_in.linear_1"), ("txtmlp.3", "txt_in.linear_2"),
        ("txtfusion.projector", "text_fusion.projector"),
        ("last.linear", "final_layer.linear"))


def _counts(config: dict) -> tuple[int, int, int]:
    """Blocks, layerwise and refiner text blocks, from the config
    synthesized from a header or the fields of a built model's config."""
    if "n_layers" in config:
        return (int(config["n_layers"]), int(config["n_layerwise"]),
                int(config["n_refiner"]))
    return (int(config["num_layers"]),
            int(config["num_layerwise_text_blocks"]),
            int(config["num_refiner_text_blocks"]))


def lora_map(config: dict) -> LoraKeyMap:
    blocks, layerwise, refiner = _counts(config)
    targets: dict = {}
    diffusers_only = set()
    groups = (("blocks", "transformer_blocks", blocks),
              ("txtfusion.layerwise_blocks",
               "text_fusion.layerwise_blocks", layerwise),
              ("txtfusion.refiner_blocks", "text_fusion.refiner_blocks",
               refiner))
    for native, diffusers, count in groups:
        for i in range(count):
            x, d = f"{native}.{i}", f"{diffusers}.{i}"
            for leaf, dname in _ATTN:
                targets[f"{x}.attn.{leaf}"] = (f"{x}.attn.{leaf}", None)
                targets[f"{d}.attn.{dname}"] = (f"{x}.attn.{leaf}", None)
                diffusers_only.add(f"{d}.attn.{dname}")
            targets[f"{d}.attn.to_out"] = (f"{x}.attn.wo", None)
            diffusers_only.add(f"{d}.attn.to_out")
            for leaf, dname in _MLP:
                targets[f"{x}.mlp.{leaf}"] = (f"{x}.mlp.{leaf}", None)
                targets[f"{d}.{dname}"] = (f"{x}.mlp.{leaf}", None)
                diffusers_only.add(f"{d}.{dname}")
    for native, dname in _TOP:
        targets[native] = (native, None)
        targets[dname] = (native, None)
        diffusers_only.add(dname)
    return LoraKeyMap(family="krea2", title="Krea 2", targets=targets,
                      diffusers_only=frozenset(diffusers_only))
