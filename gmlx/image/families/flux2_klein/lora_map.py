"""FLUX.2 klein LoRA key map.

The GGUF keeps the names of BFL's transformer, and an adapter trained on
that transformer, such as an AI-Toolkit or a kohya file, names the same
layers: in each double block ``img_attn.qkv``, ``img_attn.proj``,
``img_mlp.0`` and ``img_mlp.2`` and the same four of ``txt_``, in each
single block ``linear1`` and ``linear2``, and outside the blocks
``img_in``, ``txt_in``, ``time_in.in_layer``, ``time_in.out_layer``, the
``lin`` of ``double_stream_modulation_img``,
``double_stream_modulation_txt`` and ``single_stream_modulation``,
``final_layer.linear`` and ``final_layer.adaLN_modulation.1``. Each name
maps to itself. ``img_mlp.0`` holds the gate rows, then the up rows, and
``linear1`` the q, k and v rows, then the gate and up rows of its MLP:
a pair on either is one delta over all of its rows.

An adapter in the diffusers layout names the layers of
Flux2Transformer2DModel, which splits the fused q, k and v weight of a
double block. ``attn.to_q``, ``attn.to_k`` and ``attn.to_v`` are bands 0,
1 and 2 of ``img_attn.qkv``, each of the width in rows, and
``attn.add_q_proj``, ``attn.add_k_proj`` and ``attn.add_v_proj`` the
bands of ``txt_attn.qkv``. ``attn.to_out.0`` is ``img_attn.proj``, also
spelled ``attn.to_out``, and ``attn.to_add_out`` is ``txt_attn.proj``.
``ff.linear_in``, ``ff.linear_out`` and their ``ff_context`` twins are
the MLPs. In a single block ``attn.to_qkv_mlp_proj`` is ``linear1`` and
``attn.to_out`` is ``linear2``. Outside the blocks the names are
``x_embedder``, ``context_embedder``,
``time_guidance_embed.timestep_embedder.linear_1`` and ``linear_2``, the
``linear`` of each modulation, ``norm_out.linear`` and ``proj_out``.
diffusers' converter swaps the shift and the scale halves of the final
modulation, so ``norm_out.linear`` holds the scale rows first, and the
halves of the up rows of a pair on it are swapped back. ComfyUI also
reads ``diffusion_model.single_blocks.N.linear1_qkv`` as the q, k and v
rows of ``linear1``, and so does gmlx.

ComfyUI reads the GGUF names under ``diffusion_model.`` and
``lora_unet_``, and the diffusers names of its FLUX table under
``transformer.``, under no prefix, and with underscores under
``lycoris_`` and ``lora_transformer_``. That table has no diffusers name
for the three modulations or for the time embedder of FLUX.2, so ComfyUI
drops those pairs, and the reader warns of each stem that ComfyUI drops.

The diffusers loader converts a file of GGUF names only when every key
starts with ``diffusion_model.``. It maps a pair on
``final_layer.adaLN_modulation.1`` to ``norm_out.linear`` without the
swap, so such a pair gives another image there than in gmlx and in
ComfyUI. Its kohya converter skips the modulation and final modulation
pairs with a warning. No text encoder key resolves: the map has no
encoder targets.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

# The leaves of a double block, by GGUF name and diffusers name.
_DOUBLE = (("img_attn.proj", "attn.to_out.0"),
           ("txt_attn.proj", "attn.to_add_out"),
           ("img_mlp.0", "ff.linear_in"), ("img_mlp.2", "ff.linear_out"),
           ("txt_mlp.0", "ff_context.linear_in"),
           ("txt_mlp.2", "ff_context.linear_out"))
# The bands of each fused q, k and v weight, in the order of its rows.
_QKV = (("img_attn.qkv", ("attn.to_q", "attn.to_k", "attn.to_v")),
        ("txt_attn.qkv", ("attn.add_q_proj", "attn.add_k_proj",
                          "attn.add_v_proj")))
_SINGLE = (("linear1", "attn.to_qkv_mlp_proj"), ("linear2", "attn.to_out"))
NORM_OUT = "norm_out.linear"
_TOP = (("img_in", "x_embedder"), ("txt_in", "context_embedder"),
        ("final_layer.linear", "proj_out"),
        ("final_layer.adaLN_modulation.1", NORM_OUT))
# The layers outside the blocks that ComfyUI reads by GGUF name alone.
_TOP_GGUF_ONLY = (
    ("time_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1"),
    ("time_in.out_layer", "time_guidance_embed.timestep_embedder.linear_2"),
    *((f"{m}.lin", f"{m}.linear")
      for m in ("double_stream_modulation_img",
                "double_stream_modulation_txt", "single_stream_modulation")))
COMFY_GGUF = ("diffusion_model.", "lora_unet_")
COMFY_DIFFUSERS = ("transformer.", "", "lycoris_", "lora_transformer_")
COMFY_READS = ("the GGUF names under diffusion_model. or lora_unet_, and "
               "the diffusers names of its FLUX table, which has none for "
               "the modulations or the time embedder")


def _shape(config: dict) -> tuple[int, int, int]:
    """Double blocks, single blocks and the width. ``config`` is the
    config synthesized from a header, which a file is checked with before
    a model loads, or the fields of a built model's config."""
    if "num_layers" in config:
        return (int(config["num_layers"]), int(config["num_single_layers"]),
                int(config["num_attention_heads"])
                * int(config["attention_head_dim"]))
    return (int(config["depth"]), int(config["depth_single"]),
            int(config["dim"]))


def _spell(prefix: str, name: str) -> str:
    return prefix + (name.replace(".", "_") if prefix.endswith("_")
                     else name)


def lora_map(config: dict) -> LoraKeyMap:
    depth, single, dim = _shape(config)
    targets: dict = {}
    diffusers_only = set()
    comfy: set = set()

    def add(native: str, diffusers: str | None, band=None,
            module: str | None = None, comfy_diffusers: bool = True) -> None:
        module = module or native
        if native:
            targets[native] = (module, band)
            comfy.update(_spell(p, native) for p in COMFY_GGUF)
        if diffusers:
            targets[diffusers] = (module, band)
            diffusers_only.add(diffusers)
            if comfy_diffusers:
                comfy.update(_spell(p, diffusers) for p in COMFY_DIFFUSERS)

    for i in range(depth):
        x, d = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for leaf, dname in _DOUBLE:
            add(f"{x}.{leaf}", f"{d}.{dname}")
        add("", f"{d}.attn.to_out", module=f"{x}.img_attn.proj",
            comfy_diffusers=False)
        for leaf, names in _QKV:
            add(f"{x}.{leaf}", None)
            for band, dname in enumerate(names):
                add("", f"{d}.{dname}", band=(0, band * dim, dim),
                    module=f"{x}.{leaf}")
    for i in range(single):
        x, d = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        for leaf, dname in _SINGLE:
            add(f"{x}.{leaf}", f"{d}.{dname}")
        targets[f"{x}.linear1_qkv"] = (f"{x}.linear1", (0, 0, 3 * dim))
        comfy.add(f"diffusion_model.{x}.linear1_qkv")
    for native, dname in _TOP:
        add(native, dname)
    for native, dname in _TOP_GGUF_ONLY:
        add(native, dname, comfy_diffusers=False)
    return LoraKeyMap(family="flux2_klein", title="FLUX.2 klein",
                      targets=targets,
                      diffusers_only=frozenset(diffusers_only),
                      swapped=frozenset({NORM_OUT}),
                      comfy_stems=frozenset(comfy), comfy_reads=COMFY_READS)
