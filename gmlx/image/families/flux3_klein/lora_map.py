"""FLUX.3 klein LoRA key map.

The GGUF keeps the names of BFL's transformer, and an adapter trained on
that transformer, or read by ComfyUI, names the same layers. In each
block of the two content modes, ``content_mode_blocks.image`` and
``content_mode_blocks.image_cond``, of the text mode,
``txt_mode_blocks``, and of the single stream, ``single_blocks``, those
are ``q_proj``, ``k_proj``, ``v_proj``, ``attn_out``, ``mlp_in`` and
``mlp_out``. Outside the blocks they are the embedders ``emb_in.image``,
``emb_in.image_cond`` and ``txt_in``, the time embedder ``time_in`` and,
in the distilled DiT, the target time embedder ``ref_time_in``, the
``lin`` of each mode in ``early_stream_modulations`` and
``single_stream_modulations``, and ``final_layer.image.linear`` and
``final_layer.image.adaLN_modulation.1``. Each name maps to itself, and
every projection is its own weight in the file.

An adapter in the diffusers layout names the same layers as
Flux3Transformer2DModel does: ``mode_blocks.image``,
``mode_blocks.image_cond``, ``txt_mode_blocks`` and
``single_transformer_blocks``, each block with ``attn.to_q``,
``attn.to_k``, ``attn.to_v``, ``attn.to_out.0``, ``ff.linear_in`` and
``ff.linear_out``, and outside the blocks ``x_embedders``,
``context_embedder``, ``timestep_embedder``,
``target_timestep_embedder``, the ``linear`` of each modulation,
``norm_out.linear`` and ``proj_out``. Each maps to the module of the
same weight. Some tools write ``to_out`` without the ``.0``, and that
name maps to ``attn_out`` too. diffusers' converter swaps the shift and
the scale halves of the final modulation, so ``norm_out.linear`` holds
the scale rows first, and the halves of the up rows of a pair on it are
swapped back to the order of ``final_layer.image.adaLN_modulation.1``.
The diffusers loader renames the block layers of a file in the GGUF
names and keeps the other names, which its model does not have, so a
pair outside the blocks in the GGUF names gives another image there than
in gmlx and in ComfyUI. It also refuses a file with ``alpha`` tensors,
whose keys do not hold ``lora``, where gmlx scales the pair by alpha over
rank as ComfyUI does.

``mlp_in`` is one weight whose output rows are the gate rows, then the
up rows, each of the MLP width. A pair on ``mlp_in`` is one delta over
both halves: the block splits the sum of the base and the delta, and the
gate half takes the SiLU.

ComfyUI also loads a file that fuses the q, k and v projections of a
block into one ``qkv`` weight. A ``qkv`` key is refused by the name of
that layout, and a key on ``ref_time_in`` of a Base file by the name of
the distilled DiT. No text encoder key resolves: the map has no encoder
targets.
"""

from __future__ import annotations

from gmlx.image.lora.reader import LoraKeyMap

LEAVES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
MODES = ("image", "image_cond")
MOD_KEYS = ("image", "image_cond", "txt")
# The diffusers name of each leaf of a block.
_LEAF = {"q_proj": "attn.to_q", "k_proj": "attn.to_k",
         "v_proj": "attn.to_v", "attn_out": "attn.to_out.0",
         "mlp_in": "ff.linear_in", "mlp_out": "ff.linear_out"}
# diffusers stores the final modulation with its scale rows before its
# shift rows, the other order of the GGUF's.
NORM_OUT = "norm_out.linear"
# The layers outside the blocks and their diffusers names.
_TOP = {"emb_in.image": "x_embedders.image",
        "emb_in.image_cond": "x_embedders.image_cond",
        "txt_in": "context_embedder",
        "time_in.in_layer": "timestep_embedder.linear_1",
        "time_in.out_layer": "timestep_embedder.linear_2",
        "final_layer.image.linear": "proj_out",
        "final_layer.image.adaLN_modulation.1": NORM_OUT,
        **{f"{s}_stream_modulations.{k}.lin":
           f"{s}_stream_modulations.{k}.linear"
           for s in ("early", "single") for k in MOD_KEYS}}
# The prefixes under which ComfyUI's key map finds a FLUX.3 adapter, in
# the GGUF names only.
COMFY_PREFIXES = ("lora_unet_", "diffusion_model.")
REF_TIME = {"ref_time_in.in_layer": "target_timestep_embedder.linear_1",
            "ref_time_in.out_layer": "target_timestep_embedder.linear_2"}
_BASE = ("the target time embedder of the distilled FLUX.3 klein DiT, and "
         "this DiT file is the Base model, which takes no target time")
_FUSED = ("the fused q, k and v weight of ComfyUI's FLUX.3 layout, and this "
          "DiT file stores q_proj, k_proj and v_proj apart")


def _shape(config: dict) -> tuple[int, int, bool]:
    """Blocks per mode, single blocks and whether the DiT is the
    distilled flow map. ``config`` is the config synthesized from a
    header, which a file is checked with before a model loads, or the
    fields of a built model's config."""
    if "num_layers" in config:
        return (int(config["num_layers"]), int(config["num_single_layers"]),
                bool(config["flow_map"]))
    return (int(config["depth"]), int(config["depth_single"]),
            bool(config["flow_map"]))


def block_groups(config: dict) -> tuple:
    """(module path prefix, diffusers prefix, block count) of every group
    of blocks."""
    depth, single, _ = _shape(config)
    return (*((f"content_mode_blocks.{m}", f"mode_blocks.{m}", depth)
              for m in MODES),
            ("txt_mode_blocks", "txt_mode_blocks", depth),
            ("single_blocks", "single_transformer_blocks", single))


def lora_map(config: dict) -> LoraKeyMap:
    _, _, flow_map = _shape(config)
    targets: dict = {}
    diffusers_only = set()
    absent: dict = {}

    def add(native: str, diffusers: str) -> None:
        targets[native] = (native, None)
        targets[diffusers] = (native, None)
        if diffusers != native:
            diffusers_only.add(diffusers)

    for group, dgroup, count in block_groups(config):
        for i in range(count):
            x, d = f"{group}.{i}", f"{dgroup}.{i}"
            for leaf, dname in _LEAF.items():
                add(f"{x}.{leaf}", f"{d}.{dname}")
            add(f"{x}.attn_out", f"{d}.attn.to_out")
            absent[f"{x}.qkv"] = _FUSED
    top = dict(_TOP)
    if flow_map:
        top.update(REF_TIME)
    else:
        for native, dname in REF_TIME.items():
            absent[native] = absent[dname] = _BASE
    for native, dname in top.items():
        add(native, dname)
    return LoraKeyMap(family="flux3_klein", title="FLUX.3 klein",
                      targets=targets,
                      diffusers_only=frozenset(diffusers_only),
                      absent=absent, swapped=frozenset({NORM_OUT}),
                      comfy_prefixes=COMFY_PREFIXES)
