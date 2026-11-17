"""MiniMax-H3 LoRA key map, of the video DiT and of MiniMax-H3-Image's.

The GGUF keeps ComfyUI's module names, which ComfyUI's generic LoRA files
target, such as lightx2v's Turbo LoRAs. Every linear leaf of the DiT maps
to itself: in each block and each token refiner block ``attn.qkv_proj``,
``attn.out_proj``, ``mlp.fc1`` and ``mlp.fc2``, in each block
``adaln_proj.linear``, and outside the blocks ``condition_proj``, the two
patch projections, the two output heads and the final adaLN linear.
FastH3 (config ``vsa_sparsity``) adds ``attn.to_gate_compress`` in each
block, its VSA gate, and only FastH3's files have that module.
MiniMax-H3-Image has no audio stream, and its ``video_out`` is a bank of
one head per PDD step: as ComfyUI does for a baked bank (comfy/lora.py
426-428), a pair on the head patches head 0, the rows every step reads. A
pair on the fused qkv covers the whole weight, as those files write it:
one A over the three ranks, and a B that is zero outside each
projection's rows. ComfyUI reads these names bare, under
``diffusion_model.`` and as ``lora_unet_`` underscores, and drops every
other spelling.

The published H3-Image LoRAs spell diffusers' module names, bare or under
``transformer.``, as diffusers 0.43.0's converter writes them
(``_convert_non_diffusers_minimax_h3_lora_to_diffusers``):
``transformer_blocks.N.`` and ``token_refiner.refiner_blocks.N.`` for the
blocks; ``attn.to_q``, ``to_k`` and ``to_v`` for the thirds of the fused
qkv's rows, which map to those bands of ``attn.qkv_proj``;
``attn.to_out.0``, ``ff.net.0.proj``, ``ff.net.2`` and
``adaln_proj.linear`` for the other leaves; ``context_embedder``,
``proj_in``, ``audio_proj_in``, ``norm_out.linear``, ``proj_out`` and
``audio_proj_out`` outside the blocks. diffusers' SwiGLU holds fc1's
value rows before its gate rows, the GGUF the gate rows first, so the
halves of the up rows of a pair on ``ff.net.0.proj`` are swapped back.
gmlx applies these names, which stock ComfyUI drops. On MiniMax-H3-Image
the gmlx ComfyUI nodes read them bare or under ``transformer.``, so the
reader warns of them only under another prefix; on H3 video, where the
nodes keep ComfyUI's map, it warns of them all.

``refusal`` names the files gmlx does not run:

- a LoRA in DiffSynth's form (keys with ``.default.``), whose qkv order
  and fc1 halves gmlx has not read
- one with ``.config`` keys, which ComfyUI reads as a module's config,
  such as int8 or sparse attention in a block
- one whose head pairs have a multiple of a head's rows, as a PDD LoRA
  grows each head into a bank of heads
- on a file without FastH3's gate, one with a pair on it
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
# The modules of the audio stream, which a config without one lacks.
_AUDIO = ("audio_patch_proj", _HEADS[1])
# diffusers' name of each module outside the blocks, of each block group
# and of each block leaf but qkv, which splits into the three below.
_DIFF_TOP = {"condition_proj": "context_embedder",
             "video_patch_proj": "proj_in",
             "audio_patch_proj": "audio_proj_in",
             "final_layer.adaln_proj.linear": "norm_out.linear",
             _HEADS[0]: "proj_out", _HEADS[1]: "audio_proj_out"}
_DIFF_GROUPS = {"blocks": "transformer_blocks",
                "token_refiner.blocks": "token_refiner.refiner_blocks"}
_DIFF_LEAVES = {"attn.out_proj": "attn.to_out.0", "mlp.fc1": "ff.net.0.proj",
                "mlp.fc2": "ff.net.2",
                "adaln_proj.linear": "adaln_proj.linear"}
_DIFF_QKV = ("attn.to_q", "attn.to_k", "attn.to_v")
_COMFY = ("", "diffusion_model.", "lora_unet_")
COMFY_READS = ("the GGUF names bare, under diffusion_model. or as "
               "lora_unet_ underscores")
# The prefixes under which the gmlx ComfyUI nodes also read diffusers'
# names on a MiniMax-H3-Image model (comfyui_pack/patches.py,
# h3_diffusers_keys), so ComfyUI with the nodes drops none of those.
_PACK_DIFFUSERS = ("", "transformer.")
PACK_READS = (COMFY_READS + ", and ComfyUI with the gmlx nodes also "
              "reads diffusers' names under transformer. or bare")


def _blocks(config: dict) -> dict:
    """The fields the blocks read: a built H3-Image config holds them under
    ``block``, the other forms at the top."""
    return config.get("block", config)


def _counts(config: dict) -> tuple[int, int]:
    """Blocks and refiner blocks, from the config synthesized from a
    header or the fields of a built model's config, of either family."""
    c = _blocks(config)
    if "depth" in c:
        return int(c["depth"]), int(c["refiner_depth"])
    refiner = c.get("num_refiner_layers", c.get("token_refiner_num_layers"))
    return int(c["num_layers"]), int(refiner)


def _bank(config: dict) -> int:
    """The heads of ``video_out``: one per PDD step of H3-Image, else
    one."""
    return int(config.get("step_heads", config.get("pdd_steps", 1)))


def _audio(config: dict) -> int | None:
    """The audio head's rows, from either form of config, or None for a
    config without an audio stream."""
    audio = config.get("audio", config.get("audio_in_channels"))
    return None if audio is None else int(audio)


def _head_rows(config: dict) -> dict:
    """The output rows of one head of each output the config has, from
    any form of config."""
    c = _blocks(config)
    if "latent" in c:
        latent, patch = c["latent"], c["patch"]
    else:
        latent = c["in_channels"] if "in_channels" in c else c["latents_dim"]
        patch = c["patch_size"]
    rows = int(latent)
    for p in patch:
        rows *= int(p)
    audio = _audio(config)
    return {_HEADS[0]: rows} if audio is None else \
        {_HEADS[0]: rows, _HEADS[1]: audio}


def _spell(prefix: str, name: str) -> str:
    return prefix + (name.replace(".", "_") if prefix.endswith("_")
                     else name)


def _inner(config: dict) -> int:
    """The rows of each of q, k and v in the fused qkv."""
    c = _blocks(config)
    if "heads" in c:
        return int(c["heads"]) * int(c["head_dim"])
    return int(c["num_attention_heads"]) * int(c["attention_head_dim"])


def _diffusers(targets: dict, inner: int) -> tuple[dict, frozenset]:
    """diffusers' name of each target to its module and band, and the
    names whose up rows hold fc1's halves in the other order."""
    out, swapped = {}, set()
    for name, target in targets.items():
        if name in _DIFF_TOP:
            out[_DIFF_TOP[name]] = target
            continue
        for src, dst in _DIFF_GROUPS.items():
            if not name.startswith(src + "."):
                continue
            idx, _, leaf = name[len(src) + 1:].partition(".")
            base = f"{dst}.{idx}."
            if leaf == "attn.qkv_proj":
                for i, q in enumerate(_DIFF_QKV):
                    out[base + q] = (name, (0, i * inner, inner))
            elif leaf in _DIFF_LEAVES:
                out[base + _DIFF_LEAVES[leaf]] = target
                if leaf == "mlp.fc1":
                    swapped.add(base + _DIFF_LEAVES[leaf])
    return out, frozenset(swapped)


def _gates(config: dict) -> tuple:
    """The VSA gate of each block, as FastH3's config has them."""
    blocks, _ = _counts(config)
    return tuple(f"blocks.{i}.{_GATE}" for i in range(blocks))


def lora_map(config: dict, *, family: str = "h3",
             title: str = "MiniMax-H3",
             pack_reads_diffusers: bool = False) -> LoraKeyMap:
    """The H3 key map of ``config``. ``pack_reads_diffusers`` counts
    diffusers' names among the stems ComfyUI reads, as the gmlx nodes
    read them on MiniMax-H3-Image, so no warning names them."""
    blocks, refiner = _counts(config)
    names = [n for n in _TOP if n not in _AUDIO
             or _audio(config) is not None]
    block = _LEAVES + _BLOCK_ONLY
    if config.get("vsa_sparsity"):
        block += (_GATE,)
    for group, count, leaves in (("blocks", blocks, block),
                                 ("token_refiner.blocks", refiner, _LEAVES)):
        names += [f"{group}.{i}.{leaf}" for i in range(count)
                  for leaf in leaves]
    targets = {n: (n, None) for n in names}
    if _bank(config) > 1:
        targets[_HEADS[0]] = (_HEADS[0], (0, 0, _head_rows(config)[_HEADS[0]]))
    diffusers, swapped = _diffusers(targets, _inner(config))
    comfy = {_spell(p, n) for n in names for p in _COMFY}
    if pack_reads_diffusers:
        comfy |= {p + n for n in diffusers for p in _PACK_DIFFUSERS}
    return LoraKeyMap(
        family=family, title=title, targets={**targets, **diffusers},
        swapped=swapped, comfy_stems=frozenset(comfy),
        comfy_reads=PACK_READS if pack_reads_diffusers else COMFY_READS)


def _head_ups(header: dict, names):
    """(head, key, (rows, rank)) of each up matrix on the output heads
    ``names``, in every spelling the LoRA reader resolves."""
    heads = {h: (h, None) for h in names}
    heads.update({_DIFF_TOP[h]: (h, None) for h in names})
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
            yield heads[hit[0]][0], k, tuple(shape[:2])


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
            if split is not None and split[1] != "alpha" \
                    and _dit_target(split[0], gates, under):
                return (f"its pair {k} targets FastH3's attention gate, "
                        "which the model's file does not have",
                        "FastH3 gate")
    rows, bank = _head_rows(config), _bank(config)
    for head, k, shape in _head_ups(header, rows):
        n = rows[head]
        if shape[0] > n and shape[0] % n == 0:
            runs = "gmlx runs one plain head per stream" if bank == 1 else \
                (f"gmlx patches head 0 of the model's bank of {bank} heads "
                 "with a pair on the head, as ComfyUI does")
            return (f"its pair {k} has {shape[0]} rows, {shape[0] // n} "
                    f"heads of {n}, as a PDD LoRA grows each head into a "
                    f"bank of heads, and {runs}", "PDD")
    return None
