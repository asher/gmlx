# SPDX-License-Identifier: Apache-2.0
"""Patch tables from the weight patches stock ComfyUI nodes add.

``LoraLoader`` and ``LoraLoaderModelOnly`` call ``add_patches`` on a clone of
the gmlx MODEL, which keeps ComfyUI's tuples in ``patches``:
``(strength_patch, adapter, strength_model, offset, function)`` per weight
key. A table entry carries the key, the offset as a band, and the adapter
pairs by hash with their strengths. Plain LoRA pairs are applied; every
other kind is refused by name. ComfyUI's key map swaps the two halves of
the rows of a diffusers ``norm_out.linear`` pair's product, through
``swap_scale_shift``; such a pair goes with the halves of its up rows
swapped, which gives the same product, on a family in ``SWAPS_HALVES``
or a key of ``SWAP_KEYS``, and is refused by name on any other.

Stock ComfyUI's key map drops a MiniMax-H3-Image LoRA in diffusers'
module names. On a gmlx MiniMax-H3-Image model the pack's key map also
reads them, as diffusers 0.43.0's MiniMax-H3 LoRA converter defines them
and as ComfyUI reads diffusers' names for Flux: q, k and v as thirds of
``qkv_proj``'s rows, and ``ff.net.0.proj`` as ``fc1`` through
``swap_scale_shift``, since diffusers holds fc1's two halves in the other
order (``h3_diffusers_keys``).
"""

from __future__ import annotations

import re

import torch

from .client import GmlxRefusal, StepClient

# Adapter dtypes the contract carries; anything else goes as f32.
_CARRIED = (torch.float32, torch.float16, torch.bfloat16)
# The families whose final modulation the server holds in the order that a
# swap of the up rows' halves gives, checked against ComfyUI's own model
# by the pack's FLUX.2 klein LoRA test. The 9B and KV checkpoints share the
# 4B LoRA map.
SWAPS_HALVES = frozenset({"flux2_klein", "flux2_klein_9b", "flux2_klein_9b_kv"})
# Per family, the keys a swap of the up rows' halves gives the product of:
# MiniMax-H3-Image's fc1, under diffusers' name ff.net.0.proj.
SWAP_KEYS = {"h3_image": re.compile(
    r"diffusion_model\.(token_refiner\.)?blocks\.\d+\.mlp\.fc1\.weight")}
# The model attribute that turns on the diffusers names of h3_diffusers_keys.
H3_NAMES_FLAG = "gmlx_h3_diffusers_lora"
# MiniMax-H3-Image's diffusers names: of each module outside the blocks,
# of each block group, and of each block leaf other than qkv and fc1.
_H3_TOP = {"condition_proj": "context_embedder", "video_patch_proj": "proj_in",
           "final_layer.adaln_proj.linear": "norm_out.linear",
           "final_layer.video_out": "proj_out"}
_H3_GROUPS = (("blocks.", "transformer_blocks."),
              ("token_refiner.blocks.", "token_refiner.refiner_blocks."))
_H3_LEAVES = {"attn.out_proj": "attn.to_out.0", "mlp.fc2": "ff.net.2",
              "adaln_proj.linear": "adaln_proj.linear"}
_H3_QKV = ("attn.to_q", "attn.to_k", "attn.to_v")


def adapter_kind(adapter) -> str:
    """``"lora"`` for a plain LoRA pair, else the name of what it is."""
    if isinstance(adapter, list):
        return "a merge list (from a model merge node)"
    if isinstance(adapter, tuple):
        return f"a {adapter[0]!r} patch" if adapter else "an empty patch"
    name = getattr(adapter, "name", None)
    if name != "lora":
        return {"loha": "LoHa", "lokr": "LoKr", "glora": "GLoRA", "oft": "OFT",
                "boft": "BOFT"}.get(name, f"a {name or type(adapter).__name__} adapter")
    w = tuple(getattr(adapter, "weights", ()))
    if len(w) < 3:
        return "a LoRA without up and down weights"
    mid = w[3] if len(w) > 3 else None
    dora = w[4] if len(w) > 4 else None
    reshape = w[5] if len(w) > 5 else None
    if mid is not None:
        return "a LoRA with a mid tensor"
    if dora is not None:
        return "DoRA"
    if reshape is not None:
        return "a LoRA with a reshape"
    return "lora"


def _carried(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in _CARRIED else t.float()


def _h3_names(name: str) -> list:
    """(diffusers name, part) of the module ``name``: part None for the
    whole weight, 0 to 2 for a third of the fused qkv rows, "swap" for
    fc1's swapped halves."""
    if name in _H3_TOP:
        return [(_H3_TOP[name], None)]
    for src, dst in _H3_GROUPS:
        if not name.startswith(src):
            continue
        idx, _, leaf = name[len(src):].partition(".")
        base = f"{dst}{idx}."
        if leaf == "attn.qkv_proj":
            return [(base + q, i) for i, q in enumerate(_H3_QKV)]
        if leaf == "mlp.fc1":
            return [(base + "ff.net.0.proj", "swap")]
        if leaf in _H3_LEAVES:
            return [(base + _H3_LEAVES[leaf], None)]
    return []


def h3_diffusers_keys(model, key_map: dict) -> dict:
    """``key_map`` with MiniMax-H3-Image's diffusers names added, bare and
    under ``transformer.``, for every weight of ``model``. A name ComfyUI
    places already keeps its place."""
    import comfy.utils

    sd = model.state_dict()
    for k in sd:
        if not (k.startswith("diffusion_model.") and k.endswith(".weight")):
            continue
        name = k[len("diffusion_model."):-len(".weight")]
        for dn, part in _h3_names(name):
            if part is None:
                to = key_map.get(name, k)
            elif part == "swap":
                to = (k, None, comfy.utils.swap_scale_shift)
            else:
                n = int(sd[k].shape[0]) // 3
                to = (k, (0, part * n, n))
            for prefix in ("", "transformer."):
                key_map.setdefault(prefix + dn, to)
    return key_map


def install_key_map() -> None:
    """Wrap ComfyUI's ``model_lora_keys_unet`` once, so a model that sets
    ``H3_NAMES_FLAG`` gets MiniMax-H3-Image's diffusers names beside
    ComfyUI's own. Every other model gets ComfyUI's map unchanged. The
    flag is looked for down the ``__wrapped__`` chain, so a wrap another
    node puts on top does not hide this one and add a second."""
    import functools
    import inspect

    import comfy.lora

    def _flagged(f):
        return getattr(f, "gmlx_wrapped", False)

    real = comfy.lora.model_lora_keys_unet
    if _flagged(inspect.unwrap(real, stop=_flagged)):
        return

    @functools.wraps(real)
    def keys(model, key_map={}):  # noqa: B006 - ComfyUI's own default
        key_map = real(model, key_map)
        if getattr(model, H3_NAMES_FLAG, False):
            h3_diffusers_keys(model, key_map)
        return key_map

    keys.gmlx_wrapped = True
    comfy.lora.model_lora_keys_unet = keys


def _swaps_halves(function) -> bool:
    """Whether ``function`` is ComfyUI's ``swap_scale_shift``."""
    import comfy.utils

    return function is not None and function is getattr(comfy.utils, "swap_scale_shift", None)


def _swap_halves(t: torch.Tensor) -> torch.Tensor:
    first, second = t.chunk(2, dim=0)
    return torch.cat([second, first], dim=0)


def _alpha(a):
    if a is None:
        return None
    if torch.is_tensor(a):
        a = a.item()
    return float(a)


def build_table(model_id: str, patches: dict, client: StepClient | None = None, *,
                family: str):
    """``(table, tensors)`` for ComfyUI ``patches`` on a model of ``family``:
    the table body, or None when there are none, and the adapter matrices by
    hash. With a client, pairs the server does not hold are uploaded. Raises
    :class:`GmlxRefusal` for a patch kind the step contract does not apply."""
    from .client import tensor_hash

    groups: dict = {}
    tensors: dict = {}
    for key in sorted(patches):
        for entry in patches[key]:
            strength, adapter, strength_model, offset, function = entry[:5]
            kind = adapter_kind(adapter)
            if kind != "lora":
                raise GmlxRefusal(f"gmlx: {kind} on {key} is not supported on a gmlx model; "
                                  "only plain LoRA pairs apply")
            swap = offset is None and _swaps_halves(function)
            if function is not None and not swap:
                raise GmlxRefusal(f"gmlx: a patch function on {key} is not supported on a "
                                  "gmlx model")
            keys = SWAP_KEYS.get(family)
            if swap and family not in SWAPS_HALVES and not (keys and keys.fullmatch(key)):
                raise GmlxRefusal(f"gmlx: ComfyUI swaps the halves of the product on {key} "
                                  "(swap_scale_shift), which a gmlx model takes on FLUX.2 "
                                  "klein and on MiniMax-H3-Image's fc1 alone, not here on "
                                  f"{family}")
            if float(strength_model) != 1.0:
                raise GmlxRefusal(f"gmlx: a model strength of {strength_model} on {key} "
                                  "scales the base weight, which a gmlx model does not do")
            if float(strength) == 0.0:
                continue
            up, down = _carried(adapter.weights[0]), _carried(adapter.weights[1])
            if swap:
                up = _swap_halves(up)
            if client is not None:
                hu, hd = client.ensure_adapter(up), client.ensure_adapter(down)
            else:
                hu, hd = tensor_hash(up), tensor_hash(down)
            tensors[hu], tensors[hd] = up, down
            band = None if offset is None else [int(v) for v in offset]
            groups.setdefault((key, tuple(band) if band else None), []).append(
                {"up": hu, "down": hd, "alpha": _alpha(adapter.weights[2]),
                 "strength": float(strength), "strength_model": float(strength_model)})
    if not groups:
        return None, tensors
    entries = [{"key": k, "band": None if b is None else list(b), "patches": p}
               for (k, b), p in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or ()))]
    return {"model": model_id, "entries": entries}, tensors


def sync_table(client: StepClient, model_id: str, patches: dict, family: str) -> str | None:
    """Upload what the server lacks and define the table. Returns its id,
    or None with no patches. A 409 for adapters the server dropped
    uploads them again and retries once."""
    from .client import GmlxMissing

    table, tensors = build_table(model_id, patches, client, family=family)
    if table is None:
        return None
    try:
        return client.define_table(model_id, table)
    except GmlxMissing as e:
        client.stats["missing_retries"] += 1
        for h in e.missing:
            if h in tensors:
                client.put_tensor(h, tensors[h], adapter=True)
        return client.define_table(model_id, table, force=True)
