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
alone, and is refused by name on any other.
"""

from __future__ import annotations

import torch

from .client import GmlxRefusal, StepClient

# Adapter dtypes the contract carries; anything else goes as f32.
_CARRIED = (torch.float32, torch.float16, torch.bfloat16)
# The families whose final modulation the server holds in the order that a
# swap of the up rows' halves gives, checked against ComfyUI's own model
# by the pack's FLUX.2 klein LoRA test. The 9B checkpoints share the 4B
# LoRA map.
SWAPS_HALVES = frozenset({"flux2_klein", "flux2_klein_9b"})


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
            if swap and family not in SWAPS_HALVES:
                raise GmlxRefusal(f"gmlx: ComfyUI swaps the halves of the product on {key} "
                                  "(swap_scale_shift), which a gmlx model takes on FLUX.2 "
                                  f"klein alone, not on {family}")
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
