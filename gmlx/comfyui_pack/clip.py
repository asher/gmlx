"""A ComfyUI CLIP whose text encoder runs on the gmlx server.

``tokenize`` keeps the text and the options the stock text encode nodes
pass, and the encode calls the server, which returns real conditioning
tensors, so stock conditioning nodes work on them. The stub text encoder
has no weights, so LoRA nodes find no text encoder keys on it.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import torch

import comfy.model_patcher
import comfy.sd

from .client import GmlxRefusal, StepClient
from .patcher import CPU

log = logging.getLogger("gmlx")

_WEIGHTED = re.compile(r"\([^()]*:\s*-?[0-9.]+\s*\)")

# The families whose stock ComfyUI tokenizer encodes an empty prompt as a
# space under prevent_empty_text: QwenImageTokenizer, and Qwen3VLTokenizer
# under Qwen-Image 2.1 and Krea 2. ZImageTokenizer ignores the option.
_SPACE_FOR_EMPTY = frozenset({"qwen_image", "qwen_image21", "krea2"})


class _StubEncoder(torch.nn.Module):
    """A text encoder with no weights; ComfyUI reads its options and state dict."""

    def __init__(self):
        super().__init__()
        self.dtypes = {torch.float32}

    def reset_clip_options(self):
        pass

    def set_clip_options(self, options):
        pass


def _to_uint8(image: torch.Tensor) -> list:
    """(B, H, W, C) floats in [0, 1] to a list of (H, W, C) uint8 arrays."""
    a = image.detach().float().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    if a.ndim == 3:
        a = a[None]
    return [np.ascontiguousarray(a[i, :, :, :3]) for i in range(a.shape[0])]


class GmlxCLIP(comfy.sd.CLIP):
    gmlx_client: StepClient | None = None
    gmlx_model: str | None = None
    gmlx_family: str | None = None

    def __init__(self, client: StepClient | None = None, model_id: str | None = None,
                 family: str | None = None):
        super().__init__(no_init=True)
        if client is None:
            return
        self.gmlx_client, self.gmlx_model, self.gmlx_family = client, model_id, family
        self.cond_stage_model = _StubEncoder()
        self.tokenizer = None
        self.patcher = comfy.model_patcher.ModelPatcher(self.cond_stage_model, load_device=CPU,
                                                        offload_device=CPU, size=1)
        self.patcher.is_clip = True
        self.layer_idx = None
        self.use_clip_schedule = False
        self.apply_hooks_to_conds = None
        self.tokenizer_options = {}
        self.warned = set()

    def clone(self, disable_dynamic=False):
        n = GmlxCLIP()
        n.gmlx_client, n.gmlx_model = self.gmlx_client, self.gmlx_model
        n.gmlx_family = self.gmlx_family
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = None
        n.layer_idx = self.layer_idx
        n.tokenizer_options = self.tokenizer_options.copy()
        n.use_clip_schedule = self.use_clip_schedule
        n.apply_hooks_to_conds = self.apply_hooks_to_conds
        n.warned = self.warned
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        if strength_patch and "lora" not in self.warned:
            self.warned.add("lora")
            log.warning("gmlx: LoRA keys for the text encoder are not applied on a gmlx CLIP; "
                        "set strength_clip to 0 to silence this")
        return []

    def clip_layer(self, layer_idx):
        raise GmlxRefusal("gmlx: a CLIP layer choice is not supported on a gmlx CLIP; the "
                          "server encodes with the model's own layer")

    def load_model(self, tokens={}):
        return None

    def tokenize(self, text, return_word_ids=False, **kwargs):
        if return_word_ids:
            raise GmlxRefusal("gmlx: word ids are not available from a gmlx CLIP")
        images = list(kwargs.get("images") or [])
        if kwargs.get("image") is not None and not images:
            img = kwargs["image"]
            images = [img[i:i + 1] for i in range(img.shape[0])]
        return {"gmlx": {"text": str(text), "images": images,
                         "llama_template": kwargs.get("llama_template"),
                         "keep_vision": bool(kwargs.get("keep_vision", False)),
                         "prevent_empty_text": bool(kwargs.get("prevent_empty_text", False))}}

    def _payload(self, tokens) -> dict:
        p = tokens.get("gmlx") if isinstance(tokens, dict) else None
        if p is None:
            raise GmlxRefusal("gmlx: these tokens did not come from a gmlx CLIP's tokenize")
        if p.get("llama_template") is not None:
            raise GmlxRefusal("gmlx: a custom prompt template is not supported on a gmlx CLIP; "
                              "the server applies the model's own template")
        if p["images"] and p.get("keep_vision"):
            raise GmlxRefusal("gmlx: reference images seen by the text encoder alone are not "
                              "supported on a gmlx CLIP; connect a VAE to the text encode node")
        if "weighted" not in self.warned and _WEIGHTED.search(p["text"]):
            self.warned.add("weighted")
            log.warning("gmlx: weighted prompt syntax is passed to the server as plain text")
        return p

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        p = self._payload(tokens)
        images = []
        for im in p["images"]:
            images += _to_uint8(im)
        text = p["text"]
        if p.get("prevent_empty_text") and text == "" and self.gmlx_family in _SPACE_FOR_EMPTY:
            text = " "
        out, fields = self.gmlx_client.encode(self.gmlx_model, text, images)
        cond = out["cond"].float()
        if cond.ndim == 2:
            cond = cond[None]
        if return_dict:
            d = {"cond": cond, "pooled_output": None}
            if images and fields.get("image_slots"):
                d["image_slots"] = [int(s) for s in fields["image_slots"]]
            self.add_hooks_to_dict(d)
            return d
        if return_pooled:
            return cond, None
        return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        if self.patcher.forced_hooks is not None and self.use_clip_schedule:
            raise GmlxRefusal("gmlx: scheduled CLIP hooks are not supported on a gmlx CLIP")
        d = self.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = d.pop("cond")
        d.update(add_dict)
        return [[cond, d]]

    def encode(self, text):
        return self.encode_from_tokens(self.tokenize(text))

    def get_sd(self):
        raise GmlxRefusal("gmlx: a gmlx CLIP has no weights in ComfyUI to save")

    def state_dict_for_saving(self):
        raise GmlxRefusal("gmlx: a gmlx CLIP has no weights in ComfyUI to save")

    def load_sd(self, sd, full_model=False):
        raise GmlxRefusal("gmlx: a gmlx CLIP takes no weights")
