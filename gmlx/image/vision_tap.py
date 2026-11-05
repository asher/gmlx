# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 Qwen-Image Team and The HuggingFace Team
# Ported from calculate_dimensions, the image preprocessing of __call__ and
# the image branch of _get_qwen_prompt_embeds in diffusers'
# src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py at e0abab83
# and modified for gmlx (Apache-2.0; see licenses/diffusers-LICENSE).
"""The tap of a Qwen3-VL text encoder over a prompt with images, which
the families that condition on images share.

The tap is the text tap with images: after the last decoder layer, before
the final norm, in float32, less the rows the spec drops. The vision
tower runs one image at a time in float32, its features replace the
embeddings of the image rows, and its deepstack features are added to
those rows after each of the first decoder layers, as the model's own
forward does.

The tower's matrix products must be exact float32. Its residual stream
reaches the thousands, and a product at TF32 precision, the default of an
M5-class GPU, moves the image rows of the features by 5 to 10 percent of
their norm and single rows to a cosine under 0.5. MLX reads its TF32
switch once per process, so the tower cannot turn it off for itself. On a
device that multiplies float32 at TF32 the tower's linear layers split
each product while the tower runs: each operand into a part TF32 holds
exactly and the rest, three products summed. That product is as exact as
a float32 one and leaves every other product of the process at TF32. The
convolution of the patch embedding computes float32 exactly in either
state, and the fused attention does at some head sizes only: where it
does not at the tower's, the tower runs on the CPU.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image.float32 import (exact_matmul, f32_products_exact,
                                tf32_exact)


class ExactLinear(nn.Module):
    """A float32 linear layer computed with ``exact_matmul``. It holds the
    arrays of the layer it stands for, and whether its weight is its own
    high part."""

    def __init__(self, linear: nn.Linear, weight_exact: bool = False):
        super().__init__()
        self.weight = linear.weight
        if "bias" in linear:
            self.bias = linear.bias
        self.weight_exact = weight_exact

    def __call__(self, x: mx.array) -> mx.array:
        y = exact_matmul(x, self.weight, b_t=True,
                         b_exact=self.weight_exact)
        return y + self.bias if "bias" in self else y


class EvaluatedBlock(nn.Module):
    """A block whose output is evaluated before the next block runs, so
    the temporaries of the split products of one block are released
    before those of the next are made."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def __call__(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        mx.eval(out)
        return out


_SDPA_EXACT: dict = {}              # by default device and head size
_SCOPE = threading.RLock()


def sdpa_exact(head_dim: int) -> bool:
    """Whether the fused attention of the default device computes float32
    at float32 precision at ``head_dim``, padded as the tower pads it.
    Each device and size is measured once against the CPU."""
    from gmlx._vendor.mlx_vlm.models.base import ensure_fused_sdpa

    key = (str(mx.default_device()), int(head_dim))
    hit = _SDPA_EXACT.get(key)
    if hit is None:
        q, k, v = (mx.random.normal((1, 2, 128, head_dim),
                                    key=mx.random.key(i)) for i in range(3))
        scale = head_dim ** -0.5
        got = ensure_fused_sdpa(q, k, v, scale)
        with mx.stream(mx.cpu):
            want = ensure_fused_sdpa(q, k, v, scale)
        gap = mx.linalg.norm(got - want) / mx.linalg.norm(want)
        hit = _SDPA_EXACT[key] = float(gap) < 1e-5
    return hit


def _head_dims(module) -> set:
    return {int(m.head_dim) for _, m in module.named_modules()
            if "qkv" in m and hasattr(m, "head_dim")}


@contextmanager
def float32_products(module):
    """Inside the block, ``module`` computes at float32 precision on a
    device that multiplies float32 at TF32: its float32 linear layers
    split each product and its blocks are evaluated one at a time, or,
    where the fused attention is not float32 at its head size, it runs on
    the CPU. On a device that multiplies
    float32 exactly, nothing changes. The layers are back as they were
    after the block."""
    if f32_products_exact():
        yield module
        return
    heads = _head_dims(module)
    if not heads or not all(sdpa_exact(d) for d in heads):
        with mx.stream(mx.cpu):
            yield module
        return
    from mlx.utils import tree_unflatten

    with _SCOPE:
        found: list = []

        def visit(path: str, m) -> None:
            if (isinstance(m, nn.Linear) and m.weight.dtype == mx.float32
                    and path):
                found.append((path, m))

        module.apply_to_modules(visit)
        # A weight read from bfloat16 is its own high part: its split
        # would only add a copy of it the size of the weight. Each check
        # runs alone, so one weight's temporary is live at a time.
        module.update_modules(tree_unflatten(
            [(p, ExactLinear(m, bool(tf32_exact(m.weight).item())))
             for p, m in found]))
        stacks = [(m, m.blocks) for _, m in module.named_modules()
                  if isinstance(m.get("blocks"), list)]
        for m, blocks in stacks:
            m.blocks = [EvaluatedBlock(b) for b in blocks]
        try:
            yield module
        finally:
            for m, blocks in stacks:
                m.blocks = blocks
            module.update_modules(tree_unflatten(found))


def tap(handle, ids, blocks, *, pad: int, deepstack: int | None = None):
    """(features, rows) of the token ids ``ids`` with images. ``blocks``
    yields (pixels, grid) per image in the order of its rows, each run
    through the tower alone. The rows of token ``pad`` take the tower's
    features, ascending, as ``rows``, indices into ``ids``. ``features``
    is (L - drop, D) in the spec's activation dtype. ``deepstack`` cuts
    the count of deepstack blocks added, for a test of a wrong encoder."""
    model = handle.model
    tower = model.vision_tower
    spec = handle.spec
    # The model places the image rows by its own image token.
    own = getattr(getattr(model, "config", None), "image_token_id", pad)
    if int(own) != pad:
        raise ValueError(f"the tokenizer's image pad token {pad} is not "
                         f"the encoder's image token {int(own)}")
    act = getattr(mx, spec.activation or "float32")
    feats, deep, grids = [], [], []
    with float32_products(tower):
        for pixels, grid in blocks:
            f, d = tower(pixels.astype(mx.float32), mx.array([grid]))
            mx.eval(f, *d)
            feats.append(f.astype(act))
            deep.append([x.astype(act) for x in d])
            grids.append(grid)
    rows = np.flatnonzero(np.asarray(ids) == pad)
    image = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)
    if image.shape[0] != rows.size:
        raise ValueError(f"the tower gave {image.shape[0]} image rows for "
                         f"{rows.size} image tokens")
    count = len(deep[0]) if deepstack is None else deepstack
    stack = [mx.concatenate([d[i] for d in deep], axis=0)
             for i in range(count)]
    lm = model.language_model
    x = mx.array(np.asarray(ids, np.int32).reshape(1, -1))
    h = lm.model.embed_tokens(x).astype(act)
    h[0, mx.array(rows)] = image
    position_ids, _ = lm.get_rope_index(x, mx.array(grids), None, None)
    (layer,) = spec.tap_layers
    out = lm.model(x, inputs_embeds=h, position_ids=position_ids,
                   visual_pos_masks=x == pad,
                   deepstack_visual_embeds=stack or None,
                   stop_after_layer=layer, apply_final_norm=False)
    features = out[0, spec.drop:]
    mx.eval(features)
    return features, rows
