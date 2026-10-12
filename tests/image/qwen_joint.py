"""One full forward of the Qwen-Image 2.1 joint sequence under an explicit
block-causal mask, written apart from the model's prefill and cached
forward so that a test can hold the two against it.

It uses the model's leaf modules, its modulation vectors and the family's
rotary table function, and none of its token layout, prefill, cache or
join code. The joint sequence, the image blocks, the positions and the
mask are built here from the conditioning as the reference transformer
builds them for its uncached mode: the
encoder rows and the target's slots, each image slot expanded to four
tokens and overwritten by latents; blocks counted from the image shapes;
one position walk over the whole sequence; and a dense mask that lets a
token see every earlier token and every token of its own image block.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from gmlx.image import ops
from gmlx.image.families.qwen_image21 import layout


def sequence(n_rows: int, image_slots, ref_shapes, target) -> dict:
    """The joint sequence of ``n_rows`` encoder rows and a target of
    ``target`` (height, width) tokens: for each token whether it is an
    image token, the encoder row it expands, its image block or -1, and
    whether it is a target token."""
    th, tw = target
    slot = np.zeros(n_rows + th * tw // 4, bool)
    slot[list(image_slots)] = True
    slot[n_rows:] = True
    repeats = np.where(slot, 4, 1)
    image = np.repeat(slot, repeats)
    row = np.repeat(np.arange(slot.size), repeats)
    shapes = [*ref_shapes, (th, tw)]
    lengths = [h * w for h, w in shapes]
    where = np.flatnonzero(image)
    assert sum(lengths) == where.size
    block = np.full(image.size, -1)
    block[where] = np.repeat(np.arange(len(lengths)), lengths)
    is_target = np.zeros(image.size, bool)
    is_target[where[-lengths[-1]:]] = True
    return {"image": image, "row": row, "block": block,
            "target": is_target, "shapes": shapes}


def positions(image: np.ndarray, shapes) -> np.ndarray:
    """Positions (S, 3) from one walk over the joint sequence: a text
    token advances the shared position by one, and an image block holds
    the frame axis, centers its grid on zero and then advances by the
    larger of its height and width."""
    flags = image.tolist()
    frame, hh, ww = [], [], []
    cursor = position = 0
    for h, w in shapes:
        start = flags.index(True, cursor)
        frame.extend(range(position, position + start - cursor))
        position += start - cursor
        cursor = start + h * w
        frame.extend([position] * (h * w))
        position += max(h, w)
        hh.extend(r for r in range(-(h - h // 2), h // 2) for _ in range(w))
        ww.extend(c for _ in range(h) for c in range(-(w - w // 2), w // 2))
    frame.extend(range(position, position + len(flags) - cursor))
    pos = np.repeat(np.asarray(frame, np.int64)[:, None], 3, axis=1)
    pos[image, 1] = hh
    pos[image, 2] = ww
    return pos


def block_causal_mask(block: np.ndarray) -> np.ndarray:
    """(S, S) bool, true where the row's token sees the column's."""
    i = np.arange(block.size)
    same = (block[:, None] == block[None, :]) & (block[:, None] >= 0)
    return (i[:, None] >= i[None, :]) | same


def joint_forward(model, cond, latents: mx.array, sigma, *,
                  leak: tuple | None = None, sampled_rows=(),
                  taps: dict | None = None) -> mx.array:
    """Velocity (1, C, H, W) of ``latents`` (1, C, H, W) at ``sigma`` from
    one forward of the whole joint sequence.

    ``leak`` is (row, columns) of more true entries in the mask, and
    ``sampled_rows`` are tokens that read the sampled timestep's
    modulation where they would read that of t = 0. Both make a wrong
    model. ``taps`` collects the sequence after every block."""
    cfg = model.cfg
    act = model.act_dtype
    _, _, th, tw = latents.shape
    n_rows = int(cond.features.shape[0])
    refs = [tuple(int(d) for d in r.shape[-2:]) for r in cond.ref_latents]
    seq = sequence(n_rows, cond.image_slots, refs, (th, tw))
    image, target = seq["image"], seq["target"]

    def pack(z):
        c = z.shape[0]
        return z.reshape(c, -1).transpose(1, 0)

    txt = model.txt_in(cond.features.astype(act))
    txt = mx.concatenate(
        [txt, mx.zeros((th * tw // 4, txt.shape[1]), txt.dtype)], axis=0)
    img = mx.concatenate(
        [model.img_in(pack(z).astype(act))
         for z in (*cond.ref_latents, latents[0])], axis=0)
    nth = np.where(image, np.cumsum(image) - 1, 0)
    x = mx.where(mx.array(image)[:, None], img[mx.array(nth)],
                 txt[mx.array(seq["row"])])[None]

    cos, sin = layout.rope_tables(positions(image, seq["shapes"]),
                                  cfg.axes_dims, cfg.rope_theta)
    allowed = block_causal_mask(seq["block"])
    if leak is not None:
        allowed[leak] = True
    mask = mx.array(allowed)

    sampled = target.copy()
    sampled[list(sampled_rows)] = True
    pick = mx.array(sampled)[None, :, None]
    at_t = model.modulation_at(model.timesteps(sigma, 1))
    at_0 = model.modulation_at(np.zeros(1, np.float32))
    scale1, gate1, scale2, gate2, final = (
        mx.where(pick, a, b) for a, b in zip(at_t.arrays(), at_0.arrays()))

    for i, blk in enumerate(model.transformer_blocks):
        h = ops.layer_norm(x, cfg.eps) * (1 + scale1)
        q, k, v = blk.attn.qkv(h, cos, sin)
        x = x + gate1 * blk.attn.out(ops.sdpa(q, k, v, mask))
        h = ops.layer_norm(x, cfg.eps) * (1 + scale2)
        x = x + gate2 * blk.img_mlp(h)
        if taps is not None:
            taps[f"transformer_blocks.{i}"] = x
    out = model.proj_out(ops.layer_norm(x, cfg.eps) * (1 + final))
    out = out[:, mx.array(np.flatnonzero(target))]
    return out.transpose(0, 2, 1).reshape(1, -1, th, tw).astype(mx.float32)
