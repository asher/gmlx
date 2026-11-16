"""The rule for the clean rows an H3 DiT is conditioned on, shared by
video's anchors and the image's references.

A family's rule mixes a little noise into the rows, ``aug * z + (1 -
aug) * noise`` from a generator every anchor restarts at the same seed,
with no noise at ``aug`` 1, and runs them at a time of at least ``aug``.
Importing this module loads no mlx.
"""

from __future__ import annotations

from dataclasses import dataclass

# Seeds and their offsets wrap to 64 bits, the range of a ComfyUI seed.
SEED_RANGE = 1 << 64


@dataclass(frozen=True)
class AnchorRule:
    """A family's rule for the anchor rows of one stream: the weight of the
    clean rows, which is also the time they run at least at, and the
    offset added to the request seed."""
    aug: float
    seed_offset: int


def augment(rows: list, rule: AnchorRule, seed: int, noise=None) -> list:
    """Each anchor's rows (n, features) as the rule mixes them, in
    float32. Each anchor's noise comes from the same generator, restarted
    at the seed plus the rule's offset, wrapped to 64 bits; or it is
    ``noise``'s array for that anchor, in its rows' shape, which is how
    the pack sends the draws of ComfyUI's torch generator."""
    import mlx.core as mx

    if noise is not None and len(noise) != len(rows):
        raise ValueError(f"{len(noise)} noise arrays for {len(rows)} "
                         "anchors")
    out = []
    for i, r in enumerate(rows):
        r = r.astype(mx.float32)
        if rule.aug < 1.0:
            if noise is None:
                key = mx.random.key((seed + rule.seed_offset) % SEED_RANGE)
                n = mx.random.normal(r.shape, mx.float32, key=key)
            else:
                n = noise[i].astype(mx.float32)
                if tuple(n.shape) != tuple(r.shape):
                    raise ValueError(f"anchor noise {tuple(n.shape)} is not "
                                     f"the shape of its rows "
                                     f"{tuple(r.shape)}")
            r = rule.aug * r + (1.0 - rule.aug) * n
        out.append(r)
    return out


def pinned(t: float, rule: AnchorRule) -> float:
    """The time the stream's anchor rows run at when its target is at t."""
    return max(t, rule.aug)
