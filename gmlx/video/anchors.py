"""Anchors: clean latents a clip is conditioned on, at a frame index.

A keyframe or a guide is an anchor. Its rows enter the sequence as extra
cond and cond_audio rows on the target's grid (``layout.pack``), not in
place of target rows. A family's rule mixes a little noise into the
rows, ``aug * z + (1 - aug) * noise`` from a generator every anchor
restarts at the same seed, with no noise at ``aug`` 1, and runs them at
a time of at least ``aug``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass(frozen=True)
class Anchor:
    """A frame index of the target clip, negative from the end, with a
    video latent (1, C, T, H, W), an audio latent (1, C, 2, T), or both."""
    frame: int
    video: mx.array | None = None
    audio: mx.array | None = None

    def __post_init__(self):
        if self.video is None and self.audio is None:
            raise ValueError("an anchor needs a video or an audio latent")


@dataclass(frozen=True)
class AnchorRule:
    """A family's rule for the anchor rows of one stream: the weight of the
    clean rows, which is also the time they run at least at, and the
    offset added to the request seed."""
    aug: float
    seed_offset: int


def resolve_frame(frame: int, frames: int) -> int:
    """A frame index counted from the start, for a clip of ``frames``."""
    f = frame + frames if frame < 0 else frame
    if not 0 <= f < frames:
        raise ValueError(f"frame {frame} is outside a clip of {frames} "
                         f"frames")
    return f


def augment(rows: list, rule: AnchorRule, seed: int) -> list:
    """Each anchor's rows (n, features) as the rule mixes them, each from
    the same restarted generator, in float32."""
    import mlx.core as mx

    out = []
    for r in rows:
        r = r.astype(mx.float32)
        if rule.aug < 1.0:
            key = mx.random.key(seed + rule.seed_offset)
            noise = mx.random.normal(r.shape, mx.float32, key=key)
            r = rule.aug * r + (1.0 - rule.aug) * noise
        out.append(r)
    return out


def pinned(t: float, rule: AnchorRule) -> float:
    """The time the stream's anchor rows run at when its target is at t."""
    return max(t, rule.aug)
