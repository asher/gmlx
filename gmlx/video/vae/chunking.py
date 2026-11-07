"""Tile and chunk schedules of a video VAE, apart from any model's values.

A decoder trained on tiles of one size decodes a larger frame as
overlapping tiles of that size, blended linearly across each overlap. A
temporal decoder decodes overlapping runs of latents and crossfades the
frames two runs share. The family supplies the sizes; this module holds
the arithmetic. Importing it loads no mlx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def split_tiles(length: int, tile: int, min_overlap: int,
                unit: int) -> tuple[list[int], list[int], list[int]]:
    """Starts, lengths and overlaps of the tiles that cover ``length``
    pixels: the fewest tiles of ``tile`` pixels whose overlaps are all at
    least ``min_overlap``, with the room left over spread across the
    overlaps in steps of ``unit``, round robin from the first. A length
    one tile covers is one tile of that length."""
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) < length:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    for i in range((tile * n - sum(overlaps) - length) // unit):
        overlaps[i % (n - 1)] += unit
    starts = [0]
    for o in overlaps:
        starts.append(starts[-1] + tile - o)
    return starts, [tile] * n, overlaps


@dataclass(frozen=True)
class TimeChunks:
    """A causal video VAE's temporal pieces. The encoder takes pieces of
    ``clip`` frames and gives ``-(-clip // ratio)`` latents each; the
    concatenated latents lose their last ``drop``. The decoder runs over
    runs of ``tokens + overlap`` latents at a stride of ``tokens``, and
    each run's frames past its first ``tokens`` latents' are crossfaded
    into the next run's head."""
    clip: int           # frames of an encoder piece
    ratio: int          # frames per latent
    drop: int           # latents dropped after the pieces

    @property
    def tokens(self) -> int:
        return -(-self.clip // self.ratio)

    @property
    def overlap(self) -> int:
        return (-self.drop) % self.tokens

    @property
    def pre_frames(self) -> int:
        """Frames a decoded run starts with that belong to no frame of
        the clip: the causal pad of its first latent."""
        return (-self.clip) % self.ratio

    @property
    def fade(self) -> int:
        """Frames two runs share, crossfaded."""
        return max(self.overlap * self.ratio - self.pre_frames, 0)

    def latents(self, frames: int) -> int:
        """Latents the encoder gives a clip of ``frames`` frames."""
        if frames == 1:
            return 1
        return -(-frames // self.clip) * self.tokens - self.drop

    def runs(self, latents: int) -> tuple[int, int]:
        """The latents repeated onto the end and the decoder runs, for
        ``latents`` latents."""
        total = latents + self.drop
        pad = (-total) % self.tokens
        n = (total + pad) // self.tokens - int(self.drop > 0)
        if n < 1:
            pad += self.tokens
            n += 1
        return pad, n

    def frames(self, latents: int) -> int:
        """Frames the decoder gives ``latents`` latents."""
        if latents == 1:
            return 1
        pad, n = self.runs(latents)
        total = latents + pad
        head = self.tokens * self.ratio
        out = tail = 0
        for i in range(n):
            lo = i * self.tokens
            got = max(0, min(lo + self.tokens + self.overlap, total)
                      - min(lo, total)) * self.ratio
            out += max(0, min(head, got) - self.pre_frames)
            if self.drop > 0:
                tail = max(0, min(2 * head, got) - head - self.pre_frames)
        return out + tail - self._pad_frames(latents, pad)

    def _pad_frames(self, latents: int, pad: int) -> int:
        if pad <= 0:
            return 0
        rest = self.clip % self.ratio
        if rest == 0:
            return pad * self.ratio
        return sum(rest if (latents + k) % self.tokens == 0 else self.ratio
                   for k in range(pad))
