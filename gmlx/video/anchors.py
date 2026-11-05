"""Anchors: clean latents a clip is conditioned on, at a frame index.

A keyframe or a guide is an anchor. Its rows enter the sequence as extra
cond and cond_audio rows on the target's grid (``layout.pack``), not in
place of target rows. A family's rule mixes a little noise into the
rows, ``aug * z + (1 - aug) * noise`` from a generator every anchor
restarts at the same seed, with no noise at ``aug`` 1, and runs them at
a time of at least ``aug``.

``place`` applies ComfyUI's rule for where an anchor sits and what it
keeps, before its frames or audio are encoded: a guide clip keeps its
first frame alone under 5 frames, else its first 17 k + 5; a negative
frame counts from the end; the kept frames fit the clip; and audio keeps
the latent frames left on the clip's audio track from the anchor's time.
``encode`` then makes the anchor of a placement from its frames and its
waveform with the family's VAEs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

# Seeds and their offsets wrap to 64 bits, the range of a ComfyUI seed.
SEED_RANGE = 1 << 64


@dataclass(frozen=True)
class Anchor:
    """A frame index of the target clip, counted from the start once
    placed, with a video latent (1, C, T, H, W), an audio latent (1, C, 2,
    T), or both."""
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


@dataclass(frozen=True)
class Placement:
    """Where an anchor sits and what it keeps: its frame counted from the
    start, the video frames it keeps (0 for audio alone), and the most
    audio latent frames a channel it keeps (None without audio)."""
    frame: int
    video_frames: int = 0
    audio_t: int | None = None


def guide_frames(n: int) -> int:
    """The frames of an ``n``-frame guide an anchor keeps: its first
    frame alone under 5, else its first 17 k + 5."""
    if n < 1:
        raise ValueError("a guide holds no frames")
    return 1 if n < 5 else n - (n - 5) % 17


def place(frame: int, frames: int, audio_t: int, frame_rescale: float, *,
          video_frames: int = 0, audio: bool = False) -> Placement:
    """The placement of an anchor at ``frame`` of a clip of ``frames``
    frames and ``audio_t`` audio latent frames a channel, with
    ``video_frames`` frames of video (0 for none) and audio or not. The
    audio keeps floor(audio_t - frame_rescale x frame) latent frames."""
    if not video_frames and not audio:
        raise ValueError("an anchor needs frames or audio")
    kept = guide_frames(video_frames) if video_frames else 1
    at = frame + frames if frame < 0 else frame
    if at < 0 or at + kept > frames:
        if kept == 1:
            raise ValueError(f"frame {frame} is outside the clip's {frames} "
                             "frames")
        raise ValueError(f"a {kept}-frame guide clip at frame {frame} does "
                         f"not fit in the clip's {frames} frames")
    max_rt = None
    if audio:
        max_rt = math.floor(audio_t - frame_rescale * at)
        if max_rt < 1:
            raise ValueError(f"frame {frame} is past the end of the clip's "
                             "audio")
    return Placement(at, kept if video_frames else 0, max_rt)


def encode(placement: Placement, *, frames=None, waveform=None,
           video_vae=None, audio_vae=None) -> Anchor:
    """The anchor of ``placement``: its kept frames of ``frames``, (T, H,
    W, 3) uint8 at the canvas size, through ``video_vae``, and
    ``waveform``, (2, samples) at the audio VAE's rate, through
    ``audio_vae``, read as far as its kept latents need and cut to
    them, as a whole encode cut to them would be."""
    import mlx.core as mx

    video = audio = None
    if placement.video_frames:
        if frames is None or len(frames) < placement.video_frames:
            raise ValueError(f"a placement keeps {placement.video_frames} "
                             "frames, and fewer were given")
        x = mx.array(frames[:placement.video_frames])
        if x.dtype != mx.uint8 or x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"anchor frames are (T, H, W, 3) uint8, not "
                             f"{x.dtype} {tuple(x.shape)}")
        video = video_vae.encode((x.astype(mx.float32) / 255.0)[None]
                                 * 2.0 - 1.0)
        mx.eval(video)
    if placement.audio_t is not None:
        if waveform is None:
            raise ValueError("a placement keeps audio, and no waveform was "
                             "given")
        w = mx.array(waveform, mx.float32)
        if w.ndim != 2 or w.shape[0] != 2:
            raise ValueError(f"an anchor waveform is (2, samples), not "
                             f"{tuple(w.shape)}")
        need = audio_vae.samples_for(placement.audio_t)
        audio = audio_vae.encode(w[None, :, :need])[..., :placement.audio_t]
        mx.eval(audio)
    return Anchor(placement.frame, video, audio)


def layout_rows(anchors) -> tuple:
    """The ``layout.AnchorRows`` of placed anchors, from their latents."""
    from .layout import AnchorRows

    return tuple(AnchorRows(
        a.frame, video_t=0 if a.video is None else int(a.video.shape[2]),
        audio_t=0 if a.audio is None else int(a.audio.shape[-1]))
        for a in anchors)


def resolve_frame(frame: int, frames: int) -> int:
    """A frame index counted from the start, for a clip of ``frames``."""
    f = frame + frames if frame < 0 else frame
    if not 0 <= f < frames:
        raise ValueError(f"frame {frame} is outside a clip of {frames} "
                         f"frames")
    return f


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
