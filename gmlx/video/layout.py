"""The packed sequence of a video DiT step: its segments and positions.

One sequence holds every row of a step, in segments grouped as the
family orders them (``Timeline.order``): the text, the anchors (a cond and
a cond_audio segment per anchor), the target audio and the target video.
Each row has a position (t, h, w) for RoPE. The target clock starts after
the text, and anchors sit on it at their frame times. The family supplies
the order and the clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The segment groups ``pack`` builds.
GROUPS = ("text", "anchors", "audio", "video")


@dataclass(frozen=True)
class Timeline:
    """The positions a family gives its rows: latent frame k spans
    ``frame_rescale * frame_per_token[k % len]`` time units, each spatial
    axis spans ``grid_scale`` over the square root of the frame's area,
    video rows patchify ``patch`` latent pixels per axis, and ``order``
    lists the segment groups (``GROUPS``) in sequence order."""
    frame_per_token: tuple
    frame_rescale: float
    grid_scale: float
    patch: int
    order: tuple


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    kind: str

    @property
    def rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class AnchorRows:
    """The rows one anchor adds: video latent frames and audio latent
    frames, either may be 0, at a frame index of the target clip."""
    frame: int
    video_t: int = 0
    audio_t: int = 0


@dataclass(frozen=True)
class Layout:
    segments: tuple
    positions: np.ndarray           # (rows, 3) float64: t, h, w
    signature: tuple

    @property
    def rows(self) -> int:
        return self.segments[-1].stop

    def span(self, kind: str) -> Segment:
        """The first segment of ``kind``."""
        for s in self.segments:
            if s.kind == kind:
                return s
        raise KeyError(kind)


def axis(dim: int, patch: int, sqrt_area: float, scale: float) -> np.ndarray:
    """Centered positions of ``dim // patch`` rows along one axis."""
    ratio = dim / sqrt_area
    n = dim // patch
    return (np.arange(n, dtype=np.float64) * (ratio / n)
            + (1.0 - ratio) / 2.0) * scale


def frame_grid(tl: Timeline, lat_h: int, lat_w: int) -> tuple:
    """(h, w) of one latent frame's rows, row-major, and the w axis."""
    area = math.sqrt(lat_h * lat_w)
    hs = axis(lat_h, tl.patch, area, tl.grid_scale)
    ws = axis(lat_w, tl.patch, area, tl.grid_scale)
    hh, ww = np.meshgrid(hs, ws, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), ws


def video_times(tl: Timeline, n: int, origin: float) -> np.ndarray:
    """Start time of each of ``n`` latent frames from ``origin``."""
    fpt = tl.frame_per_token
    spans = np.array([tl.frame_rescale * fpt[k % len(fpt)] for k in range(n)],
                     np.float64)
    return float(origin) + np.concatenate([np.zeros(1), spans[:-1].cumsum()])


def video_positions(tl: Timeline, n: int, frame: np.ndarray,
                    origin: float) -> np.ndarray:
    g = np.empty((n, frame.shape[0], 3), np.float64)
    g[:, :, 0] = video_times(tl, n, origin)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


def audio_positions(cursor: float, t: int, w_low: float,
                    w_high: float) -> np.ndarray:
    """Stereo rows channel-major: t advances per latent frame, each channel
    sits at one end of the w axis, h is 0."""
    g = np.zeros((2 * t, 3), np.float64)
    g[:, 0] = np.tile(cursor + np.arange(t, dtype=np.float64), 2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def pack(tl: Timeline, text_len: int, latent_t: int, lat_h: int, lat_w: int,
         audio_t: int, anchors: tuple = ()) -> Layout:
    """The segments and positions of a step. ``lat_h`` and ``lat_w`` are
    latent pixels; ``audio_t`` counts latent frames per stereo channel."""
    if lat_h % tl.patch or lat_w % tl.patch:
        raise ValueError(f"latent {lat_h}x{lat_w} is not a multiple of the "
                         f"patch {tl.patch}")
    if sorted(tl.order) != sorted(GROUPS):
        raise ValueError(f"segment order {tl.order} is not an order of "
                         f"{', '.join(GROUPS)}")
    frame, ws = frame_grid(tl, lat_h, lat_w)
    w_ends = (float(ws[0]), float(ws[-1]))
    g = np.zeros((text_len, 3), np.float64)
    g[:, 0] = np.arange(text_len, dtype=np.float64)
    origin = float(text_len)
    anchor_rows = []
    for a in anchors:
        at = origin + tl.frame_rescale * a.frame
        if a.video_t:
            anchor_rows.append(
                ("cond", video_positions(tl, a.video_t, frame, at)))
        if a.audio_t:
            anchor_rows.append(
                ("cond_audio", audio_positions(at, a.audio_t, *w_ends)))
    groups = {
        "text": [("text", g)],
        "anchors": anchor_rows,
        "audio": [("audio", audio_positions(origin, audio_t, *w_ends))],
        "video": [("video", video_positions(tl, latent_t, frame, origin))],
    }
    segs, pos, off = [], [], 0
    for kind, p in (row for name in tl.order for row in groups[name]):
        segs.append(Segment(off, off + p.shape[0], kind))
        pos.append(p)
        off += p.shape[0]
    return Layout(segments=tuple(segs), positions=np.concatenate(pos),
                  signature=(text_len, latent_t, lat_h, lat_w, audio_t,
                             tuple(anchors)))
