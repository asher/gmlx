"""FastH3's video sparse attention: the tile map of gmlx's packed layout
against FastVideo's index arrays at three FastH3 shapes, and the refusal
of a video anchor. CPU-only."""
from __future__ import annotations

import json

import numpy as np
import pytest
from video_fixtures import FASTH3_FV_GEOMETRY, FIXTURES, load_json

from gmlx.video import layout
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import vsa

_SHAPES = {"512x512x22": (512, 512, 22), "768x768x39": (768, 768, 39),
           "1344x768x124": (1344, 768, 124)}


def _layout(width: int, height: int, frames: int, text: int = 56,
            anchors: tuple = ()) -> layout.Layout:
    vs, aus = hf.latent_shapes(frames, width, height)
    return layout.pack(hf.TIMELINE, text, *vs[2:], aus[3], anchors)


def _fv() -> tuple:
    z = np.load(FIXTURES / FASTH3_FV_GEOMETRY)
    return z, json.loads(bytes(z["meta"]).decode())


def test_the_fixture_is_fastvideos_revision():
    manifest = load_json("manifest.json")[FASTH3_FV_GEOMETRY]
    _, meta = _fv()
    assert meta["fastvideo_revision"] == manifest["revision"]
    assert meta["tile_size"] == vsa.TILE


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_tile_map_is_fastvideos(shape):
    """gmlx's layout gives FastVideo's segments: the dense text and audio
    rows, then the video grid (latent frames, H / 2, W / 2). Their tile
    map is FastVideo's, array for array."""
    z, meta = _fv()
    want = meta["shapes"][shape]
    lay = _layout(*_SHAPES[shape])
    segs = vsa.segments(lay, hf.CONSTANTS.values["patch_size"])
    assert [list(s) if isinstance(s, tuple) else s
            for s in segs] == want["segments"]
    got = vsa.tile_map(segs)
    for name in ("tile_gather_index", "prefix_gather_index", "untile_index",
                 "variable_block_sizes"):
        a, b = getattr(got, name), z[f"{shape}/{name}"]
        assert a.dtype == np.int32 and np.array_equal(a, b), name
    assert (got.num_prefix_tiles, got.prefix_length) == (
        want["num_prefix_tiles"], want["prefix_length"])
    assert [list(s) for s in got.video_tile_spans] == want["video_tile_spans"]
    assert got.untile_index.size == lay.rows == want["total_seq_length"]


def test_a_video_anchor_is_refused():
    """FastH3 is text to video: a video anchor would be a second video
    region, which its attention does not run."""
    lay = _layout(512, 512, 22, anchors=(
        layout.AnchorRows(frame=0, video_t=1, audio_t=2),))
    with pytest.raises(ValueError) as ei:
        vsa.segments(lay, (1, 2, 2))
    assert str(ei.value) == (
        "FastH3's video sparse attention runs over one video region, the "
        "target video, and this layout also has 1 video anchor segment")

