"""Keyframe and guide images at the canvas size: the resizes against
ComfyUI's MiniMax-H3 nodes (scripts/video_oracle/keyframes_h3.py), the
crop offset's rounding, and the file reader against ComfyUI's LoadImage,
through ffmpeg (skipped without it on PATH) and through Pillow. CPU-only,
no weights."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from video_oracle_fixture import committed, committed_fixture

from gmlx.video import images

NAME = "keyframes_h3"
RESIZES = committed(NAME)["inputs"]["resizes"]
STILLS = committed(NAME)["inputs"]["stills"]
# A reader: the ffmpeg it is given, None for Pillow.
READERS = {"ffmpeg": "ffmpeg", "pillow": None}


@pytest.fixture(params=sorted(READERS))
def reader(request):
    exe = READERS[request.param]
    if exe is not None and shutil.which(exe) is None:
        pytest.skip(f"no {exe} on PATH")
    return exe


@pytest.mark.parametrize("case", sorted(RESIZES))
def test_the_resizes_are_comfyuis(case):
    """The first frame stretched and the others cropped about the center,
    pixel for pixel as _resize gives them from the same 8-bit pixels."""
    arrays, _ = committed_fixture(NAME)
    src, want = arrays[f"resize_{case}_src"], arrays[f"resize_{case}_out"]
    width, height = RESIZES[case]["target"]
    assert list(src.shape[1::-1]) == RESIZES[case]["source"]
    resize = images.stretch if RESIZES[case]["crop"] == "disabled" \
        else images.cover
    got = resize(src, width, height)
    assert got.shape == want.shape == (height, width, 3)
    np.testing.assert_array_equal(got, want)


def test_a_half_pixel_crop_offset_rounds_as_python_rounds():
    """1.5 columns a side goes to 2, which the fixture's half_x case pins;
    2.5 goes to 2 as well, as Python's round takes a half to the even."""
    assert images.crop_box(103, 100, 100, 100) == (2, 0)
    assert images.crop_box(100, 103, 100, 100) == (0, 2)
    assert images.crop_box(105, 100, 100, 100) == (2, 0)
    assert images.crop_box(1920, 1080, 1344, 768) == (15, 0)
    assert images.crop_box(1080, 1920, 1344, 768) == (0, 651)
    assert images.crop_box(1344, 768, 1344, 768) == (0, 0)


def test_an_image_that_is_not_8_bit_rgb_is_refused():
    with pytest.raises(ValueError, match=r"an image is \(H, W, 3\) uint8, "
                                         r"not float32 \(4, 4, 3\)"):
        images.cover(np.zeros((4, 4, 3), np.float32), 2, 2)
    with pytest.raises(ValueError, match=r"not uint8 \(4, 4, 4\)"):
        images.stretch(np.zeros((4, 4, 4), np.uint8), 2, 2)


@pytest.mark.parametrize("still", sorted(STILLS))
def test_a_still_reads_as_comfyuis_loadimage(tmp_path, reader, still):
    """A JPEG and an RGBA PNG saved with EXIF orientation 6 come out
    upright and without alpha as LoadImage gives them: ffmpeg's decode
    exactly, Pillow's within the 8 levels it measures on the JPEG."""
    arrays, _ = committed_fixture(NAME)
    fmt = STILLS[still]["format"].lower()
    path = tmp_path / f"{still}.{fmt}"
    path.write_bytes(arrays[f"still_{still}"].tobytes())
    want = arrays[f"still_{still}_out"]
    w, h = STILLS[still]["size"]
    assert want.shape == (w, h, 3)
    got = images.read(str(path), reader)
    assert got.dtype == np.uint8 and got.shape == want.shape
    gap = np.abs(got.astype(int) - want.astype(int)).max()
    assert gap <= (8 if reader is None and fmt == "jpeg" else 0)


def test_the_reader_turns_by_exif_and_drops_alpha(tmp_path, reader):
    from PIL import Image

    rng = np.random.default_rng(3)
    rgba = rng.integers(0, 256, (6, 10, 4), dtype=np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    exif = img.getexif()
    exif[0x0112] = 6
    path = tmp_path / "turned.png"
    img.save(path, exif=exif.tobytes())
    got = images.read(str(path), reader)
    assert got.dtype == np.uint8 and got.shape == (10, 6, 3)
    np.testing.assert_array_equal(got, np.rot90(rgba[..., :3], k=-1))
    plain = tmp_path / "plain.png"
    Image.fromarray(rgba[..., :3]).save(plain)
    np.testing.assert_array_equal(images.read(str(plain), reader),
                                  rgba[..., :3])


def test_a_truncated_image_names_its_file(tmp_path, reader):
    """A PNG cut short, as a copy that stopped: Pillow's error names the
    file, as ffmpeg's does, and is the ValueError a run refuses with. The
    ffmpeg reader reads its EXIF orientation through Pillow first."""
    from PIL import Image

    rng = np.random.default_rng(5)
    path = tmp_path / "first.png"
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
        path)
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2])
    with pytest.raises(ValueError, match=f"^Pillow could not read {path}: "):
        images.read(str(path), reader)
