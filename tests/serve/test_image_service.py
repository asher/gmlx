#!/usr/bin/env python3
"""Image models in the serve pool: config family names and default sizes.
CPU-only; the loads and the request path run in the image e2e."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gmlx.serve import image_service


def test_family_names_match_on_letters_and_digits():
    from gmlx.image.registry import get_family

    z, q = get_family("z_image"), get_family("qwen_image21")
    assert image_service.family_matches("z-image", z, "turbo")
    assert image_service.family_matches("z-image-turbo", z, "turbo")
    assert image_service.family_matches("qwen-image-2.1", q, "default")
    assert not image_service.family_matches("qwen-image-2.1", z, "turbo")
    assert not image_service.family_matches("z-image-base", z, "turbo")


@pytest.mark.parametrize("size,want", [(None, (1024, 1024)),
                                       ("768x512", (768, 512))])
def test_default_size_takes_the_entry_then_the_family(size, want):
    spec = SimpleNamespace(defaults={"size": size} if size else {})
    fam = SimpleNamespace(size=(1024, 1024))
    assert image_service.default_size(spec, fam) == want


def test_request_fields_map_and_aliases_reach_the_default():
    req = image_service.parse_request("generations", {
        "prompt": " a cat ", "model": "gpt-image-1", "n": "3", "size": "768x512",
        "quality": "standard", "output_compression": "20", "stream": "false",
        "adapters": '[{"id": "eyes", "strength": 0.5}, "plain"]'})
    assert (req.prompt, req.model, req.n, req.size) == ("a cat", "", 3, (768, 512))
    assert req.quality == "medium" and req.compression == 20
    assert req.adapters == [("eyes", 0.5), ("plain", 1.0)]
    assert "output_compression" in image_service.parse_request(
        "generations", {"prompt": "x", "output_compression": 5}).ignored
    for name in ("", "default", "dall-e-3", "chatgpt-image-latest"):
        assert image_service.default_model_field(name) == ""
    assert image_service.default_model_field("zit") == "zit"


def test_png_keeps_alpha_and_jpeg_and_webp_take_compression():
    import io

    import numpy as np
    from PIL import Image

    px = np.zeros((16, 24, 4), np.uint8)
    px[..., 3] = 128
    im = Image.open(io.BytesIO(image_service.encode_output(px, "png")))
    assert im.mode == "RGBA" and im.size == (24, 16)
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)
    hi = image_service.encode_output(rgb, "jpeg", 0)
    lo = image_service.encode_output(rgb, "jpeg", 90)
    assert len(lo) < len(hi) and Image.open(io.BytesIO(hi)).format == "JPEG"
    assert Image.open(io.BytesIO(
        image_service.encode_output(px, "webp", 50))).format == "WEBP"


def test_png_metadata_rides_only_when_asked():
    import io

    import numpy as np
    from PIL import Image

    px = np.zeros((8, 8, 3), np.uint8)
    plain = Image.open(io.BytesIO(image_service.encode_output(px, "png")))
    meta = Image.open(io.BytesIO(
        image_service.encode_output(px, "png", metadata={"seed": 3})))
    assert "gmlx" not in plain.info and '"seed": 3' in meta.info["gmlx"]


def test_a_mask_edits_where_its_alpha_is_clear():
    import numpy as np

    px = np.full((4, 4, 4), 255, np.uint8)
    px[:2, :, 3] = 0
    m = image_service.mask_from_image(px, (4, 4))
    assert m[:2].tolist() == [[1.0] * 4] * 2 and m[2:].sum() == 0
    with pytest.raises(image_service.ImageRequestError):
        image_service.mask_from_image(px[..., :3], (4, 4))


def test_decode_refuses_what_is_not_an_image():
    with pytest.raises(image_service.ImageRequestError) as e:
        image_service.decode_image(b"not an image", "image")
    assert e.value.status == 400 and e.value.param == "image"


def test_partials_fall_evenly_inside_the_run():
    assert image_service.partial_steps(8, 3) == [1, 3, 5]
    assert image_service.partial_steps(8, 1) == [3]
    assert image_service.partial_steps(8, 0) == []
    assert image_service.partial_steps(2, 3) == []
    assert all(1 <= i <= 38 for i in image_service.partial_steps(40, 3))


def test_the_file_store_expires_and_caps_oldest_first(tmp_path):
    import os

    now = [1000.0]
    store = image_service.FileStore(str(tmp_path / "f"), ttl_s=60,
                                    max_bytes=10, time_fn=lambda: now[0])
    a = store.put(b"aaaa", "png")
    os.utime(store.path(a, "png"), (999, 999))
    b = store.put(b"bbbb", "png")
    os.utime(store.path(b, "png"), (1000, 1000))
    assert len(a) == image_service.FileStore.TOKEN_CHARS
    assert store.path(a, "png") and store.path(b, "png")
    c = store.put(b"cccc", "png")          # 12 bytes over a cap of 10
    assert store.path(a, "png") is None
    assert store.path(b, "png") and store.path(c, "png")
    now[0] = 1000.0 + 61
    assert store.path(b, "png") is None
    assert store.path(c, "jpeg") is None and store.path("../x", "png") is None
    assert store.sweep() >= 1


def test_quality_steps_come_from_the_family_table():
    from gmlx.image.registry import get_family

    z, q = get_family("z_image"), get_family("qwen_image21")
    assert image_service.quality_steps(z, "turbo", "high") == 8
    assert image_service.quality_steps(z, "turbo", "low") == 4
    assert image_service.quality_steps(q, "default", "medium") == 28
    assert image_service.quality_steps(q, "default", "auto") is None
