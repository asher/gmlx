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
