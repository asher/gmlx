"""The quants ``gmlx pull`` takes on each memory class, priced by serve
on the real headers: the entry plan admits the class default of every
recipe at that class's working budget, at the image size the docs give
that class. Header reads only. Skipped unless KQUANT_TEST_GGUF_DIR holds
the files under their published names."""
from __future__ import annotations

import functools
import os

import pytest

from gmlx.config import ResolvedImageModel
from gmlx.image import catalog, memfit
from gmlx.serve import capacity, image_service

pytestmark = pytest.mark.integration


def _class_budget(gb: int) -> float:
    """The working budget of a Mac of ``gb`` GB: Metal's working set, 2/3
    of RAM and 3/4 from 64 GB, less the margin, and never closer to RAM
    than the reserve."""
    ram = gb << 30
    ws = ram * (3 / 4 if gb >= 64 else 2 / 3)
    return min(ws * (1 - capacity.margin()), ram - capacity.reserve_bytes(ram))


# The image size a class default makes where the docs give one below
# the family's default: Krea 2 on 16 GB makes images up to 512x512.
_SIZE = {("krea2", 16): "512x512"}


@functools.lru_cache(maxsize=1)
def _index(root: str) -> dict:
    found: dict = {}
    for d, _, names in os.walk(root):
        for n in names:
            if n.endswith(".gguf"):
                found.setdefault(n, []).append(os.path.join(d, n))
    return found


def _file(root, name: str, near: str | None = None) -> str:
    """The file ``name`` under ``root``, the one beside ``near`` first."""
    paths = sorted(_index(str(root)).get(os.path.basename(name), ()))
    if near is not None:
        paths.sort(key=lambda p: os.path.dirname(p) != os.path.dirname(near))
    if not paths:
        pytest.skip(f"{name} is not under KQUANT_TEST_GGUF_DIR")
    return paths[0]


@pytest.mark.parametrize("gb", catalog.MEMORY_CLASSES)
@pytest.mark.parametrize("name", sorted(catalog.RECIPES))
def test_serve_admits_the_default_each_class_pulls(gguf_dir, monkeypatch,
                                                   name, gb):
    r = catalog.RECIPES[name]
    dq, eq = catalog.DEFAULT_QUANTS[name][gb]
    size = _SIZE.get((r.family, gb))
    enc = _file(gguf_dir, r.encoder_name(eq))
    spec = ResolvedImageModel(
        id=name, path=_file(gguf_dir, r.dit_name(dq)),
        vae=_file(gguf_dir, r.vae_pack), encoder=enc, encoder_model=None,
        encoder_mmproj=(_file(gguf_dir, r.mmproj_file, near=enc)
                        if r.mmproj_file else None),
        family=None, adapters=(), defaults={"size": size} if size else {},
        encoder_release="auto",
        pin=False, ttl_s=None)
    budget = _class_budget(gb)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    plan = image_service.entry_plan(spec)
    assert plan.footprint <= budget, (plan.footprint / memfit.GiB,
                                      budget / memfit.GiB)
    print(f"{name} {gb} GB {dq}/{eq} at {size or 'the default size'}: "
          f"{plan.footprint / memfit.GiB:.2f} of "
          f"{budget / memfit.GiB:.2f} GiB, staged {plan.staged}, caches at "
          f"{plan.cache_share:.2f}")


@pytest.mark.parametrize("name", sorted(
    n for n, r in catalog.RECIPES.items() if (r.family, 16) in _SIZE))
def test_serve_refuses_the_16_gb_default_above_its_documented_size(
        gguf_dir, monkeypatch, name):
    r = catalog.RECIPES[name]
    dq, eq = catalog.DEFAULT_QUANTS[name][16]
    spec = ResolvedImageModel(
        id=name, path=_file(gguf_dir, r.dit_name(dq)),
        vae=_file(gguf_dir, r.vae_pack),
        encoder=_file(gguf_dir, r.encoder_name(eq)), encoder_model=None,
        encoder_mmproj=None, family=None, adapters=(),
        defaults={"size": "1024x1024"}, encoder_release="auto", pin=False,
        ttl_s=None)
    budget = _class_budget(16)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    with pytest.raises(memfit.ImageMemoryError, match="1024x1024 image"):
        image_service.entry_plan(spec)
