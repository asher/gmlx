"""The quants ``gmlx pull`` takes on each memory class, priced by serve
on the real headers: the entry plan admits the class default of every
recipe at that class's working budget, at the ``defaults.size`` the pull
writes on that class, and refuses the defaults that make no image there.
Header reads only. Skipped unless KQUANT_TEST_GGUF_DIR holds
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
    """The working budget of a Mac of ``gb`` GB."""
    return capacity.ram_budget_bytes(gb << 30)


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


# The class defaults that make no image, as catalog.NO_IMAGE records them.
NO_IMAGE = {(n, gb) for n, classes in catalog.NO_IMAGE.items()
            for gb in classes}


@pytest.mark.parametrize("name, gb", [
    (n, gb) for n in sorted(catalog.RECIPES) for gb in catalog.MEMORY_CLASSES
    if (n, gb) not in NO_IMAGE])
def test_serve_admits_the_default_each_class_pulls(gguf_dir, monkeypatch,
                                                   name, gb):
    r = catalog.RECIPES[name]
    dq, eq = catalog.DEFAULT_QUANTS[name][gb]
    size = catalog.DEFAULT_SIZES.get(name, {}).get(gb)
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


@pytest.mark.parametrize("name, gb", sorted(NO_IMAGE))
def test_serve_refuses_the_class_default_that_makes_no_image(
        gguf_dir, monkeypatch, name, gb):
    """Not even a 256x256 image of a 9B default fits the 16 GB class."""
    r = catalog.RECIPES[name]
    dq, eq = catalog.DEFAULT_QUANTS[name][gb]
    spec = ResolvedImageModel(
        id=name, path=_file(gguf_dir, r.dit_name(dq)),
        vae=_file(gguf_dir, r.vae_pack),
        encoder=_file(gguf_dir, r.encoder_name(eq)), encoder_model=None,
        encoder_mmproj=None, family=None, adapters=(),
        defaults={"size": "256x256"}, encoder_release="auto", pin=False,
        ttl_s=None)
    budget = _class_budget(gb)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    with pytest.raises(memfit.ImageMemoryError, match="256x256 image"):
        image_service.entry_plan(spec)


@pytest.mark.parametrize("name", sorted(
    n for n in catalog.RECIPES if 16 in catalog.DEFAULT_SIZES.get(n, {})))
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


def test_the_kv_edit_workflow_fits_the_24_gb_class(gguf_dir, monkeypatch):
    """The FLUX.2 klein 9B KV edit workflow of the ComfyUI pack on the
    24 GB class default: a 1024x1024 image from two 1024x1024 references
    under the Flux KV Cache node. Its first step, which keeps the K and V
    of both references, and a later step on them each fit the working
    budget beside the entry's parts, with the tensor caches at their share
    and no adapter, as the workflow loads none."""
    from gmlx.image import api
    from gmlx.image.dtypes import image_activation

    name, gb = "flux-2-klein-9b-kv", 24
    r = catalog.RECIPES[name]
    dq, eq = catalog.DEFAULT_QUANTS[name][gb]
    spec = ResolvedImageModel(
        id=name, path=_file(gguf_dir, r.dit_name(dq)),
        vae=_file(gguf_dir, r.vae_pack),
        encoder=_file(gguf_dir, r.encoder_name(eq)), encoder_model=None,
        encoder_mmproj=None, family=None, adapters=(), defaults={},
        encoder_release="auto", pin=False, ttl_s=None)
    budget = _class_budget(gb)
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: budget)
    monkeypatch.setattr(capacity, "overcommit", lambda: False)
    entry = image_service.entry_plan(spec)
    tensor, adapter = image_service.cache_bytes()
    caches = int((2 * tensor + image_service._CACHE_BYTES["steps"] - adapter)
                 * entry.cache_share)
    act = image_activation("flux2_klein_9b_kv")
    row = memfit.PEAKS["flux2_klein_9b_kv"]
    refs = 2 * (1024 // row.prefix.token_px) ** 2
    held = row.prefix.cache_bytes(refs, act.dit.size)
    for step, cache in (("build", caches), ("read", caches + held)):
        plan = memfit.plan_memory(
            api._header_bytes(spec.path, "dit"),
            (api._header_bytes(spec.encoder, "encoder")
             if entry.keep_encoder else 0),
            api._header_bytes(spec.vae, "vae"),
            family="flux2_klein_9b_kv", height=1024, width=1024,
            dtype=act.dit, vae_dtype=act.vae, encoder_owned=False,
            prefix_rows=[refs], step_refs=[(1024, 1024)] * 2,
            ref_cache=step, cache_bytes=cache, budget=budget,
            overcommit=False)
        print(f"{name} {gb} GB {dq}, the {step} step: "
              f"{plan.peak / memfit.GiB:.2f} of "
              f"{budget / memfit.GiB:.2f} GiB")
        assert plan.fits
