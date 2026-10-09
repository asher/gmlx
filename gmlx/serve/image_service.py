"""Image models in the serve residency pool.

An image entry holds a ``ServedImage``: the pipeline that the image
``load`` built, with the plan it was priced by and its configured
adapters. Every load, forward and release of an image model runs on one
worker thread, ``image-worker``, since MLX streams are per thread and
the arrays of a pipeline belong to the thread that made them. Jobs run
one at a time in the order they come, and interleave with chat decode.

The pool counts an entry at its weights plus an activation reserve: the
peak of one image at the entry's default size, from the image memory
plan. With the text encoder released after each encode, that is the
larger of the encode phase and the denoise phase, not their sum.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .subservice import SingleWorker

_log = logging.getLogger(__name__)

WORKER = SingleWorker("image-worker")

# encoder_release auto keeps the encoder while the whole entry fits in
# this share of the working budget.
_KEEP_SHARE = 0.8
_DEFAULT_SIZE = (1024, 1024)        # width, height


@dataclass(frozen=True)
class EntryPlan:
    footprint: int                  # weights and the activation reserve
    keep_encoder: bool              # the encoder stays loaded between encodes
    family: str
    variant: str
    size: tuple                     # default (width, height)
    cfg: float


@dataclass(eq=False)
class ServedImage:
    """What an image entry holds: the pipeline, the resolved config entry,
    the plan it was priced by and its configured adapters as (adapter,
    strength) rows."""
    pipe: Any
    spec: Any
    plan: EntryPlan
    adapters: list = field(default_factory=list)


def _alnum(text: str) -> str:
    return "".join(c for c in str(text).lower() if c.isalnum())


def family_matches(name: str, family, variant: str) -> bool:
    """Whether a configured ``family`` value names this family: its name,
    its title, or its name with the variant, compared on letters and
    digits alone, so ``qwen-image-2.1`` and ``z-image-turbo`` match."""
    want = _alnum(name)
    return want in {_alnum(family.name), _alnum(family.title),
                    _alnum(family.name + variant),
                    _alnum(family.title + variant)}


def default_size(spec, defaults) -> tuple:
    """(width, height) of a request that names no size."""
    size = (spec.defaults or {}).get("size")
    if size:
        w, _, h = str(size).partition("x")
        return int(w), int(h)
    return tuple(defaults.size) if defaults is not None else _DEFAULT_SIZE


def entry_plan(spec) -> EntryPlan:
    """The footprint of an image entry and whether it keeps its encoder,
    from the headers of its files. Raises for a file that is no image
    model, a family the config does not name, and an entry that cannot
    hold one image of its default size in the working budget."""
    from gmlx.image import memfit
    from gmlx.image.api import _header_bytes, sampler_defaults
    from gmlx.image.dtypes import image_activation
    from gmlx.image.loader import identify_dit

    from .capacity import overcommit, working_budget_bytes

    ident = identify_dit(spec.path)
    fam = ident.family
    if spec.family and not family_matches(spec.family, fam, ident.variant):
        raise ValueError(
            f"image model {spec.id!r}: {os.path.basename(spec.path)} is a "
            f"{fam.title} DiT, not the configured family {spec.family!r}")
    defaults = sampler_defaults(fam.name, ident.variant)
    width, height = default_size(spec, defaults)
    cfg = (spec.defaults or {}).get("guidance")
    if cfg is None:
        cfg = defaults.cfg if defaults is not None else 1.0
    act = image_activation(fam.name)
    adapters = [2 * os.path.getsize(p) for p, _ in spec.adapters]
    dit = _header_bytes(spec.path, "dit")
    vae = _header_bytes(spec.vae, "vae")
    enc = mmproj = 0
    if spec.encoder is not None:
        enc = _header_bytes(spec.encoder, "encoder")
        mmproj = _header_bytes(spec.encoder_mmproj, "mmproj")
    common = dict(adapters=adapters, family=fam.name, height=height,
                  width=width, cfg=cfg, dtype=act.dit, vae_dtype=act.vae,
                  encoder_owned=False, overcommit=True)
    keep = memfit.plan_memory(dit, enc, vae, mmproj, **common)
    budget = working_budget_bytes()
    footprint, keeps = keep.peak, True
    if enc and (spec.encoder_release == "always" or (
            spec.encoder_release == "auto" and budget is not None
            and keep.peak > _KEEP_SHARE * budget)):
        # The encoder loads for each encode beside the DiT and the VAE,
        # and is gone before the first step.
        alone = memfit.plan_memory(dit, 0, vae, **common)
        footprint, keeps = max(keep.phases["encode"], alone.peak), False
    if budget is not None and footprint > budget and not overcommit():
        raise memfit.ImageMemoryError(
            f"image model {spec.id!r} needs {footprint / memfit.GiB:.2f} GiB "
            f"for one {width}x{height} image, over the working budget of "
            f"{budget / memfit.GiB:.2f} GiB. Set a smaller defaults.size, or "
            "encoder_release: always. GMLX_OVERCOMMIT=1 overrides.")
    return EntryPlan(footprint=int(footprint), keep_encoder=keeps,
                     family=fam.name, variant=ident.variant,
                     size=(width, height), cfg=float(cfg))


def entry_footprint(spec) -> int:
    return entry_plan(spec).footprint


def _log_line(*parts) -> None:
    _log.info(" ".join(str(p) for p in parts))


def _load(spec) -> ServedImage:
    from gmlx.image import api, lora

    plan = entry_plan(spec)
    # A shared encoder is a chat model of the pool, which this entry does
    # not load; its footprint is counted there. An owned encoder loads
    # here, and one the plan does not keep is released after each encode
    # and loaded again for the next.
    pipe = api.load(spec.path, text_encoder=spec.encoder, vae=spec.vae,
                    mmproj=spec.encoder_mmproj if spec.encoder else None,
                    memory="off", log=_log_line)
    rows = []
    if spec.adapters:
        for path, strength in spec.adapters:
            ad = lora.load_adapter(pipe, path)
            for w in ad.warnings:
                _log.warning("image model %s: %s", spec.id, w)
            rows.append((ad, strength))
        if pipe.encoder is not None:
            lora.apply_text_adapters(pipe, rows)
        pipe.default_table = api.define_patch_table(
            pipe, lora.table_entries(rows))
    return ServedImage(pipe=pipe, spec=spec, plan=plan, adapters=rows)


def load_on_worker(spec) -> ServedImage:
    """Load an image entry on the image worker and wait for it."""
    return WORKER.submit(_load, spec).result()


def _release(box: list) -> None:
    served = box.pop() if box else None
    if served is None:
        return
    pipe, served.pipe = served.pipe, None
    if pipe is not None:
        enc = pipe.encoder
        if enc is not None:
            enc.release()           # an owned encoder; a borrowed one stays
        for cache in (pipe.conditionings, pipe.captions, pipe.modulations,
                      pipe.adapters, pipe.tables, pipe.reference_latents):
            cache.clear()
    del pipe, served
    import gc

    gc.collect()
    import mlx.core as mx

    mx.clear_cache()


def release_on_worker(box: list) -> None:
    """Drop the pipeline in ``box``, a one-item list, on the image worker,
    after the jobs queued before it, and return its memory."""
    WORKER.submit(_release, box).result()
