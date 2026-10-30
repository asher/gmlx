"""A video run on a loaded DiT: its adapters, the noise and the denoise
loop, from text features to the final latents.

``VideoPipeline`` is the image pipeline's adapter state over a video DiT,
so a LoRA file loads, stacks and runs live as an image LoRA does
(``gmlx.image.lora``), and the ComfyUI pack's patch tables reach it the
same way. ``adapter_table`` reads LoRA files into a patch table and
``generate`` samples a clip under it.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Callable, cast

import mlx.core as mx

from gmlx.image.api import ImagePipeline, define_patch_table
from gmlx.image.lora import (AdapterError, UnsupportedAdapterError,
                             load_adapter, table_entries)
from gmlx.image.lora import install as lora_install
from gmlx.image.lora.install import linear_leaves

from . import api, layout
from . import loader as video_loader

if TYPE_CHECKING:
    from gmlx.image.registry import Family

    from .registry import VideoFamily


class VideoPipeline(ImagePipeline):
    """An image pipeline whose DiT is a video DiT: ``family`` holds its
    ``VideoFamily`` and ``identified`` its ``VideoIdentified``."""

    @property
    def video_family(self) -> "VideoFamily":
        return cast("VideoFamily", self.family)


def load(path: str, *, variant: str | None = None,
         act_dtype: mx.Dtype | None = None, log=print) -> VideoPipeline:
    """A pipeline over the video DiT GGUF at ``path``, as ``variant`` when
    given and the weights allow it."""
    model, vid = video_loader.load_dit(path, variant=variant,
                                       act_dtype=act_dtype, log=log)
    return VideoPipeline(dit=model, family=cast("Family", vid.family),
                         variant=vid.variant, log=log, identified=vid)


def vet_adapter(family, config: dict, path, *, modules=None) -> None:
    """Refuse a LoRA file for ``family`` from its header alone, before any
    file of a stack loads: a file that is not a safetensors LoRA, a kind
    the family does not run, keys for another family, and with
    ``modules`` a stem on a module the DiT does not hold."""
    from gmlx.image.lora import vet_header
    from gmlx.image.lora.reader import _header_names
    from gmlx.image.vae.reader import safetensors_header

    if family.lora_map is None:
        raise AdapterError(f"{family.title} has no LoRA key map in gmlx")
    path = os.path.expanduser(str(path))
    # Raises AdapterError for a file that is not a safetensors LoRA.
    _header_names(path)
    rule = family.lora_refusal
    if rule is not None:
        header, _ = safetensors_header(path)
        why = rule(header, config)
        if why is not None:
            raise UnsupportedAdapterError(
                f"{path}: gmlx video does not run this LoRA on "
                f"{family.title}: {why[0]}", why[1])
    vet_header(path, keymap=family.lora_map(config), modules=modules)


def _vet(pipe: VideoPipeline, path) -> None:
    vet_adapter(pipe.video_family, dataclasses.asdict(pipe.dit.cfg), path,
                modules=set(linear_leaves(pipe.dit)))


def adapter_table(pipe: VideoPipeline, adapters) -> str:
    """The patch table id of LoRA files at their strengths, ``adapters``
    as (path, strength) pairs; the empty table for none. Each file's
    warnings go to the pipeline's log."""
    adapters = [(path, float(strength)) for path, strength in adapters]
    for path, _ in adapters:
        _vet(pipe, path)
    loaded = []
    for path, strength in adapters:
        ad = load_adapter(pipe, path)
        for w in ad.warnings:
            pipe.log(f"warning: {w}")
        loaded.append((ad, strength))
    return define_patch_table(pipe, table_entries(loaded))


def noise(pipe: VideoPipeline, frames: int, width: int, height: int,
          seed: int) -> tuple[mx.array, mx.array]:
    """Normal video and audio noise of a clip, from ``seed``."""
    vshape, ashape = pipe.video_family.latent_shapes(frames, width, height)
    kv, ka = mx.random.split(mx.random.key(seed))
    return (mx.random.normal(vshape, key=kv),
            mx.random.normal(ashape, key=ka))


def generate(pipe: VideoPipeline, features: mx.array, *, frames: int,
             width: int, height: int, steps: int, shifts: dict,
             seed: int | None = None, start: tuple | None = None,
             table: str | None = None, sigmas: dict | None = None,
             on_step: Callable | None = None) -> tuple[mx.array, mx.array]:
    """The video (1, C, T, H, W) and audio (1, C, 2, T) latents of a clip
    from encoder ``features`` (1, L, text dim), sampled under patch table
    ``table`` from the noise of ``seed``, or from ``start`` as (video,
    audio) noise, on ``sigmas`` per stream when given (``api.sample``).
    The text rows run once, under the same table."""
    fam = pipe.video_family
    dim = int(pipe.dit.cfg.text_dim)
    if features.ndim != 3 or features.shape[0] != 1 \
            or features.shape[1] == 0 or features.shape[2] != dim:
        raise ValueError(f"features {tuple(features.shape)} are not one "
                         f"prompt's encoder rows (1, L, {dim})")
    vshape, ashape = fam.latent_shapes(frames, width, height)
    if start is None:
        if seed is None:
            raise ValueError("generate takes a seed or start latents")
        start = noise(pipe, frames, width, height, seed)
    xv, xa = start
    if tuple(xv.shape) != vshape or tuple(xa.shape) != ashape:
        raise ValueError(f"start latents {tuple(xv.shape)} and "
                         f"{tuple(xa.shape)} are not the {frames}-frame "
                         f"{width}x{height} shapes {vshape} and {ashape}")
    lay = layout.pack(fam.timeline, int(features.shape[1]), vshape[2],
                      vshape[3], vshape[4], ashape[3])
    tid = pipe.default_table if table is None else table
    if tid not in pipe.patch_tables:
        raise ValueError(f"unknown patch table {table!r}")
    model = pipe.dit
    prev = lora_install.activate(pipe, tid)
    try:
        text = model.caption(features)
        mx.eval(text)
        return api.sample(model, xv, xa, text, lay, steps=steps,
                          shifts=shifts, sigmas=sigmas, on_step=on_step)
    finally:
        lora_install.restore(pipe, prev)
