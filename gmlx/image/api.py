"""The image primitives over a loaded pipeline.

``load`` reads the DiT, the text encoder and the VAE into an
``ImagePipeline``, after gating their header sizes against the working
budget. ``load_staged`` reads the text encoder alone, runs the encodes of
a request, releases the encoder and reads the DiT and the VAE after that,
for a request whose memory plan takes the staged load. ``encode_text`` turns a prompt into a ``Conditioning`` through the
family's encoder tap and keeps it in the pipeline's conditioning cache.
A family that edits from reference images takes them in three steps:
``references`` resizes them by the family's rule, ``encode_text`` runs
the prompt with them through an encoder loaded with its vision tower, and
``encode_references`` adds their VAE latents to the conditioning, which
``encode_text`` does itself unless told to leave it for after the
encoder's release.

Latents are ``(B, C, H, W)`` float32 in model space. ``denoise_step``
evaluates the DiT once per row and returns one float32 velocity per row,
such that ``x_next = x + (sigma_next - sigma) * v``. It applies no
guidance and keeps no step counter. ``guide`` combines a positive and a
negative velocity. Rows run as sequential forwards. A DiT that splits its
step, as Z-Image does, gets the modulation of each sigma from a memo keyed
on the float32 bits of sigma and the patch table, and the caption state of
each conditioning from a cache keyed on its key and the patch table. A
family that gives ``step`` parts in its registry row runs through them
under the same two memos, whose keys then also hold the activation dtype.

``sample`` runs the whole loop inside ``memfit.bounded_cache``, so the
buffers one step frees go back to the system instead of the MLX cache,
and it evaluates the latents after every step. It calls ``on_step(i, x)``
after each update and reads ``should_stop()`` before each step; a stop
raises ``SampleStopped`` with the latents reached.

``vae_decode`` and ``vae_encode`` map between latents and images in
``[0, 1]``, whole or in tiles as the memory plan of the request says, or
as ``vae.tiling`` decides for a caller without one.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import memfit
from .conditioning import (ByteLRU, Conditioning, Reference, array_bytes,
                           conditioning_key)
from .ops import kq_routes
from .registry import Family
from .samplers import (img2img_start, inpaint_blend, make_sampler, noised,
                       noise_from_seed)
from .schedule import check_sigmas, zimage_sigmas


@dataclass(frozen=True)
class SamplerDefaults:
    steps: int
    cfg: float
    shift: float
    size: tuple = (1024, 1024)      # width, height
    strength: float = 0.6           # img2img denoise, as diffusers sets it


def sampler_defaults(family: str, variant: str) -> SamplerDefaults | None:
    """The defaults of a family variant from its sampling row, or None for
    an unknown one."""
    from .registry import get_family

    row = get_family(family).sampling.get(variant)
    if row is None:
        return None
    return SamplerDefaults(steps=row.steps, cfg=row.cfg, shift=row.shift)


def _canonical_table(entries: tuple) -> str:
    return json.dumps(
        [{"module": m, "band": None if b is None else list(b),
          "patches": [[h, st] for h, st in p]} for m, b, p in entries],
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _freeze_table(entries: tuple) -> tuple[str, Mapping]:
    """The id and the immutable table of normalized, sorted entries."""
    entries = tuple(sorted(entries, key=lambda e: (e[0], e[1] or ())))
    tid = hashlib.sha256(_canonical_table(entries).encode()).hexdigest()
    return tid, MappingProxyType({"id": tid, "entries": entries})


EMPTY_TABLE, _EMPTY = _freeze_table(())

CONDITIONING_BYTES = 256 << 20
CAPTION_BYTES = 256 << 20
MODULATION_BYTES = 64 << 20
ADAPTER_BYTES = 4 << 30
TABLE_BYTES = 2 << 30
REFERENCE_BYTES = 64 << 20


@dataclass(eq=False)
class ImagePipeline:
    dit: nn.Module
    family: Family
    variant: str
    vae: object | None = None
    encoder: object | None = None           # text_encoder.EncoderHandle
    patch_tables: dict = field(                 # id -> frozen table
        default_factory=lambda: {EMPTY_TABLE: _EMPTY})
    default_table: str = EMPTY_TABLE            # what patch_table=None means
    conditionings: ByteLRU = field(
        default_factory=lambda: ByteLRU(CONDITIONING_BYTES))
    captions: ByteLRU = field(default_factory=lambda: ByteLRU(CAPTION_BYTES))
    modulations: ByteLRU = field(
        default_factory=lambda: ByteLRU(MODULATION_BYTES))
    adapters: ByteLRU = field(                  # pair hash -> lora Pair
        default_factory=lambda: ByteLRU(ADAPTER_BYTES))
    tables: ByteLRU = field(                    # built patch tables
        default_factory=lambda: ByteLRU(TABLE_BYTES))
    reference_latents: ByteLRU = field(         # image hash -> VAE latents
        default_factory=lambda: ByteLRU(REFERENCE_BYTES))
    adapter_files: dict = field(default_factory=dict)  # file -> adapter
    lora: object | None = None                  # lora.install.PatchState
    scope: dict | None = None                   # see state_scope
    identified: object | None = None            # see load_staged
    weights: dict = field(default_factory=dict)  # dit, encoder, vae bytes
    load_seconds: dict = field(default_factory=dict)  # dit, encoder, vae
    log: Callable = print

    @property
    def vae_scale(self) -> int:
        """Image pixels per latent pixel, from the family row."""
        return int(self.family.constants.values["vae_scale_factor"])

    @property
    def size_multiple(self) -> int:
        """What an image's height and width are multiples of: the family
        row's ``size_multiple``, else one DiT patch in pixels."""
        c = self.family.constants.values
        return int(c.get("size_multiple",
                         self.vae_scale * self.dit.cfg.patch))

    @property
    def defaults(self) -> SamplerDefaults:
        row = sampler_defaults(self.family.name, self.variant)
        if row is None:
            raise ValueError(
                f"no sampler defaults for {self.family.title} "
                f"{self.variant}; pass steps and cfg")
        return row

    def sigmas(self, steps: int, tokens: int | None = None) -> np.ndarray:
        """The family's schedule of ``steps``. ``tokens`` is the image
        token count of the request, which a dynamic shift reads."""
        if self.family.schedule is not None:
            return self.family.schedule(steps, tokens=tokens,
                                        variant=self.variant)
        if self.family.name == "z_image":
            row = self.family.sampling.get(self.variant)
            shift = (self.family.constants.values["scheduler_shift"]
                     if row is None else row.shift)
            return zimage_sigmas(steps, shift)
        raise ValueError(f"no schedule for {self.family.title}; pass sigmas")


def _borrowed(encoder_model, spec):
    from .text_encoder import borrow_encoder

    if len(encoder_model) == 3:
        model, tokenizer, config = encoder_model
    else:
        model, tokenizer = encoder_model
        args = getattr(model, "args", None)
        config = dict(vars(args)) if args is not None else {}
    return borrow_encoder(model, tokenizer, config, spec)


def _header_bytes(path, kind: str) -> int:
    if path is None:
        return 0
    if kind == "vae":
        from .dtypes import vae_dtype
        from .vae import read_header

        return memfit.vae_bytes(read_header(str(path)), vae_dtype().size)
    from .text_encoder import _header_view

    if kind == "mmproj":
        # Before the family is known: the larger count, every float
        # tensor at float32.
        return memfit.mmproj_bytes(_header_view(str(path)), fp32=True)
    skip = ("output.weight",) if kind == "encoder" else ()
    return memfit.file_bytes(_header_view(str(path)), skip=skip)


def load(dit, *, text_encoder=None, vae=None, mmproj=None,
         encoder_model=None, variant=None, memory: str = "auto",
         log=print) -> ImagePipeline:
    """Load an image pipeline from a DiT GGUF, a text encoder GGUF or a
    resident chat model given as ``encoder_model=(model, tokenizer)``, and
    a VAE. ``mmproj`` is the mmproj GGUF of the text encoder: with it the
    encoder loads with its vision tower, which a request with reference
    images needs and a text request does not. With ``memory="auto"`` the
    weights of all are gated on their header sizes before anything loads;
    ``memory="off"`` skips that gate, and each load still gates its own
    weights."""
    from .dtypes import image_activation
    from .loader import identify_dit, load_dit
    from .text_encoder import encoder_spec, load_encoder
    from .vae import load_vae

    if memory not in ("auto", "off"):
        raise ValueError(f"memory must be 'auto' or 'off', not {memory!r}")
    if text_encoder is not None and encoder_model is not None:
        raise ValueError("pass text_encoder or encoder_model, not both")
    if mmproj is not None and text_encoder is None:
        raise ValueError("an mmproj loads with its text encoder GGUF; pass "
                         "text_encoder")
    weights = {"dit": _header_bytes(dit, "dit"),
               "encoder": _header_bytes(text_encoder, "encoder"),
               "vae": _header_bytes(vae, "vae")}
    if mmproj is not None:
        weights["mmproj"] = _header_bytes(mmproj, "mmproj")
    if memory == "auto":
        memfit.gate_load(sum(weights.values()), str(dit),
                         what="image model",
                         hint="Try smaller quants, or memory='off' to load "
                              "the parts one by one")
    seconds = {}
    t0 = time.perf_counter()
    ident = identify_dit(str(dit), variant=variant)
    found = ident.family
    act = image_activation(found.name)
    # A pipeline without an encoder needs no encoder spec of its family.
    spec = None
    if text_encoder is not None or encoder_model is not None:
        spec = encoder_spec(found)
    model, info = load_dit(str(dit), variant=variant, act_dtype=act.dit,
                           identified=ident, log=log)
    fam = info.identified.family
    seconds["dit"] = time.perf_counter() - t0
    handle = None
    t0 = time.perf_counter()
    if text_encoder is not None:
        handle = load_encoder(str(text_encoder), spec, log=log,
                              mmproj=None if mmproj is None else str(mmproj))
        if mmproj is not None:
            # What the tower holds as loaded, which the plan counts.
            weights["mmproj"] = handle.vision_bytes()
    elif encoder_model is not None:
        handle = _borrowed(encoder_model, spec)
    seconds["encoder"] = time.perf_counter() - t0
    ae = None
    t0 = time.perf_counter()
    if vae is not None:
        ae = load_vae(str(vae), family=fam.name, dtype=act.vae)
    seconds["vae"] = time.perf_counter() - t0
    if encoder_model is not None:
        weights["encoder"] = sum(
            int(v.nbytes) for _, v in _flat(handle.model.parameters()))
    return ImagePipeline(dit=model, family=fam,
                         variant=info.identified.variant, vae=ae,
                         encoder=handle, weights=weights,
                         load_seconds=seconds, log=log)


def load_staged(dit, *, text_encoder, encode: Callable, vae=None,
                mmproj=None, variant=None, log=print) -> tuple:
    """Load an image pipeline in the staged order, for a request whose
    memory plan says ``staged_load``, and return it with what ``encode``
    returned.

    The text encoder loads first, with its vision tower when ``mmproj``
    is given. ``encode(pipe)`` then runs every encode of the request, and
    installs the text encoder halves of its adapters before them, on a
    pipeline that holds the encoder and neither the DiT nor the VAE:
    ``pipe.dit`` and ``pipe.vae`` are None there, and ``pipe.identified``
    is the DiT's header. The encoder is released when ``encode`` returns,
    and the DiT and the VAE load after that, so the encoder is never
    resident beside them. The conditionings ``encode`` made stay in the
    pipeline's cache. Reference images get their VAE latents after the
    load, through ``encode_references``. Each load gates its own weights
    against the working budget."""
    from .dtypes import image_activation
    from .loader import identify_dit, load_dit
    from .text_encoder import encoder_spec, load_encoder
    from .vae import load_vae

    weights = {"dit": _header_bytes(dit, "dit"),
               "encoder": _header_bytes(text_encoder, "encoder"),
               "vae": _header_bytes(vae, "vae")}
    ident = identify_dit(str(dit), variant=variant)
    act = image_activation(ident.family.name)
    seconds = {}
    t0 = time.perf_counter()
    handle = load_encoder(str(text_encoder), encoder_spec(ident.family),
                          log=log,
                          mmproj=None if mmproj is None else str(mmproj))
    if mmproj is not None:
        weights["mmproj"] = handle.vision_bytes()
    seconds["encoder"] = time.perf_counter() - t0
    # The DiT is not loaded yet, and nothing of the encode stage reads it.
    pipe = ImagePipeline(dit=cast(nn.Module, None), family=ident.family,
                         variant=ident.variant, encoder=handle,
                         weights=weights, load_seconds=seconds, log=log,
                         identified=ident)
    out = encode(pipe)
    handle.release()
    t0 = time.perf_counter()
    model, info = load_dit(str(dit), variant=variant, act_dtype=act.dit,
                           identified=ident, log=log)
    pipe.dit, pipe.variant = model, info.identified.variant
    seconds["dit"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if vae is not None:
        pipe.vae = load_vae(str(vae), family=pipe.family.name, dtype=act.vae)
    seconds["vae"] = time.perf_counter() - t0
    return pipe, out


def _flat(params):
    from mlx.utils import tree_flatten

    return tree_flatten(params)


def _prefix_rows(pipe: ImagePipeline, cond: Conditioning) -> int:
    """Rows a conditioning puts before the image rows: its text rows and
    the tokens of its reference latents."""
    p = pipe.dit.cfg.patch
    if cond.pending:
        px = pipe.vae_scale * p
        refs = sum((r.height // px) * (r.width // px) for r in cond.refs)
    else:
        refs = sum((int(r.shape[-2]) // p) * (int(r.shape[-1]) // p)
                   for r in cond.ref_latents)
    return int(cond.features.shape[0]) - len(cond.image_slots) + refs


def _ref_sizes(states) -> list:
    """(height, width) in pixels of each distinct reference image of the
    conditionings, which is one VAE encode each."""
    seen: dict = {}
    for c in states:
        for r in c.refs:
            seen.setdefault(r.sha256, (r.height, r.width))
    return list(seen.values())


def plan_request(pipe: ImagePipeline, *, height: int, width: int,
                 cfg: float | None = None, caption_tokens: int | None = None,
                 conds=None, patch_table=None, init_image: bool = False,
                 keep_encoder: bool = False, budget: float | None = None,
                 overcommit: bool | None = None) -> memfit.MemoryPlan:
    """The memory plan of one request on this pipeline: its weights, an
    encoder still resident, the caches it holds, and the phase peaks.
    ``conds`` is the request's conditioning, or the positive and the
    negative one, whose text rows and reference latents size the prefix
    of a family that caches one, and ``patch_table`` the table the
    request runs under, which names the states it holds already.
    ``init_image`` says the request encodes an init image of its size,
    and ``keep_encoder`` that the caller keeps the encoder through the
    request, so that its release is no lever. The reference images of the
    conditionings are VAE encodes of their own sizes, priced with the init
    image's.
    The VAE passes are priced for the VAE this pipeline loaded, on its
    conv path, and ``vae_decode`` and ``vae_encode`` run them as the plan
    they are given says. Raises ImageMemoryError when no lever makes it
    fit."""
    enc = pipe.encoder
    resident = enc is not None and not enc.released
    act = pipe.dit.act_dtype
    vae_dtype = getattr(pipe.vae, "dtype", None)
    states = ([] if conds is None else [conds]
              if isinstance(conds, Conditioning) else list(conds))
    caches = pipe.conditionings.bytes + pipe.modulations.bytes
    parts = pipe.family.step
    other = bound = 0
    if parts is None:
        caches += pipe.captions.bytes
    else:
        # The states of a step in parts are the plan's prefix cache term.
        # Those of this request count there once, held already or not.
        table = _table_id(pipe, patch_table)
        size = (height // pipe.vae_scale, width // pipe.vae_scale)
        held = sum(pipe.captions.nbytes(key) for key in {
            _state_key(pipe, c, table, size) for c in states if c.key})
        other, bound = pipe.captions.bytes - held, parts.state_bytes
    price = None
    if pipe.vae is not None:
        from .vae import tiling

        price = {op: tiling.price_args(pipe.vae, op, 1)
                 for op in ("decode", "encode")}
    return memfit.plan_memory(
        pipe.weights.get("dit", 0),
        pipe.weights.get("encoder", 0) if resident else 0,
        pipe.weights.get("vae", 0),
        pipe.weights.get("mmproj", 0) if resident else 0,
        adapters=pipe.adapters.bytes + pipe.tables.bytes,
        family=pipe.family.name,
        height=height, width=width,
        cfg=pipe.defaults.cfg if cfg is None else cfg, dtype=act,
        vae_dtype=vae_dtype,
        encoder_owned=bool(resident and enc.owned and not keep_encoder),
        caption_tokens=caption_tokens, cache_bytes=caches,
        prefix_rows=[_prefix_rows(pipe, c) for c in states] or None,
        prefix_shared=(len(states) == 2
                       and _ident(states[0]) == _ident(states[1])),
        prefix_other=other, prefix_bound=bound, init_image=init_image,
        ref_sizes=_ref_sizes(states), vae_price=price, budget=budget,
        overcommit=overcommit)


def references(pipe: ImagePipeline, images, *,
               area: int | None = None) -> tuple:
    """The reference images of a request as the family runs them: each a
    ``Reference``, resized by the family's rule to ``area`` pixels, by
    default the family's own. An image is a PIL image or a uint8 array
    (H, W, 3) or (H, W, 4), and a ``Reference`` passes through. A family
    without reference images and a count over the family's cap are
    refused by name."""
    edit = pipe.family.edit
    images = list(images)
    if edit is None:
        if not images:
            return ()
        raise ValueError(f"{pipe.family.title} takes no reference images")
    if len(images) > edit.max_refs:
        raise ValueError(
            f"{pipe.family.title} takes at most {edit.max_refs} reference "
            f"images, and the request has {len(images)}")
    area = edit.area if area is None else int(area)
    if not edit.min_area <= area <= edit.area:
        raise ValueError(
            f"a reference area of {area} pixels is outside "
            f"{edit.min_area} to {edit.area}, which {pipe.family.title} "
            "takes")
    return tuple(r if isinstance(r, Reference) else edit.prepare(r, area)
                 for r in images)


def _encode_with_images(pipe: ImagePipeline, prompt: str,
                        refs: tuple) -> Conditioning:
    edit, enc = pipe.family.edit, pipe.encoder
    ids = edit.token_ids(enc, prompt, refs)
    key = conditioning_key(
        family=pipe.family.name, encoder=str(enc.source_key),
        adapters=getattr(enc, "adapters", ""), template=edit.template_id(),
        ids=ids, tap_layers=enc.spec.tap_layers, dtype=enc.dtype_name,
        images=[r.identity for r in refs])
    hit = pipe.conditionings.get(key)
    if hit is None:
        _, features, slots = edit.encode(enc, prompt, refs)
        hit = Conditioning(features, key, image_slots=slots, refs=refs)
        pipe.conditionings.put(key, hit, array_bytes([features]))
    return hit


def encode_references(pipe: ImagePipeline, conds, *, plan=None):
    """``conds``, one Conditioning or several, with the VAE latents of
    their reference images. Each image is encoded once, in mode, and its
    latents stay in the pipeline's cache of reference latents under the
    image's hash and the VAE's dtype. ``plan`` is the memory plan of the
    request: an encode runs whole or in tiles as it says. A conditioning
    without reference images comes back as it is."""
    import dataclasses

    one = isinstance(conds, Conditioning)
    out = []
    for c in [conds] if one else list(conds):
        if c.refs and not c.ref_latents:
            lat = tuple(_reference_latents(pipe, r, plan) for r in c.refs)
            c = dataclasses.replace(c, ref_latents=lat)
        out.append(c)
    return out[0] if one else out


def _reference_latents(pipe: ImagePipeline, ref: Reference, plan):
    vae = _need_vae(pipe)
    key = (ref.sha256, str(getattr(vae, "dtype", "")))
    hit = pipe.reference_latents.get(key)
    if hit is not None:
        return hit
    from .vae import tiling

    tiled, room = None, None
    if plan is not None and "init" in plan.phases:
        tiled, room = plan.vae_pass("encode")
        if room is None:
            tiled = None
        else:
            # A reference of one tile runs whole, and one past the pixels
            # a whole pass was measured to runs in tiles.
            tiled = (max(ref.height, ref.width) > memfit.TILE_PX
                     and (tiled or memfit.vae_row(pipe.family.name)
                          .tiles_at(ref.height * ref.width)))
    x = mx.array(ref.pixels.astype(np.float32) / 255.0)[None] * 2 - 1
    z = tiling.encode(vae, x, tiled=tiled, mode="mode", free=room)[0]
    mx.eval(z)
    pipe.reference_latents.put(key, z, array_bytes([z]))
    return z


def encode_text(pipe: ImagePipeline, prompt: str, *, images=(),
                ref_latents: bool = True) -> Conditioning:
    """The conditioning of ``prompt``, from the cache when its key is
    there, else through the encoder tap. ``images`` are the reference
    images of an edit, as ``references`` takes them, and the encoder must
    hold its vision tower. With ``ref_latents`` off the conditioning comes
    back without the VAE latents of its images, for a caller that
    releases the encoder first and then calls ``encode_references``."""
    enc = pipe.encoder
    if images:
        refs = references(pipe, images)
        if enc is None:
            raise ValueError("this pipeline has no text encoder")
        if not getattr(enc, "vision", False) and not enc.released:
            raise ValueError(
                "reference images need the encoder's vision tower: load "
                "the pipeline with the encoder's mmproj")
        cond = _encode_with_images(pipe, prompt, refs)
        return encode_references(pipe, cond) if ref_latents else cond
    if enc is None:
        raise ValueError("this pipeline has no text encoder")
    ids = enc.token_ids(prompt)
    key = conditioning_key(
        family=pipe.family.name, encoder=str(enc.source_key),
        adapters=getattr(enc, "adapters", ""),
        template=enc.spec.template_id, ids=ids,
        tap_layers=enc.spec.tap_layers, dtype=enc.dtype_name)
    hit = pipe.conditionings.get(key)
    if hit is not None:
        return hit
    tap = enc.encode(prompt)
    cond = Conditioning(tap.features, key)
    pipe.conditionings.put(key, cond, array_bytes([cond.features]))
    return cond


class SampleStopped(RuntimeError):
    """``should_stop`` ended a run before ``step``. ``latents`` are the
    latents reached, at ``sigmas[step]``."""

    def __init__(self, step: int, latents: mx.array):
        super().__init__(f"sampling stopped before step {step}")
        self.step = step
        self.latents = latents


def _linear_dims(mod) -> tuple[int, int] | None:
    """(out, in) of a linear module, for a float or a K-quant weight."""
    from .lora.install import linear_dims

    return linear_dims(mod)


def _table_entry(entry, modules: dict) -> tuple:
    if isinstance(entry, Mapping):
        unknown = set(entry) - {"module", "band", "patches"}
        if unknown:
            raise ValueError(f"patch table entry has unknown keys "
                             f"{sorted(unknown)}")
        path, band, patches = (entry.get("module"), entry.get("band"),
                               entry.get("patches"))
    else:
        path, band, patches = tuple(entry)
    if not isinstance(path, str) or path not in modules:
        raise ValueError(f"patch table names {path!r}, which is not a "
                         f"module of this DiT")
    dims = _linear_dims(modules[path])
    if dims is None:
        raise ValueError(f"patch table names {path!r}, which is not a "
                         f"linear layer")
    if band is not None:
        band = tuple(band)
        if len(band) != 3 or not all(isinstance(v, numbers.Integral)
                                     and not isinstance(v, bool)
                                     for v in band):
            raise ValueError(f"band of {path} must be three integers "
                             f"(dim, start, length), not {band!r}")
        band = tuple(int(v) for v in band)
        dim, start, length = band
        if dim not in (0, 1) or start < 0 or length <= 0 \
                or start + length > dims[dim]:
            raise ValueError(f"band {band} does not fit {path}, whose "
                             f"weight is {dims[0]}x{dims[1]} (out x in)")
    if isinstance(patches, (str, bytes)) or not patches:
        raise ValueError(f"patch table entry for {path} has no patches")
    out = []
    for p in patches:
        h, st = tuple(p)
        if not isinstance(h, str) or not h:
            raise ValueError(f"adapter hash of {path} must be a non-empty "
                             f"string, not {h!r}")
        st = float(st)
        if not math.isfinite(st):
            raise ValueError(f"strength of {h} on {path} is {st}")
        out.append((h, 0.0 if st == 0.0 else st))
    return path, band, tuple(out)


def _canonical_rows(rows: list, modules: dict) -> list:
    """One form for rows that give the same delta, so they share an id. A
    band that covers its whole axis is the whole weight, and joins a
    whole-weight row on the same module. The patches of a row sum, so
    their order is not kept. ``_table_entry`` has made band values plain
    integers and a negative zero strength zero."""
    merged: dict = {}
    for path, band, patches in rows:
        if band is not None and band[1] == 0 \
                and band[2] == _linear_dims(modules[path])[band[0]]:
            band = None
        merged.setdefault((path, band), []).extend(patches)
    return [(path, band, tuple(sorted(patches)))
            for (path, band), patches in merged.items()]


def define_patch_table(pipe: ImagePipeline, entries) -> str:
    """Define an immutable patch table from entries of module path, band
    (dim, start, length) or None, and patches of (adapter hash,
    strength), and return its id: the sha256 of its canonical JSON.
    Entries that give the same delta give the same id: the order of the
    entries and of the patches in one, a band that covers the whole
    weight against None, and a strength of -0.0 against 0.0 do not change
    it. An adapter hash names a
    LoRA pair the pipeline holds, from ``gmlx.image.lora``, and a table
    that names a pair it does not hold, or one that does not fit its
    module and band, is refused. The pipeline keeps the table under its
    id, and ``denoise_step`` and ``sample`` apply it when given the id."""
    from .lora import install

    modules = install.modules(pipe.dit)
    rows = [_table_entry(e, modules) for e in entries]
    keys = [(m, b) for m, b, _ in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("patch table names a module and band twice")
    rows = _canonical_rows(rows, modules)
    tid, table = _freeze_table(tuple(rows))
    if rows and tid not in pipe.patch_tables:
        install.check_table(pipe, table)
        install.ensure_wrapped(pipe)
    pipe.patch_tables.setdefault(tid, table)
    return tid


def _table_id(pipe: ImagePipeline, patch_table) -> str:
    """The table id a call runs under: None is the pipeline's default."""
    tid = pipe.default_table if patch_table is None else patch_table
    if tid not in pipe.patch_tables:
        raise ValueError(f"unknown patch table {patch_table!r}")
    return tid


def denoise_step(pipe: ImagePipeline, latents, sigma, conds, *,
                 patch_table=None) -> mx.array:
    """The velocity of every row of ``latents`` at ``sigma``, a scalar or
    one value per row, under ``conds``, one Conditioning or one per row.
    Outside a ``state_scope`` the conditionings of the call are its
    scope."""
    table = _table_id(pipe, patch_table)
    x = mx.array(latents).astype(mx.float32)
    if x.ndim != 4:
        raise ValueError(f"latents must be (B, C, H, W), not {x.shape}")
    b = x.shape[0]
    sig = np.asarray(sigma, np.float32).reshape(-1)
    if sig.size not in (1, b):
        raise ValueError(f"{sig.size} sigmas for {b} rows")
    sig = np.broadcast_to(sig, (b,))
    rows = [conds] * b if isinstance(conds, Conditioning) else list(conds)
    if len(rows) != b:
        raise ValueError(f"{len(rows)} conditionings for {b} rows")
    with state_scope(pipe, rows):
        out = [_row(pipe, x[i:i + 1], sig[i], rows[i], table)
               for i in range(b)]
    return out[0] if b == 1 else mx.concatenate(out, axis=0)


class StateScopeError(RuntimeError):
    """A state scope asked for inside an open scope of other
    conditionings."""


def _ident(cond: Conditioning) -> tuple:
    """What names a conditioning in a scope: its key, the hashes of its
    reference latents and its image slots. A conditioning without a key
    is named by the object it is."""
    if not cond.key:
        return ("", id(cond))
    return (cond.key, tuple(cond.ref_hashes),
            tuple(int(s) for s in cond.image_slots))


def _need_latents(pipe: ImagePipeline, cond: Conditioning) -> None:
    if cond.pending:
        raise ValueError(
            "the conditioning has reference images and not their latents "
            "yet: pass it through encode_references first")


@contextmanager
def state_scope(pipe: ImagePipeline, conds):
    """Hold the conditioning states of one request while it runs.

    A family whose step runs in parts keeps a state per conditioning, for
    Qwen-Image 2.1 the K and V of its prefix in every block. A state is
    kept under the conditioning, the target's latent size, the patch
    table, the activation dtype and the kernel routes of the step that
    built it. A scope holds one state per conditioning: the one of the
    step that last ran it, in the scope or before it. Those states are
    never shed. Every other held
    state is shed, the least recently used first, until the others fit in
    the family's bound less the bytes of the scope's own states, and a
    state that is not built yet counts the bytes it will have. That runs
    before a state is built and on every step that finds its state held.
    So the cache holds one state per conditioning in scope and at most
    that room of others at every moment, which is the prefix cache term
    of ``plan_request``.

    A step in a scope under another patch table, route set or target
    size than the step before takes the conditioning's place in the
    scope: the state of the old step is one of the others from there, and
    it is shed before the new one is built unless the bound has room for
    both. Steps that alternate two tables on one conditioning so pay a
    prefill each when the bound has no such room.

    ``sample`` opens the scope of its request: the positive conditioning,
    and the negative one when guided. A ``denoise_step`` call outside a
    scope has the conditionings of that call as its scope. So a caller
    that alternates two conditionings in separate bare calls keeps both
    states only when both fit the bound together. When they do not, each
    call sheds the other's state and builds its own again, one prefill a
    call. Such a caller opens one scope around its calls, or passes both
    conditionings as the rows of one call.

    A scope opened inside another for conditionings of the open one
    changes nothing. For any other conditioning it raises
    ``StateScopeError``: the open scope holds its states whatever the
    inner request plans, so the two would hold more than either plan.
    The request with the other conditionings runs after the open scope
    closes.
    """
    rows = [conds] if isinstance(conds, Conditioning) else list(conds)
    parts = pipe.family.step
    if parts is None:
        yield
        return
    named: dict = {}
    for c in rows:
        _need_latents(pipe, c)
        named.setdefault(_ident(c), c)
    if pipe.scope is not None:
        if any(i not in pipe.scope for i in named):
            raise StateScopeError(
                "a state scope is open for other conditionings: a request "
                "with its own conditionings runs after the open scope "
                "closes")
        yield
        return
    # ident -> [key of the state the scope holds, bytes it has or will].
    # Until a conditioning steps, its state is the one it used last.
    pipe.scope = {i: [pipe.captions.newest(lambda k, i=i: k[:3] == i)
                      if i[0] else None,
                      0 if parts.state_size is None
                      else int(parts.state_size(pipe.dit, c))]
                  for i, c in named.items()}
    try:
        yield
    finally:
        pipe.scope = None


def _state_key(pipe: ImagePipeline, cond: Conditioning, table, size):
    """The key the state of ``cond`` is kept under: the conditioning with
    the hashes of its reference latents and its image slots, the latent
    size ``(h, w)`` of the target, the patch table, the activation dtype
    and the kernel routes in force. A state is built by the ops of those
    routes, as the results ``_memo`` keeps are. None for a conditioning
    without a key, whose state is never kept."""
    if not cond.key:
        return None
    return (*_ident(cond), (int(size[0]), int(size[1])), table,
            str(pipe.dit.act_dtype), kq_routes.active())


def _shed_states(pipe: ImagePipeline, parts) -> None:
    """Shed held states by the rule of ``state_scope``. A state of the
    scope counts its held bytes, or the bytes it will have before it is
    built."""
    scope = (pipe.scope or {}).values()
    own = sum((pipe.captions.nbytes(key) if key is not None else 0) or n
              for key, n in scope)
    pipe.captions.shed_others({key for key, _ in scope if key is not None},
                              max(0, int(parts.state_bytes) - own))


def _under(pipe: ImagePipeline, table: str, fn):
    """``fn()`` with the DiT's wrapped leaves reading ``table``. The graph
    takes the table's factors while it is built, so the table need not
    stay active until the result is evaluated."""
    if pipe.lora is None:
        return fn()
    from .lora import install

    prev = install.activate(pipe, table)
    try:
        return fn()
    finally:
        install.restore(pipe, prev)


def _parts_row(pipe: ImagePipeline, parts, x: mx.array, sigma: np.float32,
               cond: Conditioning, table) -> mx.array:
    """One row through the family's step parts. The cache of conditioning
    states makes room by the rule of ``state_scope`` before the state of
    ``cond`` is read or built, so a new state is never live beside one
    the rule sheds."""
    dit = pipe.dit
    s = np.float32(sigma)
    dt = str(dit.act_dtype)
    mod = _memo(pipe.modulations, (s.tobytes(), table, dt),
                lambda: _under(pipe, table,
                               lambda: parts.modulation(dit, s)))
    _need_latents(pipe, cond)
    key = _state_key(pipe, cond, table, x.shape[2:])
    held = (pipe.scope or {}).get(_ident(cond))
    if held is not None:
        # The state of this step is the one the scope holds from here.
        held[0] = key
    _shed_states(pipe, parts)
    state = None if key is None else pipe.captions.get(key)
    if state is None:
        state = _under(pipe, table, lambda: parts.conditioning(dit, cond))
        if key is not None:
            arrays = state.arrays()
            mx.eval(arrays)
            pipe.captions.put(key, state, array_bytes(arrays), shed=False)
            # A family that gives no size before the build sheds here.
            _shed_states(pipe, parts)
    return _under(pipe, table,
                  lambda: parts.forward(dit, x, s, cond, state, mod))


def _row(pipe: ImagePipeline, x: mx.array, sigma: np.float32,
         cond: Conditioning, table) -> mx.array:
    dit = pipe.dit
    if pipe.family.step is not None:
        return _parts_row(pipe, pipe.family.step, x, sigma, cond, table)
    if not (hasattr(dit, "modulation") and hasattr(dit, "caption")):
        return _under(pipe, table, lambda: dit(
            x, np.reshape(sigma, (1,)), cond.features[None]))
    mod = _modulation(pipe, sigma, table)
    caption = _caption(pipe, cond, table)
    return _under(pipe, table, lambda: dit(
        x, np.reshape(sigma, (1,)), None, mod=mod, caption=caption))


def _memo(cache: ByteLRU, key, make):
    """``make()`` kept under ``key`` and the kernel routes in force. A
    kernel rounds at other points than the stock ops, so a result made on
    one set of routes is not the result of another."""
    key = (*key, kq_routes.active())
    hit = cache.get(key)
    if hit is None:
        hit = make()
        arrays = hit.arrays()
        mx.eval(arrays)
        cache.put(key, hit, array_bytes(arrays))
    return hit


def _modulation(pipe: ImagePipeline, sigma: np.float32, table):
    s = np.float32(sigma)
    return _memo(pipe.modulations, (s.tobytes(), table),
                 lambda: _under(pipe, table, lambda: pipe.dit.modulation(
                     np.reshape(s, (1,)), 1)))


def _caption(pipe: ImagePipeline, cond: Conditioning, table):
    make = lambda: _under(  # noqa: E731
        pipe, table, lambda: pipe.dit.caption(cond.features[None]))
    if not cond.key:
        return make()
    return _memo(pipe.captions, (cond.key, table), make)


def guide(v_pos: mx.array, v_neg: mx.array, cfg: float) -> mx.array:
    """``neg + cfg * (pos - neg)`` in float32; 1.0 means no guidance."""
    pos, neg = v_pos.astype(mx.float32), v_neg.astype(mx.float32)
    return neg + mx.array(cfg, mx.float32) * (pos - neg)


def _latent_mask(mask, shape: tuple, scale: int) -> mx.array:
    """A mask at image or latent resolution as (B or 1, 1, h, w) float32
    at latent resolution. Image masks average over each latent pixel."""
    m = np.asarray(mask, np.float32)
    if m.ndim == 2:
        m = m[None, None]
    if m.ndim != 4 or m.shape[1] != 1 or m.shape[0] not in (1, shape[0]):
        raise ValueError(f"mask must be (H, W) or (B, 1, H, W), not {m.shape}")
    h, w = shape[2:]
    if m.shape[2:] == (h * scale, w * scale):
        m = m.reshape(m.shape[0], 1, h, scale, w, scale).mean(axis=(3, 5))
    elif m.shape[2:] != (h, w):
        raise ValueError(f"mask size {m.shape[2]}x{m.shape[3]} is neither "
                         f"the image nor the latent size")
    if m.min() < 0 or m.max() > 1:
        raise ValueError("mask values must lie in [0, 1]")
    return mx.array(m)


def sample(pipe: ImagePipeline, cond: Conditioning, *,
           neg: Conditioning | None = None, cfg: float | None = None,
           height: int, width: int, steps: int | None = None,
           seed: int | None = None, noise=None, sampler: str = "euler",
           sigmas=None, init_latents=None, denoise: float = 1.0, mask=None,
           patch_table=None, on_step=None, should_stop=None,
           plan=None) -> mx.array:
    """Sample latents for ``cond``. With ``init_latents`` the run is
    img2img from strength ``denoise``, and a ``mask`` (1 where the image
    changes) makes it inpainting. ``noise`` is (B, C, H, W) float32, else
    it comes from ``seed``, else from a random seed. The batch is that of
    ``init_latents``, else of ``noise``, else 1. ``sampler`` is
    ``euler`` or ``uni_pc_bh2``, and the model sees the schedule's sigmas
    either way. ``plan`` is the memory plan of the request: where it
    runs one row at a time, a guided step evaluates the positive velocity
    before it runs the negative row, and the two give the latents of a
    step that evaluates both together."""
    patch_table = _table_id(pipe, patch_table)
    if cfg is None:
        cfg = pipe.defaults.cfg
    if cfg != 1.0 and neg is None:
        raise ValueError(f"cfg {cfg} needs a negative conditioning")
    unit = pipe.size_multiple
    if height <= 0 or width <= 0 or height % unit or width % unit:
        raise ValueError(f"height and width must be positive multiples of "
                         f"{unit}, not {height}x{width}")
    if not 0.0 < denoise <= 1.0:
        raise ValueError(f"denoise must lie in (0, 1], not {denoise}")
    if init_latents is None and (denoise < 1.0 or mask is not None):
        raise ValueError("denoise below 1 and mask need init_latents")

    if sigmas is None:
        token_px = pipe.vae_scale * pipe.dit.cfg.patch
        s = pipe.sigmas(pipe.defaults.steps if steps is None else steps,
                        (height // token_px) * (width // token_px))
    else:
        s = check_sigmas(sigmas)
        if steps is not None and steps != len(s) - 1:
            raise ValueError(f"steps {steps} disagrees with "
                             f"{len(s) - 1} steps of sigmas")
    init = None
    if init_latents is not None:
        init = mx.array(init_latents).astype(mx.float32)
        s = s[img2img_start(len(s) - 1, denoise):]
        if len(s) < 2:
            raise ValueError(f"denoise {denoise} leaves no step")

    if init is not None:
        b = init.shape[0]
    elif noise is not None and len(getattr(noise, "shape",
                                           np.shape(noise))) == 4:
        b = int(getattr(noise, "shape", np.shape(noise))[0])
    else:
        b = 1
    shape = (b, pipe.dit.cfg.in_channels, height // pipe.vae_scale,
             width // pipe.vae_scale)
    if init is not None and tuple(init.shape) != shape:
        raise ValueError(f"init_latents {tuple(init.shape)} do not match "
                         f"{shape}")
    if noise is None:
        noise = noise_from_seed(secrets.randbelow(1 << 63)
                                if seed is None else seed, shape)
    elif seed is not None:
        raise ValueError("pass seed or noise, not both")
    noise = mx.array(noise).astype(mx.float32)
    if tuple(noise.shape) != shape:
        raise ValueError(f"noise {tuple(noise.shape)} does not match {shape}")
    m = None if mask is None else _latent_mask(mask, shape, pipe.vae_scale)

    x = noise if init is None else noised(s[0], noise, init)
    smp = make_sampler(sampler, s)
    one_row = plan is not None and plan.rows == 1
    scope = state_scope(pipe, [cond] if cfg == 1.0 else [cond, neg])
    with memfit.bounded_cache(), scope:
        for i in range(smp.steps):
            if should_stop is not None and should_stop():
                raise SampleStopped(i, x)
            v = denoise_step(pipe, x, s[i], cond, patch_table=patch_table)
            if cfg != 1.0:
                if one_row:
                    mx.eval(v)
                v = guide(v, denoise_step(pipe, x, s[i], neg,
                                          patch_table=patch_table), cfg)
            x = smp.step(x, v, i)
            if m is not None:
                x = inpaint_blend(x, m, noised(s[i + 1], noise, init))
            mx.eval(x)
            if on_step is not None:
                on_step(i, x)
    return x


def _need_vae(pipe: ImagePipeline):
    if pipe.vae is None:
        raise ValueError("this pipeline has no VAE")
    return pipe.vae


def _vae_pass(plan, op: str, tiled: bool | None) -> tuple:
    """How a VAE ``op`` runs and the bytes it may take. With the ``plan``
    of the request both are the plan's, and ``tiled`` overrides its
    choice alone. Without one, or with a plan that had no budget to
    decide by, the choice is the automatic one against the free working
    set."""
    if plan is None:
        return tiled, None
    chosen, room = plan.vae_pass(op)
    if room is None:
        return tiled, None
    return (chosen if tiled is None else tiled), room


def vae_decode(pipe: ImagePipeline, latents, *, tiled: bool | None = None,
               alpha: bool = False, plan=None) -> mx.array:
    """Latents to an image float32 in [0, 1]: (B, H, W, 3), or (B, H, W, 4)
    with ``alpha``, which keeps the fourth channel of a VAE that decodes
    RGBA. ``plan`` is the memory plan of the request, from
    ``plan_request``: the decode runs whole or in tiles as it says, within
    the bytes it leaves for the pass."""
    vae = _need_vae(pipe)
    if alpha and vae.config.out_channels != 4:
        raise ValueError(f"the {pipe.family.title} VAE has no alpha channel")
    from .vae import tiling

    tiled, free = _vae_pass(plan, "decode", tiled)
    img = tiling.decode(vae, latents, tiled=tiled, alpha=alpha, free=free)
    return mx.clip(img / 2 + 0.5, 0.0, 1.0)


def vae_encode(pipe: ImagePipeline, image, *, tiled: bool | None = None,
               mode: str = "mode", noise=None, plan=None) -> mx.array:
    """An image (B, H, W, 3) in [0, 1] to latents (B, C, h, w) float32.
    ``plan`` is the memory plan of a request with an init image, and the
    encode runs as it says."""
    vae = _need_vae(pipe)
    from .vae import tiling

    tiled, free = _vae_pass(plan, "encode", tiled)
    x = mx.array(image).astype(mx.float32) * 2 - 1
    return tiling.encode(vae, x, tiled=tiled, mode=mode, noise=noise,
                         free=free)
