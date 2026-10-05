"""The image primitives over a loaded pipeline.

``load`` reads the DiT, the text encoder and the VAE into an
``ImagePipeline``, after gating their header sizes against the working
budget. ``encode_text`` turns a prompt into a ``Conditioning`` through the
family's encoder tap and keeps it in the pipeline's conditioning cache.

Latents are ``(B, C, H, W)`` float32 in model space. ``denoise_step``
evaluates the DiT once per row and returns one float32 velocity per row,
such that ``x_next = x + (sigma_next - sigma) * v``. It applies no
guidance and keeps no step counter. ``guide`` combines a positive and a
negative velocity. Rows run as sequential forwards. A DiT that splits its
step, as Z-Image does, gets the modulation of each sigma from a memo keyed
on the float32 bits of sigma and the patch table, and the caption state of
each conditioning from a cache keyed on its key and the patch table.

``sample`` runs the whole loop inside ``memfit.bounded_cache``, so the
buffers one step frees go back to the system instead of the MLX cache,
and it evaluates the latents after every step. It calls ``on_step(i, x)``
after each update and reads ``should_stop()`` before each step; a stop
raises ``SampleStopped`` with the latents reached.

``vae_decode`` and ``vae_encode`` map between latents and images in
``[0, 1]``, whole or in tiles as ``vae.tiling`` decides.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import memfit
from .conditioning import ByteLRU, Conditioning, array_bytes, conditioning_key
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
    weights: dict = field(default_factory=dict)  # dit, encoder, vae bytes
    load_seconds: dict = field(default_factory=dict)  # dit, encoder, vae
    log: Callable = print

    @property
    def vae_scale(self) -> int:
        """Image pixels per latent pixel, from the family row."""
        return int(self.family.constants.values["vae_scale_factor"])

    @property
    def defaults(self) -> SamplerDefaults:
        row = sampler_defaults(self.family.name, self.variant)
        if row is None:
            raise ValueError(
                f"no sampler defaults for {self.family.title} "
                f"{self.variant}; pass steps and cfg")
        return row

    def sigmas(self, steps: int) -> np.ndarray:
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

    skip = ("output.weight",) if kind == "encoder" else ()
    return memfit.file_bytes(_header_view(str(path)), skip=skip)


def load(dit, *, text_encoder=None, vae=None, mmproj=None,
         encoder_model=None, variant=None, memory: str = "auto",
         log=print) -> ImagePipeline:
    """Load an image pipeline from a DiT GGUF, a text encoder GGUF or a
    resident chat model given as ``encoder_model=(model, tokenizer)``, and
    a VAE. With ``memory="auto"`` the weights of all three are gated on
    their header sizes before anything loads; ``memory="off"`` skips that
    gate, and each load still gates its own weights."""
    from .dtypes import image_activation
    from .loader import identify_dit, load_dit
    from .text_encoder import encoder_spec, load_encoder
    from .vae import load_vae

    if memory not in ("auto", "off"):
        raise ValueError(f"memory must be 'auto' or 'off', not {memory!r}")
    if text_encoder is not None and encoder_model is not None:
        raise ValueError("pass text_encoder or encoder_model, not both")
    if mmproj is not None:
        raise ValueError("an mmproj serves image edits, which this gmlx "
                         "does not run yet")
    weights = {"dit": _header_bytes(dit, "dit"),
               "encoder": _header_bytes(text_encoder, "encoder"),
               "vae": _header_bytes(vae, "vae")}
    if memory == "auto":
        memfit.gate_load(sum(weights.values()), str(dit),
                         what="image model",
                         hint="Try smaller quants, or memory='off' to load "
                              "the parts one by one")
    seconds = {}
    t0 = time.perf_counter()
    act = image_activation(identify_dit(str(dit), variant=variant)
                           .family.name)
    model, info = load_dit(str(dit), variant=variant, act_dtype=act.dit,
                           log=log)
    fam = info.identified.family
    seconds["dit"] = time.perf_counter() - t0
    spec = encoder_spec(fam)
    handle = None
    t0 = time.perf_counter()
    if text_encoder is not None:
        handle = load_encoder(str(text_encoder), spec, log=log)
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


def _flat(params):
    from mlx.utils import tree_flatten

    return tree_flatten(params)


def plan_request(pipe: ImagePipeline, *, height: int, width: int,
                 cfg: float | None = None, caption_tokens: int | None = None,
                 budget: float | None = None,
                 overcommit: bool | None = None) -> memfit.MemoryPlan:
    """The memory plan of one request on this pipeline: its weights, an
    encoder still resident, the caches it holds, and the phase peaks.
    Raises ImageMemoryError when no lever makes it fit."""
    enc = pipe.encoder
    resident = enc is not None and not enc.released
    act = pipe.dit.act_dtype
    vae_dtype = getattr(pipe.vae, "dtype", None)
    caches = (pipe.conditionings.bytes + pipe.captions.bytes
              + pipe.modulations.bytes)
    return memfit.plan_memory(
        pipe.weights.get("dit", 0),
        pipe.weights.get("encoder", 0) if resident else 0,
        pipe.weights.get("vae", 0), family=pipe.family.name,
        height=height, width=width,
        cfg=pipe.defaults.cfg if cfg is None else cfg, dtype=act,
        vae_dtype=vae_dtype, encoder_owned=bool(resident and enc.owned),
        caption_tokens=caption_tokens, cache_bytes=caches, budget=budget,
        overcommit=overcommit)


def encode_text(pipe: ImagePipeline, prompt: str, *,
                images=()) -> Conditioning:
    """The conditioning of ``prompt``, from the cache when its key is
    there, else through the encoder tap."""
    if images:
        raise ValueError(f"{pipe.family.title} takes no reference images")
    enc = pipe.encoder
    if enc is None:
        raise ValueError("this pipeline has no text encoder")
    ids = enc.token_ids(prompt)
    key = conditioning_key(
        family=pipe.family.name, encoder=str(enc.source_key),
        template=enc.spec.template_id, ids=ids,
        tap_layers=enc.spec.tap_layers, dtype=enc.dtype_name)
    hit = pipe.conditionings.get(key)
    if hit is not None:
        return hit
    tap = enc.encode(prompt)
    cond = Conditioning(tap.hidden[0][0], key)
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
    w = getattr(mod, "weight", None)
    if not isinstance(w, mx.array) or w.ndim != 2:
        return None
    codec = getattr(mod, "kquant_type", None)
    if codec is None:
        return int(w.shape[0]), int(w.shape[1])
    from mlx_kquant.codec_geometry import in_features

    return int(w.shape[0]), in_features(codec, int(w.shape[1]))


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
        if len(band) != 3 or not all(isinstance(v, int)
                                     and not isinstance(v, bool)
                                     for v in band):
            raise ValueError(f"band of {path} must be three integers "
                             f"(dim, start, length), not {band!r}")
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
        out.append((h, st))
    return path, band, tuple(out)


def define_patch_table(pipe: ImagePipeline, entries) -> str:
    """Define an immutable patch table from entries of module path, band
    (dim, start, length) or None, and patches of (adapter hash,
    strength), and return its id: the sha256 of its canonical JSON.
    Equal entries give the same id in any order. The pipeline keeps the
    table under its id; no adapter is applied yet."""
    modules = dict(pipe.dit.named_modules())
    rows = [_table_entry(e, modules) for e in entries]
    keys = [(m, b) for m, b, _ in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("patch table names a module and band twice")
    tid, table = _freeze_table(tuple(rows))
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
    one value per row, under ``conds``, one Conditioning or one per row."""
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
    out = [_row(pipe, x[i:i + 1], sig[i], rows[i], table)
           for i in range(b)]
    return out[0] if b == 1 else mx.concatenate(out, axis=0)


def _row(pipe: ImagePipeline, x: mx.array, sigma: np.float32,
         cond: Conditioning, table) -> mx.array:
    dit = pipe.dit
    if not (hasattr(dit, "modulation") and hasattr(dit, "caption")):
        return dit(x, np.reshape(sigma, (1,)), cond.features[None])
    return dit(x, np.reshape(sigma, (1,)), None,
               mod=_modulation(pipe, sigma, table),
               caption=_caption(pipe, cond, table))


def _memo(cache: ByteLRU, key, make):
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
                 lambda: pipe.dit.modulation(np.reshape(s, (1,)), 1))


def _caption(pipe: ImagePipeline, cond: Conditioning, table):
    make = lambda: pipe.dit.caption(cond.features[None])  # noqa: E731
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
           patch_table=None, on_step=None, should_stop=None) -> mx.array:
    """Sample latents for ``cond``. With ``init_latents`` the run is
    img2img from strength ``denoise``, and a ``mask`` (1 where the image
    changes) makes it inpainting. ``noise`` is (B, C, H, W) float32, else
    it comes from ``seed``, else from a random seed. ``sampler`` is
    ``euler`` or ``uni_pc_bh2``, and the model sees the schedule's sigmas
    either way."""
    patch_table = _table_id(pipe, patch_table)
    if cfg is None:
        cfg = pipe.defaults.cfg
    if cfg != 1.0 and neg is None:
        raise ValueError(f"cfg {cfg} needs a negative conditioning")
    unit = pipe.vae_scale * pipe.dit.cfg.patch
    if height <= 0 or width <= 0 or height % unit or width % unit:
        raise ValueError(f"height and width must be positive multiples of "
                         f"{unit}, not {height}x{width}")
    if not 0.0 < denoise <= 1.0:
        raise ValueError(f"denoise must lie in (0, 1], not {denoise}")
    if init_latents is None and (denoise < 1.0 or mask is not None):
        raise ValueError("denoise below 1 and mask need init_latents")

    if sigmas is None:
        s = pipe.sigmas(pipe.defaults.steps if steps is None else steps)
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

    b = 1 if init is None else init.shape[0]
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
    with memfit.bounded_cache():
        for i in range(smp.steps):
            if should_stop is not None and should_stop():
                raise SampleStopped(i, x)
            v = denoise_step(pipe, x, s[i], cond, patch_table=patch_table)
            if cfg != 1.0:
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


def vae_decode(pipe: ImagePipeline, latents, *, tiled: bool | None = None,
               alpha: bool = False) -> mx.array:
    """Latents to an image (B, H, W, 3) float32 in [0, 1]."""
    vae = _need_vae(pipe)
    if alpha:
        raise ValueError(f"the {pipe.family.title} VAE has no alpha channel")
    from .vae import tiling

    img = tiling.decode(vae, latents, tiled=tiled)
    return mx.clip(img / 2 + 0.5, 0.0, 1.0)


def vae_encode(pipe: ImagePipeline, image, *, tiled: bool | None = None,
               mode: str = "mode", noise=None) -> mx.array:
    """An image (B, H, W, 3) in [0, 1] to latents (B, C, h, w) float32."""
    vae = _need_vae(pipe)
    from .vae import tiling

    x = mx.array(image).astype(mx.float32) * 2 - 1
    return tiling.encode(vae, x, tiled=tiled, mode=mode, noise=noise)
