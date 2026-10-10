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
larger of the encode phase and the denoise phase, not their sum. An
entry whose encoder does not fit beside its DiT is staged as the image
command stages a load: each encode that the cache does not hold lets
the DiT and the VAE go, loads the encoder alone, and loads them again
after its release. Its caches take the room its image leaves, up to
their bounds.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
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
    # Each encode runs with the DiT and the VAE out of memory: the owned
    # encoder loads, encodes and goes, and the DiT and the VAE load again.
    staged: bool = False
    cache_share: float = 1.0        # of each cache's configured bound
    encoder_bytes: int = 0          # an owned encoder's, from its header
    mmproj_bytes: int = 0
    load_parts: tuple | None = None   # see memfit.plan_memory, no tower
    tower_parts: tuple | None = None  # the same with the vision tower


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


# The bytes each cache of an image model may hold: conditionings and
# reference latents (tensor), request adapter pairs and built patch tables
# (adapter). server.image sets them; an entry books each at its bound.
# The step routes keep one cache of uploaded tensors and one of adapter
# matrices for the server (steps), which each image entry books too.
_CACHE_BYTES = {"tensor": 64 << 20, "adapter": 256 << 20,
                "steps": (64 + 256) << 20}


def configure_caches(settings) -> None:
    """Take the cache bounds of ``server.image``."""
    _CACHE_BYTES["tensor"] = int(settings.tensor_cache_mb) << 20
    _CACHE_BYTES["adapter"] = int(settings.adapter_cache_mb) << 20
    _CACHE_BYTES["steps"] = (_CACHE_BYTES["tensor"] + _CACHE_BYTES["adapter"]
                             if getattr(settings, "steps_api", True) else 0)


def _steps_cache_bytes() -> int:
    """What the step routes' caches hold now."""
    from .image_steps import STATE

    return STATE.tensors.bytes + STATE.adapters.bytes


def cache_bytes() -> tuple:
    """(tensor, adapter) bytes each cache of an image model may hold."""
    return _CACHE_BYTES["tensor"], _CACHE_BYTES["adapter"]


def bound_caches(pipe, configured: int = 0, share: float = 1.0) -> None:
    """Bound the caches of a served pipeline at ``share`` of their
    configured bounds. The adapter cache also holds the ``configured``
    bytes of the entry's own adapters, which the entry books on their
    own."""
    tensor, adapter = cache_bytes()
    tensor, adapter = int(tensor * share), int(adapter * share)
    pipe.conditionings.max_bytes = tensor
    pipe.reference_latents.max_bytes = tensor
    pipe.adapters.max_bytes = adapter + int(configured)
    pipe.tables.max_bytes = adapter + int(configured)


# The cache share of each loaded entry, by id of its ServedImage. The
# step routes' caches serve every entry and take the smallest share.
_SHARES: dict = {}


def _scale_step_caches() -> None:
    from .image_steps import STATE

    STATE.scale_caches(min(_SHARES.values(), default=1.0))


def _load_figures(spec, act) -> dict:
    """What the loads of an entry's parts hold over what stays resident,
    from their headers, as the image command counts them: the 16-bit
    cast of the DiT, and each part loaded with the others or alone."""
    from gmlx.commands.image_preflight import dit_transient, load_transient

    enc, vae, tower = spec.encoder, spec.vae, None
    if enc is not None:
        tower = spec.encoder_mmproj
    cast = dit_transient(str(spec.path), act.dit)
    later = max(load_transient(None, vae, None), cast)
    return {"peak": max(load_transient(enc, vae, tower), cast),
            "text_peak": max(load_transient(enc, vae, None), cast),
            "parts": (load_transient(enc, None, None), later),
            "tower_parts": (load_transient(enc, None, tower), later)}


def entry_plan(spec) -> EntryPlan:
    """The footprint of an image entry and whether it keeps its encoder,
    from the headers of its files. Raises for a file that is no image
    model, a family the config does not name, and an entry that cannot
    hold one image of its default size in the working budget."""
    from gmlx.image import memfit
    from gmlx.image.api import _header_bytes, check_steps, sampler_defaults
    from gmlx.image.dtypes import image_activation
    from gmlx.image.loader import identify_dit

    from .capacity import overcommit, working_budget_bytes

    from gmlx.image.fingerprint import ImageModelError

    try:
        ident = identify_dit(spec.path, variant=spec.variant)
    except ImageModelError as e:
        raise ImageModelError(f"image model {spec.id!r}: {e}") from None
    fam = ident.family
    if spec.family and not family_matches(spec.family, fam, ident.variant):
        raise ImageModelError(
            f"image model {spec.id!r}: {os.path.basename(spec.path)} is a "
            f"{fam.title} DiT, not the configured family {spec.family!r}")
    why = fam.variant_needed(ident.variant)
    if why:
        raise ImageModelError(f"image model {spec.id!r}: {why}")
    check_parts(spec, fam)
    steps = (spec.defaults or {}).get("steps")
    if steps is not None:
        try:
            check_steps(fam, ident.variant, steps)
        except ValueError as e:
            raise ImageModelError(
                f"image model {spec.id!r}: defaults.steps: {e}") from None
    defaults = sampler_defaults(fam.name, ident.variant)
    width, height = default_size(spec, defaults)
    cfg = (spec.defaults or {}).get("guidance")
    if cfg is None:
        cfg = defaults.cfg if defaults is not None else 1.0
    act = image_activation(fam.name)
    tensor_cache, adapter_cache = cache_bytes()
    own = [2 * os.path.getsize(p) for p, _ in spec.adapters]
    caches = 2 * tensor_cache + _CACHE_BYTES["steps"]
    dit = _header_bytes(spec.path, "dit")
    vae = _header_bytes(spec.vae, "vae")
    enc = mmproj = 0
    if spec.encoder is not None:
        enc = _header_bytes(spec.encoder, "encoder")
        mmproj = _header_bytes(spec.encoder_mmproj, "mmproj")
    load = _load_figures(spec, act)
    budget = working_budget_bytes()

    def common(share: float) -> dict:
        return dict(adapters=own + [int(2 * adapter_cache * share)],
                    family=fam.name, height=height, width=width, cfg=cfg,
                    dtype=act.dit, vae_dtype=act.vae, overcommit=True,
                    cache_bytes=int(caches * share))

    def kept(share: float):
        return memfit.plan_memory(dit, enc, vae, mmproj,
                                  encoder_owned=False,
                                  load_peak=load["peak"], **common(share))

    def price(mode: str, share: float) -> int:
        """The peak of an image at the default size, as serve runs it in
        ``mode``, with each cache at ``share`` of its bound."""
        if mode == "keep":
            return kept(share).peak
        if mode == "release":
            # The encoder loads for each encode beside the DiT and the VAE,
            # and is gone before the first step.
            keep = kept(share)
            alone = memfit.plan_memory(dit, 0, vae, **common(share))
            return max(keep.phases["encode"], keep.phases.get("load", 0),
                       alone.peak)
        # The encoder loads alone for each encode, without its vision
        # tower unless the request has reference images, and the DiT and
        # the VAE load again after it.
        return memfit.plan_memory(dit, enc, vae, 0, encoder_owned=True,
                                  load_peak=load["text_peak"],
                                  load_parts=load["parts"],
                                  **common(share)).peak

    def mode_of(release: str) -> str:
        if not enc or release == "never":
            return "keep"
        if release == "auto" and (budget is None
                                  or kept(1.0).peak <= _KEEP_SHARE * budget):
            return "keep"
        if budget is None or price("release", 1.0) <= budget:
            return "release"
        return "staged"

    def fitted(mode: str):
        """(footprint, cache share) of ``mode``, or None when it does not
        fit with its caches empty. Each cache takes the room the image
        leaves, up to its bound."""
        full = price(mode, 1.0)
        if budget is None or full <= budget:
            return full, 1.0
        bare = price(mode, 0.0)
        if bare > budget:
            return None
        steps = 64
        k = min(steps - 1, int(steps * (budget - bare) / max(1, full - bare)))
        while k > 0 and price(mode, k / steps) > budget:
            k -= 1
        return price(mode, k / steps), k / steps

    mode = mode_of(spec.encoder_release)
    got = fitted(mode)
    if got is None and overcommit():
        got = price(mode, 1.0), 1.0
    if got is None:
        need = price(mode, 0.0)
        hint = "use a smaller quant of the DiT or of the text encoder"
        if spec.encoder_release == "never" and enc and fitted(
                mode_of("auto")) is not None:
            hint = "set encoder_release: auto"
        raise memfit.ImageMemoryError(
            f"image model {spec.id!r} needs {need / memfit.GiB:.2f} GiB "
            f"for one {width}x{height} image, over the working budget of "
            f"{budget / memfit.GiB:.2f} GiB. Set a smaller defaults.size, or "
            f"{hint}. GMLX_OVERCOMMIT=1 overrides.")
    footprint, share = got
    return EntryPlan(footprint=int(footprint), keep_encoder=mode == "keep",
                     family=fam.name, variant=ident.variant,
                     size=(width, height), cfg=float(cfg),
                     staged=mode == "staged", cache_share=share,
                     encoder_bytes=int(enc), mmproj_bytes=int(mmproj),
                     load_parts=load["parts"],
                     tower_parts=load["tower_parts"])


def check_parts(spec, fam) -> None:
    """Refuse a VAE of another family, and an owned encoder or mmproj that
    is not the family's, from their headers. The pool prices an entry
    before it evicts anything, so a part that would fail at load is
    refused while the residents stay."""
    from gmlx.image.text_encoder import (
        _header_view,
        check_encoder_file,
        check_mmproj_file,
        encoder_spec,
    )
    from gmlx.image.vae import reader

    reader.identify(reader.read_header(str(spec.vae)), fam.name)
    if spec.encoder is not None:
        enc = encoder_spec(fam)
        check_encoder_file(_header_view(str(spec.encoder)), enc)
        if spec.encoder_mmproj is not None:
            check_mmproj_file(_header_view(str(spec.encoder_mmproj)), enc)


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
    # and loaded again for the next. A staged entry loads its encoder at
    # its first encode.
    owned = spec.encoder if not plan.staged else None
    pipe = api.load(spec.path, text_encoder=owned, vae=spec.vae,
                    mmproj=spec.encoder_mmproj if owned else None,
                    memory="off", variant=spec.variant, log=_log_line)
    served = ServedImage(pipe=pipe, spec=spec, plan=plan)
    try:
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
        bound_caches(pipe, sum(2 * os.path.getsize(p)
                               for p, _ in spec.adapters),
                     plan.cache_share)
        _SHARES[id(served)] = plan.cache_share
        _scale_step_caches()
    except BaseException:
        # The pipeline loaded and an adapter failed: free it here, as the
        # pool never holds it.
        del pipe
        _release([served])
        raise
    served.adapters = rows
    return served


_reservation = threading.local()


@contextlib.contextmanager
def reserved_worker():
    """Hold the image worker for the calling thread: wait here, before the
    caller takes any lock, for the jobs queued ahead to finish, then keep
    the worker idle until the block ends. ``load_on_worker`` and
    ``release_on_worker`` called inside the block run on it at once. So a
    pool build that loads or releases an image pipeline never waits on the
    worker's queue while it holds the pool's build lock. Nested blocks
    share the outer hold, and on the worker itself there is nothing to
    hold."""
    if getattr(_reservation, "run", None) is not None or \
            threading.current_thread().name.startswith("image-worker"):
        yield
        return
    import concurrent.futures
    import queue

    started = threading.Event()
    jobs: queue.SimpleQueue = queue.SimpleQueue()

    def hold():
        started.set()
        while True:
            item = jobs.get()
            if item is None:
                return
            fn, args, fut = item
            try:
                fut.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                fut.set_exception(e)

    outer = WORKER.submit(hold)
    started.wait()

    def run(fn, *args):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        jobs.put((fn, args, fut))
        return fut.result()

    _reservation.run = run
    try:
        yield
    finally:
        _reservation.run = None
        jobs.put(None)
        outer.result()


def _on_worker(fn, *args):
    run = getattr(_reservation, "run", None)
    if run is not None:
        return run(fn, *args)
    return WORKER.submit(fn, *args).result()


def load_on_worker(spec) -> ServedImage:
    """Load an image entry on the image worker and wait for it."""
    return _on_worker(_load, spec)


def _release(box: list) -> None:
    served = box.pop() if box else None
    if served is None:
        return
    if _SHARES.pop(id(served), None) is not None:
        _scale_step_caches()
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
    _on_worker(_release, box)


# Requests: the OpenAI image fields mapped onto the image primitives.

OUTPUT_FORMATS = ("png", "jpeg", "webp")
QUALITIES = ("low", "medium", "high", "auto")
_QUALITY_ALIASES = {"standard": "medium", "hd": "high"}
BACKGROUNDS = ("transparent", "opaque", "auto")
RESPONSE_FORMATS = ("b64_json", "url")
MAX_N = 10
MAX_PARTIALS = 3
MAX_REFS = 16
# The largest image a request makes: its pixels and its longer side.
MAX_PIXELS = 2048 * 2048
MAX_SIDE = 4096
# Model names an OpenAI client sends that mean the default image model.
_ALIAS_PREFIXES = ("gpt-image-", "dall-e-", "chatgpt-image")
# Accepted and ignored, with one warning line.
IGNORED_FIELDS = frozenset({"moderation", "style", "input_fidelity", "user"})
# Fields that name a file to write; refused.
REFUSED_FIELDS = ("output_path", "output_dir")
# The instruction of a variation on a family that edits by reference.
VARIATION_PROMPT = "Make a variation of this image."
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


class ImageRequestError(Exception):
    """A request the image routes refuse, with its HTTP status and the
    OpenAI error fields."""

    def __init__(self, status: int, message: str, *, param: str | None = None,
                 code: str | None = None,
                 err_type: str = "invalid_request_error",
                 headers: dict | None = None):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        self.err_type = err_type
        self.headers = headers or {}


def _bad(message: str, param: str | None = None, **kw) -> ImageRequestError:
    return ImageRequestError(400, message, param=param, **kw)


@dataclass
class ImageRequest:
    """One parsed request of the generations, edits or variations route."""
    route: str
    model: str = ""
    prompt: str = ""
    n: int = 1
    size: tuple | None = None          # (width, height), None for auto
    quality: str = "auto"
    steps: int | None = None
    seed: int | None = None
    guidance: float | None = None
    sampler: str | None = None
    negative_prompt: str | None = None
    strength: float | None = None
    background: str = "auto"
    output_format: str = "png"
    compression: int | None = None
    response_format: str = "b64_json"
    stream: bool = False
    partial_images: int = 0
    adapters: list = field(default_factory=list)   # (id, strength)
    images: list = field(default_factory=list)     # uint8 (H, W, 3 or 4)
    mask: Any = None                               # uint8 (H, W, 4)
    ignored: set = field(default_factory=set)


def default_model_field(model) -> str:
    """The model field with the OpenAI model names that mean the default
    image model turned into an empty field."""
    name = str(model or "").strip()
    if name in ("", "default") or name.startswith(_ALIAS_PREFIXES):
        return ""
    return name


def _int(fields: dict, name: str, lo: int, hi: int) -> int | None:
    v = fields.get(name)
    if v is None or v == "":
        return None
    try:
        if isinstance(v, bool) or (isinstance(v, float) and not v.is_integer()):
            raise ValueError
        i = int(v)
    except (TypeError, ValueError):
        raise _bad(f"{name} must be an integer", name) from None
    if not lo <= i <= hi:
        raise _bad(f"{name} must be from {lo} to {hi}, not {i}", name)
    return i


def _float(fields: dict, name: str, lo: float, hi: float) -> float | None:
    v = fields.get(name)
    if v is None or v == "":
        return None
    try:
        if isinstance(v, bool):
            raise ValueError
        x = float(v)
    except (TypeError, ValueError):
        raise _bad(f"{name} must be a number", name) from None
    if not (lo <= x <= hi):
        raise _bad(f"{name} must be from {lo:g} to {hi:g}, not {x:g}", name)
    return x


def _bool(fields: dict, name: str) -> bool:
    v = fields.get(name)
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    text = str(v).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise _bad(f"{name} must be true or false", name)


def _choice(fields: dict, name: str, allowed, default: str) -> str:
    v = fields.get(name)
    if v is None or v == "":
        return default
    text = str(v).strip().lower()
    if text not in allowed:
        raise _bad(f"{name} must be one of {', '.join(allowed)}, not {v!r}",
                   name)
    return text


def parse_size(value) -> tuple | None:
    """``WxH`` as (width, height), or None for ``auto`` and no value."""
    if value is None or value == "" or str(value).strip().lower() == "auto":
        return None
    w, sep, h = str(value).strip().lower().partition("x")
    try:
        if not sep:
            raise ValueError
        size = int(w), int(h)
    except ValueError:
        raise _bad(f"size must be WxH or auto, not {value!r}", "size") from None
    if min(size) <= 0:
        raise _bad(f"size {value!r} is not a positive width and height", "size")
    if size[0] * size[1] > MAX_PIXELS or max(size) > MAX_SIDE:
        raise _bad(f"size {size[0]}x{size[1]} is over the limit of "
                   f"{MAX_PIXELS} pixels (2048x2048) and a side of "
                   f"{MAX_SIDE}", "size")
    return size


def _adapters(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        import json

        try:
            value = json.loads(value)
        except ValueError:
            raise _bad("adapters must be a list of {id, strength}",
                       "adapters") from None
    if not isinstance(value, list):
        raise _bad("adapters must be a list of {id, strength}", "adapters")
    rows = []
    for item in value:
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not item["id"].strip():
            raise _bad("each adapter is {id, strength}, with id a string",
                       "adapters")
        unknown = set(item) - {"id", "strength"}
        if unknown:
            raise _bad(f"adapter fields {sorted(unknown)} are unknown; an "
                       "adapter is {id, strength}", "adapters")
        st = _float(item, "strength", -100.0, 100.0)
        rows.append((item["id"].strip(), 1.0 if st is None else st))
    return rows


def parse_request(route: str, fields: dict) -> ImageRequest:
    """An :class:`ImageRequest` from the fields of a JSON body or of a
    multipart form, whose values are strings. Images are set by the
    caller. Raises :class:`ImageRequestError` for a field it refuses."""
    for name in REFUSED_FIELDS:
        if fields.get(name) is not None:
            raise _bad(f"{name} is not accepted by this server; use "
                       "response_format b64_json or url", name)
    rf = fields.get("response_format")
    if rf == "path":
        raise _bad("response_format 'path' is not accepted by this server; "
                   "use b64_json or url", "response_format")
    prompt = fields.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise _bad("prompt must be a string", "prompt")
    prompt = (prompt or "").strip()
    if route != "variations" and not prompt:
        raise _bad("prompt is required", "prompt")
    neg = fields.get("negative_prompt")
    if neg is not None and not isinstance(neg, str):
        raise _bad("negative_prompt must be a string", "negative_prompt")
    sampler = fields.get("sampler")
    if sampler not in (None, ""):
        from gmlx.config import IMAGE_SAMPLERS

        if sampler not in IMAGE_SAMPLERS:
            raise _bad(f"sampler must be one of {', '.join(IMAGE_SAMPLERS)}, "
                       f"not {sampler!r}", "sampler")
    else:
        sampler = None
    scheduler = fields.get("scheduler")
    if scheduler not in (None, "", "default"):
        raise _bad("scheduler takes only default, the variant's own "
                   "schedule", "scheduler")
    q = fields.get("quality")
    q = _QUALITY_ALIASES.get(str(q).strip().lower(), q) if q else q
    req = ImageRequest(
        route=route,
        model=default_model_field(fields.get("model")),
        prompt=prompt,
        n=_int(fields, "n", 1, MAX_N) or 1,
        size=parse_size(fields.get("size")),
        quality=_choice({"quality": q}, "quality", QUALITIES, "auto"),
        steps=_int(fields, "steps", 1, 200),
        seed=_int(fields, "seed", 0, (1 << 63) - 1),
        guidance=_float(fields, "guidance", 1.0, 30.0),
        sampler=sampler,
        negative_prompt=neg,
        strength=_float(fields, "strength", 0.01, 1.0),
        background=_choice(fields, "background", BACKGROUNDS, "auto"),
        output_format=_choice(fields, "output_format", OUTPUT_FORMATS, "png"),
        compression=_int(fields, "output_compression", 0, 100),
        response_format=_choice(fields, "response_format", RESPONSE_FORMATS,
                                "b64_json"),
        stream=_bool(fields, "stream"),
        partial_images=_int(fields, "partial_images", 0, MAX_PARTIALS) or 0,
        adapters=_adapters(fields.get("adapters")),
    )
    if req.stream and req.n > 1:
        raise _bad("stream with n above 1 is not supported; the stream "
                   "events carry no image index", "n")
    if req.background == "transparent" and req.output_format == "jpeg":
        raise _bad("background transparent needs png or webp; jpeg holds no "
                   "alpha channel", "background")
    req.ignored = {k for k in fields if k in IGNORED_FIELDS
                   and fields[k] not in (None, "")}
    if req.compression is not None and req.output_format == "png":
        req.ignored.add("output_compression")
    return req


def decode_image(data: bytes, param: str):
    """Image bytes as uint8 (H, W, 3), or (H, W, 4) for an image with
    alpha, upright by its EXIF orientation. PNG, JPEG, WebP and GIF only,
    within the media pixel limit."""
    import io

    import numpy as np
    from PIL import Image, ImageOps

    from .patches.media_gate import MEDIA_MAX_PIXELS

    try:
        with Image.open(io.BytesIO(data),
                        formats=["PNG", "JPEG", "WEBP", "GIF"]) as im:
            if im.width * im.height > MEDIA_MAX_PIXELS:
                raise _bad(f"{param} is {im.width}x{im.height} pixels, over "
                           f"the limit of {MEDIA_MAX_PIXELS} pixels", param)
            im.load()
            im = ImageOps.exif_transpose(im)
            alpha = im.mode in ("RGBA", "LA", "PA") or (
                im.mode == "P" and "transparency" in im.info)
            im = im.convert("RGBA" if alpha else "RGB")
            return np.asarray(im, np.uint8).copy()
    except ImageRequestError:
        raise
    except Image.DecompressionBombError:
        raise _bad(f"{param} has too many pixels", param) from None
    except Exception:  # noqa: BLE001 - Pillow raises many types for bad data
        raise _bad(f"{param} is not a readable PNG, JPEG, WebP or GIF "
                   "image", param) from None


def resized(pixels, size: tuple, *, resample: str = "lanczos"):
    """uint8 (H, W, C) pixels at ``size`` (width, height)."""
    import numpy as np
    from PIL import Image

    if (pixels.shape[1], pixels.shape[0]) == tuple(size):
        return pixels
    mode = "RGBA" if pixels.shape[-1] == 4 else "RGB"
    how = (Image.Resampling.LANCZOS if resample == "lanczos"
           else Image.Resampling.BILINEAR)
    return np.asarray(Image.fromarray(pixels, mode).resize(tuple(size), how))


def mask_from_image(pixels, size: tuple):
    """An edit mask as (H, W) float32, 1 where the image changes: the
    pixels of the mask image with alpha below 128."""
    import numpy as np

    if pixels.shape[-1] != 4:
        raise _bad("mask must be an image with an alpha channel; its pixels "
                   "with alpha below 128 mark the region to edit", "mask")
    a = resized(pixels, size, resample="bilinear")[..., 3]
    return (a < 128).astype(np.float32)


def encode_output(pixels, fmt: str, compression: int | None = None,
                  metadata: dict | None = None) -> bytes:
    """uint8 (H, W, 3 or 4) pixels as png, jpeg or webp bytes.
    ``compression`` 0 to 100 is the jpeg and webp quality, as OpenAI's
    ``output_compression`` is, and None is 100."""
    import io
    import json

    from PIL import Image, PngImagePlugin

    im = Image.fromarray(pixels, "RGBA" if pixels.shape[-1] == 4 else "RGB")
    buf = io.BytesIO()
    if fmt == "png":
        info = None
        if metadata is not None:
            info = PngImagePlugin.PngInfo()
            info.add_text("gmlx", json.dumps(metadata, sort_keys=True))
        im.save(buf, format="PNG", pnginfo=info)
    else:
        quality = 100 if compression is None else int(compression)
        quality = max(1, min(100, quality))
        if fmt == "jpeg":
            im.convert("RGB").save(buf, format="JPEG", quality=quality)
        else:
            im.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def mime_type(fmt: str) -> str:
    return _MIME[fmt]


# URL responses: files under a server folder, named by a 32 byte token.

class FileStore:
    """Image files for ``response_format: url``. Each file is named by a
    random 32 byte token, which is the capability to read it. Files expire
    after ``ttl_s`` seconds, and past ``max_bytes`` the oldest go first."""

    TOKEN_CHARS = 43                    # 32 bytes, url-safe base64

    def __init__(self, root: str, ttl_s: float, max_bytes: int,
                 time_fn=None):
        import threading
        import time

        self.root = root
        self.ttl_s = float(ttl_s)
        self.max_bytes = int(max_bytes)
        self._time = time_fn or time.time
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        os.makedirs(self.root, mode=0o700, exist_ok=True)

    @classmethod
    def valid_name(cls, token: str, ext: str) -> bool:
        import re

        return (ext in OUTPUT_FORMATS and len(token) == cls.TOKEN_CHARS
                and re.fullmatch(r"[A-Za-z0-9_-]+", token) is not None)

    def put(self, data: bytes, ext: str) -> str:
        """Write ``data`` and return its token."""
        import secrets

        self._ensure()
        token = secrets.token_urlsafe(32)
        path = os.path.join(self.root, f"{token}.{ext}")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        # The new file stays even when it alone is over the byte cap: its
        # link is the answer of the request that wrote it.
        self.sweep(keep=path)
        return token

    def path(self, token: str, ext: str) -> str | None:
        """The file of ``token``, or None when it is unknown or expired."""
        if not self.valid_name(token, ext):
            return None
        path = os.path.join(self.root, f"{token}.{ext}")
        try:
            st = os.stat(path)
        except OSError:
            return None
        if self._time() - st.st_mtime > self.ttl_s:
            return None
        return path

    def sweep(self, keep: str | None = None) -> int:
        """Remove expired files, then the oldest past the byte cap, never
        the file ``keep``. Returns how many it removed."""
        with self._lock:
            try:
                names = os.listdir(self.root)
            except OSError:
                return 0
            now = self._time()
            files = []
            for name in names:
                p = os.path.join(self.root, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
            files.sort()
            removed = 0
            total = sum(size for _, size, _ in files)
            for mtime, size, p in files:
                if p == keep:
                    continue
                if now - mtime > self.ttl_s or total > self.max_bytes:
                    try:
                        os.unlink(p)
                    except OSError:
                        continue
                    removed += 1
                    total -= size
            return removed


def default_store_root(port=None) -> str:
    """The URL file folder of the server on ``port``: each server keeps its
    own, so one server's sweep never takes another's files."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    root = os.path.join(base, "gmlx", "images")
    return root if port is None else os.path.join(root, str(int(port)))


# The job: encode, plan, sample, decode, on the image worker.

@dataclass
class ImageJobResult:
    images: list                    # encoded bytes per image
    size: tuple                     # (width, height)
    quality: str
    usage: dict
    seeds: list
    steps: int
    run_steps: int = 0              # denoise steps per image
    image_s: float = 0.0            # sampling and decode, all images
    step_s: float = 0.0             # sampling alone, all images


class JobStopped(Exception):
    """The request's stop event or its deadline ended the job."""

    def __init__(self, timed_out: bool):
        super().__init__("timed out" if timed_out else "cancelled")
        self.timed_out = timed_out


def quality_steps(family, variant: str, quality: str) -> int | None:
    """The steps of a quality level from the family table, or None for
    auto and a variant without a table."""
    row = family.sampling_row(variant)
    if row is None or quality == "auto":
        return None
    return dict(row.quality).get(quality)


def _stride(pipe) -> int:
    return int(pipe.size_multiple)


def target_size(req: ImageRequest, served: ServedImage) -> tuple:
    """(width, height) of the request: its size, else the first image's
    for an edit or a variation, else the entry's default."""
    pipe = served.pipe
    unit = _stride(pipe)
    if req.size is not None:
        w, h = req.size
        if w % unit or h % unit:
            raise _bad(f"size {w}x{h}: width and height must be multiples "
                       f"of {unit}", "size")
        return w, h
    if req.images:
        h, w = req.images[0].shape[:2]
        # A large image gives the largest size of its shape in the limits.
        scale = min(1.0, (MAX_PIXELS / (w * h)) ** 0.5, MAX_SIDE / max(w, h))
        w, h = int(w * scale) // unit * unit, int(h * scale) // unit * unit
        if min(w, h) <= 0:
            raise _bad(f"image is smaller than {unit} pixels", "image")
        return w, h
    return tuple(served.plan.size)


def _guided(req: ImageRequest, served: ServedImage) -> float:
    if req.guidance is not None:
        return float(req.guidance)
    return float(served.plan.cfg)


def check_family(req: ImageRequest, served: ServedImage) -> None:
    """Refuse what the family cannot do: transparency without an alpha
    VAE, a step count other than the one a variant with a fixed schedule
    runs, more images than an edit takes, and edits without a vision
    tower."""
    from gmlx.image.api import fixed_steps

    pipe = served.pipe
    fam = pipe.family
    n = fixed_steps(fam, pipe.variant)
    if n is not None and req.steps is not None and req.steps != n:
        raise _bad(f"steps: {fam.title} {pipe.variant} runs {n} steps, the "
                   f"schedule it was distilled to, not {req.steps}", "steps")
    if req.background == "transparent":
        from gmlx.image.vae.reader import VAES

        row = VAES.get(fam.name)
        if row is None or row.image_channels != 4:
            raise _bad(f"background transparent: {fam.title} decodes no alpha "
                       "channel", "background")
    if req.route == "generations":
        return
    if not req.images:
        raise _bad("image is required", "image")
    edit = fam.edit
    if edit is None:
        if len(req.images) > 1:
            raise _bad(f"{fam.title} edits one image at a time, and the "
                       f"request has {len(req.images)}", "image")
        return
    cap = min(edit.max_refs, MAX_REFS)
    if len(req.images) > cap:
        raise _bad(f"{fam.title} takes at most {cap} images, and the request "
                   f"has {len(req.images)}", "image")
    if served.spec.encoder_model is None and not served.spec.encoder_mmproj:
        raise _bad(f"edits on {served.spec.id!r} need the encoder's vision "
                   "tower: set encoder_mmproj on the image model", "image")


def _fit_args(pipe, served: ServedImage, budget, cfg: float,
              tower: bool = False) -> dict:
    """The plan_memory arguments of a request on ``served``. A staged
    entry's request loads its encoder alone, with the vision tower when
    ``tower`` says it has reference images."""
    enc = pipe.encoder
    resident = enc is not None and not enc.released
    act = pipe.dit.act_dtype
    staged = {}
    if getattr(served.plan, "staged", False):
        plan = served.plan
        staged = dict(
            encoder=plan.encoder_bytes,
            mmproj=plan.mmproj_bytes if tower else 0, encoder_owned=True,
            load_parts=plan.tower_parts if tower else plan.load_parts)
    return dict(
        dit=pipe.weights.get("dit", 0),
        encoder=pipe.weights.get("encoder", 0) if resident else 0,
        vae=pipe.weights.get("vae", 0),
        mmproj=pipe.weights.get("mmproj", 0) if resident else 0,
        adapters=pipe.adapters.bytes + pipe.tables.bytes,
        family=pipe.family.name, cfg=cfg, dtype=act,
        vae_dtype=getattr(pipe.vae, "dtype", None),
        encoder_owned=bool(resident and enc.owned
                           and not served.plan.keep_encoder),
        cache_bytes=(pipe.conditionings.bytes + pipe.modulations.bytes
                     + _steps_cache_bytes()),
        budget=budget, overcommit=False) | staged


def fit_check(req: ImageRequest, served: ServedImage, size: tuple,
              budget) -> None:
    """Refuse a request whose memory plan does not fit ``budget`` with a
    400 that names the largest size of its aspect that fits."""
    from gmlx.image import memfit

    pipe = served.pipe
    cfg = _guided(req, served)
    refs = [] if pipe.family.edit is None else [
        r.shape[:2] for r in req.images]
    args = _fit_args(pipe, served, budget, cfg, tower=bool(refs))
    plan_memory = memfit.plan_memory
    init = bool(req.images) and (pipe.family.edit is None
                                 or req.mask is not None)

    def fits(w: int, h: int, ref_sizes) -> bool:
        try:
            return plan_memory(height=h, width=w, init_image=init,
                               ref_sizes=ref_sizes, _hint=False,
                               **args).fits
        except memfit.ImageMemoryError:
            return False

    w, h = size
    if budget is None or fits(w, h, refs):
        return
    if refs:
        # The plan's own message names the references that fit.
        try:
            plan_memory(height=h, width=w, init_image=init, ref_sizes=refs,
                        **args)
        except memfit.ImageMemoryError as e:
            raise _bad(str(e), "image") from None
    unit = _stride(pipe)

    def scaled(k: int) -> tuple:
        scale = k * unit / max(w, h)
        return (max(unit, int(w * scale) // unit * unit),
                max(unit, int(h * scale) // unit * unit))

    # A smaller size of the shape never needs more memory, so a bisection
    # over the longer side finds the largest that fits in a few plans.
    best = None
    lo, hi = 1, max(w, h) // unit - 1
    while lo <= hi:
        k = (lo + hi) // 2
        if fits(*scaled(k), refs):
            best, lo = scaled(k), k + 1
        else:
            hi = k - 1
    if best is None:
        raise _bad(f"size {w}x{h} does not fit in memory beside the models "
                   "now loaded, and no smaller size of its shape does",
                   "size")
    raise _bad(f"size {w}x{h} does not fit in memory beside the models now "
               f"loaded; the largest size of its shape that fits is "
               f"{best[0]}x{best[1]}", "size")


def size_fits(served: ServedImage, width: int, height: int, budget, *,
              cfg: float = 1.0, init: bool = False) -> bool:
    """Whether the memory plan of an image of ``width`` x ``height`` on
    ``served`` fits ``budget``, as the image routes plan a request. The
    step routes ask it before a step, a VAE decode and a VAE encode."""
    from gmlx.image import memfit

    if budget is None:
        return True
    args = _fit_args(served.pipe, served, budget, cfg)
    try:
        return memfit.plan_memory(height=int(height), width=int(width),
                                  init_image=init, _hint=False, **args).fits
    except memfit.ImageMemoryError:
        return False


def _ensure_encoder(served: ServedImage, tower: bool = True) -> None:
    """Load an owned encoder again after its release, with the text
    halves of the configured adapters, and its vision tower when
    ``tower`` says so."""
    pipe = served.pipe
    enc = pipe.encoder
    if served.spec.encoder is None or (enc is not None and not enc.released):
        return
    from gmlx.image import lora
    from gmlx.image.text_encoder import encoder_spec, load_encoder

    pipe.encoder = load_encoder(
        served.spec.encoder, encoder_spec(pipe.family), log=_log_line,
        mmproj=served.spec.encoder_mmproj if tower else None)
    if served.adapters:
        lora.apply_text_adapters(pipe, served.adapters)


def _drop_dit(served: ServedImage) -> None:
    """Let the DiT and the VAE of a staged entry go before its encoder
    loads."""
    import gc

    import mlx.core as mx

    from gmlx.image.lora import install

    pipe = served.pipe
    install.drop_built(pipe)
    pipe.dit = pipe.vae = None
    gc.collect()
    mx.clear_cache()


def _ensure_dit(served: ServedImage) -> None:
    """Load the DiT and the VAE of a staged entry again, with the DiT's
    leaves wrapped for the patch tables the pipeline holds."""
    pipe = served.pipe
    if pipe.dit is not None and pipe.vae is not None:
        return
    from gmlx.image.dtypes import image_activation
    from gmlx.image.loader import load_dit
    from gmlx.image.lora import install
    from gmlx.image.vae import load_vae

    act = image_activation(pipe.family.name)
    if pipe.dit is None:
        pipe.dit, _ = load_dit(str(served.spec.path), act_dtype=act.dit,
                               variant=served.spec.variant, log=_log_line)
        install.rewrap(pipe)
    if pipe.vae is None:
        pipe.vae = load_vae(str(served.spec.vae), family=pipe.family.name,
                            dtype=act.vae)


def _staged_encode(served: ServedImage, run, prompts: list, refs):
    """``run()`` on a staged entry. A request whose conditionings are all
    in the cache runs on the released encoder's tokenizer. Any other
    lets the DiT and the VAE go, loads the encoder, encodes, releases it
    and loads the DiT and the VAE again, so the encoder is never resident
    beside them."""
    from gmlx.image import api

    pipe = served.pipe
    if all(api.conditioning_cached(pipe, p, images=refs) for p in prompts):
        _ensure_dit(served)
        return run()
    _drop_dit(served)
    try:
        _ensure_encoder(served, tower=bool(refs))
        try:
            return run()
        finally:
            pipe.encoder.release()
    finally:
        _ensure_dit(served)


def _encode(served: ServedImage, prompt: str, neg: str | None, refs,
            borrow=None) -> tuple:
    """The positive conditioning and the negative one, or None, without
    the VAE latents of reference images. ``borrow`` runs ``fn(handle)``
    with the borrowed encoder of a shared chat model on its engine."""
    from gmlx.image import api

    pipe = served.pipe

    def run():
        cond = api.encode_text(pipe, prompt, images=refs, ref_latents=False)
        other = None if neg is None else api.encode_text(
            pipe, neg, images=refs, ref_latents=False)
        import mlx.core as mx

        mx.eval([c.features for c in (cond, other) if c is not None])
        return cond, other

    if served.spec.encoder_model is not None:
        if borrow is None:
            raise _bad(f"image model {served.spec.id!r} shares the encoder of "
                       f"{served.spec.encoder_model!r}, which is not loaded",
                       "model", err_type="server_error")
        return borrow(run)
    if getattr(served.plan, "staged", False):
        return _staged_encode(served, run,
                              [prompt] + ([] if neg is None else [neg]), refs)
    _ensure_encoder(served)
    return run()


def _adapter_table(served: ServedImage, rows: list, model_dirs) -> tuple:
    """The patch table of the configured adapters with the request's,
    each found by id: the file name of a configured adapter, or a path
    relative to a model folder."""
    from gmlx.image import api, lora

    pipe = served.pipe
    if not rows:
        return None, []
    named = {}
    for ad, _st in served.adapters:
        named[os.path.basename(ad.path)] = ad.path
        named[os.path.splitext(os.path.basename(ad.path))[0]] = ad.path
    table_rows = list(served.adapters)
    # A request that names a configured adapter sets its strength on the
    # DiT for the request, in place of the entry's, and never adds it twice.
    configured = {os.path.realpath(ad.path): i
                  for i, (ad, _st) in enumerate(served.adapters)}
    shown = []
    for aid, strength in rows:
        path = named.get(aid)
        if path is None:
            path = _adapter_under_roots(aid, model_dirs)
        at = configured.get(os.path.realpath(path))
        if at is not None:
            table_rows[at] = (table_rows[at][0], strength)
            shown.append({"id": aid, "strength": strength})
            continue
        ad = lora.load_adapter(pipe, path)
        if ad.text:
            raise _bad(f"adapter {aid!r} has text encoder layers; set it on "
                       "the image model's adapters instead", "adapters")
        table_rows.append((ad, strength))
        shown.append({"id": aid, "strength": strength})
    return api.define_patch_table(pipe, lora.table_entries(table_rows)), shown


def _adapter_under_roots(aid: str, model_dirs) -> str:
    if os.path.isabs(aid) or aid.startswith("~") or ".." in aid.split("/") \
            or "\\" in aid:
        raise _bad(f"adapter {aid!r}: an id is a configured adapter or a path "
                   "inside a model folder, never an absolute path", "adapters")
    for root in model_dirs or ():
        root = os.path.realpath(os.path.expanduser(root))
        cand = os.path.realpath(os.path.join(root, aid))
        if cand.startswith(root + os.sep) and os.path.isfile(cand):
            return cand
    raise ImageRequestError(404, f"adapter {aid!r} is not a configured "
                            "adapter or a file in a model folder",
                            param="adapters", code="adapter_not_found")


def partial_steps(steps: int, count: int) -> list:
    """The step indexes after which a stream sends its partial images:
    evenly spaced, after the first step and before the last."""
    if count <= 0 or steps < 3:
        return []
    picks = sorted({min(steps - 2, max(1, round(k * steps / (count + 1)) - 1))
                    for k in range(1, count + 1)})
    return picks[:count]


def run_job(served: ServedImage, req: ImageRequest, *, size: tuple,
            budget, stop, timeout_s: float | None = None,
            submitted: float | None = None, borrow=None, emit=None,
            model_dirs=(), embed_metadata: bool = False) -> ImageJobResult:
    """Run one request on the image worker's thread. ``stop`` is a
    threading.Event a disconnect sets; ``emit(index, bytes)`` takes the
    partial images of a stream. Raises :class:`JobStopped` when ``stop``
    ends the job between steps, or when it waits ``timeout_s`` for its
    turn after ``submitted`` or for one step."""
    import secrets
    import time

    last = [time.monotonic() if submitted is None else submitted]
    if stop.is_set():
        raise JobStopped(timed_out=False)
    if timeout_s and time.monotonic() - last[0] > timeout_s:
        raise JobStopped(timed_out=True)
    last[0] = time.monotonic()

    import mlx.core as mx
    import numpy as np

    from gmlx.image import api
    from gmlx.image.samplers import img2img_start

    pipe = served.pipe
    fam = pipe.family
    spec_defaults = served.spec.defaults or {}
    width, height = size
    cfg = _guided(req, served)
    steps = (req.steps or quality_steps(fam, pipe.variant, req.quality)
             or spec_defaults.get("steps") or pipe.defaults.steps)
    sampler = req.sampler or spec_defaults.get("sampler") or "euler"
    neg_text = None
    if cfg != 1.0:
        neg_text = req.negative_prompt
        if neg_text is None:
            neg_text = spec_defaults.get("negative_prompt") or ""
    alpha = req.background == "transparent"

    prompt = req.prompt
    refs: tuple = ()
    init = mask = None
    strength = None
    if req.route != "generations":
        first = req.images[0]
        if fam.edit is not None:
            refs = api.references(pipe, [i[..., :3] for i in req.images])
            if req.route == "variations" and not prompt:
                prompt = VARIATION_PROMPT
        if fam.edit is None or req.mask is not None:
            init = resized(first[..., :3], (width, height))
            strength = req.strength
            if strength is None:
                strength = 1.0 if req.mask is not None else (
                    pipe.defaults.strength)
        if req.mask is not None:
            mask = mask_from_image(req.mask, (width, height))

    table, adapters = _adapter_table(served, req.adapters, model_dirs)
    t0 = time.perf_counter()
    cond, neg = _encode(served, prompt, neg_text, refs, borrow)
    plan = api.plan_request(
        pipe, height=height, width=width, cfg=cfg,
        conds=[cond] if neg is None else [cond, neg], patch_table=table,
        init_image=init is not None,
        keep_encoder=served.plan.keep_encoder, budget=budget,
        overcommit=budget is None)
    enc = pipe.encoder
    if enc is not None and enc.owned and not enc.released and (
            not served.plan.keep_encoder or plan.release_encoder):
        enc.release()
    init_latents = None
    if init is not None:
        img = np.asarray(init, np.float32)[None] / 255.0
        init_latents = api.vae_encode(pipe, img, plan=plan)
    if refs:
        cond = api.encode_references(pipe, cond, plan=plan)
        if neg is not None:
            neg = api.encode_references(pipe, neg, plan=plan)
    encode_s = time.perf_counter() - t0

    def should_stop() -> bool:
        return stop.is_set() or bool(
            timeout_s and time.monotonic() - last[0] > timeout_s)

    denoise = 1.0 if strength is None else float(strength)
    token_px = pipe.vae_scale * pipe.dit.cfg.patch
    sigmas = pipe.sigmas(steps, (height // token_px) * (width // token_px))
    if init_latents is not None:
        sigmas = sigmas[img2img_start(len(sigmas) - 1, denoise):]
    run_steps = len(sigmas) - 1
    picks = set(partial_steps(run_steps, req.partial_images)) \
        if emit is not None else set()
    seed = req.seed if req.seed is not None else secrets.randbelow(1 << 32)
    seeds = [seed + i for i in range(req.n)]
    out = []
    image_s = step_s = 0.0
    meta = None
    if embed_metadata and req.output_format == "png":
        meta = {"prompt": prompt, "steps": steps, "size": f"{width}x{height}",
                "sampler": sampler, "guidance": cfg,
                "model": os.path.basename(served.spec.path),
                "adapters": adapters}

    def finish(x) -> np.ndarray:
        from gmlx.commands.image import to_uint8

        pixels = to_uint8(api.vae_decode(pipe, x, alpha=alpha, plan=plan))
        if mask is not None:
            keep = resized(req.images[0], (width, height))
            if keep.shape[-1] != pixels.shape[-1]:
                keep = (keep[..., :3] if pixels.shape[-1] == 3 else
                        np.concatenate([keep, np.full(keep.shape[:2] + (1,),
                                                      255, np.uint8)], -1))
            m = mask[..., None]
            pixels = np.round(m * pixels + (1 - m) * keep).astype(np.uint8)
        return pixels

    for s in seeds:
        prev = {}
        partial_index = [0]

        def on_step(i, x, _prev=prev, _pi=partial_index):
            last[0] = time.monotonic()
            if i in picks and "x" in _prev:
                v = (x - _prev["x"]) / float(sigmas[i + 1] - sigmas[i])
                x0 = x - float(sigmas[i + 1]) * v
                emit(_pi[0], encode_output(finish(x0), req.output_format,
                                           req.compression))
                _pi[0] += 1
            _prev["x"] = x

        t_img = time.perf_counter()
        try:
            x = api.sample(pipe, cond, neg=neg, cfg=cfg, height=height,
                           width=width, steps=steps, seed=s, sampler=sampler,
                           init_latents=init_latents, denoise=denoise,
                           mask=mask, patch_table=table, plan=plan,
                           on_step=on_step,
                           should_stop=should_stop)
        except api.SampleStopped:
            raise JobStopped(timed_out=not stop.is_set()) from None
        step_s += time.perf_counter() - t_img   # sample evaluates each step
        data = encode_output(finish(x), req.output_format, req.compression,
                             None if meta is None else {**meta, "seed": s})
        out.append(data)
        image_s += time.perf_counter() - t_img
        del x
        mx.clear_cache()

    tokens = (height // token_px) * (width // token_px)
    text_tokens = int(cond.features.shape[0]) - len(cond.image_slots) + (
        0 if neg is None else int(neg.features.shape[0]) - len(neg.image_slots))
    image_tokens = 0
    for c in (cond, neg):
        if c is not None and c.refs:
            image_tokens += len(c.image_slots) + sum(
                (int(r.shape[-2]) // pipe.dit.cfg.patch)
                * (int(r.shape[-1]) // pipe.dit.cfg.patch)
                for r in c.ref_latents)
    usage = {"input_tokens": text_tokens + image_tokens,
             "input_tokens_details": {"text_tokens": text_tokens,
                                      "image_tokens": image_tokens},
             "output_tokens": tokens * req.n,
             "total_tokens": text_tokens + image_tokens + tokens * req.n}
    _log.info("image job: %d image(s) %dx%d, %d steps, encode %.2fs",
              req.n, width, height, run_steps, encode_s)
    return ImageJobResult(images=out, size=(width, height),
                          quality=req.quality, usage=usage, seeds=seeds,
                          steps=steps, run_steps=run_steps, image_s=image_s,
                          step_s=step_s)
