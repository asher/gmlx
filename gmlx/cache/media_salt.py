# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Per-prefix APC salts for prompts that carry images.

The stock server salts every APC key of a request with one hash of all
its images, so a turn that adds an image misses every prefix the earlier
turns stored. ``MediaSalt`` keys a prefix by the images that start inside
it instead: ``at(p)`` folds the start column and content hash of each
image run that starts before column ``p`` into the salt the request's
text alone would get. A prefix that ends before the first image keys
like a text prompt.

The int value of a ``MediaSalt`` is the salt of the whole prompt, so a
site that reads it as a plain int keys every prefix by all of the
request's images, as the stock salt does. A store keys an entry on the
salt at the entry's length (``salt_at``), and a block on the salt at the
block's end. A lookup over a stock body runs once per window of prefix
lengths that share one salt, longest first (``salt_windows``).

An image hash covers its pixel rows and the rows the processor emits for
it beside them (``image_grid_thw``, ``image_grid_hws``, ``image_meta``,
``image_position_ids``, ``image_sizes``, ``pixel_attention_mask``). A
grid also splits flattened patches into images. The model and processor
key dependencies, audio, video and the tenant stay in the text salt.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import logging
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from typing import Any

_log = logging.getLogger(__name__)

_MASK = (1 << 64) - 1
# Processor outputs that hold one row per image beside the pixel rows:
# the Qwen and Kimi patch grids, the DeepSeek-V4 (n_vit_h, n_vit_w,
# n_llm_h, n_llm_w) rows, and the position, size and mask rows.
_PER_IMAGE_KEYS = ("image_grid_thw", "image_grid_hws", "image_meta",
                   "image_position_ids", "image_sizes",
                   "pixel_attention_mask")
_INSTALLED_FLAG = "_gmlx_media_salt"


def _mix(*values: int) -> int:
    h = hashlib.sha256()
    for v in values:
        h.update((int(v) & _MASK).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def _fold(base: int, spans: Sequence[tuple[int, int]]) -> int:
    if not spans:
        return base
    return _mix(base, len(spans), *(v for span in spans for v in span))


class MediaSalt(int):
    """A request's APC salt that resolves per prefix length.

    ``base`` is the salt of the request's text, ``spans`` the ``(start,
    image hash)`` of each image run in prompt order, and ``mask`` what the
    LoRA and KV wire salts XOR in on top.
    """

    base: int
    spans: tuple[tuple[int, int], ...]
    mask: int
    _starts: tuple[int, ...]
    _at: dict[int, int]

    def __new__(cls, base: int, spans: Iterable[tuple[int, int]],
                mask: int = 0) -> MediaSalt:
        ordered = tuple(sorted((int(s), int(h)) for s, h in spans))
        self = super().__new__(cls, _fold(int(base), ordered) ^ int(mask))
        self.base = int(base)
        self.spans = ordered
        self.mask = int(mask)
        self._starts = tuple(s for s, _h in ordered)
        self._at = {}
        return self

    def at(self, p: int) -> int:
        """The salt of the prefix ``[0, p)``."""
        k = bisect_left(self._starts, int(p))
        value = self._at.get(k)
        if value is None:
            value = self._at[k] = _fold(self.base, self.spans[:k]) ^ self.mask
        return value

    def __xor__(self, other: int) -> MediaSalt:  # type: ignore[override]
        return MediaSalt(self.base, self.spans, self.mask ^ int(other))

    __rxor__ = __xor__  # type: ignore[assignment]

    def __reduce__(self):
        return (MediaSalt, (self.base, self.spans, self.mask))

    def __repr__(self) -> str:
        return f"MediaSalt({int(self)}, images={len(self.spans)})"


def salt_at(salt: Any, p: int) -> Any:
    """``salt`` at prefix length ``p``. A plain salt reads the same at
    every length."""
    return salt.at(p) if isinstance(salt, MediaSalt) else salt


def as_salt(value: Any) -> int:
    """A stored salt as the stores read it: a ``MediaSalt`` whole, any
    other value as an int."""
    if isinstance(value, MediaSalt):
        return value
    return int(value or 0)


def salt_windows(salt: Any, min_prefix: int = 0,
                 max_prefix: int | None = None
                 ) -> list[tuple[Any, int, int | None]]:
    """``(salt, min_prefix, max_prefix)`` per window of prefix lengths that
    share one salt, longest first, in the stock lookup convention: a hit
    is longer than ``min_prefix`` and no longer than a positive
    ``max_prefix``. A plain salt is one window with the bounds unchanged.
    """
    if not isinstance(salt, MediaSalt):
        return [(salt, min_prefix, max_prefix)]
    cap = int(max_prefix) if max_prefix is not None and max_prefix > 0 \
        else None
    starts = sorted({s for s, _h in salt.spans})
    out = []
    for k in range(len(starts), -1, -1):
        # Window k holds the lengths whose prefix contains k image starts.
        lo = starts[k - 1] if k else 0
        hi = starts[k] if k < len(starts) else None
        if cap is not None:
            hi = cap if hi is None else min(hi, cap)
        lo_eff = max(int(min_prefix), lo)
        if hi is not None and hi <= lo_eff:
            continue
        out.append((salt.at(lo + 1), lo_eff, hi))
    return out


def _content_hash(x: Any) -> int:
    """Shape, dtype and bytes of an array, or of each array in a list. An
    mx array and a numpy array of the same values hash alike: the server
    holds one image as an mx array and a list of images of different
    sizes as numpy arrays."""
    if isinstance(x, (list, tuple)):
        return _mix(len(x), *(_content_hash(e) for e in x))
    import mlx.core as mx
    import numpy as np

    if isinstance(x, mx.array):
        dtype = str(x.dtype).rsplit(".", 1)[-1]
        arr = np.asarray(x.astype(mx.float32) if x.dtype == mx.bfloat16
                         else x)
    else:
        arr = np.asarray(x)
        dtype = str(arr.dtype)
    h = hashlib.sha256()
    h.update(repr(tuple(arr.shape)).encode("utf-8"))
    h.update(dtype.encode("utf-8"))
    h.update(np.ascontiguousarray(arr).tobytes())
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def _is_array(x: Any) -> bool:
    return hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0


def _cfg(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _id_config(config: Any) -> Any:
    """The config that names the image ids: the container's own, or its
    thinker's, where qwen3_omni_moe keeps them."""
    thinker = _cfg(config, "thinker_config")
    if thinker is None or any(_cfg(config, k) is not None for k in (
            "image_token_id", "image_token_index", "media_token_ids")):
        return config
    return thinker


def image_token_ids(config: Any) -> set[int]:
    """The placeholder ids an image expands to, from the container config."""
    config = _id_config(config)
    ids = {int(v) for v in (_cfg(config, "image_token_id"),
                            _cfg(config, "image_token_index"))
           if v is not None}
    ids.update(int(t) for t in _cfg(config, "media_token_ids") or ())
    return ids


def _prompt_row(input_ids: Any) -> Any:
    import numpy as np

    if input_ids is None:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(input_ids).reshape(-1)


def _runs(row: Any, image_ids: set[int]) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of image ids in ``row``."""
    import numpy as np

    if not image_ids or row.size == 0:
        return []
    hit = np.isin(row, np.fromiter(image_ids, dtype=row.dtype))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], hit.astype(np.int8),
                                                   [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _patch_counts(raw: dict) -> Any:
    """Pixel rows per image when the processor flattens every image's
    patches into one array, from the grid it emits beside them: t*h*w
    (``image_grid_thw``), h*w (``image_grid_hws``) or n_vit_h*n_vit_w
    (``image_meta``). None without a grid."""
    import numpy as np

    for key, width, cols in (("image_grid_thw", 3, 3),
                             ("image_grid_hws", 2, 2),
                             ("image_meta", 4, 2)):
        grid = raw.get(key)
        if grid is not None:
            rows = np.asarray(grid).astype(np.int64).reshape(-1, width)
            return rows[:, :cols].prod(axis=1)
    return None


def _image_rows(raw: dict, n: int) -> list[list[Any]] | None:
    """The arrays that make up each of ``n`` images, or None when the
    pixel payload does not split into ``n`` images."""
    import numpy as np

    pv = raw.get("pixel_values")
    if pv is None:
        return None
    counts = _patch_counts(raw)
    if counts is not None and _is_array(pv):
        # Flattened patches: each image owns its grid's rows.
        if counts.shape[0] != n or int(counts.sum()) != int(pv.shape[0]):
            return None
        ends = np.cumsum(counts)
        rows = [[pv[int(e - c):int(e)]] for c, e in zip(counts, ends)]
    elif isinstance(pv, (list, tuple)):
        if len(pv) != n:
            return None
        rows = [[p] for p in pv]
    elif _is_array(pv) and int(pv.shape[0]) == n:
        rows = [[pv[i]] for i in range(n)]
    else:
        return None
    for key in _PER_IMAGE_KEYS:
        v = raw.get(key)
        if _is_array(v) and int(v.shape[0]) == n:
            for i in range(n):
                rows[i].append(v[i])
    return rows


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _per_image(pv: Any, n: int) -> list[Any] | None:
    """The ``[C, H, W]`` pixels of each of ``n`` images, or None."""
    if isinstance(pv, (list, tuple)):
        images = list(pv)
    elif _is_array(pv) and len(pv.shape) == 4:
        images = [pv[i] for i in range(int(pv.shape[0]))]
    else:
        return None
    if len(images) != n or any(len(x.shape) != 3 for x in images):
        return None
    return images


def _expected_tokens(raw: dict, n: int, config: Any, processor: Any
                     ) -> tuple[list[int], bool] | None:
    """The tokens each of ``n`` images expands to, by the rule of the
    processor that emitted ``raw``, and whether that count is the image's
    whole run (its markers are image ids too) rather than its soft tokens.
    None when the rule is unknown. The processor's settings come first,
    then the vision config's.
    """
    import numpy as np

    ip = getattr(processor, "image_processor", None)
    vc = _cfg(_id_config(config), "vision_config")
    grid = raw.get("image_grid_thw")
    if grid is not None:
        # Qwen: t*h*w patches, merge_size**2 per token.
        merge = int(_first(_cfg(ip, "merge_size"),
                           _cfg(ip, "spatial_merge_size"),
                           _cfg(vc, "spatial_merge_size"), 2))
        rows = np.asarray(grid).astype(np.int64).reshape(-1, 3)
        return [int(r.prod()) // (merge * merge) for r in rows], False
    grid = raw.get("image_grid_hws")
    if grid is not None:
        # Kimi: h*w patches, one merge kernel per token.
        kernel = np.asarray(_first(_cfg(ip, "merge_kernel_size"),
                                   _cfg(vc, "merge_kernel_size"), (2, 2)),
                            dtype=np.int64).reshape(-1)
        per = int(kernel.prod()) if kernel.size == 2 else int(kernel[0]) ** 2
        rows = np.asarray(grid).astype(np.int64).reshape(-1, 2)
        return [int(r.prod()) // per for r in rows], False
    pos = raw.get("image_position_ids")
    if _is_array(pos) and len(pos.shape) == 3 and int(pos.shape[0]) == n:
        # Gemma-4 12B: one token per patch; padding rows sit at -1.
        arr = np.asarray(pos)
        return [int((arr[i, :, 0] >= 0).sum()) for i in range(n)], False
    meta = raw.get("image_meta")
    if meta is not None and raw.get("image_spans") is None:
        # DeepSeek-V4.1: START (IMAGE * n_llm_w + NEWLINE) * n_llm_h END,
        # every one an image id, per (n_vit_h, n_vit_w, n_llm_h, n_llm_w).
        from gmlx.models.deepseek_v41.image_block import build_image_block

        rows = np.asarray(meta).astype(np.int64).reshape(-1, 4)
        return [len(build_image_block(int(r[2]), int(r[3]))[0])
                for r in rows], True
    patch = _first(_cfg(ip, "patch_size"), _cfg(vc, "patch_size"))
    pool = _first(_cfg(ip, "pooling_kernel_size"),
                  _cfg(vc, "pooling_kernel_size"))
    pv = raw.get("pixel_values")
    if patch and pool:
        # Gemma-4: [C, H, W] per image, pool**2 patches per token.
        images = _per_image(pv, n)
        if images is None:
            return None
        p, k = int(patch), int(pool)
        return [(int(x.shape[1]) // p) * (int(x.shape[2]) // p) // (k * k)
                for x in images], False
    sizes = raw.get("image_sizes")
    merge = _first(_cfg(ip, "spatial_merge_size"),
                   _cfg(vc, "spatial_merge_size"))
    if sizes is not None and patch and merge:
        # Muse Glimmer: an image resized to h x w pixels, one token per
        # cell of patch * merge pixels on a side.
        cell = int(patch) * int(merge)
        rows = np.asarray(sizes).astype(np.int64).reshape(-1, 2)
        return [int(h // cell) * int(w // cell) for h, w in rows], False
    strategy = _first(_cfg(processor, "vision_feature_select_strategy"),
                      _cfg(config, "vision_feature_select_strategy"))
    patch = _first(_cfg(processor, "patch_size"), patch)
    if strategy is not None and patch:
        # llava: the first image's (H // patch) * (W // patch) patches plus
        # the processor's extra tokens (CLS), less the one the "default"
        # strategy drops, for every image.
        images = _per_image(pv, n)
        if images is None:
            return None
        p = int(patch)
        extra = int(_first(_cfg(processor, "num_additional_image_tokens"), 1))
        per = ((int(images[0].shape[1]) // p) * (int(images[0].shape[2]) // p)
               + extra - (1 if strategy == "default" else 0))
        return [per] * n, False
    return None


def _runs_hold_one_image_each(row: Any, runs: list[tuple[int, int]],
                              raw: dict, config: Any, processor: Any) -> bool:
    """Whether run ``i`` holds exactly image ``i``: the processor's own
    block bounds (``image_spans``, DeepSeek-V4) are the runs, or each run
    holds as many image tokens as its image expands to. Adjacent images
    merge into one run and a placeholder typed as text makes a run of its
    own, so a count alone can pair a run with another image."""
    import numpy as np

    blocks = raw.get("image_spans")
    if blocks is not None:
        return [(int(a), int(b)) for a, b in blocks] == runs
    rule = _expected_tokens(raw, len(runs), config, processor)
    if rule is None:
        return False
    expected, whole = rule
    if len(expected) != len(runs):
        return False
    # The ids an image expands to, without the begin and end markers some
    # configs list beside them.
    ids = _id_config(config)
    soft = [] if whole else [int(v) for v in (_cfg(ids, "image_token_id"),
                                              _cfg(ids, "image_token_index"))
                             if v is not None]
    for (a, b), want in zip(runs, expected):
        got = int(np.isin(row[a:b], soft).sum()) if soft else b - a
        if got != want:
            return False
    return True


def image_spans(input_ids: Any, raw: dict, config: Any, processor: Any = None
                ) -> tuple[tuple[int, int], ...] | None:
    """``(start, image hash)`` per image run of a prompt, or None when the
    prompt names no image run.

    Run ``i`` maps to image ``i`` when the runs and the images pair one to
    one (``_runs_hold_one_image_each``). Otherwise one span at the first
    run carries a hash of every image, so a prefix before the first run
    still keys like text and every longer one keys on all the images.
    """
    row = _prompt_row(input_ids)
    runs = _runs(row, image_token_ids(config))
    if not runs or raw.get("pixel_values") is None:
        return None
    rows = _image_rows(raw, len(runs))
    if rows is None or not _runs_hold_one_image_each(row, runs, raw, config,
                                                     processor):
        whole = [raw["pixel_values"]] + [
            raw[k] for k in _PER_IMAGE_KEYS if raw.get(k) is not None]
        return ((runs[0][0], _content_hash(whole)),)
    return tuple((a, _mix(*(_content_hash(x) for x in r)))
                 for (a, _b), r in zip(runs, rows))


class _ImageHash(int):
    """The image hash of a request whose image runs mapped to ``spans`` on
    the request thread. Its value folds every span, so a site that reads
    it as an int keys on all of the request's images."""

    spans: tuple[tuple[int, int], ...]

    def __new__(cls, spans: Iterable[tuple[int, int]]) -> _ImageHash:
        spans = tuple(spans)
        self = super().__new__(cls, _fold(0, spans) or 1)
        self.spans = spans
        return self

    def __reduce__(self):
        return (_ImageHash, (self.spans,))


class _Pending:
    """The preprocessed inputs of the request ``generate`` is handling,
    until its image hash reads them."""

    __slots__ = ("raw", "generator")

    def __init__(self) -> None:
        self.raw: dict | None = None
        self.generator: Any = None


_PENDING: contextvars.ContextVar[_Pending | None] = contextvars.ContextVar(
    "_gmlx_media_salt_pending", default=None)


def _wrapping(wrapper, inner):
    """``wrapper`` over ``inner``, keeping the flags other installers set
    on ``inner`` readable on the outermost function."""
    wrapper.__dict__.update(inner.__dict__)
    wrapper.__wrapped__ = inner
    return wrapper


def install_media_salt() -> bool:
    """Key the serve engine's APC by the images inside each prefix.

    ``ResponseGenerator.generate`` computes a request's salt on the
    request thread. Wrapped, its ``_preprocess_request`` hands the
    request's inputs to ``hash_image_payload``, which hashes each image
    once, maps the image runs to images (``image_spans``) and returns the
    spans in place of the stock hash of all the pixels.
    ``semantic_extra_hash`` builds the ``MediaSalt`` from the spans and
    the salt of the text, and ``BatchGenerator._apc_extra_hash`` hands it
    on whole. The GPU worker hashes nothing: the stock ``_gpu_embed``
    passes the salt through. A payload outside a request hashes as stock,
    except that a list of images of different sizes hashes by content.
    Idempotent. The APC tiers resolve the salt per prefix
    (``GmlxAPCManager``, ``gmlx.cache.snapshot``, ``SpecPrefixCache``).
    """
    apc = importlib.import_module("gmlx._vendor.mlx_vlm.apc")
    if getattr(apc, _INSTALLED_FLAG, False):
        return True
    ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
    server = importlib.import_module("gmlx._vendor.mlx_vlm.server.generation")
    cls = server.ResponseGenerator
    stock_image_hash = apc.hash_image_payload
    stock_semantic = apc.semantic_extra_hash
    stock_generate = cls.generate
    stock_preprocess = cls._preprocess_request
    stock_extra = ar.BatchGenerator._apc_extra_hash

    def generate(self, *args, **kwargs):
        token = _PENDING.set(_Pending())
        try:
            return stock_generate(self, *args, **kwargs)
        finally:
            _PENDING.reset(token)

    def _preprocess_request(self, *args, **kwargs):
        raw = stock_preprocess(self, *args, **kwargs)
        pending = _PENDING.get()
        if (pending is not None and pending.raw is None
                and getattr(self, "apc_mode", None) is not None
                and isinstance(raw, dict)
                and raw.get("pixel_values") is not None):
            pending.raw, pending.generator = raw, self
        return raw

    def _request_spans(pixel_values):
        """The spans of the request whose pixels these are, or None."""
        pending = _PENDING.get()
        if (pending is None or pending.raw is None or pixel_values is None
                or pending.raw.get("pixel_values") is not pixel_values):
            return None
        raw, gen = pending.raw, pending.generator
        pending.raw = pending.generator = None
        try:
            return image_spans(raw.get("input_ids"), raw,
                               getattr(gen.model, "config", None),
                               getattr(gen, "processor", None))
        except Exception:  # noqa: BLE001  any failure keeps the stock hash
            _log.debug("APC image salt skipped", exc_info=True)
            return None

    def hash_image_payload(pixel_values=None, image_ref=None):
        spans = _request_spans(pixel_values)
        if spans:
            return _ImageHash(spans)
        # The stock hash returns 0, the image hash of a text prompt, for a
        # list of images of different sizes, as Gemma-4's processor
        # returns them. Such a request could restore a prefix stored for
        # other images of the same sizes.
        h = stock_image_hash(pixel_values=pixel_values, image_ref=image_ref)
        if not h and isinstance(pixel_values, (list, tuple)) \
                and pixel_values:
            h = _content_hash(list(pixel_values))
        return h

    def semantic_extra_hash(*, tenant=None, image_hash=0, media=None,
                            model=None, processor=None):
        if isinstance(image_hash, _ImageHash):
            text = stock_semantic(tenant=tenant, image_hash=0, media=media,
                                  model=model, processor=processor)
            return MediaSalt(text, image_hash.spans)
        return stock_semantic(tenant=tenant, image_hash=image_hash,
                              media=media, model=model, processor=processor)

    def _apc_extra_hash(self, prompt_kwargs):
        salt = (prompt_kwargs or {}).get("_apc_semantic_hash")
        if (isinstance(salt, MediaSalt)
                and getattr(self, "apc_manager", None) is not None):
            return salt
        return stock_extra(self, prompt_kwargs)

    hash_image_payload.__wrapped__ = stock_image_hash  # type: ignore[attr-defined]
    semantic_extra_hash.__wrapped__ = stock_semantic  # type: ignore[attr-defined]
    _apc_extra_hash.__wrapped__ = stock_extra  # type: ignore[attr-defined]
    setattr(apc, "hash_image_payload", hash_image_payload)
    setattr(apc, "semantic_extra_hash", semantic_extra_hash)
    cls.generate = _wrapping(generate, stock_generate)
    cls._preprocess_request = _wrapping(_preprocess_request, stock_preprocess)
    ar.BatchGenerator._apc_extra_hash = _apc_extra_hash
    # The owned MTP view grafts the engine's helpers once; a graft taken
    # before this install gets the wrapper too.
    from gmlx.spec import engine

    if engine._L1_BOUND[0]:
        engine._L1View._apc_extra_hash = _apc_extra_hash  # type: ignore[attr-defined]
    setattr(apc, _INSTALLED_FLAG, True)
    return True
