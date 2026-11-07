"""Per-row LoRA scale under batching.

A live GGUF LoRA adapter (``modules.LoRAKQuantLinear``, the expert-stack
wrap) multiplies its delta by a per-row factor, so one resident base can
serve adapted and bare rows in the same batch, each row at its own
scale. The factor rides here: a vector published for the forward in
flight on the current thread, re-derived from the batch's row uids at
every model call and cleared right after, so a forward that nobody
published for cannot silently reuse a stale vector.

Modes:

- ``static`` (run, chat, tests): nothing publishes. A wrapper that finds
  no vector uses scale 1.0 for every row, today's behaviour. ``chat``'s
  ``/adapter`` command publishes a one-row vector to toggle or scale the
  adapter without a reload.
- ``rows`` (serve with a hot-swappable adapter): every forward must have
  a publish. A wrapper that finds no vector raises, so a forward path
  that was missed at wiring time is a crash on the first request, never
  a base-id row served through the adapter.

State. The published vector, its slot-off marks and the derived factor
cache are per thread: each serve engine publishes and runs its forwards
on its own thread. The rows a uid publish reads come from a lookup bound
for the current context (``bind``). In serve that is a view over the
generator's per-row settings, bound by a wrapper on
``BatchGenerator._next`` for the length of the step; callers outside
serve bind a ``UidRegistry``. Uids count from 0 in each generator, so a
process-wide table would mix rows of two resident entries.

A uid publish also records, per adapter slot, whether every row of the
forward has scale 0 (``slot_off``); the wrappers then run the base alone.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import threading
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from gmlx.cache.media_salt import as_salt
from gmlx.upstream.patching import link

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_lora_rows"
_ROWS_STEP_FLAG = "_kq_gguf_lora_rows_step"
_channel_installed = False

_mode = "static"
_n_slots = 1


class _Local(threading.local):
    def __init__(self):
        self.current: mx.array | None = None   # (B, S) float32 for the forward in flight
        self.off: tuple | None = None          # per slot: every published row has scale 0
        self.derived: dict = {}                # per-publish cache of cast/expanded factors


_local = _Local()
_bound: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "gmlx_lora_rows_lookup", default=None)


class LoraRowsError(RuntimeError):
    """A wrapper ran without a published row vector in ``rows`` mode, or the
    published vector does not match the batch."""


def configure(mode: str, n_slots: int = 1) -> None:
    """Set the channel mode (``static`` / ``rows``) and the adapter slot
    count, and clear this thread's published vector."""
    global _mode, _n_slots
    if mode not in ("static", "rows"):
        raise ValueError(f"lora_rows mode {mode!r} (expected 'static' or 'rows')")
    if n_slots < 1:
        raise ValueError(f"lora_rows n_slots {n_slots} (expected >= 1)")
    _mode, _n_slots = mode, int(n_slots)
    _local.current, _local.off = None, None
    _local.derived = {}


def mode() -> str:
    return _mode


def ensure_rows(n_slots: int) -> None:
    """Enter ``rows`` mode with at least ``n_slots`` adapter slots. Called
    at each adapter load; the slot count only ever grows for the process
    lifetime. Stored rows keep the length they were registered with and
    are padded with zeros when they are read."""
    global _mode, _n_slots
    n = max(int(n_slots), 1)
    _mode = "rows"
    if n > _n_slots:
        _n_slots = n


def n_slots() -> int:
    return _n_slots


def set_rows(scales, n: int | None = None) -> None:
    """Publish the row vector for the forward about to run on this thread:
    ``(B,)`` (one slot) or ``(B, S)`` float32, or ``None`` to clear. ``n``
    is the slot count the vector was built for (default: the current
    count). Published as float32 deliberately (masking and logging read
    it); wrappers cast the factor to the delta dtype before multiplying,
    never the other way round."""
    _local.derived.clear()
    _local.off = None
    if scales is None:
        _local.current = None
        return
    n = _n_slots if n is None else int(n)
    arr = mx.array(scales, dtype=mx.float32) if not isinstance(scales, mx.array) \
        else scales.astype(mx.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != n:
        raise LoraRowsError(
            f"row vector shape {tuple(arr.shape)} does not match n_slots={n}")
    _local.current = arr


def clear_rows() -> None:
    set_rows(None)


def row_scales(slot: int = 0):
    """The ``(B,)`` float32 factor for ``slot`` in the forward in flight, or
    ``None`` in ``static`` mode with nothing published (meaning scale 1.0).
    Raises in ``rows`` mode when nothing is published."""
    current = _local.current
    if current is None:
        if _mode == "rows":
            raise LoraRowsError(
                "no LoRA row vector published for this forward (rows mode); "
                "the model call path is missing a lora_rows publish")
        return None
    return current[:, slot]


def row_factor(slot: int, dtype, shape) -> mx.array | None:
    """``row_scales(slot)`` cast to ``dtype`` and reshaped to ``shape``
    (``(B, 1, ...)``), cached for the publish in flight so the wrappers pay
    no cast or reshape op per target. ``None`` when nothing is published."""
    rows = row_scales(slot)
    if rows is None:
        return None
    check_rows(int(shape[0]))
    key = ("row", int(slot), dtype, tuple(shape))
    derived = _local.derived
    arr = derived.get(key)
    if arr is None:
        arr = derived[key] = rows.astype(dtype).reshape(shape)
    return arr


def dense_row_factor(slot: int, lead: tuple, dtype) -> mx.array | None:
    """The factor for a dense target's rank-r intermediate whose leading
    dimensions are ``lead``, cast to ``dtype``. An input with a batch
    dimension, two or more leading dimensions, carries the batch first: it
    gets ``row_factor`` at ``(B, 1, ...)``, and a first dimension that is
    not the published row count raises. A flat ``(B * tokens,)`` layout has
    no batch dimension and gets one value per input row in ``(B, tokens)``
    order, as ``dense_rows`` gives the in-op epilogue. ``None`` when
    nothing is published."""
    rows = row_scales(slot)
    if rows is None:
        return None
    lead = tuple(int(n) for n in lead)
    first = lead[0] if lead else 1
    if len(lead) >= 2 or lead == (rows.shape[0],):
        return row_factor(slot, dtype, (first,) + (1,) * len(lead))
    flat = dense_rows(slot, first)
    key = ("dense", int(slot), dtype, lead)
    derived = _local.derived
    arr = derived.get(key)
    if arr is None:
        arr = derived[key] = flat.astype(dtype).reshape(lead + (1,))
    return arr


def flat_row_factor(slot: int, tokens: int, k: int, dtype, order=None):
    """Per-flat-row factor for ``tokens * k`` rows of ``B`` batch rows in
    ``(B, tokens // B, k)`` order, cast to ``dtype`` and shaped ``(R, 1, 1)``
    for the rank-r intermediate (``order`` from the sorted stock path permutes
    it the way ``_gather_sort`` permuted the rows; not cached). ``None`` when
    nothing is published."""
    rows = row_scales(slot)
    if rows is None:
        return None
    b = rows.shape[0]
    check_rows(b)
    if tokens % b:
        raise LoraRowsError(
            f"published LoRA row vector has {b} rows, forward has {tokens} "
            f"tokens (not a multiple)")
    if order is not None:
        s_tok = mx.repeat(rows, tokens // b)
        return s_tok[order // k].astype(dtype).reshape(-1, 1, 1)
    key = ("flat", int(slot), int(tokens), int(k), dtype)
    derived = _local.derived
    arr = derived.get(key)
    if arr is None:
        arr = mx.contiguous(
            mx.repeat(rows, tokens // b * k).astype(dtype).reshape(-1, 1, 1)
        )
        derived[key] = arr
    return arr


def dense_rows(slot: int, rows: int):
    """Flat ``(rows,)`` float32 factor for a dense target's ``rows`` input
    rows (``B`` batch rows times the tokens per row), cached per publish;
    the in-op LoRA epilogue's ``lora_rows``. ``None`` when nothing is
    published."""
    return _flat_rows(slot, rows, 1, (rows,))


def dense_input_rows(slot: int, lead: tuple):
    """``dense_rows`` for a dense target's input whose leading dimensions
    are ``lead``, under the layout rule of ``dense_row_factor``: with a
    batch dimension the first one must be the published row count, and a
    flat layout must hold a multiple of it."""
    total = 1
    for n in lead:
        total *= int(n)
    if len(lead) >= 2:
        check_rows(int(lead[0]))
    return dense_rows(slot, total)


def flat_rows(slot: int, tokens: int, k: int):
    """``(tokens, k)`` float32 factor for an expert target's routed rows
    (``k`` slots per token), cached per publish. ``None`` when nothing is
    published."""
    return _flat_rows(slot, tokens, k, (tokens, k))


def _flat_rows(slot, tokens, k, shape):
    rows = row_scales(slot)
    if rows is None:
        return None
    b = rows.shape[0]
    check_rows(b)
    if tokens % b:
        raise LoraRowsError(
            f"published LoRA row vector has {b} rows, forward has {tokens} "
            f"tokens (not a multiple)")
    key = ("f32", int(slot), int(tokens), int(k))
    derived = _local.derived
    arr = derived.get(key)
    if arr is None:
        # mx.contiguous: mx.repeat of a single value (B=1) evaluates to a
        # stride-0 broadcast view; the epilogue kernels walk lora_rows as
        # dense memory, so materialize once per publish.
        arr = mx.contiguous(mx.repeat(rows, tokens // b * k).reshape(shape))
        derived[key] = arr
    return arr


def slot_off(slot: int) -> bool:
    """True when every row of the published forward has scale 0 in
    ``slot``. Known only for a uid or tuple publish; otherwise False."""
    off = _local.off
    return off is not None and slot < len(off) and off[slot]


def all_off(slots) -> bool:
    return all(slot_off(s) for s in slots)


def check_rows(batch_rows: int) -> None:
    """Assert the published vector covers ``batch_rows`` rows (wrappers call
    this with their input's leading dim)."""
    current = _local.current
    if current is not None and current.shape[0] != batch_rows:
        raise LoraRowsError(
            f"published LoRA row vector has {current.shape[0]} rows, forward "
            f"has {batch_rows}")


@contextlib.contextmanager
def adapters_off():
    """Run every adapter slot off on this thread for the scope: the
    wrappers take the base path and read no row vector. Binds fresh
    state and binds the saved objects again at exit, so a publish inside
    the scope never touches the outer state. Nests."""
    saved = (_local.current, _local.off, _local.derived)
    _local.current = None
    _local.off = (True,) * _n_slots
    _local.derived = {}
    try:
        yield
    finally:
        _local.current, _local.off, _local.derived = saved


def publish_tuple(scales, rows: int) -> None:
    """Publish one scale tuple for ``rows`` rows, and mark the slots it
    sets to 0 as off."""
    n = _n_slots
    vec = np.zeros((n,), np.float32)
    vals = np.asarray(tuple(scales), np.float32).reshape(-1)
    if vals.size > n:
        raise LoraRowsError(
            f"scale tuple has {vals.size} entries, the channel has {n} slots")
    vec[:vals.size] = vals
    set_rows(mx.array(np.tile(vec, (int(rows), 1))), n=n)
    _local.off = tuple((vec == 0).tolist())


@contextlib.contextmanager
def tuple_scope(scales, rows: int):
    """Publish one scale tuple for ``rows`` rows for the scope, and clear
    it at exit. Does nothing in ``static`` mode. Refuses to nest: in
    ``rows`` mode a vector already published raises."""
    if _mode != "rows":
        yield
        return
    if _local.current is not None:
        raise LoraRowsError(
            "a LoRA row vector is already published on this thread; "
            "tuple_scope does not nest")
    publish_tuple(scales, rows)
    try:
        yield
    finally:
        clear_rows()


def row_tuple(settings) -> tuple:
    """The scale tuple of a row's settings (its ``RowLora.scales``). In
    ``rows`` mode settings with no ``RowLora`` raise; in ``static`` mode
    the tuple is empty."""
    if _mode != "rows":
        return ()
    lora = getattr(settings, "lora", None) if settings is not None else None
    if lora is None:
        raise LoraRowsError(
            "a row's settings carry no RowLora (rows mode): the request's "
            "adapter scales did not reach its prompt embedding")
    return tuple(lora.scales)


# Row salt


def row_salt(identities, scales) -> int:
    """A stable salt for a row's adapter scales, folded into the row's APC
    hash. It names each adapter that is on by the sha256 of its loaded
    bytes and its strength at float32, never by slot, so a new union, a
    rewritten file or a restart cannot reuse another adapter's KV. 0 when
    every slot is 0."""
    on = sorted((identities[i].sha256, float(np.float32(s)))
                for i, s in enumerate(scales) if s != 0.0)
    if not on:
        return 0
    key = repr(("gmlx-lora", 1, tuple(on)))
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


# Row lookups


class UidRow(NamedTuple):
    row: np.ndarray        # (S,) float32
    salt: int


class UidRegistry:
    """A plain uid -> row lookup, for callers outside serve: register rows,
    bind it with :func:`bind`, publish by uid."""

    def __init__(self):
        self._rows: dict = {}

    def register(self, uid, scales, salt: int = 0) -> None:
        self._rows[uid] = UidRow(np.asarray(tuple(scales), np.float32).reshape(-1),
                                 int(salt))

    def row(self, uid):
        return self._rows.get(uid)

    def retired(self, uid) -> bool:
        return False

    def forget(self, uid) -> None:
        self._rows.pop(uid, None)


@contextlib.contextmanager
def bind(lookup):
    """Bind ``lookup`` for uid publishes in this context. ``lookup.row(uid)``
    returns an object with ``.row`` (float32 scales) and ``.salt``, or None;
    ``lookup.retired(uid)`` tells a finished uid from an unknown one."""
    token = _bound.set(lookup)
    try:
        yield lookup
    finally:
        _bound.reset(token)


def bound():
    return _bound.get()


def register_uid(uid, scales) -> None:
    """Bind a row's per-slot scales to its uid in the bound registry
    (``None`` = bare row)."""
    reg = _bound.get()
    if not isinstance(reg, UidRegistry):
        raise LoraRowsError("register_uid needs a bound UidRegistry (lora_rows.bind)")
    tup = (0.0,) * _n_slots if scales is None else tuple(float(s) for s in scales)
    if len(tup) > _n_slots:
        raise LoraRowsError(
            f"request scale tuple has {len(tup)} entries, entry has "
            f"{_n_slots} adapter slots")
    reg.register(uid, tup)


def forget_uid(uid) -> None:
    reg = _bound.get()
    if isinstance(reg, UidRegistry):
        reg.forget(uid)


def _row_of(lookup, uid):
    got = lookup.row(uid) if lookup is not None else None
    return None if got is None or getattr(got, "row", None) is None else got


def scales_for(uid):
    """The uid's scales in the bound lookup, padded to the slot count, or
    None."""
    got = _row_of(_bound.get(), uid)
    if got is None:
        return None
    row = tuple(float(v) for v in np.asarray(got.row).tolist())
    return row + (0.0,) * max(_n_slots - len(row), 0)


def salt_for(uid):
    """The uid's APC salt in the bound lookup (0 for a bare row), or None."""
    got = _row_of(_bound.get(), uid)
    return None if got is None else int(got.salt or 0)


def _rows_array(uids, n: int) -> np.ndarray:
    """The ``(B, n)`` float32 rows of ``uids``. In rows mode a uid the
    bound lookup does not hold raises, unless the lookup reports it
    retired (a finished row still in a round reads as zeros), and so does
    a publish with no lookup bound."""
    out = np.zeros((len(uids), n), np.float32)
    if not uids:
        return out
    lookup = _bound.get()
    if lookup is None:
        if _mode == "rows":
            raise LoraRowsError(
                "uid publish with no row lookup bound (rows mode); the call "
                "runs outside its generator's step")
        return out
    for i, uid in enumerate(uids):
        got = _row_of(lookup, uid)
        if got is None:
            if _mode == "rows" and not lookup.retired(uid):
                raise LoraRowsError(
                    f"row uid {uid!r} has no LoRA scales (rows mode); its "
                    f"insert carried no RowLora")
            continue
        row = np.asarray(got.row, np.float32).reshape(-1)
        if row.size > n:
            raise LoraRowsError(
                f"row uid {uid!r} has {row.size} adapter slots, the channel has {n}")
        out[i, :row.size] = row
    return out


def vector_for_uids(uids) -> mx.array:
    """The ``(B, S)`` float32 vector for a physical row list."""
    n = _n_slots
    return mx.array(_rows_array(list(uids), n))


def publish_uids(uids) -> None:
    """Publish the row vector for a forward over ``uids`` (physical order)."""
    uids = list(uids)
    n = _n_slots
    rows = _rows_array(uids, n)
    set_rows(mx.array(rows), n=n)
    if uids:
        _local.off = tuple((rows == 0).all(axis=0).tolist())


class published:
    """``with published(uids):`` around one model call: publish before,
    clear after, so the next forward must publish its own."""

    def __init__(self, uids):
        self._uids = uids

    def __enter__(self):
        if _mode == "rows":
            publish_uids(self._uids)
        return self

    def __exit__(self, *exc):
        if _mode == "rows":
            set_rows(None)
        return False


class _GeneratorRows:
    """The bound lookup over a ``BatchGenerator``'s per-row settings: a
    row's ``RowLora`` rides its ``RowSettings.lora``, and a uid whose
    settings went with its finish response is in ``_retired_rows``."""

    __slots__ = ("_gen",)

    def __init__(self, gen):
        self._gen = gen

    def row(self, uid):
        store = getattr(self._gen, "_row_settings", None)
        rs = store.get(uid) if store else None
        return getattr(rs, "lora", None) if rs is not None else None

    def retired(self, uid) -> bool:
        return uid in (getattr(self._gen, "_retired_rows", None) or ())


def bound_step(orig):
    """``orig`` (a ``BatchGenerator._next``) run with the generator's row
    settings bound as the uid lookup for the step."""
    def _next_with_rows(self, **kwargs):
        with bind(_GeneratorRows(self)):
            return orig(self, **kwargs)
    return _next_with_rows


def _row_settings_arg(args, kwargs):
    # Positional order: prompts, max_tokens, prompt_kwargs,
    # logits_processors, thinking_budget_criteria, attention_mask,
    # row_settings.
    rows = kwargs.get("row_settings")
    if rows is None and len(args) >= 6:
        rows = args[5]
    return rows


def install_row_channel() -> None:
    """Wire the serve engine: each row's ``RowLora`` rides its row settings
    into the generator's store; the generator's step binds that store,
    and the stock batch paths publish from it per forward. Idempotent.
    The gmlx-owned MTP paths (``spec_engine``, ``speculative``) publish
    directly, inside the same step."""
    from gmlx._vendor.mlx_vlm.generate import ar as _ar

    # Process-wide flag, not a marker on the visible function: a later
    # patch that wraps insert hides the marker.
    global _channel_installed
    if _channel_installed:
        return
    _channel_installed = True

    _orig_insert = _ar.BatchGenerator.insert

    def _insert_with_lora(self, prompts, *args, **kwargs):
        rows = _row_settings_arg(args, kwargs)
        loras = [getattr(rs, "lora", None) for rs in (rows or [None] * len(prompts))]
        if _mode == "rows" and any(lo is None for lo in loras):
            raise LoraRowsError(
                "insert without a RowLora in a row's settings (rows mode): the "
                "request's adapter scales did not reach its row")
        for lo in loras:
            if lo is not None and lo.salt is None:
                raise LoraRowsError(
                    "insert with an unsalted adapted row; the engine salts each "
                    "row's RowLora before insert")
        if any(lo is not None and lo.salt for lo in loras) \
                and getattr(self, "apc_manager", None) is not None:
            # APC isolation: an adapted row and a base row must not share
            # a prefix (same tokens, different logits). Salt the row's
            # semantic hash, computing the base hash the way the engine
            # would when the request carried none.
            kws: Any = kwargs.get("prompt_kwargs")
            if kws is None and len(args) >= 2:
                kws = args[1]
            if kws is None:
                kws = [{} for _ in prompts]
                if len(args) >= 2:
                    args = args[:1] + (kws,) + args[2:]
                else:
                    kwargs["prompt_kwargs"] = kws
            for i, kw in enumerate(kws):
                lo = loras[i]
                if lo is None or not lo.salt:
                    continue
                if kw is None:
                    kw = kws[i] = {}
                base = kw.get("_apc_semantic_hash")
                if base is None:
                    base = self._apc_extra_hash(kw)
                kw["_apc_semantic_hash"] = as_salt(base) ^ lo.salt
        return _orig_insert(self, prompts, *args, **kwargs)

    _orig_next = _ar.BatchGenerator._next
    _next_with_rows = bound_step(_orig_next)

    _orig_step = _ar.GenerationBatch._step

    def _step_with_rows(self):
        with published(getattr(self, "uids", [])):
            return _orig_step(self)

    _orig_generate = _ar.PromptProcessingBatch.generate

    def _generate_with_rows(self, sampler, *args, **kwargs):
        with published(getattr(self, "uids", [])):
            return _orig_generate(self, sampler, *args, **kwargs)

    _orig_prompt_step = _ar.PromptProcessingBatch.prompt_step

    def _prompt_step_with_rows(self, *args, **kwargs):
        with published(getattr(self, "uids", [])):
            return _orig_prompt_step(self, *args, **kwargs)

    for fn in (_insert_with_lora, _next_with_rows, _step_with_rows,
               _generate_with_rows, _prompt_step_with_rows):
        fn.__dict__[_INSTALLED_FLAG] = True
    _next_with_rows.__dict__[_ROWS_STEP_FLAG] = True
    _ar.BatchGenerator.insert = link(_insert_with_lora, _orig_insert)
    _ar.BatchGenerator._next = link(_next_with_rows, _orig_next)
    _ar.GenerationBatch._step = link(_step_with_rows, _orig_step)
    _ar.PromptProcessingBatch.generate = link(_generate_with_rows, _orig_generate)
    _ar.PromptProcessingBatch.prompt_step = link(_prompt_step_with_rows,
                                                 _orig_prompt_step)
    _log.info("lora row channel installed (mode=%s, slots=%d)", _mode, _n_slots)
