"""Per-row LoRA scale under batching.

A live GGUF LoRA adapter (``modules.LoRAKQuantLinear``, the expert-stack
wrap) multiplies its delta by a per-row factor, so one resident base can
serve adapted and bare rows in the same batch, each row at its own
scale. The factor rides here: a process-global vector published for the
forward in flight, re-derived from the batch's row uids at every model
call and cleared right after, so a forward that nobody published for
cannot silently reuse a stale vector.

Modes:

- ``static`` (run, chat, tests): nothing publishes. A wrapper that finds
  no vector uses scale 1.0 for every row, today's behaviour. ``chat``'s
  ``/adapter`` command publishes a one-row vector to toggle or scale the
  adapter without a reload.
- ``rows`` (serve with a hot-swappable adapter): every forward must have
  a publish. A wrapper that finds no vector raises, so a forward path
  that was missed at wiring time is a crash on the first request, never
  a base-id row served through the adapter.

Wiring on the serve path: the request thread writes ``args._kq_lora``
(the per-slot scale tuple for that request) at the gen-args seam; the
engine thread evaluates ``_make_logits_processors(args, ...)`` and
``insert`` back to back for one request, and the hook on the former
hands the tuple to the latter through a thread-keyed slot that asserts
on every pop. Rows carry their tuple in a uid registry; the batch's
step/prefill/round hooks publish the physical row vector from the uid
list of that call.
"""
from __future__ import annotations

import logging
import threading

import mlx.core as mx

_log = logging.getLogger(__name__)

_INSTALLED_FLAG = "_kq_gguf_lora_rows"
_channel_installed = False
_MAX_UIDS = 4096

_mode = "static"
_n_slots = 1
_current: mx.array | None = None       # (B, S) float32 for the forward in flight
_registry: dict = {}                    # uid -> tuple[float, ...] (len S)
_derived: dict = {}                     # per-publish cache of cast/expanded factors
_pending = threading.local()            # per-thread handoff between hooks


class LoraRowsError(RuntimeError):
    """A wrapper ran without a published row vector in ``rows`` mode, or the
    published vector does not match the batch."""


def configure(mode: str, n_slots: int = 1) -> None:
    """Set the channel mode (``static`` / ``rows``) and the adapter slot count."""
    global _mode, _n_slots, _current
    if mode not in ("static", "rows"):
        raise ValueError(f"lora_rows mode {mode!r} (expected 'static' or 'rows')")
    if n_slots < 1:
        raise ValueError(f"lora_rows n_slots {n_slots} (expected >= 1)")
    _mode, _n_slots, _current = mode, int(n_slots), None
    _registry.clear()


def mode() -> str:
    return _mode


def ensure_rows(n_slots: int) -> None:
    """Enter ``rows`` mode with at least ``n_slots`` adapter slots, keeping
    every live registration (a resident model's in-flight rows) and padding
    its tuples with zeros when the slot count grows. Called at each adapter
    load; the slot count only ever grows for the process lifetime, so a
    request tuple built against a smaller count stays valid (padded)."""
    global _mode, _n_slots
    n = max(int(n_slots), 1)
    _mode = "rows"
    if n <= _n_slots:
        return
    pad = (0.0,) * (n - _n_slots)
    for uid, tup in list(_registry.items()):
        _registry[uid] = tup + pad
    _n_slots = n
    _derived.clear()


def n_slots() -> int:
    return _n_slots


def set_rows(scales) -> None:
    """Publish the row vector for the forward about to run: ``(B,)`` (one
    slot) or ``(B, S)`` float32, or ``None`` to clear. Published as float32
    deliberately (masking and logging read it); wrappers cast the factor to
    the delta dtype before multiplying, never the other way round."""
    global _current
    _derived.clear()
    if scales is None:
        _current = None
        return
    arr = mx.array(scales, dtype=mx.float32) if not isinstance(scales, mx.array) \
        else scales.astype(mx.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] != _n_slots:
        raise LoraRowsError(
            f"row vector shape {tuple(arr.shape)} does not match n_slots={_n_slots}")
    _current = arr


def clear_rows() -> None:
    set_rows(None)


def row_scales(slot: int = 0):
    """The ``(B,)`` float32 factor for ``slot`` in the forward in flight, or
    ``None`` in ``static`` mode with nothing published (meaning scale 1.0).
    Raises in ``rows`` mode when nothing is published."""
    if _current is None:
        if _mode == "rows":
            raise LoraRowsError(
                "no LoRA row vector published for this forward (rows mode); "
                "the model call path is missing a lora_rows publish")
        return None
    return _current[:, slot]


def row_factor(slot: int, dtype, shape) -> mx.array | None:
    """``row_scales(slot)`` cast to ``dtype`` and reshaped to ``shape``
    (``(B, 1, ...)``), cached for the publish in flight so the wrappers pay
    no cast or reshape op per target. ``None`` when nothing is published."""
    rows = row_scales(slot)
    if rows is None:
        return None
    check_rows(int(shape[0]))
    key = ("row", int(slot), dtype, tuple(shape))
    arr = _derived.get(key)
    if arr is None:
        arr = _derived[key] = rows.astype(dtype).reshape(shape)
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
    arr = _derived.get(key)
    if arr is None:
        arr = mx.contiguous(
            mx.repeat(rows, tokens // b * k).astype(dtype).reshape(-1, 1, 1)
        )
        _derived[key] = arr
    return arr


def dense_rows(slot: int, rows: int):
    """Flat ``(rows,)`` float32 factor for a dense target's ``rows`` input
    rows (``B`` batch rows times the tokens per row), cached per publish;
    the in-op LoRA epilogue's ``lora_rows``. ``None`` when nothing is
    published."""
    return _flat_rows(slot, rows, 1, (rows,))


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
    arr = _derived.get(key)
    if arr is None:
        # mx.contiguous: mx.repeat of a single value (B=1) evaluates to a
        # stride-0 broadcast view; the epilogue kernels walk lora_rows as
        # dense memory, so materialize once per publish.
        arr = mx.contiguous(mx.repeat(rows, tokens // b * k).reshape(shape))
        _derived[key] = arr
    return arr


def check_rows(batch_rows: int) -> None:
    """Assert the published vector covers ``batch_rows`` rows (wrappers call
    this with their input's leading dim)."""
    if _current is not None and _current.shape[0] != batch_rows:
        raise LoraRowsError(
            f"published LoRA row vector has {_current.shape[0]} rows, forward "
            f"has {batch_rows}")


# uid registry


def register_uid(uid, scales) -> None:
    """Bind a row's per-slot scale tuple to its uid (``None`` = bare row)."""
    if scales is None:
        tup = (0.0,) * _n_slots
    else:
        tup = tuple(float(s) for s in scales)
        if len(tup) < _n_slots:
            tup = tup + (0.0,) * (_n_slots - len(tup))   # built before a grow
        if len(tup) != _n_slots:
            raise LoraRowsError(
                f"request scale tuple has {len(tup)} entries, entry has "
                f"{_n_slots} adapter slots")
    _registry[uid] = tup
    while len(_registry) > _MAX_UIDS:
        _registry.pop(next(iter(_registry)))


def forget_uid(uid) -> None:
    _registry.pop(uid, None)


def scales_for(uid):
    return _registry.get(uid)


def vector_for_uids(uids) -> mx.array:
    """The ``(B, S)`` float32 vector for a physical row list. A uid with no
    registration is a bare row (all slots 0.0): a registry miss must never
    adapt a row."""
    zero = (0.0,) * _n_slots
    rows = [_registry.get(u, zero) for u in uids]
    if not rows:
        return mx.zeros((0, _n_slots), dtype=mx.float32)
    return mx.array(rows, dtype=mx.float32)


def publish_uids(uids) -> None:
    """Publish the row vector for a forward over ``uids`` (physical order)."""
    set_rows(vector_for_uids(list(uids)))


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


# request -> row handoff (engine thread)

def request_scales(spec):
    """The per-slot scale tuple for a request resolved to ``spec`` (a
    ResolvedModel or ``None``): 1.0 in the slot of the id's adapter within
    its resident entry's adapter list (``spec.adapters``, the sorted union
    over the ids sharing the entry; slot i holds ``adapters[i]``), all zero
    for a bare id. A later per-request ``lora`` field replaces this."""
    adapter = getattr(spec, "adapter", None) if spec is not None else None
    slots = tuple((getattr(spec, "adapters", None) if spec is not None else None)
                  or ((adapter,) if adapter else ()))
    if not adapter:
        return (0.0,) * max(_n_slots, len(slots))
    try:
        i = slots.index(adapter)
    except ValueError:
        raise LoraRowsError(
            f"request adapter {adapter!r} is not in its entry's adapter list "
            f"{slots!r}") from None
    n = max(_n_slots, len(slots))
    return tuple(1.0 if j == i else 0.0 for j in range(n))


def lora_salt(scales) -> int:
    """A stable (cross-process, the APC disk tier persists) salt for a
    row's scale tuple, folded into the row's APC ``extra_hash`` so an
    adapted row and a base row never share a prefix cache entry."""
    import hashlib
    digest = hashlib.blake2b(
        repr(tuple(float(s) for s in scales)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _stash_pending(scales) -> None:
    if getattr(_pending, "scales", None) is not None:
        # A request that failed between its hooks and insert leaves its
        # tuple behind; the next request on this thread overwrites it.
        _log.warning("lora_rows handoff overwrote an unconsumed request tuple")
    _pending.scales = scales


def _pop_pending():
    scales = getattr(_pending, "scales", None)
    _pending.scales = None
    return scales


def install_row_channel() -> None:
    """Wire the serve engine: request tuple -> uid registry -> per-forward
    publish on the stock batch paths. Idempotent. The gmlx-owned MTP paths
    (``spec_engine``, ``speculative``) publish directly."""
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator

    # Process-wide flag, not a marker on the visible function: a patch that
    # wraps insert after this install (seed rows) hides the marker, and a
    # second install would nest a lora wrapper inside it and pop the pending
    # tuple twice.
    global _channel_installed
    if _channel_installed:
        return
    _channel_installed = True

    _orig_procs = ResponseGenerator._make_logits_processors

    def _procs_with_lora(self, args, input_ids=None):
        # Evaluated as an argument of the insert call that follows, on the
        # same engine thread: the request's tuple crosses via a thread-keyed
        # slot that insert pops and asserts on.
        scales = getattr(args, "_kq_lora", None)
        _stash_pending(scales if scales is not None else ())
        return _orig_procs(self, args, input_ids)

    _orig_insert = _ar.BatchGenerator.insert

    def _insert_with_lora(self, prompts, *args, **kwargs):
        scales = _pop_pending()
        if _mode == "rows" and scales is None:
            raise LoraRowsError(
                "insert without a preceding _make_logits_processors (rows "
                "mode): the request's LoRA tuple did not reach its row")
        if scales and any(scales) and getattr(self, "apc_manager", None) is not None:
            # APC isolation: an adapted row and a base row must not share
            # a prefix (same tokens, different logits). Salt the row's
            # semantic hash with its scale tuple, computing the base hash
            # the way the engine would when the request carried none.
            kws = kwargs.get("prompt_kwargs")
            if kws is None and len(args) >= 2:
                kws = args[1]
            if kws is None:
                kws = [{} for _ in prompts]
                if len(args) >= 2:
                    args = args[:1] + (kws,) + args[2:]
                else:
                    kwargs["prompt_kwargs"] = kws
            salt = lora_salt(scales)
            for i, kw in enumerate(kws):
                if kw is None:
                    kw = kws[i] = {}
                base = kw.get("_apc_semantic_hash")
                if base is None:
                    base = self._apc_extra_hash(kw)
                kw["_apc_semantic_hash"] = int(base) ^ salt
        uids = _orig_insert(self, prompts, *args, **kwargs)
        for uid in uids:
            register_uid(uid, scales if scales else None)
        return uids

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

    for fn in (_procs_with_lora, _insert_with_lora, _step_with_rows,
               _generate_with_rows, _prompt_step_with_rows):
        fn.__dict__[_INSTALLED_FLAG] = True
    ResponseGenerator._make_logits_processors = _procs_with_lora
    _ar.BatchGenerator.insert = _insert_with_lora
    _ar.GenerationBatch._step = _step_with_rows
    _ar.PromptProcessingBatch.generate = _generate_with_rows
    _ar.PromptProcessingBatch.prompt_step = _prompt_step_with_rows
    _log.info("lora row channel installed (mode=%s, slots=%d)", _mode, _n_slots)


_GEN_ARGS_FLAG = "_kq_gguf_lora_gen_args"


def install_lora_gen_args() -> None:
    """Request thread: stamp ``args._kq_lora`` from the resolved spec."""
    from gmlx.serve.patches._common import _install_gen_args_transform
    import gmlx.serve.bridge_vlm as serving

    def _stamp(args, request, _processor):
        args._kq_lora = request_scales(serving.get_active_spec())
        return args

    _install_gen_args_transform(_GEN_ARGS_FLAG, _stamp)
