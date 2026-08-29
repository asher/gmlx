"""Streamable lookup-table components (PLE tables and kin).

Some archs carry a large embedding-style lookup table whose streaming
economics beat the routed experts': the qwen4exp PLE n-gram table is
26.8 GiB resident but a decode step gathers 16 rows (~1.4 KB). When a
model is over the wired budget, offloading such a table first can bring
the rest of the model (experts included) back under budget, so the
selection ladder in ``install_expert_streaming`` consults this module
before deciding to stream experts.

The hard constraint: Metal residency is per buffer, so any GPU-stream op
that references the table wires all of it. A streamed table must never
be a GPU-stream input; the row gather runs on a dedicated CPU stream
(not the default CPU stream, which offloaded experts occupy in compose
mode) and only the gathered rows - a fresh, small buffer - cross back to
the GPU for dequantization.

This module is the single source of truth for what is streamable on an
arch: ``every_token_ranges`` (pin_weights) and the load-time GPU warm
touch both exclude declared components through it, so a streamed table
can be neither mlocked nor GPU-touched by the load path.

``GMLX_STREAM_PLE``: unset = automatic (stream the table when the model
is over budget and the table alone clears it); ``0`` = never stream the
table; ``1`` = force table streaming even on a fits-in-RAM model (the
overhead A/B) - still subject to the compose guard: v1 refuses to
stream the table and the experts at once.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import mlx.core as mx


class StreamableTable(NamedTuple):
    """One streamable lookup-table component of an arch."""
    param_path: str   # dotted module path on the loaded model, e.g. "model.ple_embed"
    gguf_name: str    # wire tensor name, e.g. "per_layer_token_embd.weight"


# model_type -> ordered tiers, cheapest-to-stream first. Only implemented
# tiers belong here: embed_tokens is the obvious next tier on several archs
# (same lookup economics) but is not wired yet.
STREAMABLE_TABLES: dict[str, tuple[StreamableTable, ...]] = {
    "qwen4_exp": (
        StreamableTable("model.ple_embed", "per_layer_token_embd.weight"),
    ),
}


def _resolve(model, path: str):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def streamable_tables_for(model) -> list[tuple[StreamableTable, object]]:
    """The declared streamable tables present on ``model`` (missing paths
    are skipped: e.g. a build without the optional table)."""
    tiers = STREAMABLE_TABLES.get(getattr(model, "model_type", None), ())
    out = []
    for tier in tiers:
        mod = _resolve(model, tier.param_path)
        if mod is not None and getattr(mod, "weight", None) is not None:
            out.append((tier, mod))
    return out


def table_bytes(model) -> int:
    """Total bytes of the declared streamable tables on ``model``."""
    return sum(int(m.weight.nbytes) for _, m in streamable_tables_for(model))


def stream_ple_env() -> str:
    return os.environ.get("GMLX_STREAM_PLE", "")


def table_stream_selected(model, total_bytes: int, budget: int | None) -> bool:
    """The selection ladder's step-1 test, shared by the loader and the
    load-time warm touch so both see the same decision: stream the
    declared tables iff forced (``GMLX_STREAM_PLE=1``), or the model is
    over budget and the tables alone bring it back under. When the
    post-table estimate is still over budget the ladder falls back to
    expert streaming with the table resident (v1 has no compose mode),
    so this returns False there."""
    env = stream_ple_env()
    if env == "0":
        return False
    tbytes = table_bytes(model)
    if not tbytes:
        return False
    if env == "1":
        # Forced: still refuse compose (see install_table_streaming).
        return True
    if budget is None:
        return False
    return total_bytes > budget and (total_bytes - tbytes) <= budget


_TABLE_STREAM = None


def table_stream() -> mx.Stream:
    """The dedicated CPU stream table gathers run on. Not the default CPU
    stream: offloaded experts queue there in compose mode, and a shared
    stream would park the layer-1 table gather behind expert work."""
    global _TABLE_STREAM
    if _TABLE_STREAM is None:
        # Materialize the device default first: a new_stream created before
        # the CPU default exists lands at index 0, which MLX then treats as
        # the default - the dedicated stream would silently BE the stream
        # offloaded experts run on, creation-order dependent.
        default = mx.default_stream(mx.cpu)
        s = mx.new_stream(mx.cpu)
        if s == default:  # belt against the index-0 surprise
            s = mx.new_stream(mx.cpu)
        _TABLE_STREAM = s
    return _TABLE_STREAM


_WRAP_CACHE: dict[tuple[type, bool], type] = {}


def _wrapped_class(cls, kquant: bool):
    """Per-instance ``__class__`` swap target (same pattern as the expert
    CPU offload): the row gather - the only op that touches the table
    buffer - runs on the dedicated CPU stream; dequantization consumes the
    gathered rows (a fresh small buffer) on the default stream. The base is
    ``mlx_kquant.nn.KQuantEmbedding`` for kquant tables, or a plain
    ``nn.Embedding`` for native-fp loads; both expose ``weight`` and their
    forward is a row gather."""
    sub = _WRAP_CACHE.get((cls, kquant))
    if sub is not None:
        return sub

    # The gather routes via per-op ``stream=`` kwargs, never ``with
    # mx.stream(...)``: the context manager permanently rebinds the CPU
    # device's default stream (mlx 0.32.1), so one context-managed table
    # gather would silently move the experts' ``mx.stream(mx.cpu)`` calls
    # onto the dedicated stream and defeat the isolation.
    if kquant:

        class _TableStream(cls):
            _kq_table_streamed = True

            def __call__(self, x):
                import mlx_kquant as kq

                gathered = mx.take(
                    self["weight"], x, axis=0, stream=table_stream())
                flat = gathered.reshape(-1, gathered.shape[-1])
                deq = kq.dequantize(flat, self["scales"], self.kquant_type)
                return deq.reshape(
                    *gathered.shape[:-1], self.dims).astype(self.out_dtype)

            def as_linear(self, x):
                raise RuntimeError(
                    "as_linear on a streamed table would run a GPU matmul "
                    "over the whole table buffer and wire it; a streamed "
                    "table cannot back a tied lm_head"
                )

    else:

        class _TableStream(cls):
            _kq_table_streamed = True

            def __call__(self, x):
                out = mx.take(self.weight, x, axis=0, stream=table_stream())
                return out

            def as_linear(self, x):
                raise RuntimeError(
                    "as_linear on a streamed table would run a GPU matmul "
                    "over the whole table buffer and wire it"
                )

    _TableStream.__name__ = cls.__name__ + "_TableStream"
    _WRAP_CACHE[(cls, kquant)] = _TableStream
    return _TableStream


def install_table_streaming(model) -> tuple[int, list[str]]:
    """Wrap the declared streamable tables for CPU-stream gathers and
    neutralize the wired-limit sweep. Returns ``(offloaded_bytes,
    gguf_names)``; the caller (the loader's selection ladder) owns the
    residency deduction and the decision log line. Idempotent."""
    from gmlx.load.loader import _neutralize_wired_limit_sweep

    offloaded = 0
    names: list[str] = []
    for tier, mod in streamable_tables_for(model):
        if getattr(mod, "_kq_table_streamed", False):
            offloaded += int(mod.weight.nbytes)
            names.append(tier.gguf_name)
            continue
        mod.__class__ = _wrapped_class(
            mod.__class__, hasattr(mod, "kquant_type"))
        offloaded += int(mod.weight.nbytes)
        names.append(tier.gguf_name)
    if offloaded:
        _neutralize_wired_limit_sweep()
    return offloaded, names


def table_streaming_active(model) -> bool:
    return any(getattr(m, "_kq_table_streamed", False)
               for _, m in streamable_tables_for(model))


def streamed_table_array_ids(model) -> set[int]:
    """``id()``s of the arrays belonging to tables actually streamed on
    ``model`` - the exclusion set for the explicit Metal residency install
    (a residency insert would wire the buffer as surely as a GPU op)."""
    out: set[int] = set()
    for _, mod in streamable_tables_for(model):
        if not getattr(mod, "_kq_table_streamed", False):
            continue
        for attr in ("weight", "scales"):
            a = getattr(mod, attr, None)
            if a is not None:
                out.add(id(a))
    return out


def warm_touch_exclusions(model, total_bytes: int,
                          budget: int | None) -> set[int]:
    """``id()``s of arrays the load-time GPU warm touch must skip: the
    declared tables, whenever the selection ladder will stream them. The
    warm touch runs before ``install_expert_streaming``, so this re-runs
    the same ladder test; without it, a fits-in-RAM load under
    ``GMLX_STREAM_PLE=1`` would GPU-touch the table and wire it before
    streaming ever installs."""
    if not table_stream_selected(model, total_bytes, budget):
        return set()
    out: set[int] = set()
    for _, mod in streamable_tables_for(model):
        for attr in ("weight", "scales"):
            a = getattr(mod, attr, None)
            if a is not None:
                out.add(id(a))
    return out
