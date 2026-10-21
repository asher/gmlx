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
is over budget); ``0`` = never stream the table; ``1`` = force table
streaming even on a fits-in-RAM model (the overhead A/B). ``--stream-cpu``
forces it too, because that mode streams the experts whatever the model
size. When the model is still over budget without the tables, the
experts stream as well (compose); ``GMLX_STREAM_PLE_COMPOSE=0`` keeps
the tables resident and streams only the experts.
"""
from __future__ import annotations

import re
from typing import Any, Callable, NamedTuple, Union

import mlx.core as mx


class StreamableTable(NamedTuple):
    """One streamable lookup-table component of an arch."""
    param_path: str   # dotted module path on the loaded model, e.g. "model.ple_embed"
    gguf_name: str    # wire tensor name, e.g. "per_layer_token_embd.weight"


# Fixed tiers, or a callable for an arch whose tables sit on a variable
# set of layers (deepseek41 reads its engram layer ids from the config).
TableTiers = Union[
    tuple[StreamableTable, ...], Callable[[Any], "tuple[StreamableTable, ...]"]
]


class ArchTables(NamedTuple):
    """An arch's streamable tables, addressed two ways.

    ``tiers`` needs a loaded model; ``name_pattern`` matches the wire
    names and serves the header-only pricing paths (the fit planner, the
    preload gate, the residency footprint), which never hold one.
    """
    arch: str            # GGUF general.architecture
    name_pattern: str    # full-match regex over wire tensor names
    tiers: TableTiers    # ordered, cheapest-to-stream first
    # The model issues every table read before its first layer runs
    # (``prefetch``), so a table read from the file costs a load without
    # a stream flag no decode time: such a load reads it from the file
    # when that alone brings the model under the budget.
    reads_ahead: bool = False


def _deepseek_v41_tiers(model, prefix: str = "") -> tuple[StreamableTable, ...]:
    """One engram table per engram layer, or one table they all share."""
    root = _resolve(model, prefix.rstrip(".")) if prefix else model
    inner = getattr(root, "model", root)
    if getattr(inner, "engram_embed", None) is not None:
        return (StreamableTable(f"{prefix}model.engram_embed",
                                "per_layer_token_embd.weight"),)
    layers = getattr(inner, "layers", None) or ()
    return tuple(
        StreamableTable(f"{prefix}model.layers.{i}.engram.embed",
                        f"blk.{i}.engram_embd.weight")
        for i, layer in enumerate(layers)
        if getattr(layer, "engram", None) is not None
    )


def _deepseek_v41_vl_tiers(model) -> tuple[StreamableTable, ...]:
    """The same tables, under the VLM container's text tower."""
    return _deepseek_v41_tiers(model, "language_model.")


_DEEPSEEK_V41_TABLES = r"blk\.\d+\.engram_embd\.weight|per_layer_token_embd\.weight"

# model_type -> tables, cheapest-to-stream first; implemented archs only
# (embed_tokens is the next candidate, not wired yet).
STREAMABLE_TABLES: dict[str, ArchTables] = {
    "qwen4_exp": ArchTables(
        arch="qwen4exp",
        name_pattern=r"per_layer_token_embd\.weight",
        tiers=(StreamableTable("model.ple_embed",
                               "per_layer_token_embd.weight"),),
    ),
    "deepseek_v41": ArchTables(
        arch="deepseek41",
        name_pattern=_DEEPSEEK_V41_TABLES,
        tiers=_deepseek_v41_tiers,
        reads_ahead=True,
    ),
    "deepseek_v41_vl": ArchTables(
        arch="deepseek41",
        name_pattern=_DEEPSEEK_V41_TABLES,
        tiers=_deepseek_v41_vl_tiers,
        reads_ahead=True,
    ),
}

_NAME_RE_CACHE: dict[str, re.Pattern] = {}


def tables_read_ahead(arch: str | None) -> bool:
    """Whether ``arch`` (GGUF general.architecture) reads its tables
    ahead of its layers, so a table can serve from the file without a
    stream flag (``ArchTables.reads_ahead``)."""
    return any(spec.arch == arch and spec.reads_ahead
               for spec in STREAMABLE_TABLES.values())


def streamable_table_names(arch: str | None) -> re.Pattern | None:
    """Wire names of ``arch``'s streamable tables, or None when it has
    none. The name-level half of the registry: header-only callers price
    a match as streamed, not as an every-token weight."""
    if not arch:
        return None
    hit = _NAME_RE_CACHE.get(arch)
    if hit is not None:
        return hit
    for spec in STREAMABLE_TABLES.values():
        if spec.arch == arch:
            hit = re.compile(spec.name_pattern)
            _NAME_RE_CACHE[arch] = hit
            return hit
    return None


def _resolve(model, path: str):
    obj = model
    for part in path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def resident_table_bytes(mod) -> int:
    """Bytes of ``mod``'s table that live in an array. A table past the
    device buffer ceiling is read from the GGUF and has no array, so it
    weighs nothing here. Its file size is on ``_kq_table_source``."""
    w = getattr(mod, "weight", None)
    return 0 if w is None else int(w.nbytes)


def streamable_tables_for(model) -> list[tuple[StreamableTable, object]]:
    """The declared streamable tables present on ``model`` (missing paths
    are skipped: e.g. a build without the optional table)."""
    spec = STREAMABLE_TABLES.get(getattr(model, "model_type", None))
    if spec is None:
        return []
    tiers = spec.tiers(model) if callable(spec.tiers) else spec.tiers
    out = []
    for tier in tiers:
        mod = _resolve(model, tier.param_path)
        if mod is None:
            continue
        if (getattr(mod, "weight", None) is not None
                or getattr(mod, "_kq_table_source", None) is not None):
            out.append((tier, mod))
    return out


def table_bytes(model) -> int:
    """Total bytes of the declared streamable tables on ``model``.

    Array bytes only. Both callers subtract this from a total that sums
    live parameter arrays, and a figure subtracted from a total may only
    count what that total counts: ``expert_streaming`` goes negative and
    picks the wrong tier otherwise, ``budget._decode_arena_bytes`` clamps
    at 0 and silently zeroes the arena. ``plan.ModelPlan`` subtracts a
    table figure too, but from a header-derived total that does count the
    file bytes, so that one is right to count them."""
    return sum(resident_table_bytes(m) for _, m in streamable_tables_for(model))


def file_table_bytes(model) -> int:
    """Bytes of the declared tables on ``model`` that are read from the
    GGUF (``table_pread``): in no array, so in no parameter total, but in
    the text bytes the header prices."""
    return sum(src.nbytes for _, m in streamable_tables_for(model)
               for src in [getattr(m, "_kq_table_source", None)]
               if src is not None)


def streamed_table_bytes(model) -> int:
    """Bytes of tables actually streamed. A declared table left resident
    (fallback) is wired like any other weight and must stay charged in
    wired-budget accounting."""
    return sum(resident_table_bytes(m) for _, m in streamable_tables_for(model)
               if getattr(m, "_kq_table_streamed", False))


def stream_ple_env(env=None) -> str:
    from gmlx.envflags import env_value

    return env_value("GMLX_STREAM_PLE", env) or ""


_FORCED = False


def force_table_stream(on: bool = True) -> None:
    """Latch ``--stream-cpu`` or ``--stream-experts`` before the weights
    load. The load-time warm touch runs before ``install_expert_streaming``,
    so it cannot see the flag the placement call carries; without the latch
    it would GPU-touch the tables and expert stacks the placement then
    streams."""
    global _FORCED
    _FORCED = bool(on)


def table_stream_selected(model, total_bytes: int, budget: int | None,
                          force_stream: bool = False) -> bool:
    """The selection ladder's step-1 test, shared by the loader and the
    load-time warm touch so both see the same decision: stream the
    declared tables iff forced (``GMLX_STREAM_PLE=1``, or ``--stream-cpu``,
    which streams the experts whatever the model size) or the model is
    over budget. When the tables alone bring it back under, the experts
    go resident (table-only); otherwise the experts stream too (compose;
    ``GMLX_STREAM_PLE_COMPOSE=0`` keeps the table resident instead)."""
    return _selected(table_bytes(model), total_bytes, budget, force_stream)


def tables_bring_under(total_bytes: int, tbytes: int, budget: int | None,
                       env=None) -> bool:
    """The ladder's table-only step: a load over ``budget`` that its
    declared tables alone bring back under it. The tables then serve
    from disk and the experts stay resident. ``GMLX_STREAM_PLE=0`` keeps
    the tables resident."""
    if stream_ple_env(env) == "0" or not tbytes or budget is None:
        return False
    return total_bytes > budget and total_bytes - tbytes <= budget


def _selected(tbytes: int, total_bytes: int, budget: int | None,
              force_stream: bool, env=None) -> bool:
    from gmlx.envflags import env_bool

    ple = stream_ple_env(env)
    if ple == "0":
        return False
    if not tbytes:
        return False
    if ple == "1" or force_stream or _FORCED:
        return True
    if tables_bring_under(total_bytes, tbytes, budget, env):
        return True
    if budget is None or total_bytes <= budget:
        return False
    return env_bool("GMLX_STREAM_PLE_COMPOSE", True, env)


def latched() -> bool:
    """Whether a stream flag latched table streaming for the next load."""
    return _FORCED


class StreamPlan(NamedTuple):
    """What an install streams: the tables, the experts, and whether a
    selected table stays resident under GMLX_STREAM_PLE_COMPOSE=0."""
    tables: bool
    experts: bool
    table_kept: bool = False


def stream_plan(total_bytes: int, tbytes: int, budget: int | None, *,
                force_stream: bool = False, explicit: bool = False,
                latched: bool = False, env=None) -> StreamPlan:
    """The selection ladder, from byte counts alone: the installer runs it
    on the loaded model, and a serve build on the GGUF header before the
    load, so the build prices what the install will stream. ``explicit``
    (``stream: experts``) forces the experts to stream; ``latched``
    stands for the :func:`force_table_stream` latch a build sets later.
    ``env`` is a model's own env window, read over ``os.environ``: the
    install runs inside it, and the header prediction before it."""
    from gmlx.envflags import env_bool

    over = budget is not None and total_bytes > budget
    force = force_stream or explicit
    if not _selected(tbytes, total_bytes, budget, force or latched, env):
        return StreamPlan(False, force or over)
    if force or (budget is not None and total_bytes - tbytes > budget):
        if env_bool("GMLX_STREAM_PLE_COMPOSE", True, env):
            return StreamPlan(True, True)
        return StreamPlan(False, True, table_kept=True)
    return StreamPlan(True, False)


_TABLE_STREAM = None


def table_stream() -> mx.Stream:
    """The dedicated CPU stream table gathers run on. Not the default CPU
    stream: offloaded experts queue there in compose mode, and a shared
    stream would park the layer-1 table gather behind expert work."""
    global _TABLE_STREAM
    if _TABLE_STREAM is None:
        # Materialize the device default first: a stream created before it
        # lands at index 0 and becomes the default (mlx 0.32.1).
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

    # Per-op ``stream=`` only: ``with mx.stream(...)`` permanently rebinds
    # the CPU device default (mlx 0.32.1).
    if kquant:

        class _TableStream(cls):
            _kq_table_streamed = True

            def __call__(self, x):
                import mlx_kquant as kq

                gathered = mx.take(
                    self["weight"], x, axis=0, stream=table_stream())
                flat = gathered.reshape(-1, gathered.shape[-1])
                # As in KQuantEmbedding: a gather-only codec (mxfp8) can
                # pass the float16 range, so it decodes to the output dtype.
                dt = (None if kq.codec_has_matmul(self.kquant_type)
                      else self.out_dtype)
                deq = kq.dequantize(flat, self["scales"], self.kquant_type,
                                    dtype=dt)
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
                # A table whose rows are a byte blob decodes off the table
                # buffer, on the default stream.
                decode = getattr(self, "decode_gathered", None)
                return out if decode is None else decode(out)

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
    from gmlx.stream.wired_limit import _neutralize_wired_limit_sweep

    offloaded = 0
    names: list[str] = []
    for tier, mod in streamable_tables_for(model):
        if getattr(mod, "_hadamard", None) is not None:
            raise NotImplementedError(
                f"{tier.gguf_name}: table streaming re-implements the gather "
                f"and would drop the Hadamard un-rotation of its rows")
        if not getattr(mod, "_kq_table_streamed", False):
            mod.__class__ = _wrapped_class(
                mod.__class__, hasattr(mod, "kquant_type"))
        offloaded += resident_table_bytes(mod)
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
    declared tables, whenever the selection ladder will stream them, and
    the expert stacks under a stream flag latch. The warm touch runs
    before ``install_expert_streaming``, so this re-runs the same ladder
    test; without it, a fits-in-RAM load under ``GMLX_STREAM_PLE=1`` or a
    stream flag would GPU-touch the table and the experts and wire them
    before streaming ever installs."""
    out: set[int] = set()
    if table_stream_selected(model, total_bytes, budget):
        for _, mod in streamable_tables_for(model):
            for attr in ("weight", "scales"):
                a = getattr(mod, attr, None)
                if a is not None:
                    out.add(id(a))
    modules = getattr(model, "modules", None)
    if _FORCED and callable(modules):
        from mlx.utils import tree_flatten

        from gmlx.load.modules import switch_layer_types

        _, glu_types = switch_layer_types()
        for mod in modules():
            if isinstance(mod, glu_types):
                out.update(id(a) for _, a in tree_flatten(mod.parameters()))
    return out
