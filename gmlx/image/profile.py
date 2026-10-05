"""Per-op wall time of one model call, for the image op inventory.

``StepProfiler`` times every leaf module call and every kernel seam in
``gmlx.image.ops``. Leaf modules are swapped onto timed subclasses and the
seams are replaced by timed wrappers while the profiler is installed. A
timed op first evaluates its array inputs and books that time as glue, the
lazy work queued since the previous op. It then runs, evaluates its output
and books its own time. Calls nested inside a timed op run untimed, so no
time is booked twice, and the records of one step add up to its wall time.
A feed-forward pair on the fused kernel books one ``swiglu_fused`` record
in place of its two leaf calls and their ``swiglu``.

The glue records close every gap, so the records always add up to the
instrumented step's own wall time. Each timed op also pays one eval round
trip that a plain step does not, and ``sync_round_trip`` measures it in the
same run. ``summarize`` subtracts it from every op record and scores the
corrected op total against the plain step time, which checks that the
per-op split accounts for the step.
"""

from __future__ import annotations

import functools
import re
import statistics
import time
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

from . import ops

_TIMED: dict[type, type] = {}


@dataclass
class OpRecord:
    kind: str               # "module", "seam" or "glue"
    name: str               # class or seam name; for glue, the op it precedes
    path: str               # module path; a seam takes its caller's
    codec: str              # kquant codec or weight dtype, "" for seams
    in_shapes: list
    in_dtypes: list
    out_shape: list
    out_dtype: str
    seconds: float


def _arrays(tree) -> list[mx.array]:
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, (list, tuple)):
        return [a for t in tree for a in _arrays(t)]
    if isinstance(tree, dict):
        return [a for t in tree.values() for a in _arrays(t)]
    return []


def _codec(m: nn.Module) -> str:
    codec = getattr(m, "kquant_type", None)
    if codec:
        return str(codec)
    w = m.get("weight")
    return _dt(w) if isinstance(w, mx.array) else ""


def _is_leaf(m: nn.Module) -> bool:
    for v in m.values():
        if isinstance(v, nn.Module):
            return False
        if isinstance(v, (list, tuple)) and any(
                isinstance(x, nn.Module) for x in v):
            return False
        if isinstance(v, dict) and any(
                isinstance(x, nn.Module) for x in v.values()):
            return False
    return True


def _timed_class(cls: type) -> type:
    """A subclass of ``cls`` whose call books through the owning profiler.
    One per class, shared by every profiler."""
    hit = _TIMED.get(cls)
    if hit is None:
        base = cls.__call__

        def __call__(m, *args, **kwargs):
            return m._prof_owner._timed(
                "module", cls.__name__, m._prof_path, _codec(m),
                lambda *a, **k: base(m, *a, **k), args, kwargs)

        hit = type(f"Timed{cls.__name__}", (cls,),
                   {"__call__": __call__, "_prof_base": cls})
        _TIMED[cls] = hit
    return hit


def _parent(path: str) -> str:
    """The module a seam call after ``path`` most likely runs in: trailing
    list indices dropped, then one more element."""
    parts = path.split(".")
    while parts and parts[-1].isdigit():
        parts.pop()
    return ".".join(parts[:-1])


def _dt(a: mx.array) -> str:
    return str(a.dtype).replace("mlx.core.", "")


class StepProfiler:
    def __init__(self, model: nn.Module,
                 seams=ops.SEAMS + ("layer_norm", "swiglu_fused")):
        self.model = model
        self.seams = tuple(seams)
        self.records: list[OpRecord] = []
        self._depth = 0
        self._mark = 0.0
        self._caller = ""
        self._saved: dict = {}
        self._swapped: list[tuple[nn.Module, type]] = []

    # Install and remove.

    def __enter__(self) -> "StepProfiler":
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def install(self) -> None:
        for path, m in self.model.named_modules():
            if not _is_leaf(m):
                continue
            cls = type(m)
            m._prof_path = path
            m._prof_owner = self
            m.__class__ = _timed_class(cls)
            self._swapped.append((m, cls))
        for name in self.seams:
            fn = getattr(ops, name)
            self._saved[name] = fn
            setattr(ops, name, self._timed_seam(name, fn))

    def remove(self) -> None:
        for m, cls in self._swapped:
            m.__class__ = cls
            m.__dict__.pop("_prof_path", None)
            m.__dict__.pop("_prof_owner", None)
        self._swapped.clear()
        for name, fn in self._saved.items():
            setattr(ops, name, fn)
        self._saved.clear()

    def _timed_seam(self, name: str, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self._timed("seam", name, self._caller, "", fn, args,
                               kwargs)
        return wrapper

    # Timing.

    def _timed(self, kind, name, path, codec, fn, args, kwargs):
        if self._depth:
            return fn(*args, **kwargs)
        ins = _arrays(args) + _arrays(kwargs)
        mx.eval(ins)
        t_in = time.perf_counter()
        self.records.append(OpRecord(
            "glue", name, path, "", [], [], [], "", t_in - self._mark))
        self._depth += 1
        try:
            out = fn(*args, **kwargs)
        finally:
            self._depth -= 1
        if kind == "module":
            self._caller = _parent(path)
        outs = _arrays(out)
        mx.eval(outs)
        self._mark = time.perf_counter()
        o = outs[0] if outs else None
        self.records.append(OpRecord(
            kind, name, path, codec, [list(a.shape) for a in ins],
            [_dt(a) for a in ins], list(o.shape) if o is not None else [],
            _dt(o) if o is not None else "", self._mark - t_in))
        return out

    def step(self, fn) -> tuple[list[OpRecord], float]:
        """Run ``fn`` once instrumented. Returns the records and the wall
        time, which the records sum to."""
        self.records = []
        mx.synchronize()
        t0 = self._mark = time.perf_counter()
        out = fn()
        mx.eval(_arrays(out))
        end = time.perf_counter()
        self.records.append(OpRecord("glue", "tail", "", "", [], [], [], "",
                                     end - self._mark))
        return self.records, end - t0


def sync_round_trip(iters: int = 200, warmup: int = 20) -> dict:
    """Wall time of one eval round trip on a one-element add: the fixed cost
    a timed op pays for its own command buffer and wait."""
    x = mx.zeros((1,))
    mx.eval(x)
    ts: list[float] = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        x = x + 1
        mx.eval(x)
        if i >= warmup:
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_s": statistics.median(ts), "p10_s": ts[len(ts) // 10],
            "p90_s": ts[9 * len(ts) // 10], "iters": iters}


def corrected(r: OpRecord, sync_s: float) -> float:
    """An op record's time less one round trip, at least 0. Glue keeps its
    time."""
    if r.kind == "glue":
        return r.seconds
    return max(0.0, r.seconds - sync_s)


def plain_step(fn) -> float:
    """Wall time of one uninstrumented call of ``fn``."""
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(_arrays(fn()))
    return time.perf_counter() - t0


_NUM = re.compile(r"\.\d+(?=\.|$)")


def pattern(path: str) -> str:
    """Module path with block indices folded: layers.3.attention.qkv ->
    layers.N.attention.qkv."""
    return _NUM.sub(".N", path)


def _group(records, key, sync_s: float) -> list[dict]:
    total = sum(r.seconds for r in records) or 1.0
    total_c = sum(corrected(r, sync_s) for r in records) or 1.0
    out: dict = {}
    for r in records:
        k = key(r)
        g = out.setdefault(k, {"key": k, "calls": 0, "seconds": 0.0,
                               "corrected_seconds": 0.0})
        g["calls"] += 1
        g["seconds"] += r.seconds
        g["corrected_seconds"] += corrected(r, sync_s)
    rows = sorted(out.values(), key=lambda g: -g["seconds"])
    for g in rows:
        g["share"] = g["seconds"] / total
        g["corrected_share"] = g["corrected_seconds"] / total_c
    return rows


def summarize(records: list[OpRecord], instrumented: list[float],
              plain: list[float], sync_s: float = 0.0) -> dict:
    """Aggregates for the profile JSON. ``instrumented`` and ``plain`` are
    step wall times, ``records`` one instrumented step's records and
    ``sync_s`` the round trip measured in the same run. The exit ratio is
    ``corrected_ops_over_plain``. Shares come raw and corrected."""
    booked = sum(r.seconds for r in records)
    ops_s = sum(r.seconds for r in records if r.kind != "glue")
    ops_c = sum(corrected(r, sync_s) for r in records if r.kind != "glue")
    plain_med = statistics.median(plain)
    return {
        "step": {
            "plain_median_s": plain_med,
            "plain_s": plain,
            "instrumented_s": instrumented,
            "booked_s": booked,
            "ops_s": ops_s,
            "glue_s": booked - ops_s,
            "sync_s": sync_s,
            "ops_corrected_s": ops_c,
            "instrumented_over_plain": booked / plain_med,
            "corrected_ops_over_plain": ops_c / plain_med,
        },
        "by_class": _group(records, lambda r: f"{r.kind}:{r.name}:{r.codec}"
                           if r.kind != "glue" else "glue", sync_s),
        "by_pattern": _group(records, lambda r: f"{r.kind}:{pattern(r.path)}"
                             f":{r.name}", sync_s),
        "records": [asdict(r) for r in records],
    }


def sdpa_peak(seq: int, heads: int, head_dim: int, *, batch: int = 1,
              dtype=mx.bfloat16, fn=None) -> dict:
    """Peak memory of one attention call above its inputs, against the size
    of the score matrix it would materialize."""
    fn = fn or ops.sdpa
    q, k, v = (mx.random.normal((batch, heads, seq, head_dim)).astype(dtype)
               for _ in range(3))
    mx.eval(q, k, v)
    # eval can return before the temporaries it made are released, and a
    # release during the call would read as a peak below the inputs.
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = fn(q, k, v)
    mx.eval(out)
    secs = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    item = mx.zeros((1,), dtype).itemsize
    return {
        "seq": seq, "heads": heads, "head_dim": head_dim, "batch": batch,
        "dtype": str(dtype).replace("mlx.core.", ""),
        "peak_over_inputs_bytes": peak - base,
        "output_bytes": out.nbytes,
        "score_bytes": batch * heads * seq * seq * item,
        "seconds": secs,
    }
