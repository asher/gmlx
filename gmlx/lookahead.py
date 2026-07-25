"""Lookahead expert prediction for streamed MoE decode.

Layer L's MoE input is ``post_ln_L(h)`` where ``h`` is the post-attention
residual - the same vector layer L+1's router will see one sublayer later,
normalized with different gains (the residual moves by only the routed-MLP
delta in between). Running L+1's router on it therefore predicts L+1's
top-k well before L+1 executes (measured 71.6% recall on GLM-5.2
vs 41.3% for previous-token routing), and the prediction can drive expert
reads that overlap a full layer of compute. Predictions only move bytes;
routing math is untouched, so output is bit-identical.

Two input variants, selected by ``GMLX_DECODE_LOOKAHEAD_NORM``:
- ``ratio``: rescale L's normed input by ``post_ln_{L+1}(1)/post_ln_L(1)``
  elementwise - exact conversion between the two RMSNorm applications
  (RMSNorm gains are scale-independent; ``norm(ones)`` is the effective
  gain vector for both plain and Gemma-style norms).
- ``raw``: feed L's normed input as-is.

``GMLX_DECODE_LOOKAHEAD_PROBE=1`` installs the lossless recall probe: predictions
are recorded and compared against each layer's actual routing, with a
previous-token baseline for reference, and a table prints at exit. No
reads are issued - the probe exists to decide whether the prefetch is
worth building for a given model before any bytes move.
"""

from __future__ import annotations

import atexit
import time

import mlx.core as mx
import numpy as np

from .envflags import env_bool, env_choice, env_float, env_int

_VARIANTS = ("ratio", "raw")

# Sub-split of the loader's per-token ``la`` phase bucket
# (GMLX_DECODE_PHASE_STATS=1): ``build`` = replica prediction graph
# construction, ``sync`` = the shared ``mx.eval`` (the per-layer GPU
# segment wall). The numpy/gate tail is derived at dump time as
# la - build - sync. Read by loader._phase_dump.
_LA_PHASE = (
    {"build": 0.0, "sync": 0.0}
    if env_bool("GMLX_DECODE_PHASE_STATS", False)
    else None
)


class RankGate:
    """Online per-(layer, rank) reliability gate for prestage submissions.
    Tracks how often each prediction rank lands in the layer's actual
    routing (an EMA over decode calls - a pure function of the token
    stream, so gating never makes arena residency timing-dependent) and
    trims each prestage to the rank prefix whose measured hit rate clears
    ``GMLX_DECODE_LOOKAHEAD_MIN_P``. Ranks start optimistic (EMA 1.0), the
    full prediction width keeps being observed even while gated, and the
    ranking head stays reliable while the tail is where the wasted reads
    live - so the gate converts the waste tax into overlap without
    touching what the router computes."""

    _ALPHA = 1 / 64

    def __init__(self, min_p: float):
        self.min_p = min_p
        # (li, depth) -> per-rank hit EMA / last predicted row. Depth-2
        # predictions of a layer are scored apart from depth-1 ones: two
        # residual sublayers of drift make them a different reliability
        # population, and the gate must be able to trim them harder.
        self._ema: dict[tuple, np.ndarray] = {}
        self._last: dict[tuple, np.ndarray] = {}

    def note(self, li: int, pred: np.ndarray, depth: int = 1) -> None:
        rows = pred.reshape(-1, pred.shape[-1])
        if rows.shape[0] == 1:  # decode rows only; prefill passes ungated
            self._last[(li, depth)] = rows[0]

    def observe(self, li: int, actual: np.ndarray) -> None:
        keys = [k for k in self._last if k[0] == li]
        if not keys:
            return
        rows = actual.reshape(-1, actual.shape[-1])
        for key in keys:
            last = self._last.pop(key)
            if rows.shape[0] != 1:
                continue
            hits = np.isin(last, rows[0]).astype(np.float64)
            ema = self._ema.get(key)
            if ema is None or len(ema) != len(hits):
                ema = np.ones(len(hits), dtype=np.float64)
            ema += self._ALPHA * (hits - ema)
            self._ema[key] = ema

    def k(self, li: int, width: int, depth: int = 1) -> int:
        ema = self._ema.get((li, depth))
        if ema is None:
            return width
        k = 0
        for r in range(min(width, len(ema))):
            if ema[r] < self.min_p:
                break
            k += 1
        return k

    def report(self) -> None:
        if not self._ema or not getattr(self, "_stats_verbose", True):
            return
        for depth in sorted({d for _, d in self._ema}):
            emas = {li: e for (li, d), e in self._ema.items() if d == depth}
            widths = [len(e) for e in emas.values()]
            w = max(set(widths), key=widths.count)
            stack = np.stack([e for e in emas.values() if len(e) == w])
            ks = sorted(self.k(li, len(e), depth) for li, e in emas.items())
            tag = "" if depth == 1 else f" d{depth}"
            print(
                f"[lookahead] rank gate{tag} (hit EMA per prediction rank, "
                f"min_p={self.min_p:g}): "
                + "  ".join(
                    f"r{r} {v:.0%}" for r, v in enumerate(stack.mean(0)))
            )
            print(
                f"[lookahead] rank gate{tag} per-layer K: "
                f"median {ks[len(ks) // 2]}, "
                f"min {ks[0]}, max {ks[-1]} of {w}"
            )


def _norm_gains(norm) -> mx.array | None:
    """Effective gain vector of an RMSNorm-family module: ``norm(ones)``.
    ``rms(ones) == 1`` so this returns the multiplicative gains for both
    plain (``w``) and Gemma-style (``1 + w``) variants."""
    w = getattr(norm, "weight", None)
    if w is None or getattr(w, "ndim", 0) != 1:
        return None
    return norm(mx.ones_like(w))


def _gate_module_select(gate):
    """Ranked-ids selection through a DeepSeek-family gate submodule
    (deepseek_v2/v3/v4, glm_moe_dsa, glm4_moe). Calls the unwrapped
    forward when an ExpertCtl probe/mass-filter subclass is installed so
    its stats see only real routing. Monotonic post-scales (norm_topk_prob,
    routed_scaling_factor) preserve the weight order the ranking uses."""
    cls = type(gate)
    call = cls.__mro__[1].__call__ if cls.__name__.endswith("_ExpertCtl") else cls.__call__

    def select(x):
        inds, weights = call(gate, x)
        order = mx.argsort(-weights, axis=-1)
        return mx.take_along_axis(inds, order, axis=-1)

    return select


def _sigmoid_bias_select(mod):
    """Ranked-ids selection for MiniMax/MiniMax-M3-shaped blocks: sigmoid
    router plus selection bias, ranked by the selection score (score+bias),
    mirroring the stock forward's argpartition seam."""

    def select(x):
        choice = mx.sigmoid(mod.gate(x.astype(mx.float32)))
        choice = choice + mod.e_score_correction_bias
        k = mod.num_experts_per_tok
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        ch = mx.take_along_axis(choice, inds, axis=-1)
        order = mx.argsort(-ch, axis=-1)
        return mx.take_along_axis(inds, order, axis=-1)

    return select


def _softmax_select(mod):
    """Ranked-ids selection for qwen3_moe / qwen3_next-shaped blocks."""

    def select(x):
        gates = mx.softmax(mod.gate(x), axis=-1, precise=True)
        k = mod.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        sc = mx.take_along_axis(gates, inds, axis=-1)
        order = mx.argsort(-sc, axis=-1)
        return mx.take_along_axis(inds, order, axis=-1)

    return select


_SIGMOID_BIAS_BLOCKS = ("MiniMaxSparseMoeBlock", "MiniMaxM3SparseMoeBlock")
_SOFTMAX_BLOCKS = ("Qwen3MoeSparseMoeBlock", "Qwen3NextSparseMoeBlock")


def _base_block_name(owner) -> str:
    name = type(owner).__name__
    for suffix in ("_ExpertCtl",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _router_fn_for(owner):
    """Selection closure for ``owner`` (a MoE block), or None when the
    arch's routing seam is not recognized."""
    from .moe_experts import _gate_submodule

    gate = _gate_submodule(owner)
    if gate is not None:
        return _gate_module_select(gate)
    name = _base_block_name(owner)
    if name in _SIGMOID_BIAS_BLOCKS:
        return _sigmoid_bias_select(owner)
    if name in _SOFTMAX_BLOCKS:
        return _softmax_select(owner)
    return None


class _LayerPredictor:
    """Predicts layer ``dst_li``'s ranked top-k from layer ``src_li``'s MoE
    input. ``ratio`` converts between the two layers' norm gains; None
    (missing/degenerate norms) restricts the predictor to the raw variant."""

    def __init__(self, src_li: int, dst_li: int, router_fn, ratio,
                 depth: int = 1):
        self.src_li = src_li
        self.dst_li = dst_li
        self._router_fn = router_fn
        self._ratio = ratio
        self.depth = depth  # MoE-layer distance src -> dst
        self.dead = False  # set on first router_fn failure; predictor off

    def variants(self, probing: bool) -> tuple[str, ...]:
        if probing:
            return _VARIANTS if self._ratio is not None else ("raw",)
        v = env_choice("GMLX_DECODE_LOOKAHEAD_NORM", "ratio", _VARIANTS)
        if v == "ratio" and self._ratio is None:
            v = "raw"
        return (v,)

    def predict(self, x, variant: str):
        """Lazy ranked expert ids (shape ``indices``-like) for ``dst_li``."""
        if variant == "ratio":
            x = x * self._ratio.astype(x.dtype)
        return self._router_fn(x)


class LookaheadProbe:
    """Recall accumulator: predicted-vs-actual per destination layer, per
    variant, at every prefix K, plus a previous-token routing baseline.
    All comparisons run on host arrays the wrapper already materialized."""

    _KS = (1, 2, 4, 6, 8)

    def __init__(self):
        # (dst_li, variant) -> {K: [hits, total]}
        self._recall: dict = {}
        self._pending: dict = {}  # dst_li -> {variant: np ids (rows, k)}
        self._prev: dict = {}  # li -> previous token's unique ids
        self._prev_recall = [0.0, 0.0]  # hits, total
        self._token_layers = 0
        self._reported = False

    def note(self, dst_li: int, preds: dict) -> None:
        # Merge: with depth > 1 the same destination is predicted by more
        # than one source hook (labels are variant@dN, so keys never clash).
        self._pending.setdefault(dst_li, {}).update(preds)

    def actual(self, li: int, ids_np: np.ndarray) -> None:
        rows = ids_np.reshape(-1, ids_np.shape[-1])
        preds = self._pending.pop(li, None)
        if preds is not None:
            for variant, pred in preds.items():
                prows = pred.reshape(-1, pred.shape[-1])
                if prows.shape[0] != rows.shape[0]:
                    continue
                width = prows.shape[1]
                ks = [k for k in self._KS if k < width] + [width]
                for r in range(rows.shape[0]):
                    actual = np.unique(rows[r])
                    self._token_layers += 1
                    for k in ks:
                        cell = self._recall.setdefault(
                            (li, variant), {}
                        ).setdefault(k, [0.0, 0.0])
                        cell[0] += np.isin(prows[r, :k], actual).sum()
                        cell[1] += len(actual)
        if rows.shape[0] == 1:
            actual = np.unique(rows[0])
            prev = self._prev.get(li)
            if prev is not None:
                self._prev_recall[0] += np.isin(prev, actual).sum()
                self._prev_recall[1] += len(actual)
            self._prev[li] = actual

    def report(self) -> None:
        if self._reported or not self._recall:
            return
        self._reported = True
        variants = sorted({v for _, v in self._recall})
        ks = sorted({k for cells in self._recall.values() for k in cells})
        print("[lookahead] router recall probe (predicted@K vs actual routing):")
        for variant in variants:
            layer_cells = {
                li: cells
                for (li, v), cells in self._recall.items()
                if v == variant
            }
            overall = []
            for k in ks:
                h = sum(c[k][0] for c in layer_cells.values() if k in c)
                t = sum(c[k][1] for c in layer_cells.values() if k in c)
                overall.append(f"@{k} {h / t:.1%}" if t else f"@{k} -")
            print(f"  {variant:<6} overall: " + "  ".join(overall))
            kref = max(k for k in ks)
            per_layer = sorted(
                (c[kref][0] / c[kref][1], li)
                for li, c in layer_cells.items()
                if kref in c and c[kref][1]
            )
            if per_layer:
                vals = [v for v, _ in per_layer]
                lo_v, lo_li = per_layer[0]
                print(
                    f"  {variant:<6} per-layer @{kref}: median "
                    f"{vals[len(vals) // 2]:.1%}, min {lo_v:.1%} "
                    f"(layer {lo_li})"
                )
        if self._prev_recall[1]:
            print(
                "  prev-token baseline (same-layer routing reuse): "
                f"{self._prev_recall[0] / self._prev_recall[1]:.1%}"
            )


class LookaheadHook:
    """Per-module seam the offload wrapper calls at decode. Owns the one
    ``mx.eval`` batching the router read with the prediction, probe
    bookkeeping, and (when prefetching) the materialized prediction."""

    def __init__(self, li: int, predictors, probe, prefetch: bool,
                 gate: RankGate | None = None):
        self.li = li
        if predictors is None:
            predictors = []
        elif not isinstance(predictors, (list, tuple)):
            predictors = [predictors]
        self.predictors = [p for p in predictors if p is not None]
        self.probe = probe
        self.prefetch = prefetch
        self.gate = gate

    @property
    def predictor(self):
        """The depth-1 (next-MoE-layer) predictor, or None."""
        for p in self.predictors:
            if p.depth == 1:
                return p
        return None

    def on_call(self, x, indices) -> dict:
        """Evaluate ``indices`` (the fence the caller needed anyway) plus
        every live lookahead prediction in one sync. Returns a dict of
        predicted ranked ids per destination layer when prefetching, each
        trimmed to its rank gate prefix; empty when there is nothing to
        prestage."""
        probing = self.probe is not None
        ph = _LA_PHASE
        t0 = time.perf_counter() if ph is not None else 0.0
        lazy: list = []  # (predictor, variant, lazy array)
        for pred in self.predictors:
            if pred.dead:
                continue
            try:
                for variant in pred.variants(probing):
                    lazy.append((pred, variant, pred.predict(x, variant)))
            except Exception as exc:  # unsupported gate signature, bad dims
                pred.dead = True
                lazy = [t for t in lazy if t[0] is not pred]
                print(
                    f"[lookahead] predictor for layer {pred.dst_li} "
                    f"disabled: {type(exc).__name__}: {exc}"
                )
        if ph is not None:
            t1 = time.perf_counter()
            ph["build"] += t1 - t0
        try:
            mx.eval(indices, *[a for _, _, a in lazy])
        except Exception as exc:
            # Joint eval: the failing predictor is unattributable, so
            # disable every one this hook owns rather than loop forever.
            for pred, _, _ in lazy:
                pred.dead = True
            if lazy:
                print(
                    f"[lookahead] predictors at layer {self.li} disabled: "
                    f"{type(exc).__name__}: {exc}"
                )
            lazy = []
            mx.eval(indices)
        if ph is not None:
            ph["sync"] += time.perf_counter() - t1
        by_pred: dict = {}
        for pred, variant, arr in lazy:
            by_pred.setdefault(pred, {})[variant] = np.array(arr)
        actual_np = None
        if self.probe is not None or self.gate is not None:
            actual_np = np.array(indices)
        if self.gate is not None:
            self.gate.observe(self.li, actual_np)
        if self.probe is not None:
            for pred, pv in by_pred.items():
                if pred.depth == 1:
                    labeled = pv
                else:
                    labeled = {
                        f"{v}@d{pred.depth}": a for v, a in pv.items()
                    }
                self.probe.note(pred.dst_li, labeled)
            self.probe.actual(self.li, actual_np)
        out: dict = {}
        if self.prefetch:
            for pred, pv in by_pred.items():
                # With the probe co-installed both variants exist;
                # prefetch uses the configured one.
                chosen = pred.variants(False)[0]
                ids = pv.get(chosen)
                if ids is None:
                    ids = next(iter(pv.values()))
                if self.gate is not None:
                    # Full width is noted (gated-out ranks keep being
                    # scored and can re-qualify); only the submission is
                    # trimmed.
                    self.gate.note(pred.dst_li, ids, pred.depth)
                    k = self.gate.k(pred.dst_li, ids.shape[-1], pred.depth)
                    if k <= 0:
                        continue
                    if k < ids.shape[-1]:
                        ids = ids[..., :k]
                out[pred.dst_li] = ids
        return out


def install_lookahead(model, layers, *, probe: bool = False,
                      prefetch: bool = False,
                      stats_verbose: bool | None = None) -> int:
    """Wire lookahead hooks onto every offloaded MoE module whose next
    offloaded MoE layer has a recognizable router. The last MoE layer (and
    any layer followed by an unsupported arch) still gets a hook when
    probing, so its actual routing anchors the recall comparison. Returns
    the number of layers with a live predictor.

    ``GMLX_DECODE_LOOKAHEAD_DEPTH`` (default 1, max 3) adds predictors for
    the MoE layers 2..D steps ahead as well: a read predicted two layers
    early gets two layers of compute to complete instead of one, at the
    cost of extra residual drift between the source input and the
    destination router. Deeper predictions are probed, gated, and
    prestaged independently of the next-layer ones."""
    from .moe_experts import _offloaded_moe_owners

    owners: dict[int, object] = {}
    for li, owner in _offloaded_moe_owners(model):
        owners.setdefault(li, owner)
    wrapped: dict[int, list] = {}
    for li, layer in enumerate(layers):
        if li not in owners:
            continue
        mods = [
            m
            for m in layer.modules()
            if type(m).__name__.endswith("_CPUOffload")
        ]
        if mods:
            wrapped[li] = mods
    lis = sorted(wrapped)
    shared_probe = LookaheadProbe() if probe else None
    gate = (
        RankGate(env_float("GMLX_DECODE_LOOKAHEAD_MIN_P", 0.5))
        if prefetch else None)
    if gate is not None and stats_verbose is not None:
        gate._stats_verbose = stats_verbose
    depth = max(1, min(3, env_int("GMLX_DECODE_LOOKAHEAD_DEPTH", 1)))
    n_pred = 0
    unsupported: set = set()
    for pos, li in enumerate(lis):
        predictors = []
        src_norm = getattr(layers[li], "post_attention_layernorm", None)
        src_g = _norm_gains(src_norm) if src_norm is not None else None
        for d in range(1, depth + 1):
            if pos + d >= len(lis):
                break
            dst_li = lis[pos + d]
            router_fn = _router_fn_for(owners[dst_li])
            if router_fn is None:
                unsupported.add(_base_block_name(owners[dst_li]))
                continue
            dst_norm = getattr(
                layers[dst_li], "post_attention_layernorm", None
            )
            ratio = None
            dst_g = _norm_gains(dst_norm) if dst_norm is not None else None
            if src_g is not None and dst_g is not None:
                src_f = src_g.astype(mx.float32)
                ratio = mx.where(
                    mx.abs(src_f) < 1e-6,
                    mx.ones_like(src_f),
                    dst_g.astype(mx.float32) / src_f,
                )
            predictors.append(
                _LayerPredictor(li, dst_li, router_fn, ratio, d))
        if predictors:
            n_pred += 1
        elif shared_probe is None and gate is None:
            # Without a probe or gate a terminal layer has nothing to do;
            # with a gate it still observes its own routing so predictions
            # targeting it keep being scored.
            continue
        hook = LookaheadHook(li, predictors, shared_probe, prefetch, gate)
        for m in wrapped[li]:
            object.__setattr__(m, "_kq_lookahead", hook)
    if shared_probe is not None and n_pred:
        atexit.register(shared_probe.report)
    if gate is not None and n_pred:
        atexit.register(gate.report)
    if unsupported:
        print(
            "[lookahead] no router replica for block(s): "
            + ", ".join(sorted(unsupported))
        )
    return n_pred
