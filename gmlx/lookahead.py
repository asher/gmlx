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

import mlx.core as mx
import numpy as np

from .envflags import env_choice, env_float

_VARIANTS = ("ratio", "raw")


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
        self._ema: dict[int, np.ndarray] = {}  # li -> per-rank hit EMA
        self._last: dict[int, np.ndarray] = {}  # li -> last predicted row

    def note(self, li: int, pred: np.ndarray) -> None:
        rows = pred.reshape(-1, pred.shape[-1])
        if rows.shape[0] == 1:  # decode rows only; prefill passes ungated
            self._last[li] = rows[0]

    def observe(self, li: int, actual: np.ndarray) -> None:
        last = self._last.pop(li, None)
        if last is None:
            return
        rows = actual.reshape(-1, actual.shape[-1])
        if rows.shape[0] != 1:
            return
        hits = np.isin(last, rows[0]).astype(np.float64)
        ema = self._ema.get(li)
        if ema is None or len(ema) != len(hits):
            ema = np.ones(len(hits), dtype=np.float64)
        ema += self._ALPHA * (hits - ema)
        self._ema[li] = ema

    def k(self, li: int, width: int) -> int:
        ema = self._ema.get(li)
        if ema is None:
            return width
        k = 0
        for r in range(min(width, len(ema))):
            if ema[r] < self.min_p:
                break
            k += 1
        return k

    def report(self) -> None:
        if not self._ema:
            return
        widths = [len(e) for e in self._ema.values()]
        w = max(set(widths), key=widths.count)
        stack = np.stack([e for e in self._ema.values() if len(e) == w])
        ks = sorted(self.k(li, len(e)) for li, e in self._ema.items())
        print(
            "[lookahead] rank gate (hit EMA per prediction rank, "
            f"min_p={self.min_p:g}): "
            + "  ".join(f"r{r} {v:.0%}" for r, v in enumerate(stack.mean(0)))
        )
        print(
            f"[lookahead] rank gate per-layer K: median {ks[len(ks) // 2]}, "
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

    def __init__(self, src_li: int, dst_li: int, router_fn, ratio):
        self.src_li = src_li
        self.dst_li = dst_li
        self._router_fn = router_fn
        self._ratio = ratio
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
        self._pending[dst_li] = preds

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

    def __init__(self, li: int, predictor, probe, prefetch: bool,
                 gate: RankGate | None = None):
        self.li = li
        self.predictor = predictor
        self.probe = probe
        self.prefetch = prefetch
        self.gate = gate

    def on_call(self, x, indices) -> np.ndarray | None:
        """Evaluate ``indices`` (the fence the caller needed anyway) plus
        any lookahead prediction in one sync. Returns the predicted ranked
        ids for ``predictor.dst_li`` when prefetching - trimmed to the
        rank gate's reliable prefix - else None."""
        pred = self.predictor
        lazy: dict = {}
        if pred is not None and not pred.dead:
            probing = self.probe is not None
            try:
                for variant in pred.variants(probing):
                    lazy[variant] = pred.predict(x, variant)
                mx.eval(indices, *lazy.values())
            except Exception as exc:  # unsupported gate signature, bad dims
                pred.dead = True
                lazy = {}
                print(
                    f"[lookahead] predictor for layer {pred.dst_li} "
                    f"disabled: {type(exc).__name__}: {exc}"
                )
                mx.eval(indices)
        else:
            mx.eval(indices)
        preds_np = {v: np.array(a) for v, a in lazy.items()}
        actual_np = None
        if self.probe is not None or self.gate is not None:
            actual_np = np.array(indices)
        if self.gate is not None:
            self.gate.observe(self.li, actual_np)
        if self.probe is not None:
            if preds_np:
                self.probe.note(pred.dst_li, preds_np)
            self.probe.actual(self.li, actual_np)
        if self.prefetch and preds_np:
            # With the probe co-installed both variants exist; prefetch
            # uses the configured one.
            chosen = pred.variants(False)[0]
            out = preds_np.get(chosen)
            if out is None:
                out = next(iter(preds_np.values()))
            if self.gate is not None:
                # Full width is noted (gated-out ranks keep being scored
                # and can re-qualify); only the submission is trimmed.
                self.gate.note(pred.dst_li, out)
                k = self.gate.k(pred.dst_li, out.shape[-1])
                if k <= 0:
                    return None
                if k < out.shape[-1]:
                    out = out[..., :k]
            return out
        return None


def install_lookahead(model, layers, *, probe: bool = False,
                      prefetch: bool = False) -> int:
    """Wire lookahead hooks onto every offloaded MoE module whose next
    offloaded MoE layer has a recognizable router. The last MoE layer (and
    any layer followed by an unsupported arch) still gets a hook when
    probing, so its actual routing anchors the recall comparison. Returns
    the number of layers with a live predictor."""
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
    n_pred = 0
    unsupported: set = set()
    for pos, li in enumerate(lis):
        predictor = None
        if pos + 1 < len(lis):
            dst_li = lis[pos + 1]
            router_fn = _router_fn_for(owners[dst_li])
            if router_fn is None:
                unsupported.add(_base_block_name(owners[dst_li]))
            else:
                src_norm = getattr(
                    layers[li], "post_attention_layernorm", None
                )
                dst_norm = getattr(
                    layers[dst_li], "post_attention_layernorm", None
                )
                ratio = None
                src_g = _norm_gains(src_norm) if src_norm is not None else None
                dst_g = _norm_gains(dst_norm) if dst_norm is not None else None
                if src_g is not None and dst_g is not None:
                    src_f = src_g.astype(mx.float32)
                    ratio = mx.where(
                        mx.abs(src_f) < 1e-6,
                        mx.ones_like(src_f),
                        dst_g.astype(mx.float32) / src_f,
                    )
                predictor = _LayerPredictor(li, dst_li, router_fn, ratio)
                n_pred += 1
        if predictor is None and shared_probe is None and gate is None:
            # Without a probe or gate a terminal layer has nothing to do;
            # with a gate it still observes its own routing so predictions
            # targeting it keep being scored.
            continue
        hook = LookaheadHook(li, predictor, shared_probe, prefetch, gate)
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
