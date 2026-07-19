"""Per-token adaptive MoE fan-out for CPU-placed expert stacks.

The router's top-k selection spends the same expert-read budget on every
token, but routing weight is often concentrated: when the top few experts
carry nearly all the gate mass, the tail experts cost bytes (page-cache or
SSD reads on offloaded models) for almost no output contribution.

``--moe-expert-mass P`` keeps, per token, the smallest weight-sorted prefix
of the selected experts whose share of the gate mass reaches ``P`` and drops
the rest. Dropped slots are not removed - shapes stay (tokens, k) - their
index is overwritten with the token's top-1 expert and their weight zeroed,
so every downstream consumer works unchanged while the unique-id set
shrinks: the decode feeder and prefetcher dedupe ids before staging, and a
duplicated row in the CPU gather re-reads an already-hot page.

``--moe-expert-probe`` is the lossless companion: selection runs at the
trained k while the same hook records how many experts each token needed to
reach a grid of candidate P values, reported as a table at exit.

Install sites (offloaded MoE blocks only, mirroring the fixed-k override in
loader.install_moe_experts_override):
- DeepSeek-family blocks route through a gate submodule returning
  ``(inds, weights)`` - one generic gate subclass covers deepseek_v4
  (hash-routed layers included), deepseek_v2/v3, glm_moe_dsa and glm4_moe.
- qwen3_moe / qwen3_next / minimax / minimax_m3 select inline in the block
  forward - each gets a subclass with the stock forward plus the filter
  inserted between selection and the expert gather.
- hunyuan's block is already class-swapped at load (_patch_hunyuan_norm_topk);
  it calls back into _apply_expert_controls when the install marked it.
- qwen3-next-shaped blocks the loader fused at load (_FusedKQuantMoeBlock in
  modules.py) hook inside the fused forward, and their non-fused fallback
  routes through qwen3_next_moe_forward below.
"""

from __future__ import annotations

import atexit
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_PROBE_GRID_DEFAULT = (0.7, 0.8, 0.82, 0.85, 0.87, 0.9, 0.95, 0.99)


@lru_cache(maxsize=8)
def _mass_filter_fn(p: float):
    # Plain compile (not shapeless): CumSum cannot infer shapeless output
    # shapes. Decode traces once per (tokens, k); prefill retraces per chunk
    # shape, which is noise next to the expert GEMMs.
    @mx.compile
    def _filter(inds, weights):
        order = mx.argsort(-weights, axis=-1)
        w = mx.take_along_axis(weights, order, axis=-1)
        e = mx.take_along_axis(inds, order, axis=-1)
        total = w.sum(axis=-1, keepdims=True)
        prefix = mx.cumsum(w, axis=-1) - w  # mass strictly before each slot
        keep = prefix < p * total  # slot 0 always kept
        e = mx.where(keep, e, e[..., :1])  # dropped -> the token's top-1 id
        w = mx.where(keep, w, mx.zeros_like(w))
        # Survivors are rescaled to the original sum, so the routed branch
        # keeps its trained scale (arch renorm/scaling already ran).
        w = w * (total / (w.sum(axis=-1, keepdims=True) + 1e-20))
        return e, w

    return _filter


def _mass_filter(inds, weights, p: float):
    """Keep the smallest weight-sorted expert prefix reaching mass share
    ``p``; dropped slots duplicate the top-1 id at weight 0. Output stays
    (tokens, k), weight-sorted (consumers weight the gather by the aligned
    array, so order is immaterial)."""
    return _mass_filter_fn(float(p))(inds, weights)


@mx.compile
def _probe_counts(weights, grid):
    """Experts needed per token to reach each mass share in ``grid``, and
    the mass fraction actually dropped at that cut. Returns
    (counts (..., G) int32, dropped (..., G))."""
    w = -mx.sort(-weights, axis=-1)
    total = w.sum(axis=-1, keepdims=True) + 1e-20
    prefix = mx.cumsum(w, axis=-1) - w
    keep = prefix[..., None, :] < grid[..., :, None] * total[..., None]
    counts = keep.sum(axis=-1).astype(mx.int32)
    kept = mx.where(keep, w[..., None, :], mx.zeros_like(w[..., None, :]))
    dropped = 1.0 - kept.sum(axis=-1) / total
    return counts, dropped


class _ProbeBucket:
    """Fan-out accumulator for one phase (decode or prefill)."""

    def __init__(self, g: int):
        self.k = 0
        self.hist = None  # (G, k+1) tokens-needing-count histogram
        self.dropped = np.zeros(g, dtype=np.float64)
        self.tokens = 0


class ExpertProbe:
    """Accumulates fan-out counterfactuals from router weights. Recording is
    lazy (no evals added to the decode critical path); pending arrays are
    batch-evaluated every ``_FLUSH_EVERY`` records and folded into numpy.
    Decode and prefill tokens accumulate separately: the mass filter is a
    decode lever, and a long prompt would otherwise drown the decode
    distribution the table is meant to size P against."""

    _FLUSH_EVERY = 64

    def __init__(self, grid=None):
        self.grid = tuple(grid) if grid else _PROBE_GRID_DEFAULT
        self._grid_mx = mx.array(self.grid, dtype=mx.float32)
        self._pending = []
        g = len(self.grid)
        self._buckets = {"decode": _ProbeBucket(g), "prefill": _ProbeBucket(g)}
        self._reported = False

    def record(self, li: int, weights) -> None:
        counts, dropped = _probe_counts(weights.astype(mx.float32), self._grid_mx)
        k = int(weights.shape[-1])
        phase = "decode" if weights.size // k == 1 else "prefill"
        self._pending.append((phase, k, counts, dropped))
        if len(self._pending) >= self._FLUSH_EVERY:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        mx.eval(*(a for _, _, c, d in pending for a in (c, d)))
        g = len(self.grid)
        for phase, k, counts, dropped in pending:
            b = self._buckets[phase]
            if k > b.k:
                hist = np.zeros((g, k + 1), dtype=np.int64)
                if b.hist is not None:
                    hist[:, : b.hist.shape[1]] = b.hist
                b.hist, b.k = hist, k
            c = np.array(counts).reshape(-1, g)
            d = np.array(dropped, dtype=np.float64).reshape(-1, g)
            for gi in range(g):
                b.hist[gi] += np.bincount(c[:, gi], minlength=b.k + 1)
            b.dropped += d.sum(axis=0)
            b.tokens += c.shape[0]

    def report(self) -> None:
        if self._reported:
            return
        self._reported = True
        self._flush()
        for phase in ("decode", "prefill"):
            b = self._buckets[phase]
            if not b.tokens:
                continue
            print(
                f"[experts] {phase} fan-out probe over {b.tokens} "
                f"token-layer router calls (selection k={b.k}):"
            )
            print("  P      avg   p10  p50  p90  expert reads  dropped mass")
            slots = np.arange(b.k + 1)
            for gi, p in enumerate(self.grid):
                hist = b.hist[gi]
                n = hist.sum()
                avg = float((hist * slots).sum()) / n
                cum = np.cumsum(hist)
                p10 = int(np.searchsorted(cum, 0.1 * n))
                p50 = int(np.searchsorted(cum, 0.5 * n))
                p90 = int(np.searchsorted(cum, 0.9 * n))
                dropped = b.dropped[gi] / b.tokens
                print(
                    f"  {p:<5.2f}  {avg:4.1f}  {p10:3d}  {p50:3d}  {p90:3d}  "
                    f"{avg / b.k:11.0%}  {dropped:11.2%}"
                )


def _apply_expert_controls(mod, inds, weights):
    """Shared post-selection hook: probe recording and/or mass filtering,
    driven by attrs the installer set on ``mod``."""
    probe = getattr(mod, "_kq_expert_probe", None)
    if probe is not None:
        probe.record(getattr(mod, "_kq_li", -1), weights)
    p = getattr(mod, "_kq_expert_mass", None)
    if p is not None:
        inds, weights = _mass_filter(inds, weights, p)
    return inds, weights


_GATE_CLASS_CACHE: dict = {}
_BLOCK_CLASS_CACHE: dict = {}


def _expert_ctl_gate_class(cls):
    sub = _GATE_CLASS_CACHE.get(cls)
    if sub is None:

        class _ExpertCtlGate(cls):
            def __call__(self, x, *args, **kwargs):
                # Extra args pass through untouched (deepseek_v4 hands the
                # gate its input_ids for the hash-routed layers).
                inds, weights = super().__call__(x, *args, **kwargs)
                return _apply_expert_controls(self, inds, weights)

        _ExpertCtlGate.__name__ = cls.__name__ + "_ExpertCtl"
        _GATE_CLASS_CACHE[cls] = sub = _ExpertCtlGate
    return sub


def _block_class(cls, forward):
    """Cached subclass of ``cls`` whose ``__call__`` is ``forward(self, x)`` -
    the arch's stock MoE forward with the expert-controls hook inserted at the
    selection seam. The ``_ExpertCtl`` name suffix is what re-installs key on."""
    sub = _BLOCK_CLASS_CACHE.get(cls)
    if sub is None:
        sub = type(cls.__name__ + "_ExpertCtl", (cls,), {"__call__": forward})
        _BLOCK_CLASS_CACHE[cls] = sub
    return sub


def _qwen3_moe_forward(mod, x):
    # Stock Qwen3MoeSparseMoeBlock forward (mlx-lm 0.31) plus the hook
    # between selection and the expert gather. Qwen3NextSparseMoeBlock
    # shares the selection but adds a gated shared expert and sharding.
    gates = mod.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = mod.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mod.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    inds, scores = _apply_expert_controls(mod, inds, scores)
    y = mod.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def qwen3_next_moe_forward(mod, x):
    """Stock qwen3-next-shaped MoE forward (router + SwitchGLU + sigmoid-gated
    shared expert) with the expert-controls hook at the selection seam. Shared
    by the qwen3_next class swap and the fused-block fallback in modules.py
    (whose eligibility check asserts exactly this forward shape)."""
    from mlx.nn.layers.distributed import sum_gradients

    if getattr(mod, "sharding_group", None) is not None:
        x = sum_gradients(mod.sharding_group)(x)
    gates = mod.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = mod.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if mod.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    inds, scores = _apply_expert_controls(mod, inds, scores)
    y = mod.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    shared_y = mod.shared_expert(x)
    shared_y = mx.sigmoid(mod.shared_expert_gate(x)) * shared_y
    y = y + shared_y
    if getattr(mod, "sharding_group", None) is not None:
        y = mx.distributed.all_sum(y, group=mod.sharding_group)
    return y


def _minimax_forward(mod, x):
    from mlx.nn.layers.distributed import sum_gradients

    if mod.sharding_group is not None:
        x = sum_gradients(mod.sharding_group)(x)
    gates = mod.gate(x.astype(mx.float32))
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + mod.e_score_correction_bias
    k = mod.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)
    inds, scores = _apply_expert_controls(mod, inds, scores)
    y = mod.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    if mod.sharding_group is not None:
        y = mx.distributed.all_sum(y, group=mod.sharding_group)
    return y


def _minimax_m3_forward(mod, x):
    gates = mod.gate(x.astype(mx.float32))
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + mod.e_score_correction_bias
    k = mod.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(orig_scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    weights = (weights * mod.routed_scaling_factor).astype(x.dtype)
    inds, weights = _apply_expert_controls(mod, inds, weights)
    y = mod.switch_mlp(x, inds)
    y = (y * weights[..., None]).sum(axis=-2)
    return y + mod.shared_experts(x)


# Block name -> the arch's stock forward with the expert-controls hook at the
# selection seam; _install swaps the block's class for the cached _ExpertCtl
# subclass carrying that forward.
_INLINE_SWAPS = {
    "Qwen3MoeSparseMoeBlock": _qwen3_moe_forward,
    "Qwen3NextSparseMoeBlock": qwen3_next_moe_forward,
    "MiniMaxSparseMoeBlock": _minimax_forward,
    "MiniMaxM3SparseMoeBlock": _minimax_m3_forward,
}


def _offloaded_moe_owners(model):
    """Yield (layer index, owning block) for every MoE block whose expert
    container install_expert_streaming wrapped (same walk as the fixed-k
    override in loader.py)."""
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.model.layers
    for li, layer in enumerate(layers):
        for owner in layer.modules():
            for child in owner.children().values():
                candidates = child if isinstance(child, (list, tuple)) else [child]
                if any(type(c).__name__.endswith("_CPUOffload") for c in candidates):
                    yield li, owner
                    break


def _install(model, *, mass=None, probe=None) -> int:
    hooked = 0
    unsupported: set = set()
    for li, owner in _offloaded_moe_owners(model):
        gate = getattr(owner, "gate", None)
        if isinstance(gate, nn.Module) and type(gate).__name__.endswith("_ExpertCtl"):
            target = gate  # already swapped by an earlier install
        elif (
            isinstance(gate, nn.Module)
            and not isinstance(gate, nn.Linear)
            and isinstance(getattr(gate, "top_k", None), int)
        ):
            target = gate
            gate.__class__ = _expert_ctl_gate_class(type(gate))
        elif type(owner).__name__ in ("_NormTopKMoE", "_FusedKQuantMoeBlock"):
            # hunyuan renorm patch / fused kquant MoE block: both forwards
            # call back into the hook when the attrs are set
            target = owner
        elif type(owner).__name__.endswith("_ExpertCtl"):
            target = owner  # already swapped by an earlier install
        elif type(owner).__name__ in _INLINE_SWAPS:
            target = owner
            owner.__class__ = _block_class(
                type(owner), _INLINE_SWAPS[type(owner).__name__])
        else:
            unsupported.add(type(owner).__name__)
            continue
        object.__setattr__(target, "_kq_expert_mass", mass)
        object.__setattr__(target, "_kq_expert_probe", probe)
        object.__setattr__(target, "_kq_li", li)
        hooked += 1
    if unsupported:
        print(
            "[stream] MoE expert controls skipped unsupported block(s): "
            + ", ".join(sorted(unsupported))
        )
    return hooked


def install_moe_expert_mass(model, p: float) -> int:
    """Route every offloaded MoE block through the adaptive mass filter at
    share ``p``. Lossy - outputs differ from the trained router. Returns the
    number of blocks hooked; raises on p outside (0, 1]."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"MoE expert-mass share must be in (0, 1], got {p}")
    hooked = _install(model, mass=float(p))
    if hooked:
        print(
            f"[stream] MoE expert-mass {p:g}: adaptive experts/token on "
            f"{hooked} offloaded MoE layers (lossy - outputs differ from "
            "the trained router)"
        )
    else:
        print(
            "[stream] MoE expert-mass found no supported offloaded MoE "
            "block - no effect"
        )
    return hooked


def install_moe_expert_probe(model, grid=None) -> int:
    """Record lossless fan-out counterfactuals on every offloaded MoE block
    and print the table at exit. Returns the number of blocks hooked."""
    probe = ExpertProbe(grid)
    hooked = _install(model, probe=probe)
    if hooked:
        atexit.register(probe.report)
        print(
            f"[stream] MoE expert probe: recording router fan-out on "
            f"{hooked} offloaded MoE layers (lossless; table prints at exit)"
        )
    else:
        print(
            "[stream] MoE expert probe found no supported offloaded MoE "
            "block - no effect"
        )
    return hooked
