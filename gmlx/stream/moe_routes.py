"""Record and replay MoE expert routes through the expert-controls seam.

A route set is the per-layer top-k expert ids of one forward, shaped
[n_moe, B, T, k]. ``RouteRecorder`` collects them lazily at the selection
seam of every MoE block; ``RouteReplay`` feeds them back so a later forward
over the same positions selects exactly those experts while the block
computes its own mixing weights for them (its gate probabilities gathered
at the replayed ids, then its usual renormalization). Routes carry ids only,
so a replay stays valid on a model whose router has since moved, and a
forward that replays its own live routes reproduces itself bit for bit on
the eager selection paths.

Both controls install on every MoE block of the model, resident or
streamed, through the same targets the expert-mass and probe installers
use (stream/moe_experts.py). Replay needs the block to recompute weights
for arbitrary ids: the inline-swapped and call-back blocks carry that
inline, DeepSeek-shaped gate submodules need an adapter, registered here
for the mlx-lm ``group_expert_select`` gates and supplied as a
``_kq_route_weights(x, inds)`` method by the gmlx-owned gates. A block
without one makes ``install_moe_route_replay`` refuse, since a partial
replay would misalign every layer after the gap.

Position bookkeeping is the caller's: set ``RouteReplay.offset`` to the
first position of each chunk before its forward (``advance`` moves it by
the tokens consumed in that step), and keep the batch row count of the replayed
forward equal to the recorded one.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from gmlx.stream.moe_experts import (
    _apply_expert_controls,  # noqa: F401  (the seam this module drives)
    _hook_target,
    _moe_owners,
)


class RouteRecorder:
    """Collects the ids every hooked MoE block selects, per layer, across
    the forwards between two ``take`` calls. Each copy is evaluated with
    the forward that consumes its ids, so recording adds no sync to the
    forward."""

    def __init__(self):
        self.layers: list[int] = []
        self._parts: dict[int, list] = {}

    def record(self, li: int, inds):
        """Keep a copy of layer ``li``'s ids and return ``inds`` tied to
        it, so the forward that consumes the returned ids evaluates the
        copy as well. The ids are a view of the selection's [..., E] sort
        buffer, and a kept view, or a copy left unevaluated until
        ``take``, would hold that buffer for every layer until then."""
        c = mx.contiguous(inds)
        self._parts.setdefault(li, []).append(c)
        return mx.depends(inds, c)

    def take(self):
        """The recorded routes as int32 [n_moe, B, T, k], chunks joined along
        T in call order, then reset. Layers with no call are dropped from
        the stack, so check the length against ``layers`` on a model whose
        forward skips blocks."""
        out = []
        for li in self.layers:
            parts = self._parts.get(li)
            if not parts:
                continue
            mx.eval(*parts)
            arrs = [np.asarray(a).astype(np.int32) for a in parts]
            arrs = [a.reshape(-1, a.shape[-2], a.shape[-1]) if a.ndim >= 3
                    else a.reshape(1, -1, a.shape[-1]) for a in arrs]
            out.append(np.concatenate(arrs, axis=1))
        self._parts.clear()
        if not out:
            return np.zeros((0, 1, 0, 1), np.int32)
        return np.stack(out)


class RouteReplay:
    """Feeds a recorded route set back through the seam. ``routes`` is
    [n_moe, T, k] for one row or [n_moe, B, T, k]; ``layers`` lists the
    model layer index behind each leading entry, in order."""

    def __init__(self, routes, layers, offset: int = 0, n_experts: int | None = None):
        routes = np.asarray(routes)
        if n_experts is not None and routes.size and int(routes.max()) >= int(n_experts):
            raise ValueError(
                f"routes carry expert id {int(routes.max())}, the model has {int(n_experts)} experts")
        if routes.ndim == 3:
            routes = routes[:, None]
        if routes.ndim != 4:
            raise ValueError(
                f"routes must be [n_moe, T, k] or [n_moe, B, T, k], got {routes.shape}")
        if routes.shape[0] != len(layers):
            raise ValueError(
                f"routes carry {routes.shape[0]} layers, the layer list {len(layers)}")
        self.layers = list(layers)
        self.rows, self.length, self.k = routes.shape[1:]
        self._ids = {
            li: mx.array(routes[j].astype(np.int32)) for j, li in enumerate(self.layers)
        }
        self.offset = int(offset)

    def advance(self, n: int) -> None:
        self.offset += int(n)

    def ids_for(self, li: int, inds):
        """The replayed ids for layer ``li`` over the window this forward
        covers, shaped and typed like the block's own ``inds``."""
        ids = self._ids.get(li)
        if ids is None:
            raise ValueError(f"route replay carries no routes for layer {li}")
        k = inds.shape[-1]
        if k != self.k:
            raise ValueError(
                f"route replay recorded k={self.k}, layer {li} selects k={k}")
        if inds.ndim >= 3:
            rows, n = int(np.prod(inds.shape[:-2])), inds.shape[-2]
        else:
            rows, n = 1, inds.shape[0]  # flat (tokens, k) ids: one row
        if rows != self.rows:
            raise ValueError(
                f"route replay recorded {self.rows} rows, layer {li} sees {rows}")
        end = self.offset + n
        if end > self.length:
            raise ValueError(
                f"route replay window {self.offset}:{end} runs past the "
                f"recorded length {self.length}")
        return ids[:, self.offset:end].reshape(inds.shape).astype(inds.dtype)


# DeepSeek-shaped gates from mlx-lm: (module, class) -> weights at ids.
# Each mirrors the weight branch of the gate's own select function, so the
# replayed weights equal what the gate would have produced had it selected
# those ids itself.


def _group_select_weights(gate, x, inds):
    # mlx_lm deepseek_v3.group_expert_select: sigmoid scores, optional
    # renormalization, routed scaling; the group mask steers selection only.
    scores = mx.sigmoid((x @ gate.weight.T).astype(mx.float32))
    w = mx.take_along_axis(scores, inds, axis=-1)
    if gate.top_k > 1 and gate.norm_topk_prob:
        w = w / w.sum(axis=-1, keepdims=True)
    return w * gate.routed_scaling_factor


_GATE_WEIGHTS = {
    ("mlx_lm.models.deepseek_v3", "MoEGate"): _group_select_weights,
    ("mlx_lm.models.glm4_moe", "MoEGate"): _group_select_weights,
}


def _gate_weights_fn(gate):
    """The ``(gate, x, inds) -> weights`` adapter for a gate submodule, or
    None when no adapter exists for its class."""
    if hasattr(type(gate), "_kq_route_weights"):
        return lambda g, x, inds: g._kq_route_weights(x, inds)
    cls = type(gate)
    if cls.__name__.endswith("_ExpertCtl"):
        cls = cls.__mro__[1]
    return _GATE_WEIGHTS.get((cls.__module__, cls.__name__))


def moe_layers(model) -> list[int]:
    """Model layer indices that hold a MoE block, in forward order."""
    return [li for li, _ in _moe_owners(model)]


def moe_expert_counts(model) -> list[int | None]:
    """The number of experts behind each MoE layer of ``model``, in forward
    order: the leading dimension of the block's first switch linear, else
    its integer ``num_experts`` or ``n_routed_experts``, and None when
    neither is found."""
    out = []
    for _, owner in _moe_owners(model):
        n = None
        for m in owner.modules():
            w = getattr(m, "weight", None)
            if type(m).__name__.endswith("SwitchLinear") and isinstance(w, mx.array) and w.ndim == 3:
                n = int(w.shape[0])
                break
        if n is None:
            for attr in ("num_experts", "n_routed_experts"):
                v = getattr(owner, attr, None)
                if isinstance(v, int) and not isinstance(v, bool):
                    n = v
                    break
        out.append(n)
    return out


def _targets(model):
    supported, unsupported = [], []
    for li, owner in _moe_owners(model):
        target = _hook_target(owner)
        if target is None:
            unsupported.append((li, type(owner).__name__))
            continue
        object.__setattr__(target, "_kq_li", li)
        if target is not owner and getattr(target, "_kq_route_weights_fn", None) is None:
            fn = _gate_weights_fn(target)
            if fn is not None:
                object.__setattr__(target, "_kq_route_weights_fn", fn)
        supported.append((li, owner, target))
    return supported, unsupported


def install_moe_route_record(model, recorder: RouteRecorder | None = None) -> RouteRecorder:
    """Hook every supported MoE block to append its selected ids to
    ``recorder`` (a new one when None). Unsupported blocks are reported and
    left out of ``recorder.layers``."""
    recorder = recorder or RouteRecorder()
    supported, unsupported = _targets(model)
    for li, _, target in supported:
        object.__setattr__(target, "_kq_route_record", recorder)
        if li not in recorder.layers:
            recorder.layers.append(li)
    if unsupported:
        names = sorted({n for _, n in unsupported})
        print(
            "[stream] MoE route recording skipped unsupported block(s): "
            + ", ".join(names)
        )
    return recorder


def install_moe_route_replay(model, replay: RouteReplay) -> int:
    """Hook every MoE block to select ``replay``'s ids. Raises when a block
    cannot replay (no forward seam, or a gate without a weights adapter) or
    when the replay's layer list differs from the model's MoE layers.
    Returns the number of blocks hooked."""
    supported, unsupported = _targets(model)
    for li, owner, target in supported:
        if target is not owner and getattr(target, "_kq_route_weights_fn", None) is None:
            unsupported.append((li, type(target).__name__))
    if unsupported:
        raise ValueError(
            "route replay unsupported on MoE block(s): "
            + ", ".join(f"layer {li} {n}" for li, n in sorted(unsupported))
        )
    have = [li for li, _, _ in supported]
    if have != replay.layers:
        raise ValueError(
            f"route replay layers {replay.layers} do not match the model's "
            f"MoE layers {have}"
        )
    for _, _, target in supported:
        object.__setattr__(target, "_kq_route_replay", replay)
    return len(supported)


def clear_moe_route_controls(model) -> None:
    """Remove route recording and replay from every MoE block. The seam
    subclasses stay installed and are inert without the attrs."""
    for _, owner in _moe_owners(model):
        target = _hook_target(owner)
        if target is None:
            continue
        for attr in ("_kq_route_record", "_kq_route_replay"):
            if getattr(target, attr, None) is not None:
                object.__setattr__(target, attr, None)
