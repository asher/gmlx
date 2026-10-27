# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Model for IBM Granite Switch (GGUF arch ``graniteswitch``).

This is the runtime for llama.cpp's ``LLM_ARCH_GRANITE_SWITCH``
conversions (src/models/granite-switch.cpp). The base is Granite dense
with its four runtime multipliers. These parts differ:

  1. Q, K and V come from one fused ``attn_qkv``. Its Q and K rows are
     permuted for NORM rope, so rope runs ``traditional`` on the wire
     layout.
  2. Every projection carries a stack of LoRA pairs, one slot per
     adapter plus an all-zero slot 0. Each token adds the delta of its
     own slot: the mean of the slots of the control tokens at or before
     it, rounded half up, or 0 with none. The deltas reach the cached
     K and V.
  3. A control token embeds as its substitute token.

The slot rule is llama.cpp's router head without the softmax: its K lane
is +gain on a control token and -gain elsewhere, so the attended V is the
mean of the visible control slots to within e^-30. Here the router cache
keeps a 0/1 control lane and the slot lane per position, and the slot is
that mean exactly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from gmlx._vendor.mlx_lm.models.base import BaseModelArgs
from gmlx._vendor.mlx_lm.models.rope_utils import initialize_rope

from gmlx.cache.attention import (
    create_attention_mask,
    scaled_dot_product_attention,
)
from gmlx.cache.kv_cache import BatchKVCache, KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    logits_scaling: float
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    num_adapters: int
    max_lora_rank: int
    adapter_token_ids: List[int]
    adapter_substitute_ids: List[int]
    router_gain: float = 15.0
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True


class SwitchRouterCache(KVCache):
    """Per-position router lanes, ``(B, 1, T, 1)`` float32: K is 1.0 on a
    control token and 0.0 elsewhere, V is the control token's slot. A
    positional cache, so trim, batching and prefix restore carry it like
    the attention caches. It is never quantized or converted, and the KV
    layer policy skips it (``kv_policy_exempt``)."""

    kv_policy_exempt = True
    # Keeps mlx-vlm's hybrid and TurboQuant KV arms off the router lanes.
    preserve_auxiliary_kv_state = True
    # Two float32 lanes per token; serve's admission prices this.
    side_bytes_per_token = 8.0
    # Keys, values and offset are the whole state, so the decision reader's
    # prefix views and exact prefix snapshots treat it as a plain KVCache.
    plain_kv = True

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        return self

    def to_batch(self, left_padding):
        return SwitchRouterBatchCache(left_padding)

    def extract(self, idx):
        return _as_router(super().extract(idx), SwitchRouterCache)

    @classmethod
    def merge(cls, caches):
        return SwitchRouterBatchCache.merge(caches)

    def prefix_cache_merge(self, rows, prefix_lens):
        if all(isinstance(c, (KVCache, SwitchRouterCache)) for c in rows):
            return self.merge(rows)
        return None


class SwitchRouterBatchCache(BatchKVCache):
    """The batch form of :class:`SwitchRouterCache`."""

    kv_policy_exempt = True
    # Keeps mlx-vlm's hybrid and TurboQuant KV arms off the router lanes.
    preserve_auxiliary_kv_state = True
    # Two float32 lanes per token; serve's admission prices this.
    side_bytes_per_token = 8.0

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        return self

    def extract(self, idx):
        return _as_router(super().extract(idx), SwitchRouterCache)


def _as_router(cache, cls):
    out = cls()
    out.keys, out.values, out.offset = cache.keys, cache.values, cache.offset
    return out


try:
    import mlx_kquant as _kq
    from mlx_kquant.nn import KQuantLinear as _KQuantLinear
    _LORA_STACK = bool(getattr(_kq, "HAS_LORA_STACK", False))
except ImportError:                          # pragma: no cover
    _KQuantLinear = None
    _LORA_STACK = False


class SwitchLoRA(nn.Module):
    """The adapter stacks of one layer, one per projection, with the zero
    slot 0 dropped: ``<p>_a`` is ``(S - 1, in, r)`` and ``<p>_b`` is ``(S -
    1, r, out)``. The fused ``qkv`` projection joins the q, k and v ranks:
    its A holds them side by side and its B is block diagonal. The wire
    form (``<t>_a (S, r, in)``, ``<t>_b (S, out, r)`` per target) is packed
    by :meth:`Model.sanitize`."""

    TARGETS = ("q", "k", "v", "o", "gate", "up", "down")
    PROJECTIONS = ("qkv", "o", "gate", "up", "down")
    # Past this many rows the slot-sorted GatherMM beats the per-row
    # epilogue, which forms each row's x @ A once per output tile.
    EPILOGUE_MAX_ROWS = 32

    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.num_adapters
        for p, (n_in, n_out, rank) in lora_shapes(args).items():
            setattr(self, f"{p}_a", mx.zeros((n, n_in, rank)))
            setattr(self, f"{p}_b", mx.zeros((n, rank, n_out)))
        # Router slot -> stack entry; slot 0 skips the row.
        self._table = mx.arange(-1, n, dtype=mx.int32)

    def project(self, p: str, base: nn.Module, x: mx.array,
                route: Optional["SlotRoute"]) -> mx.array:
        """``base(x)`` plus each token's adapter delta for projection
        ``p``. On a K-quant base the delta rides the matmul's own epilogue
        at decode widths; otherwise it is two GatherMMs over the stack."""
        if route is None:
            return base(x)
        a, b = self[f"{p}_a"], self[f"{p}_b"]
        rows = x.size // x.shape[-1]
        if (_LORA_STACK and type(base) is _KQuantLinear
                and x.dtype in (mx.float16, mx.bfloat16)
                and rows <= self.EPILOGUE_MAX_ROWS):
            if a.dtype != x.dtype:
                a, b = a.astype(x.dtype), b.astype(x.dtype)
            return base(x, lora=(a, b, None, route.slots, self._table))
        return base(x) + self._delta(a, b, x, route)

    def _delta(self, a, b, x, route):
        # The CPU GatherMM is float32-only.
        dt = mx.float32 if mx.default_device() == mx.cpu else x.dtype
        if a.dtype != dt:
            a, b = a.astype(dt), b.astype(dt)
        *lead, n_in = x.shape
        rows = x.reshape(-1, 1, n_in)
        if rows.dtype != dt:
            rows = rows.astype(dt)
        sort = route.order is not None
        tail = sort and route.n_live is not None
        entry = route.entry
        if sort:
            rows = rows[route.order]
        if tail:
            # Slot order puts the adapted rows last.
            n0 = entry.size - route.n_live
            rows, entry = rows[n0:], entry[n0:]
        z = mx.gather_mm(rows, a, rhs_indices=entry, sorted_indices=sort)
        d = mx.gather_mm(z, b, rhs_indices=entry, sorted_indices=sort)
        if tail:
            d = mx.concatenate([mx.zeros((n0, *d.shape[1:]), d.dtype), d])
        else:
            d = d * route.live
        if sort:
            d = d[route.inv]
        return d.reshape(*lead, -1).astype(x.dtype)


def lora_shapes(args: ModelArgs) -> dict:
    """Projection -> (in, out, rank) of the packed adapter stacks."""
    r, d = args.max_lora_rank, args.hidden_size
    hd = d // args.num_attention_heads
    kv = args.num_key_value_heads * hd
    ff = args.intermediate_size
    return {"qkv": (d, d + 2 * kv, 3 * r), "o": (d, d, r),
            "gate": (d, ff, r), "up": (d, ff, r), "down": (ff, d, r)}


def pack_lora(wire: dict) -> dict:
    """The packed stacks of one layer from its wire tensors (``<t>_a (S, r,
    in)``, ``<t>_b (S, out, r)``, slot 0 zero). Every stack is dense: the
    kq LoRA route copies a strided stack on every call."""
    a = {t: wire[f"{t}_a"][1:].swapaxes(-1, -2) for t in SwitchLoRA.TARGETS}
    b = {t: wire[f"{t}_b"][1:].swapaxes(-1, -2) for t in SwitchLoRA.TARGETS}
    out = {f"{t}_{h}": mx.contiguous((a if h == "a" else b)[t])
           for t in ("o", "gate", "up", "down") for h in "ab"}
    out["qkv_a"] = mx.concatenate([a["q"], a["k"], a["v"]], axis=-1)
    widths = [b[t].shape[-1] for t in ("q", "k", "v")]
    blocks, col = [], 0
    for t, w in zip(("q", "k", "v"), widths):
        pad = [(0, 0), (0, 0), (col, sum(widths) - col - w)]
        blocks.append(mx.pad(b[t], pad))
        col += w
    out["qkv_b"] = mx.concatenate(blocks, axis=1)
    return out


class SlotRoute:
    """Each token row's router slot, for :meth:`SwitchLoRA.project`. The
    epilogue reads ``slots`` through the stack table. The GatherMM path
    reads ``entry`` (the stack entry, slot 0 clamped onto entry 0 and
    zeroed by ``live``); past a few rows its rows go in slot order, so
    GatherMM runs one segmented GEMM per entry.

    With ``count_live`` the adapted row count is read back to the host
    (``n_live``), so the GatherMM path runs on those rows alone. Under
    ``mx.compile`` the read is not allowed and ``n_live`` stays None."""

    SORT_MIN_ROWS = 16

    def __init__(self, slots: mx.array, count_live: bool = False):
        self.slots = slots.reshape(-1)
        entry = mx.maximum(self.slots.astype(mx.int32) - 1, 0).astype(mx.uint32)
        live = self.slots > 0
        self.n_live = None
        if count_live:
            try:
                self.n_live = int(live.sum().item())
            except ValueError:
                pass
        self.order = self.inv = None
        if entry.size >= self.SORT_MIN_ROWS:
            self.order = mx.argsort(self.slots)
            self.inv = mx.argsort(self.order)
            entry, live = entry[self.order], live[self.order]
        self.entry = entry
        self.live = live.reshape(-1, 1, 1)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.scale = args.attention_multiplier
        self.qkv_proj = nn.Linear(dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, dim, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            True,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(self, x, mask, cache, lora: SwitchLoRA, route):
        B, L, _ = x.shape
        qkv = lora.project("qkv", self.qkv_proj, x, route)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim :]
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return lora.project("o", self.o_proj, out, route)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x, lora: SwitchLoRA, route):
        g = lora.project("gate", self.gate_proj, x, route)
        u = lora.project("up", self.up_proj, x, route)
        return lora.project("down", self.down_proj, nn.silu(g) * u, route)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.switch_lora = SwitchLoRA(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.residual_multiplier = args.residual_multiplier

    def __call__(self, x, mask, cache, route):
        r = self.self_attn(
            self.input_layernorm(x), mask, cache, self.switch_lora, route)
        h = x + r * self.residual_multiplier
        r = self.mlp(self.post_attention_layernorm(h), self.switch_lora, route)
        return h + r * self.residual_multiplier


class GraniteSwitchModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.embedding_multiplier = args.embedding_multiplier
        n = args.vocab_size
        sub = list(range(n))
        slot = [0.0] * n
        for i, (tok, rep) in enumerate(
                zip(args.adapter_token_ids, args.adapter_substitute_ids)):
            sub[tok] = rep
            slot[tok] = float(i + 1)
        self._substitute = mx.array(sub, dtype=mx.int32)
        self._control_slot = mx.array(slot, dtype=mx.float32)
        self._n_adapters = args.num_adapters

    def embed(self, inputs: mx.array) -> mx.array:
        """Token embeddings, a control token taking its substitute's row,
        scaled by the embedding multiplier."""
        return self.embed_tokens(self._substitute[inputs]) * self.embedding_multiplier

    def slots(self, inputs: mx.array, router) -> mx.array:
        """Each input token's adapter slot, ``(B, L)`` uint32, appending the
        tokens' lanes to ``router`` when there is one."""
        v = self._control_slot[inputs]
        k = (v > 0).astype(mx.float32)
        L = inputs.shape[1]
        cnt = mx.cumsum(k, axis=-1)
        tot = mx.cumsum(v, axis=-1)
        if router is not None:
            keys, values = router.update_and_fetch(
                k[:, None, :, None], v[:, None, :, None])
            T = keys.shape[2]
            if T > L:
                cnt = cnt + keys[:, 0, : T - L, 0].sum(axis=-1, keepdims=True)
                tot = tot + values[:, 0, : T - L, 0].sum(axis=-1, keepdims=True)
        mean = tot / mx.maximum(cnt, 1.0)
        s = mx.floor(mx.clip(mean, 0.0, float(self._n_adapters)) + 0.5)
        return mx.where(cnt > 0, s, 0.0).astype(mx.uint32)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed(inputs)
        n = len(self.layers)
        if cache is None:
            cache = [None] * (n + 1)
        router = cache[n] if len(cache) > n else None
        # Decode widths skip slot-0 rows inside the matmul epilogue; wider
        # calls count the adapted rows on the host and skip a plain chunk.
        rows = inputs.size
        route = SlotRoute(self.slots(inputs, router),
                          count_live=rows > SwitchLoRA.EPILOGUE_MAX_ROWS)
        if route.n_live == 0:
            route = None
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, route)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GraniteSwitchModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.logits_scaling = args.logits_scaling

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out / self.logits_scaling

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        wire: dict = {}
        for k in list(weights):
            head, sep, leaf = k.rpartition(".switch_lora.")
            if sep and leaf.split("_")[0] in SwitchLoRA.TARGETS:
                wire.setdefault(head, {})[leaf] = weights.pop(k)
        for head, tensors in wire.items():
            for leaf, v in pack_lora(tensors).items():
                weights[f"{head}.switch_lora.{leaf}"] = v
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers] + [SwitchRouterCache()]
