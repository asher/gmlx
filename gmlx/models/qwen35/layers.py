"""Owned layer assembly for qwen3.5/3.6 MTP targets.

The dense MLP, the MoE sparse block, and both decoder layers, as
subclasses whose ``__init__`` mirrors the stock body but constructs
the owned attention/GDN/MLP classes. Dispatch-only ``__call__`` bodies
are verbatim upstream copies, source-equality-tested against the
pinned mlx-vlm release; the MLP and sparse-block forwards route
projections through the owned verify-linear family. The mirrored
constructors draw parameters in the stock order, so a seeded owned
build is weight-identical to a seeded stock build (certified by the
construction-pair tests).
"""

from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.switch_layers import SwitchGLU

from .attn import OwnedQwen3_5Attention
from .gdn import OwnedQwen3_5GatedDeltaNet
from .verify_linear import verify_linear, verify_linears

# The two MoE classes resolve lazily through the module __getattr__
# (moe_layer_classes), so only the dense pair is name-checkable here.
__all__ = ["OwnedQwen3_5MLP", "OwnedQwen3_5DecoderLayer", "moe_layer_classes"]


# --- verbatim upstream copies (activations.py / qwen3_5_moe/language.py) ---


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


def _target_verify_switch_glu(switch_mlp: SwitchGLU, x, indices, target_verify: bool):
    if not (target_verify and x.ndim == 3 and x.shape[1] > 1):
        return switch_mlp(x, indices)

    B, T, D = x.shape
    k = indices.shape[-1]
    flat_x = x.reshape(B * T, D)
    flat_indices = indices.reshape(B * T, k)
    flat_x = mx.expand_dims(flat_x, (-2, -3))

    up = switch_mlp.up_proj(flat_x, flat_indices, sorted_indices=False)
    gate = switch_mlp.gate_proj(flat_x, flat_indices, sorted_indices=False)
    out = switch_mlp.down_proj(
        switch_mlp.activation(up, gate),
        flat_indices,
        sorted_indices=False,
    )
    return out.squeeze(-2).reshape(B, T, k, -1)


class OwnedQwen3_5MLP(_L.Qwen3_5MLP):
    """Stock construction; forward mirrors upstream with the projection
    indirections routed through the owned verify-linear family."""

    def __call__(self, x, target_verify: bool = False) -> mx.array:
        gate, up = verify_linears(
            (self.gate_proj, self.up_proj), x, target_verify
        )
        return verify_linear(self.down_proj, swiglu(gate, up), target_verify)


class OwnedQwen3_5DecoderLayer(_L.Qwen3_5DecoderLayer):
    """__init__ mirrors the stock body building the owned classes; the
    dispatch-only ``__call__`` is a verbatim upstream copy."""

    def __init__(self, args, layer_idx: int):
        nn.Module.__init__(self)
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = OwnedQwen3_5GatedDeltaNet(args)
        else:
            self.self_attn = OwnedQwen3_5Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = OwnedQwen3_5MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
        gdn_sink: Optional[list] = None,
        target_verify: bool = False,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x),
                mask,
                cache,
                gdn_sink=gdn_sink,
                target_verify=target_verify,
            )
        else:
            r = self.self_attn(
                self.input_layernorm(x),
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                target_verify=target_verify,
            )
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h), target_verify)


def _moe_layer_classes():
    """MoE layer classes, lazily bound (the MoE language module imports
    the dense one; keep the import cost off dense loads)."""
    from mlx_vlm.models.qwen3_5_moe import language as _ML

    class OwnedQwen3_5MoeSparseMoeBlock(_ML.Qwen3_5MoeSparseMoeBlock):
        """__init__ mirrors the stock body (shared expert built owned);
        the forward mirrors upstream with the gate/shared-gate
        projections routed through the owned verify-linear family."""

        def __init__(self, args):
            nn.Module.__init__(self)
            dim = args.hidden_size
            intermediate_size = args.moe_intermediate_size
            shared_expert_intermediate_size = args.shared_expert_intermediate_size

            self.num_experts = num_experts = args.num_experts
            self.top_k = args.num_experts_per_tok

            self.gate = nn.Linear(dim, num_experts, bias=False)
            self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

            self.shared_expert = OwnedQwen3_5MLP(
                dim, shared_expert_intermediate_size
            )
            self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        def __call__(
            self,
            x: mx.array,
            target_verify: bool = False,
        ) -> mx.array:
            gates = verify_linear(self.gate, x, target_verify)
            gates = mx.softmax(gates, axis=-1, precise=True)

            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)

            y = _target_verify_switch_glu(
                self.switch_mlp, x, inds, target_verify
            )
            y = (y * scores[..., None]).sum(axis=-2)

            shared_y = self.shared_expert(x, target_verify)
            shared_y = (
                mx.sigmoid(
                    verify_linear(self.shared_expert_gate, x, target_verify)
                )
                * shared_y
            )

            return y + shared_y

    class OwnedQwen3_5MoeDecoderLayer(_ML.Qwen3_5MoeDecoderLayer):
        """__init__ mirrors the stock body building the owned classes;
        the dispatch-only ``__call__`` is a verbatim upstream copy."""

        def __init__(self, args, layer_idx: int):
            nn.Module.__init__(self)
            self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
            if self.is_linear:
                self.linear_attn = OwnedQwen3_5GatedDeltaNet(args)
            else:
                self.self_attn = OwnedQwen3_5Attention(args)

            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = OwnedQwen3_5MoeSparseMoeBlock(args)

        def __call__(
            self,
            x: mx.array,
            mask: Optional[mx.array] = None,
            cache: Optional[Any] = None,
            position_ids: Optional[mx.array] = None,
            position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
            gdn_sink: Optional[list] = None,
            target_verify: bool = False,
        ) -> mx.array:
            if self.is_linear:
                r = self.linear_attn(
                    self.input_layernorm(x),
                    mask,
                    cache,
                    gdn_sink=gdn_sink,
                    target_verify=target_verify,
                )
            else:
                r = self.self_attn(
                    self.input_layernorm(x),
                    mask=mask,
                    cache=cache,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    target_verify=target_verify,
                )
            h = x + r
            out = h + self.mlp(self.post_attention_layernorm(h), target_verify)
            return out

    return OwnedQwen3_5MoeSparseMoeBlock, OwnedQwen3_5MoeDecoderLayer


_MOE_LAYER_CACHE = None


def moe_layer_classes():
    global _MOE_LAYER_CACHE
    if _MOE_LAYER_CACHE is None:
        _MOE_LAYER_CACHE = _moe_layer_classes()
    return _MOE_LAYER_CACHE


def __getattr__(name):
    if name in ("OwnedQwen3_5MoeSparseMoeBlock", "OwnedQwen3_5MoeDecoderLayer"):
        block, layer = moe_layer_classes()
        return block if name == "OwnedQwen3_5MoeSparseMoeBlock" else layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
