"""Shared Hadamard rotation for the mlx-lm qwen3_next attention and MLP.

The stock ``Qwen3NextAttention`` and ``Qwen3NextMLP`` call q/k/v and
gate/up as separate module calls, so on a Hadamard-folded file each
projection rotates the same input again. ``install_hadamard_sharing``
class-swaps those instances onto subclasses whose forward mirrors the
stock body with the projection group routed through
``hadamard_modules.shared_linears``, which rotates once per group. The
op sequence is otherwise the stock one, so the output is bit-identical
to the per-module form. GDN layers take the same sharing inside
``gdn_patches``, and the owned tree takes it through ``verify_linears``.

SDPA resolves from the defining module at call time, the
``occupancy_fuse`` rule, so seam patches cover the swapped path too.
The swap runs after ``install_hadamard_modules`` and before
``install_occupancy_fuse``, which already skips folded projections.
"""

from __future__ import annotations

import sys

import mlx.core as mx

from gmlx.load.hadamard_modules import is_folded, shared_linears

_QWEN_MODULES = ("models.qwen3_5", "models.qwen3_next")


def _sdpa_of(cls):
    return sys.modules[cls.__module__].scaled_dot_product_attention


def _attention_eligible(m) -> bool:
    if type(m).__name__ not in ("Attention", "Qwen3NextAttention"):
        return False
    if not type(m).__module__.endswith(_QWEN_MODULES):
        return False
    projs = [getattr(m, p, None) for p in ("q_proj", "k_proj", "v_proj")]
    return sum(is_folded(p) for p in projs) >= 2


def _mlp_eligible(m) -> bool:
    if type(m).__name__ != "Qwen3NextMLP":
        return False
    if not type(m).__module__.endswith(_QWEN_MODULES):
        return False
    return is_folded(getattr(m, "gate_proj", None)) and is_folded(
        getattr(m, "up_proj", None))


def _make_shared_attention(base_cls):
    class _SharedQwen35Attention(base_cls):
        def __call__(self, x, mask=None, cache=None):
            B, L, D = x.shape

            q_proj_output, keys, values = shared_linears(
                (self.q_proj, self.k_proj, self.v_proj), x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
            )
            gate = gate.reshape(B, L, -1)

            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
                0, 2, 1, 3
            )
            values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
                0, 2, 1, 3
            )

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            output = _sdpa_of(base_cls)(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

            return self.o_proj(output * mx.sigmoid(gate))

    _SharedQwen35Attention.__name__ = "_SharedQwen35Attention"
    return _SharedQwen35Attention


def _make_shared_mlp(base_cls):
    from mlx_lm.models.activations import swiglu

    class _SharedQwen35MLP(base_cls):
        def __call__(self, x) -> mx.array:
            gate, up = shared_linears((self.gate_proj, self.up_proj), x)
            return self.down_proj(swiglu(gate, up))

    _SharedQwen35MLP.__name__ = "_SharedQwen35MLP"
    return _SharedQwen35MLP


def install_hadamard_sharing(model) -> int:
    """Class-swap the stock qwen3_next attention and MLP instances whose
    projections are Hadamard-folded. Returns instances swapped."""
    classes: dict = {}
    n = 0
    for _, m in model.named_modules():
        if _attention_eligible(m):
            maker = _make_shared_attention
        elif _mlp_eligible(m):
            maker = _make_shared_mlp
        else:
            continue
        base = type(m)
        sub = classes.get(base)
        if sub is None:
            sub = maker(base)
            classes[base] = sub
        m.__class__ = sub
        n += 1
    return n
