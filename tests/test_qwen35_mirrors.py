"""Substitution-normalized drift alarms for the owned mirror forwards.

The verbatim in-tree copies have plain source-equality tests in their
own modules. The MIRRORS - owned bodies that differ from upstream by a
fixed, small substitution table (renamed call indirections, an owned
memo, a hoisted import) - had no source-level alarm: only the toy
identity tests covered them, and those fire only when an upstream
change alters numerics on a route and shape the toys reach. Here each
mirror asserts exact equality after applying its substitution table to
the ast-normalized upstream source, so ANY upstream edit outside the
table fails loudly and forces a re-mirror review.

``Qwen3_5Model.__call__`` is deliberately not here: `_owned_model_call`
diverges structurally on purpose (B=1 shortcut removed as a correctness
fix, padded prefill hoisted, S=0 guard added), so its alarm is a
fingerprint pin in ``upstream_seams`` instead of an edit script that
would ossify the removed code as test literals.

The MoE alias premise the owned constructors rely on is asserted at the
bottom: the MoE module re-exports the dense layer classes, so building
the dense owned classes in the MoE tree is a pure-alias substitution.
"""

import ast
import inspect
import textwrap

import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5_moe.language")

from mlx_vlm.models.qwen3_5 import language as _L
from mlx_vlm.models.qwen3_5_moe import language as _ML
from mlx_vlm.models import rope_utils as _RU

from gmlx import qwen35_attn, qwen35_gdn, qwen35_layers, qwen35_rope


def _norm(fn) -> str:
    """ast-normalized source with the docstring dropped: whitespace and
    comment noise gone, so the substitution tables stay small."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    f = tree.body[0]
    if (
        f.body
        and isinstance(f.body[0], ast.Expr)
        and isinstance(f.body[0].value, ast.Constant)
        and isinstance(f.body[0].value.value, str)
    ):
        f.body = f.body[1:]
    return ast.unparse(tree)


def _apply(text: str, table) -> str:
    """Apply ordered (old, new, count) replacements; each old substring
    must occur exactly count times, so a table entry can never silently
    stop matching."""
    for old, new, count in table:
        found = text.count(old)
        assert found == count, (
            f"substitution {old!r} matched {found} times, expected {count} "
            f"- upstream drifted or the table is stale"
        )
        text = text.replace(old, new)
    return text


def _assert_mirror(owned_fn, upstream_fn, table, name):
    expected = _apply(_norm(upstream_fn), table)
    assert _norm(owned_fn) == expected, (
        f"owned mirror of {name} drifted from upstream beyond its "
        f"substitution table - re-mirror and update the table"
    )


def test_attention_call_mirror():
    _assert_mirror(
        qwen35_attn.OwnedQwen3_5Attention.__call__,
        _L.Qwen3_5Attention.__call__,
        [
            ("_target_verify_linears(", "verify_linears(", 1),
            (
                "self.rotary_emb.apply_rotary(queries, keys, ",
                "rope_apply_rotary(self.rotary_emb, queries, keys, ",
                1,
            ),
            (
                "_target_verify_left_padded_attention(",
                "_verify_attention(",
                1,
            ),
            ("scaled_dot_product_attention(", "_sdpa(", 2),
            ("_target_verify_linear(", "verify_linear(", 1),
        ],
        "Qwen3_5Attention.__call__",
    )


def test_mlp_call_mirror():
    _assert_mirror(
        qwen35_layers.OwnedQwen3_5MLP.__call__,
        _L.Qwen3_5MLP.__call__,
        [
            ("_target_verify_linears(", "verify_linears(", 1),
            ("_target_verify_linear(", "verify_linear(", 1),
        ],
        "Qwen3_5MLP.__call__",
    )


def test_moe_sparse_block_call_mirror():
    block_cls, _ = qwen35_layers.moe_layer_classes()
    _assert_mirror(
        block_cls.__call__,
        _ML.Qwen3_5MoeSparseMoeBlock.__call__,
        [("_target_verify_linear(", "verify_linear(", 2)],
        "Qwen3_5MoeSparseMoeBlock.__call__",
    )


def test_gdn_unfused_chain_mirror():
    """The owned unfused chain mirrors the stock GDN forward body: the
    projection indirections go through the owned verify-linear family,
    the scan routes to mlx-lm's tiled ops (the production numerics; the
    tv-or-sink hoist lives in the owned dispatch __call__), and the
    verify-decode branch uses the owned tiled with-states ops."""
    _assert_mirror(
        qwen35_gdn._owned_gdn_unfused,
        _L.Qwen3_5GatedDeltaNet.__call__,
        [
            (
                "def __call__(self, inputs: mx.array, "
                "mask: Optional[mx.array]=None, cache: Optional[Any]=None, "
                "gdn_sink: Optional[list]=None, target_verify: bool=False) "
                "-> mx.array:\n"
                "    B, S, _ = inputs.shape\n"
                "    target_verify = target_verify or gdn_sink is not None\n",
                "def _owned_gdn_unfused(self, inputs, mask, cache, "
                "gdn_sink, target_verify):\n"
                "    from mlx_lm.models import gated_delta as _gd\n"
                "    B, S, _ = inputs.shape\n",
                1,
            ),
            ("_target_verify_linears(", "verify_linears(", 1),
            (
                "_gated_delta_update_verify_decode(q, k, v, a, b, "
                "self.A_log, self.dt_bias, state, mask, "
                "use_kernel=not self.training)",
                "_gdn_update_with_states_tiled(q, k, v, a, b, "
                "self.A_log, self.dt_bias, state, mask)",
                1,
            ),
            (
                "out, state = gated_delta_update(",
                "out, state = _gd.gated_delta_update(",
                1,
            ),
            ("_target_verify_linear(", "verify_linear(", 1),
        ],
        "Qwen3_5GatedDeltaNet.__call__ (unfused chain)",
    )


def test_rope_cos_sin_mirror():
    _assert_mirror(
        qwen35_rope.rope_cos_sin,
        _RU.MRoPERotaryEmbedding.__call__,
        [
            (
                "def __call__(self, x, position_ids):",
                "def rope_cos_sin(rotary_emb, x, position_ids):",
                1,
            ),
            ("self.", "rotary_emb.", 7),
        ],
        "MRoPERotaryEmbedding.__call__",
    )


def test_rope_apply_rotary_mirror():
    _assert_mirror(
        qwen35_rope.rope_apply_rotary,
        _RU.MRoPERotaryEmbedding.apply_rotary,
        [
            (
                "def apply_rotary(self, q, k, position_ids, *, "
                "unsqueeze_dim: int=1, cast_output: bool=True):",
                "def rope_apply_rotary(rotary_emb, q, k, position_ids, *, "
                "unsqueeze_dim: int=1, cast_output: bool=True):",
                1,
            ),
            # Bare self(...) recursion first: it carries no "self." for
            # the attribute rename below to catch.
            (
                "cos, sin = self(k, position_ids)",
                "cos, sin = rope_cos_sin(rotary_emb, k, position_ids)",
                1,
            ),
            ("self.", "rotary_emb.", 9),
            # Owned per-ndim memo replaces the instance dict so a
            # stock-compiled entry can never masquerade as owned.
            (
                "compiled_apply = "
                "rotary_emb._compiled_apply.get(position_ids.ndim)",
                "memo = getattr(rotary_emb, '_gmlx_compiled_apply', None)\n"
                "        if memo is None:\n"
                "            memo = {}\n"
                "            rotary_emb._gmlx_compiled_apply = memo\n"
                "        compiled_apply = memo.get(position_ids.ndim)",
                1,
            ),
            (
                "rotary_emb._compiled_apply[position_ids.ndim] = "
                "compiled_apply",
                "memo[position_ids.ndim] = compiled_apply",
                1,
            ),
        ],
        "MRoPERotaryEmbedding.apply_rotary",
    )


def test_moe_alias_premise():
    """The owned MoE constructors build the DENSE owned classes where
    upstream names the Moe variants. Sound only while upstream keeps the
    Moe names as pure import aliases of the dense classes; if they ever
    become real subclasses, the owned tree would silently keep the dense
    behavior."""
    assert _ML.Qwen3_5MoeAttention is _L.Qwen3_5Attention
    assert _ML.Qwen3_5MoeGatedDeltaNet is _L.Qwen3_5GatedDeltaNet
    assert _ML.Qwen3_5MoeMLP is _L.Qwen3_5MLP
