"""hadamard_share: the swapped qwen3_next attention and MLP forwards
mirror the stock bodies with the projection group routed through
``shared_linears`` and the gated activation through ``glu_rotate``, and
rotate once per group. With the down and output projections unfolded they
match the per-module form bit for bit. A folded down projection takes the
fused activation and matches to bf16 rounding."""

from __future__ import annotations

import ast
import inspect
import textwrap

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import numpy as np
import pytest
from mlx_lm.models import qwen3_next as _Q
from mlx_lm.models.cache import KVCache

import gmlx.upstream.hadamard_share as hs
from gmlx.load.hadamard import FoldTarget
from gmlx.load.hadamard_modules import (
    install_hadamard_modules,
    reset_rotation_count,
    rotation_count,
)
from gmlx.load.modules import install_kquant_modules

HIDDEN = 256
BLOCK = 64


def _norm(fn) -> str:
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
    for old, new, count in table:
        found = text.count(old)
        assert found == count, (
            f"substitution {old!r} matched {found} times, expected {count} "
            f"- upstream drifted or the table is stale"
        )
        text = text.replace(old, new)
    return text


def test_attention_forward_mirrors_upstream():
    sub = hs._make_shared_attention(_Q.Qwen3NextAttention)
    expected = _apply(
        _norm(_Q.Qwen3NextAttention.__call__),
        [
            (
                "def __call__(self, x: mx.array, mask: Optional[mx.array]=None, "
                "cache: Optional[Any]=None) -> mx.array:",
                "def __call__(self, x, mask=None, cache=None):",
                1,
            ),
            (
                "q_proj_output = self.q_proj(x)",
                "q_proj_output, keys, values = shared_linears("
                "(self.q_proj, self.k_proj, self.v_proj), x)",
                1,
            ),
            ("\n    keys, values = (self.k_proj(x), self.v_proj(x))", "", 1),
            ("scaled_dot_product_attention(", "_sdpa_of(base_cls)(", 1),
            (
                "self.o_proj(output * mx.sigmoid(gate))",
                "self.o_proj(glu_rotate(output, gate, fold_of(self.o_proj), "
                "activation='sigmoid', kernel=not self.training))",
                1,
            ),
        ],
    )
    assert _norm(sub.__call__) == expected


def test_mlp_forward_mirrors_upstream():
    sub = hs._make_shared_mlp(_Q.Qwen3NextMLP)
    expected = _apply(
        _norm(_Q.Qwen3NextMLP.__call__),
        [
            (
                "return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))",
                "gate, up = shared_linears((self.gate_proj, self.up_proj), x)\n"
                "    return self.down_proj(glu_rotate(up, gate, "
                "fold_of(self.down_proj), kernel=not self.training))",
                1,
            ),
        ],
    )
    assert _norm(sub.__call__) == expected


def _args():
    return _Q.ModelArgs(
        model_type="qwen3_next", hidden_size=HIDDEN, num_hidden_layers=1,
        intermediate_size=512, num_attention_heads=4, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=0, num_experts_per_tok=0,
        decoder_sparse_step=1, shared_expert_intermediate_size=0, mlp_only_layers=[],
        moe_intermediate_size=0, rms_norm_eps=1e-6, vocab_size=64,
        num_key_value_heads=2, rope_theta=10000.0, partial_rotary_factor=0.25,
        max_position_embeddings=1024, head_dim=32,
    )


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        args = _args()
        self.self_attn = _Q.Qwen3NextAttention(args)
        self.mlp = _Q.Qwen3NextMLP(HIDDEN, args.intermediate_size)


def _quantize(model, names):
    mx.random.seed(2)
    weights, meta = {}, {}
    for name in names:
        lin = model
        for part in name.split("."):
            lin = getattr(lin, part)
        q, s = kq.quantize(lin.weight, "q8_0")
        weights[f"{name}.weight"] = q.reshape(q.shape[0], -1)
        weights[f"{name}.scales"] = s
        meta[f"{name}.weight"] = "q8_0"
    assert install_kquant_modules(model, meta) == len(names)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())


def _folded_layer(rng, *, fold=True):
    mx.random.seed(3)
    model = _Layer()
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    _quantize(model, names)
    if fold:
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), HIDDEN)
        targets = {n: FoldTarget(width=HIDDEN, block=BLOCK, signs=signs)
                   for n in names[:3] + names[4:6]}
        assert install_hadamard_modules(model, targets) == 5
    return model


def _counted(monkeypatch, fn):
    monkeypatch.setenv("GMLX_HADAMARD_TRACE", "1")
    reset_rotation_count()
    out = fn()
    mx.eval(out)
    return out, rotation_count()


def test_swap_shares_and_matches_the_per_module_form(monkeypatch):
    rng = np.random.default_rng(20)
    model = _folded_layer(rng)
    x = mx.array(rng.standard_normal((2, 3, HIDDEN)).astype(np.float32)).astype(
        mx.bfloat16)
    want_attn, n_attn = _counted(
        monkeypatch, lambda: model.self_attn(x, cache=KVCache()))
    want_mlp, n_mlp = _counted(monkeypatch, lambda: model.mlp(x))
    assert (n_attn, n_mlp) == (3, 2)

    assert hs.install_hadamard_sharing(model) == 2
    assert type(model.self_attn).__name__ == "_SharedQwen35Attention"
    assert type(model.mlp).__name__ == "_SharedQwen35MLP"
    got_attn, n_attn = _counted(
        monkeypatch, lambda: model.self_attn(x, cache=KVCache()))
    got_mlp, n_mlp = _counted(monkeypatch, lambda: model.mlp(x))
    assert (n_attn, n_mlp) == (1, 1)
    assert mx.array_equal(got_attn, want_attn)
    assert mx.array_equal(got_mlp, want_mlp)
    # The no-cache branch runs too.
    mx.eval(model.self_attn(x))


def test_unfolded_layer_is_not_swapped():
    model = _folded_layer(np.random.default_rng(21), fold=False)
    assert hs.install_hadamard_sharing(model) == 0
    assert type(model.self_attn) is _Q.Qwen3NextAttention


def test_occupancy_fuse_ignores_the_swapped_classes():
    from gmlx.upstream.occupancy_fuse import install_occupancy_fuse

    model = _folded_layer(np.random.default_rng(22))
    hs.install_hadamard_sharing(model)
    assert install_occupancy_fuse(model) == 0


@pytest.mark.parametrize("attr", ["self_attn", "mlp"])
def test_second_install_is_idempotent(attr):
    model = _folded_layer(np.random.default_rng(23))
    assert hs.install_hadamard_sharing(model) == 2
    cls = type(getattr(model, attr))
    assert hs.install_hadamard_sharing(model) == 0
    assert type(getattr(model, attr)) is cls


def test_fused_glu_feeds_a_folded_down_projection(monkeypatch):
    if getattr(kq, "glu_hadamard", None) is None:
        pytest.skip("installed mlx-kquant has no glu_hadamard")
    if mx.default_device() != mx.gpu:
        pytest.skip("the fused form runs on the GPU device only")
    rng = np.random.default_rng(24)
    mx.random.seed(3)
    model = _Layer()
    _quantize(model, ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    inter = _args().intermediate_size
    hidden = FoldTarget(width=HIDDEN, block=BLOCK,
                        signs=rng.choice(np.array([-1, 1], dtype=np.int8), HIDDEN))
    targets = {
        "mlp.gate_proj": hidden,
        "mlp.up_proj": hidden,
        "mlp.down_proj": FoldTarget(
            width=inter, block=256,
            signs=rng.choice(np.array([-1, 1], dtype=np.int8), inter)),
    }
    assert install_hadamard_modules(model, targets) == 3
    model.eval()   # the kernels serve inference only, as after a load
    x = mx.array(rng.standard_normal((2, 3, HIDDEN)).astype(np.float32)).astype(
        mx.bfloat16)
    want, n_each = _counted(monkeypatch, lambda: model.mlp(x))
    assert hs.install_hadamard_sharing(model) == 1
    calls = []
    real = kq.glu_hadamard
    monkeypatch.setattr(
        kq, "glu_hadamard", lambda *a, **k: calls.append(1) or real(*a, **k))
    got, n_fused = _counted(monkeypatch, lambda: model.mlp(x))
    assert len(calls) == 1
    # Per module: gate, up and down rotate. Swapped: one shared rotation and
    # the fused activation, whose row the down projection takes as offered.
    assert (n_each, n_fused) == (3, 2)
    gf, wf = (np.array(a.astype(mx.float32)) for a in (got, want))
    assert np.abs(gf - wf).max() <= 3e-2 * np.abs(wf).max()
