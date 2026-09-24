"""install_index_stop_gradient keeps selection ids off the gradient for a
training run, so the backward through a stock MoE block is defined, and
gmlx's own routers detach their ids without it."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.tune.indices import install_index_stop_gradient


@pytest.fixture(autouse=True)
def _cpu():
    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(dev)


def _args(**over):
    kw = dict(hidden_size=16, moe_intermediate_size=32, intermediate_size=32, num_experts=8, num_local_experts=8,
              num_experts_per_tok=4, norm_topk_prob=True, shared_expert_intermediate_size=32,
              routed_scaling_factor=1.5, swiglu_alpha=1.702, swiglu_limit=7.0, shared_intermediate_size=32,
              n_routed_experts=8, n_group=2, topk_group=1, topk_method="noaux_tc", n_shared_experts=1)
    kw.update(over)
    return SimpleNamespace(**kw)


def _input_grad(fn, x):
    g = mx.grad(lambda x: (fn(x).astype(mx.float32) ** 2).sum())(x)
    mx.eval(g)
    return np.array(g)


def _stock(name):
    a = _args()
    if name == "qwen3_moe":
        from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
        return Qwen3MoeSparseMoeBlock(a)
    if name == "qwen3_next":
        from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock
        return Qwen3NextSparseMoeBlock(a)
    if name == "qwen3_5_moe":
        from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
        return Qwen3_5MoeSparseMoeBlock(a)
    if name == "minimax":
        from mlx_lm.models.minimax import MiniMaxSparseMoeBlock
        return MiniMaxSparseMoeBlock(a)
    if name == "gpt_oss":
        from mlx_lm.models.gpt_oss import MLPBlock
        return MLPBlock(a)
    from mlx_lm.models.deepseek_v3 import DeepseekV3MoE
    return DeepseekV3MoE(a)


def test_install_detaches_the_four_ops_and_restores_them():
    """The replacements return the originals' values, a nested install
    changes nothing and its restore leaves the outer one in place, and the
    outer restore puts back the very functions it replaced."""
    names = ("argpartition", "argsort", "argmax", "argmin")
    orig = {n: getattr(mx, n) for n in names}
    x = mx.array(np.random.default_rng(0).standard_normal((3, 9)).astype(np.float32))
    ref = [orig["argpartition"](x, kth=2, axis=-1), orig["argsort"](x, axis=-1),
           orig["argmax"](x, axis=-1), orig["argmin"](x, axis=-1)]
    restore = install_index_stop_gradient()
    try:
        assert restore.count == 4
        assert all(getattr(mx, n) is not orig[n] for n in names)
        got = [mx.argpartition(x, kth=2, axis=-1), mx.argsort(x, axis=-1), mx.argmax(x, axis=-1),
               mx.argmin(x, axis=-1)]
        for r, g in zip(ref, got):
            assert g.dtype == r.dtype and mx.array_equal(g, r)
        inner = install_index_stop_gradient()
        assert inner.count == 0
        inner()
        assert all(getattr(mx, n)._gmlx_index_stop_gradient for n in names)
    finally:
        restore()
    assert all(getattr(mx, n) is orig[n] for n in names)


def test_a_gather_at_selected_ids_has_the_gradient_of_the_selected_values():
    """With the ids detached, the gradient of a top-k gather is the
    gradient of the gathered values scattered back to their positions."""
    x = mx.array(np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32))

    def top2(x):
        return mx.take_along_axis(x, mx.argpartition(-x, kth=1, axis=-1)[..., :2], axis=-1)

    restore = install_index_stop_gradient()
    try:
        g = _input_grad(top2, x)
    finally:
        restore()
    xn = np.array(x)
    want = np.zeros_like(xn)
    for r in range(xn.shape[0]):
        top = np.argsort(-xn[r])[:2]
        want[r, top] = 2 * xn[r, top]
    assert np.allclose(g, want)


@pytest.mark.parametrize("name", ["qwen3_moe", "qwen3_next", "qwen3_5_moe", "minimax", "gpt_oss", "deepseek_v3"])
def test_stock_moe_blocks_train_under_the_wrapper(name):
    """A stock block gathers its routing weights at argpartition ids that
    carry a gradient. Under the wrapper its input gradient is finite and
    nonzero, and its output is unchanged."""
    mx.random.seed(5)
    block = _stock(name)
    mx.eval(block.parameters())
    x = mx.random.normal((2, 5, 16))
    y0 = block(x)
    restore = install_index_stop_gradient()
    try:
        assert mx.array_equal(block(x), y0)
        g = _input_grad(block, x)
    finally:
        restore()
    assert np.isfinite(g).all() and np.abs(g).sum() > 0


def test_owned_routers_detach_their_ids_without_the_wrapper():
    """gmlx's eager MoE forwards and its model routers detach their ids
    themselves, so their backward is defined outside a training run too."""
    from mlx_lm.models.gpt_oss import MLPBlock
    from mlx_lm.models.minimax import MiniMaxSparseMoeBlock
    from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock
    from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

    import gmlx.stream.moe_experts as me
    from gmlx.models.minimax_m3 import MiniMaxM3SparseMoeBlock
    from gmlx.models.qwen4_exp.model import SparseMoeBlock as Qwen4ExpMoe
    from gmlx.models.qwen35.layers import OwnedQwen3_5MoeSparseMoeBlock

    assert not getattr(mx.argpartition, "_gmlx_index_stop_gradient", False)
    mx.random.seed(6)
    a = _args()
    cases = {
        "qwen3_moe forward": (me._qwen3_moe_forward, Qwen3MoeSparseMoeBlock(a)),
        "qwen3_next forward": (me.qwen3_next_moe_forward, Qwen3NextSparseMoeBlock(a)),
        "minimax forward": (me._minimax_forward, MiniMaxSparseMoeBlock(a)),
        "minimax_m3 forward": (me._minimax_m3_forward, MiniMaxM3SparseMoeBlock(a)),
        "gpt_oss forward": (me.gptoss_moe_forward, MLPBlock(a)),
        "minimax_m3 block": (None, MiniMaxM3SparseMoeBlock(a)),
        "qwen4_exp block": (None, Qwen4ExpMoe(a)),
        "qwen35 block": (None, OwnedQwen3_5MoeSparseMoeBlock(a)),
    }
    x = mx.random.normal((2, 5, 16))
    for label, (fwd, block) in cases.items():
        mx.eval(block.parameters())
        g = _input_grad((lambda x, b=block, f=fwd: f(b, x) if f else b(x)), x)
        assert np.isfinite(g).all() and np.abs(g).sum() > 0, label


def test_owned_selection_functions_detach_their_ids_without_the_wrapper():
    """The DeepSeek-V4 selections (GLM-5-Next routes through them too),
    Hunyuan-V3's and the expert-mass filter pass a gradient to their
    scores and none to the ids."""
    from gmlx.models.deepseek_v4 import model as dsv4
    from gmlx.models.hy_v3.model import expert_select as hy_select
    from gmlx.stream.moe_experts import _mass_filter_fn

    rng = np.random.default_rng(7)
    logits = mx.array(rng.standard_normal((5, 8)).astype(np.float32))
    bias = mx.array(rng.standard_normal(8).astype(np.float32) * 0.1)
    image = mx.array([True, False, True, False, False])
    ids = mx.array([1, 2, 0, 3, 4])
    tid2eid = mx.array(np.stack([rng.permutation(8)[:2] for _ in range(6)]).astype(np.int32))

    def weights(fn):
        return lambda z: fn(z)[1]

    fns = {
        "dsv4 select": weights(lambda z: dsv4._expert_select(z, bias, 2, 1.5, True, "sigmoid")),
        "dsv4 select vl": weights(lambda z: dsv4._expert_select_vl(z, bias, bias * 2, image, 2, 1.5, True, "sigmoid")),
        "dsv4 hash vl": weights(lambda z: dsv4._hash_expert_select_vl(ids, z, tid2eid, bias, image, 2, 1.5, True,
                                                                      "sigmoid")),
        "hy_v3 select": weights(lambda z: hy_select(z, bias, 2, 1.5, True)),
        "mass filter": lambda z: _mass_filter_fn(0.9)(mx.array(np.tile(np.arange(8, dtype=np.int32), (5, 1))),
                                                      mx.softmax(z, axis=-1))[1],
    }
    assert not getattr(mx.argpartition, "_gmlx_index_stop_gradient", False)
    for label, fn in fns.items():
        g = _input_grad(fn, logits)
        assert np.isfinite(g).all() and np.abs(g).sum() > 0, label


def test_qwen4_exp_kernel_block_ids_carry_no_gradient(monkeypatch):
    """The kq radix top-k is a custom kernel that the wrapper never sees.
    A stand-in selecting with the stock op keeps the ids on the tape, so
    the block mask's backward is defined only if the site detaches them."""
    from gmlx.models.qwen4_exp import model as q4

    stock = mx.argpartition
    monkeypatch.setattr(
        q4, "_kq_topk_fn",
        lambda s, k, flag: stock(-s.astype(mx.float32), kth=k - 1, axis=-1)[..., :k])
    idx = q4.QSAIndexer(SimpleNamespace(
        indexer_n_heads=4, indexer_head_dim=32, indexer_budget=2048,
        hidden_size=64, rope_theta=1e4, rms_norm_eps=1e-6), 4, 16)
    mx.random.seed(0)
    L, key_len = 8, 2080
    x = mx.random.normal((1, L, 64)).astype(mx.bfloat16)
    ik = mx.random.normal((1, key_len, 32)).astype(mx.bfloat16)

    def f(x):
        sel, _ = idx.select(x, ik, None, key_len - L)
        assert sel.shape == (1, L, 512)
        blk = mx.zeros((1, L, key_len // 4 + 1))
        v = x.astype(mx.float32).sum()
        return mx.put_along_axis(blk, sel, v, axis=-1).sum()

    mx.eval(mx.grad(f)(x))


def test_glm5_next_router_kernel_is_skipped_in_training(monkeypatch):
    """The fused router kernel has no backward."""
    import mlx_kquant as kq

    from gmlx.models.glm5_next import model as g5

    def kernel(*a, **k):
        raise AssertionError("router kernel reached")

    monkeypatch.setattr(g5, "_kq_router_available", lambda: True)
    monkeypatch.setattr(kq, "moe_router_topk", kernel, raising=False)
    gate = g5.Glm5NextMoEGate(_args(hidden_size=16, n_routed_experts=8,
                                    num_experts_per_tok=2))
    gate.weight = mx.random.normal(gate.weight.shape)
    gate.train()
    x = mx.random.normal((1, 3, 16))
    mx.eval(_input_grad(lambda x: gate(x)[1], x))
    gate.eval()
    with pytest.raises(AssertionError, match="router kernel reached"):
        gate(x)


def test_every_kernel_top_k_call_detaches_its_ids():
    """Each ``dsa_topk_indices`` call in gmlx is inside ``mx.stop_gradient``,
    since the kernel has no backward and the training wrapper covers the
    mlx.core ops only. Warm-up lambdas are exempt."""
    import ast
    from pathlib import Path

    import gmlx

    root = Path(gmlx.__file__).parent
    bare = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        parent = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dsa_topk_indices"):
                continue
            p = parent.get(node)
            while p is not None and not isinstance(p, (ast.stmt, ast.Lambda)):
                if (isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute)
                        and p.func.attr == "stop_gradient"):
                    break
                p = parent.get(p)
            if p is None or isinstance(p, ast.stmt):
                bare.append(f"{path.relative_to(root)}:{node.lineno}")
    assert bare == []
