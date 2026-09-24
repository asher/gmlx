"""DeepSeek-V4 and V4.1 training forwards.

The QAT round-trips pass the gradient straight through, the kernels with
no backward stay out of a training step, a compiled step never evaluates
inside the transform, and a row-fused projection pair serves and trains
its members' adapters. Tiny random-weight models on the GPU.
"""
from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_kquant as kq
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from test_deepseek_v41_model import _args as _v41_args
from test_deepseek_v41_model import _randomized as _v41_randomized

from gmlx.models.deepseek_v4 import hyper_connection as hc_mod
from gmlx.models.deepseek_v4 import model as v4
from gmlx.models.deepseek_v41 import model as v41
from gmlx.tune.indices import install_index_stop_gradient
from gmlx.tune.lora import prepare_lora_student

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu or not mx.metal.is_available(),
    reason="the kernels under test are Metal-only",
)

_FUSED_SLOTS = ("_qa_kv_fused", "_kv_gate_fused")


@pytest.fixture(autouse=True)
def _kernel_latches(monkeypatch):
    """Restore the kernel latches a fallback may clear for the process."""
    for table in (v4._dsa_state, v4._EMIT_QAT_NATIVE, v4._SPARSE_KERNEL,
                  v4._KQ_SKINNY_STATE, hc_mod._KQ_SKINNY, v41._QAT_FUSED):
        for key in list(table):
            monkeypatch.setitem(table, key, table[key])
    monkeypatch.setattr(v4, "_pool_grid_certified", v4._pool_grid_certified)


def _v4_args(**over):
    base = dict(
        vocab_size=64, hidden_size=64, intermediate_size=32,
        moe_intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
        q_lora_rank=32, o_groups=2, o_lora_rank=16, n_routed_experts=4,
        num_experts_per_tok=2, n_shared_experts=1, num_hash_layers=1,
        index_n_heads=2, index_head_dim=16, index_topk=4, sliding_window=8,
        compress_ratios=[0, 128, 4], hc_mult=4, hc_sinkhorn_iters=5,
        rms_norm_eps=1e-6, hc_eps=1e-5, rope_scaling=None,
    )
    base.update(over)
    return v4.ModelArgs(**base)


def _v4_model(seed=0, bf16=False, **over):
    """A V4 model with random weights. Layer 0 routes by hash, layer 1
    compresses at ratio 128 and layer 2 at ratio 4 with an indexer."""
    v4.ensure_registered()
    mx.random.seed(seed)
    model = v4.Model(_v4_args(**over))
    keep = model.cast_predicate
    params = []
    for k, v in tree_flatten(model.parameters()):
        if v.dtype in (mx.float32, mx.float16, mx.bfloat16):
            v = mx.random.normal(v.shape) * 0.1
            if bf16 and keep(k):
                v = v.astype(mx.bfloat16)
        params.append((k, v))
    model.update(tree_unflatten(params))
    gate = model.model.layers[0].ffn.gate
    rng = np.random.default_rng(seed)
    gate.tid2eid = mx.array(
        rng.integers(0, model.args.n_routed_experts, gate.tid2eid.shape)
        .astype(np.int32))
    # The expert table is an index, so a gradient cannot reach it.
    gate.freeze(keys=["tid2eid"], recurse=False)
    mx.eval(model.parameters())
    return model


def _v41_model(bf16=False, **over):
    mx.random.seed(4)
    model = _v41_randomized(v41.Model(_v41_args(**over)))
    if bf16:
        keep = model.cast_predicate
        model.update(tree_unflatten([
            (k, v.astype(mx.bfloat16) if keep(k) and v.dtype == mx.float32 else v)
            for k, v in tree_flatten(model.parameters())]))
        mx.eval(model.parameters())
    return model


def _tokens(L, seed=1):
    return mx.array(
        np.random.default_rng(seed).integers(3, 64, (1, L)).astype(np.int32))


def _loss(model, toks):
    return (model(toks).astype(mx.float32) ** 2).mean()


def _train_step(model, toks):
    """One training forward and backward. Returns |grad| summed per
    parameter path."""
    model.train()
    restore = install_index_stop_gradient()
    try:
        loss, grads = nn.value_and_grad(model, _loss)(model, toks)
        mx.eval(loss, grads)
    finally:
        restore()
    assert bool(mx.isfinite(loss))
    return {k: float(mx.abs(v).sum()) for k, v in tree_flatten(grads)}


def _spy(fn, calls, tag=None):
    """Delegate to ``fn`` and record each call it accepts."""

    def spy(*a, **k):
        out = fn(*a, **k)
        calls.append(tag)
        return out

    return spy


def _zero_lora_grads(grads, modules):
    return {m: grads.get(f"{m}.lora_b") for m in modules
            if not grads.get(f"{m}.lora_b")}


# --- straight-through QAT ----------------------------------------------------


@pytest.mark.parametrize("head_dim", [16, 128], ids=["chain", "kernel"])
def test_v41_qat_passes_the_gradient_to_the_attention_adapters(head_dim, monkeypatch):
    """With QAT on, the KV, latent and indexer round-trips pass the gradient
    straight through. The rounding chains have a zero derivative and the
    fused kernels have no backward, so without it the adapters that feed
    them never train."""
    monkeypatch.delenv("GMLX_DS41_QAT", raising=False)
    wide = head_dim == 128
    model = _v41_model(head_dim=head_dim, qk_rope_head_dim=64 if wide else 4,
                       index_head_dim=128 if wide else 8)
    if wide:
        assert v41._qat_fused("kv") and v41._qat_fused("indexer")
    prepare_lora_student(model, rank=4)
    grads = _train_step(model, _tokens(40))
    modules = [f"model.layers.{i}.attn.wkv" for i in range(6)] + [
        "model.layers.2.attn.compressor.wkv",
        "model.layers.2.attn.compressor.wgate",
        "model.layers.4.attn.compressor.wkv",
    ]
    assert _zero_lora_grads(grads, modules) == {}


def test_v4_training_step_at_the_production_rope_width(monkeypatch):
    """At head 128 and RoPE 64 the KV rows and the indexer rows take the
    fused QAT kernels, and the step still has a backward."""
    for path in ("kv_qat", "qat"):
        monkeypatch.setitem(v4._dsa_state, path, None)
    calls = []
    for name in ("dsa_kv_qat", "dsa_indexer_qat"):
        monkeypatch.setattr(kq, name, _spy(getattr(kq, name), calls, name))
    model = _v4_model(head_dim=128, qk_rope_head_dim=64, index_head_dim=128)
    prepare_lora_student(model, rank=4)
    grads = _train_step(model, _tokens(40))
    assert {"dsa_kv_qat", "dsa_indexer_qat"} <= set(calls)
    modules = [f"model.layers.{i}.attn.{p}" for i in range(3)
               for p in ("wq_a", "wkv")] + [
        "model.layers.2.attn.compressor.wkv",
        "model.layers.2.attn.compressor.wgate",
    ]
    assert _zero_lora_grads(grads, modules) == {}


# --- kernels with no backward ------------------------------------------------


def test_v4_window_and_sparse_kernels_stay_out_of_a_training_step(monkeypatch):
    """dsa_sparse_attention serves the window-only and the sparse layers
    at 64 heads of 512, and a training step takes the ops."""
    monkeypatch.setenv("GMLX_DSA_WINDOW_MIN_L", "1")
    monkeypatch.setenv("GMLX_DSA_SPARSE_MIN_L", "1")
    for path in ("window", "sparse"):
        monkeypatch.setitem(v4._dsa_state, path, True)
    # The QAT round-trips keep to the chains, so only this kernel is in
    # question.
    for path in ("kv_qat", "qat"):
        monkeypatch.setitem(v4._dsa_state, path, False)
    monkeypatch.setitem(v4._EMIT_QAT_NATIVE, "on", False)
    ratios = []
    real = kq.dsa_sparse_attention

    def spy(*a, **k):
        out = real(*a, **k)
        ratios.append(a[7])
        return out

    monkeypatch.setattr(kq, "dsa_sparse_attention", spy)
    model = _v4_model(bf16=True, num_attention_heads=64, head_dim=512,
                      qk_rope_head_dim=64, index_head_dim=128)
    toks = _tokens(40)
    _train_step(model, toks)
    assert ratios == []
    model.eval()
    mx.eval(model(toks))
    assert set(ratios) == {1 << 30, 4}


def test_v4_skinny_kernel_stays_out_of_a_training_step(monkeypatch):
    """At 2 to 16 tokens the router, the indexer weights and the hyper
    head take skinny_matmul in serve and the matmul in training."""
    monkeypatch.setitem(v4._KQ_SKINNY_STATE, "ok", True)
    monkeypatch.setitem(hc_mod._KQ_SKINNY, "ok", True)
    shapes = []
    real = kq.skinny_matmul

    def spy(x, w):
        out = real(x, w)
        shapes.append(tuple(w.shape))
        return out

    monkeypatch.setattr(kq, "skinny_matmul", spy)
    model = _v4_model()
    toks = _tokens(8)
    _train_step(model, toks)
    assert shapes == []
    model.eval()
    mx.eval(model(toks))
    inner = model.model
    want = {
        tuple(inner.layers[1].ffn.gate.weight.shape),
        tuple(inner.layers[2].attn.indexer.weights_proj.weight.shape),
        tuple(inner.hc_head.fn.shape),
    }
    assert want <= set(shapes)


def test_v4_indexer_quantized_operand_arm_stays_out_of_a_training_step(monkeypatch):
    """The prefill arm quantizes the indexer queries for the int8 scorer.
    A training step keeps to the round-tripped queries."""
    for path in ("indexer", "indexer_q"):
        monkeypatch.setitem(v4._dsa_state, path, True)
    calls = []
    monkeypatch.setattr(kq, "dsa_indexer_qat_quant",
                        _spy(kq.dsa_indexer_qat_quant, calls))
    model = _v4_model(index_head_dim=128)
    toks = _tokens(40)
    _train_step(model, toks)
    assert calls == []
    model.eval()
    mx.eval(model(toks))
    assert calls


@pytest.mark.parametrize("L, kernel", [(40, "sdpa_sparse_decode"),
                                       (80, "sdpa_sparse_prefill")])
def test_v41_sparse_kernels_stay_out_of_a_training_step(L, kernel, monkeypatch):
    """The sparse attention kernels serve a bf16 prompt at head 128, and a
    training step takes the chains."""
    for key in ("on", "wide", "prefill"):
        monkeypatch.setitem(v4._SPARSE_KERNEL, key, True)
    for key in ("kv", "indexer"):
        monkeypatch.setitem(v41._QAT_FUSED, key, False)
    calls = []
    monkeypatch.setattr(kq, kernel, _spy(getattr(kq, kernel), calls))
    model = _v41_model(bf16=True, head_dim=128)
    toks = _tokens(L)
    _train_step(model, toks)
    assert calls == []
    model.eval()
    mx.eval(model(toks))
    assert calls


def test_v41_skinny_kernel_stays_out_of_a_training_step(monkeypatch):
    """The router and the indexer weights take skinny_matmul in serve and
    the matmul in training."""
    monkeypatch.setitem(v4._KQ_SKINNY_STATE, "ok", True)
    shapes = []
    real = kq.skinny_matmul

    def spy(x, w):
        out = real(x, w)
        shapes.append(tuple(w.shape))
        return out

    monkeypatch.setattr(kq, "skinny_matmul", spy)
    model = _v41_model()
    toks = _tokens(8)
    _train_step(model, toks)
    assert shapes == []
    model.eval()
    mx.eval(model(toks))
    inner = model.model
    want = {
        tuple(inner.layers[2].ffn.gate.weight.shape),
        tuple(inner.layers[2].attn.indexer.weights_proj.weight.shape),
    }
    assert want <= set(shapes)


# --- compiled training step --------------------------------------------------


def _compiled_losses(model, toks, steps=2):
    """Losses of ``steps`` compiled optimizer steps, built the way mlx-lm's
    trainer builds them."""
    model.train()
    opt = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(model, _loss)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(t):
        loss, grads = loss_and_grad(model, t)
        opt.update(model, grads)
        return loss

    losses = []
    restore = install_index_stop_gradient()
    try:
        for _ in range(steps):
            loss = step(toks)
            mx.eval(state, loss)
            losses.append(loss.item())
    finally:
        restore()
    return losses


def _hyper_modules(model):
    return [m for _, m in model.named_modules()
            if isinstance(m, (hc_mod.HyperConnection, hc_mod.HyperHead))]


@pytest.mark.parametrize("arch", ["v4", "v41"])
def test_a_compiled_training_step_on_a_cold_model(arch):
    """No forward has run before the first compiled step, so nothing is
    cached on the modules. The step builds fn.T in the graph, because an
    eval inside the transform is refused."""
    if arch == "v4":
        model = _v4_model(bf16=True, head_dim=128, qk_rope_head_dim=64,
                          index_head_dim=128)
    else:
        model = _v41_model(bf16=True, head_dim=128)
    prepare_lora_student(model, rank=4)
    hyper = _hyper_modules(model)
    assert hyper
    losses = _compiled_losses(model, _tokens(40))
    assert all(np.isfinite(losses)) and losses[1] != losses[0]
    assert all("_fn_t" not in m and getattr(m, "_fn_t", None) is None
               for m in hyper)


# --- row fusion --------------------------------------------------------------


def _slots(model):
    return [(p, s) for p, m in model.named_modules() for s in _FUSED_SLOTS
            if getattr(m, s, None) is not None]


def test_default_key_adapters_train_through_a_row_fused_pair():
    """Row fusion runs at every load. The student setup splits each fused
    pair back into its members, so the adapters on wq_a, wkv and the
    compressor pair are the modules the forward calls."""
    model = _v4_model()
    model.eval()
    assert v4.install_gemv_row_fusion(model) > 0
    prepare_lora_student(model, rank=4)
    grads = _train_step(model, _tokens(40))
    modules = [f"model.layers.{i}.attn.{p}" for i in range(3)
               for p in ("wq_a", "wkv")] + [
        "model.layers.2.attn.compressor.wkv",
        "model.layers.2.attn.compressor.wgate",
    ]
    assert _zero_lora_grads(grads, modules) == {}
    assert _slots(model) == []
    assert not [k for k, _ in tree_flatten(model.trainable_parameters())
                if "_fused" in k]


def _serve_model(codec):
    """A V4 model in eval mode, with attention wq_a and wkv K-quant encoded
    when ``codec`` is set."""
    from mlx_kquant.nn import KQuantLinear

    model = _v4_model()
    if codec is not None:
        for layer in model.model.layers:
            for name in ("wq_a", "wkv"):
                lin = getattr(layer.attn, name)
                out_dims, in_dims = lin.weight.shape
                q = KQuantLinear(in_dims, out_dims, False, codec)
                q.weight = kq.quantize(lin.weight, codec)[0]
                setattr(layer.attn, name, q)
        mx.eval(model.parameters())
    model.eval()
    return model


def _adapter_plan(model):
    """A LoRA plan on a member of each fused pair kind: attention wkv (with
    wq_a) and the ratio-4 compressor wgate (with its wkv)."""
    from gmlx.load import adapter

    by_path = dict(model.named_modules())
    mx.random.seed(9)
    modules = {}
    for path in ("model.layers.2.attn.wkv",
                 "model.layers.2.attn.compressor.wgate"):
        lin = by_path[path]
        out_dims = lin.weight.shape[0]
        in_dims = model.args.hidden_size
        modules[path] = adapter.LoraModule(
            module_path=path, a=mx.random.normal((4, in_dims)) * 0.5,
            b=mx.random.normal((out_dims, 4)) * 0.5, rank=4, scale=1.0)
    return adapter.LoraAdapter(alpha=4.0, arch="deepseek_v4", modules=modules)


@pytest.mark.parametrize("codec", [None, "q8_0"], ids=["float", "q8_0"])
def test_an_adapter_on_a_row_fused_member_reaches_the_serve_logits(codec):
    """Installing an adapter on a fused member clears its pair, so the
    forward calls the adapted member and matches a model that was never
    fused."""
    from gmlx.load.modules import install_lora_adapter

    toks = _tokens(10)
    plain = _serve_model(codec)
    fused = _serve_model(codec)
    assert v4.install_gemv_row_fusion(fused) > 0
    base = fused(toks)
    for model in (plain, fused):
        assert install_lora_adapter(model, _adapter_plan(model)) == 2
    got = fused(toks)
    assert not mx.array_equal(got, base)
    assert mx.array_equal(got, plain(toks))
    cleared = {"model.layers.2.attn", "model.layers.2.attn.compressor"}
    assert not cleared & {p for p, _ in _slots(fused)}


@pytest.mark.parametrize("codec", [None, "q8_0"], ids=["float", "q8_0"])
def test_a_cleared_fused_pair_serves_its_members(codec):
    """With every fused slot cleared the members serve the same logits as
    a model that was never fused."""
    from gmlx.load.modules import drop_fused_children

    toks = _tokens(10)
    plain = _serve_model(codec)
    fused = _serve_model(codec)
    n = v4.install_gemv_row_fusion(fused)
    assert n > 0 and len(_slots(fused)) == n
    mx.eval(fused(toks))
    assert drop_fused_children(fused) == n
    assert _slots(fused) == []
    assert mx.array_equal(fused(toks), plain(toks))
