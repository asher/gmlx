"""One LoRA training step on a tiny K-quant model of every owned
architecture, then the served round trip of what it trained.

The step is mlx-lm's compiled trainer step under gmlx's training scopes.
Every adapted module must get a nonzero lora_b gradient, except the ones
listed in ZERO_GRAD, each with the reason its gradient is zero. The
trained adapter then goes through the GGUF export, the base is restored,
and the served model's eval-mode logits must match the trained model's,
for a full forward and for a cached one-token decode."""
from __future__ import annotations

import math
import os
from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from .tiny_train_archs import (
    CASES,
    Selections,
    base_names,
    build,
    eval_logits,
    logits_of,
    num_layers,
    rel,
    text_root,
    vocab_size,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available()
    or mx.default_device() != mx.gpu, reason="the K-quant kernels run on the GPU only")

SCALE = 20.0

# Adapted modules whose lora_b gradient is zero by construction.
ZERO_GRAD = {
    # The indexer of a full-attention layer returns pool ordinals from
    # argpartition, so nothing continuous flows back to its projections.
    "glm5_next": {f"model.layers.{i}.self_attn.indexer.{n}" for i in (1, 3)
                  for n in ("wq_b", "wk", "weights_proj", "compressor.wgate")},
    # The indexer's top-k ids only pick which keys attention reads.
    "hy_v4": {f"model.layers.{i}.self_attn.indexer.{n}" for i in (0, 2)
              for n in ("wq_b", "wk", "weights_proj")},
    "hy_v4_flash": {f"model.layers.{i}.self_attn.indexer.{n}" for i in (0, 3)
                    for n in ("wq_b", "wk", "weights_proj", "pool_value", "pool_gate")},
    # MSA scores blocks only when a cache is present. A training forward
    # has none and runs dense attention.
    "minimax_m3": {f"model.layers.{i}.self_attn.{n}" for i in (1, 2)
                   for n in ("index_q_proj", "index_k_proj")},
    # The QSA indexer's block ids only select keys.
    "qwen4_exp": {f"model.layers.3.self_attn.indexer.{n}" for n in ("q_proj", "k_proj")},
}

# Largest max-relative logits gap, trained against served, about twice
# the gap measured. The trained forward (f32 LoRA factors) and the served
# forms (bf16 factor tables, in the kq epilogue or in plain ops) round
# differently, the two served forms differ from each other by as much, and
# the random tiny models amplify a rounding step into the logits. kimi_k3
# amplifies most: a one-ulp bump of its first layer's output moves the
# logits by about 3%.
PARITY_TOL = {
    "qwen3_moe": 3e-2, "muse_glimmer": 3e-2, "muse_spark": 3e-2, "k2_horizon": 3e-2,
    "beam": 3e-2, "flint": 3e-2, "ricursive": 3e-2, "granite_switch": 3e-2,
    "inkling2": 6e-2,
    "hy_v3": 3e-2, "glm5_next": 4e-2,
    "kimi_k3": 1e-1, "hy_v4": 4e-2, "hy_v4_flash": 4e-2, "minimax_m3": 3e-2, "qwen4_exp": 2e-2,
    "qwen3_next": 4e-2, "qwen3_5": 3e-2, "qwen3_5_moe": 4e-2, "qwen3_5_hadamard": 4e-2,
}


def _batch(vocab, width, lengths, seed):
    """mlx-lm's padded batch and its (offset, length) rows."""
    r = np.random.default_rng(seed)
    arr = np.zeros((len(lengths), width), np.int32)
    for j, n in enumerate(lengths):
        arr[j, :n] = r.integers(1, vocab, n)
    return mx.array(arr), mx.array([[0, n] for n in lengths])


def _loss(model, batch, lengths):
    """mlx-lm's default_loss."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = logits_of(model(inputs))
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    return ce.astype(mx.float32).sum() / mask.sum(), mask.sum()


def _one_step(model, batch, lengths):
    """One step of mlx-lm's compiled trainer step, returning the loss and
    the gradient it applied."""
    opt = optim.Adam(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, _loss)
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, lengths):
        (loss, _), grad = loss_and_grad(model, batch, lengths)
        opt.update(model, grad)
        return loss, grad

    model.train()
    loss, grad = step(batch, lengths)
    mx.eval(state, loss, grad)
    return float(loss), grad


@pytest.mark.parametrize("arch", list(CASES))
def test_one_step_trains_exports_and_serves(arch, tmp_path):
    from gmlx._vendor.mlx_lm.tuner.lora import LoRALinear

    from gmlx.load.adapter import apply_gguf_adapter
    from gmlx.tune.attention import install_training_attention
    from gmlx.tune.gdn import install_training_gdn
    from gmlx.tune.indices import install_index_stop_gradient
    from gmlx.tune.kernels import install_training_switch_gemm
    from gmlx.tune.lora import (
        _set_module,
        adapter_refusals,
        prepare_lora_student,
        save_trained_adapter,
    )

    model, config, case = build(arch)
    adapted = prepare_lora_student(model, rank=4, scale=SCALE, num_layers=None)
    assert adapted > 0
    names = base_names(text_root(model), case.gguf_arch, num_layers(config))
    assert adapter_refusals(model, base_arch=case.gguf_arch, base_names=names) == {}

    # 2 x 64 input tokens, 512 routed rows at top-4
    vocab = vocab_size(config)
    batch, lengths = _batch(vocab, 65, [65, 50], seed=1)
    restores = [install_training_attention(model), install_index_stop_gradient(),
                install_training_switch_gemm()]
    install_training_gdn(model)
    try:
        loss, grad = _one_step(model, batch, lengths)
    finally:
        for restore in reversed(restores):
            restore()
    assert math.isfinite(loss)
    zero = set()
    for k, g in tree_flatten(grad):
        if k.endswith(".lora_b"):
            g = g.astype(mx.float32)
            assert mx.all(mx.isfinite(g)).item(), k
            if mx.abs(g).max().item() == 0.0:
                zero.add(k[: -len(".lora_b")])
    assert zero == ZERO_GRAD.get(arch, set())

    tokens = _batch(vocab, 24, [24, 24], seed=3)[0]
    sel = Selections()
    with sel:
        trained = eval_logits(model, tokens, sel, "record")
    path = str(tmp_path / "adapter.gguf")
    assert save_trained_adapter(model, config, base_arch=case.gguf_arch, out_path=path,
                                scale=SCALE, base_names=names) == adapted
    wrapped = [(p, m) for p, m in model.named_modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == adapted
    for p, m in wrapped:
        _set_module(model, p, m.linear)
    base = eval_logits(model, tokens)
    assert apply_gguf_adapter(model, config, path, base_arch=case.gguf_arch) == adapted
    with sel:
        served = eval_logits(model, tokens, sel, "replay")
    assert sel.mismatch == 0 and sel.aligned()

    tol = PARITY_TOL[arch]
    for form, t, s, b in zip(("full", "decode"), trained, served, base):
        print(f"{arch} {form}: served {rel(t, s):.4f}, adapter delta {rel(t, b):.4f}")
        assert rel(t, s) < tol, form
        # the adapter moves the logits far past the tolerance
        assert rel(t, b) > 5 * tol, form


def test_a_folded_gated_product_trains_where_its_projection_is_unadapted(monkeypatch):
    """An unadapted folded down or output projection gives the swiglu and
    the attention output gate its fold, and serve runs the product and the
    rotation as one kq op. That op has no backward, so a training step
    takes the unfused product."""
    import mlx_kquant as kq

    from gmlx.tune.attention import install_training_attention
    from gmlx.tune.gdn import install_training_gdn
    from gmlx.tune.lora import prepare_lora_student

    calls = []
    fused = kq.glu_hadamard
    monkeypatch.setattr(kq, "glu_hadamard",
                        lambda *a, **k: calls.append(1) or fused(*a, **k))
    model, config, _ = build("qwen3_5_hadamard")
    keys = ("self_attn.q_proj", "mlp.gate_proj", "mlp.up_proj")
    assert prepare_lora_student(model, rank=4, scale=SCALE, num_layers=None, keys=keys) > 0
    batch, lengths = _batch(vocab_size(config), 65, [65, 50], seed=1)
    restore = install_training_attention(model)
    install_training_gdn(model)
    try:
        loss, grad = _one_step(model, batch, lengths)
    finally:
        restore()
    assert math.isfinite(loss) and not calls
    for k, g in tree_flatten(grad):
        if k.endswith(".lora_b"):
            assert mx.abs(g.astype(mx.float32)).max().item() > 0.0, k
    eval_logits(model, _batch(vocab_size(config), 24, [24, 24], seed=3)[0])
    # serve fuses at the same sites
    assert calls


def test_train_loads_flash_next_without_the_expert_store(monkeypatch, tmp_path):
    """gmlx train on a qwen4_exp base, whose loads get a full expert store
    by default: the load builds none, so the training step runs the
    experts' own gathers on the mapped stacks. The load is the loader's
    order on a tiny model: the fused blocks install and register, then
    the store builds."""
    import contextlib

    import mlx_kquant as kq

    import gmlx._vendor.mlx_lm.tuner.datasets as datasets
    import gmlx._vendor.mlx_lm.tuner.trainer as trainer
    import gmlx.commands.train as train
    import gmlx.load.adapter as adapter
    import gmlx.load.expert_store as es
    import gmlx.load.loader as loader
    import gmlx.load.loadlog as loadlog
    import gmlx.load.preflight as preflight

    if not hasattr(kq, "moe_store_pack"):
        pytest.skip("this mlx-kquant has no expert store")
    monkeypatch.delenv("GMLX_MOE_EXPERT_STORE", raising=False)
    # The store's process registry, for this test's loads only.
    for name, empty in (("_BLOCKS", []), ("_DONE", []), ("_GONE", set()),
                        ("_BUILT", [False])):
        monkeypatch.setattr(es, name, empty)
    seen = {}

    def load_model(path, hf_source=None):
        model, config, _ = build("qwen4_exp")
        seen["stored"] = es.build_at_load(model, log=lambda s: None)
        seen["config"] = config
        return model, config, object()

    def one_step(model, opt, train_set, val_set, args):
        assert model.training
        batch, lengths = _batch(vocab_size(seen["config"]), 9, [9], seed=2)
        (loss, _), _ = nn.value_and_grad(model, _loss)(model, batch, lengths)
        seen["loss"] = float(loss)

    monkeypatch.setattr(preflight, "preflight",
                        lambda p, hf_source=None: SimpleNamespace(arch="qwen4exp"))
    monkeypatch.setattr(adapter, "base_tensor_names", lambda p: [])
    monkeypatch.setattr(loadlog, "load_ui", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(loader, "load_model", load_model)
    monkeypatch.setattr(train, "adapter_refusals", lambda *a, **k: {})
    monkeypatch.setattr(datasets, "load_dataset", lambda args, tok: ([], [], []))
    monkeypatch.setattr(trainer, "train", one_step)
    monkeypatch.setattr(train, "save_trained_adapter", lambda *a, **k: 0)
    load_model("base.gguf")
    assert seen["stored"] > 0  # a serving load stores the experts
    train.train_lora("base.gguf", str(tmp_path), str(tmp_path / "out.gguf"),
                     iters=1, rank=4)
    assert seen["stored"] == 0 and math.isfinite(seen["loss"])
