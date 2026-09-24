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
    process_patches,
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
    "qwen3_moe": 3e-2, "muse_glimmer": 3e-2, "hy_v3": 3e-2, "glm5_next": 4e-2,
    "kimi_k3": 1e-1, "hy_v4": 4e-2, "minimax_m3": 3e-2, "qwen4_exp": 2e-2,
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
    from mlx_lm.tuner.lora import LoRALinear

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

    with process_patches():
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
