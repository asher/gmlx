"""The distill head reproduces the logits of every trainable arch.

train, cache and eval build the head from the model's arguments and refuse
a model whose head parity gap exceeds HEAD_PARITY_TOL, so a logit scale the
head does not read makes that arch impossible to distill.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest

from .tiny_train_archs import CASES, build, process_patches

pytestmark = pytest.mark.skipif(
    os.environ.get("KQUANT_FORCE_CPU") == "1" or not mx.metal.is_available()
    or mx.default_device() != mx.gpu, reason="the K-quant kernels run on the GPU only")


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_distill_head_matches_the_model_logits(name):
    from gmlx.distill.head import HEAD_PARITY_TOL, head_parity_gap, head_spec_from_model

    with process_patches():
        model, _config, _case = build(name)
        head = head_spec_from_model(getattr(model, "language_model", model))
        assert head_parity_gap(model, head, mx.arange(1, 9)[None]) < HEAD_PARITY_TOL


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_distill_head_backward_is_the_gradient_of_its_forward(name):
    """train reaches the trunk through the head's closed-form backward, so
    that backward must be the gradient of the head's own forward on every
    trainable arch, a Hadamard-folded head's input rotation included."""
    from gmlx.distill.head import chunked_head_vjp, head_spec_from_model

    with process_patches():
        model, _config, _case = build(name)
        model.train()
        head = head_spec_from_model(getattr(model, "language_model", model))
        params = head.current()
        rng = np.random.default_rng(0)
        d = int(head.dense_weight().shape[1])
        h = mx.array(rng.standard_normal((16, d)).astype(np.float32) * 0.5)
        nxt = mx.array(rng.integers(0, head.V, 16).astype(np.int32))
        a = mx.array(rng.standard_normal(16).astype(np.float32))

        def onpath(hh):
            z = head.fn(params, hh).astype(mx.float32)
            z = head.softcap * mx.tanh(z / head.softcap) if head.softcap else z
            return mx.take_along_axis(z - mx.logsumexp(z, axis=-1, keepdims=True), nxt[:, None], axis=1)[:, 0]
        _, (want,) = mx.vjp(onpath, [h], [a])
        got, _ = chunked_head_vjp(h, head, nxt, n_bnd=0, target_gid=None, group_of=None, G=1, Kp=1,
                                  log_bmask=mx.zeros((head.V,)), C=8, params=params, d_onpath=a,
                                  d_Qslot=None, d_logbm=None, want_params=False)
        assert float(mx.abs(got - want).max() / mx.abs(want).max()) < 2e-2


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_startup_backward_check_runs_on_a_model_in_eval_mode_and_leaves_each_mode(name):
    """train runs the backward check on the model as the loader leaves it,
    in eval mode, where a Hadamard-folded head rotates through a kernel
    with no backward. The check runs the head in training mode and gives
    every module its own mode back."""
    from gmlx.distill.head import HEAD_PARITY_TOL, head_backward_gap, head_spec_from_model

    with process_patches():
        model, _config, _case = build(name)
        model.eval()
        inner = getattr(model, "language_model", model)
        inner.layers[0].train()
        head = head_spec_from_model(inner)
        assert head_backward_gap(model, head, mx.arange(1, 9)[None]) < HEAD_PARITY_TOL
        in_layer = {id(m) for _, m in inner.layers[0].named_modules()}
        wrong = [k for k, m in model.named_modules() if m.training != (id(m) in in_layer)]
        assert not wrong, wrong
