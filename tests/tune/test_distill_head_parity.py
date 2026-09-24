"""The distill head reproduces the logits of every trainable arch.

train, cache and eval build the head from the model's arguments and refuse
a model whose head parity gap exceeds HEAD_PARITY_TOL, so a logit scale the
head does not read makes that arch impossible to distill.
"""

from __future__ import annotations

import os

import mlx.core as mx
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
