"""A right-padded batch leaves each Mamba2 row with the state of its own
prompt.

The batch engines pad short prompts on the right and declare the row
lengths on the cache. The scan must end each row's state at the row's
length. A scan that ignores the lengths takes its final state from the
last position, a pad on a short row, and loses the short row's prompt.
Tiny random models on the CPU, one row per Mamba2 family gmlx loads.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.cache.kv_cache import ArraysCache


def _nemotron_h():
    from test_nemotron_h import _randomized, _tiny_args

    return _randomized(_tiny_args())


def _granitehybrid():
    from gmlx.load.config_synth import synthesize_config
    from gmlx.load.loader import build_model
    from test_config_synth import _granitehybrid_meta

    mx.random.seed(0)
    model, _ = build_model(synthesize_config(_granitehybrid_meta(), {}))
    mx.eval(model.parameters())
    return model


_FAMILIES = {"nemotron_h": _nemotron_h, "granitehybrid": _granitehybrid}


def _mixer(model):
    """The first Mamba2 mixer, with steps of about one so its state holds
    the prompt."""
    m = next(m for _, m in model.named_modules()
             if "dt_bias" in m and "A_log" in m)
    m.dt_bias = mx.ones_like(m.dt_bias)
    m.A_log = mx.zeros_like(m.A_log)
    mx.eval(m.parameters())
    return m


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_right_padded_rows_keep_their_own_state(family):
    mixer = _mixer(_FAMILIES[family]())
    lengths = [9, 3]
    width = mixer.in_proj.weight.shape[1]
    x = mx.random.normal((2, 9, width), key=mx.random.key(1))

    batch = ArraysCache(2)
    batch.prepare(lengths=lengths, right_padding=[0, 6])
    mixer(x, batch.make_mask(9), batch)
    for row, n in enumerate(lengths):
        alone = ArraysCache(2)
        mixer(x[row:row + 1, :n], None, alone)
        for got, want in zip(batch.cache, alone.cache):
            np.testing.assert_allclose(np.array(got[row]), np.array(want[0]),
                                       atol=1e-4, rtol=1e-4)
