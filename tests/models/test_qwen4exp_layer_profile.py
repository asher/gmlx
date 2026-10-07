"""qwen4exp GMLX_DECODE_LAYER_PROFILE: a profiled decode step records every
layer part, and the attention steps at level 2, without changing the
output."""

from __future__ import annotations

import mlx.core as mx
import pytest

from test_qwen4exp_prefill_dispatch import _train_model

import gmlx.models.qwen4_exp.model as q4


@pytest.fixture
def profile(monkeypatch):
    def arm(level):
        monkeypatch.setattr(q4, "_LAYER_PROFILE", level >= 1)
        monkeypatch.setattr(q4, "_SUB_PROFILE", level >= 2)
        q4._prof_reset()
    yield arm
    q4._prof_reset()


def _decode(model, steps=3):
    mx.random.seed(41)
    cache = model.make_cache()
    ids = mx.random.randint(0, model.args.vocab_size, (1, 24 + steps))
    outs = [model(ids[:, :24], cache=cache)[:, -1]]
    for i in range(steps):
        outs.append(model(ids[:, 24 + i:25 + i], cache=cache)[:, -1])
    mx.eval(*outs)
    return outs


def test_profile_off_records_nothing(profile):
    profile(0)
    _decode(_train_model())
    assert q4._prof_summary() == {}


@pytest.mark.parametrize("level", [1, 2])
def test_profile_parts_and_output(profile, capsys, level):
    model = _train_model()
    profile(0)
    ref = _decode(model)
    profile(level)
    got = _decode(model)
    for a, b in zip(got, ref):
        assert mx.allclose(a, b, atol=1e-4).item()
    s = q4._prof_summary()
    # the prefill chunk is not a decode step
    assert s["tokens"] == 3
    parts = s["parts"]
    assert {"hc", "attn", "gdn", "ffn"} <= set(parts)
    n_attn = sum(not layer.is_linear for layer in model.layers)
    assert len(s["layers"]["attn"]) == n_attn
    assert len(s["layers"]["ffn"]) == len(model.layers)
    subs = {"a.proj", "a.cache", "a.blocks", "a.score", "a.core", "a.out"}
    if level >= 2:
        # a.score needs the kq decode scorer; the rest always mark
        assert subs - {"a.score"} <= set(parts) | {"a.blocks"}
        assert {"a.proj", "a.cache", "a.core", "a.out"} <= set(parts)
    else:
        assert not subs & set(parts)
    q4._prof_dump()
    out = capsys.readouterr().out
    assert "[layerprof] per token ms:" in out
    assert ("[layerprof] sub per token ms:" in out) == (level >= 2)
