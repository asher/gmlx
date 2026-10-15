"""A request's logit_bias reaches the sampler only as finite numbers. NaN
turns every sampled row into NaN, so the draws are arbitrary ids, and an
infinite ban on the one token a grammar still allows (the end of sequence
once the schema is complete) leaves a row with no finite logit, which a
sampler also draws arbitrary ids from."""

import importlib
import math

import mlx.core as mx
import pytest

rn = importlib.import_module("gmlx._vendor.mlx_vlm.server.request_normalization")
schemas = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")
su = importlib.import_module("gmlx._vendor.mlx_vlm.sample_utils")


def _args(bias):
    req = schemas.ChatRequest(
        model="m", messages=[{"role": "user", "content": "hi"}],
        logit_bias=bias)
    return rn._build_gen_args(req)


@pytest.mark.parametrize("bad", [math.nan, "x", None, [1.0]])
def test_a_bias_that_is_not_a_number_is_refused(bad):
    with pytest.raises(ValueError, match="logit_bias for token 5"):
        _args({"5": bad})


def test_an_unbounded_bias_is_held_at_the_ban_value():
    got = _args({"5": -math.inf, "6": 1e39, "7": -2.5}).logit_bias
    assert got == {5: -rn._LOGIT_BIAS_MAX, 6: rn._LOGIT_BIAS_MAX, 7: -2.5}


@pytest.mark.parametrize("temp", [0.0, 0.7])
def test_a_ban_on_the_only_allowed_token_still_ends_the_row(temp):
    """The grammar left only the end of sequence (id 3), and the request
    bans it. The row still draws id 3, so the request ends."""
    V = 64
    row = mx.where(mx.arange(V)[None] == 3, 0.5, -mx.inf)
    bias = _args({"3": -math.inf}).logit_bias
    row = su.make_logits_processors(logit_bias=bias)[0](None, row)
    # the decode loop hands the sampler log-probabilities
    row = row - mx.logsumexp(row, axis=-1, keepdims=True)
    sampler = su.make_sampler(temp=temp, top_p=0.9)
    assert [int(sampler(row).item()) for _ in range(8)] == [3] * 8
