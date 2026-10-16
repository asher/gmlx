"""The FLUX.2 klein sampling loop on the CPU.

On a stub DiT, gmlx's sample holds bit for bit to the text to image path
of BFL's scripts/cli.py, as scripts/image_oracle/check6_flux2_klein.py
--self-test records it: sampling.denoise for a distilled checkpoint, and
sampling.denoise_cfg for Base, which guides the prompt's row by the empty
prompt's. The reference keeps the latents in bf16, and each loop rule of
the family is needed for that. Every other family's loop rules are
unchanged.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from oracle_fixture import committed_fixture

from gmlx.image import api
from gmlx.image.conditioning import Conditioning
from gmlx.image.registry import LoopRules, families, get_family

LOOP = "selftest_flux2_klein_loop"


@pytest.fixture(scope="module")
def loop():
    return committed_fixture(LOOP)


class _Stub:
    """The velocity of the self-test's stub: bf16((table[k] + the first
    feature) - latents), at the step k whose time is sigma."""

    def __init__(self, table, schedule):
        self.table = [mx.array(t)[None] for t in table]
        self.step = {np.float32(s).tobytes(): k
                     for k, s in enumerate(schedule[:-1])}
        assert len(self.step) == len(table)
        self.cfg = SimpleNamespace(flow_map=False,
                                   in_channels=table.shape[1], patch=1)
        self.act_dtype = mx.bfloat16
        self.calls = []

    def __call__(self, x, sigma, feats):
        k = self.step[np.float32(sigma[0]).tobytes()]
        self.calls.append((k, feats.shape[1], float(feats[0, 0, 0])))
        v = (self.table[k] + feats[0, 0, 0]) - x
        return v.astype(mx.bfloat16).astype(mx.float32)


def _bits(a) -> np.ndarray:
    return np.asarray(a, np.float32).view(np.uint32)


def _sample(arrays, m, variant: str, family=None):
    """Every step's latents of gmlx's sample on the stub, and the stub."""
    fam = family or get_family("flux2_klein")
    res = m["results"][variant]
    stub = _Stub(arrays[f"{variant}_table"], arrays[f"{variant}_schedule"])
    pipe = api.ImagePipeline(dit=stub, family=fam, variant=variant)
    cond = Conditioning(mx.array(arrays["features"]))
    neg = (Conditioning(mx.array(arrays["features_empty"]))
           if variant == "base" else None)
    _, _, h, w = arrays["noise"].shape
    after = []
    with mx.stream(mx.cpu):
        api.sample(pipe, cond, neg=neg, height=h * 16, width=w * 16,
                   steps=res["steps"], noise=mx.array(arrays["noise"]),
                   on_step=lambda i, x: after.append(np.array(x)))
    return np.concatenate(after), stub, pipe


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_the_loop_is_the_reference_loop_bit_for_bit(loop, variant):
    arrays, m = loop
    res = m["results"][variant]
    steps = res["steps"]
    schedule = arrays[f"{variant}_schedule"]
    got, stub, pipe = _sample(arrays, m, variant)
    assert pipe.defaults.cfg == res["guidance"]
    tokens = m["inputs"]["tokens"]
    assert np.array_equal(_bits(pipe.sigmas(steps, tokens)), _bits(schedule))
    want = arrays[f"{variant}_latents"]
    same = [bool(np.array_equal(_bits(g), _bits(w)))
            for g, w in zip(got, want)]
    assert len(same) == steps and all(same), same
    # The latents stay bf16 values after every step.
    assert np.array_equal(_bits(got) & 0xFFFF, np.zeros_like(_bits(got)))
    plain = float(arrays["features"][0, 0])
    empty = float(arrays["features_empty"][0, 0])
    if variant == "distilled":
        assert res["rows"] == [1] * steps
        assert [c[0] for c in stub.calls] == list(range(steps))
        assert {c[2] for c in stub.calls} == {plain}
    else:
        # denoise_cfg runs the empty prompt's row beside the prompt's.
        assert res["rows"] == [2] * steps
        assert [c[0] for c in stub.calls] == sorted(list(range(steps)) * 2)
        assert [c[2] for c in stub.calls] == [plain, empty] * steps


@pytest.mark.parametrize("variant", ["distilled", "base"])
@pytest.mark.parametrize("rules", [
    None, LoopRules(noise=None, act_update=True, act_latents=True),
    LoopRules(noise="bfloat16", act_update=False, act_latents=True),
    LoopRules(noise="bfloat16", act_update=True, act_latents=False)],
    ids=["no rules", "float32 noise", "float32 update", "float32 latents"])
def test_each_loop_rule_is_needed(loop, variant, rules):
    arrays, m = loop
    fam = dataclasses.replace(get_family("flux2_klein"), loop=rules)
    got, _, _ = _sample(arrays, m, variant, fam)
    want = arrays[f"{variant}_latents"]
    assert not np.array_equal(_bits(got[-1]), _bits(want[-1]))


def test_only_flux2_klein_keeps_its_latents_in_the_activation_dtype():
    rules = {f.name: f.loop for f in families()}
    assert rules["flux2_klein"] == LoopRules(noise="bfloat16",
                                             act_update=True,
                                             act_latents=True)
    assert rules["flux2_klein_9b"] is rules["flux2_klein"]
    assert rules["flux3_klein"] == LoopRules(noise="bfloat16",
                                             act_update=True)
    assert all(r is None or not r.act_latents
               for n, r in rules.items()
               if n not in ("flux2_klein", "flux2_klein_9b"))


def test_an_fp32_run_keeps_fp32_latents(loop):
    """With fp32 activations the latents are not rounded, as the oracle's
    fp32 run keeps them."""
    arrays, m = loop
    stub = _Stub(arrays["distilled_table"], arrays["distilled_schedule"])
    stub.act_dtype = mx.float32
    pipe = api.ImagePipeline(dit=stub, family=get_family("flux2_klein"),
                             variant="distilled")
    _, _, h, w = arrays["noise"].shape
    after = []
    with mx.stream(mx.cpu):
        api.sample(pipe, Conditioning(mx.array(arrays["features"])),
                   height=h * 16, width=w * 16,
                   steps=m["results"]["distilled"]["steps"],
                   noise=mx.array(arrays["noise"]),
                   on_step=lambda i, x: after.append(np.array(x)))
    assert (_bits(np.concatenate(after)) & 0xFFFF).any()
