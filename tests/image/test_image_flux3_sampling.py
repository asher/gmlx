"""The FLUX.3 klein sampling loop on the CPU.

On a stub DiT, gmlx's sample holds bit for bit to the loop of BFL's own
Pipeline.generate, as scripts/image_oracle/check6_flux3_klein.py
--self-test records it: the distilled flow map, which takes the time each
step goes to, and Base, which guides the prompt's row by the
unconditional one. Each loop rule of the family is needed for that. A
flow map refuses guidance and other samplers. A pipeline step builds the
reference tokens with the image rope scale of the config and a
modulation for each target time.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import flux3_gates as fg
from oracle_fixture import committed_fixture

from gmlx.image import api
from gmlx.image.conditioning import Conditioning
from gmlx.image.families.flux3_klein import model as fm
from gmlx.image.registry import LoopRules, get_family
from gmlx.image.samplers import Euler

LOOP = "selftest_flux3_klein_loop"
TINY = "selftest_flux3_klein_tiny"
# The tiny model in fp32 on the CPU against the oracle, as in
# test_image_flux3_model.py.
TOL = 4e-6


@pytest.fixture(scope="module")
def loop():
    return committed_fixture(LOOP)


class _Stub:
    """The velocity of the self-test's stub: bf16((table[k] + the first
    feature) - latents), at the step k whose time is sigma."""

    def __init__(self, table, schedule, flow_map: bool):
        self.table = [mx.array(t)[None] for t in table]
        self.step = {np.float32(s).tobytes(): k
                     for k, s in enumerate(schedule[:-1])}
        self.cfg = SimpleNamespace(flow_map=flow_map,
                                   in_channels=table.shape[1], patch=1)
        self.act_dtype = mx.bfloat16
        self.calls = []

    def __call__(self, x, sigma, feats, target=None):
        k = self.step[np.float32(sigma[0]).tobytes()]
        self.calls.append((k, None if target is None
                           else np.float32(target[0]), feats.shape[1]))
        v = (self.table[k] + feats[0, 0, 0]) - x
        return v.astype(mx.bfloat16).astype(mx.float32)


def _bits(a) -> np.ndarray:
    return np.asarray(a, np.float32).view(np.uint32)


def _sample(arrays, m, variant: str, family=None):
    """Every step's latents of gmlx's sample on the stub, and the stub."""
    fam = family or get_family("flux3_klein")
    table = arrays[f"{variant}_table"]
    stub = _Stub(table, arrays[f"{variant}_schedule"], variant == "distilled")
    pipe = api.ImagePipeline(dit=stub, family=fam, variant=variant)
    cond = Conditioning(mx.array(arrays["features"]))
    neg = (Conditioning(mx.array(arrays["features_empty"]))
           if variant == "base" else None)
    _, _, h, w = arrays["noise"].shape
    after = []
    with mx.stream(mx.cpu):
        api.sample(pipe, cond, neg=neg, height=h * 16, width=w * 16,
                   noise=mx.array(arrays["noise"]),
                   on_step=lambda i, x: after.append(np.array(x)))
    return np.concatenate(after), stub, pipe


@pytest.mark.parametrize("variant", ["distilled", "base"])
def test_the_loop_is_the_reference_loop_bit_for_bit(loop, variant):
    arrays, m = loop
    res = m["results"][variant]
    schedule = arrays[f"{variant}_schedule"]
    steps = res["steps"]
    got, stub, pipe = _sample(arrays, m, variant)
    assert pipe.defaults.steps == steps
    assert pipe.defaults.cfg == (res["guidance"] or 1.0)
    assert np.array_equal(_bits(pipe.sigmas(steps)), _bits(schedule))
    want = arrays[f"{variant}_latents"]
    same = [bool(np.array_equal(_bits(g), _bits(w)))
            for g, w in zip(got, want)]
    assert len(same) == steps and all(same), same
    feats = arrays["features"].shape[0]
    empty = arrays["features_empty"].shape[0]
    if variant == "distilled":
        assert res["prompts"] == ["plain"]
        assert [c[0] for c in stub.calls] == list(range(steps))
        assert [c[1] for c in stub.calls] == list(schedule[1:])
        assert {c[2] for c in stub.calls} == {feats}
    else:
        # The pipeline encodes the empty prompt for the unconditional row.
        assert res["prompts"] == ["", "plain"]
        assert sorted(c[0] for c in stub.calls) == sorted(
            list(range(steps)) * 2)
        assert all(c[1] is None for c in stub.calls)
        assert [c[2] for c in stub.calls] == [feats, empty] * steps


@pytest.mark.parametrize("variant", ["distilled", "base"])
@pytest.mark.parametrize("rules", [
    None, LoopRules(noise=None, act_update=True),
    LoopRules(noise="bfloat16", act_update=False)],
    ids=["no rules", "float32 noise", "float32 update"])
def test_each_loop_rule_is_needed(loop, variant, rules):
    arrays, m = loop
    fam = dataclasses.replace(get_family("flux3_klein"), loop=rules)
    got, _, _ = _sample(arrays, m, variant, fam)
    want = arrays[f"{variant}_latents"]
    assert not np.array_equal(_bits(got[-1]), _bits(want[-1]))


def test_a_flow_map_refuses_guidance_and_other_samplers(loop):
    arrays, _ = loop
    stub = _Stub(arrays["distilled_table"], arrays["distilled_schedule"],
                 True)
    pipe = api.ImagePipeline(dit=stub, family=get_family("flux3_klein"),
                             variant="distilled")
    cond = Conditioning(mx.array(arrays["features"]))
    neg = Conditioning(mx.array(arrays["features_empty"]))
    kw = dict(height=64, width=96, seed=0)
    with pytest.raises(ValueError, match="flow map, which samples without"):
        api.sample(pipe, cond, neg=neg, cfg=3.5, **kw)
    with pytest.raises(ValueError, match="euler only, not uni_pc_bh2"):
        api.sample(pipe, cond, sampler="uni_pc_bh2", **kw)
    assert not stub.calls


def test_euler_rounds_its_product_and_guide_each_op():
    s = np.array([1.0, 0.7, 0.0], np.float32)
    x = mx.array([[0.1, -2.0]], mx.float32)
    v = mx.array([[1.0, 3.0]], mx.float32)
    dt = np.float32(s[1] - s[0])
    plain = Euler(s).step(x, v, 0)
    rounded = Euler(s, product=mx.bfloat16).step(x, v, 0)
    p = (mx.array(dt) * v).astype(mx.bfloat16).astype(mx.float32)
    assert mx.array_equal(rounded, x + p).item()
    assert not mx.array_equal(rounded, plain).item()
    pos = mx.array([1.0078125, 2.0], mx.float32)
    neg = mx.array([0.0, 1.9921875], mx.float32)
    r = lambda a: a.astype(mx.bfloat16).astype(mx.float32)  # noqa: E731
    want = r(neg + r(mx.array(np.float32(3.3)) * r(pos - neg)))
    assert mx.array_equal(api.guide(pos, neg, 3.3, mx.bfloat16), want).item()
    assert not mx.array_equal(api.guide(pos, neg, 3.3), want).item()
    assert mx.array_equal(api.guide(pos, neg, 3.3, mx.float32),
                          api.guide(pos, neg, 3.3)).item()


@pytest.fixture(scope="module")
def tiny_pipe():
    weights, wm = committed_fixture(TINY + "_weights")
    arrays, _ = committed_fixture(TINY)
    model = fg.tiny_model(weights, wm["inputs"]["config"])
    pipe = api.ImagePipeline(dit=model, family=get_family("flux3_klein"),
                             variant="distilled")
    refs = tuple(mx.array(arrays[f"refs.{i}"])[0]
                 for i in range(fg.n_refs(arrays)))
    cond = Conditioning(mx.array(arrays["features"]), ref_latents=refs)
    return pipe, cond, arrays


def _step(pipe, cond, arrays, target=None):
    with mx.stream(mx.cpu):
        v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                             arrays["sigma"], cond,
                             target=arrays["target"] if target is None
                             else target)
        return np.array(v)


def test_a_step_gives_the_references_the_rope_scale_of_the_config(
        tiny_pipe, monkeypatch):
    """The reference tokens of a pipeline step take the config's image
    rope scale, BFL's IMAGE_ROPE_SCALE 0.5, and the step is the oracle's.
    Any other scale for the references alone misses the bound."""
    pipe, cond, arrays = tiny_pipe
    real = fm.References.of
    scales = []

    def spy(latents, scale):
        scales.append(scale)
        return real(latents, scale)

    monkeypatch.setattr(fm.References, "of", spy)
    v = _step(pipe, cond, arrays)
    assert scales == [0.5] and pipe.dit.cfg.image_rope_scale == 0.5
    assert get_family("flux3_klein").constants.values[
        "image_rope_scale"] == 0.5
    assert fg.rel(v, arrays["velocity"]) < TOL
    monkeypatch.setattr(fm.References, "of",
                        lambda latents, scale: real(latents, 1.0))
    assert fg.rel(_step(pipe, cond, arrays), arrays["velocity"]) > 100 * TOL


def test_the_modulation_of_a_step_is_kept_per_target(tiny_pipe):
    pipe, cond, arrays = tiny_pipe
    first = _step(pipe, cond, arrays)
    other = _step(pipe, cond, arrays, target=np.float32(0.25))
    again = _step(pipe, cond, arrays)
    assert np.array_equal(first, again)
    assert fg.rel(other, first) > 100 * TOL
    model = pipe.dit
    lat, feats, refs, sigma, _ = fg.inputs(model, arrays)
    with mx.stream(mx.cpu):
        direct = np.array(model(lat, sigma, feats, refs=refs,
                                target=np.full((1,), 0.25, np.float32)))
    assert np.array_equal(other, direct)
