"""The hooks a family's registry row gives the shared image code, on toy
families: the step in parts under the pipeline's two memos, the schedule,
the size multiple, a load without an encoder, and a conditioning with
reference images. CPU."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api, memfit
from gmlx.image.fingerprint import ImageModelError
from gmlx.image.registry import ConstantsRow, StepParts, get_family


class _State:
    def __init__(self, *arrays):
        self._arrays = list(arrays)

    def arrays(self) -> list:
        return self._arrays


def _parts_pipe(state_bytes: int = 1 << 20, rows=4, sized: bool = True):
    """A pipeline whose family gives step parts, and their call counts.
    v = latents * 2 sigma + mean of the features. A state is ``rows``
    float32 values, one count or a count per conditioning key. ``held``
    lists the bytes the state cache holds at every build of a state.
    ``sized`` gives the family the size of a state before its build."""
    calls = {"modulation": 0, "conditioning": 0, "forward": 0, "held": []}

    def n_rows(cond):
        return rows[cond.key] if isinstance(rows, dict) else rows

    def modulation(dit, sigma):
        calls["modulation"] += 1
        assert isinstance(sigma, np.float32)
        return _State(mx.array(sigma) * 2)

    def conditioning(dit, cond):
        calls["conditioning"] += 1
        calls["held"].append(pipe.captions.bytes)
        return _State(mx.zeros((n_rows(cond),), mx.float32)
                      + cond.features.mean())

    def forward(dit, x, sigma, cond, state, mod):
        calls["forward"] += 1
        return x * mod.arrays()[0] + state.arrays()[0][0]

    fam = replace(get_family("z_image"), step=StepParts(
        modulation, conditioning, forward, state_bytes,
        (lambda dit, cond: 4 * n_rows(cond)) if sized else None))
    dit = SimpleNamespace(cfg=SimpleNamespace(in_channels=4, patch=2),
                          act_dtype=mx.bfloat16)
    pipe = api.ImagePipeline(dit=dit, family=fam, variant="turbo")
    return pipe, calls


def _cond(value: float, key: str = "") -> api.Conditioning:
    return api.Conditioning(mx.full((3, 5), value, mx.float32), key)


def _counts(calls: dict) -> dict:
    return {k: v for k, v in calls.items() if k != "held"}


def test_step_parts_run_a_row_under_the_two_memos():
    pipe, calls = _parts_pipe()
    x = mx.arange(32, dtype=mx.float32).reshape(1, 4, 2, 4)
    v = api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert np.array_equal(np.array(v), np.array(x) * 1.0 + 0.25)
    api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert _counts(calls) == {"modulation": 1, "conditioning": 1,
                              "forward": 2}
    # Another sigma is another modulation, another key another state.
    api.denoise_step(pipe, x, 0.25, _cond(0.25, "a"))
    api.denoise_step(pipe, x, 0.25, _cond(0.5, "b"))
    assert _counts(calls) == {"modulation": 2, "conditioning": 2,
                              "forward": 4}
    # A conditioning without a key is never cached.
    api.denoise_step(pipe, x, 0.25, _cond(0.5))
    api.denoise_step(pipe, x, 0.25, _cond(0.5))
    assert calls["conditioning"] == 4 and len(pipe.captions) == 2
    # Both memo keys hold the activation dtype.
    pipe.dit.act_dtype = mx.float32
    api.denoise_step(pipe, x, 0.25, _cond(0.5, "b"))
    assert calls["modulation"] == 3 and calls["conditioning"] == 5
    # Cleared caches change nothing but the work.
    pipe.captions.clear()
    pipe.modulations.clear()
    again = api.denoise_step(pipe, x, 0.5, _cond(0.25, "a"))
    assert np.array_equal(np.array(again), np.array(v))


def test_step_parts_rows_take_their_own_sigma_and_conditioning():
    pipe, calls = _parts_pipe()
    x = mx.arange(64, dtype=mx.float32).reshape(2, 4, 2, 4)
    conds = [_cond(0.25, "a"), _cond(0.5, "b")]
    v = api.denoise_step(pipe, x, [0.5, 0.25], conds)
    for i, s in enumerate((0.5, 0.25)):
        one = api.denoise_step(pipe, x[i:i + 1], s, conds[i])
        assert np.array_equal(np.array(v[i:i + 1]), np.array(one))
    assert calls["modulation"] == 2 and calls["conditioning"] == 2


def _key(pipe, name: str) -> tuple:
    return (name, pipe.default_table, str(mx.bfloat16))


@pytest.mark.parametrize("pos, neg", [(80, 70), (70, 80), (75, 75)],
                         ids=["positive_larger", "negative_larger", "equal"])
def test_the_state_cache_keeps_both_states_of_a_guided_request(pos, neg):
    # A state is 4 bytes a row and the bound is 300 bytes, so the two
    # states of the request pass the bound together in every case.
    rows = {"pos": pos, "neg": neg, "old": 20}

    def run(bound):
        pipe, calls = _parts_pipe(state_bytes=bound, rows=rows)
        x0 = mx.zeros((1, 4, 2, 4), mx.float32)
        api.denoise_step(pipe, x0, 0.125, _cond(0.5, "old"))
        x = api.sample(pipe, _cond(1.0, "pos"), neg=_cond(0.25, "neg"),
                       cfg=4.0, height=32, width=32, seed=1, steps=4)
        return pipe, calls, np.array(x)

    pipe, calls, x = run(300)
    assert _counts(calls) == {"modulation": 5, "conditioning": 3,
                              "forward": 9}
    # The request's own states pass the bound, so the older state went
    # before the first of them was built, and neither shed the other.
    assert calls["held"] == [0, 0, 4 * pos]
    assert len(pipe.captions) == 2
    assert pipe.captions.bytes == 4 * (pos + neg)
    assert pipe.scope is None
    assert np.array_equal(x, run(1 << 20)[2])
    # Under a bound that holds all three the older state stays.
    assert len(run(1 << 20)[0].captions) == 3


def test_the_state_cache_sheds_other_states_to_the_room_the_scope_leaves():
    x = mx.zeros((1, 4, 2, 4), mx.float32)
    # Small states stay up to the bound: the state of a bare call is 16
    # bytes, and 284 bytes of room hold 17 others.
    pipe, calls = _parts_pipe(state_bytes=300, rows=4)
    for i in range(24):
        api.denoise_step(pipe, x, 0.5, _cond(1.0, f"k{i}"))
    assert len(pipe.captions) == 18 and pipe.captions.bytes == 288
    assert max(calls["held"]) == 272
    for i in range(6, 24):
        api.denoise_step(pipe, x, 0.5, _cond(1.0, f"k{i}"))
    assert calls["conditioning"] == 24
    # A hit sheds too: under a smaller bound the next call, whose state
    # is held, sheds the others to the new room, the oldest use first.
    pipe.family = replace(pipe.family, step=replace(pipe.family.step,
                                                    state_bytes=64))
    api.denoise_step(pipe, x, 0.5, _cond(1.0, "k9"))
    assert calls["conditioning"] == 24 and pipe.captions.bytes == 64
    assert [k[0] for k in pipe.captions._items] == ["k21", "k22", "k23",
                                                    "k9"]


def test_two_conditionings_in_separate_calls_share_the_bound_or_a_scope():
    """A state of 64 rows is 256 bytes, over half the bound of 300. Each
    bare call sheds the other's state and builds its own again. One scope
    around the calls, or one call with both rows, holds both."""
    x = mx.zeros((1, 4, 2, 4), mx.float32)
    pos, neg = _cond(1.0, "pos"), _cond(0.25, "neg")
    pipe, calls = _parts_pipe(state_bytes=300, rows=64)
    for _ in range(3):
        api.denoise_step(pipe, x, 0.5, pos)
        api.denoise_step(pipe, x, 0.5, neg)
    assert calls["conditioning"] == 6 and set(calls["held"]) == {0}
    assert len(pipe.captions) == 1 and pipe.captions.bytes == 256
    pipe, calls = _parts_pipe(state_bytes=300, rows=64)
    with api.state_scope(pipe, [pos, neg]):
        assert [n for _, n in pipe.scope] == [256, 256]
        for _ in range(3):
            api.denoise_step(pipe, x, 0.5, pos)
            api.denoise_step(pipe, x, 0.5, neg)
    assert calls["conditioning"] == 2 and calls["held"] == [0, 256]
    assert pipe.scope is None and pipe.captions.bytes == 512
    # A scope inside another changes nothing, and an error closes it.
    with pytest.raises(RuntimeError), api.state_scope(pipe, pos):
        with api.state_scope(pipe, neg):
            assert [k[0] for k, _ in pipe.scope] == ["pos"]
        assert pipe.scope is not None
        raise RuntimeError
    assert pipe.scope is None
    pipe, calls = _parts_pipe(state_bytes=300, rows=64)
    xx = mx.zeros((2, 4, 2, 4), mx.float32)
    for _ in range(3):
        api.denoise_step(pipe, xx, 0.5, [pos, neg])
    assert calls["conditioning"] == 2 and pipe.captions.bytes == 512
    # Both fit a bound of 512, so separate bare calls keep both.
    pipe, calls = _parts_pipe(state_bytes=512, rows=64)
    for _ in range(3):
        api.denoise_step(pipe, x, 0.5, pos)
        api.denoise_step(pipe, x, 0.5, neg)
    assert calls["conditioning"] == 2 and pipe.captions.bytes == 512


def test_a_state_is_never_built_beside_a_state_the_rule_sheds():
    """With the size of a state known before its build, the cache makes
    room first. Without it the cache sheds after the build."""
    x = mx.zeros((1, 4, 2, 4), mx.float32)
    for sized, at_build in ((True, [0, 0, 0]), (False, [0, 256, 256])):
        pipe, calls = _parts_pipe(state_bytes=300, rows=64, sized=sized)
        for key in ("a", "b", "c"):
            api.denoise_step(pipe, x, 0.5, _cond(1.0, key))
            assert pipe.captions.bytes == 256
            assert pipe.captions.get(_key(pipe, key)) is not None
        assert calls["held"] == at_build
    # A conditioning without a key is never held, and it is of the scope:
    # its bytes come off the room of the others.
    pipe, calls = _parts_pipe(state_bytes=300, rows={"a": 16, "": 64})
    api.denoise_step(pipe, x, 0.5, _cond(1.0, "a"))
    assert pipe.captions.bytes == 64
    api.denoise_step(pipe, x, 0.5, _cond(1.0))
    assert pipe.captions.bytes == 0 and calls["held"] == [0, 0]
    # The bound of the states is the family's, not the cache's own, and a
    # state of the scope stays over both.
    pipe, calls = _parts_pipe(state_bytes=64, rows=64)
    pipe.captions.max_bytes = 100
    for _ in range(2):
        api.denoise_step(pipe, x, 0.5, _cond(1.0, "a"))
    assert calls["conditioning"] == 1 and pipe.captions.bytes == 256


def test_a_cache_sheds_by_its_bound_or_by_the_rule_of_its_caller():
    lru = api.ByteLRU(100)
    lru.put("a", 1, 150)
    assert len(lru) == 0
    # With shed off an entry stays whatever its bytes and sheds nothing.
    lru.put("a", 1, 150, shed=False)
    lru.put("b", 2, 120, shed=False)
    lru.put("c", 3, 10, shed=False)
    assert len(lru) == 3 and lru.bytes == 280
    # The others go, the least recently used first, down to the room.
    assert lru.get("a") == 1
    lru.shed_others({"c"}, 150)
    assert lru.get("b") is None and lru.bytes == 160
    lru.shed_others({"c"}, 150)
    assert lru.bytes == 160
    lru.shed_others({"a"}, 0)
    assert lru.get("c") is None and lru.get("a") == 1 and lru.bytes == 150
    lru.shed_others((), 0)
    assert len(lru) == 0 and lru.bytes == 0
    # A put that sheds holds the bound.
    lru.put("d", 4, 60)
    lru.put("e", 5, 60)
    assert lru.get("d") is None and lru.bytes == 60


def test_a_family_schedule_reads_the_token_count_of_the_request():
    seen = []

    def schedule(steps, *, tokens, variant):
        seen.append((steps, tokens, variant))
        return np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)

    pipe, calls = _parts_pipe()
    pipe.family = replace(pipe.family, schedule=schedule)
    assert np.array_equal(pipe.sigmas(4, 96), schedule(4, tokens=96,
                                                       variant="turbo"))
    # Z-Image tokens are 16 pixels a side: 64x48 is 4 x 3 of them.
    x = api.sample(pipe, _cond(0.25, "a"), height=48, width=64, seed=3,
                   steps=2)
    assert seen[-1] == (2, 12, "turbo") and calls["forward"] == 2
    assert x.shape == (1, 4, 6, 8)
    # An outside list still replaces the family's schedule.
    n = len(seen)
    api.sample(pipe, _cond(0.25, "a"), height=48, width=64, seed=3,
               sigmas=[1.0, 0.5, 0.0])
    assert len(seen) == n and calls["forward"] == 4


def test_the_size_multiple_comes_from_the_family_row_when_it_has_one():
    pipe, _ = _parts_pipe()
    assert pipe.size_multiple == 16
    row = pipe.family.constants
    pipe.family = replace(pipe.family, constants=ConstantsRow(
        {**row.values, "size_multiple": 32}, row.repo, row.revision))
    assert pipe.size_multiple == 32
    with pytest.raises(ValueError, match="multiples of 32, not 48x64"):
        api.sample(pipe, _cond(0.25), height=48, width=64, seed=0, steps=1)
    x = api.sample(pipe, _cond(0.25), height=64, width=32, seed=0, steps=1)
    assert x.shape == (1, 4, 8, 4)


def test_a_family_without_an_encoder_loads_when_none_is_asked(monkeypatch):
    from gmlx.image import dtypes, loader

    fam = replace(get_family("z_image"), name="toy", title="Toy")
    ident = SimpleNamespace(family=fam, variant="default")
    loads = []

    def load_dit(path, **kw):
        loads.append(path)
        dit = SimpleNamespace(act_dtype=kw["act_dtype"])
        return dit, SimpleNamespace(identified=ident)

    monkeypatch.setattr(api, "_header_bytes", lambda p, kind: 0)
    monkeypatch.setattr(memfit, "gate_load", lambda *a, **k: None)
    monkeypatch.setattr(loader, "identify_dit", lambda path, **kw: ident)
    monkeypatch.setattr(loader, "load_dit", load_dit)
    monkeypatch.setattr(
        dtypes, "image_activation",
        lambda name, **kw: dtypes.Activation(mx.bfloat16, mx.bfloat16, 16))
    pipe = api.load("d.gguf")
    assert pipe.family is fam and pipe.encoder is None and pipe.vae is None
    with pytest.raises(ValueError, match="no text encoder"):
        api.encode_text(pipe, "a prompt")
    # With an encoder asked for, the family is refused before the DiT loads.
    with pytest.raises(ImageModelError, match="'toy' has no text encoder"):
        api.load("d.gguf", text_encoder="e.gguf")
    # A borrowed encoder model asks for the family's encoder too.
    with pytest.raises(ImageModelError, match="'toy' has no text encoder"):
        api.load("d.gguf", encoder_model=(object(), object()))
    assert loads == ["d.gguf"]


def test_a_conditioning_carries_reference_latents_and_image_slots():
    plain = _cond(0.5, "k")
    assert plain.ref_latents == () and plain.image_slots == ()
    ref = mx.zeros((4, 2, 4), mx.float32)
    edit = api.Conditioning(plain.features, "e", ref_latents=(ref,),
                            image_slots=(1, 2))
    assert edit.ref_latents[0] is ref and edit.image_slots == (1, 2)
    assert edit.features is plain.features and edit.key == "e"
