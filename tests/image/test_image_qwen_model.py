"""The Qwen-Image 2.1 DiT on a tiny random model, CPU.

The prefill and the cached target forward together equal one forward of
the whole joint sequence under an explicit block-causal mask, for a text
prefix and for prefixes with reference images, and three wrong models
each break the equality. qwen_joint.py holds that joint forward. A
denoise step through a loaded pipeline is a pure function: steps in any
order, with the caches cleared between some of them, give the same
arrays. The rest covers the two MLP layouts, batches, the timestep chain
and the refusals of the config and of float16.
"""
from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

import qwen_tiny as qt
from oracle_fixture import committed_fixture, fx
from qwen_joint import joint_forward

from gmlx.image import api, memfit, registry
from gmlx.image.conditioning import array_bytes
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.qwen_image21 import layout
from gmlx.image.families.qwen_image21 import model as qm
from gmlx.image.ops import kq_routes
from gmlx.image.schedule import qwen21_sigmas

# The equivalence holds to this relative L2 in fp32, and a wrong model
# must miss it by FAULT times at least.
TOL = 1e-5
FAULT = 20.0
SIGMA = float(qwen21_sigmas(40, 24)[7])


@pytest.fixture(scope="module")
def tiny():
    return qt.model_of(qt.weights(0))


def _cached(model, cond, lat, sigma, prefix_taps=None, target_taps=None):
    prefix = qm.prefill_prefix(model, cond, prefix_taps)
    v = qm.forward_target(model, lat, sigma, cond, prefix, target_taps)
    mx.eval(v)
    return v


def _rel(got, ref) -> float:
    return fx.rel_l2(np.array(got), np.array(ref))


CASES = ("text", "ref1", "ref2", "adjacent")


@pytest.mark.parametrize("case", CASES)
def test_cached_mode_equals_one_joint_forward_under_the_mask(tiny, case):
    cond, lat = qt.conditioning(case), qt.latents()
    with mx.stream(mx.cpu):
        joint, pre, tgt = {}, {}, {}
        want = joint_forward(tiny, cond, lat, SIGMA, taps=joint)
        got = _cached(tiny, cond, lat, SIGMA, pre, tgt)
        mx.eval(want, *joint.values())
    p = layout.prefix_layout(*qm.prefix_key(cond)).tokens
    worst = 0.0
    for name, x in joint.items():
        assert x.shape[1] == p + lat.shape[2] * lat.shape[3]
        worst = max(worst, _rel(pre[name], x[:, :p]), _rel(tgt[name],
                                                           x[:, p:]))
    vel = _rel(got, want)
    print(f"qwen cached against joint, {case}: prefix of {p} tokens, "
          f"blocks within {worst:.2e}, velocity within {vel:.2e}, gate "
          f"{TOL:.0e}")
    assert worst < TOL and vel < TOL


@pytest.mark.parametrize("case", CASES)
def test_the_equivalence_fails_for_each_wrong_model(tiny, case, monkeypatch):
    cond, lat = qt.conditioning(case), qt.latents()
    lay = layout.prefix_layout(*qm.prefix_key(cond))
    p = lay.tokens
    seen = {}
    with mx.stream(mx.cpu):
        got = _cached(tiny, cond, lat, SIGMA)
        # The last prefix token sees the target tokens.
        seen["a prefix token sees the target"] = _rel(got, joint_forward(
            tiny, cond, lat, SIGMA, leak=(p - 1, slice(p, None))))
        # It sees one of them. This is the weakest wrong model, held to
        # the gate alone.
        one = _rel(got, joint_forward(tiny, cond, lat, SIGMA,
                                      leak=(p - 1, p + 3)))
        # The first prefix token reads the sampled timestep's modulation.
        seen["a prefix token at the sampled timestep"] = _rel(
            got, joint_forward(tiny, cond, lat, SIGMA, sampled_rows=(0,)))
        want = joint_forward(tiny, cond, lat, SIGMA)
        with monkeypatch.context() as mp:
            mp.setattr(qm, "prefix_modulation",
                       lambda model: qm.modulation(model, SIGMA))
            seen["the prefill at the sampled timestep"] = _rel(
                _cached(tiny, cond, lat, SIGMA), want)
        if cond.ref_latents:
            def late(axes, theta, n_rows, slots, refs):
                pos = lay.positions.copy()
                for start, end, text in lay.segments:
                    if not text:
                        pos[start:end, 2] += 1
                return layout.rope_tables(pos, axes, theta)

            with monkeypatch.context() as mp:
                mp.setattr(layout, "prefix_tables", late)
                seen["the reference grid one position off"] = _rel(
                    _cached(tiny, cond, lat, SIGMA), want)
        assert _rel(_cached(tiny, cond, lat, SIGMA), want) < TOL
    print(f"qwen cached against joint, {case}, wrong models: "
          + ", ".join(f"{k} {v:.2e}" for k, v in seen.items())
          + f", a prefix token sees one target token {one:.2e}")
    assert one > 2 * TOL
    assert len(seen) == (4 if cond.ref_latents else 3)
    for name, miss in seen.items():
        assert miss > FAULT * TOL, (name, miss)


def test_no_mask_is_built_in_cached_mode(tiny, monkeypatch):
    from gmlx.image import ops

    cond, lat = qt.conditioning("ref2"), qt.latents()
    masks = []
    real = ops.sdpa

    def sdpa(q, k, v, mask=None):
        masks.append((q.shape[2], k.shape[2], mask is not None))
        return real(q, k, v, mask)

    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(tiny, cond)
        monkeypatch.setattr(ops, "sdpa", sdpa)
        monkeypatch.setattr(qm, "causal_mask", None)
        mx.eval(qm.forward_target(tiny, lat, SIGMA, cond, prefix))
    # One call per block: the target rows over the prefix and themselves.
    t = lat.shape[2] * lat.shape[3]
    assert masks == [(t, prefix.tokens + t, False)] * tiny.cfg.n_layers


def test_the_prefill_runs_one_attention_call_per_segment(tiny, monkeypatch):
    from gmlx.image import ops

    cond = qt.conditioning("ref2")
    calls = []
    real = ops.sdpa

    def sdpa(q, k, v, mask=None):
        calls.append((q.shape[2], k.shape[2], mask is not None))
        return real(q, k, v, mask)

    monkeypatch.setattr(ops, "sdpa", sdpa)
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(tiny, cond)
    # Rows 0 and 1 are text, a 2x2 reference, row 3, a 4x2 reference, and
    # four rows of text: each segment's keys end where it ends, and only
    # text segments carry a mask.
    want = [(2, 2, True), (4, 6, False), (1, 7, True), (8, 15, False),
            (4, 19, True)]
    assert calls == want * tiny.cfg.n_layers
    assert prefix.tokens == 19 and prefix.position == 2 + 2 + 1 + 4 + 4
    assert [k.shape for k in prefix.k] == [(1, 2, 19, 16)] * 2
    assert [v.shape for v in prefix.v] == [(1, 2, 19, 16)] * 2


def test_both_mlp_layouts_give_the_same_velocity():
    w = qt.weights(1)
    fused, pair = qt.model_of(w), qt.model_of(qt.unfuse(w))
    assert fused.cfg.fused_mlp and not pair.cfg.fused_mlp
    cond, lat = qt.conditioning("ref1"), qt.latents()
    with mx.stream(mx.cpu):
        a = _cached(fused, cond, lat, SIGMA)
        b = _cached(pair, cond, lat, SIGMA)
    assert _rel(b, a) < 5e-6


def test_a_batch_gives_each_row_its_own_sigma(tiny):
    cond = qt.conditioning("ref1")
    lat = qt.latents(batch=2)
    sig = np.asarray([SIGMA, 0.25], np.float32)
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(tiny, cond)
        both = qm.forward_target(tiny, lat, sig, cond, prefix)
        rows = [qm.forward_target(tiny, lat[i:i + 1], sig[i], cond, prefix)
                for i in range(2)]
        mx.eval(both, rows)
    assert both.shape == (2, 8, *qt.TARGET) and both.dtype == mx.float32
    for i in range(2):
        assert _rel(both[i:i + 1], rows[i]) < 5e-6
    assert _rel(rows[1], rows[0]) > 0.1


def test_the_timestep_chain_equals_the_reference_bit_for_bit(tiny):
    arrays, m = committed_fixture("check4_qwen_image21")
    n, (w, h) = m["inputs"]["chain"]["steps"], m["inputs"]["chain"]["size"]
    sigmas = arrays[f"sigmas_{n}_{w}x{h}"][:-1]
    assert np.array_equal(tiny.timesteps(sigmas, n), arrays["t_embedded"])
    one = tiny.timesteps(sigmas[3], 1)
    assert one.dtype == np.float32
    assert np.array_equal(one, arrays["t_embedded"][3:4])


def test_the_timestep_chain_reads_the_training_timesteps_of_the_row():
    from gmlx.image.registry import get_family

    cfg = qt.config_of(qt.weights(0))
    row = get_family("qwen_image21").constants.values
    assert cfg["num_train_timesteps"] == row["num_train_timesteps"] == 1000
    half = qm.QwenImage21DiT({**cfg, "num_train_timesteps": 500})
    s = np.array([0.3, 0.7001], np.float32)
    want = np.float32(1000) * ((s * np.float32(500)) / np.float32(1000))
    assert np.array_equal(half.timesteps(s, 2), want)
    assert not np.array_equal(
        qm.QwenImage21DiT(cfg).timesteps(s, 2), want)


def test_the_prefix_reads_t_zero_and_the_target_its_sigma(tiny):
    with mx.stream(mx.cpu):
        zero = qm.prefix_modulation(tiny)
        same = qm.modulation(tiny, 0.0)
        other = qm.modulation(tiny, SIGMA)
        mx.eval(zero.arrays(), same.arrays(), other.arrays())
    for a, b, c in zip(zero.arrays(), same.arrays(), other.arrays()):
        assert a.shape == (1, 1, tiny.cfg.dim)
        assert np.array_equal(np.array(a), np.array(b))
        assert not np.array_equal(np.array(a), np.array(c))
    # The gates are squashed once, where the modulation is formed.
    assert float(mx.abs(other.gate1).max()) <= 1.0
    assert float(mx.abs(other.gate2).max()) <= 1.0


def test_float16_and_wrong_configs_are_refused(tiny):
    with pytest.raises(ImageDtypeError, match="never runs in float16"):
        tiny.act_dtype = mx.float16
    assert tiny.act_dtype == mx.float32
    cfg = qt.config_of(qt.weights(0))
    for change, text in (
            ({"patch_size": 2}, "takes latents unpatched"),
            ({"causal_condition": False}, "causal condition"),
            ({"axes_dims_rope": [4, 6, 8]}, "sum to the head dim 16"),
            ({"axes_dims_rope": [3, 7, 6]}, "three even dims"),
            ({"modulation_dim": 96}, "widths disagree")):
        with pytest.raises(ValueError, match=text):
            qm.QwenImage21Config.from_dict({**cfg, **change})


def test_a_prefix_of_another_depth_and_bad_references_are_refused(tiny):
    cond, lat = qt.conditioning("ref1"), qt.latents()
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(tiny, cond)
        short = qm.PrefixCache(prefix.k[:1], prefix.v[:1], prefix.position,
                               prefix.built_for)
        with pytest.raises(ValueError, match="holds 1 blocks.*has 2"):
            qm.forward_target(tiny, lat, SIGMA, cond, short)
        flat = api.Conditioning(cond.features, "", ref_latents=(
            cond.ref_latents[0][:4],), image_slots=cond.image_slots)
        with pytest.raises(ValueError, match=r"must be \(8, h, w\)"):
            qm.prefill_prefix(tiny, flat)
        more = api.Conditioning(cond.features, "", ref_latents=(
            cond.ref_latents[0],), image_slots=(3, 4, 5))
        with pytest.raises(ValueError, match="3 image slots hold 12"):
            qm.prefill_prefix(tiny, more)


# A loaded pipeline with no encoder and no VAE.

@pytest.fixture
def pipe(tmp_path, monkeypatch):
    fam = qt.tiny_family()
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    path = qt.write(tmp_path / "tiny.gguf", qt.weights(0))
    return api.load(path, log=lambda *_: None)


def test_a_target_step_refuses_the_prefix_of_another_conditioning(tiny):
    cond, lat = qt.conditioning("ref1"), qt.latents()
    refused = "the prefix was built for another conditioning"
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(tiny, cond)
        assert prefix.built_for == qm.prefix_tag(cond)
        qm.forward_target(tiny, lat, SIGMA, cond, prefix)
        # Another key over the same rows, the same key over another
        # layout, and the same key and layout over other reference
        # latents.
        for other in (
                api.Conditioning(cond.features, "other",
                                 ref_latents=cond.ref_latents,
                                 image_slots=cond.image_slots),
                qt.conditioning("text", key=cond.key),
                qt.conditioning("ref2", key=cond.key),
                api.Conditioning(cond.features, cond.key,
                                 ref_latents=(cond.ref_latents[0] + 1,),
                                 image_slots=cond.image_slots)):
            with pytest.raises(ValueError, match=refused):
                qm.forward_target(tiny, lat, SIGMA, other, prefix)


def test_a_prefix_state_is_kept_under_the_reference_latents_too(pipe):
    """The key of a conditioning does not hold its reference latents, so
    the memo of prefix states is keyed by their hashes and the image
    slots beside it: one key over other reference latents is another
    state, and never the first one's prefix."""
    lat = qt.latents()
    ref = qt.conditioning("ref1", key="k")
    other = api.Conditioning(ref.features, "k",
                             ref_latents=(ref.ref_latents[0] + 1,),
                             image_slots=ref.image_slots)
    plain = qt.conditioning("text", key="k")
    got = [np.array(api.denoise_step(pipe, lat, SIGMA, c))
           for c in (ref, other, plain)]
    assert len(pipe.captions) == 3
    assert not np.array_equal(got[0], got[1])
    for c, want in zip((ref, other, plain), got):
        alone = api.Conditioning(c.features, "", ref_latents=c.ref_latents,
                                 image_slots=c.image_slots)
        assert np.array_equal(
            np.array(api.denoise_step(pipe, lat, SIGMA, alone)), want)
    assert {k[1] for k in pipe.captions._items} == {
        ref.ref_hashes, other.ref_hashes, ()}
    # One key over other text rows has no hash to tell it by, and the
    # state of the first is refused for the second.
    with pytest.raises(ValueError, match="built for another conditioning"):
        api.denoise_step(pipe, lat, SIGMA, _text("k", rows=5))


def test_a_state_is_kept_under_the_latent_size_of_its_target(pipe):
    cond = qt.conditioning("ref1", key="k")
    api.denoise_step(pipe, qt.latents(), SIGMA, cond)
    api.denoise_step(pipe, qt.latents(size=(4, 4)), SIGMA, cond)
    assert [k[3] for k in pipe.captions._items] == [tuple(qt.TARGET),
                                                    (4, 4)]
    assert _state_key(pipe, "k", (4, 4))[3] == (4, 4)
    # The plan of a request names the state of its own size: the other
    # one is another request's, and at a third size both are.
    kw = dict(conds=cond, budget=100e9, overcommit=False)
    one = pipe.captions.nbytes(_state_key(pipe, "k", (4, 4)))
    assert pipe.captions.bytes == 2 * one
    terms = [api.plan_request(pipe, height=h, width=w, **kw).terms[
        "prefix_cache"] for h, w in ((64, 64), (64, 96), (96, 96))]
    assert terms[0] == terms[1] == terms[2] - one


@pytest.mark.parametrize("case", ("ref1", "ref2", "adjacent"))
def test_a_prefix_with_references_is_a_function_of_its_conditioning(pipe,
                                                                    case):
    """The state a step keeps for a conditioning with references is equal
    in bits whatever the target of the step that built it: other latents,
    another sigma, another latent size."""
    cond = qt.conditioning(case, key="k")
    assert cond.ref_latents and cond.image_slots
    runs = [(qt.latents(), SIGMA),
            (qt.latents(seed=5), float(qwen21_sigmas(40, 24)[30])),
            (qt.latents(seed=2, size=(4, 4)), SIGMA)]
    states = []
    for lat, sigma in runs:
        pipe.captions.clear()
        api.denoise_step(pipe, lat, sigma, cond)
        ((state, held),) = pipe.captions._items.values()
        assert held == array_bytes(state.arrays())
        states.append(state)
    want = qm.prefill_prefix(pipe.dit, cond)
    for state in states:
        assert state.built_for == want.built_for == qm.prefix_tag(cond)
        assert state.position == want.position
        for got, ref in zip(state.arrays(), want.arrays()):
            assert mx.array_equal(got, ref).item()
    # The rows of the references are in it: the state of the text rows
    # alone is shorter by their tokens.
    text = api.Conditioning(cond.features[:cond.image_slots[0]], "t")
    refs = sum(int(z.shape[-2]) * int(z.shape[-1]) for z in cond.ref_latents)
    assert want.tokens - qm.prefill_prefix(pipe.dit, text).tokens == (
        refs + cond.features.shape[0] - len(cond.image_slots)
        - cond.image_slots[0])


def test_load_identifies_the_dit_once(tmp_path, monkeypatch):
    from gmlx.image import loader

    fam = qt.tiny_family()
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    calls, real = [], loader.identify
    monkeypatch.setattr(loader, "identify",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    path = qt.write(tmp_path / "tiny.gguf", qt.weights(0))
    pipe = api.load(path, log=lambda *_: None)
    assert len(calls) == 1 and pipe.family.name == "qwen_image21"
    # A DiT loaded on its own identifies its file.
    _, info = loader.load_dit(path, log=lambda *_: None)
    assert len(calls) == 2 and info.identified.family.name == "qwen_image21"


def test_a_pipeline_loads_without_an_encoder_or_a_vae(pipe):
    assert pipe.family.name == "qwen_image21" and pipe.variant == "default"
    assert pipe.encoder is None and pipe.vae is None
    assert type(pipe.dit) is qm.QwenImage21DiT
    assert pipe.dit.act_dtype == mx.bfloat16
    assert (pipe.vae_scale, pipe.size_multiple) == (16, 32)
    assert (pipe.defaults.steps, pipe.defaults.cfg) == (40, 1.0)
    with pytest.raises(ValueError, match="no text encoder"):
        api.encode_text(pipe, "a prompt")
    with pytest.raises(ValueError, match="no VAE"):
        api.vae_decode(pipe, qt.latents())


def test_the_command_reads_the_size_multiple_of_the_family_row():
    from gmlx.commands.image import _stride

    assert _stride(registry.get_family("qwen_image21")) == 32
    assert _stride(registry.get_family("z_image")) == 16


def test_denoise_step_is_pure_in_any_order_with_caches_cleared(pipe):
    conds = [qt.conditioning("ref1", key="pos"),
             qt.conditioning("text", seed=1, key="neg")]
    sig = qwen21_sigmas(6, 24)[:-1]
    xs = [qt.latents(seed=i) for i in range(len(sig))]
    calls = [(i, c) for i in range(len(sig)) for c in conds]
    base = [np.array(api.denoise_step(pipe, xs[i], sig[i], c))
            for i, c in calls]
    assert len(pipe.captions) == 2 and len(pipe.modulations) == len(sig)
    order = np.random.default_rng(3).permutation(len(calls))
    assert list(order) != sorted(order)
    cleared = 0
    for n, j in enumerate(order):
        if n % 3 == 1:
            pipe.captions.clear()
            cleared += 1
        if n % 4 == 2:
            pipe.modulations.clear()
        if n % 5 == 3:
            layout.clear_caches()
        i, c = calls[j]
        again = np.array(api.denoise_step(pipe, xs[i], sig[i], c))
        assert np.array_equal(again, base[j]), (n, i, c.key)
    assert cleared >= 3
    # A conditioning without a key is never cached and gives the same.
    i, c = calls[4]
    bare = api.Conditioning(c.features, "", ref_latents=c.ref_latents,
                            image_slots=c.image_slots)
    held = len(pipe.captions)
    assert np.array_equal(
        np.array(api.denoise_step(pipe, xs[i], sig[i], bare)), base[4])
    assert len(pipe.captions) == held


def _state_key(pipe, name: str, size=None) -> tuple:
    """The key the bf16 pipeline keeps the state of a conditioning named
    ``name`` under, which holds the hashes of its reference latents and
    its image slots, the target's latent size, and ends with the kernel
    routes in force. Read from the cache, which must hold one such state
    at ``size``, by default at any size."""
    tail = (pipe.default_table, str(mx.bfloat16), kq_routes.active())
    (key,) = [k for k in pipe.captions._items
              if k[0] == name and k[4:] == tail
              and (size is None or k[3] == tuple(size))]
    return key


def test_the_prefix_cache_is_keyed_by_conditioning_table_and_dtype(pipe):
    cond = qt.conditioning("ref1", key="pos")
    x = qt.latents()
    api.denoise_step(pipe, x, SIGMA, cond)
    api.denoise_step(pipe, x, 0.25, cond)
    key = _state_key(pipe, "pos")
    state = pipe.captions.get(key)
    assert type(state) is qm.PrefixCache and len(pipe.captions) == 1
    # 2 blocks of K and V, 15 prefix tokens, 32 wide, 2 bytes each.
    assert state.tokens == 15
    assert pipe.captions.bytes == 2 * 2 * 15 * 32 * 2
    assert len(pipe.modulations) == 2
    for s in (SIGMA, 0.25):
        assert type(pipe.modulations.get(
            (np.float32(s).tobytes(), *key[4:]))) is qm.Modulation


def test_the_prefix_cache_bound_holds_the_text_states_of_a_guided_request():
    """Both text prefixes of a guided request fit the family's bound
    twice over, so the text states of the request before stay too."""
    from gmlx.image.registry import get_family

    row = memfit.PEAKS["qwen_image21"]
    one = row.prefix.cache_bytes(row.text_rows, 2)
    assert 4 * one <= get_family("qwen_image21").step.state_bytes


def test_the_prefix_cache_bound_keeps_no_reference_state_of_another_request():
    """The ceiling of the family's bound: it is the four text states and
    no more, so the state of one 1024 pixel reference, which a request
    that does not run it would hold for nothing, is over it."""
    from gmlx.image.registry import get_family

    row = memfit.PEAKS["qwen_image21"]
    bound = get_family("qwen_image21").step.state_bytes
    assert bound == 4 * row.prefix.cache_bytes(row.text_rows, 2) == 1 << 30
    assert bound < row.prefix.cache_bytes(row.prefix.rows(0, 1, 1024), 2)
    # The plan reads the same number: other states count up to the bound
    # less the request's own.
    kw = dict(family="qwen_image21", height=1024, width=1024,
              budget=100e9, overcommit=False, prefix_bound=bound)
    own = memfit.plan_memory(4 << 30, 0, 0, **kw).terms["prefix_cache"]
    held = memfit.plan_memory(4 << 30, 0, 0, prefix_other=8 << 30,
                              **kw).terms["prefix_cache"]
    assert held == bound and own == bound // 4


def test_the_memory_plan_counts_the_prefix_cache_a_pipeline_holds(pipe):
    """Five text rows and one 2 x 2 reference, which is 32 pixels square."""
    feats = mx.array(qt.rule.features("wide")[:6])
    ref = mx.array(qt.rule.ref_latents("adjacent")[0])
    assert ref.shape == (8, 2, 2)
    cond = api.Conditioning(feats, "one", ref_latents=(ref,),
                            image_slots=(2,))
    api.denoise_step(pipe, qt.latents(), SIGMA, cond)
    state = pipe.captions.get(_state_key(pipe, "one"))
    cfg = pipe.dit.cfg
    row = memfit.PrefixRow(blocks=cfg.n_layers, dim=cfg.dim, token_px=16)
    assert state.tokens == row.rows(5, 1, 32) == 9
    assert pipe.captions.bytes == row.cache_bytes(9, 2) == 2 * 2 * 9 * 32 * 2
    # A step joins the K and V of one block over the prefix and the
    # target rows.
    t = qt.TARGET[0] * qt.TARGET[1]
    own = mx.zeros((1, cfg.n_heads, t, cfg.head_dim), mx.bfloat16)
    k, v = qm.join_prefix(state.k[0], state.v[0], own, own)
    assert k.nbytes + v.nbytes == row.join_bytes(9 + t, 2)


def test_the_plan_reads_the_prefix_rows_of_its_conditionings(pipe):
    """A conditioning with a reference plans more than the same request
    without it, by the prefix terms of the rows it adds."""
    row = memfit.PEAKS["qwen_image21"]
    feats = mx.array(qt.rule.features("wide")[:6])
    ref = mx.array(qt.rule.ref_latents("adjacent")[0])
    text = api.Conditioning(feats, "text")
    one = api.Conditioning(feats, "one", ref_latents=(ref,),
                           image_slots=(2,))
    kw = dict(height=64, width=64, budget=100e9, overcommit=False)
    image = 16
    plans = {}
    for cond, rows in ((text, 6), (one, 5 + 4)):
        p = plans[rows] = api.plan_request(pipe, conds=cond, **kw)
        cache, step, prefill = memfit.prefix_row_terms(row, image, rows, 2)
        assert p.terms["prefix_cache"] == cache
        assert p.terms["step_peak"] == max(row.step.bytes(step),
                                           row.prefill.bytes(prefill))
    assert (plans[9].terms["prefix_cache"] - plans[6].terms["prefix_cache"]
            == row.prefix.cache_bytes(3, 2) > 0)
    assert plans[9].phases["denoise"] > plans[6].phases["denoise"]
    # Without a conditioning the plan counts the row's text rows.
    assert (api.plan_request(pipe, **kw).terms["prefix_cache"]
            == row.prefix.cache_bytes(row.text_rows, 2))
    # A guided request counts the positive and the negative state.
    g = api.plan_request(pipe, conds=[one, text], cfg=4.0, **kw)
    assert g.terms["prefix_cache"] == (plans[9].terms["prefix_cache"]
                                       + plans[6].terms["prefix_cache"])


def test_the_plan_counts_a_held_prefix_state_once(pipe):
    cond = qt.conditioning("ref1", key="one")
    # The latent size of the request is that of the steps below.
    assert tuple(qt.TARGET) == (4, 6)
    kw = dict(height=64, width=96, budget=100e9, overcommit=False)
    cold = api.plan_request(pipe, conds=cond, **kw)
    assert cold.terms["caches"] == 0
    api.denoise_step(pipe, qt.latents(), SIGMA, cond)
    key = _state_key(pipe, "one")
    assert pipe.captions.nbytes(key) == pipe.captions.bytes > 0
    warm = api.plan_request(pipe, conds=cond, **kw)
    assert warm.terms["prefix_cache"] == cold.terms["prefix_cache"]
    assert warm.terms["caches"] == pipe.modulations.bytes > 0
    # The state of another conditioning stays beside the request's own.
    api.denoise_step(pipe, qt.latents(), SIGMA,
                     qt.conditioning("text", key="other"))
    other = pipe.captions.bytes - pipe.captions.nbytes(key)
    assert other > 0
    both = api.plan_request(pipe, conds=cond, **kw)
    assert both.terms["prefix_cache"] == cold.terms["prefix_cache"] + other
    assert both.terms["caches"] == pipe.modulations.bytes
    # A plan that names no conditioning counts every held state as other.
    bare = api.plan_request(pipe, **kw)
    row = memfit.PEAKS["qwen_image21"]
    assert bare.terms["prefix_cache"] == (
        row.prefix.cache_bytes(row.text_rows, 2) + pipe.captions.bytes)


# The state cache against the plan. A text state of 9 rows is 2304 bytes
# on the tiny model in bf16 and a state with one 8 x 8 reference 18432.
# The bound holds two text states and a little more.
TEXT_STATE, REF_STATE, BOUND = 2304, 18432, 2 * 2304 + 600


def _text(key: str, rows: int = 9) -> api.Conditioning:
    rng = np.random.default_rng(len(key) + rows)
    return api.Conditioning(mx.array(rng.standard_normal(
        (rows, qt.TINY["context_in_dim"])).astype(np.float32)), key)


def _with_ref(key: str) -> api.Conditioning:
    rng = np.random.default_rng(len(key))
    return api.Conditioning(
        mx.array(rng.standard_normal(
            (24, qt.TINY["context_in_dim"])).astype(np.float32)), key,
        ref_latents=(mx.array(rng.standard_normal((8, 8, 8)).astype(
            np.float32)),), image_slots=tuple(range(2, 18)))


def _watched(pipe, monkeypatch) -> dict:
    """Give the plan's prefix row the tiny model's sizes and the family
    the bound BOUND, and log the bytes of conditioning states alive at
    every prefill, the held ones and the new one, and at every forward."""
    row, cfg = memfit.PEAKS["qwen_image21"], pipe.dit.cfg
    monkeypatch.setitem(memfit.PEAKS, "qwen_image21", replace(
        row, prefix=replace(row.prefix, blocks=cfg.n_layers, dim=cfg.dim)))
    parts = pipe.family.step
    log = {"prefill": [], "forward": []}

    def conditioning(dit, cond):
        before = pipe.captions.bytes
        state = parts.conditioning(dit, cond)
        mx.eval(state.arrays())
        assert array_bytes(state.arrays()) == parts.state_size(dit, cond)
        log["prefill"].append(before + array_bytes(state.arrays()))
        return state

    def forward(dit, x, s, cond, state, mod):
        log["forward"].append(pipe.captions.bytes)
        return parts.forward(dit, x, s, cond, state, mod)

    pipe.family = replace(pipe.family, step=replace(
        parts, state_bytes=BOUND, conditioning=conditioning,
        forward=forward))
    return log


def _request(pipe, log: dict, conds: list, cfg: float, table=None) -> tuple:
    """Plan a request as the image command does, sample it, and hold the
    states alive at every prefill and forward to the plan's prefix term.
    Returns the term, the most alive at a prefill and at a forward, and
    the bytes held after."""
    log["prefill"].clear()
    log["forward"].clear()
    plan = api.plan_request(
        pipe, height=64, width=64, cfg=cfg, conds=conds, patch_table=table,
        budget=100e9, overcommit=False).terms["prefix_cache"]
    api.sample(pipe, conds[0], neg=conds[1] if len(conds) > 1 else None,
               cfg=cfg, height=64, width=64, steps=3, seed=1,
               patch_table=table)
    assert len(log["forward"]) == 3 * len(conds)
    assert max(log["prefill"] + log["forward"]) <= plan, (plan, log)
    return (plan, max(log["prefill"], default=0), max(log["forward"]),
            pipe.captions.bytes)


def test_the_state_cache_holds_what_the_plan_counts_after_references(
        pipe, monkeypatch):
    log = _watched(pipe, monkeypatch)
    rp, rn, t1 = _with_ref("rp"), _with_ref("rn"), _text("t1")
    assert _request(pipe, log, [rp, rn], 4.0) == (
        2 * REF_STATE, 2 * REF_STATE, 2 * REF_STATE, 2 * REF_STATE)
    # An unguided text request: both reference states go before its
    # state is built, since neither fits the room beside it.
    assert _request(pipe, log, [t1], 1.0) == (
        BOUND, TEXT_STATE, TEXT_STATE, TEXT_STATE)
    # The same text again finds its state held.
    assert _request(pipe, log, [t1], 1.0) == (
        TEXT_STATE, 0, TEXT_STATE, TEXT_STATE)


def test_the_state_cache_holds_what_the_plan_counts_on_a_hit(
        pipe, monkeypatch):
    """The positive of a guided request with references alone again: its
    state is held, and the negative's goes at the first step."""
    log = _watched(pipe, monkeypatch)
    rp, rn = _with_ref("rp"), _with_ref("rn")
    _request(pipe, log, [rp, rn], 4.0)
    assert _request(pipe, log, [rp], 1.0) == (
        REF_STATE, 0, REF_STATE, REF_STATE)
    key = _state_key(pipe, "rp")
    assert pipe.captions.nbytes(key) == REF_STATE


def test_the_state_cache_holds_what_the_plan_counts_text_after_text(
        pipe, monkeypatch):
    log = _watched(pipe, monkeypatch)
    assert _request(pipe, log, [_text("p1"), _text("n1")], 4.0) == (
        2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE)
    # The room beside two new text states holds neither older one.
    assert _request(pipe, log, [_text("p2"), _text("n2")], 4.0) == (
        BOUND, 2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE)
    # The room beside one holds one older state, the last one used.
    assert _request(pipe, log, [_text("p3")], 1.0) == (
        BOUND, 2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE)
    assert [k[0] for k in pipe.captions._items] == ["n2", "p3"]
    # A guided text request after a request with references.
    _request(pipe, log, [_with_ref("rp"), _with_ref("rn")], 4.0)
    assert _request(pipe, log, [_text("p"), _text("n")], 4.0) == (
        BOUND, 2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE)


def test_the_plan_reads_the_held_states_under_the_table_of_the_request(
        pipe, monkeypatch):
    from gmlx.image.lora import install

    log = _watched(pipe, monkeypatch)
    rng = np.random.default_rng(5)
    dim = pipe.dit.cfg.dim
    pair = install.register_pair(
        pipe, mx.array(rng.standard_normal((2, dim)).astype(np.float32)),
        mx.array(rng.standard_normal((dim, 2)).astype(np.float32)), None)
    table = api.define_patch_table(
        pipe, [("transformer_blocks.0.attn.to_q", None, [(pair, 0.1)])])
    cond = _text("one")
    kw = dict(height=64, width=64, cfg=1.0, conds=[cond], budget=100e9,
              overcommit=False)
    cold = api.plan_request(pipe, patch_table=table, **kw)
    assert cold.terms["prefix_cache"] == TEXT_STATE
    assert _request(pipe, log, [cond], 1.0, table) == (
        TEXT_STATE, TEXT_STATE, TEXT_STATE, TEXT_STATE)
    # Held under the request's table, the state counts once.
    warm = api.plan_request(pipe, patch_table=table, **kw)
    assert warm.terms["prefix_cache"] == TEXT_STATE
    # The same conditioning under the default table is another state,
    # and the held one is another request's.
    assert api.plan_request(pipe, **kw).terms["prefix_cache"] == (
        2 * TEXT_STATE)
    assert _request(pipe, log, [cond], 1.0) == (
        2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE, 2 * TEXT_STATE)
    assert {k[4] for k in pipe.captions._items} == {table,
                                                    pipe.default_table}
    # The scope holds the states of the request's table: one state over
    # the bound is built once and stays through the request.
    assert _request(pipe, log, [_with_ref("r")], 1.0, table) == (
        REF_STATE, REF_STATE, REF_STATE, REF_STATE)
    assert len(log["prefill"]) == 1
    with pytest.raises(ValueError, match="unknown patch table"):
        api.plan_request(pipe, patch_table="none", **kw)


def _steps(pipe, log: dict, cond, plan: int, *, table=None,
           size=(4, 4)) -> tuple:
    """Three bare steps of ``cond`` inside the open scope, with the
    states alive at every prefill and forward held to ``plan``. Returns
    the prefills and the most alive."""
    log["prefill"].clear()
    log["forward"].clear()
    for s in (0.9, 0.6, 0.3):
        api.denoise_step(pipe, qt.latents(size=size), s, cond,
                         patch_table=table)
    alive = max(log["prefill"] + log["forward"])
    assert alive <= plan, (plan, log)
    return len(log["prefill"]), alive


def test_a_scope_holds_the_state_of_the_table_its_steps_run_under(
        pipe, monkeypatch):
    """A scope opened with the state of the default table held, and
    steps under another table inside it: the state of the default table
    goes before the new one is built, and the new one stays, one prefill
    for three steps. The plan is that of the request under the table."""
    from gmlx.image.lora import install

    log = _watched(pipe, monkeypatch)
    rng = np.random.default_rng(5)
    dim = pipe.dit.cfg.dim
    pair = install.register_pair(
        pipe, mx.array(rng.standard_normal((2, dim)).astype(np.float32)),
        mx.array(rng.standard_normal((dim, 2)).astype(np.float32)), None)
    table = api.define_patch_table(
        pipe, [("transformer_blocks.0.attn.to_q", None, [(pair, 0.1)])])
    r = _with_ref("r")
    _request(pipe, log, [r], 1.0)
    plan = api.plan_request(
        pipe, height=64, width=64, cfg=1.0, conds=[r], patch_table=table,
        budget=100e9, overcommit=False).terms["prefix_cache"]
    assert plan == REF_STATE
    with api.state_scope(pipe, [r]):
        assert _steps(pipe, log, r, plan, table=table) == (1, REF_STATE)
        assert [k[4] for k in pipe.captions._items] == [table]
        # Back under the default table, and at another target size.
        assert _steps(pipe, log, r, plan) == (1, REF_STATE)
        assert _steps(pipe, log, r, plan, size=(4, 6)) == (1, REF_STATE)
        assert [(k[3], k[4]) for k in pipe.captions._items] == [
            ((4, 6), pipe.default_table)]
        # Steps that alternate two tables pay a prefill each under a
        # bound with no room for both, and stay within the plan.
        log["prefill"].clear()
        for i in range(4):
            api.denoise_step(pipe, qt.latents(), 0.5, r,
                             patch_table=table if i % 2 else None)
            assert pipe.captions.bytes == REF_STATE
        assert log["prefill"] == [REF_STATE] * 3
    assert pipe.scope is None and pipe.captions.bytes == REF_STATE


def test_a_scope_holds_the_state_of_the_routes_its_steps_run_under(
        pipe, monkeypatch):
    log = _watched(pipe, monkeypatch)
    r = _with_ref("r")
    _request(pipe, log, [r], 1.0)
    held = set(pipe.captions._items)
    with api.state_scope(pipe, [r]):
        with kq_routes.only(()):
            plan = api.plan_request(
                pipe, height=64, width=64, cfg=1.0, conds=[r],
                budget=100e9, overcommit=False).terms["prefix_cache"]
            assert plan == REF_STATE
            assert _steps(pipe, log, r, plan) == (
                1 if kq_routes.active() != next(iter(held))[-1] else 0,
                REF_STATE)
    assert pipe.captions.bytes == REF_STATE


def test_a_scope_inside_a_scope_of_other_conditionings_is_refused(
        pipe, monkeypatch):
    """An open scope never sheds its states, so a request with other
    conditionings inside it would hold them beside its own, over its
    plan. It is refused by name, with the states built or not, and runs
    within its plan once the scope is closed."""
    log = _watched(pipe, monkeypatch)
    rp, rn, t1 = _with_ref("rp"), _with_ref("rn"), _text("t1")
    refused = "a state scope is open for other conditionings"
    with api.state_scope(pipe, [rp, rn]):
        with pytest.raises(api.StateScopeError, match=refused):
            api.sample(pipe, t1, height=64, width=64, steps=3, seed=1)
    assert pipe.scope is None and len(pipe.captions) == 0
    _request(pipe, log, [rp, rn], 4.0)
    with api.state_scope(pipe, [rp, rn]):
        with pytest.raises(api.StateScopeError, match=refused):
            api.sample(pipe, t1, height=64, width=64, steps=3, seed=1)
        with pytest.raises(api.StateScopeError, match=refused):
            api.denoise_step(pipe, qt.latents(), 0.5, t1)
        # A conditioning of the open scope runs.
        api.denoise_step(pipe, qt.latents(size=(4, 4)), 0.5, rn)
        assert pipe.captions.bytes == 2 * REF_STATE
    assert pipe.scope is None
    assert _request(pipe, log, [t1], 1.0) == (
        BOUND, TEXT_STATE, TEXT_STATE, TEXT_STATE)


def test_one_conditioning_as_both_rows_of_a_guided_request_is_one_state(
        pipe, monkeypatch):
    log = _watched(pipe, monkeypatch)
    t1 = _text("t1")
    assert _request(pipe, log, [t1, t1], 4.0) == (
        TEXT_STATE, TEXT_STATE, TEXT_STATE, TEXT_STATE)
    assert len(log["prefill"]) == 1
    # Equal in key, references and slots is one state too, whatever the
    # objects.
    again = api.Conditioning(t1.features, "t1")
    assert _request(pipe, log, [t1, again], 4.0) == (
        TEXT_STATE, 0, TEXT_STATE, TEXT_STATE)


def test_rows_of_a_batch_take_their_own_conditioning(pipe):
    conds = [qt.conditioning("ref1", key="pos"),
             qt.conditioning("text", seed=1, key="neg")]
    x = qt.latents(batch=2)
    sig = [SIGMA, 0.25]
    both = api.denoise_step(pipe, x, sig, conds)
    assert both.shape == (2, 8, *qt.TARGET) and both.dtype == mx.float32
    for i in range(2):
        one = api.denoise_step(pipe, x[i:i + 1], sig[i], conds[i])
        assert np.array_equal(np.array(both[i:i + 1]), np.array(one))


def test_sample_runs_the_family_schedule_at_the_request_size(pipe):
    cond = qt.conditioning("ref1", key="pos")
    x = api.sample(pipe, cond, height=64, width=96, seed=1, steps=6)
    assert x.shape == (1, 8, 4, 6) and x.dtype == mx.float32
    assert bool(mx.isfinite(x).all())
    # 64x96 is 4 x 6 latent tokens, and the schedule shifts by that count.
    s = pipe.sigmas(6, 24)
    assert np.array_equal(s, qwen21_sigmas(6, 24))
    assert not np.array_equal(s, qwen21_sigmas(6, 4096))
    same = api.sample(pipe, cond, height=64, width=96, seed=1,
                      sigmas=[float(v) for v in s])
    assert np.array_equal(np.array(same), np.array(x))
    assert len(pipe.modulations) == 6 and len(pipe.captions) == 1
    with pytest.raises(ValueError, match="multiples of 32, not 48x96"):
        api.sample(pipe, cond, height=48, width=96, seed=1, steps=2)
    with pytest.raises(ValueError, match="needs at least 2 steps"):
        api.sample(pipe, cond, height=64, width=96, seed=1, steps=1)


def test_the_schedule_is_refused_without_the_token_count(pipe):
    with pytest.raises(ValueError, match="shifts by the image token count"):
        pipe.sigmas(4)
    with pytest.raises(ValueError, match="shifts by the image token count"):
        pipe.family.schedule(4, tokens=None)
    assert np.array_equal(pipe.sigmas(4, 16), qwen21_sigmas(4, 16))


def test_a_conditioning_without_tokens_is_refused(tiny):
    empty = api.Conditioning(mx.zeros((0, tiny.cfg.context_dim)), "")
    with pytest.raises(ValueError, match="the conditioning has no tokens"):
        qm.prefill_prefix(tiny, empty)


def test_sample_under_guidance_holds_a_prefix_cache_per_conditioning(pipe):
    cond = qt.conditioning("ref1", key="pos")
    neg = qt.conditioning("text", seed=1, key="neg")
    seen = []
    x = api.sample(pipe, cond, neg=neg, cfg=4.0, height=64, width=64,
                   seed=2, steps=3, on_step=lambda i, x: seen.append(i))
    assert x.shape == (1, 8, 4, 4) and seen == [0, 1, 2]
    assert len(pipe.captions) == 2
    plain = api.sample(pipe, cond, height=64, width=64, seed=2, steps=3)
    assert not np.array_equal(np.array(plain), np.array(x))
