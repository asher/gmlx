"""Check 3 of the image acceptance gate for Z-Image: one denoise step of the
gmlx DiT, block by block, against the diffusers oracle.

block_gates.py describes the rows and the three gates. The exact gate
here runs the base model in the oracle's fp32 arithmetic and
holds every teacher-forced row, every branch row, every free-running block
and the velocity to the oracle's fp32 arrays, at 512. At 1024 the fixture
holds the velocity alone, and the exact gate reads that. Each wrong model
of FAULTS must fail the exact gate, and only on rows its fault reaches.
The product gate runs the path a request takes and holds the block rows,
the free-running blocks and the velocity to the manifest's tolerances.
The site gate holds each op of that path to the exact op on its own
input, at 512 and at 1024, and each wrong production model of the site
faults must fail it at the site of its fault.
The arrays are read from GMLX_IMAGE_FIXTURES and must match the manifests
committed under tests/fixtures/image/oracle/.
scripts/image_oracle/check3_z_image.py writes them.

The tiny row runs on the CPU and needs only the self-test fixture. It
checks the gmlx forward's conventions against diffusers' own code in fp32.
The real rows need the Z-Image-Turbo GGUF under KQUANT_TEST_GGUF_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from block_gates import (EXACT_TOL, GROUPS, LEAVES, Sites, assert_all_dense,
                         check3_rows, exact_missed, exact_process,
                         exact_products, free_run, free_running, gate,
                         plain_velocity, rerun_exact, rerun_sites, resolve,
                         right_sites, site_process, thin, velocity)
from oracle_fixture import committed, fixture, fx


# The tiny model, against diffusers' code on the same weights.

def tiny_model(weights: dict, cfg: dict):
    from gmlx.image.families.z_image.model import ZImageDiT

    d = cfg["dim"]
    model = ZImageDiT({**cfg, "head_dim": d // cfg["n_heads"],
                       "ffn_dim": int(d / 3 * 8), "adaln_dim": min(d, 256),
                       "t_freq_dim": 256, "t_mid_dim": 1024,
                       "final_norm_eps": 1e-6, "seq_multiple": 32})
    model.load_weights([(k, mx.array(v)) for k, v in weights.items()])
    model.act_dtype = mx.float32
    model.eval()
    return model


def test_tiny_model_matches_the_diffusers_oracle_in_fp32(monkeypatch):
    weights, wm = fixture("selftest_z_image_tiny_weights")
    arrays, m = fixture("selftest_z_image_tiny_48")
    model = tiny_model(weights, wm["inputs"]["config"])
    # The CPU stream keeps fp32 products exact; the GPU's TF32 would not.
    with mx.stream(mx.cpu):
        t = model.timesteps(float(arrays["sigma"]), 1)
        branches: dict = {}
        rows = check3_rows(model, arrays, m["inputs"]["rows"], branches)
        # torch's fp32 exp misrounds some sinusoid frequencies by one ulp,
        # and t up to 1000 turns that into about 2e-6 in the t_embedder
        # row, which the random tiny blocks amplify to 1e-5. The free run
        # takes the oracle's embedding; the t_embedder row checks gmlx's.
        ad = mx.array(arrays["t_embedder.out"])
        monkeypatch.setattr(model, "adaln", lambda sigma, b: ad)
        free, vel = free_run(model, arrays, m["inputs"]["free_rows"])
    assert np.array_equal(t, arrays["t_embedder.in"])
    assert len(free) == len(model.layers) + 2 * len(model.noise_refiner) + 1
    # Measured in fp32 on the CPU: rows up to 3.1e-7, branches 4.0e-7, free
    # blocks 4.3e-7, velocity 2.8e-7. The t_embedder row carries torch's
    # sinusoid error, 2.1e-6 at this timestep, and has its own bound.
    assert sorted(branches) == sorted(m["inputs"]["branches"])
    for r, g in rows.items():
        assert g["rel_l2"] < (3e-6 if r == "t_embedder" else 2e-6), (r, g)
    for measured in (branches, free):
        worst = max(measured.items(), key=lambda kv: kv[1]["rel_l2"])
        assert worst[1]["rel_l2"] < 2e-6, worst
    assert vel["rel_l2"] < 2e-6, vel


# Z-Image-Turbo Q4_K_M, the S1 exit.

@pytest.fixture(scope="module")
def turbo(gguf_index):
    m = committed("check3_z_image_turbo_512")
    want = m["files"]["dit"]
    for p in gguf_index.get("lumina2", []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    from gmlx.image.loader import load_dit

    model, _ = load_dit(p, log=lambda *_: None)
    return model


@pytest.mark.integration
def test_check3_one_step_block_by_block_at_512(turbo):
    """The product gate."""
    arrays, m = fixture("check3_z_image_turbo_512")
    t = turbo.timesteps(float(arrays["sigma"]), 1)
    assert np.array_equal(t, arrays["t_embedder.in"])
    rows = check3_rows(turbo, arrays, m["inputs"]["rows"])
    free, vel = free_run(turbo, arrays, m["inputs"]["free_rows"])
    print(f"check3 product 512: above 0.8 of its gate: "
          f"{thin(m, rows, vel, free) or 'none'}")
    misses = gate(m, rows, vel, free)
    assert not misses, misses


@pytest.mark.integration
def test_check3_final_velocity_at_1024(turbo):
    arrays, m = fixture("check3_z_image_turbo_1024")
    vel = velocity(turbo, arrays)
    print(f"check3 product 1024: above 0.8 of its gate: "
          f"{thin(m, None, vel) or 'none'}")
    misses = gate(m, None, vel)
    assert not misses, misses


# The exact gate.

EMBEDS = ("t_embedder", "x_embed", "cap_embed")


def exact_step(model, arrays: dict, m: dict, rows=None, free: bool = True
               ) -> dict:
    """Name to measure of the step in the oracle's arithmetic: each
    teacher-forced row of ``rows`` with the branch rows of its block and,
    with ``free``, each free-running block as ``free <name>`` and the
    velocity. Without ``rows`` it is every row, and every K-quant product
    must then have run dense."""
    names = m["inputs"]["rows"] if rows is None else rows
    with exact_products(model) as used:
        branches: dict = {}
        got = dict(check3_rows(model, arrays, names, branches))
        got.update(branches)
        if rows is None:
            assert sorted(branches) == sorted(m["inputs"]["branches"])
            assert_all_dense(model, used)
        if free:
            blocks, vel = free_run(model, arrays, m["inputs"]["free_rows"],
                                   stepwise=True)
            got.update({f"free {b}": g for b, g in blocks.items()})
            got["velocity"] = vel
    return got


def _blocks(model, groups=GROUPS) -> list:
    return [blk for g in groups for blk in getattr(model, g)]


def _modulated(row: str) -> bool:
    return row.startswith(("noise_refiner.", "layers."))


def _in_block(row: str) -> bool:
    return row.startswith(GROUPS)


def norm_eps_tenfold(mp, model, arrays):
    """Every RMSNorm of every block, and its q and k norm, at ten times
    the eps."""
    for blk in _blocks(model):
        for mod in (blk.attention, blk.attention_norm1, blk.attention_norm2,
                    blk.ffn_norm1, blk.ffn_norm2):
            mp.setattr(mod, "eps", mod.eps * 10)


def qk_norm_weights_swapped(mp, model, arrays):
    for blk in _blocks(model):
        q, k = blk.attention.q_norm, blk.attention.k_norm
        qw, kw = q.weight, k.weight
        mp.setattr(q, "weight", kw)
        mp.setattr(k, "weight", qw)


def rope_one_token_late(mp, model, arrays):
    """The image rows' rotary tables shifted by one token."""
    real = model.rope

    def rope(ht, wt, n_cap):
        (xc, xs), cap = real(ht, wt, n_cap)
        return (mx.roll(xc, 1, axis=0), mx.roll(xs, 1, axis=0)), cap

    mp.setattr(model, "rope", rope)


def modulation_scales_dropped(mp, model, arrays):
    """Every modulated block keeps its gates and loses its two scales."""
    for blk in _blocks(model, ("noise_refiner", "layers")):
        def modulation(adaln, real=blk.modulation):
            s_msa, g_msa, s_mlp, g_mlp = real(adaln)
            return mx.zeros_like(s_msa), g_msa, mx.zeros_like(s_mlp), g_mlp

        mp.setattr(blk, "modulation", modulation)


def gelu_for_silu(mp, model, arrays):
    from gmlx.image import ops

    mp.setattr(ops, "swiglu", lambda a, b: nn.gelu(a) * b)


def patch_axes_transposed(mp, model, arrays):
    """The two pixel axes of a patch in the other order."""
    def patchify(latents):
        b, c, h, w = latents.shape
        p = model.cfg.patch
        x = latents.reshape(b, c, h // p, p, w // p, p)
        return x.transpose(0, 2, 4, 5, 3, 1).reshape(
            b, (h // p) * (w // p), p * p * c)

    mp.setattr(model, "patchify", patchify)


def timestep_not_reversed(mp, model, arrays):
    """t = sigma, where the model takes 1 - sigma."""
    def timesteps(sigma, batch):
        s = np.broadcast_to(np.asarray(sigma, np.float32), (batch,))
        return s * np.float32(model.cfg.t_scale)

    mp.setattr(model, "timesteps", timesteps)


def caption_rows_first(mp, model, arrays):
    """The main layers take [caption, image] rows under the tables of
    [image, caption]."""
    block = type(model.layers[0])
    real, first = block.__call__, model.layers[0]
    n_img = arrays["noise_refiner.0.in"].shape[1]

    def call(self, x, *args, **kwargs):
        if self is first:
            x = mx.concatenate([x[:, n_img:], x[:, :n_img]], axis=1)
        return real(self, x, *args, **kwargs)

    mp.setattr(block, "__call__", call)


def velocity_sign(mp, model, arrays):
    real = model.unpatchify
    mp.setattr(model, "unpatchify", lambda x, ht, wt: -real(x, ht, wt))


# A wrong model is seen when it moves a measure this many times the gate.
# The weakest one, the norm eps, moves its most moved row 53 times it.
SEEN = 2.0

# name: (the wrong model, the teacher-forced rows to run or None for all,
# whether the free run is needed to see it, the rows it can reach).
FAULTS = {
    "norm eps x10": (norm_eps_tenfold, None, False, _in_block),
    "q and k norm weights swapped": (
        qk_norm_weights_swapped, None, False, _in_block),
    "rope one token late": (rope_one_token_late, None, False, _modulated),
    "modulation scales dropped": (
        modulation_scales_dropped, None, False, _modulated),
    "gelu for silu": (
        gelu_for_silu, None, False,
        lambda r: _in_block(r) and not r.endswith(".attention")),
    "patch axes transposed": (
        patch_axes_transposed, ["x_embed"], False, lambda r: r == "x_embed"),
    "timestep not reversed": (
        timestep_not_reversed, ["t_embedder"], False,
        lambda r: r == "t_embedder"),
    "caption rows first": (
        caption_rows_first, [], True,
        lambda r: r == "velocity" or r.startswith("free layers.")
        or r == "free final_layer"),
    "velocity sign": (velocity_sign, [], True, lambda r: r == "velocity"),
}


def fault_misses(monkeypatch, model, arrays: dict, m: dict, name: str
                 ) -> dict:
    """The measures the wrong model ``name`` misses under the exact gate.
    It must miss one by SEEN times the gate at least, and none outside the
    rows its fault reaches."""
    apply, rows, free, reach = FAULTS[name]
    with monkeypatch.context() as mp:
        apply(mp, model, arrays)
        missed = exact_missed(exact_step(model, arrays, m, rows, free))
    assert missed and max(missed.values()) >= SEEN * EXACT_TOL, (name, missed)
    stray = sorted(r for r in missed if not reach(r))
    assert not stray, (name, stray)
    return missed


@pytest.mark.integration
def test_check3_exact_gate_at_512(turbo, request, monkeypatch):
    """The base model matches the oracle on every row in the oracle's
    arithmetic, and each wrong model of FAULTS fails."""
    arrays, m = fixture("check3_z_image_turbo_512")
    if not exact_process():
        return rerun_exact(request, "check3 exact")
    got = exact_step(turbo, arrays, m)
    assert len(got) == (len(m["inputs"]["rows"]) + len(m["inputs"]["branches"])
                        + len(m["inputs"]["free_rows"]) + 1)
    forced = {r: g["rel_l2"] for r, g in got.items() if not free_running(r)}
    free = {r: g["rel_l2"] for r, g in got.items() if r not in forced}
    top, ftop = max(forced, key=forced.get), max(free, key=free.get)
    print(f"check3 exact 512: within {forced[top]:.2e} of the oracle on "
          f"{len(forced)} teacher-forced rows, at {top}, and within "
          f"{free[ftop]:.2e} on {len(free)} free-running measures, at "
          f"{ftop}, gate {EXACT_TOL:.0e}")
    assert not exact_missed(got), exact_missed(got)
    assert len(FAULTS) >= 6
    for name in FAULTS:
        missed = fault_misses(monkeypatch, turbo, arrays, m, name)
        low, high = min(missed, key=missed.get), max(missed, key=missed.get)
        print(f"check3 exact 512, {name}: {len(missed)} measures miss, "
              f"from {low} at {missed[low]:.2e} to {high} at "
              f"{missed[high]:.2e}")
    # The model is the right one again.
    again = exact_step(turbo, arrays, m, ["x_embed", "t_embedder",
                                          "layers.0"], free=False)
    assert not exact_missed(again), exact_missed(again)


@pytest.mark.integration
def test_check3_exact_velocity_at_1024(turbo, request):
    """The fixture at 1024 holds no block arrays, so the exact gate there
    is the free-running velocity."""
    arrays, m = fixture("check3_z_image_turbo_1024")
    if not exact_process():
        return rerun_exact(request, "check3 exact")
    with exact_products(turbo):
        vel = free_run(turbo, arrays, stepwise=True)[1]
    print(f"check3 exact 1024: velocity within {vel['rel_l2']:.2e} of the "
          f"oracle, gate {EXACT_TOL:.0e}")
    assert vel["rel_l2"] <= EXACT_TOL, vel


# The site gate.

# A K-quant codec to one that reads its rows as the same number of
# weights. A q6_k row has no such codec.
MISREAD = {"q4_k": "q4_0", "q5_k": "q5_0"}


def gate_and_up_swapped(mp, model):
    for blk in _blocks(model):
        ff = blk.feed_forward
        w1, w3 = ff.w1, ff.w3
        mp.setattr(ff, "w1", w3)
        mp.setattr(ff, "w3", w1)


def feed_forward_gelu(mp, model):
    """gelu for silu in the feed-forward's own call, so the fault is on
    the production path whatever kernel the right call runs."""
    mp.setattr(type(model.layers[0].feed_forward), "__call__",
               lambda self, x: self.w2(nn.gelu(self.w1(x)) * self.w3(x)))


def _base(model, path: str):
    leaf = resolve(model, path)
    return getattr(leaf, "base", leaf)


def codec_leaves(model) -> dict:
    """K-quant codec to (block, leaf) of one main layer leaf in it, the
    middle one of the leaf name that has the most."""
    found: dict = {}
    for i in range(len(model.layers)):
        for leaf, path in LEAVES.items():
            codec = getattr(_base(model, f"layers.{i}.{path}"),
                            "kquant_type", None)
            if codec is not None:
                found.setdefault(codec, {}).setdefault(leaf, []).append(
                    f"layers.{i}")
    out = {}
    for codec, leaves in found.items():
        leaf = max(leaves, key=lambda k: len(leaves[k]))
        out[codec] = (leaves[leaf][len(leaves[leaf]) // 2], leaf)
    return out


def faulted_sites(monkeypatch, model, arrays: dict, apply) -> Sites:
    """The sites of the step with the wrong production model of
    ``apply``, against the right model's exact ops."""
    sites = Sites(model)
    with monkeypatch.context() as mp:
        apply(mp, model)
        sites.run(arrays)
    return sites


def _span(missed: dict) -> str:
    fin = [v for v in missed.values() if np.isfinite(v)]
    return (f"{len(missed)} sites miss, {len(missed) - len(fin)} not finite"
            + (f", from {min(fin):.1f} to {max(fin):.1f} gates" if fin else ""))


@pytest.mark.integration
def test_check3_site_gate_at_512(turbo, request, monkeypatch):
    """Every op of every block is within its gate of the exact op on its
    own input, and each wrong production model fails at its fault."""
    arrays, _ = fixture("check3_z_image_turbo_512")
    if not site_process():
        return rerun_sites(request, "check3 sites",
                           plain_velocity(turbo, arrays))
    sites = Sites(turbo)
    v = sites.run(arrays)
    right_sites(sites, v, "check3 sites 512")
    glu = {n for n in sites.sites if n.endswith(".glu")}
    assert len(glu) == len(_blocks(turbo))

    # silu lands on the up product, so the glu output of every block is
    # another function of its input. Each product and swiglu is still
    # right on its own input, and only the glu site reads the pair.
    missed = faulted_sites(monkeypatch, turbo, arrays,
                           gate_and_up_swapped).missed()
    print(f"check3 sites 512, gate and up swapped: {_span(missed)}")
    assert glu <= missed.keys(), sorted(glu - missed.keys())

    # gelu and silu differ by a few percent of the gate product where the
    # gate is near zero, and the glu site reads that against silu.
    missed = faulted_sites(monkeypatch, turbo, arrays,
                           feed_forward_gelu).missed()
    print(f"check3 sites 512, gelu for silu: {_span(missed)}")
    assert missed and missed.keys() <= glu, sorted(missed.keys() - glu)

    # A codec read as another of the same row size decodes other weights,
    # so the leaf's product is wrong on its own input. The sites before
    # the leaf pass, and the ones after it read the wrong activations.
    codecs = codec_leaves(turbo)
    assert sorted(codecs) == ["q4_k", "q5_k", "q6_k"], codecs
    for codec, other in MISREAD.items():
        block, leaf = codecs[codec]
        got = faulted_sites(
            monkeypatch, turbo, arrays, lambda mp, model: mp.setattr(
                _base(model, f"{block}.{LEAVES[leaf]}"), "kquant_type",
                other))
        missed, order = got.missed(), list(got.sites)
        print(f"check3 sites 512, {block}.{leaf} {codec} read as {other}: "
              f"{_span(missed)}")
        at = order.index(f"{block}.{leaf}")
        assert f"{block}.{leaf}" in missed
        assert all(order.index(n) >= at for n in missed), min(
            missed, key=order.index)

    # A q6_k row is 210 bytes for 256 weights and no other codec has that
    # ratio, so the product refuses the row before any kernel runs.
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    block, leaf = codecs["q6_k"]
    base = _base(turbo, f"{block}.{LEAVES[leaf]}")
    x = mx.zeros((1, 2, base.weight.shape[1] // 210 * 256), turbo.act_dtype)
    others = sorted(set(CODEC_GEOMETRY) - {"q6_k"})
    for other in others:
        with monkeypatch.context() as mp:
            mp.setattr(base, "kquant_type", other)
            with pytest.raises(ValueError):
                base(x)
    print(f"check3 sites 512, {block}.{leaf} q6_k read as any of "
          f"{len(others)} codecs: the product refuses the row")

    # The model is the right one again.
    assert Sites(turbo).run(arrays) == v


@pytest.mark.integration
def test_check3_site_gate_at_1024(turbo, request):
    arrays, _ = fixture("check3_z_image_turbo_1024")
    if not site_process():
        return rerun_sites(request, "check3 sites",
                           plain_velocity(turbo, arrays))
    sites = Sites(turbo)
    right_sites(sites, sites.run(arrays), "check3 sites 1024")
