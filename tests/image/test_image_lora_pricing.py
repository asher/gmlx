"""The dry run of the LoRA factors a table builds (install._factor_bytes,
install.table_figures) against the arrays the build makes, form by form,
at the dtype each leaf runs at."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx.image.conditioning import ByteLRU
from gmlx.image.lora import install, plan

kq = pytest.importorskip("mlx_kquant")
from mlx_kquant.nn import KQuantLinear  # noqa: E402


def kq_leaf(out: int, inp: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = mx.array((rng.standard_normal((out, inp)) / 16).astype(np.float32))
    leaf = KQuantLinear(inp, out, False, "q4_k")
    leaf.weight, leaf.scales = kq.quantize(w, "q4_k")
    mx.eval(leaf.weight, leaf.scales)
    return leaf


def float_leaf(out: int, inp: int):
    leaf = nn.Linear(inp, out, bias=False)
    mx.eval(leaf.parameters())
    return leaf


def parts(rng, dims, bands, rank, strength=1.0, dtype=mx.bfloat16):
    """One part per band (start, rows) on the output, or one whole-weight
    part for a band of None."""
    out, inp = dims
    got = []
    for band in bands:
        rows = out if band is None else band[1]
        down = mx.array(rng.standard_normal((rank, inp)).astype(np.float32))
        up = mx.array(rng.standard_normal((rows, rank)).astype(np.float32))
        got.append(plan.Part(down.astype(dtype), up.astype(dtype), 8.0,
                             None if band is None else (0, band[0], band[1]),
                             strength))
    return got


def built(path, ps, leaf, dtype, how):
    """(table bytes, kernel pair bytes) of the module once it ran, its
    stock path too."""
    dims = install.linear_dims(leaf)
    pairs = []

    def pair():
        pairs.append(install.kernel_pair(path, ps, dims, dtype))
        mx.eval(pairs[-1][0], pairs[-1][1])
        return pairs[-1]
    f = install._factors(path, ps, dims, dtype, how, leaf,
                         pair if how == install.KERNEL else None)
    mx.eval(f.arrays())
    if isinstance(f, install._Kernel):
        f.stock()
    kernel = sum(int(p[0].nbytes) + int(p[1].nbytes) for p in pairs)
    return sum(int(a.nbytes) for a in f.arrays()), kernel, type(f).__name__


def priced(ps, leaf, dtype, how):
    return install._factor_bytes(ps, install.linear_dims(leaf), dtype, how,
                                 leaf, how == install.KERNEL)


# Each form, as the reviewer's probe shapes it: the kernel at 3 and 4
# bands, FLUX.2's partial linear1_qkv band of a 3d + 2 mlp_dim leaf, the
# stock path at 0.75, block, slice, a lone pair at 0, and split with one
# band of 256 rows of a 768-row leaf at the whole weight.
D = 1024
CASES = {
    "kernel, 3 bands": ((12288, D), [(0, 4096), (4096, 4096), (8192, 4096)],
                        1.0, install.KERNEL, "_Kernel"),
    "kernel, 4 bands": ((12288, D), [(i * 3072, 3072) for i in range(4)],
                        1.0, install.KERNEL, "_Kernel"),
    "flux2 linear1_qkv": ((3 * D + 2 * 3 * D, D), [(0, 3 * D)], 1.0,
                          install.KERNEL, "_Kernel"),
    "stock path at 0.75": ((12288, D), [(0, 4096), (4096, 4096),
                                        (8192, 4096)], 0.75,
                           install.KERNEL, "_Kernel"),
    "block": ((768, 256), [(0, 256), (512, 256)], 0.75, "block", "_Block"),
    "slice": ((768, 256), [(0, 256), (512, 256)], 0.75, "slice", "_Slice"),
    "split": ((768, 256), [(0, 256)], 1.0, "split", "_Split"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_dry_run_equals_the_built_bytes_of_each_form(name):
    dims, bands, strength, how, form = CASES[name]
    rng = np.random.default_rng(1)
    leaf = kq_leaf(*dims)
    ps = parts(rng, dims, bands, 8, strength)
    table, kernel, got = built("m", ps, leaf, mx.bfloat16, how)
    assert got == form
    assert priced(ps, leaf, mx.bfloat16, how) == (table, kernel)
    if name == "split":
        # The whole weight is copied, the rows past the band too.
        assert table > int(leaf.weight.nbytes)


def test_a_lone_pair_runs_on_the_cached_pair_and_builds_nothing():
    rng = np.random.default_rng(2)
    leaf = float_leaf(96, 64)
    ps = parts(rng, (96, 64), [None], 4, 0.7, mx.float32)
    for how in ("block", "slice", "split", install.KERNEL):
        assert built("m", ps, leaf, mx.float32, how)[:3] == (0, 0, "_Own")
        assert priced(ps, leaf, mx.float32, how) == (0, 0)


class _Pipe(SimpleNamespace):
    """A pipeline's LoRA parts, which a built table holds weakly."""


def tiny_pipe(leaves: dict):
    """A pipeline of one DiT with linear ``leaves`` by path."""
    dit = nn.Module()
    for path, leaf in leaves.items():
        node = dit
        *head, last = path.split(".")
        for name in head:
            if not hasattr(node, name):
                setattr(node, name, nn.Module())
            node = getattr(node, name)
        setattr(node, last, leaf)
    return _Pipe(dit=dit, adapters=ByteLRU(1 << 40),
                           tables=ByteLRU(1 << 40), patch_tables={},
                           lora=None)


def table_of(pipe, rows) -> str:
    """A patch table of (path, band, pair, strength) rows."""
    entries = []
    for path, band, pr, strength in rows:
        key = install.register_pair(pipe, pr[0], pr[1])
        entries.append((path, band, [(key, strength, 8.0)]))
    pipe.patch_tables["t"] = {"entries": entries}
    return "t"


def test_a_table_prices_each_leaf_at_the_dtype_it_runs_at():
    """A modulation leaf of two parts at bfloat16 activations runs on
    float32 rows: priced as its float32 block, beside the bfloat16 table
    of the other leaves. The dry run names each tables-cache key the build
    holds, at its bytes."""
    rng = np.random.default_rng(3)
    mod = "double_stream_modulation_img.lin"
    pipe = tiny_pipe({mod: float_leaf(384, 64),
                      "blocks.0.attn.qkv": float_leaf(192, 64)})

    def pr(out, inp, rank):
        return (mx.array(rng.standard_normal((rank, inp)).astype(np.float32)),
                mx.array(rng.standard_normal((out, rank)).astype(np.float32)))
    tid = table_of(pipe, [(mod, (0, 0, 192), pr(192, 64, 4), 0.8),
                          (mod, (0, 192, 192), pr(192, 64, 2), 0.8),
                          ("blocks.0.attn.qkv", None, pr(192, 64, 4), 0.8)])
    how = install.form()
    figs = install.table_figures(pipe, tid, mx.bfloat16, frozenset({mod}))
    f32, b16 = (tid, mx.float32, how), (tid, mx.bfloat16, how)
    assert set(figs) == {f32, b16}
    for key, path in ((f32, mod), (b16, "blocks.0.attn.qkv")):
        t = install.build_table(pipe, *key)
        assert isinstance(t.get(path), install._Block)
        assert figs[key] == (t.nbytes, 0)
    assert figs[f32][0] == 4 * 6 * (64 + 384)
    # Once built and held, nothing more is due. A forward builds each leaf
    # at the one dtype it runs at.
    for key, path in ((f32, mod), (b16, "blocks.0.attn.qkv")):
        t = install.build_table(pipe, *key)
        t.get(path)
        install._keep_table(pipe, key, t, t.nbytes)
    assert all(f == h for f, h in install.table_figures(
        pipe, tid, mx.bfloat16, frozenset({mod})).values())


def test_a_pair_not_held_is_priced_from_its_shape():
    rng = np.random.default_rng(4)
    pipe = tiny_pipe({"blocks.0.attn.qkv": float_leaf(192, 64)})
    down = mx.array(rng.standard_normal((4, 64)).astype(np.float32))
    up = mx.array(rng.standard_normal((192, 4)).astype(np.float32))
    tid = table_of(pipe, [("blocks.0.attn.qkv", None, (down, up), 0.5)])
    (key,) = list(pipe.adapters.keys(bool))
    pipe.adapters.clear()
    shapes = {key: ((4, 64), (192, 4), mx.float32)}
    figs = install.table_figures(pipe, tid, mx.float32, shapes=shapes)
    assert figs[key] == (int(down.nbytes) + int(up.nbytes), 0)


# The trace test (F3): the leaves of each family that see float32 rows in
# a forward at bfloat16, against the family's lora_f32 list.
def serve_cast(model, keep, act=mx.bfloat16):
    """A tiny DiT of float32 weights cast as the loader casts a file's
    float tensors: float32 under a ``keep`` substring of the name, ``act``
    for the rest."""
    from mlx.utils import tree_flatten

    model.load_weights([
        (k, v.astype(mx.float32 if any(s in k for s in keep) else act))
        for k, v in tree_flatten(model.parameters())])
    model.act_dtype = act
    return model


def _image_pipe(name, model, variant):
    from gmlx.image import api
    from gmlx.image.registry import get_family

    fam = get_family(name)
    return api.ImagePipeline(dit=serve_cast(model, fam.fp32_keep),
                             family=fam, variant=variant)


def _fixture_step(name, gates, variant, **kw):
    """A FLUX.2, FLUX.3 or Krea 2 tiny DiT of its committed weights and
    the step of its committed inputs."""
    from oracle_fixture import committed_fixture

    from gmlx.image import api

    tiny = {"flux2_klein": "selftest_flux2_klein_tiny",
            "flux3_klein": "selftest_flux3_klein_tiny",
            "krea2": "selftest_krea2_tiny"}[name]
    weights, wm = committed_fixture(tiny + "_weights")
    arrays, _ = committed_fixture(tiny)
    pipe = _image_pipe(name, gates.tiny_model(
        weights, wm["inputs"]["config"], **kw), variant)
    cond = api.Conditioning(mx.array(arrays["features"]), "")
    lat = mx.array(arrays["latents"])
    extra = {"target": arrays["target"]} if "target" in arrays else {}
    return pipe, lambda tid: api.denoise_step(
        pipe, lat, arrays["sigma"], cond, patch_table=tid, **extra)


def _random_step(pipe, lat, feats, sigma):
    from gmlx.image import api

    rng = np.random.default_rng(5)
    lat = mx.array(rng.standard_normal(lat).astype(np.float32))
    cond = api.Conditioning(
        mx.array(rng.standard_normal(feats).astype(np.float32)), "")
    return pipe, lambda tid: api.denoise_step(pipe, lat, sigma, cond,
                                              patch_table=tid)


def _flux2(tmp_path, monkeypatch):
    import flux2_gates

    return _fixture_step("flux2_klein", flux2_gates, "distilled")


def _flux3(tmp_path, monkeypatch):
    import flux3_gates

    return _fixture_step("flux3_klein", flux3_gates, "distilled",
                         flow_map=True)


def _krea2(tmp_path, monkeypatch):
    import krea2_gates

    return _fixture_step("krea2", krea2_gates, "turbo")


def _moss(tmp_path, monkeypatch):
    from image_fixtures import tiny_moss_model

    return _random_step(_image_pipe("moss_image", tiny_moss_model(0), "sft"),
                        (1, 8, 3, 4), (5, 16), 0.6875)


def _z(tmp_path, monkeypatch):
    from image_fixtures import tiny_z_model

    return _random_step(_image_pipe("z_image", tiny_z_model(0)[0], "turbo"),
                        (1, 4, 6, 8), (5, 16), 0.75)


def _qwen(tmp_path, monkeypatch):
    """Through the loader, from a GGUF of the tiny weights."""
    import qwen_tiny as qt

    from gmlx.image import api, registry

    fam = qt.tiny_family()
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    pipe = api.load(qt.write(tmp_path / "tiny.gguf", qt.weights(0)),
                    log=lambda *_: None)
    lat, cond = qt.latents(), qt.conditioning("ref1")
    return pipe, lambda tid: api.denoise_step(pipe, lat, 0.6, cond,
                                              patch_table=tid)


def _h3(tmp_path, monkeypatch):
    import h3_gates
    from test_video_h3_model import TINY, tiny_weights

    from gmlx.video import pipeline
    from gmlx.video.families.h3 import family as hf

    m = h3_gates.tiny_model(tiny_weights(), TINY, mx.bfloat16)
    pipe = pipeline.VideoPipeline(dit=m, family=hf.FAMILY,  # type: ignore
                                  variant="fl2va")
    rng = np.random.default_rng(3)
    vs, aus = hf.latent_shapes(5, 64, 64)
    start = tuple(mx.array(rng.standard_normal(s).astype(np.float32))
                  for s in (vs, aus))
    feats = mx.array(rng.standard_normal((1, 8, TINY["text_dim"]))
                     .astype(np.float32))
    return pipe, lambda tid: pipeline.generate(
        pipe, feats, frames=5, width=64, height=64, steps=2,
        shifts=hf.SHIFTS, start=start, table=tid)


FAMILIES = {"flux2_klein": _flux2, "flux3_klein": _flux3, "krea2": _krea2,
            "moss_image": _moss, "z_image": _z, "qwen_image21": _qwen,
            "h3": _h3}


def every_leaf_table(pipe, paths, seed: int) -> str:
    """A table of two bfloat16 pairs on each of ``paths``."""
    from gmlx.image import api

    rng = np.random.default_rng(seed)
    leaves = install.linear_leaves(pipe.dit)
    entries = []
    for path in paths:
        out, inp = install.linear_dims(leaves[path])
        patches = []
        for rank, strength in ((2, 0.8), (1, 0.5)):
            down, up = (mx.array((rng.standard_normal(s) / 64)
                                 .astype(np.float32)).astype(mx.bfloat16)
                        for s in ((rank, inp), (out, rank)))
            patches.append((install.register_pair(pipe, down, up),
                            strength, float(rank)))
        entries.append((path, None, patches))
    return api.define_patch_table(pipe, entries)


def trace(pipe, run, tid) -> dict:
    """The dtypes each wrapped leaf sees in ``run(tid)``, by path."""
    seen: dict = {}
    real = install.ImageLoRALinear.__call__

    def rec(self, x):
        seen.setdefault(self._path, set()).add(x.dtype)
        return real(self, x)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(install.ImageLoRALinear, "__call__", rec)
        mx.eval(run(tid))
    return seen


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_each_family_lists_the_leaves_that_see_float32_rows(
        name, tmp_path, monkeypatch):
    """One forward at bfloat16 under a table on every linear leaf: the
    leaves that see float32 rows are the family's lora_f32 list. A table
    on the leaves the forward runs then builds what its dry run figures,
    each leaf at the dtype it runs at."""
    pipe, run = FAMILIES[name](tmp_path, monkeypatch)
    assert pipe.dit.act_dtype == mx.bfloat16
    every = sorted(install.linear_leaves(pipe.dit))
    seen = trace(pipe, run, every_leaf_table(pipe, every, 0))
    f32 = install.f32_leaves(pipe)
    assert f32
    assert {p for p, d in seen.items() if mx.float32 in d} == f32
    assert all(seen[p] == {mx.float32} for p in f32)
    assert all(d == {mx.bfloat16} for p, d in seen.items() if p not in f32)
    tid = every_leaf_table(pipe, sorted(seen), 1)
    before = install.table_figures(pipe, tid, mx.bfloat16, f32)
    assert before[(tid, mx.float32, install.form())][0] > 0
    mx.eval(run(tid))
    after = install.table_figures(pipe, tid, mx.bfloat16, f32)
    assert {k: h for k, (_f, h) in after.items()} == \
        {k: f for k, (f, _h) in before.items()}
