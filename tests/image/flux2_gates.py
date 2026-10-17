"""The rows of a FLUX.2 klein step against a check 3 fixture of
scripts/image_oracle/check3_flux2_klein.py: free running from the
fixture's inputs, and teacher-forced for each block and the final layer
from the fixture's float32 input to it and its modulations, and the
exact and site gates over them."""
from __future__ import annotations

import math
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import block_gates as bg
import qwen_gates as qg
from flux3_gates import (FLOORS, check_floor, exact_floats, exact_tols,
                         rel, swap_halves)

from gmlx.image.families.flux2_klein import model as fm

__all__ = ["FLOORS", "check_floor", "exact_floats", "exact_tols", "rel",
           "swap_halves"]

MODS = ("double_stream_modulation_img", "double_stream_modulation_txt",
        "single_stream_modulation", "final_layer.adaLN_modulation")


# The diffusers names of each GGUF module, written apart from the map from
# diffusers' Apache-2.0 _convert_non_diffusers_flux2_lora_to_diffusers
# (diffusers 0.41): a fused q, k and v weight splits into thirds of its
# rows, and every other module takes one name.
_D_DOUBLE = (("img_attn.proj", "attn.to_out.0"),
             ("txt_attn.proj", "attn.to_add_out"),
             ("img_mlp.0", "ff.linear_in"), ("img_mlp.2", "ff.linear_out"),
             ("txt_mlp.0", "ff_context.linear_in"),
             ("txt_mlp.2", "ff_context.linear_out"))
_D_QKV = (("img_attn.qkv", ("attn.to_q", "attn.to_k", "attn.to_v")),
          ("txt_attn.qkv", ("attn.add_q_proj", "attn.add_k_proj",
                            "attn.add_v_proj")))
_D_SINGLE = (("linear1", "attn.to_qkv_mlp_proj"), ("linear2", "attn.to_out"))
_D_TOP = (("img_in", "x_embedder"), ("txt_in", "context_embedder"),
          ("time_in.in_layer",
           "time_guidance_embed.timestep_embedder.linear_1"),
          ("time_in.out_layer",
           "time_guidance_embed.timestep_embedder.linear_2"),
          ("final_layer.linear", "proj_out"),
          ("final_layer.adaLN_modulation.1", "norm_out.linear"),
          ("single_stream_modulation.lin", "single_stream_modulation.linear"),
          ("double_stream_modulation_img.lin",
           "double_stream_modulation_img.linear"),
          ("double_stream_modulation_txt.lin",
           "double_stream_modulation_txt.linear"))


def diffusers_names(depth: int, single: int) -> dict:
    """GGUF module to its [(diffusers name, third of its rows or None)]."""
    out = {}
    for i in range(depth):
        x, d = f"double_blocks.{i}", f"transformer_blocks.{i}"
        for leaf, dname in _D_DOUBLE:
            out[f"{x}.{leaf}"] = [(f"{d}.{dname}", None)]
        for leaf, names in _D_QKV:
            out[f"{x}.{leaf}"] = [(f"{d}.{n}", j) for j, n in enumerate(names)]
    for i in range(single):
        x, d = f"single_blocks.{i}", f"single_transformer_blocks.{i}"
        for leaf, dname in _D_SINGLE:
            out[f"{x}.{leaf}"] = [(f"{d}.{dname}", None)]
    for native, dname in _D_TOP:
        out[native] = [(dname, None)]
    return out


def tiny_model(weights: dict, config: dict, act=mx.float32):
    """The DiT of float32 weights by GGUF name, in torch order."""
    m = fm.Flux2KleinDiT(config)
    m.load_weights([(k, mx.array(np.asarray(v, np.float32)))
                    for k, v in weights.items()])
    m.act_dtype = act
    m.eval()
    return m


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def n_refs(arrays: dict) -> int:
    return sum(1 for k in arrays if k.startswith("refs."))


def inputs(model, arrays: dict):
    """latents, features, references and sigma of a fixture."""
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    refs = None
    if n_refs(arrays):
        refs = model.references(
            [mx.array(arrays[f"refs.{i}"]) for i in range(n_refs(arrays))])
    sigma = np.reshape(np.float32(arrays["sigma"]), (1,))
    return lat, feats, refs, sigma


@contextmanager
def oracle_time(arrays: dict):
    """The model reads the reference's time embedding at the fixture's
    sigma, so a row measures the model and not the sinusoid. The
    reference's fp32 exp can be an ulp off the correctly rounded one,
    which can move one bf16 entry of the embedding; ``own_time_gap``
    holds gmlx's own."""
    want = np.float32(arrays["sigma"].reshape(()))
    emb = arrays["in.time_in"]
    real = fm.time_embedding

    def time_embedding(t, dim):
        t = np.reshape(np.asarray(t, np.float32), (-1,))
        assert (t == want).all(), f"no oracle embedding at t = {t!r}"
        assert emb.shape[-1] == dim
        return mx.array(np.broadcast_to(emb[:1], (t.size, dim)))

    fm.time_embedding = time_embedding
    try:
        yield
    finally:
        fm.time_embedding = real


def own_time_gap(arrays: dict) -> tuple[int, float]:
    """The entries where gmlx's time embedding differs from the
    reference's, and the largest difference in bf16 ulps of the
    reference's entry."""
    got = np.array(fm.time_embedding(arrays["sigma"].reshape(()),
                                     arrays["in.time_in"].shape[-1]))[0]
    want = arrays["in.time_in"][0]
    diff = got != want
    if not diff.any():
        return 0, 0.0
    mag = np.maximum(np.abs(want[diff]), 2.0 ** -126)
    ulp = 2.0 ** (np.floor(np.log2(mag)) - 7)
    return int(diff.sum()), float((np.abs(got - want)[diff] / ulp).max())


def mod_rows(mod: fm.StepModulation, vec: mx.array) -> dict:
    """The time vector and the modulations in the fixture's row names."""
    def cat(chunks):
        return mx.concatenate(chunks, axis=-1)[:, 0]
    return {"vec": vec,
            "double_stream_modulation_img": cat([*mod.img[0], *mod.img[1]]),
            "double_stream_modulation_txt": cat([*mod.txt[0], *mod.txt[1]]),
            "single_stream_modulation": cat(mod.single),
            "final_layer.adaLN_modulation": cat(mod.final)}


def free_rows(model, arrays: dict) -> dict:
    """Every row of one step from the fixture's inputs, and the
    velocity."""
    lat, feats, refs, sigma = inputs(model, arrays)
    taps: dict = {}
    vec = model.vector(sigma, 1)
    mod = model.modulation(sigma, 1)
    out = mod_rows(mod, vec)
    vel = model(lat, sigma, feats, taps, refs=refs, mod=mod)
    out.update(taps)
    got = {k: _np(v) for k, v in out.items()}
    got = {k: v[0] if v.ndim == 3 else v for k, v in got.items()}
    got["velocity"] = _np(vel)
    return got


def fixture_modulation(model, arrays: dict) -> fm.StepModulation:
    """The step's modulations as the oracle formed them."""
    act = model.act_dtype

    def split(name, n):
        a = mx.array(arrays[name]).astype(act)[:, None, :]
        return tuple(mx.split(a, n, axis=-1))

    img = split("double_stream_modulation_img", 6)
    txt = split("double_stream_modulation_txt", 6)
    return fm.StepModulation(
        img=(img[:3], img[3:]), txt=(txt[:3], txt[3:]),
        single=split("single_stream_modulation", 3),
        final=split("final_layer.adaLN_modulation", 2))


def ids_of(model, arrays: dict) -> tuple:
    """The position ids of the image stream, references after the image,
    and of the text."""
    lat, feats, refs, _ = inputs(model, arrays)
    _, _, h, w = lat.shape
    img = fm.image_ids(h, w)
    if refs is not None:
        img = np.concatenate([img, refs.ids])
    return img, fm.text_ids(feats.shape[1])


def forced_rows(model, arrays: dict) -> dict:
    """Each block and the final layer on the fixture's float32 input to
    it, with the fixture's modulations."""
    act = model.act_dtype
    cfg = model.cfg
    mod = fixture_modulation(model, arrays)
    img_ids, txt_ids = ids_of(model, arrays)
    rope_img = fm.rope_tables(img_ids, cfg.axes_dims, cfg.theta)
    rope_txt = fm.rope_tables(txt_ids, cfg.axes_dims, cfg.theta)

    def row(name):
        return mx.array(arrays[name])[None].astype(act)

    out = {}
    for i, blk in enumerate(model.double_blocks):
        src = ("img_in", "txt_in") if i == 0 else \
            (f"double_blocks.{i - 1}.img", f"double_blocks.{i - 1}.txt")
        img, txt = blk(row(src[0]), row(src[1]), mod.img, mod.txt, rope_img,
                       rope_txt)
        out[f"double_blocks.{i}.img"] = _np(img)[0]
        out[f"double_blocks.{i}.txt"] = _np(txt)[0]
    cos, sin = fm.rope_tables(np.concatenate([txt_ids, img_ids]),
                              cfg.axes_dims, cfg.theta)
    n = arrays["in.single_blocks.0"].shape[0]
    smod = tuple(mx.broadcast_to(t, (1, n, cfg.dim)) for t in mod.single)
    for i, blk in enumerate(model.single_blocks):
        src = "in.single_blocks.0" if i == 0 else f"single_blocks.{i - 1}"
        out[f"single_blocks.{i}"] = _np(blk(row(src), smod, cos, sin))[0]
    y = model.final_layer(row("in.final_layer"), *mod.final)
    out["final_layer"] = _np(y)[0]
    return out


# Fixture arrays that are inputs of the step, not rows of it.
INPUTS = ("latents", "features", "sigma", "velocity_bf16")
INPUT_PREFIXES = ("refs.", "ids.", "in.", "down.", "up.")


def fixture_rows(arrays: dict) -> set:
    """The rows a fixture holds: every array but the step's inputs."""
    return {k for k in arrays
            if k not in INPUTS and not k.startswith(INPUT_PREFIXES)}


def measures(model, arrays: dict, own_time: bool = False) -> dict:
    """Relative L2 of every row against the fixture: teacher-forced for
    the blocks and the final layer, free running for the rest, and
    ``free <row>`` for each block and the final layer run free. The rows
    measured are the fixture's rows, each of them. The model reads the
    reference's time embedding unless ``own_time``."""
    if own_time:
        free = free_rows(model, arrays)
    else:
        with oracle_time(arrays):
            free = free_rows(model, arrays)
    forced = forced_rows(model, arrays)
    got = {}
    for k, v in free.items():
        if k == "velocity":
            got[k] = rel(v, arrays["velocity"])
        elif k in forced:
            got[k] = rel(forced[k], arrays[k])
            got[f"free {k}"] = rel(v, arrays[k])
        elif k in arrays:
            got[k] = rel(v, arrays[k])
    rows = {k for k in got if not k.startswith("free ")}
    want = fixture_rows(arrays)
    assert rows == want, (f"rows the step did not give: {sorted(want - rows)}"
                          f"; rows the fixture lacks: {sorted(rows - want)}")
    return got


def gate(name: str, m: dict) -> float:
    """The product gate of a measure: the manifest's tolerance of a row,
    and for a block or the final layer run free, ``free <row>``, twice the
    oracle's own bf16 against fp32 gap of that row run free."""
    if name.startswith("free "):
        return 2 * m["results"]["free_bf16_vs_fp32"][name[len("free "):]]
    return m["tolerances"][name]["rel_l2"]


def missed(got: dict, m: dict) -> dict:
    """The measures over their gate, by name."""
    out = {}
    for k, v in got.items():
        g = gate(k, m)
        if not v <= g:
            out[k] = (v, g)
    return out


def forced_row(name: str) -> bool:
    return name.startswith(("double_blocks.", "single_blocks.")) \
        or name == "final_layer"


def free_row(name: str) -> bool:
    """A free-running row of ``measures``: every measure but the
    teacher-forced rows and the velocity."""
    return not forced_row(name) and name != "velocity"


# The exact gate: the step in the oracle's fp32 arithmetic.

@contextmanager
def _exact(model):
    """The exact path, after which every K-quant product of the step ran
    dense."""
    with bg.exact_products(model) as used:
        yield
        bg.assert_all_dense(model, used)


def exact_measures(model, arrays: dict) -> dict:
    with _exact(model):
        return measures(model, arrays)


def exact_free_floor(model, arrays: dict) -> dict:
    """Relative L2 of each free row of ``measures`` and of the velocity
    between the exact step on the GPU and on the CPU stream."""
    with _exact(model), oracle_time(arrays):
        gpu = free_rows(model, arrays)
        with mx.stream(mx.cpu):
            cpu = free_rows(model, arrays)
    out = {}
    for k, v in gpu.items():
        name = f"free {k}" if forced_row(k) else k
        if k in arrays or k == "velocity":
            out[name] = rel(v, cpu[k])
    return out


def exact_gate(name: str, tols: dict) -> float:
    if name == "velocity":
        return tols["velocity"]
    if name.startswith("free "):
        return tols["free"]
    return tols["free"] if free_row(name) else bg.EXACT_TOL


def exact_over(got: dict, tols: dict) -> dict:
    return {k: (v, exact_gate(k, tols)) for k, v in got.items()
            if not v <= exact_gate(k, tols)}


def exact_line(label: str, got: dict, tols: dict) -> str:
    def worst(keys):
        k = max(keys, key=got.get)
        return f"{got[k]:.2e} at {k}"
    forced = [k for k in got if forced_row(k)]
    free = [k for k in got if free_row(k) and not k.startswith("free ")]
    run = [k for k in got if k.startswith("free ")]
    return (f"{label}: teacher-forced rows within {worst(forced)}, gate "
            f"{bg.EXACT_TOL:.0e}; velocity {got['velocity']:.2e}, gate "
            f"{tols['velocity']:.2e}; free rows within {worst(free)}, "
            f"blocks run free within {worst(run)}, gate {tols['free']:.2e}")


def fault_line(label: str, bad: dict, tols: dict) -> tuple[str, float]:
    parts = []
    for kind, keep in (("teacher-forced", forced_row),
                       ("velocity", lambda k: k == "velocity"),
                       ("free", lambda k: free_row(k) and k != "velocity")):
        r = {k: bad[k] / exact_gate(k, tols) for k in bad if keep(k)}
        miss = [k for k, v in r.items() if v > 1]
        if miss:
            top = max(miss, key=r.get)
            parts.append(f"{kind} {len(miss)} rows, up to {r[top]:.0f} of "
                         f"the gate at {top}")
    top = max(bad[k] / exact_gate(k, tols) for k in bad)
    return (f"{label}: misses through " + ("; ".join(parts) or "nothing"),
            top)


# The site gate: each op of a block on the path a request takes, against
# the stock op in float32 on the same production input, within 2 n u. The
# float32 side runs on the CPU, whose float32 products are exact. A reading
# is (block, "step", op, call number) for an op of a block, with
# ``qg.HEAD`` after it for the worst head row of a q or k site, and
# (module path, call number) for a linear leaf, so the helpers of
# qwen_gates.py read it.
#
# The sites of a block: adaln, the LayerNorm and modulation in fp32 that
# round at 1 + scale and at the cast; q and k through the q and k seam;
# sdpa; glu, from a double block's modulated input to the rows of mlp.2,
# whose stock ops round the mlp.0 product and the swiglu; swiglu, the seam,
# where the stock ops run it; out, a gated product added to the residual,
# of a double block's proj or mlp.2 or a single block's linear2 on its
# attention and feed-forward rows; and every linear leaf. On the residual
# route ``out`` is one of mlx-kquant's residual epilogues, held to its own
# count, and its leaf is read through it.
OP_ROUNDINGS = {
    "adaln": 2,           # 1 + scale, then the cast of the fp32 result
    "qk": 2,              # the norm, then the cast after the fp32 rotation
    "sdpa": 1,
    "glu": 4,             # the mlp.0 product, then swiglu
    "swiglu": 3,          # silu's sigmoid, silu, the product
    "out": 3,             # the leaf product, the gate, the residual
}
# Calls of each op in one block of each kind.
BLOCK_OPS = {"double": {"adaln": 4, "qk": 2, "sdpa": 1, "glu": 2,
                        "swiglu": 2, "out": 4},
             "single": {"adaln": 1, "qk": 1, "sdpa": 1, "swiglu": 1,
                        "out": 1}}
# The leaves a step reads past their call: the time vector and the
# modulations run as float32 products on the dequantized weight.
NOT_CALLED_PREFIX = ("time_in.", *(f"{m}." for m in MODS))


def block_names(model) -> dict:
    """id of each block of ``model`` to its module path."""
    out = {id(b): f"double_blocks.{i}"
           for i, b in enumerate(model.double_blocks)}
    out.update({id(b): f"single_blocks.{i}"
                for i, b in enumerate(model.single_blocks)})
    return out


@contextmanager
def step_sites(model, sites: dict, routed: dict | None = None,
               heads: dict | None = None, built=None,
               skipped: set | None = None):
    """Fill ``sites`` with a reading for every op of every block and every
    call of a linear leaf while steps run: key to (relative L2, gate).
    ``routed`` takes the mlx-kquant op of each site a kernel route ran,
    ``heads`` the place of the worst head row of each q and k site.
    ``built`` is the built table of the patch table the step runs under,
    from ``install.build_table`` in the block form, and gives the exact
    side each leaf's delta. ``skipped`` takes the path of a leaf called on
    a float32 input, which runs in float32 and has no site."""
    import mlx_kquant as kq

    from gmlx.image import ops
    from gmlx.image.lora import install
    from gmlx.image.ops import attention, epilogues

    f = mx.float32
    act = model.act_dtype
    names = block_names(model)
    cur: list = []
    counts: dict = {}
    calls: dict = {}
    ran: list = []
    # Under a patch table a path holds the leaf's ImageLoRALinear.
    held = {p: m for p, m in install.modules(model).items()
            if install.linear_dims(m) is not None}
    paths = {id(m): p for p, m in held.items()}

    def factors(path):
        g = None if built is None else built.get(path)
        return None if g is None else (g.a_t.astype(f), g.b_t.astype(f))

    def exact_leaf(mod, x32, path=None):
        """The leaf's product in float32 on its dequantized weight, with
        its delta."""
        base = getattr(mod, "base", mod)
        codec = getattr(base, "kquant_type", None)
        w = base["weight"].astype(f) if codec is None else \
            kq.dequantize(base["weight"], base["scales"], codec, f)
        y = x32 @ w.T
        fac = factors(paths.get(id(mod)) if path is None else path)
        if fac is not None:
            y = y + (x32 @ fac[0]) @ fac[1]
        return y

    def delta_rounds(mod) -> int:
        """The roundings a leaf's delta adds on the stock ops: its two
        products."""
        return 2 if factors(paths.get(id(mod))) is not None else 0

    def read(op, got, want, rounds, by_head=False):
        key = (cur[-1], "step", op)
        n = counts[key] = counts.get(key, 0) + 1
        outs = got if isinstance(got, tuple) else (got,)
        same = all(o.dtype == act for o in outs)
        gate = 2 * rounds * qg.SITE_UNIT * same
        with mx.stream(mx.cpu):
            exact = want()
            r = qg._rel_l2(got, exact)
            if by_head:
                e2, r2 = bg._head_sums(outs, exact if isinstance(
                    exact, tuple) else (exact,))
                mx.eval(e2, r2)
        sites[(*key, n)] = (r, gate)
        if by_head:
            worst, at = bg._worst(np.array(e2), np.array(r2))
            sites[(*key, n, qg.HEAD)] = (worst, gate)
            if heads is not None:
                heads[(*key, n)] = at
        return (*key, n)

    def leaf_site(path, r, gate):
        n = calls[path] = calls.get(path, 0) + 1
        sites[(path, n)] = (r, gate)

    def within(orig):
        def run(blk, *a, **k):
            hit = names.get(id(blk))
            if hit is None:
                return orig(blk, *a, **k)
            cur.append(hit)
            try:
                return orig(blk, *a, **k)
            finally:
                cur.pop()
        return run

    def adaln(orig):
        def run(blk, x, shift, scale):
            y = orig(blk, x, shift, scale)
            if cur:
                read("adaln", y, lambda: fm.adaln(
                    x.astype(f), shift.astype(f), scale.astype(f), blk.eps,
                    f), OP_ROUNDINGS["adaln"])
            return y
        return run

    def qk(orig):
        def run(q, k, q_w, k_w, cos, sin, eps):
            before = len(ran)
            out = orig(q, k, q_w, k_w, cos, sin, eps)
            if not cur:
                return out
            kernel = "qk" in ran[before:]
            key = read("qk", tuple(out), lambda: attention.qk_norm_rope(
                q.astype(f), k.astype(f), q_w.astype(f), k_w.astype(f),
                cos, sin, eps),
                bg.KERNEL_ROUNDINGS["qk"] if kernel else OP_ROUNDINGS["qk"],
                by_head=True)
            if kernel and routed is not None:
                routed[key] = bg.KERNEL_OPS["qk"]
            return out
        return run

    def sdpa(orig):
        def run(q, k, v, mask=None):
            out = orig(q, k, v, mask)
            if cur:
                read("sdpa", out, lambda: attention.sdpa(
                    q.astype(f), k.astype(f), v.astype(f), mask),
                    OP_ROUNDINGS["sdpa"])
            return out
        return run

    def hidden(orig):
        def run(blk, mlp, h):
            before = len(ran)
            y = orig(blk, mlp, h)
            if not cur:
                return y
            kernel = "glu" in ran[before:]

            def want():
                a, b = mx.split(exact_leaf(mlp[0], h.astype(f)), 2, axis=-1)
                return epilogues.swiglu(a, b)

            key = read("glu", y, want, bg.KERNEL_ROUNDINGS["glu"] if kernel
                       else OP_ROUNDINGS["glu"] + delta_rounds(mlp[0]))
            if kernel:
                if routed is not None:
                    routed[key] = bg.KERNEL_OPS["glu"]
                leaf_site(paths[id(mlp[0])], *sites[key])
            return y
        return run

    def swiglu(orig):
        def run(a, b):
            out = orig(a, b)
            if cur:
                read("swiglu", out, lambda: epilogues.swiglu(
                    a.astype(f), b.astype(f)), OP_ROUNDINGS["swiglu"])
            return out
        return run

    def gated(lin, x, gate, rows, y, before):
        fused = "resid" in ran[before:]
        n = math.prod(x.shape[:-1])
        each = 1 if n >= qg.RESID_ONE_ROUNDING_ROWS else 3

        def want():
            return epilogues.gated_residual(
                x.astype(f), gate.astype(f), exact_leaf(lin, rows.astype(f)))

        key = read("out", y, want, each if fused
                   else OP_ROUNDINGS["out"] + delta_rounds(lin))
        if fused:
            if routed is not None:
                routed[key] = "quantized_matmul_residual"
            leaf_site(paths[id(lin)], *sites[key])

    def double_out(orig):
        def run(blk, lin, x, gate, rows):
            before = len(ran)
            y = orig(blk, lin, x, gate, rows)
            if cur:
                gated(lin, x, gate, rows, y, before)
            return y
        return run

    def single_out(orig):
        def run(blk, x, gate, rows, hid):
            before = len(ran)
            y = orig(blk, x, gate, rows, hid)
            if cur:
                gated(blk.linear2, x, gate,
                      mx.concatenate([rows, hid], axis=-1), y, before)
            return y
        return run

    def resid(orig):
        def run(x, leaf, residual, gate):
            ran.append("resid")
            return orig(x, leaf, residual, gate)
        return run

    def leaf(orig):
        def run(mod, x, *a, **k):
            before = len(ran)
            y = orig(mod, x, *a, **k)
            path = paths.get(id(mod))
            if path is None:
                return y
            if x.dtype != act:
                if skipped is not None:
                    skipped.add(path)
                return y
            with mx.stream(mx.cpu):
                r = qg._rel_l2(y, exact_leaf(mod, x.astype(f)))
            rounds = 1 + delta_rounds(mod)
            kernel = "lora" in ran[before:]
            if kernel:
                rounds = bg.KERNEL_ROUNDINGS["lora"]
            leaf_site(path, r, 2 * rounds * qg.SITE_UNIT)
            if kernel and routed is not None:
                routed[(path, calls[path])] = bg.KERNEL_OPS["lora"]
            return y
        return run

    # A leaf under a patch table is read at its ImageLoRALinear, which
    # calls its base inside.
    leaf_types = {type(m) for m in held.values()}
    hooks = [(fm.DoubleBlock, "__call__", within),
             (fm.SingleBlock, "__call__", within),
             (fm.DoubleBlock, "adaln", adaln), (fm.SingleBlock, "adaln", adaln),
             (fm.DoubleBlock, "hidden", hidden),
             (fm.DoubleBlock, "out", double_out),
             (fm.SingleBlock, "out", single_out),
             (ops, "qk_norm_rope", qk), (ops, "sdpa", sdpa),
             (ops, "swiglu", swiglu), (ops, "resid_fused", resid)]
    hooks += [(cls, "__call__", leaf) for cls in leaf_types]
    saved = []
    try:
        for obj, attr, hook in hooks:
            saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, hook(saved[-1][2]))
        with bg.kernel_calls(ran.append):
            yield sites
    finally:
        for obj, attr, orig in reversed(saved):
            setattr(obj, attr, orig)


def missing_ops(model, sites: dict, routed: dict, skipped=()) -> list:
    """What one step leaves unread: each block that ran calls every op of
    ``BLOCK_OPS`` of its kind as often as it names, the swiglu seam of a
    double block only where glu ran on the stock ops, and every linear leaf
    of a block that ran and every embedding leaf is read once."""
    from gmlx.image.lora import install

    got: dict = {}
    for block, _, op, _ in qg.op_sites(sites):
        got[(block, op)] = got.get((block, op), 0) + 1
    ran = {b for b, _ in got}
    glu_on: dict = {}
    for k, v in routed.items():
        if len(k) == 4 and k[2] == "glu":
            glu_on[k[0]] = glu_on.get(k[0], 0) + 1
    want = {}
    for name in ran:
        kind = name.split("_", 1)[0]
        for op, n in BLOCK_OPS[kind].items():
            if op == "swiglu" and kind == "double":
                n -= glu_on.get(name, 0)
            if n:
                want[(name, op)] = n
    out = sorted(f"{k}: {got.get(k, 0)} of {n}" for k, n in want.items()
                 if got.get(k, 0) != n)
    out += sorted(f"{k}: not a known op call" for k in got if k not in want)

    def called(p: str) -> bool:
        if p.startswith(NOT_CALLED_PREFIX) or p in skipped:
            return False
        if p.startswith(("double_blocks.", "single_blocks.")):
            return ".".join(p.split(".")[:2]) in ran
        return True

    leaves = {p for p in install.linear_leaves(model) if called(p)}
    read: dict = {}
    for k in sites:
        if len(k) == 2:
            read[k[0]] = read.get(k[0], 0) + 1
    out += sorted(f"leaf {p}: {read.get(p, 0)} of 1" for p in leaves
                  if read.get(p, 0) != 1)
    out += sorted(f"leaf {p}: not a leaf a step calls" for p in read
                  if p not in leaves)
    return out
