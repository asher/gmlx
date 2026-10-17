"""The rows of a FLUX.3 klein step against a check 3 fixture of
scripts/image_oracle/check3_flux3_klein.py: free running from the
fixture's inputs, and teacher-forced for each block and the final layer
from the fixture's float32 input to it and its modulations."""
from __future__ import annotations

import math
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

import block_gates as bg
import qwen_gates as qg

from gmlx.image.families.flux3_klein import model as fm

STREAMS = {"image": "content_mode_blocks.image",
           "image_cond": "content_mode_blocks.image_cond",
           "text": "txt_mode_blocks"}
EMBEDS = {"image": "emb_in.image", "image_cond": "emb_in.image_cond",
          "text": "txt_in"}


def tiny_model(weights: dict, config: dict, act=mx.float32,
               flow_map: bool = True):
    """The DiT of float32 weights by GGUF name, in torch order."""
    m = fm.Flux3KleinDiT(dict(config, flow_map=flow_map))
    m.load_weights([(k, mx.array(np.asarray(v, np.float32)))
                    for k, v in weights.items()
                    if flow_map or not k.startswith("ref_time_in.")])
    m.act_dtype = act
    m.eval()
    return m


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def rel(got, ref) -> float:
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


# The diffusers names of the GGUF's, as diffusers' FLUX.3 converter
# renames them.
ADALN = "final_layer.image.adaLN_modulation.1"
_D_GROUP = {"content_mode_blocks.image": "mode_blocks.image",
            "content_mode_blocks.image_cond": "mode_blocks.image_cond",
            "txt_mode_blocks": "txt_mode_blocks",
            "single_blocks": "single_transformer_blocks"}
_D_LEAF = {"q_proj": "attn.to_q", "k_proj": "attn.to_k",
           "v_proj": "attn.to_v", "attn_out": "attn.to_out.0",
           "mlp_in": "ff.linear_in", "mlp_out": "ff.linear_out"}
_D_TOP = {"emb_in.image": "x_embedders.image",
          "emb_in.image_cond": "x_embedders.image_cond",
          "txt_in": "context_embedder",
          "time_in.in_layer": "timestep_embedder.linear_1",
          "time_in.out_layer": "timestep_embedder.linear_2",
          "ref_time_in.in_layer": "target_timestep_embedder.linear_1",
          "ref_time_in.out_layer": "target_timestep_embedder.linear_2",
          "final_layer.image.linear": "proj_out", ADALN: "norm_out.linear"}


def diffusers_name(path: str) -> str:
    """The Flux3Transformer2DModel name of a gmlx module path."""
    if path in _D_TOP:
        return _D_TOP[path]
    if path.endswith(".lin"):
        return path[:-len(".lin")] + ".linear"
    for native, d in _D_GROUP.items():
        if path.startswith(native + "."):
            i, leaf = path[len(native) + 1:].split(".", 1)
            return f"{d}.{i}.{_D_LEAF[leaf]}"
    raise KeyError(path)


def swap_halves(a: np.ndarray) -> np.ndarray:
    """Rows of ``a`` with its two halves in the other order, as diffusers'
    swap_scale_shift turns the final modulation."""
    h = a.shape[0] // 2
    return np.concatenate([a[h:], a[:h]], axis=0)


def n_refs(arrays: dict) -> int:
    return sum(1 for k in arrays if k.startswith("refs."))


def inputs(model, arrays: dict):
    """latents, features, references, sigma and target of a fixture."""
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    refs = None
    if n_refs(arrays):
        refs = fm.References.of(
            [mx.array(arrays[f"refs.{i}"]) for i in range(n_refs(arrays))],
            model.cfg.image_rope_scale)
    sigma = np.reshape(np.float32(arrays["sigma"]), (1,))
    target = None
    if "target" in arrays:
        target = np.reshape(np.float32(arrays["target"]), (1,))
    return lat, feats, refs, sigma, target


def modes_of(arrays: dict) -> tuple:
    return tuple(m for m in fm.MODES if m != "image_cond" or n_refs(arrays))


def mod_rows(model, mod: fm.StepModulation, vec: dict, modes) -> dict:
    """The time vectors and modulations in the fixture's row names."""
    out = {}
    for m in modes:
        k = fm.MOD_KEYS[m]
        out[f"vec.{m}"] = vec[m]
        out[f"early_stream_modulations.{k}"] = mx.concatenate(
            mod.early[m], axis=-1)[:, 0]
        out[f"single_stream_modulations.{k}"] = mx.concatenate(
            mod.single[m], axis=-1)[:, 0]
    out["final_layer.image.adaLN_modulation"] = mx.concatenate(
        mod.final, axis=-1)[:, 0]
    return out


def free_rows(model, arrays: dict) -> dict:
    """Every row of one step from the fixture's inputs, and the
    velocity."""
    lat, feats, refs, sigma, target = inputs(model, arrays)
    taps: dict = {}
    vec = model.vectors(sigma, 1, target)
    mod = model.modulation(sigma, 1, target)
    out = mod_rows(model, mod, vec, modes_of(arrays))
    vel = model(lat, sigma, feats, taps, target=target, refs=refs, mod=mod)
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

    early, single = {}, {}
    for m in modes_of(arrays):
        k = fm.MOD_KEYS[m]
        early[m] = split(f"early_stream_modulations.{k}", 3)
        single[m] = split(f"single_stream_modulations.{k}", 3)
    return fm.StepModulation(
        early=early, single=single,
        final=split("final_layer.image.adaLN_modulation", 2))


def _ids(model, arrays: dict) -> dict:
    lat, feats, refs, _, _ = inputs(model, arrays)
    _, _, h, w = lat.shape
    ids = {"image": fm.image_ids(h, w, model.cfg.image_rope_scale),
           "text": fm.text_ids(feats.shape[1])}
    if refs is not None:
        ids["image_cond"] = refs.ids
    return ids


def forced_rows(model, arrays: dict) -> dict:
    """Each block and the final layer on the fixture's float32 input to
    it, with the fixture's modulations."""
    act = model.act_dtype
    mod = fixture_modulation(model, arrays)
    ids = _ids(model, arrays)
    out = {}
    modes = modes_of(arrays)
    for m in modes:
        blocks = (model.txt_mode_blocks if m == "text"
                  else model.content_mode_blocks[m])
        cos, sin = fm.rope_tables(ids[m], model.cfg.axes_dims,
                                  model.cfg.theta)
        for i, blk in enumerate(blocks):
            src = arrays[EMBEDS[m]] if i == 0 else \
                arrays[f"{STREAMS[m]}.{i - 1}"]
            y = blk(mx.array(src)[None].astype(act), mod.early[m], cos, sin)
            out[f"{STREAMS[m]}.{i}"] = _np(y)[0]
    lens = [arrays[EMBEDS[m]].shape[0] for m in modes]
    smod = tuple(mx.concatenate(
        [mx.broadcast_to(mod.single[m][j], (1, n, model.cfg.dim))
         for m, n in zip(modes, lens)], axis=1) for j in range(3))
    cos, sin = fm.rope_tables(np.concatenate([ids[m] for m in modes]),
                              model.cfg.axes_dims, model.cfg.theta)
    for i, blk in enumerate(model.single_blocks):
        src = arrays["in.single_blocks.0"] if i == 0 else \
            arrays[f"single_blocks.{i - 1}"]
        y = blk(mx.array(src)[None].astype(act), smod, cos, sin)
        out[f"single_blocks.{i}"] = _np(y)[0]
    y = model.final_layer["image"](
        mx.array(arrays["in.final_layer.image"])[None].astype(act),
        *mod.final)
    out["final_layer.image"] = _np(y)[0]
    return out


# Fixture arrays that are inputs of the step, not rows of it: a check 7
# fixture also holds the pairs of its adapter.
INPUTS = ("latents", "features", "sigma", "target", "velocity_bf16")
INPUT_PREFIXES = ("refs.", "ids.", "in.", "down.", "up.")


def fixture_rows(arrays: dict) -> set:
    """The rows a fixture holds: every array but the step's inputs."""
    return {k for k in arrays
            if k not in INPUTS and not k.startswith(INPUT_PREFIXES)}


def measures(model, arrays: dict) -> dict:
    """Relative L2 of every row against the fixture: teacher-forced for
    the blocks and the final layer, free running for the rest, and
    ``free <row>`` for each block and the final layer run free. The rows
    measured are the fixture's rows, each of them."""
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
    return name.startswith(tuple(STREAMS.values()) + ("single_blocks.",)) \
        or name == "final_layer.image"


def free_row(name: str) -> bool:
    """A free-running row of ``measures``: every measure but the
    teacher-forced rows and the velocity."""
    return not forced_row(name) and name != "velocity"


# The exact gate: the step in the oracle's fp32 arithmetic.

def exact_floats(model, path: str) -> int:
    """Every F32 and F16 tensor of the file into the model in float32,
    where the load keeps the activation dtype. BF16 tensors stay bf16:
    their values are exact in float32, and a product promotes them."""
    import gguf
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    reader = gguf.GGUFReader(path)
    floats = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
    pairs = []
    for t in reader.tensors:
        if t.tensor_type not in floats or t.name not in params:
            continue
        a = np.array(t.data, np.float32).reshape(params[t.name].shape)
        pairs.append((t.name, mx.array(a)))
    model.load_weights(pairs, strict=False)
    return len(pairs)


COND = ("emb_in.image_cond", "content_mode_blocks.image_cond.")


@contextmanager
def _exact(model, arrays: dict):
    """The exact path, after which every K-quant product of the step ran
    dense. A step with no reference image calls no leaf of the
    image_cond stream."""
    with bg.exact_products(model) as used:
        yield
        if n_refs(arrays):
            bg.assert_all_dense(model, used)
            return
        assert not used.kernel, used.kernel[:3]
        left = sorted(p for i, p in bg.kquant_leaves(model).items()
                      if i not in used)
        assert left and all(p.startswith(COND) for p in left), left[:3]


def exact_measures(model, arrays: dict) -> dict:
    with _exact(model, arrays):
        return measures(model, arrays)


def exact_free_floor(model, arrays: dict) -> dict:
    """Relative L2 of each free row of ``measures`` and of the velocity
    between the exact step on the GPU and on the CPU stream."""
    with _exact(model, arrays):
        gpu = free_rows(model, arrays)
        with mx.stream(mx.cpu):
            cpu = free_rows(model, arrays)
    out = {}
    for k, v in gpu.items():
        name = f"free {k}" if forced_row(k) else k
        if k in arrays or k == "velocity":
            out[name] = rel(v, cpu[k])
    return out


# The kinds of measure with a floor: manifest result and tolerance keys.
FLOORS = {"free": ("exact_free_floor", "exact_free"),
          "velocity": ("exact_velocity_floor", "exact_velocity")}


def exact_tols(m: dict) -> dict:
    out = {}
    for kind, (res, key) in FLOORS.items():
        floor = m["results"][res]["rel_l2"]
        tol = m["tolerances"][key]["rel_l2"]
        assert tol == max(bg.EXACT_TOL, 2 * floor), (kind, tol, floor)
        out[kind] = tol
    return out


def floor_now(floor: dict) -> dict:
    """The worst free row and the velocity of a floor measured now."""
    free = {k: v for k, v in floor.items() if k != "velocity"}
    top = max(free, key=free.get)
    return {"free": free[top], "free_at": top, "velocity": floor["velocity"]}


def check_floor(m: dict, floor: dict, label: str) -> dict:
    """The floors measured now are the floors the manifest records.
    Returns the manifest's tolerances."""
    now = floor_now(floor)
    print(f"{label}: fp32 floor of the free rows {now['free']!r} at "
          f"{now['free_at']}, of the velocity {now['velocity']!r}")
    for kind, (res, _) in FLOORS.items():
        assert res in m["results"], f"{m['name']} records no {res}"
    tols = exact_tols(m)
    for kind, (res, _) in FLOORS.items():
        rec = m["results"][res]["rel_l2"]
        print(f"{label}: recorded {kind} floor {rec:.2e}, gate "
              f"{tols[kind]:.2e}")
        assert now[kind] <= 1.25 * rec, (kind, now[kind], rec)
    return tols


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
# sdpa; glu, from the block's modulated input to the rows of mlp_out,
# whose stock ops round the mlp_in product and the swiglu, and the swiglu
# seam when the stock ops run it; out, the gated sum of the two output
# leaves added to the residual; and every linear leaf. On the residual
# route ``out`` is two of mlx-kquant's residual epilogues, each held to
# its own count, and its leaves are read through it.
OP_ROUNDINGS = {
    "adaln": 2,           # 1 + scale, then the cast of the fp32 result
    "qk": 2,              # the norm, then the cast after the fp32 rotation
    "sdpa": 1,
    "glu": 4,             # the mlp_in product, then swiglu
    "swiglu": 3,          # silu's sigmoid, silu, the product
    "out": 5,             # two leaf products, their sum, gate, residual
}
BLOCK_OPS = {"adaln": 1, "qk": 1, "sdpa": 1, "glu": 1, "swiglu": 1,
             "out": 1}
# The leaves a step reads past their call: the time vectors and the
# modulations run as float32 products on the dequantized weight.
NOT_CALLED_PREFIX = ("time_in.", "ref_time_in.", "early_stream_modulations.",
                     "single_stream_modulations.",
                     "final_layer.image.adaLN_modulation.")


def block_names(model) -> dict:
    """id of each block of ``model`` to its module path."""
    out = {}
    for m, prefix in STREAMS.items():
        blocks = (model.txt_mode_blocks if m == "text"
                  else model.content_mode_blocks[m])
        for i, b in enumerate(blocks):
            out[id(b)] = f"{prefix}.{i}"
    for i, b in enumerate(model.single_blocks):
        out[id(b)] = f"single_blocks.{i}"
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

    def factors(mod):
        g = None if built is None else built.get(paths.get(id(mod)))
        return None if g is None else (g.a_t.astype(f), g.b_t.astype(f))

    def exact_leaf(mod, x32):
        """The leaf's product in float32 on its dequantized weight, with
        its delta."""
        base = getattr(mod, "base", mod)
        codec = getattr(base, "kquant_type", None)
        w = base["weight"].astype(f) if codec is None else \
            kq.dequantize(base["weight"], base["scales"], codec, f)
        y = x32 @ w.T
        fac = factors(mod)
        if fac is not None:
            y = y + (x32 @ fac[0]) @ fac[1]
        return y

    def delta_rounds(*mods) -> int:
        """The roundings the leaves' deltas add on the stock ops: two
        products each."""
        return sum(2 for m in mods if factors(m) is not None)

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
        def run(blk, h):
            before = len(ran)
            y = orig(blk, h)
            if not cur:
                return y
            kernel = "glu" in ran[before:]

            def want():
                a, b = mx.split(exact_leaf(blk.mlp_in, h.astype(f)), 2,
                                axis=-1)
                return epilogues.swiglu(a, b)

            key = read("glu", y, want, bg.KERNEL_ROUNDINGS["glu"] if kernel
                       else OP_ROUNDINGS["glu"] + delta_rounds(blk.mlp_in))
            if kernel:
                if routed is not None:
                    routed[key] = bg.KERNEL_OPS["glu"]
                leaf_site(paths[id(blk.mlp_in)], *sites[key])
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

    def out_op(orig):
        def run(blk, x, gate, rows, hid):
            before = len(ran)
            y = orig(blk, x, gate, rows, hid)
            if not cur:
                return y
            fused = ran[before:].count("resid")
            n = math.prod(x.shape[:-1])
            each = 1 if n >= qg.RESID_ONE_ROUNDING_ROWS else 3

            def want():
                s = exact_leaf(blk.attn_out, rows.astype(f)) + \
                    exact_leaf(blk.mlp_out, hid.astype(f))
                return epilogues.gated_residual(x.astype(f), gate.astype(f),
                                                s)

            key = read("out", y, want, 2 * each if fused
                       else OP_ROUNDINGS["out"]
                       + delta_rounds(blk.attn_out, blk.mlp_out))
            if fused:
                if routed is not None:
                    routed[key] = "quantized_matmul_residual"
                for leaf in (blk.attn_out, blk.mlp_out):
                    leaf_site(paths[id(leaf)], *sites[key])
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
    hooks = [(fm.Block, "__call__", within),
             (fm.Block, "adaln", adaln), (fm.Block, "hidden", hidden),
             (fm.Block, "out", out_op),
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
    ``BLOCK_OPS`` once, the swiglu seam only where glu ran on the stock
    ops, and every linear leaf of a block that ran and every embedding
    leaf is read once."""
    from gmlx.image.lora import install

    got: dict = {}
    for block, _, op, _ in qg.op_sites(sites):
        got[(block, op)] = got.get((block, op), 0) + 1
    ran = {b for b, _ in got}
    glu_on = {k[0] for k, v in routed.items() if len(k) == 4 and k[2] == "glu"}
    want = {}
    for name in ran:
        for op, n in BLOCK_OPS.items():
            if op == "swiglu" and name in glu_on:
                continue
            want[(name, op)] = n
    out = sorted(f"{k}: {got.get(k, 0)} of {n}" for k, n in want.items()
                 if got.get(k, 0) != n)
    out += sorted(f"{k}: not a known op call" for k in got if k not in want)
    cond = "content_mode_blocks.image_cond.0" in ran

    def called(p: str) -> bool:
        if p.startswith(NOT_CALLED_PREFIX) or p in skipped:
            return False
        if p.startswith(tuple(STREAMS.values()) + ("single_blocks.",)):
            return p.rsplit(".", 1)[0] in ran
        return cond or p != "emb_in.image_cond"

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
