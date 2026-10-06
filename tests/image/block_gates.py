"""Block gates of the image acceptance checks. Check 3 reads the base
model through them and check 7 reads the model under an adapter.

A fixture holds one denoise step of the diffusers oracle in fp32: the
input and the output of every block, the outputs of each block's
attention and feed-forward modules, and the velocity. A teacher-forced
row gives a gmlx module the oracle's input to it and compares the output
with the oracle's. A late block adds a few percent to its output, so each
block also has two branch rows, the outputs of its two modules. The
free-running pass runs the whole step, which covers the plumbing between
blocks.

The exact gate checks structure. It runs the model in the oracle's fp32
arithmetic: fp32 activations, exact float32 products, every K-quant
product as a float32 product on the dequantized weight, and the stock
ops. ``assert_all_dense`` holds that no K-quant product ran on a kernel
there. Every row then matches the oracle to a few parts in a million and
the gate is EXACT_TOL, so a wrong model fails it at any size of fault
that matters. MLX reads MLX_ENABLE_TF32 once per process and the kernel
routes of ``gmlx.image.ops`` are set at import, so a test that needs the
exact path reruns itself in a child process through ``rerun_exact``.

The product gate runs the path a request takes, in the activation dtype
on the kernels the build has. It holds every block row, the free-running
blocks and the velocity to the manifest's tolerances, each twice the
oracle's own bf16 against fp32 gap at that point. It does not read the
branch rows: the oracle's gap on a branch row is one rounding sample, and
a right model lands on either side of twice that.

The site gate holds each op of the path a request takes to the same op in
the exact arithmetic on the same input. A site is an op between a block's
input and its output, read where the block calls it: a seam of
``gmlx.image.ops``, a norm, a linear leaf, or an op written in the block.
The site's output on the production input is compared with the stock op
in fp32 on that input, with exact float32 products on the dequantized
weight, so no softmax carries an earlier rounding into the reading. The
gate is a relative L2 of 2 n u over all rows, the image rows and the
caption rows, where u is the largest relative error of one rounding to
the activation dtype (2^-8 for bfloat16) and n counts the outputs the
stock path of the op writes in that dtype. Each such output is within u
of its exact value and the errors add to first order, so n u bounds a
stock op on a well-conditioned input and the factor 2 covers the
conditioning. n is a property of the op, not of the route: a kernel that
replaces a stock op is read at the same site against the same gate.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from contextlib import contextmanager

import mlx.core as mx
import numpy as np

from oracle_fixture import fx

BRANCHES = ("attention", "feed_forward")
GROUPS = ("noise_refiner", "context_refiner", "layers")

# The exact gate. Over check 3's fixture and the five of check 7 the right
# model measures 6.7e-6 at most on a teacher-forced row. A free-running
# measure carries the fp32 rounding of every block before it: the velocity
# measures 1.1e-5 at 512 and 1.3e-5 at 1024. The weakest wrong adapter
# table moves its most moved row by 6.4e-4, and the weakest structural
# fault of check 3, a norm eps ten times too large, by 2.7e-3.
EXACT_TOL = 5e-5


def _np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


@contextmanager
def recorded_branches(model, store: dict):
    """Record the attention and feed-forward outputs of every block into
    ``store`` by name, such as ``layers.3.attention``. The record is taken
    at the call of each module, whatever kernels run inside it."""
    names = {}
    for group in GROUPS:
        for i, blk in enumerate(getattr(model, group)):
            for sub in BRANCHES:
                names[id(getattr(blk, sub))] = f"{group}.{i}.{sub}"
    saved = {type(getattr(model.layers[0], sub)): None for sub in BRANCHES}

    def wrap(orig):
        def call(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            if id(self) in names:
                store[names[id(self)]] = out
            return out
        return call

    try:
        for cls in saved:
            saved[cls] = cls.__call__
            cls.__call__ = wrap(saved[cls])
        yield store
    finally:
        for cls, orig in saved.items():
            if orig is not None:
                cls.__call__ = orig


def check3_rows(model, arrays: dict, rows, branches: dict | None = None
                ) -> dict:
    """Relative L2, cosine and max abs of each row of ``rows`` against the
    oracle. With ``branches``, also fills it with the attention and
    feed-forward outputs of each block in ``rows`` against the oracle's."""
    act = model.act_dtype
    sigma = float(arrays["sigma"])
    lat = mx.array(arrays["latents"])
    cap = mx.array(arrays["cap_feats"])[None]
    adaln = mx.array(arrays["t_embedder.out"]).astype(act)
    wanted = set(rows)

    def feed(key):
        return mx.array(arrays[key]).astype(act)

    ht, wt = lat.shape[2] // model.cfg.patch, lat.shape[3] // model.cfg.patch
    n_cap = arrays["context_refiner.0.in"].shape[1]
    (xc, xs), (cc, cs) = model.rope(ht, wt, n_cap)
    cos = mx.concatenate([xc, cc], axis=0)
    sin = mx.concatenate([xs, cs], axis=0)

    got = {"t_embedder": model.adaln(sigma, 1),
           "x_embed": model.embed_image(lat),
           "cap_embed": model.embed_caption(cap)}
    ref = {"t_embedder": arrays["t_embedder.out"],
           "x_embed": arrays["noise_refiner.0.in"],
           "cap_embed": arrays["context_refiner.0.in"]}
    seen: dict = {}
    with recorded_branches(model, seen):
        for group, (c, s, mod) in {"noise_refiner": (xc, xs, True),
                                   "context_refiner": (cc, cs, False),
                                   "layers": (cos, sin, True)}.items():
            for i, blk in enumerate(getattr(model, group)):
                src = f"{group}.0.in" if i == 0 else f"{group}.{i - 1}.out"
                name = f"{group}.{i}"
                if name not in wanted:
                    continue
                got[name] = blk(feed(src), c, s, adaln if mod else None)
                ref[name] = arrays[f"{name}.out"]
    if branches is not None:
        for k, v in seen.items():
            mx.eval(v)
            branches[k] = fx.compare(_np(v), arrays[f"{k}.out"])
    last = f"layers.{len(model.layers) - 1}.out"
    got["final_layer"] = model.final_layer(feed(last), adaln)
    ref["final_layer"] = arrays["final_layer.out"]
    out = {}
    for row in rows:
        mx.eval(got[row])
        out[row] = fx.compare(_np(got[row]), ref[row])
    return out


class _EvalTaps(dict):
    """A taps dict that evaluates each block output as the forward
    records it, so one eval holds the work of one block."""

    def __setitem__(self, key, value):
        mx.eval(value)
        super().__setitem__(key, value)


def free_run(model, arrays: dict, blocks=(), stepwise: bool = False
             ) -> tuple[dict, dict]:
    """One free-running step. Returns the output of each named block and
    the velocity, each measured against the oracle's fp32 run. The block
    rows see the plumbing between blocks, which teacher forcing skips.
    ``stepwise`` evaluates block by block."""
    taps: dict = _EvalTaps() if stepwise else {}
    v = model(mx.array(arrays["latents"]), float(arrays["sigma"]),
              mx.array(arrays["cap_feats"])[None],
              taps=taps if blocks or stepwise else None)
    mx.eval(v, *(taps[b] for b in blocks))
    free = {b: fx.compare(_np(taps[b]), arrays[f"{b}.out"]) for b in blocks}
    return free, fx.compare(np.array(v), arrays["velocity"])


def velocity(model, arrays: dict) -> dict:
    return free_run(model, arrays)[1]


def gate(m: dict, rows: dict | None, vel: dict,
         free: dict | None = None) -> list[str]:
    """The product gate: rows, free-running blocks and velocity measures
    that miss the manifest's tolerances."""
    tol = m["tolerances"]
    misses = [f"{r} rel L2 {g['rel_l2']:.3e} > {tol['rows_rel_l2'][r]:.3e}"
              for r, g in (rows or {}).items()
              if not g["rel_l2"] <= tol["rows_rel_l2"][r]]
    misses += [f"free {b} rel L2 {g['rel_l2']:.3e} > "
               f"{tol['free_rel_l2'][b]:.3e}"
               for b, g in (free or {}).items()
               if not g["rel_l2"] <= tol["free_rel_l2"][b]]
    if not vel["rel_l2"] <= tol["velocity_rel_l2"]:
        misses.append(f"velocity rel L2 {vel['rel_l2']:.3e} > "
                      f"{tol['velocity_rel_l2']:.3e}")
    if not vel["cosine"] >= tol["velocity_cosine_min"]:
        misses.append(f"velocity cosine {vel['cosine']:.6f} < "
                      f"{tol['velocity_cosine_min']}")
    return misses


def thin(m: dict, rows: dict | None, vel: dict, free: dict | None = None,
         above: float = 0.8) -> list[str]:
    """Product gate readings above ``above`` of their gate. The cosine
    reading is its distance from 1 over the floor's distance from 1."""
    tol = m["tolerances"]
    out = [(g["rel_l2"] / tol[key][n], f"{kind}{n}")
           for kind, key, got in (("", "rows_rel_l2", rows),
                                  ("free ", "free_rel_l2", free))
           for n, g in (got or {}).items()]
    out.append((vel["rel_l2"] / tol["velocity_rel_l2"], "velocity rel L2"))
    out.append(((1 - vel["cosine"]) / (1 - tol["velocity_cosine_min"]),
                "velocity cosine"))
    return [f"{n} {r:.2f}" for r, n in sorted(out, reverse=True) if r > above]


# The exact path.

def tf32() -> bool:
    return os.environ.get("MLX_ENABLE_TF32", "1") != "0"


def stock_ops() -> bool:
    """Whether every image op seam is the stock op in this process. A
    build with kernel routes puts them in force at import and names the
    ones in force."""
    from gmlx.image import ops

    routes = getattr(ops, "kq_routes", None)
    return routes is None or not routes.active()


def exact_process() -> bool:
    """Whether this process can run the exact path."""
    return not tf32() and stock_ops()


def _rerun(request, tag: str, env: dict) -> None:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath,
        env=dict(os.environ, **env), capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if tag in line:
            print(line[line.index(tag):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert " passed" in r.stdout, r.stdout[-2000:]


def rerun_exact(request, tag: str) -> None:
    """Rerun this test in a child process with TF32 off and the image ops
    on the stock path, where it gates. The child's lines that carry
    ``tag`` are printed."""
    _rerun(request, tag, {"MLX_ENABLE_TF32": "0",
                          "GMLX_IMAGE_KQ_ROUTES": "0"})


def kquant_leaves(dit) -> dict:
    """id to path of every K-quant linear of ``dit``."""
    from mlx_kquant.nn import KQuantLinear

    return {id(m): path for path, m in dit.named_modules()
            if isinstance(m, KQuantLinear)}


class _Used(set):
    """The ids of the K-quant linears whose product ran dense in the
    exact process. ``direct`` counts the products that ran dense past
    their module, and ``kernel`` lists each product that ran on a kernel
    as (entry of mlx_kquant, module path or codec)."""

    def __init__(self):
        super().__init__()
        self.direct = 0
        self.kernel: list = []


@contextmanager
def exact_products(dit):
    """Run ``dit`` in the oracle's fp32 arithmetic: fp32 activations, and
    every K-quant product as a float32 product on the dequantized weight.
    The dequantized weight is a node of the lazy graph, so one eval holds
    the weights of one block. An adapter stays on the product path's
    code.

    A product runs dense here by either way a caller reaches it: the call
    of its K-quant linear, and ``kq.quantized_matmul``, which the split
    placement of an adapter table calls past the module with one band of
    the weight's rows. Every other entry of mlx_kquant is watched, and a
    call that takes the weight of a K-quant linear of ``dit`` is recorded
    as a product on a kernel. Yields the record, which
    ``assert_all_dense`` reads."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    assert exact_process(), ("the exact path needs MLX_ENABLE_TF32=0 and "
                             "GMLX_IMAGE_KQ_ROUTES=0")
    used = _Used()
    paths = kquant_leaves(dit)
    weights, owners = {}, {}
    for m in dit.modules():
        if id(m) in paths:
            weights[id(m["weight"])] = paths[id(m)]
            owners.setdefault(id(m["scales"]), []).append(id(m))

    def dense(self, x, lora=None):
        assert lora is None, "the exact path has no in-op adapter epilogue"
        used.add(id(self))
        w = kq.dequantize(self["weight"], self["scales"], self.kquant_type,
                          mx.float32)
        y = x @ w.T
        return y + self["bias"] if "bias" in self else y

    matmul = kq.quantized_matmul

    def direct(x, w, scales, kquant_type, transpose=True, **kwargs):
        # A band of a module's rows carries the module's scales array.
        owner = owners.get(id(scales), ())
        if len(owner) != 1 or not transpose or kwargs:
            used.kernel.append(("quantized_matmul", kquant_type))
            return matmul(x, w, scales, kquant_type, transpose=transpose,
                          **kwargs)
        used.add(owner[0])
        used.direct += 1
        return x @ kq.dequantize(w, scales, kquant_type, mx.float32).T

    def watched(name, fn):
        def call(*args, **kwargs):
            for a in (*args, *kwargs.values()):
                if id(a) in weights:
                    used.kernel.append((name, weights[id(a)]))
            return fn(*args, **kwargs)
        return call

    entries = {n: getattr(kq, n) for n in kq.__all__
               if n not in ("dequantize", "quantized_matmul")
               and callable(getattr(kq, n, None))
               and not isinstance(getattr(kq, n), type)}
    act, call = dit.act_dtype, KQuantLinear.__call__
    dit.act_dtype, KQuantLinear.__call__ = mx.float32, dense
    kq.quantized_matmul = direct
    for n, fn in entries.items():
        setattr(kq, n, watched(n, fn))
    try:
        yield used
    finally:
        dit.act_dtype, KQuantLinear.__call__ = act, call
        kq.quantized_matmul = matmul
        for n, fn in entries.items():
            setattr(kq, n, fn)


def assert_all_dense(dit, used) -> None:
    """No K-quant product of ``dit`` ran on a kernel in the exact process,
    and every K-quant linear ran as a dense product. A kernel that reads
    a weight past its module is recorded when it reaches mlx_kquant, and
    one that takes the place of a module's call leaves that module out."""
    assert not used.kernel, (
        f"{len(used.kernel)} K-quant products ran on a kernel in the exact "
        f"process, such as {used.kernel[:3]}")
    left = sorted(p for i, p in kquant_leaves(dit).items() if i not in used)
    assert not left, f"{len(left)} K-quant products ran off the exact " \
                     f"path, such as {left[:3]}"


def free_running(name: str) -> bool:
    """Whether a measure is free-running by its name: ``velocity`` and
    ``free <block>`` are, every other name is a teacher-forced row."""
    return name == "velocity" or name.startswith("free ")


def exact_missed(got: dict) -> dict:
    """Name to relative L2 of each measure of ``got`` over the exact
    gate."""
    return {r: g["rel_l2"] for r, g in got.items()
            if not g["rel_l2"] <= EXACT_TOL}


# The site gate.

SITE_VELOCITY = "GMLX_SITE_VELOCITY"
CHUNKS = ("scale_msa", "gate_msa", "scale_mlp", "gate_mlp")
NORMS = ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2")
LEAVES = {"qkv": "attention.qkv", "out": "attention.out",
          "w1": "feed_forward.w1", "w3": "feed_forward.w3",
          "w2": "feed_forward.w2", "adaLN_modulation": "adaLN_modulation.0"}
# The sites every block has, whatever kernels run inside them. A block
# without modulation writes its two residual sums in line.
_SEAMS = NORMS + ("qkv", "qk", "sdpa", "out", "glu", "w2")
SEAMS = {True: _SEAMS + ("adaLN_modulation", "modulate.msa", "modulate.mlp",
                         "tanh.msa", "tanh.mlp", "gated_residual.msa",
                         "gated_residual.mlp"),
         False: _SEAMS + ("residual.msa", "residual.mlp")}
# What feeds each site, as (site, input, source, output). An op between a
# source and its site would break the link, and the op would need a site.
_FED = {None: (("attention_norm1", 0, "block", 0),
               ("qkv", 0, "attention", 0), ("sdpa", 0, "qk", 2),
               ("sdpa", 1, "qk", 3), ("attention", -1, "out", -1),
               ("attention_norm2", 0, "out", -1),
               ("feed_forward", -1, "w2", -1), ("ffn_norm2", 0, "w2", -1)),
        True: (("modulate.msa", 0, "attention_norm1", -1),
               ("attention", 0, "modulate.msa", -1),
               ("gated_residual.msa", 0, "block", 0),
               ("gated_residual.msa", 2, "attention_norm2", -1),
               ("ffn_norm1", 0, "gated_residual.msa", -1),
               ("modulate.mlp", 0, "ffn_norm1", -1),
               ("feed_forward", 0, "modulate.mlp", -1),
               ("gated_residual.mlp", 0, "gated_residual.msa", -1),
               ("gated_residual.mlp", 2, "ffn_norm2", -1),
               ("block", -1, "gated_residual.mlp", -1)),
        False: (("attention", 0, "attention_norm1", -1),
                ("feed_forward", 0, "ffn_norm1", -1))}
_GLU = (("w1", 0, "feed_forward", 0), ("w3", 0, "feed_forward", 0),
        ("swiglu", 0, "w1", -1), ("swiglu", 1, "w3", -1),
        ("w2", 0, "swiglu", -1))

# n of the site gate: the outputs the stock path of an op writes in the
# activation dtype. n is a property of the op, not of the route, so a
# kernel that writes one output where the stock op writes three is held
# to the gate of three.
ROUNDINGS = {
    "sdpa": 1,            # one op
    "tanh": 1,
    "residual": 1,        # x + y
    "modulate": 2,        # 1 + scale, then the product
    "gated_residual": 2,  # gate * y, then the sum
    "swiglu": 3,          # sigmoid(a), a times it, then the product with b
    "lora.down": 1,       # x @ A^T
    "lora.add": 1,        # one addmm of the base product and z @ B^T
}


SITE_RULE = (
    "site gate: relative L2 <= 2 n u of each op's output on its own "
    "production input against the stock op in fp32, with exact float32 "
    "products on the dequantized weight, over all rows, the image rows and "
    "the caption rows. u = 2^-8 is one rounding to bfloat16, and n counts "
    "the outputs the stock path of the op writes in the activation dtype, "
    "whatever kernel runs. A wrong adapter table is visible only where the "
    "part of a leaf's delta it changes exceeds the gate of a site that "
    "reads it: a tenth of the largest leaf delta for a strength a tenth "
    "off, and all of it for a group left out")


def unit(dtype) -> float:
    """u of the site gate: the largest relative error of one rounding to
    ``dtype``."""
    return {mx.bfloat16: 2.0 ** -8, mx.float16: 2.0 ** -11}[dtype]


def norm_roundings(weight: mx.array, dtype) -> int:
    """``rms_norm`` rounds the normalized value and then the weight
    product for a weight in the activation dtype. For a weight in another
    dtype it runs in fp32 and rounds once."""
    return 2 if weight.dtype == dtype else 1


def digest(a) -> str:
    return hashlib.sha256(np.array(a).tobytes()).hexdigest()


def site_process() -> bool:
    """Whether this process can read sites: float32 products are exact.
    The kernel routes stay as the build has them, since the production
    side of a site is the path a request takes."""
    return not tf32()


def rerun_sites(request, tag: str, velocity: str | None = None) -> None:
    """Rerun this test in a child process with TF32 off and the kernel
    routes as this process has them. ``velocity`` is the digest of this
    process's velocity, which the child's production step must equal bit
    for bit."""
    _rerun(request, tag, {"MLX_ENABLE_TF32": "0",
                          **({SITE_VELOCITY: velocity} if velocity else {})})


def resolve(root, path: str):
    for part in path.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def _row_sums(got, want, rows: int = 1) -> tuple:
    """Squared error and squared exact norm per row of a site's output:
    (B, L, D), (B, H, L, D), a pair of those, or (B, D) in ``rows`` equal
    parts, the four chunks of a modulation."""
    if isinstance(got, tuple):
        e2, r2 = zip(*(_row_sums(g, w) for g, w in zip(got, want)))
        return sum(e2[1:], e2[0]), sum(r2[1:], r2[0])
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    if got.ndim == 2:
        got = got.reshape(got.shape[0], rows, -1)
        want = want.reshape(got.shape)
    axes = tuple(i for i in range(got.ndim) if i != got.ndim - 2)
    return (mx.sum(mx.square(got - want), axis=axes),
            mx.sum(mx.square(want), axis=axes))


def _rel(e2: float, r2: float) -> float:
    with np.errstate(all="ignore"):
        return 0.0 if e2 == 0 else float(np.sqrt(np.float64(e2) / r2))


class _Leaf:
    """A linear leaf of a block as the right model has it: the module the
    block holds, the module under an adapter wrapper, and its codec."""

    def __init__(self, block: str, leaf: str, obj):
        self.block, self.leaf, self.obj = block, leaf, obj
        self.name, self.path = f"{block}.{leaf}", f"{block}.{LEAVES[leaf]}"
        self.base = getattr(obj, "base", obj)
        self.codec = getattr(self.base, "kquant_type", None)
        self.chunks = leaf == "adaLN_modulation"

    def roundings(self) -> int:
        """One for the product. A K-quant leaf adds its bias in a second
        op and an ``nn.Linear`` in the same one."""
        return 2 if self.codec is not None and "bias" in self.base else 1

    def exact(self, x: mx.array, factors=None) -> tuple:
        """The product on fp32 ``x`` in float32 on the dequantized weight,
        with the delta of ``factors``, and that delta or None."""
        import mlx_kquant as kq

        b = self.base
        w = b["weight"].astype(mx.float32) if self.codec is None else \
            kq.dequantize(b["weight"], b["scales"], self.codec, mx.float32)
        y = x @ w.T
        if "bias" in b:
            y = y + b["bias"].astype(mx.float32)
        if factors is None:
            return y, None
        delta = (x @ factors[0]) @ factors[1]
        return y + delta, delta


def site_factors(pipe, entries):
    """Module path to the A^T and B^T of the table of ``entries`` as the
    stock path holds them, built in the activation dtype. The exact side
    of the site gate reads an adapter through them."""
    from gmlx.image import api
    from gmlx.image.lora import install

    built = install.build_table(pipe, api.define_patch_table(pipe, entries),
                                pipe.dit.act_dtype, "block")

    def get(path: str):
        f = built.get(path)
        return None if f is None else (f.a_t.astype(mx.float32),
                                       f.b_t.astype(mx.float32))
    return get


class _SiteTaps(dict):
    """A taps dict that reads the sites of each block as the forward
    records the block's output, so one eval holds one block."""

    def __init__(self, sites):
        super().__init__()
        self.sites = sites

    def __setitem__(self, key, value):
        self.sites.flush(value)


class Sites:
    """The site gate's reader for one free-running production step of
    ``model``. Make it on the right model, before any fault: it keeps the
    right leaves, codecs and norm weights for the exact side. ``factors``,
    from ``site_factors``, is the right adapter table.

    A site is read at the op's seam, the ``gmlx.image.ops`` function or
    the call of the leaf or norm, so a kernel route is measured at the
    same site as the stock op it replaces. n is a property of the op, not
    of the route: it is the count of ``ROUNDINGS``, ``norm_roundings`` and
    ``_Leaf.roundings`` on the stock path, whatever kernel ran. A product
    under an adapter delta counts two more than the leaf, the down product
    and the addmm. q and k norm with RoPE counts the norm and the cast
    after the fp32 rotation. The glu site, from the feed-forward input to
    the input of w2, counts its two products and ``swiglu``. Where the
    stock ops run inside a site they are sites too: w1, w3 and ``swiglu``
    inside glu, and the base product, the down product and the addmm
    inside a leaf under an adapter.

    After ``run``: ``readings`` and ``missed`` hold each site to its gate,
    ``effect`` is each adapter delta's share of its leaf's output, and
    ``broken`` lists what the block does that no site reads, which a right
    model leaves empty."""

    def __init__(self, model, factors=None):
        self.model, self.act = model, model.act_dtype
        self.u = unit(self.act)
        self._factors = factors or (lambda path: None)
        self.blocks, self.leaves, self.bases, self.norms = {}, {}, {}, {}
        self.attn, self.ffs, self.right = {}, {}, {}
        for group in GROUPS:
            for i, blk in enumerate(getattr(model, group)):
                name = f"{group}.{i}"
                rec = {"modulated": "adaLN_modulation" in blk,
                       "attention": blk.attention, "leaves": {}}
                for leaf, path in LEAVES.items():
                    if leaf == "adaLN_modulation" and not rec["modulated"]:
                        continue
                    lf = rec["leaves"][leaf] = _Leaf(name, leaf,
                                                     resolve(blk, path))
                    self.leaves[id(lf.obj)] = lf
                    if lf.base is not lf.obj:
                        self.bases[id(lf.base)] = lf
                for n in NORMS:
                    self.norms[id(getattr(blk, n))] = (name, n)
                self.attn[id(blk.attention)] = name
                self.ffs[id(blk.feed_forward)] = name
                self.blocks[id(blk)] = name
                self.right[name] = rec
        self.sites: dict = {}   # name -> (family, n, inside a seam site)
        self.chunked: set = set()  # sites that write a modulation
        self.sums: dict = {}    # name -> (error, exact norm), squared per row
        self.deltas: dict = {}  # adapter leaf -> squared delta per row
        self.broken: list = []
        self.n_img = 0
        self._pending, self._dpending, self._links = {}, {}, []
        self._state: dict = {}
        self._cur = self._leaf = None

    # Reading.

    def _st(self, block: str) -> dict:
        return self._state.setdefault(block, {"io": {}})

    def _site(self, block: str, site: str, family: str, n: int, got, want,
              ins=None, inner: bool = False, chunks: bool = False) -> None:
        name = f"{block}.{site}"
        outs = got if isinstance(got, tuple) else (got,)
        if any(o.dtype != self.act for o in outs):
            self.broken.append(f"{name} writes {outs[0].dtype}")
        self.sites[name] = (family, n, inner)
        if chunks:
            self.chunked.add(name)
        self._pending[name] = _row_sums(got, want,
                                        len(CHUNKS) if chunks else 1)
        if ins is not None:
            self._st(block)["io"][site] = (*ins, *outs)

    def _same(self, what: str, a: mx.array, b: mx.array) -> None:
        if a is b:
            return
        if a.shape != b.shape or a.dtype != b.dtype:
            self.broken.append(what)
        else:
            self._links.append((what, mx.array_equal(a, b)))

    def flush(self, *arrays) -> None:
        pend, deltas, links = self._pending, self._dpending, self._links
        self._pending, self._dpending, self._links = {}, {}, []
        mx.eval(*arrays, *(a for pair in pend.values() for a in pair),
                *deltas.values(), *(eq for _, eq in links))
        for name, pair in pend.items():
            self.sums[name] = tuple(np.array(a).astype(np.float64)
                                    for a in pair)
        for name, d2 in deltas.items():
            self.deltas[name] = np.array(d2).astype(np.float64)
        self.broken += [what for what, eq in links if not eq.item()]
        mx.clear_cache()

    def _groups(self, name: str, rows: int) -> dict:
        out = {"all": slice(None)}
        if name in self.chunked:
            out.update({c: slice(i, i + 1) for i, c in enumerate(CHUNKS)})
        elif name.startswith("layers.") and 0 < self.n_img < rows:
            out.update(image=slice(0, self.n_img),
                       caption=slice(self.n_img, None))
        return out

    def gate(self, name: str) -> float:
        return 2 * self.sites[name][1] * self.u

    def readings(self) -> dict:
        """Site to {row group: relative L2 over the site's gate}, in the
        order the step called the sites."""
        out = {}
        for name in self.sites:
            e2, r2 = self.sums[name]
            out[name] = {g: _rel(e2[s].sum(), r2[s].sum()) / self.gate(name)
                         for g, s in self._groups(name, len(e2)).items()}
        return out

    def missed(self) -> dict:
        """Site to its largest reading over its gate, for each site that
        has a row group over the gate or not finite."""
        out = {}
        for name, r in self.readings().items():
            if not all(v <= 1 for v in r.values()):
                out[name] = max(v if np.isfinite(v) else np.inf
                                for v in r.values())
        return out

    def effect(self) -> dict:
        """Adapter leaf to {row group: relative L2 of its delta against
        the leaf's exact output}."""
        out = {}
        for name, d2 in self.deltas.items():
            r2 = self.sums[name][1]
            out[name] = {g: _rel(d2[s].sum(), r2[s].sum())
                         for g, s in self._groups(name, len(d2)).items()}
        return out

    def table(self) -> dict:
        """(family, refiners or main, row group) to the count, the median
        and the largest of the readings over their gates, and the site of
        the largest."""
        cells: dict = {}
        for name, r in self.readings().items():
            part = "main" if name.startswith("layers.") else "refiners"
            for g, v in r.items():
                cells.setdefault((self.sites[name][0], part, g), []).append(
                    (v, name))
        return {k: (len(v), float(np.median([x for x, _ in v])), *max(v))
                for k, v in cells.items()}

    # Hooks.

    def _h_block(self, orig):
        s = self

        def call(blk, x, *a, **k):
            name = s.blocks.get(id(blk))
            if name is None:
                return orig(blk, x, *a, **k)
            if name.startswith("noise_refiner."):
                s.n_img = x.shape[1]
            s._cur = name
            try:
                out = orig(blk, x, *a, **k)
            finally:
                s._cur = None
            s._leave(name, x, out)
            return out
        return call

    def _h_module(self, names: dict, key: str):
        def hook(orig):
            s = self

            def call(mod, x, *a, **k):
                name = names.get(id(mod))
                if name is not None:
                    s._st(name)[key] = x
                out = orig(mod, x, *a, **k)
                if name is not None:
                    s._st(name)["io"][key] = (x, out)
                return out
            return call
        return hook

    def _h_norm(self, orig):
        from gmlx.image.ops import norms

        s, f = self, mx.float32

        def call(nrm, x):
            out = orig(nrm, x)
            hit = s.norms.get(id(nrm))
            if hit is not None:
                want = norms.rms_norm(x.astype(f), nrm.weight.astype(f),
                                      nrm.eps)
                s._site(*hit, "norm", norm_roundings(nrm.weight, x.dtype),
                        out, want, (x,))
            return out
        return call

    def _h_modulate(self, orig):
        from gmlx.image.ops import epilogues

        s, f = self, mx.float32

        def call(x, scale):
            out = orig(x, scale)
            if s._cur is not None:
                which = "mlp" if "modulate.msa" in s._st(s._cur)["io"] \
                    else "msa"
                s._site(s._cur, f"modulate.{which}", "modulate",
                        ROUNDINGS["modulate"], out,
                        epilogues.modulate(x.astype(f), scale.astype(f)),
                        (x, scale))
            return out
        return call

    def _h_gated_residual(self, orig):
        from gmlx.image.ops import epilogues

        s, f = self, mx.float32

        def call(x, gate, y):
            out = orig(x, gate, y)
            if s._cur is None:
                return out
            st = s._st(s._cur)
            which = "mlp" if "gated_residual.msa" in st["io"] else "msa"
            s._site(s._cur, f"gated_residual.{which}", "gated_residual",
                    ROUNDINGS["gated_residual"], out,
                    epilogues.gated_residual(x.astype(f), gate.astype(f),
                                             y.astype(f)), (x, gate, y))
            if "mod" in st:
                # The block writes tanh in line, on a chunk of its
                # modulation.
                c = _chunk(st["mod"], CHUNKS.index(f"gate_{which}"))
                s._site(s._cur, f"tanh.{which}", "tanh", ROUNDINGS["tanh"],
                        gate, mx.tanh(c.astype(f)))
            return out
        return call

    def _h_swiglu(self, orig):
        from gmlx.image.ops import epilogues

        s, f = self, mx.float32

        def call(a, b):
            out = orig(a, b)
            if s._cur is not None:
                s._site(s._cur, "swiglu", "swiglu", ROUNDINGS["swiglu"], out,
                        epilogues.swiglu(a.astype(f), b.astype(f)), (a, b),
                        inner=True)
            return out
        return call

    def _h_qk(self, orig):
        from gmlx.image.ops import attention

        s, f = self, mx.float32

        def call(q, k, q_w, k_w, cos, sin, eps):
            out = orig(q, k, q_w, k_w, cos, sin, eps)
            if s._cur is not None:
                att = s.right[s._cur]["attention"]
                qw, kw = att.q_norm.weight, att.k_norm.weight
                want = attention.qk_norm_rope(
                    q.astype(f), k.astype(f), qw.astype(f), kw.astype(f),
                    cos, sin, att.eps)
                n = 1 + max(norm_roundings(qw, q.dtype),
                            norm_roundings(kw, k.dtype))
                s._site(s._cur, "qk", "qk", n, tuple(out), want, (q, k))
            return out
        return call

    def _h_sdpa(self, orig):
        from gmlx.image.ops import attention

        s, f = self, mx.float32

        def call(q, k, v, mask=None):
            out = orig(q, k, v, mask=mask)
            if s._cur is not None:
                want = attention.sdpa(q.astype(f), k.astype(f), v.astype(f),
                                      mask=mask)
                s._site(s._cur, "sdpa", "sdpa", ROUNDINGS["sdpa"], out, want,
                        (q, k, v))
            return out
        return call

    def _h_leaf(self, orig):
        s, f = self, mx.float32

        def call(mod, x, *a, **k):
            lf = s.leaves.get(id(mod))
            if lf is None:
                y = orig(mod, x, *a, **k)
                lf = s.bases.get(id(mod))
                if lf is not None and s._leaf is not None \
                        and s._leaf[0] is lf \
                        and s._factors(lf.path) is not None:
                    s._site(lf.block, f"{lf.leaf}.base", "lora.base",
                            lf.roundings(), y, lf.exact(x.astype(f))[0],
                            (x,), inner=True, chunks=lf.chunks)
                return y
            prev, s._leaf = s._leaf, (lf, x)
            try:
                y = orig(mod, x, *a, **k)
            finally:
                s._leaf = prev
            s._read_leaf(lf, x, y)
            return y
        return call

    def _h_add(self, orig):
        s, f = self, mx.float32

        def call(y, z, b_t):
            out = orig(y, z, b_t)
            lf, x = s._leaf or (None, None)
            fac = None if lf is None else s._factors(lf.path)
            if fac is not None:
                s._site(lf.block, f"{lf.leaf}.down", "lora.down",
                        ROUNDINGS["lora.down"], z, x.astype(f) @ fac[0],
                        inner=True)
                s._site(lf.block, f"{lf.leaf}.add", "lora.add",
                        ROUNDINGS["lora.add"], out,
                        y.astype(f) + z.astype(f) @ fac[1], (y, z),
                        inner=True, chunks=lf.chunks)
            return out
        return call

    def _leaf_n(self, lf: _Leaf) -> int:
        return lf.roundings() + (0 if self._factors(lf.path) is None else 2)

    def _read_leaf(self, lf: _Leaf, x: mx.array, y: mx.array) -> None:
        from gmlx.image.ops import epilogues

        f, st = mx.float32, self._st(lf.block)
        want, delta = lf.exact(x.astype(f), self._factors(lf.path))
        self._site(lf.block, lf.leaf, f"leaf.{lf.leaf}", self._leaf_n(lf), y,
                   want, (x,), inner=lf.leaf in ("w1", "w3"),
                   chunks=lf.chunks)
        st[lf.leaf] = (x, want)
        if delta is not None:
            self._dpending[lf.name] = _row_sums(
                delta, delta, len(CHUNKS) if lf.chunks else 1)[1]
        if lf.leaf == "adaLN_modulation":
            st["mod"] = y
        if lf.leaf != "w2" or "feed_forward" not in st:
            return
        # The glu site: the feed-forward input to the input of w2, against
        # the block's right w1 and w3.
        x0, parts, n = st["feed_forward"], [], ROUNDINGS["swiglu"]
        for key in ("w1", "w3"):
            right = self.right[lf.block]["leaves"][key]
            n += self._leaf_n(right)
            hit = st.get(key)
            parts.append(hit[1] if hit is not None and hit[0] is x0 else
                         right.exact(x0.astype(f),
                                     self._factors(right.path))[0])
        self._site(lf.block, "glu", "glu", n, x, epilogues.swiglu(*parts))

    def _leave(self, name: str, x0: mx.array, out: mx.array) -> None:
        f = mx.float32
        st = self._state.pop(name, {"io": {}})
        io, modulated = st["io"], self.right[name]["modulated"]
        io["block"] = (x0, out)
        if not modulated and {"attention_norm2", "ffn_norm1",
                              "ffn_norm2"} <= io.keys():
            x1 = io["ffn_norm1"][0]
            for which, x, y, got in (
                    ("msa", x0, io["attention_norm2"][-1], x1),
                    ("mlp", x1, io["ffn_norm2"][-1], out)):
                self._site(name, f"residual.{which}", "residual",
                           ROUNDINGS["residual"], got,
                           x.astype(f) + y.astype(f))
        fed = list(_FED[None] + _FED[modulated])
        inner = {"w1", "w3", "swiglu"} & io.keys()
        if len(inner) == 3:
            fed += _GLU
        elif inner:
            self.broken.append(f"{name}: of the stock glu ops only "
                               f"{sorted(inner)} are called")
        for leaf in LEAVES:
            if f"{leaf}.add" in io:
                fed += [(f"{leaf}.base", 0, leaf, 0),
                        (f"{leaf}.add", 0, f"{leaf}.base", -1),
                        (leaf, -1, f"{leaf}.add", -1)]
        for site, i, src, j in fed:
            if site in io and src in io:
                self._same(f"{name}: an op between {src} and {site} is not "
                           f"a site", io[site][i], io[src][j])
        # The links through a view: the q, k and v of qkv, the heads of
        # sdpa, and the chunks of the modulation.
        if {"qkv", "qk", "sdpa", "out"} <= io.keys():
            q = io["qk"][0]
            qkv = io["qkv"][-1].reshape(*q.shape[:2], 3, *q.shape[2:])
            o = io["sdpa"][-1].transpose(0, 2, 1, 3)
            for what, a, b in (
                    ("qkv and qk", q, qkv[:, :, 0]),
                    ("qkv and qk", io["qk"][1], qkv[:, :, 1]),
                    ("qkv and sdpa", io["sdpa"][2],
                     qkv[:, :, 2].transpose(0, 2, 1, 3)),
                    ("sdpa and out", io["out"][0],
                     o.reshape(*o.shape[:2], -1))):
                self._same(f"{name}: an op between {what} is not a site",
                           a, b)
        if "mod" in st:
            for which in ("msa", "mlp"):
                if f"modulate.{which}" in io:
                    self._same(
                        f"{name}: an op between adaLN_modulation and "
                        f"modulate.{which} is not a site",
                        io[f"modulate.{which}"][1],
                        _chunk(st["mod"], CHUNKS.index(f"scale_{which}")))
        self.broken += [f"{name}: no {site} site" for site in SEAMS[modulated]
                        if f"{name}.{site}" not in self.sites]

    @contextmanager
    def _hooked(self):
        from gmlx.image import ops
        from gmlx.image.lora import install

        blk = self.model.layers[0]
        hooks = [(type(blk), "__call__", self._h_block),
                 (type(blk.attention), "__call__",
                  self._h_module(self.attn, "attention")),
                 (type(blk.feed_forward), "__call__",
                  self._h_module(self.ffs, "feed_forward")),
                 (type(blk.attention_norm1), "__call__", self._h_norm)]
        hooks += [(cls, "__call__", self._h_leaf) for cls in
                  {type(m) for lf in self.leaves.values()
                   for m in (lf.obj, lf.base)}]
        hooks += [(ops, "modulate", self._h_modulate),
                  (ops, "gated_residual", self._h_gated_residual),
                  (ops, "swiglu", self._h_swiglu),
                  (ops, "qk_norm_rope", self._h_qk),
                  (ops, "sdpa", self._h_sdpa)]
        if install.placement() == "block":
            hooks.append((install, "_add", self._h_add))
        saved = []
        try:
            for obj, attr, hook in hooks:
                saved.append((obj, attr, getattr(obj, attr)))
                setattr(obj, attr, hook(saved[-1][2]))
            yield
        finally:
            for obj, attr, orig in reversed(saved):
                setattr(obj, attr, orig)

    def run(self, arrays: dict) -> str:
        """Read every site of one free-running production step on the
        fixture's inputs. Returns the digest of the velocity."""
        with self._hooked():
            v = self.model(mx.array(arrays["latents"]),
                           float(arrays["sigma"]),
                           mx.array(arrays["cap_feats"])[None],
                           taps=_SiteTaps(self))
            self.flush(v)
        return digest(v)


def _chunk(mod: mx.array, i: int) -> mx.array:
    d = mod.shape[-1] // len(CHUNKS)
    return mod[:, None, i * d:(i + 1) * d]


def table_sites(pipe, arrays: dict, right, table=None) -> tuple[Sites, str]:
    """The sites of one step under the table of ``table``, the right one
    when None, against the exact arithmetic under the table of ``right``,
    and the digest of the velocity."""
    from gmlx.image import api
    from gmlx.image.lora import install

    tid = api.define_patch_table(pipe, right if table is None else table)
    sites = Sites(pipe.dit, site_factors(pipe, right))
    prev = install.activate(pipe, tid)
    try:
        v = sites.run(arrays)
    finally:
        install.restore(pipe, prev)
    return sites, v


def plain_velocity(model, arrays: dict) -> str:
    """The digest of the velocity of one free-running step with no site
    read, for ``rerun_sites``."""
    return digest(model(mx.array(arrays["latents"]), float(arrays["sigma"]),
                        mx.array(arrays["cap_feats"])[None]))


def right_sites(sites: Sites, velocity: str, label: str) -> None:
    """The site gate of a right model. Every op between a block's input
    and its output is a site, the step read is bit for bit the one the
    parent process ran without hooks, and every site is within its gate.
    Prints the largest reading of each family as a share of its gate."""
    want = os.environ.get(SITE_VELOCITY)
    assert want in (None, velocity), (
        f"{label}: the step the sites were read on is not the step of the "
        f"parent process")
    assert not sites.broken, sites.broken[:6]
    seams = sum(not inner for _, _, inner in sites.sites.values())
    assert seams == sum(len(SEAMS[r["modulated"]])
                        for r in sites.right.values()), seams
    top: dict = {}
    for (family, _, _), (_, _, v, _) in sites.table().items():
        top[family] = max(top.get(family, 0.0), v)
    print(f"{label}: {len(sites.sites)} sites, the largest reading over its "
          f"gate by family: "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(top.items())))
    assert not sites.missed(), (label, SITE_RULE, sites.missed())


def site_reach(effect: dict, got: Sites, part: float, where: str = "") -> float:
    """How far a wrong adapter table is over the site gate, in gates: the
    part of each leaf delta it changes, from the ``effect`` of the right
    run, over the gate of a site of ``got`` that reads the change, at its
    largest over the adapter leaves under ``where``. A strength a tenth
    off changes a tenth of every delta, and the addmm inside the leaf
    reads it on the stock path. A group left out changes all of its
    deltas, and only the leaf reads that, at the count of the leaf under
    its adapter. The table is visible only when this is over 1."""
    out = 0.0
    for leaf, groups in effect.items():
        if leaf.startswith(where):
            for site in (leaf, f"{leaf}.add"):
                if site in got.sites:
                    out = max(out, part * max(groups.values())
                              / got.gate(site))
    return out


def adapter_sites(effect: dict, where: str = "") -> set:
    """The sites a wrong table can move: each adapter leaf under
    ``where``, the stock ops inside it, and the glu site over a w1 or w3."""
    out = set()
    for leaf in effect:
        if leaf.startswith(where):
            out |= {leaf, f"{leaf}.base", f"{leaf}.down", f"{leaf}.add"}
            if leaf.rsplit(".", 1)[1] in ("w1", "w3"):
                out.add(leaf.rsplit(".", 1)[0] + ".glu")
    return out


def table_velocity(pipe, arrays: dict, entries) -> str:
    """``plain_velocity`` under the table of ``entries``."""
    from gmlx.image import api
    from gmlx.image.lora import install

    prev = install.activate(pipe, api.define_patch_table(pipe, entries))
    try:
        return plain_velocity(pipe.dit, arrays)
    finally:
        install.restore(pipe, prev)
