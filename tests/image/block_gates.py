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
ops. Every row then matches the oracle to a few parts in a million and
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
"""
from __future__ import annotations

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


def rerun_exact(request, tag: str) -> None:
    """Rerun this test in a child process with TF32 off and the image ops
    on the stock path, where it gates. The child's lines that carry
    ``tag`` are printed."""
    env = dict(os.environ, MLX_ENABLE_TF32="0", GMLX_IMAGE_KQ_ROUTES="0")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         request.node.nodeid], cwd=request.config.rootpath, env=env,
        capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if tag in line:
            print(line[line.index(tag):])
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert " passed" in r.stdout, r.stdout[-2000:]


def kquant_leaves(dit) -> dict:
    """id to path of every K-quant linear of ``dit``."""
    from mlx_kquant.nn import KQuantLinear

    return {id(m): path for path, m in dit.named_modules()
            if isinstance(m, KQuantLinear)}


@contextmanager
def exact_products(dit):
    """Run ``dit`` in the oracle's fp32 arithmetic: fp32 activations, and
    every K-quant product as a float32 product on the dequantized weight.
    The dequantized weight is a node of the lazy graph, so one eval holds
    the weights of one block. An adapter stays on the product path's
    code. Yields the ids of the K-quant linears whose product ran here,
    which ``assert_all_dense`` reads."""
    import mlx_kquant as kq
    from mlx_kquant.nn import KQuantLinear

    assert exact_process(), ("the exact path needs MLX_ENABLE_TF32=0 and "
                             "GMLX_IMAGE_KQ_ROUTES=0")
    used: set = set()

    def dense(self, x, lora=None):
        assert lora is None, "the exact path has no in-op adapter epilogue"
        used.add(id(self))
        w = kq.dequantize(self["weight"], self["scales"], self.kquant_type,
                          mx.float32)
        y = x @ w.T
        return y + self["bias"] if "bias" in self else y

    act, call = dit.act_dtype, KQuantLinear.__call__
    dit.act_dtype, KQuantLinear.__call__ = mx.float32, dense
    try:
        yield used
    finally:
        dit.act_dtype, KQuantLinear.__call__ = act, call


def assert_all_dense(dit, used: set) -> None:
    """Every K-quant linear of ``dit`` ran as a dense product. A kernel
    that reads a weight past its module would leave that one out."""
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
