"""Rows of check 3 for Qwen-Image 2.1 against the diffusers oracle.

A fixture holds three calls of the oracle in fp32 on one conditioning:
the joint forward of the whole sequence on latents a at sigma a, the
extract call that keeps each block's K and V of the prefix, and the
cached forward of the target rows on latents b at sigma b.

A teacher-forced row gives a gmlx function the oracle's input to it. The
joint row of a block is the prefill of the prefix rows followed by the
target rows over that prefill's K and V, so one row holds both functions
and their join. The cached row of a block is the target rows over the
oracle's K and V. Each block also has two branch rows per call, the
outputs of its attention and of its MLP. The free-running pass is the
path a request takes: one ``prefill_prefix`` and a ``forward_target`` per
call. Its rows carry the names the manifest gives the oracle's own
free-running rows.

Check 7 reads the same rows with a patch table active, against a fixture
of the oracle's model with the adapter merged into its weights, and adds
the site gate of every linear leaf.

Every row reads the modulation through ``modulation`` and
``prefix_modulation`` of the model module, and ``oracle_temb`` gives
those the oracle's timestep embedding, so a row measures the model and
not the sinusoid.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import numpy as np

from oracle_fixture import fx

from gmlx.image.conditioning import Conditioning
from gmlx.image.families.qwen_image21 import layout
from gmlx.image.families.qwen_image21 import model as qm

CALLS = (("joint", "a"), ("cached", "b"))
BRANCHES = ("attn", "img_mlp")


def _np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def conditioning(arrays: dict, key: str = "check3") -> Conditioning:
    """The conditioning the oracle ran: its encoder rows, the image slots
    among them and the reference latents."""
    refs = sorted((k for k in arrays if k.startswith("ref_latents.")),
                  key=lambda k: int(k.rsplit(".", 1)[1]))
    return Conditioning(
        mx.array(arrays["features"]), key,
        ref_latents=tuple(mx.array(arrays[k]) for k in refs),
        image_slots=tuple(int(i) for i in np.flatnonzero(arrays["img_mask"])))


def has_blocks(arrays: dict) -> bool:
    return "joint.in" in arrays


@contextmanager
def oracle_temb(model, arrays: dict):
    """Give ``model`` the oracle's timestep embedding at the fixture's
    two timesteps and at t = 0. The embedder inputs must equal the
    oracle's bit for bit, which the lookup checks."""
    assert np.array_equal(arrays["temb_a"][1], arrays["temb_b"][1])
    table = {np.float32(0).tobytes(): arrays["temb_a"][1]}
    for tag in "ab":
        t = np.float32(arrays[f"t_embedded_{tag}"].reshape(()))
        table[t.tobytes()] = arrays[f"temb_{tag}"][0]

    def temb(t):
        rows = []
        for v in np.asarray(t, np.float32).reshape(-1):
            assert v.tobytes() in table, f"no oracle embedding at t = {v!r}"
            rows.append(table[v.tobytes()])
        return mx.array(np.stack(rows)).astype(model.act_dtype)

    model.temb = temb
    try:
        yield
    finally:
        del model.temb


@contextmanager
def recorded_branches(model, store: list):
    """Append ``(block, branch, output)`` to ``store`` at every call of a
    block's attention output and of its MLP."""
    names = {}
    for i, blk in enumerate(model.transformer_blocks):
        names[id(blk.attn)] = (i, "attn")
        names[id(blk.img_mlp)] = (i, "img_mlp")
    first = model.transformer_blocks[0]
    attn, mlp = type(first.attn), type(first.img_mlp)
    real_out, real_call = attn.out, mlp.__call__

    def out(self, o):
        y = real_out(self, o)
        store.append((*names[id(self)], y))
        return y

    def call(self, x):
        y = real_call(self, x)
        store.append((*names[id(self)], y))
        return y

    attn.out, mlp.__call__ = out, call
    try:
        yield store
    finally:
        attn.out, mlp.__call__ = real_out, real_call


def _tables(model, cond, height: int, width: int):
    cfg = model.cfg
    key = qm.prefix_key(cond)
    lay = layout.prefix_layout(*key)
    prefix = layout.prefix_tables(cfg.axes_dims, cfg.rope_theta, *key)
    target = layout.target_tables(cfg.axes_dims, cfg.rope_theta,
                                  lay.position, height, width)
    return lay, prefix, target


def _cache_layout(a: np.ndarray) -> np.ndarray:
    """The oracle's K or V, (1, P, heads, D), as (1, heads, P, D)."""
    return np.ascontiguousarray(a.transpose(0, 2, 1, 3))


def timestep_rows(model, arrays: dict) -> dict:
    """The timestep embedding and the shared modulation of each call from
    gmlx's own sinusoid, each with its t = 0 row."""
    out = {}
    for call, tag in CALLS:
        t = np.concatenate([model.timesteps(float(arrays[f"sigma_{tag}"]), 1),
                            np.zeros(1, np.float32)])
        temb = model.temb(t)
        rows = model.modulation_rows(temb)
        mx.eval(temb, rows)
        out[f"{call}.time_text_embed"] = fx.compare(
            _np(temb), arrays[f"temb_{tag}"])
        out[f"{call}.modulation"] = fx.compare(
            _np(rows), arrays[f"modulation_{tag}"])
    return out


def check3_rows(model, arrays: dict, branches: dict | None = None) -> dict:
    """Relative L2, cosine and max abs of every teacher-forced row
    against the oracle. With ``branches``, also fills it with the
    attention and MLP outputs of every block of both calls."""
    act = model.act_dtype
    cond = conditioning(arrays)
    lat = {tag: mx.array(arrays[f"latents_{tag}"]) for tag in "ab"}
    _, _, h, w = lat["a"].shape
    lay, (pcos, psin), (tcos, tsin) = _tables(model, cond, h, w)
    p = lay.tokens
    blocks = model.transformer_blocks
    last = len(blocks) - 1
    out: dict = {}

    def feed(key):
        return mx.array(arrays[key]).astype(act)

    def row(name, value, want):
        mx.eval(value)
        out[name] = fx.compare(_np(value), want)

    row("rope_cos", mx.concatenate([pcos, tcos], axis=0), arrays["rope_cos"])
    row("rope_sin", mx.concatenate([psin, tsin], axis=0), arrays["rope_sin"])
    row("txt_in", model.embed_text(mx.array(arrays["features"]))[None],
        arrays["txt_in.out"])
    images = [model.embed_image(r[None]) for r in cond.ref_latents]
    target = {tag: model.embed_image(lat[tag]) for tag in "ab"}
    row("img_in", mx.concatenate([*images, target["a"]], axis=1),
        arrays["img_in.out"])
    row("joint.in", mx.concatenate([model.embed_prefix(cond, lay),
                                    target["a"]], axis=1), arrays["joint.in"])
    row("cached.in", target["b"], arrays["cached.in"])
    for call, tag in CALLS:
        row(f"{call}.modulation", model.modulation_rows(feed(f"temb_{tag}")),
            arrays[f"modulation_{tag}"])

    with oracle_temb(model, arrays):
        zero = qm.prefix_modulation(model)
        mod = {tag: qm.modulation(model, float(arrays[f"sigma_{tag}"]))
               for tag in "ab"}
    seen: list = []
    with recorded_branches(model, seen):
        for i, blk in enumerate(blocks):
            name = f"transformer_blocks.{i}"
            src = "in" if i == 0 else f"transformer_blocks.{i - 1}.out"
            x = feed(f"joint.{src}")
            xp, k, v = blk.prefill(x[:, :p], zero, pcos, psin, lay.segments)
            xt = blk.target(x[:, p:], mod["a"], tcos, tsin, k, v)
            row(f"joint.{name}", mx.concatenate([xp, xt], axis=1),
                arrays[f"joint.{name}.out"])
            row(f"prefill.k.{i}", k, _cache_layout(arrays[f"cache.k.{i}"]))
            row(f"prefill.v.{i}", v, _cache_layout(arrays[f"cache.v.{i}"]))
            ck, cv = (mx.array(_cache_layout(arrays[f"cache.{s}.{i}"]))
                      .astype(act) for s in "kv")
            row(f"cached.{name}", blk.target(
                feed(f"cached.{src}"), mod["b"], tcos, tsin, ck, cv),
                arrays[f"cached.{name}.out"])
    if branches is not None:
        # A block records its prefix rows, the joint target rows and the
        # cached target rows, in that order, for each branch.
        for i in range(len(blocks)):
            for sub in BRANCHES:
                pre, tgt, cached = (y for j, s, y in seen
                                    if j == i and s == sub)
                name = f"transformer_blocks.{i}.{sub}"
                mx.eval(pre, tgt, cached)
                branches[f"joint.{name}"] = fx.compare(
                    _np(mx.concatenate([pre, tgt], axis=1)),
                    arrays[f"joint.{name}.out"])
                branches[f"cached.{name}"] = fx.compare(
                    _np(cached), arrays[f"cached.{name}.out"])
    x = feed(f"joint.transformer_blocks.{last}.out")
    row("joint.proj_out", mx.concatenate(
        [model.final(x[:, :p], zero), model.final(x[:, p:], mod["a"])],
        axis=1), arrays["joint.proj_out.out"])
    row("cached.proj_out", model.final(
        feed(f"cached.transformer_blocks.{last}.out"), mod["b"]),
        arrays["cached.proj_out.out"])
    return out


def free_run(model, arrays: dict, own_temb: bool = False) -> dict:
    """The path a request takes: one prefill and a target step per call.
    Returns the two velocities and, when the fixture holds them, every
    block of both calls, the cached K and V and the final projections.
    ``own_temb`` runs gmlx's own sinusoid."""
    cond = conditioning(arrays)
    blocks = has_blocks(arrays)
    taps: dict = {"prefix": {}, "a": {}, "b": {}}
    out = {}

    def run():
        prefix = qm.prefill_prefix(model, cond,
                                   taps["prefix"] if blocks else None)
        vel = {}
        for tag in "ab":
            vel[tag] = qm.forward_target(
                model, mx.array(arrays[f"latents_{tag}"]),
                float(arrays[f"sigma_{tag}"]), cond, prefix,
                taps[tag] if blocks else None)
            mx.eval(vel[tag], *taps[tag].values())
        return prefix, vel

    if own_temb:
        prefix, vel = run()
    else:
        with oracle_temb(model, arrays):
            prefix, vel = run()
    for tag in "ab":
        out[f"velocity_{tag}"] = fx.compare(_np(vel[tag]),
                                            arrays[f"velocity_{tag}"])
    if not blocks:
        return out
    p = prefix.tokens
    for i in range(len(model.transformer_blocks)):
        name = f"transformer_blocks.{i}"
        out[f"joint.{name}"] = fx.compare(
            _np(mx.concatenate([taps["prefix"][name], taps["a"][name]],
                               axis=1)), arrays[f"joint.{name}.out"])
        out[f"cached.{name}"] = fx.compare(
            _np(taps["b"][name]), arrays[f"cached.{name}.out"])
        out[f"cache.k.{i}"] = fx.compare(
            _np(prefix.k[i]), _cache_layout(arrays[f"cache.k.{i}"]))
        out[f"cache.v.{i}"] = fx.compare(
            _np(prefix.v[i]), _cache_layout(arrays[f"cache.v.{i}"]))
    # The oracle's joint projection also holds the prefix rows, which a
    # request never forms.
    out["joint.proj_out"] = fx.compare(
        _np(taps["a"]["proj_out"]), arrays["joint.proj_out.out"][:, p:])
    out["cached.proj_out"] = fx.compare(
        _np(taps["b"]["proj_out"]), arrays["cached.proj_out.out"])
    return out


def step(model, arrays: dict) -> dict:
    """Every measure of a fixture under one name each: the teacher-forced
    rows and the branch rows by their names, and each free-running
    measure as ``free <name>``."""
    got: dict = {}
    if has_blocks(arrays):
        branches: dict = {}
        got.update(check3_rows(model, arrays, branches))
        got.update(branches)
    got.update({f"free {k}": v for k, v in free_run(model, arrays).items()})
    return got


def missed(got: dict, tol: float) -> dict:
    """Name to relative L2 of each measure of ``got`` over ``tol``."""
    return {r: g["rel_l2"] for r, g in got.items() if not g["rel_l2"] <= tol}


def product_misses(m: dict, rows: dict, free: dict) -> list[str]:
    """Measures of the product path over the manifest's tolerances, each
    twice the oracle's own bf16 against fp32 gap at that point."""
    tol = m["tolerances"]
    out = [f"{r} rel L2 {g['rel_l2']:.3e} > {tol['rows_rel_l2'][r]:.3e}"
           for r, g in rows.items() if r in tol.get("rows_rel_l2", {})
           and not g["rel_l2"] <= tol["rows_rel_l2"][r]]
    for r, g in free.items():
        if r.startswith("velocity_"):
            if not g["rel_l2"] <= tol[f"{r}_rel_l2"]:
                out.append(f"{r} rel L2 {g['rel_l2']:.3e} > "
                           f"{tol[f'{r}_rel_l2']:.3e}")
            if not g["cosine"] >= tol[f"{r}_cosine_min"]:
                out.append(f"{r} cosine {g['cosine']:.6f} < "
                           f"{tol[f'{r}_cosine_min']}")
        elif not g["rel_l2"] <= tol["free_rel_l2"][r]:
            out.append(f"free {r} rel L2 {g['rel_l2']:.3e} > "
                       f"{tol['free_rel_l2'][r]:.3e}")
    return out


@contextmanager
def under_table(pipe, table: str):
    """Run the DiT of ``pipe`` under the patch table ``table``, as a
    denoise step does, so the rows above read the model with an adapter."""
    from gmlx.image.lora import install

    prev = install.activate(pipe, table)
    try:
        yield
    finally:
        install.restore(pipe, prev)


# The site gate of a linear leaf: relative L2 at most 2 n u, where u is
# one rounding to bfloat16 and n counts the outputs the stock path writes
# in the activation dtype: the product, and under an adapter also the
# product with A and the sum with the product with B.
SITE_UNIT = 2.0 ** -8
LEAF_ROUNDINGS = {False: 1, True: 3}


@contextmanager
def leaf_sites(pipe, table: str, sites: dict):
    """Fill ``sites`` with a reading for every call of a linear leaf of
    the DiT while a step runs under ``table``: (module path, call number)
    to (relative L2, gate). A reading is the leaf's output on the path a
    request takes against the same product in float32 on the leaf's
    dequantized weight and the same input, with the table's delta through
    its factors as that path holds them. The float32 side runs on the
    CPU, whose float32 products are exact. The leaves must be wrapped,
    which the first table with entries does."""
    import mlx_kquant as kq

    from gmlx.image.lora import install

    assert pipe.lora is not None, "no table with entries was defined"
    built = None
    if pipe.patch_tables[table]["entries"]:
        built = install.build_table(pipe, table, pipe.dit.act_dtype, "block")
    counts: dict = {}
    real = install.ImageLoRALinear.__call__

    def call(self, x):
        y = real(self, x)
        base = self.base
        codec = getattr(base, "kquant_type", None)
        f = None if built is None else built.get(self._path)
        with mx.stream(mx.cpu):
            x32 = x.astype(mx.float32)
            w = base["weight"].astype(mx.float32) if codec is None else \
                kq.dequantize(base["weight"], base["scales"], codec,
                              mx.float32)
            want = x32 @ w.T
            if f is not None:
                want = want + (x32 @ f.a_t.astype(mx.float32)) \
                    @ f.b_t.astype(mx.float32)
            e2 = mx.sum(mx.square(y.astype(mx.float32) - want))
            r2 = mx.sum(mx.square(want))
            mx.eval(e2, r2)
        n = counts[self._path] = counts.get(self._path, 0) + 1
        rel = 0.0 if e2.item() == 0 else float(np.sqrt(e2.item() / r2.item()))
        sites[(self._path, n)] = (
            rel, 2 * LEAF_ROUNDINGS[f is not None] * SITE_UNIT)
        return y

    install.ImageLoRALinear.__call__ = call
    try:
        yield sites
    finally:
        install.ImageLoRALinear.__call__ = real
