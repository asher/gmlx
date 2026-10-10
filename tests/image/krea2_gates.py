"""The rows of a Krea 2 step against a check 3 fixture of
scripts/image_oracle/check3_krea2.py: free running from the fixture's
inputs, and teacher-forced for each block and the final layer from the
fixture's float32 input to it."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gmlx.image.families.krea2 import model as km

# A tiny model's config in the keys the family synthesizes, from the
# diffusers config of its fixture.
_LAYOUT = {"patch_size": 2, "num_train_timesteps": 1000, "last_mlp": False}


def config_of(diffusers_config: dict) -> dict:
    c = dict(diffusers_config, **_LAYOUT)
    c["out_channels"] = c["in_channels"]
    return c


def tiny_model(weights: dict, diffusers_config: dict, act=mx.float32):
    """The DiT of float32 weights by GGUF name, in torch order."""
    m = km.Krea2DiT(config_of(diffusers_config))
    pairs = []
    for k, v in weights.items():
        a = np.asarray(v, np.float32)
        if k.endswith("mod.lin"):
            a = a.reshape(-1)
        pairs.append((k, mx.array(a)))
    m.load_weights(pairs)
    m.act_dtype = act
    m.eval()
    return m


def _np(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def rel(got, ref) -> float:
    g = np.asarray(got, np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


def _inputs(model, arrays: dict):
    lat = mx.array(arrays["latents"])
    feats = mx.array(arrays["features"])[None]
    sigma = np.reshape(np.float32(arrays["sigma"]), (1,))
    return lat, feats, sigma


def free_rows(model, arrays: dict, mod=None) -> dict:
    """Every row of one step from the fixture's inputs, and the
    velocity."""
    lat, feats, sigma = _inputs(model, arrays)
    taps: dict = {}
    t = model.timesteps(sigma, 1)
    temb = model.temb(t)
    out = {"tmlp": temb,
           "tproj": model.tproj[1](nn.gelu_approx(temb))}
    vel = model(lat, sigma, feats, taps, mod=mod)
    out.update(taps)
    got = {k: _np(v) for k, v in out.items()}
    got["velocity"] = _np(vel)
    return got


def fixture_modulation(model, arrays: dict) -> km.StepModulation:
    """The step's modulation from the fixture's timestep embedding and
    shared vectors, as the reference formed them."""
    act = model.act_dtype
    temb = mx.array(arrays["tmlp"]).astype(act)
    tvec = mx.array(arrays["tproj"]).astype(act)
    scale, shift = model.last.scale_shift(temb)
    return km.StepModulation(
        blocks=tuple(b.modulation(tvec) for b in model.blocks),
        scale=scale, shift=shift)


def forced_rows(model, arrays: dict) -> dict:
    """Each block and the final layer on the fixture's float32 input to
    it, with the fixture's modulation."""
    act = model.act_dtype
    mod = fixture_modulation(model, arrays)
    n_txt = int(arrays["features"].shape[0])
    _, _, h, w = arrays["latents"].shape
    p = model.cfg.patch
    cos, sin = model.rope(n_txt, h // p, w // p)
    out = {}
    for i, blk in enumerate(model.blocks):
        src = arrays["in.blocks.0"] if i == 0 else arrays[f"blocks.{i - 1}"]
        y = blk(mx.array(src).astype(act), mod.blocks[i], cos, sin)
        out[f"blocks.{i}"] = _np(y)
    y = model.last(mx.array(arrays["in.last"]).astype(act), mod.scale,
                   mod.shift)
    out["last"] = _np(y)
    return out


def measures(model, arrays: dict) -> dict:
    """Relative L2 of every row against the fixture: teacher-forced for
    the blocks and the final layer, free running for the rest, and
    ``free <row>`` for each block and the final layer run free."""
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
    return got


def missed(got: dict, tolerances: dict) -> dict:
    """The measures over their tolerance, by name. A ``free`` measure
    has no tolerance of its own and is not gated."""
    out = {}
    for k, v in got.items():
        if k.startswith("free "):
            continue
        tol = tolerances[k]["rel_l2"]
        if not v <= tol:
            out[k] = (v, tol)
    return out
