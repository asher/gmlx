"""Check 3 of the image acceptance gate for Z-Image: one denoise step of the
gmlx DiT, block by block, against the diffusers oracle.

Each teacher-forced row gives a gmlx module the oracle's fp32 input to
it, cast to the activation dtype, with the oracle's timestep embedding,
and compares the output with the oracle's fp32 output. The free-running
pass runs the whole step and compares the output of every block and the
velocity, which covers the plumbing between blocks. Tolerances come from
the manifests committed under tests/fixtures/image/oracle/, where each is
twice the oracle's own bf16 against fp32 gap at the same point. The arrays are read from GMLX_IMAGE_FIXTURES and must
match those manifests. scripts/image_oracle/check3_z_image.py writes them.

The tiny row runs on the CPU and needs only the self-test fixture. It
checks the gmlx forward's conventions against diffusers' own code in fp32.
The real rows need the Z-Image-Turbo GGUF under KQUANT_TEST_GGUF_DIR.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

_ROOT = Path(__file__).resolve().parents[2]
ORACLE = _ROOT / "tests" / "fixtures" / "image" / "oracle"


def _fixtures_module():
    path = _ROOT / "scripts" / "image_oracle" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("_image_oracle_fixtures",
                                                  path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _fixtures_module()


def committed(name: str) -> dict:
    return json.loads((ORACLE / f"{name}.manifest.json").read_text())


def fixture(name: str) -> tuple[dict, dict]:
    """Arrays of a fixture, checked against its committed manifest."""
    if not os.environ.get(fx.ENV):
        pytest.skip(f"set {fx.ENV} to the image oracle fixture directory")
    m = committed(name)
    if not (fx.fixture_root() / m["npz"]).is_file():
        pytest.skip(f"{m['npz']} is not under {fx.ENV}; run "
                    f"{m['script']} to write it")
    return fx.load(name, manifest=m)


def _np(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def check3_rows(model, arrays: dict, rows) -> dict:
    """Relative L2, cosine and max abs of every row against the oracle."""
    act = model.act_dtype
    sigma = float(arrays["sigma"])
    lat = mx.array(arrays["latents"])
    cap = mx.array(arrays["cap_feats"])[None]
    adaln = mx.array(arrays["t_embedder.out"]).astype(act)

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
    for group, (c, s, mod) in {"noise_refiner": (xc, xs, True),
                               "context_refiner": (cc, cs, False),
                               "layers": (cos, sin, True)}.items():
        for i, blk in enumerate(getattr(model, group)):
            src = f"{group}.0.in" if i == 0 else f"{group}.{i - 1}.out"
            name = f"{group}.{i}"
            got[name] = blk(feed(src), c, s, adaln if mod else None)
            ref[name] = arrays[f"{name}.out"]
    last = f"layers.{len(model.layers) - 1}.out"
    got["final_layer"] = model.final_layer(feed(last), adaln)
    ref["final_layer"] = arrays["final_layer.out"]
    out = {}
    for row in rows:
        mx.eval(got[row])
        out[row] = fx.compare(_np(got[row]), ref[row])
    return out


def free_run(model, arrays: dict, blocks=()) -> tuple[dict, dict]:
    """One free-running step. Returns the output of each named block and
    the velocity, each measured against the oracle's fp32 run. The block
    rows see the plumbing between blocks, which teacher forcing skips."""
    taps: dict = {}
    v = model(mx.array(arrays["latents"]), float(arrays["sigma"]),
              mx.array(arrays["cap_feats"])[None],
              taps=taps if blocks else None)
    mx.eval(v, *(taps[b] for b in blocks))
    free = {b: fx.compare(_np(taps[b]), arrays[f"{b}.out"]) for b in blocks}
    return free, fx.compare(np.array(v), arrays["velocity"])


def velocity(model, arrays: dict) -> dict:
    return free_run(model, arrays)[1]


def gate(m: dict, rows: dict | None, vel: dict,
         free: dict | None = None) -> list[str]:
    """Rows, free-running blocks and velocity measures that miss the
    manifest's tolerances."""
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
        rows = check3_rows(model, arrays, m["inputs"]["rows"])
        # torch's fp32 exp misrounds some sinusoid frequencies by one ulp,
        # and t up to 1000 turns that into about 2e-6 in the t_embedder
        # row, which the random tiny blocks amplify to 1e-5. The free run
        # takes the oracle's embedding; the t_embedder row checks gmlx's.
        ad = mx.array(arrays["t_embedder.out"])
        monkeypatch.setattr(model, "adaln", lambda sigma, b: ad)
        free, vel = free_run(model, arrays, m["inputs"]["free_rows"])
    assert np.array_equal(t, arrays["t_embedder.in"])
    assert len(free) == len(model.layers) + 2 * len(model.noise_refiner) + 1
    for measured in (rows, free):
        worst = max(measured.items(), key=lambda kv: kv[1]["rel_l2"])
        assert worst[1]["rel_l2"] < 1e-5, worst
    assert vel["rel_l2"] < 1e-5, vel


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
    arrays, m = fixture("check3_z_image_turbo_512")
    t = turbo.timesteps(float(arrays["sigma"]), 1)
    assert np.array_equal(t, arrays["t_embedder.in"])
    rows = check3_rows(turbo, arrays, m["inputs"]["rows"])
    free, vel = free_run(turbo, arrays, m["inputs"]["free_rows"])
    misses = gate(m, rows, vel, free)
    assert not misses, misses


@pytest.mark.integration
def test_check3_final_velocity_at_1024(turbo):
    arrays, m = fixture("check3_z_image_turbo_1024")
    misses = gate(m, None, velocity(turbo, arrays))
    assert not misses, misses
