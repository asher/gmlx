"""Check 3 of the image acceptance gate for Krea 2: one denoise step of the
gmlx DiT, row by row, against the diffusers oracle on the same GGUF.

scripts/image_oracle/check3_krea2.py writes the fixtures: every row of a
step at 512 and the velocity at 1024, in fp32, with the oracle's own
bf16 against fp32 gap of each. The product gate runs the model as it
ships, in bf16 with the kernel routes at their default, and holds each
row to twice the oracle's gap: teacher-forced for each block and the
final layer, free running for the rest. The exact gate runs the model in
the oracle's fp32 arithmetic, every K-quant product dense on its
dequantized weight and every float tensor of the file in float32, and
holds each teacher-forced row to EXACT_TOL and each free-running one to
``exact_free_gate``.

The rows need the Krea 2 GGUF under KQUANT_TEST_GGUF_DIR and the
fixtures under GMLX_IMAGE_FIXTURES. They skip until both are there.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import krea2_gates as kg
from block_gates import (EXACT_TOL, assert_all_dense, exact_process,
                         exact_products, rerun_exact)
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image import loader as image_loader

ROWS = "check3_krea2_512"
VELOCITY = "check3_krea2_1024"
# Free-running rows in the exact process before the first block: the
# sinusoid of the timestep differs from torch's in the last bit.
EXACT_FREE_TOL = 2e-4
# A free-running block row carries the fp32 rounding of every block before
# it, and Krea 2 grows a difference of rounding block by block: the
# oracle's own bf16 against fp32 gap of the free-running blocks goes from
# 3.4e-2 at block 0 to 0.55 at block 26. On the 512 fixture the exact
# process reads 5.4e-4 to 3.3e-3 of that gap at every block, 2.3e-3 of
# the velocity's, and 2.3e-5 to 1.8e-3 in absolute terms. Its gate is
# FREE_SHARE of the oracle's own gap of the row, three times the largest
# right reading.
FREE_SHARE = 1e-2


def exact_free_gate(name: str, m: dict) -> float:
    """The exact gate of a free-running measure ``name`` of manifest
    ``m``."""
    r = m["results"]
    if name.startswith("free blocks."):
        return FREE_SHARE * r["free_bf16_vs_fp32"][name[len("free "):]]
    if name in ("free last", "velocity"):
        return FREE_SHARE * r["velocity_bf16_vs_fp32"]["rel_l2"]
    return EXACT_FREE_TOL


def _manifest(name: str) -> dict:
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run "
                    "scripts/image_oracle/check3_krea2.py on the Krea 2 "
                    "GGUF and commit the manifest it writes")
    return committed(name)


@pytest.fixture(scope="module")
def real(gguf_dir):
    m = _manifest(ROWS)
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, "krea2"):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    model, _ = image_loader.load_dit(p, log=lambda *_: None)
    return model, p, list(model.blocks)


def _depth(real, m: dict):
    """The model with the blocks the fixture holds."""
    model, _, blocks = real
    n = m["inputs"]["blocks"]
    assert n <= len(blocks)
    model.blocks = blocks[:n]
    return model


def exact_floats(model, path: str):
    """Every float tensor of the file into the model in float32, where the
    load keeps the activation dtype."""
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


@pytest.mark.integration
def test_check3_rows_at_512(real):
    arrays, m = fixture(_manifest(ROWS)["name"])
    model = _depth(real, m)
    got = kg.measures(model, arrays)
    out = kg.missed(got, m["tolerances"])
    ratio = {k: v / m["tolerances"][k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"krea2 check3 512: {len(ratio)} rows, worst {got[top]:.2e} at "
          f"{top}, {ratio[top]:.2f} of its gate; velocity "
          f"{got['velocity']:.2e} of gate "
          f"{m['tolerances']['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.integration
def test_check3_velocity_at_1024(real):
    arrays, m = fixture(_manifest(VELOCITY)["name"])
    model = _depth(real, m)
    got = kg.free_rows(model, arrays)["velocity"]
    rel = kg.rel(got, arrays["velocity"])
    cos = fx.cosine(got, arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"krea2 check3 1024: velocity {rel:.2e} of gate "
          f"{tol['rel_l2']:.2e}, cosine {cos:.6f} floor "
          f"{tol['cosine_min']:.6f}")
    assert rel <= tol["rel_l2"] and cos >= tol["cosine_min"]


@pytest.mark.integration
def test_check3_exact_gate_at_512(real, request):
    arrays, m = fixture(_manifest(ROWS)["name"])
    if not exact_process():
        return rerun_exact(request, "krea2 check3 exact")
    model = _depth(real, m)
    n = exact_floats(model, real[1])
    with exact_products(model) as used:
        got = kg.measures(model, arrays)
        # The reference never reads the final layer's two extra matrices.
        if model.cfg.last_mlp:
            used.update({id(model.last.up), id(model.last.down)})
        assert_all_dense(model, used)
    forced = {k: v for k, v in got.items()
              if k.startswith("blocks.") or k == "last"}
    free = {k: v for k, v in got.items() if k not in forced}
    share = {k: v / exact_free_gate(k, m) for k, v in free.items()}
    hi_f = max(forced, key=forced.get)
    hi = max(share, key=share.get)
    print(f"krea2 check3 exact 512: {n} float tensors in fp32, "
          f"teacher-forced rows within {forced[hi_f]:.2e} at {hi_f} of "
          f"gate {EXACT_TOL:.0e}; free rows at most {share[hi]:.2f} of "
          f"their gate, {free[hi]:.2e} at {hi}")
    assert forced[hi_f] <= EXACT_TOL, forced
    assert share[hi] <= 1.0, {k: v for k, v in share.items() if v > 1.0}
