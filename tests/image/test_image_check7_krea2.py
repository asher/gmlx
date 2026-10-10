"""Check 7 of the image acceptance gate for Krea 2, step half: check 3's
step at 512 with a Comfy-Org Krea 2 LoRA applied live under a patch
table, against the diffusers oracle with the same adapter applied as
PEFT applies it. scripts/image_oracle/check7_krea2_lora.py writes the
fixture, which has the layout of a check 3 fixture and its rule: each
row within twice the oracle's own bf16 against fp32 gap with the adapter
on, teacher-forced for the blocks and the final layer, free running for
the rest.

The rows see the adapter and its strength: the base model, and the
adapter at strength 0.8, miss the gate. The velocity of the request path,
api.denoise_step with the kernel routes at their default, passes its
gate too. The delta half runs on the CPU in test_image_krea2_lora_map.py.

Skipped unless the manifest is committed, GMLX_IMAGE_FIXTURES holds its
arrays, KQUANT_TEST_GGUF_DIR the DiT GGUF and GMLX_TEST_LORA_DIR the
adapter the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import krea2_gates as kg
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

pytestmark = pytest.mark.integration

NAME = "check7_krea2_lora_512"
LORA_ENV = "GMLX_TEST_LORA_DIR"


def _manifest() -> dict:
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check7_krea2_lora.py and commit "
                    "the manifest it writes")
    return committed(NAME)


def _identified(p, want: dict) -> bool:
    if Path(p).name != want["name"] or os.path.getsize(p) != want["size"]:
        return False
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return True


@pytest.fixture(scope="module")
def check7():
    _manifest()
    return fixture(NAME)


@pytest.fixture(scope="module")
def lora_path(check7):
    want = check7[1]["files"]["lora"]
    root = os.environ.get(LORA_ENV)
    if not root:
        pytest.skip(f"set {LORA_ENV} to the folder of {want['name']}")
    for p in sorted(Path(root).expanduser().rglob(want["name"])):
        if _identified(p, want):
            return str(p)
    pytest.skip(f"{want['name']} is not under {LORA_ENV}")


@pytest.fixture(scope="module")
def pipe(check7, gguf_dir):
    from gmlx.image import api

    want = check7[1]["files"]["dit"]
    for p in image_dits(gguf_dir, "krea2"):
        if _identified(p, want):
            return api.load(p, log=lambda *_: None)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


def _table(pipe, path: str, strength: float) -> str:
    from gmlx.image import api, lora

    ad = lora.load_adapter(pipe, path)
    assert ad.text == ()
    return api.define_patch_table(pipe, lora.table_entries([(ad,
                                                             strength)]))


def _rows(pipe, arrays: dict, tid: str | None) -> dict:
    from gmlx.image.lora import install

    prev = install.activate(pipe, tid) if tid is not None else None
    try:
        return kg.measures(pipe.dit, arrays)
    finally:
        if tid is not None:
            install.restore(pipe, prev)


def test_check7_adapter_step_rows_at_512(check7, pipe, lora_path):
    arrays, m = check7
    tol = m["tolerances"]
    ad_n = m["inputs"]["lora_pairs"]
    from gmlx.image import lora

    assert len(lora.load_adapter(pipe, lora_path).dit) == ad_n
    got = _rows(pipe, arrays, _table(pipe, lora_path,
                                     m["inputs"]["strength"]))
    out = kg.missed(got, tol)
    ratio = {k: v / tol[k]["rel_l2"] for k, v in got.items()
             if not k.startswith("free ")}
    top = max(ratio, key=ratio.get)
    print(f"krea2 check7 512: {ad_n} pairs, {len(ratio)} rows, worst "
          f"{got[top]:.2e} at {top}, {ratio[top]:.2f} of its gate; "
          f"velocity {got['velocity']:.2e} of gate "
          f"{tol['velocity']['rel_l2']:.2e}")
    assert not out, out


@pytest.mark.parametrize("wrong", ["base", "strength 0.8"])
def test_check7_rows_see_the_adapter_and_its_strength(check7, pipe,
                                                      lora_path, wrong):
    arrays, m = check7
    tid = None if wrong == "base" else _table(
        pipe, lora_path, 0.8 * m["inputs"]["strength"])
    got = _rows(pipe, arrays, tid)
    out = kg.missed(got, m["tolerances"])
    worst = max(out, key=lambda k: out[k][0] / out[k][1]) if out else None
    print(f"krea2 check7 {wrong}: {len(out)} rows miss"
          + (f", worst {worst} at {out[worst][0] / out[worst][1]:.1f} of "
             "its gate" if worst else ""))
    assert out


def test_check7_velocity_of_the_request_path(check7, pipe, lora_path):
    from gmlx.image import api

    arrays, m = check7
    tid = _table(pipe, lora_path, m["inputs"]["strength"])
    v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                         float(arrays["sigma"]),
                         api.Conditioning(mx.array(arrays["features"])),
                         patch_table=tid)
    got = fx.compare(np.array(v), arrays["velocity"])
    tol = m["tolerances"]["velocity"]
    print(f"krea2 check7 request velocity: rel L2 {got['rel_l2']:.3e} of "
          f"{tol['rel_l2']:.3e}, cosine {got['cosine']:.6f} of "
          f"{tol['cosine_min']:.6f}")
    assert got["rel_l2"] <= tol["rel_l2"]
    assert got["cosine"] >= tol["cosine_min"]
