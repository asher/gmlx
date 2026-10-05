"""Check 7 of the image acceptance gate for Z-Image, step half: one
denoise step with a LoRA applied live, against the diffusers oracle with
the delta merged into dequantized dense weights.

The oracle's adapter targets the diffusers layout, where q, k and v are
separate weights, so its merge places no band. The test writes the
oracle's pairs in each dialect's key spelling, loads the file into a
pipeline over the Z-Image-Turbo GGUF, and runs check 3's step at 512 under
the patch table. A variant of the fixture fixes the alpha and strength of
the dialects that carry it. The tolerance is the manifest's: twice the
oracle's own bf16 against fp32 gap with the adapter merged, check 3's rule.
scripts/image_oracle/check7_z_image_lora.py writes the fixture.

The delta half of check 7 runs on the CPU in test_image_lora_map.py. The
rows here need the GGUF under KQUANT_TEST_GGUF_DIR and the arrays under
GMLX_IMAGE_FIXTURES.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from lora_fixtures import DIALECTS, write_adapter
from oracle_fixture import committed, fixture, fx

NAME = "check7_z_image_turbo_lora_512"


def turbo_pipe(gguf_index, name: str):
    """A pipeline over the DiT GGUF that the fixture ``name`` was made
    from, or a skip without it."""
    if not (Path(__file__).resolve().parents[1] / "fixtures" / "image"
            / "oracle" / f"{name}.manifest.json").is_file():
        pytest.skip(f"no committed manifest for {name}")
    want = committed(name)["files"]["dit"]
    for p in gguf_index.get("lumina2", []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    from gmlx.image import api
    from gmlx.image.loader import load_dit
    from gmlx.image.registry import get_family

    model, _ = load_dit(p, log=lambda *_: None)
    return api.ImagePipeline(dit=model, family=get_family("z_image"),
                             variant="turbo")


@pytest.fixture(scope="module")
def pipe(gguf_index):
    return turbo_pipe(gguf_index, NAME)


def step(pipe, arrays, table=None) -> np.ndarray:
    from gmlx.image import api

    v = api.denoise_step(pipe, mx.array(arrays["latents"]),
                         float(arrays["sigma"]),
                         api.Conditioning(mx.array(arrays["cap_feats"])),
                         patch_table=table)
    return np.array(v)


@pytest.mark.integration
@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_check7_one_adapter_step_at_512(pipe, tmp_path, dialect):
    from gmlx.image import api, lora

    arrays, m = fixture(NAME)
    variants = m["inputs"]["variants"]
    variant = next(v for v, d in variants.items() if dialect in d["dialects"])
    alpha, strength = variants[variant]["alpha"], variants[variant]["strength"]
    assert (alpha is not None) == DIALECTS[dialect][4]
    adapter = {t: (arrays[f"down.{t}"], arrays[f"up.{t}"], alpha)
               for t in m["inputs"]["targets"]}
    ad = lora.load_adapter(pipe, write_adapter(
        tmp_path / f"{dialect}.safetensors", dialect, adapter))
    assert len(ad.dit) == len(adapter) and ad.text == ()
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, strength)]))
    ref = arrays[f"velocity.{variant}"]
    tol = m["tolerances"][variant]
    got = fx.compare(step(pipe, arrays, tid), ref)
    print(f"check7 step {dialect} ({variant}): rel L2 {got['rel_l2']:.4e} "
          f"of {tol['velocity_rel_l2']:.4e}, cosine {got['cosine']:.6f} "
          f"of {tol['velocity_cosine_min']:.6f}")
    assert got["rel_l2"] <= tol["velocity_rel_l2"], got
    assert got["cosine"] >= tol["velocity_cosine_min"], got
    # The base model, and the adapter without its alpha, fail the gate, so
    # it sees the adapter and its alpha.
    base = fx.compare(step(pipe, arrays), ref)
    assert base["rel_l2"] > tol["velocity_rel_l2"], base
    if alpha is not None:
        bare = {t: (d, u, None) for t, (d, u, _) in adapter.items()}
        other = lora.load_adapter(pipe, write_adapter(
            tmp_path / "bare.safetensors", "diffusers", bare))
        wrong = fx.compare(step(pipe, arrays, api.define_patch_table(
            pipe, lora.table_entries([(other, strength)]))), ref)
        assert wrong["rel_l2"] > tol["velocity_rel_l2"], wrong
        print(f"check7 step {dialect}: base {base['rel_l2']:.4e}, without "
              f"alpha {wrong['rel_l2']:.4e}")
