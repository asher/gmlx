"""Check 2 of the image acceptance gate for Z-Image: the text encoder tap.

gmlx encodes the oracle's prompts with the same Qwen3-4B GGUF. Token ids
and the rendered template must be equal, and each prompt's tap must stay
within the manifest's relative L2 and per-token cosine of the oracle's
fp32 tap. Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder GGUF the
fixture names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_z_image.py writes it."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
from oracle_fixture import fixture, fx

import mlx.core as mx

pytestmark = pytest.mark.integration

NAME = "check2_z_image"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def token_cosine(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    return (g * r).sum(-1) / (np.linalg.norm(g, axis=-1)
                              * np.linalg.norm(r, axis=-1))


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


@pytest.fixture(scope="module")
def handle(gguf_index, check2):
    from gmlx.image import text_encoder as te

    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(te.Z_IMAGE.arch, []):
        if (os.path.basename(p) == want["name"]
                and os.path.getsize(p) == want["size"]
                and _sha256(p) == want["sha256"]):
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    h = te.load_encoder(p, te.Z_IMAGE, log=lambda *_: None)
    yield h
    h.release()


def test_tap_and_template_are_the_oracle_ones(check2):
    from gmlx.image import text_encoder as te

    _, m = check2
    spec = te.Z_IMAGE
    assert m["inputs"]["max_sequence_length"] == spec.max_tokens
    assert spec.layers == 36 and spec.tap_layers == (35,)
    assert "35 of 36" in m["inputs"]["tap"]
    for checks in m["results"]["tap_checks"].values():
        for mode in checks.values():
            assert mode["hook_equals_hidden_states_minus_2"]
            assert mode["last_is_final_norm"]
    for key, prompt in m["inputs"]["prompts"].items():
        assert te.render(spec, prompt) == m["inputs"]["rendered"][key], key


def test_token_ids_equal_the_oracle(handle, check2):
    from gmlx.image import text_encoder as te

    arrays, m = check2
    for key, prompt in m["inputs"]["prompts"].items():
        ids = te.token_ids(te.Z_IMAGE, handle.tokenizer, prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key


def test_tap_is_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    tol = m["tolerances"]
    rows = {}
    for key, prompt in m["inputs"]["prompts"].items():
        (h,) = handle.encode(prompt).hidden
        got = np.array(h[0].astype(mx.float32))
        ref = arrays[f"tap_{key}"]
        assert got.shape == ref.shape, key
        cos = token_cosine(got, ref)
        rows[key] = {"rel_l2": fx.rel_l2(got, ref),
                     "token_cosine_min": float(cos.min()),
                     "worst_token": int(cos.argmin()),
                     "tol_rel_l2": tol["rel_l2"][key]}
        print(f"[check2] {key}: {rows[key]}")
    for key, r in rows.items():
        assert r["rel_l2"] <= r["tol_rel_l2"], (key, r)
        assert r["token_cosine_min"] >= tol["token_cosine_min"], (key, r)
