"""Check 2 of the image acceptance gate for MOSS-Image: the Qwen3.5-9B text
encoder read after its final norm.

gmlx encodes the oracle's prompts with the text GGUF the oracle read,
for the Q8_0 and the Q4_K_M files of unsloth/Qwen3.5-9B-GGUF. The token
ids must equal the pipeline's, and the features of each prompt must stay
within the manifest's relative L2 and per-token cosine of the oracle's
fp32 features. The pipeline runs each prompt alone at its own length,
which is the encoding gmlx runs. Each wrong encoder below must fail some
reading.

Skipped until the fixture's manifest is committed, and unless
KQUANT_TEST_GGUF_DIR holds the encoder GGUF the fixture names and
GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_moss_image.py writes it."""
from __future__ import annotations

import os

import mlx.core as mx
import numpy as np
import pytest
from oracle_fixture import ORACLE, fixture, fx

pytestmark = pytest.mark.integration

NAMES = ("check2_moss_image_q8_0", "check2_moss_image_q4_k_m")


def token_cosine(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    return (g * r).sum(-1) / (np.linalg.norm(g, axis=-1)
                              * np.linalg.norm(r, axis=-1))


@pytest.fixture(scope="module", params=NAMES)
def check2(request):
    name = request.param
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed")
    return fixture(name)


@pytest.fixture(scope="module")
def handle(gguf_index, check2):
    from gmlx.image import text_encoder as te

    spec = te.encoder_spec("moss_image")
    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(spec.arch, []):
        if (os.path.basename(p) == want["name"]
                and os.path.getsize(p) == want["size"]
                and fx.file_identity(p)["sha256"] == want["sha256"]):
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    h = te.load_encoder(p, spec, log=lambda *_: None)
    yield h
    h.release()


def test_the_encoder_is_the_identified_one(handle):
    from gmlx.image import text_encoder as te

    spec = handle.spec
    assert len(handle.model.model.layers) == spec.layers == 32
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


def test_token_ids_equal_the_pipelines(handle, check2):
    arrays, m = check2
    for key, prompt in m["inputs"]["prompts"].items():
        ids = handle.token_ids(prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert len(ids) == m["inputs"]["rows"][key]


def _reading(got: np.ndarray, ref: np.ndarray, tol: dict, key: str) -> dict:
    out = {"rows": int(got.shape[0]), "shape": got.shape,
           "want_shape": ref.shape,
           "tol_rel_l2": tol["rel_l2"][key],
           "tol_cosine": tol["token_cosine_min"][key]}
    if got.shape == ref.shape:
        cos = token_cosine(got, ref)
        out.update(rel_l2=fx.rel_l2(got, ref),
                   token_cosine_min=float(cos.min()),
                   worst_token=int(cos.argmin()))
    return out


def _caught(r: dict, ids, want_ids) -> list:
    by = []
    if list(ids) != list(want_ids):
        by.append("token ids")
    if r["shape"] != r["want_shape"]:
        by.append("shape")
    if "rel_l2" in r and not r["rel_l2"] <= r["tol_rel_l2"]:
        by.append("relative L2")
    if "rel_l2" in r and not r["token_cosine_min"] >= r["tol_cosine"]:
        by.append("token cosine")
    return by


def test_features_are_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    tol = m["tolerances"]
    for key, prompt in m["inputs"]["prompts"].items():
        enc = handle.encode(prompt)
        got = np.array(enc.features.astype(mx.float32))
        r = _reading(got, arrays[f"features_{key}"], tol, key)
        print(f"[check2] {m['name']} {key}: {r['rows']} rows, rel L2 "
              f"{r.get('rel_l2', float('nan')):.3e} of gate "
              f"{r['tol_rel_l2']:.3e}, min token cosine "
              f"{r.get('token_cosine_min', float('nan')):.6f} floor "
              f"{r['tol_cosine']:.6f}")
        assert not _caught(r, [], []), (key, r)


def _layer_read(handle, ids, layer: int, norm: bool) -> np.ndarray:
    """The output of layer ``layer`` (1-based) of the stack, with or
    without the final norm."""
    inner = handle.model.model
    x = mx.array([ids])
    pos = mx.tile(mx.arange(len(ids))[None, None], (3, 1, 1))
    sink: list = []
    inner(x, position_ids=pos, capture_layer_ids=[layer - 1],
          hidden_sink=sink)
    h = inner.norm(sink[0]) if norm else sink[0]
    return np.array(h[0].astype(mx.float32))


def _thinking_off(handle, prompt: str):
    import dataclasses

    from gmlx.image import text_encoder as te

    spec = handle.spec
    off = dataclasses.replace(spec, template=spec.template.replace(
        "<think>\n", "<think>\n\n</think>\n\n"))
    ids = te.token_ids(off, handle.tokenizer, prompt)
    return ids, np.array(te.final_hidden(handle.model, ids)[0].astype(
        mx.float32))


def _ids(handle, prompt):
    return handle.token_ids(prompt)


WRONG = {
    "the read before the final norm": (
        lambda h, p: (_ids(h, p), _layer_read(h, _ids(h, p), 32, False)),
        {"relative L2", "token cosine"}),
    "the final norm one layer early": (
        lambda h, p: (_ids(h, p), _layer_read(h, _ids(h, p), 31, True)),
        {"relative L2", "token cosine"}),
    "thinking off": (_thinking_off, {"token ids", "shape"}),
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    arrays, m = check2
    make, kinds = WRONG[name]
    missed = {}
    for key, prompt in m["inputs"]["prompts"].items():
        ids, got = make(handle, prompt)
        r = _reading(got, arrays[f"features_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
