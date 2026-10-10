"""Check 2 of the image acceptance gate for Krea 2: the text encoder tap.

gmlx encodes the oracle's prompts with the Qwen3-VL-4B text GGUF the
oracle read. The token ids must equal the ids the pipeline keeps, with
the template's first 34 ids dropped from the features, and the stacked
features of each prompt must stay within the manifest's relative L2 and
per-token cosine of the oracle's fp32 features. The pipeline pads each
prompt in the middle of its template and masks the pads, with positions
that count real tokens alone, which is the unpadded encoding that gmlx
runs. Each wrong encoder below must fail some reading.

Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder GGUF the fixture
names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_krea2.py writes it."""
from __future__ import annotations

import dataclasses
import os

import mlx.core as mx
import numpy as np
import pytest
from oracle_fixture import fixture, fx

pytestmark = pytest.mark.integration

NAME = "check2_krea2"


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

    spec = te.encoder_spec("krea2")
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
    assert len(handle.model.model.layers) == spec.layers == 36
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


def test_token_ids_equal_the_ids_the_pipeline_keeps(handle, check2):
    arrays, m = check2
    spec = handle.spec
    prefix = spec.template.partition("{}")[0]
    system = handle.tokenizer.encode(prefix, add_special_tokens=False)
    assert len(system) == spec.drop == m["inputs"]["drop_index"]
    for key, prompt in m["inputs"]["prompts"].items():
        ids = handle.token_ids(prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert ids[:spec.drop] == system, key
        assert len(ids) - spec.drop == m["inputs"]["rows"][key]


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
        tap = handle.encode(prompt)
        got = np.array(tap.features.astype(mx.float32))
        ref = arrays[f"features_{key}"]
        assert tap.features.dtype == mx.float32
        r = _reading(got, ref, tol, key)
        print(f"[check2] krea2 {key}: {r['rows']} rows, rel L2 "
              f"{r.get('rel_l2', float('nan')):.3e} of gate "
              f"{r['tol_rel_l2']:.3e}, min token cosine "
              f"{r.get('token_cosine_min', float('nan')):.6f} floor "
              f"{r['tol_cosine']:.6f}")
        assert not _caught(r, [], []), (key, r)


def _features(handle, spec, prompt: str, *, layers=None, drop=None,
              stack=True):
    from gmlx.image import text_encoder as te

    ids = te.token_ids(spec, handle.tokenizer, prompt)
    hs = te.tap(handle.model, ids, layers=layers or spec.tap_layers,
                activation=spec.activation)
    d = spec.drop if drop is None else drop
    rows = [h[0, d:] for h in hs]
    f = mx.concatenate(rows, axis=-1) if stack else rows[0]
    return ids, np.array(f.astype(mx.float32))


WRONG = {
    "every tap one layer early": (
        lambda h, s, p: _features(h, s, p, layers=tuple(
            n - 1 for n in s.tap_layers)), {"relative L2", "token cosine"}),
    "the system rows kept": (
        lambda h, s, p: _features(h, s, p, drop=0), {"shape"}),
    "the first tap alone": (
        lambda h, s, p: _features(h, s, p, stack=False),
        {"shape"}),
    "the taps in reverse": (
        lambda h, s, p: _features(h, s, p,
                                  layers=tuple(reversed(s.tap_layers))),
        {"relative L2", "token cosine"}),
    "the suffix cut with the prompt": (
        lambda h, s, p: _features(h, dataclasses.replace(s, keep_tail=0),
                                  p), {"token ids"}),
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    arrays, m = check2
    make, kinds = WRONG[name]
    missed = {}
    prompts = m["inputs"]["prompts"]
    keys = ["long"] if name == "the suffix cut with the prompt" else prompts
    for key in keys:
        ids, got = make(handle, handle.spec, prompts[key])
        r = _reading(got, arrays[f"features_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
