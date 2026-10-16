"""Check 2 of the image acceptance gate for FLUX.3 klein: the text encoder
tap.

gmlx encodes the oracle's prompts with the Qwen3-VL-4B text GGUF the
oracle read. The reference right-pads each templated prompt with the pad
token to a multiple of 80 rows, at most 4096, masks the pads as keys and
gives every pad row position 1, and keeps the pad rows. The token ids and
pad rows gmlx forms must equal the ids the reference feeds its model, its
text ids must equal the reference's, and the stacked features of every
row must stay within the manifest's relative L2 and per-token cosine of
the oracle's fp32 features. Each wrong encoder below must fail some
reading.

Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder GGUF the fixture
names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_flux3_klein.py writes it."""
from __future__ import annotations

import dataclasses
import os

import mlx.core as mx
import numpy as np
import pytest
from oracle_fixture import fixture, fx

pytestmark = pytest.mark.integration

NAME = "check2_flux3_klein"


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

    spec = te.encoder_spec("flux3_klein")
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


def _padded(spec, ids: list) -> list:
    from gmlx.image import text_encoder as te

    return list(ids) + [spec.pad_id] * te.pad_rows(spec, len(ids))


def test_the_encoder_is_the_identified_one(handle):
    from gmlx.image import text_encoder as te

    spec = handle.spec
    assert len(handle.model.model.layers) == spec.layers == 36
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


def test_the_ids_and_pads_equal_what_the_reference_feeds(handle, check2):
    """The ids and the mask of each prompt, the prompt past the token cap
    included, and the positions and text ids of every row."""
    from gmlx.image.families.flux3_klein import model as fm

    arrays, m = check2
    spec = handle.spec
    prompts = {**m["inputs"]["prompts"], **m["inputs"]["ids_only"]}
    for key, prompt in prompts.items():
        ids = handle.token_ids(prompt)
        want = arrays[f"ids_{key}"].tolist()
        assert _padded(spec, ids) == want, key
        mask = arrays[f"mask_{key}"]
        assert mask.tolist() == [1] * len(ids) + [0] * (len(want) - len(ids))
        if key in m["inputs"]["ids_only"]:
            assert len(want) == spec.max_tokens
            continue
        pos = arrays[f"positions_{key}"]
        assert pos.shape == (4, 1, len(want))
        assert (pos == (list(range(len(ids))) + [spec.pad_position]
                        * (len(want) - len(ids)))).all(), key
        assert np.array_equal(fm.text_ids(len(want)),
                              arrays[f"text_ids_{key}"]), key
    rows = m["inputs"]["rows"]
    assert rows["eighty"]["real"] == rows["eighty"]["rows"] == 80
    assert rows["long"]["rows"] > 80 and rows["empty"]["rows"] == 80


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
        assert len(tap.ids) + tap.pads == ref.shape[0]
        r = _reading(got, ref, tol, key)
        pads = ""
        if tap.pads:
            n = len(tap.ids)
            pads = f", pad rows rel L2 {fx.rel_l2(got[n:], ref[n:]):.3e}"
        print(f"[check2] flux3 {key}: {r['rows']} rows ({tap.pads} pads), "
              f"rel L2 {r.get('rel_l2', float('nan')):.3e} of gate "
              f"{r['tol_rel_l2']:.3e}, min token cosine "
              f"{r.get('token_cosine_min', float('nan')):.6f} floor "
              f"{r['tol_cosine']:.6f}{pads}")
        assert not _caught(r, [], []), (key, r)


def _features(handle, spec, prompt: str, *, layers=None, pads=True,
              causal_pads=False, stack=True):
    """The features of a prompt with one convention changed. Returns the
    ids fed the model and the features."""
    from gmlx.image import text_encoder as te

    ids = te.token_ids(spec, handle.tokenizer, prompt)
    n = te.pad_rows(spec, len(ids)) if pads else 0
    layers = layers or spec.tap_layers
    if causal_pads:
        hs = te.tap(handle.model, ids + [spec.pad_id] * n, layers=layers,
                    activation=spec.activation)
    else:
        hs = te.tap(handle.model, ids, layers=layers,
                    activation=spec.activation, pad=n, pad_id=spec.pad_id,
                    pad_position=spec.pad_position)
    rows = [h[0] for h in hs]
    f = mx.concatenate(rows, axis=-1) if stack else rows[0]
    return ids + [spec.pad_id] * n, np.array(f.astype(mx.float32))


WRONG = {
    "every tap one layer early": (
        lambda h, s, p: _features(h, s, p, layers=tuple(
            n - 1 for n in s.tap_layers)), {"relative L2", "token cosine"}),
    "the taps in reverse": (
        lambda h, s, p: _features(h, s, p,
                                  layers=tuple(reversed(s.tap_layers))),
        {"relative L2", "token cosine"}),
    "the first tap alone": (
        lambda h, s, p: _features(h, s, p, stack=False), {"shape"}),
    "the pad rows dropped": (
        lambda h, s, p: _features(h, s, p, pads=False),
        {"shape", "token ids"}),
    "the pads as causal tokens": (
        lambda h, s, p: _features(h, s, p, causal_pads=True),
        {"relative L2", "token cosine"}),
    "the pads at position 0": (
        lambda h, s, p: _features(h, dataclasses.replace(s, pad_position=0),
                                  p), {"relative L2", "token cosine"}),
    "a token cap of 512": (
        lambda h, s, p: _features(h, dataclasses.replace(s, max_tokens=512),
                                  p), {"token ids", "shape"}),
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    arrays, m = check2
    make, kinds = WRONG[name]
    missed = {}
    prompts = m["inputs"]["prompts"]
    # The prompts that have pads, or are past a cap of 512.
    keys = {"the pad rows dropped": ["plain", "long"],
            "the pads as causal tokens": ["plain", "long"],
            "the pads at position 0": ["plain", "long"],
            "a token cap of 512": ["long"]}.get(name, list(prompts))
    for key in keys:
        ids, got = make(handle, handle.spec, prompts[key])
        r = _reading(got, arrays[f"features_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
