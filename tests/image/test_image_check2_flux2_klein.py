"""Check 2 of the image acceptance gate for FLUX.2 klein: the text encoder
tap.

gmlx encodes the oracle's prompts with the Qwen3-4B GGUF the oracle read.
The reference renders each prompt with thinking off, right-pads it with
the pad token to 512 rows and cuts it there, masks the pads as keys,
gives each row its index as its position, and keeps the pad rows. The
token ids and pad rows gmlx forms must equal the ids the reference feeds
its model. The stacked features must stay within the manifest's relative
L2 of the oracle's fp32 features over all rows, over the real rows and
over the pad rows, and each real row within its per-token cosine floor.
The pad rows take no cosine floor: the oracle's own bf16 run moves single
pad rows to a cosine near 0.5. Each wrong encoder below must fail some
reading.

Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder GGUF the fixture
names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_flux2_klein.py writes it."""
from __future__ import annotations

import dataclasses
import os

import mlx.core as mx
import numpy as np
import pytest
from oracle_fixture import fixture, fx

pytestmark = pytest.mark.integration

NAME = "check2_flux2_klein"


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

    spec = te.encoder_spec("flux2_klein")
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


def test_the_ids_pads_and_positions_equal_what_the_reference_feeds(
        handle, check2):
    """The rendered text, the ids and the mask of each prompt, the prompt
    past the token cap included, and the position of every row."""
    from gmlx.image import text_encoder as te

    arrays, m = check2
    spec = handle.spec
    ins = m["inputs"]
    for key, prompt in ins["prompts"].items():
        assert te.render(spec, prompt) == ins["rendered"][key], key
        ids = handle.token_ids(prompt)
        want = arrays[f"ids_{key}"].tolist()
        assert _padded(spec, ids) == want, key
        assert len(want) == spec.max_tokens
        mask = arrays[f"mask_{key}"]
        assert mask.tolist() == [1] * len(ids) + [0] * (len(want) - len(ids))
        pos = arrays[f"positions_{key}"]
        assert pos.shape == (1, len(want))
        assert (pos[0] == np.arange(len(want))).all(), key
    assert all(m["results"]["positions_from_the_model"].values())
    rows = ins["rows"]
    assert rows["long"]["real"] == rows["long"]["rows"] == 512
    assert rows["one_pad"]["real"] == 511
    assert all(r["rows"] == 512 for r in rows.values())


def _reading(got: np.ndarray, ref: np.ndarray, tol: dict, key: str,
             real: int) -> dict:
    """The relative L2 over all rows, the real rows and the pad rows, and
    the smallest per-token cosine of the real rows, with their gates.
    ``real`` is the reference's count of real rows."""
    out = {"rows": int(got.shape[0]), "shape": got.shape,
           "want_shape": ref.shape,
           "tol_rel_l2": tol["rel_l2"][key],
           "tol_rel_l2_real": tol["rel_l2_real"][key],
           "tol_rel_l2_pads": tol["rel_l2_pads"].get(key),
           "tol_cosine": tol["token_cosine_min"][key]}
    if got.shape == ref.shape:
        cos = token_cosine(got[:real], ref[:real])
        out.update(rel_l2=fx.rel_l2(got, ref),
                   rel_l2_real=fx.rel_l2(got[:real], ref[:real]),
                   token_cosine_min=float(cos.min()),
                   worst_token=int(cos.argmin()))
        if real < ref.shape[0]:
            out["rel_l2_pads"] = fx.rel_l2(got[real:], ref[real:])
    return out


def _caught(r: dict, ids, want_ids) -> list:
    by = []
    if list(ids) != list(want_ids):
        by.append("token ids")
    if r["shape"] != r["want_shape"]:
        by.append("shape")
    if "rel_l2" not in r:
        return by
    if not (r["rel_l2"] <= r["tol_rel_l2"]
            and r["rel_l2_real"] <= r["tol_rel_l2_real"]):
        by.append("relative L2")
    if "rel_l2_pads" in r and not r["rel_l2_pads"] <= r["tol_rel_l2_pads"]:
        by.append("pad rows")
    if not r["token_cosine_min"] >= r["tol_cosine"]:
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
        assert len(tap.ids) + tap.pads == ref.shape[0] == 512
        r = _reading(got, ref, tol, key, m["inputs"]["rows"][key]["real"])
        pads = ""
        if "rel_l2_pads" in r:
            pads = (f", pad rows {r['rel_l2_pads']:.3e} of gate "
                    f"{r['tol_rel_l2_pads']:.3e}")
        print(f"[check2] flux2 {key}: {r['rows']} rows ({tap.pads} pads), "
              f"rel L2 {r['rel_l2']:.3e} of gate {r['tol_rel_l2']:.3e}, "
              f"real rows {r['rel_l2_real']:.3e} of gate "
              f"{r['tol_rel_l2_real']:.3e}, min real token cosine "
              f"{r['token_cosine_min']:.6f} floor {r['tol_cosine']:.6f}"
              f"{pads}")
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
        {"pad rows"}),
    "the pads at position 1, as FLUX.3 klein's": (
        lambda h, s, p: _features(h, dataclasses.replace(s, pad_position=1),
                                  p), {"pad rows"}),
    "the template with thinking on": (
        lambda h, s, p: _features(h, dataclasses.replace(
            s, template="<|im_start|>user\n{}<|im_end|>\n"
                        "<|im_start|>assistant\n"), p), {"token ids"}),
    "a token cap of 256": (
        lambda h, s, p: _features(h, dataclasses.replace(
            s, max_tokens=256, pad_multiple=256), p),
        {"token ids", "shape"}),
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_").replace(",", "")
                              .replace("'", "").replace(".", "")
                              for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    arrays, m = check2
    make, kinds = WRONG[name]
    missed = {}
    prompts = m["inputs"]["prompts"]
    # The prompts with more than one pad row, past a cap of 256, or with
    # the end of the template inside the cap.
    pads = ["plain", "empty", "non_ascii"]
    keys = {"the pad rows dropped": pads, "the pads as causal tokens": pads,
            "the pads at position 1, as FLUX.3 klein's": pads,
            "a token cap of 256": ["long", "one_pad"],
            "the template with thinking on": [*pads, "one_pad"]}.get(
                name, list(prompts))
    for key in keys:
        ids, got = make(handle, handle.spec, prompts[key])
        r = _reading(got, arrays[f"features_{key}"], m["tolerances"], key,
                     m["inputs"]["rows"][key]["real"])
        print(f"[check2] flux2 {name}, {key}: caught by "
              f"{_caught(r, ids, arrays[f'ids_{key}'].tolist())}, "
              + ", ".join(f"{k} {r[k]:.3e} of {r['tol_' + k]:.3e}"
                          for k in ("rel_l2", "rel_l2_real", "rel_l2_pads")
                          if k in r))
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
