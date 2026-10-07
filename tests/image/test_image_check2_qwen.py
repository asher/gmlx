"""Check 2 of the image acceptance gate for Qwen-Image 2.1: the text
encoder tap.

gmlx encodes the oracle's prompts with the Qwen3-VL-8B text GGUF the
oracle read. The token ids must equal the oracle's, the dropped rows must
be the system turn, and the features of each prompt must stay within the
manifest's relative L2 and per-token cosine of the oracle's fp32
features.

The manifest's gate is twice the oracle's own bf16 against fp32 gap, 4.5
to 17.7 percent, where the rows of this decoder move 1 to 2 percent under
a change of rounding. One row moves far more: the first kept row, the
<|im_start|> of the user turn, is 21 percent off in the oracle's bf16
arm. The tap of this family runs in float32 for that row, as the family's
encoder row says, and the test that reads the features prints that row
beside the others. The gate alone is loose, so each wrong encoder below
must fail some reading of this file. The readings are the hash of the
token ids, the row count, and the two gates. The ids and the rows hold
what changes the text or the rows kept: a word of the template, thinking
tokens after the assistant marker, system rows left in. The gates hold
what changes the values on the same ids: a tap one layer early, the
final norm applied, pad tokens before the prompt with no mask.

Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder GGUF the fixture
names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_qwen_image21.py writes it."""
from __future__ import annotations

import dataclasses
import hashlib
import os

import numpy as np
import pytest
from oracle_fixture import fixture, fx

import mlx.core as mx

pytestmark = pytest.mark.integration

NAME = "check2_qwen_image21"
PAD = 8                 # pad tokens of the padded encoder


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


def _spec():
    from gmlx.image import text_encoder as te

    return te.encoder_spec("qwen_image21")


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


@pytest.fixture(scope="module")
def handle(gguf_index, check2):
    from gmlx.image import text_encoder as te

    spec = _spec()
    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(spec.arch, []):
        if (os.path.basename(p) == want["name"]
                and os.path.getsize(p) == want["size"]
                and _sha256(p) == want["sha256"]):
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    h = te.load_encoder(p, spec, log=lambda *_: None)
    yield h
    h.release()


def _ids_digest(ids) -> str:
    """The digest the manifest holds of a prompt's token ids."""
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def test_the_encoder_is_the_identified_one_without_its_output_matrix(handle):
    from gmlx.image import text_encoder as te

    spec = handle.spec
    assert handle.owned and "lm_head" not in handle.model
    assert len(handle.model.model.layers) == spec.layers == 36
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


def test_token_ids_and_dropped_rows_equal_the_oracle(handle, check2):
    arrays, m = check2
    spec, tok = handle.spec, handle.tokenizer
    from gmlx.image.families.qwen_image21 import family as qf

    system = tok.encode(qf.PROMPT.values["system_turn"],
                        add_special_tokens=False)
    assert len(system) == spec.drop == m["inputs"]["drop_index"]
    for key, prompt in m["inputs"]["prompts"].items():
        ids = handle.token_ids(prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert _ids_digest(ids) == m["tensors"][f"ids_{key}"]["sha256"]
        assert ids[:spec.drop] == list(system), key
        assert len(ids) - spec.drop == m["inputs"]["rows"][key]


def _reading(got: np.ndarray, ref: np.ndarray, tol: dict, key: str) -> dict:
    """The two gated readings of one prompt and their gates. Features
    with another row count than the oracle's have no reading."""
    out = {"rows": int(got.shape[0]), "want_rows": int(ref.shape[0]),
           "tol_rel_l2": tol["rel_l2"][key],
           "tol_cosine": tol["token_cosine_min"][key]}
    if got.shape == ref.shape:
        cos = token_cosine(got, ref)
        out.update(rel_l2=fx.rel_l2(got, ref),
                   token_cosine_min=float(cos.min()),
                   worst_token=int(cos.argmin()),
                   first_row=fx.rel_l2(got[:1], ref[:1]),
                   other_rows=(fx.rel_l2(got[1:], ref[1:])
                               if len(got) > 1 else 0.0))
    return out


def _caught(r: dict, ids, want_ids) -> list:
    """The readings of this file that a wrong encoder fails."""
    by = []
    if list(ids) != list(want_ids):
        by.append("token ids")
    if r["rows"] != r["want_rows"]:
        by.append("row count")
    if "rel_l2" in r and not r["rel_l2"] <= r["tol_rel_l2"]:
        by.append("relative L2")
    if "rel_l2" in r and not r["token_cosine_min"] >= r["tol_cosine"]:
        by.append("token cosine")
    return by


def _show(label: str, key: str, r: dict) -> str:
    if "rel_l2" not in r:
        return (f"[check2] {label} {key}: {r['rows']} rows against "
                f"{r['want_rows']}")
    return (f"[check2] {label} {key}: rel L2 {r['rel_l2']:.4e}, "
            f"{r['rel_l2'] / r['tol_rel_l2']:.2f} of its gate "
            f"{r['tol_rel_l2']:.4e}; smallest token cosine "
            f"{r['token_cosine_min']:.6f} at row {r['worst_token']}, floor "
            f"{r['tol_cosine']:.6f}; row 0 rel L2 {r['first_row']:.4e}, the "
            f"other rows {r['other_rows']:.4e}")


def test_features_are_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    tol = m["tolerances"]
    rows = {}
    for key, prompt in m["inputs"]["prompts"].items():
        tap = handle.encode(prompt)
        got = np.array(tap.features.astype(mx.float32))
        ref = arrays[f"tap_{key}"]
        assert got.shape == ref.shape, key
        assert tap.hidden[0].shape == (1, len(tap.ids), 4096)
        assert tap.hidden[0].dtype == mx.float32 == tap.features.dtype
        rows[key] = _reading(got, ref, tol, key)
        print(_show("gmlx", key, rows[key]))
    for key, r in rows.items():
        assert not _caught(r, [], []), (key, r)


def _wrong_template(spec, old: str, new: str):
    assert spec.template.count(old) == 1
    return dataclasses.replace(spec, template=spec.template.replace(old, new))


def _features(handle, spec, prompt: str, *, layers=None, norm=False,
              drop=None, pad=0):
    """Token ids and float32 features of ``prompt`` by an encoder that
    departs from the right one as the arguments say."""
    from gmlx.image import text_encoder as te

    ids = te.token_ids(spec, handle.tokenizer, prompt)
    fed = ids
    if pad:
        (pad_id,) = handle.tokenizer.encode("<|endoftext|>",
                                            add_special_tokens=False)
        fed = [pad_id] * pad + ids
    (h,) = te.tap(handle.model, fed, layers=layers or spec.tap_layers,
                  activation=spec.activation)
    if norm:
        h = handle.model.model.norm(h)
    h = h[0, pad + (spec.drop if drop is None else drop):]
    mx.eval(h)
    return ids, np.array(h.astype(mx.float32))


# name: (the wrong encoder, the readings that must catch it)
WRONG = {
    "tap at layer 35": (
        lambda h, s, p: _features(h, s, p, layers=(35,)),
        {"relative L2", "token cosine"}),
    "the final norm applied": (
        lambda h, s, p: _features(h, s, p, norm=True),
        {"relative L2", "token cosine"}),
    "system tokens kept": (
        lambda h, s, p: _features(h, s, p, drop=0), {"row count"}),
    "one word of the template changed": (
        lambda h, s, p: _features(
            h, _wrong_template(s, "provided", "given"), p),
        {"token ids"}),
    "the prompt padded": (
        lambda h, s, p: _features(h, s, p, pad=PAD),
        {"relative L2", "token cosine"}),
    "thinking tokens left in": (
        lambda h, s, p: _features(
            h, _wrong_template(s, "assistant\n",
                               "assistant\n<think>\n\n</think>\n\n"), p),
        {"token ids", "row count"}),
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    """Every prompt of the fixture catches the wrong encoder, by a
    reading of the kind that can see it."""
    arrays, m = check2
    make, kinds = WRONG[name]
    missed = {}
    for key, prompt in m["inputs"]["prompts"].items():
        ids, got = make(handle, handle.spec, prompt)
        want_ids = arrays[f"ids_{key}"].tolist()
        r = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        by = _caught(r, ids, want_ids)
        print(_show(name, key, r) + f"; caught by {', '.join(by) or 'none'}")
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
