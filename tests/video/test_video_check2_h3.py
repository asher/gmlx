"""Check 2 of the video acceptance gate for MiniMax-H3: the text encoder
tap.

gmlx encodes the oracle's prompts with the stock Qwen3-VL-32B text GGUF
the oracle read, after decoder layer 50 of 64 and before the final norm.
The token ids must equal ComfyUI's, the seven fixed-id tokens among
them, and the features of each prompt must stay within the manifest's
relative L2 and per-token cosine of the oracle's fp32 features.

The relative L2 gates are twice the oracle's own bf16 against fp32 gap,
over every row and over the rows after the first, since the first row
holds nearly all of the norm. Each wrong encoder below must fail some
reading of this file: the token ids, the row count, or the gates. The
ids hold what changes the text: a chat template, the seven MiniMax
tokens left to the GGUF tokenizer. The gates hold what changes the
values on the same ids: a tap one layer early or late, the final norm
applied, the q and k norms swapped.

The rows on the real file need the encoder GGUF the fixture names under
KQUANT_TEST_GGUF_DIR and the fixture under GMLX_VIDEO_FIXTURES.
scripts/video_oracle/check2_h3.py writes it."""
from __future__ import annotations

import contextlib
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
from video_oracle_fixture import committed, fixture, fx

import mlx.core as mx

from gmlx.image import text_encoder as te
from gmlx.video import encoder, memfit
from gmlx.video.families.h3 import family as hf

NAME = "check2_h3"
PROMPTS = ("plain", "empty", "non_ascii", "long", "dialogue", "markup")
ROWS = tuple(k for k in PROMPTS if k != "empty")
FIXED = dict(hf.ADDED_TOKENS)
MMPROJ = "mmproj-BF16.gguf"


def token_cosine(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    return (g * r).sum(-1) / (np.linalg.norm(g, axis=-1)
                              * np.linalg.norm(r, axis=-1))


def test_the_manifest_follows_the_rule():
    """Each prompt's gates are twice its bf16 gaps and the cosine floor
    from its smallest cosine, the tap is the family's, the prompts hold
    every fixed-id token, and the name map gave the right next token."""
    m = committed(NAME)
    tol, gaps = m["tolerances"], m["results"]["bf16_vs_fp32"]
    assert tuple(m["inputs"]["prompts"]) == PROMPTS
    assert set(tol["rel_l2"]) == set(tol["token_cosine_min"]) == set(PROMPTS)
    assert set(tol["rel_l2_after_first"]) == set(ROWS)
    for key in PROMPTS:
        gap = gaps[key]
        assert tol["rel_l2"][key] == 2 * gap["rel_l2"], key
        if key in ROWS:
            assert tol["rel_l2_after_first"][key] == \
                2 * gap["rel_l2_after_first"], key
        assert tol["token_cosine_min"][key] == min(
            tol["starting"]["token_cosine_min"],
            1.0 - 4.0 * (1.0 - gap["token_cosine_min"])), key
        assert len(gap["rel_l2_by_layer"]) == hf.TAP_LAYER, key
    assert m["inputs"]["lengths"]["empty"] == 1
    assert m["inputs"]["tap"].startswith(
        f"output of decoder layer {hf.TAP_LAYER} of 64, before the final norm")
    for text in ("<d>", "</d>"):
        assert text in m["inputs"]["prompts"]["dialogue"]
    for text in FIXED:
        assert text in m["inputs"]["prompts"]["markup"]
    assert m["results"]["name_map_check"]["token"] == " Paris"


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


@pytest.fixture(scope="module")
def encoder_path(gguf_index, check2):
    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(hf.FAMILY.encoder().arch, []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return p


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
                   worst_token=int(cos.argmin()))
        if key in tol["rel_l2_after_first"]:
            out.update(rel_l2_rest=fx.rel_l2(got[1:], ref[1:]),
                       tol_rel_l2_rest=tol["rel_l2_after_first"][key])
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
    if "rel_l2_rest" in r and not r["rel_l2_rest"] <= r["tol_rel_l2_rest"]:
        by.append("relative L2 after the first row")
    if "rel_l2" in r and not r["token_cosine_min"] >= r["tol_cosine"]:
        by.append("token cosine")
    return by


def _show(label: str, key: str, r: dict) -> str:
    if "rel_l2" not in r:
        return (f"[check2] {label} {key}: {r['rows']} rows against "
                f"{r['want_rows']}")
    rest = ""
    if "rel_l2_rest" in r:
        rest = (f"; after the first row {r['rel_l2_rest']:.4e}, "
                f"{r['rel_l2_rest'] / r['tol_rel_l2_rest']:.2f} of its gate")
    return (f"[check2] {label} {key}: rel L2 {r['rel_l2']:.4e}, "
            f"{r['rel_l2'] / r['tol_rel_l2']:.2f} of its gate "
            f"{r['tol_rel_l2']:.4e}{rest}; smallest token cosine "
            f"{r['token_cosine_min']:.6f} at row {r['worst_token']}, floor "
            f"{r['tol_cosine']:.6f}")


def _ids(handle, prompt: str) -> list[int]:
    return handle.token_ids(prompt)


def _features(handle, ids, *, layer=hf.TAP_LAYER,
              norm=False) -> np.ndarray:
    """Float32 features of ``ids`` by an encoder that departs from the
    right one as the arguments say."""
    (h,) = te.tap(handle.model, ids, layers=(layer,),
                  activation=handle.spec.activation)
    if norm:
        h = handle.model.model.norm(h)
    mx.eval(h)
    return np.array(h[0].astype(mx.float32))


def _check_wrong(handle, check2, name: str, make, kinds: set, keys) -> None:
    """Each prompt in ``keys`` catches the wrong encoder ``make``, by a
    reading of the kind that can see it."""
    arrays, m = check2
    missed = {}
    for key in keys:
        ids, got = make(handle, m["inputs"]["prompts"][key])
        r = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        print(_show(name, key, r) + f"; caught by {', '.join(by) or 'none'}")
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)


# Wrong encoders that need the layers past the tap. They run first, on an
# encoder loaded without the drop, before the module's encoder loads.
PAST_TAP = {
    "tap at layer 51": lambda h, p: (
        _ids(h, p), _features(h, _ids(h, p), layer=hf.TAP_LAYER + 1)),
    # the last hidden state of transformers' model
    "every layer and the final norm": lambda h, p: (
        _ids(h, p), _features(h, _ids(h, p), layer=64, norm=True)),
}


@pytest.mark.integration
def test_a_read_past_the_tap_fails_a_reading(encoder_path, check2):
    spec = dataclasses.replace(hf.FAMILY.encoder(), drop_untapped=False)
    full = te.load_encoder(encoder_path, spec, log=lambda *_: None)
    try:
        assert len(full.model.model.layers) == spec.layers == 64
        for name, make in PAST_TAP.items():
            _check_wrong(full, check2, name, make,
                         {"relative L2", "token cosine"}, PROMPTS)
    finally:
        full.release()


@pytest.fixture(scope="module")
def handle(encoder_path):
    h = encoder.load_encoder(hf.FAMILY, encoder_path, log=lambda *_: None)
    yield h
    h.release()


@pytest.mark.integration
def test_the_encoder_is_the_identified_one_up_to_its_tap(handle):
    spec = handle.spec
    assert handle.owned and "lm_head" not in handle.model
    assert spec.layers == 64 and spec.tap_layers == (hf.TAP_LAYER,)
    assert len(handle.model.model.layers) == hf.TAP_LAYER == 50
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


@pytest.mark.integration
def test_the_mmproj_beside_it_is_the_identified_tower(encoder_path):
    from gmlx.image.fingerprint import view_from_scans
    from gmlx.load.headerscan import scan_gguf

    path = Path(encoder_path).parent / MMPROJ
    if not path.is_file():
        pytest.skip(f"{MMPROJ} is not beside {Path(encoder_path).name}")
    view = view_from_scans([scan_gguf(str(path))])
    spec = hf.FAMILY.encoder()
    te.check_mmproj_file(view, spec)
    first = spec.vision_identity[0]
    wrong = dataclasses.replace(spec, vision_identity=(
        dataclasses.replace(first, sha256="0" * 64),
        *spec.vision_identity[1:]))
    with pytest.raises(te.EncoderMismatchError):
        te.check_mmproj_file(view, wrong)


@pytest.mark.integration
def test_token_ids_equal_comfyuis(handle, check2):
    arrays, m = check2
    for key, prompt in m["inputs"]["prompts"].items():
        ids = _ids(handle, prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert len(ids) == m["inputs"]["lengths"][key], key
    dialogue = _ids(handle, m["inputs"]["prompts"]["dialogue"])
    assert dialogue.count(FIXED["<d>"]) == dialogue.count(FIXED["</d>"]) == 2
    markup = _ids(handle, m["inputs"]["prompts"]["markup"])
    assert all(markup.count(i) == 1 for i in FIXED.values())
    assert _ids(handle, "") == list(hf.EMPTY_IDS)


@pytest.mark.integration
def test_the_memory_check_counts_comfyuis_rows(encoder_path, check2):
    """The tokenizer the memory check builds from the encoder's header
    gives ComfyUI's ids, with no encoder loaded."""
    arrays, m = check2
    tok = encoder.header_tokenizer(hf.FAMILY, encoder_path)
    for key, prompt in m["inputs"]["prompts"].items():
        want = arrays[f"ids_{key}"].tolist()
        assert te.token_ids(hf.FAMILY.encoder(), tok, prompt) == want, key
        assert memfit.prompt_rows(hf.FAMILY, prompt, tok) == len(want), key


@pytest.mark.integration
def test_features_are_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    rows = {}
    for key, prompt in m["inputs"]["prompts"].items():
        tap = handle.encode(prompt)
        got = np.array(tap.features)
        assert tap.hidden[0].shape == (1, len(tap.ids), 5120), key
        assert tap.features.dtype == mx.float32, key
        rows[key] = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        print(_show("gmlx", key, rows[key]))
    for key, r in rows.items():
        assert not _caught(r, [], []), (key, r)


@contextlib.contextmanager
def _qk_norms_swapped(model):
    def swap():
        for layer in model.model.layers:
            a = layer.self_attn
            a.q_norm.weight, a.k_norm.weight = a.k_norm.weight, a.q_norm.weight

    swap()
    try:
        yield
    finally:
        swap()


def _swapped(handle, prompt: str):
    ids = _ids(handle, prompt)
    with _qk_norms_swapped(handle.model):
        return ids, _features(handle, ids)


def _template(handle, prompt: str) -> list[int]:
    return handle.tokenizer.encode(
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)


# name: (the wrong encoder, the readings that must catch it, the prompts
# that can show it)
WRONG = {
    "tap at layer 49": (
        lambda h, p: (_ids(h, p),
                      _features(h, _ids(h, p), layer=hf.TAP_LAYER - 1)),
        {"relative L2", "token cosine"}, PROMPTS),
    "the final norm applied": (
        lambda h, p: (_ids(h, p), _features(h, _ids(h, p), norm=True)),
        {"relative L2", "token cosine"}, PROMPTS),
    "the seven tokens left to the tokenizer": (
        lambda h, p: (
            h.tokenizer.encode(p, add_special_tokens=False),
            _features(h, h.tokenizer.encode(p, add_special_tokens=False))),
        {"token ids", "row count"}, ("dialogue", "markup")),
    "a chat template around the prompt": (
        lambda h, p: (_template(h, p), _features(h, _template(h, p))),
        {"token ids", "row count"}, PROMPTS),
    # Leaves the first row as it is, so only the rows after it show it.
    "the q and k norms swapped": (
        _swapped, {"relative L2 after the first row"}, ROWS),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    make, kinds, keys = WRONG[name]
    _check_wrong(handle, check2, name, make, kinds, keys)
