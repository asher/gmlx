"""Check 2 of the image acceptance gate for MiniMax-H3-Image: the text
encoder tap on image prompts.

gmlx encodes the oracle's prompts with the stock Qwen3-VL-32B text GGUF
the oracle read, after decoder layer 50 of 64 and before the final norm,
by the image family's encoder spec. The token ids must equal the ids
ComfyUI's MiniMaxH3ImageConditioning hands the encoder for a prompt
without images, and the features of each prompt must stay within the
manifest's relative L2 and per-token cosine of the oracle's fp32
features. The gates follow video H3's check 2: twice the oracle's own
bf16 against fp32 gap, over every row and over the rows after the first.

The prompts are image prompts, one asking for lettering, and rows_4096,
a prompt of the 4096 text rows the family prices. On a prompt of 1024
rows or more the encoder grows rows of very large norm by layer 44, and
from there the bf16 gap jumps: the oracle's on rows_4096 from 2e-2 to
9e-2. So rows_4096's value gates are loose, and a tap one layer early
stays under them; the short prompts catch that one. Its ids and row
count are exact. Each wrong encoder below must fail some reading. The
reads past the tap need a second load of all 64 layers, and video H3's
check 2 holds them on the same spec.

The rows on the real file need the encoder GGUF the fixture names under
KQUANT_TEST_GGUF_DIR and the fixture under GMLX_IMAGE_FIXTURES.
scripts/image_oracle/check2_h3_image.py writes it."""
from __future__ import annotations

import contextlib
import os
import time

import numpy as np
import pytest
from oracle_fixture import committed, fixture, fx

from gmlx.image.families.h3_image import family as hi
from gmlx.image.h3 import encoder as h3e

NAME = "check2_h3_image"
PROMPTS = ("plain", "empty", "non_ascii", "long", "typography", "rows_4096")
ROWS = tuple(k for k in PROMPTS if k != "empty")
SHORT = tuple(k for k in PROMPTS if k != "rows_4096")


def token_cosine(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    return (g * r).sum(-1) / (np.linalg.norm(g, axis=-1)
                              * np.linalg.norm(r, axis=-1))


def test_the_manifest_follows_the_rule():
    """Each prompt's gates are twice its bf16 gaps and the cosine floor
    from its smallest cosine, the tap is the family's, rows_4096 holds the
    rows the family prices, and the name map gave the right next token."""
    from gmlx.image import memfit

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
        assert len(gap["rel_l2_by_layer"]) == h3e.TAP_LAYER, key
    lengths = m["inputs"]["lengths"]
    assert lengths["empty"] == 1
    assert lengths["rows_4096"] == memfit.PEAKS["h3_image"].text_rows == 4096
    assert m["inputs"]["tap"].startswith(
        f"output of decoder layer {h3e.TAP_LAYER} of 64, before the final "
        "norm")
    assert "picture_label='Picture %d: '" in m["inputs"]["tokenizer"]
    assert m["results"]["name_map_check"]["token"] == " Paris"


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


@pytest.fixture(scope="module")
def handle(gguf_index, check2):
    from gmlx.image import text_encoder as te

    spec = te.encoder_spec(hi.FAMILY)
    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(spec.arch, []):
        if (os.path.basename(p) == want["name"]
                and os.path.getsize(p) == want["size"]):
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    h = te.load_encoder(p, spec, log=lambda *_: None)
    yield h
    h.release()


def _reading(got: np.ndarray, ref: np.ndarray, tol: dict, key: str) -> dict:
    """The gated readings of one prompt and their gates. Features with
    another row count than the oracle's have no reading."""
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
            f"{r['rel_l2'] / r['tol_rel_l2']:.3f} of its gate "
            f"{r['tol_rel_l2']:.4e}{rest}; smallest token cosine "
            f"{r['token_cosine_min']:.6f} at row {r['worst_token']}, floor "
            f"{r['tol_cosine']:.6f}")


@pytest.mark.integration
def test_the_encoder_is_the_identified_one_up_to_its_tap(handle):
    from gmlx.image import text_encoder as te

    spec = handle.spec
    assert handle.owned and spec.tap_layers == (h3e.TAP_LAYER,)
    assert spec.activation == "float32"
    assert len(te._inner(handle.model).layers) == h3e.TAP_LAYER == 50
    got = te.model_identity(handle.model, spec)
    assert got == {r.gguf_name: r.sha256 for r in spec.identity}


@pytest.mark.integration
def test_token_ids_equal_comfyuis(handle, check2):
    """The ids are ComfyUI's, and the memory check, which counts rows from
    the prompt's bytes before a tokenizer loads, prices at least them."""
    from gmlx.commands.image_preflight import prompt_rows

    arrays, m = check2
    for key, prompt in m["inputs"]["prompts"].items():
        ids = handle.token_ids(prompt)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert len(ids) == m["inputs"]["lengths"][key], key
        assert prompt_rows(hi.FAMILY, prompt) >= len(ids), key
    assert handle.token_ids("") == list(h3e.EMPTY_IDS)


@pytest.mark.integration
def test_features_are_within_the_calibrated_tolerance(handle, check2):
    import mlx.core as mx

    arrays, m = check2
    rows = {}
    for key, prompt in m["inputs"]["prompts"].items():
        t0 = time.perf_counter()
        tap = handle.encode(prompt)
        got = np.array(tap.features)
        secs = time.perf_counter() - t0
        assert tap.hidden[0].shape == (1, len(tap.ids), 5120), key
        assert tap.features.dtype == mx.float32, key
        rows[key] = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        print(_show("gmlx", key, rows[key]) + f"; {secs:.2f} s")
    for key, r in rows.items():
        assert not _caught(r, [], []), (key, r)


def _features(handle, ids, *, layer=h3e.TAP_LAYER, norm=False) -> np.ndarray:
    """Float32 features of ``ids`` by an encoder that departs from the
    right one as the arguments say."""
    import mlx.core as mx

    from gmlx.image import text_encoder as te

    (h,) = te.tap(handle.model, ids, layers=(layer,),
                  activation=handle.spec.activation)
    if norm:
        h = te._inner(handle.model).norm(h)
    mx.eval(h)
    return np.array(h[0].astype(mx.float32))


@contextlib.contextmanager
def _qk_norms_swapped(model):
    from gmlx.image import text_encoder as te

    def swap():
        for layer in te._inner(model).layers:
            a = layer.self_attn
            a.q_norm.weight, a.k_norm.weight = a.k_norm.weight, a.q_norm.weight

    swap()
    try:
        yield
    finally:
        swap()


def _swapped(handle, prompt: str):
    ids = handle.token_ids(prompt)
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
        lambda h, p: (h.token_ids(p),
                      _features(h, h.token_ids(p), layer=h3e.TAP_LAYER - 1)),
        {"relative L2", "token cosine"}, SHORT),
    "the final norm applied": (
        lambda h, p: (h.token_ids(p), _features(h, h.token_ids(p),
                                                norm=True)),
        {"relative L2", "token cosine"}, PROMPTS),
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
    arrays, m = check2
    make, kinds, keys = WRONG[name]
    missed = {}
    for key in keys:
        ids, got = make(handle, m["inputs"]["prompts"][key])
        r = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        print(_show(name, key, r) + f"; caught by {', '.join(by) or 'none'}")
        if not set(by) & kinds:
            missed[key] = (by, r)
    assert not missed, (name, missed)
