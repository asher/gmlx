"""Check 2 of the image acceptance gate for Qwen-Image 2.1 with reference
images: the encoder tap over the edit template.

gmlx loads the Qwen3-VL-8B text GGUF with the mmproj GGUF the oracle read
and encodes each case of the fixture from the oracle's source images. A
case is right when every reading below holds:

  reference pixels   the resized RGBA image equals the reference
                     pipeline's in bits
  tower input        the patches the vision tower gets equal the
                     reference processor's, to a float32 rounding
  token ids          equal to the oracle's, the pad of each slot repeated
                     once per image row
  row count, image slots
                     the features have the oracle's rows, and their image
                     rows are the oracle's
  relative L2, token cosine
                     the features against the oracle's fp32 features,
                     within the manifest's gates
  image rows, text rows
                     relative L2 over the image rows and over the text
                     rows alone, within the manifest's gates

The gates are twice the oracle's own gap between its float32 run and its
run with the language model in bfloat16 and the tower in float32. gmlx
runs both in float32. The oracle's run with the tower in bfloat16, as the
reference pipeline loads it, moves the image rows by about 30 percent and
sets no gate. The manifest holds that reading.

Each wrong encoder of WRONG must fail a reading of the kind named beside
it, on every case it applies to:

  wrong encoder                         caught by
  the image slot missing                token ids, row count
  a slot without its <imageN> label     token ids, row count
  two references swapped                relative L2, token cosine, image
                                        rows
  resized with bicubic                  reference pixels
  sides rounded down                    reference pixels, token ids (the
                                        cases with a reference that is
                                        not square)
  the alpha over black for the tower    tower input
  one deepstack block short             image rows
  image rows at text positions          relative L2, image rows

Skipped unless KQUANT_TEST_GGUF_DIR holds the encoder and the mmproj GGUF
the fixture names and GMLX_IMAGE_FIXTURES holds the fixture.
scripts/image_oracle/check2_qwen_image21_edit.py writes it."""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pytest
from oracle_fixture import fixture, fx

import mlx.core as mx

from test_image_check2_qwen import _sha256, token_cosine

pytestmark = pytest.mark.integration

NAME = "check2_qwen_image21_edit"
TOWER = "check2_qwen_image21_edit_tower"
PATCH_TOL = 2e-7        # two float32 roundings of a value in [-1, 1]


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


def _find(paths, want: dict):
    for p in paths:
        if (os.path.basename(p) == want["name"]
                and os.path.getsize(p) == want["size"]
                and _sha256(p) == want["sha256"]):
            return p
    return None


@pytest.fixture(scope="module")
def handle(gguf_index, gguf_dir, check2):
    from gmlx.image import text_encoder as te

    spec = te.encoder_spec("qwen_image21")
    files = check2[1]["files"]
    enc = _find(gguf_index.get(spec.arch, []), files["encoder"])
    mmproj = _find([str(p) for p in gguf_dir.rglob(files["mmproj"]["name"])],
                   files["mmproj"])
    if enc is None or mmproj is None:
        pytest.skip(f"{files['encoder']['name']} and "
                    f"{files['mmproj']['name']} are not under "
                    "KQUANT_TEST_GGUF_DIR")
    h = te.load_encoder(enc, spec, mmproj=mmproj, log=lambda *_: None)
    yield h
    h.release()


def _area(m: dict) -> int:
    return m["inputs"]["output_resolution"] ** 2


def _refs(arrays, m, case: str, order=None):
    from gmlx.image.families.qwen_image21 import edit

    names = m["inputs"]["cases"][case]["images"]
    return names, [edit.prepare(arrays[f"source_{n}"], _area(m))
                   for n in (names if order is None else order)]


def test_the_encoder_holds_its_tower_in_float32(handle):
    from mlx.utils import tree_flatten

    from gmlx.image import memfit
    from gmlx.image import text_encoder as te

    assert handle.vision and handle.owned
    tower = handle.model.vision_tower
    kinds = {v.dtype for _, v in tree_flatten(tower.parameters())}
    assert kinds == {mx.float32}
    assert "lm_head" not in handle.model.language_model
    assert len(handle.model.language_model.model.layers) == 36
    got = te.model_identity(handle.model.language_model, handle.spec)
    assert got == {r.gguf_name: r.sha256 for r in handle.spec.identity}
    assert handle.vision_bytes() > 2 << 30
    assert memfit.PEAKS["qwen_image21"].mmproj_fp32


def test_a_text_prompt_through_the_encoder_with_its_tower(handle):
    """The text tap of an encoder loaded with its tower is the tap of the
    text fixture: the tower changes nothing for a prompt without
    images."""
    arrays, m = fixture("check2_qwen_image21")
    key = "plain"
    tap = handle.encode(m["inputs"]["prompts"][key])
    assert list(tap.ids) == arrays[f"ids_{key}"].tolist()
    got = np.array(tap.features.astype(mx.float32))
    rel = fx.rel_l2(got, arrays[f"tap_{key}"])
    print(f"[check2e] text prompt with the tower loaded: rel L2 {rel:.4e}, "
          f"{rel / m['tolerances']['rel_l2'][key]:.2f} of its gate")
    assert rel <= m["tolerances"]["rel_l2"][key]


def test_a_text_prompt_taps_the_same_without_the_tower(handle, gguf_index):
    """A text prompt through the encoder loaded without its tower gives
    the conditioning of the encoder with it, in bits, so a server that
    loads the tower for edits makes the CLI's text to image results."""
    from gmlx.image import text_encoder as te

    _, m = fixture("check2_qwen_image21")
    enc = _find(gguf_index.get(handle.spec.arch, []),
                m["files"]["encoder"])
    if enc is None:
        pytest.skip(f"{m['files']['encoder']['name']} is not under "
                    "KQUANT_TEST_GGUF_DIR")
    text = te.load_encoder(enc, handle.spec, log=lambda *_: None)
    try:
        for key, prompt in m["inputs"]["prompts"].items():
            a, b = handle.encode(prompt), text.encode(prompt)
            assert a.ids == b.ids, key
            assert mx.array_equal(a.features, b.features).item(), key
    finally:
        text.release()


def test_references_are_resized_as_the_reference_pipeline_resizes(check2):
    from gmlx.image.families.qwen_image21 import edit

    arrays, m = check2
    for name, row in m["inputs"]["images"].items():
        ref = edit.prepare(arrays[f"source_{name}"], _area(m))
        want = arrays[f"resized_{name}"]
        assert ref.pixels.shape == want.shape, name
        assert np.array_equal(ref.pixels, want), name
        assert (ref.width, ref.height) != (row["width"], row["height"])
        assert edit.slots(ref) == want.shape[0] * want.shape[1] // 1024
        # An image already at its size is not resized again.
        again = edit.prepare(ref.pixels, _area(m))
        assert np.array_equal(again.pixels, ref.pixels)
        assert again.sha256 == ref.sha256


def test_the_tower_gets_the_patches_of_the_reference_processor(handle,
                                                               check2):
    from gmlx.image.families.qwen_image21 import edit

    arrays, m = check2
    for name in m["inputs"]["images"]:
        ref = edit.prepare(arrays[f"source_{name}"], _area(m))
        pixels, grid = edit._pixels(handle, ref)
        want = arrays[f"patches_{name}"]
        assert grid == [1, ref.height // 16, ref.width // 16]
        got = np.array(pixels).reshape(want.shape)
        gap = float(np.abs(got - want).max())
        print(f"[check2e] patches {name}: max abs {gap:.2e}")
        assert gap <= PATCH_TOL, name


def _ids_digest(ids) -> str:
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def test_token_ids_and_image_rows_equal_the_oracle(handle, check2):
    from gmlx.image.families.qwen_image21 import edit

    arrays, m = check2
    drop = handle.spec.drop
    assert drop == m["inputs"]["drop_index"]
    for case, row in m["inputs"]["cases"].items():
        names, refs = _refs(arrays, m, case)
        assert edit.render(row["prompt"], len(refs)) == (
            m["inputs"]["rendered"][case])
        ids = edit.token_ids(handle, row["prompt"], refs)
        assert ids == arrays[f"ids_{case}"].tolist(), case
        assert _ids_digest(ids) == m["tensors"][f"ids_{case}"]["sha256"]
        assert [edit.slots(r) for r in refs] == (
            m["inputs"]["layout"][case]["image_rows"])


def _parts(got, ref, slots) -> dict:
    is_slot = np.zeros(ref.shape[0], bool)
    is_slot[np.asarray(slots, np.int64)] = True
    return {"image_rows": fx.rel_l2(got[is_slot], ref[is_slot]),
            "text_rows": fx.rel_l2(got[~is_slot], ref[~is_slot])}


def _reading(got, ref, slots, want_slots, tol: dict, case: str) -> dict:
    out = {"rows": int(got.shape[0]), "want_rows": int(ref.shape[0]),
           "slots": list(slots) == list(want_slots),
           "tol": {k: tol[k][case] for k in (
               "rel_l2", "token_cosine_min", "rel_l2_image_rows",
               "rel_l2_text_rows")}}
    if got.shape == ref.shape:
        cos = token_cosine(got, ref)
        out.update(rel_l2=fx.rel_l2(got, ref),
                   token_cosine_min=float(cos.min()),
                   worst_token=int(cos.argmin()),
                   **_parts(got, ref, want_slots))
    return out


def _caught(r: dict, ids, want_ids) -> list:
    by = []
    if list(ids) != list(want_ids):
        by.append("token ids")
    if r["rows"] != r["want_rows"]:
        by.append("row count")
    elif not r["slots"]:
        by.append("image slots")
    if "rel_l2" in r:
        t = r["tol"]
        if not r["rel_l2"] <= t["rel_l2"]:
            by.append("relative L2")
        if not r["token_cosine_min"] >= t["token_cosine_min"]:
            by.append("token cosine")
        if not r["image_rows"] <= t["rel_l2_image_rows"]:
            by.append("image rows")
        if not r["text_rows"] <= t["rel_l2_text_rows"]:
            by.append("text rows")
    return by


def _show(label: str, case: str, r: dict) -> str:
    if "rel_l2" not in r:
        return (f"[check2e] {label} {case}: {r['rows']} rows against "
                f"{r['want_rows']}")
    t = r["tol"]
    return (f"[check2e] {label} {case}: rel L2 {r['rel_l2']:.4e}, "
            f"{r['rel_l2'] / t['rel_l2']:.3f} of its gate "
            f"{t['rel_l2']:.4e}; image rows {r['image_rows']:.4e}, "
            f"{r['image_rows'] / t['rel_l2_image_rows']:.3f} of "
            f"{t['rel_l2_image_rows']:.4e}; text rows {r['text_rows']:.4e}, "
            f"{r['text_rows'] / t['rel_l2_text_rows']:.3f} of "
            f"{t['rel_l2_text_rows']:.4e}; smallest token cosine "
            f"{r['token_cosine_min']:.6f} at row {r['worst_token']}, floor "
            f"{t['token_cosine_min']:.6f}")


def _encode(handle, prompt: str, refs, **kw):
    from gmlx.image.families.qwen_image21 import edit

    ids, feats, slots = edit.encode(handle, prompt, refs, **kw)
    assert feats.dtype == mx.float32
    return ids, np.array(feats), slots


def test_features_are_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    rows = {}
    for case, row in m["inputs"]["cases"].items():
        _, refs = _refs(arrays, m, case)
        ids, got, slots = _encode(handle, row["prompt"], refs)
        ref = arrays[f"tap_{case}"]
        assert got.shape == ref.shape, case
        rows[case] = _reading(got, ref, slots, arrays[f"slots_{case}"],
                              m["tolerances"], case)
        print(_show("gmlx", case, rows[case]))
        assert not _caught(rows[case], ids, arrays[f"ids_{case}"]), (
            case, rows[case])


def test_the_tower_of_an_edit_is_float32_against_the_float64_oracle(
        handle, check2):
    """The tower's outputs in an edit encode, against the oracle's tower in
    float64 on the same patches: each output within twice the oracle's
    own float32 against float64 relative L2 on it. A tower that multiplies
    at TF32 precision moves them by tens of those gates."""
    import mlx.nn as nn

    arrays, m = check2
    tarr, tm = fixture(TOWER)
    assert tm["inputs"]["source"] == NAME
    for k, sha in tm["inputs"]["source_npz_sha256"].items():
        assert m["tensors"][k]["sha256"] == sha, k
    tower = handle.model.vision_tower
    seen = []

    class Watched(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = tower

        def __call__(self, pixels, grid):
            f, d = tower(pixels, grid)
            seen.append([f, *d])
            return f, d

    handle.model.vision_tower = Watched()
    try:
        for case in ("one", "two"):
            seen.clear()
            names, refs = _refs(arrays, m, case)
            _encode(handle, m["inputs"]["cases"][case]["prompt"], refs)
            assert len(seen) == len(names)
            for name, outs in zip(names, seen):
                h, w = arrays[f"resized_{name}"].shape[:2]
                assert tm["inputs"]["grids"][name] == [1, h // 16, w // 16]
                for k, got in zip(tm["inputs"]["outputs"], outs):
                    tol = tm["tolerances"]["rel_l2"][name][k]
                    rel = fx.rel_l2(np.array(got), tarr[f"{k}_f64_{name}"])
                    print(f"[check2e] tower {name} {k}: rel L2 {rel:.3e}, "
                          f"{rel / tol:.2f} of its gate {tol:.3e}")
                    assert rel <= tol, (name, k)
    finally:
        handle.model.vision_tower = tower


def test_the_references_of_a_case_run_one_at_a_time_or_as_the_api_does(
        handle, check2, tmp_path):
    """encode_text on a pipeline gives the features of the family's own
    encode, with the image rows and the references on the conditioning,
    from the cache the second time."""
    from types import SimpleNamespace

    from gmlx.image import api
    from gmlx.image.registry import get_family

    arrays, m = check2
    case = "two"
    row = m["inputs"]["cases"][case]
    pipe = api.ImagePipeline(dit=SimpleNamespace(), family=get_family(
        "qwen_image21"), variant="base", encoder=handle)
    images = [arrays[f"source_{n}"] for n in row["images"]]
    refs = api.references(pipe, images, area=_area(m))
    cond = api.encode_text(pipe, row["prompt"], images=refs,
                           ref_latents=False)
    assert cond.pending and cond.refs == refs and cond.key
    assert list(cond.image_slots) == arrays[f"slots_{case}"].tolist()
    _, want, _ = _encode(handle, row["prompt"], list(refs))
    assert np.array_equal(np.array(cond.features), want)
    assert api.encode_text(pipe, row["prompt"], images=refs,
                           ref_latents=False) is cond
    swapped = api.encode_text(pipe, row["prompt"], images=refs[::-1],
                              ref_latents=False)
    assert swapped.key != cond.key
    text = api.encode_text(pipe, row["prompt"])
    assert text.key not in (cond.key, swapped.key) and not text.image_slots
    with pytest.raises(ValueError, match="encode_references first"):
        api._need_latents(pipe, cond)


def _swap(handle, arrays, m, case):
    row = m["inputs"]["cases"][case]
    names, refs = _refs(arrays, m, case, order=row["images"][::-1])
    return _encode(handle, row["prompt"], refs)


def _no_slot(handle, arrays, m, case, label: bool):
    """The prompt in a template without its image slots, or with slots
    that lack their <imageN> label."""
    from gmlx.image.families.qwen_image21 import edit
    from gmlx.image.families.qwen_image21.family import PROMPT

    row = m["inputs"]["cases"][case]
    _, refs = _refs(arrays, m, case)
    values = PROMPT.values
    old = values["image_slot"]
    values["image_slot"] = (old.replace("<image{n}>", "") if label else "")
    try:
        if label:
            return _encode(handle, row["prompt"], refs)
        from gmlx.image import text_encoder as te

        ids = handle.tokenizer.encode(
            edit.render(row["prompt"], len(refs)), add_special_tokens=False)
        (h,) = te.tap(handle.model, ids, layers=handle.spec.tap_layers,
                      activation=handle.spec.activation)
        feats = h[0, handle.spec.drop:]
        mx.eval(feats)
        return list(ids), np.array(feats), ()
    finally:
        values["image_slot"] = old


def _resized(handle, arrays, m, case, how: str):
    """The case with its references resized by another rule. Returns the
    encode and the references."""
    from gmlx.image.families.qwen_image21 import edit
    from gmlx.image.families.qwen_image21.family import REFERENCES

    row = m["inputs"]["cases"][case]
    values, dims = REFERENCES.values, edit.dimensions
    old = values["resample"]
    try:
        if how == "bicubic":
            values["resample"] = "bicubic"
        else:
            def floored(area, ratio):
                mult = values["side_multiple"]
                width = math.sqrt(area * ratio)
                return (int(width / mult) * mult,
                        int(width / ratio / mult) * mult)
            edit.dimensions = floored
        _, refs = _refs(arrays, m, case)
    finally:
        values["resample"] = old
        edit.dimensions = dims
    return _encode(handle, row["prompt"], refs), refs


def _over_black(handle, arrays, m, case):
    from gmlx.image.families.qwen_image21.family import REFERENCES

    row = m["inputs"]["cases"][case]
    _, refs = _refs(arrays, m, case)
    values = REFERENCES.values
    old = values["encoder_background"]
    values["encoder_background"] = [0, 0, 0]
    try:
        from gmlx.image.families.qwen_image21 import edit

        patches = [np.array(edit._pixels(handle, r)[0]) for r in refs]
        return _encode(handle, row["prompt"], refs), patches
    finally:
        values["encoder_background"] = old


def _flat_positions(handle, arrays, m, case):
    """The image rows at the positions of text rows: one position per
    row on every axis."""
    row = m["inputs"]["cases"][case]
    _, refs = _refs(arrays, m, case)
    lm = handle.model.language_model
    real = lm.get_rope_index
    lm.get_rope_index = lambda x, *a, **k: real(x, None, None, None)
    try:
        return _encode(handle, row["prompt"], refs)
    finally:
        del lm.get_rope_index


def _pixels_equal(arrays, m, case, refs) -> bool:
    names = m["inputs"]["cases"][case]["images"]
    return all(r.pixels.shape == arrays[f"resized_{n}"].shape
               and np.array_equal(r.pixels, arrays[f"resized_{n}"])
               for n, r in zip(names, refs))


def _run_wrong(name: str, handle, arrays, m, case):
    """(ids, features, slots, extra readings failed) of a wrong encoder
    on one case, or None when the case cannot show it."""
    images = m["inputs"]["cases"][case]["images"]
    extra = []
    if name == "the image slot missing":
        out = _no_slot(handle, arrays, m, case, label=False)
    elif name == "a slot without its <imageN> label":
        out = _no_slot(handle, arrays, m, case, label=True)
    elif name == "two references swapped":
        if len(images) < 2:
            return None
        out = _swap(handle, arrays, m, case)
    elif name in ("resized with bicubic", "sides rounded down"):
        out, refs = _resized(handle, arrays, m, case,
                             "bicubic" if "bicubic" in name else "floor")
        # A square image has the same sides under both roundings.
        if "rounded" in name and all(
                r.pixels.shape == arrays[f"resized_{n}"].shape
                for n, r in zip(images, refs)):
            return None
        if not _pixels_equal(arrays, m, case, refs):
            extra.append("reference pixels")
    elif name == "the alpha over black for the tower":
        if not any(m["inputs"]["images"][n]["alpha"] for n in images):
            return None
        out, patches = _over_black(handle, arrays, m, case)
        if any(float(np.abs(p.reshape(arrays[f"patches_{n}"].shape)
                            - arrays[f"patches_{n}"]).max()) > PATCH_TOL
               for n, p in zip(images, patches)):
            extra.append("tower input")
    elif name == "one deepstack block short":
        row = m["inputs"]["cases"][case]
        _, refs = _refs(arrays, m, case)
        out = _encode(handle, row["prompt"], refs, deepstack=2)
    elif name == "image rows at text positions":
        out = _flat_positions(handle, arrays, m, case)
    else:
        raise KeyError(name)
    return (*out, extra)


# name: the readings of which one at least must catch it.
WRONG = {
    "the image slot missing": {"token ids", "row count"},
    "a slot without its <imageN> label": {"token ids", "row count"},
    "two references swapped": {"relative L2", "token cosine", "image rows"},
    "resized with bicubic": {"reference pixels"},
    "sides rounded down": {"reference pixels", "token ids"},
    "the alpha over black for the tower": {"tower input"},
    "one deepstack block short": {"image rows"},
    "image rows at text positions": {"relative L2", "image rows"},
}


@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_").replace("<", "").replace(
                             ">", "") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, name):
    """Every case that can show the wrong encoder catches it, by a
    reading of the kind that can see it."""
    arrays, m = check2
    missed, shown = {}, 0
    for case in m["inputs"]["cases"]:
        got = _run_wrong(name, handle, arrays, m, case)
        if got is None:
            continue
        shown += 1
        ids, feats, slots, extra = got
        r = _reading(feats, arrays[f"tap_{case}"], slots,
                     arrays[f"slots_{case}"], m["tolerances"], case)
        by = extra + _caught(r, ids, arrays[f"ids_{case}"])
        print(_show(name, case, r) + f"; caught by {', '.join(by) or 'none'}")
        if not set(by) & WRONG[name]:
            missed[case] = (by, r)
    assert shown and not missed, (name, missed)
