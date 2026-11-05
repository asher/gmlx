"""Check 2 of the video acceptance gate for MiniMax-H3, its keyframe
prompts: the encoder tap of a prompt with one and with two keyframes on
a 512x512 canvas.

scripts/video_oracle/check2_h3.py --prompts keyframe keyframes writes
the fixture: ComfyUI's ids, tags and tower input of each prompt, the
seeded source images at 640x400 and 400x640 and ComfyUI's resize of each
to the canvas, transformers' vision tower on the mmproj and the tap of
transformers' layers over the prompt, its tower rows in place of the
image pad rows, its deepstack rows added after the first decoder layers
and the rows at 3D RoPE positions. gmlx resizes each source as ComfyUI
does, bit for bit, builds the prompt, the tags and the tower input from
the images and must give ComfyUI's ids and tags and its tower input
within 1e-6, the oracle's tower rows within TOWER_TOL,
and features within the manifest's gates, which follow check 2's rule.
Each wrong encoder below must fail a gate: the deepstack rows dropped,
the vision block at text positions.

The manifest also reports ComfyUI's own tower and positions against
transformers', which no test gates.

The rows on the real file need the encoder GGUF and the mmproj beside it
under KQUANT_TEST_GGUF_DIR and the fixture under GMLX_VIDEO_FIXTURES."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from test_video_check2_h3 import _caught, _reading, _show
from video_oracle_fixture import committed, fixture, fx

import mlx.core as mx

from gmlx.video import encoder, images
from gmlx.video.families.h3 import family as hf
from gmlx.video.families.h3 import keyframes as kf

NAME = "check2_h3_keyframes"
PROMPTS = ("keyframe", "keyframes")
MMPROJ = "mmproj-BF16.gguf"
# Both sides run the tower in float32, gmlx with its products split to
# float32 exactness: the gap is summation order over 27 blocks.
TOWER_TOL = 1e-4


def test_the_manifest_follows_the_rule():
    """Each prompt's gates are twice its bf16 gaps and the cosine floor
    from its smallest cosine, the prompts hold one and two keyframes
    brought to the 512x512 canvas from other sizes, the first stretched
    and the last cropped, and the name map gave the right next token."""
    m = committed(NAME)
    tol, gaps = m["tolerances"], m["results"]["bf16_vs_fp32"]
    assert tuple(m["inputs"]["prompts"]) == PROMPTS
    for key in PROMPTS:
        gap = gaps[key]
        assert tol["rel_l2"][key] == 2 * gap["rel_l2"], key
        assert tol["rel_l2_after_first"][key] == \
            2 * gap["rel_l2_after_first"], key
        assert tol["token_cosine_min"][key] == min(
            tol["starting"]["token_cosine_min"],
            1.0 - 4.0 * (1.0 - gap["token_cosine_min"])), key
        assert len(gap["rel_l2_by_layer"]) == hf.TAP_LAYER, key
    kfs = m["inputs"]["keyframes"]
    assert {k: v["sizes"] for k, v in kfs.items()} == {
        "keyframe": [[512, 512]], "keyframes": [[512, 512], [512, 512]]}
    assert {k: v["sources"] for k, v in kfs.items()} == {
        "keyframe": [[640, 400]], "keyframes": [[640, 400], [400, 640]]}
    assert {k: v["crops"] for k, v in kfs.items()} == {
        "keyframe": ["disabled"], "keyframes": ["disabled", "center"]}
    assert m["results"]["name_map_check"]["token"] == " Paris"


@pytest.fixture(scope="module")
def check2():
    return fixture(NAME)


@pytest.fixture(scope="module")
def handle(gguf_index, check2):
    want = check2[1]["files"]["encoder"]
    for p in gguf_index.get(hf.FAMILY.encoder().arch, []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    mm = Path(p).parent / MMPROJ
    if not mm.is_file():
        pytest.skip(f"{MMPROJ} is not beside {Path(p).name}")
    for path, role in ((p, "encoder"), (str(mm), "mmproj")):
        if fx.file_identity(path)["sha256"] != check2[1]["files"][role][
                "sha256"]:
            pytest.fail(f"{path} is not the file the oracle read")
    h = encoder.load_encoder(hf.FAMILY, p, mmproj=str(mm),
                             log=lambda *_: None)
    yield h
    h.release()


def _images(arrays, key: str) -> list:
    """The prompt's keyframes, (H, W, 3) uint8, first then last."""
    n = sum(1 for k in arrays if k.startswith(f"image_{key}_"))
    return [arrays[f"image_{key}_{i}"] for i in range(n)]


def test_each_keyframe_is_comfyuis_resize_of_its_source(check2):
    """gmlx stretches the first source and crops the last as
    MiniMaxH3ImageToVideo does, to the same 8-bit pixels."""
    arrays, _ = check2
    for key in PROMPTS:
        for i, im in enumerate(_images(arrays, key)):
            resize = images.stretch if i == 0 else images.cover
            got = resize(arrays[f"source_{key}_{i}"], 512, 512)
            np.testing.assert_array_equal(got, im, err_msg=f"{key} {i}")


@pytest.mark.integration
def test_the_prompt_and_the_tower_input_are_comfyuis(handle, check2):
    """gmlx's ids and vision spans are ComfyUI's ids and tags, and its
    tower input is ComfyUI's within 1e-6."""
    arrays, m = check2
    for key in PROMPTS:
        images = _images(arrays, key)
        sizes = [(int(im.shape[1]), int(im.shape[0])) for im in images]
        ids, spans = kf.keyframe_ids(handle.spec, handle.tokenizer,
                                     m["inputs"]["prompts"][key], sizes)
        assert ids == arrays[f"ids_{key}"].tolist(), key
        assert spans == kf.spans_of(arrays[f"tags_{key}"]), key
        for i, im in enumerate(images):
            px, grid = kf.pixels(handle, im)
            assert grid == m["inputs"]["keyframes"][key]["grids"][i]
            worst = float(np.abs(np.asarray(px)
                                 - arrays[f"pixels_{key}_{i}"]).max())
            assert worst <= 1e-6, (key, i, worst)


@pytest.mark.integration
def test_the_tower_is_the_oracles(handle, check2):
    from gmlx.image import vision_tap

    arrays, _ = check2
    tower = handle.model.vision_tower
    worst = {}
    for key in PROMPTS:
        for i, im in enumerate(_images(arrays, key)):
            px, grid = kf.pixels(handle, im)
            with vision_tap.float32_products(tower):
                out, deep = tower(px, mx.array([grid]))
                mx.eval(out, *deep)
            rows = {f"merger_{i}": out, **{f"deepstack_{i}_{j}": d
                                           for j, d in enumerate(deep)}}
            for name, got in rows.items():
                worst[key, name] = fx.rel_l2(np.asarray(got),
                                             arrays[f"tower_{key}_{name}"])
    print("[check2] gmlx tower against the oracle's: worst "
          f"{max(worst.values()):.2e} at {max(worst, key=worst.get)}, "
          f"gate {TOWER_TOL:.0e}")
    over = {k: v for k, v in worst.items() if not v <= TOWER_TOL}
    assert not over, over


def _features(handle, prompt: str, images, *, deepstack=None) -> tuple:
    """(ids, float32 features) of a keyframe prompt, by an encoder that
    departs from the right one as the arguments say."""
    from gmlx.image import vision_tap

    sizes = [(int(im.shape[1]), int(im.shape[0])) for im in images]
    ids, _ = kf.keyframe_ids(handle.spec, handle.tokenizer, prompt, sizes)
    feats, _ = vision_tap.tap(handle, ids, (kf.pixels(handle, im)
                                            for im in images),
                              pad=hf.IMAGE_PAD, deepstack=deepstack)
    return ids, np.array(feats.astype(mx.float32))


@pytest.mark.integration
def test_features_are_within_the_calibrated_tolerance(handle, check2):
    arrays, m = check2
    rows = {}
    for key in PROMPTS:
        images = _images(arrays, key)
        feats, spans = kf.encode(handle, m["inputs"]["prompts"][key], images)
        assert spans == kf.spans_of(arrays[f"tags_{key}"]), key
        assert feats.dtype == mx.float32, key
        rows[key] = _reading(np.asarray(feats[0]), arrays[f"tap_{key}"],
                             m["tolerances"], key)
        print(_show("gmlx", key, rows[key]))
    print(f"[check2] ComfyUI's tower and positions against transformers' "
          f"(not gated): {m['results']['comfyui_column']}")
    for key, r in rows.items():
        assert not _caught(r, [], []), (key, r)


def _flat_positions(mp, handle):
    """The vision rows at the text positions their rows would have."""
    lm = type(handle.model.language_model)

    def flat(self, input_ids, *a, **k):
        n = input_ids.shape[1]
        pos = mx.broadcast_to(mx.arange(n)[None, None], (3, 1, n))
        return pos, mx.zeros((1, 1), mx.int32)
    mp.setattr(lm, "get_rope_index", flat)


# name: (the wrong encoder, given the handle and the monkeypatch)
WRONG = {
    "the deepstack rows dropped": lambda h, mp, p, im: _features(
        h, p, im, deepstack=0),
    "the vision block at text positions": lambda h, mp, p, im: (
        _flat_positions(mp, h), _features(h, p, im))[1],
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(WRONG),
                         ids=[n.replace(" ", "_") for n in WRONG])
def test_each_wrong_encoder_fails_a_reading(handle, check2, monkeypatch,
                                            name):
    arrays, m = check2
    missed = {}
    for key in PROMPTS:
        with monkeypatch.context() as mp:
            ids, got = WRONG[name](handle, mp, m["inputs"]["prompts"][key],
                                   _images(arrays, key))
        r = _reading(got, arrays[f"tap_{key}"], m["tolerances"], key)
        by = _caught(r, ids, arrays[f"ids_{key}"].tolist())
        print(_show(name, key, r) + f"; caught by {', '.join(by) or 'none'}")
        if not set(by) & {"relative L2", "relative L2 after the first row",
                          "token cosine"}:
            missed[key] = (by, r)
    assert not missed, (name, missed)
