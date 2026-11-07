"""The keyframe encode's staged load (gmlx/video/encoder.py attach_tower):
the text encoder alone, cut after its tap, then the vision tower from the
mmproj. It must build the joint load's model: the same parameters by
name, shape and dtype, the same module classes, the tower's weights to
the bit, and the tower output and keyframe features of the 64x64 fixture
image within 1e-5. The tower holds the bytes the memory plan prices for
it. The real files are the encoder GGUF check 2's
manifest names, under KQUANT_TEST_GGUF_DIR, and mmproj-BF16.gguf beside
it."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import host_box
import numpy as np
import pytest
from video_oracle_fixture import committed, committed_fixture

from gmlx.video import encoder
from gmlx.video.families.h3 import family as hf

MMPROJ = "mmproj-BF16.gguf"


def _tiny_pair(layers: int = 3, kept: int = 2):
    """A tiny mlx-lm qwen3 text model cut to ``kept`` layers without its
    head, as the text load leaves it, and the qwen3_vl language model of
    the same sizes."""
    from gmlx._vendor.mlx_lm.models import qwen3
    from gmlx._vendor.mlx_vlm.models.qwen3_vl import config as vc
    from gmlx._vendor.mlx_vlm.models.qwen3_vl import language

    sizes = dict(hidden_size=64, num_hidden_layers=layers,
                 intermediate_size=96, num_attention_heads=4,
                 rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
                 head_dim=16, rope_theta=1e6, max_position_embeddings=4096)
    text = qwen3.Model(qwen3.ModelArgs(model_type="qwen3",
                                       tie_word_embeddings=False, **sizes))
    text.model.layers = text.model.layers[:kept]
    del text["lm_head"]
    lm = language.LanguageModel(vc.TextConfig(
        model_type="qwen3_vl", tie_word_embeddings=False,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
        **sizes))
    return text, lm


def test_the_language_model_takes_the_loaded_modules():
    """Every parameter of the multimodal language model is one the text
    load filled, its layers are the kept ones and it has no head. A slot
    the text load has no module for is refused, since it would run on
    the weights it was built with, and so is a loaded module with weights
    the model has no place for, as a load that renamed a projection would
    leave it."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    text, lm = _tiny_pair()
    loaded = {id(v) for _, v in tree_flatten(text.parameters())}
    encoder._take_text(lm, text)
    assert len(lm.model.layers) == 2 and "lm_head" not in lm
    params = tree_flatten(lm.parameters())
    assert len(params) == len(loaded)
    assert all(id(v) in loaded for _, v in params)

    text, lm = _tiny_pair()
    del text.model.layers[1].self_attn["q_norm"]
    with pytest.raises(RuntimeError, match="did not fill: model.layers.1."
                                           "self_attn.q_norm.weight"):
        encoder._take_text(lm, text)
    text, lm = _tiny_pair()
    text.model.layers[0].self_attn.qkv_proj = nn.Linear(64, 128, bias=False)
    with pytest.raises(RuntimeError, match="no place for "
                                           "layers.0.self_attn.qkv_proj"):
        encoder._take_text(lm, text)


@pytest.fixture(scope="module")
def files(gguf_index):
    want = committed("check2_h3")["files"]["encoder"]
    for p in gguf_index.get(hf.FAMILY.encoder().arch, []):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    mm = Path(p).parent / MMPROJ
    if not mm.is_file():
        pytest.skip(f"{MMPROJ} is not beside {Path(p).name}")
    host_box.real_files()
    return p, str(mm)


def _readings(handle, image) -> dict:
    """What the parity compares, read from one load as numpy."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx.image import vision_tap
    from gmlx.video.families.h3 import keyframes as kf

    model = handle.model
    tower = model.vision_tower
    pixels, grid = kf.pixels(handle, image)
    with vision_tap.float32_products(tower):
        out, deep = tower(pixels, mx.array([grid]))
        mx.eval(out, *deep)
    feats, spans = kf.encode(handle, "A red fox trots through fresh snow",
                             [image])
    return {
        "params": {k: (tuple(v.shape), str(v.dtype))
                   for k, v in tree_flatten(model.parameters())},
        "classes": {k: f"{type(m).__module__}.{type(m).__qualname__}"
                    for k, m in model.named_modules()},
        # numpy holds no bfloat16; the dtypes are in "params".
        "tower_bytes": {k: hashlib.sha256(np.asarray(
            v.astype(mx.float32) if v.dtype == mx.bfloat16 else v).tobytes(
            )).hexdigest() for k, v in tree_flatten(tower.parameters())},
        "pixels": np.asarray(pixels),
        "tower": [np.asarray(out)] + [np.asarray(d) for d in deep],
        "features": np.asarray(feats),
        "spans": spans,
    }


@pytest.mark.integration
def test_the_staged_load_is_the_joint_load(files):
    from mlx.utils import tree_flatten

    from gmlx.image import text_encoder as te
    from gmlx.video import memfit

    path, mm = files
    image = committed_fixture("keyframes_h3")[0]["image_64"]
    staged = encoder.load_encoder(hf.FAMILY, path, mmproj=mm,
                                  log=lambda *_: None)
    try:
        assert len(staged.model.language_model.model.layers) == hf.TAP_LAYER
        assert "lm_head" not in staged.model.language_model
        held = sum(int(v.nbytes) for v in dict(tree_flatten(
            staged.model.vision_tower.parameters())).values())
        assert held == memfit.tower_bytes(mm)
        got = _readings(staged, image)
    finally:
        staged.release()
    joint = te.load_encoder(path, hf.FAMILY.encoder(), mmproj=mm,
                            log=lambda *_: None, tag="video")
    try:
        want = _readings(joint, image)
    finally:
        joint.release()
    assert got["params"] == want["params"]
    assert got["classes"] == want["classes"]
    assert got["tower_bytes"] == want["tower_bytes"]
    assert got["spans"] == want["spans"]
    np.testing.assert_array_equal(got["pixels"], want["pixels"])
    assert len(got["tower"]) == len(want["tower"]) == 4
    for g, w in zip(got["tower"] + [got["features"]],
                    want["tower"] + [want["features"]]):
        assert g.shape == w.shape
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-5)
