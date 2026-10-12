"""Real-weights Qwen3-VL: the text file alone, and with its mmproj.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` to hold a Qwen3-VL text
GGUF (architecture ``qwen3vl``) with a sibling mmproj that discovery pairs
to it. The text file loads as the qwen3 class and the pair as the qwen3_vl
container. The tests hold both loads to every parameter and every tensor,
the two models to the same hidden state on a text-only prompt, an image
prompt to the token layout llama.cpp's mtmd builds for the same pair, and
the image features of the 8B to the ones llama.cpp computes.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="a real-weights Qwen3-VL forward is a GPU workload")


@pytest.fixture(scope="module")
def pair(gguf_dir):
    """The (text GGUF, mmproj) of a Qwen3-VL model, from discovery."""
    from gmlx.config import DiscoverSpec
    from gmlx.load.discovery import scan_dirs
    from gmlx.load.headerscan import scan_gguf

    spec = DiscoverSpec(
        dir=None, recursive=True, pair_mmproj=True, speculative="auto"
    )
    for m in scan_dirs([spec], [str(gguf_dir)]):
        if not m.mmproj:
            continue
        kv = scan_gguf(m.path, include_tensors=False).kv
        if kv.get("general.architecture") == "qwen3vl":
            return m.path, m.mmproj
    pytest.skip(
        "no qwen3vl model with a paired sibling mmproj under "
        "KQUANT_TEST_GGUF_DIR"
    )


@pytest.fixture(scope="module")
def text_load(pair):
    from gmlx import load_model

    return load_model(pair[0])


@pytest.fixture(scope="module")
def vlm_load(pair):
    from gmlx.load.vlm import load_vlm_model

    return load_vlm_model(pair[0], pair[1])


@_NEEDS_GPU
def test_text_load_fills_every_parameter_from_every_tensor(pair, capsys):
    """A parameter without a tensor stops the load. fail_on_unknown stops
    it on a tensor without a name map, and the load log names a tensor
    without a parameter."""
    from gmlx import load_model
    from gmlx.load.headerscan import scan_gguf

    model, config, _ = load_model(pair[0], fail_on_unknown=True, verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    kv = scan_gguf(pair[0], include_tensors=False).kv
    assert config["model_type"] == "qwen3"
    assert config["rope_theta"] == kv["qwen3vl.rope.freq_base"]
    assert len(model.model.layers) == kv["qwen3vl.block_count"]


@_NEEDS_GPU
def test_pair_load_fills_every_parameter_from_every_tensor(pair, capsys):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model

    model, config, _ = load_vlm_model(pair[0], pair[1], verbose=True)
    log = capsys.readouterr().out
    assert "redundant" not in log, log
    # the mmproj line counts the tensors the vision remap left unused.
    assert " 0 skipped" in log, log
    assert config["model_type"] == "qwen3_vl"
    kv = scan_gguf(pair[0], include_tensors=False).kv
    flags = scan_gguf(pair[1], include_tensors=False).kv[
        "clip.vision.is_deepstack_layers"]
    blocks = [i for i, flag in enumerate(flags) if flag]
    tower = model.vision_tower
    assert list(tower.deepstack_visual_indexes) == blocks
    assert len(tower.deepstack_merger_list) == len(blocks)
    assert len(blocks) == kv["qwen3vl.n_deepstack_layers"]
    sections = kv["qwen3vl.rope.dimension_sections"]
    rotary = model.language_model.model.layers[0].self_attn.rotary_emb
    assert list(rotary.mrope_section) == [int(s) for s in sections[:3]]
    assert rotary.base == kv["qwen3vl.rope.freq_base"]


# A paragraph that does not repeat. Text that repeats a sentence many times
# parks a few tokens where any change of rounding moves them, in either
# class alone, so it measures the rounding and not the two classes.
_PROSE = (
    "The harbour town woke slowly that morning. Fishing boats that had gone "
    "out before dawn were already returning, low in the water, and the gulls "
    "followed them in a ragged white cloud. On the quay an old woman set out "
    "crates of ice while her grandson counted change into a tin box. The "
    "baker across the square propped open his door, and the smell of bread "
    "drifted over the cobbles toward the church, where the sexton was "
    "winding the clock as he had done every Tuesday for forty years. Nobody "
    "hurried. A delivery van idled by the chemist, its driver reading a "
    "newspaper folded into quarters. Two schoolchildren argued about a "
    "football match, a dog slept in a doorway, and somewhere above the "
    "rooftops a window opened and a radio began to play a song that had been "
    "popular one summer long ago. By nine the fog had lifted from the "
    "headland, the ferry sounded its horn twice, and the first visitors of "
    "the day came down the hill with their cameras and their paper maps, "
    "looking for the museum that had closed the year before.")

# The two classes share every weight and differ in the rope kernel alone,
# which seeds a different rounding in the 16-bit kernels of each layer.
# Measured on Qwen3-VL-8B Q4_K_M in bfloat16: cosine 0.9998 at the worst
# token and relative L2 0.009 over the prompt. A wrong rope base puts the
# cosine far under the floor. A wrong section list or axis layout does not
# show here: a text-only prompt puts one position on all three axes, so
# every frequency pair reads the same angle whichever axis it is given.
# The unequal-position test in tests/load/test_qwen3vl.py holds those.
_COS_FLOOR = 0.998
_REL_L2_CEILING = 0.03


@_NEEDS_GPU
def test_text_model_and_language_tower_agree_before_the_final_norm(
        text_load, vlm_load):
    """For a text-only prompt, the hidden state after the last layer with
    no final norm is the same from the text model and from the language
    tower of the pair, to the rounding of the 16-bit kernels. The text
    class has no call for that state, so its layers run here in order."""
    import mlx.core as mx
    import numpy as np

    from gmlx.cache.attention import create_attention_mask

    text, _, tok = text_load
    vlm = vlm_load[0]
    ids = mx.array([tok.encode(_PROSE, add_special_tokens=False)])

    h = text.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for layer in text.model.layers:
        h = layer(h, mask, None)
    got = vlm.hidden_state_at_layer(ids, len(text.model.layers))

    a = np.array(h.astype(mx.float32))[0].astype(np.float64)
    b = np.array(got.astype(mx.float32))[0].astype(np.float64)
    assert a.shape == b.shape
    cos = (a * b).sum(-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    rel = np.linalg.norm(a - b) / np.linalg.norm(a)
    assert cos.min() > _COS_FLOOR, (cos.min(), rel)
    assert rel < _REL_L2_CEILING, (cos.min(), rel)
    # under each model's own final norm and head the two pick the same
    # token, but for a few positions where two logits tie at 16 bits.
    differ = mx.argmax(text(ids), -1) != mx.argmax(vlm(ids).logits, -1)
    assert int(differ.sum()) <= ids.shape[1] // 20
    mx.clear_cache()


@_NEEDS_GPU
@pytest.mark.parametrize("size,grid", [((448, 448), (14, 14)),
                                       ((1280, 960), (30, 40))])
def test_image_prompt_has_the_layout_mtmd_builds(vlm_load, size, grid):
    """One image turn renders as the user header, vision_start, one
    image_pad per 32 x 32 pixels, vision_end, the text and the assistant
    header: the layout and the counts llama.cpp's mtmd builds for this
    pair at these two sizes."""
    from PIL import Image

    from gmlx._vendor.mlx_vlm.prompt_utils import get_chat_template
    from gmlx._vendor.mlx_vlm.utils import (
        prepare_inputs, should_add_special_tokens)
    from gmlx.tui.chat import _vlm_message

    model, config, processor = vlm_load
    tok = processor.tokenizer
    question = "What is in this picture?"
    msgs = [_vlm_message(config["model_type"], question, "user", n_images=1)]
    prompt = get_chat_template(processor, msgs, add_generation_prompt=True)
    img = Image.new("RGB", size, (90, 120, 150))
    # the call the generation path makes.
    inputs = prepare_inputs(
        processor, images=[img], prompts=prompt,
        image_token_index=model.config.image_token_index,
        add_special_tokens=should_add_special_tokens(
            config["model_type"], processor))
    ids = inputs["input_ids"][0].tolist()

    rows, cols = grid
    assert inputs["image_grid_thw"].tolist() == [[1, 2 * rows, 2 * cols]]
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    head = tok.encode("<|im_start|>user\n<|vision_start|>",
                      add_special_tokens=False)
    tail = tok.encode(
        f"<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False)
    assert ids == head + [pad] * (rows * cols) + tail


# The image features of the 8B against llama.cpp.

_FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures"
            / "qwen3vl_8b_main_features_llamacpp.npz")

# The fixture holds every 64th column of the main features llama.cpp's
# mtmd computes for the synthetic image below (196 image tokens), with the
# Q4_K_M text file and the BF16 mmproj of Qwen3-VL-8B-Instruct. Relative L2
# against it over those columns, measured: 0.0155 with the tower in float32
# (0.0180 with MLX_ENABLE_TF32=0) and 0.147 with the tower in bfloat16. The
# bound leaves the float32 tower a factor of 2.2 and is a factor of 3.7
# under the bfloat16 tower.
_FEATURE_REL_L2 = 0.04


def _synthetic_image(side=448):
    """An image by integer arithmetic, the same bytes on every machine: two
    ramps, a checker of 32 pixel squares and a fine diagonal texture."""
    y, x = np.mgrid[0:side, 0:side]
    r = (x * 255) // (side - 1)
    g = (y * 255) // (side - 1)
    b = ((x // 32 + y // 32) % 2) * 160 + (x * 7 + y * 13) % 64
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _synthetic_inputs(vlm_load):
    """The synthetic image through the pair's own processor, with the
    fixture. Skips for a Qwen3-VL of another size."""
    import mlx.core as mx
    from PIL import Image

    model, _, processor = vlm_load
    tower = model.vision_tower
    if (len(tower.blocks), model.config.vision_config.out_hidden_size) != (
            27, 4096):
        pytest.skip("the feature fixture is of Qwen3-VL-8B")
    fixture = np.load(_FIXTURE)
    image = _synthetic_image()
    assert (hashlib.sha256(image.tobytes()).hexdigest()
            == str(fixture["image_sha256"]))
    inputs = processor.image_processor([Image.fromarray(image)])
    pixels = mx.array(inputs["pixel_values"])
    grid = mx.array(inputs["image_grid_thw"])
    assert grid.tolist() == [[1, 28, 28]]
    return pixels, grid, fixture


def _rel_l2(got, want):
    got, want = got.astype(np.float64), want.astype(np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@_NEEDS_GPU
def test_main_image_features_follow_llama_cpp(vlm_load):
    """The main features the container joins to the token embeddings, for
    an image through the pair's own processor, are the ones llama.cpp
    computes for the same bytes. A processor with another rescale, mean or
    std misses the bound, and so does a tower that computes in 16 bits."""
    import mlx.core as mx

    pixels, grid, fixture = _synthetic_inputs(vlm_load)
    model = vlm_load[0]
    entry = model.encode_images(pixels, image_grid_thw=grid)
    main = np.array(entry[0].astype(mx.float32))
    assert main.shape == (196, 4096)
    rel = _rel_l2(main[:, fixture["columns"]], fixture["main"])
    assert rel < _FEATURE_REL_L2, rel
    mx.clear_cache()


@_NEEDS_GPU
def test_a_bfloat16_tower_misses_the_feature_bound(vlm_load):
    """The same tower with every parameter and the pixels in bfloat16, the
    form a load without the float32 rule gives, is off by several times
    the bound. This is why the tower computes in float32."""
    import mlx.core as mx
    from mlx.utils import tree_map

    pixels, grid, fixture = _synthetic_inputs(vlm_load)
    tower = vlm_load[0].vision_tower
    loaded = tower.parameters()
    tower.update(tree_map(lambda p: p.astype(mx.bfloat16), loaded))
    try:
        main, _ = tower(pixels.astype(mx.bfloat16), grid)
        main = np.array(main.astype(mx.float32))
    finally:
        tower.update(loaded)
    rel = _rel_l2(main[:, fixture["columns"]], fixture["main"])
    assert rel > 2 * _FEATURE_REL_L2, rel
    mx.clear_cache()


class _Recorder:
    """A module behind a record of its last input."""

    def __init__(self, module):
        self.module = module
        self.seen = None

    def __call__(self, x):
        self.seen = x
        return self.module(x)


# The merger with the exact GELU against the one with the tanh form, in
# float64 on the real input: 1.14e-4 relative L2 apart. The model's merger
# on the CPU stream is 1.5e-6 from the exact form. The feature bound above
# is far too wide to tell the two forms apart; this check can.
_MERGER_REL_L2 = 2e-5


@_NEEDS_GPU
def test_real_merger_is_the_exact_gelu_merger_in_float64(vlm_load):
    """The main merger on the hidden state the tower hands it, against
    norm, linear, exact GELU, linear in float64 from the same weights."""
    import mlx.core as mx

    pixels, grid, _ = _synthetic_inputs(vlm_load)
    tower = vlm_load[0].vision_tower
    merger = tower.merger
    recorder = _Recorder(merger)
    tower.merger = recorder
    try:
        mx.eval(tower(pixels.astype(mx.float32), grid)[0])
    finally:
        tower.merger = merger
    x = recorder.seen
    # With TF32 on, the GPU's float32 products put the merger 7.9e-4 from
    # the exact form, more than the two forms differ, so the model side
    # runs on the CPU.
    with mx.stream(mx.cpu):
        got = merger(x.astype(mx.float32))
        mx.eval(got)
    got = np.array(got.astype(mx.float32))

    def f64(a):
        return np.array(a.astype(mx.float32)).astype(np.float64)

    h = f64(x)
    mean = h.mean(-1, keepdims=True)
    var = ((h - mean) ** 2).mean(-1, keepdims=True)
    h = (h - mean) / np.sqrt(var + 1e-6)
    h = h * f64(merger.norm.weight) + f64(merger.norm.bias)
    w1 = f64(merger.linear_fc1.weight)
    h = h.reshape(-1, w1.shape[1]) @ w1.T + f64(merger.linear_fc1.bias)
    erf = np.vectorize(math.erf)
    exact = 0.5 * h * (1.0 + erf(h / math.sqrt(2.0)))
    tanh = 0.5 * h * (1.0 + np.tanh(
        math.sqrt(2.0 / math.pi) * (h + 0.044715 * h ** 3)))
    w2, b2 = f64(merger.linear_fc2.weight), f64(merger.linear_fc2.bias)
    assert _rel_l2(got, exact @ w2.T + b2) < _MERGER_REL_L2
    assert _rel_l2(tanh @ w2.T + b2, exact @ w2.T + b2) > 3 * _MERGER_REL_L2
    mx.clear_cache()
