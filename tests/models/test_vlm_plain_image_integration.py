"""Real-weights coverage of the PLAIN (non-MTP) VLM image path.

The raw --image path loads through ``load_vlm_model`` with no drafter and
runs mlx-vlm's stock qwen3_5 classes. Since mlx-vlm 0.6.15 vendored the
gated_delta functions (0.6.4 from-imported mlx-lm's), the mlx-lm tiled-V
patch alone no longer reaches this path: without the mlx-vlm-side rebind
the recurrent state is built against the wrong K heads and an image turn
degrades to repetitive text with an early eos, while staying superficially
grounded. This file pins both the seam (module globals rebound after a
plain load) and the symptom (greedy image turn does not eos early).

``integration`` + ``slow``; needs ``KQUANT_TEST_GGUF_DIR`` with a qwen3.x
model paired to a sibling mmproj GGUF (same fixture family as
test_vlm_mtp_integration).
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="real-weights VLM image turn is a GPU workload")


def _qwen_vlm_pair(gguf_dir):
    from gmlx.config import DiscoverSpec
    from gmlx.load.discovery import scan_dirs
    from gmlx.load.headerscan import scan_gguf

    spec = DiscoverSpec(
        dir=None, recursive=True, pair_mmproj=True, speculative="auto"
    )
    paired = [m for m in scan_dirs([spec], [str(gguf_dir)]) if m.mmproj]
    for m in paired:
        kv = scan_gguf(m.path, include_tensors=False).kv
        if str(kv.get("general.architecture", "")).startswith("qwen3"):
            return m
    pytest.skip(
        "no qwen3.x model with a paired sibling mmproj under "
        "KQUANT_TEST_GGUF_DIR"
    )


@pytest.fixture(scope="module")
def plain_vlm_load(gguf_dir):
    from gmlx.load.vlm import load_vlm_model

    pair = _qwen_vlm_pair(gguf_dir)
    model, config, processor = load_vlm_model(pair.path, pair.mmproj)
    return model, config, processor


@_NEEDS_GPU
def test_plain_load_gives_the_model_the_tiled_head_order(plain_vlm_load):
    """A plain load patches the mlx-lm module and gives each gated delta
    layer the tiled order of its GGUF; the scan module the copied Qwen3.5
    path shares keeps its own functions."""
    import importlib

    import gmlx.upstream.gdn_patches as gp

    model = plain_vlm_load[0]
    vgd = importlib.import_module("gmlx.models.qwen35.gated_delta")
    assert gp._tiled_v_patch_applied()
    layers = [layer.linear_attn for layer in model.language_model.model.layers
              if layer.is_linear]
    assert layers and all(layer.gdn_kv_tiled for layer in layers)
    assert model.language_model.args.kv_head_layout == "tiled"
    for name in ("gated_delta_ops", "gated_delta_kernel"):
        assert getattr(vgd, name).__module__ == vgd.__name__


@_NEEDS_GPU
def test_plain_image_turn_greedy_does_not_collapse(plain_vlm_load):
    """Greedy image turn runs to the cap instead of degrading to an early
    eos. The broken K->V mapping produced repetitive prose and an eos
    around 40 tokens; a healthy model asked for great detail does not stop
    inside 64."""
    from PIL import Image

    from gmlx._vendor.mlx_vlm import generate
    from gmlx._vendor.mlx_vlm.prompt_utils import apply_chat_template

    model, config, processor = plain_vlm_load

    img = Image.new("RGB", (128, 128))
    img.putdata(
        [(2 * x, 2 * y, 128) for y in range(128) for x in range(128)]
    )

    prompt = apply_chat_template(
        processor,
        config,
        [{"role": "user", "content": "Describe this image in great detail."}],
        num_images=1,
    )
    result = generate(
        model, processor, prompt, image=[img],
        max_tokens=64, temperature=0.0, verbose=False,
    )
    assert getattr(result, "finish_reason", None) == "length", (
        result.finish_reason,
        result.text,
    )
