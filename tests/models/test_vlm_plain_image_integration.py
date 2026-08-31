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

pytest.importorskip("mlx_vlm")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

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
def test_plain_load_arms_tiled_v_on_both_modules(plain_vlm_load):
    """The vendored mlx-vlm gated_delta module is rebound by a plain load."""
    import importlib

    import gmlx.upstream.gdn_patches as gp

    vgd = importlib.import_module("mlx_vlm.models.qwen3_5.gated_delta")
    assert gp._tiled_v_patch_applied()
    # Qualname/module, not identity against gd: a later load can re-close
    # the mlx-lm patch, leaving vgd on an older (equally tiled) instance.
    # The kernel rebind points at mlx-lm's dispatch, which reads the
    # patched (tiled) private kernels at call time.
    assert "_tiled_gated_delta_ops" in vgd.gated_delta_ops.__qualname__
    assert vgd.gated_delta_kernel.__module__ == "mlx_lm.models.gated_delta"
    assert vgd._gated_delta_with_states_kernel is None
    assert vgd._gated_delta_with_states_ops is gp._tiled_gd_with_states_ops


@_NEEDS_GPU
def test_plain_image_turn_greedy_does_not_collapse(plain_vlm_load):
    """Greedy image turn runs to the cap instead of degrading to an early
    eos. The broken K->V mapping produced repetitive prose and an eos
    around 40 tokens; a healthy model asked for great detail does not stop
    inside 64."""
    from PIL import Image

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

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
