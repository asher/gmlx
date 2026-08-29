"""Real-weights VLM integration: mmproj pairing + one image turn + one
audio turn.

``integration`` + ``slow``; needs ``KQUANT_TEST_GGUF_DIR`` to contain a chat
model with a sibling mmproj GGUF that discovery pairs to it (the same pairing
``serve`` config discovery does in production). The media-turn tests then load
the pair through ``load_vlm_model`` and answer one prompt about the bundled
test image / audio clip. The audio turn additionally needs the paired mmproj
to carry an audio encoder (``clip.has_audio_encoder`` or ``clip.audio.*``
metadata - Qwen omni or gemma-4 E-series/unified projectors); a vision-only
projector skips it. Skips carry the exact missing piece in their reason.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# A 30B-class VLM forward is a GPU workload; the forced-CPU stream keeps to
# logic tests.
_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="real-weights VLM forward is a GPU workload")

CATS = Path(__file__).parents[1] / "e2e" / "assets" / "cats.jpg"
SPEECH = Path(__file__).parents[1] / "e2e" / "assets" / "speech.wav"


def _paired_models(gguf_dir):
    from gmlx.config import DiscoverSpec
    from gmlx.load.discovery import scan_dirs

    spec = DiscoverSpec(
        dir=None, recursive=True, pair_mmproj=True, speculative="auto"
    )
    return [m for m in scan_dirs([spec], [str(gguf_dir)]) if m.mmproj]


def _mmproj_has_audio(mmproj_path: str) -> bool:
    from gmlx.load.headerscan import scan_gguf

    kv = scan_gguf(mmproj_path, include_tensors=False).kv
    return bool(kv.get("clip.has_audio_encoder")) or any(
        k.startswith("clip.audio.") for k in kv
    )


@pytest.fixture(scope="module")
def vlm_pair(gguf_dir):
    """(model_path, mmproj_path) from discovery over the test dir."""
    paired = _paired_models(gguf_dir)
    if not paired:
        pytest.skip(
            "no model with a paired sibling mmproj under KQUANT_TEST_GGUF_DIR"
        )
    return paired[0]


@pytest.fixture(scope="module")
def audio_pair(gguf_dir):
    """A paired model whose mmproj carries an audio encoder."""
    paired = _paired_models(gguf_dir)
    if not paired:
        pytest.skip(
            "no model with a paired sibling mmproj under KQUANT_TEST_GGUF_DIR"
        )
    audio = [m for m in paired if _mmproj_has_audio(m.mmproj)]
    if not audio:
        pytest.skip(
            "no paired mmproj with an audio encoder under KQUANT_TEST_GGUF_DIR "
            "(vision-only projectors: "
            + ", ".join(os.path.basename(m.mmproj) for m in paired)
            + ")"
        )
    return audio[0]


def test_discovery_pairs_sibling_mmproj(vlm_pair):
    assert os.path.isfile(vlm_pair.path)
    assert vlm_pair.mmproj and os.path.isfile(vlm_pair.mmproj)
    assert "mmproj" in os.path.basename(vlm_pair.mmproj).lower()
    # Named projectors pair by id-prefix; the pair must share a directory.
    assert os.path.dirname(vlm_pair.mmproj) == os.path.dirname(vlm_pair.path)


@_NEEDS_GPU
def test_vlm_image_turn(vlm_pair):
    if not CATS.is_file():
        pytest.skip(f"bundled test image missing: {CATS}")
    import mlx.core as mx

    from gmlx.tui.chat import _vlm_message
    from gmlx.load.vlm import load_vlm_model

    model, config, processor = load_vlm_model(
        vlm_pair.path, vlm_pair.mmproj, return_tokenizer=False
    )
    model_type = getattr(config, "model_type", None) or (
        config.get("model_type") if isinstance(config, dict) else ""
    )

    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import get_chat_template

    msgs = [_vlm_message(model_type, "What animals are in this picture?",
                         "user", n_images=1)]
    # enable_thinking=False: a thinking-mode pair (qwen3.6 templates) burns
    # the whole token budget inside its reasoning preamble and never names
    # the animals; templates without the knob ignore it.
    prompt = get_chat_template(processor, msgs, add_generation_prompt=True,
                               enable_thinking=False)

    chunks = []
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        image=[str(CATS)],
        max_tokens=24,
        temperature=0.0,
    ):
        chunks.append(chunk.text)
    reply = "".join(chunks).strip()
    assert reply, "image turn produced no text"
    # Content-bearing check: a non-empty reply alone passes even when the
    # image never reaches the encoder. Greedy on cats.jpg must mention the
    # cats; couples the test to checkpoint competence, which a curated
    # test dir provides. Word-boundary match: "locate"/"indicates" carry
    # the bare substring.
    assert re.search(r"\bcats?\b", reply.lower()), (
        f"image not grounded in reply: {reply!r}"
    )
    mx.clear_cache()


@_NEEDS_GPU
def test_vlm_audio_turn(audio_pair):
    if not SPEECH.is_file():
        pytest.skip(f"bundled test clip missing: {SPEECH}")
    import mlx.core as mx

    from gmlx.tui.chat import _vlm_message
    from gmlx.load.vlm import load_vlm_model

    model, config, processor = load_vlm_model(
        audio_pair.path, audio_pair.mmproj, return_tokenizer=False
    )
    model_type = getattr(config, "model_type", None) or (
        config.get("model_type") if isinstance(config, dict) else ""
    )

    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import get_chat_template

    msgs = [_vlm_message(model_type, "What is said in this recording?",
                         "user", n_audios=1)]
    prompt = get_chat_template(processor, msgs, add_generation_prompt=True)

    chunks = []
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        audio=[str(SPEECH)],
        max_tokens=24,
        temperature=0.0,
    ):
        chunks.append(chunk.text)
    reply = "".join(chunks).strip()
    assert reply, "audio turn produced no text"
    # The clip says "peanut butter and jelly sandwich"; a grounded reply
    # mentions at least one phrase word even when it paraphrases.
    words = ("peanut", "butter", "jelly", "sandwich")
    assert any(w in reply.lower() for w in words), (
        f"audio not grounded in reply: {reply!r}"
    )
    mx.clear_cache()
