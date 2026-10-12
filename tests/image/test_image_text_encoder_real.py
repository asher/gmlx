"""The Z-Image text encoder on real weights: a Qwen3-4B GGUF passes the
header and identity checks, the tap after the last layer equals the
forward of the copied mlx-lm Qwen3 class before its final norm, and release
drops the model.
Skipped unless KQUANT_TEST_GGUF_DIR holds a GGUF that passes the Z-Image
encoder checks."""
from __future__ import annotations

import pytest

import mlx.core as mx

pytestmark = pytest.mark.integration


def _encoder_file(gguf_index: dict) -> str:
    from gmlx.image import text_encoder as te

    for p in gguf_index.get(te.Z_IMAGE.arch, []):
        try:
            te.check_encoder_file(te._header_view(p), te.Z_IMAGE)
        except te.EncoderMismatchError:
            continue
        return p
    pytest.skip("no Z-Image text encoder GGUF under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def handle(gguf_index):
    from gmlx.image import text_encoder as te

    h = te.load_encoder(_encoder_file(gguf_index), te.Z_IMAGE,
                        log=lambda *_: None)
    yield h
    h.release()


def test_encode_shapes_and_template(handle):
    got = handle.encode("a red bicycle")
    (hidden,) = got.hidden
    assert hidden.shape == (1, len(got.ids), 2560)
    assert hidden.dtype in (mx.bfloat16, mx.float16)
    assert bool(mx.all(mx.isfinite(hidden)))
    tok = handle.tokenizer
    assert tok.decode(list(got.ids)) == (
        "<|im_start|>user\na red bicycle<|im_end|>\n<|im_start|>assistant\n")


def test_tap_after_every_layer_is_the_stock_forward_before_its_norm(handle):
    from gmlx.image import text_encoder as te

    m = handle.model
    ids = list(handle.encode("two cats on a sofa").ids)
    (last,) = te.tap(m, ids, layers=(36,))
    stock = m.model(mx.array([ids]))
    normed = m.model.norm(last)
    assert mx.array_equal(normed, stock).item()


def test_release_drops_the_last_reference_to_the_model(gguf_index):
    import gc
    import weakref

    from gmlx.image import text_encoder as te

    h = te.load_encoder(_encoder_file(gguf_index), te.Z_IMAGE,
                        log=lambda *_: None)
    h.encode("a")
    ref = weakref.ref(h.model)
    assert h.release() is True
    gc.collect()
    assert ref() is None
