"""A real Qwen-Image 2.1 DiT GGUF through load_dit: one KQuantLinear per
quantized matrix, the fp32 norm leaves, and one small prefill and cached
step with and without a reference image, whose prefix cache has the bytes
the memory plan counts. Skipped unless KQUANT_TEST_GGUF_DIR holds a
Qwen-Image 2.1 DiT GGUF (arch qwen_image)."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

pytestmark = pytest.mark.integration


def _qwen_file(gguf_index: dict) -> tuple[str, object]:
    from gmlx.image import fingerprint as fp
    from gmlx.load.headerscan import scan_gguf

    for p in gguf_index.get("qwen_image", []):
        try:
            got = fp.identify(fp.view_from_scans([scan_gguf(p)]))
        except ValueError:
            continue
        if got.family.name == "qwen_image21":
            return p, got
    pytest.skip("no Qwen-Image 2.1 DiT GGUF under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def loaded(gguf_index):
    path, ident = _qwen_file(gguf_index)
    from gmlx.image.loader import load_dit

    model, info = load_dit(path, log=lambda *_: None)
    return ident, model, info


def test_quantized_leaf_count_and_fp32_norms(loaded):
    ident, model, info = loaded
    floats = ("F32", "F16", "BF16")
    tensors = ident.view.tensors
    quant = sum(1 for n, t in tensors.items() if len(t.shape) == 2
                and t.codec not in floats and "norm" not in n)
    assert info.n_kquant == quant
    assert info.n_owned_casts == sum(1 for n, t in tensors.items()
                                     if "norm" in n and t.codec != "F32")
    assert model.act_dtype == mx.bfloat16
    assert len(model.transformer_blocks) == 32
    blk = model.transformer_blocks[0]
    for leaf in (model.txt_in.text_norm, model.norm_out.linear,
                 blk.attn.norm_q, blk.attn.norm_k):
        assert leaf.weight.dtype == mx.float32


# name: (text rows, image slots, reference shape in latent tokens)
STEPS = {"text": (24, (), ()),
         "ref1": (40, tuple(range(10, 18)), ((4, 8),))}


@pytest.mark.parametrize("name", list(STEPS))
def test_one_small_cached_step_is_finite(loaded, name):
    from gmlx.image import memfit
    from gmlx.image.conditioning import Conditioning, array_bytes
    from gmlx.image.families.qwen_image21 import model as qm

    _, model, _ = loaded
    rows, slots, refs = STEPS[name]
    cfg = model.cfg
    rng = np.random.default_rng(0)

    def normal(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    cond = Conditioning(
        normal(rows, cfg.context_dim), name,
        ref_latents=tuple(normal(cfg.in_channels, h, w) for h, w in refs),
        image_slots=slots)
    prefix = qm.prefill_prefix(model, cond)
    tokens = rows - len(slots) + sum(h * w for h, w in refs)
    assert prefix.tokens == tokens
    row = memfit.PEAKS["qwen_image21"].prefix
    assert array_bytes(prefix.arrays()) == row.cache_bytes(tokens, 2)
    lat = normal(1, cfg.in_channels, 16, 16)
    outs = [qm.forward_target(model, lat, s, cond, prefix)
            for s in (0.8, 0.3)]
    mx.eval(outs)
    for out in outs:
        assert out.shape == lat.shape and out.dtype == mx.float32
        assert bool(mx.all(mx.isfinite(out)))
    assert not np.array_equal(np.array(outs[0]), np.array(outs[1]))
