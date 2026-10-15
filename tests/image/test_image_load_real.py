"""A real Z-Image DiT GGUF through load_dit: the variant from the identity
hash, zero-copy views, one KQuantLinear per quantized matrix, the owned
casts, and one small forward. Skipped unless KQUANT_TEST_GGUF_DIR holds a
Z-Image DiT GGUF (arch lumina2)."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = [pytest.mark.integration, pytest.mark.host_gb(16)]


def _z_image_file(gguf_index: dict) -> tuple[str, object]:
    from gmlx.image import fingerprint as fp
    from gmlx.load.headerscan import scan_gguf

    for p in gguf_index.get("lumina2", []):
        try:
            got = fp.identify(fp.view_from_scans([scan_gguf(p)]))
        except ValueError:
            continue
        if got.family.name == "z_image":
            return p, got
    pytest.skip("no Z-Image DiT GGUF under KQUANT_TEST_GGUF_DIR")


@pytest.fixture(scope="module")
def loaded(gguf_index):
    path, ident = _z_image_file(gguf_index)
    from gmlx.image.loader import load_dit

    model, info = load_dit(path, log=lambda *_: None)
    return path, ident, model, info


def test_variant_and_quantized_leaf_count(loaded):
    _, ident, _, info = loaded
    quant = sum(1 for t in ident.view.tensors.values()
                if len(t.shape) == 2 and t.codec not in ("F32", "F16", "BF16"))
    assert info.n_kquant == quant
    assert info.identified.variant in ("turbo", "base")
    assert info.identified.identity_hash in ident.family.variants


def test_owned_casts_and_zero_copy_matrices(loaded):
    _, ident, model, _ = loaded
    assert model.t_embedder.mlp[0].weight.dtype == mx.float32
    assert model.t_embedder.mlp[2].weight.dtype == mx.float32
    assert model.cap_embedder[0].weight.dtype == mx.float32
    assert model.x_pad_token.shape == (1, 3840)
    assert model.x_pad_token.dtype == mx.bfloat16
    assert model.layers[0].attention_norm1.weight.dtype == mx.bfloat16
    if ident.view.tensors["x_embedder.weight"].codec == "BF16":
        assert model.x_embedder.weight.dtype == mx.bfloat16


def test_one_small_step_is_finite(loaded):
    _, _, model, _ = loaded
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 16, 32, 32)).astype(np.float32))
    cap = mx.array(rng.standard_normal((1, 7, 2560)).astype(np.float32))
    out = model(lat, 0.8, cap)
    mx.eval(out)
    assert out.shape == (1, 16, 32, 32) and out.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(out)))
