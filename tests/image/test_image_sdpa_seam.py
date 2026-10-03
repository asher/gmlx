"""The image SDPA seam calls stock MLX, even after a text model has
installed its process-wide SDPA wrapper on mx.fast."""
from __future__ import annotations

import numpy as np

import mlx.core as mx

from gmlx.image import ops
from gmlx.image.ops import attention
from gmlx.upstream import attn_hd512


def test_the_image_seam_skips_the_text_sdpa_wrapper(monkeypatch):
    monkeypatch.setenv("GMLX_HD512", "1")
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention",
                        mx.fast.scaled_dot_product_attention)
    monkeypatch.setattr(attn_hd512, "_installed", False)
    monkeypatch.setattr(attn_hd512, "_orig_sdpa", attn_hd512._orig_sdpa)
    assert attn_hd512.install_hd512_sdpa()
    routed = []
    monkeypatch.setattr(attn_hd512, "_route",
                        lambda name, *_: routed.append(name))
    rng = np.random.default_rng(0)
    q, k, v = (mx.array(rng.standard_normal((1, 2, 8, 32)), mx.float32)
               for _ in range(3))
    out = ops.sdpa(q, k, v)
    mx.eval(out)
    assert routed == []
    want = attn_hd512._orig_sdpa(q, k, v, scale=32 ** -0.5)
    assert np.array_equal(np.array(out), np.array(want))
    # The wrapper is live for every other caller.
    mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0))
    assert routed == ["stock"]
    assert attention._HD512_FLAG == attn_hd512._FLAG
