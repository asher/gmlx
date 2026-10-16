"""A sampled VAE encode of every family's autoencoder on the CPU: the
noise comes in the layout of the model-space latents, and the encode adds
it at the standard deviation of the encoder's distribution. Tiny VAEs on
their initial weights. A tiled encode adds the noise as a whole one
does, after the same lines."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image.vae import tiling


def _vae(family: str):
    """A tiny VAE of the family's class on its initial weights, in
    float32, and the side of the smallest image whose latents take the
    model's layout."""
    if family == "z_image":
        from gmlx.image.vae import flux1_ae

        vae = flux1_ae.Flux1AE(flux1_ae.Flux1AEConfig(
            in_channels=3, out_channels=3, latent_channels=16,
            block_out_channels=(32, 64), layers_per_block=1,
            norm_num_groups=32, eps=1e-6, scaling_factor=0.3611,
            shift_factor=0.1159))
        return vae, 2
    if family == "flux3_klein":
        from gmlx.image.vae import flux2_ae

        vae = flux2_ae.Flux2AE(flux2_ae.Flux2AEConfig(
            in_channels=3, out_channels=3, latent_channels=8,
            block_out_channels=(32, 64), layers_per_block=1,
            norm_num_groups=32, eps=1e-6))
        rng = np.random.default_rng(3)
        n = vae.config.model_channels
        vae.bn.running_mean = mx.array(rng.standard_normal(n), mx.float32)
        vae.bn.running_var = mx.array(rng.uniform(0.5, 2.0, n), mx.float32)
        return vae, 4
    if family == "krea2":
        from gmlx.image.vae import qwen_image_vae as q

        vae = q.QwenImageVAE(q.QwenImageVAEConfig(
            in_channels=3, out_channels=3, latent_channels=16, base_dim=4,
            dim_mult=(1, 2), num_res_blocks=1, temporal=(False,)))
        return vae, 2
    import qwen_vae_fixtures as qv

    return qv.build(qv.tiny_config()), 2


FAMILIES = ["z_image", "flux3_klein", "krea2", "qwen_image21"]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_sampled_encode_adds_the_noise_at_the_deviation(family):
    """The model-space latents less those of the mean are the noise times
    the deviation carried to model space. The maps to model space are
    affine, so that deviation is the map of the mean plus the deviation
    less the map of the mean."""
    vae, unit = _vae(family)
    s = vae.config.scale * unit
    rng = np.random.default_rng(4)
    x = mx.array(rng.uniform(-1, 1, (1, 6 * s, 4 * s, 3)).astype(np.float32))
    # The plan of the pass is not under test.
    kw = dict(tiled=False, free=1e12)
    with mx.stream(mx.cpu):
        mode = tiling.encode(vae, x, **kw)
        noise = rng.standard_normal(mode.shape).astype(np.float32)
        got = tiling.encode(vae, x, mode="sample", noise=noise, **kw)
        zero = tiling.encode(vae, x, mode="sample",
                             noise=np.zeros_like(noise), **kw)
        mean, logvar = vae.encode(x)
        std = mx.exp(0.5 * logvar.astype(mx.float32))
        dev = (vae.to_model_space(mean + std) - vae.to_model_space(mean))
        mx.eval(mode, got, zero, dev)
    assert got.shape == mode.shape
    assert mx.array_equal(zero, mode).item()
    step = np.array(got) - np.array(mode)
    want = np.array(dev) * noise
    np.testing.assert_allclose(step, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("family", FAMILIES)
def test_noise_of_another_shape_is_refused(family):
    vae, unit = _vae(family)
    s = vae.config.scale * unit
    x = mx.zeros((1, 2 * s, 2 * s, 3))
    with mx.stream(mx.cpu):
        mode = tiling.encode(vae, x, free=1e12)
        b, c, h, w = mode.shape
        with pytest.raises(ValueError, match="noise"):
            tiling.encode(vae, x, mode="sample", free=1e12,
                          noise=np.zeros((b, c, h + 1, w), np.float32))
