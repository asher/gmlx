"""Pause points of the image forwards: a pacer that evaluates at every
point between blocks sees one point per block, and the velocity and the
decoded image equal those of the forward without a pacer, bit for bit.
Tiny models on the CPU stream."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

import qwen_tiny as qt
from image_fixtures import tiny_z_model

from gmlx.image import pacing


class _Evaluating:
    def __init__(self):
        self.points = 0
        self.arrays = 0

    def __call__(self, arrays):
        self.points += 1
        self.arrays += len(arrays)
        if arrays:
            mx.eval(*arrays)


def test_z_image_forward_pauses_per_block_with_the_same_velocity():
    m, _ = tiny_z_model()
    rng = np.random.default_rng(3)
    lat = mx.array(rng.standard_normal((1, 4, 12, 10)).astype(np.float32))
    cap = mx.array(rng.standard_normal((1, 5, 16)).astype(np.float32))
    with mx.stream(mx.cpu):
        plain = m(lat, 0.6, cap)
        mx.eval(plain)
        pacer = _Evaluating()
        with pacing.paced(pacer):
            paced = m(lat, 0.6, cap)
        mx.eval(paced)
    assert pacer.points == len(m.noise_refiner) + len(m.layers)
    assert pacer.arrays == pacer.points
    assert np.array_equal(np.array(plain), np.array(paced))


def test_qwen_image_prefill_and_target_pause_per_block_with_the_same_velocity():
    from gmlx.image.families.qwen_image21 import model as qm

    model = qt.model_of(qt.weights())
    cond = qt.conditioning("text")
    lat = qt.latents()
    n = len(model.transformer_blocks)
    with mx.stream(mx.cpu):
        prefix = qm.prefill_prefix(model, cond)
        plain = qm.forward_target(model, lat, 0.6, cond, prefix)
        mx.eval(plain)
        pacer = _Evaluating()
        with pacing.paced(pacer):
            prefix2 = qm.prefill_prefix(model, cond)
            paced = qm.forward_target(model, lat, 0.6, cond, prefix2)
        mx.eval(paced)
    # A bare point after each evaluated prefill block, then one per target
    # block with the rows it carries.
    assert pacer.points == 2 * n and pacer.arrays == n
    assert np.array_equal(np.array(plain), np.array(paced))


def _tiny_vae(family: str):
    if family == "z_image":
        from gmlx.image.vae import flux1_ae

        vae = flux1_ae.Flux1AE(flux1_ae.Flux1AEConfig(
            in_channels=3, out_channels=3, latent_channels=16,
            block_out_channels=(32, 64), layers_per_block=1,
            norm_num_groups=32, eps=1e-6))
        return vae, 16, len(vae.decoder.up_blocks)
    import qwen_vae_fixtures as qv

    vae = qv.build(qv.tiny_config())
    return vae, vae.config.latent_channels, len(vae.decoder.up_blocks)


@pytest.mark.parametrize("family", ["z_image", "qwen_image21"])
def test_vae_decode_pauses_per_up_block_with_the_same_image(family):
    vae, c, blocks = _tiny_vae(family)
    rng = np.random.default_rng(5)
    z = mx.array(rng.standard_normal((1, 4, 4, c)).astype(np.float32))
    with mx.stream(mx.cpu):
        plain = vae.decode(z)
        mx.eval(plain)
        pacer = _Evaluating()
        with pacing.paced(pacer):
            paced = vae.decode(z)
        mx.eval(paced)
    assert pacer.points == blocks
    assert np.array_equal(np.array(plain), np.array(paced))
