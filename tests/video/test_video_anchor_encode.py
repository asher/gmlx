"""Anchors through the VAEs: the audio encode's convolutions windowed with
their halo and a guide's audio read only as far as its kept latents need,
both against the whole encode, and an anchor's frames and waveform as
the VAEs take them. Seeded tiny weights, no files."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from gmlx.video import anchors
from gmlx.video.vae import h3_audio

SECONDS = 4


@pytest.fixture(scope="module")
def audio_vae():
    """The H3 audio VAE's rates and kernels at a tiny width."""
    cfg = h3_audio.Config(encoder_dim=4, latent_dim=32, z=8, heads=2,
                          decoder_dim=128)
    m = h3_audio.H3AudioVAE(cfg)
    rng = np.random.default_rng(0)
    params = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith(".alpha"):
            a = 1.0 + 0.1 * rng.standard_normal(v.shape)
        elif "norm" in k:
            a = np.ones(v.shape) if k.endswith("weight") else \
                np.zeros(v.shape)
        else:
            fan = int(np.prod(v.shape[1:])) if v.ndim > 1 else 1
            a = rng.standard_normal(v.shape) / np.sqrt(fan)
        params.append((k, mx.array(a.astype(np.float32))))
    m.load_weights(params)
    m.eval()
    return m


def _wave(n: int, seed: int = 1) -> mx.array:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / h3_audio.SAMPLE_RATE
    w = 0.5 * np.sin(2 * np.pi * np.array([[220.0], [330.0]]) * t) \
        + 0.1 * rng.standard_normal((2, n))
    return mx.array(w[None].astype(np.float32))


def test_the_halo_is_the_reach_of_the_convolutions():
    """9317 samples before a latent and 9125 after it, 12 latents each
    side; the shipped widths have the tiny model's kernels and rates."""
    enc = h3_audio.Encoder(h3_audio.Config())
    assert h3_audio.reach(enc) == (9317, 9125)


def test_the_windowed_encode_is_the_whole_encode(audio_vae):
    """4 s and a part latent, its convolutions 40 latents at a time,
    within 1e-5 of them all at once."""
    w = _wave(SECONDS * h3_audio.SAMPLE_RATE + 123)
    assert audio_vae.halo == 12
    whole = audio_vae.encode(w, window=None)
    got = audio_vae.encode(w, window=40)
    assert got.shape == whole.shape == (1, 8, 2, SECONDS * 40 + 1)
    np.testing.assert_allclose(np.array(got), np.array(whole), rtol=0,
                               atol=1e-5)
    assert float(mx.abs(whole).max()) > 0.1


def test_the_first_latents_need_their_samples_and_the_halo_alone(
        audio_vae):
    """A guide's audio is cut to the latents left on the clip: reading its
    first 37 latents and 12 more gives the whole 4 s encode's first 37."""
    w = _wave(SECONDS * h3_audio.SAMPLE_RATE)
    whole = audio_vae.encode(w, window=None)[..., :37]
    need = audio_vae.samples_for(37)
    assert need == (37 + 12) * h3_audio.HOP
    got = audio_vae.encode(w[..., :need], window=None)[..., :37]
    np.testing.assert_allclose(np.array(got), np.array(whole), rtol=0,
                               atol=1e-5)


class _Seen:
    """A video VAE that keeps what it is given."""

    def __init__(self):
        self.seen = None

    def encode(self, x):
        self.seen = x
        return mx.zeros((1, 24, 1, 2, 2))


def test_an_anchor_takes_its_kept_frames_and_latents(audio_vae):
    """Its kept frames in [-1, 1] as ComfyUI's VAE takes [0, 1] pixels,
    and its audio as the whole encode cut to its latents."""
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, (9, 32, 32, 3), dtype=np.uint8)
    wave = _wave(SECONDS * h3_audio.SAMPLE_RATE)
    seen = _Seen()
    a = anchors.encode(anchors.Placement(3, 5, 7), frames=frames,
                       waveform=wave[0], video_vae=seen,
                       audio_vae=audio_vae)
    assert a.frame == 3
    want = (frames[:5].astype(np.float32) / 255.0)[None] * 2.0 - 1.0
    np.testing.assert_array_equal(np.array(seen.seen), want)
    whole = audio_vae.encode(wave, window=None)[..., :7]
    assert a.audio.shape == (1, 8, 2, 7)
    np.testing.assert_allclose(np.array(a.audio), np.array(whole), rtol=0,
                               atol=1e-5)
    only = anchors.encode(anchors.Placement(0, 0, 3), waveform=wave[0],
                          audio_vae=audio_vae)
    assert only.video is None and only.audio.shape[-1] == 3


def test_anchor_inputs_of_the_wrong_form_are_refused(audio_vae):
    p = anchors.Placement(0, 5)
    with pytest.raises(ValueError, match="keeps 5 frames, and fewer were "
                                         "given"):
        anchors.encode(p, frames=np.zeros((4, 32, 32, 3), np.uint8),
                       video_vae=_Seen())
    with pytest.raises(ValueError, match=r"\(T, H, W, 3\) uint8, not "
                                         r"mlx.core.float32"):
        anchors.encode(p, frames=np.zeros((5, 32, 32, 3), np.float32),
                       video_vae=_Seen())
    with pytest.raises(ValueError, match=r"\(2, samples\), not \(1, 800\)"):
        anchors.encode(anchors.Placement(0, 0, 1),
                       waveform=np.zeros((1, 800), np.float32),
                       audio_vae=audio_vae)
    with pytest.raises(ValueError, match="keeps audio, and no waveform was "
                                         "given"):
        anchors.encode(anchors.Placement(0, 0, 1), audio_vae=audio_vae)
