"""Anchors through the VAEs: the audio encode's convolutions windowed with
their halo, the video encode in windows of whole pieces, and a guide's
audio read only as far as its kept latents need, each against the whole
encode, and an anchor's frames and waveform as the VAEs take them. Seeded
tiny weights, no files."""
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


@pytest.fixture(scope="module")
def video_vae(tmp_path_factory):
    """The seeded tiny video VAE of check 5's self-test."""
    from test_video_check5_h3 import seeded
    from video_oracle_fixture import committed

    from gmlx.video.vae.reader import load_vae

    shapes = committed("selftest_h3_vae")["inputs"]["shapes"]
    path = str(tmp_path_factory.mktemp("h3_vae") / "vae.safetensors")
    seeded.write_safetensors(path, seeded.weights(shapes["vae"]))
    return load_vae(path, family="h3", dtype=mx.float32)


@pytest.mark.parametrize("pieces", [1, 2])
def test_windows_of_whole_pieces_are_the_whole_video_encode(video_vae,
                                                            pieces):
    """Each 17-frame piece encodes alone and the clip's end drops 3
    latents, so 56 frames in open windows of whole pieces, kept whole,
    and the clip's last 22 frames give the whole encode bit for bit."""
    rng = np.random.default_rng(3)
    x = mx.array(rng.uniform(-1, 1, (1, 56, 32, 32, 3)).astype(np.float32))
    whole = video_vae.encode(x)
    parts = [video_vae.encode(x[:, a:a + 17 * pieces], drop=False)
             for a in range(0, 34, 17 * pieces)]
    parts.append(video_vae.encode(x[:, 34:]))
    assert [p.shape[2] for p in parts] == [5 * pieces] * (2 // pieces) + [7]
    assert whole.shape[2] == 17
    np.testing.assert_array_equal(np.array(mx.concatenate(parts, axis=2)),
                                  np.array(whole))


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


@pytest.mark.parametrize("fits", [True, False])
def test_a_jobs_guide_past_one_window_encodes_in_windows_as_the_whole(
        video_vae, monkeypatch, fits):
    """A video job's guide clip goes through ffmpeg a frame at a time:
    when the whole clip does not fit, in open windows of the pieces that
    do and the clip's end, which give the whole encode; the frames read
    are dropped once encoded."""
    from types import SimpleNamespace

    from gmlx.serve import video_jobs as vj
    from gmlx.serve.patches import image_steps as pis
    from gmlx.video import images, io
    from gmlx.video.layout import AnchorRows
    from gmlx.video.vae import reader

    rng = np.random.default_rng(5)
    clip = rng.integers(0, 256, (56, 32, 32, 3), dtype=np.uint8)
    held = []

    def each_frame(path, *, most, ffmpeg, each):
        for f in clip[:most]:
            each(f)
            held.append(1)
        return most

    monkeypatch.setattr(io, "each_guide_frame", each_frame)
    monkeypatch.setattr(images, "cover", lambda f, w, h: f)
    monkeypatch.setattr(reader, "load_vae", lambda *a, **k: video_vae)
    sizes = []

    def fit(hold, served, *, n, width, height, level, is_open):
        sizes.append((n, level, is_open))
        return (0, 1, 3) if fits else (2, 1, 1)

    monkeypatch.setattr(pis, "_window_fit", fit)
    monkeypatch.setattr(pis, "_check_clip", lambda *a, **k: None)
    encoded = []
    real = video_vae.encode
    monkeypatch.setattr(video_vae, "encode", lambda x, drop=True: (
        encoded.append((x.shape[1], drop)) or real(x, drop=drop)))
    placed = SimpleNamespace(frame=0, video_frames=56, audio_t=None)
    t = SimpleNamespace(video="guide.mp4", ffmpeg="ffmpeg", still=None,
                        wave=None, placed=placed, label="guides[0] (g)",
                        rows=AnchorRows(0, 17, 0))
    job = SimpleNamespace(stop=SimpleNamespace(is_set=lambda: False),
                          params={"settings": {"width": 32, "height": 32,
                                               "frames": 56}})
    served = SimpleNamespace(spec=SimpleNamespace(vae="vae.safetensors",
                                                  audio_vae=None))
    got = vj.encode_anchors(job, None, served, [t])
    assert sizes == [(56, 1, False)]
    assert encoded == ([(56, True)] if fits else
                       [(17, False), (17, False), (17, False), (5, True)])
    whole = real((mx.array(clip).astype(mx.float32) / 255.0)[None]
                 * 2.0 - 1.0)
    (frame, video, audio), = got["anchors"]
    assert (frame, audio) == (0, None)
    np.testing.assert_array_equal(video, np.array(whole))
