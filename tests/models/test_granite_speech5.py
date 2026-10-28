"""Granite Speech 5.0 TurboCTC, the speech companion of Granite Switch: the
converter, the audio frontend, the encoder and the greedy CTC decode.

References: transformers 5.16.1 ``models/granite_speech5`` (the encoder runs
as the oracle on a tiny random checkpoint; its feature extractor needs
torchaudio, so the frontend oracle restates it on ``torch.stft``), mlx-audio
0.5.9 ``dsp.mel_filters`` as a cross-check of the filterbank, and
granite-switch 43ccd291 ``vllm/audio/chunking.py`` for the long-clip
windows."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

from gmlx.models.granite_speech5 import chunking
from gmlx.models.granite_speech5 import model as gs5

_SCRIPT = Path(__file__).parents[2] / "scripts" / "convert_granite_speech5_ctc.py"


@pytest.fixture(scope="module")
def conv():
    spec = importlib.util.spec_from_file_location("_convert_gs5", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def _cpu():
    with mx.stream(mx.cpu):
        prev = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            yield
        finally:
            mx.set_default_device(prev)


def test_the_filterbank_matches_mlx_audio(conv):
    dsp = pytest.importorskip("mlx_audio.dsp")
    want = np.array(dsp.mel_filters(16000, 512, 80, mel_scale="htk", precise=True))
    got = conv.mel_filters(80)
    assert got.shape == (257, 80)
    np.testing.assert_allclose(got, want.T if want.shape[0] == 80 else want,
                               atol=1e-5)


def _torch_features(wave, fb, window_len=400, n_fft=512, hop=160):
    """The feature extractor's ``_extract_features`` with torchaudio's
    MelSpectrogram and compute_deltas restated on torch."""
    torch = pytest.importorskip("torch")
    audio = torch.from_numpy(wave)[None]
    mel_frames = audio.shape[1] // hop
    num_frames = 2 * -(-mel_frames // 2)
    need = (num_frames - 1) * hop + 1
    if audio.shape[1] < need:
        audio = torch.nn.functional.pad(audio, (0, need - audio.shape[1]))
    spec = torch.stft(audio, n_fft, hop_length=hop, win_length=window_len,
                      window=torch.hann_window(window_len), center=True,
                      pad_mode="reflect", return_complex=True)
    mel = (spec.abs() ** 2).transpose(-1, -2) @ torch.from_numpy(fb)
    mel = mel.transpose(-1, -2)[..., :num_frames]
    logmel = mel.clamp_min(1e-10).log10()
    logmel = torch.maximum(logmel, logmel.amax(dim=(-2, -1), keepdim=True) - 8.0) / 4 + 1
    padded = torch.nn.functional.pad(logmel, (1, 1), mode="replicate")
    deltas = (padded[..., 2:] - padded[..., :-2]) / 2
    feats = torch.cat((logmel, deltas), dim=-2).transpose(-1, -2)
    return feats.reshape(1, -1, 2 * feats.shape[-1])[0].numpy()


@pytest.mark.parametrize("seconds", [0.37, 1.0, 2.013])
def test_features_match_the_extractor(conv, _cpu, seconds):
    rng = np.random.default_rng(1)
    wave = (rng.standard_normal(int(16000 * seconds)) * 0.1).astype(np.float32)
    fb = conv.mel_filters(80)
    want = _torch_features(wave, fb)
    got = np.array(gs5.features(mx.array(wave), mx.array(fb), mx.array(conv.stft_window()),
                                gs5.Config()))
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=2e-4, rtol=1e-4)


def test_a_clip_under_one_hop_has_no_frames(conv, _cpu):
    got = gs5.features(mx.zeros((100,)), mx.array(conv.mel_filters(80)),
                       mx.array(conv.stft_window()), gs5.Config())
    assert got.shape == (0, 320)


# A tiny random checkpoint, converted and run against transformers.

TINY = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4,
            num_attention_heads=2, head_dim=32, max_position_embeddings=16,
            context_size=8, subsample_layers=[0, 1], vocab_size=40)
PIECES = (["<unk>", "<|nospeech|>", "<0xC3>", "<0xA9>"]
          + [f"\u2581w{i}" if i % 3 == 0 else f"p{i}" for i in range(4, 40)])


def _tokenizer_json():
    return {
        "added_tokens": [{"id": i, "content": p, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}
                         for i, p in enumerate(PIECES[:2])],
        "normalizer": None, "pre_tokenizer": None, "post_processor": None,
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>",
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                  "vocab": {p: i for i, p in enumerate(PIECES)}, "merges": []},
        "version": "1.0", "truncation": None, "padding": None,
    }


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    from transformers.models.granite_speech5.configuration_granite_speech5 import (
        GraniteSpeech5EncoderConfig,
    )

    enc = GraniteSpeech5EncoderConfig(**TINY)
    cfg = tf.GraniteSpeech5CTCConfig(vocab_size=TINY["vocab_size"], encoder_config=enc)
    torch.manual_seed(0)
    model = tf.GraniteSpeech5ForCTC(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.ndim == 1 else 0.15))
            # bfloat16-exact values, so the bf16 GGUF is lossless.
            p.copy_(p.to(torch.bfloat16).float())
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(torch.randn_like(b).to(torch.bfloat16).float())
            elif name.endswith("running_var"):
                b.copy_((torch.rand_like(b) + 0.5).to(torch.bfloat16).float())
    d = tmp_path_factory.mktemp("tiny-turboctc")
    model.save_pretrained(d)
    (d / "processor_config.json").write_text(json.dumps({"feature_extractor": {
        "sampling_rate": 16000, "n_fft": 512, "win_length": 400, "hop_length": 160}}))
    (d / "tokenizer.json").write_text(json.dumps(_tokenizer_json()))
    return d, model


def _logits_torch(model, feats):
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(np.array(feats))[None]).logits[0].numpy()


@pytest.mark.parametrize("outtype,tol", [("bf16", 2e-4), ("f16", 2e-2), ("q8_0", 6e-2)])
def test_a_converted_tiny_checkpoint_matches_transformers(conv, tiny, tmp_path, _cpu,
                                                          outtype, tol):
    d, model = tiny
    out = tmp_path / "tiny.gguf"
    conv.convert(d, out, outtype, conv.REPO, conv.REVISION)
    if outtype == "q8_0":
        pytest.importorskip("mlx_kquant")
    speech = gs5.load(out, dtype=mx.float32)
    rng = np.random.default_rng(2)
    wave = (rng.standard_normal(int(16000 * 0.93)) * 0.1).astype(np.float32)
    feats = gs5.features(mx.array(wave), speech.mel_filters, speech.window, speech.cfg)
    assert feats.shape[0] > 2 * TINY["context_size"]   # several blocks, padded
    got = np.array(speech.encoder(feats, mx.float32))
    want = _logits_torch(model, feats)
    assert got.shape == want.shape
    err = np.abs(got - want).max() / np.abs(want).max()
    assert err < tol


def test_the_converter_records_the_source_and_frontend(conv, tiny, tmp_path):
    from gguf import GGUFReader

    from gmlx.load.gguf_meta import as_kv_dict

    out = tmp_path / "tiny.gguf"
    conv.convert(tiny[0], out, "bf16", conv.REPO, conv.REVISION)
    r = GGUFReader(str(out))
    kv = as_kv_dict(r)
    assert kv["general.architecture"] == gs5.ARCH
    assert kv["general.license"] == "apache-2.0"
    assert conv.REVISION in kv["general.source.url"]
    assert kv["tokenizer.ggml.tokens"] == PIECES
    names = {t.name for t in r.tensors}
    assert {"frontend.mel_filters", "frontend.window"} <= names
    assert not any(n.startswith(("ctc_head", "encoder.")) for n in names)
    assert gs5.is_companion(out)


@pytest.mark.parametrize("ids", [[5, 6, 7], [2, 3, 4, 8], [1, 9, 0, 9], [12, 2, 3]])
def test_decode_matches_the_tokenizer(ids):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_str(json.dumps(_tokenizer_json()))
    types = [3, 3, 6, 6] + [1] * (len(PIECES) - 4)
    vocab = gs5.CTCVocab(PIECES, types)
    assert vocab.decode(ids) == tok.decode(ids, skip_special_tokens=True)


def test_the_greedy_path_collapses_repeats_and_drops_blanks():
    vocab = gs5.CTCVocab(PIECES, [3, 3, 6, 6] + [1] * 36)
    assert vocab.collapse([0, 5, 5, 0, 5, 6, 6, 0, 0, 7]) == [5, 5, 6, 7]
    assert vocab.collapse([0, 0]) == []


def test_a_short_clip_is_one_window():
    wave = np.zeros(16000 * 30, dtype=np.float32)
    assert len(chunking.split_waveform(wave, 16000)) == 1


def test_long_clips_overlap_by_five_seconds():
    wave = np.arange(16000 * 250, dtype=np.float32)
    segs = chunking.split_waveform(wave, 16000)
    assert [len(s) for s in segs] == [16000 * 120, 16000 * 120, 16000 * 20]
    assert segs[1][0] == 16000 * 115 and segs[2][0] == 16000 * 230


def test_the_merge_keeps_one_copy_of_each_seam():
    got = chunking.merge_transcripts(
        ["the cat sat on the mat.", "On the mat, it slept", "", "slept soundly"])
    assert got == "the cat sat on the mat. it slept soundly"


# The real checkpoint, on a clip that macOS speech synthesis reads aloud.

SPOKEN = "The quick brown fox jumps over the lazy dog near the river bank."


@pytest.fixture(scope="module")
def turboctc_hf():
    import os

    d = os.environ.get("GMLX_TEST_TURBOCTC_HF")
    if not d or not (Path(d).expanduser() / "model.safetensors").is_file():
        pytest.skip("set GMLX_TEST_TURBOCTC_HF to a snapshot of "
                    "ibm-granite/granite-speech-5.0-470m-turboctc")
    return Path(d).expanduser()


@pytest.fixture(scope="module")
def spoken_clip(tmp_path_factory):
    import shutil
    import subprocess
    import wave

    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("needs macOS say and afconvert")
    d = tmp_path_factory.mktemp("spoken")
    subprocess.run(["say", "-o", str(d / "a.aiff"), SPOKEN], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(d / "a.aiff"), str(d / "a.wav")], check=True)
    with wave.open(str(d / "a.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def test_the_real_checkpoint_transcribes_as_transformers(conv, turboctc_hf, spoken_clip,
                                                         tmp_path):
    tf = pytest.importorskip("transformers")
    out = tmp_path / "turboctc.gguf"
    conv.convert(turboctc_hf, out, "bf16", conv.REPO, conv.REVISION)
    speech = gs5.load(out)
    text = speech.transcribe(spoken_clip)
    feats = gs5.features(mx.array(spoken_clip), speech.mel_filters, speech.window,
                         speech.cfg)
    ref = tf.GraniteSpeech5ForCTC.from_pretrained(turboctc_hf, dtype="float32").eval()
    want = _logits_torch(ref, feats)
    path = want.argmax(-1).tolist()
    assert text == speech.vocab.decode(speech.vocab.collapse(path)).strip()
    got = np.array(speech.encoder(feats, speech.dtype).astype(mx.float32))
    assert (got.argmax(-1) == want.argmax(-1)).mean() > 0.98
    words = set(text.lower().replace(".", "").split())
    assert {"quick", "brown", "fox", "lazy", "dog"} <= words
