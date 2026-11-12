"""The Gemma 4 processor writes one audio placeholder per frame the audio
tower emits.

The tower's two stride-2 convs turn the mel frames into soft tokens, and
the model scatters them over the placeholders in order. A count past the
tower's frames wraps the scatter: the bundled 1.92 s speech clip got 49
placeholders for 48 frames, and the E2B model then answered that it could
not hear audio. transformers counts the same way. The unified processor's
extractor emits one frame per token, and its count stays the frame count.
"""

from __future__ import annotations

import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from gmlx._vendor.mlx_vlm.models.gemma4.audio import SubSampleConvProjection
from gmlx._vendor.mlx_vlm.models.gemma4.audio_feature_extractor import (
    Gemma4AudioFeatureExtractor,
)
from gmlx._vendor.mlx_vlm.models.gemma4.config import AudioConfig
from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor
from gmlx._vendor.mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
    Gemma4UnifiedAudioFeatureExtractor,
    Gemma4UnifiedProcessor,
)

SPEECH = Path(__file__).parents[1] / "e2e" / "assets" / "speech.wav"


def _speech() -> np.ndarray:
    with wave.open(str(SPEECH)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
            16000, 1, 2)
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _tower_frames(fe, wav: np.ndarray) -> int:
    out = fe([wav], sampling_rate=16000, return_attention_mask=True)
    feats = mx.array(np.asarray(out["input_features"], np.float32))
    valid = mx.array(np.asarray(out["input_features_mask"], bool))
    mx.random.seed(0)
    sscp = SubSampleConvProjection(AudioConfig(hidden_size=32))
    _, pad = sscp(feats, ~valid)
    return int((~pad).sum())


@pytest.mark.parametrize("extra", [0, -1, 1, -16000, 9000])
def test_placeholders_match_the_tower_frames(extra):
    speech = _speech()
    if extra < 0:
        wav = speech[:extra]
    else:
        wav = np.concatenate([speech, np.zeros(extra, np.float32)])
    fe = Gemma4AudioFeatureExtractor()
    proc = Gemma4Processor(feature_extractor=fe)
    frames = _tower_frames(fe, wav)
    assert proc._compute_audio_num_tokens(wav, 16000) == frames
    if extra == 0:
        assert frames == 48


def test_unified_placeholders_stay_one_per_frame():
    wav = _speech()
    fe = Gemma4UnifiedAudioFeatureExtractor()
    proc = Gemma4UnifiedProcessor(feature_extractor=fe)
    frames = int(fe([wav])["input_features_mask"].sum())
    assert proc._compute_audio_num_tokens(wav, 16000) == frames == 49
