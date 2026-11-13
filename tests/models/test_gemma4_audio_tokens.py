"""The Gemma 4 processor writes one audio placeholder per frame the audio
tower emits.

The tower's two stride-2 convs turn the mel frames into soft tokens, and
the model scatters them over the placeholders in order. A count past the
tower's frames wraps the scatter: the bundled 1.92 s speech clip got 49
placeholders for 48 frames, and the E2B model then answered that it could
not hear audio. transformers counts the same way. The unified processor's
extractor emits one frame per token, and its count stays the frame count.

Clips in one prompt are padded to the longest, and each clip's
placeholders must take that clip's frames, not an earlier clip's padding.
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
_AUDIO_ID = 121


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


# The clip is 30756 samples. 30720 is 48 * 640 and 30920 is 200 past it:
# the duration count and the frame count agree on both, and the clip itself
# is a length where they differ.
@pytest.mark.parametrize("extra, tokens", [
    (0, 48), (-36, 48), (164, 49), (-1, None), (1, None), (-16000, None),
    (9000, None)])
def test_placeholders_match_the_tower_frames(extra, tokens):
    speech = _speech()
    if extra < 0:
        wav = speech[:extra]
    else:
        wav = np.concatenate([speech, np.zeros(extra, np.float32)])
    fe = Gemma4AudioFeatureExtractor()
    proc = Gemma4Processor(feature_extractor=fe)
    frames = _tower_frames(fe, wav)
    assert proc._compute_audio_num_tokens(wav, 16000) == frames
    if tokens is not None:
        assert frames == tokens


def _tiny_audio_container():
    from mlx.utils import tree_flatten, tree_unflatten

    from gmlx._vendor.mlx_vlm.models import gemma4
    from gmlx._vendor.mlx_vlm.utils import update_module_configs

    text = dict(
        model_type="gemma4_text", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, head_dim=16,
        global_head_dim=32, rms_norm_eps=1e-6, vocab_size=128,
        vocab_size_per_layer_input=128, num_key_value_heads=2,
        num_kv_shared_layers=0, hidden_size_per_layer_input=0,
        sliding_window=8, sliding_window_pattern=2, tie_word_embeddings=True)
    config = {
        "model_type": "gemma4", "vocab_size": 128, "eos_token_id": 1,
        "image_token_id": 120, "audio_token_id": _AUDIO_ID,
        "text_config": text,
        "vision_config": dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            global_head_dim=16, position_embedding_size=64,
            default_output_length=4),
        "audio_config": dict(hidden_size=32, num_hidden_layers=2,
                             num_attention_heads=2, output_proj_dims=48),
    }
    cfg = update_module_configs(
        gemma4.ModelConfig.from_dict(config), gemma4, config,
        ["text", "vision", "projector", "audio"])
    mx.random.seed(5)
    model = gemma4.Model(cfg)
    # Random per-dim scales and norms, so frames differ from each other.
    model.update(tree_unflatten([
        (k, v + 0.3 * mx.random.normal(v.shape))
        if "audio" in k and v.ndim == 1 else (k, v)
        for k, v in tree_flatten(model.parameters())]))
    mx.eval(model.parameters())
    return model


def test_each_clip_gets_its_own_frames():
    """Two clips of different lengths in one prompt: each clip's
    placeholders hold the frames the tower makes from that clip alone."""
    speech = _speech()
    clips = [speech[:12000], speech]
    fe = Gemma4AudioFeatureExtractor()
    proc = Gemma4Processor(feature_extractor=fe)
    model = _tiny_audio_container()
    counts = [proc._compute_audio_num_tokens(c, 16000) for c in clips]
    assert counts[0] < counts[1]

    def features(waves):
        out = fe(waves, sampling_rate=16000, return_attention_mask=True)
        return (mx.array(np.asarray(out["input_features"], np.float32)),
                mx.array(np.asarray(out["input_features_mask"], bool)))

    # The CPU stream: a GPU batch of two rows rounds unlike a row alone.
    with mx.stream(mx.cpu):
        alone = []
        for clip, n in zip(clips, counts):
            feats, valid = features([clip])
            enc, pad = model.audio_tower(feats, ~valid)
            assert int((~pad).sum()) == n
            alone.append(np.array(model.embed_audio(enc)[0, :n]))

        ids = ([2, 5] + [_AUDIO_ID] * counts[0] + [6, 7]
               + [_AUDIO_ID] * counts[1] + [8])
        feats, valid = features(clips)
        out = model.get_input_embeddings(
            input_ids=mx.array([ids]), input_features=feats,
            input_features_mask=valid).inputs_embeds
        out = np.array(out[0])
    first = 2
    second = first + counts[0] + 2
    np.testing.assert_allclose(out[first:first + counts[0]], alone[0],
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out[second:second + counts[1]], alone[1],
                               rtol=1e-4, atol=1e-4)


def test_unified_placeholders_stay_one_per_frame():
    wav = _speech()
    fe = Gemma4UnifiedAudioFeatureExtractor()
    proc = Gemma4UnifiedProcessor(feature_extractor=fe)
    frames = int(fe([wav])["input_features_mask"].sum())
    assert proc._compute_audio_num_tokens(wav, 16000) == frames == 49
