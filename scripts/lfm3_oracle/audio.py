"""Record the LFM3-Omni speech-input reference that
tests/models/test_lfm3_audio_live.py checks gmlx against. Runs in an
environment with torch and transformers, on a folder holding the
LiquidAI/LFM3-Omni-2B checkpoint and its code. The output holds arrays
and token ids only.

    python scripts/lfm3_oracle/audio.py <model_dir> tests/fixtures/lfm3_oracle/audio.npz

It records, in float32 on the CPU, for two synthetic clips of 16 kHz
audio (1.31 s and 0.52 s, neither a whole number of frames):

- the samples as int16;
- the reference feature extractor's spectrogram of each clip;
- the projected features of each clip encoded alone, and of the two
  encoded as one padded batch;
- the token ids of a prompt holding both clips, as the chat template and
  the reference processor write them.

Spectrograms and features are stored as float16.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

QUESTION = "What do you hear?"
RATE = 16000
LENGTHS = (20931, 8333)


def clip(samples: int, seed: int) -> np.ndarray:
    """int16 samples: three gliding tones under a slow envelope, over
    quiet noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float64) / RATE
    wave = np.zeros(samples)
    for base, glide, level in ((180.0, 90.0, 0.30), (720.0, -260.0, 0.20),
                               (2400.0, 900.0, 0.10)):
        phase = 2 * np.pi * (base * t + 0.5 * glide * t * t)
        wave += level * np.sin(phase + seed)
    wave *= 0.55 + 0.45 * np.sin(2 * np.pi * 3.1 * t)
    wave += 0.01 * rng.standard_normal(samples)
    return np.round(np.clip(wave, -1, 1) * 32767).astype(np.int16)


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    extractor = processor.feature_extractor

    out: dict = {}
    waves = []
    for i, samples in enumerate(LENGTHS):
        pcm = clip(samples, seed=i + 1)
        wave = pcm.astype(np.float32) / 32768.0
        waves.append(wave)
        pre = extractor([wave], sampling_rate=RATE, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_audio_features(
                pre["input_features"], pre["feature_attention_mask"])
        out[f"clip{i}"] = pcm
        out[f"clip{i}_mel"] = pre["input_features"][0].numpy().astype(
            np.float16)
        out[f"clip{i}_features"] = feats.numpy().astype(np.float16)

    pre = extractor(waves, sampling_rate=RATE, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_audio_features(
            pre["input_features"], pre["feature_attention_mask"])
    out["batch_features"] = feats.numpy().astype(np.float16)

    messages = [{"role": "user", "content": [
        {"type": "audio"}, {"type": "text", "text": QUESTION},
        {"type": "audio"}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(text=[text], audio=waves, sampling_rate=RATE,
                    return_tensors="pt")
    out["prompt_ids"] = enc["input_ids"].numpy().reshape(-1)

    np.savez_compressed(out_path, **out)
    for key, value in out.items():
        print(f"{key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
