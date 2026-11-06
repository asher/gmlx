#!/usr/bin/env python3
"""Goldens of the tiny random Nemotron speech encoder, committed for the
CPU test tests/models/test_nemotron_transcribe_speech.py.

  python scripts/speech_oracle/run.py --python <oracle python> \\
      tiny_encoder.py --out tests/fixtures/nemotron_transcribe

Writes the tiny GGUF of ``tiny_pee.py`` (seed ``SEED``), runs transformers'
reference on it in float32 and float64, with speaker cache ties broken as
gmlx breaks them (``oracle.stable_speaker_cache_topk``), and writes one
fixture with the weights, three synthetic clips and the float32 outputs:

- ``long``: 24.93 s; 16 windows, and the speaker cache compresses twice;
- ``short``: 3.3 s, ending in a partial 8-frame group; one window. Its
  spectrogram is kept too;
- ``whole``: 4 s, a whole number of 8-frame groups, so the reference
  carries one masked group past the audio. The reference's speaker head
  convolution reads that group's states at the last real frame, where
  NeMo zeroes it first (NeMo-Speech.cpp ``PeeGraph::speaker_probs``), so
  the test leaves the last frame's speaker outputs out.

Tolerances are twice the reference's own float32-vs-float64 gap, and at
least 1e-5 relative L2.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import oracle
import tiny_pee

SEED = 13
CASES = {"long": 2493 * 160, "short": 52_800, "whole": 400 * 160}
REL_L2_FLOOR = 1e-5


def synthetic_clip(samples: int, seed: int) -> np.ndarray:
    """Tones that switch on and off over noise, as 16-bit PCM."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 16000.0
    x = 0.02 * rng.standard_normal(samples)
    for i in range(6):
        f0 = 150.0 + 90.0 * i
        gate = (np.sin(2 * np.pi * (0.13 + 0.07 * i) * t + i) > 0.3)
        x += 0.15 * gate * np.sin(2 * np.pi * f0 * t * (1 + 0.02 * np.sin(t)))
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def mel_bank() -> np.ndarray:
    """The extractor's slaney bank rounded to bfloat16, as the real file
    stores it."""
    import torch

    fe = oracle.feature_extractor()
    return fe.mel_filters.to(torch.bfloat16).float().numpy()


def run(model, feats, mask, torch, dtype, compress_calls):
    audio = model.get_audio_features(feats.to(dtype), mask)
    n = int(audio.last_hidden_state.shape[1])
    logits = model.get_word_timestamp_logits(audio.asr_last_hidden_state, mask)
    v = int(mask.sum())
    return {
        "embeds": audio.pooler_output.double().numpy(),
        "asr_states": audio.asr_last_hidden_state.double()[0].numpy(),
        "speaker_logits": audio.speaker_logits.double()[0, :v].numpy(),
        "speaker_probs": audio.speaker_logits.double().sigmoid()
        .unflatten(1, (n, -1)).mean(2)[0].numpy(),
        "ctc_logp": torch.log_softmax(logits.double()[0], -1).numpy(),
        "compressions": compress_calls[0],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    torch = oracle.setup(1)
    oracle.stable_speaker_cache_topk()
    from transformers.models.nemotron3_diarization import \
        modeling_nemotron3_diarization as diar

    calls = [0]
    compress = diar.Nemotron3DiarizationSpeakerCache._compress

    def counted(self, *args, **kw):
        calls[0] += 1
        return compress(self, *args, **kw)

    diar.Nemotron3DiarizationSpeakerCache._compress = counted

    weights = tiny_pee.tiny_weights(SEED, mel_bank())
    with tempfile.TemporaryDirectory() as tmp:
        meta, tensors = oracle.read_gguf(
            tiny_pee.write_gguf(Path(tmp) / "tiny.gguf", weights))
    models = {"fp32": oracle.reference_model(meta, tensors, "fp32"),
              "fp64": oracle.reference_model(meta, tensors, "fp32").double()}

    w = oracle.writer("selftest_pee_tiny", check="encoder",
                      script="tiny_encoder.py")
    w.inputs.update({"seed": SEED, "fields": tiny_pee.FIELDS,
                     "cases": CASES})
    for k, v in weights.items():
        w.add(f"w/{k}", v)
    for case, samples in CASES.items():
        pcm = synthetic_clip(samples, SEED + samples)
        audio = pcm.astype(np.float32) / 32768.0
        feats, mask = oracle.features(audio)
        w.add(f"{case}/pcm", pcm)
        if case == "short":
            w.add(f"{case}/features", feats[0].numpy().astype(np.float32))
        outs = {}
        for mode, model in models.items():
            calls[0] = 0
            outs[mode] = run(model, feats, mask, torch,
                             torch.float64 if mode == "fp64" else torch.float32,
                             calls)
        w.results[f"{case}_compressions"] = outs["fp32"].pop("compressions")
        outs["fp64"].pop("compressions")
        probs = outs["fp32"]["speaker_probs"]
        w.results[f"{case}_active_fraction"] = float((probs > 0.5).mean())
        w.results[f"{case}_threshold_margin"] = float(np.abs(probs - 0.5).min())
        for k, ref in outs["fp32"].items():
            w.add(f"{case}/{k}", ref.astype(np.float32))
            gap = oracle.fixtures_module().compare(ref, outs["fp64"][k])
            w.tolerances[f"{case}/{k}"] = {
                "rel_l2": max(2.0 * gap["rel_l2"], REL_L2_FLOOR)}
    w.note("Tolerances are twice the reference's own float32-vs-float64 "
           f"gap, at least {REL_L2_FLOOR} relative L2.")
    path = w.write(a.out)
    print(f"[tiny_encoder] wrote {path}")
    for k, v in w.results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
