#!/usr/bin/env python3
"""Reference encoder outputs of Nemotron 3.5 Transcribe for one clip.

  python scripts/speech_oracle/run.py --python <oracle python> \\
      ref_encoder.py --encoder <...-Encoder-F16.gguf> --audio <clip> \\
      --name <fixture name>

Runs transformers' reference (oracle.TRANSFORMERS_REV) with the encoder
GGUF's weights in float32 and in bfloat16, and writes one fixture:

- ``audio``: the 16 kHz samples the run used, ``features`` and ``mask``:
  the processor's spectrogram and its valid frames;
- per mode (``fp32_`` / ``bf16_`` prefix): ``embeds`` (the projected rows
  that replace the audio tokens), ``fused`` (the audio tower output before
  the projector), ``asr_states`` (the ASR encoder output, the CTC head's
  input), ``speaker_logits`` (10 ms) and ``speaker_probs`` (80 ms means);
- the CTC head per mode: the full log-softmax rows when they are small
  enough, else ``ctc_argmax``, ``ctc_logsumexp`` and the logits of the
  blank and of ``--ctc-columns`` fixed class ids.

The manifest records the fp32-vs-bf16 gap of each output and a tolerance
of twice that gap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import oracle

# Above this many CTC logits the fixture keeps a summary, not every row.
FULL_CTC_LIMIT = 64_000_000


def run_mode(model, feats, mask, mode: str, full_ctc: bool, columns):
    import torch

    dtype = oracle.torch_dtype(mode)
    x = feats.to(dtype)
    t0 = time.perf_counter()
    # The audio tower's output, with the projected rows as pooler_output.
    audio = model.get_audio_features(x, mask)
    embeds = audio.pooler_output
    n_frames = int(audio.last_hidden_state.shape[1])
    probs = (audio.speaker_logits.float().sigmoid()
             .unflatten(1, (n_frames, -1)).mean(2))
    logits = model.get_word_timestamp_logits(audio.asr_last_hidden_state, mask)
    took = time.perf_counter() - t0
    logits = logits.float()[0]
    out = {
        "embeds": embeds.float().numpy(),
        "fused": audio.last_hidden_state.float()[0].numpy(),
        "asr_states": audio.asr_last_hidden_state.float()[0].numpy(),
        "speaker_logits": audio.speaker_logits.float()[0].numpy(),
        "speaker_probs": probs[0].numpy(),
    }
    logp = torch.log_softmax(logits, dim=-1)
    if full_ctc:
        out["ctc_logp"] = logp.numpy()
    else:
        out["ctc_argmax"] = logits.argmax(-1).numpy().astype(np.int32)
        out["ctc_logsumexp"] = torch.logsumexp(logits, -1).numpy()
        out["ctc_columns_logp"] = logp[:, columns].numpy()
    return out, took


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, type=Path)
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float,
                    help="keep only the first this many seconds")
    ap.add_argument("--ctc-columns", type=int, default=512,
                    help="fixed class ids kept when the CTC rows are summarized")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--stable-topk", action="store_true",
                    help="break speaker cache score ties by the lowest index, "
                    "as gmlx does")
    a = ap.parse_args(argv)
    torch = oracle.setup(a.threads)
    if a.stable_topk:
        oracle.stable_speaker_cache_topk()

    meta, tensors = oracle.read_gguf(a.encoder)
    audio = oracle.load_audio(a.audio)
    if a.seconds:
        audio = audio[: int(a.seconds * 16000)]
    feats, mask = oracle.features(audio)

    w = oracle.writer(a.name, check="encoder", script="ref_encoder.py")
    w.add_file("encoder", a.encoder)
    w.add_file("audio", a.audio)
    w.inputs.update({"seconds": len(audio) / 16000, "samples": len(audio),
                     "mel_frames": int(mask.sum()),
                     "padded_frames": int(feats.shape[1])})
    w.add("audio", audio)
    w.add("features", feats[0].numpy().astype(np.float32))
    w.add("mask", mask[0].numpy().astype(np.int32))

    vocab = int(meta["ctc.vocab_size"])
    n_ctc = (feats.shape[1] // oracle.SUBSAMPLING) * int(meta["ctc.upsampling_factor"])
    full_ctc = n_ctc * vocab <= FULL_CTC_LIMIT
    rng = np.random.default_rng(0)
    columns = np.sort(np.concatenate([
        [vocab - 1], rng.choice(vocab - 1, a.ctc_columns, replace=False)]))
    if not full_ctc:
        w.add("ctc_columns", columns.astype(np.int32))
    w.inputs["ctc"] = "full" if full_ctc else "summary"
    w.inputs["stable_topk"] = a.stable_topk

    fx = oracle.fixtures_module()
    outs = {}
    for mode in oracle.MODES:
        model = oracle.reference_model(meta, tensors, mode)
        outs[mode], took = run_mode(model, feats, mask, mode, full_ctc, columns)
        w.results[f"{mode}_seconds"] = round(took, 2)
        print(f"[ref_encoder] {mode}: {took:.1f} s", flush=True)
        for k, v in outs[mode].items():
            w.add(f"{mode}_{k}", v)
        del model

    gaps = {}
    for k, ref in outs["fp32"].items():
        got = outs["bf16"][k]
        if k == "ctc_argmax":
            gaps[k] = {"agree": float(np.mean(got == ref))}
            w.tolerances[k] = {"agree": 1.0 - 2.0 * (1.0 - gaps[k]["agree"])}
            continue
        gaps[k] = fx.compare(got, ref)
        w.tolerances[k] = {"rel_l2": 2.0 * gaps[k]["rel_l2"],
                           "max_abs": 2.0 * gaps[k]["max_abs"]}
    w.results["bf16_vs_fp32"] = gaps
    w.note("Tolerances are twice the reference's own fp32-vs-bf16 gap.")
    path = w.write(oracle.fixture_dir())
    print(f"[ref_encoder] wrote {path}")
    for k, g in gaps.items():
        print(f"  {k}: {g}")
    del torch
    return 0


if __name__ == "__main__":
    sys.exit(main())
