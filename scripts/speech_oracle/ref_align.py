#!/usr/bin/env python3
"""Goldens of NeMo's word timestamp aligner, committed for the CPU test
tests/models/test_nemotron_transcribe_align.py.

  python scripts/speech_oracle/run.py --python <oracle python> \\
      ref_align.py --nemo <ctc_timestamp_utils.py> \\
      --out tests/fixtures/nemotron_transcribe

``--nemo`` is NeMo Speech's
``nemo/collections/speechlm2/parts/ctc_timestamp_utils.py`` at 979cc6c6.
Its NeMo imports are stubbed: the aligner's search, mapping and refinement
need numpy, torch and scipy only. NeMo's NumPy forward runs (no numba), with
the checkpoint's alignment parameters and calibration.

The cases come from ``align_cases.py`` and a SentencePiece model trained
here; the fixture stores the model, every input and NeMo's output per case:
words with their speaker tag and times, the 10 ms diarization segments, the
tag to diarization column map and the streams with no legal path, or the
error NeMo raises.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np

import align_cases
import oracle

# HF config.json ctc_timestamps.alignment_config.parameters at c1766764.
CONFIG = Path(__file__).resolve().parents[2] / "gmlx" / "models" / "nemotron_h" / "align.py"


def _stub(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod


def nemo_aligner(path: Path):
    """NeMo's ctc_timestamp_utils with its NeMo imports stubbed."""
    import torch

    for pkg in ("nemo", "nemo.collections", "nemo.collections.asr",
                "nemo.collections.asr.modules", "nemo.collections.common",
                "nemo.collections.common.tokenizers", "nemo.core",
                "nemo.core.utils"):
        _stub(pkg)
    _stub("nemo.collections.asr.modules.conv_asr", ConvASRDecoder=torch.nn.Module)
    _stub("nemo.collections.asr.modules.transformer_encoder", TransformerEncoder=object)
    _stub("nemo.collections.common.tokenizers.sentencepiece_tokenizer",
          SentencePieceTokenizer=object)
    _stub("nemo.core.utils.optional_libs", NUMBA_AVAILABLE=False)
    _stub("nemo.utils", logging=logging.getLogger("nemo"))
    _stub("omegaconf", OmegaConf=None)
    spec = importlib.util.spec_from_file_location("nemo_ctc_timestamp_utils", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def checkpoint_parameters() -> dict:
    """gmlx's pinned copy of the checkpoint parameters, read as text so
    the oracle never imports gmlx."""
    ns: dict = {}
    src = CONFIG.read_text()
    start = src.index("PARAMETERS: dict[str, Any] = {")
    end = src.index("\n}\n", start) + 3
    exec("from typing import Any\n" + src[start:end], ns)  # noqa: S102
    return dict(ns["PARAMETERS"], interpolation_enabled=True)


class _Tokenizer:
    """NeMo's SentencePieceTokenizer with legacy=False: the model's encode."""

    def __init__(self, sp):
        self.tokenizer = sp

    def text_to_ids(self, text):
        return self.tokenizer.encode(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nemo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    torch = oracle.setup(1)
    import sentencepiece as spm

    mod = nemo_aligner(a.nemo)
    params = checkpoint_parameters()
    proto = align_cases.train_spm()
    sp = spm.SentencePieceProcessor(model_proto=proto)
    blank = sp.get_piece_size()
    aligner = mod.MultiSpeakerSOTWordTimestampAligner(
        tokenizer=_Tokenizer(sp), blank_id=blank, **params)
    # The checkpoint's head upsamples encoder frames 2x; the aligner reads
    # its word cap through it (_ctc_frames). The head itself never runs.
    aligner.ctc_decoder = types.SimpleNamespace(upsampling_factor=2)

    w = oracle.writer("align_nemo_tiny", check="alignment", script="ref_align.py")
    w.add_file("nemo_ctc_timestamp_utils", a.nemo)
    w.inputs.update({"seed": align_cases.SEED, "parameters": params})
    w.add("spm", np.frombuffer(proto, np.uint8))
    for name, case in align_cases.cases().items():
        text = align_cases.transcript(case)
        arrays = align_cases.inputs(case, sp, name)
        frames = case["frames"]
        duration = frames * 0.04
        for k, v in arrays.items():
            w.add(f"{name}/{k}", v)
        w.inputs[name] = {"transcript": text, "duration": duration}
        probs = arrays.get("speaker_probs")
        labels = arrays.get("labels")
        try:
            prepared = aligner.prepare_alignment(
                torch.from_numpy(arrays["logp"].astype(np.float32))[None],
                None if probs is None else torch.from_numpy(probs)[None], [text],
                audio_durations=[duration],
                diarization_labels=None if labels is None else torch.from_numpy(labels)[None])
            (result,) = aligner.align_prepared(prepared)
        except ValueError as e:
            w.results[name] = {"error": str(e)}
            continue
        words = [{"word": x["word"], "speaker": x["speaker"], "start": x["start"],
                  "end": x["end"]}
                 for ws in result["speaker_word_timestamps"].values() for x in ws]
        words.sort(key=lambda x: (x["start"], x["end"]))
        w.results[name] = {
            "words": words,
            "diarization": result["diarization_timestamps"],
            "mapping": {str(k): v for k, v in result["speaker_tag_to_sortformer_column"].items()},
            "failed": result["alignment_diagnostics"]["failed_streams"],
            "banded": [{k: d[k] for k in ("speaker", "retained_frames", "states", "half_width")}
                       for d in result["alignment_diagnostics"]["banded"]],
        }
    w.note("NeMo ctc_timestamp_utils.py at 979cc6c6, NumPy forward, checkpoint "
           "alignment parameters and calibration; times compare to 1e-9 s.")
    path = w.write(a.out)
    print(f"[ref_align] wrote {path}")
    print(json.dumps({k: (v.get("error") or f"{len(v['words'])} words, "
                      f"failed {v['failed']}") for k, v in w.results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
