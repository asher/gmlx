#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "gguf>=0.19", "huggingface_hub"]
# ///
"""Convert IBM's Granite Speech 5.0 TurboCTC checkpoint to a GGUF that gmlx
pairs with Granite Switch as its speech companion.

The source is the Hugging Face repo ibm-granite/granite-speech-5.0-470m-turboctc
(Apache-2.0): a 470M conformer CTC encoder in bfloat16 safetensors. The GGUF
keeps every tensor under its Hugging Face name without the ``encoder.``
prefix, adds the audio frontend (mel filterbank and STFT window) as tensors,
and stores the CTC vocabulary under ``tokenizer.ggml.*``.

Usage:
  uv run convert_granite_speech5_ctc.py --out granite-speech-5.0-470m-turboctc.gguf

Without ``--hf-dir`` the script downloads the pinned revision with
huggingface_hub. ``--outtype`` is bf16 (lossless, the default), f16 or q8_0
(linear weights only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType, GGUFWriter

ARCH = "granite-speech5-ctc"
REPO = "ibm-granite/granite-speech-5.0-470m-turboctc"
REVISION = "1dd23f46f6d4e6eae2531982ba2e33220086dd97"
GGUF_REPO = "ashbash/granite-speech-5.0-470m-turboctc-GGUF"
Q = GGMLQuantizationType

# torchaudio.transforms.MelSpectrogram as the feature extractor builds it.
SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
DELTA_WIN_LENGTH = 3
LOGMEL_FLOOR_DB = 8.0
FRAME_STACKING = 2


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file as numpy; bfloat16 widens to
    float32 exactly."""
    raw = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64}
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        for name, info in header.items():
            start, end = info["data_offsets"]
            f.seek(base + start)
            a = np.frombuffer(f.read(end - start), dtype=raw[info["dtype"]])
            a = a.reshape(info["shape"])
            if info["dtype"] == "BF16":
                a = (a.astype(np.uint32) << 16).view(np.float32)
            out[name] = a
    return out


def mel_filters(n_mels: int) -> np.ndarray:
    """torchaudio's ``melscale_fbanks`` (HTK scale, no norm) for the
    extractor's settings: ``(n_fft // 2 + 1, n_mels)`` float32."""
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, n_freqs, dtype=np.float32)

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE / 2), n_mels + 2,
                        dtype=np.float32)
    f_pts = mel_to_hz(m_pts).astype(np.float32)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)


def stft_window() -> np.ndarray:
    """The periodic Hann window of ``WIN_LENGTH`` centred in ``N_FFT``, as
    torch.stft pads it."""
    n = np.arange(WIN_LENGTH, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / WIN_LENGTH)
    left = (N_FFT - WIN_LENGTH) // 2
    out = np.zeros(N_FFT, dtype=np.float64)
    out[left:left + WIN_LENGTH] = hann
    return out.astype(np.float32)


def _token_type(tok: dict, piece: str) -> int:
    if tok.get(piece, {}).get("special"):
        return gguf.TokenType.CONTROL
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        return gguf.TokenType.BYTE
    return gguf.TokenType.NORMAL


def write_tokenizer(w: GGUFWriter, tokenizer: dict, vocab_size: int) -> None:
    model = tokenizer["model"]
    vocab = model["vocab"]
    pieces = [""] * vocab_size
    for piece, i in (vocab.items() if isinstance(vocab, dict) else vocab):
        if i < vocab_size:
            pieces[i] = piece
    added = {t["content"]: t for t in tokenizer.get("added_tokens", [])}
    for t in added.values():
        if t["id"] < vocab_size:
            pieces[t["id"]] = t["content"]
    if any(p == "" for p in pieces):
        raise SystemExit("error: the tokenizer does not cover the CTC vocabulary")
    w.add_tokenizer_model("llama")
    w.add_token_list(pieces)
    w.add_token_types([int(_token_type(added, p)) for p in pieces])
    w.add_uint32(f"{ARCH}.ctc.blank_token_id", 0)
    w.add_pad_token_id(0)
    w.add_unk_token_id(0)


def _tensor(w: GGUFWriter, name: str, a: np.ndarray, outtype: str) -> None:
    a = np.ascontiguousarray(a)
    if a.dtype == np.int64:
        return
    if a.ndim == 1 or name.startswith("frontend.") or ".conv.norm." in name:
        w.add_tensor(name, a.astype(np.float32))
        return
    linear = a.ndim == 2 and name.endswith(".weight") and "rel_pos_emb" not in name
    if outtype == "q8_0" and linear and a.shape[-1] % 32 == 0:
        q = gguf.quants.quantize(a.astype(np.float32), Q.Q8_0)
        w.add_tensor(name, q, raw_dtype=Q.Q8_0)
    elif outtype == "f16":
        w.add_tensor(name, a.astype(np.float16))
    else:
        bf = (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
        w.add_tensor(name, bf, raw_dtype=Q.BF16)


def convert(hf_dir: Path, out: Path, outtype: str, repo: str, revision: str) -> None:
    config = json.loads((hf_dir / "config.json").read_text())
    if config.get("model_type") != "granite_speech5_ctc":
        raise SystemExit(f"error: {hf_dir} is not a granite_speech5_ctc checkpoint")
    enc = config["encoder_config"]
    fe = json.loads((hf_dir / "processor_config.json").read_text()).get("feature_extractor", {})
    expected = {"sampling_rate": SAMPLE_RATE, "n_fft": N_FFT, "win_length": WIN_LENGTH,
                "hop_length": HOP_LENGTH, "delta_win_length": DELTA_WIN_LENGTH,
                "logmel_floor_db": LOGMEL_FLOOR_DB}
    for k, v in expected.items():
        if k in fe and fe[k] != v:
            raise SystemExit(f"error: feature extractor {k}={fe[k]}, this script expects {v}")
    tokenizer = json.loads((hf_dir / "tokenizer.json").read_text())
    weights = read_safetensors(hf_dir / "model.safetensors")

    w = GGUFWriter(str(out), ARCH)
    w.add_name("Granite Speech 5.0 470M TurboCTC")
    w.add_license("apache-2.0")
    w.add_source_repo_url(f"https://huggingface.co/{repo}")
    w.add_source_url(f"https://huggingface.co/{repo}/tree/{revision}")
    p = f"{ARCH}."
    w.add_uint32(p + "block_count", enc["num_hidden_layers"])
    w.add_uint32(p + "embedding_length", enc["hidden_size"])
    w.add_uint32(p + "feed_forward_length", enc["intermediate_size"])
    w.add_uint32(p + "attention.head_count", enc["num_attention_heads"])
    w.add_uint32(p + "attention.head_count_kv", enc["num_key_value_heads"])
    w.add_uint32(p + "attention.key_length", enc["head_dim"])
    w.add_uint32(p + "attention.context_size", enc["context_size"])
    w.add_uint32(p + "attention.max_relative_position", enc["max_position_embeddings"])
    w.add_float32(p + "attention.layer_norm_epsilon", 1e-5)
    w.add_uint32(p + "conv.kernel_size", enc["conv_kernel_size"])
    w.add_uint32(p + "conv.expansion_factor", enc["conv_expansion_factor"])
    w.add_float32(p + "conv.batch_norm_epsilon", 1e-5)
    w.add_array(p + "subsample_layers", [int(i) for i in enc["subsample_layers"]])
    w.add_uint32(p + "vocab_size", config["vocab_size"])
    w.add_uint32(p + "audio.sample_rate", SAMPLE_RATE)
    w.add_uint32(p + "audio.n_fft", N_FFT)
    w.add_uint32(p + "audio.win_length", WIN_LENGTH)
    w.add_uint32(p + "audio.hop_length", HOP_LENGTH)
    w.add_uint32(p + "audio.num_mel_bins", enc["num_mel_bins"])
    w.add_uint32(p + "audio.delta_win_length", DELTA_WIN_LENGTH)
    w.add_float32(p + "audio.logmel_floor_db", LOGMEL_FLOOR_DB)
    w.add_uint32(p + "audio.frame_stacking", FRAME_STACKING)
    write_tokenizer(w, tokenizer, config["vocab_size"])

    _tensor(w, "frontend.mel_filters", mel_filters(enc["num_mel_bins"]), outtype)
    _tensor(w, "frontend.window", stft_window(), outtype)
    for name in sorted(weights):
        if name.startswith("ctc_head."):
            continue        # tied to encoder.out
        if not name.startswith("encoder."):
            raise SystemExit(f"error: unexpected tensor {name}")
        _tensor(w, name[len("encoder."):], weights[name], outtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


README = """---
license: apache-2.0
base_model: {repo}
tags:
- gguf
- speech-recognition
- ctc
---

# Granite Speech 5.0 470M TurboCTC (GGUF)

IBM's [Granite Speech 5.0 TurboCTC](https://huggingface.co/{repo}) speech
recognizer as a GGUF, for use with [gmlx](https://asher.github.io/gmlx/) on
macOS with Apple silicon. gmlx pairs it with Granite Switch, so the chat model
can read audio, and serves it for speech to text. Other GGUF runtimes, such as
llama.cpp, do not load this file.

## Use it with gmlx

Download it with the Hugging Face CLI:

```sh
hf download {gguf_repo}
```

gmlx pairs it with a Granite Switch model that also came from the Hugging Face
cache. For a model in a folder of your own, put the file in the model's folder,
or pass its path:

```sh
gmlx serve granite-switch-4.1-8b-Q4_K_M.gguf --asr {name}
```

To serve `/v1/audio/transcriptions` with it, set `server.stt: granite-speech`
in the gmlx config. [Granite Switch audio](https://asher.github.io/gmlx/granite-switch.html#audio)
in the gmlx docs covers pairing and audio in chat requests.

## Files

| File | Type | SHA-256 |
|------|------|---------|
| `{name}` | {outtype} | `{sha256}` |

The weights are IBM's, unchanged. The file adds the audio frontend (mel
filterbank and STFT window) as tensors and the CTC vocabulary as tokenizer
metadata.

## Rebuild it

`convert_granite_speech5_ctc.py` in this repo downloads IBM's checkpoint at
revision `{revision}` and writes the same file:

```sh
uv run convert_granite_speech5_ctc.py --out {name} --outtype {outtype}
```

## License

The weights are Apache 2.0, from
[{repo}](https://huggingface.co/{repo}). The conversion script is MIT.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-dir", type=Path, help="a local snapshot of the source repo")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--outtype", choices=("bf16", "f16", "q8_0"), default="bf16")
    ap.add_argument("--readme", action="store_true",
                    help="also write README.md beside the output")
    a = ap.parse_args(argv)
    hf_dir = a.hf_dir
    if hf_dir is None:
        from huggingface_hub import snapshot_download

        hf_dir = Path(snapshot_download(a.repo, revision=a.revision))
    convert(hf_dir.expanduser(), a.out.expanduser(), a.outtype, a.repo, a.revision)
    if a.readme:
        out = a.out.expanduser()
        (out.parent / "README.md").write_text(README.format(
            repo=a.repo, revision=a.revision, outtype=a.outtype, name=out.name,
            gguf_repo=GGUF_REPO, sha256=_sha256(out)))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
