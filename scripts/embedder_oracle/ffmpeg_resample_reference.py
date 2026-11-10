"""Record torchcodec's 16 kHz mono decode of short int16 clips at several
rates, the reference tests/serve/test_media_decode.py holds
media_decode.ffmpeg_resample to. Runs where torchcodec and FFmpeg's shared
libraries are installed, and never imports gmlx.

    python -I scripts/embedder_oracle/ffmpeg_resample_reference.py
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from torchcodec.decoders import AudioDecoder

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ffmpeg_resample" / "reference.npz"
RATES = (8000, 11025, 22050, 44100, 48000)


def main() -> None:
    rng = np.random.default_rng(0)
    store = {}
    for rate in RATES:
        n = rate // 10 + 7          # an odd length, cut part-way through a phase
        t = np.arange(n) / rate
        x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * rate * 0.45 * t)
             + 0.1 * rng.standard_normal(n))
        pcm = np.clip(np.round(x * 26000), -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        y = AudioDecoder(buf.getvalue(), num_channels=1,
                         sample_rate=16000).get_all_samples().data.numpy()[0]
        store[f"in.{rate}"] = pcm
        store[f"out.{rate}"] = y.astype(np.float32)
        print(rate, len(pcm), len(y))
    np.savez_compressed(OUT, **store)


if __name__ == "__main__":
    main()
