"""Write the short clips tests/serve/test_media_decode.py reads with ffmpeg,
and record what torchcodec gives transformers for each: the frame count,
rate and duration of its exact mode, the size, and a hash of the frames at
a set of indices. Runs where ffmpeg, torchcodec and FFmpeg's shared
libraries are installed, and never imports gmlx. Also records two clips of
the EmbeddingGemma 2 reference, which embeddinggemma2_media.py writes.

    python -I scripts/embedder_oracle/video_decode_reference.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from torchcodec.decoders import VideoDecoder

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
OUT = FIXTURES / "video_decode"
# Clips of the EmbeddingGemma 2 reference, recorded here too: the HD one is
# tagged BT.709.
SHARED = [FIXTURES / "embeddinggemma2_reference" / "media" / name
          for name in ("ball_3s.mp4", "square_hd709_3s.mp4")]
SRC = "testsrc2=size=160x120:rate={rate}:duration={seconds}"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


def clips() -> None:
    # VP9 in WebM: the header holds no frame count.
    _ffmpeg("-f", "lavfi", "-i", SRC.format(rate=24, seconds=3), "-c:v", "libvpx-vp9",
            "-b:v", "150k", "-pix_fmt", "yuv420p", str(OUT / "vp9.webm"))
    # 2 s at 10 fps, then 1 s at 30 fps, with the timestamps kept.
    _ffmpeg("-f", "lavfi", "-i", SRC.format(rate=10, seconds=2), "-f", "lavfi", "-i",
            SRC.format(rate=30, seconds=1), "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
            "-map", "[v]", "-fps_mode", "vfr", "-c:v", "libx264", "-crf", "35",
            "-pix_fmt", "yuv420p", str(OUT / "vfr.mp4"))
    # Shown turned a quarter: 120 wide, 160 high.
    plain = OUT / "plain.mp4"
    _ffmpeg("-f", "lavfi", "-i", SRC.format(rate=12, seconds=2), "-c:v", "libx264",
            "-crf", "35", "-pix_fmt", "yuv420p", str(plain))
    _ffmpeg("-display_rotation", "90", "-i", str(plain), "-c", "copy", str(OUT / "rot90.mp4"))
    plain.unlink()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips()
    record = {}
    for path in sorted(OUT.glob("*.webm")) + sorted(OUT.glob("*.mp4")) + SHARED:
        d = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=0)
        m = d.metadata
        n = int(m.num_frames)
        indices = sorted({0, n // 3, n // 2, n - 1})
        frames = d.get_frames_at(indices).data.numpy()
        record[path.name] = {
            "frames": n, "fps": float(m.average_fps), "duration": float(m.duration_seconds),
            "width": int(m.width), "height": int(m.height), "indices": indices,
            "sha256": hashlib.sha256(frames.tobytes()).hexdigest()}
    (OUT / "torchcodec.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
