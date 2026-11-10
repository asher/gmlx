"""Write the synthetic media of the EmbeddingGemma 2 reference check to
tests/fixtures/embeddinggemma2_reference/media: drawn images, tone clips
and videos of moving shapes. Needs Pillow, numpy and ffmpeg.

    python scripts/embedder_oracle/embeddinggemma2_media.py
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "embeddinggemma2_reference" / "media"


def images():
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((170, 90, 470, 390), fill=(220, 20, 30))
    im.save(OUT / "red_circle.png", optimize=True)

    im = Image.new("RGB", (1024, 576), (250, 230, 60))
    d = ImageDraw.Draw(im)
    for x in range(0, 1024, 128):
        d.rectangle((x, 0, x + 63, 575), fill=(30, 60, 200))
    d.polygon([(512, 80), (760, 500), (264, 500)], fill=(30, 170, 60))
    im.save(OUT / "stripes_triangle.png", optimize=True)


def _tones(seconds, rate, notes, channels=1):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = np.zeros(n)
    step = seconds / len(notes)
    for i, f in enumerate(notes):
        on = (t >= i * step) & (t < (i + 1) * step)
        env = np.sin(np.pi * ((t - i * step) / step).clip(0, 1)) ** 2
        x += on * env * (0.5 * np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t))
    x = (0.6 * x / np.abs(x).max() * 32767).astype(np.int16)
    if channels == 2:
        x = np.stack([x, (x * 0.7).astype(np.int16)], axis=1)
    return x


def _flac(name, x, rate):
    wav = OUT / (name + ".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1 if x.ndim == 1 else x.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "flac",
                    "-compression_level", "12", str(OUT / (name + ".flac"))], check=True)
    wav.unlink()


def audio():
    c_major = [262, 330, 392, 523, 392, 330, 262, 196, 262, 330]
    _flac("arpeggio_10s", _tones(10, 16000, c_major), 16000)
    # 44.1 kHz, past the 30 s cut. Stereo, held to 8-bit steps so FLAC
    # stores it small.
    x = _tones(34, 44100, [440, 494, 523, 587, 659, 587, 523, 494] * 4 + [440],
               channels=2)
    _flac("melody_34s", (x // 256 * 256).astype(np.int16), 44100)
    _flac("beep_1s", _tones(1.5, 16000, [880]), 16000)
    _flac("hum_4s", _tones(4, 16000, [110, 147]), 16000)


# An HD clip tagged BT.709, where a decoder that converts with the BT.601
# matrix, as OpenCV does, is off by tens of levels.
_BT709 = ["-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
          "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


def _video(name, seconds, fps, draw, size=(256, 192), background=None, bt709=False):
    w, h = size
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *(_BT709 if bt709 else []),
         "-c:v", "libx264", "-preset", "veryslow", "-crf", "30", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(OUT / name)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i in range(int(seconds * fps)):
        im = background.copy() if background is not None else Image.new("RGB", (w, h), "white")
        draw(ImageDraw.Draw(im), i / fps, w, h)
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait():
        sys.exit("ffmpeg failed")


def videos():
    def ball(d, t, w, h):
        x = 20 + (w - 80) * t / 3
        d.ellipse((x, 70, x + 50, 120), fill=(220, 20, 30))

    def scenes(d, t, w, h):
        colors = [(220, 20, 30), (30, 60, 200), (30, 170, 60), (240, 200, 20)]
        c = colors[int(t // 10) % len(colors)]
        x = (w - 60) * ((t % 10) / 10)
        d.rectangle((x, 60, x + 60, 120), fill=c)
        d.rectangle((0, h - 20, w * t / 40, h), fill=(80, 80, 80))

    _video("ball_3s.mp4", 3, 10, ball)
    _video("scenes_40s.mp4", 40, 5, scenes)
    hd_video()


def hd_video():
    # Saturated color ramps behind a turning blue square.
    w, h = 1280, 720
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    ramp = np.concatenate([np.broadcast_to(x, (h, w, 1)), np.broadcast_to(y, (h, w, 1)),
                           np.broadcast_to(1 - x, (h, w, 1))], axis=2)
    background = Image.fromarray((ramp * 255).round().astype(np.uint8))

    def square(d, t, w, h):
        cx, cy, r = w / 2 + 300 * np.cos(t), h / 2 + 200 * np.sin(t), 90
        d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=(20, 40, 210))

    _video("square_hd709_3s.mp4", 3, 24, square, size=(w, h), background=background,
           bt709=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    images()
    audio()
    videos()
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size:>9} {p.name}")
