#!/usr/bin/env python3
"""Goldens of EmbeddingGemma 2's media preprocessing, committed for the CPU
test tests/models/test_gemma_embedding2_processor.py.

  <oracle python> -I scripts/embedder_oracle/embeddinggemma2_processor.py \\
      --out tests/fixtures/embeddinggemma2_oracle

The oracle environment holds torch, torchvision, transformers 5.22.2 and
numpy, and never gmlx or mlx. It writes ``processor.npz``:

- ``image.src``: a smooth 75x100 uint8 image; per budget ``b``,
  ``image.<b>.tokens``, and at budget 70 ``image.70.resized`` (the
  reference's resize, uint8 [H, W, 3]);
- ``sizes``: rows of (height, width, budget, tokens) from the reference
  count, for odd and extreme aspect ratios;
- ``frames``: rows of (total, fps * 1000, duration * 1000), and
  ``frames.<i>`` the indices the reference samples for row ``i``;
- ``audio.<n>.mask``: the reference extractor's mask for a clip of ``n``
  samples of a seeded noise clip, for each of ``audio.lengths``; and
  ``audio.<n>.features`` for clips under 2 s, which are prefixes of
  ``audio.wave`` (int16).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TRANSFORMERS = "5.22.2"
BUDGETS = (70, 140, 280, 560, 1120)
SIZES = [(75, 100), (100, 75), (1, 4000), (4000, 1), (47, 47), (48, 48),
         (480, 640), (1080, 1920), (1920, 1080), (300, 7000), (7000, 300),
         (333, 333), (2048, 2048)]
FRAMES = [(300, 30.0, 10.0), (10, 30.0, 0.333), (90, 29.97, 3.003),
          (3600, 30.0, 120.0), (40, 2.0, 20.0), (25, 25.0, 1.0),
          (1, 30.0, 0.0333), (2000, 24.0, 2000 / 24.0)]
AUDIO_LENGTHS = (1, 159, 160, 320, 321, 481, 1000, 16000, 16001, 23456,
                 480_000, 480_127, 500_000)


def smooth_image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / max(h, w)
    out = np.zeros((h, w, 3))
    for c in range(3):
        a, b, p = rng.uniform(1, 5, 3)
        out[..., c] = 127.5 + 100 * np.sin(a * x * 6.28 + p) * np.cos(b * y * 6.28)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import torch
    import transformers as tf
    from transformers.models.embedding_gemma2.processing_embedding_gemma2 import (
        get_aspect_ratio_preserving_size,
    )
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        EmbeddingGemma2VideoProcessor,
    )
    from transformers.models.gemma4.feature_extraction_gemma4 import (
        Gemma4AudioFeatureExtractor,
    )
    from transformers.models.gemma4.image_processing_gemma4 import (
        Gemma4ImageProcessor,
    )
    from transformers.video_utils import VideoMetadata

    if tf.__version__ != TRANSFORMERS:
        raise SystemExit(f"transformers {tf.__version__}, want {TRANSFORMERS}")
    torch.set_num_threads(1)
    out = {}

    src = smooth_image(75, 100)
    out["image.src"] = src
    ip = Gemma4ImageProcessor(do_normalize=False, image_mean=[0, 0, 0],
                              image_std=[1, 1, 1], resample=3)
    for b in BUDGETS:
        r = ip(images=[src], max_soft_tokens=b, return_tensors="np")
        n = int(r["num_soft_tokens_per_image"][0])
        th, tw = get_aspect_ratio_preserving_size(75, 100, 16, b * 9, 3)
        pv, pos = r["pixel_values"][0], r["image_position_ids"][0]
        real = pv[(pos >= 0).all(-1)]
        ph, pw = th // 16, tw // 16
        img = real.reshape(ph, pw, 16, 16, 3).transpose(0, 2, 1, 3, 4)
        img = img.reshape(th, tw, 3)
        if b == 70:
            out[f"image.{b}.resized"] = np.clip(
                np.round(img * 255), 0, 255).astype(np.uint8)
        out[f"image.{b}.tokens"] = np.array(n)

    rows = []
    for h, w in SIZES:
        for b in BUDGETS:
            th, tw = get_aspect_ratio_preserving_size(h, w, 16, b * 9, 3)
            rows.append((h, w, b, (th // 16) * (tw // 16) // 9))
    out["sizes"] = np.array(rows, dtype=np.int64)

    vp = EmbeddingGemma2VideoProcessor()
    rows = []
    for i, (total, fps, duration) in enumerate(FRAMES):
        meta = VideoMetadata(total_num_frames=total, fps=fps, duration=duration)
        idx = vp.sample_frames(meta, fps=vp.fps, max_frames=vp.max_frames,
                               overflow_strategy=vp.overflow_strategy)
        out[f"frames.{i}"] = np.asarray(idx, dtype=np.int64)
        rows.append((total, round(fps * 1000), round(duration * 1000)))
    out["frames"] = np.array(rows, dtype=np.int64)

    fe = Gemma4AudioFeatureExtractor()
    rng = np.random.default_rng(5)
    wave = np.clip(np.round(rng.standard_normal(max(AUDIO_LENGTHS)) * 3000),
                   -32768, 32767).astype(np.int16)
    out["audio.wave"] = wave[:32_000]
    out["audio.lengths"] = np.array(AUDIO_LENGTHS)
    for n in AUDIO_LENGTHS:
        f = fe([wave[:n].astype(np.float32) / 32768.0], return_tensors="np")
        if n < 32_000:
            out[f"audio.{n}.features"] = f["input_features"][0].astype(np.float32)
        out[f"audio.{n}.mask"] = f["input_features_mask"][0].astype(bool)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "processor.npz", **out)
    print({k: v.shape for k, v in out.items() if "audio" not in k})


if __name__ == "__main__":
    main()
