"""Record how the LFM3-Omni reference processor writes a video with its
soundtrack, for tests/models/test_lfm3_audio.py. Runs in an environment
with torch and transformers, on a folder holding the LiquidAI/LFM3-Omni-2B
checkpoint's processor and its code. The output holds token ids and
lengths only.

    python scripts/lfm3_oracle/soundtrack.py <model_dir> tests/fixtures/lfm3_oracle/soundtrack.npz

The video is 7 frames of 96x160 sampled from a 3 s clip at 30 fps. The
token ids depend on the frame geometry and on the length of each audio
clip, not on their content, so the media are zeros. It records the token
ids of three prompts, each written with ``use_audio_in_video``:

- ``full``: the video with a soundtrack as long as the video;
- ``short``: the video with a soundtrack that ends before the second
  frame group starts;
- ``mixed``: a clip, the video with its soundtrack, a question and a
  second clip, with the frame count of each clip the feature extractor
  was given, in the order it was given them.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

QUESTION = "What happens?"
RATE = 16000
SHAPE = (7, 3, 96, 160)
INDICES = [0, 15, 30, 45, 60, 75, 89]
FPS = 30.0
TOTAL = 90
FULL, SHORT = 48000, 14000
BEFORE, AFTER = 8333, 20931


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    frames = torch.zeros(SHAPE, dtype=torch.uint8)

    def encode(content, lengths):
        meta = VideoMetadata(
            total_num_frames=TOTAL, fps=FPS, frames_indices=list(INDICES),
            width=SHAPE[3], height=SHAPE[2], duration=TOTAL / FPS)
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True)
        return processor(
            text=[text], videos=[frames], video_metadata=[meta],
            audio=[np.zeros(n, dtype=np.float32) for n in lengths],
            do_sample_frames=False, use_audio_in_video=True,
            sampling_rate=RATE, return_tensors="pt")

    question = {"type": "text", "text": QUESTION}
    pair = [{"type": "video"}, {"type": "audio"}]
    out: dict = {
        "shape": np.asarray(SHAPE), "indices": np.asarray(INDICES),
        "fps": np.asarray(FPS), "total": np.asarray(TOTAL),
        "full_samples": np.asarray(FULL), "short_samples": np.asarray(SHORT),
        "mixed_samples": np.asarray([BEFORE, FULL, AFTER]),
    }
    for name, samples in (("full", FULL), ("short", SHORT)):
        enc = encode(pair + [question], [samples])
        out[f"{name}_ids"] = enc["input_ids"].numpy().reshape(-1)
    enc = encode([{"type": "audio"}] + pair + [question, {"type": "audio"}],
                 [BEFORE, FULL, AFTER])
    out["mixed_ids"] = enc["input_ids"].numpy().reshape(-1)
    out["mixed_frames"] = enc["feature_attention_mask"].sum(-1).numpy()

    np.savez_compressed(out_path, **out)
    for key, value in out.items():
        print(f"{key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
