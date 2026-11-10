"""Record the LFM3-Omni vision reference that tests/models/test_lfm3_vision_live.py
checks gmlx against. Runs in an environment with torch, torchvision and
transformers, on a folder holding the LiquidAI/LFM3-Omni-2B checkpoint and
its code. The output holds arrays and token ids only.

    python scripts/lfm3_oracle/vision.py <model_dir> tests/fixtures/lfm3_oracle/vision.npz

It records, in float32 on the CPU:

- a 256x256 image (no resize runs at that size), the reference
  processor's patches and grid, and the projected features;
- a 7-frame clip, its patches and grid, and the projected features;
- the token ids of an image prompt and a video prompt, as the chat
  template and the reference processor write them.

It then casts the tower and the projector to bfloat16, the dtype the
checkpoint stores, and records the features of the same patches again
(``*_features_bf16``). That is the reference's own 16-bit result.

Features are stored as float16.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.video_utils import VideoMetadata

IMAGE_QUESTION = "Describe this image."
VIDEO_QUESTION = "What happens?"
CLIP_INDICES = [0, 15, 30, 45, 60, 75, 89]
CLIP_FPS = 30.0
CLIP_TOTAL = 90


def image() -> np.ndarray:
    """``[256, 256, 3]`` uint8: two blocks and stripes on a gradient."""
    y, x = np.mgrid[0:256, 0:256]
    img = np.stack([x, y, (x + y) // 2], axis=-1).astype(np.float64)
    img[30:130, 20:120] = (200, 40, 40)
    img[110:210, 140:240] = (40, 90, 200)
    img[(x // 8 + y // 8) % 5 == 0] *= 0.6
    return np.clip(img, 0, 255).astype(np.uint8)


def clip() -> np.ndarray:
    """``[7, 3, 96, 160]`` uint8: a bar that moves and brightens over a
    gradient."""
    y, x = np.mgrid[0:96, 0:160]
    frames = np.zeros((7, 3, 96, 160), dtype=np.uint8)
    for t in range(7):
        frames[t] = np.stack([x // 4, y // 3, (x + y + 9 * t) % 40])
        frames[t, :, 20:76, 10 + 20 * t:30 + 20 * t] = 60 + 25 * t
    return frames


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32).eval()

    def features(pixel_values, grid, dtype=torch.float32) -> np.ndarray:
        with torch.no_grad():
            hidden = model.vision_tower(
                pixel_values.to(dtype), grid_thw=grid).last_hidden_state
            return model.multi_modal_projector(hidden).float().numpy()

    out: dict = {}
    pixels = image()
    pil = Image.fromarray(pixels)
    pre = processor.image_processor(images=[pil], return_tensors="pt")
    out["image"] = pixels
    out["image_pixel_values"] = pre["pixel_values"].numpy()
    out["image_grid"] = pre["image_grid_thw"].numpy()
    out["image_features"] = features(
        pre["pixel_values"], pre["image_grid_thw"]).astype(np.float16)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": IMAGE_QUESTION}]}]
    enc = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    out["image_prompt_ids"] = enc["input_ids"].numpy().reshape(-1)

    frames = clip()
    meta = VideoMetadata(
        total_num_frames=CLIP_TOTAL, fps=CLIP_FPS,
        frames_indices=list(CLIP_INDICES), width=160, height=96,
        duration=CLIP_TOTAL / CLIP_FPS)
    vid = processor.video_processor(
        videos=[torch.from_numpy(frames)], video_metadata=[meta],
        do_sample_frames=False, return_tensors="pt")
    out["clip"] = frames
    out["clip_indices"] = np.asarray(CLIP_INDICES)
    out["clip_fps"] = np.asarray(CLIP_FPS)
    out["clip_total"] = np.asarray(CLIP_TOTAL)
    out["clip_pixel_values"] = vid["pixel_values_videos"].numpy()
    out["clip_grid"] = vid["video_grid_thw"].numpy()
    out["clip_features"] = features(
        vid["pixel_values_videos"], vid["video_grid_thw"]).astype(np.float16)
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": VIDEO_QUESTION}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc = processor(
        text=[text], videos=[torch.from_numpy(frames)],
        video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
    out["clip_prompt_ids"] = enc["input_ids"].numpy().reshape(-1)

    model.vision_tower.to(torch.bfloat16)
    model.multi_modal_projector.to(torch.bfloat16)
    out["image_features_bf16"] = features(
        pre["pixel_values"], pre["image_grid_thw"],
        torch.bfloat16).astype(np.float16)
    out["clip_features_bf16"] = features(
        vid["pixel_values_videos"], vid["video_grid_thw"],
        torch.bfloat16).astype(np.float16)

    np.savez_compressed(out_path, **out)
    for key, value in out.items():
        print(f"{key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
