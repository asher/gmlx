#!/usr/bin/env python3
"""Keyframe prompt fixture for MiniMax-H3: ComfyUI's ids, token tags and
tower input of a prompt with keyframes, CPU, no weights.

For each case ComfyUI's MiniMaxH3Tokenizer builds the prompt with its
images; each image's embed entry stands for (W/32)(H/32) rows, which the
fixture writes as the image pad id, the id the tower's rows replace in
transformers' and gmlx's prompt. The tags are ComfyUI's
token_tags_from_embeds_info over those rows. The fixture also holds each
text piece's ids as ComfyUI tokenizes it alone, so a test can build the
prompt from them without the tokenizer, and the tower input
process_qwen2vl_images makes of one 64x64 image at Qwen3-VL's patch and
normalization.

  python scripts/video_oracle/run.py --python <comfyui venv python> \\
      keyframes_h3.py --comfy-dir <ComfyUI checkout>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("torch", "comfy_kitchen")

NAME = "keyframes_h3"
PAD = 151655
PROMPT = "A red fox trots through fresh snow at dawn, then looks back"
# case: (prompt, the images as (width, height), first then last)
CASES = {
    "first_512": (PROMPT, [(512, 512)]),
    "last_512": (PROMPT, [(512, 512)]),
    "both_512": (PROMPT, [(512, 512), (512, 512)]),
    "first_1344x768": (PROMPT, [(1344, 768)]),
    "both_1344x768": (PROMPT, [(1344, 768), (1344, 768)]),
    "both_768_empty": ("", [(768, 768), (768, 768)]),
}


def expand(entries, sizes) -> tuple:
    """(ids, embeds_info) of ComfyUI's entries with each image entry as
    its rows of pad ids, as its encode expands them."""
    ids, info, k = [], [], 0
    for tok, _w in entries:
        if isinstance(tok, dict):
            w, h = sizes[k]
            n = (w // 32) * (h // 32)
            info.append({"type": "image", "index": len(ids), "size": n})
            ids += [PAD] * n
            k += 1
        else:
            ids.append(int(tok))
    return ids, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comfy-dir", type=Path, required=True,
                    help="ComfyUI checkout (its H3 tokenizer)")
    ap.add_argument("--out", type=Path,
                    help="fixture directory (default $GMLX_VIDEO_FIXTURES)")
    a = ap.parse_args(argv)
    oracle.setup()
    commit = oracle.start_comfy(a.comfy_dir)
    import torch

    from comfy.text_encoders import minimax
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images

    tok = minimax.MiniMaxH3Tokenizer()
    inner = tok.qwen3vl_32b
    fx = fixtures.FixtureWriter(
        NAME, check=2, family="h3",
        script=f"scripts/video_oracle/{Path(__file__).name}",
        argv=[Path(__file__).name, "--comfy-dir", "<ComfyUI checkout>"])
    cases = {}
    for name, (prompt, sizes) in CASES.items():
        images = [torch.zeros((1, h, w, 3)) for w, h in sizes]
        entries = tok.tokenize_with_weights(prompt, images=images)
        entries = entries["qwen3vl_32b"]
        if len(entries) != 1:
            raise SystemExit(f"error: {name}: {len(entries)} token batches")
        ids, info = expand(entries[0], sizes)
        tags = minimax.token_tags_from_embeds_info(len(ids), info)
        pieces = {}
        for text in [f"<Picture {i}>: " for i in range(1, len(sizes) + 1)] \
                + ([prompt] if prompt else []):
            batch = inner.tokenize_with_weights(text, return_word_ids=False,
                                                disable_weights=True)
            pieces[text] = [int(t) for t, _ in batch[0]]
        cases[name] = {"prompt": prompt, "sizes": [list(s) for s in sizes],
                       "ids": ids, "tags": [int(t) for t in tags],
                       "pieces": pieces}
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    pix, grid = process_qwen2vl_images(
        torch.from_numpy(image.astype(np.float32) / 255.0)[None],
        patch_size=16, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    fx.add("image_64", image)
    fx.add("pixels_64", oracle.to_numpy(pix).astype(np.float32))
    fx.add("grid_64", np.asarray(grid, np.int64))
    fx.inputs.update(cases={k: {"prompt": p, "sizes": [list(s) for s in v]}
                            for k, (p, v) in CASES.items()},
                     pad_id=PAD, image="uint8 64x64x3, numpy default_rng(0)")
    fx.results.update(cases=cases, comfyui_commit=commit)
    fx.tolerances.update(ids="exact", tags="exact", pixels_64="max abs 1e-6",
                         rule="gmlx's ids and spans equal ComfyUI's; its "
                              "tower input within 1e-6 of "
                              "process_qwen2vl_images")
    print(f"wrote {fx.write(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
