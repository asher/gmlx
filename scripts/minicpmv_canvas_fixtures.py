#!/usr/bin/env python
"""Record MiniCPM-V 4.7 canvas positions from the transformers reference as
a test fixture.

    python scripts/minicpmv_canvas_fixtures.py /path/to/transformers \
        --fixture tests/fixtures/minicpmv_canvas_positions.json

The path is a transformers checkout at commit 674c0e3, the first with
``minicpmv4_7``. Its ``get_rope_index`` runs on the CPU (MPS rounds the
canvas differently). The fixture stores each prompt, its crop sizes and the
reference positions, so the pinned test needs neither torch nor the
checkout.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from types import SimpleNamespace

PAD, IMG, IMG_END, SLICE, SLICE_END = 248056, 248078, 248079, 248088, 248089
ID, ID_END, NEWLINE = 248090, 248091, 198
DIGITS = {0: 15, 1: 16}


def image(overview, grid=None, slice_hw=None, *, merge=4, image_id=None):
    """Tokens and crop sizes of one image block, as the processor writes it.
    ``overview`` and ``slice_hw`` are patch grids, ``grid`` (rows, cols)."""
    ids = [] if image_id is None else [ID, DIGITS[image_id], ID_END]
    per = (overview[0] // merge) * (overview[1] // merge)
    ids += [IMG] + [PAD] * per + [IMG_END]
    sizes = [list(overview)]
    if grid:
        rows, cols = grid
        n = (slice_hw[0] // merge) * (slice_hw[1] // merge)
        for r in range(rows):
            if r:
                ids.append(NEWLINE)
            for _c in range(cols):
                ids += [SLICE] + [PAD] * n + [SLICE_END]
                sizes.append(list(slice_hw))
    return ids, sizes


def text(n, start=1000):
    return list(range(start, start + n))


def cases():
    out = []

    def add(name, parts, mode="16x", pad=0):
        ids, sizes = [], []
        for p in parts:
            if isinstance(p, tuple):
                ids += p[0]
                sizes += p[1]
            else:
                ids += p
        out.append({"name": name, "downsample_mode": mode, "pad": pad,
                    "input_ids": ids, "target_sizes": sizes})

    add("overview_only", [text(3), image((32, 32), image_id=0), [NEWLINE],
                          text(4)])
    add("grid_2x2", [text(2), image((32, 44), (2, 2), (28, 36), image_id=0),
                     [NEWLINE], text(5)])
    add("grid_1x3", [text(2), image((20, 56), (1, 3), (32, 28), image_id=0),
                     [NEWLINE], text(3)])
    add("two_images", [text(2), image((32, 32), image_id=0), [NEWLINE],
                       image((24, 48), (2, 2), (28, 28), image_id=1),
                       [NEWLINE], text(6)])
    add("image_first", [image((28, 36), image_id=0), [NEWLINE], text(4)])
    add("no_image_id", [text(2), image((32, 32), (1, 2), (32, 32)),
                        text(3)])
    add("text_only", [text(9)])
    # The overview stretched over a taller canvas, where float32 linspace
    # rounds a half the other way from exact arithmetic.
    add("linspace_48_23", [text(1), image((92, 4), (2, 1), (96, 4)),
                           text(2)])
    add("linspace_90_39", [text(1), image((156, 4), (3, 1), (120, 4)),
                           text(2)])
    add("linspace_96_47", [text(1), image((188, 4), (4, 1), (96, 4)),
                           text(2)])
    add("linspace_wide_48_23", [text(1), image((4, 92), (1, 2), (4, 96)),
                                text(2)])
    add("mode_4x", [text(2), image((16, 20), (1, 2), (14, 16), merge=2,
                                   image_id=0), [NEWLINE], text(3)],
        mode="4x")
    add("left_padded", [text(2), image((32, 32), (1, 2), (32, 28),
                                       image_id=0), [NEWLINE], text(2)],
        pad=5)
    return out


def run(checkout: str, case: dict) -> dict:
    sys.path.insert(0, f"{checkout}/src")
    import torch
    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7 as m

    cls = m.MiniCPMV4_7Model
    cfg = SimpleNamespace(
        image_token_id=PAD, video_token_id=248057, image_start_id=IMG,
        image_end_id=IMG_END, slice_start_id=SLICE, slice_end_id=SLICE_END,
        newline_id=NEWLINE, downsample_mode=case["downsample_mode"])
    self = SimpleNamespace(config=cfg, _frame_end_idx=cls._frame_end_idx)
    for name in ("_group_visual_frames", "get_vision_position_ids"):
        setattr(self, name, functools.partial(getattr(cls, name), self))
    ids = [0] * case["pad"] + case["input_ids"]
    mask = [0] * case["pad"] + [1] * len(case["input_ids"])
    input_ids = torch.tensor([ids], dtype=torch.long)
    pos, deltas = cls.get_rope_index(
        self, input_ids,
        mm_token_type_ids=(input_ids == PAD).int(),
        target_sizes=(torch.tensor(case["target_sizes"], dtype=torch.long)
                      if case["target_sizes"] else None),
        attention_mask=torch.tensor([mask]) if case["pad"] else None)
    return {**case, "position_ids": pos[:, 0].tolist(),
            "rope_deltas": int(deltas[0, 0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkout", help="transformers checkout at 674c0e3")
    ap.add_argument("--fixture", required=True)
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", args.checkout, "rev-parse", "HEAD"],
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    import torch

    data = {"transformers_commit": commit, "torch": torch.__version__,
            "cases": [run(args.checkout, c) for c in cases()]}
    with open(args.fixture, "w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {len(data['cases'])} cases to {args.fixture}")


if __name__ == "__main__":
    main()
