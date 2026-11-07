#!/usr/bin/env python3
"""FastVideo's FastH3 tile map at three shapes, the fixture of gmlx's
tile map tests (tests/fixtures/video/fasth3_fv_geometry.npz).

FastVideo's build_h3_segment_geometry, in
fastvideo/mlx_runtime/minimax_h3_vsa.py, runs on the segments of gmlx's
packed layout of each shape: 56 text rows, the audio rows, then the
video grid (latent frames, H / 2, W / 2). The module loads by path from
a FastVideo checkout with stub modules for its logger and for mlx.core,
whose array constructor and int32 its index properties use on the numpy
arrays they build. The oracle environment so needs numpy alone.

Saved per shape: the four index arrays that mlx-kquant's VSAGeometry
takes, as FastVideo's properties build them, in int32, and the scalar
fields in ``meta`` (JSON bytes), compressed.

  python scripts/video_oracle/run.py --python <oracle python> \\
      --fixtures <dir> fasth3_geometry.py --fastvideo <FastVideo checkout>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

import fixtures
import oracle

ORACLE_NEEDS = ("numpy",)

NAME = "fasth3_fv_geometry"
MODULE = "fastvideo/mlx_runtime/minimax_h3_vsa.py"
TILE = 64
# Per shape, the dense segments then the video grid, in packed order.
SHAPES = {
    "512x512x22": (56, 74, (7, 16, 16)),
    "768x768x39": (56, 130, (12, 24, 24)),
    "1344x768x124": (56, 414, (37, 24, 42)),
}
INDEXES = ("tile_gather_index", "prefix_gather_index", "untile_index")


def _stubs() -> None:
    """fastvideo.logger, and the two names of mlx.core that the index
    properties use, over numpy."""
    pkg = types.ModuleType("fastvideo")
    pkg.__path__ = []
    lg = types.ModuleType("fastvideo.logger")

    def init_logger(name):
        log = logging.getLogger(name)
        log.warning_once = log.warning
        log.info_once = log.info
        return log

    lg.init_logger = init_logger
    core = types.ModuleType("mlx.core")
    core.int32 = np.int32
    core.array = lambda a, dtype=None: np.asarray(a, dtype=dtype)
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    mlx.core = core
    sys.modules.update({"fastvideo": pkg, "fastvideo.logger": lg,
                        "mlx": mlx, "mlx.core": core})


def load_vsa(checkout: Path):
    """FastVideo's minimax_h3_vsa module from ``checkout``."""
    _stubs()
    spec = importlib.util.spec_from_file_location("fv_vsa",
                                                  checkout / MODULE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: no {MODULE} under {checkout}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fv_vsa"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fastvideo", type=Path, required=True,
                    help="FastVideo checkout")
    ap.add_argument("--out", type=Path,
                    help="the .npz to write (default "
                         f"$GMLX_VIDEO_FIXTURES/{NAME}.npz)")
    a = ap.parse_args(argv)
    out = a.out or fixtures.fixture_root(create=True) / f"{NAME}.npz"
    checkout = a.fastvideo.expanduser().resolve()
    rev = oracle.git_head(checkout)
    fv = load_vsa(checkout)
    arrays, meta = {}, {"fastvideo_revision": rev, "tile_size": TILE,
                        "shapes": {}}
    for name, segs in SHAPES.items():
        g = fv.build_h3_segment_geometry(segs, TILE)
        for k in INDEXES:
            arrays[f"{name}/{k}"] = np.array(getattr(g, k)).astype(np.int32)
        arrays[f"{name}/variable_block_sizes"] = \
            g.variable_block_sizes.astype(np.int32)
        meta["shapes"][name] = {
            "segments": [list(s) if isinstance(s, tuple) else s
                         for s in segs],
            "num_prefix_tiles": g.num_prefix_tiles,
            "num_video_tiles": g.num_video_tiles,
            "prefix_length": g.prefix_length,
            "total_seq_length": g.total_seq_length,
            "video_tile_spans": [list(s) for s in g.video_tile_spans],
            "dense_rows_lead": bool(g.dense_rows_lead)}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(out, **arrays)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps(meta, indent=1))
    print(json.dumps({"script": f"scripts/video_oracle/{Path(__file__).name}",
                      "argv": fixtures.portable_argv(
                          sys.argv if argv is None else [__file__, *argv])}))
    print(f"wrote {out.name}: {os.path.getsize(out)} bytes, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
