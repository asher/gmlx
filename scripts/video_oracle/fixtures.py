#!/usr/bin/env python3
"""Oracle fixtures for the video acceptance checks: writer, reader and
manifest.

The format and the commands are those of scripts/image_oracle/fixtures.py,
which this module loads by path and points at GMLX_VIDEO_FIXTURES. The
manifest's package list adds the ComfyUI kernels package, since the DiT
and VAE oracles run ComfyUI's model code. The repository keeps a copy of
each manifest that a test reads under ``tests/fixtures/video/oracle/``.

  python fixtures.py list
  python fixtures.py verify <name> [--manifest <committed copy>]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PACKAGES = ("torch", "diffusers", "transformers", "tokenizers",
             "accelerate", "huggingface-hub", "gguf", "safetensors",
             "numpy", "comfy-kitchen")


def _image_fixtures():
    path = Path(__file__).resolve().parent.parent / "image_oracle" / \
        "fixtures.py"
    spec = importlib.util.spec_from_file_location("_video_oracle_fixtures",
                                                  path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ENV = "GMLX_VIDEO_FIXTURES"
    mod.__doc__ = __doc__
    base = mod.environment

    def environment() -> dict:
        env = base()
        env["packages"] = mod.package_versions(_PACKAGES)
        return env

    mod.environment = environment
    return mod


_fx = _image_fixtures()

ENV = _fx.ENV
FixtureError = _fx.FixtureError
FixtureWriter = _fx.FixtureWriter
fixture_root = _fx.fixture_root
read_manifest = _fx.read_manifest
load = _fx.load
array_sha256 = _fx.array_sha256
file_identity = _fx.file_identity
rel_l2 = _fx.rel_l2
cosine = _fx.cosine
max_abs = _fx.max_abs
psnr = _fx.psnr
compare = _fx.compare


if __name__ == "__main__":
    try:
        sys.exit(_fx._main())
    except FixtureError as e:
        sys.exit(f"error: {e}")
