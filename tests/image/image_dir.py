"""Image model files that are not text GGUFs, such as a VAE's safetensors,
found under the directory that GMLX_TEST_IMAGE_DIR names. A test that
needs one skips while the variable is unset or no file there matches."""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import pytest

ENV = "GMLX_TEST_IMAGE_DIR"


def image_dir() -> Path:
    d = os.environ.get(ENV)
    if not d:
        pytest.skip(f"set {ENV} to a directory of image model files")
    p = Path(d).expanduser()
    if not p.is_dir():
        pytest.skip(f"{ENV}={d!r} is not a directory")
    return p


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def vae_files(family: str) -> list[Path]:
    """GGUF and safetensors files under the directory whose headers hold
    the VAE of ``family``. Only headers are read."""
    from gmlx.image.vae import reader

    found = []
    for p in sorted(image_dir().rglob("*")):
        if p.suffix not in (".gguf", ".safetensors") or not p.is_file():
            continue
        try:
            reader.identify(reader.read_header(str(p)), family)
        except (reader.VAEError, ValueError, KeyError, OSError,
                struct.error):
            continue
        found.append(p)
    return found
