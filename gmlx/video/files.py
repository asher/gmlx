"""The video files a model folder holds, found by their weights.

A DiT is found by its fingerprint and variant, the text encoder and its
vision tower by the identity digests of the family's encoder spec, the
video VAE, the audio VAE and the preview decoder by their tensor names
and shapes, and a LoRA by its sha256 against a recipe's pin.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

ROLES = ("dit", "encoder", "mmproj", "vae", "audio_vae", "tiny", "lora")
# The video reader row of each VAE role.
VAE_ROWS = {"vae": "h3", "audio_vae": "h3_audio", "tiny": "h3_tiny"}


@dataclass(frozen=True)
class VideoFile:
    path: str
    role: str
    family: str
    variant: str | None = None
    layout: str | None = None        # a DiT's qkv layout, or "unverified"


def _gguf_role(path: str, family) -> VideoFile | None:
    from gmlx.image.catalog import _kv, _view
    from gmlx.image.fingerprint import ImageModelError
    from gmlx.image.text_encoder import check_encoder_file, check_mmproj_file

    from . import fingerprint as vfp

    kv = _kv(path)
    arch = kv.get("general.architecture")
    if arch is None or arch in family.arch_strings:
        try:
            got = vfp.identify(_view(path))
        except (vfp.VideoModelError, OSError, ValueError):
            got = None
        if got is not None and got.family.name == family.name:
            return VideoFile(path, "dit", family.name, got.variant,
                             got.layout)
    spec = family.encoder()
    try:
        if arch == "clip":
            check_mmproj_file(_view(path), spec)
            return VideoFile(path, "mmproj", family.name)
        if arch == spec.arch and kv.get(f"{arch}.block_count") == spec.layers:
            check_encoder_file(_view(path), spec)
            return VideoFile(path, "encoder", family.name)
    except (ImageModelError, OSError, ValueError):
        pass
    return None


def _vae_role(path: str, family) -> VideoFile | None:
    from gmlx.image.vae import reader as image_reader

    from .vae.reader import VAES

    for role, label in VAE_ROWS.items():
        row = VAES[label]
        try:
            header = image_reader.read_header(path, own_names=row.own_names)
            image_reader.identify(header, label, rows=VAES)
        except (image_reader.VAEError, OSError, ValueError):
            continue
        return VideoFile(path, role, family.name)
    return None


def scan(dirs, family) -> list:
    """The video files of ``family`` under ``dirs``, first shards and
    safetensors files. A file that cannot be read is skipped."""
    from gmlx.load.preflight import is_first_shard

    out: list = []
    seen: set = set()
    for d in dirs or ():
        root = os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = sorted(n for n in dirnames if not n.startswith("."))
            for n in sorted(names):
                p = os.path.join(dirpath, n)
                real = os.path.realpath(p)
                if real in seen or not os.path.isfile(p):
                    continue
                seen.add(real)
                low = n.lower()
                try:
                    if low.endswith(".gguf") and is_first_shard(n):
                        got = _gguf_role(p, family)
                    elif low.endswith(".safetensors"):
                        got = _vae_role(p, family)
                    else:
                        got = None
                except Exception as e:  # noqa: BLE001 (a bad file is skipped)
                    print(f"[video] skipping {p}: {e}", file=sys.stderr)
                    got = None
                if got is not None:
                    out.append(got)
    return out


def find_pinned(dirs, pin) -> str | None:
    """A file under ``dirs`` with the pin's size and sha256, by name
    first."""
    from gmlx.image.catalog import sha256_file

    hits: list = []
    for d in dirs or ():
        root = os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [n for n in dirnames if not n.startswith(".")]
            for n in names:
                p = os.path.join(dirpath, n)
                if os.path.isfile(p) and os.path.getsize(p) == pin.size:
                    hits.append(p)
    hits.sort(key=lambda p: (os.path.basename(p) != pin.name, p))
    for p in hits:
        try:
            if sha256_file(p) == pin.sha256:
                return p
        except OSError as e:
            print(f"[video] skipping {p}: {e}", file=sys.stderr)
    return None
