"""``gmlx image``: image generation from GGUF diffusion models.

``pack-vae`` writes a lossless VAE GGUF from a safetensors VAE and checks
it bit for bit. The library behind each action is ``gmlx.image``.

Exit codes follow ``gmlx run``: 0 on success, 1 for a refusal or a
runtime failure, 2 for a usage error and 130 for an interrupt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ACTIONS = ("pack-vae",)
_ACTION_DESC = {
    "pack-vae": "write a lossless VAE GGUF from a safetensors VAE",
}
_PACK_DTYPES = {"f32": "F32", "f16": "F16", "bf16": "BF16"}


def _print_help(prog: str) -> None:
    lines = [f"usage: {prog} <action> [options]", "",
             "image generation from GGUF diffusion models.", "", "actions:"]
    for a in _ACTIONS:
        lines.append(f"  {a:<10} {_ACTION_DESC[a]}")
    lines += ["", f"run `{prog} <action> --help` for an action's options."]
    print("\n".join(lines))


def _refuse(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _pack_vae_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Pack a safetensors VAE into a VAE GGUF. Tensors keep "
                    "their official names, torch layout and, by default, "
                    "their dtype, so the file is lossless. The GGUF is read "
                    "back and every tensor compared bit for bit before it "
                    "replaces --out.")
    p.add_argument("src", metavar="SRC", help="The safetensors VAE.")
    p.add_argument("--out", required=True, metavar="FILE.gguf",
                   help="The GGUF to write.")
    p.add_argument("--dtype", choices=sorted(_PACK_DTYPES), default=None,
                   help="Store every tensor in this dtype (default: the "
                        "source dtype). A narrower dtype is not lossless, "
                        "and the file records that.")
    p.add_argument("--family", default=None, metavar="NAME",
                   help="The image family whose VAE this is (default: the "
                        "one family whose VAE the file fills).")
    p.add_argument("--source-repo", default="unknown", metavar="ID",
                   help="Record this repository as the source (default "
                        "unknown).")
    p.add_argument("--source-revision", default="unknown", metavar="REV",
                   help="Record this revision of the source repository "
                        "(default unknown).")
    p.add_argument("--source-config", default=None, metavar="FILE",
                   help="Record this VAE config.json (default: config.json "
                        "beside SRC when there is one).")
    p.add_argument("--force", action="store_true",
                   help="Replace --out when it exists.")
    p.add_argument("--json", action="store_true",
                   help="Print a JSON record of the packed file on stdout.")
    return p


def cmd_pack_vae(argv: list[str], prog: str = "gmlx image pack-vae") -> int:
    a = _pack_vae_parser(prog).parse_args(argv)
    from gmlx.image.vae import VAEError, pack_vae

    cfg_path = a.source_config
    if cfg_path is None:
        beside = os.path.join(os.path.dirname(os.path.abspath(a.src)),
                              "config.json")
        cfg_path = beside if os.path.isfile(beside) else None
    config_json = "{}"
    if cfg_path is not None:
        try:
            with open(cfg_path, encoding="utf-8") as f:
                config_json = f.read()
        except OSError as e:
            return _refuse(f"--source-config {cfg_path}: {e.strerror}")
    try:
        rep = pack_vae(a.src, a.out, source_repo=a.source_repo,
                       source_revision=a.source_revision,
                       config_json=config_json, family=a.family,
                       dtype=_PACK_DTYPES.get(a.dtype), overwrite=a.force)
    except VAEError as e:
        return _refuse(str(e))
    record = {"path": rep.path, "family": rep.family, "tensors": rep.tensors,
              "payload_bytes": rep.payload_bytes,
              "source_sha256": rep.source_sha256, "dtype": rep.dtype,
              "lossless": rep.lossless, "verified": True}
    if a.json:
        print(json.dumps(record))
    else:
        print(f"wrote {rep.path}: {rep.tensors} tensors, "
              f"{rep.payload_bytes / 1e6:.1f} MB, {rep.family} VAE, "
              f"verified bit for bit")
    if not rep.lossless:
        print(f"note: {a.dtype} is narrower than the source, so "
              f"{rep.path} is not lossless", file=sys.stderr)
    return 0


HANDLERS = {"pack-vae": cmd_pack_vae}


def cmd_image(argv: list[str], prog: str = "gmlx image") -> int:
    """Dispatch ``gmlx image <action> ...``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help(prog)
        return 0
    action, rest = argv[0], argv[1:]
    if action not in HANDLERS:
        print(f"error: unknown action {action!r}; one of "
              f"{', '.join(_ACTIONS)}", file=sys.stderr)
        return 2
    return HANDLERS[action](rest, prog=f"{prog} {action}")
