#!/usr/bin/env python3
"""Repack an MLX checkpoint stored for RecursiveSI's Rec runtime into the
layout stock MLX loaders read, such as oMLX, mlx-lm and mlx-vlm.

Rec stores each quantized weight as U8 and keeps per-module widths (8-bit
embeddings beside 4-bit layers) outside config.json. Stock loaders expect
uint32-packed weights and per-module "quantization" entries. The repack
relabels each U8 weight that has a .scales sibling as U32 (last axis / 4),
infers each module's width from its byte count and scales, and writes the
widths that differ from the default into config.json. Tensor bytes are
copied unchanged; other files are hard links.

  ./repack-rec-mlx.py SRC_DIR DST_DIR [--dry-run]
"""
import argparse
import glob
import json
import os
import shutil
import struct
import sys


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Rec checkpoint folder")
    ap.add_argument("dst", help="output folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the widths, write nothing")
    a = ap.parse_args(argv)

    with open(os.path.join(a.src, "config.json")) as fh:
        cfg = json.load(fh)
    q = dict(cfg.get("quantization") or {})
    gs, default_bits = q.get("group_size"), q.get("bits")
    if not gs or not default_bits:
        sys.exit("config.json has no quantization group_size/bits")

    shards = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
    headers = {p: read_header(p) for p in shards}
    scales = {k: v["shape"] for _, (_, h) in headers.items() for k, v in h.items()
              if k.endswith(".scales")}

    widths = {}
    for _, (_, h) in headers.items():
        for k, v in h.items():
            if k == "__metadata__" or not k.endswith(".weight") or v["dtype"] != "U8":
                continue
            mod = k[: -len(".weight")]
            sc = scales.get(mod + ".scales")
            if sc is None:
                continue
            cols, in_dim = v["shape"][-1], sc[-1] * gs
            bits = cols * 8 // in_dim
            if cols * 8 != bits * in_dim or cols % 4:
                sys.exit(f"{k}: {cols} bytes per row does not fit group {gs} x {sc[-1]}")
            widths[mod] = bits
            v["dtype"] = "U32"
            v["shape"] = v["shape"][:-1] + [cols // 4]

    overrides = {m: {"group_size": gs, "bits": b} for m, b in sorted(widths.items())
                 if b != default_bits}
    counts = {}
    for b in widths.values():
        counts[b] = counts.get(b, 0) + 1
    print(f"{len(widths)} quantized weights relabeled U8 -> U32; widths {counts}; "
          f"default {default_bits}-bit, {len(overrides)} overrides:")
    for m, o in overrides.items():
        print(f"  {m}: {o['bits']}-bit")
    if a.dry_run:
        return

    os.makedirs(a.dst, exist_ok=True)
    for f in os.listdir(a.src):
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if f.endswith(".safetensors") or f == "config.json" or f.startswith(".") \
                or os.path.isdir(s) or os.path.exists(d):
            continue
        os.link(s, d)

    for path, (n, h) in headers.items():
        blob = json.dumps(h, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        out = os.path.join(a.dst, os.path.basename(path))
        with open(path, "rb") as src, open(out + ".tmp", "wb") as dst:
            src.seek(8 + n)
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            shutil.copyfileobj(src, dst, 64 << 20)
        os.replace(out + ".tmp", out)
        print(f"wrote {out}")

    q.update(overrides)
    cfg["quantization"] = q
    with open(os.path.join(a.dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    print(f"wrote {os.path.join(a.dst, 'config.json')}")


if __name__ == "__main__":
    main()
