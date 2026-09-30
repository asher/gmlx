#!/usr/bin/env python3
"""Load a DiT GGUF through diffusers' from_single_file with the GGUF
quantization config, and report what it built.

The report gives the GGUF linear leaves by codec, the plain parameters by
dtype, and for one matrix of each codec the gap between diffusers'
dequantization and gguf-py's reference dequantization of the same bytes,
once with stock float16 block arithmetic and once with the oracle's
float32 arithmetic.

Run it through run.py:

  python scripts/image_oracle/run.py --python <oracle python> \\
      probe_load.py --dit <file.gguf> --config <official repo dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import fixtures
import oracle

# First matrix of each codec among these names is compared. Their
# diffusers names equal the GGUF names.
_CANDIDATES = (
    [f"layers.{i}.feed_forward.w{j}.weight" for i in range(30)
     for j in (1, 2, 3)]
    + [f"layers.{i}.adaLN_modulation.0.weight" for i in range(30)]
    + ["noise_refiner.0.feed_forward.w1.weight", "cap_embedder.1.weight"])


def _gap(got: np.ndarray, ref: np.ndarray) -> dict:
    diff = np.abs(got - ref)
    return {"rel_l2": fixtures.rel_l2(got, ref),
            "max_abs": float(diff.max()),
            "max_abs_over_absmax": float(diff.max() / np.abs(ref).max()),
            "frac_differ": float(np.mean(diff > 0))}


def dequant_gaps(model, gguf_path: Path) -> dict:
    from diffusers.quantizers.gguf.utils import (GGUFParameter,
                                                 dequantize_gguf_tensor)
    from gguf import GGUFReader, quants

    params = dict(model.named_parameters())
    reader = GGUFReader(str(gguf_path))
    by_name = {t.name: t for t in reader.tensors}
    out: dict = {}
    for name in _CANDIDATES:
        t = by_name.get(name)
        p = params.get(name)
        if t is None or not isinstance(p, GGUFParameter):
            continue
        codec = t.tensor_type.name
        if codec in out:
            continue
        ref = quants.dequantize(t.data, t.tensor_type).astype(np.float64)
        entry: dict = {"tensor": name, "shape": list(ref.shape)}
        for kind in oracle.DEQUANT:
            oracle.set_dequant(kind)
            got = oracle.to_numpy(dequantize_gguf_tensor(p), np.float64)
            entry[kind] = _gap(got.reshape(ref.shape), ref)
        out[codec] = entry
    oracle.set_dequant("exact")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dit", required=True, type=Path, help="DiT GGUF file")
    ap.add_argument("--config", required=True, type=Path,
                    help="local copy of the official repository")
    ap.add_argument("--class-name", default="ZImageTransformer2DModel")
    ap.add_argument("--mode", choices=oracle.MODES, default="fp32")
    ap.add_argument("--hash", action="store_true",
                    help="also record the file's sha256")
    ap.add_argument("--out", type=Path,
                    help="JSON report (default: <fixtures>/probe_load_"
                         "<file stem>.json)")
    a = ap.parse_args(argv)
    oracle.setup()
    t0 = time.perf_counter()
    model = oracle.load_dit(a.dit, a.config, a.mode, a.class_name)
    load_s = time.perf_counter() - t0
    ident = (fixtures.file_identity(a.dit) if a.hash
             else {"name": a.dit.name, "size": a.dit.stat().st_size})
    report = {
        "file": ident,
        "class": type(model).__name__,
        "mode": a.mode,
        "dequant": oracle.dequant_mode(),
        "load_seconds": round(load_s, 2),
        **oracle.gguf_census(model),
        "dequant_vs_gguf_py": dequant_gaps(model, a.dit),
        "oracle": fixtures.environment(),
    }
    out = a.out or (fixtures.fixture_root(create=True)
                    / f"probe_load_{a.dit.stem}.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "oracle"},
                     indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
