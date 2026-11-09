"""Write the float32 copy of a quantized EmbeddingGemma 2 build's weights
that embeddinggemma2_reference.py records a build's reference from, as an
npz keyed by google/embeddinggemma-2's tensor names.

- A GGUF: every text tensor, dequantized with gguf-py.
- An MLX build: every quantized tensor, tower linears included (affine,
  mx.dequantize). Its other tensors match google's.

The rest stays google's, so a build's floor measures gmlx and not the
build's quantization.

    python scripts/embedder_oracle/embeddinggemma2_dequant.py <build> <out.npz>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def _gguf(path: str) -> dict:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    from gmlx.load.remap import ARCH_PRIORITY_OVERRIDES

    table = ARCH_PRIORITY_OVERRIDES["GEMMA_EMBEDDING2"]
    out = {}
    for t in GGUFReader(path).tensors:
        for rx, target, _ in table:
            m = re.match(rx, t.name)
            if m:
                name = target.format(bid=m.group(1)) if m.groups() else target
                break
        else:
            raise SystemExit(f"no google name for {t.name}")
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        out[name] = a.reshape([int(d) for d in reversed(t.shape)]) if a.ndim == 1 and \
            len(t.shape) > 1 else a
    return out


def _mlx(path: str) -> dict:
    import mlx.core as mx

    folder = Path(path)
    q = json.loads((folder / "config.json").read_text())["quantization"]
    w = mx.load(str(folder / "model.safetensors"))
    out = {}
    for k in w:
        if not k.endswith(".scales"):
            continue
        base = k[: -len(".scales")]
        a = mx.dequantize(w[base + ".weight"], w[k], w[base + ".biases"],
                          group_size=q["group_size"], bits=q["bits"])
        out[base + ".weight"] = np.array(a.astype(mx.float32))
    return out


def main(build: str, out: str) -> None:
    weights = _gguf(build) if build.endswith(".gguf") else _mlx(build)
    np.savez(out, **weights)
    print(f"{len(weights)} tensors to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
