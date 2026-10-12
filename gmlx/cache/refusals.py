# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Members that keep a KVCache subclass out of batching and KV quantization.

A gmlx KVCache subclass whose rows carry state a plain KVCache row lacks
(an indexer key stream, or a layer held at fp16) passes mlx-vlm's
isinstance checks against KVCache. These members keep it out of the paths
that would drop that state:

- ``to_batch``: the mlx-vlm batch constructor calls it before its
  isinstance chain. It raises the ValueError that chain raises for a class
  it does not know.
- ``preserve_auxiliary_kv_state``: the hybrid and TurboQuant arms of the
  mlx-vlm KV quantizer pass the layer through.
- ``extract``: hidden, so the prefix cache probes for a per-row
  ``extract`` find none.

``to_quantized`` stays as each class defines it, so the uniform arm and
gmlx's safe quantizer keep calling it.
"""

from __future__ import annotations


def refuse_batch(self, left_padding):
    raise ValueError(f"{type(self)} does not yet support batching")


def _no_extract(self):
    raise AttributeError(
        f"{type(self).__name__!r} object has no attribute 'extract'")


no_extract = property(_no_extract)

# For classes built with type(), such as the fp16 holds in kv_policy.
MEMBERS = {
    "to_batch": refuse_batch,
    "preserve_auxiliary_kv_state": True,
    "extract": no_extract,
}
