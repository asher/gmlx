# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A tiny LFM3 GGUF for header tests: the family's metadata keys and the
three tensors the config synthesis reads shapes from. It holds no model
weights and no template text."""

from __future__ import annotations

import numpy as np

ARCH = "lfm3"
KV_HEADS = [0, 0, 2, 0, 2]
SLIDING = [False, False, False, False, True]


def write_lfm3_header(path, *, context: int = 128000, templates=(),
                      causal: bool | None = None,
                      decision: str | None = None) -> str:
    """Write the GGUF and return its path. ``templates`` are the names in
    ``tokenizer.chat_templates``. ``causal`` and ``decision`` are the two
    keys a d1 decision model carries."""
    import gguf

    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_uint32(f"{ARCH}.embedding_length", 64)
    w.add_uint32(f"{ARCH}.block_count", len(KV_HEADS))
    w.add_uint32(f"{ARCH}.attention.head_count", 4)
    w.add_array(f"{ARCH}.attention.head_count_kv", list(KV_HEADS))
    w.add_uint32(f"{ARCH}.feed_forward_length", 128)
    w.add_uint32(f"{ARCH}.context_length", context)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32(f"{ARCH}.attention.key_length", 16)
    w.add_uint32(f"{ARCH}.attention.value_length", 16)
    w.add_uint32(f"{ARCH}.attention.sliding_window", 3)
    w.add_array(f"{ARCH}.attention.sliding_window_pattern", list(SLIDING))
    w.add_float32(f"{ARCH}.rope.freq_base", 1000000.0)
    w.add_float32(f"{ARCH}.rope.freq_base_swa", 10000.0)
    w.add_uint32(f"{ARCH}.shortconv.l_cache", 7)
    w.add_array("tokenizer.ggml.tokens", ["t"] * 32)
    if templates:
        w.add_array("tokenizer.chat_templates", list(templates))
    if causal is not None:
        w.add_bool(f"{ARCH}.attention.causal", causal)
    if decision is not None:
        w.add_string(f"{ARCH}.decision.type", decision)
    w.add_tensor("rope_freqs.weight",
                 np.array([1.0] * 4 + [1e30] * 4, dtype=np.float32))
    for name in ("post_attention_norm", "post_ffw_norm"):
        w.add_tensor(f"blk.0.{name}.weight", np.ones(64, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)
