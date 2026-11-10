# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A tiny LFM3 GGUF for header tests: the family's metadata keys and the
three tensors the config synthesis reads shapes from. It holds no model
weights and no template text. Also the lookup of a real LFM3 model and its
projector, for the integration tier."""

from __future__ import annotations

import numpy as np

ARCH = "lfm3"
KV_HEADS = [0, 0, 2, 0, 2]
SLIDING = [False, False, False, False, True]
# The layer schedule and attention geometry of the shipped 2B models:
# conv at 0, 1 and every odd layer, sliding attention at 4, 8, ..., 20,
# full attention at 2, 6, ..., 22.
SHIPPED_KV_HEADS = [0, 0] + [0 if i % 2 else 4 for i in range(2, 24)]
SHIPPED_SLIDING = [i >= 4 and i % 4 == 0 for i in range(24)]


def write_lfm3_header(path, *, context: int = 128000, templates=(),
                      causal: bool | None = None,
                      decision: str | None = None,
                      shipped: bool = False) -> str:
    """Write the GGUF and return its path. ``templates`` are the names in
    ``tokenizer.chat_templates``. ``causal`` and ``decision`` are the two
    keys a d1 decision model carries. ``shipped`` writes the geometry of
    the shipped 2B models (24 layers, 16 query and 4 KV heads, head dim
    128, window 256) in place of the tiny one."""
    import gguf

    kv_heads = SHIPPED_KV_HEADS if shipped else KV_HEADS
    sliding = SHIPPED_SLIDING if shipped else SLIDING
    hidden, heads, head_dim, ffn, window = (
        (2048, 16, 128, 7680, 256) if shipped else (64, 4, 16, 128, 3))
    w = gguf.GGUFWriter(str(path), ARCH)
    w.add_uint32(f"{ARCH}.embedding_length", hidden)
    w.add_uint32(f"{ARCH}.block_count", len(kv_heads))
    w.add_uint32(f"{ARCH}.attention.head_count", heads)
    w.add_array(f"{ARCH}.attention.head_count_kv", list(kv_heads))
    w.add_uint32(f"{ARCH}.feed_forward_length", ffn)
    w.add_uint32(f"{ARCH}.context_length", context)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32(f"{ARCH}.attention.key_length", head_dim)
    w.add_uint32(f"{ARCH}.attention.value_length", head_dim)
    w.add_uint32(f"{ARCH}.attention.sliding_window", window)
    w.add_array(f"{ARCH}.attention.sliding_window_pattern", list(sliding))
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
    half = head_dim // 2
    w.add_tensor("rope_freqs.weight",
                 np.array([1.0] * (half // 2) + [1e30] * (half - half // 2),
                          dtype=np.float32))
    for name in ("post_attention_norm", "post_ffw_norm"):
        w.add_tensor(f"blk.0.{name}.weight",
                     np.ones(hidden, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def find_vision_pair(gguf_dir, gguf_index) -> tuple[str, str] | None:
    """The (text GGUF, projector) of a real LFM3 model with vision under
    the integration folder, matched by the base name both headers carry.
    The projector can sit in any subfolder."""
    from gmlx.load.headerscan import scan_gguf

    texts: dict = {}
    for path in gguf_index.get(ARCH, ()):
        kv = scan_gguf(path, include_tensors=False).kv
        texts.setdefault(kv.get("general.basename"), path)
    for path in sorted(gguf_dir.rglob("*mmproj*.gguf")):
        kv = scan_gguf(str(path), include_tensors=False).kv
        if (kv.get("clip.vision.projector_type") == "lfm3vl"
                and kv.get("general.basename") in texts):
            return texts[kv["general.basename"]], str(path)
    return None
