"""Minted GGUF headers for the adapter refusal tests: a tiny base, tied or
untied, and a LoRA adapter with or without a token embedding pair."""

from __future__ import annotations

import numpy as np

V, D, R = 32, 8, 2


def write(path, arch, tensors, kv=()):
    import gguf

    w = gguf.GGUFWriter(str(path), arch)
    for key, kind, value in kv:
        getattr(w, f"add_{kind}")(key, value)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def mint_base(path, *, arch="llama", tied=False, vocab=V, kv=()):
    tensors = {"token_embd.weight": np.zeros((vocab, D), np.float32),
               "blk.0.attn_q.weight": np.zeros((D, D), np.float32)}
    if not tied:
        tensors["output.weight"] = np.zeros((vocab, D), np.float32)
    return write(path, arch, tensors, kv)


def mint_adapter(path, *, arch="llama", embd=True, vocab=V):
    tensors = {"blk.0.attn_q.weight.lora_a": np.zeros((R, D), np.float32),
               "blk.0.attn_q.weight.lora_b": np.zeros((D, R), np.float32)}
    if embd:
        tensors["token_embd.weight.lora_a"] = np.zeros((vocab, R), np.float32)
        tensors["token_embd.weight.lora_b"] = np.zeros((D, R), np.float32)
    kv = (("adapter.type", "string", "lora"), ("adapter.lora.alpha", "float32", 2.0))
    return write(path, arch, tensors, kv)
