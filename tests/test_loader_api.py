#!/usr/bin/env python3
"""Public ``load_model`` contract, end to end on a synthetic GGUF.

docs/python.md promises ``load_model`` returns ``(model, config, tokenizer)``
and that GGUF-declared stop tokens (eot/eom) fold into the tokenizer wrapper's
public ``eos_token_ids`` stop set - the one the samplers read. Both are pinned
here through the REAL load path: a tiny fully-valid llama-arch GGUF (F32
weights, minimal ByteLevel BPE vocab) minted with gguf-py, no model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from gguf import GGUFWriter  # noqa: E402
from tokenizers import pre_tokenizers  # noqa: E402

# Vocab mirrors tests/test_tokenizer.py: 3 specials + the byte alphabet + two
# merged tokens, so encode() round-trips ASCII with no download.
_SPECIALS = ["<s>", "</s>", "<pad>"]
_ALPHABET = sorted(pre_tokenizers.ByteLevel.alphabet())
_MERGED = ["He", "wo"]
_MERGES = ["H e", "w o"]
_TOKENS = _SPECIALS + _ALPHABET + _MERGED

_HID, _LAYERS, _HEADS, _KV, _FFN, _HD = 64, 2, 4, 2, 128, 16


def _mint_tiny_llama(path: str) -> None:
    """A complete, loadable llama-arch GGUF: full KV metadata + every tensor
    the 2-layer model needs (tied embeddings; F32 so no quant kernels)."""
    vocab = len(_TOKENS)
    w = GGUFWriter(path, "llama")
    w.add_uint32("llama.embedding_length", _HID)
    w.add_uint32("llama.block_count", _LAYERS)
    w.add_uint32("llama.attention.head_count", _HEADS)
    w.add_uint32("llama.attention.head_count_kv", _KV)
    w.add_uint32("llama.feed_forward_length", _FFN)
    w.add_uint32("llama.context_length", 1024)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32("llama.rope.dimension_count", _HD)
    w.add_float32("llama.rope.freq_base", 10000.0)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "llama-bpe")
    w.add_array("tokenizer.ggml.tokens", _TOKENS)
    w.add_array("tokenizer.ggml.merges", _MERGES)
    w.add_array("tokenizer.ggml.token_type", [3, 3, 3] + [1] * (vocab - 3))
    w.add_uint32("tokenizer.ggml.bos_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    # A declared turn-end token distinct from eos (the GLM shape): must land
    # in the wrapper's public stop set, not just tokenizer-private state.
    w.add_uint32("tokenizer.ggml.eot_token_id", 2)

    rng = np.random.default_rng(0)

    def t(name, *shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32) * 0.02)

    t("token_embd.weight", vocab, _HID)
    t("output_norm.weight", _HID)
    for i in range(_LAYERS):
        t(f"blk.{i}.attn_norm.weight", _HID)
        t(f"blk.{i}.ffn_norm.weight", _HID)
        t(f"blk.{i}.attn_q.weight", _HEADS * _HD, _HID)
        t(f"blk.{i}.attn_k.weight", _KV * _HD, _HID)
        t(f"blk.{i}.attn_v.weight", _KV * _HD, _HID)
        t(f"blk.{i}.attn_output.weight", _HID, _HEADS * _HD)
        t(f"blk.{i}.ffn_gate.weight", _FFN, _HID)
        t(f"blk.{i}.ffn_up.weight", _FFN, _HID)
        t(f"blk.{i}.ffn_down.weight", _HID, _FFN)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """One real load_model() call shared by the contract tests."""
    import gmlx

    p = tmp_path_factory.mktemp("loader_api") / "tiny-llama.gguf"
    _mint_tiny_llama(str(p))
    return gmlx.load_model(str(p))


def test_load_model_returns_triple(loaded):
    # The documented return shape: (model, config, tokenizer).
    assert isinstance(loaded, tuple) and len(loaded) == 3
    model, config, tokenizer = loaded
    assert isinstance(config, dict)
    assert config["model_type"] == "llama"
    assert config["vocab_size"] == len(_TOKENS)
    ids = tokenizer.encode("Hello")
    assert isinstance(ids, list) and ids
    # The model is a live mlx-lm module: a forward yields vocab-wide logits.
    import mlx.core as mx

    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, len(_TOKENS))


def test_load_model_folds_gguf_eot_into_wrapper_stop_set(loaded):
    # The promise is the PUBLIC wrapper stop set (what the samplers read),
    # not the private _gguf_eos_token_ids stash on the raw tokenizer.
    _, _, tokenizer = loaded
    assert 1 in tokenizer.eos_token_ids            # primary eos
    assert 2 in tokenizer.eos_token_ids            # declared eot folded in
