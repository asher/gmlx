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


def _mint_tiny_llama(path: str, *, yarn: bool = False) -> None:
    """A complete, loadable llama-arch GGUF: full KV metadata + every tensor
    the 2-layer model needs (tied embeddings; F32 so no quant kernels).
    ``yarn`` adds rope-scaling KVs so the built model carries a YarnRoPE with
    a precomputed ``_freqs`` - a lazy non-parameter array, the gemma-4 shape
    the background-load test needs."""
    vocab = len(_TOKENS)
    w = GGUFWriter(path, "llama")
    if yarn:
        w.add_string("llama.rope.scaling.type", "yarn")
        w.add_float32("llama.rope.scaling.factor", 2.0)
        w.add_uint32("llama.rope.scaling.original_context_length", 512)
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


def test_background_thread_load_generates_on_main_thread(tmp_path_factory):
    """Chat background load and the server preload/keep-warm load on one
    thread and generate on another. MLX default streams are per-thread: any
    array a load leaves lazy is bound to a stream only the loading thread can
    evaluate, and the first forward elsewhere dies with "There is no
    Stream(gpu, N) in current thread" (gemma-4's ProportionalRoPE ``_freqs``
    was the first casualty). load_model must hand back a fully materialized
    tree, non-parameter attributes included."""
    import threading

    import mlx.core as mx
    from mlx.utils import tree_flatten

    import gmlx

    p = tmp_path_factory.mktemp("loader_bg") / "tiny-llama-bg.gguf"
    _mint_tiny_llama(str(p), yarn=True)
    box: list = []
    t = threading.Thread(target=lambda: box.append(gmlx.load_model(str(p))))
    t.start()
    t.join()
    model, _config, _tokenizer = box[0]
    # Deep-eval the FULL module tree (underscore attrs included, which
    # parameters() filters) and run a forward - all on this thread.
    mx.eval([a for _, a in tree_flatten(dict(model)) if isinstance(a, mx.array)])
    mx.eval(model(mx.array([[1, 2, 3]])))


def test_materialize_module_arrays_reaches_non_parameters():
    """The loader guard must materialize arrays parameters() skips. The
    hazard is pinned first: an unmaterialized lazy from a dead thread raises
    at eval. The raise comes from Metal's thread-local encoder maps, so where
    MLX runs on CPU (GPU-less CI runners) or a future mlx makes cross-thread
    eval legal, the hazard is absent and the pin skips - a skip on Metal means
    this test and ``materialize_module_arrays`` can both be retired."""
    import threading

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from gmlx.loader import materialize_module_arrays

    class Rope(nn.Module):
        def __init__(self):
            super().__init__()
            self._freqs = 2.0 ** mx.arange(4, dtype=mx.float32)

    # parameters() filters underscore attrs - the reason the load-path
    # weight evals never reach _freqs and a dedicated guard exists.
    assert not tree_flatten(Rope().parameters())

    def build(materialize: bool):
        box: list = []

        def bg():
            m = Rope()
            if materialize:
                materialize_module_arrays(m)
            box.append(m)

        t = threading.Thread(target=bg)
        t.start()
        t.join()
        return box[0]

    # The guard must be a no-op for correctness everywhere, hazard or not.
    assert build(materialize=True)._freqs.tolist() == [1.0, 2.0, 4.0, 8.0]

    try:
        mx.eval(build(materialize=False)._freqs)
    except RuntimeError as e:
        assert "in current thread" in str(e)          # the pinned hazard
    else:
        pytest.skip("cross-thread eval is legal on this mlx/device - "
                    "no hazard to pin here")
