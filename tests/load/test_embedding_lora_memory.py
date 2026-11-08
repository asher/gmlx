"""The prompt embedding's memory under a token embedding adapter: a 32k
token prompt embedded through Qwen3.5-9B's table with a rank-16 adapter
peaks within one extra prompt-sized array of the bare lookup, plus 10
percent. The serve engine embeds a whole prompt in one call, so the
delta must not build a float32 copy of the rows.

``integration``; finds Qwen3.5-9B-Q4_K_M.gguf under ``KQUANT_TEST_GGUF_DIR``
and skips without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# A device import, so the GPU gate takes this file.
import mlx.core as mx  # noqa: E402

pytestmark = pytest.mark.integration

GGUF = "Qwen3.5-9B-Q4_K_M.gguf"
T = 32768
RANK = 16


def _peak(fn) -> int:
    """Peak bytes over the active memory before ``fn`` runs."""
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    y = fn()
    mx.eval(y)
    mx.synchronize()
    peak = mx.get_peak_memory() - before
    del y
    return peak


def test_the_adapted_prompt_embedding_peaks_within_one_extra_array(gguf_index):
    from gmlx.load.loader import load_model
    from gmlx.load.modules import LoRAKQuantEmbedding
    from gmlx.serve.bridge_vlm import _find_token_embedding

    path = next((p for p in gguf_index.get("qwen35", []) if Path(p).name == GGUF), None)
    if path is None:
        pytest.skip(f"{GGUF} is not under KQUANT_TEST_GGUF_DIR")
    model, _config, _tok = load_model(path)
    try:
        base = _find_token_embedding(model)
        n_vocab, dims = int(base.num_embeddings), int(base.dims)
        a = mx.ones((n_vocab, RANK)) * 1e-3
        b = mx.ones((dims, RANK)) * 1e-3
        wrap = LoRAKQuantEmbedding(base, a, b, 1.0)
        ids = (mx.arange(T, dtype=mx.uint32) * 7919) % n_vocab
        mx.eval(ids)
        bare = _peak(lambda: base(ids[None]))
        adapted = _peak(lambda: wrap(ids[None]))
        row_bytes = T * dims * wrap(ids[:1][None]).dtype.size
        print(f"[embd-lora] bare {bare / 2**20:.0f} MiB, adapted "
              f"{adapted / 2**20:.0f} MiB, one prompt array {row_bytes / 2**20:.0f} MiB")
        assert adapted <= bare + 1.1 * row_bytes
    finally:
        del model
        mx.clear_cache()
