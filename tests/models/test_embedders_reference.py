"""gmlx's sentence encoders against reference vectors recorded with
mlx-embeddings and stored in tests/fixtures/embedders_reference, so the check
runs without that package.

Integration tier. ``GMLX_TEST_MLX_DIR`` names a folder of the mlx-community
encoder checkpoints, each as ``mlx-community__<name>`` or ``<name>``, and
``KQUANT_TEST_GGUF_DIR`` supplies the EmbeddingGemma GGUF. An absent
checkpoint skips.

A fixture holds each text's reference vector, recorded for that text alone.
The test embeds each text alone, and the batch texts in one padded request,
through the server's embedding drivers with mlx_embeddings imports blocked,
and holds each cosine to the model's floor:

- The XLM-RoBERTa references are the recorded first-token hidden states,
  normalized, because BGE-M3 and Arctic-Embed pool the first token.
- The EmbeddingGemma texts stay within 257 tokens, where full attention and
  the model's 512-token sliding window give the same result.
- gmlx runs EmbeddingGemma in bfloat16, so its references are recorded in
  float32: from the bf16 checkpoint (which stores float16 weights) and from
  a float32 copy of the 8-bit one. The floor allows for bfloat16 rounding.
  The GGUF, a Q8_0 quantization also run in bfloat16, is held to the bf16
  checkpoint's references.
- The bf16 ModernBERT floor allows for bfloat16 rounding, which moves both
  implementations about as far from a float32 run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "embedders_reference"

# checkpoint name -> (fixture, text family, cosine floor)
_MODELS = {
    "embeddinggemma-300m-bf16": ("embeddinggemma-300m-f32", "gemma", 0.999),
    "embeddinggemma-300m-8bit": ("embeddinggemma-300m-8bit-f32", "gemma", 0.999),
    "nomicai-modernbert-embed-base-bf16": (
        "nomicai-modernbert-embed-base-bf16", "modernbert", 0.999),
    "nomicai-modernbert-embed-base-8bit": (
        "nomicai-modernbert-embed-base-8bit", "modernbert", 0.9999),
    "bge-m3-mlx-8bit": ("bge-m3-mlx-8bit", "xlmr", 0.9999),
    "snowflake-arctic-embed-l-v2.0-8bit": (
        "snowflake-arctic-embed-l-v2.0-8bit", "xlmr", 0.9999),
    "Qwen3-Embedding-0.6B-8bit": ("Qwen3-Embedding-0.6B-8bit", "qwen3", 0.9999),
}
_GGUF_REFERENCE = "embeddinggemma-300m-f32"
_GGUF_FLOOR = 0.999


@pytest.fixture
def mlx_dir() -> Path:
    d = os.environ.get("GMLX_TEST_MLX_DIR")
    if not d:
        pytest.skip("set GMLX_TEST_MLX_DIR to a folder of MLX encoder checkpoints")
    p = Path(d).expanduser()
    if not p.is_dir():
        pytest.skip(f"GMLX_TEST_MLX_DIR={d!r} is not a directory")
    return p


class _Block:
    """A meta path finder that refuses mlx_embeddings and records the try."""

    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, name, path=None, target=None):
        if name == "mlx_embeddings" or name.startswith("mlx_embeddings."):
            self.attempts.append(name)
            raise ImportError(f"blocked import of {name}")
        return None


@pytest.fixture
def no_mlx_embeddings(monkeypatch):
    block = _Block()
    monkeypatch.setattr(sys, "meta_path", [block, *sys.meta_path])
    for name in [m for m in sys.modules if m.split(".")[0] == "mlx_embeddings"]:
        monkeypatch.delitem(sys.modules, name)
    yield
    assert block.attempts == []
    assert not [m for m in sys.modules if m.split(".")[0] == "mlx_embeddings"]


def _texts(family: str) -> dict:
    data = json.loads((_FIXTURES / "texts.json").read_text())
    return {**data["texts"], **data["long"][family]}


def _reference(name: str) -> dict:
    with np.load(_FIXTURES / f"{name}.npz") as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def _check(ref: dict, texts: dict, embed, floor: float):
    """Embed each text alone, and the batch texts in one request, and hold
    every cosine to ``floor``. Returns the worst cosine."""
    ids = sorted(ref)
    got = {i: embed([texts[i]])[0] for i in ids}
    batch = [i for i in ids if i.startswith("b8_")]
    rows = embed([texts[i] for i in batch])
    worst = 1.0
    for i, v in [*got.items(), *((f"{i} in batch", r) for i, r in zip(batch, rows))]:
        r = ref[i.split(" ")[0]]
        v = np.asarray(v, dtype=np.float64)
        cos = float(v @ r / (np.linalg.norm(v) * np.linalg.norm(r)))
        assert cos >= floor, f"{i}: cosine {cos:.6f} < {floor}"
        worst = min(worst, cos)
    return worst


@pytest.mark.parametrize("name", list(_MODELS))
def test_safetensors_encoder_matches_the_reference(name, mlx_dir, no_mlx_embeddings):
    import gmlx.serve.embeddings as emb
    from gmlx.models.embedders import load

    path = next((p for p in (mlx_dir / f"mlx-community__{name}", mlx_dir / name)
                 if p.is_dir()), None)
    if path is None:
        pytest.skip(f"no {name} under GMLX_TEST_MLX_DIR")
    fixture, family, floor = _MODELS[name]
    model, tok = load(str(path))
    _check(_reference(fixture), _texts(family),
           lambda t: emb._embed_texts(model, tok, t)[0], floor)


def test_embeddinggemma_gguf_matches_the_reference(gguf_index, no_mlx_embeddings):
    import gmlx.serve.embeddings as emb
    from gmlx.load.loader import load_model

    paths = gguf_index.get("gemma-embedding")
    if not paths:
        pytest.skip("no gemma-embedding GGUF under KQUANT_TEST_GGUF_DIR")
    path = paths[0]
    model, _config, tok = load_model(path, verbose=False)
    assert emb._is_gguf_encoder(model)
    _check(_reference(_GGUF_REFERENCE), _texts("gemma"),
           lambda t: emb._embed_texts_gguf_encoder(model, tok, t)[0],
           _GGUF_FLOOR)
