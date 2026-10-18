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
  a float32 copy of the 8-bit one. Each floor allows 1.5 times the gap the
  reference implementation itself shows between a bfloat16 copy of the
  same weights and float32, rounded down. Its worst cosines are 0.99970
  for the bf16 checkpoint and 0.99934 for the 8-bit one, on the GPU and the
  CPU. On the CPU the quantized EmbeddingGemma runs in float32 and sits
  closer. Another quantized checkpoint runs in float32 on the CPU only when
  it stores bfloat16; one stored in float16 stays float16.
- The GGUF, a Q8_0 quantization, is held to the bf16 checkpoint's
  references in bfloat16 and in float32 (a float16 request). Its floor sits
  below the worst cosine measured, 0.99878 in bfloat16 on the CPU, where
  the Q8_0 matmul is less exact than on the GPU in every dtype.
- The bf16 ModernBERT floor allows for bfloat16 rounding, which moves both
  implementations about as far from a float32 run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import host_box
import numpy as np
import pytest

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "embedders_reference"

# checkpoint name -> (fixture, text family, cosine floor)
_MODELS = {
    "embeddinggemma-300m-bf16": ("embeddinggemma-300m-f32", "gemma", 0.9995),
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
_GGUF_FLOOR = 0.9985


@pytest.fixture
@host_box.gate
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
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-4, f"{i}: norm {norm:.6f}, not 1"
        cos = float(v @ r / (norm * np.linalg.norm(r)))
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


@pytest.mark.parametrize("name", ["bge-m3-mlx-8bit",
                                  "snowflake-arctic-embed-l-v2.0-8bit"])
def test_a_copy_without_the_model_card_pools_the_first_token(name, mlx_dir,
                                                            tmp_path):
    # A copy without README.md, as a download that keeps only the weights,
    # configs and tokenizer files leaves it. Its config.json names no model,
    # so the folder name gives the family.
    import shutil

    from gmlx.models.embedders import checkpoint

    src = mlx_dir / f"mlx-community__{name}"
    if not src.is_dir():
        pytest.skip(f"no mlx-community__{name} under GMLX_TEST_MLX_DIR")
    config = json.loads((src / "config.json").read_text())
    assert not config.get("_name_or_path")
    dst = tmp_path / src.name
    dst.mkdir()
    for f in ("config.json", "modules.json"):
        shutil.copyfile(src / f, dst / f)
    assert checkpoint.pooling_mode(str(dst), config["model_type"]) == "cls"


@pytest.mark.parametrize("policy", ["auto", "float16"])
def test_embeddinggemma_gguf_matches_the_reference(policy, gguf_index,
                                                    no_mlx_embeddings,
                                                    monkeypatch):
    import mlx.core as mx

    import gmlx.serve.embeddings as emb
    from gmlx.load import dtypes
    from gmlx.load.loader import load_model

    paths = gguf_index.get("gemma-embedding")
    if not paths:
        pytest.skip("no gemma-embedding GGUF under KQUANT_TEST_GGUF_DIR")
    path = paths[0]
    # A float16 request runs EmbeddingGemma in float32.
    monkeypatch.setenv(dtypes.ENV_VAR, policy)
    model, _config, tok = load_model(path, verbose=False)
    assert emb._is_gguf_encoder(model)
    want = mx.float32 if policy == "float16" else dtypes.activation_dtype("gemma_embedding")
    assert model.model.embed_tokens(mx.array([[1, 2]])).dtype == want
    _check(_reference(_GGUF_REFERENCE), _texts("gemma"),
           lambda t: emb._embed_texts_gguf_encoder(model, tok, t)[0],
           _GGUF_FLOOR)
