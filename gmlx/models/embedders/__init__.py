"""Sentence encoders for ``POST /v1/embeddings``.

gmlx builds these encoders itself: EmbeddingGemma (safetensors
``gemma3_text`` and the ``gemma_embedding`` GGUF), BERT, XLM-RoBERTa and
ModernBERT. A model takes ``(input_ids, attention_mask)`` and returns an
``EncoderOutput`` whose ``text_embeds`` are the pooled, L2-normalized
sentence vectors. :func:`load` reads an MLX safetensors checkpoint, plain
or quantized, with its tokenizer (see :mod:`.checkpoint`).

Importing this package loads nothing: the modules import MLX on use.
"""


def load(model_ref: str):
    """``(model, tokenizer)`` for an encoder folder or Hub repo id."""
    from .checkpoint import load as _load

    return _load(model_ref)
