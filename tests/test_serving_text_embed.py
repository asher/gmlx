#!/usr/bin/env python3
"""Plain-text serving wrap: the token-embedding probe that lets mlx-vlm's batched
engine GPU-embed an mlx-lm text model.

mlx-vlm's ``text_only.LanguageModel._token_embedding`` probes only the top level
and one ``.model`` hop - the canonical mlx-lm dense layout
(``Model.model.embed_tokens``). The hybrid qwen3_5/3.6 *dense* family keeps a
VLM-shaped ``Model.language_model.model.embed_tokens`` nesting even for its text
checkpoint, one hop deeper, so the engine's GPU-embed step raised
``ValueError: ... does not expose token embeddings`` at serve time (a crash no
prior test caught: every text tier ran a shallow-layout model, and the one
deep-nested family in the matrix only ran the MTP target, which embeds itself).

``serving._ensure_text_embedding_probe`` closes that: a no-op when the stock
probe already resolves, a one-instance override when it doesn't, and a fail-fast
at *load* when no embedding is reachable at all. CPU-only - fake module trees, no
GGUF, no GPU, no real model load."""
from __future__ import annotations

import types

import pytest

pytest.importorskip("mlx_vlm")

from gmlx import server_bridge_vlm as serving  # noqa: E402


def _emb(tag):
    """A stand-in embedding: callable like ``nn.Embedding`` (``emb(ids)``)."""
    return lambda ids: (tag, ids)


def _raw_top(tag="top"):
    """Embedding at the top level (``Model.embed_tokens``)."""
    return types.SimpleNamespace(embed_tokens=_emb(tag), model_type="top")


def _raw_shallow(tag="shallow"):
    """Canonical mlx-lm dense layout (``Model.model.embed_tokens``: llama / qwen2
    / qwen3 / gemma / ...) - reachable by mlx-vlm's stock probe."""
    inner = types.SimpleNamespace(embed_tokens=_emb(tag))
    return types.SimpleNamespace(model=inner, model_type="qwen3")


def _raw_deep(tag="deep"):
    """VLM-shaped dense layout (``Model.language_model.model.embed_tokens``: hybrid
    qwen3_5/3.6 dense) - one hop past the stock probe."""
    inner = types.SimpleNamespace(embed_tokens=_emb(tag))
    lang = types.SimpleNamespace(model=inner)
    return types.SimpleNamespace(language_model=lang, model_type="qwen3_5")


def _wrap(raw):
    """Wrap a raw mlx-lm-shaped model the way ``load_serveable_model`` does."""
    return serving.TextOnlyModel(
        raw, config={"model_type": getattr(raw, "model_type", "x")})


# _find_token_embedding: walks every known nesting
@pytest.mark.parametrize("raw_fn", [_raw_top, _raw_shallow, _raw_deep])
def test_find_token_embedding_walks_known_nestings(raw_fn):
    emb = serving._find_token_embedding(raw_fn("t"))
    assert emb is not None and emb("ids") == ("t", "ids")


def test_find_token_embedding_none_when_absent():
    assert serving._find_token_embedding(types.SimpleNamespace()) is None


# _ensure_text_embedding_probe: override deep, leave shallow, fail-fast on none
def test_deep_nesting_gets_probe_override():
    raw = _raw_deep("deep")
    model = _wrap(raw)
    # stock probe cannot reach language_model.model.embed_tokens -> would crash
    assert model.language_model._token_embedding() is None
    serving._ensure_text_embedding_probe(model, raw)
    # now resolvable: the engine's get_input_embeddings -> input_embeds works
    assert model.language_model.input_embeds("ids") == ("deep", "ids")


def test_shallow_nesting_left_untouched():
    raw = _raw_shallow("shallow")
    model = _wrap(raw)
    before = model.language_model._token_embedding()
    assert before is not None                       # stock probe already resolves
    serving._ensure_text_embedding_probe(model, raw)   # no-op
    assert model.language_model._token_embedding() is before
    assert model.language_model.input_embeds("ids") == ("shallow", "ids")


def test_top_level_embedding_left_untouched():
    raw = _raw_top("top")
    model = _wrap(raw)
    serving._ensure_text_embedding_probe(model, raw)
    assert model.language_model.input_embeds("ids") == ("top", "ids")


def test_no_embedding_anywhere_fails_fast_at_load():
    raw = types.SimpleNamespace(model_type="mystery")   # no embedding at all
    model = _wrap(raw)
    with pytest.raises(RuntimeError, match="no token embedding"):
        serving._ensure_text_embedding_probe(model, raw)
