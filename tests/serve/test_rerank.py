#!/usr/bin/env python3
"""Reranking core (``gmlx.rerank``): GGUF ref resolution, request validation,
the yes/no single-token guard, the P(yes) scoring math (fake model on CPU), and
the run_rerank driver (sorting / top_n / response shape). CPU only - the loader is
stubbed, no real model is loaded; only tiny scoring math touches mlx (forced onto
the CPU device)."""
from __future__ import annotations

import math

import pytest

from gmlx import rerank as rr  # noqa: E402
from gmlx.config import ConfigError  # noqa: E402
from gmlx.config import build_config  # noqa: E402

GGUF = "/models/Qwen3-Reranker-4B.Q6_K.gguf"
YES_ID, NO_ID, VOCAB = 10, 11, 16


@pytest.fixture
def cpu_mx():
    mx = pytest.importorskip("mlx.core")
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(prev)


class _Tok:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"yes": YES_ID, "no": NO_ID}.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


# resolve_rerank_model
def test_resolve_rerank_local_alias_and_bad(tmp_path, monkeypatch):
    f = tmp_path / "r.gguf"
    f.write_bytes(b"GGUF")
    assert rr.resolve_rerank_model(str(f)) == str(f)
    with pytest.raises(ConfigError):
        rr.resolve_rerank_model("some/repo")          # not a GGUF ref / alias
    # aliases (+ True/default) resolve their hf: ref via config.resolve_path;
    # stub it so no HF cache is needed.
    monkeypatch.setattr("gmlx.config.resolve_path",
                        lambda v, dirs: f"LOCAL:{v}")
    default_ref = rr.RERANK_ALIASES[rr.DEFAULT_RERANK_ALIAS]
    for v in (True, "default", "qwen3-rerank-0.6b"):
        assert rr.resolve_rerank_model(v) == f"LOCAL:{default_ref}"
    for alias, ref in rr.RERANK_ALIASES.items():
        assert rr.resolve_rerank_model(alias) == f"LOCAL:{ref}"


def test_rerank_preset_lookup():
    for p in rr.RERANK_PRESETS:
        got = rr.rerank_preset(p["alias"])
        assert got is p
        # the bare alias resolves to the preset's default rung
        assert rr.RERANK_ALIASES[p["alias"]] == p["quants"][p["default_quant"]]
    assert rr.rerank_preset("no-such-alias") is None
    assert rr.rerank_preset(None) is None


def test_resolve_rerank_relative_path_searches_model_dirs(tmp_path):
    sub = tmp_path / "org__repo-GGUF"
    sub.mkdir()
    (sub / "rerank.gguf").write_bytes(b"GGUF")
    got = rr.resolve_rerank_model("org__repo-GGUF/rerank.gguf", [str(tmp_path)])
    assert got == str(sub / "rerank.gguf")


def test_config_rerank_key_round_trips():
    assert build_config({"server": {"rerank": "/x/r.gguf"}}).rerank == "/x/r.gguf"
    assert build_config({"server": {}}).rerank is None
    assert build_config({"server": {"rerank": False}}).rerank is None


# request validation
def test_normalize_query():
    assert rr._normalize_query("hi") == "hi"
    for bad in (None, "", "  ", 5):
        with pytest.raises(rr.RerankRequestError, match="query"):
            rr._normalize_query(bad)


def test_normalize_documents():
    assert rr._normalize_documents(["a", "b"]) == ["a", "b"]
    assert rr._normalize_documents([{"text": "x"}, "y"]) == ["x", "y"]
    for bad, m in [(None, "required"), ([], "non-empty"),
                   ([1, 2], "string"), ("a", "non-empty list")]:
        with pytest.raises(rr.RerankRequestError, match=m):
            rr._normalize_documents(bad)


# yes/no single-token resolution + guard
def test_single_token_id_prefers_vocab_then_falls_back():
    class T:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"yes": YES_ID}.get(token, 0)

        def encode(self, text, add_special_tokens=False):
            return [99]

    assert rr._single_token_id(T(), "yes") == YES_ID       # exact vocab entry
    assert rr._single_token_id(T(), "missing") == 99       # 1-token encode fallback


def test_yes_no_ids_guard_rejects_non_reranker():
    class Bad:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return 0                                        # always unk

        def encode(self, text, add_special_tokens=False):
            return [1, 2]                                   # never a single token

    with pytest.raises(RuntimeError, match="Qwen3-Reranker"):
        rr._yes_no_ids(Bad())


# scoring math
def test_score_documents_is_sigmoid_of_yes_minus_no(cpu_mx):
    class Model:
        def __call__(self, ids_arr):
            import numpy as np
            T = ids_arr.shape[1]
            arr = np.zeros((1, T, VOCAB), dtype=np.float32)
            arr[0, -1, YES_ID] = 2.0
            arr[0, -1, NO_ID] = 0.0
            return cpu_mx.array(arr)

    scores, n_tokens = rr._score_documents(
        Model(), _Tok(), "q", ["d1", "d2"], "instr")
    expected = 1.0 / (1.0 + math.exp(-2.0))
    assert len(scores) == 2
    assert all(abs(s - expected) < 1e-5 for s in scores)
    assert n_tokens == 2 * (3 + 3 + 3)                     # prefix+middle+suffix per doc


# run_rerank driver (scorer stubbed)
def _stub_scorer(monkeypatch, scores, n_tokens=30):
    monkeypatch.setattr(rr._GGUFRerankHolder, "get",
                        classmethod(lambda cls, p: ("M", "T")))
    monkeypatch.setattr(rr, "_score_documents",
                        lambda m, t, q, docs, instr: (list(scores), n_tokens))


def test_run_rerank_sorts_truncates_and_shapes(monkeypatch):
    _stub_scorer(monkeypatch, [0.1, 0.9, 0.5], n_tokens=30)
    out = rr.run_rerank("q", ["a", "b", "c"], configured_model=GGUF, top_n=2)
    assert [r["index"] for r in out["results"]] == [1, 2]   # desc, truncated to top-2
    assert out["results"][0]["relevance_score"] == 0.9
    assert out["results"][0]["document"]["text"] == "b"
    assert out["usage"]["total_tokens"] == 30
    assert out["model"] == "reranker"           # model='' -> advertised id, not path


def test_run_rerank_echoes_requested_model_and_can_drop_documents(monkeypatch):
    _stub_scorer(monkeypatch, [0.2, 0.8])
    out = rr.run_rerank("q", ["a", "b"], configured_model=GGUF,
                        model="rerank-english-v3.0", return_documents=False)
    assert out["model"] == "rerank-english-v3.0"
    assert "document" not in out["results"][0]


def test_run_rerank_validation(monkeypatch):
    _stub_scorer(monkeypatch, [0.0])
    with pytest.raises(rr.RerankRequestError, match="query"):
        rr.run_rerank("", ["a"], configured_model=GGUF)
    with pytest.raises(rr.RerankRequestError, match="documents"):
        rr.run_rerank("q", [], configured_model=GGUF)
    with pytest.raises(rr.RerankRequestError, match="top_n"):
        rr.run_rerank("q", ["a"], configured_model=GGUF, top_n=0)
    with pytest.raises(rr.RerankRequestError, match="top_n"):
        rr.run_rerank("q", ["a"], configured_model=GGUF, top_n="x")
    # bool is an int subclass; float() truncates. Both used to coerce silently
    # (`true` -> 1 result, `2.9` -> 2) instead of the documented 400.
    with pytest.raises(rr.RerankRequestError, match="top_n"):
        rr.run_rerank("q", ["a"], configured_model=GGUF, top_n=True)
    with pytest.raises(rr.RerankRequestError, match="top_n"):
        rr.run_rerank("q", ["a"], configured_model=GGUF, top_n=2.9)


def test_run_rerank_wraps_backend_errors(monkeypatch):
    monkeypatch.setattr(rr._GGUFRerankHolder, "get",
                        classmethod(lambda cls, p: ("M", "T")))

    def boom(*a, **k):
        raise RuntimeError("reranker exploded")

    monkeypatch.setattr(rr, "_score_documents", boom)
    with pytest.raises(RuntimeError, match="reranking failed: reranker exploded"):
        rr.run_rerank("q", ["a"], configured_model=GGUF)


def test_run_rerank_missing_file_propagates(monkeypatch):
    # Mirror of the embeddings contract: FileNotFoundError reaches the endpoint
    # unwrapped so its 404 model_file_missing branch fires instead of a 500.
    def gone(cls, p):
        raise FileNotFoundError(2, "No such file or directory", p)

    monkeypatch.setattr(rr._GGUFRerankHolder, "get", classmethod(gone))
    with pytest.raises(FileNotFoundError):
        rr.run_rerank("q", ["a"], configured_model=GGUF)


# prewarm
def test_prewarm_loads_in_background(monkeypatch):
    loaded = []
    monkeypatch.setattr(rr, "_load_rerank_model", lambda p: loaded.append(p))
    rr.prewarm(GGUF).result(timeout=5)
    assert loaded == [GGUF]


def test_prewarm_is_best_effort_on_load_failure(monkeypatch, capsys):
    def boom(_p):
        raise RuntimeError("load failed")

    monkeypatch.setattr(rr, "_load_rerank_model", boom)
    rr.prewarm(GGUF).result(timeout=5)                        # must not raise
    assert "rerank prewarm failed" in capsys.readouterr().err
