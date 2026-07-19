#!/usr/bin/env python3
"""Text-embeddings core (``gmlx.embeddings``): alias resolution, the
request-model policy, input normalization, vector encoding, both embed drivers
(stubbed mlx-embeddings + the GGUF decoder-LM backend with a stubbed loader),
and the background pre-warm. CPU only - neither mlx-embeddings nor a real model
is ever loaded; the GGUF tests exercise pooling math on a fake trunk forced onto
the CPU device."""
from __future__ import annotations

import base64
import sys

import numpy as np
import pytest

from gmlx import embeddings as emb  # noqa: E402
from gmlx.config import build_config  # noqa: E402

EGEMMA = "mlx-community/embeddinggemma-300m-8bit"       # an mlx-tier (encoder) repo
QWEN06_REF = emb._qwen_emb_ref("0.6B", "Q8_0")          # the gguf-tier default hf: ref


# resolve_embeddings_model
def test_resolve_aliases_and_default(monkeypatch):
    # mlx-tier aliases resolve straight to their repo id (no local file needed).
    assert emb.resolve_embeddings_model("embeddinggemma") == EGEMMA
    assert (emb.resolve_embeddings_model("ARCTIC-L")
            == "mlx-community/snowflake-arctic-embed-l-v2.0-8bit")
    # gguf-tier values are hf: refs resolved to a local file via config.resolve_path
    # (stub it so no HF cache is needed); the default + back-compat alias are gguf.
    monkeypatch.setattr("gmlx.config.resolve_path",
                        lambda v, dirs: f"LOCAL:{v}")
    for v in (True, "default", "true"):
        assert emb.resolve_embeddings_model(v) == f"LOCAL:{QWEN06_REF}"
    assert emb.resolve_embeddings_model("qwen3-embed") == f"LOCAL:{QWEN06_REF}"
    # every preset alias resolves: mlx-tier -> repo id, gguf-tier -> LOCAL:<ref>.
    for alias, value in emb.EMBEDDINGS_ALIASES.items():
        expect = f"LOCAL:{value}" if emb._is_gguf_ref(value) else value
        assert emb.resolve_embeddings_model(alias) == expect


def test_embeddinggemma_gguf_is_a_gguf_tier_preset(monkeypatch):
    p = emb.embedding_preset("embeddinggemma-gguf")
    assert p is not None and p["tier"] == "gguf"
    assert list(p["quants"]) == ["Q8_0"]              # the canonical ggml-org rung
    # the alias value is a GGUF ref (encoder GGUF backend), not a safetensors repo
    assert emb._is_gguf_ref(emb.EMBEDDINGS_ALIASES["embeddinggemma-gguf"])
    # ... and resolves to a LOCAL gguf file, like the Qwen gguf-tier refs.
    monkeypatch.setattr("gmlx.config.resolve_path",
                        lambda v, dirs: f"LOCAL:{v}")
    val = emb.resolve_embeddings_model("embeddinggemma-gguf")
    assert val.startswith("LOCAL:hf:ggml-org/embeddinggemma-300M-GGUF/")


def test_resolve_passthrough_repo_and_local_dir(tmp_path):
    repo = "nomic-ai/nomic-embed-text-v1.5"          # not an alias -> passthrough
    assert emb.resolve_embeddings_model(repo) == repo
    d = tmp_path / "my-embed"
    d.mkdir()
    assert emb.resolve_embeddings_model(str(d)) == str(d)


# effective_model (request `model` policy)
def test_effective_model_accepts_conventional_names():
    for n in ("", "text-embedding-3-small", "text-embedding-3-large",
              "text-embedding-ada-002", "default", EGEMMA):
        assert emb.effective_model(n, EGEMMA) == EGEMMA
    assert emb.effective_model("embeddinggemma", EGEMMA) == EGEMMA  # alias -> configured


def test_effective_model_rejects_other_repos():
    with pytest.raises(emb.EmbeddingsRequestError) as exc:
        emb.effective_model("some/other-repo", EGEMMA)
    assert exc.value.status_code == 400


# _normalize_input
def test_normalize_input_string_and_list():
    assert emb._normalize_input("hello") == ["hello"]
    assert emb._normalize_input(["a", "b"]) == ["a", "b"]


def test_normalize_input_rejects_bad_shapes():
    for bad, match in [
        (None, "'input' is required"),
        ("   ", "must not be empty"),
        ([], "empty list"),
        ([1, 2], "list of strings"),
        (42, "must be a string or a list"),
    ]:
        with pytest.raises(emb.EmbeddingsRequestError, match=match):
            emb._normalize_input(bad)


# encode_embedding
def test_encode_embedding_float_is_plain_list():
    out = emb.encode_embedding(np.array([0.0, 0.5, -1.0], dtype=np.float32),
                               "float")
    assert out == [0.0, 0.5, -1.0]
    assert all(isinstance(x, float) for x in out)


def test_encode_embedding_base64_round_trips():
    vec = np.array([0.0, 1.0, -2.5], dtype=np.float32)
    out = emb.encode_embedding(vec, "base64")
    assert isinstance(out, str)
    back = np.frombuffer(base64.b64decode(out), dtype="<f4")
    assert np.array_equal(back, vec)


# run_embeddings (driver; mlx-embeddings stubbed)
def _stub_embed(monkeypatch, matrix, n_tokens=7):
    monkeypatch.setattr(emb, "import_mlx_embeddings", lambda: None)
    monkeypatch.setattr(emb._EmbeddingsModelHolder, "get",
                        classmethod(lambda cls, p: ("MODEL", "TOK")))
    cap = {}

    def fake_embed(model_obj, tokenizer, texts):
        cap.update(model=model_obj, tok=tokenizer, texts=list(texts))
        return np.asarray(matrix, dtype=np.float32), n_tokens

    monkeypatch.setattr(emb, "_embed_texts", fake_embed)
    return cap


def test_run_embeddings_happy_path_float(monkeypatch):
    # exactly-representable float32 values so the JSON floats compare cleanly
    cap = _stub_embed(monkeypatch, [[0.5, -0.25], [0.125, 0.75]], n_tokens=5)
    out = emb.run_embeddings(["foo", "bar"], configured_model=EGEMMA,
                             model="text-embedding-3-small")
    assert out["object"] == "list"
    assert out["model"] == "text-embedding-3-small"      # echo, never the path
    assert out["usage"] == {"prompt_tokens": 5, "total_tokens": 5}
    assert [d["index"] for d in out["data"]] == [0, 1]
    assert out["data"][0]["embedding"] == [0.5, -0.25]
    assert out["data"][1]["object"] == "embedding"
    assert cap["texts"] == ["foo", "bar"]


def test_run_embeddings_single_string_and_base64(monkeypatch):
    _stub_embed(monkeypatch, [[1.0, 0.0]], n_tokens=3)
    out = emb.run_embeddings("just one", configured_model=EGEMMA,
                             encoding_format="base64")
    assert len(out["data"]) == 1
    back = np.frombuffer(base64.b64decode(out["data"][0]["embedding"]),
                         dtype="<f4")
    assert np.array_equal(back, np.array([1.0, 0.0], dtype=np.float32))


def test_run_embeddings_field_validation(monkeypatch):
    _stub_embed(monkeypatch, [[0.0]])
    with pytest.raises(emb.EmbeddingsRequestError, match="'input' is required"):
        emb.run_embeddings(None, configured_model=EGEMMA)
    with pytest.raises(emb.EmbeddingsRequestError, match="encoding_format"):
        emb.run_embeddings("hi", configured_model=EGEMMA,
                           encoding_format="float16")


def test_run_embeddings_wraps_backend_errors(monkeypatch):
    monkeypatch.setattr(emb, "import_mlx_embeddings", lambda: None)
    monkeypatch.setattr(emb._EmbeddingsModelHolder, "get",
                        classmethod(lambda cls, p: ("M", "T")))

    def boom(*a, **k):
        raise RuntimeError("bert exploded")

    monkeypatch.setattr(emb, "_embed_texts", boom)
    with pytest.raises(RuntimeError, match="embedding failed: bert exploded"):
        emb.run_embeddings("hi", configured_model=EGEMMA)


def test_run_embeddings_missing_file_propagates(monkeypatch):
    # FileNotFoundError must NOT be masked into RuntimeError: the endpoint's
    # 404 model_file_missing branch keys on it (a missing service GGUF returns
    # a clean 404, not a raw-errno 500).
    monkeypatch.setattr(emb, "import_mlx_embeddings", lambda: None)

    def gone(cls, p):
        raise FileNotFoundError(2, "No such file or directory", p)

    monkeypatch.setattr(emb._EmbeddingsModelHolder, "get", classmethod(gone))
    with pytest.raises(FileNotFoundError):
        emb.run_embeddings("hi", configured_model=EGEMMA)


# prewarm (background model load; never touches a real GPU/HF here)
def test_prewarm_loads_in_background(monkeypatch):
    monkeypatch.setattr(emb, "import_mlx_embeddings", lambda: None)
    loaded = []
    monkeypatch.setattr(emb, "_load_embeddings_model", lambda p: loaded.append(p))
    fut = emb.prewarm(EGEMMA)
    fut.result(timeout=5)
    assert fut.done()
    assert loaded == [EGEMMA]


def test_prewarm_is_best_effort_on_load_failure(monkeypatch, capsys):
    monkeypatch.setattr(emb, "import_mlx_embeddings", lambda: None)

    def boom(_p):
        raise RuntimeError("HF offline")

    monkeypatch.setattr(emb, "_load_embeddings_model", boom)
    emb.prewarm(EGEMMA).result(timeout=5)        # must not raise
    assert "embeddings prewarm failed" in capsys.readouterr().err


def test_prewarm_is_best_effort_when_extra_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)
    emb.prewarm(EGEMMA).result(timeout=5)
    assert "embeddings prewarm failed" in capsys.readouterr().err


def test_load_embeddings_model_warms_holder(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        emb._EmbeddingsModelHolder, "get",
        classmethod(lambda cls, p: seen.update(path=p)))
    emb._load_embeddings_model(EGEMMA)
    assert seen["path"] == EGEMMA


# import gate + config key
def test_import_gate_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_embeddings", None)
    # mlx-embeddings is a core dependency now - a missing one is a broken install.
    with pytest.raises(ImportError, match=r"core gmlx dependency"):
        emb.import_mlx_embeddings()


def test_config_embeddings_key_round_trips():
    cfg = build_config({"server": {"embeddings": "embeddinggemma"}})
    assert cfg.embeddings == "embeddinggemma"   # raw; resolved at serve time
    assert build_config({"server": {}}).embeddings is None
    assert build_config({"server": {"embeddings": False}}).embeddings is None
    assert build_config({"server": {"embeddings": True}}).embeddings is True


# ---------------------------------------------------------------------------
# GGUF decoder-LM backend (Qwen3-Embedding et al.). The loader is stubbed, no
# model is loaded; only the tiny pooling math touches mlx (forced onto the CPU).
# ---------------------------------------------------------------------------
GGUF_PATH = "/models/Qwen3-Embedding-4B.Q6_K.gguf"


@pytest.fixture
def cpu_mx():
    mx = pytest.importorskip("mlx.core")
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(prev)


class _FakeTok:
    eos_token_id = 99

    def __init__(self, ids):
        self._ids = ids

    def encode(self, text):
        return list(self._ids)


class _FakeTrunk:
    """A trunk whose last-token hidden row is a known vector (so pooling + L2-norm
    are checkable), recording the ids it was called with (so EOS-append /
    truncation are checkable)."""

    def __init__(self, mx, last_vec):
        self._mx = mx
        self._last = last_vec
        self.seen = []

    def __call__(self, ids_arr):
        self.seen.append(np.asarray(ids_arr).tolist()[0])
        T = ids_arr.shape[1]
        arr = np.zeros((1, T, len(self._last)), dtype=np.float32)
        arr[0, -1, :] = self._last
        return self._mx.array(arr)


class _FakeModel:
    def __init__(self, trunk):
        self.model = trunk


class _FakeEncoderOut:
    def __init__(self, text_embeds):
        self.text_embeds = text_embeds


class _FakeEncoder:
    """Stands in for an mlx-embeddings Model: records the ids + mask each forward
    sees and returns a fixed pooled ``text_embeds`` (the real model mean-pools +
    dense + L2-norms; the driver just forwards). ``__module__`` is spoofed so
    ``_is_gguf_encoder`` routes to it, mirroring the real class' module."""

    __module__ = "mlx_embeddings.models.gemma3_text"

    def __init__(self, mx, vec):
        self._mx = mx
        self._vec = vec
        self.seen_ids = []
        self.seen_mask = []

    def __call__(self, input_ids, attention_mask=None):
        self.seen_ids.append(np.asarray(input_ids).tolist()[0])
        self.seen_mask.append(np.asarray(attention_mask).tolist()[0])
        return _FakeEncoderOut(
            self._mx.array(np.asarray([self._vec], dtype=np.float32)))


# _is_gguf_ref
def test_is_gguf_ref_truth_table():
    assert emb._is_gguf_ref("/x/Qwen3-Embedding.gguf")
    assert emb._is_gguf_ref("hf:org/repo/file.gguf")
    assert emb._is_gguf_ref("hf:org/repo/file.gguf@main")
    assert emb._is_gguf_ref("MODEL.GGUF")             # case-insensitive
    assert not emb._is_gguf_ref("qwen3-embed")
    assert not emb._is_gguf_ref(EGEMMA)
    assert not emb._is_gguf_ref("/some/dir")
    assert not emb._is_gguf_ref(True)


# resolve_embeddings_model (GGUF refs)
def test_resolve_gguf_local_path(tmp_path):
    f = tmp_path / "embed.gguf"
    f.write_bytes(b"GGUF")
    assert emb.resolve_embeddings_model(str(f)) == str(f)


def test_resolve_gguf_relative_path_searches_model_dirs(tmp_path):
    # A model_dirs-relative path (how the init wizard writes every entry) must
    # resolve like a models: entry - not demand an absolute path.
    sub = tmp_path / "org__repo-GGUF"
    sub.mkdir()
    (sub / "embed.gguf").write_bytes(b"GGUF")
    got = emb.resolve_embeddings_model("org__repo-GGUF/embed.gguf",
                                       [str(tmp_path)])
    assert got == str(sub / "embed.gguf")
    from gmlx.config import ConfigError
    with pytest.raises(ConfigError, match=str(tmp_path)):   # names the dirs searched
        emb.resolve_embeddings_model("org__repo-GGUF/nope.gguf", [str(tmp_path)])


def test_resolve_gguf_uncached_hf_ref_raises():
    from gmlx.config import ConfigError
    with pytest.raises(ConfigError):
        emb.resolve_embeddings_model("hf:no-such-org/no-such-repo/x.gguf")


# _eos_id
def test_eos_id_shapes():
    class T1:
        eos_token_id = 7

    class T2:
        eos_token_id = [7, 8]

    class T3:
        eos_token_id = None

    class T4:
        eos_token_id = True            # bool is an int subclass - must not count

    assert emb._eos_id(T1) == 7
    assert emb._eos_id(T2) == 7
    assert emb._eos_id(T3) is None
    assert emb._eos_id(T4) is None


# _embed_texts_gguf
def test_embed_texts_gguf_pools_last_token_and_normalizes(cpu_mx):
    trunk = _FakeTrunk(cpu_mx, [3.0, 4.0])             # ||[3,4]|| = 5 -> [0.6, 0.8]
    matrix, n = emb._embed_texts_gguf(
        _FakeModel(trunk), _FakeTok([1, 2, 3]), ["hello"])
    assert matrix.shape == (1, 2)
    np.testing.assert_allclose(matrix[0], [0.6, 0.8], atol=1e-5)
    assert trunk.seen == [[1, 2, 3, 99]]              # EOS appended
    assert n == 4


def test_embed_texts_gguf_truncates_to_cap(cpu_mx, monkeypatch):
    monkeypatch.setattr(emb, "_GGUF_MAX_TOKENS", 2)
    trunk = _FakeTrunk(cpu_mx, [1.0, 0.0])
    emb._embed_texts_gguf(_FakeModel(trunk), _FakeTok([1, 2, 3, 4, 5]), ["x"])
    assert trunk.seen == [[1, 2, 99]]                 # truncated to 2, then EOS


def test_embed_texts_gguf_no_double_eos(cpu_mx):
    trunk = _FakeTrunk(cpu_mx, [0.0, 1.0])
    emb._embed_texts_gguf(_FakeModel(trunk), _FakeTok([1, 2, 99]), ["x"])
    assert trunk.seen == [[1, 2, 99]]                 # already ends in EOS


def test_embed_texts_gguf_batches_each_text(cpu_mx):
    trunk = _FakeTrunk(cpu_mx, [1.0, 0.0])
    matrix, n = emb._embed_texts_gguf(
        _FakeModel(trunk), _FakeTok([5, 6]), ["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert len(trunk.seen) == 3                       # one forward per text
    assert n == 3 * 3                                 # (5,6,99) each


# _is_gguf_encoder (encoder vs decoder discrimination by loaded class' module)
def test_is_gguf_encoder_distinguishes_backends(cpu_mx):
    assert emb._is_gguf_encoder(_FakeEncoder(cpu_mx, [1.0])) is True
    assert emb._is_gguf_encoder(_FakeModel(None)) is False    # mlx-lm-style decoder


# _embed_texts_gguf_encoder (model pools internally; driver just forwards)
def test_embed_texts_gguf_encoder_forwards_and_returns_pooled(cpu_mx):
    enc = _FakeEncoder(cpu_mx, [0.6, 0.8])            # the model already L2-norms
    matrix, n = emb._embed_texts_gguf_encoder(
        enc, _FakeTok([1, 2, 3]), ["hello"])
    assert matrix.shape == (1, 2)
    np.testing.assert_allclose(matrix[0], [0.6, 0.8], atol=1e-6)
    assert enc.seen_ids == [[1, 2, 3]]               # no EOS appended (encoder)
    assert enc.seen_mask == [[1, 1, 1]]              # all-ones mask, no padding
    assert n == 3


def test_embed_texts_gguf_encoder_caps_at_ceiling(cpu_mx, monkeypatch):
    monkeypatch.setattr(emb, "_ENCODER_MAX_TOKENS", 2)
    enc = _FakeEncoder(cpu_mx, [1.0, 0.0])
    emb._embed_texts_gguf_encoder(enc, _FakeTok([1, 2, 3, 4, 5]), ["x"])
    assert enc.seen_ids == [[1, 2]]                  # truncated to ceiling, no EOS
    assert enc.seen_mask == [[1, 1]]


def test_embed_texts_gguf_encoder_respects_tokenizer_window(cpu_mx):
    class _CapTok(_FakeTok):
        model_max_length = 2                         # shorter than the ceiling

    enc = _FakeEncoder(cpu_mx, [1.0, 0.0])
    emb._embed_texts_gguf_encoder(enc, _CapTok([1, 2, 3, 4]), ["x"])
    assert enc.seen_ids == [[1, 2]]


def test_embed_texts_gguf_encoder_batches_each_text(cpu_mx):
    enc = _FakeEncoder(cpu_mx, [1.0, 0.0])
    matrix, n = emb._embed_texts_gguf_encoder(
        enc, _FakeTok([5, 6]), ["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert enc.seen_ids == [[5, 6], [5, 6], [5, 6]]  # one forward per text
    assert n == 3 * 2


# run_embeddings (GGUF dispatch; loader stubbed, mlx-embeddings forbidden)
def test_run_embeddings_gguf_backend(monkeypatch):
    def _boom():
        raise AssertionError("mlx-embeddings must not be imported for a GGUF")

    monkeypatch.setattr(emb, "import_mlx_embeddings", _boom)
    monkeypatch.setattr(emb._GGUFEmbeddingsHolder, "get",
                        classmethod(lambda cls, p: ("MODEL", "TOK")))
    cap = {}

    def fake_embed(model_obj, tok, texts):
        cap.update(model=model_obj, tok=tok, texts=list(texts))
        return np.asarray([[0.5, -0.5]], dtype=np.float32), 4

    monkeypatch.setattr(emb, "_embed_texts_gguf", fake_embed)
    out = emb.run_embeddings("hello", configured_model=GGUF_PATH,
                             model="text-embedding-3-small")
    assert out["model"] == "text-embedding-3-small"      # echo, never the path
    assert out["data"][0]["embedding"] == [0.5, -0.5]
    assert out["usage"] == {"prompt_tokens": 4, "total_tokens": 4}
    assert cap["texts"] == ["hello"] and cap["model"] == "MODEL"


def test_run_embeddings_gguf_encoder_dispatch(monkeypatch, cpu_mx):
    # an encoder-class GGUF must route to _embed_texts_gguf_encoder, NOT the
    # decoder driver, and must not import mlx-embeddings (it loads via the loader).
    def _boom():
        raise AssertionError("mlx-embeddings must not be imported for a GGUF")

    monkeypatch.setattr(emb, "import_mlx_embeddings", _boom)
    enc = _FakeEncoder(cpu_mx, [0.5, -0.25])
    monkeypatch.setattr(emb._GGUFEmbeddingsHolder, "get",
                        classmethod(lambda cls, p: (enc, _FakeTok([1, 2, 3]))))

    def _decoder_boom(*a, **k):
        raise AssertionError("decoder driver called for an encoder-class model")

    monkeypatch.setattr(emb, "_embed_texts_gguf", _decoder_boom)
    out = emb.run_embeddings("hello", configured_model=GGUF_PATH,
                             model="text-embedding-3-small")
    assert out["model"] == "text-embedding-3-small"      # echo, never the path
    assert out["data"][0]["embedding"] == [0.5, -0.25]
    assert enc.seen_ids == [[1, 2, 3]]


# prewarm / _load_embeddings_model (GGUF dispatch)
def test_load_embeddings_model_gguf_uses_gguf_holder(monkeypatch):
    seen = {}
    monkeypatch.setattr(emb._GGUFEmbeddingsHolder, "get",
                        classmethod(lambda cls, p: seen.update(path=p)))
    monkeypatch.setattr(emb._EmbeddingsModelHolder, "get",
                        classmethod(lambda cls, p: seen.update(wrong=True)))
    emb._load_embeddings_model(GGUF_PATH)
    assert seen == {"path": GGUF_PATH}


def test_prewarm_gguf_skips_mlx_embeddings(monkeypatch):
    def _boom():
        raise AssertionError("mlx-embeddings must not be imported for a GGUF")

    monkeypatch.setattr(emb, "import_mlx_embeddings", _boom)
    loaded = []
    monkeypatch.setattr(emb, "_load_embeddings_model", lambda p: loaded.append(p))
    emb.prewarm(GGUF_PATH).result(timeout=5)
    assert loaded == [GGUF_PATH]
