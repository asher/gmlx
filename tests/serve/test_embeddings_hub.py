"""The safetensors encoders behind ``/v1/embeddings`` on a repo id from the
Hugging Face cache.

Each test caches a tiny random encoder at commit C1 in a scratch HF cache
in tmp_path, with the real huggingface_hub code, from a fake Hub on
127.0.0.1 (fake_hub.py) whose head has moved on to C2. Requests go through
the real route code (gmlx.serve.embeddings.run_embeddings), on the CPU
device. Neither the network nor the real cache is touched.
"""
from __future__ import annotations

import json
import os
import threading

import pytest

import gmlx.serve.embeddings as emb
import gmlx.serve.hf_cache as hf_cache
from fake_hub import BlackHole, FakeHub, before_each_request, cache_state, use_hub

C1 = "1" * 40
C2 = "2" * 40


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """A fresh failure cache and model holder, a CPU device, and quiet
    progress bars."""
    import mlx.core as mx

    monkeypatch.setattr(emb, "_UNLOADABLE", {})
    _new_start(monkeypatch)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _new_start(monkeypatch):
    """What a server restart clears: the loaded model."""
    for attr in ("model", "tokenizer", "model_path"):
        monkeypatch.setattr(emb._EmbeddingsModelHolder, attr, None)


def _modules(pooling_path: str | None = "1_Pooling") -> str:
    mods = [{"idx": 0, "name": "0", "path": "",
             "type": "sentence_transformers.models.Transformer"}]
    if pooling_path is not None:
        mods.append({"idx": 1, "name": "1", "path": pooling_path,
                     "type": "sentence_transformers.models.Pooling"})
    return json.dumps(mods)


def _tiny(kind: str, tmp_path) -> dict:
    """The files of a tiny random checkpoint of ``kind``, with a
    tokenizer, as name -> bytes."""
    import test_embedders as te

    from gmlx.models.embedders import bert, gemma_embedding, modernbert, qwen3

    module, config = {
        "bert": (bert, te._tiny_bert_config()),
        "modernbert": (modernbert, dict(
            model_type="modernbert", vocab_size=te._VOCAB, hidden_size=64,
            intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            local_attention=4)),
        "gemma3_text": (gemma_embedding, te._TINY_GEMMA),
        "qwen3": (qwen3, te._TINY_QWEN3),
    }[kind]
    model = module.Model(module.ModelArgs.from_dict(config))
    path = te._write(tmp_path / f"build-{kind}", config, te._params(model))
    te._tokenizer(path)
    out = {}
    for d, _, files in os.walk(path):
        for f in files:
            full = os.path.join(d, f)
            with open(full, "rb") as fh:
                out[os.path.relpath(full, path)] = fh.read()
    return out


def _cached(monkeypatch, tmp_path, repo: str, c1: dict, cached, c2=None):
    """A fake Hub with ``repo`` at C1 (files ``c1``) and at the head C2,
    and a scratch cache holding the files ``cached`` of C1, refs/main at
    C1."""
    import huggingface_hub

    hub = FakeHub()
    hub.commit(repo, C1, c1, head=False)
    hub.commit(repo, C2, c2 if c2 is not None else c1)
    cache = str(tmp_path / "hf")
    use_hub(monkeypatch, hub.endpoint, cache)
    huggingface_hub.snapshot_download(repo, revision=C1,
                                      allow_patterns=list(cached))
    store = os.path.join(cache, "models--" + repo.replace("/", "--"))
    os.makedirs(os.path.join(store, "refs"), exist_ok=True)
    with open(os.path.join(store, "refs", "main"), "w") as f:
        f.write(C1)
    hub.requests.clear()
    return hub, cache


def _embed(repo: str) -> list:
    out = emb.run_embeddings("alpha beta gamma", configured_model=repo)
    return out["data"][0]["embedding"]


def test_a_cached_encoder_fetches_its_card_at_the_cached_commit(
        monkeypatch, tmp_path, capsys):
    # The cached C1 snapshot lists a Pooling module at 2_Pool without its
    # config, and lacks the model card, and the Hub's head is C2. The
    # server fetches both at C1, so refs/main stays at C1 and no other
    # snapshot appears, and the model loads on both starts.
    repo = "acme/tiny-encoder"
    files = _tiny("bert", tmp_path)
    c1 = {**files, "modules.json": _modules("2_Pool"),
          "2_Pool/config.json": json.dumps({"pooling_mode_cls_token": True}),
          "README.md": "# tiny encoder\n"}
    hub, cache = _cached(monkeypatch, tmp_path, repo, c1,
                         cached=[*files, "modules.json"],
                         c2={**c1, "README.md": "# newer\n"})
    for start in range(2):
        _new_start(monkeypatch)
        assert len(_embed(repo)) == 64, start
        assert emb._EmbeddingsModelHolder.model.args.pooling == "cls"
    assert cache_state(cache, repo) == {
        "main": C1, "snapshots": {C1: sorted([*files, "modules.json",
                                              "2_Pool/config.json",
                                              "README.md"])}}
    assert sorted(hub.requests) == [
        ("GET", f"/{repo}/resolve/{C1}/2_Pool/config.json"),
        ("GET", f"/{repo}/resolve/{C1}/README.md")]
    assert hub.tokens == []
    err = capsys.readouterr().err
    assert "ships no sentence-transformers Pooling config" not in err
    hub.close()


@pytest.mark.parametrize("repo,kind,modules,pooling", [
    ("BAAI/bge-small-en-v1.5", "bert", None, "cls"),
    ("mlx-community/bge-m3-mlx-8bit", "bert", "1_Pooling", "cls"),
    ("mlx-community/nomicai-modernbert-embed-base-8bit", "modernbert",
     "1_Pooling", "mean"),
    ("mlx-community/embeddinggemma-300m-8bit", "gemma3_text", "1_Pooling",
     "mean"),
    ("mlx-community/Qwen3-Embedding-0.6B-8bit", "qwen3", "1_Pooling", "last"),
])
def test_a_cached_encoder_its_names_settle_makes_no_request(
        monkeypatch, tmp_path, repo, kind, modules, pooling):
    # The repo name, the model type, or the modules the snapshot lists
    # settle the pooling, so the load makes no Hub request at all, on the
    # first start or a later one.
    files = _tiny(kind, tmp_path)
    if modules is not None:
        files["modules.json"] = _modules(modules)
    c1 = {**files, "README.md": "# card\n"}
    hub, cache = _cached(monkeypatch, tmp_path, repo, c1, cached=files)
    for _ in range(2):
        _new_start(monkeypatch)
        assert len(_embed(repo)) == 64
        assert emb._EmbeddingsModelHolder.model.args.pooling == pooling
    assert hub.requests == []
    assert cache_state(cache, repo)["main"] == C1
    hub.close()


def test_an_undecided_encoder_on_a_black_holed_hub_loads_after_one_short_try(
        monkeypatch, tmp_path, capsys):
    # The Hub accepts and never answers. The card fetch gives up after its
    # timeout on its one connection, holds no lock while it waits, and the
    # model loads with the default pooling. A second start does not try
    # again.
    repo = "acme/tiny-encoder"
    files = _tiny("bert", tmp_path)
    hub, cache = _cached(monkeypatch, tmp_path, repo,
                         {**files, "README.md": "# card\n"}, cached=files)
    hub.close()
    hole = BlackHole()
    use_hub(monkeypatch, hole.endpoint, cache)
    monkeypatch.setattr(hf_cache, "FETCH_TIMEOUT", 1.0)
    monkeypatch.setattr(hf_cache, "_clock", lambda: 0.0)
    result: list = []
    entered = threading.Event()
    timeouts, during = [], []

    def other():
        with hf_cache.network_fetch_allowed():
            entered.set()

    def fetch(timeout):
        timeouts.append(timeout)
        threading.Thread(target=other, daemon=True).start()
        during.append(entered.wait(30))

    before_each_request(monkeypatch, fetch)

    def request():
        try:
            result.append(len(_embed(repo)))
        except BaseException as exc:            # noqa: BLE001
            result.append(exc)

    t = threading.Thread(target=request, daemon=True)
    t.start()
    try:
        t.join(60)
        assert result == [64]
        assert during == [True], "the card fetch held the offline lock"
        assert timeouts == [1.0]           # one try, given the whole timeout
        assert hole.wait_accepted(1)
        assert hole.accepted == 1
        assert emb._EmbeddingsModelHolder.model.args.pooling == "mean"
        assert "modules.json not fetched" in capsys.readouterr().err
        _new_start(monkeypatch)
        assert len(_embed(repo)) == 64
        assert hole.accepted == 1 and len(timeouts) == 1
        assert cache_state(cache, repo)["main"] == C1
    finally:
        hole.close()


def test_a_cached_qwen3_named_by_its_card_is_left_to_the_load(
        monkeypatch, tmp_path):
    # The startup check cannot tell this qwen3 snapshot from a chat model:
    # only the model card the snapshot lacks names Qwen3-Embedding. The
    # check leaves it to the load, which fetches the card at C1 and loads
    # the embedder.
    repo = "acme/q3-tiny"
    files = _tiny("qwen3", tmp_path)
    card = "---\nbase_model: Qwen/Qwen3-Embedding-0.6B\n---\n"
    hub, cache = _cached(monkeypatch, tmp_path, repo,
                         {**files, "README.md": card}, cached=files)
    assert emb.check_embeddings_model(repo) is None
    assert emb._UNLOADABLE == {}
    assert len(_embed(repo)) == 64
    assert emb._EmbeddingsModelHolder.model.args.pooling == "last"
    assert sorted(hub.requests) == [
        ("GET", f"/{repo}/resolve/{C1}/README.md"),
        ("GET", f"/{repo}/resolve/{C1}/modules.json")]
    assert cache_state(cache, repo)["main"] == C1
    hub.close()
