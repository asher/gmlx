"""The /tokenize and /detokenize routes on a prompt string: the ids equal
the ones /v1/completions sends for that prompt. CPU-only, a stub engine
with a real GGUF tokenizer."""
from __future__ import annotations

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

from gmlx.serve.patches.tokenize import install_tokenize_routes  # noqa: E402
from tests.serve.test_completions_route import (  # noqa: E402,F401
    _BOS,
    _client,
    _encode,
    _restore,
)


def _tokenizing_client(**kw):
    client, rg = _client(**kw)
    install_tokenize_routes()
    return client, rg


@pytest.mark.parametrize("path", ["/tokenize", "/v1/tokenize"])
def test_a_prompt_tokenizes_as_completions_sends_it(path):
    client, rg = _tokenizing_client()
    r = client.post(path, json={"model": "m", "prompt": "hi there"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["tokens"] == [_BOS] + _encode("hi there", special=False)
    assert out["count"] == len(out["tokens"])
    assert "token_strs" not in out
    r = client.post("/v1/completions", json={"model": "m", "prompt": "hi there"})
    assert rg.calls[-1][0] == out["tokens"]


def test_add_special_tokens_false_drops_the_bos():
    client, _ = _tokenizing_client()
    r = client.post("/tokenize", json={"model": "m", "prompt": "hi",
                                       "add_special_tokens": False})
    assert r.json()["tokens"] == _encode("hi", special=False)


def test_token_strs_name_each_id():
    client, rg = _tokenizing_client()
    r = client.post("/tokenize", json={"model": "m", "prompt": "hi",
                                       "return_token_strs": True})
    out = r.json()
    assert len(out["token_strs"]) == out["count"]
    assert out["token_strs"][0] == rg.processor.tokenizer.convert_ids_to_tokens([_BOS])[0]


@pytest.mark.parametrize("path", ["/detokenize", "/v1/detokenize"])
def test_detokenize_round_trips(path):
    client, rg = _tokenizing_client()
    ids = client.post("/tokenize", json={"model": "m", "prompt": "hi there"}).json()["tokens"]
    r = client.post(path, json={"model": "m", "tokens": ids})
    assert r.status_code == 200, r.text
    assert r.json() == {"prompt": rg.processor.tokenizer.decode(ids)}
    assert r.json()["prompt"].endswith("hi there")


@pytest.mark.parametrize("tokens,fragment", [
    ("abc", "list of token ids"), ([1, True], "list of token ids"),
    ([1, 2.5], "list of token ids"), ([-1], "out of range"), ([261], "out of range")])
def test_detokenize_refuses_bad_ids(tokens, fragment):
    client, _ = _tokenizing_client()
    r = client.post("/detokenize", json={"model": "m", "tokens": tokens})
    assert r.status_code == 400 and fragment in r.json()["detail"]
