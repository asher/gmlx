"""`gmlx systemone` on the server path: flag checks, the posted body, the
key header, the printed rows and the error messages. No server runs."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

import gmlx.commands.systemone as so

_BODY = {"model": "jev-latest", "state": "s", "seed": 3,
         "questions": {"urgent": {"type": "noul"}}}
_ANSWER = {
    "model": "dg",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.91},
        "team": {"type": "choice", "choice": "infra",
                 "probabilities": {"infra": 0.8, "billing": 0.2},
                 "confidence": 0.8},
        "severity": {"type": "score", "score": 1.5, "legend": {},
                     "probabilities": {}, "confidence": 0.6},
        "why": None,
    },
    "usage": {"input_tokens": 10, "output_tokens": 4},
    "diagnostics": {},
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def request_file(tmp_path):
    p = tmp_path / "req.json"
    p.write_text(json.dumps(_BODY))
    return str(p)


@pytest.fixture
def posted(monkeypatch):
    seen = {}

    def fake_urlopen(req, *a, **k):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["headers"] = dict(req.header_items())
        return _Resp(json.dumps(_ANSWER).encode())

    monkeypatch.setattr(so.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GMLX_API_KEY", raising=False)
    return seen


def test_rows_are_printed_per_question(request_file, posted, capsys):
    assert so.cmd_systemone([request_file, "--url", "http://h:1/v1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["urgent:   0.910", "team:     infra (0.80)",
                   "severity: 1.50 (0.60)", "why:      skipped"]
    assert posted["url"] == "http://h:1/v1/systemone"
    assert posted["body"] == _BODY


def test_seed_replaces_the_body_seed_and_the_key_is_sent(
        request_file, posted, monkeypatch, capsys):
    monkeypatch.setenv("GMLX_API_KEY", "k1")
    assert so.cmd_systemone([request_file, "--url", "http://h:1", "--seed", "9",
                             "--json"]) == 0
    assert posted["body"]["seed"] == 9
    assert posted["headers"]["Authorization"] == "Bearer k1"
    assert json.loads(capsys.readouterr().out) == _ANSWER


@pytest.mark.parametrize("extra", [["--url", "http://h:1"], ["--host", "h"],
                                   ["--port", "1"]])
def test_model_excludes_a_server_target(request_file, extra, capsys):
    with pytest.raises(SystemExit) as e:
        so.cmd_systemone([request_file, "--model", "m.gguf", *extra])
    assert e.value.code == 2
    assert "--model" in capsys.readouterr().err


def test_host_and_port_together_are_allowed(request_file, posted, monkeypatch):
    import gmlx.serve.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "auto_target", lambda h, p: (h, p))
    assert so.cmd_systemone([request_file, "--host", "h", "--port", "7"]) == 0
    assert posted["url"] == "http://h:7/v1/systemone"


def test_config_needs_model(request_file, capsys):
    with pytest.raises(SystemExit):
        so.cmd_systemone([request_file, "--config", "c.yaml"])


def test_an_unreadable_request_file_fails(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    assert so.cmd_systemone([str(bad), "--url", "http://h:1"]) == 1
    assert so.cmd_systemone([str(tmp_path / "none.json"), "--url", "http://h:1"]) == 1


def _http_error(code, body):
    return urllib.error.HTTPError("http://h:1/v1/systemone", code, "x", {},
                                  io.BytesIO(json.dumps(body).encode()))


@pytest.mark.parametrize("code,body,needle", [
    (401, {}, "API key"),
    (422, {"error": {"type": "validation_error", "message": "state: required"}},
     "state: required"),
])
def test_http_errors_are_reported(request_file, monkeypatch, capsys, code, body, needle):
    def fake_urlopen(req, *a, **k):
        raise _http_error(code, body)

    monkeypatch.setattr(so.urllib.request, "urlopen", fake_urlopen)
    assert so.cmd_systemone([request_file, "--url", "http://h:1"]) == 1
    assert needle in capsys.readouterr().err


def test_an_unreachable_server_is_reported(request_file, monkeypatch, capsys):
    def fake_urlopen(req, *a, **k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(so.urllib.request, "urlopen", fake_urlopen)
    assert so.cmd_systemone([request_file, "--url", "http://h:1"]) == 1
    assert "gmlx serve" in capsys.readouterr().err
