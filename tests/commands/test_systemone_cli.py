"""`gmlx systemone` on the server path: flag checks, the posted body, the
key header, the printed rows and the error messages. No server runs. The
offline path on a text model runs a scripted letter reader."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import types
import urllib.error
import zlib

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
                 "confidence": 0.6},
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
    assert out == ["urgent:   0.910", "team:     infra (0.60)",
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


# offline on a text model

class _CharTok:
    bos_token = None

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def apply_chat_template(self, msgs, **kwargs):
        return "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant: "


class _LetterReader:
    def __init__(self, model, letter_ids):
        self.forwards = 0
        self.path = "rows"

    def check(self, rows_scope=None):
        return self.path

    def bind(self, rows_scope=None):
        return self

    def prefill(self, ids):
        self.forwards += 1
        yield
        return ids

    def tails(self, prefix, prefix_ids, tails):
        self.forwards += 1
        yield
        return [[zlib.crc32(repr((t, i)).encode()) / 2**32 for i in range(52)]
                for t in tails]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    import gmlx.gen.diffusion as diffusion
    import gmlx.load.loader as loader
    import gmlx.serve.bridge_vlm as bridge
    import gmlx.systemone.ar_reader as ar_reader
    from gmlx.config import SystemoneCfg

    model = types.SimpleNamespace(diffusion=False)
    monkeypatch.setattr(so, "_offline_settings", lambda path: (None, SystemoneCfg()))
    monkeypatch.setattr(loader, "load_model", lambda path, verbose: (model, None, None))
    monkeypatch.setattr(bridge, "_make_text_processor", lambda tok: _CharTok())
    monkeypatch.setattr(diffusion, "is_diffusion_model", lambda m: m.diffusion)
    monkeypatch.setattr(ar_reader, "LetterReader", _LetterReader)
    monkeypatch.setattr(importlib.import_module("mlx_lm.generate"), "wired_limit",
                        lambda m: contextlib.nullcontext())
    gguf = tmp_path / "text.gguf"
    gguf.write_bytes(b"")

    def run(body, *flags):
        req = tmp_path / "letters.json"
        req.write_text(json.dumps(body))
        return so.cmd_systemone([str(req), "--model", str(gguf), *flags])
    return run


_LETTER_BODY = {"state": "s", "questions": {
    "urgent": {"type": "noul", "instructions": "Urgent?"},
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": None, "infra": None}}}}


def test_offline_a_text_model_answers_with_the_letter_readout(offline, capsys):
    assert offline(_LETTER_BODY, "--json") == 0
    out = capsys.readouterr()
    body = json.loads(out.out)
    assert body["model"] == "text.gguf"
    assert body["diagnostics"]["readout"] == "letters"
    assert set(body["answers"]) == {"urgent", "team"}
    assert body["usage"]["output_tokens"] == 0
    assert "forwards in" in out.err


def test_offline_letter_errors_name_the_request(offline, capsys):
    assert offline(_BODY) == 1
    assert "instructions is required" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="text-only"):
        offline(dict(_LETTER_BODY, state={"image": "data:image/png;base64,AA"}))
