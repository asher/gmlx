#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Every path that generates refuses a decision model.

A d1 GGUF is arch ``lfm3`` with ``lfm3.attention.causal`` false. A
causal pass would answer wrongly with no error, so each entry point that
generates refuses the file with a message that names the systemone
route, and the causal forward itself refuses as a backstop. The fixture
carries the header facts of a d1 file and no weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.lfm3_header import write_lfm3_header, write_projector_header

_ROUTE = "/v1/systemone"


@pytest.fixture()
def d1_gguf(tmp_path) -> str:
    return write_lfm3_header(tmp_path / "d1.gguf", templates=("systemone",),
                             causal=False, decision="lfm3-d1")


@pytest.fixture()
def chat_gguf(tmp_path) -> str:
    """A chat model that carries a decision type, as Granite Switch
    does: the record keys on the attention, not on that key."""
    return write_lfm3_header(tmp_path / "chat.gguf", causal=True,
                             decision="granite-switch")


def test_the_record_marks_a_d1_file_and_no_chat_file(d1_gguf, chat_gguf):
    from gmlx.load.decision import decision_header
    from gmlx.load.discovery import classify_gguf, header_meta
    from gmlx.load.headerscan import scan_gguf

    assert decision_header(scan_gguf(d1_gguf).kv)
    assert header_meta(d1_gguf)["decision"] is True
    assert classify_gguf(d1_gguf).decision is True
    assert not decision_header(scan_gguf(chat_gguf).kv)
    assert header_meta(chat_gguf)["decision"] is False


@pytest.mark.parametrize("argv", [
    ["run", "--prompt", "hi", "--max-tokens", "2"],
    ["run", "--bench", "16"],
    ["chat"],
])
def test_run_bench_and_chat_refuse_a_d1_file(d1_gguf, capsys, argv):
    from gmlx.commands.cli import umbrella_main

    try:
        code = umbrella_main([argv[0], d1_gguf, *argv[1:]])
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    assert code not in (0, None)
    assert _ROUTE in out.err + out.out


def test_the_loaders_refuse_unless_the_caller_runs_the_decision_forward(
        d1_gguf):
    from gmlx.load.decision import DecisionModelOnly
    from gmlx.load.loader import load_model
    from gmlx.load.preflight import preflight
    from gmlx.serve.bridge_vlm import load_serveable_model

    assert preflight(d1_gguf).decision
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        load_model(d1_gguf)
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        load_serveable_model(d1_gguf)
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        preflight(d1_gguf).refuse_decision()
    preflight(d1_gguf).refuse_decision(allowed=True)


def test_the_mtp_loaders_refuse_a_d1_file(d1_gguf):
    from gmlx.load.decision import DecisionModelOnly
    from gmlx.spec.mtp_load import load_mtp_model

    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        load_mtp_model(d1_gguf)


def test_the_server_refuses_a_d1_file_outside_the_decision_scope(d1_gguf,
                                                                 chat_gguf):
    from fastapi import HTTPException

    from gmlx.load.decision import decision_scope
    from gmlx.serve.residency import _refuse_decision_model

    spec = SimpleNamespace(id="d1-2b")
    with pytest.raises(HTTPException) as exc:
        _refuse_decision_model(d1_gguf, spec)
    assert exc.value.status_code == 400
    error = exc.value.detail["error"]
    assert error["type"] == "decision_model_only"
    assert _ROUTE in error["message"] and "d1-2b" in error["message"]
    with decision_scope():
        _refuse_decision_model(d1_gguf, spec)
    _refuse_decision_model(chat_gguf, spec)


def test_the_chat_warm_stays_refused_and_keep_and_preload_may_load(
        monkeypatch, d1_gguf):
    """``_load_resident`` opens the scope for keep and preload
    (``images=True``) only, never for the chat routes' warm."""
    import importlib

    from gmlx.load.decision import in_decision_scope
    from gmlx.serve.patches import routes

    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    seen = []
    monkeypatch.setattr(app, "get_cached_model",
                        lambda *a, **k: seen.append(in_decision_scope()))
    monkeypatch.setattr(routes.serving, "is_image_model", lambda _m: False)
    routes._load_resident("d1-2b")
    routes._load_resident("d1-2b", images=True)
    assert seen == [False, True]


def test_the_systemone_route_opens_the_scope_only_when_asked(monkeypatch):
    import importlib

    from gmlx.load.decision import in_decision_scope
    from gmlx.serve import job_route

    app = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
    seen = []
    monkeypatch.setattr(app, "get_cached_model",
                        lambda *a, **k: seen.append(in_decision_scope()))
    runtime = SimpleNamespace(response_generator=SimpleNamespace(
        _rg=object(), _hold=None))
    job_route.resolve_engine(lambda: "m", None, runtime)
    job_route.resolve_engine(lambda: "m", None, runtime, decisions=True)
    assert seen == [False, True]


def test_speech_services_skip_a_d1_model(tmp_path, d1_gguf):
    from gmlx.serve.native_tts import tts_adapter
    from gmlx.serve.stt import stt_adapter

    omni = write_projector_header(tmp_path / "mmproj-d1-omni.gguf",
                                  vision=True, audio=True)
    assert stt_adapter(d1_gguf, omni) is None
    assert tts_adapter(d1_gguf, omni) is None


def test_the_embeddings_route_names_the_systemone_route(d1_gguf):
    from gmlx.serve.embeddings import _gguf_refusal

    reason = _gguf_refusal(d1_gguf)
    assert reason and _ROUTE in reason and "causal LM" not in reason


def test_models_entry_marks_d1_and_chat_clients_skip_it(monkeypatch, d1_gguf,
                                                        chat_gguf):
    from gmlx.commands.launch import chat_models
    from gmlx.serve.patches.routes import _decision
    from gmlx.talk import client

    assert _decision(SimpleNamespace(path=d1_gguf))
    assert not _decision(SimpleNamespace(path=chat_gguf))
    entries = [{"id": "d1-2b", "decision": True, "default": True},
               {"id": "chat", "owned_by": "gmlx"}]
    assert [m["id"] for m in chat_models(entries)] == ["chat"]
    monkeypatch.setattr(client, "_http_get_json",
                        lambda *a, **k: {"data": entries})
    caps = client.probe_capabilities("http://h/v1")
    assert caps["chat_ids"] == ["chat"]
    assert caps["default"] is None


def test_the_context_of_a_d1_file_is_the_state_cap(d1_gguf):
    from gmlx.load.decision import DECISION_STATE_CAP
    from gmlx.serve.capacity import trained_context_length

    assert DECISION_STATE_CAP == 65536
    assert trained_context_length(d1_gguf) == DECISION_STATE_CAP


def test_no_drafter_pairs_with_a_d1_model(d1_gguf):
    from gmlx.load import discovery

    target = discovery.classify_gguf(d1_gguf)
    entry = SimpleNamespace(id="d1-2b", stream=None, draft_gguf=None)
    drafter = discovery.ClassifiedGguf(
        "/x/drafter.gguf", "drafter", "lfm3", False, None, False)
    assert discovery._drafter_targets(drafter, [(entry, target)]) == []


def _tiny_args(**over):
    from gmlx.models.lfm3.model import ModelArgs

    base = dict(model_type="lfm3", hidden_size=32, num_hidden_layers=3,
                intermediate_size=64, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, vocab_size=64,
                layer_types=["conv", "full_attention", "sliding_attention"],
                sliding_window=4)
    base.update(over)
    return ModelArgs(**base)


def test_the_causal_forward_refuses_a_decision_model():
    """The backstop: a path nobody listed fails with the message, never
    a causal answer."""
    import mlx.core as mx

    from gmlx.load.decision import DecisionModelOnly, is_decision_model
    from gmlx.models.lfm3.model import Model
    from gmlx.models.lfm3.vlm_model import LanguageModel

    ids = mx.array([[1, 2, 3]])
    plain = Model(_tiny_args())
    assert not is_decision_model(plain)
    plain(ids)
    d1 = Model(_tiny_args(decision=True))
    assert is_decision_model(d1)
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        d1(ids)
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        d1.model(ids)
    lm = LanguageModel(_tiny_args(decision=True))
    assert is_decision_model(SimpleNamespace(language_model=lm))
    with pytest.raises(DecisionModelOnly, match=_ROUTE):
        lm(ids)


def test_the_synthesized_config_carries_the_mark(d1_gguf, chat_gguf):
    from gmlx.commands.tool_preflight import synth_config_from_scans
    from gmlx.load.headerscan import scan_gguf
    from gmlx.models.lfm3.model import ModelArgs

    config = synth_config_from_scans([scan_gguf(d1_gguf,
                                                include_tensors=True)])
    assert config["decision"] is True
    assert ModelArgs.from_dict(config).decision is True
    plain = synth_config_from_scans([scan_gguf(chat_gguf,
                                               include_tensors=True)])
    assert not plain.get("decision")
