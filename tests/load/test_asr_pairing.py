"""The speech companion of a model that reads audio through one (Granite
Switch): how a companion GGUF is classified, how a model finds its
companion, and the prompt splice."""

from __future__ import annotations

import re
import types
from pathlib import Path

import numpy as np
import pytest
from gguf import GGUFWriter

from gmlx import asr
from gmlx.config import ConfigError, build_config, resolve_model
from gmlx.load import arch_table
from gmlx.load.discovery import classify_gguf, find_asr_companion

_TEMPLATE = (Path(__file__).parents[1] / "fixtures"
             / "granite_switch_template.jinja").read_text()
_REPO = arch_table.ASR_COMPANION_REPOS["graniteswitch"]


def _gguf(path, arch):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(path), arch)
    w.add_tensor("x", np.zeros(4, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _switch(folder):
    return _gguf(folder / "granite-switch-4.1-8b-Q4_K_M.gguf", "graniteswitch")


def _speech(folder):
    return _gguf(folder / "granite-speech-5.0-470m-turboctc.gguf",
                 "granite-speech5-ctc")


def test_a_companion_classifies_as_asr_and_never_runs(tmp_path):
    path = _speech(tmp_path)
    assert classify_gguf(path).kind == "asr"
    with pytest.raises(arch_table.UnsupportedArchError, match="--asr PATH"):
        arch_table.gate("granite-speech5-ctc")


def test_a_companion_in_the_model_folder_pairs(tmp_path):
    model = _switch(tmp_path)
    speech = _speech(tmp_path)
    assert find_asr_companion(model) == speech
    assert asr.companion_path(model) == speech


def test_a_companion_in_a_sibling_repo_folder_pairs(tmp_path):
    model = _switch(tmp_path / "ibm-granite__granite-switch-4.1-8b-GGUF")
    speech = _speech(tmp_path / f"someone__{_REPO}")
    assert find_asr_companion(model) == speech


def test_a_companion_in_the_hf_cache_pairs(tmp_path):
    model = _switch(tmp_path / "models--ibm-granite--granite-switch-4.1-8b-GGUF"
                    / "snapshots" / "abc")
    speech = _speech(tmp_path / f"models--someone--{_REPO}" / "snapshots" / "def")
    assert find_asr_companion(model) == speech


def test_only_a_model_that_pairs_gets_a_companion(tmp_path):
    other = _gguf(tmp_path / "llama.gguf", "llama")
    _speech(tmp_path)
    assert not asr.pairs_with_companion(other)
    assert asr.companion_path(other) is None
    model = _switch(tmp_path / "solo")
    assert asr.companion_path(model) is None
    assert asr.companion_path(model, explicit=str(tmp_path / "x.gguf")) \
        == str(tmp_path / "x.gguf")
    assert asr.companion_path(model, explicit="x.gguf", disabled=True) is None


def test_the_asr_config_key(tmp_path):
    model = _switch(tmp_path)
    speech = _speech(tmp_path / "elsewhere")
    cfg = build_config({"models": {"a": {"path": model, "asr": speech},
                                   "b": {"path": model, "asr": False},
                                   "c": {"path": model}}})
    a, b, c = (resolve_model(m, cfg) for m in "abc")
    assert (a.asr, a.no_asr) == (speech, False)
    assert (b.asr, b.no_asr) == (None, True)
    assert (c.asr, c.no_asr) == (None, False)
    with pytest.raises(ConfigError, match="asr"):
        build_config({"models": {"a": {"path": model, "asr": 3}}})


def _render(messages):
    from transformers.utils.chat_template_utils import render_jinja_template

    out = render_jinja_template([messages], chat_template=_TEMPLATE,
                                add_generation_prompt=True)
    rendered = out[0] if isinstance(out, tuple) else out
    return rendered[0] if isinstance(rendered, list) else rendered


@pytest.mark.parametrize("content", [
    [{"type": "text", "text": "Hi."}, {"type": "input_audio"},
     {"type": "text", "text": "What?"}],
    [{"type": "audio"}, {"type": "audio"}, {"type": "text", "text": "Both?"}],
    [{"type": "text", "text": "Only"}, {"type": "audio"}],
])
def test_marked_text_renders_as_the_content_list(content):
    as_list = _render([{"role": "user", "content": content}])
    as_text = _render([{"role": "user", "content": asr.marked_text(content)}])
    assert as_text == as_list
    assert as_list.count(asr.MARKER) == sum(map(asr.is_audio_part, content))


def test_splice_replaces_each_marker_in_order():
    ids = [1, 99, 2, 99, 3]
    out = asr.splice(ids, 99, ["ab", ""], lambda t: [ord(c) for c in t])
    assert out == [1, 97, 98, 2, ord(" "), 3]


def test_splice_refuses_a_count_mismatch_and_special_tokens():
    with pytest.raises(ValueError, match="2 audio marker"):
        asr.splice([99, 99], 99, ["one"], list)
    with pytest.raises(ValueError, match=re.escape("<|end_of_text|>")):
        asr.splice([99], 99, ["stop <|end_of_text|>"], list)


def test_run_without_a_companion_names_the_flag(tmp_path, monkeypatch):
    from gmlx.commands import cli

    model = _switch(tmp_path)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    args = types.SimpleNamespace(gguf=model, audio=[str(clip)], image=None,
                                 video=None, asr=None, no_asr=False)
    monkeypatch.setattr(cli, "_media_inputs", lambda a: ([], [str(clip)], []))
    with pytest.raises(ValueError, match="--asr PATH"):
        cli._audio_companion(args)
    args.no_asr = True
    with pytest.raises(ValueError, match="--asr PATH"):
        cli._audio_companion(args)
    speech = _speech(tmp_path)
    args.no_asr = False
    assert cli._audio_companion(args) == speech


def test_the_marker_id_reads_through_either_tokenizer_layer():
    """Serve hands over an HF fast tokenizer, whose _tokenizer is the raw
    tokenizers object; run hands over a wrapper around the HF one."""
    hf = types.SimpleNamespace(convert_tokens_to_ids={asr.MARKER: 77}.get,
                               unk_token_id=0, _tokenizer=object())
    assert asr.marker_id(hf) == 77
    assert asr.marker_id(types.SimpleNamespace(_tokenizer=hf)) == 77
    assert asr.marker_id(types.SimpleNamespace(
        convert_tokens_to_ids=lambda t: 0, unk_token_id=0)) is None
