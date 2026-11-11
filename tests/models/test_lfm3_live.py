#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LFM3 checks that need a real GGUF (KQUANT_TEST_GGUF_DIR).

The head's mask finds its tokens by name, so a renamed marker would turn
the mask off with no failure. These tests resolve the names against the
shipped vocabulary. They also load an NVFP4 file, whose FFN weights carry
a scale tensor the loader maps and an input scale it skips.
"""

from __future__ import annotations

import logging

import pytest

import mlx.core as mx

from gmlx.models.lfm3.model import MEDIA_PLACEHOLDERS, MODE_MARKERS

_NAMES = MODE_MARKERS + MEDIA_PLACEHOLDERS


def _lfm3_files(gguf_index) -> list[str]:
    paths = gguf_index.get("lfm3")
    if not paths:
        pytest.skip("no 'lfm3' GGUF under KQUANT_TEST_GGUF_DIR")
    return list(paths)


def _nvfp4_file(gguf_index) -> str:
    from gmlx.load.headerscan import scan_gguf

    for path in _lfm3_files(gguf_index):
        scan = scan_gguf(path, include_tensors=True)
        if any(t.name.endswith(".input_scale") for t in scan.tensors):
            return path
    pytest.skip("no NVFP4 'lfm3' GGUF under KQUANT_TEST_GGUF_DIR")


def test_every_masked_name_resolves_on_the_shipped_vocabulary(gguf_index):
    """The six names are tokens of every LFM3 file but LFM3-VL's, which has
    no audio tokens. The loaded model masks exactly the ones it has."""
    from gmlx import load_model

    for path in _lfm3_files(gguf_index):
        model, _config, tok = load_model(path, verbose=False)
        raw = getattr(tok, "_tokenizer", tok)
        vocab = raw.get_vocab()
        # LFM3-VL's vocabulary has no audio tokens.
        missing = {n for n in _NAMES if n not in vocab}
        assert missing <= {"<|audio_out|>", "<|audio|>"}, (path, missing)
        names = [n for n in _NAMES if n in vocab]
        ids = sorted(vocab[n] for n in names)
        assert len(set(ids)) == len(names)
        assert list(model.suppressed_tokens) == ids, path
        assert sorted(raw.convert_ids_to_tokens(ids)) == sorted(names)

        logits = model(mx.array([tok.encode("Hello")]))[0, -1]
        masked = logits[mx.array(ids)].astype(mx.float32)
        assert float(masked.max()) < -2.0e4, path
        assert int(logits.argmax()) not in ids
        del model
        mx.clear_cache()


def test_nvfp4_scales_map_and_input_scales_skip(gguf_index):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.remap import RemapDecision, parse_gguf_name

    path = _nvfp4_file(gguf_index)
    names = [t.name for t in scan_gguf(path, include_tensors=True).tensors]
    scales = [n for n in names if n.endswith(".scale")]
    inputs = [n for n in names if n.endswith(".input_scale")]
    assert scales and len(scales) == len(inputs)
    for name in scales:
        dec = parse_gguf_name("lfm3", name)
        assert dec.kind == RemapDecision.KIND_MAP, name
        assert dec.hf_name.endswith(".out_scale"), (name, dec.hf_name)
    for name in inputs:
        dec = parse_gguf_name("lfm3", name)
        assert dec.kind == RemapDecision.KIND_SKIP, name
        # An expected skip, not the Unsloth metadata rule.
        assert "Unsloth" not in (dec.reason or ""), (name, dec.reason)


def test_nvfp4_file_loads_with_no_warning(gguf_index, capfd, caplog):
    from gmlx import load_model

    path = _nvfp4_file(gguf_index)
    with caplog.at_level(logging.WARNING):
        model, _config, tok = load_model(path, verbose=True)
        logits = model(mx.array([tok.encode("The capital of France is")]))
        mx.eval(logits)
    out = capfd.readouterr()
    text = (out.out + out.err).lower()
    assert "warning" not in text, text[-2000:]
    assert "unsloth" not in text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert bool(mx.isfinite(logits.astype(mx.float32)).all())
    word = tok.decode([int(logits[0, -1].argmax())])
    assert "Paris" in word, word


def test_every_shipped_template_takes_tool_arguments_as_a_mapping(gguf_index):
    """An OpenAI client replays a tool call's arguments as a JSON string.
    Each shipped template raises on that, which is why the serve render
    decodes the string for this model type."""
    from gmlx import load_model
    from gmlx.serve.patches import render

    assert "lfm3" in render._DICT_TOOL_ARGUMENTS
    tools = [{"type": "function", "function": {
        "name": "weather", "description": "Read the weather.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}}}}]

    def turns(arguments):
        return [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "weather", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}]

    for path in _lfm3_files(gguf_index):
        _model, _config, tok = load_model(path, verbose=False)

        def render_turns(arguments, tok=tok):
            return tok.apply_chat_template(
                turns(arguments), tools=tools, add_generation_prompt=True,
                tokenize=False)

        assert "Lisbon" in render_turns({"city": "Lisbon"}), path
        with pytest.raises(Exception, match="mapping"):
            render_turns('{"city": "Lisbon"}')
        del _model
        mx.clear_cache()
