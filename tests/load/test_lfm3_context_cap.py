#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Every reader of an LFM3 file's trained context gives the same value.

A speaking checkpoint (one with a named ``tts`` chat template) is served
at 32,768 tokens. The synthesized config, the serve capacity reader and
the pull catalog all read the trained context, so all three apply the
cap. The cases are the header facts of the four shipped models.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.lfm3_header import write_lfm3_header

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "refresh_pull_catalog.py"


@pytest.fixture(scope="module")
def refresh():
    spec = importlib.util.spec_from_file_location("_refresh_pull_catalog_lfm3", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name, templates, context, want", [
    ("omni", ("tts",), 128000, 32768),
    ("audio", ("tts", "asr"), 128000, 32768),
    ("vl", (), 128000, 128000),
    ("text", (), 262144, 262144),
])
def test_every_reader_gives_the_same_trained_context(
        tmp_path, refresh, name, templates, context, want):
    from gmlx.commands.tool_preflight import synth_config_from_scans
    from gmlx.load.headerscan import scan_gguf
    from gmlx.serve.capacity import trained_context_length

    path = write_lfm3_header(tmp_path / f"{name}.gguf", context=context,
                             templates=templates)
    scan = scan_gguf(path, include_tensors=True)
    config = synth_config_from_scans([scan])
    assert config["max_position_embeddings"] == want
    assert trained_context_length(path) == want
    assert refresh.recorded_context(scan.kv, "lfm3") == want


def test_a_context_under_the_cap_stays(tmp_path, refresh):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.serve.capacity import trained_context_length

    path = write_lfm3_header(tmp_path / "small.gguf", context=4096,
                             templates=("tts",))
    assert trained_context_length(path) == 4096
    kv = scan_gguf(path, include_tensors=False).kv
    assert refresh.recorded_context(kv, "lfm3") == 4096


def test_another_arch_with_a_tts_template_is_not_capped():
    from gmlx.load.config_synth import trained_context_cap

    kv = {"general.architecture": "qwen2",
          "tokenizer.chat_templates": ["tts"]}
    assert trained_context_cap(kv) is None
