#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""A d1 decision model is refused on every load path.

A d1 GGUF is arch ``lfm3`` with ``lfm3.attention.causal`` false. The
causal forward would run it and answer wrongly with no error, so run,
chat and serve each refuse the file by name. The fixture carries the
header facts of a d1 file and no weights.
"""

from __future__ import annotations

import pytest

from tests.lfm3_header import write_lfm3_header

_MESSAGE = "d1 decision models are not supported yet"


@pytest.fixture()
def d1_gguf(tmp_path) -> str:
    return write_lfm3_header(tmp_path / "d1.gguf", templates=("systemone",),
                             causal=False, decision="lfm3-d1")


@pytest.mark.parametrize("argv", [
    ["run", "--prompt", "hi", "--max-tokens", "2"],
    ["chat"],
])
def test_run_and_chat_refuse_a_d1_file(d1_gguf, capsys, argv):
    from gmlx.commands.cli import umbrella_main

    try:
        code = umbrella_main([argv[0], d1_gguf, *argv[1:]])
    except SystemExit as exc:
        code = exc.code
    out = capsys.readouterr()
    assert code not in (0, None)
    assert _MESSAGE in out.err + out.out


def test_serve_refuses_a_d1_file(d1_gguf):
    from gmlx.serve.bridge_vlm import load_serveable_model

    with pytest.raises(NotImplementedError, match=_MESSAGE):
        load_serveable_model(d1_gguf)


def test_a_causal_header_is_not_refused(tmp_path):
    """The key set to true, as a later converter may write it, loads."""
    from gmlx.commands.tool_preflight import synth_config_from_scans
    from gmlx.load.headerscan import scan_gguf

    path = write_lfm3_header(tmp_path / "causal.gguf", causal=True)
    config = synth_config_from_scans([scan_gguf(path, include_tensors=True)])
    assert config["model_type"] == "lfm3"
