#!/usr/bin/env python3
"""Opt-in integration: the interactive chat TUI driven in a real pty.

Runs only when ``KQUANT_TEST_GGUF_DIR`` is set (the shared integration gate) -
it loads a small real model and drives ``gmlx chat`` through the scripted
session in ``tests/e2e/run_chat_pty_e2e.py``: quiet-load spinner + ``[load]``
summary, the prompt_toolkit banner, streaming turns with the v2 stat line,
the /command surface (/model /stats /theme /render /system /thinking-budget
/copy /retry /undo), Esc-cancel, /save + /export + /sessions, and a second
launch with ``--resume``. NOTE: /copy touches the real clipboard when a
clipboard tool is installed.

The deterministic CPU-only loop coverage lives in ``test_chat_e2e.py`` (mocked
model, no tty); this tier is the tty truth those mocks can't reach. The
standalone form (plus the optional multimodal arm) remains directly runnable::

    python tests/e2e/run_chat_pty_e2e.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))

import run_chat_pty_e2e as harness  # noqa: E402
from models import ModelRegistry  # noqa: E402

_HANDLES = ("qwen3_0_6b_q4", "qwen3_0_6b_q8")


def test_chat_tui_pty_session(gguf_dir, tmp_path):
    reg = ModelRegistry(root=str(gguf_dir))
    gguf = next((p for h in _HANDLES if (p := reg.find(h))), None)
    if gguf is None:
        pytest.skip(
            "no qwen3-0.6b GGUF under KQUANT_TEST_GGUF_DIR (fetch: gmlx pull "
            "hf:unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf "
            f"--to {gguf_dir}/qwen3-0.6b)"
        )
    try:
        cli = harness._cli(sys.executable)
    except FileNotFoundError as e:
        pytest.skip(str(e))

    checks = harness.Check()
    xdg = tmp_path / "xdg"
    harness._run_text_arm(
        cli, gguf, sys.executable, str(tmp_path / "chat-text.log"),
        checks, xdg=xdg,
    )
    harness._run_resume_arm(
        cli, gguf, str(tmp_path / "chat-resume.log"), checks, xdg=xdg
    )
    failed = [
        f"{name} - {detail.splitlines()[0] if detail else ''}"
        for name, passed, gated, detail in checks.rows
        if gated and not passed
    ]
    assert not failed, (
        f"{len(failed)} gated pty check(s) failed "
        f"(logs under {tmp_path}):\n" + "\n".join(failed)
    )
