"""Faithful history render: key merge over the stock per-model rebuild."""

import importlib

import pytest

from gmlx.server_patches.render import install_faithful_history

openai_mod = importlib.import_module("mlx_vlm.server.openai")
prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")

QWEN_CFG = {"model_type": "qwen3_5"}
PLAIN_CFG = {"model_type": "some_text_model"}


@pytest.fixture(autouse=True)
def _stock_render():
    orig = openai_mod.apply_chat_template
    openai_mod.apply_chat_template = prompt_utils.apply_chat_template
    yield
    openai_mod.apply_chat_template = orig


class _Proc:
    """Records the messages the chat template is asked to render."""

    chat_template = "stub"

    def __init__(self):
        self.seen = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.seen = [dict(m) if isinstance(m, dict) else m for m in messages]
        return "RENDERED"


def _history():
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer",
         "reasoning_content": "let me think"},
        {"role": "user", "content": "and then?"},
    ]


def test_reasoning_content_survives_rebuild():
    install_faithful_history()
    msgs = openai_mod.apply_chat_template(
        _Proc(), QWEN_CFG, _history(), return_messages=True)
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["reasoning_content"] == "let me think"


def test_rebuilt_keys_win_over_sent_keys():
    install_faithful_history()
    history = _history()
    history[2]["content"] = [{"type": "text", "text": "and then?"}]
    msgs = openai_mod.apply_chat_template(
        _Proc(), QWEN_CFG, history, return_messages=True)
    assert msgs[2]["content"] != history[2]["content"]


def test_tool_passthrough_unchanged():
    install_faithful_history()
    history = _history()
    history[1]["tool_calls"] = [{
        "id": "c1", "type": "function",
        "function": {"name": "f", "arguments": "{}"}}]
    msgs = openai_mod.apply_chat_template(
        _Proc(), QWEN_CFG, history, return_messages=True)
    assert msgs[1]["tool_calls"][0]["id"] == "c1"
    assert msgs[1]["reasoning_content"] == "let me think"


def test_reasoning_reaches_template():
    install_faithful_history()
    proc = _Proc()
    out = openai_mod.apply_chat_template(proc, QWEN_CFG, _history())
    assert out == "RENDERED"
    assert proc.seen[1]["reasoning_content"] == "let me think"


def test_non_model_config_type_matches_stock():
    install_faithful_history()
    stock = prompt_utils.apply_chat_template(
        _Proc(), PLAIN_CFG, _history(), return_messages=True)
    patched = openai_mod.apply_chat_template(
        _Proc(), PLAIN_CFG, _history(), return_messages=True)
    assert patched == stock


def test_unreadable_item_skips_merge():
    install_faithful_history()
    history = _history() + [42]
    msgs = openai_mod.apply_chat_template(
        _Proc(), QWEN_CFG, history, return_messages=True)
    assert len(msgs) == 3
    assert "reasoning_content" not in msgs[1]


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("GMLX_FAITHFUL_HISTORY", "0")
    install_faithful_history()
    assert openai_mod.apply_chat_template is prompt_utils.apply_chat_template


def test_idempotent_install():
    install_faithful_history()
    once = openai_mod.apply_chat_template
    install_faithful_history()
    assert openai_mod.apply_chat_template is once


def test_retire_capture_wraps_faithful_render():
    from gmlx.server_patches.apc import install_retire_render_capture
    install_faithful_history()
    install_retire_render_capture()
    outer = openai_mod.apply_chat_template
    assert getattr(outer, "_kq_retire_capture", False)
    inner = [c.cell_contents for c in outer.__closure__
             if callable(getattr(c, "cell_contents", None))]
    assert any(getattr(f, "_kq_faithful_history", False) for f in inner)
