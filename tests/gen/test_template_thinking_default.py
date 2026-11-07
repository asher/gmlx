"""With no thinking control, a render through mlx-lm's TokenizerWrapper
follows the chat template's own default, in gmlx run, gmlx chat and serve
alike. The wrapper is real and wraps a GGUF tokenizer; only the model
stream is mocked."""
import importlib
import types
from pathlib import Path

import pytest

from gmlx.gen.generation import generate
from gmlx.tui import chat

mlg = importlib.import_module("gmlx._vendor.mlx_lm.generate")

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_THINK = ("<think>", "</think>")

# Thinks unless told not to.
_DEFAULT_ON = ("{{ messages[-1]['content'] }}|"
               "{% if enable_thinking is defined and enable_thinking is false %}"
               "NOTHINK{% else %}THINK{% endif %}")
# Thinks only when asked.
_DEFAULT_OFF = ("{{ messages[-1]['content'] }}|"
                "{% if enable_thinking is defined and enable_thinking %}"
                "THINK{% else %}NOTHINK{% endif %}")


def _wrapper(template: str, specials=()):
    """A real TokenizerWrapper over a byte-level GGUF tokenizer whose chat
    template is ``template``, with ``specials`` as single tokens."""
    from tokenizers import pre_tokenizers

    from gmlx._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper
    from gmlx.load.tokenizer import load_tokenizer_from_gguf

    head = ["<s>", "</s>", "<pad>", *specials]
    tokens = head + sorted(pre_tokenizers.ByteLevel.alphabet()) + ["He"]
    backend = load_tokenizer_from_gguf({
        "general.architecture": "qwen2",
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": ["H e"],
        "tokenizer.ggml.token_type": [3] * len(head)
        + [1] * (len(tokens) - len(head)),
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 2,
        "tokenizer.ggml.add_bos_token": False,
    }, "qwen2")
    backend.chat_template = template
    return TokenizerWrapper(backend)


# (template, vocab specials, has_thinking, default render)
_CASES = [
    pytest.param(_DEFAULT_ON, (), False, "THINK", id="default-on-no-think-vocab"),
    pytest.param(_DEFAULT_OFF, _THINK, True, "NOTHINK",
                 id="default-off-think-vocab"),
]


def _run_prompt(monkeypatch, tok, **kw):
    """The prompt gmlx run's generate() hands the model."""
    seen = []

    def fake(model, tokenizer, prompt, **_kw):
        seen.append(prompt)
        yield types.SimpleNamespace(
            text="4", prompt_tokens=1, prompt_tps=1.0, generation_tokens=1,
            generation_tps=1.0, peak_memory=0.1)

    monkeypatch.setattr(mlg, "stream_generate", fake)
    generate(object(), tok, "q", **kw)
    prompt = seen[0]
    return prompt if isinstance(prompt, str) else tok.decode(prompt)


@pytest.mark.parametrize("template,specials,has_thinking,default", _CASES)
def test_the_wrapper_keeps_the_templates_thinking_default(
        template, specials, has_thinking, default):
    tok = _wrapper(template, specials)
    assert tok.has_thinking is has_thinking
    msgs = [{"role": "user", "content": "q"}]
    assert tok.apply_chat_template(msgs, tokenize=False) == f"q|{default}"
    for flag, text in ((True, "THINK"), (False, "NOTHINK")):
        assert tok.apply_chat_template(
            msgs, tokenize=False, enable_thinking=flag) == f"q|{text}"


@pytest.mark.parametrize("template,specials,has_thinking,default", _CASES)
def test_run_keeps_the_templates_thinking_default(
        monkeypatch, template, specials, has_thinking, default):
    tok = _wrapper(template, specials)
    assert _run_prompt(monkeypatch, tok) == f"q|{default}"
    for flag, text in ((True, "THINK"), (False, "NOTHINK")):
        assert _run_prompt(monkeypatch, tok,
                           template_kwargs={"enable_thinking": flag}) == \
            f"q|{text}"


@pytest.mark.parametrize("template,specials,has_thinking,default", _CASES)
def test_chat_keeps_the_templates_thinking_default(
        template, specials, has_thinking, default):
    tok = _wrapper(template, specials)
    turn = [{"role": "user", "content": "q"}]
    assert chat._render_turn(tok, chat.ChatState(), turn, False, {}) == \
        f"q|{default}"
    for flag, text in ((True, "THINK"), (False, "NOTHINK")):
        assert chat._render_turn(tok, chat.ChatState(), turn, False,
                                 {"enable_thinking": flag}) == f"q|{text}"


@pytest.mark.parametrize("template,specials,has_thinking,default", _CASES)
def test_serve_keeps_the_templates_thinking_default(
        template, specials, has_thinking, default):
    from gmlx.serve import bridge_vlm
    from gmlx.serve.patches import install_chat_template_kwargs

    install_chat_template_kwargs()
    pu = importlib.import_module("gmlx._vendor.mlx_vlm.prompt_utils")
    tok = _wrapper(template, specials)
    proc = bridge_vlm._make_text_processor(tok)
    msgs = [{"role": "user", "content": "q"}]
    assert pu.get_chat_template(proc, msgs, True) == f"q|{default}"
    for flag, text in ((True, "THINK"), (False, "NOTHINK")):
        assert pu.get_chat_template(proc, msgs, True,
                                    enable_thinking=flag) == f"q|{text}"


def _thinking_args(template_path, thinking):
    return types.SimpleNamespace(
        thinking=thinking, reasoning_effort=None, thinking_budget=None,
        chat_template=str(template_path), gguf=None)


# Text-only Qwen3.5-9B and Gemma 4 templates think only when asked; their
# vocabularies have think tokens. The Qwen3 template thinks by default.
_QWEN35 = _FIXTURES / "qwen35_9b_template.jinja"
_GEMMA4 = _FIXTURES / "gemma4_12b_template.jinja"


def test_run_thinks_on_qwen35_9b_only_with_thinking_on(monkeypatch):
    tok = _wrapper(_QWEN35.read_text(), ("<|im_start|>", "<|im_end|>", *_THINK))
    assert tok.has_thinking
    assert _run_prompt(monkeypatch, tok).endswith("<think>\n\n</think>\n\n")
    tkw = chat.fold_thinking_flag(_thinking_args(_QWEN35, "on"), {})
    assert _run_prompt(monkeypatch, tok, template_kwargs=tkw).endswith(
        "assistant\n<think>\n")


def test_run_thinks_on_gemma4_only_with_thinking_on(monkeypatch):
    tok = _wrapper(_GEMMA4.read_text(),
                   ("<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>"))
    assert tok.has_thinking
    assert "<|think|>" not in _run_prompt(monkeypatch, tok)
    tkw = chat.fold_thinking_flag(_thinking_args(_GEMMA4, "on"), {})
    assert "<|think|>" in _run_prompt(monkeypatch, tok, template_kwargs=tkw)


def test_run_thinks_by_default_on_a_default_on_template(monkeypatch):
    path = _FIXTURES / "qwen3_template.jinja"
    tok = _wrapper(path.read_text(), ("<|im_start|>", "<|im_end|>"))
    assert not tok.has_thinking
    assert _run_prompt(monkeypatch, tok).endswith("<|im_start|>assistant\n")
    tkw = chat.fold_thinking_flag(_thinking_args(path, "off"), {})
    assert _run_prompt(monkeypatch, tok, template_kwargs=tkw).endswith(
        "<think>\n\n</think>\n\n")


# Fixtures that are not chat templates: a systemone question template and
# an excerpt of MiniMax-M3.1's thinking block.
_NOT_CHAT = {"granite_switch_systemone", "minimax_m31_thinking"}


@pytest.mark.parametrize(
    "path", [p for p in sorted(_FIXTURES.glob("*.jinja"))
             if p.stem not in _NOT_CHAT], ids=lambda p: p.stem)
def test_every_template_fixture_renders_with_no_thinking_control(path):
    tok = _wrapper(path.read_text(), _THINK)
    msgs = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2+2?"}]
    out = tok.apply_chat_template(msgs, tokenize=False,
                                  add_generation_prompt=True)
    assert "2+2" in out
