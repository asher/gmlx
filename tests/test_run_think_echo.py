"""Raw-path think-tag echo (generation.generate, verbose): a chat template
that pre-opens a thinking block leaves only the close tag in the generated
stream, so the run verb echoes the open tag before streaming. Stream and
mlx_lm.generate are mocked; no model loads."""
import importlib
import types

from gmlx.generation import generate

mlx_lm_pkg = importlib.import_module("mlx_lm")
mlg = importlib.import_module("mlx_lm.generate")


class _Tok:
    chat_template = None
    bos_token = None
    eos_token_id = 2
    eos_token_ids = {2}

    def encode(self, text, add_special_tokens=True):
        return [1]


def _resp(text):
    return types.SimpleNamespace(
        text=text, prompt_tokens=3, prompt_tps=1.0,
        generation_tokens=2, generation_tps=10.0, peak_memory=0.1,
    )


def _patch_stream(monkeypatch, texts):
    def fake(model, tokenizer, prompt, **kw):
        for t in texts:
            yield _resp(t)

    monkeypatch.setattr(mlg, "stream_generate", fake)


def test_open_prompt_echoes_tag_before_stream(monkeypatch, capsys):
    _patch_stream(monkeypatch, ["reasoning", "</think>", "answer"])
    out = generate(
        object(), _Tok(), "<|assistant|><think>\n",
        apply_chat_template=False, verbose=True,
    )
    assert out == "reasoning</think>answer"
    printed = capsys.readouterr().out
    assert printed.startswith("=" * 10 + "\n<think>\n")
    assert printed.index("<think>") < printed.index("</think>")


def test_open_prompt_suffixed_spelling_is_echoed(monkeypatch, capsys):
    _patch_stream(monkeypatch, ["x</think:opensource>y"])
    generate(
        object(), _Tok(), "<|hy_Assistant:opensource|><think:opensource>",
        apply_chat_template=False, verbose=True,
    )
    printed = capsys.readouterr().out
    assert "\n<think:opensource>\n" in printed


def test_closed_prompt_delegates_untouched(monkeypatch, capsys):
    calls = []

    def fake_generate(model, tokenizer, prompt, verbose=False, **kw):
        calls.append(prompt)
        return "plain"

    monkeypatch.setattr(mlx_lm_pkg, "generate", fake_generate)
    out = generate(
        object(), _Tok(), "<|assistant|>\n",
        apply_chat_template=False, verbose=True,
    )
    assert out == "plain"
    assert calls == ["<|assistant|>\n"]
    assert "<think>" not in capsys.readouterr().out


def test_stop_path_echoes_tag(monkeypatch, capsys):
    _patch_stream(monkeypatch, ["reasoning</think>", "answer STOP tail"])
    out = generate(
        object(), _Tok(), "<|assistant|><think>\n",
        apply_chat_template=False, verbose=True, stop=["STOP"],
    )
    assert out == "reasoning</think>answer "
    printed = capsys.readouterr().out
    assert "<think>\n" in printed
    assert printed.index("<think>") < printed.index("reasoning")


def test_quiet_run_does_not_echo(monkeypatch, capsys):
    def fake_generate(model, tokenizer, prompt, verbose=False, **kw):
        return "quiet"

    monkeypatch.setattr(mlx_lm_pkg, "generate", fake_generate)
    out = generate(
        object(), _Tok(), "<|assistant|><think>\n",
        apply_chat_template=False, verbose=False,
    )
    assert out == "quiet"
    assert "<think>" not in capsys.readouterr().out
