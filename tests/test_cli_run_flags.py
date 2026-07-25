#!/usr/bin/env python3
"""`cli.main` run-path flag plumbing: --help-documented flags reach
generation.generate on the plain (non-MTP, non-VLM) path, --prompt-file feeds the
prompt, --report-only inventories without building, and the documented exit
codes (1 = load refusal, 2 = bad invocation) hold. CPU-only: every load seam
is a stub."""

from __future__ import annotations

import pytest

from gmlx import cli


@pytest.fixture(autouse=True)
def no_family_defaults(monkeypatch):
    """Keep argparse defaults: family seeding would fill flags we didn't pass."""
    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def gen(monkeypatch):
    """Stub the plain-path load/generate seams; returns the recorded generate call."""
    from gmlx import generation, loader

    seen = {}

    def fake_generate(model, tok, prompt, **kwargs):
        seen["prompt"] = prompt
        seen.update(kwargs)

    monkeypatch.setattr(loader, "load_model",
                        lambda *a, **k: (object(), {}, object()))
    monkeypatch.setattr(generation, "generate", fake_generate)
    return seen


# flag forwarding on the plain generate path
def test_run_forwards_sampling_and_kv_flags(gguf, gen):
    rc = cli.main([gguf,
                   "--stop", "END", "--stop", "DONE",
                   "--kv-bits", "8", "--kv-group-size", "32",
                   "--max-kv-size", "4096", "--quantized-kv-start", "16",
                   "--thinking-budget", "100",
                   "--repetition-penalty", "1.1",
                   "--presence-penalty", "0.25", "--frequency-penalty", "0.5",
                   "--xtc-probability", "0.5", "--xtc-threshold", "0.1",
                   "--prefill-step-size", "512",
                   "--system-prompt", "sys"])
    assert rc == 0
    assert gen["stop"] == ["END", "DONE"]              # repeatable, in order
    assert gen["kv_bits"] == 8 and gen["kv_group_size"] == 32
    assert gen["max_kv_size"] == 4096 and gen["quantized_kv_start"] == 16
    assert gen["thinking_budget"] == 100
    assert gen["repetition_penalty"] == 1.1
    assert gen["presence_penalty"] == 0.25 and gen["frequency_penalty"] == 0.5
    assert gen["xtc_probability"] == 0.5 and gen["xtc_threshold"] == 0.1
    assert gen["prefill_step_size"] == 512
    assert gen["system_prompt"] == "sys"
    for k in ("kv_bits", "kv_group_size", "max_kv_size", "thinking_budget",
              "prefill_step_size"):
        assert isinstance(gen[k], int)                 # type=int survived


def test_run_no_chat_template_and_seed(gguf, gen, monkeypatch):
    import mlx.core as mx

    seeded = []
    monkeypatch.setattr(mx.random, "seed", lambda v: seeded.append(v))
    rc = cli.main([gguf, "--no-chat-template", "--seed", "7", "--prompt", "hi"])
    assert rc == 0
    assert gen["apply_chat_template"] is False
    assert gen["prompt"] == "hi"
    assert seeded == [7]                               # --seed reached mx.random.seed


# --prompt-file
def test_prompt_file_reads_prompt(gguf, gen, tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("prompt from file\n")
    assert cli.main([gguf, "--prompt-file", str(pf)]) == 0
    assert gen["prompt"] == "prompt from file\n"       # verbatim, newline kept


def test_prompt_file_missing_exits_2(gguf, gen, capsys):
    assert cli.main([gguf, "--prompt-file", "/no/such/prompt.txt"]) == 2
    err = capsys.readouterr().err
    assert "--prompt-file" in err and "error:" in err


# exit 1 = load refusal (docs/cli.md)
def test_run_codec_refusal_exits_1(gguf, monkeypatch, capsys):
    from gmlx import loader
    from gmlx.preflight import UnsupportedCodecError

    exc = UnsupportedCodecError("gemma4", {"TQ1_0": 3})

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(loader, "load_model", boom)
    assert cli.main([gguf]) == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    # the refusal itself is one line naming the codec
    assert err.strip().splitlines()[-1] == f"error: {exc}"
    assert "TQ1_0" in str(exc)


# exit 2 = flag conflicts refused before any load. --speculative
# --stream-experts is NOT here anymore: streaming composes with MTP
# (placement after load_mtp_model; see resolve_speculative).
@pytest.mark.parametrize("extra,named", [
    (["--adapter", "/x.gguf", "--speculative"], "--speculative"),
    (["--speculative", "--stream-cpu"], "--stream-cpu"),
])
def test_run_adapter_conflicts_exit_2(gguf, extra, named, capsys):
    assert cli.main([gguf, *extra]) == 2
    err = capsys.readouterr().err
    assert named in err and "not supported yet" in err


# --report-only
@pytest.fixture
def report_stubs(monkeypatch):
    """Stub the _report_only seams (wire bytes + remap + inventory + tokenizer);
    poison the model-build/generate seams. Returns the preflight module."""
    import gmlx.preflight as preflight_mod
    import gmlx.tokenizer as tok_mod
    from gmlx import generation, loader

    monkeypatch.setattr(preflight_mod, "preflight", lambda path, arch=None: None)
    monkeypatch.setattr(
        loader, "load_gguf_wire_bytes",
        lambda path, zero_copy=True: ({"t": 0}, {"t": {}}, "gemma4", {}, {}))
    from gmlx import gguf_meta
    monkeypatch.setattr(gguf_meta, "read_int", lambda meta, key: None)
    monkeypatch.setattr(gguf_meta, "first_nonzero_int", lambda meta, key: None)
    monkeypatch.setattr(
        loader, "remap_arrays",
        lambda arrays, kq, arch, no_remap=False, n_head=None, n_head_kv=None:
        (arrays, {}, {}))
    monkeypatch.setattr(loader, "print_inventory",
                        lambda *a, **k: print("[inventory] stub"))
    monkeypatch.setattr(loader, "_resolve_chat_template", lambda x: None)

    class _Tok:
        chat_template = "T"

        def apply_chat_template(self, msgs, tokenize=False,
                                add_generation_prompt=True):
            return "RENDERED: " + msgs[0]["content"]

    monkeypatch.setattr(tok_mod, "load_tokenizer_from_gguf",
                        lambda meta, arch, chat_template_override=None: _Tok())
    monkeypatch.setattr(
        loader, "load_model",
        lambda *a, **k: pytest.fail("--report-only must not build the model"))
    monkeypatch.setattr(
        generation, "generate",
        lambda *a, **k: pytest.fail("--report-only must not generate"))
    return preflight_mod


def test_report_only_inventories_and_skips_generate(gguf, report_stubs, capsys):
    assert cli.main([gguf, "--report-only", "--prompt", "hello"]) == 0
    out = capsys.readouterr().out
    assert "[inventory] stub" in out
    assert "=== rendered prompt ===" in out and "RENDERED: hello" in out


def test_report_only_swallows_arch_error(gguf, report_stubs, monkeypatch, capsys):
    # an arch the loader can't build must still inventory (rc 0)
    from gmlx.arch_table import UnsupportedArchError

    def boom(path, arch=None):
        raise UnsupportedArchError("arch 'foo' not supported")

    monkeypatch.setattr(report_stubs, "preflight", boom)
    assert cli.main([gguf, "--report-only"]) == 0
    assert "[inventory] stub" in capsys.readouterr().out


def test_report_only_codec_error_still_refuses(gguf, report_stubs, monkeypatch,
                                               capsys):
    from gmlx.preflight import UnsupportedCodecError

    def boom(path, arch=None):
        raise UnsupportedCodecError("gemma4", {"TQ1_0": 3})

    monkeypatch.setattr(report_stubs, "preflight", boom)
    assert cli.main([gguf, "--report-only"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err and "TQ1_0" in err


# --bench-depths list validation (before any load)
def test_bench_depths_bad_list_exits_2(gguf, monkeypatch, capsys):
    from gmlx import loader, mtp_load

    monkeypatch.setattr(loader, "load_model",
                        lambda *a, **k: pytest.fail("must not load"))
    monkeypatch.setattr(mtp_load, "load_mtp_model",
                        lambda *a, **k: pytest.fail("must not load"))
    assert cli.main([gguf, "--bench-depths", "4096,abc"]) == 2
    err = capsys.readouterr().err
    assert "--bench-depths" in err and "not a comma-separated int list" in err


# --max-tokens: unset = generate until EOS (huge effective cap); N = cap
def test_max_tokens_default_uncapped(gguf, gen, capsys):
    assert cli.main([gguf]) == 0
    assert gen["max_tokens"] == cli._UNCAPPED_MAX_TOKENS
    assert "max_tokens=until-eos" in capsys.readouterr().out


def test_max_tokens_explicit_cap(gguf, gen, capsys):
    assert cli.main([gguf, "--max-tokens", "64"]) == 0
    assert gen["max_tokens"] == 64
    assert "max_tokens=64" in capsys.readouterr().out


def test_warn_cap_hit_only_when_capped(capsys):
    class A:
        pass

    a = A()
    a.max_tokens = 64
    a._max_tokens_capped = True
    cli.warn_cap_hit(a, 64)
    assert "--max-tokens cap (64)" in capsys.readouterr().err
    a.max_tokens = cli._UNCAPPED_MAX_TOKENS
    a._max_tokens_capped = False
    cli.warn_cap_hit(a, 10)
    assert capsys.readouterr().err == ""
