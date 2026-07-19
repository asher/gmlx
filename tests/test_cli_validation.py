#!/usr/bin/env python3
"""`cli.main` pre-load validation: fail-fast path checks (before the heavy
loader imports), --mmproj flag-compatibility errors/warnings, and the
--speculative ignored-flags warning. CPU-only: every load seam is a stub.
"""

from __future__ import annotations

import sys

import pytest

from gmlx import cli  # noqa: E402

# The modules main() imports for its dispatch exception types - transitively
# heavy (transformers). Poisoning them proves a check runs *before* them.
_HEAVY = ("gmlx.arch_table", "gmlx.preflight", "gmlx.vlm")


@pytest.fixture
def no_heavy_imports(monkeypatch):
    """Make the loader-stack imports explode so a test fails loudly if the
    pre-load validation no longer runs first."""
    for mod in _HEAVY:
        monkeypatch.setitem(sys.modules, mod, None)


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def mmproj(tmp_path):
    p = tmp_path / "mmproj.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


# fail-fast path validation (before any heavy import)
def test_missing_gguf_fails_before_heavy_imports(no_heavy_imports, capsys):
    assert cli.main(["/no/such/model.gguf"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_remote_ref_hint_fails_before_heavy_imports(no_heavy_imports, capsys):
    assert cli.main(["hf:org/repo/model.gguf"]) == 2
    assert "gmlx pull" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--mmproj", "--draft-gguf", "--adapter"])
def test_missing_companion_file_fails_fast(no_heavy_imports, gguf, flag, capsys):
    assert cli.main([gguf, flag, "/no/such/companion.gguf"]) == 2
    err = capsys.readouterr().err
    assert flag in err and "no such file" in err


# --mmproj flag compatibility
@pytest.mark.parametrize("extra,named", [
    (["--bench", "512"], "--bench"),
    (["--bench-depths", "0,4096"], "--bench-depths"),
    (["--report-only"], "--report-only"),
    (["--stream-cpu"], "--stream-cpu"),
    (["--stream-experts"], "--stream-experts"),
])
def test_mmproj_rejects_unsupported_modes(no_heavy_imports, gguf, mmproj,
                                          extra, named, capsys):
    assert cli.main([gguf, "--mmproj", mmproj, *extra]) == 2
    err = capsys.readouterr().err
    assert named in err and "not supported with --mmproj" in err


def test_mmproj_error_names_all_offending_flags(no_heavy_imports, gguf, mmproj,
                                                capsys):
    rc = cli.main([gguf, "--mmproj", mmproj, "--bench", "512", "--stream-cpu"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--bench" in err and "--stream-cpu" in err


def test_mmproj_warns_on_text_only_sampling_flags(gguf, mmproj, monkeypatch,
                                                  capsys):
    monkeypatch.setattr(cli, "_run_vlm", lambda args: 0)
    rc = cli.main([gguf, "--mmproj", mmproj, "--stop", "END",
                   "--xtc-probability", "0.5"])
    assert rc == 0                       # warned, still dispatched to VLM
    err = capsys.readouterr().err
    assert "ignored in VLM mode" in err
    assert "--stop" in err and "--xtc-probability" in err
    assert "--xtc-threshold" not in err  # unset flags are not named


def test_mmproj_no_warning_when_flags_unset(gguf, mmproj, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_run_vlm", lambda args: 0)
    assert cli.main([gguf, "--mmproj", mmproj]) == 0
    assert "ignored in VLM mode" not in capsys.readouterr().err


# --speculative ignored-flags warning
@pytest.fixture
def spec_stubs(monkeypatch):
    """Stub the MTP load/generate seams so the speculative branch runs on CPU."""
    from gmlx import generation, mtp_load

    class _Tok:
        chat_template = None

    monkeypatch.setattr(mtp_load, "load_mtp_model",
                        lambda *a, **k: (object(), object(), {}, _Tok()))
    seen = {}

    def fake_generate_speculative(model, drafter, tok, prompt, **kwargs):
        seen.update(kwargs)
        return {"text": "", "tokens": 1, "elapsed_s": 0.1, "prefill_s": 0.0,
                "decode_tps": 10.0, "accept_rate": 0.5, "mean_accept_len": 1.5,
                "rounds": 1}
    monkeypatch.setattr(generation, "generate_speculative",
                        fake_generate_speculative)
    return seen


def test_speculative_warns_on_dropped_flags(gguf, spec_stubs, capsys):
    rc = cli.main([gguf, "--speculative", "--stop", "END", "--kv-bits", "8",
                   "--presence-penalty", "0.5", "--system-prompt", "be brief"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "not applied on the MTP path" in err
    for flag in ("--stop", "--presence-penalty"):
        assert flag in err
    assert "--logit-bias" not in err          # unset flags are not named
    assert "--system-prompt" not in err       # supported on MTP (baked), not dropped
    # --kv-bits is handled by the MTP engine itself (pooled packing or an
    # accurate per-model note), not the generic dropped-flags warning.
    assert "--kv-bits" not in err
    assert spec_stubs["kv_bits"] == 8         # forwarded to the engine
    assert spec_stubs["system_prompt"] == "be brief"   # forwarded to the walk


def test_speculative_no_warning_on_supported_flags(gguf, spec_stubs, capsys):
    rc = cli.main([gguf, "--speculative", "--temp", "0.7", "--top-k", "40"])
    assert rc == 0
    assert "not applied on the MTP path" not in capsys.readouterr().err
    # the supported sampling surface is forwarded
    assert spec_stubs["temp"] == 0.7 and spec_stubs["top_k"] == 40


# native-head MTP auto-enable (resolve_speculative)
@pytest.fixture
def native_head(monkeypatch):
    """Pretend the positional GGUF carries a native MTP head (skip the GGUF read)."""
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: True)


def _args(argv):
    return cli._build_parser().parse_args(argv)


def test_resolve_auto_on_for_native_head_clean(native_head, gguf):
    on, note = cli.resolve_speculative(_args([gguf]), gguf)
    assert on and "native MTP head detected" in note


def test_resolve_auto_off_without_native_head(gguf, monkeypatch):
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: False)
    on, note = cli.resolve_speculative(_args([gguf]), gguf)
    assert not on and note == ""


@pytest.mark.parametrize("flag", [
    ["--max-kv-size", "4096"], ["--repetition-penalty", "1.1"],
    ["--stop", "END"], ["--kv-bits", "8"], ["--logit-bias", '{"1": -100}'],
    ["--xtc-probability", "0.5"],
])
def test_resolve_auto_stays_on_despite_soft_flag(native_head, gguf, flag):
    # sticky AUTO: a soft sampler flag is dropped+warned at dispatch, NOT deferred,
    # so a habitual penalty never silently disables auto-MTP.
    on, note = cli.resolve_speculative(_args([gguf, *flag]), gguf)
    assert on and "native MTP head detected" in note


def test_resolve_stays_on_with_system_prompt(native_head, gguf):
    # both surfaces keep MTP on; the system prompt is baked into the templated turn.
    a = _args([gguf, "--system-prompt", "be brief", "--stop", "END"])
    assert cli.resolve_speculative(a, gguf)[0] is True
    assert cli.resolve_speculative(a, gguf)[0] is True


def test_resolve_hard_flag_defers_silently(native_head, gguf):
    on, note = cli.resolve_speculative(_args([gguf, "--adapter", "/x.gguf"]),
                                       gguf)
    assert not on and note == ""        # no note: respecting an incompatible request


def test_resolve_explicit_mtp_forces_on_despite_flag(native_head, gguf):
    on, note = cli.resolve_speculative(
        _args([gguf, "--mtp", "--repetition-penalty", "1.3"]), gguf)
    assert on and note == ""            # explicit: the warning happens at dispatch


def test_resolve_no_mtp_forces_off(native_head, gguf):
    assert cli.resolve_speculative(_args([gguf, "--no-mtp"]),
                                   gguf)[0] is False


def test_resolve_config_false_disables_auto(native_head, gguf):
    a = _args([gguf])
    a.speculative = False               # config 'speculative: false' overlay
    on, note = cli.resolve_speculative(a, gguf)
    assert not on and note == ""


def test_main_auto_enables_mtp_for_native_head(gguf, spec_stubs, native_head,
                                               capsys):
    rc = cli.main([gguf, "--temp", "0.7"])   # no --speculative
    assert rc == 0
    assert "native MTP head detected" in capsys.readouterr().out
    assert spec_stubs["temp"] == 0.7         # dispatched through generate_speculative


# VLM x MTP: a loaded VLM serves text-only requests through the MTP path
@pytest.fixture
def draft(tmp_path):
    p = tmp_path / "draft.gguf"
    p.write_bytes(b"GGUF")
    return str(p)


@pytest.fixture
def vlm_mtp_stubs(monkeypatch):
    """Stub the VLM x MTP load/generate seams + the plain-VLM fallback (CPU)."""
    from gmlx import generation, mtp_load

    class _Tok:
        chat_template = None

    calls = {"mtp": 0, "vlm": 0, "gen_kwargs": None}

    def fake_load_vlm_mtp_model(*a, **k):
        return object(), object(), {}, _Tok(), object()

    def fake_generate_speculative(model, drafter, tok, prompt, **kwargs):
        calls["mtp"] += 1
        calls["gen_kwargs"] = kwargs
        return {"text": "", "tokens": 1, "elapsed_s": 0.1, "prefill_s": 0.0,
                "decode_tps": 10.0, "accept_rate": 0.5, "mean_accept_len": 1.5,
                "rounds": 1}

    def fake_run_vlm(args):
        calls["vlm"] += 1
        return 0

    monkeypatch.setattr(mtp_load, "load_vlm_mtp_model", fake_load_vlm_mtp_model)
    monkeypatch.setattr(generation, "generate_speculative", fake_generate_speculative)
    monkeypatch.setattr(cli, "_run_vlm", fake_run_vlm)
    return calls


def test_vlm_text_only_with_draft_routes_to_mtp(gguf, mmproj, draft,
                                                vlm_mtp_stubs, capsys):
    rc = cli.main([gguf, "--mmproj", mmproj, "--draft-gguf", draft,
                   "--prompt", "hi", "--system-prompt", "be brief"])
    assert rc == 0
    assert vlm_mtp_stubs["mtp"] == 1 and vlm_mtp_stubs["vlm"] == 0
    assert vlm_mtp_stubs["gen_kwargs"]["system_prompt"] == "be brief"
    assert "MTP speculative" in capsys.readouterr().out


def test_vlm_image_with_draft_routes_to_plain_vlm(gguf, mmproj, draft,
                                                  vlm_mtp_stubs):
    # an image request keeps the VLM path; the drafter is unused that request
    rc = cli.main([gguf, "--mmproj", mmproj, "--draft-gguf", draft,
                   "--image", "/some/pic.png", "--prompt", "describe"])
    assert rc == 0
    assert vlm_mtp_stubs["vlm"] == 1 and vlm_mtp_stubs["mtp"] == 0


def test_vlm_native_head_text_only_routes_to_mtp(gguf, mmproj, vlm_mtp_stubs,
                                                 monkeypatch):
    # no --draft-gguf: a native MTP head in the LLM GGUF drives VLM x MTP for a
    # text-only request (qwen3.5/3.6).
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: True)
    rc = cli.main([gguf, "--mmproj", mmproj, "--prompt", "hi"])
    assert rc == 0
    assert vlm_mtp_stubs["mtp"] == 1 and vlm_mtp_stubs["vlm"] == 0


def test_vlm_no_drafter_routes_to_plain_vlm(gguf, mmproj, vlm_mtp_stubs,
                                            monkeypatch):
    # no drafter available (no --draft-gguf, no native head) -> plain VLM
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: False)
    rc = cli.main([gguf, "--mmproj", mmproj, "--prompt", "hi"])
    assert rc == 0
    assert vlm_mtp_stubs["vlm"] == 1 and vlm_mtp_stubs["mtp"] == 0


def test_vlm_native_head_image_request_stays_plain_vlm(gguf, mmproj, vlm_mtp_stubs,
                                                       monkeypatch):
    # even with a native head, an image request stays on the VLM path
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: True)
    rc = cli.main([gguf, "--mmproj", mmproj, "--image", "/p.png", "--prompt", "x"])
    assert rc == 0
    assert vlm_mtp_stubs["vlm"] == 1 and vlm_mtp_stubs["mtp"] == 0


# _vlm_mtp_drafter_available precedence
def test_drafter_available_native_head(gguf, mmproj, monkeypatch):
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: True)
    assert cli._vlm_mtp_drafter_available(_args([gguf, "--mmproj", mmproj])) is True


def test_drafter_available_assistant_draft(gguf, mmproj, draft, monkeypatch):
    # an assistant drafter qualifies without consulting the GGUF for a native head
    monkeypatch.setattr(cli, "_has_native_mtp_head",
                        lambda *a, **k: pytest.fail("should not peek with --draft-gguf"))
    a = _args([gguf, "--mmproj", mmproj, "--draft-gguf", draft])
    assert cli._vlm_mtp_drafter_available(a) is True


def test_drafter_available_no_mtp_opts_out(gguf, mmproj, monkeypatch):
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: True)
    a = _args([gguf, "--mmproj", mmproj, "--no-mtp"])
    assert cli._vlm_mtp_drafter_available(a) is False


def test_drafter_available_none_without_head_or_draft(gguf, mmproj, monkeypatch):
    monkeypatch.setattr(cli, "_has_native_mtp_head", lambda *a, **k: False)
    assert cli._vlm_mtp_drafter_available(_args([gguf, "--mmproj", mmproj])) is False


def test_speculative_forwards_template_config_and_warns_new_drops(
        gguf, spec_stubs, capsys):
    rc = cli.main([gguf, "--speculative",
                   "--chat-template-config", '{"enable_thinking": false}',
                   "--thinking-budget", "256",
                   "--prefill-step-size", "1024"])
    assert rc == 0
    err = capsys.readouterr().err
    # forwarded, not dropped
    assert spec_stubs["template_kwargs"] == {"enable_thinking": False}
    assert "--chat-template-config" not in err
    # newly-named drops fire the warning + --no-mtp hint
    for flag in ("--thinking-budget", "--prefill-step-size"):
        assert flag in err
    assert "not applied on the MTP path" in err


def test_vlm_mtp_forwards_kv_bits_and_template_config(gguf, mmproj, draft,
                                                      vlm_mtp_stubs):
    rc = cli.main([gguf, "--mmproj", mmproj, "--draft-gguf", draft,
                   "--prompt", "hi", "--kv-bits", "8",
                   "--chat-template-config", '{"enable_thinking": false}'])
    assert rc == 0
    kw = vlm_mtp_stubs["gen_kwargs"]
    assert kw["kv_bits"] == 8 and kw["kv_group_size"] == 64
    assert kw["template_kwargs"] == {"enable_thinking": False}
