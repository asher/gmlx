"""``--adapter`` on the MTP paths of run, bench and chat installs through the
load's ``before_wire`` hook, on the trunk, before placement. A plain run
installs once after the load. Every load seam is a stub; no GGUF, no GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gmlx.commands.cli as cli


@pytest.fixture(autouse=True)
def no_family_defaults(monkeypatch):
    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")


@pytest.fixture
def files(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    lora = tmp_path / "a.gguf"
    lora.write_bytes(b"GGUF")
    return str(gguf), str(lora)


class _Tok:
    chat_template = None


@pytest.fixture
def seen(monkeypatch):
    """Stubs the loads, the adapter install and the placement, and records
    each in order."""
    import gmlx.gen.generation as generation
    import gmlx.load.loader as loader
    import gmlx.spec.mtp_load as mtp_load

    events = []
    trunk = object()
    model = SimpleNamespace(language_model=trunk)

    def mtp(*a, before_wire=None, **k):
        events.append("load")
        if before_wire is not None:
            before_wire(trunk, {})
        return model, object(), {}, _Tok()

    def plain(*a, **k):
        events.append("load")
        return model, {}, _Tok()

    monkeypatch.setattr(mtp_load, "load_mtp_model", mtp)
    monkeypatch.setattr(loader, "load_model", plain)
    monkeypatch.setattr(cli, "_apply_cli_adapter",
                        lambda args, m, cfg, **k: events.append(("adapter", m)))
    monkeypatch.setattr(cli, "_apply_placement",
                        lambda args, m: events.append(("placement", m)))
    monkeypatch.setattr(generation, "generate", lambda *a, **k: None)
    monkeypatch.setattr(
        generation, "generate_speculative",
        lambda *a, **k: {"text": "", "tokens": 1, "elapsed_s": 0.1,
                         "prefill_s": 0.0, "decode_tps": 10.0,
                         "accept_rate": 0.5, "mean_accept_len": 1.5,
                         "rounds": 1})
    return events, model, trunk


def test_an_mtp_run_installs_the_adapter_before_placement(files, seen):
    gguf, lora = files
    events, _model, trunk = seen
    assert cli.main([gguf, "--prompt", "hi", "--mtp", "--adapter", lora]) == 0
    assert events == ["load", ("adapter", trunk), ("placement", trunk)]


def test_a_plain_run_installs_the_adapter_once(files, seen):
    gguf, lora = files
    events, model, _trunk = seen
    assert cli.main([gguf, "--prompt", "hi", "--no-mtp", "--adapter", lora]) == 0
    assert [e for e in events if e[0] == "adapter"] == [("adapter", model)]
    assert events.index(("adapter", model)) > events.index("load")


@pytest.mark.parametrize("mtp", [True, False], ids=["mtp", "plain"])
def test_bench_installs_the_adapter_once_before_placement(files, seen,
                                                         monkeypatch, mtp):
    import gmlx.commands.tool_preflight as tool_preflight
    import gmlx.gen.benchmarks as benchmarks

    gguf, lora = files
    events, model, trunk = seen
    monkeypatch.setattr(tool_preflight, "check_or_exit", lambda *a, **k: None)
    row = {"prefill_tps": 1.0, "tg_tps": 1.0, "spec_tps": 1.0,
           "accept_rate": 0.5, "mean_accept_len": 1.5, "speedup": 1.0}
    monkeypatch.setattr(benchmarks, "bench_tg_depth",
                        lambda m, t, depths, **k: {d: row for d in depths})
    argv = [gguf, "--bench-depths", "16", "--adapter", lora]
    assert cli.main(argv + (["--mtp"] if mtp else [])) == 0
    target = trunk if mtp else model
    assert events == ["load", ("adapter", target), ("placement", trunk)]


def test_the_chat_mtp_backend_installs_the_adapter_before_placement(
        files, seen, monkeypatch):
    import gmlx.tui.chat as chat

    gguf, lora = files
    events, _model, trunk = seen
    monkeypatch.setattr(chat, "_require_chat_template", lambda tok: None)
    args = chat._build_parser("gmlx chat").parse_args([gguf, "--adapter", lora])
    chat._backend_mtp_text(args, {})
    assert events == ["load", ("adapter", trunk), ("placement", trunk)]
