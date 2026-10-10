"""The rope scaling options from the command line and the serve config to
the load: flags, the `rope_scaling` block, the load signature, the
synthesized config's context length, and the arch gate. No model files."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from gmlx import config as cfgmod
from gmlx.commands import cli
from gmlx.config import ConfigError, build_config, resolve_model
from gmlx.load import rope_scaling as rs
from gmlx.load.rope_scaling import RopeOverride

CARD = RopeOverride(rope_scaling="yarn", rope_scale=4.0, yarn_orig_ctx=262144)


def _parse(argv):
    ap = argparse.ArgumentParser()
    cli.add_rope_args(ap)
    return ap.parse_args(argv)


def test_flags_parse_to_an_override():
    args = _parse(["--rope-scaling", "yarn", "--rope-scale", "4",
                   "--yarn-orig-ctx", "262144"])
    assert cli.rope_override_from_args(args) == CARD
    assert cli.rope_override_from_args(_parse([])) is None
    with pytest.raises(SystemExit, match="disagree"):
        cli.rope_override_from_args(
            _parse(["--rope-scale", "4", "--rope-freq-scale", "0.5"]))
    with pytest.raises(SystemExit):
        _parse(["--rope-scaling", "dynamic"])


@pytest.mark.parametrize("prog", ["run", "chat"])
def test_run_and_chat_carry_the_flags(prog):
    if prog == "run":
        ap = cli._build_parser()
    else:
        from gmlx.tui.chat import _build_parser
        ap = _build_parser()
    dests = {a.dest for a in ap._actions}
    assert set(cli.ROPE_FLAGS) <= dests


def _doc(**rope):
    return {"models": {"m": {"path": "/abs/m.gguf", **rope}}}


def test_config_block_parses_and_joins_the_signature():
    plain = resolve_model("m", build_config(_doc()))
    four = resolve_model("m", build_config(_doc(rope_scaling={
        "type": "yarn", "factor": 4, "original_context": 262144})))
    two = resolve_model("m", build_config(_doc(rope_scaling={
        "type": "yarn", "factor": 2})))
    empty = resolve_model("m", build_config(_doc(rope_scaling={})))
    assert plain.rope_scaling is None and empty.rope_scaling is None
    assert four.rope_scaling == CARD
    assert plain.load_signature() == empty.load_signature()
    assert len({plain.load_signature(), four.load_signature(),
                two.load_signature()}) == 3


def test_signature_without_options_is_unchanged():
    common = dict(path="/p", load={}, cache={}, system=None, speculative=False,
                  mmproj=None, draft_gguf=None, pin=False, ttl_s=None)
    a = cfgmod.ResolvedModel(id="a", sampling={}, **common)
    assert len(a.load_signature()) == 18


@pytest.mark.parametrize("block,needle", [
    ({"rope_type": "yarn"}, "unknown key"),
    ({"type": "dynamic"}, "not one of"),
    ({"factor": 0}, "positive"),
    ("yarn", "mapping"),
])
def test_config_block_errors(block, needle):
    with pytest.raises(ConfigError, match=needle):
        build_config(_doc(rope_scaling=block))


def test_config_block_reaches_run_args_and_flags_win():
    rm = resolve_model("m", build_config(_doc(rope_scaling={
        "type": "yarn", "factor": 4, "beta_fast": 16})))
    args = _parse(["--rope-freq-scale", "0.5"])
    applied = cli._apply_resolved_to_args(args, rm, {"rope_freq_scale"})
    assert "rope-scaling" in applied and "yarn-beta-fast" in applied
    ov = cli.rope_override_from_args(args)
    assert ov is not None
    assert ov.rope_scaling == "yarn" and ov.freq_scale == 0.5
    assert ov.rope_scale is None and ov.yarn_beta_fast == 16


def _serve_args(**kw):
    import gmlx.serve.server as server

    ap = argparse.ArgumentParser()
    server._add_serve_args(ap)
    a = ap.parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_positional_serve_takes_the_flags(tmp_path):
    import gmlx.serve.server as server

    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    a = _serve_args(model=str(p), rope_scaling="yarn", rope_scale=4.0,
                    yarn_orig_ctx=262144)
    cfg = server._single_model_cfg(a)
    assert cfg.models[cfg.defaults.model].rope_scaling == CARD
    argv = server._bg_serve_args(a, None)
    i = argv.index("--rope-scale")
    assert argv[i + 1] == "4.0" and "--yarn-orig-ctx" in argv


def test_config_mode_refuses_the_flags(tmp_path):
    import gmlx.serve.server as server

    a = _serve_args(config=str(tmp_path / "c.yaml"), rope_scale=4.0)
    with pytest.raises(ConfigError, match="rope_scaling block"):
        server._resolve_mode_cfg(a)


def _meta(arch="qwen3", ctx=40960):
    return {"general.architecture": arch,
            f"{arch}.context_length": ctx,
            f"{arch}.rope.freq_base": 1e6}


def test_config_carries_the_effective_context(monkeypatch):
    from gmlx.load import config_synth

    monkeypatch.setattr(rs, "WIRED_ARCHES", {"qwen3"})
    cfg = {"max_position_embeddings": 40960}
    config_synth._add_rope_scaling(_meta(), cfg, "qwen3",
                                   RopeOverride(rope_scaling="yarn",
                                                rope_scale=4.0))
    assert cfg["max_position_embeddings"] == 163840
    assert cfg[rs.TRAINED_CONTEXT_KEY] == 40960
    s = rs.from_config(cfg)
    assert s.factor == 4.0 and s.original_context == 40960
    plain = {"max_position_embeddings": 40960}
    config_synth._add_rope_scaling(_meta(), plain, "qwen3", None)
    assert plain == {"max_position_embeddings": 40960}
    assert rs.from_config(plain) is rs.NO_SCALING


def test_options_on_an_unwired_arch_refuse(monkeypatch):
    from gmlx.load import config_synth

    monkeypatch.setattr(rs, "WIRED_ARCHES", set())
    with pytest.raises(NotImplementedError, match="not supported for the qwen3"):
        config_synth._add_rope_scaling(_meta(), {}, "qwen3", CARD)
    # The active scope reaches the synthesizer without a parameter.
    with rs.override_scope(CARD), pytest.raises(NotImplementedError):
        config_synth._add_rope_scaling(_meta(), {}, "qwen3", None)
    config_synth._add_rope_scaling(_meta(), {}, "qwen3", None)


def test_scoped_entry_point_sets_and_restores():
    seen = []

    @rs.scoped
    def load(path, *, rope_override=None):
        seen.append(rs.active_override())

    load("x", rope_override=CARD)
    load("x")
    assert seen == [CARD, rs.NO_OVERRIDE]
    assert rs.active_override() is rs.NO_OVERRIDE


def test_reported_context_follows_the_options(monkeypatch, tmp_path):
    from gmlx.serve import capacity

    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF")
    meta = _meta("qwen4exp", 262144)
    monkeypatch.setattr("gmlx.load.headerscan.scan_gguf",
                        lambda path, include_tensors=False:
                        SimpleNamespace(kv=meta))
    capacity._CTX_CACHE.clear()
    assert capacity.trained_context_length(str(p)) == 262144
    assert capacity.effective_context_length(str(p), CARD) == 1048576
    assert capacity.effective_context_length(
        str(p), RopeOverride(rope_scaling="yarn", rope_scale=2.0)) == 524288
    capacity._CTX_CACHE.clear()
