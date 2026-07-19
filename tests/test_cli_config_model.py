#!/usr/bin/env python3
"""`run` / `chat` by-name config resolution: a positional that isn't an on-disk
file is looked up in the server config by id/alias, and that model's resolved
path + merged sampling/template/load settings are overlaid onto the CLI args
(explicit flags win). CPU-only - no model is loaded; the resolution and overlay
are pure."""
from __future__ import annotations

from gmlx import chat, cli


def _config(tmp_path, *, body=None, model_path="m-Q4_K_M.gguf"):
    """Write a config with a `coder` profile + model `m` (path under model_dirs),
    and create the model file. Returns (config_path, model_abspath)."""
    gguf = tmp_path / model_path
    gguf.write_bytes(b"GGUF")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(body or (
        f"server:\n  model_dirs: [{tmp_path}]\n"
        "profiles:\n"
        "  coder: {sampling: {temperature: 0.2, top_p: 0.9, max_tokens: 4096},"
        " load: {kv_bits: 8}, system: 'be terse', chat_template: 'TMPL'}\n"
        "models:\n"
        f"  m: {{path: {model_path}, profile: coder}}\n"
        "aliases: {fast: m@coder}\n"))
    return str(cfg), str(gguf)


def _resolve(argv):
    parser = cli._build_parser("gmlx run")
    args = parser.parse_args(argv)
    rc = cli.maybe_load_from_config(args, parser, argv)
    return rc, args


def test_by_id_applies_profile(tmp_path):
    cfg, gguf = _config(tmp_path)
    rc, args = _resolve(["m", "--config", cfg])
    assert rc is None
    assert args.gguf == gguf                          # resolved to the model path
    assert args.temp == 0.2 and args.top_p == 0.9 and args.max_tokens == 4096
    assert args.kv_bits == 8
    assert args.system_prompt == "be terse"
    assert args.chat_template == "TMPL"


def test_explicit_flag_beats_config(tmp_path):
    cfg, _ = _config(tmp_path)
    _, args = _resolve(["m", "--config", cfg, "--temp", "1.5"])
    assert args.temp == 1.5                            # explicit wins
    assert args.top_p == 0.9                           # config still fills the rest


def test_abbreviated_explicit_flag_beats_config(tmp_path):
    # argparse accepts unique prefixes by default; the explicit-dest scan must
    # see them too, or the config overlay clobbers a value the user typed.
    cfg, _ = _config(tmp_path)
    _, args = _resolve(["m", "--config", cfg, "--max-tok", "100"])
    assert args.max_tokens == 100                      # abbreviated, still explicit
    _, args = _resolve(["m", "--config", cfg, "--max-tok=100"])
    assert args.max_tokens == 100                      # --flag=value form too


def test_alias_resolves(tmp_path):
    cfg, gguf = _config(tmp_path)
    _, args = _resolve(["fast", "--config", cfg])
    assert args.gguf == gguf and args.temp == 0.2      # fast -> m@coder


def test_unknown_name_falls_through(tmp_path):
    cfg, _ = _config(tmp_path)
    rc, args = _resolve(["nope", "--config", cfg])
    assert rc is None and args.gguf == "nope"          # unchanged -> file-miss later


def test_pathlike_positional_not_looked_up(tmp_path):
    # A `.gguf` positional is a path, never a config id - even one named like a model.
    cfg, _ = _config(tmp_path)
    rc, args = _resolve(["m.gguf", "--config", cfg])
    assert rc is None and args.gguf == "m.gguf"        # not resolved to model `m`


def test_unknown_profile_errors(tmp_path, capsys):
    cfg, _ = _config(tmp_path)
    rc, _ = _resolve(["m@nope", "--config", cfg])
    assert rc == 2
    assert "unknown profile" in capsys.readouterr().err


def test_missing_config_path_errors(tmp_path, capsys):
    rc, _ = _resolve(["m", "--config", str(tmp_path / "none.yaml")])
    assert rc == 2
    assert "--config not found" in capsys.readouterr().err


def test_hf_cache_model_resolves_to_cache_file(tmp_path, monkeypatch):
    import huggingface_hub
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"GGUF")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: str(cached))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "server:\n  hf_cache: true\n"
        "models:\n  m: {path: hf:org/repo/cached.gguf}\n")
    _, args = _resolve(["m", "--config", str(cfg)])
    assert args.gguf == str(cached)                    # hf: -> local cache file


def test_chat_unknown_name_reports_miss(tmp_path, capsys):
    cfg, _ = _config(tmp_path)
    rc = chat.cmd_chat(["nope", "--config", cfg])      # wiring: same resolver
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_run_dispatches_resolved_path(tmp_path, monkeypatch):
    cfg, gguf = _config(tmp_path)
    seen = {}
    monkeypatch.setattr(cli, "_run_generate",
                        lambda args: seen.__setitem__("gguf", args.gguf) or 0)
    rc = cli.main(["m", "--config", cfg])
    assert rc == 0
    assert seen["gguf"] == gguf                        # resolved path reached dispatch


def _resolve_chat(argv):
    parser = chat._build_parser("gmlx chat")
    args = parser.parse_args(argv)
    rc = cli.maybe_load_from_config(args, parser, argv)
    return rc, args


def _spec_config(tmp_path):
    gguf = tmp_path / "m-Q6_K.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"server:\n  model_dirs: [{tmp_path}]\n"
        "models:\n"
        "  m: {path: m-Q6_K.gguf, speculative: true}\n")
    return str(cfg), str(gguf)


def test_chat_speculative_config_applies(tmp_path):
    # Regression: a config `speculative: true` must reach the *chat* arg surface.
    # It once silently no-op'd because chat's parser lacked --speculative, so the
    # config overlay's hasattr() guard skipped the key (the reported MTP-in-chat bug).
    cfg, gguf = _spec_config(tmp_path)
    rc, args = _resolve_chat(["m", "--config", cfg])
    assert rc is None
    assert args.gguf == gguf
    assert args.speculative is True


def test_run_speculative_config_applies(tmp_path):
    # The same config must keep applying on `run` (unchanged behavior, shared surface).
    cfg, gguf = _spec_config(tmp_path)
    rc, args = _resolve(["m", "--config", cfg])
    assert rc is None and args.gguf == gguf and args.speculative is True


def test_run_chat_share_speculative_surface():
    # Both parsers compose the same speculative group, so they can't drift again.
    need = {"speculative", "no_speculative", "draft_gguf", "draft_block_size"}
    run_dests = {a.dest for a in cli._build_parser()._actions}
    chat_dests = {a.dest for a in chat._build_parser()._actions}
    assert need <= run_dests
    assert need <= chat_dests


def test_chat_speculative_rejects_mmproj(tmp_path, capsys):
    # MTP in chat is the text-only mlx-vlm path; --mmproj (and --adapter/--stream-*)
    # are refused before any model load, mirroring `run`.
    gguf = tmp_path / "m-Q6_K.gguf"
    gguf.write_bytes(b"GGUF")
    rc = chat.cmd_chat(
        [str(gguf), "--speculative", "--mmproj", str(tmp_path / "mm.gguf")]
    )
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "mmproj" in err and "mtp" in err


def test_chat_draft_gguf_implies_speculative(tmp_path):
    # --draft-gguf alone implies the speculative path (same as `run`).
    args = chat._build_parser().parse_args(
        ["m.gguf", "--draft-gguf", str(tmp_path / "d.gguf")]
    )
    assert args.draft_gguf and not args.speculative   # pre-cmd_chat: not yet implied
    # the implication is applied inside cmd_chat; assert the flag wiring is present
    assert hasattr(args, "no_speculative")


# Bare-path family defaults (model-card sampling seeded from the GGUF header),
# --profile intents, and path@intent addressing.
def _seed(argv, monkeypatch, meta):
    """Parse run argv and apply split_path_intent + apply_family_defaults with
    a monkeypatched header read (no real GGUF needed)."""
    import gmlx.discovery as disc
    monkeypatch.setattr(disc, "header_meta", lambda p: meta)
    parser = cli._build_parser("gmlx run")
    args = parser.parse_args(argv)
    cli.split_path_intent(args)
    rc = cli.apply_family_defaults(args, parser, argv)
    return rc, args


_GEMMA = {"arch": "gemma4", "name": "Gemma 4", "kind": "model", "mtp": False}
_QWEN = {"arch": "qwen35", "name": "Qwen3.6", "kind": "model", "mtp": True}
_OSS = {"arch": "gpt-oss", "name": "gpt-oss 20b", "kind": "model", "mtp": False}


def test_bare_path_seeds_family_base(tmp_path, monkeypatch, capsys):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f)], monkeypatch, _GEMMA)
    assert rc is None
    assert (args.temp, args.top_p, args.top_k) == (1.0, 0.95, 64)
    # The banner is deferred so it never trails a failed load; the run/chat
    # paths print it via print_family_note after a successful load.
    assert "[family]" not in capsys.readouterr().out
    cli.print_family_note(args)
    assert "[family] Gemma defaults:" in capsys.readouterr().out


def test_bare_path_explicit_flag_wins(tmp_path, monkeypatch):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    _, args = _seed([str(f), "--temp", "0.3"], monkeypatch, _GEMMA)
    assert args.temp == 0.3                            # explicit wins
    assert args.top_k == 64                            # rest still seeded


def test_bare_path_profile_intent(tmp_path, monkeypatch):
    f = tmp_path / "q.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f), "--profile", "coding"], monkeypatch, _QWEN)
    assert rc is None
    assert args.temp == 0.6 and args.top_p == 0.95 and args.top_k == 20


def test_bare_path_reasoning_intent_sets_template_config(tmp_path, monkeypatch):
    import json
    f = tmp_path / "o.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f), "--profile", "reasoning-high"], monkeypatch, _OSS)
    assert rc is None
    assert (args.temp, args.top_p) == (1.0, 1.0)
    assert json.loads(args.chat_template_config) == {"reasoning_effort": "high"}


def test_intent_without_family_delta_notes_base(tmp_path, monkeypatch, capsys):
    # gemma defines no @creative delta: the base defaults apply, and the banner
    # must say so instead of claiming the intent was applied.
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f), "--profile", "creative"], monkeypatch, _GEMMA)
    assert rc is None
    assert args.temp == 1.0                            # family base still seeded
    out, err = capsys.readouterr()
    assert "no @creative tuning" in err
    assert "@creative defaults" not in out             # banner does not claim it


def test_bare_path_unknown_profile_errors(tmp_path, monkeypatch, capsys):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, _ = _seed([str(f), "--profile", "nope"], monkeypatch, _GEMMA)
    assert rc == 2
    assert "built-in intents" in capsys.readouterr().err


def test_no_family_defaults_keeps_argparse_defaults(tmp_path, monkeypatch):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f), "--no-family-defaults"], monkeypatch, _GEMMA)
    assert rc is None
    assert args.temp == 0.0                            # greedy, as before


def test_no_family_defaults_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([str(f)], monkeypatch, _GEMMA)
    assert rc is None and args.temp == 0.0


def test_no_family_defaults_with_profile_errors(tmp_path, monkeypatch, capsys):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    rc, _ = _seed([str(f), "--profile", "coding", "--no-family-defaults"],
                  monkeypatch, _GEMMA)
    assert rc == 2


def test_config_resolved_skips_family_seeding(tmp_path, monkeypatch):
    f = tmp_path / "g.gguf"
    f.write_bytes(b"x")
    import gmlx.discovery as disc
    monkeypatch.setattr(disc, "header_meta", lambda p: _GEMMA)
    parser = cli._build_parser("gmlx run")
    args = parser.parse_args([str(f)])
    args._config_resolved = True
    assert cli.apply_family_defaults(args, parser, [str(f)]) is None
    assert args.temp == 0.0                            # untouched


def test_path_at_intent_splits(tmp_path, monkeypatch):
    f = tmp_path / "q.gguf"
    f.write_bytes(b"x")
    rc, args = _seed([f"{f}@coding"], monkeypatch, _QWEN)
    assert rc is None
    assert args.gguf == str(f) and args.profile == "coding"
    assert args.temp == 0.6


def test_path_at_intent_ignores_non_intent_suffix(tmp_path, monkeypatch):
    f = tmp_path / "q.gguf"
    f.write_bytes(b"x")
    _, args = _seed([f"{f}@main", "--no-family-defaults"], monkeypatch, _QWEN)
    assert args.gguf == f"{f}@main"                    # left for the file-miss error


def test_path_at_intent_prefers_whole_filename(tmp_path, monkeypatch):
    f = tmp_path / "model@coding"                      # an actual @ in the name
    f.write_bytes(b"x")
    rc, args = _seed([str(f)], monkeypatch, _QWEN)
    assert args.gguf == str(f) and args.profile is None


def test_config_profile_flag_overrides_model_profile(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"server: {{model_dirs: [{tmp_path}]}}\n"
        "profiles:\n"
        "  coder: {sampling: {temperature: 0.2}}\n"
        "  hot:   {sampling: {temperature: 1.9}}\n"
        "models:\n"
        "  m: {path: m.gguf, profile: coder}\n")
    rc, args = _resolve(["m", "--config", str(cfg), "--profile", "hot"])
    assert rc is None and args.temp == 1.9             # --profile replaces coder


def test_config_model_without_profile_gets_family_base(tmp_path):
    """The fake GGUF is unreadable as a header -> family None -> generic base
    (t=0.7/top_p=0.95) seeds the overlay, mirroring the server's lowest layer."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"server: {{model_dirs: [{tmp_path}]}}\n"
        "models:\n"
        "  m: {path: m.gguf}\n")
    rc, args = _resolve(["m", "--config", str(cfg)])
    assert rc is None
    assert (args.temp, args.top_p) == (0.7, 0.95)


def test_chat_parser_has_family_flags():
    args = chat._build_parser("gmlx chat").parse_args(
        ["x.gguf", "--profile", "coding", "--no-family-defaults"])
    assert args.profile == "coding" and args.no_family_defaults
