#!/usr/bin/env python3
"""Tests for the interactive `gmlx init` wizard, its flag mirrors, and the
extras install/detection helper. CPU-only; no model, no server, no real pip -
the scan is faked and pip is mocked, so nothing touches the network or the venv.

The wizard funnels all input through ``WizardIO._read``; ``_ScriptIO`` overrides
it to replay a fixed answer list, driving the flow headlessly (the same pattern
the chat REPL uses for its loop tests)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from gmlx import config, discovery, extras, server, wizard
from gmlx.config import ModelCfg

yaml = pytest.importorskip("yaml")


class _ScriptIO(wizard.WizardIO):
    """A WizardIO that replays a scripted list of answers and swallows output."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def _read(self, prompt: str) -> str:
        return self.answers.pop(0) if self.answers else ""

    def note(self, msg: str = "") -> None:  # keep test output quiet
        pass


def _fake_scan(models):
    return lambda specs, dirs, **kw: list(models)


# Wizard end-to-end
def test_wizard_curation_and_knobs(monkeypatch, tmp_path):
    """A full scripted run: curate ids (rename / default / alias), enable the
    disk cache, never-evict TTL, a 30m request timeout - output must parse."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([
        ModelCfg(id="qwen3-0.6b", path="/m/qwen3-0.6b-Q4_K_M.gguf"),
        ModelCfg(id="gemma-e4b", path="/m/gemma-4-E4B-it-Q6_K.gguf"),
    ]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([], []))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "/m",            # scan dir
        "",              # recurse -> default yes
        "rename 1 qwen",  # curation
        "default 1",
        "alias fast 2",
        "done",
        "y",             # enable on-disk prompt cache
        "100",           # max on-disk cache size (GB)
        "n", "n",        # decline stt / tts
        "n",             # decline embeddings
        "n",             # decline rerank
        "5",             # idle TTL -> never (0)
        "2",             # request timeout -> 30m (1800)
        "1",             # output -> user config
        "",              # write? -> default yes
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert outcome is not None and outcome.out == out
    assert {m.id for m in outcome.models} == {"qwen", "gemma-e4b"}

    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.defaults.model == "qwen"          # rename propagated to default
    assert cfg.aliases == {"fast": "gemma-e4b"}
    assert cfg.defaults.ttl_s == 0               # never-evict sentinel
    assert cfg.token_queue_timeout_s == 1800
    assert cfg.cache.get("enabled") and cfg.cache.get("disk")
    assert cfg.cache["disk"]["max_gb"] == 100   # the follow-up size propagated


def test_wizard_drop_clears_default_and_aliases(monkeypatch, tmp_path):
    """Dropping the model that holds the default and an alias must not leave
    dangling references - the written config still builds."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([
        ModelCfg(id="qwen3-0.6b", path="/m/qwen3-0.6b-Q4_K_M.gguf"),
        ModelCfg(id="gemma-e4b", path="/m/gemma-4-E4B-it-Q6_K.gguf"),
    ]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([], []))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "/m", "",        # scan dir, recurse
        "default 1",     # curation: default + alias on row 1...
        "alias fast 1",
        "drop 1",        # ...then drop it
        "done",
        "n",             # disk cache
        "n", "n",        # decline stt / tts
        "n", "n",        # decline embeddings / rerank
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert {m.id for m in outcome.models} == {"gemma-e4b"}
    cfg = config.build_config(yaml.safe_load(outcome.text))  # must not reject
    assert cfg.defaults.model is None            # dropped -> default cleared
    assert cfg.aliases == {}                     # dropped -> alias gone


def test_wizard_hf_cache_adoption(monkeypatch, tmp_path):
    """Answering yes to the hf-cache prompt adopts cache entries (deduped by
    known_ids) and writes server.hf_cache: true."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([
        ModelCfg(id="qwen3-0.6b", path="/m/qwen3-0.6b-Q4_K_M.gguf"),
    ]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: True)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([], []))
    seen_ids = {}

    def fake_hf_scan(*, known_ids=frozenset(), **kw):
        seen_ids["ids"] = set(known_ids)
        # scan_hf_cache dedupes against known_ids; a disk-adopted id yields
        # only the genuinely new entry
        return [ModelCfg(id="gemma-e4b", path="hf:org/repo/gemma-e4b.gguf")]

    monkeypatch.setattr(discovery, "scan_hf_cache", fake_hf_scan)

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "/m", "",        # scan dir, recurse
        "y",             # include hf cache
        "",              # curation done
        "n",             # disk cache
        "n", "n",        # decline stt / tts
        "n", "n",        # decline embeddings / rerank
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert seen_ids["ids"] == {"qwen3-0.6b"}     # dedupe fed the disk ids
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.hf_cache is True
    assert set(cfg.models) == {"qwen3-0.6b", "gemma-e4b"}
    assert cfg.models["gemma-e4b"].path.startswith("hf:")


def test_wizard_port_override(monkeypatch, tmp_path):
    """run_wizard(port=...) (the -i --port seed) overrides the scaffold's
    server.port."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)

    io = _ScriptIO(["", "", "n", "n", "n", "n", "n", "", "", "", ""])
    outcome = wizard.run_wizard(
        default_out=str(tmp_path / "cfg.yaml"), io=io, port=9317)

    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.port == 9317


def test_wizard_offers_install_for_missing_extra(monkeypatch, tmp_path):
    """Configuring a service whose extra is absent offers the install and, on
    yes, calls install_extra exactly once for that extra."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(extras, "extra_installed", lambda e: False)
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)
    installed = []
    monkeypatch.setattr(extras, "install_extra",
                        lambda e, **k: (installed.append(e) or True))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "",              # no scan dir
        "",              # recurse
        # no models -> curation skipped
        "n",             # disk cache
        "y", "", "y",    # configure stt, default model, install yes
        "n",             # decline tts
        "n",             # decline embeddings
        "n",             # decline rerank
        "", "", "",      # ttl / timeout / output -> defaults
        "",              # write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert installed == ["stt"]
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.stt == wizard.stt.DEFAULT_STT_ALIAS


def test_wizard_allow_install_false_never_installs(monkeypatch, tmp_path):
    """allow_install=False (the --no-install mirror) configures the service but
    never offers to install - install_extra is not called."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(extras, "extra_installed", lambda e: False)
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)
    called = []
    monkeypatch.setattr(extras, "install_extra",
                        lambda e, **k: (called.append(e) or True))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "", "",          # dir, recurse
        "n",             # disk cache
        "y", "",         # configure stt, default model (no install prompt)
        "n",             # decline tts
        "n",             # decline embeddings
        "n",             # decline rerank
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(
        default_out=str(out), io=io, allow_install=False)

    assert called == []
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.stt == wizard.stt.DEFAULT_STT_ALIAS


def test_wizard_embeddings_preset_quant_and_rerank_inherits(monkeypatch, tmp_path):
    """Pick a GGUF embedder preset + a non-default quant, then a reranker whose
    quant defaults to the embedder's chosen rung - the written config carries both
    concrete hf: refs at that rung."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "", "",          # no scan dir, recurse
        "n",             # disk cache
        "n", "n",        # decline stt / tts
        "y",             # configure embeddings
        "2",             # pick qwen3-embed-4b (2nd preset row)
        "3",             # quant -> Q6_K (3rd rung)
        "y",             # configure reranking
        "1",             # reranker size -> qwen3-rerank-0.6b
        "",              # reranker quant -> default == embedder's Q6_K
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.embeddings == wizard.embeddings._qwen_emb_ref("4B", "Q6_K")
    # the reranker inherited the embedder's Q6_K rung (not its own Q8_0 default)
    assert cfg.rerank == wizard.rerank._qwen_rerank_ref("0.6B", "Q6_K")


def test_wizard_adopts_found_retrieval_gguf(monkeypatch, tmp_path):
    """When the scan already found an embedder / reranker GGUF on disk, the wizard
    offers to adopt each (default yes) and writes their paths - no preset picker."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    emb = discovery.ClassifiedGguf(
        "/m/Qwen3-Embedding-0.6B-Q8_0.gguf", "embedding", "qwen3", False, "Q8_0", False)
    rr = discovery.ClassifiedGguf(
        "/m/Qwen3-Reranker-0.6B.Q8_0.gguf", "reranker", "qwen3", False, "Q8_0", False)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([emb], [rr]))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "/m", "",        # scan dir, recurse
        "n",             # disk cache
        "n", "n",        # decline stt / tts
        "",              # adopt the found embedder (default yes)
        "",              # adopt the found reranker (default yes)
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.embeddings == emb.path
    assert cfg.rerank == rr.path


def test_wizard_declined_final_write_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    # dir, recurse, disk-cache, stt, tts, embeddings, rerank, ttl, timeout, out, write->no
    io = _ScriptIO(["", "", "n", "n", "n", "n", "n", "", "", "", "n"])
    assert wizard.run_wizard(default_out=str(tmp_path / "c.yaml"), io=io) is None


# Profiles step (3.5): family summary + optional pinned intent
def _profiles_wizard(monkeypatch, tmp_path, answers):
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([
        ModelCfg(id="qw", path="/m/qw.gguf", family="qwen3.6"),
        ModelCfg(id="gm", path="/m/gm.gguf", family="gemma"),
    ]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([], []))
    io = _ScriptIO(answers)
    return wizard.run_wizard(default_out=str(tmp_path / "cfg.yaml"), io=io)


def test_wizard_profiles_step_pins_intent(monkeypatch, tmp_path):
    """Answering yes to the pin prompt asks once per family WITH intents
    (qwen3.6 here - gemma has none) and writes the pick as the model's
    `profile:`; the scaffold output stays valid."""
    outcome = _profiles_wizard(monkeypatch, tmp_path, [
        "/m", "",        # scan dir, recurse
        "",              # curation done
        "y",             # pin a default intent?
        "2",             # qwen3.6 choice: 1=family default, 2=coding, 3=instruct
        "n",             # disk cache
        "n", "n", "n", "n",   # stt / tts / embeddings / rerank
        "", "", "", "",  # ttl, timeout, output, write
    ])
    assert outcome is not None
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.models["qw"].profile == "coding"      # pinned
    assert cfg.models["gm"].profile is None          # no intents -> no prompt
    assert "# qwen3.6: t=1.0" in outcome.text        # family comment rendered


def test_wizard_profiles_step_enter_through(monkeypatch, tmp_path):
    """Enter at the pin prompt (default no) skips straight on - exactly one
    extra answer consumed, nothing pinned."""
    outcome = _profiles_wizard(monkeypatch, tmp_path, [
        "/m", "",        # scan dir, recurse
        "",              # curation done
        "",              # pin? -> default no (no per-family choice follows)
        "n",             # disk cache
        "n", "n", "n", "n",   # stt / tts / embeddings / rerank
        "", "", "", "",  # ttl, timeout, output, write
    ])
    assert outcome is not None
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert all(m.profile is None for m in cfg.models.values())


def test_wizard_no_eligible_family_no_prompt(monkeypatch, tmp_path):
    """Family-less models (or families without intents) get the summary notes
    only - no prompt is added, so the historic answer script still fits."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([
        ModelCfg(id="plain", path="/m/plain.gguf"),
    ]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(discovery, "find_retrieval_models",
                        lambda dirs, **kw: ([], []))
    io = _ScriptIO([
        "/m", "",        # scan dir, recurse
        "",              # curation done
        "n",             # disk cache (no pin prompt in between)
        "n", "n", "n", "n",
        "", "", "", "",
    ])
    outcome = wizard.run_wizard(default_out=str(tmp_path / "cfg.yaml"), io=io)
    assert outcome is not None
    assert config.build_config(yaml.safe_load(outcome.text)).models


# Non-interactive flag path
def test_flag_path_unchanged_without_new_flags(monkeypatch, tmp_path):
    """Bare flag-driven init (no new flags) still writes today's shape: ttl 900,
    services left as commented hints."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    out = tmp_path / "cfg.yaml"
    rc = server._cmd_init(["--models-dir", str(tmp_path), "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    cfg = config.load_config(out)
    assert cfg.defaults.ttl_s == 900
    assert cfg.stt is None and cfg.tts is None and cfg.embeddings is None
    assert "# stt: whisper-turbo" in text       # hint, commented


def test_flag_path_mirrors_wizard_knobs(monkeypatch, tmp_path):
    """--with-* / --idle-ttl / --request-timeout write the same keys the wizard
    would, and --install routes only to extras that need it (not a GGUF embedder)."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(extras, "extra_installed", lambda e: False)
    installed = []
    monkeypatch.setattr(extras, "install_extra",
                        lambda e, **k: (installed.append(e) or True))

    out = tmp_path / "cfg.yaml"
    rc = server._cmd_init([
        "--models-dir", str(tmp_path), "--out", str(out),
        "--with-stt", "--with-tts", "kokoro-4bit",
        "--with-embeddings", "/m/Qwen3-Embedding-4B.Q6_K.gguf",  # GGUF -> no extra
        "--idle-ttl", "none", "--request-timeout", "1h", "--install",
    ])
    assert rc == 0
    cfg = config.load_config(out)
    assert cfg.stt == "whisper-turbo" and cfg.tts == "kokoro-4bit"
    assert cfg.embeddings.endswith(".gguf")
    assert cfg.defaults.ttl_s == 0 and cfg.token_queue_timeout_s == 3600
    assert sorted(installed) == ["stt", "tts"]    # GGUF embeddings needs no extra


def test_flag_path_bad_duration_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    with pytest.raises(SystemExit):
        server._cmd_init(["--models-dir", str(tmp_path),
                          "--out", str(tmp_path / "c.yaml"),
                          "--idle-ttl", "banana"])


# Dispatch predicates
def _ns(**kw):
    base = dict(models_dir=None, from_hf_cache=False, out=None, disk_cache=None,
                recursive=False, force=False, with_stt=None, with_tts=None,
                with_embeddings=None, with_rerank=None, default_model=None,
                port=None, idle_ttl=None, request_timeout=None, install=False,
                no_install=False, interactive=False, no_interactive=False)
    return SimpleNamespace(**{**base, **kw})


def test_scaffold_intent_predicate():
    assert server._has_scaffold_intent(_ns()) is False
    assert server._has_scaffold_intent(_ns(models_dir=["/x"])) is True
    assert server._has_scaffold_intent(_ns(with_stt=server._SVC_DEFAULT)) is True
    assert server._has_scaffold_intent(_ns(with_rerank=server._SVC_DEFAULT)) is True
    assert server._has_scaffold_intent(_ns(idle_ttl="15m")) is True
    assert server._has_scaffold_intent(_ns(port=9090)) is True


def test_want_interactive():
    assert server._want_interactive(_ns(interactive=True)) is True
    assert server._want_interactive(
        _ns(interactive=True, no_interactive=True)) is False
    # any scaffolding flag opts out of the auto-wizard regardless of tty
    assert server._want_interactive(_ns(models_dir=["/x"])) is False


# extras helper
def test_extras_packages_mirror_pyproject():
    assert extras.extra_packages("stt") == ["mlx-whisper", "python-multipart"]
    # vlm + embeddings are core deps now - empty back-compat extras.
    assert extras.extra_packages("embeddings") == []
    assert extras.extra_packages("vlm") == []
    with pytest.raises(KeyError):
        extras.extra_packages("nope")


def test_install_extra_uses_runner():
    seen = {}

    def runner(cmd):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    assert extras.install_extra("stt", runner=runner) is True
    assert seen["cmd"][:4] == [sys.executable, "-m", "pip", "install"]
    assert "mlx-whisper" in seen["cmd"]


def test_install_extra_reports_failure():
    runner = lambda cmd: SimpleNamespace(returncode=1)  # noqa: E731
    assert extras.install_extra("tts", runner=runner) is False


@pytest.mark.parametrize("text,secs", [
    ("none", 0), ("never", 0), ("0", 0), ("900", 900),
    ("15m", 900), ("1h", 3600), ("2.5m", 150),
])
def test_parse_duration(text, secs):
    assert wizard.parse_duration(text) == secs


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        wizard.parse_duration("banana")


# Talk step (voice chat): offered only when BOTH audio services configured
def test_wizard_talk_step_after_both_audio_services(monkeypatch, tmp_path):
    """Configuring stt AND tts triggers the talk step; the picks land in a
    top-level talk: block that parses into TalkCfg."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(extras, "extra_installed", lambda e: True)
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)
    monkeypatch.setattr(wizard.tts, "available_voices",
                        lambda m: ["af_heart", "am_adam", "bf_emma"])

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "", "",          # no scan dir, recurse
        "n",             # disk cache
        "y", "",         # configure stt, default model
        "y", "",         # configure tts, default model
        "n",             # decline embeddings
        "n",             # decline rerank
        "",              # set up voice chat? -> default yes
        "",              # voice -> default (af_heart, present in the list)
        "computer",      # wake phrase
        "2",             # mode -> vad (open mic)
        "2",             # hotkey modifier -> right-command
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.talk.voice == "af_heart"
    assert cfg.talk.wake_word == "computer"
    assert cfg.talk.mode == "vad"
    assert cfg.talk.push_to_talk_modifier == "right-command"


def test_wizard_talk_offers_install_when_extra_missing(monkeypatch, tmp_path):
    """A missing [talk] extra gets the install offer; yes routes to
    install_extra('talk') (stt/tts installs also fire - extras all 'missing')."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(extras, "extra_installed", lambda e: False)
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)
    monkeypatch.setattr(wizard.tts, "available_voices", lambda m: [])
    installed = []
    monkeypatch.setattr(extras, "install_extra",
                        lambda e, **k: (installed.append(e) or True))

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "", "",          # no scan dir, recurse
        "n",             # disk cache
        "y", "", "n",    # configure stt, default model, decline install
        "y", "", "n",    # configure tts, default model, decline install
        "n",             # decline embeddings
        "n",             # decline rerank
        "y",             # set up voice chat
        "",              # voice -> default af_heart (no enumeration)
        "",              # wake phrase -> default "hey assistant"
        "",              # mode -> default wake
        "",              # hotkey modifier -> default globe
        "y",             # install the [talk] extra
        "", "", "", "",  # ttl / timeout / output / write
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert installed == ["talk"]
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.talk.voice == "af_heart"
    assert cfg.talk.wake_word == "hey assistant"
    assert cfg.talk.mode == "wake"
    assert cfg.talk.push_to_talk_modifier == "globe"


def test_wizard_talk_skipped_without_both_services(monkeypatch, tmp_path):
    """STT alone (no TTS) never shows the talk step - the historic answer
    script still fits and the scaffold keeps the commented hint."""
    monkeypatch.setattr(discovery, "scan_dirs", _fake_scan([]))
    monkeypatch.setattr(wizard, "_hf_cache_has_gguf", lambda: False)
    monkeypatch.setattr(extras, "extra_installed", lambda e: True)
    monkeypatch.setattr(extras, "ffmpeg_present", lambda: True)

    out = tmp_path / "cfg.yaml"
    io = _ScriptIO([
        "", "",          # no scan dir, recurse
        "n",             # disk cache
        "y", "",         # configure stt, default model
        "n",             # decline tts
        "n",             # decline embeddings
        "n",             # decline rerank
        "", "", "", "",  # ttl / timeout / output / write (no talk prompts)
    ])
    outcome = wizard.run_wizard(default_out=str(out), io=io)

    assert "# talk:" in outcome.text
    cfg = config.build_config(yaml.safe_load(outcome.text))
    assert cfg.talk.voice is None                # defaults, block commented out
