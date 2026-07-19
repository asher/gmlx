#!/usr/bin/env python3
"""Friendly-id resolution layer in server_bridge_vlm.py: register_resolved_models fills the
id tables + path-keyed companions, resolve_request_model maps id / id@profile /
profile-field to a concrete path + merged spec, and unknown ids/profiles raise
clean errors (never an HF fetch). CPU-only - no model load, absolute paths so
resolve_path passes through unchecked."""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")
pytest.importorskip("mlx_vlm")

from gmlx import server_bridge_vlm as serving  # noqa: E402
from gmlx.config import build_config  # noqa: E402


def _doc():
    return {
        "server": {"model_dirs": ["/models"], "defaults": {"profile": "base"}},
        "profiles": {
            "base": {"sampling": {"temperature": 0.7}},
            "coder": {"extends": "base", "sampling": {"temperature": 0.2},
                      "load": {"kv_bits": 8}},
        },
        "models": {
            "qwen": {"path": "/abs/qwen.gguf", "profile": "base"},
            "qwen-mtp": {"path": "/abs/qwen-mtp.gguf", "speculative": True},
            "gemma-vlm": {"path": "/abs/gemma.gguf",
                          "mmproj": "/abs/mmproj.gguf"},
        },
    }


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    serving.clear_resolved_models()
    serving._GGUF_VLM_REGISTRY.clear()
    serving._GGUF_MTP_REGISTRY.clear()
    yield
    serving.clear_resolved_models()
    serving._GGUF_VLM_REGISTRY.clear()
    serving._GGUF_MTP_REGISTRY.clear()


@pytest.fixture
def registered():
    cfg = build_config(_doc())
    serving.register_resolved_models(cfg)
    return cfg


# registration fills the id tables + path-keyed companions
def test_register_fills_id_tables(registered):
    assert set(serving.resolved_models()) == {"qwen", "qwen-mtp", "gemma-vlm"}
    assert serving._PATH_TO_IDS["/abs/qwen.gguf"] == ["qwen"]


def test_register_wires_vlm_companion_by_path(registered):
    spec = serving._resolve_vlm_spec("/abs/gemma.gguf")
    assert spec is not None
    assert spec["mmproj_path"] == "/abs/mmproj.gguf"


def test_register_wires_mtp_companion_by_path(registered):
    spec = serving._resolve_mtp_spec("/abs/qwen-mtp.gguf")
    assert spec is not None                       # marked speculative


def test_non_speculative_model_has_no_mtp_spec(registered):
    assert serving._resolve_mtp_spec("/abs/qwen.gguf") is None


# resolve_request_model - plain id
def test_resolve_plain_id(registered):
    path, rm = serving.resolve_request_model("qwen")
    assert path == "/abs/qwen.gguf"
    assert rm.sampling["temperature"] == 0.7      # base profile


def test_resolve_inline_profile_overrides(registered):
    path, rm = serving.resolve_request_model("qwen@coder")
    assert path == "/abs/qwen.gguf"
    assert rm.sampling["temperature"] == 0.2      # coder via @profile
    assert rm.load["kv_bits"] == 8                # coder load params


def test_resolve_profile_field(registered):
    _path, rm = serving.resolve_request_model("qwen", profile_field="coder")
    assert rm.sampling["temperature"] == 0.2


def test_inline_profile_beats_profile_field(registered):
    _path, rm = serving.resolve_request_model("qwen@coder", profile_field="base")
    assert rm.sampling["temperature"] == 0.2      # inline @coder wins


# addressing robustness
def test_split_last_at_only_when_profile_known(registered):
    # right side is a known profile -> split
    assert serving.split_profile_address("qwen@coder") == ("qwen", "coder")
    # right side not a profile -> whole string is the id (hf org/model@rev safe)
    assert serving.split_profile_address("org/model@abc123") == (
        "org/model@abc123", None)


def test_unknown_model_raises_modelnotfound(registered):
    with pytest.raises(serving.ModelNotFound) as e:
        serving.resolve_request_model("does-not-exist")
    assert "qwen" in e.value.available            # 404 body can list options


def test_at_unknown_profile_on_known_id_raises_unknownprofile(registered):
    # "qwen@ghost": qwen IS a model, ghost is not a profile -> the helpful error
    # is "unknown profile", not "unknown model id 'qwen@ghost'".
    with pytest.raises(serving.UnknownProfile) as e:
        serving.resolve_request_model("qwen@ghost")
    assert "coder" in e.value.available


def test_at_on_unknown_head_stays_modelnotfound(registered):
    # "org/model@rev": head isn't a configured id, so the last-@ rule keeps the
    # whole string intact (hf revision safety) -> unknown id, not unknown profile.
    with pytest.raises(serving.ModelNotFound):
        serving.resolve_request_model("org/model@rev")


def test_unknown_profile_field_raises(registered):
    with pytest.raises(serving.UnknownProfile) as e:
        serving.resolve_request_model("qwen", profile_field="ghost")
    assert "coder" in e.value.available


# default-model resolution for an empty field
def test_empty_field_uses_defaults_model():
    doc = _doc()
    doc["server"]["defaults"]["model"] = "qwen"
    serving.register_resolved_models(build_config(doc))
    path, _rm = serving.resolve_request_model("")
    assert path == "/abs/qwen.gguf"
    path2, _ = serving.resolve_request_model(None)    # missing field
    assert path2 == "/abs/qwen.gguf"


def test_empty_field_uses_sole_model():
    serving.register_resolved_models(build_config(
        {"models": {"only": {"path": "/abs/only.gguf"}}}))
    path, _rm = serving.resolve_request_model("")
    assert path == "/abs/only.gguf"


def test_empty_field_ambiguous_raises(registered):
    with pytest.raises(serving.NoModelSpecified):
        serving.resolve_request_model("")             # 3 models, no default


def test_defaults_model_unknown_fails_at_load():
    # An unknown server.defaults.model is now caught fail-fast at config build,
    # not lazily at request time.
    from gmlx.config import ConfigError
    doc = _doc()
    doc["server"]["defaults"]["model"] = "ghost"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "ghost" in str(e.value)


# aliases: friendly name / profile preset resolution
def _alias_doc():
    doc = _doc()
    doc["aliases"] = {
        "big": "qwen",                  # bare rename
        "coder-preset": "qwen@coder",   # profile preset
    }
    return doc


def test_alias_bare_rename_resolves_like_target():
    serving.register_resolved_models(build_config(_alias_doc()))
    path, rm = serving.resolve_request_model("big")
    assert path == "/abs/qwen.gguf"
    # "big" == qwen, so it inherits qwen's own profile resolution (base default)
    assert rm.sampling["temperature"] == 0.7


def test_alias_profile_preset_applies_profile():
    serving.register_resolved_models(build_config(_alias_doc()))
    path, rm = serving.resolve_request_model("coder-preset")
    assert path == "/abs/qwen.gguf"
    assert rm.sampling["temperature"] == 0.2        # coder profile baked into alias


def test_alias_inline_profile_overrides_baked():
    serving.register_resolved_models(build_config(_alias_doc()))
    _path, rm = serving.resolve_request_model("coder-preset@base")
    assert rm.sampling["temperature"] == 0.7        # explicit @base beats baked coder


def test_alias_unknown_inline_profile_raises():
    serving.register_resolved_models(build_config(_alias_doc()))
    with pytest.raises(serving.UnknownProfile):
        serving.resolve_request_model("coder-preset@ghost")


def test_aliases_accessor_maps_to_target_and_profile():
    serving.register_resolved_models(build_config(_alias_doc()))
    a = serving.aliases()
    assert a["big"] == ("qwen", None)
    assert a["coder-preset"] == ("qwen", "coder")


# default_model_id - the soft /v1/models + launch default hint
def test_default_model_id_prefers_configured_default():
    doc = _doc()
    doc["server"]["defaults"]["model"] = "qwen"
    serving.register_resolved_models(build_config(doc))
    assert serving.default_model_id() == "qwen"


def test_default_model_id_lone_pinned():
    doc = _doc()
    doc["models"]["qwen-mtp"]["pin"] = True
    serving.register_resolved_models(build_config(doc))
    assert serving.default_model_id() == "qwen-mtp"


def test_default_model_id_sole_model():
    serving.register_resolved_models(build_config(
        {"models": {"only": {"path": "/abs/only.gguf"}}}))
    assert serving.default_model_id() == "only"


def test_default_model_id_ambiguous_is_none():
    serving.register_resolved_models(build_config(_doc()))   # 3 models, none pinned
    assert serving.default_model_id() is None


# active-spec ContextVar discipline
def test_active_spec_set_get_reset(registered):
    assert serving.get_active_spec() is None
    _path, rm = serving.resolve_request_model("qwen")
    tok = serving.set_active_spec(rm)
    assert serving.get_active_spec() is rm
    serving.reset_active_spec(tok)
    assert serving.get_active_spec() is None


def test_resolve_without_config_raises():
    serving.clear_resolved_models()
    with pytest.raises(RuntimeError):
        serving.resolve_request_model("qwen")


# reload coherence: re-registering resets the path-keyed companion registries
# (a /v1/reload that removes/adds an mmproj or drafter must take effect on the
# next cold load) and drops the drafter stash.
def test_reload_drops_removed_vlm_companion(registered):
    assert serving._resolve_vlm_spec("/abs/gemma.gguf") is not None
    doc = _doc()
    del doc["models"]["gemma-vlm"]["mmproj"]            # config edit: mmproj gone
    serving.register_resolved_models(build_config(doc))  # what /v1/reload runs
    assert serving._resolve_vlm_spec("/abs/gemma.gguf") is None


def test_reload_adds_new_vlm_companion(registered):
    assert serving._resolve_vlm_spec("/abs/qwen.gguf") is None
    doc = _doc()
    doc["models"]["qwen"]["mmproj"] = "/abs/qwen-mmproj.gguf"
    serving.register_resolved_models(build_config(doc))
    spec = serving._resolve_vlm_spec("/abs/qwen.gguf")
    assert spec["mmproj_path"] == "/abs/qwen-mmproj.gguf"


def test_reload_drops_removed_mtp_companion_and_stash(registered):
    assert serving._resolve_mtp_spec("/abs/qwen-mtp.gguf") is not None
    serving._MTP_DRAFTER_STASH["/abs/qwen-mtp.gguf"] = (object(), "mtp")
    doc = _doc()
    doc["models"]["qwen-mtp"]["speculative"] = False
    serving.register_resolved_models(build_config(doc))
    assert serving._resolve_mtp_spec("/abs/qwen-mtp.gguf") is None
    assert serving._MTP_DRAFTER_STASH == {}             # stash never survives reload


def test_reload_then_serve_speculative_still_resolves(registered):
    # The promise behind the stash-clearing: after a reload (and an eviction)
    # of a still-speculative model, speculative serving resolves a FRESH spec
    # and no stale drafter lingers in the stash.
    assert serving._resolve_mtp_spec("/abs/qwen-mtp.gguf") is not None
    serving._MTP_DRAFTER_STASH["/abs/qwen-mtp.gguf"] = (object(), "mtp")
    serving.register_resolved_models(build_config(_doc()))  # /v1/reload, model kept
    assert serving._MTP_DRAFTER_STASH == {}                 # no leaked drafter
    spec = serving._resolve_mtp_spec("/abs/qwen-mtp.gguf")
    assert spec == {"draft_gguf_path": None}                # still speculative
    # Residency eviction (drop_mtp_stash) must not unregister the model either.
    serving._MTP_DRAFTER_STASH["/abs/qwen-mtp.gguf"] = (object(), "mtp")
    serving.drop_mtp_stash("/abs/qwen-mtp.gguf")
    assert serving._MTP_DRAFTER_STASH == {}
    assert serving._resolve_mtp_spec("/abs/qwen-mtp.gguf") == {
        "draft_gguf_path": None}


def test_reload_adds_mtp_companion(registered):
    assert serving._resolve_mtp_spec("/abs/qwen.gguf") is None
    doc = _doc()
    doc["models"]["qwen"]["speculative"] = True
    serving.register_resolved_models(build_config(doc))
    assert serving._resolve_mtp_spec("/abs/qwen.gguf") is not None


# _fill_families - registration-time family detection from GGUF headers.
def _fam_doc():
    return {
        "server": {"model_dirs": ["/models"]},
        "models": {
            "auto": {"path": "/abs/auto.gguf"},
            "pinned": {"path": "/abs/pinned.gguf", "family": "glm"},
        },
    }


def test_register_fills_family_from_header(monkeypatch):
    import gmlx.discovery as disc
    monkeypatch.setattr(
        disc, "header_meta",
        lambda p: {"arch": "gemma4", "name": None, "kind": "model", "mtp": False})
    cfg = build_config(_fam_doc())
    serving.register_resolved_models(cfg)
    try:
        assert cfg.models["auto"].family == "gemma"
        assert serving.resolved_models()["auto"].family == "gemma"
        # Explicit YAML family: wins over detection.
        assert cfg.models["pinned"].family == "glm"
        _path, rm = serving.resolve_request_model("auto@coding")
        assert rm.sampling["temperature"] == 1.0    # gemma coding == base
    finally:
        serving.clear_resolved_models()


def test_register_missing_file_keeps_family_none(monkeypatch, capsys):
    import gmlx.discovery as disc
    monkeypatch.setattr(disc, "header_meta", lambda p: None)
    cfg = build_config(_fam_doc())
    serving.register_resolved_models(cfg)
    try:
        assert cfg.models["auto"].family is None
        # Family detection stays silent; the only permitted stderr is the
        # model_dirs root-missing warning (the doc's /models doesn't exist).
        err = capsys.readouterr().err
        assert "family" not in err.lower()
        assert all(ln.startswith("[server] model_dirs root missing")
                   for ln in err.splitlines() if ln)
    finally:
        serving.clear_resolved_models()


def test_register_kill_switch_skips_detection(monkeypatch):
    import gmlx.discovery as disc

    def _boom(p):
        raise AssertionError("header read attempted with family_defaults off")

    monkeypatch.setattr(disc, "header_meta", _boom)
    doc = _fam_doc()
    doc["server"]["family_defaults"] = False
    doc["models"]["pinned"].pop("family")           # keep names resolvable
    cfg = build_config(doc)
    serving.register_resolved_models(cfg)
    serving.clear_resolved_models()


def test_unknown_profile_error_lists_builtin_intents():
    cfg = build_config(_fam_doc())
    serving.register_resolved_models(cfg)
    try:
        with pytest.raises(serving.UnknownProfile) as e:
            serving.resolve_request_model("auto@nope")
        assert "coding" in str(e.value)
    finally:
        serving.clear_resolved_models()


# warn-and-skip: a missing file must never take registration (= startup AND
# config reload) down; the entry hides from /v1/models and self-heals.
def _ghost_doc(tmp_path):
    (tmp_path / "real.gguf").write_bytes(b"GGUF")
    return {
        "server": {"model_dirs": [str(tmp_path)]},
        "models": {"real": {"path": "real.gguf"},
                   "ghost": {"path": "gone.gguf"}},
    }


def test_register_skips_models_missing_on_disk(tmp_path, capsys):
    cfg = build_config(_ghost_doc(tmp_path))
    serving.register_resolved_models(cfg)          # must not raise
    assert set(serving.resolved_models()) == {"real"}
    err = capsys.readouterr().err
    assert "ghost" in err and "sync-models" in err


def test_request_for_skipped_model_raises_typed_error(tmp_path):
    cfg = build_config(_ghost_doc(tmp_path))
    serving.register_resolved_models(cfg)
    with pytest.raises(serving.ModelFileMissing) as ei:
        serving.resolve_request_model("ghost")
    assert "sync-models" in str(ei.value)
    path, _rm = serving.resolve_request_model("real")   # sibling unaffected
    assert path.endswith("real.gguf")


def test_skipped_model_self_heals_when_file_returns(tmp_path):
    cfg = build_config(_ghost_doc(tmp_path))
    serving.register_resolved_models(cfg)
    (tmp_path / "gone.gguf").write_bytes(b"GGUF")
    path, _rm = serving.resolve_request_model("ghost")  # per-request re-resolve
    assert path.endswith("gone.gguf")


def test_reregister_heals_listing_when_file_returns(tmp_path, capsys):
    # The request path self-heals, but /v1/models reads the registration table;
    # reregister_missing_models is what folds a restored file back into it.
    cfg = build_config(_ghost_doc(tmp_path))
    serving.register_resolved_models(cfg)
    assert set(serving.resolved_models()) == {"real"}
    assert serving.reregister_missing_models() is False   # still gone: no-op
    (tmp_path / "gone.gguf").write_bytes(b"GGUF")
    assert serving.reregister_missing_models() is True
    assert set(serving.resolved_models()) == {"real", "ghost"}
    assert "re-registered" in capsys.readouterr().err
    assert serving.reregister_missing_models() is False   # idempotent
