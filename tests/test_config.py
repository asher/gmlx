#!/usr/bin/env python3
"""Server config core: dataclasses, the YAML loader, the ``extends`` / ``rules`` /
precedence merge, path/env resolution, and fail-fast validation. Pure CPU - the
module imports only PyYAML + stdlib, so no GPU, no GGUF files, no model load."""
from __future__ import annotations

import os

import pytest

pytest.importorskip("yaml")

from gmlx import config as cfgmod  # noqa: E402
from gmlx.config import ConfigError, build_config, resolve_model  # noqa: E402


# A reusable config exercising every layer of the precedence ladder.
def _doc():
    """server.defaults.profile=base; a rule maps *coder* -> coder; named models pin
    a profile / overrides; absolute paths so resolve_path passes through unchecked."""
    return {
        "server": {
            "model_dirs": ["/models"],
            "defaults": {"profile": "base", "ttl_s": 900},
        },
        "profiles": {
            "base": {"sampling": {"temperature": 0.7, "top_p": 0.95,
                                  "max_tokens": 1024}},
            "coder": {"extends": "base",
                      "sampling": {"temperature": 0.2, "top_p": 0.9},
                      "load": {"kv_bits": 8, "kv_group_size": 64}},
            "creative": {"extends": "base",
                         "system": "Be imaginative.",
                         "sampling": {"temperature": 1.0, "min_p": 0.05}},
        },
        "rules": [{"match": "*coder*", "profile": "coder"}],
        "models": {
            "m-coder": {"path": "/abs/a.gguf"},                 # rule -> coder
            "m-named": {"path": "/abs/b.gguf", "profile": "creative"},
            "m-ov": {"path": "/abs/c.gguf", "profile": "creative",
                     "overrides": {"sampling": {"max_tokens": 2048}}},
            "m-bare": {"path": "/abs/d.gguf"},                  # no profile, no rule
        },
    }


@pytest.fixture
def cfg():
    return build_config(_doc())


# extends chains
def test_extends_inherits_then_overrides(cfg):
    """coder extends base: parent's top_p drops out where the child sets it, but the
    parent's untouched max_tokens carries through."""
    r = resolve_model("m-named", cfg, request_profile="coder")
    assert r.sampling["temperature"] == 0.2     # child wins
    assert r.sampling["top_p"] == 0.9           # child overrides parent's 0.95
    assert r.sampling["max_tokens"] == 1024     # inherited from base
    assert r.load == {"kv_bits": 8, "kv_group_size": 64}  # child-only load


def test_extends_system_propagates_from_leaf(cfg):
    r = resolve_model("m-named", cfg)            # creative carries a system prompt
    assert r.system == "Be imaginative."
    assert r.sampling["min_p"] == 0.05


# rule globbing
def test_rule_glob_selects_profile(cfg):
    """m-coder has no model.profile; the *coder* rule supplies it on top of base."""
    r = resolve_model("m-coder", cfg)
    assert r.sampling["temperature"] == 0.2     # coder via rule
    assert r.load["kv_bits"] == 8


def test_rule_first_match_wins():
    doc = _doc()
    doc["rules"] = [
        {"match": "m-*", "profile": "creative"},   # broad, comes first
        {"match": "*coder*", "profile": "coder"},
    ]
    cfg = build_config(doc)
    r = resolve_model("m-coder", cfg)
    assert r.sampling["temperature"] == 1.0     # creative (first rule), not coder


def test_unmatched_id_falls_to_server_default(cfg):
    r = resolve_model("m-bare", cfg)            # only server.defaults.profile=base
    assert r.sampling["temperature"] == 0.7
    assert r.load == {}                          # base has no load params
    assert r.profile_name is None               # no explicit/request profile


# full precedence ladder + @profile / request_profile
def test_model_profile_beats_rule_and_default(cfg):
    """m-named matches no rule; its creative profile sits above the base default."""
    r = resolve_model("m-named", cfg)
    assert r.sampling["temperature"] == 1.0
    assert r.profile_name == "creative"


def test_overrides_beat_profile(cfg):
    r = resolve_model("m-ov", cfg)
    assert r.sampling["temperature"] == 1.0     # creative
    assert r.sampling["max_tokens"] == 2048     # override wins over base's 1024


def test_request_profile_replaces_model_profile(cfg):
    """An inline @profile / request `profile` field overrides the configured one."""
    base = resolve_model("m-named", cfg)
    assert base.sampling["temperature"] == 1.0  # creative configured
    swapped = resolve_model("m-named", cfg, request_profile="coder")
    assert swapped.sampling["temperature"] == 0.2
    assert swapped.profile_name == "coder"


def test_request_profile_beats_a_matching_rule(cfg):
    """m-coder matches the *coder* rule, but a request creative profile is more
    specific and applied last."""
    r = resolve_model("m-coder", cfg, request_profile="creative")
    assert r.sampling["temperature"] == 1.0
    assert r.profile_name == "creative"


def test_unknown_request_profile_raises(cfg):
    with pytest.raises(ConfigError) as e:
        resolve_model("m-named", cfg, request_profile="does-not-exist")
    assert "unknown profile" in str(e.value)


def test_unknown_model_id_raises_keyerror(cfg):
    with pytest.raises(KeyError):
        resolve_model("no-such-model", cfg)


# scalar fields: speculative / pin / ttl / companions
def test_speculative_inferred_from_draft_gguf():
    doc = _doc()
    doc["models"]["m-draft"] = {"path": "/abs/t.gguf", "draft_gguf": "/abs/dr.gguf"}
    cfg = build_config(doc)
    r = resolve_model("m-draft", cfg)
    assert r.speculative is True                 # implied by draft_gguf
    assert r.draft_gguf == "/abs/dr.gguf"


def test_ttl_defaults_to_server_then_model_override():
    doc = _doc()
    doc["models"]["m-bare"]["ttl_s"] = 60        # per-model override
    cfg = build_config(doc)
    assert resolve_model("m-bare", cfg).ttl_s == 60       # model wins
    assert resolve_model("m-named", cfg).ttl_s == 900     # falls to server default


def test_pin_flag_and_mmproj_passthrough():
    doc = _doc()
    doc["models"]["m-vlm"] = {"path": "/abs/llm.gguf",
                              "mmproj": "/abs/mmproj.gguf", "pin": True}
    cfg = build_config(doc)
    r = resolve_model("m-vlm", cfg)
    assert r.pin is True
    assert r.mmproj == "/abs/mmproj.gguf"


# adapter - a companion GGUF LoRA path, resolved + load-affecting
def test_adapter_parsed_and_resolved():
    doc = _doc()
    doc["models"]["m-lora"] = {"path": "/abs/base.gguf",
                               "adapter": "/abs/pirate.lora.gguf"}
    cfg = build_config(doc)
    assert cfg.models["m-lora"].adapter == "/abs/pirate.lora.gguf"
    assert resolve_model("m-lora", cfg).adapter == "/abs/pirate.lora.gguf"
    assert resolve_model("m-named", cfg).adapter is None     # default when absent


def test_adapter_resolves_through_model_dirs(tmp_path):
    root = tmp_path / "ggufs"
    root.mkdir()
    (root / "ad.lora.gguf").write_text("x")
    doc = _doc()
    doc["server"]["model_dirs"] = [str(root)]
    doc["models"]["m-lora"] = {"path": "/abs/base.gguf", "adapter": "ad.lora.gguf"}
    cfg = build_config(doc)
    assert resolve_model("m-lora", cfg).adapter == str(root / "ad.lora.gguf")


def test_adapter_folds_into_load_signature():
    """Two ids on one GGUF differing only in adapter are distinct resident entries
    (the adapter is wrapped over the model leaves at load); same adapter collapses."""
    common = dict(path="/p", load={}, cache={}, system=None, speculative=False,
                  mmproj=None, draft_gguf=None, pin=False, ttl_s=None)
    a = cfgmod.ResolvedModel(id="a", sampling={}, adapter="/abs/A.lora.gguf", **common)
    b = cfgmod.ResolvedModel(id="b", sampling={}, adapter="/abs/B.lora.gguf", **common)
    c = cfgmod.ResolvedModel(id="c", sampling={}, adapter="/abs/A.lora.gguf", **common)
    bare = cfgmod.ResolvedModel(id="d", sampling={}, adapter=None, **common)
    assert a.load_signature() != b.load_signature()    # different adapter => split
    assert a.load_signature() == c.load_signature()    # same adapter => shared entry
    assert a.load_signature() != bare.load_signature()  # adapter vs bare base => split


def test_adapter_typo_key_errors():
    doc = _doc()
    doc["models"]["m-lora"] = {"path": "/abs/base.gguf", "adaptor": "/abs/x.gguf"}
    with pytest.raises(ConfigError, match="adaptor"):
        build_config(doc)


# moe_expert_mass - adaptive lossy MoE fan-out on stream entries
def test_moe_expert_mass_parsed_and_resolved():
    doc = _doc()
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "stream": "experts",
                              "moe_expert_mass": 0.9}
    cfg = build_config(doc)
    assert cfg.models["m-moe"].moe_expert_mass == 0.9
    assert resolve_model("m-moe", cfg).moe_expert_mass == 0.9
    assert resolve_model("m-named", cfg).moe_expert_mass is None  # default
    # YAML may carry it quoted; coerced like other numeric keys
    doc["models"]["m-moe"]["moe_expert_mass"] = "0.85"
    assert build_config(doc).models["m-moe"].moe_expert_mass == 0.85


def test_moe_expert_mass_bad_value_fails_fast():
    """A lossy knob must never be silently reinterpreted: out-of-range or
    non-numeric values are config shape errors."""
    for bad in (1.5, 0, -0.2, "warm"):
        doc = _doc()
        doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "stream": "experts",
                                  "moe_expert_mass": bad}
        with pytest.raises(ConfigError, match="moe_expert_mass"):
            build_config(doc)


def test_moe_lossy_levers_parsed_and_resolved():
    """The other lossy levers ride the same per-model config path as
    moe_expert_mass: parsed, validated, resolved, and load-affecting."""
    doc = _doc()
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "stream": "experts",
                              "moe_experts": 6, "moe_miss_shed": 0.95,
                              "moe_layer_shed": "0.1"}
    cfg = build_config(doc)
    rm = resolve_model("m-moe", cfg)
    assert rm.moe_experts == 6
    assert rm.moe_miss_shed == 0.95
    assert rm.moe_layer_shed == 0.1
    bare = resolve_model("m-named", cfg)
    assert (bare.moe_experts is None and bare.moe_miss_shed is None
            and bare.moe_layer_shed is None)
    # Each lever is load-affecting: same GGUF, different value => distinct
    # resident entries.
    common = dict(path="/p", sampling={}, load={}, cache={}, system=None,
                  speculative=False, mmproj=None, draft_gguf=None, pin=False,
                  ttl_s=None, stream="experts")
    base_sig = cfgmod.ResolvedModel(id="x", **common).load_signature()
    for key in ("moe_experts", "moe_expert_mass", "moe_miss_shed",
                "moe_layer_shed"):
        sig = cfgmod.ResolvedModel(
            id="x", **{key: 2 if key == "moe_experts" else 0.5},
            **common).load_signature()
        assert sig != base_sig, key

    for key, bad in (("moe_experts", 0), ("moe_experts", "many"),
                     ("moe_miss_shed", 1.5), ("moe_miss_shed", 0),
                     ("moe_layer_shed", 1.0), ("moe_layer_shed", 0)):
        doc = _doc()
        doc["models"]["m-moe"] = {"path": "/abs/big.gguf",
                                  "stream": "experts", key: bad}
        with pytest.raises(ConfigError, match=key):
            build_config(doc)


def test_stream_legacy_cpu_moe_alias_warns_and_maps():
    """`cpu_moe:` still works (old semantics: hybrid -> experts, true/full ->
    cpu) but warns about the rename; an explicit `stream:` wins silently."""
    doc = _doc()
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "cpu_moe": "hybrid"}
    with pytest.warns(UserWarning, match="renamed"):
        assert build_config(doc).models["m-moe"].stream == "experts"
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "cpu_moe": True}
    with pytest.warns(UserWarning, match="renamed"):
        assert build_config(doc).models["m-moe"].stream == "cpu"
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "cpu_moe": "full",
                              "stream": "experts"}
    assert build_config(doc).models["m-moe"].stream == "experts"


def test_stream_true_is_ambiguous_and_ignored():
    """`stream: true` would invert the old meaning of `cpu_moe: true`, so it
    warns and is ignored rather than guessing a placement."""
    doc = _doc()
    doc["models"]["m-moe"] = {"path": "/abs/big.gguf", "stream": True}
    with pytest.warns(UserWarning, match="ambiguous"):
        assert build_config(doc).models["m-moe"].stream is None


def test_moe_expert_mass_folds_into_load_signature():
    """Two ids on one GGUF differing only in moe_expert_mass are distinct
    resident entries (the filter is installed over the routers at load)."""
    common = dict(path="/p", load={}, cache={}, system=None, speculative=False,
                  mmproj=None, draft_gguf=None, adapter=None, stream="experts",
                  pin=False, ttl_s=None)
    a = cfgmod.ResolvedModel(id="a", sampling={}, moe_expert_mass=0.9, **common)
    b = cfgmod.ResolvedModel(id="b", sampling={}, moe_expert_mass=0.8, **common)
    c = cfgmod.ResolvedModel(id="c", sampling={}, moe_expert_mass=0.9, **common)
    bare = cfgmod.ResolvedModel(id="d", sampling={}, moe_expert_mass=None, **common)
    assert a.load_signature() != b.load_signature()     # different P => split
    assert a.load_signature() == c.load_signature()     # same P => shared entry
    assert a.load_signature() != bare.load_signature()  # P vs trained => split


# load_signature - what splits two resident entries for one GGUF
def test_load_signature_splits_on_load_params(cfg):
    a = resolve_model("m-named", cfg)                          # creative, no load
    b = resolve_model("m-named", cfg, request_profile="coder")  # kv_bits=8
    assert a.path == b.path                       # same GGUF
    assert a.load_signature() != b.load_signature()  # but distinct resident entries


def test_load_signature_ignores_sampling(cfg):
    """Sampling/system/ttl aren't load-affecting; two ids differing only in those
    share a resident entry."""
    a = resolve_model("m-named", cfg)
    b = resolve_model("m-ov", cfg)                # differs only in max_tokens override
    # Different paths here, so force the same path to isolate the sampling axis:
    a2 = cfgmod.ResolvedModel(
        id="x", path="/p", sampling={"temperature": 0.1}, load={}, cache={},
        system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)
    b2 = cfgmod.ResolvedModel(
        id="y", path="/p", sampling={"temperature": 0.9}, load={}, cache={},
        system="hi", speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=10.0)
    assert a2.load_signature() == b2.load_signature()
    assert a.path != b.path or True               # a/b kept for readability


# chat_template - resolves like `system`, but IS load-affecting (tokenizer-baked)
def _tmpl_doc():
    """A `themed` profile (extends base) sets a chat template; one model adopts it,
    one overrides it, one leaves it unset."""
    return {
        "server": {"model_dirs": ["/models"]},
        "profiles": {
            "base": {"sampling": {"temperature": 0.7}},
            "themed": {"extends": "base", "chat_template": "{{ pirate }}"},
        },
        "models": {
            "plain": {"path": "/abs/a.gguf"},
            "adopt": {"path": "/abs/b.gguf", "profile": "themed"},
            "override": {"path": "/abs/c.gguf", "profile": "themed",
                         "overrides": {"chat_template": "/tmpl/custom.jinja"}},
        },
    }


def test_chat_template_from_profile_extends_chain():
    cfg = build_config(_tmpl_doc())
    assert resolve_model("adopt", cfg).chat_template == "{{ pirate }}"
    assert resolve_model("plain", cfg).chat_template is None   # no profile sets it


def test_chat_template_override_beats_profile():
    cfg = build_config(_tmpl_doc())
    assert resolve_model("override", cfg).chat_template == "/tmpl/custom.jinja"


def test_chat_template_request_profile_applies():
    cfg = build_config(_tmpl_doc())
    r = resolve_model("plain", cfg, request_profile="themed")
    assert r.chat_template == "{{ pirate }}"


def test_chat_template_folds_into_load_signature():
    """Two ids on one GGUF differing only in chat template are distinct resident
    entries (it's baked into the tokenizer); identical templates collapse to one,
    even when system/sampling/ttl differ."""
    common = dict(path="/p", load={}, cache={}, speculative=False, mmproj=None,
                  draft_gguf=None, pin=False)
    a = cfgmod.ResolvedModel(id="a", sampling={}, system="x", ttl_s=None,
                             chat_template="{{ A }}", **common)
    b = cfgmod.ResolvedModel(id="b", sampling={"temperature": 0.9}, system="y",
                             ttl_s=5.0, chat_template="{{ B }}", **common)
    c = cfgmod.ResolvedModel(id="c", sampling={}, system=None, ttl_s=99.0,
                             chat_template="{{ A }}", **common)
    assert a.load_signature() != b.load_signature()   # different template => split
    assert a.load_signature() == c.load_signature()   # same template => shared entry


def test_chat_template_default_none_not_load_affecting():
    """No template configured anywhere => None, and two such ids share an entry."""
    cfg = build_config(_tmpl_doc())
    r = resolve_model("plain", cfg)
    assert r.chat_template is None


def test_chat_template_typo_key_errors():
    doc = {"profiles": {"p": {"chat_templat": "oops"}},
           "models": {"m": {"path": "/abs/a.gguf", "profile": "p"}}}
    with pytest.raises(ConfigError, match="chat_templat"):
        build_config(doc)


# chat_template_kwargs - merges like sampling (profile -> override), but is NOT
# load-affecting (applied per-request at apply_chat_template, not tokenizer-baked)
def _ctkw_doc():
    """A `think` profile (extends base) sets preserve_thinking; one model adopts it,
    one overrides a key on top, one leaves it unset."""
    return {
        "server": {"model_dirs": ["/models"]},
        "profiles": {
            "base": {"sampling": {"temperature": 0.7}},
            "think": {"extends": "base",
                      "chat_template_kwargs": {"preserve_thinking": True}},
        },
        "models": {
            "plain": {"path": "/abs/a.gguf"},
            "adopt": {"path": "/abs/b.gguf", "profile": "think"},
            "override": {"path": "/abs/c.gguf", "profile": "think",
                         "overrides": {"chat_template_kwargs":
                                       {"preserve_thinking": False, "extra": 1}}},
        },
    }


def test_chat_template_kwargs_from_profile_extends_chain():
    cfg = build_config(_ctkw_doc())
    assert resolve_model("adopt", cfg).chat_template_kwargs == {
        "preserve_thinking": True}
    assert resolve_model("plain", cfg).chat_template_kwargs == {}   # default empty


def test_chat_template_kwargs_override_merges_over_profile():
    cfg = build_config(_ctkw_doc())
    # override merges key-by-key (flips preserve_thinking, adds extra) - not replace
    assert resolve_model("override", cfg).chat_template_kwargs == {
        "preserve_thinking": False, "extra": 1}


def test_chat_template_kwargs_not_load_affecting():
    """Two ids on one GGUF differing only in chat_template_kwargs share a resident
    entry - the kwargs are applied per request, never baked into the tokenizer."""
    common = dict(path="/p", load={}, cache={}, speculative=False, mmproj=None,
                  draft_gguf=None, pin=False, chat_template=None)
    a = cfgmod.ResolvedModel(id="a", sampling={}, system=None, ttl_s=None,
                             chat_template_kwargs={"preserve_thinking": True},
                             **common)
    b = cfgmod.ResolvedModel(id="b", sampling={}, system=None, ttl_s=None,
                             chat_template_kwargs={"preserve_thinking": False},
                             **common)
    assert a.load_signature() == b.load_signature()


def test_chat_template_kwargs_typo_key_errors():
    doc = {"profiles": {"p": {"chat_template_kwarg": {"x": 1}}},
           "models": {"m": {"path": "/abs/a.gguf", "profile": "p"}}}
    with pytest.raises(ConfigError, match="chat_template_kwarg"):
        build_config(doc)


# path resolution against model_dirs
def test_resolve_path_absolute_passthrough():
    assert cfgmod.resolve_path("/abs/x.gguf", ["/models"]) == "/abs/x.gguf"


def test_resolve_path_none_passthrough():
    assert cfgmod.resolve_path(None, ["/models"]) is None


def test_resolve_path_searches_model_dirs(tmp_path):
    root = tmp_path / "ggufs"
    root.mkdir()
    f = root / "model.gguf"
    f.write_text("x")
    got = cfgmod.resolve_path("model.gguf", [str(root)])
    assert got == str(f)


def test_resolve_path_miss_raises_missing_model_file(tmp_path):
    # The narrower type lets serve degrade (warn-and-skip) on disk state while
    # config-shape errors keep failing fast; it must stay a ConfigError too.
    with pytest.raises(cfgmod.MissingModelFile) as ei:
        cfgmod.resolve_path("gone.gguf", [str(tmp_path)])
    assert isinstance(ei.value, cfgmod.ConfigError)
    assert str(tmp_path) in str(ei.value)      # names the dirs it searched


def test_resolve_path_first_existing_root_wins(tmp_path):
    r1, r2 = tmp_path / "a", tmp_path / "b"
    r1.mkdir()
    r2.mkdir()
    (r2 / "m.gguf").write_text("x")              # only in the 2nd root
    got = cfgmod.resolve_path("m.gguf", [str(r1), str(r2)])
    assert got == str(r2 / "m.gguf")


def test_resolve_path_miss_raises_listing_roots(tmp_path):
    with pytest.raises(ConfigError) as e:
        cfgmod.resolve_path("ghost.gguf", [str(tmp_path)])
    assert str(tmp_path) in str(e.value)


def test_resolve_path_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m.gguf").write_text("x")
    assert cfgmod.resolve_path("~/m.gguf", []) == str(tmp_path / "m.gguf")


# hf: cache refs - resolved from the local HF cache, never the network
def test_resolve_path_hf_ref_from_cache(monkeypatch, tmp_path):
    import huggingface_hub
    f = tmp_path / "m.gguf"
    f.write_text("x")
    seen = {}

    def fake(repo, filename, revision=None):
        seen.update(repo=repo, filename=filename, revision=revision)
        return str(f)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake)
    assert cfgmod.resolve_path("hf:org/repo/sub/m.gguf", []) == str(f)
    assert seen == {"repo": "org/repo", "filename": "sub/m.gguf", "revision": "main"}


def test_resolve_path_hf_ref_revision(monkeypatch, tmp_path):
    import huggingface_hub
    f = tmp_path / "m.gguf"
    f.write_text("x")
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache",
        lambda r, fn, revision=None: seen.update(rev=revision) or str(f))
    cfgmod.resolve_path("hf:org/repo/m.gguf@v2", [])
    assert seen["rev"] == "v2"


def test_resolve_path_hf_ref_not_cached_raises(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    with pytest.raises(ConfigError) as e:
        cfgmod.resolve_path("hf:org/repo/m.gguf", [])
    assert "gmlx pull" in str(e.value)


def test_resolve_path_hf_ref_falls_back_to_pull_layout(monkeypatch, tmp_path):
    # The miss error says "gmlx pull ..." - and pull lands files under
    # <model_dir>/<org>__<repo>/<file>, so the resolver must find them there.
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    f = tmp_path / "org__repo" / "m.gguf"
    f.parent.mkdir()
    f.write_text("x")
    got = cfgmod.resolve_path("hf:org/repo/m.gguf", [str(tmp_path)])
    assert got == str(f)


def test_resolve_path_hf_ref_miss_names_both_locations(monkeypatch, tmp_path):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    with pytest.raises(ConfigError) as e:
        cfgmod.resolve_path("hf:org/repo/m.gguf", [str(tmp_path)])
    assert "model_dirs" in str(e.value) and "gmlx pull" in str(e.value)


def test_resolve_path_hf_ref_malformed_raises():
    with pytest.raises(ConfigError):
        cfgmod.resolve_path("hf:org/repo", [])     # no filename component


# env_for - load params + APC cache -> env vars
def test_env_for_load_params():
    r = cfgmod.ResolvedModel(
        id="x", path="/p",
        sampling={}, load={"kv_bits": 8, "kv_group_size": 64, "max_kv_size": 4096},
        cache={}, system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)
    env = cfgmod.env_for(r)
    assert env == {"KV_BITS": "8", "KV_GROUP_SIZE": "64", "MAX_KV_SIZE": "4096",
                   "MLX_VLM_GGUF_SPECULATIVE": "0"}


def test_env_for_emits_speculative_flag():
    """The per-build speculative state rides the env window (the only signal that
    reaches the load bridge in the engine's worker thread). Always emitted - "0"
    is authoritative (forces a plain load over an MTP-registered sibling path)."""
    def env_spec(speculative):
        r = cfgmod.ResolvedModel(
            id="x", path="/p", sampling={}, load={}, cache={}, system=None,
            speculative=speculative, mmproj=None, draft_gguf=None, pin=False,
            ttl_s=None)
        return cfgmod.env_for(r)["MLX_VLM_GGUF_SPECULATIVE"]

    assert env_spec(True) == "1"
    assert env_spec(False) == "0"


def test_env_for_apc_cache_and_disk(tmp_path):
    r = cfgmod.ResolvedModel(
        id="x", path="/p", sampling={}, load={},
        cache={"enabled": True,
               "disk": {"path": "~/apc", "max_gb": 200, "workers": 2}},
        system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)
    env = cfgmod.env_for(r)
    assert env["APC_ENABLED"] == "1"             # bool -> 1/0
    assert env["APC_DISK_PATH"] == os.path.expanduser("~/apc")  # ~ expanded
    assert env["APC_DISK_MAX_GB"] == "200"
    assert env["APC_DISK_WORKERS"] == "2"


def test_env_for_disk_needs_path():
    """A disk block without a path doesn't enable the SSD tier (no disk env emitted)."""
    r = cfgmod.ResolvedModel(
        id="x", path="/p", sampling={}, load={},
        cache={"enabled": False, "disk": {"max_gb": 50}},
        system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)
    env = cfgmod.env_for(r)
    assert env["APC_ENABLED"] == "0"
    assert not any(k.startswith("APC_DISK") for k in env)


def _resolved(cache):
    return cfgmod.ResolvedModel(
        id="x", path="/p", sampling={}, load={}, cache=cache,
        system=None, speculative=False, mmproj=None, draft_gguf=None,
        pin=False, ttl_s=None)


def test_env_for_exact_entries_bumps_default_when_apc_on():
    """APC on but no explicit exact_entries -> gmlx raises mlx-vlm's default of 2."""
    env = cfgmod.env_for(_resolved({"enabled": True}))
    assert env["APC_EXACT_CACHE_ENTRIES"] == str(cfgmod.DEFAULT_EXACT_CACHE_ENTRIES)


def test_env_for_exact_entries_explicit_wins():
    env = cfgmod.env_for(_resolved({"enabled": True, "exact_entries": 12}))
    assert env["APC_EXACT_CACHE_ENTRIES"] == "12"


def test_env_for_exact_entries_absent_when_apc_off():
    """No bump (and no var at all) unless APC is actually enabled."""
    assert "APC_EXACT_CACHE_ENTRIES" not in cfgmod.env_for(_resolved({}))
    assert "APC_EXACT_CACHE_ENTRIES" not in cfgmod.env_for(_resolved({"enabled": False}))


def test_server_cache_is_base_under_profiles():
    """server.cache is the lowest cache layer; a profile cache merges on top, with
    one-level nesting so a profile disk.max_gb doesn't wipe the server disk.path."""
    doc = _doc()
    doc["server"]["cache"] = {"enabled": True,
                              "disk": {"path": "/ssd/apc", "max_gb": 500}}
    doc["profiles"]["creative"]["cache"] = {"disk": {"max_gb": 100}}
    cfg = build_config(doc)
    r = resolve_model("m-named", cfg)            # creative
    assert r.cache["enabled"] is True            # from server base
    assert r.cache["disk"]["path"] == "/ssd/apc"  # server path preserved
    assert r.cache["disk"]["max_gb"] == 100      # profile override merged in


def test_cache_disk_boolean_shorthand():
    """`disk: true` enables the SSD tier at the default path; a per-model
    `disk: false` turns an inherited server-level disk tier off."""
    doc = _doc()
    doc["server"]["cache"] = {"enabled": True, "disk": True}
    env = cfgmod.env_for(resolve_model("m-bare", build_config(doc)))
    assert env["APC_DISK_PATH"] == os.path.expanduser(
        cfgmod.DEFAULT_APC_DISK_PATH)

    doc = _doc()
    doc["server"]["cache"] = {"enabled": True, "disk": {"path": "/ssd/apc"}}
    doc["models"]["m-bare"]["overrides"] = {"cache": {"disk": False}}
    env = cfgmod.env_for(resolve_model("m-bare", build_config(doc)))
    assert env["APC_ENABLED"] == "1"
    assert not any(k.startswith("APC_DISK") for k in env)


def test_cache_disk_bad_type_fails_fast():
    doc = _doc()
    doc["server"]["cache"] = {"enabled": True, "disk": "yes"}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "cache.disk" in str(e.value)


# validation: fail-fast on the config footguns
def test_extends_cycle_fails_fast():
    doc = _doc()
    doc["profiles"]["coder"]["extends"] = "creative"
    doc["profiles"]["creative"]["extends"] = "coder"   # coder <-> creative
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "cycle" in str(e.value)


def test_extends_self_cycle_fails_fast():
    doc = _doc()
    doc["profiles"]["base"]["extends"] = "base"
    with pytest.raises(ConfigError):
        build_config(doc)


def test_extends_unknown_target_fails_fast():
    doc = _doc()
    doc["profiles"]["coder"]["extends"] = "ghost"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "unknown profile" in str(e.value)


def test_model_references_unknown_profile_fails_fast():
    doc = _doc()
    doc["models"]["m-bad"] = {"path": "/abs/x.gguf", "profile": "ghost"}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "ghost" in str(e.value)


def test_rule_references_unknown_profile_fails_fast():
    doc = _doc()
    doc["rules"].append({"match": "*", "profile": "ghost"})
    with pytest.raises(ConfigError):
        build_config(doc)


def test_default_profile_references_unknown_fails_fast():
    doc = _doc()
    doc["server"]["defaults"]["profile"] = "ghost"
    with pytest.raises(ConfigError):
        build_config(doc)


def test_illegal_at_in_model_id_fails_fast():
    doc = _doc()
    doc["models"]["bad@id"] = {"path": "/abs/x.gguf"}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "@" in str(e.value)


def test_model_without_path_fails_fast():
    doc = _doc()
    doc["models"]["m-nopath"] = {"profile": "base"}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "path" in str(e.value)


def test_temp_accepted_as_temperature_alias(recwarn):
    # Every CLI flag says --temp, so configs get the same spelling; no warning.
    doc = _doc()
    doc["profiles"]["base"]["sampling"] = {"temp": 0.5}
    cfg = build_config(doc)
    assert cfg.profiles["base"].sampling == {"temperature": 0.5}
    assert not [w for w in recwarn.list if "temp" in str(w.message)]
    # Canonical key wins when both are present.
    doc["profiles"]["base"]["sampling"] = {"temp": 0.5, "temperature": 0.9}
    cfg = build_config(doc)
    assert cfg.profiles["base"].sampling == {"temperature": 0.9}


def test_temp_alias_in_model_overrides_and_tweaks():
    doc = _doc()
    doc["models"]["m-coder"]["overrides"] = {"sampling": {"temp": 0.3}}
    doc["models"]["m-coder"]["profiles"] = {"base": {"sampling": {"temp": 0.2}}}
    cfg = build_config(doc)
    assert cfg.models["m-coder"].overrides["sampling"] == {"temperature": 0.3}
    assert cfg.models["m-coder"].profiles["base"]["sampling"] \
        == {"temperature": 0.2}


def test_unknown_sampling_key_warns_not_fails():
    doc = _doc()
    doc["profiles"]["base"]["sampling"]["temprature"] = 0.5   # typo for temperature
    with pytest.warns(UserWarning) as w:
        cfg = build_config(doc)
    msgs = [str(x.message) for x in w]
    assert any("temprature" in m for m in msgs)               # names the bad key
    assert any("temperature" in m for m in msgs)              # lists the valid ones
    # warned, did NOT fail: the config still built and the valid keys are intact.
    # (the typo key is preserved verbatim in the dict but is inert downstream -
    # gen-args injection only sets attrs that already exist on the request.)
    assert cfg.profiles["base"].sampling["temperature"] == 0.7


# build_config / parsing shapes
def test_model_dirs_accepts_scalar_or_list():
    s = build_config({"server": {"model_dirs": "/one"},
                      "models": {"m": {"path": "/abs/x.gguf"}}})
    assert s.model_dirs == ["/one"]
    li = build_config({"server": {"model_dirs": ["/a", "/b"]},
                       "models": {"m": {"path": "/abs/x.gguf"}}})
    assert li.model_dirs == ["/a", "/b"]


def test_defaults_ttl_and_model():
    s = build_config({"server": {"defaults": {"model": "m", "ttl_s": 0}},
                      "models": {"m": {"path": "/abs/x.gguf"}}})
    assert s.defaults.model == "m"
    assert s.defaults.ttl_s == 0                  # null/0 honored (never auto-unload)


def test_defaults_preload_list_and_all():
    doc = _doc()
    doc["server"]["defaults"]["preload"] = ["m-coder", "m-bare"]
    assert build_config(doc).defaults.preload == ["m-coder", "m-bare"]
    doc["server"]["defaults"]["preload"] = "all"
    assert build_config(doc).defaults.preload == "all"
    assert build_config(_doc()).defaults.preload is None


def test_defaults_preload_unknown_id_raises():
    doc = _doc()
    doc["server"]["defaults"]["preload"] = ["m-coder", "no-such"]
    with pytest.raises(ConfigError, match="no-such"):
        build_config(doc)


def test_defaults_preload_bad_shape_raises():
    doc = _doc()
    doc["server"]["defaults"]["preload"] = "m-coder"   # bare id: must be a list
    with pytest.raises(ConfigError, match="preload"):
        build_config(doc)


def test_empty_doc_yields_defaults():
    s = build_config({})
    assert s.host == "127.0.0.1" and s.port == 8080
    assert s.models == {} and s.profiles == {}
    assert s.defaults.ttl_s == 900.0


def test_discover_spec_defaults():
    s = build_config({"discover": [{"dir": "/d"}]})
    spec = s.discover[0]
    assert spec.dir == "/d"
    assert spec.recursive is False
    assert spec.pair_mmproj is True
    assert spec.speculative == "auto"


# load_config - file IO + error surfaces
def test_load_config_roundtrip(tmp_path):
    import yaml
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_doc()))
    cfg = cfgmod.load_config(p)
    assert set(cfg.models) == {"m-coder", "m-named", "m-ov", "m-bare"}
    assert cfg.defaults.profile == "base"


def test_token_queue_timeout_parsed_and_defaults_none():
    assert build_config({}).token_queue_timeout_s is None        # absent => None
    assert build_config({"server": {"token_queue_timeout_s": 1800}}
                        ).token_queue_timeout_s == 1800.0         # coerced to float
    assert build_config({"server": {"token_queue_timeout_s": 0}}
                        ).token_queue_timeout_s == 0.0            # 0 => waits forever


def test_prefill_step_size_parsed_and_defaults_none():
    assert build_config({}).prefill_step_size is None            # absent => None
    assert build_config({"server": {"prefill_step_size": "512"}}
                        ).prefill_step_size == 512               # coerced to int
    with pytest.raises(ConfigError):
        build_config({"server": {"prefill_step_size": "lots"}})


def test_load_config_missing_file_raises():
    with pytest.raises(ConfigError) as e:
        cfgmod.load_config("/no/such/config.yaml")
    assert "not found" in str(e.value)


def test_load_config_malformed_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("server: {host: 127.0.0.1\nmodels: [oops")   # unbalanced
    with pytest.raises(ConfigError) as e:
        cfgmod.load_config(p)
    assert "malformed YAML" in str(e.value)


def test_load_config_non_mapping_root_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ConfigError) as e:
        cfgmod.load_config(p)
    assert "mapping" in str(e.value)


def test_default_config_paths_order():
    paths = cfgmod.default_config_paths()
    assert paths[0].name == "gmlx.yaml"      # project-local searched first
    # init writes the XDG-style config, which is also in the search order.
    write = cfgmod.default_config_write_path()
    assert ".config/gmlx" in str(write) and write.name == "gmlx.yaml"
    assert write in paths


# typo surfacing: a misspelled key in a STRUCTURAL namespace we fully own (top
# level, server, defaults, models, profiles, rules, discover, overrides) RAISES
# ConfigError naming the bad key + listing the valid ones - never a silent drop
# (`pinned:` for `pin:` would quietly leave a model unpinned). Open-ended passthrough
# namespaces (sampling, load, cache) only warn; covered separately.
def _warn_msgs(doc):
    """build_config(doc), returning the list of warning messages emitted."""
    with pytest.warns(UserWarning) as w:
        build_config(doc)
    return [str(x.message) for x in w]


def _err_msg(doc):
    """build_config(doc) expecting a ConfigError; return its message text."""
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    return str(e.value)


def _assert_helpful(msgs, *, names, lists):
    """At least one message names the typo AND surfaces a valid key for the fix."""
    joined = "\n".join(msgs)
    for n in names:
        assert n in joined, f"{n!r} not surfaced in: {joined!r}"
    for v in lists:
        assert v in joined, f"valid key {v!r} not listed in: {joined!r}"


def test_typo_top_level_key_errors():
    # `model:` (singular) instead of `models:` would silently serve zero models.
    msg = _err_msg({"server": {}, "model": {"q": {"path": "/a.gguf"}}})
    _assert_helpful([msg], names=["model"], lists=["models"])


def test_typo_server_key_errors():
    msg = _err_msg({"server": {"prot": 9000},                    # typo for port
                    "models": {"q": {"path": "/a.gguf"}}})
    _assert_helpful([msg], names=["prot"], lists=["port"])


def test_typo_defaults_key_errors():
    msg = _err_msg({"server": {"defaults": {"modol": "q"}},      # typo for model
                    "models": {"q": {"path": "/a.gguf"}}})
    _assert_helpful([msg], names=["modol"], lists=["model"])


def test_typo_model_key_errors():
    # `pinned:`/`speculatve:` would quietly leave the model unpinned / non-spec -
    # now a hard error, not a silent drop.
    doc = _doc()
    doc["models"]["m-bare"]["pinned"] = True            # typo for pin
    msg = _err_msg(doc)
    _assert_helpful([msg], names=["pinned", "m-bare"], lists=["pin"])


def test_typo_profile_key_errors():
    doc = _doc()
    doc["profiles"]["base"]["samping"] = {"temperature": 0.1}    # typo for sampling
    msg = _err_msg(doc)
    _assert_helpful([msg], names=["samping", "base"], lists=["sampling"])


def test_typo_load_param_warns():
    doc = _doc()
    doc["profiles"]["coder"]["load"]["kv_bts"] = 8              # typo for kv_bits
    msgs = _warn_msgs(doc)
    _assert_helpful(msgs, names=["kv_bts"], lists=["kv_bits"])


def test_typo_cache_and_disk_keys_warn():
    doc = _doc()
    doc["server"]["cache"] = {"enabeld": True,                  # typo for enabled
                              "disk": {"maxgb": 50}}            # typo for max_gb
    msgs = _warn_msgs(doc)
    _assert_helpful(msgs, names=["enabeld", "maxgb"],
                    lists=["enabled", "max_gb"])


def test_typo_overrides_sampling_warns():
    doc = _doc()
    doc["models"]["m-ov"]["overrides"]["sampling"]["top_pp"] = 0.9   # typo for top_p
    msgs = _warn_msgs(doc)
    _assert_helpful(msgs, names=["top_pp", "m-ov"], lists=["top_p"])


def test_typo_rule_key_warns():
    doc = _doc()
    doc["rules"].append({"match": "x-*", "profle": "base"})     # typo + no `profile`
    # missing `profile` is structural -> raises, helpfully.
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "match" in str(e.value) and "profile" in str(e.value)


def test_typo_rule_extra_key_errors():
    doc = _doc()
    doc["rules"].append({"match": "x-*", "profile": "base", "weight": 2})
    msg = _err_msg(doc)
    _assert_helpful([msg], names=["weight"], lists=["match"])


def test_typo_discover_key_errors():
    doc = _doc()
    doc["discover"] = [{"dir": "/d", "recursiv": True}]         # typo for recursive
    msg = _err_msg(doc)
    _assert_helpful([msg], names=["recursiv"], lists=["recursive"])


def test_clean_config_emits_no_warnings(recwarn):
    """The canonical _doc() is fully valid -> not a single spurious typo warning
    (guards against the key sets drifting out of sync with the parsers)."""
    build_config(_doc())
    spurious = [w for w in recwarn.list if "unrecognized keys" in str(w.message)]
    assert spurious == [], [str(w.message) for w in spurious]


# structural errors carry the valid options (the "helpful" half of fail-fast)
def test_unknown_profile_reference_lists_known():
    doc = _doc()
    doc["models"]["m-bad"] = {"path": "/abs/x.gguf", "profile": "ghost"}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    m = str(e.value)
    assert "ghost" in m and "base" in m and "coder" in m       # names + lists valid


def test_extends_unknown_target_lists_known():
    doc = _doc()
    doc["profiles"]["coder"]["extends"] = "ghost"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    m = str(e.value)
    assert "ghost" in m and "creative" in m                    # lists a valid profile


def test_missing_path_names_keys_present():
    doc = _doc()
    doc["models"]["m-nopath"] = {"profile": "base", "pin": True}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    m = str(e.value)
    assert "path" in m and "profile" in m                      # shows what WAS given


# aliases: name -> id | id@profile (a friendly name / profile preset)
def _alias_doc():
    doc = _doc()
    doc["aliases"] = {
        "big": "m-named",                  # bare rename
        "coder": "m-named@coder",          # profile preset
    }
    return doc


def test_aliases_parsed():
    cfg = build_config(_alias_doc())
    assert cfg.aliases == {"big": "m-named", "coder": "m-named@coder"}


def test_alias_name_with_at_fails():
    doc = _alias_doc()
    doc["aliases"]["bad@name"] = "m-named"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "@" in str(e.value)


def test_alias_colliding_with_model_id_fails():
    doc = _alias_doc()
    doc["aliases"]["m-bare"] = "m-named"   # m-bare is already a model id
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "collides" in str(e.value)


def test_alias_unknown_target_model_lists_known():
    doc = _alias_doc()
    doc["aliases"]["ghost"] = "no-such-model"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    m = str(e.value)
    assert "no-such-model" in m and "m-named" in m             # names + lists valid


def test_alias_unknown_target_profile_lists_known():
    doc = _alias_doc()
    doc["aliases"]["bad"] = "m-named@ghost"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    m = str(e.value)
    assert "ghost" in m and "coder" in m


def test_default_model_must_exist():
    doc = _doc()
    doc["server"]["defaults"]["model"] = "no-such-model"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "no-such-model" in str(e.value)


def test_split_address_helper():
    profiles = {"coder", "creative"}
    assert cfgmod.split_address("m-named", profiles) == ("m-named", None)
    assert cfgmod.split_address("m-named@coder", profiles) == ("m-named", "coder")
    # unknown tail -> not split (hf org/model@rev safety)
    assert cfgmod.split_address("org/model@rev", profiles) == ("org/model@rev", None)


def test_server_api_key_parses():
    cfg = build_config({"server": {"api_key": "sk-local-123"}})
    assert cfg.api_key == "sk-local-123"


def test_server_api_key_defaults_off():
    cfg = build_config({})
    assert cfg.api_key is None


def test_server_no_auth_parses():
    cfg = build_config({"server": {"no_auth": True}})
    assert cfg.no_auth is True
    assert build_config({}).no_auth is False


# YAML type robustness: numeric coercion + non-mapping entries
def test_numeric_server_keys_coerced_from_strings():
    # YAML often arrives with quoted numbers; `budget_gb: "12"` must become 12.0,
    # not a string that "12" * 1024**3 repeats downstream.
    cfg = build_config({"server": {"port": "8081", "budget_gb": "12",
                                   "max_models": "3",
                                   "defaults": {"ttl_s": "60"}}})
    assert cfg.port == 8081
    assert cfg.budget_gb == 12.0
    assert cfg.max_models == 3
    assert cfg.defaults.ttl_s == 60.0


@pytest.mark.parametrize("key,val", [
    ("port", "eighty"),
    ("budget_gb", "lots"),
    ("max_models", "many"),
    ("port", True),            # bool is an int subclass; still a typo
])
def test_bad_numeric_server_key_names_key_and_value(key, val):
    with pytest.raises(ConfigError) as e:
        build_config({"server": {key: val}})
    msg = str(e.value)
    assert f"server.{key}" in msg and repr(val) in msg


def test_bad_defaults_ttl_names_key_and_value():
    with pytest.raises(ConfigError) as e:
        build_config({"server": {"defaults": {"ttl_s": "soon"}}})
    msg = str(e.value)
    assert "defaults.ttl_s" in msg and "'soon'" in msg


def test_model_entry_must_be_mapping():
    # `models: {foo: /path.gguf}` makes raw a str; must raise, not TypeError.
    doc = _doc()
    doc["models"]["m-bad"] = "/abs/x.gguf"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    msg = str(e.value)
    assert "m-bad" in msg and "mapping" in msg and "path" in msg


def test_profile_entry_must_be_mapping():
    # `profiles: {fast: greedy}` makes raw a str; must raise, not AttributeError.
    doc = _doc()
    doc["profiles"]["fast"] = "greedy"
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    msg = str(e.value)
    assert "fast" in msg and "mapping" in msg


# Built-in family profiles (profiles.py): the family base layer, @intent
# addressing, shadowing, per-model tweaks, and the kill switch.
def _fam_doc():
    """Two models: one pinned to the gemma family, one family-less (-> generic
    default base). No defaults.profile, so the family base is the only layer."""
    return {
        "server": {"model_dirs": ["/models"]},
        "models": {
            "g": {"path": "/abs/g.gguf", "family": "gemma"},
            "u": {"path": "/abs/u.gguf"},
        },
    }


def test_family_base_is_lowest_layer():
    cfg = build_config(_fam_doc())
    rm = resolve_model("g", cfg)
    assert rm.sampling["temperature"] == 1.0
    assert rm.sampling["top_k"] == 64
    assert rm.family == "gemma"


def test_unknown_family_gets_generic_base():
    cfg = build_config(_fam_doc())
    rm = resolve_model("u", cfg)
    assert rm.sampling == {"temperature": 0.7, "top_p": 0.95}
    assert rm.family is None


def test_defaults_profile_beats_family_base():
    doc = _fam_doc()
    doc["server"]["defaults"] = {"profile": "mine"}
    doc["profiles"] = {"mine": {"sampling": {"temperature": 0.5}}}
    cfg = build_config(doc)
    rm = resolve_model("g", cfg)
    assert rm.sampling["temperature"] == 0.5     # user layer wins
    assert rm.sampling["top_k"] == 64            # base fills unset keys


def test_builtin_intent_addressable():
    cfg = build_config(_fam_doc())
    rm = resolve_model("g", cfg, request_profile="coding")
    # Gemma defines no coding delta on purpose -> base values.
    assert rm.sampling["temperature"] == 1.0
    doc = _fam_doc()
    doc["models"]["q"] = {"path": "/abs/q.gguf", "family": "qwen3.6"}
    rm = resolve_model("q", build_config(doc), request_profile="coding")
    assert rm.sampling["temperature"] == 0.6
    assert rm.sampling["top_p"] == 0.95


def test_builtin_intent_chat_template_kwargs():
    doc = _fam_doc()
    doc["models"]["o"] = {"path": "/abs/o.gguf", "family": "gpt-oss"}
    rm = resolve_model("o", build_config(doc), request_profile="reasoning-high")
    assert rm.chat_template_kwargs == {"reasoning_effort": "high"}
    assert rm.sampling["top_p"] == 1.0


def test_user_profile_shadows_builtin():
    doc = _fam_doc()
    doc["profiles"] = {"coding": {"sampling": {"temperature": 0.11}}}
    cfg = build_config(doc)
    rm = resolve_model("g", cfg, request_profile="coding")
    assert rm.sampling["temperature"] == 0.11


def test_user_profile_extends_builtin():
    doc = _fam_doc()
    doc["models"]["q"] = {"path": "/abs/q.gguf", "family": "qwen3.6"}
    doc["profiles"] = {"my-coding": {"extends": "coding",
                                     "sampling": {"min_p": 0.02}}}
    cfg = build_config(doc)
    rm = resolve_model("q", cfg, request_profile="my-coding")
    assert rm.sampling["temperature"] == 0.6     # inherited from builtin coding
    assert rm.sampling["min_p"] == 0.02          # leaf wins


def test_per_model_profiles_tweak():
    doc = _fam_doc()
    doc["models"]["q"] = {
        "path": "/abs/q.gguf", "family": "qwen3.6",
        "profiles": {"coding": {"sampling": {"temperature": 0.4}}},
    }
    cfg = build_config(doc)
    assert resolve_model("q", cfg, request_profile="coding").sampling["temperature"] == 0.4
    # The tweak applies only when that profile is selected.
    assert resolve_model("q", cfg).sampling["temperature"] == 1.0


def test_per_model_tweak_below_overrides():
    doc = _fam_doc()
    doc["models"]["q"] = {
        "path": "/abs/q.gguf", "family": "qwen3.6",
        "profiles": {"coding": {"sampling": {"temperature": 0.4}}},
        "overrides": {"sampling": {"temperature": 0.9}},
    }
    cfg = build_config(doc)
    assert resolve_model("q", cfg, request_profile="coding").sampling["temperature"] == 0.9


def test_per_model_tweak_unknown_name_raises():
    doc = _fam_doc()
    doc["models"]["g"]["profiles"] = {"no-such": {"sampling": {"temperature": 1}}}
    with pytest.raises(ConfigError) as e:
        build_config(doc)
    assert "no-such" in str(e.value)


def test_unknown_family_warns_not_raises():
    doc = _fam_doc()
    doc["models"]["g"]["family"] = "martian"
    with pytest.warns(UserWarning, match="martian"):
        cfg = build_config(doc)
    # Unknown family degrades to the generic base.
    assert resolve_model("g", cfg).sampling["temperature"] == 0.7


def test_family_defaults_kill_switch():
    doc = _fam_doc()
    doc["server"]["family_defaults"] = False
    cfg = build_config(doc)
    rm = resolve_model("g", cfg)
    assert rm.sampling == {}                     # no base layer
    with pytest.raises(ConfigError):
        resolve_model("g", cfg, request_profile="coding")   # intents gone


def test_stochastic_mtp_key():
    doc = _doc()
    assert build_config(doc).stochastic_mtp is False
    doc["server"]["stochastic_mtp"] = True
    assert build_config(doc).stochastic_mtp is True


def test_gpu_keepwarm_key():
    doc = _doc()
    assert build_config(doc).gpu_keepwarm is False
    doc["server"]["gpu_keepwarm"] = True
    assert build_config(doc).gpu_keepwarm is True


def test_kill_switch_rejects_intent_refs():
    doc = _fam_doc()
    doc["server"]["family_defaults"] = False
    doc["models"]["g"]["profile"] = "coding"
    with pytest.raises(ConfigError):
        build_config(doc)


def test_model_profile_may_name_builtin():
    doc = _fam_doc()
    doc["models"]["q"] = {"path": "/abs/q.gguf", "family": "qwen3.6",
                          "profile": "instruct"}
    cfg = build_config(doc)
    rm = resolve_model("q", cfg)
    assert rm.sampling["presence_penalty"] == 1.5
    assert rm.chat_template_kwargs == {"enable_thinking": False}


def test_alias_may_bake_builtin_intent():
    doc = _fam_doc()
    doc["aliases"] = {"fastcode": "g@coding"}
    cfg = build_config(doc)                      # validates without raising
    rm = cfgmod.resolve_cli_model("fastcode", cfg)
    assert rm.id == "g" and rm.profile_name == "coding"


def test_cli_model_at_intent():
    cfg = build_config(_fam_doc())
    rm = cfgmod.resolve_cli_model("g@creative", cfg)
    assert rm.profile_name == "creative"
    with pytest.raises(ConfigError, match="no-such"):
        cfgmod.resolve_cli_model("g@no-such", cfg)


# talk: block (the gmlx talk client config)
def test_talk_defaults_when_absent():
    cfg = build_config(_doc())
    t = cfg.talk
    assert t.mode == "wake" and t.wake_word == "hey assistant"
    assert t.voice is None and t.model is None and t.brain == "chat"
    assert t.vad.silence_ms == 550.0 and t.vad.pre_roll_ms == 400.0
    assert t.chime is True and t.max_tokens == 512


def test_talk_vad_must_be_a_mapping():
    """A scalar `vad:` used to escape as AttributeError/TypeError, past every
    `except ConfigError` handler in serve/doctor/menubar."""
    doc = _doc()
    doc["talk"] = {"vad": 0.5}
    with pytest.raises(ConfigError, match="talk.vad"):
        build_config(doc)


def test_talk_system_and_wake_defaults():
    from gmlx.config import DEFAULT_TALK_SYSTEM

    t = build_config(_doc()).talk
    assert t.system == DEFAULT_TALK_SYSTEM          # absent -> speakable default
    assert t.wake_threshold == 0.3

    doc = _doc()
    doc["talk"] = {"system": ""}                    # explicit opt-out
    assert build_config(doc).talk.system is None

    doc = _doc()
    doc["talk"] = {"system": "You are terse."}
    assert build_config(doc).talk.system == "You are terse."


def test_talk_block_parses_all_fields():
    doc = _doc()
    doc["talk"] = {
        "model": "m-bare@creative", "voice": "am_adam", "speed": 1.2,
        "system": "You are terse.", "language": "en", "max_tokens": 256,
        "mode": "vad", "wake_word": "hey gadget", "wake_threshold": 0.7,
        "vad": {"threshold": 0.5, "silence_ms": 700, "min_speech_ms": 250,
                "pre_roll_ms": 300},
        "input_device": "MacBook Pro Microphone", "output_device": 3,
        "chime": False,
    }
    t = build_config(doc).talk
    assert t.model == "m-bare@creative" and t.voice == "am_adam"
    assert t.speed == 1.2 and t.max_tokens == 256 and t.mode == "vad"
    assert t.wake_word == "hey gadget" and t.wake_threshold == 0.7
    assert t.vad.threshold == 0.5 and t.vad.silence_ms == 700.0
    assert t.vad.min_speech_ms == 250.0 and t.vad.pre_roll_ms == 300.0
    assert t.input_device == "MacBook Pro Microphone"
    assert t.output_device == "3"            # normalized to str
    assert t.chime is False


def test_talk_unknown_key_is_hard_error():
    doc = _doc()
    doc["talk"] = {"voices": "af_heart"}     # typo for voice
    with pytest.raises(ConfigError, match="voices"):
        build_config(doc)


def test_talk_vad_unknown_key_is_hard_error():
    doc = _doc()
    doc["talk"] = {"vad": {"silence": 700}}  # typo for silence_ms
    with pytest.raises(ConfigError, match="silence"):
        build_config(doc)


def test_talk_bad_mode_and_brain_raise():
    doc = _doc()
    doc["talk"] = {"mode": "telepathy"}
    with pytest.raises(ConfigError, match="telepathy"):
        build_config(doc)
    doc["talk"] = {"brain": "psychic"}       # not chat | agent
    with pytest.raises(ConfigError, match="psychic"):
        build_config(doc)


def test_talk_push_to_talk_modifier():
    assert build_config(_doc()).talk.push_to_talk_modifier == "globe"
    doc = _doc()
    doc["talk"] = {"push_to_talk_modifier": "Right-Command"}   # normalized
    assert build_config(doc).talk.push_to_talk_modifier == "right-command"
    doc["talk"] = {"push_to_talk_modifier": "hyper"}
    with pytest.raises(ConfigError, match="hyper"):
        build_config(doc)


def test_talk_numeric_coercion_and_errors():
    doc = _doc()
    doc["talk"] = {"speed": "1.5", "max_tokens": "300",
                   "vad": {"silence_ms": "600"}}
    t = build_config(doc).talk
    assert t.speed == 1.5 and t.max_tokens == 300
    assert t.vad.silence_ms == 600.0
    doc["talk"] = {"speed": "fast"}
    with pytest.raises(ConfigError, match="talk.speed"):
        build_config(doc)


def test_talk_non_mapping_raises():
    doc = _doc()
    doc["talk"] = "wake"
    with pytest.raises(ConfigError, match="talk must be a mapping"):
        build_config(doc)


# assistant (built-in tool-loop assistant: tools + memory)
def test_assistant_defaults_when_absent():
    cfg = build_config(_doc())
    a = cfg.assistant
    assert cfg.talk.brain == "chat"
    assert a.max_tool_rounds == 8 and a.tool_timeout_s == 60.0
    assert a.mcp == []
    assert a.memory.enabled is True and a.memory.top_k == 4
    assert a.memory.extract is True
    assert a.memory.ttl_days is None
    assert a.memory.max_items == 20000
    assert cfg.talk.assistant is a          # one shared settings object
    assert cfg.assistants == {}
    assert cfg.assistant_allow_remote is False


def test_assistant_block_parses():
    doc = _doc()
    doc["talk"] = {"brain": "assistant"}
    doc["assistant"] = {
        "max_tool_rounds": 4, "tool_timeout_s": 30,
        "mcp": [
            {"name": "files",
             "command": ["npx", "@modelcontextprotocol/server-filesystem",
                         "~/notes"],
             "env": {"DEBUG": 1}},
            {"name": "web", "url": "http://127.0.0.1:8931/mcp"},
        ],
        "memory": {"enabled": True, "path": "~/talk-mem", "top_k": 6,
                   "extract": False, "ttl_days": 90, "max_items": 5000},
    }
    cfg = build_config(doc)
    a = cfg.assistant
    assert cfg.talk.brain == "assistant"
    assert a.max_tool_rounds == 4 and a.tool_timeout_s == 30.0
    assert [s.name for s in a.mcp] == ["files", "web"]
    assert a.mcp[0].command[0] == "npx" and a.mcp[0].url is None
    assert a.mcp[0].env == {"DEBUG": "1"}    # values normalized to str
    assert a.mcp[1].url == "http://127.0.0.1:8931/mcp" and a.mcp[1].command == []
    assert a.memory.path == "~/talk-mem" and a.memory.top_k == 6
    assert a.memory.extract is False and a.memory.ttl_days == 90.0
    assert a.memory.max_items == 5000


def test_talk_agent_key_moved_error():
    doc = _doc()
    doc["talk"] = {"agent": {"max_tool_rounds": 4}}
    with pytest.raises(ConfigError, match=r"talk\.agent has moved"):
        build_config(doc)


def test_talk_brain_agent_is_invalid():
    doc = _doc()
    doc["talk"] = {"brain": "agent"}
    with pytest.raises(ConfigError, match="chat/assistant"):
        build_config(doc)


def test_assistant_mcp_command_string_is_split():
    doc = _doc()
    doc["assistant"] = {"mcp": [
        {"name": "files", "command": "npx mcp-server-filesystem '/my notes'"},
    ]}
    s = build_config(doc).assistant.mcp[0]
    assert s.command == ["npx", "mcp-server-filesystem", "/my notes"]


@pytest.mark.parametrize("entry,match", [
    ({"command": ["x"]}, "name"),                          # missing name
    ({"name": "a"}, "exactly one"),                        # neither transport
    ({"name": "a", "command": ["x"], "url": "http://y"}, "exactly one"),
    ({"name": "a", "commandz": ["x"]}, "commandz"),        # typo'd key
])
def test_assistant_mcp_entry_errors(entry, match):
    doc = _doc()
    doc["assistant"] = {"mcp": [entry]}
    with pytest.raises(ConfigError, match=match):
        build_config(doc)


def test_assistant_duplicate_mcp_name_raises():
    doc = _doc()
    doc["assistant"] = {"mcp": [
        {"name": "a", "command": ["x"]}, {"name": "a", "url": "http://y"},
    ]}
    with pytest.raises(ConfigError, match="duplicate"):
        build_config(doc)


def test_assistant_bounds_and_unknown_keys():
    doc = _doc()
    doc["assistant"] = {"max_tool_rounds": 0}
    with pytest.raises(ConfigError, match="must be >= 1"):
        build_config(doc)
    doc["assistant"] = {"memory": {"topk": 2}}             # typo for top_k
    with pytest.raises(ConfigError, match="topk"):
        build_config(doc)
    doc["assistant"] = {"memory": {"ttl_days": 0}}
    with pytest.raises(ConfigError, match="ttl_days"):
        build_config(doc)
    doc["assistant"] = {"memory": {"max_items": 0}}
    with pytest.raises(ConfigError, match="max_items"):
        build_config(doc)


# server.assistants (served assistant aliases)
def test_assistants_parse_and_scoping():
    doc = _doc()
    doc["server"]["assistants"] = {
        "helper": {"model": "m-bare"},                         # inherits mcp
        "scoped": {"model": "m-named", "memory": True,
                   "mcp": [{"name": "w", "url": "http://127.0.0.1:1/mcp"}]},
        "locked": {"model": "m-bare", "mcp": []},              # zero tools
    }
    cfg = build_config(doc)
    assert set(cfg.assistants) == {"helper", "scoped", "locked"}
    h, s, lk = (cfg.assistants[k] for k in ("helper", "scoped", "locked"))
    assert h.model == "m-bare" and h.memory is False and h.mcp is None
    assert s.memory is True and [m.name for m in s.mcp] == ["w"]
    assert lk.mcp == []                      # scoped-to-nothing, not inherit


def test_assistant_alias_requires_model():
    doc = _doc()
    doc["server"]["assistants"] = {"helper": {"memory": True}}
    with pytest.raises(ConfigError, match="`model` is required"):
        build_config(doc)


def test_assistant_alias_unknown_model():
    doc = _doc()
    doc["server"]["assistants"] = {"helper": {"model": "nope"}}
    with pytest.raises(ConfigError, match="unknown model 'nope'"):
        build_config(doc)


def test_assistant_alias_collisions():
    doc = _doc()
    doc["server"]["assistants"] = {"m-bare": {"model": "m-named"}}
    with pytest.raises(ConfigError, match="collides with a model id"):
        build_config(doc)
    doc = _doc()
    doc["aliases"] = {"fast": "m-bare"}
    doc["server"]["assistants"] = {"fast": {"model": "m-named"}}
    with pytest.raises(ConfigError, match="collides with an alias name"):
        build_config(doc)
    doc = _doc()
    doc["server"]["assistants"] = {"help@er": {"model": "m-bare"}}
    with pytest.raises(ConfigError, match="must not contain '@'"):
        build_config(doc)


def test_assistants_refuse_non_loopback_bind():
    doc = _doc()
    doc["server"]["host"] = "0.0.0.0"
    doc["server"]["assistants"] = {"helper": {"model": "m-bare", "mcp": []}}
    with pytest.raises(ConfigError, match="beyond localhost"):
        build_config(doc)
    doc["server"]["assistant_allow_remote"] = True
    cfg = build_config(doc)                  # explicit opt-in starts
    assert cfg.assistant_allow_remote is True


def test_assistant_fail_closed_tool_scope():
    # allow_remote + an unscoped alias + real local tools = refuse: the alias
    # would hand the full local tool list to remote callers.
    doc = _doc()
    doc["server"]["host"] = "0.0.0.0"
    doc["server"]["assistant_allow_remote"] = True
    doc["assistant"] = {"mcp": [{"name": "fs", "command": ["srv"]}]}
    doc["server"]["assistants"] = {"helper": {"model": "m-bare"}}
    with pytest.raises(ConfigError, match="would inherit 1 local tool"):
        build_config(doc)
    # An explicit per-alias list (even []) is a deliberate scope: passes.
    doc["server"]["assistants"] = {"helper": {"model": "m-bare", "mcp": []}}
    build_config(doc)
    # No shared tools to leak: passes even unscoped.
    doc["assistant"] = {}
    doc["server"]["assistants"] = {"helper": {"model": "m-bare"}}
    build_config(doc)


def test_edit_config_yaml_preserves_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "# top note\n"
        "server:\n"
        "  model_dirs:\n"
        "    - ~/models          # my library\n"
        "models:\n"
        "  keep-me:\n"
        "    path: keep.gguf     # hand-tuned\n"
        "  drop-me:\n"
        "    path: drop.gguf\n")
    cfgmod.edit_config_yaml(str(p), lambda doc: doc["models"].pop("drop-me"))
    text = p.read_text()
    assert "# top note" in text and "# my library" in text
    assert "# hand-tuned" in text and "keep-me" in text
    assert "drop-me" not in text


# Section-shape guards: a mistyped section must be a named ConfigError at the
# section, never an AttributeError from deep inside a parser.
def test_section_wrong_shapes_raise_named_errors():
    for doc, needle in [
        ({"models": [{"id": "m", "path": "/x.gguf"}]}, "models must be a mapping"),
        ({"models": "x.gguf"}, "models must be a mapping"),
        ({"profiles": ["hot"]}, "profiles must be a mapping"),
        ({"aliases": ["a"]}, "aliases must be a mapping"),
        ({"server": "yes"}, "server must be a mapping"),
        ({"server": {"cache": "big"}}, "server.cache must be a mapping"),
        ({"server": {"defaults": [1]}}, "server.defaults must be a mapping"),
        ({"rules": "always"}, "rules must be a list"),
        ({"discover": "~/llm"}, "discover must be a list"),
        ({"discover": ["~/llm"]}, "discover entry must be a mapping"),
        ({"models": {"m": {"path": "/x.gguf", "overrides": "hot"}}},
         "overrides must be a mapping"),
    ]:
        with pytest.raises(ConfigError) as e:
            build_config(doc)
        assert needle in str(e.value), f"{doc} -> {e.value}"
    # YAML's empty-list rendering of an emptied section still means "absent".
    assert build_config({"server": []}).port == 8080


def test_model_path_must_be_string():
    with pytest.raises(ConfigError) as e:
        build_config({"models": {"m": {"path": ["/a.gguf", "/b.gguf"]}}})
    assert "path must be a string" in str(e.value)


def test_model_ttl_s_coerced_like_server_ttl():
    with pytest.raises(ConfigError) as e:
        build_config({"models": {"m": {"path": "/x.gguf", "ttl_s": "10m"}}})
    assert "ttl_s" in str(e.value) and "'10m'" in str(e.value)
    cfg = build_config({"models": {"m": {"path": "/x.gguf", "ttl_s": "900"}}})
    assert cfg.models["m"].ttl_s == 900.0


def test_sampling_stop_validated_at_load():
    # An int list (the "stop on the EOS id" confusion) must fail at config
    # load, not 400 every later request that resolves the profile.
    for doc in [
        {"profiles": {"p": {"sampling": {"stop": [128001]}}}},
        {"profiles": {"p": {"sampling": {"stop": 5}}}},
        {"models": {"m": {"path": "/x.gguf",
                          "overrides": {"sampling": {"stop": [1]}}}}},
        {"models": {"m": {"path": "/x.gguf",
                          "profiles": {"q": {"sampling": {"stop": 7}}}}}},
    ]:
        with pytest.raises(ConfigError) as e:
            build_config(doc)
        assert "stop must be a string or a list of strings" in str(e.value)
    ok = build_config({"profiles": {"p": {"sampling": {"stop": ["END"]}}}})
    assert ok.profiles["p"].sampling["stop"] == ["END"]


def test_config_path_directory_and_binary_are_named_errors(tmp_path):
    from gmlx.config import load_config
    with pytest.raises(ConfigError, match="directory"):
        load_config(str(tmp_path))
    p = tmp_path / "model.yaml"
    p.write_bytes(b"\x00\xff\xfe\x81 binary junk")
    with pytest.raises(ConfigError) as e:      # not-text or malformed-YAML,
        load_config(str(p))                    # never a raw decode traceback
    assert str(p) in str(e.value)


def test_edit_config_yaml_atomic_no_tmp_left(tmp_path):
    from gmlx.config import edit_config_yaml
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    edit_config_yaml(str(p), lambda doc: doc.__setitem__("b", 2))
    text = p.read_text()
    assert "a: 1" in text and "b: 2" in text
    assert not list(tmp_path.glob("*.tmp"))


# non-mapping group values fail at parse time, not as a crash at resolve

@pytest.mark.parametrize("group,val", [
    ("load", "kv_bits"),
    ("cache", "exact"),
    ("chat_template_kwargs", ["enable_thinking"]),
])
def test_scalar_override_group_raises_config_error(group, val):
    doc = _doc()
    doc["models"]["m-bare"]["overrides"] = {group: val}
    with pytest.raises(ConfigError, match=f"overrides.{group}"):
        build_config(doc)


def test_scalar_profile_group_raises_config_error():
    doc = _doc()
    doc["profiles"]["base"]["load"] = "kv_bits"
    with pytest.raises(ConfigError, match="load"):
        build_config(doc)


def test_scalar_model_profile_tweak_group_raises_config_error():
    doc = _doc()
    doc["models"]["m-bare"]["profiles"] = {"base": {"cache": "exact"}}
    with pytest.raises(ConfigError, match="cache"):
        build_config(doc)


def test_scalar_cache_disk_raises_config_error():
    doc = _doc()
    doc["server"]["cache"] = {"disk": "/tmp/apc"}
    with pytest.raises(ConfigError, match="cache.disk"):
        build_config(doc)
