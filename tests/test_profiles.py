#!/usr/bin/env python3
"""Built-in family profile table: shape invariants, arch mapping, intent
fallback, and the describe() surface the docs test keys off. Pure CPU."""
from __future__ import annotations

import pytest

from gmlx import profiles
from gmlx.config import SAMPLING_KEYS
from gmlx.config_synth import GGUF_ARCH_TO_MODEL_TYPE


# Table shape
def test_every_arch_is_a_real_gguf_arch():
    """A typo'd arch string would silently never match - tripwire against the
    loader's own arch table."""
    known = set(GGUF_ARCH_TO_MODEL_TYPE)
    for key, fam in profiles.FAMILIES.items():
        unknown = set(fam["arches"]) - known
        assert not unknown, f"family {key!r} maps unknown arches {sorted(unknown)}"


def test_arches_disjoint_across_families():
    seen: dict = {}
    for key, fam in profiles.FAMILIES.items():
        for arch in fam["arches"]:
            assert arch not in seen, f"{arch} in both {seen[arch]} and {key}"
            seen[arch] = key


def test_group_sampling_keys_are_valid():
    """Every sampling key any family/intent sets must be a real profile
    sampling key, or resolve_model would inject a knob the server drops."""
    for row in profiles.describe():
        groups = [row["base"], *row["intents"].values()]
        for g in groups:
            bad = set(g.get("sampling", {})) - SAMPLING_KEYS
            assert not bad, f"family {row['family']!r} sets unknown {sorted(bad)}"


def test_group_shapes():
    for key, fam in profiles.FAMILIES.items():
        assert set(fam) == {"label", "arches", "base", "intents"}
        for g in (fam["base"], *fam["intents"].values()):
            assert set(g) <= {"sampling", "chat_template_kwargs"}, (key, g)


def test_default_family_defines_core_intents():
    assert {"coding", "creative", "instruct"} <= set(
        profiles.FAMILIES["default"]["intents"])
    assert {"coding", "creative", "instruct",
            "reasoning-low", "reasoning-medium",
            "reasoning-high"} <= profiles.BUILTIN_INTENTS


def test_families_never_set_max_tokens():
    for row in profiles.describe():
        for g in (row["base"], *row["intents"].values()):
            assert "max_tokens" not in g.get("sampling", {})


# detect_family
@pytest.mark.parametrize("arch,family", [
    ("qwen35", "qwen3.6"), ("qwen35moe", "qwen3.6"), ("qwen3next", "qwen3.6"),
    ("qwen3", "qwen3"), ("qwen3moe", "qwen3"),
    ("qwen2", "qwen2.5"),
    ("gemma4", "gemma"), ("gemma3", "gemma"), ("diffusion-gemma", "gemma"),
    ("gpt-oss", "gpt-oss"),
    ("glm4moe", "glm"), ("glm-dsa", "glm"),
    ("deepseek2", "deepseek"), ("deepseek4", "deepseek"),
    ("minimax-m2", "minimax"),
    ("minimax-m3", "minimax"),
    ("nemotron_h_moe", "nemotron"),
    ("hunyuan-moe", "hunyuan"),
    ("llama", "llama"), ("smollm3", "llama"),
    ("mistral3", "mistral"),
    ("phi3", "default"),          # no card entry -> generic
    ("no-such-arch", "default"),
    (None, "default"),
    ("", "default"),
])
def test_detect_family(arch, family):
    assert profiles.detect_family(arch) == family


def test_detect_family_ignores_name_without_refinements():
    assert profiles.detect_family("gemma4", name="whatever") == "gemma"


# Intent resolution / fallback
def test_every_family_resolves_every_intent():
    for key in profiles.FAMILIES:
        intents = profiles.family_intents(key)
        assert set(intents) == set(profiles.BUILTIN_INTENTS)


def test_intent_without_card_value_resolves_to_base():
    """Gemma defines no coding point on purpose (the card says low temperature
    is harmful) - @coding on a gemma model must equal its base."""
    assert profiles.groups_for("gemma", "coding") == profiles.groups_for("gemma")
    assert profiles.groups_for("gemma")["sampling"]["temperature"] == 1.0
    assert profiles.groups_for("gemma")["sampling"]["top_k"] == 64


def test_intent_delta_merges_over_base():
    g = profiles.groups_for("qwen3.6", "coding")
    assert g["sampling"]["temperature"] == 0.6
    assert g["sampling"]["top_p"] == 0.95      # inherited from base
    g = profiles.groups_for("qwen3.6", "instruct")
    assert g["sampling"]["presence_penalty"] == 1.5
    assert g["chat_template_kwargs"] == {"enable_thinking": False}


def test_gpt_oss_reasoning_intents():
    base = profiles.groups_for("gpt-oss")
    assert base["sampling"] == {"temperature": 1.0, "top_p": 1.0}
    assert "top_k" not in base["sampling"]     # card: disable top_k
    hi = profiles.groups_for("gpt-oss", "reasoning-high")
    assert hi["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert hi["sampling"] == base["sampling"]


def test_unknown_family_and_intent_fall_back():
    assert profiles.family_base(None) == profiles.FAMILIES["default"]["base"]
    assert profiles.family_base("martian") == profiles.FAMILIES["default"]["base"]
    assert (profiles.groups_for("qwen3.6", "no-such-intent")
            == profiles.groups_for("qwen3.6"))


def test_default_coding_and_creative_deltas():
    assert profiles.groups_for("default", "coding")["sampling"]["temperature"] == 0.3
    creative = profiles.groups_for("default", "creative")["sampling"]
    assert creative["temperature"] == 1.0 and creative["min_p"] == 0.05


# describe()
def test_describe_complete():
    rows = profiles.describe()
    assert {r["family"] for r in rows} == set(profiles.FAMILIES)
    for r in rows:
        assert set(r["intents"]) == set(profiles.BUILTIN_INTENTS)
        assert r["label"]
