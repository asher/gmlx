#!/usr/bin/env python3
"""Tripwire for `docs/server-config.md`: every embedded YAML example must parse,
and the complete examples (first line `# doctest: build`) must build + validate
through the real loader - so a renamed key or changed default can't silently drift
the reference from the code. CPU-only; no model, no server."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

pytest.importorskip("yaml")
import yaml  # noqa: E402

from gmlx import config  # noqa: E402

_DOC = Path(__file__).resolve().parent.parent / "docs" / "server-config.md"
_FENCE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)


def _yaml_blocks() -> list:
    if not _DOC.is_file():
        return []
    return _FENCE.findall(_DOC.read_text())


def test_doc_exists():
    assert _DOC.is_file(), f"missing {_DOC}"


def test_has_yaml_examples():
    assert _yaml_blocks(), "no ```yaml examples found - extractor or doc drifted"


@pytest.mark.parametrize("block", _yaml_blocks())
def test_every_yaml_block_parses(block):
    """No syntactically broken YAML in the reference."""
    doc = yaml.safe_load(block)
    assert doc is None or isinstance(doc, (dict, list))


def test_full_config_examples_build_cleanly():
    """Blocks marked `# doctest: build` go through the real build_config under
    warnings-as-errors: an unknown/typo'd key (which the loader would warn about)
    fails the test, and so does any structural validation error."""
    built = 0
    for block in _yaml_blocks():
        if not block.lstrip().startswith("# doctest: build"):
            continue
        doc = yaml.safe_load(block)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # a typo'd key warning -> failure
            cfg = config.build_config(doc)
        assert isinstance(cfg, config.ServerCfg)
        built += 1
    assert built >= 1, "no `# doctest: build` example found to validate"


# --- drift tests: the doc's key tables / family table mirror the code ---

def _render_group(group: dict) -> str:
    """The doc family table's value renderer - must match how the table was
    generated so a value change in profiles.py fails this test."""
    parts = [f"{k}={v}" for k, v in (group.get("sampling") or {}).items()]
    parts += [f"{k}={v}"
              for k, v in (group.get("chat_template_kwargs") or {}).items()]
    return " ".join(parts)


def test_family_table_in_sync_with_profiles_py():
    """Every family row in the doc table is regenerated from profiles.describe()
    and must appear verbatim - a changed base value, arch list, or intent delta
    in code fails here until the doc row is updated."""
    from gmlx import profiles as fp
    doc = _DOC.read_text()
    for row in fp.describe():
        fam = row["family"]
        arches = ", ".join(f"`{a}`" for a in row["arches"]) or "*(anything else)*"
        base = _render_group(row["base"])
        own = fp.FAMILIES[fam]["intents"]
        ints = "; ".join(f"`@{n}`: {_render_group(fp.groups_for(fam, n))}"
                         for n in sorted(own)) or "-"
        line = f"| `{fam}` | {arches} | {base} | {ints} |"
        assert line in doc, f"family table row drifted for {fam!r}:\n{line}"


def _param_reference_ticks() -> set:
    """All `backticked` tokens inside the Param key reference section."""
    doc = _DOC.read_text()
    body = doc.split("## Param key reference", 1)[1].split("## Residency", 1)[0]
    return set(re.findall(r"`([A-Za-z_0-9.]+)`", body))


def test_sampling_keys_documented():
    missing = set(config.SAMPLING_KEYS) - _param_reference_ticks()
    assert not missing, f"sampling keys missing from the docs: {sorted(missing)}"


def test_load_keys_documented():
    ticks = _param_reference_ticks()
    missing = set(config.LOAD_ENV) - ticks
    assert not missing, f"load keys missing from the docs: {sorted(missing)}"
    missing_env = set(config.LOAD_ENV.values()) - ticks
    assert not missing_env, f"load env vars missing: {sorted(missing_env)}"


def test_cache_keys_documented():
    ticks = _param_reference_ticks()
    missing = set(config.CACHE_ENV) - ticks
    assert not missing, f"cache keys missing from the docs: {sorted(missing)}"
    missing_disk = set(config.CACHE_DISK_ENV) - ticks
    assert not missing_disk, f"cache disk keys missing: {sorted(missing_disk)}"


def test_builtin_intents_documented():
    """Every addressable built-in intent is named in the profiles section."""
    from gmlx import profiles as fp
    doc = _DOC.read_text()
    for name in fp.BUILTIN_INTENTS:
        assert f"@{name}" in doc, f"intent @{name} not documented"


# --- docs/talk.md: its YAML examples are full configs and must build too ---

_TALK_DOC = Path(__file__).resolve().parent.parent / "docs" / "talk.md"


def _talk_yaml_blocks() -> list:
    if not _TALK_DOC.is_file():
        return []
    return _FENCE.findall(_TALK_DOC.read_text())


def test_talk_doc_examples_build_cleanly():
    """Every talk.md YAML example (the server stt/tts lines, the full worked
    example, the talk: reference block) goes through the real build_config
    under warnings-as-errors, so a renamed talk key or drifted default fails
    here."""
    blocks = _talk_yaml_blocks()
    assert len(blocks) >= 2, "talk.md yaml examples missing - doc drifted"
    for block in blocks:
        doc = yaml.safe_load(block)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = config.build_config(doc)
        assert isinstance(cfg, config.ServerCfg)


# --- docs/assistant.md: its YAML examples are full configs and must build too ---

_ASSISTANT_DOC = Path(__file__).resolve().parent.parent / "docs" / "assistant.md"


def _assistant_yaml_blocks() -> list:
    if not _ASSISTANT_DOC.is_file():
        return []
    return _FENCE.findall(_ASSISTANT_DOC.read_text())


def test_assistant_doc_examples_build_cleanly():
    """Every assistant.md YAML example (the assistant: block reference and the
    served-assistants config) goes through the real build_config under
    warnings-as-errors, so a renamed assistant key or drifted default fails
    here."""
    blocks = _assistant_yaml_blocks()
    assert len(blocks) >= 2, "assistant.md yaml examples missing - doc drifted"
    for block in blocks:
        doc = yaml.safe_load(block)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = config.build_config(doc)
        assert isinstance(cfg, config.ServerCfg)
