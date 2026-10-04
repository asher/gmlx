#!/usr/bin/env python3
"""The upstream seam contract: every mlx-vlm/mlx-lm symbol gmlx patches
or deep-imports must exist and match its pinned source fingerprint. A failure
here names the exact seam that drifted - re-audit the using site, then
`python -m gmlx.upstream.seams --regen` after a deliberate bump."""
from __future__ import annotations

import importlib.metadata as md
import sys
import types

import pytest

import gmlx.upstream.seams as us


def _impl_a(x):
    return x


def _impl_b(x):
    return x + 1


def _probe_module(monkeypatch, **attrs):
    mod = types.ModuleType("_gmlx_seam_probe")
    for name, value in attrs.items():
        setattr(mod, name, value)
    monkeypatch.setitem(sys.modules, "_gmlx_seam_probe", mod)
    return mod


def test_all_seams_present_and_unchanged():
    problems = us.check_seams()
    assert not problems, "\n".join(problems)


def test_every_owned_name_resolves_to_its_gmlx_module():
    problems = us.owned_module_problems()
    assert not problems, "\n".join(problems)


def test_fingerprint_drift_is_named(monkeypatch):
    # A seam over a symbol no gmlx installer ever patches, so the check
    # can't be skipped as "ours" when other tests installed patches first.
    seam = us.Seam("mlx_vlm.prompt_utils", "_content_media_count", "test")
    key = us._key(seam)
    monkeypatch.setattr(us, "SEAMS", (seam,))
    monkeypatch.setattr(
        us, "load_pinned",
        lambda: {"generated_with": {}, "fingerprints": {key: "0" * 64}})
    problems = us.check_seams()
    assert any(key in p and "changed under the pin" in p for p in problems)


def test_a_removed_row_is_named(monkeypatch):
    """A row dropped from SEAMS leaves its pin in seams.json, and the check
    names it, so a seam cannot stop being checked unnoticed."""
    dropped = us.SEAMS[0]
    monkeypatch.setattr(us, "SEAMS", us.SEAMS[1:])
    problems = us.check_seams()
    assert any(p.startswith(us._key(dropped) + ":") and "no row" in p
               for p in problems)


def test_every_seam_has_a_pin():
    # Values without source (constants, instances, properties) pin their
    # value; a null pin would let any drift under it pass.
    fps = us.load_pinned()["fingerprints"]
    assert None not in fps.values()


def test_values_without_source_get_distinct_pins():
    assert us._fingerprint(16) != us._fingerprint(2048)
    assert (us._fingerprint(types.SimpleNamespace(a=_impl_a))
            != us._fingerprint(types.SimpleNamespace(b=_impl_a)))
    assert (us._fingerprint(property(_impl_a))
            != us._fingerprint(property(_impl_b)))
    assert us._fingerprint(object()) == us._fingerprint(object())


def test_forwarding_lambda_needs_impl(monkeypatch):
    _probe_module(monkeypatch, fwd=lambda *a: _impl_a(*a))
    seam = us.Seam("_gmlx_seam_probe", "fwd", "test")
    monkeypatch.setattr(us, "SEAMS", (seam,))
    monkeypatch.setattr(
        us, "load_pinned", lambda: {"generated_with": {}, "fingerprints": {}})
    problems = us.check_seams()
    assert any("_gmlx_seam_probe:fwd" in p and "forwarding lambda" in p
               for p in problems)
    with pytest.raises(RuntimeError, match="forwarding lambda"):
        us.collect_fingerprints()


def test_forwarding_seam_pins_its_implementation(monkeypatch):
    mod = _probe_module(monkeypatch, fwd=lambda *a: mod.impl(*a),
                        impl=_impl_a)
    seam = us.Seam("_gmlx_seam_probe", "fwd", "test",
                   impl="_gmlx_seam_probe:impl")
    monkeypatch.setattr(us, "SEAMS", (seam,))
    pinned = us.collect_fingerprints()
    monkeypatch.setattr(
        us, "load_pinned",
        lambda: {"generated_with": {}, "fingerprints": pinned})
    assert us.check_seams() == []
    mod.impl = _impl_b
    problems = us.check_seams()
    assert any("changed under the pin" in p for p in problems)
    del mod.impl
    problems = us.check_seams()
    assert any("implementation _gmlx_seam_probe:impl missing" in p
               for p in problems)


_TOOL_TABLES = ("mlx_vlm.tools.registry:SPECS",
                "mlx_vlm.tools.registry:_BY_NAME", "mlx_vlm.tools:SPECS")


def _tool_table_problems():
    return sorted(k for k in _TOOL_TABLES
                  if any(p.startswith(k + ":") for p in us.check_seams()))


def _pin_tool_tables(monkeypatch):
    import mlx_vlm.tools as tools
    import mlx_vlm.tools.registry as reg

    monkeypatch.setattr(reg, "SPECS", reg.SPECS)
    monkeypatch.setattr(reg, "_BY_NAME", dict(reg._BY_NAME))
    monkeypatch.setattr(tools, "SPECS", tools.SPECS)
    return reg, tools


def test_tool_parser_tables_pin_the_stock_specs(monkeypatch):
    """An upstream spec added to the tool-parser table changes which parser
    a template routes to, so the three table rows trip."""
    reg, tools = _pin_tool_tables(monkeypatch)
    assert _tool_table_problems() == []
    spec = reg.ParserSpec("atem", (("<atem:invoke",),), priority=300)
    reg.SPECS = tuple(x for x in reg.SPECS if x.name != "atem") + (spec,)
    reg._BY_NAME["atem"] = spec
    tools.SPECS = reg.SPECS
    assert _tool_table_problems() == sorted(_TOOL_TABLES)


def test_tool_parser_tables_leave_out_gmlx_specs(monkeypatch):
    import mlx_vlm.tools.parsers as parsers

    import gmlx.upstream.tool_parsers as tp

    reg, _ = _pin_tool_tables(monkeypatch)
    monkeypatch.setattr(tp, "_REGISTERED", set(tp._REGISTERED))
    probe = types.ModuleType("_gmlx_seam_probe_parser")
    monkeypatch.setitem(sys.modules, "mlx_vlm.tools.parsers._gmlx_seam_probe",
                        probe)
    monkeypatch.setattr(parsers, "_gmlx_seam_probe", probe, raising=False)
    tp.register_tool_parser("_gmlx_seam_probe", probe, (("<probe>",),))
    assert reg.SPECS[0].name == "_gmlx_seam_probe"
    assert _tool_table_problems() == []


def test_missing_attr_is_named(monkeypatch):
    bogus = us.Seam("mlx_vlm.generate.ar", "definitely_not_a_thing",
                    "test", critical=True)
    monkeypatch.setattr(us, "SEAMS", us.SEAMS + (bogus,))
    problems = us.check_seams()
    assert any("definitely_not_a_thing" in p and "missing" in p
               for p in problems)


def test_version_gate_below_floor_raises(monkeypatch):
    real = md.version

    def fake(pkg):
        return "0.30.7" if pkg == "mlx-lm" else real(pkg)

    monkeypatch.setattr(us.sys.modules["importlib.metadata"], "version", fake)
    with pytest.raises(RuntimeError, match="mlx-lm 0.30.7.*floor"):
        us.check_upstream_versions(quiet=True)


def test_version_gate_newer_than_qualified_warns(monkeypatch):
    real = md.version

    def fake(pkg):
        return "99.0.0" if pkg == "mlx-vlm" else real(pkg)

    monkeypatch.setattr(us.sys.modules["importlib.metadata"], "version", fake)
    warnings = us.check_upstream_versions(quiet=True)
    assert any("mlx-vlm 99.0.0" in w and "newer than the qualified" in w
               for w in warnings)


def test_version_gate_missing_metadata_warns_not_raises(monkeypatch):
    real = md.version

    def fake(pkg):
        if pkg == "mlx-lm":
            raise md.PackageNotFoundError(pkg)
        return real(pkg)

    monkeypatch.setattr(us.sys.modules["importlib.metadata"], "version", fake)
    warnings = us.check_upstream_versions(quiet=True)
    assert any("mlx-lm" in w and "source install" in w for w in warnings)
