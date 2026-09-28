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


def test_no_vendored_module_shipped_upstream():
    hits = us.vendored_upstream_collisions()
    assert not hits, "\n".join(hits)


def test_fingerprint_drift_is_named(monkeypatch):
    # A seam over a symbol no gmlx installer ever patches, so the check
    # can't be skipped as "ours" when other tests installed patches first.
    seam = us.Seam("mlx_vlm.prompt_utils", "apply_chat_template", "test")
    key = us._key(seam)
    monkeypatch.setattr(us, "SEAMS", (seam,))
    monkeypatch.setattr(
        us, "load_pinned",
        lambda: {"generated_with": {}, "fingerprints": {key: "0" * 64}})
    problems = us.check_seams()
    assert any(key in p and "changed under the pin" in p for p in problems)


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
