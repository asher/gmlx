"""scripts/drift_report.py: the registry, the unit split and who changed
each unit. The installed mlx-vlm is read only when it is the baseline
version; the split is checked on a small synthetic package. CPU."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "drift_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("_drift_report", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # the dataclass looks its module up
    spec.loader.exec_module(mod)
    return mod


dr = _load()


def test_each_registered_file_has_its_baseline():
    assert {e.path for e in dr.REGISTRY} >= {"gmlx/cache/kv_cache.py",
                                             "gmlx/cache/attention.py"}
    for e in dr.REGISTRY:
        assert (_ROOT / e.path).is_file(), e.path
        base = json.loads((_ROOT / e.baseline).read_text())
        assert e.source in base["sources"], e
        assert e.path in base["outputs"], e
        assert base["outputs"][e.path]["source"].startswith(e.source)


def test_units_split_classes_and_module_names():
    text = ('"""doc"""\nimport os\nX = 1\n\n\nclass A:\n    """A."""\n'
            "    k = 2\n\n    @property\n    def f(self):\n        return 1\n\n\n"
            "def g():\n    pass\n")
    u = dr.units(text)
    assert set(u) == {"<module>", "X", "A", "A.f", "g"}
    assert u["<module>"] == "import os\n"
    assert "@property" in u["A.f"]
    assert "k = 2" in u["A"] and "def f" not in u["A"]


_PROP = """class A:
    @property
    def p(self):
        return 1

    @p.setter
    def p(self, v):
        self.v = v

    @p.deleter
    def p(self):
        del self.v
"""


def test_units_key_a_setter_and_a_deleter_apart_from_the_getter():
    u = dr.units(_PROP)
    assert set(u) == {"<module>", "A", "A.p", "A.p.setter", "A.p.deleter"}
    assert "return 1" in u["A.p"]
    assert "self.v = v" in u["A.p.setter"]
    assert "del self.v" in u["A.p.deleter"]


_OLD = """import mlx

def keep():
    return 1

def up():
    return 1

def fold():
    return 1

def both():
    return 1

def gone():
    return 1

def text():
    return "a \\u2014 b"
"""

_NEW = _OLD.replace("def up():\n    return 1", "def up():\n    return 2") \
    .replace("def both():\n    return 1", "def both():\n    return 2") \
    .replace("def gone():\n    return 1\n\n", "") + "\ndef fresh():\n    return 1\n"

_OWNED = _OLD.replace("def fold():\n    return 1", "def fold():\n    return 3") \
    .replace("def both():\n    return 1", "def both():\n    return 3") \
    .replace("\\u2014", "--") + "\ndef ours():\n    return 1\n"


_EDITS = [{"old": "\\u2014", "new": "--", "count": 1}]


def _tree(tmp_path, new_text, old_text=_OLD, owned_text=_OWNED, edits=_EDITS):
    pkg_old, pkg_new = tmp_path / "old" / "pkg", tmp_path / "new" / "pkg"
    for pkg, text in ((pkg_old, old_text), (pkg_new, new_text)):
        (pkg / "models").mkdir(parents=True)
        (pkg / "models" / "m.py").write_text(text)
        (pkg / "version.py").write_text('__version__ = "1.0"\n')
    (tmp_path / "owned.py").write_text(owned_text)
    (tmp_path / "base.json").write_text(json.dumps({
        "version": "1.0",
        "sources": {"models/m.py": {"sha256": dr.sha256(old_text)}},
        "outputs": {"owned.py": {"source": "models/m.py", "edits": edits}},
    }))
    return dr.Owned("owned.py", "pkg", "models/m.py", "base.json"), \
        pkg_old, pkg_new


def test_the_baseline_source_splits_upstream_from_gmlx(tmp_path):
    entry, old, new = _tree(tmp_path, _NEW)
    r = dr.report(entry, new, old, root=tmp_path)
    rows = {row["unit"]: row["change"] for row in r["rows"]}
    assert rows == {"up": "upstream", "both": "both", "fold": "gmlx",
                    "fresh": "upstream added", "gone": "upstream removed",
                    "ours": "gmlx added"}
    assert r["split"] and not r["new"]["unchanged"] and not r["stale_edits"]


@pytest.mark.parametrize("old,new,unit", [
    ("        return 1\n", "        v = 1\n        return v\n", "A.p"),
    ("        self.v = v\n", "        self.v = int(v)\n", "A.p.setter"),
    ("        del self.v\n", "        self.v = None\n", "A.p.deleter"),
], ids=["getter", "setter", "deleter"])
def test_an_upstream_property_change_shows_as_its_own_row(tmp_path, old, new,
                                                          unit):
    entry, pkg_old, pkg_new = _tree(tmp_path, _PROP.replace(old, new),
                                    old_text=_PROP, owned_text=_PROP, edits=[])
    r = dr.report(entry, pkg_new, pkg_old, root=tmp_path)
    assert r["rows"] == [{"unit": unit, "change": "upstream"}]


def test_an_unchanged_upstream_needs_no_old_source(tmp_path):
    entry, _, new = _tree(tmp_path, _OLD)
    r = dr.report(entry, new, None, root=tmp_path)
    rows = {row["unit"]: row["change"] for row in r["rows"]}
    assert rows == {"fold": "gmlx", "both": "gmlx", "ours": "gmlx added"}
    assert r["split"] and r["new"]["unchanged"]


def test_without_the_old_source_the_rows_are_unsplit(tmp_path):
    entry, _, new = _tree(tmp_path, _NEW.replace("\\u2014", "-"))
    r = dr.report(entry, new, None, root=tmp_path)
    assert not r["split"]
    assert r["stale_edits"] == ["\\u2014"]
    assert {row["change"] for row in r["rows"]} <= {"changed",
                                                    "added or removed"}


def test_an_old_source_that_is_not_the_baseline_is_refused(tmp_path):
    entry, old, new = _tree(tmp_path, _NEW)
    (old / "models" / "m.py").write_text(_OLD + "\n")
    with pytest.raises(dr.Mismatch):
        dr.report(entry, new, old, root=tmp_path)


def test_the_installed_baseline_version_shows_only_gmlx_changes():
    try:
        pkg = dr.package_dir("gmlx._vendor.mlx_vlm")
    except dr.Mismatch:
        pytest.skip("mlx-vlm is not installed")
    for e in dr.REGISTRY:
        base = json.loads((_ROOT / e.baseline).read_text())
        if dr.package_version(pkg) != base["version"]:
            pytest.skip("the installed mlx-vlm is not the baseline version")
        r = dr.report(e, pkg, None)
        assert r["new"]["unchanged"] and not r["stale_edits"], e.path
        kinds = {row["change"] for row in r["rows"]}
        assert kinds <= {"gmlx", "gmlx added"}, (e.path, kinds)
