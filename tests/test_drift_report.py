# vendor_upstream: skip
"""scripts/drift_report.py: the unit split, the file classes against the
manifest, and who changed each unit. The split is checked on a small
synthetic copy; the installed stock packages are read only when they are
the releases the copy holds. CPU."""
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


def test_each_extract_takes_named_units_of_a_copied_file():
    files = dr._tool().read_manifest(_ROOT)["files"]
    for ex in dr.EXTRACTS:
        assert f"{ex.package}/{ex.source}" in files, ex
        owned = dr.units((_ROOT / ex.path).read_text())
        renamed = dict(ex.renames)
        assert not [n for n in ex.names if renamed.get(n, n) not in owned], ex
        text = (_ROOT / dr.VENDOR / ex.package / ex.source).read_text()
        alias = dr._ALIAS.search(text)
        if alias is not None:  # the owned file of the alias uses the units
            assert files[f"{ex.package}/{ex.source}"]["status"] == "alias"
            owner = _ROOT / (alias.group(1).replace(".", "/") + ".py")
            assert Path(ex.path).stem in dr.units(owner.read_text())[
                "<module>"], ex
            assert not ex.mirror, ex
            continue
        copied = dr.units(text)
        if ex.mirror:  # the copy keeps its own
            assert not [n for n in ex.names if n not in copied], ex
            if not ex.names:
                assert set(owned) - {"<module>"} & set(copied), ex
        else:  # the copy imports the owned units
            by = dr.owners(text, _ROOT)
            assert not [n for n in ex.names if by.get(n) != ex.path], ex


def test_the_owned_and_mirrored_files_the_plan_names_have_entries():
    taken = {(ex.package, ex.source): ex for ex in dr.EXTRACTS}
    for package, source, path in [
            ("mlx_lm", "models/base.py", "gmlx/cache/attention.py"),
            ("mlx_lm", "models/cache.py", "gmlx/cache/kv_cache.py"),
            ("mlx_lm", "models/kimi_linear.py", "gmlx/models/kimi_k3.py"),
            ("mlx_lm", "models/gemma4_text.py",
             "gmlx/models/gemma4/lm_sanitize.py"),
            ("mlx_vlm", "apc.py", "gmlx/cache/apc_manager.py")]:
        assert [ex for ex in dr.EXTRACTS if (ex.package, ex.source, ex.path)
                == (package, source, path)], (package, source)
    assert taken["mlx_lm", "models/kimi_linear.py"].names == ("ShortConv1d",)
    assert taken["mlx_vlm", "apc.py"].renames == (
        ("from_env", "build_apc_manager"),)


def test_each_alias_of_the_manifest_names_its_owned_file():
    files = dr._tool().read_manifest(_ROOT)["files"]
    for package, source, path in [
            ("mlx_vlm", "models/cache.py", "gmlx/cache/kv_cache.py"),
            ("mlx_vlm", "models/qwen3_5/language.py",
             "gmlx/models/qwen35/language.py"),
            ("mlx_vlm", "models/qwen3_5/gated_delta.py",
             "gmlx/models/qwen35/gated_delta.py"),
            ("mlx_vlm", "models/qwen3_5_moe/language.py",
             "gmlx/models/qwen35/moe_language.py"),
            ("mlx_vlm", "models/gemma4/language.py",
             "gmlx/models/gemma4/language.py"),
            ("mlx_vlm", "models/gemma4_unified/gemma4_unified.py",
             "gmlx/models/gemma4/unified.py"),
            ("mlx_lm", "models/gemma4_text.py",
             "gmlx/models/gemma4/text.py")]:
        assert files[f"{package}/{source}"]["status"] == "alias", source
        m = dr._ALIAS.search(
            (_ROOT / dr.VENDOR / package / source).read_text())
        assert m and m.group(1).replace(".", "/") + ".py" == path, source
        assert not [ex for ex in dr.EXTRACTS if ex.mirror
                    and (ex.package, ex.source) == (package, source)], source


_METHOD = ("class Model:\n    k = 1\n\n    def sanitize(self, w):\n"
           "        return w\n\n    def other(self):\n        return 1\n")
_FUNCTION = "import os\n\n\ndef clean(self, w):\n    return w\n"


def test_a_method_held_as_a_module_function_gets_the_methods_row():
    ex = dr.Extract("gmlx/x.py", "mlx_lm", "models/m.py",
                    ("Model.sanitize",), (("Model.sanitize", "clean"),))
    old = _METHOD.encode()
    assert dr._unit_rows(_FUNCTION, old, old, ex=ex) == []
    new = _METHOD.replace("return w", "return dict(w)").encode()
    assert dr._unit_rows(_FUNCTION, new, old, ex=ex) == [
        {"unit": "Model.sanitize", "change": "upstream"}]
    assert dr._unit_rows(_FUNCTION.replace("return w", "return {}"),
                         new, old, ex=ex) == [
        {"unit": "Model.sanitize", "change": "both"}]
    other = _METHOD.replace("return 1", "return 2").encode()
    assert dr._unit_rows(_FUNCTION, other, old, ex=ex) == []


def test_the_gemma4_text_sanitize_is_the_recorded_upstream_method():
    ex, = [e for e in dr.EXTRACTS
           if e.path == "gmlx/models/gemma4/lm_sanitize.py"]
    files = dr._tool().read_manifest(_ROOT)["files"]
    old = (_ROOT / "tests/cache/upstream" / ex.package
           / ex.source).read_bytes()
    assert dr.sha256(old) == files[f"{ex.package}/{ex.source}"]["upstream"]
    owned = (_ROOT / ex.path).read_text()
    assert dr._unit_rows(owned, old, old, ex=ex) == []
    new = old.replace(b'"self_attn.rotary_emb",', b'"rotary_emb",')
    assert new != old
    assert dr._unit_rows(owned, new, old, ex=ex) == [
        {"unit": "Model.sanitize", "change": "upstream"}]


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


_OLD = """import mlx.core as mx
from mlx_vlm.utils import helper

def keep():
    return 1

def up():
    return 1

def fold():
    return 1

def gone():
    return 1

def both():
    return 1
"""

_NEW = _OLD.replace("def up():\n    return 1", "def up():\n    return 2") \
    .replace("def both():\n    return 1", "def both():\n    return 2") \
    .replace("def gone():\n    return 1\n\n", "") + "\ndef fresh():\n    return 1\n"

_OWNED = _OLD.replace("from mlx_vlm.", "from gmlx._vendor.mlx_vlm.") \
    .replace("def fold():\n    return 1", "def fold():\n    return 3  # gmlx:") \
    .replace("def both():\n    return 1", "def both():\n    return 3") \
    + "\ndef ours():\n    return 1\n"

_USED = "from mlx_vlm.models.m import keep\n\n\ndef f():\n    return 1\n"
_LICENSE = "MIT License\n"


def _write(base: Path, files: dict[str, str]):
    for rel, text in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(text)


def _tree(tmp_path, new_m=_NEW, old_m=_OLD, owned_m=_OWNED):
    """A copy of one package under tmp_path/gmlx/_vendor, its manifest, a
    gmlx module that imports used.py, and old and new releases. The old
    release is the recorded one; the new one edits models/m.py as given,
    changes used.py, plain.py and version.py, drops gone.py and adds
    fresh.py. Both wheels keep the LICENSE in their dist-info."""
    old = {"__init__.py": "", "models/m.py": old_m, "used.py": _USED,
           "plain.py": "X = 1\n", "same.py": "S = 1\n",
           "gone.py": "Y = 1\n", "skip.py": "Z = 1\n",
           "version.py": '__version__ = "1.0"\n'}
    new = dict(old, **{"models/m.py": new_m,
                       "used.py": _USED.replace("return 1", "return 2"),
                       "plain.py": "X = 2\n", "fresh.py": "W = 1\n",
                       "version.py": '__version__ = "1.1"\n'})
    del new["gone.py"]
    copy = {k: v for k, v in old.items() if k != "skip.py"}
    copy.update({"models/m.py": owned_m,
                 "used.py": dr._tool().rewrite_text(_USED),
                 "LICENSE": _LICENSE})
    status = {"models/m.py": "edited", "used.py": "codemod",
              "skip.py": "excluded"}
    for side, files in (("old", old), ("new", new)):
        _write(tmp_path / side / "mlx_vlm", files)
        _write(tmp_path / side / "mlx_vlm-1.0.dist-info",
               {"licenses/LICENSE": _LICENSE})
    _write(tmp_path / "gmlx" / "_vendor" / "mlx_vlm", copy)
    _write(tmp_path / "gmlx", {
        "user.py": "from gmlx._vendor.mlx_vlm.used import f\n"})
    entries = {f"mlx_vlm/{rel}": {"status": status.get(rel, "verbatim"),
                                  "upstream": dr.sha256(text.encode())}
               for rel, text in old.items()}
    entries["mlx_vlm/LICENSE"] = {"status": "license",
                                  "upstream": dr.sha256(_LICENSE.encode())}
    _write(tmp_path, {"gmlx/_vendor/manifest.json": json.dumps({
        "prefix": dr.PREFIX, "packages": {"mlx_vlm": {"version": "1.0"}},
        "files": entries})})
    return tmp_path / "old" / "mlx_vlm", tmp_path / "new" / "mlx_vlm"


def _report(tmp_path, old, new):
    return dr.report_package("mlx_vlm", new, old, root=tmp_path)


def _rows(r, path):
    (s,) = [s for s in r["units"] if s["path"] == path]
    return s, {row["unit"]: row["change"] for row in s["rows"]}


def test_each_copied_file_gets_its_change(tmp_path):
    old, new = _tree(tmp_path)
    r = _report(tmp_path, old, new)
    changes = {f["path"]: f["change"] for f in r["files"]}
    assert changes == {
        "LICENSE": "unchanged", "__init__.py": "unchanged",
        "same.py": "unchanged", "skip.py": "excluded",
        "gone.py": "removed upstream", "models/m.py": "both",
        "plain.py": "upstream only", "used.py": "upstream only",
        "version.py": "upstream only", "fresh.py": "new upstream file"}
    assert (r["copy"], r["new"]["version"], r["old"]["version"]) == \
        ("1.0", "1.1", "1.0")
    assert [s["path"] for s in r["units"]] == ["models/m.py", "used.py"]
    md = dr.markdown([r])
    assert "- `gone.py` (verbatim)" in md and "- `fresh.py`" in md


def test_the_old_release_splits_upstream_from_gmlx(tmp_path):
    old, new = _tree(tmp_path)
    r = _report(tmp_path, old, new)
    s, rows = _rows(r, "models/m.py")
    assert rows == {"up": "upstream", "both": "both", "fold": "gmlx",
                    "fresh": "upstream added", "gone": "upstream removed",
                    "ours": "gmlx added"}
    assert s["split"]
    assert _rows(r, "used.py")[1] == {"f": "upstream"}


def test_the_codemod_import_rewrite_is_not_a_gmlx_change(tmp_path):
    old, new = _tree(tmp_path, new_m=_OLD,
                     owned_m=_OLD.replace("from mlx_vlm.",
                                          "from gmlx._vendor.mlx_vlm."))
    s, rows = _rows(_report(tmp_path, None, new), "models/m.py")
    assert s["change"] == "gmlx only" and rows == {}


@pytest.mark.parametrize("old,new,unit", [
    ("        return 1\n", "        v = 1\n        return v\n", "A.p"),
    ("        self.v = v\n", "        self.v = int(v)\n", "A.p.setter"),
    ("        del self.v\n", "        self.v = None\n", "A.p.deleter"),
], ids=["getter", "setter", "deleter"])
def test_an_upstream_property_change_shows_as_its_own_row(tmp_path, old, new,
                                                          unit):
    pkg_old, pkg_new = _tree(tmp_path, new_m=_PROP.replace(old, new),
                             old_m=_PROP, owned_m=_PROP)
    s = _rows(_report(tmp_path, pkg_old, pkg_new), "models/m.py")[0]
    assert s["rows"] == [{"unit": unit, "change": "upstream"}]


def test_an_unchanged_upstream_needs_no_old_release(tmp_path):
    _, new = _tree(tmp_path, new_m=_OLD)
    r = _report(tmp_path, None, new)
    s, rows = _rows(r, "models/m.py")
    assert rows == {"fold": "gmlx", "both": "gmlx", "ours": "gmlx added"}
    assert s["split"] and s["change"] == "gmlx only" and r["old"] is None


def test_without_the_old_release_the_rows_are_unsplit(tmp_path):
    _, new = _tree(tmp_path)
    s, rows = _rows(_report(tmp_path, None, new), "models/m.py")
    assert not s["split"]
    assert set(rows.values()) <= {"changed", "added or removed"}


def test_an_old_file_that_is_not_the_recorded_one_is_listed(tmp_path):
    old, new = _tree(tmp_path)
    (old / "models" / "m.py").write_text(_OLD + "\n")
    r = _report(tmp_path, old, new)
    (entry,) = [f for f in r["files"] if f["path"] == "models/m.py"]
    assert entry["old"] == "unverifiable"
    s, rows = _rows(r, "models/m.py")
    assert not s["split"]
    assert set(rows.values()) <= {"changed", "added or removed"}
    md = dr.markdown([r])
    assert "Unverifiable" in md and "- `models/m.py`" in md


def test_the_default_source_is_the_stock_distribution_not_the_copy(
        monkeypatch):
    try:
        pkg = dr.package_dir("mlx_vlm")
    except dr.UsageError:
        pytest.skip("mlx-vlm is not installed")
    assert not pkg.resolve().is_relative_to(_ROOT), pkg
    assert (pkg / "version.py").is_file()

    def absent(name):
        raise dr.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dr.metadata, "distribution", absent)
    with pytest.raises(dr.UsageError, match="not installed: pass --new"):
        dr.package_dir("mlx_vlm")


@pytest.mark.parametrize("package", ["mlx_vlm", "mlx_lm"])
def test_the_installed_release_of_the_copy_shows_only_gmlx_changes(package):
    try:
        pkg = dr.package_dir(package)
    except dr.UsageError:
        pytest.skip(f"{package} is not installed")
    copy = dr._tool().read_manifest(_ROOT)["packages"][package]["version"]
    if dr.package_version(package, pkg) != copy:
        pytest.skip(f"the installed {package} is not the copied release")
    r = dr.report_package(package, pkg, None)
    changes = {f["change"] for f in r["files"]}
    assert changes <= {"gmlx only", "excluded", "unchanged"}, changes
    for s in r["units"]:
        assert s["split"] and isinstance(s["rows"], list), s["path"]
        kinds = {row["change"] for row in s["rows"]}
        assert kinds <= {"gmlx", "gmlx added", "gmlx removed"}, \
            (s["path"], kinds)


_IMPORTS_UP = _OWNED.replace("def up():\n    return 1\n\n", "") \
    .replace("def keep():\n    return 1\n\n", "") \
    .replace("import mlx.core as mx\n",
             "import mlx.core as mx\nfrom gmlx.owned import keep, up\n")


def test_a_unit_gmlx_took_over_still_shows_an_upstream_change(tmp_path):
    old, new = _tree(tmp_path, owned_m=_IMPORTS_UP)
    _write(tmp_path, {"gmlx/owned.py": "def keep():\n    return 1\n\n\n"
                                       "def up():\n    return 1\n"})
    s = _rows(_report(tmp_path, old, new), "models/m.py")[0]
    got = {row["unit"]: row for row in s["rows"]}
    assert got["up"] == {"unit": "up", "change": "upstream",
                         "owner": "gmlx/owned.py"}
    assert got["keep"] == {"unit": "keep", "change": "gmlx removed",
                           "owner": "gmlx/owned.py"}
    md = dr.markdown([_report(tmp_path, old, new)])
    assert "| `up` | upstream | `gmlx/owned.py` |" in md


_MIRROR = "def g():\n    return 1\n\n\ndef h():\n    return 1\n"


def _mirror(tmp_path, monkeypatch):
    old, new = _tree(tmp_path)
    _write(tmp_path, {"gmlx/mirror.py": _MIRROR})
    monkeypatch.setattr(dr, "EXTRACTS", (dr.Extract(
        "gmlx/mirror.py", "mlx_vlm", "used.py", renames=(("f", "g"),),
        mirror=True),))
    return old, new


def test_a_mirror_gets_rows_under_the_upstream_names(tmp_path, monkeypatch):
    _, new = _mirror(tmp_path, monkeypatch)
    r = _report(tmp_path, None, new)
    (s,) = [s for s in r["units"] if s["status"] == "mirror"]
    assert (s["path"], s["owned"], s["split"]) == \
        ("used.py", "gmlx/mirror.py", True)
    # The copy's used.py is the recorded release, so it splits the rows
    # with no old release; h is gmlx's own and not a row.
    assert s["rows"] == [{"unit": "f", "change": "upstream"}]


def test_the_listing_limit_never_hides_an_owned_or_mirrored_file(
        tmp_path, monkeypatch):
    old, new = _mirror(tmp_path, monkeypatch)
    monkeypatch.setattr(dr, "LISTED", 0)
    md = dr.markdown([_report(tmp_path, old, new)])
    assert "- `models/m.py` (edited)" in md
    assert "- `used.py` (codemod, mirror gmlx/mirror.py)" in md
    assert "- `plain.py`" not in md
    assert "- and 2 more, listed in the --json output" in md


def test_a_checkout_keeps_its_license_at_the_root(tmp_path):
    import shutil
    _, new = _tree(tmp_path)
    checkout = tmp_path / "checkout"
    shutil.copytree(new, checkout / "mlx_vlm")
    (checkout / ".git").mkdir()
    (checkout / "LICENSE").write_text(_LICENSE)
    r = _report(tmp_path, None, checkout / "mlx_vlm")
    (entry,) = [f for f in r["files"] if f["path"] == "LICENSE"]
    assert entry["change"] == "unchanged"


def _stock_lm():
    try:
        pkg = dr.package_dir("mlx_lm")
    except dr.UsageError:
        pytest.skip("mlx-lm is not installed")
    copy = dr._tool().read_manifest(_ROOT)["packages"]["mlx_lm"]["version"]
    if dr.package_version("mlx_lm", pkg) != copy:
        pytest.skip("the installed mlx-lm is not the copied release")
    return pkg


def test_upstream_changes_to_owned_attention_and_cache_units_show(tmp_path):
    """A release that changes scaled_dot_product_attention in base.py and
    KVCache.update_and_fetch in cache.py, both units gmlx owns and the
    copy imports: each is an upstream row naming its gmlx owner."""
    import shutil
    stock = _stock_lm()
    new = tmp_path / "mlx_lm"
    shutil.copytree(stock, new, ignore=shutil.ignore_patterns("__pycache__"))
    base = new / "models" / "base.py"
    text = base.read_text()
    base.write_text(text.replace("def scaled_dot_product_attention(",
                                 "def scaled_dot_product_attention(\n"
                                 "    *_unused,", 1))
    cache = new / "models" / "cache.py"
    text = cache.read_text()
    at = text.index("def update_and_fetch(", text.index("class KVCache("))
    cache.write_text(text[:at] + text[at:].replace(
        "):\n", "):\n        keys = keys + 0\n", 1))
    r = dr.report_package("mlx_lm", new, stock)

    def rows(path, status):
        return {(row["unit"], row["change"], row.get("owner"))
                for s in r["units"] if (s["path"], s["status"])
                == (path, status) for row in s["rows"]
                if row["change"] != "gmlx" and not row["change"].startswith(
                    "gmlx ")}

    assert rows("models/base.py", "edited") == {
        ("scaled_dot_product_attention", "upstream",
         "gmlx/cache/attention.py")}
    assert rows("models/cache.py", "edited") == {
        ("KVCache.update_and_fetch", "upstream", "gmlx/cache/kv_cache.py")}
    assert ("scaled_dot_product_attention", "both", None) in \
        rows("models/base.py", "extract")
    # gmlx edits both units in its copy too (the KV append routes in place).
    assert ("KVCache.update_and_fetch", "both", None) in \
        rows("models/cache.py", "extract")


def test_a_usage_error_returns_2(tmp_path, monkeypatch, capsys):
    _, new = _tree(tmp_path)
    assert dr.main(["--new", f"nope={new}"]) == 2
    assert dr.main(["--new", f"mlx_vlm={tmp_path}"]) == 2
    assert dr.main(["--new", f"mlx_vlm={new}", "--package", "nope"]) == 2

    def absent(name):
        raise dr.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dr.metadata, "distribution", absent)
    assert dr.main(["--package", "mlx_lm"]) == 2
    assert "not installed: pass --new mlx_lm=DIR" in capsys.readouterr().err
