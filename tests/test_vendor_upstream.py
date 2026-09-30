# vendor_upstream: skip
"""scripts/vendor_upstream.py: the rewrite rules, the copy from a RECORD,
the manifest and its verify. Synthetic packages only. CPU."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "vendor_upstream.py"


def _load():
    spec = importlib.util.spec_from_file_location("_vendor_upstream", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


vu = _load()


def rw(text: str) -> str:
    return vu.rewrite_text(text)


# -- rules -------------------------------------------------------------------


def test_r1_rewrites_absolute_import_heads_and_leaves_relative_ones():
    src = ("from mlx_vlm.models.cache import KVCache\n"
           "import mlx_lm.models.base as base\n"
           "from mlx_lm import load\n"
           "from ..cache import ArraysCache\n"
           "from . import utils\n")
    assert rw(src) == (
        "from gmlx._vendor.mlx_vlm.models.cache import KVCache\n"
        "import gmlx._vendor.mlx_lm.models.base as base\n"
        "from gmlx._vendor.mlx_lm import load\n"
        "from ..cache import ArraysCache\n"
        "from . import utils\n")


def test_r1_keeps_formatting_of_multiline_and_mixed_imports():
    src = ("import os, mlx_vlm.utils as vu, sys\n"
           "from mlx_vlm.generate import (  # keep\n"
           "    stream_generate,\n"
           ")\n")
    assert rw(src) == (
        "import os, gmlx._vendor.mlx_vlm.utils as vu, sys\n"
        "from gmlx._vendor.mlx_vlm.generate import (  # keep\n"
        "    stream_generate,\n"
        ")\n")


def test_r2_rewrites_the_name_a_no_alias_import_binds():
    src = "import mlx_vlm.utils\n\nx = mlx_vlm.utils.load\n"
    assert rw(src) == ("import gmlx._vendor.mlx_vlm.utils\n\n"
                       "x = gmlx._vendor.mlx_vlm.utils.load\n")


def test_r2_leaves_a_name_no_import_binds_and_reports_it():
    src = "mlx_lm = object()\nprint(mlx_lm)\n"
    new, _, _, review = vu.analyze(src)
    assert new == src
    assert {r["kind"].split(" (")[0] for r in review} == {
        "name mlx_lm not bound by a no-alias import"}


def test_r3_rewrites_module_path_strings_and_logger_names():
    src = ('import importlib, logging, sys\n'
           'importlib.import_module("mlx_vlm.models.cache")\n'
           "sys.modules.get('mlx_lm.models.base')\n"
           'target = "mlx_vlm.apc:_resolve"\n'
           'prefix = "mlx_vlm.models."\n'
           'log = logging.getLogger("mlx_vlm")\n')
    assert rw(src) == (
        'import importlib, logging, sys\n'
        'importlib.import_module("gmlx._vendor.mlx_vlm.models.cache")\n'
        "sys.modules.get('gmlx._vendor.mlx_lm.models.base')\n"
        'target = "gmlx._vendor.mlx_vlm.apc:_resolve"\n'
        'prefix = "gmlx._vendor.mlx_vlm.models."\n'
        'log = logging.getLogger("gmlx._vendor.mlx_vlm")\n')


def test_r3_reports_a_rewritten_bare_root_for_review():
    new, counts, _, review = vu.analyze('ROOTS = ("mlx_vlm", "gmlx")\n')
    assert new == 'ROOTS = ("gmlx._vendor.mlx_vlm", "gmlx")\n'
    assert counts == {"R3 bare root string": 1}
    assert review[0]["kind"].startswith("bare root string")


def test_r4_rewrites_the_head_of_an_f_string_only():
    src = ('t = "x"\n'
           'a = f"mlx_lm.models.{t}"\n'
           'b = f"see mlx_vlm {t} docs"\n')
    assert rw(src) == ('t = "x"\n'
                       'a = f"gmlx._vendor.mlx_lm.models.{t}"\n'
                       'b = f"see mlx_vlm {t} docs"\n')


def test_r5_rewrites_a_child_process_script():
    src = ('CODE = r"""\n'
           'import sys\n'
           'from mlx_vlm.models import cache\n'
           'print(sys.modules["mlx_vlm.models.cache"] is cache)\n'
           '"""\n')
    assert rw(src) == (
        'CODE = r"""\n'
        'import sys\n'
        'from gmlx._vendor.mlx_vlm.models import cache\n'
        'print(sys.modules["gmlx._vendor.mlx_vlm.models.cache"] is cache)\n'
        '"""\n')


def test_comments_docstrings_messages_and_paths_are_left():
    src = ('"""Docs name mlx_vlm.utils."""\n'
           '# mlx_lm.models.base is upstream\n'
           'MSG = "install mlx_vlm for this"\n'
           'PATH = "mlx_lm/models/qwen2.py"\n'
           'URL = "https://github.com/Blaizzy/mlx-vlm"\n'
           'DIST = "mlx-vlm"\n')
    new, counts, left, _ = vu.analyze(src)
    assert new == src
    assert {x["kind"] for x in left} == {"docstring", "message or other",
                                          "path-like"}
    assert counts["left comment"] == 1


def test_module_name_comparisons_are_listed_for_review():
    src = ('def f(obj):\n'
           '    return obj.__module__.startswith("gmlx")\n')
    _, _, _, review = vu.analyze(src)
    assert [r["kind"] for r in review] == ["module-name comparison"]


def test_startswith_on_a_root_string_is_listed_for_review():
    src = ('def f(name):\n'
           '    a = name.startswith("mlx_vlm.")\n'
           '    b = name.startswith(("os.", "mlx_lm"))\n'
           '    return a or b or name.startswith("mlx_vlmx")\n')
    new, _, _, review = vu.analyze(src)
    assert [r["line"] for r in review
            if r["kind"] == "startswith test on a root string"] == [2, 3]
    assert new == ('def f(name):\n'
                   '    a = name.startswith("gmlx._vendor.mlx_vlm.")\n'
                   '    b = name.startswith(("os.", "gmlx._vendor.mlx_lm"))\n'
                   '    return a or b or name.startswith("mlx_vlmx")\n')


def test_a_stock_name_line_keeps_its_names():
    src = ('import mlx_vlm.utils  # gmlx: stock-name\n'
           'OLD = "mlx_vlm.models."  # gmlx: stock-name\n'
           'NEW = "mlx_vlm.models."\n'
           'u = mlx_vlm.utils  # gmlx: stock-name\n'
           'v = mlx_vlm.utils\n')
    new, counts, _, _ = vu.analyze(src)
    assert new == ('import mlx_vlm.utils  # gmlx: stock-name\n'
                   'OLD = "mlx_vlm.models."  # gmlx: stock-name\n'
                   'NEW = "gmlx._vendor.mlx_vlm.models."\n'
                   'u = mlx_vlm.utils  # gmlx: stock-name\n'
                   'v = mlx_vlm.utils\n')
    assert counts["R3 module-path string"] == 1
    assert "R1 import mlx_vlm" not in counts


def test_bytes_and_prefixed_names_are_not_rewritten():
    src = ('a = b"mlx_vlm.utils"\n'
           'b = "gmlx._vendor.mlx_vlm.utils"\n'
           'c = "my_mlx_vlm.utils"\n')
    assert rw(src) == src


def test_skip_pragma_leaves_the_file():
    src = '# vendor_upstream: skip\nimport mlx_vlm\n'
    assert rw(src) == src


def test_rewrite_is_idempotent_and_reversible():
    src = ('import importlib\n'
           'import mlx_vlm.utils\n'
           'from mlx_lm.models import cache as c\n'
           'm = importlib.import_module("mlx_lm.models.llama")\n'
           't = "x"\n'
           'n = f"mlx_vlm.models.{t}"\n'
           'u = mlx_vlm.utils\n'
           '# mlx_vlm comment\n')
    once = rw(src)
    assert once != src
    assert rw(once) == once
    assert vu.reverse_text(once) == src


# -- copy, manifest, verify --------------------------------------------------


def _record_hash(data: bytes) -> str:
    d = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + d.decode()


_PKGS = {
    "mlx_vlm": ("0.7.3", "version.py", {
        "__init__.py": "from .version import __version__\n",
        "version.py": '__version__ = "0.7.3"\n',
        "py.typed": "",
        "models/__init__.py": "",
        "models/cache.py": (
            "from mlx_lm.models.cache import KVCache as _LM\n\n\n"
            "class KVCache(_LM):\n    pass\n"),
        "models/base.py": ('"""Built on mlx_lm.models.base."""\n'
                           "from .cache import KVCache\n"),
        "utils.py": (
            "import importlib\n\n\n"
            "def get(t):\n"
            '    return importlib.import_module(f"mlx_vlm.models.{t}")\n'),
    }),
    "mlx_lm": ("0.31.3", "_version.py", {
        "__init__.py": "from ._version import __version__\n",
        "_version.py": '__version__ = "0.31.3"\n',
        "models/__init__.py": "",
        "models/cache.py": "class KVCache:\n    pass\n",
    }),
}


def _site_packages(tmp: Path, tamper: str | None = None,
                   files: dict[str, str] | None = None) -> Path:
    """Both distributions, each with a RECORD. ``files`` replaces a file's
    text before the RECORD is written; ``tamper`` replaces it after."""
    sp = tmp / "site-packages"
    for pkg, (version, _, pkg_files) in _PKGS.items():
        dist = sp / f"{pkg}-{version}.dist-info"
        (dist / "licenses").mkdir(parents=True)
        rows = []
        for rel, text in pkg_files.items():
            text = (files or {}).get(f"{pkg}/{rel}", text)
            p = sp / pkg / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            rows.append(f"{pkg}/{rel},{_record_hash(text.encode())},"
                        f"{len(text)}")
        pyc = sp / pkg / "__pycache__" / "__init__.cpython-312.pyc"
        pyc.parent.mkdir()
        pyc.write_bytes(b"\0")
        rows.append(f"{pkg}/__pycache__/__init__.cpython-312.pyc,,")
        lic = f"MIT License for {pkg}\n".encode()
        (dist / "licenses" / "LICENSE").write_bytes(lic)
        rows.append(f"{dist.name}/licenses/LICENSE,{_record_hash(lic)},"
                    f"{len(lic)}")
        (dist / "METADATA").write_text(
            f"Name: {pkg}\nVersion: {version}\nRequires-Dist: mlx>=0.32.2\n")
        rows.append(f"{dist.name}/RECORD,,")
        (dist / "RECORD").write_text("\n".join(rows) + "\n")
    if tamper:
        (sp / tamper).write_text("tampered = True\n")
    return sp


def _wheels(sp: Path, out: Path) -> Path:
    out.mkdir()
    for pkg, (version, _, _) in _PKGS.items():
        with zipfile.ZipFile(out / f"{pkg}-{version}-py3-none-any.whl",
                             "w") as z:
            for p in sorted(sp.rglob("*")):
                rel = p.relative_to(sp).as_posix()
                if p.is_file() and (rel.startswith(f"{pkg}/")
                                    or rel.startswith(f"{pkg}-{version}.")):
                    z.write(p, rel)
    return out


def _repo(tmp: Path) -> Path:
    root = tmp / "repo"
    (root / "gmlx" / "_vendor").mkdir(parents=True)
    return root


def _run(*argv) -> int:
    return vu.main([str(a) for a in argv])


@pytest.mark.parametrize("source", ["site-packages", "wheels"])
def test_copy_takes_every_record_file_and_writes_the_manifest(tmp_path,
                                                              source):
    sp = _site_packages(tmp_path)
    src = sp if source == "site-packages" else _wheels(sp, tmp_path / "w")
    root = _repo(tmp_path)
    assert _run("--root", root, "copy", src) == 0
    vendor = root / "gmlx" / "_vendor"
    for pkg, (version, _, files) in _PKGS.items():
        for rel, text in files.items():
            assert (vendor / pkg / rel).read_text() == text
        assert not (vendor / pkg / "__pycache__").exists()
        assert (vendor / pkg / "LICENSE").read_text().startswith("MIT")
    m = json.loads((vendor / "manifest.json").read_text())
    assert m["prefix"] == "gmlx._vendor."
    assert m["packages"]["mlx_vlm"]["version"] == "0.7.3"
    assert m["packages"]["mlx_lm"]["requires"] == ["mlx>=0.32.2"]
    assert m["files"]["mlx_vlm/utils.py"]["status"] == "verbatim"
    assert m["files"]["mlx_lm/LICENSE"]["status"] == "license"
    assert len(m["files"]) == sum(len(f) for _, _, f in _PKGS.values()) + 2


@pytest.mark.parametrize("source", ["site-packages", "wheels"])
def test_copy_refuses_a_file_that_does_not_match_its_record(tmp_path, source):
    sp = _site_packages(tmp_path, tamper="mlx_vlm/utils.py")
    src = sp if source == "site-packages" else _wheels(sp, tmp_path / "w")
    with pytest.raises(SystemExit, match="utils.py does not match RECORD"):
        _run("--root", _repo(tmp_path), "copy", src)


def test_a_refused_copy_leaves_the_tree_as_it_was(tmp_path):
    """mlx_vlm copies first and is sound; mlx_lm then fails its RECORD."""
    root = _repo(tmp_path)
    sp = _site_packages(tmp_path, tamper="mlx_lm/models/cache.py")
    with pytest.raises(SystemExit, match="does not match RECORD"):
        _run("--root", root, "copy", sp)
    assert list((root / "gmlx/_vendor").iterdir()) == []


def test_copy_refuses_a_license_that_does_not_match_its_record(tmp_path):
    sp = _site_packages(
        tmp_path, tamper="mlx_lm-0.31.3.dist-info/licenses/LICENSE")
    with pytest.raises(SystemExit, match="LICENSE does not match RECORD"):
        _run("--root", _repo(tmp_path), "copy", sp)


def test_copy_refuses_upstream_text_that_holds_the_prefix(tmp_path):
    """The reverse could not tell such text from the rewrite's."""
    sp = _site_packages(tmp_path, files={
        "mlx_vlm/models/base.py": '"""See gmlx._vendor.mlx_vlm."""\n'})
    with pytest.raises(SystemExit, match="already contains"):
        _run("--root", _repo(tmp_path), "copy", sp)


def _copied_and_rewritten(tmp_path) -> Path:
    root = _repo(tmp_path)
    assert _run("--root", root, "copy", _site_packages(tmp_path)) == 0
    assert _run("--root", root, "rewrite", "gmlx/_vendor") == 0
    assert _run("--root", root, "manifest") == 0
    return root


def test_after_the_rewrite_the_manifest_verifies(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    m = json.loads((root / "gmlx/_vendor/manifest.json").read_text())
    status = {k: v["status"] for k, v in m["files"].items()}
    assert status["mlx_vlm/models/cache.py"] == "codemod"
    assert status["mlx_vlm/utils.py"] == "codemod"
    assert status["mlx_vlm/models/base.py"] == "verbatim"
    assert "sha256" in m["files"]["mlx_vlm/utils.py"]
    assert "sha256" not in m["files"]["mlx_vlm/models/base.py"]
    assert vu.verify(root) == []


def test_verify_before_the_rewrite_names_each_unrewritten_file(tmp_path):
    root = _repo(tmp_path)
    _run("--root", root, "copy", _site_packages(tmp_path))
    assert sorted(vu.verify(root)) == [
        "mlx_vlm/models/cache.py: a module reference is not rewritten",
        "mlx_vlm/utils.py: a module reference is not rewritten"]


def test_verify_names_an_edit_the_manifest_does_not_know(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    f = root / "gmlx/_vendor/mlx_lm/models/cache.py"
    f.write_text(f.read_text() + "X = 1\n")
    assert vu.verify(root) == [
        "mlx_lm/models/cache.py: is edited, manifest says verbatim"]


def test_an_edit_that_only_adds_the_prefix_is_an_edit(tmp_path):
    """It reverses to upstream's bytes, but the rewrite never produces it."""
    root = _copied_and_rewritten(tmp_path)
    f = root / "gmlx/_vendor/mlx_vlm/models/base.py"
    f.write_text(f.read_text().replace("on mlx_lm", "on gmlx._vendor.mlx_lm"))
    assert vu.verify(root) == [
        "mlx_vlm/models/base.py: is edited, manifest says verbatim"]
    _run("--root", root, "manifest")
    assert vu.verify(root) == [
        "mlx_vlm/models/base.py: edited without a '# gmlx:' comment"]


def test_an_edit_needs_the_marker(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    f = root / "gmlx/_vendor/mlx_lm/models/cache.py"
    f.write_text(f.read_text() + "X = 1\n")
    _run("--root", root, "manifest")
    assert vu.verify(root) == [
        "mlx_lm/models/cache.py: edited without a '# gmlx:' comment"]
    f.write_text(f.read_text() + "# gmlx: X for the test.\n")
    _run("--root", root, "manifest")
    assert vu.verify(root) == []


def test_verify_takes_a_stock_name_only_on_a_marked_line(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    f = root / "gmlx/_vendor/mlx_lm/models/cache.py"
    base = f.read_text()
    problem = ["mlx_lm/models/cache.py: a module reference is not rewritten"]
    for tail, want in (
            ('# gmlx: the stock prefix.\nP = "mlx_lm."\n', problem),
            ('P = "mlx_lm."  # gmlx: stock-name\n', []),
            ('# gmlx: stock-name\nP = "mlx_lm."\n', problem),
            ('P = "mlx_lm."  # gmlx: stock-name\nQ = "mlx_lm.x"\n', problem)):
        f.write_text(base + tail)
        _run("--root", root, "manifest")
        assert vu.verify(root) == want, tail


def test_verify_names_an_edit_made_after_the_manifest(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    f = root / "gmlx/_vendor/mlx_lm/models/cache.py"
    f.write_text(f.read_text() + "# gmlx: X for the test.\nX = 1\n")
    _run("--root", root, "manifest")
    f.write_text(f.read_text() + "Y = 2\n")
    assert vu.verify(root) == [
        "mlx_lm/models/cache.py: sha256 differs from the manifest"]


def test_a_license_must_stay_the_one_upstream_shipped(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    lic = root / "gmlx/_vendor/mlx_lm/LICENSE"
    lic.write_text(lic.read_text() + "Amended.\n")
    problem = ["mlx_lm/LICENSE: differs from the LICENSE upstream shipped"]
    assert vu.verify(root) == problem
    assert _run("--root", root, "manifest") == 0
    assert vu.verify(root) == problem
    path = root / "gmlx/_vendor/manifest.json"
    m = json.loads(path.read_text())
    m["files"]["mlx_lm/LICENSE"]["sha256"] = hashlib.sha256(
        lic.read_bytes()).hexdigest()
    path.write_text(json.dumps(m))
    assert vu.verify(root) == problem


def test_verify_checks_the_manifest_prefix(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    path = root / "gmlx/_vendor/manifest.json"
    m = json.loads(path.read_text())
    m["prefix"] = "gmlx.vendor."
    path.write_text(json.dumps(m))
    assert vu.verify(root) == [
        "manifest prefix 'gmlx.vendor.' is not 'gmlx._vendor.'"]


def test_alias_and_excluded_files(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    vendor = root / "gmlx/_vendor"
    (vendor / "mlx_vlm/models/base.py").write_text(
        "# gmlx: the owned module replaces this file.\n"
        "import sys\n\nimport gmlx.owned as _owned\n\n"
        "sys.modules[__name__] = _owned\n")
    (vendor / "mlx_lm/models/cache.py").unlink()
    assert sorted(vu.verify(root)) == [
        "mlx_lm/models/cache.py: is excluded, manifest says verbatim",
        "mlx_vlm/models/base.py: is alias, manifest says verbatim"]
    _run("--root", root, "manifest")
    assert vu.verify(root) == []
    m = json.loads((vendor / "manifest.json").read_text())
    assert m["files"]["mlx_lm/models/cache.py"] == {
        "status": "excluded",
        "upstream": hashlib.sha256(
            _PKGS["mlx_lm"][2]["models/cache.py"].encode()).hexdigest()}


def test_verify_names_a_file_upstream_did_not_ship(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    (root / "gmlx/_vendor/mlx_vlm/extra.py").write_text("X = 1\n")
    assert vu.verify(root) == [
        "mlx_vlm/extra.py: under the copy but not in the manifest"]
    assert _run("--root", root, "manifest") == 1


def test_a_one_package_copy_keeps_the_other_package_verified(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    vendor = root / "gmlx/_vendor"
    before = json.loads((vendor / "manifest.json").read_text())
    for p in sorted((vendor / "mlx_lm").rglob("*"), reverse=True):
        p.rmdir() if p.is_dir() else p.unlink()
    (vendor / "mlx_lm").rmdir()
    assert _run("--root", root, "copy", tmp_path / "site-packages",
                "--packages", "mlx_lm") == 0
    assert _run("--root", root, "manifest") == 0
    after = json.loads((vendor / "manifest.json").read_text())
    assert after == before
    (vendor / "mlx_vlm/models/base.py").write_text("X = 1\n")
    assert vu.verify(root) == [
        "mlx_vlm/models/base.py: is edited, manifest says verbatim"]


def test_verify_names_a_package_the_manifest_does_not_list(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    vendor = root / "gmlx/_vendor"
    m = json.loads((vendor / "manifest.json").read_text())
    del m["packages"]["mlx_vlm"]
    m["files"] = {k: v for k, v in m["files"].items()
                  if not k.startswith("mlx_vlm/")}
    (vendor / "manifest.json").write_text(json.dumps(m))
    (vendor / "misaki").mkdir()
    (vendor / "misaki" / "__init__.py").write_text("")
    assert vu.verify(root) == [
        "mlx_vlm: a package under the copy that the manifest does not list"]


def test_verify_checks_the_version_files(tmp_path):
    root = _copied_and_rewritten(tmp_path)
    (root / "gmlx/_vendor/mlx_lm/_version.py").write_text(
        '__version__ = "0.31.4"\n# gmlx: bumped by hand.\n')
    _run("--root", root, "manifest")
    assert vu.verify(root) == [
        "mlx_lm: version file says 0.31.4, manifest says 0.31.3"]


def test_check_mode_fails_until_the_tree_is_rewritten(tmp_path, capsys):
    root = _repo(tmp_path)
    f = root / "tests" / "test_x.py"
    f.parent.mkdir()
    f.write_text("from mlx_vlm.utils import load\n")
    (root / "tests" / "test_skip.py").write_text(
        "# vendor_upstream: skip\nimport mlx_lm\n")
    assert _run("--root", root, "rewrite", "--check") == 1
    assert "would rewrite: tests/test_x.py" in capsys.readouterr().out
    assert f.read_text() == "from mlx_vlm.utils import load\n"
    assert _run("--root", root, "rewrite", "tests") == 0
    assert _run("--root", root, "rewrite", "--check") == 0
    assert (root / "tests" / "test_skip.py").read_text().endswith(
        "import mlx_lm\n")
