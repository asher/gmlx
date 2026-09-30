"""gmlx declares what its copy of mlx-vlm and mlx-lm imports, and never the
stock packages. Each requirement of the copied releases, as the manifest
recorded them, has a decision: declared with at least the copy's floor,
brought by another declared requirement, moved to an extra, or dropped."""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]

# A requirement of the copied releases that gmlx does not declare in core:
# the requirement that brings it, the extra that holds it, or why it goes.
_NOT_CORE = {
    "miniaudio": "brought by mlx-audio",
    "tqdm": "brought by mlx-audio",
    "opencv-python": "extra video",
    "websockets": "dropped: only the stock /v1/realtime route used it",
}


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def _project() -> dict:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def _core() -> dict[str, Requirement]:
    return {_canonical(r.name): r
            for r in map(Requirement, _project()["dependencies"])}


def _copied_requirements() -> list[Requirement]:
    manifest = json.loads(
        (_ROOT / "gmlx" / "_vendor" / "manifest.json").read_text())
    out = []
    for package in manifest["packages"].values():
        for text in package["requires"]:
            r = Requirement(text)
            if r.marker is None or r.marker.evaluate({"extra": ""}):
                out.append(r)
    return out


def test_no_requirement_or_extra_names_a_stock_package():
    project = _project()
    names = [Requirement(r).name for r in project["dependencies"]]
    names += [Requirement(r).name
              for reqs in project["optional-dependencies"].values()
              for r in reqs]
    assert not {_canonical(n) for n in names} & {"mlx-vlm", "mlx-lm"}


def test_every_copied_requirement_has_a_decision():
    core, extras = _core(), _project()["optional-dependencies"]
    missing = []
    for req in _copied_requirements():
        name = _canonical(req.name)
        if name in _NOT_CORE:
            how = _NOT_CORE[name]
            if how.startswith("brought by "):
                assert how.removeprefix("brought by ") in core, how
            elif how.startswith("extra "):
                extra = extras[how.removeprefix("extra ")]
                assert name in {_canonical(Requirement(r).name)
                                for r in extra}, how
            assert name not in core, f"{name} is core and in _NOT_CORE"
            continue
        if name == "mlx":
            # mlx-kquant pins mlx; check_mlx_version holds the copy's floor.
            assert name in core
            continue
        if name not in core:
            missing.append(str(req))
            continue
        floors = [s.version for s in req.specifier if s.operator == ">="]
        ours = [s.version for s in core[name].specifier if s.operator == ">="]
        for floor in floors:
            assert ours and Version(max(ours, key=Version)) >= Version(floor), (
                f"gmlx declares {core[name]}, below the copy's {req}")
    assert not missing, missing


_VENDOR = _ROOT / "gmlx" / "_vendor"


def _module(path: Path) -> str:
    rel = path.relative_to(_ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _is_opencv(node) -> bool:
    """``cv2 = opencv(...)``: the copy's way to take cv2."""
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cv2"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "opencv")


def _imports_cv2(node) -> bool:
    return isinstance(node, ast.Import) and any(
        a.name.split(".")[0] == "cv2" for a in node.names) or (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] == "cv2")


def _outer_functions(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield None, node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name, sub


def _opencv_modules() -> dict[str, ast.Module]:
    """module -> tree for each copied module that names cv2."""
    out = {}
    for path in sorted(_VENDOR.rglob("*.py")):
        text = path.read_text()
        if "cv2" not in text:
            continue
        tree = ast.parse(text)
        if any(isinstance(n, ast.Name) and n.id == "cv2"
               for n in ast.walk(tree)):
            out[_module(path)] = tree
    return out


def _guarded(tree) -> set[int]:
    """ids of the cv2 imports inside a try that handles ImportError: a
    fallback runs without OpenCV there."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                h.type is not None and "ImportError" in ast.dump(h.type)
                for h in node.handlers):
            out |= {id(n) for stmt in node.body for n in ast.walk(stmt)
                    if _imports_cv2(n)}
    return out


def test_the_copy_takes_opencv_only_through_the_video_extra():
    """Every function of the copy that uses cv2 takes it from
    gmlx.commands.extras.opencv, whose error names the video extra, or
    imports it under a try that falls back without it. No module imports
    cv2 when it loads."""
    modules = _opencv_modules()
    assert "gmlx._vendor.mlx_vlm.models.multi_modality.vision" in modules
    bad = []
    for name, tree in modules.items():
        guarded = _guarded(tree)
        bad += [f"{name}:{n.lineno} imports cv2 at load"
                for n in tree.body if _imports_cv2(n)]
        for cls, fn in _outer_functions(tree):
            nodes = list(ast.walk(fn))
            names = [n for n in nodes if isinstance(n, ast.Name)
                     and n.id == "cv2"]
            stores = [n for n in names if isinstance(n.ctx, ast.Store)]
            if not names or len(stores) > sum(map(_is_opencv, nodes)):
                continue  # no cv2, or a local of that name
            bad += [f"{name}:{n.lineno} imports cv2 unguarded"
                    for n in nodes if _imports_cv2(n) and id(n) not in guarded]
            if not any(_is_opencv(n) for n in nodes) and not any(
                    id(n) in guarded for n in nodes):
                bad.append(f"{name}.{fn.name} uses cv2 it never takes")
    assert not bad, bad


def test_every_copy_module_that_uses_opencv_imports_without_it():
    names = sorted(_opencv_modules())
    code = ("import importlib, sys\n"
            "sys.modules['cv2'] = None\n"
            "bad = []\n"
            f"for name in {names!r}:\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except Exception as e:\n"
            "        bad.append(f'{name}: {type(e).__name__}: {e}')\n"
            "print('\\n'.join(bad))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", out.stdout


def _first_takes_opencv(fn) -> bool:
    body = fn.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return bool(body) and _is_opencv(body[0])


def _opencv_first_functions():
    for name, tree in _opencv_modules().items():
        for cls, fn in _outer_functions(tree):
            if _first_takes_opencv(fn):
                yield pytest.param(name, cls, fn.name,
                                   id=f"{name.rsplit('.', 2)[-2]}."
                                      f"{cls + '.' if cls else ''}{fn.name}")


@pytest.fixture
def no_opencv(monkeypatch):
    from gmlx.commands import extras

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr(extras, "install_route", lambda: extras.ROUTE_PIP)


@pytest.mark.parametrize("module,cls,func", list(_opencv_first_functions()))
def test_each_opencv_path_names_the_video_extra(no_opencv, module, cls, func):
    """With cv2 absent, a function that takes cv2 first raises the extras
    error before it reads any argument."""
    obj = importlib.import_module(module)
    fn = getattr(getattr(obj, cls), func) if cls else getattr(obj, func)
    params = [p for p in inspect.signature(fn).parameters.values()
              if p.default is p.empty
              and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    with pytest.raises(ImportError,
                       match=r" needs OpenCV: pip install 'gmlx\[video\]'$"):
        out = fn(*[None] * len(params))
        if inspect.isgenerator(out):
            next(out)


def test_the_named_opencv_paths_are_swept():
    """The four modules that imported cv2 at load, and DeepSeek-VL's image
    resize among them."""
    swept = {(p.values[0].rsplit(".", 1)[-1], p.values[2])
             for p in _opencv_first_functions()}
    assert {("vision", "resize_image"), ("annotators", "_resize_mask"),
            ("overlay", "render_mesh_overlay"), ("video", "draw_bbox"),
            ("video", "process_video")} <= swept


def test_load_video_names_the_video_extra(no_opencv):
    from gmlx._vendor.mlx_vlm.utils import load_video

    with pytest.raises(ImportError, match=r"gmlx\[video\]"):
        load_video("clip.mp4")


def test_the_opencv_hint_follows_the_install_route(monkeypatch):
    """A keg and a uv tool install get their own command, not pip."""
    from gmlx._vendor.mlx_vlm.utils import load_video
    from gmlx.commands import extras

    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr(extras, "install_route", lambda: extras.ROUTE_BREW)
    with pytest.raises(ImportError, match=r"^video input needs OpenCV: "
                       r"brew reinstall asher/gmlx/gmlx$"):
        load_video("clip.mp4")
