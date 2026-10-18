"""The ComfyUI node pack in gmlx/comfyui_pack: its version is gmlx's, the
wheel ships every file of it, gmlx never imports it, gmlx's pytest never
collects its suites, and its example workflows are UI-format graphs that
carry the install command and match the API-format copies the headless
suite runs. CPU-only, reads files only."""
from __future__ import annotations

import ast
import fnmatch
import json
import runpy
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PACK = _ROOT / "gmlx" / "comfyui_pack"
_UI = _PACK / "example_workflows"
_API = _ROOT / "tests" / "comfyui_pack" / "workflows"


def _literals(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def _pyproject() -> dict:
    with open(_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def test_pack_version_is_gmlx_version():
    want = _pyproject()["project"]["version"]
    got = _literals(_PACK / "version.py")["__version__"]
    assert got == want, (f"set __version__ in gmlx/comfyui_pack/version.py to {want}, "
                         "gmlx's version")


def test_pack_versions_are_dotted_numbers():
    meta = _literals(_PACK / "version.py")
    for key in ("COMFYUI_FLOOR", "COMFYUI_TESTED"):
        parts = meta[key].split(".")
        assert len(parts) == 3 and all(p.isdigit() for p in parts), (key, meta[key])
    floor, tested = (tuple(int(p) for p in meta[k].split("."))
                     for k in ("COMFYUI_FLOOR", "COMFYUI_TESTED"))
    assert floor <= tested


def test_the_tested_comfyui_runs_every_family_the_pack_serves():
    """The pack serves FLUX.3 klein, whose model ComfyUI has from 0.41, and
    the docs' launch line shows the tested release."""
    meta = _literals(_PACK / "version.py")
    tested = tuple(int(p) for p in meta["COMFYUI_TESTED"].split("."))
    assert tested >= (0, 41, 1)
    page = (_ROOT / "docs" / "comfyui.md").read_text()
    assert f"(a ComfyUI folder, ComfyUI {meta['COMFYUI_TESTED']})" in page
    assert "FLUX.3 klein needs ComfyUI 0.41 or newer" in page


# The README title of each family the step contract serves.
TITLES = {"z_image": "Z-Image", "qwen_image21": "Qwen-Image 2.1",
          "krea2": "Krea 2", "flux2_klein": "FLUX.2 klein",
          "flux2_klein_9b": "FLUX.2 klein 9B",
          "flux2_klein_9b_kv": "FLUX.2 klein 9B KV",
          "flux3_klein": "FLUX.3 klein", "moss_image": "MOSS-Image"}


def _served_families() -> set:
    tree = ast.parse((_ROOT / "gmlx" / "serve" / "image_steps.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and \
                [getattr(t, "id", None) for t in node.targets] == ["COMFY"]:
            return set(ast.literal_eval(node.value))
    raise AssertionError("no COMFY table in gmlx/serve/image_steps.py")


def test_the_readme_lists_every_family_the_pack_serves_with_its_parity_rows():
    """Each family of the step contract is in the README's families list
    and has a row in the step table and the final image table."""
    assert set(TITLES) == _served_families()
    text = (_PACK / "README.md").read_text()
    families = text.split("Supported model families:", 1)[1].split("\n## ", 1)[0]
    parity = text.split("### Parity with the stock loaders", 1)[1]
    step, final = parity.split("The final image check", 1)
    for fam, title in TITLES.items():
        assert f"\n- {title}" in families, fam
        for name, table in (("step", step), ("final image", final)):
            assert any(ln.startswith(f"| {title}") for ln in table.splitlines()), \
                (fam, name)



def test_the_moss_image_step_rows_sit_under_headers_that_name_their_cells():
    """MOSS-Image reads its step distances from stock fp32 against the
    larger of the stock bf16 and in-process gmlx ones, so its rows sit in
    a table of their own whose headers name both."""
    parity = (_PACK / "README.md").read_text().split(
        "### Parity with the stock loaders", 1)[1]
    step = parity.split("The final image check", 1)[0]
    header = None
    rows = []
    for ln in step.splitlines():
        if ln.startswith("| Model |"):
            header = ln
        elif ln.startswith("| MOSS-Image |"):
            rows.append((header, ln))
    assert rows
    for header, row in rows:
        assert header is not None
        cols = [c.strip() for c in header.strip("|").split("|")]
        assert "gmlx in process, same step" in cols, header
        assert "stock bf16, same step" in cols, header
        assert len(row.strip("|").split("|")) == len(cols), row

def _pack_files() -> list[str]:
    out = []
    for p in sorted(_PACK.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or p.suffix == ".pyc" \
                or p.name == ".DS_Store":
            continue
        out.append(p.relative_to(_ROOT).as_posix())
    return out


def test_the_wheel_ships_every_pack_file():
    """A module ships because the pack is a package; any other file only
    through a package-data glob, which the CI wheel check then holds to
    the built wheel."""
    tool = _pyproject()["tool"]["setuptools"]
    assert any(fnmatch.fnmatch("gmlx.comfyui_pack", g)
               for g in tool["packages"]["find"]["include"])
    globs = [k.replace(".", "/") + "/" + g
             for k, gs in tool["package-data"].items() for g in gs]
    files = _pack_files()
    unshipped = [f for f in files if not f.endswith(".py")
                 and not any(fnmatch.fnmatch(f, g) for g in globs)]
    assert not unshipped, f"no package-data glob ships: {unshipped}"
    modules = [f for f in files if f.endswith(".py")]
    assert all(Path(_ROOT / f).parent == _PACK for f in modules), modules
    for want in ("gmlx/comfyui_pack/LICENSE", "gmlx/comfyui_pack/__init__.py",
                 "gmlx/comfyui_pack/version.py"):
        assert want in files
    assert len([f for f in files if f.startswith("gmlx/comfyui_pack/example_workflows/")]) == 11


def test_the_pack_keeps_its_apache_license():
    """The pack is Apache-2.0 inside a BUSL repository: its own LICENSE, the
    root LICENSE's exception for its folder, an SPDX line in each module and
    its LICENSE among the wheel's license files."""
    text = (_PACK / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    project = _pyproject()["project"]
    assert "Apache-2.0" in project["license"]
    assert "gmlx/comfyui_pack/LICENSE" in project["license-files"]
    root = " ".join((_ROOT / "LICENSE").read_text().split())
    assert ("Files in gmlx/comfyui_pack, the ComfyUI node pack, are licensed under the "
            "Apache License, Version 2.0 (see the LICENSE file in that folder), not under "
            "this License.") in root
    missing = [p.name for p in sorted(_PACK.rglob("*.py"))
               if p.read_text().splitlines()[0] != "# SPDX-License-Identifier: Apache-2.0"]
    assert not missing, f"no Apache-2.0 SPDX line first in: {missing}"


def test_gmlx_never_imports_the_pack():
    hits = []
    for py in sorted((_ROOT / "gmlx").rglob("*.py")):
        rel = py.relative_to(_ROOT)
        if "comfyui_pack" in rel.parts or "_vendor" in rel.parts:
            continue
        text = py.read_text()
        for node in ast.walk(ast.parse(text)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            if any("comfyui_pack" in n.split(".") for n in names):
                hits.append(f"{rel}:{node.lineno}")
        if '"gmlx.comfyui_pack' in text or "'gmlx.comfyui_pack" in text:
            hits.append(f"{rel}: names the module")
    assert not hits, hits


def test_gmlx_pytest_collects_no_pack_suite(monkeypatch):
    conftest = str(_ROOT / "tests" / "comfyui_pack" / "conftest.py")
    monkeypatch.delenv("COMFYUI_DIR", raising=False)
    assert runpy.run_path(conftest)["collect_ignore_glob"] == ["*"]
    monkeypatch.setenv("COMFYUI_DIR", "/x")
    assert runpy.run_path(conftest)["collect_ignore_glob"] == []


# Widget names in widgets_values order, for the node types the example
# workflows use. control_after_generate and upload are frontend widgets
# with no API input.
_WIDGETS = {
    "GmlxModelLoader": ["model_name", "server_url", "start_server"],
    "GmlxCLIPLoader": ["model_name", "server_url", "start_server"],
    "GmlxVAELoader": ["model_name", "server_url", "start_server"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "CLIPTextEncode": ["text"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name",
                 "scheduler", "denoise"],
    "VAEDecode": [],
    "SaveImage": ["filename_prefix"],
    "LoadImage": ["image", "upload"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "ConditioningZeroOut": [],
    "EmptyFlux2LatentImage": ["width", "height", "batch_size"],
    # A dynamic combo: the option, then the inputs of that option.
    "ReferenceMultiLatent": ["resolution", "resolution.megapixels"],
    "Flux2Scheduler": ["steps", "width", "height"],
    "BasicScheduler": ["scheduler", "steps", "denoise"],
    "RandomNoise": ["noise_seed", "control_after_generate"],
    "KSamplerSelect": ["sampler_name"],
    "CFGGuider": ["cfg"],
    "SamplerCustomAdvanced": [],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "VAEEncode": [],
    "ReferenceLatent": [],
    "GetImageSize": [],
    "FluxKVCache": [],
    "ModelSamplingSD3": ["shift"],
    "SkipLayerGuidanceDiTSimple": ["double_layers", "single_layers", "start_percent", "end_percent"],
}
_FRONTEND_ONLY = {"control_after_generate", "upload"}


def to_api(ui: dict) -> dict:
    """The API-format prompt of a UI-format graph, as ComfyUI's frontend
    queues it: Note nodes drop out, links become [node id, slot] pairs and
    widget values take their input names, but for a widget a link feeds."""
    links = {lk[0]: lk for lk in ui["links"]}
    out = {}
    for n in ui["nodes"]:
        if n["type"] == "Note":
            continue
        inputs = {}
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                lk = links[i["link"]]
                inputs[i["name"]] = [str(lk[1]), lk[2]]
        names = _WIDGETS[n["type"]]
        values = n.get("widgets_values", [])
        assert len(values) == len(names), (n["type"], values)
        for name, value in zip(names, values):
            if name not in _FRONTEND_ONLY and name not in inputs:
                inputs[name] = value
        out[str(n["id"])] = {"inputs": inputs, "class_type": n["type"],
                             "_meta": {"title": n.get("title", n["type"])}}
    return out


def _ui_files() -> list[Path]:
    return sorted(_UI.glob("*.json"))


@pytest.mark.parametrize("path", _ui_files(), ids=lambda p: p.stem)
def test_example_workflow_is_a_ui_graph_with_the_install_note(path):
    ui = json.loads(path.read_text())
    assert ui["version"] == 0.4
    nodes = {n["id"]: n for n in ui["nodes"]}
    assert len(nodes) == len(ui["nodes"])
    notes = [n for n in ui["nodes"] if n["type"] == "Note"]
    assert len(notes) == 1
    text = notes[0]["widgets_values"][0]
    assert "gmlx comfyui install" in text and "gmlx pull " in text
    assert text.isascii()
    assert ui["last_node_id"] == max(nodes)
    assert ui["last_link_id"] == len(ui["links"])
    for lid, src, sslot, dst, dslot, typ in ui["links"]:
        assert lid in nodes[src]["outputs"][sslot]["links"]
        inp = nodes[dst]["inputs"][dslot]
        assert inp["link"] == lid and inp["type"] == typ == nodes[src]["outputs"][sslot]["type"]
    for n in ui["nodes"]:
        assert n["type"] == "Note" or n["type"] in _WIDGETS, n["type"]
        for i in n.get("inputs", []):
            assert i["link"] is None or any(lk[0] == i["link"] for lk in ui["links"])
    to_api(ui)


@pytest.mark.parametrize("path", sorted(_API.glob("*.json")), ids=lambda p: p.stem)
def test_api_copy_is_its_example_workflow(path):
    ui = _UI / path.name
    assert ui.is_file(), f"{path.name} has no example workflow"
    assert json.loads(path.read_text()) == to_api(json.loads(ui.read_text()))


def test_every_example_workflow_names_a_known_model_id():
    from gmlx.image.catalog import RECIPES

    ids = set(RECIPES)
    for path in _ui_files():
        for n in json.loads(path.read_text())["nodes"]:
            if n["type"].startswith("Gmlx"):
                assert n["widgets_values"][0] in ids, (path.name, n["widgets_values"][0])


def test_ascii_only():
    bad = [f for f in _pack_files() if not (_ROOT / f).read_bytes().isascii()]
    assert not bad, bad
