"""The copied model files that are aliases of owned gmlx modules.

Each alias is imported in a child process in every order: the copied
package first, the copied file first, the owned package first, the owned
file first. In each order the copy path and the owned path are one module
object and the copied container holds the owned classes. Every name a
copied file imports from an aliased path is a name of the owned module,
and the upstream file of each alias that a reference test loads is the
file the manifest records.
"""

import ast
import hashlib
import json
import os
import subprocess
import sys

from pathlib import Path

import pytest

from pristine import ALIASED, upstream_file

_ROOT = Path(__file__).resolve().parents[2]
_VLM = "gmlx._vendor.mlx_vlm.models."
_LM = "gmlx._vendor.mlx_lm.models."
_DRAFT = "gmlx._vendor.mlx_vlm.speculative.drafters."

# copy path -> (owned path, names the copied package binds from the alias
# as (module that binds, name there, name in the owned module, or None for
# the owned module itself))
ALIASES = {
    _VLM + "qwen3_5.language": ("gmlx.models.qwen35.language", (
        (_VLM + "qwen3_5", "LanguageModel", "LanguageModel"),
        (_VLM + "qwen3_5.qwen3_5", "LanguageModel", "LanguageModel"),
        (_VLM + "qwen3_5_text.language", "Qwen3_5LanguageModel",
         "LanguageModel"),
        (_DRAFT + "qwen3_5_mtp.qwen3_5_mtp", "Qwen3_5DecoderLayer",
         "Qwen3_5DecoderLayer"))),
    _VLM + "qwen3_5_moe.language": ("gmlx.models.qwen35.moe_language", (
        (_VLM + "qwen3_5_moe", "LanguageModel", "LanguageModel"),
        (_VLM + "qwen3_5_moe.qwen3_5_moe", "LanguageModel", "LanguageModel"),
        (_DRAFT + "qwen3_5_mtp.qwen3_5_mtp", "Qwen3_5MoeDecoderLayer",
         "Qwen3_5MoeDecoderLayer"))),
    _VLM + "qwen3_5.gated_delta": ("gmlx.models.qwen35.gated_delta", (
        (_VLM + "qwen3_5.speculative_verifier", "gated_delta_update",
         "gated_delta_update"),)),
    _VLM + "gemma4.language": ("gmlx.models.gemma4.language", (
        (_VLM + "gemma4", "LanguageModel", "LanguageModel"),
        (_VLM + "gemma4.gemma4", "LanguageModel", "LanguageModel"),
        (_VLM + "gemma4.gemma4", "RMSNormNoScale", "RMSNormNoScale"),
        (_VLM + "diffusion_gemma.language", "RMSNormNoScale",
         "RMSNormNoScale"),
        (_DRAFT + "gemma4_assistant.gemma4_assistant", "DecoderLayer",
         "DecoderLayer"),
        (_DRAFT + "gemma4_dspark.gemma4_dspark", "Gemma4MLP", "MLP"))),
    _VLM + "gemma4_unified.gemma4_unified": ("gmlx.models.gemma4.unified", (
        (_VLM + "gemma4_unified", "LanguageModel", "LanguageModel"),
        (_VLM + "gemma4_unified", "Model", "Model"),
        (_VLM + "gemma4_unified", "VisionEmbedder", "VisionEmbedder"),
        (_VLM + "gemma4_unified", "VisionModel", "VisionEmbedder"))),
    _LM + "gemma4_text": ("gmlx.models.gemma4.text", (
        (_LM + "gemma4", "gemma4_text", None),)),
}

_CHILD = r"""
import importlib
import json
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
first, aliases = json.loads(sys.argv[1]), json.loads(sys.argv[2])
for name in first:
    importlib.import_module(name)
out = {}
for copy, (owned, bound) in aliases.items():
    c, o = importlib.import_module(copy), importlib.import_module(owned)
    parent, _, leaf = copy.rpartition(".")
    row = {"same": c is o and sys.modules[copy] is sys.modules[owned],
           "parent": getattr(sys.modules[parent], leaf) is o,
           "name": o.__name__ == owned,
           "bound": [f"{m}.{n}" for m, n, on in bound
                     if getattr(importlib.import_module(m), n)
                     is not (o if on is None else getattr(o, on))]}
    classes = [v for v in vars(o).values() if isinstance(v, type)
               and v.__module__ in (copy, owned)]
    row["copy_named"] = sorted(v.__name__ for v in classes
                               if v.__module__ != owned)
    out[copy] = row
out["stock"] = sorted(m for m in sys.modules
                      if m.partition(".")[0] in ("mlx_vlm", "mlx_lm"))
print(json.dumps(out))
"""


def _orders():
    for copy, (owned, _bound) in ALIASES.items():
        yield copy, [copy.rpartition(".")[0]]
        yield copy, [copy]
        yield copy, [owned.rpartition(".")[0]]
        yield copy, [owned]


@pytest.mark.parametrize(
    "copy,first", list(_orders()),
    ids=[f"{c.removeprefix(_VLM).removeprefix(_LM)}"
         f"<-{f[0].removeprefix('gmlx.')}" for c, f in _orders()])
def test_every_import_order_gives_one_module(copy, first):
    r = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(first),
                        json.dumps(ALIASES)],
                       capture_output=True, text=True, timeout=600,
                       env=dict(os.environ))
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out.pop("stock") == []
    assert set(out) == set(ALIASES)
    for path, row in out.items():
        assert row == {"same": True, "parent": True, "name": True,
                       "bound": [], "copy_named": []}, path


def _manifest() -> dict:
    return json.loads(
        (_ROOT / "gmlx/_vendor/manifest.json").read_text())["files"]


def _bound_names(path: Path) -> set[str]:
    """The names a module file binds at its top level."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).partition(".")[0]
                      for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def test_every_name_a_copied_file_imports_from_an_alias_is_owned():
    vendor = _ROOT / "gmlx" / "_vendor"
    owned = {copy: _bound_names(_ROOT / (path.replace(".", "/") + ".py"))
             for copy, (path, _bound) in ALIASES.items()}
    missing, seen = [], 0
    for file in sorted(vendor.rglob("*.py")):
        rel = file.relative_to(_ROOT).with_suffix("")
        package = list(rel.parts[:-1])
        for node in ast.walk(ast.parse(file.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ""
            if node.level:
                base = package[:len(package) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            for a in node.names:
                if target in owned:
                    seen += 1
                    if a.name not in owned[target]:
                        missing.append(f"{rel}: {target}.{a.name}")
    assert seen >= 10
    assert missing == []


@pytest.mark.parametrize("origin", ALIASED)
def test_the_upstream_reference_file_of_an_alias_is_the_shipped_file(origin):
    assert origin in ALIASES
    rel = origin.removeprefix("gmlx._vendor.").replace(".", "/") + ".py"
    entry = _manifest()[rel]
    assert entry["status"] == "alias"
    digest = hashlib.sha256(upstream_file(origin).read_bytes()).hexdigest()
    assert digest == entry["upstream"], rel


def test_the_table_covers_every_alias_of_a_model_file():
    root = _ROOT
    manifest = _manifest()
    aliased = {"gmlx._vendor." + name.removesuffix(".py").replace("/", ".")
               for name, e in manifest.items() if e["status"] == "alias"}
    aliased.discard("gmlx._vendor.mlx_vlm.models.cache")  # its own test
    assert aliased == set(ALIASES)
    for copy, (owned, _bound) in ALIASES.items():
        text = (root / (copy.replace(".", "/") + ".py")).read_text()
        assert f"import {owned} as _owned\n" in text, copy
        assert "sys.modules[__name__] = _owned\n" in text, copy
