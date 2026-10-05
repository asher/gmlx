"""The copied model files that are aliases of owned gmlx modules.

Each alias is imported in a child process in every order: the copied
package first, the copied file first, the owned package first, the owned
file first. In each order the copy path and the owned path are one module
object and the copied container holds the owned classes.
"""

import json
import os
import subprocess
import sys

import pytest

_VLM = "gmlx._vendor.mlx_vlm.models."
_DRAFT = "gmlx._vendor.mlx_vlm.speculative.drafters."

# copy path -> (owned path, names the copied package binds from the alias
# as (module that binds, name there, name in the owned module))
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
                     is not getattr(o, on)]}
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
    ids=[f"{c.removeprefix(_VLM).removeprefix('gmlx._vendor.')}"
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


def test_the_table_covers_every_alias_of_a_model_file():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "gmlx/_vendor/manifest.json").read_text())["files"]
    aliased = {"gmlx._vendor." + name.removesuffix(".py").replace("/", ".")
               for name, e in manifest.items() if e["status"] == "alias"}
    aliased.discard("gmlx._vendor.mlx_vlm.models.cache")  # its own test
    assert aliased == set(ALIASES)
    for copy, (owned, _bound) in ALIASES.items():
        text = (root / (copy.replace(".", "/") + ".py")).read_text()
        assert f"import {owned} as _owned\n" in text, copy
        assert "sys.modules[__name__] = _owned\n" in text, copy
