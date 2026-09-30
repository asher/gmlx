"""gmlx's entries in the copy's registries: the tool-parser table and when
gmlx's parsers join it, the chat message formats of the owned model types,
the owned model modules the copy's class lookup imports, and the APC
capabilities of gmlx's cache classes. The copy reads each table at call
time, so an entry gmlx adds routes as the copy's own do."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import gmlx._vendor.mlx_vlm.tools as tools
import gmlx._vendor.mlx_vlm.tools.registry as reg
import gmlx.upstream.tool_parsers as tp

_STOCK_PRIORITY = 100


def _fresh_tables(monkeypatch):
    monkeypatch.setattr(reg, "SPECS", reg.SPECS)
    monkeypatch.setattr(reg, "_BY_NAME", dict(reg._BY_NAME))
    monkeypatch.setattr(reg, "GMLX_PARSERS", dict(reg.GMLX_PARSERS))
    monkeypatch.setattr(tools, "SPECS", tools.SPECS)
    monkeypatch.setattr(tp, "_REGISTERED", set(tp._REGISTERED))


def test_the_copy_holds_the_stock_parser_specs_below_gmlx_ones():
    ours = tp.registered_names()
    stock = [s for s in reg.SPECS if s.name not in ours]
    assert stock[-1].name == "json_tools" and stock[-1].priority == 0
    assert all(s.priority == _STOCK_PRIORITY for s in stock[:-1])
    assert {s.name for s in stock} <= set(reg._BY_NAME)
    assert tp._GMLX_PRIORITY > _STOCK_PRIORITY
    assert all(reg._BY_NAME[n].priority == tp._GMLX_PRIORITY for n in ours)


def test_a_registered_parser_loads_and_takes_its_markers(monkeypatch):
    _fresh_tables(monkeypatch)
    probe = types.ModuleType("_gmlx_parser_probe")
    tp.register_tool_parser("_gmlx_probe", probe, (("<probe>",),))
    assert reg.SPECS[0].name == "_gmlx_probe"
    assert tools.SPECS is reg.SPECS
    assert reg.load_tool_module("_gmlx_probe") is probe
    assert reg._infer_tool_parser("x <probe> y") == "_gmlx_probe"
    assert reg._infer_tool_parser(None, override="_gmlx_probe") == "_gmlx_probe"
    assert tp.registered_names() >= {"_gmlx_probe"}


def test_a_stock_name_registered_keeps_its_markers_and_loads_gmlx(
        monkeypatch):
    _fresh_tables(monkeypatch)
    probe = types.ModuleType("_gmlx_parser_probe")
    before = reg.SPECS
    tp.register_tool_parser("glm47", probe, (("<other>",),))
    assert reg.SPECS == before
    assert reg.load_tool_module("glm47") is probe
    assert reg._infer_tool_parser("<arg_key>") == "glm47"
    assert "glm47" not in tp.registered_names()


def test_every_owned_vlm_type_has_a_message_format():
    from gmlx._vendor.mlx_vlm.prompt_utils import MODEL_CONFIG
    from gmlx.models.owned import OWNED_VLM_MODELS

    assert not [t for t in OWNED_VLM_MODELS if t not in MODEL_CONFIG]


def test_the_gmlx_caches_register_their_apc_capability():
    import gmlx._vendor.mlx_vlm.apc_adapters as adapters
    from gmlx.cache.apc_pooling import install_pooling_apc_support
    from gmlx.cache.kvarn_apc import install_kvarn_apc
    from gmlx.cache.kvarn_cache import BatchKVarNKVCache, KVarNKVCache
    from gmlx.models.deepseek_v4.cache import (
        DS4BatchPoolingCache,
        DS4PoolingCache,
    )

    install_pooling_apc_support()
    install_kvarn_apc()
    for cls in (DS4PoolingCache, DS4BatchPoolingCache, KVarNKVCache,
                BatchKVarNKVCache):
        assert adapters._CAPABILITY.get(cls) is \
            adapters.Capability.CHECKPOINT, cls.__name__


def test_an_import_error_inside_an_owned_model_module_is_chained(
        tmp_path, monkeypatch):
    from gmlx._vendor.mlx_lm.utils import _get_classes
    from gmlx.models import owned

    (tmp_path / "_gmlx_owned_probe.py").write_text(
        "import _gmlx_absent_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(owned.OWNED_LM_MODELS, "probe_type",
                        "_gmlx_owned_probe")
    with pytest.raises(ValueError, match="probe_type not supported") as e:
        _get_classes({"model_type": "probe_type"})
    cause = e.value.__cause__
    assert isinstance(cause, ModuleNotFoundError)
    assert cause.name == "_gmlx_absent_dependency"


_ATEM_CHILD = """
import gmlx.commands.cli, gmlx.load.loader, gmlx.serve.patches
import gmlx._vendor.mlx_vlm.tools.registry as reg

template = '<atem:function_calls>\\n<atem:invoke name="f">'
first = (reg._infer_tool_parser(template), sorted(reg.GMLX_PARSERS),
         reg.load_tool_module("atem").__name__)
import gmlx.models.muse_glimmer.tools as mg

mg.ensure_registered()
print(repr((first, reg._infer_tool_parser(template))))
"""


def test_atem_resolves_to_the_copy_parser_until_a_muse_model_loads():
    """No gmlx import registers the Muse Glimmer parser: an ATEM template
    takes the copy's ``atem`` parser until a Muse Glimmer model loads."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), KQUANT_FORCE_CPU="1")
    r = subprocess.run([sys.executable, "-c", _ATEM_CHILD],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    first, after = ast.literal_eval(r.stdout.strip().splitlines()[-1])
    assert first == ("atem", [],
                     "gmlx._vendor.mlx_vlm.tools.parsers.atem"), first
    assert after == "muse_glimmer"
