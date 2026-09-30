"""gmlx's entries in the copy's registries: the tool-parser table, the
chat message formats of the owned model types, and the APC capabilities of
gmlx's cache classes. The copy reads each table at call time, so an entry
gmlx adds routes as the copy's own do."""
from __future__ import annotations

import types

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
