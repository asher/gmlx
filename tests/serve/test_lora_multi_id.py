#!/usr/bin/env python3
"""P4 serve binding: ids on one GGUF that differ only in adapter share one
resident entry (the adapter union is the signature's adapter component and
the row channel's slot list), each request selects its slot, adapted rows
salt their APC hash, and a second adapter installs on an already adapted
model as a further slot on every target. CPU-only."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from gmlx import config as cfgmod, lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


def _rm(mid, adapter_path, **kw):
    common = dict(path="/p/base.gguf", load={}, cache={}, system=None,
                  speculative=False, mmproj=None, draft_gguf=None, pin=False,
                  ttl_s=None, sampling={})
    common.update(kw)
    return cfgmod.ResolvedModel(id=mid, adapter=adapter_path, **common)


# signature: the adapter union is the adapter component

def test_union_signature_shares_entry_and_forks_on_change():
    a = _rm("a", "/abs/A.gguf")
    b = _rm("b", "/abs/B.gguf")
    bare = _rm("d", None)
    assert a.load_signature() != b.load_signature()          # unfilled: own adapter
    assert a.base_signature() == b.base_signature() == bare.base_signature()
    for rm in (a, b, bare):
        rm.adapters = ("/abs/A.gguf", "/abs/B.gguf")
    assert a.load_signature() == b.load_signature() == bare.load_signature()
    c = _rm("c", "/abs/C.gguf")
    c.adapters = ("/abs/A.gguf", "/abs/B.gguf", "/abs/C.gguf")
    assert c.load_signature() != a.load_signature()          # new union: new entry
    other = _rm("e", "/abs/A.gguf", chat_template="x")
    assert other.base_signature() != a.base_signature()       # different load: no share


def test_registry_fills_union_over_siblings(tmp_path):
    pytest.importorskip("mlx_vlm")
    import gmlx.serve.bridge_vlm as serving
    from gmlx.config import build_config
    base = tmp_path / "base.gguf"
    base.write_bytes(b"GGUF")
    (tmp_path / "A.gguf").write_bytes(b"GGUF")
    (tmp_path / "B.gguf").write_bytes(b"GGUF")
    doc = {
        "server": {"model_dirs": [str(tmp_path)]},
        "profiles": {"kv8": {"load": {"kv_bits": 8}}},
        "models": {
            "base": {"path": str(base)},
            "a": {"path": str(base), "adapter": str(tmp_path / "A.gguf")},
            "b": {"path": str(base), "adapter": str(tmp_path / "B.gguf")},
            "a-kv8": {"path": str(base), "adapter": str(tmp_path / "A.gguf"),
                      "profile": "kv8"},
        },
    }
    serving.clear_resolved_models()
    try:
        serving.register_resolved_models(build_config(doc))
        rms = serving._RESOLVED_MODELS
        union = (str(tmp_path / "A.gguf"), str(tmp_path / "B.gguf"))
        assert rms["base"].adapters == union
        assert rms["a"].adapters == union and rms["b"].adapters == union
        assert rms["a-kv8"].adapters == (str(tmp_path / "A.gguf"),)   # other load
        sig = {rms[k].load_signature() for k in ("base", "a", "b")}
        assert len(sig) == 1                                  # one resident entry
        assert rms["a-kv8"].load_signature() not in sig
        # request-time resolution fills the same union
        _path, rm = serving.resolve_request_model("b")
        assert rm.adapters == union
        assert lora_rows.request_scales(rm) == (0.0, 1.0)
        _path, rm = serving.resolve_request_model("base")
        assert lora_rows.request_scales(rm) == (0.0, 0.0)
    finally:
        serving.clear_resolved_models()


# row channel: slots grow, tuples index the union, adapted rows salt APC

def test_request_scales_index_the_entry_union():
    lora_rows.ensure_rows(2)
    spec = SimpleNamespace(adapter="/B", adapters=("/A", "/B"))
    assert lora_rows.request_scales(spec) == (0.0, 1.0)
    spec = SimpleNamespace(adapter="/A", adapters=("/A", "/B"))
    assert lora_rows.request_scales(spec) == (1.0, 0.0)
    assert lora_rows.request_scales(SimpleNamespace(adapter=None, adapters=("/A",))) \
        == (0.0, 0.0)
    with pytest.raises(lora_rows.LoraRowsError):
        lora_rows.request_scales(SimpleNamespace(adapter="/C", adapters=("/A", "/B")))


def test_ensure_rows_grows_without_dropping_registrations():
    lora_rows.ensure_rows(1)
    lora_rows.register_uid(7, (1.0,))
    lora_rows.ensure_rows(3)
    assert lora_rows.mode() == "rows" and lora_rows.n_slots() == 3
    assert lora_rows.scales_for(7) == (1.0, 0.0, 0.0)
    lora_rows.register_uid(8, (0.0, 1.0))                  # short tuple pads
    assert lora_rows.scales_for(8) == (0.0, 1.0, 0.0)
    lora_rows.ensure_rows(2)                               # never shrinks
    assert lora_rows.n_slots() == 3


def test_lora_salt_is_stable_and_distinct():
    assert lora_rows.lora_salt((1.0, 0.0)) == lora_rows.lora_salt([1, 0])
    assert lora_rows.lora_salt((1.0, 0.0)) != lora_rows.lora_salt((0.0, 1.0))


def test_insert_salts_apc_hash_for_adapted_rows_only():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.generate import ar as _ar
    from mlx_vlm.server.generation import ResponseGenerator as RG
    lora_rows.install_row_channel()
    lora_rows.ensure_rows(2)

    def fake_bg():
        return SimpleNamespace(max_tokens=8, logits_processors=None,
                               _unprocessed_sequences=[], uid_count=100,
                               apc_manager=object(),
                               _apc_extra_hash=lambda kw: 1234)

    def args(scales):
        return SimpleNamespace(
            logit_bias=None, repetition_penalty=None, repetition_context_size=20,
            presence_penalty=None, presence_context_size=20,
            frequency_penalty=None, frequency_context_size=20,
            logits_processors=None, enable_thinking=False, _kq_lora=scales)

    bg = fake_bg()
    kw = {"_apc_semantic_hash": 99}
    RG._make_logits_processors(SimpleNamespace(), args((0.0, 0.0)), None)
    _ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw])
    assert kw["_apc_semantic_hash"] == 99                  # bare row: untouched
    kw = {"_apc_semantic_hash": 99}
    RG._make_logits_processors(SimpleNamespace(), args((1.0, 0.0)), None)
    _ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw])
    assert kw["_apc_semantic_hash"] == 99 ^ lora_rows.lora_salt((1.0, 0.0))
    kw = {}
    RG._make_logits_processors(SimpleNamespace(), args((0.0, 1.0)), None)
    _ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw])
    assert kw["_apc_semantic_hash"] == 1234 ^ lora_rows.lora_salt((0.0, 1.0))
    RG._make_logits_processors(SimpleNamespace(), args((0.0, 1.0)), None)
    _ar.BatchGenerator.insert(bg, [[1, 2]])                # no kwargs at all
    seq_kw = bg._unprocessed_sequences[-1][3]
    assert seq_kw["_apc_semantic_hash"] == 1234 ^ lora_rows.lora_salt((0.0, 1.0))


# two adapters on one model: dense and expert targets

class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


def _dense_plan(a, b, scale, path="q_proj"):
    lm = adapter.LoraModule(module_path=path, a=a, b=b, rank=a.shape[0],
                            scale=scale, transform="passthrough")
    return adapter.LoraAdapter(alpha=scale * a.shape[0], arch="qwen3",
                               modules={path: lm})


def test_second_dense_adapter_installs_as_slot_and_rows_select_it():
    mx.random.seed(1)
    m = _Tiny()
    base = m.q_proj
    x = mx.random.normal((3, 4))
    a1, b1 = mx.random.normal((2, 4)), mx.random.normal((4, 2))
    a2, b2 = mx.random.normal((1, 4)), mx.random.normal((4, 1))
    assert modules.install_lora_adapter(m, _dense_plan(a1, b1, 0.5), slot=0) == 1
    assert modules.install_lora_adapter(m, _dense_plan(a2, b2, 2.0), slot=1) == 1
    assert isinstance(m.q_proj, modules.LoRAKQuantLinear)
    assert m.q_proj.base is base and m.q_proj.slots == (0, 1)
    d1 = 0.5 * ((x @ a1.T) @ b1.T)
    d2 = 2.0 * ((x @ a2.T) @ b2.T)
    y0 = base(x)
    assert mx.allclose(m.q_proj(x), y0 + d1 + d2, atol=1e-5)      # static: all on
    lora_rows.ensure_rows(2)
    lora_rows.set_rows([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    y = m.q_proj(x)
    assert mx.allclose(y[0], (y0 + d1)[0], atol=1e-5)
    assert mx.allclose(y[1], (y0 + d2)[1], atol=1e-5)
    assert mx.allclose(y[2], y0[2], atol=1e-5)
    with pytest.raises(ValueError):
        modules.install_lora_adapter(m, _dense_plan(a2, b2, 2.0), slot=1)


def test_second_expert_adapter_installs_as_slot():
    from mlx_lm.models.switch_layers import SwitchGLU

    class _Holder(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.switch_mlp = glu

    mx.random.seed(2)
    d, inter, E, k = 8, 16, 4, 2
    glu = SwitchGLU(d, inter, E)
    for name in ("gate_proj", "up_proj", "down_proj"):
        leaf = getattr(glu, name)
        leaf.weight = mx.random.normal(leaf.weight.shape) * 0.3
    glu.eval()
    model = _Holder(glu)
    x = mx.random.normal((2, 3, d))
    idx = mx.array(np.random.default_rng(0).integers(0, E, size=(2, 3, k)).astype(np.uint32))
    y0 = glu(x, idx)
    a1, b1 = mx.random.normal((E, 1, inter)), mx.random.normal((E, d, 1))
    a2, b2 = mx.random.normal((E, 1, inter)), mx.random.normal((E, d, 1))

    def plan(a, b, scale):
        lm = adapter.LoraModule(module_path="switch_mlp.down_proj", a=a, b=b,
                                rank=1, scale=scale, experts=True)
        return adapter.LoraAdapter(alpha=scale, arch="qwen3moe",
                                   modules={"switch_mlp.down_proj": lm})

    def ref(a, b, scale):
        xe = mx.expand_dims(x, (-2, -3))
        h = glu.activation(glu.up_proj(xe, idx), glu.gate_proj(xe, idx))   # [B,L,k,1,I]
        hn, an, bn, idn = (np.array(h), np.array(a), np.array(b), np.array(idx))
        out = np.zeros((2, 3, k, d), np.float32)
        for bi in range(2):
            for li in range(3):
                for ki in range(k):
                    e = int(idn[bi, li, ki])
                    out[bi, li, ki] = scale * ((hn[bi, li, ki, 0] @ an[e].T) @ bn[e].T)
        return mx.array(out)

    modules.install_lora_adapter(model, plan(a1, b1, 0.5), slot=0)
    modules.install_lora_adapter(model, plan(a2, b2, 1.5), slot=1)
    glu = model.switch_mlp
    assert type(glu).__name__ == "_LoRASwitchGLU"
    assert [lo.slot for lo in glu.down_proj._kq_lora_extra] == [1]
    r1, r2 = ref(a1, b1, 0.5), ref(a2, b2, 1.5)
    assert mx.allclose(glu(x, idx), y0 + r1 + r2, atol=1e-4)
    lora_rows.ensure_rows(2)
    lora_rows.set_rows([[1.0, 0.0], [0.0, 1.0]])
    y = glu(x, idx)
    assert mx.allclose(y[0], (y0 + r1)[0], atol=1e-4)
    assert mx.allclose(y[1], (y0 + r2)[1], atol=1e-4)
    with pytest.raises(ValueError):
        modules.install_lora_adapter(model, plan(a2, b2, 1.5), slot=1)


def test_single_model_serve_registers_bare_base_sibling(tmp_path):
    pytest.importorskip("yaml")
    import gmlx.serve.server as server
    base = tmp_path / "Qwen3-0.6B-Q8_0.gguf"
    base.write_bytes(b"GGUF")
    lora = tmp_path / "pirate.gguf"
    lora.write_bytes(b"GGUF")
    a = SimpleNamespace(model=str(base), adapter=str(lora), mmproj=None,
                        draft_gguf=None, speculative=False, host=None, port=None,
                        budget_gb=None, max_models=None, hf_cache=None,
                        chat_template=None)
    cfg = server._single_model_cfg(a)
    ids = set(cfg.models)
    assert len(ids) == 2 and cfg.defaults.model in ids
    bare = [m for m in cfg.models.values() if m.adapter is None]
    assert len(bare) == 1 and bare[0].id.endswith("-base")
    assert bare[0].path == cfg.models[cfg.defaults.model].path
    assert cfg.models[cfg.defaults.model].adapter == str(lora)
