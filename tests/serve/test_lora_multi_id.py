#!/usr/bin/env python3
"""Serve binding of text adapters: ids on one GGUF that differ only in
adapters share one resident entry (the adapter union is the signature's
adapter component and the row channel's slot list), the registry swaps in
whole, each request sets its strengths with ``adapters`` or ``lora``,
adapted rows salt their APC hash by file content, and a second adapter
installs on an already adapted model as a further slot on every target.
CPU-only."""
from __future__ import annotations

import importlib
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


def _refs(*paths, strengths=None):
    strengths = strengths or [1.0] * len(paths)
    return tuple(cfgmod.AdapterRef(p, p, float(st)) for p, st in zip(paths, strengths))


def _rm(mid, *adapter_paths, strengths=None, **kw):
    common = dict(path="/p/base.gguf", load={}, cache={}, system=None,
                  speculative=False, mmproj=None, draft_gguf=None, pin=False,
                  ttl_s=None, sampling={})
    common.update(kw)
    return cfgmod.ResolvedModel(id=mid, adapter_list=_refs(*adapter_paths,
                                                            strengths=strengths),
                                **common)


# signature: the adapter union is the adapter component

def test_union_signature_shares_entry_and_forks_on_change():
    a = _rm("a", "/abs/A.gguf")
    b = _rm("b", "/abs/B.gguf")
    bare = _rm("d")
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


def _write(tmp_path, *names):
    for n in names:
        p = tmp_path / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"GGUF" + n.encode())


@pytest.fixture
def serving():
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.serve.bridge_vlm as serving
    serving.clear_resolved_models()
    yield serving
    serving.clear_resolved_models()


def test_registry_fills_union_over_siblings(tmp_path, serving):
    from gmlx.config import build_config
    from gmlx.serve.lora_request import request_lora
    _write(tmp_path, "base.gguf", "A.gguf", "B.gguf")
    base = tmp_path / "base.gguf"
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
    assert rm.adapters == union and rm.adapter_slots == (1,)
    assert request_lora(rm, {}).scales == (0.0, 1.0)
    _path, rm = serving.resolve_request_model("base")
    assert request_lora(rm, {}).scales == (0.0, 0.0)


def test_a_shared_path_is_one_slot_with_each_entrys_default(tmp_path, serving):
    from gmlx.config import build_config
    _write(tmp_path, "base.gguf", "p/s.gguf", "p/t.gguf", "p/u.gguf")
    doc = {"server": {"model_dirs": [str(tmp_path)]},
           "models": {
               "x": {"path": "base.gguf", "adapters": [
                   {"path": "p/u.gguf", "strength": 0.5}, "p/s.gguf"]},
               "y": {"path": "base.gguf", "adapters": [
                   {"path": "p/s.gguf", "strength": 0}, "p/t.gguf"]}}}
    serving.register_resolved_models(build_config(doc))
    x, y = serving._RESOLVED_MODELS["x"], serving._RESOLVED_MODELS["y"]
    want = tuple(str(tmp_path / "p" / n) for n in ("s.gguf", "t.gguf", "u.gguf"))
    assert x.adapters == y.adapters == want
    assert x.adapter_slots == (2, 0) and y.adapter_slots == (0, 1)
    from gmlx.serve.lora_request import request_lora
    assert request_lora(x, {}).scales == (1.0, 0.0, 0.5)
    assert request_lora(y, {}).scales == (0.0, 1.0, 0.0)


def test_the_request_path_reuses_the_registered_adapter_list(tmp_path, serving):
    from gmlx.config import build_config
    _write(tmp_path, "base.gguf", "a.gguf", "b.gguf")
    doc = {"server": {"model_dirs": [str(tmp_path)]},
           "models": {"m": {"path": "base.gguf", "adapters": ["a.gguf", "b.gguf"]}}}
    serving.register_resolved_models(build_config(doc))
    before = serving._RESOLVED_MODELS["m"].adapter_list
    (tmp_path / "b.gguf").unlink()
    _path, rm = serving.resolve_request_model("m")
    assert rm.adapter_list == before


def _three_ids(tmp_path):
    from gmlx.config import build_config
    _write(tmp_path, "base.gguf", "A.gguf", "B.gguf")
    return build_config({"server": {"model_dirs": [str(tmp_path)]},
                         "models": {"base": {"path": "base.gguf"},
                                    "a": {"path": "base.gguf", "adapter": "A.gguf"},
                                    "b": {"path": "base.gguf", "adapter": "B.gguf"}}})


def test_a_request_during_a_reload_sees_the_old_registry_whole(tmp_path, serving,
                                                               monkeypatch):
    cfg = _three_ids(tmp_path)
    serving.register_resolved_models(cfg)
    _p, old = serving.resolve_request_model("b")
    real = serving._resolve_served
    seen = []

    def resolving(mid, c, *a, **k):
        if mid == "a" and not seen:
            seen.append(serving.resolve_request_model("b")[1])
        return real(mid, c, *a, **k)

    monkeypatch.setattr(serving, "_resolve_served", resolving)
    serving.register_resolved_models(cfg)
    assert seen and seen[0].adapters == old.adapters
    assert seen[0].load_signature() == old.load_signature()


def test_a_refused_reload_leaves_the_old_generation_whole(tmp_path, serving):
    from gmlx.config import ConfigError, build_config
    _write(tmp_path, "base.gguf", "A.gguf", "mm.gguf")
    good = build_config({"server": {"model_dirs": [str(tmp_path)]},
                         "models": {"v": {"path": "base.gguf", "mmproj": "mm.gguf"},
                                    "a": {"path": "base.gguf", "adapter": "A.gguf"}}})
    serving.register_resolved_models(good)
    reg = serving._REGISTRY
    vlm = dict(serving._GGUF_VLM_REGISTRY)
    assert vlm
    bad = build_config({"server": {"model_dirs": [str(tmp_path)]},
                        "models": {"a": {"path": "base.gguf",
                                         "adapters": ["A.gguf",
                                                      str(tmp_path / "A.gguf")]},
                                   "v": {"path": "base.gguf", "mmproj": "mm.gguf"}}})
    with pytest.raises(ConfigError, match="name the same file"):
        serving.register_resolved_models(bad)
    assert serving._REGISTRY is reg
    assert serving.server_config() is good
    assert serving._GGUF_VLM_REGISTRY == vlm


def test_the_config_rebinds_with_the_registry(tmp_path, serving, monkeypatch):
    cfg = _three_ids(tmp_path)
    serving.register_resolved_models(cfg)
    new = _three_ids(tmp_path)
    real = serving._resolve_served
    seen = []

    def resolving(mid, c, *a, **k):
        seen.append(serving.server_config() is serving._REGISTRY.cfg)
        return real(mid, c, *a, **k)

    monkeypatch.setattr(serving, "_resolve_served", resolving)
    serving.register_resolved_models(new)
    assert seen and all(seen)
    assert serving.server_config() is new is serving._REGISTRY.cfg


def test_a_request_during_the_self_heal_sees_one_registry(tmp_path, serving,
                                                         monkeypatch):
    cfg = _three_ids(tmp_path)
    (tmp_path / "A.gguf").unlink()
    serving.register_resolved_models(cfg)
    assert "a" not in serving._RESOLVED_MODELS
    _p, old = serving.resolve_request_model("b")
    _write(tmp_path, "A.gguf")
    real = serving._resolve_served
    seen = []

    def resolving(mid, c, *a, **k):
        if mid == "a" and not seen:
            seen.append(serving.resolve_request_model("b")[1])
        return real(mid, c, *a, **k)

    monkeypatch.setattr(serving, "_resolve_served", resolving)
    assert serving.reregister_missing_models()
    assert seen[0].adapters == old.adapters
    _p, new = serving.resolve_request_model("b")
    assert len(new.adapters) == 2


# request fields: adapters and lora


def _spec(tmp_path, items, *, model_id="m", union=None):
    paths = [str(tmp_path / w) for w, _st in items]
    refs = tuple(cfgmod.AdapterRef(w, p, float(st))
                 for (w, st), p in zip(items, paths))
    union = tuple(sorted(set(paths) | set(union or ())))
    return SimpleNamespace(id=model_id, adapter_list=refs, adapters=union,
                           adapter_slots=tuple(union.index(p) for p in paths))


def _req(spec, model_dirs=(), **fields):
    from gmlx.serve.lora_request import request_lora
    return request_lora(spec, fields, model_dirs=list(model_dirs)).scales


def _refused(spec, match, model_dirs=(), **fields):
    from gmlx.serve.lora_request import AdapterRequestError, request_lora
    with pytest.raises(AdapterRequestError, match=match):
        request_lora(spec, fields, model_dirs=list(model_dirs))


def test_adapters_set_listed_strengths_and_keep_the_rest(tmp_path):
    spec = _spec(tmp_path, [("p/a.gguf", 0.0), ("p/b.gguf", 0.25)])
    assert _req(spec) == (0.0, 0.25)
    assert _req(spec, adapters=[{"id": "p/a.gguf", "strength": 0.8}]) == (0.8, 0.25)
    assert _req(spec, adapters=["b.gguf"]) == (0.0, 1.0)            # bare: 1
    assert _req(spec, adapters=[{"id": "a", "strength": -2}]) == (-2.0, 0.25)   # stem


def test_lora_sets_listed_scales_and_zeroes_the_rest(tmp_path):
    spec = _spec(tmp_path, [("p/a.gguf", 0.5), ("p/b.gguf", 0.25)])
    assert _req(spec, lora=[{"id": 1, "scale": 1.5}]) == (0.0, 1.5)
    assert _req(spec, lora=[]) == (0.5, 0.25)                        # defaults


def test_ids_name_paths_across_folders_and_refuse_ambiguity(tmp_path):
    spec = _spec(tmp_path, [("one/s.gguf", 0.0), ("two/s.gguf", 0.0)])
    assert _req(spec, adapters=[{"id": "two/s.gguf", "strength": 1}]) == (0.0, 1.0)
    assert _req(spec, adapters=[{"id": "two/./s.gguf", "strength": 1}]) == (0.0, 1.0)
    _refused(spec, "adapter 's.gguf' matches adapters 0 and 1 of model 'm'; name it "
             "by its path in the config, or use lora with its index",
             adapters=["s.gguf"])


def test_an_id_under_a_model_folder_names_its_file(tmp_path):
    root = tmp_path / "root"
    _write(root, "deep/x.gguf")
    spec = SimpleNamespace(id="m", adapter_list=(cfgmod.AdapterRef(
        str(root / "deep" / "x.gguf"), str(root / "deep" / "x.gguf"), 0.0),),
        adapters=(str(root / "deep" / "x.gguf"),), adapter_slots=(0,))
    assert _req(spec, [str(root)], adapters=["deep/x.gguf"]) == (1.0,)


def test_unknown_sibling_and_absolute_ids_are_refused(tmp_path):
    spec = _spec(tmp_path, [("p/a.gguf", 0.0)], union=[str(tmp_path / "q/sib.gguf")])
    _refused(spec, "adapter 'sib.gguf' is not an adapter of model 'm'; a request "
             "sets the strength of the adapters its model entry loads",
             adapters=["sib.gguf"])
    _refused(spec, "never an absolute path", adapters=[str(tmp_path / "p/a.gguf")])
    for bad in ("~/a.gguf", "../a.gguf", "p\\a.gguf"):
        _refused(spec, "never an absolute path", adapters=[bad])


@pytest.mark.parametrize("fields,match", [
    ({"adapters": "a"}, "adapters must be a list of {id, strength}"),
    ({"adapters": [3]}, "each adapter is {id, strength}, with id a string"),
    ({"adapters": [{"id": "a", "x": 1}]},
     r"adapter fields \['x'\] are unknown; an adapter is {id, strength}"),
    ({"adapters": [{"id": "a", "strength": 101}]},
     r"adapters\[0\].strength must be a finite number from -100 to 100"),
    ({"adapters": ["a", "p/a.gguf"]}, "adapter 'p/a.gguf' is listed twice"),
    ({"lora": {"id": 0}}, "lora must be a list of {id, scale}"),
    ({"lora": [{"id": "0", "scale": 1}]}, r"lora\[0\].id must be an integer"),
    ({"lora": [{"id": 5, "scale": 1}]},
     "lora id 5 is out of range; model 'm' has 2 adapters \\(ids 0 to 1\\)"),
    ({"lora": [{"id": 0, "scale": 1}, {"id": 0, "scale": 2}]},
     "lora id 0 is listed twice"),
    ({"lora": [{"id": 0}]}, r"lora\[0\].scale must be a finite number from -100 to 100"),
])
def test_malformed_fields_are_refused(tmp_path, fields, match):
    spec = _spec(tmp_path, [("p/a.gguf", 0.0), ("p/b.gguf", 0.0)])
    _refused(spec, match, **fields)


def test_both_spellings_are_refused(tmp_path):
    from fastapi import HTTPException

    from gmlx.serve.lora_request import request_lora
    spec = _spec(tmp_path, [("p/a.gguf", 0.0)])
    with pytest.raises(HTTPException) as e:
        request_lora(spec, {"adapters": [], "lora": []}, model_dirs=[])
    assert e.value.status_code == 400
    assert e.value.detail == "Set adapters or lora, not both."
    assert _req(spec, adapters=None, lora=[{"id": 0, "scale": 2}]) == (2.0,)


def test_a_model_without_adapters_takes_only_empty_fields(tmp_path):
    bare = SimpleNamespace(id="m", adapter_list=(), adapters=("/u/x.gguf",),
                           adapter_slots=())
    for spec in (bare, None):
        assert _req(spec, adapters=[], lora=None) == ((0.0,) if spec else ())
        assert _req(spec, lora=[]) == ((0.0,) if spec else ())
    _refused(bare, "model 'm' loads no adapters; adapters and lora apply only to a "
             "model entry with adapters", lora=[{"id": 0, "scale": 1}])
    _refused(None, "loads no adapters", adapters=["a"])


# row salt: names files by content, never by slot


def _identity(path):
    import hashlib
    from gmlx.load.adapter import AdapterIdentity
    data = open(path, "rb").read()
    return AdapterIdentity(str(path), hashlib.sha256(data).hexdigest(), len(data))


def _inserted_hash(lora, base=99):
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    from gmlx._vendor.mlx_vlm.generate import ar as _ar
    from gmlx.serve.row_sampling import RowSettings
    lora_rows.install_row_channel()
    bg = SimpleNamespace(max_tokens=8, logits_processors=None,
                         _unprocessed_sequences=[], uid_count=0,
                         apc_manager=object(), _apc_extra_hash=lambda kw: 1234)
    kw = {"_apc_semantic_hash": base}
    _ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw],
                              row_settings=[RowSettings(lora=lora)])
    return kw["_apc_semantic_hash"]


def _salted_row(spec, identities, **fields):
    from gmlx.serve.lora_request import request_lora, salted
    return salted(request_lora(spec, fields, model_dirs=[]), identities)


def test_two_entries_on_one_gguf_with_other_files_salt_apart(tmp_path):
    """X with [A] and Y with [B] under different load profiles: one slot
    each, one disk namespace. The slot tuples are equal; the salts are
    not."""
    _write(tmp_path, "A.gguf", "B.gguf")
    lora_rows.ensure_rows(1)
    x = _spec(tmp_path, [("A.gguf", 1.0)])
    y = _spec(tmp_path, [("B.gguf", 1.0)])
    hx = _inserted_hash(_salted_row(x, (_identity(tmp_path / "A.gguf"),)))
    hy = _inserted_hash(_salted_row(y, (_identity(tmp_path / "B.gguf"),)))
    assert hx != hy and hx != 99 and hy != 99


def test_one_entry_restarted_on_another_file_salts_apart(tmp_path):
    _write(tmp_path, "A.gguf", "B.gguf")
    lora_rows.ensure_rows(1)
    first = _spec(tmp_path, [("A.gguf", 1.0)])
    second = _spec(tmp_path, [("B.gguf", 1.0)])
    assert _inserted_hash(_salted_row(first, (_identity(tmp_path / "A.gguf"),))) \
        != _inserted_hash(_salted_row(second, (_identity(tmp_path / "B.gguf"),)))


def test_a_rewrite_of_the_same_path_salts_apart(tmp_path):
    from gmlx.load.adapter import AdapterIdentity

    path = tmp_path / "A.gguf"

    def loaded(a):
        adapter.save_lora_adapter(
            str(path), [("model.layers.0.mlp.down_proj", a, mx.ones((4, 1)))],
            alpha=2.0, base_arch="llama", n_head=4, n_head_kv=2, n_layers=1)
        plan = adapter.load_lora_adapter(str(path), base_arch="llama")
        return (AdapterIdentity(str(path), plan.sha256, plan.nbytes),)

    spec = _spec(tmp_path, [("A.gguf", 1.0)])
    before = _salted_row(spec, loaded(mx.ones((1, 8)))).salt
    after = _salted_row(spec, loaded(-mx.ones((1, 8)))).salt
    assert before != after


def test_a_slot_count_grow_keeps_the_salt(tmp_path):
    _write(tmp_path, "A.gguf")
    spec = _spec(tmp_path, [("A.gguf", 1.0)])
    ids = (_identity(tmp_path / "A.gguf"),)
    lora_rows.ensure_rows(1)
    before = _inserted_hash(_salted_row(spec, ids))
    lora_rows.ensure_rows(5)
    assert _inserted_hash(_salted_row(spec, ids)) == before


def test_base_rows_salt_zero_and_need_no_identity(tmp_path):
    from gmlx.serve.lora_request import AdapterIdentityMissing
    spec = _spec(tmp_path, [("A.gguf", 0.0)])
    row = _salted_row(spec, None)
    assert row.salt == 0 and _inserted_hash(row) == 99
    with pytest.raises(AdapterIdentityMissing,
                       match="model 'm' has no loaded adapter identities; this is a "
                             "gmlx defect"):
        _salted_row(spec, None, adapters=["A.gguf"])
    _write(tmp_path, "A.gguf")
    with pytest.raises(AdapterIdentityMissing):        # another union's identities
        _salted_row(spec, (_identity(tmp_path / "A.gguf"),) * 2, adapters=["A.gguf"])


def test_the_seam_never_raises_a_missing_identity(tmp_path):
    from gmlx.serve.lora_request import request_lora
    spec = _spec(tmp_path, [("A.gguf", 0.0)])
    row = request_lora(spec, {"adapters": ["A.gguf"]}, model_dirs=[])
    assert row.salt is None and row.scales == (1.0,)


def test_ensure_rows_only_grows():
    lora_rows.ensure_rows(1)
    lora_rows.ensure_rows(3)
    assert lora_rows.mode() == "rows" and lora_rows.n_slots() == 3
    lora_rows.ensure_rows(2)                               # never shrinks
    assert lora_rows.n_slots() == 3


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
    from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

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
    bare = [m for m in cfg.models.values() if not m.adapters]
    assert len(bare) == 1 and bare[0].id.endswith("-base")
    assert bare[0].path == cfg.models[cfg.defaults.model].path
    assert cfg.models[cfg.defaults.model].adapters == ((str(lora), 1.0),)


# /v1/models: index, name, default strength, sha256 of the loaded bytes


class _ListingPool:
    """The pool surface ``_models_payload`` reads, with the real
    ``resident_chat_entry`` over a ``figures`` snapshot."""

    def __init__(self, entries=()):
        from gmlx.serve import residency
        self.figures = residency._Figures(entries=tuple(entries))
        self.resident_chat_entry = residency._ResidencyPool.resident_chat_entry.__get__(self)

    def stats(self):
        return {"resident": [{"model_path": e.model_path, "pinned": False,
                              "kind": "chat", "footprint_bytes": 1, "idle_s": 0.0}
                             for e in self.figures.entries]}


def _listing_cfg(tmp_path):
    from gmlx.config import build_config
    _write(tmp_path, "base.gguf", "loras/a.gguf", "loras/b.gguf")
    return build_config({"server": {"model_dirs": [str(tmp_path)]},
                         "models": {"m": {"path": "base.gguf", "adapters": [
                             {"path": "loras/a.gguf", "strength": 0.0},
                             {"path": "loras/b.gguf", "strength": 0.5}]},
                                    "bare": {"path": "base.gguf"}}})


def _resident(tmp_path, serving, identities):
    from gmlx.serve import residency
    rm = serving._RESOLVED_MODELS["m"]
    model = SimpleNamespace()
    if identities is not None:
        object.__setattr__(model, "gmlx_adapter_identities", identities)
    return residency._Entry(cache_key=(rm.path, None, "auto") + rm.load_signature(),
                            model_path=rm.path,
                            response_generator=SimpleNamespace(model=model))


def _payload(pool):
    from gmlx.serve.patches import routes as sp_routes
    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    saved = getattr(pkg, "_kq_residency_pool", None)
    pkg._kq_residency_pool = pool
    try:
        return {m["id"]: m for m in sp_routes._models_payload()["data"]}
    finally:
        pkg._kq_residency_pool = saved


def test_models_lists_adapters_with_the_loaded_hash(tmp_path, serving):
    import json
    serving.register_resolved_models(_listing_cfg(tmp_path))
    ids = tuple(_identity(tmp_path / "loras" / n) for n in ("a.gguf", "b.gguf"))
    entry = _resident(tmp_path, serving, ids)
    by_id = _payload(_ListingPool([entry]))
    assert by_id["m"]["adapters"] == [
        {"index": 0, "name": "a.gguf", "strength": 0.0, "sha256": ids[0].sha256},
        {"index": 1, "name": "b.gguf", "strength": 0.5, "sha256": ids[1].sha256}]
    assert by_id["bare"]["adapters"] == []
    assert str(tmp_path) not in json.dumps(by_id)


def test_models_lists_null_hashes_with_no_resident_copy(tmp_path, serving):
    serving.register_resolved_models(_listing_cfg(tmp_path))
    for pool in (None, _ListingPool()):                   # never loaded, unloaded
        got = _payload(pool)["m"]["adapters"]
        assert [a["sha256"] for a in got] == [None, None]
        assert [a["index"] for a in got] == [0, 1]


def test_models_lists_null_hashes_for_an_entry_mid_teardown(tmp_path, serving):
    serving.register_resolved_models(_listing_cfg(tmp_path))
    entry = _resident(tmp_path, serving, None)
    entry.response_generator = None
    got = _payload(_ListingPool([entry]))["m"]["adapters"]
    assert [a["sha256"] for a in got] == [None, None]
