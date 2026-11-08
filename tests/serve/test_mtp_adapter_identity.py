"""An MTP entry installs its adapter through the load's ``before_wire`` hook
and still carries the adapter identity: the returned model holds the sha256
of the loaded bytes, ``/v1/models`` lists it, and an adapted row's salt
names it. The MTP load is faked; the adapter is a real minted file."""
from __future__ import annotations

import hashlib
import importlib
import types
from types import SimpleNamespace

import mlx.nn as nn
import numpy as np
import pytest

from helpers import mint_lora_gguf

from gmlx import lora_rows


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


@pytest.fixture
def serving(monkeypatch):
    pytest.importorskip("gmlx._vendor.mlx_vlm")
    import gmlx.serve.bridge_vlm as serving
    serving.clear_resolved_models()
    monkeypatch.setattr(serving, "_build_detokenizer", lambda backend: object())
    monkeypatch.setattr(serving, "StoppingCriteria", lambda eos, backend: object())
    yield serving
    serving.clear_resolved_models()


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = [nn.Module()]
        self.model.layers[0].self_attn = nn.Module()
        self.model.layers[0].self_attn.o_proj = nn.Linear(8, 6, bias=False)


class _Target:
    def __init__(self):
        self.config = {"model_type": "qwen3_5_moe"}
        self.language_model = _Trunk()


def _fake_mtp_load(monkeypatch, seen):
    import gmlx.spec.mtp_load as mtp_load

    def fake(gguf_path, *, draft_gguf_path=None, chat_template=None,
             verbose=True, before_wire=None):
        target = _Target()
        if before_wire is not None:
            before_wire(target.language_model, dict(target.config))
            seen.append(type(target.language_model.model.layers[0].self_attn.o_proj))
        tok = types.SimpleNamespace(eos_token_ids={1})
        return target, object(), dict(target.config), tok

    monkeypatch.setattr(mtp_load, "load_mtp_model", fake)


def _files(tmp_path):
    import gguf

    base = tmp_path / "base.gguf"
    w = gguf.GGUFWriter(str(base), "qwen35moe")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    rng = np.random.default_rng(0)
    ad = tmp_path / "a.gguf"
    mint_lora_gguf(ad, "qwen35moe", {"blk.0.attn_output.weight": (
        rng.standard_normal((2, 8)), rng.standard_normal((6, 2)))})
    return base, ad


def _models_payload(pool):
    from gmlx.serve.patches import routes as sp_routes

    pkg = importlib.import_module("gmlx._vendor.mlx_vlm.server")
    saved = getattr(pkg, "_kq_residency_pool", None)
    pkg._kq_residency_pool = pool
    try:
        return {m["id"]: m for m in sp_routes._models_payload()["data"]}
    finally:
        pkg._kq_residency_pool = saved


class _Pool:
    def __init__(self, entry):
        from gmlx.serve import residency

        self.figures = residency._Figures(entries=(entry,))
        self.resident_chat_entry = (
            residency._ResidencyPool.resident_chat_entry.__get__(self))

    def stats(self):
        e = self.figures.entries[0]
        return {"resident": [{"model_path": e.model_path, "pinned": False,
                              "kind": "chat", "footprint_bytes": 1,
                              "idle_s": 0.0}]}


def test_an_mtp_entry_keeps_its_adapter_identity(tmp_path, serving, monkeypatch):
    from gmlx.config import build_config
    from gmlx.serve import residency

    base, ad = _files(tmp_path)
    seen = []
    _fake_mtp_load(monkeypatch, seen)
    model, _proc, _cfg = serving._load_serveable_mtp(str(base), adapter_gguf=(str(ad),))
    assert [t.__name__ for t in seen] == ["LoRAKQuantLinear"]
    sha = hashlib.sha256(ad.read_bytes()).hexdigest()
    ids = model.gmlx_adapter_identities
    assert [i.sha256 for i in ids] == [sha]

    serving.register_resolved_models(build_config({
        "server": {"model_dirs": [str(tmp_path)]},
        "models": {"m": {"path": "base.gguf", "speculative": True,
                         "adapters": [{"path": "a.gguf", "strength": 1.0}]}}}))
    rm = serving._RESOLVED_MODELS["m"]
    entry = residency._Entry(
        cache_key=(rm.path, None, "auto") + rm.load_signature(),
        model_path=rm.path, response_generator=SimpleNamespace(model=model))
    listed = _models_payload(_Pool(entry))["m"]["adapters"]
    assert [a["sha256"] for a in listed] == [sha]

    from gmlx.load.adapter import AdapterIdentity

    want = lora_rows.row_salt((AdapterIdentity(str(ad), sha, ad.stat().st_size),), (1.0,))
    assert lora_rows.row_salt(ids, (1.0,)) == want != 0


def test_an_mtp_entry_without_an_adapter_stamps_no_identity(tmp_path, serving,
                                                            monkeypatch):
    base, _ad = _files(tmp_path)
    seen = []
    _fake_mtp_load(monkeypatch, seen)
    model, _proc, _cfg = serving._load_serveable_mtp(str(base))
    assert seen == []
    assert model.gmlx_adapter_identities == ()
