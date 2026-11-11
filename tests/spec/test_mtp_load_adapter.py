"""An MTP load installs an adapter through ``before_wire``, before the table
install and the wire. A table-only plan wraps every SwitchGLU in the
streaming offload class, so an expert adapter installed after the load
meets that wrapper and refuses. Every load stage before the wire is faked;
no GGUF, no GPU."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU

from gmlx import lora_rows
import gmlx.load.adapter as adapter
import gmlx.load.modules as modules

PATH = "layers.0.mlp.switch_mlp.down_proj"


@pytest.fixture(autouse=True)
def _static():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Module()]
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.switch_mlp = SwitchGLU(8, 16, 4)


def _plan():
    lm = adapter.LoraModule(module_path=PATH, a=mx.zeros((4, 1, 16)),
                            b=mx.zeros((4, 8, 1)), rank=1, scale=1.0,
                            experts=True)
    return adapter.LoraAdapter(alpha=1.0, arch="qwen35moe", modules={PATH: lm})


class _Tok:
    eos_token_id = 2
    chat_template = None

    def get_vocab(self):
        return {"</s>": 2}


def _offload(trunk, *a, **k):
    """What a table-only plan does to the expert owners."""
    for _, m in trunk.named_modules():
        if isinstance(m, SwitchGLU):
            m.__class__ = type(type(m).__name__ + "_CPUOffload", (type(m),), {})


@pytest.fixture
def fake_load(monkeypatch):
    import gmlx.gen.prefill_decay as prefill_decay
    import gmlx.load.config_synth as config_synth
    import gmlx.load.hadamard as hadamard
    import gmlx.load.tokenizer as tokenizer
    import gmlx.load.wire as wire
    import gmlx.spec.mtp_load as ml
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.table_pread as table_pread

    trunk = _Trunk()
    model = SimpleNamespace(language_model=trunk)
    order = []
    nothing = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(ml, "preflight", lambda path, arch=None: SimpleNamespace(
        arch="qwen35moe", shards=[path],
        refuse_decision=lambda allowed=False: None))
    monkeypatch.setattr(ml, "maybe_populate_for_load", nothing)
    monkeypatch.setattr(ml, "load_gguf_wire_bytes",
                        lambda *a, **k: ({}, {}, {}, {}, {}))
    monkeypatch.setattr(ml, "coalesce_split_experts", lambda a, m: (a, m, 0))
    monkeypatch.setattr(config_synth, "synthesize_config", lambda meta, shapes: {
        "model_type": "qwen3_5_moe", "mtp_num_hidden_layers": 1})
    monkeypatch.setattr(ml, "_resolve_companion_drafter", nothing)
    monkeypatch.setattr(ml, "require_native_head_tensors", nothing)
    monkeypatch.setattr(ml, "read_int", lambda *a: 1)
    monkeypatch.setattr(ml, "first_nonzero_int", lambda *a: 1)
    monkeypatch.setattr(ml, "remap_arrays", lambda *a, **k: ({}, {}, {}))
    monkeypatch.setattr(wire, "strip_nextn_trunk_overflow", lambda *a: 0)
    monkeypatch.setattr(ml, "build_model", lambda cfg, mtp=False: (model, cfg))
    monkeypatch.setattr(ml.rope_swap, "apply", lambda *a: 0)
    monkeypatch.setattr(hadamard, "hadamard_targets_for", nothing)
    monkeypatch.setattr(table_pread, "deferred_tables", nothing)
    monkeypatch.setattr(ml, "_install_and_load", nothing)
    monkeypatch.setattr(ml, "_arm_spec_target", nothing)
    monkeypatch.setattr(ml, "_load_mtp_drafter", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(ml, "_stamp_draft_source", nothing)
    monkeypatch.setattr(tokenizer, "load_tokenizer_from_gguf",
                        lambda *a, **k: _Tok())
    monkeypatch.setattr(ml, "materialize_module_arrays", nothing)
    monkeypatch.setattr(ml, "wait_for_populate", nothing)
    monkeypatch.setattr(ml, "collect_after_load", nothing)
    for name in ("note_resident_model", "note_drafter_held"):
        monkeypatch.setattr(prefill_decay, name, nothing)
    monkeypatch.setattr(es, "install_unflagged_tables",
                        lambda m, *a, **k: (order.append("tables"), _offload(m)))
    monkeypatch.setattr(ml, "_wire_big_model", lambda m: order.append("wire"))
    return ml, trunk, order


def test_before_wire_installs_the_adapter_before_the_table_wrap(fake_load):
    ml, trunk, order = fake_load

    def install(lm, config):
        order.append("adapter")
        assert lm is trunk and config["model_type"] == "qwen3_5_moe"
        modules.install_lora_adapter(lm, _plan())

    ml.load_mtp_model("/m/t.gguf", before_wire=install)
    assert order == ["adapter", "tables", "wire"]
    glu = trunk.layers[0].mlp.switch_mlp
    names = [c.__name__ for c in type(glu).__mro__]
    assert names[0].endswith("_CPUOffload") and "_LoRASwitchGLU" in names
    assert glu.down_proj._kq_lora is not None


def test_an_adapter_after_the_table_wrap_refuses(fake_load):
    ml, trunk, _order = fake_load
    ml.load_mtp_model("/m/t.gguf")
    with pytest.raises(NotImplementedError, match="before placement"):
        modules.install_lora_adapter(trunk, _plan())


def test_the_cli_hook_installs_the_adapter_once(monkeypatch):
    import gmlx.commands.cli as cli
    import gmlx.load.discovery as discovery

    applied = []
    monkeypatch.setattr(adapter, "apply_gguf_adapter",
                        lambda trunk, cfg, path, **k: applied.append(path) or 1)
    monkeypatch.setattr(discovery, "header_meta", lambda p: {"arch": "qwen35moe"})
    assert cli._cli_adapter_hook(SimpleNamespace(adapter=None)) is None
    args = SimpleNamespace(adapter=[("/m/a.gguf", 1.0)], gguf="/m/t.gguf", arch=None)
    trunk = _Trunk()
    cli._cli_adapter_hook(args)(trunk, {})
    cli._apply_cli_adapter(args, SimpleNamespace(language_model=trunk), {})
    assert applied == ["/m/a.gguf"]


def test_an_adapter_failure_names_the_adapter_stage(fake_load, capsys):
    import gmlx.load.loadlog as loadlog

    ml, _trunk, order = fake_load

    def refuse(lm, config):
        raise ValueError("adapter arch llama != base arch qwen35moe")

    with pytest.raises(ValueError):
        with loadlog.load_ui(False, "/m/t.gguf"):
            ml.load_mtp_model("/m/t.gguf", before_wire=refuse)
    err = capsys.readouterr().err
    assert "load failed applying adapter: adapter arch llama" in err
    assert order == []
