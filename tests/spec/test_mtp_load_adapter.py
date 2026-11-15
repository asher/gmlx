"""An MTP load installs an adapter through ``before_wire``, before the table
install and the wire. A table-only plan wraps every SwitchGLU in the
streaming offload class, so an expert adapter installed after the load
meets that wrapper and refuses. A load whose install wires its weights in
chunks leaves them to it, untouched by the warm touch. Every load stage
before the wire is faked; no GGUF, no GPU."""
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


def test_a_load_whose_install_wires_in_chunks_stays_resident_after_its_touch(
        fake_load, monkeypatch):
    """A direct load of an arch that gathers its tables, with room for its
    resident set but not for the set twice. Its install judges the set
    against memory not wired, then wires it in chunks. The load's GPU warm
    touch wires each weight it touches, and a touch ahead of the check
    counts the set twice, once as wired memory and once as the set to
    wire, so the experts stream. The touch leaves the weights to the
    install, and the set stays resident. A load with no such install
    keeps its touch."""
    from mlx.utils import tree_flatten

    import gmlx.load.loader as loader
    import gmlx.serve.capacity as capacity
    import gmlx.stream.budget as budget
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.table_stream as ts

    ml, trunk, _order = fake_load
    w = sum(a.nbytes for _, a in tree_flatten(trunk.parameters()))
    room = budget.wire_keep_bytes(0) + w + w // 2
    unwired = [room]
    monkeypatch.setattr(budget, "WIRE_EXEMPT_BYTES", 0)
    monkeypatch.setattr(budget, "unwired_ram_bytes", lambda: unwired[0])

    def touch(arrays, total, **k):
        unwired[0] -= sum(a.nbytes for a in arrays)

    monkeypatch.setattr(loader, "_warm_touch_pass", touch)
    monkeypatch.setattr(ml, "_install_and_load", lambda lm, *a, source_key, **k:
                        loader._warm_mmap_residency(lm, log=print,
                                                    source_key=source_key))
    checks = []
    monkeypatch.setattr(es, "install_unflagged_tables", lambda m, path, **k:
                        checks.append(budget.check_resident_wire(
                            w, room_bytes=0, budget=None)))
    monkeypatch.setattr(capacity, "gathered_table_bytes",
                        lambda path, *a, **k: 1 << 30)
    ml.load_mtp_model("/m/t.gguf")
    assert checks[-1].fits and unwired[0] == room
    assert not ts.resident_wire_guarded("/m/t.gguf")
    monkeypatch.setattr(capacity, "gathered_table_bytes", lambda *a, **k: 0)
    ml.load_mtp_model("/m/t.gguf")
    assert unwired[0] == room - w
