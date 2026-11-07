"""Expert LoRA on a fused MoE block needs the routed experts in memory, so a
load that streams them refuses: in the streaming install, before any table
wrap, wire or credit, and before the load from the headers, at the serve
pool's explicit and for-memory sites and in the run and chat verbs. Faked budgets, plans and builds over
minted headers; no GPU."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.nn as nn
import numpy as np
import pytest

from gmlx._vendor.mlx_lm.models.switch_layers import SwitchGLU
from helpers import mint_lora_gguf

import gmlx.load.modules as modules
from gmlx.load.adapter import fused_expert_lora_refusal

GB = 1024**3


class _Model(nn.Module):
    def __init__(self, fused_expert=True, dense_only=False):
        super().__init__()
        self.layers = [nn.Module()]
        mlp = nn.Module()
        mlp.switch_mlp = SwitchGLU(8, 16, 4)
        self.layers[0].mlp = mlp
        if fused_expert or dense_only:
            object.__setattr__(mlp, "_kq_lora_slots", (0,))
        if fused_expert:
            object.__setattr__(mlp, "_kq_expert_lora", True)


def test_the_refusal_names_the_first_adapted_fused_block():
    with pytest.raises(NotImplementedError, match="1 block carries expert targets, "
                       "the first at layers.0.mlp"):
        modules.refuse_streamed_fused_expert_lora(_Model())


@pytest.mark.parametrize("model", [
    _Model(fused_expert=False), _Model(fused_expert=False, dense_only=True)],
    ids=["unfused", "dense_only"])
def test_unfused_and_dense_only_adapters_pass(model):
    modules.refuse_streamed_fused_expert_lora(model)


# the install's call sites

class _Past(Exception):
    """The install got past every refusal point."""


@pytest.fixture
def install(monkeypatch):
    import gmlx.stream.expert_streaming as es
    import gmlx.stream.table_stream as ts

    seen = []
    monkeypatch.setattr(es, "stream_budget_bytes", lambda: 100 * GB)
    monkeypatch.setattr(ts, "file_table_bytes", lambda m: 0)
    monkeypatch.setattr(ts, "table_bytes", lambda m: 0)
    monkeypatch.setattr(ts, "tables_for_memory", lambda p: False)
    monkeypatch.setattr(ts, "streamed_table_array_ids", lambda m: ())
    monkeypatch.setattr(ts, "install_table_streaming",
                        lambda m: (seen.append("tables"), (0, ["t"]))[1])
    monkeypatch.setattr(es, "credit_streamed",
                        lambda *a, **k: (seen.append("credit"), 0.0)[1])

    def past(*a, **k):
        raise _Past

    monkeypatch.setattr(es, "end_build_credit", past)

    def run(plan, wired=True):
        monkeypatch.setattr(ts, "stream_plan", lambda *a, **k: plan)
        monkeypatch.setattr(es, "_wire_resident_experts",
                            lambda *a, **k: (seen.append("wire"), wired)[1])
        model = _Model()
        try:
            es.install_expert_streaming(model)
        finally:
            run.model = model
        return seen

    return run, seen


def test_a_streamed_plan_refuses_before_the_table_install(install):
    run, seen = install
    with pytest.raises(NotImplementedError, match="expert LoRA on a fused MoE"):
        run(SimpleNamespace(tables=True, experts=True, table_kept=False))
    assert seen == []
    assert "_kq_streaming" not in vars(run.model)


def test_a_table_only_plan_whose_wire_fails_refuses_before_the_credits(install):
    run, seen = install
    with pytest.raises(NotImplementedError, match="expert LoRA on a fused MoE"):
        run(SimpleNamespace(tables=True, experts=False, table_kept=False),
            wired=False)
    assert seen == ["tables", "wire"]


def test_a_table_only_plan_whose_wire_holds_loads(install):
    run, seen = install
    with pytest.raises(_Past):
        run(SimpleNamespace(tables=True, experts=False, table_kept=False))
    assert seen == ["tables", "wire", "credit"]


# the header check

E, R, D, INTER = 4, 1, 8, 16


def _base(tmp_path, arch):
    import gguf

    path = tmp_path / f"{arch}.gguf"
    w = gguf.GGUFWriter(str(path), arch)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _adapter(tmp_path, name, expert):
    rng = np.random.default_rng(0)
    pairs = {"blk.0.attn_output.weight": (rng.standard_normal((2, D)),
                                          rng.standard_normal((6, 2)))}
    if expert:
        pairs["blk.0.ffn_down_exps.weight"] = (
            rng.standard_normal((E, R, INTER)), rng.standard_normal((E, D, R)))
    path = tmp_path / name
    mint_lora_gguf(path, "qwen35moe", pairs)
    return str(path)


@pytest.fixture
def files(tmp_path):
    return SimpleNamespace(
        qwen35moe=_base(tmp_path, "qwen35moe"), gptoss=_base(tmp_path, "gpt-oss"),
        expert=_adapter(tmp_path, "expert.gguf", True),
        dense=_adapter(tmp_path, "dense.gguf", False))


@pytest.mark.parametrize("base,adapters,env,mtp,refuses", [
    ("qwen35moe", ("expert",), {}, False, True),
    ("gptoss", ("expert",), {}, False, False),
    ("qwen35moe", ("expert",), {"GMLX_FUSED_MOE_BLOCK": "0"}, False, False),
    ("qwen35moe", ("dense",), {}, False, False),
    ("qwen35moe", ("dense", "expert"), {}, False, True),
    ("qwen35moe", ("expert",), {}, True, False),
    ("qwen35moe", ("expert",), {"MLX_VLM_GGUF_SPECULATIVE": "0"}, True, True),
], ids=["expert", "gpt-oss", "block-off", "dense", "union", "mtp", "mtp-off"])
def test_the_header_check(files, monkeypatch, base, adapters, env, mtp, refuses):
    for name in ("GMLX_FUSED_MOE", "GMLX_FUSED_MOE_BLOCK", "MLX_VLM_GGUF_SPECULATIVE"):
        monkeypatch.delenv(name, raising=False)
    why = fused_expert_lora_refusal(getattr(files, base),
                                    [getattr(files, a) for a in adapters],
                                    env, mtp=mtp)
    assert (why is not None) is refuses
    if refuses:
        assert "blk.0.ffn_down_exps.weight" in why and "GMLX_FUSED_MOE_BLOCK=0" in why


# the serve pool sites

def _pool(monkeypatch, files):
    from gmlx.serve import residency
    from gmlx.serve.residency import _ResidencyPool, _RuntimeProxy

    proxy = _RuntimeProxy(SimpleNamespace(metrics=None))

    def stock_get(model_path, adapter_path, *, model_kind="auto"):
        proxy.apc_manager = proxy.response_generator = object()
        proxy.model_cache = {"cache_key": (model_path, adapter_path, model_kind),
                             "model_path": model_path, "model": object()}

    pool = _ResidencyPool(proxy, stock_get, lambda: True, int(40 * GB), (),
                          footprint_fn=lambda p: int(10 * GB))
    monkeypatch.setattr(residency, "_streaming_footprint",
                        lambda p, fb, env, held_bytes=0: fb)
    monkeypatch.setattr(residency, "_plain_footprint", lambda p, fb, env=None: fb)
    hits = []
    pool._seam = hits.append
    return pool, hits


def _spec(files, stream, speculative=False):
    return SimpleNamespace(stream=stream, speculative=speculative,
                           effective_adapters=lambda: (files.expert,))


@pytest.mark.host_gb(64)
def test_an_explicit_stream_refuses_before_any_eviction(files, monkeypatch):
    pool, hits = _pool(monkeypatch, files)
    idle = pool.acquire("/m/other.gguf", None, "auto")
    pool.release(idle)
    built = []
    real_build = pool._build
    pool._build = lambda *a, **k: built.append(a) or real_build(*a, **k)
    with pytest.raises(ValueError, match="expert LoRA on a fused MoE block"):
        pool.acquire(files.qwen35moe, None, "auto",
                     build_spec=_spec(files, "experts"))
    assert idle.cache_key in pool._entries
    assert hits.count("room") == 1          # the idle entry's own build only
    assert built == []


@pytest.mark.host_gb(64)
@pytest.mark.parametrize("speculative,mtp_off", [(False, False), (True, True)],
                         ids=["plain", "mtp_off"])
def test_a_for_memory_stream_refuses_before_it_latches(files, monkeypatch,
                                                       speculative, mtp_off):
    import gmlx.stream.table_stream as ts

    pool, _hits = _pool(monkeypatch, files)
    keep = SimpleNamespace(fits=False, reason="budget", mtp_off=mtp_off,
                           table_bytes=0, mtp_off_line=lambda s: "")
    monkeypatch.setattr(pool, "_keep_check",
                        lambda path, env, spec: (keep, None, 0))
    latched = []
    monkeypatch.setattr(ts, "stream_experts_for_memory",
                        lambda path, why="": why and latched.append(why))
    gated = []
    monkeypatch.setattr(pool, "_gate", lambda *a, **k: gated.append(a))
    with pytest.raises(ValueError, match="expert LoRA on a fused MoE block"):
        pool.acquire(files.qwen35moe, None, "auto",
                     build_spec=_spec(files, None, speculative))
    assert latched == [] and gated == []
    assert pool._building == 0 and pool._loading is None


@pytest.mark.parametrize("flags,refuses", [
    ({"stream_experts": True}, True), ({"stream_cpu": True}, True),
    ({"stream_experts": True, "speculative": True}, False),
    ({"stream_experts": True, "native_mtp": True}, False),
    ({"stream_experts": True, "draft_gguf": "/m/d.gguf"}, False),
    ({"stream_experts": True, "draft_gguf": "/m/d.gguf", "draft_gguf_auto": True}, True),
    ({"stream_experts": True, "speculative": True, "no_speculative": True}, True),
    ({}, False)],
    ids=["experts", "cpu", "mtp", "native-mtp", "draft", "auto-draft", "no-mtp", "plain"])
def test_the_cli_checks_a_streamed_adapter_before_the_load(files, monkeypatch,
                                                          flags, refuses):
    import gmlx.commands.cli as cli

    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    args = SimpleNamespace(gguf=files.qwen35moe, adapter=[(files.expert, 1.0)],
                           stream_experts=False, stream_cpu=False,
                           speculative=None)
    vars(args).update(flags)
    assert (cli._streamed_adapter_refusal(args) is not None) is refuses


@pytest.mark.parametrize("flags,refuses", [
    (["--stream-experts"], True), (["--stream-experts", "--mtp"], False)],
    ids=["plain", "mtp"])
def test_chat_checks_a_streamed_adapter_before_the_load(files, monkeypatch, capsys,
                                                       flags, refuses):
    import gmlx.tui.chat as chat

    monkeypatch.setenv("GMLX_NO_FAMILY_DEFAULTS", "1")
    monkeypatch.delenv("MLX_VLM_GGUF_SPECULATIVE", raising=False)
    monkeypatch.setattr("gmlx.commands.cli.maybe_load_from_config",
                        lambda *a, **k: None)

    class _Checked(Exception):
        """Chat got past its checks before the load."""

    def past(*a, **k):
        raise _Checked

    monkeypatch.setattr(chat, "_wire_input", past)
    argv = [files.qwen35moe, "--adapter", files.expert, *flags]
    if refuses:
        assert chat.cmd_chat(argv) == 2
        assert "--adapter: adapter expert.gguf adapts expert tensor" in \
            capsys.readouterr().err
    else:
        with pytest.raises(_Checked):
            chat.cmd_chat(argv)
