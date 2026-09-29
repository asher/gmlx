"""gmlx reads the Metal device's info whatever the default device is.

``mx.device_info()`` answers for the default device. ``gmlx distill ...
--cpu`` and a program of its own make the CPU the default device before a
load, and the CPU's answer has none of the Metal keys. Each reader here
runs in that state on a Mac and must see the Metal device's values. Where
a test needs a small value, it replaces that key in the Metal device's
answer only, so a read that asks the default device still gets the CPU's
answer.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="needs a Metal device beside the CPU")

GB = 10**9
WS = "max_recommended_working_set_size"


@pytest.fixture
def on_cpu():
    """The CPU as the default device, set the way ``gmlx distill ... --cpu``
    sets it. The restore after each test puts the default device back."""
    from gmlx.commands.distill import _cpu

    gpu = dict(mx.device_info(mx.gpu))
    _cpu(SimpleNamespace(cpu=True))
    assert WS not in mx.device_info()
    return gpu


def _metal(monkeypatch, **keys):
    """The Metal device's info with ``keys`` in place of its own values."""
    stock = mx.device_info

    def device_info(d=None):
        info = dict(stock(d))
        if (mx.default_device() if d is None else d) == mx.gpu:
            info.update(keys)
        return info

    monkeypatch.setattr(mx, "device_info", device_info)


# The helper.


def test_metal_info_is_the_metal_device_answer(on_cpu):
    from gmlx.device import cpu_default_device, metal_info, metal_value

    assert cpu_default_device()
    assert metal_info() == on_cpu
    assert metal_value("max_buffer_length") == on_cpu["max_buffer_length"]


def test_no_metal_gives_no_info(on_cpu, monkeypatch):
    from gmlx.device import metal_info, metal_value

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert metal_info() is None
    with pytest.raises(KeyError):
        metal_value(WS)


def test_a_replacement_answers_for_the_metal_device(on_cpu, monkeypatch):
    from gmlx.device import metal_info

    _metal(monkeypatch, **{WS: 9})
    assert metal_info()[WS] == 9
    # The configure_stream_cpu rebind and most test doubles take no device.
    monkeypatch.setattr(mx, "device_info", lambda: {WS: 7})
    assert metal_info() == {WS: 7}


def test_every_device_info_call_goes_through_the_helper():
    root = Path(__file__).resolve().parents[1] / "gmlx"
    found = []
    for path in sorted(root.rglob("*.py")):
        if path == root / "device.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "device_info"):
                continue
            base = node.func.value
            if (isinstance(base, ast.Name) and base.id == "mx") or (
                    isinstance(base, ast.Attribute) and base.attr == "metal"):
                found.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    assert found == []


# The table ceiling.


def test_max_buffer_bytes(on_cpu, monkeypatch):
    from gmlx.stream.table_pread import max_buffer_bytes

    monkeypatch.delenv("GMLX_TABLE_MAX_BUFFER", raising=False)
    assert max_buffer_bytes() == on_cpu["max_buffer_length"]


@pytest.fixture
def table_gguf(tmp_path):
    """A deepseek41 GGUF whose one engram table is 33 KB of I8 rows."""
    gguf = pytest.importorskip("gguf")
    path = str(tmp_path / "t.gguf")
    w = gguf.GGUFWriter(path, "deepseek41")
    w.add_tensor("blk.1.engram_embd.weight", np.ones((1000, 33), np.int8),
                 raw_dtype=gguf.GGMLQuantizationType.I8)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), np.float32),
                 raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_the_wire_load_skips_a_table_past_the_ceiling(
        on_cpu, table_gguf, monkeypatch):
    from gmlx.load.wire import load_gguf_wire_bytes

    monkeypatch.delenv("GMLX_TABLE_MAX_BUFFER", raising=False)
    _metal(monkeypatch, max_buffer_length=1000)
    arrays, *_ = load_gguf_wire_bytes(table_gguf, expect_quant=False)
    assert "blk.1.engram_embd.weight" not in arrays
    assert "token_embd.weight" in arrays


# The GPU architecture.


def test_gpu_arch_gen(on_cpu, monkeypatch):
    import gmlx.load.dtypes as dtypes

    monkeypatch.setattr(dtypes, "_arch_gen", None)
    match = dtypes._ARCH_RE.search(on_cpu["architecture"])
    assert match, on_cpu["architecture"]
    want = int(match.group(1))
    assert dtypes.gpu_arch_gen() == want


def test_qwen35_arch_suffix(on_cpu):
    from gmlx.models.qwen35 import attn, language

    for mod in (attn, language):
        mod._qwen3_5_device_arch_suffix.cache_clear()
        try:
            assert mod._qwen3_5_device_arch_suffix() == \
                on_cpu["architecture"][-1:], mod.__name__
        finally:
            mod._qwen3_5_device_arch_suffix.cache_clear()


# The working set.


def test_prefill_score_cap(on_cpu, monkeypatch):
    import gmlx.gen.prefill_decay as pd

    monkeypatch.delenv("GMLX_PREFILL_SCORE_CAP_GB", raising=False)
    monkeypatch.setattr(pd, "_WS_CAP_BYTES", None)
    assert pd._cap_bytes() == pd.cap_bytes_for(on_cpu[WS])


def test_prefill_headroom(on_cpu):
    from gmlx.gen.prefill_decay import headroom_bytes

    assert headroom_bytes() is not None


def test_warm_touch_threshold(on_cpu, monkeypatch):
    from gmlx.load.loader import _warm_touch_threshold_bytes

    monkeypatch.delenv("GMLX_WARM_TOUCH_GB", raising=False)
    _metal(monkeypatch, **{WS: 30 << 30})
    assert _warm_touch_threshold_bytes() == 10 << 30


def test_warm_touch_leaves_a_streaming_bound_model(on_cpu, monkeypatch):
    import gmlx.load.loader as loader

    monkeypatch.delenv("GMLX_RESIDENCY_WARM", raising=False)
    _metal(monkeypatch, **{WS: 100})
    calls = []
    monkeypatch.setattr(loader, "start_populate", lambda p, log: calls.append(p))
    loader._warm_touch_pass([], 1000, log=lambda *_: None, paths=["m.gguf"],
                            batch_bytes=1, threshold_bytes=10**6)
    assert calls == []


def test_native_fp_auto_picks_wire_past_the_working_set(on_cpu, monkeypatch):
    import gmlx.load.loader as loader

    monkeypatch.setenv("GMLX_NATIVE_FP", "auto")
    monkeypatch.setattr(loader, "kq", SimpleNamespace(
        codecs=lambda: ["q4_k", "mxfp4", "nvfp4"]))
    _metal(monkeypatch, **{WS: 50})
    assert loader._resolve_native_fp_wire(
        {"blk.0.ffn_gate_exps.weight": np.zeros(64, np.uint8)},
        {"blk.0.ffn_gate_exps.weight": "mxfp4"}, lambda *_: None)


def test_early_populate(on_cpu, monkeypatch, tmp_path):
    import gmlx.load.populate as populate

    monkeypatch.delenv("GMLX_RESIDENCY_WARM", raising=False)
    monkeypatch.delenv("GMLX_POPULATE_EARLY", raising=False)
    calls = []
    monkeypatch.setattr(populate, "start_populate",
                        lambda p, log: calls.append(p))
    shard = tmp_path / "m.gguf"
    shard.write_bytes(b"\0" * 64)
    populate.maybe_populate_for_load([str(shard)], log=lambda *_: None)
    assert calls == [[str(shard)]]


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
def test_a_teacher_on_the_cpu_device_is_never_over_budget(on_cpu, monkeypatch,
                                                          moe):
    # `gmlx distill cache --cpu` loads the teacher resident: nothing is
    # wired on the CPU device, and expert streaming runs on the GPU.
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import SwitchGLU

    from gmlx.distill.teacher import teacher_over_budget

    model = SwitchGLU(4, 8, 4) if moe else nn.Linear(4, 4)
    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    _metal(monkeypatch, **{WS: 10})
    assert teacher_over_budget(model) == (False, total, 0)
    mx.set_default_device(mx.gpu)
    assert teacher_over_budget(model) == (True, total, 9)


def test_capacity_working_set_and_ceiling(on_cpu, monkeypatch):
    from gmlx.serve import capacity

    monkeypatch.delenv("GMLX_GOV_RESERVE_GB", raising=False)
    assert capacity.working_set_bytes() == on_cpu[WS]
    _metal(monkeypatch, memory_size=64 * GB)
    # The RAM reserve clamps Metal's recommendation: 64 GB less 8 GB.
    assert capacity.ceiling_bytes(100 * GB, 0.05) == 56 * GB


def test_tool_preflight_working_set(on_cpu):
    from gmlx.commands.tool_preflight import working_set_bytes

    assert working_set_bytes() == on_cpu[WS]


def test_drained_admission_room(on_cpu):
    from gmlx.serve.mem_preflight import available_drained_bytes

    assert available_drained_bytes() is not None


def test_extra_preload_over_budget(on_cpu, monkeypatch, tmp_path):
    import gmlx.serve.bridge_vlm as serving
    from gmlx.serve.patches.routes import _preload_extra_over_budget

    shard = tmp_path / "m.gguf"
    shard.write_bytes(b"\0" * 1000)
    monkeypatch.setattr(serving, "resolved_models",
                        lambda: {"m": SimpleNamespace(path=str(shard))})
    _metal(monkeypatch, **{WS: 100})
    assert _preload_extra_over_budget("m")


# Physical RAM.


def test_governor_kernel_floor(on_cpu, monkeypatch):
    from gmlx.serve.governor import _kernel_floor_bytes

    monkeypatch.delenv("GMLX_GOV_KERNEL_FLOOR_GB", raising=False)
    _metal(monkeypatch, memory_size=16 * GB)
    assert _kernel_floor_bytes() == pytest.approx(1.6 * GB)


def test_decode_arena_ram_fraction(on_cpu, monkeypatch):
    import gmlx.stream.budget as budget

    for var in ("GMLX_DECODE_ARENA_GB", "GMLX_GOV_RESERVE_GB",
                "GMLX_GOV_MARGIN", "GMLX_DECODE_RAM_FLOOR_GB",
                "GMLX_DECODE_PAGECACHE_GB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GMLX_DECODE_ARENA_RAM_FRAC", "0.25")
    monkeypatch.setattr(budget, "_available_ram_bytes", lambda *a, **k: None)
    ram = 128 * GB
    _metal(monkeypatch, **{WS: 100 * GB, "memory_size": ram})
    arena = budget._decode_arena_bytes(
        201 * GB, {0: [(0, 0, 200 * GB)]}, 90 * GB, room_bytes=0)
    assert arena == ram // 4 - GB - budget._ram_floor_bytes(ram)


# Wiring: nothing to wire on the CPU device.


def test_mtp_wires_on_the_gpu_only(on_cpu, monkeypatch):
    from gmlx.spec.mtp_load import _wire_big_model

    calls = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda v: calls.append(v) or 0)
    _wire_big_model(None)
    assert calls == []
    mx.set_default_device(mx.gpu)
    _wire_big_model(None)
    assert calls == [on_cpu[WS]]


def test_generate_guard_on_the_cpu_device(on_cpu, monkeypatch):
    import importlib

    from gmlx.distill.eval import generate_guard

    gen = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(gen, "wired_limit", gen.wired_limit)
    stock = gen.wired_limit
    generate_guard()
    assert gen.wired_limit is not stock
    with gen.wired_limit(None, None):
        pass
