"""A row's adapter strength on a whole model: strength s matches the same
adapter written with s times its B at strength 1, and a batch of rows at
different strengths matches each row run alone. The tolerances absorb
rounding of the bfloat16 activations and of batched matmuls."""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx import lora_rows
from gmlx.load import adapter

from test_loader_api import _FFN, _HID, _mint_tiny_llama

_TARGETS = ("model.layers.0.mlp.down_proj", "model.layers.1.self_attn.o_proj")


def _write(path, scale: float):
    mx.random.seed(11)
    pairs = []
    for name in _TARGETS:
        n_in = _FFN if name.endswith("down_proj") else _HID
        a = mx.random.normal((1, n_in))
        b = mx.random.normal((_HID, 1))
        pairs.append((name, a, b * scale))
    adapter.save_lora_adapter(str(path), pairs, alpha=1.0, base_arch="llama",
                              n_head=4, n_head_kv=2, n_layers=2)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    import gmlx

    d = tmp_path_factory.mktemp("strength")
    _mint_tiny_llama(str(d / "tiny.gguf"))
    for name, scale in (("one", 1.0), ("half", 0.5), ("neg", -1.5)):
        _write(d / f"{name}.gguf", scale)

    def model(adapter_name, cli=None):
        m, cfg, _tok = gmlx.load_model(str(d / "tiny.gguf"))
        if adapter_name and cli is None:
            adapter.apply_gguf_adapter(m, cfg, str(d / f"{adapter_name}.gguf"),
                                       base_arch="llama")
        elif adapter_name:
            from gmlx.commands.adapter_flag import parse_adapter
            from gmlx.commands.cli import _apply_cli_adapter

            path = f"{d / adapter_name}.gguf:{cli['strength']}"
            args = SimpleNamespace(adapter=[parse_adapter(path)], arch="llama",
                                   gguf=str(d / "tiny.gguf"))
            _apply_cli_adapter(args, m, cfg, fold=cli["fold"])
        return m

    return model


_X = mx.array([[5, 9, 40, 41, 77, 3]])


def _rows_forward(m, scales):
    lora_rows.configure("rows", 1)
    try:
        lora_rows.set_rows([[s] for s in scales])
        out = m(mx.concatenate([_X] * len(scales), axis=0))
        mx.eval(out)
        return out
    finally:
        lora_rows.configure("static", 1)


@pytest.mark.parametrize("strength,written", [(0.5, "half"), (-1.5, "neg")])
def test_strength_s_matches_s_times_the_delta(rig, strength, written):
    adapted = rig("one")
    want = _rows_forward(rig(written), [1.0])
    got = _rows_forward(adapted, [strength])
    base = rig(None)(_X)
    assert mx.array_equal(_rows_forward(adapted, [0.0]), base)
    moved = mx.abs(got - base).max().item()
    off = mx.abs(got - want).max().item()
    assert off < 0.02 * moved, (off, moved)


def test_a_mixed_strength_batch_matches_each_row_alone(rig):
    m = rig("one")
    scales = [0.0, 1.0, 0.5, -1.5]
    batch = _rows_forward(m, scales)
    for i, s in enumerate(scales):
        alone = _rows_forward(m, [s])
        assert mx.abs(batch[i:i + 1] - alone).max().item() <= 1e-3 * mx.abs(alone).max().item()


def _gap(got, want, base):
    return mx.abs(got - want).max().item(), mx.abs(got - base).max().item()


def test_run_folds_a_flag_strength_into_the_static_scale(rig):
    """gmlx run --adapter PATH:0.5 publishes nothing and gives half the
    delta."""
    lora_rows.configure("static", 1)
    got = rig("one", cli={"strength": 0.5, "fold": True})(_X)
    want = rig("half")(_X)
    off, moved = _gap(got, want, rig(None)(_X))
    assert off < 0.02 * moved, (off, moved)


def test_chat_publishes_a_flag_strength_over_strength_one(rig):
    """gmlx chat --adapter PATH:0.5 installs at strength 1 and publishes
    0.5, which gives half the delta."""
    m = rig("one", cli={"strength": 0.5, "fold": False})
    lora_rows.configure("static", 1)
    try:
        assert mx.allclose(m(_X), rig("one")(_X))       # installed at 1
        lora_rows.publish_tuple((0.5,), 1)
        got = m(_X)
    finally:
        lora_rows.configure("static", 1)
    off, moved = _gap(got, rig("half")(_X), rig(None)(_X))
    assert off < 0.02 * moved, (off, moved)
