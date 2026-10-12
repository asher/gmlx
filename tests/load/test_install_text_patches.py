"""_install_and_load keywords for non-text callers: text_patches=False skips
every text-model install between the Hadamard install and the weight
partition, and act_dtype sets the dtype the cast loop targets. With the
defaults a load installs them all and casts to activation_dtype()."""
from __future__ import annotations

import ast
import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx.load import loader
from gmlx.load.dtypes import activation_dtype
from gmlx.load.invariant_linear import BatchInvariantLinear


class _T(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = [nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4)]


class _F(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 4)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_embedder = _T()
        self.final_layer = _F()
        self.norm_weight = mx.zeros((16,))


def _weights() -> dict:
    w = {
        "t_embedder.mlp.0.weight": mx.ones((8, 4), mx.float32),
        "t_embedder.mlp.0.bias": mx.ones((8,), mx.float32),
        "t_embedder.mlp.2.weight": mx.ones((4, 8), mx.float32),
        "t_embedder.mlp.2.bias": mx.ones((4,), mx.float32),
        "final_layer.linear.weight": mx.ones((4, 16), mx.bfloat16),
        "final_layer.linear.bias": mx.ones((4,), mx.float32),
        "norm_weight": mx.ones((16,), mx.bfloat16),
    }
    mx.eval(list(w.values()))
    return w


def _load(model, **kw):
    loader._install_and_load(model, _weights(), {}, log=lambda *_: None,
                             sanitize=False, **kw)


def _installer_names() -> list[str]:
    """The installs the block calls: each install_* function, and each
    function of a module the loader imports, as module.function."""
    tree = ast.parse(inspect.getsource(loader._install_text_patches))
    names = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id.startswith("install_"):
            names.add(f.id)
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            names.add(f"{f.value.id}.{f.attr}")
    return sorted(names)


def _count_installs(monkeypatch) -> list[str]:
    import gmlx.gen.media_spans as media_spans

    calls: list[str] = []
    for name in _installer_names():
        if "." in name:
            module, attr = name.split(".")
            owner = getattr(loader, module)
        else:
            owner = loader if hasattr(loader, name) else media_spans
            attr = name
        monkeypatch.setattr(owner, attr,
                            lambda *a, _n=name, **k: calls.append(_n) or 0)
    return calls


def test_the_block_holds_the_text_installs():
    names = _installer_names()
    assert "install_batch_invariant_linears" in names
    assert "install_hd512_sdpa" in names
    assert "install_occupancy_fuse" in names
    assert "gemma4_dispatch.configure" in names
    assert len(names) >= 12


def test_text_patches_false_installs_nothing(monkeypatch):
    calls = _count_installs(monkeypatch)
    _load(_Toy(), text_patches=False)
    assert calls == []


def test_the_default_installs_every_patch(monkeypatch):
    calls = _count_installs(monkeypatch)
    _load(_Toy())
    # The cascade stamp installs only after the cascade route does.
    expected = set(_installer_names()) - {"install_cascade_stamp"}
    assert set(calls) == expected


def test_batch_invariant_swap_is_skipped_without_text_patches(monkeypatch):
    monkeypatch.setenv("GMLX_BATCH_INVARIANT", "1")
    m = _Toy()
    _load(m, text_patches=False)
    assert type(m.final_layer.linear) is nn.Linear
    assert type(m.t_embedder.mlp[0]) is nn.Linear
    assert type(m.t_embedder.mlp[2]) is nn.Linear
    control = _Toy()
    _load(control)
    assert type(control.final_layer.linear) is BatchInvariantLinear
    assert type(control.t_embedder.mlp[2]) is BatchInvariantLinear


@pytest.mark.parametrize("act", [None, mx.float16, mx.bfloat16])
def test_act_dtype_sets_the_cast_target(act):
    m = _Toy()
    _load(m, act_dtype=act, text_patches=False, fp32_keep=("t_embedder.",))
    want = activation_dtype() if act is None else act
    assert m.final_layer.linear.bias.dtype == want
    # The loop never touches BF16, and the keep list holds F32.
    assert m.final_layer.linear.weight.dtype == mx.bfloat16
    assert m.norm_weight.dtype == mx.bfloat16
    assert m.t_embedder.mlp[0].weight.dtype == mx.float32
    assert m.t_embedder.mlp[2].bias.dtype == mx.float32
