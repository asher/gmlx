"""Both loader install blocks configure the owned Gemma 4 attention
dispatch, after the installs whose state its gates read.

``_install_and_load`` and ``load_model`` each run one block. A spy on
``gmlx.models.gemma4.dispatch.configure`` records the install state when
the loader calls it, which must be the state the load leaves.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.upstream.attn_hd512 as attn_hd512
from gmlx.models.gemma4 import dispatch
from gmlx.upstream import cascade_sdpa

from test_loader_api import _mint_tiny_llama


def _state():
    return (bool(attn_hd512._installed), bool(cascade_sdpa._installed_route))


@pytest.fixture
def calls(monkeypatch):
    seen = []
    real = dispatch.configure

    def spy():
        seen.append(_state())
        real()

    monkeypatch.setattr(dispatch, "configure", spy)
    # The gates the real configure sets go back afterwards.
    for name in ("_SPARSE", "_CASCADE", "_BATCHED"):
        monkeypatch.setattr(dispatch, name, getattr(dispatch, name))
    return seen


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)


def test_install_and_load_configures_the_dispatch(calls):
    from gmlx.load.loader import _install_and_load

    w = {"a.weight": mx.zeros((4, 4)), "a.bias": mx.zeros((4,))}
    _install_and_load(_Linear(), w, {}, log=lambda *_a: None)
    assert calls == [_state()]


def test_load_model_configures_the_dispatch(calls, tmp_path):
    import gmlx

    gguf = tmp_path / "tiny-llama.gguf"
    _mint_tiny_llama(str(gguf))
    gmlx.load_model(str(gguf))
    assert calls == [_state()]
