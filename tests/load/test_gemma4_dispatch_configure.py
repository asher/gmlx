"""Both loader install blocks configure the owned Gemma 4 attention
dispatch, after the installs whose state its gates read.

``_install_and_load`` and ``load_model`` each run one block. A spy on
``gmlx.models.gemma4.dispatch.configure`` records the install state when
the loader calls it, which must be the state the load leaves. A load of
the owned text model logs one line naming the open routes, and a load of
another model logs none.
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

    def spy(*args, **kwargs):
        seen.append(_state())
        real(*args, **kwargs)

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


def _gemma4_lines(model) -> list[str]:
    from mlx.utils import tree_flatten

    from gmlx.load.loader import _install_and_load

    lines = []
    _install_and_load(model, dict(tree_flatten(model.parameters())), {},
                      log=lambda *a, **_k: lines.append(" ".join(map(str, a))),
                      sanitize=False)
    return [line for line in lines if "gemma-4" in line]


def test_a_load_of_the_owned_model_names_its_open_routes(calls):
    from gmlx.models.gemma4 import text

    model = text.Model(text.ModelArgs.from_dict(dict(
        model_type="gemma4_text", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, head_dim=16,
        global_head_dim=32, vocab_size=128, vocab_size_per_layer_input=128,
        num_key_value_heads=2, num_kv_shared_layers=0,
        hidden_size_per_layer_input=0, sliding_window=8,
        sliding_window_pattern=2)))
    lines = _gemma4_lines(model)
    names = [name for gate, name in dispatch._ROUTE_NAMES
             if dispatch.gates()[gate]]
    assert names
    assert lines == ["[install] gemma-4 attention routes: " + ", ".join(names)]


def test_a_load_of_another_model_names_no_gemma4_route(calls):
    assert _gemma4_lines(_Linear()) == []
