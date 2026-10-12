"""A GGUF load stops when its weights leave a model parameter unfilled.

load_model and _install_and_load drop GGUF tensors that match no parameter,
as before. A parameter with no tensor would keep its random initialization,
so the load raises with the missing and unmatched counts and the first
names. A tensor whose shape differs from its parameter's also stops the
load, with both shapes. The tiny llama GGUF from test_loader_api is loaded
with an hf_source config.json that builds a different parameter tree.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from test_loader_api import _FFN, _HID, _TOKENS, _mint_tiny_llama


@pytest.fixture(scope="module")
def gguf(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("unfilled") / "tiny-llama.gguf"
    _mint_tiny_llama(str(p))
    return str(p)


def _source(tmp_path, gguf: str, **changes) -> str:
    """A local hf_source dir holding the GGUF's own config with changes."""
    from gmlx.commands.tool_preflight import _synth_config

    cfg = dict(_synth_config(gguf), **changes)
    d = tmp_path / "src"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def _refused(gguf: str, src: str,
             error: str = "UnfilledParametersError") -> str:
    import gmlx

    with pytest.raises(RuntimeError) as ei:
        gmlx.load_model(gguf, hf_source=src)
    assert type(ei.value).__name__ == error
    return str(ei.value)


def test_a_parameter_without_a_tensor_stops_the_load(gguf, tmp_path):
    # Untied embeddings add lm_head.weight; the tied GGUF ships no output.
    msg = _refused(gguf, _source(tmp_path, gguf, tie_word_embeddings=False))
    assert "Parameters without a tensor (1): lm_head.weight." in msg
    assert "Tensors without a parameter (0)." in msg
    assert "hf_source" in msg and "'llama'" in msg


def test_the_message_counts_both_sides_and_names_the_first(gguf, tmp_path):
    msg = _refused(gguf, _source(tmp_path, gguf, num_hidden_layers=3))
    # Layer 2 holds 9 parameters (two norms, q/k/v/o, gate/up/down).
    assert "fill 20 of 29 model parameters" in msg
    assert "Parameters without a tensor (9): model.layers.2." in msg
    assert "(+4 more)" in msg


def test_a_tensor_of_another_shape_stops_the_load(gguf, tmp_path):
    # A config FFN wider than the GGUF's.
    ffn = _FFN + 8
    msg = _refused(gguf, _source(tmp_path, gguf, intermediate_size=ffn),
                   error="ParameterShapeError")
    assert ("6 of 20 model parameters get a GGUF tensor of another shape: "
            f"model.layers.0.mlp.down_proj.weight (parameter ({_HID}, {ffn}), "
            f"tensor ({_HID}, {_FFN})), ") in msg
    assert "hf_source" in msg and "'llama'" in msg


def test_the_vocab_size_comes_from_the_token_embd_rows(gguf, tmp_path):
    # A config vocab wider than the GGUF's token_embd rows loads with the
    # rows, as a load without hf_source does.
    import gmlx

    v = len(_TOKENS)
    _model, config, _tok = gmlx.load_model(
        gguf, hf_source=_source(tmp_path, gguf, vocab_size=v + 8))
    assert config["vocab_size"] == v


def test_tensors_without_a_parameter_are_dropped(gguf, tmp_path):
    import gmlx

    model, _config, _tok = gmlx.load_model(
        gguf, hf_source=_source(tmp_path, gguf, num_hidden_layers=1))
    assert len(model.model.layers) == 1
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape[:2] == (1, 3)


class _Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4, bias=False)


def test_install_and_load_stops_on_the_same_gap():
    from gmlx.load.loader import _install_and_load

    w = {"a.weight": mx.zeros((4, 4)), "a.bias": mx.zeros((4,)),
         "extra.weight": mx.zeros((4, 4))}
    with pytest.raises(RuntimeError) as ei:
        _install_and_load(_Pair(), w, {}, log=lambda *_a: None)
    assert type(ei.value).__name__ == "UnfilledParametersError"
    msg = str(ei.value)
    assert "fill 2 of 3 model parameters" in msg
    assert "Parameters without a tensor (1): b.weight." in msg
    assert "Tensors without a parameter (1): extra.weight." in msg


def test_install_and_load_stops_on_a_tensor_of_another_shape():
    from gmlx.load.loader import _install_and_load

    w = {"a.weight": mx.zeros((3, 4)), "a.bias": mx.zeros((4,)),
         "b.weight": mx.zeros((4, 4))}
    with pytest.raises(RuntimeError) as ei:
        _install_and_load(_Pair(), w, {}, log=lambda *_a: None)
    assert type(ei.value).__name__ == "ParameterShapeError"
    assert ("1 of 3 model parameters get a GGUF tensor of another shape: "
            "a.weight (parameter (4, 4), tensor (3, 4)).") in str(ei.value)
