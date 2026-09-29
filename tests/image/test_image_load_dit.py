"""load_dit on small synthetic GGUFs through a test family: refusals before
any bytes are read, strict name equality, the owned casts, and the text
patches kept out of an image load."""
from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from image_fixtures import bf16_bits, write_gguf

from gmlx.image import fingerprint as fp
from gmlx.image import loader as image_loader
from gmlx.image import registry
from gmlx.image.fingerprint import Match
from gmlx.image.registry import ConstantsRow, Family


class _T(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = [nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4)]


class _F(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 16)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.ones((64,))


class _Toy(nn.Module):
    def __init__(self, drop_pad=False):
        super().__init__()
        self.proj = nn.Linear(64, 32)
        self.t_embedder = _T()
        self.norm = _Norm()
        self.final_layer = _F()
        if not drop_pad:
            self.x_pad_token = mx.zeros((1, 64))


def _family(build=lambda cfg: _Toy()) -> Family:
    def fingerprint(view):
        return Match() if view.has("proj.weight") else None

    return Family(
        name="toy", title="Toy", order=0, arch_strings=frozenset({None}),
        strip_prefixes=("model.diffusion_model.",), fingerprint=fingerprint,
        synth_config=lambda view: {}, constants=ConstantsRow({}, "r", "v"),
        build=build, fp32_keep=("t_embedder.",))


@pytest.fixture
def toy(monkeypatch):
    fam = _family()
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    return fam


def _tensors(prefix: str = "") -> dict:
    import gguf

    rng = np.random.default_rng(0)

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    q = gguf.quants.quantize(f32(32, 64), gguf.GGMLQuantizationType.Q8_0)
    t = {
        "proj.weight": (q, "Q8_0"),
        "proj.bias": (f32(32), "F32"),
        "t_embedder.mlp.0.weight": (bf16_bits(f32(8, 4)), "BF16"),
        "t_embedder.mlp.0.bias": (f32(8), "F32"),
        "t_embedder.mlp.2.weight": (bf16_bits(f32(4, 8)), "BF16"),
        "t_embedder.mlp.2.bias": (f32(4), "F32"),
        "norm.weight": (bf16_bits(f32(64)), "BF16"),
        "final_layer.linear.weight": (bf16_bits(f32(16, 64)), "BF16"),
        "final_layer.linear.bias": (f32(16), "F32"),
        "x_pad_token": (f32(1, 64), "F32"),
    }
    return {prefix + k: v for k, v in t.items()}


def _load(path, **kw):
    return image_loader.load_dit(str(path), family="toy",
                                 log=lambda *_: None, **kw)


def test_load_keeps_wire_bytes_and_makes_the_owned_casts(toy, tmp_path,
                                                        monkeypatch):
    monkeypatch.setenv("GMLX_BATCH_INVARIANT", "1")
    p = write_gguf(tmp_path / "toy.gguf", _tensors("model.diffusion_model."))
    model, info = _load(p)
    from mlx_kquant.nn import KQuantLinear

    assert type(model.proj) is KQuantLinear and info.n_kquant == 1
    assert info.identified.prefix == "model.diffusion_model."
    # BF16 timestep matrices become owned fp32, their F32 biases stay fp32.
    assert model.t_embedder.mlp[0].weight.dtype == mx.float32
    assert model.t_embedder.mlp[2].bias.dtype == mx.float32
    assert info.n_owned_casts == 2
    # A BF16 rank 1 norm weight passes the loop untouched.
    assert model.norm.weight.dtype == mx.bfloat16
    assert model.final_layer.linear.weight.dtype == mx.bfloat16
    assert model.final_layer.linear.bias.dtype == mx.bfloat16
    assert model.x_pad_token.shape == (1, 64)
    assert model.x_pad_token.dtype == mx.bfloat16
    # No text patch reached the small float linears.
    assert type(model.final_layer.linear) is nn.Linear
    assert type(model.t_embedder.mlp[2]) is nn.Linear


def test_fp16_casts_bf16_matrices_through_fp32(toy, tmp_path):
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    model, info = _load(p, act_dtype=mx.float16)
    assert model.final_layer.linear.weight.dtype == mx.float16
    assert model.norm.weight.dtype == mx.float16
    assert model.final_layer.linear.bias.dtype == mx.float16
    assert model.t_embedder.mlp[0].weight.dtype == mx.float32
    assert info.n_owned_casts == 4


def test_values_survive_the_casts(toy, tmp_path):
    t = _tensors()
    p = write_gguf(tmp_path / "toy.gguf", t)
    model, _ = _load(p, act_dtype=mx.float16)
    bits = t["final_layer.linear.weight"][0]
    want = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    got = np.array(model.final_layer.linear.weight)
    assert np.array_equal(got, want)


def test_a_name_mismatch_names_both_sides(monkeypatch, tmp_path):
    fam = _family(build=lambda cfg: _Toy(drop_pad=True))
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    with pytest.raises(fp.ImageModelError) as ei:
        _load(p)
    assert str(ei.value) == (
        "the Toy module tree and the file disagree. Parameters without a "
        "tensor (0): none. Tensors without a parameter (1): x_pad_token.")


def test_a_family_without_a_model_is_refused(monkeypatch, tmp_path):
    fam = _family(build=None)
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    with pytest.raises(fp.ImageModelError,
                       match="this is a Toy GGUF, which this gmlx cannot"):
        _load(p)


def test_reshaped_tensors_are_refused(toy, tmp_path):
    import gguf

    path = tmp_path / "reshaped.gguf"
    w = gguf.GGUFWriter(str(path), "none")
    w.kv_data[0].pop("general.architecture", None)
    w.add_array("comfy.gguf.orig_shape.proj.weight", [32, 2, 32])
    for name, (arr, codec) in _tensors().items():
        w.add_tensor(name, np.ascontiguousarray(arr),
                     raw_dtype=getattr(gguf.GGMLQuantizationType, codec))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(fp.ImageModelError, match="reshaped by the converter"):
        _load(path)


def test_an_unsupported_codec_is_refused_by_name(toy, tmp_path):
    import gguf
    from gmlx.load.preflight import UnsupportedCodecError

    t = _tensors()
    tsize = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.TQ1_0][1]
    t["extra.weight"] = (np.zeros((4, tsize), np.uint8), "TQ1_0")
    p = write_gguf(tmp_path / "tq.gguf", t)
    with pytest.raises(UnsupportedCodecError, match="TQ1_0x1"):
        _load(p)


def test_a_load_on_one_thread_runs_on_another(toy, tmp_path, monkeypatch):
    """MLX default streams are per thread. With the view check off nothing
    else evaluates the owned casts, so load_dit must return a materialized
    tree or the first forward on another thread fails."""
    import threading

    monkeypatch.setenv("GMLX_VERIFY_VIEWS", "0")
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    box: list = []
    t = threading.Thread(
        target=lambda: box.append(_load(p, act_dtype=mx.float16)))
    t.start()
    t.join()
    model, _ = box[0]
    mlp = model.t_embedder.mlp
    h = mlp[2](mlp[1](mlp[0](mx.ones((1, 4)))))
    y = model.final_layer.linear(model.norm.weight[None] * model.x_pad_token)
    mx.eval(h, y)
    assert h.dtype == mx.float32 and y.dtype == mx.float16


def test_act_dtype_must_be_half(toy, tmp_path):
    with pytest.raises(ValueError, match="bfloat16 or float16"):
        _load(tmp_path / "unused.gguf", act_dtype=mx.float32)
