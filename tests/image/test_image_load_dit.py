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


@pytest.mark.parametrize("dims", [[32, 2, 32], [1] * 3000],
                         ids=["read", "over the scan's array limit"])
def test_reshaped_tensors_are_refused(toy, tmp_path, dims):
    import gguf

    path = tmp_path / "reshaped.gguf"
    w = gguf.GGUFWriter(str(path), "none")
    w.kv_data[0].pop("general.architecture", None)
    w.add_array("comfy.gguf.orig_shape.proj.weight", dims)
    for name, (arr, codec) in _tensors().items():
        w.add_tensor(name, np.ascontiguousarray(arr),
                     raw_dtype=getattr(gguf.GGMLQuantizationType, codec))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(fp.ImageModelError, match="reshaped by the converter"):
        _load(path)


def _forbid_tensor_reads(monkeypatch):
    """Fail any read of tensor bytes: the wire load and the identity reads.
    The header scan maps the file and reads only the header."""
    import os

    import gmlx.load.wire as wire

    def boom(*a, **k):
        raise AssertionError("tensor bytes were read")

    monkeypatch.setattr(wire, "load_gguf_wire_bytes", boom)
    monkeypatch.setattr(os, "pread", boom)


def _identity_family(monkeypatch) -> Family:
    """The toy family with an identity tensor, so identify would read."""
    fam = _family()
    fam = Family(**{**fam.__dict__, "identity_tensors": ("proj.bias",)})
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    return fam


def test_a_truncated_file_is_refused_before_any_bytes_are_read(
        tmp_path, monkeypatch):
    import os

    _identity_family(monkeypatch)
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    with open(p, "r+b") as f:
        f.truncate(os.path.getsize(p) - 16)
    _forbid_tensor_reads(monkeypatch)
    with pytest.raises(fp.ImageModelError, match="truncated GGUF"):
        _load(p)


def test_the_codec_refusal_comes_before_any_bytes_are_read(
        tmp_path, monkeypatch):
    import gguf
    from gmlx.load.preflight import UnsupportedCodecError

    _identity_family(monkeypatch)
    t = _tensors()
    tsize = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.TQ1_0][1]
    t["extra.weight"] = (np.zeros((4, tsize), np.uint8), "TQ1_0")
    p = write_gguf(tmp_path / "tq.gguf", t)
    _forbid_tensor_reads(monkeypatch)
    with pytest.raises(UnsupportedCodecError, match="TQ1_0x1"):
        _load(p)


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


def test_a_file_cut_inside_its_header_is_refused_as_truncated(toy, tmp_path):
    p = write_gguf(tmp_path / "toy.gguf", _tensors())
    with open(p, "r+b") as f:
        f.truncate(40)
    with pytest.raises(fp.ImageModelError,
                       match=r"truncated GGUF \(the file ends inside its "
                             r"header, at 40 bytes\)"):
        _load(p)


class _WideT(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = [nn.Linear(64, 8), nn.SiLU(), nn.Linear(8, 4)]


def test_a_quantized_keep_tensor_loads_as_owned_fp32(monkeypatch, tmp_path):
    import gguf
    from mlx_kquant.nn import KQuantLinear

    def build(cfg):
        m = _Toy()
        m.t_embedder = _WideT()
        return m

    fam = _family(build=build)
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    rng = np.random.default_rng(1)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    q = gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q8_0)
    t = _tensors()
    t["t_embedder.mlp.0.weight"] = (q, "Q8_0")
    t["t_embedder.mlp.2.weight"] = (bf16_bits(
        rng.standard_normal((4, 8)).astype(np.float32)), "BF16")
    p = write_gguf(tmp_path / "toy.gguf", t)
    model, info = _load(p)
    lin = model.t_embedder.mlp[0]
    assert type(lin) is nn.Linear and info.n_kquant == 1
    assert type(model.proj) is KQuantLinear
    assert lin.weight.dtype == mx.float32 and lin.weight.shape == (8, 64)
    want = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0)
    assert np.array_equal(np.array(lin.weight), want)
    assert info.n_owned_casts == 2


def _q2_0(rows: int, cols: int, seed: int):
    """Wire bytes of a ``rows`` x ``cols`` Q2_0 matrix, and the values kq
    reads from them."""
    kq = pytest.importorskip("mlx_kquant")
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    if "q2_0" not in CODEC_GEOMETRY:
        pytest.skip("this mlx-kquant has no q2_0 codec")
    rng = np.random.default_rng(seed)
    w = mx.array((rng.standard_normal((rows, cols)) * 0.05).astype(np.float32))
    wire, scales = kq.quantize(w, "q2_0")
    values = kq.dequantize(wire, scales, "q2_0", mx.float32).reshape(rows, cols)
    return np.array(wire).astype(np.uint8).reshape(rows, -1), values


def test_a_q2_0_matrix_loads_as_a_kquant_leaf_outside_the_kernel_routes(
        toy, tmp_path):
    """Q2_0 is a codec of the text loader. A DiT file that holds it loads
    its matrix as wire bytes on the plain K-quant matmul, and no kernel
    route of gmlx image takes the codec."""
    from mlx_kquant.nn import KQuantLinear

    from gmlx.image.ops import kq_routes
    from gmlx.load.headerscan import scan_gguf

    wire, values = _q2_0(32, 64, 3)
    t = _tensors()
    t["proj.weight"] = (wire, "Q2_0")
    p = write_gguf(tmp_path / "q2.gguf", t)
    named = {m.name: m.type_name for m in scan_gguf(p).tensors}
    assert named["proj.weight"] == "Q2_0"
    model, info = _load(p)
    assert type(model.proj) is KQuantLinear and info.n_kquant == 1
    assert model.proj.kquant_type == "q2_0"
    x = mx.array(np.random.default_rng(4).standard_normal((3, 64))
                 .astype(np.float32)).astype(mx.bfloat16)
    want = x.astype(mx.float32) @ values.T + model.proj.bias.astype(mx.float32)
    got = model.proj(x)
    assert got.dtype == mx.bfloat16
    err = mx.abs(got.astype(mx.float32) - want).max() / mx.abs(want).max()
    assert err.item() < 2e-2
    codecs = {c for c, _ in kq_routes._GLU_K_TILE} | {
        c for c, _ in kq_routes._LORA_ROWS}
    assert "q2_0" not in codecs
    assert not kq_routes.lora_static(model.proj, mx.bfloat16, 16, None)


def test_a_q2_0_keep_tensor_loads_as_owned_fp32(monkeypatch, tmp_path):
    def build(cfg):
        m = _Toy()
        m.t_embedder = _WideT()
        return m

    fam = _family(build=build)
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    wire, values = _q2_0(8, 64, 5)
    rng = np.random.default_rng(6)
    t = _tensors()
    t["t_embedder.mlp.0.weight"] = (wire, "Q2_0")
    t["t_embedder.mlp.2.weight"] = (bf16_bits(
        rng.standard_normal((4, 8)).astype(np.float32)), "BF16")
    model, info = _load(write_gguf(tmp_path / "q2.gguf", t))
    lin = model.t_embedder.mlp[0]
    assert type(lin) is nn.Linear and info.n_kquant == 1
    assert lin.weight.dtype == mx.float32 and lin.weight.shape == (8, 64)
    assert mx.array_equal(lin.weight, values).item()


def test_the_memory_gate_runs_on_header_sizes_before_any_bytes_are_read(
        toy, tmp_path, monkeypatch):
    from gmlx.image.memfit import ImageMemoryError
    from gmlx.serve import capacity

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    t = _tensors()
    p = write_gguf(tmp_path / "toy.gguf", t)
    size = sum(a.nbytes for a, _ in t.values())
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: size - 1)
    _forbid_tensor_reads(monkeypatch)
    with pytest.raises(ImageMemoryError,
                       match="The Toy DiT weights of .* exceed the working "
                             "budget .* Try a smaller DiT quant"):
        _load(p)
    seen = []

    def defer(weight_bytes, model_id):
        seen.append((weight_bytes, model_id))
        raise capacity.LoadDeferred("busy")

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: size)
    monkeypatch.setattr(capacity, "preload_gate", defer)
    with pytest.raises(capacity.LoadDeferred):
        _load(p)
    assert seen == [(float(size), str(p))]


def test_the_memory_gate_counts_the_tensors_the_load_casts(toy, tmp_path,
                                                          monkeypatch):
    """A 16-bit tensor of the other float type is held beside its copy in
    the activation dtype until the load ends, so the gate counts it
    twice."""
    from gmlx.serve import capacity

    monkeypatch.delenv("GMLX_OVERCOMMIT", raising=False)
    t = _tensors()
    name = "final_layer.linear.weight"
    w16 = np.random.default_rng(1).standard_normal((16, 64)).astype(
        np.float16)
    t[name] = (w16, "F16")
    p = write_gguf(tmp_path / "toy.gguf", t)
    size = sum(a.nbytes for a, _ in t.values())
    seen = []

    def defer(weight_bytes, model_id):
        seen.append(weight_bytes)
        raise capacity.LoadDeferred("busy")

    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1 << 30)
    monkeypatch.setattr(capacity, "preload_gate", defer)
    _forbid_tensor_reads(monkeypatch)
    with pytest.raises(capacity.LoadDeferred):
        _load(p)
    assert seen == [float(size + w16.nbytes)]


def test_overcommit_skips_the_memory_gate(toy, tmp_path, monkeypatch):
    from gmlx.serve import capacity

    def boom(*a, **k):
        raise AssertionError("the gate ran")

    monkeypatch.setenv("GMLX_OVERCOMMIT", "1")
    monkeypatch.setattr(capacity, "working_budget_bytes", lambda: 1)
    monkeypatch.setattr(capacity, "preload_gate", boom)
    _, info = _load(write_gguf(tmp_path / "toy.gguf", _tensors()))
    assert info.n_kquant == 1
