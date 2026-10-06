"""load_dit on small synthetic Qwen-Image 2.1 GGUFs in both MLP layouts:
every parameter filled from its tensor, every tensor consumed, a missing
and an extra tensor refused by name, the casts of the family's fp32 keep
list, float16 refused, and the relatives refused through the real
fingerprint. The real file's names and shapes are held to the module
tree through the header inventory, without a model of that size."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import qwen_tiny as qt
from image_fixtures import Q_HEADER, load_json, official_config

from gmlx.image import fingerprint as fp
from gmlx.image import loader as image_loader
from gmlx.image import registry
from gmlx.image.dtypes import ImageDtypeError
from gmlx.image.families.qwen_image21.model import (QwenImage21Config,
                                                    QwenImage21DiT)


@pytest.fixture
def tiny(monkeypatch):
    fam = qt.tiny_family()
    monkeypatch.setattr(registry, "families", lambda: (fam,))
    return fam


def _load(path, **kw):
    return image_loader.load_dit(str(path), log=lambda *_: None, **kw)


def _params(model) -> dict:
    return dict(tree_flatten(model.parameters()))


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_every_parameter_is_filled_and_every_tensor_consumed(tiny, tmp_path,
                                                            fused):
    w = qt.weights(0, fused=fused)
    model, info = _load(qt.write(tmp_path / "q.gguf", w))
    assert type(model) is QwenImage21DiT
    assert info.identified.family is tiny
    assert info.identified.layout == ("fused" if fused else "unfused")
    assert info.identified.prefix == qt.PREFIX
    assert model.cfg.fused_mlp is fused and model.act_dtype == mx.bfloat16
    params = _params(model)
    assert sorted(params) == sorted(w)
    assert len(w) == 9 + (8 if fused else 9) * qt.TINY["num_layers"]
    for name, p in params.items():
        # The weights are values bfloat16 holds, so either dtype is exact.
        assert np.array_equal(np.array(p.astype(mx.float32)), w[name]), name
        keep = "norm" in name
        assert p.dtype == (mx.float32 if keep else mx.bfloat16), name
    mlp = model.transformer_blocks[1].img_mlp
    assert hasattr(mlp, "gate_up") is fused
    assert hasattr(mlp, "gate_layer") is (not fused)
    assert info.n_kquant == 0 and info.n_owned_casts == 0


def test_the_codecs_of_the_real_file_load_into_the_same_tree(tiny, tmp_path):
    from mlx_kquant.nn import KQuantLinear

    w = qt.weights(2)
    b = "transformer_blocks.0."
    codecs = {"txt_in.text_norm.weight": "BF16",
              "img_in.weight": "BF16", "txt_in.in_layer.weight": "BF16",
              b + "attn.to_q.weight": "Q8_0", "proj_out.weight": "Q8_0",
              "modulation.1.weight": "Q8_0"}
    model, info = _load(qt.write(tmp_path / "q.gguf", w, codecs=codecs))
    # The BF16 text norm weight becomes an owned fp32 array, since the
    # reference adds one to it in fp32. The F32 final scale matrix stays.
    assert model.txt_in.text_norm.weight.dtype == mx.float32
    assert np.array_equal(np.array(model.txt_in.text_norm.weight),
                          w["txt_in.text_norm.weight"])
    assert model.norm_out.linear.weight.dtype == mx.float32
    assert model.img_in.weight.dtype == mx.bfloat16
    assert info.n_owned_casts == 1 and info.n_kquant == 3
    assert type(model.transformer_blocks[0].attn.to_q) is KQuantLinear
    assert type(model.modulation[1]) is KQuantLinear
    assert type(model.transformer_blocks[0].attn.to_k) is nn.Linear


def test_a_quantized_norm_matrix_loads_as_owned_fp32(tiny, tmp_path):
    import gguf

    w = qt.weights(2)
    name = "norm_out.linear.weight"
    model, info = _load(qt.write(tmp_path / "q.gguf", w,
                                 codecs={name: "Q8_0"}))
    lin = model.norm_out.linear
    assert type(lin) is nn.Linear and lin.weight.dtype == mx.float32
    q = gguf.quants.quantize(w[name], gguf.GGMLQuantizationType.Q8_0)
    want = gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0)
    assert np.array_equal(np.array(lin.weight), want)
    assert info.n_kquant == 0 and info.n_owned_casts == 1


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_a_missing_and_an_extra_tensor_are_refused_by_name(tiny, tmp_path,
                                                          fused):
    w = qt.weights(0, fused=fused)
    gone = "transformer_blocks.1.attn.to_v.weight"
    short = {k: v for k, v in w.items() if k != gone}
    with pytest.raises(fp.ImageModelError) as ei:
        _load(qt.write(tmp_path / "short.gguf", short))
    assert str(ei.value) == (
        "the Qwen-Image 2.1 module tree and the file disagree. Parameters "
        f"without a tensor (1): {gone}. Tensors without a parameter (0): "
        "none.")
    extra = "transformer_blocks.0.attn.to_q.bias"
    more = {**w, extra: np.zeros(32, np.float32)}
    with pytest.raises(fp.ImageModelError) as ei:
        _load(qt.write(tmp_path / "more.gguf", more))
    assert str(ei.value) == (
        "the Qwen-Image 2.1 module tree and the file disagree. Parameters "
        f"without a tensor (0): none. Tensors without a parameter (1): "
        f"{extra}.")


def test_a_file_with_both_mlp_layouts_is_refused_by_name(tiny, tmp_path):
    w = qt.weights(0)
    both = {**w, **{k: v for k, v in qt.unfuse(w).items() if k not in w}}
    with pytest.raises(fp.ImageModelError,
                       match=r"Tensors without a parameter \(4\): "
                             r"transformer_blocks.0.img_mlp.gate_layer"):
        _load(qt.write(tmp_path / "both.gguf", both))


def test_float16_is_refused(tiny, tmp_path):
    p = qt.write(tmp_path / "q.gguf", qt.weights(0))
    with pytest.raises(ImageDtypeError, match="never runs in float16"):
        _load(p, act_dtype=mx.float16)


def test_the_real_fingerprint_refuses_tiny_shapes_and_the_relatives(
        tmp_path):
    w = qt.weights(0)
    z = np.zeros((4, 4), np.float32)
    b = "transformer_blocks.0."

    def refused(name, tensors) -> str:
        with pytest.raises(fp.ImageModelError) as ei:
            _load(qt.write(tmp_path / f"{name}.gguf", tensors))
        return str(ei.value)

    msg = refused("tiny", w)
    assert "a Qwen-Image relative that gmlx image does not support" in msg
    assert "img_in.weight is (8, 32), Qwen-Image 2.1 has (64, 4096)" in msg
    assert "it has 2 blocks, Qwen-Image 2.1 has 32" in msg
    assert "a Lens GGUF" in refused("lens", {
        **w, b + "attn.norm_added_q.weight": z, b + "img_mlp.w1.weight": z})
    assert "a Mage-Flow GGUF" in refused("mage", {
        "txt_norm.weight": np.zeros(2560, np.float32),
        "proj_out.weight": np.zeros((128, 4), np.float32)})
    q20 = {**w, "txt_norm.weight": np.zeros(4, np.float32),
           b + "attn.add_q_proj.weight": z}
    assert "a Qwen-Image 20B GGUF" in refused("q20", q20)
    assert "Qwen-Image-Edit 2511" in refused("edit", {
        **q20, "__index_timestep_zero__": np.zeros(1, np.float32)})
    assert "Qwen-Image-Layered" in refused("layered", {
        **q20, "time_text_embed.addition_t_embedding.weight": z})


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_the_real_names_and_shapes_follow_from_the_module_tree(fused):
    """qt.shapes is held to the module tree on the tiny config, and to the
    header inventory of the real file on the official config. A tree of
    the real depth at the tiny width has the real file's names."""
    w = qt.weights(0, fused=fused)
    built = {k: tuple(v.shape) for k, v in _params(qt.model_of(w)).items()}
    assert built == qt.shapes(qt.TINY, fused)

    official = official_config("qwen_image21_transformer_config.json")
    header = {n.removeprefix(qt.PREFIX): tuple(reversed(shape))
              for n, shape, _ in load_json(Q_HEADER)["tensors"]}
    if not fused:
        header = _unfused_shapes(header)
    assert header == qt.shapes(official, fused)

    deep = {**qt.TINY, "num_layers": official["num_layers"]}
    names = set(_params(QwenImage21DiT(qt.config_of(
        {k: np.zeros(s, np.float32)
         for k, s in qt.shapes(deep, fused).items()}))))
    assert names == set(header) and len(names) == (265 if fused else 297)


def _unfused_shapes(header: dict) -> dict:
    out = {}
    for name, shape in header.items():
        if name.endswith("img_mlp.gate_up.weight"):
            stem = name[:-len("gate_up.weight")]
            half = (shape[0] // 2, shape[1])
            out[stem + "gate_layer.weight"] = half
            out[stem + "proj.weight"] = half
        else:
            out[name] = shape
    return out


def test_the_real_config_builds_without_a_model():
    from image_fixtures import fixture_view

    got = fp.identify(fixture_view(Q_HEADER))
    fam = got.family
    cfg = QwenImage21Config.from_dict(got.config)
    assert (cfg.dim, cfg.n_heads, cfg.head_dim, cfg.n_layers) == (
        4096, 32, 128, 32)
    assert (cfg.in_channels, cfg.out_channels, cfg.patch) == (64, 64, 1)
    assert (cfg.context_dim, cfg.ffn_dim, cfg.t_freq_dim) == (
        4096, 12288, 256)
    assert cfg.axes_dims == (16, 56, 56) and cfg.rope_theta == 10000.0
    assert cfg.fused_mlp and cfg.eps == 1e-6
    assert fam.fp32_keep == ("norm",) and fam.build is not None
    kept = sorted({n.split(".")[-2] for n, _, _ in
                   load_json(Q_HEADER)["tensors"] if "norm" in n})
    assert kept == ["linear", "norm_k", "norm_q", "text_norm"]
