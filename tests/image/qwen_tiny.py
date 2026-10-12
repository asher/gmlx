"""A tiny Qwen-Image 2.1 DiT for the CPU tests. The config, the tensor
names and shapes, the weights and the inputs come from
scripts/image_oracle/qwen_image21_tiny.py, which the oracle self-test
reads too. This module adds the gmlx side: a registry row that accepts
tiny shapes, models, GGUF files and conditionings."""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

_B0 = "transformer_blocks.0."


def _rule():
    path = (Path(__file__).resolve().parents[2] / "scripts" / "image_oracle"
            / "qwen_image21_tiny.py")
    spec = importlib.util.spec_from_file_location("_qwen_image21_tiny", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rule = _rule()
TINY, CASES, TARGET, PREFIX = rule.TINY, rule.CASES, rule.TARGET, rule.PREFIX
shapes, weights, unfuse = rule.shapes, rule.weights, rule.unfuse


def view_of(w: dict):
    """A header view of float32 tensors ``w``."""
    from gmlx.image.fingerprint import HeaderView, Tensor

    return HeaderView(kv={}, tensors={
        k: Tensor(tuple(reversed(np.shape(v))), "F32") for k, v in w.items()})


def tiny_family(axes=TINY["axes_dims_rope"]):
    """The Qwen-Image 2.1 registry row with a fingerprint that takes any
    width and with the rotary axes of the tiny config."""
    from gmlx.image.fingerprint import Match
    from gmlx.image.registry import get_family

    real = get_family("qwen_image21")

    def layout(view) -> str:
        fused = view.has(_B0 + "img_mlp.gate_up.weight")
        return "fused" if fused else "unfused"

    def fingerprint(view):
        if not view.has("txt_in.text_norm.weight"):
            return None
        return Match(layout=layout(view))

    def synth_config(view):
        return {**real.synth_config(view), "axes_dims_rope": list(axes),
                "mlp_layout": layout(view)}

    return replace(real, fingerprint=fingerprint, synth_config=synth_config)


def config_of(w: dict, axes=TINY["axes_dims_rope"]) -> dict:
    """The config the family synthesizes for weights ``w``."""
    return tiny_family(axes).synth_config(view_of(w))


def model_of(w: dict, axes=TINY["axes_dims_rope"]):
    """The DiT of float32 weights ``w`` with fp32 activations."""
    import mlx.core as mx

    from gmlx.image.families.qwen_image21.model import QwenImage21DiT

    m = QwenImage21DiT(config_of(w, axes))
    m.load_weights([(k, mx.array(np.asarray(v, np.float32)))
                    for k, v in w.items()])
    m.act_dtype = mx.float32
    m.eval()
    return m


def write(path, w: dict, prefix: str = PREFIX, codecs: dict | None = None,
          arch: str | None = "qwen_image") -> str:
    """Write weights ``w`` as a GGUF. ``codecs`` maps a name to BF16 or
    Q8_0, and the rest are F32."""
    import gguf

    from image_fixtures import bf16_bits, write_gguf

    tensors = {}
    for name, a in w.items():
        codec = (codecs or {}).get(name, "F32")
        if codec == "BF16":
            a = bf16_bits(a)
        elif codec != "F32":
            a = gguf.quants.quantize(
                np.asarray(a, np.float32),
                getattr(gguf.GGMLQuantizationType, codec))
        tensors[prefix + name] = (a, codec)
    return write_gguf(path, tensors, arch=arch)


def conditioning(case: str, seed: int = 0, key: str | None = None):
    """The conditioning of a case: its features and reference latents."""
    import mlx.core as mx

    from gmlx.image.conditioning import Conditioning

    return Conditioning(
        mx.array(rule.features(case, seed)), case if key is None else key,
        ref_latents=tuple(mx.array(z) for z in rule.ref_latents(case, seed)),
        image_slots=tuple(CASES[case][1]))


def latents(seed: int = 0, batch: int = 1, size=TARGET):
    import mlx.core as mx

    return mx.array(rule.latents(seed, batch, size))
