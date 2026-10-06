"""Helpers for the Qwen-Image 2.1 VAE tests: the tiny model of the check 5
self-test as a safetensors file in the layout of the official one, tiny
models built from a config of their own, and the wrong models that the
parity gate must fail."""
from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FAMILY = "qwen_image21"


def _script(name: str):
    path = ROOT / "scripts" / "image_oracle" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_image_oracle_{name}",
                                                  path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tiny = _script("qwen_image21_vae_tiny")
ST_NAME = "diffusion_pytorch_model.safetensors"


def tiny_file(directory, tensors: dict | None = None,
              name: str = ST_NAME) -> str:
    """The tiny VAE, or ``tensors``, as an F32 safetensors file."""
    return tiny.write_safetensors(Path(directory) / name,
                                  tiny.weights() if tensors is None
                                  else tensors)


def tiny_config(**changes) -> dict:
    return {**tiny.TINY, **changes}


def pruned() -> dict:
    """The tiny file's tensors without all, or some, of its time_conv."""
    t = tiny.weights()
    return {
        "all_12": {k: v for k, v in t.items() if ".time_conv." not in k},
        "the_weights": {k: v for k, v in t.items()
                        if not k.endswith(".time_conv.weight")},
        "the_decoder_side": {k: v for k, v in t.items() if not (
            k.startswith("decoder.") and ".time_conv." in k)},
        "one_weight": {k: v for k, v in t.items() if k !=
                       "decoder.up_blocks.0.upsampler.time_conv.weight"},
    }


def weights(vae) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    return {k: np.array(v.astype(mx.float32))
            for k, v in tree_flatten(vae.parameters())}


def module_config(config: dict):
    """The module config of a diffusers config, its time flags included."""
    from gmlx.image.vae.qwen_image21_vae import QwenImage21VAEConfig

    return QwenImage21VAEConfig(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["z_dim"], base_dim=config["base_dim"],
        decoder_base_dim=config["decoder_base_dim"],
        dim_mult=tuple(config["dim_mult"]),
        num_res_blocks=config["num_res_blocks"],
        temporal=tuple(config["temperal_downsample"]))


def build(config: dict, tensors: dict | None = None, dtype=None):
    """The VAE of a diffusers config, built from the config itself and
    filled with ``tensors`` in torch layout (``tiny.weights(config)`` when
    None). The reader gives a file the official time flags and number of
    levels, and this keeps the config's own."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from gmlx.image.vae.qwen_image21_vae import QwenImage21VAE

    vae = QwenImage21VAE(module_config(config))
    src = tiny.weights(config) if tensors is None else tensors
    filled = []
    for name, _ in tree_flatten(vae.parameters()):
        a = src[name]
        a = mx.array(a.transpose(0, 2, 3, 1) if a.ndim == 4 else a)
        filled.append((name, a.astype(dtype or mx.float32)))
    vae.load_weights(filled, strict=True)
    return vae


def gguf_with_config(path, tensors: dict, text: str | None) -> str:
    """``tensors`` (torch layout, float32) as an F32 GGUF that carries
    ``text`` as its VAE config, as a packed VAE does."""
    import gguf

    w = gguf.GGUFWriter(str(path), "autoencoder")
    w.add_type("vae")
    if text is not None:
        w.add_string("gmlx.vae.config_json", text)
    for name, a in tensors.items():
        w.add_tensor(name, np.ascontiguousarray(a, np.float32),
                     raw_shape=a.shape,
                     raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


# Wrong models. Each takes pytest's monkeypatch and a loaded VAE and leaves
# one departure from the reference in place.


def _other_repeat_order(monkeypatch, vae):
    """The up shortcut repeats the channel list whole, not each channel in
    turn."""
    import mlx.core as mx

    from gmlx.image.vae import qwen_image21_vae as q

    def dup_up(x, cout, factor_t, factor_s):
        b, h, w, c = x.shape
        s = factor_s
        f = factor_t * s * s
        at = (np.arange(cout)[:, None] * f + (factor_t - 1) * s * s
              + np.arange(s * s)[None, :])
        src = mx.array((at % c).reshape(-1).astype(np.int32))
        x = mx.take(x, src, axis=-1).reshape(b, h, w, cout, s, s)
        return x.transpose(0, 1, 4, 2, 5, 3).reshape(b, h * s, w * s, cout)

    monkeypatch.setattr(q, "dup_up", dup_up)


def _strided_pick(monkeypatch, vae):
    """The down shortcut reads the first pixel of each block, not its
    mean."""
    import mlx.core as mx

    from gmlx.image.vae import qwen_image21_vae as q

    right = q.avg_down

    def avg_down(x, cout, factor_t, factor_s):
        s = factor_s
        if s > 1:
            x = mx.repeat(mx.repeat(x[:, ::s, ::s], s, axis=1), s, axis=2)
        return right(x, cout, factor_t, factor_s)

    monkeypatch.setattr(q, "avg_down", avg_down)


def _pad_top_left(monkeypatch, vae):
    from gmlx.image.vae import qwen_image21_vae as q

    monkeypatch.setattr(q, "DOWN_PAD", ((0, 0), (1, 0), (1, 0), (0, 0)))


def _bilinear(monkeypatch, vae):
    import mlx.core as mx

    from gmlx.image.vae import qwen_image21_vae as q

    def up(x, axis):
        n = x.shape[axis]
        idx = mx.arange(n)
        lo = mx.take(x, mx.maximum(idx - 1, 0), axis=axis)
        hi = mx.take(x, mx.minimum(idx + 1, n - 1), axis=axis)
        pair = mx.stack([0.75 * x + 0.25 * lo, 0.75 * x + 0.25 * hi],
                        axis=axis + 1)
        shape = list(x.shape)
        shape[axis] = 2 * n
        return pair.reshape(shape).astype(x.dtype)

    monkeypatch.setattr(q, "upsample_nearest", lambda x: up(up(x, 1), 2))


def _gamma_without_scale(monkeypatch, vae):
    from gmlx.image.vae import qwen_image21_vae as q

    right = q.rms_norm
    monkeypatch.setattr(
        q, "rms_norm",
        lambda x, gamma: right(x, gamma) * (1 / math.sqrt(x.shape[-1])))


def _mean_std_swapped(monkeypatch, vae):
    cfg = vae.config
    vae.config = replace(cfg, latents_mean=cfg.latents_std,
                         latents_std=cfg.latents_mean)


def _last_half(monkeypatch, vae):
    import mlx.core as mx

    c = vae.config.latent_channels
    monkeypatch.setattr(
        type(vae), "split_moments",
        lambda self, m: (m[..., c:], mx.clip(m[..., :c], -30.0, 20.0)))


def _alpha_dropped(monkeypatch, vae):
    import mlx.core as mx

    from gmlx.image.vae import qwen_image21_vae as q

    def finish(self, x):
        x = q.clamp(x)
        return mx.concatenate([x[..., :3], mx.ones_like(x[..., :1])],
                              axis=-1)

    monkeypatch.setattr(type(vae), "finish", finish)


def _no_clamp(monkeypatch, vae):
    import mlx.core as mx

    from gmlx.image.vae import qwen_image21_vae as q

    monkeypatch.setattr(q, "clamp", lambda x: x.astype(mx.float32))


# name -> (the departure, the gates it must fail)
WRONG = {
    "up_shortcut_repeats_in_the_other_order": (_other_repeat_order,
                                               ("decode",)),
    "down_shortcut_by_strided_pick": (_strided_pick, ("encode",)),
    "padding_on_the_other_side": (_pad_top_left, ("encode",)),
    "bilinear_for_nearest": (_bilinear, ("decode",)),
    "gamma_without_sqrt_c": (_gamma_without_scale, ("decode", "encode")),
    "mean_and_std_swapped": (_mean_std_swapped, ("decode", "latents")),
    "last_64_of_128_encode_channels": (_last_half, ("encode",)),
    "alpha_dropped_on_decode": (_alpha_dropped, ("decode",)),
    "clamp_left_out": (_no_clamp, ("decode",)),
}
