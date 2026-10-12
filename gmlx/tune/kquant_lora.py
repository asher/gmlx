# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""LoRA modules on a K-quant base.

The vendored mlx-lm ``linear_to_lora_layers`` calls ``lora_from_kquant``
for every layer it adapts, before it consults a ``to_lora`` method, so a
``KQuantLinear``, ``KQuantSwitchLinear`` or ``KQuantEmbedding`` base gets a
LoRA module of the vendored tuner whatever another package has set on those
classes. The base stays frozen K-quant, and mlx-kquant's gradient with
respect to the input of the quantized matmul carries the gradient to the
adapter. Nothing here changes an mlx-kquant class.
"""

from __future__ import annotations


def is_kquant(layer) -> bool:
    """True iff ``lora_from_kquant`` builds a LoRA module for ``layer``."""
    from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

    return isinstance(layer, (KQuantLinear, KQuantSwitchLinear, KQuantEmbedding))


def lora_from_kquant(layer, *, r: int, scale: float, dropout: float):
    """The vendored tuner's LoRA module around the K-quant ``layer``, or
    None when ``layer`` is not a K-quant class. The input width comes from
    the codec geometry, since the K-quant modules store only the packed row
    width."""
    from mlx_kquant.codec_geometry import in_features
    from mlx_kquant.nn import KQuantEmbedding, KQuantLinear, KQuantSwitchLinear

    from gmlx._vendor.mlx_lm.tuner.lora import (
        LoRAEmbedding,
        LoRALinear,
        LoRASwitchLinear,
    )

    if isinstance(layer, KQuantSwitchLinear):
        num_experts, out_dims, row_bytes = layer.weight.shape
        lora = LoRASwitchLinear(
            input_dims=in_features(layer.kquant_type, row_bytes),
            output_dims=out_dims, num_experts=num_experts, r=r,
            dropout=dropout, scale=scale)
        return lora.update_modules({"linear": layer})
    if isinstance(layer, KQuantLinear):
        out_dims, row_bytes = layer.weight.shape
        lora = LoRALinear(
            input_dims=in_features(layer.kquant_type, row_bytes),
            output_dims=out_dims, r=r, dropout=dropout, scale=scale)
        return lora.update_modules({"linear": layer})
    if isinstance(layer, KQuantEmbedding):
        lora = LoRAEmbedding(
            num_embeddings=layer.num_embeddings, dims=layer.dims, r=r,
            dropout=dropout, scale=scale)
        return lora.update_modules({"embedding": layer})
    return None
