# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Int8 linears: a DiT's quantized linears converted to int8 at load.

Each weight row, one output channel, has its own float32 scale over all
its columns. Each call quantizes its activation rows to int8, one scale a
row or a row's group of ``act_group`` columns, with mlx-kquant's
``int8_quantize_rows``, and multiplies the two in ``int8_matmul``: exact
int32 sums, scaled to the activation dtype. The form is lossy against the
file's own product.

An ``Int8Linear`` is not a K-quant leaf, so no K-quant route of
``kq_routes`` takes it, and a LoRA on it runs on the stock path.
"""

from __future__ import annotations

import fnmatch

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq


def _kq(name: str):
    """mlx-kquant's op ``name``. The video loader refuses int8 linears on
    a build without the ops before any weight loads."""
    return getattr(kq, name)


class Int8Linear(nn.Module):
    """``x @ W^T`` with W int8 (out, in) and one float32 scale a row, and
    x quantized to int8 a call. ``act_group`` is the columns of one
    activation scale, or None for the whole row."""

    def __init__(self, weight: mx.array, scales: mx.array,
                 act_group: int | None = None):
        super().__init__()
        if weight.dtype != mx.int8 or weight.ndim != 2:
            raise ValueError(f"an int8 linear takes an int8 (out, in) "
                             f"weight, not {weight.dtype} {weight.shape}")
        if scales.dtype != mx.float32 or scales.shape != weight.shape[:1]:
            raise ValueError(f"an int8 linear takes one float32 scale an "
                             f"output row, not {scales.dtype} "
                             f"{scales.shape}")
        self.weight = weight
        self.scales = scales
        self._act_group = act_group
        self.freeze()

    @property
    def act_group(self) -> int | None:
        return self._act_group

    def __call__(self, x: mx.array) -> mx.array:
        g = self._act_group
        x8, sx = _kq("int8_quantize_rows")(x, group_size=g)
        return _kq("int8_matmul")(x8, sx, self.weight, self.scales,
                                  group_size=g, out_dtype=x.dtype)


def int8_paths(model: nn.Module, patterns: dict) -> dict:
    """Module path to activation group, for each leaf of ``model`` whose
    path matches a pattern of ``patterns`` (fnmatch, as a family's
    ``int8_linears``)."""
    out = {}
    for path, _m in model.named_modules():
        for pat, group in patterns.items():
            if fnmatch.fnmatchcase(path, pat):
                out[path] = group
                break
    return out


def _in_features(leaf) -> int:
    codec = getattr(leaf, "kquant_type", None)
    w = leaf["weight"]
    if codec is None:
        return int(w.shape[1])
    from mlx_kquant.codec_geometry import in_features

    return in_features(codec, int(w.shape[1]))


def _float_weight(path: str, leaf) -> mx.array:
    """The float32 (out, in) weight of a K-quant or float linear leaf."""
    from mlx_kquant.nn import KQuantLinear

    if isinstance(leaf, KQuantLinear):
        return kq.dequantize(leaf["weight"], leaf["scales"],
                             leaf.kquant_type, mx.float32)
    return leaf.weight.astype(mx.float32)


def _set(root: nn.Module, path: str, new: nn.Module) -> None:
    *parts, last = path.split(".")
    obj = root
    for part in parts:
        obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
    if isinstance(obj, list):
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def convert(model: nn.Module, patterns: dict) -> int:
    """Replace each leaf that ``patterns`` names with an ``Int8Linear``:
    its weight read at float32, then ``int8_quantize_rows`` over each
    output row, one evaluation a matrix, the leaf dropped as it goes.
    Returns the count. Raises ValueError for a leaf that is no K-quant or
    float linear, has a bias or takes an input width off a multiple of
    32, before any leaf changes."""
    from mlx_kquant.nn import KQuantLinear

    paths = int8_paths(model, patterns)
    leaves = dict(model.named_modules())
    for path in paths:
        leaf = leaves[path]
        if not isinstance(leaf, KQuantLinear) and type(leaf) is not nn.Linear:
            raise ValueError(f"{path} is a {type(leaf).__name__}, which int8 "
                             f"linears do not convert")
        if "bias" in leaf:
            raise ValueError(f"{path} has a bias, which int8 linears do not "
                             f"take")
        k = _in_features(leaf)
        if k % 32:
            raise ValueError(f"{path} takes {k} columns, and int8 linears "
                             f"take a multiple of 32")
    quantize = _kq("int8_quantize_rows")
    for path, group in paths.items():
        w = _float_weight(path, leaves.pop(path))
        codes, scales = quantize(w)
        scales = scales.reshape(-1)
        mx.eval(codes, scales)
        del w
        _set(model, path, Int8Linear(codes, scales, group))
    del leaves
    mx.clear_cache()
    return len(paths)
