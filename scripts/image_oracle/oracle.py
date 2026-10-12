"""Oracle-side helpers, imported only by oracle scripts in the oracle
environment: the offline guard, the torch setup, and a DiT loaded from its
GGUF through diffusers' ``from_single_file`` with the GGUF quantization
config.

The oracle computes on the CPU. ``fp32`` runs activations and dequantized
weights in float32. ``bf16`` runs them in bfloat16, and the gap between
the two modes calibrates each tolerance.

Stock diffusers dequantizes GGUF blocks with their scales left in float16,
which rounds each K-quant weight away from ggml's float32 dequantization
of the same bytes. ``dequant="exact"`` passes float32 to diffusers'
per-codec dequant functions, whose arithmetic then equals ggml's. The
mode in force is recorded by ``dequant_mode``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODES = ("fp32", "bf16")
DEQUANT = ("exact", "stock")


def require_offline() -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("error: oracle runs need HF_HUB_OFFLINE=1. Start "
                         "them with scripts/image_oracle/run.py")


def setup(threads: int | None = None):
    """Import torch for an oracle run on the CPU and return it."""
    require_offline()
    import torch

    n = threads or int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if n:
        torch.set_num_threads(n)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    return torch


def torch_dtype(mode: str):
    import torch

    return {"fp32": torch.float32, "bf16": torch.bfloat16}[mode]


def set_dequant(kind: str) -> None:
    """Select float32 (``exact``) or stock float16 block arithmetic for
    every GGUF dequantization in this process."""
    import functools

    import torch
    from diffusers.quantizers.gguf import utils as gu

    if kind not in DEQUANT:
        raise ValueError(f"dequant must be one of {DEQUANT}, not {kind!r}")
    stock = getattr(gu, "_oracle_stock_dequant", None)
    if stock is None:
        stock = gu._oracle_stock_dequant = dict(gu.dequantize_functions)
    for qt, fn in stock.items():
        gu.dequantize_functions[qt] = (
            functools.partial(fn, dtype=torch.float32) if kind == "exact"
            else fn)
    gu._oracle_dequant_mode = kind


def dequant_mode() -> str:
    from diffusers.quantizers.gguf import utils as gu

    return getattr(gu, "_oracle_dequant_mode", "stock")


def load_dit(gguf: str | os.PathLike, config_dir: str | os.PathLike,
             mode: str, cls_name: str = "ZImageTransformer2DModel",
             dequant: str = "exact"):
    """The DiT from a GGUF file with GGUF linears that dequantize to the
    mode's dtype on each call. ``config_dir`` is a local copy of the
    official repository, read for ``transformer/config.json``."""
    import diffusers

    set_dequant(dequant)
    dtype = torch_dtype(mode)
    cls = getattr(diffusers, cls_name)
    qc = diffusers.GGUFQuantizationConfig(compute_dtype=dtype)
    model = cls.from_single_file(
        str(Path(gguf).expanduser()), quantization_config=qc,
        torch_dtype=dtype, config=str(Path(config_dir).expanduser()),
        subfolder="transformer", local_files_only=True)
    return model.eval()


_NATIVE_LINEAR: dict = {}


def emulate_bf16_matmul(enable: bool = True) -> None:
    """Run each bfloat16 linear product as a float32 GEMM over the
    bfloat16 operands, with one rounding of the result to bfloat16.

    Products of bfloat16 values are exact in float32, so the result
    differs from a native bfloat16 GEMM with float32 accumulation only in
    summation order. torch's CPU bfloat16 GEMM runs on one thread at about
    1.5 GFLOPS, which puts a bfloat16 DiT forward at hours. SDPA and the
    elementwise ops stay native bfloat16.
    """
    import torch
    import torch.nn.functional as F

    native = _NATIVE_LINEAR.setdefault("linear", F.linear)
    if not enable:
        F.linear = native
        return

    def linear(x, w, b=None):
        if x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16:
            y = native(x.float(), w.float(),
                       None if b is None else b.float())
            return y.to(torch.bfloat16)
        return native(x, w, b)

    linear._oracle_emulated = True
    F.linear = linear


def matmul_mode() -> str:
    import torch.nn.functional as F

    return ("bf16 operands, float32 GEMM, one bf16 rounding"
            if getattr(F.linear, "_oracle_emulated", False) else "native")


def gguf_census(model) -> dict:
    """GGUF linear leaves by codec and plain parameters by dtype."""
    from collections import Counter

    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

    linears = Counter()
    plain = Counter()
    for _, mod in model.named_modules():
        if isinstance(mod, GGUFLinear):
            linears[mod.weight.quant_type.name] += 1
    for _, p in model.named_parameters():
        if not isinstance(p, GGUFParameter):
            plain[str(p.dtype).removeprefix("torch.")] += 1
    return {"gguf_linears": dict(sorted(linears.items())),
            "n_gguf_linears": sum(linears.values()),
            "plain_params": dict(sorted(plain.items()))}


def to_numpy(t, dtype=np.float32) -> np.ndarray:
    """A tensor as a numpy array; floating tensors pass through float32."""
    import torch

    t = t.detach().cpu()
    if t.is_floating_point():
        t = t.to(torch.float32)
    return t.numpy().astype(dtype, copy=False)
