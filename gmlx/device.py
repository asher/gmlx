"""The Metal device's info, whatever the default device is.

``mx.device_info()`` answers for the default device. With the CPU as the
default device on a Mac (``gmlx distill ... --cpu``, or a program that sets
it before a load), the answer names the CPU and has none of the Metal keys:
``max_buffer_length``, ``max_recommended_working_set_size``,
``memory_size`` and ``resource_limit``. Every gmlx read of those keys, and
of the GPU architecture, goes through ``metal_info`` or ``metal_value``.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def cpu_default_device() -> bool:
    """Whether the CPU is the MLX default device."""
    return mx.default_device() == mx.cpu


def metal_info() -> dict[str, Any] | None:
    """The Metal device's info, or None on a machine without Metal."""
    info = mx.device_info
    if (getattr(info, "__module__", None) or "").startswith("mlx.core"):
        return dict(info(mx.gpu)) if mx.metal.is_available() else None
    # A replacement answers for the Metal device, as the one
    # configure_stream_cpu installs does. One that takes a device is asked
    # for the GPU.
    try:
        return dict(info(mx.gpu))
    except TypeError:
        return dict(info())


def metal_value(key: str) -> Any:
    """``key`` of the Metal device's info. Raises KeyError without Metal or
    without the key, as ``mx.device_info()[key]`` does."""
    info = metal_info()
    if info is None:
        raise KeyError(key)
    return info[key]


def stream_budget_bytes() -> int | None:
    """The weight bytes a model may hold resident: 0.9 of the Metal
    working set. A model over it streams its experts, and discovery marks
    a MoE over it ``stream: experts``. None without Metal."""
    try:
        return int(0.9 * metal_value("max_recommended_working_set_size"))
    except (KeyError, TypeError, ValueError):
        return None
