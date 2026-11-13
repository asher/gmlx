"""The footprint of a whole FLUX.1 VAE decode at 512x512 in bf16 stays
under the memfit peak, which is fitted to measured whole decodes, and the
MLX cache ends at or under its bound with the previous limit restored.
The weights are random at the official shapes, so no file is needed.
The footprint is the kernel's peak since a reset, so a short peak is
never missed, and the baseline is the peak over a quiet moment after the
cache is cleared. Metal only."""
from __future__ import annotations

import ctypes
import gc
import os
import time

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_map

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")) or not mx.metal.is_available(),
    reason="measures a Metal decode")

_LIBC = ctypes.CDLL(None)


def _peak_footprint() -> int:
    """This process's peak physical footprint since _reset_peak:
    rusage_info_v4 ri_interval_max_phys_footprint, bytes 280:288."""
    buf = ctypes.create_string_buffer(512)
    assert _LIBC.proc_pid_rusage(os.getpid(), 4, buf) == 0
    return int.from_bytes(buf.raw[280:288], "little")


def _reset_peak() -> None:
    assert _LIBC.proc_reset_footprint_interval(os.getpid()) == 0


def _limit_now() -> int:
    lim = mx.set_cache_limit(0)
    mx.set_cache_limit(lim)
    return lim


def test_a_512_decode_stays_under_the_fitted_footprint():
    from gmlx.image import memfit
    from gmlx.image.vae import flux1_ae, tiling

    cfg = flux1_ae.Flux1AEConfig(
        in_channels=3, out_channels=3, latent_channels=16,
        block_out_channels=(128, 256, 512, 512), layers_per_block=2,
        norm_num_groups=32, eps=1e-6, scaling_factor=0.3611,
        shift_factor=0.1159)
    vae = flux1_ae.Flux1AE(cfg)
    vae.update(tree_map(lambda a: a.astype(mx.bfloat16), vae.parameters()))
    lat = mx.array(np.random.default_rng(0).standard_normal(
        (1, 16, 64, 64)).astype(np.float32))
    mx.eval(vae.parameters(), lat)
    tiling.decode(vae, lat, tiled=False, free=1e12)     # compile kernels
    bound = memfit.vae_peak("decode", 512, 512, family="z_image",
                            itemsize=2)
    # Arrays an earlier test left in reference cycles would otherwise
    # reach the cache at the next collection, inside the measurement.
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    before = _limit_now()
    time.sleep(1.0)
    _reset_peak()
    time.sleep(0.2)
    start = _peak_footprint()
    _reset_peak()
    out = tiling.decode(vae, lat, tiled=False, free=1e12)
    mx.synchronize()
    peak = _peak_footprint()
    assert out.shape == (1, 512, 512, 3)
    assert bool(mx.all(mx.isfinite(out)))
    grew = peak - start
    assert 0 < grew <= bound, (grew / 2**30, bound / 2**30)
    # A free that finds the cache under its limit keeps the buffer, so
    # the cache may end one buffer over the bound.
    assert mx.get_cache_memory() <= memfit.CACHE_BOUND + (256 << 20)
    assert _limit_now() == before
