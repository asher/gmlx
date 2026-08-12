"""The activation dtype knob: resolution, the auto rule, and what it reaches.

One dtype seeds the graph. The embedding's out_dtype and the loader's cast of
non-quantized params both read it, and every downstream kquant matmul returns
its activation dtype. bfloat16 stays the default on every device; float16 is
opt-in for Apple GPUs before Apple9, which lack native bfloat16 arithmetic.
CPU-only.
"""

from __future__ import annotations

import argparse
import os

import mlx.core as mx
import mlx.nn as nn
import mlx_kquant as kq
import pytest

from gmlx import dtypes
from gmlx.cli import add_load_args
from gmlx.modules import install_kquant_modules


@pytest.fixture(autouse=True)
def _clean_env():
    """Each case starts with the variable unset and the arch probe uncached.

    The CLI action writes os.environ directly rather than through monkeypatch,
    so the teardown clears it unconditionally instead of relying on an undo.
    """
    os.environ.pop(dtypes.ENV_VAR, None)
    dtypes._arch_gen = None
    yield
    os.environ.pop(dtypes.ENV_VAR, None)
    dtypes._arch_gen = None


# --------------------------------------------------------------- resolution


def test_default_is_bfloat16():
    assert dtypes.activation_dtype() == mx.bfloat16
    assert dtypes.activation_dtype_name() == "bf16"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("float16", mx.float16),
        ("fp16", mx.float16),
        ("bfloat16", mx.bfloat16),
        ("bf16", mx.bfloat16),
        ("FLOAT16", mx.float16),
        ("  fp16  ", mx.float16),
    ],
)
def test_explicit_values_and_aliases(monkeypatch, value, expected):
    monkeypatch.setenv(dtypes.ENV_VAR, value)
    assert dtypes.activation_dtype() == expected


@pytest.mark.parametrize("value", ["garbage", "float32", "int8", ""])
def test_malformed_degrades_to_default(monkeypatch, value):
    """A bad value must not crash a serve boot; it falls back to bfloat16."""
    monkeypatch.setenv(dtypes.ENV_VAR, value)
    assert dtypes.activation_dtype() == mx.bfloat16


# ------------------------------------------------------------------- auto


@pytest.mark.parametrize(
    "gen,expected",
    [
        (13, mx.float16),   # M1, no native bfloat16
        (14, mx.float16),   # M2, no native bfloat16
        (15, mx.bfloat16),  # M3, Apple9 onward
        (17, mx.bfloat16),  # M5
        (0, mx.bfloat16),   # unknown device keeps the default
    ],
)
def test_auto_follows_gpu_generation(monkeypatch, gen, expected):
    monkeypatch.setenv(dtypes.ENV_VAR, "auto")
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: gen)
    assert dtypes.activation_dtype() == expected


def test_explicit_value_overrides_the_auto_rule(monkeypatch):
    """An M1-class device still honours an explicit bfloat16 request."""
    monkeypatch.setattr(dtypes, "gpu_arch_gen", lambda: 13)
    monkeypatch.setenv(dtypes.ENV_VAR, "bfloat16")
    assert dtypes.activation_dtype() == mx.bfloat16


# ------------------------------------------------------------- arch probe


@pytest.mark.parametrize(
    "arch,gen",
    [
        ("applegpu_g13s", 13),
        ("applegpu_g14p", 14),
        ("applegpu_g17s", 17),
        ("", 0),
        ("some_other_gpu", 0),
    ],
)
def test_arch_string_parses_to_generation(monkeypatch, arch, gen):
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    assert dtypes.gpu_arch_gen() == gen


def test_arch_probe_survives_a_raising_device_info(monkeypatch):
    """A device without Metal must not take the loader down."""

    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(mx, "device_info", _boom)
    assert dtypes.gpu_arch_gen() == 0
    assert dtypes.has_native_bfloat16() is True


def test_arch_probe_is_cached(monkeypatch):
    calls = []

    def _info():
        calls.append(1)
        return {"architecture": "applegpu_g13s"}

    monkeypatch.setattr(mx, "device_info", _info)
    assert dtypes.gpu_arch_gen() == 13
    assert dtypes.gpu_arch_gen() == 13
    assert len(calls) == 1


# ------------------------------------------------------- what it reaches


class EmbedBlock(nn.Module):
    def __init__(self, vocab, dims):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dims)


def _swapped_embedding(vocab=64, dims=256):
    mx.random.seed(0)
    model = EmbedBlock(vocab, dims)
    table = mx.random.normal((vocab, dims))
    wq, scales = kq.quantize(table, "q8_0")
    n = install_kquant_modules(model, {"embed_tokens.weight": "q8_0"})
    assert n == 1
    emb = model.embed_tokens
    emb.weight = wq.reshape(vocab, -1)
    emb.scales = scales
    return emb


@pytest.mark.parametrize(
    "value,expected",
    [(None, mx.bfloat16), ("float16", mx.float16), ("bfloat16", mx.bfloat16)],
)
def test_embedding_emits_the_activation_dtype(monkeypatch, value, expected):
    """The embedding seeds the graph, so its output dtype is the knob."""
    if value is not None:
        monkeypatch.setenv(dtypes.ENV_VAR, value)
    emb = _swapped_embedding()
    assert emb.out_dtype == expected
    out = emb(mx.array([0, 5, 17]))
    assert out.dtype == expected
    assert out.shape == (3, 256)


def test_embedding_values_match_across_dtypes(monkeypatch):
    """Switching dtype changes width, not what the table decodes to."""
    monkeypatch.setenv(dtypes.ENV_VAR, "bfloat16")
    ref = _swapped_embedding()(mx.array([0, 5, 17])).astype(mx.float32)
    monkeypatch.setenv(dtypes.ENV_VAR, "float16")
    got = _swapped_embedding()(mx.array([0, 5, 17])).astype(mx.float32)
    # float16 carries more mantissa than bfloat16, so it is at least as close
    # to the bfloat16 rendering as bfloat16's own rounding step is wide.
    assert mx.allclose(ref, got, rtol=8e-3, atol=8e-3)


# --------------------------------------------------------------------- cli


@pytest.mark.parametrize("value", ["float16", "bfloat16", "auto"])
def test_cli_flag_publishes_the_env_var(monkeypatch, value):
    ap = argparse.ArgumentParser()
    add_load_args(ap)
    args = ap.parse_args(["--dtype", value])
    assert args.dtype == value
    assert os.environ.get(dtypes.ENV_VAR) == value


def test_cli_without_the_flag_leaves_the_env_alone():
    ap = argparse.ArgumentParser()
    add_load_args(ap)
    args = ap.parse_args([])
    assert args.dtype is None
    assert dtypes.ENV_VAR not in os.environ
