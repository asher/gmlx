#!/usr/bin/env python3
"""Preflight: codec classification (IQ hard-fail) and the happy path.

Reads only the GGUF header (no tensor data, no GPU). Mints tiny GGUFs whose
tensor *headers* declare a codec - the IQ buffer is all-zeros because preflight
classifies by the type enum and never dequantizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from gguf import GGUFWriter, GGMLQuantizationType as GT, quants  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

from gmlx.preflight import (  # noqa: E402
    SUPPORTED_QUANT_TYPES,
    UnsupportedCodecError,
    find_split_shards,
    preflight,
)


def _finish(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _mint_unsupported(path):
    """A GGUF whose only weight declares the (unkernelled) TQ1_0 codec.

    The whole IQ family (iq1-iq4) now has kernels; the ternary TQ codecs stay
    unkernelled, so TQ1_0 is the standing example of a codec the loader must
    refuse."""
    _, tsize = GGML_QUANT_SIZES[GT.TQ1_0]
    w = GGUFWriter(str(path), "llama")
    w.add_tensor("blk.0.attn_q.weight",
                 np.zeros((4, 2 * tsize), dtype=np.uint8), raw_dtype=GT.TQ1_0)
    _finish(w)


def _mint_ok(path):
    """A GGUF with only supported codecs: one F32 + one Q4_0 weight."""
    w = GGUFWriter(str(path), "llama")
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32), raw_dtype=GT.F32)
    s = (np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32) * 0.1)
    w.add_tensor("blk.0.attn_q.weight", quants.quantize(s, GT.Q4_0), raw_dtype=GT.Q4_0)
    _finish(w)


def test_unsupported_codec_hard_fails(tmp_path):
    p = tmp_path / "tq.gguf"
    _mint_unsupported(p)
    with pytest.raises(UnsupportedCodecError) as ei:
        preflight(str(p))
    err = ei.value
    assert "TQ1_0" in err.unsupported
    msg = str(err)
    assert "unsupported tensor types" in msg
    assert "TQ1_0" in msg
    assert "Re-quantize" in msg  # actionable remedy named


def test_iq_codecs_now_pass(tmp_path):
    """The whole IQ family (iq1-iq4) gained kernels - preflight must accept it."""
    for gt in (
        GT.IQ4_NL, GT.IQ4_XS, GT.IQ3_S, GT.IQ3_XXS,
        GT.IQ2_XXS, GT.IQ2_XS, GT.IQ2_S, GT.IQ1_S, GT.IQ1_M,
    ):
        wpb, tsize = GGML_QUANT_SIZES[gt]
        p = tmp_path / f"{gt.name}.gguf"
        w = GGUFWriter(str(p), "llama")
        w.add_tensor("blk.0.attn_q.weight",
                     np.zeros((4, (256 // wpb) * tsize), dtype=np.uint8),
                     raw_dtype=gt)
        _finish(w)
        pf = preflight(str(p))                      # must not raise
        assert pf.codec_histogram.get(gt.name) == 1


def test_supported_codecs_pass(tmp_path):
    p = tmp_path / "ok.gguf"
    _mint_ok(p)
    pf = preflight(str(p))
    assert pf.arch == "llama"
    assert pf.entry.model_type == "llama"
    assert pf.shards == [str(p)]
    assert pf.codec_histogram.get("Q4_0") == 1
    assert pf.codec_histogram.get("F32") == 1
    assert pf.n_tensors == 2


def test_supported_allowlist_matches_kernels():
    # preflight's allowlist must not drift from the codecs the kernels implement.
    kq = pytest.importorskip("mlx_kquant")
    assert {c.lower() for c in SUPPORTED_QUANT_TYPES} == set(kq.codecs())


# split-shard discovery
def test_find_split_shards_non_split_passthrough(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_text("x")
    assert find_split_shards(str(p)) == [str(p)]      # no -NNNNN-of-MMMMM => as-is


def test_find_split_shards_complete_set_in_order(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"model-{i:05d}-of-00003.gguf").write_text("x")
    got = find_split_shards(str(tmp_path / "model-00001-of-00003.gguf"))
    assert got == [str(tmp_path / f"model-{i:05d}-of-00003.gguf") for i in (1, 2, 3)]


def test_find_split_shards_missing_middle_shard_hard_fails(tmp_path):
    # Shard 2 of 3 absent: must NOT limp on with a partial set (which would
    # load_weights(strict=False) leaving the gap at random init) - fail loud.
    (tmp_path / "model-00001-of-00003.gguf").write_text("x")
    (tmp_path / "model-00003-of-00003.gguf").write_text("x")
    with pytest.raises(FileNotFoundError) as ei:
        find_split_shards(str(tmp_path / "model-00001-of-00003.gguf"))
    msg = str(ei.value)
    assert "1/3 shard(s) missing" in msg
    assert "model-00002-of-00003.gguf" in msg          # names the missing shard


def test_find_split_shards_missing_count_truncated(tmp_path):
    # Only shard 1 of 4 present: reports the count and lists at most three names.
    (tmp_path / "model-00001-of-00004.gguf").write_text("x")
    with pytest.raises(FileNotFoundError) as ei:
        find_split_shards(str(tmp_path / "model-00001-of-00004.gguf"))
    assert "3/4 shard(s) missing" in str(ei.value)


# pure message-formatting unit (no GGUF)
def test_codec_error_message_lists_unsupported():
    e = UnsupportedCodecError("llama", {"TQ1_0": 12, "TQ2_0": 3})
    msg = str(e)
    assert "unsupported tensor types" in msg
    assert "TQ1_0" in msg and "TQ2_0" in msg
    assert "(15 tensors)" in msg  # total unsupported tensor count reported
