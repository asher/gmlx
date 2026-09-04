#!/usr/bin/env python3
"""Fused-MoE codec eligibility for codecs the installed gguf-py does not know.

STQ1_0 is not an upstream ggml type, so two seams have to agree before a
STQ1_0 expert stack reaches the fused decode kernels: the capability probe
must list the codec, and the wire-K helper must find its block geometry
without gguf-py. A model whose gate/up stacks miss this runs the stock
SwitchGLU on every affected layer.
"""

from __future__ import annotations

import pytest

from gmlx.load.headerscan import QUANT_TYPE_FALLBACK
from gmlx.load.modules import _FusedMoeCaps, _kq_wire_k

STQ1_0_BLOCK, STQ1_0_BYTES = dict(QUANT_TYPE_FALLBACK.values())["STQ1_0"]


class _Wire:
    def __init__(self, last: int):
        self.shape = (256, 4096, last)


def _kq_has_stq1_0_glu() -> bool:
    import mlx_kquant as kq

    probe = getattr(kq, "codec_has_moe_glu", None)
    return probe is not None and probe("stq1_0")


def test_stq1_0_joins_the_fused_codecs_when_kq_covers_it():
    if not _kq_has_stq1_0_glu():
        pytest.skip("installed mlx-kquant has no stq1_0 moe_glu kernel")
    assert "stq1_0" in _FusedMoeCaps().kq_fused_codecs


def test_wire_k_reads_stq1_0_geometry_without_gguf_py():
    from gguf.constants import GGMLQuantizationType

    assert "STQ1_0" not in GGMLQuantizationType.__members__
    assert _kq_wire_k(_Wire(STQ1_0_BYTES * 24), "stq1_0") == STQ1_0_BLOCK * 24


def test_wire_k_rejects_a_partial_stq1_0_block():
    assert _kq_wire_k(_Wire(STQ1_0_BYTES * 24 + 1), "stq1_0") == -1


def test_wire_k_declines_an_unknown_codec_instead_of_raising():
    assert _kq_wire_k(_Wire(4096), "not_a_codec") == -1
