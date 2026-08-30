#!/usr/bin/env python3
"""_classify_local was rewritten from gguf-py's GGUFReader onto headerscan;
these tests pin the old report shape plus the STQ1_0 case the rewrite is for.
"""

from __future__ import annotations

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType as GT
from gguf.constants import GGML_QUANT_SIZES

from gmlx.commands.manage import _classify_local
from gmlx.load.headerscan import QUANT_TYPE_FALLBACK


def _finish(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _mint(path, codec=GT.Q4_0, arch="llama", gguf_type=None):
    wpb, tsize = GGML_QUANT_SIZES[codec]
    w = GGUFWriter(str(path), arch)
    if gguf_type is not None:
        w.add_type(gguf_type)
    w.add_tensor("plain.f32", np.zeros((4, 16), dtype=np.float32),
                 raw_dtype=GT.F32)
    w.add_tensor("blk.0.attn_q.weight",
                 np.zeros((4, (256 // wpb) * tsize), dtype=np.uint8),
                 raw_dtype=codec)
    _finish(w)


def test_classify_local_supported(tmp_path):
    p = tmp_path / "ok.gguf"
    _mint(p)
    rep = _classify_local(str(p))
    assert rep.arch == "llama"
    assert rep.histogram == {"F32": 1, "Q4_0": 1}
    assert rep.unsupported == {}
    assert rep.loadable_codecs is True
    assert rep.n_tensors == 2
    assert rep.gguf_type is None


def test_classify_local_unsupported_named(tmp_path):
    p = tmp_path / "tq.gguf"
    _mint(p, codec=GT.TQ1_0)
    rep = _classify_local(str(p))
    assert rep.unsupported == {"TQ1_0": 1}
    assert rep.loadable_codecs is False


def test_classify_local_gguf_type(tmp_path):
    p = tmp_path / "ad.gguf"
    _mint(p, gguf_type="adapter")
    assert _classify_local(str(p)).gguf_type == "adapter"


def test_classify_local_multi_shard(tmp_path):
    """Histograms accumulate across shards; arch and general.type come from
    shard 0 only."""
    p1 = tmp_path / "model-00001-of-00002.gguf"
    p2 = tmp_path / "model-00002-of-00002.gguf"
    _mint(p1, gguf_type="adapter")
    _, q6_tsize = GGML_QUANT_SIZES[GT.Q6_K]
    _, tq_tsize = GGML_QUANT_SIZES[GT.TQ1_0]
    w = GGUFWriter(str(p2), "qwen3")  # must NOT win over shard 0's "llama"
    w.add_tensor("blk.1.ffn_up.weight",
                 np.zeros((4, q6_tsize), dtype=np.uint8), raw_dtype=GT.Q6_K)
    w.add_tensor("blk.1.ffn_down.weight",
                 np.zeros((4, tq_tsize), dtype=np.uint8), raw_dtype=GT.TQ1_0)
    _finish(w)

    rep = _classify_local(str(p1))
    assert rep.arch == "llama"
    assert rep.gguf_type == "adapter"
    assert rep.histogram == {"F32": 1, "Q4_0": 1, "Q6_K": 1, "TQ1_0": 1}
    assert rep.unsupported == {"TQ1_0": 1}
    assert rep.n_tensors == 4


def test_classify_local_stq1_0_via_fallback(tmp_path):
    """A type-43 file classifies: GGUFReader rejects the id, headerscan's
    fallback names it."""
    (type_id, (name, (wpb, tsize))), = [
        (k, v) for k, v in QUANT_TYPE_FALLBACK.items() if v[0] == "STQ1_0"
    ]

    class _Stq:
        def __int__(self):
            return type_id

        def __index__(self):
            return type_id

    _Stq.name, _Stq.value = name, type_id
    sentinel = _Stq()
    GGML_QUANT_SIZES[sentinel] = (wpb, tsize)
    try:
        p = tmp_path / "stq1_0.gguf"
        w = GGUFWriter(str(p), "llama")
        w.add_tensor("blk.0.attn_q.weight",
                     np.zeros((4, (256 // wpb) * tsize), dtype=np.uint8),
                     raw_dtype=sentinel)
        _finish(w)
    finally:
        del GGML_QUANT_SIZES[sentinel]
    rep = _classify_local(str(p))
    assert rep.histogram == {"STQ1_0": 1}
    assert rep.unsupported == {}
    assert rep.loadable_codecs is True
