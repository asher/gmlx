"""GGML type id 43 names two formats: STQ1_0 (ternary weights) and MXFP8
(gather-only lookup tables). The bytes a tensor spans tell them apart."""
from __future__ import annotations

import pytest
from helpers import mint_type43_gguf

import gmlx.load.remote as remote
from gmlx.load.headerscan import TYPE_43_FORMATS, scan_gguf, type_43_format

ALIGN = 32


def _size(n: int, fmt: str) -> int:
    block, tsize = TYPE_43_FORMATS[fmt]
    return n // block * tsize


def _mint(path, tensors, **kw) -> bytes:
    return mint_type43_gguf(
        path, [(name, (n,), fmt, None) for name, n, fmt in tensors], **kw)


# An STQ1_0 weight and an MXFP8 table in one file, in both orders, so each
# format is resolved once between tensors and once at the end of the file.
_ORDERS = [
    [("blk.0.ffn_up.weight", 1024, "STQ1_0"),
     ("per_layer_token_embd.weight", 512, "MXFP8")],
    [("per_layer_token_embd.weight", 512, "MXFP8"),
     ("blk.0.ffn_up.weight", 1024, "STQ1_0")],
]


@pytest.mark.parametrize("tensors", _ORDERS)
@pytest.mark.parametrize("pad_last", [True, False])
def test_scan_resolves_each_type43_tensor(tmp_path, tensors, pad_last):
    path = tmp_path / "m.gguf"
    _mint(path, tensors, pad_last=pad_last)
    h = scan_gguf(str(path))
    assert [(t.name, t.type_name, t.nbytes) for t in h.tensors] == [
        (name, fmt, _size(n, fmt)) for name, n, fmt in tensors]
    assert not h.truncated


def test_remote_report_resolves_both_formats(tmp_path):
    buf = _mint(tmp_path / "m.gguf", _ORDERS[0])
    rep = remote.classify_header(buf, len(buf))
    assert rep.histogram == {"STQ1_0": 1, "MXFP8": 1}
    assert rep.loadable_codecs


def test_a_truncated_table_at_the_end_reads_as_mxfp8(tmp_path):
    # A cut download: the short span still lands nearest MXFP8, and the
    # size check flags the file.
    path = tmp_path / "m.gguf"
    _mint(path, _ORDERS[0], cut=100)
    h = scan_gguf(str(path))
    assert h.tensors[-1].type_name == "MXFP8"
    assert h.truncated


def test_a_span_that_fits_neither_format_raises():
    n = 1024
    with pytest.raises(ValueError, match="neither STQ1_0"):
        type_43_format(n, _size(n, "STQ1_0") + 3 * ALIGN, ALIGN)
    # A header prefix with no file size cannot see the last span; it reads
    # as STQ1_0 and the local scan decides again.
    assert type_43_format(n, None, ALIGN) == "STQ1_0"
