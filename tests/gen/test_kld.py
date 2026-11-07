"""gmlx.gen.kld against a scalar port of llama-perplexity's loops
(tools/perplexity/perplexity.cpp): the uint16 records, the KL sum, the
argmax agreement, the chunking and the reference file layout."""

import io
import math
import struct

import numpy as np
import pytest

from gmlx.gen import kld as K

f32 = np.float32


def _nearest_int(x) -> int:
    val = f32(x) + f32(12582912.0)
    i = struct.unpack("<i", struct.pack("<f", val))[0]
    return (i & 0x007FFFFF) - 0x00400000


def _ref_records(row) -> list[int]:
    """llama.cpp's log_softmax(n_vocab, logits, uint16 *, tok), one row."""
    row = [f32(v) for v in row]
    max_l = min_l = row[0]
    for v in row[1:]:
        max_l = max(max_l, v)
        min_l = min(min_l, v)
    min_l = max(min_l, f32(max_l - f32(16)))
    sum_exp = 0.0
    for v in row:
        sum_exp += float(np.exp(f32(v - max_l)))
    lse = f32(math.log(sum_exp))
    min_log_prob = f32(f32(min_l - max_l) - lse)
    scale = f32(f32(max_l - min_l) / f32(65535))
    out = [0] * K.record_width(len(row))
    out[:4] = struct.unpack("<4H", struct.pack("<ff", scale, min_log_prob))
    if scale:
        inv = f32(f32(1) / scale)
        for i, v in enumerate(row):
            if v > min_l:
                out[4 + i] = _nearest_int(f32(inv * f32(v - min_l))) & 0xFFFF
    return out


def _ref_kld(row, rec) -> tuple[float, bool]:
    """llama.cpp's KL log_softmax: the float64 sum and the argmax match."""
    row = [f32(v) for v in row]
    max_l, imax = row[0], 0
    for i, v in enumerate(row[1:], 1):
        if v > max_l:
            max_l, imax = v, i
    sum_exp = 0.0
    for v in row:
        sum_exp += float(np.exp(f32(v - max_l)))
    lse = f32(math.log(sum_exp))
    scale, min_log_prob = struct.unpack("<ff", struct.pack("<4H", *rec[:4]))
    scale, min_log_prob = f32(scale), f32(min_log_prob)
    top = f32(max_l + lse)
    total, imax_base, p_max = 0.0, -1, f32(0)
    for i, v in enumerate(row):
        plb = f32(f32(scale * f32(rec[4 + i])) + min_log_prob)
        if i == 0 or plb > p_max:
            p_max, imax_base = plb, i
        if plb > f32(-16):
            total += float(f32(f32(np.exp(plb)) * f32(f32(plb - v) + top)))
    return total, imax == imax_base


def _logits(n: int, V: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((n, V)) * 4).astype(np.float32)
    z[1, : V // 2] -= 40          # half the row under max - 16
    z[2] = 1.5                    # a flat row: scale 0
    return z


@pytest.mark.parametrize("V", [37, 64])
def test_records_match_llama_cpp(V):
    z = _logits(5, V, 1)
    got = K.quantize_logprobs(z)
    assert got.shape == (5, K.record_width(V)) and got.dtype == np.uint16
    want = np.array([_ref_records(r) for r in z], dtype=np.uint16)
    np.testing.assert_array_equal(got, want)
    assert not got[2, 4:].any()


def test_kld_matches_llama_cpp():
    V = 37
    ref = _logits(5, V, 2)
    test = ref + np.random.default_rng(3).standard_normal(ref.shape).astype(np.float32)
    rec = K.quantize_logprobs(ref)
    kld, same = K.kld_rows(test, rec)
    want = [_ref_kld(t, r) for t, r in zip(test, rec.tolist())]
    np.testing.assert_allclose(kld, [w[0] for w in want], rtol=1e-6, atol=1e-9)
    assert same.tolist() == [w[1] for w in want]
    assert kld.dtype == np.float64


def test_kld_of_the_reference_itself_is_near_zero():
    ref = _logits(4, 64, 4)
    kld, same = K.kld_rows(ref, K.quantize_logprobs(ref))
    assert np.abs(kld).max() < 1e-4
    assert same.all()


def test_kld_sums_report_like_llama_cpp():
    ref = _logits(30, 40, 5)
    test = ref * 1.1
    sums = K.KldSums()
    rec = K.quantize_logprobs(ref)
    sums.add(test[:12], rec[:12])
    sums.add(test[12:], rec[12:])
    kld, same = K.kld_rows(test, rec)
    mean, unc = sums.mean_kld()
    assert sums.count == 30
    assert mean == pytest.approx(kld.mean())
    var = (kld * kld).mean() - kld.mean() ** 2
    assert unc == pytest.approx(math.sqrt(var / 29))
    p, p_unc = sums.same_top()
    assert p == pytest.approx(same.mean())
    assert p_unc == pytest.approx(math.sqrt(p * (1 - p) / 29))


def test_scored_span_and_chunk_count():
    assert K.scored_span(512) == (256, 255)
    assert K.chunk_count(4449, 512) == 8
    assert K.chunk_count(4449, 512, 3) == 3
    assert K.chunk_count(1024, 512, 9) == 2
    with pytest.raises(ValueError, match="at least 1024"):
        K.chunk_count(1023, 512)


def test_chunk_ids_swap_bos_on_a_copy():
    tokens = np.arange(10, 30, dtype=np.int32)
    ids = K.chunk_ids(tokens, 1, 8, bos=1)
    assert ids.tolist() == [1, *range(19, 26)]
    assert tokens[8] == 18
    assert K.chunk_ids(tokens, 1, 8).tolist() == list(range(18, 26))


def test_read_text_drops_one_trailing_newline(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(b"one\n\n")
    assert K.read_text(str(p)) == "one\n"
    p.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not UTF-8"):
        K.read_text(str(p))


def test_reference_file_round_trip():
    n_ctx, V, n_chunk = 8, 37, 2
    tokens = np.arange(20, dtype=np.int32)
    _, n = K.scored_span(n_ctx)
    chunks = [K.quantize_logprobs(_logits(5, V, 10 + c)[:n]) for c in range(n_chunk)]
    buf = io.BytesIO()
    K.write_reference_header(buf, n_ctx, V, n_chunk, tokens)
    for c in chunks:
        buf.write(c.tobytes())
    assert len(buf.getvalue()) == K.reference_bytes(n_ctx, V, n_chunk)
    buf.seek(0)
    head = K.read_reference_header(buf)
    assert head[:3] == (n_ctx, V, n_chunk)
    assert head[3].tolist() == list(range(16))
    for c in chunks:
        np.testing.assert_array_equal(K.read_reference_chunk(buf, n_ctx, V), c)
    with pytest.raises(ValueError, match="ends inside a chunk"):
        K.read_reference_chunk(buf, n_ctx, V)
