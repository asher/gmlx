"""headerscan: fast GGUF metadata scan vs gguf-py ground truth.

Parity on every KV type, skip accounting for large arrays, tensor-table
agreement, expected-size / truncation detection, and the preflight refusal
for truncated downloads.
"""

import shutil

import numpy as np
import pytest

from gmlx.headerscan import scan_gguf


def _mint(path, *, big_arrays=False, tensors=True):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "llama")  # sets general.architecture
    w.add_string("general.name", "scan-fixture")
    w.add_uint32("llama.block_count", 2)
    w.add_uint32("llama.context_length", 4096)
    w.add_float32("llama.rope.freq_base", 10000.0)
    w.add_bool("tokenizer.ggml.add_bos_token", True)
    w.add_array("llama.small_ints", [1, 2, 3])
    w.add_array("llama.small_floats", [0.5, 1.5])
    w.add_array("llama.small_bools", [True, False, True])
    w.add_array("llama.small_strs", ["a", "bb", "ccc"])
    if big_arrays:
        w.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(5000)])
        w.add_array("tokenizer.ggml.token_type", [1] * 5000)
    if tensors:
        w.add_tensor("token_embd.weight", np.zeros((32, 64), dtype=np.float32))
        w.add_tensor("blk.0.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_kv_parity_vs_ggufreader(tmp_path):
    p = _mint(tmp_path / "m.gguf")
    scan = scan_gguf(p, include_tensors=False)

    assert scan.kv["general.architecture"] == "llama"
    assert scan.kv["general.name"] == "scan-fixture"
    assert scan.kv["llama.block_count"] == 2
    assert scan.kv["llama.rope.freq_base"] == pytest.approx(10000.0)
    assert scan.kv["tokenizer.ggml.add_bos_token"] is True
    assert scan.kv["llama.small_ints"] == [1, 2, 3]
    assert scan.kv["llama.small_floats"] == pytest.approx([0.5, 1.5])
    assert scan.kv["llama.small_bools"] == [True, False, True]
    assert scan.kv["llama.small_strs"] == ["a", "bb", "ccc"]
    assert scan.skipped == {}


def test_large_arrays_skipped_not_read(tmp_path):
    p = _mint(tmp_path / "m.gguf", big_arrays=True)
    scan = scan_gguf(p, include_tensors=False)
    assert scan.skipped == {"tokenizer.ggml.tokens": 5000,
                            "tokenizer.ggml.token_type": 5000}
    assert "tokenizer.ggml.tokens" not in scan.kv
    # keys after the skipped arrays must still parse (order-independence)
    assert scan.kv["general.architecture"] == "llama"
    # raising the limit reads them for real
    full = scan_gguf(p, include_tensors=False, array_limit=10_000)
    assert full.kv["tokenizer.ggml.tokens"][4999] == "tok4999"
    assert full.skipped == {}


def test_tensor_table_matches_ggufreader(tmp_path):
    from gguf import GGUFReader

    p = _mint(tmp_path / "m.gguf", big_arrays=True)
    scan = scan_gguf(p)
    r = GGUFReader(p, "r")
    assert len(scan.tensors) == len(r.tensors) == scan.n_tensors
    for mine, ref in zip(scan.tensors, r.tensors):
        assert mine.name == ref.name
        assert mine.type_name == ref.tensor_type.name
        assert mine.nbytes == int(ref.n_bytes)
    assert scan.expected_size == scan.size
    assert not scan.truncated


def test_truncated_file_detected(tmp_path):
    p = _mint(tmp_path / "m.gguf")
    whole = scan_gguf(p)
    cut = tmp_path / "cut.gguf"
    shutil.copy(p, cut)
    with open(cut, "r+b") as f:
        f.truncate(whole.size - 128)
    scan = scan_gguf(str(cut))
    assert scan.truncated
    assert scan.expected_size == whole.expected_size


def test_preflight_refuses_truncated(tmp_path):
    from gmlx.preflight import preflight

    p = _mint(tmp_path / "m.gguf")
    with open(p, "r+b") as f:
        f.truncate(scan_gguf(p).size - 128)
    with pytest.raises(ValueError, match="truncated"):
        preflight(p, arch="llama")


def test_bad_magic_raises(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError, match="bad magic"):
        scan_gguf(str(p))
