#!/usr/bin/env python3
"""The wire format of the image step contract: the golden request and
response bodies, the tensor hash, the bodies it refuses and the version
header. numpy only, nothing runs on the GPU.

The golden files under tests/fixtures/image_steps are what a client and
the server must write for the tensors and fields of ``golden()``, byte for
byte; ``python tests/serve/test_image_wire.py --write`` writes them again
after a deliberate format change."""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import pytest

from gmlx.serve import image_wire as w

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures",
                        "image_steps")


def _cond():
    return (np.arange(6 * 8, dtype=np.float32).reshape(1, 6, 8) / 7).astype(
        np.float32)


def _bf16():
    bits = (np.arange(12, dtype=np.uint16) * 97 + 16256).reshape(3, 4)
    return ("BF16", (3, 4), bits.astype("<u2").tobytes())


def golden() -> dict:
    """name -> (tensors, fields) of each golden body."""
    x = np.linspace(-1.5, 1.5, 2 * 16 * 2 * 2, dtype=np.float32).reshape(
        2, 16, 2, 2)
    cond_hash = w.tensor_hash("F32", (1, 6, 8), _cond().tobytes())
    step = ({"x": x, "timestep": np.array([0.75, 0.75], np.float32)},
            {"model": "zit", "run": "0123456789abcdef", "convention": "comfy",
             "patch_table": None,
             "rows": [{"cond": cond_hash, "num_tokens": 6},
                      {"cond": cond_hash, "num_tokens": 4}]})
    out = ({"out": (x * -0.5).astype(np.float32)}, None)
    cond = ({"cond": _cond()}, None)
    bf16 = ({"w": _bf16()}, None)
    image = ({"image_0": np.full((4, 6, 3), 200, np.uint8)},
             {"model": "qi", "text": "a red lighthouse"})
    return {"step-request": step, "step-response": out,
            "tensor-cond": cond, "tensor-bf16": bf16, "encode-request": image}


HASHES = {
    "tensor-cond": "6a5330a4fb32d8a2e612fc4b6983639c9de5d7837ef9e121f6ce16238941541f",
    "tensor-bf16": "6251a21245a4a21767bc4f22a07167f346e686f86d9bc6b9a6eb5f60923409a0",
}


def _fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, f"{name}.safetensors"), "rb") as f:
        return f.read()


@pytest.mark.parametrize("name", sorted(golden()))
def test_a_body_is_the_golden_bytes_and_reads_back(name):
    tensors, fields = golden()[name]
    body = w.write(tensors, fields)
    assert body == _fixture(name)
    got, got_fields = w.read(body)
    assert list(got) == list(tensors)
    assert got_fields == (fields or {})
    for k, t in got.items():
        want = tensors[k]
        if isinstance(want, tuple):
            assert (t.dtype, t.shape, bytes(t.raw)) == (want[0], want[1],
                                                         want[2])
        else:
            assert t.shape == want.shape
            assert np.array_equal(t.numpy(), want)
    # The header is padded with spaces to 8 bytes and the fields are
    # sorted compact JSON.
    (n,) = struct.unpack("<Q", body[:8])
    assert n % 8 == 0
    if fields:
        meta = json.loads(body[8:8 + n])["__metadata__"]["gmlx"]
        assert meta == json.dumps(fields, sort_keys=True,
                                  separators=(",", ":"))


@pytest.mark.parametrize("name", sorted(HASHES))
def test_the_tensor_hash_is_pinned(name):
    tensors, _ = golden()[name]
    (t,) = w.read(w.write(tensors))[0].values()
    assert t.hash() == HASHES[name]


def test_bf16_reads_as_its_bits_and_becomes_bfloat16_in_mlx():
    mx = pytest.importorskip("mlx.core")
    (t,) = w.read(_fixture("tensor-bf16"))[0].values()
    a = t.mx()
    assert a.dtype == mx.bfloat16 and a.shape == (3, 4)
    again = w.write({"w": a})
    assert again == _fixture("tensor-bf16")


def _header(spec: dict, data: bytes = b"") -> bytes:
    hj = json.dumps(spec).encode()
    return struct.pack("<Q", len(hj)) + hj + data


@pytest.mark.parametrize("body, match", [
    (b"\0\0", "shorter than"),
    (struct.pack("<Q", 1 << 40), "out of range"),
    (struct.pack("<Q", 3) + b"abc", "not JSON"),
    (_header([1]), "not a JSON object"),
    (_header({"a": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}},
             b"\0" * 8), "dtype 'I64'"),
    (_header({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
             b"\0" * 4), "do not hold 2 F32"),
    (_header({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
              "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]}},
             b"\0" * 8), "overlap"),
    (_header({"a": {"dtype": "F32", "shape": [1]}}), "needs dtype"),
    (_header({"__metadata__": {"gmlx": "[1]"}}), "not a JSON object"),
    (_header({"__metadata__": {"gmlx": 3}}), "map names to strings"),
])
def test_a_malformed_body_is_refused_by_name(body, match):
    with pytest.raises(w.WireError, match=match) as e:
        w.read(body)
    assert e.value.status == 400


def test_a_body_over_the_limit_is_413():
    with pytest.raises(w.WireError) as e:
        w.read(_fixture("step-request"), max_bytes=64)
    assert e.value.status == 413


def test_a_dtype_the_contract_does_not_carry_is_refused_on_write():
    with pytest.raises(w.WireError, match="int64"):
        w.write({"a": np.zeros(2, np.int64)})


@pytest.mark.parametrize("value, match", [
    (None, "has no X-Gmlx-Steps-Contract header"),
    ("2.0", "Upgrade the gmlx server"),
    ("0.9", "Upgrade the client"),
    ("one", "MAJOR.MINOR"),
])
def test_the_version_header_names_the_side_to_upgrade(value, match):
    with pytest.raises(w.WireError, match=match):
        w.check_version(value)


@pytest.mark.parametrize("value", ["1.0", "1.7", "1"])
def test_any_minor_of_the_major_is_served(value):
    w.check_version(value)


if __name__ == "__main__" and "--write" in sys.argv:
    os.makedirs(FIXTURES, exist_ok=True)
    for name, (tensors, fields) in golden().items():
        body = w.write(tensors, fields)
        with open(os.path.join(FIXTURES, f"{name}.safetensors"), "wb") as f:
            f.write(body)
        (t,) = list(w.read(body)[0].values())[:1]
        print(name, len(body), t.hash())
