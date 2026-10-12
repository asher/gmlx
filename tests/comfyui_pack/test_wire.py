"""The wire format against the server's golden bodies: the pack writes the
same bytes for the same tensors and fields, reads them back, and pins the
tensor hash. Needs torch and numpy only."""

from __future__ import annotations

import json
import os
import struct
import unittest

import numpy as np
import torch

from support import FIXTURES, load_client

C = load_client()
GOLDEN = os.path.join(FIXTURES, "golden")


def fixture(name: str) -> bytes:
    with open(os.path.join(GOLDEN, f"{name}.safetensors"), "rb") as f:
        return f.read()


def _cond():
    return torch.from_numpy((np.arange(6 * 8, dtype=np.float32).reshape(1, 6, 8) / 7).astype(np.float32))


def _bf16():
    bits = (np.arange(12, dtype=np.uint16) * 97 + 16256).reshape(3, 4).astype("<u2")
    return torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16)


def golden() -> dict:
    x = torch.from_numpy(np.linspace(-1.5, 1.5, 2 * 16 * 2 * 2, dtype=np.float32).reshape(2, 16, 2, 2))
    ch = C.tensor_hash(_cond())
    step = ({"x": x, "timestep": torch.tensor([0.75, 0.75], dtype=torch.float32)},
            {"model": "zit", "run": "0123456789abcdef", "convention": "comfy", "patch_table": None,
             "rows": [{"cond": ch, "num_tokens": 6}, {"cond": ch, "num_tokens": 4}]})
    return {"step-request": step,
            "step-response": ({"out": (x * -0.5).float()}, None),
            "tensor-cond": ({"cond": _cond()}, None),
            "tensor-bf16": ({"w": _bf16()}, None),
            "encode-request": ({"image_0": torch.full((4, 6, 3), 200, dtype=torch.uint8)},
                               {"model": "qi", "text": "a red lighthouse"})}


HASHES = {"tensor-cond": "6a5330a4fb32d8a2e612fc4b6983639c9de5d7837ef9e121f6ce16238941541f",
          "tensor-bf16": "6251a21245a4a21767bc4f22a07167f346e686f86d9bc6b9a6eb5f60923409a0"}


class GoldenBodies(unittest.TestCase):
    def test_each_body_is_the_golden_bytes(self):
        for name, (tensors, fields) in golden().items():
            with self.subTest(name):
                self.assertEqual(C.pack(tensors, fields), fixture(name))

    def test_each_golden_body_reads_back(self):
        for name, (tensors, fields) in golden().items():
            with self.subTest(name):
                got, got_fields = C.unpack(fixture(name))
                self.assertEqual(list(got), list(tensors))
                self.assertEqual(got_fields, fields or {})
                for k, t in got.items():
                    self.assertEqual(t.dtype, tensors[k].dtype)
                    self.assertTrue(torch.equal(t, tensors[k]))

    def test_numpy_input_writes_the_same_bytes(self):
        tensors, fields = golden()["encode-request"]
        self.assertEqual(C.pack({k: v.numpy() for k, v in tensors.items()}, fields),
                         fixture("encode-request"))

    def test_the_header_is_padded_and_the_fields_sorted(self):
        body = fixture("step-request")
        (n,) = struct.unpack("<Q", body[:8])
        self.assertEqual(n % 8, 0)
        meta = json.loads(body[8:8 + n])["__metadata__"]["gmlx"]
        self.assertEqual(meta, json.dumps(golden()["step-request"][1], sort_keys=True,
                                          separators=(",", ":")))

    def test_the_tensor_hash_is_pinned(self):
        for name, h in HASHES.items():
            with self.subTest(name):
                (t,) = golden()[name][0].values()
                self.assertEqual(C.tensor_hash(t), h)

    def test_a_noncontiguous_tensor_hashes_as_its_contiguous_copy(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertEqual(C.tensor_hash(t.t()), C.tensor_hash(t.t().contiguous()))

    def test_a_dtype_the_contract_does_not_carry_is_refused(self):
        with self.assertRaises(C.GmlxError):
            C.pack({"a": torch.zeros(2, dtype=torch.int64)})

    def test_table_id_is_the_sha256_of_the_canonical_json(self):
        import hashlib
        t = {"model": "m", "entries": [{"patches": [], "key": "k", "band": None}]}
        tid, body = C.table_id(t)
        self.assertEqual(body, b'{"entries":[{"band":null,"key":"k","patches":[]}],"model":"m"}')
        self.assertEqual(tid, hashlib.sha256(body).hexdigest())


class StepCallWritesTheGoldenBody(unittest.TestCase):
    def test_step_sends_the_golden_request_and_reads_the_golden_response(self):
        sent = {}

        class Capture(C.StepClient):
            def request(self, method, path, body=None, headers=None, **kw):
                sent.update(method=method, path=path, body=body, headers=headers)
                return 200, {"x-gmlx-forward-ms": "1.0"}, fixture("step-response")

        c = Capture("http://127.0.0.1:9")
        (tensors, fields) = golden()["step-request"]
        out, hdrs = c.step(tensors["x"], tensors["timestep"], fields)
        self.assertEqual(sent["path"], C.PREFIX + "/step")
        self.assertEqual(sent["headers"]["Content-Type"], C.CONTENT_TYPE)
        self.assertEqual(sent["body"], fixture("step-request"))
        self.assertTrue(torch.equal(out, tensors["x"] * -0.5))


if __name__ == "__main__":
    unittest.main()
