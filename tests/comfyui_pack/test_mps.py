"""Loads with the MPS device on: a stock model resident on MPS stays
resident while a gmlx model loads and samples, and ComfyUI frees no memory
on the gmlx model's behalf.

Runs only with ``GMLX_TEST_MPS=1`` and ``COMFYUI_DIR``; it opens a Metal
device, so it runs on its own."""

from __future__ import annotations

import os
import unittest

MPS = os.environ.get("GMLX_TEST_MPS") == "1"


@unittest.skipUnless(MPS, "the MPS run needs GMLX_TEST_MPS=1")
class MpsEviction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import support
        support.setup_comfy()
        cls.P = support.load_pack()
        import comfy.model_management as mm
        cls.mm = mm
        if mm.get_torch_device().type != "mps":
            raise unittest.SkipTest(f"ComfyUI chose {mm.get_torch_device()}, not mps")
        cls.stub = support.Stub()
        cls.client = cls.P.client.StepClient(cls.stub.url)
        cls.client.handshake()

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def test_a_resident_mps_model_survives_a_gmlx_load_and_sample(self):
        import torch

        import comfy.model_patcher
        import nodes as comfy_nodes
        mm = self.mm
        dev = mm.get_torch_device()
        lin = torch.nn.Linear(256, 256)
        other = comfy.model_patcher.ModelPatcher(lin, load_device=dev, offload_device=torch.device("cpu"))
        mm.load_models_gpu([other])
        self.assertEqual(lin.weight.device.type, "mps")
        m = self.P.model.load_model(self.client, "z-image-turbo")
        calls = []
        orig = mm.free_memory

        def spy(*a, **k):
            calls.append((a, k))
            return orig(*a, **k)
        mm.free_memory = spy
        try:
            mm.load_models_gpu([m])
            pos, _ = self.client.encode("z-image-turbo", "a lighthouse at dusk")
            neg, _ = self.client.encode("z-image-turbo", "blurry")
            latent = {"samples": torch.zeros(1, 16, 8, 8)}
            (out,) = comfy_nodes.common_ksampler(m, 1, 3, 2.0, "euler", "simple", [[pos["cond"], {}]],
                                                 [[neg["cond"], {}]], latent)
        finally:
            mm.free_memory = orig
        self.assertEqual(calls, [])
        loaded = [lm.model for lm in mm.current_loaded_models]
        self.assertTrue(any(x is other for x in loaded))
        self.assertEqual(lin.weight.device.type, "mps")
        self.assertEqual(out["samples"].device.type, "cpu")
        self.assertEqual(len(self.stub.events("step")), 3)
        self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))


if __name__ == "__main__":
    unittest.main()
