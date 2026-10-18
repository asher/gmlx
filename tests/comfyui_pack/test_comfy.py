"""The pack inside ComfyUI against the stub server: the weightless models,
loads that touch no weights, patch capture and table ids, clones, the
sampler reaching the remote forward, refusals by name, the 409 path,
interrupt, the CLIP and VAE, the variant sigmas and the round trip.

Needs ``COMFYUI_DIR``. Run with ``python -m unittest`` from this folder."""

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import statistics
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid

import numpy as np
import torch

import support

support.setup_comfy()
P = support.load_pack()

import comfy.latent_formats  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.supported_models  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.weight_adapter  # noqa: E402
import folder_paths  # noqa: E402
import nodes as comfy_nodes  # noqa: E402
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # noqa: E402

C = P.client
ZID, QID, QTID = "z-image-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo"
KID = "krea-2-turbo"
F2ID, F2BID = "flux-2-klein", "flux-2-klein-base"
F29ID, F29BID = "flux-2-klein-9b", "flux-2-klein-base-9b"
F29KID = "flux-2-klein-9b-kv"
F3ID, F3BID = "flux-3-klein", "flux-3-klein-base"
MID = "moss-image"
STEPS = 4
LORAS = tempfile.mkdtemp(prefix="gmlx-pack-loras-")
folder_paths.add_model_folder_path("loras", LORAS)
LORA = "zimage-test-lora.safetensors"
LORA_TARGETS = support.write_zimage_lora(os.path.join(LORAS, LORA), support.manifest("z_image"))
LORA_F3 = "flux3-test-lora.safetensors"
LORA_F3_KEYS = {f"diffusion_model.{s}.weight" for s in
                support.write_flux3_lora(os.path.join(LORAS, LORA_F3), support.manifest("flux3_klein"))}
LORA_F2 = "flux2-test-lora.safetensors"
LORA_F2_KEYS = {f"diffusion_model.{s}.weight" for s in
                support.write_flux2_lora(os.path.join(LORAS, LORA_F2), support.manifest("flux2_klein"))}
LORA_F29 = "flux2-9b-test-lora.safetensors"
support.write_flux2_lora(os.path.join(LORAS, LORA_F29), support.manifest("flux2_klein_9b"))
LORA_M = "moss-test-lora.safetensors"
LORA_M_KEYS = {f"diffusion_model.{s}.weight" for s in
               support.write_moss_lora(os.path.join(LORAS, LORA_M), support.manifest("moss_image"))}
# FLUX.3 came to ComfyUI after the pinned checkout; there its models are
# refused by name.
HAS_FLUX3 = hasattr(comfy.supported_models, "Flux3")
needs_flux3 = unittest.skipUnless(HAS_FLUX3, "this ComfyUI has no FLUX.3")


def rss() -> int:
    import psutil
    return psutil.Process().memory_info().rss


class Base(unittest.TestCase):
    stub_args: tuple = ()

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub(*cls.stub_args)
        cls.client = C.StepClient(cls.stub.url)
        cls.client.handshake()

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def setUp(self):
        self.stub.forget()
        self.client.forget()
        self.stub.reset()

    def model(self, mid=ZID):
        return P.model.load_model(self.client, mid)

    def cond(self, mid, text, tokens=None):
        if tokens is None:
            out, _ = self.client.encode(mid, text)
            return [[out["cond"], {}]]
        s, _, d = self.client.request("POST", C.PREFIX + "/encode",
                                      f'{{"model": "{mid}", "text": "{text}", "num_tokens": {tokens}}}'.encode(),
                                      {"Content-Type": "application/json"})
        self.assertEqual(s, 200)
        out, fields = C.unpack(d)
        self.client.known_tensors.add(fields["cond"])
        return [[out["cond"], {}]]

    def sample(self, model, mid=ZID, seed=1, cfg=2.0, neg_tokens=None, steps=STEPS, size=(64, 64),
               pos=None, neg=None):
        pos = pos or self.cond(mid, "a red bicycle in the rain")
        neg = neg or self.cond(mid, "blurry", neg_tokens)
        ch = model.gmlx.status["latent"]["channels"]
        stride = model.gmlx.status["latent"]["stride"]
        latent = {"samples": torch.zeros(1, ch, size[0] // stride, size[1] // stride)}
        (out,) = comfy_nodes.common_ksampler(model, seed, steps, cfg, "euler", "simple", pos, neg, latent)
        return out["samples"]


class Compat(unittest.TestCase):
    def test_every_internal_the_pack_touches_is_present(self):
        self.assertIsNone(P.compat.first_missing(), P.compat.comfyui_version())

    def test_a_missing_internal_is_named_with_the_version(self):
        names = P.compat.NAMES
        try:
            P.compat.NAMES = names + (("comfy.model_patcher", "ModelPatcher.no_such_name"),)
            with self.assertRaises(ImportError) as e:
                P.compat.check()
            self.assertIn("comfy.model_patcher.ModelPatcher.no_such_name", str(e.exception))
            self.assertIn(P.compat.comfyui_version(), str(e.exception))
        finally:
            P.compat.NAMES = names


class StandIns(unittest.TestCase):
    """The nodes that load when the full import fails: the same names,
    inputs and outputs, and an error that names the change and the upgrade."""

    def setUp(self):
        self.stub = importlib.import_module(f"{P.__name__}.stub")
        self.env = os.environ.get("GMLX_COMFY_SETTINGS")
        d = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(d, "comfyui.json")
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_install": "homebrew", "gmlx_command": sys.executable}, f)
        # The model list probes in the background: one answer for both sides.
        P.discovery.forget()
        P.discovery.refresh().join(30)

    def tearDown(self):
        if self.env is None:
            os.environ.pop("GMLX_COMFY_SETTINGS", None)
        else:
            os.environ["GMLX_COMFY_SETTINGS"] = self.env

    def test_the_stand_ins_match_the_nodes(self):
        fakes, names = self.stub.nodes(ImportError("probe"))
        self.assertEqual(set(fakes), set(P.NODE_CLASS_MAPPINGS))
        self.assertEqual(set(names), set(P.NODE_DISPLAY_NAME_MAPPINGS))
        for name, real in P.NODE_CLASS_MAPPINGS.items():
            fake = fakes[name]
            self.assertEqual(fake.RETURN_TYPES, real.RETURN_TYPES, name)
            self.assertEqual(getattr(fake, "RETURN_NAMES", None), getattr(real, "RETURN_NAMES", None))
            self.assertEqual(bool(getattr(fake, "OUTPUT_NODE", False)),
                             bool(getattr(real, "OUTPUT_NODE", False)), name)
            ri, fi = real.INPUT_TYPES(), fake.INPUT_TYPES()
            self.assertEqual({k: list(v) for k, v in ri.items()},
                             {k: list(v) for k, v in fi.items()}, name)
            for sect, inputs in ri.items():
                for key, spec in inputs.items():
                    other = fi[sect][key]
                    self.assertEqual(spec[0], other[0], (name, key))
                    if len(spec) > 1 and "default" in spec[1]:
                        self.assertEqual(spec[1]["default"], other[1]["default"], (name, key))

    def test_a_stand_in_names_the_internal_and_the_upgrade(self):
        names = P.compat.NAMES
        try:
            P.compat.NAMES = names + (("comfy.model_patcher", "ModelPatcher.no_such_name"),)
            fakes, _ = self.stub.nodes(ImportError("probe"))
        finally:
            P.compat.NAMES = names
        with self.assertRaises(RuntimeError) as e:
            fakes["GmlxModelLoader"]().run(model_name="z-image-turbo")
        msg = str(e.exception)
        for want in ("comfy.model_patcher.ModelPatcher.no_such_name", P.compat.comfyui_version(),
                     P.version.COMFYUI_TESTED, "brew upgrade gmlx",
                     "then run `gmlx comfyui install` and restart ComfyUI"):
            self.assertIn(want, msg)
        self.assertNotIn("not installed", msg)
        shown = fakes["GmlxServerStatus"]().run(model_name="z-image-turbo")
        self.assertEqual(shown["result"], (msg,))


class ServerStatus(unittest.TestCase):
    """The status node: the server's view of a model, the note about a
    server from another gmlx, and how gmlx manages the nodes."""

    def setUp(self):
        self.env = os.environ.get("GMLX_COMFY_SETTINGS")
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(tempfile.mkdtemp(), "comfyui.json")

    def tearDown(self):
        if self.env is None:
            os.environ.pop("GMLX_COMFY_SETTINGS", None)
        else:
            os.environ["GMLX_COMFY_SETTINGS"] = self.env

    def test_the_status_carries_the_version_note_and_how_the_nodes_update(self):
        stub = support.Stub("--gmlx-version", "99.0.0")
        try:
            out = P.nodes.GmlxServerStatus().status(ZID, server_url=stub.url)
        finally:
            stub.close()
        text = out["result"][0]
        self.assertEqual(out["ui"]["text"], [text])
        lines = text.splitlines()
        self.assertEqual(len(lines), 3, lines)
        self.assertIn("resident", lines[0])
        self.assertTrue(lines[1].startswith("gmlx: the server runs gmlx 99.0.0"), lines[1])
        self.assertIn("Restart ComfyUI", lines[1])
        self.assertEqual(lines[2], P.nodes.MANAGED)
        self.assertIn("ComfyUI-Manager lists them as an unknown pack", lines[2])


class Weightless(Base):
    """Weightless builds on the meta device, and the class check against the server's answer."""

    def test_every_family_builds_on_meta_under_200_mb(self):
        for mid, cls in ((ZID, "ZImage"), (QID, "QwenImage21"), (KID, "Krea2")):
            with self.subTest(mid):
                gc.collect()
                r0 = rss()
                m = self.model(mid)
                gc.collect()
                grew = (rss() - r0) / 2 ** 20
                params = list(m.model.parameters())
                self.assertEqual(type(m.model.model_config).__name__, cls)
                self.assertTrue(params and all(p.device.type == "meta" for p in params))
                self.assertLess(grew, 200.0)
                self.assertIsInstance(m, P.patcher.GmlxModelPatcher)
                self.assertEqual(m.model.get_dtype(), torch.float32)

    def test_zimage_detection_reads_the_inline_norm(self):
        m = self.model(ZID)
        man = self.client.manifest(ZID)
        (name,) = man["inline"]
        t = P.model.manifest_state_dict(man)[name]
        want = torch.std(t, unbiased=False).item() < 0.42
        self.assertEqual(m.model.model_config.unet_config["allow_fp16"], want)

    def test_a_class_mismatch_names_both(self):
        with self.assertRaises(C.GmlxError) as e:
            P.model.build_weightless(self.client.manifest(ZID), "QwenImage21")
        self.assertIn("ZImage", str(e.exception))
        self.assertIn("QwenImage21", str(e.exception))

    def test_flux2_builds_on_meta_and_reads_the_reference_rule_of_its_dit(self):
        for mid, depths in ((F2ID, (5, 20)), (F2BID, (5, 20)), (F29ID, (8, 24)), (F29BID, (8, 24))):
            with self.subTest(mid):
                gc.collect()
                r0 = rss()
                m = self.model(mid)
                gc.collect()
                grew = (rss() - r0) / 2 ** 20
                uc = m.model.model_config.unet_config
                self.assertEqual(type(m.model.model_config).__name__, "Flux2")
                self.assertEqual((uc["guidance_embed"], uc["depth"], uc["depth_single_blocks"]),
                                 (False, *depths))
                f = m.model.diffusion_model.forward
                self.assertEqual((f.reads_guidance, f.ref_method, f.flow_map), (False, "index", False))
                self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))
                self.assertLess(grew, 200.0)
                self.assertEqual(m.model.get_dtype(), torch.float32)

    @needs_flux3
    def test_flux3_builds_on_meta_with_its_flow_map_from_the_manifest(self):
        for mid, flow in ((F3ID, True), (F3BID, False)):
            with self.subTest(mid):
                names = [t["name"] for t in self.client.manifest(mid)["tensors"]]
                count = lambda g: len({n.split(".")[1] for n in names if n.startswith(g)})  # noqa: E731
                gc.collect()
                r0 = rss()
                m = self.model(mid)
                gc.collect()
                grew = (rss() - r0) / 2 ** 20
                uc = m.model.model_config.unet_config
                self.assertEqual(type(m.model.model_config).__name__, "Flux3")
                self.assertEqual((uc["flow_map"], uc["fused_qkv"], uc["depth"], uc["depth_single_blocks"]),
                                 (flow, False, count("txt_mode_blocks."), count("single_blocks.")))
                self.assertIs(m.model.diffusion_model.flow_map, flow)
                self.assertIs(m.model.diffusion_model.forward.flow_map, flow)
                self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))
                self.assertLess(grew, 200.0)
                self.assertEqual(m.model.get_dtype(), torch.float32)

    @needs_flux3
    def test_the_fused_qkv_flux3_layout_is_refused_by_name(self):
        man = self.client.manifest(F3ID)
        fused = []
        for t in man["tensors"]:
            n = t["name"]
            if n.endswith(".q_proj.weight"):
                fused.append({**t, "name": n.replace(".q_proj.", ".qkv."),
                              "shape": [3 * t["shape"][0], t["shape"][1]]})
            elif not n.endswith((".k_proj.weight", ".v_proj.weight")):
                fused.append(t)
        with self.assertRaises(C.GmlxRefusal) as e:
            P.model.build_weightless({**man, "tensors": fused}, "Flux3")
        self.assertIn("fused qkv layout (single_blocks.0.qkv.weight)", str(e.exception))

    @unittest.skipIf(HAS_FLUX3, "this ComfyUI has FLUX.3")
    def test_flux3_on_a_comfyui_without_it_is_refused_by_name(self):
        for mid in (F3ID, F3BID):
            with self.subTest(mid):
                with self.assertRaises(C.GmlxError) as e:
                    self.model(mid)
                for want in (f"ComfyUI {P.compat.comfyui_version()} has no Flux3 model",
                             repr(mid), "update ComfyUI"):
                    self.assertIn(want, str(e.exception))
        # The nodes load beside it: no internal the pack needs is missing.
        self.assertIsNone(P.compat.first_missing())
        self.assertEqual(type(self.model(KID).model.model_config).__name__, "Krea2")


class Loads(Base):
    """Loads that free nothing on the cpu device, and clones that keep the class."""

    def test_load_models_gpu_touches_no_weights_and_evicts_nothing(self):
        cpu = torch.device("cpu")
        other = comfy.model_patcher.ModelPatcher(torch.nn.Linear(256, 256), cpu, cpu)
        mm.load_models_gpu([other])
        target = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.8)[0]
        calls = []
        orig = mm.free_memory

        def spy(*a, **k):
            calls.append(a)
            return orig(*a, **k)
        mm.free_memory = spy
        try:
            mm.load_models_gpu([target])
            mm.load_models_gpu([target])
        finally:
            mm.free_memory = orig
        loaded = [lm.model for lm in mm.current_loaded_models]
        self.assertTrue(any(m is other for m in loaded))
        self.assertTrue(any(m is target for m in loaded))
        self.assertEqual(calls, [])
        self.assertTrue(all(p.device.type == "meta" for p in target.model.parameters()))
        self.assertEqual(target.model.current_weight_patches_uuid, target.patches_uuid)
        self.assertIsNotNone(target.gmlx_table_id)
        self.assertEqual(len(self.stub.events("put_table")), 1)

    def test_clone_keeps_the_class_through_stock_nodes(self):
        base = self.model()
        lora = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(base, LORA, 0.8)[0]
        aura = ModelSamplingAuraFlow().patch_aura(lora, 6.0)[0]
        for m in (base.clone(), lora, aura, base.clone(disable_dynamic=True)):
            self.assertIsInstance(m, P.patcher.GmlxModelPatcher)
            self.assertIs(m.gmlx, base.gmlx)


class Capture(Base):
    """Patch capture from the stock LoRA nodes, and tables that follow content, not uuids."""

    def test_stock_lora_patches_become_a_banded_table(self):
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.8)[0]
        offsets = {t[3] for v in m.patches.values() for t in v}
        self.assertIn((0, 0, 3840), offsets)
        self.assertIn((0, 7680, 3840), offsets)
        tid = m.sync_patches()
        table, _ = P.patches.build_table(ZID, m.patches, family="z_image")
        self.assertEqual(tid, C.table_id(table)[0])
        (put,) = self.stub.events("put_table")
        self.assertEqual(put["id"], tid)
        self.assertEqual(put["strengths"], [0.8])
        self.assertEqual(put["bands"], 2 * 3)
        self.assertEqual(sum(len(e["patches"]) for e in table["entries"]), LORA_TARGETS)
        qkv = [e for e in table["entries"] if e["key"].endswith("layers.0.attention.qkv.weight")]
        self.assertEqual(sorted(e["band"][1] for e in qkv), [0, 3840, 7680])
        self.assertEqual(len(self.stub.events("put_adapter")), 2 * LORA_TARGETS)

    def test_patches_made_under_inference_mode_sync_and_sample(self):
        # ComfyUI runs every node under torch.inference_mode, so LoRA tensors
        # are inference tensors there.
        with torch.inference_mode():
            m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.9)[0]
            tid = m.sync_patches()
            self.sample(m)
        self.assertIsNotNone(tid)
        self.assertEqual(len(self.stub.events("put_table")), 1)
        self.assertEqual(len(self.stub.events("step")), STEPS)

    def test_patches_uuid_and_table_follow_content(self):
        LL = comfy_nodes.LoraLoaderModelOnly()
        base = self.model()
        base.sync_patches()
        puts = lambda: len(self.stub.events("put_table"))  # noqa: E731
        seq = []

        def step(name, child, content_changed):
            before = puts()
            child.sync_patches()
            seq.append((name, content_changed, puts() - before, child.gmlx_table_id))
            return child
        c1 = step("clone", base.clone(), False)
        m08 = step("lora 0.8", LL.load_lora_model_only(c1, LORA, 0.8)[0], True)
        again = step("lora 0.8 again", LL.load_lora_model_only(c1, LORA, 0.8)[0], True)
        step("auraflow", ModelSamplingAuraFlow().patch_aura(m08, 6.0)[0], False)
        stacked = step("stacked 0.5", LL.load_lora_model_only(m08, LORA, 0.5)[0], True)
        m05 = step("lora 0.5", LL.load_lora_model_only(c1, LORA, 0.5)[0], True)
        cd = m05.clone()
        cd.set_model_compute_dtype(torch.float32)
        step("compute dtype", cd, False)
        by = {s[0]: s for s in seq}
        self.assertIsNone(by["clone"][3])
        self.assertEqual(by["lora 0.8"][2], 1)
        self.assertEqual(by["lora 0.8 again"][2], 0)
        self.assertEqual(again.gmlx_table_id, m08.gmlx_table_id)
        self.assertEqual(by["auraflow"][2], 0)
        self.assertEqual(by["stacked 0.5"][2], 1)
        self.assertEqual(by["lora 0.5"][2], 1)
        self.assertEqual(by["compute dtype"][2], 0)
        self.assertEqual(by["compute dtype"][3], m05.gmlx_table_id)
        table, _ = P.patches.build_table(ZID, stacked.patches, family="z_image")
        self.assertEqual({len(e["patches"]) for e in table["entries"]}, {2})
        self.assertEqual(len({s[3] for s in seq if s[3]}), 3)

    @needs_flux3
    def test_a_flux3_lora_on_native_keys_becomes_a_table(self):
        LL = comfy_nodes.LoraLoaderModelOnly()
        m = LL.load_lora_model_only(self.model(F3ID), LORA_F3, 0.8)[0]
        self.assertEqual(set(m.patches), LORA_F3_KEYS)
        self.assertEqual({t[3] for v in m.patches.values() for t in v}, {None})
        tid = m.sync_patches()
        table, _ = P.patches.build_table(F3ID, m.patches, family="flux3_klein")
        self.assertEqual(tid, C.table_id(table)[0])
        (put,) = self.stub.events("put_table")
        self.assertEqual((put["id"], put["entries"], put["bands"], put["strengths"]),
                         (tid, len(LORA_F3_KEYS), 0, [0.8]))
        self.assertEqual(len(self.stub.events("put_adapter")), 2 * len(LORA_F3_KEYS))
        # Base has no target time embedder, so ComfyUI's key map drops that pair.
        b = LL.load_lora_model_only(self.model(F3BID), LORA_F3, 0.8)[0]
        self.assertEqual(set(b.patches), LORA_F3_KEYS - {"diffusion_model.ref_time_in.in_layer.weight"})

    def test_a_flux2_lora_in_three_spellings_becomes_a_table(self):
        """ComfyUI's FLUX.2 key map puts diffusers q, k and v on thirds of the
        fused weight and norm_out.linear on the final modulation with the
        halves of its product swapped, and the kohya and native names on their
        layers."""
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(F2ID), LORA_F2, 0.8)[0]
        self.assertEqual(set(m.patches), LORA_F2_KEYS)
        qkv = m.patches["diffusion_model.double_blocks.0.img_attn.qkv.weight"]
        self.assertEqual(sorted(t[3] for t in qkv), [(0, 0, 3072), (0, 3072, 3072), (0, 6144, 3072)])
        tid = m.sync_patches()
        table, tensors = P.patches.build_table(F2ID, m.patches, family="flux2_klein")
        self.assertEqual(tid, C.table_id(table)[0])
        (put,) = self.stub.events("put_table")
        self.assertEqual((put["id"], put["entries"], put["bands"], put["strengths"]),
                         (tid, len(support.FLUX2_PAIRS), 4, [0.8]))
        self.assertEqual(len(self.stub.events("put_adapter")), 2 * len(support.FLUX2_PAIRS))
        sd = comfy.utils.load_torch_file(os.path.join(LORAS, LORA_F2), safe_load=True)
        up = sd["transformer.norm_out.linear.lora_up.weight"]
        (e,) = [e for e in table["entries"]
                if e["key"] == "diffusion_model.final_layer.adaLN_modulation.1.weight"]
        (pp,) = e["patches"]
        self.assertTrue(torch.equal(tensors[pp["up"]], torch.cat([up[3072:], up[:3072]])))
        self.assertTrue(torch.equal(tensors[pp["down"]], sd["transformer.norm_out.linear.lora_down.weight"]))

    def test_a_dropped_adapter_is_uploaded_again_on_409(self):
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.7)[0]
        # The client believes the server holds every pair; the server holds none.
        self.client.known_adapters.update(P.patches.build_table(ZID, m.patches, family="z_image")[1])
        m.sync_patches()
        self.assertEqual(len(self.stub.events("table_missing")), 1)
        self.assertEqual(len(self.stub.events("put_table")), 1)
        self.assertEqual(len(self.stub.events("put_adapter")), 2 * LORA_TARGETS)


class Refusals(Base):
    """Refusals by name, at registration and at forward time."""

    def test_registrations_that_run_inside_the_forward_raise_by_name(self):
        m = self.model()
        cases = {
            "the model patch": lambda: m.set_model_patch(lambda *a: a, "double_block"),
            "the model patch replace": lambda: m.set_model_patch_replace(lambda *a: a, "dit", "single_block", 0),
            "RoPE": lambda: m.set_model_rope_options(1, 0, 1, 0, 1, 0),
            "attention override": lambda: m.set_model_optimized_attention(lambda *a: a),
            "weight wrapper": lambda: m.add_weight_wrapper("diffusion_model.x", lambda w: w),
            "object patch on diffusion_model": lambda: m.add_object_patch("diffusion_model", torch.nn.Identity()),
            "DIFFUSION_MODEL wrapper": lambda: m.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "k", lambda *a: a),
            "hook patches": lambda: m.add_hook_patches(None, {}),
        }
        for name, fn in cases.items():
            with self.subTest(name):
                with self.assertRaises(C.GmlxRefusal) as e:
                    fn()
                self.assertIn(name, str(e.exception))
        m.add_object_patch("model_sampling", m.get_model_object("model_sampling"))

    def test_patch_kinds_version_1_does_not_apply_are_refused_by_name(self):
        base = self.model()
        key = "diffusion_model.layers.0.attention.out.weight"
        up, down = torch.zeros(3840, 4), torch.zeros(4, 3840)
        LoRA = comfy.weight_adapter.LoRAAdapter
        kinds = {
            "a LoRA with a mid tensor": (LoRA(set(), (up, down, 1.0, torch.zeros(4, 4, 1, 1), None, None)), 1.0, None),
            "DoRA": (LoRA(set(), (up, down, 1.0, None, torch.ones(3840), None)), 1.0, None),
            "a LoRA with a reshape": (LoRA(set(), (up, down, 1.0, None, None, (3840, 3840))), 1.0, None),
            "a 'diff' patch": (("diff", (torch.zeros(3840, 3840),)), 1.0, None),
            "a 'set' patch": (("set", (torch.zeros(3840, 3840),)), 1.0, None),
            "a 'model_as_lora' patch": (("model_as_lora", (torch.zeros(3840, 3840),)), 1.0, None),
            "a merge list": ([(1.0, torch.zeros(1))], 1.0, None),
            "a model strength of 0.5": (LoRA(set(), (up, down, 1.0, None, None, None)), 0.5, None),
            "a patch function": (LoRA(set(), (up, down, 1.0, None, None, None)), 1.0, lambda w: w),
        }
        for cls_name, label in (("LoHaAdapter", "LoHa"), ("LoKrAdapter", "LoKr"), ("GLoRAAdapter", "GLoRA"),
                                ("OFTAdapter", "OFT"), ("BOFTAdapter", "BOFT")):
            kinds[label] = (getattr(comfy.weight_adapter, cls_name)(set(), ()), 1.0, None)
        for name, (adapter, sm, fn) in kinds.items():
            with self.subTest(name):
                m = base.clone()
                m.patches = {key: [(1.0, adapter, sm, None, fn)]}
                m.patches_uuid = uuid.uuid4()
                with self.assertRaises(C.GmlxRefusal) as e:
                    m.sync_patches()
                self.assertIn(name, str(e.exception))
                self.assertIn(key, str(e.exception))

    def test_a_swap_of_the_halves_is_refused_on_a_family_not_verified_for_it(self):
        """ComfyUI's swap_scale_shift on a patch is taken on FLUX.2 klein
        alone, whose LoRA test checks the order, and refused by name on
        every other family."""
        LoRA = comfy.weight_adapter.LoRAAdapter
        up, down = torch.zeros(64, 4), torch.zeros(4, 32)
        patches = {"diffusion_model.final_layer.adaLN_modulation.1.weight": [
            (1.0, LoRA(set(), (up, down, 1.0, None, None, None)), 1.0, None,
             comfy.utils.swap_scale_shift)]}
        self.assertEqual(P.patches.SWAPS_HALVES, {"flux2_klein", "flux2_klein_9b",
                                                  "flux2_klein_9b_kv"})
        for family in sorted(P.patches.SWAPS_HALVES):
            table, tensors = P.patches.build_table(F2ID, patches, family=family)
            (e,) = table["entries"]
            self.assertTrue(torch.equal(tensors[e["patches"][0]["up"]], torch.cat([up[32:], up[:32]])))
        for family in ("z_image", "qwen_image21", "krea2", "flux3_klein"):
            with self.subTest(family), self.assertRaises(C.GmlxRefusal) as err:
                P.patches.build_table("m", patches, family=family)
            self.assertIn("swap_scale_shift", str(err.exception))
            self.assertIn(family, str(err.exception))
        m = self.model().clone()
        m.patches, m.patches_uuid = patches, uuid.uuid4()
        with self.assertRaises(C.GmlxRefusal) as err:
            m.sync_patches()
        self.assertIn("not on z_image", str(err.exception))

    def forward(self, m, **kw):
        cond = self.cond(ZID, "x")[0][0]
        f = m.model.diffusion_model.forward
        args = dict(context=cond, transformer_options={"cond_or_uncond": [0]}, num_tokens=cond.shape[1])
        args.update(kw)
        return f(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]), **args)

    def test_forward_time_refusals_name_what_reached_the_forward(self):
        m = self.model()
        m.model.current_patcher = m
        try:
            cases = {"ControlNet": dict(control={"output": []}),
                     "transformer_options['patches_replace']":
                         dict(transformer_options={"patches_replace": {"dit": {("double_block", 0): 1}}}),
                     "transformer_options['patches']": dict(transformer_options={"patches": {"post_input": [1]}}),
                     "transformer_options['rope_options']": dict(transformer_options={"rope_options": {"scale_x": 2}}),
                     "DIFFUSION_MODEL wrapper": dict(transformer_options={"wrappers": {
                         comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL: {"k": [lambda *a: a]}}}),
                     "'siglip_feats'": dict(siglip_feats=[torch.zeros(1, 2, 2, 8)]),
                     "the DiT input 'guidance'": dict(guidance=torch.tensor([3.5])),
                     "reference latents": dict(ref_latents=[torch.zeros(1, 16, 4, 4)])}
            for name, kw in cases.items():
                with self.subTest(name):
                    with self.assertRaises(C.GmlxRefusal) as e:
                        self.forward(m, **kw)
                    self.assertIn(name, str(e.exception))
            self.assertEqual(self.stub.events("step"), [])
        finally:
            m.model.current_patcher = None

    def test_a_flux2_forward_reads_no_guidance_and_refuses_another_reference_method(self):
        """ComfyUI hands a FLUX.2 forward the guidance on every call, which its
        DiT reads only through a guidance embedder klein has none of. The
        server places the references by the DiT's own index rule, so another
        method, which FluxKontextMultiReferenceLatentMethod sets, is refused by
        name."""
        f = self.model(F2ID).model.diffusion_model.forward
        cond = self.cond(F2ID, "x")[0][0]
        x, t, g = torch.zeros(1, 128, 4, 4), torch.tensor([1.0]), torch.tensor([3.5])
        ref = torch.zeros(1, 128, 2, 2)
        f(x, t, context=cond, guidance=g)
        f(x, t, context=cond, guidance=g, ref_latents=[ref], ref_latents_method="index")
        self.assertEqual([s["refs"] for s in self.stub.events("step")], [[0], [1]])
        for method in ("offset", "uxo", "index_timestep_zero"):
            with self.subTest(method):
                with self.assertRaises(C.GmlxRefusal) as e:
                    f(x, t, context=cond, guidance=g, ref_latents=[ref], ref_latents_method=method)
                self.assertIn(f"{F2ID!r} places its reference latents by the 'index' method, and "
                              f"this call asks for {method!r}", str(e.exception))
        # Without references the stock DiT reads no method either.
        f(x, t, context=cond, guidance=g, ref_latents_method="offset")
        self.assertEqual(len(self.stub.events("step")), 3)

    @needs_flux3
    def test_flux3_refusals_name_what_reached_the_forward(self):
        m = self.model(F3ID)
        f = m.model.diffusion_model.forward
        cond = self.cond(F3ID, "x")[0][0]
        ss = torch.tensor([1.0, 0.5, 0.0])
        cases = {"the DiT input 'txt_states'": (C.GmlxRefusal, dict(txt_states=torch.zeros(1, 32, 3072))),
                 "the DiT input 'ref_states'": (C.GmlxRefusal, dict(ref_states=torch.zeros(1, 16, 3072))),
                 "transformer_options['patches_replace']": (C.GmlxRefusal, dict(transformer_options={
                     "sample_sigmas": ss, "patches_replace": {"dit": {("single_block", 0): 1}}})),
                 "transformer_options['sample_sigmas']": (C.GmlxError, dict(transformer_options={}))}
        for name, (cls, kw) in cases.items():
            with self.subTest(name):
                args = {"context": cond, "transformer_options": {"sample_sigmas": ss}, **kw}
                with self.assertRaises(cls) as e:
                    f(torch.zeros(1, 128, 4, 4), torch.tensor([1.0]), **args)
                self.assertIn(name, str(e.exception))
        self.assertEqual(self.stub.events("step"), [])
        # The DiT's conditioning blocks never run in ComfyUI.
        with self.assertRaises(C.GmlxRefusal) as e:
            m.model.diffusion_model.preprocess_conds(cond)
        self.assertIn("conditioning blocks of the Flux3 DiT", str(e.exception))

    def test_unknown_options_warn_once_and_the_cache_maps(self):
        m = self.model(QID)
        cond = self.cond(QID, "x")[0][0]
        f = m.model.diffusion_model.forward
        x = torch.zeros(1, 64, 4, 4)
        with self.assertLogs("gmlx", level=logging.WARNING) as logs:
            f(x, torch.tensor([0.5]), context=cond, transformer_options={"brand_new_key": 1})
            f(x, torch.tensor([0.5]), context=cond, transformer_options={"brand_new_key": 1})
        self.assertEqual(sum("brand_new_key" in r for r in logs.output), 1)
        f(x, torch.tensor([0.5]), context=cond, transformer_options={"qwen_image21_cache": {"device": "off"}})
        self.assertEqual(self.stub.events("step")[-1]["prefix_cache"], [False])
        with self.assertRaises(C.GmlxRefusal) as e:
            f(x, torch.tensor([0.5]), context=cond,
              transformer_options={"qwen_image21_cache": {"device": "auto", "dtype": "int8"}})
        self.assertIn("int8", str(e.exception))

    def test_pag_is_caught_at_forward_time_in_a_sampler(self):
        from comfy_extras.nodes_pag import PerturbedAttentionGuidance
        m = PerturbedAttentionGuidance.execute(self.model(), 3.0)[0]
        with self.assertRaises(C.GmlxRefusal) as e:
            self.sample(m)
        self.assertIn("patches_replace", str(e.exception))


class Sampling(Base):
    """The sampler reaching the forward, model sampling patches, the 409 path in a step
    and the variant sigmas."""

    def test_ksampler_reaches_the_forward_with_rows_batched(self):
        for mid in (ZID, QID):
            with self.subTest(mid):
                self.stub.reset()
                m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.8)[0] \
                    if mid == ZID else self.model(mid)
                out = self.sample(m, mid)
                steps = self.stub.events("step")
                self.assertEqual(len(steps), STEPS)
                self.assertEqual({s["rows"] for s in steps}, {2})
                self.assertTrue(all(s["convention"] == "comfy" for s in steps))
                self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens")})
                self.assertEqual({s["x_dtype"] for s in steps}, {"F32"})
                self.assertTrue(bool(torch.isfinite(out).all()))
                kinds = [e["kind"] for e in self.stub.log() if e["kind"] in ("run_begin", "run_end")]
                self.assertEqual(kinds, ["run_begin", "run_end"])
                self.assertEqual(len({s["run"] for s in steps}), 1)
                tables = {s["patch_table"] for s in steps}
                self.assertEqual(len(tables), 1)
                self.assertEqual(tables.pop() is None, mid != ZID)
                self.assertEqual(len(self.stub.events("put_tensor")), 0)

    def test_krea2_latents_with_a_frame_axis_reach_the_forward(self):
        """ComfyUI gives Krea 2 latents (B, C, 1, h, w), as for the Wan 2.1 latent
        format. The step sends (B, C, h, w) and the sampler gets the frame axis back."""
        out = self.sample(self.model(KID), KID, cfg=1.0)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), STEPS)
        self.assertEqual({tuple(s["x_shape"]) for s in steps}, {(1, 16, 8, 8)})
        self.assertEqual(tuple(out.shape), (1, 16, 1, 8, 8))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_a_frame_axis_of_more_than_one_frame_is_refused(self):
        fwd = self.model(KID).model.diffusion_model.forward
        with self.assertRaises(C.GmlxError) as e:
            fwd(torch.zeros(1, 16, 2, 8, 8), torch.ones(1), context=torch.zeros(1, 4, 30720))
        self.assertIn("(B, C, h, w)", str(e.exception))

    def test_a_flux2_sample_sends_its_512_text_rows_and_no_schedule(self):
        """ComfyUI's Flux2 model pads the text rows at the front to 512 and
        hands the forward a guidance the step does not carry. The DiT is no
        flow map, so no schedule goes."""
        for mid, cfg, rows in ((F2ID, 1.0, 1), (F2BID, 4.0, 2)):
            with self.subTest(mid):
                self.stub.reset()
                with torch.inference_mode():
                    out = self.sample(self.model(mid), mid, cfg=cfg)
                steps = self.stub.events("step")
                self.assertEqual(len(steps), STEPS)
                self.assertEqual({s["rows"] for s in steps}, {rows})
                self.assertEqual({tuple(s["tensors"]) for s in steps}, {("timestep", "x")})
                self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens")})
                self.assertEqual({tuple(s["num_tokens"]) for s in steps}, {(512,) * rows})
                self.assertEqual({tuple(s["x_shape"]) for s in steps}, {(rows, 128, 4, 4)})
                self.assertEqual(tuple(out.shape), (1, 128, 4, 4))
                self.assertTrue(bool(torch.isfinite(out).all()))

    def test_a_kv_edit_under_the_flux_kv_cache_node_steps_on_the_runs_cache(self):
        """ComfyUI's Flux KV Cache node on a gmlx model patches nothing in
        ComfyUI: each step of an edit under it asks the server for the run's
        cache of reference K and V, under the reference method the node sets,
        and an edit without the node does not."""
        from comfy_extras.nodes_flux import FluxKVCache
        import node_helpers

        m = self.model(F29KID)
        (kv,) = FluxKVCache.execute(m).result
        self.assertNotIn("patches", kv.model_options["transformer_options"])
        ref = torch.zeros(1, 128, 2, 2)
        pos = node_helpers.conditioning_set_values(
            self.cond(F29KID, "a red bicycle"), {"reference_latents": [ref]}, append=True)
        for model, cached in ((kv, True), (m, False)):
            with self.subTest(cached=cached):
                self.stub.reset()
                with torch.inference_mode():
                    out = self.sample(model, F29KID, cfg=1.0, pos=pos)
                steps = self.stub.events("step")
                self.assertEqual(len(steps), STEPS)
                self.assertEqual({s["ref_cache"] for s in steps}, {cached})
                self.assertEqual({tuple(s["refs"]) for s in steps}, {(1,)})
                self.assertEqual(len({s["run"] for s in steps}), 1)
                self.assertTrue(bool(torch.isfinite(out).all()))

    @needs_flux3
    def test_a_flux3_flow_map_step_carries_the_sampler_schedule(self):
        """Each step holds ComfyUI's sample_sigmas, so the server's target is
        the entry after the step's timestep, as ComfyUI's FLUX.3 forward takes it."""
        m = self.model(F3ID)
        with torch.inference_mode():
            out = self.sample(m, F3ID, cfg=1.0)
        sig = [float(v) for v in comfy.samplers.calculate_sigmas(
            m.get_model_object("model_sampling"), "simple", STEPS)]
        steps = self.stub.events("step")
        self.assertEqual(len(steps), STEPS)
        for i, s in enumerate(steps):
            self.assertEqual(s["sample_sigmas"], sig)
            self.assertAlmostEqual(s["timestep"][0], sig[i], places=6)
            self.assertEqual(s["targets"], [sig[i + 1]])
        self.assertEqual({tuple(s["tensors"]) for s in steps}, {("sample_sigmas", "timestep", "x")})
        self.assertEqual({tuple(s["x_shape"]) for s in steps}, {(1, 128, 4, 4)})
        self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens")})
        self.assertEqual(tuple(out.shape), (1, 128, 4, 4))
        self.assertTrue(bool(torch.isfinite(out).all()))

    @needs_flux3
    def test_a_flux3_base_sample_under_inference_mode_sends_its_conditioning(self):
        """ComfyUI runs the conditioning blocks of a FLUX.3 DiT that is no flow
        map once in extra_conds under inference mode. On a gmlx model the step
        takes the conditioning and runs them on the server, and sends no schedule."""
        m = self.model(F3BID)
        with torch.inference_mode():
            out = self.sample(m, F3BID, cfg=3.5)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), STEPS)
        self.assertEqual({s["rows"] for s in steps}, {2})
        self.assertEqual({tuple(s["tensors"]) for s in steps}, {("timestep", "x")})
        self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens")})
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertIs(m.model.diffusion_model.flow_map, False)
        kw = dict(cross_attn=torch.zeros(1, 32, 20480), device=torch.device("cpu"),
                  noise=torch.zeros(1, 128, 4, 4))
        with torch.inference_mode():
            self.assertEqual(set(m.model.extra_conds(**kw)), {"c_crossattn"})
            # ComfyUI's own extra_conds would run the blocks; the call is refused by name.
            with self.assertRaises(C.GmlxRefusal) as e:
                type(m.model).extra_conds(m.model, **kw)
        self.assertIn("conditioning blocks of the Flux3 DiT", str(e.exception))

    def test_unequal_token_counts_send_single_rows(self):
        self.sample(self.model(), neg_tokens=17)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), 2 * STEPS)
        self.assertEqual({s["rows"] for s in steps}, {1})
        self.assertEqual(sorted({s["num_tokens"][0] for s in steps}), [17, 32])

    def test_a_padded_stock_conditioning_on_zimage_sends_every_row(self):
        pos = self.cond(ZID, "a red bicycle in the rain")
        mask = torch.ones(1, pos[0][0].shape[1])
        mask[:, -5:] = 0
        pos = [[pos[0][0], {"attention_mask": mask}]]
        self.sample(self.model(), pos=pos)
        steps = self.stub.events("step")
        self.assertTrue(steps)
        # The stock DiT reads neither the mask nor its count, so all 32 rows go.
        self.assertEqual({s["num_tokens"][0] for s in steps}, {32})
        self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens")})

    def test_auraflow_changes_the_timesteps_sent(self):
        m = self.model()
        aura = ModelSamplingAuraFlow().patch_aura(m, 6.0)[0]

        def expected(model):
            ms = model.get_model_object("model_sampling")
            s = comfy.samplers.calculate_sigmas(ms, "simple", STEPS)
            return [float(ms.timestep(v)) for v in s[:-1]]
        got = {}
        for name, mod in (("default", m), ("aura", aura)):
            self.stub.reset()
            self.sample(mod, seed=3)
            got[name] = [s["timestep"][0] for s in self.stub.events("step")]
            np.testing.assert_allclose(got[name], expected(mod), atol=1e-6)
        self.assertNotEqual(got["default"], got["aura"])

    def test_a_step_after_the_server_dropped_its_tensors_uploads_again(self):
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.6)[0]
        mm.load_models_gpu([m])
        m.pre_run()
        try:
            f = m.model.diffusion_model.forward
            cond = self.cond(ZID, "z")[0][0]
            to = {"cond_or_uncond": [0], "uuids": [uuid.uuid4()]}
            f(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]), context=cond, transformer_options=to)
            self.stub.forget()
            self.stub.reset()
            out = f(torch.zeros(1, 16, 4, 4), torch.tensor([0.4]), context=cond, transformer_options=to)
            self.assertEqual(tuple(out.shape), (1, 16, 4, 4))
        finally:
            m.cleanup()
        kinds = [e["kind"] for e in self.stub.log()]
        self.assertEqual(kinds[0], "step_missing")
        self.assertEqual(kinds.count("put_tensor"), 1)
        self.assertEqual(kinds.count("put_table"), 1)
        self.assertEqual(kinds.count("put_adapter"), 2 * LORA_TARGETS)
        self.assertEqual(kinds.count("step"), 1)

    def test_variant_sigmas(self):
        turbo = self.model(QTID)
        sig, cfg = P.nodes.GmlxSigmas().get_sigmas(turbo)
        self.assertEqual([round(float(s), 6) for s in sig],
                         [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0])
        self.assertEqual(cfg, 1.0)
        with self.assertRaises(C.GmlxRefusal):
            P.nodes.GmlxSigmas().get_sigmas(turbo, steps=20)
        self.assertEqual(len(P.nodes.GmlxSigmas().get_sigmas(turbo, 0, 0.5)[0]), 5)
        # Qwen-Image 2.1's schedule follows the image size, and ends on its
        # terminal sigma before the zero.
        base = self.model(QID)
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxSigmas().get_sigmas(base)
        self.assertIn("width and height", str(e.exception))
        self.assertNotIn("Flux2Scheduler", str(e.exception))
        sig, _ = P.nodes.GmlxSigmas().get_sigmas(base, 0, 1.0, 1024, 1024)
        self.assertEqual(len(sig), 41)
        self.assertAlmostEqual(float(sig[-2]), 0.02, places=6)
        self.assertEqual(float(sig[-1]), 0.0)

    def test_variant_sigmas_of_flux2_follow_the_image_size(self):
        model = self.model(F2ID)
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxSigmas().get_sigmas(model)
        self.assertIn("Flux2Scheduler", str(e.exception))
        small, _ = P.nodes.GmlxSigmas().get_sigmas(model, 0, 1.0, 512, 512)
        large, _ = P.nodes.GmlxSigmas().get_sigmas(model, 0, 1.0, 1024, 1024)
        self.assertEqual((len(small), len(large)), (5, 5))
        self.assertEqual((float(small[-1]), float(large[-1])), (0.0, 0.0))
        # A larger image shifts the schedule up.
        self.assertTrue(all(float(a) < float(b) for a, b in zip(small[1:-1], large[1:-1])))

    def test_sampler_custom_samples_on_the_fixed_sigmas(self):
        from comfy_extras.nodes_custom_sampler import SamplerCustom
        turbo = self.model(QTID)
        sig, _ = P.nodes.GmlxSigmas().get_sigmas(turbo)
        pos, neg = self.cond(QTID, "p"), self.cond(QTID, "n")
        sampler = comfy.samplers.sampler_object("euler")
        latent = {"samples": torch.zeros(1, 64, 4, 4)}
        out = SamplerCustom.execute(turbo, True, 5, 1.0, pos, neg, sampler, sig, latent)
        self.assertEqual(tuple(out[0]["samples"].shape), (1, 64, 4, 4))
        np.testing.assert_allclose([s["timestep"][0] for s in self.stub.events("step")],
                                   [float(s) for s in sig[:-1]], atol=1e-6)


class Interrupt(Base):
    """Interrupt in process: the flag reaches the server
    within one poll, and the run ends."""
    DELAY_S = 1.0
    stub_args = ("--step-delay-ms", str(int(DELAY_S * 1000)))

    def test_interrupt_reaches_the_server_and_stops_the_run(self):
        m = self.model()
        done = threading.Event()

        def watch():
            while not done.is_set():
                # Interrupt once the third step is under way on the server,
                # not between steps, where ComfyUI's own check stops the
                # run first.
                if len(self.stub.events("step_begin")) >= 3:
                    mm.interrupt_current_processing(True)
                    return
                time.sleep(0.02)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        t0 = time.monotonic()
        with self.assertRaises(mm.InterruptProcessingException):
            self.sample(m, steps=8)
        done.set()
        took = time.monotonic() - t0
        time.sleep(0.3)
        kinds = [e["kind"] for e in self.stub.log()]
        self.assertGreaterEqual(kinds.count("interrupt"), 1)
        self.assertIn("run_end", kinds)
        self.assertLess(kinds.count("step"), 8)
        self.assertGreaterEqual(kinds.count("step_interrupted"), 1)
        self.assertLess(took, 8 * self.DELAY_S)


class ClipAndVae(Base):
    def test_clip_text_encode_gives_real_conditioning(self):
        clip = P.nodes.GmlxCLIPLoader().load(ZID, self.stub.url, False)[0]
        (cond,) = comfy_nodes.CLIPTextEncode().encode(clip, "a lighthouse")
        self.assertEqual(tuple(cond[0][0].shape), (1, 32, 2560))
        self.assertEqual(cond[0][1].get("pooled_output", None), None)
        self.assertEqual(self.stub.events("encode")[0]["text"], "a lighthouse")
        self.assertIsInstance(clip.clone(), P.clip.GmlxCLIP)

    def test_an_empty_prompt_that_must_not_be_empty_encodes_a_space(self):
        clip = P.nodes.GmlxCLIPLoader().load(QID, self.stub.url, False)[0]
        clip.encode_from_tokens_scheduled(clip.tokenize("", prevent_empty_text=True))
        clip.encode_from_tokens_scheduled(clip.tokenize(""))
        self.assertEqual([e["text"] for e in self.stub.events("encode")], [" ", ""])

    def test_a_z_image_clip_keeps_an_empty_prompt_and_a_qwen_clip_sends_a_space(self):
        z = P.nodes.GmlxCLIPLoader().load(ZID, self.stub.url, False)[0]
        q = P.nodes.GmlxCLIPLoader().load(QID, self.stub.url, False)[0].clone()
        k = P.nodes.GmlxCLIPLoader().load(KID, self.stub.url, False)[0]
        z.encode_from_tokens_scheduled(z.tokenize("", prevent_empty_text=True))
        q.encode_from_tokens_scheduled(q.tokenize("", prevent_empty_text=True))
        q.encode_from_tokens_scheduled(q.tokenize("  ", prevent_empty_text=True))
        # Krea2Tokenizer passes the option to Qwen3VLTokenizer.
        k.encode_from_tokens_scheduled(k.tokenize("", prevent_empty_text=True))
        self.assertEqual([(e["model"], e["text"]) for e in self.stub.events("encode")],
                         [(ZID, ""), (QID, " "), (QID, "  "), (KID, " ")])

    def test_clip_refusals_by_name(self):
        clip = P.nodes.GmlxCLIPLoader().load(QID, self.stub.url, False)[0]
        with self.assertRaises(C.GmlxRefusal) as e:
            comfy_nodes.CLIPSetLastLayer().set_last_layer(clip, -2)
        self.assertIn("CLIP layer", str(e.exception))
        img = torch.rand(1, 32, 32, 3)
        with self.assertRaises(C.GmlxRefusal) as e:
            clip.encode_from_tokens_scheduled(clip.tokenize("x", images=[img], keep_vision=True))
        self.assertIn("connect a VAE", str(e.exception))
        with self.assertRaises(C.GmlxRefusal) as e:
            clip.encode_from_tokens_scheduled(clip.tokenize("x", llama_template="{}"))
        self.assertIn("template", str(e.exception))

    def test_stock_lora_loader_with_the_gmlx_clip(self):
        model = self.model()
        clip = P.nodes.GmlxCLIPLoader().load(ZID, self.stub.url, False)[0]
        with self.assertLogs("gmlx", level=logging.WARNING) as logs:
            m2, c2 = comfy_nodes.LoraLoader().load_lora(model, clip, LORA, 0.8, 1.0)
        self.assertTrue(any("text encoder" in r for r in logs.output))
        self.assertIsInstance(m2, P.patcher.GmlxModelPatcher)
        self.assertIsInstance(c2, P.clip.GmlxCLIP)
        self.assertEqual(len(m2.patches), LORA_TARGETS - 2 * 2)

    def test_qwen_edit_with_two_references_reaches_the_step(self):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        clip = P.nodes.GmlxCLIPLoader().load(QID, self.stub.url, False)[0]
        vae = P.nodes.GmlxVAELoader().load(QID, self.stub.url, False)[0]
        ims = {"image_1": torch.rand(1, 96, 64, 3), "image_2": torch.rand(1, 64, 64, 4)}
        out = TextEncodeQwenImage21.execute(clip, "make it blue", "", vae, 0, ims)
        pos, neg, latent = out[0], out[1], out[2]
        self.assertEqual(len(pos[0][1]["reference_latents"]), 2)
        self.assertEqual(pos[0][1]["image_slots"], [2, 5])
        self.assertEqual([e["images"] for e in self.stub.events("encode")], [2, 2])
        self.assertEqual(len(self.stub.events("vae_encode")), 2)
        self.stub.reset()
        model = self.model(QID)
        (res,) = comfy_nodes.common_ksampler(model, 1, 2, 2.0, "euler", "simple", pos, neg, latent)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), 2)
        self.assertEqual({tuple(s["refs"]) for s in steps}, {(2, 2)})
        self.assertEqual({str(s["image_slots"]) for s in steps}, {str([[2, 5], [2, 5]])})
        self.assertEqual(list(latent["samples"].shape), [1, 64, 6, 4])
        self.assertEqual(steps[0]["x_shape"], [2, 64, 6, 4])
        self.assertEqual(tuple(res["samples"].shape), (1, 64, 6, 4))

    def test_flux2_clip_and_vae_carry_its_features_and_model_space_latents(self):
        clip = P.nodes.GmlxCLIPLoader().load(F2ID, self.stub.url, False)[0]
        (cond,) = comfy_nodes.CLIPTextEncode().encode(clip, "a lighthouse")
        self.assertEqual(tuple(cond[0][0].shape), (1, 32, 7680))
        # KleinTokenizer applies its chat template before prevent_empty_text
        # reads the prompt, so an empty prompt goes as it is.
        clip.encode_from_tokens_scheduled(clip.tokenize("", prevent_empty_text=True))
        self.assertEqual([e["text"] for e in self.stub.events("encode")], ["a lighthouse", ""])
        vae = P.nodes.GmlxVAELoader().load(F2ID, self.stub.url, False)[0]
        (img,) = comfy_nodes.VAEDecode().decode(vae, {"samples": torch.zeros(1, 128, 4, 6)})
        self.assertEqual(tuple(img.shape), (1, 64, 96, 3))
        (lat,) = comfy_nodes.VAEEncode().encode(vae, torch.rand(1, 70, 100, 3))
        self.assertEqual(tuple(lat["samples"].shape), (1, 128, 4, 6))
        self.assertEqual(self.stub.events("vae_encode")[-1]["shape"], [1, 64, 96, 3])

    def test_a_flux2_edit_through_reference_latent_reaches_the_step(self):
        """ComfyUI's FLUX.2 klein edit template: the text encode, its zeroed
        negative, and the reference's VAEEncode put on both by
        ReferenceLatent. FluxKontextMultiReferenceLatentMethod at the DiT's
        own rule passes, and another method is refused by name."""
        from comfy_extras.nodes_edit_model import ReferenceLatent
        from comfy_extras.nodes_flux import FluxKontextMultiReferenceLatentMethod
        clip = P.nodes.GmlxCLIPLoader().load(F2ID, self.stub.url, False)[0]
        vae = P.nodes.GmlxVAELoader().load(F2ID, self.stub.url, False)[0]
        (pos,) = comfy_nodes.CLIPTextEncode().encode(clip, "make it blue")
        (neg,) = comfy_nodes.ConditioningZeroOut().zero_out(pos)
        (lat,) = comfy_nodes.VAEEncode().encode(vae, torch.rand(1, 96, 64, 3))
        pos, neg = ReferenceLatent.execute(pos, lat)[0], ReferenceLatent.execute(neg, lat)[0]
        latent = {"samples": torch.zeros(1, 128, 4, 4)}
        for method in (None, "index"):
            with self.subTest(method):
                self.stub.reset()
                p = pos if method is None else FluxKontextMultiReferenceLatentMethod.execute(pos, method)[0]
                with torch.inference_mode():
                    (res,) = comfy_nodes.common_ksampler(self.model(F2ID), 1, 2, 2.0, "euler", "simple",
                                                         p, neg, latent)
                steps = self.stub.events("step")
                self.assertEqual(sum(s["rows"] for s in steps), 4)
                for s in steps:
                    self.assertEqual(s["ref_shapes"], [[[1, 128, 6, 4]]] * s["rows"])
                    self.assertEqual(s["row_keys"], ["cond", "num_tokens", "ref_latents"])
                self.assertEqual(tuple(res["samples"].shape), (1, 128, 4, 4))
        p = FluxKontextMultiReferenceLatentMethod.execute(pos, "offset")[0]
        with self.assertRaises(C.GmlxRefusal) as e, torch.inference_mode():
            comfy_nodes.common_ksampler(self.model(F2ID), 1, 2, 2.0, "euler", "simple", p, neg, latent)
        self.assertIn("by the 'index' method, and this call asks for 'offset'", str(e.exception))

    def test_stock_lora_loader_with_the_gmlx_flux2_clip(self):
        model = self.model(F2ID)
        clip = P.nodes.GmlxCLIPLoader().load(F2ID, self.stub.url, False)[0]
        m2, c2 = comfy_nodes.LoraLoader().load_lora(model, clip, LORA_F2, 0.8, 1.0)
        self.assertIsInstance(m2, P.patcher.GmlxModelPatcher)
        self.assertIsInstance(c2, P.clip.GmlxCLIP)
        self.assertEqual(set(m2.patches), LORA_F2_KEYS)

    def test_flux3_clip_and_vae_carry_its_features_and_model_space_latents(self):
        clip = P.nodes.GmlxCLIPLoader().load(F3ID, self.stub.url, False)[0]
        (cond,) = comfy_nodes.CLIPTextEncode().encode(clip, "a lighthouse")
        self.assertEqual(tuple(cond[0][0].shape), (1, 32, 20480))
        # Flux3Tokenizer passes prevent_empty_text to Qwen3VLTokenizer.
        clip.encode_from_tokens_scheduled(clip.tokenize("", prevent_empty_text=True))
        self.assertEqual([e["text"] for e in self.stub.events("encode")], ["a lighthouse", " "])
        vae = P.nodes.GmlxVAELoader().load(F3ID, self.stub.url, False)[0]
        (img,) = comfy_nodes.VAEDecode().decode(vae, {"samples": torch.zeros(1, 128, 4, 6)})
        self.assertEqual(tuple(img.shape), (1, 64, 96, 3))
        (lat,) = comfy_nodes.VAEEncode().encode(vae, torch.rand(1, 70, 100, 3))
        self.assertEqual(tuple(lat["samples"].shape), (1, 128, 4, 6))
        self.assertEqual(self.stub.events("vae_encode")[-1]["shape"], [1, 64, 96, 3])

    @needs_flux3
    def test_a_flux3_edit_through_reference_multi_latent_reaches_the_step(self):
        """ComfyUI's FLUX.3 edit template: the text encode, its zeroed negative,
        and ReferenceMultiLatent encoding each reference with the VAE."""
        from comfy_extras.nodes_edit_model import ReferenceMultiLatent
        clip = P.nodes.GmlxCLIPLoader().load(F3ID, self.stub.url, False)[0]
        vae = P.nodes.GmlxVAELoader().load(F3ID, self.stub.url, False)[0]
        (pos,) = comfy_nodes.CLIPTextEncode().encode(clip, "make it blue")
        (neg,) = comfy_nodes.ConditioningZeroOut().zero_out(pos)
        ims = {"image1": torch.rand(1, 96, 64, 3), "image2": torch.rand(1, 64, 64, 3)}
        out = ReferenceMultiLatent.execute(pos, {"resolution": "keep resolution"}, neg, vae, ims)
        pos, neg, latent = out[0], out[1], out[2]
        self.assertEqual([tuple(r.shape) for r in pos[0][1]["reference_latents"]],
                         [(1, 128, 6, 4), (1, 128, 4, 4)])
        self.assertEqual([e["shape"] for e in self.stub.events("vae_encode")],
                         [[1, 96, 64, 3], [1, 64, 64, 3]])
        self.stub.reset()
        with torch.inference_mode():
            (res,) = comfy_nodes.common_ksampler(self.model(F3ID), 1, 2, 1.0, "euler", "simple",
                                                 pos, neg, latent)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), 2)
        self.assertEqual({tuple(s["refs"]) for s in steps}, {(2,)})
        self.assertEqual({tuple(s["x_shape"]) for s in steps}, {(1, 128, 6, 4)})
        self.assertEqual({tuple(s["row_keys"]) for s in steps}, {("cond", "num_tokens", "ref_latents")})
        self.assertEqual(tuple(res["samples"].shape), (1, 128, 6, 4))

    @needs_flux3
    def test_a_flux3_step_sends_its_references_in_their_order(self):
        """Two references of other sizes reach every row of the step in the
        order of the conditioning, the positive and the zeroed negative."""
        from comfy_extras.nodes_edit_model import ReferenceMultiLatent
        clip = P.nodes.GmlxCLIPLoader().load(F3ID, self.stub.url, False)[0]
        vae = P.nodes.GmlxVAELoader().load(F3ID, self.stub.url, False)[0]
        (pos,) = comfy_nodes.CLIPTextEncode().encode(clip, "put the cat on the sofa")
        (neg,) = comfy_nodes.ConditioningZeroOut().zero_out(pos)
        ims = {"image1": torch.full((1, 64, 96, 3), 0.25), "image2": torch.full((1, 96, 64, 3), 0.75)}
        pos, neg, latent = ReferenceMultiLatent.execute(pos, {"resolution": "keep resolution"}, neg, vae, ims)[:3]
        want = [list(r.shape) for r in pos[0][1]["reference_latents"]]
        self.assertEqual(want, [[1, 128, 4, 6], [1, 128, 6, 4]])
        self.stub.reset()
        with torch.inference_mode():
            comfy_nodes.common_ksampler(self.model(F3ID), 1, 1, 2.0, "euler", "simple", pos, neg, latent)
        steps = self.stub.events("step")
        self.assertEqual(sum(s["rows"] for s in steps), 2)
        for s in steps:
            self.assertEqual(s["ref_shapes"], [want] * s["rows"])

    @needs_flux3
    def test_stock_lora_loader_with_the_gmlx_flux3_clip(self):
        model = self.model(F3ID)
        clip = P.nodes.GmlxCLIPLoader().load(F3ID, self.stub.url, False)[0]
        m2, c2 = comfy_nodes.LoraLoader().load_lora(model, clip, LORA_F3, 0.8, 1.0)
        self.assertIsInstance(m2, P.patcher.GmlxModelPatcher)
        self.assertIsInstance(c2, P.clip.GmlxCLIP)
        self.assertEqual(set(m2.patches), LORA_F3_KEYS)

    def test_vae_nodes(self):
        zvae = P.nodes.GmlxVAELoader().load(ZID, self.stub.url, False)[0]
        qvae = P.nodes.GmlxVAELoader().load(QID, self.stub.url, False)[0]
        (img,) = comfy_nodes.VAEDecode().decode(zvae, {"samples": torch.zeros(1, 16, 8, 8)})
        self.assertEqual(tuple(img.shape), (1, 64, 64, 3))
        (img,) = comfy_nodes.VAEDecode().decode(qvae, {"samples": torch.zeros(1, 64, 4, 4)})
        self.assertEqual(tuple(img.shape), (1, 64, 64, 4))
        self.assertEqual(self.stub.events("vae_decode")[-1]["alpha"], True)
        (lat,) = comfy_nodes.VAEEncode().encode(zvae, torch.rand(1, 70, 68, 3))
        self.assertEqual(tuple(lat["samples"].shape), (1, 16, 8, 8))
        (lat,) = comfy_nodes.VAEEncode().encode(qvae, torch.rand(1, 64, 64, 3))
        self.assertEqual(tuple(lat["samples"].shape), (1, 64, 4, 4))
        self.assertEqual(self.stub.events("vae_encode")[-1]["shape"], [1, 64, 64, 4])
        with self.assertRaises(C.GmlxRefusal):
            zvae.get_sd()


@needs_flux3
class Upsampled(Base):
    """A server that upsamples the prompt of a text encode, which sees no
    reference images: an edit and a typed negative sample on the prompt as
    typed, and a plain prompt on the caption."""
    stub_args = ("--upsample",)

    def setUp(self):
        super().setUp()
        self.clip = P.nodes.GmlxCLIPLoader().load(F3ID, self.stub.url, False)[0]

    def encode(self, text):
        (c,) = comfy_nodes.CLIPTextEncode().encode(self.clip, text)
        return c, next(e for e in self.stub.events("encode") if e["text"] == text)

    def sampled(self, pos, neg, latent, cfg):
        self.stub.reset()
        with torch.inference_mode():
            comfy_nodes.common_ksampler(self.model(F3ID), 1, 2, cfg, "euler", "simple", pos, neg, latent)
        return {h for s in self.stub.events("step") for h in s["conds"]}

    def test_a_plain_prompt_samples_on_the_caption(self):
        pos, e = self.encode("a lighthouse")
        (neg,) = comfy_nodes.ConditioningZeroOut().zero_out(pos)
        got = self.sampled(pos, neg, {"samples": torch.zeros(1, 128, 4, 4)}, 1.0)
        self.assertEqual(got, {e["cond"]})

    def test_an_edit_samples_on_the_prompt_as_typed(self):
        from comfy_extras.nodes_edit_model import ReferenceMultiLatent
        vae = P.nodes.GmlxVAELoader().load(F3ID, self.stub.url, False)[0]
        pos, e = self.encode("make it blue")
        (neg,) = comfy_nodes.ConditioningZeroOut().zero_out(pos)
        pos, neg, latent = ReferenceMultiLatent.execute(
            pos, {"resolution": "keep resolution"}, neg, vae, {"image1": torch.rand(1, 64, 64, 3)})[:3]
        got = self.sampled(pos, neg, latent, 1.0)
        self.assertNotEqual(e["cond"], e["cond_typed"])
        self.assertEqual(got, {e["cond_typed"]})

    def test_a_typed_negative_samples_on_the_prompt_as_typed(self):
        pos, ep = self.encode("a lighthouse")
        neg, en = self.encode("blurry")
        got = self.sampled(pos, neg, {"samples": torch.zeros(1, 128, 4, 4)}, 2.0)
        self.assertEqual(got, {ep["cond"], en["cond_typed"]})


def slg_graph(*nodes) -> dict:
    """A submitted graph: the gmlx loader, ModelSamplingSD3 at shift 5, the
    SkipLayerGuidanceDiTSimple nodes of ``nodes`` in a chain, each the
    widgets it takes, and the KSampler, node 9."""
    g = {"1": {"class_type": "GmlxModelLoader", "inputs": {"model_name": MID}},
         "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 5.0}}}
    up = "2"
    for i, widgets in enumerate(nodes):
        nid = str(3 + i)
        g[nid] = {"class_type": "SkipLayerGuidanceDiTSimple",
                  "inputs": {"model": [up, 0], "double_layers": "", "single_layers": "22",
                             "start_percent": 0.0, "end_percent": 1.0, **widgets}}
        up = nid
    g["9"] = {"class_type": "KSampler", "inputs": {"model": [up, 0], "cfg": 3.0}}
    return {"prompt": g}


class MossImage(Base):
    """MOSS-Image, which ComfyUI has no class for: the pack's weightless model
    with the OpenMOSS plugin's settings, its LoRA key map, and block 22 left
    out of the negative pass under SkipLayerGuidanceDiTSimple."""

    def running(self, graph: dict):
        """The pack sees ``graph`` as the prompt whose KSampler runs, as
        ComfyUI's queue holds it."""
        from unittest import mock
        item = (0, "p1", graph["prompt"], {}, ["9"], [])
        queue = types.SimpleNamespace(get_current_queue_volatile=lambda: ([item], []))
        fake = types.SimpleNamespace(PromptServer=types.SimpleNamespace(
            instance=types.SimpleNamespace(last_node_id="9", last_prompt_id="p1",
                                           prompt_queue=queue)))
        return mock.patch.dict(sys.modules, {"server": fake})

    def slg(self, model, single="22", double="", start=0.0, end=1.0):
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
        m = ModelSamplingSD3().patch(model, 5.0)[0]
        return SkipLayerGuidanceDiTSimple.execute(m, start, end, double, single)[0]

    def test_moss_image_builds_on_meta_with_the_plugins_settings(self):
        gc.collect()
        r0 = rss()
        m = self.model(MID)
        gc.collect()
        grew = (rss() - r0) / 2 ** 20
        cfg = m.model.model_config
        self.assertEqual(type(cfg).__name__, "MOSSImageConfig")
        self.assertEqual(type(m.model).__name__, "MOSSImage")
        self.assertEqual(m.model.model_type, comfy.model_base.ModelType.FLOW)
        self.assertIsInstance(m.model.latent_format, comfy.latent_formats.Flux2)
        ms = m.get_model_object("model_sampling")
        self.assertEqual((float(ms.shift), float(ms.multiplier)), (1.0, 1000.0))
        self.assertEqual(m.model.diffusion_model.blocks, 42)
        names = {t["name"] for t in self.client.manifest(MID)["tensors"]}
        sd = m.model.diffusion_model.state_dict()
        self.assertEqual(set(sd), names)
        self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))
        self.assertLess(grew, 200.0)
        self.assertEqual(m.model.get_dtype(), torch.float32)

    def test_a_manifest_that_is_no_moss_image_dit_is_refused_by_name(self):
        man = self.client.manifest(MID)
        cut = {**man, "tensors": [t for t in man["tensors"]
                                  if t["name"] != "layers.0.proj.ffn_up_gate.weight"]}
        with self.assertRaises(C.GmlxError) as e:
            P.model.build_weightless(cut, "MOSSImage")
        self.assertIn("not a MOSS-Image DiT: it has no layers.0.proj.ffn_up_gate.weight",
                      str(e.exception))

    def test_a_moss_lora_on_its_training_names_becomes_a_table(self):
        """ComfyUI's generic key map reads the DiT's own names under the kohya
        and dotted forms; the fused feed-forward input and the refiner's qkv
        take their pairs whole."""
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(MID), LORA_M, 0.8)[0]
        self.assertEqual(set(m.patches), LORA_M_KEYS)
        self.assertEqual({t[3] for v in m.patches.values() for t in v}, {None})
        tid = m.sync_patches()
        table, _ = P.patches.build_table(MID, m.patches, family="moss_image")
        self.assertEqual(tid, C.table_id(table)[0])
        (put,) = self.stub.events("put_table")
        self.assertEqual((put["id"], put["entries"], put["bands"], put["strengths"]),
                         (tid, len(LORA_M_KEYS), 0, [0.8]))
        self.assertEqual(len(self.stub.events("put_adapter")), 2 * len(LORA_M_KEYS))

    def test_a_guided_sample_leaves_block_22_out_of_the_negative_pass_alone(self):
        m = self.slg(self.model(MID))
        with self.running(slg_graph({})):
            out = self.sample(m, MID, cfg=3.0, steps=3)
        steps = self.stub.events("step")
        self.assertEqual(len(steps), 2 * 3)
        self.assertEqual({s["rows"] for s in steps}, {1})
        self.assertEqual([s["skip"] for s in steps], [None, [22]] * 3)
        self.assertEqual(["skip" in s["fields"] for s in steps], [False, True] * 3)
        self.assertEqual({tuple(s["x_shape"]) for s in steps}, {(1, 128, 4, 4)})
        sig = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 3)
        want = [float(np.float32(v) * np.float32(1000.0)) for v in sig[:-1].tolist() for _ in (0, 1)]
        np.testing.assert_allclose([s["timestep"][0] for s in steps], want, rtol=1e-6)
        self.assertGreater(want[0] / 1000.0, 0.99)
        self.assertTrue(bool(torch.isfinite(out).all()))
        # Without the node both passes run whole, in one call.
        self.stub.reset()
        self.sample(self.model(MID), MID, cfg=3.0, steps=1)
        (s,) = self.stub.events("step")
        self.assertEqual((s["rows"], s["skip"]), (2, None))

    def test_skip_layer_guidance_refusals_name_each_cause(self):
        base = self.model(MID)
        node = "the SkipLayerGuidanceDiTSimple node 3"
        cases = {
            "double layers": (dict(double="7"), slg_graph({"double_layers": "7"}),
                              f"{node} skips double_layers [7]"),
            "window": (dict(start=0.0, end=0.5), slg_graph({"end_percent": 0.5}),
                       f"{node} skips its blocks from 0.0 to 0.5 of the run"),
            "other blocks than the widget": ({}, slg_graph({"single_layers": "21"}),
                                             "this call skips single blocks [22], which are not "
                                             f"the single_layers of {node}"),
            "linked widget": ({}, slg_graph({"single_layers": ["12", 0]}),
                              f"{node} takes single_layers from another node"),
            "past the last block": (dict(single="42"), slg_graph({"single_layers": "42"}),
                                    f"{node} skips blocks [42], and the DiT has blocks 0 to 41"),
            "no node": ({}, {"prompt": {"9": {"class_type": "KSampler", "inputs": {}}}},
                        "from a node other than SkipLayerGuidanceDiTSimple"),
            "two nodes that differ": ({}, slg_graph({}, {"single_layers": "21"}),
                                      "the SkipLayerGuidanceDiTSimple nodes 3, 4 skip other blocks"),
        }
        for name, (kw, graph, want) in cases.items():
            with self.subTest(name):
                m = self.slg(base, **kw)
                with self.running(graph), self.assertRaises(C.GmlxRefusal) as e:
                    self.sample(m, MID, cfg=3.0, steps=1)
                self.assertIn(want, str(e.exception))
        self.assertEqual([s["skip"] for s in self.stub.events("step")], [None] * len(cases))

    def test_a_skip_the_server_or_the_call_cannot_take_is_refused(self):
        f = self.model(MID).model.diffusion_model.forward
        cond = self.cond(MID, "x")[0][0]
        x, t = torch.zeros(1, 128, 4, 4), torch.tensor([500.0])

        def skip(args, extra):
            return args
        skip.__qualname__ = P.model.SLG_SKIP
        neg = {"cond_or_uncond": [1], "patches_replace": {"dit": {("single_block", 22): skip}}}
        with self.running(slg_graph({})):
            f(x, t, context=cond, transformer_options=neg)
            self.assertEqual(self.stub.events("step")[-1]["skip"], [22])
            cases = {
                "a positive row": ({**neg, "cond_or_uncond": [0]},
                                   "skips blocks of a positive row"),
                "another replacement": ({**neg, "patches_replace": {"dit": {
                    ("single_block", 22): lambda a, e: a}}},
                    "entry other than the block skips of SkipLayerGuidanceDiTSimple"),
                "another kind of patch": ({**neg, "patches_replace": {
                    "dit": {("single_block", 22): skip}, "attn2": {}}},
                    "entry other than the block skips of SkipLayerGuidanceDiTSimple")}
            for name, (to, want) in cases.items():
                with self.subTest(name), self.assertRaises(C.GmlxRefusal) as e:
                    f(x, t, context=cond, transformer_options=to)
                self.assertIn(want, str(e.exception))
            caps = self.client.capabilities
            self.client.capabilities = {**caps, "features": [
                v for v in caps["features"] if v != "skip_blocks"]}
            try:
                with self.assertRaises(C.GmlxRefusal) as e:
                    f(x, t, context=cond, transformer_options=neg)
                self.assertIn("this gmlx server skips no blocks in a step; update it",
                              str(e.exception))
            finally:
                self.client.capabilities = caps
            # A family whose DiT skips no blocks keeps its refusal of the replacement.
            z = self.model(ZID).model.diffusion_model.forward
            with self.assertRaises(C.GmlxRefusal) as e:
                z(torch.zeros(1, 16, 4, 4), torch.tensor([0.5]), context=self.cond(ZID, "x")[0][0],
                  transformer_options=neg)
            self.assertIn("transformer_options['patches_replace']", str(e.exception))
        self.assertEqual(len(self.stub.events("step")), 1)

    def test_a_masked_conditioning_is_refused_and_a_whole_one_samples(self):
        m = self.model(MID)
        pos = self.cond(MID, "a red bicycle")
        n = pos[0][0].shape[1]
        whole = [[pos[0][0], {"attention_mask": torch.ones(1, n)}]]
        self.sample(m, MID, cfg=1.0, steps=1, pos=whole)
        self.assertEqual(len(self.stub.events("step")), 1)
        mask = torch.ones(1, n)
        mask[0, -2:] = 0
        with self.assertRaises(C.GmlxRefusal) as e:
            self.sample(m, MID, cfg=1.0, steps=1, pos=[[pos[0][0], {"attention_mask": mask}]])
        self.assertIn("masks some of its text rows", str(e.exception))

    def test_the_moss_clip_sends_an_empty_prompt_as_it_is(self):
        # MOSSImageTokenizer applies its chat template first.
        clip = P.nodes.GmlxCLIPLoader().load(MID, self.stub.url, False)[0]
        (cond,) = comfy_nodes.CLIPTextEncode().encode(clip, "a lighthouse")
        self.assertEqual(tuple(cond[0][0].shape), (1, 32, 4096))
        clip.encode_from_tokens_scheduled(clip.tokenize("", prevent_empty_text=True))
        self.assertEqual([e["text"] for e in self.stub.events("encode")], ["a lighthouse", ""])

    def test_the_lora_key_map_equals_the_openmoss_plugins(self):
        """The patches ComfyUI makes of one LoRA on the pack's model and on
        the OpenMOSS plugin's own model of the same file are the same."""
        import importlib.util
        root = os.environ.get("GMLX_TEST_MOSS_PLUGIN")
        if not root:
            self.skipTest("the plugin comparison needs GMLX_TEST_MOSS_PLUGIN, the OpenMOSS "
                          "plugin's moss-image folder")
        spec = importlib.util.spec_from_file_location(
            "moss_image_plugin", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
        mod = sys.modules.get("moss_image_plugin")
        if mod is None:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["moss_image_plugin"] = mod
            spec.loader.exec_module(mod)
        loaders = importlib.import_module("moss_image_plugin.loaders")
        man = self.client.manifest(MID)
        state = {t["name"]: torch.empty(t["shape"], device="meta") for t in man["tensors"]}
        cfg = loaders.model_config(state)
        cfg.set_inference_dtype(torch.float32, None, device=torch.device("cpu"))
        theirs = comfy.model_patcher.ModelPatcher(cfg.get_model(state, device=torch.device("meta")),
                                                  torch.device("cpu"), torch.device("cpu"))
        ours = self.model(MID)
        self.assertEqual(set(ours.model.diffusion_model.state_dict()),
                         set(theirs.model.diffusion_model.state_dict()))
        sd = comfy.utils.load_torch_file(os.path.join(LORAS, LORA_M), safe_load=True)
        mw, _ = comfy.sd.load_lora_for_models(ours, None, sd, 0.8, 0.0)
        mp, _ = comfy.sd.load_lora_for_models(theirs, None, sd, 0.8, 0.0)
        self.assertEqual(set(mw.patches), set(mp.patches))
        self.assertEqual(set(mw.patches), LORA_M_KEYS)
        for k in mw.patches:
            for ta, tb in zip(mw.patches[k], mp.patches[k]):
                self.assertEqual((ta[0], ta[2], ta[3]), (tb[0], tb[2], tb[3]))
                for x, y in zip(ta[1].weights, tb[1].weights):
                    self.assertTrue((x is None and y is None) or torch.equal(torch.as_tensor(x), torch.as_tensor(y)))


class Widgets(unittest.TestCase):
    """The loaders list the server's image models, the default first, and
    show the address they use; a name the server lacks is refused with its
    `gmlx pull`; Cancel stops the wait for a starting server."""

    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()

    @classmethod
    def tearDownClass(cls):
        cls.stub.close()

    def setUp(self):
        keys = ("GMLX_BASE_URL", "GMLX_COMFY_SETTINGS", "GMLX_DEFAULT_IMAGE_MODEL", "PATH")
        self.env = {k: os.environ.get(k) for k in keys}
        self.dir = tempfile.mkdtemp()
        os.environ["GMLX_COMFY_SETTINGS"] = os.path.join(self.dir, "comfyui.json")
        os.environ["GMLX_BASE_URL"] = self.stub.url
        os.environ["GMLX_DEFAULT_IMAGE_MODEL"] = QID
        os.environ["PATH"] = "/nonexistent"
        self.probed()

    @staticmethod
    def probed():
        """Drop the kept model list and wait for a probe from now."""
        P.discovery.forget()
        P.discovery.refresh().join(30)

    def tearDown(self):
        mm.interrupt_current_processing(False)
        P.discovery.forget()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_the_loaders_list_the_server_models_default_first(self):
        want = [QID, ZID, QTID, KID, F2ID, F2BID, F29ID, F29BID, F29KID, F3ID, F3BID, MID]
        for name in ("GmlxModelLoader", "GmlxCLIPLoader", "GmlxVAELoader", "GmlxServerStatus"):
            node = P.NODE_CLASS_MAPPINGS[name]
            req = node.INPUT_TYPES()["required"]
            names, opts = req["model_name"]
            self.assertEqual(names, want, name)
            self.assertEqual(opts["default"], QID, name)
            self.assertEqual(req["server_url"][0], "STRING")
            self.assertEqual(req["server_url"][1]["default"], self.stub.url, name)
            # A saved name the list lacks passes validation and reaches the node.
            self.assertIs(node.VALIDATE_INPUTS("qwen-image"), True)

    def test_the_server_default_image_model_comes_first(self):
        stub = support.Stub("--default-model", KID)
        try:
            os.environ["GMLX_BASE_URL"] = stub.url
            os.environ.pop("GMLX_DEFAULT_IMAGE_MODEL")
            self.probed()
            names, opts = P.nodes.GmlxModelLoader.INPUT_TYPES()["required"]["model_name"]
            self.assertEqual(names, [KID, ZID, QID, QTID, F2ID, F2BID, F29ID, F29BID, F29KID,
                                     F3ID, F3BID, MID])
            self.assertEqual(opts["default"], KID)
        finally:
            stub.close()

    def test_a_model_the_server_lacks_is_refused_with_its_pull(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxModelLoader().load("flux", self.stub.url, False)
        self.assertIn("`gmlx pull z-image-turbo`", str(e.exception))
        self.assertIn(ZID, str(e.exception))
        # The server has qwen-image as qwen-image-2.1: the refusal names that
        # id, since `gmlx pull qwen-image` would add no qwen-image id.
        text = P.nodes.GmlxServerStatus().status("qwen-image", server_url=self.stub.url)["result"][0]
        self.assertIn(f"pick {QID!r}", text)
        self.assertIn("or run `gmlx pull qwen-image` and pick the id it adds", text)
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxModelLoader().load("qwen-image", self.stub.url, False)
        self.assertIn(f"It has that model as {QID!r}", str(e.exception))

    def test_the_status_node_says_when_the_default_address_gives_way(self):
        out = P.nodes.GmlxServerStatus().status(ZID, server_url=C.DEFAULT_URL)
        text = out["result"][0]
        if self.stub.url != C.DEFAULT_URL:
            self.assertIn(f"the loader's server_url is the default {C.DEFAULT_URL}, so the "
                          f"nodes use the gmlx server at {self.stub.url}", text)
        self.assertTrue(text.startswith(f"{ZID}: z_image turbo"), text)

    def test_cancel_stops_the_wait_for_a_starting_server(self):
        import socket
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            port = sk.getsockname()[1]
        os.environ["GMLX_BASE_URL"] = f"http://127.0.0.1:{port}"
        # The server loads: gmlx status says it runs.
        status = json.dumps({"running": True, "port": port})
        exe = os.path.join(self.dir, "gmlx")
        with open(exe, "w") as f:
            f.write(f"#!/bin/sh\ncase \"$1\" in\n  serve) exit 0 ;;\n  *) echo '{status}' ;;\nesac\n")
        os.chmod(exe, 0o755)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": exe}, f)
        timer = threading.Timer(0.5, mm.interrupt_current_processing, (True,))
        timer.start()
        t0 = time.monotonic()
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                P.nodes.GmlxModelLoader().load(ZID, "", True)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - t0, 2.5)


class RoundTrip(Base):
    """The stub round trip at 1024x1024 latents."""

    def test_median_round_trip_under_25_ms(self):
        m = self.model()
        m.pre_run()
        res = {}
        try:
            f = m.model.diffusion_model.forward
            for label, B in (("rows2", 2), ("rows1", 1)):
                g = torch.Generator().manual_seed(0)
                x = torch.randn(B, 16, 128, 128, generator=g)
                ctx = torch.randn(B, 32, 2560, generator=g)
                to = {"cond_or_uncond": list(range(B)), "uuids": [uuid.uuid4() for _ in range(B)]}
                for _ in range(10):
                    f(x, torch.full((B,), 0.5), context=ctx, transformer_options=to)
                f.calls.clear()
                for _ in range(100):
                    f(x, torch.full((B,), 0.5), context=ctx, transformer_options=to)
                walls = [c["wall_ms"] for c in f.calls]
                res[label] = statistics.median(walls)
        finally:
            m.cleanup()
        print(f"\nround trip median ms: {res}")
        for label, v in res.items():
            self.assertLess(v, 25.0, label)


class AgainstComfyGGUF(Base):
    """The weightless model's unet_config and LoRA patches equal those of a
    ComfyUI-GGUF model of the same file, the one the stub's manifest of the
    model was read from."""

    def test_unet_config_and_patches_equal_comfyui_gguf(self):
        self.against("GMLX_TEST_ZIMAGE_GGUF", ZID, LORA)

    def test_flux2_unet_config_and_patches_equal_comfyui_gguf(self):
        self.against("GMLX_TEST_FLUX2_GGUF", F2ID, LORA_F2)

    def test_flux2_9b_unet_config_and_patches_equal_comfyui_gguf(self):
        self.against("GMLX_TEST_FLUX2_9B_GGUF", F29ID, LORA_F29)

    @needs_flux3
    def test_flux3_unet_config_and_patches_equal_comfyui_gguf(self):
        self.against("GMLX_TEST_FLUX3_GGUF", F3ID, LORA_F3)

    def against(self, var, mid, lora):
        import importlib.util
        import sys
        path = os.environ.get(var)
        if not path:
            self.skipTest(f"the ComfyUI-GGUF comparison needs {var} and ComfyUI-GGUF in the checkout")
        gdir = os.path.join(support.setup_comfy(), "custom_nodes", "ComfyUI-GGUF")
        if not os.path.isdir(gdir):
            self.skipTest("ComfyUI-GGUF is not in this checkout")
        folder_paths.add_model_folder_path("diffusion_models", os.path.dirname(path))
        mod = sys.modules.get("ComfyUI_GGUF")
        if mod is None:
            spec = importlib.util.spec_from_file_location("ComfyUI_GGUF", os.path.join(gdir, "__init__.py"),
                                                          submodule_search_locations=[gdir])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["ComfyUI_GGUF"] = mod
            spec.loader.exec_module(mod)
        (gg,) = mod.nodes.UnetLoaderGGUF().load_unet(os.path.basename(path))
        ours = self.model(mid)
        a = {k: v for k, v in ours.model.model_config.unet_config.items() if k != "dtype"}
        b = {k: v for k, v in gg.model.model_config.unet_config.items() if k != "dtype"}
        self.assertEqual(a, b)
        sd = comfy.utils.load_torch_file(os.path.join(LORAS, lora), safe_load=True)
        mw, _ = comfy.sd.load_lora_for_models(ours, None, sd, 0.8, 0.0)
        mg, _ = comfy.sd.load_lora_for_models(gg, None, sd, 0.8, 0.0)
        self.assertEqual(set(mw.patches), set(mg.patches))
        for k in mw.patches:
            for ta, tb in zip(mw.patches[k], mg.patches[k]):
                self.assertEqual((ta[0], ta[2], ta[3]), (tb[0], tb[2], tb[3]))
                for x, y in zip(ta[1].weights, tb[1].weights):
                    self.assertTrue((x is None and y is None) or torch.equal(torch.as_tensor(x), torch.as_tensor(y)))


if __name__ == "__main__":
    unittest.main()
