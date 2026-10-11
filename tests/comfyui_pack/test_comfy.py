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
import unittest
import uuid

import numpy as np
import torch

import support

support.setup_comfy()
P = support.load_pack()

import comfy.model_management as mm  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.weight_adapter  # noqa: E402
import folder_paths  # noqa: E402
import nodes as comfy_nodes  # noqa: E402
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow  # noqa: E402

C = P.client
ZID, QID, QTID = "z-image-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo"
KID = "krea-2-turbo"
STEPS = 4
LORAS = tempfile.mkdtemp(prefix="gmlx-pack-loras-")
folder_paths.add_model_folder_path("loras", LORAS)
LORA = "zimage-test-lora.safetensors"
LORA_TARGETS = support.write_zimage_lora(os.path.join(LORAS, LORA), support.manifest("z_image"))


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
        table, _ = P.patches.build_table(ZID, m.patches)
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
        table, _ = P.patches.build_table(ZID, stacked.patches)
        self.assertEqual({len(e["patches"]) for e in table["entries"]}, {2})
        self.assertEqual(len({s[3] for s in seq if s[3]}), 3)

    def test_a_dropped_adapter_is_uploaded_again_on_409(self):
        m = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(self.model(), LORA, 0.7)[0]
        # The client believes the server holds every pair; the server holds none.
        self.client.known_adapters.update(P.patches.build_table(ZID, m.patches)[1])
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
                     "reference latents": dict(ref_latents=[torch.zeros(1, 16, 4, 4)])}
            for name, kw in cases.items():
                with self.subTest(name):
                    with self.assertRaises(C.GmlxRefusal) as e:
                        self.forward(m, **kw)
                    self.assertIn(name, str(e.exception))
            self.assertEqual(self.stub.events("step"), [])
        finally:
            m.model.current_patcher = None

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
        base, _ = P.nodes.GmlxSigmas().get_sigmas(self.model(QID))
        self.assertEqual(len(base), 41)

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
    stub_args = ("--step-delay-ms", "400")

    def test_interrupt_reaches_the_server_and_stops_the_run(self):
        m = self.model()
        done = threading.Event()

        def watch():
            while not done.is_set():
                if len(self.stub.events("step")) >= 2:
                    # A step event is logged as its step ends; interrupt
                    # inside the next step's delay, not between steps,
                    # where ComfyUI's own check stops the run first.
                    time.sleep(0.15)
                    mm.interrupt_current_processing(True)
                    return
                time.sleep(0.05)
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
        self.assertLess(took, 8 * 0.4)


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
        P.discovery.forget()

    def tearDown(self):
        mm.interrupt_current_processing(False)
        P.discovery.forget()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_the_loaders_list_the_server_models_default_first(self):
        want = [QID, ZID, QTID, KID]
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

    def test_a_model_the_server_lacks_is_refused_with_its_pull(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxModelLoader().load("qwen-image", self.stub.url, False)
        self.assertIn("`gmlx pull qwen-image`", str(e.exception))
        self.assertIn(ZID, str(e.exception))
        out = P.nodes.GmlxServerStatus().status("qwen-image", server_url=self.stub.url)
        self.assertIn("`gmlx pull qwen-image`", out["result"][0])

    def test_cancel_stops_the_wait_for_a_starting_server(self):
        exe = os.path.join(self.dir, "gmlx")
        with open(exe, "w") as f:
            f.write("#!/bin/sh\ncase \"$1\" in\n  serve) exit 0 ;;\n  *) echo '{}' ;;\nesac\n")
        os.chmod(exe, 0o755)
        with open(os.environ["GMLX_COMFY_SETTINGS"], "w") as f:
            json.dump({"gmlx_command": exe}, f)
        import socket
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            os.environ["GMLX_BASE_URL"] = f"http://127.0.0.1:{sk.getsockname()[1]}"
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


@unittest.skipUnless(os.environ.get("GMLX_TEST_ZIMAGE_GGUF"),
                     "the ComfyUI-GGUF comparison needs GMLX_TEST_ZIMAGE_GGUF and ComfyUI-GGUF in the checkout")
class AgainstComfyGGUF(Base):
    """The weightless model's unet_config and LoRA patches equal
    those of a ComfyUI-GGUF model of the same file."""

    def test_unet_config_and_patches_equal_comfyui_gguf(self):
        import importlib.util
        import sys
        path = os.environ["GMLX_TEST_ZIMAGE_GGUF"]
        gdir = os.path.join(support.setup_comfy(), "custom_nodes", "ComfyUI-GGUF")
        if not os.path.isdir(gdir):
            self.skipTest("ComfyUI-GGUF is not in this checkout")
        folder_paths.add_model_folder_path("diffusion_models", os.path.dirname(path))
        spec = importlib.util.spec_from_file_location("ComfyUI_GGUF", os.path.join(gdir, "__init__.py"),
                                                      submodule_search_locations=[gdir])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ComfyUI_GGUF"] = mod
        spec.loader.exec_module(mod)
        (gg,) = mod.nodes.UnetLoaderGGUF().load_unet(os.path.basename(path))
        ours = self.model()
        a = {k: v for k, v in ours.model.model_config.unet_config.items() if k != "dtype"}
        b = {k: v for k, v in gg.model.model_config.unet_config.items() if k != "dtype"}
        self.assertEqual(a, b)
        sd = comfy.utils.load_torch_file(os.path.join(LORAS, LORA), safe_load=True)
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
