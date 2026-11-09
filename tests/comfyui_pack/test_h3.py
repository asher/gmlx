"""MiniMax-H3 and FastH3 through the pack against the stub server: the
weightless model and the ComfyUI keys its forward reads, a sample of
ComfyUI's nested video and audio latent through the AV step with the shifts
and audio scale ComfyUI's forward reads, FastH3 on its trained sigmas, a
LoRA from the stock loader, the model loader's attention on each AV step,
the 409 retry and the interrupt of an AV step, the video and audio VAE
nodes, keyframes and guides from the stock nodes through the gmlx CLIP and
VAEs to the AV step, and the refusals of what a
gmlx model does not run: references, FastH3 with keyframes, the noise masks
and the Fun ControlNet. Every ComfyUI the pack supports has MiniMax-H3,
which came in v0.30.0.

Needs ``COMFYUI_DIR``. Run with ``python -m unittest`` from this folder."""

from __future__ import annotations

import threading
import time
import unittest
import uuid

import torch

import support

support.setup_comfy()
P = support.load_pack()

import comfy.model_base  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import node_helpers  # noqa: E402
import nodes as comfy_nodes  # noqa: E402
from comfy_extras import nodes_custom_sampler as CS  # noqa: E402
from comfy_extras import nodes_minimax_h3 as H  # noqa: E402
from comfy_extras.nodes_audio import vae_decode_audio  # noqa: E402

C = P.client
HID, FID, ZID = "h3", "fasth3", "z-image-turbo"
# The 5-frame 64x64 clip, the smallest the latent node makes.
SIZE, FRAMES = 64, 5
NO_REFS = ("the gmlx nodes run MiniMax-H3 from the text prompt with keyframes and guides, and "
           "take no references")


def _out(node_output):
    return node_output.result if hasattr(node_output, "result") else node_output


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

    def model(self, mid=HID):
        return P.model.load_model(self.client, mid)

    def cond(self, text="a fox crossing a frozen river at dawn"):
        out, fields = self.client.encode(HID, text)
        return [[out["cond"], {}]], fields["cond"]

    def latent(self):
        return _out(H.EmptyMiniMaxH3LatentAV.execute(width=SIZE, height=SIZE, length=FRAMES))[0]

    def sample(self, model, pos=None, latent=None, steps=3):
        pos = pos or self.cond()[0]
        (out,) = comfy_nodes.common_ksampler(model, 1, steps, 1.0, "res_multistep", "simple",
                                             pos, pos, latent or self.latent())
        return out


class Model(Base):
    def test_h3_and_fasth3_build_on_meta_from_each_of_the_four_loaders(self):
        for mid in (HID, FID):
            with self.subTest(mid):
                m = P.nodes.GmlxModelLoader().load(mid, self.stub.url, False)[0]
                self.assertEqual(type(m.model.model_config).__name__, "MiniMaxH3")
                self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))
                self.assertIs(m.model.diffusion_model.forward.av, True)
                clip = P.nodes.GmlxCLIPLoader().load(mid, self.stub.url, False)[0]
                self.assertEqual(clip.gmlx_model, mid)
                for node in (P.nodes.GmlxVideoVAELoader, P.nodes.GmlxAudioVAELoader):
                    self.assertEqual(node().load(mid, self.stub.url, False)[0].gmlx_model, mid)

    def test_a_key_comfyui_stops_writing_is_named_at_the_load(self):
        """The forward reads the shifts and the payload keys with defaults,
        so a ComfyUI that renames one would run the clip at 12 and 3 or an
        audio scale of 1 with no error. The load names the key instead."""
        shift, conds = H.MiniMaxH3SigmaShift.execute, comfy.model_base.MiniMaxH3.extra_conds

        def renamed_shift(key):
            def execute(model, shift_video, shift_audio):
                m = _out(shift(model, shift_video, shift_audio))[0]
                to = m.model_options["transformer_options"]
                to[key + "_v2"] = to.pop(key)
                return (m,)
            return execute

        def renamed_payload(key):
            def extra_conds(self, **kwargs):
                out = conds(self, **kwargs)
                if key == "minimax_payload":
                    out[key + "_v2"] = out.pop(key)
                else:
                    held = out["minimax_payload"].cond
                    held[key + "_v2"] = held.pop(key)
                return out
            return extra_conds

        def renamed_input(key):
            def extra_conds(self, **kwargs):
                kwargs[key + "_v2"] = kwargs.pop(key)
                return conds(self, **kwargs)
            return extra_conds

        def reads_device(self, **kwargs):
            kwargs["device"]
            return conds(self, **kwargs)

        def new_shift_inputs(model, shift_v, shift_a):
            return shift(model, shift_v, shift_a)

        def written(key, given):
            return f"minimax_payload[{key!r}]" + (f" from {given}" if given else "")

        cases = [(H.MiniMaxH3SigmaShift, "execute", renamed_shift(k),
                  f"ModelSamplingMiniMaxH3 does not write transformer_options[{k!r}]")
                 for k in P.model.H3_SHIFTS.values()]
        cases += [(comfy.model_base.MiniMaxH3, "extra_conds", renamed_payload("minimax_payload"),
                   "MiniMaxH3 model does not write minimax_payload")]
        cases += [(comfy.model_base.MiniMaxH3, "extra_conds", renamed_payload(k),
                   f"MiniMaxH3 model does not write {written(k, given)}")
                  for k, (_what, given) in P.model.H3_PAYLOAD.items()]
        # A ComfyUI that renames an input the payload is written from.
        cases += [(comfy.model_base.MiniMaxH3, "extra_conds", renamed_input(given),
                   f"MiniMaxH3 model does not write {written(k, given)}")
                  for k, (_what, given) in P.model.H3_PAYLOAD.items() if given]
        # A node whose inputs are renamed.
        cases += [(H.MiniMaxH3SigmaShift, "execute", new_shift_inputs,
                   "ModelSamplingMiniMaxH3 fails on a model and shifts of 10 and 3, which the "
                   "gmlx nodes check it with (TypeError: ")]
        for owner, name, fake, want in cases:
            with self.subTest(want):
                real = owner.__dict__[name]
                setattr(owner, name, fake)
                try:
                    with self.assertRaises(C.GmlxError) as e:
                        self.model()
                finally:
                    setattr(owner, name, real)
                self.assertIn(f"gmlx: ComfyUI {P.compat.comfyui_version()}'s {want}",
                              str(e.exception))
        # The load passes the device every sampler passes.
        real = comfy.model_base.MiniMaxH3.__dict__["extra_conds"]
        comfy.model_base.MiniMaxH3.extra_conds = reads_device
        try:
            self.model()
        finally:
            comfy.model_base.MiniMaxH3.extra_conds = real

    def test_a_keyframe_key_comfyui_stops_writing_is_named_at_the_load(self):
        """The AV step reads a guide's frame, latents and sound latents by
        key; a ComfyUI whose MiniMaxH3AddGuide renames one is named at the
        load."""
        real = H.MiniMaxH3AddGuide.__dict__["execute"]
        add = H.MiniMaxH3AddGuide.execute

        def renamed(key):
            def execute(positive, latent, frame_idx, **kw):
                pos = _out(add(positive, latent, frame_idx, **kw))[0]
                kf = pos[0][1]["minimax_keyframes"][-1]
                if key in kf:
                    kf[key + "_v2"] = kf.pop(key)
                return (pos,)
            return execute

        for key in ("resolved_frame_index", "latent", "audio_latent"):
            with self.subTest(key):
                H.MiniMaxH3AddGuide.execute = renamed(key)
                try:
                    with self.assertRaises(C.GmlxError) as e:
                        self.model()
                finally:
                    H.MiniMaxH3AddGuide.execute = real
                self.assertIn(f"gmlx: ComfyUI {P.compat.comfyui_version()}'s MiniMaxH3AddGuide "
                              f"does not write a keyframe's {key!r}", str(e.exception))

    def test_a_comfyui_whose_anchor_noise_mix_differs_is_named_at_the_load(self):
        """The server mixes the anchor rows at ComfyUI's defaults, 0.999 for
        the video and 1 for the audio; a ComfyUI that changes one is named
        at the load, not run at the old one."""
        import comfy.ldm.minimax.model as h3_model

        real = h3_model.VISUAL_COND_TIMESTEP
        h3_model.VISUAL_COND_TIMESTEP = 0.995
        try:
            with self.assertRaises(C.GmlxError) as e:
                self.model()
        finally:
            h3_model.VISUAL_COND_TIMESTEP = real
        self.assertIn(f"gmlx: ComfyUI {P.compat.comfyui_version()} mixes the video anchor rows "
                      "at 0.995 (VISUAL_COND_TIMESTEP), and a gmlx model at 0.999",
                      str(e.exception))

    def test_a_missing_h3_internal_is_named_with_the_version(self):
        self.assertIsNone(P.compat.first_missing(P.compat.H3_NAMES))
        names = P.compat.H3_NAMES
        try:
            P.compat.H3_NAMES = names + (("comfy.ldm.minimax.model", "MiniMaxH3Model.gone"),)
            with self.assertRaises(C.GmlxError) as e:
                self.model()
        finally:
            P.compat.H3_NAMES = names
        for want in ("comfy.ldm.minimax.model.MiniMaxH3Model.gone",
                     f"ComfyUI {P.compat.comfyui_version()}", "update ComfyUI"):
            self.assertIn(want, str(e.exception))


class Sampling(Base):
    def test_a_sample_steps_the_clip_with_the_shifts_and_audio_scale_it_carries(self):
        """Without ModelSamplingMiniMaxH3 the DiT's own shifts 12 and 3 hold;
        with it, the node's. The step gets the encoder's features as they
        were encoded, the timestep at sigma x 1000 and the audio scale
        shift / audio_shift."""
        pos, cond = self.cond()
        for shifts in (None, (6.0, 3.0)):
            with self.subTest(shifts=shifts):
                self.stub.reset()
                model = self.model()
                if shifts:
                    model = _out(H.MiniMaxH3SigmaShift.execute(model, *shifts))[0]
                out = self.sample(model, pos)
                video, audio = (tuple(t.shape) for t in out["samples"].unbind())
                want_v, want_a = (tuple(t.shape) for t in self.latent()["samples"].unbind())
                self.assertEqual((video, audio), (want_v, want_a))
                sv, sa = shifts or (12.0, 3.0)
                sig = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),
                                                      "simple", 3)
                steps = self.stub.events("av_step")
                self.assertEqual(len(steps), 3)
                for e, s in zip(steps, sig[:-1].tolist()):
                    self.assertAlmostEqual(e["timestep"], s * 1000, places=2)
                    self.assertEqual(e["shifts"], {"video": sv, "audio": sa})
                    self.assertAlmostEqual(e["audio_scale"], sv / sa, places=5)
                    self.assertEqual(e["conds"], [cond])
                    self.assertEqual((tuple(e["video_shape"]), tuple(e["audio_shape"])),
                                     (want_v, want_a))
                    self.assertEqual(e["video_dtype"], "F32")
                    self.assertIsNotNone(e["run"])

    def custom(self, model, sigmas, pos=None):
        sampler = comfy.samplers.sampler_object("euler")
        pos = pos or self.cond()[0]
        return _out(CS.SamplerCustom.execute(model, True, 42, 1.0, pos, pos, sampler, sigmas,
                                             self.latent()))[0]

    def test_fasth3_samples_on_its_trained_sigmas_at_its_shifts(self):
        """No stock scheduler makes FastH3's eight sigmas, so GmlxSigmas
        hands the sampler those of the server's status, and the steps run
        at sigma x 1000 with FastH3's shifts and audio scale."""
        model = _out(H.MiniMaxH3SigmaShift.execute(self.model(FID), 10.0, 3.0))[0]
        sig, cfg = P.nodes.GmlxSigmas().get_sigmas(model)
        self.assertEqual(sig.tolist(), torch.tensor(self.client.status(FID)["sampling"]["sigmas"],
                                                    dtype=torch.float32).tolist())
        self.assertEqual((len(sig), cfg), (9, 1.0))
        self.custom(model, sig)
        steps = self.stub.events("av_step")
        self.assertEqual([e["timestep"] for e in steps], (sig[:-1] * 1000).tolist())
        for e in steps:
            self.assertEqual(e["shifts"], {"video": 10.0, "audio": 3.0})
            self.assertAlmostEqual(e["audio_scale"], 10.0 / 3.0, places=6)

    def test_fasth3_on_its_trained_sigmas_at_other_shifts_is_refused(self):
        """Without ModelSamplingMiniMaxH3 ComfyUI's forward would derive the
        audio sigma at 12 and 3, off FastH3's trained audio sigmas, with no
        error, and at a wrong audio shift alone off them too. A split of the
        trained sigmas is refused at its first sampler. A schedule of the
        user's own runs at the shifts it is given."""
        sig = P.nodes.GmlxSigmas().get_sigmas(self.model(FID))[0]
        tail = P.nodes.GmlxSigmas().get_sigmas(self.model(FID), 0, 0.5)[0]
        head = _out(CS.SplitSigmas.execute(sig, 4))[0]
        for shifts, sigmas, ran in ((None, sig, "12 and 3"), ((6.0, 3.0), sig, "6 and 3"),
                                    ((10.0, 4.0), sig, "10 and 4"),
                                    (None, tail, "12 and 3"), (None, head, "12 and 3")):
            with self.subTest(shifts=shifts, steps=len(sigmas) - 1):
                self.stub.reset()
                model = self.model(FID)
                if shifts:
                    model = _out(H.MiniMaxH3SigmaShift.execute(model, *shifts))[0]
                with self.assertRaises(C.GmlxRefusal) as e:
                    self.custom(model, sigmas)
                self.assertEqual(str(e.exception), "gmlx: 'fasth3' samples on its trained "
                                 f"sigmas at the shifts 10 and 3, and this run is at {ran}; put "
                                 "ModelSamplingMiniMaxH3 on the model at 10 and 3")
                self.assertEqual(self.stub.events("av_step"), [])
                self.assertEqual(self.stub.events("put_tensor"), [])
        self.stub.reset()
        model = _out(H.MiniMaxH3SigmaShift.execute(self.model(FID), 6.0, 3.0))[0]
        own = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"),
                                              "simple", 4)
        self.custom(model, own)
        self.assertEqual(len(self.stub.events("av_step")), 4)

    def test_fasth3_on_basic_scheduler_without_the_shift_node_runs_at_12_and_3(self):
        """BasicScheduler on FastH3 with no ModelSamplingMiniMaxH3 makes a
        schedule of its own at the DiT's shifts, 12 and 3, which is no run of
        FastH3's trained sigmas. The guard lets it run, and every step goes at
        12 and 3 as ComfyUI's forward would take them, with one warning in
        the log that names the trained shifts."""
        model = self.model(FID)
        sig = _out(CS.BasicScheduler.execute(model, "simple", 8, 1.0))[0]
        with self.assertLogs("gmlx", "WARNING") as logs:
            self.custom(model, sig)
        self.assertEqual(logs.output, [
            "WARNING:gmlx:gmlx: 'fasth3' samples on its trained sigmas at the shifts 10 and 3, "
            "and this run is at 12 and 3 on a schedule of its own; to sample as trained, put "
            "ModelSamplingMiniMaxH3 on the model at 10 and 3 and sample on gmlx Sigmas"])
        steps = self.stub.events("av_step")
        self.assertEqual(len(steps), 8)
        for e, s in zip(steps, sig[:-1].tolist()):
            self.assertAlmostEqual(e["timestep"], s * 1000, places=2)
            self.assertEqual(e["shifts"], {"video": 12.0, "audio": 3.0})
            self.assertAlmostEqual(e["audio_scale"], 4.0, places=5)

    def _forward(self, model, to: dict):
        """One forward of the model's DiT with ``to``, as a sampler calls it."""
        mm.load_models_gpu([model])
        model.pre_run()
        pos, _ = self.cond()
        video, audio = self.latent()["samples"].unbind()
        try:
            return model.model.diffusion_model.forward(
                [video, audio], torch.tensor([999.9]), context=pos[0][0],
                transformer_options={"cond_or_uncond": [0], "uuids": [uuid.uuid4()], **to},
                minimax_payload={"audio_scale": 10.0 / 3.0})
        finally:
            model.cleanup()

    def test_a_fasth3_call_with_no_schedule_is_refused_at_other_shifts(self):
        """A ComfyUI that stops setting transformer_options['sample_sigmas']
        would turn the guard off. A FastH3 call without it at shifts other
        than FastH3's is refused by the key it lacks, and one at FastH3's
        shifts steps."""
        with self.assertRaises(C.GmlxError) as e:
            self._forward(self.model(FID), {})
        self.assertEqual(str(e.exception), "gmlx: 'fasth3' samples on its trained sigmas at "
                         "the shifts 10 and 3, and this run is at 12 and 3 with no "
                         "transformer_options['sample_sigmas'] to tell its schedule from them; "
                         "put ModelSamplingMiniMaxH3 on the model at 10 and 3")
        self.assertEqual(self.stub.events("av_step"), [])
        model = _out(H.MiniMaxH3SigmaShift.execute(self.model(FID), 10.0, 3.0))[0]
        to = model.model_options["transformer_options"]
        self._forward(model, {k: to[k] for k in P.model.H3_SHIFTS.values()})
        self.assertEqual(len(self.stub.events("av_step")), 1)

    def test_an_av_step_after_the_server_dropped_its_tensors_uploads_again(self):
        """A server restart or eviction between steps drops the cond and the
        LoRA's table; the step puts them again and runs once."""
        names = {t["name"]: t["shape"] for t in support.manifest("h3")["tensors"]}
        dout, din = names["blocks.0.attn.qkv_proj.weight"]
        k = "diffusion_model.blocks.0.attn.qkv_proj"
        sd = {f"{k}.lora_A.weight": torch.full((4, din), 0.01).bfloat16(),
              f"{k}.lora_B.weight": torch.full((dout, 4), 0.01).bfloat16(),
              f"{k}.alpha": torch.tensor(2.0)}
        m, _ = comfy.sd.load_lora_for_models(self.model(), None, sd, 0.8, 0.0)
        mm.load_models_gpu([m])
        m.pre_run()
        pos, _ = self.cond()
        video, audio = self.latent()["samples"].unbind()
        to = {"cond_or_uncond": [0], "uuids": [uuid.uuid4()]}
        retries = self.client.stats["missing_retries"]
        try:
            f = m.model.diffusion_model.forward
            for i, t in enumerate((600.0, 400.0)):
                if i:
                    self.stub.forget()
                    self.stub.reset()
                out = f([video, audio], torch.tensor([t]), context=pos[0][0],
                        transformer_options=dict(to), minimax_payload={"audio_scale": 2.0})
            self.assertEqual([tuple(o.shape) for o in out],
                             [tuple(video.shape), tuple(audio.shape)])
        finally:
            m.cleanup()
        kinds = [e["kind"] for e in self.stub.log()]
        self.assertEqual(kinds[0], "step_missing")
        self.assertEqual((kinds.count("put_tensor"), kinds.count("put_table"),
                          kinds.count("av_step")), (1, 1, 1))
        self.assertEqual(self.client.stats["missing_retries"], retries + 1)

    def test_a_lora_from_the_stock_loader_steps_under_its_table(self):
        names = {t["name"]: t["shape"] for t in support.manifest("h3")["tensors"]}
        stems = ("blocks.0.attn.qkv_proj", "blocks.49.mlp.fc2",
                 "token_refiner.blocks.1.attn.out_proj")
        g = torch.Generator().manual_seed(0)
        sd = {}
        for stem in stems:
            dout, din = names[f"{stem}.weight"]
            k = f"diffusion_model.{stem}"
            sd[f"{k}.lora_A.weight"] = (torch.randn(4, din, generator=g) * 0.01).bfloat16()
            sd[f"{k}.lora_B.weight"] = (torch.randn(dout, 4, generator=g) * 0.01).bfloat16()
            sd[f"{k}.alpha"] = torch.tensor(2.0)
        lora, _ = comfy.sd.load_lora_for_models(self.model(), None, sd, 0.8, 0.0)
        self.sample(lora, steps=1)
        (put,) = self.stub.events("put_table")
        self.assertEqual((put["entries"], put["strengths"]), (len(stems), [0.8]))
        (step,) = self.stub.events("av_step")
        self.assertEqual(step["patch_table"], put["id"])


class Int8(Base):
    """The model loader's int8 input: entry sends nothing and each other
    value goes on each AV step of a server that takes it; a value whose
    linears differ from the entry's is refused at the load when the
    status names the entry's setting, else at the first step; an older
    server runs the entry's with one warning a model when the loader asks
    for a value other than entry or off."""

    def setUp(self):
        super().setUp()
        self.stub.int8("off")

    def tearDown(self):
        self.stub.int8("off")

    def loaded(self, int8, mid=HID):
        return P.nodes.GmlxModelLoader().load(mid, self.stub.url, False, int8)[0]

    def first_step(self, model):
        model = _out(H.MiniMaxH3SigmaShift.execute(model, 10.0, 3.0))[0]
        sig = P.nodes.GmlxSigmas().get_sigmas(model)[0]
        pos = self.cond()[0]
        CS.SamplerCustom.execute(model, True, 42, 1.0, pos, pos,
                                 comfy.samplers.sampler_object("euler"), sig, self.latent())

    def test_the_loaders_int8_defaults_to_the_entrys_and_reaches_each_step(self):
        spec = P.nodes.GmlxModelLoader.INPUT_TYPES()["optional"]["int8"]
        self.assertEqual((spec[0], spec[1]["default"]),
                         (["entry", "off", "attention", "linears", "all"], "entry"))
        for entry, values in (("off", ("entry", "off", "attention")),
                              ("linears", ("entry", "linears", "all"))):
            self.stub.int8(entry)
            for value in values:
                with self.subTest(entry=entry, value=value):
                    self.stub.reset()
                    self.sample(self.loaded(value), steps=2)
                    self.assertEqual([e["int8"] for e in self.stub.events("av_step")],
                                     [None if value == "entry" else value] * 2)

    def test_a_value_whose_linears_differ_is_refused_at_the_load(self):
        """The status names the entry's setting, so the loader refuses
        before any step, with the server's text."""
        for entry, value, a, b in (("off", "linears", "off", "attention"),
                                   ("off", "all", "off", "attention"),
                                   ("linears", "off", "linears", "all"),
                                   ("all", "attention", "linears", "all")):
            self.stub.int8(entry)
            with self.subTest(entry=entry, value=value):
                with self.assertRaises(C.GmlxRefusal) as e:
                    self.loaded(value)
                self.assertEqual(str(e.exception), (
                    f"gmlx: this model's entry loads its DiT at int8 {entry}, so a request "
                    f"names int8 {a} or {b}, not {value}; set the entry's defaults.int8 to "
                    f"run {value}"))
        self.assertEqual(self.stub.events("av_step"), [])

    def test_a_status_without_the_entrys_setting_refuses_at_the_first_step(self):
        self.stub.int8("off", status=False)
        model = self.loaded("linears")
        with self.assertRaises(C.GmlxError) as e:
            self.first_step(model)
        self.assertEqual(str(e.exception), (
            "gmlx: the step failed (400): this model's entry loads its DiT at int8 off, so a "
            "request names int8 off or attention, not linears; set the entry's defaults.int8 "
            "to run linears"))
        self.assertEqual(self.stub.events("av_step"), [])

    def test_a_server_without_the_int8_setting_warns_once_a_model_past_entry_and_off(self):
        caps = self.client.capabilities
        had = caps["features"]
        caps["features"] = [f for f in had if f != "int8"]
        try:
            with self.assertLogs("gmlx", "WARNING") as logs:
                for model in (P.model.load_model(self.client, HID, "attention"),
                              P.model.load_model(self.client, HID, "attention")):
                    self.sample(model, steps=2)
                    self.sample(model, steps=1)
                for value in ("entry", "off"):
                    self.sample(P.model.load_model(self.client, HID, value), steps=1)
        finally:
            caps["features"] = had
        self.assertEqual([e["int8"] for e in self.stub.events("av_step")], [None] * 8)
        self.assertEqual(logs.output, [
            "WARNING:gmlx:gmlx: this gmlx server has no int8 setting, so 'h3' runs as its "
            "entry says; update it"] * 2)

    def test_fasth3_runs_int8_linears_and_refuses_int8_all(self):
        """FastH3 on an entry at int8 linears runs its steps at linears,
        and all is refused at the first step with the server's text."""
        self.stub.int8("linears")
        self.first_step(self.loaded("linears", FID))
        got = {e["int8"] for e in self.stub.events("av_step")}
        self.assertEqual(got, {"linears"})
        self.stub.reset()
        with self.assertRaises(C.GmlxError) as e:
            self.first_step(self.loaded("all", FID))
        self.assertEqual(str(e.exception), "gmlx: the step failed (400): MiniMax-H3 fasth3 runs "
                         "video sparse attention, and int8 attention applies to dense attention "
                         "only")
        self.assertEqual(self.stub.events("av_step"), [])

    def test_a_refused_int8_shows_the_servers_text(self):
        with self.assertRaises(C.GmlxError) as e:
            self.first_step(self.loaded("attention", FID))
        self.assertEqual(str(e.exception), "gmlx: the step failed (400): MiniMax-H3 fasth3 runs "
                         "video sparse attention, and int8 attention applies to dense attention "
                         "only")
        self.assertEqual(self.stub.events("av_step"), [])

    def test_int8_on_an_image_model_is_refused_at_the_load(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            self.loaded("attention", ZID)
        self.assertEqual(str(e.exception), "gmlx: int8 attention runs on MiniMax-H3, and "
                         "'z-image-turbo' is an image model; set int8 to entry")
        self.loaded("off", ZID)


class Interrupt(Base):
    """Cancel in ComfyUI during an AV step: the flag reaches the server
    within one poll and the run ends."""
    DELAY_S = 1.0
    stub_args = ("--step-delay-ms", str(int(DELAY_S * 1000)))

    def test_an_interrupt_reaches_the_server_and_stops_the_clip(self):
        model = self.model()
        done = threading.Event()

        def watch():
            while not done.is_set():
                # Once the third step is under way on the server, not between
                # steps, where ComfyUI's own check stops the run first.
                if len(self.stub.events("step_begin")) >= 3:
                    mm.interrupt_current_processing(True)
                    return
                time.sleep(0.02)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                self.sample(model, steps=8)
        finally:
            done.set()
            mm.interrupt_current_processing(False)
        # The stub logs the stopped step from its own thread.
        deadline = time.monotonic() + 30
        while "step_interrupted" not in (kinds := [e["kind"] for e in self.stub.log()]) \
                and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreaterEqual(kinds.count("interrupt"), 1)
        self.assertGreaterEqual(kinds.count("step_interrupted"), 1)
        self.assertIn("run_end", kinds)
        self.assertLess(kinds.count("av_step"), 8)

    def test_a_step_the_server_ends_after_a_cancel_ends_the_clip_as_comfyuis_interrupt(self):
        """The server can end the step with a 409 interrupted between the
        user's Cancel and the client's next poll. ComfyUI then shows its
        cancel, not a gmlx error. The client's poll is off, so only the 409
        tells the sampler."""
        model = self.model()
        other = C.StepClient(self.stub.url)
        done = threading.Event()

        def watch():
            while not done.is_set():
                if len(self.stub.events("step_begin")) >= 2:
                    mm.interrupt_current_processing(True)
                    other.interrupt(None)
                    return
                time.sleep(0.02)
        C.set_interrupt_check(None)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                self.sample(model, steps=8)
        finally:
            done.set()
            mm.interrupt_current_processing(False)
            C.set_interrupt_check(mm.throw_exception_if_processing_interrupted)
        self.assertIn("step_interrupted", [e["kind"] for e in self.stub.log()])


class EncodeInterrupt(Base):
    """Cancel in ComfyUI while the server rewrites the prompt: the
    interrupt naming the encode reaches the server, which ends the
    rewrite, and the node raises ComfyUI's interrupt."""
    stub_args = ("--encode-delay-ms", "5000")

    def test_a_cancel_during_the_rewrite_interrupts_that_encode_on_the_server(self):
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        done = threading.Event()

        def watch():
            while not done.is_set():
                if self.stub.events("encode_begin"):
                    mm.interrupt_current_processing(True)
                    return
                time.sleep(0.02)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        t0 = time.monotonic()
        try:
            with self.assertRaises(mm.InterruptProcessingException):
                comfy_nodes.CLIPTextEncode().encode(clip, "a fox")
        finally:
            done.set()
            mm.interrupt_current_processing(False)
        self.assertLess(time.monotonic() - t0, 3.0)
        (begin,) = self.stub.events("encode_begin")
        self.assertTrue(begin["interrupt_id"])
        self.assertEqual([e["run"] for e in self.stub.events("interrupt")],
                         [begin["interrupt_id"]])
        deadline = time.monotonic() + 10
        while not self.stub.events("encode_interrupted") and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(self.stub.events("encode_interrupted")), 1)


class Vaes(Base):
    def test_the_video_and_audio_vae_nodes_decode_a_sampled_clip(self):
        out = self.sample(self.model(), steps=1)
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        audio = P.nodes.GmlxAudioVAELoader().load(HID, self.stub.url, False)[0]
        (frames,) = comfy_nodes.VAEDecode().decode(video, out)
        self.assertEqual(tuple(frames.shape), (FRAMES, SIZE, SIZE, 3))
        track = vae_decode_audio(audio, out)
        a_t = out["samples"].unbind()[1].shape[-1]
        self.assertEqual(tuple(track["waveform"].shape), (1, 2, a_t * 800))
        self.assertEqual(track["sample_rate"], 32000)
        self.assertEqual([e["shape"][2] for e in self.stub.events("video_decode")],
                         [out["samples"].unbind()[0].shape[2]])

    def test_the_video_vae_counts_frames_and_latents_on_comfyuis_grid(self):
        """ComfyUI sizes a clip from the VAE's counts: 5 k + 2 latent frames
        for 17 k + 5 frames, as its MiniMax-H3 latent node counts them, and
        one latent frame for a still image."""
        vae = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        frames, latents = vae.upscale_ratio[0], vae.downscale_ratio[0]
        for n in range(5, 400):
            t = H.video_latent_t(n)
            self.assertEqual(latents(n), t, n)
            self.assertEqual(frames(t), 17 * ((n - 5) // 17) + 5, n)
        self.assertEqual((latents(1), frames(1)), (1, 1))
        self.assertEqual((vae.upscale_ratio[1:], vae.downscale_ratio[1:]), ((16, 16), (16, 16)))

    def test_the_stock_vae_encode_node_crops_a_frame_to_whole_latents(self):
        """A loaded image reaches the video VAE through the stock VAE
        Encode node; the encode cuts it to whole latents about its middle
        and drops its alpha, as ComfyUI's encode does."""
        vae = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        image = torch.rand(1, SIZE + 12, SIZE + 4, 4)
        (out,) = comfy_nodes.VAEEncode().encode(vae, image)
        self.assertEqual(out["samples"].tolist(),
                         vae.encode(image[:, 6:6 + SIZE, 2:2 + SIZE, :3]).tolist())
        self.assertEqual([e["shape"] for e in self.stub.events("video_encode")],
                         [[1, SIZE, SIZE, 3]] * 2)

    def test_each_vae_loader_refuses_the_other_kind_of_model(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxVAELoader().load(HID, self.stub.url, False)
        self.assertEqual(str(e.exception), "gmlx: 'h3' has a video VAE and an audio VAE; load "
                         "them with the gmlx Video VAE Loader and the gmlx Audio VAE Loader")
        for node in (P.nodes.GmlxVideoVAELoader, P.nodes.GmlxAudioVAELoader):
            with self.subTest(node.__name__), self.assertRaises(C.GmlxRefusal) as e:
                node().load(ZID, self.stub.url, False)
            self.assertIn("'z-image-turbo' is an image model", str(e.exception))


class GuideWire(Base):
    """A guide clip on the wire: uint8 levels when its frames are such
    levels exactly, float32 otherwise, and a long clip in windows of whole
    pieces that tile it in order, each within the server's body limit, a
    window refused for memory sent again once at the window named."""

    def setUp(self):
        super().setUp()
        self.stub.video_encode()
        self.vae = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        self.vae.gmlx_client.handshake()

    def windows(self):
        return [(e["shape"][0], e["at"], e["open"], e["dtype"])
                for e in self.stub.events("video_encode")]

    def test_frames_go_as_uint8_only_when_they_are_levels_exactly(self):
        """Levels k / 255, as an image read gives them, go as uint8 and
        come back as the same values; a frame of 0.5 is no level."""
        levels = torch.randint(0, 256, (5, SIZE, SIZE, 3)).float() / 255
        self.assertEqual(C.wire_frames(levels).dtype, torch.uint8)
        self.assertTrue(torch.equal(C.wire_frames(levels).float() / 255, levels))
        half = levels.clone()
        half[2, 3, 4, 1] = 0.5
        self.assertEqual(C.wire_frames(half).dtype, torch.float32)
        self.vae.encode(levels)
        self.vae.encode(half)
        self.assertEqual(self.windows(), [(5, 0, False, "U8"), (5, 0, False, "F32")])

    def test_add_guide_frames_are_levels_and_go_as_uint8(self):
        """MiniMaxH3AddGuide resizes a clip through PIL's Lanczos at 8 bits,
        so its frames are levels and go as uint8."""
        pos, _ = self.cond()
        _out(H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, vae=self.vae,
                                         image=torch.rand(FRAMES, SIZE + 8, SIZE, 3)))
        self.assertEqual(self.windows(), [(FRAMES, 0, False, "U8")])

    def clip(self, n):
        """``n`` frames whose pieces of 17 frames each hold one level, its
        index plus 1, so a latent's value names its piece."""
        piece = torch.arange(n) // 17 + 1
        return (piece.float() / 255)[:, None, None, None].expand(n, SIZE, SIZE, 3).contiguous()

    def assert_tiles(self, lat, n):
        """The clip's 5 k + 2 latents, each window's in turn: the stub
        gives a window's latents its frames' mean level, which differs
        from window to window, so a window out of place shows."""
        self.assertEqual(lat.shape[2], 5 * -(-n // 17) - 3)
        clip, want = self.clip(n), []
        for size, at, is_open, _ in self.windows():
            level = clip[at:at + size, 0, 0, 0].mean().item()
            want += [level] * (size // 17 * 5 + (0 if is_open else 2))
        got = lat[0, 0, :, 0, 0].tolist()
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=5)

    def test_a_362_frame_guide_goes_in_three_windows(self):
        """The longest guide ComfyUI makes at 15 s: windows of 7 pieces,
        two open ones of 119 frames and the last 124, uint8, to 107
        latents."""
        lat = self.vae.encode(self.clip(362))
        self.assertEqual(self.windows(), [(119, 0, True, "U8"), (119, 119, True, "U8"),
                                          (124, 238, False, "U8")])
        self.assert_tiles(lat, 362)

    def test_a_window_fits_the_servers_body_limit(self):
        """Float32 frames past the body limit at 7 pieces go as many pieces
        a window as fit beside the 5 frames of the clip's end and the
        header: 2 at 39 frames of 49152 bytes, 1 at 36."""
        frames = self.clip(73)
        frames[0, 0, 0, 0] = 0.5
        for fit, want in ((39, [(34, 0, True), (39, 34, False)]),
                          (36, [(17, 0, True), (17, 17, True), (17, 34, True),
                                (22, 51, False)])):
            with self.subTest(fit=fit):
                self.stub.reset()
                self.stub.video_encode(max_body=(1 << 20) + fit * SIZE * SIZE * 3 * 4)
                self.vae.gmlx_client.handshake()
                lat = self.vae.encode(frames)
                self.assertEqual([w[:3] for w in self.windows()], want)
                self.assertEqual({w[3] for w in self.windows()}, {"F32"})
                self.assertEqual(lat.shape[2], 22)

    def test_a_window_refused_for_memory_goes_again_at_the_window_named(self):
        """The first window is refused from its header, naming 3 pieces;
        the clip then goes 51 frames a window, in order, once more."""
        self.stub.video_encode(refuse=[3])
        lat = self.vae.encode(self.clip(362))
        (refused,) = self.stub.events("video_encode_refused")
        self.assertEqual((refused["shape"][0], refused["at"], refused["window"]), (119, 0, 3))
        self.assertEqual([w[:3] for w in self.windows()],
                         [(51, at, True) for at in range(0, 306, 51)] + [(56, 306, False)])
        self.assert_tiles(lat, 362)

    def test_a_later_window_refused_goes_again_from_its_own_frame(self):
        """The second window is refused, naming 3 pieces: the first window,
        encoded already, goes once, and the rest of the clip goes from the
        refused window's frame 119, 51 frames a window."""
        self.stub.video_encode(refuse=[None, 3])
        lat = self.vae.encode(self.clip(362))
        (refused,) = self.stub.events("video_encode_refused")
        self.assertEqual((refused["shape"][0], refused["at"], refused["window"]), (119, 119, 3))
        self.assertEqual([w[:3] for w in self.windows()],
                         [(119, 0, True)] + [(51, at, True) for at in range(119, 323, 51)]
                         + [(39, 323, False)])
        self.assert_tiles(lat, 362)

    def test_a_second_refusal_shows_the_servers_text(self):
        """A window refused again after the retry, or one naming no window
        that fits, stops with the server's text, drained and answered, not
        a lost connection."""
        for refuse in ([3, 2], [0]):
            with self.subTest(refuse=refuse):
                self.stub.reset()
                self.stub.video_encode(refuse=refuse)
                with self.assertRaises(C.GmlxError) as e:
                    self.vae.encode(self.clip(362))
                self.assertIn("does not fit in memory beside the models now loaded",
                              str(e.exception))
                self.assertNotIn("did not answer", str(e.exception))
                self.assertEqual(self.windows(), [])

def _rows_noise(z: torch.Tensor, seed: int) -> list:
    """The noise ComfyUI's forward mixes into the rows of video latent
    ``z`` at ``seed``: a CPU generator restarted for it, in the rows'
    shape at the DiT's patch (1, 2, 2)."""
    _, c, t, h, w = z.shape
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((t * (h // 2) * (w // 2), c * 4), generator=g).tolist()


class Anchors(Base):
    """Keyframes and guides as the stock nodes make them, through the gmlx
    CLIP and VAEs, reach the AV step: each anchor's frame and latents by
    hash, the noise of its video rows and the prompt's vision spans."""

    def nodes(self, mid=HID):
        clip = P.nodes.GmlxCLIPLoader().load(mid, self.stub.url, False)[0]
        video = P.nodes.GmlxVideoVAELoader().load(mid, self.stub.url, False)[0]
        audio = P.nodes.GmlxAudioVAELoader().load(mid, self.stub.url, False)[0]
        return clip, video, audio

    def step_of(self, pos, latent=None, seed=1):
        (out,) = comfy_nodes.common_ksampler(self.model(), seed, 1, 1.0, "res_multistep", "simple",
                                             pos, pos, latent or self.latent())
        (step,) = self.stub.events("av_step")
        return step

    def test_image_to_video_sends_its_frames_to_the_encoder_and_the_step(self):
        """MiniMaxH3ImageToVideo shows the encoder its first frame
        stretched and its last cover-cropped to the canvas, and its VAE
        encodes them. The encoder's vision spans come back as ComfyUI's
        tags, and the step anchors the frames at 0 and the clip's last,
        each with the noise of its rows at the sampler's seed."""
        clip, video, _a = self.nodes()
        first, last = torch.rand(1, 48, 80, 3), torch.rand(1, 80, 48, 4)
        pos, latent = _out(H.MiniMaxH3ImageToVideo.execute(clip, video, "a dog", SIZE, SIZE,
                                                           FRAMES, first_frame=first,
                                                           last_frame=last))
        (enc,) = self.stub.events("encode")
        shown = [H._resize(first[:1], SIZE, SIZE, "disabled"),
                 H._resize(last[:1], SIZE, SIZE, "center")]
        self.assertEqual(enc["image_shapes"], [[SIZE, SIZE, 3]] * 2)
        self.assertEqual(enc["image_sums"], [
            int(im.clamp(0, 1).mul(255).round().to(torch.uint8).long().sum()) for im in shown])
        tags = pos[0][1]["minimax_token_tags"]
        self.assertEqual((tags.dtype, tags.numel()), (torch.long, pos[0][0].shape[1]))
        self.assertEqual(P.patcher.tag_spans(tags), enc["spans"])
        self.assertEqual(enc["spans"], [[3, 9], [12, 18]])
        self.assertEqual([e["shape"] for e in self.stub.events("video_encode")],
                         [[1, SIZE, SIZE, 3]] * 2)
        keyframes = pos[0][1]["minimax_keyframes"]
        step = self.step_of(pos, latent, seed=11)
        self.assertEqual([a["frame"] for a in step["anchors"]], [0, FRAMES - 1])
        self.assertEqual(step["spans"], enc["spans"])
        for a, kf in zip(step["anchors"], keyframes):
            got = step["anchor_tensors"]
            self.assertEqual(sorted(a), ["frame", "noise", "video"])
            self.assertEqual(got[a["video"]], kf["latent"].tolist())
            self.assertEqual(got[a["noise"]], _rows_noise(kf["latent"], 11))

    def test_a_guide_image_clip_and_sound_reach_the_step(self):
        """MiniMaxH3AddGuide anchors an image, a clip cut to 17 k + 5
        frames and a sound cut to the clip's audio from its frame; the
        gmlx VAEs encode them, the sound mono on both channels and cut to
        whole latents about its middle, as ComfyUI's encode does."""
        clip, video, audio = self.nodes()
        pos, _ = self.cond()
        latent = self.latent()
        image = torch.rand(1, SIZE, SIZE, 3)
        frames = torch.rand(FRAMES + 3, SIZE, SIZE, 3)
        wave = torch.rand(1, 1, 3000) * 2 - 1
        track = {"waveform": wave, "sample_rate": 32000}
        pos = _out(H.MiniMaxH3AddGuide.execute(pos, latent, -1, vae=video, image=image))[0]
        pos = _out(H.MiniMaxH3AddGuide.execute(pos, latent, 0, vae=video, image=frames,
                                               audio_vae=audio, audio=track))[0]
        self.assertEqual([e["shape"] for e in self.stub.events("video_encode")],
                         [[1, SIZE, SIZE, 3], [FRAMES, SIZE, SIZE, 3]])
        self.assertEqual([e["shape"] for e in self.stub.events("audio_encode")], [[1, 2, 2400]])
        keyframes = pos[0][1]["minimax_keyframes"]
        a_t = latent["samples"].unbind()[1].shape[-1]
        self.assertEqual([tuple(k["latent"].shape) for k in keyframes],
                         [(1, 24, 1, 4, 4), (1, 24, 2, 4, 4)])
        sound = keyframes[1]["audio_latent"]
        self.assertEqual(tuple(sound.shape), (1, 32, 2, min(3, a_t)))
        mid = wave[0, 0, 300:2700].mean().item()
        self.assertAlmostEqual(sound[0, 0, 1, 0].item(), mid, places=5)
        step = self.step_of(pos, latent)
        self.assertEqual([a["frame"] for a in step["anchors"]], [FRAMES - 1, 0])
        self.assertEqual(sorted(step["anchors"][1]), ["audio", "frame", "noise", "video"])
        got = step["anchor_tensors"]
        self.assertEqual(got[step["anchors"][1]["audio"]], sound.tolist())
        self.assertEqual(got[step["anchors"][1]["noise"]], _rows_noise(keyframes[1]["latent"], 1))
        self.assertEqual(step["spans"], [])

    def test_a_guide_from_a_stock_vae_and_a_stock_encoders_tags_step(self):
        """A stock VAE's latent in minimax_keyframes and a stock CLIP's tags
        reach the step as the gmlx nodes' do: tags of text and vision as
        the spans of their vision runs."""
        pos, _ = self.cond()
        n = pos[0][0].shape[1]
        tags = torch.ones(n, dtype=torch.long)
        tags[3:9] = 0
        z = torch.rand(1, 24, 1, SIZE // 16, SIZE // 16)
        pos = node_helpers.conditioning_set_values(pos, {
            "minimax_token_tags": tags,
            "minimax_keyframes": [{"resolved_frame_index": 2, "latent": z}]})
        step = self.step_of(pos)
        self.assertEqual(step["spans"], [[3, 9]])
        (a,) = step["anchors"]
        self.assertEqual((a["frame"], step["anchor_tensors"][a["video"]]), (2, z.tolist()))

    def test_an_anchor_the_server_dropped_is_put_again(self):
        """A 409 names the anchors' tensors too; the step puts them and
        runs once."""
        pos, _ = self.cond()
        z = torch.rand(1, 24, 1, SIZE // 16, SIZE // 16)
        payload = {"audio_scale": 2.0, "seed": 3,
                   "keyframes": [{"resolved_frame_index": 0, "latent": z}]}
        model = self.model()
        mm.load_models_gpu([model])
        model.pre_run()
        video, audio = self.latent()["samples"].unbind()
        to = {"cond_or_uncond": [0], "uuids": [uuid.uuid4()]}
        f = model.model.diffusion_model.forward
        try:
            f([video, audio], torch.tensor([600.0]), context=pos[0][0],
              transformer_options=dict(to), minimax_payload=dict(payload))
            self.stub.forget()
            self.stub.reset()
            f([video, audio], torch.tensor([400.0]), context=pos[0][0],
              transformer_options=dict(to), minimax_payload=dict(payload))
        finally:
            model.cleanup()
        kinds = [e["kind"] for e in self.stub.log()]
        self.assertEqual(kinds[0], "step_missing")
        self.assertEqual((kinds.count("put_tensor"), kinds.count("av_step")), (3, 1))

    def test_a_server_without_anchors_refuses_keyframes(self):
        pos, _ = self.cond()
        z = torch.rand(1, 24, 1, SIZE // 16, SIZE // 16)
        pos = node_helpers.conditioning_set_values(pos, {
            "minimax_keyframes": [{"resolved_frame_index": 0, "latent": z}]})
        model = self.model()
        caps = model.gmlx.client.capabilities
        had = caps["features"]
        caps["features"] = [f for f in had if f != "anchors"]
        try:
            with self.assertRaises(C.GmlxRefusal) as e:
                comfy_nodes.common_ksampler(model, 1, 1, 1.0, "res_multistep", "simple", pos,
                                            pos, self.latent())
        finally:
            caps["features"] = had
        self.assertEqual(str(e.exception), "gmlx: this gmlx server runs no keyframes or guides; "
                         "update it")
        self.assertEqual(self.stub.events("av_step"), [])


class KeyframeCaptions(Base):
    """GmlxH3ImageToVideo runs ComfyUI's MiniMaxH3ImageToVideo and tells
    the encode the clip's length and the role of each frame, which an
    upsampler on the server's entry rewrites the prompt with."""

    stub_args = ("--upsample",)
    FIRST, LAST = torch.rand(1, 48, 80, 3), torch.rand(1, 80, 48, 3)

    def nodes(self):
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        return clip, video

    def run_node(self, node, clip, video, **frames):
        out = _out(node(clip, video, "a dog", SIZE, SIZE, FRAMES, **frames))
        (enc,) = self.stub.events("encode")
        self.stub.reset()
        return out, enc

    def test_the_node_tells_the_encode_the_clips_length_and_each_frames_role(self):
        """Each frame's role reaches the encode, a last frame alone too,
        with the clip's frame count. The encoder, the VAE and the step get
        what ComfyUI's own node gives them, which sends neither."""
        gmlx = P.nodes.GmlxH3ImageToVideo().encode
        for frames, roles in (({"first_frame": self.FIRST}, ["first"]),
                              ({"last_frame": self.LAST}, ["last"]),
                              ({"first_frame": self.FIRST, "last_frame": self.LAST},
                               ["first", "last"])):
            with self.subTest(roles=roles):
                clip, video = self.nodes()
                self.stub.reset()
                (pos, latent), enc = self.run_node(gmlx, clip, video, **frames)
                (want, want_latent), stock = self.run_node(H.MiniMaxH3ImageToVideo.execute,
                                                           clip, video, **frames)
                self.assertEqual((enc["frames"], enc["keyframes"]), (FRAMES, roles))
                self.assertEqual((stock["frames"], stock["keyframes"]), (None, None))
                for k in ("image_shapes", "image_sums", "spans"):
                    self.assertEqual(enc[k], stock[k], k)
                got, wanted = pos[0][1]["minimax_keyframes"], want[0][1]["minimax_keyframes"]
                self.assertEqual([kf["resolved_frame_index"] for kf in got],
                                 [kf["resolved_frame_index"] for kf in wanted])
                for a, b in zip(got, wanted):
                    self.assertTrue(torch.equal(a["latent"], b["latent"]))
                self.assertTrue(torch.equal(pos[0][1]["minimax_token_tags"],
                                            want[0][1]["minimax_token_tags"]))
                self.assertEqual(latent["samples"].shape, want_latent["samples"].shape)

    def test_a_keyframe_prompt_samples_on_the_servers_caption(self):
        """On an entry with an upsampler, the prompt with a first frame is
        encoded as the caption the server wrote for the clip's length and
        keyframe, and the step samples on that caption's conditioning."""
        clip, video = self.nodes()
        (pos, latent), enc = self.run_node(P.nodes.GmlxH3ImageToVideo().encode, clip, video,
                                           first_frame=self.FIRST)
        self.assertEqual(enc["revised"], f"a caption of {FRAMES} frames, ['first']: a dog")
        (_out_,) = comfy_nodes.common_ksampler(P.model.load_model(self.client, HID), 1, 1, 1.0,
                                               "res_multistep", "simple", pos, pos, latent)
        (step,) = self.stub.events("av_step")
        self.assertEqual(step["conds"], [enc["cond"]])

    def test_a_server_without_keyframe_upsampling_encodes_as_before_with_one_warning(self):
        """A server that takes no clip length or keyframes gets the encode
        ComfyUI's own node sends, and the CLIP warns once that the prompt
        with keyframes runs as typed."""
        clip, video = self.nodes()
        caps = clip.gmlx_client.capabilities
        had = caps["features"]
        caps["features"] = [f for f in had if f != "keyframe_upsampling"]
        try:
            with self.assertLogs("gmlx", "WARNING") as logs:
                encs = [self.run_node(P.nodes.GmlxH3ImageToVideo().encode, clip, video,
                                      first_frame=self.FIRST)[1] for _ in range(2)]
        finally:
            caps["features"] = had
        self.assertEqual([(e["frames"], e["keyframes"], e["revised"]) for e in encs],
                         [(None, None, None)] * 2)
        self.assertEqual(logs.output, [
            "WARNING:gmlx:gmlx: this gmlx server does not tell its upsampler the clip's "
            "length and keyframes, so a prompt with keyframes runs as typed; update it"])

    def test_comfyuis_own_node_with_keyframes_is_noted_once(self):
        """ComfyUI's own node sends keyframes with no clip length or roles:
        the CLIP says once that the gmlx node sends both."""
        clip, video = self.nodes()
        with self.assertLogs("gmlx", "WARNING") as logs:
            for _ in range(2):
                self.run_node(H.MiniMaxH3ImageToVideo.execute, clip, video,
                              first_frame=self.FIRST)
        self.assertEqual(logs.output, [
            "WARNING:gmlx:gmlx: ComfyUI's MiniMax H3 Image to Video sends the server no clip "
            "length or keyframes; gmlx MiniMax-H3 Image to Video sends both, and an upsampler "
            "on the server's entry rewrites the prompt with them"])


class Refusals(Base):
    """What a gmlx model does not run of what MiniMax-H3 conditions on,
    each as the stock node that adds it builds it, and the Fun
    ControlNet."""

    def refused(self, want: str, model=None, **kw):
        with self.assertRaises(C.GmlxRefusal) as e:
            self.sample(model or self.model(), steps=1, **kw)
        self.assertIn(want, str(e.exception))
        self.assertEqual(self.stub.events("av_step"), [])
        return str(e.exception)

    def test_references_are_refused_by_name(self):
        """MiniMaxH3ReferenceToVideo with a stock VAE puts its encoded
        images in minimax_refs."""
        pos, _ = self.cond()
        ref = [{"kind": "image", "latent_h": SIZE // 16, "latent_w": SIZE // 16,
                "latent": torch.zeros(1, 24, 1, SIZE // 16, SIZE // 16)}]
        pos = node_helpers.conditioning_set_values(pos, {"minimax_refs": ref})
        self.assertIn(NO_REFS, self.refused("references (minimax_refs", pos=pos))

    def test_tags_other_than_text_and_vision_are_refused(self):
        pos, _ = self.cond()
        tags = torch.ones(pos[0][0].shape[1], dtype=torch.long)
        tags[3:9] = 2
        pos = node_helpers.conditioning_set_values(pos, {"minimax_token_tags": tags})
        self.refused("gmlx: the prompt's minimax_token_tags hold tags other than text (1) and "
                     "vision (0), which no stock node makes", pos=pos)

    def test_tags_of_another_prompt_are_refused(self):
        """ConditioningConcat joins the features of two prompts and keeps
        the first one's tags."""
        pos, _ = self.cond()
        n = pos[0][0].shape[1]
        pos = node_helpers.conditioning_set_values(pos, {
            "minimax_token_tags": torch.ones(n - 1, dtype=torch.long)})
        with self.assertRaises(C.GmlxError) as e:
            self.sample(self.model(), pos=pos, steps=1)
        self.assertEqual(str(e.exception), f"gmlx: the prompt's minimax_token_tags hold {n - 1} "
                         f"tags for its {n} rows; encode the prompt and its keyframes with one "
                         "text encode node")
        self.assertEqual(self.stub.events("av_step"), [])

    def test_an_anchor_noise_mix_of_a_custom_node_is_refused_by_name(self):
        """No stock node sets the cond noise augs; the server mixes the
        anchor rows at ComfyUI's defaults."""
        pos, _ = self.cond()
        for key, want in (("minimax_visual_cond_noise_aug", "a video anchor noise mix of 0.99 "
                           "(minimax_payload['visual_cond_noise_aug']"),
                          ("minimax_audio_cond_noise_aug", "an audio anchor noise mix of 0.99 "
                           "(minimax_payload['audio_cond_noise_aug']")):
            with self.subTest(key):
                self.stub.reset()
                p = node_helpers.conditioning_set_values(pos, {key: 0.99})
                self.refused(want, pos=p)

    def test_fasth3_refuses_keyframes_and_guides_by_name(self):
        """FastH3 runs text to video only: a guide, or keyframes in the
        prompt, stop before any upload."""
        pos, _ = self.cond()
        z = torch.rand(1, 24, 1, SIZE // 16, SIZE // 16)
        tags = torch.ones(pos[0][0].shape[1], dtype=torch.long)
        tags[3:9] = 0
        for given, held in (({"minimax_keyframes": [{"resolved_frame_index": 0, "latent": z}]},
                             "1 keyframe or guide"),
                            ({"minimax_token_tags": tags}, "keyframes in its prompt")):
            with self.subTest(held):
                self.stub.reset()
                p = node_helpers.conditioning_set_values(pos, given)
                text = self.refused("gmlx: 'fasth3' is FastH3", self.model(FID), pos=p)
                self.assertEqual(text, "gmlx: 'fasth3' is FastH3, which runs text to video "
                                 f"only, and this run has {held}; run it without keyframes or "
                                 "guides, or on h3")
                self.assertEqual(self.stub.events("put_tensor"), [])

    def test_a_noise_mask_on_either_stream_is_refused_by_name(self):
        """SetLatentNoiseMask masks the video; a mask beside an audio
        track, as LTXVConcatAVLatent joins them, masks the audio."""
        for stream in ("denoise_mask", "audio_denoise_mask"):
            with self.subTest(stream):
                latent = self.latent()
                v, a = latent["samples"].unbind()
                if stream == "denoise_mask":
                    mask = torch.ones(1, SIZE, SIZE)
                    mask[:, :, : SIZE // 2] = 0.0
                    latent = _out(comfy_nodes.SetLatentNoiseMask().set_mask(latent, mask))[0]
                else:
                    keep = torch.ones(a.shape[0], 1, *a.shape[2:])
                    keep[..., :2] = 0.0
                    latent = {**latent, "noise_mask": comfy.nested_tensor.NestedTensor(
                        (torch.ones(v.shape[0], 1, *v.shape[2:]), keep))}
                text = self.refused(f"a noise mask on the MiniMax-H3 latent reached the DiT "
                                    f"forward as {stream}", latent=latent)
                self.assertIn("samples the whole clip", text)

    def test_the_fun_controlnet_is_refused_by_name(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3FunControlNetApply.execute(
                self.model(), model_patch=None, vae=None, strength=1.0, start_percent=0.0,
                end_percent=1.0, control_video=torch.zeros(FRAMES, SIZE, SIZE, 3))
        self.assertEqual(str(e.exception), "gmlx: the MiniMax-H3 Fun ControlNet "
                         "(MiniMaxH3FunControlNetApply) runs inside the DiT forward and is not "
                         "supported on a gmlx model")

    def test_references_on_the_gmlx_nodes_are_refused_by_name(self):
        """MiniMaxH3ReferenceToVideo shows its references to the gmlx CLIP,
        after its VAE encoded them when it has one."""
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        image = torch.rand(1, SIZE, SIZE, 3)
        frames = torch.rand(FRAMES, SIZE, SIZE, 3)
        for given in ({"ref_images": {"ref_image_1": image}},
                      {"vae": video, "ref_videos": {"ref_video_1": frames}}):
            with self.subTest(sorted(given)), self.assertRaises(C.GmlxRefusal) as e:
                H.MiniMaxH3ReferenceToVideo.execute(clip, "a dog", SIZE, SIZE, FRAMES, **given)
            self.assertEqual(str(e.exception), "gmlx: references (minimax_ref_items, from "
                             f"MiniMaxH3ReferenceToVideo) reached a gmlx CLIP; {NO_REFS}")
        self.assertEqual(self.stub.events("encode"), [])

    def test_a_keyframe_of_keys_the_step_does_not_read_is_refused(self):
        """A keyframe that names none of the keys the step reads, or a
        latent with no frame, is refused, not dropped. ComfyUI's own layout
        reads each keyframe's frame first, so the payload goes to the DiT
        forward as a sampler calls it. A frame with no latent anchors
        nothing, as in ComfyUI's forward."""
        z = torch.rand(1, 24, 1, SIZE // 16, SIZE // 16)

        def forward(keyframes):
            model = self.model()
            mm.load_models_gpu([model])
            model.pre_run()
            pos, _ = self.cond()
            video, audio = self.latent()["samples"].unbind()
            try:
                model.model.diffusion_model.forward(
                    [video, audio], torch.tensor([999.9]), context=pos[0][0],
                    transformer_options={"cond_or_uncond": [0], "uuids": [uuid.uuid4()]},
                    minimax_payload={"audio_scale": 10.0 / 3.0, "keyframes": keyframes})
            finally:
                model.cleanup()

        for kf in ({"frame_index": 0, "video_latent": z}, {"latent": z}):
            with self.subTest(sorted(kf)):
                self.stub.reset()
                with self.assertRaises(Exception) as e:
                    forward([kf])
                self.assertIsInstance(e.exception, C.GmlxError)
                self.assertIn(f"gmlx: a MiniMax-H3 keyframe holds {sorted(kf)}, and the gmlx "
                              "nodes read resolved_frame_index, latent, audio_latent",
                              str(e.exception))
                self.assertEqual(self.stub.events("av_step"), [])
        self.stub.reset()
        forward([{"resolved_frame_index": 4}, {"resolved_frame_index": 0, "latent": z}])
        (step,) = self.stub.events("av_step")
        (a,) = step["anchors"]
        self.assertEqual((a["frame"], step["anchor_tensors"][a["video"]]), (0, z.tolist()))

    def running(self, node):
        """Run the test body as if ComfyUI ran the stock node ``node``."""
        real = P.model._running
        P.model._running = lambda: ({"9": {"class_type": node, "inputs": {}}}, "9")
        self.addCleanup(setattr, P.model, "_running", real)

    def test_references_are_refused_before_the_vaes_encode_them(self):
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        audio = P.nodes.GmlxAudioVAELoader().load(HID, self.stub.url, False)[0]
        self.running("MiniMaxH3ReferenceToVideo")
        sound = {"waveform": torch.rand(1, 2, 32000), "sample_rate": 32000}
        for given in ({"vae": video, "ref_images": {"ref_image_1": torch.rand(1, SIZE, SIZE, 3)}},
                      {"audio_vae": audio, "ref_audios": {"ref_audio_1": sound}}):
            with self.subTest(sorted(given)), self.assertRaises(C.GmlxRefusal) as e:
                H.MiniMaxH3ReferenceToVideo.execute(clip, "a dog", SIZE, SIZE, FRAMES, **given)
            self.assertEqual(str(e.exception), "gmlx: references (from MiniMaxH3ReferenceToVideo) "
                             f"reached a gmlx VAE; {NO_REFS}")
        self.assertEqual(self.stub.events("video_encode") + self.stub.events("audio_encode"), [])

    def test_fasth3_refuses_a_guide_before_its_vae_encodes_it(self):
        video = P.nodes.GmlxVideoVAELoader().load(FID, self.stub.url, False)[0]
        pos, _ = self.cond()
        self.running("MiniMaxH3AddGuide")
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, vae=video,
                                        image=torch.rand(1, SIZE, SIZE, 3))
        self.assertEqual(str(e.exception), "gmlx: 'fasth3' is FastH3, which runs text to video "
                         "only, and MiniMaxH3AddGuide encodes a keyframe or guide on its VAE; "
                         "run it without keyframes or guides, or on h3")
        self.assertEqual(self.stub.events("video_encode"), [])

    def test_a_plain_vae_encode_on_fasth3_and_a_guide_on_h3_are_sent(self):
        image = torch.rand(1, SIZE, SIZE, 3)
        self.running("VAEEncode")
        fast = P.nodes.GmlxVideoVAELoader().load(FID, self.stub.url, False)[0]
        comfy_nodes.VAEEncode().encode(fast, image)
        self.running("MiniMaxH3AddGuide")
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        H.MiniMaxH3AddGuide.execute(self.cond()[0], self.latent(), 0, vae=video, image=image)
        self.assertEqual(len(self.stub.events("video_encode")), 2)

    def test_fasth3_refuses_keyframes_at_the_encode(self):
        clip = P.nodes.GmlxCLIPLoader().load(FID, self.stub.url, False)[0]
        video = P.nodes.GmlxVideoVAELoader().load(FID, self.stub.url, False)[0]
        with self.assertRaises(C.GmlxError) as e:
            H.MiniMaxH3ImageToVideo.execute(clip, video, "a dog", SIZE, SIZE, FRAMES,
                                            first_frame=torch.rand(1, SIZE, SIZE, 3))
        self.assertIn("fasth3 runs text to video only", str(e.exception))


if __name__ == "__main__":
    unittest.main()
