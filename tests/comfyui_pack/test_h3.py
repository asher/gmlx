"""MiniMax-H3 and FastH3 through the pack against the stub server: the
weightless model and the ComfyUI keys its forward reads, a sample of
ComfyUI's nested video and audio latent through the AV step with the shifts
and audio scale ComfyUI's forward reads, FastH3 on its trained sigmas, a
LoRA from the stock loader, the 409 retry and the interrupt of an AV step,
the video and audio VAE nodes, and the refusals of what MiniMax-H3
conditions on beside the text prompt and the Fun ControlNet. Every ComfyUI
the pack supports has MiniMax-H3, which came in v0.30.0.

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
TEXT_ONLY = "the gmlx nodes run MiniMax-H3 from the text prompt alone"
VIDEO_VAE = ("gmlx: a gmlx video VAE decodes and does not encode, so it takes no first frame, "
             "last frame, guide image or guide clip, reference image or reference video; "
             f"{TEXT_ONLY}")


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

    def test_each_vae_loader_refuses_the_other_kind_of_model(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            P.nodes.GmlxVAELoader().load(HID, self.stub.url, False)
        self.assertEqual(str(e.exception), "gmlx: 'h3' has a video VAE and an audio VAE; load "
                         "them with the gmlx Video VAE Loader and the gmlx Audio VAE Loader")
        for node in (P.nodes.GmlxVideoVAELoader, P.nodes.GmlxAudioVAELoader):
            with self.subTest(node.__name__), self.assertRaises(C.GmlxRefusal) as e:
                node().load(ZID, self.stub.url, False)
            self.assertIn("'z-image-turbo' is an image model", str(e.exception))


class Refusals(Base):
    """What MiniMax-H3 conditions on beside the text prompt, each as the
    stock node that adds it builds it, and the Fun ControlNet."""

    def refused(self, want: str, **kw):
        with self.assertRaises(C.GmlxRefusal) as e:
            self.sample(self.model(), steps=1, **kw)
        self.assertIn(want, str(e.exception))
        self.assertIn(TEXT_ONLY if "noise mask" not in want else "samples the whole clip",
                      str(e.exception))
        self.assertEqual(self.stub.events("av_step"), [])

    def test_keyframes_and_guides_are_refused_by_name(self):
        """MiniMaxH3AddGuide with a stock VAE puts the encoded frame in
        minimax_keyframes."""
        pos, _ = self.cond()
        guide = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, SIZE // 16, SIZE // 16)}]
        pos = node_helpers.conditioning_set_values(pos, {"minimax_keyframes": guide})
        self.refused("keyframes or guides (minimax_keyframes", pos=pos)

    def test_references_are_refused_by_name(self):
        """MiniMaxH3ReferenceToVideo with a stock VAE puts its encoded
        images in minimax_refs."""
        pos, _ = self.cond()
        ref = [{"kind": "image", "latent_h": SIZE // 16, "latent_w": SIZE // 16,
                "latent": torch.zeros(1, 24, 1, SIZE // 16, SIZE // 16)}]
        pos = node_helpers.conditioning_set_values(pos, {"minimax_refs": ref})
        self.refused("references (minimax_refs", pos=pos)

    def test_vision_tokens_in_the_prompt_are_refused(self):
        """A stock CLIP shown references without a VAE tags their vision
        tokens 0 and adds no minimax_refs."""
        pos, _ = self.cond()
        n = pos[0][0].shape[1]
        tags = torch.ones(n, dtype=torch.long)
        tags[3:9] = 0
        pos = node_helpers.conditioning_set_values(pos, {"minimax_token_tags": tags})
        self.refused("vision tokens (minimax_token_tags)", pos=pos)

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
                self.refused(f"a noise mask on the MiniMax-H3 latent reached the DiT forward "
                             f"as {stream}", latent=latent)

    def test_the_fun_controlnet_is_refused_by_name(self):
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3FunControlNetApply.execute(
                self.model(), model_patch=None, vae=None, strength=1.0, start_percent=0.0,
                end_percent=1.0, control_video=torch.zeros(FRAMES, SIZE, SIZE, 3))
        self.assertEqual(str(e.exception), "gmlx: the MiniMax-H3 Fun ControlNet "
                         "(MiniMaxH3FunControlNetApply) runs inside the DiT forward and is not "
                         "supported on a gmlx model")

    def test_a_first_or_last_frame_on_the_gmlx_nodes_is_refused_by_name(self):
        """MiniMaxH3ImageToVideo shows its frames to the text encoder before
        its VAE encodes them, so with the gmlx CLIP and video VAE wired, as
        the example workflow is, the CLIP refuses them before any encode."""
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        vae = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        image = torch.rand(1, SIZE, SIZE, 3)
        for frame in ("first_frame", "last_frame"):
            with self.subTest(frame), self.assertRaises(C.GmlxRefusal) as e:
                H.MiniMaxH3ImageToVideo.execute(clip, vae, "a dog", SIZE, SIZE, FRAMES,
                                                **{frame: image})
            self.assertEqual(str(e.exception), "gmlx: a first or last frame (first_frame or "
                             f"last_frame of MiniMaxH3ImageToVideo) reached a gmlx CLIP; "
                             f"{TEXT_ONLY}")
        self.assertEqual(self.stub.events("encode"), [])

    def test_reference_and_guide_nodes_on_gmlx_nodes_are_refused_by_name(self):
        """MiniMaxH3ReferenceToVideo without a VAE shows its references to
        the gmlx CLIP; MiniMaxH3AddGuide hands its image to the gmlx video
        VAE and its audio to the gmlx audio VAE."""
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        image = torch.rand(1, SIZE, SIZE, 3)
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3ReferenceToVideo.execute(clip, "a dog", SIZE, SIZE, FRAMES,
                                                ref_images={"ref_image_1": image})
        self.assertEqual(str(e.exception), "gmlx: references (minimax_ref_items, from "
                         f"MiniMaxH3ReferenceToVideo) reached a gmlx CLIP; {TEXT_ONLY}")
        self.assertEqual(self.stub.events("encode"), [])
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        audio = P.nodes.GmlxAudioVAELoader().load(HID, self.stub.url, False)[0]
        pos, _ = self.cond()
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, vae=video, image=image)
        self.assertEqual(str(e.exception), VIDEO_VAE)
        frames = torch.rand(FRAMES, SIZE, SIZE, 3)
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, vae=video, image=frames)
        self.assertEqual(str(e.exception), VIDEO_VAE)
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3ReferenceToVideo.execute(clip, "a dog", SIZE, SIZE, FRAMES, vae=video,
                                                ref_videos={"ref_video_1": frames})
        self.assertEqual(str(e.exception), VIDEO_VAE)
        track = {"waveform": torch.zeros(1, 2, 3200), "sample_rate": 32000}
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, audio_vae=audio, audio=track)
        self.assertEqual(str(e.exception), "gmlx: a gmlx audio VAE decodes and does not "
                         f"encode, so it takes no guide audio or reference audio; {TEXT_ONLY}")


if __name__ == "__main__":
    unittest.main()
