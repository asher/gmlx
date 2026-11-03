"""MiniMax-H3 through the pack against the stub server: the weightless
model, a sample of ComfyUI's nested video and audio latent through the AV
step with the shifts and audio scale ComfyUI's forward reads, a LoRA from
the stock loader, the video and audio VAE nodes, and the refusals of what
MiniMax-H3 conditions on beside the text prompt, the Fun ControlNet and a
FastH3 DiT. Every ComfyUI the pack supports has MiniMax-H3, which came in
v0.30.0.

Needs ``COMFYUI_DIR``. Run with ``python -m unittest`` from this folder."""

from __future__ import annotations

import unittest

import torch

import support

support.setup_comfy()
P = support.load_pack()

import comfy.nested_tensor  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import node_helpers  # noqa: E402
import nodes as comfy_nodes  # noqa: E402
from comfy_extras import nodes_minimax_h3 as H  # noqa: E402
from comfy_extras.nodes_audio import vae_decode_audio  # noqa: E402

C = P.client
HID, FID, ZID = "h3", "fasth3", "z-image-turbo"
# The 5-frame 64x64 clip, the smallest the latent node makes.
SIZE, FRAMES = 64, 5
TEXT_ONLY = "the gmlx nodes run MiniMax-H3 from the text prompt alone"


def _out(node_output):
    return node_output.result if hasattr(node_output, "result") else node_output


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = support.Stub()
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
    def test_h3_builds_on_meta_and_refuses_a_fasth3_dit_by_name(self):
        m = self.model()
        self.assertEqual(type(m.model.model_config).__name__, "MiniMaxH3")
        self.assertTrue(all(p.device.type == "meta" for p in m.model.parameters()))
        self.assertIs(m.model.diffusion_model.forward.av, True)
        with self.assertRaises(C.GmlxRefusal) as e:
            self.model(FID)
        self.assertEqual(str(e.exception), "gmlx: 'fasth3' is a FastH3 DiT; the gmlx nodes "
                         "run the MiniMax-H3 recipes h3, h3-fast and h3-base")

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

    def test_reference_and_guide_nodes_on_gmlx_nodes_are_refused_by_name(self):
        """MiniMaxH3ReferenceToVideo without a VAE shows its references to
        the gmlx CLIP; MiniMaxH3AddGuide hands its image to the gmlx video
        VAE."""
        clip = P.nodes.GmlxCLIPLoader().load(HID, self.stub.url, False)[0]
        image = torch.rand(1, SIZE, SIZE, 3)
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3ReferenceToVideo.execute(clip, "a dog", SIZE, SIZE, FRAMES,
                                                ref_images={"ref_image_1": image})
        self.assertEqual(str(e.exception), "gmlx: references (minimax_ref_items, from "
                         f"MiniMaxH3ReferenceToVideo) reached a gmlx CLIP; {TEXT_ONLY}")
        self.assertEqual(self.stub.events("encode"), [])
        video = P.nodes.GmlxVideoVAELoader().load(HID, self.stub.url, False)[0]
        pos, _ = self.cond()
        with self.assertRaises(C.GmlxRefusal) as e:
            H.MiniMaxH3AddGuide.execute(pos, self.latent(), 0, vae=video, image=image)
        self.assertEqual(str(e.exception),
                         f"gmlx: a gmlx video VAE decodes and does not encode; {TEXT_ONLY}")


if __name__ == "__main__":
    unittest.main()
