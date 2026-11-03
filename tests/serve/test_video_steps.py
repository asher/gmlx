"""The step routes on a tiny MiniMax-H3 pipeline: the AV step against
ComfyUI's forward rule for the audio its sampler carries, the refined text
rows kept by cond and patch table, the step's refusals, the refusal of
images on the H3 encode, and the video model's status. Every MLX op runs on
the image worker."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")
mx = pytest.importorskip("mlx.core")

from fastapi.testclient import TestClient  # noqa: E402

from gmlx.config import ImageServerCfg  # noqa: E402
from gmlx.serve import image_steps as steps  # noqa: E402
from gmlx.serve import image_wire as wire  # noqa: E402
from gmlx.serve.image_service import WORKER  # noqa: E402
from gmlx.serve.image_steps_client import (StepsClient,  # noqa: E402
                                           StepsClientError)
from gmlx.serve.patches import image_steps as routes  # noqa: E402
from gmlx.serve.video_service import ServedVideo, VideoEntryPlan  # noqa: E402
from gmlx.video import api as vapi  # noqa: E402
from gmlx.video import layout, pipeline  # noqa: E402
from gmlx.video.families.h3 import family as hf  # noqa: E402
from gmlx.video.families.h3 import model as hm  # noqa: E402
from test_video_h3_model import TINY, tiny_weights  # noqa: E402

_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
P = wire.PREFIX
V = {wire.CONTRACT_HEADER: "1.0"}
FRAMES, SIZE = 5, 64
# The h3 recipe's shifts, which MiniMaxH3SigmaShift puts in the options,
# and the audio scale ComfyUI's sampler carries the audio at.
SHIFTS = {"video": 6.0, "audio": 3.0}
SCALE = SHIFTS["video"] / SHIFTS["audio"]


def _on_worker(fn, *args):
    return WORKER.submit(fn, *args).result()


def _tiny_pipe():
    m = hm.H3Transformer(TINY)
    m.load_weights(list(tiny_weights().items()))
    m.act_dtype = mx.float32
    m.eval()
    return pipeline.VideoPipeline(dit=m, family=hf.FAMILY,  # type: ignore
                                  variant="fl2va")


def _served(pipe):
    spec = SimpleNamespace(id="h3", path="/abs/h3.gguf", variant="fl2va",
                           encoder="/abs/enc.gguf", vae="/abs/vae.st",
                           audio_vae="/abs/audio.st", adapters=(),
                           encoder_model=None, defaults={})
    plan = VideoEntryPlan(
        footprint=0, family="h3", variant="fl2va", size=(SIZE, SIZE),
        frames=FRAMES, steps=8, shifts=dict(SHIFTS), dit_bytes=0,
        encoder_bytes=0, vae_bytes=0, audio_vae_bytes=0)
    return ServedVideo(pipe=pipe, spec=spec, plan=plan)


class _Host(routes.ServeHost):
    """The tiny H3 pipeline served as id h3."""

    budget_bytes = None

    def __init__(self, served):
        self.served = served

    def resolve(self, model):
        if (model or "h3") != "h3":
            raise LookupError(f"unknown model {model!r}")
        return self.served.spec.path, self.served.spec

    def acquire(self, model):
        from gmlx.serve.residency import _BusyHold

        self.resolve(model)

        class Pool:
            def release(self, entry):
                pass

        return _BusyHold(Pool(), SimpleNamespace(pipeline=self.served,
                                                 footprint=0))

    def models(self):
        return {"h3": self.served.spec}

    def resident(self, path):
        return True

    def plan(self, spec):
        return self.served.plan

    def budget(self, hold):
        return self.budget_bytes


@pytest.fixture(scope="module")
def pipe():
    return _on_worker(_tiny_pipe)


@pytest.fixture
def host(pipe, request):
    h = _Host(_served(pipe))
    routes.mount(_APP.app, ImageServerCfg(), h)

    def done():
        steps.STATE.reset()
        routes.mount(_APP.app, ImageServerCfg(steps_api=False))
    request.addfinalizer(done)
    return h


def _client() -> StepsClient:
    http = TestClient(_APP.app, base_url="http://127.0.0.1")

    def call(method, path, body, headers):
        r = http.request(method, path, content=body, headers=headers)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, \
            r.content
    return StepsClient(call)


def _clip(seed=0):
    """Video and audio latents of a 5-frame 64x64 clip, the audio as the
    sampler carries it, and one prompt's encoder features."""
    rng = np.random.default_rng(seed)
    vs, aus = hf.latent_shapes(FRAMES, SIZE, SIZE)
    video = rng.standard_normal(vs).astype(np.float32)
    audio = rng.standard_normal(aus).astype(np.float32)
    feats = rng.standard_normal((1, 6, TINY["text_dim"])).astype(np.float32)
    return video, audio, feats


def _comfy_sigmas(t: float) -> tuple:
    """sigma_v and sigma_a of ComfyUI's MiniMax-H3 forward at timestep t,
    in float32 as torch takes them (time_shift_sigma)."""
    f = np.float32
    sv = max(f(t) / f(1000.0), f(1e-6))
    a, b = SHIFTS["video"], SHIFTS["audio"]
    base = sv / (f(a) + sv * f(1.0 - a))
    sa = f(b) * base / (f(1.0) + f(b - 1.0) * base)
    return f(sv), f(sa)


def _comfy_rule(pipe, video, audio, t, feats):
    """ComfyUI's forward around gmlx's network: the audio carried at
    sigma_a / sigma_v into the network, and its velocity converted back
    to the carried variable's."""
    sv, sa = _comfy_sigmas(t)
    carry = sa / sv
    xa = (audio * carry).astype(np.float32)

    def run():
        dit = pipe.dit
        text = dit.caption(mx.array(feats))
        lay = layout.pack(hf.FAMILY.timeline, feats.shape[1],
                          *video.shape[2:], audio.shape[3])
        vv, va = vapi.denoise_step(dit, mx.array(video), mx.array(xa),
                                   {"video": float(sv), "audio": float(sa)},
                                   text, lay)
        return np.array(vv.astype(mx.float32)), np.array(
            va.astype(mx.float32))

    vv, va = _on_worker(run)
    f = np.float32
    va = f(1.0 - SCALE) * xa + (f(1.0) + f(SCALE - 1.0) * sa) * va
    return vv, va


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("sigma", [1.0, 0.6])
def test_an_av_step_gives_comfyuis_output_for_the_carried_audio(host, pipe,
                                                              sigma):
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    t = np.float32(sigma) * np.float32(1000)
    vv, va = c.av_step("h3", None, video, audio, t, row, shifts=SHIFTS,
                       audio_scale=SCALE)
    want_v, want_a = _comfy_rule(pipe, video, audio, t, feats)
    assert vv.shape == video.shape and va.shape == audio.shape
    assert _rel(vv, want_v) < 1e-5
    assert _rel(va, want_a) < 1e-5


def test_the_refined_rows_are_kept_by_cond_and_patch_table(host, pipe,
                                                          tmp_path):
    """A second step of the same cond reuses its refined text rows; a step
    under another patch table refines them again, since a LoRA on the
    refiner changes them."""
    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    first = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE)
    again = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE)
    path = tmp_path / "lora.safetensors"
    write_keys(path, _comfy_keys(_adapter(5)))
    core = _on_worker(pipeline.adapter_table, pipe, [(str(path), 1.0)])
    steps.STATE.tables_of("h3", pipe)["t1"] = core
    other = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE, patch_table="t1")
    assert [r.prefix_cache for r in c.records] == ["miss", "hit", "miss"]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert _rel(other[0], first[0]) > 1e-3


def test_a_lora_from_comfyuis_patches_steps_as_its_file(host, pipe, tmp_path):
    """The pack sends a stock LoRA loader's patches as a contract table on
    ComfyUI's keys; a step under it equals one under the same file."""
    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    adapter = _adapter(5)
    path = tmp_path / "lora.safetensors"
    write_keys(path, _comfy_keys(adapter))
    c = _client()
    entries = [{"key": f"diffusion_model.{target}.weight", "band": None,
                "patches": [{"up": c.put(up, kind="adapters"),
                             "down": c.put(down, kind="adapters"),
                             "alpha": float(alpha), "strength": 1.0,
                             "strength_model": 1.0}]}
               for target, (down, up, alpha) in sorted(adapter.items())]
    tid = c.define_table("h3", {"model": "h3", "entries": entries})
    core = _on_worker(pipeline.adapter_table, pipe, [(str(path), 1.0)])
    steps.STATE.tables_of("h3", pipe)["file"] = core
    video, audio, feats = _clip()
    row = {"cond": c.put(feats)}
    got = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                    audio_scale=SCALE, patch_table=tid)
    want = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                     audio_scale=SCALE, patch_table="file")
    plain = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE)
    for g, w, p in zip(got, want, plain):
        assert _rel(g, w) < 1e-5 and _rel(g, p) > 1e-3


@pytest.mark.parametrize("change, match", [
    # A batch of two, as RepeatLatentBatch makes; ComfyUI's own H3 forward
    # takes batch 1.
    ("batch", "one clip"),
    # An audio track of another length beside the video, as
    # LTXVConcatAVLatent can join them.
    ("audio_len", "no MiniMax-H3 clip gmlx runs"),
])
def test_an_av_step_on_latents_that_are_no_clip_is_refused(host, change,
                                                          match):
    video, audio, feats = _clip()
    if change == "batch":
        video, audio = (np.concatenate([a, a]) for a in (video, audio))
    else:
        audio = np.concatenate([audio, audio[..., :3]], axis=-1)
    c = _client()
    row = {"cond": c.put(feats)}
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                  audio_scale=SCALE)
    assert e.value.status == 400 and match in str(e.value.body["error"])


def test_a_step_that_does_not_fit_beside_the_models_is_refused(host):
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    host.budget_bytes = 1 << 20
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                  audio_scale=SCALE)
    assert e.value.status == 400
    assert "does not fit in memory" in e.value.body["error"]
    host.budget_bytes = 64 << 30
    c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
              audio_scale=SCALE)


def test_the_h3_encode_refuses_images_by_name(host):
    """An image-to-video workflow wires its keyframe through GmlxCLIP,
    which sends the image with the prompt."""
    image = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(StepsClientError) as e:
        _client().encode("h3", "a cat", images=[image])
    assert e.value.status == 400
    assert e.value.body["error"] == (
        "the gmlx server encodes text alone for MiniMax-H3; it takes no "
        "images with the prompt, which keyframes and guides send")


def test_a_video_model_status_names_comfyuis_class_and_its_shifts(
        host, tmp_path, monkeypatch):
    """The pack builds its model from the status's class and refuses a
    variant it does not run."""
    spec = host.served.spec
    for name in ("path", "encoder", "vae", "audio_vae"):
        f = tmp_path / name
        f.write_bytes(b"x" * 16)
        setattr(spec, name, str(f))
    monkeypatch.setattr(steps, "family_of", lambda s: (hf.FAMILY, "fl2va"))
    st = _client().status("h3")
    assert st["family"] == "h3" and st["variant"] == "fl2va"
    assert st["comfy"] == {"model_class": "MiniMaxH3",
                           "image_model": "minimax_h3"}
    assert st["latent"]["channels"] == 24
    assert st["audio"]["sample_rate"] == 32000
    assert st["sampling"]["steps"] == 8
    assert (st["sampling"]["shift"], st["sampling"]["audio_shift"]) == (6, 3)
