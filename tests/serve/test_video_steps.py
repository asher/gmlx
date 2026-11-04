"""The step routes on a tiny MiniMax-H3 pipeline: the AV step against
ComfyUI's forward rule for the audio its sampler carries, the refined text
rows kept by cond and patch table, the memory checks of the step and the
decodes, the encode with the DiT and its LoRA let go and loaded again,
the entry's cache bounds, the refusals of each route, the decodes'
layouts, and the video model's status. Every MLX op runs on the image
worker."""
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
from gmlx.serve import video_service as vs  # noqa: E402
from gmlx.serve.image_service import WORKER, ServedImage  # noqa: E402
from gmlx.serve.image_steps_client import (MissingTensors,  # noqa: E402
                                           StepsClient, StepsClientError,
                                           array_hash)
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


def _tiny_dit(weights=None):
    m = hm.H3Transformer(TINY)
    m.load_weights(list((weights or tiny_weights()).items()))
    m.act_dtype = mx.float32
    m.eval()
    return m


def _tiny_pipe():
    return pipeline.VideoPipeline(dit=_tiny_dit(), family=hf.FAMILY,  # type: ignore
                                  variant="fl2va")


def _own_pipe() -> tuple:
    """A tiny pipeline of a test's own and its weights, seeded and built
    on this thread: the image worker's stream does not outlive the
    test."""
    weights = tiny_weights()
    mx.eval(list(weights.values()))
    return _on_worker(lambda: pipeline.VideoPipeline(
        dit=_tiny_dit(weights), family=hf.FAMILY, variant="fl2va")), weights


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
    """The tiny H3 pipeline served as id h3, and an image entry as id zit
    for the refusals of the other kind."""

    budget_bytes = None

    def __init__(self, served):
        self.served = served
        image = ServedImage(
            pipe=SimpleNamespace(family=SimpleNamespace(name="zimage")),
            spec=SimpleNamespace(id="zit", path="/abs/zit.safetensors"),
            plan=None, facts=None)  # type: ignore
        self.entries = {"h3": served, "zit": image}

    def resolve(self, model):
        got = self.entries.get(model or "h3")
        if got is None:
            raise LookupError(f"unknown model {model!r}")
        return got.spec.path, got.spec

    def acquire(self, model):
        from gmlx.serve.residency import _BusyHold

        self.resolve(model)

        class Pool:
            def release(self, entry):
                pass

        return _BusyHold(Pool(), SimpleNamespace(
            pipeline=self.entries[model or "h3"], footprint=0))

    def models(self):
        return {k: v.spec for k, v in self.entries.items()}

    def resident(self, path):
        return True

    def plan(self, spec):
        return self.served.plan

    def budget(self, hold):
        return self.budget_bytes


@pytest.fixture(scope="module")
def pipe():
    return _on_worker(_tiny_pipe)


def _mount(h, request):
    routes.mount(_APP.app, ImageServerCfg(), h)

    def done():
        steps.STATE.reset()
        routes.mount(_APP.app, ImageServerCfg(steps_api=False))
    request.addfinalizer(done)
    return h


@pytest.fixture
def host(pipe, request):
    return _mount(_Host(_served(pipe)), request)


@pytest.fixture(scope="module")
def vaes(tmp_path_factory):
    """The seeded tiny video and audio VAE files of check 5's self-test."""
    from test_video_check5_h3 import seeded
    from video_oracle_fixture import committed

    shapes = committed("selftest_h3_vae")["inputs"]["shapes"]
    d = tmp_path_factory.mktemp("h3_vaes")
    files = {}
    for role in ("vae", "audio_vae"):
        files[role] = str(d / f"{role}.safetensors")
        seeded.write_safetensors(files[role], seeded.weights(shapes[role]))
    return files


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
    bare = vs.held_bytes(host.served)
    first = c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE)
    # The step prices the rows it keeps among the caches.
    (kept,) = host.served.captions.values()
    assert vs.held_bytes(host.served) == (bare[0], bare[1] + kept.nbytes)
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


def test_the_negative_of_a_guided_step_is_refined_from_its_own_cond(host):
    """At cfg above 1 ComfyUI runs the model on the positive cond and then
    on the negative one, each a step of its own under the same table."""
    video, audio, pos = _clip()
    neg = _clip(1)[2]
    c = _client()
    rows = [{"cond": c.put(f)} for f in (pos, neg)]
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    c.av_step("h3", None, video, audio, 600.0, rows[0], **kw)
    after = c.av_step("h3", None, video, audio, 600.0, rows[1], **kw)
    host.served.captions.clear()
    alone = c.av_step("h3", None, video, audio, 600.0, rows[1], **kw)
    assert [r.prefix_cache for r in c.records] == ["miss", "miss", "miss"]
    assert all(np.array_equal(a, b) for a, b in zip(after, alone))


def test_a_lora_on_the_text_refiner_reaches_the_refined_rows(host, pipe,
                                                            tmp_path):
    """lightx2v's 8-step LoRA patches the text projection and refiner
    too; a table of those pairs alone changes the step."""
    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    text = {k: v for k, v in _adapter(5).items()
            if k.startswith(("token_refiner.", "condition_proj"))}
    path = tmp_path / "refiner.safetensors"
    write_keys(path, _comfy_keys(text))
    core = _on_worker(pipeline.adapter_table, pipe, [(str(path), 1.0)])
    steps.STATE.tables_of("h3", pipe)["refiner"] = core
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    plain = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    got = c.av_step("h3", None, video, audio, 600.0, row, patch_table="refiner",
                    **kw)
    assert _rel(got[0], plain[0]) > 1e-3 and _rel(got[1], plain[1]) > 1e-3


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


def _plain_need(served, text: int = 6) -> int:
    """The step's bytes by the entry's plan alone."""
    return vs.plan_clip(served.plan, frames=FRAMES, width=SIZE, height=SIZE,
                        text=text, decode=False, budget=1).phases["denoise"]


@pytest.mark.parametrize("held", ["uploads", "pairs", "table"])
def test_a_step_counts_what_the_server_holds_and_runs_beside_the_dit(
        tmp_path, request, held):
    """A step fits by the entry's plan, and then not: beside the LoRA
    matrices the pack uploads before its table, beside the pairs of a
    ComfyUI LoRA's table past what the plan prices for the entry's own
    LoRA, or under such a table on an entry with no LoRA of its own,
    whose step then runs a LoRA."""
    import dataclasses

    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    pipe, _w = _own_pipe()
    served = _served(pipe)
    if held == "pairs":
        # A lone bf16 LoRA of the entry, priced at its file of one byte.
        served.plan = dataclasses.replace(served.plan, loras=(1,),
                                          lora_dtypes=("BF16",))
    host = _mount(_Host(served), request)
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    margin = {"uploads": 64 << 20, "pairs": 8 << 10, "table": 100 << 20}[held]
    host.budget_bytes = _plain_need(served) + margin
    c.av_step("h3", None, video, audio, 600.0, row, **kw)
    table = None
    if held == "uploads":
        c.put(np.ones((8192, 3072), np.float32), kind="adapters")
    else:
        path = tmp_path / "lora.safetensors"
        write_keys(path, _comfy_keys(_adapter(5)))
        table = "lora"
        steps.STATE.tables_of("h3", pipe)[table] = _on_worker(
            pipeline.adapter_table, pipe, [(str(path), 1.0)])
        assert held == "table" or pipe.adapters.bytes > 2 * margin
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", None, video, audio, 600.0, row, patch_table=table,
                  **kw)
    assert e.value.status == 400
    assert "does not fit in memory" in e.value.body["error"]


def _lora_file(tmp_path, seed: int = 5, name: str = "lora.safetensors"):
    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    path = tmp_path / name
    write_keys(path, _comfy_keys(_adapter(seed)))
    return path


def test_a_step_counts_the_entrys_own_lora_once(tmp_path, request):
    """The plan prices the entry's own LoRA, and its pairs are what the
    entry holds of it: a step on a budget that fits the DiT, the LoRA,
    the caches and the step runs."""
    import dataclasses

    path = _lora_file(tmp_path)
    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    served.plan = dataclasses.replace(served.plan,
                                      loras=(path.stat().st_size,),
                                      lora_dtypes=("BF16",))
    _on_worker(vs._load_adapters, served)
    host = _mount(_Host(served), request)
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    plan = vs.plan_clip(served.plan, frames=FRAMES, width=SIZE, height=SIZE,
                        text=6, decode=False, budget=1)
    lora = plan.terms["denoise"]["lora"]
    adapters, caches = vs.held_bytes(served)
    assert lora == path.stat().st_size and lora // 2 < adapters <= lora
    host.budget_bytes = lora + caches + plan.terms["denoise"]["step"] + 4096
    c.av_step("h3", None, video, audio, 600.0, row, shifts=SHIFTS,
              audio_scale=SCALE)


def test_a_runs_first_step_counts_the_clips_decodes(host):
    """A clip whose decodes do not fit beside the models loaded stops at
    its run's first step, not after its last. The steps after it count
    the step alone, as each decode is counted when it runs."""
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    served = host.served
    plan = vs.plan_clip(served.plan, frames=FRAMES, width=SIZE, height=SIZE,
                        text=6, budget=1)
    step = plan.terms["denoise"]["step"]
    decodes = max(plan.phases["audio"], plan.phases["decode"])
    assert decodes > step
    adapters, caches = vs.held_bytes(served)
    host.budget_bytes = adapters + caches + decodes - 1
    run = c.begin("h3")
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", run, video, audio, 600.0, row, **kw)
    assert e.value.status == 400 and e.value.body["error"] == (
        "a clip of 64x64 at 5 frames does not fit in memory beside the "
        "models now loaded")
    host.budget_bytes += 1
    c.av_step("h3", run, video, audio, 600.0, row, **kw)
    adapters, caches = vs.held_bytes(served)
    host.budget_bytes = adapters + caches + step
    c.av_step("h3", run, video, audio, 600.0, row, **kw)


@pytest.mark.parametrize("ask, want", [
    # An image-to-video request sends its first or last frame with the
    # prompt; the gmlx CLIP refuses them before it calls the server.
    ("images", "the gmlx server encodes text alone for MiniMax-H3; it takes "
     "no images with the prompt, such as a first or last frame"),
    ("prompt_upsampling", "video model 'h3' takes no prompt upsampling; "
     "send the request without prompt_upsampling"),
])
def test_the_h3_encode_refuses_what_it_does_not_run_by_name(host, ask, want):
    c = _client()
    with pytest.raises(StepsClientError) as e:
        if ask == "images":
            c.encode("h3", "a cat", images=[np.zeros((32, 32, 3), np.uint8)])
        else:
            c.container("encode", {}, {"model": "h3", "text": "a cat",
                                       "prompt_upsampling": True})
    assert e.value.status == 400 and e.value.body["error"] == want


def test_an_encode_loads_the_encoder_without_the_dit_or_any_lora(
        tmp_path, monkeypatch, request):
    """The H3 encoder and the DiT with its 8-step LoRA do not fit a 32 GB
    Mac together: the encoder loads with neither the DiT nor a LoRA pair
    held. The configured LoRA comes back with the DiT at the next step,
    and a table the pack sent is answered as missing until the pack sends
    it again."""
    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    from gmlx.video import encoder as video_encoder
    from gmlx.video import loader

    path = tmp_path / "lora.safetensors"
    write_keys(path, _comfy_keys(_adapter(5)))
    pipe, weights = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 0.75),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)

    def strengths():
        entries = pipe.patch_tables[pipe.default_table]["entries"]
        return {st for _path, _band, patches in entries for _k, st in patches}

    assert strengths() == {0.75}
    video, audio, feats = _clip()
    c = _client()
    entries = [{"key": f"diffusion_model.{target}.weight", "band": None,
                "patches": [{"up": c.put(up, kind="adapters"),
                             "down": c.put(down, kind="adapters"),
                             "alpha": float(alpha), "strength": 1.0,
                             "strength_model": 1.0}]}
               for target, (down, up, alpha) in sorted(_adapter(6).items())]
    body = {"model": "h3", "entries": entries}
    tid = c.define_table("h3", body)
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    before = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    sent = c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid,
                     **kw)
    seen = {}

    def load_encoder(fam, path, log=None):
        seen.update(dit=served.pipe.dit, pairs=served.pipe.adapters.bytes)
        feats = np.zeros((4, TINY["text_dim"]), np.float32)
        return SimpleNamespace(encode=lambda text: SimpleNamespace(
            features=feats), release=lambda: None)

    monkeypatch.setattr(video_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(loader, "load_dit",
                        lambda *a, **k: (_tiny_dit(weights), None))
    c.encode("h3", "a cat")
    assert seen == {"dit": None, "pairs": 0} and pipe.dit is None
    after = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert strengths() == {0.75}
    assert all(_rel(a, b) < 1e-6 for a, b in zip(after, before))
    with pytest.raises(MissingTensors) as e:
        c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid, **kw)
    assert e.value.body["missing"] == [tid]
    assert c.define_table("h3", body) == tid
    again = c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid,
                      **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(again, sent))


def test_the_encode_refuses_a_prompt_whose_rows_do_not_fit(
        tmp_path, monkeypatch, request):
    """The encode prices a prompt by its rows, by the encoder's tokenizer
    from its header as the video command prices it, beside the caches the
    server holds. A short prompt encodes, a long one is refused with the
    command's hint, and a tensor uploaded since counts."""
    from test_video_encoder import _gguf

    pipe, weights = _own_pipe()
    served = _served(pipe)
    served.spec.encoder = _gguf(tmp_path / "enc.gguf")
    host = _mount(_Host(served), request)
    _encoder_by_text(monkeypatch, weights)
    short, long = "a cat", "a cat on a mat " * 2000
    rows = vs.prompt_rows(served, long)

    def encode_need(text):
        return vs.plan_clip(served.plan, frames=FRAMES, width=SIZE,
                            height=SIZE, text=vs.prompt_rows(served, text),
                            decode=False, budget=1).phases["encode"]

    c = _client()
    _adapters, caches = vs.held_bytes(served)
    host.budget_bytes = caches + encode_need(short) + (4 << 20)
    c.encode("h3", short)
    with pytest.raises(StepsClientError) as e:
        c.encode("h3", long)
    assert e.value.status == 400 and e.value.body["error"] == (
        f"the encode of {rows} text rows does not fit in memory beside the "
        "models now loaded; try a shorter prompt")
    c.put(np.zeros((1, 2 << 20), np.float32))
    with pytest.raises(StepsClientError) as e:
        c.encode("h3", short)
    assert e.value.status == 400


def _table_body(c, adapter: dict, alphas: bool = True) -> dict:
    """A contract table of ``adapter``'s pairs on ComfyUI's keys, as the
    pack sends a stock LoRA loader's patches; without ``alphas``, every
    pair's alpha is None."""
    entries = [{"key": f"diffusion_model.{target}.weight", "band": None,
                "patches": [{"up": c.put(up, kind="adapters"),
                             "down": c.put(down, kind="adapters"),
                             "alpha": float(alpha) if alphas else None,
                             "strength": 1.0, "strength_model": 1.0}]}
               for target, (down, up, alpha) in sorted(adapter.items())]
    return {"model": "h3", "entries": entries}


def _stub_encode(monkeypatch, weights):
    """The H3 encoder stood in, and the DiT loaded after an encode as the
    tiny DiT of ``weights``."""
    from gmlx.video import encoder as video_encoder
    from gmlx.video import loader

    def load_encoder(fam, path, log=None):
        feats = np.zeros((4, TINY["text_dim"]), np.float32)
        return SimpleNamespace(encode=lambda text: SimpleNamespace(
            features=feats), release=lambda: None)

    monkeypatch.setattr(video_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(loader, "load_dit",
                        lambda *a, **k: (_tiny_dit(weights), None))


def test_a_lora_that_does_not_load_again_with_the_dit_fails_the_step(
        tmp_path, monkeypatch, request):
    """The configured LoRA loads again with the DiT at the first step
    after an encode. A file moved since fails that step with its path and
    leaves the entry with no DiT and no table of the LoRA, so the next
    step loads both again once the file is back."""
    from gmlx.image import api

    path = _lora_file(tmp_path)
    pipe, weights = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)
    _stub_encode(monkeypatch, weights)
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    before = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    moved = path.rename(tmp_path / "moved.safetensors")
    c.encode("h3", "a cat")
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert e.value.status == 400
    assert str(path) in e.value.body["error"]
    assert pipe.dit is None and pipe.default_table == api.EMPTY_TABLE
    moved.rename(path)
    after = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(after, before))


@pytest.mark.parametrize("between", ["encode", "define"])
def test_a_table_let_go_before_its_step_runs_is_missing(monkeypatch, request,
                                                       between):
    """A second client's encode, or the define of its table on a pair
    cache with room for one table, can run on the image worker between a
    step's check of its table and the step itself. The step answers the
    table as missing, and the pack sends it again."""
    from test_video_h3_lora import _adapter

    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    _stub_encode(monkeypatch, weights)
    c = _client()
    body = _table_body(c, _adapter(6))
    theirs = _table_body(_client(), _adapter(7))
    pipe.adapters.max_bytes = sum(d.nbytes + u.nbytes
                                  for d, u, _a in _adapter(6).values())
    tid = c.define_table("h3", body)
    video, audio, feats = _clip()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE, patch_table=tid)
    sent = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    ensure, ran = vs.ensure_dit, []

    def other_first(served):
        if not ran:
            ran.append(True)
            if between == "encode":
                vs.drop_dit(served)     # what the encode leaves
            else:
                steps.define_table(served, theirs, steps.STATE.adapters)
        ensure(served)

    monkeypatch.setattr(vs, "ensure_dit", other_first)
    with pytest.raises(MissingTensors) as e:
        c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert ran and e.value.body["missing"] == [tid]
    assert c.define_table("h3", body) == tid
    again = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(again, sent))


def test_a_table_whose_pairs_another_table_shed_is_missing_until_sent_again(
        request):
    """Two ComfyUI clients whose LoRAs each fill the entry's pair room, with
    its built tables bounded as a loaded entry bounds them: B's table
    sheds the pairs of A's. A's next step, on a cond the server lacks
    too, names both in one 409, and A steps as before once the pack sends
    its table again."""
    from test_video_h3_lora import _adapter

    pipe, _w = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    a = _adapter(6)
    need = sum(d.nbytes + u.nbytes for d, u, _a in a.values())
    pipe.adapters.max_bytes = pipe.tables.max_bytes = need
    ca, cb = _client(), _client()
    body = _table_body(ca, a)
    video, audio, feats = _clip()
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    ta = ca.define_table("h3", body)
    row = {"cond": ca.put(feats)}
    first = ca.av_step("h3", None, video, audio, 600.0, row, patch_table=ta,
                       **kw)
    tb = cb.define_table("h3", _table_body(cb, _adapter(7)))
    cb.av_step("h3", None, video, audio, 600.0, {"cond": cb.put(feats)},
               patch_table=tb, **kw)
    lacked = array_hash(_clip(1)[2])
    with pytest.raises(MissingTensors) as e:
        ca.av_step("h3", None, video, audio, 600.0, {"cond": lacked},
                   patch_table=ta, **kw)
    assert sorted(e.value.body["missing"]) == sorted([lacked, ta])
    assert ca.define_table("h3", body) == ta
    again = ca.av_step("h3", None, video, audio, 600.0, row, patch_table=ta,
                       **kw)
    assert all(_rel(x, y) < 1e-6 for x, y in zip(again, first))


def _encoder_by_text(monkeypatch, weights) -> list:
    """The H3 encoder stood in with features of the prompt's own seed, and
    the DiT loaded after an encode as the tiny DiT of ``weights``. Returns
    the list each DiT load appends to."""
    import zlib

    from gmlx.video import encoder as video_encoder
    from gmlx.video import loader

    def load_encoder(fam, path, log=None):
        def encode(text):
            rng = np.random.default_rng(zlib.crc32(text.encode()))
            return SimpleNamespace(features=rng.standard_normal(
                (4, TINY["text_dim"])).astype(np.float32))
        return SimpleNamespace(encode=encode, release=lambda: None)

    loads = []
    monkeypatch.setattr(video_encoder, "load_encoder", load_encoder)
    monkeypatch.setattr(loader, "load_dit", lambda *a, **k: (
        loads.append(1), (_tiny_dit(weights), None))[1])
    return loads


def test_two_clients_alternating_encode_and_step_get_their_own_steps(
        monkeypatch, request):
    """Two ComfyUI clients that alternate their encodes and steps on one
    entry each get their own steps back. Each encode lets the DiT go, as
    the entry's plan prices an encode with the DiT out, and the first
    define or step after it loads it again, once. A client's table that
    the other's encode let go is answered as missing, and steps once sent
    again."""
    from test_video_h3_lora import _adapter

    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    loads = _encoder_by_text(monkeypatch, weights)
    video, audio, _feats = _clip()
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    clients = {}
    for name, seed in (("a", 6), ("b", 7)):
        c = _client()
        clients[name] = SimpleNamespace(c=c, body=_table_body(c, _adapter(seed)),
                                        row=None, tid=None)

    def encode(name, text):
        me = clients[name]
        me.row = {"cond": me.c.encode("h3", text)[0]["cond"]}

    def step(name):
        me = clients[name]
        if me.tid is None:
            me.tid = me.c.define_table("h3", me.body)
        try:
            got = me.c.av_step("h3", None, video, audio, 600.0, me.row,
                               patch_table=me.tid, **kw)
            sent = False
        except MissingTensors as e:
            assert e.body["missing"] == [me.tid]
            me.c.define_table("h3", me.body)
            got = me.c.av_step("h3", None, video, audio, 600.0, me.row,
                               patch_table=me.tid, **kw)
            sent = True
        return got, sent

    encode("a", "a red fox")
    first_a, _ = step("a")
    encode("b", "a blue bird")
    again_a, sent_a = step("a")
    first_b, _ = step("b")
    encode("a", "a red fox")
    again_b, sent_b = step("b")
    third_a, sent_a3 = step("a")
    assert len(loads) == 3
    assert (sent_a, sent_b, sent_a3) == (True, True, True)
    for got, want in ((again_a, first_a), (third_a, first_a),
                      (again_b, first_b)):
        assert all(_rel(x, y) < 1e-6 for x, y in zip(got, want))
    assert _rel(first_a[0], first_b[0]) > 1e-3


def test_a_cfg_runs_two_encodes_load_the_dit_once_for_its_step(
        monkeypatch, request):
    """A run with cfg encodes its prompt and its negative prompt back to
    back, then steps. The DiT stays out between the encodes and loads
    once, at the step."""
    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    loads = _encoder_by_text(monkeypatch, weights)
    c = _client()
    pos = c.encode("h3", "a red fox")[0]["cond"]
    c.encode("h3", "blurry, low quality")
    assert pipe.dit is None and loads == []
    video, audio, _feats = _clip()
    c.av_step("h3", None, video, audio, 600.0, {"cond": pos},
              shifts=SHIFTS, audio_scale=SCALE)
    assert len(loads) == 1


def test_a_video_entry_released_with_a_client_table_frees_its_pipeline(
        monkeypatch, request):
    """The pool's release of a video entry a client defined a table on
    frees its pipeline and DiT, so the next model is admitted beside
    memory that is free."""
    import gc
    import weakref

    from test_video_h3_lora import _adapter

    from gmlx.serve import image_service

    pipe, _w = _own_pipe()
    plan, held = _served(pipe).plan, [pipe]
    monkeypatch.setattr(vs, "entry_plan", lambda spec: plan)
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: held.pop())
    monkeypatch.setattr(image_service, "_SHARES", {})
    box = [_on_worker(vs._load, _served(pipe).spec)]
    host = _mount(_Host(box[0]), request)
    c = _client()
    tid = c.define_table("h3", _table_body(c, _adapter(6)))
    assert tid in steps.STATE.tables_of("h3", pipe)
    refs = [weakref.ref(pipe), weakref.ref(pipe.dit)]
    del pipe
    host.entries.pop("h3")
    host.served = None
    vs.release_on_worker(box)
    gc.collect()
    assert [r() for r in refs] == [None, None]


def test_a_loaded_video_entry_bounds_its_caches_at_its_share(tmp_path,
                                                            monkeypatch):
    """The pool admits an entry with each cache at the share of its bound
    that its clip leaves room for. The entry bounds its pipeline's caches
    at that share beside the bytes its plan books for its own LoRA, which
    hold its pairs, and the step routes' caches take the smallest share of
    the entries loaded until it is released."""
    import dataclasses

    from gmlx.serve import image_service

    path = _lora_file(tmp_path)
    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    size = path.stat().st_size
    plan = dataclasses.replace(served.plan, loras=(size,), lora_term=size,
                               cache_share=0.25)
    monkeypatch.setattr(vs, "entry_plan", lambda spec: plan)
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: pipe)
    monkeypatch.setattr(image_service, "_SHARES", {})
    scaled = []
    monkeypatch.setattr(steps.STATE, "scale_caches", scaled.append)
    got = _on_worker(vs._load, served.spec)
    tensor, adapter = image_service.cache_bytes()
    assert 0 < pipe.adapters.bytes <= size
    assert pipe.conditionings.max_bytes == tensor // 4
    assert pipe.reference_latents.max_bytes == tensor // 4
    assert pipe.adapters.max_bytes == pipe.tables.max_bytes == (
        adapter // 4 + size)
    assert scaled == [0.25]
    _on_worker(vs._release, [got])
    assert scaled == [0.25, 1.0]


def test_the_step_tensor_cache_keeps_a_loaded_video_entrys_cond(
        monkeypatch, request):
    """The step routes' caches take the smallest share of the entries
    loaded, and their tensor cache the largest cond floor: beside an image
    entry at a share of 1/16, a video entry's cond of the priced rows of
    H3's 5120-wide features still uploads. Released, the video entry's
    floor goes with it."""
    import dataclasses

    from gmlx.serve import image_service

    pipe, _w = _own_pipe()
    served = _served(pipe)
    rows = np.zeros((1, vs.PRICED_TEXT_ROWS, 5120), np.float32)
    plan = dataclasses.replace(served.plan, cache_share=0.5,
                               tensor_floor=rows.nbytes)
    monkeypatch.setattr(vs, "entry_plan", lambda spec: plan)
    monkeypatch.setattr(pipeline, "load", lambda *a, **k: pipe)
    _mount(_Host(served), request)
    tensor, adapter = image_service.cache_bytes()
    # An image entry loaded at a share of 1/16.
    shares = {0: (1 / 16, 0)}
    monkeypatch.setattr(image_service, "_SHARES", shares)
    try:
        image_service._scale_step_caches()
        got = _on_worker(vs._load, served.spec)
        bounds = steps.STATE.tensors.max_bytes, steps.STATE.adapters.max_bytes
        _client().put(rows)
        _on_worker(vs._release, [got])
        after = steps.STATE.tensors.max_bytes
        with pytest.raises(StepsClientError) as e:
            _client().put(rows)
    finally:
        shares.clear()
        image_service._scale_step_caches()
    assert bounds == (rows.nbytes, adapter // 16)
    assert after == tensor // 16 and e.value.status == 413


@pytest.mark.parametrize("route", ["upload", "encode"])
def test_a_cond_over_the_share_the_models_leave_is_413_naming_it(
        route, monkeypatch, request):
    """The step routes' tensor cache takes the smallest share of the
    entries loaded. Under the share that holds one cond of the priced rows
    of H3's 5120-wide features, the cond a pack uploads, or an encode
    keeps, is refused with the share as its cause and its remedy, and a
    tensor over the configured bound with the key that sets it."""
    from gmlx.video import encoder as video_encoder

    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    _stub_encode(monkeypatch, weights)
    rows = np.zeros((1, vs.PRICED_TEXT_ROWS, 5120), np.float32)
    monkeypatch.setattr(video_encoder, "load_encoder", lambda *a, **k: (
        SimpleNamespace(encode=lambda text: SimpleNamespace(features=rows[0]),
                        release=lambda: None)))
    c = _client()
    send = {"upload": lambda: c.put(rows),
            "encode": lambda: c.encode("h3", "a cat")}[route]
    configured = steps.STATE.tensors.max_bytes
    try:
        steps.STATE.scale_caches(0.25)
        with pytest.raises(StepsClientError) as e:
            send()
        with pytest.raises(StepsClientError) as over:
            c.put(np.zeros((1, 5 * configured // 16), np.float32))
        with pytest.raises(StepsClientError) as edge:
            c.put(np.zeros((1, configured // 4 + 1), np.float32))
        steps.STATE.scale_caches(rows.nbytes / configured)
        send()
    finally:
        steps.STATE.scale_caches(1.0)
    assert (e.value.status, over.value.status) == (413, 413)
    assert e.value.body["error"] == (
        "a tensor of 20.0 MiB is over the step routes' tensor cache of 16.0 "
        "MiB: the models loaded leave it 25% of its configured 64 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")
    assert over.value.body["error"] == (
        "a tensor of 80.0 MiB is over the configured 64 MiB of the step "
        "routes' tensor cache; raise server.image.tensor_cache_mb")
    # A tensor a few bytes over the bound reads above it.
    assert edge.value.body["error"] == (
        "a tensor of 64.1 MiB is over the configured 64 MiB of the step "
        "routes' tensor cache; raise server.image.tensor_cache_mb")


def test_the_adapter_cache_and_the_upload_limit_are_named_in_a_413(host):
    """An adapter matrix over the share of the adapter cache is refused
    with that cache named, and a tensor over a route's own upload limit
    with the limit."""
    from gmlx.serve.image_wire import read, write

    c = _client()
    try:
        steps.STATE.scale_caches(0.25)
        with pytest.raises(StepsClientError) as e:
            c.put(np.zeros((1, 20 << 20), np.float32), kind="adapters")
        with pytest.raises(StepsClientError) as edge:
            c.put(np.zeros((1, (16 << 20) + 1), np.float32), kind="adapters")
        # A share whose bound is 69.99 MiB.
        steps.STATE.scale_caches(0.2734)
        with pytest.raises(StepsClientError) as share:
            c.put(np.zeros((1, 70 << 18), np.float32), kind="adapters")
    finally:
        steps.STATE.scale_caches(1.0)
    limit, just = [], []
    for n, got in ((1 << 19, limit), ((1 << 18) + 1, just)):
        (t,) = read(write({"t": np.zeros((1, n), np.float32)}))[0].values()
        with pytest.raises(steps.StepError) as refused:
            steps.STATE.put_tensor(steps.STATE.tensors, t.hash(), {"t": t},
                                   1 << 20)
        got.append(str(refused.value))
    assert e.value.status == 413
    assert e.value.body["error"] == (
        "a tensor of 80.0 MiB is over the step routes' adapter cache of 64.0 "
        "MiB: the models loaded leave it 25% of its configured 256 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")
    assert limit == ["a tensor of 2.0 MiB is over the upload limit of 1.0 MiB"]
    # A tensor a few bytes over a bound reads above it, and a bound under a
    # tenth of a MiB more reads under it.
    assert just == ["a tensor of 1.1 MiB is over the upload limit of 1.0 MiB"]
    assert edge.value.body["error"] == (
        "a tensor of 64.1 MiB is over the step routes' adapter cache of 64.0 "
        "MiB: the models loaded leave it 25% of its configured 256 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")
    assert share.value.body["error"] == (
        "a tensor of 70.0 MiB is over the step routes' adapter cache of 69.9 "
        "MiB: the models loaded leave it 27% of its configured 256 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")


def _big_table(c, pairs: int, width: int = 8192) -> dict:
    """A ComfyUI LoRA's table of ``pairs`` pairs in f32, each a rank 16
    pair over a ``width`` wide weight: 1 MiB at the default width."""
    from test_video_h3_lora import _adapter

    rng = np.random.default_rng(9)
    entries = []
    for target in sorted(_adapter(6))[:pairs]:
        down = rng.standard_normal((16, width)).astype(np.float32)
        up = rng.standard_normal((width, 16)).astype(np.float32)
        entries.append({"key": f"diffusion_model.{target}.weight",
                        "band": None, "patches": [{
                            "up": c.put(up, kind="adapters"),
                            "down": c.put(down, kind="adapters"),
                            "alpha": 16.0, "strength": 1.0,
                            "strength_model": 1.0}]})
    return {"model": "h3", "entries": entries}


@pytest.mark.parametrize("configured_mb, share, want", [
    (8, 0.25, "the LoRA pairs of this patch table take 4.0 MiB, over the "
     "2.0 MiB the pair cache of 'h3' keeps for a request's pairs: its "
     "default clip leaves the cache 25% of its configured 8 MiB. A smaller "
     "defaults.size or defaults.seconds on 'h3' gives it more."),
    (2, 0.25, "the LoRA pairs of this patch table take 4.0 MiB, over the "
     "0.5 MiB the pair cache of 'h3' keeps for a request's pairs: its "
     "default clip leaves the cache 25% of its configured 2 MiB. Raise "
     "server.image.adapter_cache_mb to 4 or more, and set a smaller "
     "defaults.size or defaults.seconds on 'h3' so that it leaves the cache "
     "more of it."),
    (2, 1.0, "the LoRA pairs of this patch table take 4.0 MiB, over the "
     "2.0 MiB the pair cache of 'h3' keeps for a request's pairs; raise "
     "server.image.adapter_cache_mb to 4 or more")])
def test_a_table_over_the_entrys_pair_room_is_413_naming_it(
        monkeypatch, request, configured_mb, share, want):
    """The pairs of a ComfyUI LoRA's table that the entry's pair cache
    cannot hold at once are refused with a 413 that names the cache, the
    sizes and the remedy: under a share, the share the entry's clip leaves
    the cache, and the configured bound, with the MiB it must reach, when
    the bound is under the pairs too."""
    import dataclasses

    from gmlx.serve import image_service

    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.plan = dataclasses.replace(served.plan, cache_share=share)
    _mount(_Host(served), request)
    monkeypatch.setitem(image_service._CACHE_BYTES, "adapter",
                        configured_mb << 20)
    image_service.bound_caches(pipe, 0, share)
    c = _client()
    body = _big_table(c, 4)
    with pytest.raises(StepsClientError) as e:
        c.define_table("h3", body)
    assert e.value.status == 413 and e.value.body["error"] == want
    assert len(pipe.adapters) == 0


@pytest.mark.parametrize("width, room, need, room_mib, bound", [
    (8192, (4 << 20) - 1, "4.0", "3.9", 4), (8200, 4 << 20, "4.1", "4.0", 5)])
def test_a_table_just_over_the_pair_room_reads_its_need_above_it(
        request, width, room, need, room_mib, bound):
    """The 413 rounds the pairs' MiB up and the room's down, so a table a
    byte over the room, or a few KiB, never reads at or under it."""
    pipe, _w = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    c = _client()
    body = _big_table(c, 4, width)
    pipe.adapters.max_bytes = room
    with pytest.raises(StepsClientError) as e:
        c.define_table("h3", body)
    assert e.value.status == 413 and e.value.body["error"] == (
        f"the LoRA pairs of this patch table take {need} MiB, over the "
        f"{room_mib} MiB the pair cache of 'h3' keeps for a request's pairs; "
        f"raise server.image.adapter_cache_mb to {bound} or more")


def test_a_table_that_fills_the_entrys_pair_room_steps(tmp_path, request):
    """A table whose pairs fill the room the entry's pair cache keeps for a
    request's pairs beside its own LoRA holds them all and steps, and one
    a byte over that room is refused."""
    from test_video_h3_lora import _adapter

    path = _lora_file(tmp_path)
    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)
    c = _client()
    adapter = _adapter(6)
    need = sum(d.nbytes + u.nbytes for d, u, _a in adapter.values())
    own = pipe.adapters.bytes
    pipe.adapters.max_bytes = own + need
    tid = c.define_table("h3", _table_body(c, adapter))
    assert pipe.adapters.bytes == own + need
    video, audio, feats = _clip()
    c.av_step("h3", None, video, audio, 600.0, {"cond": c.put(feats)},
              shifts=SHIFTS, audio_scale=SCALE, patch_table=tid)
    pipe.adapters.max_bytes -= 1
    with pytest.raises(StepsClientError) as e:
        c.define_table("h3", _table_body(c, _adapter(7)))
    assert e.value.status == 413


def _entry_with_own(tmp_path, request, seed: int = 5):
    """A mounted entry with a LoRA of its own, the file of ``_adapter``
    ``seed`` at strength 1."""
    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(_lora_file(tmp_path, seed)), 1.0),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)
    return pipe


def _stack_body(c, *loras) -> dict:
    """A contract table of the pairs of each (adapter, strength) of
    ``loras`` on each ComfyUI key, as a chain of stock LoRA loaders sends
    them."""
    rows: dict = {}
    for adapter, strength in loras:
        for e in _table_body(c, adapter)["entries"]:
            for patch in e["patches"]:
                patch["strength"] = strength
            rows.setdefault(e["key"], {**e, "patches": []})["patches"] += (
                e["patches"])
    return {"model": "h3", "entries": list(rows.values())}


def test_a_table_of_the_entrys_own_lora_takes_no_pair_room(tmp_path,
                                                           request):
    """A workflow that keeps the recipe's LoRA in a stock loader sends the
    entry's own pairs as its table. The cache holds them already, so the
    table defines with 1 KiB of room beside them, adds no pair, and steps
    as the entry does without it."""
    from test_video_h3_lora import _adapter

    pipe = _entry_with_own(tmp_path, request, seed=6)
    held = set(pipe.adapters.keys(lambda k: True))
    pipe.adapters.max_bytes = pipe.adapters.bytes + (1 << 10)
    c = _client()
    tid = c.define_table("h3", _table_body(c, _adapter(6)))
    assert set(pipe.adapters.keys(lambda k: True)) == held
    video, audio, feats = _clip()
    row, kw = {"cond": c.put(feats)}, dict(shifts=SHIFTS, audio_scale=SCALE)
    got = c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid,
                    **kw)
    plain = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(g, p) < 1e-5 for g, p in zip(got, plain))


def test_the_old_pairs_of_a_lora_file_written_again_take_the_request_room(
        tmp_path, monkeypatch, request):
    """The entry's LoRA file is written again between two encodes, and
    ComfyUI, which keeps a LoRA it loaded by its path, sends the old pairs
    again with alpha None. The entry reads the new file with its DiT, so
    the old pairs are no longer its own: they take the request room, and
    a table of them over it is a 413 with its remedy."""
    import os

    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    from gmlx.serve.image_service import _pair_keys

    path = _lora_file(tmp_path)
    pipe, weights = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)
    _stub_encode(monkeypatch, weights)
    held = pipe.adapters.bytes
    pipe.adapters.max_bytes = held + held // 2
    c = _client()
    body = _table_body(c, _adapter(5), alphas=False)
    c.define_table("h3", body)
    c.encode("h3", "a cat")
    st = path.stat()
    write_keys(path, _comfy_keys(_adapter(6)))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    with pytest.raises(StepsClientError) as e:
        c.define_table("h3", body)
    assert e.value.status == 413
    assert e.value.body["error"].startswith(
        "the LoRA pairs of this patch table take ")
    assert "; raise server.image.adapter_cache_mb to " in e.value.body["error"]
    assert set(pipe.adapters.keys(lambda k: True)) == _pair_keys(
        served.adapters)


def test_a_lora_loaded_twice_takes_the_room_of_one(request):
    """Two stock loaders of one LoRA at half strength each send each pair
    twice on its key. The pairs fill a room of one copy and step as the
    LoRA at full strength."""
    from test_video_h3_lora import _adapter

    pipe, _w = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    a = _adapter(6)
    pipe.adapters.max_bytes = sum(d.nbytes + u.nbytes for d, u, _a in
                                  a.values())
    c = _client()
    twice = c.define_table("h3", _stack_body(c, (a, 0.5), (a, 0.5)))
    video, audio, feats = _clip()
    row, kw = {"cond": c.put(feats)}, dict(shifts=SHIFTS, audio_scale=SCALE)
    got = c.av_step("h3", None, video, audio, 600.0, row, patch_table=twice,
                    **kw)
    once = c.define_table("h3", _table_body(c, a))
    want = c.av_step("h3", None, video, audio, 600.0, row, patch_table=once,
                     **kw)
    # The halves sum in the module's pair, rounded on their own.
    assert all(_rel(g, w) < 1e-3 for g, w in zip(got, want))


def test_a_patch_alpha_that_is_no_number_is_a_400(host):
    """A patch's alpha is ComfyUI's scalar, sent as a number or null."""
    from test_video_h3_lora import _adapter

    c = _client()
    body = _table_body(c, _adapter(6))
    body["entries"][0]["patches"][0]["alpha"] = [2.0]
    with pytest.raises(StepsClientError) as e:
        c.define_table("h3", body)
    assert e.value.status == 400 and e.value.body["error"] == (
        "a patch's alpha is a number or null, not [2.0]")


def test_a_client_table_stacks_on_the_entrys_own_lora(tmp_path, request):
    """A ComfyUI table adds to the entry's own LoRA, as a request's
    adapters do on the images API: LoRA B on the entry steps as a table
    of the entry's LoRA and B on the entry without it. A table that names
    the entry's LoRA sets its strength for the step, applied once."""
    from test_video_h3_lora import _adapter

    own, b = _adapter(5), _adapter(7)
    video, audio, feats = _clip()
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)

    def step(with_own: bool, *loras):
        if with_own:
            _entry_with_own(tmp_path, request, seed=5)
        else:
            _mount(_Host(_served(_own_pipe()[0])), request)
        c = _client()
        tid = c.define_table("h3", _stack_body(c, *loras))
        got = c.av_step("h3", None, video, audio, 600.0,
                        {"cond": c.put(feats)}, patch_table=tid, **kw)
        steps.STATE.reset()
        return got

    stacked = step(True, (b, 1.0))
    for got, want in ((stacked, step(False, (own, 1.0), (b, 1.0))),
                      (step(True, (own, 0.5)), step(False, (own, 0.5)))):
        assert all(_rel(g, w) < 1e-5 for g, w in zip(got, want))
    assert _rel(stacked[0], step(False, (b, 1.0))[0]) > 1e-3


def _latents(route: str):
    """The latents of a 22-frame 32x32 clip, those the tiny VAEs take."""
    vshape, ashape = hf.latent_shapes(22, 32, 32)
    rng = np.random.default_rng(3)
    return rng.standard_normal(vshape if route == "video" else ashape).astype(
        np.float32)


@pytest.mark.parametrize("route", ["video", "audio"])
def test_a_decode_that_does_not_fit_beside_the_models_is_refused(host, vaes,
                                                                 route):
    """A clip's decode loads its VAE beside the DiT, which stays. Each is
    priced at the latents it was sent, a clip other than the entry's
    default: the audio decode at their length, not the entry's."""
    served = host.served
    served.spec.vae = vaes["vae"]
    served.spec.audio_vae = vaes["audio_vae"]
    c = _client()
    lat = _latents(route)
    if route == "video":
        clip = dict(frames=22, width=32, height=32)
    else:
        assert lat.shape[3] != hf.latent_shapes(FRAMES, SIZE, SIZE)[1][3]
        clip = dict(frames=FRAMES, width=SIZE, height=SIZE,
                    audio=int(lat.shape[3]))
    phase = "decode" if route == "video" else "audio"
    work = vs.plan_clip(served.plan, text=1, budget=1, **clip).phases[phase]
    adapters, caches = vs.held_bytes(served)
    host.budget_bytes = adapters + caches + work - 1
    with pytest.raises(StepsClientError) as e:
        c.container(f"vae/{route}/decode", {"latents": lat}, {"model": "h3"})
    assert e.value.status == 400
    assert "does not fit in memory" in e.value.body["error"]
    host.budget_bytes += 1
    c.container(f"vae/{route}/decode", {"latents": lat}, {"model": "h3"})


def test_the_audio_decode_gives_comfyuis_waveform_layout(host, vaes):
    """ComfyUI's AUDIO holds the waveform (batch, channels, samples)."""
    from gmlx.video.vae.reader import load_vae

    host.served.spec.audio_vae = vaes["audio_vae"]
    lat = _latents("audio")
    got, _f, _h = _client().container("vae/audio/decode", {"latents": lat},
                                      {"model": "h3"})

    def direct():
        avae = load_vae(vaes["audio_vae"], family="h3_audio",
                        dtype=mx.float32)
        return np.array(avae.decode(mx.array(lat)).astype(mx.float32))

    want = _on_worker(direct)
    wave = got["waveform"]
    assert wave.shape[:2] == (1, 2) and wave.shape[2] > 2
    assert np.array_equal(wave, want)


@pytest.mark.parametrize("route, model, want", [
    ("vae/decode", "h3", "'h3' is a video model, and vae/decode takes an "
     "image model"),
    ("step", "h3", "'h3' is a video model, and a step on x takes an image "
     "model"),
    ("av step", "zit", "'zit' is no video model; its step takes x"),
])
def test_each_route_refuses_a_model_of_the_other_kind(host, route, model,
                                                      want):
    """A raw client of the step API reaches these refusals. The pack sends
    VAEDecode on the gmlx video VAE to vae/video/decode, and the gmlx VAE
    Loader refuses a video model."""
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    host.budget_bytes = 64 << 30
    with pytest.raises(StepsClientError) as e:
        if route == "vae/decode":
            c.container("vae/decode", {"latents": video[:, :, 0]},
                        {"model": model})
        elif route == "step":
            c.step(model, None, video[:, :, 0], np.float32(600.0), [row])
        else:
            c.av_step(model, None, video, audio, 600.0, row, shifts=SHIFTS,
                      audio_scale=SCALE)
    assert e.value.status == 400 and e.value.body["error"] == want


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
    assert st["sampling"]["sigmas"] is None


def test_a_fasth3_status_hands_the_sampler_its_trained_sigmas(
        host, tmp_path, monkeypatch):
    """No stock scheduler makes FastH3's eight sigmas, so GmlxSigmas hands
    ComfyUI's sampler the video stream's from the status, as float32
    values, at the shifts the entry is configured with, which the status
    names beside them. At another step count the entry samples the even
    grid."""
    import dataclasses

    from gmlx.video import schedule

    spec = host.served.spec
    for name in ("path", "encoder", "vae", "audio_vae"):
        f = tmp_path / name
        f.write_bytes(b"x" * 16)
        setattr(spec, name, str(f))
    monkeypatch.setattr(steps, "family_of", lambda s: (hf.FAMILY, hf.FASTH3))
    plan = host.served.plan
    for shift in (10.0, 8.0):
        shifts = {"video": shift, "audio": 3.0}
        want = schedule.stream_sigmas(8, shifts, hf.FASTH3_RUNGS)["video"]
        for n, sigmas in ((8, [*map(float, want)]), (6, None)):
            monkeypatch.setattr(host.served, "plan", dataclasses.replace(
                plan, variant=hf.FASTH3, steps=n, shifts=shifts))
            st = _client().status("h3")
            assert st["variant"] == hf.FASTH3 and st["sampling"]["steps"] == n
            assert (st["sampling"]["shift"], st["sampling"]["audio_shift"]) == (shift, 3.0)
            assert st["sampling"]["sigmas"] == sigmas
        assert np.array_equal(np.asarray([*map(float, want)], np.float32), want)
