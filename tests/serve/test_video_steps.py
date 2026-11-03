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
                                           StepsClient, StepsClientError)
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
            plan=None)  # type: ignore
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
    # An image-to-video workflow wires its keyframe through GmlxCLIP,
    # which sends the image with the prompt.
    ("images", "the gmlx server encodes text alone for MiniMax-H3; it takes "
     "no images with the prompt, which keyframes and guides send"),
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
    held. The configured LoRA comes back with the DiT, and a table the
    pack sent is answered as missing until the pack sends it again."""
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
    assert seen == {"dit": None, "pairs": 0}
    assert strengths() == {0.75}
    after = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(after, before))
    with pytest.raises(MissingTensors) as e:
        c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid, **kw)
    assert e.value.body["missing"] == [tid]
    assert c.define_table("h3", body) == tid
    again = c.av_step("h3", None, video, audio, 600.0, row, patch_table=tid,
                      **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(again, sent))


def _table_body(c, adapter: dict) -> dict:
    """A contract table of ``adapter``'s pairs on ComfyUI's keys, as the
    pack sends a stock LoRA loader's patches."""
    entries = [{"key": f"diffusion_model.{target}.weight", "band": None,
                "patches": [{"up": c.put(up, kind="adapters"),
                             "down": c.put(down, kind="adapters"),
                             "alpha": float(alpha), "strength": 1.0,
                             "strength_model": 1.0}]}
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


def test_a_lora_that_does_not_load_again_after_an_encode_leaves_no_dit(
        tmp_path, monkeypatch, request):
    """The configured LoRA loads again with the DiT after each encode. A
    file moved since fails that encode and leaves the entry with no DiT
    and no table of the LoRA, so the next step loads both again once the
    file is back."""
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
    with pytest.raises(StepsClientError) as e:
        c.encode("h3", "a cat")
    assert e.value.status == 400
    assert str(path) in e.value.body["error"]
    assert pipe.dit is None and pipe.default_table == api.EMPTY_TABLE
    moved.rename(path)
    after = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(after, before))


def test_a_table_an_encode_lets_go_before_its_step_runs_is_missing(
        monkeypatch, request):
    """A second client's encode can run on the image worker between a
    step's check of its table and the step itself. The step answers the
    table as missing, and the pack sends it again."""
    from test_video_h3_lora import _adapter

    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    _stub_encode(monkeypatch, weights)
    c = _client()
    body = _table_body(c, _adapter(6))
    tid = c.define_table("h3", body)
    video, audio, feats = _clip()
    row = {"cond": c.put(feats)}
    kw = dict(shifts=SHIFTS, audio_scale=SCALE, patch_table=tid)
    sent = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    ensure, ran = vs.ensure_dit, []

    def encode_first(served):
        if not ran:
            ran.append(True)
            vs.drop_dit(served)         # what the encode leaves
        ensure(served)

    monkeypatch.setattr(vs, "ensure_dit", encode_first)
    with pytest.raises(MissingTensors) as e:
        c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert ran and e.value.body["missing"] == [tid]
    assert c.define_table("h3", body) == tid
    again = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert all(_rel(a, b) < 1e-6 for a, b in zip(again, sent))


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
        "a tensor of 80.0 MiB is over the step routes' tensor cache of 16.0 "
        "MiB; raise server.image.tensor_cache_mb")


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
    finally:
        steps.STATE.scale_caches(1.0)
    (t,) = read(write({"t": np.zeros((1, 1 << 19), np.float32)}))[0].values()
    with pytest.raises(steps.StepError) as limit:
        steps.STATE.put_tensor(steps.STATE.tensors, t.hash(), {"t": t},
                               1 << 20)
    assert e.value.status == limit.value.status == 413
    assert e.value.body["error"] == (
        "a tensor of 80.0 MiB is over the step routes' adapter cache of 64.0 "
        "MiB: the models loaded leave it 25% of its configured 256 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")
    assert str(limit.value) == (
        "a tensor of 2.0 MiB is over the upload limit of 1.0 MiB")


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
