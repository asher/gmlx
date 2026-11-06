"""The step routes on a tiny MiniMax-H3 pipeline: the AV step against
ComfyUI's forward rule for the audio its sampler carries, with anchors as
the gmlx pipeline runs them, the refined text rows kept by cond and patch
table, the memory checks of the step, the encodes and the decodes, the
encode with the DiT and its LoRA let go and loaded again, with keyframes
through the entry's mmproj, the entry's cache bounds, the refusals of
each route, the VAE routes against the VAEs, and the video model's
status. Every MLX op runs on the image worker."""
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


def _comfy_rule(pipe, video, audio, t, feats, anchors=(), noise=(),
                spans=()):
    """ComfyUI's forward around gmlx's network: the audio carried at
    sigma_a / sigma_v into the network, and its velocity converted back
    to the carried variable's. ``anchors`` (frame, video, audio) placed,
    with the ``noise`` of each one's video rows, and the cond's vision
    ``spans`` run as the gmlx pipeline runs them."""
    from gmlx.video.anchors import Anchor

    sv, sa = _comfy_sigmas(t)
    carry = sa / sv
    xa = (audio * carry).astype(np.float32)

    def run():
        dit = pipe.dit
        text = dit.caption(mx.array(feats))
        rows = tuple(layout.AnchorRows(
            f, 0 if v is None else v.shape[2], 0 if a is None else a.shape[3])
            for f, v, a in anchors)
        lay = layout.pack(hf.FAMILY.timeline, feats.shape[1],
                          *video.shape[2:], audio.shape[3], rows, spans)
        placed = tuple(Anchor(f, None if v is None else mx.array(v),
                              None if a is None else mx.array(a))
                       for f, v, a in anchors)
        cv, ca = dit.cond_rows(placed, 0, noise=[mx.array(n) for n in noise])
        vv, va = vapi.denoise_step(dit, mx.array(video), mx.array(xa),
                                   {"video": float(sv), "audio": float(sa)},
                                   text, lay, cond_video=cv, cond_audio=ca)
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


def test_the_refined_rows_keep_to_the_bytes_the_entry_books():
    """The entry keeps refined rows of any count within the bytes it
    books for them, CAPTIONS prompts of the priced rows, and drops the
    oldest past that; rows over the bound alone are not kept and drop
    nothing."""
    dim = 4
    dit = SimpleNamespace(cfg=SimpleNamespace(dim=dim), caption=lambda f: f)
    served = ServedVideo(pipe=SimpleNamespace(dit=dit), spec=None,
                         plan=None)  # type: ignore

    def put(key, rows):
        feats = mx.zeros((1, rows, dim), mx.float32)
        mx.eval(feats)
        return _on_worker(vs.caption, served, key, feats, "t")[0]
    assert vs.caption_bytes(dim) == 64 << 10
    for i in range(vs.CAPTIONS + 2):
        put(f"c{i}", 64)                # 1 KiB each
    assert len(served.captions) == vs.CAPTIONS + 2
    put("big", 3840)                    # 60 KiB
    assert [k for k, _ in served.captions] == [f"c{i}" for i in range(6, 10)
                                               ] + ["big"]
    assert put("huge", 4200).shape == (1, 4200, dim)
    assert [k for k, _ in served.captions] == [f"c{i}" for i in range(6, 10)
                                               ] + ["big"]


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


def _anchor_parts(seed=4):
    """A lone first frame's video latent with its sound and the noise of
    its video rows, and a lone frame's latent and noise for the last
    frame, on the 5-frame 64x64 clip."""
    rng = np.random.default_rng(seed)
    vshape, _a = hf.latent_shapes(FRAMES, SIZE, SIZE)
    one = (1, vshape[1], 1, *vshape[3:])
    # The rows of a lone frame at patch (1, 2, 2): (h/2 * w/2, C * 4).
    rows = (vshape[3] * vshape[4] // 4, vshape[1] * 4)

    def draw(shape):
        return rng.standard_normal(shape).astype(np.float32)
    return {"first": draw(one), "first_noise": draw(rows),
            "sound": draw((1, 32, 2, 3)), "last": draw(one),
            "last_noise": draw(rows)}


def _anchored(c, parts) -> tuple:
    """The step's anchors by hash, and ComfyUI's (frame, video, audio)
    and noise of them placed."""
    h = {k: c.put(v) for k, v in parts.items()}
    sent = [{"frame": 0, "video": h["first"], "noise": h["first_noise"],
             "audio": h["sound"]},
            {"frame": -1, "video": h["last"], "noise": h["last_noise"]}]
    placed = [(0, parts["first"], parts["sound"]),
              (FRAMES - 1, parts["last"], None)]
    return sent, placed, [parts["first_noise"], parts["last_noise"]]


def test_an_av_step_with_anchors_and_spans_runs_them_as_the_pipeline(
        host, pipe):
    """The anchors are placed by ComfyUI's guide rule, the last frame's
    counted from the end, their rows mixed with the noise each was sent
    with, and the cond's vision spans run with the video tag, as the
    gmlx pipeline runs a clip with keyframes and a guide."""
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    t = np.float32(600)
    sent, placed, noise = _anchored(c, _anchor_parts())
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    vv, va = c.av_step("h3", None, video, audio, t, row, anchors=sent,
                       spans=[[1, 3]], **kw)
    want_v, want_a = _comfy_rule(pipe, video, audio, t, feats, placed, noise,
                                 ((1, 3),))
    assert _rel(vv, want_v) < 1e-5 and _rel(va, want_a) < 1e-5
    plain_v, plain_a = c.av_step("h3", None, video, audio, t, row, **kw)
    assert _rel(vv, plain_v) > 1e-3 and _rel(va, plain_a) > 1e-3


def _refused_step(c, row, fields_of, parts=None):
    """The refusal of a step whose anchors and spans ``fields_of`` makes
    from the anchors sent, in a raw body as any client may send it."""
    video, audio, _f = _clip()
    sent, _p, _n = _anchored(c, parts or _anchor_parts())
    anchors, spans = fields_of(sent)
    fields = {"model": "h3", "rows": [row], "audio_scale": SCALE,
              "shifts": SHIFTS}
    for k, v in (("anchors", anchors), ("spans", spans)):
        if v is not None:
            fields[k] = v
    tensors = {"video": video, "audio": audio,
               "timestep": np.asarray([600.0], np.float32)}
    with pytest.raises(StepsClientError) as e:
        c.container("step", tensors, fields)
    return e.value


def _with(i, **kw):
    def fields_of(sent):
        sent = [dict(a) for a in sent]
        sent[i].update(kw)
        for k in [k for k, v in sent[i].items() if v is None]:
            del sent[i][k]
        return sent, None
    return fields_of


@pytest.mark.parametrize("case, want", [
    ("list", "an AV step's anchors are a list"),
    ("frame", "anchors[1] is an object with an integer frame"),
    ("noise", "anchors[0] names the noise of its video rows with its video "
     "latent, and with it alone"),
    ("canvas", "anchors[1]'s video latent is [1, 24, 1, 2, 2], not (1, 24, "
     "T, 4, 4) on the clip's canvas"),
    ("guide", "anchors[1]'s video latent holds 3 latent frames, no "
     "MiniMax-H3 guide clip: 1, or 5 k + 2"),
    ("outside", "anchors[1]: frame 5 is outside the clip's 5 frames"),
    ("sound", None),
    ("noise_shape", "anchor noise (4, 95) is not the shape of its rows "
     "(4, 96)"),
    ("spans", "vision spans [(4, 9)] are not ascending spans of a 6-row "
     "prefix"),
])
def test_an_av_step_refuses_anchors_that_are_no_part_of_its_clip(
        host, monkeypatch, case, want):
    """Each refusal names the anchor and what is wrong with it, before
    the step's turn on the worker, which would load the DiT an encode let
    go. The noise is checked against the rows the DiT makes, in the
    turn."""
    _v, audio, feats = _clip()
    c = _client()
    if case != "noise_shape":
        def turn(served):
            raise AssertionError("the refused step took its turn")
        monkeypatch.setattr(vs, "ensure_dit", turn)
    row = {"cond": c.put(feats)}
    parts = _anchor_parts()
    if case == "list":
        fields_of = lambda sent: ({"frame": 0}, None)  # noqa: E731
    elif case == "frame":
        fields_of = _with(1, frame=1.5)
    elif case == "noise":
        fields_of = _with(0, noise=None)
    elif case == "canvas":
        fields_of = _with(1, video=c.put(parts["last"][..., :2, :2]))
    elif case == "guide":
        fields_of = _with(1, video=c.put(np.repeat(parts["last"], 3, 2)))
    elif case == "outside":
        fields_of = _with(1, frame=FRAMES)
    elif case == "sound":
        t = audio.shape[3]
        want = (f"anchors[0]'s audio holds {t + 1} latent frames, past the "
                f"{t} the clip's audio holds from frame 0")
        fields_of = _with(0, audio=c.put(np.zeros((1, 32, 2, t + 1),
                                                  np.float32)))
    elif case == "noise_shape":
        fields_of = _with(1, noise=c.put(parts["last_noise"][:, :95]))
    else:
        fields_of = lambda sent: (sent, [[4, 9]])  # noqa: E731
    e = _refused_step(c, row, fields_of, parts)
    assert e.status == 400 and e.body["error"] == want


def test_an_av_step_names_the_anchor_tensors_it_does_not_hold(host):
    """One 409 names each tensor of the step the server does not hold,
    the cond and the anchors', so the client sends them all at once."""
    video, audio, _f = _clip()
    c = _client()
    # A cond no test has sent: the tensor store outlives a test.
    cond = array_hash(np.full((1, 6, TINY["text_dim"]), 3.0, np.float32))
    sent, _p, _n = _anchored(c, _anchor_parts())
    lost = array_hash(np.full((1, 32, 2, 2), 7.0, np.float32))
    sent[0]["audio"] = lost
    with pytest.raises(MissingTensors) as e:
        c.av_step("h3", None, video, audio, 600.0, {"cond": cond},
                  anchors=sent, shifts=SHIFTS, audio_scale=SCALE)
    assert e.value.missing == sorted([cond, lost])


@pytest.mark.parametrize("ask", ["anchors", "spans"])
def test_a_fasth3_step_refuses_anchors_and_vision_spans(host, monkeypatch,
                                                         ask):
    """FastH3 runs text to video only; a vision span is a keyframe's
    image, so it counts as an anchor."""
    import dataclasses

    monkeypatch.setattr(host.served, "plan", dataclasses.replace(
        host.served.plan, variant=hf.FASTH3))
    c = _client()
    row = {"cond": c.put(_clip()[2])}

    def fields_of(sent):
        return (sent, None) if ask == "anchors" else (None, [[1, 3]])
    e = _refused_step(c, row, fields_of)
    n = "2 anchors" if ask == "anchors" else "1 anchor"
    assert e.status == 400 and e.body["error"] == (
        f"MiniMax-H3 fasth3 runs text to video only, and this request has "
        f"{n}; run it without keyframes, guides or audio anchors")


def test_a_step_with_anchors_counts_their_rows(host):
    """The anchors' rows join the packed sequence the step is priced at:
    a budget that fits the clip's step alone refuses it with anchors."""
    video, audio, feats = _clip()
    c = _client()
    row = {"cond": c.put(feats)}
    sent, _p, _n = _anchored(c, _anchor_parts())
    served = host.served
    rows = (layout.AnchorRows(0, 1, 3), layout.AnchorRows(FRAMES - 1, 1, 0))

    def need(anchors):
        return vs.plan_clip(served.plan, frames=FRAMES, width=SIZE,
                            height=SIZE, text=6, decode=False, budget=1,
                            anchors=anchors).terms["denoise"]["step"]
    assert need(rows) > need(())
    kw = dict(shifts=SHIFTS, audio_scale=SCALE)
    # The first step keeps the cond's refined rows, which the caches count.
    c.av_step("h3", None, video, audio, 600.0, row, **kw)
    adapters, caches = vs.held_bytes(served)
    host.budget_bytes = adapters + caches + need(rows) - 1
    c.av_step("h3", None, video, audio, 600.0, row, **kw)
    with pytest.raises(StepsClientError) as e:
        c.av_step("h3", None, video, audio, 600.0, row, anchors=sent, **kw)
    assert e.value.status == 400 and e.value.body["error"] == (
        "a step of 64x64 at 5 frames does not fit in memory beside the "
        "models now loaded")
    host.budget_bytes += 1
    c.av_step("h3", None, video, audio, 600.0, row, anchors=sent, **kw)


@pytest.mark.parametrize("ask, want", [
    # An image-to-video request sends its first or last frame with the
    # prompt, which the encoder's vision tower reads.
    ("images", "video model 'h3' names no encoder_mmproj, the vision tower "
     "of its text encoder that reads keyframes; name the encoder's mmproj "
     "GGUF as encoder_mmproj in its entry"),
    # ComfyUI's node resizes a keyframe to the clip's canvas first.
    ("canvas", "keyframe image 1 is 48x32, no MiniMax-H3 canvas: a keyframe "
     "is the clip's canvas, in multiples of 32"),
    ("fasth3", "MiniMax-H3 fasth3 runs text to video only, and this request "
     "has 2 anchors; run it without keyframes, guides or audio anchors"),
    ("prompt_upsampling", "video model 'h3' takes no prompt upsampling; "
     "send the request without prompt_upsampling"),
])
def test_the_h3_encode_refuses_what_it_does_not_run_by_name(
        host, monkeypatch, ask, want):
    import dataclasses

    c = _client()
    images = [np.zeros((32, 32, 3), np.uint8)]
    if ask != "images":
        host.served.spec.encoder_mmproj = "/abs/mmproj.gguf"
    if ask == "canvas":
        images.append(np.zeros((32, 48, 3), np.uint8))
    if ask == "fasth3":
        images.append(images[0])
        monkeypatch.setattr(host.served, "plan", dataclasses.replace(
            host.served.plan, variant=hf.FASTH3))
    with pytest.raises(StepsClientError) as e:
        if ask == "prompt_upsampling":
            c.container("encode", {}, {"model": "h3", "text": "a cat",
                                       "prompt_upsampling": True})
        else:
            c.encode("h3", "a cat", images=images)
    assert e.value.status == 400 and e.value.body["error"] == want


def _keyframe_encoder(monkeypatch, served) -> list:
    """The encoder load replaced by tests/video's tiny Qwen3-VL with its
    tower, built on the image worker at each load; returns the (path,
    mmproj, DiT held) of each load."""
    import os

    from test_video_keyframes import _Pieces, _tiny_h3_encoder

    from gmlx.video import encoder as video_encoder

    served.spec.encoder_mmproj = "~/mmproj.gguf"
    served.tokenizer = _Pieces({"<Picture 1>: ": [11, 12, 13],
                                "<Picture 2>: ": [11, 12, 14],
                                "a fox": [21, 22]})
    loads = []

    def load_encoder(fam, path, mmproj=None, log=None):
        loads.append((path, mmproj, served.pipe.dit is not None))
        return _tiny_h3_encoder()

    monkeypatch.setattr(video_encoder, "load_encoder", load_encoder)
    assert os.path.expanduser("~/mmproj.gguf") != "~/mmproj.gguf"
    return loads


def test_an_encode_with_keyframes_reads_them_through_the_entrys_mmproj(
        monkeypatch, request):
    """The encoder loads alone with the entry's mmproj, and the features
    and vision spans are the keyframe encode's on the images, their alpha
    dropped, as the step takes them back."""
    import os

    from test_video_keyframes import _tiny_h3_encoder

    from gmlx.video.families.h3 import keyframes as kf

    pipe, _w = _own_pipe()
    served = _served(pipe)
    _mount(_Host(served), request)
    loads = _keyframe_encoder(monkeypatch, served)
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8),
              rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)]
    fields, got = _client().encode("h3", "a fox", images=images)
    rgb = [im[..., :3] for im in images]

    def direct():
        feats, spans = kf.encode(_tiny_h3_encoder(), "a fox", rgb)
        return np.array(feats), spans

    want, spans = _on_worker(direct)
    assert loads == [("/abs/enc.gguf", os.path.expanduser("~/mmproj.gguf"),
                      False)]
    assert fields["spans"] == [list(sp) for sp in spans] == [[3, 11],
                                                             [14, 20]]
    assert fields["num_tokens"] == want.shape[1]
    np.testing.assert_allclose(got["cond"], want, rtol=1e-6, atol=1e-6)
    text, _ = _client().encode("h3", "a fox")
    assert "spans" not in text and loads[-1][1] is None


def test_an_encode_with_keyframes_counts_their_rows_and_the_tower(
        monkeypatch, request):
    """The encode of a prompt with keyframes is priced at the prompt's
    rows with their vision blocks through the tower, as the video command
    prices it."""
    import dataclasses

    from gmlx.video import memfit

    pipe, _w = _own_pipe()
    served = _served(pipe)
    served.plan = dataclasses.replace(served.plan, mmproj_bytes=8 << 20,
                                      tower_bytes=16 << 20)
    host = _mount(_Host(served), request)
    _keyframe_encoder(monkeypatch, served)
    sizes = [(64, 64)]
    rows = vs.prompt_rows(served, "a fox", sizes)
    need = vs.plan_clip(
        served.plan, frames=FRAMES, width=SIZE, height=SIZE, text=rows,
        decode=False, budget=1,
        blocks=memfit.keyframe_blocks(hf.FAMILY, sizes)).phases["encode"]
    assert need > vs.plan_clip(served.plan, frames=FRAMES, width=SIZE,
                               height=SIZE, text=rows, decode=False,
                               budget=1).phases["encode"]
    _adapters, caches = vs.held_bytes(served)
    host.budget_bytes = caches + need - 1
    images = [np.zeros((64, 64, 3), np.uint8)]
    c = _client()
    with pytest.raises(StepsClientError) as e:
        c.encode("h3", "a fox", images=images)
    assert e.value.status == 400 and e.value.body["error"] == (
        f"the encode of {rows} text rows with 1 keyframe does not fit in "
        "memory beside the models now loaded; try a shorter prompt or a "
        "smaller size, or one keyframe")
    host.budget_bytes += 1
    c.encode("h3", "a fox", images=images)


def test_an_encode_loads_the_encoder_without_the_dit_or_any_lora(
        tmp_path, monkeypatch, request):
    """The H3 encoder and the DiT with its 8-step LoRA do not fit a 32 GB
    Mac together: the encoder loads with neither the DiT nor a LoRA pair
    held. The configured LoRA comes back with the DiT at the next step,
    and a table the pack sent is defined again at its next step from the
    body the server keeps."""
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
        return {st for _path, _band, patches in entries
                for _k, st, _a in patches}

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


@pytest.mark.parametrize("between", ["encode", "define", "lost"])
def test_a_table_let_go_before_its_step_runs_steps(monkeypatch, request,
                                                   between):
    """A second client's encode, or the define of its table on a pair
    cache with room for one table, can run on the image worker between a
    step's check of its table and the step itself. The step defines the
    table again in its own turn, from the body the server keeps, and
    steps as before. With the table's matrices gone from the adapter
    cache too, the step answers the table as missing, and steps as
    before once the pack sends it again."""
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
            if between == "lost":
                steps.STATE.adapters.clear()
        ensure(served)

    monkeypatch.setattr(vs, "ensure_dit", other_first)
    if between == "lost":
        with pytest.raises(MissingTensors) as e:
            c.av_step("h3", None, video, audio, 600.0, row, **kw)
        assert e.value.body["missing"] == [tid]
        # The client sends the matrices again, as the pack's sync does.
        c._uploaded.clear()
        assert c.define_table("h3", _table_body(c, _adapter(6))) == tid
    again = c.av_step("h3", None, video, audio, 600.0, row, **kw)
    assert ran and all(_rel(a, b) < 1e-6 for a, b in zip(again, sent))


def test_a_table_whose_pairs_another_table_shed_steps_in_its_turn(request):
    """Two ComfyUI clients whose LoRAs each fill the entry's pair room, with
    its built tables bounded as a loaded entry bounds them: B's table
    sheds the pairs of A's. A's next step, on a cond the server lacks, is
    a 409 that names the cond alone, since the server holds the table's
    body and matrices, and A's step with its cond defines the table again
    in its own turn and steps as before. Once B's step sheds A's pairs
    again and the adapter cache lets the matrices go, one 409 names the
    cond and the table, and A steps as before once the pack sends its
    table again."""
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

    def lacking() -> list:
        with pytest.raises(MissingTensors) as e:
            ca.av_step("h3", None, video, audio, 600.0, {"cond": lacked},
                       patch_table=ta, **kw)
        return sorted(e.value.body["missing"])

    assert lacking() == [lacked]
    again = ca.av_step("h3", None, video, audio, 600.0, row, patch_table=ta,
                       **kw)
    assert all(_rel(x, y) < 1e-6 for x, y in zip(again, first))
    cb.av_step("h3", None, video, audio, 600.0, {"cond": cb.put(feats)},
               patch_table=tb, **kw)
    steps.STATE.adapters.clear()
    assert lacking() == sorted([lacked, ta])
    ca._uploaded.clear()
    assert ca.define_table("h3", _table_body(ca, a)) == ta
    again = ca.av_step("h3", None, video, audio, 600.0, row, patch_table=ta,
                       **kw)
    assert all(_rel(x, y) < 1e-6 for x, y in zip(again, first))


def _two_av_clients(request, monkeypatch=None):
    """An entry whose pair cache holds one of two clients' LoRAs, so each
    define sheds the other's pairs, with the H3 encoder stood in when
    ``monkeypatch`` is given. Returns the pipeline and, for each client,
    a step of its table, its define and its first step's output."""
    from test_video_h3_lora import _adapter

    pipe, weights = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    if monkeypatch is not None:
        _stub_encode(monkeypatch, weights)
    pipe.adapters.max_bytes = sum(d.nbytes + u.nbytes
                                  for d, u, _a in _adapter(6).values())
    video, audio, feats = _clip()
    out = []
    for seed in (6, 7):
        c = _client()
        body, row = _table_body(c, _adapter(seed)), {"cond": c.put(feats)}
        tid = c.define_table("h3", body)

        def step(c=c, row=row, tid=tid):
            return c.av_step("h3", None, video, audio, 600.0, row,
                             patch_table=tid, shifts=SHIFTS,
                             audio_scale=SCALE)

        def define(c=c, body=body, tid=tid):
            assert c.define_table("h3", body) == tid

        out.append((step, define, step()))
    return pipe, out


def _av_steps_as_first(clients) -> None:
    for step, _define, first in clients:
        assert all(_rel(x, y) < 1e-6 for x, y in zip(step(), first))


def test_two_av_clients_whose_defines_interleave_step_without_a_409(request):
    """Two ComfyUI clients alone on a video entry whose pair cache holds
    one of their LoRAs. A step on a table whose pairs the other's define
    shed defines it again in its own turn and steps as before, and so
    does a step after both define again, A then B, before A's step
    reaches the server."""
    pipe, clients = _two_av_clients(request)
    _av_steps_as_first(clients)
    for _step, define, _first in clients:
        define()
    _av_steps_as_first(clients)
    assert pipe.adapters.bytes <= pipe.adapters.max_bytes


def test_av_tables_a_third_clients_encode_let_go_step_without_a_409(
        monkeypatch, request):
    """A third client's encode lets the DiT and every LoRA pair go. Each of
    two clients' next steps defines its table again in its own turn and
    steps as before, and so does each after both define again, A then B,
    before A's step reaches the server."""
    pipe, clients = _two_av_clients(request, monkeypatch)
    _client().encode("h3", "a cat")
    assert pipe.dit is None and pipe.adapters.bytes == 0
    _av_steps_as_first(clients)
    for _step, define, _first in clients:
        define()
    _av_steps_as_first(clients)
    assert pipe.adapters.bytes <= pipe.adapters.max_bytes


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
    the other's encode let go is defined again at its next step, in the
    step's own turn."""
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
        return me.c.av_step("h3", None, video, audio, 600.0, me.row,
                            patch_table=me.tid, **kw)

    encode("a", "a red fox")
    first_a = step("a")
    encode("b", "a blue bird")
    again_a = step("a")
    first_b = step("b")
    encode("a", "a red fox")
    again_b = step("b")
    third_a = step("a")
    assert len(loads) == 3
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


@pytest.mark.parametrize("cap_mb", [None, 256])
def test_a_loaded_video_entry_bounds_its_caches_at_its_share(tmp_path,
                                                            monkeypatch,
                                                            cap_mb):
    """The pool admits an entry with each tensor cache at the share of its
    bound that its clip leaves room for. The entry bounds its pipeline's
    tensor caches at that share, and its pair and tables caches at the cap,
    or no bound with none, beside the bytes its plan books for its own
    LoRA, which hold its pairs: the pool books a request's pairs. The step
    routes' caches take the smallest share of the entries loaded until it
    is released."""
    import dataclasses

    from gmlx.serve import image_service

    monkeypatch.setitem(image_service._CACHE_BYTES, "adapter",
                        None if cap_mb is None else cap_mb << 20)

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
    tensor, cap = image_service.cache_bytes()
    room = steps.UNBOUNDED if cap is None else cap
    assert 0 < pipe.adapters.bytes <= size
    assert pipe.conditionings.max_bytes == tensor // 4
    assert pipe.reference_latents.max_bytes == tensor // 4
    assert pipe.adapters.max_bytes == pipe.tables.max_bytes == room + size
    assert scaled == [0.25]
    _on_worker(vs._release, [got])
    assert scaled == [0.25, 1.0]


def test_the_step_tensor_cache_keeps_a_loaded_video_entrys_cond(
        monkeypatch, request):
    """The step routes' tensor cache takes the smallest share of the
    entries loaded and the largest cond floor: beside an image
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
    tensor, _cap = image_service.cache_bytes()
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
    # With no cap the upload store has no bound at any share.
    assert bounds == (rows.nbytes, steps.UNBOUNDED)
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
        # A share whose bound is 17.49 MiB.
        steps.STATE.scale_caches(0.2734)
        with pytest.raises(StepsClientError) as share:
            c.put(np.zeros((1, 70 << 16), np.float32))
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
    # A bound under a tenth of a MiB below the tensor reads under it.
    assert share.value.body["error"] == (
        "a tensor of 17.5 MiB is over the step routes' tensor cache of 17.4 "
        "MiB: the models loaded leave it 27% of its configured 64 MiB. A "
        "smaller defaults.size or defaults.seconds on the tightest entry "
        "gives it more.")


def test_the_adapter_cache_and_the_upload_limit_are_named_in_a_413(host):
    """The share the models loaded leave bounds the step routes' tensor
    cache alone: at a quarter, an adapter matrix of 80 MiB is held. An
    adapter matrix over a set server.image.adapter_cache_mb is refused
    with the key named, and a tensor over a route's own upload limit with
    the limit."""
    from gmlx.serve.image_wire import read, write

    c = _client()
    big = np.zeros((1, 20 << 20), np.float32)
    try:
        steps.STATE.scale_caches(0.25)
        h = c.put(big, kind="adapters")
        held = steps.STATE.adapters.nbytes(h)
        steps.STATE.configure(ImageServerCfg(adapter_cache_mb=16))
        with pytest.raises(StepsClientError) as edge:
            c.put(np.zeros((1, (4 << 20) + 1), np.float32), kind="adapters")
    finally:
        steps.STATE.scale_caches(1.0)
        steps.STATE.configure(ImageServerCfg())
    limit, just = [], []
    for n, got in ((1 << 19, limit), ((1 << 18) + 1, just)):
        (t,) = read(write({"t": np.zeros((1, n), np.float32)}))[0].values()
        with pytest.raises(steps.StepError) as refused:
            steps.STATE.put_tensor(steps.STATE.tensors, t.hash(), {"t": t},
                                   1 << 20)
        got.append(str(refused.value))
    assert held == big.nbytes
    assert limit == ["a tensor of 2.0 MiB is over the upload limit of 1.0 MiB"]
    # A tensor a few bytes over a bound reads above it.
    assert just == ["a tensor of 1.1 MiB is over the upload limit of 1.0 MiB"]
    assert edge.value.status == 413
    assert edge.value.body["error"] == (
        "a tensor of 16.1 MiB is over the configured 16 MiB of the step "
        "routes' adapter cache; raise server.image.adapter_cache_mb")


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


_CAP_413 = ("the LoRA pairs of this patch table take 4.0 MiB, over the "
            "2.0 MiB the pair cache of 'h3' keeps for a request's pairs; "
            "raise server.image.adapter_cache_mb to 4 or more")


@pytest.mark.parametrize("configured_mb, share, want", [
    (2, 0.25, _CAP_413), (2, 1.0, _CAP_413)])
def test_a_table_over_the_entrys_pair_room_is_413_naming_it(
        monkeypatch, request, configured_mb, share, want):
    """The pairs of a ComfyUI LoRA's table over a set
    server.image.adapter_cache_mb are refused with a 413 that names the
    cache, the sizes and the MiB the key must reach, at any share, before
    a pair is made."""
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


def test_a_table_under_the_cap_defines_at_any_share(monkeypatch, request):
    """The share the entry's clip leaves its tensor caches bounds its pair
    cache no more: at a share of a quarter, a table whose pairs fill a set
    server.image.adapter_cache_mb holds them all."""
    from test_video_h3_lora import _adapter

    from gmlx.serve import image_service

    pipe, _w = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    a = _adapter(6)
    need = sum(d.nbytes + u.nbytes for d, u, _a in a.values())
    monkeypatch.setitem(image_service._CACHE_BYTES, "adapter", need)
    image_service.bound_caches(pipe, 0, 0.25)
    c = _client()
    c.define_table("h3", _table_body(c, a))
    assert pipe.adapters.bytes == need and len(pipe.adapters) == len(a)


@pytest.mark.parametrize("width, room, need, room_mib, bound", [
    (8192, (4 << 20) - 1, "4.0", "3.9", 4), (8200, 4 << 20, "4.1", "4.0", 5)])
def test_a_table_just_over_the_pair_room_reads_its_need_above_it(
        monkeypatch, request, width, room, need, room_mib, bound):
    """The 413 rounds the pairs' MiB up and the cap's down, so a table a
    byte over the cap, or a few KiB, never reads at or under it."""
    from gmlx.serve import image_service

    pipe, _w = _own_pipe()
    _mount(_Host(_served(pipe)), request)
    c = _client()
    body = _big_table(c, 4, width)
    monkeypatch.setitem(image_service._CACHE_BYTES, "adapter", room)
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
    the old pairs are no longer its own: they count against a set
    server.image.adapter_cache_mb, and a table of them over it is a 413
    with its remedy."""
    import os

    from lora_fixtures import write_keys
    from test_video_h3_lora import _adapter, _comfy_keys

    from gmlx.serve import image_service
    from gmlx.serve.image_service import _pair_keys

    path = _lora_file(tmp_path)
    pipe, weights = _own_pipe()
    served = _served(pipe)
    served.spec.adapters = ((str(path), 1.0),)
    _on_worker(vs._load_adapters, served)
    _mount(_Host(served), request)
    _stub_encode(monkeypatch, weights)
    held = pipe.adapters.bytes
    monkeypatch.setitem(image_service._CACHE_BYTES, "adapter", held // 2)
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


def _encode_inputs(route: str, n: int = 22):
    """Frames in [0, 1] of an ``n``-frame 32x32 clip, those the tiny VAE
    takes, or a waveform of 1000 samples, past one latent's 800."""
    rng = np.random.default_rng(5)
    if route == "video":
        return {"frames": rng.random((n, 32, 32, 3), np.float32)}
    return {"waveform": (rng.random((1, 2, 1000), np.float32) * 2 - 1)}


@pytest.mark.parametrize("route, n", [("video", 22), ("video", 1),
                                      ("audio", 0)])
def test_the_vae_encodes_are_the_vaes_on_comfyuis_layouts(host, vaes, route,
                                                          n):
    """The pack sends ComfyUI's IMAGE frames and AUDIO waveform; the
    latents are the gmlx VAEs' on them, the frames in [-1, 1] as
    ComfyUI's video VAE takes them, and the waveform padded to whole
    latents."""
    from gmlx.video.vae.reader import load_vae

    host.served.spec.vae = vaes["vae"]
    host.served.spec.audio_vae = vaes["audio_vae"]
    sent = _encode_inputs(route, n)
    got, _f, _h = _client().container(f"vae/{route}/encode", sent,
                                      {"model": "h3"})

    def direct():
        if route == "video":
            vae = load_vae(vaes["vae"], family="h3", dtype=mx.float16)
            x = mx.array(sent["frames"])[None] * 2.0 - 1.0
        else:
            vae = load_vae(vaes["audio_vae"], family="h3_audio",
                           dtype=mx.float32)
            x = mx.array(sent["waveform"])
        return np.array(vae.encode(x).astype(mx.float32))

    want = _on_worker(direct)
    lat = got["latents"]
    if route == "video":
        t = 1 if n == 1 else hf.latent_shapes(n, 32, 32)[0][2]
        assert lat.shape == (1, 24, t, 2, 2)
    else:
        assert lat.shape == (1, 32, 2, 2)
    assert np.array_equal(lat, want)


@pytest.mark.parametrize("route", ["video", "audio"])
def test_a_vae_encode_that_does_not_fit_beside_the_models_is_refused(
        host, vaes, route):
    """An anchor's encode loads its VAE beside the DiT, which stays, and
    is priced as the video command prices the anchors of a run: a guide
    clip at its frames and size, a sound at its latents."""
    served = host.served
    served.spec.vae = vaes["vae"]
    served.spec.audio_vae = vaes["audio_vae"]
    c = _client()
    sent = _encode_inputs(route)
    wire_bytes = 0
    if route == "video":
        clip = dict(frames=22, width=32, height=32,
                    anchors=(layout.AnchorRows(0, 7, 0),))
        # The body and its MLX copy.
        wire_bytes = 2 * sent["frames"].nbytes
    else:
        clip = dict(frames=FRAMES, width=SIZE, height=SIZE,
                    anchors=(layout.AnchorRows(0, 0, 2),))
    work = vs.plan_clip(served.plan, text=1, budget=1, **clip
                        ).phases["anchors"]
    adapters, caches = vs.held_bytes(served)
    host.budget_bytes = adapters + caches + work + wire_bytes - 1
    with pytest.raises(StepsClientError) as e:
        c.container(f"vae/{route}/encode", sent, {"model": "h3"})
    assert e.value.status == 400
    if route == "video":
        need = (work + wire_bytes) / (1 << 30)
        free = (work + wire_bytes - 1) / (1 << 30)
        assert e.value.body == {"error": (
            "a video encode of a guide clip of 22 frames of 32x32 does not "
            "fit in memory beside the models now loaded: its window of 22 "
            f"frames, f32 on the wire, needs {need:.2f} GiB, and "
            f"{free:.2f} GiB is free; no window fits"), "window": 0}
    else:
        assert e.value.body["error"] == (
            "an audio encode of 1000 samples does not fit in memory beside "
            "the models now loaded")
    host.budget_bytes += 1
    c.container(f"vae/{route}/encode", sent, {"model": "h3"})


@pytest.mark.parametrize("route, sent, model, want", [
    ("video", np.zeros((7, 32, 32, 3), np.float32), "h3",
     "frames [7, 32, 32, 3] are no MiniMax-H3 keyframe or guide clip: 1 or "
     "17 k + 5 frames on a canvas in multiples of 32"),
    ("video", np.zeros((1, 32, 48, 3), np.float32), "h3",
     "frames [1, 32, 48, 3] are no MiniMax-H3 keyframe or guide clip: 1 or "
     "17 k + 5 frames on a canvas in multiples of 32"),
    ("video", np.zeros((1, 32, 32, 3), np.bool_), "h3",
     "vae/video/encode takes frames (T, H, W, 3) in [0, 1] or uint8 levels, "
     "not BOOL [1, 32, 32, 3]"),
    ("video", np.zeros((1, 32, 32, 3), np.float32), "zit",
     "'zit' is an image model; vae/video/encode takes a video model"),
    ("audio", np.zeros((1, 1, 800), np.float32), "h3",
     "a waveform is (1, 2, n) at 32000 Hz, not [1, 1, 800]"),
    ("audio", np.zeros((2, 800), np.float32), "h3",
     "vae/audio/encode takes the waveform of one sound (1, 2, n)"),
])
def test_a_vae_encode_refuses_what_is_no_anchor(host, route, sent, model,
                                                want):
    key = "frames" if route == "video" else "waveform"
    with pytest.raises(StepsClientError) as e:
        _client().container(f"vae/{route}/encode", {key: sent},
                            {"model": model})
    assert e.value.status == 400 and e.value.body["error"] == want


def _levels(n: int) -> np.ndarray:
    return np.random.default_rng(6).integers(0, 256, (n, 32, 32, 3),
                                             dtype=np.uint8)


def test_uint8_levels_encode_as_their_float32(host, vaes):
    """Frames as uint8 levels give the latents of the same frames as
    float32 k / 255, bit for bit, at a keyframe and a clip."""
    host.served.spec.vae = vaes["vae"]
    c = _client()
    for n in (1, 22):
        u8 = _levels(n)
        f32 = u8.astype(np.float32) / np.float32(255)
        got = [c.container("vae/video/encode", {"frames": f}, {"model": "h3"}
                           )[0]["latents"] for f in (u8, f32)]
        assert got[0].shape == got[1].shape and np.array_equal(*got), n


def test_windows_that_tile_a_clip_encode_as_the_whole_clip(host, vaes):
    """39 frames as two open windows of one piece and the 5 frames of the
    clip's end give the latents of the clip in one body, bit for bit."""
    host.served.spec.vae = vaes["vae"]
    c = _client()
    u8 = _levels(39)
    whole = c.container("vae/video/encode", {"frames": u8},
                        {"model": "h3"})[0]["latents"]
    parts = [c.container("vae/video/encode", {"frames": u8[at:at + n]},
                         {"model": "h3", "at": at, "clip_frames": 39,
                          "open": is_open})[0]["latents"]
             for at, n, is_open in ((0, 17, True), (17, 17, True),
                                    (34, 5, False))]
    assert [p.shape[2] for p in parts] == [5, 5, 2]
    assert np.array_equal(np.concatenate(parts, axis=2), whole)


RULES = ("breaks the rules: a guide clip goes in windows, each at frame "
         "`at` of `clip_frames`: an open window is 17 m frames from a "
         "multiple of 17, 1 <= m <= 7, ending before the clip does; the "
         "last window is the clip's 1 frame or 17 j + 5 frames from a "
         "multiple of 17 to the clip's end, j <= 7")


@pytest.mark.parametrize("n, at, clip, is_open", [
    (16, 0, 39, True),        # an open window of no whole piece
    (17, 5, 39, True),        # off a piece boundary
    (34, 0, 34 + 5, False),   # a last window that is not 17 j + 5
    (17, 0, 22, False),       # a last window short of the clip's end
    (136, 0, 141, True),      # 8 pieces in an open window
    (141, 0, 141, False),     # 8 pieces and the end in one body
    (22, 17, 22, False),      # past the clip's end
])
def test_a_window_off_the_rules_is_refused_by_them(host, n, at, clip,
                                                   is_open):
    with pytest.raises(StepsClientError) as e:
        _client().container("vae/video/encode", {"frames": _levels(n)},
                            {"model": "h3", "at": at, "clip_frames": clip,
                             "open": is_open})
    open_ = " (open)" if is_open else ""
    assert e.value.status == 400 and e.value.body["error"] == (
        f"a window of {n} frames at frame {at} of a {clip}-frame clip{open_} "
        + RULES)


def _asgi_request(body: bytes, seen: list, length: int | None = None,
                  step: int = 4096):
    """A Request whose body arrives ``step`` bytes at a time after its
    safetensors header, each pull of it booked in ``seen``."""
    from starlette.requests import Request

    head = 8 + int.from_bytes(body[:8], "little")
    chunks = [body[:head]] + [body[i:i + step]
                              for i in range(head, len(body), step)]

    async def receive():
        i = sum(1 for x in seen if x.startswith("chunk"))
        seen.append(f"chunk {i}")
        return {"type": "http.request", "body": chunks[i],
                "more_body": i + 1 < len(chunks)}

    n = len(body) if length is None else length
    scope = {"type": "http", "method": "POST", "path": "/", "query_string":
             b"", "headers": [(b"content-type", wire.CONTENT_TYPE.encode()),
                              (b"content-length", str(n).encode())]}
    return Request(scope, receive), len(chunks)


def test_a_body_past_the_limit_is_refused_before_a_byte_is_read(
        host, monkeypatch):
    """Its Content-Length alone refuses it, 413, with no pull of the body
    and no hold of the model."""
    import asyncio

    monkeypatch.setattr(routes, "MAX_BODY_BYTES", 1 << 16)
    seen: list = []
    monkeypatch.setattr(host, "acquire", lambda m: seen.append("acquire"))
    body = wire.write({"frames": _levels(22)}, {"model": "h3"})
    req, _ = _asgi_request(body, seen)
    with pytest.raises(steps.StepError) as e:
        asyncio.run(routes.video_encode(req))
    assert e.value.status == 413 and str(e.value) == (
        f"the body is {len(body)} bytes, over the limit of 65536")
    assert seen == []


def test_a_window_is_priced_from_its_header_and_its_frames_drained(
        host, monkeypatch):
    """The window's price is read before a byte of its frames: the body
    arrives a chunk at a time, the refusal comes after the header alone,
    and the rest is read and dropped so a client still sending gets it.
    The refusal names the largest window that fits, 3 pieces."""
    import asyncio

    served = host.served
    adapters, caches = vs.held_bytes(served)

    def need(f, t):
        work = vs.plan_clip(served.plan, frames=max(served.plan.frames, f),
                            width=32, height=32, text=1, budget=1,
                            anchors=(layout.AnchorRows(0, t, 0),)
                            ).phases["anchors"]
        return work + 2 * f * 32 * 32 * 3

    host.budget_bytes = adapters + caches + need(56, 17)
    seen: list = []
    budget = host.budget
    monkeypatch.setattr(host, "budget",
                        lambda hold: seen.append("price") or budget(hold))
    monkeypatch.setattr(vs, "encode_video", lambda *a, **k: seen.append(
        "encode"))
    body = wire.write({"frames": _levels(119)},
                      {"model": "h3", "at": 0, "clip_frames": 362,
                       "open": True})
    req, chunks = _asgi_request(body, seen)
    with pytest.raises(steps.StepError) as e:
        asyncio.run(routes.video_encode(req))
    assert e.value.status == 400 and e.value.extra == {"window": 3}
    assert "does not fit in memory beside the models now loaded" in str(
        e.value)
    assert seen.index("price") < seen.index("chunk 1")
    assert seen.count("price") >= 1 and "encode" not in seen
    assert seen[-1] == f"chunk {chunks - 1}"

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
