#!/usr/bin/env python3
"""The image step contract routes over a tiny Z-Image pipeline: the
handshake and version refusals, the loopback guard, uploads by hash, the
409 that names what is missing, patch table ids, the convention adapter,
the prefix cache answer, interrupts and runs. A sampling loop of the
reference client gives the latents of ``sample`` at the same noise, bit
for bit. The model is tiny and every MLX op runs on the image worker."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")
mx = pytest.importorskip("mlx.core")

from fastapi.testclient import TestClient  # noqa: E402

from gmlx.config import ImageServerCfg  # noqa: E402
from gmlx.image import api  # noqa: E402
from gmlx.image.registry import get_family  # noqa: E402
from gmlx.image.samplers import noise_from_seed  # noqa: E402
from gmlx.serve import image_steps as steps  # noqa: E402
from gmlx.serve import image_wire as wire  # noqa: E402
from gmlx.serve.image_service import WORKER  # noqa: E402
from gmlx.serve.image_steps_client import (MissingTensors,  # noqa: E402
                                           StepsClient, StepsClientError,
                                           array_hash, overhead_ms_median)
from gmlx.serve.patches import image_steps as routes  # noqa: E402

_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
P = wire.PREFIX
V = {wire.CONTRACT_HEADER: "1.0"}
SIGMAS = np.array([1.0, 0.75, 0.5, 0.25, 0.0], np.float32)
SHAPE = (1, 4, 4, 6)                  # 32x48 pixels at the tiny VAE scale


def _on_worker(fn, *args):
    return WORKER.submit(fn, *args).result()


def _tiny_pipe():
    from image_fixtures import tiny_z_model

    m, _ = tiny_z_model()
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


class _Host(routes.ServeHost):
    """Tiny pipelines served by id."""

    def __init__(self, pipes: dict):
        self.served = {
            mid: SimpleNamespace(
                pipe=p, adapters=(),
                spec=SimpleNamespace(id=mid, path=f"/abs/{mid}.gguf",
                                     encoder=None, vae="/abs/vae.gguf",
                                     encoder_model=None),
                plan=SimpleNamespace(keep_encoder=True))
            for mid, p in pipes.items()}
        self.held = 0

    def resolve(self, model):
        served = self.served.get(model or "zit")
        if served is None:
            raise LookupError(f"unknown image model {model!r}")
        return served.spec.path, served.spec

    def acquire(self, model):
        served = self.served[self.resolve(model)[1].id]
        self.held += 1
        host = self

        class Hold:
            _entry = SimpleNamespace(pipeline=served)

            def release(self):
                host.held -= 1

        return Hold()

    def models(self):
        return {mid: s.spec for mid, s in self.served.items()}

    def resident(self, path):
        return True

    def plan(self, spec):
        return None


@pytest.fixture(scope="module")
def pipe():
    return _on_worker(_tiny_pipe)


@pytest.fixture(scope="module")
def qpipe():
    def build():
        import qwen_tiny as qt

        return api.ImagePipeline(dit=qt.model_of(qt.weights()),
                                 family=get_family("qwen_image21"),
                                 variant="base")
    return _on_worker(build)


@pytest.fixture
def host(pipe, qpipe, request):
    h = _Host({"zit": pipe, "qit": qpipe})
    routes.mount(_APP.app, ImageServerCfg(), h)

    def done():
        steps.STATE.reset()
        for p in (pipe, qpipe):
            p.captions.clear()
        routes.mount(_APP.app, ImageServerCfg(steps_api=False))
    request.addfinalizer(done)
    return h


def _http(**kw):
    return TestClient(_APP.app, base_url="http://127.0.0.1", **kw)


def _client(http=None) -> StepsClient:
    http = http or _http()

    def call(method, path, body, headers):
        r = http.request(method, path, content=body, headers=headers)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, \
            r.content
    return StepsClient(call)


def _features(seed=0, n=5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, n, 16)).astype(np.float32)


def _direct(pipe, feats, *, neg=None, cfg=1.0, table=None, seed=3):
    """``sample`` latents from the seeded noise over SIGMAS, on the
    worker."""
    def run():
        c = api.Conditioning(mx.array(feats[0]), "direct")
        n = None if neg is None else api.Conditioning(mx.array(neg[0]),
                                                      "direct-neg")
        x = api.sample(pipe, c, neg=n, cfg=cfg, height=32, width=48,
                       noise=mx.array(noise_from_seed(seed, SHAPE)),
                       sigmas=SIGMAS, patch_table=table)
        return np.array(x)
    return _on_worker(run)


# Handshake and refusals.

def test_capabilities_answer_without_the_version_header(host):
    r = _http().get(f"{P}/capabilities")
    assert r.status_code == 200
    caps = r.json()
    assert caps["contract"] == {"major": 1, "minor": 0}
    assert caps["families"] == ["qwen_image21", "z_image"]
    assert caps["conventions"] == ["comfy"]
    assert caps["wire"]["container"] == "safetensors"


@pytest.mark.parametrize("value, match", [
    (None, "has no X-Gmlx-Steps-Contract header"),
    ("2.0", "Upgrade the gmlx server"),
    ("0.9", "Upgrade the client"),
])
def test_a_request_in_another_major_is_refused_naming_the_side(host, value,
                                                              match):
    headers = {} if value is None else {wire.CONTRACT_HEADER: value}
    r = _http().get(f"{P}/models", headers=headers)
    assert r.status_code == 400 and match in r.json()["error"]
    assert _http().get(f"{P}/models", headers=V).json()["models"] == [
        {"model": "zit", "family": None}, {"model": "qit", "family": None}]


def test_a_peer_off_loopback_is_refused_unless_steps_remote(host):
    far = _http(client=("10.0.0.5", 1))
    for path in ("capabilities", "models"):
        r = far.get(f"{P}/{path}", headers=V)
        assert r.status_code == 403
        assert "10.0.0.5" in r.json()["error"]
        assert "server.image.steps_remote" in r.json()["error"]
    routes.mount(_APP.app, ImageServerCfg(steps_remote=True), host)
    assert far.get(f"{P}/capabilities").status_code == 200


def test_steps_api_off_removes_the_routes(host):
    assert _http().get(f"{P}/capabilities").status_code == 200
    routes.mount(_APP.app, ImageServerCfg(steps_api=False), host)
    assert _http().get(f"{P}/capabilities").status_code == 404
    assert _http().post(f"{P}/step", headers=V).status_code in (404, 405)
    routes.mount(_APP.app, ImageServerCfg(), host)
    assert _http().get(f"{P}/capabilities").status_code == 200


def test_the_serve_install_reads_server_image(monkeypatch):
    seen = []
    monkeypatch.setattr(routes, "mount", lambda app, s, host=None,
                        model_dirs=(): seen.append((s, model_dirs)))
    cfg = SimpleNamespace(image=ImageServerCfg(steps_remote=True),
                          model_dirs=("/m",))
    routes.install_image_step_routes(cfg)
    assert seen == [(cfg.image, ("/m",))]


# Uploads by hash.

def test_a_tensor_is_put_once_by_its_hash(host):
    feats = _features()
    h = array_hash(feats)
    c = _http()
    assert c.head(f"{P}/tensors/{h}", headers=V).status_code == 404
    body = wire.write({"cond": feats})
    ct = {**V, "Content-Type": wire.CONTENT_TYPE}
    r = c.put(f"{P}/tensors/{h}", content=body, headers=ct)
    assert r.status_code == 200 and r.json() == {"hash": h}
    assert c.head(f"{P}/tensors/{h}", headers=V).status_code == 200
    # The wrong hash, a JSON body, two tensors and a bad digest.
    r = c.put(f"{P}/tensors/{'0' * 64}", content=body, headers=ct)
    assert r.status_code == 400 and r.json()["got"] == h
    r = c.put(f"{P}/tensors/{h}", content=b"{}",
              headers={**V, "Content-Type": "application/json"})
    assert r.status_code == 415
    two = wire.write({"a": feats, "b": feats})
    assert c.put(f"{P}/tensors/{h}", content=two,
                 headers=ct).status_code == 400
    assert c.head(f"{P}/tensors/XYZ", headers=V).status_code == 400


def test_a_binary_body_passes_the_serve_middleware(host):
    from gmlx.serve.patches.hardening import (
        install_api_key_auth,
        install_json_content_type_tolerance,
    )

    install_json_content_type_tolerance(_APP.app)
    install_api_key_auth("sekret")
    feats = _features(1)
    h = array_hash(feats)
    auth = {"Authorization": "Bearer sekret"}
    c = _http()
    ct = {**V, **auth, "Content-Type": wire.CONTENT_TYPE}
    assert c.put(f"{P}/tensors/{h}", content=wire.write({"c": feats}),
                 headers=ct).status_code == 200
    assert c.put(f"{P}/tensors/{h}", content=wire.write({"c": feats}),
                 headers={**V, "Content-Type": wire.CONTENT_TYPE}
                 ).status_code == 401
    # A JSON body sent without a content type is read as JSON.
    r = c.post(f"{P}/runs/begin", content=json.dumps({"model": "zit"}),
               headers={**V, **auth})
    assert r.status_code == 200 and r.json()["run"]


# Steps.

def _rows(client, feats, n=None):
    h = client.put(feats)
    return [{"cond": h, "num_tokens": n or feats.shape[1]}]


def test_a_step_names_every_hash_the_server_lacks_in_a_409(host):
    c = _client()
    feats = _features()
    ref = np.zeros((1, 4, 4, 6), np.float32)
    rows = [{"cond": array_hash(feats), "ref_latents": [array_hash(ref)]}]
    with pytest.raises(MissingTensors) as e:
        c.step("zit", None, np.zeros(SHAPE, np.float32), [0.5], rows,
               patch_table="ab" * 32)
    assert e.value.status == 409
    assert sorted(e.value.missing) == sorted(
        [array_hash(feats), array_hash(ref), "ab" * 32])
    # Upload what it named and the same step runs.
    c.put(feats)
    out = c.step("zit", None, np.zeros(SHAPE, np.float32), [0.5],
                 [{"cond": array_hash(feats)}])
    assert out.shape == SHAPE and out.dtype == np.float32
    assert host.held == 0


def test_the_contract_loop_gives_the_latents_of_sample(host, pipe):
    c = _client()
    feats = _features()
    run = c.begin("zit")
    got = c.sample("zit", noise_from_seed(3, SHAPE), SIGMAS,
                   _rows(c, feats), run=run)
    c.end(run)
    assert np.array_equal(got, _direct(pipe, feats))
    # Z-Image keeps no prefix state between steps.
    assert [r.prefix_cache for r in c.records] == ["none"] * (len(SIGMAS) - 1)
    assert all(r.forward_ms > 0 and r.wall_ms >= r.forward_ms
               for r in c.records)
    assert overhead_ms_median(c.records) >= 0


def test_a_guided_loop_gives_the_latents_of_sample(host, pipe):
    c = _client()
    pos, neg = _features(0), _features(1, n=3)
    got = c.sample("zit", noise_from_seed(3, SHAPE), SIGMAS,
                   _rows(c, pos), cfg=3.0, neg_rows=_rows(c, neg))
    assert np.array_equal(got, _direct(pipe, pos, neg=neg, cfg=3.0))


def test_num_tokens_and_an_attention_mask_trim_the_cond(host, pipe):
    c = _client()
    feats = _features(n=8)
    mask = np.array([[1, 1, 1, 1, 1, 0, 0, 0]], np.float32)
    rows = [{"cond": c.put(feats), "attention_mask": c.put(mask)}]
    got = c.sample("zit", noise_from_seed(3, SHAPE), SIGMAS, rows)
    assert np.array_equal(got, _direct(pipe, feats[:, :5]))
    bad = np.array([[1, 0, 1, 0, 0, 0, 0, 0]], np.float32)
    with pytest.raises(StepsClientError, match="prefix of the tokens"):
        c.step("zit", None, np.zeros(SHAPE, np.float32), [0.5],
               [{"cond": rows[0]["cond"], "attention_mask": c.put(bad)}])


def test_the_convention_maps_the_timestep_and_signs_the_output(
        host, pipe, monkeypatch):
    # ComfyUI hands both families sigma as the timestep and wants the
    # velocity back, so both conventions are the identity.
    for fam in ("z_image", "qwen_image21"):
        assert steps.CONVENTIONS["comfy"][fam] == steps.Convention()
    monkeypatch.setitem(steps.CONVENTIONS["comfy"], "z_image",
                        steps.Convention(scale=-1.0, shift=1.0, sign=-1.0))
    c = _client()
    feats = _features()
    x = noise_from_seed(5, SHAPE)
    out = c.step("zit", None, x, [0.25], _rows(c, feats))
    want = _on_worker(lambda: np.array(api.denoise_step(
        pipe, mx.array(x), 0.75, api.Conditioning(mx.array(feats[0]), "k"))))
    assert np.array_equal(out, -want)


# Patch tables.

def _table_body(c, scale=0.2, strength=0.8):
    from lora_fixtures import make_pair

    down, up, _ = make_pair(np.random.default_rng(4), 64, 64, 2,
                            scale=scale)
    return {"entries": [{
        "key": "diffusion_model.layers.0.attention.out.weight",
        "band": None,
        "patches": [{"up": c.put(up, kind="adapters"),
                     "down": c.put(down, kind="adapters"), "alpha": 2.0,
                     "strength": strength}]}]}, (down, up)


def test_a_patch_table_is_named_by_the_hash_of_its_json(host, pipe):
    c = _client()
    body, (down, up) = _table_body(c)
    tid = c.define_table("zit", body)
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert tid == hashlib.sha256(canon.encode()).hexdigest()
    # Another spelling of the same JSON has the same id.
    loose = json.dumps(body, indent=2).encode()
    r = _http().put(f"{P}/patch-tables/{tid}?model=zit", content=loose,
                    headers={**V, "Content-Type": "application/json"})
    assert r.status_code == 200 and r.json() == {"id": tid}
    r = _http().put(f"{P}/patch-tables/{'1' * 64}?model=zit", content=loose,
                    headers={**V, "Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["got"] == tid
    # The table steps as the core table of the same pair does.
    core = steps.STATE.tables_of("zit", pipe)[tid]
    feats = _features()
    got = c.sample("zit", noise_from_seed(3, SHAPE), SIGMAS, _rows(c, feats),
                   patch_table=tid)
    assert np.array_equal(got, _direct(pipe, feats, table=core))
    assert not np.array_equal(got, _direct(pipe, feats))


def test_a_table_whose_matrices_are_not_held_is_a_409(host):
    c = _client()
    body = {"entries": [{"key": "diffusion_model.layers.0.attention.out."
                         "weight", "band": None,
                         "patches": [{"up": "a" * 64, "down": "b" * 64,
                                      "strength": 1.0}]}]}
    with pytest.raises(MissingTensors) as e:
        c.define_table("zit", body)
    assert sorted(e.value.missing) == ["a" * 64, "b" * 64]


def _qwen_rows(c, case="ref2"):
    import qwen_tiny as qt

    rule = qt.rule
    feats = rule.features(case)[None]
    refs = [z[None] for z in rule.ref_latents(case)]
    return [{"cond": c.put(feats), "num_tokens": feats.shape[1],
             "ref_latents": [c.put(z) for z in refs],
             "image_slots": list(qt.CASES[case][1])}]


def test_a_qwen_run_keeps_its_prefix_and_a_table_change_misses_it(host,
                                                                   qpipe):
    import qwen_tiny as qt
    from lora_fixtures import make_pair

    c = _client()
    rows = _qwen_rows(c)
    down, up, _ = make_pair(np.random.default_rng(4), 32, 32, 2)
    tid = c.define_table("qit", {"entries": [{
        "key": "diffusion_model.transformer_blocks.0.attn.to_q.weight",
        "band": None,
        "patches": [{"up": c.put(up, kind="adapters"),
                     "down": c.put(down, kind="adapters"), "alpha": 2.0,
                     "strength": 0.8}]}]})
    run = c.begin("qit")
    x = np.asarray(qt.latents())
    plain = c.step("qit", run, x, [0.75], rows)
    plain2 = c.step("qit", run, x, [0.5], rows)
    patched = c.step("qit", run, x, [0.75], rows, patch_table=tid)
    patched2 = c.step("qit", run, x, [0.75], rows, patch_table=tid)
    c.end(run)
    # The table id is in the key of the prefix state.
    assert [r.prefix_cache for r in c.records] == ["miss", "hit", "miss",
                                                   "hit"]
    assert np.array_equal(patched, patched2)
    assert not np.array_equal(plain, patched)
    core = steps.STATE.tables_of("qit", qpipe)[tid]

    def want(sig, table=None):
        return _on_worker(lambda: np.array(api.denoise_step(
            qpipe, qt.latents(), sig, qt.conditioning("ref2", key="direct"),
            patch_table=table)))
    assert np.array_equal(plain, want(0.75))
    assert np.array_equal(plain2, want(0.5))
    assert np.array_equal(patched, want(0.75, core))


# Runs and interrupts.

def test_an_interrupt_stops_the_step_in_flight(host, monkeypatch):
    c = _client()
    rows = _rows(c, _features())
    run = c.begin("zit")
    real = steps.step_pacer

    def pacer(r, gen):
        fn = real(r, gen)

        def first(arrays):
            steps.STATE.interrupt(run)     # the client's interrupt lands
            fn(arrays)
        return first

    monkeypatch.setattr(steps, "step_pacer", pacer)
    x = noise_from_seed(3, SHAPE)
    with pytest.raises(StepsClientError) as e:
        c.step("zit", run, x, [0.5], rows)
    assert e.value.status == 409 and e.value.body["code"] == "interrupted"
    monkeypatch.setattr(steps, "step_pacer", real)
    # The run goes on with its next step.
    assert c.step("zit", run, x, [0.5], rows).shape == SHAPE
    assert c.interrupt(run) == 1 and c.interrupt("nope") == 0
    assert host.held == 1                     # the run's own hold


def test_a_run_ends_once_and_its_steps_are_refused_after(host):
    c = _client()
    rows = _rows(c, _features())
    run = c.begin("zit")
    assert host.held == 1
    c.step("zit", run, np.zeros(SHAPE, np.float32), [0.5], rows)
    assert c.end(run) == {"run": run, "ended": True}
    assert c.end(run) == {"run": run, "ended": False}
    assert host.held == 0
    with pytest.raises(StepsClientError, match="no run") as e:
        c.step("zit", run, np.zeros(SHAPE, np.float32), [0.5], rows)
    assert e.value.status == 404


def test_an_idle_run_ends_at_the_next_begin(host, monkeypatch):
    c = _client()
    now = [100.0]
    monkeypatch.setattr(steps.STATE, "clock", lambda: now[0])
    old = c.begin("zit")
    now[0] += steps.RUN_IDLE_S + 1
    c.begin("zit")
    assert old not in steps.STATE.runs and host.held == 1


# Status and encode.

def test_status_carries_the_turbo_sampling_schedule(host, monkeypatch):
    fam = {"zit": "z_image", "qit": "qwen_image21"}
    monkeypatch.setattr(steps, "family_of",
                        lambda spec: (get_family(fam[spec.id]), "turbo"))
    monkeypatch.setattr(api, "_header_bytes", lambda path, part: 7)
    c = _client()
    st = c.status("qit")
    row = get_family("qwen_image21").sampling_row("turbo")
    assert st["comfy"]["model_class"] == "QwenImage21"
    assert st["latent"] == {"channels": 64, "stride": 16, "patch": 1}
    assert st["sampling"]["sigmas"] == [*map(float, row.sigmas), 0.0]
    assert st["sampling"]["steps"] == len(st["sampling"]["sigmas"]) - 1
    assert st["memory"]["dit_bytes"] == 7
    # Z-Image Turbo samples on its family schedule, which has no fixed
    # sigmas.
    z = c.status("zit")
    assert z["comfy"]["model_class"] == "ZImage"
    assert z["sampling"]["sigmas"] is None and z["sampling"]["steps"] == 8


@pytest.mark.parametrize("as_container", [False, True])
def test_encode_answers_the_tensors_and_holds_them(host, monkeypatch,
                                                   as_container):
    seen = []
    cond = _features()
    ref = np.ones((1, 4, 4, 6), np.float32)

    def encode(served, text, images, borrow=None):
        seen.append((text, [i.shape for i in images]))
        return {"cond": cond, "refs": [ref], "image_slots": [2]}

    monkeypatch.setattr(steps, "encode", encode)
    c = _client()
    img = np.full((8, 8, 3), 9, np.uint8)
    if as_container:
        fields, got = c.encode("zit", "a cat", [img])
    else:
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        r = _http().post(f"{P}/encode", headers=V, json={
            "model": "zit", "text": "a cat",
            "images": [base64.b64encode(buf.getvalue()).decode()]})
        assert r.headers["content-type"] == wire.CONTENT_TYPE
        tensors, fields = wire.read(r.content)
        got = {k: t.numpy() for k, t in tensors.items()}
    assert seen == [("a cat", [(8, 8, 3)])]
    assert np.array_equal(got["cond"], cond)
    assert np.array_equal(got["ref_0"], ref)
    assert fields == {"cond": array_hash(cond),
                      "ref_latents": [array_hash(ref)],
                      "image_slots": [2], "num_tokens": 5}
    for h in (array_hash(cond), array_hash(ref)):
        assert h in steps.STATE.tensors


def test_cache_sizes_follow_server_image(host):
    routes.mount(_APP.app, dataclasses.replace(
        ImageServerCfg(), tensor_cache_mb=3, adapter_cache_mb=5), host)
    assert steps.STATE.tensors.max_bytes == 3 << 20
    assert steps.STATE.adapters.max_bytes == 5 << 20
