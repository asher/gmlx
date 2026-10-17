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
        from gmlx.serve.residency import _BusyHold

        served = self.served[self.resolve(model)[1].id]
        self.held += 1
        host = self

        class Pool:
            def release(self, entry):
                host.held -= 1

        return _BusyHold(Pool(), SimpleNamespace(pipeline=served))

    def models(self):
        return {mid: s.spec for mid, s in self.served.items()}

    def resident(self, path):
        return True

    def plan(self, spec):
        return None

    budget_bytes = None

    def budget(self, hold):
        return self.budget_bytes


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


def _client(http=None, **kw) -> StepsClient:
    http = http or _http()

    def call(method, path, body, headers):
        r = http.request(method, path, content=body, headers=headers)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, \
            r.content
    return StepsClient(call, **kw)


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
    assert caps["families"] == ["flux2_klein", "flux2_klein_9b",
                                "flux2_klein_9b_kv", "flux3_klein", "krea2",
                                "qwen_image21", "z_image"]
    assert caps["conventions"] == ["comfy", "reference"]
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


def test_the_model_list_names_the_default_image_model(host, monkeypatch):
    import gmlx.serve.bridge_vlm as serving
    monkeypatch.setattr(serving, "default_image_model_id", lambda: "qit")
    assert _http().get(f"{P}/models", headers=V).json()["default_model"] == "qit"
    monkeypatch.setattr(serving, "default_image_model_id", lambda: None)
    assert _http().get(f"{P}/models", headers=V).json()["default_model"] is None


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


def test_the_media_gate_passes_tensor_bodies_and_still_reads_json(host):
    """The media gate parses every body as JSON; a step route's container
    passes it, up to the routes' own limit, and a JSON body to the same
    prefix is still checked."""
    from fastapi import FastAPI

    from gmlx.serve.patches import media_gate as mg
    from gmlx.serve.patches.hardening import (
        install_json_content_type_tolerance,
    )

    app = FastAPI()
    routes.mount(app, ImageServerCfg(), host)
    install_json_content_type_tolerance(app)
    mg.install_media_gate(False, app=app)
    c = TestClient(app, base_url="http://127.0.0.1")
    feats = _features(2)
    h = array_hash(feats)
    ct = {**V, "Content-Type": wire.CONTENT_TYPE}
    r = c.put(f"{P}/tensors/{h}", content=wire.write({"c": feats}), headers=ct)
    assert r.status_code == 200, r.text
    assert c.head(f"{P}/tensors/{h}", headers=V).status_code == 200
    # A step body passes too, and its answer is a container.
    body = wire.write({"x": np.zeros(SHAPE, np.float32),
                       "timestep": np.array([0.5], np.float32)},
                      {"model": "zit", "rows": [{"cond": h}]})
    r = c.post(f"{P}/step", content=body, headers=ct)
    assert r.status_code == 200 and r.headers["content-type"] == \
        wire.CONTENT_TYPE
    # The same bytes typed as JSON are refused by the gate.
    r = c.put(f"{P}/tensors/{h}", content=wire.write({"c": feats}),
              headers={**V, "Content-Type": "application/json"})
    assert r.status_code == 400 and "not valid JSON" in r.text
    assert mg._is_step_tensors(SimpleNamespace(
        url=SimpleNamespace(path="/v1/chat/completions"),
        headers={"content-type": wire.CONTENT_TYPE})) is False


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
    # ComfyUI hands every family sigma as the timestep and wants the
    # velocity back, so every convention maps the identity. Its FLUX.2
    # forward alone embeds the fp32 time, at both sizes and on KV.
    for fam in ("z_image", "qwen_image21", "krea2", "flux3_klein"):
        assert steps.CONVENTIONS["comfy"][fam] == steps.Convention()
    for fam in ("flux2_klein", "flux2_klein_9b", "flux2_klein_9b_kv"):
        assert steps.CONVENTIONS["comfy"][fam] == steps.Convention(
            time="fp32")
    # The reference convention is comfy's on the reference time rule.
    assert steps.CONVENTIONS["reference"] == {
        fam: dataclasses.replace(c, time="reference")
        for fam, c in steps.CONVENTIONS["comfy"].items()}
    monkeypatch.setitem(steps.CONVENTIONS["comfy"], "z_image",
                        steps.Convention(scale=-1.0, shift=1.0, sign=-1.0))
    c = _client()
    feats = _features()
    x = noise_from_seed(5, SHAPE)
    out = c.step("zit", None, x, [0.25], _rows(c, feats))
    want = _on_worker(lambda: np.array(api.denoise_step(
        pipe, mx.array(x), 0.75, api.Conditioning(mx.array(feats[0]), "k"))))
    assert np.array_equal(out, -want)


# FLUX.3 klein: a flow map reads the time its step goes to from the
# client's schedule, and the VAE routes carry model-space latents.

F3_SHAPE = (1, 8, 4, 6)                 # 96x64 pixels at the stride of 16


@pytest.fixture(scope="module")
def fpipes():
    """The tiny FLUX.3 klein DiT of the check 3 self-test, as a flow map
    and as Base."""
    def build():
        import flux3_gates as fg
        from oracle_fixture import committed_fixture

        weights, m = committed_fixture("selftest_flux3_klein_tiny_weights")
        cfg = m["inputs"]["config"]
        fam = get_family("flux3_klein")
        return {flow: api.ImagePipeline(
            dit=fg.tiny_model(weights, cfg, flow_map=flow), family=fam,
            variant="distilled" if flow else "base")
            for flow in (True, False)}
    return _on_worker(build)


def _serve(host, mid, pipe):
    """``pipe`` served as ``mid`` beside the host's own."""
    host.served[mid] = SimpleNamespace(
        pipe=pipe, adapters=(),
        spec=SimpleNamespace(id=mid, path=f"/abs/{mid}.gguf", encoder=None,
                             vae="/abs/vae.gguf", encoder_model=None),
        plan=SimpleNamespace(keep_encoder=True))


def _f3_velocity(pipe, x, sigma, cond, target=None):
    return _on_worker(lambda: np.array(api.denoise_step(
        pipe, mx.array(x), sigma, cond, target=target)))


def _f3_cond(feats, ref=None):
    return api.Conditioning(
        mx.array(feats[0]), "k",
        ref_latents=() if ref is None else (mx.array(ref[0]),))


def test_a_flux3_flow_map_step_goes_to_the_next_time_of_the_schedule(
        host, fpipes):
    """Each step of a flow map goes to the entry of ``sample_sigmas``
    after the one nearest its timestep, as ComfyUI's forward takes it,
    with and without a reference, and two rows go together."""
    pipe = fpipes[True]
    _serve(host, "f3d", pipe)
    c = _client()
    rng = np.random.default_rng(7)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    ref = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    rows = _rows(c, feats)
    for i in range(len(SIGMAS) - 1):
        out = c.step("f3d", None, x, [SIGMAS[i]], rows, sample_sigmas=SIGMAS)
        assert np.array_equal(out, _f3_velocity(
            pipe, x, SIGMAS[i], _f3_cond(feats), SIGMAS[i + 1])), i
    # A timestep between two entries goes after the nearest, and the last
    # entry to itself.
    for t, to in ((0.55, 0.25), (0.0, 0.0)):
        out = c.step("f3d", None, x, [t], rows, sample_sigmas=SIGMAS)
        assert np.array_equal(out, _f3_velocity(
            pipe, x, t, _f3_cond(feats), to)), t
    with_ref = [{"cond": rows[0]["cond"], "ref_latents": [c.put(ref)]}]
    out = c.step("f3d", None, x, [0.75], with_ref, sample_sigmas=SIGMAS)
    assert np.array_equal(out, _f3_velocity(
        pipe, x, 0.75, _f3_cond(feats, ref), 0.5))
    two = c.step("f3d", None, np.concatenate([x, x]), [0.75, 0.75],
                 rows + with_ref, sample_sigmas=SIGMAS)
    assert np.array_equal(two[1:], out)
    assert host.held == 0


def test_a_flux3_step_takes_its_references_in_the_order_of_the_row(
        host, fpipes):
    """Two references of other shapes reach the DiT in the order of the
    row, and the other order gives another step."""
    pipe = fpipes[True]
    _serve(host, "f3d", pipe)
    c = _client()
    rng = np.random.default_rng(11)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    a = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    b = rng.standard_normal((1, 8, 3, 2)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    row = {"cond": _rows(c, feats)[0]["cond"],
           "ref_latents": [c.put(a), c.put(b)]}
    out = c.step("f3d", None, x, [0.75], [row], sample_sigmas=SIGMAS)

    def cond(*refs):
        # No prefix key, which the other steps here hold with other rows.
        return api.Conditioning(mx.array(feats[0]), "", ref_latents=tuple(
            mx.array(r[0]) for r in refs))

    want = _f3_velocity(pipe, x, 0.75, cond(a, b), 0.5)
    assert np.array_equal(out, want)
    assert not np.array_equal(out, _f3_velocity(pipe, x, 0.75, cond(b, a),
                                                0.5))
    assert host.held == 0


def test_a_flux3_flow_map_step_without_its_schedule_is_refused(host,
                                                                fpipes):
    _serve(host, "f3d", fpipes[True])
    c = _client()
    rows = _rows(c, np.ones((1, 10, 24), np.float32))
    x = np.zeros(F3_SHAPE, np.float32)
    with pytest.raises(StepsClientError, match="is a flow map: a step body "
                       "holds sample_sigmas"):
        c.step("f3d", None, x, [0.5], rows)
    for bad in ([0.5], [1.0, np.nan]):
        with pytest.raises(StepsClientError, match="2 or more finite"):
            c.step("f3d", None, x, [0.5], rows, sample_sigmas=bad)
    assert host.held == 0


def test_a_flux3_base_step_reads_no_schedule(host, fpipes):
    pipe = fpipes[False]
    _serve(host, "f3b", pipe)
    c = _client()
    feats = np.random.default_rng(8).standard_normal(
        (1, 10, 24)).astype(np.float32)
    x = noise_from_seed(6, F3_SHAPE)
    want = _f3_velocity(pipe, x, 0.5, _f3_cond(feats))
    for sched in (None, SIGMAS):
        out = c.step("f3b", None, x, [0.5], _rows(c, feats),
                     sample_sigmas=sched)
        assert np.array_equal(out, want)


def test_a_flux3_table_on_comfyuis_native_keys_steps_as_the_core_table(
        host, fpipes):
    """The pack names a FLUX.3 pair by ComfyUI's weight key, the GGUF name
    under diffusion_model., in a block and outside the blocks."""
    from lora_fixtures import make_pair

    pipe = fpipes[True]
    _serve(host, "f3d", pipe)
    c = _client()
    rng = np.random.default_rng(10)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    plain = _f3_velocity(pipe, x, 0.75, _f3_cond(feats), 0.5)
    for name, (dout, din) in (("single_blocks.0.q_proj", (64, 64)),
                              ("ref_time_in.in_layer", (64, 256)),
                              ("txt_in", (64, 24))):
        down, up, _ = make_pair(rng, dout, din, 2)
        tid = c.define_table("f3d", {"entries": [{
            "key": f"diffusion_model.{name}.weight", "band": None,
            "patches": [{"up": c.put(up, kind="adapters"),
                         "down": c.put(down, kind="adapters"), "alpha": 2.0,
                         "strength": 0.8}]}]})
        core = steps.STATE.tables_of("f3d", pipe)[tid]
        out = c.step("f3d", None, x, [0.75], _rows(c, feats),
                     patch_table=tid, sample_sigmas=SIGMAS)
        want = _on_worker(lambda: np.array(api.denoise_step(
            pipe, mx.array(x), 0.75, _f3_cond(feats), patch_table=core,
            target=0.5)))
        assert np.array_equal(out, want), name
        assert not np.array_equal(out, plain), name


def test_the_flux3_status_names_comfyuis_flux3_on_model_space_latents(
        host, fpipes, monkeypatch):
    monkeypatch.setattr(steps, "family_of", lambda spec: (
        get_family("flux3_klein"), "distilled"))
    monkeypatch.setattr(api, "_header_bytes", lambda path, part: 7)
    _serve(host, "f3d", fpipes[True])
    st = _client().status("f3d")
    assert st["comfy"] == {"model_class": "Flux3", "image_model": "flux3"}
    # ComfyUI's FLUX.3 edit hands the step the latents of its references.
    assert st["edit_capable"]
    assert st["latent"] == {"channels": 128, "stride": 16, "patch": 1}
    assert st["vae"] == {"channels_out": 3, "stride": 16, "space": "model"}


def test_the_flux2_9b_status_names_comfyuis_flux2_as_4b_does(
        host, f2pipe, monkeypatch):
    """The step contract carries the 9B checkpoints of FLUX.2 klein, which
    check 8 ran through ComfyUI, as ComfyUI's Flux2 on model space latents
    and the size rule of the 4B schedule."""
    monkeypatch.setattr(steps, "family_of", lambda spec: (
        get_family("flux2_klein_9b"), "distilled"))
    monkeypatch.setattr(api, "_header_bytes", lambda path, part: 7)
    _serve(host, "f2n", f2pipe)
    st = _client().status("f2n")
    assert st["family"] == "flux2_klein_9b"
    assert st["comfy"] == {"model_class": "Flux2", "image_model": "flux2"}
    assert st["edit_capable"]
    assert st["latent"] == {"channels": 128, "stride": 16, "patch": 1}
    assert st["vae"] == {"channels_out": 3, "stride": 16, "space": "model"}
    assert st["sampling"] == {"steps": 4, "cfg": 1.0, "shift": 1.0,
                              "sigmas": None, "schedule": {"rule": "flux2"}}


def test_the_flux3_vae_routes_carry_model_space_latents(host, fpipes,
                                                         monkeypatch):
    """ComfyUI's FLUX.3 latents are the model's: the routes hand them to
    the core VAE passes as they are, where the other families map them
    to and from the VAE's own space."""
    _model_space_vae_routes(host, fpipes[True], "f3d", monkeypatch)


def _model_space_vae_routes(host, pipe, mid, monkeypatch):
    def no_map(_z):
        raise AssertionError(f"{pipe.family.title} latents are mapped to no "
                             "other space")

    pipe = dataclasses.replace(pipe, vae=SimpleNamespace(
        config=SimpleNamespace(in_channels=3), to_model_space=no_map,
        to_vae_space=no_map))
    _serve(host, mid, pipe)
    seen = []

    def decode(p, z, *, alpha=False, **kw):
        seen.append(np.array(z))
        return mx.full((1, 32, 48, 3), 0.25)

    lat = np.random.default_rng(9).standard_normal(
        (2, 128, 2, 3)).astype(np.float32)
    monkeypatch.setattr(api, "vae_decode", decode)
    monkeypatch.setattr(api, "vae_encode", lambda p, img, **kw: mx.array(
        lat[:1]))
    c = _client()
    got, _, _ = c.container("vae/decode", {"latents": lat}, {"model": mid})
    assert got["image"].shape == (2, 32, 48, 3)
    assert [z.shape for z in seen] == [(1, 128, 2, 3)] * 2
    assert np.array_equal(np.concatenate(seen), lat)
    got, _, _ = c.container("vae/encode",
                            {"image": np.zeros((1, 32, 48, 3), np.float32)},
                            {"model": mid})
    assert np.array_equal(got["latents"], lat[:1])


def test_flow_targets_follow_comfyuis_rule():
    s = np.array([1.0, 0.75, 0.5, 0.0], np.float32)
    assert steps.flow_targets(s, [1.0, 0.8, 0.6, 0.5, 0.3, 0.0]) == [
        0.75, 0.5, 0.0, 0.0, 0.0, 0.0]
    # A tie goes to the first of the two nearest, as torch's argmin does.
    assert steps.flow_targets(s, [0.625]) == [0.5]




# FLUX.2 klein: ComfyUI's Flux2 forward takes sigma as the timestep,
# embeds it in fp32 and returns the native velocity, reads the reference
# latents of a row in its order, and its VAE routes carry model-space
# latents, as FLUX.3's do.

@pytest.fixture(scope="module")
def f2pipe():
    """The tiny FLUX.2 klein DiT of the check 3 self-test."""
    def build():
        import flux2_gates as fg
        from oracle_fixture import committed_fixture

        weights, m = committed_fixture("selftest_flux2_klein_tiny_weights")
        return api.ImagePipeline(
            dit=fg.tiny_model(weights, m["inputs"]["config"]),
            family=get_family("flux2_klein"), variant="distilled")
    return _on_worker(build)


@pytest.fixture(scope="module")
def kvpipe():
    """The tiny FLUX.2 klein DiT of the check 3 self-test on the KV row,
    whose own step runs in parts."""
    def build():
        import flux2_gates as fg
        from oracle_fixture import committed_fixture

        weights, m = committed_fixture("selftest_flux2_klein_tiny_weights")
        return api.ImagePipeline(
            dit=fg.tiny_model(weights, m["inputs"]["config"]),
            family=get_family("flux2_klein_9b_kv"), variant="distilled")
    return _on_worker(build)


def _f2_cond(feats, *refs, key="k"):
    return api.Conditioning(mx.array(feats[0]), key, ref_latents=tuple(
        mx.array(r[0]) for r in refs))


def _f2_velocity(pipe, x, sigma, cond, table=None, time="fp32"):
    return _on_worker(lambda: np.array(api.denoise_step(
        pipe, mx.array(x), sigma, cond, patch_table=table, time=time)))


def test_a_flux2_step_is_the_core_step_with_and_without_references(
        host, f2pipe):
    """Each step equals denoise_step at the timestep as sigma on the fp32
    time rule, with no schedule read, with a reference and with two rows.
    At 0.75 the reference's rule embeds 752, the bf16 of 750, and the
    step embeds 750."""
    _serve(host, "f2d", f2pipe)
    c = _client()
    rng = np.random.default_rng(7)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    ref = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    rows = _rows(c, feats)
    for t in SIGMAS[:-1]:
        for sched in (None, SIGMAS):
            out = c.step("f2d", None, x, [t], rows, sample_sigmas=sched)
            assert np.array_equal(out, _f2_velocity(
                f2pipe, x, t, _f2_cond(feats))), t
    with_ref = [{"cond": rows[0]["cond"], "ref_latents": [c.put(ref)]}]
    assert not np.array_equal(c.step("f2d", None, x, [0.75], rows),
                              _f2_velocity(f2pipe, x, 0.75, _f2_cond(feats),
                                           time="reference"))
    out = c.step("f2d", None, x, [0.75], with_ref)
    assert np.array_equal(out, _f2_velocity(f2pipe, x, 0.75,
                                            _f2_cond(feats, ref)))
    assert not np.array_equal(out, _f2_velocity(f2pipe, x, 0.75,
                                                _f2_cond(feats)))
    two = c.step("f2d", None, np.concatenate([x, x]), [0.75, 0.75],
                 rows + with_ref)
    assert np.array_equal(two[1:], out)
    assert host.held == 0


def test_a_kv_edit_under_comfyuis_kv_cache_node_extracts_once_a_run(
        host, kvpipe):
    """Without ComfyUI's Flux KV Cache node a KV step runs whole, its
    references at the step's time. Under the node the first step of a run
    runs them at time zero and keeps their K and V, which the later steps
    of the run read. The step caches count them until the run ends."""
    from gmlx.serve import image_service

    _serve(host, "kv", kvpipe)
    c = _client()
    rng = np.random.default_rng(12)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    ref = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    x0, x1 = noise_from_seed(5, F3_SHAPE), noise_from_seed(6, F3_SHAPE)
    row = {"cond": _rows(c, feats)[0]["cond"], "ref_latents": [c.put(ref)]}

    def want():
        m = kvpipe.dit
        f, refs = mx.array(feats), m.references([mx.array(ref)])
        s0, s1 = np.full(1, 0.75, np.float32), np.full(1, 0.5, np.float32)
        m0, m1 = m.modulation(s0, 1, "fp32"), m.modulation(s1, 1, "fp32")
        v0, kv = m.extract(mx.array(x0), s0, f, refs=refs, mod=m0)
        return [np.array(v) for v in (
            m(mx.array(x0), s0, f, refs=refs, mod=m0), v0,
            m(mx.array(x1), s1, f, mod=m1, cache=kv))]
    plain, v0, v1 = _on_worker(want)
    try:
        assert np.array_equal(c.step("kv", None, x0, [0.75], [row]), plain)
        run = c.begin("kv")
        assert np.array_equal(
            c.step("kv", run, x0, [0.75], [row], ref_cache=True), v0)
        (held,) = steps.STATE.runs[run].ref_cache.values()
        caches = steps.STATE.tensors.bytes + steps.STATE.adapters.bytes
        assert image_service._steps_cache_bytes() == caches + held.nbytes
        assert held.nbytes == kvpipe.dit.ref_cache_bytes([ref[0]])
        assert np.array_equal(
            c.step("kv", run, x1, [0.5], [row], ref_cache=True), v1)
        c.end(run)
        assert image_service._steps_cache_bytes() == caches
    finally:
        kvpipe.captions.clear()
    assert host.held == 0


def test_each_row_of_a_kv_step_under_the_node_keeps_its_own_cache(
        host, kvpipe):
    """The K and V of a reference depend on the latents of the row, as in
    ComfyUI, whose cache keeps the batch axis. Two rows on one
    conditioning keep two caches, and the second row of a step gets what
    it gets alone."""
    _serve(host, "kv", kvpipe)
    c = _client()
    rng = np.random.default_rng(15)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    ref = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    x0, x1 = noise_from_seed(5, F3_SHAPE), noise_from_seed(6, F3_SHAPE)
    row = {"cond": _rows(c, feats)[0]["cond"], "ref_latents": [c.put(ref)]}
    try:
        one, two = c.begin("kv"), c.begin("kv")
        alone = c.step("kv", one, x1, [0.75], [row], ref_cache=True)
        both = c.step("kv", two, np.concatenate([x0, x1]), [0.75],
                      [row, row], ref_cache=True)
        assert np.array_equal(both[1:], alone)
        assert len(steps.STATE.runs[two].ref_cache) == 2
        c.end(one)
        c.end(two)
    finally:
        kvpipe.captions.clear()
    assert host.held == 0


def test_a_kv_edit_step_under_the_node_is_priced_with_the_cache_it_keeps(
        host, kvpipe, monkeypatch):
    """The first step of a run under the Flux KV Cache node runs whole, the
    reference tokens joined to its rows, and keeps the K and V of its
    references. Its plan prices the step by the larger of the KV row's
    fits of that step and of a later one on those K and V, and the K and V
    it keeps, to the byte, and its refusal names what fits."""
    import re

    from gmlx.image import memfit
    from gmlx.serve import image_service

    _serve(host, "kv", kvpipe)
    served = host.served["kv"]
    c = _client()
    rng = np.random.default_rng(14)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    ref = rng.standard_normal((1, 8, 64, 64)).astype(np.float32)
    row = {"cond": _rows(c, feats)[0]["cond"], "ref_latents": [c.put(ref)]}
    x = np.zeros(F3_SHAPE, np.float32)
    w, h = routes._px(served, "latent", *F3_SHAPE[2:])
    p, scale = kvpipe.dit.cfg.patch, kvpipe.vae_scale
    terms = dict(caption_tokens=None, prefix_rows=[(64 // p) ** 2],
                 prefix_shared=False, prefix_other=0, prefix_bound=0,
                 prefix_held=0, ref_sizes=[],
                 step_refs=[(64 * scale, 64 * scale)], ref_cache="build")
    price = _least_budget(served, h, w, 1.0, terms)

    def plan(budget):
        args = image_service._fit_args(kvpipe, served, budget, 1.0)
        return memfit.plan_memory(height=h, width=w, held_in=("denoise",),
                                  **{**args, **terms})

    kv, act = memfit.PEAKS["flux2_klein_9b_kv"], kvpipe.dit.act_dtype.size
    image, refs = memfit.image_rows(kv, h, w, None), (64 // p) ** 2
    _, later, _ = memfit.prefix_row_terms(kv, image, refs, act)
    t = plan(price).terms
    assert t["prefix_cache"] == kv.prefix.cache_bytes(refs, act)
    assert t["step_peak"] == max(
        kv.ref_build.bytes((image + refs) * kv.width * act),
        kv.ref_read.bytes(later, 1, kv.prefix.join_bytes(refs, act)))
    with pytest.raises(memfit.ImageMemoryError) as refused:
        plan(price - 1)
    hint = refused.value.hint
    assert hint.startswith("The largest term is") and (
        "of the 1 reference images" in hint)
    run = c.begin("kv")
    step = lambda: c.step("kv", run, x, [0.75], [row],  # noqa: E731
                          ref_cache=True)
    try:
        host.budget_bytes = price - 1
        _refused_by_its_plan(monkeypatch, step, re.escape(hint))
        host.budget_bytes = price
        _reaches_the_dit(monkeypatch, step)
    finally:
        host.budget_bytes = None
        c.end(run)
    assert host.held == 0


def test_a_flux2_loop_on_the_reference_convention_gives_the_latents_of_sample(
        host, f2pipe):
    """The reference convention embeds the bf16 time of BFL's rule, as
    sample does. Comfy's fp32 time gives other latents. The noise is in
    bf16, as FLUX.2's loop rounds it, and the tiny DiT runs in float32,
    so the rest of the loop takes no rounding."""
    _serve(host, "f2d", f2pipe)
    feats = np.random.default_rng(7).standard_normal(
        (1, 10, 24)).astype(np.float32)
    noise = np.array(mx.array(noise_from_seed(3, F3_SHAPE)).astype(
        mx.bfloat16).astype(mx.float32))

    def direct():
        return np.array(api.sample(
            f2pipe, _f2_cond(feats, key="f2-direct"), height=64, width=96,
            noise=mx.array(noise), sigmas=SIGMAS))
    want = _on_worker(direct)
    got = {}
    for conv in ("reference", "comfy"):
        c = _client(convention=conv)
        got[conv] = c.sample("f2d", noise, SIGMAS, _rows(c, feats))
    assert np.isfinite(want).all()
    assert np.array_equal(got["reference"], want)
    assert not np.array_equal(got["comfy"], want)
    assert host.held == 0


def test_a_flux2_step_takes_its_references_in_the_order_of_the_row(
        host, f2pipe):
    _serve(host, "f2d", f2pipe)
    c = _client()
    rng = np.random.default_rng(11)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    a = rng.standard_normal((1, 8, 2, 3)).astype(np.float32)
    b = rng.standard_normal((1, 8, 3, 2)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    row = {"cond": _rows(c, feats)[0]["cond"],
           "ref_latents": [c.put(a), c.put(b)]}
    out = c.step("f2d", None, x, [0.75], [row])
    want = _f2_velocity(f2pipe, x, 0.75, _f2_cond(feats, a, b, key=""))
    assert np.array_equal(out, want)
    assert not np.array_equal(out, _f2_velocity(
        f2pipe, x, 0.75, _f2_cond(feats, b, a, key="")))
    assert host.held == 0


def test_a_flux2_table_on_comfyuis_native_keys_steps_as_the_core_table(
        host, f2pipe):
    """ComfyUI names a FLUX.2 pair by the GGUF name under diffusion_model.,
    a diffusers q, k or v pair as a band of the fused qkv, and the core
    table of those keys gives the same step."""
    from lora_fixtures import make_pair

    _serve(host, "f2d", f2pipe)
    c = _client()
    rng = np.random.default_rng(10)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    x = noise_from_seed(5, F3_SHAPE)
    plain = _f2_velocity(f2pipe, x, 0.75, _f2_cond(feats))
    for name, band, (dout, din) in (
            ("double_blocks.0.img_attn.qkv", [0, 64, 64], (64, 64)),
            ("double_blocks.1.txt_attn.qkv", None, (192, 64)),
            ("single_blocks.0.linear1", None, (576, 64)),
            ("txt_in", None, (64, 24))):
        down, up, _ = make_pair(rng, dout, din, 2)
        tid = c.define_table("f2d", {"entries": [{
            "key": f"diffusion_model.{name}.weight", "band": band,
            "patches": [{"up": c.put(up, kind="adapters"),
                         "down": c.put(down, kind="adapters"), "alpha": 2.0,
                         "strength": 0.8}]}]})
        core = steps.STATE.tables_of("f2d", f2pipe)[tid]
        out = c.step("f2d", None, x, [0.75], _rows(c, feats),
                     patch_table=tid)
        want = _f2_velocity(f2pipe, x, 0.75, _f2_cond(feats), core)
        assert np.array_equal(out, want), name
        assert not np.array_equal(out, plain), name


def test_the_flux2_status_names_comfyuis_flux2_on_model_space_latents(
        host, f2pipe, monkeypatch):
    monkeypatch.setattr(steps, "family_of", lambda spec: (
        get_family("flux2_klein"), "distilled"))
    monkeypatch.setattr(api, "_header_bytes", lambda path, part: 7)
    _serve(host, "f2d", f2pipe)
    st = _client().status("f2d")
    assert st["comfy"] == {"model_class": "Flux2", "image_model": "flux2"}
    assert st["edit_capable"]
    assert st["latent"] == {"channels": 128, "stride": 16, "patch": 1}
    assert st["vae"] == {"channels_out": 3, "stride": 16, "space": "model"}
    assert st["sampling"] == {"steps": 4, "cfg": 1.0, "shift": 1.0,
                              "sigmas": None, "schedule": {"rule": "flux2"}}


def test_the_flux2_vae_routes_carry_model_space_latents(host, f2pipe,
                                                         monkeypatch):
    """ComfyUI's FLUX.2 latents are the model's, as FLUX.3's are."""
    _model_space_vae_routes(host, f2pipe, "f2d", monkeypatch)


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

    # The fixture is in the core's layout; the wire takes ComfyUI's.
    rule = qt.rule
    feats = np.asarray(rule.features(case))
    refs = [np.asarray(z)[None] for z in rule.ref_latents(case)]
    text, before = steps.slots_to_wire(
        feats.shape[0], qt.CASES[case][1], [z.shape[-2:] for z in refs])
    feats = np.ascontiguousarray(feats[text])[None]
    return [{"cond": c.put(feats), "num_tokens": feats.shape[1],
             "ref_latents": [c.put(z) for z in refs],
             "image_slots": before}]


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
    # Every later step of the run is refused, and the run ends as usual.
    with pytest.raises(StepsClientError) as e:
        c.step("zit", run, x, [0.5], rows)
    assert e.value.status == 409 and e.value.body["code"] == "interrupted"
    assert c.interrupt(run) == 1 and c.interrupt("nope") == 0
    assert host.held == 1                     # the run's own hold
    assert c.end(run)["ended"] and host.held == 0


@pytest.mark.parametrize("lands", ["interrupt", "end"])
def test_a_stop_that_lands_during_the_eval_discards_the_step(host,
                                                             monkeypatch,
                                                             lands):
    c = _client()
    rows = _rows(c, _features())
    run = c.begin("zit")
    held = steps.STATE.runs[run]
    gen = held.gen
    real = api.denoise_step

    def denoise_step(*a, **kw):
        # The graph is built; the stop lands before its one evaluation.
        out = real(*a, **kw)
        if lands == "interrupt":
            steps.STATE.interrupt(run)
        else:
            steps.STATE.end(held)
        return out

    monkeypatch.setattr(api, "denoise_step", denoise_step)
    x = noise_from_seed(3, SHAPE)
    with pytest.raises(StepsClientError) as e:
        c.step("zit", run, x, [0.5], rows)
    assert e.value.status == 409 and e.value.body["code"] == "interrupted"
    assert held.steps == 0 and held.gen == gen + 1
    monkeypatch.setattr(api, "denoise_step", real)
    with pytest.raises(StepsClientError) as e:
        c.step("zit", run, x, [0.5], rows)
    if lands == "end":
        assert held.ended and run not in steps.STATE.runs
        assert e.value.status == 404 and host.held == 0
    else:
        assert e.value.status == 409 and host.held == 1


def test_a_step_gives_chat_its_share_of_the_gpu(host, monkeypatch):
    from gmlx.serve import gpu_share

    seen = []
    monkeypatch.setattr(gpu_share, "pacer", lambda: seen.append)
    c = _client()
    rows = _rows(c, _features())
    c.step("zit", None, np.zeros(SHAPE, np.float32), [0.5], rows)
    # The chat pacer is called at each point between the DiT blocks.
    assert seen and all(isinstance(a, tuple) for a in seen)


def test_an_interrupt_naming_no_run_reaches_none(host):
    c = _client()
    rows = _rows(c, _features())
    runs = [c.begin("zit"), c.begin("zit")]
    for named in (None, ""):
        r = _http().post(f"{P}/interrupt", json={"run": named}, headers=V)
        assert r.status_code == 200 and r.json() == {"interrupted": 0}
    for run in runs:
        assert c.step("zit", run, np.zeros(SHAPE, np.float32), [0.5],
                      rows).shape == SHAPE


def test_an_ended_run_sheds_its_prefix_states(host, qpipe):
    import qwen_tiny as qt

    c = _client()
    rows = _qwen_rows(c)
    run = c.begin("qit")

    def held():
        return [k for k in list(qpipe.captions._items)
                if isinstance(k, tuple) and str(k[0]).startswith("wire:")]

    c.step("qit", run, np.asarray(qt.latents()), [0.75], rows)
    assert held()
    c.end(run)
    _on_worker(lambda: None)                 # the shed queued before it
    assert held() == [] and host.held == 0


def test_an_idle_run_ends_on_the_timer(host, monkeypatch):
    import time

    c = _client()
    now = [100.0]
    monkeypatch.setattr(steps.STATE, "clock", lambda: now[0])
    monkeypatch.setattr(steps.STATE, "sweep_s", 0.01)
    run = c.begin("zit")
    assert host.held == 1
    now[0] += steps.RUN_IDLE_S + 1
    deadline = time.monotonic() + 5
    while run in steps.STATE.runs and time.monotonic() < deadline:
        time.sleep(0.01)
    assert run not in steps.STATE.runs and host.held == 0


def test_the_runs_held_at_once_are_bounded(host, monkeypatch):
    monkeypatch.setattr(steps, "MAX_RUNS", 2)
    c = _client()
    c.begin("zit")
    c.begin("zit")
    r = _http().post(f"{P}/runs/begin", json={"model": "zit"}, headers=V)
    assert r.status_code == 429 and "2 runs are held" in r.json()["error"]
    assert host.held == 2


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
    fam = {"zit": "z_image", "qit": "qwen_image21", "k2t": "krea2"}
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
    assert st["sampling"]["schedule"] is None
    assert st["memory"]["dit_bytes"] == 7
    assert st["edit_capable"]
    # Z-Image Turbo samples on its family schedule, which has no fixed
    # sigmas.
    z = c.status("zit")
    assert z["comfy"]["model_class"] == "ZImage"
    assert z["sampling"]["sigmas"] is None and z["sampling"]["steps"] == 8
    assert z["sampling"]["schedule"] is None
    assert not z["edit_capable"]
    # Krea 2 Turbo: ComfyUI's Krea2 class on 16 channel latents at 8x, in
    # 2x2 patches, an RGB VAE, 8 steps on its family schedule, no guidance.
    q = host.served["qit"]
    host.served["k2t"] = SimpleNamespace(**{**vars(q), "spec": SimpleNamespace(
        **{**vars(q.spec), "id": "k2t", "path": "/abs/k2t.gguf"})})
    k = c.status("k2t")
    assert k["comfy"] == {"model_class": "Krea2", "image_model": "krea2"}
    assert k["latent"] == {"channels": 16, "stride": 8, "patch": 2}
    assert k["vae"] == {"channels_out": 3, "stride": 8}
    assert k["sampling"]["sigmas"] is None and k["sampling"]["steps"] == 8
    assert k["sampling"]["cfg"] == 1.0 and not k["edit_capable"]


@pytest.mark.parametrize("kv, ne", [
    ({}, [12, 1]),
    ({"comfy.gguf.orig_shape.txtfusion.projector.weight": [1, 12]}, [12]),
])
def test_the_manifest_gives_the_krea2_projector_its_torch_shape(kv, ne):
    """ComfyUI detects Krea 2 by the projector and reads its second dim
    as the encoder layer count, from either converter's layout."""
    assert steps._torch_shape(kv, "txtfusion.projector.weight", ne) == \
        [1, 12]


@pytest.mark.parametrize("as_container", [False, True])
def test_encode_answers_the_tensors_and_holds_them(host, monkeypatch,
                                                   as_container):
    seen = []
    cond = _features()
    ref = np.ones((1, 4, 4, 6), np.float32)

    def encode(served, text, images, borrow=None, ref_latents=True):
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


def test_encode_with_refs_false_skips_the_reference_latents(host, monkeypatch):
    seen = []
    cond = _features()

    def encode(served, text, images, borrow=None, ref_latents=True):
        seen.append(ref_latents)
        refs = [np.ones((1, 4, 4, 6), np.float32)] if ref_latents else []
        return {"cond": cond, "refs": refs, "image_slots": [2]}

    monkeypatch.setattr(steps, "encode", encode)
    img = np.full((8, 8, 3), 9, np.uint8)
    body = wire.write({"image_0": img},
                      {"model": "qit", "text": "a cat", "refs": False})
    r = _http().post(f"{P}/encode", content=body,
                     headers={**V, "Content-Type": wire.CONTENT_TYPE})
    assert r.status_code == 200, r.text
    tensors, fields = wire.read(r.content)
    assert seen == [False]
    assert set(tensors) == {"cond"} and fields["ref_latents"] == []
    assert fields["image_slots"] == [2]
    r = _http().post(f"{P}/encode", headers=V,
                     json={"model": "qit", "text": "a", "refs": "no"})
    assert r.status_code == 400 and "refs" in r.text


def test_an_upsampled_encode_also_answers_the_prompt_as_typed(host, monkeypatch):
    """A client that adds the references after the encode, as ComfyUI
    does, gets the typed prompt's conditioning beside the caption's."""
    seen = []
    cond, typed = _features(), _features() + 1.0

    def encode(served, text, images, borrow=None, ref_latents=True, also=None):
        seen.append((text, also))
        return {"cond": cond, "refs": [], "image_slots": [], "typed": typed}

    monkeypatch.setattr(steps, "encode", encode)
    monkeypatch.setattr(host, "revise", lambda model, text, requested, has_refs:
                        "a caption of " + text, raising=False)
    fields, got = _client().encode("zit", "a cat")
    assert seen == [("a caption of a cat", "a cat")]
    assert fields["revised_prompt"] == "a caption of a cat"
    assert np.array_equal(got["cond_typed"], typed)
    assert fields["cond_typed"] == array_hash(typed) != fields["cond"]
    assert fields["cond_typed"] in steps.STATE.tensors


def _stub_step_encode(monkeypatch):
    """steps.encode on the Qwen-Image 2.1 family's reference rules with the
    text encoder and the VAE stubbed. Returns the served model, the
    references the encoder saw and the VAE encode calls."""
    from gmlx.serve import image_service

    seen, vae = [], []

    def stub_encode(served, text, neg, refs, borrow):
        # Two text rows, each reference's slot rows, three text rows: a
        # slot covers 32x32 pixels.
        seen.append(refs)
        n = sum((r.pixels.shape[0] // 32) * (r.pixels.shape[1] // 32)
                for r in refs)
        feats = mx.concatenate([mx.ones((2, 16)), mx.zeros((n, 16)),
                                mx.full((3, 16), 2.0)])
        return SimpleNamespace(features=feats, ref_latents=(),
                               image_slots=tuple(range(2, 2 + n))), None

    monkeypatch.setattr(image_service, "_encode", stub_encode)
    monkeypatch.setattr(api, "encode_references",
                        lambda pipe, c: vae.append(1) or c)
    fam = get_family("qwen_image21")
    served = SimpleNamespace(
        pipe=SimpleNamespace(family=fam, encoder=None),
        plan=SimpleNamespace(keep_encoder=True))
    return served, seen, vae


def test_step_encode_runs_no_vae_encode_without_reference_latents(
        monkeypatch):
    served, _, vae = _stub_step_encode(monkeypatch)
    img = np.zeros((256, 256, 3), np.uint8)
    got = _on_worker(lambda: steps.encode(served, "a", [img],
                                          ref_latents=False))
    assert vae == [] and got["refs"] == [] and got["image_slots"] == [2]
    assert np.array_equal(got["cond"][0, :, 0], [1, 1, 2, 2, 2])
    _on_worker(lambda: steps.encode(served, "a", [img]))
    assert vae == [1]


def test_step_encode_keeps_each_reference_at_its_own_size(monkeypatch):
    # A client that sizes its references gives its VAE the same pixels, so
    # the encoder's image slots match the latents it sends with the step.
    served, seen, _ = _stub_step_encode(monkeypatch)
    a = np.full((384, 512, 3), 7, np.uint8)
    b = np.full((512, 512, 3), 9, np.uint8)
    got = _on_worker(lambda: steps.encode(served, "a", [a, b]))
    assert [r.pixels.shape[:2] for r in seen[0]] == [(384, 512), (512, 512)]
    # The answer is ComfyUI's: text rows, and each reference's text row.
    assert got["cond"].shape == (1, 5, 16) and got["image_slots"] == [2, 2]
    assert np.all(seen[0][0].pixels[..., :3] == 7)
    big = np.zeros((1056, 1024, 3), np.uint8)
    small = np.zeros((128, 128, 3), np.uint8)
    for im in (big, small):
        with pytest.raises(steps.StepError) as e:
            _on_worker(lambda im=im: steps.encode(served, "a", [im]))
        assert e.value.status == 400


@pytest.mark.parametrize("fam, title", [("flux2_klein", "FLUX.2 klein"),
                                        ("flux3_klein", "FLUX.3 klein")])
def test_a_klein_step_encode_takes_the_prompt_and_no_images(monkeypatch, fam,
                                                            title):
    """The FLUX.2 and FLUX.3 klein encoders read the prompt alone, and
    their steps take the reference latents of the VAE route, so images on
    the encode are refused by that rule."""
    served, seen, _ = _stub_step_encode(monkeypatch)
    served.pipe.family = get_family(fam)
    img = np.zeros((256, 256, 3), np.uint8)
    with pytest.raises(steps.StepError, match=f"the {title} encoder "
                       "reads the prompt alone; a step takes the reference "
                       "latents the VAE route encodes") as e:
        _on_worker(lambda: steps.encode(served, "a", [img]))
    assert e.value.status == 400 and seen == []
    _on_worker(lambda: steps.encode(served, "a", []))
    assert seen == [()]


def test_wire_slots_are_comfyuis_and_map_onto_the_core_layout():
    xs = np.arange(5 * 2, dtype=np.float32).reshape(5, 2) + 1
    shapes = [(2, 4), (4, 4)]          # 2 and 4 slot rows

    def expand(before, shapes=shapes):
        got, slots = steps.slots_from_wire(mx.array(xs), before, shapes)
        return np.array(got), slots

    got, slots = _on_worker(lambda: expand([1, 3]))
    assert slots == (1, 2, 5, 6, 7, 8)
    assert got.shape == (11, 2) and not got[list(slots)].any()
    text = [i for i in range(11) if i not in slots]
    assert np.array_equal(got[text], xs)
    back, before = steps.slots_to_wire(11, slots, shapes)
    assert list(back) == text and before == [1, 3]
    # As in ComfyUI, a reference without a slot goes after the text.
    assert _on_worker(lambda: expand([1]))[1] == (1, 2, 7, 8, 9, 10)
    for bad, sh in (([3, 1], shapes), ([0, 6], shapes), ([0], [(3, 3)])):
        with pytest.raises(steps.StepError) as e:
            _on_worker(lambda bad=bad, sh=sh: expand(bad, sh))
        assert e.value.status == 400


def test_cache_sizes_follow_server_image(host):
    routes.mount(_APP.app, dataclasses.replace(
        ImageServerCfg(), tensor_cache_mb=3, adapter_cache_mb=5), host)
    assert steps.STATE.tensors.max_bytes == 3 << 20
    assert steps.STATE.adapters.max_bytes == 5 << 20
    limits = _http().get(f"{P}/capabilities").json()["limits"]
    assert limits["max_tensor_bytes"] == 3 << 20
    assert limits["max_adapter_bytes"] == 5 << 20


def test_the_default_step_caches_are_the_configured_ones():
    cfg = ImageServerCfg()
    fresh = steps.StepsState()
    assert fresh.tensors.max_bytes == cfg.tensor_cache_mb << 20
    assert fresh.adapters.max_bytes == cfg.adapter_cache_mb << 20


def test_a_tensor_over_the_cache_is_refused_before_it_is_copied(
        host, monkeypatch):
    routes.mount(_APP.app, dataclasses.replace(
        ImageServerCfg(), tensor_cache_mb=1), host)
    copied = []
    monkeypatch.setattr(steps, "_wire_copy", lambda t: copied.append(t) or t)
    big = np.zeros((1, 300_000, 1), np.float32)
    r = _http().put(f"{P}/tensors/{array_hash(big)}",
                    content=wire.write({"cond": big}),
                    headers={**V, "Content-Type": wire.CONTENT_TYPE})
    assert r.status_code == 413 and copied == []


# Size limits and the memory plan.

def _put_container(path, tensors, fields=None):
    return _http().post(f"{P}/{path}", content=wire.write(tensors, fields),
                        headers={**V, "Content-Type": wire.CONTENT_TYPE})


@pytest.mark.parametrize("path, tensors, fields", [
    # 80x4800 pixels at Z-Image's stride of 8: a side over 4096.
    ("step", {"x": np.zeros((1, 4, 600, 10), np.float32),
              "timestep": np.zeros((1,), np.float32)},
     {"model": "zit", "rows": [{"cond": "0" * 64}]}),
    ("vae/decode", {"latents": np.zeros((1, 4, 600, 10), np.float32)},
     {"model": "zit"}),
    ("vae/decode", {"latents": np.zeros((17, 4, 2, 2), np.float32)},
     {"model": "zit"}),
    ("vae/encode", {"image": np.zeros((1, 4800, 8, 3), np.float32)},
     {"model": "zit"}),
])
def test_a_step_or_vae_pass_over_the_size_limits_is_413(host, path, tensors,
                                                         fields):
    r = _put_container(path, tensors, fields)
    assert r.status_code == 413, r.text
    assert host.held == 0


@pytest.mark.parametrize("path, tensors, fields", [
    ("step", {"x": np.zeros(SHAPE, np.float32),
              "timestep": np.zeros((1,), np.float32)},
     {"model": "zit", "rows": [{"cond": "0" * 64}]}),
    ("vae/decode", {"latents": np.zeros((1, 4, 4, 6), np.float32)},
     {"model": "zit"}),
    ("vae/encode", {"image": np.zeros((1, 32, 48, 3), np.float32)},
     {"model": "zit"}),
])
def test_a_step_or_vae_pass_that_does_not_fit_is_400(host, path, tensors,
                                                     fields):
    host.budget_bytes = 1
    try:
        r = _put_container(path, tensors, fields)
    finally:
        host.budget_bytes = None
    assert r.status_code == 400, r.text
    assert "does not fit in memory" in r.json()["error"]
    assert host.held == 0


def test_a_vae_encode_is_planned_with_its_image_and_a_decode_without(
        host, monkeypatch):
    from gmlx.serve import image_service

    seen = []
    monkeypatch.setattr(image_service, "size_fits",
                        lambda served, w, h, budget, cfg=1.0, init=False,
                        held_in=None: seen.append((init, held_in)) or False)
    for path, tensors in (
            ("vae/encode", {"image": np.zeros((1, 32, 48, 3), np.float32)}),
            ("vae/decode", {"latents": np.zeros((1, 4, 4, 6), np.float32)})):
        r = _put_container(path, tensors, {"model": "zit"})
        assert r.status_code == 400, r.text
    # Each counts the held states in the phase it runs alone.
    assert seen == [(True, ("init",)), (False, ("decode",))]


def _bare_and_full(served, h, w, **full):
    """The step plan peaks of ``served`` at h x w pixels without its rows'
    conditioning, as the size check before the rows load prices it, and
    with ``full``, the terms of its references."""
    from gmlx.image import memfit
    from gmlx.serve import image_service

    args = image_service._fit_args(served.pipe, served, 1 << 50, 1.0)
    bare = memfit.plan_memory(height=h, width=w, _hint=False, **args).peak
    return bare, memfit.plan_memory(height=h, width=w, _hint=False,
                                    **args, **full).peak


def _refused_by_its_plan(monkeypatch, call, hint: str = "") -> None:
    """``call`` is refused by the plan of its rows before the DiT runs,
    with ``hint`` after the refusal. A step the plan lets through stops at
    the DiT, whose forward on these references would take gigabytes."""
    def reached(*a, **k):
        raise AssertionError("the step reached the DiT")

    with monkeypatch.context() as m:
        m.setattr(steps, "run_step", reached)
        with pytest.raises(StepsClientError, match="with the references of "
                           "its rows does not fit in memory beside the "
                           f"models now loaded. {hint}"):
            call()


def test_a_flux3_step_prices_the_reference_latents_of_its_rows(
        host, fpipes, monkeypatch):
    """Four reference latents of 1024 pixels join every step's rows, so a
    step that fits as text to image is refused with them, and more than
    the family's 10 references in a row are refused by count."""
    from gmlx.image import memfit

    _serve(host, "f3d", fpipes[True])
    served = host.served["f3d"]
    c = _client()
    rng = np.random.default_rng(9)
    feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
    x = np.zeros(F3_SHAPE, np.float32)
    refs = [c.put(rng.standard_normal((1, 8, 64, 64)).astype(np.float32))
            for _ in range(4)]
    rows = _rows(c, feats)
    with_refs = [{**rows[0], "ref_latents": refs}]
    caption = 10 if memfit.prices_prompt("flux3_klein") else None
    w, h = routes._px(served, "latent", *F3_SHAPE[2:])
    bare, full = _bare_and_full(served, h, w, caption_tokens=caption,
                                step_refs=[(1024, 1024)] * 4)
    assert full > bare
    host.budget_bytes = (bare + full) / 2
    try:
        assert c.step("f3d", None, x, [0.75], rows,
                      sample_sigmas=SIGMAS).shape == F3_SHAPE
        _refused_by_its_plan(monkeypatch, lambda: c.step(
            "f3d", None, x, [0.75], with_refs, sample_sigmas=SIGMAS))
    finally:
        host.budget_bytes = None
    small = c.put(np.zeros((1, 8, 2, 3), np.float32))
    with pytest.raises(StepsClientError, match="FLUX.3 klein takes at most "
                       "10 reference latents in a row, and row 0 has 11"):
        c.step("f3d", None, x, [0.75], [{**rows[0],
                                         "ref_latents": [small] * 11}],
               sample_sigmas=SIGMAS)
    assert host.held == 0


def test_each_row_of_a_step_takes_ten_references_and_no_more(
        host, fpipes, monkeypatch):
    """The family's count of references is allowed in every row, and one
    more in any row is refused with that row's index."""
    _serve(host, "f3d", fpipes[True])
    c = _client()
    feats = np.random.default_rng(13).standard_normal((1, 10, 24)).astype(
        np.float32)
    x = np.zeros((2, *F3_SHAPE[1:]), np.float32)
    small = c.put(np.zeros((1, 8, 2, 3), np.float32))
    row = _rows(c, feats)[0]
    ten = [{**row, "ref_latents": [small] * 10} for _ in range(2)]
    _reaches_the_dit(monkeypatch, lambda: c.step(
        "f3d", None, x, [0.75], ten, sample_sigmas=SIGMAS))
    for i in range(2):
        rows = [dict(r) for r in ten]
        rows[i]["ref_latents"] = [small] * 11
        with pytest.raises(StepsClientError, match="FLUX.3 klein takes at "
                           f"most 10 reference latents in a row, and row {i} "
                           "has 11"):
            c.step("f3d", None, x, [0.75], rows, sample_sigmas=SIGMAS)
    assert host.held == 0


def test_a_qwen_step_prices_the_reference_prefix_of_its_rows(host,
                                                            monkeypatch):
    """Two reference latents of 1024 pixels in a Qwen-Image 2.1 row put
    their tokens in the prefix of every block, which the plan of the step
    counts."""
    from gmlx.image import memfit

    served = host.served["qit"]
    c = _client()
    rng = np.random.default_rng(10)
    text = 9
    feats = rng.standard_normal((1, text, 24)).astype(np.float32)
    refs = [c.put(rng.standard_normal((1, 8, 64, 64)).astype(np.float32))
            for _ in range(2)]
    row = {"cond": c.put(feats), "num_tokens": text, "ref_latents": refs,
           "image_slots": [2, 5]}
    x = np.zeros((1, 8, 4, 6), np.float32)
    caption = text if memfit.prices_prompt("qwen_image21") else None
    w, h = routes._px(served, "latent", *x.shape[2:])
    bare, full = _bare_and_full(served, h, w, caption_tokens=caption,
                                prefix_rows=[text + 2 * 64 * 64])
    assert full > bare
    host.budget_bytes = (bare + full) / 2
    try:
        plain = {"cond": row["cond"], "num_tokens": text}
        assert c.step("qit", None, x, [0.75], [plain]).shape == x.shape
        _refused_by_its_plan(monkeypatch,
                             lambda: c.step("qit", None, x, [0.75], [row]))
    finally:
        host.budget_bytes = None
    assert host.held == 0


def _least_budget(served, h, w, cfg, terms) -> int:
    """The least budget at which the plan of a step of ``served`` with
    ``terms`` fits, its levers applied, as the step route plans one."""
    from gmlx.image import memfit
    from gmlx.serve import image_service

    def fits(budget):
        args = image_service._fit_args(served.pipe, served, budget, cfg)
        args.update(terms)
        try:
            return memfit.plan_memory(height=h, width=w, _hint=False,
                                      **args).fits
        except memfit.ImageMemoryError:
            return False

    lo, hi = 0, 1 << 50
    assert fits(hi) and not fits(lo)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def _reaches_the_dit(monkeypatch, call) -> None:
    """``call`` passes the plan of its rows and reaches the DiT, which a
    stub stands in for, as these references would take gigabytes."""
    seen = []

    def stub(served, x, *a, **k):
        seen.append(tuple(x.shape))
        return mx.zeros(x.shape, mx.float32), 0.0, "none"

    with monkeypatch.context() as m:
        m.setattr(steps, "run_step", stub)
        call()
    assert seen


def _priced_to_the_byte(host, monkeypatch, served, h, w, terms, step):
    """``step`` runs on the least budget of the plan of ``terms`` with
    guidance, and is refused by the plan of its rows one byte below it."""
    price = _least_budget(served, h, w, 2.0, terms)
    # The rows' plan, not the size check before the rows load, decides.
    assert _least_budget(served, h, w, 2.0, {}) < price - 1
    try:
        host.budget_bytes = price - 1
        _refused_by_its_plan(monkeypatch, step)
        host.budget_bytes = price
        _reaches_the_dit(monkeypatch, step)
    finally:
        host.budget_bytes = None
    assert host.held == 0
    return price


def test_a_guided_flux3_step_is_priced_at_its_rows_to_the_byte(
        host, fpipes, monkeypatch):
    """Two rows whose second holds the most reference pixels: one small
    reference in the first row and four of 1024 pixels in the second.
    Pricing the references of the first row, or the family's most text
    rows, misses the budget of the plan by a byte or more."""
    _serve(host, "f3d", fpipes[True])
    served = host.served["f3d"]
    pipe = served.pipe
    c = _client()
    rng = np.random.default_rng(11)
    rows = []
    for shapes in ([(16, 16)], [(64, 64)] * 4):
        feats = rng.standard_normal((1, 10, 24)).astype(np.float32)
        rows.append({"cond": c.put(feats), "num_tokens": 10, "ref_latents": [
            c.put(rng.standard_normal((1, 8) + hw).astype(np.float32))
            for hw in shapes]})
    x = np.zeros((2,) + F3_SHAPE[1:], np.float32)
    w, h = routes._px(served, "latent", *F3_SHAPE[2:])
    p, scale = pipe.dit.cfg.patch, pipe.vae_scale
    terms = dict(caption_tokens=10,
                 prefix_rows=[10 + (16 // p) ** 2, 10 + 4 * (64 // p) ** 2],
                 prefix_shared=False, prefix_other=0, prefix_bound=0,
                 ref_sizes=[], step_refs=[(64 * scale, 64 * scale)] * 4)
    conds = _on_worker(lambda: [steps.conditioning(
        r, steps.STATE.tensors, "flux3_klein") for r in rows])
    assert api.plan_terms(pipe, conds, height=h, width=w) == terms
    price = _priced_to_the_byte(host, monkeypatch, served, h, w, terms,
                                lambda: c.step("f3d", None, x, [0.75], rows,
                                               sample_sigmas=SIGMAS))
    assert _least_budget(served, h, w, 2.0, {
        **terms, "step_refs": [(16 * scale, 16 * scale)]}) < price
    assert _least_budget(served, h, w, 2.0, {
        **terms, "caption_tokens": None}) > price


def test_a_guided_qwen_step_is_priced_at_both_prefix_states_to_the_byte(
        host, monkeypatch):
    """A guided pair of two conditionings, each with two reference latents
    of 1024 pixels, holds two prefix states. Pricing one, as an unguided
    step would, misses the budget of the plan by a byte or more."""
    served = host.served["qit"]
    pipe = served.pipe
    c = _client()
    rng = np.random.default_rng(12)
    refs = [c.put(rng.standard_normal((1, 8, 64, 64)).astype(np.float32))
            for _ in range(2)]
    rows = [{"cond": c.put(rng.standard_normal((1, 9, 24)).astype(
        np.float32)), "num_tokens": 9, "ref_latents": refs,
        "image_slots": [2, 5]} for _ in range(2)]
    x = np.zeros((2, 8, 4, 6), np.float32)
    w, h = routes._px(served, "latent", *x.shape[2:])
    p, scale = pipe.dit.cfg.patch, pipe.vae_scale
    # The wire cond holds the text rows alone: the route sets the two
    # image slots into them.
    terms = dict(caption_tokens=9,
                 prefix_rows=[9 + 2 * (64 // p) ** 2] * 2,
                 prefix_shared=False, prefix_other=pipe.captions.bytes,
                 prefix_bound=pipe.family.step.state_bytes,
                 prefix_held=pipe.captions.bytes, ref_sizes=[],
                 step_refs=[(64 * scale, 64 * scale)] * 2)
    conds = _on_worker(lambda: [steps.conditioning(
        r, steps.STATE.tensors, "qwen_image21") for r in rows])
    assert api.plan_terms(pipe, conds, height=h, width=w) == terms
    price = _priced_to_the_byte(host, monkeypatch, served, h, w, terms,
                                lambda: c.step("qit", None, x, [0.75], rows))
    assert _least_budget(served, h, w, 1.0, terms) < price


def test_the_step_route_plans_its_rows_under_its_table_and_guidance(
        host, monkeypatch):
    """The plan of a step's rows takes the step's patch table, which
    names the prefix states its run holds, and the guidance of its two
    rows."""
    from lora_fixtures import make_pair

    from gmlx.image import memfit
    from gmlx.serve import image_service

    c = _client()
    rows = _qwen_rows(c)
    down, up, _ = make_pair(np.random.default_rng(4), 32, 32, 2)
    tid = c.define_table("qit", {"entries": [{
        "key": "diffusion_model.transformer_blocks.0.attn.to_q.weight",
        "band": None,
        "patches": [{"up": c.put(up, kind="adapters"),
                     "down": c.put(down, kind="adapters"), "alpha": 2.0,
                     "strength": 0.8}]}]})
    core = steps.STATE.tables_of("qit", host.served["qit"].pipe)[tid]
    seen = []

    def size_fits(served, w, h, budget, *, cfg=1.0, init=False, conds=None,
                  patch_table=None, joined=False, ref_cache=None, building=0,
                  held_in=None, hint=False):
        seen.append(dict(cfg=cfg, conds=conds, patch_table=patch_table))
        if conds is not None:
            raise memfit.ImageMemoryError("refused", None, "a hint")
        return True

    monkeypatch.setattr(image_service, "size_fits", size_fits)
    import qwen_tiny as qt

    x = np.asarray(qt.latents())
    pair = [rows[0], rows[0]]
    _refused_by_its_plan(monkeypatch, lambda: c.step(
        "qit", None, np.concatenate([x, x]), [0.75], pair, patch_table=tid))
    bare, planned = seen
    assert bare["cfg"] == planned["cfg"] == 2.0 and bare["conds"] is None
    assert len(planned["conds"]) == 2 and planned["patch_table"] is core


def test_the_plan_of_a_qwen_run_counts_its_held_prefix_once(host, qpipe):
    """After a step of a guided pair of one conditioning under a table,
    the plan of the next step holds one prefix state for the pair, and
    counts the state its run holds under that table as its own, not as
    another conditioning's."""
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
    core = steps.STATE.tables_of("qit", qpipe)[tid]
    x = np.asarray(qt.latents())
    pair = [rows[0], rows[0]]
    run = c.begin("qit")
    c.step("qit", run, np.concatenate([x, x]), [0.75], pair, patch_table=tid)
    conds = _on_worker(lambda: [steps.conditioning(
        r, steps.STATE.tensors, "qwen_image21") for r in pair])
    w, h = routes._px(host.served["qit"], "latent", *x.shape[2:])
    held = qpipe.captions.bytes
    assert held > 0
    terms = api.plan_terms(qpipe, conds, height=h, width=w,
                           patch_table=core)
    assert terms["prefix_shared"] is True and terms["prefix_other"] == 0
    # Under no table the state is another conditioning's.
    assert api.plan_terms(qpipe, conds, height=h, width=w)[
        "prefix_other"] == held
    (other,) = _qwen_rows(c, "ref1")
    two = _on_worker(lambda: [steps.conditioning(
        r, steps.STATE.tensors, "qwen_image21") for r in (rows[0], other)])
    assert api.plan_terms(qpipe, two, height=h, width=w,
                          patch_table=core)["prefix_shared"] is False
    c.end(run)
    assert host.held == 0


def test_a_patch_table_body_that_is_no_object_is_400(host):
    body = b"[1]"
    r = _http().put(f"{P}/patch-tables/{steps.table_id(body)}?model=zit",
                    content=body, headers={**V,
                                           "Content-Type": "application/json"})
    assert r.status_code == 400 and "JSON object" in r.json()["error"]
