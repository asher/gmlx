"""The sampling loop of gmlx.image.api on a toy DiT: Euler from seeded
noise, guidance, img2img and inpainting, the per-step hooks, per-row
denoise steps, and the cache bound around the loop. CPU."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api, memfit
from gmlx.image.registry import get_family
from gmlx.image.samplers import img2img_start, noise_from_seed
from gmlx.image.schedule import zimage_sigmas


class _Dit:
    """v = latents * (1 + sigma) + mean of the caption features, so the
    velocity depends on every argument."""

    def __init__(self):
        self.cfg = SimpleNamespace(in_channels=4, patch=2)
        self.act_dtype = mx.bfloat16
        self.calls = 0
        self.bounded = []

    def __call__(self, latents, sigma, cap_feats):
        self.calls += 1
        self.bounded.append(_bound[0])
        s = mx.array(np.asarray(sigma, np.float32)).reshape(-1, 1, 1, 1)
        return (latents * (1 + s) + cap_feats.mean()).astype(mx.float32)


_bound = [False]


@pytest.fixture
def pipe(monkeypatch):
    @contextmanager
    def bound():
        _bound[0] = True
        try:
            yield
        finally:
            _bound[0] = False

    monkeypatch.setattr(memfit, "bounded_cache", bound)
    return api.ImagePipeline(dit=_Dit(), family=get_family("z_image"),
                             variant="turbo")


def _cond(value: float) -> api.Conditioning:
    return api.Conditioning(mx.full((3, 5), value, mx.float32))


def _euler(x, sigmas, velocity):
    for i in range(len(sigmas) - 1):
        dt = sigmas[i + 1] - sigmas[i]
        x = x + mx.array(dt, mx.float32) * velocity(x, sigmas[i])
    return x


def test_sample_is_euler_from_the_seeded_noise_under_the_cache_bound(pipe):
    x = api.sample(pipe, _cond(0.5), height=32, width=48, seed=7)
    s = zimage_sigmas(8, 3.0)
    noise = mx.array(noise_from_seed(7, (1, 4, 4, 6)))
    want = _euler(noise, s, lambda x, sig: x * (1 + mx.array(sig)) + 0.5)
    assert x.dtype == mx.float32 and x.shape == (1, 4, 4, 6)
    assert np.array_equal(np.array(x), np.array(want))
    assert pipe.dit.calls == 8 and all(pipe.dit.bounded)


def test_guidance_runs_a_negative_row_and_guides(pipe):
    with pytest.raises(ValueError, match="needs a negative conditioning"):
        api.sample(pipe, _cond(1.0), height=32, width=32, seed=1, cfg=3.0)
    x = api.sample(pipe, _cond(1.0), neg=_cond(0.25), cfg=3.0, steps=2,
                   height=32, width=32, seed=1)
    s = zimage_sigmas(2, 3.0)

    def v(x, sig):
        base = x * (1 + mx.array(sig))
        return api.guide(base + 1.0, base + 0.25, 3.0)

    want = _euler(mx.array(noise_from_seed(1, (1, 4, 4, 4))), s, v)
    assert np.array_equal(np.array(x), np.array(want))
    assert pipe.dit.calls == 4
    g = api.guide(mx.array([2.0]), mx.array([1.0]), 3.0)
    assert g.item() == 4.0


def test_img2img_starts_at_the_strength_index(pipe):
    assert [img2img_start(8, d) for d in (1.0, 0.75, 0.5, 0.1)] == \
        [0, 2, 4, 7]
    init = mx.array(np.linspace(-1, 1, 64, dtype=np.float32)
                    .reshape(1, 4, 4, 4))
    noise = mx.array(noise_from_seed(3, (1, 4, 4, 4)))
    x = api.sample(pipe, _cond(0.0), height=32, width=32, noise=noise,
                   init_latents=init, denoise=0.5)
    s = zimage_sigmas(8, 3.0)[4:]
    s0 = mx.array(s[0])
    start = s0 * noise + (1 - s0) * init
    want = _euler(start, s, lambda x, sig: x * (1 + mx.array(sig)))
    assert np.allclose(np.array(x), np.array(want), rtol=0, atol=1e-6)
    assert pipe.dit.calls == 4
    with pytest.raises(ValueError, match="need init_latents"):
        api.sample(pipe, _cond(0.0), height=32, width=32, seed=0,
                   denoise=0.5)


def test_inpainting_keeps_the_unmasked_latents(pipe):
    init = mx.array(np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4))
    mask = np.zeros((32, 32), np.float32)
    mask[:, 16:] = 1.0                       # the right half changes
    x = api.sample(pipe, _cond(0.0), height=32, width=32, seed=2,
                   init_latents=init, mask=mask)
    plain = api.sample(pipe, _cond(0.0), height=32, width=32, seed=2,
                       init_latents=init)
    x, plain = np.array(x), np.array(plain)
    assert np.array_equal(x[..., :2], np.array(init)[..., :2])
    assert np.array_equal(x[..., 2:], plain[..., 2:])


def test_hooks_see_every_step_and_a_stop_returns_the_latents(pipe):
    seen = []
    api.sample(pipe, _cond(0.0), height=32, width=32, seed=0, steps=3,
               on_step=lambda i, x: seen.append((i, x.shape)))
    assert seen == [(0, (1, 4, 4, 4)), (1, (1, 4, 4, 4)), (2, (1, 4, 4, 4))]
    calls = iter([False, False, True])
    with pytest.raises(api.SampleStopped) as e:
        api.sample(pipe, _cond(0.0), height=32, width=32, seed=0, steps=3,
                   should_stop=lambda: next(calls))
    assert e.value.step == 2 and e.value.latents.shape == (1, 4, 4, 4)


def test_denoise_step_rows_equal_single_rows(pipe):
    x = mx.array(noise_from_seed(4, (2, 4, 4, 4)))
    v = api.denoise_step(pipe, x, [0.75, 0.25], [_cond(1.0), _cond(2.0)])
    a = api.denoise_step(pipe, x[:1], 0.75, _cond(1.0))
    b = api.denoise_step(pipe, x[1:], 0.25, _cond(2.0))
    assert np.array_equal(np.array(v), np.concatenate([a, b]))
    with pytest.raises(ValueError, match="3 sigmas for 2 rows"):
        api.denoise_step(pipe, x, [1, 2, 3], _cond(1.0))
    with pytest.raises(ValueError, match="unknown patch table"):
        api.denoise_step(pipe, x, 0.5, _cond(1.0), patch_table="ab")


def test_sample_refuses_bad_requests(pipe):
    with pytest.raises(ValueError, match="multiples of 16"):
        api.sample(pipe, _cond(0.0), height=40, width=32, seed=0)
    with pytest.raises(ValueError, match="seed or noise"):
        api.sample(pipe, _cond(0.0), height=32, width=32, seed=0,
                   noise=noise_from_seed(0, (1, 4, 4, 4)))
    with pytest.raises(ValueError, match="disagrees"):
        api.sample(pipe, _cond(0.0), height=32, width=32, seed=0, steps=2,
                   sigmas=[1.0, 0.5, 0.25, 0.0])
    with pytest.raises(ValueError, match="no sampler defaults"):
        api.ImagePipeline(dit=pipe.dit, family=pipe.family,
                          variant="other").defaults


def _tiny_pipe():
    from image_fixtures import tiny_z_model

    m, _ = tiny_z_model()
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


def test_a_split_step_equals_the_plain_forward():
    pipe = _tiny_pipe()
    m = pipe.dit
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 6, 8)).astype(np.float32))
    cap = mx.array(rng.standard_normal((1, 5, 16)).astype(np.float32))
    sig = np.float32(0.75)
    plain = np.array(m(lat, sig, cap))
    split = m(lat, sig, None, mod=m.modulation(np.reshape(sig, (1,)), 1),
              caption=m.caption(cap))
    assert np.array_equal(plain, np.array(split))

    cond = api.Conditioning(cap[0], "k1")
    for _ in range(2):
        v = api.denoise_step(pipe, lat, sig, cond)
        assert np.array_equal(plain, np.array(v))
    assert len(pipe.modulations) == 1 and len(pipe.captions) == 1
    api.denoise_step(pipe, lat, 0.5, cond)
    api.denoise_step(pipe, lat, sig, api.Conditioning(cap[0]))
    assert len(pipe.modulations) == 2 and len(pipe.captions) == 1


def test_byte_lru_evicts_the_least_recent_entries():
    from gmlx.image.conditioning import ByteLRU

    c = ByteLRU(10)
    c.put("a", 1, 4)
    c.put("b", 2, 4)
    assert c.get("a") == 1
    c.put("c", 3, 4)
    assert c.get("b") is None and c.get("a") == 1 and c.bytes == 8
    c.put("big", 4, 11)
    assert c.get("big") is None and len(c) == 2


class _Encoder:
    """An encoder handle over fixed features per character."""

    def __init__(self):
        from gmlx.image.text_encoder import Z_IMAGE

        self.spec = Z_IMAGE
        self.source_key = "enc"
        self.dtype_name = "bfloat16"
        self.model = object()
        self.encoded = []

    def token_ids(self, prompt):
        return [ord(c) for c in prompt]

    def encode(self, prompt):
        if self.model is None:
            raise RuntimeError("the qwen3-4b text encoder was released")
        self.encoded.append(prompt)
        h = mx.array(np.array(self.token_ids(prompt), np.float32)[:, None]
                     * np.ones((1, 4), np.float32))
        return SimpleNamespace(hidden=(h[None],))


def test_encode_text_caches_by_key_and_survives_release(pipe):
    pipe.encoder = enc = _Encoder()
    a = api.encode_text(pipe, "ab")
    assert a.features.shape == (2, 4) and len(a.key) == 64
    assert api.encode_text(pipe, "ab") is a and enc.encoded == ["ab"]
    enc.model = None                        # released
    assert api.encode_text(pipe, "ab") is a
    with pytest.raises(RuntimeError, match="released"):
        api.encode_text(pipe, "cd")
    with pytest.raises(ValueError, match="no reference images"):
        api.encode_text(pipe, "ab", images=[np.zeros((8, 8, 3))])


def test_the_conditioning_key_covers_its_inputs():
    from gmlx.image.conditioning import conditioning_key

    kw = dict(family="z_image", encoder="e", template="t", ids=[1, 2],
              tap_layers=(35,), dtype="bfloat16")
    base = conditioning_key(**kw)
    assert base == conditioning_key(**kw)
    for change in (dict(family="q"), dict(encoder="f"), dict(template="u"),
                   dict(ids=[1, 3]), dict(tap_layers=(34,)),
                   dict(dtype="float16"), dict(adapters="x"),
                   dict(images=[("h", 8, 8)])):
        assert conditioning_key(**{**kw, **change}) != base, change


def test_load_gates_the_header_sizes_before_any_load(monkeypatch):
    calls = []
    sizes = {"dit": 5, "encoder": 3, "vae": 1}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda p, kind: 0 if p is None else sizes[kind])

    def refuse(weight_bytes, model_id, **kw):
        calls.append(weight_bytes)
        raise memfit.ImageMemoryError("no room")

    monkeypatch.setattr(memfit, "gate_load", refuse)
    from gmlx.image import loader

    monkeypatch.setattr(loader, "load_dit", lambda *a, **k: 1 / 0)
    with pytest.raises(memfit.ImageMemoryError):
        api.load("d.gguf", text_encoder="e.gguf", vae="v.gguf")
    assert calls == [9]
    with pytest.raises(ValueError, match="not both"):
        api.load("d.gguf", text_encoder="e", encoder_model=(1, 2))
    with pytest.raises(ValueError, match="memory must be"):
        api.load("d.gguf", memory="maybe")


def test_the_request_plan_releases_an_owned_encoder_and_tiles(pipe):
    G = memfit.GiB
    pipe.weights = {"dit": int(4.67 * G), "encoder": int(2.32 * G),
                    "vae": 167_000_000}
    pipe.encoder = SimpleNamespace(released=False, owned=True)
    budget = 17_179_869_184 - 8e9
    p = api.plan_request(pipe, height=1024, width=1024, budget=budget,
                         overcommit=False)
    assert p.fits and p.release_encoder and p.tiled_decode
    pipe.encoder = SimpleNamespace(released=True, owned=True)
    p = api.plan_request(pipe, height=1024, width=1024, budget=budget,
                         overcommit=False)
    assert p.terms["encoder"] == 0 and not p.release_encoder
    pipe.encoder = SimpleNamespace(released=False, owned=False)
    with pytest.raises(memfit.ImageMemoryError, match="borrowed encoder"):
        api.plan_request(pipe, height=1024, width=1024, budget=7 * G,
                         overcommit=False)
