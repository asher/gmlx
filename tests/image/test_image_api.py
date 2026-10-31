"""The sampling loop of gmlx.image.api on a toy DiT: Euler and UniPC from
seeded noise, guidance, img2img and inpainting, the per-step hooks, per-row
denoise steps, and the cache bound around the loop. CPU."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api, memfit
from gmlx.image.registry import get_family
from gmlx.image.samplers import img2img_start, make_sampler, noise_from_seed
from gmlx.image.schedule import zimage_sigmas
from gmlx.image.text_encoder import TextTap


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


def test_sample_runs_unipc_with_the_model_at_the_schedule_sigmas(pipe):
    x = api.sample(pipe, _cond(0.5), height=32, width=48, seed=7,
                   sampler="uni_pc_bh2")
    s = zimage_sigmas(8, 3.0)
    smp = make_sampler("uni_pc_bh2", s)
    want = mx.array(noise_from_seed(7, (1, 4, 4, 6)))
    for i in range(8):
        v = want * (1 + mx.array(s[i])) + 0.5
        want = smp.step(want, v, i)
    assert np.array_equal(np.array(x), np.array(want))
    with pytest.raises(ValueError, match="unknown sampler 'dpm'"):
        api.sample(pipe, _cond(0.5), height=32, width=48, seed=7,
                   sampler="dpm")


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


def test_an_unknown_variant_needs_steps_and_cfg(pipe):
    unknown = api.ImagePipeline(dit=pipe.dit, family=pipe.family,
                                variant="unknown")
    for kw in (dict(), dict(steps=2), dict(cfg=1.0)):
        with pytest.raises(ValueError, match="pass steps and cfg"):
            api.sample(unknown, _cond(0.0), height=32, width=32, seed=0,
                       **kw)
    x = api.sample(unknown, _cond(0.0), height=32, width=32, seed=0,
                   steps=2, cfg=1.0)
    assert x.shape == (1, 4, 4, 4) and pipe.dit.calls == 2


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


def _qkv_dims(pipe):
    return api._linear_dims(dict(pipe.dit.named_modules())[
        "layers.0.attention.qkv"])


def _spec(pipe, out: int):
    """Entries on a q band and on w1, and the hashes of their pairs."""
    from gmlx.image import lora

    rng = np.random.default_rng(0)
    pair = lambda o, i: (  # noqa: E731
        rng.standard_normal((2, i)).astype(np.float32),
        rng.standard_normal((o, 2)).astype(np.float32))
    _, inp = _qkv_dims(pipe)
    w1 = "layers.0.feed_forward.w1"
    a = lora.register_pair(pipe, *pair(out // 3, inp))
    b = lora.register_pair(pipe, *pair(
        *api._linear_dims(dict(pipe.dit.named_modules())[w1])))
    return [{"module": "layers.0.attention.qkv", "band": [0, 0, out // 3],
             "patches": [[a, 0.8]]}, (w1, None, [(b, 1.0)])], a, b


def test_a_patch_table_is_named_by_its_content_and_kept_frozen():
    pipe = _tiny_pipe()
    out, _ = _qkv_dims(pipe)
    assert pipe.default_table == api.EMPTY_TABLE
    assert pipe.patch_tables[api.EMPTY_TABLE]["entries"] == ()
    assert api.define_patch_table(pipe, []) == api.EMPTY_TABLE
    spec, a, _ = _spec(pipe, out)
    tid = api.define_patch_table(pipe, spec)
    assert len(tid) == 64 and tid != api.EMPTY_TABLE
    assert api.define_patch_table(pipe, _spec(pipe, out)[0][::-1]) == tid
    table = pipe.patch_tables[tid]
    spec[0]["patches"][0][1] = 0.5
    spec[0]["band"][1] = out // 3
    assert pipe.patch_tables[tid] is table
    assert table["entries"][0] == ("layers.0.attention.qkv",
                                   (0, 0, out // 3), ((a, 0.8),))
    with pytest.raises(TypeError):
        table["entries"] = ()
    other = api.define_patch_table(pipe, spec)
    assert other not in (tid, api.EMPTY_TABLE)
    assert pipe.patch_tables[tid]["entries"][0][2] == ((a, 0.8),)


def test_rows_that_give_the_same_delta_share_a_table_id():
    from gmlx.image import lora

    pipe = _tiny_pipe()
    out, inp = _qkv_dims(pipe)
    qkv, third = "layers.0.attention.qkv", out // 3
    rng = np.random.default_rng(0)
    pair = lambda o, r: lora.register_pair(  # noqa: E731
        pipe, rng.standard_normal((r, inp)).astype(np.float32),
        rng.standard_normal((o, r)).astype(np.float32))
    a, b, c = pair(third, 2), pair(third, 3), pair(out, 2)
    band = (0, third, third)
    tid = api.define_patch_table(pipe, [(qkv, band, [(a, 0.5), (b, -0.2)])])
    # The patches of a row sum, so their order does not name the table.
    assert api.define_patch_table(
        pipe, [(qkv, band, [(b, -0.2), (a, 0.5)])]) == tid
    assert pipe.patch_tables[tid]["entries"][0][2] == tuple(
        sorted(((a, 0.5), (b, -0.2))))
    # A band of numpy integers is the band of plain ones.
    assert api.define_patch_table(
        pipe, [(qkv, tuple(np.int64(v) for v in band),
                [(a, 0.5), (b, -0.2)])]) == tid
    assert api.define_patch_table(pipe, [(qkv, np.array(band, np.int32),
                                          [(a, 0.5), (b, -0.2)])]) == tid
    assert type(pipe.patch_tables[tid]["entries"][0][1][2]) is int
    with pytest.raises(ValueError, match="three integers"):
        api.define_patch_table(pipe, [(qkv, (0, np.float32(third), third),
                                       [(a, 0.5)])])
    with pytest.raises(ValueError, match="three integers"):
        api.define_patch_table(pipe, [(qkv, (np.bool_(0), 0, third),
                                       [(a, 0.5)])])
    # A strength of -0.0 is 0.0.
    zero = api.define_patch_table(pipe, [(qkv, band, [(a, 0.0), (b, 1.0)])])
    assert api.define_patch_table(
        pipe, [(qkv, band, [(a, -0.0), (b, 1.0)])]) == zero
    # A band over the whole axis is the whole weight, on either axis, and
    # joins a whole-weight row on the same module.
    whole = api.define_patch_table(pipe, [(qkv, None, [(c, 0.7)])])
    assert api.define_patch_table(pipe, [(qkv, (0, 0, out), [(c, 0.7)])]) \
        == whole
    assert api.define_patch_table(pipe, [(qkv, (1, 0, inp), [(c, 0.7)])]) \
        == whole
    assert pipe.patch_tables[whole]["entries"] == ((qkv, None, ((c, 0.7),)),)
    both = api.define_patch_table(pipe, [(qkv, None, [(c, 0.7), (c, 0.1)])])
    assert api.define_patch_table(pipe, [(qkv, (0, 0, out), [(c, 0.1)]),
                                         (qkv, None, [(c, 0.7)])]) == both
    # Different deltas keep different ids.
    assert len({tid, zero, whole, both}) == 4
    assert api.define_patch_table(pipe, [(qkv, band, [(a, 0.5)])]) != tid
    assert api.define_patch_table(
        pipe, [(qkv, (0, 0, third), [(a, 0.5), (b, -0.2)])]) != tid


def test_a_bad_patch_table_entry_is_refused():
    pipe = _tiny_pipe()
    out, inp = _qkv_dims(pipe)
    qkv = "layers.0.attention.qkv"
    for entry, match in (
            ((("layers.99.attention.qkv", None, [("a", 1.0)])),
             "not a module of this DiT"),
            (("layers.0.attention", None, [("a", 1.0)]), "not a linear"),
            ((qkv, (2, 0, 1), [("a", 1.0)]), "does not fit"),
            ((qkv, (0, out - 1, 2), [("a", 1.0)]), "does not fit"),
            ((qkv, (1, 0, inp + 1), [("a", 1.0)]), "does not fit"),
            ((qkv, (0, 0), [("a", 1.0)]), "three integers"),
            ((qkv, None, []), "no patches"),
            ((qkv, None, [("", 1.0)]), "non-empty string"),
            ((qkv, None, [("a", float("nan"))]), "strength"),
            ({"module": qkv, "patches": [("a", 1)], "scale": 2},
             "unknown keys")):
        with pytest.raises(ValueError, match=match):
            api.define_patch_table(pipe, [entry])
    with pytest.raises(ValueError, match="twice"):
        api.define_patch_table(pipe, [(qkv, None, [("a", 1.0)]),
                                      (qkv, None, [("b", 1.0)])])
    assert list(pipe.patch_tables) == [api.EMPTY_TABLE]


def test_none_and_the_default_table_share_memo_entries():
    pipe = _tiny_pipe()
    rng = np.random.default_rng(0)
    lat = mx.array(rng.standard_normal((1, 4, 6, 8)).astype(np.float32))
    cond = api.Conditioning(
        mx.array(rng.standard_normal((5, 16)).astype(np.float32)), "k1")
    a = api.denoise_step(pipe, lat, 0.75, cond)
    b = api.denoise_step(pipe, lat, 0.75, cond,
                         patch_table=pipe.default_table)
    assert np.array_equal(np.array(a), np.array(b))
    assert len(pipe.modulations) == 1 and len(pipe.captions) == 1
    keys = list(pipe.modulations._items) + list(pipe.captions._items)
    assert all(k[1] == api.EMPTY_TABLE for k in keys)
    out, _ = _qkv_dims(pipe)
    tid = api.define_patch_table(pipe, _spec(pipe, out)[0])
    api.denoise_step(pipe, lat, 0.75, cond, patch_table=tid)
    assert len(pipe.modulations) == 2 and len(pipe.captions) == 2
    with pytest.raises(ValueError, match="unknown patch table"):
        api.denoise_step(pipe, lat, 0.75, cond, patch_table="0" * 64)
    with pytest.raises(ValueError, match="unknown patch table"):
        api.sample(pipe, cond, height=32, width=32, seed=0, steps=1,
                   cfg=1.0, patch_table=tid[::-1])


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


def test_byte_lru_never_sheds_a_kept_entry():
    from gmlx.image.conditioning import ByteLRU

    c = ByteLRU(10)
    c.put("own", 1, 4)
    c.keep({"own"})
    c.put("a", 2, 4)
    c.put("b", 3, 4)
    assert c.peek("own") == 1 and c.peek("a") is None and c.bytes == 8
    # An entry that does not fit beside the kept one sheds nothing.
    c.put("c", 4, 7)
    assert c.peek("c") is None and c.peek("b") == 3 and c.bytes == 8


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
        return TextTap(ids=tuple(self.token_ids(prompt)),
                       hidden=(h[None],), template_id="", tap_layers=())


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


def test_a_released_encoder_tells_a_cached_prompt_without_encoding(pipe):
    assert not api.conditioning_cached(pipe, "ab")      # no encoder yet
    pipe.encoder = enc = _Encoder()
    api.encode_text(pipe, "ab")
    enc.model = None                        # released
    assert api.conditioning_cached(pipe, "ab")
    assert not api.conditioning_cached(pipe, "cd")
    enc.adapters = "other"                  # the text adapters are keyed
    assert not api.conditioning_cached(pipe, "ab")
    assert enc.encoded == ["ab"]


class _ReleasedEncoder(_Encoder):
    owned = True

    @property
    def released(self):
        return self.model is None

    def release(self):
        self.model = None
        return True


def test_a_staged_load_holds_no_dit_beside_the_encoder(pipe, monkeypatch):
    """``load_staged`` on stub loaders: the encode stage sees the encoder
    and no DiT or VAE, the encoder is gone when the DiT loads, and the
    pipeline samples what a pipeline loaded whole samples."""
    from gmlx.image import loader, text_encoder
    from gmlx.image import vae as vae_pkg

    fam = get_family("z_image")
    ident = SimpleNamespace(family=fam, variant="turbo", config={},
                            view=SimpleNamespace(tensors={}))
    enc, seen, staged = _ReleasedEncoder(), [], []

    def state():
        p = staged[0]
        return (p.dit is not None, p.vae is not None,
                not p.encoder.released)

    def load_dit(path, **kw):
        seen.append(("dit", state()))
        assert kw["identified"] is ident
        return _Dit(), SimpleNamespace(identified=ident)

    monkeypatch.setattr(loader, "identify_dit",
                        lambda path, variant=None: ident)
    monkeypatch.setattr(loader, "load_dit", load_dit)
    monkeypatch.setattr(
        text_encoder, "load_encoder",
        lambda path, spec, log=print, mmproj=None:
        seen.append(("encoder", path, mmproj)) or enc)
    monkeypatch.setattr(vae_pkg, "load_vae", lambda path, family, dtype:
                        seen.append(("vae", state())) or "the vae")
    monkeypatch.setattr(api, "_header_bytes",
                        lambda path, kind: 0 if path is None else 7)

    def encode(p):
        staged.append(p)
        seen.append(("encode", state()))
        assert p.dit is None and p.vae is None and p.identified is ident
        return api.encode_text(p, "ab")

    got, cond = api.load_staged("d", text_encoder="e", vae="v",
                                encode=encode)
    # (the DiT is loaded, the VAE is loaded, the encoder is resident)
    assert seen == [("encoder", "e", None),
                    ("encode", (False, False, True)),
                    ("dit", (False, False, False)),
                    ("vae", (True, False, False))]
    assert got is staged[0] and got.encoder.released
    assert isinstance(got.dit, _Dit) and got.vae == "the vae"
    assert got.weights == {"dit": 7, "encoder": 7, "vae": 7}
    assert sorted(got.load_seconds) == ["dit", "encoder", "vae"]
    # The conditioning of the encode stage serves the loaded pipeline.
    assert api.encode_text(got, "ab") is cond
    x = api.sample(got, cond, height=32, width=48, seed=7)
    pipe.encoder = _Encoder()
    want = api.sample(pipe, api.encode_text(pipe, "ab"), height=32,
                      width=48, seed=7)
    assert np.array_equal(np.array(x), np.array(want))
    # An mmproj goes to the encoder's load, and the pipeline counts the
    # tower as loaded.
    enc2 = _ReleasedEncoder()
    enc2.vision_bytes = lambda: 11
    monkeypatch.setattr(
        text_encoder, "load_encoder",
        lambda path, spec, log=print, mmproj=None:
        seen.append(("encoder", path, mmproj)) or enc2)
    got, _ = api.load_staged("d", text_encoder="e", mmproj="m",
                             encode=lambda p: None)
    assert ("encoder", "e", "m") in seen and got.weights["mmproj"] == 11
    assert got.vae is None and got.encoder.released
    # An encode that raises releases the encoder and loads no DiT.
    enc3 = _ReleasedEncoder()
    monkeypatch.setattr(text_encoder, "load_encoder",
                        lambda path, spec, log=print, mmproj=None: enc3)
    n = len(seen)

    def fail(p):
        raise ValueError("bad prompt")

    with pytest.raises(ValueError, match="bad prompt"):
        api.load_staged("d", text_encoder="e", encode=fail)
    assert enc3.released and len(seen) == n


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
    # The loader binds memfit.gate_load by name when it is first imported,
    # so it is imported before the gate is stubbed: a stub bound there
    # would outlive this test and refuse every later load.
    from gmlx.image import loader

    calls = []
    sizes = {"dit": 5, "encoder": 3, "vae": 1}
    monkeypatch.setattr(api, "_header_bytes",
                        lambda p, kind: 0 if p is None else sizes[kind])

    def refuse(weight_bytes, model_id, **kw):
        calls.append(weight_bytes)
        raise memfit.ImageMemoryError("no room")

    monkeypatch.setattr(memfit, "gate_load", refuse)
    monkeypatch.setattr(loader, "load_dit", lambda *a, **k: 1 / 0)
    with pytest.raises(memfit.ImageMemoryError):
        api.load("d.gguf", text_encoder="e.gguf", vae="v.gguf")
    assert calls == [9]
    with pytest.raises(ValueError, match="not both"):
        api.load("d.gguf", text_encoder="e", encoder_model=(1, 2))
    with pytest.raises(ValueError, match="memory must be"):
        api.load("d.gguf", memory="maybe")


def test_the_load_gate_stub_does_not_outlive_its_test():
    """After the test above, in a process where it was the first to
    import the loader: the loader gates with the real gate."""
    from gmlx.image import loader

    assert loader.gate_load is memfit.gate_load
    assert memfit.gate_load.__module__ == "gmlx.image.memfit"


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
    with pytest.raises(memfit.ImageMemoryError, match="is not released in this request"):
        api.plan_request(pipe, height=1024, width=1024, budget=7 * G,
                         overcommit=False)


def _small(pipe):
    """A small pipeline for the plan: weights in bytes, an owned encoder
    with an mmproj, and a VAE of the family's row that no pass runs, at
    the 8 pixels per latent pixel of the family's VAE."""
    G = memfit.GiB
    pipe.weights = {"dit": G, "encoder": G, "mmproj": G // 2,
                    "vae": 100 << 20}
    pipe.encoder = SimpleNamespace(released=False, owned=True)
    pipe.vae = SimpleNamespace(
        family="z_image", dtype=mx.bfloat16, conv="exact",
        config=SimpleNamespace(out_channels=3, scale=8),
        channels=lambda op: 128)
    return dict(height=1024, width=1024, cfg=3.0, overcommit=False)


def test_each_lever_has_a_budget_that_forces_it(pipe):
    """The levers in the plan's order on a small model, and under the
    last of them the refusal, which names every term in GiB."""
    kw = _small(pipe)
    G = memfit.GiB
    rows, release, tiles = memfit.LEVERS[:3]
    free = api.plan_request(pipe, budget=100 * G, **kw)
    assert free.levers == () and free.rows == 2
    assert free.terms["mmproj"] == G // 2 and free.peak == free.phases[
        "decode"]
    # One byte under the whole decode: the rows go one at a time, which
    # frees nothing of the decode, and then the encoder goes with its
    # mmproj.
    released = api.plan_request(pipe, budget=free.peak - 1, **kw)
    assert released.levers == (rows, release) and released.rows == 1
    assert released.release_encoder and not released.tiled_decode
    assert released.phases["decode"] == free.peak - G - G // 2
    assert released.phases["encode"] == free.phases["encode"]
    # One byte under that: the decode runs in tiles.
    tiled = api.plan_request(pipe, budget=released.peak - 1, **kw)
    assert tiled.levers == (rows, release, tiles) and tiled.tiled_decode
    assert tiled.peak < released.peak
    # An encoder the caller keeps is no lever, and tiles come in its place.
    kept = api.plan_request(pipe, budget=free.peak - 1, keep_encoder=True,
                            **kw)
    assert kept.levers == (rows, tiles) and not kept.release_encoder
    # One byte under the last plan nothing is left, and the refusal names
    # every term in GiB.
    with pytest.raises(memfit.ImageMemoryError) as e:
        api.plan_request(pipe, budget=tiled.peak - 1, **kw)
    msg, plan = str(e.value), e.value.plan
    assert plan.levers == (rows, release, tiles) and not plan.fits
    for name, value in plan.terms.items():
        assert (f"{name} {value / G:.2f} GiB" in msg) == bool(value), name
    for name in ("dit", "encoder", "mmproj", "vae", "step_peak",
                 "decode_peak", "encode_peak"):
        assert plan.terms[name] > 0, name
    assert f"budget of {(tiled.peak - 1) / G:.2f} GiB" in msg
    assert "The largest term is decode_peak; try a lower resolution" in msg


def test_sample_runs_one_row_at_a_time_where_the_plan_says_so(pipe,
                                                              monkeypatch):
    kw = _small(pipe)
    both = api.plan_request(pipe, budget=100 * memfit.GiB, **kw)
    one = api.plan_request(pipe, budget=both.peak - 1, **kw)
    assert (both.rows, one.rows) == (2, 1)
    real, seen = mx.eval, []

    def spy(*a, **k):
        seen.append(pipe.dit.calls)
        return real(*a, **k)

    def run(plan):
        pipe.dit.calls = 0
        del seen[:]
        x = api.sample(pipe, _cond(1.0), neg=_cond(-1.0), cfg=3.0, height=64,
                       width=64, steps=2, seed=3, plan=plan)
        return np.array(x), list(seen)

    monkeypatch.setattr(mx, "eval", spy)
    together, at = run(both)
    # One evaluation per step, after both forwards.
    assert at == [2, 4] and run(None)[1] == [2, 4]
    apart, at = run(one)
    # The positive velocity is evaluated before the negative forward.
    assert at == [1, 2, 3, 4]
    assert np.array_equal(apart, together)
    # A request without guidance has one row either way.
    pipe.dit.calls = 0
    del seen[:]
    api.sample(pipe, _cond(1.0), cfg=1.0, height=64, width=64, steps=2,
               seed=3, plan=one)
    assert seen == [1, 2]


def test_the_vae_passes_run_as_the_plan_says(pipe, monkeypatch):
    from gmlx.image.vae import tiling

    kw = _small(pipe)
    G = memfit.GiB
    calls = []
    monkeypatch.setattr(tiling, "decode", lambda vae, x, **k: calls.append(
        ("decode", k)) or mx.zeros((1, 8, 8, 3)))
    monkeypatch.setattr(tiling, "encode", lambda vae, x, **k: calls.append(
        ("encode", k)) or mx.zeros((1, 4, 1, 1)))
    lat, img = mx.zeros((1, 4, 1, 1)), np.zeros((1, 8, 8, 3), np.float32)
    whole = api.plan_request(pipe, budget=100 * G, init_image=True, **kw)
    assert not whole.tiled_decode and not whole.tiled_encode
    under = whole.phases["decode"] - G - G // 2
    tiled = api.plan_request(pipe, budget=under - 1, init_image=True, **kw)
    assert tiled.tiled_decode and tiled.release_encoder
    for plan in (whole, tiled):
        del calls[:]
        api.vae_decode(pipe, lat, plan=plan)
        api.vae_encode(pipe, img, plan=plan)
        held = plan.phases["decode"] - plan.terms["decode_peak"]
        assert held == plan.phases["init"] - plan.terms["init_peak"]
        assert [(op, k["tiled"], k["free"]) for op, k in calls] == [
            ("decode", plan.tiled_decode, plan.budget - held),
            ("encode", plan.tiled_encode, plan.budget - held)]
    # The room of the plan is what its pass is priced against, so the
    # pass the plan admits is the one that runs.
    assert tiled.budget - held >= tiled.terms["decode_peak"]
    # A caller can still force a pass, within the room of the plan, and
    # without a plan the guarded pass chooses against what is free.
    del calls[:]
    api.vae_decode(pipe, lat, plan=whole, tiled=True)
    api.vae_decode(pipe, lat)
    api.vae_encode(pipe, img, tiled=False)
    assert [(k["tiled"], k["free"]) for _, k in calls] == [
        (True, whole.vae_pass("decode")[1]), (None, None), (False, None)]
    # A plan that had no budget decided nothing, so its pass is chosen
    # against what is free, as for a caller without a plan.
    from dataclasses import replace

    del calls[:]
    blind = replace(whole, budget=None)
    assert blind.vae_pass("decode") == (False, None)
    api.vae_decode(pipe, lat, plan=blind)
    api.vae_encode(pipe, img, plan=blind, tiled=True)
    assert [(k["tiled"], k["free"]) for _, k in calls] == [
        (None, None), (True, None)]
    # A plan without an init image has no encode to run.
    plain = api.plan_request(pipe, budget=100 * G, **kw)
    with pytest.raises(ValueError, match="this plan has no VAE encode"):
        api.vae_encode(pipe, img, plan=plain)


def test_the_plan_prices_the_vae_the_pipeline_loaded(pipe):
    kw = _small(pipe)
    G = memfit.GiB
    exact = api.plan_request(pipe, budget=100 * G, init_image=True, **kw)
    pipe.vae.conv = "winograd"
    wino = api.plan_request(pipe, budget=100 * G, init_image=True, **kw)
    for op, term in (("decode", "decode_peak"), ("encode", "init_peak")):
        assert exact.terms[term] == memfit.vae_peak(
            op, 1024, 1024, family="z_image", itemsize=2, conv="exact")
        assert wino.terms[term] == memfit.vae_peak(
            op, 1024, 1024, family="z_image", itemsize=2, conv="winograd")
        assert wino.terms[term] > exact.terms[term]
    # A released encoder holds nothing, and neither does its mmproj.
    pipe.encoder = SimpleNamespace(released=True, owned=True)
    gone = api.plan_request(pipe, budget=100 * G, **kw)
    assert gone.terms["encoder"] == gone.terms["mmproj"] == 0


def test_the_vae_scale_comes_from_the_family_row(pipe):
    from dataclasses import replace

    from gmlx.commands.image import _stride
    from gmlx.image.registry import ConstantsRow

    fam = pipe.family
    assert pipe.vae_scale == fam.constants.values["vae_scale_factor"] == 8
    assert _stride(fam) == 16
    row = fam.constants
    wide = replace(fam, constants=ConstantsRow(
        {**row.values, "vae_scale_factor": 16}, row.repo, row.revision))
    p = api.ImagePipeline(dit=pipe.dit, family=wide, variant="turbo")
    assert p.vae_scale == 16 and _stride(wide) == 32
    with pytest.raises(ValueError, match="multiples of 32"):
        api.sample(p, _cond(0.0), height=48, width=32, seed=0)
    x = api.sample(p, _cond(0.0), height=64, width=32, seed=0, steps=1)
    assert x.shape == (1, 4, 4, 2)


def test_load_picks_the_activation_from_the_header_and_loads_the_dit_once(
        monkeypatch):
    from gmlx.image import dtypes, loader

    fam = get_family("z_image")
    monkeypatch.setattr(api, "_header_bytes", lambda p, kind: 0)
    monkeypatch.setattr(memfit, "gate_load", lambda *a, **k: None)
    monkeypatch.setattr(
        loader, "identify_dit",
        lambda path, **kw: SimpleNamespace(family=fam, variant="turbo"))
    monkeypatch.setattr(
        dtypes, "image_activation",
        lambda name, **kw: dtypes.Activation(mx.float16, mx.float32, 14))
    calls = []

    def load_dit(path, **kw):
        calls.append(kw.get("act_dtype"))
        ident = SimpleNamespace(family=fam, variant="turbo")
        return _Dit(), SimpleNamespace(identified=ident,
                                       act_dtype=kw.get("act_dtype"))

    monkeypatch.setattr(loader, "load_dit", load_dit)
    p = api.load("d.gguf")
    assert calls == [mx.float16]
    assert p.family is fam and p.variant == "turbo"


def test_variant_defaults_follow_their_official_sampling_rows(pipe):
    fam = pipe.family
    turbo, base = fam.sampling["turbo"], fam.sampling["base"]
    assert (turbo.steps, turbo.cfg, turbo.shift) == (8, 1.0, 3.0)
    assert (base.steps, base.cfg, base.shift) == (50, 5.0, 6.0)
    assert turbo.repo == "Tongyi-MAI/Z-Image-Turbo"
    assert base.repo == "Tongyi-MAI/Z-Image"
    assert all(len(r.revision) == 40 for r in (turbo, base))
    d = api.sampler_defaults("z_image", "base")
    assert (d.steps, d.cfg, d.shift) == (50, 5.0, 6.0)
    b = api.ImagePipeline(dit=pipe.dit, family=fam, variant="base")
    assert np.array_equal(b.sigmas(4), zimage_sigmas(4, 6.0))
    assert np.array_equal(pipe.sigmas(4), zimage_sigmas(4, 3.0))
    unknown = api.ImagePipeline(dit=pipe.dit, family=fam, variant="other")
    assert np.array_equal(unknown.sigmas(4), zimage_sigmas(
        4, fam.constants.values["scheduler_shift"]))


def test_sample_takes_the_batch_from_the_noise(pipe):
    noise = noise_from_seed(9, (2, 4, 4, 4))
    x = api.sample(pipe, _cond(0.5), height=32, width=32, noise=noise,
                   steps=2)
    assert x.shape == (2, 4, 4, 4)
    for i in range(2):
        one = api.sample(pipe, _cond(0.5), height=32, width=32,
                         noise=noise[i:i + 1], steps=2)
        assert np.array_equal(np.array(x[i:i + 1]), np.array(one))
    with pytest.raises(ValueError, match="does not match"):
        api.sample(pipe, _cond(0.5), height=32, width=64, noise=noise,
                   steps=2)


def test_load_passes_the_borrowed_models_gguf_to_its_encoder(monkeypatch):
    """A borrowed encoder model with its GGUF reads the F32 norms a
    MOSS-Image encode runs on, as the CLI and serve do."""
    from gmlx.image import loader

    fam = get_family("moss_image")
    ident = SimpleNamespace(family=fam, variant=None)
    monkeypatch.setattr(api, "_header_bytes", lambda p, kind: 0)
    monkeypatch.setattr(loader, "identify_dit", lambda *a, **k: ident)
    monkeypatch.setattr(loader, "load_dit", lambda *a, **k: (
        None, SimpleNamespace(identified=ident)))
    got = []

    def borrowed(pair, spec, gguf=None):
        got.append((pair, spec.fp32_norms, gguf))
        raise LookupError("stop")

    monkeypatch.setattr(api, "_borrowed", borrowed)
    with pytest.raises(LookupError):
        api.load("d.gguf", encoder_model=(1, 2), encoder_gguf="chat.gguf")
    assert got == [((1, 2), True, "chat.gguf")]
    with pytest.raises(ValueError, match="encoder_gguf"):
        api.load("d.gguf", text_encoder="e.gguf", encoder_gguf="chat.gguf")
