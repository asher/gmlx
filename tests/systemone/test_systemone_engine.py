"""Structured reads on a tiny DiffusionGemma: the read views, the one-step
and multi-step reads, prefix extension, the thought, the tokenizer adapter
and the engine scope."""

from __future__ import annotations

import math
import types

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from mlx_vlm.models.cache import RotatingKVCache  # noqa: E402
from mlx_vlm.models.diffusion_gemma.language import _cache_state  # noqa: E402
from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer  # noqa: E402
from mlx_vlm.utils import StoppingCriteria  # noqa: E402
from test_diffusion_gemma import ARCH, _tiny_meta  # noqa: E402

import gmlx.systemone.denoise as denoise  # noqa: E402
import gmlx.systemone.engine as engine  # noqa: E402
from gmlx.load.config_synth import synthesize_config  # noqa: E402
from gmlx.load.loader import build_model  # noqa: E402
from gmlx.systemone.engine import (  # noqa: E402
    ChatTokens,
    StructuredReader,
    engine_scope,
)
from gmlx.systemone.reads import (  # noqa: E402
    Cancelled,
    ReadRequest,
    Slot,
    build_canvas,
    label_id_union,
    pinned_positions,
)

WIDTH = 16
TEMPLATE = (20, 21, 22, 23, 24)
SLOTS = (Slot(1, (30, 31)), Slot(3, (40, 41, 42)))
PROMPT = [2] + list(range(5, 25))          # 21 ids, past the window of 8
EOS = 1
VOCAB = 128


@pytest.fixture(scope="module", autouse=True)
def _cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def _build(full: bool):
    meta = _tiny_meta()
    # The seed canvas always carries the turn-close id (106), so the vocabulary
    # must reach past it.
    meta["tokenizer.ggml.tokens"] = ["t"] * VOCAB
    shapes = {}
    if full:
        meta[f"{ARCH}.attention.sliding_window_pattern"] = [True, False, True]
        shapes = {"blk.1.attn_k.weight": (32, 16)}
    mx.random.seed(0)
    model, _ = build_model(synthesize_config(meta, shapes))
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def sliding_model(_cpu):
    return _build(False)


@pytest.fixture(scope="module")
def full_model(_cpu):
    return _build(True)


@pytest.fixture(params=["sliding", "full"])
def model(request, sliding_model, full_model):
    return sliding_model if request.param == "sliding" else full_model


def _reader(model, step=4):
    return StructuredReader(model, prefill_step_size=step)


def _req(seeds, **kw):
    kw.setdefault("rows_per_pass", WIDTH * len(seeds))
    return ReadRequest(template=TEMPLATE, slots=kw.pop("slots", SLOTS),
                       width=WIDTH, seeds=tuple(seeds), **kw)


def _assert_reads_close(a, b, atol=1e-5):
    assert len(a.samples) == len(b.samples)
    for sa, sb in zip(a.samples, b.samples):
        assert sa.seed == sb.seed and sa.canvas_in == sb.canvas_in
        for x, y in zip(sa.slots, sb.slots):
            assert x.argmax_id == y.argmax_id
            assert x.top.keys() == y.top.keys()
            for t in x.top:
                assert math.isclose(x.top[t], y.top[t], abs_tol=atol), (t, x.top[t], y.top[t])


def test_read_views_leave_the_decoder_no_masks(model):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    views = prompt.views_for(3)
    h = mx.zeros((3, WIDTH, model.config.text_config.hidden_size))
    masks = model.model.decoder._make_decoder_masks(h, views, None)
    assert set(masks) == set(model.config.text_config.layer_types)
    assert all(m is None for m in masks.values())


def test_read_views_match_the_stock_cache_path(model):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    canvas = mx.array([build_canvas(TEMPLATE, SLOTS, WIDTH, 7, reader.vocab)],
                      dtype=mx.int32)
    stock = reader.decoder(canvas, cache=prompt.cache, decoder_attention_mask=None)
    viewed = reader.decoder(canvas, cache=prompt.views_for(1),
                            decoder_attention_mask=None)
    assert mx.allclose(stock, viewed, atol=1e-5).item()


def test_slot_reads_follow_positions_not_list_order(sliding_model, monkeypatch):
    real = engine.build_canvas

    def by_position(template, slots, width, seed, vocab):
        return real(template, sorted(slots, key=lambda s: s.pos), width, seed, vocab)

    monkeypatch.setattr(engine, "build_canvas", by_position)
    reader = _reader(sliding_model)
    prompt = reader.prefill(PROMPT)
    fwd = reader.read(prompt, _req((5, 6)))
    rev = reader.read(prompt, _req((5, 6), slots=SLOTS[::-1]))
    for a, b in zip(fwd.samples, rev.samples):
        assert a.canvas_in == b.canvas_in
        for x, y in zip(a.slots, b.slots[::-1]):
            assert x.argmax_id == y.argmax_id
            assert x.top == pytest.approx(y.top, abs=1e-6)


def test_constrained_read_renormalises_the_unconstrained_one(model):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    allowed = label_id_union(SLOTS)
    con = reader.read(prompt, _req((1, 2, 3), constrained=True))
    unc = reader.read(prompt, _req((1, 2, 3), constrained=False))
    worst = 0.0
    for sc, su in zip(con.samples, unc.samples):
        for c, u in zip(sc.slots, su.slots):
            assert set(c.top) == set(allowed)
            assert c.argmax_id in allowed
            assert set(allowed) <= set(u.top) and u.argmax_id in u.top
            sub = {t: u.top[t] for t in allowed}
            assert c.argmax_id == max(sub, key=sub.__getitem__)
            norm = math.log(sum(math.exp(v) for v in sub.values()))
            for t in allowed:
                worst = max(worst, abs(math.exp(sub[t] - norm) - math.exp(c.top[t])))
    # Measured at most 1.8e-7 over 32 seeds on both fp32 fixtures; the two
    # paths share the pre-softcap GEMM.
    assert worst < 1e-6


def test_constrained_top_sums_to_one(model):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    for steps in (1, 3):
        res = reader.read(prompt, _req((1, 2), steps=steps, constrained=True))
        for s in res.samples:
            for slot in s.slots:
                assert sum(math.exp(v) for v in slot.top.values()) == pytest.approx(1.0, abs=1e-5)


def test_multistep_constrained_read_samples_only_allowed_ids(sliding_model):
    reader = _reader(sliding_model)
    reader.config.entropy_bound = 1e9          # accept every position
    prompt = reader.prefill(PROMPT)
    allowed = set(label_id_union(SLOTS))
    res = reader.read(prompt, _req((1, 2), steps=3, trace=True))
    pins = pinned_positions(SLOTS, WIDTH, 3)
    for s in res.samples:
        *canvases, final = s.trace
        assert len(canvases) == 3 and canvases[0] == s.canvas_in
        for c in canvases[1:]:
            assert all(c[slot.pos] in allowed for slot in SLOTS)
            assert all(c[p] == s.canvas_in[p] for p in pins)
        assert set(final) <= allowed


def test_pins_hold_and_soft_embedding_is_zero_at_pins(sliding_model):
    reader = _reader(sliding_model)
    prompt = reader.prefill(PROMPT)
    real = reader.decoder
    seen = []

    class Recording:
        def __call__(self, canvas, *, cache, self_conditioning_embeddings=None,
                     decoder_attention_mask=None):
            seen.append(self_conditioning_embeddings)
            return real(canvas, cache=cache,
                        self_conditioning_embeddings=self_conditioning_embeddings,
                        decoder_attention_mask=decoder_attention_mask)

        def __getattr__(self, name):
            return getattr(real, name)

    reader.decoder = Recording()
    res = reader.read(prompt, _req((1, 2), steps=3, trace=True))
    pins = pinned_positions(SLOTS, WIDTH, 3)
    free = [s.pos for s in SLOTS]
    for s in res.samples:
        for c in s.trace[:-1]:
            assert [c[p] for p in pins] == [s.canvas_in[p] for p in pins]
    assert seen[0] is None and len(seen) == 3
    for sc in seen[1:]:
        assert mx.all(sc[:, mx.array(pins)] == 0).item()
        assert mx.any(sc[:, mx.array(free)] != 0).item()


def _mean_entropy_at_first_step(reader, prompt, seeds):
    allowed = label_id_union(SLOTS)
    canvas = mx.array([build_canvas(TEMPLATE, SLOTS, WIDTH, s, reader.vocab)
                       for s in seeds], dtype=mx.int32)
    h = reader.decoder(canvas, cache=prompt.views_for(len(seeds)),
                       decoder_attention_mask=None)
    logits = denoise.unembed(reader, h, reader._label_rows(allowed))
    scaled = logits / reader.config.temperature(0)
    lp = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
    return (-(mx.exp(lp) * lp).sum(axis=-1)).mean(axis=-1).tolist()


def test_samples_converge_at_their_own_step(sliding_model):
    reader = _reader(sliding_model)
    prompt = reader.prefill(PROMPT)
    seeds = (11, 12)
    ent = _mean_entropy_at_first_step(reader, prompt, seeds)
    assert ent[0] != ent[1]
    early, late = (0, 1) if ent[0] < ent[1] else (1, 0)
    # One-canvas history makes every sample stable; the threshold between the
    # two first-step entropies converges one sample at step 1 only.
    reader.config.history = 1
    reader.config.confidence_threshold = (ent[0] + ent[1]) / 2
    res = reader.read(prompt, _req(seeds, steps=4, trace=True))
    fast, slow = res.samples[early], res.samples[late]
    assert len(fast.trace) == 2
    assert len(slow.trace) > 2

    one_step = reader.read(prompt, _req((seeds[early],)))
    for x, y in zip(fast.slots, one_step.samples[0].slots):
        assert x.argmax_id == y.argmax_id
        assert x.top == pytest.approx(y.top, abs=1e-5)
    alone = reader.read(prompt, _req((seeds[late],), steps=4, trace=True))
    assert alone.samples[0].trace == slow.trace
    _assert_reads_close(types.SimpleNamespace(samples=(slow,)), alone)


@pytest.mark.parametrize("steps", [1, 3])
def test_batched_sub_batched_and_sequential_reads_agree(model, steps):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    seeds = (1, 2, 3, 4)
    one_pass = reader.read(prompt, _req(seeds, steps=steps))
    sub = reader.read(prompt, _req(seeds, steps=steps, rows_per_pass=2 * WIDTH))
    _assert_reads_close(one_pass, sub)
    for i, sd in enumerate(seeds):
        alone = reader.read(prompt, _req((sd,), steps=steps))
        _assert_reads_close(types.SimpleNamespace(samples=(one_pass.samples[i],)),
                            alone)


def test_one_prefill_serves_many_reads(model):
    reader = _reader(model)
    prompt = reader.prefill(PROMPT)
    for seeds in ((1, 2), (3,), (1, 2)):
        shared = reader.read(prompt, _req(seeds))
        fresh = reader.read(reader.prefill(PROMPT), _req(seeds))
        _assert_reads_close(shared, fresh, atol=1e-6)


@pytest.mark.parametrize("step", [0, -1])
def test_a_step_of_zero_or_less_prefills_in_one_pass(model, step):
    reader = _reader(model, step=step)
    assert reader.prefill_step_size is None
    whole = reader.read(reader.prefill(PROMPT), _req((1, 2)))
    chunked = _reader(model).read(_reader(model).prefill(PROMPT), _req((1, 2)))
    _assert_reads_close(whole, chunked, atol=1e-4)


@pytest.mark.parametrize("n,path", [(1, "_update_in_place"), (8, "_update_concat")])
def test_extend_matches_a_prefill_of_the_whole_prompt(model, n, path, monkeypatch):
    reader = _reader(model)
    base = PROMPT[:12]
    more = list(range(40, 40 + n))
    grown = reader.prefill(base)
    calls = []
    real = getattr(RotatingKVCache, path)

    def spy(self, *args, **kwargs):
        calls.append(path)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(RotatingKVCache, path, spy)
    grown.extend(more)
    monkeypatch.undo()
    assert calls, f"extend of {n} did not take {path}"
    whole = reader.prefill(base + more)
    assert grown.ids == whole.ids
    for layer_type, a, b in zip(model.config.text_config.layer_types,
                                grown.views_for(1), whole.views_for(1)):
        assert a.offset == b.offset == len(base) + n
        assert a.keys.shape == b.keys.shape, layer_type
        assert mx.allclose(a.keys, b.keys, atol=1e-4).item(), layer_type
        assert mx.allclose(a.decoder_state[1], b.decoder_state[1], atol=1e-4).item()
    _assert_reads_close(reader.read(grown, _req((1, 2), steps=2)),
                        reader.read(whole, _req((1, 2), steps=2)), atol=1e-4)


def test_extend_keeps_the_sliding_cache_in_temporal_order(sliding_model):
    reader = _reader(sliding_model)
    grown = reader.prefill(PROMPT[:12])
    for t in range(40, 45):
        grown.extend([t])
    whole = reader.prefill(PROMPT[:12] + list(range(40, 45)))
    for a, b in zip(grown.cache, whole.cache):
        ka, kb = _cache_state(a)[0], _cache_state(b)[0]
        w = min(ka.shape[2], kb.shape[2])
        assert mx.allclose(ka[:, :, -w:], kb[:, :, -w:], atol=1e-4).item()


class _Backend:
    """The callable-tokenizer half of the processor pair: decode plus the
    stopping criteria the denoiser reads."""

    def __init__(self, eos=(EOS,)):
        self.stopping_criteria = StoppingCriteria(list(eos), self)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]


def _pair():
    backend = _Backend()
    processor = types.SimpleNamespace(
        tokenizer=backend, detokenizer=NaiveStreamingDetokenizer(backend))
    return processor, backend


def _think(reader, stop_id, seed=3, budget=24, should_stop=None):
    processor, backend = _pair()
    mx.random.seed(seed)
    ids, info = reader.think(PROMPT, budget, stop_id=stop_id, canvas_width=WIDTH,
                             processor=processor, backend=backend,
                             should_stop=should_stop)
    return ids, info, backend


def test_think_runs_the_denoiser_and_restores_the_stop_set(sliding_model):
    reader = _reader(sliding_model)
    ids, info, backend = _think(reader, stop_id=63)
    assert info["tokens"] == len(ids) and len(ids) <= 24
    assert ids or info["closed"]
    assert all(isinstance(t, int) and 0 <= t < reader.vocab for t in ids)
    # The denoiser adds the config EOS ids to the list; the call undoes it.
    assert backend.stopping_criteria.eos_token_ids == [EOS]


def test_think_stops_at_the_stop_id(sliding_model):
    reader = _reader(sliding_model)
    eos = {EOS, *sliding_model.config.generation_config["eos_token_id"]}
    first, _, _ = _think(reader, stop_id=63)
    k, stop = next((i, t) for i, t in enumerate(first) if t not in eos and t != 63)
    ids, info, _ = _think(reader, stop_id=stop)
    assert ids == first[:k]
    assert info["closed"] is True


def test_think_honours_should_stop(sliding_model):
    reader = _reader(sliding_model)
    processor, backend = _pair()
    calls = []

    def stop():
        calls.append(1)
        return True

    mx.random.seed(3)
    with pytest.raises(Cancelled):
        reader.think(PROMPT, 24, stop_id=63, canvas_width=WIDTH,
                     processor=processor, backend=backend, should_stop=stop)
    assert calls == [1]
    assert backend.stopping_criteria.eos_token_ids == [EOS]


def _result(token, n, *, finish=None, draft=False, block=False):
    return types.SimpleNamespace(token=token, generation_tokens=n,
                                 finish_reason=finish, is_draft=draft,
                                 diffusion_block_complete=block)


@pytest.mark.parametrize("script,want_ids,closed", [
    # Drafts and block markers carry no new token.
    ([_result(None, 0, draft=True), _result(7, 1), _result(8, 2),
      _result(8, 2, block=True), _result(9, 3, finish="stop")], [7, 8], True),
    # An end-of-turn stop other than the close tag stays in the thought.
    ([_result(7, 1), _result(EOS, 2, finish="stop")], [7, EOS], False),
    ([_result(7, 1), _result(8, 2), _result(8, 2, finish="length")], [7, 8], False),
])
def test_think_reads_the_denoiser_results(sliding_model, monkeypatch, script,
                                          want_ids, closed):
    processor, backend = _pair()
    seen = {}

    def fake(model, proc, back, input_ids, *args, **kwargs):
        seen["eos"] = list(back.stopping_criteria.eos_token_ids)
        seen["kwargs"] = kwargs
        seen["ids"] = input_ids.tolist()
        try:
            yield from script
        finally:
            seen["closed"] = True

    monkeypatch.setattr(engine, "stream_diffusion_generate", fake)
    ids, info = _reader(sliding_model).think(
        [2, 5], 16, stop_id=9, canvas_width=WIDTH, processor=processor,
        backend=backend)
    assert ids == want_ids and info["closed"] is closed
    assert seen["eos"] == [EOS, 9]
    assert seen["ids"] == [[2, 5]] and seen["closed"]
    kw = seen["kwargs"]
    assert kw["temperature"] == 1.0 and kw["diffusion_sampler"] == "entropy-bound"
    assert kw["diffusion_max_canvas_length"] == WIDTH and kw["max_tokens"] == 16
    assert backend.stopping_criteria.eos_token_ids == [EOS]


class _Lock:
    def __init__(self):
        self.held = False
        self.entries = 0

    def __enter__(self):
        assert not self.held, "lock taken twice"
        self.held = True
        self.entries += 1

    def __exit__(self, *exc):
        self.held = False


class _Wrapper:
    bos_token = "<bos>"

    def __init__(self, lock):
        self.lock = lock
        self.calls = []
        self.renders = []

    def apply_chat_template(self, msgs, **kwargs):
        assert self.lock.held
        self.renders.append((msgs, kwargs))
        return "<bos> sys " + msgs[0]["content"] + " user " + msgs[1]["content"]

    def encode(self, text, add_special_tokens=True):
        assert self.lock.held
        self.calls.append(add_special_tokens)
        ids = [2 if w == "<bos>" else len(w) for w in text.split()]
        return ([2] + ids) if add_special_tokens else ids

    def decode(self, ids):
        assert self.lock.held
        return "|".join(str(i) for i in ids)


def test_chat_tokens_render_encode_and_lock():
    lock = _Lock()
    wrapper = _Wrapper(lock)
    processor = types.SimpleNamespace(_wrapper=wrapper)
    renders = wrapper.renders
    tokens = ChatTokens(processor, "the state", lock=lock)

    assert tokens.enc("a bb") == [1, 2]
    assert tokens.decode([3, 4]) == "3|4"
    assert tokens.render("S", 1) == "<bos> sys S user the state"
    msgs, kwargs = renders[-1]
    assert msgs == [{"role": "system", "content": "S"},
                    {"role": "user", "content": "the state"}]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                      "enable_thinking": True}
    ids = tokens.chat_ids("S", False)
    assert renders[-1][1]["enable_thinking"] is False
    assert ids.count(2) == 1 and ids[0] == 2
    assert wrapper.calls == [False, False]
    assert lock.entries == 5 and not lock.held


def test_chat_tokens_without_a_lock_use_the_processor():
    wrapper = _Wrapper(types.SimpleNamespace(held=True))
    tokens = ChatTokens(wrapper)
    assert tokens.wrapper is wrapper
    assert tokens.enc("abc") == [3]


def test_engine_scope_on_cpu_seeds_and_clears_without_a_wired_limit(
        sliding_model, monkeypatch):
    cleared = []
    imported = []
    monkeypatch.setattr(mx, "clear_cache", lambda: cleared.append(1))
    monkeypatch.setattr(engine.importlib, "import_module",
                        lambda name: imported.append(name))

    def draw(seed):
        with engine_scope(sliding_model, seed):
            return mx.random.uniform(shape=(4,)).tolist()

    assert draw(5) == draw(5)
    assert draw(5) != draw(6)
    assert imported == []
    assert len(cleared) == 4
    with pytest.raises(RuntimeError):
        with engine_scope(sliding_model, 1):
            raise RuntimeError("job failed")
    assert len(cleared) == 5
