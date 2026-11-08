"""The scoring reader: prompt-token logprobs on tiny models.

The reference for bit-equality is the same model run chunked at the
reader's widths, with the float32 log-softmax gathered at the next id.
Chunked prefill in bf16 is not bit-equal to one forward, so against one
forward the checks use ``TOL``, measured on these models (largest gap
1e-6 over 20 prompts) with a margin. A live adapter is not bit-equal to
its merged weights; ``LORA_TOL`` is measured the same way (gap 0.0013)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx import lora_rows
from gmlx._vendor.mlx_lm.models import granite, llama, minicpm
from gmlx.cache.kv_cache import make_prompt_cache
from gmlx.gen import score_reader as sr
from gmlx.gen.logprobs import raw_logprobs, top_logprobs
from gmlx.models.vlm_text_only import Model as TextOnlyModel
from gmlx.systemone.ar_reader import run_to_end

_V = 64
_K = 3
TOL = 1e-3
LORA_TOL = 0.01


def _llama(dtype=mx.bfloat16, vocab=_V):
    mx.random.seed(0)
    inner = llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4,
        num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=vocab))
    m = TextOnlyModel(inner)
    m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def _granite():
    mx.random.seed(1)
    inner = granite.Model(granite.ModelArgs(
        model_type="granite", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-5,
        vocab_size=_V, logits_scaling=3.0, attention_multiplier=0.25,
        embedding_multiplier=2.0, residual_multiplier=0.5,
        max_position_embeddings=512, num_key_value_heads=2,
        attention_bias=False, mlp_bias=False, rope_theta=10000.0))
    m = TextOnlyModel(inner)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def _minicpm():
    mx.random.seed(2)
    inner = minicpm.Model(minicpm.ModelArgs(
        model_type="minicpm", hidden_size=32, dim_model_base=8,
        num_hidden_layers=2, intermediate_size=64, num_attention_heads=4,
        rms_norm_eps=1e-5, vocab_size=_V, num_key_value_heads=2,
        scale_depth=1.4, scale_emb=12.0, max_position_embeddings=512))
    m = TextOnlyModel(inner)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def _ids(n, seed=3):
    mx.random.seed(seed)
    return mx.random.randint(0, _V, (n,)).tolist()


def _score(reader, ids, k=_K, scope=None):
    def job():
        if not reader.built:
            yield from reader.build(scope)
        return (yield from reader.score(ids, k, scope))
    return run_to_end(job())


def _chunked(model, ids, steps, r, k=_K):
    """The reference: the language model run at the reader's widths."""
    lm = model.language_model
    cache = make_prompt_cache(lm)
    out = []
    for depth, w in steps:
        logits = lm(mx.array([ids[depth:depth + w]]), cache=cache).logits[0]
        targets = ids[depth + 1:depth + w + 1]
        for a in range(0, w, r):
            lp = raw_logprobs(logits[a:a + r])
            t = mx.array(targets[a:a + r])[:, None]
            chosen = mx.take_along_axis(lp, t, axis=-1)[:, 0]
            tid, tval = top_logprobs(lp, k)
            out.extend(zip(chosen.tolist(), tid.tolist(), tval.tolist()))
    return out


def _one_shot(model, ids):
    lm = model.language_model
    lp = raw_logprobs(lm(mx.array([ids[:-1]]), cache=make_prompt_cache(lm)).logits[0])
    return mx.take_along_axis(lp, mx.array(ids[1:])[:, None], axis=-1)[:, 0]


def _gap(entries, ref):
    return mx.max(mx.abs(mx.array([e[0] for e in entries]) - ref)).item()


@pytest.fixture
def small(monkeypatch):
    """Forwards of 16 tokens and reductions of 4 columns at V = 64."""
    monkeypatch.setattr(sr, "FORWARD_TOKENS", 16)
    monkeypatch.setattr(sr, "REDUCE_CAP", 4 * _V * 16)


def test_every_position_equals_the_chunked_reference_bit_for_bit():
    model = _llama()
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    ids = _ids(40)
    got = _score(reader, ids)
    assert len(got) == len(ids) - 1
    assert got == _chunked(model, ids, reader.plan(len(ids) - 1), reader.reduce)
    assert _gap(got, _one_shot(model, ids)) <= TOL


def test_several_forwards_at_small_widths_stay_continuous(small):
    model = _llama()
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    ids = _ids(50)
    got = _score(reader, ids)
    steps = reader.plan(len(ids) - 1)
    assert [w for _, w in steps] == [16, 16, 16, 1] and reader.reduce == 4
    assert len(got) == len(ids) - 1
    assert got == _chunked(model, ids, steps, 4)
    assert _gap(got, _one_shot(model, ids)) <= TOL


def test_two_runs_are_bit_equal(small):
    model = _llama()
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    ids = _ids(37)
    assert _score(reader, ids) == _score(reader, ids)


def test_k_zero_gives_the_token_logprob_alone():
    model = _llama()
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    ids = _ids(12)
    got = _score(reader, ids, k=0)
    assert all(e[1] is None and e[2] is None for e in got)
    assert [e[0] for e in got] == [e[0] for e in _score(reader, ids)]


def test_an_adapter_at_half_strength_scores_as_the_merged_model(monkeypatch):
    import gmlx.load.modules as modules

    model = _llama()
    merged = _llama()
    attn = model.language_model._model.model.layers[0].self_attn
    mx.random.seed(5)
    a = (mx.random.normal((2, 32)) * 0.5).astype(mx.bfloat16)
    b = (mx.random.normal((32, 2)) * 0.5).astype(mx.bfloat16)
    base = attn.q_proj
    attn.q_proj = modules.LoRAKQuantLinear(base, a, b, 1.0)
    target = merged.language_model._model.model.layers[0].self_attn.q_proj
    target.weight = (base.weight.astype(mx.float32)
                     + 0.5 * (b.astype(mx.float32) @ a.astype(mx.float32))
                     ).astype(mx.bfloat16)
    lora_rows.configure("rows", 1)
    try:
        reader = sr.ScoreReader(model, ws=16e9, streamed=False)
        ids = _ids(30)
        got = _score(reader, ids, scope=lora_rows.rows_scope((0.5,)))
    finally:
        lora_rows.configure("static", 1)
    ref = _chunked(merged, ids, reader.plan(len(ids) - 1), reader.reduce)
    assert _gap(got, mx.array([e[0] for e in ref])) <= LORA_TOL
    plain = _score(sr.ScoreReader(_llama(), ws=16e9, streamed=False), ids)
    assert _gap(got, mx.array([e[0] for e in plain])) > 10 * LORA_TOL


def test_a_resident_model_takes_the_full_path():
    reader = sr.ScoreReader(_granite(), ws=16e9, streamed=False)
    _score(reader, _ids(10))
    assert reader.split is None


def test_a_streamed_granite_takes_the_split_and_matches_its_full_path(
        small, monkeypatch):
    monkeypatch.setattr(sr, "adopted_step", lambda model: 16)
    ids = _ids(50)
    full = sr.ScoreReader(_granite(), ws=16e9, streamed=False)
    want = _score(full, ids)
    split = sr.ScoreReader(_granite(), ws=16e9, streamed=True)
    got = _score(split, ids)
    assert split.split is not None and split.split[2] is not None
    assert split.plan(len(ids) - 1) == full.plan(len(ids) - 1)
    if mx.default_device() == mx.gpu:
        assert got == want
        return
    # The CPU matmul rounds by row count: the head over a 4-column
    # sub-chunk and over the 16-column forward differ in the last bit.
    assert [e[1] for e in got] == [e[1] for e in want]
    flat = lambda es: mx.array([[e[0], *e[2]] for e in es])  # noqa: E731
    assert mx.max(mx.abs(flat(got) - flat(want))).item() <= 1e-5


def test_a_streamed_minicpm_fails_the_check_and_takes_the_full_path(small):
    model = _minicpm()
    reader = sr.ScoreReader(model, ws=16e9, streamed=True)
    ids = _ids(20)
    got = _score(reader, ids)
    assert reader.split is None
    assert got == _chunked(model, ids, reader.plan(len(ids) - 1), reader.reduce)


class _OneColumn(nn.Module):
    """A language model that returns logits for its last column only."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, cache=None, **kwargs):
        out = self.inner(x, cache=cache, **kwargs)
        return SimpleNamespace(logits=out.logits[:, -1:])

    def make_cache(self):
        return self.inner.make_cache()


def test_a_forward_that_cuts_columns_is_refused_and_the_verdict_kept():
    base = _llama()
    model = SimpleNamespace(language_model=_OneColumn(base.language_model))
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    with pytest.raises(sr.ScoringRefused, match="every prompt token"):
        _score(reader, _ids(10))
    assert reader.refusal == sr.NO_COLUMNS
    with pytest.raises(sr.ScoringRefused):
        run_to_end(reader.score(_ids(4), 0))


def test_the_widths():
    v = 248_320
    assert sr.logits_width(v, 2, 16e9) == 512
    assert sr.logits_width(v, 2, 5.7e9) == 512
    assert sr.logits_width(v, 4, 16e9) == 256
    assert sr.reduce_width(v, 16e9) == 128
    assert sr.reduce_width(v, 5.7e9) == 64
    assert sr.reduce_width(49_152, 64e9) == 1024
    steps = sr.plan(512, 131_072, 32, 16e9)
    assert dict(steps)[0] == 512
    assert dict(steps).get(32_768) == 512
    assert dict(steps)[65_536] == 256


def test_a_halving_inside_the_prompt_is_priced_at_the_wider_forward():
    steps = sr.plan(512, 65_536, 32, 16e9)
    assert steps[-1][1] == 256
    final = 32 * 256 * (65_536 + 256) * 2.0
    wide = max(32 * w * (d + w) * 2.0 for d, w in steps if w == 512)
    assert sr.attention_bytes(steps, 32) == wide > final
    assert wide == pytest.approx(2.0e9, rel=0.01)


def test_the_price_before_the_reader_is_the_larger_itemsize(monkeypatch):
    v = 248_320
    model = SimpleNamespace(config=SimpleNamespace(
        vocab_size=v, num_attention_heads=16, num_hidden_layers=2,
        num_key_value_heads=4, head_dim=256))
    monkeypatch.setattr("gmlx.stream.expert_streaming.experts_streamed",
                        lambda m: False)
    each = {size: sr.transient_bytes(
        2048, vocab=v, itemsize=size, width=sr.logits_width(v, size, 5.7e9),
        reduce=64, split=False, heads=16, ws=5.7e9) for size in (2, 4)}
    assert sr.price_bytes(model, 2048, None, ws=5.7e9) == max(each.values())
    reader = SimpleNamespace(built=True, vocab=v, itemsize=4, width=256,
                             reduce=64, split=None, heads=16, ws=5.7e9)
    assert sr.price_bytes(model, 2048, reader) == each[4]  # type: ignore[arg-type]


def test_the_reader_records_the_logits_dtype_when_built(monkeypatch):
    monkeypatch.setattr(sr, "LOGITS_CAP", 100_000)
    model = _llama(dtype=mx.float32)
    widths = []
    lm = model.language_model
    call = type(lm).__call__

    def counting(self, x, *a, **k):
        widths.append(x.shape[1])
        return call(self, x, *a, **k)

    monkeypatch.setattr(type(lm), "__call__", counting)
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    assert reader.dtype is None
    ids = _ids(300)
    for _ in range(2):
        widths.clear()
        _score(reader, ids, k=0)
        assert reader.dtype == mx.float32 and reader.width == 256
        assert max(widths) == 256


def test_every_scoring_forward_runs_on_the_language_model():
    model = _llama()
    outer_calls = []

    class Outer:
        language_model = model.language_model
        get_input_embeddings = model.get_input_embeddings

        def __call__(self, *a, **k):
            outer_calls.append(1)
            raise AssertionError("the outer class was called")

    reader = sr.ScoreReader(Outer(), ws=16e9, streamed=False)
    _score(reader, _ids(20))
    assert reader.refusal is None and not outer_calls


class _ScaledEmbeds:
    """An outer class whose text embedding is not the inner lookup."""

    def __init__(self, model, scale):
        self.language_model = model.language_model
        self.inner = model
        self.scale = scale

    def get_input_embeddings(self, ids, pixel_values=None, **kwargs):
        got = self.inner.get_input_embeddings(ids, pixel_values)
        return SimpleNamespace(to_dict=lambda: {
            "inputs_embeds": got.inputs_embeds * self.scale})


def test_the_ids_and_embeddings_probe():
    passes = sr.ScoreReader(_ScaledEmbeds(_llama(), 1), ws=16e9, streamed=False)
    run_to_end(passes.build())
    assert passes.refusal is None
    refused = sr.ScoreReader(_ScaledEmbeds(_llama(), 2), ws=16e9, streamed=False)
    forwards = refused.forwards
    run_to_end(refused.build())
    assert refused.refusal == sr.EMBEDS_DIFFER
    with pytest.raises(sr.ScoringRefused):
        run_to_end(refused.score(_ids(5), 0))
    assert refused.forwards == forwards


_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="qwen3_5 GDN forward is Metal-only")


@_NEEDS_GPU
def test_a_hybrid_scores_as_its_chunked_reference(small):
    from tests.systemone.test_systemone_ar_reader import _model

    model = SimpleNamespace(language_model=_model())
    reader = sr.ScoreReader(model, ws=16e9, streamed=False)
    ids = [i % 58 + 2 for i in _ids(45)]
    got = _score(reader, ids)
    lm = model.language_model
    cache = lm.make_cache()
    ref = []
    for depth, w in reader.plan(len(ids) - 1):
        out = lm(mx.array([ids[depth:depth + w]]), cache=cache)
        logits = getattr(out, "logits", out)[0]
        for a in range(0, w, reader.reduce):
            lp = raw_logprobs(logits[a:a + reader.reduce])
            t = mx.array(ids[depth + 1 + a:depth + 1 + a + reader.reduce])[:, None]
            ref.extend(mx.take_along_axis(lp, t, axis=-1)[:, 0].tolist())
    assert [e[0] for e in got] == ref
