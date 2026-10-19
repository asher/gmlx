"""The owned MTP prompt step keeps the stock prompt-step contract.

Runs the stock PromptProcessingBatch with the gmlx MTP prefill installed and
a stub model that records every forward. A right-padded mixed batch ends a
chunk at each shorter row's last real column and keeps that row's logits,
mrope position ids are sliced on their sequence axis, and the APC
coordinator sees the cache after every chunk. CPU only: the stub runs no
kernel beyond array ops.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from gmlx._vendor.mlx_vlm.models.cache import BatchKVCache

from gmlx.models.gemma4.language import LanguageModel as Gemma4LanguageModel

_V = 32
_D = 4


class _Out:
    def __init__(self, logits, hidden):
        self.logits = logits
        self.hidden_states = [hidden]
        self.shared_kv_states = None


class _Model:
    """Each column's logits are the one-hot of that column's token, so a
    greedy first token is the last real prompt token of its row."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
        b, n = input_ids.shape
        self.calls.append({"n": n, **kw})
        kv = mx.zeros((b, 1, n, _D))
        for c in cache:
            c.update_and_fetch(kv, kv)
        logits = (mx.arange(_V)[None, None, :]
                  == input_ids[..., None]).astype(mx.float32)
        keep = kw.get("logits_to_keep")
        if keep:
            logits = logits[:, -int(keep):]
        return _Out(logits, inputs_embeds)


class _Coordinator:
    is_checkpoint = False

    def __init__(self):
        self.observed = []

    def prepare_prefill(self, *args, **kwargs):
        pass

    def observe_cache(self, prompt_cache, token_count, *, batch_size=1):
        self.observed.append((token_count, batch_size))


class _SpecBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def ar(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar

    import gmlx.spec.engine as engine
    import gmlx.spec.mtp_prefill as mtp_prefill

    mtp_prefill.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(engine, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(ar, "SpeculativeGenerationBatch", _SpecBatch)
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    return ar


def _embeds(rows, width):
    out = []
    for ids in rows:
        e = mx.array(ids + [0] * (width - len(ids)), dtype=mx.float32)
        out.append(mx.broadcast_to(e[:, None], (width, _D)))
    return mx.stack(out)


def _run(batch):
    steps = []
    while batch.needs_processing():
        steps.append(batch.prompt_step())
    gen = batch.generate(lambda x: mx.argmax(x, axis=-1), None)
    return steps, gen


def test_right_padded_rows_keep_their_last_real_logits(ar, monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "16")
    rows = [[5, 6, 7], [11, 12, 13, 14, 15, 16, 17]]
    model, coordinator = _Model(), _Coordinator()
    batch = ar.PromptProcessingBatch(
        model, [0, 1], rows, [4, 4], _embeds(rows, 7), {},
        warm_cache=[BatchKVCache(left_padding=[0, 0])],
        apc_coordinator=coordinator,
        right_pad_per_row=[4, 0], suffix_lens=[3, 7],
        draft_model=object(), draft_kind="mtp")

    steps, gen = _run(batch)

    # The first chunk stops at row 0's end; the final forward is one column.
    assert steps == [3, 3]
    assert [c["n"] for c in model.calls] == [3, 3, 1]
    assert gen.first_tokens.tolist() == [7, 17]
    assert coordinator.observed == [(3, 2), (6, 2)]


def test_mrope_position_ids_slice_on_the_sequence_axis(ar, monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    ids = [1, 2, 3, 4, 5, 6]
    pos = mx.stack([mx.arange(6) + 10 * k for k in (1, 2, 3)])[:, None, :]
    model = _Model()
    batch = ar.PromptProcessingBatch(
        model, [0], [ids], [4], _embeds([ids], 6), {"position_ids": pos},
        warm_cache=[BatchKVCache(left_padding=[0])],
        draft_model=object(), draft_kind="mtp")

    _run(batch)

    seen = [c["position_ids"] for c in model.calls]
    assert [c["n"] for c in model.calls] == [2, 2, 1, 1]
    assert [p.shape for p in seen] == [(3, 1, 2), (3, 1, 2), (3, 1, 1),
                                       (3, 1, 1)]
    assert mx.concatenate(seen, axis=2).tolist() == pos.tolist()


class _BidirModel(_Model):
    """The owned gemma-4 chunking policy, on a bidirectional image
    config."""

    config = SimpleNamespace(use_bidirectional_attention="vision")

    def chunked_prefill_policy(self, **kwargs):
        return Gemma4LanguageModel.chunked_prefill_policy(self, **kwargs)


@pytest.mark.parametrize("types,calls", [
    ([0, 1, 1, 1, 0, 0, 0, 0], [1, 3, 2, 1, 1]),
    ([0] * 8, [2, 2, 2, 1, 1]),
])
def test_mtp_chunks_an_image_prompt_at_block_edges(
        ar, monkeypatch, types, calls):
    """Under bidirectional image attention an MTP batch built the way
    serve builds it chunks at the MTP step, with the boundaries at the
    image edges: a boundary inside the image moves to its start, and the
    image, longer than the step, prefills in one chunk."""
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    ids = [1, 5, 5, 5, 2, 3, 4, 6]
    model = _BidirModel()
    batch = ar.PromptProcessingBatch(
        model, [0], [ids], [4], _embeds([ids], 8),
        {"mm_token_type_ids": mx.array([types])},
        warm_cache=[BatchKVCache(left_padding=[0])], prefill_step_size=None,
        draft_model=object(), draft_kind="mtp")

    _run(batch)

    assert [c["n"] for c in model.calls] == calls


@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
def test_a_row_ending_inside_another_rows_image_keeps_its_logits(
        ar, monkeypatch, mtp):
    """A right-padded batch whose shorter row ends inside the longer row's
    image. The chunk keeps the image whole, runs past that row end, and
    the row keeps the logits of its last real column."""
    from gmlx.gen import media_spans

    assert media_spans.install_span_aware_prompt_step()
    monkeypatch.setenv("PREFILL_STEP_SIZE", "16")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    rows = [[5, 6, 7], [11, 12, 13, 14, 15, 16, 17]]
    types = [[0] * 7, [0, 1, 1, 1, 1, 1, 0]]
    model = _BidirModel()
    kw = dict(draft_model=object(), draft_kind="mtp") if mtp else dict(
        prefill_step_size=16)
    batch = ar.PromptProcessingBatch(
        model, [0, 1], rows, [4, 4], _embeds(rows, 7),
        {"mm_token_type_ids": mx.array(types)},
        warm_cache=[BatchKVCache(left_padding=[0, 0])],
        right_pad_per_row=[4, 0], suffix_lens=[3, 7], **kw)

    steps = []
    while batch.needs_processing():
        steps.append(batch.prompt_step())

    # The first chunk stops at the image start, the second takes the
    # whole image past row 0's end at column 3.
    assert steps == [1, 5]
    assert int(mx.argmax(batch._finished_prompt_logits[0]).item()) == 7
    assert batch._suffix_lens == [3, 7]


def test_l0_restore_trims_mrope_position_ids(ar, monkeypatch):
    import gmlx.spec.engine as engine
    import gmlx.spec.mtp_prefill as mtp_prefill
    from gmlx.cache.prefix_cache import SpecPrefixCache

    monkeypatch.setattr(engine, "_SPEC_APC_DISABLED", False)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", False)
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    pos = mx.stack([mx.arange(9) + 10 * k for k in (1, 2, 3)])[:, None, :]
    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    # The first prompt stores an L0 entry that the second one extends.
    for ids in ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8, 9]):
        model.calls.clear()
        batch = ar.PromptProcessingBatch(
            model, [0], [ids], [4], _embeds([ids], len(ids)),
            {"position_ids": pos[:, :, :len(ids)]},
            warm_cache=[BatchKVCache(left_padding=[0])],
            draft_model=object(), draft_kind="mtp")
        _run(batch)

    assert batch._mtp_apc_prefix_len == 6
    seen = [c["position_ids"] for c in model.calls]
    assert [c["n"] for c in model.calls] == [2, 1]
    assert mx.concatenate(seen, axis=2).tolist() == pos[:, :, 6:].tolist()


def test_a_multi_row_batch_logs_the_skipped_owned_lookup(monkeypatch, caplog):
    """A multi-row prompt batch skips the owned prefix lookup only. Its rows
    keep what the stock prefix cache restored, so the log does not say the
    APC was skipped."""
    import logging

    import gmlx.spec.mtp_prefill as mtp_prefill

    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", False)
    batch = SimpleNamespace(
        model=_Model(), _input_ids=mx.array([[1, 2, 3], [4, 5, 6]]),
        _inputs_embeds=mx.zeros((2, 3, _D)), _prompt_kwargs={},
        _apc_meta=None, prompt_cache=[])
    with caplog.at_level(logging.WARNING, logger="gmlx.spec.mtp_prefill"):
        mtp_prefill._mtp_prefill_init(batch)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "owned MTP prefix lookup skipped" in text
    assert "APC skipped" not in text


def test_prefill_step_env_override(monkeypatch):
    """Stock nulls prefill_step_size for speculative batches; the MTP
    re-enable must resolve None from the serve-side PREFILL_STEP_SIZE env
    override rather than pin the stock default. Exercised directly on the
    patched prompt_step (no model needed): the resolution runs before the
    needs_processing gate."""
    import types

    from gmlx._vendor.mlx_vlm.generate.ar import PromptProcessingBatch

    import gmlx.spec.mtp_prefill as mtp_prefill

    mtp_prefill.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(mtp_prefill, "_mtp_prefill_init", lambda s: None)

    def fake_batch():
        # _inputs_embeds: the span-aware wrapper (media_spans) reads it
        # when a serve test has layered it over prompt_step.
        return types.SimpleNamespace(
            draft_kind="mtp",
            prefill_step_size=None,
            needs_processing=lambda: False,
            _inputs_embeds=None,
        )

    monkeypatch.setenv("PREFILL_STEP_SIZE", "97")
    fake = fake_batch()
    assert PromptProcessingBatch.prompt_step(fake) == 0
    assert fake.prefill_step_size == 97

    monkeypatch.delenv("PREFILL_STEP_SIZE")
    fake = fake_batch()
    PromptProcessingBatch.prompt_step(fake)
    assert fake.prefill_step_size == 2048

    # A streamed model takes the step serve adopted for it.
    fake = fake_batch()
    fake.model = types.SimpleNamespace(_gmlx_adopted_prefill_step=8192)
    PromptProcessingBatch.prompt_step(fake)
    assert fake.prefill_step_size == 8192
