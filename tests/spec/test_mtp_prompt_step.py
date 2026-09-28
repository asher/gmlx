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
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.models.gemma4.language import LanguageModel as Gemma4LanguageModel

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
    from mlx_vlm.generate import ar

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
    """The stock gemma-4 chunking policy on a bidirectional image config."""

    config = SimpleNamespace(use_bidirectional_attention="vision")
    chunked_prefill_policy = Gemma4LanguageModel.chunked_prefill_policy


@pytest.mark.parametrize("types,calls", [
    ([0, 1, 1, 1, 0, 0, 0, 0], [8]),
    ([0] * 8, [2, 2, 2, 1, 1]),
])
def test_mtp_keeps_one_pass_when_the_target_declines_chunking(
        ar, monkeypatch, types, calls):
    """Asked without a drafter, the gemma-4 policy declines to chunk an
    image prompt under bidirectional image attention, so stock plain
    serve prefills it in one forward. An MTP batch built the way serve
    builds it keeps that one forward and chunks a text prompt at the MTP
    step."""
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
