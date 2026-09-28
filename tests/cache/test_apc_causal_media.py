"""Qwen3.5 on the serve engine: a turn that adds an image restores the
earlier turns, and the suffix after the restored prefix, image included,
prefills with the logits, mrope position ids and rope deltas of a cold
run. Run and chat keep the stock media floor."""

import dataclasses
import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm import apc as _apc
from mlx_vlm.utils import StoppingCriteria

import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.apc_media import (
    CausalMedia,
    WholeMediaBlocks,
    install_causal_serve_media,
    install_media_token_ids,
    safe_prefix,
)
from gmlx.cache.media_salt import install_media_salt
from gmlx.cache.prefix_cache import SpecPrefixCache
from gmlx.gen import media_spans
from gmlx.load.vlm import _swap_spec_language_model
from gmlx.spec import engine
from gmlx.spec.admission import install_continuous_batch_admission
from gmlx.spec.engine import install_owned_spec_engine
from gmlx.spec.mtp_drafter import QwenMTPDrafter
from gmlx.spec.mtp_load import _arm_spec_target

from test_vlm_spec_target_swap import _q35_vlm

ar = importlib.import_module("mlx_vlm.generate.ar")
server = importlib.import_module("mlx_vlm.server.generation")

# The model is f32. Warm and cold prefill the suffix in different GEMM
# shapes, and f32 GEMM runs at TF32 on the GPU. A row of a B=2 batch
# runs at other shapes again, padded to the longer row.
_ATOL = 1e-2
_ATOL_B2 = 2e-2
_V = 128
_IMG = [126] + [124] * 4 + [127]
_GRID = [1, 4, 4]


@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "8")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert media_spans.install_span_aware_prompt_step()
    install_media_token_ids()
    assert install_media_salt()
    assert install_causal_serve_media()


_MODEL = []


def _model():
    if not _MODEL:
        mx.random.seed(5)
        model = _q35_vlm()
        mx.eval(model.parameters())
        _MODEL.append(model)
    return _MODEL[0]


def _img(seed):
    return mx.random.normal((16, 1176), key=mx.random.key(seed))


def _request(model, ids, pixels):
    """``_gpu_embed`` as the server calls it, with the salt its request
    handler computes."""
    raw = {"input_ids": mx.array([ids])}
    pv = None
    if pixels:
        pv = raw["pixel_values"] = mx.concatenate(pixels, axis=0)
        raw["image_grid_thw"] = mx.array([_GRID] * len(pixels))
    salt = _apc.semantic_extra_hash(
        tenant=None,
        image_hash=_apc.hash_image_payload(pixel_values=pv) if pixels else 0,
        media={"audio": None, "video": None}, model=model.language_model,
        processor=None)
    rg = SimpleNamespace(model=model, vision_cache=None)
    _ids, kw = server.ResponseGenerator._gpu_embed(
        rg, raw, None, apc_semantic_hash=salt)
    return ids, kw


class _Positions:
    """The mrope position ids and rope deltas each text-model forward
    reads, and the hidden states it returns, per call."""

    def __init__(self, monkeypatch, lm):
        self.calls = []
        inner = type(lm.model)
        stock = inner.__call__

        def forward(this, inputs, *a, **kw):
            pos = kw.get("position_ids")
            deltas = lm._rope_deltas
            out = stock(this, inputs, *a, **kw)
            x = inputs if inputs is not None else kw["inputs_embeds"]
            h = out[0] if isinstance(out, (tuple, list)) else out
            self.calls.append((
                int(x.shape[1]),
                None if pos is None else np.array(pos),
                None if deltas is None else np.array(deltas),
                np.array(h.astype(mx.float32))))
            return out

        monkeypatch.setattr(inner, "__call__", forward)

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def _run(model, manager, rows, head=None):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    spec = {"top_logprobs_k": _V} if head is None else {
        "draft_model": head, "draft_kind": "mtp", "greedy_sampling": True}
    gen = ar.BatchGenerator(model.language_model, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=4, apc_manager=manager, **spec)
    uids = gen.insert([r[0] for r in rows], [4] * len(rows),
                      prompt_kwargs=[r[1] for r in rows])
    out = {u: [] for u in uids}
    while gen.has_work:
        _p, resp = gen.next()
        for r in resp:
            if r.uid in out:
                out[r.uid].append((int(r.token), dict(r.top_logprobs or ())))
    gen.close()
    return [out[u] for u in uids]


def _manager():
    return GmlxAPCManager(num_blocks=512, block_size=16)


def _matched(manager):
    return int(manager.stats_snapshot().get("matched_tokens", 0) or 0)


def _assert_same(warm, cold, atol=_ATOL):
    assert len(warm) == len(cold) == 4
    assert [t for t, _l in warm] == [t for t, _l in cold]
    for (_t, lw), (_c, lc) in zip(warm, cold):
        assert max(abs(lw[k] - lc[k]) for k in lc) < atol


def _turns():
    """A 96-token first turn with image A, and a second turn that adds
    image B after more text."""
    turn1 = ([1 + i % 50 for i in range(40)] + [60, 61] + _IMG
             + [62 + i % 30 for i in range(48)])
    turn2 = (turn1 + [70 + i % 20 for i in range(30)] + _IMG
             + [83 + i % 10 for i in range(20)])
    return turn1, turn2


def test_turn_two_restores_the_prefix_before_its_new_image(serve,
                                                           monkeypatch):
    model = _model()
    pos = _Positions(monkeypatch, model.language_model)
    turn1, turn2 = _turns()
    m = _manager()
    _run(model, m, [_request(model, turn1, [_img(1)])])
    before = _matched(m)
    pos.take()
    warm = _run(model, m, [_request(model, turn2, [_img(1), _img(2)])])
    warm_pos = pos.take()
    assert _matched(m) - before == len(turn1)
    cold = _run(model, _manager(),
                [_request(model, turn2, [_img(1), _img(2)])])
    cold_pos = pos.take()
    _assert_same(warm[0], cold[0])

    # The mrope risk: the suffix's position ids, image B's included, and
    # the rope deltas every decode step reads.
    def prefill(calls):
        return np.concatenate([p for n, p, _d, _h in calls if n > 1],
                              axis=-1)

    assert np.array_equal(prefill(warm_pos),
                          prefill(cold_pos)[..., len(turn1):])
    decode_w = [(p, d) for n, p, d, _h in warm_pos if n == 1]
    decode_c = [(p, d) for n, p, d, _h in cold_pos if n == 1]
    assert len(decode_w) == len(decode_c) >= 3
    for (pw, dw), (pc, dc) in zip(decode_w, decode_c):
        assert np.array_equal(pw, pc) and np.array_equal(dw, dc)
    # Image B moves the positions after it: the check has teeth.
    assert int(decode_c[0][1].reshape(-1)[0]) != 0


def test_a_mixed_batch_restores_one_row_and_prefills_the_other(serve):
    model = _model()
    turn1, turn2 = _turns()
    other = ([3 + i % 40 for i in range(50)] + _IMG
             + [5 + i % 7 for i in range(30)])
    m = _manager()
    _run(model, m, [_request(model, turn1, [_img(1)])])
    before = _matched(m)
    pair = _run(model, m, [_request(model, turn2, [_img(1), _img(2)]),
                           _request(model, other, [_img(5)])])
    assert _matched(m) - before == len(turn1)
    _assert_same(pair[0], _run(model, _manager(), [
        _request(model, turn2, [_img(1), _img(2)])])[0], _ATOL_B2)
    _assert_same(pair[1], _run(model, _manager(), [
        _request(model, other, [_img(5)])])[0], _ATOL_B2)


def test_the_serve_guard_reads_causal_media_for_qwen35_only(serve):
    model = _model()
    bg = SimpleNamespace(model=model.language_model)
    ids = ar.BatchGenerator._apc_media_token_ids(bg)
    assert isinstance(ids, CausalMedia) and 124 in ids
    assert safe_prefix([1, 124, 124, 2], ids)(2)
    assert _apc.media_safe_prefix_min([1, 124, 2], ids) == 0
    assert _apc.prefix_leaves_text_only_suffix([1, 124, 2], 1, ids)
    assert _apc.adjust_prefix_to_text_suffix_boundary([1, 124, 2, 3], 1,
                                                      ids) == 1
    # The owned MTP path reads the same helper through its view.
    engine._bind_l1_view()
    view = engine._L1View(model.language_model, None, None)
    assert isinstance(view._apc_media_token_ids(), CausalMedia)
    # Another model type keeps its rule.
    other = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        model_type="qwen3_vl", image_token_id=124)))
    got = ar.BatchGenerator._apc_media_token_ids(other)
    assert not isinstance(got, (CausalMedia, WholeMediaBlocks))
    assert _apc.media_safe_prefix_min([1, 124, 2], got) == 2


def test_run_and_chat_keep_the_stock_floor(serve):
    """Run and chat read the container config, not the serve engine's
    helper, and drop the pixel values when they reuse a prefix."""
    model = _model()
    ids = _apc.multimodal_token_ids_from_config(model.config)
    assert not isinstance(ids, CausalMedia)
    assert _apc.media_safe_prefix_min([1, 124, 124, 2], ids) == 3


_MTP = []


def _mtp_model():
    """A second tiny Qwen3.5, armed as an MTP target the way the VLM
    loader arms one, with a random-weight MTP head bound to it."""
    if not _MTP:
        mx.random.seed(5)
        model = _q35_vlm()

        def quiet(*_a, **_k):
            return None

        _swap_spec_language_model(model, "qwen3_5", log=quiet)
        _arm_spec_target(model, {"model_type": "qwen3_5"},
                         stock_vlm_regime=True, log=quiet)
        text = model.config.text_config
        cfg = importlib.import_module(
            "mlx_vlm.speculative.drafters.qwen3_5_mtp.config")
        head = QwenMTPDrafter(cfg.Qwen3_5MTPConfig.from_dict({
            "model_type": "qwen3_5_mtp",
            "text_config": {f.name: getattr(text, f.name)
                            for f in dataclasses.fields(text)}}))
        mx.eval(model.parameters(), head.parameters())
        head.bind(model.language_model)
        _MTP.append((model, head))
    return _MTP[0]


def _leading(calls, n):
    """Split off the leading forwards that prefill ``n`` tokens."""
    total = 0
    for k, call in enumerate(calls):
        if total == n:
            return calls[:k], calls[k:]
        total += call[0]
    raise AssertionError(f"no forward boundary at {n} tokens")


@pytest.mark.parametrize("tier", ["l0", "l1"])
def test_the_mtp_target_restores_the_prefix_before_its_new_image(
        serve, monkeypatch, tier):
    """The owned MTP path reads the guard through ``_L1View``. Turn 2
    restores turn 1 from the MTP prefix cache (l0) or, with that cache
    emptied, from the checkpoint tier (l1)."""
    install_owned_spec_engine()
    install_continuous_batch_admission()
    model, head = _mtp_model()
    lm = model.language_model
    pos = _Positions(monkeypatch, lm)
    hits = []
    lookup = SpecPrefixCache.lookup

    def record(self, *a, **kw):
        hit = lookup(self, *a, **kw)
        hits.append(0 if hit is None else hit[0])
        return hit

    monkeypatch.setattr(SpecPrefixCache, "lookup", record)
    turn1, turn2 = _turns()
    lm._spec_prefix_cache = None
    m = _manager()
    _run(model, m, [_request(model, turn1, [_img(1)])], head=head)
    if tier == "l1":
        lm._spec_prefix_cache = None
    before = _matched(m)
    pos.take()
    hits.clear()
    warm = _run(model, m, [_request(model, turn2, [_img(1), _img(2)])],
                head=head)
    warm_calls = pos.take()
    if tier == "l0":
        assert hits == [len(turn1)]
    else:
        assert hits == [0] and _matched(m) - before == len(turn1)
    lm._spec_prefix_cache = None
    cold = _run(model, _manager(),
                [_request(model, turn2, [_img(1), _img(2)])], head=head)
    cold_calls = pos.take()
    assert warm == cold

    # Turn 2 prefills only its new tokens, image B included, with the
    # hidden states and mrope position ids of the same tokens run cold.
    warm_pre, warm_rest = _leading(warm_calls, len(turn2) - len(turn1))
    cold_pre, cold_rest = _leading(cold_calls, len(turn2))
    hidden_w = np.concatenate([h for *_x, h in warm_pre], axis=1)
    hidden_c = np.concatenate([h for *_x, h in cold_pre], axis=1)
    assert np.abs(hidden_w - hidden_c[:, len(turn1):]).max() < _ATOL
    pos_w = np.concatenate([p for _n, p, _d, _h in warm_pre], axis=-1)
    pos_c = np.concatenate([p for _n, p, _d, _h in cold_pre], axis=-1)
    assert np.array_equal(pos_w, pos_c[..., len(turn1):])
    # The verify forwards after it see the same positions and deltas.
    assert [c[0] for c in warm_rest] == [c[0] for c in cold_rest]
    assert len(warm_rest) >= 3
    for (_n, pw, dw, hw), (_m, pc, dc, hc) in zip(warm_rest, cold_rest):
        assert np.array_equal(pw, pc) and np.array_equal(dw, dc)
        assert np.abs(hw - hc).max() < _ATOL
