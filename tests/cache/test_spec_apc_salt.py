"""The owned MTP APC tiers key a prompt by its salt as well as its ids.

Token ids do not identify a prompt's KV: two images of one size give the
same placeholder ids, and a LoRA adapter changes every layer's keys. The
stock cold admission salts each row (`_apc_extra_hash`: the server's image
hash, the tenant, the gmlx LoRA salt) in the row's cold meta, and the
batch-merged prompt kwargs drop that key. L0 (SpecPrefixCache) and the
owned L1 lookup read the salt from the cold meta; a media prompt with no
cold meta skips L0. CPU only.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.apc import APCManager, hash_image_payload
from mlx_vlm.models.cache import BatchKVCache

import gmlx.lora_rows as lora_rows
import gmlx.spec.ckpt as ckpt
import gmlx.spec.engine as se
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.prefix_cache import SpecPrefixCache

from test_ckpt_tier import make_hybrid_cache

_V = 32
_D = 4
_IMG = 9


class _Out:
    def __init__(self, logits, hidden):
        self.logits = logits
        self.hidden_states = [hidden]
        self.shared_kv_states = None


class _Model:
    def __init__(self, config=None):
        self.config = config

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
        b, n = input_ids.shape
        kv = mx.zeros((b, 1, n, _D))
        for c in cache:
            c.update_and_fetch(kv, kv)
        logits = (mx.arange(_V)[None, None, :]
                  == input_ids[..., None]).astype(mx.float32)
        keep = kw.get("logits_to_keep")
        if keep:
            logits = logits[:, -int(keep):]
        return _Out(logits, inputs_embeds)


class _SpecBatch:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def ar(monkeypatch):
    from mlx_vlm.generate import ar

    mtp_prefill.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(se, "_SPEC_APC_DISABLED", False)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", False)
    monkeypatch.setattr(ar, "SpeculativeGenerationBatch", _SpecBatch)
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("PREFILL_STEP_SIZE", "2")
    return ar


def _cold_meta(ids, salt):
    return [{"full_input_ids": list(ids), "prefix_len": 0,
             "extra_hash": int(salt), "apc_blocks": [], "checkpoint_len": 0}]


def _prefill(ar, model, ids, *, uid=0, kwargs=None, meta=None):
    e = mx.array(ids, dtype=mx.float32)
    embeds = mx.broadcast_to(e[None, :, None], (1, len(ids), _D))
    batch = ar.PromptProcessingBatch(
        model, [uid], [ids], [4], embeds, dict(kwargs or {}),
        warm_cache=[BatchKVCache(left_padding=[0])], apc_meta=meta,
        draft_model=object(), draft_kind="mtp")
    while batch.needs_processing():
        batch.prompt_step()
    batch.generate(lambda x: mx.argmax(x, axis=-1), None)
    return int(getattr(batch, "_mtp_apc_prefix_len", 0) or 0)


_TURN1 = [1, _IMG, _IMG, _IMG, 2, 3]
_TURN2 = _TURN1 + [4, 5]


def test_l0_keys_on_the_cold_meta_salt(ar):
    """Another image of the same size does not restore this image's KV;
    the same image still hits."""
    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    red = hash_image_payload(pixel_values=mx.full((1, 3, 4, 4), 1.0))
    blue = hash_image_payload(pixel_values=mx.full((1, 3, 4, 4), 2.0))
    assert red != blue

    assert _prefill(ar, model, _TURN1, meta=_cold_meta(_TURN1, red)) == 0
    assert _prefill(ar, model, _TURN2, meta=_cold_meta(_TURN2, blue)) == 0
    assert _prefill(ar, model, _TURN2, meta=_cold_meta(_TURN2, red)) == 6


def test_l0_skips_media_prompts_without_a_stock_salt(ar):
    """No cold meta means no APC manager and no image hash: a prompt with
    pixel payloads or media tokens neither stores nor restores, while a
    text prompt keeps its L0 reuse."""
    model = _Model(config=SimpleNamespace(image_token_id=_IMG))
    cache = SpecPrefixCache(min_prefix=2)
    model._spec_prefix_cache = cache
    pixels = {"pixel_values": mx.full((1, 3, 4, 4), 1.0)}

    assert _prefill(ar, model, _TURN1, kwargs=pixels) == 0
    assert _prefill(ar, model, _TURN2, kwargs=pixels) == 0
    assert len(cache) == 0

    text1 = [1, 5, 6, 7, 2, 3]
    assert _prefill(ar, model, text1) == 0
    assert _prefill(ar, model, text1 + [4, 5]) == 6


def test_l0_keeps_adapted_and_bare_rows_apart(ar):
    """Without an APC manager lora_rows salts nothing, so the L0 key
    carries the row's adapter scales itself."""
    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    text1 = [1, 5, 6, 7, 2, 3]
    lora_rows.configure("rows", 1)
    try:
        lora_rows.register_uid(1, (1.0,))
        lora_rows.register_uid(2, None)
        lora_rows.register_uid(3, (1.0,))
        assert _prefill(ar, model, text1, uid=1) == 0
        assert _prefill(ar, model, text1 + [4, 5], uid=2) == 0
        assert _prefill(ar, model, text1 + [4, 5], uid=3) == 6
    finally:
        lora_rows.configure("static", 1)


def _l1_batch(model, ids, salt, **extra):
    return SimpleNamespace(
        model=model, _input_ids=mx.array([ids]),
        _inputs_embeds=mx.zeros((1, len(ids), 4)),
        prompt_cache=make_hybrid_cache(len(ids)), _prompt_kwargs={},
        prefill_step_size=32, _prompt_length_aware_keys=(),
        _apc_meta=_cold_meta(ids, salt), **extra)


def test_owned_l1_keys_on_the_cold_meta_salt():
    """The merged batch kwargs lack `_apc_semantic_hash`, so the owned L1
    must not recompute the salt from them: a checkpoint stored under one
    salt restores only for that salt, and the armed meta keeps it."""
    se._bind_l1_view()
    man = APCManager(num_blocks=256, block_size=16)
    model = SimpleNamespace(
        _kq_apc_manager=man, _kq_apc_mode="exact", config=SimpleNamespace(),
        make_cache=lambda: make_hybrid_cache(0))
    shared = list(range(900, 996))

    batch_a = _l1_batch(model, shared + list(range(50, 74)), 111)
    mtp_prefill._mtp_prefill_init(batch_a)
    meta = batch_a._apc_meta[0]
    assert meta["extra_hash"] == 111
    cl = int(meta["checkpoint_len"])
    batch_a.prompt_cache = make_hybrid_cache(cl, seed=1)
    batch_a._row_real_tokens_processed = lambda idx: cl
    ckpt._ckpt_mid_prefill_store(batch_a)
    assert meta["ckpt_last_stored"] == cl == 96

    other = _l1_batch(model, shared + list(range(700, 740)), 222)
    mtp_prefill._mtp_prefill_init(other)
    assert other._mtp_l1_prefix_len == 0

    same = _l1_batch(model, shared + list(range(700, 740)), 111)
    mtp_prefill._mtp_prefill_init(same)
    assert same._mtp_l1_prefix_len == 96
