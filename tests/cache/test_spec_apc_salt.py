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
from gmlx._vendor.mlx_vlm.apc import APCManager, hash_image_payload
from gmlx._vendor.mlx_vlm.models.cache import BatchKVCache

import gmlx.lora_rows as lora_rows
import gmlx.spec.ckpt as ckpt
import gmlx.spec.engine as se
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.media_salt import MediaSalt
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
    from gmlx._vendor.mlx_vlm.generate import ar

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


def test_l0_restores_only_where_the_media_guard_allows(ar):
    """A turn that adds an image finds the last turn's entry under its
    per-prefix salt. On a model whose guard wants every image inside the
    restored prefix, L0 refuses it as the L1 tiers do, and a later turn
    whose suffix is text restores."""
    model = _Model(config=SimpleNamespace(image_token_id=_IMG))
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    turn2 = _TURN2 + [_IMG, _IMG, 6]
    turn3 = turn2 + [7, 8]
    spans = [(1, 11), (8, 22)]

    def meta(ids):
        return [{"full_input_ids": list(ids), "prefix_len": 0,
                 "extra_hash": MediaSalt(0, spans[:1 + (len(ids) > 8)]),
                 "apc_blocks": [], "checkpoint_len": 0}]

    assert _prefill(ar, model, _TURN2, meta=meta(_TURN2)) == 0
    assert _prefill(ar, model, turn2, meta=meta(turn2)) == 0
    assert _prefill(ar, model, turn3, meta=meta(turn3)) == len(turn2)


def _gemma4_spec_config(model_type):
    """The config of the language model an owned gemma4 container holds,
    the spec target of the VLMxMTP loader, from the real container config.
    It names no media ids."""
    import importlib

    from gmlx._vendor.mlx_vlm.utils import update_module_configs

    from gmlx.models.gemma4.language import LanguageModel

    module = importlib.import_module(f"gmlx._vendor.mlx_vlm.models.{model_type}")
    text = ("gemma4_unified_text" if model_type == "gemma4_unified"
            else "gemma4_text")
    raw = {"model_type": model_type,
           "text_config": {"model_type": text, "hidden_size": 16,
                           "num_hidden_layers": 2, "intermediate_size": 32,
                           "num_attention_heads": 2, "head_dim": 8,
                           "global_head_dim": 8, "num_key_value_heads": 1,
                           "num_kv_shared_layers": 0,
                           "hidden_size_per_layer_input": 0,
                           "sliding_window": 16, "sliding_window_pattern": 2},
           "vision_config": {}}
    config = update_module_configs(
        module.ModelConfig.from_dict(raw), module, raw,
        ["text", "vision", "projector", "audio"])
    return (LanguageModel(config.text_config).config,
            int(config.image_token_id))


def _muse_glimmer_config():
    """The muse_glimmer language model's config: the text config, which
    names no image id."""
    from gmlx.models.muse_glimmer.vlm_model import ModelConfig, TextConfig

    config = ModelConfig(text_config=TextConfig(vocab_size=32),
                         image_token_id=_IMG)
    return config.text_config, config.image_token_id


_IMAGE_KWARGS = {
    # What the server forwards for an image turn once it strips the pixels.
    "gemma4": ("mm_token_type_ids",),
    "gemma4_unified": ("mm_token_type_ids", "image_position_ids"),
    "muse_glimmer": ("image_sizes",),
}


@pytest.mark.parametrize("arch,keys", [
    ("gemma4", _IMAGE_KWARGS["gemma4"]),
    ("gemma4_unified", _IMAGE_KWARGS["gemma4_unified"]),
    ("gemma4_unified", ("image_position_ids",)),
    ("muse_glimmer", _IMAGE_KWARGS["muse_glimmer"]),
])
def test_l0_skips_image_turns_whose_config_names_no_image_id(ar, arch, keys):
    """Without a cold meta an image turn skips L0 on these archs too: the
    language model's config names no image id, so the forwarded kwargs
    mark the turn. A gemma4 text turn carries token types too, all zero,
    and keeps its reuse."""
    from gmlx._vendor.mlx_vlm import apc

    if arch == "muse_glimmer":
        config, img = _muse_glimmer_config()
    else:
        config, img = _gemma4_spec_config(arch)
    assert img not in apc.multimodal_token_ids_from_config(config)
    model = _Model(config=config)
    cache = SpecPrefixCache(min_prefix=2)
    model._spec_prefix_cache = cache

    def kwargs(ids):
        out = {}
        if "mm_token_type_ids" in keys:
            out["mm_token_type_ids"] = mx.array(
                [[1 if t == img else 0 for t in ids]])
        if img in ids:
            if "image_position_ids" in keys:
                out["image_position_ids"] = mx.zeros((1, 4, 2), mx.int32)
            if "image_sizes" in keys:
                out["image_sizes"] = mx.array([[28, 28]])
        return out

    turn1 = [1, img, img, img, 2, 3]
    turn2 = turn1 + [4, 5]
    assert _prefill(ar, model, turn1, kwargs=kwargs(turn1)) == 0
    assert _prefill(ar, model, turn2, kwargs=kwargs(turn2)) == 0
    assert len(cache) == 0

    text1 = [1, 5, 6, 7, 2, 3]
    text2 = text1 + [4, 5]
    assert _prefill(ar, model, text1, kwargs=kwargs(text1)) == 0
    assert _prefill(ar, model, text2, kwargs=kwargs(text2)) == 6


def test_expanded_media_ids_mark_a_turn_without_the_fold(monkeypatch):
    """DeepSeek-V4 VL expands an image into ids past the vocab and forwards
    only its image layout. Its text config lists those ids as
    media_token_ids, which mark the turn even when the serve fold has not
    taught the stock reader about them."""
    from gmlx._vendor.mlx_vlm import apc

    from gmlx.models.deepseek_v4.model import ModelArgs
    from gmlx.models.deepseek_v4.vlm_model import ModelConfig

    text = ModelArgs(model_type="deepseek_v4", vocab_size=32, hidden_size=8,
                     intermediate_size=8, num_hidden_layers=1,
                     num_attention_heads=1)
    ModelConfig(text_config=text, vocab_size=32)
    monkeypatch.setattr(apc, "multimodal_token_ids_from_config",
                        lambda config: set())
    ids = [1, 2, 32, 33, 33, 34, 36, 3]
    batch = SimpleNamespace(
        model=SimpleNamespace(config=text), _input_ids=mx.array([ids]),
        _prompt_kwargs={"image_meta": mx.array([[2, 2, 0, 0]])})
    assert mtp_prefill._carries_media(batch)
    batch._input_ids = mx.array([[1, 2, 3]])
    batch._prompt_kwargs = {}
    assert not mtp_prefill._carries_media(batch)


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


def _generator_steps(ar, monkeypatch, insert):
    """Stub generators through the real tenant wraps: ``insert`` hands out
    the uids, and ``_next`` runs the prefill it is given, as the stock
    step builds a prompt batch and runs its first prompt step."""
    monkeypatch.setattr(ar.BatchGenerator, "insert", insert)
    monkeypatch.setattr(ar.BatchGenerator, "_next",
                        lambda self, **k: k["run"]())
    mtp_prefill._install_l0_tenant_rows()

    def step(gen, model, ids, uid):
        return ar.BatchGenerator._next(
            gen, run=lambda: _prefill(ar, model, ids, uid=uid))

    return step


def test_l0_keeps_tenants_apart_without_a_manager(ar, monkeypatch):
    """Without an APC manager the server hands each row its tenant as a
    prompt kwarg, and the batch merge drops it. Insert records the tenant
    for each row, and the L0 key carries it."""
    uids = iter([21, 22, 23, 24])
    step = _generator_steps(ar, monkeypatch, lambda self, prompts, *a, **k:
                            [next(uids) for _ in prompts])
    gen = SimpleNamespace()
    ar.BatchGenerator.insert(
        gen, [[1], [1], [1]], [4, 4, 4],
        [{"_apc_tenant": "acme"}, {"_apc_tenant": "globex"},
         {"_apc_tenant": "acme"}])
    ar.BatchGenerator.insert(gen, [[1]], max_tokens=[4], prompt_kwargs=[{}])

    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    text1 = [1, 5, 6, 7, 2, 3]
    assert step(gen, model, text1, 21) == 0
    assert step(gen, model, text1 + [4, 5], 22) == 0
    assert step(gen, model, text1 + [4, 5], 23) == 6
    assert step(gen, model, text1 + [4, 5], 24) == 0


def test_l0_tenant_rows_belong_to_their_generator(ar, monkeypatch):
    """Each generator numbers its rows from 0, so a uid names a row only
    within its generator. A row inserted without a tenant reads none, even
    when another generator's row with the same uid had one."""
    step = _generator_steps(ar, monkeypatch, lambda self, prompts, *a, **k:
                            list(range(len(prompts))))
    acme, bare = SimpleNamespace(), SimpleNamespace()
    ar.BatchGenerator.insert(acme, [[1]], [4], [{"_apc_tenant": "acme"}])
    ar.BatchGenerator.insert(bare, [[1]], [4], [{}])

    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    text1 = [1, 5, 6, 7, 2, 3]
    assert step(acme, model, text1, 0) == 0
    assert step(bare, model, text1 + [4, 5], 0) == 0
    assert step(acme, model, text1 + [4, 5], 0) == 6


def test_l0_skips_a_row_its_generator_no_longer_holds(ar, monkeypatch):
    """The tenant rows are bounded. A row they dropped has no known tenant,
    so L0 neither restores nor stores it."""
    monkeypatch.setattr(mtp_prefill, "_MAX_TENANT_ROWS", 2)
    uids = iter(range(100, 110))
    step = _generator_steps(ar, monkeypatch, lambda self, prompts, *a, **k:
                            [next(uids) for _ in prompts])
    gen = SimpleNamespace()
    ar.BatchGenerator.insert(gen, [[1], [1], [1]], [4, 4, 4], [{}, {}, {}])

    model = _Model()
    model._spec_prefix_cache = SpecPrefixCache(min_prefix=2)
    text1 = [1, 5, 6, 7, 2, 3]
    assert step(gen, model, text1, 101) == 0
    assert step(gen, model, text1 + [4, 5], 100) == 0
    assert step(gen, model, text1 + [4, 5], 102) == 6


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
