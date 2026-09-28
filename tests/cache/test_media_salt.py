"""Per-prefix image salts: a turn that adds an image restores the prefix an
earlier turn stored, and a prefix whose images differ never restores."""

import importlib
import pickle
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm import apc as _apc
from mlx_vlm.utils import StoppingCriteria

import gmlx.lora_rows as lora_rows
import gmlx.load.vlm as gvlm
import gmlx.models.gemma4.sync as gemma4_sync
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache import snapshot
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.cache.apc_media import install_media_token_ids
from gmlx.cache.media_salt import (
    MediaSalt,
    image_spans,
    install_media_salt,
    salt_at,
    salt_windows,
)
from gmlx.cache.prefix_cache import SpecPrefixCache
from gmlx.gen import media_spans

from test_apc_media import _HIDDEN, _gemma4_config
from test_ckpt_tier import KVCache

ar = importlib.import_module("mlx_vlm.generate.ar")
server = importlib.import_module("mlx_vlm.server.generation")

# Warm and cold prefill the suffix in different GEMM shapes, and f32 GEMM
# runs at TF32 on the GPU, so their logprobs differ by about 4e-3 on a
# text-only prompt too. A wrong restore moves them by 0.5 or more.
_ATOL = 1e-2
_V = 128
_BS = 16


# -- the salt ---------------------------------------------------------------

def test_a_prefix_keys_on_the_images_that_start_inside_it():
    s = MediaSalt(7, [(40, 111), (10, 222)])
    assert s.spans == ((10, 222), (40, 111))
    assert s.at(0) == s.at(10) == 7
    assert s.at(11) == s.at(40) == MediaSalt(7, [(10, 222)]).at(11)
    assert s.at(41) == int(s) == s.at(10 ** 6)
    assert len({s.at(10), s.at(11), s.at(41)}) == 3
    # The start and the hash both count.
    assert MediaSalt(7, [(10, 222)]).at(50) != MediaSalt(7, [(11, 222)]).at(50)
    assert MediaSalt(7, [(10, 222)]).at(50) != MediaSalt(7, [(10, 223)]).at(50)


def test_xor_masks_every_prefix_and_keeps_the_type():
    s = MediaSalt(7, [(10, 222)])
    m = s ^ 5
    assert isinstance(m, MediaSalt) and isinstance(5 ^ s, MediaSalt)
    assert int(m) == int(s) ^ 5
    assert m.at(4) == 7 ^ 5 and m.at(20) == s.at(20) ^ 5
    assert (m ^ 5).at(20) == s.at(20)
    back = pickle.loads(pickle.dumps(m))
    assert isinstance(back, MediaSalt) and back.at(4) == m.at(4)
    assert salt_at(99, 4) == 99 and salt_at(m, 4) == m.at(4)


def test_windows_run_longest_first_inside_the_caller_bounds():
    s = MediaSalt(7, [(10, 1), (40, 2)])
    assert salt_windows(s) == [(s.at(41), 40, None), (s.at(11), 10, 40),
                               (s.at(1), 0, 10)]
    assert salt_windows(s, 20, 30) == [(s.at(11), 20, 30)]
    assert salt_windows(s, 0, 0) == salt_windows(s)
    assert salt_windows(99, 3, 8) == [(99, 3, 8)]


def _cfg():
    return SimpleNamespace(image_token_id=9)


def test_image_spans_map_runs_to_images_in_order():
    ids = mx.array([[1, 9, 9, 2, 9, 9, 9, 3]])
    a, b = mx.full((3, 2, 2), 1.0), mx.full((3, 2, 2), 2.0)
    stacked = image_spans(ids, {"pixel_values": mx.stack([a, b])}, _cfg())
    listed = image_spans(ids, {"pixel_values": [a, b]}, _cfg())
    assert [s for s, _h in stacked] == [1, 4]
    assert stacked == listed
    assert stacked[0][1] != stacked[1][1]
    swapped = image_spans(ids, {"pixel_values": mx.stack([b, a])}, _cfg())
    assert swapped[0][1] == stacked[1][1] and swapped[1][1] == stacked[0][1]
    assert image_spans(mx.array([[1, 2, 3]]), {"pixel_values": a[None]},
                       _cfg()) is None
    assert image_spans(ids, {}, _cfg()) is None


def test_image_spans_split_flattened_patches_by_grid():
    ids = mx.array([[1, 9, 9, 2, 9, 3]])
    pv = mx.arange(6 * 4, dtype=mx.float32).reshape(6, 4)
    grid = mx.array([[1, 2, 2], [1, 1, 2]])
    spans = image_spans(ids, {"pixel_values": pv, "image_grid_thw": grid},
                        _cfg())
    assert [s for s, _h in spans] == [1, 4]
    # The grid row is part of an image's hash: the same patches under a
    # different grid tokenize differently.
    other = image_spans(ids, {"pixel_values": pv,
                              "image_grid_thw": mx.array([[1, 1, 4],
                                                          [1, 1, 2]])},
                        _cfg())
    assert other[0][1] != spans[0][1] and other[1][1] == spans[1][1]


def test_unmatched_runs_key_every_image_at_the_first_run():
    ids = mx.array([[1, 9, 2, 9, 3, 9]])
    pv = mx.stack([mx.full((3, 2, 2), 1.0), mx.full((3, 2, 2), 2.0)])
    spans = image_spans(ids, {"pixel_values": pv}, _cfg())
    assert len(spans) == 1 and spans[0][0] == 1


# -- the block and exact tiers ---------------------------------------------

def _kv(n, seed):
    k = mx.random.normal((1, 2, n, 8), key=mx.random.key(seed))
    return [k], [k + 1]


def _turns():
    """Turn 1 holds image A at 40; turn 2 repeats it and adds B at 80."""
    t1 = list(range(1, 65))
    t2 = t1 + list(range(65, 97)) + [120] * 16 + list(range(97, 113))
    a, b = (40, 111), (80, 222)
    return t1, t2, a, b


def test_the_block_walk_restores_the_prefix_before_a_new_image():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m.release(m.store_kv_blocks(t1, *_kv(len(t1), 1),
                                extra_hash=MediaSalt(7, [a])))
    blocks, n = m.lookup_prefix(t2, extra_hash=MediaSalt(7, [a, b]))
    m.release(blocks)
    assert n == len(t1)
    assert (m.stats.hits, m.stats.misses, m.stats.matched_tokens) == (1, 0, 64)
    # Another image at A's column, another tenant salt, another LoRA mask.
    for salt in (MediaSalt(7, [(40, 112), b]), MediaSalt(8, [a, b]),
                 MediaSalt(7, [a, b]) ^ lora_rows.lora_salt((1.0, 0.0))):
        blocks, n = m.lookup_prefix(t2, extra_hash=salt)
        m.release(blocks)
        assert n == 0
    assert (m.stats.hits, m.stats.misses) == (1, 3)


def test_the_block_store_covers_the_prefix_a_warm_turn_skipped():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m.release(m.store_kv_blocks(t1, *_kv(len(t1), 1),
                                extra_hash=MediaSalt(7, [a])))
    s2 = MediaSalt(7, [a, b])
    m.release(m.store_kv_blocks(t2, *_kv(len(t2), 2), extra_hash=s2,
                                skip_first_n_tokens=len(t1)))
    t3 = t2 + [121] * 16 + [5] * 16
    blocks, n = m.lookup_prefix(t3, extra_hash=MediaSalt(7, [a, b, (128, 3)]))
    m.release(blocks)
    assert n == 128


def _row(n, seed):
    c = KVCache()
    c.update_and_fetch(*(x[0] for x in _kv(n, seed)))
    return [c]


def test_the_exact_tier_restores_an_entry_under_fewer_images():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    m._exact_cache_max = 8
    assert m.store_exact_cache(t1, _row(len(t1), 1),
                               extra_hash=MediaSalt(7, [a]))
    got, p = m.lookup_exact_cache(t2, extra_hash=MediaSalt(7, [a, b]))
    assert got is not None and p == len(t1)
    for salt in (MediaSalt(7, [(40, 112), b]), MediaSalt(8, [a, b]),
                 MediaSalt(7, [a, b]) ^ lora_rows.lora_salt((1.0, 0.0))):
        assert m.lookup_exact_cache(t2, extra_hash=salt) == (None, 0)
    # A text entry ending before the first image restores for that prefix.
    m.store_exact_cache(t1[:32], _row(32, 2), extra_hash=7)
    got, p = m.lookup_exact_cache(t2, extra_hash=MediaSalt(7, [(40, 112), b]))
    assert got is not None and p == 32


def test_disk_lookups_run_one_bounded_stock_call_per_window(monkeypatch):
    calls = []

    def stock(self, token_ids, extra_hash=0, max_prefix_tokens=None,
              min_prefix_tokens=0, allow_memory_overlap=False):
        calls.append((extra_hash, min_prefix_tokens, max_prefix_tokens))
        return None, 0

    monkeypatch.setattr(_apc.APCManager, "lookup_prefix_disk_cache", stock)
    m = GmlxAPCManager(num_blocks=8, block_size=_BS)
    s = MediaSalt(7, [(40, 1), (80, 2)])
    assert m.lookup_prefix_disk_cache(list(range(120)), extra_hash=s,
                                      max_prefix_tokens=100,
                                      min_prefix_tokens=16) == (None, 0)
    assert calls == [(s.at(81), 80, 100), (s.at(41), 40, 80),
                     (s.at(1), 16, 40)]


def test_the_spec_prefix_cache_keys_each_entry_at_its_length():
    t1, t2, a, b = _turns()
    cache = SpecPrefixCache(max_entries=4, min_prefix=8)
    hidden = mx.zeros((1, 1, 4))
    cache.store(mx.array(t1), _row(len(t1), 1), hidden,
                salt=(MediaSalt(7, [a]), 0))
    hit = cache.lookup(mx.array(t2), salt=(MediaSalt(7, [a, b]), 0))
    assert hit is not None and hit[0] == len(t1)
    for salt in ((MediaSalt(7, [(40, 112), b]), 0), (MediaSalt(7, [a, b]), 1)):
        assert cache.lookup(mx.array(t2), salt=salt) is None


def test_an_anchor_keys_on_its_own_length():
    t1, t2, a, b = _turns()
    m = GmlxAPCManager(num_blocks=64, block_size=_BS)
    assert snapshot.anchor_exact_store(m, t1, _row(len(t1), 1),
                                       extra_hash=MediaSalt(7, [a]))
    got, p = snapshot.anchor_exact_lookup(m, t2,
                                          extra_hash=MediaSalt(7, [a, b]))
    assert got is not None and p == len(t1)
    assert snapshot.anchor_exact_lookup(
        m, t2, extra_hash=MediaSalt(7, [(40, 112), b])) == (None, 0)


@pytest.fixture
def _static_rows():
    lora_rows.configure("static", 1)
    yield
    lora_rows.configure("static", 1)


def test_a_lora_row_keeps_the_per_prefix_salt(_static_rows):
    lora_rows.install_row_channel()
    lora_rows.ensure_rows(2)
    bg = SimpleNamespace(max_tokens=8, logits_processors=None,
                         _unprocessed_sequences=[], uid_count=100,
                         apc_manager=object(),
                         _apc_extra_hash=lambda kw: 1234)
    args = SimpleNamespace(
        logit_bias=None, repetition_penalty=None, repetition_context_size=20,
        presence_penalty=None, presence_context_size=20,
        frequency_penalty=None, frequency_context_size=20,
        logits_processors=None, enable_thinking=False, _kq_lora=(1.0, 0.0))
    s = MediaSalt(7, [(40, 111)])
    kw = {"_apc_semantic_hash": s}
    server.ResponseGenerator._make_logits_processors(SimpleNamespace(), args,
                                                     None)
    ar.BatchGenerator.insert(bg, [[1, 2]], prompt_kwargs=[kw])
    got = kw["_apc_semantic_hash"]
    mask = lora_rows.lora_salt((1.0, 0.0))
    assert isinstance(got, MediaSalt)
    assert got.at(40) == 7 ^ mask and got.at(41) == s.at(41) ^ mask


# -- the serve path, on a tiny gemma-4 --------------------------------------

@pytest.fixture
def serve(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "8")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    # The loader's installs, then the server's.
    gemma4_sync.install_gemma4_nosync()
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert media_spans.install_span_aware_prompt_step()
    install_media_token_ids()
    assert install_media_salt()


_RIGS = {}


def _rig(bidirectional: bool):
    rig = _RIGS.get(bidirectional)
    if rig is not None:
        return rig
    mx.random.seed(3)
    text = {"use_bidirectional_attention": "vision"} if bidirectional else {}
    model, _ = gvlm.build_vlm_model(_gemma4_config("gemma4", **text))
    lm = model.language_model
    mx.eval(lm.parameters())
    img = int(model.config.image_token_id)
    table = mx.random.normal((_V, _HIDDEN), key=mx.random.key(1)) * 0.5
    runs = mx.random.normal((64, _HIDDEN), key=mx.random.key(2)) * 0.5

    def embed(input_ids, pixel_values=None, mask=None, **_kw):
        ids = input_ids.reshape(-1).tolist()
        emb = table[mx.array([min(t, _V - 1) for t in ids])][None]
        starts = [i for i, t in enumerate(ids)
                  if t == img and (i == 0 or ids[i - 1] != img)]
        for k, a in enumerate(starts):
            b = a
            while b < len(ids) and ids[b] == img:
                b += 1
            # An image's rows depend on its pixels.
            emb[:, a:b] = runs[:b - a] * (1.0 + mx.mean(pixel_values[k]))
        return SimpleNamespace(to_dict=lambda: {"inputs_embeds": emb})

    rg = SimpleNamespace(model=SimpleNamespace(config=model.config,
                                               get_input_embeddings=embed),
                         vision_cache=None)
    rig = _RIGS[bidirectional] = SimpleNamespace(lm=lm, img=img, rg=rg)
    return rig


def _pix(v):
    return mx.full((3, 4, 4), float(v))


def _request(rig, ids, pixels, tenant=None):
    """``_gpu_embed`` as the server calls it, with the salt its request
    handler computes."""
    raw = {"input_ids": mx.array([ids]),
           "mm_token_type_ids": mx.array([[int(t == rig.img) for t in ids]])}
    pv = mx.stack(pixels) if pixels else None
    if pv is not None:
        raw["pixel_values"] = pv
    salt = _apc.semantic_extra_hash(
        tenant=tenant,
        image_hash=_apc.hash_image_payload(pixel_values=pv) if pixels else 0,
        media={"audio": None, "video": None}, model=rig.lm, processor=None)
    _ids, kw = server.ResponseGenerator._gpu_embed(
        rig.rg, raw, None, apc_semantic_hash=salt)
    return ids, kw


def _run(rig, manager, rows):
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(rig.lm, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=4, apc_manager=manager,
                            top_logprobs_k=_V)
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
    return GmlxAPCManager(num_blocks=512, block_size=_BS)


def _matched(manager):
    return int(manager.stats_snapshot().get("matched_tokens", 0) or 0)


def _assert_same(warm, cold):
    assert [t for t, _l in warm] == [t for t, _l in cold]
    for (_t, lw), (_c, lc) in zip(warm, cold):
        assert max(abs(lw[k] - lc[k]) for k in lc) < _ATOL


def _prompts(img):
    """A 96-token first turn with image A at 42, and a second turn that
    repeats it and adds image B at 126."""
    sysp = [1 + i % 50 for i in range(40)]
    turn1 = (sysp + [60, 61] + [img] * 20 + [62, 63, 64, 65]
             + [66 + i % 30 for i in range(30)])
    turn2 = (turn1 + [70, 71, 72] + [73 + i % 20 for i in range(26)]
             + [80] + [img] * 20 + [81, 82] + [83 + i % 10 for i in range(14)])
    return turn1, turn2


@pytest.mark.parametrize("bidirectional", [False, True],
                         ids=["causal", "bidirectional"])
def test_turn_two_restores_the_prefix_before_its_new_image(serve,
                                                           bidirectional):
    rig = _rig(bidirectional)
    turn1, turn2 = _prompts(rig.img)
    m = _manager()
    _run(rig, m, [_request(rig, turn1, [_pix(1)])])
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, turn2, [_pix(1), _pix(2)])])
    assert _matched(m) - before == len(turn1)
    cold = _run(rig, _manager(), [_request(rig, turn2, [_pix(1), _pix(2)])])
    _assert_same(warm[0], cold[0])


@pytest.mark.parametrize("change", ["other_image", "moved", "tenant"])
def test_a_prefix_with_other_images_does_not_restore(serve, change):
    rig = _rig(False)
    turn1, turn2 = _prompts(rig.img)
    m = _manager()
    # Two images, so a swap moves each one to the other's column.
    first = turn1[:62] + [5] * 10 + [rig.img] * 20 + [6] * 4
    _run(rig, m, [_request(rig, first, [_pix(1), _pix(2)])])
    second = first + [7] * 30
    pixels, tenant = [_pix(1), _pix(2)], None
    if change == "other_image":
        pixels = [_pix(3), _pix(2)]
    elif change == "moved":
        pixels = [_pix(2), _pix(1)]
    else:
        tenant = "other"
    before = _matched(m)
    warm = _run(rig, m, [_request(rig, second, pixels, tenant)])
    restored = _matched(m) - before
    # The text before the first image may restore, never more.
    assert restored <= 42
    if tenant is not None:
        assert restored == 0
    cold = _run(rig, _manager(), [_request(rig, second, pixels, tenant)])
    _assert_same(warm[0], cold[0])


def test_a_text_prompt_keeps_the_stock_salt(serve):
    rig = _rig(False)
    turn1, _turn2 = _prompts(rig.img)
    text = [t if t != rig.img else 90 for t in turn1]
    _ids, kw = _request(rig, text, [])
    stock = _apc.semantic_extra_hash.__wrapped__(
        tenant=None, image_hash=0, media={"audio": None, "video": None},
        model=rig.lm, processor=None)
    assert type(kw["_apc_semantic_hash"]) is int
    assert kw["_apc_semantic_hash"] == stock
    # An image prompt keys its text before the first image the same way.
    _ids, kw = _request(rig, turn1, [_pix(1)])
    salt = kw["_apc_semantic_hash"]
    assert isinstance(salt, MediaSalt)
    assert salt.at(42) == stock and salt.at(43) != stock
