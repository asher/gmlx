"""The checkpoint tier never cuts a prompt where the APC media guard
refuses a prefix.

A guard that sees media ids refuses a prefix below the end of the last
media span, and a record there is one no lookup can restore. The guard of
a gemma-4 text config under bidirectional image attention reads its ids
as whole blocks and refuses only a prefix that ends inside an image: the
prefill keeps each image in one chunk, so a cut before or after it leaves
the logits unchanged. Guards that see no ids keep their full schedule.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm import apc as _apc
from mlx_vlm.apc import APCManager
from mlx_vlm.generate import ar

import gmlx.cache.retire_key as retire_key
import gmlx.load.vlm as gvlm
import gmlx.spec.ckpt as ckpt
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.cache.apc_media import install_media_token_ids
from gmlx.cache.snapshot import ckpt_lookup, ckpt_store

from test_apc_media import _HIDDEN, _gemma4_config, _guard, _model
from test_apc_schedule_invariant import _make_from_tags


@pytest.mark.parametrize("model_type,text", [
    ("gemma4", {"use_bidirectional_attention": "vision"}),
    ("gemma4_unified", {}),
])
def test_checkpoint_tier_leaves_a_cold_image_prefill_unchanged(
        model_type, text, monkeypatch):
    """A cold image request on a checkpoint-tier model: the guard column
    sits past the image, but its grid point falls inside it. The prefill
    keeps the image in one chunk, and the checkpoint tier must not cut it,
    so the last logits match the same batch served without the prefix
    cache."""
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", False)
    install_media_token_ids()
    mtp_prefill.install_full_prompt_mtp_prefill()
    model, _ = gvlm.build_vlm_model(_gemma4_config(model_type, **text))
    lm = model.language_model
    mx.eval(lm.parameters())
    image = int(model.config.image_token_id)
    n, lo, hi = 40, 10, 30
    ids = [image if lo <= i < hi else 1 + i for i in range(n)]
    types = mx.array([[int(lo <= i < hi) for i in range(n)]])
    embeds = mx.random.normal((1, n, _HIDDEN), key=mx.random.key(0)) * 0.5
    media = _guard(lm)._apc_media_token_ids()
    guard = _apc.adjust_prefix_to_text_suffix_boundary(
        ids, n - 16, media, max_prefix_tokens=n - 1)
    assert guard == hi
    meta = {"full_input_ids": ids, "prefix_len": 0, "extra_hash": 0,
            "apc_blocks": [], "checkpoint_len": guard}

    def served(**apc) -> tuple[mx.array, list]:
        # The remaining columns go through one forward, as generate does.
        batch = ar.PromptProcessingBatch(
            model=lm, uids=[0], input_ids=[ids], max_tokens=[4],
            inputs_embeds=embeds, prompt_kwargs={"mm_token_type_ids": types},
            prefill_step_size=2048, **apc)
        armed = [p for p, _kind in meta.get("ckpt_boundaries") or ()]
        while batch.needs_processing():
            if batch.prompt_step() <= 0:
                break
        out = lm(None, inputs_embeds=batch._inputs_embeds,
                 cache=batch.prompt_cache,
                 mm_token_type_ids=batch._prompt_kwargs["mm_token_type_ids"])
        return out.logits[:, -1, :].astype(mx.float32), armed

    plain, _ = served()
    with_apc, armed = served(
        apc_meta=[meta], apc_mode="exact",
        apc_manager=APCManager(num_blocks=64, block_size=16))
    assert "ckpt_boundaries" in meta
    assert mx.max(mx.abs(with_apc - plain)).item() <= 1e-4, armed
    assert not any(lo < p < hi for p in armed)


# A 9000-token prompt on a sliding-window layout (block grid, no prefill
# step): a sys anchor at 272, interval points at 4096 and 8192, the turn
# boundaries at 8976 and 8990, the terminal at 8992 and the replay at 8999.
_TAGS = ("kv", "rot:64:0")
_N, _LO, _HI = 9000, 300, 5000
_FULL = [(272, "anchor"), (4096, "boundary"), (8192, "boundary"),
         (8976, "boundary"), (8990, "boundary"), (8992, "boundary"),
         (8999, "replay")]


def _schedule(config, ids, monkeypatch) -> list:
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda i: {"stub": True})
    monkeypatch.setattr(retire_key, "prompt_stable_lcp",
                        lambda ctx, i: 8990)
    monkeypatch.setattr(retire_key, "system_prefix_lcp",
                        lambda ctx, i: 280)
    batch = SimpleNamespace(
        prefill_step_size=None,
        model=SimpleNamespace(config=config, _kq_apc_ckpt_layout=_TAGS))
    meta = {"full_input_ids": list(ids)}
    ckpt._ckpt_arm_schedule(batch, meta, len(ids), 0, 16)
    return meta["ckpt_boundaries"]


def _image_prompt(image: int) -> list:
    return [image if _LO <= i < _HI else 1 + i % 900 for i in range(_N)]


@pytest.mark.parametrize("model_type,text", [
    ("gemma4", {}), ("glm5_next", None), ("muse_glimmer", None)])
def test_guards_without_media_ids_keep_the_full_schedule(
        model_type, text, monkeypatch):
    """Causal gemma-4, glm5_next and muse_glimmer guards see no image id,
    so an image prompt keeps every boundary, those inside the image
    included."""
    install_media_token_ids()
    model = _model(model_type, text)
    ids = _image_prompt(int(model.config.image_token_id))
    assert _schedule(None, ids, monkeypatch) == _FULL
    assert _schedule(model.language_model.config, ids, monkeypatch) == _FULL


_IMG = 7000


def test_a_text_prompt_keeps_the_full_schedule_under_a_media_guard(
        monkeypatch):
    install_media_token_ids()
    config = SimpleNamespace(image_token_id=_IMG)
    ids = [1 + i % 900 for i in range(_N)]
    assert _schedule(config, ids, monkeypatch) == _FULL


@pytest.mark.parametrize("image,lcp,want", [
    (_IMG, 280, None), (_IMG, 5200, 5200), (None, 280, 280)])
def test_exact_anchor_stays_at_or_above_the_media_floor(
        image, lcp, want, monkeypatch):
    """The exact-tier anchor pauses the prefill at the sibling divergence,
    so a divergence below the end of the image arms no anchor."""
    install_media_token_ids()
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda i: {"stub": True})
    monkeypatch.setattr(retire_key, "system_prefix_lcp",
                        lambda ctx, i: lcp)
    ids = _image_prompt(image) if image else [1 + i % 900 for i in range(_N)]
    batch = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(image_token_id=_IMG)))
    meta = {"full_input_ids": ids}
    assert ckpt._exact_anchor_boundary(batch, meta, _N - 1, 0) == want


def test_boundaries_below_the_image_are_ones_no_lookup_restores(monkeypatch):
    """A guard that sees the image id drops the boundaries below the end of
    the image and keeps the rest. Each dropped boundary is unreachable: the
    same image leaves media in the suffix, and another image or a text
    prompt salts the lookup differently."""
    install_media_token_ids()
    config = SimpleNamespace(image_token_id=_IMG)
    ids = _image_prompt(_IMG)
    got = _schedule(config, ids, monkeypatch)
    assert got == [e for e in _FULL if e[0] >= _HI]
    dropped = [p for p, _kind in _FULL if p < _HI]
    assert dropped == [272, 4096]

    gen = _guard(SimpleNamespace(config=config))
    turn2 = ids + [1 + i for i in range(40)]
    red, blue = (_apc.semantic_extra_hash(image_hash=_apc.hash_image_payload(
        pixel_values=mx.full((1, 3, 4, 4), v))) for v in (1.0, 2.0))
    text = _apc.semantic_extra_hash()
    assert len({red, blue, text}) == 3
    for p in dropped:
        for prompt in (ids, turn2):
            assert not gen._apc_suffix_is_text_only(prompt, p)
        man = APCManager(num_blocks=1024, block_size=16)
        assert ckpt_store(man, ids[:p], _make_from_tags(_TAGS, p),
                          extra_hash=red) == p
        query = ids[:p] + [901, 902, 903]
        assert ckpt_lookup(man, query, extra_hash=red, layout=_TAGS)[1] == p
        for salt in (blue, text):
            warm, got_p = ckpt_lookup(man, query, extra_hash=salt,
                                      layout=_TAGS)
            assert (warm, got_p) == (None, 0), (p, salt)


# A gemma-4 text config under bidirectional image attention: its guard
# reads the image id as a whole block.
_WHOLE = SimpleNamespace(model_type="gemma4_text", image_token_id=_IMG,
                         use_bidirectional_attention="vision")


@pytest.mark.parametrize("lcp,want", [(280, 280), (4000, None), (5200, 5200)])
def test_exact_anchor_stays_outside_the_image_under_a_whole_block_guard(
        lcp, want, monkeypatch):
    install_media_token_ids()
    monkeypatch.setattr(retire_key, "lookup_render_ctx",
                        lambda i: {"stub": True})
    monkeypatch.setattr(retire_key, "system_prefix_lcp",
                        lambda ctx, i: lcp)
    batch = SimpleNamespace(model=SimpleNamespace(config=_WHOLE))
    meta = {"full_input_ids": _image_prompt(_IMG)}
    assert ckpt._exact_anchor_boundary(batch, meta, _N - 1, 0) == want


def test_a_whole_block_guard_drops_only_the_boundaries_inside_the_image(
        monkeypatch):
    """The boundary at 4096 falls inside the image and goes. The anchor at
    272 sits before it, and a lookup with the same image restores it with
    the image left to the suffix. The APC salt hashes the request's
    images, so a request with another image or none misses it."""
    install_media_token_ids()
    ids = _image_prompt(_IMG)
    got = _schedule(_WHOLE, ids, monkeypatch)
    assert got == [e for e in _FULL if not _LO < e[0] < _HI]
    assert [p for p, _kind in _FULL if _LO < p < _HI] == [4096]

    gen = _guard(SimpleNamespace(config=_WHOLE))
    assert gen._apc_suffix_is_text_only(ids, 272)
    assert not gen._apc_suffix_is_text_only(ids, 4096)
    red, blue = (_apc.semantic_extra_hash(image_hash=_apc.hash_image_payload(
        pixel_values=mx.full((1, 3, 4, 4), v))) for v in (1.0, 2.0))
    man = APCManager(num_blocks=1024, block_size=16)
    assert ckpt_store(man, ids[:272], _make_from_tags(_TAGS, 272),
                      extra_hash=red) == 272
    assert ckpt_lookup(man, ids, extra_hash=red, layout=_TAGS)[1] == 272
    assert ckpt_lookup(man, ids, extra_hash=blue, layout=_TAGS) == (None, 0)
