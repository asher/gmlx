"""A requeued row's prompt kwargs, extended for the tokens it delivered."""

import types

import mlx.core as mx
import pytest

from gmlx.serve import replay_prompt as rp
from gmlx.serve.replay_prompt import ReplayUnsupported, replay_prompt_kwargs

D = 4
L = 5


def _embed(ids, **extra):
    """A stub text embedder: one row per id, valued by the id."""
    n = len(ids)
    out = {"inputs_embeds": mx.broadcast_to(
        mx.array(ids, dtype=mx.float32)[None, :, None], (1, n, D))}
    for k, shape in extra.items():
        out[k] = mx.full((1, n) + shape, 7.0)
    return out


def _prompt(**kw):
    out = {"inputs_embeds": mx.zeros((1, L, D))}
    out.update(kw)
    return out


def _lm(fam):
    cls = type("LanguageModel", (), {"__module__":
                                     f"gmlx._vendor.mlx_vlm.models.{fam}.language"})
    return cls()


def test_nothing_delivered_or_no_embeddings_returns_the_same_kwargs():
    kw = _prompt()
    assert replay_prompt_kwargs(kw, L, [], _embed) is kw
    kw = {"x": 1}
    assert replay_prompt_kwargs(kw, L, [3], _embed) is kw
    assert replay_prompt_kwargs({}, L, [3], _embed) == {}


def test_inputs_embeds_grow_by_the_tail():
    kw = _prompt()
    out = replay_prompt_kwargs(kw, L, [3, 4], _embed)
    e = out["inputs_embeds"]
    assert e.shape == (1, L + 2, D)
    assert mx.array_equal(e[:, :L], kw["inputs_embeds"]).item()
    assert e[0, L:, 0].tolist() == [3.0, 4.0]
    assert kw["inputs_embeds"].shape == (1, L, D)     # the prompt's is kept


def test_per_layer_inputs_and_zero_deepstack_grow_by_the_tail():
    kw = _prompt(per_layer_inputs=mx.zeros((1, L, 2, 3)),
                 deepstack_visual_embeds=mx.ones((1, L, 2, D)))
    out = replay_prompt_kwargs(
        kw, L, [3], lambda ids: _embed(ids, per_layer_inputs=(2, 3)))
    assert out["per_layer_inputs"].shape == (1, L + 1, 2, 3)
    assert out["per_layer_inputs"][0, L].tolist() == [[7.0] * 3] * 2
    ds = out["deepstack_visual_embeds"]
    assert ds.shape == (1, L + 1, 2, D)
    assert mx.array_equal(ds[:, L:], mx.zeros((1, 1, 2, D))).item()
    with pytest.raises(ReplayUnsupported, match="per_layer_inputs"):
        replay_prompt_kwargs(kw, L, [3], _embed)


def test_positions_append_after_the_largest_prompt_position():
    mrope = mx.array([[[0, 1, 2, 2, 3]], [[0, 1, 4, 5, 6]], [[0, 1, 2, 3, 3]]])
    flat = mx.array([[0, 1, 2, 3, 9]])
    deltas = mx.array([[2]])
    kw = _prompt(position_ids=mrope, rope_deltas=deltas)
    out = replay_prompt_kwargs(kw, L, [3, 4, 5], _embed)
    assert out["position_ids"][:, 0, L:].tolist() == [[7, 8, 9]] * 3
    assert out["rope_deltas"] is deltas
    out = replay_prompt_kwargs(_prompt(position_ids=flat), L, [3, 4], _embed)
    assert out["position_ids"].tolist() == [[0, 1, 2, 3, 9, 10, 11]]


@pytest.mark.parametrize("n", [1, 5, 17])
def test_the_position_rule_matches_qwen2_vl_over_the_whole_replay(n):
    """Qwen2-VL's own index over a prompt with an image and n appended text
    tokens equals the rule, and the rope delta does not change."""
    from gmlx._vendor.mlx_vlm.models.qwen2_vl.language import LanguageModel

    img, vid, vst = 151655, 151656, 151652
    lm = LanguageModel.__new__(LanguageModel)
    lm.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2),
        image_token_id=img, video_token_id=vid, vision_start_token_id=vst)
    grid = mx.array([[1, 4, 6]])
    prompt = [10, 11, vst] + [img] * 6 + [12, 13, 14]
    tail = list(range(30, 30 + n))
    pos, delta = lm.get_rope_index(mx.array([prompt]), grid, None, None)
    whole, whole_delta = lm.get_rope_index(mx.array([prompt + tail]), grid,
                                           None, None)
    kw = {"inputs_embeds": mx.zeros((1, len(prompt), D)),
          "position_ids": pos, "rope_deltas": delta}
    out = replay_prompt_kwargs(kw, len(prompt), tail, _embed)
    assert mx.array_equal(out["position_ids"], whole).item()
    assert mx.array_equal(out["rope_deltas"], whole_delta).item()


def test_masks_and_token_types_get_text_values():
    kw = _prompt(visual_pos_masks=mx.ones((1, L), dtype=mx.bool_),
                 mm_token_type_ids=mx.ones((1, L), dtype=mx.int32),
                 token_type_ids=mx.ones((1, L), dtype=mx.int32),
                 attention_mask=mx.ones((1, L), dtype=mx.int32))
    out = replay_prompt_kwargs(kw, L, [3, 4], _embed)
    assert out["visual_pos_masks"][0, L:].tolist() == [False, False]
    assert out["mm_token_type_ids"][0, L:].tolist() == [0, 0]
    assert out["token_type_ids"][0, L:].tolist() == [0, 0]
    assert out["attention_mask"][0, L:].tolist() == [1, 1]


def test_aligned_keys_reach_the_replay_length_and_the_rest_are_kept():
    grid = mx.array([[1, 4, 6]])
    hidden = [mx.zeros((3, D))]
    kw = _prompt(position_ids=mx.zeros((3, 1, L), dtype=mx.int32),
                 visual_pos_masks=mx.zeros((1, L), dtype=mx.bool_),
                 attention_mask=mx.ones((1, L)),
                 image_grid_thw=grid, deepstack_visual_embeds=hidden,
                 attention_mask_4d=mx.zeros((1, 1, L, L)),
                 _apc_semantic_hash=12, small=mx.zeros((1, 2)))
    out = replay_prompt_kwargs(kw, L, [3, 4, 5], _embed)
    assert out["inputs_embeds"].shape[1] == L + 3
    assert out["position_ids"].shape[2] == L + 3
    assert out["visual_pos_masks"].shape[1] == L + 3
    assert out["attention_mask"].shape[1] == L + 3
    for k in ("image_grid_thw", "deepstack_visual_embeds",
              "attention_mask_4d", "_apc_semantic_hash", "small"):
        assert out[k] is kw[k]


class _Rows:
    """Stands in for unbuilt prompt rows."""
    shape = (1, L, D)


@pytest.mark.parametrize("kw,lm,match", [
    (_prompt(), _lm("glm4v"), "glm4v keeps"),
    (_prompt(), _lm("minicpmo"), "minicpmo keeps"),
    (_prompt(decoder_inputs_embeds=mx.zeros((1, 2, D))), None,
     "encoder-decoder"),
    (_prompt(cross_attention_mask=mx.zeros((1, 1, L, 4))), None,
     "cross-attention"),
    (_prompt(full_text_row_masked_out_mask=mx.zeros((1, 1, L, 1))), None,
     "cross-attention"),
    ({"inputs_embeds": _Rows()}, None, "unbuilt"),
    (_prompt(pos_hw=mx.zeros((1, L, 2))), None, "pos_hw"),
    (_prompt(something_new=mx.zeros((2, 3, L))), None, "something_new"),
    (_prompt(attention_mask_4d=mx.zeros((1, 1, L, L))), _lm("falcon_ocr"),
     "falcon_ocr"),
], ids=["glm4v", "minicpmo", "decoder", "mllama-cross", "mllama-rows",
        "prompt-rows", "pos_hw", "unknown-aligned", "falcon_ocr"])
def test_rows_with_no_rule_are_refused(kw, lm, match):
    with pytest.raises(ReplayUnsupported, match=match):
        replay_prompt_kwargs(kw, L, [3], _embed, lm)


def test_position_families_replay_rows_that_carry_their_positions():
    kw = _prompt(position_ids=mx.zeros((3, 1, L), dtype=mx.int32))
    out = replay_prompt_kwargs(kw, L, [3], _embed, _lm("glm4v"))
    assert out["position_ids"].shape == (3, 1, L + 1)


def test_a_qwen4_exp_text_row_gets_plain_positions():
    lm = type("LanguageModel", (),
              {"__module__": "gmlx.models.qwen4_exp.vlm_model"})()
    out = replay_prompt_kwargs(_prompt(), L, [3, 4], _embed, lm)
    assert out["position_ids"].tolist() == [[list(range(L + 2))]] * 3
    assert out["rope_deltas"].tolist() == [[0]]
    image = _prompt(position_ids=mx.array([[[0, 1, 1, 1, 2]]] * 3),
                    rope_deltas=mx.array([[-2]]))
    out = replay_prompt_kwargs(image, L, [3], _embed, lm)
    assert out["position_ids"][:, 0, L:].tolist() == [[3]] * 3
    assert out["rope_deltas"] is image["rope_deltas"]


def test_ngram_replays_and_says_so_once(caplog, monkeypatch):
    monkeypatch.setattr(rp, "_NGRAM_WARNED", [False])
    lm = _lm("longcat_flash_ngram")
    for _ in range(2):
        out = replay_prompt_kwargs(_prompt(), L, [3], _embed, lm)
        assert out["inputs_embeds"].shape[1] == L + 1
    assert sum("n-gram" in r.message for r in caplog.records) == 1


class _Feats:
    def __init__(self, **kw):
        self.kw = kw

    def to_dict(self):
        return dict(self.kw)


def test_embed_text_puts_back_the_position_state_and_scopes_the_row(
        monkeypatch):
    from gmlx import lora_rows

    lm = types.SimpleNamespace(_position_ids="P", _rope_deltas="R")
    scopes = []

    class _Scope:
        def __init__(self, tup, rows):
            self.args = (tup, rows)

        def __enter__(self):
            scopes.append(("in", self.args))

        def __exit__(self, *exc):
            scopes.append(("out", self.args))

    monkeypatch.setattr(lora_rows, "row_tuple", lambda rs: ("tuple", rs))
    monkeypatch.setattr(lora_rows, "tuple_scope", _Scope)

    class _Model:
        language_model = lm

        def get_input_embeddings(self, ids, pixel_values):
            assert pixel_values is None
            assert scopes and scopes[-1][0] == "in"
            lm._position_ids = lm._rope_deltas = None
            return _Feats(inputs_embeds=mx.zeros((1, ids.shape[1], D)),
                          position_ids=None)

    out = rp.embed_text(_Model(), [3, 4], "settings")
    assert set(out) == {"inputs_embeds"}
    assert out["inputs_embeds"].shape == (1, 2, D)
    assert (lm._position_ids, lm._rope_deltas) == ("P", "R")
    assert scopes == [("in", (("tuple", "settings"), 1)),
                      ("out", (("tuple", "settings"), 1))]
