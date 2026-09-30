"""Hooks the owned Gemma 4 language model defines for gmlx's engine.

``speculative_argmax_from_hidden`` lets the greedy verify walk take the
argmax of every verify position in one op instead of one host sync per
draft position. Stock gemma4 has only ``speculative_logits_from_hidden``,
and the owned class defines the argmax. The text MTP target builds the
owned language model, and the embeddings it hands the MTP engine carry
the scale of the ids path.
"""

from __future__ import annotations

import mlx.core as mx
from gmlx._vendor.mlx_vlm.models.gemma4 import language as stock

from gmlx.load import mtp_target
from gmlx.models.gemma4 import language as owned

import gemma4_probes as probes


def test_the_owned_argmax_is_the_argmax_of_the_verify_logits():
    lm = probes.tiny(owned)
    hidden = lm(mx.array([probes.PROMPT]), return_hidden=True).hidden_states[-1]
    got = lm.speculative_argmax_from_hidden(hidden)
    want = mx.argmax(lm.speculative_logits_from_hidden(hidden), axis=-1)
    assert got.shape == (1, len(probes.PROMPT))
    assert mx.array_equal(got, want).item()


def test_the_stock_class_has_no_argmax():
    assert not hasattr(stock.LanguageModel, "speculative_argmax_from_hidden")


def test_the_mtp_target_builds_the_owned_language_model():
    """The text MTP target is the owned class, whose hook row requires the
    argmax, and the build adds nothing to the instance."""
    cls, _build = mtp_target._mtp_target_classes("gemma4_text")
    assert cls is owned.LanguageModel
    assert ("speculative_argmax_from_hidden"
            in mtp_target._MTP_TARGET_HOOKS_BY_TYPE["gemma4_text"])
    wrapper, _config = mtp_target._build_mtp_target(dict(probes.BASE))
    lm = wrapper.language_model
    assert type(lm) is owned.LanguageModel
    assert "speculative_argmax_from_hidden" not in vars(lm)


def test_the_mtp_target_embeddings_carry_the_ids_path_scale():
    """The MTP engine embeds on the top-level model and hands the language
    model the embeddings, which its embeddings path does not scale. The
    target scales them as the ids path does, so both give the same
    logits."""
    wrapper, _config = mtp_target._build_mtp_target(dict(probes.BASE))
    lm = wrapper.language_model
    mx.eval(lm.parameters())
    ids = mx.array([probes.PROMPT])
    embeds = wrapper.get_input_embeddings(input_ids=ids).inputs_embeds
    scale = lm.model.embed_scale
    assert scale == probes.BASE["hidden_size"] ** 0.5
    assert mx.array_equal(embeds, lm.model.embed_tokens(ids) * scale).item()
    via_embeds = lm(ids, inputs_embeds=embeds).logits
    assert mx.allclose(via_embeds, lm(ids).logits, rtol=1e-5, atol=1e-5).item()
