"""Which callers reach the per-layer-input slice of the Gemma 4 text model.

``Gemma4TextModel.__call__`` slices whole-prompt ``per_layer_inputs`` at
``int(c.offset)`` when they are longer than the chunk. On a batch cache
the offset is an array, and ``int()`` raises at B>1 and syncs at B=1.
This pins every caller that passes ``per_layer_inputs`` on a tiny model
with per-layer inputs:

- the container's ``__call__`` computes them from the same ids as the
  embeddings, so they are never longer than the chunk;
- the batch engine's prompt step, gmlx's MTP prompt step and the media
  span step slice them to the chunk with the other sequence-aligned
  prompt keywords;
- the single-sequence ``generate_step`` passes the whole prompt's with
  every chunk and reaches the slice, on a cache with int offsets.

So no caller reaches the slice with a batch cache. The last test shows
what the slice does with one.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from gmlx._vendor.mlx_vlm.generate import ar
from gmlx._vendor.mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig

from gmlx.models.gemma4 import gemma4 as g4
from gmlx.models.gemma4 import language as lang

from test_gemma4_image_audio import serve_env  # noqa: F401

_TEXT = dict(
    model_type="gemma4_text", hidden_size=64, num_hidden_layers=4,
    intermediate_size=128, num_attention_heads=4, head_dim=16,
    global_head_dim=16, num_key_value_heads=2, num_kv_shared_layers=2,
    hidden_size_per_layer_input=8, vocab_size=128,
    vocab_size_per_layer_input=128, sliding_window=64,
    sliding_window_pattern=2,
)
_ROWS = [list(range(1, 21)), list(range(30, 44))]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    m = g4.Model(ModelConfig(text_config=TextConfig(**_TEXT),
                             vision_config=VisionConfig()))
    mx.eval(m.language_model.parameters())
    return m


@pytest.fixture
def seen(monkeypatch):
    """(per_layer_inputs longer than the chunk, the cache offset's kind)
    for each text model call."""
    calls = []
    orig = lang.Gemma4TextModel.__call__

    def spy(self, inputs=None, inputs_embeds=None, mask=None, cache=None,
            per_layer_inputs=None, **kw):
        h = inputs_embeds if inputs_embeds is not None else inputs
        off = next((c.offset for c in (cache or [])
                    if c is not None and hasattr(c, "offset")), None)
        longer = (per_layer_inputs is not None
                  and per_layer_inputs.shape[1] != h.shape[1])
        calls.append((longer, type(off).__name__))
        return orig(self, inputs, inputs_embeds=inputs_embeds, mask=mask,
                    cache=cache, per_layer_inputs=per_layer_inputs, **kw)

    monkeypatch.setattr(lang.Gemma4TextModel, "__call__", spy)
    return calls


def _padded(model):
    n = max(len(r) for r in _ROWS)
    ids = mx.array([[0] * (n - len(r)) + r for r in _ROWS])
    emb = model.get_input_embeddings(input_ids=ids)
    return ids, emb.inputs_embeds, emb.per_layer_inputs


class _Drafter:
    """No teacher-force or seed hooks."""


def _prompt_steps(lm, ids, embeds, pli, mtp):
    kw = dict(model=lm, uids=[0, 1], input_ids=_ROWS, max_tokens=[2, 2],
              inputs_embeds=embeds,
              prompt_kwargs={"per_layer_inputs": pli,
                             "mm_token_type_ids": mx.zeros(ids.shape,
                                                           mx.int32)})
    if mtp:
        kw.update(prefill_step_size=None, draft_model=_Drafter(),
                  draft_kind="mtp")
    else:
        kw.update(prefill_step_size=6)
    batch = ar.PromptProcessingBatch(**kw)
    steps = 0
    while batch.needs_processing() and batch.prompt_step() > 0:
        steps += 1
    return steps


def test_the_container_passes_chunk_width_inputs(model, seen):
    ids, _e, _p = _padded(model)
    cache = ar._make_cache(model.language_model, [0, 6])
    mx.eval(model(ids, cache=cache).logits)
    assert seen and not any(longer for longer, _ in seen)
    assert {kind for _, kind in seen} == {"array"}


@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
def test_the_prompt_steps_slice_to_the_chunk(model, seen, serve_env, mtp):  # noqa: F811
    ids, embeds, pli = _padded(model)
    steps = _prompt_steps(model.language_model, ids, embeds, pli, mtp)
    assert steps >= (1 if mtp else 3)
    assert seen and not any(longer for longer, _ in seen)
    assert {kind for _, kind in seen} == {"array"}


def test_generate_step_reaches_the_slice_with_int_offsets(model, seen):
    ids = mx.array([_ROWS[0]])
    step = ar.generate_step(ids, model, None, None, max_tokens=1,
                            prefill_step_size=6)
    next(step)
    reached = [kind for longer, kind in seen if longer]
    assert reached and set(reached) == {"int"}


def test_the_slice_raises_on_a_batch_cache(model):
    ids, embeds, pli = _padded(model)
    lm = model.language_model
    cache = ar._make_cache(lm, [0, 6])
    mx.eval(lm(None, inputs_embeds=embeds[:, :4], cache=cache,
               per_layer_inputs=pli[:, :4]).logits)
    with pytest.raises(ValueError, match="length-1"):
        lm(None, inputs_embeds=embeds[:, 4:8], cache=cache,
           per_layer_inputs=pli)
