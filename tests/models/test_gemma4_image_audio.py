"""A gemma-4 prompt holding an image and audio, under bidirectional image
attention.

transformers applies the image overlay whenever the setting is "vision",
with or without audio, and keeps audio causal. The plain and MTP language
models agree on such a prompt, and a served batch holding it, alone or
beside an audio row, chunks with its boundaries at the image edges and
matches one pass.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from gmlx._vendor.mlx_vlm.generate import ar

import gmlx.load.vlm as gvlm
import gmlx.models.gemma4.sync as gemma4_sync
import gmlx.spec.engine as engine
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx.gen import media_spans
from gmlx.load.mtp_target import _vlm_spec_language_model

from test_apc_media import _HIDDEN, _gemma4_config

_N, _IMAGE, _AUDIO = 40, (10, 30), (32, 37)
_BIDIRECTIONAL = [
    ("gemma4", {"use_bidirectional_attention": "vision"}),
    ("gemma4_unified", {}),
]


def _models(model_type, text):
    """The container, its stock language model and the MTP target, on the
    same weights."""
    assert gemma4_sync.install_gemma4_nosync()
    mtp_prefill.install_full_prompt_mtp_prefill()
    model, _ = gvlm.build_vlm_model(_gemma4_config(model_type, **text))
    lm = model.language_model
    mx.eval(lm.parameters())
    _cls, build = _vlm_spec_language_model(model_type)
    target = build(model.config)
    target.load_weights(list(tree_flatten(lm.parameters())))
    mx.eval(target.parameters())
    return model, lm, target


def _prompt(model, image: bool, audio: bool):
    image_id = int(model.config.image_token_id)
    audio_id = int(model.config.audio_token_id)
    ids, types = [], []
    for i in range(_N):
        if image and _IMAGE[0] <= i < _IMAGE[1]:
            ids.append(image_id)
            types.append(1)
        elif audio and _AUDIO[0] <= i < _AUDIO[1]:
            ids.append(audio_id)
            types.append(3)
        else:
            ids.append(1 + i)
            types.append(0)
    return ids, types


def _stamp(model, m, ids, types) -> None:
    """The server embeds each request through the container, and
    gemma4_unified stamps its chunking answer on the language model it
    holds, the MTP target under MTP."""
    decline = getattr(model, "_should_disable_chunked_prefill", None)
    if decline is not None:
        m.no_chunked_prefill = bool(
            decline(mx.array([ids]), mm_token_type_ids=mx.array([types])))


def _one_pass(m, embeds, types) -> mx.array:
    out = m(None, inputs_embeds=embeds, cache=m.make_cache(),
            mm_token_type_ids=types)
    return out.logits[:, -1, :].astype(mx.float32)


class _Drafter:
    """No teacher-force or seed hooks: the MTP step keeps the last chunk."""


def _served(m, rows, embeds, types, mtp: bool):
    """Prompt steps the batch runs, then the last logits of one forward
    over the remaining columns, as generate does."""
    kw = dict(model=m, uids=list(range(len(rows))), input_ids=rows,
              max_tokens=[4] * len(rows), inputs_embeds=embeds,
              prompt_kwargs={"mm_token_type_ids": types})
    if mtp:
        # Serve-shaped: stock nulls the step for a drafter batch.
        kw.update(prefill_step_size=None, draft_model=_Drafter(),
                  draft_kind="mtp")
    else:
        kw.update(prefill_step_size=16)
    batch = ar.PromptProcessingBatch(**kw)
    chunks = []
    while batch.needs_processing():
        n = batch.prompt_step()
        if n <= 0:
            break
        chunks.append(n)
    out = m(None, inputs_embeds=batch._inputs_embeds, cache=batch.prompt_cache,
            mm_token_type_ids=batch._prompt_kwargs["mm_token_type_ids"])
    return chunks, out.logits[:, -1, :].astype(mx.float32)


@pytest.fixture
def serve_env(monkeypatch):
    monkeypatch.setenv("PREFILL_STEP_SIZE", "16")
    monkeypatch.setenv("GMLX_PREFILL_MIN_TAIL", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setattr(engine, "_SPEC_APC_DISABLED", True)
    monkeypatch.setattr(mtp_prefill, "_SPEC_APC_DISABLED", True)
    # Serve's order: the span wrapper goes outside the MTP body.
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert media_spans.install_span_aware_prompt_step()


@pytest.mark.parametrize("model_type,text", _BIDIRECTIONAL)
def test_plain_and_mtp_agree_on_an_image_and_audio_prompt(model_type, text):
    model, lm, target = _models(model_type, text)
    _ids, types = _prompt(model, image=True, audio=True)
    embeds = mx.random.normal((1, _N, _HIDDEN), key=mx.random.key(7)) * 0.5
    t = mx.array([types])
    d = mx.max(mx.abs(_one_pass(lm, embeds, t) - _one_pass(target, embeds, t)))
    assert d.item() <= 1e-4


# At a step of 16 the first chunk ends at the image start (10), the next
# takes the whole 20-token image, and the last runs to the final column.
_IMAGE_CHUNKS = [10, 20, 9]


@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
@pytest.mark.parametrize("model_type,text", _BIDIRECTIONAL)
def test_a_served_image_and_audio_prompt_chunks_at_the_image_edges(
        model_type, text, mtp, serve_env):
    model, lm, target = _models(model_type, text)
    m = target if mtp else lm
    ids, types = _prompt(model, image=True, audio=True)
    _stamp(model, m, ids, types)
    embeds = mx.random.normal((1, _N, _HIDDEN), key=mx.random.key(7)) * 0.5
    t = mx.array([types])
    with mx.stream(mx.cpu):
        chunks, got = _served(m, [ids], embeds, t, mtp)
        ref = _one_pass(m, embeds, t)
    assert chunks == _IMAGE_CHUNKS
    assert mx.max(mx.abs(got - ref)).item() <= 1e-4


@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
@pytest.mark.parametrize("model_type,text", _BIDIRECTIONAL)
def test_an_image_row_beside_an_audio_row_chunks_at_the_image_edges(
        model_type, text, mtp, serve_env):
    """The group's token types hold an image in one row and audio in the
    other, and the audio request embedded last."""
    model, lm, target = _models(model_type, text)
    m = target if mtp else lm
    image_ids, image_types = _prompt(model, image=True, audio=False)
    audio_ids, audio_types = _prompt(model, image=False, audio=True)
    _stamp(model, m, audio_ids, audio_types)
    embeds = mx.random.normal((2, _N, _HIDDEN), key=mx.random.key(7)) * 0.5
    t = mx.array([image_types, audio_types])
    with mx.stream(mx.cpu):
        chunks, got = _served(m, [image_ids, audio_ids], embeds, t, mtp)
        ref = _one_pass(m, embeds, t)
    assert chunks == _IMAGE_CHUNKS
    for row in range(2):
        assert mx.max(mx.abs(got[row] - ref[row])).item() <= 1e-4, row


@pytest.mark.parametrize("image,audio", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_the_unified_container_never_declines_chunking(image, audio):
    """From the token types, and from the ids alone when the request
    carries none."""
    model, _lm, _target = _models("gemma4_unified", {})
    ids, types = _prompt(model, image=image, audio=audio)
    decline = model._should_disable_chunked_prefill
    assert decline(mx.array([ids]),
                   mm_token_type_ids=mx.array([types])) is False
    assert decline(mx.array([ids])) is False


def test_causal_gemma4_keeps_chunking_an_image_and_audio_prompt():
    model, lm, _target = _models("gemma4", {})
    _ids, types = _prompt(model, image=True, audio=True)
    assert lm.chunked_prefill_policy(
        prefill_kwargs={"mm_token_type_ids": mx.array([types])})
