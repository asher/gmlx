"""Real weights: an image chat retired mid-reply resumes as it would have
gone on.

MiniCPM-V 4.6 with its mmproj. The row's prompt kwargs carry the image's
embeddings and its positions, as the server builds them. After a few tokens
the row is retired through ``_rebuild_row`` and replays its prompt plus what
it delivered, with APC on. Its resumed tokens must follow the uninterrupted
run; the replay prefills the delivered tokens where the first run decoded
them, so a first difference is allowed only at a near tie.

``integration``; needs ``KQUANT_TEST_GGUF_DIR`` with MiniCPM-V 4.6 and its
mmproj side by side.
"""

from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm")

pytestmark = pytest.mark.integration

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="a real-weights image turn is a GPU workload")

CUT = 12            # tokens delivered before the retire
TOTAL = 40
TIE = 0.5           # nats: a first difference must be this close a call


@pytest.fixture(scope="module")
def minicpmv(gguf_dir):
    from gmlx.config import DiscoverSpec
    from gmlx.load.discovery import scan_dirs
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.vlm import load_vlm_model

    spec = DiscoverSpec(dir=None, recursive=True, pair_mmproj=True,
                        speculative="auto")
    for m in scan_dirs([spec], [str(gguf_dir)]):
        if not m.mmproj:
            continue
        kv = scan_gguf(m.mmproj, include_tensors=False).kv
        if str(kv.get("clip.projector_type", "")) == "minicpmv4_6":
            return load_vlm_model(m.path, m.mmproj)
    pytest.skip("no MiniCPM-V 4.6 GGUF with its mmproj under "
                "KQUANT_TEST_GGUF_DIR")


def _kwargs(model, config, processor):
    """The prompt's ids and kwargs, as the server's _gpu_embed builds them."""
    from PIL import Image

    from gmlx._vendor.mlx_vlm.prompt_utils import apply_chat_template
    from gmlx._vendor.mlx_vlm.server.generation import ResponseGenerator
    from gmlx._vendor.mlx_vlm.utils import (
        prepare_inputs,
        should_add_special_tokens,
    )

    img = Image.new("RGB", (224, 160))
    img.putdata([(x, (2 * y) % 256, 96) for y in range(160) for x in range(224)])
    prompt = apply_chat_template(
        processor, config,
        [{"role": "user", "content": "Describe this image in detail."}],
        num_images=1)
    raw = prepare_inputs(
        processor, images=[img], prompts=prompt,
        image_token_index=getattr(model.config, "image_token_index", None),
        add_special_tokens=should_add_special_tokens(config["model_type"],
                                                     processor))
    server = types.SimpleNamespace(model=model, vision_cache=None)
    ids, kw = ResponseGenerator._gpu_embed(server, raw, None,
                                           apc_semantic_hash=7)
    return ids[0].tolist(), kw


def _run(model, processor, ids, kw, *, retire=False):
    """Greedy tokens with their top-2, and the APC prefix of each pick."""
    import mlx.core as mx

    import gmlx.serve.tick_guard as tg
    from gmlx._vendor.mlx_vlm.apc import APCManager
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.serve.replay_prompt import embed_text
    from gmlx.serve.row_sampling import RowSettings

    gen = ar.BatchGenerator(model.language_model, processor,
                            sampler=lambda lp: mx.argmax(lp, axis=-1),
                            greedy_sampling=True, logprobs_cap=2,
                            apc_manager=APCManager(num_blocks=512,
                                                   block_size=16))
    gen._kq_embed_text = lambda t, rs: embed_text(model, t, rs)
    picks = []
    orig = gen._apc_pick_for

    def pick(seq):
        out = orig(seq)
        picks.append(out["prefix_len"] if out else 0)
        return out

    gen._apc_pick_for = pick
    (uid,) = gen.insert([ids], TOTAL, prompt_kwargs=[kw],
                        row_settings=[RowSettings(logprobs=True, top_k=2)])
    st = tg._state(gen)
    tg._harvest_pending(gen, st)
    out = []
    ticks = 0
    try:
        while gen.has_work:
            ticks += 1
            assert ticks < 4 * TOTAL, "the run stopped making progress"
            resp = gen.next()[1]
            tg._harvest_responses(st, resp)
            out += resp
            if retire and len(out) == CUT:
                retire = False
                # The embedder leaves the language model's position state
                # as it was.
                lm = model.language_model
                held = (lm._position_ids, lm._rope_deltas)
                marks = (object(), object())
                lm._position_ids, lm._rope_deltas = marks
                tg._rebuild_row(gen, st, uid)
                assert (lm._position_ids, lm._rope_deltas) == marks
                lm._position_ids, lm._rope_deltas = held
                assert [e[0] for e in gen._unprocessed_sequences] == [uid]
    finally:
        gen.close()
    return out, picks


@_NEEDS_GPU
def test_a_retired_image_chat_resumes_as_it_would_have_gone_on(minicpmv):
    model, config, processor = minicpmv
    ids, kw = _kwargs(model, config, processor)
    assert "position_ids" in kw
    ref, _ = _run(model, processor, ids, kw)
    assert len(ref) == TOTAL
    got, picks = _run(model, processor, ids, kw, retire=True)
    assert len(got) == TOTAL
    assert picks[-1] >= len(ids) // 2          # the replay hit its prompt
    assert [r.token for r in got[:CUT]] == [r.token for r in ref[:CUT]]
    for i in range(CUT, TOTAL):
        a, b = ref[i], got[i]
        if a.token == b.token:
            assert abs(a.token_logprob - b.token_logprob) < TIE, i
            continue
        # A first difference only where the reference was a near tie.
        top = dict(a.top_logprobs)
        assert b.token in top, (i, a.token, b.token)
        assert top[a.token] - top[b.token] < TIE, (i, top)
        break
