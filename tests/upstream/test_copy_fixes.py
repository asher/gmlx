"""In-place fixes to the copy for the names that ruff found undefined or
defined twice. Each test drives the fixed code. The copy lint baseline is
empty, so tests/test_vendor_upstream.py::test_the_copy_has_only_its_baseline_findings
also fails on each unfixed file."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
import typing
from queue import Empty, Queue

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

_EVALUATE = "gmlx._vendor.mlx_lm.evaluate"


def _tiny_llama(vocab_size: int, seed: int = 0):
    from gmlx._vendor.mlx_lm.models import llama

    cfg = dict(model_type="llama", hidden_size=32, num_hidden_layers=2,
               intermediate_size=64, num_attention_heads=4,
               num_key_value_heads=2, rms_norm_eps=1e-5,
               vocab_size=vocab_size, tie_word_embeddings=True,
               max_position_embeddings=256, rope_theta=10000.0)
    mx.random.seed(seed)
    return llama.Model(llama.ModelArgs.from_dict(cfg)), cfg


def _byte_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from gmlx._vendor.mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

    BPEStreamingDetokenizer.make_byte_decoder()
    chars = sorted(BPEStreamingDetokenizer._byte_decoder.items(),
                   key=lambda kv: kv[1])
    tok = Tokenizer(models.BPE(
        vocab={ch: i for i, (ch, _) in enumerate(chars)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=True)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens({"eos_token": "<eos>"})
    return fast


def _checkpoint(path):
    """A two-layer llama over a byte tokenizer, as the copy's load reads it."""
    from mlx.utils import tree_flatten

    tok = _byte_tokenizer()
    model, cfg = _tiny_llama(len(tok))
    path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(path)
    mx.save_safetensors(str(path / "model.safetensors"),
                        dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(cfg))
    return path


def _collect(rqueue: Queue):
    """(prompt, tokens, text) of one request, as the server's generate reads
    its queue, with a timeout so a dead generation thread fails the test."""
    def get():
        try:
            item = rqueue.get(timeout=60)
        except Empty:
            pytest.fail("the server's generation thread sent nothing")
        if isinstance(item, Exception):
            raise item
        return item

    ctx = get()
    tokens, text = [], ""
    while (item := get()) is not None:
        if isinstance(item, tuple):
            continue
        tokens.append(item.token)
        text += item.text
    return ctx.prompt, tokens, text


def test_a_batched_request_on_the_bridge_server_uses_the_batch_tokenizer(
        tmp_path, monkeypatch):
    """Two requests join one batch on the copied mlx-lm server with the GGUF
    bridge installed, and each matches the same request served alone."""
    if mx.default_device() == mx.cpu:
        pytest.skip("the copy's BatchGenerator reads the Metal working set "
                    "from the default device")
    import gmlx.serve.bridge_lm as bridge
    from gmlx._vendor.mlx_lm import server as srv

    monkeypatch.setattr(srv.ModelProvider, bridge._BRIDGE_FLAG, False,
                        raising=False)
    monkeypatch.setattr(srv.ModelProvider, "_load", srv.ModelProvider._load)
    bridge.install_gguf_bridge()

    inserted = []

    class Counting(srv.BatchGenerator):
        def insert_segments(self, segments, **kw):
            inserted.append(segments)
            return super().insert_segments(segments, **kw)

    monkeypatch.setattr(srv, "BatchGenerator", Counting)

    cli = argparse.Namespace(
        model=str(_checkpoint(tmp_path / "tiny")), adapter_path=None,
        draft_model=None, trust_remote_code=False, chat_template="",
        use_default_chat_template=False, pipeline=False,
        decode_concurrency=4, prompt_concurrency=2, prefill_step_size=64,
        prompt_cache_bytes=None, prompt_cache_size=4, chat_template_args={})
    provider = srv.ModelProvider(cli)

    def args(seed):
        return srv.GenerationArguments(
            model=srv.ModelDescription(model="default_model",
                                       draft="default_model", adapter=None),
            sampling=srv.SamplingArguments(
                temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                xtc_probability=0.0, xtc_threshold=0.0),
            logits=srv.LogitsProcessorArguments(
                logit_bias=None, repetition_penalty=0.0,
                repetition_context_size=20, presence_penalty=0.0,
                presence_context_size=20, frequency_penalty=0.0,
                frequency_context_size=20),
            stop_words=[], max_tokens=4, num_draft_tokens=0,
            logprobs=False, top_logprobs=-1, seed=seed,
            chat_template_kwargs=None)

    def request(prompt):
        return srv.CompletionRequest(request_type="text", prompt=prompt,
                                     messages=[], tools=None,
                                     role_mapping=None)

    prompts = ["abcd", "wxyz"]
    rg = srv.ResponseGenerator(provider, srv.LRUPromptCache(4))
    try:
        queues = [Queue() for _ in prompts]
        for q, p in zip(queues, prompts):
            rg.requests.put((q, request(p), args(None)))
        batched = [_collect(q) for q in queues]
        assert provider.is_batchable and len(inserted) == 2
        alone = []
        for p in prompts:
            q = Queue()
            rg.requests.put((q, request(p), args(0)))
            alone.append(_collect(q))
    finally:
        rg.stop_and_join()

    assert batched == alone
    for (prompt, tokens, text), p in zip(batched, prompts):
        assert prompt == provider.tokenizer.encode(p)
        assert tokens and text == provider.tokenizer.decode(tokens)


def _import_evaluate(monkeypatch):
    """The copy's evaluate module over stand-ins for the two lm-eval names it
    imports, removed again after the test."""
    import gmlx._vendor.mlx_lm as pkg

    class LM:
        def __init__(self):
            pass

    stubs = {name: types.ModuleType(name) for name in (
        "lm_eval", "lm_eval.api", "lm_eval.api.model", "lm_eval.api.registry")}
    stubs["lm_eval.api.model"].LM = LM
    stubs["lm_eval.api.registry"].register_model = \
        lambda *names: (lambda cls: cls)
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, _EVALUATE, None)
    monkeypatch.setattr(pkg, "evaluate", None, raising=False)
    del sys.modules[_EVALUATE]
    return importlib.import_module(_EVALUATE)


def test_a_question_whose_prompt_truncates_away_scores_minus_infinity(
        monkeypatch):
    """lm-eval's loglikelihood on the copy's MLXLM: a continuation that pushes
    the whole context past max_tokens scores -inf and not greedy, in its
    request's place, beside a scored one."""
    ev = _import_evaluate(monkeypatch)

    class Tok:
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            return [2 + ord(c) % 60 for c in text]

    model, _ = _tiny_llama(64)
    monkeypatch.setattr(ev, "load", lambda path, tokenizer_config=None:
                        (model, Tok()))
    lm = ev.MLXLM("tiny", max_tokens=4)
    req = [types.SimpleNamespace(args=("x", "y" * 10)),
           types.SimpleNamespace(args=("ab", "c"))]
    (long_score, long_greedy), (score, _greedy) = lm.loglikelihood(req)
    assert long_score == -float("inf") and long_greedy is False
    assert np.isfinite(score) and score < 0


def test_a_later_conv_stack_level_adds_the_resampled_earlier_one():
    from gmlx._vendor.mlx_vlm.models.moge3.config import ConvStackConfig
    from gmlx._vendor.mlx_vlm.models.moge3.conv_stack import ConvStack

    mx.random.seed(0)
    stack = ConvStack(ConvStackConfig(dim_in=[4, 4], dim_res_blocks=[8, 4],
                                      dim_out=[2, 2], resamplers="nearest"))
    fine = mx.random.normal((1, 8, 8, 4))
    a = stack([mx.zeros((1, 4, 4, 4)), fine])
    b = stack([mx.random.normal((1, 4, 4, 4)), fine])
    assert [o.shape for o in a] == [(1, 4, 4, 2), (1, 8, 8, 2)]
    assert not mx.allclose(a[1], b[1]).item()


def test_the_ming_image_encoder_annotation_resolves_to_its_config():
    from gmlx._vendor.mlx_vlm.models.ming_image.config import MingImageConfig
    from gmlx._vendor.mlx_vlm.models.ming_image.text_encoder import (
        MingImageTextEncoder,
    )

    hints = typing.get_type_hints(MingImageTextEncoder.__init__)
    assert hints["config"] is MingImageConfig


def test_the_trainer_exports_the_utils_save_adapter(tmp_path):
    import gmlx._vendor.mlx_vlm.trainer as trainer
    from gmlx._vendor.mlx_vlm.trainer import utils

    assert trainer.save_adapter is utils.save_adapter

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(lora={"rank": 2})
            self.proj = nn.Linear(2, 2)

    model = Tiny()
    out = tmp_path / "run" / "adapters.safetensors"
    trainer.save_adapter(model, out)
    assert json.loads((out.parent / "adapter_config.json").read_text()) == \
        {"rank": 2}
    assert set(mx.load(str(out))) == {"proj.weight", "proj.bias"}


def _leaves(state):
    from gmlx._vendor.mlx_vlm import turboquant as tq

    out = []
    tq._map_state(state, lambda a, ndim: (out.append(np.array(a)), a)[1])
    return out


def test_the_batch_turboquant_cache_zeroes_one_row_tail():
    from gmlx._vendor.mlx_vlm import turboquant as tq

    mx.random.seed(0)
    cache = tq.BatchTurboQuantKVCache(left_padding=[0, 0], bits=4)
    cache.update_and_fetch(mx.random.normal((2, 2, 6, 64)),
                           mx.random.normal((2, 2, 6, 64)))
    before = _leaves(cache.keys) + _leaves(cache.values)
    cache.zero_row_tail(1, 6, 6)
    assert all(np.array_equal(x, y) for x, y in
               zip(before, _leaves(cache.keys) + _leaves(cache.values)))
    cache.zero_row_tail(1, 2, 5)
    after = _leaves(cache.keys) + _leaves(cache.values)
    for old, new in zip(before, after):
        assert not new[1, :, 2:5].any()
        assert np.array_equal(new[0], old[0])
        assert np.array_equal(new[1, :, :2], old[1, :, :2])
        assert np.array_equal(new[1, :, 5:], old[1, :, 5:])
