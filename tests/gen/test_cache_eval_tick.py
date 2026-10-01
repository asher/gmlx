"""The serve engine's periodic cache tick evaluates the cache states
asynchronously. The decode step queued just before it writes those states,
so a blocking eval would wait for that step and leave the GPU idle while
the next one is built."""

import importlib
from types import SimpleNamespace

import mlx.core as mx

from gmlx._vendor.mlx_lm.models import llama
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.models.vlm_text_only import TextOnlyModel

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")


class _Spy:
    """Stands in for mlx.core in the engine module and records what each
    eval call received."""

    def __init__(self, calls):
        self._calls = calls

    def __getattr__(self, name):
        return getattr(mx, name)

    def eval(self, *args):
        self._calls.append(("eval", args))
        return mx.eval(*args)

    def async_eval(self, *args):
        self._calls.append(("async_eval", args))
        return mx.async_eval(*args)


def test_the_cache_tick_does_not_wait_for_the_queued_step(monkeypatch):
    monkeypatch.setenv("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "2")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    mx.random.seed(0)
    model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=64)))
    mx.eval(model.parameters())

    states = []
    stock = ar.GenerationBatch.cache_states

    def cache_states(self):
        states.append(stock(self))
        return states[-1]

    calls = []
    monkeypatch.setattr(ar.GenerationBatch, "cache_states", cache_states)
    monkeypatch.setattr(ar, "mx", _Spy(calls))
    proc = SimpleNamespace(tokenizer=SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    gen = ar.BatchGenerator(model.language_model, proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            max_tokens=9)
    prompts = [[3, 9, 4, 7, 1], [5, 2, 8]]
    lm = model.language_model
    gen.insert(prompts, 9, prompt_kwargs=[
        {"inputs_embeds": lm.input_embeds(mx.array([p]))} for p in prompts])
    while gen.has_work:
        gen.next()
    gen.close()

    ticks = [[name for name, args in calls if args and args[0] is s]
             for s in states]
    ticks = [t for t in ticks if t]
    assert len(ticks) >= 3
    assert all(t == ["async_eval"] for t in ticks), ticks
