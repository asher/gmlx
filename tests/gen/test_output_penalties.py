"""Presence and frequency penalties count every generated token and never
the prompt, as vLLM applies them, on plain decode, batched rows and the
speculative verify rounds."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from gmlx._vendor.mlx_lm import sample_utils as lm_sampling
from gmlx._vendor.mlx_lm.generate import generate_step, speculative_generate_step
from gmlx._vendor.mlx_vlm import sample_utils as vlm_sampling
from gmlx._vendor.mlx_vlm.generate import ar
from gmlx._vendor.mlx_vlm.server import schemas
from pydantic import ValidationError

from gmlx.spec.row_procs import SpecRowProcessors

from test_copy_fixes import _tiny_llama

V = 64
PROMPT = [7, 8, 9]
P = 1.5

COPIES = pytest.mark.parametrize("sampling", [lm_sampling, vlm_sampling],
                                 ids=["lm-copy", "vlm-copy"])
# The first call's context: mlx-lm's generate_step passes the last prompt
# token, the batch prefill the whole prompt.
CONTEXTS = pytest.mark.parametrize("prompt", [PROMPT[-1:], PROMPT],
                                   ids=["prompt-tail", "prompt"])


def _penalties(logits):
    row = logits.reshape(-1).tolist()
    return {t: -v for t, v in enumerate(row) if v != 0}


def _counted(tokens, frequency):
    out = {}
    for t in tokens:
        out[t] = out.get(t, 0) + P if frequency else P
    return out


def _decode(procs, generated, prompt):
    """Each step's penalties under plain decode: the first call scores the
    first generated token, and each later call adds the token before it."""
    steps = []
    for i in range(len(generated) + 1):
        logits = mx.zeros((1, V))
        for proc in procs:
            logits = proc(mx.array(prompt + generated[:i]), logits)
        steps.append(_penalties(logits))
    return steps


@COPIES
@CONTEXTS
@pytest.mark.parametrize("context_size", [None, 0])
def test_presence_lasts_the_whole_output_and_skips_the_prompt(
        sampling, prompt, context_size):
    gen = list(range(10, 50)) + [8]
    steps = _decode(sampling.make_logits_processors(
        presence_penalty=P, presence_context_size=context_size), gen, prompt)
    assert steps[31][10] == P
    for i, pen in enumerate(steps):
        assert pen == _counted(set(gen[:i]), False)


@COPIES
@CONTEXTS
def test_frequency_counts_each_generated_occurrence(sampling, prompt):
    gen = [10, 8, 10, 11] + list(range(20, 45)) + [10]
    steps = _decode(sampling.make_logits_processors(frequency_penalty=P),
                    gen, prompt)
    assert steps[-1][10] == 3 * P
    for i, pen in enumerate(steps):
        assert pen == _counted(gen[:i], True)


@COPIES
@pytest.mark.parametrize("frequency", [False, True],
                         ids=["presence", "frequency"])
def test_a_context_size_counts_the_last_generated_tokens(sampling, frequency):
    key = "frequency" if frequency else "presence"
    procs = sampling.make_logits_processors(
        **{f"{key}_penalty": P, f"{key}_context_size": 3})
    gen = [10, 11, 10, 12]
    steps = _decode(procs, gen, PROMPT)
    for i, pen in enumerate(steps):
        assert pen == _counted(gen[max(0, i - 3):i], frequency)


def test_a_negative_context_size_is_refused():
    with pytest.raises(ValueError):
        lm_sampling.make_presence_penalty(P, -1)
    with pytest.raises(ValidationError):
        schemas.ChatRequest(model="m", messages=[], presence_penalty=P,
                            presence_context_size=-1)
    req = schemas.ChatRequest(model="m", messages=[], presence_penalty=P,
                              frequency_context_size=0)
    assert req.presence_context_size is None
    assert req.frequency_context_size == 0


@pytest.mark.parametrize("frequency", [False, True],
                         ids=["presence", "frequency"])
def test_a_refused_draft_leaves_the_penalty(frequency):
    procs = lm_sampling.make_logits_processors(
        **{("frequency" if frequency else "presence") + "_penalty": P})
    rows = SpecRowProcessors.from_processors(procs, PROMPT)
    rows.apply(mx.zeros((1, V)), rows.plan([]))
    rows.start(20)

    def verify(drafts):
        out = rows.apply(mx.zeros((len(drafts) + 1, V)), rows.plan(drafts))
        return [_penalties(out[j]) for j in range(len(drafts) + 1)]

    committed = [20]
    drafts = [21, 20, 23]
    assert verify(drafts) == [_counted(committed + drafts[:j], frequency)
                              for j in range(4)]
    # 20 is refused and the target's 30 takes its place; 23 is dropped.
    rows.commit([21, 30])
    committed += [21, 30]
    assert verify([22]) == [_counted(committed, frequency),
                            _counted(committed + [22], frequency)]
    # Every draft accepted, then the bonus token.
    rows.commit([22, 31])
    committed += [22, 31]
    assert verify([]) == [_counted(committed, frequency)]


class _RowLM:
    """Row r of a batch prefers token r, then r + 1, and so on."""

    def __call__(self, x, cache=None, **kw):
        ids = mx.arange(V)
        logits = mx.stack([-((ids - r) % V).astype(mx.float32)
                           for r in range(x.shape[0])])
        return SimpleNamespace(logits=logits[:, None, :])


def test_batched_rows_count_only_their_own_output():
    # Ragged prompts that hold each row's first preferred tokens.
    prompts = [[3, 0, 1, 2], [5, 1]]
    procs = [lm_sampling.make_logits_processors(presence_penalty=100.0)
             for _ in prompts]
    lm = _RowLM()
    # The prefill scores each row's first token with its prompt as context.
    first = lm(mx.zeros((2, 1))).logits[:, -1, :]
    firsts = []
    for r, prompt in enumerate(prompts):
        row = first[r:r + 1]
        for proc in procs[r]:
            row = proc(mx.array(prompt), row)
        firsts.append(int(mx.argmax(row, axis=-1).item()))
    batch = ar.GenerationBatch(
        lm, uids=[0, 1], inputs=mx.array(firsts), prompt_cache=[],
        sampler=lambda lp: mx.argmax(lp, axis=-1), stop_criteria=None,
        max_tokens=[64, 64], token_context=prompts, logits_processors=procs)
    out = [batch._step()[0] for _ in range(30)]
    assert [t[0] for t in out] == list(range(30))
    assert [t[1] for t in out] == list(range(1, 31))


def test_a_drafter_with_a_narrower_vocab_shares_the_penalties():
    # mlx-lm runs the same processors on the drafter's 56-wide logits and
    # the target's 64-wide ones, over a history that holds target tokens
    # the drafter cannot emit.
    target, _ = _tiny_llama(64, seed=0)
    drafter, _ = _tiny_llama(56, seed=1)
    prompt = mx.array([3, 9, 27, 40], dtype=mx.int32)

    def greedy(logprobs):
        return mx.argmax(logprobs, axis=-1)

    def procs():
        return lm_sampling.make_logits_processors(presence_penalty=4.0)

    plain = [int(t) for t, _ in generate_step(
        prompt, target, max_tokens=24, sampler=greedy,
        logits_processors=procs())]
    spec = [int(t) for t, _, _ in speculative_generate_step(
        prompt, target, drafter, num_draft_tokens=3, max_tokens=24,
        sampler=greedy, logits_processors=procs())]
    assert spec == plain
    assert len(set(plain)) == len(plain) and max(plain) >= 56
