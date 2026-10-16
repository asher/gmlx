"""Context copy on targets whose rollback undoes only narrow verifies.

DeepSeek-V4 and glm5_next keep an undo log for verify updates up to
``MTP_UNDO_WIDTH`` positions (the pooling caches and the rotating stash).
A copy round wider than that cannot roll back a long rejected tail, so the
copy is clamped to the target's ``spec_rollback_width``. Each test proposes
copies of 8 with a wrong token at 1 (a long rejected tail on every copy
round) and checks the greedy chain and the verify widths.
"""

import mlx.core as mx
import pytest

import gmlx.spec.speculative as spec
from gmlx.cache.kv_cache import MTP_UNDO_WIDTH


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_CONTEXT_COPY", "GMLX_CONTEXT_COPY_MAX",
                "MLX_VLM_GGUF_CONTEXT_COPY", "GMLX_ROUND_LOG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(spec, "_STOCH_ACCEPT", False)


class _Proposer:
    """Stands in for PromptCopy: proposes the next 8 tokens of the greedy
    chain with the token at 1 wrong, clamped to the loop's budget."""

    def __init__(self, ref):
        self.ref = ref
        self.seen = []
        self.records = []
        self.rounds = self.copied = self.accepted = 0

    def extend(self, tokens):
        self.seen.extend(int(t) for t in tokens)

    def propose(self, shortest, budget):
        nxt = list(self.ref[len(self.seen):len(self.seen) + 8])
        if len(nxt) > 1:
            nxt[1] = (nxt[1] + 1) % 30
        nxt = nxt[:budget]
        return nxt if len(nxt) >= max(shortest, 1) else []

    def drafted(self, tokens, seconds):
        pass

    def record(self, copied, accepted, seconds=None):
        self.records.append((copied, accepted))
        self.rounds += 1
        self.copied += copied
        self.accepted += accepted


def _arm(monkeypatch, ref):
    prop = _Proposer(ref)
    built = []
    monkeypatch.setenv("GMLX_CONTEXT_COPY", "1")
    monkeypatch.setattr(spec, "PromptCopy",
                        lambda p, **k: built.append(k) or prop)
    widths = []
    orig = spec._mtp_verify_target

    def verify(lm_, ids, *a, **k):
        widths.append(int(ids.shape[1]))
        return orig(lm_, ids, *a, **k)

    monkeypatch.setattr(spec, "_mtp_verify_target", verify)
    return prop, built, widths


def _run(target, lm, drafter, prompt, n, block, prefill_chunk):
    drafter.reset(target)
    return list(spec.stream_speculative(
        target, drafter, prompt, prompt_cache=lm.make_cache(),
        max_tokens=n, sampler=None, draft_block_size=block,
        prefill_chunk=prefill_chunk))


def _check(prop, built, widths):
    assert built and built[0]["longest"] == MTP_UNDO_WIDTH - 1
    assert prop.records, "no copy round ran"
    assert any(a < c - 1 for c, a in prop.records)   # long rejected tails
    assert max(widths) <= MTP_UNDO_WIDTH


@pytest.mark.parametrize("block", [2, 3])
def test_dsv4_mtp_copy_rounds_stay_within_the_undo_width(monkeypatch, block):
    from test_deepseek_v4_mtp import (PREFILL_CHUNK, _build_drafter,
                                      _build_target, _greedy_reference,
                                      _tiny_config)

    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    drafter = _build_drafter(cfg, block_size=block)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)
    n = 24
    ref = _greedy_reference(lm, prompt, n + 8)
    prop, built, widths = _arm(monkeypatch, ref)
    out = _run(target, lm, drafter, prompt, n, block, PREFILL_CHUNK)
    assert out == ref[:n]
    _check(prop, built, widths)


def _dspark_drafter(cfg, lm):
    import gmlx.models.deepseek_v4.model as v4
    from gmlx.models.deepseek_v4.dspark import (DeepseekV4DSparkConfig,
                                                DeepseekV4DSparkDrafter)
    from test_deepseek_v4_mtp import _randomize_zero_params

    args = v4.ModelArgs.from_dict(cfg)
    n = int(args.num_hidden_layers)
    args.compress_ratios = list(args.compress_ratios)[:n] + [0, 0, 0]
    layers = tuple(range(n))
    drafter = DeepseekV4DSparkDrafter(DeepseekV4DSparkConfig(
        text=args, n_stages=3, draft_len=5,
        noise_token_id=int(cfg["vocab_size"]) - 1,
        target_layer_ids=layers, markov_rank=8, block_size=4))
    drafter._confidence_tau = 0.0
    mx.eval(drafter.parameters())
    _randomize_zero_params(drafter)
    lm._dspark_capture = layers
    return drafter


def test_dsv4_dspark_copy_rounds_stay_within_the_undo_width(monkeypatch):
    from test_deepseek_v4_mtp import (PREFILL_CHUNK, _build_target,
                                      _greedy_reference, _tiny_config)

    mx.random.seed(7)
    cfg = _tiny_config()
    target, lm = _build_target(cfg)
    prompt = mx.random.randint(0, cfg["vocab_size"], (1, 41)).astype(mx.int32)
    n = 24
    ref = _greedy_reference(lm, prompt, n + 8)
    drafter = _dspark_drafter(cfg, lm)
    prop, built, widths = _arm(monkeypatch, ref)
    out = _run(target, lm, drafter, prompt, n, 4, PREFILL_CHUNK)
    assert out == ref[:n]
    _check(prop, built, widths)


def test_glm5_next_copy_rounds_stay_within_the_undo_width(monkeypatch):
    from test_glm5_next_mtp import (N_GEN, PREFILL_CHUNK, _build_drafter,
                                    _build_lm, _greedy_reference,
                                    _wrap_target)

    lm = _build_lm(seed=5, scale=0.2)
    target = _wrap_target(lm)
    drafter = _build_drafter(lm.args)
    mx.random.seed(7)
    prompt = mx.random.randint(0, lm.args.vocab_size, (1, 41)).astype(mx.int32)
    ref, gaps = _greedy_reference(lm, prompt, N_GEN + 8)
    prop, built, widths = _arm(monkeypatch, ref)
    out = _run(target, lm, drafter, prompt, N_GEN, 2, PREFILL_CHUNK)
    k = next((i for i, g in enumerate(gaps) if g < 1e-3), N_GEN)
    assert out[:k] == ref[:k]
    _check(prop, built, widths)


def test_targets_declare_their_rollback_width():
    from gmlx.models.deepseek_v4.mtp import DeepseekV4SpecHooks
    from gmlx.models.glm5_next.mtp import Glm5NextSpecHooks

    assert DeepseekV4SpecHooks.spec_rollback_width == MTP_UNDO_WIDTH
    assert Glm5NextSpecHooks.spec_rollback_width == MTP_UNDO_WIDTH
