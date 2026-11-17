"""The engine admits pending requests in arrival order."""

import threading
import types

import pytest

from gmlx._vendor.mlx_vlm.generate import ar

import gmlx.cache.fresh_gate as fg
import gmlx.serve.tick_guard as tg


def _engine(monkeypatch, **kw):
    import mlx.core as mx

    from gmlx._vendor.mlx_lm.models import llama
    from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
    from gmlx.models.vlm_text_only import TextOnlyModel

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    mx.random.seed(0)
    model = TextOnlyModel(llama.Model(llama.ModelArgs(
        model_type="llama", hidden_size=32, num_hidden_layers=2,
        intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
        rms_norm_eps=1e-5, vocab_size=64)))
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    return ar.BatchGenerator(model.language_model, proc,
                             sampler=lambda lp: mx.argmax(lp, axis=-1),
                             prefill_step_size=None, **kw)


def _pending(gen):
    return [s[0] for s in gen._unprocessed_sequences]


def test_pending_keeps_arrival_order(monkeypatch):
    gen = _engine(monkeypatch)
    (long_uid,) = gen.insert([list(range(1, 40))], 4)
    (short_uid,) = gen.insert([[5, 6]], 4)
    many = gen.insert([[7] * 9, [8], [9] * 3], 4)
    assert _pending(gen) == [long_uid, short_uid, *many]


def test_a_requeued_row_stays_at_the_head(monkeypatch):
    gen = _engine(monkeypatch)
    (first,) = gen.insert([list(range(1, 30))], 4)
    st = tg._state(gen)
    tg._harvest_pending(gen, st)
    # The row leaves the queue as an admission would take it.
    gen._unprocessed_sequences.pop(0)
    (second,) = gen.insert([[3, 4]], 4)
    tg._requeue(gen, st, [first])
    (third,) = gen.insert([[5]], 4)
    assert _pending(gen) == [first, second, third]


def _add(gen, ids, max_tokens):
    import mlx.core as mx

    embeds = gen.model.input_embeds(mx.array([ids]))
    (uid,) = gen.insert([ids], max_tokens,
                        prompt_kwargs=[{"inputs_embeds": embeds}])
    return uid


def test_a_long_request_is_not_overtaken_by_shorter_arrivals(monkeypatch):
    """One slot, one row per admission, and a shorter request arrives on
    every tick: the long request is admitted after the rows ahead of it,
    before every row that came after it."""
    gen = _engine(monkeypatch, completion_batch_size=1, prefill_batch_size=1)
    order = [_add(gen, [2, 3, 4], 2)]
    long_uid = _add(gen, list(range(1, 40)), 1)
    order.append(long_uid)
    admitted = []
    for tick in range(24):
        order.append(_add(gen, [5 + tick % 7], 1))
        before = set(_pending(gen))
        gen.next()
        admitted.extend(u for u in order if u in before
                        and u not in _pending(gen))
        if long_uid in admitted:
            break
    assert long_uid in admitted
    assert admitted == order[:len(admitted)]


# The fresh-prefix gate with arrivals out of length order.

class _Manager:
    block_size = 16

    def __init__(self):
        self.lock = threading.RLock()
        self.hash_table = {}
        self._exact_cache = {}
        self._kq_anchor_cache = {}


SHARED = list(range(1000, 1512))


def _fake_next(self, **kw):
    n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
    self.admitted.extend(s[0] for s in self._unprocessed_sequences[:n])
    self._unprocessed_sequences = self._unprocessed_sequences[n:]
    return [], []


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setattr(ar.BatchGenerator, "_next", _fake_next)
    fg.install_fresh_admission_gate()
    return ar.BatchGenerator._next


def test_fresh_gate_cut_follows_arrival_order(monkeypatch, gated):
    """The leader arrives first and is the longest; a short unrelated row
    and a shorter sibling follow. The leader and the unrelated row are
    admitted, and the sibling waits for the leader's stores."""
    gen = _engine(monkeypatch)
    # An earlier test may have installed the serve init, which sets groups
    # of 1; this test needs a group of three.
    gen.prefill_batch_size, gen.completion_batch_size = 4, 8
    gen.apc_manager = _Manager()
    gen.admitted = []
    monkeypatch.setattr(gen, "_apc_extra_hash", lambda kw: 0, raising=False)
    leader, loner, sibling = gen.insert(
        [SHARED + list(range(1, 200)), list(range(5000, 5040)),
         SHARED + [7, 8, 9]], 4)
    assert fg._keep_count(gen) == 2
    gated(gen)
    assert gen.admitted == [leader, loner]
    assert _pending(gen) == [sibling]


def test_fresh_gate_arrival_during_the_tick_lands_behind_the_held_tail(
        monkeypatch):
    """insert() appends to the list the gate hands the stock body, and the
    arrival still goes behind the held rows."""
    gen = _engine(monkeypatch)
    gen.apc_manager = _Manager()
    gen.admitted = []
    monkeypatch.setattr(gen, "_apc_extra_hash", lambda kw: 0, raising=False)
    leader, sibling = gen.insert(
        [SHARED + [1, 2, 3], SHARED + [7, 8, 9]], 4)
    late = []

    def _next_with_arrival(self, **kw):
        late.extend(self.insert([[42]], 4))
        self.admitted.append(self._unprocessed_sequences[0][0])
        self._unprocessed_sequences = self._unprocessed_sequences[1:]
        return [], []

    monkeypatch.setattr(ar.BatchGenerator, "_next", _next_with_arrival)
    fg.install_fresh_admission_gate()
    ar.BatchGenerator._next(gen)
    assert gen.admitted == [leader]
    assert _pending(gen) == [sibling, *late]

