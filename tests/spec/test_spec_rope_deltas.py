"""Per-row mrope deltas through the batched MTP rounds loop.

An image row decodes at offset + its own (negative) delta. The loop owns
the deltas next to positions: it seeds them from the rows' own prefill,
concatenates an injected entry's deltas, slices them with the same keep
array as the cache, and passes them as a kwarg on every target forward. A
prefill that writes the model attribute between rounds must not reach the
batch.

The fake target records the delta each row's forward actually uses: the
kwarg when given, else the attribute, sliced or broadcast to the batch
width as the qwen3.5 language model does.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from gmlx.spec.speculative import _owned_decode_rounds_batch, _width_cap_logged
import gmlx.spec.speculative as spec

from test_mtp_preempt_resume import _ArmableDrafter, _EchoDrafter
from test_mtp_width_cap import _FakeCache

VOCAB = 32
IMG = -300


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GMLX_MTP_WIDTH_CAP", "MLX_VLM_GGUF_SPEC_WIDTH_CAP",
                "GMLX_MTP_RESUME"):
        monkeypatch.delenv(var, raising=False)
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()
    yield
    spec._width_cap_memo = ("", None)
    _width_cap_logged.clear()


def _deltas(*vals):
    return mx.array([[v] for v in vals], dtype=mx.int32)


class _DeltaLM:
    """Echo target that records ``(kind, per-row deltas)`` per forward."""

    def __init__(self):
        self.seen = []
        self._rope_deltas = None

    def _effective(self, B, kw):
        rd = kw.get("rope_deltas")
        if rd is not None:
            self._rope_deltas = rd
        else:
            rd = self._rope_deltas
        if rd is None:
            return [0] * B
        vals = [int(v) for v in rd.reshape(-1).tolist()]
        return vals * B if len(vals) == 1 else vals[:B]

    def __call__(self, x, cache=None, return_hidden=False,
                 return_shared_kv=False, **kw):
        B, S = x.shape
        self.seen.append(("verify" if return_hidden else "plain",
                          self._effective(B, kw)))
        nxt = (x + 1) % VOCAB
        logits = (mx.arange(VOCAB)[None, None, :] == nxt[:, :, None]).astype(
            mx.float32) * 10.0
        if return_hidden:
            return SimpleNamespace(
                logits=logits, hidden_states=[mx.zeros((B, S, 8))],
                shared_kv_states={"full": (mx.zeros((B, 2, S, 4)),
                                           mx.zeros((B, 2, S, 4)))},
                gdn_states=None)
        return SimpleNamespace(logits=logits)


def _inject(model, lm, deltas, *, max_tokens):
    """Queue one prefilled row. Its prefill left the attribute at its own
    width-1 deltas, as the real prefill does."""
    lm._rope_deltas = deltas
    model._generator_injections = [{
        "uids": ["u"],
        "prompt_cache": [_FakeCache(width=1, offset=9)],
        "hidden": mx.zeros((1, 1, 8)),
        "prompt_tokens": mx.zeros((1, 4), dtype=mx.int32),
        "first_tokens": mx.array([20], dtype=mx.int32),
        "first_tokens_list": [20],
        "shared_kv_states": {
            "full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))},
        "max_tokens": [max_tokens],
        "rope_deltas": deltas,
    }]


def _drive(drafter, lm, model, deltas, *, max_tokens, stop_check=None):
    B = int(deltas.shape[0])
    lm._rope_deltas = deltas
    gen = _owned_decode_rounds_batch(
        model, drafter, lm, [_FakeCache(width=B)],
        hidden=mx.zeros((B, 1, 8)),
        b=list(range(1, B + 1)),
        shared_kv={"full": (mx.zeros((B, 2, 4, 4)), mx.zeros((B, 2, 4, 4)))},
        seed_tokens=None,
        emitted=[1] * B,
        max_tokens=max_tokens,
        sampler=None,
        draft_block_size=None,
        stop_check=stop_check,
        rope_deltas=deltas,
    )
    return list(gen)


def _by_width(seen):
    out = {}
    for kind, rows in seen:
        out.setdefault(len(rows), set()).add((kind, tuple(rows)))
    return out


def test_image_row_joins_text_batch():
    """A text row speculating alone admits an image row: the width-2 plain
    stretch and the image row's solo speculation after the text row leaves
    both use the image row's own delta."""
    lm, model = _DeltaLM(), SimpleNamespace()
    # Enough budget left after the text row leaves for the batch to re-arm.
    _inject(model, lm, _deltas(IMG), max_tokens=spec._RESUME_MIN_REMAINING + 8)
    _drive(_ArmableDrafter(cap=1), lm, model, _deltas(0), max_tokens=3)
    widths = _by_width(lm.seen)
    assert widths[2] == {("plain", (0, IMG))}
    assert ("verify", (IMG,)) in widths[1]
    assert all(rows in ((0,), (IMG,)) for _, rows in widths[1])
    # the solo stretch after the text row leaves is the image row's
    assert [rows for _, rows in lm.seen[-3:]] == [[IMG]] * 3


def test_text_row_joins_image_batch():
    lm, model = _DeltaLM(), SimpleNamespace()
    _inject(model, lm, _deltas(0), max_tokens=12)
    _drive(_ArmableDrafter(cap=1), lm, model, _deltas(IMG), max_tokens=3)
    widths = _by_width(lm.seen)
    assert widths[2] == {("plain", (IMG, 0))}
    assert [rows for _, rows in lm.seen[-3:]] == [[0]] * 3


def test_finished_row_zero_leaves_its_neighbors_deltas():
    """Row 0 of three finishes first. The two survivors keep their own
    deltas, not the first two of the old batch, and a prefill that writes
    the attribute between rounds does not reach them."""
    lm, model = _DeltaLM(), SimpleNamespace()

    def stop(orig, tok):
        lm._rope_deltas = _deltas(123)      # a concurrent prefill
        return orig == 0

    _drive(_ArmableDrafter(cap=1), lm, model, _deltas(-5, -7, -9),
           max_tokens=4, stop_check=stop)
    widths = _by_width(lm.seen)
    assert widths[3] == {("plain", (-5, -7, -9))}
    assert widths[2] == {("plain", (-7, -9))}


class _ClobberDrafter(_EchoDrafter):
    """Writes the target's attribute before each draft, as another
    request's prefill between rounds does."""

    def __init__(self, lm, **kw):
        super().__init__(**kw)
        self._lm = lm

    def draft_block(self, b, hidden, kv, n, sampler, dtype, **kw):
        self._lm._rope_deltas = _deltas(0)
        return super().draft_block(b, hidden, kv, n, sampler, dtype, **kw)


def test_scalar_loop_keeps_its_row_delta():
    """A single image request decoding on the B=1 serve loop keeps its own
    delta while another request prefills between its rounds."""
    lm = _DeltaLM()
    lm._rope_deltas = _deltas(IMG)
    shared = {"full": (mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))}
    gen = spec._owned_decode_rounds(
        SimpleNamespace(), _ClobberDrafter(lm, cap=0), lm,
        [_FakeCache(width=1)], hidden=mx.zeros((1, 4, 8)), b=5,
        shared_kv=shared, seed_tokens=None, emitted=1, max_tokens=9,
        sampler=None, draft_block_size=None, rope_deltas=_deltas(IMG))
    assert len(list(gen)) == 8
    assert lm.seen and all(rows == [IMG] for _, rows in lm.seen)


@pytest.mark.parametrize("B, inner", [(1, "owned_server_rounds"),
                                      (2, "owned_server_rounds_batch")])
def test_spec_batch_hands_its_deltas_to_the_rounds(monkeypatch, B, inner):
    """The spec batch passes the deltas its prefill captured through the
    engine wrapper into both serve loops."""
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.spec.engine import install_owned_spec_engine

    got = []

    def fake_inner(model, drafter, prompt_cache, hidden, **kw):
        got.append(kw.get("rope_deltas"))
        yield [11] * B, None

    def stock(*a, **kw):  # unflagged stand-in so install re-wraps
        yield [0] * B, None

    monkeypatch.setattr(spec, inner, fake_inner)
    monkeypatch.setattr(ar, "run_speculative_server_rounds", stock)
    install_owned_spec_engine()
    batch = ar.SpeculativeGenerationBatch(
        model=object(), draft_model=_ArmableDrafter(), draft_kind="mtp",
        uids=list(range(B)), first_tokens=mx.array([5] * B),
        prompt_cache=[_FakeCache(width=B)], sampler=None,
        stop_criteria=lambda tok: False, max_tokens=[4] * B,
        hidden=mx.zeros((B, 1, 8)), shared_kv_states=None,
        prompt_tokens=mx.zeros((B, 3), dtype=mx.int32),
        greedy_sampling=True)
    batch._rope_deltas = _deltas(*([IMG] * B))
    batch.next()                    # first tokens
    batch.next()                    # starts the rounds
    assert len(got) == 1 and got[0] is batch._rope_deltas


def test_injection_entry_and_prestart_compaction_carry_deltas(monkeypatch):
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.spec.admission import install_continuous_batch_admission

    install_continuous_batch_admission()

    class _Cache(_FakeCache):
        def filter(self, keep):
            self.width = int(keep.size)

    def make(B, deltas):
        b = ar.SpeculativeGenerationBatch(
            model=SimpleNamespace(), draft_model=_ArmableDrafter(),
            draft_kind="mtp", uids=list(range(10 * B, 10 * B + B)),
            first_tokens=mx.array([5] * B),
            prompt_cache=[_Cache(width=B)], sampler=None,
            stop_criteria=lambda tok: False, max_tokens=[4] * B,
            hidden=mx.zeros((B, 1, 8)), shared_kv_states=None,
            prompt_tokens=mx.zeros((B, 3), dtype=mx.int32),
            greedy_sampling=True)
        b._rope_deltas = deltas
        return b

    host = make(3, _deltas(-5, -7, -9))
    host.filter([1, 2])             # pre-start cancel of row 0
    assert host._rope_deltas.reshape(-1).tolist() == [-7, -9]

    monkeypatch.setattr(ar, "run_speculative_server_rounds",
                        lambda *a, **kw: iter([([None, None, None], None)]))
    host.next()                     # first tokens
    host.extend(make(1, _deltas(IMG)))
    host.next()                     # drains the pending row into the queue
    entry = host.model._generator_injections[0]
    assert entry["rope_deltas"].reshape(-1).tolist() == [IMG]
