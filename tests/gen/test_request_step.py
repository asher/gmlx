"""The request step: the prefill chunk an arch that prices its rows runs,
from the room the resident models leave and what the request holds."""

import gc
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import gmlx.gen.prefill_decay as pd
import gmlx.serve.capacity as capacity

MIB = 2**20
ROW = 1 * MIB


@pytest.fixture(autouse=True)
def _room(monkeypatch):
    monkeypatch.setattr(pd, "_RESIDENT", {})
    monkeypatch.setattr(pd, "_ROOM", [])
    monkeypatch.setitem(pd._ROW_COSTS, "rows_x", lambda cfg: ROW)
    monkeypatch.delenv("GMLX_PREFILL_MIN_STEP", raising=False)
    monkeypatch.delenv("GMLX_PREFILL_DECAY", raising=False)
    monkeypatch.delenv("PREFILL_STEP_SIZE", raising=False)
    monkeypatch.setattr(capacity, "working_set_bytes", lambda: 100 * MIB)
    monkeypatch.setattr(capacity, "ceiling_bytes", lambda ws, m=None: ws)


def _model(model_type="rows_x", **cfg):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type, **cfg))


# The fake models stay alive for the test: a freed one's id can return.
_KEEP: list = []


def _weights(mib):
    m = SimpleNamespace()
    _KEEP.append(m)
    pd.note_resident_model(m, nbytes=mib * MIB)


def _resident(mib):
    _weights(mib)


def test_the_step_is_the_largest_tier_that_fits_the_room_less_held():
    # A working set of 100 MiB less -1000 MiB of weights leaves a room of
    # 1100 MiB, at 1 MiB a row.
    _weights(-1000)
    m = _model()
    assert pd.request_step(m, 4096, 0) == 1024
    assert pd.request_step(m, 2048, 500 * MIB) == 512
    assert pd.request_step(m, 2048, 800 * MIB) == 256


def test_no_step_when_even_the_floor_does_not_fit():
    _weights(-200)  # room 300
    assert pd.request_step(_model(), 2048, 100 * MIB) is None
    assert pd.request_step(_model(), 2048, 44 * MIB) == 256


def test_the_step_passes_through_without_a_row_cost_a_room_or_the_decay(
        monkeypatch):
    assert pd.request_step(_model(), 2048, 0) == 2048          # no room yet
    _resident(99)
    assert pd.request_step(_model("dense_x"), 2048, 0) == 2048
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    assert pd.request_step(_model(), 2048, 0) == 2048


def test_the_room_follows_the_resident_models():
    class Holder(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.w = w

    shared = mx.zeros((MIB // 4,), mx.float32)            # 1 MiB
    a = Holder(shared)
    b = Holder(shared)
    pd.note_resident_model(a)
    assert pd.prefill_room_bytes() == 99 * MIB
    pd.note_resident_model(b)                  # the shared array counts once
    assert pd.prefill_room_bytes() == 99 * MIB
    c = Holder(mx.zeros((MIB // 2,), mx.float32))
    pd.note_resident_model(c)
    assert pd.prefill_room_bytes() == 97 * MIB
    del c
    gc.collect()
    assert pd.prefill_room_bytes() == 99 * MIB


def test_a_load_without_a_drafter_notes_nothing():
    pd.note_resident_model(None)
    pd.note_resident_model("drafter")
    assert pd.prefill_room_bytes() is None


def _batch(model, prompt, gen, base=2048):
    return SimpleNamespace(
        model=model, prefill_step_size=base, max_tokens=[gen],
        _prompt_tokens_per_row=[prompt],
        _inputs_embeds=mx.zeros((1, prompt, 1)), _input_ids=None,
        prompt_cache=[])


def test_a_batch_keeps_its_step_when_the_room_changes():
    _weights(-1000)  # room 1100
    m = _model(hidden_size=0)
    batch = _batch(m, 100, 0)
    assert pd.batch_request_step(batch, 2048) == 1024
    _weights(900)    # room 200
    assert pd.batch_request_step(batch, 2048) == 1024
    assert pd.batch_request_step(_batch(m, 100, 0), 2048) == 256


def test_held_counts_kv_states_rows_and_audio():
    from gmlx.serve.mem_preflight import kv_layer_costs, prompt_kv_bytes

    from tests.models.test_nemotron_h import _tiny_args

    args = _tiny_args()
    m = SimpleNamespace(config=args, audio_held_bytes=lambda n: 7 * n)
    kv = prompt_kv_bytes(kv_layer_costs(m, 2.0), 150)
    assert pd.request_held_bytes(m, 100, 50) == kv
    assert pd.request_held_bytes(m, 100, 50, audio_tokens=10,
                                 embedded_rows=100) == (
        kv + 100 * args.hidden_size * 2 + 70)


def test_nemotron_h_prices_its_rows_by_the_hidden_size():
    from gmlx.models.nemotron_h import model as nh

    from tests.models.test_nemotron_h import _tiny_args

    m = nh.Model(_tiny_args())
    assert pd.row_bytes(m) == nh.ROW_BYTES_PER_HIDDEN * 64
    assert nh.ROW_BYTES_PER_HIDDEN * 2688 == 0.75 * MIB


def _rg(text_tokens):
    tok = SimpleNamespace(encode=lambda s: [0] * text_tokens)
    return SimpleNamespace(tokenizer=tok)


def test_the_memory_check_refuses_a_request_the_floor_cannot_fit(
        monkeypatch):
    from gmlx._vendor.mlx_vlm.server.generation import PromptTooLongError
    from gmlx.serve import mem_preflight as mp

    monkeypatch.setattr(pd, "request_held_bytes",
                        lambda model, p, g, a=0, r=0: (p + g) * MIB)
    _weights(-200)  # room 300
    m = _model()
    args = SimpleNamespace(max_tokens=10)
    assert mp._check_request_fit(_rg(30), m, "x", 0, 0, 0, args)
    with pytest.raises(PromptTooLongError, match="max_tokens"):
        mp._check_request_fit(_rg(100), m, "x", 0, 0, 0, args)
    # What other requests hold leaves less room, and the check refuses.
    with pytest.raises(PromptTooLongError):
        mp._check_request_fit(_rg(30), m, "x", 0, 0, 50 * MIB, args)
    # A model without a row cost takes the stock check.
    assert not mp._check_request_fit(_rg(100), _model("dense_x"), "x", 0, 0,
                                     0, args)


def test_a_text_run_chunks_at_the_request_step(monkeypatch):
    from gmlx._vendor.mlx_lm.generate import generate_step
    from gmlx.models.nemotron_h import model as nh

    from tests.models.test_nemotron_h import _tiny_args

    monkeypatch.setenv("GMLX_PREFILL_MIN_STEP", "16")
    mx.random.seed(0)
    m = nh.Model(_tiny_args())
    mx.eval(m.parameters())
    row = pd.row_bytes(m)
    held = pd.request_held_bytes(m, 300, 1)
    # Room for 64 rows of chunk and no more.
    _weights((100 * MIB - held - 64 * row - 1) / MIB)
    widths = []
    orig = nh.Model.__call__

    def call(self, inputs, cache=None, **kw):
        widths.append(int(inputs.shape[1]))
        return orig(self, inputs, cache=cache, **kw)

    monkeypatch.setattr(nh.Model, "__call__", call)
    prompt = mx.array([i % 40 for i in range(300)])
    for _ in generate_step(prompt, m, max_tokens=1, prefill_step_size=256):
        break
    assert widths[:4] == [64, 64, 64, 64]


def test_an_until_eos_budget_holds_no_output_unless_the_context_bounds_it():
    text = _model()
    assert pd.output_budget(text, 100, 400) == 400
    assert pd.output_budget(text, 100, None) == 0
    assert pd.output_budget(text, 100, 1 << 25) == 0      # run's until-EOS
    assert pd.output_budget(text, 100, 1 << 30) == 0      # serve's default
    speech = SimpleNamespace(config=text.config, context_tokens=1000)
    assert pd.output_budget(speech, 100, 400) == 400
    assert pd.output_budget(speech, 800, 400) == 200
    assert pd.output_budget(speech, 100, 1 << 30) == 900
    assert pd.output_budget(speech, 1200, None) == 0
