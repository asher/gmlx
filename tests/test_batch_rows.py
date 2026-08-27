"""batch_rows: decision code must never len() the generation batch.

On the speculative path len() is the promotion mechanism (release +
buffered-injection swap via __dict__.update), so a gate or watcher
calling it can swap the live batch out from under its own decision.
These tests pin the pure read and fail loudly on any new len() call
site in the decision modules.
"""

from pathlib import Path
from types import SimpleNamespace

import gmlx
from gmlx.batch_rows import batch_rows, batch_rows_of

DECISION_MODULES = (
    "admit_gate", "server_memory", "batch_sched", "auto_ratio", "fresh_gate",
)


class _LenRaisesBatch:
    """Stands in for the retrofitted SpecBatch: __len__ is a mutation, so
    a pure reader must never call it."""

    def __init__(self, finished):
        self._finished = list(finished)
        self.prompt_cache = []

    def __len__(self):
        raise AssertionError(
            "len() on the generation batch promotes; decision code must "
            "read batch_rows()")


def _gen(batch, **attrs):
    g = SimpleNamespace(_generation_batch=batch)
    for k, v in attrs.items():
        setattr(g, k, v)
    return g


def test_counts_finished_flags_without_calling_len():
    batch = _LenRaisesBatch([False, True, False])
    assert batch_rows(_gen(batch)) == 2
    assert batch_rows_of(batch) == 2


def test_plain_batch_falls_back_to_len():
    class Plain:
        def __len__(self):
            return 3

    assert batch_rows_of(Plain()) == 3


def test_missing_or_none_batch_reads_zero():
    assert batch_rows(SimpleNamespace()) == 0
    assert batch_rows_of(None) == 0


def test_no_len_on_generation_batch_in_decision_modules():
    # The loud tripwire for new call sites. The engine-side len() calls in
    # spec_engine are the promotion mechanism and are exempt by design.
    root = Path(gmlx.__file__).parent
    offenders = []
    for name in DECISION_MODULES:
        for i, line in enumerate(
                (root / f"{name}.py").read_text().splitlines(), 1):
            if "len(" in line and "_generation_batch" in line:
                offenders.append(f"{name}.py:{i}: {line.strip()}")
    assert not offenders, offenders


def test_should_decline_never_lens_the_batch():
    # Drives the admission decision directly with a raising-len batch:
    # the full predicate (num_to_add, empty check, projection width) must
    # run on pure reads. Rates are absent so the projection returns None
    # and the gate admits, exercising every read on the way.
    from gmlx.admit_gate import _should_decline

    gen = _gen(
        _LenRaisesBatch([False]),
        _unprocessed_sequences=[("uid-1", [1, 2, 3], 64)],
        _prompt_batch=None,
        completion_batch_size=8,
        prefill_batch_size=1,
    )
    assert _should_decline(gen) is False


def test_project_admission_never_lens_the_batch():
    from gmlx.server_memory import project_admission, update_kv_rates

    gen = _gen(_LenRaisesBatch([False, False]))
    update_kv_rates(gen)  # empty prompt_cache: measures nothing, purely
    assert project_admission(gen, [("uid-1", [1, 2, 3], 64)]) is None


def test_keep_count_never_lens_the_batch():
    from gmlx.fresh_gate import _keep_count

    gen = _gen(
        _LenRaisesBatch([False]),
        apc_manager=object(),
        _unprocessed_sequences=[("u1", [1, 2], 8), ("u2", [1, 2], 8)],
        _prompt_batch=None,
        completion_batch_size=8,
        prefill_batch_size=4,
    )
    _keep_count(gen)  # must complete without touching __len__


def test_auto_ratio_c_term_never_lens_the_batch():
    from gmlx import auto_ratio

    gen = _gen(_LenRaisesBatch([False, False]), _kq_last_chunk_time=0.0)
    st = auto_ratio._AutoState()
    wants, c = auto_ratio._c_term(gen, st, now=0.0)
    assert (wants, c) == (False, 0.0)
