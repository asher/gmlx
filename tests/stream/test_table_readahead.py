"""Table read-ahead: the pages a prefill chunk and the two after it read,
read beside its forward."""

import os
import threading
import types

import mlx.core as mx
import numpy as np

import gmlx.stream.table_readahead as tra
from gmlx.stream import installs


class _RowIsToken:
    """A hasher stand-in: a token reads the row of its own id, with one
    token of history."""

    context_len = 1
    eos_token_id = 0

    def __call__(self, hist):
        return np.asarray(hist)[:, 1:, None]


def test_the_read_ahead_reads_the_cold_pages_its_rows_read(tmp_path,
                                                           monkeypatch):
    page = os.sysconf("SC_PAGE_SIZE")
    path = tmp_path / "table.gguf"
    path.write_bytes(b"\0" * (page * 64))
    offset, row_bytes = 100, page // 2
    read, warm = [], set(range(0, 64, 2))
    monkeypatch.setattr(tra.TableReadAhead, "_resident",
                        lambda self, off, vec: off // page in warm)
    monkeypatch.setattr(tra.TableReadAhead, "_pread",
                        lambda self, off, buf: read.append(off // page))
    ra = tra.TableReadAhead(str(path), offset, 120, row_bytes, _RowIsToken(),
                            workers=3)
    try:
        tokens = [5, 9, 5, 40, 77, 2]
        # Rows 9, 5 and 40 with token 5 before them, then row 5 with none.
        ra.submit([(tokens, 1, 4), (tokens, 0, 1)])
        assert ra.wait(10.0)
    finally:
        ra.close()
    want = set()
    for r in (9, 5, 40):
        start = offset + r * row_bytes
        want |= {start // page, (start + row_bytes - 1) // page}
    assert sorted(read) == sorted(want - warm)
    assert ra.stats["pages"] == len(want)
    assert ra.stats["cold"] == len(want - warm)


def _batch(full, suffix, columns, done, step, right=None):
    return types.SimpleNamespace(
        _apc_meta=[{"full_input_ids": f} for f in full],
        _inputs_embeds=types.SimpleNamespace(shape=(len(full), columns, 8)),
        _input_ids=None, _processed_prompt_columns=done,
        prefill_step_size=step, _right_pad_per_row=right,
        _suffix_lens=suffix)


def _plan(batch, n):
    return [[(s, e) for _, s, e in spans]
            for spans in tra.plan_spans(batch, n)]


def test_each_chunk_plans_its_own_rows_then_two_steps_ahead():
    # A cold 20-token prompt in steps of 4, planned just before each
    # chunk's forward: the chunk itself, then the next two the first time,
    # then one step more each chunk.
    full = list(range(1, 21))
    b = _batch([full], [20], columns=20, done=0, step=4)
    assert _plan(b, 4) == [[(0, 4)], [(4, 12)]]
    b._inputs_embeds.shape, b._processed_prompt_columns = (1, 16, 8), 4
    assert _plan(b, 4) == [[(12, 16)]]
    b._inputs_embeds.shape, b._processed_prompt_columns = (1, 12, 8), 8
    assert _plan(b, 4) == [[(16, 20)]]
    b._inputs_embeds.shape, b._processed_prompt_columns = (1, 8, 8), 12
    assert _plan(b, 4) == []
    # The MTP prefill restored 8 tokens and trimmed its columns before its
    # first chunk: what the cache restored is never read.
    b = _batch([full], [20], columns=12, done=8, step=4)
    assert _plan(b, 4) == [[(8, 12)], [(12, 20)]]
    # A right-padded batch: a warm row with 20 of its 30 tokens restored
    # and a cold row of 16, each from its own next token.
    warm, cold = list(range(1, 31)), list(range(1, 17))
    b = _batch([warm, cold], [10, 16], columns=16, done=0, step=4,
               right=[6, 0])
    assert _plan(b, 4) == [[(20, 24), (0, 4)], [(24, 30), (4, 12)]]


class _Recorder:
    """A read-ahead stand-in that records what it is sent, in ``order``
    beside the forwards."""

    def __init__(self, order=None):
        self.sent, self.closed = [], False
        self.order = [] if order is None else order

    def submit(self, spans):
        self.order.append("submit")
        self.sent.append([(s, e) for _, s, e in spans])

    def close(self):
        self.closed = True


def _served(owner):
    # The text-only adapter a server holds: language_model._model is the
    # model the loader hung its helpers on.
    return types.SimpleNamespace(
        language_model=types.SimpleNamespace(_model=owner))


class _ServedForward:
    """That adapter over a stub forward that records itself in ``order``."""

    def __init__(self, owner, order):
        self.language_model = types.SimpleNamespace(_model=owner)
        self.order = order

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
        self.order.append("forward")
        b, n = input_ids.shape
        kv = mx.zeros((b, 1, n, 4))
        for c in cache:
            c.update_and_fetch(kv, kv)
        return mx.zeros((b, n, 8))


def test_a_prefill_chunk_queues_its_rows_before_its_forward(monkeypatch):
    # The table gather reads the chunk's rows inside its forward, so the
    # chunk's own rows go to the read-ahead first, then the next two steps.
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx._vendor.mlx_vlm.models.cache import BatchKVCache

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    order = []
    ra, owner = _Recorder(order), types.SimpleNamespace()
    setattr(owner, tra._ATTR, ra)
    ids = list(range(1, 21))
    batch = ar.PromptProcessingBatch(
        _ServedForward(owner, order), [0], [ids], [4], mx.zeros((1, 20, 4)),
        {}, warm_cache=[BatchKVCache(left_padding=[0])], prefill_step_size=4,
        suffix_lens=[20])
    batch._apc_meta = [{"full_input_ids": ids}]
    assert batch.prompt_step() == 4
    assert order == ["submit", "submit", "forward"]
    assert ra.sent == [[(0, 4)], [(4, 12)]]


class _PlannedForward(_ServedForward):
    """A served model whose text model takes row plans."""

    def __init__(self, order):
        owner = types.SimpleNamespace(table_rows_ahead=self._plan)
        super().__init__(owner, order)
        self.plans = []

    def _plan(self, next_ids, n):
        self.order.append("plan")
        self.plans.append((n, np.array(next_ids).tolist()))


def test_a_prefill_chunk_names_the_next_chunk_before_its_forward(monkeypatch):
    # The model starts the next chunk's row reads behind its own, so the
    # plan reaches it before the forward. The last token is no chunk's.
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx._vendor.mlx_vlm.models.cache import BatchKVCache

    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    order = []
    model = _PlannedForward(order)
    ids = list(range(1, 12))
    batch = ar.PromptProcessingBatch(
        model, [0], [ids], [4], mx.zeros((1, 11, 4)),
        {}, warm_cache=[BatchKVCache(left_padding=[0])], prefill_step_size=4,
        suffix_lens=[11])
    assert [batch.prompt_step() for _ in range(3)] == [4, 4, 2]
    assert order == ["plan", "forward", "plan", "forward", "forward"]
    assert model.plans == [(4, [[5, 6, 7, 8]]), (4, [[9, 10]])]

    monkeypatch.setenv("GMLX_TABLE_READAHEAD", "0")
    tra.plan_rows_ahead(model, mx.array([ids]), 4, 4)
    assert len(model.plans) == 2


class _HeldHasher(_RowIsToken):
    """A hasher that says when it starts its first submission and holds
    it until ``go`` is set."""

    def __init__(self):
        self.started, self.go = threading.Event(), threading.Event()

    def __call__(self, hist):
        self.started.set()
        self.go.wait(10.0)
        return super().__call__(hist)


def test_a_close_with_reads_queued_drops_them_and_stops(tmp_path):
    # A release closes the read-ahead while a submission is being hashed
    # and another waits. The hash ends as close shuts the pool down. The
    # coordinator drops both and exits, so no thread is left blocked on
    # reads handed to a pool that close shut down.
    path = tmp_path / "table.gguf"
    path.write_bytes(b"\0" * (os.sysconf("SC_PAGE_SIZE") * 8))
    hasher = _HeldHasher()
    ra = tra.TableReadAhead(str(path), 0, 64, 90, hasher, workers=2)
    tokens = list(range(1, 33))
    ra.submit([(tokens, 0, 16)])
    ra.submit([(tokens, 16, 32)])
    assert hasher.started.wait(10.0)
    shutdown = ra._pool.shutdown

    def shut_then_release(*args, **kwargs):
        shutdown(*args, **kwargs)
        hasher.go.set()

    ra._pool.shutdown = shut_then_release
    ra.close()
    ra._thread.join(10.0)
    assert not ra._thread.is_alive()
    assert ra.wait(0) and ra._pending == 0


def test_a_release_closes_the_read_ahead_its_owner_holds(tmp_path):
    path = tmp_path / "table.gguf"
    path.write_bytes(b"\0" * os.sysconf("SC_PAGE_SIZE"))
    ra = tra.TableReadAhead(str(path), 0, 1, 1, _RowIsToken(), workers=1)
    owner = types.SimpleNamespace()
    object.__setattr__(owner, tra._ATTR, ra)
    served = _served(owner)
    assert tra.readahead_of(served) is ra
    installs.release(served)
    assert ra._closed and tra.readahead_of(served) is None
