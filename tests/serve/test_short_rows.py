"""Short-lived rows: a client that caps its reply at a few tokens.

On a checkpoint-tier model a short-lived row makes no prompt cache lookup
and stores nothing, and concurrent short-lived rows prefill in one group.
Every other row, and every row on a block-tier or exact-tier model, keeps
the path it had.
"""

import dataclasses
import importlib
import types

import mlx.core as mx
import numpy as np
import pytest

import gmlx.cache.fresh_gate as fg
import gmlx.serve.admit_gate as ag
import gmlx.serve.bridge_vlm as serving
import gmlx.serve.governor as gov
import gmlx.serve.memory as server_memory
import gmlx.serve.short_rows as sr
import gmlx.spec.engine as engine
import gmlx.spec.mtp_prefill as mtp_prefill
from gmlx._vendor.mlx_vlm.utils import StoppingCriteria
from gmlx.cache.apc_manager import GmlxAPCManager
from gmlx.serve.patches import _common as sp_common
from gmlx.serve.patches import sampling as sp_sampling
from gmlx.serve.row_sampling import RowSettings

from test_prefill_fold import _V, _mint_q35_adapter, _qwen35, _row, _rows_channel  # noqa: F401

ar = importlib.import_module("gmlx._vendor.mlx_vlm.generate.ar")
_APP = importlib.import_module("gmlx._vendor.mlx_vlm.server.app")
_SCHEMAS = importlib.import_module("gmlx._vendor.mlx_vlm.server.schemas")

_BS = 16


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GMLX_PREFILL_DECAY", "0")
    monkeypatch.setenv("GMLX_CB_PHASE", "0")
    monkeypatch.setenv("GMLX_TICK_GUARD", "0")


def _short(**kw):
    return RowSettings(logprobs=True, top_k=_V, short_lived=True, **kw)


def _long(**kw):
    return RowSettings(logprobs=True, top_k=_V, **kw)


# -- classification -----------------------------------------------------------

def _chat(**kw):
    return _SCHEMAS.ChatRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kw)


def _responses(**kw):
    return _SCHEMAS.OpenAIRequest(model="m", input="hi", **kw)


def _anthropic(**kw):
    return _SCHEMAS.AnthropicRequest(
        model="m", messages=[{"role": "user", "content": "hi"}], **kw)


def _spec(**sampling):
    return types.SimpleNamespace(sampling=sampling)


@pytest.fixture
def seam():
    """The gen-args seam in the server's order."""
    sp_sampling.install_gen_args_profile_injection()
    sp_sampling.install_max_completion_tokens()
    sp_sampling.install_short_lived_mark()
    fn = _APP._build_gen_args
    assert getattr(fn, sp_common._PATCH_FLAG, False)
    assert getattr(fn, sp_sampling._SHORT_LIVED_FLAG, False)

    def build(request, spec=None):
        tok = serving.set_active_spec(spec)
        try:
            return _APP._build_gen_args(request)
        finally:
            serving.reset_active_spec(tok)
    return build


@pytest.mark.parametrize("request_, spec, want", [
    (lambda: _chat(max_tokens=16), None, True),
    (lambda: _chat(max_tokens=1), _spec(max_tokens=4096), True),
    (lambda: _chat(max_completion_tokens=4), None, True),
    (lambda: _chat(max_tokens=200, max_completion_tokens=8), None, True),
    (lambda: _responses(max_output_tokens=16), None, True),
    (lambda: _anthropic(max_tokens=2), None, True),
    (lambda: _chat(max_tokens=17), None, False),
    (lambda: _chat(max_tokens=4, max_completion_tokens=64), None, False),
    (lambda: _chat(), None, False),
    (lambda: _chat(), _spec(max_tokens=4), False),
    (lambda: _chat(max_completion_tokens=0), _spec(max_tokens=4), False),
    (lambda: _responses(), None, False),
    (lambda: _anthropic(max_tokens=17), None, False),
], ids=["max-tokens-16", "client-beats-profile", "max-completion",
        "max-completion-wins", "responses", "messages", "max-tokens-17",
        "final-cap-17-plus", "unset", "profile-only", "bad-alias-profile",
        "responses-default", "messages-17"])
def test_a_client_cap_of_16_or_less_marks_the_request(seam, request_, spec,
                                                      want):
    args = seam(request_(), spec)
    assert args._kq_short_lived is want


def test_the_mark_rides_the_row_settings():
    from gmlx._vendor.mlx_vlm.server.generation import (
        GenerationArguments,
        ResponseGenerator,
    )

    rg = object.__new__(ResponseGenerator)
    rg.__dict__.update(top_logprobs_k=0, model=types.SimpleNamespace(),
                       draft_model=None)
    marked = GenerationArguments(max_tokens=4)
    marked._kq_short_lived = True
    assert rg._row_settings(marked).short_lived is True
    assert rg._row_settings(GenerationArguments(max_tokens=4)).short_lived \
        is False


@pytest.mark.parametrize("n, settings, want", [
    (sr._SHORT_CAP, _short(), True),
    (sr._SHORT_CAP + 1, _short(), False),
    (8, _long(), False),
    (8, None, False),
], ids=["at-cap", "over-cap", "not-marked", "no-settings"])
def test_a_marked_row_at_or_under_the_cap_is_short_lived(n, settings, want):
    gen = types.SimpleNamespace(_kq_ckpt_tier=True, draft_model=None,
                                _row_settings={})
    if settings is not None:
        gen._row_settings[0] = settings
    assert sr.is_short_lived(gen, (0, list(range(n)), 4, {}, None, None)) \
        is want
    gen._kq_ckpt_tier = False
    assert not sr.is_short_lived(gen, (0, list(range(n)), 4, {}, None, None))


def test_the_cap_stays_under_the_fresh_gate_floor():
    assert sr._SHORT_CAP < fg._hold_min()


# -- the checkpoint-tier engine ----------------------------------------------

@pytest.fixture
def ckpt(monkeypatch):
    """A Qwen3.5 dense model on the checkpoint tier, as serve runs it."""
    mtp_prefill.install_full_prompt_mtp_prefill()
    monkeypatch.setattr(engine, "_SPEC_APC_CKPT_DISABLED", False)
    return _Rig(monkeypatch)


class _Rig:
    def __init__(self, monkeypatch, dtype=mx.float32, **load):
        self.model = _qwen35(monkeypatch, dtype=dtype, **load)
        self.lm = self.model.language_model
        self.proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
            stopping_criteria=StoppingCriteria([], None)))
        self.batches, self.sizes = [], []
        batches, sizes = self.batches, self.sizes
        init = ar.PromptProcessingBatch.__init__

        def spy(batch, *a, **k):
            init(batch, *a, **k)
            batches.append(batch)
            sizes.append(len(batch.uids))

        monkeypatch.setattr(ar.PromptProcessingBatch, "__init__", spy)

    def manager(self, **kw):
        m = GmlxAPCManager(num_blocks=256, block_size=_BS, **kw)
        m._disk_min_free_ram_bytes = 0
        return m

    def gen(self, manager, **kw):
        kw.setdefault("completion_batch_size", 8)
        gen = ar.BatchGenerator(self.lm, self.proc,
                                sampler=lambda x: mx.argmax(x, axis=-1),
                                top_logprobs_k=_V, apc_manager=manager, **kw)
        assert gen._kq_ckpt_tier and gen.prefill_batch_size == 1
        return gen

    def insert(self, gen, prompts, settings, max_tokens=2):
        return gen.insert(
            prompts, max_tokens,
            prompt_kwargs=[{"inputs_embeds": self.lm.input_embeds(
                mx.array([p])), "_apc_semantic_hash": 0} for p in prompts],
            row_settings=settings)

    def drain(self, gen, out=None):
        out = {} if out is None else out
        while gen.has_work:
            _p, resp = gen.next()
            for r in resp:
                out.setdefault(r.uid, []).append(
                    (int(r.token), dict(r.top_logprobs or ())))
        return out

    def run(self, manager, prompts, settings, max_tokens=2, **kw):
        gen = self.gen(manager, **kw)
        try:
            uids = self.insert(gen, prompts, settings, max_tokens)
            out = self.drain(gen)
        finally:
            gen.close()
        return [out[u] for u in uids]

    def widths(self):
        return list(self.sizes)

    def clear(self):
        self.batches.clear()
        self.sizes.clear()


def _prompt(n, seed=1):
    return [1 + (i * 7 + seed) % (_V - 2) for i in range(n)]


def _ckpt_stats(manager):
    snap = manager.stats_snapshot()
    return snap.get("ckpt_stores", 0), snap.get("ckpt_hits", 0)


@pytest.mark.parametrize("n, settings", [
    (40, _long()),
    (sr._SHORT_CAP + 8, _short()),
], ids=["not-marked", "over-cap"])
def test_a_row_over_either_limit_stores_and_hits_as_before(ckpt, n,
                                                           settings):
    m = ckpt.manager()
    prompt = _prompt(n)
    ckpt.run(m, [prompt], [settings])
    stores, hits = _ckpt_stats(m)
    assert stores > 0
    ckpt.run(m, [prompt + [5, 6, 7]], [settings])
    assert _ckpt_stats(m)[1] > hits


class _Calls:
    """Counts the calls of each named function or method, wherever gmlx
    bound it."""

    def __init__(self, monkeypatch):
        self.mp = monkeypatch
        self.n = {}

    def _wrap(self, name, orig):
        n = self.n
        n.setdefault(name, 0)

        def call(*a, **k):
            n[name] += 1
            return orig(*a, **k)
        return call

    def method(self, cls, name):
        self.mp.setattr(cls, name, self._wrap(name, getattr(cls, name)))

    def function(self, module, name):
        import sys

        orig = getattr(module, name)
        wrapped = self._wrap(name, orig)
        for modname, mod in list(sys.modules.items()):
            if modname.startswith("gmlx") and mod is not None \
                    and vars(mod).get(name) is orig:
                self.mp.setattr(mod, name, wrapped)


@pytest.fixture
def calls(monkeypatch):
    import gmlx.cache.snapshot as snapshot
    from gmlx._vendor.mlx_vlm import apc
    from gmlx._vendor.mlx_vlm.apc_coordinator import APCCoordinator

    c = _Calls(monkeypatch)
    c.method(GmlxAPCManager, "lookup_exact_cache")
    c.method(apc.APCManager, "lookup_exact_cache")
    c.method(apc.DiskBlockStore, "find_exact_prefix")
    c.method(APCCoordinator, "store_checkpoint")
    c.method(APCCoordinator, "commit")
    for name in ("ckpt_lookup", "ckpt_peek", "ckpt_store",
                 "decode_ckpt_tick"):
        c.function(snapshot, name)
    return c


def _no_stash(rig):
    for b in rig.batches:
        assert not getattr(b, "_kq_ckpt_armed", False)
        for c in b.prompt_cache or ():
            assert getattr(c, "_kq_apc_retire", None) is None


def test_a_short_lived_row_makes_no_lookup_and_stores_nothing(
        ckpt, calls, tmp_path):
    from gmlx._vendor.mlx_vlm import apc

    disk = apc.DiskBlockStore(tmp_path, namespace="t")
    m = ckpt.manager(disk=disk)
    prompt = _prompt(40)
    # A record the short row could hit, stored by a longer request.
    ckpt.run(m, [prompt], [_long()])
    m.disk.flush()
    stored = sorted(p.name for p in tmp_path.rglob("*"))
    ckpt.clear()
    for k in calls.n:
        calls.n[k] = 0
    gen = ckpt.gen(m)
    try:
        ckpt.insert(gen, [prompt + [9]], [_short()])
        ckpt.drain(gen)
        registry = getattr(gen._generation_batch, "_kq_apc_retire_rows",
                           None)
    finally:
        gen.close()
    m.disk.flush()
    assert calls.n == dict.fromkeys(calls.n, 0)
    assert not registry
    _no_stash(ckpt)
    assert sorted(p.name for p in tmp_path.rglob("*")) == stored


def _close(a, b, atol):
    assert len(a) == len(b)
    (tok_a, lp_a), (tok_b, lp_b) = a[0], b[0]
    assert tok_a == tok_b
    keys = sorted(lp_b)
    np.testing.assert_allclose([lp_a[k] for k in keys],
                               [lp_b[k] for k in keys], atol=atol)


# The group prefills each row at the batch's width, left padded to the
# longest row, so a row's logits move at float tolerance against the row
# alone. f32 GEMM runs at TF32 on the GPU, and bf16 allows two rounding
# steps at the size of these log probabilities (4 to 8).
_ATOL = {mx.float32: 2e-2, mx.bfloat16: 6.25e-2}


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16],
                         ids=["f32", "bf16"])
def test_a_group_matches_each_row_alone(monkeypatch, calls, dtype):
    mtp_prefill.install_full_prompt_mtp_prefill()
    rig = _Rig(monkeypatch, dtype=dtype)
    prompts = [_prompt(5, 1), _prompt(23, 2), _prompt(12, 3)]
    m = rig.manager()
    grouped = rig.run(m, prompts, [_short() for _ in prompts])
    assert rig.widths() == [3]
    group_calls = dict(calls.n)
    rig.clear()
    alone = [rig.run(m, [p], [_short()])[0] for p in prompts]
    assert rig.widths() == [1, 1, 1]
    assert group_calls == calls.n == dict.fromkeys(calls.n, 0)
    for g, a in zip(grouped, alone):
        _close(g, a, _ATOL[dtype])


def test_a_group_of_mrope_text_rows_matches_each_row_alone(monkeypatch):
    """The Qwen3.5 image model's text rows carry position ids and zero rope
    deltas, which the group pads with the rows."""
    from test_vlm_spec_target_swap import _q35_vlm

    mtp_prefill.install_full_prompt_mtp_prefill()
    rig = _Rig(monkeypatch)
    mx.random.seed(5)
    model = _q35_vlm()
    mx.eval(model.parameters())
    rig.lm = model.language_model

    def kwargs(p):
        emb = model.get_input_embeddings(mx.array([p]))
        kw = {k: v for k, v in emb.to_dict().items() if v is not None}
        assert not kw["rope_deltas"].any().item()
        return {**kw, "_apc_semantic_hash": 0}

    def run(prompts):
        gen = rig.gen(rig.manager())
        try:
            uids = gen.insert(prompts, 2,
                              prompt_kwargs=[kwargs(p) for p in prompts],
                              row_settings=[_short() for _ in prompts])
            out = rig.drain(gen)
        finally:
            gen.close()
        return [out[u] for u in uids]

    prompts = [_prompt(5, 1), _prompt(23, 2), _prompt(12, 3)]
    grouped = run(prompts)
    assert rig.widths() == [3]
    for g, p in zip(grouped, prompts):
        _close(g, run([p])[0], _ATOL[mx.float32])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16],
                         ids=["f32", "bf16"])
def test_a_group_of_adapted_and_bare_rows_matches_each_row_alone(
        monkeypatch, tmp_path, _rows_channel, dtype):  # noqa: F811
    mtp_prefill.install_full_prompt_mtp_prefill()
    adapter = _mint_q35_adapter(tmp_path / "adapter.gguf")
    rig = _Rig(monkeypatch, dtype=dtype, adapter_gguf=adapter)
    assert _rows_channel.mode() == "rows"
    prompts = [_prompt(12, 1), _prompt(7, 2), _prompt(19, 3)]
    rows = [dataclasses.replace(_row(0.5, 11), short_lived=True),
            dataclasses.replace(_row(1.5, 22), short_lived=True),
            dataclasses.replace(_row(0.0, 0), short_lived=True)]
    m = rig.manager()
    grouped = rig.run(m, prompts, rows)
    assert rig.widths() == [3]
    rig.clear()
    alone = [rig.run(m, [p], [r])[0] for p, r in zip(prompts, rows)]
    atol = _ATOL[dtype]
    for g, a in zip(grouped, alone):
        _close(g, a, atol)
    # The adapter moves the adapted rows by more than the tolerance.
    bare = [rig.run(m, [p], [rows[2]])[0] for p in prompts[:2]]
    for g, b in zip(grouped[:2], bare):
        lp_g, lp_b = g[0][1], b[0][1]
        assert max(abs(lp_g[k] - lp_b[k]) for k in lp_b) > 5 * atol


def test_a_cold_group_runs_no_column_split_and_no_commit(ckpt, calls):
    m = ckpt.manager()
    gen = ckpt.gen(m)
    try:
        ckpt.insert(gen, [_prompt(30, 1), _prompt(60, 2)],
                    [_short(), _short()])
        gen.next()
        (batch,) = ckpt.batches
        assert batch._apc_harvest_enabled is False
        assert all(meta["checkpoint_len"] == 0
                   and meta["checkpoint_lengths"] == []
                   for meta in batch._apc_meta)
        assert batch._next_apc_checkpoint_column() is None
        ckpt.drain(gen)
    finally:
        gen.close()
    assert calls.n["commit"] == 0 and calls.n["store_checkpoint"] == 0


def test_an_ineligible_head_goes_alone_before_the_group(ckpt):
    m = ckpt.manager()
    ckpt.run(m, [_prompt(40, 1), _prompt(9, 2), _prompt(14, 3),
                 _prompt(6, 4)], [_long(), _short(), _short(), _short()])
    assert ckpt.widths() == [1, 3]


def test_groups_reach_the_saved_width(ckpt):
    m = ckpt.manager()
    gen = ckpt.gen(m, completion_batch_size=8)
    assert gen.prefill_batch_size == 1 and gen._kq_short_group_width == 8
    try:
        ckpt.insert(gen, [_prompt(5 + i, i) for i in range(10)],
                    [_short() for _ in range(10)], max_tokens=4)
        ckpt.drain(gen)
    finally:
        gen.close()
    assert ckpt.widths()[0] == 8
    assert sum(ckpt.widths()) == 10


def test_a_layout_without_recurrent_state_keeps_groups_of_one(ckpt,
                                                              monkeypatch):
    """A sliding-window-only layout (gemma 4) changes a row's tokens with
    the prefill width, so its short rows prefill alone and still skip the
    prompt cache."""
    tags = tuple(engine._ckpt_layout_for(ckpt.lm, _BS))
    assert "arr" in tags
    monkeypatch.setattr(ckpt.lm, "_kq_apc_ckpt_layout",
                        tuple("rot:64" if t == "arr" else t for t in tags),
                        raising=False)
    m = ckpt.manager()
    gen = ckpt.gen(m, completion_batch_size=8)
    assert gen._kq_short_group_width == 1
    try:
        ckpt.insert(gen, [_prompt(5 + i, i) for i in range(4)],
                    [_short() for _ in range(4)], max_tokens=2)
        ckpt.drain(gen)
    finally:
        gen.close()
    assert ckpt.widths() == [1, 1, 1, 1]
    assert _ckpt_stats(m) == (0, 0)


def test_closed_loop_clients_form_groups(ckpt, monkeypatch):
    monkeypatch.setenv("GMLX_DECODE_BATCH", "8")
    m = ckpt.manager()
    gen = ar.BatchGenerator(ckpt.lm, ckpt.proc,
                            sampler=lambda x: mx.argmax(x, axis=-1),
                            apc_manager=m)
    assert gen.completion_batch_size == 8
    clients = 8
    done = 0
    try:
        for c in range(clients):
            ckpt.insert(gen, [_prompt(6 + c, c)], [_short()], max_tokens=1)
        for _tick in range(200):
            if done >= 32:
                break
            _p, resp = gen.next()
            for r in resp:
                if r.finish_reason is not None:
                    done += 1
                    ckpt.insert(gen, [_prompt(5 + done % 9, done)],
                                [_short()], max_tokens=1)
    finally:
        gen.close()
    assert done >= 32
    assert max(ckpt.widths()) > 1


# -- the four admission sites ------------------------------------------------

S, L = "short", "long"


def _site_gen(monkeypatch, kinds, completion=3, **attrs):
    """A checkpoint-tier generator, by its flags, with nothing decoding and
    ``kinds`` pending in arrival order."""
    from test_arrival_order import _engine

    gen = _engine(monkeypatch, completion_batch_size=completion,
                  prefill_batch_size=1)
    gen.prefill_batch_size = 1
    gen._kq_ckpt_tier = True
    gen._kq_short_group_width = 8
    for k, v in attrs.items():
        setattr(gen, k, v)
    for i, kind in enumerate(kinds):
        ids = [2 + i, 3 + i, 4 + i]
        gen.insert([ids], 2, prompt_kwargs=[
            {"inputs_embeds": gen.model.input_embeds(mx.array([ids]))}],
            row_settings=[_short() if kind == S else _long()])
    return gen


_SEVEN = [S, L, S, S, S, L, S]
_GROUP = [0, 2, 3]
_AFTER = [0, 2, 3, 1, 4, 5, 6]


def _pending(gen):
    return [s[0] for s in gen._unprocessed_sequences]


def test_the_stock_body_takes_the_group(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN)
    try:
        gen.next()
        assert _pending(gen) == [1, 4, 5, 6]
        assert sorted(gen._generation_batch.uids) == _GROUP
    finally:
        gen.close()


def test_the_admit_gate_prices_the_group(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN)
    priced = []
    monkeypatch.setattr(server_memory, "project_admission",
                        lambda g, cands: priced.append(
                            [s[0] for s in cands]))
    assert ag._should_decline(gen) is False
    assert priced == [_GROUP]
    assert ag._candidate_uids(gen) == _GROUP
    assert _pending(gen) == _AFTER
    gen.close()


def test_the_fresh_gate_walks_the_group(monkeypatch):
    """Rows 0 and 3 share a prefix past the floor, so the gate holds row 3
    for row 0's stores: it walked the group, not the arrival prefix."""
    from test_arrival_order import _Manager

    monkeypatch.setenv("GMLX_APC_FRESH_MIN", "2")
    gen = _site_gen(monkeypatch, _SEVEN)
    gen.apc_manager = _Manager()
    monkeypatch.setattr(gen, "_apc_extra_hash", lambda kw: 0, raising=False)
    seq = gen._unprocessed_sequences
    seq[3] = (seq[3][0], [2, 3, 4, 9], *seq[3][2:])
    assert fg._keep_count(gen) == 2
    assert sr.admit_width(gen) == 3
    assert _pending(gen) == _AFTER
    gen.close()


def test_the_governor_sees_the_group(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN)
    monkeypatch.setattr(gov, "admission_hold_reason", lambda g: None)
    assert gov._prompt_may_step(gen) is True
    assert sr.admit_width(gen) == 3
    assert _pending(gen) == _AFTER
    gen.close()


def test_every_site_picks_the_same_group_and_repeats_it(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN)
    assert sr.admit_group(gen) == 3
    first = list(gen._unprocessed_sequences)
    for _ in range(3):
        assert sr.admit_group(gen) == 3
        assert gen._unprocessed_sequences == first
    gen.close()


def test_the_group_reorders_the_list_the_gates_hold(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN)
    held = gen._unprocessed_sequences
    sr.admit_group(gen)
    assert gen._unprocessed_sequences is held
    gen.close()


def _media(seq):
    return (seq[0], seq[1], seq[2], {**seq[3], "image_grid_thw": mx.array(
        [[1, 2, 2]])}, *seq[4:])


def _mrope(seq):
    return (seq[0], seq[1], seq[2], {**seq[3], "rope_deltas": mx.array(
        [[3]])}, *seq[4:])


@pytest.mark.parametrize("make", ["long", "media", "mrope", "speech"])
def test_an_ineligible_row_prefills_alone(monkeypatch, make):
    gen = _site_gen(monkeypatch, [S, S, S, S])
    seq = gen._unprocessed_sequences
    for i in (0, 2):
        uid = seq[i][0]
        if make == "long":
            gen._row_settings[uid] = _long()
        elif make == "speech":
            gen._row_settings[uid] = _short(speech=object())
        else:
            seq[i] = (_media if make == "media" else _mrope)(seq[i])
    assert sr.admit_group(gen) == 1
    assert _pending(gen) == [0, 1, 2, 3]
    gen._unprocessed_sequences.pop(0)
    assert sr.admit_group(gen) == 2
    assert _pending(gen) == [1, 3, 2]
    gen.close()


def test_zero_rope_deltas_are_text(monkeypatch):
    gen = _site_gen(monkeypatch, [S, S])
    seq = gen._unprocessed_sequences
    for i in (0, 1):
        seq[i] = (*seq[i][:3], {**seq[i][3],
                                "rope_deltas": mx.zeros((1, 1))}, *seq[i][4:])
    assert sr.admit_group(gen) == 2
    gen.close()


@pytest.mark.parametrize("attrs", [
    {"draft_model": object()},
    {"kv_quant_scheme": "kvarn"},
    {"_kq_short_group_width": 1},
], ids=["drafter", "kvarn", "pooling-width"])
def test_a_generator_that_cannot_group_admits_one_row(monkeypatch, attrs):
    gen = _site_gen(monkeypatch, [S, S, S], **attrs)
    assert sr.admit_group(gen) == 1
    gen.close()


def test_a_drafter_entry_keeps_its_lookup(monkeypatch):
    gen = types.SimpleNamespace(_kq_ckpt_tier=True, draft_model=object(),
                                _row_settings={0: _short()})
    assert not sr.is_short_lived(gen, (0, [1, 2, 3], 2, {}, None, None))


def test_a_pooling_model_keeps_groups_of_one(monkeypatch):
    import gmlx.cache.apc_pooling as pooling
    from gmlx._vendor.mlx_vlm.apc import APCManager
    from test_ckpt_tier import make_hybrid_cache

    class _ProbeBG:
        def __init__(self, model, processor, **kwargs):
            self.prefill_batch_size = kwargs.get("prefill_batch_size", 8)
            self.completion_batch_size = kwargs.get("completion_batch_size")

    monkeypatch.setattr(ar, "BatchGenerator", _ProbeBG)
    engine._install_apc_manager_stash()
    engine._bind_l1_view()
    pooling.install_pooled_prefill_batch_gate()
    monkeypatch.setattr(pooling, "model_has_pools", lambda m: True)
    hybrid = types.SimpleNamespace(make_cache=lambda: make_hybrid_cache(0))
    bg = ar.BatchGenerator(hybrid, None, completion_batch_size=8,
                           apc_manager=APCManager(num_blocks=64,
                                                  block_size=16))
    assert bg._kq_ckpt_tier
    assert bg.prefill_batch_size == 1 and bg._kq_short_group_width == 1


def test_off_the_ckpt_tier_the_helper_leaves_admission_alone(monkeypatch):
    gen = _site_gen(monkeypatch, _SEVEN, _kq_ckpt_tier=False)
    gen.prefill_batch_size = 2
    assert sr.admit_group(gen) is None
    assert sr.admit_width(gen) == 2
    assert _pending(gen) == list(range(7))
    gen.close()


# -- block and exact tiers ---------------------------------------------------

def _llama(monkeypatch):
    from test_arrival_order import _engine

    return _engine(monkeypatch).model


def _tier_rig(monkeypatch, tier):
    if tier == "block":
        model = _llama(monkeypatch)
        return model, (lambda p: {"inputs_embeds": model.input_embeds(
            mx.array([p]))})
    from test_prompt_cache_memory_checks import _kw, _rig

    monkeypatch.setattr(engine, "_SPEC_APC_CKPT_DISABLED", True)
    return _rig().lm, _kw


@pytest.mark.parametrize("tier", ["block", "exact"])
@pytest.mark.parametrize("width", [None, 32, 4], ids=["default", "32", "4"])
def test_off_the_ckpt_tier_a_short_row_runs_as_before(monkeypatch, tier,
                                                      width):
    mtp_prefill.install_full_prompt_mtp_prefill()
    if width is not None:
        monkeypatch.setenv("GMLX_DECODE_BATCH", str(width))
    lm, kw = _tier_rig(monkeypatch, tier)
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        stopping_criteria=StoppingCriteria([], None)))
    sizes = []
    init = ar.PromptProcessingBatch.__init__

    def spy(batch, *a, **k):
        init(batch, *a, **k)
        sizes.append(len(batch.uids))

    monkeypatch.setattr(ar.PromptProcessingBatch, "__init__", spy)
    prompts = [[1 + (i * 5 + j) % 40 for j in range(20 + i)]
               for i in range(6)]

    def run(make):
        m = GmlxAPCManager(num_blocks=128, block_size=_BS)
        m._exact_cache_max = 8
        m._disk_min_free_ram_bytes = 0
        sizes.clear()
        gen = ar.BatchGenerator(lm, proc,
                                sampler=lambda x: mx.argmax(x, axis=-1),
                                apc_manager=m)
        assert not getattr(gen, "_kq_ckpt_tier", False)
        out = {}
        try:
            uids = gen.insert(prompts, 2,
                              prompt_kwargs=[kw(p) for p in prompts],
                              row_settings=[make() for _ in prompts])
            while gen.has_work:
                _p, resp = gen.next()
                for r in resp:
                    out.setdefault(r.uid, []).append(int(r.token))
        finally:
            gen.close()
        snap = m.stats_snapshot()
        held = sum(b.ref_cnt for b in m.pool)
        return ([out[u] for u in uids], list(sizes), held,
                snap["stores"], snap["exact_stores"])

    short, before = run(_short), run(_long)
    assert short == before
    assert short[2] == 0
    assert short[3] + short[4] > 0
    if width == 32:
        assert max(short[1]) > 1
