"""Drive the real prefill loops on a stub model, CPU only.

A PromptProcessingBatch runs the stock prompt_step and needs_processing
under the installed decay wrapper (plain and folding), or the owned MTP
body under it, with a model that records each forward width. The
checkpoint schedule arms where serve arms it: at batch init on the plain
path, inside the first MTP body call on the MTP path. ckpt_store records
each store and succeeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

BS = 16
_ONE = mx.array(0.0)


class Cols:
    """Stand-in for the [1, L, ...] ids and embeddings: slices and shape."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    @property
    def shape(self):
        return (1, self.b - self.a, 1)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self
        _, s = key
        n = self.b - self.a
        start = 0 if s.start is None else min(max(s.start, 0), n)
        stop = n if s.stop is None else min(max(s.stop, 0), n)
        return Cols(self.a + start, self.a + max(start, stop))

    def tolist(self):
        return list(range(self.a, self.b))


class Cache:
    def __init__(self):
        self.offset = 0
        self.state = _ONE


class Model:
    def __init__(self, tags=("rot:2048:0", "kv"), *, folds=False,
                 mt="dense_probe", heads=16, streamed=True):
        self._kq_streaming = streamed
        self._kq_apc_ckpt_layout = tuple(tags)
        self._gmlx_adopted_prefill_step = 8192 if streamed else None
        self.supports_logits_to_keep = folds
        self.config = SimpleNamespace(model_type=mt,
                                      num_attention_heads=heads)
        self.calls: list = []

    def __call__(self, input_ids, cache=None, inputs_embeds=None, **kw):
        n = input_ids.shape[1]
        self.calls.append(n)
        for c in cache:
            c.offset += n
        return SimpleNamespace(logits=_ONE, hidden_states=[_ONE])


class _Manager:
    block_size = BS

    def store_exact_cache(self, *a, **k):
        raise AssertionError("stock exact store ran on an armed batch")

    def release(self, *a, **k):
        pass


def stock_batch_class(mp):
    """Puts PromptProcessingBatch back to the vendored copy's methods, with
    no install flag set, so a test can build the install chain in an order
    of its own. ``mp`` puts the process's chain back after the test."""
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx.upstream.patching import root

    cls = ar.PromptProcessingBatch
    for name, value in list(vars(cls).items()):
        if callable(value) and root(value) is not value:
            mp.setattr(cls, name, root(value))
        elif value is True and name.startswith(("_gmlx_", "_kq_")):
            mp.setattr(cls, name, False)


def install(mp, state, stored):
    """Install the serve chain with the render and store stubs. ``state``
    holds the next-turn prediction ``p`` and system prefix ``sys``;
    ``stored`` collects (position, kind, skeleton_disk) per store."""
    import gmlx.cache.retire_key as rk
    import gmlx.cache.snapshot as snapshot
    import gmlx.gen.prefill_decay as pd
    import gmlx.spec.ckpt as ckpt
    import gmlx.spec.engine as engine
    import gmlx.spec.mtp_prefill as mtp_prefill

    def layout(batch, block_size=BS):
        return tuple(getattr(batch.model, "_kq_apc_ckpt_layout", ())) or None

    def store(manager, ids, cache, **kw):
        stored.append((len(ids), kw.get("kind"), kw.get("skeleton_disk")))
        return True

    mp.setenv("GMLX_CB_PHASE", "0")
    mp.setattr(rk, "lookup_render_ctx", lambda ids: {"stub": True})
    mp.setattr(rk, "prompt_stable_lcp", lambda ctx, ids: state["p"])
    mp.setattr(rk, "system_prefix_lcp", lambda ctx, ids: state["sys"])
    mp.setattr(snapshot, "ckpt_store", store)
    mp.setattr(snapshot, "ckpt_note_armed", lambda m: None)
    mp.setattr(ckpt, "_ckpt_layout_live", layout)
    mp.setattr(engine, "_ckpt_layout_live", layout)
    mtp_prefill.install_full_prompt_mtp_prefill()
    assert pd.install_prefill_decay()
    ckpt._install_ckpt_checkpoint_store()
    mp.setattr(mtp_prefill, "_mtp_seed_stream_init", lambda b: None)


def make_batch(model, n, step=8192, mtp=False, guard=None, apc=True):
    from gmlx._vendor.mlx_vlm.generate import ar
    from gmlx._vendor.mlx_vlm.speculative.utils import SpeculativePrefill

    b = object.__new__(ar.PromptProcessingBatch)
    draft = object() if mtp else None
    meta = {"full_input_ids": list(range(n)), "prefix_len": 0,
            "extra_hash": 0, "checkpoint_len": guard}
    b.__dict__.update(
        model=model, uids=[0], _prompt_uids=[0], max_tokens=[1],
        prefill_step_size=step, draft_model=draft,
        draft_kind="mtp" if mtp else None, draft_block_size=None,
        greedy_sampling=False, _right_pad_per_row=None, _suffix_lens=[n],
        _left_padding_per_row=[0], _input_ids=Cols(0, n),
        _inputs_embeds=Cols(0, n), _total_prompt_tokens=n,
        _processed_prompt_columns=0, _finished_prompt_logits={},
        logits_processors=[], thinking_budget_criteria=[], _token_context=[],
        _prompt_kwargs={}, _prompt_length_aware_keys=[],
        _apc_meta=[meta] if apc else [],
        _apc_manager=_Manager() if apc else None,
        _apc_mode="exact" if apc else None, _apc_coordinator=None,
        prompt_cache=[Cache()], _cached_tokens_per_row=[0],
        _speculative_prefill=SpeculativePrefill(
            "mtp" if mtp else None, draft))
    return b


def drain(batch, limit=10000):
    """Run the loop to generate(). Returns the forward widths and the
    columns left for generate(), or raises on a stall."""
    ticks = 0
    while batch.needs_processing():
        ticks += 1
        if batch.prompt_step() == 0 or ticks > limit:
            raise AssertionError(
                f"prefill stalled after {batch.model.calls}")
    return list(batch.model.calls) + [batch._inputs_embeds.shape[1]]


def run(mp, model, n, restored, *, mtp=False, step=8192, arm=None):
    """Prefill ``n`` tokens past a ``restored`` prefix with the schedule
    ``arm(batch, meta, guard, restored)`` sets (default: the shipped one).
    Returns the chunks, the armed meta and the chunks ``_ckpt_chunks``
    predicted at arm time."""
    import gmlx.spec.ckpt as ckpt
    import gmlx.spec.engine as engine
    import gmlx.spec.mtp_prefill as mtp_prefill

    arm = arm or (lambda b, m, g, r: ckpt._ckpt_arm_schedule(b, m, g, r, BS))
    guard = n - 16
    model.calls = []
    out: dict = {}
    batch = make_batch(model, n, step, mtp, guard)

    def arm_here(b):
        meta = {"full_input_ids": list(range(n)), "prefix_len": 0,
                "extra_hash": 0, "apc_blocks": [], "checkpoint_len": guard}
        b._apc_meta = [meta]
        arm(b, meta, guard, restored)
        b._apc_harvest_enabled = False
        b._kq_ckpt_armed = True
        out["meta"] = dict(meta, ckpt_boundaries=list(
            meta["ckpt_boundaries"]))
        out["pred"] = ckpt._ckpt_chunks(
            b, [p for p, _ in meta["ckpt_boundaries"]], restored, n)

    if mtp:
        def init(b):
            if hasattr(b, "_mtp_full_input_ids"):
                return
            b._mtp_full_input_ids = b._input_ids
            b._mtp_chunk_hiddens = []
            b._mtp_l1_prefix_len = restored
            for c in b.prompt_cache:
                c.offset = restored
            arm_here(b)
            if restored > 0:
                engine._trim_prompt_columns(b, restored)
                b._processed_prompt_columns = restored
                b._mtp_apc_prefix_len = restored

        mp.setattr(mtp_prefill, "_mtp_prefill_init", init)
    else:
        for c in batch.prompt_cache:
            c.offset = restored
        if restored > 0:
            engine._trim_prompt_columns(batch, restored)
            batch._processed_prompt_columns = restored
        arm_here(batch)
    out["chunks"] = drain(batch)
    return out


def cost(chunks):
    """Expert passes and routed partial calls, as the schedule counts them."""
    from gmlx.load.loader import (
        _STREAM_PREFETCH_MIN_TOKENS,
        _arena_stage_max_tokens,
    )

    small = _arena_stage_max_tokens()
    return (sum(c > small for c in chunks),
            sum(_STREAM_PREFETCH_MIN_TOKENS <= c <= small for c in chunks))
