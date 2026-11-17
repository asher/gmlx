"""Short-lived rows on checkpoint-tier models.

A row is short-lived when its client capped its reply at
``_SHORT_LIMIT`` tokens or fewer and its prompt is at most ``_SHORT_CAP``
tokens. A missing cap, the server default and a profile value never
count. A gen-args transform (``serve.patches.sampling``) marks the
request, and ``RowSettings.short_lived`` carries the mark to the row.

On a model whose prompt cache runs the checkpoint tier, a short-lived
row makes no prompt cache lookup and stores nothing: a record holds the
whole recurrent state at any prompt length, and these rows are rarely
extended. On a layout with recurrent state, short-lived rows that are
text-only, with no drafter and no kvarn KV, also prefill together: ``admit_group`` moves up to
``min(saved width, free slots)`` of them to the head of the pending list,
and the admission sites slice pending by ``admit_width``. Every other row
prefills alone, as before. Block-tier and exact-tier models keep their
admission unchanged.
"""

from __future__ import annotations

from .batch_rows import batch_rows

_SHORT_LIMIT = 16
_SHORT_CAP = 128  # must stay under GMLX_APC_FRESH_MIN (fresh_gate)


def client_short_lived(client_set: bool, max_tokens) -> bool:
    """Whether a request whose client set its output cap (``client_set``)
    and whose final cap is ``max_tokens`` is short-lived."""
    return (client_set and isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and max_tokens <= _SHORT_LIMIT)


def is_short_lived(gen, seq) -> bool:
    """Whether pending ``seq`` is a short-lived row on a checkpoint-tier
    generator without a drafter."""
    if not getattr(gen, "_kq_ckpt_tier", False):
        return False
    if getattr(gen, "draft_model", None) is not None:
        return False
    settings = (getattr(gen, "_row_settings", None) or {}).get(seq[0])
    if settings is None or not getattr(settings, "short_lived", False):
        return False
    try:
        return len(seq[1]) <= _SHORT_CAP
    except TypeError:
        return False


def mark_short_meta(gen, sequences, apc_meta) -> None:
    """Mark the cold meta of each short-lived row in ``sequences``."""
    if not apc_meta or not getattr(gen, "_kq_ckpt_tier", False):
        return
    for meta, seq in zip(apc_meta, sequences):
        if meta is not None and is_short_lived(gen, seq):
            meta["short_lived"] = True


def all_short_lived(apc_meta) -> bool:
    """Whether every row of a prompt batch's meta is short-lived."""
    return bool(apc_meta) and all(
        m is not None and m.get("short_lived") for m in apc_meta)


def _media_ids(gen) -> set:
    ids = getattr(gen, "_kq_short_media_ids", None)
    if ids is None:
        from gmlx.spec.mtp_prefill import media_token_ids

        ids = gen._kq_short_media_ids = media_token_ids(
            getattr(gen, "model", None))
    return ids


def _text_only(gen, seq) -> bool:
    from gmlx.spec.mtp_prefill import row_carries_media

    kwargs = seq[3] or {}
    deltas = kwargs.get("rope_deltas")
    if deltas is not None:
        import mlx.core as mx

        if bool(mx.any(mx.array(deltas) != 0).item()):
            return False
    ids = seq[1]
    if not isinstance(ids, list):
        ids = ids.tolist()
    return not row_carries_media(kwargs, ids, _media_ids(gen))


def _eligible(gen, seq) -> bool:
    if not is_short_lived(gen, seq):
        return False
    if getattr(gen, "kv_quant_scheme", None) == "kvarn":
        return False
    settings = gen._row_settings[seq[0]]
    if getattr(settings, "speech", None) is not None:
        return False
    try:
        return _text_only(gen, seq)
    except Exception:
        return False


def admit_group(gen) -> int | None:
    """Form this tick's group on a checkpoint-tier generator and return
    its size, or None when the generator keeps its stock admission.

    An ineligible head is admitted alone. Otherwise up to
    ``min(_kq_short_group_width, free slots)`` eligible rows move, in
    arrival order, to the head of pending, in place: the gates hold the
    list. The size is kept for ``admit_width`` until the next call."""
    if not getattr(gen, "_kq_ckpt_tier", False):
        return None
    gen._kq_admit_n = None
    pending = gen._unprocessed_sequences
    if not pending or getattr(gen, "_prompt_batch", None) is not None:
        return None
    free = gen.completion_batch_size - batch_rows(gen)
    if free < 1:
        return None
    width = min(int(getattr(gen, "_kq_short_group_width", 1) or 1), free)
    if width <= 1 or not _eligible(gen, pending[0]):
        gen._kq_admit_n = 1
        return 1
    snap = list(pending)
    picked = [0]
    for i in range(1, len(snap)):
        if len(picked) == width:
            break
        if _eligible(gen, snap[i]):
            picked.append(i)
    if picked[-1] != len(picked) - 1:
        chosen = set(picked)
        # Only the rows seen here move; a row appended meanwhile stays last.
        pending[:len(snap)] = ([snap[i] for i in picked]
                               + [s for i, s in enumerate(snap)
                                  if i not in chosen])
    gen._kq_admit_n = len(picked)
    return len(picked)


def admit_width(gen) -> int:
    """Rows the admission sites take from the head of pending this tick."""
    k = getattr(gen, "_kq_admit_n", None)
    return gen.prefill_batch_size if k is None else k
