"""Row count of the live generation batch, without side effects.

On the speculative path, len() on the generation batch is a mutation:
the continuous-batching retrofit (spec_engine) makes SpecBatch.__len__
release heavy request state and promote a buffered injection by
swapping the whole batch state (__dict__.update). That is the engine's
own promotion mechanism and is by design; but gmlx decision and
watcher code (admission gates, memory projections, schedulers) reading
len() could swap the live batch out from under its own decision,
between two reads inside one predicate. Decision code calls
batch_rows() instead: the same active-row count, read from the
_finished flags, never promoting.

A pending buffered injection therefore reads as 0 rows until the
engine's own next len() promotes it, which is the decision-side view
of the batch as it currently is.
"""


def batch_rows(gen) -> int:
    """Active rows in ``gen._generation_batch``; 0 when there is none.
    Pure: never triggers the spec batch's release/promotion side
    effects, so it is safe from gates, projections, and watchers."""
    return batch_rows_of(getattr(gen, "_generation_batch", None))


def batch_rows_of(batch) -> int:
    """Active rows of a generation batch object (see ``batch_rows``).

    A speculative batch carries ``_finished`` flags; counting them
    matches the stock ``SpecBatch.__len__`` without the promotion
    side effect. A plain batch has no flags and a pure ``__len__``.
    """
    if batch is None:
        return 0
    finished = getattr(batch, "_finished", None)
    if finished is not None:
        return sum(1 for done in finished if not done)
    return len(batch)
