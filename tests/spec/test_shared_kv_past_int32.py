"""The shared-KV tail pad of a gemma-4 assistant drafter at an output past
2^31 elements.

On a GPU stream of mlx 0.32.3 a concatenate on a later axis misplaces input
elements past 2^31 output elements (docs/upstream-defects.md), and
_pad_shared_kv_seq goes through gmlx.mlx_guards. When the last rows of an
MTP batch finish while two prefilled rows wait to join, the joining rows'
full-attention keys are padded to the old batch's depth. The test pads the
keys of gemma-4-E4B's full-attention layer (2 KV heads of 512, bfloat16) at
the first depth past the limit, and checks every key of the output and of
the input against the formula of tests/past_int32.py.
"""
from __future__ import annotations

import mlx.core as mx

import past_int32 as p32
from gmlx.spec.speculative import _pad_shared_kv_seq

pytestmark = p32.MARKS
# The fixture MARKS uses, where pytest finds it.
settled = p32.settled

H, D = 2, 512


def _pad(n: int, w: int, s: int) -> tuple:
    """Keys of ``n`` rows at ``w`` tokens padded to ``s``: the output's
    shape, the wrong elements of the last two tokens of row n - 1, head 1,
    of every key and of the input."""
    rows = n * H
    keys = p32.bf16((n, H, w, D))
    out = _pad_shared_kv_seq(keys, s)
    mx.eval(out)
    region = p32.wrong(out[n - 1:, H - 1:, w - 2:w], (1, 2, D),
                       lambda r, s0, m: p32.pattern(rows - 1, 1, w - 2 + s0, m, D)[0])
    whole = p32.wrong(out, (rows, s, D), lambda r, s0, m: p32.placed(r, s0, m, D, 0, w))
    own = p32.wrong(keys, (rows, w, D), lambda r, s0, m: p32.pattern(r, 1, s0, m, D)[0])
    return tuple(out.shape), region, whole, own


def test_joining_rows_padded_past_2_31_elements_keep_every_key():
    """Two rows at 1,048,575 tokens pad to 1,048,577. The last two tokens
    of row 1, head 1 end past INT32_MAX, which the stock concatenate left
    unwritten and wrote 8 GiB below."""
    shape, region, whole, own = _pad(2, 1_048_575, 1_048_577)
    assert shape == (2, H, 1_048_577, D)
    # {(row, first token of a span): wrong elements}, all three at once.
    assert {"region": region, "keys": whole, "input": own} == {
        "region": {}, "keys": {}, "input": {}}
