"""The owned cache classes and SDPA helpers behave as stock, except where
an owned fix changes a result.

cache_equivalence_ops.py runs one fixed operation sequence per cache class
(``update_and_fetch`` at widths 1 and N, ``make_mask``, ``trim``, the state
and meta_state round trip, ``from_state``, ``extract``, ``merge``,
``filter``, ``extend``, ``prepare``, ``finalize`` and ``to_quantized``),
over the module helpers and over the SDPA helpers, in three children on
the CPU device: one on gmlx.cache.kv_cache and gmlx.cache.attention, and
one each on stock mlx-vlm and mlx-lm that never import gmlx. The owned run
must equal the mlx-vlm run in every result, a raised exception included,
except in the records OWNED_FIXES names: there an owned fix changes the
result, and the two runs must differ. Against mlx-lm it must agree in
every other method whose body the two packages share.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cache_equivalence_ops import GROUPS

DRIVER = Path(__file__).with_name("cache_equivalence_ops.py")

# Methods whose mlx-lm 0.31.3 body is identical to the mlx-vlm 0.7.3 one.
# A class that does not define a method inherits the _BaseCache entry.
LM_SAME = {
    "_BaseCache": {"empty", "from_state", "is_trimmable", "meta_state",
                   "meta_state.setter", "nbytes", "size", "state",
                   "state.setter"},
    "ConcatenateKVCache": {"__init__", "empty", "is_trimmable", "make_mask",
                           "nbytes", "state", "update_and_fetch"},
    "QuantizedKVCache": {"__init__", "empty", "is_trimmable", "make_mask",
                         "meta_state", "meta_state.setter", "nbytes", "state",
                         "state.setter", "trim", "update_and_fetch"},
    "KVCache": {"__init__", "empty", "is_trimmable", "make_mask", "merge",
                "nbytes", "size", "to_quantized", "trim", "update_and_fetch"},
    "RotatingKVCache": {"__init__", "_temporal_order", "_trim",
                        "_update_concat", "_update_in_place", "empty",
                        "is_trimmable", "make_mask", "merge", "meta_state",
                        "meta_state.setter", "nbytes", "size", "state.setter",
                        "to_quantized", "trim", "update_and_fetch"},
    "ArraysCache": {"__getitem__", "__init__", "__setitem__", "extend",
                    "filter", "finalize", "make_mask", "prepare", "state",
                    "state.setter"},
    "ChunkedKVCache": {"__init__", "empty", "is_trimmable", "meta_state",
                       "meta_state.setter", "nbytes", "state", "state.setter",
                       "trim", "update_and_fetch"},
    "CacheList": {"__getitem__", "__init__", "empty", "extend", "extract",
                  "filter", "finalize", "from_state", "is_trimmable", "merge",
                  "meta_state", "meta_state.setter", "nbytes", "prepare",
                  "size", "state", "state.setter", "trim"},
    "BatchKVCache": {"empty", "extend", "extract", "is_trimmable",
                     "make_mask", "nbytes", "prepare", "size", "trim",
                     "update_and_fetch"},
    "BatchRotatingKVCache": {"_temporal_order", "_trim", "_update_concat",
                             "_update_in_place", "empty", "extend", "extract",
                             "finalize", "is_trimmable", "meta_state",
                             "meta_state.setter", "nbytes", "prepare", "size",
                             "state", "state.setter", "to_quantized", "trim"},
}
# Methods each class overrides with a body that differs from mlx-lm, or
# that mlx-lm lacks; the _BaseCache entry does not apply to them.
LM_OWN = {
    "ConcatenateKVCache": {"state.setter", "trim"},
    "KVCache": {"state", "state.setter", "extract"},
    "RotatingKVCache": {"state"},
    "ArraysCache": {"advance", "batch_size", "empty", "extract", "merge",
                    "nbytes", "left_padding", "lengths"},
    "ChunkedKVCache": {"maybe_trim_front"},
    "BatchKVCache": {"filter", "finalize", "merge", "state", "state.setter",
                     "batch_size", "is_single_row"},
    "BatchRotatingKVCache": {"filter", "make_mask", "merge",
                             "update_and_fetch", "batch_size",
                             "is_single_row"},
}
# Record names that call more than one method.
LM_CALLS = {"from_state": {"from_state", "state.setter", "meta_state.setter"},
            "state.setter": {"state.setter", "meta_state.setter"}}
# The mlx-lm body of these calls leaves different state behind (valid
# tokens dropped from the front, left padding subtracted in place, stale
# keys kept past a trim), so each group is compared up to that call.
LM_DIVERGES = {"ChunkedKVCache": "maybe_trim_front", "ArraysCache": "advance",
               "ConcatenateKVCache": "trim"}
# Module helpers both packages define with one body. The two-row GQA
# quantized SDPA is left out: mlx-lm broadcasts its pad mask wrongly.
LM_FUNCTIONS = {"create_causal_mask", "create_attention_mask.cache",
                "dynamic_roll", "make_prompt_cache", "sdpa", "quantized_sdpa",
                "create_attention_mask.base", "create_ssm_mask"}
LM_GROUPS = ("KVCache", "RotatingKVCache", "RotatingKVCache keep",
             "ConcatenateKVCache", "QuantizedKVCache 4", "QuantizedKVCache 8",
             "ChunkedKVCache", "BatchKVCache", "BatchRotatingKVCache",
             "ArraysCache", "CacheList", "functions", "sdpa")
# Records whose owned result differs from stock on purpose, per group: the
# record an owned fix changes, and the later records that carry its state.
OWNED_FIXES = {
    # from_state seeds _lengths, which the stock restore leaves unset.
    "BatchRotatingKVCache": {"BatchRotatingKVCache/031 from_state"},
    "CacheList": {"CacheList/024 batch from_state"},
}


def _child(mode: str) -> dict:
    r = subprocess.run([sys.executable, str(DRIVER), mode], capture_output=True,
                       text=True, env=dict(os.environ), timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def runs():
    return {mode: _child(mode) for mode in ("owned", "vlm", "lm")}


def _rows(run: dict, group: str) -> dict:
    return {row["key"]: row for row in run["rows"] if row["group"] == group}


def _lm_same(cls: str, method: str) -> bool:
    if not cls:
        return method in LM_FUNCTIONS
    if cls not in LM_SAME:
        return False
    same, own = LM_SAME[cls], LM_OWN.get(cls, set())
    return all(m in same or (m in LM_SAME["_BaseCache"] and m not in own)
               for m in LM_CALLS.get(method, {method}))


def test_children_load_one_package_each(runs):
    # The owned modules import nothing from mlx-vlm or mlx-lm, and the
    # stock children never see gmlx.
    assert runs["owned"]["loaded"] == ["gmlx"]
    assert runs["owned"]["cache_module"] == "gmlx.cache.kv_cache"
    assert runs["vlm"]["loaded"] == ["mlx_vlm"]
    assert runs["lm"]["loaded"] == ["mlx_lm"]


@pytest.mark.parametrize("group", GROUPS)
def test_owned_matches_mlx_vlm(runs, group):
    own, vlm = _rows(runs["owned"], group), _rows(runs["vlm"], group)
    assert list(own) == list(vlm)
    assert len(own) >= 9
    diff = {key for key in own if own[key]["full"] != vlm[key]["full"]}
    assert diff == OWNED_FIXES.get(group, set()), sorted(diff)[:10]


@pytest.mark.parametrize("group", LM_GROUPS)
def test_owned_matches_mlx_lm_where_bodies_agree(runs, group):
    own, lm = _rows(runs["owned"], group), _rows(runs["lm"], group)
    assert list(own) == list(lm)
    compared, diff = 0, []
    for key, row in own.items():
        if LM_DIVERGES.get(row["cls"]) == row["method"]:
            break
        if (row["method"] == "__dict__" or key in OWNED_FIXES.get(group, ())
                or not _lm_same(row["cls"], row["method"])):
            continue
        compared += 1
        if row["reduced"] != lm[key]["reduced"]:
            diff.append(key)
    assert compared >= 9
    assert not diff, diff[:10]


def test_sdpa_cases_cover_dtypes_masks_sinks_and_bits(runs):
    keys = [row["key"].split(" ", 1)[1] for row in runs["owned"]["rows"]
            if row["group"] == "sdpa"]
    for dtype in ("float16", "bfloat16"):
        for mask in ("none", "causal", "bool", "additive"):
            assert f"{dtype} L4 {mask}" in keys
            assert f"{dtype} L4 {mask} sinks" in keys
            for bits in (4, 8):
                assert f"{dtype} L1 bits{bits} {mask}" in keys
