"""A drafter binds the target's base LM head, never an adapter's wrapper:
a draft call publishes no row vector, so a wrapped head would raise in rows
mode, and llama.cpp drafts without the adapters too. The source guard holds
every head bind to ``lora_base``. CPU-only."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx import lora_rows
from gmlx.load.modules import LoRAKQuantLinear, lora_base

_ROOT = Path(__file__).resolve().parents[2] / "gmlx"


def _wrapped_head():
    mx.random.seed(0)
    base = nn.Linear(8, 16, bias=False)
    a = mx.random.normal((2, 8)) * 0.1
    b = mx.random.normal((16, 2)) * 0.1
    return base, LoRAKQuantLinear(base, a, b, 1.0)


def test_lora_base_unwraps_only_a_wrapper():
    base, head = _wrapped_head()
    assert lora_base(head) is base and lora_base(base) is base
    assert lora_base(None) is None


def test_the_mtp_drafter_binds_the_base_head_in_rows_mode():
    from gmlx.spec.mtp_drafter import QwenMTPDrafter

    base, head = _wrapped_head()
    target = SimpleNamespace(embed_tokens=nn.Embedding(16, 8), lm_head=head)
    drafter = QwenMTPDrafter.__new__(QwenMTPDrafter)
    nn.Module.__init__(drafter)
    drafter.bind(target)
    x = mx.random.normal((2, 1, 8))
    lora_rows.configure("rows", 1)
    try:
        with pytest.raises(lora_rows.LoraRowsError):
            head(x)                                  # the wrapper needs a publish
        assert mx.array_equal(drafter._lm_head_fn(x), base(x))
    finally:
        lora_rows.configure("static", 1)


def _bare_head_reads(source: str) -> list:
    """Lines in a ``bind`` method that read a target's ``lm_head`` outside a
    ``lora_base(...)`` call."""
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def reads_head(n) -> bool:
        if isinstance(n, ast.Attribute) and n.attr == "lm_head" \
                and isinstance(n.ctx, ast.Load):
            return not (isinstance(n.value, ast.Name) and n.value.id == "self")
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "getattr" and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value == "lm_head"
                and not (isinstance(n.args[0], ast.Name) and n.args[0].id == "self"))

    def in_lora_base(n) -> bool:
        while n in parents:
            n = parents[n]
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "lora_base":
                return True
            if isinstance(n, ast.FunctionDef):
                return False
        return False

    bad = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "bind":
            bad += [n.lineno for n in ast.walk(fn)
                    if reads_head(n) and not in_lora_base(n)]
    return bad


def _drafter_sources() -> list:
    files = list((_ROOT / "spec").rglob("*.py"))
    files += list((_ROOT / "models").glob("*/mtp.py"))
    files += list((_ROOT / "models").glob("*/dspark.py"))
    files += list((_ROOT / "_vendor" / "mlx_vlm" / "speculative").rglob("*.py"))
    return sorted(files)


def test_every_drafter_head_bind_goes_through_lora_base():
    files = _drafter_sources()
    assert len(files) > 10
    bad = {str(f.relative_to(_ROOT)): lines for f in files
           if (lines := _bare_head_reads(f.read_text()))}
    assert not bad, f"lm_head reads in bind outside lora_base: {bad}"


def test_the_guard_finds_a_bare_head_bind():
    sample = (
        "class D:\n"
        "    def bind(self, target_model):\n"
        "        self.head = getattr(target_model, 'lm_head', None)\n"
        "        self.other = target_model.language_model.lm_head\n"
        "        self.ok = lora_base(target_model.lm_head)\n"
        "        self.mine = self.lm_head\n")
    assert _bare_head_reads(sample) == [3, 4]
