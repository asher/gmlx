"""A drafter binds the target's base LM head and base token embedding,
never an adapter's wrapper: a draft call publishes no row vector, so a
wrapped head or table would raise in rows mode, and llama.cpp drafts
without the adapters too. The source guard holds every head and embedding
bind to ``lora_base``. CPU-only."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from gmlx import lora_rows
from gmlx.load.modules import LoRAKQuantEmbedding, LoRAKQuantLinear, lora_base

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


# The token embedding names the serve probe knows (bridge_vlm._EMBED_ATTR_NAMES).
_EMBED_NAMES = ("embed_tokens", "embeddings", "wte", "tok_embeddings", "word_embeddings")


_HEAD = ("lm_head",)


def _bare_head_reads(source: str, names=_HEAD) -> list:
    """Lines in a ``bind`` method that read a target's ``lm_head`` (or one
    of ``names``) outside a ``lora_base(...)`` call."""
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def own(v) -> bool:
        # self.x, or for an embedding self.model.x: the drafter's own table.
        while names is not _HEAD and isinstance(v, ast.Attribute):
            v = v.value
        return isinstance(v, ast.Name) and v.id == "self"

    def reads_head(n) -> bool:
        if isinstance(n, ast.Attribute) and n.attr in names \
                and isinstance(n.ctx, ast.Load):
            return not own(n.value)
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "getattr" and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant)
                and n.args[1].value in names and not own(n.args[0]))

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


def test_every_drafter_embedding_bind_goes_through_lora_base():
    from gmlx.serve.bridge_vlm import _EMBED_ATTR_NAMES

    assert set(_EMBED_ATTR_NAMES) == set(_EMBED_NAMES)
    bad = {str(f.relative_to(_ROOT)): lines for f in _drafter_sources()
           if (lines := _bare_head_reads(f.read_text(), _EMBED_NAMES))}
    assert not bad, f"embedding reads in bind outside lora_base: {bad}"


def test_the_guard_finds_a_bare_embedding_bind():
    sample = (
        "class D:\n"
        "    def bind(self, target_model):\n"
        "        self.e = inner.embed_tokens\n"
        "        self.f = getattr(backbone, 'embeddings', None)\n"
        "        self.ok = lora_base(inner.embed_tokens)\n"
        "        self.mine = self.embed_tokens\n"
        "        self.own = self.model.embed_tokens\n")
    assert _bare_head_reads(sample, _EMBED_NAMES) == [3, 4]


# the embedding binds


def _wrapped_embedding():
    base = nn.Embedding(16, 8)
    a = mx.ones((16, 2)) * 0.5
    b = mx.ones((8, 2)) * 0.25
    return base, LoRAKQuantEmbedding(base, a, b, 1.0)


def _bare(cls, **attrs):
    d = cls.__new__(cls)
    nn.Module.__init__(d)
    for k, v in attrs.items():
        setattr(d, k, v)
    return d


def _dflash():
    from gmlx.spec.dflash_drafter import DFlashDrafter
    return _bare(DFlashDrafter, config=SimpleNamespace(input_embedding_scale=1.0))


def _qwen_mtp():
    from gmlx.spec.mtp_drafter import QwenMTPDrafter
    return _bare(QwenMTPDrafter)


def _mtp_base():
    from gmlx._vendor.mlx_vlm.speculative.drafters.mtp_base import (
        AutoregressiveMTPDraftModel,
    )
    return _bare(AutoregressiveMTPDraftModel)


def _qwen3_dflash():
    from gmlx._vendor.mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
        DFlashDraftModel,
    )
    d = _bare(DFlashDraftModel)
    d.embed_tokens = None
    return d


def _bound_table(d):
    for name in ("embed_tokens", "_input_embed"):
        t = getattr(d, name, None)
        if t is not None:
            return t
    raise AssertionError("no bound table")


@pytest.mark.parametrize("make", [_dflash, _qwen_mtp, _mtp_base, _qwen3_dflash],
                         ids=["dflash", "qwen_mtp", "mtp_base", "qwen3_dflash"])
def test_the_drafter_binds_the_base_embedding_in_rows_mode(make):
    base, emb = _wrapped_embedding()
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=emb),
                             lm_head=nn.Linear(8, 16, bias=False))
    d = make()
    d.bind(target)
    ids = mx.array([[1, 5]])
    lora_rows.configure("rows", 1)
    try:
        with pytest.raises(lora_rows.LoraRowsError):
            emb(ids)
        assert _bound_table(d) is base
        assert mx.array_equal(_bound_table(d)(ids), base(ids))
    finally:
        lora_rows.configure("static", 1)


def test_the_qwen3_dflash_bind_takes_the_base_of_a_backbone_table():
    base, emb = _wrapped_embedding()
    target = SimpleNamespace(language_model=SimpleNamespace(
        backbone=SimpleNamespace(embeddings=emb), lm_head=nn.Linear(8, 16, bias=False)))
    d = _qwen3_dflash()
    d.bind(target)
    assert d.embed_tokens is base
