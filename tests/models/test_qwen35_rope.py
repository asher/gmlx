"""Owned MRoPE apply chain: upstream drift tripwires + mirror parity.

The verbatim copies must stay byte-identical to the pinned mlx-vlm
release. The two free-function mirrors must be exact against the stock
``MRoPERotaryEmbedding`` forwards on every route the qwen3.5 family
reaches: fused Metal apply (2D and 3D position ids), the cos/sin
fallback, partial rotary dims (pass-through tail), and both dtypes.
"""

import ast
import inspect
import os
import re
import textwrap

import mlx.core as mx
import pytest

pytest.importorskip("gmlx._vendor.mlx_vlm.models.rope_utils")

from gmlx._vendor.mlx_vlm.models import rope_utils as _R
from gmlx.models.qwen35.language import Qwen3_5RotaryEmbedding

import gmlx.models.qwen35.rope as rp

_NEEDS_GPU = pytest.mark.skipif(
    bool(os.environ.get("KQUANT_FORCE_CPU")),
    reason="the fused MRoPE apply is Metal-only")

_COPIES = (
    "_interleaved_position_selector",
    "_chunked_position_selector",
    "_selected_mrope_freqs",
    "mrope_position_selector",
    "_selects_frequency_by_position",
    "_is_sectioned_style",
    "_has_mrope_apply_selector",
    "_uses_even_odd_pairing",
    "_needs_even_odd_layout",
    "_pairing_for_style",
    "_mrope_apply_kernel",
    "_mrope_apply_cos_sin",
    "_mrope_apply",
    "_fast_mrope_apply",
    "_compiled_mrope_apply",
    "get_mrope_section",
    "compute_inv_freq",
    "_apply_selected_mrope_frequency_layout",
    "apply_mrope_frequency_layout",
    "compute_mrope_frequencies",
    "rotate_half",
    "rotate_half_even_odd",
    "_apply_rotary_embedding",
    "_apply_interleaved_rotary_pos_emb_axis1",
    "apply_multimodal_rotary_pos_emb",
)

_CONSTANTS = ("_HAS_METAL", "_HALF_SPLIT", "_EVEN_ODD", "_HALF_COS", "_FULL_COS")


def _norm(fn):
    return [
        line.rstrip()
        for line in textwrap.dedent(inspect.getsource(fn)).splitlines()
        if line.strip()
    ]


def _assign_source(mod, name):
    src = inspect.getsource(mod)
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found at top level of {mod.__name__}")


def test_copies_match_upstream_source():
    for name in _COPIES:
        assert _norm(getattr(rp, name)) == _norm(getattr(_R, name)), (
            f"owned copy of {name} drifted from the upstream original"
        )


def test_constants_match_upstream_source():
    for name in _CONSTANTS:
        assert _assign_source(rp, name) == _assign_source(_R, name), (
            f"owned constant {name} drifted from the upstream original"
        )


# ---------------------------------------------------------------------------
# free-function mirror parity vs the stock rotary forwards
# ---------------------------------------------------------------------------


def _emb(dim=16):
    return Qwen3_5RotaryEmbedding(
        dim, max_position_embeddings=2048, base=100000, mrope_section=[2, 1, 1]
    )


def _qk(dtype, head=32):
    mx.random.seed(3)
    q = mx.random.normal((2, 4, 5, head)).astype(dtype)
    k = mx.random.normal((2, 2, 5, head)).astype(dtype)
    return q, k


def _positions(ndim):
    if ndim == 2:
        return mx.tile(mx.arange(5)[None, :], (2, 1))
    return mx.tile(mx.arange(5)[None, None, :], (3, 2, 1))


@_NEEDS_GPU
@pytest.mark.parametrize("ndim", (2, 3))
@pytest.mark.parametrize("dtype", (mx.bfloat16, mx.float32))
def test_fused_apply_parity(ndim, dtype):
    emb = _emb()
    q, k = _qk(dtype)
    pos = _positions(ndim)
    qs, ks = emb.apply_rotary(q, k, pos)
    qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
    mx.eval(qs, ks, qo, ko)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()


@pytest.mark.parametrize("ndim", (2, 3))
def test_cos_sin_and_unfused_apply_parity(ndim):
    emb = _emb()
    q, k = _qk(mx.bfloat16)
    pos = _positions(ndim)
    cs, ss = emb(k, pos)
    co, so = rp.rope_cos_sin(emb, k, pos)
    mx.eval(cs, ss, co, so)
    assert mx.array_equal(cs, co).item() and mx.array_equal(ss, so).item()

    # Force the non-fused route on both arms; the mirrors must agree
    # through the owned interleaved apply.
    emb.fused_apply = False
    try:
        qs, ks = emb.apply_rotary(q, k, pos)
        qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
        mx.eval(qs, ks, qo, ko)
        assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()
    finally:
        emb.fused_apply = True


@_NEEDS_GPU
def test_partial_rotary_pass_through_parity():
    # rotary dim 8 under head dim 32: the tail must pass through
    # untouched on the fused kernel route.
    emb = _emb(dim=8)
    q, k = _qk(mx.bfloat16)
    pos = _positions(3)
    qs, ks = emb.apply_rotary(q, k, pos)
    qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
    mx.eval(qs, ks, qo, ko)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()
    assert mx.array_equal(qs[..., 8:], q[..., 8:]).item()


@_NEEDS_GPU
def test_owned_memo_is_separate():
    """The owned mirror memoizes its compiled apply under an owned attr,
    so a stock-compiled entry can never masquerade as the owned path."""
    emb = _emb()
    q, k = _qk(mx.bfloat16)
    rp.rope_apply_rotary(emb, q, k, _positions(3))
    assert len(emb._gmlx_compiled_apply) == 1
    assert len(emb._compiled_apply) == 0, (
        "owned mirror wrote into the stock compiled-apply memo"
    )


@_NEEDS_GPU
def test_fused_route_is_active_on_the_gpu():
    assert rp.fused_rope_active(_emb())


def test_cpu_default_device_takes_the_cos_sin_route(monkeypatch):
    """The fused apply is a Metal kernel: stock MLX raises when it is
    dispatched while the default device is the CPU. The mirror routes
    around it there and lands on the cos/sin apply."""
    emb = _emb()
    if not emb.fused_apply:
        pytest.skip("no Metal device")

    def _no_kernel(*args, **kwargs):
        raise AssertionError("fused apply dispatched on the CPU device")

    monkeypatch.setattr(rp, "_compiled_mrope_apply", _no_kernel)
    q, k = _qk(mx.bfloat16)
    pos = _positions(3)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not rp.fused_rope_active(emb)
        qo, ko = rp.rope_apply_rotary(emb, q, k, pos)
        emb.fused_apply = False
        qs, ks = emb.apply_rotary(q, k, pos)
        mx.eval(qs, ks, qo, ko)
    finally:
        emb.fused_apply = True
        mx.set_default_device(prev)
    assert mx.array_equal(qs, qo).item() and mx.array_equal(ks, ko).item()


# ---------------------------------------------------------------------------
# every-run guard: the fused kernels' index arithmetic is 64-bit
# ---------------------------------------------------------------------------

# This reads the Metal source each kernel builder generates, for every
# variant it builds, with whitespace normalized; it launches nothing. It
# is a tripwire for the forms these kernels use, not a proof, and it is
# closed: the forms it does not read are listed at the end. It checks:
#
# 1. Each 32-bit integer declared at the start of a statement is one of
#    the names below, all of which the launch grid or the shape bounds,
#    and no auto is declared there.
# 2. These have no sum, difference, product, quotient or remainder of two
#    32-bit operands, and are 64-bit:
#    - each subscript of x, x_out, cos and sin;
#    - each offset added to or taken from one of those names, as in
#      "x + off": the text after the operator, up to the next ";", ")",
#      "," or "]" outside brackets;
#    - each initializer of a declared long, int64_t, ulong, uint64_t or
#      size_t, at the start of a statement;
#    - each right side of = to a name declared so, at the start of a
#      statement.
# 3. A pointer declared from one of those names or from another such
#    pointer, as in "T* p = x + off" or "T* p = &x[i]", is an alias.
#    These have no operation of two 32-bit operands, but may be 32-bit,
#    as an offset from a 64-bit base may be:
#    - each subscript of an alias, and each offset added to or taken
#      from one;
#    - each right side of +=, -= or *= to a 64-bit name, one of the four
#      or an alias, at the start of a statement.
#
# A 64-bit operand is a name declared so, a 64-bit cast (static_cast,
# C-style or functional) of a name or a literal, or an operation with a
# 64-bit operand. A 64-bit cast of anything else is refused, though a
# bound might make it correct, and a 32-bit cast is 32-bit.
#
# It does not read:
# - a subscript of, or an offset from, a name that is neither one of the
#   four nor an alias;
# - an offset written before the name, as in "off + x";
# - a declaration or an assignment that does not start a statement, as
#   in a for header or after an if condition;
# - a declaration whose type is two words, such as unsigned int;
# - the position_ids index, which reads 3 x B x L entries.

_GRID_INTS = frozenset((
    "elem", "half_dim", "q_bsz", "q_heads", "q_len", "q_dim", "slots",
    "work_size", "local", "slot", "tmp", "t", "h", "b", "pass_d",
    "freq_idx", "d", "pair_d", "axis"))
_INT32 = ("int", "uint", "unsigned", "short", "ushort", "int32_t",
          "uint32_t")
_INT64 = ("long", "int64_t", "ulong", "uint64_t", "size_t")
_TOKEN = re.compile(
    r"\s*(static_cast\s*<\s*\w+\s*>|\w+|\S)")


def _kernel_sources():
    """(name, Metal source) of every variant of the three builders."""
    import itertools

    out = []

    def build(mod, fn, *args):
        seen = []
        prev = (mod._HAS_METAL, mx.fast.metal_kernel)
        mod._HAS_METAL = True
        mx.fast.metal_kernel = lambda **k: seen.append(k["source"])
        try:
            getattr(mod, fn).__wrapped__(*args)
        finally:
            mod._HAS_METAL, mx.fast.metal_kernel = prev
        out.append((f"{mod.__name__}.{fn}{args}", seen[0]))

    for mod in (rp, _R):
        for ndim, pairing in itertools.product((2, 3), (_R._EVEN_ODD,
                                                       _R._HALF_SPLIT)):
            build(mod, "_mrope_apply_kernel", 64, ndim, pairing)
    for pairing, cos_ndim, sectioned, layout in itertools.product(
            (_R._EVEN_ODD, _R._HALF_SPLIT), (3, 4), (False, True),
            (_R._HALF_COS, _R._FULL_COS)):
        build(_R, "_rotary_apply_kernel", 64, pairing, cos_ndim, sectioned,
              layout)
    return out


class _Expr:
    """A recursive descent over one index expression: the type of each
    sum and product, int or long, and whether one ran in int."""

    def __init__(self, text: str, wide: set):
        self.toks = _TOKEN.findall(text)
        self.i, self.wide, self.narrow = 0, wide, []

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, want=None):
        tok = self.peek()
        if want is not None and tok != want:
            raise AssertionError(f"expected {want!r} at {tok!r} in "
                                 f"{' '.join(self.toks)}")
        self.i += 1
        return tok

    def parse(self) -> str:
        kind = self.sum()
        assert self.peek() is None, f"trailing {self.toks[self.i:]}"
        return kind

    def binary(self, ops, operand):
        kind = operand()
        while self.peek() in ops:
            op = self.take()
            rhs = operand()
            if kind == rhs == "int":
                self.narrow.append(op)
            kind = "long" if "long" in (kind, rhs) else "int"
        return kind

    def sum(self):
        return self.binary(("+", "-"), self.product)

    def product(self):
        return self.binary(("*", "/", "%"), self.atom)

    def cast(self, to: str) -> str:
        """The type of a cast to ``to`` of the atom that follows. A 64-bit
        cast is of one name or literal."""
        start = self.i
        self.atom()
        if to not in _INT64:
            return "int"
        tok = self.toks[start]
        assert self.i - start == 1 and (
            tok in _GRID_INTS or tok in self.wide or tok.isdigit()), (
            f"a 64-bit cast of an expression, not of a name or a literal, "
            f"in {' '.join(self.toks)}")
        return "long"

    def atom(self):
        tok = self.take()
        m = re.fullmatch(r"static_cast\s*<\s*(\w+)\s*>", tok)
        if m or (tok in _INT64 + _INT32 and self.peek() == "("):
            # static_cast<T>(e) or T(e)
            self.take("(")
            if self.toks[self.i + 1] == ")":
                kind = self.cast(m.group(1) if m else tok)
            else:
                to = m.group(1) if m else tok
                kind = self.sum()
                if to in _INT64:
                    raise AssertionError(
                        f"a 64-bit cast of an expression, not of a name or "
                        f"a literal, in {' '.join(self.toks)}")
                kind = "int"
            self.take(")")
            return kind
        if tok == "(":
            if self.peek() in _INT64 + _INT32 and self.toks[self.i + 1] == ")":
                # (T)e
                to = self.take()
                self.take(")")
                return self.cast(to)
            kind = self.sum()
            self.take(")")
            return kind
        if tok == "-":
            return self.atom()
        if tok.isdigit():
            return "int"
        if tok in self.wide:
            return "long"
        if tok == "x_shape":
            self.take("[")
            self.take()
            self.take("]")
            return "int"
        assert tok in _GRID_INTS, f"{tok!r} is not a checked name"
        return "int"


def _subscripts(src: str, names):
    """The index text of each subscript of ``names`` in ``src``."""
    for m in re.finditer(r"\b(%s)\s*\[" % "|".join(names), src):
        depth, j = 1, m.end()
        while depth:
            depth += {"[": 1, "]": -1}.get(src[j], 0)
            j += 1
        yield m.group(1), src[m.end():j - 1]


def _offsets(src: str, names):
    """The text of each offset added to or taken from ``names`` in
    ``src``: from the operator to the next ";", ")", "," or "]" outside
    brackets."""
    for m in re.finditer(r"(?<![\w.])(%s)\b\s*([+-])(?![+=-])"
                         % "|".join(names), src):
        depth, j = 0, m.end()
        while j < len(src) and not (depth == 0 and src[j] in ";),]"):
            depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(src[j], 0)
            j += 1
        yield m.group(1), m.group(2), src[m.end():j].strip()


_BUFFERS = ("x", "x_out", "cos", "sin")


def _aliases(src: str) -> list:
    """The pointers declared from a buffer or from another such pointer,
    as ``T* p = x + off`` or ``T* p = &x[i]`` declares them."""
    names = set(_BUFFERS)
    ptr = re.compile(r"\*\s*(\w+)\s*=\s*(?:&\s*(\w+)\s*\[|(\w+)\b"
                     r"(?!\s*\[))")
    while True:
        new = {m.group(1) for m in ptr.finditer(src)
               if (m.group(2) or m.group(3)) in names} - names
        if not new:
            return sorted(names - set(_BUFFERS))
        names |= new


def _index_faults(src: str) -> list:
    """What breaks the three rules in one kernel's source."""
    src = " ".join(src.split())
    faults, wide = [], set()
    decl = re.compile(r"(?:^|(?<=[;{}]))\s*(?:const\s+)?(\w+)\s+(\w+)\s*"
                      r"=\s*([^;]*);")
    for m in decl.finditer(src):
        kind, name, init = m.groups()
        if kind in _INT32 and name not in _GRID_INTS:
            faults.append(f"32-bit {kind} {name} = {init}")
        elif kind == "auto":
            faults.append(f"auto {name} = {init}")
        elif kind in _INT64:
            e = _Expr(init, wide)
            try:
                if e.parse() != "long" or e.narrow:
                    faults.append(f"{kind} {name} = {init}")
            except AssertionError as err:
                faults.append(str(err))
            wide.add(name)
    assign = re.compile(r"(?:^|(?<=[;{}]))\s*(\w+)\s*(=|\+=|-=|\*=)(?!=)"
                        r"\s*([^;]*);")
    checked = [(f"{a}[{i}]", i, True) for a, i in _subscripts(src, _BUFFERS)]
    checked += [(f"{a} {op} {i}", i, True)
                for a, op, i in _offsets(src, _BUFFERS)]
    checked += [(f"{name} {op} {rhs}", rhs, op == "=")
                for name, op, rhs in assign.findall(src) if name in wide]
    aliases = _aliases(src)
    if aliases:
        checked += [(f"{a}[{i}]", i, False)
                    for a, i in _subscripts(src, aliases)]
        checked += [(f"{a} {op} {i}", i, False)
                    for a, op, i in _offsets(src, aliases)]
    pointers = set(_BUFFERS) | set(aliases)
    checked += [(f"{name} {op} {rhs}", rhs, False)
                for name, op, rhs in assign.findall(src)
                if name in pointers and op != "="]
    for what, text, whole in checked:
        e = _Expr(text, wide)
        try:
            if (e.parse() != "long" and whole) or e.narrow:
                faults.append(what)
        except AssertionError as err:
            faults.append(str(err))
    return faults


_SOURCES = _kernel_sources()


@pytest.mark.parametrize("name,src", _SOURCES, ids=[n for n, _ in _SOURCES])
def test_fused_kernel_index_arithmetic_is_64_bit(name, src):
    assert _index_faults(src) == []


# ---------------------------------------------------------------------------
# the fused rotary apply against its pure-MLX route
# ---------------------------------------------------------------------------

_ROTARY_CASES = {
    # sections of the 4-D cos/sin, each frequency's axis by its section
    "sectioned": dict(cos_ndim=4, pairing=_R._HALF_SPLIT, sections=(16, 24, 24)),
    # one 4-D slab of cos/sin already selected, read at its (b, t) rows
    "4d preselected": dict(cos_ndim=4, pairing=_R._EVEN_ODD, sections=None),
    # a rotary dim under the head dim, whose tail passes through
    "partial": dict(cos_ndim=3, pairing=_R._HALF_SPLIT, sections=None,
                    layout=_R._FULL_COS, tail=32),
}


@_NEEDS_GPU
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("case", _ROTARY_CASES)
def test_fused_rotary_apply_matches_the_precomputed_route(case, dtype):
    c = _ROTARY_CASES[case]
    B, L, RD = 2, 7, 128
    layout = c.get("layout", _R._HALF_COS)
    shape = (3, B, L, RD) if c["sections"] else (B, L, RD)
    ang = mx.random.uniform(-3.0, 3.0, shape, key=mx.random.key(1))
    cos, sin = mx.cos(ang), mx.sin(ang)
    D = RD + c.get("tail", 0)
    q = mx.random.normal((B, 4, L, D), key=mx.random.key(2)).astype(dtype)
    k = mx.random.normal((B, 2, L, D), key=mx.random.key(3)).astype(dtype)
    fc, fs = (cos[None], sin[None]) if c["cos_ndim"] == 4 and not c[
        "sections"] else (cos, sin)
    fast = _R._maybe_fast_precomputed_rotary(
        q, k, fc, fs, pairing=c["pairing"], cos_layout=layout,
        mrope_section=c["sections"])
    assert fast is not None, "the fused route was not taken"
    ref = _R._precomputed_rotary(
        q, k, cos, sin, pairing=c["pairing"], cos_layout=layout,
        sectioned=c["sections"] is not None,
        mrope_section=None if c["sections"] is None else list(c["sections"]))
    tol = 1e-5 if dtype == mx.float32 else 2e-2
    for a, b in zip(fast, ref):
        assert a.dtype == dtype
        err = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        assert err <= tol, (case, err)
