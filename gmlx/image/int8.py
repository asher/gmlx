"""The int8 setting of an H3 DiT, and its refusals and pricing by the
model's title, its variant's facts and a header view, which video and
image families share."""

from __future__ import annotations


# The DiT's two int8 properties: its attention, a setting of the loaded
# DiT, and the form of its block linears, a property of its weights that
# a load sets (``file`` runs them as the GGUF stores them).
ATTENTION = ("dense", "int8")
LINEARS = ("file", "int8")
# The one int8 setting a run takes: ``--int8``, an entry's
# ``defaults.int8``, and the ``int8`` field of a job or a step. Each value
# maps to the two properties (``int8_parts``).
INT8 = ("off", "attention", "linears", "all")
_PARTS = {"off": ("dense", "file"), "attention": ("int8", "file"),
          "linears": ("dense", "int8"), "all": ("int8", "int8")}


def int8_parts(value: str) -> tuple[str, str]:
    """The DiT's (attention, linears) of an int8 setting. Raises
    ValueError for a value off ``INT8``, a request's list or object
    among them."""
    if not isinstance(value, str) or value not in _PARTS:
        raise ValueError(f"int8 is {', '.join(INT8[:-1])} or {INT8[-1]}, "
                         f"not {value!r}")
    return _PARTS[value]


def int8_of(dit) -> str:
    """The int8 setting a loaded DiT runs at, from its two properties."""
    parts = (getattr(dit, "attention", "dense"),
             getattr(dit, "linears", "file"))
    return next(v for v in INT8 if _PARTS[v] == parts)


def int8_served(entry: str) -> tuple[str, str]:
    """The int8 values a request may name on an entry at ``entry``: the
    two whose linears match its loaded DiT's."""
    linears = int8_parts(entry)[1]
    a, b = (v for v in INT8 if _PARTS[v][1] == linears)
    return a, b


def int8_request_refusal(entry: str, value: str) -> str | None:
    """Why a job or a step naming int8 ``value`` does not run on an entry
    loaded at ``entry``, or None. The linears are a property of the loaded
    DiT, so a request may change the attention alone."""
    allowed = int8_served(entry)
    if value in allowed:
        return None
    return (f"this model's entry loads its DiT at int8 {entry}, so a "
            f"request names int8 {allowed[0]} or {allowed[1]}, not "
            f"{value}; set the entry's defaults.int8 to run {value}")


def attention_refusal(title: str, variant: str, attention: str, *,
                      sparse: bool = False) -> str | None:
    """Why this install or Mac does not run ``variant`` of the model
    ``title`` with ``attention``, or None. Int8 attention replaces dense
    attention only, not a ``sparse`` variant's, and mlx-kquant's
    ``sdpa_qk8`` runs it, on a GPU with NAX units (M5 and later)."""
    if attention == "dense":
        return None
    import mlx_kquant as kq

    if sparse:
        return (f"{title} {variant} runs video sparse attention, and "
                f"int8 attention applies to dense attention only")
    what = f"{title} {variant} runs with int8 attention"
    if getattr(kq, "sdpa_qk8", None) is None:
        return (f"{what}, and the installed mlx-kquant {kq.__version__} "
                f"has no sdpa_qk8; update mlx-kquant to 0.5.10 or later")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def int8_fallback(sparse: bool) -> str:
    """The int8 settings that keep the file's linears: off alone where a
    ``sparse`` variant refuses int8 attention."""
    return "off" if sparse else "off or attention"


def linears_refusal(title: str, variant: str, linears: str,
                    patterns) -> str | None:
    """Why this install or Mac does not run ``variant`` of the model
    ``title`` with ``linears``, or None. A model converts the linears
    ``patterns`` names, none when empty. mlx-kquant's
    ``int8_quantize_rows`` and ``int8_matmul`` run int8 linears, the
    matmul on a GPU with NAX units (M5 and later), with dense or video
    sparse attention."""
    if linears == "file":
        return None
    import mlx_kquant as kq

    if not patterns:
        return f"gmlx runs {title} with its file's linears only"
    what = f"{title} {variant} runs with int8 linears"
    if (getattr(kq, "int8_matmul", None) is None
            or getattr(kq, "int8_quantize_rows", None) is None):
        return (f"{what}, and the installed mlx-kquant {kq.__version__} "
                f"has no int8_matmul; update mlx-kquant to 0.5.11 or later")
    if not kq.nax_available():
        return (f"{what}, whose kernel needs a GPU with NAX units (M5 or "
                f"newer), and this Mac's GPU has none")
    return None


def int8_refusal(title: str, variant: str, int8: str, patterns, *,
                 sparse: bool = False) -> str | None:
    """Why this install or Mac does not run ``variant`` of the model
    ``title`` at the int8 setting ``int8``, or None: its attention, then
    its linears."""
    attention, linears = int8_parts(int8)
    return (attention_refusal(title, variant, attention, sparse=sparse)
            or linears_refusal(title, variant, linears, patterns))


def int8_matrices(view, strip_prefixes, patterns) -> list:
    """(out, in, file bytes) of each tensor that the int8 linears
    ``patterns`` convert, from a DiT header, its names with or without
    one of ``strip_prefixes``."""
    import fnmatch

    from gmlx.image.memfit import tensor_bytes

    out = []
    for name, t in view.tensors.items():
        for pre in strip_prefixes:
            if name.startswith(pre):
                name = name[len(pre):]
                break
        path, _, kind = name.rpartition(".")
        if kind != "weight" or len(t.shape) != 2 or not any(
                fnmatch.fnmatchcase(path, pat) for pat in patterns):
            continue
        k, n = t.shape
        out.append((int(n), int(k), t.nbytes or tensor_bytes(t.shape,
                                                             t.codec)))
    return out


def int8_delta(view, strip_prefixes, patterns) -> int:
    """The bytes a DiT with int8 linears holds over its file: each
    converted matrix's int8 weight and float32 scale a row, less its file
    bytes, from the header ``view``. Exact, not a fit."""
    return sum(n * k + 4 * n - b for n, k, b in
               int8_matrices(view, strip_prefixes, patterns))


def int8_transient(view, strip_prefixes, patterns) -> int:
    """The bytes a conversion holds beside the loaded DiT: the largest
    converted matrix at float32, its int8 codes and its scales."""
    return max((4 * n * k + n * k + 4 * n for n, k, _ in
                int8_matrices(view, strip_prefixes, patterns)), default=0)
