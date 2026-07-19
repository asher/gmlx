"""Typed readers over GGUF KV metadata, shared by config synthesis, tokenizer
building, and the loader.

Dual-mode: ``meta`` is either a decoded GGUF KV metadata dict (from
``mlx_kquant.load_gguf``: scalars are plain int/str, arrays are lists) or a
gguf-py ``GGUFReader``. A reader is detected by its ``.fields`` attribute; in
that mode the value lives at ``f.parts[f.data[0]]`` (the ``[0]`` element),
strings are UTF-8 byte arrays, and array fields index ``f.data`` per element -
which is also why ``f.parts[f.data[0]][0]`` yields the first element for
array-typed fields.
"""

from __future__ import annotations


def is_reader(meta) -> bool:
    return hasattr(meta, "fields")


def scalar(meta, key):
    """The scalar value for ``key``, the first element if it's an array, or
    None if absent - matching gguf-py's first-element semantics."""
    if is_reader(meta):
        f = meta.fields.get(key)
        return None if f is None else f.parts[f.data[0]][0]
    v = meta.get(key)
    if v is None:
        return None
    return v[0] if isinstance(v, list) and v else (None if isinstance(v, list) else v)


def read_int(meta, key: str) -> int | None:
    s = scalar(meta, key)
    return None if s is None else int(s)


def read_float(meta, key: str) -> float | None:
    s = scalar(meta, key)
    return None if s is None else float(s)


def read_bool(meta, key: str) -> bool | None:
    s = scalar(meta, key)
    return None if s is None else bool(s)


def read_string(meta, key: str) -> str | None:
    if is_reader(meta):
        f = meta.fields.get(key)
        return None if f is None else bytes(f.parts[f.data[0]]).decode("utf-8")
    v = meta.get(key)
    if v is None:
        return None
    return v if isinstance(v, str) else str(v)


def _read_array(meta, key: str, cast) -> list | None:
    """Array of ``cast`` values for ``key``, a one-element list for a scalar,
    or None if absent."""
    if is_reader(meta):
        f = meta.fields.get(key)
        return None if f is None else [cast(f.parts[i][0]) for i in f.data]
    v = meta.get(key)
    if v is None:
        return None
    return [cast(x) for x in v] if isinstance(v, list) else [cast(v)]


def read_int_array(meta, key: str) -> list[int] | None:
    return _read_array(meta, key, int)


def read_bool_array(meta, key: str) -> list[bool] | None:
    return _read_array(meta, key, bool)


def read_float_array(meta, key: str) -> list[float] | None:
    return _read_array(meta, key, float)


def read_str_array(meta, key: str) -> list[str] | None:
    if is_reader(meta):
        f = meta.fields.get(key)
        if f is None:
            return None
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    v = meta.get(key)
    if v is None:
        return None
    return list(v) if isinstance(v, list) else [str(v)]


def array_len(meta, key: str) -> int | None:
    if is_reader(meta):
        f = meta.fields.get(key)
        return None if f is None else len(f.data)
    v = meta.get(key)
    if v is None:
        return None
    return len(v) if isinstance(v, list) else 1


def first_nonzero_int(meta, key: str) -> int | None:
    """First non-zero int for ``key`` (per-layer arrays like head_count_kv use
    0 to mark non-attention layers); falls back to the first element."""
    vals = read_int_array(meta, key)
    if not vals:
        return None
    return next((x for x in vals if x > 0), vals[0])
