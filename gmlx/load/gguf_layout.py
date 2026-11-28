# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Read-time rewrites that give newer GGUF layouts the names the remap knows.

``normalize`` runs on the tensors ``load_gguf_wire_bytes`` returns, so every
load path (text, VLM, native MTP head) sees the same names.

Shared-input merges. llama.cpp's converter (``_SHARED_INPUT_MERGES``) writes
two matrices that read the same input as one tensor ``[a | b]`` along the
output rows: ``attn_qkv_gate`` is ``attn_qkv`` then ``attn_gate``,
``ssm_beta_alpha`` is ``ssm_beta`` then ``ssm_alpha``, and on qwen4exp
``hc_{attn,ffn}_down_inject`` is ``hc_*_down`` then ``hc_*_inject``. Each
output row is whole in the wire bytes, so a row slice gives the two original
tensors exactly. Older files keep the split names and pass through.

Looped layouts. A ``sensenova_loop`` file writes the physical layers before
the loop end under their own block, the layers after it under their visit
index, and the NextN head after the last visit. ``fold_sensenova_loop``
renames those blocks to physical indices with the head right after the
last physical layer, the layout of a plain qwen35 file.
"""

from __future__ import annotations

import re

import mlx.core as mx

from .gguf_meta import read_int, read_string

_MERGES = (
    # (merged suffix, first part, second part, header key for the second
    #  part's row count; None splits in half)
    ("attn_qkv_gate", "attn_qkv", "attn_gate", "ssm.inner_size"),
    ("ssm_beta_alpha", "ssm_beta", "ssm_alpha", None),
    ("hc_attn_down_inject", "hc_attn_down", "hc_attn_inject",
     "hyper_connection.count"),
    ("hc_ffn_down_inject", "hc_ffn_down", "hc_ffn_inject",
     "hyper_connection.count"),
)
_MERGE_RE = re.compile(
    r"^blk\.(\d+)\.(" + "|".join(m[0] for m in _MERGES) + r")\.weight$")
_MERGE_PARTS = {m[0]: m[1:] for m in _MERGES}


def _scales_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".scales"


def shared_input_parts(name: str) -> tuple[str, str] | None:
    """The two tensors a merged shared-input tensor ``name`` holds, first
    rows first, or None for any other name."""
    m = _MERGE_RE.match(name)
    if m is None:
        return None
    bid, kind = m.groups()
    first, second, _key = _MERGE_PARTS[kind]
    return f"blk.{bid}.{first}.weight", f"blk.{bid}.{second}.weight"


def shared_input_rows(name: str, total: int, meta: dict,
                      arch: str | None) -> tuple[int, int]:
    """The output rows of the two parts of merged tensor ``name`` with
    ``total`` rows."""
    kind = _MERGE_RE.match(name).group(2)
    first, second, key = _MERGE_PARTS[kind]
    if key is None:
        n_second = total // 2
    else:
        n_second = read_int(meta, f"{arch}.{key}") or 0
    if not (0 < n_second < total) or (key is None and total % 2):
        raise ValueError(
            f"{name}: cannot split {total} rows into {first} and "
            f"{second} ({arch}.{key} = {n_second})")
    return total - n_second, n_second


def split_shared_input(arrays: dict, kquant_meta: dict, meta: dict,
                       shapes: dict, arch: str | None) -> int:
    """Split every merged shared-input tensor in place. Returns the count."""
    merged = [n for n in arrays if _MERGE_RE.match(n)]
    for name in merged:
        a_name, b_name = shared_input_parts(name)
        arr = arrays.pop(name)
        total = arr.shape[-2] if arr.ndim >= 2 else 0
        n_first, n_second = shared_input_rows(name, total, meta, arch)
        arrays[a_name] = arr[..., :n_first, :]
        arrays[b_name] = arr[..., n_first:, :]
        codec = kquant_meta.pop(name, None)
        if codec is not None:
            arrays.pop(_scales_name(name), None)
            for part in (a_name, b_name):
                kquant_meta[part] = codec
                arrays[_scales_name(part)] = mx.zeros((1,), dtype=mx.uint8)
        shape = shapes.pop(name, None)
        if shape is not None:
            shape = list(shape)
            shapes[a_name] = shape[:-1] + [n_first]
            shapes[b_name] = shape[:-1] + [n_second]
    return len(merged)


def sensenova_loop_physical(visit: int, start: int, end: int,
                            count: int) -> int:
    """The physical layer that loop visit ``visit`` runs."""
    width = end - start
    if visit < start:
        return visit
    if visit < start + count * width:
        return start + (visit - start) % width
    return visit - (count - 1) * width


def sensenova_loop_plan(meta: dict) -> tuple[int, int, int, int, int]:
    """``(visits, physical, start, end, count)`` from a sensenova_loop
    header, with llama.cpp's refusals."""
    arch = "sensenova_loop"

    def need(key):
        v = read_int(meta, f"{arch}.{key}")
        if v is None:
            raise ValueError(f"sensenova_loop: missing GGUF field {arch}.{key}")
        return v

    blocks = need("block_count")
    nextn = read_int(meta, f"{arch}.nextn_predict_layers") or 0
    start, end, count = need("loop.start"), need("loop.end"), need("loop.count")
    visits = blocks - nextn
    extra = (count - 1) * (end - start)
    if (nextn > 1 or count < 1 or start >= end or end > visits
            or extra > visits - end):
        raise ValueError(
            f"sensenova_loop: invalid loop [{start}, {end}) x {count} for "
            f"{visits} layer visits")
    return visits, visits - extra, start, end, count


_BLK_RE = re.compile(r"^blk\.(\d+)\.")


def sensenova_loop_block(bid: int, start: int, end: int, count: int) -> int:
    """The physical layer that file block ``bid`` holds (the NextN head
    lands right after the trunk). Raises for a block a looped file never
    writes."""
    extra = (count - 1) * (end - start)
    if bid < end:
        return bid
    if bid >= end + extra:
        return bid - extra
    raise ValueError(
        f"sensenova_loop: block {bid} is a repeated loop visit; the "
        f"file should hold each looped layer once")


def fold_sensenova_loop(arrays: dict, kquant_meta: dict, meta: dict,
                        shapes: dict) -> int:
    """Rename visit-indexed blocks to physical ones. Returns the count."""
    _visits, _physical, start, end, count = sensenova_loop_plan(meta)
    moved = 0
    for table in (arrays, kquant_meta, shapes):
        renames = {}
        for name in table:
            m = _BLK_RE.match(name)
            if m is None:
                continue
            bid = int(m.group(1))
            new = sensenova_loop_block(bid, start, end, count)
            if new != bid:
                renames[name] = f"blk.{new}." + name[m.end():]
        for old, new in renames.items():
            table[new] = table.pop(old)
        if table is arrays:
            moved = sum(1 for n in renames if not n.endswith(".scales"))
    return moved


def loaded_tensor_names(shapes: dict, meta: dict,
                        arch: str | None) -> dict[str, tuple]:
    """What the reader loads each tensor of a file as: tensor name ->
    ``((name as loaded, output rows), ...)``, one entry for most tensors
    and two for a merged shared-input tensor. ``shapes`` holds the file's
    shapes in GGUF order (rows second)."""
    loop = None
    if arch == "sensenova_loop" and read_string(meta, "general.type") != "adapter":
        loop = sensenova_loop_plan(meta)[2:]
    out = {}
    for name, ne in shapes.items():
        loaded = name
        m = _BLK_RE.match(name)
        if loop is not None and m is not None:
            loaded = f"blk.{sensenova_loop_block(int(m.group(1)), *loop)}." + name[m.end():]
        rows = int(ne[1]) if len(ne) >= 2 else None
        parts = shared_input_parts(loaded)
        if parts is None or rows is None:
            out[name] = ((loaded, rows),)
        else:
            out[name] = tuple(zip(parts, shared_input_rows(loaded, rows, meta, arch)))
    return out


def normalize(arrays: dict, kquant_meta: dict, meta: dict, shapes: dict,
              arch: str | None) -> None:
    """All read-time rewrites for ``arch``, in place. A LoRA adapter keeps
    its own names: its header has no layout to fold by."""
    if arch == "sensenova_loop" and read_string(meta, "general.type") != "adapter":
        fold_sensenova_loop(arrays, kquant_meta, meta, shapes)
    split_shared_input(arrays, kquant_meta, meta, shapes, arch)
