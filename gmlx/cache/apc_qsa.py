"""APC disk participation for the qwen4exp QSA cache stack.

Served qwen4exp stacks carry QSAKVCache entries (KVCache plus the QSA
indexer key stream ``ik`` and optional mrope ``pos``). The in-memory APC
exact tiers reach them through the state/meta_state contract
(QSAKVCache.from_state), but mlx-vlm's disk shard writer dispatches on its
own cache classes and has no arm for them, so every disk exact store on a
qwen4exp stack raised "unsupported exact-cache entry". The installer below
adds writer/reader arms that carry all streams; a snapshot missing ik would
restore a cache whose offset covers tokens the indexer stream does not
have, which crashes the next QSA update.

Install after install_pooling_apc_support: each installer chains the
method it found at install time.
"""

from __future__ import annotations

import importlib


def install_qsa_apc_support() -> None:
    """Add QSAKVCache arms to the APC disk shard writer and reader.

    Idempotent. A missing upstream symbol raises: silently skipping would
    leave disk APC refusing every qwen4exp store (see gmlx.upstream.seams).
    """
    apc = importlib.import_module("mlx_vlm.apc")
    if getattr(apc, "_kq_qsa_apc", False):
        return
    try:
        disk_cls = apc.DiskBlockStore
        stock_snap = disk_cls._snapshot_exact_cache_entry
        stock_load = disk_cls._load_exact_cache_entry
        _read = apc._read_safetensors_tensor
        stock_dtype_info = apc._safetensors_dtype_info
    except AttributeError as e:
        raise RuntimeError(
            "APC QSA support cannot install: mlx-vlm apc surface "
            f"changed ({e}) - re-audit against the pinned seams "
            "(gmlx.upstream.seams)") from e

    # The stock shard reader maps only BF16/F16/F32 while the writer
    # accepts anything mx.save_safetensors takes (the PLE ArraysCache
    # carries an I64 slot). Extend the reader, not the writer.
    import numpy as np
    import mlx.core as mx
    _int_dtypes = {
        "I64": (np.dtype("<i8"), mx.int64, None),
        "U64": (np.dtype("<u8"), mx.uint64, None),
        "I32": (np.dtype("<i4"), mx.int32, None),
        "U32": (np.dtype("<u4"), mx.uint32, None),
        "I16": (np.dtype("<i2"), mx.int16, None),
        "U16": (np.dtype("<u2"), mx.uint16, None),
        "I8": (np.dtype("<i1"), mx.int8, None),
        "U8": (np.dtype("<u1"), mx.uint8, None),
    }

    def dtype_info(dtype):
        got = stock_dtype_info(dtype)
        if got is not None:
            return got
        return _int_dtypes.get(dtype)

    apc._safetensors_dtype_info = dtype_info

    def _qsa_cls():
        from gmlx.models.qwen4_exp.model import QSAKVCache
        return QSAKVCache

    def snap_entry(self, c, prefix, arrays, metadata):
        if not isinstance(c, _qsa_cls()):
            return stock_snap(self, c, prefix, arrays, metadata)
        off = int(getattr(c, "offset", 0) or 0)
        metadata[f"{prefix}_kind"] = "kq_qsa"
        metadata[f"{prefix}_ratio"] = str(int(c.ratio))
        metadata[f"{prefix}_offset"] = str(off)
        if c.ik is None or off <= 0:
            metadata[f"{prefix}_empty"] = "1"
            return True
        # Exact-tier snapshots are detached clones whose widths equal
        # offset; ckpt disk skeletons stamp offset=p but keep only the
        # unaligned k/v tail (the chain holds the rest) with ik full to
        # p. Write each stream at min(offset, stored width).
        if c.keys is not None and c.keys.shape[2] > 0:
            kw = min(off, int(c.keys.shape[2]))
            arrays[f"{prefix}_k"] = c.keys[..., :kw, :]
            arrays[f"{prefix}_v"] = c.values[..., :kw, :]
        arrays[f"{prefix}_ik"] = c.ik[:, :min(off, int(c.ik.shape[1]))]
        if c.pos is not None:
            arrays[f"{prefix}_pos"] = \
                c.pos[:, :, :min(off, int(c.pos.shape[2]))]
        return True

    def load_entry(
        self,
        path,
        tensor_entries,
        metadata,
        data_start,
        prefix,
        *,
        min_capacity_tokens,
        eval_targets,
    ):
        if metadata.get(f"{prefix}_kind") != "kq_qsa":
            return stock_load(
                self,
                path,
                tensor_entries,
                metadata,
                data_start,
                prefix,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
        try:
            ratio = int(metadata[f"{prefix}_ratio"])
        except (KeyError, TypeError, ValueError):
            return None
        out = _qsa_cls()(ratio)
        if metadata.get(f"{prefix}_empty") == "1":
            return out
        ik_ent = tensor_entries.get(f"{prefix}_ik")
        if ik_ent is None:
            return None
        ik = _read(path, data_start, ik_ent)
        if ik is None:
            return None
        loaded = [ik]
        k_ent = tensor_entries.get(f"{prefix}_k")
        v_ent = tensor_entries.get(f"{prefix}_v")
        if k_ent is not None and v_ent is not None:
            k = _read(path, data_start, k_ent)
            v = _read(path, data_start, v_ent)
            if k is None or v is None:
                return None
            out.keys, out.values = k, v
            loaded.extend([k, v])
        pos_ent = tensor_entries.get(f"{prefix}_pos")
        if pos_ent is not None:
            pos = _read(path, data_start, pos_ent)
            if pos is None:
                return None
            out.pos = pos
            loaded.append(pos)
        out.ik = ik
        # Skeleton entries stamp offset=p over tail-width k/v; shards
        # from the pre-offset writer fall back to the widths on disk.
        try:
            off = int(metadata.get(f"{prefix}_offset", "-1"))
        except (TypeError, ValueError):
            off = -1
        if off < 0:
            off = (int(out.keys.shape[2]) if out.keys is not None
                   else int(ik.shape[1]))
        out.offset = off
        eval_targets.extend(loaded)
        return out

    disk_cls._snapshot_exact_cache_entry = snap_entry
    disk_cls._load_exact_cache_entry = load_entry
    apc._kq_qsa_apc = True
