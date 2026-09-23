"""The on-disk cache: shard fields and dtypes, the row sidecar, the manifest,
the atomic writer and the validator. mlx-kld reads the same field names by
convention."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import FORMAT_VERSION, FRAME_KINDS, GB, NEG_INF
from .corpus import same_reply

# ---------------------------------------------------------------------------
# cache format
# ---------------------------------------------------------------------------

SHARD_FIELDS = ("top_k_log_softmax", "top_k_indices", "token_ids",
                "attention_mask", "token_end_byte", "onpath_log_p",
                "onpath_mask", "tail_log_mass", "log_boundary_mass",
                "text_bytes", "text_offsets")


def bytes_per_position(K: int, floor: bool = False, text_bytes: float = 4.4,
                       routes_bytes: float = 0.0, hidden_bytes: float = 0.0) -> float:
    return 6 * K + 22 + (4 if floor else 0) + text_bytes + routes_bytes + hidden_bytes


def estimate_cache_bytes(n_tokens: int, K: int, floor: bool = False, routes_bytes: float = 0.0,
                         hidden_bytes: float = 0.0) -> float:
    return n_tokens * bytes_per_position(K, floor, routes_bytes=routes_bytes, hidden_bytes=hidden_bytes)


# ---------------------------------------------------------------------------
# MoE routes: the `routes` field and the gmlx record/replay seam
# ---------------------------------------------------------------------------

ROUTES_FIELD = "routes"
HIDDEN_FIELD = "hidden"   # [B, L, dim] float16 sketch of the teacher's final hidden state


def routes_dtype(n_experts: int):
    """uint8 when every expert id fits, uint16 otherwise."""
    return np.uint8 if n_experts <= 256 else np.uint16


def install_route_recording(model):
    """(recorder, reason): a gmlx RouteRecorder hooked on every MoE block of
    ``model``, or (None, why) when gmlx lacks the seam or the model has no
    supported MoE block."""
    try:
        from gmlx.stream import moe_routes as _mr
    except ImportError:
        return None, "gmlx has no stream.moe_routes (route record and replay seam)"
    rec = _mr.install_moe_route_record(model)
    if not rec.layers:
        _mr.clear_moe_route_controls(model)
        return None, "no supported MoE block found (dense model, or an unsupported router)"
    missing = sorted(set(_mr.moe_layers(model)) - set(rec.layers))
    if missing:
        # a routes field over some layers would fail the replay's layer
        # check, and eval would score without replay. The recorder comes
        # off again, or every later forward would feed it
        _mr.clear_moe_route_controls(model)
        return None, f"route recording unsupported on MoE layers {missing}"
    return rec, ""


def take_routes_blt(rec) -> np.ndarray:
    """The recorder's routes as [B, L, n_moe, k] (the shard layout)."""
    r = rec.take()  # [n_moe, B, T, k]
    return np.ascontiguousarray(np.transpose(r, (1, 2, 0, 3)))


class pin_routes:
    """Context manager: replay ``routes_blt`` ([B, L, n_moe, k], or [L, n_moe,
    k] for one row) through ``model``'s MoE blocks for the forwards inside
    the block; the caller sets ``.replay.offset`` per chunk."""

    def __init__(self, model, routes_blt: np.ndarray, layers: list[int], n_experts: int | None = None):
        from gmlx.stream.moe_routes import RouteReplay
        r = np.asarray(routes_blt)
        if r.ndim == 3:
            r = r[None]
        self.model = model
        self.replay = RouteReplay(np.transpose(r, (2, 0, 1, 3)).astype(np.int32), list(layers),
                                  n_experts=n_experts)

    def __enter__(self):
        from gmlx.stream.moe_routes import install_moe_route_replay
        install_moe_route_replay(self.model, self.replay)
        return self.replay

    def __exit__(self, *exc):
        from gmlx.stream.moe_routes import clear_moe_route_controls
        clear_moe_route_controls(self.model)
        return False


def routing_block(manifest: dict) -> dict | None:
    return (manifest.get("gmlx_distill") or {}).get("routing")


def replay_layers_for(model, manifest: dict) -> list[int] | None:
    """The manifest's MoE layer list when ``model`` carries the same layers
    in the same order with the same number of experts behind each (the
    cache's own teacher), else None. A layer whose expert count cannot be
    read is not compared."""
    rb = routing_block(manifest)
    if not rb:
        return None
    try:
        from gmlx.stream.moe_routes import moe_expert_counts, moe_layers
    except ImportError:
        return None
    inner = getattr(model, "language_model", model)
    if moe_layers(inner) != list(rb["moe_layers"]):
        return None
    n = rb.get("n_experts")
    if n is not None and any(c is not None and c != int(n) for c in moe_expert_counts(inner)):
        return None
    return list(rb["moe_layers"])


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Temp file in the same directory, fsync, os.replace, parent fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def write_json_atomic(path: Path, obj: Any) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def teacher_fingerprint(cache_dir: Path) -> dict | None:
    """The size and content hash of the teacher a cache was made with, the
    teacher block of the run fingerprint in progress.json; None when the
    cache has no progress.json or it records none."""
    try:
        run = read_json(Path(cache_dir) / "progress.json").get("run") or {}
    except (OSError, ValueError, AttributeError):
        return None
    t = run.get("teacher") if isinstance(run, dict) else None
    return dict(t) if isinstance(t, dict) else None


def free_bytes(path: Path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


@dataclass
class RowMeta:
    row_id: int
    doc_id: str
    window: int
    n_tokens: int
    source: str = "human"
    generator_id: str = ""
    suffix_start_byte: int = 0
    prefix_n_tokens: int = 0
    frame: str = ""
    messages: list | None = None
    spans: list | None = None
    student_messages: list | None = None
    content_start: int | None = None
    zero_width: list | None = None      # token indices that span no bytes (a dummy prefix after a special)
    turns: int | None = None            # per-turn rows: how many turns the conversation has, window counts from 0

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if d["messages"] is None:
            d.pop("frame")
            d.pop("messages")
            d.pop("spans")
        if d.get("student_messages") is None:
            d.pop("student_messages", None)
        if d.get("zero_width") is None:
            d.pop("zero_width", None)
        if d.get("content_start") is None:
            d.pop("content_start", None)
        if d.get("turns") is None:
            d.pop("turns", None)
        return d


def pack_shard(rows: list[dict[str, np.ndarray]], texts: list[bytes], K: int,
               floor: bool) -> dict[str, np.ndarray]:
    """Pad B per-row reductions to the shard max L and pack the texts.

    Each row dict carries 1-D per-position arrays (length n_tokens) named
    as in SHARD_FIELDS plus [n_tokens, K] top_k arrays. Pads: top-K values
    -inf and indices -1 in the excluded region, masks False, offsets equal
    to the row's last offset, tails and boundary masses 0."""
    B = len(rows)
    L = max(int(r["token_ids"].shape[0]) for r in rows)
    out = {
        "top_k_log_softmax": np.full((B, L, K), NEG_INF, dtype=np.float16),
        "top_k_indices": np.full((B, L, K), -1, dtype=np.int32),
        "token_ids": np.zeros((B, L), dtype=np.int32),
        "attention_mask": np.zeros((B, L), dtype=bool),
        "token_end_byte": np.zeros((B, L), dtype=np.uint32),
        "onpath_log_p": np.zeros((B, L), dtype=np.float32),
        "onpath_mask": np.zeros((B, L), dtype=bool),
        "tail_log_mass": np.zeros((B, L), dtype=np.float32),
        "log_boundary_mass": np.zeros((B, L), dtype=np.float32),
    }
    if floor:
        out["floor_kld"] = np.zeros((B, L), dtype=np.float32)
    if ROUTES_FIELD in rows[0]:
        r0 = rows[0][ROUTES_FIELD]
        out[ROUTES_FIELD] = np.zeros((B, L) + r0.shape[1:], dtype=r0.dtype)
    if HIDDEN_FIELD in rows[0]:
        h0 = rows[0][HIDDEN_FIELD]
        out[HIDDEN_FIELD] = np.zeros((B, L, h0.shape[-1]), dtype=np.float16)
    for b, r in enumerate(rows):
        n = int(r["token_ids"].shape[0])
        if ROUTES_FIELD in out:
            out[ROUTES_FIELD][b, :n] = r[ROUTES_FIELD]
        if HIDDEN_FIELD in out:
            out[HIDDEN_FIELD][b, :n] = r[HIDDEN_FIELD]
        out["top_k_log_softmax"][b, :n] = r["top_k_log_softmax"]
        out["top_k_indices"][b, :n] = r["top_k_indices"]
        out["token_ids"][b, :n] = r["token_ids"]
        out["attention_mask"][b, :n] = True
        out["token_end_byte"][b, :n] = r["token_end_byte"]
        out["token_end_byte"][b, n:] = r["token_end_byte"][-1]
        out["onpath_log_p"][b, :n] = r["onpath_log_p"]
        out["onpath_mask"][b, :n] = r["onpath_mask"]
        out["tail_log_mass"][b, :n] = r["tail_log_mass"]
        out["log_boundary_mass"][b, :n] = r["log_boundary_mass"]
        if floor:
            out["floor_kld"][b, :n] = r["floor_kld"]
    offs = np.zeros(B + 1, dtype=np.int64)
    for b, t in enumerate(texts):
        offs[b + 1] = offs[b] + len(t)
    out["text_bytes"] = np.frombuffer(b"".join(texts), dtype=np.uint8).copy() \
        if texts else np.zeros(0, dtype=np.uint8)
    out["text_offsets"] = offs
    return out


def save_safetensors_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    from safetensors.numpy import save
    return save({k: np.ascontiguousarray(v) for k, v in arrays.items()})


def load_shard(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(str(path))


def shard_texts(shard: dict[str, np.ndarray]) -> list[bytes]:
    tb = shard["text_bytes"].tobytes()
    offs = shard["text_offsets"]
    return [tb[int(offs[i]):int(offs[i + 1])] for i in range(len(offs) - 1)]


def read_rows_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class ShardWriter:
    """Atomic shard writer with progress.json. The caller hands it packed
    arrays; it never touches the model."""

    def __init__(self, cache_dir: Path, K: int, floor: bool,
                 max_disk_gb: float | None = None):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.K = K
        self.floor = floor
        self.max_disk_bytes = None if max_disk_gb is None else max_disk_gb * GB
        self.progress_path = self.dir / "progress.json"
        self.progress = read_json(self.progress_path) if self.progress_path.exists() \
            else {"shards": [], "tokens": 0, "bytes": 0, "wall_s": 0.0,
                  "min_step": None, "trunk_chunk": None,
                  "bytes_per_v_element": None, "format_version": FORMAT_VERSION}

    @property
    def n_done(self) -> int:
        return len(self.progress["shards"])

    def shard_path(self, i: int) -> Path:
        return self.dir / f"batch-{i:05d}.safetensors"

    def rows_path(self, i: int) -> Path:
        return self.dir / f"rows-{i:05d}.jsonl"

    def write(self, i: int, arrays: dict[str, np.ndarray], rows: list[RowMeta],
              wall_s: float, step: int | None = None, trunk_chunk: int | None = None,
              max_id: int | None = None) -> dict:
        data = save_safetensors_bytes(arrays)
        if self.max_disk_bytes is not None and self.progress["bytes"] + len(data) > self.max_disk_bytes:
            raise RuntimeError(
                f"--max-disk-gb {self.max_disk_bytes / GB:.1f} would be exceeded "
                f"by shard {i} ({(self.progress['bytes'] + len(data)) / GB:.2f} GB)")
        if free_bytes(self.dir) < 2 * len(data):
            raise RuntimeError(f"free space under two shards ({free_bytes(self.dir) / GB:.2f} GB)")
        write_bytes_atomic(self.shard_path(i), data)
        rows_data = ("".join(json.dumps(r.as_dict(), sort_keys=True) + "\n" for r in rows)).encode()
        write_bytes_atomic(self.rows_path(i), rows_data)
        entry = {"index": i, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                 "rows_sha256": hashlib.sha256(rows_data).hexdigest(),
                 "rows": len(rows), "tokens": int(sum(r.n_tokens for r in rows)),
                 "wall_s": wall_s}
        self.progress["shards"].append(entry)
        self.progress["tokens"] += entry["tokens"]
        self.progress["bytes"] += entry["bytes"]
        self.progress["wall_s"] += wall_s
        if step is not None:
            ms = self.progress["min_step"]
            self.progress["min_step"] = step if ms is None else min(ms, step)
        if trunk_chunk is not None:
            self.progress["trunk_chunk"] = trunk_chunk
        if max_id is not None:
            # the largest token id any shard holds, what a student head
            # must be wider than to gather the cache's ids
            self.progress["max_top_k_id"] = max(int(max_id), int(self.progress.get("max_top_k_id") or -1))
        write_json_atomic(self.progress_path, self.progress)
        return entry

    def set_constant(self, key: str, value: Any) -> None:
        self.progress[key] = value
        write_json_atomic(self.progress_path, self.progress)

    def verified_shards(self) -> int:
        """Shards whose file and rows sidecar exist with the recorded
        sha256 (a sidecar written before the sidecar hash existed is taken
        as it is); progress is truncated to that prefix. Resume continues
        after it."""
        good = []
        for e in self.progress["shards"]:
            p = self.shard_path(e["index"])
            rp = self.rows_path(e["index"])
            if p.exists() and rp.exists() and sha256_file(p) == e["sha256"] \
                    and (not e.get("rows_sha256") or sha256_file(rp) == e["rows_sha256"]):
                good.append(e)
            else:
                break
        if len(good) != len(self.progress["shards"]):
            self.progress["shards"] = good
            self.progress["tokens"] = int(sum(e["tokens"] for e in good))
            self.progress["bytes"] = int(sum(e["bytes"] for e in good))
            self.progress["wall_s"] = float(sum(float(e.get("wall_s", 0.0)) for e in good))
            write_json_atomic(self.progress_path, self.progress)
        return len(good)


def write_manifest(cache_dir: Path, *, teacher_path: str, dataset: str,
                   num_samples: int, max_seq_len: int, seed: int, top_k: int,
                   vocab_size: int, config_vocab_size: int | None,
                   tokenizer_hash: str, batch_size: int, gmlx_distill: dict) -> dict:
    progress = read_json(Path(cache_dir) / "progress.json")
    manifest = {
        "format_version": FORMAT_VERSION,
        "teacher_path": teacher_path,
        "dataset": dataset,
        "num_samples": num_samples,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "top_k": top_k,
        "score_window": [0, max_seq_len],
        "vocab_size": vocab_size,
        "config_vocab_size": config_vocab_size,
        "tokenizer_hash": tokenizer_hash,
        "num_batches": len(progress["shards"]),
        "batch_size": batch_size,
        "logit_dtype": "float16",
        "max_top_k_id": progress.get("max_top_k_id"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gmlx_distill": dict(gmlx_distill, format_version=FORMAT_VERSION,
                             shards=progress["shards"], tokens=progress["tokens"],
                             bytes=progress["bytes"]),
    }
    write_json_atomic(Path(cache_dir) / "manifest.json", manifest)
    return manifest


def manifest_sha256(cache_dir: Path) -> str:
    return sha256_file(Path(cache_dir) / "manifest.json")


def validate_cache(cache_dir: Path, check_sha: bool = True) -> list[str]:
    """Problems found, empty when the cache is valid."""
    cache_dir = Path(cache_dir)
    problems: list[str] = []
    mp = cache_dir / "manifest.json"
    if not mp.exists():
        if (cache_dir / "manifest.invalid.json").exists():
            return ["manifest.json missing, manifest.invalid.json holds the one the validator rejected "
                    "(a resume rewrites it, a bad shard needs a fresh --out)"]
        return ["manifest.json missing (pass incomplete)"]
    manifest = read_json(mp)
    gd = manifest.get("gmlx_distill", {})
    allow_prefix = not gd.get("mlx_kld_compatible", True)
    progress = read_json(cache_dir / "progress.json") if (cache_dir / "progress.json").exists() else None
    by_index = {e["index"]: e for e in (progress or {"shards": []})["shards"]}
    K = manifest["top_k"]
    n_batches = manifest["num_batches"]
    routing = gd.get("routing")
    hidden_blk = gd.get("hidden")
    for i in range(n_batches):
        sp = cache_dir / f"batch-{i:05d}.safetensors"
        rp = cache_dir / f"rows-{i:05d}.jsonl"
        if not sp.exists():
            problems.append(f"shard {i} missing")
            continue
        if not rp.exists():
            problems.append(f"rows sidecar {i} missing")
            continue
        if check_sha:
            e = by_index.get(i)
            if e is None:
                problems.append(f"shard {i} not in progress.json")
            elif sha256_file(sp) != e["sha256"]:
                problems.append(f"shard {i} sha256 mismatch")
                continue
            elif e.get("rows_sha256") and sha256_file(rp) != e["rows_sha256"]:
                problems.append(f"rows sidecar {i} sha256 mismatch")
                continue
        try:
            sh = load_shard(sp)
            rows = read_rows_jsonl(rp)
        except Exception as e:  # noqa: BLE001 - a torn file raises the loader's own type
            problems.append(f"shard {i} unreadable ({e})")
            continue
        B, L = sh["token_ids"].shape
        if len(rows) != B:
            problems.append(f"shard {i}: {len(rows)} rows in sidecar, {B} in shard")
        if sh["top_k_log_softmax"].shape != (B, L, K) or sh["top_k_indices"].shape != (B, L, K):
            problems.append(f"shard {i}: top-K shapes inconsistent with K={K}")
        for f in ("onpath_log_p", "tail_log_mass", "log_boundary_mass"):
            v = sh[f][sh["attention_mask"]]
            if not np.all(np.isfinite(v) | (v == NEG_INF)):
                problems.append(f"shard {i}: {f} has NaN or +inf")
        am = sh["attention_mask"]
        om = sh["onpath_mask"]
        if not np.all(np.isfinite(sh["onpath_log_p"][om])):
            problems.append(f"shard {i}: on-path not finite where onpath_mask")
        if np.any(om & ~am):
            problems.append(f"shard {i}: onpath_mask outside attention_mask")
        tk = sh["top_k_log_softmax"].astype(np.float32)
        valid = am[..., None] & (sh["top_k_indices"] >= 0)
        tk = np.where(valid, tk, 0.0)
        d = np.diff(tk, axis=-1)
        both = valid[..., 1:] & valid[..., :-1]
        if np.any(d[both] > 0):
            problems.append(f"shard {i}: top-K not descending")
        if routing:
            rt = sh.get(ROUTES_FIELD)
            want = (B, L, len(routing["moe_layers"]), int(routing["k"]))
            if rt is None:
                problems.append(f"shard {i}: routes field missing with a routing block in the manifest")
            elif rt.shape != want:
                problems.append(f"shard {i}: routes shape {rt.shape} != {want}")
            else:
                v = rt[am].astype(np.int64)
                if v.size and v.max() >= int(routing["n_experts"]):
                    problems.append(f"shard {i}: routes carry an id >= n_experts {routing['n_experts']}")
                if v.size and np.any(np.sort(v, axis=-1)[..., 1:] == np.sort(v, axis=-1)[..., :-1]):
                    problems.append(f"shard {i}: routes repeat an expert within a position")
        elif ROUTES_FIELD in sh:
            problems.append(f"shard {i}: routes field present without a routing block in the manifest")
        if hidden_blk:
            hv = sh.get(HIDDEN_FIELD)
            want_h = (B, L, int(hidden_blk["dim"]))
            if hv is None:
                problems.append(f"shard {i}: hidden field missing with a hidden block in the manifest")
            elif hv.shape != want_h:
                problems.append(f"shard {i}: hidden shape {hv.shape} != {want_h}")
            elif not np.all(np.isfinite(hv[am].astype(np.float32))):
                problems.append(f"shard {i}: hidden not finite where attention_mask")
        elif HIDDEN_FIELD in sh:
            problems.append(f"shard {i}: hidden field present without a hidden block in the manifest")
        texts = shard_texts(sh)
        for b in range(B):
            n = int(am[b].sum())
            ends = sh["token_end_byte"][b, :n].astype(np.int64)
            r = rows[b] if b < len(rows) else {}
            if n and (np.any(np.diff(ends) < 0) or ends[-1] != len(texts[b])):
                problems.append(f"shard {i} row {b}: byte offsets not monotone or last != text length")
            if n:
                # strictly increasing after the special prefix, except at
                # the indices the row sidecar names as zero width (a dummy
                # prefix after a mid-row special spans no bytes)
                zw = [int(z) for z in (r.get("zero_width") or [])] if r else []
                if any(z < 1 or z >= n or ends[z] != ends[z - 1] for z in zw):
                    problems.append(f"shard {i} row {b}: zero_width names a token that spans bytes")
                keep = np.setdiff1d(np.arange(n), np.asarray(zw, dtype=np.int64))
                e2 = ends[keep]
                spec_prefix = int(np.sum(e2 == 0))
                if np.any(np.diff(e2[max(spec_prefix - 1, 0):]) <= 0):
                    problems.append(f"shard {i} row {b}: byte offsets not strictly increasing")
            if r and (r.get("prefix_n_tokens", 0) or r.get("suffix_start_byte", 0)) and not allow_prefix:
                problems.append(f"shard {i} row {b}: prefix fields nonzero with mlx_kld_compatible true")
            if r and r.get("messages") is not None:
                spans = r.get("spans") or []
                pn = int(r.get("prefix_n_tokens", 0) or 0)
                if not spans or pn < 1 or r.get("frame") not in FRAME_KINDS:
                    problems.append(f"shard {i} row {b}: framed row without spans, frame kind or a frame prefix")
                elif n:
                    om = sh["onpath_mask"][b, :n]
                    if om[:pn - 1].any():
                        problems.append(f"shard {i} row {b}: on-path mask set inside the frame")
                if r.get("frame") == "reply" and len(spans) != 1:
                    problems.append(f"shard {i} row {b}: reply row with {len(spans)} spans")
                st = r.get("student_messages")
                if st is not None:
                    if r.get("frame") not in ("reply", "reply-think"):
                        problems.append(f"shard {i} row {b}: student_messages on a {r.get('frame')} row (reply only)")
                    if not same_reply(r["messages"], st):
                        problems.append(f"shard {i} row {b}: student_messages end on a different reply")
            t = r.get("turns") if r else None
            if t is not None:
                w = r.get("window", 0)
                if not (isinstance(t, int) and not isinstance(t, bool) and isinstance(w, int)
                        and not isinstance(w, bool) and 0 <= w < t):
                    problems.append(f"shard {i} row {b}: window {w!r} outside its turns {t!r}")
            if r and r.get("n_tokens") != n:
                problems.append(f"shard {i} row {b}: n_tokens {r.get('n_tokens')} != {n}")
    return problems


