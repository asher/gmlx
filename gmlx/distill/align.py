"""Cross-tokenizer alignment: the tables built once per tokenizer pair and
the group projection of a teacher top-K onto student groups at a shared
boundary."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gmlx.load.tokenizer import (
    backend,
    byte_decoder,
    eos_ids,
    hf_inner,
    is_bytelevel,
    special_ids,
    token_bytes,
    vocab_map_hash,
    whitespace_start_mask,
)

from .constants import NEG_INF, TABLES_VERSION
from .format import read_json, sha256_file, write_json_atomic
from .tokens import bos_id, identity_pair, specials_digest

# ---------------------------------------------------------------------------
# alignment tables and projection
# ---------------------------------------------------------------------------

def _inverse_byte_decoder() -> dict[int, str]:
    return {b: ch for ch, b in byte_decoder().items()}


def _has_dummy_prefix(tokenizer, tb: list[bytes | None]) -> bool:
    ids = backend(tokenizer).encode("a", add_special_tokens=False).ids
    cat = b"".join(tb[i] or b"" for i in ids)
    return cat == b" a"


def _first_token_of_bytes(tokenizer, tb: list[bytes | None], bytelevel: bool,
                          byte_token_ids: dict[int, int], dummy_prefix: bool,
                          b: bytes) -> int:
    """First token id the tokenizer emits for the byte string b, with no
    special tokens and no dummy prefix. The full pipeline runs when the
    bytes decode and the tokenizer adds no dummy prefix; otherwise the BPE
    model tokenizes the piece directly (byte-chars for byte-level, U+2581
    for SPM), and a non-UTF-8 SPM fragment maps to the <0xNN> id of its
    first byte. -1 when nothing can be emitted."""
    if not b:
        return -1
    be = backend(tokenizer)
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        s = None
    if s is not None and not dummy_prefix:
        ids = be.encode(s, add_special_tokens=False).ids
        return int(ids[0]) if ids else -1
    if bytelevel:
        inv = _inverse_byte_decoder()
        s2 = "".join(inv[x] for x in b)
    else:
        if s is None:
            return int(byte_token_ids.get(b[0], -1))
        s2 = s.replace(" ", "\u2581")
    toks = be.model.tokenize(s2)
    return int(toks[0].id) if toks else -1


def _byte_token_ids(tokenizer, tb: list[bytes | None]) -> dict[int, int]:
    """SPM <0xNN> ids by byte value (empty for byte-level vocabularies)."""
    inner = hf_inner(tokenizer)
    out: dict[int, int] = {}
    toks = inner.convert_ids_to_tokens(list(range(len(inner))))
    for tid, tok in enumerate(toks):
        if tok and tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
            try:
                out.setdefault(int(tok[3:5], 16), tid)
            except ValueError:
                pass
    return out


@dataclass
class Tables:
    """Cache-free projection tables for one (teacher, student) tokenizer pair."""
    v1: np.ndarray            # [V_S] int32 teacher-first token of each student token
    u1: np.ndarray            # [V_T] int32 student-first token of each teacher token
    group_of: np.ndarray      # [V_S] int32 in [0, G)
    target_g: np.ndarray      # [V_T] int32 group or -1
    group_key: np.ndarray     # [G] int32 teacher id keying the group, -1 for the unmapped group
    group_size: np.ndarray    # [G] int32
    nonsingleton_ids: np.ndarray  # [N_ns] int32
    bmask_S: np.ndarray       # [V_S] bool
    own: np.ndarray           # [V_T] bool: a group keyed by v exists
    roles: dict[str, Any]
    teacher_hash: str
    student_hash: str
    V_T: int
    V_S: int
    identity: bool
    t_len: np.ndarray | None = None   # [V_T] int32 bytes each teacher token spells, -1 for a special or hole
    s_len: np.ndarray | None = None   # [V_S] int32 the same for the student

    @property
    def G(self) -> int:
        return int(self.group_size.shape[0])

    def meta(self) -> dict:
        return {"tables_version": TABLES_VERSION, "teacher_hash": self.teacher_hash,
                "student_hash": self.student_hash, "V_T": self.V_T, "V_S": self.V_S,
                "G": self.G, "N_ns": int(self.nonsingleton_ids.shape[0]),
                "identity": self.identity, "roles": self.roles}


def token_lengths(tb: list[bytes | None], V: int) -> np.ndarray:
    """[V] int32 byte length of each token, -1 for a special or a hole."""
    out = np.full(V, -1, dtype=np.int32)
    for i, b in enumerate(tb[:V]):
        if b is not None:
            out[i] = len(b)
    return out


def identity_tables(V: int, bmask: np.ndarray, teacher_hash: str, student_hash: str, *,
                    V_T: int | None = None, t_len: np.ndarray | None = None,
                    s_len: np.ndarray | None = None) -> Tables:
    """Identity tables at the student width V. A teacher head narrower than
    the student's (V_T < V, pad-style surplus ids on the student) keeps its
    ids as a prefix, so the teacher-side arrays run to V_T only."""
    V_T = V if V_T is None else V_T
    if V_T > V:
        raise ValueError(f"identity tables need V_T <= V_S, got V_T={V_T} V_S={V}")
    ar = np.arange(V, dtype=np.int32)
    at = np.arange(V_T, dtype=np.int32)
    return Tables(v1=ar.copy(), u1=at.copy(), group_of=ar.copy(), target_g=at.copy(),
                  group_key=ar.copy(), group_size=np.ones(V, dtype=np.int32),
                  nonsingleton_ids=np.zeros(0, dtype=np.int32), bmask_S=bmask.astype(bool),
                  own=np.ones(V_T, dtype=bool), roles={"identity": True},
                  teacher_hash=teacher_hash, student_hash=student_hash, V_T=V_T, V_S=V,
                  identity=True, t_len=t_len, s_len=s_len)


def build_tables(teacher_tok, student_tok, *, V_T: int | None = None,
                 V_S: int | None = None, teacher_tb=None, student_tb=None) -> Tables:
    """Group projection tables from the two tokenizers alone.

    Groups partition the student vocab by v1[u], the teacher-first token of
    each student token's bytes. Specials map by role only (EOS set -> EOS,
    BOS -> BOS, identical added tokens 1:1); every other special and every
    hole sits in one unmapped group that no teacher token targets."""
    ti, si = hf_inner(teacher_tok), hf_inner(student_tok)
    V_T = V_T or len(ti)
    V_S = V_S or len(si)
    ttb = teacher_tb if teacher_tb is not None else token_bytes(teacher_tok, V_T)
    stb = student_tb if student_tb is not None else token_bytes(student_tok, V_S)
    th, sh = vocab_map_hash(ti), vocab_map_hash(si)
    ident, _why = identity_pair(ti, si)
    if ident and V_T == V_S:
        return identity_tables(V_S, whitespace_start_mask(student_tok, V_S, stb), th, sh,
                               t_len=token_lengths(ttb, V_T), s_len=token_lengths(stb, V_S))

    t_bytelevel, s_bytelevel = is_bytelevel(teacher_tok), is_bytelevel(student_tok)
    t_byte_ids = {} if t_bytelevel else _byte_token_ids(teacher_tok, ttb)
    s_byte_ids = {} if s_bytelevel else _byte_token_ids(student_tok, stb)
    t_dummy, s_dummy = _has_dummy_prefix(teacher_tok, ttb), _has_dummy_prefix(student_tok, stb)

    v1 = np.full(V_S, -1, dtype=np.int32)
    memo: dict[bytes, int] = {}
    for u in range(V_S):
        b = stb[u]
        if b is None:
            continue
        r = memo.get(b)
        if r is None:
            r = _first_token_of_bytes(teacher_tok, ttb, t_bytelevel, t_byte_ids, t_dummy, b)
            memo[b] = r
        v1[u] = r
    u1 = np.full(V_T, -1, dtype=np.int32)
    memo = {}
    for v in range(V_T):
        b = ttb[v]
        if b is None:
            continue
        r = memo.get(b)
        if r is None:
            r = _first_token_of_bytes(student_tok, stb, s_bytelevel, s_byte_ids, s_dummy, b)
            memo[b] = r
        u1[v] = r

    # Role map for specials.
    roles: dict[str, Any] = special_roles(teacher_tok, student_tok)
    t_eos, s_eos = eos_ids(teacher_tok), eos_ids(student_tok)
    t_bos, s_bos = bos_id(teacher_tok), bos_id(student_tok)
    t_special = special_ids(teacher_tok)
    t_str = {int(k): str(v) for k, v in getattr(ti, "added_tokens_decoder", {}).items()}
    s_str = {str(v): int(k) for k, v in getattr(si, "added_tokens_decoder", {}).items()}
    # Student specials get a pseudo teacher key so they form groups.
    if s_eos and t_eos:
        for u in s_eos:
            if u < V_S:
                v1[u] = t_eos[0]
    if s_bos is not None and t_bos is not None and s_bos < V_S and s_bos not in s_eos:
        v1[s_bos] = t_bos
    same_added = []
    for v, s in t_str.items():
        u = s_str.get(s)
        # a special with a role on either side keeps its role: the
        # student's EOS string may exist as a plain added token on the
        # teacher, and mapping it by string would drop the teacher's EOS
        # target
        if u is not None and v < V_T and u < V_S and v not in t_eos and v != t_bos \
                and u not in s_eos and u != s_bos:
            v1[u] = v
            same_added.append([v, u])
    roles["identical_added"] = same_added

    # Groups: distinct v1 keys plus one unmapped group (key -1) for
    # student ids with v1 == -1 (specials without a role, holes).
    keys, inverse = np.unique(v1, return_inverse=True)
    group_key = keys.astype(np.int32)
    group_of = inverse.astype(np.int32)
    G = int(group_key.shape[0])
    if group_key[0] != -1:
        # ensure an unmapped group exists so no teacher token can target it
        group_key = np.concatenate([np.array([-1], dtype=np.int32), group_key])
        group_of = group_of + 1
        G += 1
    group_size = np.bincount(group_of, minlength=G).astype(np.int32)
    nonsingleton_ids = np.nonzero(group_size[group_of] > 1)[0].astype(np.int32)

    key_to_group = {int(k): g for g, k in enumerate(group_key) if k >= 0}
    own = np.zeros(V_T, dtype=bool)
    target_g = np.full(V_T, -1, dtype=np.int32)
    for v in range(V_T):
        g = key_to_group.get(v)
        if g is not None:
            target_g[v] = g
            own[v] = True
            continue
        if v in t_special:
            continue    # specials only by role (handled through key_to_group above)
        u = int(u1[v])
        if u >= 0 and group_key[group_of[u]] >= 0:
            # a student special with no role keys no group; a teacher
            # token spelling it stays dropped rather than targeting group 0
            target_g[v] = group_of[u]
    # every teacher end-of-sequence id (a model can carry an end-of-turn
    # id beside its eos) lands in the student's EOS group by role, as own
    # mass: only the first is a group key, and the rest would otherwise
    # reach the group through a byte prefix or not at all
    eos_group = key_to_group.get(int(t_eos[0])) if t_eos else None
    for v in t_eos:
        if v < V_T and eos_group is not None:
            target_g[v] = eos_group
            own[v] = True
    bmask = whitespace_start_mask(student_tok, V_S, stb)
    return Tables(v1=v1, u1=u1, group_of=group_of, target_g=target_g, group_key=group_key,
                  group_size=group_size, nonsingleton_ids=nonsingleton_ids, bmask_S=bmask,
                  own=own, roles=roles, teacher_hash=th, student_hash=sh, V_T=V_T, V_S=V_S,
                  identity=False, t_len=token_lengths(ttb, V_T), s_len=token_lengths(stb, V_S))


def special_roles(teacher_tok, student_tok) -> dict[str, Any]:
    """The EOS and BOS ids of a pair and the digests of both sides'
    special token strings, the part of a tables artifact the vocab hash
    does not cover (special ids are left out of it), so a tables
    artifact is reused only for a student with the same roles."""
    roles: dict[str, Any] = {"specials": {"teacher": specials_digest(teacher_tok),
                                          "student": specials_digest(student_tok)}}
    t_eos, s_eos = eos_ids(teacher_tok), eos_ids(student_tok)
    t_bos, s_bos = bos_id(teacher_tok), bos_id(student_tok)
    if s_eos and t_eos:
        roles["eos"] = {"teacher": [int(v) for v in t_eos], "student": [int(u) for u in s_eos]}
    if s_bos is not None and t_bos is not None and s_bos not in s_eos:
        roles["bos"] = {"teacher": int(t_bos), "student": int(s_bos)}
    return roles


def same_roles(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ("eos", "bos", "specials"))


def save_tables(dirpath: Path, t: Tables) -> None:
    from safetensors.numpy import save_file
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    tmp = dirpath / "tables.safetensors.tmp"
    try:
        save_file({"v1": t.v1, "u1": t.u1, "group_of": t.group_of, "target_g": t.target_g,
                   "group_key": t.group_key, "group_size": t.group_size,
                   "nonsingleton_ids": t.nonsingleton_ids, "bmask_S": t.bmask_S,
                   "own": t.own, **({"t_len": t.t_len, "s_len": t.s_len}
                                    if t.t_len is not None and t.s_len is not None else {})}, str(tmp))
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dirpath / "tables.safetensors")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # tables.json names the arrays it was written with, so the two files
    # replaced one after the other never pass as a pair when a kill
    # separates them
    write_json_atomic(dirpath / "tables.json",
                      dict(t.meta(), safetensors_sha256=sha256_file(dirpath / "tables.safetensors")))


def load_tables(dirpath: Path) -> Tables:
    """The tables under dirpath; ValueError when a file is missing or
    unreadable, or tables.json does not name the tables.safetensors
    beside it."""
    from safetensors.numpy import load_file
    dirpath = Path(dirpath)
    try:
        m = read_json(dirpath / "tables.json")
        digest = sha256_file(dirpath / "tables.safetensors")
    except (OSError, ValueError) as e:
        raise ValueError(f"the tables under {dirpath} are unreadable ({e})") from None
    if m.get("safetensors_sha256") != digest:
        raise ValueError(f"the tables under {dirpath} are torn (tables.safetensors is not the one "
                         "tables.json names)")
    a = load_file(str(dirpath / "tables.safetensors"))
    return Tables(v1=a["v1"], u1=a["u1"], group_of=a["group_of"], target_g=a["target_g"],
                  group_key=a["group_key"], group_size=a["group_size"],
                  nonsingleton_ids=a["nonsingleton_ids"], bmask_S=a["bmask_S"].astype(bool),
                  own=a["own"].astype(bool), roles=m.get("roles", {}),
                  teacher_hash=m["teacher_hash"], student_hash=m["student_hash"],
                  V_T=m["V_T"], V_S=m["V_S"], identity=bool(m["identity"]),
                  t_len=a.get("t_len"), s_len=a.get("s_len"))


def ends_from_ids(ids: np.ndarray, tb: list[bytes | None], special: set[int]) -> np.ndarray:
    ends = np.zeros(len(ids), dtype=np.int64)
    pos = 0
    for i, t in enumerate(ids):
        if int(t) in special:
            ends[i] = pos
            continue
        b = tb[int(t)]
        if b is None:
            raise ValueError(f"id {t} has no bytes")
        pos += len(b)
        ends[i] = pos
    return ends


@dataclass
class Alignment:
    t_pos: np.ndarray    # [J] teacher positions ending at each shared boundary (with a successor)
    s_pos: np.ndarray    # [J] student positions likewise
    ends: np.ndarray     # [J] byte offsets of the shared boundaries
    n_teacher: int
    n_student: int

    @property
    def J(self) -> int:
        return int(self.t_pos.shape[0])


def shared_boundaries(t_ends: np.ndarray, s_ends: np.ndarray) -> Alignment:
    """Intersection of end offsets over tokens that have a successor.
    Positions are token indices; BOS-style tokens with end 0 count only if
    both sides have one. Duplicate end offsets (specials at 0) keep the last
    token with that end."""
    Lt, Ls = len(t_ends), len(s_ends)
    t_cand = {int(e): i for i, e in enumerate(t_ends[:-1])}   # last wins
    s_cand = {int(e): i for i, e in enumerate(s_ends[:-1])}
    common = sorted(set(t_cand) & set(s_cand))
    t_pos = np.array([t_cand[e] for e in common], dtype=np.int32)
    s_pos = np.array([s_cand[e] for e in common], dtype=np.int32)
    return Alignment(t_pos=t_pos, s_pos=s_pos, ends=np.array(common, dtype=np.int64),
                     n_teacher=Lt, n_student=Ls)


def align_row(teacher_ids: np.ndarray, student_ids: np.ndarray, text_bytes: bytes, *,
              teacher_tb, student_tb, teacher_special: set[int], student_special: set[int],
              teacher_ends: np.ndarray | None = None,
              student_ends: np.ndarray | None = None) -> Alignment:
    """Shared boundaries and chunk spans from two tokenizations of the same
    bytes. Takes ids and bytes only; no cache row in its signature."""
    te = teacher_ends if teacher_ends is not None else ends_from_ids(teacher_ids, teacher_tb, teacher_special)
    se = student_ends if student_ends is not None else ends_from_ids(student_ids, student_tb, student_special)
    if len(te) and te[-1] != len(text_bytes):
        raise ValueError(f"teacher offsets end at {te[-1]}, text has {len(text_bytes)} bytes")
    if len(se) and se[-1] != len(text_bytes):
        raise ValueError(f"student offsets end at {se[-1]}, text has {len(text_bytes)} bytes")
    return shared_boundaries(np.asarray(te), np.asarray(se))


def project_topk(top_log_p: np.ndarray, top_idx: np.ndarray, tables: Tables,
                 Kp: int | None = None) -> dict[str, np.ndarray]:
    """Project [P, K] teacher top-K onto groups at P boundaries.

    Duplicate groups are summed in the log domain by one argsort over the
    P x K entries keyed by boundary * (G + 1) + gid and logaddexp.reduceat.
    Returns gid [P, Kp] (sentinel G at pads), log_p [P, Kp] (-inf pads),
    log_M [P], n_groups [P], and per-boundary mass fractions own, redirect,
    singleton, dropped and capped (all in [0, 1], relative to the captured
    mass M_K; own + redirect + dropped = 1, and capped is the mass of the
    mapped groups a Kp cap left out of the target)."""
    P, K = top_log_p.shape
    G = tables.G
    lp = top_log_p.astype(np.float64)
    valid = (top_idx >= 0) & np.isfinite(lp)
    idx = np.where(valid, top_idx, 0)
    gid = np.where(valid, tables.target_g[idx], -1)
    own = np.where(valid, tables.own[idx], False)
    single = np.where(valid & (gid >= 0), tables.group_size[np.maximum(gid, 0)] == 1, False)
    pw = np.exp(np.where(valid, lp, -np.inf))
    MK = pw.sum(axis=1)
    MKs = np.maximum(MK, 1e-300)
    kept = valid & (gid >= 0)
    dropped = (pw * (valid & ~kept)).sum(axis=1) / MKs
    capped = np.zeros(P)
    own_frac = (pw * (own & kept)).sum(axis=1) / MKs
    redirect_frac = (pw * (~own & kept)).sum(axis=1) / MKs
    single_frac = (pw * (single & kept)).sum(axis=1) / MKs

    rows = np.repeat(np.arange(P), K)
    g = gid.reshape(-1)
    v = lp.reshape(-1)
    m = kept.reshape(-1)
    rows, g, v = rows[m], g[m], v[m]
    key = rows.astype(np.int64) * (G + 1) + g
    order = np.argsort(key, kind="stable")
    key, v = key[order], v[order]
    if key.size:
        starts = np.concatenate([[0], np.nonzero(np.diff(key))[0] + 1])
        sums = np.logaddexp.reduceat(v, starts)
        skey = key[starts]
    else:
        starts = np.zeros(0, dtype=np.int64)
        sums = np.zeros(0)
        skey = np.zeros(0, dtype=np.int64)
    srow = skey // (G + 1)
    sg = (skey % (G + 1)).astype(np.int32)
    n_groups = np.bincount(srow, minlength=P).astype(np.int32)
    if Kp is None:
        Kp = int(n_groups.max()) if P else 1
    Kp = max(Kp, 1)
    out_gid = np.full((P, Kp), G, dtype=np.int32)
    out_lp = np.full((P, Kp), NEG_INF, dtype=np.float32)
    # rank groups within a boundary by mass, descending, so a --kprime cap
    # keeps the heaviest
    if skey.size:
        o2 = np.lexsort((-sums, srow))
        srow2, sg2, sums2 = srow[o2], sg[o2], sums[o2]
        first = np.concatenate([[0], np.nonzero(np.diff(srow2))[0] + 1]) if srow2.size else np.zeros(0, dtype=np.int64)
        rank = np.arange(srow2.size) - np.repeat(first, np.diff(np.concatenate([first, [srow2.size]])))
        keep = rank < Kp
        out_gid[srow2[keep], rank[keep]] = sg2[keep]
        out_lp[srow2[keep], rank[keep]] = sums2[keep].astype(np.float32)
        if np.any(~keep):
            np.add.at(capped, srow2[~keep], np.exp(sums2[~keep]))
    # own, redirect and dropped partition the captured mass whatever the
    # cap: own and redirect are properties of the tokenizer pair, which
    # the align gate reads, and the cap's loss is reported on its own
    capped = capped / MKs
    with np.errstate(divide="ignore"):
        log_M = np.log(np.exp(out_lp.astype(np.float64)).sum(axis=1)).astype(np.float32)
    return {"gid": out_gid, "log_p": out_lp, "log_M": log_M, "n_groups": n_groups,
            "own": own_frac.astype(np.float32), "redirect": redirect_frac.astype(np.float32),
            "singleton": single_frac.astype(np.float32), "dropped": dropped.astype(np.float32),
            "capped": capped.astype(np.float32), "M_K": MK.astype(np.float32)}


def tokenization_bias_check(proj: dict[str, np.ndarray], onpath_gid: np.ndarray,
                            onpath_log_p: np.ndarray, onpath_in_topk: np.ndarray) -> tuple[float, float]:
    """Tokenization-bias diagnostic over the boundaries whose teacher
    on-path token is inside the cached top-K (elsewhere the mass sits in
    the tail bucket and says nothing about the projection). Returns the
    fraction of those boundaries where the projected mass on the student's
    next-token group is at least the teacher's on-path probability, and
    the covered fraction itself. Target >= 0.99 on the first.
    The cache stores top-K log-probs in float16 and the on-path value in
    float32, so the comparison is against the float16 rounding of the
    on-path value with one float16 ulp of slack."""
    gid, lp = proj["gid"], proj["log_p"]
    hit = (gid == onpath_gid[:, None])
    with np.errstate(invalid="ignore"):
        m = np.where(hit, lp, NEG_INF).max(axis=1)
    ref = onpath_log_p.astype(np.float16).astype(np.float32)
    slack = np.abs(ref) * 2.0 ** -10 + 1e-6
    ok = (m >= ref - slack) | (onpath_gid < 0)
    cov = onpath_in_topk.astype(bool)
    n_cov = int(cov.sum())
    return (float(ok[cov].mean()) if n_cov else 1.0), (n_cov / len(ok) if len(ok) else 1.0)

