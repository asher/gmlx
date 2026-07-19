"""GGUF directory discovery - header-only classification + id synthesis.

Walks one or more directories, reads each GGUF's *header* (architecture +
``nextn_predict_layers``, never tensor data), and classifies it as a servable
**model**, a VLM **mmproj** companion, or an assistant **drafter**. Synthesizes a
friendly, deterministic id from each model's filename, pairs sibling mmproj files
into VLM entries, and emits a starter YAML config.

Classification (header-only):

- ``general.architecture == "clip"`` (or a ``mmproj*`` filename) -> **mmproj**:
  a VLM companion, not a standalone model.
- arch is an assistant shape (``gemma4_assistant`` / ``gemma4-assistant`` /
  ``gemma4_mtp``, or carries a target-backbone field) -> **drafter**: only paired
  when a model explicitly names it via ``draft_gguf``; never standalone.
- ``<arch>.nextn_predict_layers > 0`` -> a **native-head MTP** model (the drafter
  lives inside the target GGUF; ``speculative: auto`` enables it).
- otherwise -> a plain text **model**.

Reuses :func:`preflight.find_split_shards` (sharded-model collapse) and the
dual-mode GGUF KV readers from :mod:`config_synth`, so the same value access works
on a live ``GGUFReader`` or a plain metadata dict (the latter keeps this CPU-testable).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

from .textfmt import plural_s
from . import profiles as _family_profiles
from .config import DiscoverSpec, ModelCfg
from .config_synth import supported_arches
from .gguf_meta import read_int, read_string
from .preflight import is_first_shard as _is_first_shard
from .preflight import strip_shard_suffix as _strip_shard_suffix

# Assistant-shape drafter arches (a separate GGUF that drafts for a target whose
# hidden size it carries as a "backbone" field). The set is the fast path; the
# backbone-field probe in `_looks_like_drafter` catches future naming.
_DRAFTER_ARCHES = frozenset({"gemma4_assistant", "gemma4-assistant", "gemma4_mtp"})
_BACKBONE_FIELDS = ("backbone_embedding_length", "embedding_length_out",
                    "n_embd_backbone")

# A trailing quant tag on a filename stem: optional UD- (Unsloth) prefix, then a
# K-quant / legacy / IQ / float codec name. Used to strip the tag from the id and
# to disambiguate two ids that collapse to the same base.
_QUANT_TRAILING = re.compile(
    r"[-._]"
    r"(?:UD[-_])?"
    r"(?P<q>"
    r"IQ\d+(?:_[A-Za-z0-9]+)*"
    r"|Q\d+(?:_[A-Za-z0-9]+)*"
    r"|BF16|FP16|F16|FP32|F32|MXFP4|NVFP4"
    r")$",
    re.IGNORECASE,
)
# Markers stripped from a stem before slugifying the id: kind markers
# (mmproj/assistant/...) plus imatrix provenance (mradermacher `i1`, `imatrix`) -
# that's quant provenance, not part of the model name, and leaving it in splits
# one model's quants across two id prefixes (`...instruct.i1-*` vs `...instruct-*`).
_ID_MARKERS = ("mmproj", "assistant", "draft", "mtp", "gguf",
               "imatrix", "imat", "i1")


# Encoder/embedding architectures (not generative decoders). A GGUF on one of
# these - or carrying a pooling_type, or named *embedding* / *reranker* - is an
# embedder/reranker, not a chat model, so it is never a servable model or an
# mmproj target.
_ENCODER_ARCHES = frozenset({
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "roberta",
    "xlm-roberta", "gte", "modernbert",
    "gemma-embedding",   # EmbeddingGemma: gemma3 backbone, runs on the encoder path
})
_RERANK_RE = re.compile(r"(?i)(?:^|[-._])rerank(?:er)?(?:[-._]|$)")
_EMBED_RE = re.compile(r"(?i)(?:^|[-._])embed(?:ding)?(?:[-._]|$)")


@dataclass(frozen=True)
class ClassifiedGguf:
    """The header-only verdict for one GGUF file."""
    path: str               # absolute path
    kind: str               # "model" | "mmproj" | "drafter" | "embedding" | "reranker" | "adapter"
    arch: str | None
    mtp: bool               # native-head MTP (model kind only)
    quant: str | None    # quant tag from the filename, or None
    loadable: bool          # arch builds a model with no hf override (model kind)


# Classification
def _looks_like_drafter(meta, arch: str | None) -> bool:
    if arch in _DRAFTER_ARCHES:
        return True
    if arch and ("assistant" in arch or "_mtp" in arch):
        return True
    if arch:
        for suf in _BACKBONE_FIELDS:
            if read_int(meta, f"{arch}.{suf}") is not None:
                return True
    return False


def _embedding_kind(meta, basename: str, arch: str | None) -> str | None:
    """``"reranker"`` / ``"embedding"`` if this GGUF is an encoder-style retrieval
    model rather than a generative chat model, else ``None``. Signals (any one):
    a *reranker* / *embedding* filename token, an encoder architecture, or a
    ``<arch>.pooling_type`` >= 1 (llama.cpp sets it on embedding conversions;
    ``0`` = none = a normal decoder)."""
    if _RERANK_RE.search(basename):
        return "reranker"
    if _EMBED_RE.search(basename):
        return "embedding"
    if arch in _ENCODER_ARCHES:
        return "embedding"
    if arch and (read_int(meta, f"{arch}.pooling_type") or 0) >= 1:
        return "embedding"
    return None


def _classify_meta(meta, *, basename: str, path: str) -> ClassifiedGguf:
    """Classify from already-read metadata (a live ``GGUFReader`` or a plain dict).
    Split out so the verdict logic is exercised without a real GGUF on disk."""
    arch = read_string(meta, "general.architecture")
    quant = quant_tag(basename)
    ap = os.path.abspath(path)

    if arch == "clip" or basename.lower().startswith("mmproj"):
        return ClassifiedGguf(ap, "mmproj", arch, False, quant, False)
    # A LoRA adapter (`gmlx train` / llama.cpp export) carries its base model's
    # `general.architecture`, so without this check it classifies as a loadable
    # chat model and serves a phantom id that 500s on request.
    if (read_string(meta, "general.type") == "adapter"
            or read_string(meta, "adapter.type") is not None):
        return ClassifiedGguf(ap, "adapter", arch, False, quant, False)
    if _looks_like_drafter(meta, arch):
        return ClassifiedGguf(ap, "drafter", arch, False, quant, False)
    emb = _embedding_kind(meta, basename, arch)
    if emb:
        return ClassifiedGguf(ap, emb, arch, False, quant, False)

    nextn = read_int(meta, f"{arch}.nextn_predict_layers") if arch else None
    mtp = bool(nextn and nextn > 0)
    loadable = bool(arch) and arch in supported_arches()
    return ClassifiedGguf(ap, "model", arch, mtp, quant, loadable)


def classify_gguf(path: str) -> ClassifiedGguf | None:
    """Read ``path``'s header and classify it. ``None`` if it can't be opened/read
    as a GGUF (skipped by the scanner). Cheap on multi-GB files - header only."""
    from .headerscan import scan_gguf
    try:
        kv = scan_gguf(path, include_tensors=False).kv
    except Exception as e:                       # not a GGUF / unreadable
        print(f"[discover] skipping {path}: {e}", file=sys.stderr)
        return None
    return _classify_meta(kv, basename=os.path.basename(path), path=path)


# Header-meta cache - the cheap serve-time answer to "what is this GGUF?"
# (arch / general.name / kind / native-MTP), backing family detection at
# registration and run/chat auto-detection. Two tiers: a process memo, and a
# persistent JSON cache keyed by (mtime, size) so repeat startups and SIGHUP
# re-registers are stat-only even for 30-model configs on slow volumes.
_HEADER_MEMO: dict[str, dict | None] = {}
_HEADER_DISK: dict | None = None


def _header_cache_path() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache_home, "gmlx", "header-meta.json")


def _load_header_cache() -> dict:
    global _HEADER_DISK
    if _HEADER_DISK is None:
        try:
            import json
            with open(_header_cache_path()) as f:
                loaded = json.load(f)
            _HEADER_DISK = loaded if isinstance(loaded, dict) else {}
        except Exception:                        # absent / corrupt: start fresh
            _HEADER_DISK = {}
    return _HEADER_DISK


def _save_header_cache(cache: dict) -> None:
    """Best-effort atomic write; an unwritable cache dir is silently ignored."""
    try:
        import json
        import tempfile
        p = _header_cache_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, p)
    except Exception:
        pass  # best-effort cache write; discovery just re-scans next time


def _read_header(path: str) -> tuple:
    """The slow path behind :func:`header_meta`: open the GGUF and return
    ``(ClassifiedGguf, general.name)``. Raises when unreadable."""
    from .headerscan import scan_gguf
    kv = scan_gguf(path, include_tensors=False).kv
    c = _classify_meta(kv, basename=os.path.basename(path), path=path)
    return c, read_string(kv, "general.name")


def header_meta(path: str) -> dict | None:
    """``{"arch", "name", "kind", "mtp"}`` from a GGUF's header, or ``None``
    when the file is missing/unreadable. Unlike :func:`classify_gguf` this is
    **silent** on failure - registration over a config that references a
    not-yet-pulled file must not spam stderr. A sharded model's configured path
    is its first shard, which carries the full KV block, so a plain read of the
    given path suffices."""
    ap = os.path.abspath(os.path.expanduser(path))
    if ap in _HEADER_MEMO:
        return _HEADER_MEMO[ap]
    try:
        st = os.stat(ap)
    except OSError:
        return None                              # not memoized: it may appear later
    disk = _load_header_cache()
    ent = disk.get(ap)
    if (isinstance(ent, dict) and ent.get("mtime") == int(st.st_mtime)
            and ent.get("size") == st.st_size):
        meta = {k: ent.get(k) for k in ("arch", "name", "kind", "mtp")}
        _HEADER_MEMO[ap] = meta
        return meta
    try:
        c, name = _read_header(ap)
    except Exception:
        _HEADER_MEMO[ap] = None                  # unreadable-as-GGUF is stable
        return None
    meta = {"arch": c.arch, "name": name, "kind": c.kind, "mtp": c.mtp}
    _HEADER_MEMO[ap] = meta
    disk[ap] = {"mtime": int(st.st_mtime), "size": st.st_size, **meta}
    _save_header_cache(disk)
    return meta


def find_mtp_companion(path: str, drafter_arch: str = "deepseek4_mtp_support") -> str | None:
    """Path of an MTP drafter GGUF (arch ``drafter_arch``) sitting in the same
    directory as ``path``, or ``None``. Header-only peeks through
    :func:`header_meta`'s stat-validated cache, so a directory scan costs one
    stat per already-seen sibling. Lexically first match wins on a tie."""
    ap = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(ap)
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return None
    for name in names:
        if not name.endswith(".gguf"):
            continue
        p = os.path.join(parent, name)
        if p == ap:
            continue
        meta = header_meta(p)
        if meta and meta.get("arch") == drafter_arch:
            return p
    return None


def fill_families(cfg) -> None:
    """Fill each config model's sampling ``family`` (profiles.py key) from its
    GGUF header, in place. Shared by serve registration and the run/chat config
    overlay. An explicit YAML ``family:`` wins; a path that doesn't resolve or
    read (not yet pulled, fake test path) silently stays ``None`` (generic
    defaults). No-op when ``server.family_defaults`` is off. Header reads ride
    the stat-validated cache above, so repeat calls are cheap."""
    if not cfg.family_defaults:
        return
    from . import profiles as _profiles
    from .config import ConfigError, resolve_path
    for mc in cfg.models.values():
        if mc.family is not None:
            continue
        try:
            path = resolve_path(mc.path, cfg.model_dirs)
        except ConfigError:
            continue
        meta = header_meta(path)
        if meta and meta.get("arch"):
            mc.family = _profiles.detect_family(meta.get("arch"),
                                                meta.get("name"))


# Id derivation
def quant_tag(filename: str) -> str | None:
    """The trailing quant codec in a filename (``Q4_K_S``, ``Q6_K_L``, ``BF16`` ...),
    or ``None``. Split-shard suffix is stripped first."""
    stem = _strip_ext_and_shard(filename)
    m = _QUANT_TRAILING.search(stem)
    return m.group("q").upper() if m else None    # canonical codec case (BF16, Q4_K_S)


def _strip_ext_and_shard(filename: str) -> str:
    name = _strip_shard_suffix(os.path.basename(filename))
    if name.lower().endswith(".gguf"):
        name = name[:-len(".gguf")]
    return name


def _slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9.]+", "-", s)           # keep dots (qwen3.6); rest -> '-'
    s = re.sub(r"\.{2,}", ".", s)
    return re.sub(r"-{2,}", "-", s).strip("-.")


def derive_id(filename: str) -> tuple[str, str | None]:
    """Derive a friendly id + the quant tag from a filename. The stem has its
    split suffix, trailing quant tag, and kind markers (mmproj/assistant/...)
    removed, then is lowercased + slugified. Deterministic - same name, same id."""
    stem = _strip_ext_and_shard(filename)
    m = _QUANT_TRAILING.search(stem)
    qtag = m.group("q").upper() if m else None
    if m:
        stem = stem[:m.start()]
    for marker in _ID_MARKERS:
        stem = re.sub(rf"(?i)(?:^|[-._]){marker}(?=[-._]|$)", "-", stem)
    return _slug(stem), qtag


# A leading family + bit-width, e.g. Q4 from Q4_K_M, IQ2 from IQ2_XXS, Q8 from
# Q8_0, BF16 from BF16. Used to compress a codec to its compact id form.
_CODEC_FAMILY = re.compile(r"(?i)^([A-Za-z]+\d+)")


def _id_codecs(filename: str) -> tuple[str, str]:
    """Two slug forms of a filename's trailing quant codec, for id synthesis:
    a compact ``base`` (family + bit width only - ``q4``, ``iq2``, ``q8``,
    ``bf16``) and the ``full`` codec (``q4-k-m``). A leading Unsloth ``UD-``
    recipe marker is preserved on both (``ud-q4`` / ``ud-q4-k-xl``), since UD is a
    distinct recipe, not a size variant. ``("", "")`` if the name has no codec."""
    stem = _strip_ext_and_shard(filename)
    m = _QUANT_TRAILING.search(stem)
    if not m:
        return "", ""
    qtag = m.group("q")                                   # Q4_K_M, IQ2_XXS, BF16
    ud = "ud-" if re.search(r"(?i)UD[-_]$", stem[:m.start("q")]) else ""
    fam = _CODEC_FAMILY.match(qtag)
    base = fam.group(1) if fam else qtag                  # Q4_K_M -> Q4
    return ud + _slug(base), ud + _slug(qtag)


# Directory scan
def _iter_gguf_files(root: str, recursive: bool):
    """Yield canonical *.gguf paths under ``root`` (first shard only), sorted."""
    root = os.path.abspath(os.path.expanduser(os.path.expandvars(root)))
    if not os.path.isdir(root):
        print(f"[discover] not a directory: {root}", file=sys.stderr)
        return
    found = []
    if recursive:
        for dirpath, _dirs, files in os.walk(root):
            found += [os.path.join(dirpath, f) for f in files]
    else:
        try:
            found = [os.path.join(root, f) for f in os.listdir(root)]
        except OSError as e:
            # Mirror the recursive branch (os.walk swallows errors): an
            # unreadable dir is skipped with a note, it must not abort serve/init.
            print(f"[discover] cannot read {root}: {e.strerror}", file=sys.stderr)
            return
    for p in sorted(found):
        if not p.lower().endswith(".gguf"):
            continue
        if not _is_first_shard(os.path.basename(p)):  # non-first shard -> skip
            continue
        yield p


def scan_dirs(
    specs,
    model_dirs,
    *,
    known_ids=frozenset(),
    known_paths=frozenset(),
    progress=False,
    stats=None,
) -> list[ModelCfg]:
    """Discover servable models from ``specs`` (each a :class:`config.DiscoverSpec`).

    A spec with ``dir=None`` scans ``model_dirs``. Native-head MTP models get
    ``speculative`` per the spec (``auto``/``True`` -> on for MTP; ``False`` -> off);
    assistant drafters are reported but never auto-wired (they need an explicit
    ``draft_gguf``). Sibling mmproj files pair into the model they best match when
    ``pair_mmproj``. Every id carries its quant codec (see :func:`_assign_ids`).
    ``known_ids`` /
    ``known_paths`` (from configured ``models:``) are skipped/deduped against, as
    are paths an earlier spec/root in this same call already emitted.
    ``progress`` streams per-file scan feedback to stderr (used by ``init``).
    ``stats``, if given, is a dict that receives ``skipped``: the count of
    .gguf files seen but unreadable as GGUF (so callers can distinguish an
    empty dir from a dir of truncated downloads)."""
    known_paths = {os.path.abspath(os.path.expanduser(p)) for p in known_paths}
    used_ids = set(known_ids)
    out: list[ModelCfg] = []
    skipped = 0
    # Overlapping roots (`dir: null` + an explicit subdir, or `-r` over a parent)
    # would otherwise emit the same GGUF twice, under two ids.
    seen: set[str] = set()

    for spec in specs:
        roots = [spec.dir] if spec.dir else list(model_dirs)
        for root in roots:
            if progress:
                print(f"[discover] scanning {root} ...", file=sys.stderr)
            paths = []
            for p in _iter_gguf_files(root, spec.recursive):
                ap = os.path.abspath(p)
                if ap in known_paths or ap in seen:
                    continue
                seen.add(ap)
                paths.append(p)
            classified = [c for c in _classify_each(paths, progress=progress) if c]
            skipped += len(paths) - len(classified)
            _emit_dir(classified, spec, used_ids, out)
    if stats is not None:
        stats["skipped"] = skipped
    return out


def find_retrieval_models(dirs, *, recursive=True):
    """Classify GGUFs under ``dirs`` and return the embedder / reranker ones this
    runtime can serve - the files :func:`scan_dirs` deliberately drops (it returns
    only chat models, printing a ``[discover]`` note for retrieval GGUFs). Used by
    ``gmlx init`` to offer an embedder / reranker the user already has on disk.

    Returns the retrieval GGUFs whose arch is in :func:`supported_arches` -- the
    ones this runtime can actually serve: Qwen3-Embedding / Qwen3-Reranker on the
    GGUF decoder path (last-token pooling / yes-no), and EmbeddingGemma on the
    encoder path. Encoder GGUFs we can't build (bert / xlm-roberta, not in
    supported_arches) are skipped. Returns ``(embedders, rerankers)`` -- two
    :class:`ClassifiedGguf` lists, de-duplicated by path, in scan order."""
    supported = supported_arches()
    embedders: list[ClassifiedGguf] = []
    rerankers: list[ClassifiedGguf] = []
    seen: set[str] = set()
    for root in dirs:
        for p in _iter_gguf_files(root, recursive):
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            c = classify_gguf(p)
            if c is None or c.kind not in ("embedding", "reranker"):
                continue
            if c.arch not in supported:        # arch we can't build -> skip
                continue
            (rerankers if c.kind == "reranker" else embedders).append(c)
    return embedders, rerankers


# Hugging Face cache scan
def _pick_cache_revision(repo):
    """Pick the cache revision to surface for ``repo`` and a portable ref name for it:
    the ``main`` branch when present (most portable), else a named tag/branch, else the
    most-recently-modified revision's commit hash. Returns ``(revision, ref_name)``, or
    ``(None, None)`` for an empty repo."""
    revs = list(repo.revisions)
    if not revs:
        return None, None
    for r in revs:
        if "main" in r.refs:
            return r, "main"
    newest = max(revs, key=lambda r: r.last_modified)
    refs = sorted(newest.refs)
    return newest, (refs[0] if refs else newest.commit_hash)


def scan_hf_cache(*, known_ids=frozenset(), known_refs=frozenset(),
                  progress=False) -> list[ModelCfg]:
    """Discover servable GGUF models from the **local** Hugging Face cache as portable
    ``hf:<org>/<repo>/<file.gguf>`` entries that :func:`config.resolve_path` resolves
    back to cache files (never the network).

    Scans every cached *model* repo's preferred revision (``main`` when present),
    classifies each GGUF header (model / mmproj / drafter), pairs sibling mmproj files,
    and synthesizes ids - the same pipeline as :func:`scan_dirs`, but emitting hf refs.
    ``known_ids`` / ``known_refs`` (from a configured ``models:``) are skipped/deduped."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print("[discover] huggingface_hub not installed; skipping hf-cache scan",
              file=sys.stderr)
        return []
    try:
        info = scan_cache_dir()
    except Exception as e:                       # corrupt cache / no cache dir
        print(f"[discover] could not scan the hf cache: {e}", file=sys.stderr)
        return []

    ref_by_path: dict[str, str] = {}
    classified: list[ClassifiedGguf] = []
    repos = sorted((r for r in info.repos if r.repo_type == "model"),
                   key=lambda r: r.repo_id)
    for repo in repos:
        rev, rev_name = _pick_cache_revision(repo)
        if rev is None:
            continue
        ggufs = [f for f in rev.files
                 if f.file_name.lower().endswith(".gguf")
                 and _is_first_shard(f.file_name)]
        if not ggufs:
            continue
        if progress:
            print(f"[discover] hf cache: {repo.repo_id}@{rev_name} "
                  f"({len(ggufs)} GGUF model "
                  f"file{plural_s(len(ggufs))})", file=sys.stderr)
        for f in sorted(ggufs, key=lambda f: f.file_name):
            suffix = "" if rev_name == "main" else f"@{rev_name}"
            ref = f"hf:{repo.repo_id}/{f.file_name}{suffix}"
            if ref in known_refs:
                continue
            c = classify_gguf(os.path.abspath(str(f.file_path)))
            if c is None:
                continue
            ref_by_path[c.path] = ref
            classified.append(c)

    spec = DiscoverSpec(dir=None, recursive=True, pair_mmproj=True, speculative="auto")
    out: list[ModelCfg] = []
    _emit_dir(classified, spec, set(known_ids), out)
    for mc in out:                               # cache file paths -> portable hf refs
        mc.path = ref_by_path.get(mc.path, mc.path)
        if mc.mmproj:
            mc.mmproj = ref_by_path.get(mc.mmproj, mc.mmproj)
    return out


def _classify_each(paths, *, progress):
    """Classify each GGUF in turn, reporting per-file progress to stderr when
    ``progress`` is set (header reads on a large directory are otherwise silent)."""
    n = len(paths)
    if progress:
        print(f"[discover] {n} GGUF file{plural_s(n)} to inspect",
              file=sys.stderr)
    for i, p in enumerate(paths, 1):
        if progress:
            print(f"[discover]   ({i}/{n}) {os.path.basename(p)}", file=sys.stderr)
        yield classify_gguf(p)


def _emit_dir(classified, spec, used_ids, out):
    """Build ModelCfgs for one scan, pairing mmproj/draft per directory."""
    by_dir: dict[str, list[ClassifiedGguf]] = {}
    for c in classified:
        by_dir.setdefault(os.path.dirname(c.path), []).append(c)

    # Assign every id up front across all loadable models in this scan, so the
    # compact-vs-full quant-codec decision sees all siblings (even across subdirs).
    all_models = [c for _d, g in sorted(by_dir.items())
                  for c in g if c.kind == "model" and c.loadable]
    id_for = _assign_ids(all_models, used_ids)

    for _dir, group in sorted(by_dir.items()):
        models = [c for c in group if c.kind == "model" and c.loadable]
        for c in group:
            if c.kind == "model" and not c.loadable:
                print(f"[discover] skip (unsupported arch {c.arch!r}): {c.path}",
                      file=sys.stderr)
            if c.kind == "drafter":
                print(f"[discover] assistant drafter (configure via "
                      f"draft_gguf: on its model): {c.path}", file=sys.stderr)
            if c.kind in ("embedding", "reranker"):
                wire = ("server.embeddings:" if c.kind == "embedding"
                        else "the /v1/rerank endpoint")
                print(f"[discover] {c.kind} model - not a chat model (serve via "
                      f"{wire}): {c.path}", file=sys.stderr)
            if c.kind == "adapter":
                print(f"[discover] LoRA adapter - not a chat model (attach via "
                      f"`adapter:` per model, or run/chat/serve --adapter): "
                      f"{c.path}", file=sys.stderr)

        made: list[tuple[ModelCfg, ClassifiedGguf]] = []
        for c in models:
            spec_flag = False if spec.speculative is False else bool(c.mtp)
            fam = _family_profiles.detect_family(c.arch)
            mc = ModelCfg(id=id_for[c.path], path=c.path, speculative=spec_flag,
                          family=fam if fam != "default" else None)
            made.append((mc, c))
            out.append(mc)

        if spec.pair_mmproj:
            for mm in (c for c in group if c.kind == "mmproj"):
                tgt = _best_mmproj_target(mm, made)
                if tgt is not None and tgt.mmproj is None:
                    tgt.mmproj = mm.path


def _assign_ids(models, used_ids: set) -> dict[str, str]:
    """Map each model's path to a unique friendly id, quant tag always included.

    Every id carries its quant codec. The compact form (``q4``, ``iq2``, ``q8``)
    is the default; the full codec (``q4-k-m``) is substituted for any base id
    whose siblings would otherwise collapse onto the same compact codec (e.g.
    ``Q4_K_S`` + ``Q4_K_M`` both compress to ``q4``), so the suffix stays
    meaningful instead of degrading to a numeric tiebreak. A numeric ``-N`` is the
    last resort for a genuine full-codec clash. Mutates ``used_ids``."""
    info = []                                            # (path, base_id, compact, full)
    for c in models:
        bn = os.path.basename(c.path)
        base, qtag = derive_id(bn)
        compact, full = _id_codecs(bn)
        if base:
            info.append((c.path, base, compact, full))
        else:                                            # name was essentially just a codec
            codec_id = full or (_slug(qtag) if qtag else "") or "model"
            info.append((c.path, codec_id, "", ""))

    collide: dict[tuple[str, str], int] = {}             # (base_id, compact) -> count
    for _p, base_id, compact, _f in info:
        if compact:
            collide[(base_id, compact)] = collide.get((base_id, compact), 0) + 1

    assigned: dict[str, str] = {}
    for path, base_id, compact, full in info:
        if not full:                                     # no codec to append
            cand = base_id
        elif collide.get((base_id, compact), 0) > 1:     # compact would clash -> full codec
            cand = f"{base_id}-{full}"
        else:
            cand = f"{base_id}-{compact}"
        final, n = cand, 2
        while final in used_ids:
            final = f"{cand}-{n}"
            n += 1
        used_ids.add(final)
        assigned[path] = final
    return assigned


def _best_mmproj_target(mm: ClassifiedGguf, made):
    """Pick the model a sibling mmproj belongs to. ``None`` if no confident match
    (logged). Two cases:

    * A **generic** projector name (``mmproj-F16.gguf`` -> no model name once the
      ``mmproj`` / codec markers are stripped) carries no signal, so it pairs to
      the sole model in the directory - and stays unpaired if there are several.
    * A **named** projector (``mmproj-Qwen2-VL-2B-Instruct-f16``) must actually
      match a model: the shared id-prefix has to cover most of the projector's
      name (>=70%, >=6 chars). A bare family overlap like ``qwen`` between a
      Qwen2-VL projector and a Qwen3-Embedding model is *not* enough - that loose
      match is exactly what mis-paired projectors onto unrelated models before."""
    if not made:
        return None
    core, _ = derive_id(os.path.basename(mm.path))   # markers (mmproj) stripped
    # llama.cpp's conversion default is mmproj-model-f16.gguf: a residual
    # core of "model" is the generic-name case, not a model called "model".
    if not core or core == "model":                  # generic projector, no name
        if len(made) == 1:
            return made[0][0]
        print(f"[discover] mmproj unpaired (ambiguous; several models): "
              f"{mm.path}", file=sys.stderr)
        return None
    best, best_len = None, 0
    for mc, _c in made:
        n = _common_prefix_len(core, mc.id)
        if n > best_len:
            best, best_len = mc, n
    if best is not None and best_len >= max(6, int(0.7 * len(core))):
        return best
    print(f"[discover] mmproj unpaired (no model name matches {core!r}): "
          f"{mm.path}", file=sys.stderr)
    return None


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# Scaffold
def _rel(path: str, model_dirs) -> str:
    """A path relative to a model_dirs root (for a portable config), else absolute.
    An ``hf:`` cache ref is already portable - pass it through untouched."""
    if isinstance(path, str) and path.startswith("hf:"):
        return path
    ap = os.path.abspath(path)
    for d in model_dirs:
        dd = os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
        if ap == dd or ap.startswith(dd + os.sep):
            return os.path.relpath(ap, dd)
    return ap


def model_to_entry(mc: ModelCfg, model_dirs) -> dict:
    """A plain config-entry dict for one :class:`ModelCfg` (``path`` plus any
    companions), for ``sync-models`` to splice into a YAML ``models:`` block. Paths render
    relative to a ``model_dirs`` root when possible, matching :func:`scaffold_yaml`."""
    entry: dict = {"path": _rel(mc.path, model_dirs)}
    if mc.mmproj:
        entry["mmproj"] = _rel(mc.mmproj, model_dirs)
    if mc.speculative:
        entry["speculative"] = True
    return entry


def _fmt_num(v) -> str:
    """Render a YAML number without a trailing ``.0`` on whole floats (900.0 -> 900)."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def family_comment(mc: ModelCfg) -> str:
    """A one-line family summary for a model entry's trailing comment
    (``qwen3.6: t=1.0 top_p=0.95 top_k=20``); "" when the family is unknown.
    Documentation only -- detection re-runs from the GGUF header at serve time."""
    fam = mc.family
    if not fam or fam == "default":
        return ""
    base = _family_profiles.family_base(fam).get("sampling", {})
    short = (("temperature", "t"), ("top_p", "top_p"), ("top_k", "top_k"),
             ("min_p", "min_p"), ("repetition_penalty", "rep"),
             ("presence_penalty", "presence"))
    parts = [f"{s}={base[k]}" for k, s in short if base.get(k)]
    return f"{fam}: {' '.join(parts)}" if parts else fam


def _aligned(rows: list, indent: str = "  ", gap: int = 3, cap: int = 40) -> list[str]:
    """``rows``: ``(keyval_text, comment_text_or_None)``. The comment column is
    computed from this group's own longest ``keyval_text`` up to ``cap`` chars, so
    it stays aligned regardless of live vs hint values or a user's own strings --
    not hardcoded spacing. A row past ``cap`` (a long HF id, an absolute path a
    user passed for e.g. ``embeddings:``) is a one-off outlier: it gets a plain
    single-gap comment instead of stretching the whole group's column out to
    match it."""
    width = max((len(kv) for kv, c in rows if c is not None and len(kv) <= cap),
               default=0)
    out = []
    for kv, c in rows:
        if c is None:
            out.append(f"{indent}{kv}")
        elif len(kv) <= width:
            out.append(f"{indent}{kv.ljust(width)}{' ' * gap}# {c}")
        else:
            out.append(f"{indent}{kv}{' ' * gap}# {c}")
    return out


def _scaffold_server_block(dirs, *, hf_cache, port, token_queue_timeout_s) -> list[str]:
    """Header comment + server: core, model_dirs, and discovery/capacity rows."""
    lines: list[str] = []
    lines.append("# gmlx server config -- created by `gmlx init`.")
    lines.append("# Every option below is live or a commented example with its "
                 "default. Inspect the")
    lines.append("# effective config: `gmlx serve --print-config` | sampling "
                 "table: `gmlx profiles`")
    lines.append("# | full reference: docs/server-config.md")
    lines.append("server:")
    lines.extend(_aligned([
        ("host: 127.0.0.1", None),
        (f"port: {int(port or 8080)}", None),
        ("# api_key: change-me",
         "optional for the default localhost bind, required otherwise"),
        ("# no_auth: true",
         "explicit opt-out of that requirement (e.g. auth handled by a "
         "front proxy/mTLS)"),
        ("# menubar: false",
         "disable the menu-bar monitor raised when `serve` backgrounds "
         "(macOS only, default on)"),
    ]))
    lines.append("")
    lines.append("  # Model discovery + capacity")
    lines.append("  model_dirs:")
    if dirs:
        for d in dirs:
            lines.append(f"    - {d}")
    else:
        lines.append("    []      # no local dirs; models below resolve from the "
                     "hf cache")
    hf_val = "true" if hf_cache else "false"
    discovery_rows = [
        (f"hf_cache: {hf_val}",
         "hf model paths resolve from the local cache, no downloads"),
        ("# budget_gb: 96",
         "GB cap on resident model weights (default: auto-sized to RAM)"),
        ("# max_models: 2",
         "max resident models at once (checked after budget_gb)"),
        ("# family_defaults: true",
         "built-in model architecture sampling defaults (false to disable)"),
    ]
    if token_queue_timeout_s is not None:
        discovery_rows.append(
            (f"token_queue_timeout_s: {_fmt_num(token_queue_timeout_s)}",
             "abort if no new token received in this time (0 = never, "
             "includes prefill time for first token)"))
    else:
        discovery_rows.append(
            ("# token_queue_timeout_s: 600",
             "abort if no new token received in this time (0 = never, "
             "includes prefill time for first token)"))
    lines.extend(_aligned(discovery_rows))
    lines.append("")
    return lines


def _scaffold_services_block(stt, tts, embeddings, rerank) -> list[str]:
    """Optional service endpoints; a chosen value uncomments its line."""
    lines: list[str] = []
    # Optional service models. A chosen value uncomments one line; otherwise a short
    # hint keeps the knob discoverable. Endpoints are OpenAI-compatible.
    lines.append("  # Optional service endpoints (all OpenAI-compatible)")
    svc_rows = []
    if stt:
        svc_rows.append((f"stt: {stt}",
                         "speech-to-text (POST /v1/audio/transcriptions)"))
    else:
        svc_rows.append(("# stt: whisper-turbo",
                         "speech-to-text: pip install 'gmlx[stt]' + ffmpeg"))
    if tts:
        svc_rows.append((f"tts: {tts}", "text-to-speech (POST /v1/audio/speech)"))
    else:
        svc_rows.append(("# tts: kokoro",
                         "text-to-speech: pip install 'gmlx[tts]' + ffmpeg"))
    if embeddings:
        svc_rows.append((f"embeddings: {embeddings}",
                         "text embeddings (POST /v1/embeddings)"))
    else:
        svc_rows.append(("# embeddings: qwen3-embed-0.6b",
                         "text embeddings: GGUF embedders work out of the box, "
                         "pip install 'gmlx[embeddings]' for safetensors"))
    if rerank:
        svc_rows.append((f"rerank: {rerank}", "reranking (POST /v1/rerank)"))
    else:
        svc_rows.append(("# rerank: qwen3-rerank-0.6b",
                         "reranking: Qwen3-Reranker GGUF works out of the box"))
    lines.extend(_aligned(svc_rows))
    lines.append("")
    return lines


def _scaffold_defaults_block(ttl_s, default_model) -> list[str]:
    """server.defaults: ttl + fallback model + profile hint."""
    lines: list[str] = []
    lines.append("  defaults:")
    ttl_val = 900 if ttl_s is None else ttl_s
    defaults_rows = [
        (f"ttl_s: {_fmt_num(ttl_val)}",
         "idle-unload timeout, seconds; 0/null = never unload"),
    ]
    if default_model:
        defaults_rows.append((f"model: {default_model}",
                              "fallback model when a request omits `model`"))
    else:
        defaults_rows.append(("# model: <id>",
                              "fallback model when a request omits `model`"))
    lines.extend(_aligned(defaults_rows, indent="    "))
    lines.append("    # profile: <name>    # global fallback profile (rules + "
                 "per-model settings win over this); rarely")
    lines.append("    #                    # needed -- each model already "
                 "starts from its family defaults")
    lines.append("")
    return lines


def _scaffold_cache_block(disk_cache, disk_cache_gb) -> list[str]:
    """APC prompt-cache block, with the SSD tier when requested."""
    lines: list[str] = []
    lines.append("  # APC prompt cache: reuse a shared prompt prefix across "
                 "requests. An exact-prefix")
    lines.append("  # hit skips the prefill recompute (~90% faster TTFT; "
                 "storing it costs ~1% of")
    lines.append("  # prefill). See docs/server-config.md \"Cache keys\". "
                 "Overridable per profile/model.")
    lines.append("  cache:")
    cache_rows = [
        ("enabled: true", None),
        ("# exact_entries: 4",
         "exact-prefix snapshots kept in memory for hybrid/recurrent "
         "models (default 4)"),
    ]
    if disk_cache:
        # `init --disk-cache` adds the SSD tier, so prefix reuse survives an
        # idle-unload / restart out of the box.
        gb = 50 if disk_cache_gb is None else disk_cache_gb
        cache_rows.append(
            ("disk:", "SSD tier: persists cache across an idle-unload or restart"))
        lines.extend(_aligned(cache_rows, indent="    "))
        lines.extend(_aligned([
            ("path: ~/.cache/gmlx/apc", None),
            (f"max_gb: {_fmt_num(gb)}",
             "disk cap per model (worst case: max_gb * resident models)"),
        ], indent="      "))
    else:
        cache_rows.append(
            ("disk: false",
             "true persists the cache to ~/.cache/gmlx/apc so reuse "
             "survives an idle-unload or restart; a mapping sets "
             "{path, max_gb, ...}"))
        lines.extend(_aligned(cache_rows, indent="    "))
    lines.append("")
    return lines


def _scaffold_profiles_block() -> list[str]:
    """Sampling/profiles reference comments + profiles/rules hints."""
    lines: list[str] = []
    lines.append("# Sampling. Every model automatically starts from its family's "
                 "model-card")
    lines.append("# recommended sampling (the trailing comment on each model "
                 "below; `gmlx profiles`")
    lines.append("# prints the full table). Built-in intents work on any model "
                 "with zero config,")
    lines.append("# as the request `model` (`<id>@coding`), a request `profile` "
                 "field, or")
    lines.append("# `run/chat --profile`: @coding @instruct @creative "
                 "@reasoning-low|-medium|-high")
    lines.append("# Profiles you define here layer on top of the family base. "
                 "Three override levels:")
    lines.append("#   1. reuse a built-in name (e.g. `coding:`) -> replaces that "
                 "intent everywhere")
    lines.append("#   2. `extends: <name>`                      -> compose a "
                 "variant of one")
    lines.append("#   3. a model's `profiles:` block            -> reshape an "
                 "intent for one model")
    lines.append("profiles:")
    lines.append("  # brief:     {sampling: {max_tokens: 512}}")
    lines.append("  # my-coding: {extends: coding, sampling: {min_p: 0.05}, "
                 "load: {kv_bits: 8}}")
    lines.append("  # agents:    {chat_template_kwargs: {preserve_thinking: "
                 "true}}   # keep prior-turn <think> (Qwen3.6, Gemma-4)")
    lines.append("  # narrator:  {system: \"Answer in second person.\", "
                 "chat_template: ~/.config/gmlx/templates/narrator.jinja}")
    lines.append("")
    lines.append("# Assign a profile to every id a glob matches "
                 "(a model's own `profile:` wins):")
    lines.append("# rules:")
    lines.append("#   - {match: \"*-coder-*\", profile: my-coding}")
    lines.append("")
    return lines


def _scaffold_aliases_block(aliases) -> list[str]:
    """aliases: block (real entries, or a commented hint)."""
    lines: list[str] = []
    if aliases:
        lines.append("aliases:    # friendly request names, listed in /v1/models")
        for name, target in aliases.items():
            lines.append(f"  {name}: {target}")
    else:
        lines.append("# Friendly request names, listed in /v1/models:")
        lines.append("# aliases: {fast: <id>, coder: <id>@coding}")
    lines.append("")
    return lines


def _scaffold_discover_block() -> list[str]:
    """Commented discover: scan-on-start example."""
    lines: list[str] = []
    lines.append("# Auto-register models by scanning a directory on every server "
                 "start, instead")
    lines.append("# of listing them by hand (skip this if you're happy curating "
                 "`models:` below")
    lines.append("# -- `gmlx init` already scanned once to produce that list).")
    lines.append("# discover:")
    lines.extend(_aligned([
        ("- dir: ~/llm/gguf", "default: server.model_dirs"),
        ("  recursive: true", "default: false"),
        ("  pair_mmproj: true",
         "auto-pair a sibling mmproj into a VLM entry (default: true)"),
        ("  speculative: auto",
         "auto | true | false -- wire in native-head MTP drafters "
         "(default: auto)"),
    ], indent="#   "))
    lines.append("")
    return lines


def _scaffold_models_block(models, dirs) -> list[str]:
    """models: per-model key reference + one entry per discovered model."""
    lines: list[str] = []
    lines.append("models:")
    # Per-model knobs, documented once instead of repeated under every entry.
    lines.append("  # Optional per-model keys:")
    lines.append("  #   profile: <name> (default profile for this model) | "
                 "family: <key> (override")
    lines.append("  #   auto-detection) | profiles: {coding: {sampling: {...}}} "
                 "(reshape one intent")
    lines.append("  #   for this model) | overrides: {sampling: {...}, "
                 "load: {...}, ...} (also:")
    lines.append("  #   cache, system, chat_template(_kwargs); always wins "
                 "over any profile) |")
    # Every wrapped line below must carry a `<placeholder>` (or `{...}`) so
    # _uncomment_hints's per-line heuristic (tests/test_discovery.py) leaves this
    # prose-only reference block commented instead of splicing a bare
    # `key: value | key: value` fragment into the parsed YAML.
    lines.append("  #   adapter: <lora.gguf> | pin: true (never auto-unload) | "
                 "ttl_s: 600")
    lines.append("  #   (overrides defaults.ttl_s) | speculative: true "
                 "(native-head MTP) |")
    lines.append("  #   draft_gguf: <assistant.gguf> (assistant-drafter MTP) |")
    lines.append("  #   mmproj: <file> (VLM) | stream: experts (over-RAM MoE: "
                 "experts stream from")
    lines.append("  #   disk, rest of the model + KV on GPU) | stream: cpu "
                 "(whole model on CPU) |")
    lines.append("  #   moe_expert_mass: <P> (adaptive lossy fan-out on "
                 "streamed experts; size P")
    lines.append("  #   with `gmlx run --moe-expert-probe`)")
    if not models:
        lines.append("  # (none discovered) -- point model_dirs at a folder of "
                     ".gguf files")
    for mc in sorted(models, key=lambda m: m.id):
        note = family_comment(mc)
        lines.append(f"  {mc.id}:" + (f"        # {note}" if note else ""))
        lines.append(f"    path: {_rel(mc.path, dirs)}")
        if mc.profile:
            lines.append(f"    profile: {mc.profile}    # pinned default "
                         "(requests can still switch @intent)")
        if mc.mmproj:
            lines.append(f"    mmproj: {_rel(mc.mmproj, dirs)}    # VLM companion")
        if mc.speculative:
            lines.append("    speculative: true       # native-head MTP "
                         "(drafter inside the target GGUF)")
    lines.append("")
    return lines


def _scaffold_talk_block(talk) -> list[str]:
    """talk: voice-chat block (real values, or the commented field reference)."""
    from .hotkey import PUSH_TO_TALK_MODIFIERS
    modifiers = " | ".join(PUSH_TO_TALK_MODIFIERS)
    alternates = " | ".join(m for m in PUSH_TO_TALK_MODIFIERS if m != "globe")
    lines: list[str] = []
    if talk:
        lines.append("talk:                     # voice chat client: `gmlx talk` "
                     "(wake word -> STT -> chat -> TTS)")
        lines.append(f"  voice: {talk['voice']}              # kokoro preset or "
                     "a qwen3-tts speaker name")
        lines.append(f"  wake_word: \"{talk['wake_word']}\"")
        lines.append(f"  mode: {talk['mode']}    # wake=say the phrase | "
                     "vad=just start talking | ptt=space in the terminal | "
                     "text=typed")
        modifier = talk.get("push_to_talk_modifier", "globe")
        lines.append(f"  push_to_talk_modifier: {modifier}    # menu bar "
                     f"hotkey is <key>+Space: {modifiers}")
    else:
        lines.append("# Voice chat client (`gmlx talk`): needs stt + tts above "
                     "+ `pip install 'gmlx[talk]'`")
        lines.append("# talk:")
        talk_field_rows = [
            ("model: <id>@profile",
             "which model to talk to (default: server's default model)"),
            ("voice: af_heart", "kokoro preset or a qwen3-tts speaker name"),
            ("speed: 1.0", "TTS playback speed multiplier"),
            ("wake_word: \"hey assistant\"", "any phrase works, no training needed"),
            ("wake_threshold: 0.3",
             "detection confidence needed to trigger (0-1)"),
            ("mode: wake",
             "wake=say the phrase | vad=just start talking | "
             "ptt=space in the terminal | text=typed"),
            ("push_to_talk_modifier: globe",
             "menu bar hotkey is <key>+Space; keyboards without a Globe key: "
             f"{alternates}"),
            ("system: <prompt>", "spoken persona (default: a speakable-output prompt; \"\" disables)"),
            ("language: en", "whisper STT language hint (default: auto-detect)"),
            ("max_tokens: 512", "cap on spoken reply length"),
            ("chime: true", "audio chime on wake / turn end"),
            ("input_device: <name>",
             "sounddevice name/index (default: system default mic)"),
            ("output_device: <name>",
             "sounddevice name/index (default: system default output)"),
        ]
        lines.extend(_aligned(talk_field_rows, indent="#   "))
        lines.append("#")
        lines.append("#   vad:                           # endpointing tuning, "
                     "only used in `mode: vad`")
        vad_rows = [
            ("threshold: 0.6", "speech-probability threshold (0-1)"),
            ("silence_ms: 550", "trailing silence that ends an utterance"),
            ("min_speech_ms: 300", "shorter utterances are discarded as noise"),
            ("pre_roll_ms: 400", "audio kept from before speech onset"),
        ]
        lines.extend(_aligned(vad_rows, indent="#     "))
        lines.append("#")
        lines.append("#   brain: chat    # chat=plain turn-based | "
                     "assistant=adds MCP tools + long-term memory")
    lines.append("")
    return lines


def _scaffold_assistant_block() -> list[str]:
    """Commented assistant: tool-loop reference block."""
    lines: list[str] = []
    lines.append("# The built-in tool-loop assistant, shared by `talk.brain: "
                 "assistant`,")
    lines.append("# `chat --assistant`, and served `server.assistants` aliases "
                 "(NOT the external")
    lines.append("# coding agents `gmlx launch` points at the server):")
    lines.append("# assistant:")
    assistant_rows = [
        ("max_tool_rounds: 8", "tool-call round-trips per turn"),
        ("tool_timeout_s: 60", "per-tool execution timeout"),
    ]
    lines.extend(_aligned(assistant_rows, indent="#   "))
    lines.append("#   mcp:                         # MCP servers "
                 "providing tools")
    lines.append("#     - name: my-tools")
    lines.append("#       command: [npx, -y, some-mcp-server]   # stdio "
                 "transport (or use url: for HTTP)")
    lines.append("#   memory:")
    memory_rows = [
        ("enabled: true", "long-term memory (sqlite + embeddings)"),
        ("top_k: 4", "memories recalled per turn"),
        ("ttl_days: <n>", "expire memories older than this (default: never)"),
        ("max_items: 20000",
         "store size cap, evicts least-recalled oldest first"),
    ]
    lines.extend(_aligned(memory_rows, indent="#     "))
    lines.append("")
    return lines


def scaffold_yaml(models: list[ModelCfg], *, model_dirs,
                  hf_cache: bool = False,
                  disk_cache: bool = False, disk_cache_gb=None,
                  stt=None, tts=None, embeddings=None, rerank=None,
                  default_model=None, aliases=None, ttl_s=None,
                  token_queue_timeout_s=None, talk=None, port=None) -> str:
    """Emit a starter YAML config from discovered models. Local paths render relative
    to ``model_dirs`` when possible; ``hf:`` cache refs pass through portably. The
    output parses cleanly via :func:`config.load_config`; commented hints show the
    optional knobs. ``hf_cache`` writes ``server.hf_cache: true`` (set by
    ``init --from-hf-cache`` so the cache-resident entries resolve).

    The wizard / ``init --with-*`` knobs uncomment what would otherwise be a hint:
    ``stt`` / ``tts`` / ``embeddings`` / ``rerank`` write ``server.<svc>``; ``ttl_s`` and
    ``token_queue_timeout_s`` write their server knobs (``0`` => never / wait
    forever); ``default_model`` writes ``server.defaults.model``; ``aliases``
    (``{name: id}``) writes a real ``aliases:`` block. ``disk_cache`` turns the APC
    SSD tier on, with ``disk_cache_gb`` as the per-namespace ``max_gb`` cap (default
    50 when unset). All must reference ids that exist in ``models`` - the caller
    curates that. ``talk`` (``{voice, wake_word, mode}``) writes a top-level
    ``talk:`` block for the voice-chat client; ``None`` leaves a commented hint.
    ``port`` overrides the default 8080 (``init --port``)."""
    # Anchor cwd-relative roots: the server resolves `model_dirs` against its
    # own cwd (launchd runs at `/`), so a root written as typed ("models")
    # silently yields a zero-model server from any other directory. `~` and
    # `$VAR` forms stay verbatim -- they expand at load and stay portable.
    dirs = [d if d.startswith(("~", "$")) or os.path.isabs(os.path.expandvars(d))
            else os.path.abspath(d)
            for d in model_dirs]
    lines: list[str] = []
    lines += _scaffold_server_block(
        dirs, hf_cache=hf_cache, port=port,
        token_queue_timeout_s=token_queue_timeout_s)
    lines += _scaffold_services_block(stt, tts, embeddings, rerank)
    lines += _scaffold_defaults_block(ttl_s, default_model)
    lines += _scaffold_cache_block(disk_cache, disk_cache_gb)
    lines += _scaffold_profiles_block()
    lines += _scaffold_aliases_block(aliases)
    lines += _scaffold_discover_block()
    lines += _scaffold_models_block(models, dirs)
    lines += _scaffold_talk_block(talk)
    lines += _scaffold_assistant_block()
    return "\n".join(lines)
