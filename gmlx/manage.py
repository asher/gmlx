#!/usr/bin/env python3
"""``gmlx validate`` / ``gmlx pull`` - check that a GGUF will load before
committing to a multi-GB download, and (for ``pull``) fetch it to a plain local
file rather than the Hugging Face blob cache.

``validate`` accepts a local path, an ``hf:<org>/<repo>/<file.gguf>[@rev]`` id, or
an ``http(s)://`` URL. A local file is classified by reading its header with
gguf-py; a remote ref is checked by range-reading just the GGUF header (a few MB)
- see :mod:`gmlx.remote`. Both report the architecture (and whether the
installed ``mlx-lm`` implements it) plus the per-tensor codec histogram, and they
agree by reusing preflight's codec sets.

``pull`` runs the same remote header validation first and refuses an unloadable
GGUF unless ``--force``, then downloads (all shards of a split file) - never the
HF cache. It lands in the server config's first ``model_dirs`` root by default
(under ``<dir>/<org>__<repo>/`` for hf refs, so ``serve`` discovery / ``sync-models``
find it), or into ``--to DIR`` exactly. Several files fetch in one go - multipart
GGUFs expand automatically, and extra bare filenames resolve in the first ref's
repo (an mmproj companion, a second quant). Interrupted transfers resume.

``list`` tables the local GGUFs a directory holds (the same header-only scan
``serve`` discovery uses); ``ps`` shows the models resident in a running server.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

from .textfmt import plural_s
from . import remote
from .preflight import (
    NATIVE_FP_TYPES,
    NATIVE_TYPES,
    SPLIT_SHARD_RE,
    SUPPORTED_QUANT_TYPES,
    find_split_shards,
    shard_names,
)


# Classification (local) + verdict shared by both verbs
def _classify_local(path: str, *, arch: str | None = None) -> remote.HeaderReport:
    """Build the same :class:`remote.HeaderReport` from a local GGUF (all shards),
    reusing preflight's codec sets so a local verdict matches the remote one."""
    from gguf import GGUFReader

    from .gguf_meta import read_string
    from .remap import detect_arch

    shards = find_split_shards(path)
    reader0 = GGUFReader(shards[0], "r")
    detected = arch or detect_arch(reader0)
    gguf_type = read_string(reader0, "general.type")
    hist: dict[str, int] = {}
    unsup: dict[str, int] = {}
    for i, shard in enumerate(shards):
        reader = reader0 if i == 0 else GGUFReader(shard, "r")
        for t in reader.tensors:
            tn = t.tensor_type.name
            hist[tn] = hist.get(tn, 0) + 1
            if (tn not in SUPPORTED_QUANT_TYPES and tn not in NATIVE_TYPES
                    and tn not in NATIVE_FP_TYPES):
                unsup[tn] = unsup.get(tn, 0) + 1
    return remote.HeaderReport(detected, hist, unsup, sum(hist.values()),
                               gguf_type)


def _arch_status(arch: str | None, *, hf_source: str | None = None):
    """``(supported: bool, error: str | None)`` - runs the same arch gate the
    loader does (mapping -> not-disabled -> mlx-lm has the model -> config synth)."""
    if not arch:
        return False, "could not determine architecture from GGUF metadata"
    from .arch_table import UnsupportedArchError, gate
    try:
        gate(arch, hf_source=hf_source)
        return True, None
    except UnsupportedArchError as e:
        return False, str(e)


def _build_report(ref: remote.Ref, *, arch: str | None = None,
                  max_mb: int | None = None) -> tuple[remote.HeaderReport, int]:
    """Classify a GGUF, **across all shards** of a split file, and return
    ``(report, n_shards)``. Local reads every on-disk shard; remote range-reads
    each shard's header and unions them - so a codec used by a single tensor in
    any shard can't slip through."""
    if ref.kind == "local":
        if not os.path.exists(ref.raw):
            raise remote.RemoteError(
                f"no such file: {ref.raw} (validate takes a local path, "
                "hf:<org>/<repo>/<file.gguf>, or an http(s):// URL)")
        try:
            return _classify_local(ref.raw, arch=arch), len(find_split_shards(ref.raw))
        except FileNotFoundError as e:
            # An incomplete split set - surface it as a clean verdict, not a traceback.
            raise remote.RemoteError(str(e)) from e
        except IsADirectoryError as e:
            raise remote.RemoteError(
                f"{ref.raw} is a directory, not a GGUF file") from e
        except Exception as e:
            # Vetting dubious files is this verb's job: any header-parse
            # failure is a verdict, never a traceback.
            raise remote.RemoteError(
                f"cannot read {ref.raw} as GGUF ({e}) - truncated download "
                f"or not a GGUF file?") from e

    kwargs = {}
    if max_mb is not None:
        if max_mb <= 0:
            raise remote.RemoteError(f"--max-mb must be positive (got {max_mb})")
        kwargs["max_bytes"] = max_mb * 1024 * 1024
        kwargs["initial"] = min(4 * 1024 * 1024, max_mb * 1024 * 1024)
    urls = _remote_shard_urls(ref)
    reports = [remote.fetch_header(u, **kwargs) for u in urls]
    report = remote.aggregate_reports(reports)
    if arch:
        report.arch = arch
    return report, len(urls)


def _verdict(ref: remote.Ref, report: remote.HeaderReport, *,
             hf_source: str | None = None, n_shards: int = 1) -> dict:
    # An mmproj (vision/audio projector) carries general.architecture="clip".
    # It is not a standalone model, so the arch gate doesn't apply - but it's a
    # perfectly valid file for its purpose (pair it with the LLM GGUF).
    mmproj = report.arch == "clip"
    # A LoRA adapter carries its base model's arch (general.type = "adapter"),
    # so like mmproj it's a valid companion file, never a standalone model.
    adapter = report.gguf_type == "adapter"
    if mmproj or adapter:
        arch_ok, arch_err = False, None
    else:
        arch_ok, arch_err = _arch_status(report.arch, hf_source=hf_source)
    return {
        "ref": ref.raw,
        "kind": ref.kind,
        "arch": report.arch,
        "arch_supported": arch_ok,
        "arch_error": arch_err,
        "mmproj": mmproj,
        "adapter": adapter,
        "n_shards": n_shards,
        "n_tensors": report.n_tensors,
        "codecs": dict(sorted(report.histogram.items())),
        "unsupported_codecs": dict(sorted(report.unsupported.items())),
        "codecs_loadable": report.loadable_codecs,
        "loadable": report.loadable_codecs and arch_ok,
        # loadable = runs standalone; usable also admits a healthy companion
        # (mmproj / LoRA adapter).
        "usable": (report.loadable_codecs and arch_ok)
                  or ((mmproj or adapter) and report.loadable_codecs),
    }


def _print_report(v: dict) -> None:
    print(f"GGUF validation: {v['ref']}")
    print(f"  source: {v['kind']}")
    arch = v["arch"] or "?"
    if v.get("mmproj"):
        print(f"  architecture: {arch}  [mmproj companion]")
    elif v.get("adapter"):
        print(f"  architecture: {arch}  [LoRA adapter]")
    elif v["arch_supported"]:
        print(f"  architecture: {arch}  [supported]")
    else:
        print(f"  architecture: {arch}  [unsupported]")
        if v["arch_error"]:
            print(f"    {v['arch_error']}")
    tline = f"  tensors: {v['n_tensors']}"
    if v.get("n_shards", 1) > 1:
        tline += f"   (across {v['n_shards']} shards)"
    print(tline)
    print("  codecs:")
    for name, n in v["codecs"].items():
        mark = "   <- no kernel" if name in v["unsupported_codecs"] else ""
        print(f"    {name:<8} x{n}{mark}")
    if v["loadable"]:
        print("  => loadable")
    elif v.get("mmproj") and v["codecs_loadable"]:
        print("  => mmproj companion: a vision/audio projector, not a standalone")
        print("     model - pair it with its LLM GGUF: --mmproj on run/chat/serve,")
        print("     or `mmproj:` per model in the server config.")
    elif v.get("adapter") and v["codecs_loadable"]:
        print("  => adapter companion: a LoRA adapter, not a standalone model -")
        print("     attach it to its base GGUF: --adapter on run/chat/serve, or")
        print("     `adapter:` per model in the server config.")
    else:
        reasons = []
        if not v["codecs_loadable"]:
            bad = ", ".join(f"{k}x{n}"
                            for k, n in v["unsupported_codecs"].items())
            reasons.append(f"unsupported codecs ({bad})")
        if not v["arch_supported"] and not (v.get("mmproj") or v.get("adapter")):
            reasons.append("architecture not supported")
        print(f"  => not loadable: {'; '.join(reasons)}")


# validate
def cmd_validate(argv: list | None = None, prog: str = "gmlx validate") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Check that a GGUF will load - locally, or by range-reading "
                    "just the header of a remote file - without a full download.")
    ap.add_argument("ref", help="Local path, hf:<org>/<repo>/<file.gguf>[@rev], "
                                "or an http(s):// URL.")
    ap.add_argument("--arch", default=None,
                    help="Override architecture detection.")
    ap.add_argument("--hf-source", default=None,
                    help="Treat the arch as loadable with this config override "
                         "(matches `run --hf-source`).")
    ap.add_argument("--max-mb", type=int, default=None,
                    help="Cap the remote header range-read (MB; default 128).")
    ap.add_argument("--json", action="store_true",
                    help="Emit the verdict as JSON instead of a report.")
    a = ap.parse_args(argv)

    try:
        ref = _resolve_to_file(remote.parse_ref(a.ref))
        report, n_shards = _build_report(ref, arch=a.arch, max_mb=a.max_mb)
    except remote.AmbiguousRepo as e:
        # A repo with several models is a listing, not a failure: print the
        # ready-to-paste refs and succeed (the README promises exactly this).
        if a.json:
            print(json.dumps({"repo": e.where, "models": e.refs}, indent=2))
        else:
            print(f"{e.where} has {len(e.refs)} GGUF models:")
            for r in e.refs:
                print(f"  {r}")
        return 0
    except remote.RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    v = _verdict(ref, report, hf_source=a.hf_source, n_shards=n_shards)
    if a.json:
        print(json.dumps(v, indent=2))
    else:
        _print_report(v)
    return 0 if v["usable"] else 1


# pull - validate the header, then download (not into the HF cache)
def _shard_names(filename: str) -> list[str]:
    """Expand a split-GGUF basename to every shard name; non-split -> ``[filename]``.
    ``preflight.shard_names`` with its malformed-name ValueError as a RemoteError."""
    try:
        return shard_names(filename)
    except ValueError as exc:
        raise remote.RemoteError(str(exc)) from None


def _group_shard_sets(paths: list[str]) -> dict[str, str]:
    """Map each GGUF model in ``paths`` to one representative shard - the
    lowest-index shard of a split set, or the file itself."""
    singles: dict[str, str] = {}
    split_min: dict[str, tuple[int, str]] = {}
    for p in paths:
        m = SPLIT_SHARD_RE.search(p)
        if not m:
            singles[p] = p
            continue
        idx, prefix = int(m.group(1)), p[:m.start()]
        cur = split_min.get(prefix)
        if cur is None or idx < cur[0]:
            split_min[prefix] = (idx, p)
    out = dict(singles)
    for prefix, (_idx, p) in split_min.items():
        out[prefix] = p
    return out


def _resolve_to_file(ref: remote.Ref) -> remote.Ref:
    """Turn an hf *directory* / repo ref into a concrete first-shard file ref by
    listing it: a sole GGUF model auto-resolves (noted on stderr); several models
    raise listing the pickable refs; none raises (naming any subfolders to drill
    into). File refs (and non-hf refs) pass through untouched."""
    if ref.kind != "hf" or not ref.is_dir:
        return ref
    entries = remote.hf_list_dir(ref.repo, ref.path_in_repo or "", ref.revision)
    where = f"{ref.repo}/{ref.path_in_repo}" if ref.path_in_repo else ref.repo
    ggufs = [p for p, typ, _ in entries if typ == "file" and p.endswith(".gguf")]
    if not ggufs:
        dirs = sorted(p for p, typ, _ in entries if typ == "directory")
        hint = ("\nsubfolders to try:\n"
                + "\n".join(f"  hf:{ref.repo}/{d}" for d in dirs[:40])) if dirs else ""
        raise remote.RemoteError(f"no .gguf files under {where}{hint}")
    reps = sorted(_group_shard_sets(ggufs).values())
    if len(reps) == 1:
        rep = reps[0]
        print(f"[resolved] {where} -> hf:{ref.repo}/{rep}", file=sys.stderr)
        return remote._make_hf_ref(ref.repo, rep, ref.revision,
                                   f"hf:{ref.repo}/{rep}")
    listing = "\n".join(f"  hf:{ref.repo}/{r}" for r in reps[:40])
    more = "" if len(reps) <= 40 else f"\n  ...and {len(reps) - 40} more"
    raise remote.AmbiguousRepo(
        f"{where} has {len(reps)} GGUF models - pass one:\n{listing}{more}",
        where, [f"hf:{ref.repo}/{r}" for r in reps])


def _remote_shard_urls(ref: remote.Ref) -> list[str]:
    """Every shard's fetch URL for a remote ref (one entry for a non-split file).
    The given ref can point at any shard; the whole ``-of-000NN`` set is derived."""
    if ref.kind == "hf":
        names = _shard_names(ref.path_in_repo)
        if len(names) == 1:
            return [ref.url]
        return [remote.hf_resolve_url(ref.repo, n, ref.revision) for n in names]
    names = _shard_names(ref.filename)
    if len(names) == 1:
        return [ref.url]                     # keep the original URL (query intact)
    base = ref.url.rsplit("/", 1)[0]
    return [f"{base}/{n}" for n in names]


def _hf_download(repo: str, filename: str, revision: str, dest_dir: str) -> str:
    """Download one repo file into ``dest_dir`` via URL + ``.part``-file resume.
    Module-level seam: monkeypatched in tests."""
    url = remote.hf_resolve_url(repo, filename, revision)
    dest_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    return _url_download(url, dest_path)


def _url_download(url: str, dest_path: str) -> str:
    """Stream a URL to ``dest_path``, resuming an interrupted transfer.

    Downloads to a sibling ``.part`` file, requesting a byte ``Range`` to continue
    where a previous run left off (the server must honour it; a ``200`` answer means
    it didn't, so we restart from byte 0). The ``.part`` is renamed into place only
    on completion -- a failure leaves it behind so a re-run resumes rather than
    restarts. A finished ``dest_path`` short-circuits (idempotent re-pull).
    Module-level seam: monkeypatched in tests."""
    if os.path.exists(dest_path):
        return dest_path
    part = dest_path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(remote._auth_headers(url))
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = remote.http_open(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have > 0:
            # 416 = our .part already covers the remote range. Only an exact
            # size match proves completion; a stale .part larger than the
            # remote file must not be promoted to a corrupt final download.
            cr = (e.headers.get("Content-Range") or "").strip() if e.headers else ""
            m = re.match(r"bytes \*/(\d+)$", cr)
            if m and int(m.group(1)) == have:
                os.replace(part, dest_path)
                return dest_path
            if m:
                raise remote.RemoteError(
                    f"stale partial download: {part} has {have} bytes but the "
                    f"remote file is {m.group(1)} - delete the .part and "
                    f"re-pull") from e
            raise remote.RemoteError(
                f"range not satisfiable and no Content-Range to confirm "
                f"{part} ({have} bytes) is complete - delete the .part and "
                f"re-pull") from e
        raise
    with resp:
        status = getattr(resp, "status", 200)
        resumed = have > 0 and status == 206
        total = _download_total(resp, have, resumed)
        fname = os.path.basename(dest_path)
        if resumed:
            print(f"  resuming {fname} from {_human_gb(have, 2)}"
                  + (f" / {_human_gb(total, 2)}" if total else ""),
                  file=sys.stderr)
        t0 = time.monotonic()
        session_bytes = 0
        last_print = 0.0
        with open(part, "ab" if resumed else "wb") as f:
            written = have if resumed else 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                session_bytes += len(chunk)
                now = time.monotonic()
                if now - last_print >= 2.0:
                    last_print = now
                    stats = _transfer_stats(t0, session_bytes,
                                            (total - written) if total else None)
                    if total:
                        pct = written * 100 // total
                        print(f"\r  {fname}: {_human_gb(written, 2)} / "
                              f"{_human_gb(total, 2)} ({pct}%) {stats}",
                              end="", file=sys.stderr, flush=True)
                    else:
                        print(f"\r  {fname}: {_human_gb(written, 2)} {stats}",
                              end="", file=sys.stderr, flush=True)
            if total or written:
                print(file=sys.stderr)
    if total and written != total:
        # A clean early close reads as EOF (read() returns b"" instead of
        # raising), so an unchecked rename would publish a truncated file
        # that the completed-download short-circuit then makes permanent.
        raise remote.RemoteError(
            f"connection closed early: got {written} of {total} bytes for "
            f"{os.path.basename(dest_path)} - re-run to resume from the .part")
    os.replace(part, dest_path)
    return dest_path


def _fmt_duration(s: float) -> str:
    s = int(s)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _transfer_stats(t0: float, session_bytes: int,
                    remaining: int | None) -> str:
    elapsed = time.monotonic() - t0
    if elapsed < 0.5:
        return ""
    rate = session_bytes / elapsed
    if rate < 1:
        return f"[{_fmt_duration(elapsed)}]"
    rate_s = (f"{rate / 1e6:.1f} MB/s" if rate < 1e9
              else f"{rate / 1e9:.2f} GB/s")
    if remaining is not None and remaining > 0:
        eta = remaining / rate
        return f"[{_fmt_duration(elapsed)}<{_fmt_duration(eta)}, {rate_s}]"
    return f"[{_fmt_duration(elapsed)}, {rate_s}]"


def _download_total(resp, have: int, resumed: bool) -> int | None:
    """Extract the total file size from the response headers."""
    cr = getattr(resp, "getheader", lambda _: None)("Content-Range")
    if cr:
        try:
            return int(cr.rsplit("/", 1)[-1])
        except (ValueError, IndexError):
            pass
    cl = getattr(resp, "getheader", lambda _: None)("Content-Length")
    if cl:
        try:
            return int(cl) + (have if resumed else 0)
        except ValueError:
            pass
    return None


def _model_dir_dest(config_path: str | None = None) -> str:
    """Resolve the default download dir from a server config's first ``model_dirs``
    root (where ``pull`` lands files unless ``--to`` is given). Searches the standard
    config locations unless ``config_path`` is given. Raises :class:`remote.RemoteError`
    with an actionable message when there's no config or no ``model_dirs``."""
    from . import config as cfgmod

    if config_path:
        config_path = os.path.expanduser(config_path)
        if not os.path.exists(config_path):
            raise remote.RemoteError(f"--config not found: {config_path}")
        path = config_path
    else:
        path = next((str(c) for c in cfgmod.default_config_paths()
                     if os.path.exists(c)), None)
        if path is None:
            raise remote.RemoteError(
                "pull needs a server config to find your model dir, but none was "
                "found in the default locations; run `gmlx init --models-dir "
                "DIR` first, pass --config FILE, or use --to DIR.")
    try:
        cfg = cfgmod.load_config(path)
    except cfgmod.ConfigError as e:
        raise remote.RemoteError(f"could not load config {path}: {e}")
    if not cfg.model_dirs:
        raise remote.RemoteError(
            f"config {path} defines no server.model_dirs; pass --to DIR.")
    return os.path.expanduser(os.path.expandvars(cfg.model_dirs[0]))


def _download_ref(ref: remote.Ref, dest_dir: str) -> list[str]:
    os.makedirs(dest_dir, exist_ok=True)
    out: list[str] = []
    if ref.kind == "hf":
        for name in _shard_names(ref.path_in_repo):
            out.append(_hf_download(ref.repo, name, ref.revision, dest_dir))
    elif ref.kind == "url":
        names = _shard_names(ref.filename)
        if len(names) == 1:                  # keep the original URL (query intact)
            out.append(_url_download(ref.url,
                                     os.path.join(dest_dir, ref.filename)))
        else:
            base = ref.url.rsplit("/", 1)[0]
            for name in names:
                out.append(_url_download(f"{base}/{name}",
                                         os.path.join(dest_dir, name)))
    else:
        raise remote.RemoteError(
            "pull needs an hf: ref or an http(s):// URL "
            "(a local path is already on disk)")
    return out


def _expand_refs(raw_refs: list[str]) -> list[remote.Ref]:
    """Expand a pull's positional refs into concrete :class:`remote.Ref` targets.

    The first scheme ref (``hf:`` / ``http(s)://``) or existing local path is parsed
    normally; any following **bare filename** (no scheme, not a local path) is taken
    as a sibling in the most recent hf repo - so a model and its mmproj (or a second
    quant) download in one go::

        gmlx pull hf:org/repo/model-Q4_K_M.gguf mmproj-F16.gguf model-Q6_K.gguf
        gmlx pull hf:org/repo/ model-Q4_K_M.gguf mmproj-F16.gguf

    A bare name inherits the anchor's revision and subfolder. When bare names are
    attached to a directory/repo ref, that ref is dropped (it served only as the
    anchor - its files are named explicitly). A bare name with no preceding hf
    anchor is an error."""
    out: list[remote.Ref] = []
    anchor = None                        # (repo, revision, subdir, dir_ref_index|None)
    anchored_dirs: set[int] = set()
    for raw in raw_refs:
        # A plain filename (no scheme, no path separator, not on disk) is a sibling
        # in the most recent hf repo; everything else parses as a normal ref.
        bare = ("/" not in raw and not raw.startswith(("~", "hf:", "http://",
                "https://")) and not os.path.exists(os.path.expanduser(raw)))
        if not bare and "/" not in raw \
                and not raw.startswith(("~", "hf:", "http://", "https://")):
            # A would-be sibling name that happens to match a file in CWD: it
            # parses as a local ref, and pull skips it (already on disk). Say
            # so - silence here flips behavior with the working directory.
            # (No anchor - e.g. after an http(s) ref - still deserves the note.)
            hint = (f"; write hf:{anchor[0]}/{raw} to fetch the repo sibling "
                    "instead" if anchor is not None else "")
            print(f"note: {raw!r} matches a local file - skipping it{hint}",
                  file=sys.stderr)
        if bare:
            if anchor is None:
                raise remote.RemoteError(
                    f"{raw!r} is not a model ref - use "
                    "hf:<org>/<repo>/<file.gguf> or an http(s):// URL (bare "
                    "filenames only name siblings after an hf: ref)")
            repo, rev, subdir, dir_idx = anchor
            path = f"{subdir}/{raw}" if subdir else raw
            out.append(remote._make_hf_ref(repo, path, rev, f"hf:{repo}/{path}"))
            if dir_idx is not None:
                anchored_dirs.add(dir_idx)
            continue
        ref = remote.parse_ref(raw)
        out.append(ref)
        if ref.kind == "hf":
            pir = ref.path_in_repo or ""
            if ref.is_dir:
                anchor = (ref.repo, ref.revision, pir.rstrip("/"), len(out) - 1)
            else:
                parent = pir.rsplit("/", 1)[0] if "/" in pir else ""
                anchor = (ref.repo, ref.revision, parent, None)
        else:
            anchor = None
    return [r for i, r in enumerate(out) if i not in anchored_dirs]


def _repo_subdir(ref: remote.Ref) -> str | None:
    """``<org>__<repo>`` for an hf ref (a filesystem-safe repo name), else ``None``."""
    if ref.kind == "hf" and ref.repo:
        return ref.repo.replace("/", "__")
    return None


def _dest_for_ref(ref: remote.Ref, base_dir: str, nest: bool) -> str:
    """The directory ``ref`` downloads into. With ``nest`` (the model-dir default),
    an hf file lands under ``<base_dir>/<org>__<repo>/`` so siblings group and
    discovery's recursive scan finds them; ``--to`` (``nest=False``) writes straight
    into ``base_dir``."""
    sub = _repo_subdir(ref) if nest else None
    return os.path.join(base_dir, sub) if sub else base_dir


def _disk_free(path: str) -> int:
    """Free bytes on the volume holding ``path`` (nearest existing ancestor).
    Module-level seam: monkeypatched in tests."""
    p = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free


def _planned_download_bytes(ref: remote.Ref, dest_dir: str,
                            listing_cache: dict) -> int | None:
    """Bytes still needed to land ``ref`` (all shards) in ``dest_dir``, or
    ``None`` when unknown (non-hf ref, listing failed, or a shard is missing
    from the listing). Existing final files and ``.part`` resume files are
    subtracted, so a re-pull or resume needs only the remainder."""
    if ref.kind != "hf":
        return None
    key = (ref.repo, ref.revision)
    if key not in listing_cache:
        try:
            listing_cache[key] = {
                p: size for p, typ, size in
                remote.hf_list_dir(ref.repo, "", ref.revision)
                if typ == "file"}
        except remote.RemoteError:
            listing_cache[key] = None
    sizes = listing_cache[key]
    if sizes is None:
        return None
    total = 0
    for name in _shard_names(ref.path_in_repo):
        size = sizes.get(name)
        if not isinstance(size, int) or size <= 0:
            return None
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest):
            size -= os.path.getsize(dest)
        elif os.path.exists(dest + ".part"):
            size -= os.path.getsize(dest + ".part")
        total += max(0, size)
    return total


def cmd_pull(argv: list | None = None, prog: str = "gmlx pull") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Validate a remote GGUF's header, then download it (all shards) "
                    "to your model dir - not the HF cache. By default it lands in "
                    "the server config's first model_dirs root, under "
                    "<dir>/<org>__<repo>/. Pass several files to fetch them together "
                    "(extra bare names resolve in the first ref's repo - e.g. an "
                    "mmproj or a second quant). Files landing under a model_dirs "
                    "root are registered in the config and served immediately. "
                    "Refuses an unloadable GGUF unless --force.")
    ap.add_argument("refs", nargs="+", metavar="REF",
                    help="hf:<org>/<repo>/<file.gguf>[@rev] or an http(s):// URL; "
                         "additional bare filenames are fetched from the first "
                         "ref's repo.")
    ap.add_argument("--to", "--out", default=None, metavar="DIR",
                    help="Download into DIR exactly (default: the server config's "
                         "first model_dirs root, nesting hf files under "
                         "<dir>/<org>__<repo>/).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to read model_dirs from for the default "
                         "destination (default: search the standard config "
                         "locations).")
    ap.add_argument("--force", action="store_true",
                    help="Download even if the header check says it won't load "
                         "or the disk-space check fails.")
    ap.add_argument("--hf-source", default=None,
                    help="Treat the arch as loadable with this config override.")
    ap.add_argument("--max-mb", type=int, default=None,
                    help="Cap the remote header range-read (MB; default 128).")
    ap.add_argument("--json", action="store_true",
                    help="Emit each verdict as JSON before downloading.")
    ap.add_argument("--no-register", action="store_true",
                    help="Don't add the downloaded file(s) to the server "
                         "config's models.")
    a = ap.parse_args(argv)

    try:
        refs = _expand_refs(a.refs)
    except remote.RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    # A local ref is already on disk: skip it and pull the rest (a bare sibling
    # name colliding with a CWD file already printed its own note upstream).
    for ref in refs:
        if ref.kind == "local" and ("/" in ref.raw or ref.raw.startswith("~")):
            print(f"note: {ref.raw!r} is already on disk - skipping",
                  file=sys.stderr)
    refs = [ref for ref in refs if ref.kind != "local"]
    if not refs:
        print("error: pull needs an hf: ref or an http(s):// URL "
              "(a local path is already on disk)", file=sys.stderr)
        return 2

    # Resolve the destination base once (model-dir default nests; --to is literal).
    try:
        base_dir = (os.path.expanduser(os.path.expandvars(a.to)) if a.to
                    else _model_dir_dest(a.config))
    except remote.RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    nest = a.to is None

    all_ok = True
    downloaded: list[str] = []
    listing_cache: dict = {}
    for raw_ref in refs:
        try:
            ref = _resolve_to_file(raw_ref)
            report, n_shards = _build_report(ref, max_mb=a.max_mb)
        except remote.RemoteError as e:
            print(f"error: {e}", file=sys.stderr)
            all_ok = False
            continue

        v = _verdict(ref, report, hf_source=a.hf_source, n_shards=n_shards)
        if a.json:
            print(json.dumps(v, indent=2))
        else:
            _print_report(v)
        if not v["usable"] and not a.force:
            reasons = []
            if not v["codecs_loadable"] and v["unsupported_codecs"]:
                reasons.append(
                    "no kernel for codec(s): " + ", ".join(v["unsupported_codecs"]))
            if not v["arch_supported"] and not (v.get("mmproj") or v.get("adapter")):
                reasons.append(f"unsupported arch: {v['arch'] or '?'}")
            why = f" - {'; '.join(reasons)}" if reasons else ""
            print(f"\nrefusing to download an unloadable GGUF{why}. Pass --force to "
                  "override, or pick a different quant.", file=sys.stderr)
            all_ok = False
            continue

        dest_dir = _dest_for_ref(ref, base_dir, nest)
        need = _planned_download_bytes(ref, dest_dir, listing_cache)
        if need and not a.force:
            free = _disk_free(dest_dir)
            if free < need:
                print(f"error: not enough disk space for {ref.filename}: need "
                      f"{_human_gb(need)}, {_human_gb(free)} free at {dest_dir} "
                      "(free space, pass --to DIR on another volume, or --force)",
                      file=sys.stderr)
                all_ok = False
                continue

        try:
            downloaded += _download_ref(ref, dest_dir)
        except (remote.RemoteError, OSError) as e:
            print(f"error: download failed: {e}", file=sys.stderr)
            all_ok = False
            continue

    if downloaded:
        print(f"\ndownloaded {len(downloaded)} "
              f"file{plural_s(len(downloaded))}:")
        for p in downloaded:
            print(f"  {p}")
        if not a.no_register:
            # Close pull's own loop: a file that landed under a model_dirs
            # root becomes a served model without a separate sync-models run.
            from .server import register_downloads
            register_downloads(downloaded, a.config)
    return 0 if all_ok else 1


# rm - delete a model's files and its config entry
def _try_resolve(p, model_dirs) -> str | None:
    """resolve_path that returns None instead of raising on a miss."""
    if not p:
        return None
    from .config import resolve_path
    try:
        return resolve_path(p, model_dirs)
    except Exception:                        # noqa: BLE001 - missing is fine here
        return None


def _shard_files(path: str) -> list[str]:
    """Every existing shard of ``path``'s split set (tolerant of gaps, unlike
    preflight), plus any ``.part`` resume leftovers."""
    d, base = os.path.split(path)
    out = []
    for name in _shard_names(base):
        p = os.path.join(d, name)
        for cand in (p, p + ".part"):
            if os.path.exists(cand):
                out.append(cand)
    return out


def _rm_resolve_target(cfg, requested):
    """Resolve a requested id against aliases, config entries, and the
    discovery scan. Returns (target, model, is_configured, disc, notes);
    model is None when the id is unknown."""
    notes: list[str] = []
    target = requested
    if target in cfg.aliases and target not in cfg.models:
        real = cfg.aliases[target].split("@", 1)[0]
        notes.append(f"alias {target} -> {real}")
        target = real

    entry = cfg.models.get(target)
    disc: list = []
    if cfg.discover:
        from . import discovery
        try:
            disc = discovery.scan_dirs(
                cfg.discover, cfg.model_dirs, known_ids=set(cfg.models),
                known_paths={m.path for m in cfg.models.values()})
        except Exception:                    # noqa: BLE001 - a flaky scan dir
            disc = []
    m = entry or next((d for d in disc if d.id == target), None)
    return target, m, entry is not None, disc, notes


def _rm_deletion_plan(cfg, target, m, disc, *, keep_files, notes):
    """Files to delete (all shards + companions no other model references)
    and their sizes; explanatory notes append to ``notes``."""
    # Paths any other model (configured or discovered) still resolves to.
    others = [mc for mid, mc in cfg.models.items() if mid != target]
    others += [mc for mc in disc if mc.id != target]
    ref_paths = set()
    for mc in others:
        for p in (mc.path, mc.mmproj, mc.draft_gguf, mc.adapter):
            rp = _try_resolve(p, cfg.model_dirs)
            if rp:
                ref_paths.add(os.path.abspath(rp))

    to_delete: list[str] = []
    if not keep_files:
        if str(m.path).startswith("hf:"):
            notes.append("path is an hf: cache ref - the cached file stays "
                         "(the HF cache manages its own blobs)")
        else:
            main = _try_resolve(m.path, cfg.model_dirs)
            if main is None:
                notes.append(f"model file not found ({m.path}) - removing "
                             "the config entry only")
            elif os.path.abspath(main) in ref_paths:
                notes.append("model file kept: another model references it")
            else:
                to_delete += _shard_files(main)
        for label, comp in (("mmproj", m.mmproj), ("draft", m.draft_gguf),
                            ("adapter", m.adapter)):
            if not comp or str(comp).startswith("hf:"):
                continue
            rp = _try_resolve(comp, cfg.model_dirs)
            if rp is None:
                continue
            if os.path.abspath(rp) in ref_paths:
                notes.append(f"{label} kept: another model references it")
            else:
                to_delete += _shard_files(rp)
    to_delete = list(dict.fromkeys(to_delete))
    sizes = {p: os.path.getsize(p) for p in to_delete if os.path.exists(p)}
    return to_delete, sizes


def _rm_print_plan(target, cfg_path, *, to_delete, sizes, keep_files,
                   is_configured, aliases_to_drop, default_cleared,
                   notes) -> None:
    print(f"removing {target}:")
    for p in to_delete:
        print(f"  {p}  ({_human_gb(sizes.get(p, 0))})")
    if to_delete:
        print(f"  total {_human_gb(sum(sizes.values()))}")
    elif not keep_files:
        print("  (no files to delete)")
    if is_configured:
        print(f"  config entry: {target} (removed from {cfg_path})")
    else:
        print("  discovered model - no config entry; it leaves "
              "`gmlx list` once the file is gone")
    for name in aliases_to_drop:
        print(f"  note: dropping alias {name}")
    if default_cleared:
        print(f"  note: clearing server.defaults.model ({target})")
    for n in notes:
        print(f"  note: {n}")


def _rm_confirm() -> int | None:
    """Interactive confirmation; returns an exit code to bail with, else
    None."""
    if not sys.stdin.isatty():
        print("error: refusing to prompt without a tty; pass --yes",
              file=sys.stderr)
        return 2
    try:
        ans = input("remove? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans.strip().lower() not in ("y", "yes"):
        print("aborted", file=sys.stderr)
        return 1
    return None


def _rm_delete_files(to_delete):
    """Unlink the planned files. Returns (deleted paths, exit code)."""
    rc = 0
    deleted: list[str] = []
    for p in to_delete:
        try:
            os.remove(p)
            deleted.append(p)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"error: could not delete {p}: {e}", file=sys.stderr)
            rc = 1
    return deleted, rc


def _rm_update_config(cfg_path, target, aliases_to_drop, default_cleared, *,
                      skip_reload) -> None:
    """Drop the model entry, dead aliases, dangling assistants, and the
    cleared default from the YAML, then live-reload a running server."""
    from . import config as cfgmod

    assistants_dropped: list[str] = []

    def mutate(doc):
        models = doc.get("models")
        if isinstance(models, dict) and target in models:
            del models[target]
        al = doc.get("aliases")
        if isinstance(al, dict):
            for name in aliases_to_drop:
                if name in al:
                    del al[name]
        srv = doc.get("server")
        if isinstance(srv, dict):
            # An assistants entry whose model points at the removed id
            # (or a dropped alias) would fail validation on next load.
            gone = {target, *aliases_to_drop}
            asst = srv.get("assistants")
            if isinstance(asst, dict):
                for name in [k for k, v in asst.items()
                             if isinstance(v, dict)
                             and v.get("model") in gone]:
                    del asst[name]
                    assistants_dropped.append(name)
            if default_cleared:
                d = srv.get("defaults")
                if isinstance(d, dict) and d.get("model") == target:
                    del d["model"]

    cfgmod.edit_config_yaml(cfg_path, mutate)
    if assistants_dropped:
        print("removed dangling assistant alias(es): "
              + ", ".join(sorted(assistants_dropped)), file=sys.stderr)
    # Same live-reload as init/sync-models/pull: without it, a running
    # server keeps advertising the removed id until the next restart.
    from .server import _reload_running
    _reload_running(cfg_path, skip=skip_reload)


def _resolve_config_path(a, cfgmod) -> str | None:
    """The config path for a manage subcommand: an explicit ``--config`` (which
    must exist) else the first existing default. Prints the error and returns
    ``None`` on failure, so the caller can ``return 2``. (``cmd_profiles``
    deliberately tolerates a missing config and does not use this.)"""
    if a.config:
        cfg_path = os.path.abspath(os.path.expanduser(a.config))
        if not os.path.exists(cfg_path):
            print(f"error: no config file at {cfg_path}", file=sys.stderr)
            return None
        return cfg_path
    cfg_path = next((str(p) for p in cfgmod.default_config_paths()
                     if p.exists()), None)
    if cfg_path is None:
        searched = ", ".join(str(p) for p in cfgmod.default_config_paths())
        print(f"error: no server config found (looked at: {searched}).\n"
              "  create one with `gmlx init`, or pass `--config FILE`.",
              file=sys.stderr)
    return cfg_path


def cmd_rm(argv: list | None = None, prog: str = "gmlx rm") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Delete a model's GGUF file(s) from disk and remove its "
                    "entry from the server config. All shards are included, "
                    "plus mmproj/draft/adapter companions no other model "
                    "references. Never touches the Hugging Face cache.")
    ap.add_argument("id", metavar="ID",
                    help="Model id (or alias) from the config, or a discovered "
                         "model's id (see `gmlx list`).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to read (default: the bare-start search path, "
                         "e.g. ~/.config/gmlx/gmlx.yaml).")
    ap.add_argument("--keep-files", action="store_true",
                    help="Remove only the config entry; leave files on disk.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt.")
    ap.add_argument("--json", action="store_true",
                    help="Emit the removal result as JSON (requires --yes).")
    ap.add_argument("--no-reload", action="store_true",
                    help="Don't SIGHUP a server already running this config to "
                         "pick up the removal.")
    a = ap.parse_args(argv)

    if a.json and not a.yes:
        print("error: --json requires --yes (no prompt in JSON mode)",
              file=sys.stderr)
        return 2

    from . import config as cfgmod

    cfg_path = _resolve_config_path(a, cfgmod)
    if cfg_path is None:
        return 2

    try:
        cfg = cfgmod.load_config(cfg_path)
    except cfgmod.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    target, m, is_configured, disc, notes = _rm_resolve_target(cfg, a.id)
    if m is None:
        print(f"error: unknown model id {a.id!r} (see: gmlx list)",
              file=sys.stderr)
        return 2

    to_delete, sizes = _rm_deletion_plan(
        cfg, target, m, disc, keep_files=a.keep_files, notes=notes)
    aliases_to_drop = [name for name, tgt in cfg.aliases.items()
                       if tgt == target or tgt.startswith(target + "@")]
    default_model = cfg.defaults.model if cfg.defaults else None
    default_cleared = default_model == target

    if not a.json:
        _rm_print_plan(target, cfg_path, to_delete=to_delete, sizes=sizes,
                       keep_files=a.keep_files, is_configured=is_configured,
                       aliases_to_drop=aliases_to_drop,
                       default_cleared=default_cleared, notes=notes)

    if not a.yes:
        bail = _rm_confirm()
        if bail is not None:
            return bail

    deleted, rc = _rm_delete_files(to_delete)

    if is_configured or aliases_to_drop or default_cleared:
        _rm_update_config(cfg_path, target, aliases_to_drop, default_cleared,
                          skip_reload=a.no_reload)

    freed = sum(sizes[p] for p in deleted if p in sizes)
    if a.json:
        print(json.dumps({
            "id": target, "config": cfg_path,
            "files_deleted": deleted, "bytes_freed": freed,
            "config_entry_removed": is_configured,
            "aliases_removed": aliases_to_drop,
            "default_cleared": default_cleared, "notes": notes}, indent=2))
    elif deleted or is_configured:
        what = ([f"{len(deleted)} file{plural_s(len(deleted))}, "
                 f"{_human_gb(freed)}"] if deleted else [])
        if is_configured:
            what.append("config entry")
        print(f"removed {', '.join(what)}")
    return rc


# list - the server's discovery scan, as a table
def _human_gb(n_bytes: int, decimals: int = 1) -> str:
    from .lifecycle import human_gb
    return human_gb(n_bytes, decimals)


def _model_flags(m) -> list[str]:
    """The one-line tags for a configured/discovered model (id-addressing aside)."""
    flags = []
    if getattr(m, "mmproj", None):
        flags.append("vlm")
    if getattr(m, "speculative", False) or getattr(m, "draft_gguf", None):
        flags.append("mtp")
    if getattr(m, "adapter", None):
        flags.append("lora")
    stream = getattr(m, "stream", None)
    if stream == "cpu":
        flags.append("stream-cpu")
    elif stream == "experts":
        flags.append("stream-experts")
    if getattr(m, "pin", False):
        flags.append("pinned")
    return flags


def cmd_list(argv: list | None = None, prog: str = "gmlx list") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="List the models your server config defines - the ids you can "
                    "address in a request: the explicit `models:` entries plus any "
                    "`discover:` scan, with their source paths, aliases, and the "
                    "default model. (To list GGUF files on disk, use your shell or "
                    "`gmlx init`/`sync-models` to fold them into a config.)")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to read (default: the bare-start search path, "
                         "e.g. ~/.config/gmlx/gmlx.yaml).")
    ap.add_argument("--json", action="store_true",
                    help="Emit the listing as JSON.")
    a = ap.parse_args(argv)

    from . import config as cfgmod

    cfg_path = _resolve_config_path(a, cfgmod)
    if cfg_path is None:
        return 2

    try:
        cfg = cfgmod.load_config(cfg_path)
    except cfgmod.ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    default_model = cfg.defaults.model if cfg.defaults else None
    rows = []
    for mid, m in cfg.models.items():
        rows.append({"id": mid, "path": m.path, "profile": m.profile,
                     "source": "config", "default": mid == default_model,
                     "flags": _model_flags(m)})
    if cfg.discover:
        from . import discovery
        try:
            disc = discovery.scan_dirs(
                cfg.discover, cfg.model_dirs, known_ids=set(cfg.models),
                known_paths={m.path for m in cfg.models.values()})
        except Exception as e:                       # noqa: BLE001 - a flaky scan dir
            disc = []
            print(f"[list] discovery scan failed: {e}", file=sys.stderr)
        for m in disc:
            rows.append({"id": m.id, "path": m.path, "profile": m.profile,
                         "source": "discovered", "default": m.id == default_model,
                         "flags": _model_flags(m)})

    rows.sort(key=lambda r: r["id"])
    aliases = dict(cfg.aliases)

    if a.json:
        print(json.dumps({"config": cfg_path, "models": rows, "aliases": aliases,
                          "default_model": default_model}, indent=2))
        return 0

    n_disc = sum(1 for r in rows if r["source"] == "discovered")
    summ = f"{len(rows)} model{plural_s(len(rows))}"
    if n_disc:
        summ += f" ({len(rows) - n_disc} configured + {n_disc} discovered)"
    if aliases:
        summ += f", {len(aliases)} alias(es)"
    print(f"config: {cfg_path}   {summ}")
    if not rows:
        print("\n  (no models) - add entries under `models:` or a `discover:` scan, "
              "then `gmlx sync-models`.")
        return 0
    print()
    wid = max(len(r["id"]) for r in rows)
    for r in rows:
        mark = "*" if r["default"] else " "
        extra = list(r["flags"])
        if r["source"] == "discovered":
            extra.append("discovered")
        note = f"   [{', '.join(extra)}]" if extra else ""
        prof = f"  @{r['profile']}" if r["profile"] else ""
        print(f"{mark} {r['id']:<{wid}}{prof}{note}")
        print(f"  {' ' * wid}  {r['path']}")
    if aliases:
        print("\naliases:")
        wa = max(len(k) for k in aliases)
        for name, target in sorted(aliases.items()):
            print(f"  {name:<{wa}}  ->  {target}")
    if default_model:
        print(f"\ndefault model (when a request omits `model`): {default_model}  "
              "(* above)")
    return 0


# ps - the residency view of a running server
def cmd_ps(argv: list | None = None, prog: str = "gmlx ps") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Show the models resident in a running gmlx server "
                    "(reads /v1/metrics).")
    ap.add_argument("--url", default=None, metavar="URL",
                    help="Server base URL (default: the single managed server "
                         "if there's one, else the config's host/port, else "
                         "http://127.0.0.1:8080).")
    ap.add_argument("--host", default=None, help="Server host (alternative to "
                    "--url).")
    ap.add_argument("--port", type=int, default=None, help="Server port "
                    "(alternative to --url).")
    ap.add_argument("--api-key", default=None, metavar="KEY",
                    help="API key for a key-protected server (default: the "
                         "GMLX_API_KEY env var).")
    ap.add_argument("--json", action="store_true",
                    help="Emit the resident-model list as JSON.")
    a = ap.parse_args(argv)

    if a.url:
        root = a.url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
    else:
        # Same target resolution as status/stop/logs, so every bare lifecycle
        # verb reports on the same server.
        from . import lifecycle
        host, port = lifecycle.auto_target(a.host, a.port)
        root = f"http://{host}:{port}"
    api_key = a.api_key or os.environ.get("GMLX_API_KEY")
    req = urllib.request.Request(
        root + "/v1/metrics",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("error: the server requires an API key - pass --api-key "
                  "(or set GMLX_API_KEY)", file=sys.stderr)
            return 1
        print(f"error: {root}/v1/metrics returned HTTP {e.code}",
              file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", None) or e
        print(f"error: no gmlx server reachable at {root} ({reason}) - "
              f"start one with `gmlx serve`", file=sys.stderr)
        return 3   # not-running is a status, like `gmlx status` (actions use 1)

    srv = payload.get("server") if isinstance(payload, dict) else None
    if not isinstance(srv, dict):
        # Valid JSON that isn't our metrics shape: some other service answered.
        print(f"error: {root}/v1/metrics did not return gmlx metrics - "
              f"is something else listening there?", file=sys.stderr)
        return 1
    resident = srv.get("resident_models") or []
    resident = ([e for e in resident if isinstance(e, dict)]
                if isinstance(resident, list) else [])
    if a.json:
        print(json.dumps(resident, indent=2))
        return 0
    if not resident:
        print(f"server at {root} is up; no models resident")
        return 0
    print(f"server at {root}:")
    names = [", ".join(e.get("ids") or [])
             or os.path.basename(e.get("model_path") or "?")
             for e in resident]
    wid = max(len(n) for n in names)
    print(f"{'ID':<{wid}}  {'SIZE':>8}  {'IDLE':>7}  {'TTL':>6}  PINNED  KEPT")
    for name, e in zip(names, resident):
        ttl = e.get("ttl_s")
        ttl_str = "-" if not ttl else f"{int(ttl)}s"
        pin = "yes" if e.get("pinned") else "no"
        kept = "yes" if e.get("kept") else "no"
        print(f"{name:<{wid}}  {_human_gb(e.get('footprint_bytes') or 0):>8}  "
              f"{e.get('idle_s', 0):>6.0f}s  {ttl_str:>6}  {pin:>6}  {kept}")
        print(f"{'':<{wid}}  {e.get('model_path') or '?'}")
    return 0


# profiles - the built-in family sampling table + per-model resolution
def _fmt_pairs(sampling: dict | None, ctk: dict | None = None) -> str:
    """One-line `k=v` render of a sampling group (+ chat_template_kwargs)."""
    parts = [f"{k}={v}" for k, v in (sampling or {}).items()]
    parts += [f"{k}={v}" for k, v in (ctk or {}).items()]
    return " ".join(parts) if parts else "(none)"


def _profiles_for_model(a, cfg, cfg_path: str) -> int:
    """The `gmlx profiles <id>` body: every addressable profile resolved for
    one configured model, plus the config layers that shape the merge."""
    from . import config as cfgmod
    from . import profiles as fam_profiles

    names = cfgmod.profile_names(cfg)
    head, _req = cfgmod.split_address(a.model, names)
    head = cfg.aliases.get(head, head)
    head, _alias_req = cfgmod.split_address(head, names)
    if head not in cfg.models:
        print(f"error: unknown model {head!r} (known: {sorted(cfg.models)})",
              file=sys.stderr)
        return 2
    mc = cfg.models[head]
    fam = mc.family or "default"

    addr = (sorted(fam_profiles.BUILTIN_INTENTS | set(cfg.profiles))
            if cfg.family_defaults else sorted(cfg.profiles))
    rows = [("base", None)] + [(f"@{n}", n) for n in addr]
    resolved = {}
    for label, req in rows:
        rm = cfgmod.resolve_model(head, cfg, request_profile=req)
        resolved[label] = {"sampling": rm.sampling,
                           "chat_template_kwargs": rm.chat_template_kwargs}

    if a.json:
        print(json.dumps({"config": cfg_path, "model": head, "family": fam,
                          "resolved": resolved}, indent=2))
        return 0

    print(f"{head} - family {fam}   ({mc.path})")
    # The layers shaping the merge, so a surprising value is traceable.
    if not cfg.family_defaults:
        print("  family_defaults: false - family base + built-in intents disabled")
    if cfg.defaults.profile:
        print(f"  server default profile:  {cfg.defaults.profile}")
    ruled = cfgmod._matched_rule_profile(head, cfg.rules)
    if ruled:
        print(f"  matched rule profile:    {ruled}")
    if mc.profile:
        print(f"  model profile:           {mc.profile}")
    if mc.profiles:
        print(f"  per-model profile tweaks: {', '.join(sorted(mc.profiles))}")
    ov = (mc.overrides or {}).get("sampling") or {}
    if ov:
        print(f"  model overrides:         {_fmt_pairs(ov)}")
    print()
    wid = max(len(label) for label, _r in rows)
    for label, _r in rows:
        g = resolved[label]
        print(f"  {label:<{wid}}  "
              f"{_fmt_pairs(g['sampling'], g['chat_template_kwargs'])}")
    return 0


def cmd_profiles(argv: list | None = None, prog: str = "gmlx profiles") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Show the built-in per-family sampling table (model-card "
                    "defaults plus the addressable @intents), your config's "
                    "user profiles, and each configured model's family. With a "
                    "model id: that model's fully resolved sampling for its "
                    "base and every addressable profile.")
    ap.add_argument("model", nargs="?", default=None,
                    help="A configured model id or alias - print its resolved "
                         "sampling per profile (needs a config).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to read (default: the bare-start search path, "
                         "e.g. ~/.config/gmlx/gmlx.yaml).")
    ap.add_argument("--json", action="store_true",
                    help="Emit the table / resolution as JSON.")
    a = ap.parse_args(argv)

    from . import config as cfgmod
    from . import profiles as fam_profiles

    # The table form works with no config; the per-model form needs one.
    cfg, cfg_path = None, None
    if a.config:
        cfg_path = os.path.abspath(os.path.expanduser(a.config))
        if not os.path.exists(cfg_path):
            print(f"error: no config file at {cfg_path}", file=sys.stderr)
            return 2
    else:
        cfg_path = next((str(p) for p in cfgmod.default_config_paths()
                         if p.exists()), None)
    if cfg_path:
        try:
            cfg = cfgmod.load_config(cfg_path)
        except cfgmod.ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        from . import discovery
        discovery.fill_families(cfg)         # header-only reads; silent on misses

    if a.model:
        if cfg is None:
            print("error: `profiles <id>` needs a server config (none found - "
                  "pass --config FILE, or `gmlx init`)", file=sys.stderr)
            return 2
        return _profiles_for_model(a, cfg, cfg_path)

    if a.json:
        doc: dict = {"families": fam_profiles.describe(),
                     "intents": sorted(fam_profiles.BUILTIN_INTENTS)}
        if cfg is not None:
            doc["config"] = cfg_path
            doc["profiles"] = {
                n: {k: v for k, v in vars(p).items() if k != "name" and v}
                for n, p in cfg.profiles.items()}
            doc["models"] = {mid: (m.family or "default")
                             for mid, m in cfg.models.items()}
        print(json.dumps(doc, indent=2))
        return 0

    print("built-in sampling: model-card defaults, applied per model family")
    print("an intent a family doesn't list resolves to its base; every intent "
          "is\naddressable on every model (`<id>@coding`, a request `profile` "
          "field,\nor `run/chat --profile coding`)")
    for row in fam_profiles.describe():
        fam = row["family"]
        arches = ", ".join(row["arches"]) or "any other architecture"
        print(f"\n{fam} - {row['label']}  ({arches})")
        labels = ["base"] + [f"@{n}" for n in
                             sorted(fam_profiles.FAMILIES[fam]["intents"])]
        wid = max(len(lb) for lb in labels)
        base = row["base"]
        print(f"  {'base':<{wid}}  "
              f"{_fmt_pairs(base.get('sampling'), base.get('chat_template_kwargs'))}")
        for lb in labels[1:]:
            g = row["intents"][lb[1:]]
            print(f"  {lb:<{wid}}  "
                  f"{_fmt_pairs(g.get('sampling'), g.get('chat_template_kwargs'))}")

    if cfg is None:
        print("\n(no server config found - user profiles / model families "
              "omitted; `gmlx init` creates one)")
        return 0
    print(f"\nconfig: {cfg_path}")
    if not cfg.family_defaults:
        print("  family_defaults: false - the table above is disabled for "
              "this config")
    if cfg.profiles:
        print("user profiles:")
        for n, p in cfg.profiles.items():
            bits = []
            if p.extends:
                bits.append(f"extends {p.extends}")
            s = _fmt_pairs(p.sampling, p.chat_template_kwargs)
            if s != "(none)":
                bits.append(s)
            if p.load:
                bits.append("load: " + " ".join(f"{k}={v}"
                                                for k, v in p.load.items()))
            shadow = ("   (replaces the built-in intent)"
                      if cfg.family_defaults and n in fam_profiles.BUILTIN_INTENTS
                      else "")
            print(f"  {n}: {' | '.join(bits) or '(empty)'}{shadow}")
    if cfg.models:
        wid = max(len(mid) for mid in cfg.models)
        print("models:")
        for mid, m in sorted(cfg.models.items()):
            prof = f"   profile: {m.profile}" if m.profile else ""
            print(f"  {mid:<{wid}}  {m.family or 'default'}{prof}")
    return 0
