#!/usr/bin/env python3
"""Refresh the pull catalog's measured facts from Hugging Face.

  python scripts/refresh_pull_catalog.py                  # print the FACTS block and the walk
  python scripts/refresh_pull_catalog.py --only qwen3.8-27b
  python scripts/refresh_pull_catalog.py --check          # exit 1 when a recorded fact disagrees
  python scripts/refresh_pull_catalog.py --cache DIR      # keep header reads between runs

For each entry of gmlx/load/catalog.py it reads the model card, lists the
repo at the head of its main branch, which is what a pull downloads, range-reads
every shard header of each rung, the vision tower, the drafter and the
speech companion, and
prints what gmlx/load/catalog_facts.py should hold: the files each
quality tier maps to, their sizes, text and lookup-table bytes, the
memory class a MoE streams from, the full-precision cache cost and the
license. It also verifies what registration would
check (a supported arch, a head on a wired arch, a drafter that serves
the base and declares it) and flags a vision tower or drafter the repo
ships that the entry leaves out. Then it prints, per memory class, the
default plan each RECOMMENDED name gets, and the image models' DiT sizes
for gmlx/image/catalog_facts.py, with the encoder sizes the image catalog
records checked against their listings.

It prints; it never rewrites code. It runs at release prep, never in CI.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def strip_env() -> None:
    """Drop the cache and runtime switches, so a recorded row never
    depends on the shell that ran the script."""
    for k in list(os.environ):
        if k.startswith(("KV_", "GMLX_")) or k in ("MAX_KV_SIZE", "MLX_VLM_GGUF_SPECULATIVE"):
            del os.environ[k]


strip_env()

from gmlx.load import arch_table, catalog, remote  # noqa: E402
from gmlx.load.catalog import MEMORY_CLASSES, TIERS, Seed  # noqa: E402
from gmlx.load.discovery import (_classify_meta, declared_base_ids,  # noqa: E402
                                 _drafter_kind, _looks_like_drafter,
                                 _target_ids)
from gmlx.load.config_synth import supported_arches, trained_context_cap  # noqa: E402
from gmlx.load.gguf_meta import read_int, read_string  # noqa: E402
from gmlx.gen.prefill_decay import transient_declaration  # noqa: E402
from gmlx.load.headerscan import scan_bytes  # noqa: E402
from gmlx.load.preflight import shard_names  # noqa: E402
from gmlx.load.repo_files import (file_for_quant, is_companion, model_files,  # noqa: E402,F401
                                  quant_of)

HOST = remote._HF_HOST
GiB = 1 << 30

# Network seams, monkeypatched in tests.
hf_list_dir = remote.hf_list_dir
fetch_header_bytes = remote.fetch_header_bytes


def _hf_api_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=remote._auth_headers(url))
    with remote.http_open(req, timeout=60) as resp:
        return json.load(resp)


def _url_ok(url: str) -> bool:
    """Whether a license page answers 200."""
    req = urllib.request.Request(url, method="HEAD", headers=remote._auth_headers(url))
    try:
        with remote.http_open(req, timeout=60) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


# The quants each tier takes, best file first. A uniform file wins over a
# mixed ``_XL`` one of the same bits, with or without a ``UD-`` prefix.
TIER_QUANTS = {
    "fastest": {"dense": ("Q4_0", "IQ4_XS"), "moe": ("IQ4_XS",)},
    "balanced": ("Q4_K_M", "Q4_K_XL"),
    "high": ("Q6_K", "Q6_K_XL"),
    "xhigh": ("Q8_0", "Q8_K_XL"),
}
_SUB4 = re.compile(r"^(?:UD-)?(?:IQ|PTQ|PQ|Q)([123])_")
USABLE = ("streams", "resident", "tables_from_file")


class Problem(Exception):
    pass


# -- fetching ------------------------------------------------------------

def card(repo: str) -> dict:
    """License, gating and the head commit of a repo."""
    info = _hf_api_get(f"{HOST}/api/models/{repo}")
    data = info.get("cardData") or {}
    lic = data.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    base = data.get("base_model") or ""
    if isinstance(base, list):
        base = base[0] if base else ""
    return {"license": (lic or "").lower(), "license_name": data.get("license_name") or "",
            "license_link": data.get("license_link") or "", "gated": info.get("gated"),
            "sha": info.get("sha") or "main", "modified": str(info.get("lastModified") or ""),
            "base": str(base)}


def listing(repo: str, revision: str) -> dict:
    """``{path: size}`` of every file in the repo at ``revision``."""
    out = {}
    for entry in hf_list_dir(repo, "", revision):
        path, kind, size = entry[0], entry[1], entry[2]
        if kind == "file":
            out[path] = int(size or 0)
    return out


def header(repo: str, path: str, revision: str, cache: str | None):
    """The header scan of one repo file."""
    url = remote.hf_resolve_url(repo, path, revision)
    key = None
    if cache:
        os.makedirs(cache, exist_ok=True)
        # One entry per file and revision: a repo that re-uploads a file
        # under the same name must not answer from the older header.
        key = os.path.join(cache, (repo + "__" + path + "@" + revision).replace("/", "__"))
        if os.path.exists(key + ".bin"):
            with open(key + ".bin", "rb") as f:
                buf = f.read()
            with open(key + ".total") as f:
                total = int(f.read() or 0)
            return scan_bytes(buf, url, total)
    buf, total = fetch_header_bytes(url)
    if key:
        with open(key + ".bin", "wb") as f:
            f.write(buf)
        with open(key + ".total", "w") as f:
            f.write(str(total or 0))
    return scan_bytes(buf, url, total or 0)


def scans_for(repo: str, path: str, revision: str, cache: str | None) -> list:
    """Every shard's header scan, the metadata shard first."""
    folder = os.path.dirname(path)
    names = shard_names(os.path.basename(path))
    return [header(repo, os.path.join(folder, n) if folder else n, revision, cache)
            for n in names]


# -- the listing's model files ----------------------------------------------

def sub4(quant: str) -> bool:
    return bool(_SUB4.match(quant))


def reads_as_drafter(scan) -> bool:
    """Discovery's test for a drafter file: a drafter arch or backbone
    fields, or a head-only sidecar that keeps its base model's arch (nextn
    tensors and no trunk), the probe ``_probe_mtp_only`` makes on disk."""
    darch = read_string(scan.kv, "general.architecture")
    if _looks_like_drafter(scan.kv, darch):
        return True
    if not darch or (read_int(scan.kv, f"{darch}.nextn_predict_layers") or 0) <= 0:
        return False
    names = {t.name for t in scan.tensors}
    return (any(".nextn." in n for n in names)
            and not any(n.startswith("blk.0.") for n in names))


def resolve_tier(groups: dict, tier: str, moe: bool) -> tuple:
    """``(first shard path, quant, note)`` for a tier, or ``(None, None,
    note)`` when the repo ships no file for it."""
    quants = TIER_QUANTS[tier]
    if isinstance(quants, dict):
        quants = quants["moe" if moe else "dense"]
    for q in quants:
        for candidate in (q, "UD-" + q):
            first = file_for_quant(groups, candidate)
            if first is not None:
                quant = groups[first][0]
                note = "" if quant == quants[0] else f"{tier} falls back to {quant}"
                return first, quant, note
    return None, None, f"{tier}: no file for {' or '.join(quants)}"


# -- one entry -------------------------------------------------------------

def _license_url(repo: str, revision: str, c: dict, files: dict) -> str:
    """The first license page that answers: the card's link, a LICENSE
    file in the repo, the base model's LICENSE, the base model's page,
    else the repo page. Cards link to files their repos never had."""
    candidates = []
    link = c["license_link"]
    if link.startswith("http"):
        candidates.append(link)
    elif link:
        candidates.append(f"{HOST}/{repo}/blob/{revision}/{link.lstrip('./')}")
    for path in files:
        if os.path.basename(path).upper().startswith("LICENSE"):
            candidates.append(f"{HOST}/{repo}/blob/{revision}/{path}")
    if c.get("base"):
        candidates.append(f"{HOST}/{c['base']}/blob/main/LICENSE")
        candidates.append(f"{HOST}/{c['base']}")
    for url in candidates:
        if _url_ok(url):
            return url
    return f"{HOST}/{repo}"


def _streams_from(mp, ram_classes=MEMORY_CLASSES) -> tuple:
    """The smallest class from which every larger class runs the plan,
    and the verdict per class."""
    from gmlx.serve.capacity import working_set_for_ram
    from gmlx.stream.plan import box_plan

    ladder = {}
    for cls in ram_classes:
        ram = cls * GiB
        bp = box_plan(mp, ram_bytes=ram, ws_bytes=working_set_for_ram(ram),
                      max_buffer_bytes=int(0.63 * ram))
        ladder[cls] = bp.verdict if bp else "too_big"
    streams_from = None
    for i, cls in enumerate(ram_classes):
        if all(ladder[c] in USABLE for c in ram_classes[i:]):
            streams_from = cls
            break
    return streams_from, ladder


def recorded_context(kv, arch: str) -> int:
    """The trained context the catalog records: the header's, under the
    cap gmlx serves the file at."""
    ctx = int(read_int(kv, f"{arch}.context_length") or 0)
    cap = trained_context_cap(kv)
    return min(ctx, cap) if cap else ctx


def record_entry(seed: Seed, recorded: dict | None, cache: str | None) -> tuple:
    """``(facts, problems, notes)`` for one seed. ``recorded`` is the
    current FACTS row, which names the compact quants to keep. Every repo
    reads at the commit its main branch is at now, so one run sees one
    snapshot, and the facts record no revision: a pull reads main."""
    strip_env()
    from gmlx.load.vlm import resolve_vlm_model_type
    from gmlx.serve.capacity import boot_costs
    from gmlx.stream.plan import model_plan

    problems: list = []
    notes: list = []
    c = card(seed.repo)
    if c["gated"] not in (False, None):
        problems.append(f"{seed.repo} is gated ({c['gated']!r}); a pull needs a login")
    revision = c["sha"]
    files = listing(seed.repo, revision)
    groups = model_files(files)
    if not groups:
        raise Problem(f"{seed.repo} at {revision[:8]} lists no model GGUF")

    tiered = seed.preferred in TIERS
    first_any = next(iter(groups))
    probe = header(seed.repo, first_any, revision, cache)
    arch = read_string(probe.kv, "general.architecture") or ""
    if arch not in supported_arches():
        problems.append(f"{seed.name}: arch {arch!r} is not in ARCH_TABLE")
    moe = bool(read_int(probe.kv, f"{arch}.expert_count") or 0)

    # Which files make the rungs.
    chosen: list = []                 # (quant, tier, first shard path)
    if tiered:
        for tier in TIERS:
            first, quant, note = resolve_tier(groups, tier, moe)
            if note:
                notes.append(f"{seed.name}: {note}")
            if first is not None:
                chosen.append((quant, tier, first))
        if seed.quants:
            compacts = list(seed.quants)
        elif recorded and recorded.get("rungs"):
            compacts = [r[0] for r in recorded["rungs"] if not r[1]]
        else:
            compacts = [q for _f, (q, _ud, _s, _n) in groups.items() if sub4(q)]
        for quant in compacts:
            first = file_for_quant(groups, quant)
            if first is None:
                problems.append(f"{seed.name}: compact quant {quant} is not in the listing")
            elif not sub4(quant):
                problems.append(f"{seed.name}: compact quant {quant} is not below 4 bits")
            else:
                chosen.append((quant, "", first))
        if not any(t == seed.preferred for _q, t, _f in chosen):
            problems.append(f"{seed.name}: preferred tier {seed.preferred} has no file")
    else:
        for quant in seed.quants:
            first = file_for_quant(groups, quant)
            if first is None:
                problems.append(f"{seed.name}: quant {quant} is not in the listing")
            else:
                chosen.append((quant, "", first))
        if seed.preferred not in seed.quants:
            problems.append(f"{seed.name}: preferred quant {seed.preferred} is not listed")
    for cls, want in seed.preferred_at:
        if not any((t or q) == want for q, t, _f in chosen):
            problems.append(f"{seed.name}: preferred {want} at {cls} GB has no file")

    rungs: list = []
    scans_by_quant: dict = {}
    for quant, tier, first in chosen:
        _q, _ud, shards, nbytes = groups[first]
        scans = scans_for(seed.repo, first, revision, cache)
        scans_by_quant[quant] = scans
        mp = model_plan(scans)
        nextn = read_int(scans[0].kv, f"{arch}.nextn_predict_layers") or 0
        head = nextn > 0 and arch_table.mtp_wired(arch)
        if nextn > 0 and not head:
            notes.append(f"{seed.name} {quant}: a head on an arch with no wired target; "
                         "recorded without it")
        streams_from = None
        if moe:
            streams_from, ladder = _streams_from(mp)
            notes.append(f"{seed.name} {quant}: " + "  ".join(
                f"{cls}:{v}" for cls, v in ladder.items()))
        rungs.append((quant, tier, first, int(nbytes), int(mp.total_bytes),
                      int(mp.table_bytes), streams_from, bool(head)))
    rungs.sort(key=lambda r: r[3])
    if tiered:
        tiers_ = [r for r in rungs if r[1]]
        order = [TIERS.index(r[1]) for r in tiers_]
        if order != sorted(order):
            problems.append(f"{seed.name}: tiers do not ascend in bytes")

    # The entry's cost row, from its preferred rung.
    pref_quant = next((q for q, t, _f in chosen if (t or q) == seed.preferred), None)
    scans = scans_by_quant.get(pref_quant) or next(iter(scans_by_quant.values()))
    got = boot_costs(None, env={}, scans=scans)
    kv_cost, heads, fused_head_dim = (), 0, 0
    if got:
        cfg, costs, heads, _kv_fp16 = got
        kv_cost = catalog.serialise_costs(costs).costs
        decl = transient_declaration(cfg)
        fused_head_dim = int(decl.head_dim) if decl else 0
    else:
        problems.append(f"{seed.name}: boot_costs priced nothing from the header")
    trained_ctx = recorded_context(scans[0].kv, arch)
    params = read_string(scans[0].kv, "general.size_label") or ""
    n_params = sum(math.prod(t.shape) for s in scans for t in s.tensors)
    model_cls = _classify_meta(scans[0].kv, basename=os.path.basename(chosen[0][2]),
                               path=chosen[0][2])
    targets = _target_ids(model_cls)

    # The vision tower.
    mmproj_bytes = 0
    modalities: tuple = ()
    if seed.mmproj:
        if seed.mmproj not in files:
            problems.append(f"{seed.name}: {seed.mmproj} is not in the listing")
        else:
            mmproj_bytes = files[seed.mmproj]
            mm = header(seed.repo, seed.mmproj, revision, cache)
            try:
                resolve_vlm_model_type(arch, mm.kv)
            except Exception as e:  # the pair gmlx cannot load
                problems.append(f"{seed.name}: tower {seed.mmproj} does not load with "
                                f"{arch}: {e}")
            modalities = tuple(word for word, key in (
                ("vision", "clip.has_vision_encoder"),
                ("audio", "clip.has_audio_encoder"),
                ("speech", "clip.has_gen_audio_encoder")) if mm.kv.get(key))
            if seed.mmproj_f16:
                # The F16 twin a pull takes on M1 and M2: the same tensors,
                # so the same size to within its header.
                twin = files.get(seed.mmproj_f16)
                if twin is None:
                    problems.append(f"{seed.name}: {seed.mmproj_f16} is not in the listing")
                elif abs(twin - mmproj_bytes) > 1 << 20:
                    problems.append(
                        f"{seed.name}: {seed.mmproj_f16} is {twin} bytes and "
                        f"{seed.mmproj} is {mmproj_bytes}")
    else:
        towers = [p for p in files if os.path.basename(p).lower().startswith("mmproj")
                  and p not in seed.ignored_files]
        if towers:
            problems.append(f"{seed.name}: the repo ships a vision tower the entry does "
                            f"not list: {', '.join(sorted(towers))}")

    # The drafter.
    drafter = None
    native = bool(rungs) and all(r[7] for r in rungs)
    if seed.drafter_file:
        if seed.drafter_repo == seed.repo:
            d_rev, d_files = revision, files
        else:
            d_rev = card(seed.drafter_repo)["sha"]
            d_files = listing(seed.drafter_repo, d_rev)
        if seed.drafter_file not in d_files:
            problems.append(f"{seed.name}: drafter {seed.drafter_file} is not in "
                            f"{seed.drafter_repo}")
        else:
            d = header(seed.drafter_repo, seed.drafter_file, d_rev, cache)
            darch = read_string(d.kv, "general.architecture")
            kind = _drafter_kind(d.kv, darch) or ""
            if not reads_as_drafter(d):
                problems.append(f"{seed.name}: {seed.drafter_file} does not read as a drafter")
            if not arch_table.mtp_wired(arch):
                problems.append(f"{seed.name}: arch {arch} has no wired speculation target "
                                "for a drafter")
            if arch_table.drafter_serves(darch, arch) is False:
                problems.append(f"{seed.name}: a {darch} drafter does not serve {arch}")
            if seed.drafter_repo != seed.repo and not seed.cross_base:
                declared = set(declared_base_ids(d.kv))
                if not declared & targets:
                    problems.append(f"{seed.name}: drafter declares {sorted(declared)}, the "
                                    f"model answers to {sorted(targets)}; set cross_base")
            drafter = (kind, int(d_files[seed.drafter_file]))
    elif not native:
        for path in files:
            if not path.endswith(".gguf") or path in seed.ignored_files:
                continue
            if not is_companion(path) or os.path.basename(path).lower().startswith("mmproj"):
                continue
            d = header(seed.repo, path, revision, cache)
            if reads_as_drafter(d):
                problems.append(f"{seed.name}: the repo ships a drafter the entry does not "
                                f"list: {path}")

    # The speech companion.
    asr = None
    if seed.asr_file:
        a_rev = card(seed.asr_repo)["sha"]
        a_files = listing(seed.asr_repo, a_rev)
        if seed.asr_file not in a_files:
            problems.append(f"{seed.name}: speech companion {seed.asr_file} is not in "
                            f"{seed.asr_repo}")
        else:
            a = header(seed.asr_repo, seed.asr_file, a_rev, cache)
            aarch = read_string(a.kv, "general.architecture")
            if aarch not in arch_table.ASR_ARCHES:
                problems.append(f"{seed.name}: {seed.asr_file} is arch {aarch!r}, not a "
                                "speech companion")
            if arch_table.ASR_COMPANION_REPOS.get(arch) != seed.asr_repo.split("/")[-1]:
                problems.append(f"{seed.name}: {arch} does not pair with {seed.asr_repo}")
            asr = int(a_files[seed.asr_file])

    facts = {
        "params": params, "n_params": n_params, "arch": arch, "moe": moe,
        "native_mtp": native, "license": c["license"],
        "license_url": _license_url(seed.repo, "main", c, files),
        "license_name": c["license_name"] if c["license"] == "other" else "",
        "mmproj_bytes": int(mmproj_bytes), "drafter": drafter,
        "heads": int(heads or 0), "trained_ctx": trained_ctx,
        "kv_cost": tuple(kv_cost), "rungs": tuple(rungs),
    }
    if asr is not None:
        facts["asr"] = asr
    if fused_head_dim:
        facts["prefill_fused_head_dim"] = fused_head_dim
    if set(modalities) - {"vision"}:
        # A projector that is more than a vision tower.
        facts["modalities"] = modalities
    if c["license"] == "other" and not seed.license_note:
        problems.append(f"{seed.name}: license is other ({c['license_name']}); the seed "
                        "needs a license_note")
    return facts, problems, notes


# -- output ----------------------------------------------------------------

def facts_text(facts: dict) -> str:
    lines = ['"""The measured facts of the pull catalog, printed whole by',
             '``scripts/refresh_pull_catalog.py`` and pasted here. Edit nothing by',
             'hand; ``catalog._SEED`` holds the words."""', "", "FACTS: dict = {"]
    for name, f in facts.items():
        lines.append(f"    {name!r}: {{")
        for key in ("params", "n_params", "arch", "moe", "native_mtp", "license",
                    "license_url", "license_name", "mmproj_bytes", "drafter", "heads",
                    "trained_ctx", "kv_cost"):
            lines.append(f"        {key!r}: {f[key]!r},")
        if f.get("asr"):
            lines.append(f"        'asr': {f['asr']!r},")
        if f.get("prefill_fused_head_dim"):
            lines.append("        'prefill_fused_head_dim': "
                         f"{f['prefill_fused_head_dim']!r},")
        if f.get("modalities"):
            lines.append(f"        'modalities': {tuple(f['modalities'])!r},")
        lines.append("        'rungs': (")
        for r in f["rungs"]:
            lines.append(f"            {tuple(r)!r},")
        lines.append("        ),")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compare(name: str, recorded: dict | None, fresh: dict) -> list:
    """What differs between the recorded row and the fresh one."""
    if recorded is None:
        return [f"{name}: no recorded facts"]
    diffs = []
    for key, value in fresh.items():
        old = recorded.get(key)
        if key == "rungs":
            old_set, new_set = set(map(tuple, old or ())), set(map(tuple, value))
            for r in sorted(new_set - old_set):
                diffs.append(f"{name}: rung now {r}")
            for r in sorted(old_set - new_set):
                diffs.append(f"{name}: rung gone {r}")
        elif key == "kv_cost":
            if set(map(tuple, old or ())) != set(map(tuple, value)):
                diffs.append(f"{name}: kv_cost changed")
        elif old != value:
            diffs.append(f"{name}: {key} {old!r} -> {value!r}")
    return diffs


def record_images() -> tuple:
    """``(dit_bytes, problems)``: every image recipe's DiT file size per
    quant from its repo listing at main, for ``gmlx/image/catalog_facts
    .DIT_BYTES``, and the encoder sizes the image catalog records checked
    against their listings."""
    from gmlx.image import catalog as ic

    out, problems, lists = {}, [], {}

    def files(repo: str) -> dict:
        if repo not in lists:
            lists[repo] = listing(repo, "main")
        return lists[repo]

    for r in ic.RECIPES.values():
        row = {}
        for q in r.dit_quants:
            name = r.dit_name(q)
            if name in files(r.dit_repo):
                row[q] = files(r.dit_repo)[name]
            else:
                problems.append(f"{r.name}: {name} is not in {r.dit_repo}")
        out[r.name] = row
        enc = files(r.encoder_repo)
        tower = enc.get(r.mmproj_file, 0) if r.mmproj_file else 0
        for q in r.encoder_quants:
            name = r.encoder_name(q)
            if name not in enc:
                problems.append(f"{r.name}: {name} is not in {r.encoder_repo}")
                continue
            recorded = ic.ENCODER_BYTES.get(name, r.encoder_bytes if q == "Q8_0" else None)
            if recorded is not None and recorded != enc[name] + tower:
                problems.append(f"{r.name}: {name} with its tower is {enc[name] + tower} "
                                f"bytes; the image catalog records {recorded}")
    return out, problems


def image_facts_text(dit: dict) -> str:
    lines = ['"""The measured facts of the image catalog, printed whole by',
             '``scripts/refresh_pull_catalog.py`` and pasted here. Edit nothing by',
             'hand; ``catalog.RECIPES`` holds the words."""', "",
             "# Each recipe's DiT file size per quant, from its repo listing at main.",
             "DIT_BYTES: dict = {"]
    for name, row in dit.items():
        lines.append(f"    {name!r}: {{")
        for q, n in row.items():
            lines.append(f"        {q!r}: {n},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def compare_images(recorded: dict, fresh: dict) -> list:
    """What differs between the recorded DiT sizes and the fresh ones."""
    diffs = []
    for name, row in fresh.items():
        old = recorded.get(name)
        if old is None:
            diffs.append(f"{name}: no recorded DiT sizes")
            continue
        for q, n in row.items():
            if old.get(q) != n:
                diffs.append(f"{name}: {q} DiT {old.get(q)!r} -> {n!r}")
        for q in set(old) - set(row):
            diffs.append(f"{name}: {q} DiT gone")
    return diffs


def walk_text(models: dict) -> str:
    """Per memory class, the default plan of each RECOMMENDED name."""
    lines = ["Default plans per memory class (rung, add-ons, state, room with the "
             "add-ons / with them off):"]
    for cls in catalog.MEMORY_CLASSES:
        box = catalog.box_for_class(cls)
        for name in catalog.RECOMMENDED.get(cls, ()):
            m = models.get(name)
            if m is None:
                lines.append(f"{cls:>4} GB  {name:<26} not in the catalog")
                continue
            dp = catalog.default_plan(m, box)
            if dp is None:
                need = catalog.needs_class_model(m)
                lines.append(f"{cls:>4} GB  {name:<26} no plan; needs {need} GB")
                continue
            addons = ", ".join(a for a, on in (("images", dp.mmproj), ("faster replies", dp.drafter)) if on) or "no add-ons"
            state = "streams" if dp.streaming else "resident"
            with_on = catalog.context_estimate(m, dp.rung, box, catalog.plan_addon_bytes(dp),
                                               catalog.plan_speculates(dp))
            with_off = catalog.context_estimate(m, dp.rung, box, 0,
                                                catalog.speculates(m, dp.rung, False,
                                                                   dp.streaming))

            def k(v):
                return "n/a" if v is None else f"{v // 1024}K"

            lines.append(f"{cls:>4} GB  {name:<26} {dp.rung.tier or dp.rung.quant:<10} "
                         f"{dp.rung.quant:<14} {catalog.human_gb(catalog.plan_bytes(dp)):>9}  "
                         f"{addons:<24} {state:<9} {k(with_on)} / {k(with_off)}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", action="append", default=[], help="an entry name; repeatable")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when a recorded fact disagrees or a check fails")
    ap.add_argument("--cache", default=None,
                    help="a folder that keeps header reads, one per file and revision")
    a = ap.parse_args(argv)
    strip_env()
    from gmlx.load.catalog_facts import FACTS

    seeds = [s for s in catalog._SEED if not a.only or s.name in a.only]
    fresh, failures, notes = {}, [], []
    for seed in seeds:
        print(f"== {seed.name} ({seed.repo})", file=sys.stderr)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                facts, problems, seed_notes = record_entry(seed, FACTS.get(seed.name), a.cache)
        except (Problem, remote.RemoteError) as e:
            failures.append(str(e))
            continue
        fresh[seed.name] = facts
        failures.extend(problems)
        notes.extend(seed_notes)
        if a.check:
            failures.extend(compare(seed.name, FACTS.get(seed.name), facts))
            if not _url_ok(facts["license_url"]):
                failures.append(f"{seed.name}: license page {facts['license_url']} does not answer")
    image_dit = None
    if not a.only:
        from gmlx.image.catalog_facts import DIT_BYTES

        print("== image models", file=sys.stderr)
        try:
            image_dit, problems = record_images()
            failures.extend(problems)
            if a.check:
                failures.extend(compare_images(DIT_BYTES, image_dit))
        except remote.RemoteError as e:
            failures.append(f"image models: {e}")
    for line in notes:
        print("note: " + line, file=sys.stderr)
    for line in failures:
        print("FAIL: " + line, file=sys.stderr)
    if not a.check:
        merged = dict(FACTS)
        merged.update(fresh)
        print(facts_text({s.name: merged[s.name] for s in catalog._SEED if s.name in merged}))
        if image_dit is not None:
            print("# ---- gmlx/image/catalog_facts.py ----")
            print(image_facts_text(image_dit))
    models = {name: catalog._build(catalog.SEEDS[name], f)
              for name, f in {**FACTS, **fresh}.items() if name in catalog.SEEDS}
    print(walk_text(models))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
