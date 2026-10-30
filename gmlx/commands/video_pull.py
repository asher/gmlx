"""``gmlx video pull <recipe>``: the files a video recipe runs on.

Each file a model folder already holds is used where it is, found by its
weights (``gmlx.video.files``). The rest downloads after a summary of
files, sizes and free disk, into a staging folder beside its path, where
an interrupted transfer resumes, and lands only once its sha256 is the
recipe's. The entry is then added under ``video_models:`` in the server
config.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

import gmlx.load.remote as remote
from gmlx.video import catalog

_ROLE = {"dit": "DiT", "encoder": "text encoder", "mmproj": "vision tower",
         "vae": "video VAE", "audio_vae": "audio VAE",
         "tiny": "preview decoder", "lora": "LoRA"}
# The files the smallest run prices (_smallest_run).
_PRICED = frozenset({"dit", "encoder", "tiny", "audio_vae", "lora"})


@dataclass
class Part:
    role: str
    pin: catalog.Pin
    path: str
    found: bool                     # on disk already
    need: int = 0                   # bytes still to download


@dataclass
class Plan:
    recipe: catalog.VideoRecipe
    parts: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    # Each role's parts to choose from: the files found, then the download.
    choices: dict = field(default_factory=dict, repr=False)

    def part(self, role: str) -> Part | None:
        return next((p for p in self.parts if p.role == role), None)


def _dest(pin: catalog.Pin, base_dir: str) -> str:
    return os.path.join(base_dir, pin.repo.replace("/", "__"), pin.file)


def _stage(dest: str, pin: catalog.Pin) -> str:
    """The staging folder of a download to ``dest``. It stays after a
    failed transfer, so the next pull resumes there."""
    return os.path.join(os.path.dirname(dest),
                        f".gmlx-pull-{pin.sha256[:16]}")


def _staged(dest: str, pin: catalog.Pin) -> str:
    return os.path.join(_stage(dest, pin), pin.name if pin.url else pin.file)


def _remaining(dest: str, pin: catalog.Pin) -> int:
    got = _staged(dest, pin)
    if os.path.exists(got):
        return 0
    if os.path.exists(got + ".part"):
        return max(0, pin.size - os.path.getsize(got + ".part"))
    return pin.size


def _plan(r: catalog.VideoRecipe, pick: dict, choices: dict) -> Plan:
    parts = list(pick.values())
    return Plan(r, parts, [p for p in parts
                           if not p.found and os.path.lexists(p.path)],
                choices=choices)


def plan_pull(r: catalog.VideoRecipe, *, roots: list, base_dir: str) -> Plan:
    """What the pull uses from disk and what it downloads: for each file,
    the recipe's file found on disk, or else another file of its weights,
    or else the download."""
    from gmlx.video import files
    from gmlx.video.registry import get_family

    fam = get_family(r.family)
    found = files.scan(roots, fam)
    choices = {}
    for role, pin in r.pins().items():
        if role == "lora":
            path = files.find_pinned(roots, pin)
            paths = [] if path is None else [path]
        else:
            # A DiT of unverified qkv layout is not taken for the recipe.
            hits = [f for f in found if f.role == role and (
                role != "dit" or (f.variant == r.variant
                                  and f.layout != "unverified"))]
            # The pinned file first, then any file of the same weights.
            hits.sort(key=lambda f: (os.path.basename(f.path) != pin.name,
                                     os.path.getsize(f.path) != pin.size,
                                     f.path))
            paths = [f.path for f in hits]
        dest = _dest(pin, base_dir)
        choices[role] = [Part(role, pin, p, True) for p in paths] + [
            Part(role, pin, dest, False, _remaining(dest, pin))]
    return _plan(r, {role: c[0] for role, c in choices.items()}, choices)


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _need(plan: Plan) -> int:
    return sum(p.need for p in plan.parts if not p.found)


def _print_plan(plan: Plan, free: int | None) -> None:
    r = plan.recipe
    print(f"{r.name}: {r.title}")
    for p in plan.parts:
        how = "found" if p.found else f"download {_gb(p.need)}"
        print(f"  {_ROLE[p.role]:<16} {p.pin.name}")
        print(f"  {'':<16} {how}: {p.path}")
    need = _need(plan)
    if need:
        print(f"to download: {_gb(need)}"
              + (f", {_gb(free)} free" if free is not None else ""))
    print(f"license: {r.license} ({r.license_url})")
    if r.license_note:
        print(f"  {r.license_note}")
    print(f"encoder license: {r.encoder_license}; video and audio VAEs: "
          f"{r.vae_license}; preview decoder: {r.tiny_license}"
          + ("" if r.lora is None else
             f"; LoRA: {r.lora_license or 'none stated on its card'}"))


def _plan_json(plan: Plan, free: int | None) -> dict:
    r = plan.recipe
    return {"recipe": r.name, "download_bytes": _need(plan),
            "free_bytes": free, "license": r.license,
            "license_url": r.license_url,
            "license_note": r.license_note or None,
            "encoder_license": r.encoder_license,
            "vae_license": r.vae_license, "tiny_license": r.tiny_license,
            "lora_license": (r.lora_license or None) if r.lora else None,
            "blocked": [p.path for p in plan.blocked],
            "parts": [{"role": p.role, "file": p.pin.name, "path": p.path,
                       "found": p.found, "need": p.need}
                      for p in plan.parts]}


def _smallest_run(plan: Plan):
    """The memory plan of the smallest run, 5 frames at 32x32 from a
    one-token prompt with the preview decoder, on the files the pull
    registers: a file found on disk at its own size, a download at its
    pin's."""
    from gmlx.video import memfit
    from gmlx.video.families.h3 import family as hf
    from gmlx.video.registry import get_family

    fam = get_family(plan.recipe.family)
    size = {p.role: memfit.file_bytes(p.path) if p.found else p.pin.size
            for p in plan.parts}
    side = 32
    rows, audio = memfit.clip_rows(fam, frames=hf.MIN_FRAMES, width=side,
                                   height=side, text=1)
    return memfit.plan_memory(
        fam.name, dit=size["dit"], encoder=size["encoder"], vae=size["tiny"],
        tiny=True, audio_vae=size["audio_vae"],
        loras=[size["lora"]] if "lora" in size else [], rows=rows, text=1,
        audio_latents=audio, pixels=side * side)


def memory_refusal(plan: Plan) -> str | None:
    """Why no clip of the recipe fits this Mac's memory, or None."""
    from gmlx.serve import capacity
    from gmlx.video.registry import get_family

    priced = _smallest_run(plan)
    if priced.fits or capacity.overcommit():
        return None
    return (f"the smallest {get_family(plan.recipe.family).title} clip does "
            f"not fit in memory: {priced.describe()}")


def fit_plan(plan: Plan) -> Plan:
    """``plan``, or when no clip fits this Mac's memory with its files,
    a plan one fits with, its notes naming each file it leaves.

    Each file the check prices, in turn, takes the first of its choices a
    clip fits with: the file ``plan`` took, the other files found from
    the largest down, then the download. The files not yet chosen count at
    their smallest choice."""
    from gmlx.video import memfit

    if memory_refusal(plan) is None:
        return plan
    r, choices = plan.recipe, plan.choices

    def size(p: Part) -> int:
        return memfit.file_bytes(p.path) if p.found else p.pin.size

    def fits(pick: dict) -> bool:
        return memory_refusal(_plan(r, pick, choices)) is None

    pick = {role: min(c, key=size) if role in _PRICED else c[0]
            for role, c in choices.items()}
    if not fits(pick):
        return plan
    for role, c in choices.items():
        if role in _PRICED:
            tries = [c[0], *sorted(c[1:-1], key=size, reverse=True), c[-1]]
            pick[role] = next(t for t in tries if fits({**pick, role: t}))
    out = _plan(r, pick, choices)
    for p, q in zip(plan.parts, out.parts):
        if p.path != q.path:
            how = f"uses {q.path}" if q.found else f"downloads {q.pin.name}"
            out.notes.append(f"{p.path} does not fit this Mac's memory, so "
                             f"the pull {how}")
    return out


def _made_dirs(path: str) -> list:
    """The folders ``os.makedirs(path)`` would make, deepest first."""
    made = []
    d = os.path.abspath(path)
    while not os.path.exists(d) and os.path.dirname(d) != d:
        made.append(d)
        d = os.path.dirname(d)
    return made


def _fetch(part: Part) -> str:
    """Download ``part`` into its staging folder, check its sha256 there,
    and move it into place. A failed transfer keeps the folder, so the
    next pull resumes; a wrong sha256 removes it."""
    from gmlx.commands import manage
    from gmlx.image.catalog import sha256_file

    pin = part.pin
    stage = _stage(part.path, pin)
    made = _made_dirs(os.path.dirname(stage))
    os.makedirs(stage, exist_ok=True)
    print(f"downloading {pin.repo}/{pin.file}", file=sys.stderr)
    if pin.url:
        got = _staged(part.path, pin)
        _http_download(pin.url, got)
    else:
        got = manage._hf_download(pin.repo, pin.file, pin.revision, stage)
    sha = sha256_file(got)
    if sha != pin.sha256:
        shutil.rmtree(stage, ignore_errors=True)
        for d in made:
            try:
                os.rmdir(d)
            except OSError:
                break
        raise remote.RemoteError(
            f"the download has sha256 {sha}, and {pin.file} of "
            f"{pin.repo} at {pin.revision[:12]} has {pin.sha256}")
    if os.path.lexists(part.path):
        shutil.rmtree(stage, ignore_errors=True)
        raise remote.RemoteError(f"{part.path} appeared during the pull")
    os.replace(got, part.path)
    shutil.rmtree(stage, ignore_errors=True)
    part.found = True
    return part.path


def _http_download(url: str, dest: str) -> None:
    """A URL to ``dest``, resuming from a ``.part`` beside it."""
    from gmlx.commands import manage

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    manage._url_download(url, dest)


def _entry(plan: Plan, cfg) -> dict:
    from gmlx.load.discovery import _rel

    dirs = cfg.model_dirs
    r = plan.recipe
    out = {"path": _rel(plan.part("dit").path, dirs),
           "encoder": _rel(plan.part("encoder").path, dirs),
           "encoder_mmproj": _rel(plan.part("mmproj").path, dirs),
           "vae": _rel(plan.part("vae").path, dirs),
           "audio_vae": _rel(plan.part("audio_vae").path, dirs),
           "tiny_vae": _rel(plan.part("tiny").path, dirs)}
    lora = plan.part("lora")
    if lora is not None:
        out["adapters"] = [_rel(lora.path, dirs)]
    out["defaults"] = {"steps": r.steps, "shift": r.shift, "size": r.size,
                       "seconds": r.seconds}
    return out


def _entry_id(name: str, cfg) -> str:
    """The recipe's name, or the first ``name-N`` no other block takes."""
    taken = (set(cfg.models) | set(cfg.image_models) | set(cfg.aliases)
             | set(cfg.assistants) | set(cfg.video_models))
    cand, n = name, 2
    while cand in taken:
        cand = f"{name}-{n}"
        n += 1
    return cand


def _registered(plan: Plan, cfg) -> str | None:
    """The video entry that already runs this DiT with these adapters."""
    from gmlx.commands.manage import _try_resolve

    def real(p):
        rp = _try_resolve(p, cfg.model_dirs)
        return os.path.realpath(rp) if rp else None

    dit = os.path.realpath(plan.part("dit").path)
    lora = plan.part("lora")
    loras = [os.path.realpath(lora.path)] if lora is not None else []
    for vid, m in cfg.video_models.items():
        if real(m.path) == dit and [real(a) for a, _st in m.adapters] == loras:
            return vid
    return None


def register(plan: Plan, cfg_path: str, out=None) -> str | None:
    """Add the recipe under ``video_models:``, named as the recipe unless
    another entry takes that id. Returns the id, or None when an entry
    runs this DiT with these adapters already or a file lies outside
    every model_dirs root. Its lines go to ``out`` (default stdout)."""
    from gmlx.config import edit_config_yaml, load_config

    out = out or sys.stdout
    cfg = load_config(cfg_path)
    roots = [os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
             for d in cfg.model_dirs]
    for p in plan.parts:
        ap = os.path.abspath(p.path)
        if not any(ap == r or ap.startswith(r + os.sep) for r in roots):
            print(f"note: {p.path} is outside the config's model_dirs, so "
                  f"{plan.recipe.name} is not registered", file=sys.stderr)
            return None
    entry = _entry(plan, cfg)
    name = plan.recipe.name
    old = _registered(plan, cfg)
    if old is not None:
        print(f"{name} is already configured as {old} in {cfg_path}",
              file=out)
        return None
    vid = _entry_id(name, cfg)

    def mutate(doc):
        from ruamel.yaml.comments import CommentedMap

        block = doc.get("video_models")
        if not isinstance(block, dict):
            block = CommentedMap()
            doc["video_models"] = block
        block[vid] = CommentedMap(entry)

    edit_config_yaml(cfg_path, mutate)
    print(f"registered {vid} in {cfg_path}", file=out)
    return vid


def _parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Fetch the files of a video recipe, or find them in "
                    "the model folders, and add it under video_models in "
                    "the config. Recipes: " + ", ".join(catalog.RECIPES))
    p.add_argument("recipe", metavar="RECIPE", choices=sorted(
        catalog.RECIPES), help="The recipe, such as h3.")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="The config to register the entry in (default: "
                        "the usual config search).")
    p.add_argument("--dir", default=None, metavar="DIR",
                   help="Where downloads go (default: the config's first "
                        "model_dirs folder).")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Download without asking.")
    p.add_argument("--force", action="store_true",
                   help="Download even when the free disk space looks too "
                        "small, or this Mac's chip or memory does not run "
                        "the model.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what the pull finds and downloads, and stop.")
    p.add_argument("--json", action="store_true",
                   help="Print the plan as JSON on stdout.")
    return p


def cmd_pull(argv: list[str], prog: str = "gmlx video pull") -> int:
    import gmlx.config as cfgmod

    a = _parser(prog).parse_args(argv)
    r = catalog.recipe(a.recipe)
    cfg_path = (os.path.expanduser(a.config) if a.config
                else next((str(c) for c in cfgmod.default_config_paths()
                           if os.path.exists(c)), None))
    cfg = cfgmod.load_config(cfg_path) if cfg_path else None
    roots = list(cfg.model_dirs) if cfg else []

    def expand(d):
        return os.path.expanduser(os.path.expandvars(d))

    base = expand(a.dir) if a.dir else (expand(roots[0]) if roots else None)
    if base is None:
        why = ("no config was found" if cfg is None else
               f"{cfg_path} has no model_dirs")
        print(f"error: {why}; pass --dir, or run gmlx init", file=sys.stderr)
        return 2
    if a.dir and a.dir not in roots:
        roots.append(a.dir)
    from gmlx.video.registry import chip_refusal, get_family

    chip = chip_refusal(get_family(r.family))
    plan = plan_pull(r, roots=roots, base_dir=base)
    # On a chip it refuses, or with an entry on this DiT and LoRA
    # already, the pull keeps the files it finds.
    if (a.force or not chip) and not (cfg and _registered(plan, cfg)):
        plan = fit_plan(plan)
        for n in plan.notes:
            print(f"note: {n}", file=sys.stderr)
    from gmlx.commands import manage

    free = manage._disk_free(base)
    # Under --json, stdout carries the plan alone.
    talk = sys.stderr if a.json else sys.stdout
    if a.json:
        print(json.dumps(_plan_json(plan, free), indent=2))
    else:
        _print_plan(plan, free)
    sys.stdout.flush()
    why = chip or memory_refusal(plan)
    if why and not a.force:
        print(f"error: {why}; pass --force to download anyway",
              file=sys.stderr)
        return 1
    if plan.blocked:
        for p in plan.blocked:
            print(f"error: {p.path} is there and is not the {r.title} "
                  f"{_ROLE[p.role]}; move it away and pull again",
                  file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    todo = [p for p in plan.parts if not p.found]
    need = _need(plan)
    if need and free < need and not a.force:
        print(f"error: not enough disk space: need {_gb(need)}, "
              f"{_gb(free)} free at {base} (free space, pass --dir on "
              "another volume, or --force)", file=sys.stderr)
        return 1
    if todo and not a.yes:
        if not sys.stdin.isatty():
            print("error: pass --yes to download without a terminal",
                  file=sys.stderr)
            return 1
        print("download? [y/N] ", end="", file=talk, flush=True)
        if input().strip().lower() not in ("y", "yes"):
            return 1
    try:
        for p in todo:
            _fetch(p)
    except (remote.RemoteError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if cfg_path:
        register(plan, cfg_path, out=talk)
    else:
        print(f"note: no config was found, so {r.name} is not registered",
              file=sys.stderr)
    return 0
