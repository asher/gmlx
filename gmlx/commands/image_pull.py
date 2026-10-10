"""``gmlx pull <image model>``: a DiT, its text encoder and its VAE.

A recipe name (``gmlx pull z-image-turbo``) pulls the three files an image
model needs. Each part a model folder already holds is used where it is:
the folders are searched by weight identity (the DiT fingerprint, the
encoder and vision tower identity digests, the VAE source digest), and a
configured chat model that is the encoder becomes the entry's shared
encoder. The rest downloads after a summary of files, sizes and free disk.
The VAE comes as safetensors from its source repository, is checked
against the recipe's sha256, packed to GGUF and checked again, and the
source is deleted unless ``--keep-source``. The entry is then added under
``image_models:`` in the server config.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import gmlx.load.remote as remote
from gmlx.image import catalog

PARTS = ("dit", "encoder", "vae")
_ROLE = {"dit": "DiT", "encoder": "text encoder", "mmproj": "vision tower",
         "vae": "VAE"}


@dataclass
class Part:
    """One file of the pull: where it is on disk, or what downloads."""
    role: str                       # dit, encoder, mmproj, vae
    name: str                       # file name shown in the summary
    path: str | None = None         # on disk, or where it lands
    repo: str | None = None         # set when it downloads
    files: tuple = ()               # repo files to download
    root: str | None = None         # the folder they download into
    revision: str = "main"
    size: int | None = None         # bytes of the download, or on disk
    need: int | None = None         # bytes still to download
    chat_id: str | None = None      # the configured chat model it is
    wanted: bool = True             # inside --only

    @property
    def download(self) -> bool:
        return self.repo is not None


@dataclass
class Plan:
    recipe: catalog.Recipe
    dit_quant: str
    encoder_quant: str
    memory_class: int
    parts: list = field(default_factory=list)
    default: bool = True            # the quants of the memory class
    blocked: list = field(default_factory=list)  # parts whose path is taken
    size: str | None = None         # the defaults.size the entry gets

    def part(self, role: str) -> Part | None:
        return next((p for p in self.parts if p.role == role), None)


def _parse_only(raw: str | None) -> tuple:
    if not raw:
        return PARTS
    got = tuple(dict.fromkeys(s.strip().lower() for s in raw.split(",")
                              if s.strip()))
    bad = [s for s in got if s not in PARTS]
    if bad or not got:
        raise ValueError(f"--only takes a comma list of {', '.join(PARTS)} "
                         f"(got {raw!r})")
    return got


def _quant_of(path: str) -> str:
    from gmlx.load.discovery import quant_tag

    return (quant_tag(os.path.basename(path)) or "").upper()


def _same_quant(path: str, quant: str) -> bool:
    q = quant.upper()
    return _quant_of(path) == q or os.path.basename(path).upper().endswith(
        f"-{q}.GGUF")


def _repo_sizes(repo: str, revision: str, cache: dict) -> dict:
    key = (repo, revision)
    if key not in cache:
        try:
            cache[key] = {p: s for p, typ, s in
                          remote.hf_list_dir(repo, "", revision)
                          if typ == "file"}
        except remote.RemoteError:
            cache[key] = {}
    return cache[key]


def _remaining(dest: str, size: int | None) -> int | None:
    if size is None:
        return None
    if os.path.exists(dest):
        return 0
    if os.path.exists(dest + ".part"):
        return max(0, size - os.path.getsize(dest + ".part"))
    return size


def _dest(repo: str, base_dir: str, nest: bool) -> str:
    return os.path.join(base_dir, repo.replace("/", "__")) if nest \
        else base_dir


def _chat_encoders(cfg, family: str) -> dict:
    """Resolved path -> id of the configured chat models that are the
    family's encoder and can be shared."""
    if cfg is None:
        return {}
    from gmlx.config import ConfigError, resolve_path

    out = {}
    for mid, mc in cfg.models.items():
        if mc.adapter or str(mc.path).startswith("hf:"):
            continue
        try:
            rp = resolve_path(mc.path, cfg.model_dirs)
        except ConfigError:
            continue
        if rp and os.path.isfile(rp) and catalog.encoder_matches(rp, family):
            out[os.path.abspath(rp)] = mid
    return out


def _download(plan: Plan, part: Part, dest: str) -> Part:
    """``part`` downloads to ``dest``. A file already there that the
    identity scan did not take is in the way: the pull refuses it rather
    than take it unchecked. A ``.part`` beside it resumes."""
    if os.path.exists(dest):
        plan.blocked.append(part)
    plan.parts.append(part)
    return part


def plan_pull(r: catalog.Recipe, *, dit_quant: str, encoder_quant: str,
              encoder_quant_given: bool, only: tuple, roots: list,
              base_dir: str, nest: bool, cfg=None,
              ram: int | None = None) -> Plan:
    """What the pull uses from disk and what it downloads."""
    plan = Plan(r, dit_quant, encoder_quant, catalog.memory_class(ram))
    found = catalog.scan(roots, families=(r.family,))
    sizes: dict = {}

    def of(role):
        return [f for f in found if f.role == role and f.family == r.family]

    # The DiT: this variant at this quant. A renamed file whose name has no
    # quant is that quant when its size is the published file's.
    name = r.dit_name(dit_quant)
    want = _repo_sizes(r.dit_repo, "main", sizes).get(name)
    dits = [f for f in of("dit") if f.variant == r.variant and (
        _same_quant(f.path, dit_quant) or (
            not _quant_of(f.path) and want is not None
            and os.path.getsize(f.path) == want))]
    if dits:
        p = dits[0].path
        plan.parts.append(Part("dit", os.path.basename(p), p,
                               size=os.path.getsize(p), need=0))
    else:
        dest = os.path.join(_dest(r.dit_repo, base_dir, nest), name)
        _download(plan, Part("dit", name, dest, repo=r.dit_repo,
                             files=(name,),
                             root=_dest(r.dit_repo, base_dir, nest),
                             size=want,
                             need=_remaining(dest, want)), dest)

    # The encoder: the quant asked for or the class default first, then a
    # configured chat model, then the file nearest the DiT.
    shared = _chat_encoders(cfg, r.family)
    encs = catalog.nearest(plan.parts[0].path, of("encoder"))
    if encoder_quant_given:
        encs = [f for f in encs if _same_quant(f.path, encoder_quant)]
    encs = sorted(encs, key=lambda f: (
        not _same_quant(f.path, encoder_quant),
        os.path.abspath(f.path) not in shared))
    pick = encs[0].path if encs else None
    name = r.encoder_name(encoder_quant)
    if pick is not None and not _same_quant(pick, encoder_quant):
        # Another quant on disk serves; the plan names the one it takes.
        got = _quant_of(pick)
        plan.encoder_quant = next((q for q in r.encoder_quants
                                   if q.upper() == got), got or "?")
    if pick is not None:
        chat_id = shared.get(os.path.abspath(pick))
        if chat_id and r.mmproj_file and not cfg.models[chat_id].mmproj:
            # A chat model without its vision tower cannot be shared: the
            # entry names the file and its own tower, so it can edit.
            chat_id = None
        plan.parts.append(Part(
            "encoder", os.path.basename(pick), pick,
            size=os.path.getsize(pick), need=0, chat_id=chat_id))
    else:
        dest = os.path.join(_dest(r.encoder_repo, base_dir, nest), name)
        size = _repo_sizes(r.encoder_repo, "main", sizes).get(name)
        _download(plan, Part("encoder", name, dest, repo=r.encoder_repo,
                             files=(name,),
                             root=_dest(r.encoder_repo, base_dir, nest),
                             size=size,
                             need=_remaining(dest, size)), dest)

    # The vision tower, unless the shared chat model brings its own.
    enc = plan.part("encoder")
    if r.mmproj_file and not (enc and enc.chat_id
                              and cfg.models[enc.chat_id].mmproj):
        towers = catalog.nearest(enc.path, of("mmproj")) if enc else []
        if towers:
            p = towers[0].path
            plan.parts.append(Part("mmproj", os.path.basename(p), p,
                                   size=os.path.getsize(p), need=0))
        else:
            dest = os.path.join(_dest(r.encoder_repo, base_dir, nest),
                                r.mmproj_file)
            size = _repo_sizes(r.encoder_repo, "main", sizes).get(
                r.mmproj_file)
            _download(plan, Part("mmproj", r.mmproj_file, dest,
                                 repo=r.encoder_repo,
                                 files=(r.mmproj_file,),
                                 root=_dest(r.encoder_repo, base_dir, nest),
                                 size=size,
                                 need=_remaining(dest, size)), dest)

    # The VAE: a packed GGUF of the same source, else the source to pack.
    vaes = [f for f in of("vae") if f.sha256 == r.vae_sha256]
    if vaes:
        p = catalog.nearest(plan.parts[0].path, vaes)[0].path
        plan.parts.append(Part("vae", os.path.basename(p), p,
                               size=os.path.getsize(p), need=0))
    else:
        d = _dest(r.vae_repo, base_dir, nest)
        dest = os.path.join(d, r.vae_pack)
        listing = _repo_sizes(r.vae_repo, r.vae_revision, sizes)
        size = listing.get(r.vae_file)
        cfg_size = (listing.get(r.vae_config) or 0) if r.vae_config else 0
        need = _remaining(os.path.join(d, r.vae_file), size)
        _download(plan, Part(
            "vae", r.vae_file, dest, repo=r.vae_repo,
            files=(r.vae_file, r.vae_config) if r.vae_config
            else (r.vae_file,), root=d, revision=r.vae_revision,
            size=None if size is None else size + cfg_size,
            need=None if need is None else need + cfg_size), dest)

    for p in plan.parts:
        role = "encoder" if p.role == "mmproj" else p.role
        p.wanted = role in only
    return plan


def _gb(n: int | None, digits: int = 1) -> str:
    from gmlx.serve.lifecycle import human_gb

    return "?" if n is None else human_gb(n, digits)


def _combined(plan: Plan) -> int | None:
    sizes = [p.size for p in plan.parts]
    if any(s is None for s in sizes):
        return None
    return sum(sizes)  # type: ignore[arg-type]


def _print_plan(plan: Plan, *, ram: int | None, free: int | None,
                dest: str) -> None:
    r = plan.recipe
    head = (f"pull {r.name} ({r.title}): DiT {plan.dit_quant}, encoder "
            f"{plan.encoder_quant}")
    if ram and plan.default:
        head += f", the default for a {plan.memory_class} GB Mac" \
            if plan.memory_class < 64 else ", the default for 64 GB and up"
    print(head)
    wid = max(len(p.name) for p in plan.parts)
    for p in plan.parts:
        if not p.wanted:
            state = "skipped (--only)" if p.download else "on disk"
        elif not p.download:
            state = "on disk" + (f", chat model {p.chat_id}" if p.chat_id
                                 else "")
        elif p.role == "vae":
            state = f"download, packed to {r.vae_pack}"
        else:
            state = "download"
        print(f"  {p.role:<8} {p.name:<{wid}}  {_gb(p.size):>8}  {state}")
        if not p.download:
            print(f"  {'':<8} {p.path}")
    need = _need(plan)
    line = f"  download {_gb(need)}"
    if free is not None:
        line += f", {_gb(free)} free at {dest}"
    print(line)
    total = _combined(plan)
    if total and ram:
        from gmlx.load.memfit import FIT_FITS, FIT_TIGHT, classify_fit

        cast = _cast(plan)
        fit = classify_fit(total + cast, ram)
        where = {FIT_FITS: "fit", FIT_TIGHT: "are tight on"}.get(
            fit, "are more than") if fit else None
        if where:
            load = (f", {_gb(total + cast)} while the DiT loads" if cast
                    else "")
            print(f"  the three parts: {_gb(total)}{load}, which {where} "
                  f"this Mac's {_gb(ram, 0)} RAM")
    if plan.size:
        print(f"  the entry gets defaults.size {plan.size}, the largest image "
              f"this DiT makes on a {plan.memory_class} GB Mac")
    note = f", {r.license_note}" if r.license_note else ""
    print(f"  license: {r.title} {r.license}{note} ({r.license_url}); "
          f"encoder {r.encoder_license}")


def _cast(plan: Plan) -> int:
    """What the load of the DiT holds beside it while it casts its 16-bit
    tensors, as the load gate counts it: from the header of a DiT on
    disk, else from its quant."""
    from .image_preflight import dit_cast_bytes, quant_cast_bytes

    dit = next(p for p in plan.parts if p.role == "dit")
    if not dit.download and dit.path and os.path.isfile(dit.path):
        return dit_cast_bytes(dit.path, plan.recipe.family)
    return quant_cast_bytes(plan.dit_quant, dit.size or 0,
                            plan.recipe.family)


def _need(plan: Plan) -> int | None:
    total = 0
    for p in plan.parts:
        if not (p.download and p.wanted):
            continue
        if p.need is None:
            return None
        total += p.need
    return total


def _fetch(plan: Plan, part: Part) -> list:
    from gmlx.commands import manage

    d = part.root or os.path.dirname(part.path)
    out = []
    for name in part.files:
        print(f"downloading {part.repo}/{name}", file=sys.stderr)
        out.append(manage._hf_download(part.repo, name, part.revision, d))
    return out


def _pack(plan: Plan, part: Part, got: list, *, keep_source: bool) -> str:
    """Check the VAE source against the recipe, pack it, check the
    GGUF, and delete the source unless ``keep_source``. A VAE source
    without a config.json packs with an empty one."""
    from gmlx.image.vae import pack_vae

    r = plan.recipe
    src, conf = got[0], (got[1] if len(got) > 1 else None)
    sha = catalog.sha256_file(src)
    if sha != r.vae_sha256:
        raise remote.RemoteError(
            f"{src} has sha256 {sha}, and the {r.title} VAE at "
            f"{r.vae_revision[:12]} has {r.vae_sha256}; delete it and pull "
            "again")
    config_json = "{}"
    if conf is not None:
        with open(conf, encoding="utf-8") as f:
            config_json = f.read()
    pack_vae(src, part.path, source_repo=r.vae_repo,
             source_revision=r.vae_revision, config_json=config_json,
             family=r.family, overwrite=True)
    if catalog.vae_identity(part.path) != (r.family, r.vae_sha256):
        raise remote.RemoteError(f"the packed VAE {part.path} does not read "
                                 "back as its source")
    if not keep_source:
        for p in (src, conf):
            if p is not None:
                os.remove(p)
        sub = os.path.dirname(src)
        if sub != os.path.dirname(part.path) and not os.listdir(sub):
            os.rmdir(sub)
    part.size = os.path.getsize(part.path)
    return part.path


def _entry_id(r: catalog.Recipe, plan: Plan, cfg) -> str:
    taken = set(cfg.models) | set(cfg.aliases) | set(cfg.assistants)
    if r.name not in taken and r.name not in cfg.image_models:
        return r.name
    from gmlx.load.discovery import _slug

    cand = f"{r.name}-{_slug(plan.dit_quant)}"
    n = 2
    base = cand
    while cand in taken or cand in cfg.image_models:
        cand = f"{base}-{n}"
        n += 1
    return cand


def _entry(plan: Plan, cfg) -> dict:
    from gmlx.load.discovery import _rel

    dirs = cfg.model_dirs
    enc = plan.part("encoder")
    mm = plan.part("mmproj")
    out = {"path": _rel(plan.part("dit").path, dirs),
           "encoder": enc.chat_id or _rel(enc.path, dirs)}
    if mm is not None and not enc.chat_id:
        out["encoder_mmproj"] = _rel(mm.path, dirs)
    out["vae"] = _rel(plan.part("vae").path, dirs)
    if plan.size:
        out["defaults"] = {"size": plan.size}
    return out


def _registered(plan: Plan, cfg) -> str | None:
    """The image entry that already serves this DiT, or None."""
    from gmlx.config import ConfigError, resolve_path

    dit = os.path.abspath(plan.part("dit").path)
    for iid, m in cfg.image_models.items():
        try:
            rp = resolve_path(m.path, cfg.model_dirs)
        except ConfigError:
            continue
        if rp and os.path.abspath(rp) == dit:
            return iid
    return None


def register(plan: Plan, cfg_path: str) -> str | None:
    """Add the pulled model under ``image_models:`` and make it the image
    default when none is set. Returns the id, or None when the entry was
    there already or the DiT lies outside every model_dirs root."""
    from gmlx.config import edit_config_yaml, load_config

    cfg = load_config(cfg_path)
    roots = [os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
             for d in cfg.model_dirs]
    for p in plan.parts:
        ap = os.path.abspath(p.path)
        if p.chat_id is None and not any(
                ap == r or ap.startswith(r + os.sep) for r in roots):
            print(f"note: {p.path} is outside the config's model_dirs, so "
                  f"{plan.recipe.name} is not registered", file=sys.stderr)
            return None
    old = _registered(plan, cfg)
    if old is not None:
        print(f"{plan.recipe.name} is already configured as {old}")
        return None
    iid = _entry_id(plan.recipe, plan, cfg)
    entry = _entry(plan, cfg)
    set_default = not cfg.defaults.image_model

    def mutate(doc):
        from ruamel.yaml.comments import CommentedMap

        block = doc.get("image_models")
        if not isinstance(block, dict):
            block = CommentedMap()
            doc["image_models"] = block
        block[iid] = CommentedMap(entry)
        if set_default:
            srv = doc.get("server")
            if not isinstance(srv, dict):
                srv = CommentedMap()
                doc["server"] = srv
            d = srv.get("defaults")
            if not isinstance(d, dict):
                d = CommentedMap()
                srv["defaults"] = d
            d["image_model"] = iid

    edit_config_yaml(cfg_path, mutate)
    print(f"registered {iid} in {cfg_path}"
          + (" (the image default)" if set_default else ""))
    return iid


def _config_path(a) -> str | None:
    import gmlx.config as cfgmod

    if a.config:
        p = os.path.expanduser(a.config)
        return p if os.path.exists(p) else None
    return next((str(c) for c in cfgmod.default_config_paths()
                 if os.path.exists(c)), None)


def pull_recipe(a, r: catalog.Recipe) -> int:
    """``gmlx pull`` of the recipe ``r`` with the parsed flags ``a``."""
    import gmlx.config as cfgmod
    from gmlx.commands import manage
    from gmlx.load.memfit import total_ram_bytes

    try:
        only = _parse_only(a.only)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    ram = total_ram_bytes()
    dit_q, enc_q = catalog.default_quants(r, ram)
    if a.quant:
        dit_q = a.quant
    if a.encoder_quant:
        enc_q = a.encoder_quant
    for what, q, have in (("--quant", dit_q, r.dit_quants),
                          ("--encoder-quant", enc_q, r.encoder_quants)):
        match = [h for h in have if h.upper() == q.upper()]
        if not match:
            print(f"error: {what} {q} is not a quant of {r.name}; choose "
                  f"one of {', '.join(have)}", file=sys.stderr)
            return 2
    dit_q = next(h for h in r.dit_quants if h.upper() == dit_q.upper())
    enc_q = next(h for h in r.encoder_quants if h.upper() == enc_q.upper())

    cfg_path = _config_path(a)
    cfg = None
    if cfg_path:
        try:
            cfg = cfgmod.load_config(cfg_path)
        except cfgmod.ConfigError as e:
            print(f"error: could not load config {cfg_path}: {e}",
                  file=sys.stderr)
            return 2
    try:
        base_dir = (os.path.expanduser(os.path.expandvars(a.to)) if a.to
                    else manage._model_dir_dest(a.config))
    except remote.RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    nest = a.to is None
    roots = list(cfg.model_dirs) if cfg is not None else []
    if base_dir not in roots:
        roots.append(base_dir)

    plan = plan_pull(r, dit_quant=dit_q, encoder_quant=enc_q,
                     encoder_quant_given=bool(a.encoder_quant), only=only,
                     roots=roots, base_dir=base_dir, nest=nest, cfg=cfg,
                     ram=ram)
    plan.default = (plan.dit_quant, plan.encoder_quant) == \
        catalog.default_quants(r, ram)
    plan.size = catalog.default_size(r, ram, plan.dit_quant)
    need = _need(plan)
    free = manage._disk_free(base_dir)
    if a.json:
        print(json.dumps({
            "recipe": r.name, "dit_quant": plan.dit_quant,
            "encoder_quant": plan.encoder_quant,
            "memory_class": plan.memory_class, "default_size": plan.size,
            "download_bytes": need,
            "free_bytes": free, "license": r.license,
            "license_note": r.license_note or None,
            "encoder_license": r.encoder_license,
            "parts": [{"role": p.role, "name": p.name, "path": p.path,
                       "size_bytes": p.size, "download": p.download
                       and p.wanted, "on_disk": not p.download,
                       "chat_model": p.chat_id} for p in plan.parts]},
            indent=2))
    else:
        _print_plan(plan, ram=ram, free=free, dest=base_dir)
    blocked = [p for p in plan.blocked if p.wanted]
    if blocked:
        for p in blocked:
            print(f"error: {p.path} is not the {r.title} {_ROLE[p.role]}, "
                  "and a download would land on it; move or delete it, and "
                  "pull again", file=sys.stderr)
        return 1
    # The packed VAE is written while its source is still on disk.
    vae = plan.part("vae")
    if need is not None and vae is not None and vae.download and vae.wanted:
        need += r.vae_bytes
    if need and free is not None and free < need and not a.force:
        print(f"error: not enough disk space: need {_gb(need)}, "
              f"{_gb(free)} free at {base_dir} (free space, pass --to DIR "
              "on another volume, or --force)", file=sys.stderr)
        return 1

    for part in plan.parts:
        if not (part.download and part.wanted):
            continue
        try:
            got = _fetch(plan, part)
            if part.role == "vae":
                _pack(plan, part, got, keep_source=a.keep_source)
            part.repo = None
            part.need = 0
            if part.size is None or part.role != "vae":
                part.size = os.path.getsize(part.path)
        except (remote.RemoteError, OSError, ValueError) as e:
            print(f"error: {part.role} failed: {e}", file=sys.stderr)
            return 1
        print(f"  {part.role}: {part.path}")

    missing = [p.role for p in plan.parts if p.download]
    if missing:
        print(f"not registered: {', '.join(missing)} not on disk yet; "
              f"`gmlx pull {r.name}` fetches the rest", file=sys.stderr)
        return 0
    if a.no_register or cfg_path is None:
        return 0
    try:
        iid = register(plan, cfg_path)
    except (cfgmod.ConfigError, cfgmod.ConfigWriteError) as e:
        print(f"warning: could not register {r.name} in the server config: "
              f"{e}", file=sys.stderr)
        return 0
    if iid is not None:
        from gmlx.serve.server import _reload_running

        _reload_running(cfg_path, skip=False)
        print(f"try it: gmlx image generate {iid} --prompt \"a red bicycle\" "
              "--output out.png")
    return 0

