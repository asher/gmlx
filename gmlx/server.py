"""``gmlx`` batched OpenAI/Anthropic server - the platform's serving layer.

Composes the gmlx loader bridge (:func:`server_bridge_vlm.install_gguf_server_bridge`),
the multi-model residency pool (:func:`residency.install_gguf_residency_pool`),
the config-driven HTTP surface (:func:`server_patches.install_server_patches`),
and mlx-vlm's FastAPI app + ``BatchGenerator`` continuous-batching engine into one
GGUF-only HTTP server - **text**, **VLM** (LLM GGUF + float ``mmproj``), and
**speculative/MTP** models.

Start modes (resolution order):

* ``init --models-dir DIR`` - discover the GGUFs under ``--models-dir`` and scaffold a
  starter YAML config to ``--out`` (default ``~/.config/gmlx/gmlx.yaml``; the
  only path that writes a file; refuses overwrite without ``--force``).
* ``sync-models`` - reconcile an existing config's ``models:`` with what's on disk
  (keep, drop gone, add new); preserves comments. With ``--from-hf-cache`` (or a
  config already carrying ``hf_cache: true``) it also reconciles cache-resident GGUFs.
  Default config unless ``--config``.
* ``launch <harness>`` - point a coding harness (opencode, ...) at a **running** server
  and exec it (see :mod:`launch`). No auto-install; the server must already be up.
* ``--config FILE`` - serve a YAML config (named models + profiles).
* ``--models-dir DIR`` - serve a discovery scan of a directory (in-memory config).
* a positional ``model.gguf`` - serve a single model (wrapped as a one-model config).
* bare - load the first existing default config, else discovery-scan the current
  directory.

Every mode converges on one :class:`config.ServerCfg`: register it, install the
bridge + residency pool + HTTP patches (before the lifespan preload), then
``uvicorn`` serves the ``mlx_vlm.server:app`` FastAPI app in-process (single
worker, no reload) so the patches hold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import discovery
from .config import (
    DEFAULT_CONFIG_WRITE,
    LOOPBACK_HOSTS,
    ConfigError,
    DiscoverSpec,
    MissingModelFile,
    ModelCfg,
    ServerCfg,
    ServerDefaults,
    default_config_paths,
    default_config_write_path,
    edit_config_yaml,
    load_config,
    resolve_path,
)
from .envflags import env_bool

_DEFAULT_DISCOVER_DIR = "."          # zero-config bare start scans the cwd


def _has_uvloop() -> bool:
    try:
        import uvloop  # noqa: F401
        return True
    except ImportError:
        return False
_LOOPBACK = LOOPBACK_HOSTS

# Sentinel for `--with-<svc>` given with no value (use the service's default alias).
_SVC_DEFAULT = object()


def _import_serving():
    """Import the mlx-vlm-backed serving stack, lazily so ``init`` / ``launch`` /
    ``--help`` never need it, and a missing mlx-vlm names the fix instead of a
    bare ModuleNotFoundError."""
    try:
        from . import residency, server_patches, server_bridge_vlm  # noqa: F401
    except ImportError as exc:
        root = (exc.name or "").split(".")[0]
        if root in ("mlx_vlm", "fastapi", "uvicorn", "starlette"):
            raise ImportError(
                "gmlx serve needs its serving dependencies (mlx-vlm, "
                "FastAPI, uvicorn), which install with gmlx and are required "
                "even for text-only serving. A missing one usually means a "
                "broken install - reinstall gmlx.") from exc
        raise


def main(argv: list | None = None, prog: str | None = None) -> int:
    """``prog`` overrides the help/usage program name (the ``gmlx`` umbrella
    passes ``gmlx <verb>``; a bare ``python -m gmlx.server`` keeps the
    ``gmlx serve`` defaults)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        return _cmd_init(argv[1:], prog=prog or "gmlx init")
    if argv and argv[0] == "sync-models":
        return _cmd_sync(argv[1:], prog=prog or "gmlx sync-models")
    if argv and argv[0] == "launch":
        from .launch import cmd_launch
        return cmd_launch(argv[1:], prog=prog or "gmlx launch")
    # Lifecycle verbs for a backgrounded server - explicit branches before the serve
    # fallthrough so they aren't parsed as the positional GGUF.
    if argv and argv[0] == "stop":
        return _cmd_stop(argv[1:], prog=prog or "gmlx stop")
    if argv and argv[0] == "restart":
        return _cmd_restart(argv[1:], prog=prog or "gmlx restart")
    if argv and argv[0] == "status":
        return _cmd_status(argv[1:], prog=prog or "gmlx status")
    if argv and argv[0] == "logs":
        return _cmd_logs(argv[1:], prog=prog or "gmlx logs")
    if argv and argv[0] == "service":
        return _cmd_service(argv[1:], prog=prog or "gmlx service")
    return _cmd_serve(argv, prog=prog or "gmlx serve")


def _reload_running(config_path, *, skip: bool) -> None:
    """After init/sync rewrites ``config_path``, SIGHUP any server running from it so it
    picks up the change without a restart (it keeps resident models). Best-effort - a
    reload hiccup must never fail the write that already succeeded."""
    if skip:
        return
    try:
        from . import lifecycle
        target = os.path.abspath(os.path.expanduser(str(config_path)))
        signalled = lifecycle.reload_config(target)
    except Exception:
        return
    for host, port, pid in signalled:
        print(f"reloaded the running server at {host}:{port} (pid {pid}) "
              f"to pick up the config change")


# init - scaffold a starter config (guided wizard, or flag-driven)
def _cmd_init(argv: list, prog: str = "gmlx init") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Scaffold a starter YAML config from a directory of GGUFs "
                    "and/or the local Hugging Face cache. Run with no flags on a "
                    "terminal for a guided wizard.",
    )
    ap.add_argument("--models-dir", action="append", default=None, metavar="DIR",
                    help="Directory of GGUFs to scan (repeatable). Required unless "
                         "--from-hf-cache is given.")
    ap.add_argument("--from-hf-cache", "--hf-cache", action="store_true",
                    help="Also scan the local Hugging Face cache for GGUFs and add "
                         "them as portable hf:<org>/<repo>/<file> entries (sets "
                         "server.hf_cache; resolved from the cache, never downloaded).")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help=f"Where to write the config (default: "
                         f"{DEFAULT_CONFIG_WRITE}).")
    ap.add_argument("--disk-cache", nargs="?", const=50.0, type=float,
                    default=None, metavar="GB",
                    help="Persist the prompt cache to disk in the generated config "
                         "(the in-memory cache is on in every generated config): "
                         "cached prefixes land under ~/.cache/gmlx/apc and survive "
                         "an idle-unload / restart. A bare --disk-cache caps it at "
                         "50 GB per model, or pass a size (e.g. --disk-cache 100).")
    ap.add_argument("-r", "--recursive", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Recurse into subdirectories when scanning --models-dir "
                         "(default: shallow).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing config.")
    ap.add_argument("--no-reload", action="store_true",
                    help="Don't SIGHUP a server already running this config to pick "
                         "up the change.")
    # Interactivity. Bare `init` on a terminal runs the wizard; flags or a
    # non-terminal use the flag-driven path. -i forces the wizard either way.
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="Run the guided wizard (the default when init is given no "
                         "scaffolding flags on a terminal).")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Never run the wizard; scaffold from the flags as given.")
    # Optional services. A bare --with-stt uses the default alias; a value overrides
    # the model (alias / HF repo id / local path; embeddings also accepts a GGUF).
    ap.add_argument("--with-stt", nargs="?", const=_SVC_DEFAULT, default=None,
                    metavar="MODEL", help="Configure speech-to-text (server.stt).")
    ap.add_argument("--with-tts", nargs="?", const=_SVC_DEFAULT, default=None,
                    metavar="MODEL", help="Configure text-to-speech (server.tts).")
    ap.add_argument("--with-embeddings", nargs="?", const=_SVC_DEFAULT, default=None,
                    metavar="MODEL",
                    help="Configure text embeddings (server.embeddings).")
    ap.add_argument("--with-rerank", nargs="?", const=_SVC_DEFAULT, default=None,
                    metavar="MODEL",
                    help="Configure reranking (server.rerank); a Qwen3-Reranker "
                         "GGUF, no extra.")
    ap.add_argument("--install", action="store_true",
                    help="pip-install the missing extras for the chosen --with-* "
                         "services after writing the config.")
    ap.add_argument("--no-install", action="store_true",
                    help="Don't offer to install extras in the wizard.")
    ap.add_argument("--default-model", default=None, metavar="ID",
                    help="Set server.defaults.model (used when a request omits it).")
    ap.add_argument("--port", type=int, default=None,
                    help="Set server.port (default 8080).")
    ap.add_argument("--idle-ttl", default=None, metavar="SECONDS|none",
                    help="Set server.defaults.ttl_s - idle auto-unload; "
                         "`none` keeps models resident (manual / LRU eviction).")
    ap.add_argument("--request-timeout", default=None, metavar="DURATION|none",
                    help="Set server.token_queue_timeout_s - give up if no new "
                         "token for this long (e.g. 10m, 1h); `none` waits forever.")
    a = ap.parse_args(argv)

    if _want_interactive(a):
        return _init_interactive(a)
    # No scaffolding intent and not interactive -> the help affordance (preserves
    # bare `init` on a non-terminal / under --no-interactive).
    if not _has_scaffold_intent(a):
        ap.print_help()
        return 0
    return _init_scaffold(a, ap)


def _has_scaffold_intent(a) -> bool:
    """True if any flag asked init to scaffold something (vs. a bare invocation)."""
    return bool(
        a.models_dir or a.from_hf_cache or a.out or a.disk_cache is not None
        or a.recursive
        or a.force or a.with_stt is not None or a.with_tts is not None
        or a.with_embeddings is not None or a.with_rerank is not None
        or a.default_model or a.port is not None
        or a.idle_ttl is not None or a.request_timeout is not None
        or a.install or a.no_install)


def _want_interactive(a) -> bool:
    """Decide between the wizard and the flag-driven path."""
    if a.no_interactive:
        return False
    if a.interactive:
        return True
    if _has_scaffold_intent(a):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _init_interactive(a) -> int:
    from . import wizard
    default_out = a.out or DEFAULT_CONFIG_WRITE
    try:
        outcome = wizard.run_wizard(
            default_out=default_out, seed_dirs=a.models_dir,
            allow_install=not a.no_install, port=a.port)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 1
    if outcome is None:
        return 1
    return _finish_write(outcome.out, outcome.text, outcome.models,
                         no_reload=a.no_reload)


def _resolve_services(a):
    """Map the --with-* flags to scaffold values (default alias when bare).
    Returns ``(stt, tts, embeddings, rerank)``."""
    from . import embeddings as _emb, rerank as _rr, stt as _stt, tts as _tts

    def pick(v, default_alias):
        if v is None:
            return None
        return default_alias if v is _SVC_DEFAULT else v

    return (pick(a.with_stt, _stt.DEFAULT_STT_ALIAS),
            pick(a.with_tts, _tts.DEFAULT_TTS_ALIAS),
            pick(a.with_embeddings, _emb.DEFAULT_EMBEDDINGS_ALIAS),
            pick(a.with_rerank, _rr.DEFAULT_RERANK_ALIAS))


def _resolve_duration(flag: str, value, ap):
    if value is None:
        return None
    from .wizard import parse_duration
    try:
        return parse_duration(value)
    except ValueError:
        ap.error(f"{flag}: expected a duration like 900, 15m, 1h, or 'none' "
                 f"(got {value!r})")


def _install_for_services(stt_v, tts_v, emb_v) -> None:
    from . import embeddings as _emb, extras
    wanted = []
    if stt_v:
        wanted.append("stt")
    if tts_v:
        wanted.append("tts")
    if emb_v and not _emb._is_gguf_ref(emb_v):     # GGUF embedder needs no extra
        wanted.append("embeddings")
    for extra in wanted:
        if not extras.extra_installed(extra):
            extras.install_extra(extra)


def _init_scaffold(a, ap) -> int:
    dirs = a.models_dir or []
    if not dirs and not a.from_hf_cache:
        ap.error("need --models-dir DIR (repeatable) or --from-hf-cache "
                 "(or run `gmlx init` with no flags for the guided wizard)")
    out = Path(os.path.expanduser(a.out)) if a.out else default_config_write_path()
    if out.exists() and not a.force:
        print(f"refusing to overwrite {out} (use --force)", file=sys.stderr)
        return 1

    models = []
    scan_stats: dict = {}
    if dirs:
        specs = [DiscoverSpec(dir=d, recursive=a.recursive) for d in dirs]
        models += discovery.scan_dirs(specs, dirs, progress=True,
                                      stats=scan_stats)
    if a.from_hf_cache:
        models += discovery.scan_hf_cache(
            known_ids={m.id for m in models}, progress=True)

    # The ids are GENERATED during discovery, so a hand-typed --default-model
    # is easy to get wrong - and every consumer hard-rejects a config whose
    # default names no model. Fail here, where the real ids can be shown.
    if a.default_model and all(m.id != a.default_model for m in models):
        print(f"error: --default-model {a.default_model!r} matches none of the "
              f"discovered model ids: {sorted(m.id for m in models)}",
              file=sys.stderr)
        return 2

    if a.disk_cache is not None and a.disk_cache <= 0:
        ap.error(f"--disk-cache: expected a positive size in GB (got {a.disk_cache})")
    if a.port is not None and not (0 < a.port < 65536):
        ap.error(f"--port: expected 1-65535 (got {a.port})")
    stt_v, tts_v, emb_v, rerank_v = _resolve_services(a)
    ttl_s = _resolve_duration("--idle-ttl", a.idle_ttl, ap)
    timeout_s = _resolve_duration("--request-timeout", a.request_timeout, ap)
    text = discovery.scaffold_yaml(
        models, model_dirs=dirs, hf_cache=a.from_hf_cache,
        disk_cache=a.disk_cache is not None, disk_cache_gb=a.disk_cache,
        stt=stt_v, tts=tts_v, embeddings=emb_v, rerank=rerank_v,
        default_model=a.default_model or None, ttl_s=ttl_s,
        token_queue_timeout_s=timeout_s, port=a.port)

    rc = _finish_write(out, text, models, no_reload=a.no_reload,
                       skipped=scan_stats.get("skipped", 0))
    if a.install:
        _install_for_services(stt_v, tts_v, emb_v)
    return rc


def _finish_write(out: Path, text: str, models, *, no_reload: bool,
                  skipped: int = 0) -> int:
    """Commit the rendered config: write it, print the summary + next step, and
    SIGHUP a server already running it. Shared by the wizard and the flag path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # tmp + rename: with --force this replaces an existing config, and a bare
    # write_text would truncate it before the new text lands.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(models)} model(s) discovered)")
    _print_models({m.id: m for m in models})
    if not models:
        if skipped:
            print(f"\nfound {skipped} .gguf file(s) but none were readable as "
                  f"GGUF (see the [discover] skip reasons above - truncated "
                  f"downloads?). Fix or re-pull them, then run "
                  f"`gmlx sync-models`.")
        else:
            print("\nno GGUFs found yet - drop some into your model dir (or "
                  "`gmlx pull` into it), then run `gmlx sync-models` to "
                  "add them.")
    else:
        print("\nsampling: every model starts from its family's model-card "
              "defaults; request\n`<id>@coding` (or @instruct / @creative / "
              "@reasoning-*) to switch intent.\n`gmlx profiles` prints the "
              "full table.")
    _reload_running(out, skip=no_reload)
    # When the config lands where a bare `serve` would find it first, the `--config`
    # flag is redundant - show the shorter command.
    first = next((p for p in default_config_paths() if p.exists()), None)
    bare = first is not None and os.path.realpath(first) == os.path.realpath(out)
    cfg_arg = "" if bare else f" --config {out}"
    print(f"\nnext: gmlx serve{cfg_arg}")
    # On macOS, point at the launchd agent for a server that starts at every login
    # (service is macOS-only; the hint would be a dead end elsewhere).
    if sys.platform == "darwin":
        print(f"  to start it now and at every login: gmlx service install{cfg_arg}")
    return 0


# sync - reconcile an existing config's models with what's on disk
def _cmd_sync(argv: list, prog: str = "gmlx sync-models") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Reconcile an existing config's models with what's on disk: "
                    "keep configured models that still exist, drop ones whose file "
                    "is gone, and add newly-discovered GGUFs. Comments and "
                    "hand-edits are preserved.",
    )
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Config to sync (default: the first existing default "
                         "location).")
    ap.add_argument("--models-dir", action="append", default=None, metavar="DIR",
                    help="Dirs to scan, overriding the config's server.model_dirs "
                         "(repeatable).")
    ap.add_argument("--from-hf-cache", "--hf-cache", action="store_true",
                    help="Also reconcile cache-resident GGUFs (add new hf: entries, "
                         "drop ones no longer cached). Implied when the config "
                         "already carries server.hf_cache: true.")
    ap.add_argument("-r", "--recursive", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Recurse into subdirectories when scanning (default: deep, "
                         "since `pull` nests under <dir>/<org>__<repo>/). "
                         "--no-recursive for a shallow scan.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing the config.")
    ap.add_argument("--no-reload", action="store_true",
                    help="Don't SIGHUP a server already running this config to pick "
                         "up the change.")
    a = ap.parse_args(argv)

    if a.config:
        path = Path(os.path.expanduser(a.config))
        if not path.exists():
            print(f"error: --config not found: {path}", file=sys.stderr)
            return 2
    else:
        path = next((p for p in default_config_paths() if p.exists()), None)
        if path is None:
            print("error: no config found in the default locations; run "
                  "`gmlx init --models-dir DIR` first (or pass --config).",
                  file=sys.stderr)
            return 2

    try:
        cfg = load_config(path)
    except ConfigError as e:
        print(f"error: could not load {path}: {e}", file=sys.stderr)
        return 2

    dirs = a.models_dir or cfg.model_dirs
    scan_cache = a.from_hf_cache or cfg.hf_cache
    if not dirs and not scan_cache:
        print(f"error: {path} defines no server.model_dirs and neither "
              "--models-dir nor --from-hf-cache was given; nothing to scan.",
              file=sys.stderr)
        return 2

    recursive = a.recursive

    # Keep models whose file still resolves (local path or cache-resident hf: ref);
    # drop the ones gone from disk / no longer cached. Removal is a destructive
    # config write, so an entry is only dropped when its verification substrate
    # is intact: an unreadable hf cache or a missing model_dirs root (unmounted
    # disk, wrong shell env) keeps the entries it covers, with a warning.
    hf_cache_ok = _hf_cache_readable()
    missing_roots = [d for d in cfg.model_dirs
                     if not os.path.isdir(os.path.expanduser(os.path.expandvars(d)))]
    kept, removed, unverified, known_paths = [], [], [], set()
    for mid, mc in cfg.models.items():
        is_hf = str(mc.path).startswith("hf:")
        try:
            rp = resolve_path(mc.path, cfg.model_dirs)
        except MissingModelFile:
            if is_hf and not hf_cache_ok:
                unverified.append((mid, "hf cache unreadable"))
                kept.append(mid)
            elif not is_hf and missing_roots:
                unverified.append((mid, "model_dirs root missing"))
                kept.append(mid)
            else:
                removed.append(mid)
            continue
        except ConfigError:
            # A shape problem, not disk state - never silently delete it.
            unverified.append((mid, "entry does not parse; fix it by hand"))
            kept.append(mid)
            continue
        if rp and not os.path.exists(rp):
            removed.append(mid)                    # dangling absolute path
            continue
        kept.append(mid)
        if rp:
            known_paths.add(rp)
    if missing_roots:
        print(f"warning: model_dirs root(s) not on disk right now: "
              f"{', '.join(missing_roots)} - their entries are kept unverified",
              file=sys.stderr)
    if not hf_cache_ok and any(r == "hf cache unreadable" for _, r in unverified):
        print("warning: the local Hugging Face cache is unreadable - hf: entries "
              "are kept unverified", file=sys.stderr)

    # Discover, skipping anything already configured (by id or by resolved path).
    discovered = []
    if dirs:
        specs = [DiscoverSpec(dir=d, recursive=recursive) for d in dirs]
        discovered += discovery.scan_dirs(
            specs, dirs,
            known_ids=set(cfg.models), known_paths=known_paths, progress=True)
    if scan_cache:
        known_refs = {mc.path for mc in cfg.models.values()
                      if str(mc.path).startswith("hf:")}
        discovered += discovery.scan_hf_cache(
            known_ids=set(cfg.models) | {m.id for m in discovered},
            known_refs=known_refs, progress=True)

    print(f"\nsync-models {path}:")
    print(f"  kept:    {len(kept)} model(s)")
    for mid, reason in unverified:
        print(f"  keep:    {mid}  (unverified - {reason})")
    for mid in removed:
        print(f"  remove:  {mid}  ({cfg.models[mid].path} - gone)")
    for m in discovered:
        print(f"  add:     {m.id}  ({discovery._rel(m.path, dirs)})")
    if not removed and not discovered:
        print("  (already in sync)")
        return 0
    if a.dry_run:
        print("\n(dry run - no changes written)")
        return 0

    new_roots = ([d for d in dirs if d not in cfg.model_dirs]
                 if a.models_dir else [])
    _apply_sync(path, removed, discovered, dirs, new_roots=new_roots)
    print(f"\nupdated {path} (+{len(discovered)} / -{len(removed)})")
    _reload_running(path, skip=a.no_reload)
    return 0


def _hf_cache_readable() -> bool:
    """Whether the local Hugging Face cache can be scanned at all. False means
    hf: entries cannot be VERIFIED - which must never read as 'gone'."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return os.path.isdir(HF_HUB_CACHE)
    except Exception:
        return False


def _apply_sync(path, removed, discovered, dirs, new_roots=()) -> None:
    """Rewrite ``path``'s ``models:`` block in place, preserving comments and
    hand-edits (ruamel round-trip): delete ``removed`` ids, splice in the newly
    ``discovered`` models. Untouched entries keep their exact formatting.
    ``new_roots`` are --models-dir override roots absent from
    ``server.model_dirs``; they're appended so the new entries resolve."""
    from ruamel.yaml.comments import CommentedMap

    def mutate(doc):
        if new_roots and discovered:
            srv = doc.get("server")
            if not isinstance(srv, dict):
                srv = CommentedMap()
                doc["server"] = srv
            roots = srv.get("model_dirs")
            if not isinstance(roots, list):
                roots = []
                srv["model_dirs"] = roots
            for d in new_roots:
                if d not in roots:
                    roots.append(d)
        models = doc.get("models")
        if not isinstance(models, dict):
            models = CommentedMap()
            doc["models"] = models
        for mid in removed:
            if mid in models:
                del models[mid]
        pos = 0
        for mc in discovered:
            entry = discovery.model_to_entry(mc, dirs)
            if mc.id in models:
                models[mc.id] = entry          # update keeps its position
            else:
                # Insert at the top of the block: appending would land the
                # entry after the scaffold's trailing commented hints (talk:,
                # assistant:), which ride the last entry's comment token.
                models.insert(pos, mc.id, entry)
                pos += 1
            note = discovery.family_comment(mc)
            if note:
                models.yaml_add_eol_comment(note, key=mc.id)
        # Cache-resident entries need server.hf_cache to resolve from the cache.
        if any(str(mc.path).startswith("hf:") for mc in discovered):
            srv = doc.get("server")
            if isinstance(srv, dict) and not srv.get("hf_cache"):
                srv["hf_cache"] = True

    edit_config_yaml(path, mutate)


def register_downloads(paths: list, config_path=None) -> None:
    """Fold freshly downloaded GGUF files into the server config - ``gmlx
    pull``'s auto-register, a mini sync-models scoped to exactly these paths.

    Quiet no-op when there is no config, when a file landed outside every
    ``model_dirs`` root (an explicit ``--to`` elsewhere - the server could
    never discover it), or when the file is already configured. Otherwise the
    same machinery as sync-models end to end: id derivation, mmproj pairing,
    speculative detection, comment-preserving splice, and a SIGHUP so a
    running server serves the new entries immediately. Best-effort by
    contract: the download already succeeded, so a registration problem warns
    and returns instead of failing ``pull``."""
    try:
        if config_path:
            path = Path(os.path.expanduser(str(config_path)))
        else:
            path = next((p for p in default_config_paths() if p.exists()), None)
        if path is None or not path.exists():
            return
        cfg = load_config(path)
        roots = [os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
                 for d in cfg.model_dirs]
        wanted = set()
        for p in paths:
            ap = os.path.abspath(str(p))
            if any(ap == r or ap.startswith(r + os.sep) for r in roots):
                wanted.add(ap)
        if not wanted:
            return
        known_paths = set()
        for mc in cfg.models.values():
            try:
                known_paths.add(resolve_path(mc.path, cfg.model_dirs))
            except ConfigError:
                pass
        # Scan the parent dirs (mmproj pairing needs the siblings), then keep
        # only models whose file is one we just downloaded - a neighbouring
        # file the user left unregistered stays unregistered.
        parents = sorted({os.path.dirname(p) for p in wanted})
        specs = [DiscoverSpec(dir=d, recursive=False) for d in parents]
        found = discovery.scan_dirs(specs, cfg.model_dirs,
                                    known_ids=set(cfg.models),
                                    known_paths=known_paths)
        newly = [m for m in found if os.path.abspath(m.path) in wanted]
        if not newly:
            return                     # already configured (a re-pull) - quiet
        _apply_sync(path, [], newly, cfg.model_dirs)
        for m in newly:
            extras = [w for w, on in (("vlm", m.mmproj),
                                      ("mtp", m.speculative)) if on]
            note = f"  ({', '.join(extras)})" if extras else ""
            print(f"registered {m.id} in {path}{note}")
        _reload_running(path, skip=False)
    except Exception as e:             # noqa: BLE001 - never fail a good pull
        print(f"warning: could not register the download(s) in the server "
              f"config: {e}; run `gmlx sync-models`", file=sys.stderr)


# serve
def _add_serve_args(ap: argparse.ArgumentParser) -> None:
    from .cli import mass_share

    ap.add_argument("model", nargs="?", default=None,
                    help="A single GGUF to serve (sharded ok).")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="Serve a YAML config (named models + profiles).")
    ap.add_argument("--print-config", action="store_true",
                    help="Resolve the effective config for the chosen start mode "
                         "(config / discovery / single model), print it as YAML with "
                         "every key and default filled in, and exit without serving. "
                         "Use it to introspect the schema or check a config.")
    ap.add_argument("--models-dir", action="append", default=None, metavar="DIR",
                    help="Serve a discovery scan of a directory (repeatable).")
    ap.add_argument("-r", "--recursive", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Recurse when discovering (--models-dir / bare; default: "
                         "shallow).")
    ap.add_argument("--hf-cache", "--from-hf-cache", action="store_true",
                    help="Resolve named hf repo ids from the LOCAL hf cache only "
                         "(never the network). Off => no HF access at all.")
    ap.add_argument("--mmproj", default=None,
                    help="Float mmproj GGUF - makes a single positional model a VLM.")
    ap.add_argument("--hf-source", default=None,
                    help="Processor/config override for a single VLM model (rarely "
                         "needed; the processor is synthesized from the GGUFs).")
    ap.add_argument("--speculative", action="store_true",
                    help="Serve a single positional model with MTP speculative "
                         "decoding (native-head qwen3.5/3.6; gemma4 also needs "
                         "--draft-gguf).")
    ap.add_argument("--draft-gguf", default=None,
                    help="Companion drafter GGUF for assistant-shape MTP (gemma4); "
                         "implies --speculative.")
    ap.add_argument("--stochastic-mtp", action="store_true",
                    help="Accept MTP drafts by p/q rejection sampling on sampled "
                         "requests (server-wide): same output distribution as "
                         "non-speculative sampling, not token-identical; higher "
                         "acceptance at temp > 0. Same as config "
                         "server.stochastic_mtp; greedy requests unaffected.")
    ap.add_argument("--draft-block-size", type=int, default=None, metavar="N",
                    help="MTP draft tokens per round (analogous to llama-server "
                         "--spec-draft-n-max). Default: the drafter's own block "
                         "size. Also via GMLX_DRAFT_BLOCK_SIZE.")
    ap.add_argument("--chat-template", default=None, metavar="STR|PATH",
                    help="Inline Jinja template, or a path to a .jinja/.txt file, "
                         "replacing a single positional model's GGUF template "
                         "(config mode: set it per profile/model instead).")
    ap.add_argument("--adapter", default=None, metavar="PATH",
                    help="GGUF LoRA adapter applied live over a single positional "
                         "model at load - base stays K-quant, no merge (config mode: "
                         "set `adapter:` per model instead).")
    placement = ap.add_mutually_exclusive_group()
    placement.add_argument("--stream-experts", action="store_true",
                    help="Stream a single positional MoE model's routed-expert "
                         "stacks from disk (unwired, page-cache resident) while "
                         "the every-token layers + KV cache stay on GPU. Slower "
                         "than --stream-cpu at short context; wins at long "
                         "context with a quantized KV cache (config mode: set "
                         "`stream: experts` per model).")
    placement.add_argument("--stream-cpu", action="store_true",
                    help="Run a single positional model entirely on the CPU device "
                         "with every weight streamed from the page cache - serves "
                         "models larger than the wired limit "
                         "(config mode: set `stream: cpu` per model).")
    ap.add_argument("--moe-expert-mass", type=mass_share, default=None,
                    metavar="P",
                    help="Lossy: adaptive experts-per-token for a single positional "
                         "model on the streamed MoE layers (--stream-experts / "
                         "--stream-cpu) - each token keeps only the smallest "
                         "set of its routed experts covering share P (0 < P <= 1) "
                         "of the router's gate mass. Size P with `gmlx run "
                         "--moe-expert-probe` (config mode: set `moe_expert_mass: "
                         "P` per model).")
    ap.add_argument("--moe-experts", type=int, default=None, metavar="K",
                    help="Lossy: cap the router at K experts per token for a "
                         "single positional model on the streamed MoE layers. "
                         "Composes with --moe-expert-mass (config mode: set "
                         "`moe_experts: K` per model).")
    ap.add_argument("--moe-miss-shed", type=mass_share, default=None,
                    metavar="P",
                    help="Lossy: at decode, drop routed experts that would "
                         "demand-miss the expert arena, lowest scores first, "
                         "keeping at least share P (0 < P <= 1) of each token's "
                         "gate mass (config mode: set `moe_miss_shed: P` per "
                         "model).")
    ap.add_argument("--moe-layer-shed", type=float, default=None, metavar="P",
                    help="Lossy: at decode, skip each streamed MoE layer's routed "
                         "experts with probability P (0 < P < 1) per token; the "
                         "shared expert still runs on shed layers (config mode: "
                         "set `moe_layer_shed: P` per model).")
    ap.add_argument("--prefill-feeder", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Faster prompt processing for streaming models "
                         "(--stream-experts / --stream-cpu past the wired budget): "
                         "stage prefill expert layers straight from the GGUF into "
                         "GPU-visible ring slots. Default on; config mode: set "
                         "`prefill_feeder: false` per model.")
    ap.add_argument("--decode-feeder", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Faster decode for --stream-experts models: keep the "
                         "most-used experts in a wired, popularity-managed GPU "
                         "arena and read only the misses from the GGUF. Default on "
                         "for --stream-experts (needs the every-token layers on "
                         "GPU); config mode: set `decode_feeder: false` per "
                         "model.")
    ap.add_argument("--host", default=None, help="Bind address (default from config "
                    "or 127.0.0.1).")
    ap.add_argument("--port", type=int, default=None, help="Port (default 8080).")
    ap.add_argument("--budget-gb", type=float, default=None,
                    help="Resident weight-byte budget across all models "
                         "(default: 0.8x the GPU recommended working set).")
    ap.add_argument("--max-models", type=int, default=None,
                    help="Optional secondary cap on resident model count.")
    ap.add_argument("--pin", action="append", default=[], metavar="ID_OR_PATH",
                    help="Pin a model (id or path) so it is never evicted (repeatable).")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Server default max completion tokens.")
    ap.add_argument("--prefill-step-size", type=int, default=None, metavar="N",
                    help="Prefill chunk size in tokens (default 2048) - lower "
                         "it to cap peak memory on long prompts, at some "
                         "prefill-throughput cost. Also via PREFILL_STEP_SIZE; "
                         "config mode: server.prefill_step_size.")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Never stop on EOS; decode every request to max_tokens "
                         "(forced-length throughput benchmarking; mirrors "
                         "llama-server --ignore-eos). Also via GMLX_IGNORE_EOS=1.")
    ap.add_argument("--no-auth", action="store_true",
                    help="Serve a non-loopback bind without an API key - an "
                         "explicit opt-out for setups that authenticate in "
                         "front (mTLS, reverse proxy). Loopback binds never "
                         "need it. (The required key itself is set in the config "
                         "as `server.api_key`.)")
    ap.add_argument("--stt", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Speech-to-text: serve POST /v1/audio/transcriptions via "
                         "mlx-whisper (pip install 'gmlx[stt]'; ffmpeg on "
                         "PATH). MODEL is an alias (whisper-turbo, "
                         "whisper-turbo-q4, whisper-large/medium/small/base/tiny), "
                         "any HF repo in MLX-whisper format, or a local model dir; "
                         "bare --stt picks whisper-turbo - give it a value or put "
                         "it after the positional model, or the model path is "
                         "parsed as the STT model. Works in every serve mode and "
                         "overrides a config `server.stt:`.")
    ap.add_argument("--tts", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Text-to-speech: serve POST /v1/audio/speech via "
                         "mlx-audio (pip install 'gmlx[tts]'; non-wav "
                         "formats need ffmpeg on PATH). MODEL is an alias "
                         "(kokoro, kokoro-8bit/4bit, qwen3-tts), any HF repo in "
                         "MLX-audio format, or a local model dir; bare --tts "
                         "picks kokoro. Works in every serve mode and overrides "
                         "a config `server.tts:`.")
    ap.add_argument("--embeddings", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Serve POST /v1/embeddings from a GGUF decoder-LM "
                         "embedder (alias qwen3-embed-0.6b/-4b/-8b, *.gguf, or "
                         "hf:<org>/<repo>/<file>.gguf; no extra) or an "
                         "mlx-embeddings encoder (alias embeddinggemma/arctic-l/"
                         "nomic-embed/bge-m3, HF repo, or local dir; needs the "
                         "embeddings extra). Bare flag picks qwen3-embed-0.6b; "
                         "overrides config server.embeddings:.")
    ap.add_argument("--rerank", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Serve POST /v1/rerank (Cohere/Jina shape) from a "
                         "Qwen3-Reranker GGUF (alias qwen3-rerank-0.6b/-4b/-8b, "
                         "*.gguf, or hf:<org>/<repo>/<file>.gguf; no extra). Bare "
                         "flag picks qwen3-rerank-0.6b; overrides config "
                         "server.rerank:.")
    ap.add_argument("-f", "--foreground", action="store_true",
                    help="Run the server attached to this terminal (blocking) "
                         "instead of the default detached background start. Manage "
                         "a background server with `gmlx status`/`stop`/"
                         "`restart`/`logs`.")
    ap.add_argument("--no-menubar", action="store_true",
                    help="Don't auto-start the macOS menu-bar monitor alongside a "
                         "background server (same as `server.menubar: false`).")
    ap.add_argument("--log", default=None, metavar="FILE",
                    help="Background log file (default "
                         "~/.cache/gmlx/server-<host>-<port>.log).")
    ap.add_argument("--log-level", default=None, metavar="LEVEL",
                    choices=("critical", "error", "warning", "info", "debug",
                             "trace"),
                    help="Server log verbosity: critical, error, warning, info "
                         "(default), debug, or trace. debug/trace add uvicorn's "
                         "connection-level detail.")
    ap.add_argument("--start-timeout", type=float, default=40.0, metavar="S",
                    help="Seconds to wait for a background server to become ready "
                         "(default 40; a slow model load keeps starting past it).")


def _bg_serve_args(a, cfg_path) -> list:
    """Rebuild a clean, absolute-pathed serve argv from the parsed args, for a
    faithful detached/launchd relaunch from any cwd. Omits host/port (the caller bakes
    the resolved values in) and the background-only flags; bakes an absolute --config."""
    def _abs(p):
        return os.path.abspath(os.path.expanduser(p))

    out: list = []
    if a.model:
        out.append(_abs(a.model))
    if cfg_path:
        out += ["--config", cfg_path]
    for d in (a.models_dir or []):
        out += ["--models-dir", _abs(d)]
    if a.recursive:
        out.append("-r")
    if a.hf_cache:
        out.append("--hf-cache")
    if a.mmproj:
        out += ["--mmproj", _abs(a.mmproj)]
    if a.hf_source:
        out += ["--hf-source", a.hf_source]
    if a.speculative:
        out.append("--speculative")
    if a.draft_gguf:
        out += ["--draft-gguf", _abs(a.draft_gguf)]
    if getattr(a, "draft_block_size", None):
        out += ["--draft-block-size", str(a.draft_block_size)]
    if getattr(a, "chat_template", None):
        ct = a.chat_template
        out += ["--chat-template",
                _abs(ct) if os.path.exists(os.path.expanduser(ct)) else ct]
    if a.adapter:
        out += ["--adapter", _abs(a.adapter)]
    if getattr(a, "stream_cpu", False):
        out.append("--stream-cpu")
    if getattr(a, "stream_experts", False):
        out.append("--stream-experts")
    if getattr(a, "moe_expert_mass", None) is not None:
        out += ["--moe-expert-mass", str(a.moe_expert_mass)]
    if getattr(a, "moe_experts", None) is not None:
        out += ["--moe-experts", str(a.moe_experts)]
    if getattr(a, "moe_miss_shed", None) is not None:
        out += ["--moe-miss-shed", str(a.moe_miss_shed)]
    if getattr(a, "moe_layer_shed", None) is not None:
        out += ["--moe-layer-shed", str(a.moe_layer_shed)]
    if a.budget_gb is not None:
        out += ["--budget-gb", str(a.budget_gb)]
    if a.max_models is not None:
        out += ["--max-models", str(a.max_models)]
    for p in a.pin:
        out += ["--pin", p]
    if a.max_tokens is not None:
        out += ["--max-tokens", str(a.max_tokens)]
    if getattr(a, "prefill_step_size", None) is not None:
        out += ["--prefill-step-size", str(a.prefill_step_size)]
    if getattr(a, "ignore_eos", False):
        out.append("--ignore-eos")
    if a.no_auth:
        out.append("--no-auth")
    if getattr(a, "stt", None):
        out += ["--stt"] if a.stt == "default" else ["--stt", a.stt]
    if getattr(a, "tts", None):
        out += ["--tts"] if a.tts == "default" else ["--tts", a.tts]
    if getattr(a, "embeddings", None):
        out += (["--embeddings"] if a.embeddings == "default"
                else ["--embeddings", a.embeddings])
    if getattr(a, "rerank", None):
        out += ["--rerank", a.rerank]
    # Tri-state feeder flags: None means loader default; an explicit
    # opt-in/opt-out must survive the background re-exec.
    for flag, val in (("prefill-feeder", getattr(a, "prefill_feeder", None)),
                      ("decode-feeder", getattr(a, "decode_feeder", None))):
        if val is not None:
            out.append(f"--{flag}" if val else f"--no-{flag}")
    if getattr(a, "log_level", None):
        out += ["--log-level", a.log_level]
    return out


def _bg_plan(a) -> tuple:
    """For the background start / `service install`: resolve (host, port,
    config_abspath, serve_args, api_key, menubar) cheaply (config YAML only - no engine
    import). The config is the sole key source, so a key-protected server's key comes
    from there; `menubar` reflects `server.menubar` (default on). A broken config
    raises :class:`ConfigError` - backgrounding anyway would just spawn a child
    that dies on the same error, with the message buried in the log tail."""
    host, port, cfg_path, api_key = a.host, a.port, None, None
    cfg = None
    if a.config:
        cfg_path = os.path.abspath(os.path.expanduser(a.config))
        cfg = load_config(cfg_path)
    elif not a.models_dir and not a.model:
        for p in default_config_paths():
            if p.exists():
                cfg_path = str(p)
                cfg = load_config(p)
                break
    if cfg is not None:
        host = host or cfg.host
        port = port or cfg.port
        api_key = getattr(cfg, "api_key", None)
    menubar = getattr(cfg, "menubar", True) if cfg is not None else True
    host = host or "127.0.0.1"
    port = int(port or 8080)
    return host, port, cfg_path, _bg_serve_args(a, cfg_path), api_key, menubar


def _add_target_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--host", default=None, help="Bind host (default: the single "
                    "managed server if there's one, else the config's, else "
                    "127.0.0.1).")
    ap.add_argument("--port", type=int, default=None, help="Port (default: the "
                    "single managed server if there's one, else the config's, "
                    "else 8080).")


def _ambiguous_runs(a) -> list | None:
    """The managed runfiles when a bare lifecycle verb has more than one server
    to choose from (no --host/--port given), else None."""
    if a.host is not None or a.port is not None:
        return None
    from . import lifecycle
    runs = lifecycle.list_runs()
    return runs if len(runs) > 1 else None


def _refuse_ambiguous(prog: str, runs: list) -> int:
    urls = ", ".join(r.get("url") or f"http://{r.get('host')}:{r.get('port')}"
                     for r in runs)
    print(f"error: {len(runs)} managed servers are running ({urls}) - "
          f"pick one with --port", file=sys.stderr)
    return 2


def _cmd_stop(argv: list, prog: str = "gmlx stop") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Stop a backgrounded gmlx server (SIGTERM, then SIGKILL).")
    _add_target_args(ap)
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="Seconds to wait for graceful shutdown before SIGKILL "
                         "(SIGKILL cuts any in-flight generation). Default 15.")
    a = ap.parse_args(argv)
    from . import lifecycle
    runs = _ambiguous_runs(a)
    if runs:
        return _refuse_ambiguous(prog, runs)
    host, port = lifecycle.auto_target(a.host, a.port)
    return lifecycle.stop(host, port, timeout=a.timeout)


def _cmd_restart(argv: list, prog: str = "gmlx restart") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Restart a backgrounded server with its original arguments.")
    _add_target_args(ap)
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="Graceful-stop timeout before SIGKILL (default 15).")
    ap.add_argument("--start-timeout", type=float, default=40.0, metavar="S",
                    help="Readiness wait for the new process (default 40).")
    a = ap.parse_args(argv)
    from . import lifecycle
    runs = _ambiguous_runs(a)
    if runs:
        return _refuse_ambiguous(prog, runs)
    host, port = lifecycle.auto_target(a.host, a.port)
    return lifecycle.restart(host, port, timeout=a.timeout,
                             start_timeout=a.start_timeout)


def _cmd_status(argv: list, prog: str = "gmlx status") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Show whether a backgrounded server is running (process + "
                    "auth-exempt /health; no API key needed). `ps` lists models.")
    _add_target_args(ap)
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    a = ap.parse_args(argv)
    from . import lifecycle
    runs = _ambiguous_runs(a)
    if runs:
        # Several managed servers: report on all of them instead of guessing.
        if a.json:
            infos = [lifecycle.status_info(r.get("host"), r.get("port"))
                     for r in runs]
            infos = [i for i in infos if i]
            print(json.dumps(infos, indent=2))
            return 0 if any(i["running"] for i in infos) else 3
        rcs = [lifecycle.status(r.get("host"), r.get("port")) for r in runs]
        return 0 if 0 in rcs else 3
    host, port = lifecycle.auto_target(a.host, a.port)
    return lifecycle.status(host, port, as_json=a.json)


def _cmd_logs(argv: list, prog: str = "gmlx logs") -> int:
    ap = argparse.ArgumentParser(
        prog=prog, description="Show a backgrounded server's log.")
    _add_target_args(ap)
    ap.add_argument("-n", "--lines", type=int, default=40,
                    help="Lines of history to print (default 40).")
    ap.add_argument("-f", "--follow", action="store_true",
                    help="Follow the log (tail -f; Ctrl-C to stop).")
    ap.add_argument("--clear", action="store_true",
                    help="Truncate the log file and exit (keeps the file).")
    a = ap.parse_args(argv)
    from . import lifecycle
    runs = _ambiguous_runs(a)
    if runs:
        return _refuse_ambiguous(prog, runs)
    host, port = lifecycle.auto_target(a.host, a.port)
    return lifecycle.tail_log(host, port, n=a.lines, follow=a.follow, clear=a.clear)


def _cmd_service(argv: list, prog: str = "gmlx service") -> int:
    actions = ("install", "uninstall", "status")
    usage = (f"usage: {prog} {{install|uninstall|status}} [options]\n"
             "  install    write + load a launchd LaunchAgent for the menu bar "
             "(start at login; it starts the server too unless --no-autostart); "
             "takes the same options as `serve`. --headless installs a "
             "server-only agent instead (SSH boxes, restarts on crash)\n"
             "  uninstall  unload + remove the agent(s)\n"
             "  status     show the agents' launchd state")
    if argv and argv[0] in ("-h", "--help"):
        print(usage)
        return 0
    if not argv or argv[0] not in actions:
        print(usage, file=sys.stderr)
        return 2
    action, rest = argv[0], argv[1:]
    from . import lifecycle
    if action == "install":
        ap = argparse.ArgumentParser(
            prog=f"{prog} install",
            description="Install a launchd LaunchAgent: the menu bar at login "
                        "(which starts the server when it isn't up), or with "
                        "--headless a server-only agent.")
        _add_serve_args(ap)
        ap.add_argument("--headless", action="store_true",
                        help="Install a server-only agent (no menu bar; for "
                             "GUI-less use). Restarts the server on crash.")
        ap.add_argument("--no-autostart", dest="autostart",
                        action="store_false", default=True,
                        help="Menu-bar agent only: don't start the server at "
                             "login (start it from the menu bar).")
        grp = ap.add_mutually_exclusive_group()
        grp.add_argument("--keepalive", dest="keepalive", action="store_true",
                         default=True,
                         help="--headless: restart on crash (default).")
        grp.add_argument("--no-keepalive", dest="keepalive", action="store_false",
                         help="--headless: don't auto-restart on crash.")
        a = ap.parse_args(rest)
        for flag, value in (("model", a.model), ("--mmproj", a.mmproj),
                            ("--draft-gguf", a.draft_gguf), ("--adapter", a.adapter),
                            ("--config", a.config)):
            if value and not os.path.exists(os.path.expanduser(value)):
                print(f"error: {flag}: no such file: {value}", file=sys.stderr)
                return 2
        try:
            host, port, cfg_path, serve_args, api_key, _menubar = _bg_plan(a)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if a.headless:
            return lifecycle.service_install(
                serve_args, host=host, port=port, config_abspath=cfg_path,
                log=a.log, keepalive=a.keepalive, api_key_set=bool(api_key))
        return lifecycle.service_install_menubar(
            serve_args, host=host, port=port, config_abspath=cfg_path,
            log=a.log, autostart=a.autostart,
            start_timeout=a.start_timeout, api_key=api_key)
    ap = argparse.ArgumentParser(prog=f"{prog} {action}")
    _add_target_args(ap)
    a = ap.parse_args(rest)
    host, port = lifecycle.auto_target(a.host, a.port)
    return (lifecycle.service_uninstall(host, port) if action == "uninstall"
            else lifecycle.service_status(host, port))


def _cmd_serve(argv: list, prog: str = "gmlx serve") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="The gmlx server: continuously batched, multi-model, "
                    "OpenAI/Anthropic-compatible (text + VLM + MTP).",
    )
    _add_serve_args(ap)
    # Internal: set in the --headless LaunchAgent plist (service_install).
    ap.add_argument("--launchd", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.launchd and a.foreground and sys.platform == "darwin":
        # launchd-parented: re-exec through the (just refreshed) renamed stub
        # so ps reads "gmlx" - see procname.launchd_reexec.
        from . import procname
        procname.launchd_reexec(procname.named_python, ["serve", *argv])

    n_modes = sum(bool(x) for x in (a.config, a.models_dir, a.model))
    if n_modes > 1:
        ap.error("choose one of: --config FILE | --models-dir DIR | a positional GGUF")
    # --mmproj (VLM) + --speculative/--draft-gguf (MTP) now coexist: a resident VLM
    # speculates on text-only requests and serves image/audio requests on the VLM
    # forward (gemma4 needs --draft-gguf; qwen3.5/3.6 use the native head).
    if a.model and a.mmproj and a.adapter:
        ap.error("--adapter (live GGUF LoRA) on a VLM (--mmproj) is not supported yet.")

    # Validate every positional/flag file cheaply before the engine import and
    # the port bind, so a path typo gets one clear line instead of a half-up
    # server failing inside the uvicorn worker.
    for flag, value in (("model", a.model), ("--mmproj", a.mmproj),
                        ("--draft-gguf", a.draft_gguf), ("--adapter", a.adapter),
                        ("--config", a.config)):
        if value and not os.path.exists(os.path.expanduser(value)):
            hint = ""
            if str(value).startswith(("hf:", "http://", "https://")):
                hint = (" (remote refs work with `gmlx validate` / "
                        "`gmlx pull`; serve needs a local file)")
            what = "no such file" if flag == "model" else f"{flag}: no such file"
            print(f"error: {what}: {value}{hint}", file=sys.stderr)
            return 2

    # --print-config: resolve the effective config and dump it, no engine, no spawn.
    if a.print_config:
        try:
            cfg, _ = _resolve_cfg(a)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        _overlay_cli_flags(cfg, a)       # flags beat the file, same as at startup
        discovery.fill_families(cfg)     # header-only read; shows detected family
        print(_dump_cfg_yaml(cfg))
        return 0

    # Default start is detached: spawn a foreground `serve` child, wait for it to be
    # ready, return - no engine import in this (parent) process. A non-loopback /
    # missing-config error in the child surfaces via the child-death log tail. On a
    # macOS GUI session we also raise the menu-bar monitor (unless opted out), then
    # the parent exits and the shell is free for `gmlx launch`.
    if not a.foreground:
        from . import lifecycle
        try:
            host, port, cfg_path, serve_args, api_key, menubar = _bg_plan(a)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        rc = lifecycle.start_background(
            serve_args, host=host, port=port, config_abspath=cfg_path,
            log=a.log, start_timeout=a.start_timeout, api_key=api_key)
        if rc == 0 and menubar and not a.no_menubar \
                and lifecycle.gui_session_available():
            lifecycle.start_menubar(auto=True)  # one machine-wide bar; tracks the primary
        return rc

    try:
        _import_serving()
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        cfg, reload_fn = _resolve_cfg(a)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return _serve(cfg, a, reload_fn)


def _resolve_cfg(a) -> tuple:
    """Return ``(ServerCfg, reload_fn)`` for the selected start mode."""
    if a.config:
        path = a.config
        return load_config(path), _make_reload_fn(path)
    if a.models_dir:
        return _discovery_cfg(a.models_dir, a), None
    if a.model:
        return _single_model_cfg(a), None
    # bare: first existing default config, else discovery-scan the default dir.
    # Informational notes go to stderr: `--print-config > file` must leave
    # stdout pure YAML (the emitted header promises --config round-trips).
    for p in default_config_paths():
        if p.exists():
            print(f"[server] loading config {p}", file=sys.stderr)
            return load_config(p), _make_reload_fn(str(p))
    print("[server] no config found; discovering the current directory "
          "(pass --models-dir DIR, or `gmlx init` to save a config)",
          file=sys.stderr)
    return _discovery_cfg([_DEFAULT_DISCOVER_DIR], a), None


def _make_reload_fn(path):
    def _reload():
        from .server_bridge_vlm import register_resolved_models
        cfg = load_config(path)
        register_resolved_models(cfg)        # warm entries persist (keyed by path)
        return {"models": len(cfg.models)}
    return _reload


def _overlay_cli_flags(cfg, a) -> None:
    """Mirror the flag-beats-config precedence :func:`_serve` applies at startup
    onto ``cfg``, so ``--print-config`` shows the settings the server would run
    with. Service values print as passed (a bare ``--stt`` as its default
    alias); startup additionally resolves them to models."""
    if a.host:
        cfg.host = a.host
    if a.port:
        cfg.port = int(a.port)
    if a.budget_gb is not None:
        cfg.budget_gb = a.budget_gb
    if a.max_models is not None:
        cfg.max_models = a.max_models
    if a.no_auth:
        cfg.no_auth = True
    if getattr(a, "hf_cache", False):
        cfg.hf_cache = True
    if any(getattr(a, k, None) for k in ("stt", "tts", "embeddings", "rerank")):
        from . import embeddings as _emb, rerank as _rr, stt as _stt, tts as _tts
        for key, alias in (("stt", _stt.DEFAULT_STT_ALIAS),
                           ("tts", _tts.DEFAULT_TTS_ALIAS),
                           ("embeddings", _emb.DEFAULT_EMBEDDINGS_ALIAS),
                           ("rerank", _rr.DEFAULT_RERANK_ALIAS)):
            v = getattr(a, key, None)
            if v:
                setattr(cfg, key, alias if v == "default" else v)


def _dump_cfg_yaml(cfg: ServerCfg) -> str:
    """Serialize a resolved :class:`ServerCfg` (defaults filled in) to YAML - every
    schema key with its effective value, for `serve --print-config`. The output
    uses the on-disk schema shape (server / profiles / rules / models / aliases),
    so it feeds back through `--config` / :func:`load_config` unchanged."""
    import dataclasses

    import yaml
    d = dataclasses.asdict(cfg)
    server = {k: d.pop(k) for k in
              ("host", "port", "api_key", "no_auth", "model_dirs", "budget_gb",
               "max_models", "hf_cache", "menubar", "token_queue_timeout_s",
               "prefill_step_size", "cache_limit_gb", "family_defaults",
               "stochastic_mtp", "stt", "tts", "embeddings", "rerank",
               "defaults", "cache", "assistants", "assistant_allow_remote")}
    # Profile/model names double as dict keys; drop the redundant fields so the
    # entries match the file schema (which has no `name`/`id` keys).
    profiles = {n: {k: v for k, v in p.items() if k != "name"}
                for n, p in d.pop("profiles").items()}
    models = {mid: {k: v for k, v in m.items() if k != "id"}
              for mid, m in d.pop("models").items()}
    talk = d.pop("talk")
    talk.pop("assistant", None)     # attached shared object, not a talk: key
    doc = {"server": server, "profiles": profiles, "rules": d.pop("rules"),
           "models": models, "aliases": d.pop("aliases"),
           "discover": d.pop("discover"), "talk": talk,
           "assistant": d.pop("assistant")}
    assert not d, f"ServerCfg fields missing from --print-config: {sorted(d)}"
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False,
                          width=2 ** 16)
    return ("# effective gmlx config (every key + default shown); "
            "valid as a --config file\n" + body)


def _discovery_cfg(dirs: list, a) -> ServerCfg:
    specs = [DiscoverSpec(dir=d, recursive=a.recursive) for d in dirs]
    models = discovery.scan_dirs(specs, dirs)
    return ServerCfg(
        host=a.host or "127.0.0.1",
        port=a.port or 8080,
        model_dirs=list(dirs),
        budget_gb=a.budget_gb,
        max_models=a.max_models,
        hf_cache=a.hf_cache,
        defaults=ServerDefaults(),
        models={m.id: m for m in models},
    )


def _single_model_cfg(a) -> ServerCfg:
    mp = os.path.abspath(os.path.expanduser(a.model))
    mid, _q = discovery.derive_id(os.path.basename(a.model))
    mid = mid or "model"
    speculative = bool(a.speculative or a.draft_gguf)
    # A --chat-template override rides on the model's overrides (the same slot a
    # config `overrides: {chat_template: ...}` uses), so it flows through resolve_model
    # -> the load bridge identically.
    overrides = ({"chat_template": a.chat_template}
                 if getattr(a, "chat_template", None) else {})
    model = ModelCfg(
        id=mid,
        path=mp,
        mmproj=os.path.abspath(os.path.expanduser(a.mmproj)) if a.mmproj else None,
        draft_gguf=(os.path.abspath(os.path.expanduser(a.draft_gguf))
                    if a.draft_gguf else None),
        adapter=(os.path.abspath(os.path.expanduser(a.adapter))
                 if a.adapter else None),
        speculative=speculative,
        overrides=overrides,
        stream=("cpu" if getattr(a, "stream_cpu", False)
                else ("experts" if getattr(a, "stream_experts", False)
                      else None)),
        moe_experts=getattr(a, "moe_experts", None),
        moe_expert_mass=getattr(a, "moe_expert_mass", None),
        moe_miss_shed=getattr(a, "moe_miss_shed", None),
        moe_layer_shed=getattr(a, "moe_layer_shed", None),
        prefill_feeder=getattr(a, "prefill_feeder", None),
        decode_feeder=getattr(a, "decode_feeder", None),
        pin=True,                            # the single model is always pinned
    )
    return ServerCfg(
        host=a.host or "127.0.0.1",
        port=a.port or 8080,
        budget_gb=a.budget_gb,
        max_models=a.max_models,
        hf_cache=a.hf_cache,
        defaults=ServerDefaults(model=mid),
        models={mid: model},
    )


def _preload_id(cfg: ServerCfg) -> str | None:
    """The model the lifespan preloads at startup: a pinned model, else the default
    model, else the sole model, else none (lazy load on first request)."""
    for mid, m in cfg.models.items():
        if m.pin:
            return mid
    if cfg.defaults.model:
        return cfg.defaults.model
    if len(cfg.models) == 1:
        return next(iter(cfg.models))
    return None


def _resolve_service(key: str, resolver, value, model_dirs):
    """Resolve one optional service model (embeddings / rerank). A missing
    *file* degrades - warn to the log and disable the service (None) - so a
    deleted GGUF doesn't take the whole server down; any other config error
    still fails fast, naming the config key it came from."""
    try:
        return resolver(value, model_dirs)
    except MissingModelFile as e:
        print(f"[server] {key} disabled - {e}", file=sys.stderr)
        return None
    except ConfigError as e:
        raise ConfigError(f"{key}: {e}") from None


def _serve(cfg: ServerCfg, a, reload_fn) -> int:
    from . import server_bridge_vlm as serving
    from .residency import install_gguf_residency_pool
    from .server_patches import install_server_patches
    from .server_bridge_vlm import (
        install_gguf_server_bridge,
        register_gguf_vlm,
        register_resolved_models,
    )

    register_resolved_models(cfg)
    # MLX buffer-cache limit: explicit env/config wins, else the auto policy
    # bounds the cache when the biggest configured model leaves little slack
    # (deep-context safety; see server_memory).
    from .server_memory import apply_cache_limit
    apply_cache_limit(cfg)
    # Single-model --hf-source override (niche): re-register the VLM with the
    # explicit processor source (ModelCfg carries no hf_source field).
    if a.model and a.hf_source and a.mmproj:
        register_gguf_vlm(os.path.abspath(os.path.expanduser(a.model)),
                          os.path.abspath(os.path.expanduser(a.mmproj)),
                          hf_source=a.hf_source)
    if a.max_tokens is not None:
        os.environ["MLX_VLM_MAX_TOKENS"] = str(a.max_tokens)
    if getattr(a, "draft_block_size", None):
        # Read lazily at drafter-load time (server_bridge_vlm.load_drafter).
        os.environ["GMLX_DRAFT_BLOCK_SIZE"] = str(a.draft_block_size)
    if cfg.stochastic_mtp or getattr(a, "stochastic_mtp", False):
        # Startup-only, like the timeout below; a config reload does not
        # re-apply it.
        from .speculative import set_stoch_accept

        set_stoch_accept(True)
        print("[server] stochastic MTP acceptance: on (sampled requests keep "
              "the sampling distribution but are not token-identical)")
    # The config's token-queue timeout is authoritative for this server: it drives
    # mlx-vlm's per-request "wait for the next token" guard (default 600s; <=0 waits
    # forever). Read per request from the env, so setting it here is enough.
    if cfg.token_queue_timeout_s is not None:
        os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = str(cfg.token_queue_timeout_s)
        secs = cfg.token_queue_timeout_s
        human = "disabled (waits forever)" if secs <= 0 else f"{secs:g}s"
        print(f"[server] token-queue timeout: {human}")
    elif "MLX_VLM_TOKEN_QUEUE_TIMEOUT" not in os.environ:
        # mlx-vlm's 600s default is shorter than a deep-context dense prefill
        # (>10 min to the first token); the SSE keepalive keeps clients on the
        # line, so the server must not give up first. An exported env wins.
        os.environ["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = "1800"
        print("[server] token-queue timeout: 1800s (default)")

    # Decode-loop housekeeping (mx.eval(cache) + mx.clear_cache) interval:
    # mlx-vlm's default of every 50 steps costs ~2.5 ms per event (~0.05
    # ms/tok amortized at B=1). 256 keeps the memory bound while cutting the
    # amortized cost ~5x. An exported env wins.
    os.environ.setdefault("MLX_VLM_BATCH_CACHE_EVAL_INTERVAL", "256")

    # Prefill chunk size: flag > config > exported env > upstream default
    # (2048). Read per request from the env (both the stock serve path and the
    # MTP re-enable in spec_engine), so setting it here is enough.
    step = getattr(a, "prefill_step_size", None)
    if step is None:
        step = cfg.prefill_step_size
    if step is not None:
        if step <= 0:
            print(f"[server] ignoring non-positive prefill step size {step}")
        else:
            os.environ["PREFILL_STEP_SIZE"] = str(step)
            print(f"[server] prefill step size: {step} tokens")

    resolved = serving.resolved_models()
    preload = _preload_id(cfg)
    if preload:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = preload

    pinned_paths = {rm.path for rm in resolved.values() if rm.pin}
    for p in a.pin:
        rp = resolved.get(p)
        pinned_paths.add(rp.path if rp is not None
                         else os.path.abspath(os.path.expanduser(p)))

    install_gguf_server_bridge()
    budget_gb = a.budget_gb if a.budget_gb is not None else cfg.budget_gb
    budget_bytes = int(budget_gb * 1024**3) if budget_gb else None
    install_gguf_residency_pool(
        budget_bytes=budget_bytes,
        max_models=a.max_models if a.max_models is not None else cfg.max_models,
        pinned=pinned_paths or None,
    )
    # Speech-to-text: CLI --stt overrides config `server.stt:`; resolve aliases
    # and fail fast (with install guidance) before binding the port.
    stt_value = getattr(a, "stt", None) or getattr(cfg, "stt", None)
    if stt_value:
        from . import stt as stt_mod
        cfg.stt = stt_mod.resolve_stt_model(stt_value)
        stt_mod.import_mlx_whisper()
        # Warm the Whisper model in the background so the first transcription
        # request is a cache hit, not a cold HF download + load. Best-effort
        # and non-blocking: the server (and LLM load) come up immediately.
        stt_mod.prewarm(cfg.stt)
    else:
        cfg.stt = None

    # Text-to-speech: CLI --tts overrides config `server.tts:`; resolve aliases
    # and fail fast (with install guidance) before binding the port, then warm
    # in the background like STT.
    tts_value = getattr(a, "tts", None) or getattr(cfg, "tts", None)
    if tts_value:
        from . import tts as tts_mod
        cfg.tts = tts_mod.resolve_tts_model(tts_value)
        tts_mod.import_mlx_audio()
        tts_mod.prewarm(cfg.tts)
    else:
        cfg.tts = None

    # Text embeddings: CLI --embeddings overrides config `server.embeddings:`;
    # resolve aliases and fail fast (with install guidance) before binding the
    # port, then warm in the background like STT/TTS. Relative GGUF paths
    # search server.model_dirs, same as a models: entry; a missing file
    # degrades to a disabled service (see _resolve_service).
    model_dirs = getattr(cfg, "model_dirs", None) or []
    emb_value = getattr(a, "embeddings", None) or getattr(cfg, "embeddings", None)
    cfg.embeddings = None
    if emb_value:
        from . import embeddings as emb_mod
        cfg.embeddings = _resolve_service(
            "server.embeddings", emb_mod.resolve_embeddings_model,
            emb_value, model_dirs)
        if cfg.embeddings:
            if not emb_mod._is_gguf_ref(cfg.embeddings):
                emb_mod.import_mlx_embeddings()   # GGUF embedders use the runtime loader
            emb_mod.prewarm(cfg.embeddings)

    # Reranker: CLI --rerank overrides config `server.rerank:`; same degrade
    # rule as embeddings.
    rerank_value = getattr(a, "rerank", None) or getattr(cfg, "rerank", None)
    cfg.rerank = None
    if rerank_value:
        from . import rerank as rerank_mod
        cfg.rerank = _resolve_service(
            "server.rerank", rerank_mod.resolve_rerank_model,
            rerank_value, model_dirs)
        if cfg.rerank:
            rerank_mod.prewarm(cfg.rerank)

    # API key: the config `server.api_key` is the sole source (so a managed /
    # menu-bar client can read the same key the server enforces from the same file;
    # the runfile never stores the key itself). Policy: loopback binds need no key; a
    # non-loopback bind REFUSES to start without one unless --no-auth / server.no_auth
    # opts out explicitly (for setups that authenticate in front - mTLS, reverse proxy).
    cfg.api_key = getattr(cfg, "api_key", None) or None
    no_auth = a.no_auth or getattr(cfg, "no_auth", False)

    host = a.host or cfg.host
    port = a.port or cfg.port
    if host not in _LOOPBACK and not cfg.api_key and not no_auth:
        print(f"error: binding {host} exposes this server beyond localhost - "
              f"set `server.api_key` in your config, or opt out explicitly with "
              f"--no-auth if auth is handled in front (mTLS, reverse proxy).",
              file=sys.stderr)
        return 2
    # Same posture for served assistants on the CLI-resolved host (config
    # validation already refused a non-loopback `server.host`; this covers
    # a --host override).
    if getattr(cfg, "assistants", None) and host not in _LOOPBACK \
            and not getattr(cfg, "assistant_allow_remote", False):
        print(f"error: binding {host} exposes the assistant tool loop beyond "
              f"localhost - remove `server.assistants`, bind a loopback host, "
              f"or set `server.assistant_allow_remote: true` (anyone holding "
              f"the API key can then drive tool execution on this host).",
              file=sys.stderr)
        return 2

    # The patches need the *resolved* bind (CLI may override the config): the
    # loopback host guard keys off cfg.host.
    cfg.host, cfg.port = host, port
    install_server_patches(cfg, reload_fn=reload_fn)
    if getattr(a, "ignore_eos", False) or env_bool("GMLX_IGNORE_EOS", False):
        from .server_patches import install_ignore_eos
        install_ignore_eos()
        print("[server] ignore-eos: decode runs to max_tokens (EOS suppressed)")

    if reload_fn is not None:
        import signal

        def _on_sighup(_sig, _frame):
            try:
                print(f"[server] SIGHUP config reload: {reload_fn() or {}}")
            except Exception as exc:
                print(f"[server] SIGHUP config reload failed: {exc}",
                      file=sys.stderr)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _on_sighup)
            print(f"[server] config reload: POST /v1/reload, or "
                  f"kill -HUP {os.getpid()}")

    if cfg.api_key:
        print("[server] api-key auth on (every endpoint except /health)")
    elif host not in _LOOPBACK:
        print(f"[server] warning: binding {host} with no auth (--no-auth)")
    print(f"[server] {len(cfg.models)} model(s) configured"
          + (f", preloading {preload} in background" if preload else ""))
    _print_models(cfg.models)
    if cfg.stt:
        print(f"[server] stt: {cfg.stt}  (POST /v1/audio/transcriptions, "
              f"pre-warming in background)")
    if cfg.tts:
        print(f"[server] tts: {cfg.tts}  (POST /v1/audio/speech, "
              f"pre-warming in background)")
    if getattr(cfg, "embeddings", None):
        print(f"[server] embeddings: {cfg.embeddings}  (POST /v1/embeddings, "
              f"pre-warming in background)")
    if getattr(cfg, "rerank", None):
        print(f"[server] rerank: {cfg.rerank}  (POST /v1/rerank, "
              f"pre-warming in background)")

    import uvicorn

    from . import server_patches

    # Preload off the startup path: mlx-vlm's lifespan loads MLX_VLM_PRELOAD_MODEL
    # synchronously *before* the port accepts connections, so a big model makes the
    # whole server unreachable (menu bar / health probes see "down") until it
    # finishes. The env was already consumed by the residency-pool install above
    # (it pins the preload); pop it now so the lifespan skips the blocking load, and
    # warm the model in a background thread instead - the port binds and /health
    # answers immediately while the load runs (the retained hold keeps it resident,
    # exactly as the lifespan hold would have). defaults.preload extras warm after
    # the primary, LRU-evictable.
    pre = cfg.defaults.preload
    extras = [m for m in (list(cfg.models) if pre == "all" else list(pre or ()))
              if m != preload]
    if preload or extras:
        os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
        server_patches.spawn_preload_warm(preload, extras)
        if extras:
            print(f"[server] preload: warming {', '.join(extras)} in background")

    loop = "uvloop" if _has_uvloop() else "auto"
    uvicorn.run("mlx_vlm.server:app", host=host, port=port, workers=1,
                server_header=False, loop=loop,
                # log_level re-levels uvicorn's own loggers after the
                # dictConfig; the config itself carries the level to the
                # gmlx/mlx_vlm loggers, so the flag governs the whole server
                # while the timestamped formatters and noise filters stay.
                log_level=getattr(a, "log_level", None),
                log_config=server_patches.uvicorn_log_config(
                    getattr(a, "log_level", None)))
    return 0


def _print_models(models: dict) -> None:
    """Print the id -> path table so the addressable ids are obvious on start."""
    if not models:
        print("  (no models)")
        return
    width = max(len(mid) for mid in models)
    for mid, m in models.items():
        tags = []
        if getattr(m, "speculative", False):
            tags.append("mtp")
        if getattr(m, "mmproj", None):
            tags.append("vlm")
        if getattr(m, "pin", False):
            tags.append("pinned")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {mid.ljust(width)}  {m.path}{suffix}")


if __name__ == "__main__":
    sys.exit(main())
