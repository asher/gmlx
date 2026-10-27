"""``gmlx`` batched OpenAI/Anthropic server - the platform's serving layer.

Composes the gmlx loader bridge (:func:`server_bridge_vlm.install_gguf_server_bridge`),
the multi-model residency pool (:func:`residency.install_gguf_residency_pool`),
the config-driven HTTP surface (:func:`gmlx.serve.patches.install_server_patches`),
and mlx-vlm's FastAPI app + ``BatchGenerator`` continuous-batching engine into one
GGUF-only HTTP server - **text**, **VLM** (LLM GGUF + ``mmproj``), and
**speculative/MTP** models.

Start modes (resolution order):

* ``init --models-dir DIR`` - discover the GGUFs under ``--models-dir`` and scaffold a
  starter YAML config to ``--out`` (default ``~/.config/gmlx/gmlx.yaml``; the
  only path that writes a file; refuses overwrite without ``--force``).
* ``sync-models`` - reconcile an existing config's ``models:`` with what's on disk
  (keep, drop gone, add new); preserves comments. With ``--from-hf-cache`` (or a
  config already carrying ``hf_cache: true``) it also reconciles cache-resident GGUFs.
  Default config unless ``--config``.
* ``launch <client>`` - point a client (opencode, ...) at a **running** server
  and exec it (see :mod:`gmlx.commands.launch`). No auto-install; the server must
  already be up.
* ``--config FILE`` - serve a YAML config (named models + profiles).
* ``--models-dir DIR`` - serve a discovery scan of a directory (in-memory config).
* a positional ``model.gguf`` - serve a single model (wrapped as a one-model config).
* bare - load the first existing default config, else exit 2 with the way to
  make one.

Every mode converges on one :class:`config.ServerCfg`: register it, install the
bridge + residency pool + HTTP patches (before the lifespan preload), then
``uvicorn`` serves the ``mlx_vlm.server:app`` FastAPI app in-process (single
worker, no reload) so the patches hold.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

import gmlx.load.discovery as discovery
from gmlx.config import (
    DEFAULT_CONFIG_WRITE,
    LOAD_ENV,
    LOOPBACK_HOSTS,
    ConfigError,
    ConfigWriteError,
    DiscoverSpec,
    MissingModelFile,
    ModelCfg,
    ServerCfg,
    ServerDefaults,
    config_write_target,
    default_config_paths,
    default_config_write_path,
    edit_config_yaml,
    load_config,
    replace_config_text,
    resolve_model,
    resolve_path,
    speculative_off,
)
from gmlx.envflags import env_bool

NO_CONFIG_LINE = ("No gmlx config yet. Run gmlx init to create ~/.config/gmlx/gmlx.yaml, "
                  "or serve one model with gmlx serve <file.gguf>.")


def _refuse_bare_without_config(a) -> bool:
    """A start with no --config, --models-dir or GGUF needs a config in a
    default location. Without one, print the way to make one and return True."""
    if a.config or a.models_dir or a.model:
        return False
    if any(p.exists() for p in default_config_paths()):
        return False
    print(NO_CONFIG_LINE, file=sys.stderr)
    return True


def _ratio_flag(raw: str):
    """--decode-prefill-ratio value: a float or the literal 'auto'."""
    if raw.strip().lower() == "auto":
        return "auto"
    try:
        return float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number or 'auto', got {raw!r}")


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
    """Import the serving stack, lazily so ``init`` / ``launch`` / ``--help``
    never need it, and a missing serving dependency names the fix instead of
    a bare ModuleNotFoundError."""
    try:
        from . import residency  # noqa: F401
        from . import patches  # noqa: F401
        from . import bridge_vlm as server_bridge_vlm  # noqa: F401
    except ImportError as exc:
        root = (exc.name or "").split(".")[0]
        if root in ("fastapi", "uvicorn", "starlette"):
            raise ImportError(
                "gmlx serve needs its serving dependencies (FastAPI, "
                "uvicorn), which install with gmlx and are required "
                "even for text-only serving. A missing one usually means a "
                "broken install - reinstall gmlx.") from exc
        raise


def main(argv: list | None = None, prog: str | None = None) -> int:
    """``prog`` overrides the help/usage program name (the ``gmlx`` umbrella
    passes ``gmlx <verb>``; a bare ``python -m gmlx.serve.server`` keeps the
    ``gmlx serve`` defaults)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        return _cmd_init(argv[1:], prog=prog or "gmlx init")
    if argv and argv[0] == "sync-models":
        return _cmd_sync(argv[1:], prog=prog or "gmlx sync-models")
    if argv and argv[0] == "launch":
        from gmlx.commands.launch import cmd_launch
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
        offsets = lifecycle.log_offsets()
        signalled = lifecycle.reload_config(target)
    except Exception:
        return
    for host, port, pid in signalled:
        print(f"reloaded the running server at {host}:{port} (pid {pid}) "
              f"to pick up the config change")
        log, size = offsets.get((host, port), (None, 0))
        for line in lifecycle.services_off_after(log, size) if log else []:
            print(f"note: {line.removeprefix('[server] ')}", file=sys.stderr)
    if signalled:
        return
    # A server with no runfile, such as `gmlx serve --foreground`, reloads
    # over HTTP at the config's address.
    try:
        got = lifecycle.reload_over_http(target)
    except Exception:
        return
    if got is None:
        return
    where, why, off = got
    if why is None:
        print(f"reloaded the running server at {where} to pick up the "
              "config change")
        for key, hint in sorted(off.items()):
            print(f"note: server.{key} is off, because its optional {key} extra "
                  f"is not installed. To add it: {hint}", file=sys.stderr)
    else:
        print(f"note: the server at {where} did not reload ({why}); "
              "restart it to pick up the config change", file=sys.stderr)


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
                         "cached prefixes land under $XDG_CACHE_HOME/gmlx/apc "
                         "(default ~/.cache) and survive "
                         "an idle-unload / restart. A bare --disk-cache caps it at "
                         "50 GB per model, or pass a size (e.g. --disk-cache 100).")
    ap.add_argument("-r", "--recursive", action=argparse.BooleanOptionalAction,
                    default=None,
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
                         "scaffolding flags on a terminal). Other flags pre-fill "
                         "its answers.")
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
        return _init_interactive(a, ap)
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


def _init_seeds(a, ap):
    """The wizard answers that the init flags pre-fill."""
    import gmlx.commands.wizard as wizard

    services = {}
    for key, value in (("stt", a.with_stt), ("tts", a.with_tts),
                       ("embeddings", a.with_embeddings),
                       ("rerank", a.with_rerank)):
        if value is not None:
            services[key] = None if value is _SVC_DEFAULT else value
    return wizard.Seeds(
        dirs=a.models_dir, recursive=a.recursive, hf_cache=a.from_hf_cache,
        disk_cache_gb=a.disk_cache, services=services,
        default_model=a.default_model,
        ttl_s=_resolve_duration("--idle-ttl", a.idle_ttl, ap),
        timeout_s=_resolve_duration("--request-timeout", a.request_timeout, ap),
        overwrite=a.force, install=a.install, port=a.port)


def _out_refused(out) -> bool:
    """Print the error of the write and give True when the write refuses
    the config path ``out``: a link that a container client can change, a
    folder that gmlx cannot write, or a path that is not a file, such as a
    folder. The checks run in the order of the write, so a link that a
    client planted is named as such, also when it leads to a folder. Init
    stops before the overwrite question, the scan and the wizard, which
    lead only to that error."""
    from gmlx.config import _not_a_file

    path = os.path.abspath(os.path.expanduser(str(out)))
    try:
        config_write_target(path, "--out")
        if os.path.exists(path) and not os.path.isfile(path):
            raise _not_a_file(path, "--out")
    except ConfigWriteError as e:
        print(f"error: {e}", file=sys.stderr)
        return True
    return False


def _init_interactive(a, ap) -> int:
    import gmlx.commands.wizard as wizard
    default_out = a.out or DEFAULT_CONFIG_WRITE
    if _out_refused(default_out):
        return 1
    seeds = _init_seeds(a, ap)
    io = wizard.make_io()
    try:
        outcome = wizard.run_wizard(
            default_out=default_out, seeds=seeds,
            allow_install=not a.no_install, io=io)
        if outcome is None:
            return 1
        rc = _finish_write(outcome.out, outcome.text, outcome.models,
                           no_reload=a.no_reload)
        if rc != 0:
            return rc
        models = outcome.models
        if not models and _offer_pull(io, outcome.out):
            from gmlx.commands import manage
            print()
            rc = manage.cmd_pull(["--config", str(outcome.out)])
            # A cancel in the picker (130) leaves init at 0 with the pull hint.
            if rc not in (0, 130):
                return rc
            models = list(load_config(outcome.out).models.values())
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 1
    _print_next_steps(outcome.out, models)
    return 0


def _offer_pull(io, out: Path) -> bool:
    """After a zero-model init on a terminal, ask whether to pull a recommended
    model now. A config with no model_dirs has nowhere to put one."""
    from gmlx.commands import manage
    if not manage._want_interactive():
        return False
    try:
        if not load_config(out).model_dirs:
            return False
    except Exception:
        return False
    return io.yesno("Download a recommended model now?", default=True)


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
    from gmlx.commands.wizard import parse_duration
    try:
        return parse_duration(value)
    except ValueError:
        ap.error(f"{flag}: expected a duration like 900, 15m, 1h, or 'none' "
                 f"(got {value!r})")


def _install_for_services(stt_v, tts_v) -> None:
    # Embeddings and rerank need no extra: every backend is core.
    import gmlx.commands.extras as extras
    wanted = []
    if stt_v:
        wanted.append("stt")
    if tts_v:
        wanted.append("tts")
    for extra in wanted:
        if not extras.extra_installed(extra):
            extras.install_extra(extra)


def _init_scaffold(a, ap) -> int:
    dirs = a.models_dir or []
    if not dirs and not a.from_hf_cache:
        ap.error("need --models-dir DIR (repeatable) or --from-hf-cache "
                 "(or run `gmlx init` with no flags for the guided wizard)")
    out = Path(os.path.expanduser(a.out)) if a.out else default_config_write_path()
    if _out_refused(out):
        return 1
    if out.exists() and not a.force:
        print(f"refusing to overwrite {out} (use --force)", file=sys.stderr)
        return 1
    # A link that leads to no file still names a file, which the write would
    # make. So it needs --force too.
    if out.is_symlink() and not a.force:
        print(f"refusing to write through {out}, a link to {os.readlink(out)}, which does "
              "not exist (use --force)", file=sys.stderr)
        return 1

    models = []
    scan_stats: dict = {}
    missing = discovery.missing_dirs(dirs)
    for d in missing:
        print(f"{d} does not exist yet - `gmlx pull` creates it")
    specs = [DiscoverSpec(dir=d, recursive=bool(a.recursive))
             for d in dirs if d not in missing]
    if specs:
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
    if rc != 0:
        return rc
    _print_next_steps(out, models)
    if a.install:
        _install_for_services(stt_v, tts_v)
    return 0


def _finish_write(out: Path, text: str, models, *, no_reload: bool,
                  skipped: int = 0) -> int:
    """Commit the rendered config: write it, print the summary + next step, and
    SIGHUP a server already running it. Shared by the wizard and the flag path."""
    from gmlx.container.settings import _tilde

    # With --force this replaces an existing config. A config link stays a
    # link, and the file that it leads to gets the new text.
    try:
        real = config_write_target(out, "--out")
        try:
            os.makedirs(os.path.dirname(real), exist_ok=True)
        except OSError as e:
            raise ConfigWriteError(
                f"could not make the folder {_tilde(os.path.dirname(real))} for the config "
                f"{_tilde(os.path.abspath(out))} ({e.strerror or e}).",
                "Check that you can make that folder, then try again.") from e
        replace_config_text(real, text, "--out")
    except ConfigWriteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out} ({len(models)} model(s) discovered)")
    _print_models({m.id: m for m in models})
    if not models:
        if skipped:
            print(f"\nfound {skipped} .gguf file(s) but none were readable as "
                  f"GGUF (see the [discover] skip reasons above - truncated "
                  f"downloads?). Fix or re-pull them, then run "
                  f"`gmlx sync-models`.")
        else:
            print("\nno GGUFs found yet. `gmlx pull` downloads one into your model "
                  "dir and adds it. For GGUFs you copy in, run `gmlx sync-models`.")
    else:
        print("\nsampling: every model starts from its family's model-card "
              "defaults; request\n`<id>@coding` (or @instruct / @creative / "
              "@reasoning-*) to switch intent.\n`gmlx profiles` prints the "
              "full table.")
    _reload_running(out, skip=no_reload)
    return 0


def _print_next_steps(out: Path, models) -> None:
    """The next commands after init wrote ``out``: a pull first when it holds
    no model, then serve."""
    # When the config lands where a bare `serve` would find it first, the `--config`
    # flag is redundant - show the shorter command.
    # No ./gmlx.yaml line here: the next step names --out with --config.
    first = next((p for p in default_config_paths(note_local=False) if p.exists()), None)
    bare = first is not None and os.path.realpath(first) == os.path.realpath(out)
    cfg_arg = "" if bare else f" --config {out}"
    pull = "" if models else f"gmlx pull{cfg_arg}  ->  "
    print(f"\nnext: {pull}gmlx serve{cfg_arg}")
    # On macOS, point at the launchd agent for a server that starts at every login
    # (service is macOS-only; the hint would be a dead end elsewhere).
    if sys.platform == "darwin":
        print(f"  to start it now and at every login: gmlx service install{cfg_arg}")


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
    ap.add_argument("--add-companions", action="store_true",
                    help="Also add an mmproj or drafter the scan pairs with a "
                         "model already in the config, when its entry has no "
                         "such key, live or commented out. No drafter goes to "
                         "an entry set to speculative: false.")
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
    known_models = {}                              # resolved path -> ModelCfg
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
            known_models[rp] = _resolved_drafter(mc, cfg.model_dirs)
    if missing_roots:
        print(f"warning: model_dirs root(s) not on disk right now: "
              f"{', '.join(missing_roots)} - their entries are kept unverified",
              file=sys.stderr)
    if not hf_cache_ok and any(r == "hf cache unreadable" for _, r in unverified):
        print("warning: the local Hugging Face cache is unreadable - hf: entries "
              "are kept unverified", file=sys.stderr)

    # A `stream:` key discovery wrote holds only while the model is over the
    # memory budget; the budget moves with iogpu.wired_limit_mb, so each
    # sync judges it again.
    restream = stale_stream_marks(                 # id -> {"stream", "speculative"}
        path, {mc.id: rp for rp, mc in known_models.items()})
    # The scan pairs a drafter only with a resident model, as the entry
    # reads once the key is gone.
    for rp, mc in known_models.items():
        if mc.id in restream:
            known_models[rp] = dataclasses.replace(mc, stream=None)

    # Discover, skipping anything already configured (by id or by resolved path).
    # `draft_pairs` and `mmproj_pairs` are the drafters and mmprojs that pair
    # into an entry already configured that can take one. Without
    # --add-companions they are only named, as hints.
    discovered, draft_pairs, mmproj_pairs, images = [], {}, {}, []
    if dirs:
        specs = [DiscoverSpec(dir=d, recursive=recursive) for d in dirs]
        stats = {}
        discovered += discovery.scan_dirs(
            specs, dirs,
            known_ids=set(cfg.models) | set(cfg.image_models),
            known_paths=known_paths | _image_paths(cfg),
            known_models=known_models, progress=True, stats=stats,
            images=images, encoder_ids=_encoder_ids(known_models))
        draft_pairs = _open_pairs(path, "draft_gguf",
                                  stats.get("draft_pairs"))
        mmproj_pairs = _open_pairs(path, "mmproj", stats.get("mmproj_pairs"))
    hints = {}
    if not a.add_companions:
        hints = _companion_hints(draft_pairs, mmproj_pairs)
        draft_pairs, mmproj_pairs = {}, {}
    if scan_cache:
        known_refs = {mc.path for mc in cfg.models.values()
                      if str(mc.path).startswith("hf:")}
        discovered += discovery.scan_hf_cache(
            known_ids=(set(cfg.models) | set(cfg.image_models)
                       | {m.id for m in discovered}),
            known_refs=known_refs, progress=True)

    print(f"\nsync-models {path}:")
    print(f"  kept:    {len(kept)} model(s)")
    for mid, reason in unverified:
        print(f"  keep:    {mid}  (unverified - {reason})")
    for mid in removed:
        print(f"  remove:  {mid}  ({cfg.models[mid].path} - gone)")
    for m in discovered:
        print(f"  add:     {m.id}  ({discovery._rel(m.path, dirs)})")
    for im in images:
        print(f"  add:     {im.id}  (image, {discovery._rel(im.path, dirs)})")
    for mid, dpath in draft_pairs.items():
        print(f"  update:  {mid}  (draft_gguf: {discovery._rel(dpath, dirs)})")
    for mid, mpath in mmproj_pairs.items():
        print(f"  update:  {mid}  (mmproj: {discovery._rel(mpath, dirs)})")
    for mid in restream:
        print(f"  update:  {mid}  (stream: experts dropped, the model fits "
              "the memory budget now)")
    for (mid, what), cpath in hints.items():
        print(f"  hint:    {mid}  ({what} {discovery._rel(cpath, dirs)} pairs "
              "with it; --add-companions adds it)")
    if (not removed and not discovered and not draft_pairs and not images
            and not restream and not mmproj_pairs):
        print("  (already in sync)")
        return 0
    if a.dry_run:
        print("\n(dry run - no changes written)")
        return 0
    # The scan writes nothing, so a config that gmlx cannot write is refused
    # only when it needs a change.
    try:
        config_write_target(path)
    except ConfigWriteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    new_roots = ([d for d in dirs if d not in cfg.model_dirs]
                 if a.models_dir else [])
    try:
        _apply_sync(path, removed, discovered, dirs, new_roots=new_roots,
                    draft_pairs=draft_pairs, images=images,
                    restream=restream, mmproj_pairs=mmproj_pairs)
    except ConfigWriteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    n_updated = len(draft_pairs) + len(restream) + len(mmproj_pairs)
    changed = (f"\nupdated {path} (+{len(discovered) + len(images)} / "
               f"-{len(removed)}"
               + (f" / ~{n_updated}" if n_updated else "") + ")")
    print(changed)
    _reload_running(path, skip=a.no_reload)
    return 0


def _pinned_paths(resolved: dict, images: dict, cli_pins) -> tuple:
    """The pool's pinned paths, from `pin: true` on chat and image models and
    from --pin, and the pinned image models as {id: path}."""
    paths = {rm.path for rm in resolved.values() if rm.pin}
    image_pins = {iid: rm.path for iid, rm in images.items() if rm.pin}
    paths |= set(image_pins.values())
    for p in cli_pins:
        rp = resolved.get(p)
        paths.add(rp.path if rp is not None
                  else os.path.abspath(os.path.expanduser(p)))
    return paths, image_pins


def _image_paths(cfg) -> set:
    """The resolved DiT paths of the configured image models."""
    out = set()
    for m in cfg.image_models.values():
        try:
            rp = resolve_path(m.path, cfg.model_dirs)
        except ConfigError:
            continue
        if rp:
            out.add(rp)
    return out


def _encoder_ids(known_models: dict) -> dict:
    """Resolved path -> id of the configured chat models an image entry
    may share as its encoder: those without an adapter."""
    return {p: mc.id for p, mc in known_models.items() if not mc.adapter}


def _image_entry(im, dirs) -> dict:
    """The ``image_models:`` entry of a paired image model."""
    entry = {"path": discovery._rel(im.path, dirs),
             "encoder": (im.encoder if not os.path.isabs(im.encoder)
                         else discovery._rel(im.encoder, dirs))}
    if im.encoder_mmproj:
        entry["encoder_mmproj"] = discovery._rel(im.encoder_mmproj, dirs)
    entry["vae"] = discovery._rel(im.vae, dirs)
    return entry


def _hf_cache_readable() -> bool:
    """Whether the local Hugging Face cache can be scanned at all. False means
    hf: entries cannot be VERIFIED - which must never read as 'gone'."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return os.path.isdir(HF_HUB_CACHE)
    except Exception:
        return False


def stale_stream_marks(path, path_of: dict) -> dict:
    """The entries of the config at ``path`` whose ``stream:`` key
    discovery wrote and a fresh scan no longer gives: id -> the scan's
    ``stream`` and ``speculative`` values. ``path_of`` maps an id to its
    resolved model file. Sync drops these keys."""
    out = {}
    for mid in _marked_stream_ids(path):
        rp = path_of.get(mid)
        fresh = discovery.rescan_entry(rp) if rp else None
        if fresh is not None and fresh["stream"] is None:
            out[mid] = fresh
    return out


def _warn_stale_stream_marks(cfg, path) -> None:
    """At the start, a warning for each entry of the config at ``path``
    whose ``stream: experts`` key a folder scan wrote and a fresh scan no
    longer gives (``stale_stream_marks``): the key streams the experts of
    a model that now fits, until ``gmlx sync-models`` drops it. Reads the
    headers of the marked entries only. A read that fails says nothing."""
    if not path or not getattr(cfg, "models", None):
        return
    path_of = {}
    for mid, m in cfg.models.items():
        try:
            path_of[mid] = resolve_path(m.path, cfg.model_dirs)
        except Exception:                              # noqa: BLE001 - advisory
            continue
    try:
        stale = sorted(stale_stream_marks(path, path_of))
    except Exception:                                  # noqa: BLE001 - advisory
        return
    from .capacity import gathered_table_bytes

    for mid in stale:
        fits = ("fits with its table read from the file"
                if gathered_table_bytes(path_of[mid]) else "fits in memory")
        print(f"[server] warning: {mid}: a folder scan set stream: experts, "
              f"and the model now {fits}, so its experts stream. Run gmlx "
              "sync-models to keep them in memory.", file=sys.stderr)


def _marked_key(node, key, note: str) -> bool:
    """Whether ``key`` of the ruamel map ``node`` carries ``note`` in its
    end-of-line comment: the mark that tells a key gmlx wrote from one
    the user wrote."""
    ca = getattr(node, "ca", None)
    item = ca.items.get(key) if ca is not None else None
    tok = item[2] if item and len(item) > 2 else None
    return tok is not None and note in str(tok.value)


def _marked_stream_ids(path) -> list:
    """The ids of ``path``'s model entries whose ``stream:`` key carries
    discovery's mark comment."""
    from ruamel.yaml import YAML

    try:
        doc = YAML().load(Path(path).read_text())
    except Exception:
        return []
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        return []
    return [str(mid) for mid, ent in models.items()
            if _marked_key(ent, "stream", discovery.STREAM_MARK_NOTE)
            and ent.get("stream") == "experts"]


def _marked_alias_names(path) -> set:
    """The names in ``path``'s ``aliases:`` map that ``gmlx pull`` wrote,
    by the mark comment on each."""
    from ruamel.yaml import YAML

    try:
        doc = YAML().load(Path(path).read_text())
    except Exception:
        return set()
    aliases = doc.get("aliases") if isinstance(doc, dict) else None
    if not isinstance(aliases, dict):
        return set()
    return {str(name) for name in aliases
            if _marked_key(aliases, name, discovery.PULL_ALIAS_NOTE)}


def _resolved_drafter(mc, model_dirs):
    """A copy of ``mc`` whose ``draft_gguf`` is the absolute path the scan
    names the file by, so a pairing that is in place already reads as one.
    A value that does not resolve stays as written."""
    if not mc.draft_gguf:
        return mc
    try:
        rp = resolve_path(mc.draft_gguf, model_dirs)
    except ConfigError:
        return mc
    return dataclasses.replace(mc, draft_gguf=rp)


def _companion_hints(draft_pairs, mmproj_pairs) -> dict:
    """{(id, "drafter" or "mmproj"): path} for the companions a sync leaves
    out without --add-companions."""
    hints = {(mid, "mmproj"): p for mid, p in (mmproj_pairs or {}).items()}
    hints.update({(mid, "drafter"): p for mid, p in (draft_pairs or {}).items()})
    return dict(sorted(hints.items()))


def _open_pairs(path, key, pairs) -> dict:
    """The ``pairs`` (id -> companion path) whose entry in ``path`` can take
    ``key``. A key already there is the user's, and a commented-out line
    such as ``#mmproj:`` keeps the entry without one on purpose. A drafter
    also skips an entry set to ``speculative: false``, which it would turn
    on."""
    hold = _ids_with_key(path, key)
    if key == "draft_gguf":
        hold |= _ids_speculative_off(path)
    return {mid: p for mid, p in (pairs or {}).items() if mid not in hold}


def _ids_speculative_off(path) -> set:
    """The ids of ``path``'s model entries set to ``speculative: false``,
    in any spelling the loader reads (``speculative_off``)."""
    from ruamel.yaml import YAML

    try:
        doc = YAML().load(Path(path).read_text())
    except Exception:
        return set()
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        return set()
    return {str(mid) for mid, ent in models.items()
            if isinstance(ent, dict) and speculative_off(ent.get("speculative"))}


def _ids_with_key(path, key) -> set:
    """The ids of ``path``'s model entries that carry ``key``, live or as a
    commented-out line below the entry's key: ``#mmproj: ...`` indented
    past the entry's column, or ``#    mmproj: ...`` with the ``#`` at
    column 0, as a comment toggle writes it."""
    import re

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    try:
        text = Path(path).read_text()
        doc = YAML().load(text)
    except Exception:
        return set()
    models = doc.get("models") if isinstance(doc, CommentedMap) else None
    if not isinstance(doc, CommentedMap) or not isinstance(
            models, CommentedMap) or not models:
        return set()
    out = {str(mid) for mid, ent in models.items()
           if isinstance(ent, dict) and key in ent}
    lines = text.splitlines()
    # ruamel files a comment line under whichever key precedes it, so the
    # entry's span in the text decides.
    starts: list[tuple[int, int, str]] = sorted(
        (int(models.lc.key(mid)[0]), int(models.lc.key(mid)[1]), str(mid))
        for mid in models)
    block = doc.lc.key("models")[0]
    after = [doc.lc.key(k)[0] for k in doc if doc.lc.key(k)[0] > block]
    stop = min(after) if after else len(lines)
    pat = re.compile(r"^(\s*)#(\s*)(.*)$")
    is_key = re.compile(rf"{re.escape(key)}\s*:")
    for i, (line, col, mid) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else stop
        for raw in lines[line + 1:end]:
            m = pat.match(raw)
            if not m:
                continue
            lead = len(m.group(1))
            # Where the text stood before a `#` went in front of it.
            text = len(m.group(2))
            if lead == 0 and text <= col:
                break          # a commented-out entry of its own starts here
            # A `#` at the entry's column is a note between entries, such
            # as the hint lines `gmlx init` writes.
            if is_key.match(m.group(3)) and (
                    lead > col or (lead == 0 and text > col)):
                out.add(mid)
                break
    return out


def _drop_key_keep_trailing(ent, key) -> None:
    """Delete ``key`` from a ruamel map. Its end-of-line comment token also
    holds the lines after it (a blank line, the next entry's comment), so
    those move onto the key before it."""
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    i = list(ent).index(key)
    item = ent.ca.items.pop(key, None)
    del ent[key]
    tok = item[2] if item and len(item) > 2 else None
    rest = tok.value.split("\n", 1)[1] if tok is not None and "\n" in tok.value else ""
    if not rest or i == 0:
        return
    prev = ent.ca.items.setdefault(list(ent)[i - 1], [None, None, None, None])
    if prev[2] is not None:
        prev[2].value += rest
    else:
        prev[2] = CommentToken("\n" + rest, CommentMark(0), None)


def _insert_key_keep_trailing(ent, after, key, value,
                              note: str | None = None) -> None:
    """Insert ``key: value  # note`` after ``after`` in a ruamel map, with
    no comment when ``note`` is None. The lines that rode ``after``'s
    end-of-line comment token (a blank line, the next entry's comment)
    move onto the new key, so they stay below it. A flow-style map
    (``{path: ..., pin: true}``) gets no comment: it would land inside the
    braces."""
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    item = ent.ca.items.get(after)
    tok = item[2] if item and len(item) > 2 else None
    rest = ""
    if tok is not None and "\n" in tok.value:
        head, rest = tok.value.split("\n", 1)
        if rest:
            if head.strip():
                tok.value = head + "\n"
            else:
                item[2] = None
    ent.insert(list(ent).index(after) + 1, key, value)
    if note is not None and not ent.fa.flow_style():
        try:
            column = ent.lc.key(after)[1] + len(f"{key}: {value}") + 2
        except Exception:  # noqa: BLE001 - a map built in memory has no positions
            column = None
        ent.yaml_add_eol_comment(note, key, column=column)
    if rest:
        new = ent.ca.items.setdefault(key, [None, None, None, None])
        if new[2] is not None:
            new[2].value = new[2].value.rstrip("\n") + "\n" + rest
        else:
            new[2] = CommentToken("\n" + rest, CommentMark(0), None)


# The comment lines above an entry of ``models:``, and the lines after the
# block, live in one of three places in the ruamel tree: the map's own head
# (above the first key), the end-of-line token of the previous entry's last
# line, or the key's before-comment list. These helpers read them as text
# lines, each with its indentation and newline, and write them back to one
# place: the head for the first entry, the key's list for the others.

def _tail_slot(models, key):
    """``(ca.items, key, index)`` of the comment slot after the last line
    of entry ``key``: the last key's end-of-line token, nested maps and
    lists followed down to their last item."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    holder, k, node = models, key, models[key]
    while (isinstance(node, (CommentedMap, CommentedSeq)) and len(node)
           and not node.fa.flow_style()):
        holder = node
        k = list(node)[-1] if isinstance(node, CommentedMap) else len(node) - 1
        node = node[k]
    return holder.ca.items, k, 0 if isinstance(holder, CommentedSeq) else 2


def _token_lines(tokens, skip_first=False) -> list[str]:
    """The text lines of comment tokens. ``skip_first`` drops what the
    first line of an end-of-line token holds, the comment on the line."""
    lines = []
    for n, tok in enumerate(tokens):
        if tok is None:
            continue
        text = tok.value
        if n == 0 and skip_first:
            text = text.split("\n", 1)[1] if "\n" in text else ""
        elif text.strip():
            text = " " * tok.column + text
        lines += text.splitlines(keepends=True)
    return lines


def _comment_tokens(lines) -> list:
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    out = []
    for line in lines:
        body = line.lstrip(" ")
        col = len(line) - len(body) if body.strip() else 0
        out.append(CommentToken(body if body.strip() else "\n",
                                CommentMark(col), None))
    return out


def _take_rest(models, key) -> list[str]:
    """Remove and return the lines after entry ``key``'s last line."""
    items, k, i = _tail_slot(models, key)
    row = items.get(k)
    tok = row[i] if row and len(row) > i else None
    if tok is None:
        return []
    lines = _token_lines([tok], skip_first=True)
    head = tok.value.split("\n", 1)[0]
    if head.strip():
        tok.value = head + "\n"
    else:
        row[i] = None
    return lines


def _set_rest(models, key, lines) -> None:
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    _take_rest(models, key)
    if not lines:
        return
    items, k, i = _tail_slot(models, key)
    row = items.setdefault(k, [None, None, None, None])
    if row[i] is not None:
        row[i].value = row[i].value.rstrip("\n") + "\n" + "".join(lines)
    else:
        row[i] = CommentToken("\n" + "".join(lines), CommentMark(0), None)


def _head_slots(doc, models):
    """The row and indexes of the ``models:`` line's end-of-line token and
    of the comment list above the first key. The dump reads the row that
    ``doc`` keeps for the ``models`` key when there is one."""
    row = doc.ca.items.get("models")
    if row is not None:
        return row, 2, 3
    if models.ca.comment is None:
        models.ca.comment = [None, None]
    return models.ca.comment, 0, 1


def _take_head(doc, models) -> list[str]:
    """Remove and return the comment lines above the first key of
    ``models``, below the ``models:`` line."""
    row, e, h = _head_slots(doc, models)
    eol = row[e]
    lines = _token_lines([eol], skip_first=True) + _token_lines(row[h] or [])
    if eol is not None:
        head = eol.value.split("\n", 1)[0]
        row[e] = eol if head.strip() else None
        if head.strip():
            eol.value = head + "\n"
    _set_head(doc, models, [])
    return lines


def _set_head(doc, models, lines) -> None:
    row, e, h = _head_slots(doc, models)
    row[h] = _comment_tokens(lines) or None
    if row is not models.ca.comment:
        models.ca.comment = [row[e], row[h]]


def _take_before(models, key) -> list[str]:
    row = models.ca.items.get(key)
    if not row or not row[1]:
        return []
    lines = _token_lines(row[1])
    row[1] = None
    return lines


def _take_above(doc, models, i) -> list[str]:
    """Remove and return the comment lines above entry ``i``."""
    keys = list(models)
    lines = (_take_head(doc, models) if i == 0
             else _take_rest(models, keys[i - 1]))
    return lines + _take_before(models, keys[i])


def _put_above(doc, models, i, lines) -> None:
    keys = list(models)
    if i == 0:
        _set_head(doc, models, lines)
        return
    row = models.ca.items.setdefault(keys[i], [None, None, None, None])
    row[1] = _comment_tokens(lines) or None


def _joined(a, b) -> list[str]:
    """``a`` then ``b``, one blank line where both meet with one."""
    if a and b and not a[-1].strip() and not b[0].strip():
        b = b[1:]
    return a + b


def _drop_entry(doc, models, key) -> None:
    """Delete entry ``key``. The sampling comment discovery wrote above it
    goes with it; the user's lines above it and the lines above the next
    entry stay where they were."""
    keys = list(models)
    i = keys.index(key)
    above = [line for line in _take_above(doc, models, i)
             if not discovery.is_family_comment(line)]
    after = _take_rest(models, key)
    if i + 1 < len(keys):
        after += _take_before(models, keys[i + 1])
    del models[key]
    models.ca.items.pop(key, None)
    lines = _joined(above, after)
    if i + 1 < len(keys):
        _put_above(doc, models, i, lines)
    elif i > 0:
        _set_rest(models, keys[i - 1], lines)
    else:
        _set_head(doc, models, lines)


def _place_entry(doc, models, pos, key, entry, note) -> None:
    """Insert the new entry ``key`` at ``pos`` with the sampling comment
    ``note`` directly above it. The comment that stood above the entry now
    after it moves along with that entry."""
    mine = [f"  # {note}\n"] if note else []
    keys = list(models)
    if pos >= len(keys):
        trailing = _take_rest(models, keys[-1]) if keys else []
        models.insert(pos, key, entry)
        if keys:
            _put_above(doc, models, pos, mine)
            _set_rest(models, key, trailing)
        else:
            _set_head(doc, models, _take_head(doc, models) + mine)
        return
    above = _take_above(doc, models, pos)
    stay = 0
    if pos == 0:
        # The head holds the block's own comment, then the first entry's:
        # the lines after its last blank line, or with no blank line, the
        # lines from its first sampling comment on.
        blank = [n + 1 for n, line in enumerate(above) if not line.strip()]
        if blank:
            stay = blank[-1]
        else:
            stay = next((n for n, line in enumerate(above)
                         if discovery.is_family_comment(line)), len(above))
    models.insert(pos, key, entry)
    _put_above(doc, models, pos, above[:stay] + mine)
    _put_above(doc, models, pos + 1, above[stay:])


def _open_models_block(doc):
    """Make a null or missing ``models:`` block an empty map and return it.
    The comment lines indented under the null value become the map's head.
    The lines from the first one at column 0 on stay above the next key,
    or at the end of the file."""
    from ruamel.yaml.comments import CommentedMap

    keys = list(doc)
    row = doc.ca.items.get("models")
    lines = []
    if row is not None and row[2] is not None:
        eol = row[2]
        lines = _token_lines([eol], skip_first=True)
        first = eol.value.split("\n", 1)[0]
        if first.strip():
            eol.value = first + "\n"
        else:
            row[2] = None
    i = keys.index("models") + 1 if "models" in keys else len(keys)
    nxt = keys[i] if i < len(keys) else None
    if nxt is not None:
        lines += _take_before(doc, nxt)
    elif "models" in keys and doc.ca.end:
        lines += _token_lines(doc.ca.end)
        doc.ca.end = []
    cut = next((n for n, line in enumerate(lines)
                if line.strip() and not line.startswith(" ")), len(lines))
    head, rest = lines[:cut], lines[cut:]
    gap = bool(head) and not head[-1].strip()
    while head and not head[-1].strip():
        rest.insert(0, head.pop())
    if head and gap:
        head.append("\n")
    if rest and nxt is not None:
        doc.ca.items.setdefault(nxt, [None, None, None, None])[1] = (
            _comment_tokens(rest))
    elif rest:
        doc.ca.end = _comment_tokens(rest)
    models = CommentedMap()
    doc["models"] = models
    _set_head(doc, models, head)
    return models


def _empty_models_block(doc, models) -> None:
    """Write an emptied ``models:`` block that has comment lines as a null
    value under them. The dump would put the lines above the ``{}`` it
    writes for an empty map, where the file no longer parses."""
    from ruamel.yaml.error import CommentMark
    from ruamel.yaml.tokens import CommentToken

    lines = _take_head(doc, models)
    if not lines:
        return
    doc["models"] = None
    row = doc.ca.items.setdefault("models", [None, None, None, None])
    if row[2] is not None:
        row[2].value = row[2].value.rstrip("\n") + "\n" + "".join(lines)
    else:
        row[2] = CommentToken("\n" + "".join(lines), CommentMark(0), None)


def _apply_sync(path, removed, discovered, dirs, new_roots=(),
                draft_pairs=None, images=(), restream=None,
                mmproj_pairs=None,
                companion_note: str | None = discovery.COMPANION_MARK_NOTE,
                aliases=None) -> None:
    """Rewrite ``path``'s ``models:`` block in place, preserving comments and
    hand-edits (ruamel round-trip): delete ``removed`` ids, splice in the newly
    ``discovered`` models. Untouched entries keep their exact formatting.
    ``new_roots`` are --models-dir override roots absent from
    ``server.model_dirs``; they're appended so the new entries resolve.
    ``images`` (paired :class:`ImageModelCfg`) go under ``image_models:``.
    ``restream`` (id -> a fresh scan's ``stream`` and ``speculative``)
    drops the ``stream:`` key discovery wrote, and turns speculative
    decoding back on where the fresh scan would. ``draft_pairs`` and
    ``mmproj_pairs`` (configured id -> companion path) add a ``draft_gguf``
    or ``mmproj`` key to an entry that has none, with the end-of-line
    comment ``companion_note``; a drafter skips an entry set to
    ``speculative: false``. ``aliases`` (``{name: id}``) land in the
    top-level ``aliases:`` map, made after ``models:`` when there is none,
    each with :data:`discovery.PULL_ALIAS_NOTE` as its end-of-line comment.
    An alias whose target is a removed id goes with it, with a
    ``remove:`` line."""
    from ruamel.yaml.comments import CommentedMap

    def edit_models(doc, models):
        for mid in removed:
            if mid in models:
                _drop_entry(doc, models, mid)
        pos = 0
        for mid, fresh in (restream or {}).items():
            ent = models.get(mid)
            if not isinstance(ent, dict) or ent.get("stream") != "experts":
                continue
            if (fresh["speculative"] and "speculative" not in ent
                    and not ent.get("draft_gguf")):
                ent.insert(list(ent).index("stream"), "speculative", "auto")
            _drop_key_keep_trailing(ent, "stream")
        for mc in discovered:
            entry = CommentedMap(discovery.model_to_entry(mc, dirs))
            if "stream" in entry:
                entry.yaml_add_eol_comment(discovery.STREAM_MARK_NOTE, "stream")
            # The scan skips configured ids, so each one is new: it goes at
            # the top of the block, below its head comment.
            _place_entry(doc, models, pos, mc.id, entry,
                         discovery.family_comment(mc))
            pos += 1
        # A companion goes after `path` (a drafter after an `mmproj`), as
        # discovery writes an entry: the last key's comment token holds the
        # next entry's comment, so a key appended there would land below it.
        for mid, dpath in (draft_pairs or {}).items():
            ent = models.get(mid)
            if (isinstance(ent, dict) and "path" in ent
                    and "draft_gguf" not in ent
                    and not speculative_off(ent.get("speculative"))):
                _insert_key_keep_trailing(
                    ent, "mmproj" if "mmproj" in ent else "path",
                    "draft_gguf", discovery._rel(dpath, dirs), companion_note)
                # As a scan writes a paired drafter: a load leaves it out
                # when it does not fit beside the experts.
                if "speculative" not in ent:
                    _insert_key_keep_trailing(ent, "draft_gguf",
                                              "speculative", "auto")
        for mid, mpath in (mmproj_pairs or {}).items():
            ent = models.get(mid)
            if (isinstance(ent, dict) and "path" in ent
                    and "mmproj" not in ent):
                _insert_key_keep_trailing(ent, "path", "mmproj",
                                          discovery._rel(mpath, dirs),
                                          companion_note)
        if not models:
            _empty_models_block(doc, models)

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
        # A null or missing block stays as it is unless the sync adds to it.
        models = doc.get("models")
        if discovered:
            discovery.drop_no_models_comment(doc)
            if not isinstance(models, dict):
                models = _open_models_block(doc)
        if isinstance(models, dict):
            edit_models(doc, models)
        block = doc.get("aliases")
        if removed and isinstance(block, dict):
            for name in [n for n in block
                         if str(block[n]).split("@", 1)[0] in removed]:
                print(f"  remove:  alias {name} -> {block[name]}")
                del block[name]
                block.ca.items.pop(name, None)
        if aliases:
            if not isinstance(block, dict):
                block = CommentedMap()
                keys = list(doc)
                at = keys.index("models") + 1 if "models" in keys else len(keys)
                doc.insert(at, "aliases", block)
                doc.yaml_set_comment_before_after_key("aliases", before="\n")
            for name, target in aliases.items():
                # A fresh mark at the new value's column; the lines that
                # rode the old comment token stay below the key.
                item = block.ca.items.pop(name, None)
                tok = item[2] if item and len(item) > 2 else None
                rest = (tok.value.split("\n", 1)[1]
                        if tok is not None and "\n" in tok.value else "")
                block[name] = target
                block.yaml_add_eol_comment(discovery.PULL_ALIAS_NOTE, name,
                                           column=2 + len(f"{name}: {target}") + 2)
                new = block.ca.items[name]
                if item and len(item) > 1:
                    new[1] = item[1]
                if rest:
                    new[2].value = new[2].value.rstrip("\n") + "\n" + rest
        if images:
            block = doc.get("image_models")
            if not isinstance(block, dict):
                block = CommentedMap()
                doc["image_models"] = block
            for im in images:
                block[im.id] = CommentedMap(_image_entry(im, dirs))
        # Cache-resident entries need server.hf_cache to resolve from the cache.
        if any(str(mc.path).startswith("hf:") for mc in discovered):
            srv = doc.get("server")
            if isinstance(srv, dict) and not srv.get("hf_cache"):
                srv["hf_cache"] = True

    edit_config_yaml(path, mutate)


@dataclasses.dataclass
class Registered:
    """What :func:`register_downloads` did for a pull. ``ids`` maps each
    downloaded model path to its configured id, new or already there.
    ``alias`` is the catalog name ``gmlx chat`` can use, None when none
    was written or kept. ``moved_from`` is ``(id, path)`` of the model
    the alias named before this pull repointed it."""
    ids: dict = dataclasses.field(default_factory=dict)
    alias: str | None = None
    moved_from: tuple | None = None


# Why a drafter stays off an entry that streams its experts.
_STREAMS_REASON = "it runs from the SSD (stream: experts)"


def _names_file(value, path, model_dirs) -> bool:
    """Whether a config entry's companion ``value`` resolves to ``path``."""
    if not value:
        return False
    try:
        return os.path.abspath(resolve_path(value, model_dirs)) == os.path.abspath(path)
    except ConfigError:
        return False


def _alias_plan(cfg, path, pairs, ids) -> tuple:
    """The alias a catalog pull asks for, as ``(to_write, alias,
    moved_from)``. ``to_write`` is ``{name: id}`` for the config, empty
    when the alias is in place already or is held back; ``alias`` is the
    name ``gmlx chat`` can use after the write, None when it is held
    back; ``moved_from`` is ``(id, path)`` of the model a repointed alias
    named before. A name that is a model, image or assistant id, or an
    alias without the pull mark, is the user's: the config would not
    load with a second meaning for it, so the pull says which id runs
    the model instead."""
    for ap, pr in pairs.items():
        name = pr.get("alias")
        if not name or ap not in ids:
            continue
        target = ids[ap]
        if name == target:
            return {}, name, None
        taken = next((what for what, keys in (
            ("a model", cfg.models), ("an image model", cfg.image_models),
            ("a served assistant", cfg.assistants)) if name in keys), None)
        if taken:
            print(f"{name} is already {taken} in {path}; run it as "
                  f"gmlx chat {target}")
            return {}, None, None
        old = cfg.aliases.get(name)
        if old is None:
            return {name: target}, name, None
        old_id = old.split("@", 1)[0]
        if old_id == target:
            return {}, name, None
        if name not in _marked_alias_names(path):
            print(f"{name} is already an alias in {path}; run it as "
                  f"gmlx chat {target}")
            return {}, None, None
        moved = None
        if old_id in cfg.models:
            try:
                moved = (old_id, resolve_path(cfg.models[old_id].path,
                                              cfg.model_dirs))
            except ConfigError:
                moved = (old_id, cfg.models[old_id].path)
        return {name: target}, name, moved
    return {}, None, None


def register_downloads(paths: list, config_path=None, *,
                       pairs=None) -> Registered:
    """Fold freshly downloaded GGUF files into the server config - ``gmlx
    pull``'s auto-register, a mini sync-models scoped to exactly these paths.

    Quiet no-op when there is no config, when a file landed outside every
    ``model_dirs`` root (an explicit ``--to`` elsewhere - the server could
    never discover it), or when the file is already configured. Otherwise the
    same machinery as sync-models end to end: id derivation, mmproj pairing,
    drafter pairing, speculative detection, comment-preserving splice, and a
    SIGHUP so a running server serves the new entries immediately.

    ``pairs`` maps a downloaded model path to the companions a catalog pull
    knows go with it, ``{"mmproj": path, "draft_gguf": path, "alias":
    name}``, each key optional. A new entry takes them as they are, except
    a drafter on an entry that streams. An entry already in the config
    takes a companion only where its key is open (see :func:`_open_pairs`)
    and it does not stream; a held-back one gets a line that says why. The
    alias lands in the top-level ``aliases:`` map with the pull mark; see
    :func:`_alias_plan` for the names it leaves alone. Every line a
    catalog pull prints says "faster replies" and "image input", never the
    file kinds.

    Best-effort by contract: the download already succeeded, so a
    registration problem warns and returns an empty result instead of
    failing ``pull``."""
    done = Registered()
    try:
        if config_path:
            path = Path(os.path.expanduser(str(config_path)))
        else:
            path = next((p for p in default_config_paths() if p.exists()), None)
        if path is None or not path.exists():
            return done
        cfg = load_config(path)
        roots = [os.path.abspath(os.path.expanduser(os.path.expandvars(d)))
                 for d in cfg.model_dirs]
        wanted = set()
        for p in paths:
            ap = os.path.abspath(str(p))
            if any(ap == r or ap.startswith(r + os.sep) for r in roots):
                wanted.add(ap)
        if not wanted:
            return done
        pairs = {os.path.abspath(str(p)): dict(v or {})
                 for p, v in (pairs or {}).items()}
        known_paths, known_models, id_of = set(), {}, {}
        for mid, mc in cfg.models.items():
            try:
                rp = resolve_path(mc.path, cfg.model_dirs)
            except ConfigError:
                continue
            known_paths.add(rp)
            if rp:
                known_models[rp] = _resolved_drafter(mc, cfg.model_dirs)
                id_of[os.path.abspath(rp)] = mid
        # Scan the parent dirs (mmproj pairing needs the siblings), then keep
        # only models whose file is one we just downloaded - a neighbouring
        # file the user left unregistered stays unregistered. The drafter a
        # catalog pull names is configured below, so the scan says nothing
        # about configuring it.
        parents = sorted({os.path.dirname(p) for p in wanted})
        specs = [DiscoverSpec(dir=d, recursive=False) for d in parents]
        stats = {}
        images: list = []
        named = {v["draft_gguf"] for v in pairs.values() if v.get("draft_gguf")}
        found = discovery.scan_dirs(specs, cfg.model_dirs,
                                    known_ids=set(cfg.models) | set(cfg.image_models),
                                    known_paths=known_paths | _image_paths(cfg),
                                    known_models=known_models, stats=stats,
                                    images=images,
                                    encoder_ids=_encoder_ids(known_models),
                                    named_drafters=named)
        newly = [m for m in found if os.path.abspath(m.path) in wanted]
        images = [im for im in images if os.path.abspath(im.path) in wanted]
        # The companions a catalog pull names replace the scan's on a new
        # entry.
        for m in newly:
            pr = pairs.get(os.path.abspath(m.path), {})
            if pr.get("mmproj"):
                m.mmproj = pr["mmproj"]
            if pr.get("draft_gguf"):
                if m.stream:
                    print(f"{os.path.basename(pr['draft_gguf'])} is not set for "
                          f"faster replies on {m.id}: {_STREAMS_REASON}")
                else:
                    m.draft_gguf = pr["draft_gguf"]
                    m.speculative = True
        # Only a drafter from this download may change a configured entry: a
        # file the user left beside a model stays unregistered.
        pulled = {mid: dp for mid, dp in (stats.get("draft_pairs") or {}).items()
                  if os.path.abspath(dp) in wanted}
        asked_mm: dict = {}
        for ap, pr in pairs.items():
            mid = id_of.get(ap)
            if mid is None:
                continue
            # A companion the entry names already is in place: quiet.
            cur = cfg.models[mid]
            if pr.get("draft_gguf") and not _names_file(
                    cur.draft_gguf, pr["draft_gguf"], cfg.model_dirs):
                pulled[mid] = pr["draft_gguf"]
            if pr.get("mmproj") and not _names_file(
                    cur.mmproj, pr["mmproj"], cfg.model_dirs):
                asked_mm[mid] = pr["mmproj"]
        streams = {mid for mid, mc in cfg.models.items() if mc.stream}
        drafts = {mid: dp for mid, dp in _open_pairs(path, "draft_gguf", pulled).items()
                  if mid not in streams}
        for mid, dp in pulled.items():
            if mid not in drafts:
                why = (_STREAMS_REASON if mid in streams else
                       f"its entry in {path} has a draft_gguf key or "
                       "speculative: false")
                print(f"{os.path.basename(dp)} is not set for faster replies "
                      f"on {mid}: {why}")
        mmprojs = _open_pairs(path, "mmproj", asked_mm)
        for mid, mp in asked_mm.items():
            if mid not in mmprojs:
                print(f"{os.path.basename(mp)} is not set for image input on "
                      f"{mid}: its entry in {path} has an mmproj key")
        # The companions beside a configured entry that a pull leaves out:
        # a drafter it did not download, and an mmproj.
        beside = {mid: dp for mid, dp in (stats.get("draft_pairs") or {}).items()
                  if mid not in pulled}
        scan_mm = {mid: mp for mid, mp in (stats.get("mmproj_pairs") or {}).items()
                   if mid not in asked_mm}
        hints = _companion_hints(_open_pairs(path, "draft_gguf", beside),
                                 _open_pairs(path, "mmproj", scan_mm))
        for (mid, what), cp in hints.items():
            gives = "faster replies" if what == "drafter" else "image input"
            print(f"{os.path.basename(cp)} can give {mid} {gives}: "
                  "gmlx sync-models --add-companions adds it")
        ids = {ap: id_of[ap] for ap in wanted if ap in id_of}
        ids.update({os.path.abspath(m.path): m.id for m in newly})
        to_write, alias, moved = _alias_plan(cfg, path, pairs, ids)
        if not newly and not drafts and not mmprojs and not images \
                and not to_write:
            return Registered(ids, alias)   # already configured (a re-pull) - quiet
        _apply_sync(path, [], newly, cfg.model_dirs, draft_pairs=drafts,
                    mmproj_pairs=mmprojs, images=images, companion_note=None,
                    aliases=to_write)
        for mid, dp in drafts.items():
            print(f"registered {os.path.basename(dp)} for faster replies on "
                  f"{mid} in {path}")
        for mid, mp in mmprojs.items():
            print(f"registered {os.path.basename(mp)} for image input on "
                  f"{mid} in {path}")
        for m in newly:
            extras = [w for w, on in (("images", m.mmproj),
                                      ("faster replies", m.speculative)) if on]
            note = f"  ({', '.join(extras)})" if extras else ""
            print(f"registered {m.id} in {path}{note}")
        for im in images:
            print(f"registered {im.id} in {path}  (image)")
        _reload_running(path, skip=False)
        return Registered(ids, alias, moved)
    except ConfigWriteError as e:
        # sync-models would meet the same refusal, so the message gives its own step.
        print(f"warning: could not register the download(s) in the server "
              f"config: {e}", file=sys.stderr)
    except Exception as e:             # noqa: BLE001 - never fail a good pull
        print(f"warning: could not register the download(s) in the server "
              f"config: {e}; run `gmlx sync-models`", file=sys.stderr)
    return done


# serve
# (argparse dest, sampling key) for the positional-model sampling flags: the
# server's SAMPLING_KEYS minus the three serve covers elsewhere (max_tokens is
# the server-wide --max-tokens, thinking_budget has its own flag, and
# enable_thinking is --thinking). Named as `gmlx run` names them.
_SAMPLING_FLAGS = (
    ("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
    ("min_p", "min_p"), ("seed", "seed"),
    ("repetition_penalty", "repetition_penalty"),
    ("repetition_context_size", "repetition_context_size"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"), ("stop", "stop"),
    ("xtc_probability", "xtc_probability"), ("xtc_threshold", "xtc_threshold"),
    ("thinking_start_token", "thinking_start_token"),
    ("thinking_end_token", "thinking_end_token"),
)


def _template_kwargs(raw: str) -> dict:
    """argparse type for --chat-template-config: a JSON object, refused at
    parse time so a typo never reaches a model load. A key that names a
    parameter of the template call is dropped with a line."""
    import json

    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not valid JSON: {e}") from e
    if not isinstance(out, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    from gmlx.config import TEMPLATE_CALL_KEYS, template_call_key_refusal

    # As in a config file, such a key is dropped with a line, so a server
    # that a login item starts with it still starts.
    msg = template_call_key_refusal("--chat-template-config", out)
    if msg:
        print(f"warning: {msg}. The server ignores it.", file=sys.stderr)
    return {k: v for k, v in out.items() if k not in TEMPLATE_CALL_KEYS}


def _add_serve_args(ap: argparse.ArgumentParser) -> None:
    from gmlx.commands.cli import mass_share

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
                    help="mmproj GGUF - makes a single positional model a VLM.")
    asr = ap.add_mutually_exclusive_group()
    asr.add_argument("--asr", default=None, metavar="PATH",
                     help="Speech companion GGUF for a single positional model that "
                          "reads audio through one (Granite Switch).")
    asr.add_argument("--no-asr", action="store_true",
                     help="Do not pair a speech companion with the single model.")
    ap.add_argument("--hf-source", default=None,
                    help="Processor/config override for a single VLM model (rarely "
                         "needed; the processor is synthesized from the GGUFs).")
    ap.add_argument("--speculative", action="store_true",
                    help="Serve a single positional model with MTP speculative "
                         "decoding (native-head qwen3.5/3.6; gemma4 also needs "
                         "--draft-gguf).")
    ap.add_argument("--draft-gguf", default=None,
                    help="Companion drafter GGUF for assistant-shape MTP (gemma4, "
                         "DFlash); implies --speculative.")
    ap.add_argument("--native-mtp", action="store_true",
                    help="Draft with the model's own MTP head even when "
                         "--draft-gguf is set. Same as the per-model config key "
                         "native_mtp.")
    ap.add_argument("--stochastic-mtp", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Accept MTP drafts by p/q rejection sampling on sampled "
                         "requests, server-wide (the default): same output "
                         "distribution as non-speculative sampling, not "
                         "token-identical; higher acceptance at temp > 0. "
                         "--no-stochastic-mtp accepts only exact matches. Same "
                         "as config server.stochastic_mtp; greedy requests "
                         "unaffected.")
    ap.add_argument("--gpu-keepwarm", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Hold GPU clocks up while a streamed model is decoding "
                         "(tiny heartbeat kernel; parks when no request is "
                         "decoding, so an idle server pays nothing). Only acts "
                         "on models streaming with a decode feeder, where it is "
                         "on by default; --no-gpu-keepwarm turns it off. Same "
                         "as config server.gpu_keepwarm.")
    ap.add_argument("--draft-block-size", type=int, default=None, metavar="N",
                    help="Speculative block size. Each round drafts N-1 tokens "
                         "and checks them in one N-token target pass, so "
                         "llama-server --spec-draft-n-max M matches N = M+1. "
                         "Raises or lowers the drafter's own default, up to the "
                         "deepest block it can produce. Also via "
                         "GMLX_DRAFT_BLOCK_SIZE.")
    ap.add_argument("--chat-template", default=None, metavar="STR|PATH",
                    help="Inline Jinja template, or a path to a .jinja/.txt file, "
                         "replacing a single positional model's GGUF template "
                         "(config mode: set it per profile/model instead).")
    ap.add_argument("--thinking", choices=("on", "off", "adaptive"), default=None,
                    help="Reasoning switch for a single positional model, mapped "
                         "onto the variable its chat template reads (config mode: "
                         "the per-profile/model `thinking:` key). Default: the "
                         "template's own default.")
    ap.add_argument("--thinking-budget", type=int, default=None, metavar="N",
                    help="Cap a single positional model's thinking tokens per "
                         "request; 0 closes thinking at once (config mode: the "
                         "sampling `thinking_budget:` key; requests may override).")
    ap.add_argument("--adapter", default=None, metavar="PATH",
                    help="GGUF LoRA adapter applied live over a single positional "
                         "model at load - base stays K-quant, no merge (config mode: "
                         "set `adapter:` per model instead).")
    ap.add_argument("--system-prompt", default=None, metavar="STR",
                    help="System prompt for a single positional model, used when "
                         "a request sends none (config mode: the per-profile/model "
                         "`system:` key).")
    ap.add_argument("--chat-template-config", type=_template_kwargs, default=None,
                    metavar="JSON",
                    help="JSON object of extra chat-template variables for a "
                         "single positional model, passed through verbatim, "
                         "e.g. '{\"enable_thinking\": false}' (config mode: the "
                         "`chat_template_kwargs:` key).")
    from gmlx.commands.cli import add_rope_args

    add_rope_args(
        ap, title="Rope scaling of a single positional model",
        description="llama.cpp's rope flags, each over its GGUF header key. "
        "In config mode they are the model's `rope_scaling:` block; see "
        "docs/config.md.")
    kv = ap.add_argument_group(
        "KV cache of a single positional model",
        "Each flag is a `load:` key in config mode; see docs/config.md.")
    kv.add_argument("--kv-bits", type=int, default=None, metavar="N",
                    help="Quantize the KV cache to N bits: 2, 3, 4, 6 or 8 affine, "
                         "or 2, 3, 4, 5, 6 or 8 under kvarn (`load.kv_bits`).")
    kv.add_argument("--kv-group-size", type=int, default=None, metavar="N",
                    help="Affine KV quantization group size, default 64 "
                         "(`load.kv_group_size`).")
    kv.add_argument("--kv-quant-scheme", choices=("auto", "uniform", "kvarn"),
                    default=None,
                    help="KV cache quantization scheme for --kv-bits: 'auto' "
                         "(default) picks kvarn on recurrent and sliding-window "
                         "models and affine elsewhere, 'uniform' is affine, and "
                         "'kvarn' is variance-normalized, where --kv-bits defaults "
                         "to 6 and the newest --kv-tail-tokens stay fp16 "
                         "(`load.kv_quant_scheme`).")
    kv.add_argument("--kv-tail-tokens", type=int, default=None, metavar="N",
                    help="Under kvarn, the newest N tokens kept fp16, a multiple "
                         "of 128, default 1024 (`load.kv_tail_tokens`).")
    kv.add_argument("--max-kv-size", type=int, default=None, metavar="N",
                    help="Cap the request context budget at N tokens "
                         "(`load.max_kv_size`).")
    kv.add_argument("--quantized-kv-start", type=int, default=None, metavar="N",
                    help="Keep the KV cache fp16 until it holds N tokens, "
                         "default 0; not applied under kvarn or to batches "
                         "(`load.quantized_kv_start`).")
    sg = ap.add_argument_group(
        "Sampling defaults of a single positional model",
        "Used when a request omits the field; a request that sends it wins. "
        "Each is a `sampling:` key in config mode, and `gmlx profiles` prints "
        "the family defaults these sit on top of.")
    sg.add_argument("--profile", default=None, metavar="NAME",
                    help="Sampling profile: a built-in intent (coding, creative, "
                         "instruct, reasoning-low/-medium/-high) resolved for the "
                         "model's family (config mode: the model `profile:` key).")
    sg.add_argument("--temp", type=float, default=None, metavar="T",
                    help="Sampling temperature (`temperature`).")
    sg.add_argument("--top-p", type=float, default=None, metavar="P",
                    help="Nucleus probability; 0 disables the filter (`top_p`).")
    sg.add_argument("--top-k", type=int, default=None, metavar="N",
                    help="Candidate count; 0 disables the filter (`top_k`).")
    sg.add_argument("--min-p", type=float, default=None, metavar="P",
                    help="Minimum probability relative to the best token; 0 "
                         "disables (`min_p`).")
    sg.add_argument("--seed", type=int, default=None, metavar="N",
                    help="Sampling seed for every request that sends none "
                         "(`seed`).")
    sg.add_argument("--repetition-penalty", type=float, default=None, metavar="X",
                    help="Penalty over the last --repetition-context-size tokens "
                         "(`repetition_penalty`).")
    sg.add_argument("--repetition-context-size", type=int, default=None,
                    metavar="N",
                    help="Window for the repetition penalty, default 20 "
                         "(`repetition_context_size`).")
    sg.add_argument("--presence-penalty", type=float, default=None, metavar="X",
                    help="Penalty on any token already generated "
                         "(`presence_penalty`).")
    sg.add_argument("--frequency-penalty", type=float, default=None, metavar="X",
                    help="Penalty scaled by how often a token was generated "
                         "(`frequency_penalty`).")
    sg.add_argument("--stop", action="append", default=None, metavar="STR",
                    help="Stop sequence, repeatable; chat completions only "
                         "(`stop`).")
    sg.add_argument("--xtc-probability", type=float, default=None, metavar="P",
                    help="XTC sampling probability (`xtc_probability`).")
    sg.add_argument("--xtc-threshold", type=float, default=None, metavar="T",
                    help="XTC sampling threshold (`xtc_threshold`).")
    sg.add_argument("--thinking-start-token", default=None, metavar="STR",
                    help="The model's opening reasoning marker when it is not "
                         "<think> (`thinking_start_token`).")
    sg.add_argument("--thinking-end-token", default=None, metavar="STR",
                    help="The model's closing reasoning marker when it is not "
                         "</think> (`thinking_end_token`).")
    sg.add_argument("--reasoning-effort", default=None, metavar="LEVEL",
                    help="Reasoning level for models whose template grades "
                         "thinking, such as low, medium or high (config mode: "
                         "the `reasoning_effort:` key).")
    placement = ap.add_mutually_exclusive_group()
    placement.add_argument("--stream-experts", action="store_true",
                    help="Stream a single positional MoE model's routed-expert "
                         "stacks from disk (unwired, page-cache resident) while "
                         "the every-token layers + KV cache stay on GPU "
                         "(config mode: set `stream: experts` per model).")
    placement.add_argument("--stream-cpu", action="store_true",
                    help="Run a single positional model entirely on the CPU device "
                         "with every weight streamed from the page cache - serves "
                         "models larger than the wired limit "
                         "(config mode: set `stream: cpu` per model).")
    ap.add_argument("--stream-fast-disk", choices=("auto", "on", "off"),
                    default=None,
                    help="Streamed-decode prefetch recipe for a single positional "
                         "--stream-experts model: 'auto' measures the drive at "
                         "load, 'on' forces the aggressive recipe, 'off' the "
                         "conservative one (config mode: set `stream_fast_disk:` "
                         "per model).")
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
                         "experts with probability P (0 < P < 1) per token. A "
                         "shed layer still runs its shared expert or dense MLP. "
                         "On a layer with neither, Muse Spark runs its top "
                         "expert and other models run the layer in full "
                         "(config mode: set `moe_layer_shed: P` per model).")
    ap.add_argument("--moe-prestage", choices=("ranked", "keepers"),
                    default=None,
                    help="Lookahead prestage targeting for a streamed model; "
                         "default ranked. keepers filters predictions through "
                         "the active miss-shed policy and stages predicted "
                         "keepers demand-grade; needs --moe-miss-shed (config "
                         "mode: set `moe_prestage: keepers` per model).")
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
                    help="Cap a request that sets no output cap of its own "
                         "(default: until EOS or the context fills).")
    ap.add_argument("--no-family-defaults", action="store_true",
                    help="Do not seed each model family's model-card sampling "
                         "under profiles and requests (config mode: "
                         "`server.family_defaults: false`, which a config "
                         "reload restores).")
    ap.add_argument("--prefill-step-size", type=int, default=None, metavar="N",
                    help="Prefill chunk size in tokens (default 2048, 4096 for a sparse MoE "
                         "from the M5 on when memory allows, 8192 for a streamed "
                         "model) - lower "
                         "it to cap peak memory on long prompts, at some "
                         "prefill-throughput cost. Also via PREFILL_STEP_SIZE; "
                         "config mode: server.prefill_step_size.")
    ap.add_argument("--dtype", choices=("auto", "bfloat16", "float16"),
                    default=None, metavar="DTYPE",
                    help="Activation dtype for every model this server loads "
                         "(default auto). auto selects float16 on M1 and M2, "
                         "whose GPUs have no native bfloat16 arithmetic, and "
                         "bfloat16 on all other devices. Also via "
                         "GMLX_ACTIVATION_DTYPE; config mode: server.dtype.")
    ap.add_argument("--decode-prefill-ratio", type=_ratio_flag,
                    default=None, metavar="R",
                    help="Decode GPU-time share per prefill chunk under load "
                         "(default auto: paced or stock per tick from "
                         "measured load; numeric pins a static split, "
                         "1.0 ~= 50/50, 0 = stock scheduling). Also via "
                         "GMLX_DECODE_PREFILL_RATIO; config mode: "
                         "server.decode_prefill_ratio.")
    ap.add_argument("--prefill-tick-ms", type=float, default=None,
                    metavar="MS",
                    help="Wall-clock budget per prefill chunk while decode "
                         "rows are live; chunks are halved to fit (default "
                         "500; 0 = full chunks). Also via "
                         "GMLX_PREFILL_TICK_MS; config mode: "
                         "server.prefill_tick_ms.")
    ap.add_argument("--speculative-width-cap", type=int, default=None,
                    metavar="N",
                    help="Speculate on at most N rows of the live decode "
                         "batch (0 = uncapped; default: per drafter family). "
                         "Process-wide, overriding every model's config key. "
                         "Also via GMLX_MTP_WIDTH_CAP; config mode: "
                         "models[].speculative_width_cap.")
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
                         "mlx-whisper (needs the stt extra, and ffmpeg on the "
                         "server's PATH). MODEL is an alias (whisper-turbo, "
                         "whisper-turbo-q4, whisper-large/medium/small/base/tiny), "
                         "any HF repo in MLX-whisper format, or a local model dir; "
                         "bare --stt picks whisper-turbo - give it a value or put "
                         "it after the positional model, or the model path is "
                         "parsed as the STT model. Works in every serve mode and "
                         "overrides a config `server.stt:`.")
    ap.add_argument("--tts", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Text-to-speech: serve POST /v1/audio/speech via "
                         "mlx-audio (needs the tts extra; non-wav formats need "
                         "ffmpeg on the server's PATH). MODEL is an alias "
                         "(kokoro, kokoro-8bit/4bit, qwen3-tts, qwen3-tts-small), "
                         "any HF repo in "
                         "MLX-audio format, or a local model dir; bare --tts "
                         "picks kokoro. Works in every serve mode and overrides "
                         "a config `server.tts:`.")
    ap.add_argument("--embeddings", nargs="?", const="default", default=None,
                    metavar="MODEL",
                    help="Serve POST /v1/embeddings from a GGUF decoder-LM "
                         "embedder (alias qwen3-embed-0.6b/-4b/-8b, *.gguf, or "
                         "hf:<org>/<repo>/<file>.gguf; no extra) or a "
                         "safetensors encoder (alias embeddinggemma/arctic-l/"
                         "nomic-embed/bge-m3, HF repo, or local dir; no extra "
                         "needed). Bare flag picks qwen3-embed-0.6b; "
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


def _rope_override(a):
    """The positional model's rope options, or None."""
    from gmlx.commands.cli import rope_override_from_args

    return rope_override_from_args(a)


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
    if getattr(a, "asr", None):
        out += ["--asr", _abs(a.asr)]
    if getattr(a, "no_asr", False):
        out.append("--no-asr")
    if a.hf_source:
        out += ["--hf-source", a.hf_source]
    if a.speculative:
        out.append("--speculative")
    if a.draft_gguf:
        out += ["--draft-gguf", _abs(a.draft_gguf)]
    if getattr(a, "native_mtp", False):
        out.append("--native-mtp")
    if getattr(a, "draft_block_size", None):
        out += ["--draft-block-size", str(a.draft_block_size)]
    if getattr(a, "stochastic_mtp", None) is not None:
        out.append("--stochastic-mtp" if a.stochastic_mtp
                   else "--no-stochastic-mtp")
    if getattr(a, "chat_template", None):
        ct = a.chat_template
        out += ["--chat-template",
                _abs(ct) if os.path.exists(os.path.expanduser(ct)) else ct]
    if a.adapter:
        out += ["--adapter", _abs(a.adapter)]
    if getattr(a, "thinking", None) is not None:
        out += ["--thinking", a.thinking]
    if getattr(a, "thinking_budget", None) is not None:
        out += ["--thinking-budget", str(a.thinking_budget)]
    if getattr(a, "system_prompt", None) is not None:
        out += ["--system-prompt", a.system_prompt]
    if getattr(a, "chat_template_config", None):
        import json
        out += ["--chat-template-config", json.dumps(a.chat_template_config)]
    for key in LOAD_ENV:
        val = getattr(a, key, None)
        if val is not None:
            out += ["--" + key.replace("_", "-"), str(val)]
    if getattr(a, "profile", None):
        out += ["--profile", a.profile]
    for dest, _key in _SAMPLING_FLAGS:
        val = getattr(a, dest, None)
        if val is None:
            continue
        flag = "--" + dest.replace("_", "-")
        for v in (val if isinstance(val, list) else [val]):
            out += [flag, str(v)]
    if getattr(a, "reasoning_effort", None) is not None:
        out += ["--reasoning-effort", a.reasoning_effort]
    if getattr(a, "stream_cpu", False):
        out.append("--stream-cpu")
    if getattr(a, "stream_experts", False):
        out.append("--stream-experts")
    if getattr(a, "stream_fast_disk", None) is not None:
        out += ["--stream-fast-disk", a.stream_fast_disk]
    if getattr(a, "moe_expert_mass", None) is not None:
        out += ["--moe-expert-mass", str(a.moe_expert_mass)]
    if getattr(a, "moe_experts", None) is not None:
        out += ["--moe-experts", str(a.moe_experts)]
    if getattr(a, "moe_miss_shed", None) is not None:
        out += ["--moe-miss-shed", str(a.moe_miss_shed)]
    if getattr(a, "moe_layer_shed", None) is not None:
        out += ["--moe-layer-shed", str(a.moe_layer_shed)]
    if getattr(a, "moe_prestage", None) is not None:
        out += ["--moe-prestage", str(a.moe_prestage)]
    from gmlx.commands.cli import ROPE_FLAGS

    for dest in ROPE_FLAGS:
        if getattr(a, dest, None) is not None:
            out += ["--" + dest.replace("_", "-"), str(getattr(a, dest))]
    if a.budget_gb is not None:
        out += ["--budget-gb", str(a.budget_gb)]
    if a.max_models is not None:
        out += ["--max-models", str(a.max_models)]
    for p in a.pin:
        out += ["--pin", p]
    if a.max_tokens is not None:
        out += ["--max-tokens", str(a.max_tokens)]
    if getattr(a, "no_family_defaults", False):
        out.append("--no-family-defaults")
    if getattr(a, "prefill_step_size", None) is not None:
        out += ["--prefill-step-size", str(a.prefill_step_size)]
    if getattr(a, "decode_prefill_ratio", None) is not None:
        out += ["--decode-prefill-ratio", str(a.decode_prefill_ratio)]
    if getattr(a, "prefill_tick_ms", None) is not None:
        out += ["--prefill-tick-ms", str(a.prefill_tick_ms)]
    if getattr(a, "speculative_width_cap", None) is not None:
        out += ["--speculative-width-cap", str(a.speculative_width_cap)]
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


def _ambiguous_runs(a, *, prune: bool = False) -> list | None:
    """The live managed runfiles when a bare lifecycle verb has more than
    one server to choose from (no --host/--port given), else None. Stale
    runfiles (dead or recycled pid) never make a verb ambiguous: with
    ``prune`` they are cleared here and reported; otherwise they are just
    left out of the count (``status`` lists them itself)."""
    if a.host is not None or a.port is not None:
        return None
    from . import lifecycle
    live, stale = lifecycle.classify_runs()
    if prune and stale:
        removed = lifecycle.prune_stale_runs()
        if removed:
            print(_stale_summary(removed, "cleared"), file=sys.stderr)
    return live if len(live) > 1 else None


def _stale_summary(runs: list, verb: str) -> str:
    what = ", ".join(f"{r.get('host')}:{r.get('port')} ({r.get('stale_reason')}, "
                     f"{r.get('age')})" for r in runs)
    return f"{verb} {len(runs)} stale runfile{'s' if len(runs) != 1 else ''}: {what}"


def _refuse_ambiguous(prog: str, runs: list) -> int:
    urls = ", ".join(r.get("url") or f"http://{r.get('host')}:{r.get('port')}"
                     for r in runs)
    print(f"error: {len(runs)} managed servers are running ({urls}) - "
          f"pick one with --port", file=sys.stderr)
    return 2


def _cmd_stop_stale(prog: str) -> int:
    from . import lifecycle
    _live, stale = lifecycle.classify_runs()
    if not stale:
        print("no stale runfiles")
        return 0
    removed = lifecycle.prune_stale_runs()
    print(_stale_summary(removed, "cleared") if removed else "no stale runfiles")
    return 0


def _cmd_stop(argv: list, prog: str = "gmlx stop") -> int:
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Stop a backgrounded gmlx server (SIGTERM, then SIGKILL).")
    _add_target_args(ap)
    ap.add_argument("--timeout", type=float, default=15.0,
                    help="Seconds to wait for graceful shutdown before SIGKILL "
                         "(SIGKILL cuts any in-flight generation). Default 15.")
    ap.add_argument("--stale", action="store_true",
                    help="Only clear runfiles whose server is gone (exited or "
                         "pid recycled); signals nothing. A bare `stop` clears "
                         "them too, on its way to the live server.")
    a = ap.parse_args(argv)
    from . import lifecycle
    if a.stale:
        return _cmd_stop_stale(prog)
    runs = _ambiguous_runs(a, prune=True)
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
    runs = _ambiguous_runs(a, prune=True)
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
    stale = [] if (a.host is not None or a.port is not None) \
        else lifecycle.classify_runs()[1]
    runs = _ambiguous_runs(a)
    if runs:
        # Several managed servers: report on all of them instead of guessing.
        if a.json:
            infos = [lifecycle.status_info(r.get("host"), r.get("port"))
                     for r in runs]
            infos = [i for i in infos if i]
            print(json.dumps({"servers": infos, "stale": stale}, indent=2))
            return 0 if any(i["running"] for i in infos) else 3
        rcs = [lifecycle.status(r.get("host"), r.get("port")) for r in runs]
        _print_stale(stale)
        _print_launch_sessions()
        return 0 if 0 in rcs else 3
    host, port = lifecycle.auto_target(a.host, a.port)
    rc = lifecycle.status(host, port, as_json=a.json)
    # A bare status on the one live server still owes the user the stale
    # leftovers (an e2e harness or a crash leaves them; nothing else lists
    # them), minus the target itself, which status() just described.
    if not a.json:
        _print_stale([r for r in stale
                      if (r.get("host"), r.get("port")) != (host, port)])
        _print_launch_sessions()
    return rc


def _print_launch_sessions() -> None:
    """The launch sessions that run, since a detached one has no window
    that shows it. The exit code stays the server's."""
    from gmlx.commands.launch_container import status_lines
    for line in status_lines():
        print(line)


def _print_stale(stale: list) -> None:
    if not stale:
        return
    for r in stale:
        print(f"stale runfile {r.get('host')}:{r.get('port')}: "
              f"{r.get('stale_reason')}, {r.get('age')}")
    print(f"  {len(stale)} stale runfile{'s' if len(stale) != 1 else ''} - "
          f"`gmlx stop --stale` clears them")


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
                            ("--asr", a.asr),
                            ("--draft-gguf", a.draft_gguf), ("--adapter", a.adapter),
                            ("--config", a.config)):
            if value and not os.path.exists(os.path.expanduser(value)):
                print(f"error: {flag}: no such file: {value}", file=sys.stderr)
                return 2
        if _refuse_bare_without_config(a):
            return 2
        try:
            host, port, cfg_path, serve_args, api_key, _menubar = _bg_plan(a)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if a.headless:
            return lifecycle.service_install(
                serve_args, host=host, port=port, config_abspath=cfg_path,
                log=a.log, keepalive=a.keepalive, api_key_set=bool(api_key),
                cwd=os.getcwd())
        return lifecycle.service_install_menubar(
            serve_args, host=host, port=port, config_abspath=cfg_path,
            log=a.log, autostart=a.autostart,
            start_timeout=a.start_timeout, api_key=api_key, cwd=os.getcwd())
    ap = argparse.ArgumentParser(prog=f"{prog} {action}")
    _add_target_args(ap)
    a = ap.parse_args(rest)
    host, port = lifecycle.auto_target(a.host, a.port)
    return (lifecycle.service_uninstall(host, port) if action == "uninstall"
            else lifecycle.service_status(host, port))


def _line_buffered_stdout() -> None:
    """Write each line of standard output as it ends. A background server's
    output goes to its log file, where Python would hold the ``[server]``
    lines in a block buffer until something flushes it, so ``gmlx logs``
    would not show them. This covers every way a server starts, including a
    LaunchAgent that an older gmlx wrote."""
    stream = sys.stdout
    try:
        if not stream.isatty() and not stream.line_buffering:
            stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass                    # a replaced or closed stream keeps its mode


def _refresh_comfyui_pack() -> None:
    """A gmlx upgrade reaches the ComfyUI node pack here: a server start is
    the one gmlx step sure to come before the pack's next request. The
    detached start does it too, so the line shows in the terminal."""
    from gmlx.commands.comfyui import refresh_for_serve

    line = refresh_for_serve()
    if line:
        print(f"[server] {line}")


def _cmd_serve(argv: list, prog: str = "gmlx serve") -> int:
    _line_buffered_stdout()
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
                        ("--asr", a.asr),
                        ("--draft-gguf", a.draft_gguf), ("--adapter", a.adapter),
                        ("--config", a.config)):
        if value and not os.path.exists(os.path.expanduser(value)):
            hint = ""
            if str(value).startswith(("hf:", "http://", "https://")):
                hint = (" (remote refs work with `gmlx validate` / "
                        "`gmlx pull`; serve needs a local file)")
            what = "no such file" if flag == "model" else f"{flag}: no such file"
            print(f"error: {what}: {value}{hint}", file=sys.stderr)
            if (a.launchd and flag == "--config"
                    and not os.path.isabs(os.path.expanduser(value))):
                # An older gmlx recorded --config gmlx.yaml relative to the
                # folder it ran in, and launchd runs a headless agent in /.
                # Such a start never works, so it ends with success, and
                # launchd does not start it again every 10 seconds.
                print("A login start cannot find a relative --config, so it stops "
                      "here. Run gmlx doctor for the steps.", file=sys.stderr)
                return 0
            return 2
    if _refuse_bare_without_config(a):
        # launchd starts a headless agent again after a failed exit, every 10
        # seconds. This start cannot work until a config exists, so it ends
        # with success and launchd leaves it stopped.
        return 0 if a.launchd else 2

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
        _refresh_comfyui_pack()
        rc = lifecycle.start_background(
            serve_args, host=host, port=port, config_abspath=cfg_path,
            log=a.log, start_timeout=a.start_timeout, api_key=api_key,
            cwd=os.getcwd())
        if rc == 0 and menubar and not a.no_menubar \
                and lifecycle.gui_session_available():
            lifecycle.start_menubar(auto=True)  # one machine-wide bar; tracks the primary
        return rc

    try:
        _import_serving()
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # The file that the config path leads to now, which the server reads at
    # its start. The runfile records it for the key that the server keeps.
    a.config_real = os.path.realpath(os.path.expanduser(a.config)) if a.config else None
    try:
        cfg, reload_fn = _resolve_cfg(a)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _refresh_comfyui_pack()
    return _serve(cfg, a, reload_fn)


def _resolve_cfg(a) -> tuple:
    """Return ``(ServerCfg, reload_fn)`` for the selected start mode, with
    ``--no-family-defaults`` applied to whichever config that is."""
    cfg, reload_fn = _resolve_mode_cfg(a)
    if getattr(a, "no_family_defaults", False):
        cfg.family_defaults = False
    return cfg, reload_fn


def _resolve_mode_cfg(a) -> tuple:
    if ((a.config or a.models_dir or not a.model)
            and _rope_override(a) is not None):
        # One set of rope options cannot fit every configured model.
        raise ConfigError(
            "the rope scaling flags apply to a single positional model; "
            "set a model's rope_scaling block in the config instead")
    if a.config:
        path = a.config
        return _load_with_discover(path), _make_reload_fn(path, a)
    if a.models_dir:
        return _discovery_cfg(a.models_dir, a), None
    if a.model:
        cfg = _single_model_cfg(a)
        _check_positional(cfg)
        return cfg, None
    # bare: the first existing default config. Informational notes go to
    # stderr: `--print-config > file` must leave stdout pure YAML (the emitted
    # header promises --config round-trips).
    for p in default_config_paths():
        if p.exists():
            print(f"[server] loading config {p}", file=sys.stderr)
            # The runfile records this file as if the start named it, since
            # the argv of a bare start names no config.
            a.config_default = os.path.abspath(p)
            a.config_real = os.path.realpath(p)
            return _load_with_discover(p), _make_reload_fn(str(p), a)
    raise ConfigError(NO_CONFIG_LINE)


def _load_with_discover(path) -> ServerCfg:
    """Load the config and add the models its ``discover`` scans find."""
    cfg = load_config(path)
    found = discovery.merge_discovered(cfg)
    if found:
        print(f"[server] discover: {', '.join(m.id for m in found)}",
              file=sys.stderr)
    return cfg


def _make_reload_fn(path, a=None):
    """The reload of the config at ``path``. With the start flags ``a``, it
    keeps ``--no-family-defaults`` and the service flags, and turns each
    service on, off or to another model as the file now says. A service
    that fails to resolve fails the reload and changes nothing. The result
    names, under ``services_off``, each configured speech service the reload
    left off because its extra is missing, with the command that adds it."""
    def _reload():
        from .bridge_vlm import register_resolved_models
        from .patches import routes
        cfg = _load_with_discover(path)
        off: dict = {}
        if a is not None:
            if getattr(a, "no_family_defaults", False):
                cfg.family_defaults = False
            missing = routes.clear_missing_services()
            try:
                _resolve_service_models(cfg, a)
            except Exception:
                # What the failed resolve noted goes; what was noted before
                # comes back.
                routes.clear_missing_services()
                for key, extra in missing.items():
                    routes.note_missing_service(key, extra=extra)
                raise
            off = routes.missing_extras()
        register_resolved_models(cfg)        # warm entries persist (keyed by path)
        if a is not None:
            routes.install_service_routes(cfg)
            _warm_service_models(cfg)
        out: dict = {"models": len(cfg.models)}
        if off:
            out["services_off"] = off
        return out
    return _reload


def _recording_reload(reload_fn, host, port, config_given: str, *,
                      launchd: bool = False):
    """``reload_fn``, which also records in the runfile the file that a
    reload read through ``config_given``. A link there can lead to another
    file than at the start, and launch reads the models and profiles of the
    server from the file it read last. ``launchd`` tells that launchd
    started this server."""
    from . import lifecycle

    def _reload():
        real = os.path.realpath(config_given)
        out = reload_fn()
        # The reload worked, so a runfile that cannot be written only
        # keeps the earlier file.
        with contextlib.suppress(OSError):
            lifecycle.note_config_reload(host, port, config_given=config_given,
                                         config_real=real, launchd=launchd)
        return out
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


def cors_origin_lines(origins) -> list[str]:
    """The start-up lines for the listed origins that are not loopback: a web
    origin lets browser pages call the server, a wildcard entry lets every
    extension of one browser, and any other origin lets an app."""
    from gmlx.config import origin_is_loopback
    from gmlx.serve.patches.hardening import EXTENSION_SCHEMES
    browsers = {"chrome-extension": "Chrome, Edge and other Chromium browsers",
                "moz-extension": "Firefox", "safari-web-extension": "Safari"}
    lines = []
    for origin in origins:
        if origin_is_loopback(origin):
            continue
        if origin.endswith("://*"):
            lines.append(f"[server] every browser extension in "
                         f"{browsers[origin.partition('://')[0]]} may call this server "
                         f"({origin} in server.cors_origins)")
        elif origin.startswith(("http://", "https://")):
            lines.append(f"[server] browser pages at {origin} may call this server "
                         "(server.cors_origins)")
        elif origin.partition("://")[0] in EXTENSION_SCHEMES:
            lines.append(f"[server] the browser extension at {origin} may call this "
                         "server (server.cors_origins)")
        else:
            lines.append(f"[server] the app that sends Origin {origin} may call this "
                         "server (server.cors_origins)")
    return lines


def _dump_cfg_yaml(cfg: ServerCfg) -> str:
    """Serialize a resolved :class:`ServerCfg` (defaults filled in) to YAML - every
    schema key with its effective value, for `serve --print-config`. The output
    uses the on-disk schema shape (server / profiles / rules / models / aliases),
    so it feeds back through `--config` / :func:`load_config` unchanged."""
    import dataclasses

    import yaml
    d = dataclasses.asdict(cfg)
    server = {k: d.pop(k) for k in
              ("host", "port", "api_key", "no_auth", "media_urls", "cors_origins",
               "model_dirs", "budget_gb",
               "max_models", "hf_cache", "menubar", "token_queue_timeout_s",
               "prefill_step_size", "dtype",
               "decode_prefill_ratio", "prefill_tick_ms",
               "cache_limit_gb", "family_defaults",
               "stochastic_mtp", "gpu_keepwarm", "stt", "tts", "embeddings",
               "rerank", "systemone",
               "defaults", "cache", "assistants", "assistant_allow_remote",
               "image")}
    # Profile/model names double as dict keys; drop the redundant fields so the
    # entries match the file schema (which has no `name`/`id` keys). An
    # auto drafter writes back as `speculative: auto`.
    profiles = {n: {k: v for k, v in p.items() if k != "name"}
                for n, p in d.pop("profiles").items()}
    models = {}
    for mid, m in d.pop("models").items():
        m.pop("id")
        if m.pop("speculative_auto") and m.get("speculative"):
            m["speculative"] = "auto"
        models[mid] = m
    images = {}
    for iid, m in d.pop("image_models").items():
        m.pop("id")
        m["adapters"] = [{"path": p, "strength": st} for p, st in m["adapters"]]
        if m["ttl_s"] is None:
            # An absent key inherits the server's; a written null never unloads.
            m.pop("ttl_s")
        images[iid] = m
    talk = d.pop("talk")
    talk.pop("assistant", None)     # attached shared object, not a talk: key
    doc = {"server": server, "profiles": profiles, "rules": d.pop("rules"),
           "models": models, "aliases": d.pop("aliases"),
           "image_models": images,
           "discover": d.pop("discover"), "talk": talk,
           "launch": d.pop("launch"),
           "assistant": d.pop("assistant"),
           "theme": d.pop("theme"), "themes": d.pop("themes")}
    assert not d, f"ServerCfg fields missing from --print-config: {sorted(d)}"
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False,
                          width=2 ** 16)
    return ("# effective gmlx config (every key + default shown); "
            "valid as a --config file\n" + body)


def _discovery_cfg(dirs: list, a) -> ServerCfg:
    specs = [DiscoverSpec(dir=d, recursive=a.recursive) for d in dirs]
    images: list = []
    models = discovery.scan_dirs(specs, dirs, images=images)
    return ServerCfg(
        host=a.host or "127.0.0.1",
        port=a.port or 8080,
        model_dirs=list(dirs),
        budget_gb=a.budget_gb,
        max_models=a.max_models,
        hf_cache=a.hf_cache,
        defaults=ServerDefaults(),
        models={m.id: m for m in models},
        image_models={im.id: im for im in images},
    )


def _single_model_cfg(a) -> ServerCfg:
    mp = os.path.abspath(os.path.expanduser(a.model))
    mid, _q = discovery.derive_id(os.path.basename(a.model))
    mid = mid or "model"
    native_mtp = bool(getattr(a, "native_mtp", False))
    speculative = bool(a.speculative or a.draft_gguf or native_mtp)
    # A --chat-template override rides on the model's overrides (the same slot a
    # config `overrides: {chat_template: ...}` uses), so it flows through resolve_model
    # -> the load bridge identically.
    overrides = ({"chat_template": a.chat_template}
                 if getattr(a, "chat_template", None) else {})
    if getattr(a, "thinking", None) is not None:
        overrides["thinking"] = a.thinking
    sampling = {key: getattr(a, dest, None) for dest, key in _SAMPLING_FLAGS}
    sampling = {k: v for k, v in sampling.items() if v is not None}
    if getattr(a, "thinking_budget", None) is not None:
        sampling["thinking_budget"] = a.thinking_budget
    if sampling:
        overrides["sampling"] = sampling
    if getattr(a, "reasoning_effort", None) is not None:
        overrides["reasoning_effort"] = a.reasoning_effort
    if getattr(a, "system_prompt", None) is not None:
        overrides["system"] = a.system_prompt
    if getattr(a, "chat_template_config", None):
        overrides["chat_template_kwargs"] = dict(a.chat_template_config)
    # The load keys (kv_bits, kv_quant_scheme, ...) keyed exactly as the
    # config `load:` section, so they resolve and price like a config model.
    load = {k: getattr(a, k, None) for k in LOAD_ENV}
    load = {k: v for k, v in load.items() if v is not None}
    if load:
        overrides["load"] = load
    model = ModelCfg(
        id=mid,
        path=mp,
        profile=getattr(a, "profile", None),
        mmproj=os.path.abspath(os.path.expanduser(a.mmproj)) if a.mmproj else None,
        asr=(False if getattr(a, "no_asr", False)
             else os.path.abspath(os.path.expanduser(a.asr))
             if getattr(a, "asr", None) else None),
        draft_gguf=(os.path.abspath(os.path.expanduser(a.draft_gguf))
                    if a.draft_gguf and not native_mtp else None),
        native_mtp=native_mtp,
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
        moe_prestage=getattr(a, "moe_prestage", None),
        prefill_feeder=getattr(a, "prefill_feeder", None),
        decode_feeder=getattr(a, "decode_feeder", None),
        stream_fast_disk=getattr(a, "stream_fast_disk", None),
        rope_scaling=_rope_override(a),
        pin=True,                            # the single model is always pinned
    )
    models = {mid: model}
    if model.adapter:
        # The bare base under `<id>-base` on the same resident entry (the
        # adapter is a per-request slot, not a second load), so a client or a
        # bench config can address both without a config file. Ids may not
        # carry `@` (it addresses profiles), hence the `-base` suffix.
        models[f"{mid}-base"] = dataclasses.replace(model, id=f"{mid}-base",
                                                    adapter=None)
    return ServerCfg(
        host=a.host or "127.0.0.1",
        port=a.port or 8080,
        budget_gb=a.budget_gb,
        max_models=a.max_models,
        hf_cache=a.hf_cache,
        family_defaults=not getattr(a, "no_family_defaults", False),
        defaults=ServerDefaults(model=mid),
        models=models,
    )


def _check_positional(cfg: ServerCfg) -> None:
    """Run the positional model through the checks a config file gets at
    parse, so an unknown --profile is refused at start instead of inside the
    engine's first load, then resolve it once for the rest."""
    from gmlx.config import _validate
    _validate(cfg)
    for mid in cfg.models:
        resolve_model(mid, cfg)


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
        from .patches.routes import note_missing_service
        print(f"[server] {key} disabled - {e}", file=sys.stderr)
        note_missing_service(key.removeprefix("server."))
        return None
    except ConfigError as e:
        raise ConfigError(f"{key}: {e}") from None


_SERVICE_KEYS = ("stt", "tts", "embeddings", "rerank")
# The model each service runs, as of the start or the last reload.
_LIVE_SERVICES: dict[str, str] = {}


# The top-level modules of the packages each speech extra installs. Only a
# missing one of these turns a configured service off.
_EXTRA_MODULES = {
    "stt": frozenset({"mlx_whisper", "multipart"}),
    "tts": frozenset({"mlx_audio", "spacy", "num2words", "addict", "phonemizer",
                      "espeakng_loader"}),
}

def _extra_missing(key: str, e: ImportError) -> bool:
    """Whether ``e``, or the error it was raised from, is a module of the
    ``key`` extra that is not installed."""
    for err in (e, e.__cause__):
        if isinstance(err, ModuleNotFoundError) and \
                (err.name or "").split(".")[0] in _EXTRA_MODULES[key]:
            return True
    return False


def _speech_service(key: str, flag, cfg, resolve, check_extra):
    """The model of the speech service ``key``, or None when it is off. A
    missing extra stops the start when ``--stt`` or ``--tts`` asked for the
    service. When only the config names it, the server warns and starts
    without it, and its routes answer 404 with the install command. Any
    other import error stops the start."""
    value = flag or getattr(cfg, key, None)
    if not value:
        return None
    model = resolve(value)
    try:
        check_extra(model)
    except ImportError as e:
        if flag or not _extra_missing(key, e):
            raise
        from gmlx.commands.extras import install_hint

        from .lifecycle import SERVICE_OFF_MARK
        from .patches.routes import note_missing_service
        hint = install_hint(key)
        print(f"[server] server.{key}{SERVICE_OFF_MARK}{key} extra is not installed. "
              f"To add it: {hint}", file=sys.stderr)
        note_missing_service(key, extra=hint)
        return None
    return model


def _resolve_service_models(cfg, a) -> None:
    """Resolve the four service keys of ``cfg`` in place, where a ``--stt``,
    ``--tts``, ``--embeddings`` or ``--rerank`` flag overrides the config
    value. A malformed value raises, naming the config key. A speech
    service whose extra is missing raises for a flag and turns the service
    off for a config value (see :func:`_speech_service`). A missing
    embeddings or rerank file turns that service off (see
    :func:`_resolve_service`)."""
    from . import stt as stt_mod
    from . import tts as tts_mod

    cfg.stt = _speech_service("stt", getattr(a, "stt", None), cfg,
                              lambda v: (stt_mod.paired_companion(cfg)
                                         if str(v).strip().lower()
                                         == stt_mod.ALIAS_COMPANION
                                         else stt_mod.resolve_stt_model(v)),
                              stt_mod.check_backend)
    cfg.tts = _speech_service("tts", getattr(a, "tts", None), cfg,
                              tts_mod.resolve_tts_model,
                              lambda _model: tts_mod.import_mlx_audio())

    # Relative GGUF paths search server.model_dirs, as a models: entry does.
    model_dirs = getattr(cfg, "model_dirs", None) or []
    emb_value = getattr(a, "embeddings", None) or getattr(cfg, "embeddings", None)
    cfg.embeddings = None
    if emb_value:
        from . import embeddings as emb_mod
        cfg.embeddings = _resolve_service(
            "server.embeddings", emb_mod.resolve_embeddings_model,
            emb_value, model_dirs)

    rerank_value = getattr(a, "rerank", None) or getattr(cfg, "rerank", None)
    cfg.rerank = None
    if rerank_value:
        from . import rerank as rerank_mod
        cfg.rerank = _resolve_service(
            "server.rerank", rerank_mod.resolve_rerank_model,
            rerank_value, model_dirs)


def _warm_service_models(cfg) -> None:
    """Load each service model in the background, so the first request
    finds it loaded. The server comes up without waiting. A service that a
    reload turned off or moved to another model drops its old model first."""
    import importlib
    for key in _SERVICE_KEYS:
        new, old = getattr(cfg, key, None), _LIVE_SERVICES.get(key)
        if not (new or old):
            continue
        mod = importlib.import_module(f".{key}", __package__)
        if old and old != new:
            mod.release()
        if new:
            mod.prewarm(new)
            _LIVE_SERVICES[key] = new
        else:
            _LIVE_SERVICES.pop(key, None)


def _serve(cfg: ServerCfg, a, reload_fn) -> int:
    from . import bridge_vlm as serving
    from .residency import install_gguf_residency_pool
    from .patches import install_server_patches
    from .bridge_vlm import (
        install_gguf_server_bridge,
        register_gguf_vlm,
        register_resolved_models,
    )

    register_resolved_models(cfg)
    # MLX buffer-cache limit: explicit env/config wins, else the auto policy
    # bounds the cache when the biggest configured model leaves little slack
    # (deep-context safety; see server_memory).
    from .memory import apply_cache_limit
    apply_cache_limit(cfg)
    from .cb_phase import (FINE, STREAMED_DECODE, STREAMED_PREFILL, decode_caps,
                           install_cb_phase_flips)
    if install_cb_phase_flips():
        print(f"[server] command buffer caps: per-phase (decode {decode_caps(None)}, "
              f"prefill {FINE}; streamed decode {STREAMED_DECODE}, "
              f"streamed prefill {STREAMED_PREFILL})")
    else:
        print("[server] command buffer caps: not installed "
              "(GMLX_CB_PHASE=0, MLX_MAX_OPS_PER_BUFFER or "
              "MLX_MAX_MB_PER_BUFFER set, or mlx-kquant lacks set_cb_caps); "
              "every phase keeps the caps set at start")
    from .gpu_share import install_chat_ticks
    install_chat_ticks()
    # Single-model --hf-source override (niche): re-register the VLM with the
    # explicit processor source (ModelCfg carries no hf_source field).
    if a.model and a.hf_source and a.mmproj:
        register_gguf_vlm(os.path.abspath(os.path.expanduser(a.model)),
                          os.path.abspath(os.path.expanduser(a.mmproj)),
                          hf_source=a.hf_source)
    if a.max_tokens is not None:
        os.environ["MLX_VLM_MAX_TOKENS"] = str(a.max_tokens)
    else:
        from .patches.sampling import UNTIL_EOS
        os.environ.setdefault("MLX_VLM_MAX_TOKENS", str(UNTIL_EOS))
    if getattr(a, "draft_block_size", None):
        # Read lazily at drafter-load time (server_bridge_vlm.load_drafter).
        os.environ["GMLX_DRAFT_BLOCK_SIZE"] = str(a.draft_block_size)
    # Startup-only, like the timeout below; a config reload does not
    # re-apply it.
    from gmlx.spec.speculative import set_stoch_accept, stoch_accept_enabled

    stoch = getattr(a, "stochastic_mtp", None)
    if stoch is None:
        stoch = cfg.stochastic_mtp
    if stoch is not None:
        set_stoch_accept(stoch)
    if not stoch_accept_enabled():
        print("[server] stochastic MTP acceptance: off (sampled requests "
              "accept only exact matches)")
    # Activation dtype: flag > config > exported env. Server-wide because the
    # reason to leave bfloat16 is a property of the GPU, not of a model. Read
    # at each model's load, so setting it here covers every model this server
    # loads, including ones registered by a later config reload.
    dtype = getattr(a, "dtype", None) or cfg.dtype
    if dtype is not None:
        os.environ["GMLX_ACTIVATION_DTYPE"] = str(dtype)
        print(f"[server] activation dtype: {dtype}")
    keepwarm = getattr(a, "gpu_keepwarm", None)
    if keepwarm is None:
        keepwarm = cfg.gpu_keepwarm
    if keepwarm is not None:
        # Startup-only. The loader's gate starts the heartbeat when a
        # streamed model with a decode feeder loads; the heartbeat itself
        # parks whenever no request is decoding (keepwarm.touch()).
        os.environ["GMLX_GPU_KEEPWARM"] = "1" if keepwarm else "0"
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

    # Decode-loop housekeeping interval (mx.async_eval of the cache states,
    # then mx.clear_cache): mlx-vlm's default of every 50 steps costs ~2.5 ms
    # per event (~0.05 ms/tok amortized at B=1). 256 keeps the memory bound
    # while cutting the amortized cost ~5x. An exported env wins.
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

    # Decode-priority pacing ratio: flag > config > exported env > default
    # (1.0 in batch_sched). Read per tick from the env, so setting it here
    # is enough regardless of patch-install order.
    ratio = getattr(a, "decode_prefill_ratio", None)
    if ratio is None:
        ratio = cfg.decode_prefill_ratio
    if ratio is not None:
        if isinstance(ratio, str):
            from .auto_ratio import (c_threshold, deadline_s, floor_rho,
                                     paced_ratio)
            os.environ["GMLX_DECODE_PREFILL_RATIO"] = "auto"
            print(f"[server] decode-prefill pacing ratio: auto "
                  f"(floor {floor_rho():.2f} -> paced {paced_ratio():.2f}, "
                  f"chunk threshold {c_threshold():.1f}, "
                  f"deadline {deadline_s():.1f}s)")
        elif ratio < 0:
            print(f"[server] ignoring negative decode-prefill ratio {ratio}")
        else:
            os.environ["GMLX_DECODE_PREFILL_RATIO"] = str(ratio)
            print(f"[server] decode-prefill pacing ratio: {ratio}")

    # Prefill tick budget: flag > config > exported env > default (500 in
    # prefill_decay). Read per chunk from the env, so setting it here is
    # enough regardless of patch-install order.
    tick = getattr(a, "prefill_tick_ms", None)
    if tick is None:
        tick = cfg.prefill_tick_ms
    if tick is not None:
        if tick < 0:
            print(f"[server] ignoring negative prefill tick budget {tick}")
        else:
            os.environ["GMLX_PREFILL_TICK_MS"] = str(tick)
            print(f"[server] prefill tick budget: "
                  f"{f'{tick:g} ms' if tick else 'off (full chunks)'}")

    # MTP width cap: the flag is process-wide and outranks per-model config,
    # so say so when both are in play rather than letting the key look active.
    width_cap = getattr(a, "speculative_width_cap", None)
    if width_cap is not None:
        if width_cap < 0:
            print(f"[server] ignoring negative speculative width cap "
                  f"{width_cap}")
        else:
            os.environ["GMLX_MTP_WIDTH_CAP"] = str(width_cap)
            print(f"[server] speculative width cap: "
                  f"{width_cap or 'uncapped'} (process-wide)")
            if any(m.speculative_width_cap is not None
                   for m in getattr(cfg, "models", {}).values()):
                print("[server] note: --speculative-width-cap overrides the "
                      "per-model speculative_width_cap config key")

    resolved = serving.resolved_models()
    pinned_paths, image_pins = _pinned_paths(resolved, serving.image_models(),
                                             a.pin)

    # Only an explicit pin (`pin: true`, --pin, or the single-GGUF positional)
    # holds the preload for the process lifetime. A default or sole model
    # preloads too, but stays TTL- and budget-evictable.
    preload = _preload_id(cfg)
    preload_rm = resolved.get(preload) if preload else None
    preload_pinned = preload_rm is not None and preload_rm.path in pinned_paths
    if preload and preload_pinned:
        os.environ["MLX_VLM_PRELOAD_MODEL"] = preload

    install_gguf_server_bridge()
    budget_gb = a.budget_gb if a.budget_gb is not None else cfg.budget_gb
    budget_bytes = int(budget_gb * 1024**3) if budget_gb else None
    install_gguf_residency_pool(
        budget_bytes=budget_bytes,
        max_models=a.max_models if a.max_models is not None else cfg.max_models,
        pinned=pinned_paths or None,
    )
    # The services resolve before the port binds, so a malformed value, or
    # a missing speech extra that a flag asked for, stops the start.
    _resolve_service_models(cfg, a)
    _warm_service_models(cfg)

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
    config = getattr(a, "config", None)
    config_given = os.path.abspath(os.path.expanduser(config)) if config else None
    # A bare start reads the first default config, at its start and at each
    # reload.
    bare = config_given is None and bool(getattr(a, "config_default", None))
    if bare:
        config_given = a.config_default
    # A login agent owns a runfile that records no pid.
    launchd = bool(getattr(a, "launchd", False))
    if reload_fn is not None and config_given:
        reload_fn = _recording_reload(reload_fn, host, port, config_given,
                                      launchd=launchd)
    # A request may name media files in this folder, and in no other.
    from gmlx.serve.media_sinks import ensure_media_root
    ensure_media_root()
    install_server_patches(cfg, reload_fn=reload_fn)
    for line in cors_origin_lines(getattr(cfg, "cors_origins", None) or ()):
        print(line)
    if getattr(a, "ignore_eos", False) or env_bool("GMLX_IGNORE_EOS", False):
        from .patches import install_ignore_eos
        install_ignore_eos()
        print("[server] ignore-eos: decode runs to max_tokens (EOS suppressed)")

    import signal

    if reload_fn is None and hasattr(signal, "SIGHUP"):
        # No file to read again. Without a handler, SIGHUP would stop the server.
        def _ignore_sighup(_sig, _frame):
            print("[server] SIGHUP ignored: started without a config file, "
                  "so there is nothing to reload")

        signal.signal(signal.SIGHUP, _ignore_sighup)
    if reload_fn is not None:
        def _on_sighup(_sig, _frame):
            try:
                print(f"[server] SIGHUP config reload: {reload_fn() or {}}", flush=True)
            except Exception as exc:
                print(f"[server] SIGHUP config reload failed: {exc}",
                      file=sys.stderr, flush=True)

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
    _warn_stale_stream_marks(cfg, config_given)
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
    # Audio requests run these two. A PATH entry that a container client
    # can write is skipped, and the log says so.
    from .media_programs import log_programs
    log_programs(cfg)

    import uvicorn

    from . import patches

    # Preload off the startup path: mlx-vlm's lifespan loads MLX_VLM_PRELOAD_MODEL
    # synchronously *before* the port accepts connections, so a big model makes the
    # whole server unreachable (menu bar / health probes see "down") until it
    # finishes. The env was already consumed by the residency-pool install above
    # (it pins a pinned preload); pop it now so the lifespan skips the blocking load,
    # and warm the model in a background thread instead - the port binds and /health
    # answers immediately while the load runs. A pinned primary keeps a retained
    # hold, exactly as the lifespan hold would have; an unpinned one is released
    # after the warm. The other pinned models warm next, in file order, and the
    # pool's pinned set keeps them resident; defaults.preload extras follow,
    # LRU-evictable.
    pre = cfg.defaults.preload
    warm_paths = {preload_rm.path} if preload_rm is not None else set()
    pinned_ids = []
    for mid, rm in resolved.items():
        if rm.path in pinned_paths and rm.path not in warm_paths:
            warm_paths.add(rm.path)
            pinned_ids.append(mid)
    for iid, path in image_pins.items():
        if path not in warm_paths:
            warm_paths.add(path)
            pinned_ids.append(iid)
    extras = []
    for m in pinned_ids + (list(cfg.models) if pre == "all" else list(pre or ())):
        if m != preload and m not in extras:
            extras.append(m)
    if preload or extras:
        os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
        patches.spawn_preload_warm(preload, extras, retain=preload_pinned)
        if extras:
            print(f"[server] preload: warming {', '.join(extras)} in background")

    # Record what code this server actually booted with. The launcher stamps
    # at spawn, but launchd respawns bypass it (login, crash recovery); the
    # stamp lets status/launch flag a server that predates a source change.
    # The config file it read goes in the runfile for the same reason. This
    # runs before the bind, so only the server that the runfile records
    # writes it.
    from . import lifecycle

    lifecycle.stamp_run(host, port, config_given=config_given,
                        config_real=getattr(a, "config_real", None), bare=bare,
                        launchd=launchd)
    # Each client connection holds a descriptor, and the soft limit of 256
    # that Terminal gives would let a few hundred idle connections reset
    # every other client.
    from gmlx.rlimit import low_limit_warning, raise_nofile_limit
    low = low_limit_warning(raise_nofile_limit(), "the server")
    if low:
        print(f"[server] warning: {low}", file=sys.stderr)

    # Session sockets of an earlier run have no listener. This removes them now,
    # and this run's sockets when the server stops.
    from .patches.session_sockets import prepare_session_sockets
    prepare_session_sockets(host, port)

    loop = "uvloop" if _has_uvloop() else "auto"
    uvicorn.run("gmlx._vendor.mlx_vlm.server:app", host=host, port=port, workers=1,
                server_header=False, loop=loop,
                # The host guard of a loopback bind reads the socket peer,
                # which X-Forwarded-For from a local proxy must not replace.
                proxy_headers=host not in _LOOPBACK,
                # log_level re-levels uvicorn's own loggers after the
                # dictConfig; the config itself carries the level to the
                # gmlx/mlx_vlm loggers, so the flag governs the whole server
                # while the timestamped formatters and noise filters stay.
                log_level=getattr(a, "log_level", None),
                log_config=patches.uvicorn_log_config(
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
