"""Interactive ``gmlx init`` - a guided front door over the same discovery and
scaffold machinery as the flag-driven path.

Bare ``gmlx init`` on a TTY runs :func:`run_wizard`: it walks the user through
picking model directories, scans them (header-only), lets them curate the
auto-synthesized ids (rename / drop / set a default / add aliases), then offers
the on-disk prompt cache, the optional STT / TTS / embeddings services (with an
offer to ``pip``-install the missing extra), a voice-chat (``gmlx talk``) step
when both audio services are on, and the residency knobs (idle TTL, per-request
token timeout). It previews the YAML and confirms before writing.

Every decision here has a mirror ``--flag`` on ``gmlx init`` (see
:func:`server._cmd_init`), so nothing the wizard does is unreachable from a
script; the wizard is the human affordance, not a separate code path - it ends
by handing a rendered config back for the shared writer to commit.

I/O goes through :class:`WizardIO` (one ``_read`` primitive built on
``input()`` / prompt_toolkit), which tests subclass to drive the flow headlessly
- mirroring the chat REPL's scripted-input seam.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from . import discovery, embeddings, extras, rerank, stt, tts
from .config import DiscoverSpec, ModelCfg

# Common places people keep a local GGUF folder; the first that already exists is
# offered as the dir prompt's default. If none exist, ~/models is suggested anyway -
# accepting it writes server.model_dirs pointing there, and `gmlx pull` then lands
# files into it (init establishes the convention; pull reads it back).
_DIR_CANDIDATES = ("~/models", "~/gguf", "~/.lmstudio/models",
                   "~/.cache/lm-studio/models")
_FALLBACK_DIR = "~/models"

# (config key, human label, alias table, default alias, needs-ffmpeg, what-it-does).
# STT / TTS share the generic _configure_service flow; embeddings and rerank have
# dedicated flows (tiered presets, a quant follow-up, and auto-pickup of a
# retrieval GGUF already on disk) - see _configure_embeddings / _configure_rerank.
_SERVICES = (
    ("stt", "speech-to-text (POST /v1/audio/transcriptions)",
     stt.STT_ALIASES, stt.DEFAULT_STT_ALIAS, True,
     "transcribe uploaded audio with mlx-whisper"),
    ("tts", "text-to-speech (POST /v1/audio/speech)",
     tts.TTS_ALIASES, tts.DEFAULT_TTS_ALIAS, True,
     "synthesize speech from text with mlx-audio"),
)


@dataclass
class Outcome:
    """What the wizard hands back for the shared writer to commit."""
    out: Path
    text: str
    models: list = field(default_factory=list)


# Console I/O (test seam)
class WizardIO:
    """Line-oriented console I/O for the wizard. Everything funnels through
    :meth:`_read`; tests subclass and override it to script the session."""

    def __init__(self, reader=None):
        self._reader = reader            # a prompt_toolkit prompt callable, or None

    def _read(self, prompt: str) -> str:
        if self._reader is not None:
            try:
                return self._reader(prompt)
            except EOFError:
                return ""
        try:
            return input(prompt)
        except EOFError:
            return ""

    def note(self, msg: str = "") -> None:
        print(msg)

    def text(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default not in (None, "") else ""
        raw = self._read(f"{prompt}{suffix}: ").strip()
        return raw or (default or "")

    def yesno(self, prompt: str, default: bool = False) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            raw = self._read(f"{prompt} {hint} ").strip().lower()
            if not raw:
                return default
            if raw in ("y", "yes"):
                return True
            if raw in ("n", "no"):
                return False
            self.note("  please answer y or n")

    def choice(self, prompt: str, options: list, default: int = 0):
        """``options`` is a list of ``(label, value)``; returns the chosen value."""
        self.note(prompt)
        for i, (label, _v) in enumerate(options):
            mark = "  (default)" if i == default else ""
            self.note(f"  {i + 1}) {label}{mark}")
        while True:
            raw = self._read(f"  choice [1-{len(options)}] ").strip()
            if not raw:
                return options[default][1]
            # isdecimal, not isdigit: int() rejects superscripts like "2"'s
            # Unicode siblings that isdigit accepts, which would abort the wizard.
            if raw.isdecimal() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1][1]
            self.note("  enter a number from the list")


def make_io() -> WizardIO:
    """A :class:`WizardIO` upgraded to prompt_toolkit line editing when the
    ``[chat]`` extra is installed and stdin is a real terminal, else plain input."""
    import sys
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import prompt as ptk_prompt
            return WizardIO(reader=ptk_prompt)
        except ImportError:
            pass
    return WizardIO()


# Helpers
def parse_duration(value) -> float:
    """A duration like ``none`` / ``0`` / ``900`` / ``15m`` / ``1h`` -> seconds.
    ``none``/``never`` -> ``0`` (the disable sentinel). Raises ``ValueError`` on
    anything else - the caller surfaces it."""
    s = str(value).strip().lower()
    if s in ("none", "never"):
        return 0.0
    unit = 1.0
    if s and s[-1] in "smh":
        unit = {"s": 1.0, "m": 60.0, "h": 3600.0}[s[-1]]
        s = s[:-1]
    return float(s) * unit


def _ask_cache_gb(io, default: float = 50.0) -> float:
    """Follow-up for the on-disk cache cap once the disk tier is enabled. The cap is
    per model namespace, so the worst-case footprint is ~size * N_models. Loops
    until a positive number; bare Enter takes ``default``."""
    while True:
        raw = io.text(
            "  Maximum on-disk cache size in GB (per model - worst case "
            "~size x N models)", default=str(int(default)))
        try:
            gb = float(str(raw).strip().rstrip("gGbB ").strip())
            if gb > 0:
                return gb
        except ValueError:
            pass
        io.note("  enter a positive number of GB (e.g. 20, 50, 100)")


def _suggested_dir() -> str:
    """The first candidate dir that already exists, else the generic fallback."""
    for d in _DIR_CANDIDATES:
        if os.path.isdir(os.path.expanduser(d)):
            return d
    return _FALLBACK_DIR


def _hf_cache_has_gguf() -> bool:
    """True if the local HF cache holds any GGUF (metadata-only check - no header
    reads), so the wizard only offers the cache scan when it would find something."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
    except Exception:
        return False
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            if any(f.file_name.lower().endswith(".gguf") for f in rev.files):
                return True
    return False


def _print_model_table(io: WizardIO, models: list) -> None:
    if not models:
        io.note("  (no models)")
        return
    width = max(len(m.id) for m in models)
    for i, mc in enumerate(models, 1):
        flags = []
        if getattr(mc, "mmproj", None):
            flags.append("vlm")
        if getattr(mc, "speculative", False):
            flags.append("mtp")
        fam = getattr(mc, "family", None)
        if fam:
            flags.append(fam)
        tag = f"  [{','.join(flags)}]" if flags else ""
        io.note(f"  {i:>2}) {mc.id:<{width}}  {os.path.basename(mc.path)}{tag}")


def _curate(io: WizardIO, models: list):
    """Let the user rename / drop discovered models, set a default, and add
    aliases. Returns ``(models, default_model, aliases)`` over the surviving ids."""
    default_model: str | None = None
    aliases: dict = {}
    ids = {m.id for m in models}
    io.note("\nDiscovered models:")
    _print_model_table(io, models)
    io.note(
        "\nKeep them as-is by pressing Enter, or adjust the list:\n"
        "  rename <#> <new-name>     give a model a shorter / clearer id\n"
        "  drop <#>                  leave it out of the config\n"
        "  default <#>               serve it when a request names no model\n"
        "  alias <name> <#>          add a friendly alias for it")
    while True:
        raw = io.text("edit (or Enter when done)").strip()
        if not raw or raw.lower() == "done":
            break
        parts = raw.split()
        cmd = parts[0].lower()
        try:
            if cmd == "rename" and len(parts) == 3:
                idx = int(parts[1]) - 1
                new = discovery._slug(parts[2])
                if not (0 <= idx < len(models)):
                    io.note("  no such row")
                    continue
                if not new or (new in ids and new != models[idx].id):
                    io.note(f"  id {new!r} is empty or already taken")
                    continue
                old = models[idx].id
                ids.discard(old)
                models[idx].id = new
                ids.add(new)
                if default_model == old:
                    default_model = new
                aliases = {n: (new if t == old else t) for n, t in aliases.items()}
            elif cmd == "drop" and len(parts) == 2:
                idx = int(parts[1]) - 1
                if not (0 <= idx < len(models)):
                    io.note("  no such row")
                    continue
                gone = models.pop(idx).id
                ids.discard(gone)
                if default_model == gone:
                    default_model = None
                aliases = {n: t for n, t in aliases.items() if t != gone}
            elif cmd == "default" and len(parts) == 2:
                idx = int(parts[1]) - 1
                if not (0 <= idx < len(models)):
                    io.note("  no such row")
                    continue
                default_model = models[idx].id
            elif cmd == "alias" and len(parts) == 3:
                name = discovery._slug(parts[1])
                idx = int(parts[2]) - 1
                if not name or not (0 <= idx < len(models)):
                    io.note("  bad alias name or row")
                    continue
                if name in ids:
                    io.note(f"  {name!r} collides with a model id")
                    continue
                aliases[name] = models[idx].id
            else:
                io.note("  didn't catch that - use: rename <#> <new-name>, "
                        "drop <#>, default <#>, alias <name> <#> (Enter = done)")
                continue
        except ValueError:
            io.note("  the <#> must be a row number from the list above")
            continue
        _print_model_table(io, models)
    return models, default_model, aliases


def _profiles_step(io: WizardIO, models: list) -> None:
    """Summarize each model's detected sampling family (model-card defaults +
    built-in intents), then offer to pin a default intent per family that has
    card-specific intents. Pinning sets ``mc.profile`` in place; the scaffold
    renders it. No prompt appears when no family has intents to pin."""
    if not models:
        return
    from . import profiles as fam_profiles
    by_fam: dict = {}
    for mc in models:
        by_fam.setdefault(getattr(mc, "family", None) or "default", []).append(mc)
    io.note("\nSampling comes from each model's family card - no setup needed:")
    for fam in sorted(by_fam):
        entry = fam_profiles.FAMILIES.get(fam, fam_profiles.FAMILIES["default"])
        base = entry["base"].get("sampling", {})
        vals = " ".join(f"{k}={v}" for k, v in base.items())
        intents = sorted(entry["intents"])
        extra = f"   intents: {', '.join(intents)}" if intents else ""
        io.note(f"  {fam}: {len(by_fam[fam])} model(s) - {vals}{extra}")
    io.note("Requests switch intent per call: `<id>@coding` as the model name, "
            "a `profile`\nfield, or `run/chat --profile coding`. Full table: "
            "`gmlx profiles`.")
    eligible = [f for f in sorted(by_fam)
                if f != "default" and fam_profiles.FAMILIES.get(f, {}).get("intents")]
    if not eligible:
        return
    if not io.yesno("Pin a default intent for a family now? (requests can still "
                    "switch per-call)", default=False):
        return
    for fam in eligible:
        intents = sorted(fam_profiles.FAMILIES[fam]["intents"])
        pick = io.choice(
            f"Default intent for the {len(by_fam[fam])} {fam} model(s):",
            [("family default (recommended)", None)] + [(n, n) for n in intents],
            default=0)
        if pick:
            for mc in by_fam[fam]:
                mc.profile = pick


def _configure_service(io: WizardIO, key, label, alias_table, default_alias,
                       needs_ffmpeg, blurb, *, allow_install: bool):
    """Describe one service (what it does, what it installs, its default model),
    ask whether to configure it; on yes, pick a model and (optionally) install its
    extra. Returns the chosen model value, or ``None`` if skipped. Used for STT /
    TTS - embeddings and rerank have their own richer flows."""
    pkgs = ", ".join(extras.extra_packages(key))
    io.note(f"\n{label}")
    io.note(f"  {blurb}; default model `{default_alias}`.")
    ffmpeg = " + ffmpeg on PATH" if needs_ffmpeg else ""
    io.note(f"  installs the [{key}] extra ({pkgs}){ffmpeg} if not already present.")
    if not io.yesno("Configure it?", default=False):
        return None
    io.note(f"  presets: {', '.join(alias_table)}  "
            "(or an HF repo id / local path)")
    model = io.text(f"  {key} model", default=default_alias)
    if not extras.extra_installed(key):
        pkgs = " ".join(extras.extra_packages(key))
        if allow_install and io.yesno(
                f"  the {key} extra ({pkgs}) isn't installed - install it now?",
                default=False):
            ok = extras.install_extra(key)
            io.note("  installed." if ok else
                    f"  install failed - configure anyway; retry later with: "
                    f"pip install 'gmlx[{key}]'")
        else:
            io.note(f"  not installed - the endpoint errors until you run: "
                    f"pip install 'gmlx[{key}]'")
    if needs_ffmpeg and not extras.ffmpeg_present():
        io.note("  note: audio needs ffmpeg on PATH - `brew install ffmpeg`")
    return model


# Retrieval services (embeddings / rerank): tiered presets + a quant follow-up +
# auto-pickup of a retrieval GGUF the dir scan already found.
def _ctx_label(ctx: int) -> str:
    """A compact context-window label: 32768 -> ``32k``, 2048 -> ``2k``."""
    k = ctx / 1024
    return f"{int(k)}k" if k == int(k) else f"{k:.0f}k"


def _preset_row(n: int, p: dict) -> str:
    """One numbered guidance row for an embedding preset (dim / ctx / size / blurb).
    Size is the default rung's on-disk GB - the quant follow-up refines it."""
    size = p["sizes"][p["default_quant"]]
    return (f"    {n}) {p['alias']:<16} {p['dim']:>4}-dim  "
            f"{_ctx_label(p['ctx']):>4} ctx  {size:>4.1f} GB   {p['blurb']}")


def _offer_found(io: WizardIO, found, label: str):
    """If ``found`` (classified retrieval GGUFs already on disk) is non-empty, offer
    to adopt one. Returns the chosen file path, or ``None`` to fall through to the
    preset picker."""
    if not found:
        return None
    plural = "s" if len(found) > 1 else ""
    io.note(f"\n  found {label}{plural} you already have:")
    for i, c in enumerate(found, 1):
        tag = f"  [{c.arch}]" if c.arch else ""
        io.note(f"    {i}) {os.path.basename(c.path)}{tag}")
    if len(found) == 1:
        return found[0].path if io.yesno(
            f"  use this {label}?", default=True) else None
    raw = io.text(f"  use one of these {label}s? (number, or Enter to skip)",
                  default="").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(found):
        return found[int(raw) - 1].path
    return None


def _pick_embedding(io: WizardIO):
    """Print the tiered guidance table and read a choice. Returns a preset dict, or
    a raw string (a custom HF repo id / local path the user typed)."""
    io.note("\nPick an embedding model "
            "(dim = vector width, ctx = max input tokens):")
    rows: list = []
    io.note("  GGUF embedders (run on this runtime's loader):")
    for p in embeddings.EMBEDDING_PRESETS:
        if p["tier"] == "gguf":
            rows.append(p)
            io.note(_preset_row(len(rows), p))
    io.note("  safetensors encoders (via mlx-embeddings):")
    for p in embeddings.EMBEDDING_PRESETS:
        if p["tier"] == "mlx":
            rows.append(p)
            io.note(_preset_row(len(rows), p))
    io.note("  (or type an HF repo id / local path)")
    n = len(rows)
    while True:
        raw = io.text(f"  embedding model [1-{n} or repo/path]",
                      default="1").strip()
        if raw.isdigit():
            if 1 <= int(raw) <= n:
                return rows[int(raw) - 1]
            io.note(f"  enter a number from 1 to {n}")
            continue
        return raw                              # custom HF repo id / local path


def _pick_quant(io: WizardIO, preset: dict, *, prefer=None) -> str:
    """Offer ``preset``'s quant rungs and return the chosen rung name. When the
    preset has only one rung (e.g. the 0.6B embedder ships Q8_0/f16 only) the menu
    collapses. ``prefer`` (the embedder's chosen rung, for the reranker) is the
    pre-selected default when the preset has it, else the preset's own default."""
    rungs = list(preset["quants"])
    if len(rungs) == 1:
        return rungs[0]
    default = prefer if prefer in preset["quants"] else preset["default_quant"]
    opts = [(f"{r:<7} {preset['sizes'][r]:>4.1f} GB", r) for r in rungs]
    return io.choice(
        f"\nQuant for {preset['alias']} "
        "(smaller = less RAM / disk, slightly lower quality):",
        opts, default=rungs.index(default))


def _maybe_install_embeddings_extra(io: WizardIO, value: str, *, allow_install):
    """Offer to install the [embeddings] extra unless ``value`` is a GGUF ref (the
    decoder-LM backend needs no extra). ``value`` is the already-resolved concrete
    ref/repo, so the GGUF test is correct - the old bug tested a raw alias."""
    if embeddings._is_gguf_ref(value):
        return                                  # GGUF decoder-LM backend - no extra
    if extras.extra_installed("embeddings"):
        return
    pkgs = " ".join(extras.extra_packages("embeddings"))
    if allow_install and io.yesno(
            f"  the embeddings extra ({pkgs}) isn't installed - install it now?",
            default=False):
        ok = extras.install_extra("embeddings")
        io.note("  installed." if ok else
                "  install failed - configure anyway; retry later with: "
                "pip install 'gmlx[embeddings]'")
    else:
        io.note("  not installed - the endpoint errors until you run: "
                "pip install 'gmlx[embeddings]'")


def _configure_embeddings(io: WizardIO, *, allow_install: bool, found):
    """Configure text embeddings: adopt an embedder GGUF already on disk, else pick
    a tiered preset + quant (or a custom repo/path), installing the [embeddings]
    extra when a safetensors encoder needs it. Returns ``(value, quant_rung)`` -
    ``value`` is the config string (hf: ref / repo id / path) or ``None`` if
    skipped; ``quant_rung`` is the chosen rung (so the reranker can match it) or
    ``None`` for a custom / adopted pick."""
    io.note("\ntext embeddings (POST /v1/embeddings)")
    io.note("  a local RAG embedder - vector search / retrieval for Open WebUI "
            "and other clients.")
    adopt = _offer_found(io, found, "embedder")
    if adopt is not None:
        return adopt, None
    if not io.yesno("Configure text embeddings?", default=False):
        return None, None
    pick = _pick_embedding(io)
    if isinstance(pick, str):                   # custom HF repo id / local path
        _maybe_install_embeddings_extra(io, pick, allow_install=allow_install)
        return pick, None
    rung = _pick_quant(io, pick)
    value = pick["quants"][rung]
    if pick["tier"] == "mlx":
        _maybe_install_embeddings_extra(io, value, allow_install=allow_install)
    return value, rung


def _configure_rerank(io: WizardIO, *, found, embed_quant):
    """Configure reranking (POST /v1/rerank): adopt a reranker GGUF already on
    disk, else pick a Qwen3-Reranker size + quant. The quant defaults to
    ``embed_quant`` (the embedder's chosen rung) when that rung exists, else the
    reranker default. Never needs an extra. Returns the config value or ``None``."""
    io.note("\nreranking (POST /v1/rerank)")
    io.note("  re-scores retrieved passages against the query (RAG stage 2) - "
            "Open WebUI's external reranker. A Qwen3-Reranker GGUF; no extra.")
    adopt = _offer_found(io, found, "reranker")
    if adopt is not None:
        return adopt
    if not io.yesno("Configure reranking?", default=False):
        return None
    opts = [(f"{p['label']} - {p['blurb']}", p) for p in rerank.RERANK_PRESETS]
    preset = io.choice("\nPick a reranker size:", opts, default=0)
    rung = _pick_quant(io, preset, prefer=embed_quant)
    return preset["quants"][rung]


def _configure_talk(io: WizardIO, *, stt_model, tts_model,
                    allow_install: bool):
    """Offer the ``gmlx talk`` voice-chat client - only when both STT and TTS
    were just configured (the voice loop needs both). Picks a voice (enumerated
    from the TTS model when it is already cached locally), a wake phrase, a
    listening mode, and the menu bar tap-to-talk hotkey modifier; offers the
    [talk] extra (client audio + wake word). Returns a ``{voice, wake_word,
    mode, push_to_talk_modifier}`` dict for the scaffold, or ``None``."""
    if not (stt_model and tts_model):
        return None
    io.note("\nvoice chat (`gmlx talk`)")
    io.note("  a hands-free voice loop on this server: wake word -> transcribe "
            "-> chat -> speak.")
    if not io.yesno("Set up voice chat?", default=True):
        return None
    voices = tts.available_voices(tts.resolve_tts_model(tts_model))
    default_voice = (tts.DEFAULT_VOICE
                     if not voices or tts.DEFAULT_VOICE in voices
                     else voices[0])
    if voices:
        preview = ", ".join(voices[:8]) + (", ..." if len(voices) > 8 else "")
        io.note(f"  voices: {preview}")
    voice = io.text("  voice", default=default_voice)
    wake = io.text("  wake phrase (any text)", default="hey assistant")
    mode = io.choice(
        "  listen mode:",
        [("wake word - mic waits for the phrase, hands-free", "wake"),
         ("open mic - any speech starts a turn", "vad"),
         ("push-to-talk - Space in the terminal client starts / ends a turn",
          "ptt")],
        default=0)
    io.note("  separate from the mode above, the menu bar app offers a global "
            "tap-to-talk hotkey:\n  <modifier>+Space from any app (enable it "
            "from the menu bar).")
    # Options derive from the canonical tuple so the wizard can never write a
    # modifier the config loader rejects.
    from .hotkey import PUSH_TO_TALK_MODIFIERS
    labels = {"globe": "Globe/fn - Apple keyboards",
              "right-command": "Right Command - for keyboards without a "
                               "Globe key",
              "right-option": "Right Option",
              "control": "Control"}
    modifier = io.choice(
        "  hotkey modifier:",
        [(labels.get(m, m), m) for m in PUSH_TO_TALK_MODIFIERS],
        default=0)
    if not extras.extra_installed("talk"):
        pkgs = " ".join(extras.extra_packages("talk"))
        if allow_install and io.yesno(
                f"  the talk extra ({pkgs}) isn't installed - install it now?",
                default=False):
            ok = extras.install_extra("talk")
            io.note("  installed." if ok else
                    "  install failed - configure anyway; retry later with: "
                    "pip install 'gmlx[talk]'")
        else:
            io.note("  not installed - `gmlx talk` errors until you run: "
                    "pip install 'gmlx[talk]'")
    return {"voice": voice, "wake_word": wake, "mode": mode,
            "push_to_talk_modifier": modifier}


# The wizard
def run_wizard(*, default_out, io: WizardIO | None = None,
               seed_dirs=None, allow_install: bool = True,
               port=None) -> Outcome | None:
    """Walk the guided init flow and return an :class:`Outcome` to write, or
    ``None`` if the user cancels at the final confirm. ``default_out`` is the path
    a bare ``gmlx serve`` would find; ``seed_dirs`` pre-seeds the dir prompt (from
    ``-i --models-dir``). ``allow_install`` gates the pip-install offers; ``port``
    (from ``-i --port``) overrides the scaffold's server.port."""
    io = io or make_io()
    io.note("gmlx init - let's build a server config.\n")

    # 1. Model directories + scan.
    default_dir = (seed_dirs or [None])[0] or _suggested_dir()
    answer = io.text("Directory of GGUFs to scan (space-separated for several, "
                     "quote paths with spaces)", default=default_dir)
    # shlex keeps the space-separated multi-dir affordance while letting a
    # quoted "~/My Models" through in one piece.
    dirs = shlex.split(answer) if answer else []
    recursive = io.yesno("Recurse into subdirectories?", default=True)

    models: list[ModelCfg] = []
    if dirs:
        specs = [DiscoverSpec(dir=d, recursive=recursive) for d in dirs]
        models += discovery.scan_dirs(specs, dirs, progress=True)

    # 2. Hugging Face cache (only when it actually holds GGUFs).
    hf_cache = False
    if _hf_cache_has_gguf() and io.yesno(
            "\nAlso include GGUFs from your Hugging Face cache?", default=False):
        models += discovery.scan_hf_cache(
            known_ids={m.id for m in models}, progress=True)
        hf_cache = True

    # 3. Curate ids (rename / drop / default / aliases).
    default_model, aliases = None, {}
    if models:
        models, default_model, aliases = _curate(io, models)
    else:
        io.note("\nNo GGUFs found yet - writing a valid zero-model config; "
                "`gmlx pull` some in, then `gmlx sync-models`.")

    # 3.5. Sampling families: model-card defaults + optional pinned intent.
    _profiles_step(io, models)

    # 4. Prompt-cache SSD tier. The in-memory prompt cache is on in every
    # generated config; this decides only whether it also persists to disk.
    disk_cache = io.yesno(
        "\nThe prompt cache reuses prompt prefixes across requests (on by "
        "default).\nAlso persist it to disk at ~/.cache/gmlx/apc, so reuse "
        "survives an\nidle-unload or restart?",
        default=False)
    disk_cache_gb = _ask_cache_gb(io) if disk_cache else None

    # 5. Optional services (+ install). STT / TTS use the generic flow; embeddings
    # and rerank have richer flows with auto-pickup of a retrieval GGUF already on
    # disk (a header-only re-scan of the same dirs - cheap).
    svc_values = {}
    for key, label, table, default_alias, needs_ffmpeg, blurb in _SERVICES:
        svc_values[key] = _configure_service(
            io, key, label, table, default_alias, needs_ffmpeg, blurb,
            allow_install=allow_install)
    found_emb, found_rerank = (
        discovery.find_retrieval_models(dirs, recursive=recursive)
        if dirs else ([], []))
    emb_value, emb_quant = _configure_embeddings(
        io, allow_install=allow_install, found=found_emb)
    rerank_value = _configure_rerank(
        io, found=found_rerank, embed_quant=emb_quant)
    talk_value = _configure_talk(
        io, stt_model=svc_values["stt"], tts_model=svc_values["tts"],
        allow_install=allow_install)

    # 6. Residency & limits.
    io.note("\nUnloading frees the model and its in-memory KV cache. If the "
            "on-disk prompt\ncache is enabled, cached prefixes survive there and "
            "reload quickly.")
    ttl_s = io.choice(
        "Auto-unload an idle model after:",
        [("10 minutes", 600), ("15 minutes", 900), ("30 minutes", 1800),
         ("1 hour", 3600),
         ("never (keep resident; evict manually or under memory pressure)", 0)],
        default=1)
    io.note("\nA single request (prefill + generation) can be slow for some "
            "model + context\ncombinations. Some cases may need significant time "
            "to generate the first token.\nThis caps the wait for the NEXT token "
            "before the server gives up (each token\nresets it, so a healthy long "
            "reply is never cut off).")
    timeout_s = io.choice(
        "Give up on a request if no new token arrives for:",
        [("10 minutes", 600), ("30 minutes", 1800), ("1 hour", 3600),
         ("4 hours", 14400), ("never (wait forever)", 0)],
        default=1)   # matches the config-less server default (1800s)

    # 7. Output path (+ overwrite).
    user_out = Path(os.path.expanduser(str(default_out)))
    where = io.choice(
        "\nWhere should the config live?",
        [(f"user config ({default_out}) - found by `gmlx serve`", "user"),
         ("project-local (./gmlx.yaml)", "project")],
        default=0)
    out = user_out if where == "user" else Path("gmlx.yaml").resolve()
    if out.exists() and not io.yesno(f"\n{out} exists - overwrite?", default=False):
        io.note("aborted (existing config left in place).")
        return None

    # 8. Preview + confirm.
    text = discovery.scaffold_yaml(
        models, model_dirs=dirs, hf_cache=hf_cache, disk_cache=disk_cache,
        disk_cache_gb=disk_cache_gb,
        stt=svc_values["stt"], tts=svc_values["tts"],
        embeddings=emb_value, rerank=rerank_value, default_model=default_model,
        aliases=aliases, ttl_s=ttl_s, token_queue_timeout_s=timeout_s,
        talk=talk_value, port=port)
    io.note("\n----- config preview -----")
    io.note(text)
    io.note("----- end preview -----")
    if not io.yesno(f"Write this to {out}?", default=True):
        io.note("aborted (nothing written).")
        return None
    return Outcome(out=out, text=text, models=models)
