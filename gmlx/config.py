"""Server configuration model for ``gmlx serve``.

A composable YAML config describes the models the server serves and the reusable
**profiles** (sampling + load + cache params) applied to them. This module is the
pure-Python core - dataclasses, the YAML loader, the ``extends`` / ``rules`` /
precedence merge, and path/env resolution. It imports nothing heavy (no mlx-vlm,
no mlx), so it loads and tests on any machine.

Shape (see ``docs/config.md`` for the full reference)::

    server:    {host, port, api_key, no_auth, media_urls, cors_origins, model_dirs, budget_gb, max_models, hf_cache, cache, defaults, stt, tts, embeddings, rerank, systemone, menubar, token_queue_timeout_s, prefill_step_size, dtype, decode_prefill_ratio, prefill_tick_ms, cache_limit_gb, family_defaults, stochastic_mtp, gpu_keepwarm, assistants, assistant_allow_remote}
    profiles:  {<name>: {extends, sampling, load, cache, system}}
    rules:     [{match: <glob>, profile: <name>}]
    models:    {<id>: {path, profile, family, profiles, mmproj, draft_gguf, adapter, stream, moe_experts, moe_expert_mass, moe_miss_shed, moe_layer_shed, moe_prestage, stream_fast_disk, speculative, speculative_width_cap, overrides, pin, ttl_s}}
    aliases:   {<name>: <id> | <id>@<profile>}    # friendly name / profile preset
    discover:  [{dir, recursive, pair_mmproj, speculative}]
    talk:      {model, voice, speed, system, language, max_tokens, mode, wake_word, wake_threshold, vad, input_device, output_device, chime, brain, push_to_talk_modifier}
    launch:    {container: {enabled, mount_cwd, mounts, volumes, forward, network, cpus, memory, ssh_agent, env, open_browser, clipboard, clients}}
    assistant: {max_tool_rounds, tool_timeout_s, mcp, memory}   # shared tool-loop assistant
    theme:     <name>                             # chat default theme (--theme overrides)
    themes:    {<name>: {<slot>: {bold, dim, italic, underline, fg16, rgb}, extends, code_theme, ptk_toolbar}}

Precedence (low -> high) for the param groups of a request:
``family base (built-in, see profiles.py) -> server.defaults.profile -> matched
rule.profile -> model.profile (+extends) -> model.profiles[<selected>] tweak ->
model.overrides -> per-request fields``. A request ``@profile`` (inline in the model
string) replaces the model's configured profile in that chain and may name a
built-in intent (``coding``, ``creative``, ...); a user profile with the same name
shadows the built-in. ``server.family_defaults: false`` removes the built-in layer
and names. Per-request fields are applied later, at the gen-args seam
(``server_patches``), so they are not modelled here.
"""

from __future__ import annotations

import contextlib
import errno
import fnmatch
import functools
import io
import os
import re
import secrets
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import gmlx.gen.profiles as _family_profiles
from gmlx.systemone.extensions import THINK_BUDGET, THINK_THRESHOLD
from .cache.kv_policy import SCHEMES as KV_QUANT_SCHEMES
from .envflags import env_bool
from .safe_path import canonical, fd_path, path_inside

# Canonical key sets / env mappings
# Sampling keys a profile may carry, as a plain dict so profiles compose by
# dict-merge. Most are request fields mlx-vlm honours directly (injected at the
# gen-args seam); `stop` and the xtc_* keys are honoured by gmlx's own
# server seams (server_patches) - mlx-vlm has no native support for them.
SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "max_tokens", "seed",
    "repetition_penalty", "presence_penalty", "frequency_penalty",
    "repetition_context_size", "enable_thinking", "thinking_budget",
    "thinking_start_token", "thinking_end_token",
    "stop", "xtc_probability", "xtc_threshold",
})

# Load-param key -> the env var mlx-vlm's server reads at model-build time
# (server/generation.py getters). Applied through the residency env window at load.
LOAD_ENV = {
    "kv_bits": "KV_BITS",
    "kv_group_size": "KV_GROUP_SIZE",
    "kv_quant_scheme": "KV_QUANT_SCHEME",
    "kv_tail_tokens": "KV_TAIL_TOKENS",
    "max_kv_size": "MAX_KV_SIZE",
    "quantized_kv_start": "QUANTIZED_KV_START",
}

# Accepted `server.dtype` values. float32 is deliberately absent: it is a
# certification reference arm reachable through the env var, not something a
# server should be configured into. Validated at parse time, because an
# unrecognized value would otherwise fall back to bfloat16 at load with
# nothing said about it.
SERVER_DTYPES = ("auto", "bfloat16", "bf16", "float16", "fp16")

# Accepted `load.kv_quant_scheme` values come from the policy resolver
# that implements them (KV_QUANT_SCHEMES, imported above). Other values
# are refused at parse time: an unchecked one reaches mlx-vlm and builds
# caches no gmlx path can read.

# APC prompt-cache (+ SSD disk tier) key -> env var (mlx-vlm apc.from_env). The disk
# sub-block maps to APC_DISK_*; without a namespace, each model's path is its namespace.
CACHE_ENV = {
    "enabled": "APC_ENABLED",
    "block_size": "APC_BLOCK_SIZE",
    "num_blocks": "APC_NUM_BLOCKS",
    "exact_entries": "APC_EXACT_CACHE_ENTRIES",
    "hash": "APC_HASH",
}

# In-memory exact-prefix snapshots kept per model (hybrid/recurrent archs use these;
# pure-attention archs use the block cache instead). mlx-vlm defaults to 2, which is
# low for a multi-turn / multi-conversation session - a third distinct prefix evicts
# the first. We raise the default when APC is on; each entry is a full prompt-cache
# clone, so it's memory for reuse. Override per config with cache.exact_entries.
DEFAULT_EXACT_CACHE_ENTRIES = 4
def default_apc_disk_path() -> str:
    """Where `disk: true` (the boolean shorthand for the SSD tier) puts the
    cache: under ``$XDG_CACHE_HOME``, like the other gmlx caches."""
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache",
                        "gmlx", "apc")


CACHE_DISK_ENV = {
    "path": "APC_DISK_PATH",
    "max_gb": "APC_DISK_MAX_GB",
    "workers": "APC_DISK_WORKERS",
    "read_mode": "APC_DISK_READ_MODE",
    "namespace": "APC_DISK_NAMESPACE",
}

# The complete, documented key surface of each config namespace. A key outside its
# set is a typo or an unsupported knob; since the parsers read with .get(), such a
# key would otherwise be silently dropped (e.g. `pinned:` instead of `pin:` quietly
# leaves a model unpinned) - so we warn loudly instead. Structural breakage (missing
# path, unknown profile reference, extends cycle) is what *raises*; see _validate.
_TOP_KEYS = frozenset({"server", "profiles", "rules", "models", "aliases",
                       "discover", "talk", "assistant", "theme", "themes",
                       "launch"})
_SERVER_KEYS = frozenset({"host", "port", "api_key", "no_auth", "media_urls",
                          "cors_origins", "model_dirs",
                          "budget_gb", "max_models", "hf_cache", "cache",
                          "defaults", "stt", "tts", "embeddings", "rerank",
                          "systemone", "menubar", "token_queue_timeout_s", "prefill_step_size",
                          "dtype",
                          "decode_prefill_ratio", "prefill_tick_ms",
                          "cache_limit_gb", "family_defaults", "stochastic_mtp",
                          "gpu_keepwarm", "assistants", "assistant_allow_remote"})
_DEFAULTS_KEYS = frozenset({"profile", "ttl_s", "model", "preload"})
_SYSTEMONE_KEYS = frozenset({"model", "canvas", "constrained", "max_questions",
                             "max_samples", "think", "think_threshold",
                             "think_budget"})
_PROFILE_KEYS = frozenset({"extends", "sampling", "load", "cache", "system",
                           "chat_template", "chat_template_kwargs",
                           "thinking", "reasoning_effort"})
_OVERRIDE_KEYS = frozenset({"sampling", "load", "cache", "system",
                            "chat_template", "chat_template_kwargs",
                            "thinking", "reasoning_effort"})
# chat_template_kwargs keys that name a parameter of the template call rather
# than a template variable. chat_template would swap the model's template for
# the caller's Jinja source, and the rest change what the call returns or
# collide with the arguments the server passes itself.
TEMPLATE_CALL_KEYS = frozenset({
    "chat_template", "conversation", "messages", "prompt", "processor", "config",
    "tools", "tool_choice", "documents", "add_generation_prompt",
    "continue_final_message", "tokenize", "padding", "truncation", "max_length",
    "return_tensors", "return_dict", "return_assistant_tokens_mask",
    "tokenizer_kwargs", "return_messages", "num_images", "num_audios", "video",
    "max_pixels", "fps", "audio_token", "processor_kwargs",
    "load_audio_from_video"})
_MODEL_KEYS = frozenset({"path", "profile", "family", "profiles", "mmproj",
                         "draft_gguf", "native_mtp", "adapter", "stream",
                         "cpu_moe",  # deprecated alias for `stream:`
                         "moe_experts", "moe_expert_mass",
                         "moe_miss_shed", "moe_layer_shed", "moe_prestage",
                         "prefill_feeder", "decode_feeder",
                         "stream_fast_disk", "speculative",
                         "speculative_width_cap",
                         "overrides", "pin", "ttl_s"})
_RULE_KEYS = frozenset({"match", "profile"})
_DISCOVER_KEYS = frozenset({"dir", "recursive", "pair_mmproj", "speculative"})
_CACHE_KEYS = frozenset(set(CACHE_ENV) | {"disk"})
_TALK_KEYS = frozenset({"model", "voice", "speed", "system", "language",
                        "max_tokens", "mode", "wake_word", "wake_threshold",
                        "vad", "input_device", "output_device", "chime",
                        "brain", "push_to_talk_modifier"})
_TALK_VAD_KEYS = frozenset({"threshold", "silence_ms", "min_speech_ms",
                            "pre_roll_ms"})
_ASSISTANT_KEYS = frozenset({"max_tool_rounds", "tool_timeout_s", "mcp",
                             "memory"})
_MCP_KEYS = frozenset({"name", "command", "url", "env"})
_MEMORY_KEYS = frozenset({"enabled", "path", "top_k", "extract",
                          "ttl_days", "max_items"})
_ASSISTANT_ALIAS_KEYS = frozenset({"model", "memory", "mcp"})
TALK_MODES = ("wake", "vad", "ptt", "text")
TALK_BRAINS = ("chat", "assistant")

# The clients `gmlx launch` runs, in the order its help lists them. The
# launch.container.clients keys must be one of these.
LAUNCH_CLIENTS = ("opencode", "pi", "omp", "hermes", "goose", "claude-code",
                  "aichat", "elia", "open-webui", "dsh")
LAUNCH_NETWORKS = ("default", "none")
LAUNCH_CLIPBOARD = ("off", "images")
_LAUNCH_KEYS = frozenset({"container"})
# Settings both levels take: the client value wins for a single value, and
# the two lists add up.
_LAUNCH_SHARED_KEYS = frozenset({"enabled", "mount_cwd", "mounts", "volumes",
                                 "forward", "network", "cpus", "memory",
                                 "ssh_agent", "env", "open_browser",
                                 "clipboard"})
_LAUNCH_CONTAINER_KEYS = _LAUNCH_SHARED_KEYS | {"clients"}
_LAUNCH_CLIENT_KEYS = _LAUNCH_SHARED_KEYS | {"image", "build", "command",
                                             "packages", "seed", "assistants"}
# Launch sets these guest variables itself. A second HOME would name a Mac
# path that is not shared.
LAUNCH_RESERVED_ENV = frozenset({"HOME", "TERM", "COLORTERM", "LANG", "TZ",
                                 "PATH", "SSH_AUTH_SOCK"})

# Host names that count as a loopback bind for the serve auth policy and the
# DNS-rebinding host guard (shared here because server.py must stay importable
# without the fastapi extra).
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_DEFAULT_PORTS = {"http": 80, "https": 443}
_ORIGIN_HOST = re.compile(r"[a-z0-9]([a-z0-9.-]*[a-z0-9.])?")
# The origin of a desktop app's own pages: a scheme other than http(s) and
# a plain name, as in tauri://localhost, app://. or file://.
_APP_ORIGIN = re.compile(r"([a-z][a-z0-9+.-]*)://([a-z0-9._-]*)")
# The schemes that Electron, Tauri and VS Code webview apps send in Origin.
# A web page in a browser cannot send one, so the server answers them as it
# answers a loopback page.
APP_ORIGIN_SCHEMES = frozenset({"app", "file", "tauri", "vscode-file", "vscode-webview"})


def _ipv6_origin_host(host: str) -> str:
    """An IPv6 address as a browser writes it in an origin: lower-case hex
    groups, the first longest run of two or more zero groups as ``::``, and
    never a dotted IPv4 tail."""
    import ipaddress

    packed = ipaddress.IPv6Address(host).packed
    groups = [format(int.from_bytes(packed[i:i + 2], "big"), "x") for i in range(0, 16, 2)]
    best, run = (0, 0), 0
    for i, g in enumerate(groups + ["end"]):
        if g == "0":
            run += 1
            continue
        if run > best[1]:
            best = (i - run, run)
        run = 0
    start, length = best
    if length < 2:
        return ":".join(groups)
    return ":".join(groups[:start]) + "::" + ":".join(groups[start + length:])


# The server.cors_origins entries that let every extension of one browser
# call the server. Safari gives an extension a new ID at each launch, so an
# entry with one Safari extension's ID lasts only until Safari quits.
EXTENSION_WILDCARDS = ("chrome-extension://*", "moz-extension://*",
                       "safari-web-extension://*")


def normalize_cors_entry(text: str) -> str:
    """A ``server.cors_origins`` entry: an origin normalized by
    :func:`normalize_origin`, or one of :data:`EXTENSION_WILDCARDS` in lower
    case. Raises ValueError for any other entry with a ``*``, saying what to
    write instead."""
    value = text.strip().lower()
    if value in EXTENSION_WILDCARDS:
        return value
    if "*" in value:
        wildcard = f"{value.partition('://')[0]}://*"
        if wildcard in EXTENSION_WILDCARDS:
            raise ValueError(f"to let every extension of this browser call the server, "
                             f"write {wildcard} with nothing after it")
        raise ValueError("a wildcard names no single origin, so list each site's origin, "
                         "such as https://chat.example.com. The only wildcard entries "
                         "are chrome-extension://*, moz-extension://* and "
                         "safari-web-extension://*, each of which lets every extension "
                         "of one browser call the server")
    return normalize_origin(text)


def normalize_origin(text: str) -> str:
    """The origin ``text`` in the form a browser sends it:
    ``scheme://host[:port]`` with the scheme and host in lower case and no
    default port, or, for a scheme other than http(s), ``scheme://name`` in
    lower case, the origin of a desktop app's pages. Raises ValueError
    naming the problem for anything else, including ``*`` and ``null``,
    which name no single origin."""
    import urllib.parse

    value = text.strip()
    if value in ("*", "null"):
        raise ValueError(f"{value} names no single origin")
    if not value.isascii():
        raise ValueError("write the host in its ASCII (punycode) form")
    app = _APP_ORIGIN.fullmatch(value.lower())
    if app and app.group(1) not in _DEFAULT_PORTS:
        return app.group(0)
    try:
        split = urllib.parse.urlsplit(value)
        port = split.port
    except ValueError:
        raise ValueError("it is not scheme://host[:port]") from None
    scheme = split.scheme             # urlsplit lowercases it
    if scheme not in _DEFAULT_PORTS:
        raise ValueError("an origin with a scheme other than http or https is only "
                         "scheme://name, such as tauri://localhost")
    if "@" in split.netloc:
        raise ValueError("an origin has no user name or password")
    if split.path not in ("", "/") or split.query or split.fragment \
            or value.endswith(("?", "#")):
        raise ValueError("an origin is only scheme://host[:port], with nothing after it")
    host = split.hostname or ""
    if not host:
        raise ValueError("it has no host")
    if split.netloc.startswith("["):
        try:
            host = f"[{_ipv6_origin_host(host)}]"
        except ValueError:
            raise ValueError(f"{host!r} is not an IPv6 address") from None
    elif not _ORIGIN_HOST.fullmatch(host):
        raise ValueError("the host must be a name or an address, with no wildcard")
    if port == 0:
        raise ValueError("port 0 is not a port a page can use")
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def origin_is_loopback(origin: str) -> bool:
    """Whether the normalized ``origin`` is a page on this machine's
    loopback: ``localhost``, 127.0.0.0/8 or ``::1``. Only a process on the
    machine can serve such a page."""
    import ipaddress
    import urllib.parse

    split = urllib.parse.urlsplit(origin)
    if split.scheme not in _DEFAULT_PORTS:
        return False
    host = split.hostname or ""
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped and mapped.is_loopback)


def origin_is_app(origin: str) -> bool:
    """Whether the normalized ``origin`` has one of the
    :data:`APP_ORIGIN_SCHEMES` that desktop apps send and web pages cannot."""
    return origin.partition("://")[0] in APP_ORIGIN_SCHEMES

# Bare-start config search order (first existing wins). The XDG-style
# ``~/.config`` location, where ``gmlx init`` writes, is the default, and the
# legacy ``~/.gmlx.yaml`` dotfile is a fallback. A ``./gmlx.yaml`` is never
# searched: a file in the folder you work in can name commands the server
# runs, and a container client can write that folder.
_DEFAULT_CONFIG_PATHS = (
    "~/.config/gmlx/gmlx.yaml",
    "~/.gmlx.yaml",
)

# The name a project-local config had while gmlx still searched for it.
_LOCAL_CONFIG = "gmlx.yaml"
_LOCAL_CONFIG_NOTE = ("gmlx no longer reads ./gmlx.yaml. Move it to "
                      "~/.config/gmlx/gmlx.yaml to use it.")
_local_config_noted = False

# Where ``gmlx init`` writes by default - the XDG-style location bare
# ``gmlx serve`` then finds via the search order above.
DEFAULT_CONFIG_WRITE = "~/.config/gmlx/gmlx.yaml"


class ConfigError(ValueError):
    """A malformed or self-inconsistent config (bad YAML, extends cycle, unknown
    profile reference, duplicate id, illegal id). Raised by :func:`load_config`."""


class MissingModelFile(ConfigError):
    """A model path/ref that parses fine but has no file behind it right now -
    disk state, not config shape. Consumers that can degrade catch this narrower
    type (serve skips the model with a warning instead of failing startup);
    everything else keeps seeing it as a :class:`ConfigError`."""


# Dataclasses
@dataclass
class Rule:
    match: str          # fnmatch glob, tested against the model id
    profile: str


@dataclass
class Profile:
    name: str
    extends: str | None = None
    sampling: dict = field(default_factory=dict)
    load: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)   # may carry a nested "disk" block
    system: str | None = None
    chat_template: str | None = None         # inline Jinja or path to .jinja/.txt
    # Extra variables passed to apply_chat_template per request (e.g.
    # {preserve_thinking: true} for Qwen3.6 / Gemma-4 agent turns). Applied at the
    # gen-args seam, so - unlike chat_template - not load-affecting.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Dedicated thinking controls, mapped per model at request time onto
    # whatever switch its chat template reads (thinking_mode /
    # enable_thinking / the Hy3 reasoning_effort dialect) - unlike
    # chat_template_kwargs entries, which pass through verbatim. thinking:
    # on|off|adaptive (bools accepted); reasoning_effort: a level name the
    # model's own template validates (low/medium/high, no_think, max, ...).
    thinking: Any = None
    reasoning_effort: str | None = None


@dataclass
class ModelCfg:
    id: str
    path: str
    profile: str | None = None
    # Sampling family (profiles.py key). Usually auto-detected from the GGUF
    # header at registration/scan; an explicit YAML `family:` wins (also the
    # escape hatch for a not-yet-pulled file or a mis-detected family).
    family: str | None = None
    # Per-model tweaks of NAMED profiles: {profile-or-intent-name: {sampling,
    # load, cache, system, chat_template, chat_template_kwargs}}. Merged when
    # that name is the selected profile for a request - e.g. reshape what
    # `@coding` means for this one model.
    profiles: dict = field(default_factory=dict)
    mmproj: str | None = None
    draft_gguf: str | None = None
    # Draft with the GGUF's own MTP head even when draft_gguf is set (an
    # external drafter otherwise wins).
    native_mtp: bool = False
    adapter: str | None = None          # GGUF LoRA adapter applied live at load
    # Execution placement (normalized by _normalize_stream): "experts" streams
    # only the routed-expert stacks from disk (every-token layers + KV cache
    # stay on GPU); "cpu" runs the whole model on the CPU device, all weights
    # streamed through the page cache. None = all-GPU.
    stream: Any = None
    # Lossy MoE fan-out levers on the streamed expert stacks (the
    # `--moe-experts K` / `--moe-expert-mass P` / `--moe-miss-shed P` /
    # `--moe-layer-shed P` CLI levers). Require stream; None = trained
    # fan-out / no shedding. `moe_prestage: keepers` retargets the lookahead
    # prestage through the miss-shed policy (needs `moe_miss_shed`).
    moe_experts: int | None = None
    moe_expert_mass: float | None = None
    moe_miss_shed: float | None = None
    moe_layer_shed: float | None = None
    moe_prestage: str | None = None
    # Streaming-model feeder overrides (tri-state: None = loader default -
    # prefill feeder on, decode feeder on under `stream: experts`).
    prefill_feeder: bool | None = None
    decode_feeder: bool | None = None
    # Streamed-decode prefetch recipe: auto (probe the drive), on, off.
    stream_fast_disk: str | None = None
    speculative: bool = False
    # Batch-width cap for speculative decode: MTP runs only while the live
    # decode batch is this wide, wider batches decode plain (drafter stays
    # loaded). None = the drafter family's measured default; 0 = uncapped;
    # N = cap. Clamped by a per-drafter hard limit (B=1-only drafters).
    speculative_width_cap: int | None = None
    overrides: dict = field(default_factory=dict)   # {sampling, load, cache, system}
    pin: bool = False
    ttl_s: float | None = None      # None => server default; 0 => never


@dataclass
class DiscoverSpec:
    dir: str | None = None       # None => scan server.model_dirs
    recursive: bool = False
    pair_mmproj: bool = True
    speculative: Any = "auto"       # "auto" | True | False


@dataclass
class ServerDefaults:
    profile: str | None = None
    ttl_s: float | None = 900.0  # idle auto-unload (15 min); None/0 => never
    model: str | None = None     # used when a request omits/empties `model`
    preload: object = None          # model ids to warm at startup; "all" | list


@dataclass(frozen=True)
class SystemoneCfg:
    """``server.systemone``: structured decisions on POST /v1/systemone.
    ``canvas``, ``constrained`` and the think keys apply to DiffusionGemma
    only; the letter readout on other models ignores them."""
    model: str | None = None     # used when the request's model is absent or unknown
    canvas: int = 64             # served canvas rows; a positive multiple of 16
    constrained: bool = True     # read over the label ids only
    max_questions: int = 64      # per request
    max_samples: int = 32        # per question: samples, or letter orderings
    think: int | str = 0         # request default: a budget, or "auto"
    think_threshold: float = THINK_THRESHOLD  # request default for "auto"
    think_budget: int = THINK_BUDGET          # request default for "auto"

    def request_defaults(self) -> dict:
        """The values a request takes for the think fields it omits."""
        return {"think": self.think, "think_threshold": self.think_threshold,
                "think_budget": self.think_budget}


@dataclass
class TalkVad:
    """Endpointing knobs for the ``gmlx talk`` listener."""
    threshold: float = 0.6      # speech probability above which a frame is speech
    silence_ms: float = 550.0   # trailing-silence hangover that ends an utterance
    min_speech_ms: float = 300.0  # utterances shorter than this are discarded
    pre_roll_ms: float = 400.0  # audio kept from before speech onset


@dataclass
class McpServerCfg:
    """One MCP server the assistant connects to for tools: a stdio ``command``
    (argv list) or a streamable-HTTP ``url`` - exactly one of the two."""
    name: str
    command: list = field(default_factory=list)  # stdio transport: argv
    url: str | None = None                    # HTTP transport: endpoint
    env: dict = field(default_factory=dict)      # stdio: extra environment


@dataclass
class AssistantMemory:
    """The assistant's long-term memory store (RAG over the server's own
    /v1/embeddings + /v1/rerank)."""
    enabled: bool = True
    path: str | None = None  # sqlite store; default: XDG data dir at runtime
    top_k: int = 4              # memories retrieved + injected per turn
    extract: bool = True        # distill turns into facts via the chat model
    ttl_days: float | None = None  # expire memories older than this
    max_items: int = 20000      # size cap; evicts least-recalled oldest


@dataclass
class AssistantCfg:
    """Top-level ``assistant:`` - tools + memory for the built-in tool-loop
    assistant used by ``gmlx talk`` (``talk.brain: assistant``), ``gmlx chat
    --assistant``, and ``server.assistants`` - not the external coding agents
    ``gmlx launch`` points at the server."""
    max_tool_rounds: int = 8      # tool-call round cap per turn
    tool_timeout_s: float = 60.0  # per tool invocation
    mcp: list = field(default_factory=list)      # [McpServerCfg]
    memory: AssistantMemory = field(default_factory=AssistantMemory)


@dataclass
class AssistantAlias:
    """One ``server.assistants:`` entry - a served pseudo-model id that wraps a
    configured model with the assistant tool loop, server-side."""
    model: str                  # underlying configured model id
    # Server-side memory is off by default. When enabled it is one shared
    # store across every client of this alias.
    memory: bool = False
    # None = inherit the full local-convenience `assistant.mcp` tool list;
    # scope remote-exposed aliases with an explicit list ([] = no tools).
    mcp: list | None = None  # [McpServerCfg] | None


# Default spoken-persona prompt: steers models away from markdown and
# symbols the TTS front-ends can't speak. `system: ""` in the talk block
# (or `--system ""`) opts out; any other value replaces it.
DEFAULT_TALK_SYSTEM = (
    "You are a helpful voice assistant. Your output is synthesized as "
    "speech, so avoid using markdown or any special characters."
)


@dataclass
class TalkCfg:
    """Client-side config for the ``gmlx talk`` voice loop (top-level ``talk:``
    block - it configures the *client*, not the server, so it does not live under
    ``server:``). Most fields have a CLI flag that overrides them
    (``pre_roll_ms`` and ``push_to_talk_modifier`` are config-only)."""
    model: str | None = None       # id[@profile]; default: the server's default
    voice: str | None = None       # TTS voice preset; default: server default
    speed: float = 1.0
    system: str | None = DEFAULT_TALK_SYSTEM
    language: str | None = None    # whisper language hint
    max_tokens: int | None = None     # None = until the model stops
    mode: str = "wake"                # wake | vad | ptt | text
    wake_word: str = "hey assistant"  # any text phrase (sherpa-onnx KWS)
    wake_threshold: float = 0.3
    # Menu bar global hotkey: <modifier>+Space (see gmlx.talk.hotkey's
    # PUSH_TO_TALK_MODIFIERS; right-side keys for Globe-less keyboards).
    push_to_talk_modifier: str = "globe"
    vad: TalkVad = field(default_factory=TalkVad)
    input_device: str | None = None   # sounddevice name substring or index
    output_device: str | None = None
    chime: bool = True
    brain: str = "chat"               # chat | assistant (tools + memory)
    # The shared top-level assistant: block, attached here by build_config so
    # talk consumers get one settings object (talk parses no assistant keys).
    assistant: AssistantCfg = field(default_factory=AssistantCfg)


@dataclass
class LaunchClientCfg:
    """One ``launch.container.clients.<client>`` block. ``None`` and empty
    lists mean unset, so the global value applies."""
    enabled: bool | None = None
    image: str | None = None          # an OCI reference; excludes build
    build: str | None = None          # a Containerfile or a context folder
    command: list[str] | str | None = None   # an argv list, or "image"
    mount_cwd: bool | None = None
    mounts: list[str] = field(default_factory=list)     # PATH[:DST][:ro]
    volumes: list[str] = field(default_factory=list)    # NAME:/path[:SIZE]
    forward: list[int] = field(default_factory=list)
    network: str | None = None
    cpus: int | None = None
    memory: str | None = None
    ssh_agent: bool | str | None = None   # true, false or an agent socket path
    env: list[str] = field(default_factory=list)        # NAME or NAME=VALUE
    open_browser: bool | None = None
    clipboard: str | None = None
    packages: list[str] = field(default_factory=list)   # Debian package names
    seed: list[str] = field(default_factory=list)       # files under $HOME
    assistants: list[str] = field(default_factory=list)  # server assistant alias ids


@dataclass
class LaunchContainerCfg:
    """The ``launch.container`` block: how ``gmlx launch --container`` runs a
    client in an Apple container. ``mount_cwd`` stays ``None`` when unset,
    because each client has its own built-in default."""
    enabled: bool = False
    mount_cwd: bool | None = None
    mounts: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    forward: list[int] = field(default_factory=list)
    network: str = "default"
    cpus: int = 4
    memory: str = "4G"
    ssh_agent: bool | str = False
    env: list[str] = field(default_factory=list)
    open_browser: bool = True
    clipboard: str = "off"
    clients: dict[str, LaunchClientCfg] = field(default_factory=dict)

    def for_client(self, client: str) -> LaunchClientCfg:
        """The effective settings of one client: its own value for each
        single setting, else the global one, and the global list followed by
        the client's, with exact duplicates dropped."""
        own = self.clients.get(client) or LaunchClientCfg()

        def pick(name):
            value = getattr(own, name)
            return getattr(self, name) if value is None else value

        def join(name):
            return list(dict.fromkeys([*getattr(self, name), *getattr(own, name)]))

        return LaunchClientCfg(
            enabled=pick("enabled"), image=own.image, build=own.build,
            command=own.command, mount_cwd=pick("mount_cwd"),
            mounts=join("mounts"), volumes=join("volumes"),
            forward=join("forward"), network=pick("network"), cpus=pick("cpus"),
            memory=pick("memory"), ssh_agent=pick("ssh_agent"), env=join("env"),
            open_browser=pick("open_browser"), clipboard=pick("clipboard"),
            packages=list(own.packages), seed=list(own.seed),
            assistants=list(own.assistants))


@dataclass
class LaunchCfg:
    """The top-level ``launch:`` block. ``gmlx launch`` reads it only from the
    user-level config (:func:`load_launch_settings`)."""
    container: LaunchContainerCfg = field(default_factory=LaunchContainerCfg)


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8080
    model_dirs: list[str] = field(default_factory=list)
    budget_gb: float | None = None
    max_models: int | None = None
    hf_cache: bool = False
    cache: dict = field(default_factory=dict)
    # Optional speech-to-text model (POST /v1/audio/transcriptions; needs the
    # `stt` extra). An alias (`whisper-turbo`), an HF repo id in MLX-whisper
    # format, a local model dir, or `true` for the default alias - resolved by
    # stt.resolve_stt_model at serve time.
    stt: str | None = None
    # Optional text-to-speech model (POST /v1/audio/speech; needs the `tts`
    # extra). An alias (`kokoro`), an HF repo id in MLX-audio format, a local
    # model dir, or `true` for the default alias - resolved by
    # tts.resolve_tts_model at serve time.
    tts: str | None = None
    # Optional text-embeddings model (POST /v1/embeddings). A GGUF decoder-LM
    # embedder (alias `qwen3-embed-0.6b`, a *.gguf path, or
    # hf:<org>/<repo>/<file>.gguf, loaded by the runtime - no extra), or an
    # mlx-embeddings safetensors encoder (alias `embeddinggemma`/`bge-m3`, an HF
    # MLX-embeddings repo, a local dir, or `true` for the default alias - needs
    # the `embeddings` extra) - resolved by embeddings.resolve_embeddings_model
    # at serve time.
    embeddings: str | None = None
    # Optional reranker model (POST /v1/rerank; Cohere/Jina shape). A Qwen3-Reranker
    # GGUF (alias `qwen3-rerank-0.6b`, a *.gguf path, or hf:<org>/<repo>/<file>.gguf)
    # - a causal Qwen3 LM scored by its yes/no logits, loaded by the runtime (no
    # extra). Resolved by rerank.resolve_rerank_model at serve time.
    rerank: str | None = None
    # POST /v1/systemone settings (structured decisions).
    systemone: SystemoneCfg = field(default_factory=SystemoneCfg)
    # Optional static API key: every endpoint except /health requires it
    # (Authorization: Bearer, or x-api-key). This config field is the sole
    # server-side source - there is no CLI flag or env override. A non-loopback
    # bind refuses to start with no key unless `no_auth` opts out explicitly
    # (for auth handled in front: mTLS, reverse proxy).
    api_key: str | None = None
    no_auth: bool = False
    # A request may name media by an http(s) URL, which the server then
    # fetches. File paths are refused either way (patches/media_gate.py).
    media_urls: bool = False
    # Origins, besides loopback and desktop-app ones, whose pages may call
    # the server (normalized by normalize_origin; patches/hardening.py).
    cors_origins: list[str] = field(default_factory=list)
    # macOS menu-bar companion: a background `serve` auto-starts it (GUI session
    # only) unless this is set false. No effect off macOS / headless.
    menubar: bool = True
    # Seconds the request loop waits for the *next* generated token before it gives
    # up, cancels the generation, and returns an error (mlx-vlm's token-queue
    # timeout; default 600). Raise it for very long prefills on big/over-RAM models;
    # 0 (or negative) disables the timeout (wait forever). None => leave the env /
    # mlx-vlm default in place.
    token_queue_timeout_s: float | None = None
    # Prefill chunk size in tokens for every model this server runs (upstream
    # default 2048). Lower it to cap the per-request prefill transient on long
    # prompts. Server-wide by design: the engine reads it per request from the
    # env, after the per-model load window has closed. None => leave the env /
    # upstream default in place.
    prefill_step_size: int | None = None
    # Activation dtype for every model this server loads: "auto" (the runtime
    # default), "bfloat16" or "float16", plus the "bf16"/"fp16" spellings.
    # This key is server-wide by design. The reason to leave bfloat16 is that
    # the GPU has no native bfloat16 arithmetic. That is a property of the
    # machine, not of one model. None keeps the env or the runtime default.
    dtype: str | None = None
    # Decode-priority prefill pacing ratio: a live decode batch gets this
    # multiple of each prefill chunk's GPU time before the next chunk is
    # admitted (1.0 ~= 50/50 split; 0 = stock 1 decode step : 1 chunk).
    # None => leave the env / branch default in place, which is "auto"
    # (dynamic pacing via auto_ratio); a number pins a static split.
    decode_prefill_ratio: float | str | None = None
    # Prefill tick budget in wall-clock ms: while decode rows are live, each
    # prefill chunk is halved until its predicted wall time (from the last
    # observed chunk cost) fits this budget, bounding the per-chunk decode
    # stall. 0 disables the term. None => leave the env / default (500) in
    # place. Inert with no live decode, so single-stream TTFT is untouched.
    prefill_tick_ms: float | None = None
    # MLX buffer-cache cap in GiB (mx.set_cache_limit). None => auto policy:
    # bounded automatically when the biggest configured model leaves little
    # working-set slack (deep-context safety), unlimited otherwise. Negative
    # => force unlimited (suppress auto). 0 => disable the cache entirely.
    # The GMLX_CACHE_LIMIT_GB env overrides this key (see server_memory).
    cache_limit_gb: float | None = None
    # Built-in per-family sampling defaults + intents (profiles.py). False removes
    # the family base layer and the built-in profile names (@coding etc.).
    family_defaults: bool = True
    # Stochastic MTP acceptance for sampled requests, process-wide: drafts are
    # accepted by p/q rejection sampling, so output follows the same
    # distribution as non-speculative sampling but is not token-identical to
    # it; acceptance (and decode speed) rises at temp > 0. Off = default MTP,
    # token-identical. Greedy requests are unaffected either way.
    stochastic_mtp: bool = False
    # Hold GPU clocks up while a streamed model is decoding (loader gate;
    # only acts on models with a decode feeder). The heartbeat parks when
    # no request is decoding, so an idle server pays nothing.
    gpu_keepwarm: bool | None = None   # None => on with the decode feeder
    defaults: ServerDefaults = field(default_factory=ServerDefaults)
    profiles: dict[str, Profile] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    models: dict[str, ModelCfg] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)   # name -> id | id@profile
    discover: list[DiscoverSpec] = field(default_factory=list)
    talk: TalkCfg = field(default_factory=TalkCfg)
    launch: LaunchCfg = field(default_factory=LaunchCfg)
    assistant: AssistantCfg = field(default_factory=AssistantCfg)
    # Served assistant aliases: pseudo-model id -> AssistantAlias. Requests to
    # an alias on /v1/chat/completions run the tool loop server-side.
    assistants: dict[str, AssistantAlias] = field(default_factory=dict)
    # Assistants on a non-loopback bind refuse to start unless this is true:
    # anyone holding the API key can drive tool execution on this host.
    assistant_allow_remote: bool = False
    # Chat UI: the default color theme (``--theme`` overrides) and user-defined
    # theme specs (name -> slot/style mapping). Carried raw here; the chat
    # startup validates and registers them via theme.register_user_themes so a
    # malformed theme warns in chat instead of failing serve.
    theme: str | None = None
    themes: dict = field(default_factory=dict)


@dataclass
class ResolvedModel:
    """A model with every layer merged - what the server actually serves. ``sampling``
    is still applied per-key against the request at the gen-args seam; ``load``/``cache``
    feed the residency env window; the scalars drive load dispatch + residency."""
    id: str
    path: str                       # resolved absolute path
    sampling: dict
    load: dict
    cache: dict                     # merged (server.cache base; may carry nested "disk")
    system: str | None
    speculative: bool
    mmproj: str | None           # resolved abspath or None
    draft_gguf: str | None       # resolved abspath or None
    pin: bool
    ttl_s: float | None
    profile_name: str | None = None
    # inline Jinja or path to a .jinja/.txt; baked into the tokenizer at load, so it
    # is load-affecting (folded into load_signature) - unlike system/sampling.
    chat_template: str | None = None
    # Extra apply_chat_template variables merged into the per-request template kwargs
    # at the gen-args seam (request fields win). Per-request - not load-affecting.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Dedicated thinking controls (see Profile); mapped onto this model's
    # template spelling at the gen-args seam. Per-request - not load-affecting.
    thinking: Any = None
    reasoning_effort: str | None = None
    # resolved GGUF LoRA adapter abspath, applied live over the base at load. Ids on
    # one GGUF that differ only in adapter share one resident entry: `adapters` is
    # the sorted union of their adapters (filled by the serving registry; slot i of
    # the row channel holds adapters[i]) and is what enters load_signature.
    adapter: str | None = None
    adapters: tuple = ()
    # Execution placement ("experts" = routed experts stream, rest of the model
    # + KV on GPU; "cpu" = whole model on CPU); load-affecting - it restructures
    # which device/stream the model runs on (and what gets wired).
    stream: Any = None
    # Lossy MoE fan-out levers for the streamed expert stacks; load-affecting -
    # the filters/hooks are installed over the routers and wrappers at load.
    moe_experts: int | None = None
    moe_expert_mass: float | None = None
    moe_miss_shed: float | None = None
    moe_layer_shed: float | None = None
    moe_prestage: str | None = None
    # Feeder overrides for streaming models (None = loader default); load-
    # affecting - they decide the ring slots / wired arena built at load.
    prefill_feeder: bool | None = None
    decode_feeder: bool | None = None
    stream_fast_disk: str | None = None
    # Sampling family (profiles.py key) the spec resolved under; informational
    # (surfaced by /v1/models and `gmlx profiles`), not load-affecting.
    family: str | None = None
    # Speculative batch-width cap: None = drafter-family default, 0 = uncapped,
    # N = speculate only while the live decode batch is <= N wide.
    # Load-affecting - it is stamped onto the drafter at load.
    speculative_width_cap: int | None = None

    def effective_adapters(self) -> tuple:
        """The adapters the resident entry serving this id carries: the
        registry-filled union, else this id's own adapter alone."""
        return tuple(self.adapters) or ((self.adapter,) if self.adapter else ())

    def base_signature(self) -> tuple:
        """:meth:`load_signature` with the adapter component blanked: ids
        that agree on it can share one resident entry (their adapters
        become that entry's slots)."""
        return self._signature(None)

    def load_signature(self) -> tuple:
        """Identity for the residency cache_key: two ids backed by the same GGUF but
        loaded differently (kv bits, mmproj, drafter, speculative, chat template,
        adapters) are distinct resident entries. ``chat_template`` is load-affecting
        (baked into the tokenizer). The adapter component is the entry's adapter
        union (:meth:`effective_adapters`): ids differing only in adapter share
        the entry and select their slot per request, so a union that changes
        (a reload adding an id with a new adapter) forks a new entry.
        Sampling/system/ttl do not change the loaded model and are excluded.

        The stream-riding keys enter as their effective values, not their
        config spellings, so an explicitly written default never forks a
        resident entry from an unset key: without a ``stream`` placement the
        MoE levers and feeder overrides are inert (announced as ignored at
        load) and collapse to None; ``moe_prestage`` collapses to None
        whenever it resolves to ranked behavior (``ranked``, or ``keepers``
        without ``moe_miss_shed``); the feeder tri-states resolve through the
        same explicit-then-env-then-default policy as
        loader._resolve_feeder_defaults. The env reads happen in the serving
        process, which is also where the load happens, so signature and load
        always see the same values."""
        return self._signature(self.effective_adapters())

    def _signature(self, adapters) -> tuple:
        stream = self.stream or None
        prestage = self.moe_prestage if self.moe_prestage == "keepers" else None
        if self.moe_miss_shed is None:
            prestage = None
        if stream:
            pf = (self.prefill_feeder if self.prefill_feeder is not None
                  else env_bool("GMLX_FEEDER_PREFILL", True))
            df = (self.decode_feeder if self.decode_feeder is not None
                  else env_bool("GMLX_FEEDER_DECODE", stream == "experts"))
            levers = (str(self.moe_experts), str(self.moe_expert_mass),
                      str(self.moe_miss_shed), str(self.moe_layer_shed),
                      str(prestage), str(pf), str(df),
                      str(self.stream_fast_disk))
        else:
            levers = (str(None),) * 8
        return (
            self.path,
            self.mmproj,
            self.draft_gguf,
            bool(self.speculative),
            str(self.speculative_width_cap),
            self.chat_template,
            adapters,
            str(stream),
            *levers,
            tuple(sorted((k, str(v)) for k, v in self.load.items())),
            tuple(sorted((k, str(v)) for k, v in _flatten_cache(self.cache).items())),
        )


# Path resolution
def resolve_path(p: str | None, model_dirs: list[str]) -> str | None:
    """Resolve a model/mmproj/draft path. ``None`` passes through. An
    ``hf:<org>/<repo>/<file.gguf>[@rev]`` ref resolves to its file in the **local**
    Hugging Face cache (never the network). An absolute or ``~``/``$VAR`` path is
    expanded and returned as-is; a bare/relative path is searched against
    ``model_dirs`` in order (first existing match wins). A miss raises a
    :class:`ConfigError` listing the roots searched (or the `pull` fix for an hf ref)."""
    if p is None:
        return None
    if isinstance(p, str) and p.startswith("hf:"):
        return _resolve_hf_cache_path(p, model_dirs)
    expanded = os.path.expanduser(os.path.expandvars(p))
    if os.path.isabs(expanded):
        return expanded
    roots = [os.path.expanduser(os.path.expandvars(d)) for d in model_dirs]
    for root in roots:
        cand = os.path.join(root, expanded)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    # Last resort: relative to cwd (lets a bare run work without model_dirs); only if
    # it exists, else report the misses so the user knows where we looked.
    if os.path.exists(expanded):
        return os.path.abspath(expanded)
    searched = roots or ["<no model_dirs set>"]
    raise MissingModelFile(
        f"model path {p!r} not found under model_dirs: {searched}")


def _resolve_hf_cache_path(ref: str, model_dirs: list | None = None) -> str:
    """Resolve an ``hf:<org>/<repo>/<file.gguf>[@rev]`` model path to a concrete
    file - never the network. Looks in the **local** Hugging Face cache first,
    then under ``model_dirs`` in ``gmlx pull``'s layout
    (``<root>/<org>__<repo>/<file>``): the miss error tells people to run
    ``pull``, so the resolver must find what ``pull`` downloads. Backs configs
    that reference cache-resident GGUFs (see ``gmlx init --from-hf-cache``)."""
    body = ref[len("hf:"):]
    revision: str | None = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)
    parts = [s for s in body.split("/") if s]
    if len(parts) < 3:
        raise ConfigError(
            f"hf model path {ref!r} must be hf:<org>/<repo>/<file.gguf> "
            f"(optionally @<revision>)")
    repo = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        raise ConfigError(
            f"resolving {ref!r} needs huggingface_hub (pip install huggingface_hub)")
    cached = try_to_load_from_cache(repo, filename, revision=revision or "main")
    if isinstance(cached, str) and os.path.exists(cached):
        return cached
    flat = repo.replace("/", "__")
    for d in model_dirs or []:
        root = os.path.expanduser(os.path.expandvars(d))
        cand = os.path.join(root, flat, filename)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    raise MissingModelFile(
        f"hf model {ref!r} is not in the local Hugging Face cache or under "
        f"model_dirs; `gmlx pull hf:{repo}/{filename}` to fetch it, or "
        f"remove the entry.")


def default_config_paths(*, note_local: bool = True) -> list[Path]:
    """Bare-start config search order (first existing wins). With
    ``note_local``, a ``./gmlx.yaml`` that this search skips gets one line
    on stderr per process."""
    paths = [Path(os.path.expanduser(p)) for p in _DEFAULT_CONFIG_PATHS]
    if note_local:
        note_local_config(paths=paths)
    return paths


def note_local_config(argv: Sequence[str] | None = None, *,
                      paths: list[Path] | None = None) -> None:
    """Say once that ``./gmlx.yaml`` is not read, when it exists and no
    user-level config does, so the line stops once the file is moved. It is
    silent when ``argv``, the command line by default, asks for help or names
    a config with ``--config``, and when ``./gmlx.yaml`` is one of the search
    ``paths``, as when you work in ``~/.config/gmlx``."""
    global _local_config_noted
    if _local_config_noted:
        return
    words = list(sys.argv[1:] if argv is None else argv)
    words = words[:words.index("--")] if "--" in words else words
    if any(w in ("-h", "--help", "--config") or w.startswith("--config=") for w in words):
        return
    if paths is None:
        paths = [Path(os.path.expanduser(p)) for p in _DEFAULT_CONFIG_PATHS]
    try:
        local = Path(_LOCAL_CONFIG)
        # is_file never opens the file, so a named pipe cannot block here.
        if not local.is_file() or any(p.is_file() for p in paths):
            return
    except (OSError, RuntimeError):
        return
    _local_config_noted = True
    print(_LOCAL_CONFIG_NOTE, file=sys.stderr)


def _launch_block(path: Path):
    """The raw ``launch`` block of a config file, or None when it has none.
    An unknown top-level key that holds a ``container`` block, such as a
    misspelled ``launch``, is a ConfigError, because the block may have
    meant to turn container mode on."""
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read {path}: {e}")
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {path}: {e}")
    if doc is None:
        return None
    if not isinstance(doc, dict):
        raise ConfigError(f"{path} holds a {type(doc).__name__}, not a mapping of "
                          "settings.")
    if "container" in doc:
        raise ConfigError(f"{path} has a container block at the top level. Did you "
                          "mean launch: container:?")
    for key, value in doc.items():
        if key not in _TOP_KEYS and isinstance(value, dict) and "container" in value:
            raise ConfigError(f"{path} has the unknown top-level key {key!r} with a "
                              "container block under it. Did you mean launch?")
    return doc.get("launch")


def launch_block_enables(client: str) -> tuple[bool | None, Path | None]:
    """Whether the raw ``launch`` block of the user-level config turns
    container mode on for ``client``, read without the full checks, and the
    file. Three levels are checked: ``launch``, ``launch.container`` and
    ``launch.container.clients.<client>``, each against its own key set.
    Every key under ``clients`` must name a known client. True only when the
    client's own ``enabled`` is true, or when it is absent and the global
    one is true, as in ``for_client``. False only when each of the three
    levels is absent or a mapping of known keys, and every ``enabled``
    there is exactly ``false``. None, which stops the launch, for any other
    value or shape, and for an unknown top-level key that holds a
    ``container`` block, since the block may have meant to turn container
    mode on. The keys of other clients' blocks are not checked."""
    found = next((q for q in default_config_paths() if q.is_file()), None)
    if found is None:
        return False, None
    try:
        block = _launch_block(found)
    except ConfigError:
        return None, found
    if block is None:
        return False, found
    box = block.get("container") if isinstance(block, dict) else None
    clients = box.get("clients") if isinstance(box, dict) else None
    own = clients.get(client) if isinstance(clients, dict) else None
    missing = object()
    values = [box.get("enabled", missing) if isinstance(box, dict) else missing,
              own.get("enabled", missing) if isinstance(own, dict) else missing]
    shared, own_value = values
    if own_value is True or (own_value is missing and shared is True):
        return True, found
    if not isinstance(block, dict) or set(block) - _LAUNCH_KEYS:
        return None, found
    if box is None:
        return False, found
    if not isinstance(box, dict) or set(box) - _LAUNCH_CONTAINER_KEYS:
        return None, found
    if clients is not None and (not isinstance(clients, dict)
                                or set(clients) - set(LAUNCH_CLIENTS)):
        return None, found
    if own is not None and (not isinstance(own, dict) or set(own) - _LAUNCH_CLIENT_KEYS):
        return None, found
    return (False if all(v is missing or v is False for v in values) else None), found


def load_launch_settings(*, note_local: bool = True) -> LaunchCfg:
    """The ``launch`` block of the user-level config: the first of
    ``~/.config/gmlx/gmlx.yaml`` and ``~/.gmlx.yaml`` that exists. Only the
    ``launch`` block is parsed. ``note_local`` as in
    :func:`default_config_paths`. Every error names the file."""
    found = next((q for q in default_config_paths(note_local=note_local)
                  if q.is_file()), None)
    if found is None:
        return LaunchCfg()
    block = _launch_block(found)
    try:
        return _parse_launch(block)
    except ConfigError as e:
        raise ConfigError(f"{found}: {e}") from None


def default_config_write_path() -> Path:
    """Where ``gmlx init`` writes when ``--out`` is not given."""
    return Path(os.path.expanduser(DEFAULT_CONFIG_WRITE))


def edit_config_yaml(path, mutate, flag: str = "--config") -> None:
    """Round-trip edit of a config file: load with ruamel, call ``mutate(doc)``
    (a CommentedMap), write back. Comments, quoting, and untouched entries keep
    their exact formatting. ruamel is imported lazily.

    The edit reads and writes the file that a config link leads to, so the
    link stays a link. :func:`config_write_target` refuses a link that a
    container client can change, and ``flag`` names the option that gives
    ``path`` in that message. The read and the write use one open folder,
    so they change one file."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    yaml = YAML()
    yaml.preserve_quotes = True
    # Match the scaffold's 2-space mapping / 4-space block-sequence style so an
    # untouched list (e.g. model_dirs) isn't reflowed into the diff.
    yaml.indent(mapping=2, sequence=4, offset=2)
    # ruamel's default 80-col wrap folds long scalars (hf: cache paths) onto a
    # continuation line - one value per line, never wrapped.
    yaml.width = 2 ** 16
    real = config_write_target(path, flag)
    name = os.path.basename(real)
    with _config_folder(real) as folder:
        try:
            text, _st = _read_in(folder, name, real, flag)
        except FileNotFoundError as e:
            raise _read_error(real, e) from e
        doc = yaml.load(text)
        if doc is None:
            doc = CommentedMap()
        mutate(doc)
        out = io.StringIO()
        yaml.dump(doc, out)
        _write_in(folder, name, out.getvalue(), real, flag)


class ConfigWriteError(OSError):
    """gmlx will not write a config file, or could not read or write it.
    The message names the file and the next step. ``reason`` is the
    message without a ``step`` that was given apart, for a caller that
    gives its own step."""

    def __init__(self, reason: str, step: str | None = None):
        super().__init__(f"{reason} {step}" if step else reason)
        self.reason = reason


def _shown(path: str) -> str:
    """``path`` as a message shows it, with ~ for the home folder."""
    from gmlx.container.settings import _tilde

    return _tilde(path)


_HOMES = "where launch keeps the private homes of the clients"
_SHARED = "which a container session shares or once shared read-write"


def _client_folders() -> list[tuple[str, str]]:
    """The folders that a container client can write, each with what it is:
    the private homes, and each folder that a container session shares or
    once shared read-write. Launch records a share before its session
    starts."""
    from gmlx.container.settings import shared_history
    from gmlx.container.state import data_path

    return [(canonical(data_path()), _HOMES), *((f, _SHARED) for f in shared_history())]


def _name_with(flag: str | None) -> str:
    """How the user gives gmlx another config path: with ``flag``, or with
    the server's --config when the path comes from the server."""
    return f"pass {flag} with" if flag else "start the server with --config and"


def config_target(path, flag: str | None = "--config") -> tuple[str, str | None]:
    """The real path of the config ``path``, and why gmlx does not write
    it, or None. A writer changes the real path, so a config link stays a
    link.

    A container client can replace a file or a folder in a folder that it
    writes with a link to any file of yours. So gmlx does not write through
    ``path`` when it, or a link on the way to it, lies in such a folder and
    the real path leads out of that folder. ``flag`` names the option that
    gives the path, for the next step in the message."""
    from gmlx.container.settings import _resolution_paths, _tilde

    written = os.path.abspath(os.path.expanduser(str(path)))
    real = canonical(written)
    # The paths that the resolution visits, found once for every folder,
    # since the share history can hold hundreds of folders.
    trail = list(dict.fromkeys([written, *_resolution_paths(written)]))
    for folder, what in _client_folders():
        if path_inside(real, folder):
            continue
        # As settings._link_in gives it: a link first.
        hits = [p for p in trail if path_inside(p, folder)]
        if not hits:
            continue
        link = next((p for p in hits if os.path.islink(p)), hits[0])
        given = os.path.join(canonical(os.path.dirname(written)), os.path.basename(written))
        if link in (written, given):
            head = (f"the config {_tilde(written)} lies in {_tilde(folder)}, {what}, "
                    f"and it leads to {_tilde(real)}.")
        else:
            head = (f"the config {_tilde(written)} leads to {_tilde(real)} through "
                    f"{_tilde(link)}, in {_tilde(folder)}, {what}.")
        return real, (f"{head} A container client can change where it leads, so gmlx does "
                      "not write through it. Remove the link if you did not make it, or "
                      f"{_name_with(flag)} a path that does not go through the link.")
    return real, None


def config_folder_refusal(path, real: str, flag: str | None = "--config") -> str | None:
    """Why gmlx cannot write the config at the real path ``real``, which
    the config ``path`` leads to, or None: its folder exists and gmlx
    cannot write it, as when a link leads into a read-only folder that a
    tool such as home-manager manages. The step to change the config where
    it is managed comes only when an entry is at ``real``: for a new config
    there is nothing to change."""
    from gmlx.container.settings import _tilde

    folder = os.path.dirname(real)
    if not os.path.isdir(folder) or os.access(folder, os.W_OK | os.X_OK):
        return None
    written = os.path.abspath(os.path.expanduser(str(path)))
    via = "" if _tilde(written) == _tilde(real) else f", which {_tilde(written)} leads to"
    other = "a file in a folder that you can write."
    if os.path.lexists(real):
        step = f"Change the config where it is managed, or {_name_with(flag)} {other}"
    else:
        step = f"{_name_with(flag).capitalize()} {other}"
    return (f"gmlx cannot write {_tilde(folder)}, the folder of the config {_tilde(real)}"
            f"{via}. {step}")


def config_write_target(path, flag: str | None = "--config") -> str:
    """The real path of the config ``path``, where a writer puts the new
    text. Raises :class:`ConfigWriteError` when gmlx does not write through
    ``path`` (see :func:`config_target`) or cannot write the folder. A
    command calls it before a step that it cannot undo, such as ``gmlx rm``
    before it deletes a model file."""
    real, why = config_target(path, flag)
    why = why or config_folder_refusal(path, real, flag)
    if why is not None:
        raise ConfigWriteError(why)
    return real


@contextlib.contextmanager
def _config_folder(real: str, use: str = "write"):
    """An open descriptor of the folder of the real path ``real``. Raises
    :class:`ConfigWriteError` when the folder is no longer at that path,
    as when a client makes a folder on the way a link after the check. A
    read and a write through the descriptor stay in the folder that the
    check saw. ``use`` is what the caller does with the config, for the
    message."""
    folder = os.path.dirname(real)
    try:
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as e:
        raise ConfigWriteError(f"could not open {_shown(folder)}, the folder of the config "
                               f"{_shown(real)} ({e.strerror or e}).",
                               "Check the folder, then try again.") from e
    try:
        try:
            now = fd_path(fd) or folder
        except OSError:
            now = folder
        if not (path_inside(now, folder) and path_inside(folder, now)):
            raise ConfigWriteError(f"the folder of the config {_shown(real)} changed after "
                                   f"gmlx checked it, so gmlx did not {use} the config.",
                                   "Check the folder, then try again.")
        yield fd
    finally:
        os.close(fd)


def _read_error(real: str, e: OSError) -> ConfigWriteError:
    """The error for a read of the config ``real`` that failed with ``e``."""
    return ConfigWriteError(f"could not read the config {_shown(real)} ({e.strerror or e}).",
                            "Check the file, then try again.")


def _not_a_file(real: str, flag: str | None) -> ConfigWriteError:
    """The error for a config ``real`` that is not a file, such as a
    folder. ``flag`` names the option that gives the config."""
    return ConfigWriteError(f"the config {_shown(real)} is not a file.",
                            f"{_name_with(flag).capitalize()} the path of a config file.")


def _read_in(folder: int, name: str, real: str,
             flag: str | None) -> tuple[str, os.stat_result]:
    """The text of the config file ``name`` in the open ``folder``, and its
    status. The read does not follow a link, so it reads the file that the
    check saw, never a link that a client puts in its place after the
    check. Raises FileNotFoundError when there is no file, and
    :class:`ConfigWriteError` when gmlx cannot read it to the end, it is
    not text, or it is not a file. ``real`` is the path of the config, for
    the message, and ``flag`` names the option that gives it."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=folder)
    except FileNotFoundError:
        raise
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ConfigWriteError(f"the config {_shown(real)} became a link after gmlx "
                                   "checked it, so gmlx did not read it.",
                                   "Remove the link if you did not make it, then try "
                                   "again.") from e
        raise _read_error(real, e) from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _not_a_file(real, flag)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd) as f:
        try:
            return f.read(), st
        except UnicodeDecodeError as e:
            code = e.encoding.upper()
            raise ConfigWriteError(f"the config {_shown(real)} is not {code} text, so gmlx "
                                   "did not read it.",
                                   f"Convert it to {code} text, then try again.") from e
        except OSError as e:
            raise _read_error(real, e) from e


def read_config_text(real: str) -> tuple[str, int]:
    """The text of the config at the real path ``real``, which
    :func:`config_target` gives, and its time of change in nanoseconds.
    A link that a client puts at ``real`` after the check is refused, not
    followed. Raises FileNotFoundError when there is no file, and
    :class:`ConfigWriteError` when gmlx cannot read it."""
    with _config_folder(real, "read") as folder:
        text, st = _read_in(folder, os.path.basename(real), real, None)
    return text, st.st_mtime_ns


def _write_in(folder: int, name: str, text: str, real: str, flag: str | None) -> None:
    """Replace the file ``name`` in the open ``folder`` with ``text``
    through a new file in that folder, so a crash or a full disk never
    leaves the config half written. The file keeps its mode. A new file
    gets mode 0600, because a config can hold the server's key. An error
    names the config ``real``, not the new file, and ``flag`` names the
    option that gives the config."""
    try:
        _replace_in(folder, name, text, real, flag)
    except ConfigWriteError:
        raise
    except OSError as e:
        raise ConfigWriteError(f"could not write the config {_shown(real)} "
                               f"({e.strerror or e}).",
                               "Check that you can write its folder, then try again.") from e


def _replace_in(folder: int, name: str, text: str, real: str, flag: str | None) -> None:
    """The write of :func:`_write_in`, with the error of the system call.
    Raises :class:`ConfigWriteError` when ``name`` is not a file or a link,
    such as a folder."""
    try:
        st = os.stat(name, dir_fd=folder, follow_symlinks=False)
    except FileNotFoundError:
        mode = 0o600
    else:
        if stat.S_ISREG(st.st_mode):
            mode = stat.S_IMODE(st.st_mode)
        elif stat.S_ISLNK(st.st_mode):
            mode = 0o600
        else:
            raise _not_a_file(real, flag)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    for _ in range(100):
        tmp = f".gmlx-config-{secrets.token_hex(4)}"
        try:
            fd = os.open(tmp, flags, 0o600, dir_fd=folder)
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(errno.EEXIST, "no free name for a new file in the folder")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            os.fchmod(f.fileno(), mode)
        os.replace(tmp, name, src_dir_fd=folder, dst_dir_fd=folder)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=folder)
        raise


def replace_config_text(real: str, text: str, flag: str | None = "--config") -> None:
    """Replace the config at the real path ``real``, which
    :func:`config_write_target` gives, with ``text``. A link at ``real``
    is replaced, never followed. ``flag`` names the option that gives the
    config, for the message."""
    with _config_folder(real) as folder:
        _write_in(folder, os.path.basename(real), text, real, flag)


# Merge helpers
def _merge_dict(base: dict, over: dict) -> dict:
    """Shallow merge with one level of nesting for the cache ``disk`` block, so a
    profile that sets only ``cache.disk.max_gb`` doesn't wipe a server ``disk.path``."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _flatten_cache(cache: dict) -> dict:
    """Flatten ``{..., disk: {...}}`` to ``{..., disk.path, disk.max_gb, ...}`` for a
    stable signature/diff."""
    flat = {k: v for k, v in cache.items() if k != "disk"}
    for k, v in (cache.get("disk") or {}).items():
        flat[f"disk.{k}"] = v
    return flat


@functools.lru_cache(maxsize=None)
def _builtin_profiles(family: str | None) -> dict[str, Profile]:
    """The built-in intents materialized as :class:`Profile` objects for one
    family. Each carries only the intent's *delta* - the family base is merged
    as its own (lowest) layer in :func:`resolve_model`, so delta-over-base
    reproduces the full card value. Cached per family (the table is static)."""
    out: dict[str, Profile] = {}
    for name, delta in _family_profiles.family_intents(family).items():
        out[name] = Profile(
            name=name,
            sampling=dict(delta.get("sampling") or {}),
            chat_template_kwargs=dict(delta.get("chat_template_kwargs") or {}),
        )
    return out


def profile_names(cfg: "ServerCfg") -> set[str]:
    """Every name a ``@profile`` suffix / ``profile`` field / ``extends`` target
    may legally use: the user-defined profiles plus (unless
    ``server.family_defaults: false``) the built-in intents."""
    names = set(cfg.profiles)
    if cfg.family_defaults:
        names |= _family_profiles.BUILTIN_INTENTS
    return names


def _resolve_profile_chain(name: str, profiles: dict[str, Profile]) -> dict:
    """Merge a profile and its ``extends`` ancestors (root applied first, leaf last)
    into ``{sampling, load, cache, system, chat_template}``. Assumes the chain is
    acyclic + known (validated at load); a link that isn't in ``profiles`` (a
    kill-switched built-in) simply ends the chain."""
    chain: list[Profile] = []
    seen: set[str] = set()
    cur: str | None = name
    while cur is not None:
        prof = profiles.get(cur)
        if prof is None:
            break
        chain.append(prof)
        seen.add(cur)
        cur = prof.extends if prof.extends not in seen else None
    groups = {"sampling": {}, "load": {}, "cache": {}, "system": None,
              "chat_template": None, "chat_template_kwargs": {},
              "thinking": None, "reasoning_effort": None}
    for prof in reversed(chain):          # root -> leaf
        groups["sampling"] = _merge_dict(groups["sampling"], prof.sampling)
        groups["load"] = _merge_dict(groups["load"], prof.load)
        groups["cache"] = _merge_dict(groups["cache"], prof.cache)
        groups["chat_template_kwargs"] = _merge_dict(
            groups["chat_template_kwargs"], prof.chat_template_kwargs)
        if prof.system is not None:
            groups["system"] = prof.system
        if prof.chat_template is not None:
            groups["chat_template"] = prof.chat_template
        if prof.thinking is not None:
            groups["thinking"] = prof.thinking
        if prof.reasoning_effort is not None:
            groups["reasoning_effort"] = prof.reasoning_effort
    return groups


def _matched_rule_profile(model_id: str, rules: list[Rule]) -> str | None:
    """First rule (by order) whose glob matches the id; ``None`` if none match."""
    for rule in rules:
        if fnmatch.fnmatch(model_id, rule.match):
            return rule.profile
    return None


def split_address(field: str, profiles) -> tuple[str, str | None]:
    """Split an ``id@profile`` (or ``alias@profile``) address on the **last** ``@``,
    treating the tail as a profile only if it names a known one - so hf ``org/model@rev``
    and ids/aliases containing ``@``-like text stay intact. Pure (takes the profile
    name set), so config validation and the request resolver share one rule."""
    if "@" in field:
        head, tail = field.rsplit("@", 1)
        if head and tail in profiles:
            return head, tail
    return field, None


# Resolution
def resolve_model(
    model_id: str, cfg: ServerCfg, request_profile: str | None = None
) -> ResolvedModel:
    """Merge every layer for ``model_id`` into a :class:`ResolvedModel`.

    Precedence low -> high: ``family base (built-in) -> server.defaults.profile ->
    matched rule.profile -> (request_profile or model.profile, +extends) ->
    model.profiles[<selected>] tweak -> model.overrides``. ``server.cache`` is the
    cache base (below all profiles). ``request_profile`` (an ``@profile`` or the
    request ``profile`` field) replaces the model's configured profile and may
    name a built-in intent; a user profile shadows a built-in of the same name.
    Raises :class:`KeyError` for an unknown ``model_id`` and :class:`ConfigError`
    for an unknown ``request_profile``."""
    model = cfg.models[model_id]
    # User profiles shadow built-in intents by name; built-ins resolve per this
    # model's family, so `coding` means the right thing for each model.
    profs = dict(cfg.profiles)
    if cfg.family_defaults:
        profs = {**_builtin_profiles(model.family), **cfg.profiles}
    if request_profile is not None and request_profile not in profs:
        raise ConfigError(
            f"unknown profile {request_profile!r}; "
            f"available: {sorted(profs)}"
        )

    sampling: dict = {}
    load: dict = {}
    cache: dict = dict(cfg.cache)       # server.cache is the base
    system: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict = {}
    thinking: Any = None
    reasoning_effort: str | None = None

    # Lowest layer: the family's model-card base. Merged directly (not a named
    # pseudo-profile), so it can never be addressed or shadowed - anything a
    # user profile sets wins over it.
    if cfg.family_defaults:
        base = _family_profiles.family_base(model.family)
        sampling = _merge_dict(sampling, base.get("sampling", {}))
        load = _merge_dict(load, base.get("load", {}))
        chat_template_kwargs = _merge_dict(
            chat_template_kwargs, base.get("chat_template_kwargs", {}))
        # The GGUF's own embedded model-card sampling (general.sampling.*)
        # refines the arch-family guess; profiles/overrides still win.
        from gmlx.load.discovery import header_sampling
        try:
            hs = header_sampling(resolve_path(model.path, cfg.model_dirs))
        except ConfigError:
            hs = {}
        if hs:
            sampling = _merge_dict(sampling, hs)

    layers: list[str] = []
    if cfg.defaults.profile:
        layers.append(cfg.defaults.profile)
    ruled = _matched_rule_profile(model_id, cfg.rules)
    if ruled:
        layers.append(ruled)
    effective = request_profile or model.profile
    if effective:
        layers.append(effective)

    for prof_name in layers:
        groups = _resolve_profile_chain(prof_name, profs)
        sampling = _merge_dict(sampling, groups["sampling"])
        load = _merge_dict(load, groups["load"])
        cache = _merge_dict(cache, groups["cache"])
        chat_template_kwargs = _merge_dict(
            chat_template_kwargs, groups["chat_template_kwargs"])
        if groups["system"] is not None:
            system = groups["system"]
        if groups["chat_template"] is not None:
            chat_template = groups["chat_template"]
        if groups["thinking"] is not None:
            thinking = groups["thinking"]
        if groups["reasoning_effort"] is not None:
            reasoning_effort = groups["reasoning_effort"]

    # Per-model tweak of the *selected* named profile (the highest-precedence
    # name that applied): reshape what e.g. `@coding` means for this one model.
    selected = effective or ruled or cfg.defaults.profile
    tweak = (model.profiles or {}).get(selected) if selected else None
    if tweak:
        sampling = _merge_dict(sampling, tweak.get("sampling", {}))
        load = _merge_dict(load, tweak.get("load", {}))
        cache = _merge_dict(cache, tweak.get("cache", {}))
        chat_template_kwargs = _merge_dict(
            chat_template_kwargs, tweak.get("chat_template_kwargs", {}))
        if tweak.get("system") is not None:
            system = tweak["system"]
        if tweak.get("chat_template") is not None:
            chat_template = tweak["chat_template"]
        if tweak.get("thinking") is not None:
            thinking = tweak["thinking"]
        if tweak.get("reasoning_effort") is not None:
            reasoning_effort = tweak["reasoning_effort"]

    ov = model.overrides or {}
    sampling = _merge_dict(sampling, ov.get("sampling", {}))
    load = _merge_dict(load, ov.get("load", {}))
    cache = _merge_dict(cache, ov.get("cache", {}))
    chat_template_kwargs = _merge_dict(
        chat_template_kwargs, ov.get("chat_template_kwargs", {}))
    if ov.get("system") is not None:
        system = ov["system"]
    if ov.get("chat_template") is not None:
        chat_template = ov["chat_template"]
    if ov.get("thinking") is not None:
        thinking = ov["thinking"]
    if ov.get("reasoning_effort") is not None:
        reasoning_effort = ov["reasoning_effort"]

    scheme = load.get("kv_quant_scheme")
    if scheme is not None:
        norm = str(scheme).strip().lower()
        if norm not in KV_QUANT_SCHEMES:
            # An unchecked value reaches mlx-vlm and builds caches no
            # gmlx path can read.
            raise ConfigError(
                f"load.kv_quant_scheme must be one of "
                f"{', '.join(KV_QUANT_SCHEMES)} (got {scheme!r})")
        load["kv_quant_scheme"] = norm

    ttl_s = model.ttl_s if model.ttl_s is not None else cfg.defaults.ttl_s
    return ResolvedModel(
        id=model_id,
        path=resolve_path(model.path, cfg.model_dirs),
        sampling=sampling,
        load=load,
        cache=cache,
        system=system,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        speculative=bool(model.speculative or model.draft_gguf or model.native_mtp),
        speculative_width_cap=model.speculative_width_cap,
        mmproj=resolve_path(model.mmproj, cfg.model_dirs),
        draft_gguf=(None if model.native_mtp
                    else resolve_path(model.draft_gguf, cfg.model_dirs)),
        adapter=resolve_path(model.adapter, cfg.model_dirs),
        stream=model.stream or None,
        moe_experts=model.moe_experts,
        moe_expert_mass=model.moe_expert_mass,
        moe_miss_shed=model.moe_miss_shed,
        moe_layer_shed=model.moe_layer_shed,
        moe_prestage=model.moe_prestage,
        prefill_feeder=model.prefill_feeder,
        decode_feeder=model.decode_feeder,
        stream_fast_disk=model.stream_fast_disk,
        pin=bool(model.pin),
        ttl_s=ttl_s,
        profile_name=effective,
        family=model.family,
    )


def load_cli_config(config_path: str | None = None) -> tuple:
    """Load a server config for the ``run``/``chat`` by-name lookup: an explicit
    ``config_path``, else the first existing default location. Returns
    ``(cfg, path)`` or ``(None, None)`` when no config is present. Raises
    :class:`ConfigError` for a bad explicit path or a malformed config."""
    if config_path:
        p = Path(os.path.expanduser(config_path))
        if not p.exists():
            raise ConfigError(f"--config not found: {p}")
        return load_config(p), str(p)
    for p in default_config_paths():
        if p.exists():
            return load_config(p), str(p)
    return None, None


def resolve_cli_model(name: str, cfg: ServerCfg,
                      request_profile: str | None = None
                      ) -> ResolvedModel | None:
    """Resolve a CLI model *name* against ``cfg`` - by id or alias, with an optional
    ``@profile`` - to a :class:`ResolvedModel` (path + merged sampling/load/template).
    ``None`` when the name matches no model/alias (the caller then reports the file
    miss). ``request_profile`` is the ``--profile`` flag; precedence mirrors the
    server: inline ``@profile`` > ``--profile`` > an alias's baked profile. Raises
    :class:`ConfigError` for an unknown profile on a known model."""
    raw = (name or "").strip()
    known = profile_names(cfg)
    head, inline_profile = split_address(raw, known)
    base_profile = None
    if head in cfg.aliases:                       # expand alias -> id (+ baked profile)
        head, base_profile = split_address(cfg.aliases[head], known)
    if head not in cfg.models:
        # `<id|alias>@suffix` with a real head but an unknown suffix isn't split off as
        # a profile (last-@ keeps hf org/model@rev intact); surface it as the unknown
        # profile it really is rather than a confusing file-miss.
        if "@" in raw:
            h, t = raw.rsplit("@", 1)
            if h in cfg.models or h in cfg.aliases:
                raise ConfigError(
                    f"unknown profile {t!r}; available: {sorted(known)}")
        return None
    effective = inline_profile or request_profile or base_profile
    if effective is not None and effective not in known:
        raise ConfigError(
            f"unknown profile {effective!r}; available: {sorted(known)}")
    return resolve_model(head, cfg, request_profile=effective)


def env_for(resolved: ResolvedModel) -> dict[str, str]:
    """The env vars to set in the residency window for this model's load params + APC
    prompt-cache config. Keys absent/``None`` are omitted; booleans render ``1``/``0``;
    a ``~`` disk path is expanded. The two speculative keys are the exceptions:
    both are always emitted (an empty/``0`` value is meaningful) so a sibling
    id's setting can never linger in the process env and be inherited."""
    env: dict[str, str] = {}
    for k, v in resolved.load.items():
        if v is not None and k in LOAD_ENV:
            env[LOAD_ENV[k]] = str(v)
    cache = resolved.cache or {}
    for k, v in cache.items():
        if k == "disk" or v is None:
            continue
        if k in CACHE_ENV:
            env[CACHE_ENV[k]] = "1" if v is True else "0" if v is False else str(v)
    # Raise the in-memory exact-cache pool above mlx-vlm's default of 2 when APC is on
    # and the user hasn't set it explicitly (see DEFAULT_EXACT_CACHE_ENTRIES).
    if cache.get("enabled") and "exact_entries" not in cache:
        env["APC_EXACT_CACHE_ENTRIES"] = str(DEFAULT_EXACT_CACHE_ENTRIES)
    disk = cache.get("disk") or {}
    # A present disk.path is what enables the SSD tier; only emit disk vars with a path.
    if disk.get("path"):
        for k, v in disk.items():
            if v is None or k not in CACHE_DISK_ENV:
                continue
            val = os.path.expanduser(str(v)) if k == "path" else str(v)
            env[CACHE_DISK_ENV[k]] = val
    # The per-id speculative state must reach the load bridge, which runs in the
    # engine's generation worker thread - a request-thread ContextVar is invisible
    # there, but this env window (held across the blocking stock load) is not. An
    # explicit "0" forces a non-speculative load even when a sibling id registered
    # the same GGUF for MTP (the lossless-oracle case: one GGUF, spec-on + spec-off).
    env["MLX_VLM_GGUF_SPECULATIVE"] = "1" if resolved.speculative else "0"
    # The drafter is per id too: two ids on one GGUF may differ (a companion
    # drafter vs native_mtp), and the path registry keeps only the last one.
    # Always emitted; "" means "this id drafts with the GGUF's own head".
    env["MLX_VLM_GGUF_DRAFT"] = resolved.draft_gguf or ""
    # Same reasoning, same always-emit rule: "" means "this id declares no cap,
    # use the drafter family default". Emitting only when set would let a
    # sibling id's cap linger in the process env and be inherited here.
    cap = resolved.speculative_width_cap
    env["MLX_VLM_GGUF_SPEC_WIDTH_CAP"] = "" if cap is None else str(cap)
    return env


# YAML loading + validation
def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _warn_unknown_keys(where: str, raw, known, *, strict: bool = False) -> None:
    """Reject (``strict``) or warn on keys outside a namespace's documented surface -
    a typo or unsupported knob that .get()-based parsing would otherwise drop silently.
    The message names the location, the offending keys, and the valid ones so the fix
    is obvious. Structural namespaces we fully own (server, models, profiles, rules,
    defaults, discover, overrides) pass ``strict=True`` so a typo like ``pinned:`` for
    ``pin:`` is a hard error, not a silently-unpinned model; the open-ended passthrough
    namespaces (sampling, load, cache) only warn, since mlx-lm / the APC may accept
    knobs we don't enumerate. No-op when ``raw`` isn't a mapping (shape errors surface
    elsewhere)."""
    if not isinstance(raw, dict):
        return
    bad = sorted(str(k) for k in set(raw) - set(known))
    if not bad:
        return
    msg = (f"{where}: unknown key{'s' if len(bad) > 1 else ''} {', '.join(bad)} "
           f"(known: {', '.join(sorted(known))})")
    if strict:
        raise ConfigError(msg)
    import warnings
    warnings.warn(msg, stacklevel=3)


def template_call_key_refusal(where: str, kwargs, keys=TEMPLATE_CALL_KEYS) -> str | None:
    """The message refusing the ``chat_template_kwargs`` keys in ``kwargs``
    that name a parameter of the template call, or None when there are none."""
    bad = sorted(k for k in kwargs if k in keys) if isinstance(kwargs, dict) else []
    if not bad:
        return None
    what = (("is a parameter", "a template variable") if len(bad) == 1
            else ("are parameters", "template variables"))
    msg = (f"{where} names {', '.join(map(repr, bad))}, which {what[0]} of "
           f"the chat template call, not {what[1]}")
    if "chat_template" in bad:
        msg += ". Set the model's template with the chat_template key instead"
    return msg


def drop_template_call_keys(where: str, kwargs: dict) -> dict:
    """``kwargs`` without the keys that name a parameter of the template
    call, with one warning that names them. A config that 0.4.19 loaded
    therefore still loads, and no request renders with those keys."""
    bad = [k for k in kwargs if k in TEMPLATE_CALL_KEYS]
    if not bad:
        return kwargs
    import warnings
    warnings.warn(f"{template_call_key_refusal(where, kwargs)}. The server ignores "
                  f"{'it' if len(bad) == 1 else 'them'}.", stacklevel=3)
    return {k: v for k, v in kwargs.items() if k not in TEMPLATE_CALL_KEYS}


def _section_mapping(where: str, raw) -> dict:
    """A config section that must be a mapping. ``None`` and the empty list
    (YAML's other rendering of an emptied-out section) mean "absent"; any other
    non-mapping is named here instead of surfacing as an ``AttributeError``
    deep inside a parser."""
    if raw is None or raw == [] or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{where} must be a mapping (got {type(raw).__name__}: {raw!r})")
    return raw


def _section_list(where: str, raw) -> list:
    """A config section that must be a list of entries (rules, discover)."""
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ConfigError(
            f"{where} must be a list of entries (got {type(raw).__name__}: "
            f"{raw!r})")
    return list(raw)


def _validate_stop(where: str, sampling: dict) -> None:
    """``sampling.stop`` must be a string or a list of strings at load time.
    Left unvalidated it only surfaces as an HTTP 400 on every request that
    resolves the profile, with no pointer back to the config line."""
    stop = sampling.get("stop")
    if stop is None or isinstance(stop, str):
        return
    if isinstance(stop, (list, tuple)) and all(isinstance(s, str) for s in stop):
        return
    raise ConfigError(
        f"{where} sampling.stop must be a string or a list of strings "
        f"(got {stop!r}); token *ids* are not accepted - use the token's text")


def _validate_server_dtype(dtype):
    """``server.dtype`` must name an activation dtype we actually offer.

    Unlike an unknown key, a bad value here is silent: the loader resolves the
    env var through env_choice, which falls back to bfloat16 without a word.
    Someone setting float16 to test a pre-Apple9 box would get the default back
    and never know, so a typo raises at parse time instead.
    """
    if dtype is None:
        return None
    value = str(dtype).strip().lower()
    if value not in SERVER_DTYPES:
        raise ConfigError(
            f"server.dtype must be one of {', '.join(SERVER_DTYPES)} "
            f"(got {dtype!r})")
    return value


def _normalize_optional_bool(value, key: str, where: str = "model"):
    """Normalize a tri-state boolean config value to None / True / False.
    None (absent) means "use the built-in default"; unrecognized values warn
    and fall back to None rather than silently flipping a feature."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "yes", "on", "1"):
        return True
    if s in ("false", "no", "off", "0"):
        return False
    import warnings
    warnings.warn(
        f"{where}: unrecognized {key} value {value!r} "
        "(expected true or false); using the default",
        stacklevel=3,
    )
    return None


def _normalize_speculative_width_cap(value, where: str = "model"):
    """Validate a ``speculative_width_cap``: None (family default), 0
    (uncapped), or a positive batch width. Bad values raise rather than
    default - a silently-ignored cap would serve the losing regime."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(
            f"{where}.speculative_width_cap: expected an int batch width, "
            f"got {value!r}")
    n = _coerce_num("speculative_width_cap", value, int, where=where)
    if n is not None and n < 0:
        raise ConfigError(
            f"{where}.speculative_width_cap: expected null, 0 (uncapped), or "
            f"a positive batch width, got {value!r}")
    return n


def _normalize_moe_expert_mass(value, where: str = "model",
                               key: str = "moe_expert_mass"):
    """Validate a gate-mass share in (0, 1] (``moe_expert_mass`` /
    ``moe_miss_shed``). ``None`` passes; a non-numeric or out-of-range value
    raises - a lossy knob must fail fast, not be silently reinterpreted."""
    p = _coerce_num(key, value, float, where=where)
    if p is not None and not 0.0 < p <= 1.0:
        raise ConfigError(
            f"{where}.{key}: expected a mass share in (0, 1], "
            f"got {value!r}")
    return p


def _normalize_moe_experts(value, where: str = "model"):
    """Validate a ``moe_experts`` per-token expert cap: a positive int."""
    k = _coerce_num("moe_experts", value, int, where=where)
    if k is not None and k < 1:
        raise ConfigError(
            f"{where}.moe_experts: expected a positive expert count, "
            f"got {value!r}")
    return k


def _normalize_moe_layer_shed(value, where: str = "model"):
    """Validate a ``moe_layer_shed`` skip probability in (0, 1)."""
    p = _coerce_num("moe_layer_shed", value, float, where=where)
    if p is not None and not 0.0 < p < 1.0:
        raise ConfigError(
            f"{where}.moe_layer_shed: expected a probability in (0, 1), "
            f"got {value!r}")
    return p


def _normalize_moe_prestage(value, where: str = "model"):
    """Validate a ``moe_prestage`` targeting mode: "ranked" or "keepers"."""
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in ("ranked", "keepers"):
        raise ConfigError(
            f"{where}.moe_prestage: expected 'ranked' or 'keepers', "
            f"got {value!r}")
    return mode


def _normalize_fast_disk(value, where: str = "model"):
    """Validate a ``stream_fast_disk`` recipe: "auto", "on" or "off"."""
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ConfigError(
            f"{where}.stream_fast_disk: expected 'auto', 'on' or 'off', "
            f"got {value!r}")
    return mode


def _normalize_stream(value, where: str = "model", legacy=None):
    """Normalize a ``stream:`` value to one of None / "experts" / "cpu".

    "experts" (the ``--stream-experts`` placement) streams only the
    routed-expert stacks from disk while the every-token layers + KV cache
    stay on GPU; "cpu" (``--stream-cpu``) runs the whole model on the CPU
    device with all weights streamed through the page cache.

    ``legacy`` carries a deprecated ``cpu_moe:`` value, honored with a rename
    warning when ``stream:`` itself is unset (old semantics preserved:
    true/full -> "cpu", hybrid -> "experts"; the removed integer partial
    offload -> "experts"). A bare ``stream: true`` is ambiguous between the
    two placements (and would invert the old meaning of ``cpu_moe: true``),
    so it warns and is ignored. Unrecognized values warn."""
    import warnings
    if value is None and legacy not in (None, False, ""):
        if legacy is True:
            mapped = "cpu"
        elif isinstance(legacy, int):  # legacy `cpu_moe: N` partial offload
            mapped = "experts"
        else:
            s = str(legacy).strip().lower()
            if s in ("full", "true", "cpu", "all", "yes", "on"):
                mapped = "cpu"
            elif s in ("hybrid", "gpu-spine", "spine"):
                mapped = "experts"
            else:
                mapped = None
        if mapped:
            msg = (f"{where}: `cpu_moe: {legacy}` is renamed; use "
                   f"`stream: {mapped}`")
        else:
            msg = (f"{where}: unrecognized cpu_moe value {legacy!r}; ignoring "
                   "(the key is renamed to `stream: experts|cpu`)")
        warnings.warn(msg, stacklevel=3)
        return mapped
    if value is None or value is False or value == "":
        return None
    if value is True:
        warnings.warn(
            f"{where}: `stream: true` is ambiguous; use `stream: experts` "
            "(experts stream, rest of the model + KV on GPU) or "
            "`stream: cpu` (whole model on the CPU device); ignoring",
            stacklevel=3,
        )
        return None
    s = str(value).strip().lower()
    if s in ("experts", "hybrid", "gpu"):
        return "experts"
    if s in ("cpu", "full", "all"):
        return "cpu"
    if s in ("false", "none", "off", "no"):
        return None
    warnings.warn(
        f"{where}: unrecognized stream value {value!r} "
        "(expected experts, cpu, or false); ignoring",
        stacklevel=3,
    )
    return None


def _normalize_cache(where: str, raw) -> dict:
    """Validate a ``cache:`` block (typo'd keys warn) and normalize its ``disk``
    tier. ``where`` names the block itself (e.g. ``server.cache``). ``disk``
    accepts a boolean shorthand: ``true`` enables the SSD tier at
    :func:`default_apc_disk_path`; ``false`` disables it, overriding any
    inherited ``disk.path`` (the tier is keyed on a present path)."""
    cache = dict(_section_mapping(where, raw))
    _warn_unknown_keys(where, cache, _CACHE_KEYS)
    if "disk" in cache:
        disk = cache["disk"]
        if disk is True:
            disk = {"path": default_apc_disk_path()}
        elif disk is False:
            disk = {"path": None}
        else:
            disk = dict(_section_mapping(f"{where}.disk", disk))
        _warn_unknown_keys(f"{where}.disk", disk, CACHE_DISK_ENV)
        cache["disk"] = disk
    return cache


def _coerce_ratio(key: str, v, *, where: str = "server"):
    """decode_prefill_ratio: a float, or the literal string "auto"
    (case-insensitive, surrounding whitespace stripped)."""
    if isinstance(v, str) and v.strip().lower() == "auto":
        return "auto"
    return _coerce_num(key, v, float, where=where)


def _parse_systemone(raw) -> SystemoneCfg:
    raw = _section_mapping("server.systemone", raw)
    _warn_unknown_keys("server.systemone", raw, _SYSTEMONE_KEYS, strict=True)
    where = "server.systemone"
    canvas = _coerce_num("canvas", raw.get("canvas", 64), int, where=where)
    if canvas is None or canvas <= 0 or canvas % 16:
        raise ConfigError(
            f"{where}.canvas: expected a positive multiple of 16, got {canvas!r}")
    counts = {}
    for key, default in (("max_questions", 64), ("max_samples", 32)):
        n = _coerce_num(key, raw.get(key, default), int, where=where)
        if n is None or n < 1:
            raise ConfigError(f"{where}.{key}: expected a positive int, got {n!r}")
        counts[key] = n
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise ConfigError(f"{where}.model: expected a model id, got {model!r}")
    think = raw.get("think", 0)
    if think != "auto":
        think = _coerce_num("think", think, int, where=where)
        if think is None or not 0 <= think <= 4096:
            raise ConfigError(
                f'{where}.think: expected 0 to 4096 or "auto", got {raw.get("think")!r}')
    threshold = _coerce_num("think_threshold",
                            raw.get("think_threshold", THINK_THRESHOLD), float,
                            where=where)
    if threshold is None or not 0 < threshold <= 1:
        raise ConfigError(f"{where}.think_threshold: expected a number above 0 "
                          f"and at most 1, got {threshold!r}")
    budget = _coerce_num("think_budget", raw.get("think_budget", THINK_BUDGET), int,
                         where=where)
    if budget is None or not 1 <= budget <= 4096:
        raise ConfigError(f"{where}.think_budget: expected 1 to 4096, got {budget!r}")
    return SystemoneCfg(model=model or None, canvas=canvas,
                        constrained=bool(raw.get("constrained", True)),
                        think=think, think_threshold=threshold, think_budget=budget,
                        **counts)


def _coerce_num(key: str, v, cast, *, where: str = "server"):
    """Coerce a numeric config key (YAML may carry it quoted as a string),
    raising a ConfigError naming the key and the bad value. ``None`` passes."""
    if v is None:
        return None
    try:
        if isinstance(v, bool):   # bool is an int subclass; `port: true` is a typo
            raise ValueError
        return cast(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}.{key}: expected a {cast.__name__}, got {v!r}")


# The CLI's own flag spellings, accepted in config sampling blocks too.
_SAMPLING_KEY_ALIASES = {"temp": "temperature"}


def _normalize_sampling(where: str, raw) -> dict:
    """A sampling mapping with alias spellings canonicalized (``temp`` is what
    every CLI flag says, so configs get it too); canonical key wins if both
    appear."""
    out = dict(_section_mapping(where, raw))
    for alias, canon in _SAMPLING_KEY_ALIASES.items():
        if alias in out:
            out.setdefault(canon, out[alias])
            del out[alias]
    return out


def _parse_profile(name: str, raw: dict) -> Profile:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"profile {name!r} must be a mapping, e.g. `{name}: {{sampling: "
            f"{{temperature: 0.7}}}}` (got {type(raw).__name__}: {raw!r})")
    _warn_unknown_keys(f"profile {name!r}", raw, _PROFILE_KEYS, strict=True)
    sampling = _normalize_sampling(f"profile {name!r} sampling",
                                   raw.get("sampling"))
    _warn_unknown_keys(f"profile {name!r} sampling", sampling, SAMPLING_KEYS)
    _validate_stop(f"profile {name!r}", sampling)
    load = _section_mapping(f"profile {name!r} load", raw.get("load"))
    cache = _normalize_cache(f"profile {name!r} cache", raw.get("cache"))
    ctk = drop_template_call_keys(
        f"profile {name!r} chat_template_kwargs",
        _section_mapping(f"profile {name!r} chat_template_kwargs",
                         raw.get("chat_template_kwargs")))
    _warn_unknown_keys(f"profile {name!r} load", load, LOAD_ENV)
    return Profile(
        name=name,
        extends=raw.get("extends"),
        sampling=sampling,
        load=dict(load),
        cache=dict(cache),
        system=raw.get("system"),
        chat_template=raw.get("chat_template"),
        chat_template_kwargs=dict(ctk),
        thinking=raw.get("thinking"),
        reasoning_effort=raw.get("reasoning_effort"),
    )


def _parse_model(model_id: str, raw: dict) -> ModelCfg:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"model {model_id!r} must be a mapping, e.g. `{model_id}: "
            f"{{path: {raw!r}}}` (got {type(raw).__name__}: {raw!r})")
    if "path" not in raw:
        raise ConfigError(
            f"model {model_id!r} has no `path` (keys present: {sorted(raw)})")
    if not isinstance(raw["path"], str):
        raise ConfigError(
            f"model {model_id!r} path must be a string "
            f"(got {type(raw['path']).__name__}: {raw['path']!r})")
    _warn_unknown_keys(f"model {model_id!r}", raw, _MODEL_KEYS, strict=True)
    ov = dict(_section_mapping(f"model {model_id!r} overrides",
                               raw.get("overrides")))
    _warn_unknown_keys(f"model {model_id!r} overrides", ov, _OVERRIDE_KEYS,
                       strict=True)
    if "sampling" in ov:
        ov["sampling"] = _normalize_sampling(
            f"model {model_id!r} overrides.sampling", ov.get("sampling"))
    _warn_unknown_keys(f"model {model_id!r} overrides.sampling",
                       ov.get("sampling") or {}, SAMPLING_KEYS)
    _validate_stop(f"model {model_id!r} overrides", ov.get("sampling") or {})
    for g in ("load", "chat_template_kwargs"):
        if g in ov:
            ov[g] = _section_mapping(f"model {model_id!r} overrides.{g}",
                                     ov.get(g))
    if ov.get("chat_template_kwargs"):
        ov["chat_template_kwargs"] = drop_template_call_keys(
            f"model {model_id!r} overrides.chat_template_kwargs", ov["chat_template_kwargs"])
    if "cache" in ov:
        ov["cache"] = _normalize_cache(f"model {model_id!r} overrides.cache",
                                       ov.get("cache"))
    _warn_unknown_keys(f"model {model_id!r} overrides.load",
                       ov.get("load") or {}, LOAD_ENV)
    tweaks = raw.get("profiles") or {}
    if not isinstance(tweaks, dict):
        raise ConfigError(
            f"model {model_id!r} profiles must be a mapping of "
            f"{{profile-name: {{sampling: ...}}}} (got {type(tweaks).__name__})")
    norm_tweaks = {}
    for pname, pv in tweaks.items():
        pv = dict(_section_mapping(f"model {model_id!r} profiles.{pname!r}", pv))
        _warn_unknown_keys(f"model {model_id!r} profiles.{pname!r}", pv,
                           _OVERRIDE_KEYS, strict=True)
        if "sampling" in pv:
            pv["sampling"] = _normalize_sampling(
                f"model {model_id!r} profiles.{pname!r} sampling",
                pv.get("sampling"))
        _warn_unknown_keys(f"model {model_id!r} profiles.{pname!r} sampling",
                           pv.get("sampling") or {}, SAMPLING_KEYS)
        _validate_stop(f"model {model_id!r} profiles.{pname!r}",
                       pv.get("sampling") or {})
        for g in ("load", "chat_template_kwargs"):
            if g in pv:
                pv[g] = _section_mapping(
                    f"model {model_id!r} profiles.{pname!r}.{g}", pv.get(g))
        if pv.get("chat_template_kwargs"):
            pv["chat_template_kwargs"] = drop_template_call_keys(
                f"model {model_id!r} profiles.{pname!r}.chat_template_kwargs",
                pv["chat_template_kwargs"])
        if "cache" in pv:
            pv["cache"] = _normalize_cache(
                f"model {model_id!r} profiles.{pname!r}.cache", pv.get("cache"))
        _warn_unknown_keys(f"model {model_id!r} profiles.{pname!r} load",
                           pv.get("load") or {}, LOAD_ENV)
        norm_tweaks[str(pname)] = pv
    return ModelCfg(
        id=model_id,
        path=raw["path"],
        profile=raw.get("profile"),
        family=raw.get("family"),
        profiles=norm_tweaks,
        mmproj=raw.get("mmproj"),
        draft_gguf=raw.get("draft_gguf"),
        native_mtp=bool(raw.get("native_mtp", False)),
        adapter=raw.get("adapter"),
        stream=_normalize_stream(raw.get("stream"), f"model {model_id!r}",
                                 legacy=raw.get("cpu_moe")),
        moe_experts=_normalize_moe_experts(
            raw.get("moe_experts"), f"model {model_id!r}"),
        moe_expert_mass=_normalize_moe_expert_mass(
            raw.get("moe_expert_mass"), f"model {model_id!r}"),
        moe_miss_shed=_normalize_moe_expert_mass(
            raw.get("moe_miss_shed"), f"model {model_id!r}",
            key="moe_miss_shed"),
        moe_layer_shed=_normalize_moe_layer_shed(
            raw.get("moe_layer_shed"), f"model {model_id!r}"),
        moe_prestage=_normalize_moe_prestage(
            raw.get("moe_prestage"), f"model {model_id!r}"),
        prefill_feeder=_normalize_optional_bool(
            raw.get("prefill_feeder"), "prefill_feeder", f"model {model_id!r}"),
        decode_feeder=_normalize_optional_bool(
            raw.get("decode_feeder"), "decode_feeder", f"model {model_id!r}"),
        stream_fast_disk=_normalize_fast_disk(
            raw.get("stream_fast_disk"), f"model {model_id!r}"),
        speculative=bool(raw.get("speculative", False)),
        speculative_width_cap=_normalize_speculative_width_cap(
            raw.get("speculative_width_cap"), f"model {model_id!r}"),
        overrides=dict(ov),
        pin=bool(raw.get("pin", False)),
        # An explicit `ttl_s: null` means never unload, as it does in
        # server.defaults. Only an absent key inherits the server value.
        ttl_s=(0.0 if "ttl_s" in raw and raw["ttl_s"] is None
               else _coerce_num("ttl_s", raw.get("ttl_s"), float,
                                where=f"model {model_id!r}")),
    )


def _parse_rule(raw: dict) -> Rule:
    raw = _section_mapping("rules entry", raw)
    if "match" not in raw or "profile" not in raw:
        raise ConfigError(
            f"rule {raw!r} needs both `match` and `profile`")
    _warn_unknown_keys(f"rule {raw.get('match')!r}", raw, _RULE_KEYS, strict=True)
    return Rule(match=str(raw["match"]), profile=str(raw["profile"]))


def _parse_preload(raw):
    """``server.defaults.preload``: model ids to warm at startup - the string
    ``"all"`` or a list of ids. Ids are checked against ``models:`` in
    ``_validate`` (aliases are not addressable here)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == "all":
            return "all"
        raise ConfigError(
            f"server.defaults.preload must be \"all\" or a list of model ids, "
            f"got {raw!r}")
    if isinstance(raw, (list, tuple)):
        return [str(m) for m in raw]
    raise ConfigError(
        f"server.defaults.preload must be \"all\" or a list of model ids, "
        f"got {type(raw).__name__}")


def _parse_discover(raw: dict) -> DiscoverSpec:
    raw = _section_mapping("discover entry", raw)
    _warn_unknown_keys("discover entry", raw, _DISCOVER_KEYS, strict=True)
    return DiscoverSpec(
        dir=raw.get("dir"),
        recursive=bool(raw.get("recursive", False)),
        pair_mmproj=bool(raw.get("pair_mmproj", True)),
        speculative=raw.get("speculative", "auto"),
    )


def _parse_mcp_list(mcp_raw, where: str) -> list:
    """Parse a list of MCP server entries (shared by the ``assistant:`` block
    and per-alias ``server.assistants.<id>.mcp`` scoping lists)."""
    servers: list = []
    if not isinstance(mcp_raw, list):
        raise ConfigError(f"{where} must be a list of server entries")
    for i, entry in enumerate(mcp_raw):
        here = f"{where}[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{here}: expected a mapping, got {entry!r}")
        _warn_unknown_keys(here, entry, _MCP_KEYS, strict=True)
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{here}: `name` is required")
        if any(s.name == name for s in servers):
            raise ConfigError(f"{here}: duplicate server name {name!r}")
        command, url = entry.get("command"), entry.get("url")
        if bool(command) == bool(url):
            raise ConfigError(
                f"{here}: exactly one of `command` (stdio) or `url` (HTTP) "
                "is required")
        if isinstance(command, str):
            import shlex
            command = shlex.split(command)
        if command is not None and (
                not isinstance(command, list)
                or not all(isinstance(c, (str, int, float)) for c in command)):
            raise ConfigError(f"{here}.command: expected an argv list")
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            raise ConfigError(f"{here}.env: expected a mapping")
        servers.append(McpServerCfg(
            name=name,
            command=[str(c) for c in (command or [])],
            url=str(url) if url else None,
            env={str(k): str(v) for k, v in env.items()}))
    return servers


def _parse_assistant(raw) -> AssistantCfg:
    """Parse + validate the top-level ``assistant:`` block (tools + memory for
    the built-in tool-loop assistant)."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"assistant must be a mapping (got {type(raw).__name__}: {raw!r})")
    _warn_unknown_keys("assistant", raw, _ASSISTANT_KEYS, strict=True)

    def num(where: str, v, cast, default, minimum):
        if v is None:
            return default
        try:
            if isinstance(v, bool):
                raise ValueError
            v = cast(v)
        except (TypeError, ValueError):
            raise ConfigError(
                f"assistant.{where}: expected a {cast.__name__}, got {v!r}")
        if v < minimum:
            raise ConfigError(f"assistant.{where}: must be >= {minimum}")
        return v

    servers = _parse_mcp_list(raw.get("mcp") or [], "assistant.mcp")

    mem_raw = raw.get("memory") or {}
    if not isinstance(mem_raw, dict):
        raise ConfigError("assistant.memory must be a mapping")
    _warn_unknown_keys("assistant.memory", mem_raw, _MEMORY_KEYS, strict=True)
    memory = AssistantMemory(
        enabled=bool(mem_raw.get("enabled", True)),
        path=str(mem_raw["path"]) if mem_raw.get("path") else None,
        top_k=num("memory.top_k", mem_raw.get("top_k"), int, 4, 1),
        extract=bool(mem_raw.get("extract", True)),
        ttl_days=num("memory.ttl_days", mem_raw.get("ttl_days"),
                     float, None, 0.01),
        max_items=num("memory.max_items", mem_raw.get("max_items"),
                      int, 20000, 1))
    return AssistantCfg(
        max_tool_rounds=num("max_tool_rounds", raw.get("max_tool_rounds"),
                            int, 8, 1),
        tool_timeout_s=num("tool_timeout_s", raw.get("tool_timeout_s"),
                           float, 60.0, 1.0),
        mcp=servers, memory=memory)


def _parse_cors_origins(raw) -> list[str]:
    """``server.cors_origins``: browser origins, normalized, duplicates
    dropped."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(o, str) for o in raw):
        raise ConfigError("server.cors_origins: expected a list of origins, such as "
                          "[https://chat.example.com]")
    out = []
    for entry in raw:
        try:
            origin = normalize_cors_entry(entry)
        except ValueError as e:
            raise ConfigError(f"server.cors_origins entry {entry!r}: {e}") from None
        if origin not in out:
            out.append(origin)
    return out


def _parse_assistant_aliases(raw) -> dict:
    """Parse ``server.assistants:`` into ``{alias_id: AssistantAlias}``.
    Cross-checks against models/aliases happen in :func:`_validate`."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"server.assistants must be a mapping of alias id -> entry "
            f"(got {type(raw).__name__}: {raw!r})")
    out: dict = {}
    for aid, entry in raw.items():
        aid = str(aid)
        where = f"server.assistants.{aid}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: expected a mapping, got {entry!r}")
        _warn_unknown_keys(where, entry, _ASSISTANT_ALIAS_KEYS, strict=True)
        model = entry.get("model")
        if not model:
            raise ConfigError(f"{where}: `model` is required (the underlying "
                              "configured model id)")
        mcp = entry.get("mcp", None)
        out[aid] = AssistantAlias(
            model=str(model),
            memory=bool(entry.get("memory", False)),
            mcp=None if mcp is None else _parse_mcp_list(mcp, f"{where}.mcp"))
    return out


def _parse_talk(raw) -> TalkCfg:
    """Parse + validate the top-level ``talk:`` block into a :class:`TalkCfg`."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"talk must be a mapping, e.g. `talk: {{voice: af_heart}}` "
            f"(got {type(raw).__name__}: {raw!r})")
    if "agent" in raw:
        raise ConfigError(
            "talk.agent has moved: define a top-level `assistant:` block "
            "(same fields) and set `talk.brain: assistant`")
    _warn_unknown_keys("talk", raw, _TALK_KEYS, strict=True)
    vad_raw = _section_mapping("talk.vad", raw.get("vad"))
    _warn_unknown_keys("talk.vad", vad_raw, _TALK_VAD_KEYS, strict=True)

    def num(where: str, v, cast, default):
        if v is None:
            return default
        try:
            if isinstance(v, bool):
                raise ValueError
            return cast(v)
        except (TypeError, ValueError):
            raise ConfigError(f"talk.{where}: expected a {cast.__name__}, got {v!r}")

    mode = str(raw.get("mode", "wake")).strip().lower()
    if mode not in TALK_MODES:
        raise ConfigError(
            f"talk.mode: {mode!r} is not one of {'/'.join(TALK_MODES)}")
    brain = str(raw.get("brain", "chat")).strip().lower()
    if brain not in TALK_BRAINS:
        raise ConfigError(
            f"talk.brain: {brain!r} is not one of {'/'.join(TALK_BRAINS)}")
    from gmlx.talk.hotkey import PUSH_TO_TALK_MODIFIERS
    ptt = str(raw.get("push_to_talk_modifier", "globe")).strip().lower()
    if ptt not in PUSH_TO_TALK_MODIFIERS:
        raise ConfigError(
            f"talk.push_to_talk_modifier: {ptt!r} is not one of "
            f"{'/'.join(PUSH_TO_TALK_MODIFIERS)}")
    from gmlx.serve.tts import SPEED_MAX, SPEED_MIN
    speed = num("speed", raw.get("speed"), float, 1.0)
    if not SPEED_MIN <= speed <= SPEED_MAX:
        raise ConfigError(
            f"talk.speed: {speed:g} is not between {SPEED_MIN:g} and {SPEED_MAX:g}")
    dev = lambda v: None if v is None else str(v)  # noqa: E731
    return TalkCfg(
        model=str(raw["model"]) if raw.get("model") else None,
        voice=str(raw["voice"]) if raw.get("voice") else None,
        speed=speed,
        # Absent -> speakable-output default; an explicit empty string is
        # the opt-out for "no system prompt at all".
        system=(str(raw["system"]) if raw.get("system")
                else (None if "system" in raw else DEFAULT_TALK_SYSTEM)),
        language=str(raw["language"]) if raw.get("language") else None,
        max_tokens=num("max_tokens", raw.get("max_tokens"), int, None),
        mode=mode,
        wake_word=str(raw.get("wake_word") or "hey assistant"),
        wake_threshold=num("wake_threshold", raw.get("wake_threshold"),
                           float, 0.3),
        push_to_talk_modifier=ptt,
        vad=TalkVad(
            threshold=num("vad.threshold", vad_raw.get("threshold"),
                          float, 0.6),
            silence_ms=num("vad.silence_ms", vad_raw.get("silence_ms"),
                           float, 550.0),
            min_speech_ms=num("vad.min_speech_ms",
                              vad_raw.get("min_speech_ms"), float, 300.0),
            pre_roll_ms=num("vad.pre_roll_ms", vad_raw.get("pre_roll_ms"),
                            float, 400.0),
        ),
        input_device=dev(raw.get("input_device")),
        output_device=dev(raw.get("output_device")),
        chime=bool(raw.get("chime", True)),
        brain=brain,
    )


_LAUNCH_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Apple container's volume name pattern (VolumeConfiguration.swift).
LAUNCH_VOLUME_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# The longest volume name, so the project's name for it and its lock file
# name stay within the 255 bytes of a file name.
LAUNCH_VOLUME_NAME_MAX = 200
# Debian package names; they reach a build argument, so nothing else passes.
_LAUNCH_PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.-]+")
# A size as Apple container parses it, with the unit required here, so that a
# bare 4096 cannot mean 4096 bytes.
_LAUNCH_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([kmgtp])(?:i?b)?\s*", re.I)
_SIZE_SHIFT = {"k": 10, "m": 20, "g": 30, "t": 40, "p": 50}
LAUNCH_VOLUME_MIN_BYTES = 1 << 20


def parse_size_bytes(text: str) -> int | None:
    """Bytes for a size such as ``8G``, ``512M`` or ``1.5GiB``, or None when
    the text is not one."""
    m = _LAUNCH_SIZE.fullmatch(str(text))
    if m is None:
        return None
    return int(float(m.group(1)) * (1 << _SIZE_SHIFT[m.group(2).lower()]))


def parse_volume_spec(spec: str) -> tuple[str, str, str | None]:
    """``(name, guest path, size or None)`` for a ``NAME:/path[:SIZE]`` volume
    entry. Raises :class:`ConfigError` naming what is wrong."""
    parts = str(spec).split(":")
    if len(parts) not in (2, 3):
        raise ConfigError(f"volume {spec}: write NAME:/path or NAME:/path:SIZE, such as "
                          "pgdata:/var/lib/postgresql:8G")
    name, path = parts[0], parts[1]
    size = parts[2] if len(parts) == 3 else None
    if not LAUNCH_VOLUME_NAME.fullmatch(name):
        raise ConfigError(f"volume {spec}: the name must start with a letter "
                          f"or digit and use only letters, digits, _ . and -")
    if len(name) > LAUNCH_VOLUME_NAME_MAX:
        raise ConfigError(f"volume {spec}: the name has {len(name)} characters, and the "
                          f"most is {LAUNCH_VOLUME_NAME_MAX}. Use a shorter name.")
    if not path.startswith("/"):
        raise ConfigError(f"volume {spec}: the container path must start with /")
    if size is not None:
        nbytes = parse_size_bytes(size)
        if nbytes is None:
            raise ConfigError(f"volume {spec}: {size} is not a size such as 512M, 8G "
                              "or 1T")
        if nbytes < LAUNCH_VOLUME_MIN_BYTES:
            raise ConfigError(f"volume {spec}: the smallest size is 1M")
    return name, path, size


def _check_launch_volumes(where: str, entries: list) -> None:
    """Refuse one volume name at two guest paths or with two sizes, since the
    VM would attach one disk image twice."""
    seen: dict = {}
    for entry in entries:
        try:
            name, path, size = parse_volume_spec(entry)
        except ConfigError as e:
            raise ConfigError(f"{where}: {e}") from None
        if name in seen:
            path0, size0 = seen[name]
            if path0 != path:
                raise ConfigError(f"{where}: volume {name} is used at both {path0} and "
                                  f"{path}")
            if size0 != size:
                raise ConfigError(f"{where}: volume {name} has two sizes, "
                                  f"{size0 or 'the default'} and "
                                  f"{size or 'the default'}")
        seen[name] = (path, size)


def _parse_launch_level(where: str, raw: dict, keys) -> dict:
    """Shape checks for one level of ``launch.container``. Returns the parsed
    values of the keys present; nothing here touches the filesystem."""
    _warn_unknown_keys(where, raw, keys, strict=True)
    # A path, a name or a command with a NUL fails later without naming its key.
    for key, value in raw.items():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, str) and "\0" in v:
                raise ConfigError(f"{where}.{key}: {v!r} holds a NUL character, which no "
                                  "path, name or command can hold. Remove it.")
    out: dict = {}

    def strings(key) -> list:
        value = raw.get(key)
        if value is None:
            return []
        if not isinstance(value, list) or not all(
                isinstance(v, str) and v.strip() for v in value):
            raise ConfigError(f"{where}.{key}: expected a list of strings, "
                              f"got {value!r}")
        if key == "env":        # a value keeps its spaces, as the shell would pass it
            return [(lambda n, eq, v: n.strip() + eq + v)(*e.partition("=")) if "=" in e
                    else e.strip() for e in value]
        return [v.strip() for v in value]

    def flag(key):
        value = raw.get(key)
        if value is not None and not isinstance(value, bool):
            raise ConfigError(f"{where}.{key}: expected true or false, got {value!r}")
        return value

    def choice(key, choices):
        value = raw.get(key)
        if value is None:
            return None
        if key == "clipboard" and value is False:    # YAML reads a bare off as false
            value = "off"
        if isinstance(value, bool):
            raise ConfigError(f"{where}.{key} takes {' or '.join(choices)}, not true or "
                              "false. YAML reads a bare yes, no, on or off as true or "
                              "false, so write the word you mean.")
        if not isinstance(value, str) or value not in choices:
            raise ConfigError(f"{where}.{key}: {value!r} is not one of "
                              f"{'/'.join(choices)}")
        return value

    for key in ("enabled", "mount_cwd", "open_browser"):
        if key in raw:
            out[key] = flag(key)
    if "ssh_agent" in raw:
        agent = raw["ssh_agent"]
        if isinstance(agent, str):
            agent = agent.strip()
            if not agent.startswith(("/", "~")):
                raise ConfigError(f"{where}.ssh_agent: expected true, false or the full path "
                                  f"of an SSH agent socket, got {raw['ssh_agent']!r}")
        elif agent is not None and not isinstance(agent, bool):
            raise ConfigError(f"{where}.ssh_agent: expected true, false or the full path "
                              f"of an SSH agent socket, got {agent!r}")
        out["ssh_agent"] = agent
    for key in ("mounts", "volumes", "env", "packages", "seed", "assistants"):
        if key in raw:
            out[key] = list(dict.fromkeys(strings(key)))
    if "network" in raw:
        out["network"] = choice("network", LAUNCH_NETWORKS)
    if "clipboard" in raw:
        out["clipboard"] = choice("clipboard", LAUNCH_CLIPBOARD)
    if raw.get("cpus") is not None:
        cpus = raw["cpus"]
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            raise ConfigError(f"{where}.cpus: expected a whole number of at "
                              f"least 1, got {cpus!r}")
        out["cpus"] = cpus
    if raw.get("memory") is not None:
        memory = raw["memory"]
        if not isinstance(memory, str) or parse_size_bytes(memory) is None:
            raise ConfigError(f"{where}.memory: {memory!r} is not a size such "
                              f"as 4G or 6144M")
        out["memory"] = memory.strip()
    if raw.get("forward") is not None:
        ports = raw["forward"]
        if not isinstance(ports, list) or not all(
                isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65535
                for v in ports):
            raise ConfigError(f"{where}.forward: expected a list of port "
                              f"numbers from 1 to 65535, got {ports!r}")
        out["forward"] = list(dict.fromkeys(ports))
    for entry in out.get("env", []):
        name = entry.split("=", 1)[0]
        if not _LAUNCH_ENV_NAME.fullmatch(name):
            raise ConfigError(f"{where}.env: {entry!r} is not NAME or NAME=VALUE")
        if name in LAUNCH_RESERVED_ENV:
            raise ConfigError(f"{where}.env: launch sets {name} in the "
                              f"container itself, so it cannot be configured")
    if "volumes" in out:
        _check_launch_volumes(f"{where}.volumes", out["volumes"])
    for name in out.get("packages", []):
        if not _LAUNCH_PACKAGE.fullmatch(name):
            raise ConfigError(f"{where}.packages: {name!r} is not a Debian "
                              f"package name")
    for key in ("image", "build"):
        if raw.get(key) is not None:
            value = raw[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{where}.{key}: expected a non-empty string, "
                                  f"got {value!r}")
            out[key] = value.strip()
    if raw.get("command") is not None:
        command = raw["command"]
        if command != "image" and not (
                isinstance(command, list) and command
                and all(isinstance(v, str) and v for v in command)):
            raise ConfigError(f"{where}.command: expected a list of strings "
                              f"such as [my-client, --flag], or the word image, "
                              f"got {command!r}")
        out["command"] = command if command == "image" else list(command)
    return out


def _parse_launch(raw) -> LaunchCfg:
    """Parse + validate the top-level ``launch:`` block. Shape only: paths are
    checked when a launch resolves them."""
    raw = _section_mapping("launch", raw)
    _warn_unknown_keys("launch", raw, _LAUNCH_KEYS, strict=True)
    box = _section_mapping("launch.container", raw.get("container"))
    values = _parse_launch_level("launch.container", box, _LAUNCH_CONTAINER_KEYS)
    cfg = LaunchContainerCfg(**{k: v for k, v in values.items() if v is not None})
    for client, client_raw in _section_mapping(
            "launch.container.clients", box.get("clients")).items():
        if client not in LAUNCH_CLIENTS:
            raise ConfigError(f"launch.container.clients: {client!r} is not a "
                              f"launch client (known: {', '.join(sorted(LAUNCH_CLIENTS))})")
        where = f"launch.container.clients.{client}"
        client_values = _parse_launch_level(
            where, _section_mapping(where, client_raw), _LAUNCH_CLIENT_KEYS)
        if client_values.get("image") and client_values.get("build"):
            raise ConfigError(f"{where}: image and build cannot both be set. Keep one "
                              "of them.")
        if client_values.get("image") and client_values.get("packages"):
            raise ConfigError(f"{where}: packages apply only to the image gmlx "
                              f"builds, not to image {client_values['image']!r}")
        cfg.clients[client] = LaunchClientCfg(**client_values)
        _check_launch_volumes(f"{where}.volumes", cfg.for_client(client).volumes)
    return LaunchCfg(container=cfg)


def _parse_launch_leniently(raw) -> LaunchCfg:
    """The ``launch`` block for the server's own config. Only ``gmlx launch``
    uses the block, and it parses the block strictly itself, so an error here
    is one warning and never stops the server."""
    try:
        return _parse_launch(raw)
    except ConfigError as e:
        import warnings
        warnings.warn(f"{e}. The server ignores the launch block. Until it is fixed, "
                      "gmlx launch refuses a client that the block runs in a container, "
                      "and runs the others on the Mac.", stacklevel=3)
        return LaunchCfg()


def build_config(doc: dict) -> ServerCfg:
    """Build (and validate) a :class:`ServerCfg` from a parsed YAML mapping. Split out
    from :func:`load_config` so discovery / tests can build a config in memory."""
    doc = doc or {}
    if "container" in doc:
        raise ConfigError("config (top level): unknown key container. Did you mean "
                          "launch: container:?")
    _warn_unknown_keys("config (top level)", doc, _TOP_KEYS, strict=True)
    srv = _section_mapping("server", doc.get("server"))
    _warn_unknown_keys("server", srv, _SERVER_KEYS, strict=True)
    srv_cache = _normalize_cache("server.cache", srv.get("cache"))
    md = srv.get("model_dirs")
    model_dirs = _as_list(md) if not isinstance(md, str) else [md]
    dft = _section_mapping("server.defaults", srv.get("defaults"))
    _warn_unknown_keys("server.defaults", dft, _DEFAULTS_KEYS, strict=True)
    cfg = ServerCfg(
        host=srv.get("host", "127.0.0.1"),
        port=_coerce_num("port", srv.get("port", 8080), int),
        model_dirs=[str(d) for d in model_dirs],
        budget_gb=_coerce_num("budget_gb", srv.get("budget_gb"), float),
        max_models=_coerce_num("max_models", srv.get("max_models"), int),
        hf_cache=bool(srv.get("hf_cache", False)),
        cache=srv_cache,
        stt=srv.get("stt") or None,   # raw; resolved (aliases etc.) at serve time
        tts=srv.get("tts") or None,   # raw; resolved (aliases etc.) at serve time
        embeddings=srv.get("embeddings") or None,   # raw; resolved at serve time
        rerank=srv.get("rerank") or None,           # raw; resolved at serve time
        systemone=_parse_systemone(srv.get("systemone")),
        api_key=str(srv["api_key"]) if srv.get("api_key") else None,
        no_auth=bool(srv.get("no_auth", False)),
        media_urls=bool(srv.get("media_urls", False)),
        cors_origins=_parse_cors_origins(srv.get("cors_origins")),
        menubar=bool(srv.get("menubar", True)),
        token_queue_timeout_s=_coerce_num(
            "token_queue_timeout_s", srv.get("token_queue_timeout_s"), float),
        prefill_step_size=_coerce_num(
            "prefill_step_size", srv.get("prefill_step_size"), int),
        dtype=_validate_server_dtype(srv.get("dtype")),
        decode_prefill_ratio=_coerce_ratio(
            "decode_prefill_ratio", srv.get("decode_prefill_ratio")),
        prefill_tick_ms=_coerce_num(
            "prefill_tick_ms", srv.get("prefill_tick_ms"), float),
        cache_limit_gb=_coerce_num(
            "cache_limit_gb", srv.get("cache_limit_gb"), float),
        family_defaults=bool(srv.get("family_defaults", True)),
        stochastic_mtp=bool(srv.get("stochastic_mtp", False)),
        gpu_keepwarm=(None if srv.get("gpu_keepwarm") is None
                      else bool(srv.get("gpu_keepwarm"))),
        defaults=ServerDefaults(
            profile=dft.get("profile"),
            ttl_s=_coerce_num("defaults.ttl_s", dft.get("ttl_s", 900.0), float),
            model=dft.get("model"),
            preload=_parse_preload(dft.get("preload")),
        ),
        profiles={n: _parse_profile(n, r) for n, r in
                  _section_mapping("profiles", doc.get("profiles")).items()},
        rules=[_parse_rule(r) for r in _section_list("rules", doc.get("rules"))],
        models={mid: _parse_model(mid, r) for mid, r in
                _section_mapping("models", doc.get("models")).items()},
        aliases={str(k): str(v) for k, v in
                 _section_mapping("aliases", doc.get("aliases")).items()},
        discover=[_parse_discover(d) for d in
                  _section_list("discover", doc.get("discover"))],
        talk=_parse_talk(doc.get("talk")),
        launch=_parse_launch_leniently(doc.get("launch")),
        assistant=_parse_assistant(doc.get("assistant")),
        assistants=_parse_assistant_aliases(srv.get("assistants")),
        assistant_allow_remote=bool(srv.get("assistant_allow_remote", False)),
        theme=str(doc["theme"]) if doc.get("theme") else None,
        themes=_section_mapping("themes", doc.get("themes")),
    )
    cfg.talk.assistant = cfg.assistant   # one shared settings object
    _validate(cfg)
    return cfg


def load_config(path) -> ServerCfg:
    """Load + validate a YAML config file into a :class:`ServerCfg`."""
    p = Path(os.path.expanduser(str(path)))
    try:
        with open(p) as f:
            doc = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {p}")
    except IsADirectoryError:
        raise ConfigError(f"config path is a directory, expected a YAML "
                          f"file: {p}")
    except UnicodeDecodeError:
        raise ConfigError(f"config file is not text (YAML expected): {p}")
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {p}: {e}")
    if doc is not None and not isinstance(doc, dict):
        raise ConfigError(f"config root must be a mapping, got {type(doc).__name__}")
    return build_config(doc)


def _validate(cfg: ServerCfg) -> None:
    """Fail fast on the config footguns: unknown profile refs, ``extends`` cycles,
    illegal ids. (Duplicate ids can't occur within a single YAML mapping; the
    config-vs-discovery dedupe lives in discovery.)"""
    known = profile_names(cfg)

    # Illegal ids: `@` would break `id@profile` addressing.
    for mid in cfg.models:
        if "@" in mid:
            raise ConfigError(f"model id {mid!r} must not contain '@'")

    # extends cycles + extends targets exist. A chain may legally end at a
    # built-in intent (they never extend further); shadowing is replacement, so
    # a user profile named like a built-in cannot also extend it (self-cycle).
    for name, prof in cfg.profiles.items():
        seen: set[str] = set()
        cur: str | None = name
        while cur is not None:
            if cur not in cfg.profiles:
                if cur in known:
                    break
                raise ConfigError(
                    f"profile {name!r} extends unknown profile {cur!r}; "
                    f"known: {sorted(known)}")
            if cur in seen:
                raise ConfigError(f"profile extends cycle through {cur!r}")
            seen.add(cur)
            cur = cfg.profiles[cur].extends

    # Profile references resolve.
    def _check_ref(where: str, ref: str | None):
        if ref is not None and ref not in known:
            raise ConfigError(
                f"{where} references unknown profile {ref!r}; known: {sorted(known)}")

    _check_ref("server.defaults.profile", cfg.defaults.profile)
    for r in cfg.rules:
        _check_ref(f"rule {r.match!r}", r.profile)
    for mid, m in cfg.models.items():
        _check_ref(f"model {mid!r}", m.profile)
        # Per-model tweaks must target a name that can actually be selected.
        for pname in (m.profiles or {}):
            _check_ref(f"model {mid!r} profiles.{pname!r}", pname)
        # An unknown family is a warning, not an error: configs written by a
        # newer package (more families) must stay loadable by an older one.
        if m.family is not None and m.family not in _family_profiles.FAMILIES:
            import warnings
            warnings.warn(
                f"model {mid!r}: unknown family {m.family!r} "
                f"(known: {sorted(_family_profiles.FAMILIES)}); "
                "using generic defaults", stacklevel=2)

    # Aliases: a name -> `id` | `id@profile`. The name must be addressable (no `@`)
    # and unambiguous (not also a model id); the target's id + profile must exist.
    for name, target in cfg.aliases.items():
        if "@" in name:
            raise ConfigError(f"alias name {name!r} must not contain '@'")
        if name in cfg.models:
            raise ConfigError(
                f"alias {name!r} collides with a model id; rename one")
        tid, tprof = split_address(target, known)
        if tid not in cfg.models:
            raise ConfigError(
                f"alias {name!r} -> unknown model {tid!r}; "
                f"known: {sorted(cfg.models)}")
        if tprof is not None and tprof not in known:
            raise ConfigError(
                f"alias {name!r} -> unknown profile {tprof!r}; "
                f"known: {sorted(known)}")

    # The default model, if named, must exist (the empty-`model` request fallback).
    if cfg.defaults.model and cfg.defaults.model not in cfg.models:
        raise ConfigError(
            f"server.defaults.model {cfg.defaults.model!r} is not a configured "
            f"model; known: {sorted(cfg.models)}")

    # The systemone fallback model, if named, must be a model id or an alias,
    # with an optional known profile, so the fallback can always resolve.
    so_model = cfg.systemone.model
    if so_model:
        head, prof = split_address(so_model, known)
        if head not in cfg.models and head not in cfg.aliases:
            raise ConfigError(
                f"server.systemone.model {so_model!r} is not a configured model "
                f"or alias; known: {sorted(cfg.models) + sorted(cfg.aliases)}")
        if prof is not None and prof not in known:
            raise ConfigError(
                f"server.systemone.model {so_model!r} names unknown profile "
                f"{prof!r}; known: {sorted(known)}")

    # Preload ids must be configured models.
    if isinstance(cfg.defaults.preload, list):
        for mid in cfg.defaults.preload:
            if mid not in cfg.models:
                raise ConfigError(
                    f"server.defaults.preload names unknown model {mid!r}; "
                    f"known: {sorted(cfg.models)}")

    # Assistant aliases: served pseudo-model ids. Each id must be addressable
    # and unambiguous against models and aliases (the request wrapper must be
    # able to claim it), and wrap a real configured model.
    for aid, alias in cfg.assistants.items():
        where = f"server.assistants.{aid}"
        if "@" in aid:
            raise ConfigError(f"assistant id {aid!r} must not contain '@'")
        if aid in cfg.models:
            raise ConfigError(
                f"assistant {aid!r} collides with a model id; rename one")
        if aid in cfg.aliases:
            raise ConfigError(
                f"assistant {aid!r} collides with an alias name; rename one")
        if alias.model not in cfg.models:
            raise ConfigError(
                f"{where}.model: unknown model {alias.model!r}; "
                f"known: {sorted(cfg.models)}")
    if cfg.assistants and cfg.host not in LOOPBACK_HOSTS \
            and not cfg.assistant_allow_remote:
        raise ConfigError(
            f"binding {cfg.host} exposes the assistant tool loop beyond "
            "localhost - remove `server.assistants`, bind a loopback host, or "
            "set `server.assistant_allow_remote: true` (anyone holding the "
            "API key can then drive tool execution on this host)")
    if cfg.assistant_allow_remote and cfg.assistant.mcp:
        unscoped = sorted(a for a, al in cfg.assistants.items()
                          if al.mcp is None)
        if unscoped:
            n = len(cfg.assistant.mcp)
            raise ConfigError(
                f"remote-exposed assistant(s) {', '.join(map(repr, unscoped))} "
                f"would inherit {n} local tool server(s) from `assistant.mcp`; "
                "give each an explicit `mcp:` list (use `mcp: []` for none)")
    # (Typo'd / unsupported keys are surfaced at parse time by _warn_unknown_keys,
    # which still sees the raw dict before .get() drops them - see _parse_*.)
