"""Server configuration model for ``gmlx serve``.

A composable YAML config describes the models the server serves and the reusable
**profiles** (sampling + load + cache params) applied to them. This module is the
pure-Python core - dataclasses, the YAML loader, the ``extends`` / ``rules`` /
precedence merge, and path/env resolution. It imports nothing heavy (no mlx-vlm,
no mlx), so it loads and tests on any machine.

Shape (see ``docs/server-config.md`` for the full reference)::

    server:    {host, port, api_key, no_auth, model_dirs, budget_gb, max_models, hf_cache, cache, defaults, stt, tts, embeddings, rerank, menubar, token_queue_timeout_s, prefill_step_size, cache_limit_gb, family_defaults, stochastic_mtp, gpu_keepwarm, assistants, assistant_allow_remote}
    profiles:  {<name>: {extends, sampling, load, cache, system}}
    rules:     [{match: <glob>, profile: <name>}]
    models:    {<id>: {path, profile, family, profiles, mmproj, draft_gguf, adapter, stream, moe_experts, moe_expert_mass, moe_miss_shed, moe_layer_shed, speculative, overrides, pin, ttl_s}}
    aliases:   {<name>: <id> | <id>@<profile>}    # friendly name / profile preset
    discover:  [{dir, recursive, pair_mmproj, speculative}]
    talk:      {model, voice, speed, system, language, max_tokens, mode, wake_word, wake_threshold, vad, input_device, output_device, chime, brain, push_to_talk_modifier}
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

import fnmatch
import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import profiles as _family_profiles

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
    "max_kv_size": "MAX_KV_SIZE",
    "quantized_kv_start": "QUANTIZED_KV_START",
}

# APC prompt-cache (+ SSD disk tier) key -> env var (mlx-vlm apc.from_env). The disk
# sub-block maps to APC_DISK_*; the namespace defaults to the model path downstream.
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
# Where `disk: true` (the boolean shorthand for the SSD tier) puts the cache.
DEFAULT_APC_DISK_PATH = "~/.cache/gmlx/apc"
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
                       "discover", "talk", "assistant", "theme", "themes"})
_SERVER_KEYS = frozenset({"host", "port", "api_key", "no_auth", "model_dirs",
                          "budget_gb", "max_models", "hf_cache", "cache",
                          "defaults", "stt", "tts", "embeddings", "rerank",
                          "menubar", "token_queue_timeout_s", "prefill_step_size",
                          "cache_limit_gb", "family_defaults", "stochastic_mtp",
                          "gpu_keepwarm", "assistants", "assistant_allow_remote"})
_DEFAULTS_KEYS = frozenset({"profile", "ttl_s", "model", "preload"})
_PROFILE_KEYS = frozenset({"extends", "sampling", "load", "cache", "system",
                           "chat_template", "chat_template_kwargs"})
_OVERRIDE_KEYS = frozenset({"sampling", "load", "cache", "system",
                            "chat_template", "chat_template_kwargs"})
_MODEL_KEYS = frozenset({"path", "profile", "family", "profiles", "mmproj",
                         "draft_gguf", "adapter", "stream",
                         "cpu_moe",  # deprecated alias for `stream:`
                         "moe_experts", "moe_expert_mass",
                         "moe_miss_shed", "moe_layer_shed",
                         "prefill_feeder", "decode_feeder", "speculative",
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

# Host names that count as a loopback bind for the serve auth policy and the
# DNS-rebinding host guard (shared here because server.py must stay importable
# without the fastapi extra).
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Bare-start config search order (first existing wins). Every location uses the
# same ``gmlx.yaml`` basename: project-local ``./gmlx.yaml`` is searched
# first so a repo can override the user-level config; the XDG-style ``~/.config``
# location (where ``gmlx init`` writes and which fits alongside other local AI
# services) is the default; the legacy ``~/.gmlx.yaml`` dotfile is a fallback.
_DEFAULT_CONFIG_PATHS = (
    "./gmlx.yaml",
    "~/.config/gmlx/gmlx.yaml",
    "~/.gmlx.yaml",
)

# Where ``gmlx init`` writes by default - the XDG-style location bare
# ``gmlx serve`` then finds via the search order above (after a project-local
# ``./gmlx.yaml``, if one exists).
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
    adapter: str | None = None          # GGUF LoRA adapter applied live at load
    # Execution placement (normalized by _normalize_stream): "experts" streams
    # only the routed-expert stacks from disk (every-token layers + KV cache
    # stay on GPU); "cpu" runs the whole model on the CPU device, all weights
    # streamed through the page cache. None = all-GPU.
    stream: Any = None
    # Lossy MoE fan-out levers on the streamed expert stacks (the
    # `--moe-experts K` / `--moe-expert-mass P` / `--moe-miss-shed P` /
    # `--moe-layer-shed P` CLI levers). Require stream; None = trained
    # fan-out / no shedding.
    moe_experts: int | None = None
    moe_expert_mass: float | None = None
    moe_miss_shed: float | None = None
    moe_layer_shed: float | None = None
    # Streaming-model feeder overrides (tri-state: None = loader default -
    # prefill feeder on, decode feeder on under `stream: experts`).
    prefill_feeder: bool | None = None
    decode_feeder: bool | None = None
    speculative: bool = False
    overrides: dict = field(default_factory=dict)   # {sampling, load, cache, system}
    pin: bool = False
    ttl_s: float | None = None


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
    max_tokens: int = 512             # spoken replies should be short
    mode: str = "wake"                # wake | vad | ptt | text
    wake_word: str = "hey assistant"  # any text phrase (sherpa-onnx KWS)
    wake_threshold: float = 0.3
    # Menu bar global hotkey: <modifier>+Space (see gmlx.hotkey's
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
    # Optional static API key: every endpoint except /health requires it
    # (Authorization: Bearer, or x-api-key). This config field is the sole
    # server-side source - there is no CLI flag or env override. A non-loopback
    # bind refuses to start with no key unless `no_auth` opts out explicitly
    # (for auth handled in front: mTLS, reverse proxy).
    api_key: str | None = None
    no_auth: bool = False
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
    gpu_keepwarm: bool = False
    defaults: ServerDefaults = field(default_factory=ServerDefaults)
    profiles: dict[str, Profile] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)
    models: dict[str, ModelCfg] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)   # name -> id | id@profile
    discover: list[DiscoverSpec] = field(default_factory=list)
    talk: TalkCfg = field(default_factory=TalkCfg)
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
    # resolved GGUF LoRA adapter abspath, applied live over the base at load; two ids
    # on one GGUF with different adapters are distinct resident entries (load-affecting).
    adapter: str | None = None
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
    # Feeder overrides for streaming models (None = loader default); load-
    # affecting - they decide the ring slots / wired arena built at load.
    prefill_feeder: bool | None = None
    decode_feeder: bool | None = None
    # Sampling family (profiles.py key) the spec resolved under; informational
    # (surfaced by /v1/models and `gmlx profiles`), not load-affecting.
    family: str | None = None

    def load_signature(self) -> tuple:
        """Identity for the residency cache_key: two ids backed by the same GGUF but
        loaded differently (kv bits, mmproj, drafter, speculative, chat template,
        adapter) are distinct resident entries. ``chat_template`` and ``adapter`` are
        both load-affecting - the template is baked into the tokenizer and the adapter
        is wrapped over the model leaves at load, so profiles differing in either need
        their own resident entry. Sampling/system/ttl do not change the loaded model
        and are excluded."""
        return (
            self.path,
            self.mmproj,
            self.draft_gguf,
            bool(self.speculative),
            self.chat_template,
            self.adapter,
            str(self.stream),
            str(self.moe_experts),
            str(self.moe_expert_mass),
            str(self.moe_miss_shed),
            str(self.moe_layer_shed),
            str(self.prefill_feeder),
            str(self.decode_feeder),
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


def default_config_paths() -> list[Path]:
    """Bare-start config search order (first existing wins)."""
    return [Path(os.path.expanduser(p)) for p in _DEFAULT_CONFIG_PATHS]


def default_config_write_path() -> Path:
    """Where ``gmlx init`` writes when ``--out`` is not given."""
    return Path(os.path.expanduser(DEFAULT_CONFIG_WRITE))


def edit_config_yaml(path, mutate) -> None:
    """Round-trip edit of a config file: load with ruamel, call ``mutate(doc)``
    (a CommentedMap), write back. Comments, quoting, and untouched entries keep
    their exact formatting. ruamel is imported lazily."""
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
    with open(path) as f:
        doc = yaml.load(f)
    if doc is None:
        doc = CommentedMap()
    mutate(doc)
    # tmp + rename: `open(path, "w")` would truncate the live config before
    # the dump, so a crash or full disk mid-write destroys it.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        yaml.dump(doc, f)
    os.replace(tmp, path)


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
              "chat_template": None, "chat_template_kwargs": {}}
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

    # Lowest layer: the family's model-card base. Merged directly (not a named
    # pseudo-profile), so it can never be addressed or shadowed - anything a
    # user profile sets wins over it.
    if cfg.family_defaults:
        base = _family_profiles.family_base(model.family)
        sampling = _merge_dict(sampling, base.get("sampling", {}))
        load = _merge_dict(load, base.get("load", {}))
        chat_template_kwargs = _merge_dict(
            chat_template_kwargs, base.get("chat_template_kwargs", {}))

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
        speculative=bool(model.speculative or model.draft_gguf),
        mmproj=resolve_path(model.mmproj, cfg.model_dirs),
        draft_gguf=resolve_path(model.draft_gguf, cfg.model_dirs),
        adapter=resolve_path(model.adapter, cfg.model_dirs),
        stream=model.stream or None,
        moe_experts=model.moe_experts,
        moe_expert_mass=model.moe_expert_mass,
        moe_miss_shed=model.moe_miss_shed,
        moe_layer_shed=model.moe_layer_shed,
        prefill_feeder=model.prefill_feeder,
        decode_feeder=model.decode_feeder,
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
    a ``~`` disk path is expanded."""
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
    bad = sorted(set(raw) - set(known))
    if not bad:
        return
    msg = f"{where}: unrecognized key(s) {bad} (known: {sorted(known)})"
    if strict:
        raise ConfigError(msg)
    import warnings
    warnings.warn(msg, stacklevel=3)


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
    :data:`DEFAULT_APC_DISK_PATH`; ``false`` disables it, overriding any
    inherited ``disk.path`` (the tier is keyed on a present path)."""
    cache = dict(_section_mapping(where, raw))
    _warn_unknown_keys(where, cache, _CACHE_KEYS)
    if "disk" in cache:
        disk = cache["disk"]
        if disk is True:
            disk = {"path": DEFAULT_APC_DISK_PATH}
        elif disk is False:
            disk = {"path": None}
        else:
            disk = dict(_section_mapping(f"{where}.disk", disk))
        _warn_unknown_keys(f"{where}.disk", disk, CACHE_DISK_ENV)
        cache["disk"] = disk
    return cache


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
    ctk = _section_mapping(f"profile {name!r} chat_template_kwargs",
                           raw.get("chat_template_kwargs"))
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
        prefill_feeder=_normalize_optional_bool(
            raw.get("prefill_feeder"), "prefill_feeder", f"model {model_id!r}"),
        decode_feeder=_normalize_optional_bool(
            raw.get("decode_feeder"), "decode_feeder", f"model {model_id!r}"),
        speculative=bool(raw.get("speculative", False)),
        overrides=dict(ov),
        pin=bool(raw.get("pin", False)),
        ttl_s=_coerce_num("ttl_s", raw.get("ttl_s"), float,
                          where=f"model {model_id!r}"),
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
    from .hotkey import PUSH_TO_TALK_MODIFIERS
    ptt = str(raw.get("push_to_talk_modifier", "globe")).strip().lower()
    if ptt not in PUSH_TO_TALK_MODIFIERS:
        raise ConfigError(
            f"talk.push_to_talk_modifier: {ptt!r} is not one of "
            f"{'/'.join(PUSH_TO_TALK_MODIFIERS)}")
    dev = lambda v: None if v is None else str(v)  # noqa: E731
    return TalkCfg(
        model=str(raw["model"]) if raw.get("model") else None,
        voice=str(raw["voice"]) if raw.get("voice") else None,
        speed=num("speed", raw.get("speed"), float, 1.0),
        # Absent -> speakable-output default; an explicit empty string is
        # the opt-out for "no system prompt at all".
        system=(str(raw["system"]) if raw.get("system")
                else (None if "system" in raw else DEFAULT_TALK_SYSTEM)),
        language=str(raw["language"]) if raw.get("language") else None,
        max_tokens=num("max_tokens", raw.get("max_tokens"), int, 512),
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


def build_config(doc: dict) -> ServerCfg:
    """Build (and validate) a :class:`ServerCfg` from a parsed YAML mapping. Split out
    from :func:`load_config` so discovery / tests can build a config in memory."""
    doc = doc or {}
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
        api_key=str(srv["api_key"]) if srv.get("api_key") else None,
        no_auth=bool(srv.get("no_auth", False)),
        menubar=bool(srv.get("menubar", True)),
        token_queue_timeout_s=_coerce_num(
            "token_queue_timeout_s", srv.get("token_queue_timeout_s"), float),
        prefill_step_size=_coerce_num(
            "prefill_step_size", srv.get("prefill_step_size"), int),
        cache_limit_gb=_coerce_num(
            "cache_limit_gb", srv.get("cache_limit_gb"), float),
        family_defaults=bool(srv.get("family_defaults", True)),
        stochastic_mtp=bool(srv.get("stochastic_mtp", False)),
        gpu_keepwarm=bool(srv.get("gpu_keepwarm", False)),
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
