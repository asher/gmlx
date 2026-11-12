"""Text embeddings for the server: the engine behind ``POST /v1/embeddings``.

Three backends, chosen by what ``embeddings:`` points at:

* GGUF decoder LM (a ``*.gguf`` file or ``hf:<org>/<repo>/<file>.gguf`` ref),
  such as ``Qwen3-Embedding-{0.6,4,8}B``. These are a decoder trunk plus
  pooling and an L2 norm, so the runtime's own
  :func:`gmlx.load.loader.load_model` loads them, and one prefill forward over
  ``model.model`` (the trunk) gives the hidden states to pool. They pool as
  ``<arch>.pooling_type`` sets: 1 the mean, 2 the first token, 3 the last
  token. Without it they pool the last token after an EOS, as Qwen3-Embedding
  does. One that sets ``<arch>.attention.causal`` to false runs without the
  causal mask. No separate download or encoder framework is needed. This
  path is not capped at the BERT window, so long documents (30k-token
  transcripts) embed without silent truncation. The GGUF must set
  ``<arch>.pooling_type`` or carry an embed or embedding token in its file
  name, as GGUF discovery reads them; a causal LM does neither.

* GGUF encoder (a ``*.gguf`` or ``hf:`` ref to an encoder arch), such as
  EmbeddingGemma. The same :func:`gmlx.load.loader.load_model` builds a gmlx
  encoder (:mod:`gmlx.models.embedders`: a bidirectional backbone, mean
  pooling, a dense head and an L2 norm) that returns pooled ``text_embeds``
  itself, so each text is just forwarded.

* safetensors encoder (an alias, an mlx-community repo id or a local dir):
  EmbeddingGemma, EmbeddingGemma 2, BERT, XLM-RoBERTa, ModernBERT and
  Qwen3-Embedding checkpoints, loaded by :func:`gmlx.models.embedders.load`. It returns
  pooled, L2-normalized vectors in ``outputs.text_embeds``, pooled as the
  checkpoint's sentence-transformers config sets (see
  :mod:`gmlx.models.embedders.checkpoint`).

Either way the loaded model is cached process-wide, so repeat requests don't
reload. This module is import-safe without MLX loaded: the encoders import
lazily, on the first load.

The point of this endpoint is to give Open WebUI (and other OpenAI clients) a
local RAG embedder, so no sentence-transformers model is downloaded from
HuggingFace.
"""

from __future__ import annotations

import base64
import concurrent.futures
import os
import sys
from dataclasses import dataclass

import numpy as np

from . import subservice
from .embeddings_errors import EmbeddingsModelError, EmbeddingsRequestError
from .hf_cache import offline_resolve
from .subservice import GGUFModelHolder, SingleWorker, preset_for, service_display

# Embedding model presets surfaced by the ``gmlx init`` wizard, in display order.
# Two tiers:
#
#  * tier ``gguf``: GGUFs that run on this runtime's own loader, so they need
#    no extra. The Qwen3-Embedding decoder LMs (last-token pooling and an L2
#    norm) point into the official Qwen GGUF repos, and ``quants`` lists only
#    the rungs those repos ship (0.6B has just Q8_0 and f16).
#  * tier ``mlx``: safetensors encoders (EmbeddingGemma, XLM-RoBERTa,
#    ModernBERT) on gmlx's own encoders. Values are mlx-community repo ids,
#    and ``quants`` are the bit-width variants.
#
# ``dim`` is the vector width (bigger = larger index + more storage/RAM per
# vector), ``ctx`` the max input tokens, ``default_quant`` the rung the wizard
# pre-selects and the bare alias resolves to, ``sizes`` the on-disk GB per rung
# (for the wizard guidance table + quant follow-up). EMBEDDINGS_ALIASES is derived
# from these so the presets are the single source of truth.

def _qwen_emb_ref(size: str, quant: str) -> str:
    """An ``hf:`` ref into the official Qwen3-Embedding GGUF repo for a rung."""
    return (f"hf:Qwen/Qwen3-Embedding-{size}-GGUF/"
            f"Qwen3-Embedding-{size}-{quant}.gguf")


_EG2_REPO = "hf:unsloth/embeddinggemma-2-GGUF"

EMBEDDING_PRESETS = [
    {"alias": "embeddinggemma-2-gguf", "label": "EmbeddingGemma 2 (GGUF)",
     "tier": "gguf", "dim": 768, "ctx": 8192,
     "blurb": "text, images, audio and video in one space (Google)",
     "default_quant": "Q8_0",
     "quants": {"Q8_0": f"{_EG2_REPO}/embeddinggemma-2-Q8_0.gguf"},
     # The projector by chip (see preset_mmproj_ref); both are 16-bit.
     "mmproj": {"bf16": f"{_EG2_REPO}/mmproj-BF16.gguf",
                "f16": f"{_EG2_REPO}/mmproj-F16.gguf"},
     # The text file and this Mac's projector.
     "sizes": {"Q8_0": 1.29}},
    {"alias": "qwen3-embed-0.6b", "label": "Qwen3-Embedding 0.6B",
     "tier": "gguf", "dim": 1024, "ctx": 32768,
     "blurb": "small, fast, multilingual, strong retrieval per byte",
     "default_quant": "Q8_0",
     "quants": {"Q8_0": _qwen_emb_ref("0.6B", "Q8_0"),
                "f16": _qwen_emb_ref("0.6B", "f16")},
     "sizes": {"Q8_0": 0.64, "f16": 1.20}},
    {"alias": "qwen3-embed-4b", "label": "Qwen3-Embedding 4B",
     "tier": "gguf", "dim": 2560, "ctx": 40960,
     "blurb": "higher retrieval quality; 2560-dim vectors (larger index)",
     "default_quant": "Q8_0",
     "quants": {"Q4_K_M": _qwen_emb_ref("4B", "Q4_K_M"),
                "Q5_K_M": _qwen_emb_ref("4B", "Q5_K_M"),
                "Q6_K": _qwen_emb_ref("4B", "Q6_K"),
                "Q8_0": _qwen_emb_ref("4B", "Q8_0")},
     "sizes": {"Q4_K_M": 2.50, "Q5_K_M": 2.89, "Q6_K": 3.31, "Q8_0": 4.28}},
    {"alias": "qwen3-embed-8b", "label": "Qwen3-Embedding 8B",
     "tier": "gguf", "dim": 4096, "ctx": 40960,
     "blurb": "best of the Qwen family; 4096-dim (largest index, most RAM)",
     "default_quant": "Q8_0",
     "quants": {"Q4_K_M": _qwen_emb_ref("8B", "Q4_K_M"),
                "Q5_K_M": _qwen_emb_ref("8B", "Q5_K_M"),
                "Q6_K": _qwen_emb_ref("8B", "Q6_K"),
                "Q8_0": _qwen_emb_ref("8B", "Q8_0")},
     "sizes": {"Q4_K_M": 4.68, "Q5_K_M": 5.42, "Q6_K": 6.21, "Q8_0": 8.05}},
    {"alias": "embeddinggemma-gguf", "label": "EmbeddingGemma 300M (GGUF)",
     "tier": "gguf", "dim": 768, "ctx": 2048,
     "blurb": "tiny multilingual retrieval (Google); K-quant GGUF on this runtime",
     "default_quant": "Q8_0",
     "quants": {"Q8_0": ("hf:ggml-org/embeddinggemma-300M-GGUF/"
                         "embeddinggemma-300M-Q8_0.gguf")},
     "sizes": {"Q8_0": 0.33}},
    {"alias": "embeddinggemma-2", "label": "EmbeddingGemma 2",
     "tier": "mlx", "dim": 768, "ctx": 8192,
     "blurb": "text, images, audio and video in one space (Google)",
     "default_quant": "8bit",
     "quants": {"8bit": "mlx-community/embeddinggemma-2-8bit",
                "bf16": "mlx-community/embeddinggemma-2-bf16"},
     "sizes": {"8bit": 1.23, "bf16": 1.49}},
    {"alias": "embeddinggemma", "label": "EmbeddingGemma 300M",
     "tier": "mlx", "dim": 768, "ctx": 2048,
     "blurb": "tiny, strong multilingual retrieval (Google)",
     "default_quant": "8bit",
     "quants": {"4bit": "mlx-community/embeddinggemma-300m-4bit",
                "6bit": "mlx-community/embeddinggemma-300m-6bit",
                "8bit": "mlx-community/embeddinggemma-300m-8bit",
                "bf16": "mlx-community/embeddinggemma-300m-bf16"},
     "sizes": {"4bit": 0.17, "6bit": 0.25, "8bit": 0.33, "bf16": 0.62}},
    {"alias": "arctic-l", "label": "Snowflake Arctic-Embed-L v2.0",
     "tier": "mlx", "dim": 1024, "ctx": 8192,
     "blurb": "multilingual long-context retrieval (XLM-RoBERTa)",
     "default_quant": "8bit",
     "quants": {"4bit": "mlx-community/snowflake-arctic-embed-l-v2.0-4bit",
                "6bit": "mlx-community/snowflake-arctic-embed-l-v2.0-6bit",
                "8bit": "mlx-community/snowflake-arctic-embed-l-v2.0-8bit",
                "bf16": "mlx-community/snowflake-arctic-embed-l-v2.0-bf16"},
     "sizes": {"4bit": 0.32, "6bit": 0.46, "8bit": 0.60, "bf16": 1.14}},
    {"alias": "nomic-embed", "label": "Nomic Embed (ModernBERT base)",
     "tier": "mlx", "dim": 768, "ctx": 8192,
     "blurb": "popular long-context English embedder (ModernBERT)",
     "default_quant": "8bit",
     "quants": {"4bit": "mlx-community/nomicai-modernbert-embed-base-4bit",
                "6bit": "mlx-community/nomicai-modernbert-embed-base-6bit",
                "8bit": "mlx-community/nomicai-modernbert-embed-base-8bit",
                "bf16": "mlx-community/nomicai-modernbert-embed-base-bf16"},
     "sizes": {"4bit": 0.08, "6bit": 0.12, "8bit": 0.16, "bf16": 0.30}},
    {"alias": "bge-m3", "label": "BGE-M3",
     "tier": "mlx", "dim": 1024, "ctx": 8192,
     "blurb": "multilingual long-context retrieval (XLM-RoBERTa)",
     "default_quant": "8bit",
     "quants": {"4bit": "mlx-community/bge-m3-mlx-4bit",
                "6bit": "mlx-community/bge-m3-mlx-6bit",
                "8bit": "mlx-community/bge-m3-mlx-8bit",
                "fp16": "mlx-community/bge-m3-mlx-fp16"},
     "sizes": {"4bit": 0.32, "6bit": 0.46, "8bit": 0.60, "fp16": 1.14}},
]

# The model of ``embeddings: true``, ``default`` and a bare ``--embeddings``.
DEFAULT_EMBEDDINGS_ALIAS = "qwen3-embed-0.6b"
# The model that ``gmlx init``, its wizard and ``--with-embeddings`` set up.
INIT_EMBEDDINGS_ALIAS = "embeddinggemma-2-gguf"

# alias -> default-rung concrete value (gguf ``hf:`` ref or mlx-community repo id).
EMBEDDINGS_ALIASES = {
    p["alias"]: p["quants"][p["default_quant"]] for p in EMBEDDING_PRESETS
}
# Hidden back-compat: the bare ``qwen3-embed`` historically meant the 0.6B. Keep
# it resolvable (not shown in the wizard, which lists explicitly-sized ids).
EMBEDDINGS_ALIASES["qwen3-embed"] = EMBEDDINGS_ALIASES["qwen3-embed-0.6b"]


def embedding_preset(alias):
    """The preset dict for a base alias (e.g. ``qwen3-embed-4b``), or ``None``."""
    return preset_for(alias, EMBEDDING_PRESETS)


def preset_mmproj_ref(preset: dict) -> str | None:
    """The ``hf:`` ref of a preset's projector for this Mac: BF16 where the
    chip runs bfloat16 natively (M3 and later), F16 on M1 and M2. The chip
    decides, never ``GMLX_ACTIVATION_DTYPE``."""
    mmproj = preset.get("mmproj")
    if not mmproj:
        return None
    from gmlx.load.dtypes import has_native_bfloat16

    return mmproj["bf16" if has_native_bfloat16() else "f16"]


def pull_hint(value, projector: bool = True) -> str | None:
    """The ``gmlx pull`` command for a GGUF preset alias or one of its refs
    that has a projector, or None. Without ``projector``, the text file
    alone."""
    v = str(value).strip()
    for p in EMBEDDING_PRESETS:
        if "mmproj" not in p:
            continue
        rungs = [q for q, ref in p["quants"].items() if ref == v]
        if v.lower() == p["alias"]:
            rungs = [p["default_quant"]]
        if rungs:
            args = preset_pull_args(p, rungs[0])
            return "gmlx pull " + " ".join(args if projector else args[:1])
    return None


def preset_pull_args(preset: dict, quant: str | None = None) -> list[str]:
    """The ``gmlx pull`` arguments that fetch a GGUF preset's rung and its
    projector for this Mac: the ref, then the projector's file name in the
    same repo."""
    ref = preset["quants"][quant or preset["default_quant"]]
    mm = preset_mmproj_ref(preset)
    if mm is None:
        return [ref]
    return [ref, mm.rsplit("/", 1)[1]]


# Request `model` values that mean "whatever this server has configured".
# OpenAI clients conventionally send "text-embedding-3-small" / "-3-large" /
# "-ada-002"; Open WebUI's openai RAG engine sends `RAG_EMBEDDING_MODEL`.
_CONFIGURED_NAMES = frozenset({
    "", "text-embedding-3-small", "text-embedding-3-large",
    "text-embedding-ada-002", "default"})

# OpenAI `encoding_format` values we serve. `float` is a JSON array of numbers;
# `base64` is base64 of the little-endian float32 bytes (what the openai-python
# client requests by default).
ENCODING_FORMATS = ("float", "base64")

# Per-input token ceiling for the safetensors encoder backend. The old
# 512-ctx BERTs are gone; the remaining encoders carry 2k-8k windows, so cap by
# each model's own ``model_max_length`` (guarding the ~1e30 "no limit" sentinel)
# up to this ceiling. A flat 512 would silently truncate long documents on the
# long-context encoders (arctic / nomic / bge-m3) that are the reason to pick them.
_ENCODER_MAX_TOKENS = 8192

# The GGUF decoder-LM embedders (Qwen3-Embedding) carry an 8k-32k context, so the
# 512 BERT cap would silently throw away most of a long document. Cap generously
# instead: long transcripts are exactly why someone reaches for these.
_GGUF_MAX_TOKENS = 32768

# All model work (the load and every forward) routes through one persistent
# single-worker thread. subservice.SingleWorker says why (per-thread MLX
# streams; an uncaught cross-thread abort otherwise).
_EMBED_WORKER = SingleWorker("embeddings-worker")


# Model path -> the reason it cannot load. A file that could not be read is
# not cached here, since a later try may read it.
_UNLOADABLE: dict[str, str] = {}


class _EmbeddingsModelHolder:
    """Process-wide single-model cache for the safetensors encoders. Mirrors
    mlx-whisper's ModelHolder: reload only on a path change. A model that
    cannot load raises :class:`EmbeddingsModelError`, from the cache after
    the first try unless a file could not be read."""

    model = None
    tokenizer = None
    model_path = None

    @classmethod
    def get(cls, model_path: str):
        if model_path in _UNLOADABLE:
            raise EmbeddingsModelError(_UNLOADABLE[model_path])
        if cls.model is None or model_path != cls.model_path:
            from gmlx.models.embedders import load
            from gmlx.models.embedders.checkpoint import (
                UnsupportedEncoderError,
                card_files_to_fetch,
            )
            try:
                with offline_resolve(model_path, wanted=card_files_to_fetch):
                    cls.model, cls.tokenizer = load(model_path)
            except UnsupportedEncoderError as exc:
                if not exc.transient:
                    _UNLOADABLE[model_path] = exc.reason
                raise EmbeddingsModelError(exc.reason) from None
            cls.model_path = model_path
        return cls.model, cls.tokenizer


def _local_checkpoint(model_path: str) -> str | None:
    """The local folder of a safetensors model ref, or None when it is not
    on disk yet (a Hub repo id not in the local cache)."""
    expanded = os.path.expanduser(model_path)
    if os.path.isdir(expanded):
        return expanded
    try:
        from .hf_cache import cached_snapshot

        return cached_snapshot(model_path, local_files_only=True)
    except Exception:                     # noqa: BLE001 (not cached, or no id)
        return None


_GGUF_SUPPORTED = ("Supported GGUFs: Qwen3-Embedding, as the qwen3-embed-0.6b, "
                   "qwen3-embed-4b and qwen3-embed-8b aliases, EmbeddingGemma, "
                   "as the embeddinggemma-gguf alias, and EmbeddingGemma 2, "
                   "as the embeddinggemma-2-gguf alias.")


# How a decoder GGUF pools, by llama.cpp's <arch>.pooling_type: 1 the mean,
# 2 the first token, 3 the last token. A GGUF that is an embedder by its file
# name and sets no pooling type, or sets none (0) or unspecified (-1), pools
# its last token after an EOS, as the Qwen3-Embedding GGUFs that set none do.
_GGUF_POOLING = {None: "last", -1: "last", 0: "last", 1: "mean", 2: "cls",
                 3: "last"}

# The decoder architectures whose trunk runs without the causal mask, for a
# GGUF that sets <arch>.attention.causal to false: the trunk asks each
# layer's cache for its mask, and _BidirectionalCache gives none. The
# others give the 501.
_BIDIRECTIONAL_ARCHES = frozenset({"llama", "mistral3", "qwen2", "qwen3"})


def _gguf_header(path: str) -> dict:
    """The KV block of the GGUF at ``path``. Reads no tensors."""
    from gmlx.load.headerscan import scan_gguf

    return scan_gguf(path, include_tensors=False).kv


def _gguf_pooling(kv: dict) -> str | None:
    """How a decoder GGUF with the header ``kv`` pools: "mean", "cls" or
    "last", or None for a pooling type that gives no embedding."""
    from gmlx.load.gguf_meta import read_int

    arch = kv.get("general.architecture")
    return _GGUF_POOLING.get(read_int(kv, f"{arch}.pooling_type"))


def _gguf_causal(kv: dict) -> bool:
    """Whether a decoder GGUF with the header ``kv`` attends causally: all
    do but one that sets ``<arch>.attention.causal`` to false."""
    from gmlx.load.gguf_meta import read_bool

    arch = kv.get("general.architecture")
    return read_bool(kv, f"{arch}.attention.causal") is not False


def _gguf_refusal(path: str, kv: dict | None = None) -> str | None:
    """Why the GGUF at ``path`` is not an embeddings model that gmlx runs,
    from its header ``kv`` (read from ``path`` when None) and file name, or
    None. Reads no tensors.

    A GGUF is an embeddings model by the signals that GGUF discovery uses
    (gmlx.load.discovery.embedding_kind): an encoder architecture, a pooling
    type of 1 or more, or an embed or embedding token in the file name, as the
    Qwen3-Embedding GGUFs that set no pooling type carry. Of the encoder
    architectures, gmlx builds only those that its loader supports, and
    those pool by the mean, so one must set no other pooling type. A decoder
    must pool by the mean, the first token or the last token, and one that
    attends in both directions must be of an architecture in
    :data:`_BIDIRECTIONAL_ARCHES`, with no sliding window."""
    from gmlx.load.config_synth import supported_arches
    from gmlx.load.discovery import ENCODER_ARCHES, embedding_kind
    from gmlx.load.gguf_meta import read_int

    if kv is None:
        kv = _gguf_header(path)
    arch = kv.get("general.architecture")
    if not arch:
        return ("this GGUF sets no general.architecture, so it is not an "
                f"embeddings model. {_GGUF_SUPPORTED}")
    arch = str(arch)
    from gmlx.load.decision import DecisionModelOnly, decision_header

    if decision_header(kv):
        return f"{DecisionModelOnly('this GGUF')}. {_GGUF_SUPPORTED}"
    kind = embedding_kind(kv, os.path.basename(path), arch)
    if kind == "reranker":
        return (f"this {arch} GGUF is a reranker, not an embeddings model. "
                f"{_GGUF_SUPPORTED}")
    if kind is None:
        pooling = kv.get(f"{arch}.pooling_type")
        sets = (f"sets no {arch}.pooling_type" if pooling is None
                else f"sets {arch}.pooling_type to {pooling} (no pooling)")
        return (f"this {arch} GGUF {sets} and its file name names no "
                "embedding model, so it is a causal LM, not an embeddings "
                f"model. {_GGUF_SUPPORTED}")
    if arch not in supported_arches():
        what = "encoder" if arch in ENCODER_ARCHES else "model"
        return (f"gmlx cannot run a {arch} {what} GGUF. {_GGUF_SUPPORTED}")
    pooling = read_int(kv, f"{arch}.pooling_type")
    if pooling == 4:
        return (f"this {arch} GGUF sets {arch}.pooling_type to 4 (rank), so "
                f"it is a reranker, not an embeddings model. {_GGUF_SUPPORTED}")
    if arch in ENCODER_ARCHES:
        if pooling not in (None, -1, 1):
            return (f"this {arch} GGUF sets {arch}.pooling_type to {pooling}, "
                    f"and gmlx runs a {arch} encoder only with mean pooling "
                    f"(1). {_GGUF_SUPPORTED}")
        return None
    if _gguf_pooling(kv) is None:
        return (f"this {arch} GGUF sets {arch}.pooling_type to {pooling}, "
                f"which is no pooling type that gmlx knows. {_GGUF_SUPPORTED}")
    if not _gguf_causal(kv):
        sets = f"this {arch} GGUF sets {arch}.attention.causal to false"
        if arch not in _BIDIRECTIONAL_ARCHES:
            return (f"{sets}, and gmlx runs a {arch} model only with the "
                    f"causal mask. {_GGUF_SUPPORTED}")
        if (read_int(kv, f"{arch}.attention.sliding_window") or 0) > 0:
            return (f"{sets} and sets a sliding window, and gmlx runs "
                    "sliding-window attention only with the causal mask. "
                    f"{_GGUF_SUPPORTED}")
    return None


def _check(model_path: str) -> tuple[str | None, bool]:
    """:func:`check_embeddings_model` before the cache: the reason and
    whether a later try may pass. May raise."""
    if _is_gguf_ref(model_path):
        if not os.path.isfile(model_path):
            return None, False
        return _gguf_refusal(model_path), False
    folder = _local_checkpoint(model_path)
    if folder is None:
        return None, False
    from gmlx.models.embedders.checkpoint import (
        UnsupportedEncoderError,
        card_files_to_fetch,
        check,
    )
    try:
        check(folder)
    except UnsupportedEncoderError as exc:
        if (not os.path.isdir(os.path.expanduser(model_path))
                and card_files_to_fetch(folder)):
            return None, False            # the load fetches what settles it
        return exc.reason, exc.transient
    return None, False


def check_embeddings_model(model_path: str) -> str | None:
    """Why the configured model cannot load, from its config.json or its
    GGUF header, or None.

    Reads no weights. A repo id not in the local cache yet gives None: the
    load checks it. A reason found is cached, so a request returns it
    without a load, unless a file could not be read, which a later load may
    read. The check never raises: a check that fails is logged, and its
    failure is the reason, cached unless it is an OSError."""
    if model_path in _UNLOADABLE:
        return _UNLOADABLE[model_path]
    try:
        reason, transient = _check(model_path)
    except FileNotFoundError:
        return None                       # the load reports the missing file
    except Exception as exc:              # noqa: BLE001 (keeps the server up)
        print(f"[server] embeddings: the startup check of {model_path} "
              f"failed: {exc!r}", file=sys.stderr)
        reason = f"the startup check failed ({type(exc).__name__})"
        transient = isinstance(exc, OSError)
    if reason is not None and not transient:
        _UNLOADABLE[model_path] = reason
    return reason


class _GGUFEmbeddingsHolder(GGUFModelHolder):
    """The GGUF backends' model cache (subservice.GGUFModelHolder). A GGUF
    that is not an embeddings model raises :class:`EmbeddingsModelError`,
    from the cache after the first try. ``pooling`` is how the loaded
    decoder pools, and ``causal`` whether it attends causally, both from its
    header (:func:`_gguf_pooling`, :func:`_gguf_causal`)."""

    pooling = "last"
    causal = True

    @classmethod
    def get(cls, model_path: str):
        if model_path in _UNLOADABLE:
            raise EmbeddingsModelError(_UNLOADABLE[model_path])
        if cls.model is None or model_path != cls.model_path:
            kv = _gguf_header(model_path)
            reason = _gguf_refusal(model_path, kv)
            if reason is not None:
                _UNLOADABLE[model_path] = reason
                raise EmbeddingsModelError(reason)
            loaded = super().get(model_path)
            cls.pooling = _gguf_pooling(kv) or "last"
            cls.causal = _gguf_causal(kv)
            return loaded
        return super().get(model_path)


def _load_embeddings_model(model_path: str) -> None:
    """Populate the model cache for ``model_path``, dispatching to the GGUF or
    safetensors holder by ref shape. Runs on :data:`_EMBED_WORKER` (like every
    request), so a request that races a background warm waits on this load
    instead of kicking off a second one."""
    if _is_gguf_ref(model_path):
        _GGUFEmbeddingsHolder.get(model_path)
    else:
        _EmbeddingsModelHolder.get(model_path)


def prewarm(model_path: str) -> concurrent.futures.Future:
    """Check the configured embeddings model's config.json, then
    background-load it at server startup (best-effort; see
    :func:`subservice.prewarm`). A model that cannot load is not loaded:
    the server logs why, and /v1/embeddings returns the reason."""
    reason = check_embeddings_model(model_path)
    if reason is not None:
        print(f"[server] embeddings: {EmbeddingsModelError(reason)}",
              file=sys.stderr)
        done: concurrent.futures.Future = concurrent.futures.Future()
        done.set_result(None)
        return done

    def _load():
        _load_embeddings_model(model_path)

    return subservice.prewarm(
        _EMBED_WORKER, _load, "embeddings",
        missing_hint="/v1/embeddings returns 404 until the file is restored, "
                     "the config is updated or `gmlx sync-models` runs")


def release() -> concurrent.futures.Future:
    """Drop the cached embeddings models of both backends and the media
    towers (see :func:`subservice.release`)."""
    _FAMILIES.clear()
    return subservice.release(_EMBED_WORKER, _EmbeddingsModelHolder,
                              _GGUFEmbeddingsHolder, _TowersHolder)


def _is_gguf_ref(value) -> bool:
    """True for an embeddings value that names a GGUF file, a local ``*.gguf``
    path or an ``hf:<org>/<repo>/<file>.gguf`` ref, which the GGUF backends
    load. Everything else (alias, mlx-community repo id, local dir) is the
    safetensors encoder backend."""
    if not isinstance(value, str):
        return False
    low = value.strip().lower()
    if low.startswith("hf:"):
        low = low.split("@", 1)[0]          # drop any @revision
    return low.endswith(".gguf")


def resolve_embeddings_model(value, model_dirs=()) -> str:
    """Normalize a configured embeddings value to a backend-ready model id/path.

    Accepts a friendly alias (``qwen3-embed-0.6b``, ``embeddinggemma``), an HF
    repo id (``mlx-community/bge-m3-mlx-8bit``), a local model directory, a GGUF
    decoder-LM ref (a ``*.gguf`` path or ``hf:<org>/<repo>/<file>.gguf``), or
    ``True``/``"default"`` (YAML ``embeddings: true`` / bare ``--embeddings``)
    for the default alias. A GGUF ref, passed directly or reached via a
    gguf-tier alias, is resolved to a concrete local file (hf: refs resolve
    from the local HF cache, never the network; a miss raises a ``gmlx pull``
    hint), so the loader can open it directly. A relative ``*.gguf`` path is
    searched under ``model_dirs``, same as a ``models:`` entry (serve passes
    ``server.model_dirs``).
    """
    if value is True:
        value = DEFAULT_EMBEDDINGS_ALIAS
    v = str(value).strip()
    low = v.lower()
    if low in ("default", "true"):
        v = EMBEDDINGS_ALIASES[DEFAULT_EMBEDDINGS_ALIAS]
    elif low in EMBEDDINGS_ALIASES:
        v = EMBEDDINGS_ALIASES[low]         # alias -> default-rung concrete value
    # ``v`` is now concrete: a gguf ref (``hf:`` / ``*.gguf``), a repo id, or a
    # dir. A gguf ref (whether passed directly or reached via a gguf-tier alias)
    # resolves to its local file; a repo id / dir is the safetensors backend.
    if _is_gguf_ref(v):
        from gmlx.config import MissingModelFile, resolve_path
        try:
            return resolve_path(v, list(model_dirs))  # hf: -> cache; path -> abs
        except MissingModelFile:
            hint = pull_hint(v)
            if hint is None:
                raise
            raise MissingModelFile(
                f"hf model {v!r} is not in the local Hugging Face cache or under "
                f"model_dirs; `{hint}` fetches it and its projector for this "
                "Mac, or remove the entry.") from None
    expanded = os.path.expanduser(v)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    return v  # HF repo id (or a path that will fail loudly at load)


def effective_model(requested: str, configured: str) -> str:
    """Map a request's ``model`` field onto the configured embeddings model
    (conventional names pass; anything else is a 400)."""
    return subservice.effective_model(
        requested, configured, accepted_names=_CONFIGURED_NAMES,
        resolver=resolve_embeddings_model, error_cls=EmbeddingsRequestError,
        kind="embeddings", hint="text-embedding-3-small")


def _embed_texts(model_obj, tokenizer, texts: list,
                 add_special_tokens: bool = True):
    """Run a safetensors encoder over ``texts`` and return ``(matrix,
    n_tokens)``: a float32 ``[len(texts), dim]`` numpy array of pooled,
    L2-normalized vectors (the checkpoint's own pooling), plus the total
    non-padding token count (for the usage field)."""
    import mlx.core as mx

    cap = getattr(tokenizer, "model_max_length", None)
    if not isinstance(cap, int) or cap <= 0 or cap > _ENCODER_MAX_TOKENS:
        cap = _ENCODER_MAX_TOKENS           # sentinel / unknown -> the ceiling
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=cap,
                       add_special_tokens=add_special_tokens,
                       return_tensors="np")
    mask = np.asarray(inputs["attention_mask"])
    # An all-real batch (one text, or equal lengths) needs no attention mask.
    outputs = model_obj(mx.array(np.asarray(inputs["input_ids"])),
                        attention_mask=None if mask.all() else mx.array(mask))
    embeds = outputs.text_embeds.astype(mx.float32)
    mx.eval(embeds)
    matrix = np.asarray(embeds)
    if matrix.ndim == 1:                         # a model that returns [dim]
        matrix = matrix[None, :]
    return matrix, int(mask.sum())


def _eos_id(tokenizer):
    """The tokenizer's EOS id (the pooling token for Qwen3-Embedding), or None.
    Accepts the int or the list-of-ids shape some backends expose."""
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, bool):                  # guard: bool is an int subclass
        return None
    if isinstance(eos, int):
        return eos
    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])
    return None


def _encode_for_embed(tokenizer, text: str, add_special_tokens: bool = True) -> list:
    """Token ids for one input. Tolerates backends whose ``encode`` rejects the
    ``add_special_tokens`` kwarg (the GGUF-synthesized fast tokenizer accepts it)."""
    if not add_special_tokens:
        return list(tokenizer.encode(text, add_special_tokens=False))
    try:
        ids = tokenizer.encode(text)
    except TypeError:
        ids = tokenizer.encode(text, add_special_tokens=True)
    return list(ids)


class _BidirectionalCache:
    """Stands in for one layer's KV cache in a trunk forward without the
    causal mask. The trunk asks it for the attention mask and gets none, so
    each token attends to every token. The attention reads its offset, 0,
    and passes the keys and values through it, which it does not keep."""

    offset = 0

    def make_mask(self, n, return_array=False, window_size=None):
        return None

    def update_and_fetch(self, keys, values):
        return keys, values


def _embed_texts_gguf(model_obj, tokenizer, texts: list, pooling: str = "last",
                      causal: bool = True, add_special_tokens: bool = True):
    """Embed ``texts`` with the GGUF decoder-LM backend and return
    ``(matrix, n_tokens)``: a float32 ``[len(texts), dim]`` of pooled,
    L2-normalized hidden states, plus the total token count.

    ``pooling`` is the GGUF's (:func:`_gguf_pooling`): "mean" over the
    tokens, "cls" the first token, or "last" the last token, after an EOS
    appended to a text that does not end in one, as Qwen3-Embedding pools.
    With ``causal`` false (:func:`_gguf_causal`) every token attends to
    every token, through a :class:`_BidirectionalCache` per layer.

    One prefill forward over the trunk per text (batch=1). A batch would need
    left or right padding and the last-non-pad bookkeeping, and the embed lock
    serializes the GPU work anyway; RAG indexing is bound by throughput, not
    latency. Batched (mask-aware) forwards are a later optimization."""
    import mlx.core as mx

    trunk = getattr(model_obj, "model", model_obj)   # mlx-lm Model.model = decoder
    eos = _eos_id(tokenizer)
    vecs: list = []
    n_tokens = 0
    for text in texts:
        ids = _encode_for_embed(tokenizer, text, add_special_tokens)
        if len(ids) > _GGUF_MAX_TOKENS:
            ids = ids[:_GGUF_MAX_TOKENS]
        if (pooling == "last" and eos is not None
                and (not ids or ids[-1] != eos)):
            ids.append(eos)                          # Qwen3-Embedding pools EOS
        if not ids:
            raise EmbeddingsRequestError(400, "input encodes to zero tokens")
        n_tokens += len(ids)
        if causal:
            hidden = trunk(mx.array([ids]))          # [1, T, H]
        else:
            hidden = trunk(mx.array([ids]), cache=[
                _BidirectionalCache() for _ in trunk.layers])
        # float32: numpy can't buffer bf16, and the mean sums in float32
        if pooling == "mean":
            pooled = mx.mean(hidden[0].astype(mx.float32), axis=0)
        elif pooling == "cls":
            pooled = hidden[0, 0, :].astype(mx.float32)
        else:
            pooled = hidden[0, -1, :].astype(mx.float32)
        norm = mx.sqrt(mx.sum(pooled * pooled)) + 1e-12
        pooled = pooled / norm
        mx.eval(pooled)
        vecs.append(np.asarray(pooled))
    return np.stack(vecs, axis=0), n_tokens


def _is_gguf_encoder(model_obj) -> bool:
    """True if a GGUF loaded into a gmlx encoder (EmbeddingGemma: pooled
    ``text_embeds`` interface), vs an mlx-lm decoder trunk we pool by hand."""
    return type(model_obj).__module__.startswith("gmlx.models.embedders")


def _embed_texts_gguf_encoder(model_obj, tokenizer, texts: list,
                              add_special_tokens: bool = True):
    """Embed ``texts`` with a GGUF encoder backend (EmbeddingGemma: a gemma3
    backbone run as a bidirectional sentence encoder, K-quant leaves loaded into
    the gmlx encoder) and return ``(matrix, n_tokens)``.

    The model mean-pools, applies its dense head and L2-norms internally, so one
    forward per text yields the final vector in ``out.text_embeds``, with no
    extra pooling here. batch=1 sidesteps padding / batch_encode_plus the synthesized
    tokenizer may not implement; the embed lock serializes the GPU work anyway.

    A text is cut to the model's context from the GGUF metadata (the
    synthesized tokenizer reports no limit), keeping its closing EOS."""
    import mlx.core as mx

    cap = getattr(tokenizer, "model_max_length", None)
    if not isinstance(cap, int) or cap <= 0 or cap > _ENCODER_MAX_TOKENS:
        cap = _ENCODER_MAX_TOKENS           # sentinel / unknown -> the ceiling
    ctx = getattr(getattr(model_obj, "args", None), "max_position_embeddings", None)
    if isinstance(ctx, int) and 0 < ctx < cap:
        cap = ctx
    eos = _eos_id(tokenizer)
    vecs: list = []
    n_tokens = 0
    for text in texts:
        ids = _encode_for_embed(tokenizer, text, add_special_tokens)
        if len(ids) > cap:
            keep_eos = eos is not None and ids[-1] == eos
            ids = ids[:cap - 1] + [eos] if keep_eos else ids[:cap]
        if not ids:
            raise EmbeddingsRequestError(400, "input encodes to zero tokens")
        n_tokens += len(ids)
        out = model_obj(mx.array([ids]))             # [1, T]: no padding
        vec = out.text_embeds[0].astype(mx.float32)  # model pools + dense + L2;
        mx.eval(vec)                                 # f32 cast: numpy can't buffer bf16
        vecs.append(np.asarray(vec))
    return np.stack(vecs, axis=0), n_tokens


# Families with prompt tables and Matryoshka widths

# The sentence-transformers prompt tables of the GGUF families, from each
# model's config_sentence_transformers.json: google/embeddinggemma-2 (main,
# 2026-10-28), google/embeddinggemma-300m (as mlx-community copies it) and
# Qwen/Qwen3-Embedding-0.6B (the 4B and 8B repos ship the same table). A
# safetensors checkpoint's own file takes their place.
_GEMMA2_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
}
_GEMMA_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "BitextMining": "task: search result | query: ",
    "Clustering": "task: clustering | query: ",
    "Classification": "task: classification | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: summarization | query: ",
}
_QWEN3_PROMPTS = {
    "query": ("Instruct: Given a web search query, retrieve relevant passages "
              "that answer the query\nQuery:"),
    "document": "",
}

# The most tokens of one EmbeddingGemma 2 input, media included.
_GEMMA2_MAX_TOKENS = 8192

# The widths EmbeddingGemma's Matryoshka training keeps.
_GEMMA_DIMS = (768, 512, 256, 128)
# The narrowest width Qwen3-Embedding takes.
_QWEN3_MIN_DIMS = 32


@dataclass(frozen=True)
class _Family:
    """What the configured model takes beyond plain text: ``key`` is
    ``gemma2``, ``gemma``, ``qwen3`` or empty for any other model."""

    label: str
    key: str = ""
    prompts: dict | None = None
    width: int | None = None


# Model path -> its family.
_FAMILIES: dict[str, _Family] = {}


def _gguf_family(path: str) -> _Family:
    from gmlx.load.gguf_meta import read_int

    kv = _gguf_header(path)
    arch = str(kv.get("general.architecture") or "")
    if arch == "gemma-embedding2":
        return _Family("EmbeddingGemma 2", "gemma2", _GEMMA2_PROMPTS, 768)
    if arch == "gemma-embedding":
        return _Family("EmbeddingGemma", "gemma", _GEMMA_PROMPTS, 768)
    if arch == "qwen3":
        return _Family("Qwen3-Embedding", "qwen3", _QWEN3_PROMPTS,
                       read_int(kv, "qwen3.embedding_length"))
    return _Family(service_display(path))


def _checkpoint_family(path: str, folder: str) -> _Family:
    from gmlx.models.embedders import checkpoint

    config = checkpoint._read_json(folder, "config.json")
    model_type = config.get("model_type") if isinstance(config, dict) else None
    own = checkpoint.prompts(folder)
    if model_type == "embedding_gemma2":
        return _Family("EmbeddingGemma 2", "gemma2", own or _GEMMA2_PROMPTS, 768)
    if model_type == "gemma3_text":
        return _Family("EmbeddingGemma", "gemma", own or _GEMMA_PROMPTS, 768)
    if model_type == "qwen3":
        width = config.get("hidden_size") if isinstance(config, dict) else None
        return _Family("Qwen3-Embedding", "qwen3", own or _QWEN3_PROMPTS,
                       width if isinstance(width, int) else None)
    return _Family(checkpoint.FAMILIES.get(str(model_type), service_display(path)),
                   prompts=own)


def _family(target: str) -> _Family:
    """The configured model's family, from its GGUF header or config.json.
    A Hub checkpoint not on disk yet loads first."""
    if target in _FAMILIES:
        return _FAMILIES[target]
    if _is_gguf_ref(target):
        try:
            family = _gguf_family(target)
        except OSError:
            # The load reports a missing or unreadable file.
            return _Family(service_display(target))
    else:
        folder = _local_checkpoint(target)
        if folder is None:
            subservice.run_on_worker(
                _EMBED_WORKER, lambda: _load_embeddings_model(target),
                error_cls=EmbeddingsRequestError, what="embedding model load")
            folder = _local_checkpoint(target)
        if folder is None:
            return _Family(service_display(target))
        from gmlx.models.embedders.checkpoint import UnsupportedEncoderError

        try:
            family = _checkpoint_family(target, folder)
        except UnsupportedEncoderError:
            # The load reports a config that cannot be read.
            return _Family(service_display(target))
    _FAMILIES[target] = family
    return family


def _prompt(family: _Family, name: str | None) -> str:
    if name is None:
        return ""
    if not family.prompts:
        raise EmbeddingsRequestError(
            400, f"prompt_name: {family.label} has no prompt table")
    if name not in family.prompts:
        raise EmbeddingsRequestError(
            400, f"prompt_name {name!r} is not one of "
                 f"{', '.join(sorted(family.prompts))}")
    return family.prompts[name]


def _check_dimensions(family: _Family, dims: int | None) -> int | None:
    """The width to cut each vector to, or None. Models without a
    Matryoshka head ignore ``dimensions``."""
    if dims is None:
        return None
    if family.key in ("gemma", "gemma2"):
        if dims not in _GEMMA_DIMS:
            raise EmbeddingsRequestError(
                400, f"dimensions must be one of "
                     f"{', '.join(map(str, _GEMMA_DIMS))} for {family.label}")
        return dims
    if family.key == "qwen3":
        top = family.width or dims
        if not _QWEN3_MIN_DIMS <= dims <= top:
            raise EmbeddingsRequestError(
                400, f"dimensions must be from {_QWEN3_MIN_DIMS} to {top} for "
                     f"{family.label}")
        return dims
    return None


def _cut(matrix: np.ndarray, dims: int | None) -> np.ndarray:
    """Each vector cut to ``dims`` and normalized again (Matryoshka)."""
    if dims is None or dims >= matrix.shape[1]:
        return matrix
    cut = matrix[:, :dims]
    return cut / np.maximum(np.linalg.norm(cut, axis=1, keepdims=True), 1e-12)


# EmbeddingGemma 2 media

@dataclass(frozen=True)
class MediaSetting:
    """Where the EmbeddingGemma 2 service finds its media towers: an mmproj
    ``path`` (GGUF), the checkpoint's own (``path`` None for a safetensors
    model), or ``reason`` it runs text-only."""

    path: str | None = None
    reason: str | None = None


def media_refusal(reason: str) -> str:
    return f"media is not supported: the embeddings model runs text-only ({reason})"


class ProjectorFileMissing(FileNotFoundError):
    """The embeddings model's projector file is gone from disk."""


class _TowersHolder:
    """The towers of the loaded EmbeddingGemma 2, loaded on its first media
    request. ``model_path`` keys them by model and projector; ``reason``
    says why they cannot load."""

    model = None
    model_path = None
    reason: str | None = None

    @classmethod
    def attach(cls, model_obj, target: str, setting: MediaSetting) -> None:
        key = f"{target}\0{setting.path}"
        if cls.model_path != key:
            # The key is kept once the towers load or the projector is
            # found not to fit. Any other failure loads again next time.
            cls.model, cls.reason, cls.model_path = None, None, None
            try:
                cls.model = _load_towers(model_obj, target, setting)
            except FileNotFoundError as exc:
                if setting.path is not None and not os.path.exists(setting.path):
                    raise ProjectorFileMissing(str(exc)) from exc
                raise
            except ValueError as exc:
                name = os.path.basename(setting.path or target)
                cls.reason = f"{name} is not a projector for this model: {exc}"
                print(f"[server] embeddings: {media_refusal(cls.reason)}",
                      file=sys.stderr)
            cls.model_path = key
        if cls.reason is not None:
            raise EmbeddingsRequestError(400, media_refusal(cls.reason))
        model_obj.towers = cls.model


def _load_towers(model_obj, target: str, setting: MediaSetting):
    if setting.path is not None:
        from gmlx.load.dtypes import activation_dtype
        from gmlx.load.embedder_mmproj import load_towers

        return load_towers(setting.path,
                           text_hidden=model_obj.args.text.hidden_size,
                           act_dtype=activation_dtype("gemma_embedding2"),
                           log=lambda *a: None)
    from gmlx.models.embedders import checkpoint

    folder = _local_checkpoint(target)
    assert folder is not None
    return checkpoint.load_towers(folder)


def resolve_media(model_path: str, value=None, model_dirs=()) -> MediaSetting:
    """The :class:`MediaSetting` of the configured embeddings model, from
    ``server.embeddings_mmproj`` (``value``): False turns media off, a
    path or ``hf:`` ref names a GGUF projector, and None takes the preset's
    projector for this Mac or one paired beside the model. Only
    EmbeddingGemma 2 takes media; any other model gets an empty setting.
    Raises :class:`gmlx.config.ConfigError` for a projector set on a model
    that takes none."""
    from gmlx.config import ConfigError

    if value is False:
        return MediaSetting(reason="server.embeddings_mmproj is false")
    if not _is_gguf_ref(model_path):
        if value is not None:
            raise ConfigError(
                "it names a GGUF projector, and the embeddings model is a "
                "safetensors checkpoint, which carries its own towers. Set it "
                "to false for text only, or remove it.")
        folder = _local_checkpoint(model_path)
        if folder is None:
            return MediaSetting()
        from gmlx.models.embedders import checkpoint

        try:
            config = checkpoint._read_json(folder, "config.json")
            if not (isinstance(config, dict)
                    and config.get("model_type") == "embedding_gemma2"):
                return MediaSetting()
            return MediaSetting(reason=checkpoint.media_mismatch(folder, config))
        except checkpoint.UnsupportedEncoderError as exc:
            return MediaSetting(reason=exc.reason)
    if not os.path.isfile(model_path):
        return MediaSetting()
    if _gguf_family(model_path).key != "gemma2":
        if value is not None:
            raise ConfigError(
                f"it is set, and the embeddings model "
                f"({service_display(model_path)}) takes text only")
        return MediaSetting()
    if value is not None:
        from gmlx.config import resolve_path

        return _checked_mmproj(resolve_path(str(value), list(model_dirs)),
                               model_path)
    found, hint = _paired_mmproj(model_path, model_dirs)
    if found is None:
        return MediaSetting(reason=f"no mmproj found; {hint}")
    return _checked_mmproj(found, model_path)


def _checked_mmproj(path: str, model_path: str) -> MediaSetting:
    from gmlx.load.embedder_mmproj import mmproj_mismatch

    name = os.path.basename(path)
    if not os.path.isfile(path):
        return MediaSetting(reason=f"{name} does not exist")
    try:
        kv = _gguf_header(path)
    except Exception as exc:              # noqa: BLE001 (any unreadable file)
        return MediaSetting(reason=f"{name} cannot be read ({type(exc).__name__})")
    from gmlx.load.gguf_meta import read_int

    width = read_int(_gguf_header(model_path), "gemma-embedding2.embedding_length")
    reason = mmproj_mismatch(kv, int(width or 0))
    if reason is not None:
        return MediaSetting(reason=f"{name} is not a projector for this model: "
                                   f"{reason}")
    return MediaSetting(path=path)


def _paired_mmproj(model_path: str, model_dirs=()) -> tuple[str | None, str]:
    """The projector for an EmbeddingGemma 2 GGUF without a configured one:
    the preset's projector for this Mac when the model is a preset's file,
    else one beside the model (gmlx.load.discovery). Gives the hint for a
    log line when there is none."""
    preset = _preset_of(model_path, model_dirs)
    if preset is not None:
        ref = preset_mmproj_ref(preset)
        assert ref is not None
        from gmlx.config import ConfigError, resolve_path

        try:
            return resolve_path(ref, list(model_dirs)), ""
        except ConfigError:
            return None, f"run gmlx pull {ref}"
    from gmlx.load.discovery import embedder_mmproj_beside

    found = embedder_mmproj_beside(model_path)
    if found is not None:
        return found, ""
    return None, "set server.embeddings_mmproj or --embeddings-mmproj"


def _preset_of(model_path: str, model_dirs=()) -> dict | None:
    """The preset whose GGUF ``model_path`` is: the local file of one of
    its ``quants`` refs."""
    from gmlx.config import ConfigError, resolve_path

    real = os.path.realpath(model_path)
    for p in EMBEDDING_PRESETS:
        if "mmproj" not in p:
            continue
        for ref in p["quants"].values():
            try:
                if os.path.realpath(resolve_path(ref, list(model_dirs))) == real:
                    return p
            except ConfigError:
                continue
    return None


# Requests

def _cut_text(ids: list, cap: int, protected: list) -> list:
    """``ids`` cut to ``cap`` tokens by dropping unprotected tokens from the
    end: text goes, BOS, EOS and media blocks stay."""
    over = len(ids) - cap
    keep = [True] * len(ids)
    for i in range(len(ids) - 1, -1, -1):
        if over <= 0:
            break
        if not protected[i]:
            keep[i] = False
            over -= 1
    return [t for t, k in zip(ids, keep) if k]


def _protected(ids: list, markers) -> list:
    """Which of an item's ids are BOS, EOS or inside a media block."""
    out, inside = [], False
    opens = {markers.boi, markers.boa}
    closes = {markers.eoi, markers.eoa}
    for i, t in enumerate(ids):
        if t in opens:
            inside = True
        out.append(inside or t in (markers.bos, markers.eos))
        if t in closes:
            inside = False
    return out


def _embed_gemma2(model_obj, tokenizer, items: list, *, prompt: str,
                  add_special_tokens: bool, options, attach):
    """Embed EmbeddingGemma 2 items (lists of text and decoded media) and
    return ``(matrix, n_tokens)``. ``attach`` loads the towers into the
    model on the first item with media."""
    import mlx.core as mx

    from gmlx.models.embedders import gemma_embedding2_processor as proc

    cap = model_obj.args.max_position_embeddings
    markers = proc.Markers.from_args(
        model_obj.args, int(tokenizer.bos_token_id), int(_eos_id(tokenizer)))
    vecs, n_tokens = [], 0
    for i, parts in enumerate(items):
        try:
            plan = proc.plan_item(tokenizer, parts, markers, options,
                                  prompt=prompt,
                                  add_special_tokens=add_special_tokens)
        except ValueError as exc:
            raise EmbeddingsRequestError(400, f"input {i}: {exc}") from None
        if len(plan.ids) > cap:
            protected = _protected(plan.ids, markers)
            if sum(protected) > cap:
                raise EmbeddingsRequestError(
                    400, f"input {i} has {sum(protected)} media tokens, over the "
                         f"{cap} tokens one input holds")
            plan.ids = _cut_text(plan.ids, cap, protected)
        if not plan.ids:
            raise EmbeddingsRequestError(400, "input encodes to zero tokens")
        n_tokens += len(plan.ids)
        if plan.media:
            attach()
            out = proc.encode_item(model_obj, plan)
        else:
            out = model_obj(mx.array([plan.ids]))
        vec = out.text_embeds[0].astype(mx.float32)
        mx.eval(vec)
        vecs.append(np.asarray(vec))
    return np.stack(vecs, axis=0), n_tokens


def _decode_media(request, family: _Family, setting: MediaSetting, options,
                  is_gguf: bool):
    """The request's items with their media decoded, on the request thread;
    each item's media tokens are checked against the context first."""
    from . import media_decode
    from .embeddings_media import decode, media_tokens

    if family.key != "gemma2":
        raise EmbeddingsRequestError(
            400, f"media is not supported: {family.label} takes text only")
    reason = setting.reason
    if reason is None and setting.path is None and is_gguf:
        reason = "no mmproj found; set server.embeddings_mmproj or --embeddings-mmproj"
    if reason is not None:
        raise EmbeddingsRequestError(400, media_refusal(reason))
    cap = _GEMMA2_MAX_TOKENS
    out = []
    with media_decode.decode_pass():
        for i, item in enumerate(request.items):
            parts = [p if isinstance(p, str) else decode(p, options, cap)
                     for p in item.parts]
            try:
                n = media_tokens(parts, options)
            except ValueError as exc:
                raise EmbeddingsRequestError(400, f"input {i}: {exc}") from None
            n += 2 if request.add_special_tokens else 0
            if n > cap:
                raise EmbeddingsRequestError(
                    400, f"input {i} has {n} media tokens, over the {cap} tokens "
                         "one input holds. Send fewer or shorter media, or a "
                         "lower max_soft_tokens in mm_processor_kwargs.")
            out.append(parts)
    return out


def encode_embedding(vec, encoding_format: str):
    """Encode one vector per OpenAI ``encoding_format``: a list of floats, or a
    base64 string of the little-endian float32 bytes."""
    if encoding_format == "base64":
        return base64.b64encode(
            np.asarray(vec, dtype="<f4").tobytes()).decode("ascii")
    return [float(x) for x in vec]


def run_embeddings(inputs=None, *, configured_model: str, model: str = "",
                   encoding_format: str = "", body: dict | None = None,
                   media: MediaSetting | None = None) -> dict:
    """Validate fields, embed the inputs, and return the OpenAI response
    dict (``object``/``data``/``model``/``usage``). Raises
    :class:`EmbeddingsRequestError` for 4xx problems.

    ``body`` is the request's JSON body (:mod:`.embeddings_request`); without
    it, ``inputs`` is the ``input`` value and ``encoding_format`` the format.
    ``media`` says where EmbeddingGemma 2 finds its towers
    (:func:`resolve_media`)."""
    from . import embeddings_request

    if body is None:
        if inputs is None:
            raise EmbeddingsRequestError(400, "field 'input' is required")
        body = {"input": inputs, "encoding_format": encoding_format or None}
    model = str(body.get("model") or model or "")
    request = embeddings_request.parse(body)
    target = effective_model(model, configured_model)
    family = _family(target)
    if request.chat and family.key != "gemma2":
        raise EmbeddingsRequestError(
            400, f"chat messages are not supported: {family.label} takes plain "
                 "inputs")
    prompt = _prompt(family, request.prompt_name)
    dims = _check_dimensions(family, request.dimensions)
    setting = media or MediaSetting()
    if family.key == "gemma2":
        from gmlx.models.embedders import gemma_embedding2_processor as proc

        try:
            options = proc.parse_options(request.mm_processor_kwargs)
        except ValueError as exc:
            raise EmbeddingsRequestError(400, str(exc)) from None
        items = (_decode_media(request, family, setting, options,
                               _is_gguf_ref(target))
                 if request.has_media else [item.parts for item in request.items])
    elif request.has_media:
        raise EmbeddingsRequestError(
            400, f"media is not supported: {family.label} takes text only")
    else:
        texts = [prompt + "".join(item.parts) for item in request.items]
    is_gguf = _is_gguf_ref(target)
    special = request.add_special_tokens

    def _job():
        # Load (if needed) and embed on the same dedicated thread (see
        # subservice.SingleWorker). The drivers return host (numpy) values,
        # so nothing GPU-bound crosses back to the caller.
        holder = _GGUFEmbeddingsHolder if is_gguf else _EmbeddingsModelHolder
        model_obj, tokenizer = holder.get(target)
        if family.key == "gemma2":
            def attach():
                _TowersHolder.attach(model_obj, target, setting)
            return _embed_gemma2(model_obj, tokenizer, items, prompt=prompt,
                                 add_special_tokens=special, options=options,
                                 attach=attach)
        if is_gguf:
            if _is_gguf_encoder(model_obj):
                return _embed_texts_gguf_encoder(model_obj, tokenizer, texts,
                                                 add_special_tokens=special)
            return _embed_texts_gguf(model_obj, tokenizer, texts,
                                     pooling=_GGUFEmbeddingsHolder.pooling,
                                     causal=_GGUFEmbeddingsHolder.causal,
                                     add_special_tokens=special)
        return _embed_texts(model_obj, tokenizer, texts,
                            add_special_tokens=special)

    matrix, n_tokens = subservice.run_on_worker(
        _EMBED_WORKER, _job, error_cls=EmbeddingsRequestError,
        what="embedding")
    matrix = _cut(matrix, dims)
    fmt = request.encoding_format
    data = [{"object": "embedding", "index": i,
             "embedding": encode_embedding(matrix[i], fmt)}
            for i in range(matrix.shape[0])]
    return {
        "object": "list",
        "data": data,
        # Echo the requested name, else the advertised id, and never the
        # resolved local path.
        "model": (model.strip() if model and model.strip()
                  else "text-embedding-3-small"),
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }
