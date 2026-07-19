"""Text embeddings for the server: the engine behind ``POST /v1/embeddings``.

Three backends, chosen by what ``embeddings:`` points at:

* **GGUF decoder-LM** (a ``*.gguf`` file or ``hf:<org>/<repo>/<file>.gguf`` ref) -
  e.g. ``Qwen3-Embedding-{0.6,4,8}B``. These *are* the Qwen3 dense decoder trunk
  plus last-token (EOS) pooling and an L2-norm, so the runtime's own
  :func:`gmlx.loader.load_model` loads them and one prefill forward over
  ``model.model`` (the trunk) gives the hidden states we pool - no separate
  download, no encoder framework. This path is **not** capped at the BERT window,
  so long documents (30k-token transcripts) embed without silent truncation.

* **GGUF encoder** (a ``*.gguf`` / ``hf:`` ref to an encoder arch), e.g.
  EmbeddingGemma. Same :func:`gmlx.loader.load_model`, but the GGUF builds an
  mlx-embeddings ``Model`` (a bidirectional backbone + mean-pool + dense head +
  L2-norm) that returns pooled ``text_embeds`` itself, so we just forward each
  text.

* **mlx-embeddings safetensors** (an alias / mlx-community repo id / local dir) -
  BERT / XLM-RoBERTa / ModernBERT encoders that are not mlx-lm decoder arches, via
  the optional `mlx-embeddings <https://pypi.org/project/mlx-embeddings/>`_ package
  (``pip install 'gmlx[embeddings]'``). It returns pooled, L2-normalized
  vectors in ``outputs.text_embeds`` (the pooling is the model's own - mean for
  BERT, last-token for Qwen3 - so that code is arch-agnostic).

Either way the loaded model is cached process-wide, so repeat requests don't
reload. This module is import-safe without mlx-embeddings installed: that import
is lazy and only the safetensors path touches it (the GGUF path never does).

The point of this endpoint is to give Open WebUI (and other OpenAI clients) a
local RAG embedder, so no sentence-transformers model is downloaded from
HuggingFace.
"""

from __future__ import annotations

import base64
import concurrent.futures
import os

import numpy as np

from . import subservice
from .hf_cache import offline_resolve
from .subservice import (
    GGUFModelHolder,
    SingleWorker,
    SubserviceRequestError,
    preset_for,
)

# Embedding model presets surfaced by the ``gmlx init`` wizard, in display order.
# Two tiers:
#
#  * tier ``gguf`` - Qwen3-Embedding decoder-LMs that run on this runtime's own
#    loader (last-token pooling + L2-norm), so they need no extra. Values are
#    ``hf:`` refs into the official Qwen GGUF repos; ``quants`` lists only the
#    rungs those repos actually ship (0.6B has just Q8_0 / f16).
#  * tier ``mlx`` - mlx-embeddings safetensors encoders (Gemma3 / XLM-RoBERTa /
#    ModernBERT) that need the optional ``[embeddings]`` extra. Values are
#    mlx-community repo ids; ``quants`` are the bit-width variants.
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


EMBEDDING_PRESETS = [
    {"alias": "qwen3-embed-0.6b", "label": "Qwen3-Embedding 0.6B",
     "tier": "gguf", "dim": 1024, "ctx": 32768,
     "blurb": "small, fast, multilingual - strong retrieval per byte; the default",
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

DEFAULT_EMBEDDINGS_ALIAS = "qwen3-embed-0.6b"

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

# Per-input token ceiling for the mlx-embeddings (encoder) backend. The old
# 512-ctx BERTs are gone; the remaining encoders carry 2k-8k windows, so cap by
# each model's own ``model_max_length`` (guarding the ~1e30 "no limit" sentinel)
# up to this ceiling - a flat 512 would silently truncate long documents on the
# long-context encoders (arctic / nomic / bge-m3) that are the reason to pick them.
_ENCODER_MAX_TOKENS = 8192

# The GGUF decoder-LM embedders (Qwen3-Embedding) carry an 8k-32k context, so the
# 512 BERT cap would silently throw away most of a long document. Cap generously
# instead - long transcripts are exactly why someone reaches for these.
_GGUF_MAX_TOKENS = 32768

# All model work (the load and every forward) routes through one persistent
# single-worker thread - see subservice.SingleWorker for why (per-thread MLX
# streams; an uncaught cross-thread abort otherwise).
_EMBED_WORKER = SingleWorker("embeddings-worker")


class EmbeddingsRequestError(SubserviceRequestError):
    """A client-side problem with an embeddings request (HTTP 4xx)."""


def import_mlx_embeddings():
    """Import mlx-embeddings (a core dependency), with guidance when missing."""
    try:
        import mlx_embeddings
    except ImportError as exc:
        raise ImportError(
            "text embeddings require mlx-embeddings, a core gmlx dependency - "
            "a missing one usually means a broken install; reinstall "
            "gmlx.") from exc
    return mlx_embeddings


class _EmbeddingsModelHolder:
    """Process-wide single-model cache (mlx-embeddings' ``load`` has none of its
    own). Mirrors mlx-whisper's ModelHolder: reload only on a path change."""

    model = None
    tokenizer = None
    model_path = None

    @classmethod
    def get(cls, model_path: str):
        if cls.model is None or model_path != cls.model_path:
            from mlx_embeddings.utils import load
            with offline_resolve(model_path):
                cls.model, cls.tokenizer = load(model_path)
            cls.model_path = model_path
        return cls.model, cls.tokenizer


class _GGUFEmbeddingsHolder(GGUFModelHolder):
    """The GGUF decoder-LM backend's model cache (subservice.GGUFModelHolder)."""


def _load_embeddings_model(model_path: str) -> None:
    """Populate the model cache for ``model_path``, dispatching to the GGUF or
    mlx-embeddings holder by ref shape. Runs on :data:`_EMBED_WORKER` (like every
    request), so a request that races a background warm waits on this load
    instead of kicking off a second one."""
    if _is_gguf_ref(model_path):
        _GGUFEmbeddingsHolder.get(model_path)
    else:
        _EmbeddingsModelHolder.get(model_path)


def prewarm(model_path: str) -> concurrent.futures.Future:
    """Background-load the configured embeddings model at server startup
    (best-effort; see :func:`subservice.prewarm`)."""
    def _load():
        if not _is_gguf_ref(model_path):
            import_mlx_embeddings()       # install guidance if the extra is gone
        _load_embeddings_model(model_path)

    return subservice.prewarm(
        _EMBED_WORKER, _load, "embeddings",
        missing_hint="/v1/embeddings returns 404 until the file is restored - "
                     "or update the config / run `gmlx sync-models`")


def _is_gguf_ref(value) -> bool:
    """True for an embeddings value that names a GGUF file - a local ``*.gguf``
    path or an ``hf:<org>/<repo>/<file>.gguf`` ref - i.e. the decoder-LM backend.
    Everything else (alias, mlx-community repo id, local dir) is the
    mlx-embeddings safetensors backend."""
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
    for the default alias. A GGUF ref - passed directly or reached via a
    gguf-tier alias - is resolved to a concrete **local** file (hf: refs resolve
    from the local HF cache, never the network - a miss raises a ``gmlx pull``
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
    # resolves to its local file; a repo id / dir is the mlx-embeddings backend.
    if _is_gguf_ref(v):
        from .config import resolve_path
        return resolve_path(v, list(model_dirs))  # hf: -> cache; path -> abs
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


def _normalize_input(value) -> list:
    """Coerce the request ``input`` field to a non-empty list of strings.

    OpenAI also allows arrays of token ids; we don't decode those (RAG clients,
    including Open WebUI, send text) and reject them with a clear message.
    """
    if value is None:
        raise EmbeddingsRequestError(400, "field 'input' is required")
    if isinstance(value, str):
        if not value.strip():
            raise EmbeddingsRequestError(400, "field 'input' must not be empty")
        return [value]
    if isinstance(value, list):
        if not value:
            raise EmbeddingsRequestError(
                400, "field 'input' must not be an empty list")
        if all(isinstance(x, str) for x in value):
            return value
        raise EmbeddingsRequestError(
            400, "field 'input' must be a string or a list of strings "
                 "(arrays of token ids are not supported)")
    raise EmbeddingsRequestError(
        400, "field 'input' must be a string or a list of strings")


def _embed_texts(model_obj, tokenizer, texts: list):
    """Run mlx-embeddings over ``texts`` and return ``(matrix, n_tokens)``: a
    float32 ``[len(texts), dim]`` numpy array of pooled, L2-normalized vectors
    (the model's own pooling), plus the total non-padding token count (for the
    usage field)."""
    import mlx.core as mx

    cap = getattr(tokenizer, "model_max_length", None)
    if not isinstance(cap, int) or cap <= 0 or cap > _ENCODER_MAX_TOKENS:
        cap = _ENCODER_MAX_TOKENS           # sentinel / unknown -> the ceiling
    inputs = tokenizer.batch_encode_plus(
        texts, return_tensors="mlx", padding=True, truncation=True,
        max_length=cap)
    outputs = model_obj(inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
    embeds = outputs.text_embeds
    mx.eval(embeds)
    matrix = np.asarray(embeds, dtype=np.float32)
    if matrix.ndim == 1:                         # a model that returns [dim]
        matrix = matrix[None, :]
    n_tokens = int(np.asarray(inputs["attention_mask"]).sum())
    return matrix, n_tokens


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


def _encode_for_embed(tokenizer, text: str) -> list:
    """Token ids for one input. Tolerates backends whose ``encode`` rejects the
    ``add_special_tokens`` kwarg (the GGUF-synthesized fast tokenizer accepts it)."""
    try:
        ids = tokenizer.encode(text)
    except TypeError:
        ids = tokenizer.encode(text, add_special_tokens=True)
    return list(ids)


def _embed_texts_gguf(model_obj, tokenizer, texts: list):
    """Embed ``texts`` with the GGUF decoder-LM backend and return
    ``(matrix, n_tokens)``: a float32 ``[len(texts), dim]`` of last-token-pooled,
    L2-normalized hidden states, plus the total token count.

    One prefill forward over the trunk per text (batch=1) - left/right padding +
    last-non-pad bookkeeping would be the only reason to batch, and the embed lock
    serializes the GPU work anyway; RAG indexing is throughput- not latency-bound.
    Batched (mask-aware) forwards are a later optimization."""
    import mlx.core as mx

    trunk = getattr(model_obj, "model", model_obj)   # mlx-lm Model.model = decoder
    eos = _eos_id(tokenizer)
    vecs: list = []
    n_tokens = 0
    for text in texts:
        ids = _encode_for_embed(tokenizer, text)
        if len(ids) > _GGUF_MAX_TOKENS:
            ids = ids[:_GGUF_MAX_TOKENS]
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)                          # Qwen3-Embedding pools EOS
        if not ids:
            raise EmbeddingsRequestError(400, "input encodes to zero tokens")
        n_tokens += len(ids)
        hidden = trunk(mx.array([ids]))              # [1, T, H]
        pooled = hidden[0, -1, :].astype(mx.float32)  # numpy can't buffer bf16
        norm = mx.sqrt(mx.sum(pooled * pooled)) + 1e-12
        pooled = pooled / norm
        mx.eval(pooled)
        vecs.append(np.asarray(pooled))
    return np.stack(vecs, axis=0), n_tokens


def _is_gguf_encoder(model_obj) -> bool:
    """True if a GGUF loaded into an mlx-embeddings encoder Model (EmbeddingGemma:
    pooled ``text_embeds`` interface), vs an mlx-lm decoder trunk we pool by hand."""
    return type(model_obj).__module__.startswith("mlx_embeddings")


def _embed_texts_gguf_encoder(model_obj, tokenizer, texts: list):
    """Embed ``texts`` with a GGUF encoder backend (EmbeddingGemma: a gemma3
    backbone run as a bidirectional sentence encoder, K-quant leaves loaded into
    the mlx-embeddings Model) and return ``(matrix, n_tokens)``.

    The model mean-pools + applies its dense head + L2-norms internally, so one
    forward per text yields the final vector in ``out.text_embeds`` -- no extra
    pooling here. batch=1 sidesteps padding / batch_encode_plus the synthesized
    tokenizer may not implement; the embed lock serializes the GPU work anyway."""
    import mlx.core as mx

    cap = getattr(tokenizer, "model_max_length", None)
    if not isinstance(cap, int) or cap <= 0 or cap > _ENCODER_MAX_TOKENS:
        cap = _ENCODER_MAX_TOKENS           # sentinel / unknown -> the ceiling
    vecs: list = []
    n_tokens = 0
    for text in texts:
        ids = _encode_for_embed(tokenizer, text)
        if len(ids) > cap:
            ids = ids[:cap]
        if not ids:
            raise EmbeddingsRequestError(400, "input encodes to zero tokens")
        n_tokens += len(ids)
        input_ids = mx.array([ids])                  # [1, T]
        attention_mask = mx.ones((1, len(ids)))      # all-ones: no padding
        out = model_obj(input_ids, attention_mask=attention_mask)
        vec = out.text_embeds[0].astype(mx.float32)  # model pools + dense + L2;
        mx.eval(vec)                                 # f32 cast: numpy can't buffer bf16
        vecs.append(np.asarray(vec))
    return np.stack(vecs, axis=0), n_tokens


def encode_embedding(vec, encoding_format: str):
    """Encode one vector per OpenAI ``encoding_format``: a list of floats, or a
    base64 string of the little-endian float32 bytes."""
    if encoding_format == "base64":
        return base64.b64encode(
            np.asarray(vec, dtype="<f4").tobytes()).decode("ascii")
    return [float(x) for x in vec]


def run_embeddings(inputs, *, configured_model: str, model: str = "",
                   encoding_format: str = "") -> dict:
    """Validate fields, embed ``inputs``, and return the OpenAI response dict
    (``object``/``data``/``model``/``usage``). Raises
    :class:`EmbeddingsRequestError` for 4xx problems.

    ``inputs`` is the raw request ``input`` value (string or list of strings).
    """
    texts = _normalize_input(inputs)
    target = effective_model(model, configured_model)
    fmt = (encoding_format or "float").strip().lower()
    if fmt not in ENCODING_FORMATS:
        raise EmbeddingsRequestError(
            400, f"unsupported encoding_format {fmt!r} "
                 f"(supported: {', '.join(ENCODING_FORMATS)})")

    is_gguf = _is_gguf_ref(target)
    if not is_gguf:
        import_mlx_embeddings()         # gate only the safetensors backend

    def _job():
        # Load (if needed) and embed on the same dedicated thread - see
        # subservice.SingleWorker. _embed_texts* return host (numpy) values,
        # so nothing GPU-bound crosses back to the caller.
        if is_gguf:
            model_obj, tokenizer = _GGUFEmbeddingsHolder.get(target)
            if _is_gguf_encoder(model_obj):
                return _embed_texts_gguf_encoder(model_obj, tokenizer, texts)
            return _embed_texts_gguf(model_obj, tokenizer, texts)
        model_obj, tokenizer = _EmbeddingsModelHolder.get(target)
        return _embed_texts(model_obj, tokenizer, texts)

    matrix, n_tokens = subservice.run_on_worker(
        _EMBED_WORKER, _job, error_cls=EmbeddingsRequestError,
        what="embedding")

    data = [{"object": "embedding", "index": i,
             "embedding": encode_embedding(matrix[i], fmt)}
            for i in range(matrix.shape[0])]
    return {
        "object": "list",
        "data": data,
        # Echo the requested name, else the advertised id - never the resolved
        # local path.
        "model": (model.strip() if model and model.strip()
                  else "text-embedding-3-small"),
        "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
    }
