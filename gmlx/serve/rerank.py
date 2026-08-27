"""Reranking for the server: the engine behind ``POST /v1/rerank``.

``/v1/rerank`` is the Cohere/Jina shape (not an OpenAI endpoint): given a
``query`` and a list of ``documents``, return each document's relevance score,
sorted best-first. It is the second RAG stage - a vector search returns a coarse
top-N, the reranker re-scores those jointly against the query, and the best few
go to the LLM. Open WebUI calls it as an external reranker
(``RAG_RERANKING_ENGINE=external`` / ``RAG_EXTERNAL_RERANKER_URL``).

The model is a **Qwen3-Reranker GGUF** - a Qwen3 causal LM fine-tuned to answer
"yes"/"no" to whether a document satisfies a query. The relevance score is the
probability it assigns "yes" over "no" at the final position (a softmax over the
two token logits, i.e. ``sigmoid(yes - no)``) - the model's native mechanism. So
the runtime loads it like any decoder GGUF (:func:`gmlx.loader.load_model`);
there is no classifier head and no extra package. (BGE/Jina BERT cross-encoders,
which llama.cpp reranks via a ``cls.output.weight`` head, are not mlx-lm arches
and are out of scope here.)

The loaded model is cached process-wide, kept separate from the chat residency
pool so a rerank burst and chat never evict each other.
"""

from __future__ import annotations

import concurrent.futures

from . import subservice
from .embeddings import _is_gguf_ref       # shared "is this a GGUF ref" predicate
from .subservice import (
    GGUFModelHolder,
    SingleWorker,
    SubserviceRequestError,
    preset_for,
)

# The Qwen3-Reranker prompt scaffold (verbatim from the model card). The yes/no
# instruction lives in the system turn; the (instruction, query, document) triple
# is the user turn; the assistant turn is primed so the next token is the verdict.
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    "on the Query and the Instruct provided. Note that the answer can only be "
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query")

# Cap the (instruction, query, document) middle. The reranker carries a long
# context, but a single RAG chunk is small; this bounds a pathological document.
_MAX_DOC_TOKENS = 8192

# All model work (the load and every forward) routes through one persistent
# single-worker thread - see subservice.SingleWorker for why (per-thread MLX
# streams; an uncaught cross-thread abort otherwise).
_RERANK_WORKER = SingleWorker("rerank-worker")


class RerankRequestError(SubserviceRequestError):
    """A client-side problem with a rerank request (HTTP 4xx)."""


class _GGUFRerankHolder(GGUFModelHolder):
    """The Qwen3-Reranker GGUF's model cache (subservice.GGUFModelHolder)."""


def _qwen_rerank_ref(size: str, quant: str) -> str:
    """An ``hf:`` ref into the mradermacher Qwen3-Reranker GGUF repo for a rung
    (note the ``.`` before the quant - the mradermacher filenames are
    ``Qwen3-Reranker-<size>.<QUANT>.gguf``, unlike the dash-joined embedder repos)."""
    return (f"hf:mradermacher/Qwen3-Reranker-{size}-GGUF/"
            f"Qwen3-Reranker-{size}.{quant}.gguf")


# Reranker presets surfaced by the ``gmlx init`` wizard. One tier: Qwen3-Reranker
# GGUFs scored on this runtime (sigmoid(yes - no) at the final token), so they
# need no extra. ``quants`` are the pure K-quant rungs the mradermacher repos
# ship; ``default_quant`` is what the bare alias resolves to and the wizard
# pre-selects, ``sizes`` the on-disk GB per rung (for the quant follow-up).
# RERANK_ALIASES is derived from these so the presets are the single source of
# truth. The reranker is independent of the embedder, but the wizard defaults its
# quant to the embedder's chosen rung.
_RERANK_RUNGS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

RERANK_PRESETS = [
    {"alias": "qwen3-rerank-0.6b", "label": "Qwen3-Reranker 0.6B", "params": "0.6B",
     "blurb": "small, fast reranker - the default; pairs with any embedder",
     "default_quant": "Q8_0",
     "quants": {q: _qwen_rerank_ref("0.6B", q) for q in _RERANK_RUNGS},
     "sizes": {"Q4_K_M": 0.40, "Q5_K_M": 0.44, "Q6_K": 0.49, "Q8_0": 0.64}},
    {"alias": "qwen3-rerank-4b", "label": "Qwen3-Reranker 4B", "params": "4B",
     "blurb": "higher reranking accuracy at more memory / latency",
     "default_quant": "Q8_0",
     "quants": {q: _qwen_rerank_ref("4B", q) for q in _RERANK_RUNGS},
     "sizes": {"Q4_K_M": 2.50, "Q5_K_M": 2.89, "Q6_K": 3.31, "Q8_0": 4.28}},
    {"alias": "qwen3-rerank-8b", "label": "Qwen3-Reranker 8B", "params": "8B",
     "blurb": "best reranking accuracy of the family; most memory",
     "default_quant": "Q8_0",
     "quants": {q: _qwen_rerank_ref("8B", q) for q in _RERANK_RUNGS},
     "sizes": {"Q4_K_M": 5.03, "Q5_K_M": 5.85, "Q6_K": 6.72, "Q8_0": 8.71}},
]

DEFAULT_RERANK_ALIAS = "qwen3-rerank-0.6b"

# alias -> default-rung concrete ``hf:`` ref (single source of truth: the presets).
RERANK_ALIASES = {
    p["alias"]: p["quants"][p["default_quant"]] for p in RERANK_PRESETS
}


def rerank_preset(alias):
    """The preset dict for a base alias (e.g. ``qwen3-rerank-4b``), or ``None``."""
    return preset_for(alias, RERANK_PRESETS)


def resolve_rerank_model(value, model_dirs=()) -> str:
    """Resolve a configured rerank value to a concrete **local** GGUF file.

    Accepts a friendly alias (``qwen3-rerank-0.6b``/``-4b``/``-8b``), a Qwen3-
    Reranker ``*.gguf`` path or ``hf:<org>/<repo>/<file>.gguf`` ref, or
    ``True``/``"default"`` for the default alias. A gguf ref - passed directly or
    reached via an alias - resolves from the local HF cache (never the network);
    a relative path is searched under ``model_dirs``, same as a ``models:``
    entry. Anything else raises a :class:`~gmlx.config.ConfigError`."""
    from .config import ConfigError, resolve_path
    if value is True:
        value = DEFAULT_RERANK_ALIAS
    v = str(value).strip()
    low = v.lower()
    if low in ("default", "true"):
        v = RERANK_ALIASES[DEFAULT_RERANK_ALIAS]
    elif low in RERANK_ALIASES:
        v = RERANK_ALIASES[low]             # alias -> default-rung concrete ref
    if v and _is_gguf_ref(v):
        return resolve_path(v, list(model_dirs))
    raise ConfigError(
        f"rerank model {value!r} must be a Qwen3-Reranker GGUF - an alias "
        f"(qwen3-rerank-0.6b/-4b/-8b), a *.gguf path, or "
        f"hf:<org>/<repo>/<file>.gguf")


def _load_rerank_model(model_path: str) -> None:
    """Populate the rerank model cache. Runs on :data:`_RERANK_WORKER` (like
    every request), so a request that races a background warm waits on this
    load instead of starting a second."""
    _GGUFRerankHolder.get(model_path)


def prewarm(model_path: str) -> concurrent.futures.Future:
    """Background-load the configured reranker at server startup
    (best-effort; see :func:`subservice.prewarm`)."""
    return subservice.prewarm(
        _RERANK_WORKER, lambda: _load_rerank_model(model_path), "rerank")


def _normalize_query(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerankRequestError(400, "field 'query' is required and must be a "
                                      "non-empty string")
    return value


def _normalize_documents(value) -> list:
    """Coerce the request ``documents`` field to a non-empty list of strings.
    Accepts plain strings or Cohere/Jina ``{"text": ...}`` objects."""
    if value is None:
        raise RerankRequestError(400, "field 'documents' is required")
    if not isinstance(value, list) or not value:
        raise RerankRequestError(400, "field 'documents' must be a non-empty list")
    out: list = []
    for d in value:
        if isinstance(d, str):
            out.append(d)
        elif isinstance(d, dict) and isinstance(d.get("text"), str):
            out.append(d["text"])
        else:
            raise RerankRequestError(
                400, "each document must be a string or an object with a "
                     "string 'text' field")
    return out


def _single_token_id(tokenizer, token_str: str):
    """The vocab id for ``token_str`` if it is exactly one token, else None. Tries
    the exact vocab entry first (what Qwen3-Reranker uses), then a 1-token encode."""
    unk = getattr(tokenizer, "unk_token_id", None)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    try:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(token_str)
    return int(ids[0]) if len(ids) == 1 else None


def _yes_no_ids(tokenizer):
    """The ('yes', 'no') token ids, or raise if this model can't be scored as a
    Qwen3-Reranker (no single yes/no token - likely the wrong model)."""
    yes_id = _single_token_id(tokenizer, "yes")
    no_id = _single_token_id(tokenizer, "no")
    if yes_id is None or no_id is None:
        raise RuntimeError(
            "this model does not expose single 'yes'/'no' tokens - it does not "
            "look like a Qwen3-Reranker; point `rerank:` at a Qwen3-Reranker GGUF")
    return yes_id, no_id


def _encode(tokenizer, text: str, *, add_special_tokens: bool) -> list:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def _score_documents(model, tokenizer, query: str, documents: list,
                     instruction: str):
    """Score each (query, document) pair as the model's P(yes) at the final
    position. Returns ``(scores, n_tokens)`` with ``scores`` aligned to
    ``documents``. One forward per document (the rerank lock serializes them)."""
    import mlx.core as mx

    yes_id, no_id = _yes_no_ids(tokenizer)
    prefix_ids = _encode(tokenizer, _PREFIX, add_special_tokens=False)
    suffix_ids = _encode(tokenizer, _SUFFIX, add_special_tokens=False)
    scores: list = []
    n_tokens = 0
    for doc in documents:
        middle = _encode(
            tokenizer,
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}",
            add_special_tokens=False)
        if len(middle) > _MAX_DOC_TOKENS:
            middle = middle[:_MAX_DOC_TOKENS]
        ids = prefix_ids + middle + suffix_ids
        n_tokens += len(ids)
        logits = model(mx.array([ids]))[0, -1, :]      # [vocab]
        score = mx.sigmoid(logits[yes_id] - logits[no_id])   # P(yes) over {no,yes}
        mx.eval(score)
        scores.append(float(score))
    return scores, n_tokens


def run_rerank(query, documents, *, configured_model: str, model: str = "",
               top_n=None, instruction=None, return_documents: bool = True) -> dict:
    """Validate fields, score ``documents`` against ``query``, and return the
    Cohere/Jina rerank response dict (``results`` sorted best-first, plus
    ``model`` and ``usage``). Raises :class:`RerankRequestError` for 4xx problems.

    The server serves one configured reranker; ``model`` is echoed back but the
    configured model is always used (a request can never load a different one)."""
    q = _normalize_query(query)
    docs = _normalize_documents(documents)
    instr = instruction if (isinstance(instruction, str) and instruction.strip()) \
        else _DEFAULT_INSTRUCTION
    if top_n is not None:
        # bool is an int subclass and float() truncates: `true` and `2.9` would
        # both coerce to a plausible-looking count instead of the documented 400.
        bad = isinstance(top_n, bool) or (
            isinstance(top_n, float) and not top_n.is_integer())
        try:
            if bad:
                raise ValueError
            top_n = int(top_n)
        except (TypeError, ValueError):
            raise RerankRequestError(400, f"top_n {top_n!r} is not an integer")
        if top_n <= 0:
            raise RerankRequestError(400, "top_n must be a positive integer")

    def _job():
        # Load (if needed) and score on the same dedicated thread - see
        # subservice.SingleWorker. _score_documents returns host floats, so
        # nothing GPU-bound crosses back to the caller.
        model_obj, tokenizer = _GGUFRerankHolder.get(configured_model)
        return _score_documents(model_obj, tokenizer, q, docs, instr)

    scores, n_tokens = subservice.run_on_worker(
        _RERANK_WORKER, _job, error_cls=RerankRequestError, what="reranking")

    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[:top_n]
    results = []
    for i in order:
        entry = {"index": i, "relevance_score": scores[i]}
        if return_documents:
            entry["document"] = {"text": docs[i]}
        results.append(entry)
    return {
        # Echo the requested name, else the advertised id - never the resolved
        # local path.
        "model": (model or "reranker"),
        "results": results,
        "usage": {"total_tokens": n_tokens},
    }
