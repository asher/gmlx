"""uvicorn log config and the per-request timing log."""

from __future__ import annotations

import importlib
import logging
import time


# Per-request timing log
#
# mlx-vlm already builds a full timing envelope per request (ttft, prefill/decode
# tok/s, elapsed) and funnels it through ServerMetricsStore.record_success - but only
# into the /v1/metrics aggregate, never the log. We emit one line per request from
# that funnel, and suppress uvicorn's own access line for the same endpoints so a
# request is one log event, not two. The poll endpoints (/health, /v1/metrics) are
# dropped entirely - they are the menu-bar's once-a-second noise.
_REQUEST_LOG_FLAG = "_kq_gguf_request_log"

# Endpoints we log ourselves (so their uvicorn access line is suppressed - one event
# per request). The generation paths emit via the ServerMetricsStore funnel above; the
# audio + embeddings paths bypass that funnel, so they emit via _log_aux_request at the
# route handler instead. Anything listed here must emit its own line or it goes dark.
_TIMED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses",
                "/v1/messages", "/v1/audio/transcriptions", "/v1/audio/speech",
                "/v1/embeddings", "/v1/images/generations", "/v1/images/edits",
                "/v1/systemone")

# Pure-poll endpoints with no useful access signal. The models listing is
# polled by the menu bar (4s) and by web clients (Open WebUI), so it floods
# `logs -f` the same way health/metrics do.
_SILENT_PATHS = ("/health", "/v1/metrics", "/metrics", "/v1/models", "/models")


class _AccessNoiseFilter(logging.Filter):
    """Drop uvicorn access lines for the poll endpoints and for the generation
    endpoints we log ourselves - so the log carries one event per request and no
    once-a-second health/metrics flood. uvicorn's access record carries the request
    as ``args = (client_addr, method, full_path, http_version, status_code)``."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        path = str(args[2]).split("?", 1)[0]
        drop = path in _SILENT_PATHS or path in _TIMED_PATHS
        return not drop


def uvicorn_log_config(level: str | None = None) -> dict:
    """A uvicorn ``log_config`` that timestamps every access/default line and attaches
    :class:`_AccessNoiseFilter`. The serve path hands this to ``uvicorn.run`` so the
    standard access lines gain a timestamp and the poll/generation noise is filtered;
    the generation lines come from :func:`install_request_timing_log` instead.

    ``level`` (``--log-level``) sets the application loggers (``gmlx`` and
    ``mlx_vlm``) too - uvicorn's own ``log_level`` kwarg only re-levels the
    ``uvicorn.*`` loggers after this config is applied. ``trace`` maps to
    ``DEBUG`` here (stdlib logging has no TRACE tier)."""
    import copy

    import uvicorn

    cfg = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    for name in ("default", "access"):
        f = cfg["formatters"].get(name)
        if f and "%(asctime)s" not in f.get("fmt", ""):
            f["fmt"] = "%(asctime)s " + f["fmt"]
            f.setdefault("datefmt", "%Y-%m-%d %H:%M:%S")
    cfg.setdefault("filters", {})["kq_access_noise"] = {
        "()": "gmlx.serve.patches.observability._AccessNoiseFilter"}
    acc = cfg["loggers"].setdefault("uvicorn.access", {})
    acc["filters"] = [*acc.get("filters", []), "kq_access_noise"]
    # Route the package's own loggers (residency, speculative, spec_engine, ...)
    # through uvicorn's timestamped stderr handler; without an entry here their
    # INFO records die in logging's WARNING-level lastResort handler.
    app_level = "INFO"
    if level:
        app_level = "DEBUG" if level.lower() == "trace" else level.upper()
    for name in ("gmlx", "mlx_vlm"):
        cfg["loggers"][name] = {
            "handlers": ["default"], "level": app_level, "propagate": False}
    return cfg


def _format_timing_line(envelope: dict) -> str:
    """One human-readable line from a metrics envelope: timestamp, endpoint, model,
    token counts, and the timing fields (missing values render as ``-``)."""
    e = envelope
    ts = time.strftime("%Y-%m-%d %H:%M:%S",
                       time.localtime(e.get("timestamp_unix") or time.time()))

    def n(value, fmt: str) -> str:
        return fmt.format(value) if isinstance(value, (int, float)) else "-"

    parts = [
        ts, str(e.get("endpoint") or "?"), str(e.get("model") or "?"),
        f"prompt={int(e.get('prompt_tokens') or 0)}",
        f"gen={int(e.get('generated_tokens') or 0)}",
        f"ttft={n(e.get('ttft_s'), '{:.2f}')}s",
        f"prefill={n(e.get('prefill_tok_s'), '{:.0f}')}t/s",
        f"decode={n(e.get('decode_tok_s'), '{:.1f}')}t/s",
        f"total={n(e.get('request_elapsed_s'), '{:.2f}')}s",
    ]
    reason = e.get("finish_reason")
    if reason and reason != "stop":
        parts.append(f"finish={reason}")
    # Allocator state at completion = what the next request inherits. active is
    # MLX-tracked live bytes, cache the freed-buffer reuse pool (both wired).
    try:
        import mlx.core as mx
        parts.append(f"active={mx.get_active_memory() / 1e9:.1f}G")
        parts.append(f"cache={mx.get_cache_memory() / 1e9:.1f}G")
    except Exception:  # noqa: S110 - memory readout is cosmetic
        pass
    return "[req] " + " ".join(parts)


def install_request_timing_log() -> None:
    """Emit one timestamped timing line per completed request by wrapping the metrics
    store's ``record_success`` / ``record_failure`` - the single funnel every chat /
    completions / responses / messages / transcriptions path calls once it finishes.
    Idempotent and best-effort: a logging hiccup never disturbs the response, and an
    upstream rename degrades to no timing line rather than a crash."""
    try:
        gen = importlib.import_module("mlx_vlm.server.generation")
        store = gen.ServerMetricsStore
        if getattr(store.record_success, _REQUEST_LOG_FLAG, False):
            return
        orig_success = store.record_success
        orig_failure = store.record_failure
    except Exception:
        return

    def record_success(self, envelope):
        orig_success(self, envelope)
        try:
            print(_format_timing_line(envelope), flush=True)
        except Exception:  # noqa: S110 - a logging hiccup must never disturb the metrics store
            pass

    def record_failure(self, *, endpoint, model, stream, error):
        orig_failure(self, endpoint=endpoint, model=model, stream=stream, error=error)
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[req] {ts} {endpoint} {model} failed {error}", flush=True)
        except Exception:  # noqa: S110 - a logging hiccup must never disturb the metrics store
            pass

    record_success.__dict__[_REQUEST_LOG_FLAG] = True
    store.record_success = record_success
    store.record_failure = record_failure


# Decode progress rate
#
# The stock progress line divides the newest chunk's tokens by the gap since
# the chunk before it. Speculative rounds deliver several tokens microseconds
# apart and then wait a whole verify, so that figure swings between thousands
# of tok/s and a fraction of the real rate. The owned copy below reports the
# rate since the previous progress line instead.
_PROGRESS_RATE_FLAG = "_kq_gguf_progress_rate"


def _log_decode_progress(uid, info: dict, *, token: int, text: str,
                         finish_reason, token_count: int = 1) -> float:
    gen = importlib.import_module("mlx_vlm.server.generation")
    log = gen.logger
    now = time.perf_counter()
    previous_tokens = int(info.get("generated_tokens", 0) or 0)
    emitted_tokens = max(0, int(token_count or 0))
    generated_tokens = previous_tokens + emitted_tokens
    info["generated_tokens"] = generated_tokens
    request_id = info.get("request_id", uid)
    if emitted_tokens > 0:
        info["last_token_at"] = now

    decode_started_at = info.get("decode_started_at")
    if decode_started_at is None:
        decode_started_at = now
        info["decode_started_at"] = now
        queued_at = float(info.get("queued_at", now) or now)
        log.info(
            "Decode started: request=%s time_to_first_token=%.3fs",
            request_id,
            max(0.0, now - queued_at),
        )

    elapsed = max(0.0, now - decode_started_at)
    first_chunk_tokens = int(info.get("decode_first_chunk_tokens", 0) or 0)
    if first_chunk_tokens == 0:
        first_chunk_tokens = emitted_tokens
        info["decode_first_chunk_tokens"] = emitted_tokens
    measured_tokens = max(0, generated_tokens - first_chunk_tokens)
    rate = measured_tokens / elapsed if elapsed > 0 else 0.0

    mark_tokens, mark_at = info.get("progress_mark") or (
        first_chunk_tokens, decode_started_at)
    window = now - mark_at
    window_tokens = generated_tokens - mark_tokens
    progress_rate_text = (f"{window_tokens / window:.1f} tok/s"
                          if window > 0 and window_tokens > 0 else "n/a")

    interval = gen.get_log_progress_interval()
    crossed_interval = interval > 0 and (
        generated_tokens // interval > previous_tokens // interval)
    if crossed_interval:
        info["progress_mark"] = (generated_tokens, now)
    debug_enabled = log.isEnabledFor(logging.DEBUG)
    if debug_enabled:
        log.debug(
            "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
            "rate=%s token_number=%d token_id=%s text=%r",
            request_id, generated_tokens, elapsed, progress_rate_text,
            generated_tokens, token, text,
        )

    if finish_reason is not None:
        log.info(
            "Decode completed: request=%s generated_tokens=%d elapsed=%.3fs "
            "rate=%.1f tok/s finish_reason=%s",
            request_id, generated_tokens, elapsed, rate, finish_reason,
        )
    elif crossed_interval and not debug_enabled:
        log.info(
            "Decode progress: request=%s generated_tokens=%d elapsed=%.3fs "
            "rate=%s",
            request_id, generated_tokens, elapsed, progress_rate_text,
        )
    return now


def install_decode_progress_rate() -> None:
    """Replace ``ResponseGenerator._log_decode_progress`` with the owned
    copy above. Idempotent. Must run before the live-requests install,
    which wraps the method it finds."""
    gen = importlib.import_module("mlx_vlm.server.generation")
    cls = gen.ResponseGenerator
    if cls.__dict__.get(_PROGRESS_RATE_FLAG, False):
        return
    cls._log_decode_progress = staticmethod(_log_decode_progress)
    setattr(cls, _PROGRESS_RATE_FLAG, True)
