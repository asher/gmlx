"""Structured decisions on DiffusionGemma: the Jev decision API.

A decision seeds the denoise canvas with an answer template, leaves one
label position per question as noise, runs one denoise step and reads the
label log-probabilities at each position. ``decide`` holds the decision
logic ported from vLLM, ``extensions`` adds ``think: "auto"`` and the
server's request defaults, and ``engine.StructuredReader`` runs the reads.
This package module imports no MLX, so the decision logic loads without
it."""

from .contract import (
    jev_answer,
    jev_answers,
    jev_response,
    jev_state,
    log_labels,
    usage,
)
from .decide import ReadEngine, slot_distribution
from .extensions import (
    GMLX_EXTENSIONS,
    decide,
    ignored_fields,
    request_schema,
)
from .reads import (
    Cancelled,
    ReadRequest,
    ReadResult,
    SampleRead,
    Slot,
    SlotRead,
    build_canvas,
    label_id_union,
)
from .schema import (
    JEV_EXTENSIONS,
    Limits,
    SchemaError,
    jev_schema,
    parse_seed,
)
from .template import TemplateResolver, system_text

__all__ = [
    "Cancelled",
    "GMLX_EXTENSIONS",
    "JEV_EXTENSIONS",
    "Limits",
    "ReadEngine",
    "ReadRequest",
    "ReadResult",
    "SampleRead",
    "SchemaError",
    "Slot",
    "SlotRead",
    "TemplateResolver",
    "build_canvas",
    "decide",
    "ignored_fields",
    "jev_answer",
    "jev_answers",
    "jev_response",
    "jev_schema",
    "jev_state",
    "label_id_union",
    "log_labels",
    "parse_seed",
    "request_schema",
    "slot_distribution",
    "system_text",
    "usage",
]
