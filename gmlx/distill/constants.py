"""Numeric constants and default knobs shared by the distill modules: the
format version, the decimal GB, the log-domain floors and the loss defaults.
Import from here rather than redefining a value."""
from __future__ import annotations

import math
import sys
from typing import Any

GB = 1e9
FORMAT_VERSION = 1
TABLES_VERSION = 3
LOG_FLOOR = math.log(2.0 ** -126)   # the f32 min-normal underflow floor
NEG_INF = float("-inf")

# Loss defaults. The CLI flags override them per run.
DEFAULT_KNOBS: dict[str, Any] = dict(
    lambda_dk=1.0, lambda_alm=1.0, lambda_ce=0.0, w_mid=0.5, T_dk=1.0,
    tau_alm=1.0, gamma=1e-3, max_chunk_len=8, redirect_cut=0.5,
    loss_mode="bucketed",
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

FRAME_KINDS = ("continue", "continue-closed", "chat", "reply", "reply-think")
