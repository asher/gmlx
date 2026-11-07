import json
import weakref
from typing import Any

import mlx.core as mx
import numpy as np

_LLGUIDANCE_MASK_KERNEL = mx.fast.metal_kernel(
    name="mlx_vlm_llguidance_mask",
    input_names=["logits", "mask"],
    output_names=["out"],
    source="""
        uint batch = thread_position_in_grid.y;
        uint token = thread_position_in_grid.x;
        uint word = token >> 5;
        uint bit = token & 31;
        bool allowed = word < mask_shape[1] &&
            ((as_type<uint>(mask[batch * mask_shape[1] + word]) >> bit) & 1u);
        uint offset = batch * logits_shape[1] + token;
        out[offset] = allowed ? logits[offset] : -metal::numeric_limits<T>::infinity();
    """,
)


def _apply_llguidance_mask(logits: mx.array, mask: mx.array) -> mx.array:
    if logits.ndim == 1:
        logits = logits[None, :]
    if mask.ndim == 1:
        mask = mask[None, :]
    return _LLGUIDANCE_MASK_KERNEL(
        inputs=[logits, mask],
        template=[("T", logits.dtype)],
        grid=(logits.shape[1], logits.shape[0], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
    )[0]


def _allocate_shared_bitmask(batch_size: int, vocab_size: int):
    """Allocate a CPU-writable mask backed by MLX shared memory."""
    mask = mx.full(
        (batch_size, (vocab_size + 31) // 32),
        -1,
        dtype=mx.int32,
    )
    mx.eval(mask)
    view = np.array(mask, copy=False)
    if not view.flags["C_CONTIGUOUS"] or not view.flags["WRITEABLE"]:
        raise RuntimeError("MLX bitmask must expose writable contiguous memory")
    return mask, view


class LLGuidanceLogitsProcessor:
    """MLX logits processor backed by llguidance.

    Accepts a single sequence or a batch. Expected shapes are input_ids as
    (seq_len,) or (batch, seq_len), and logits as (vocab,), (1, vocab), or
    (batch, vocab).
    """

    # Do not hold a latency-sensitive constrained token behind unrelated
    # prompt-prefill work after it has been decoded.
    requires_immediate_decode_yield = True

    def __init__(self, grammar: str, llg_tokenizer) -> None:
        self.grammar = grammar
        self.llg_tokenizer = llg_tokenizer
        self.is_first_token = True
        self._mask_cursor = -1

    def clone(self) -> "LLGuidanceLogitsProcessor":
        return LLGuidanceLogitsProcessor(self.grammar, self.llg_tokenizer)

    def reset(self):
        self.is_first_token = True
        self.ll_matchers = None
        self.bitmask = None
        self.bitmask_mx = None
        self._mask_buffers = None
        self._mask_cursor = -1

    def _setup(self, batch_size: int) -> None:
        from llguidance import LLMatcher

        self.ll_matchers = [
            LLMatcher(self.llg_tokenizer, self.grammar) for _ in range(batch_size)
        ]
        self._mask_buffers = [
            _allocate_shared_bitmask(batch_size, self.llg_tokenizer.vocab_size)
            for _ in range(2)
        ]
        self._mask_cursor = -1

    def _consume_tokens(self, last_tokens: list[int]) -> None:
        for i, last_token in enumerate(last_tokens):
            self.ll_matchers[i].consume_token(last_token)
            error = self.ll_matchers[i].get_error()
            if error:
                raise ValueError(f"LLGuidance matcher error: {error}")

    def _fill_next_token_mask(self) -> mx.array:
        import llguidance.numpy

        self._mask_cursor = (self._mask_cursor + 1) % len(self._mask_buffers)
        self.bitmask_mx, self.bitmask = self._mask_buffers[self._mask_cursor]
        for i, matcher in enumerate(self.ll_matchers):
            llguidance.numpy.fill_next_token_bitmask(matcher, self.bitmask, i)
        return self.bitmask_mx

    def _apply_bitmask(self, logits: mx.array) -> mx.array:
        return _apply_llguidance_mask(logits, self._fill_next_token_mask())

    def prepare_next_token_mask(self, last_token: int) -> mx.array:
        """Advance one matcher and return its packed MLX token mask."""
        if self.is_first_token:
            self._setup(1)
            self.is_first_token = False
        else:
            self._consume_tokens([last_token])
        return self._fill_next_token_mask()

    def process_last_token(self, last_token: int, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            return self.process_last_token(last_token, mx.expand_dims(logits, 0))[0]

        if self.is_first_token:
            self._setup(logits.shape[0])
            self.is_first_token = False
        else:
            self._consume_tokens([last_token])

        return self._apply_bitmask(logits)

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if input_ids.ndim == 1 and logits.ndim == 2 and logits.shape[0] == 1:
            input_ids = mx.expand_dims(input_ids, 0)
        elif logits.ndim == 1:
            return self(mx.expand_dims(input_ids, 0), mx.expand_dims(logits, 0))[0]

        batch_size = input_ids.shape[0]
        if self.is_first_token:
            self._setup(batch_size)
            self.is_first_token = False
        else:
            self._consume_tokens(input_ids[:, -1].tolist())

        return self._apply_bitmask(logits)


class ThinkingAwareLogitsProcessor:
    """Delay a logits processor until model thinking has ended.

    Structured grammars constrain the final answer, not the private thinking
    tokens. This wrapper keeps the wrapped processor inactive until it observes
    the configured thinking end token, then lets the wrapped processor constrain
    the next sampled token as the first token of the final answer.

    gmlx: the wrapper waits for the whole end marker, and for a format whose
    answer has a header of its own (ATEM), for that header: the <|eom|> that
    ends reasoning also ends commentary and tool calls. ``prompt_ids`` gives
    the prompt's last tokens, so a header the prompt begins completes on the
    generated rest.

    gmlx: ``maybe_seq`` (the start marker's ids) is for a prompt that neither
    opens nor closes a block, so the reply may open one first. A grammar
    then masks the first step with its own mask plus the first start token.
    A sampled token the grammar accepts keeps the grammar on. The first
    start token, when the grammar refuses it, starts a block: the grammar
    resets, the next steps allow only the rest of the start marker, and the
    wrapper waits for the end marker. Another processor (XTC) skips the
    first step, then waits for the end marker after the first start token
    and runs from the next step after any other token.
    """

    def __init__(
        self,
        processor,
        tokenizer,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
        enable_thinking: bool = False,
        prompt_ids=None,
        maybe_seq=None,
    ) -> None:
        # gmlx: gate on the whole answer opener, see the class docstring.
        from gmlx.gen.thinking_budget import AnswerGate, answer_opener_seqs

        self.processor = processor
        self.tokenizer = tokenizer
        self.thinking_start_token = thinking_start_token
        self.thinking_end_token = thinking_end_token
        self.enable_thinking = enable_thinking
        self._active_context: list[int] = []

        self.thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        self.thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]
        seqs = answer_opener_seqs(tokenizer, thinking_start_token, thinking_end_token)
        self._prompt_ids = [int(t) for t in (prompt_ids or ())][-16:]
        self.answer_gate = AnswerGate(seqs, self._prompt_ids)
        self._n_seen = None
        self._maybe_seq = tuple(int(t) for t in (maybe_seq or ()))
        self._active = not enable_thinking or (
            bool(self._prompt_ids) and self.answer_gate.opened()
        )
        self._start_maybe()

    def _start_maybe(self) -> None:
        # "first" until the first step, then "pending" until the first
        # sampled token decides; None after that, or with no maybe_seq.
        self._maybe = "first" if self._maybe_seq and not self._active else None
        self._force: list[int] = []
        self._grammar_row = None

    def _is_grammar(self) -> bool:
        return isinstance(self.processor, LLGuidanceLogitsProcessor)

    def clone(self) -> "ThinkingAwareLogitsProcessor":
        processor = (
            self.processor.clone()
            if hasattr(self.processor, "clone")
            else self.processor
        )
        return ThinkingAwareLogitsProcessor(
            processor=processor,
            tokenizer=self.tokenizer,
            thinking_start_token=self.thinking_start_token,
            thinking_end_token=self.thinking_end_token,
            enable_thinking=self.enable_thinking,
            prompt_ids=self._prompt_ids,
            maybe_seq=self._maybe_seq,
        )

    def reset(self):
        self.answer_gate = self.answer_gate.restart(self._prompt_ids)
        self._n_seen = None
        self._active = not self.enable_thinking or (
            bool(self._prompt_ids) and self.answer_gate.opened()
        )
        self._start_maybe()
        self._active_context = []
        if hasattr(self.processor, "reset"):
            self.processor.reset()

    def _activate(self) -> None:
        self._active = True
        self._active_context = []

    def _delegate_process_last_token(
        self, last_token: int, logits: mx.array
    ) -> mx.array:
        if hasattr(self.processor, "process_last_token"):
            return self.processor.process_last_token(last_token, logits)

        self._active_context.append(last_token)
        return self.processor(mx.array(self._active_context), logits)

    def _first_step(self, logits: mx.array, masked) -> mx.array:
        """The maybe state's first step: the grammar's mask plus the first
        start token, or no processing for another processor."""
        self._maybe = "pending"
        if not self._is_grammar():
            return logits
        masked = masked()
        self._grammar_row = masked
        first = self._maybe_seq[0]
        keep = mx.arange(logits.shape[-1]) == first
        return mx.where(keep, logits.reshape(masked.shape), masked).reshape(
            logits.shape
        )

    def _decide(self, token: int) -> bool:
        """The first sampled token of a maybe state. True when it starts
        a block."""
        self._maybe = None
        row, self._grammar_row = self._grammar_row, None
        if row is not None:
            flat = row.reshape(-1, row.shape[-1])[0]
            if bool(mx.isfinite(flat[token]).item()):
                return False
        if token != self._maybe_seq[0]:
            return False
        if hasattr(self.processor, "reset"):
            self.processor.reset()
        if self._is_grammar():
            self._force = list(self._maybe_seq[1:])
        return True

    def _forced(self, logits: mx.array) -> mx.array:
        keep = mx.arange(logits.shape[-1]) == self._force[0]
        return mx.where(keep, logits, mx.array(-float("inf"), logits.dtype))

    def _after_forced(self, token: int) -> None:
        if self._force and token == self._force[0]:
            self._force.pop(0)
        else:
            self._force = []

    def process_last_token(self, last_token: int, logits: mx.array) -> mx.array:
        if self._maybe == "first":
            # Called with a token before the first step: a seed, not a
            # sample.
            self.answer_gate.feed(last_token)
            return self._first_step(
                logits,
                lambda: self._delegate_process_last_token(last_token, logits),
            )
        if self._maybe == "pending":
            self.answer_gate.feed(last_token)
            if not self._decide(last_token):
                self._activate()
                return self._delegate_process_last_token(last_token, logits)
            return self._forced(logits) if self._force else logits
        if not self._active:
            if self._force:
                self._after_forced(last_token)
                self.answer_gate.feed(last_token)
                return self._forced(logits) if self._force else logits
            if not self.answer_gate.feed(last_token):
                return logits
            self._activate()
        return self._delegate_process_last_token(last_token, logits)

    def _observe_context(self, input_ids: mx.array) -> bool:
        """Feed the tokens ``input_ids`` adds since the last call. The decode
        loops pass the growing context: on the first call the prompt (or its
        last token), then one more token each step."""
        gate = self.answer_gate
        if input_ids.ndim == 0:
            return gate.feed(int(input_ids.item()))
        if input_ids.size == 0:
            return False
        row = input_ids.reshape(-1, input_ids.shape[-1])[0]
        n = row.shape[0]
        if self._n_seen is None:
            # Prompt tokens. The seed already holds a shorter tail.
            self._n_seen = n
            if n >= gate.keep or not self._prompt_ids:
                gate.tail = [int(t) for t in row[-gate.keep :].tolist()]
            return gate.opened()
        new = row[max(self._n_seen, n - gate.keep) :].tolist()
        self._n_seen = n
        return gate.advance(new) is not None

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if self._active:
            return self.processor(input_ids, logits)

        if self._maybe == "first":
            self._observe_context(input_ids)
            return self._first_step(
                logits, lambda: self.processor(input_ids, logits)
            )
        if self._maybe == "pending":
            self._observe_context(input_ids)
            token = int(input_ids.reshape(-1)[-1].item())
            if not self._decide(token):
                self._active = True
                return self.processor(input_ids, logits)
            return self._forced(logits) if self._force else logits
        if self._force:
            self._observe_context(input_ids)
            self._after_forced(int(input_ids.reshape(-1)[-1].item()))
            return self._forced(logits) if self._force else logits

        if self._observe_context(input_ids):
            self._activate()
            return self.processor(input_ids, logits)

        return logits


def _serialize_schema(schema: str | dict[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema)


# Building an llguidance tokenizer walks the entire vocab (~1.5s for a 150k
# token model), so we keep the result around.
# gmlx: an entry lives as long as its tokenizer. A bare id(tokenizer) key gave
# a new tokenizer at a freed id the old vocab.
_llg_tokenizer_cache = {}


def _cached_llg_tokenizer(tokenizer):
    key = id(tokenizer)
    entry = _llg_tokenizer_cache.get(key)
    if entry is not None and entry[0]() is tokenizer:
        return entry[1]
    import llguidance.hf

    llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)

    def _drop(ref, key=key):
        if _llg_tokenizer_cache.get(key, (None,))[0] is ref:
            del _llg_tokenizer_cache[key]

    try:
        ref = weakref.ref(tokenizer, _drop)
    except TypeError:
        return llg_tokenizer
    _llg_tokenizer_cache[key] = (ref, llg_tokenizer)
    return llg_tokenizer


def build_json_schema_logits_processor(tokenizer, schema: str | dict[str, Any]):
    try:
        import llguidance as llg
        import llguidance.hf
    except ImportError as exc:
        raise ImportError(
            "llguidance is required for structured response_format generation. "
            "Install mlx-vlm with llguidance available."
        ) from exc

    llg_tokenizer = _cached_llg_tokenizer(tokenizer)  # gmlx: weak entry

    schema_text = _serialize_schema(schema)
    if hasattr(llg, "JsonCompiler"):
        grammar = llg.JsonCompiler(
            separators=(", ", ": "),
            whitespace_pattern="",
        ).compile(schema_text)
    else:
        grammar = llg.grammar_from("json_schema", schema_text)
    return LLGuidanceLogitsProcessor(grammar, llg_tokenizer)
