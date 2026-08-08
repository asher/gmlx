"""Faithful chat-history rendering for the OpenAI server routes.

mlx-vlm's ``prompt_utils.apply_chat_template`` rebuilds every non-tool
message for model types in its ``MODEL_CONFIG`` table from ``content`` and
``role`` alone, discarding all other keys the client sent -- most notably
``reasoning_content``, which silently defeats ``preserve_thinking`` on
every non-tool turn (the chat template never sees the reasoning it is
asked to preserve). Tool-metadata messages take a passthrough branch and
are unaffected, which is why tool chains replay verbatim while plain
turns lose their reasoning. Model types outside the table already take a
key-preserving path, so which behavior a text request gets today depends
on the model_type string alone.

The wrapper keeps the upstream rebuild (role mapping, media token
placement, multimodal text extraction) and restores only the keys the
rebuild did not produce: it calls the stock function with
``return_messages=True``, merges each input message's unconsumed keys
back into its rebuilt counterpart (rebuilt keys win), and renders the
merged list through the same tail the stock function uses. The chat
template stays the second gate -- with ``preserve_thinking`` unset it
drops prior-turn reasoning exactly as before, so this restores the
documented option rather than changing default output.
Kill: ``GMLX_FAITHFUL_HISTORY=0``.
"""

from __future__ import annotations

import importlib
import os

# Stock tail: these model types return only the last message when a
# rendered prompt is requested.
_LAST_MESSAGE_ONLY = ("paligemma", "molmo", "florence2", "falcon_ocr")


def install_faithful_history() -> None:
    """Wrap every captured ``apply_chat_template`` binding with the key
    merge: the openai and anthropic protocol modules plus
    ``app._protocol_deps`` each hold their own reference.

    Install before ``install_retire_render_capture`` so the retirement
    memo captures this wrapper as its render callable -- next-turn
    predictions must see the same faithful render the server produces.
    """
    if os.environ.get("GMLX_FAITHFUL_HISTORY") == "0":
        return
    from ._common import _render_target_modules
    prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")

    def _wrap(orig):
        def apply_chat_template(processor, config, prompt,
                                add_generation_prompt=True,
                                return_messages=False,
                                num_images=0, num_audios=0, **kwargs):
            msgs = orig(processor, config, prompt,
                        add_generation_prompt=add_generation_prompt,
                        return_messages=True, num_images=num_images,
                        num_audios=num_audios, **kwargs)
            if not isinstance(msgs, list):
                # A wrapped render that ignores return_messages (test
                # fakes, exotic overrides) already produced final output.
                return msgs
            items = prompt if isinstance(prompt, list) else [prompt]
            # The stock rebuild appends one message per input item; items
            # it cannot read (no role/content) are skipped, breaking the
            # 1:1 mapping -- fall back to the unmerged list in that case.
            if len(msgs) == len(items):
                for i, (built, sent) in enumerate(zip(msgs, items)):
                    if isinstance(built, dict) and isinstance(sent, dict):
                        extra = {k: v for k, v in sent.items()
                                 if k not in built}
                        if extra:
                            msgs[i] = {**extra, **built}
            cfg = config if isinstance(config, dict) else config.__dict__
            if cfg.get("model_type") == "gpt_oss":
                # The harmony template reads prior-turn reasoning from a
                # ``thinking`` key; alias the OpenAI-style spelling so
                # preserve_thinking works for standard clients.
                for i, m in enumerate(msgs):
                    if (isinstance(m, dict) and "thinking" not in m
                            and m.get("reasoning_content")):
                        msgs[i] = {**m, "thinking": m["reasoning_content"]}
            if return_messages:
                return msgs
            if cfg.get("model_type") in _LAST_MESSAGE_ONLY:
                return msgs[-1]
            return prompt_utils.get_chat_template(
                processor, msgs, add_generation_prompt, **kwargs)

        apply_chat_template._kq_faithful_history = True
        return apply_chat_template

    for target in _render_target_modules():
        fn = getattr(target, "apply_chat_template", None)
        if fn is not None and not getattr(fn, "_kq_faithful_history", False):
            target.apply_chat_template = _wrap(fn)
