"""One BOS on prompts that mlx-vlm tokenizes.

mlx-vlm tokenizes a rendered prompt in ``mlx_vlm.utils.prepare_inputs``. Its
callers pass ``add_special_tokens=True`` for every model type except a few
gemma families, so a chat template that already opens with BOS gets a second
one from the tokenizer. ``gmlx run --mmproj`` and the server both reach it.

The wrapper applies the rule that ``gmlx.gen.generation.encode_prompt`` uses
on the text paths: a prompt that opens with the tokenizer's BOS string is
tokenized without special tokens. It only turns ``add_special_tokens`` off,
never on, so a prompt without a leading BOS keeps the tokenizer's own.

mlx-vlm modules bind ``prepare_inputs`` by name at import. ``install`` patches
``mlx_vlm.utils`` and rebinds every imported ``mlx_vlm`` module that still
holds the stock function. Modules imported later bind the wrapper. Idempotent.
"""

from __future__ import annotations

import functools
import importlib
import sys

_FLAG = "_gmlx_single_bos"


def _bos(processor) -> str | None:
    tokenizer = getattr(processor, "tokenizer", processor)
    return (getattr(tokenizer, "bos_token", None)
            or getattr(processor, "bos_token", None))


def opens_with_bos(processor, prompts) -> bool:
    """True when every prompt is a string that starts with the BOS string."""
    bos = _bos(processor)
    if not bos or prompts is None:
        return False
    rows = [prompts] if isinstance(prompts, str) else list(prompts)
    return bool(rows) and all(
        isinstance(p, str) and p.startswith(bos) for p in rows)


def install() -> bool:
    """Wrap ``prepare_inputs`` for a single BOS. Returns True once installed."""
    utils = importlib.import_module("mlx_vlm.utils")
    current = utils.prepare_inputs
    if getattr(current, _FLAG, False):
        return True
    stock = current

    @functools.wraps(stock)
    def prepare_inputs(processor, *args, **kwargs):
        if (kwargs.get("add_special_tokens")
                and opens_with_bos(processor, kwargs.get("prompts"))):
            kwargs["add_special_tokens"] = False
        return stock(processor, *args, **kwargs)

    setattr(prepare_inputs, _FLAG, True)
    # functools.wraps copied mlx_vlm's module; the seam check reads this
    # attribute to tell a gmlx replacement from the stock function.
    prepare_inputs.__module__ = __name__
    utils.prepare_inputs = prepare_inputs
    for name, module in list(sys.modules.items()):
        if (name.startswith("mlx_vlm") and module is not None
                and getattr(module, "prepare_inputs", None) is stock):
            module.prepare_inputs = prepare_inputs
    return True
