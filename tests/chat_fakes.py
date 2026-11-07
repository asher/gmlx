"""Stub tokenizers and processors that render a chat template as the served
ones do. Import after ``pytest.importorskip("gmlx._vendor.mlx_vlm")``."""

from __future__ import annotations

from transformers.utils.chat_template_utils import render_jinja_template

from gmlx._vendor.mlx_vlm.models.gemma4.processing_gemma4 import Gemma4Processor


class Tokenizer:
    """Renders the template as an HF tokenizer does, with its special
    tokens in the template context."""

    def __init__(self, template, bos=None):
        self.chat_template = template
        self._special = {"bos_token": bos} if bos else {}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        out = render_jinja_template(
            [messages], chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
            **{**self._special, **kw})
        rendered = out[0] if isinstance(out, tuple) else out
        return rendered[0] if isinstance(rendered, list) else rendered


class Processor:
    """A processor that forwards to its tokenizer, as the GGUF text
    processor does."""

    def __init__(self, template, bos=None):
        self._inner = Tokenizer(template, bos)

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class Gemma4VisionProcessor:
    """Renders as mlx-vlm's Gemma4Processor does: the stock rebuild once
    more, then the tokenizer."""

    apply_chat_template = Gemma4Processor.apply_chat_template
    _extract_media_from_messages = Gemma4Processor._extract_media_from_messages
    _as_list = staticmethod(Gemma4Processor._as_list)

    def __init__(self, template, model_type="gemma4_unified", bos="<bos>"):
        self.model_type = model_type
        self.tokenizer = Tokenizer(template, bos)
        self.chat_template = template
