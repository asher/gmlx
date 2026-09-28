#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
"""Write the mlx-vlm 0.6.15 qwen3.5 text modules into gmlx/models/qwen35.

SRC is the ``mlx_vlm`` package directory of mlx-vlm 0.6.15, for example
from ``pip download mlx-vlm==0.6.15 --no-deps`` and an unzip of the wheel.
Each edit is an exact, counted string replacement, so a source of another
shape fails here instead of producing a different copy. ``--check``
compares against the files in the tree and exits 1 on a difference.
"""

import argparse
import sys
from pathlib import Path

HEADER = """# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Asher Feldman
# Portions copyright (c) 2025 Prince Canuma (mlx-vlm, MIT)
# Vendored from mlx-vlm 0.6.15, mlx_vlm/models/{src}.
"""

# Import rewrites as single-line literals, so no line of this file starts
# with an upstream import (tests/test_pyright_scope.py scans for those).
LANG_IMPORTS = [
    ("from ..activations ", "from mlx_vlm.models.activations "),
    ("from ..base import (", "from mlx_vlm.models.base import ("),
    ("from ..cache ", "from mlx_vlm.models.cache "),
    ("from ..rope_utils import MRoPERotaryEmbedding\n",
     "from mlx_vlm.models.qwen3_5.config import ModelConfig\n"
     "from mlx_vlm.models.rope_utils import MRoPERotaryEmbedding\n"),
    ("from ..rope_utils import apply_", "from mlx_vlm.models.rope_utils import apply_"),
    ("from .config import ModelConfig, TextConfig\n",
     "\nfrom gmlx.eval_guard import guard\n\n"
     "from .cache import Qwen35ArraysCache\nfrom .config import TextConfig\n"),
]
# Evals under an except route through gmlx.eval_guard (tests/test_eval_guard.py
# bans the bare form in gmlx), and comments are made ASCII
# (tests/test_ascii_hygiene.py).
LANG_EDITS = [
    ("            mx.eval(outputs)\n",
     '            guard.eval(outputs, site="qwen3_5_try_launch", owner="scratch")\n'),
    ("(N, n, H, D) \u2014 one kernel", "(N, n, H, D), one kernel"),
]
GD_EDITS = [
    ('name=f"qwen3_5_gated_delta_with_states{suffix}"',
     'name=f"gmlx_qwen3_5_gated_delta_with_states{suffix}"'),
    ("import mlx.nn as nn\n", "import mlx.nn as nn\n\nfrom gmlx.eval_guard import guard\n"),
    ("        mx.async_eval(Y, S)\n",
     '        guard.async_eval(Y, S, site="gated_delta_chunked", owner="scratch")\n'),
    ("O(1) entries \u2014 exactly", "O(1) entries, exactly"),
    ("captured CUDA graph \u2014 that capture", "captured CUDA graph, since that capture"),
]
MOE_IMPORTS = [
    ("from ..qwen3_5.language import LanguageModel ",
     "from mlx_lm.models.switch_layers import SwitchGLU\n"
     "from mlx_vlm.models.qwen3_5_moe.config import ModelConfig\n\n"
     "from .language import LanguageModel "),
    ("from ..qwen3_5.language import Qwen3_5Attention ",
     "from .language import Qwen3_5Attention "),
    ("from ..qwen3_5.language import Qwen3_5GatedDeltaNet ",
     "from .language import Qwen3_5GatedDeltaNet "),
    ("from ..qwen3_5.language import Qwen3_5MLP ",
     "from .language import Qwen3_5MLP "),
    ("from ..qwen3_5.language import Qwen3_5Model,",
     "from .language import Qwen3_5Model,"),
    ("from ..switch_layers import SwitchGLU\n", ""),
    ("from .config import ModelConfig, TextConfig\n",
     "from .moe_config import TextConfig\n"),
]
CONFIG_IMPORTS = (
    "from dataclasses import dataclass, field\n"
    "from typing import Dict, List, Optional, Union\n\n"
    "from mlx_vlm.models.base import BaseModelConfig\n\n\n"
)


def sub(text, old, new):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"expected 1 x {old!r}, found {n}")
    return text.replace(old, new)


def sub_all(text, pairs):
    for old, new in pairs:
        text = sub(text, old, new)
    return text


def text_config(src):
    start = src.index("@dataclass\nclass TextConfig(BaseModelConfig):")
    end = src.index("@dataclass\nclass ModelConfig(BaseModelConfig):")
    return CONFIG_IMPORTS + src[start:end].rstrip() + "\n"


def build(models):
    out = {}

    lang = (models / "qwen3_5/language.py").read_text()
    lang = sub_all(lang, LANG_IMPORTS)
    lang = sub_all(lang, LANG_EDITS)
    lang = sub(lang, """        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]""",
               """        return [
            Qwen35ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers
        ]""")
    out["language.py"] = ("qwen3_5/language.py", """mlx-vlm 0.6.15 qwen3.5 text model, the base the owned classes subclass.

Edits to the upstream file: imports are absolute, ``TextConfig`` comes from
the vendored ``config``, ``make_cache`` builds ``Qwen35ArraysCache`` for the
linear-attention layers, the launch probe in ``_qwen3_5_try_launch`` evals
through ``gmlx.eval_guard``, and comments are ASCII.
""", lang)

    gd = (models / "qwen3_5/gated_delta.py").read_text()
    gd = sub_all(gd, GD_EDITS)
    out["gated_delta.py"] = ("qwen3_5/gated_delta.py", """mlx-vlm 0.6.15 gated-delta ops and kernels for the vendored qwen3.5 model.

Includes the state-capture, state-replay and accept-states kernels the
vendored rollback runs. Edits to the upstream file: the state-capture
kernel is named ``gmlx_qwen3_5_gated_delta_with_states``, because mlx-vlm
0.7.3 compiles a kernel under the upstream name with the same template list
and a different state stride, and MLX keys compiled kernels by name and
template values; the ``async_eval`` in ``gated_delta_chunked`` goes through
``gmlx.eval_guard`` (the graph-transformation refusal it catches schedules
nothing, so the returned arrays are not async survivors); comments are ASCII.
""", gd)

    out["config.py"] = ("qwen3_5/config.py", """mlx-vlm 0.6.15 qwen3.5 dense ``TextConfig``, the text half only.

The multimodal ``ModelConfig`` and ``VisionConfig`` stay mlx-vlm's.
""", text_config((models / "qwen3_5/config.py").read_text()))

    out["moe_config.py"] = ("qwen3_5_moe/config.py", """mlx-vlm 0.6.15 qwen3.5 MoE ``TextConfig``, the text half only.

The multimodal ``ModelConfig`` and ``VisionConfig`` stay mlx-vlm's.
""", text_config((models / "qwen3_5_moe/config.py").read_text()))

    ml = (models / "qwen3_5_moe/language.py").read_text()
    ml = sub_all(ml, MOE_IMPORTS)
    out["moe_language.py"] = ("qwen3_5_moe/language.py", """mlx-vlm 0.6.15 qwen3.5 MoE text model over the vendored dense layers.

Edits to the upstream file: imports are absolute or point at the vendored
dense modules, and ``SwitchGLU`` is mlx-lm's, whose body matches the
mlx-vlm 0.6.15 copy (mlx-vlm 0.7.3's copy adds a decode-block route).
""", ml)

    files = {name: HEADER.format(src=rel) + f'"""{doc}"""\n\n' + body
             for name, (rel, doc, body) in out.items()}
    for name, text in files.items():
        if not text.isascii():
            raise SystemExit(f"{name}: non-ASCII left after the edits")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path,
                    help="mlx_vlm package directory of mlx-vlm 0.6.15")
    ap.add_argument("--dst", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "gmlx/models/qwen35",
                    help="output directory (default: gmlx/models/qwen35)")
    ap.add_argument("--check", action="store_true",
                    help="compare with the files in --dst instead of writing")
    args = ap.parse_args()
    files = build(args.src / "models")
    stale = []
    for name, text in files.items():
        path = args.dst / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if stale:
        print("differs from the generated copy: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
